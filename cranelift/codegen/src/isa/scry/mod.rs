//! Scry Instruction Set Architecture.

use crate::CodegenError;
use crate::dominator_tree::DominatorTree;
use crate::ir::pcc;
use crate::ir::{Function, Type};
use crate::isa::scry::inst::{EmitInfo, MInst};
use crate::isa::scry::settings as scry_settings;
use crate::isa::unwind::systemv;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, TargetIsa,
};
use crate::machinst::isle::{Writable, WritableReg};
use crate::machinst::{
    BlockLoweringOrder, Callee, CompiledCode, CompiledCodeStencil, MachInst,
    MachTextSectionBuilder, Reg, SigSet, TextSectionBuilder, VCode, VCodeBuildDirection,
    VCodeBuilder, VRegAllocator,
};
use crate::result::CodegenResult;
use crate::settings::{self as shared_settings, Flags};
use crate::timing;
use crate::trace;
use crate::{VCodeConstants, ir};
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use cranelift_control::ControlPlane;
use graphene::algo::Bfs;
use graphene::core::Graph;
use graphene::core::GraphMut;
use graphene::core::property::Rooted;
use regalloc2::Function as RegFunc;
use std::collections::{HashMap, HashSet};
use target_lexicon::{Architecture, Triple};

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;
mod vcode_cfg;
use vcode_cfg::*;

/// A Scry backend.
pub struct ScryBackend {
    triple: Triple,
    flags: shared_settings::Flags,
    isa_flags: scry_settings::Flags,
}

impl ScryBackend {
    /// Create a new scry backend with the given (shared) flags.
    pub fn new_with_flags(
        triple: Triple,
        flags: shared_settings::Flags,
        isa_flags: scry_settings::Flags,
    ) -> ScryBackend {
        ScryBackend {
            triple,
            flags,
            isa_flags,
        }
    }

    /// This performs lowering to VCode, register-allocates the code, computes block layout and
    /// finalizes branches. The result is ready for binary emission.
    fn compile_vcode(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<(VCode<inst::MInst>, VRegAllocator<MInst>)> {
        let emit_info = EmitInfo::new(self.flags.clone(), self.isa_flags.clone());
        let sigs = SigSet::new::<abi::ScryMachineDeps>(func, &self.flags)?;
        let abi = Callee::<abi::ScryMachineDeps>::new(func, self, &self.isa_flags, &sigs)?;

        // ------ The below code is copied from cranelift/codegen/src/machinst/compile.rs ------
        // Compute lowered block order.
        let block_order = BlockLoweringOrder::new(func, domtree, ctrl_plane);

        // Build the lowering context.
        let lower = crate::machinst::Lower::new(
            func,
            abi,
            emit_info,
            block_order,
            sigs,
            self.flags().clone(),
        )?;

        // Lower the IR.
        let mut vcode = {
            log::debug!(
                "Number of CLIF instructions to lower: {}",
                func.dfg.num_insts()
            );
            log::debug!("Number of CLIF blocks to lower: {}", func.dfg.num_blocks());

            let _tt = timing::vcode_lower();
            lower.lower(self, ctrl_plane)?
        };

        log::debug!(
            "Number of lowered vcode instructions: {}",
            vcode.num_insts()
        );
        log::debug!("Number of lowered vcode blocks: {}", vcode.num_blocks());
        trace!("vcode from lowering: \n{:?}", vcode);

        // Perform validation of proof-carrying-code facts, if requested.
        if self.flags().enable_pcc() {
            pcc::check_vcode_facts(func, &mut vcode, self).map_err(CodegenError::Pcc)?;
        }
        // ------ The above code is copied from cranelift/codegen/src/machinst/compile.rs ------

        // To be able to create more virtual registers, we crate a new vreg allocator
        // that will generate new vregs that don't class with the existing ones in the vcode.
        let mut new_vregs = VRegAllocator::<MInst>::with_capacity(vcode.num_vregs());
        let mut existing_vregs = HashSet::new();

        let mut worklist = vec![vcode.entry_block()];
        let mut done = HashSet::new();

        // Walk the CFG, looking for vregs
        while let Some(b) = worklist.pop() {
            done.insert(b);
            for inst in vcode.block_insns(b).iter() {
                for op in vcode.inst_operands(inst).iter() {
                    existing_vregs.insert(op.vreg());
                }
            }
            for succ in vcode.block_succs(b) {
                if !done.contains(&succ) && !worklist.contains(&succ) {
                    worklist.push(*succ)
                }
            }
        }

        // Find the maximum index between the used vregs
        let max_vreg_idx = existing_vregs
            .iter()
            .fold(0, |acc, r| std::cmp::max(r.vreg(), acc));

        // Keeps making vregs in new allocator, until it exceeds the existing vreg indices
        while max_vreg_idx
            > new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
                .to_virtual_reg()
                .unwrap()
                .index()
        {}

        Ok((vcode, new_vregs))
    }
}

fn prepare_block_params(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    // Handle entry block first, moving MInst:Args to the params
    let entry_bb = cfg.0.root_weight_mut();

    match entry_bb.inst.remove(0) {
        MInst::Args { args } => {
            entry_bb.params = args.into_iter().map(|r| r.vreg.to_reg()).collect();
        }
        inst => unreachable!("Entry did not include MInst::Args: {:?}", inst),
    }

    let mut bfs = Bfs::new(&cfg.0);
    while bfs.next().is_some() {}
    let pred = bfs
        .predecessor_tree()
        .all_edges()
        .map(|(so, si, _)| (so, si))
        .collect::<HashMap<_, _>>();

    // Insert params for non-entry blocks
    for (v, p) in pred.iter() {
        let bb = cfg.0.vertex_weight(v).unwrap();
        let pred_bb = cfg.0.vertex_weight(p).unwrap();

        if bb.params.is_empty() && !pred_bb.params.is_empty() {
            cfg.0.vertex_weight_mut(v).unwrap().params = pred_bb.params.clone();
        }
    }

    for (_, bb) in cfg.0.all_vertices_weighted_mut() {
        // Insert echos for handling params
        if bb.params.len() >= 1 {
            bb.inst.insert(
                0,
                MInst::Echo {
                    rss: vec![],
                    rds: bb
                        .params
                        .iter()
                        .map(|r| WritableReg::from_reg(*r))
                        .collect(),
                    outs: bb.params.iter().map(|_| 0).collect(),
                },
            )
        }

        // Convert ImmJump to issue/trigger combo
        let mut i = 0;
        while i < bb.inst.len() {
            let inst = &mut bb.inst[i];
            if let MInst::ImmJump { dst } = inst {
                let link = new_vreg();
                *inst = MInst::JumpIssue {
                    link: Writable::from_reg(link),
                    dst: *dst,
                };
                bb.inst.insert(
                    i + 1,
                    MInst::JumpTrigger {
                        link,
                        args: bb.branch_params.clone(),
                    },
                );
            }
            i += 1;
        }
    }

    log::trace!("VCodeCFG: {:?}", cfg);
}

/// Replaces the first use of the `find` register with the `replace` register
fn replace_first_use(bb: &mut VCodeBB<MInst>, find: Reg, replace: Reg) -> bool {
    for inst in bb.inst.iter_mut() {
        if let Some(r) = inst.get_uses_mut().find(|r| **r == find) {
            *r = replace;
            return true;
        }
    }
    false
}

/// Looks for registers that are used multiple times and inserts `dup` instructions at their definition
/// and makes the uses use the resulting, unique registers. I.e. eliminates multiple uses of the same register.
fn insert_duplicates(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("insert_duplicates");
    let mut find_and_replace = |bb: &mut VCodeBB<MInst>, reg, dup_idx| {
        let rd1 = new_vreg();
        let rd2 = new_vreg();

        // Replace uses with uses of new vregs
        assert!(replace_first_use(bb, reg, rd1));
        assert!(replace_first_use(bb, reg, rd2));

        // Insert duplication of original register to new vregs
        bb.inst.insert(
            dup_idx,
            MInst::Duplicate {
                rd1: Writable::from_reg(rd1),
                rd2: Writable::from_reg(rd2),
                rs: reg,
                out1: 0,
                out2: 0,
            },
        );
    };

    'a: loop {
        let entry = cfg.0.root();
        for (v, bb) in cfg.0.all_vertices_weighted_mut() {
            // Check each param for multiple uses
            for p in &bb.params {
                if bb.reg_uses(*p).count() > 1 {
                    let insert_idx = if v == entry {
                        1 // After the Minst::Args in the entry
                    } else {
                        0
                    };
                    find_and_replace(bb, *p, insert_idx);
                    continue 'a;
                }
            }

            // Check each def for multiple uses
            for (inst_idx, reg_def) in bb.inst_defs().collect::<Vec<_>>() {
                if bb.reg_uses(reg_def).count() > 1 {
                    find_and_replace(bb, reg_def, inst_idx + 1);
                    continue 'a;
                }
            }
        }
        // No changes were made, finish
        break;
    }
    log::trace!("VCodeCFG: {:?}", cfg);
}

fn insert_ref_distances(cfg: &mut VCodeCFG<MInst>) {
    log::debug!("insert_ref_distances");
    for (_, bb) in cfg.0.all_vertices_weighted_mut() {
        log::trace!("BB: {:?}", bb);

        let mut use_pos = HashMap::<Reg, Vec<usize>>::new();
        for (inst_idx, inst) in bb.inst.iter_mut().rev().enumerate() {
            log::trace!("inst: {:?}", inst);
            // Record all uses, we can do this on-the-fly as we assume no use comes before its def
            inst.get_uses().for_each(|r| {
                use_pos.entry(*r).or_insert_with(Vec::new).push(inst_idx);
            });

            if inst.get_defs().count() >= 1 {
                let ref_dists = inst
                    .get_defs()
                    .enumerate()
                    .map(|(i, def)| {
                        let use_idx = use_pos[&def][0];
                        (i, (inst_idx - use_idx - 1) as u16)
                    })
                    .collect::<HashMap<_, _>>();

                match inst {
                    MInst::Add { out, .. } | MInst::Load { out, .. } => {
                        *out = ref_dists[&0];
                    }
                    MInst::Echo { outs, .. } => outs
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, out)| *out = ref_dists[&i]),
                    MInst::Duplicate { out1, out2, .. } => {
                        *out1 = ref_dists[&0];
                        *out2 = ref_dists[&1];
                    }
                    MInst::CallArgs { rets, .. } if ref_dists[&0] > 0 => {
                        *inst = MInst::Echo {
                            rss: vec![],
                            rds: rets.iter().map(|g| g.vreg).collect(),
                            outs: rets
                                .iter()
                                .enumerate()
                                .map(|(i, _)| ref_dists[&i])
                                .collect(),
                        }
                    }
                    _ => (),
                };
            }
        }

        log::trace!("BB: {:?}", bb);
    }
    log::trace!("VCodeCFG: {:?}", cfg);
}

fn fix_orderings(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("fix_orderings");
    for (_, bb) in cfg.0.all_vertices_weighted_mut() {
        'a: loop {
            let mut def_pos = HashMap::<Reg, usize>::new();
            for (inst_idx, inst) in bb.inst.iter_mut().enumerate() {
                // Record def positions
                for def in inst.get_defs() {
                    def_pos.insert(def, inst_idx);
                }

                match inst {
                    MInst::Store { rd, rs } => {
                        // Reorder if the address precedes the value
                        if def_pos[rd] < def_pos[rs] {
                            // Create new vregs for the reorder
                            let rd1 = new_vreg();
                            let rd2 = new_vreg();

                            let rd_old = *rd;
                            let rs_old = *rs;

                            // assign reordered vregs to store
                            *rd = rd1;
                            *rs = rd2;

                            // Insert reorder instruction before store
                            bb.inst.insert(
                                inst_idx,
                                MInst::Reorder {
                                    rd1: Writable::from_reg(rd1),
                                    rd2: Writable::from_reg(rd2),
                                    rs1: rd_old,
                                    rs2: rs_old,
                                    out: 0,
                                },
                            );

                            // Start over
                            continue 'a;
                        }
                    }
                    _ => (),
                }
            }
            break;
        }
    }
    log::trace!("VCodeCFG: {:?}", cfg);
}

impl TargetIsa for ScryBackend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        log::debug!("Beginning Scry compile");
        log::trace!("func: {:?}", func);
        let (vcode, mut new_vregs) = self.compile_vcode(func, domtree, ctrl_plane)?;
        let mut new_vreg = || {
            new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
        };

        let mut cfg = VCodeCFG::from_vcode(&vcode);

        log::trace!("VCodeCFG: {:?}", cfg);

        prepare_block_params(&mut cfg, &mut new_vreg);

        // Insert `ret` instruction as movable trigger
        cfg.0
            .all_vertices_weighted_mut()
            .find(|(_, bb)| {
                bb.inst
                    .iter()
                    .find(|i| {
                        if let MInst::Rets { .. } = i {
                            true
                        } else {
                            false
                        }
                    })
                    .is_some()
            })
            .map(|(_, bb)| {
                let inst_count = bb.inst.len();
                bb.inst.insert(inst_count - 1, MInst::Ret { trig: 0 })
            });

        log::trace!("VCodeCFG: {:?}", cfg);

        insert_duplicates(&mut cfg, &mut new_vreg);
        insert_ref_distances(&mut cfg);
        fix_orderings(&mut cfg, &mut new_vreg);

        let sigs = SigSet::new::<abi::ScryMachineDeps>(func, &self.flags)?;
        let abi = Callee::<abi::ScryMachineDeps>::new(func, self, &self.isa_flags, &sigs)?;
        let mut builder = VCodeBuilder::<inst::MInst>::new(
            sigs,
            abi,
            EmitInfo::new(self.flags.clone(), self.isa_flags.clone()),
            BlockLoweringOrder::new(func, domtree, ctrl_plane),
            VCodeConstants::with_capacity(vcode.constants.len()),
            VCodeBuildDirection::Backward,
            2,
        );

        cfg.build_vcode(&mut builder);

        let vreg_alloc = VRegAllocator::with_capacity(vcode.num_vregs());
        let vcode2 = builder.build(vreg_alloc);

        log::trace!("VCode2: {:?}", vcode2);

        let want_disasm = want_disasm || log::log_enabled!(log::Level::Debug);
        let emit_result = vcode2.emit(
            &regalloc2::Output::default(),
            want_disasm,
            &self.flags,
            ctrl_plane,
        );
        let value_labels_ranges = emit_result.value_labels_ranges;
        let buffer = emit_result.buffer;

        if let Some(disasm) = emit_result.disasm.as_ref() {
            log::debug!("disassembly:\n{disasm}");
        }

        Ok(CompiledCodeStencil {
            buffer,
            vcode: emit_result.disasm,
            value_labels_ranges,
            bb_starts: emit_result.bb_offsets,
            bb_edges: emit_result.bb_edges,
        })
    }

    fn name(&self) -> &'static str {
        "scry"
    }
    fn dynamic_vector_bytes(&self, _dynamic_ty: ir::Type) -> u32 {
        unimplemented!()
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    fn flags(&self) -> &shared_settings::Flags {
        &self.flags
    }

    fn isa_flags(&self) -> Vec<shared_settings::Value> {
        self.isa_flags.iter().collect()
    }

    fn isa_flags_hash_key(&self) -> IsaFlagsHashKey<'_> {
        IsaFlagsHashKey(self.isa_flags.hash_key())
    }

    #[cfg(feature = "unwind")]
    fn emit_unwind_info(
        &self,
        _result: &CompiledCode,
        _kind: crate::isa::unwind::UnwindInfoKind,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        Ok(None)
    }

    #[cfg(feature = "unwind")]
    fn create_systemv_cie(&self) -> Option<gimli::write::CommonInformationEntry> {
        None
    }

    fn text_section_builder(&self, num_funcs: usize) -> Box<dyn TextSectionBuilder> {
        Box::new(MachTextSectionBuilder::<inst::MInst>::new(num_funcs))
    }

    #[cfg(feature = "unwind")]
    fn map_regalloc_reg_to_dwarf(&self, _reg: Reg) -> Result<u16, systemv::RegisterMappingError> {
        unimplemented!()
    }

    fn function_alignment(&self) -> FunctionAlignment {
        inst::MInst::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        debug_assert_eq!(1 << 12, 0x1000);
        12
    }

    fn pretty_print_reg(&self, _reg: Reg, _size: u8) -> String {
        unimplemented!()
    }

    fn has_native_fma(&self) -> bool {
        false
    }

    fn has_round(&self) -> bool {
        false
    }

    fn has_blendv_lowering(&self, _: Type) -> bool {
        false
    }

    fn has_x86_pshufb_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmulhrsw_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmaddubsw_lowering(&self) -> bool {
        false
    }

    fn default_argument_extension(&self) -> ir::ArgumentExtension {
        ir::ArgumentExtension::Sext
    }

    fn remove_constant_phis(&self) -> bool {
        false
    }
}

impl fmt::Display for ScryBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.name())
            .field("triple", &self.triple())
            .field("flags", &format!("{}", self.flags()))
            .finish()
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    match triple.architecture {
        Architecture::Scry(_) => {}
        _ => unreachable!(),
    }
    IsaBuilder {
        triple,
        setup: scry_settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    triple: Triple,
    shared_flags: Flags,
    builder: &shared_settings::Builder,
) -> CodegenResult<OwnedTargetIsa> {
    let isa_flags = scry_settings::Flags::new(&shared_flags, builder);

    let backend = ScryBackend::new_with_flags(triple, shared_flags, isa_flags);
    Ok(backend.wrapped())
}
