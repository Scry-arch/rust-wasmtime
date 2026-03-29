//! Scry Instruction Set Architecture.

use crate::CodegenError;
use crate::dominator_tree::DominatorTree;
use crate::ir::pcc;
use crate::ir::{Function, RelSourceLoc, Type};
use crate::isa::scry::inst::{EmitInfo, MInst};
use crate::isa::scry::settings as scry_settings;
use crate::isa::scry::vcode_patches::{Patch, PatchIterator, VCodePatches};
use crate::isa::unwind::systemv;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, TargetIsa,
};
use crate::machinst::isle::Writable;
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
use regalloc2::Function as RegFunc;
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use target_lexicon::{Architecture, Triple};

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;
mod vcode_patches;

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

impl TargetIsa for ScryBackend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        dbg!(func);
        let (vcode, mut new_vregs) = self.compile_vcode(func, domtree, ctrl_plane)?;
        let mut new_vreg = || {
            new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
        };
        dbg!(&vcode);
        dbg!(vcode.constants.len());
        vcode.constants.iter().for_each(|c| {
            dbg!(c.1);
        });

        let emit_info = EmitInfo::new(self.flags.clone(), self.isa_flags.clone());
        let sigs = SigSet::new::<abi::ScryMachineDeps>(func, &self.flags)?;
        let abi = Callee::<abi::ScryMachineDeps>::new(func, self, &self.isa_flags, &sigs)?;

        let mut patches = VCodePatches::new();
        patches.insert(
            vcode
                .block_insns(vcode.entry_block())
                .iter()
                .rev()
                .next()
                .unwrap(),
            vec![Patch::Before(MInst::Ret { trig: 0 })],
        );

        let entry = vcode.entry_block();

        'a: loop {
            dbg!();
            // Track the position if each register use
            let mut use_pos = HashMap::<Reg, Vec<usize>>::new();
            let mut def_pos = HashMap::<Reg, usize>::new();
            'b: for (i, (minst, inst, idx)) in PatchIterator::new(&vcode, &patches, entry)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .enumerate()
            {
                dbg!(minst);
                // Record all uses
                minst.get_uses().for_each(|r| {
                    use_pos.entry(*r).or_insert_with(Vec::new).push(i);
                });
                if minst.get_defs().next().is_some() {
                    for def in minst.get_defs() {
                        def_pos.insert(def, i);
                        let use_idxs = &use_pos[&def];
                        assert!(use_idxs.iter().all(|idx| *idx <= i));
                        assert!(use_idxs.len() <= 2);

                        dbg!(use_idxs);
                        if use_idxs.len() > 1 {
                            let rd1 = new_vreg();
                            let rd2 = new_vreg();
                            let mut rds = vec![rd1, rd2];
                            let mut replacements = vec![];

                            // Find all old uses and replace them
                            for (i2, (minst2, inst2, _)) in
                                PatchIterator::new(&vcode, &patches, entry)
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .enumerate()
                            {
                                if i == i2 || rds.len() == 0 {
                                    break;
                                }

                                if minst2.get_uses().any(|u| *u == def) {
                                    let mut minst2_clone = minst2.clone();

                                    minst2_clone.get_uses_mut().filter(|u| **u == def).for_each(
                                        |u| {
                                            if let Some(r) = rds.pop() {
                                                *u = r;
                                            }
                                        },
                                    );

                                    replacements.push((inst2, idx, minst2_clone));
                                }
                            }

                            for (inst, idx, repl) in replacements.into_iter() {
                                if let Some(patches) = patches.get_mut(&inst) {
                                    if let Some(idx) = idx {
                                        *patches[idx].extract_mut() = repl;
                                    } else {
                                        if let Some(r) = patches.iter_mut().find(|p| p.is_replace())
                                        {
                                            *r = Patch::Replace(repl);
                                        } else {
                                            patches.push(Patch::Replace(repl));
                                        }
                                    }
                                } else {
                                    patches.insert(inst, vec![Patch::Replace(repl)]);
                                }
                            }

                            let dup = Patch::After(MInst::Duplicate {
                                rd1: Writable::from_reg(rd1),
                                rd2: Writable::from_reg(rd2),
                                rs: def,
                                out1: 0,
                                out2: 0,
                            });
                            match idx {
                                None => {
                                    if let Some(patches) = patches.get_mut(&inst) {
                                        patches.push(dup);
                                    } else {
                                        patches.insert(inst, vec![dup]);
                                    }
                                }
                                _ => unimplemented!(),
                            }

                            continue 'a;
                        }
                    }

                    let get_ref_dist = |def_idx| {
                        let use_idx = use_pos[&minst.get_defs().skip(def_idx).next().unwrap()][0];
                        (i - use_idx - 1) as u16
                    };

                    let patch = match minst {
                        MInst::Add { rd, rs1, rs2, out } if *out != get_ref_dist(0) => {
                            vec![Patch::Replace(MInst::Add {
                                rd: *rd,
                                rs1: *rs1,
                                rs2: *rs2,
                                out: get_ref_dist(0),
                            })]
                        }
                        MInst::Load { rd, rs, out } if *out != get_ref_dist(0) => {
                            vec![Patch::Replace(MInst::Load {
                                rd: *rd,
                                rs: *rs,
                                out: get_ref_dist(0),
                            })]
                        }
                        MInst::Duplicate {
                            rd1,
                            rd2,
                            rs,
                            out1,
                            out2,
                        } if *out1 != get_ref_dist(0) || *out2 != get_ref_dist(1) => {
                            vec![Patch::Replace(MInst::Duplicate {
                                rd1: *rd1,
                                rd2: *rd2,
                                rs: *rs,
                                out1: get_ref_dist(0),
                                out2: get_ref_dist(1),
                            })]
                        }
                        MInst::Args { args }
                            if args
                                .iter()
                                .enumerate()
                                .any(|(i, _)| get_ref_dist(i) != get_ref_dist(0)) =>
                        {
                            vec![Patch::Replace(MInst::Echo {
                                rss: vec![],
                                rds: args.iter().map(|g| g.vreg).collect(),
                                outs: args
                                    .iter()
                                    .enumerate()
                                    .map(|(i, _)| get_ref_dist(i))
                                    .collect(),
                            })]
                        }
                        MInst::Args { args } => vec![Patch::Replace(MInst::Echo {
                            rss: vec![],
                            rds: args.iter().map(|g| g.vreg).collect(),
                            outs: args
                                .iter()
                                .enumerate()
                                .map(|(i, _)| get_ref_dist(i))
                                .collect(),
                        })],
                        MInst::CallArgs { rets, .. } if get_ref_dist(0) > 0 => {
                            vec![Patch::After(MInst::Echo {
                                rss: vec![],
                                rds: rets.iter().map(|g| g.vreg).collect(),
                                outs: rets
                                    .iter()
                                    .enumerate()
                                    .map(|(i, _)| get_ref_dist(i))
                                    .collect(),
                            })]
                        }
                        _ => continue 'b,
                    };
                    dbg!(&patch);

                    let patches = patches.entry(inst).or_insert_with(Vec::new);
                    match idx {
                        None => {
                            for patch in patch {
                                if let (Some(p), true) = (
                                    patches.iter_mut().find(|p| p.is_replace()),
                                    patch.is_replace(),
                                ) {
                                    *p = patch;
                                } else {
                                    patches.push(patch);
                                }
                            }
                        }
                        Some(idx) => {
                            for (i, patch) in patch.into_iter().enumerate() {
                                if patch.is_replace() {
                                    *patches[idx + i].extract_mut() = patch.extract().clone();
                                } else {
                                    patches.insert(idx + i, patch);
                                }
                            }
                        }
                    }
                    continue 'a;
                }
            }

            // Correct any ordering issues
            for (minst, inst, idx) in PatchIterator::new(&vcode, &patches, entry)
                .collect::<Vec<_>>()
                .into_iter()
            {
                use MInst::*;
                match minst {
                    Store { rd, rs } => {
                        if def_pos[rd] > def_pos[rs] {
                            // Reorder is the address precedes the value
                            let rd = *rd;
                            let rs = *rs;
                            let rd1 = new_vreg();
                            let rd2 = new_vreg();

                            patches
                                .entry(inst)
                                .or_insert_with(Vec::new)
                                .push(Patch::Before(Reorder {
                                    rd1: Writable::from_reg(rd1),
                                    rd2: Writable::from_reg(rd2),
                                    rs1: rd,
                                    rs2: rs,
                                    out: 0,
                                }));
                            let new_store = Store { rd: rd1, rs: rd2 };
                            if let Some(idx) = idx {
                                *patches.get_mut(&inst).unwrap()[idx].extract_mut() = new_store;
                            } else {
                                if let Some(og) = patches
                                    .get_mut(&inst)
                                    .unwrap()
                                    .iter_mut()
                                    .find(|p| p.is_replace())
                                {
                                    *og.extract_mut() = new_store;
                                } else {
                                    patches
                                        .get_mut(&inst)
                                        .unwrap()
                                        .push(Patch::Replace(new_store));
                                }
                            }
                            continue 'a;
                        }
                    }
                    _ => (),
                }
            }

            break;
        }

        let mut builder = VCodeBuilder::<inst::MInst>::new(
            sigs,
            abi,
            emit_info,
            BlockLoweringOrder::new(func, domtree, ctrl_plane),
            VCodeConstants::with_capacity(vcode.constants.len()),
            VCodeBuildDirection::Backward,
            2,
        );

        builder.set_entry(entry);
        for inst in vcode.block_insns(entry).iter().rev() {
            if let Some(patches) = patches.get(&inst) {
                patches.iter().for_each(|p| {
                    if let Patch::After(mi) = p {
                        builder.push(mi.clone(), RelSourceLoc::default());
                    }
                })
            }

            let minst = if let Some(rep) = patches
                .get(&inst)
                .map_or(None, |ps| ps.iter().find(|p| p.is_replace()))
            {
                Some(rep.extract())
            } else if patches
                .get(&inst)
                .map_or(true, |ps| !ps.iter().any(Patch::is_delete))
            {
                Some(vcode.index(inst))
            } else {
                None
            };

            if let Some(minst) = minst {
                builder.push(minst.clone(), RelSourceLoc::default());
                // vcode.inst_operands(inst).iter().for_each(|op| {
                //     if let Some(f) = vcode.vreg_fact(op.vreg()) {
                //         builder.vcode.set_vreg_fact(op.vreg(), f.clone());
                //     }
                // });
            }

            if let Some(patches) = patches.get(&inst) {
                patches.iter().for_each(|p| {
                    if let Patch::Before(mi) = p {
                        builder.push(mi.clone(), RelSourceLoc::default());
                    }
                })
            }
        }
        builder.end_bb();

        let vreg_alloc = VRegAllocator::with_capacity(vcode.num_vregs());
        let vcode2 = builder.build(vreg_alloc);

        dbg!(&vcode);
        dbg!(&vcode2);

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

        dbg!(&buffer);
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
