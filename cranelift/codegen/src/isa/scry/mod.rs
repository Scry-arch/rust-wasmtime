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
use std::collections::HashMap;
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
    ) -> CodegenResult<VCode<inst::MInst>> {
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

        Ok(vcode)
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
        let vcode = self.compile_vcode(func, domtree, ctrl_plane)?;
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
            // Track the position if each register use
            let mut use_pos = HashMap::<Reg, usize>::new();
            for (i, (minst, inst)) in
                dbg!(PatchIterator::new(&vcode, &patches, entry).collect::<Vec<_>>())
                    .into_iter()
                    .rev()
                    .enumerate()
            {
                dbg!(minst).get_uses().for_each(|r| {
                    assert!(
                        use_pos.insert(r, i).is_none(),
                        "Value used multiple times: {:?}",
                        r
                    )
                });
                dbg!(inst);
                if minst.get_defs().next().is_some() && inst.is_some() {
                    let use_idx = use_pos[&minst.get_defs().next().unwrap()];
                    assert!(use_idx < i);

                    let ref_dist = (i - use_idx - 1) as u16;
                    let new_minst = match minst {
                        MInst::Add { rd, rs1, rs2, .. } => MInst::Add {
                            rd: *rd,
                            rs1: *rs1,
                            rs2: *rs2,
                            out: ref_dist,
                        },
                        MInst::Args { args } => {
                            assert_eq!(args.len(), 1);
                            MInst::EchoLong {
                                ins2: vec![],
                                outs: args.iter().map(|g| g.vreg).collect(),
                                out: ref_dist,
                            }
                        }
                        _ => unreachable!(),
                    };
                    dbg!(&new_minst);
                    if let Some(patches) = patches.get_mut(&inst.unwrap()) {
                        patches.push(Patch::Delete);
                        patches.insert(0, Patch::Before(new_minst));
                    } else {
                        patches
                            .insert(inst.unwrap(), vec![Patch::Delete, Patch::Before(new_minst)]);
                    }
                    continue 'a;
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

            if patches
                .get(&inst)
                .map_or(true, |ps| !ps.iter().any(Patch::is_delete))
            {
                let minst = vcode.index(inst).clone();
                builder.push(minst, RelSourceLoc::default());
                vcode.inst_operands(inst).iter().for_each(|op| {
                    if let Some(f) = vcode.vreg_fact(op.vreg()) {
                        builder.vcode.set_vreg_fact(op.vreg(), f.clone());
                    }
                });
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
        unimplemented!()
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
        Architecture::Scry => {}
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
