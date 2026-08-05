//! ISLE integration glue code for scry lowering.

// Pull in the ISLE generated code.
pub mod generated_code;
use generated_code::MInst;

// Types that the generated ISLE code uses via `use super::*`.
use crate::isa::scry::{IsaType, ScryBackend};
use crate::machinst::Reg;
use crate::machinst::{MachInst, isle::*};
use crate::machinst::{VCodeConstant, VCodeConstantData};
use crate::{
    ir::{
        BlockCall, ExternalName, Inst, InstructionData, MemFlags, Opcode, TrapCode, Value,
        ValueList, immediates::*, types::*,
    },
    isa::scry::inst::*,
    machinst::{ArgPair, CallArgList, CallRetList, InstOutput},
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use regalloc2::PReg;

type BoxExternalName = Box<ExternalName>;
type VecArgPair = Vec<ArgPair>;
type RegVec = Vec<Reg>;
type WritableRegVec = Vec<WritableReg>;
type U16Vec = Vec<u16>;
type AluVariant = scry_isa::AluVariant;
type Alu2Variant = scry_isa::Alu2Variant;
type Alu2OutputVariant = scry_isa::Alu2OutputVariant;

pub(crate) struct ScryIsleContext<'a, 'b, I, B>
where
    I: VCodeInst,
    B: LowerBackend,
{
    pub lower_ctx: &'a mut Lower<'b, I>,
    pub backend: &'a B,
}

impl<'a, 'b> ScryIsleContext<'a, 'b, MInst, ScryBackend> {
    fn new(lower_ctx: &'a mut Lower<'b, MInst>, backend: &'a ScryBackend) -> Self {
        Self { lower_ctx, backend }
    }

    pub(crate) fn dfg(&self) -> &crate::ir::DataFlowGraph {
        &self.lower_ctx.f.dfg
    }
}

impl generated_code::Context for ScryIsleContext<'_, '_, MInst, ScryBackend> {
    isle_lower_prelude_methods!();

    fn emit(&mut self, arg0: &MInst) -> Unit {
        self.lower_ctx.emit(arg0.clone());
    }

    fn emit_ret(&mut self, _arg0: ValueSlice) -> InstOutput {
        unreachable!()
    }

    fn emit_nop_and_empty(&mut self) -> InstOutput {
        self.lower_ctx.emit(MInst::Nop);
        smallvec::smallvec![] // empty InstOutput
    }

    fn emit_nop_unit(&mut self) {
        self.lower_ctx.emit(MInst::Nop);
    }
    fn lower_error(&mut self, arg0: Inst) -> InstOutput {
        let instr_data: InstructionData = self.inst_data_value(arg0);
        unreachable!("No valid lowering rule for instruction: {:?}", instr_data);
    }

    /// Catches any branch instruction with no lowering rule. Throws relevant error
    fn lower_branch_error(&mut self, arg0: Inst, arg1: &MachLabelSlice) -> Unit {
        unreachable!(
            "No valid lowering rule for branch: {:?}, {:?}",
            self.inst_data_value(arg0),
            arg1
        )
    }

    fn gen_machlabel(&mut self, labels: &MachLabelSlice) -> MachLabel {
        assert!(!labels.is_empty());
        labels[0]
    }

    fn isatype_invalid(&mut self) -> IsaType {
        IsaType::Invalid
    }

    #[inline]
    fn imm64_from_offset(&mut self, off: Offset32) -> Imm64 {
        Imm64::new(i64::from(off))
    }

    fn nonzero_imm64(&mut self, imm: Imm64) -> Option<Imm64> {
        if imm.bits() != 0 { Some(imm) } else { None }
    }

    fn get_signature(&mut self, arg0: SigRef) -> Signature {
        self.lower_ctx.dfg().signatures[arg0].clone()
    }

    fn block_call_regs(&mut self, block_call: BlockCall) -> RegVec {
        log::trace!("block_call_regs: {block_call:?}");
        let args: Vec<_> = block_call
            .args(&self.lower_ctx.dfg().value_lists)
            .map(|arg| arg.as_value().unwrap())
            .collect();
        args.into_iter()
            .map(|arg| self.lower_ctx.put_value_in_regs(arg).only_reg().unwrap())
            .collect()
    }
}

/// The main entry point for lowering with ISLE.
pub(crate) fn lower(
    lower_ctx: &mut Lower<MInst>,
    backend: &ScryBackend,
    inst: Inst,
) -> Option<InstOutput> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = ScryIsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower(&mut isle_ctx, inst)
}

/// The main entry point for branch lowering with ISLE.
pub(crate) fn lower_branch(
    lower_ctx: &mut Lower<MInst>,
    backend: &ScryBackend,
    branch: Inst,
    targets: &[MachLabel],
) -> Option<()> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = crate::isa::scry::lower::isle::ScryIsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower_branch(&mut isle_ctx, branch, targets)
}
