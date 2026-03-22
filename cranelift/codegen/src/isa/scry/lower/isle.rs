//! ISLE integration glue code for riscv64 lowering.

// Pull in the ISLE generated code.
pub mod generated_code;
use generated_code::MInst;

// Types that the generated ISLE code uses via `use super::*`.
use crate::isa::scry::ScryBackend;
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

pub(crate) struct ScryIsleContext<'a, 'b, I, B>
where
    I: VCodeInst,
    B: LowerBackend,
{
    pub lower_ctx: &'a mut Lower<'b, I>,
    #[allow(unused)]
    pub backend: &'a B,
}

impl<'a, 'b> ScryIsleContext<'a, 'b, MInst, ScryBackend> {
    fn new(lower_ctx: &'a mut Lower<'b, MInst>, backend: &'a ScryBackend) -> Self {
        Self { lower_ctx, backend }
    }

    #[allow(unused)]
    pub(crate) fn dfg(&self) -> &crate::ir::DataFlowGraph {
        &self.lower_ctx.f.dfg
    }
}

impl generated_code::Context for ScryIsleContext<'_, '_, MInst, ScryBackend> {
    isle_lower_prelude_methods!();

    fn emit(&mut self, arg0: &MInst) -> Unit {
        dbg!(arg0);
        self.lower_ctx.emit(arg0.clone());
    }

    fn emit_ret(&mut self, _arg0: ValueSlice) -> InstOutput {
        unreachable!()
    }

    fn emit_nop_and_empty(&mut self) -> InstOutput {
        dbg!();
        self.lower_ctx.emit(MInst::Nop);
        smallvec::smallvec![] // empty InstOutput
    }

    fn emit_nop_unit(&mut self) {
        dbg!();
        self.lower_ctx.emit(MInst::Nop);
    }
    fn lower_error(&mut self, arg0: Inst) -> InstOutput {
        let instr_data: InstructionData = self.inst_data_value(arg0);
        unreachable!("No valid lowering rule for instruction: {:?}", instr_data);
    }
    fn lower_branch_error(&mut self, arg0: Inst, arg1: &MachLabelSlice) -> Unit {
        unreachable!("No valid lowering rule for branch: {:?}, {:?}", arg0, arg1);
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
