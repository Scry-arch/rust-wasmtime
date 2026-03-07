//! Lowering rules for Scry.
use crate::ir::Inst as IRInst;
use crate::isa::scry::ScryBackend;
use crate::isa::scry::lower::isle::generated_code::MInst;
use crate::machinst::lower::*;
use crate::machinst::*;
pub mod isle;

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for ScryBackend {
    type MInst = MInst;

    fn lower(&self, ctx: &mut Lower<MInst>, ir_inst: IRInst) -> Option<InstOutput> {
        isle::lower(ctx, self, ir_inst)
    }

    fn lower_branch(
        &self,
        ctx: &mut Lower<MInst>,
        ir_inst: IRInst,
        targets: &[MachLabel],
    ) -> Option<()> {
        isle::lower_branch(ctx, self, ir_inst, targets)
    }

    fn maybe_pinned_reg(&self) -> Option<Reg> {
        None
    }

    type FactFlowState = ();
}
