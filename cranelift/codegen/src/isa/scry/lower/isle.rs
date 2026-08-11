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
        BlockCall, ExternalName, Inst, InstructionData, JumpTable, MemFlags, Opcode, TrapCode,
        Value, ValueList, immediates::*, types::*,
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
        let (ty, instr_data) = self.inst_data_value(arg0);
        unreachable!("No valid lowering rule for instruction: {:?}, {:?}", ty, instr_data);
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

    /// Lowers a `br_table` into one issue per branch, all linked to a single
    /// `JumpTrigger`.
    ///
    /// The default is issued first as an unconditional jump and each case then
    /// issues a conditional jump guarded by `idx == case_index`. Since all of
    /// them name the same trigger, the ISA's rule that the last control flow
    /// issued for a trigger address wins means a matching case overrides the
    /// default, and the case tests are mutually exclusive so at most one of them
    /// ever does. Sharing the trigger also keeps the number of executed
    /// instructions equal on every path, so the block call arguments reach the
    /// target with the same remaining flight time whichever branch wins.
    fn lower_br_table(&mut self, idx: Value, table: JumpTable, targets: &MachLabelSlice) -> Unit {
        let default_call = {
            let branches = &self.lower_ctx.dfg().jump_tables[table];
            assert_eq!(
                targets.len(),
                branches.all_branches().len(),
                "A branch label per jump table branch, the default first"
            );
            branches.default_block()
        };
        let (default_target, case_targets) = targets.split_first().unwrap();

        let idx_reg = self.lower_ctx.put_value_in_regs(idx).only_reg().unwrap();
        let idx_ty = self.lower_ctx.value_ty(idx);

        // The conditions come first so that the branch issues stay contiguous
        let conds: Vec<Reg> = (0..case_targets.len())
            .map(|case| {
                let case_const = self.lower_ctx.alloc_tmp(idx_ty).only_reg().unwrap();
                self.lower_ctx.emit(MInst::Const {
                    rd: case_const,
                    ty: IsaType::Invalid,
                    imm: Imm64::new(case as i64),
                });
                let cond = self.lower_ctx.alloc_tmp(I8).only_reg().unwrap();
                self.lower_ctx.emit(MInst::IntCmp {
                    cc: IntCC::Equal,
                    rd: cond,
                    rs1: idx_reg,
                    rs2: case_const.to_reg(),
                });
                cond.to_reg()
            })
            .collect();

        let link = self.lower_ctx.alloc_tmp(I64).only_reg().unwrap();

        // The default comes before the conditionals
        self.lower_ctx.emit(MInst::JumpIssue {
            link,
            dst: *default_target,
            trig: 0,
        });
        for (case_target, cond) in case_targets.iter().zip(conds) {
            self.lower_ctx.emit(MInst::BranchIssue {
                link,
                dst: *case_target,
                dir: false, // Takes the branch on a logical true
                cond,
                trig: 0,
            });
        }

        let args = self.block_call_regs(default_call);
        self.lower_ctx.emit(MInst::JumpTrigger {
            link: link.to_reg(),
            args,
        })
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
