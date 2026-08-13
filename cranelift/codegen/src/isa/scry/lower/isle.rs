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
        BlockCall, ExternalName, Inst, InstructionData, JumpTable, MemFlags, Opcode, StackSlot,
        TrapCode, Value, ValueList, immediates::*, types::*,
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

    /// The frame offset (relative to the frame base) of the given stack slot
    /// location. Negative offsets (possible through an out-of-bounds access
    /// offset) are represented as-is and can only be reached through address
    /// materialization, never through a scaled index.
    fn stack_frame_offset(&self, ss: StackSlot, slot_off: Offset32, off: Offset32) -> i64 {
        self.lower_ctx.abi().sized_stackslot_offset(ss) as i64
            + i32::from(slot_off) as i64
            + i32::from(off) as i64
    }

    /// Materializes the address of the given frame offset as a value register.
    ///
    /// Emits an `SAddr` reaching the offset as a scaled index. If the offset
    /// cannot be expressed as a scaled index, the frame base address (index 0)
    /// is materialized and the offset added onto it.
    ///
    /// Note that saddr's result is not cast or otherwise adjusted to the CLIF
    /// address type (i32): the machine types it as a full-width unsigned
    /// integer, and narrowing it would corrupt addresses wider than the CLIF
    /// pointer type.
    fn emit_stack_addr(&mut self, frame_offset: i64) -> Reg {
        let scaled = scaled_index(frame_offset);
        let (scale_pow2, idx) = scaled.unwrap_or((0, 0));

        let saddr_rd = self.lower_ctx.alloc_tmp(I32).only_reg().unwrap();
        self.lower_ctx.emit(MInst::SAddr {
            rd: saddr_rd,
            scale_pow2,
            idx,
        });

        if scaled.is_some() {
            saddr_rd.to_reg()
        } else {
            let off_rd = self.lower_ctx.alloc_tmp(I32).only_reg().unwrap();
            self.lower_ctx.emit(MInst::Const {
                ty: IsaType::Invalid,
                rd: off_rd,
                imm: Imm64::new(frame_offset),
            });
            let sum_rd = self.lower_ctx.alloc_tmp(I32).only_reg().unwrap();
            self.lower_ctx.emit(MInst::BinaryAlu {
                op: BinaryAluOp::IntAddWrap,
                rd: sum_rd,
                rs1: saddr_rd.to_reg(),
                rs2: off_rd.to_reg(),
            });
            sum_rd.to_reg()
        }
    }
}

/// Expresses the given frame offset as a (power-of-two scale, index) pair
/// encodable by the stack instructions, preferring the exact access size
/// caller-side by trying the largest scales first. Returns `None` if the
/// offset is not reachable by any scaled index.
fn scaled_index(frame_offset: i64) -> Option<(u16, u16)> {
    if frame_offset < 0 {
        return None;
    }
    (0..=3u16).rev().find_map(|scale_pow2| {
        let scale = 1i64 << scale_pow2;
        let idx = frame_offset / scale;
        (frame_offset % scale == 0 && idx <= 31).then(|| (scale_pow2, idx as u16))
    })
}

/// The scaled stack index for an access of `size` bytes at `frame_offset`, if
/// the offset is a size-aligned index in range of the 5-bit index field.
fn access_index(frame_offset: i64, size: u32) -> Option<u16> {
    let size = size as i64;
    let idx = frame_offset / size;
    (frame_offset >= 0 && frame_offset % size == 0 && idx <= 31).then(|| idx as u16)
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

    fn lower_stack_addr(&mut self, ss: StackSlot, off: Offset32) -> Reg {
        let frame_offset = self.stack_frame_offset(ss, off, Offset32::new(0));
        self.emit_stack_addr(frame_offset)
    }

    fn lower_stack_load(&mut self, ty: Type, ss: StackSlot, slot_off: Offset32, off: Offset32) -> Reg {
        let frame_offset = self.stack_frame_offset(ss, slot_off, off);
        let rd = self.lower_ctx.alloc_tmp(ty).only_reg().unwrap();
        match access_index(frame_offset, ty.bytes()) {
            Some(idx) => {
                self.lower_ctx.emit(MInst::LoadStack {
                    ty: IsaType::Invalid,
                    rd,
                    idx,
                });
            }
            None => {
                let addr = self.emit_stack_addr(frame_offset);
                self.lower_ctx.emit(MInst::Load {
                    ty: IsaType::Invalid,
                    rd,
                    rs: addr,
                    out: 0,
                });
            }
        }
        rd.to_reg()
    }

    fn lower_stack_store(
        &mut self,
        val: Value,
        ss: StackSlot,
        slot_off: Offset32,
        off: Offset32,
    ) -> InstOutput {
        let frame_offset = self.stack_frame_offset(ss, slot_off, off);
        let ty = self.lower_ctx.value_ty(val);
        let rs = self.lower_ctx.put_value_in_regs(val).only_reg().unwrap();
        match access_index(frame_offset, ty.bytes()) {
            Some(idx) => {
                self.lower_ctx.emit(MInst::StoreStack { rs, idx });
            }
            None => {
                let addr = self.emit_stack_addr(frame_offset);
                self.lower_ctx.emit(MInst::Store { rs, rd: addr });
            }
        }
        smallvec::smallvec![]
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
