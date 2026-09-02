//! This module defines sc-specific machine instruction types.

use crate::binemit::{Addend, CodeOffset, Reloc};
pub use crate::ir::condcodes::IntCC;

pub use crate::ir::Type;
use crate::isa::FunctionAlignment;
use crate::machinst::*;
use crate::{CodegenError, CodegenResult, settings};

pub use crate::ir::condcodes::FloatCC;

use alloc::string::String;
use alloc::vec::Vec;
use regalloc2::{RegClass, VReg};
use scry_isa::{Alu2Variant, AluVariant, Bits, Instruction};
use std::cmp::min;
use std::iter::once;
use std::ops::RangeInclusive;

pub mod args;
pub mod emit;
pub use self::emit::*;

use crate::isa::scry::abi::ScryMachineDeps;

pub use crate::isa::scry::lower::isle::generated_code::{
    BinaryAluOp, DoubleAluOp, IssueKind, MInst, ResizeVariant, UnaryAluOp,
};
use crate::opts::{I8, I16, I32, I64};

use crate::machinst::isle::WritableReg;
use byteorder::{ByteOrder, LittleEndian};
//=============================================================================
// Instructions (top level): definition

impl MachInst for MInst {
    type LabelUse = LabelUse;
    type ABIMachineSpec = ScryMachineDeps;

    // https://github.com/riscv/riscv-isa-manual/issues/850
    // all zero will cause invalid opcode.
    const TRAP_OPCODE: &'static [u8] = &[0; 4];

    fn gen_dummy_use(_reg: Reg) -> Self {
        unimplemented!()
    }

    fn canonical_type_for_rc(_rc: RegClass) -> Type {
        unimplemented!()
    }

    fn is_safepoint(&self) -> bool {
        false
    }

    fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        use MInst::*;
        match self {
            Nop | Trap | Ret { .. } | ImmJump { .. } | StackAdjust { .. } => (),
            StoreStack { rs, .. } | StoreStackArg { rs, .. } => {
                collector.reg_use(rs);
            }
            SAddr { rd, .. } | LoadStack { rd, .. } | LoadExtName { rd, .. } => {
                collector.reg_def(rd);
            }
            Discard { rss } => {
                rss.iter_mut().for_each(|r| {
                    collector.reg_use(r);
                });
            }
            Args { args } => {
                // We just treat function arguments as definition points
                for p in args {
                    collector.reg_def(&mut p.vreg);
                }
            }
            Alu1 { rd, rss, .. } => {
                collector.reg_def(rd);
                rss.iter_mut().for_each(|r| {
                    collector.reg_use(r);
                });
            }
            BinaryAlu { rd, rs1, rs2, .. } | IntCmp { rd, rs1, rs2, .. } => {
                collector.reg_def(rd);
                collector.reg_use(rs1);
                collector.reg_use(rs2);
            }
            DoubleAlu {
                rdl, rdh, rs1, rs2, ..
            } => {
                collector.reg_def(rdl);
                collector.reg_def(rdh);
                collector.reg_use(rs1);
                collector.reg_use(rs2);
            }
            Rets { rets } => {
                for p in rets {
                    collector.reg_use(&mut p.vreg);
                }
            }
            Const { rd, .. } => {
                collector.reg_def(rd);
            }
            Echo { rds, rss, .. } | EchoLong { rds, rss, .. } | Alu2 { rds, rss, .. } => {
                rds.iter_mut().for_each(|r| collector.reg_def(r));
                rss.iter_mut().for_each(|r| collector.reg_use(r));
            }
            EchoChain {
                rd1,
                rd2,
                rs1,
                rs2,
                rd_chain,
                rs_chain,
                ..
            } => {
                collector.reg_def(rd1);
                collector.reg_def(rd2);
                rd_chain.iter_mut().for_each(|r| collector.reg_def(r));
                collector.reg_use(rs1);
                collector.reg_use(rs2);
                rs_chain.iter_mut().for_each(|r| collector.reg_use(r));
            }
            EchoSplit {
                rd1, rd2, rs1, rs2, ..
            } => {
                collector.reg_def(rd1);
                collector.reg_def(rd2);
                collector.reg_use(rs1);
                collector.reg_use(rs2);
            }
            Duplicate { rd1, rd2, rs, .. } => {
                collector.reg_def(rd1);
                collector.reg_def(rd2);
                collector.reg_use(rs);
            }
            Reorder {
                rd1, rd2, rs1, rs2, ..
            } => {
                collector.reg_def(rd1);
                collector.reg_def(rd2);
                collector.reg_use(rs1);
                collector.reg_use(rs2);
            }
            Store { rd, rs } => {
                collector.reg_use(rd);
                collector.reg_use(rs);
            }
            UnaryAlu { rd, rs, .. }
            | Load { rd, rs, .. }
            | Resize { rd, rs, .. }
            | Cast { rd, rs, .. } => {
                collector.reg_def(rd);
                collector.reg_use(rs);
            }
            Pick {
                rd,
                cond,
                if_zero,
                if_nonzero,
                ..
            } => {
                collector.reg_use(cond);
                collector.reg_use(if_zero);
                collector.reg_use(if_nonzero);
                collector.reg_def(rd);
            }
            Call { link, fun, .. } => {
                collector.reg_def(link);
                collector.reg_use(fun);
            }
            CallArgs {
                link, rets, args, ..
            } => {
                collector.reg_use(link);
                rets.iter_mut().for_each(|p| collector.reg_def(&mut p.vreg));
                args.iter_mut().for_each(|p| collector.reg_use(&mut p.vreg));
            }
            JumpIssue { link, kind, .. } => {
                collector.reg_def(link);
                match kind {
                    IssueKind::Jump => {}
                    IssueKind::Branch { cond, .. } => collector.reg_use(cond),
                    IssueKind::Far { cond, offset, .. } => {
                        collector.reg_use(cond);
                        collector.reg_use(offset);
                    }
                }
            }
            JumpOffset { rd, .. } => {
                collector.reg_def(rd);
            }
            TrapNz { cond } => {
                collector.reg_use(cond);
            }
            JumpTrigger { link, args, .. } => {
                collector.reg_use(link);
                for r in args {
                    collector.reg_use(r);
                }
            }
        }
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        use MInst::*;
        match self {
            EchoLong { rds, rss, .. } | Echo { rds, rss, .. }
                if rds.len() == 1 && rss.len() == 1 =>
            {
                Some((rds[0], rss[0]))
            }
            Nop
            | Trap
            | Args { .. }
            | Ret { .. }
            | Rets { .. }
            | Const { .. }
            | LoadExtName { .. }
            | Store { .. }
            | Load { .. }
            | StoreStack { .. }
            | StoreStackArg { .. }
            | LoadStack { .. }
            | SAddr { .. }
            | StackAdjust { .. }
            | Call { .. }
            | CallArgs { .. }
            | Duplicate { .. }
            | Reorder { .. }
            | JumpIssue { .. }
            | JumpOffset { .. }
            | TrapNz { .. }
            | JumpTrigger { .. }
            | ImmJump { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | BinaryAlu { .. }
            | DoubleAlu { .. }
            | UnaryAlu { .. }
            | Pick { .. }
            | IntCmp { .. }
            | Resize { .. }
            | Cast { .. }
            | Discard { .. }
            | EchoSplit { .. }
            | EchoChain { .. }
            | Echo { .. }
            | EchoLong { .. } => None,
        }
    }

    fn is_jmp(&self) -> bool {
        matches!(self, MInst::ImmJump { .. })
    }

    fn is_included_in_clobbers(&self) -> bool {
        // Scry does not have to worry about clobbers (no registers)
        false
    }

    fn is_trap(&self) -> bool {
        matches!(self, MInst::Trap | MInst::TrapNz { .. })
    }

    fn is_args(&self) -> bool {
        matches!(self, MInst::Args { .. })
    }

    fn call_type(&self) -> CallType {
        CallType::None
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            MInst::Rets { .. } => MachTerminator::Ret,
            _ => MachTerminator::None,
        }
    }

    fn is_mem_access(&self) -> bool {
        unimplemented!()
    }

    fn gen_move(_to_reg: Writable<Reg>, _from_reg: Reg, _ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_nop(_preferred_size: usize) -> MInst {
        unimplemented!()
    }

    fn gen_nop_units() -> Vec<Vec<u8>> {
        let mut bytes = [0; 2];
        LittleEndian::write_u16(&mut bytes, Instruction::NoOp.encode());
        vec![bytes.to_vec()]
    }

    fn rc_for_type(ty: Type) -> CodegenResult<(&'static [RegClass], &'static [Type])> {
        match ty {
            I8 => Ok((&[RegClass::Int], &[I8])),
            I16 => Ok((&[RegClass::Int], &[I16])),
            I32 => Ok((&[RegClass::Int], &[I32])),
            I64 => Ok((&[RegClass::Int], &[I64])),
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {ty}"
            ))),
        }
    }

    fn gen_jump(dst: MachLabel) -> MInst {
        MInst::ImmJump { dst }
    }

    fn worst_case_size() -> CodeOffset {
        2
    }

    fn worst_case_island_growth() -> CodeOffset {
        // TODO: Copied from RISC-V target with no analysis
        128
    }

    fn function_alignment() -> FunctionAlignment {
        FunctionAlignment {
            minimum: 2,
            preferred: 2,
        }
    }
}

//=============================================================================
// Pretty-printing of instructions.

pub fn reg_name(reg: Reg) -> String {
    format!("v({})", reg.to_virtual_reg().unwrap().index())
}
pub fn wreg_name(reg: Writable<Reg>) -> String {
    format!("v({})", reg.to_reg().to_virtual_reg().unwrap().index())
}

impl IssueKind {
    /// Whether the jump is conditional (a branch).
    pub fn is_conditional(&self) -> bool {
        match self {
            IssueKind::Jump => false,
            IssueKind::Branch { .. } => true,
            IssueKind::Far { conditional, .. } => *conditional,
        }
    }
}

impl MInst {
    fn print_with_state(&self, _state: &mut EmitState) -> String {
        fn join(name: &str, list: impl Iterator<Item = String>) -> String {
            let mut res: String = name.into();
            res.push('(');
            list.for_each(|s| {
                res.push_str(s.as_str());
                res.push_str(", ");
            });
            res.push(')');
            res
        }
        use MInst::*;
        match self {
            Args { args } => join("Args", args.iter().map(|p| wreg_name(p.vreg))),
            Discard { rss } => join(
                "Discard",
                once("rss:".into()).chain(rss.iter().map(|r| reg_name(*r))),
            ),
            Nop => "Nop".into(),
            Trap => "Trap".into(),
            TrapNz { cond } => join("TrapNz", ["cond:".into(), reg_name(*cond)].into_iter()),
            Ret { trig } => join("Ret", once(format!("trig: {trig}"))),
            Rets { rets } => join("Rets", rets.iter().map(|p| reg_name(p.vreg))),
            Alu1 { var, rd, rss, out } => join(
                "Alu1",
                [format!("var: {var:?}"), "rd: ".into(), wreg_name(*rd)]
                    .into_iter()
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("out: {out:?}"))),
            ),
            Alu2 {
                var,
                out_var,
                rds,
                rss,
                outs,
            } => join(
                "Alu2",
                [format!("var: {var:?}, var_out {out_var:?}")]
                    .into_iter()
                    .chain(once("rds:".into()))
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("outs: {outs:?}"))),
            ),
            Const { ty, rd, imm } => join(
                "Const",
                [
                    format!("ty: {ty:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("imm: {}", imm.bits()),
                ]
                .into_iter(),
            ),
            LoadExtName { rd, name, offset } => join(
                "LoadExtName",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("name: {name:?}"),
                    format!("offset: {offset}"),
                ]
                .into_iter(),
            ),
            Echo { rds, rss } => join(
                "Echo",
                once("rds:".into())
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r))),
            ),
            EchoChain {
                rd1,
                rd2,
                rs1,
                rs2,
                out1,
                out2,
                rd_chain,
                rs_chain,
            } => join(
                "EchoChain",
                [
                    "rd1:".into(),
                    wreg_name(*rd1),
                    "rd2:".into(),
                    wreg_name(*rd2),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                    format!("out1: {out1}"),
                    format!("out2: {out2}"),
                ]
                .into_iter()
                .chain(once("rd_chain:".into()))
                .chain(rd_chain.iter().map(|r| wreg_name(*r)))
                .chain(once("rs_chain:".into()))
                .chain(rs_chain.iter().map(|r| reg_name(*r))),
            ),
            EchoSplit {
                rd1,
                rd2,
                rs1,
                rs2,
                out1,
                out2,
            } => join(
                "EchoSplit",
                [
                    "rd1:".into(),
                    wreg_name(*rd1),
                    "rd2:".into(),
                    wreg_name(*rd2),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                    format!("out1: {out1}"),
                    format!("out2: {out2}"),
                ]
                .into_iter(),
            ),
            EchoLong { rds, rss, out } => join(
                "EchoLong",
                once("rds:".into())
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("out: {out}"))),
            ),
            Duplicate {
                rd1,
                rd2,
                rs,
                out1,
                out2,
            } => join(
                "Duplicate",
                [
                    "rd1:".into(),
                    wreg_name(*rd1),
                    "rd2:".into(),
                    wreg_name(*rd2),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out1: {out1}"),
                    format!("out2: {out2}"),
                ]
                .into_iter(),
            ),
            Reorder {
                rd1,
                rd2,
                rs1,
                rs2,
                out,
            } => join(
                "Reorder",
                [
                    "rd1:".into(),
                    wreg_name(*rd1),
                    "rd2:".into(),
                    wreg_name(*rd2),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                    format!("out: {out}"),
                ]
                .into_iter(),
            ),
            Store { rd, rs } => join(
                "Store",
                ["rs:".into(), reg_name(*rs), "rd:".into(), reg_name(*rd)].into_iter(),
            ),
            StoreStack { rs, idx } => join(
                "StoreStack",
                ["rs:".into(), reg_name(*rs), format!("idx: {idx}")].into_iter(),
            ),
            StoreStackArg {
                rs,
                offset,
                scale_pow2,
            } => join(
                "StoreStackArg",
                [
                    "rs:".into(),
                    reg_name(*rs),
                    format!("offset: {offset}"),
                    format!("scale_pow2: {scale_pow2}"),
                ]
                .into_iter(),
            ),
            LoadStack { ty, rd, idx } => join(
                "LoadStack",
                [
                    format!("ty: {ty:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("idx: {idx}"),
                ]
                .into_iter(),
            ),
            SAddr {
                rd,
                scale_pow2,
                idx,
            } => join(
                "SAddr",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("scale_pow2: {scale_pow2}"),
                    format!("idx: {idx}"),
                ]
                .into_iter(),
            ),
            StackAdjust {
                reserve,
                private,
                amount_pow2,
            } => join(
                "StackAdjust",
                [
                    format!("reserve: {reserve}"),
                    format!("private: {private}"),
                    format!("amount_pow2: {amount_pow2}"),
                ]
                .into_iter(),
            ),
            Load { ty, rd, rs, out } => join(
                "Load",
                [
                    format!("ty: {ty:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out: {out}"),
                ]
                .into_iter(),
            ),
            Cast { rd, ty, rs, out } => join(
                "Cast",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("ty: {ty:?}"),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out: {out}"),
                ]
                .into_iter(),
            ),
            Resize { var, rd, rs } => join(
                "Resize",
                [
                    format!("var: {var:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs:".into(),
                    reg_name(*rs),
                ]
                .into_iter(),
            ),
            BinaryAlu { op, rd, rs1, rs2 } => join(
                "BinaryAlu",
                [
                    format!("op: {op:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                ]
                .into_iter(),
            ),
            DoubleAlu {
                op,
                rdl,
                rdh,
                rs1,
                rs2,
            } => join(
                "DoubleAlu",
                [
                    format!("op: {op:?}"),
                    "rdl:".into(),
                    wreg_name(*rdl),
                    "rdh:".into(),
                    wreg_name(*rdh),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                ]
                .into_iter(),
            ),
            Pick {
                rd,
                cond,
                if_zero,
                if_nonzero,
                out,
            } => join(
                "Pick",
                [
                    "rd:".into(),
                    reg_name(rd.to_reg()),
                    "cond:".into(),
                    reg_name(*cond),
                    "if_zero:".into(),
                    reg_name(*if_zero),
                    "if_nonzero:".into(),
                    reg_name(*if_nonzero),
                    format!("out: {out}"),
                ]
                .into_iter(),
            ),
            UnaryAlu { op, rd, rs, .. } => join(
                "UnaryAlu",
                [
                    format!("op: {op:?}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs:".into(),
                    reg_name(*rs),
                ]
                .into_iter(),
            ),
            IntCmp { cc, rd, rs1, rs2 } => join(
                "IntCmp",
                [
                    format!("cc: {cc}"),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                ]
                .into_iter(),
            ),
            Call { link, fun, trig } => join(
                "Call",
                [
                    "link:".into(),
                    wreg_name(*link),
                    "fn_ptr:".into(),
                    reg_name(*fun),
                    format!("trig: {trig}"),
                ]
                .into_iter(),
            ),
            CallArgs {
                link,
                rets,
                args,
                sig,
            } => join(
                "CallArgs",
                ["link:".into(), reg_name(*link), "rets:".into()]
                    .into_iter()
                    .chain(rets.iter().map(|p| wreg_name(p.vreg)))
                    .chain(once("args:".into()))
                    .chain(args.iter().map(|p| reg_name(p.vreg)))
                    .chain(once(format!("sig: {sig:?}"))),
            ),
            JumpIssue {
                link,
                dst,
                kind,
                trig,
            } => {
                let kind = match kind {
                    IssueKind::Jump => String::from("Jump"),
                    IssueKind::Branch { dir, cond } => {
                        format!("Branch(dir: {dir:?}, cond: {})", reg_name(*cond))
                    }
                    IssueKind::Far {
                        conditional,
                        dir,
                        cond,
                        offset,
                    } => format!(
                        "Far(conditional: {conditional:?}, dir: {dir:?}, cond: {}, offset: {})",
                        reg_name(*cond),
                        reg_name(*offset)
                    ),
                };
                join(
                    "JumpIssue",
                    [
                        "link:".into(),
                        wreg_name(*link),
                        format!("kind: {kind}, dst: {dst:?}, trig: {trig}"),
                    ]
                    .into_iter(),
                )
            }
            JumpOffset {
                rd,
                dst,
                size_pow2,
                gap,
                trig,
            } => join(
                "JumpOffset",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("dst: {dst:?}, size_pow2: {size_pow2}, gap: {gap}, trig: {trig}"),
                ]
                .into_iter(),
            ),
            JumpTrigger { link, args } => join(
                "JumpTrigger",
                ["link:".into(), reg_name(*link)]
                    .into_iter()
                    .chain(once("args:".into()))
                    .chain(args.iter().map(|r| reg_name(*r))),
            ),
            ImmJump { dst } => join("ImmJump", [format!("dst: {dst:?}")].into_iter()),
        }
    }

    #[duplicate::duplicate_item[
        name            reference(type) iterate;
        [get_uses]      [& type]        [iter];
        [get_uses_mut]  [&mut type]     [iter_mut];
    ]]
    /// Returns the registers used by this instruction
    pub(crate) fn name(self: reference([Self])) -> impl Iterator<Item = reference([Reg])> {
        use MInst::*;
        match self {
            Nop
            | Trap
            | Ret { .. }
            | Args { .. }
            | Const { .. }
            | LoadExtName { .. }
            | JumpIssue {
                kind: IssueKind::Jump,
                ..
            }
            | JumpOffset { .. }
            | ImmJump { .. }
            | LoadStack { .. }
            | SAddr { .. }
            | StackAdjust { .. } => {
                vec![]
            }
            StoreStack { rs, .. } | StoreStackArg { rs, .. } => vec![rs],
            Reorder { rs1, rs2, .. } => {
                vec![rs1, rs2]
            }
            Rets { rets } => rets
                .iterate()
                .map(|p| reference([(p.vreg)]))
                .collect::<Vec<_>>(),
            Echo { rss, .. }
            | EchoLong { rss, .. }
            | Alu1 { rss, .. }
            | Alu2 { rss, .. }
            | Discard { rss } => rss.iterate().collect::<Vec<_>>(),
            EchoChain {
                rs1, rs2, rs_chain, ..
            } => {
                let mut uses = vec![rs1, rs2];
                uses.extend(rs_chain.iterate());
                uses
            }
            EchoSplit { rs1, rs2, .. } => vec![rs1, rs2],
            BinaryAlu { rs1, rs2, .. } | DoubleAlu { rs1, rs2, .. } | IntCmp { rs1, rs2, .. } => {
                vec![rs1, rs2]
            }

            UnaryAlu { rs, .. }
            | Duplicate { rs, .. }
            | Load { rs, .. }
            | Resize { rs, .. }
            | Cast { rs, .. } => {
                vec![rs]
            }
            Pick {
                cond,
                if_zero,
                if_nonzero,
                ..
            } => vec![cond, if_zero, if_nonzero],
            Store { rd, rs } => vec![rs, rd],
            Call { fun, .. } => vec![fun],
            CallArgs { args, .. } => {
                let mut uses = Vec::new();
                uses.extend(args.iterate().map(|p| reference([(p.vreg)])));
                uses
            }
            JumpIssue {
                kind: IssueKind::Branch { cond, .. },
                ..
            } => vec![cond],
            // The machine's operand order: condition first, then the target.
            JumpIssue {
                kind: IssueKind::Far { cond, offset, .. },
                ..
            } => vec![cond, offset],
            TrapNz { cond } => vec![cond],
            JumpTrigger { args, .. } => args.iterate().collect(),
        }
        .into_iter()
    }

    pub(crate) fn use_order_meaningful(&self) -> bool {
        use MInst::*;
        match self {
            // Commutative operations
            BinaryAlu { op, .. }
                if matches!(
                    op,
                    BinaryAluOp::IntAddWrap
                        | BinaryAluOp::IntMulWrap
                        | BinaryAluOp::BitAnd
                        | BinaryAluOp::BitOr
                        | BinaryAluOp::BitXor
                ) =>
            {
                false
            }
            DoubleAlu { op, .. }
                if matches!(
                    op,
                    DoubleAluOp::SaddOverflow
                        | DoubleAluOp::UaddOverflow
                        | DoubleAluOp::UmulHi
                        | DoubleAluOp::SmulHi
                ) =>
            {
                false
            }
            Alu2 { var, .. } if matches!(var, Alu2Variant::Add | Alu2Variant::Multiply) => false,
            Alu1 { var, .. }
                if matches!(
                    var,
                    AluVariant::BitAnd | AluVariant::BitOr | AluVariant::BitXor
                ) =>
            {
                false
            }
            _ => true,
        }
    }

    /// Returns the registers defined by this instruction.
    ///
    /// The defs are returned in physical production order, i.e. the order in which the
    /// values are emitted by the machine and therefore arrive at a shared consumer.
    /// Ordering passes rely on this.
    pub(crate) fn get_defs(&self) -> impl Iterator<Item = Reg> + use<> {
        self.get_def_wregs()
            .map(|wr| wr.to_reg())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[duplicate::duplicate_item[
        name            reference(type) iterate;
        [get_def_wregs] [& type]        [iter];
        [get_defs_mut]  [&mut type]     [iter_mut];
    ]]
    /// Returns the writable registers defined by this instruction, in
    /// physical production order (see [Self::get_defs]).
    pub(crate) fn name(self: reference([Self])) -> impl Iterator<Item = reference([WritableReg])> {
        use MInst::*;
        match self {
            Nop
            | Trap
            | Rets { .. }
            | Ret { .. }
            | Store { .. }
            | StoreStack { .. }
            | StoreStackArg { .. }
            | StackAdjust { .. }
            | JumpTrigger { .. }
            | ImmJump { .. }
            | Call { .. }
            | JumpIssue { .. }
            | TrapNz { .. }
            | Discard { .. } => vec![],
            Echo { rds, .. } | EchoLong { rds, .. } | Alu2 { rds, .. } => {
                rds.iterate().collect::<Vec<_>>()
            }
            Duplicate { rd1, rd2, .. } | EchoSplit { rd1, rd2, .. } => {
                vec![rd1, rd2]
            }
            DoubleAlu { rdl, rdh, .. } => {
                // Physical production order: when both outputs go to the same
                // consumer, the machine delivers the low output before the
                // high one (the resolved Alu2 uses LowFirst in that case).
                vec![rdl, rdh]
            }
            Reorder { rd1, rd2, .. } => {
                // Physical production order: a reorder is an echo with both outputs on
                // the same target, which emits its second input before its first.
                vec![rd2, rd1]
            }
            EchoChain {
                rd1, rd2, rd_chain, ..
            } => {
                // Physical production order: the echo emits its second input before its
                // first, and the chained values are produced by the following echo.
                let mut defs = vec![rd2, rd1];
                defs.extend(rd_chain.iterate());
                defs
            }
            Args { args } => args
                .iterate()
                .map(|a| reference([(a.vreg)]))
                .collect::<Vec<_>>(),
            Alu1 { rd, .. }
            | UnaryAlu { rd, .. }
            | Pick { rd, .. }
            | Load { rd, .. }
            | LoadStack { rd, .. }
            | SAddr { rd, .. }
            | LoadExtName { rd, .. }
            | JumpOffset { rd, .. }
            | BinaryAlu { rd, .. }
            | IntCmp { rd, .. }
            | Resize { rd, .. }
            | Cast { rd, .. }
            | Const { rd, .. } => {
                vec![rd]
            }
            CallArgs { rets, .. } => rets
                .iterate()
                .map(|p| reference([(p.vreg)]))
                .collect::<Vec<_>>(),
        }
        .into_iter()
    }

    /// The bytes of a constant that must be explicitly emitted, in little-endian
    /// order: the last one is the `const` instruction's immediate, the ones before
    /// it are the immediates of successive `grow` instructions (most significant
    /// first). The bytes not returned are produced by the `const` instruction's
    /// sign/zero extension of its immediate.
    pub(crate) fn const_emit_bytes(ty: scry_isa::Type, imm: i64) -> Vec<u8> {
        let size = 1usize << ty.size_pow2();
        let bytes: Vec<u8> = imm.to_le_bytes()[..size].to_vec();
        let signed = ty.is_signed_int();
        let ext_of = |b: u8| if signed && b >= 0x80 { 0xFF } else { 0x00 };
        for m in 1..=size {
            if bytes[m..].iter().all(|b| *b == ext_of(bytes[m - 1])) {
                return bytes[..m].to_vec();
            }
        }
        bytes
    }

    /// Returns how much reference distances increase for in-flight operands crossing this instruction.
    pub(crate) fn reference_length(&self) -> usize {
        use MInst::*;
        match self {
            Args { .. } => 0,
            Const { .. } | LoadExtName { .. } | JumpOffset { .. } | Echo { .. } => {
                self.emitted_length()
            }
            Nop
            | Trap
            | Rets { .. }
            | JumpTrigger { .. }
            | UnaryAlu { .. }
            | Pick { .. }
            | BinaryAlu { .. }
            | IntCmp { .. }
            | Resize { .. }
            | Ret { .. }
            | Call { .. }
            | CallArgs { .. }
            | Reorder { .. }
            | Duplicate { .. }
            | Store { .. }
            | Load { .. }
            | StoreStack { .. }
            | LoadStack { .. }
            | SAddr { .. }
            | StackAdjust { .. }
            | Cast { .. }
            | JumpIssue { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | Discard { .. }
            | EchoLong { .. }
            | EchoSplit { .. }
            | EchoChain { .. } => 1,
            // Flight times count EXECUTED instructions. The trap is only
            // executed on the halting path, so for every value that gets past
            // this instruction only the jump counts.
            TrapNz { .. } => 1,
            DoubleAlu { .. } | ImmJump { .. } | StoreStackArg { .. } => unreachable!(),
        }
    }

    /// Returns how many machine instructions this instruction emits.
    pub(crate) fn emitted_length(&self) -> usize {
        use MInst::*;
        match self {
            Args { .. } | CallArgs { .. } | JumpTrigger { .. } => 0,
            // The return terminator is replaced by the (empty) epilogue at
            // emission.
            Rets { .. } => 0,
            // A constant wider than the 8-bit immediate is emitted as a
            // `const`+`grow` chain (see `const_emit_bytes`).
            Const { ty, imm, .. } => match ty.get_known() {
                Some(t) => Self::const_emit_bytes(t, imm.bits()).len(),
                // Not yet resolved (only happens before type resolution, whose
                // callers do not depend on the exact length).
                None => 1,
            },
            // An address is materialized as a const + 3*grow chain patched by
            // the ScryAbs32 relocation.
            LoadExtName { .. } => 4,
            // A const + grow chain of one instruction per byte of the offset
            JumpOffset { size_pow2, .. } => 1 << size_pow2,
            Echo { rds, rss } => Self::receive_chain(
                rds.iter()
                    .cloned()
                    .zip(rss.iter().cloned())
                    .collect::<Vec<_>>()
                    .as_slice(),
                || Reg::from_virtual_reg(VReg::new(0, RegClass::Int)),
            )
            .len(),
            Nop
            | Trap
            | UnaryAlu { .. }
            | Pick { .. }
            | Ret { .. }
            | Call { .. }
            | Reorder { .. }
            | Duplicate { .. }
            | Store { .. }
            | Load { .. }
            | StoreStack { .. }
            | LoadStack { .. }
            | SAddr { .. }
            | StackAdjust { .. }
            | Cast { .. }
            | JumpIssue { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | Discard { .. }
            | EchoLong { .. }
            | EchoSplit { .. }
            | EchoChain { .. } => 1,
            // A jump over a trap plus the trap.
            TrapNz { .. } => 2,
            // Pseudo-instructions that must have been eliminated by now.
            ImmJump { .. }
            | IntCmp { .. }
            | BinaryAlu { .. }
            | DoubleAlu { .. }
            | Resize { .. }
            | StoreStackArg { .. } => {
                unreachable!("Pseudo-instruction was not eliminated: {:?}", self)
            }
        }
    }

    /// Constructs the echo chain that receives a set of delivered operands
    /// (block parameters or call return values) and reroutes each to its
    /// consumer.
    ///
    /// `ops` are `(destination, delivered operand)` pairs in wire order. Since
    /// one instruction can ingest at most [`QUEUE_CAPACITY`] operands, the
    /// producers deliver the operands spread over the chain's instructions
    /// according to [`delivery_group`]: the first 4 to the first echo, then 2
    /// per following echo. Each echo's ready queue holds the newly delivered
    /// operands (first, their producers having executed earlier) followed by
    /// the operands chained on from the previous echo; it terminally outputs
    /// the first 2 with individual references and chains the rest to the next
    /// instruction. Every following echo therefore has up to 2 queue slots
    /// occupied by the chain, which is what limits the delivery groups after
    /// the first to 2 operands.
    pub(crate) fn receive_chain(
        ops: &[(WritableReg, Reg)],
        mut new_vreg: impl FnMut() -> Reg,
    ) -> Vec<Self> {
        let mut result = Vec::new();

        // The first echo's queue is filled entirely with delivered operands.
        let first_group = min(QUEUE_CAPACITY, ops.len());
        let mut queue: Vec<(WritableReg, Reg)> = ops[..first_group].to_vec();
        let mut remaining = &ops[first_group..];

        while !queue.is_empty() {
            let mut rds = vec![];
            let mut rss = vec![];

            // The first 2 queue operands are terminally output
            let taken = min(2, queue.len());
            for op in &queue[..taken] {
                rds.push(op.0);
                rss.push(op.1);
            }

            // The rest are chained on to the following echo
            let chained = queue[taken..]
                .iter()
                .map(|op| {
                    let new_rd = new_vreg();

                    rds.push(WritableReg::from_reg(new_rd));
                    rss.push(op.1);

                    (op.0, new_rd)
                })
                .collect::<Vec<_>>();

            result.push(if rds.len() == 1 {
                MInst::EchoLong { rds, rss, out: 0 }
            } else if rds.len() == 2 {
                MInst::EchoSplit {
                    rd1: rds[0],
                    rd2: rds[1],
                    rs1: rss[0],
                    rs2: rss[1],
                    out1: 0,
                    out2: 0,
                }
            } else {
                MInst::EchoChain {
                    rd1: rds[0],
                    rd2: rds[1],
                    rs1: rss[0],
                    rs2: rss[1],
                    out1: 0,
                    out2: 0,
                    rd_chain: rds[2..].iter().cloned().collect(),
                    rs_chain: rss[2..].iter().cloned().collect(),
                }
            });

            // The next echo receives as many newly delivered operands as the
            // chained operands leave queue capacity for, followed by the
            // chained operands themselves.
            let next_group = min(remaining.len(), QUEUE_CAPACITY - chained.len());
            queue = remaining[..next_group]
                .iter()
                .cloned()
                .chain(chained)
                .collect();
            remaining = &remaining[next_group..];
        }

        result
    }
}

/// The maximum number of operands an instruction's ready queue can hold;
/// operands delivered beyond this are dropped by the machine.
pub(crate) const QUEUE_CAPACITY: usize = 4;

/// Returns which instruction of a receiving echo chain the operand at
/// wire-order position `wire_idx` must be delivered to (0 meaning the chain's
/// first instruction, e.g. a block's first executed instruction).
///
/// Must mirror [`MInst::receive_chain`], which builds the receiving side of
/// this schedule: the first echo has queue capacity for 4 delivered operands,
/// every following echo has 2 slots occupied by the previous echo's chained
/// operands and so can receive only 2.
pub(crate) fn delivery_group(wire_idx: usize) -> usize {
    if wire_idx < QUEUE_CAPACITY {
        0
    } else {
        1 + (wire_idx - QUEUE_CAPACITY) / 2
    }
}

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
    /// The 7-bit immediate of a short [`Instruction::Jump`].
    JmpLoc7,
    /// 6-bit jump trigger offset. Used in [`Instruction::Jump`] for short range
    JmpTrig6,
    /// The signed offset operand of a far jump (the two-operand
    /// [`Instruction::Jump`]), materialized by a `const` + `grow` chain of
    /// `2^size_pow2` instructions (one per byte, most significant first). The
    /// jump follows `gap` instructions after the chain, and its trigger offset
    /// is `trig`.
    ///
    /// The offset has the same meaning as the short jump's immediate: relative
    /// to the jump instruction when jumping backward (zero or negative), and
    /// past the trigger address when jumping forward (positive). Its width is
    /// chosen by `widen_far_jumps`.
    JmpFar { size_pow2: u16, gap: u16, trig: u16 },
}

impl LabelUse {
    /// The number of instructions in a far jump's offset chain: one per byte
    /// of the offset.
    fn far_chain_len(size_pow2: u16) -> i64 {
        1i64 << size_pow2
    }

    /// The largest positive value a far jump's offset can hold.
    fn far_offset_max(size_pow2: u16) -> i64 {
        (1i64 << ((8 << size_pow2) - 1)) - 1
    }

    /// The range of jump offsets (see [`Self::jump_offset`]) this label use can
    /// encode: the single definition of a jump's reach. `widen_far_jumps`
    /// decides against it, `patch` asserts it, and the backward range reported
    /// to the [`MachBuffer`] derives from it.
    pub(crate) fn offset_range(self) -> RangeInclusive<i64> {
        use LabelUse::*;
        match self {
            JmpLoc7 => -(1 << (7 - 1))..=(1 << (7 - 1)) - 1,
            JmpFar { size_pow2, .. } => {
                let max = Self::far_offset_max(size_pow2);
                -(max + 1)..=max
            }
            JmpTrig6 => unreachable!("Not a jump offset"),
        }
    }

    /// How many instructions before the jump the patched instructions start.
    fn lead(self) -> i64 {
        use LabelUse::*;
        match self {
            JmpLoc7 => 0,
            JmpFar { size_pow2, gap, .. } => Self::far_chain_len(size_pow2) + gap as i64,
            JmpTrig6 => unreachable!("Not a jump offset"),
        }
    }

    /// The offset (immediate or far operand) a jump at instruction address
    /// `jump`, triggering `trig` instructions later, needs to reach the
    /// instruction address `target`, per the ISA's jump semantics. All in
    /// instructions, not bytes.
    ///
    /// Returns `None` for a target that no offset reaches: the jump itself
    /// (offset 0 means "jump to self", not fall-through), anything up to its
    /// trigger, and the fall-through instruction right after the trigger.
    pub(crate) fn jump_offset(jump: i64, trig: i64, target: i64) -> Option<i64> {
        if target <= jump {
            // Backward: doubled and added to the jump's own address (offset 0
            // targets the jump itself).
            Some(target - jump)
        } else {
            // Forward: doubled, incremented, and added to the trigger address.
            let trigger = jump + trig;
            (target >= trigger + 2).then(|| target - trigger - 1)
        }
    }
}

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 2;

    fn max_pos_range(self) -> CodeOffset {
        // Deliberately unlimited. The buffer uses this for the deadline at
        // which it emits an island of veneers for the pending forward jumps,
        // and only otherwise to assert a bound label. This backend has no
        // veneers, and an island (fenced by the `gen_jump` pseudo, and
        // shifting every address after it) would break the exact layout
        // `widen_far_jumps` decided on. That pass enforces the forward reach
        // against [`Self::offset_range`] instead, and `patch` asserts it.
        CodeOffset::MAX / 2
    }

    fn max_neg_range(self) -> CodeOffset {
        use LabelUse::*;
        match self {
            JmpTrig6 => ((1 << (6 - 1)) - 1) * 2,
            // Backward offsets are relative to the jump, `lead` instructions
            // past the patched instructions.
            _ => ((-self.offset_range().start() - self.lead()) * 2) as CodeOffset,
        }
    }

    fn patch_size(self) -> CodeOffset {
        use LabelUse::*;
        match self {
            JmpLoc7 | JmpTrig6 => 2,
            JmpFar { size_pow2, .. } => (Self::far_chain_len(size_pow2) * 2) as CodeOffset,
        }
    }

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        use LabelUse::*;

        assert_eq!(buffer.len(), self.patch_size() as usize);
        assert!(use_offset.is_multiple_of(2) && label_offset.is_multiple_of(2));
        // Instruction addresses of the patched instructions and the target.
        let use_addr = use_offset as i64 / 2;
        let target = label_offset as i64 / 2;

        let insts: Vec<Instruction> = buffer
            .chunks_exact(2)
            .map(|slot| Instruction::decode(LittleEndian::read_u16(slot)))
            .collect();
        log::debug!("Patching {self:?}: use({use_offset:?}) label({label_offset:?}), {insts:?}");

        let patched: Vec<Instruction> = match (self, insts.as_slice()) {
            (JmpTrig6, [Instruction::Jump(imm, _)]) => {
                if target >= use_addr {
                    // Trigger after instruction
                    vec![Instruction::Jump(
                        *imm,
                        ((target - use_addr) as i32).try_into().unwrap(),
                    )]
                } else {
                    unimplemented!()
                }
            }
            (JmpLoc7, [Instruction::Jump(_, trig)]) => {
                vec![
                    match Self::jump_offset(use_addr, trig.value as i64, target) {
                        Some(offset) => Instruction::Jump(
                            (offset as i32).try_into().unwrap_or_else(|_| {
                                panic!(
                                    "Short jump offset {offset} out of range (widen_far_jumps \
                                     should have made this a far jump)"
                                )
                            }),
                            *trig,
                        ),
                        // The jmp instruction cannot express a fall-through.
                        None if target == use_addr + trig.value as i64 + 1 => Instruction::NoOp,
                        None => panic!("Jump at {use_addr} cannot target {target}"),
                    },
                ]
            }
            (
                JmpFar {
                    size_pow2,
                    gap,
                    trig,
                },
                chain,
            ) => {
                // The chain's jump comes `gap` instructions after the chain.
                let jump = use_addr + chain.len() as i64 + gap as i64;
                let offset = Self::jump_offset(jump, trig as i64, target)
                    .unwrap_or_else(|| panic!("Far jump at {jump} cannot target {target}"));
                assert!(
                    self.offset_range().contains(&offset),
                    "Far jump offset {offset} does not fit in {} bits",
                    8 << size_pow2
                );

                // The chain's const takes the most significant byte, each grow
                // the next one down (see `Const` emission).
                let bytes = offset.to_le_bytes();
                chain
                    .iter()
                    .enumerate()
                    .map(|(i, inst)| {
                        let byte = Bits::try_from(bytes[chain.len() - 1 - i] as i32).unwrap();
                        match inst {
                            Instruction::Constant(ty, _) if i == 0 => {
                                Instruction::Constant(*ty, byte)
                            }
                            Instruction::Grow(_) if i > 0 => Instruction::Grow(byte),
                            i => unreachable!("Invalid far jump offset chain instruction: {i:?}"),
                        }
                    })
                    .collect()
            }
            (_, i) => unreachable!("Invalid LabelUse for instruction: {:?}, {:?}", self, i),
        };
        log::debug!("Patched: {patched:?}");
        for (slot, inst) in buffer.chunks_exact_mut(2).zip(patched) {
            LittleEndian::write_u16(slot, inst.encode());
        }
    }

    fn supports_veneer(self) -> bool {
        false
    }

    fn veneer_size(self) -> CodeOffset {
        unimplemented!()
    }

    fn worst_case_veneer_size() -> CodeOffset {
        4
    }

    fn generate_veneer(
        self,
        _buffer: &mut [u8],
        _veneer_offset: CodeOffset,
    ) -> (CodeOffset, LabelUse) {
        unimplemented!()
    }

    fn from_reloc(_reloc: Reloc, _addend: Addend) -> Option<LabelUse> {
        unimplemented!()
    }
}
