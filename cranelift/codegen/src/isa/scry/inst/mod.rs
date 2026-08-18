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
use scry_isa::{Alu2Variant, Instruction};
use std::cmp::min;
use std::iter::once;

pub mod args;
pub mod emit;
pub use self::emit::*;

use crate::isa::scry::abi::ScryMachineDeps;

pub use crate::isa::scry::lower::isle::generated_code::{
    BinaryAluOp, DoubleAluOp, MInst, ResizeVariant, UnaryAluOp,
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
            Nop | Ret { .. } | ImmJump { .. } | StackAdjust { .. } => (),
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
            JumpIssue { link, .. } => {
                collector.reg_def(link);
            }
            BranchIssue { link, cond, .. } => {
                collector.reg_def(link);
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
            | BranchIssue { .. }
            | JumpTrigger { .. }
            | ImmJump { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | BinaryAlu { .. }
            | DoubleAlu { .. }
            | UnaryAlu { .. }
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
        // TODO: implement for the trap instruction
        false
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
            LoadExtName { rd, name } => join(
                "LoadExtName",
                ["rd:".into(), wreg_name(*rd), format!("name: {name:?}")].into_iter(),
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
            UnaryAlu { op, rd, rs } => join(
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
            JumpIssue { link, dst, trig } => join(
                "JumpIssue",
                [
                    "link:".into(),
                    wreg_name(*link),
                    format!("dst: {dst:?}, trig: {trig}"),
                ]
                .into_iter(),
            ),
            BranchIssue {
                link,
                cond,
                dir,
                dst,
                trig,
            } => join(
                "BranchIssue",
                [
                    "link:".into(),
                    wreg_name(*link),
                    "cond:".into(),
                    reg_name(*cond),
                    format!("dir: {dir:?}, dst: {dst:?}, trig: {trig}"),
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
            | Ret { .. }
            | Args { .. }
            | Const { .. }
            | LoadExtName { .. }
            | JumpIssue { .. }
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
            Store { rd, rs } => vec![rs, rd],
            Call { fun, .. } => vec![fun],
            CallArgs { args, .. } => {
                let mut uses = Vec::new();
                uses.extend(args.iterate().map(|p| reference([(p.vreg)])));
                uses
            }
            BranchIssue { cond, .. } => vec![cond],
            JumpTrigger { args, .. } => args.iterate().collect(),
        }
        .into_iter()
    }

    pub(crate) fn use_order_meaningful(&self) -> bool {
        use MInst::*;
        match self {
            BinaryAlu { op, .. } if *op == BinaryAluOp::IntAddWrap => false,
            Alu2 { var, .. } if *var == Alu2Variant::Add => false,
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
            | BranchIssue { .. }
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
                // high one (the resolved Alu2 uses FirstLow in that case).
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
            | Load { rd, .. }
            | LoadStack { rd, .. }
            | SAddr { rd, .. }
            | LoadExtName { rd, .. }
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
            Const { .. } | LoadExtName { .. } | Echo { .. } => self.emitted_length(),
            Nop
            | Rets { .. }
            | JumpTrigger { .. }
            | UnaryAlu { .. }
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
            | BranchIssue { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | Discard { .. }
            | EchoLong { .. }
            | EchoSplit { .. }
            | EchoChain { .. } => 1,
            DoubleAlu { .. } | ImmJump { .. } | StoreStackArg { .. } => unreachable!(),
        }
    }

    /// Returns how many machine instructions this instruction emits.
    pub(crate) fn emitted_length(&self) -> usize {
        use MInst::*;
        match self {
            Args { .. } | CallArgs { .. } | JumpTrigger { .. } => 0,
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
            | UnaryAlu { .. }
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
            | BranchIssue { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | Discard { .. }
            | EchoLong { .. }
            | EchoSplit { .. }
            | EchoChain { .. } => 1,
            // Pseudo-instructions that must have been eliminated by now.
            Rets { .. }
            | ImmJump { .. }
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
    /// 7-bit jump location offset. Used in [`Instruction::Jump`] for short range
    JmpLoc7,
    /// 6-bit jump trigger offset. Used in [`Instruction::Jump`] for short range
    JmpTrig6,
}

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 2;

    fn max_pos_range(self) -> CodeOffset {
        use LabelUse::*;
        match self {
            JmpLoc7 => {
                // The positive range is calculated after the trigger address and therefore
                // gets a higher range based on it
                ((1 << (6 + 7 - 1)) - 1) * 2
            }
            JmpTrig6 => ((1 << (6 - 1)) - 1) * 2,
        }
    }

    fn max_neg_range(self) -> CodeOffset {
        use LabelUse::*;
        match self {
            JmpLoc7 => ((1 << (7 - 1)) - 1) * 2,
            JmpTrig6 => ((1 << (6 - 1)) - 1) * 2,
        }
    }

    fn patch_size(self) -> CodeOffset {
        use LabelUse::*;
        match self {
            JmpLoc7 | JmpTrig6 => 2,
        }
    }

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        use LabelUse::*;
        assert_eq!(buffer.len(), 2);

        let inst = Instruction::decode(LittleEndian::read_u16(buffer));

        log::debug!("Patching {self:?}: use({use_offset:?}) label({label_offset:?}), {inst:?}");

        let patched = match (self, inst) {
            (JmpTrig6, Instruction::Jump(imm, _)) => {
                if label_offset >= use_offset {
                    // Trigger after instruction
                    let diff = label_offset - use_offset;
                    assert!(diff.is_multiple_of(2));
                    let diff = diff / 2;
                    Instruction::Jump(imm, (diff as i32).try_into().unwrap())
                } else {
                    unimplemented!()
                }
            }
            (JmpLoc7, Instruction::Jump(_, trig)) => {
                if trig.value >= 0 {
                    // Trigger after instruction
                    let trig_offset = use_offset + trig.value as u32 * 2;

                    if label_offset >= trig_offset {
                        let diff = label_offset - trig_offset;
                        assert!(diff.is_multiple_of(2));
                        let diff = diff / 2;

                        if diff == 1 {
                            // The jmp instruction cannot be used for fallthrough
                            Instruction::NoOp
                        } else {
                            Instruction::Jump((diff as i32 - 1).try_into().unwrap(), trig)
                        }
                    } else {
                        let diff = trig_offset - label_offset;
                        assert!(diff.is_multiple_of(2));
                        let diff = diff / 2;

                        Instruction::Jump((-(diff as i32)).try_into().unwrap(), trig)
                    }
                } else {
                    unimplemented!()
                }
            }
            (_, i) => unreachable!("Invalid LabelUse for instruction: {:?}, {:?}", self, i),
        };
        log::debug!("Patched: {patched:?}");
        LittleEndian::write_u16(buffer, patched.encode());
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
