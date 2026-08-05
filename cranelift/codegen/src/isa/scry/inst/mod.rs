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
    BinaryAluOp, MInst, ResizeVariant, UnaryAluOp,
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
            Nop | Ret { .. } | ImmJump { .. } => (),
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
            | Store { .. }
            | Load { .. }
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
        match self {
            MInst::ImmJump { .. } => true,
            _ => false,
        }
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
        match self {
            MInst::Args { .. } => true,
            _ => false,
        }
    }

    fn call_type(&self) -> CallType {
        match self {
            _ => CallType::None,
        }
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

    fn ref_type_regclass(_settings: &settings::Flags) -> RegClass {
        unimplemented!()
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
#[allow(unused)]
pub fn vreg_name(reg: VReg) -> String {
    format!("v({})", reg.vreg())
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
            Ret { trig } => join("Ret", once(format!("trig: {}", trig))),
            Rets { rets } => join("Rets", rets.iter().map(|p| reg_name(p.vreg))),
            Alu1 { var, rd, rss, out } => join(
                "Alu1",
                [format!("var: {:?}", var), "rd: ".into(), wreg_name(*rd)]
                    .into_iter()
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("out: {:?}", out))),
            ),
            Alu2 {
                var,
                out_var,
                rds,
                rss,
                outs,
            } => join(
                "Alu2",
                [format!("var: {:?}, var_out {:?}", var, out_var)]
                    .into_iter()
                    .chain(once("rds:".into()))
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("outs: {:?}", outs))),
            ),
            Const { ty, rd, imm } => join(
                "Const",
                [
                    format!("ty: {:?}", ty),
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("imm: {}", imm.bits()),
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
                    format!("out1: {}", out1),
                    format!("out2: {}", out2),
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
                    format!("out1: {}", out1),
                    format!("out2: {}", out2),
                ]
                .into_iter(),
            ),
            EchoLong { rds, rss, out } => join(
                "EchoLong",
                once("rds:".into())
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("out: {}", out))),
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
                    format!("out1: {}", out1),
                    format!("out2: {}", out2),
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
                    format!("out: {}", out),
                ]
                .into_iter(),
            ),
            Store { rd, rs } => join(
                "Store",
                ["rs:".into(), reg_name(*rs), "rd:".into(), reg_name(*rd)].into_iter(),
            ),
            Load { ty, rd, rs, out } => join(
                "Load",
                [
                    format!("ty: {:?}", ty).into(),
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out: {}", out),
                ]
                .into_iter(),
            ),
            Cast { rd, ty, rs, out } => join(
                "Cast",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    format!("ty: {:?}", ty),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out: {}", out),
                ]
                .into_iter(),
            ),
            Resize { var, rd, rs } => join(
                "Resize",
                [
                    format!("var: {:?}", var),
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
                    format!("op: {:?}", op),
                    "rd:".into(),
                    wreg_name(*rd),
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
                    format!("op: {:?}", op).into(),
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
                    format!("cc: {}", cc),
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
                    format!("trig: {}", trig),
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
                    .chain(once(format!("sig: {:?}", sig))),
            ),
            JumpIssue { link, dst } => join(
                "JumpIssue",
                ["link:".into(), wreg_name(*link), format!("dst: {:?}", dst)].into_iter(),
            ),
            BranchIssue {
                link,
                cond,
                dir,
                dst,
            } => join(
                "BranchIssue",
                [
                    "link:".into(),
                    wreg_name(*link),
                    "cond:".into(),
                    reg_name(*cond),
                    format!("dir: {:?}, dst: {:?}", dir, dst),
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
            ImmJump { dst } => join("ImmJump", [format!("dst: {:?}", dst)].into_iter()),
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
            Nop | Ret { .. } | Args { .. } | Const { .. } | JumpIssue { .. } | ImmJump { .. } => {
                vec![]
            }
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
            BinaryAlu { rs1, rs2, .. } | IntCmp { rs1, rs2, .. } => vec![rs1, rs2],

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
        use MInst::*;
        match self {
            Nop
            | Rets { .. }
            | Ret { .. }
            | Store { .. }
            | JumpTrigger { .. }
            | ImmJump { .. }
            | Call { .. }
            | JumpIssue { .. }
            | BranchIssue { .. }
            | Discard { .. } => vec![],
            Echo { rds, .. } | EchoLong { rds, .. } | Alu2 { rds, .. } => {
                rds.iter().map(|wr| wr.to_reg()).collect::<Vec<_>>()
            }
            Duplicate { rd1, rd2, .. } | EchoSplit { rd1, rd2, .. } => {
                vec![rd1.to_reg(), rd2.to_reg()]
            }
            Reorder { rd1, rd2, .. } => {
                // Physical production order: a reorder is an echo with both outputs on
                // the same target, which emits its second input before its first.
                vec![rd2.to_reg(), rd1.to_reg()]
            }
            EchoChain {
                rd1, rd2, rd_chain, ..
            } => {
                // Physical production order: the echo emits its second input before its
                // first, and the chained values are produced by the following echo.
                let mut defs = vec![rd2.to_reg(), rd1.to_reg()];
                defs.extend(rd_chain.iter().map(|r| r.to_reg()));
                defs
            }
            Args { args } => args.iter().map(|a| a.vreg.to_reg()).collect(),
            Alu1 { rd, .. }
            | UnaryAlu { rd, .. }
            | Load { rd, .. }
            | BinaryAlu { rd, .. }
            | IntCmp { rd, .. }
            | Resize { rd, .. }
            | Cast { rd, .. }
            | Const { rd, .. } => {
                vec![rd.to_reg()]
            }
            CallArgs { rets, .. } => rets.iter().map(|p| p.vreg.to_reg()).collect(),
        }
        .into_iter()
    }

    /// Returns how much reference distances increase for in-flight operands crossing this instruction.
    pub(crate) fn reference_length(&self) -> usize {
        use MInst::*;
        match self {
            Args { .. } => 0,
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
            | Const { .. }
            | Reorder { .. }
            | Duplicate { .. }
            | Store { .. }
            | Load { .. }
            | Cast { .. }
            | JumpIssue { .. }
            | BranchIssue { .. }
            | Alu1 { .. }
            | Alu2 { .. }
            | Discard { .. }
            | EchoLong { .. }
            | EchoSplit { .. }
            | EchoChain { .. } => 1,
            Echo { rds, rss } => Self::echo_chain(
                rds.iter()
                    .cloned()
                    .zip(rss.iter().cloned())
                    .collect::<Vec<_>>()
                    .as_slice(),
                || Reg::from_virtual_reg(VReg::new(0, RegClass::Int)),
            )
            .len(),
            ImmJump { .. } => unreachable!(),
        }
    }

    /// Constructs an echo chain that splits all input/output/ref triplets into a most 3 per
    pub(crate) fn echo_chain(
        ops: &[(WritableReg, Reg)],
        new_vreg: impl FnMut() -> Reg,
    ) -> Vec<Self> {
        Self::echo_chain_impl(ops, new_vreg, Vec::new())
    }

    /// Constructs an echo chain that splits all input/output/ref triplets into a most 3 per
    pub fn echo_chain_impl(
        ops: &[(WritableReg, Reg)],
        mut new_vreg: impl FnMut() -> Reg,
        mut result: Vec<Self>,
    ) -> Vec<Self> {
        if ops.len() == 0 {
            return result;
        }

        let mut rds = vec![];
        let mut rss = vec![];

        // The first 2 are kept as is
        let mut insert = |op: &(WritableReg, Reg)| {
            rds.push(op.0);
            rss.push(op.1);
        };
        let ops_taken = min(2, ops.len());
        ops.iter().take(ops_taken).for_each(&mut insert);

        // The rest are rerouted to a following echo
        let map_to_next = |op: (WritableReg, Reg)| {
            let new_rd = new_vreg();

            rds.push(WritableReg::from_reg(new_rd));
            rss.push(op.1);

            (op.0, new_rd)
        };

        let next_ops = ops[(ops_taken)..]
            .iter()
            .cloned()
            .map(map_to_next)
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

        Self::echo_chain_impl(&next_ops, new_vreg, result)
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
            JmpLoc7 => ((1 << 7 - 1) - 1) * 2,
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

        log::debug!(
            "Patching {:?}: use({:?}) label({:?}), {:?}",
            self,
            use_offset,
            label_offset,
            inst
        );

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
        log::debug!("Patched: {:?}", patched);
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
