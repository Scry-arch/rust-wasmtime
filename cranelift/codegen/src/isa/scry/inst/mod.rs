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
use scry_isa::Instruction;
use std::iter::once;

pub mod args;
pub mod emit;
pub use self::emit::*;

use crate::isa::scry::abi::ScryMachineDeps;

pub use crate::isa::scry::lower::isle::generated_code::MInst;
use crate::opts::{I8, I16, I32, I64};

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
            Nop | Ret { .. } => (),
            Args { args } => {
                // We just treat function arguments as definition points
                for p in args {
                    collector.reg_def(&mut p.vreg);
                }
            }
            Add { rd, rs1, rs2, .. } => {
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
            Echo { rds, rss, .. } => {
                rds.iter_mut().for_each(|r| collector.reg_def(r));
                rss.iter_mut().for_each(|r| {
                    collector.reg_use(r);
                });
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
            Load { rd, rs, .. } => {
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
        }
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        use MInst::*;
        match self {
            Nop
            | Args { .. }
            | Ret { .. }
            | Rets { .. }
            | Add { .. }
            | Const { .. }
            | Store { .. }
            | Load { .. }
            | Call { .. }
            | CallArgs { .. }
            | Duplicate { .. }
            | Reorder { .. } => None,
            Echo { rds, rss, .. } => {
                if rds.len() == 1 && rds.len() == rss.len() {
                    Some((rds[0], rss[0]))
                } else {
                    None
                }
            }
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

    fn gen_jump(_target: MachLabel) -> MInst {
        unimplemented!()
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
            Nop => "Nop".into(),
            Ret { trig } => join("Ret", once(format!("trig: {}", trig))),
            Rets { rets } => join("Rets", rets.iter().map(|p| reg_name(p.vreg))),
            Add { rd, rs1, rs2, out } => join(
                "Add",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs1:".into(),
                    reg_name(*rs1),
                    "rs2:".into(),
                    reg_name(*rs2),
                    format!("out: {}", out),
                ]
                .into_iter(),
            ),
            Const { rd, imm } => join(
                "Const",
                ["rd:".into(), wreg_name(*rd), format!("imm: {}", imm.bits())].into_iter(),
            ),
            Echo { rds, rss, outs } => join(
                "Echo",
                once("rds:".into())
                    .chain(rds.iter().map(|r| wreg_name(*r)))
                    .chain(once("rss:".into()))
                    .chain(rss.iter().map(|r| reg_name(*r)))
                    .chain(once(format!("outs: {:?}", outs))),
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
                ["rd:".into(), reg_name(*rd), "rs:".into(), reg_name(*rs)].into_iter(),
            ),
            Load { rd, rs, out } => join(
                "Load",
                [
                    "rd:".into(),
                    wreg_name(*rd),
                    "rs:".into(),
                    reg_name(*rs),
                    format!("out: {}", out),
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
            CallArgs { link, rets, args } => join(
                "CallArgs",
                ["link:".into(), reg_name(*link), "rets:".into()]
                    .into_iter()
                    .chain(rets.iter().map(|p| wreg_name(p.vreg)))
                    .chain(once("args:".into()))
                    .chain(args.iter().map(|p| reg_name(p.vreg))),
            ),
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
            Nop | Ret { .. } | Args { .. } | Const { .. } => vec![],
            Add { rs1, rs2, .. } | Reorder { rs1, rs2, .. } => {
                vec![rs1, rs2]
            }
            Rets { rets } => rets
                .iterate()
                .map(|p| reference([(p.vreg)]))
                .collect::<Vec<_>>(),
            Echo { rss, .. } => rss.iterate().collect::<Vec<_>>(),
            Duplicate { rs, .. } => vec![rs],
            Store { rd, rs } => vec![rd, rs],
            Load { rs, .. } => vec![rs],
            Call { fun, .. } => vec![fun],
            CallArgs { link, args, .. } => {
                let mut uses = vec![link];
                uses.extend(args.iterate().map(|p| reference([(p.vreg)])));
                uses
            }
        }
        .into_iter()
    }

    /// Returns the registers defined by this instruction
    pub(crate) fn get_defs(&self) -> impl Iterator<Item = Reg> + use<> {
        use MInst::*;
        match self {
            Nop | Rets { .. } | Ret { .. } | Const { .. } | Store { .. } => vec![],
            Add { rd, .. } | Load { rd, .. } => {
                vec![rd.to_reg()]
            }
            Args { args } => args.iter().map(|p| p.vreg.to_reg()).collect::<Vec<_>>(),
            Echo { rds, .. } => rds.iter().map(|wr| wr.to_reg()).collect::<Vec<_>>(),
            Duplicate { rd1, rd2, .. } | Reorder { rd1, rd2, .. } => {
                vec![rd1.to_reg(), rd2.to_reg()]
            }
            Call { link, .. } => vec![link.to_reg()],
            CallArgs { rets, .. } => rets.iter().map(|p| p.vreg.to_reg()).collect(),
        }
        .into_iter()
    }
}

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {}

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 2;

    fn max_pos_range(self) -> CodeOffset {
        unimplemented!()
    }

    fn max_neg_range(self) -> CodeOffset {
        unimplemented!()
    }

    fn patch_size(self) -> CodeOffset {
        unimplemented!()
    }

    fn patch(self, _buffer: &mut [u8], _use_offset: CodeOffset, _label_offset: CodeOffset) {
        unimplemented!()
    }

    fn supports_veneer(self) -> bool {
        unimplemented!()
    }

    fn veneer_size(self) -> CodeOffset {
        unimplemented!()
    }

    fn worst_case_veneer_size() -> CodeOffset {
        unimplemented!()
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
