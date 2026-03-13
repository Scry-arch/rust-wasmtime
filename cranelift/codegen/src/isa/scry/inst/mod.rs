//! This module defines sc-specific machine instruction types.

use crate::binemit::{Addend, CodeOffset, Reloc};
pub use crate::ir::condcodes::IntCC;

pub use crate::ir::Type;
use crate::isa::FunctionAlignment;
use crate::machinst::*;
use crate::{CodegenResult, settings, CodegenError};

pub use crate::ir::condcodes::FloatCC;

use alloc::string::String;
use alloc::vec::Vec;
use regalloc2::{RegClass, VReg};
use scry_isa::Instruction;

pub mod args;
pub mod emit;
pub use self::emit::*;

use crate::isa::scry::abi::ScryMachineDeps;


pub use crate::isa::scry::lower::isle::generated_code::MInst;
use crate::opts::{I16, I32, I64, I8};

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
            Nop | Ret => (),
            Args { args } => {
                // We just treat function arguments as definition points
                for p in args {
                    collector.reg_def(&mut p.vreg);
                }
            }
            Add { rd, rs1, rs2 } => {
                collector.reg_def(rd);
                collector.reg_use(rs1);
                collector.reg_use(rs2);
            }
            Rets { rets } => {
                for p in rets {
                    collector.reg_use(&mut p.vreg);
                }
            }
            Const { rd, ..} => {
                collector.reg_def(rd);
            }
            
        }
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        use MInst::*;
        match self {
            Nop | Args {..} | Ret | Rets {..} | Add {..} | Const {..} => None,
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
        unimplemented!()
    }

    fn call_type(&self) -> CallType {
        match self {
            _ => CallType::None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            MInst::Ret => MachTerminator::Ret,
            _ => MachTerminator::None
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

    fn gen_nop_units() -> Vec<Vec<u8>>{
        let mut bytes = [0;2];
        LittleEndian::write_u16(&mut bytes, Instruction::NoOp.encode());
        vec![bytes.to_vec()]
    }

    fn rc_for_type(ty: Type) -> CodegenResult<(&'static [RegClass], &'static [Type])> {
        match ty {
            I8 => Ok((&[RegClass::Int], &[I8])),
            I16 => Ok((&[RegClass::Int], &[I16])),
            I32 => Ok((&[RegClass::Int], &[I32])),
            I64 => Ok((&[RegClass::Int], &[I64])),
            _=> Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {ty}"
            )))
        }
    }

    fn gen_jump(_target: MachLabel) -> MInst{
        unimplemented!()
    }

    fn worst_case_size() -> CodeOffset {
        2
    }

    fn ref_type_regclass(_settings: &settings::Flags) -> RegClass{
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
            Args { args } => {
                join("Args", args.iter().map(|p| wreg_name(p.vreg)))
            },
            Nop => "Nop".into(),
            Ret => "Ret".into(),
            Rets { rets } => {
                join("Rets", rets.iter().map(|p| reg_name(p.vreg)))
            },
            Add { rd, rs1, rs2 } => {
                join("Add", [wreg_name(*rd), reg_name(*rs1), reg_name(*rs2)].into_iter())
            },
            Const {rd, imm} => {
                join("Const", [wreg_name(*rd),format!("{}", imm.bits())].into_iter())
            }
        }
    }
}

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
}

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

    fn worst_case_veneer_size() -> CodeOffset{
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
