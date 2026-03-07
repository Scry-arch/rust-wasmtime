//! This module defines sc-specific machine instruction types.

use crate::binemit::{Addend, CodeOffset, Reloc};
pub use crate::ir::condcodes::IntCC;

pub use crate::ir::{ExternalName, MemFlags, Type};
use crate::isa::{FunctionAlignment};
use crate::machinst::*;
use crate::{CodegenResult, settings};

pub use crate::ir::condcodes::FloatCC;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;
use regalloc2::RegClass;

pub mod args;
pub use self::args::*;
pub mod emit;
pub use self::emit::*;

use crate::isa::scry::abi::ScryMachineDeps;

use core::fmt::Display;

pub use crate::isa::scry::lower::isle::generated_code::MInst;

//=============================================================================
// Instructions (top level): definition

impl MachInst for MInst {
    type LabelUse = LabelUse;
    type ABIMachineSpec = ScryMachineDeps;

    // https://github.com/riscv/riscv-isa-manual/issues/850
    // all zero will cause invalid opcode.
    const TRAP_OPCODE: &'static [u8] = &[0; 4];

    fn gen_dummy_use(reg: Reg) -> Self {
        unimplemented!()
    }

    fn canonical_type_for_rc(rc: RegClass) -> Type {
        unimplemented!()
    }

    fn is_safepoint(&self) -> bool {
        unimplemented!()
    }

    fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        unimplemented!()
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        unimplemented!()
    }

    fn is_included_in_clobbers(&self) -> bool {
        unimplemented!()
    }

    fn is_trap(&self) -> bool {
        unimplemented!()
    }

    fn is_args(&self) -> bool {
        unimplemented!()
    }

    fn call_type(&self) -> CallType {
        unimplemented!()
    }

    fn is_term(&self) -> MachTerminator {
        unimplemented!()
    }

    fn is_mem_access(&self) -> bool {
        unimplemented!()
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_nop(preferred_size: usize) -> MInst {
        unimplemented!()
    }

    fn gen_nop_units() -> Vec<Vec<u8>>{
        unimplemented!()
    }

    fn rc_for_type(ty: Type) -> CodegenResult<(&'static [RegClass], &'static [Type])> {
        unimplemented!()
    }

    fn gen_jump(target: MachLabel) -> MInst{
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

impl MInst {
    fn print_with_state(&self, _state: &mut EmitState) -> String {
        unimplemented!()
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

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
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
        buffer: &mut [u8],
        veneer_offset: CodeOffset,
    ) -> (CodeOffset, LabelUse) {
        unimplemented!()
    }

    fn from_reloc(reloc: Reloc, addend: Addend) -> Option<LabelUse> {
        unimplemented!()
    }
}
