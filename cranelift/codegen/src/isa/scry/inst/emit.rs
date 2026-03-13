//! Riscv64 ISA: binary code emission.

use scry_isa::{AluVariant, Bits, CallVariant, Instruction};
use crate::ir::{self};
use crate::isa::scry::inst::*;
use crate::isa::scry::lower::isle::generated_code::MInst;
use cranelift_control::ControlPlane;

pub struct EmitInfo {
    #[expect(dead_code, reason = "may want to be used in the future")]
    shared_flag: settings::Flags,
    #[allow(unused)]
    isa_flags: super::super::scry_settings::Flags,
}

impl EmitInfo {
    pub(crate) fn new(
        shared_flag: settings::Flags,
        isa_flags: super::super::scry_settings::Flags,
    ) -> Self {
        Self {
            shared_flag,
            isa_flags,
        }
    }
}

/// State carried between emissions of a sequence of instructions.
#[derive(Default, Clone, Debug)]
pub struct EmitState {
    /// The user stack map for the upcoming instruction, as provided to
    /// `pre_safepoint()`.
    user_stack_map: Option<ir::UserStackMap>,

    /// Only used during fuzz-testing. Otherwise, it is a zero-sized struct and
    /// optimized away at compiletime. See [cranelift_control].
    ctrl_plane: ControlPlane,

    frame_layout: FrameLayout,
}

impl MachInstEmitState<MInst> for EmitState {
    fn new(
        abi: &Callee<crate::isa::scry::abi::ScryMachineDeps>,
        ctrl_plane: ControlPlane,
    ) -> Self {
        EmitState {
            user_stack_map: None,
            ctrl_plane,
            frame_layout: abi.frame_layout().clone(),
        }
    }

    fn pre_safepoint(&mut self, user_stack_map: Option<ir::UserStackMap>) {
        self.user_stack_map = user_stack_map;
    }

    fn ctrl_plane_mut(&mut self) -> &mut ControlPlane {
        &mut self.ctrl_plane
    }

    fn take_ctrl_plane(self) -> ControlPlane {
        self.ctrl_plane
    }

    fn on_new_block(&mut self) {
    }

    fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }
}

impl MachInstEmit for MInst {
    type State = EmitState;
    type Info = EmitInfo;

    fn emit(&self, sink: &mut MachBuffer<MInst>, _emit_info: &Self::Info, _state: &mut EmitState) {
        dbg!(self);
        
        match self {
            MInst::Nop => {
                sink.put2(Instruction::NoOp.encode())
            }
            MInst::Args { .. } => {}
            MInst::Rets { .. } => {}
            MInst::Ret => {
                sink.put2(Instruction::Call(CallVariant::Ret, Bits::try_from(0).unwrap()).encode())
            }
            MInst::Add { .. } => {
                sink.put2(Instruction::Alu(AluVariant::Add, Bits::try_from(0).unwrap()).encode())
            }
            MInst::Const { rd, imm } => {
                sink.put2(Instruction::Constant(Bits::try_from(0).unwrap(), Bits::try_from(imm.bits() as i32).unwrap()).encode())
                
            }
        }
    }

    fn pretty_print_inst(&self, state: &mut Self::State) -> String {
        self.print_with_state(state)
    }
}