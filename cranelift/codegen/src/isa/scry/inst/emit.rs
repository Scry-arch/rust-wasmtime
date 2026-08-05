//! Scry ISA: binary code emission.

use crate::ir::{self};
use crate::isa::scry::inst::*;
use crate::isa::scry::lower::isle::generated_code::MInst;
use cranelift_control::ControlPlane;
use scry_isa::{AluVariant, Bits, CallVariant, Instruction};

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
    fn new(abi: &Callee<crate::isa::scry::abi::ScryMachineDeps>, ctrl_plane: ControlPlane) -> Self {
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

    fn on_new_block(&mut self) {}

    fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }
}

impl MachInstEmit for MInst {
    type State = EmitState;
    type Info = EmitInfo;

    fn emit(&self, sink: &mut MachBuffer<MInst>, _emit_info: &Self::Info, _state: &mut EmitState) {
        use MInst::*;
        let insts = match self {
            Rets { .. }
            | ImmJump { .. }
            | IntCmp { .. }
            | BinaryAlu { .. }
            | Resize { .. }
            | Echo { .. } => {
                unreachable!("Pseudo-instruction was not eliminated: {:?}", self)
            }
            Args { .. } | CallArgs { .. } | JumpTrigger { .. } => vec![],
            Nop | Discard { .. } => vec![Instruction::NoOp],
            Ret { trig } => {
                vec![Instruction::Call(
                    CallVariant::Ret,
                    Bits::try_from(*trig as i32).unwrap(),
                )]
            }
            Alu1 { var, out, .. } => {
                vec![Instruction::Alu(*var, Bits::try_from(*out as i32).unwrap())]
            }
            Alu2 {
                var, out_var, outs, ..
            } => {
                assert_eq!(outs.len(), 1);
                vec![Instruction::Alu2(
                    *var,
                    *out_var,
                    Bits::try_from(outs[0] as i32).unwrap(),
                )]
            }
            UnaryAlu { .. } => {
                vec![Instruction::Alu(
                    AluVariant::Equal,
                    Bits::try_from(0i32).unwrap(),
                )] // Logical negation just uses "x == 0", where 0 is implicit
            }
            Const { ty, imm, .. } => {
                let bits = (imm.bits() & 0b1111_1111) as u8; // Extract only relevant bits, we do not support value > 8 bits yet.
                vec![Instruction::Constant(
                    ty.get_known()
                        .expect("Missing a well-defined type for constant")
                        .try_into()
                        .unwrap(),
                    Bits::try_from(bits as i32).unwrap(),
                )]
            }
            EchoLong { out, .. } => {
                vec![Instruction::EchoLong(Bits::try_from(*out as i32).unwrap())]
            }
            EchoSplit { out1, out2, .. } => vec![if out1 == out2 {
                // Using a long echo ensures operand order is not changed when targetting the same instruction
                // unlike Instruction::Echo
                Instruction::EchoLong(Bits::try_from(*out1 as i32).unwrap())
            } else {
                Instruction::Echo(
                    false,
                    Bits::try_from(*out1 as i32).unwrap(),
                    Bits::try_from(*out2 as i32).unwrap(),
                )
            }],
            EchoChain { out1, out2, .. } => {
                vec![Instruction::Echo(
                    true,
                    Bits::try_from(*out1 as i32).unwrap(),
                    Bits::try_from(*out2 as i32).unwrap(),
                )]
            }
            Reorder { out, .. } => {
                // Can use splitting echo with the same target to reorder
                vec![Instruction::Echo(
                    false,
                    Bits::try_from(*out as i32).unwrap(),
                    Bits::try_from(*out as i32).unwrap(),
                )]
            }
            Duplicate { out1, out2, .. } => vec![Instruction::Duplicate(
                false,
                Bits::try_from(*out1 as i32).unwrap(),
                Bits::try_from(*out2 as i32).unwrap(),
            )],
            Store { .. } => vec![Instruction::Store],
            Load { ty, out, .. } => vec![Instruction::Load(
                ty.get_known().unwrap().try_into().unwrap(),
                Bits::try_from(*out as i32).unwrap(),
            )],
            Cast { out, ty, .. } => vec![Instruction::Cast(
                ty.get_known()
                    .unwrap_or(scry_isa::Type::Uint(ty.size_pow2()))
                    .try_into()
                    .unwrap(),
                Bits::try_from(*out as i32).unwrap(),
            )],
            Call { trig, .. } => {
                vec![Instruction::Call(
                    CallVariant::Call,
                    Bits::try_from(*trig as i32).unwrap(),
                )]
            }
            JumpIssue { dst, .. } | BranchIssue { dst, .. } => {
                sink.use_label_at_offset(sink.cur_offset(), *dst, LabelUse::JmpLoc7);
                vec![Instruction::Jump(
                    0.try_into().unwrap(),
                    0.try_into().unwrap(),
                )]
            }
        };
        for inst in insts {
            sink.put2(inst.encode());
        }
    }

    fn pretty_print_inst(&self, state: &mut Self::State) -> String {
        self.print_with_state(state)
    }
}
