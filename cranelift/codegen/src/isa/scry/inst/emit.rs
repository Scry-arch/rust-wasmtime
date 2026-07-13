//! Riscv64 ISA: binary code emission.

use crate::ir::{self};
use crate::isa::scry::IsaType;
use crate::isa::scry::inst::*;
use crate::isa::scry::lower::isle::generated_code::MInst;
use cranelift_control::ControlPlane;
use scry_isa::{Bits, CallVariant, Instruction};

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
        let instr = match self {
            Rets { .. } | ImmJump { .. } | IntCmp { .. } | IntAdd { .. } | UnsignedExt { .. } => {
                unreachable!("Pseudo-instruction was not eliminated: {:?}", self)
            }
            Args { .. } | CallArgs { .. } | JumpTrigger { .. } => return,
            Nop => Instruction::NoOp,
            Ret { trig } => {
                Instruction::Call(CallVariant::Ret, Bits::try_from(*trig as i32).unwrap())
            }
            Alu1 { var, out, .. } => Instruction::Alu(*var, Bits::try_from(*out as i32).unwrap()),
            Const { imm, .. } => Instruction::Constant(
                Bits::try_from(0).unwrap(),
                Bits::try_from(imm.bits() as i32).unwrap(),
            ),
            Echo { rds, outs, .. } => {
                assert_eq!(
                    rds.len(),
                    outs.len(),
                    "Registers do not have correct number of outputs: {:?} != {:?}",
                    rds,
                    outs
                );

                if outs.iter().all(|o| *o == outs[0]) {
                    // All outputs go to the same destination, use long echo
                    Instruction::EchoLong(Bits::try_from(outs[0] as i32).unwrap())
                } else if outs.len() == 2 {
                    // Two outputs going different destinations, use splitting echo
                    Instruction::Echo(
                        false,
                        Bits::try_from(outs[0] as i32).unwrap(),
                        Bits::try_from(outs[1] as i32).unwrap(),
                    )
                } else {
                    unimplemented!()
                }
            }
            Reorder { out, .. } => {
                // Can use splitting echo with the same target to reorder
                Instruction::Echo(
                    false,
                    Bits::try_from(*out as i32).unwrap(),
                    Bits::try_from(*out as i32).unwrap(),
                )
            }
            Duplicate { out1, out2, .. } => Instruction::Duplicate(
                false,
                Bits::try_from(*out1 as i32).unwrap(),
                Bits::try_from(*out2 as i32).unwrap(),
            ),
            Store { .. } => Instruction::Store,
            Load { out, .. } => Instruction::Load(
                scry_isa::Type::Int(2).try_into().unwrap(),
                Bits::try_from(*out as i32).unwrap(),
            ),
            Cast { out, ty, .. } => {
                let ty = match ty {
                    IsaType::Known(t) => t,
                    _ => unimplemented!("ty: {:?}", ty),
                };
                Instruction::Cast(
                    ty.clone().try_into().unwrap(),
                    Bits::try_from(*out as i32).unwrap(),
                )
            }
            Call { trig, .. } => {
                Instruction::Call(CallVariant::Call, Bits::try_from(*trig as i32).unwrap())
            }
            JumpIssue { dst, .. } | BranchIssue { dst, .. } => {
                // sink.use_label_at_offset(sink.cur_offset(), *dst, LabelUse::JmpTrig6);
                sink.use_label_at_offset(sink.cur_offset(), *dst, LabelUse::JmpLoc7);
                Instruction::Jump(0.try_into().unwrap(), 0.try_into().unwrap())
            }
        };
        sink.put2(instr.encode());
    }

    fn pretty_print_inst(&self, state: &mut Self::State) -> String {
        self.print_with_state(state)
    }
}
