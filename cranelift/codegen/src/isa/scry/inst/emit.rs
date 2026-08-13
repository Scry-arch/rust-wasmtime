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

/// Encodes an output reference.
///
/// Panics of the reference is out of bounds
fn out_ref<const N: u32>(out: u16) -> Bits<N, false> {
    Bits::try_from(out as i32)
        .expect(format!("Output reference out of bounds: was {out}, limit: {N} bits").as_str())
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
                vec![Instruction::Alu(*var, out_ref(*out))]
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
                let t = ty
                    .get_known()
                    .expect("Missing a well-defined type for constant");

                // The `const` instruction provides the most significant explicit
                // byte (sign/zero extension fills everything above it); each
                // subsequent `grow` shifts the value left by a byte and inserts the
                // next one.
                let bytes = MInst::const_emit_bytes(t, imm.bits());
                let (last, rest) = bytes.split_last().unwrap();
                let mut insts = vec![Instruction::Constant(
                    t.try_into().unwrap(),
                    Bits::try_from(*last as i32).unwrap(),
                )];
                insts.extend(
                    rest.iter()
                        .rev()
                        .map(|b| Instruction::Grow(Bits::try_from(*b as i32).unwrap())),
                );
                insts
            }
            EchoLong { out, .. } => {
                vec![Instruction::EchoLong(out_ref(*out))]
            }
            EchoSplit { out1, out2, .. } => vec![if out1 == out2 {
                // Using a long echo ensures operand order is not changed when targetting the same instruction
                // unlike Instruction::Echo
                Instruction::EchoLong(out_ref(*out1))
            } else {
                Instruction::Echo(false, out_ref(*out1), out_ref(*out2))
            }],
            EchoChain { out1, out2, .. } => {
                vec![Instruction::Echo(true, out_ref(*out1), out_ref(*out2))]
            }
            Reorder { out, .. } => {
                // Can use splitting echo with the same target to reorder
                vec![Instruction::Echo(false, out_ref(*out), out_ref(*out))]
            }
            Duplicate { out1, out2, .. } => vec![Instruction::Duplicate(
                false,
                out_ref(*out1),
                out_ref(*out2),
            )],
            Store { .. } => vec![Instruction::Store],
            Load { ty, out, .. } => vec![Instruction::Load(
                ty.get_known().unwrap().try_into().unwrap(),
                out_ref(*out),
            )],
            StoreStack { idx, .. } => vec![Instruction::StoreStack(
                Bits::try_from(*idx as i32).expect("Stack index out of bounds"),
            )],
            LoadStack { ty, idx, .. } => vec![Instruction::LoadStack(
                ty.get_known().unwrap().try_into().unwrap(),
                Bits::try_from(*idx as i32).expect("Stack index out of bounds"),
            )],
            SAddr {
                scale_pow2, idx, ..
            } => vec![Instruction::StackAddr(
                Bits::try_from(*scale_pow2 as i32).expect("Scale out of bounds"),
                Bits::try_from(*idx as i32).expect("Stack index out of bounds"),
            )],
            StackAdjust {
                reserve,
                amount_pow2,
            } => {
                // Reserves target the private frame (which claims fresh memory);
                // frees target the shared frame, which releases the memory that
                // freeing the private frame alone would only merge into the
                // shared frame.
                vec![Instruction::StackRes(
                    *reserve,
                    Bits::try_from(*amount_pow2 as i32).expect("Stack amount out of bounds"),
                    *reserve,
                )]
            }
            Cast { out, ty, .. } => vec![Instruction::Cast(
                ty.get_known()
                    .unwrap_or(scry_isa::Type::Uint(ty.size_pow2()))
                    .try_into()
                    .unwrap(),
                out_ref(*out),
            )],
            Call { trig, .. } => {
                vec![Instruction::Call(
                    CallVariant::Call,
                    Bits::try_from(*trig as i32).unwrap(),
                )]
            }
            JumpIssue { dst, trig, .. } | BranchIssue { dst, trig, .. } => {
                sink.use_label_at_offset(sink.cur_offset(), *dst, LabelUse::JmpLoc7);
                vec![Instruction::Jump(
                    0.try_into().unwrap(),
                    Bits::try_from(*trig as i32)
                        .expect(format!("Trigger offset out of bounds: {trig}").as_str()),
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
