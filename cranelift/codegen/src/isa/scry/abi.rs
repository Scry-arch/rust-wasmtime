//! Implementation of the Scry ABI.

use crate::ir;
use crate::ir::types::*;

use crate::isa;

use crate::machinst::*;

use crate::CodegenResult;
use crate::ir::{ArgumentExtension, ArgumentPurpose, Signature};
use crate::isa::scry::settings::Flags as ScryFlags;
use crate::settings;
use alloc::vec::Vec;
use regalloc2::{MachineEnv, PReg, PRegSet};

use crate::isa::scry::inst::QUEUE_CAPACITY;
use crate::isa::scry::lower::isle::generated_code::MInst;
use crate::isa::scry::type_to_isatype;
use smallvec::{SmallVec, smallvec};

/// Byte offsets of the stack-passed values (the ones beyond the first
/// [`QUEUE_CAPACITY`], see the ABI's calling convention) within their area,
/// plus the area's total size. Each value is placed directly after the
/// previous one, at the next offset aligned to its scale.
pub(crate) fn stack_values_layout(values: &[ir::AbiParam]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::new();
    let mut size = 0u32;
    for p in values.iter().skip(QUEUE_CAPACITY) {
        let scale = p.value_type.bytes();
        size = size.next_multiple_of(scale);
        offsets.push(size);
        size += scale;
    }
    (offsets, size)
}

/// The frame offset of a function's stack-argument area: the return area
/// (which sits at offset 0 of the private frame) rounded up to the 16-byte
/// frame alignment, which keeps every argument's natural alignment intact.
pub(crate) fn arg_area_base(sig: &Signature) -> u32 {
    stack_values_layout(&sig.returns)
        .1
        .next_multiple_of(16)
}

/// The total size of the caller-reserved stack area for calling a function of
/// the given signature: the return area at offset 0, then the argument area
/// at [`arg_area_base`].
pub(crate) fn stack_area_size(sig: &Signature) -> u32 {
    arg_area_base(sig) + stack_values_layout(&sig.params).1
}

/// The frame offset at which a function's own stack slots start: its incoming
/// caller-reserved area (return area + argument area, at offset 0) rounded up
/// to the 16-byte frame alignment, which keeps every slot's natural alignment
/// intact.
pub(crate) fn stack_locals_base(sig: &Signature) -> u32 {
    stack_area_size(sig).next_multiple_of(16)
}

/// Scry-specific ABI behavior. This struct just serves as an implementation
/// point for the trait; it is never actually instantiated.
pub struct ScryMachineDeps;

impl IsaFlags for ScryFlags {}

impl ABIMachineSpec for ScryMachineDeps {
    type I = MInst;
    type F = ScryFlags;

    /// This is the limit for the size of argument and return-value areas on the
    /// stack. We place a reasonable limit here to avoid integer overflow issues
    /// with 32-bit arithmetic: for now, 128 MB.
    const STACK_ARG_RET_SIZE_LIMIT: u32 = 128 * 1024 * 1024;

    fn word_bits() -> u32 {
        32
    }

    fn stack_align(_call_conv: isa::CallConv) -> u32 {
        16
    }

    fn compute_arg_locs(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        params: &[ir::AbiParam],
        args_or_rets: ArgsOrRets,
        _add_ret_area_ptr: bool,
        mut args: ArgsAccumulator,
    ) -> CodegenResult<(u32, Option<usize>)> {
        // The first QUEUE_CAPACITY arguments are passed as call operands,
        // delivered to the callee's first instruction; any further arguments
        // are passed on the stack per the ABI's calling convention.
        //
        // For the operand arguments, we use the pregs to encode the parameter's
        // position in the parameter list. This is needed after instruction
        // selection, where MInst::Args only gets the parameters that are
        // actually used. The pregs and their positions are therefore used to
        // get the unused parameters to, e.g., discard them correctly.
        //
        // Return values beyond the QUEUE_CAPACITY-th are stack-passed too, but
        // are deliberately given Reg slots here anyway: the common machinery
        // would route Stack return slots through a hidden return-area pointer,
        // which the ABI does not use (the area's location is implied by the
        // frame handoff). `lower_stack_args` splits the excess return values
        // off the `Rets`/`CallArgs` pseudo-instructions into stack accesses.
        let (stack_offsets, stack_size) = stack_values_layout(params);
        for (i, p) in params.iter().enumerate() {
            assert_eq!(p.purpose, ArgumentPurpose::Normal);

            let slot = if i < QUEUE_CAPACITY || args_or_rets == ArgsOrRets::Rets {
                ABIArgSlot::Reg {
                    reg: Reg::from_real_reg(PReg::new(i, RegClass::Int))
                        .to_real_reg()
                        .unwrap(),
                    ty: p.value_type,
                    extension: p.extension,
                }
            } else {
                ABIArgSlot::Stack {
                    offset: stack_offsets[i - QUEUE_CAPACITY] as i64,
                    ty: p.value_type,
                    extension: p.extension,
                }
            };

            args.push(ABIArg::Slots {
                slots: SmallVec::<[ABIArgSlot; 1]>::from_vec(vec![slot]),
                purpose: p.purpose,
            });
        }

        // The return area is managed entirely by `lower_stack_args` (no
        // common-machinery return-area pointer), so no ret space is reported.
        let stack_size = if args_or_rets == ArgsOrRets::Rets {
            0
        } else {
            stack_size
        };

        Ok((stack_size, None))
    }

    fn gen_load_stack(mem: StackAMode, into_reg: Writable<Reg>, ty: Type) -> MInst {
        // Only used for the callee side of stack-passed arguments: the
        // incoming argument area sits at the base of the private frame
        // (offset 0). 
        match mem {
            StackAMode::IncomingArg(offset, _) => {
                let scale = ty.bytes() as i64;
                assert_eq!(
                    offset % scale,
                    0,
                    "stack argument offset not aligned to its scale"
                );
                let idx = offset / scale;
                assert!(
                    idx <= 31,
                    "stack argument at offset {offset} is beyond the stack-load index range"
                );
                MInst::LoadStack {
                    ty: type_to_isatype(ty),
                    rd: into_reg,
                    idx: idx as u16,
                }
            }
            mem => unimplemented!("gen_load_stack: {mem:?}"),
        }
    }

    fn gen_store_stack(mem: StackAMode, from_reg: Reg, ty: Type) -> MInst {
        // Only used for the caller side of stack-passed arguments. The
        // outgoing argument area lives in the caller's shared frame, whose
        // frame offset (the caller's own private frame size) is not known
        // during lowering; `lower_stack_args` later resolves this
        // pseudo-instruction into a concrete stack access.
        match mem {
            StackAMode::OutgoingArg(offset) => MInst::StoreStackArg {
                rs: from_reg,
                offset: u16::try_from(offset).expect("outgoing stack argument offset too large"),
                scale_pow2: ty.bytes().ilog2() as u16,
            },
            mem => unimplemented!("gen_store_stack: {mem:?}"),
        }
    }

    fn gen_move(_to_reg: Writable<Reg>, _from_reg: Reg, _ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_extend(
        _to_reg: Writable<Reg>,
        _from_reg: Reg,
        _signed: bool,
        _from_bits: u8,
        _to_bits: u8,
    ) -> MInst {
        unreachable!("Scry should not need any function argument extension")
    }

    fn get_ext_mode(
        _call_conv: isa::CallConv,
        _specified: ir::ArgumentExtension,
    ) -> ir::ArgumentExtension {
        // Scry does not need to extend any function arguments
        ArgumentExtension::None
    }

    fn gen_args(args: Vec<ArgPair>) -> MInst {
        MInst::Args { args }
    }

    fn gen_rets(rets: Vec<RetPair>) -> MInst {
        MInst::Rets { rets }
    }

    fn get_stacklimit_reg(_call_conv: isa::CallConv) -> Reg {
        unimplemented!()
    }

    fn gen_add_imm(
        _call_conv: isa::CallConv,
        _into_reg: Writable<Reg>,
        _from_reg: Reg,
        _imm: u32,
    ) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_stack_lower_bound_trap(_limit_reg: Reg) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_get_stack_addr(_mem: StackAMode, _into_reg: Writable<Reg>) -> MInst {
        unimplemented!()
    }

    fn gen_load_base_offset(
        _into_reg: Writable<Reg>,
        _base: Reg,
        _offset: i32,
        _ty: Type,
    ) -> MInst {
        unimplemented!()
    }

    fn gen_store_base_offset(_base: Reg, _offset: i32, _from_reg: Reg, _ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_sp_reg_adjust(_amount: i32) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_prologue_frame_setup(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        smallvec![]
    }

    fn gen_epilogue_frame_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        smallvec![]
    }

    fn gen_return(
        _call_conv: isa::CallConv,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        smallvec![]
    }

    fn gen_probestack(_insts: &mut SmallInstVec<Self::I>, _frame_size: u32) {
        unimplemented!()
    }

    fn gen_clobber_save(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _frame_layout: &FrameLayout,
    ) -> SmallVec<[MInst; 16]> {
        smallvec![]
    }

    fn gen_clobber_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _frame_layout: &FrameLayout,
    ) -> SmallVec<[MInst; 16]> {
        smallvec![]
    }

    fn gen_memcpy<F: FnMut(Type) -> Writable<Reg>>(
        _call_conv: isa::CallConv,
        _dst: Reg,
        _src: Reg,
        _size: usize,
        _alloc_tmp: F,
    ) -> SmallVec<[Self::I; 8]> {
        unimplemented!()
    }

    fn get_number_of_spillslots_for_value(
        _rc: RegClass,
        _target_vector_bytes: u32,
        _isa_flags: &ScryFlags,
    ) -> u32 {
        unimplemented!()
    }

    fn get_machine_env(_flags: &settings::Flags, _call_conv: isa::CallConv) -> &MachineEnv {
        static MACHINE_ENV: MachineEnv = MachineEnv {
            preferred_regs_by_class: [PRegSet::empty(), PRegSet::empty(), PRegSet::empty()],
            non_preferred_regs_by_class: [PRegSet::empty(), PRegSet::empty(), PRegSet::empty()],
            scratch_by_class: [None, None, None],
            fixed_stack_slots: vec![],
        };
        &MACHINE_ENV
    }

    fn get_regs_clobbered_by_call(
        _call_conv_of_callee: isa::CallConv,
        _is_exception: bool,
    ) -> PRegSet {
        unimplemented!()
    }

    fn compute_frame_layout(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _sig: &Signature,
        _regs: &[Writable<RealReg>],
        _function_calls: FunctionCalls,
        _incoming_args_size: u32,
        _tail_args_size: u32,
        _stackslots_size: u32,
        _fixed_frame_storage_size: u32,
        _outgoing_args_size: u32,
    ) -> FrameLayout {
        // TODO
        FrameLayout::default()
    }

    fn gen_inline_probestack(
        _insts: &mut SmallInstVec<Self::I>,
        _call_conv: isa::CallConv,
        _frame_size: u32,
        _guard_size: u32,
    ) {
        unimplemented!()
    }

    fn retval_temp_reg(_call_conv_of_callee: isa::CallConv) -> Writable<Reg> {
        unimplemented!()
    }

    fn exception_payload_regs(_call_conv: isa::CallConv) -> &'static [Reg] {
        unimplemented!()
    }
}
