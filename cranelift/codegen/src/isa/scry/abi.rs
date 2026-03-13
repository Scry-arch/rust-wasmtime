//! Implementation of a standard Riscv64 ABI.

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

use smallvec::{smallvec, SmallVec};
use crate::isa::scry::lower::isle::generated_code::MInst;

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
        _args_or_rets: ArgsOrRets,
        _add_ret_area_ptr: bool,
        mut args: ArgsAccumulator,
    ) -> CodegenResult<(u32, Option<usize>)> {
        for p in params {
            assert_eq!(p.purpose, ArgumentPurpose::Normal);
            assert_eq!(p.extension, ArgumentExtension::None);
            assert_eq!(p.value_type, Type::int(32).unwrap());
            
            args.push(ABIArg::Slots {
                slots: SmallVec::<[ABIArgSlot; 1]>::from_vec(vec![ABIArgSlot::Reg {
                    reg: Reg::from_real_reg(PReg::new(0, RegClass::Int)).to_real_reg().unwrap(),
                    ty: Default::default(),
                    extension: ArgumentExtension::None,
                }]),
                purpose: p.purpose,
            } );
        }
        
        Ok((0,None))
    }

    fn gen_load_stack(_mem: StackAMode, _into_reg: Writable<Reg>, _ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_store_stack(_mem: StackAMode, _from_reg: Reg, _ty: Type) -> MInst {
        unimplemented!()
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
        unimplemented!()
    }

    fn get_ext_mode(
        _call_conv: isa::CallConv,
        specified: ir::ArgumentExtension,
    ) -> ir::ArgumentExtension {
        specified
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

    fn gen_load_base_offset(_into_reg: Writable<Reg>, _base: Reg, _offset: i32, _ty: Type) -> MInst {
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
        let insts = SmallVec::new();
        insts
    }
    
    fn gen_epilogue_frame_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_return(
        _call_conv: isa::CallConv,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst>{
        smallvec![MInst::Ret {}]
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
        unimplemented!()
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
            preferred_regs_by_class: [PRegSet::empty(),PRegSet::empty(),PRegSet::empty(),],
            non_preferred_regs_by_class: [PRegSet::empty(),PRegSet::empty(),PRegSet::empty(),],
            scratch_by_class: [None,None,None],
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
