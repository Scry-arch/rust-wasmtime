//! Implementation of a standard Riscv64 ABI.

use crate::ir;
use crate::ir::types::*;

use crate::isa;

use crate::isa::scry::inst::*;
use crate::machinst::*;

use crate::CodegenResult;
use crate::ir::Signature;
use crate::isa::scry::settings::Flags as ScryFlags;
use crate::settings;
use alloc::vec::Vec;
use regalloc2::{MachineEnv, PRegSet};

use smallvec::SmallVec;
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
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        params: &[ir::AbiParam],
        args_or_rets: ArgsOrRets,
        add_ret_area_ptr: bool,
        args: ArgsAccumulator,
    ) -> CodegenResult<(u32, Option<usize>)> {
        unimplemented!()
    }

    fn gen_load_stack(mem: StackAMode, into_reg: Writable<Reg>, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_store_stack(mem: StackAMode, from_reg: Reg, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_extend(
        to_reg: Writable<Reg>,
        from_reg: Reg,
        signed: bool,
        from_bits: u8,
        to_bits: u8,
    ) -> MInst {
        unimplemented!()
    }

    fn get_ext_mode(
        _call_conv: isa::CallConv,
        specified: ir::ArgumentExtension,
    ) -> ir::ArgumentExtension {
        unimplemented!()
    }

    fn gen_args(args: Vec<ArgPair>) -> MInst {
        unimplemented!()
    }

    fn gen_rets(rets: Vec<RetPair>) -> MInst {
        unimplemented!()
    }

    fn get_stacklimit_reg(_call_conv: isa::CallConv) -> Reg {
        unimplemented!()
    }

    fn gen_add_imm(
        _call_conv: isa::CallConv,
        into_reg: Writable<Reg>,
        from_reg: Reg,
        imm: u32,
    ) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_stack_lower_bound_trap(limit_reg: Reg) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_get_stack_addr(mem: StackAMode, into_reg: Writable<Reg>) -> MInst {
        unimplemented!()
    }

    fn gen_load_base_offset(into_reg: Writable<Reg>, base: Reg, offset: i32, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_store_base_offset(base: Reg, offset: i32, from_reg: Reg, ty: Type) -> MInst {
        unimplemented!()
    }

    fn gen_sp_reg_adjust(amount: i32) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_prologue_frame_setup(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        _isa_flags: &ScryFlags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        unimplemented!()
    }
    
    fn gen_epilogue_frame_restore(
        call_conv: isa::CallConv,
        _flags: &settings::Flags,
        _isa_flags: &ScryFlags,
        frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst> {
        unimplemented!()
    }

    fn gen_return(
        _call_conv: isa::CallConv,
        _isa_flags: &ScryFlags,
        _frame_layout: &FrameLayout,
    ) -> SmallInstVec<MInst>{
        unimplemented!()
    }

    fn gen_probestack(insts: &mut SmallInstVec<Self::I>, frame_size: u32) {
        unimplemented!()
    }

    fn gen_clobber_save(
        _call_conv: isa::CallConv,
        flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[MInst; 16]> {
        unimplemented!()
    }

    fn gen_clobber_restore(
        _call_conv: isa::CallConv,
        _flags: &settings::Flags,
        frame_layout: &FrameLayout,
    ) -> SmallVec<[MInst; 16]> {
        unimplemented!()
    }

    fn gen_memcpy<F: FnMut(Type) -> Writable<Reg>>(
        call_conv: isa::CallConv,
        dst: Reg,
        src: Reg,
        size: usize,
        mut alloc_tmp: F,
    ) -> SmallVec<[Self::I; 8]> {
        unimplemented!()
    }

    fn get_number_of_spillslots_for_value(
        rc: RegClass,
        _target_vector_bytes: u32,
        isa_flags: &ScryFlags,
    ) -> u32 {
        unimplemented!()
    }

    fn get_machine_env(_flags: &settings::Flags, _call_conv: isa::CallConv) -> &MachineEnv {
        unimplemented!()
    }

    fn get_regs_clobbered_by_call(
        call_conv_of_callee: isa::CallConv,
        is_exception: bool,
    ) -> PRegSet {
        unimplemented!()
    }

    fn compute_frame_layout(
        call_conv: isa::CallConv,
        flags: &settings::Flags,
        _sig: &Signature,
        regs: &[Writable<RealReg>],
        function_calls: FunctionCalls,
        incoming_args_size: u32,
        tail_args_size: u32,
        stackslots_size: u32,
        fixed_frame_storage_size: u32,
        outgoing_args_size: u32,
    ) -> FrameLayout {
        unimplemented!()
    }

    fn gen_inline_probestack(
        insts: &mut SmallInstVec<Self::I>,
        _call_conv: isa::CallConv,
        frame_size: u32,
        guard_size: u32,
    ) {
        unimplemented!()
    }

    fn retval_temp_reg(_call_conv_of_callee: isa::CallConv) -> Writable<Reg> {
        unimplemented!()
    }

    fn exception_payload_regs(call_conv: isa::CallConv) -> &'static [Reg] {
        unimplemented!()
    }
}
