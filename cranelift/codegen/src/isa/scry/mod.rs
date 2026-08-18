//! Scry Instruction Set Architecture.

use crate::MachLabel;
use crate::dominator_tree::DominatorTree;
use crate::ir::{AbiParam, ArgumentExtension, Signature};
use crate::ir::{Function, Type};
use crate::isa::scry::inst::{
    BinaryAluOp, DoubleAluOp, EmitInfo, MInst, QUEUE_CAPACITY, ResizeVariant, UnaryAluOp,
    delivery_group,
};
use crate::isa::scry::settings as scry_settings;
use crate::isa::unwind::systemv;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, TargetIsa,
};
use crate::machinst::isle::{Writable, WritableReg};
use crate::machinst::{
    ArgPair, BlockLoweringOrder, Callee, CompiledCode, CompiledCodeStencil, MachInst,
    MachTextSectionBuilder, Reg, SigSet, TextSectionBuilder, VCode, VCodeBuildDirection,
    VCodeBuilder, VRegAllocator,
};
use crate::opts::IntCC;
use crate::result::CodegenResult;
use crate::settings::{self as shared_settings, Flags};
use crate::timing;
use crate::trace;
use crate::{VCodeConstants, ir};
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use core::fmt::{Debug, Formatter};
use cranelift_control::ControlPlane;
use cranelift_entity::EntityRef;
use graphene::core::GraphMut;
use graphene::core::property::Rooted;
use graphene::core::{Graph, MaybeOwned};
use regalloc2::{Block, Function as RegFunc};
use scry_isa::{Alu2OutputVariant, Alu2Variant, AluVariant};
use std::cmp::max;
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::once;
use target_lexicon::{Architecture, Triple};
use vcode_cfg::*;

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;
mod vcode_cfg;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum IsaType {
    /// Something went wrong in type resolution and the value cannot be determined to have a valid type.
    Invalid,

    /// Either signed or unsigned integer of the given power of 2 size (bytes)
    Integer(u8),

    /// A specific known type
    Known(scry_isa::Type),
}

impl IsaType {
    /// Power of 2 size of the type in bytes
    fn size_pow2(&self) -> u8 {
        match self {
            IsaType::Integer(s) => *s,
            IsaType::Known(t) => t.size_pow2(),
            _ => unimplemented!(),
        }
    }

    /// Whether the type is an integer (signed, unsigned, or undetermined)
    fn is_int(&self) -> bool {
        match self {
            IsaType::Integer(_) => true,
            IsaType::Known(t) => t.is_signed_int() || t.is_unsigned_int(),
            _ => false,
        }
    }

    fn is_known(&self) -> bool {
        matches!(self, IsaType::Known(_))
    }

    fn is_signed_int(&self) -> bool {
        matches!(self, IsaType::Known(t) if t.is_signed_int())
    }

    fn is_same_signedness(&self, other: &IsaType) -> bool {
        match (self, other) {
            (IsaType::Known(a), IsaType::Known(b)) => {
                (a.is_unsigned_int() && b.is_unsigned_int())
                    || (a.is_signed_int() && b.is_signed_int())
            }
            _ => false,
        }
    }

    fn refine(self, t: IsaType) -> Option<IsaType> {
        use IsaType::*;

        if t == Invalid {
            Some(self)
        } else if self == Invalid {
            Some(t)
        } else if self.size_pow2() == t.size_pow2() {
            match (self, t) {
                (Integer(_), _) if t.is_int() => Some(t),
                (_, Integer(_)) if self.is_int() => Some(self),
                (Known(scry_isa::Type::Int(_)), Known(scry_isa::Type::Int(_))) => Some(self),
                (Known(scry_isa::Type::Uint(_)), Known(scry_isa::Type::Uint(_))) => Some(self),
                _ => None,
            }
        } else {
            None
        }
    }

    fn new_known_int(size_pow2: u8, signed: bool) -> IsaType {
        if signed {
            IsaType::Known(scry_isa::Type::Int(size_pow2))
        } else {
            IsaType::Known(scry_isa::Type::Uint(size_pow2))
        }
    }

    fn get_known(&self) -> Option<scry_isa::Type> {
        match self {
            IsaType::Known(t) => Some(*t),
            _ => None,
        }
    }
}

/// Maps registers to IsaTypes
///
/// Takes a closure that returns the CLIF type of a register, which is used to assign default types
/// to registers.
struct TypeMap<F: Fn(Reg) -> Option<Type>> {
    map: HashMap<Reg, IsaType>,
    vreg_type: F,
}

impl<F: Fn(Reg) -> Option<Type>> TypeMap<F> {
    fn new(vreg_type: F) -> Self {
        Self {
            map: HashMap::new(),
            vreg_type,
        }
    }

    /// Returns the assigned type of the register or the default type if none was assigned.
    fn get(&self, reg: Reg) -> IsaType {
        self.map
            .get(&reg)
            .copied()
            .unwrap_or_else(|| (self.vreg_type)(reg).map_or(IsaType::Invalid, type_to_isatype))
    }

    /// Updates the type assigned to the register.
    ///
    /// Returns whether the new value is different from the existing value or default if none
    fn update(&mut self, reg: Reg, ty: IsaType) -> bool {
        let ty_old = self.get(reg);
        if ty != ty_old {
            log::trace!("New type assignment: {reg:?}({ty_old:?}) <- {ty:?}");
            self.map.insert(reg, ty);
            return true;
        }
        false
    }
}

impl<F: Fn(Reg) -> Option<Type>> Debug for TypeMap<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.map.fmt(f)
    }
}

/// A Scry backend.
pub struct ScryBackend {
    triple: Triple,
    flags: shared_settings::Flags,
    isa_flags: scry_settings::Flags,
}

impl ScryBackend {
    /// Create a new scry backend with the given (shared) flags.
    pub fn new_with_flags(
        triple: Triple,
        flags: shared_settings::Flags,
        isa_flags: scry_settings::Flags,
    ) -> ScryBackend {
        ScryBackend {
            triple,
            flags,
            isa_flags,
        }
    }

    /// This performs lowering to VCode, register-allocates the code, computes block layout and
    /// finalizes branches. The result is ready for binary emission.
    fn compile_vcode(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<(VCode<inst::MInst>, VRegAllocator<MInst>)> {
        let emit_info = EmitInfo::new(self.flags.clone(), self.isa_flags.clone());
        let sigs = SigSet::new::<abi::ScryMachineDeps>(func, &self.flags)?;
        let abi = Callee::<abi::ScryMachineDeps>::new(func, self, &self.isa_flags, &sigs)?;

        // ------ The below code is copied from cranelift/codegen/src/machinst/compile.rs ------
        // Compute lowered block order.
        let block_order = BlockLoweringOrder::new(func, domtree, ctrl_plane);

        // Build the lowering context.
        let lower = crate::machinst::Lower::new(
            func,
            abi,
            emit_info,
            block_order,
            sigs,
            self.flags().clone(),
        )?;

        // Lower the IR.
        let vcode = {
            log::debug!(
                "Number of CLIF instructions to lower: {}",
                func.dfg.num_insts()
            );
            log::debug!("Number of CLIF blocks to lower: {}", func.dfg.num_blocks());

            let _tt = timing::vcode_lower();
            lower.lower(self, ctrl_plane)?
        };

        log::debug!(
            "Number of lowered vcode instructions: {}",
            vcode.num_insts()
        );
        log::debug!("Number of lowered vcode blocks: {}", vcode.num_blocks());
        trace!("vcode from lowering: \n{:?}", vcode);
        // ------ The above code is copied from cranelift/codegen/src/machinst/compile.rs ------

        // To be able to create more virtual registers, we crate a new vreg allocator
        // that will generate new vregs that don't class with the existing ones in the vcode.
        let mut new_vregs = VRegAllocator::<MInst>::with_capacity(vcode.num_vregs());
        let mut existing_vregs = HashSet::new();

        let mut worklist = vec![vcode.entry_block()];
        let mut done = HashSet::new();

        // Walk the CFG, looking for vregs
        while let Some(b) = worklist.pop() {
            done.insert(b);
            for inst in vcode.block_insns(b).iter() {
                for op in vcode.inst_operands(inst).iter() {
                    existing_vregs.insert(op.vreg());
                }
            }
            for succ in vcode.block_succs(b) {
                if !done.contains(&succ) && !worklist.contains(&succ) {
                    worklist.push(*succ)
                }
            }
        }

        // Find the maximum index between the used vregs
        let max_vreg_idx = existing_vregs
            .iter()
            .fold(0, |acc, r| std::cmp::max(r.vreg(), acc));

        // Keeps making vregs in new allocator, until it exceeds the existing vreg indices
        while max_vreg_idx
            > new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
                .to_virtual_reg()
                .unwrap()
                .index()
        {}

        Ok((vcode, new_vregs))
    }
}

fn prepare_block_params(
    cfg: &mut VCodeCFG<MInst>,
    func_sig: &Signature,
    mut new_vreg: impl FnMut() -> Reg,
) {
    // Handle entry block first, moving MInst:Args to the params
    let entry_bb = cfg.graph.root_weight_mut();

    // Lowering emits no Args at all when none of the function's parameters are
    // used. Insert an empty one; the gap-filling below reconstructs the full
    // positional list from the signature either way.
    if !matches!(entry_bb.inst.first(), Some(MInst::Args { .. })) {
        entry_bb.inst.insert(0, MInst::Args { args: vec![] });
    }

    match &entry_bb.inst[0] {
        MInst::Args { args } => {
            // Unused parameters are not lowered into Args, so reconstruct the full
            // positional list from the signature, using each ArgPair's preg (which
            // encodes the parameter index) and filling gaps with fresh, unused vregs
            // that will be discarded below.
            let mut params: Vec<Option<Reg>> = vec![None; func_sig.params.len()];
            for p in args {
                params[p.preg.to_real_reg().unwrap().hw_enc() as usize] = Some(p.vreg.to_reg());
            }
            entry_bb.params = params
                .into_iter()
                .map(|p| p.unwrap_or_else(&mut new_vreg))
                .collect();

            // Rewrite Args to define all parameters in positional order, so gap vregs
            // have a definition and def order matches the physical arrival order.
            entry_bb.inst[0] = MInst::Args {
                args: entry_bb
                    .params
                    .iter()
                    .map(|p| ArgPair {
                        vreg: WritableReg::from_reg(*p),
                        preg: *p,
                    })
                    .collect(),
            };
        }
        inst => unreachable!("Entry did not include MInst::Args at the start: {:?}", inst),
    }

    let dfg = cfg.dataflow_graph();

    // Resolve parameter orders. Every block's outgoing edges are processed at
    // least once; blocks are re-added whenever an order they participate in
    // changes.
    let mut worklist: HashSet<usize> = cfg.graph.all_vertices().collect();

    // Entry block has identical param order to its actual params
    let entry_param_order = cfg
        .graph
        .root_weight()
        .params
        .iter()
        .cloned()
        .map(|x| Some(x))
        .collect();
    cfg.graph.root_weight_mut().param_order = entry_param_order;

    log::trace!("Resolving parameter ordering");
    log::trace!("CFG: {cfg:?}");
    while let Some(bb_v) = worklist.iter().next().copied() {
        worklist.remove(&bb_v);

        log::trace!("Resolving block: {bb_v}");
        for succ_v in cfg
            .graph
            .edges_sourced_in(bb_v)
            .map(|(v, _)| v)
            .collect::<Vec<_>>()
            .into_iter()
        {
            log::trace!("Successor: {succ_v}");
            let bb = cfg.graph.vertex_weight(bb_v).unwrap();
            let succ_bb = cfg.graph.vertex_weight(succ_v).unwrap();

            let br_params = &bb.branch_params[&succ_bb.vcode_bb];
            assert_eq!(br_params.len(), succ_bb.params.len());

            log::trace!("Block {bb_v} branch params: {br_params:?}");
            log::trace!(
                "Block {bb_v} branch param order: {:?}",
                bb.branch_param_order
            );
            log::trace!("Block {succ_v} params: {:?}", succ_bb.params);
            log::trace!("Block {succ_v} param order: {:?}", succ_bb.param_order);

            // Get a map of this block's register outputs and successors register inputs
            let mut reg_map = HashMap::new();

            for (out_idx, out_r) in br_params.iter().enumerate() {
                let others = dfg
                    .edges_between(bb_v, succ_v)
                    .find(|edge| edge.0 == out_idx)
                    .unwrap()
                    .1
                    .iter()
                    .cloned()
                    .filter(|r| r != out_r)
                    .collect::<Vec<_>>();
                assert!(others.len() == 1);
                reg_map.insert((out_idx, *out_r), others[0]);
            }

            log::trace!("Parameter register mapping: {reg_map:?}");

            // Unify existing param order.
            let mut bb_branch_param_order = bb.branch_param_order.clone();
            let mut succ_param_order = succ_bb.param_order.clone();

            // Equalize lengths, then propagate known positions across the edge.
            let len = max(succ_param_order.len(), bb_branch_param_order.len());
            bb_branch_param_order.resize(len, None);
            succ_param_order.resize(len, None);

            for idx in 0..len {
                match (bb_branch_param_order[idx], succ_param_order[idx]) {
                    (Some(bb_p), Some(succ_p)) => {
                        // Both sides positioned: they must be each other's
                        // counterpart on this edge.
                        assert!(
                            reg_map
                                .iter()
                                .any(|((_, p1), p2)| *p1 == bb_p && *p2 == succ_p),
                            "Incongruence in this block and succ"
                        );
                    }
                    (Some(bb_p), None) => {
                        // If this edge passes bb_p (unambiguously), the successor's
                        // corresponding register takes the same position.
                        let mut targets = reg_map
                            .iter()
                            .filter(|((_, p1), _)| *p1 == bb_p)
                            .map(|(_, r)| *r);
                        match (targets.next(), targets.next()) {
                            (Some(succ_r), None) if !succ_param_order.contains(&Some(succ_r)) => {
                                succ_param_order[idx] = Some(succ_r);
                            }
                            // Not passed on this edge, ambiguous (duplicate
                            // arguments), or already positioned: leave it to the
                            // parameter loop below.
                            _ => (),
                        }
                    }
                    (None, Some(succ_p)) => {
                        // The successor's register is one of this edge's parameters;
                        // the corresponding passed register takes the position.
                        if let Some(((_, bb_r), _)) = reg_map.iter().find(|(_, r)| **r == succ_p) {
                            bb_branch_param_order[idx] = Some(*bb_r);
                        }
                    }
                    (None, None) => (),
                }
            }

            // Ensure every parameter passed on this edge has a position. Keyed on
            // the successor's parameter registers, which are unique, so that
            // duplicate arguments (the same register passed at several positions)
            // are supported.
            for (param_idx, param) in br_params.iter().enumerate() {
                let succ_param_reg = reg_map[&(param_idx, *param)];
                if let Some(order_idx) = succ_param_order
                    .iter()
                    .position(|p| *p == Some(succ_param_reg))
                {
                    // The successor has a position for this parameter.
                    match bb_branch_param_order[order_idx] {
                        None => bb_branch_param_order[order_idx] = Some(*param),
                        Some(p) => {
                            assert_eq!(p, *param, "Incongruence in this block and succ")
                        }
                    }
                } else {
                    // Neither side has a position for this parameter yet. Add to both.
                    bb_branch_param_order.push(Some(*param));
                    succ_param_order.push(Some(succ_param_reg));
                }
            }

            assert_eq!(bb_branch_param_order.len(), succ_param_order.len());

            // Every parameter of the successor now has a position: its parameters
            // and this edge's passed registers map one-to-one.
            debug_assert!(
                succ_bb
                    .params
                    .iter()
                    .all(|p| succ_param_order.contains(&Some(*p)))
            );

            if bb.branch_param_order != bb_branch_param_order {
                // This block's output order changed: re-evaluate its edges to all
                // successors (which share the order).
                worklist.insert(bb_v);
            }

            if succ_bb.param_order != succ_param_order {
                // The successor's input order changed: re-evaluate the edges from
                // all its predecessors.
                for (pred_v, _) in cfg.graph.edges_sinked_in(succ_v) {
                    worklist.insert(pred_v);
                }
            }

            // Update orders (if changed)
            cfg.graph
                .vertex_weight_mut(bb_v)
                .unwrap()
                .branch_param_order = bb_branch_param_order;
            cfg.graph.vertex_weight_mut(succ_v).unwrap().param_order = succ_param_order;

            let bb = cfg.graph.vertex_weight(bb_v).unwrap();
            let succ_bb = cfg.graph.vertex_weight(succ_v).unwrap();
            log::trace!(
                "Block {bb_v} branch params: {:?}",
                bb.branch_params[&succ_bb.vcode_bb]
            );
            log::trace!(
                "Block {bb_v} branch param order: {:?}",
                bb.branch_param_order
            );
            log::trace!("Block {succ_v} params: {:?}", succ_bb.params);
            log::trace!("Block {succ_v} param order: {:?}", succ_bb.param_order);
        }
    }
    log::trace!("Final parameter ordering:");
    for (bb_v, bb) in cfg.graph.all_vertices_weighted() {
        log::trace!("Block {bb_v} params: {:?}", bb.params);
        log::trace!("Block {bb_v} param order: {:?}", bb.param_order);
        log::trace!("Block {bb_v} branch params: {:?}", bb.branch_params);
        log::trace!(
            "Block {bb_v} branch param order: {:?}",
            bb.branch_param_order
        );

        // Assert all parameters are present in the order exactly once
        assert!(
            bb.params
                .iter()
                .all(|p| { bb.param_order.iter().filter(|po| **po == Some(*p)).count() == 1 })
        );

        // Assert every branch parameter has at least as many positions in the order
        // as the number of times it is passed on any single edge (a register may be
        // passed at several positions of the same branch)
        assert!(bb.branch_params.values().all(|params| {
            params.iter().all(|p| {
                bb.branch_param_order
                    .iter()
                    .filter(|po| **po == Some(*p))
                    .count()
                    >= params.iter().filter(|p2| *p2 == p).count()
            })
        }));
    }

    let entry_v = cfg.graph.root();
    for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
        let params = bb
            .param_order
            .iter()
            .map(|r| {
                // A None position is not used by this block; a fresh vreg is created for
                // it and later discarded (it will have no uses).
                r.unwrap_or_else(|| new_vreg())
            })
            .collect::<Vec<_>>();

        if bb_v != entry_v {
            // Insert MInst::Args for block params
            bb.inst.insert(
                0,
                MInst::Args {
                    args: params
                        .iter()
                        .map(|p| ArgPair {
                            vreg: WritableReg::from_reg(*p),
                            preg: *p,
                        })
                        .collect(),
                },
            )
        }

        // Insert echos for handling params
        let echo_regs = if !bb.param_order.is_empty() {
            let echo_regs = params
                .iter()
                .map(|p| {
                    let new_vr = new_vreg();
                    replace_all_uses(bb, *p, new_vr);
                    new_vr
                })
                .collect::<Vec<_>>();

            let rds = echo_regs
                .iter()
                .map(|r| WritableReg::from_reg(*r))
                .collect();

            bb.inst.insert(
                1, // Insert after Args
                MInst::Echo { rss: params, rds },
            );

            echo_regs
        } else {
            vec![]
        };

        // Update jump trigger inputs
        if let Some((_, trigger)) = get_jmp_issues(bb) {
            // Every position of the output order must carry a value of this block's
            // own. A None position would mean a successor's input order reserves a
            // position for another predecessor's value, which cannot happen as long
            // as all critical edges are split (see `from_vcode`): every successor
            // then either has this block as its only predecessor, or is a dedicated
            // jump block. Should a future pass reintroduce unsplit critical edges,
            // this block would have to send a padding value (e.g. a duplicate of a
            // real argument) at each None position to keep the wire positions of
            // its tuple aligned with the successor's input order.
            assert!(
                bb.branch_param_order.iter().all(|r| r.is_some()),
                "Output order contains positions carrying no value of this block's \
                 (would require padding): {:?}",
                bb.branch_param_order
            );

            match &mut bb.inst[trigger] {
                MInst::JumpTrigger { args, .. } => {
                    *args = bb.branch_param_order.iter().map(|r| r.unwrap()).collect();
                }
                _ => unreachable!(),
            }
        }

        // Any echoed parameter without a use left in the block (an unused parameter or
        // an unused parameter order position) must be explicitly discarded so its value
        // does not linger in the operand queue.
        // For consistency and the ability to easily assert that no instructions get >4
        // inputs, we only give the discard the same number of values to drop as other
        // instructions can accept.
        let unused = echo_regs
            .into_iter()
            .filter(|r| bb.reg_uses(*r).next().is_none())
            .collect::<Vec<_>>();
        for chunk in unused.chunks(QUEUE_CAPACITY) {
            // Insert discard directly after the echo, which routes the values to it
            bb.inst.insert(2, MInst::Discard { rss: chunk.to_vec() });
        }
    }

    log::trace!("VCodeCFG: {cfg:?}");
}

/// Inserts the stack frame reservation and release instructions.
///
/// The frame (sized by the function's stack slots) is reserved on the private
/// frame at function entry and released before every return by freeing the
/// shared frame (which, being empty, shrinks the private frame and actually
/// releases the memory).
///
/// The reserve/free instructions can only encode power-of-two amounts, so one
/// instruction per set bit of the frame size is emitted.
///
/// Must run before [`insert_ref_distances`] so in-flight operands crossing the
/// inserted instructions get correct distances. The reserve is placed after the
/// entry block's parameter-echo prefix: the function's arguments arrive at the
/// first executed instruction, and a reserve must never have operands delivered
/// to it.
fn insert_frame_limits(cfg: &mut VCodeCFG<MInst>, frame_bytes: u32) {
    if frame_bytes == 0 {
        return;
    }
    assert!(
        frame_bytes < (1 << 16),
        "Stack frame too large: {frame_bytes} bytes"
    );
    let pows: Vec<u16> = (0..16u16).filter(|k| frame_bytes & (1 << k) != 0).collect();

    // Reserve at entry, after the leading Args/Echo/Discard parameter handling.
    let entry_bb = cfg.graph.root_weight_mut();
    let mut pos = 0;
    while pos < entry_bb.inst.len()
        && matches!(
            entry_bb.inst[pos],
            MInst::Args { .. } | MInst::Echo { .. } | MInst::Discard { .. }
        )
    {
        pos += 1;
    }
    for k in pows.iter() {
        entry_bb.inst.insert(
            pos,
            MInst::StackAdjust {
                reserve: true,
                amount_pow2: *k,
            },
        );
        pos += 1;
    }

    // Free directly before every return.
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        if let Some(ret_idx) = bb.inst.iter().position(|i| matches!(i, MInst::Ret { .. })) {
            for (i, k) in pows.iter().enumerate() {
                bb.inst.insert(
                    ret_idx + i,
                    MInst::StackAdjust {
                        reserve: false,
                        amount_pow2: *k,
                    },
                );
            }
        }
    }

    log::trace!("VCodeCFG: {cfg:?}");
}

/// Replaces the all uses of the `find` register with the `replace` register
fn replace_all_uses(bb: &mut VCodeBB<MInst>, find: Reg, replace: Reg) {
    while replace_first_use(bb, find, replace) {}
}

/// Replaces the first use of the `find` register with the `replace` register, returning whether any replacement was made
fn replace_first_use(bb: &mut VCodeBB<MInst>, find: Reg, replace: Reg) -> bool {
    // replace in all instructions
    for inst in bb.inst.iter_mut() {
        if let Some(r) = inst.get_uses_mut().find(|r| **r == find) {
            *r = replace;
            return true;
        }
    }
    // replace in parameter ordering
    while let Some(Some(r)) = bb.branch_param_order.iter_mut().find(|r| **r == Some(find)) {
        *r = replace;
        // If we managed to find it in the order, we should also look for it in the regular params
        bb.branch_params.iter_mut().for_each(|(_, params)| {
            if let Some(p) = params.iter_mut().find(|r| **r == find) {
                *p = replace;
            }
        });
        return true;
    }
    false
}

/// Looks for registers that are used multiple times and inserts `dup` instructions at their definition
/// and makes the uses use the resulting, unique registers. I.e. eliminates multiple uses of the same register.
fn insert_duplicates(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("insert_duplicates");
    let mut find_and_replace = |bb: &mut VCodeBB<MInst>, reg, dup_idx| {
        let rd1 = new_vreg();
        let rd2 = new_vreg();

        // Replace uses with uses of new vregs
        assert!(replace_first_use(bb, reg, rd1));
        assert!(replace_first_use(bb, reg, rd2));

        // Insert duplication of original register to new vregs
        bb.inst.insert(
            dup_idx,
            MInst::Duplicate {
                rd1: Writable::from_reg(rd1),
                rd2: Writable::from_reg(rd2),
                rs: reg,
                out1: 0,
                out2: 0,
            },
        );
    };

    'a: loop {
        for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
            log::trace!("bb {bb_v}: {bb:?}");
            // Check each def for multiple uses
            for (inst_idx, reg_def) in bb.inst_defs().collect::<Vec<_>>() {
                if bb.reg_uses(reg_def).count() > 1 {
                    find_and_replace(bb, reg_def, inst_idx + 1);
                    continue 'a;
                }
            }
        }
        // No changes were made, finish
        break;
    }
    log::trace!("VCodeCFG: {cfg:?}");
}

fn insert_ref_distances(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("insert_ref_distances");
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        'a: loop {
            log::trace!("BB: {bb:?}");

            // reg -> (instruction index, reference distance, extra delivery distance).
            // The extra distance is non-zero for branch arguments delivered to a later
            // instruction of the successor's receiving echo chain (see `delivery_group`).
            let mut use_pos = HashMap::<Reg, (usize, u16, u16)>::new();
            let mut ref_dist = 0;
            for (inst_idx, inst) in bb.inst.iter_mut().rev().enumerate() {
                log::trace!("inst: {inst:?}");

                // Expand any echo into echo chains and start over. Pre-existing echoes
                // were already expanded by expand_echoes; this only handles the echoes
                // this pass inserts itself (see the CallArgs arm below).
                match &inst {
                    MInst::Echo { rds, rss } => {
                        for i in MInst::receive_chain(
                            rds.iter()
                                .cloned()
                                .zip(rss.iter().cloned())
                                .collect::<Vec<_>>()
                                .as_slice(),
                            &mut new_vreg,
                        )
                        .into_iter()
                        {
                            bb.inst.insert(bb.inst.len() - 1 - inst_idx, i);
                        }
                        bb.inst.remove(bb.inst.len() - 1 - inst_idx);
                        continue 'a;
                    }
                    _ => (),
                }

                ref_dist += inst.reference_length() as u16;

                // Record all uses, we can do this on-the-fly as we assume no use comes before its def
                match inst {
                    // A jump trigger's reference distance points at the successor's
                    // first executed instruction, whose queue only has capacity for
                    // the first arguments; the following arguments are delivered to
                    // later instructions of the successor's receiving echo chain,
                    // recorded here as extra delivery distance by wire position.
                    MInst::JumpTrigger { args, .. } => {
                        for (wire_idx, r) in args.iter().enumerate() {
                            use_pos.entry(*r).or_insert((
                                inst_idx,
                                ref_dist,
                                delivery_group(wire_idx) as u16,
                            ));
                        }
                    }
                    _ => inst.get_uses().for_each(|r| {
                        use_pos.entry(*r).or_insert((inst_idx, ref_dist, 0));
                    }),
                }

                if inst.get_defs().count() >= 1 {
                    // A def without any use would leave its value stranded in
                    // the operand queue. A two-output Alu2 can simply not emit
                    // the dead output (the Low/High single-output variants);
                    // anything else routes the value to an explicit discard.
                    let dead = inst
                        .get_defs()
                        .enumerate()
                        .find(|(_, def)| !use_pos.contains_key(def));
                    if let Some((dead_slot, dead)) = dead {
                        match inst {
                            MInst::Alu2 {
                                out_var, rds, outs, ..
                            } if rds.len() == 2 => {
                                rds.remove(dead_slot);
                                outs.remove(dead_slot);
                                *out_var = if dead_slot == 1 {
                                    Alu2OutputVariant::Low
                                } else {
                                    Alu2OutputVariant::High
                                };
                            }
                            _ => {
                                bb.inst.insert(
                                    bb.inst.len() - inst_idx,
                                    MInst::Discard { rss: vec![dead] },
                                );
                            }
                        }
                        continue 'a;
                    }

                    let ref_dists = inst
                        .get_defs()
                        .enumerate()
                        .map(|(i, def)| {
                            let use_idx = use_pos[&def];
                            (
                                i,
                                ref_dist - use_idx.1 - (inst.reference_length() as u16)
                                    + use_idx.2,
                            )
                        })
                        .collect::<HashMap<_, _>>();

                    // A def whose value the machine can only deliver to the
                    // next instruction, while its consumer sits further away,
                    // is bridged with an echo that carries the remaining
                    // distance.
                    let bridge_reg = match inst {
                        // These deliver their output to the next instruction
                        // (their encodings have no output reference field).
                        MInst::Const { rd, .. }
                        | MInst::LoadExtName { rd, .. }
                        | MInst::LoadStack { rd, .. }
                        | MInst::SAddr { rd, .. }
                            if ref_dists[&0] > 0 =>
                        {
                            Some(rd.to_reg())
                        }
                        // The two-output Alu2 encoding sends both outputs to
                        // one shared reference, or one output to the next
                        // instruction and the other to the reference (see
                        // emission). Two distinct non-zero references cannot
                        // be encoded: bridge the second (high) output, which
                        // preserves the physical production order.
                        MInst::Alu2 { rds, .. }
                            if rds.len() == 2
                                && ref_dists[&0] != ref_dists[&1]
                                && ref_dists[&0] > 0
                                && ref_dists[&1] > 0 =>
                        {
                            Some(rds[1].to_reg())
                        }
                        _ => None,
                    };
                    if let Some(rd) = bridge_reg {
                        let fresh = new_vreg();
                        replace_all_uses(bb, rd, fresh);
                        bb.inst.insert(
                            bb.inst.len() - inst_idx,
                            MInst::EchoLong {
                                rds: vec![Writable::from_reg(fresh)],
                                rss: vec![rd],
                                out: 0,
                            },
                        );
                        continue 'a;
                    }

                    match inst {
                        MInst::Alu1 { out, .. }
                        | MInst::Load { out, .. }
                        | MInst::Cast { out, .. }
                        | MInst::EchoLong { out, .. } => {
                            *out = ref_dists[&0];
                        }
                        MInst::Alu2 { outs, .. } => outs
                            .iter_mut()
                            .enumerate()
                            .for_each(|(i, out)| *out = ref_dists[&i]),
                        MInst::EchoChain { out1, out2, .. } => {
                            // EchoChain's get_defs order is [rd2, rd1, chain..] (physical
                            // production order), so out1 (routing rs1->rd1) takes the
                            // distance of def index 1 and out2 that of def index 0.
                            *out1 = ref_dists[&1];
                            *out2 = ref_dists[&0];
                        }
                        MInst::EchoSplit { out1, out2, .. }
                        | MInst::Duplicate { out1, out2, .. } => {
                            *out1 = ref_dists[&0];
                            *out2 = ref_dists[&1];
                        }
                        MInst::Reorder {
                            rd1,
                            rd2,
                            rs1,
                            rs2,
                            out,
                        } => {
                            if ref_dists[&0] == ref_dists[&1] {
                                *out = ref_dists[&0];
                            } else {
                                // The two outputs go to different targets (e.g. a chain of
                                // reorders), so this is a split rather than a same-target
                                // swap; arrival order between different targets does not
                                // matter, so an EchoSplit expresses it. Reorder's get_defs
                                // order is [rd2, rd1] (physical production order), so out1
                                // (routing rs1->rd1) takes the distance of def index 1.
                                *inst = MInst::EchoSplit {
                                    rd1: *rd1,
                                    rd2: *rd2,
                                    rs1: *rs1,
                                    rs2: *rs2,
                                    out1: ref_dists[&1],
                                    out2: ref_dists[&0],
                                };
                            }
                        }
                        MInst::CallArgs { rets, .. } if ref_dists.iter().any(|(_, d)| *d > 0) => {
                            // Some call arguments aren't going to the following instruction

                            let rss = rets.iter().map(|g| g.vreg.to_reg()).collect::<Vec<_>>();

                            let rds = rss
                                .iter()
                                .map(|rs| {
                                    let rd = new_vreg();

                                    replace_all_uses(bb, *rs, rd);

                                    WritableReg::from_reg(rd)
                                })
                                .collect();

                            bb.inst
                                .insert(bb.inst.len() - inst_idx, MInst::Echo { rds, rss });
                            continue 'a;
                        }
                        _ => (),
                    };
                }
            }
            break;
        }
        log::trace!("BB: {bb:?}");
    }
    log::trace!("VCodeCFG: {cfg:?}");
}

/// Checks that no instruction consumes more operands than its ready queue can
/// hold ([`QUEUE_CAPACITY`]); the machine silently drops operands delivered
/// beyond that.
///
/// Deliveries are counted at the consuming instruction via the use relation,
/// so values leaving the block or frame (branch arguments, call arguments,
/// return values) are not counted at their trigger pseudo-instruction: branch
/// and call arguments are counted where they arrive, at the receiving block's
/// parameter echoes (its `Args` defs), and return values at the caller's
/// `CallArgs` defs' consumers.
///
/// Must run after [`insert_ref_distances`], when all pseudo-instructions with
/// unbounded operand lists have been expanded.
fn assert_queue_capacities(cfg: &mut VCodeCFG<MInst>) {
    for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
        // Consumer instruction (index from block end) of each used reg.
        let mut use_pos = HashMap::<Reg, usize>::new();
        // Consumers whose operands physically leave the block or frame.
        let mut boundary = HashSet::<usize>::new();
        for (inst_idx, inst) in bb.inst.iter().rev().enumerate() {
            if matches!(
                inst,
                MInst::JumpTrigger { .. } | MInst::CallArgs { .. } | MInst::Rets { .. }
            ) {
                boundary.insert(inst_idx);
            }
            inst.get_uses().for_each(|r| {
                use_pos.entry(*r).or_insert(inst_idx);
            });
        }

        let mut arrivals = HashMap::<usize, usize>::new();
        for inst in bb.inst.iter() {
            for def in inst.get_defs() {
                if let Some(idx) = use_pos.get(&def) {
                    if !boundary.contains(idx) {
                        *arrivals.entry(*idx).or_default() += 1;
                    }
                }
            }
        }

        for (idx, count) in arrivals {
            assert!(
                count <= QUEUE_CAPACITY,
                "{count} operands delivered to one instruction (capacity \
                 {QUEUE_CAPACITY}) in block {bb_v}: {:?}",
                bb.inst[bb.inst.len() - 1 - idx]
            );
        }
    }
}

/// Expands all Echo pseudo-instructions into EchoChain/EchoSplit/EchoLong sequences.
///
/// This must run before [`fix_orderings`] so that ordering decisions can see the
/// physical production order of the concrete echo instructions.
fn expand_echoes(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("expand_echoes");
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        let mut idx = 0;
        while idx < bb.inst.len() {
            if let MInst::Echo { rds, rss } = &bb.inst[idx] {
                let expansion = MInst::receive_chain(
                    rds.iter()
                        .cloned()
                        .zip(rss.iter().cloned())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    &mut new_vreg,
                );
                let expansion_len = expansion.len();
                bb.inst.splice(idx..idx + 1, expansion);
                idx += expansion_len;
            } else {
                idx += 1;
            }
        }
    }
    log::trace!("VCodeCFG: {cfg:?}");
}

fn fix_orderings(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("fix_orderings");
    for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
        log::trace!("bb {bb_v}: {bb:?}");
        'a: loop {
            // Def positions at (instruction index, production slot) granularity.
            let mut def_pos = HashMap::<Reg, (usize, usize)>::new();
            for (inst_idx, inst) in bb.inst.iter().enumerate() {
                for (slot, def) in inst.get_defs().enumerate() {
                    def_pos.insert(def, (inst_idx, slot));
                }
            }

            for (inst_idx, inst) in bb.inst.iter().enumerate() {
                log::trace!("Inst {inst_idx}: {inst:?}");

                if inst.use_order_meaningful() && inst.get_uses().count() > 1 {
                    // Find the first adjacent pair of uses that will arrive in the wrong
                    // order. Repeated passes sort any permutation pairwise.
                    let uses = inst.get_uses().cloned().collect::<Vec<_>>();
                    let wrong_order_pair =
                        (0..uses.len() - 1).find(|i| def_pos[&uses[i + 1]] < def_pos[&uses[*i]]);

                    if let Some(pair_idx) = wrong_order_pair {
                        // If the reorder is needed for branch or call arguments, the reorder
                        // instruction must come before the branch or call issue. For a block with
                        // several issues (a lowered `br_table`) that means before the first of
                        // them, so that the run of issues stays contiguous.
                        let insert_idx = match inst {
                            MInst::JumpTrigger { .. } => {
                                get_jmp_issues(bb).expect("JumpTrigger without issue").0[0]
                            }
                            MInst::CallArgs { .. } => bb.inst[..inst_idx]
                                .iter()
                                .rposition(|i| matches!(i, MInst::Call { .. }))
                                .expect("CallArgs without preceding Call"),
                            // Return values are consumed by the ret trigger, so
                            // their reorder must execute before the Ret issue.
                            MInst::Rets { .. } => bb.inst[..inst_idx]
                                .iter()
                                .rposition(|i| matches!(i, MInst::Ret { .. }))
                                .expect("Rets without preceding Ret"),
                            _ => inst_idx,
                        };

                        // Create new vregs for the reorder
                        let first_new = new_vreg();
                        let second_new = new_vreg();

                        let first_old = uses[pair_idx + 1];
                        let second_old = uses[pair_idx];

                        // Assign reordered vregs to the consumer
                        let inst = &mut bb.inst[inst_idx];
                        let mut uses_mut = inst.get_uses_mut().collect::<Vec<_>>();
                        *uses_mut[pair_idx] = second_new;
                        *uses_mut[pair_idx + 1] = first_new;

                        // Insert reorder instruction before the consumer (or its issue)
                        bb.inst.insert(
                            insert_idx,
                            MInst::Reorder {
                                rd1: Writable::from_reg(first_new),
                                rd2: Writable::from_reg(second_new),
                                rs1: first_old,
                                rs2: second_old,
                                out: 0,
                            },
                        );

                        // Start over
                        continue 'a;
                    }
                }
            }
            break;
        }
    }
    log::trace!("VCodeCFG: {cfg:?}");
}

fn type_to_isatype(t: Type) -> IsaType {
    use crate::ir::types::*;
    use IsaType::*;
    match t {
        I8 => Integer(0),
        I16 => Integer(1),
        I32 => Integer(2),
        I64 => Integer(3),
        _ => unimplemented!(),
    }
}

fn abi_param_to_isatype(p: &AbiParam) -> IsaType {
    let t = type_to_isatype(p.value_type);
    match p.extension {
        ArgumentExtension::None => t,
        ArgumentExtension::Sext => IsaType::new_known_int(t.size_pow2(), true),
        ArgumentExtension::Uext => IsaType::new_known_int(t.size_pow2(), false),
    }
}

/// Demanded re-tag casts, produced by [type_analysis_phase] and applied by
/// [apply_cast_demands]: for the instruction at (block vertex, instruction
/// index), the register at the given slot must be re-tagged to the given
/// type. Whether the slot names a use or a def of the instruction decides
/// where the re-tag happens: a use is re-tagged on its way INTO the
/// instruction (cast before it, the slot redirected to the cast's output),
/// a def on its way OUT (the slot redirected to a fresh register that the
/// cast re-tags into the original directly after).
///
/// Demands are per-SLOT, not per-register: an instruction may use the same
/// register at several slots with different type requirements (e.g. the same
/// value passed to two parameters with different extension annotations).
type CastDemands = HashMap<(usize, usize), Vec<(usize, Reg, IsaType)>>;

/// Records a demanded re-tag to `ty` of the register at use/def slot `slot`
/// of the instruction at (`bb_v`, `inst_idx`), ignoring duplicates
/// (instructions are re-analyzed many times).
fn push_demand(
    demands: &mut CastDemands,
    bb_v: usize,
    inst_idx: usize,
    slot: usize,
    reg: Reg,
    ty: IsaType,
) {
    let entry = demands.entry((bb_v, inst_idx)).or_default();
    if !entry.iter().any(|(s, r, _)| *s == slot && *r == reg) {
        log::trace!("Cast demand: {reg:?} -> {ty:?} at ({bb_v}, {inst_idx}), slot {slot}");
        entry.push((slot, reg, ty));
    }
}

fn type_analysis<F: Fn(Reg) -> Option<Type>>(
    cfg: &mut VCodeCFG<MInst>,
    vreg_type: F,
    func_sig: &Signature,
    mut new_vreg: impl FnMut() -> Reg,
) -> TypeMap<F> {
    log::trace!("Type Analysis");

    let mut type_map = TypeMap::new(vreg_type);

    // Map function parameters to entry block registers
    for (r, p) in cfg
        .graph
        .root_weight()
        .params
        .iter()
        .zip(func_sig.params.iter())
    {
        let ty = abi_param_to_isatype(p);
        let refined = type_map.get(*r).refine(ty).unwrap();
        type_map.update(*r, refined);
    }

    // For every register that crosses a block boundary, the blocks whose type
    // reasoning can be affected by it: both endpoints of every dataflow edge
    // naming it. A block reasons about a register of its own *and* about the
    // registers on the other side of its edges (see the `JumpTrigger` arm, which
    // types its successors' parameters), so a change to any of them must
    // re-queue every block on an edge carrying it, not just the neighbours of
    // the block that made the change. Registers absent here are block-local and
    // are handled entirely by the instruction worklist below. (Cast insertion
    // below never adds cross-block registers, so the map stays valid.)
    let mut reg_blocks: HashMap<Reg, HashSet<usize>> = HashMap::new();
    {
        let bb_dfg = cfg.dataflow_graph();
        for (src, sink, weight) in bb_dfg.all_edges() {
            for r in weight.1.iter() {
                let blocks = reg_blocks.entry(*r).or_insert_with(HashSet::new);
                blocks.insert(src);
                blocks.insert(sink);
            }
        }
    }

    // The fixpoint runs in two phases. The first propagates only hard
    // constraints (ABI attributes, extends, comparisons, ...). The second
    // additionally enables the best-effort unifications (BinaryAlu, icmp
    // eq/ne, extend results), which are only preferences: running them after
    // every hard constraint has settled keeps the result order-independent,
    // as a preference can never commit a signedness that a hard constraint
    // would later contradict.
    //
    // A hard constraint on an operand whose type is already established with
    // the opposite signedness records a cast demand instead: the operand is
    // re-tagged (same-size cast) for that consumer only, and the analysis is
    // rerun until no demands remain.
    let mut demands: CastDemands = HashMap::new();
    let mut rounds = 0;
    loop {
        type_analysis_phase(cfg, func_sig, &reg_blocks, &mut type_map, &mut demands, false);
        type_analysis_phase(cfg, func_sig, &reg_blocks, &mut type_map, &mut demands, true);

        if demands.is_empty() {
            break;
        }
        rounds += 1;
        assert!(rounds < 8, "Signedness cast insertion did not converge");
        apply_cast_demands(cfg, &mut demands, &mut type_map, &mut new_vreg);
    }

    log::trace!("Resolved types: {type_map:?}");

    type_map
}

/// Applies (and drains) the demanded re-tag casts: each demanded operand is
/// routed through a fresh register defined by a same-size `Cast` to the
/// demanded type, inserted before the consuming instruction (before the
/// issuing instruction, for consumers that are trigger pseudos), and only
/// that instruction's uses are redirected. The fresh register is seeded in
/// the type map with the demanded type.
fn apply_cast_demands<F: Fn(Reg) -> Option<Type>>(
    cfg: &mut VCodeCFG<MInst>,
    demands: &mut CastDemands,
    type_map: &mut TypeMap<F>,
    mut new_vreg: impl FnMut() -> Reg,
) {
    // Group per block and apply highest instruction index first, so earlier
    // insertions do not shift pending positions.
    let mut by_bb: HashMap<usize, Vec<(usize, Vec<(usize, Reg, IsaType)>)>> = HashMap::new();
    for ((bb_v, inst_idx), regs) in demands.drain() {
        by_bb.entry(bb_v).or_default().push((inst_idx, regs));
    }

    for (bb_v, mut groups) in by_bb {
        groups.sort_by(|a, b| b.0.cmp(&a.0));
        let bb = cfg.graph.vertex_weight_mut(bb_v).unwrap();

        for (inst_idx, regs) in groups {
            // For operand casts: the re-tagged value must exist before its
            // consumer executes; for trigger pseudos the consuming position
            // is the trigger, so the cast goes before the issuing
            // instruction.
            let insert_at = match &bb.inst[inst_idx] {
                MInst::Rets { .. } => bb.inst[..inst_idx]
                    .iter()
                    .rposition(|i| matches!(i, MInst::Ret { .. }))
                    .expect("Rets without preceding Ret"),
                MInst::CallArgs { .. } => bb.inst[..inst_idx]
                    .iter()
                    .rposition(|i| matches!(i, MInst::Call { .. }))
                    .expect("CallArgs without preceding Call"),
                MInst::JumpTrigger { .. } => {
                    get_jmp_issues(bb).expect("JumpTrigger without issue").0[0]
                }
                _ => inst_idx,
            };

            let mut inst_idx = inst_idx;
            for (slot, reg, ty) in regs {
                let fresh = new_vreg();

                // Whether the slot names a def or a use of the instruction
                // decides the direction of the re-tag.
                if bb.inst[inst_idx].get_defs().nth(slot) == Some(reg) {
                    // Output: redirect the def slot to the fresh register
                    // (whose type the next analysis round derives from the
                    // instruction itself) and re-tag into the original
                    // register right after.
                    *bb.inst[inst_idx]
                        .get_defs_mut()
                        .nth(slot)
                        .expect("Demanded def slot out of range") =
                        WritableReg::from_reg(fresh);
                    bb.inst.insert(
                        inst_idx + 1,
                        MInst::Cast {
                            rd: WritableReg::from_reg(reg),
                            ty,
                            rs: fresh,
                            out: 0,
                        },
                    );
                } else {
                    // Input: re-tag before the instruction and redirect only
                    // the demanded use slot; other slots may require the
                    // register at its original type.
                    type_map.update(fresh, ty);
                    bb.inst.insert(
                        insert_at,
                        MInst::Cast {
                            rd: WritableReg::from_reg(fresh),
                            ty,
                            rs: reg,
                            out: 0,
                        },
                    );
                    inst_idx += 1;
                    let u = bb.inst[inst_idx]
                        .get_uses_mut()
                        .nth(slot)
                        .expect("Demanded use slot out of range");
                    assert_eq!(*u, reg, "Demanded use slot holds another register");
                    *u = fresh;
                }
            }
        }
    }
}

/// One phase of the [type_analysis] fixpoint: analyzes every block, and
/// re-analyzes affected instructions and blocks on every type change, until
/// the assignment stabilizes.
///
/// `enable_soft` gates the best-effort unifications (see the BinaryAlu, icmp
/// eq/ne and extend-result arms). Hard operand constraints that conflict with
/// an established type are recorded in `demands` (see [apply_cast_demands])
/// rather than applied.
fn type_analysis_phase<F: Fn(Reg) -> Option<Type>>(
    cfg: &VCodeCFG<MInst>,
    func_sig: &Signature,
    reg_blocks: &HashMap<Reg, HashSet<usize>>,
    type_map: &mut TypeMap<F>,
    demands: &mut CastDemands,
    enable_soft: bool,
) {
    let bb_dfg = cfg.dataflow_graph();

    let mut bb_worklist: VecDeque<usize> = VecDeque::new();

    // All blocks analyzed at least once
    bb_worklist.extend(cfg.graph.all_vertices());

    while let Some((bb_v, bb)) = bb_worklist
        .pop_front()
        .map(|bb_v| (bb_v, cfg.graph.vertex_weight(bb_v).unwrap()))
    {
        log::trace!("BB idx: {bb_v}");
        let mut changed_regs = HashSet::<Reg>::new();

        let mut inst_worklist = HashSet::new();

        // All instructions are analyzed at least once
        inst_worklist.extend(0..bb.inst.len());

        while let Some(inst_idx) = inst_worklist.iter().next().copied() {
            inst_worklist.remove(&inst_idx);
            let inst = &bb.inst[inst_idx];
            log::trace!("Inst {inst_idx}: {inst:?}");

            // Checks if the given type is different from the given registers existing type.
            // If so, assigns it and updates worklists for instructions and BBs
            let mut update_changed = |r: &Reg, new_type: IsaType, map: &mut TypeMap<_>| {
                let refined = map.get(*r).refine(new_type).unwrap();
                if map.update(*r, refined) {
                    changed_regs.insert(*r);
                    // Re-analyze every instruction touching the register, not just the
                    // dfg neighbors of the current instruction: sibling consumers (e.g.
                    // a JumpTrigger sharing an operand with the current instruction)
                    // must also observe the new type.
                    bb.reg_uses(*r).chain(bb.reg_defs(*r)).for_each(|dep_i| {
                        log::trace!("To worklist: {}({:?})", dep_i, bb.inst[dep_i]);
                        inst_worklist.insert(dep_i);
                    })
                }
            };

            use MInst::*;
            match inst {
                BinaryAlu { rd, rs1, rs2, .. } => {
                    // Wrapping add/sub is signedness-agnostic: the machine
                    // accepts mixed-signedness operands, computing with the
                    // effective input type (unsigned unless both operands are
                    // signed). Unifying operands and result here is therefore
                    // only a preference: it runs in the soft phase (once all
                    // hard constraints have settled) and relaxes on a
                    // signedness conflict.
                    if enable_soft {
                        let t1 = type_map.get(*rs1);
                        let t2 = type_map.get(*rs2);
                        let td = type_map.get(rd.to_reg());

                        match t1.refine(t2).and_then(|t12| t12.refine(td)) {
                            Some(refined) => {
                                update_changed(rs1, refined, type_map);
                                update_changed(rs2, refined, type_map);
                                update_changed(&rd.to_reg(), refined, type_map);
                            }
                            None => {
                                // On a conflict the operands keep their own
                                // types, and the result carries the effective
                                // input type — once that is determined: it is
                                // unsigned as soon as either operand is known
                                // unsigned (an operand left undetermined
                                // resolves to unsigned, see Const), and signed
                                // only with both operands known signed. If the
                                // established result type disagrees, the
                                // result is re-tagged on its way out.
                                let known_unsigned = |t: IsaType| {
                                    t.get_known().is_some_and(|k| k.is_unsigned_int())
                                };
                                let eff_sign = if known_unsigned(t1) || known_unsigned(t2) {
                                    Some(false)
                                } else if t1.is_signed_int() && t2.is_signed_int() {
                                    Some(true)
                                } else {
                                    None
                                };
                                if let (Some(sign), true) = (eff_sign, t1.is_int()) {
                                    let eff = IsaType::new_known_int(t1.size_pow2(), sign);
                                    match td.refine(eff) {
                                        Some(refined) => {
                                            update_changed(&rd.to_reg(), refined, type_map)
                                        }
                                        None => push_demand(
                                            demands,
                                            bb_v,
                                            inst_idx,
                                            0,
                                            rd.to_reg(),
                                            td,
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
                DoubleAlu {
                    op: DoubleAluOp::SaddOverflow,
                    rdl,
                    rdh,
                    rs1,
                    rs2,
                } => {
                    // The machine's Add produces the signed-overflow flag only
                    // for signed operands, so the operands' signedness is
                    // hard: a conflicting operand must be re-tagged.
                    for (slot, r) in [rs1, rs2].into_iter().enumerate() {
                        let t = type_map.get(*r);
                        if t.is_int() {
                            let target = IsaType::new_known_int(t.size_pow2(), true);
                            match t.refine(target) {
                                Some(refined) => update_changed(r, refined, type_map),
                                None => push_demand(
                                    demands,
                                    bb_v,
                                    inst_idx,
                                    slot,
                                    *r,
                                    target,
                                ),
                            }
                        }
                    }

                    // The value result carries the (signed) effective type.
                    let td = type_map.get(rdl.to_reg());
                    if td.is_int() {
                        update_changed(
                            &rdl.to_reg(),
                            td.refine(IsaType::new_known_int(td.size_pow2(), true))
                                .unwrap(),
                            type_map,
                        );
                    }

                    // The flag (the machine's high output) is a u8 at runtime,
                    // but its signedness is left to its consumers: nothing
                    // emits the flag's compile-time type, and forcing it
                    // unsigned would conflict with signed consumers (the
                    // 0/1 value widens identically either way).
                    let _ = rdh;
                }
                IntCmp { rd, rs1, rs2, cc } => {
                    // The comparison result is a u8 boolean.
                    let td = type_map
                        .get(rd.to_reg())
                        .refine(IsaType::Known(scry_isa::Type::Uint(0)))
                        .unwrap();
                    update_changed(&rd.to_reg(), td, type_map);

                    let required_sign = match cc {
                        IntCC::UnsignedGreaterThan
                        | IntCC::UnsignedLessThan
                        | IntCC::UnsignedGreaterThanOrEqual
                        | IntCC::UnsignedLessThanOrEqual => Some(false),
                        IntCC::SignedGreaterThan
                        | IntCC::SignedLessThan
                        | IntCC::SignedGreaterThanOrEqual
                        | IntCC::SignedLessThanOrEqual => Some(true),
                        // Equality compares bits only.
                        IntCC::Equal | IntCC::NotEqual => None,
                    };
                    match required_sign {
                        // The machine compares according to the operands'
                        // runtime tags, so the signedness requirement is hard:
                        // a conflicting operand must be re-tagged.
                        Some(sign) => {
                            for (slot, r) in [rs1, rs2].into_iter().enumerate() {
                                let t = type_map.get(*r);
                                if t.is_int() {
                                    let target =
                                        IsaType::new_known_int(t.size_pow2(), sign);
                                    match t.refine(target) {
                                        Some(refined) => {
                                            update_changed(r, refined, type_map)
                                        }
                                        None => push_demand(
                                            demands,
                                            bb_v,
                                            inst_idx,
                                            slot,
                                            *r,
                                            target,
                                        ),
                                    }
                                }
                            }
                        }
                        // Unifying equality operands is only a preference
                        // (it types e.g. compared constants): soft phase,
                        // skipped on conflict.
                        None => {
                            if enable_soft {
                                let t1 = type_map.get(*rs1);
                                let t2 = type_map.get(*rs2);
                                if let Some(refined) = t1.refine(t2) {
                                    update_changed(rs1, refined, type_map);
                                    update_changed(rs2, refined, type_map);
                                }
                            }
                        }
                    }
                }
                Resize { var, rd, rs }
                    if *var == ResizeVariant::Uextend || *var == ResizeVariant::Sextend =>
                {
                    let sign = match var {
                        ResizeVariant::Uextend => false,
                        ResizeVariant::Sextend => true,
                        _ => unreachable!(),
                    };
                    let rs_t = type_map.get(*rs);
                    let rd_t = type_map.get(rd.to_reg());

                    // The machine picks the extension from the INPUT's runtime
                    // tag, so the input's signedness is hard: a conflicting
                    // input must be re-tagged.
                    if rs_t.is_int() {
                        let target = IsaType::new_known_int(rs_t.size_pow2(), sign);
                        match rs_t.refine(target) {
                            Some(refined) => update_changed(rs, refined, type_map),
                            None => push_demand(
                                demands,
                                bb_v,
                                inst_idx,
                                0,
                                *rs,
                                target,
                            ),
                        }
                    }
                    // The output's tag is whatever target type the cast is
                    // emitted with, so matching the extension's signedness is
                    // only a preference: soft phase, skipped on conflict.
                    if enable_soft && rd_t.is_int() {
                        if let Some(refined) =
                            rd_t.refine(IsaType::new_known_int(rd_t.size_pow2(), sign))
                        {
                            update_changed(&rd.to_reg(), refined, type_map);
                        }
                    }
                }
                Resize {
                    var: ResizeVariant::Reduce,
                    rd,
                    rs,
                } => {
                    let rs_t = type_map.get(*rs);
                    let rd_t = type_map.get(rd.to_reg());

                    if rs_t.is_same_signedness(&rd_t) {
                        // They are the same, just assign rd
                        update_changed(&rd.to_reg(), rd_t, type_map);
                    } else if rs_t.is_known() && rd_t.is_known() {
                        panic!("Incompatible type requirements: rs_t {rs_t:?}, rd_t {rd_t:?}");
                    } else if rs_t.is_known() || rd_t.is_known() {
                        // One is known, so use its signedness
                        let sign = rs_t.is_signed_int() || rd_t.is_signed_int();

                        if rs_t.is_int() {
                            update_changed(
                                rs,
                                rs_t.refine(IsaType::new_known_int(rs_t.size_pow2(), sign))
                                    .unwrap(),
                                type_map,
                            );
                        }
                        if rd_t.is_int() {
                            update_changed(
                                &rd.to_reg(),
                                rd_t.refine(IsaType::new_known_int(rd_t.size_pow2(), sign))
                                    .unwrap(),
                                type_map,
                            );
                        }
                    }
                }
                Const { rd, .. } => {
                    let rd_t = type_map.get(rd.to_reg());
                    update_changed(&rd.to_reg(), rd_t, type_map);
                }
                Echo { rds, rss, .. } => {
                    if rss.len() != 0 {
                        assert_eq!(rds.len(), rss.len());
                        for (slot, (rs, rd)) in rss.iter().zip(rds.iter()).enumerate() {
                            // Unify in both directions: a type may become known on
                            // the output side first (e.g. a block parameter whose
                            // echoed copy is refined by a later consumer).
                            match type_map.get(*rs).refine(type_map.get(rd.to_reg())) {
                                Some(merged) => {
                                    update_changed(rs, merged, type_map);
                                    update_changed(&rd.to_reg(), merged, type_map);
                                }
                                // Conflicting hard demands meet across this
                                // move (e.g. a parameter's ABI attribute vs a
                                // return's): keep moving the input as-is and
                                // re-tag it to the output's established type
                                // right after the move.
                                None => push_demand(
                                    demands,
                                    bb_v,
                                    inst_idx,
                                    slot,
                                    rd.to_reg(),
                                    type_map.get(rd.to_reg()),
                                ),
                            }
                        }
                    } else {
                        assert!(!rds.is_empty());
                        // This echo handles block parameters. Do nothing.
                    }
                }
                Reorder {
                    rd1, rd2, rs1, rs2, ..
                } => {
                    let merged1 = type_map
                        .get(*rs1)
                        .refine(type_map.get(rd1.to_reg()))
                        .unwrap();
                    update_changed(rs1, merged1, type_map);
                    update_changed(&rd1.to_reg(), merged1, type_map);
                    let merged2 = type_map
                        .get(*rs2)
                        .refine(type_map.get(rd2.to_reg()))
                        .unwrap();
                    update_changed(rs2, merged2, type_map);
                    update_changed(&rd2.to_reg(), merged2, type_map);
                }
                Duplicate { rd1, rd2, rs, .. } => {
                    let rs_t = type_map.get(*rs);
                    let rd1_t = type_map.get(rd1.to_reg());
                    let rd2_t = type_map.get(rd2.to_reg());

                    let merged_t = rs_t.refine(rd1_t).unwrap().refine(rd2_t).unwrap();

                    update_changed(rs, merged_t, type_map);
                    update_changed(&rd1.to_reg(), merged_t, type_map);
                    update_changed(&rd2.to_reg(), merged_t, type_map);
                }
                Load { rs, .. } => {
                    update_changed(
                        rs,
                        type_map
                            .get(*rs)
                            .refine(IsaType::Known(scry_isa::Type::Uint(2)))
                            .expect("Load source refine fail"),
                        type_map,
                    );

                    // We don't assign the type of the destination since we must get the type requirements from other instructions
                }
                CallArgs {
                    rets, args, sig, ..
                } => {
                    assert_eq!(rets.len(), sig.returns.len());
                    assert_eq!(args.len(), sig.params.len());

                    for (r, p) in rets.iter().map(|p| p.vreg).zip(sig.returns.iter()) {
                        let ty = abi_param_to_isatype(p);
                        update_changed(
                            &r.to_reg(),
                            type_map.get(r.to_reg()).refine(ty).unwrap(),
                            type_map,
                        );
                    }

                    for (slot, (r, p)) in args
                        .iter()
                        .map(|p| p.vreg)
                        .zip(sig.params.iter())
                        .enumerate()
                    {
                        let ty = abi_param_to_isatype(p);
                        // The callee assumes its parameters arrive with their
                        // ABI-annotated types, so the attribute is hard: a
                        // conflicting argument must be re-tagged.
                        match type_map.get(r).refine(ty) {
                            Some(refined) => update_changed(&r, refined, type_map),
                            None => push_demand(demands, bb_v, inst_idx, slot, r, ty),
                        }
                    }
                }
                Rets { rets } => {
                    assert_eq!(rets.len(), func_sig.returns.len());
                    for (slot, (r, p)) in rets
                        .iter()
                        .map(|p| p.vreg)
                        .zip(func_sig.returns.iter())
                        .enumerate()
                    {
                        let ty = abi_param_to_isatype(p);
                        // The ABI attribute is hard; a conflicting return
                        // value must be re-tagged.
                        match type_map.get(r).refine(ty) {
                            Some(refined) => update_changed(&r, refined, type_map),
                            None => push_demand(
                                demands,
                                bb_v,
                                inst_idx,
                                slot,
                                r,
                                ty,
                            ),
                        }
                    }
                }
                JumpTrigger { args, .. } => {
                    for arg_r in args.iter() {
                        let arg_ty = type_map.get(*arg_r);

                        let shared_ty = bb_dfg
                            .edges_sourced_in(bb_v)
                            .filter(|(_, deps)| deps.1.contains(arg_r))
                            .flat_map(|(_, deps)| {
                                deps.into_borrowed()
                                    .unwrap()
                                    .1
                                    .iter()
                                    .map(|d| type_map.get(*d))
                            })
                            .fold(arg_ty, |acc_ty, d_ty| acc_ty.refine(d_ty).unwrap());

                        once(*arg_r)
                            .chain(
                                bb_dfg
                                    .edges_sourced_in(bb_v)
                                    .filter_map(|(_, deps)| {
                                        if deps.1.contains(arg_r) {
                                            Some(deps.into_borrowed().unwrap().1.iter().cloned())
                                        } else {
                                            None
                                        }
                                    })
                                    .flatten(),
                            )
                            .for_each(|r| update_changed(&r, shared_ty, type_map));
                    }
                }
                _ => (),
            }
        }

        // Re-queue every other block that reasons about a register this block
        // updated. This block itself is skipped: its instruction worklist above
        // already ran.
        for changed_r in changed_regs.into_iter() {
            for other_v in reg_blocks.get(&changed_r).into_iter().flatten() {
                if *other_v != bb_v && bb_worklist.iter().all(|v| other_v != v) {
                    bb_worklist.push_back(*other_v);
                }
            }
        }
    }
}

fn resolve_instruction_types(
    cfg: &mut VCodeCFG<MInst>,
    vreg_type: impl Fn(Reg) -> Option<Type>,
    func_sig: &Signature,
    mut new_vreg: impl FnMut() -> Reg,
) {
    let type_map = type_analysis(cfg, vreg_type, func_sig, &mut new_vreg);

    log::trace!("Resolve instruction types");

    // Resolve type specific instructions into Scry instructions
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        for inst in bb.inst.iter_mut() {
            match inst {
                // A signedness mismatch between the result's effective input
                // type and its established type was already reconciled by a
                // cast demand during type analysis.
                MInst::BinaryAlu { op, rd, rs1, rs2 } => {
                    let binary_alu_to_alu2 = |b| match b {
                        BinaryAluOp::IntAddWrap => Alu2Variant::Add,
                        BinaryAluOp::IntSubWrap => Alu2Variant::Sub,
                    };

                    *inst = MInst::Alu2 {
                        var: binary_alu_to_alu2(*op),
                        out_var: Alu2OutputVariant::Low,
                        rds: vec![*rd],
                        rss: vec![*rs1, *rs2],
                        outs: vec![0],
                    };
                }
                MInst::DoubleAlu {
                    op,
                    rdl,
                    rdh,
                    rs1,
                    rs2,
                } => {
                    let var = match op {
                        DoubleAluOp::SaddOverflow => Alu2Variant::Add,
                    };

                    *inst = MInst::Alu2 {
                        var,
                        // Placeholder for the two-output form: the final
                        // output variant is chosen at emission from the
                        // output references (see emit.rs).
                        out_var: Alu2OutputVariant::FirstLow,
                        rds: vec![*rdl, *rdh],
                        rss: vec![*rs1, *rs2],
                        outs: vec![0, 0],
                    };
                }
                MInst::IntCmp { cc, rd, rs1, rs2 } => {
                    let var = match cc {
                        IntCC::Equal => AluVariant::Equal,
                        IntCC::SignedLessThan | IntCC::UnsignedLessThan => AluVariant::LessThan,
                        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => {
                            AluVariant::GreaterThan
                        }
                        _ => unimplemented!("icmp condition {:?} is not yet supported", cc),
                    };
                    *inst = MInst::Alu1 {
                        var,
                        rd: *rd,
                        rss: vec![*rs1, *rs2],
                        out: 0,
                    };
                }
                MInst::Resize { rd, rs, var } => {
                    let rd_t = type_map.get(rd.to_reg());
                    let rs_t = type_map.get(*rs);

                    assert_ne!(rd_t, rs_t);
                    // For extends, the machine picks the extension from the
                    // INPUT's runtime tag (guaranteed by type analysis); the
                    // output type only tags the result, so its signedness is
                    // free to differ from the input's.
                    assert!(*var != ResizeVariant::Reduce || (rd_t.is_int() && rs_t.is_int()));

                    *inst = MInst::Cast {
                        rd: *rd,
                        ty: rd_t,
                        rs: *rs,
                        out: 0,
                    };
                }
                MInst::Const { ty, rd, .. } => {
                    let rd_t = type_map.get(rd.to_reg());

                    assert!(rd_t.is_int());

                    // A constant whose consumers never constrain its signedness
                    // (e.g. one only stored to memory) defaults to unsigned:
                    // only the raw bits matter to such consumers.
                    *ty = match rd_t {
                        IsaType::Integer(size) => IsaType::new_known_int(size, false),
                        t => t,
                    };
                }
                MInst::Load { ty, rd, .. } | MInst::LoadStack { ty, rd, .. } => {
                    let rd_t = type_map.get(rd.to_reg());

                    if rd_t.is_known() {
                        *ty = rd_t;
                    } else {
                        panic!("Invalid load type: {rd_t:?}");
                    }
                }
                _ => (),
            }
        }

    }
}

/// Returns the blocks list of the indices of issuing instructions and their shared trigger
pub fn get_jmp_issues(block: &VCodeBB<MInst>) -> Option<(Vec<usize>, usize)> {
    let mut issues = Vec::new();
    let mut trigger = None;

    for idx in 0..block.inst.len() {
        match block.inst[idx] {
            MInst::BranchIssue { .. } | MInst::JumpIssue { .. } => {
                assert!(trigger.is_none(), "Jump issue after its trigger");
                issues.push(idx);
            }
            MInst::JumpTrigger { .. } => {
                assert!(trigger.is_none(), "Multiple jump triggers");
                trigger = Some(idx);
            }
            MInst::ImmJump { .. } => {
                assert!(issues.is_empty(), "Multiple jump issues");
                assert!(trigger.is_none(), "Multiple jump triggers");
                trigger = Some(idx);
                issues.push(idx);
            }
            _ => (),
        }
    }

    match trigger {
        Some(trigger) => {
            assert!(!issues.is_empty(), "Jump trigger without issue");
            Some((issues, trigger))
        }
        None => {
            assert!(issues.is_empty(), "Jump issue without trigger");
            None
        }
    }
}

/// The machine-dependent block-exit query for [`VCodeCFG::compute_layout`].
///
/// A block with several issues (a lowered `br_table`) names every one of its
/// successors, the default included, so it has no fall-through and imposes no
/// layout constraint. It reports the unconditional issue's target as a
/// [`BlockExit::Jump`], which the layout only treats as a preference.
fn block_exit(bb: &VCodeBB<MInst>) -> Option<BlockExit> {
    let issues = get_jmp_issues(bb)?.0;

    if issues.len() > 1 {
        return issues.iter().find_map(|issue| match bb.inst[*issue] {
            MInst::JumpIssue { dst, .. } => Some(BlockExit::Jump(Block::new(dst.index()))),
            _ => None,
        });
    }

    match bb.inst[issues[0]] {
        MInst::BranchIssue { dst, .. } => Some(BlockExit::Branch(Block::new(dst.index()))),
        MInst::JumpIssue { dst, .. } => Some(BlockExit::Jump(Block::new(dst.index()))),
        _ => None,
    }
}

/// Fixes the branch conditions to match the final block layout.
///
/// The layout is not stored anywhere: [`VCodeCFG::build_vcode`] deterministically
/// recomputes it (see [`VCodeCFG::compute_layout`]) when the blocks are emitted.
/// This pass computes the same layout to learn each branch's direction: lowering
/// emits every `BranchIssue` with backward-jump semantics (taken if the condition
/// is non-zero, see the ISA spec on `jmp`); if the branch target ends up *after*
/// the block, the emitted jump offset is positive and the ISA gives it
/// forward-jump semantics (taken if the condition is zero), so the condition must
/// be negated. Only conditions are rewritten — branch targets and block structure
/// are untouched — so the layout recomputed at emission is identical.
///
/// Each conditional issue is handled on its own, so the arms of a lowered
/// `br_table` may end up jumping in different directions.
fn fix_branch_conditions(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    let layout = cfg.compute_layout(block_exit);
    let position: HashMap<usize, usize> = layout.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    // Find every conditional issue's branch target.
    let mut forward_issues = HashMap::new(); // bb -> its issues that jump forward
    for (bb_v, bb) in cfg.graph.all_vertices_weighted() {
        let Some((issues, _)) = get_jmp_issues(bb) else {
            continue;
        };
        let forward: Vec<usize> = issues
            .into_iter()
            .filter(|issue| match bb.inst[*issue] {
                MInst::BranchIssue { dst, .. } => {
                    let target_bb_v = cfg
                        .vertex_of_block(Block::new(dst.index()))
                        .expect("Branch target is not a block");
                    position[&target_bb_v] > position[&bb_v]
                }
                // Unconditional issues have no condition to fix.
                _ => false,
            })
            .collect();
        if !forward.is_empty() {
            forward_issues.insert(bb_v, forward);
        }
    }

    // Fix branch conditions to match the layout direction. Later issues first, so
    // that inserting a negation does not shift the indices still to be handled.
    for (bb_v, issues) in forward_issues {
        let bb_mut = cfg.graph.vertex_weight_mut(bb_v).unwrap();
        for issue in issues.into_iter().rev() {
            // Forward jump: taken if the condition is zero, so negate it.
            match bb_mut.inst[issue] {
                MInst::BranchIssue {
                    dir,
                    dst,
                    cond,
                    link,
                    trig,
                } => {
                    assert!(!dir, "Lowering must emit backward-jump conditions");
                    let neg_cond_r = new_vreg();
                    let negation = MInst::UnaryAlu {
                        op: UnaryAluOp::LogNeg,
                        rd: WritableReg::from_reg(neg_cond_r),
                        rs: cond,
                    };
                    bb_mut.inst[issue] = MInst::BranchIssue {
                        dir: true,
                        dst,
                        cond: neg_cond_r,
                        link,
                        trig,
                    };
                    bb_mut.inst.insert(issue, negation);
                }
                _ => unreachable!(),
            }
        }
        // Backward (or self-loop) jumps are taken if the condition is non-zero,
        // which is what lowering emitted.
    }
}

/// Fills in each branch issue's trigger offset.
///
/// Must run last: any pass that adds or removes instructions after an issue
/// invalidates the offsets.
fn set_trigger_offsets(cfg: &mut VCodeCFG<MInst>) {
    log::debug!("set_trigger_offsets");
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        let Some((issues, trigger)) = get_jmp_issues(bb) else {
            continue;
        };
        for issue in issues {
            if issue == trigger {
                // An `ImmJump` is its own issue and trigger.
                continue;
            }
            let offset: usize = bb.inst[issue + 1..trigger]
                .iter()
                .map(MInst::emitted_length)
                .sum();
            assert!(
                offset <= ((1 << 6) - 1),
                "{offset} instructions between a branch issue and its trigger."
            );
            let offset = offset as u16;
            match &mut bb.inst[issue] {
                MInst::BranchIssue { trig, .. } | MInst::JumpIssue { trig, .. } => *trig = offset,
                _ => unreachable!(),
            }
        }
    }
    log::trace!("VCodeCFG: {cfg:?}");
}

impl TargetIsa for ScryBackend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        _regalloc_ctx: &mut regalloc2::Ctx,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        log::debug!("Beginning Scry compile");
        log::trace!("func: {func:?}");
        let (vcode, mut new_vregs) = self.compile_vcode(func, domtree, ctrl_plane)?;

        log::trace!("func signature: {:?}", vcode.abi.signature());

        let reg_type = |r: Reg| vcode.vreg_type_maybe(r.to_virtual_reg().unwrap().into());

        // Creates a new unique vreg
        let mut new_vreg = || {
            new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
        };

        // Machine-dependent part of edge-block promotion (see `from_vcode`):
        // replace a promoted block's single ImmJump with an explicit
        // JumpIssue/JumpTrigger pair passing the block's fresh parameter registers.
        let replace_jump = |inst: &MInst, args: Vec<Reg>, new_vreg: &mut dyn FnMut() -> Reg| {
            let dst = match inst {
                MInst::ImmJump { dst } => *dst,
                inst => unreachable!("Not a promotable jump block: {:?}", inst),
            };
            let link = new_vreg();
            vec![
                MInst::JumpIssue {
                    link: WritableReg::from_reg(link),
                    dst,
                    trig: 0,
                },
                MInst::JumpTrigger { link, args },
            ]
        };

        let mut cfg = VCodeCFG::from_vcode(&vcode, &mut new_vreg, replace_jump);

        log::trace!("VCodeCFG: {cfg:?}");

        prepare_block_params(&mut cfg, vcode.abi.signature(), &mut new_vreg);

        // Insert `ret` instruction as movable trigger, in every returning block
        cfg.graph.all_vertices_weighted_mut().for_each(|(_, bb)| {
            if let Some(rets_idx) = bb
                .inst
                .iter()
                .rposition(|i| matches!(i, MInst::Rets { .. }))
            {
                bb.inst.insert(rets_idx, MInst::Ret { trig: 0 });
            }
        });

        log::trace!("VCodeCFG: {cfg:?}");

        // The stack frame size: the end of the stack slot region (slot offsets
        // are assigned by `Callee::new`, which the stack access lowerings also
        // use), rounded up so the frame stays 16-aligned. The frame base is
        // 16-aligned by ABI guarantee, and keeping every frame's size aligned
        // preserves that guarantee for callees (a callee's frame base is this
        // frame's split point).
        let frame_bytes = func
            .sized_stack_slots
            .iter()
            .zip(vcode.abi.sized_stackslot_offsets().iter())
            .map(|((_, data), (_, offset))| offset + data.size)
            .max()
            .unwrap_or(0)
            .next_multiple_of(16);

        insert_frame_limits(&mut cfg, frame_bytes);

        resolve_instruction_types(&mut cfg, reg_type, vcode.abi.signature(), &mut new_vreg);
        insert_duplicates(&mut cfg, &mut new_vreg);
        expand_echoes(&mut cfg, &mut new_vreg);
        fix_orderings(&mut cfg, &mut new_vreg);
        fix_branch_conditions(&mut cfg, &mut new_vreg);
        insert_ref_distances(&mut cfg, &mut new_vreg);
        assert_queue_capacities(&mut cfg);
        set_trigger_offsets(&mut cfg);

        let sigs = SigSet::new::<abi::ScryMachineDeps>(func, &self.flags)?;
        let abi = Callee::<abi::ScryMachineDeps>::new(func, self, &self.isa_flags, &sigs)?;
        let mut builder = VCodeBuilder::<inst::MInst>::new(
            sigs,
            abi,
            EmitInfo::new(self.flags.clone(), self.isa_flags.clone()),
            BlockLoweringOrder::new(func, domtree, ctrl_plane),
            VCodeConstants::with_capacity(vcode.constants.len()),
            VCodeBuildDirection::Backward,
            2,
        );

        cfg.build_vcode(&mut builder, block_exit, |inst, label_idx_map| match inst {
            MInst::BranchIssue {
                link,
                dst,
                dir,
                cond,
                trig,
            } => MInst::BranchIssue {
                link,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
                dir,
                cond,
                trig,
            },
            MInst::JumpIssue { link, dst, trig } => MInst::JumpIssue {
                link,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
                trig,
            },
            i => i,
        });

        let vreg_alloc = VRegAllocator::with_capacity(vcode.num_vregs());
        let vcode2 = builder.build(vreg_alloc);

        log::trace!("VCode2: {vcode2:?}");

        let want_disasm = want_disasm || log::log_enabled!(log::Level::Debug);
        let emit_result = vcode2.emit(
            &regalloc2::Output::default(),
            want_disasm,
            &self.flags,
            ctrl_plane,
        )?;
        let value_labels_ranges = emit_result.value_labels_ranges;
        let buffer = emit_result.buffer;

        if let Some(disasm) = emit_result.disasm.as_ref() {
            log::debug!("disassembly:\n{disasm}");
        }

        Ok(CompiledCodeStencil {
            buffer,
            vcode: emit_result.disasm,
            value_labels_ranges,
            bb_starts: emit_result.bb_offsets,
            bb_edges: emit_result.bb_edges,
        })
    }

    fn name(&self) -> &'static str {
        "scry"
    }
    fn dynamic_vector_bytes(&self, _dynamic_ty: ir::Type) -> u32 {
        unimplemented!()
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    fn flags(&self) -> &shared_settings::Flags {
        &self.flags
    }

    fn isa_flags(&self) -> Vec<shared_settings::Value> {
        self.isa_flags.iter().collect()
    }

    fn isa_flags_hash_key(&self) -> IsaFlagsHashKey<'_> {
        IsaFlagsHashKey(self.isa_flags.hash_key())
    }

    #[cfg(feature = "unwind")]
    fn emit_unwind_info(
        &self,
        _result: &CompiledCode,
        _kind: crate::isa::unwind::UnwindInfoKind,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        Ok(None)
    }

    #[cfg(feature = "unwind")]
    fn create_systemv_cie(&self) -> Option<gimli::write::CommonInformationEntry> {
        None
    }

    fn text_section_builder(&self, num_funcs: usize) -> Box<dyn TextSectionBuilder> {
        Box::new(MachTextSectionBuilder::<inst::MInst>::new(num_funcs))
    }

    #[cfg(feature = "unwind")]
    fn map_regalloc_reg_to_dwarf(&self, _reg: Reg) -> Result<u16, systemv::RegisterMappingError> {
        unimplemented!()
    }

    fn function_alignment(&self) -> FunctionAlignment {
        inst::MInst::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        debug_assert_eq!(1 << 12, 0x1000);
        12
    }

    fn pretty_print_reg(&self, _reg: Reg, _size: u8) -> String {
        unimplemented!()
    }

    fn has_native_fma(&self) -> bool {
        false
    }

    fn has_round(&self) -> bool {
        false
    }

    fn has_blendv_lowering(&self, _: Type) -> bool {
        false
    }

    fn has_x86_pshufb_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmulhrsw_lowering(&self) -> bool {
        false
    }

    fn has_x86_pmaddubsw_lowering(&self) -> bool {
        false
    }

    fn default_argument_extension(&self) -> ir::ArgumentExtension {
        ir::ArgumentExtension::None
    }

    fn remove_constant_phis(&self) -> bool {
        false
    }
}

impl fmt::Display for ScryBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.name())
            .field("triple", &self.triple())
            .field("flags", &format!("{}", self.flags()))
            .finish()
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    match triple.architecture {
        Architecture::Scry(_) => {}
        _ => unreachable!(),
    }
    IsaBuilder {
        triple,
        setup: scry_settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    triple: Triple,
    shared_flags: Flags,
    builder: &shared_settings::Builder,
) -> CodegenResult<OwnedTargetIsa> {
    let isa_flags = scry_settings::Flags::new(&shared_flags, builder);

    let backend = ScryBackend::new_with_flags(triple, shared_flags, isa_flags);
    Ok(backend.wrapped())
}
