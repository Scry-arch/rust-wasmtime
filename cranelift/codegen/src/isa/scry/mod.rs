//! Scry Instruction Set Architecture.

use crate::MachLabel;
use crate::dominator_tree::DominatorTree;
use crate::ir::immediates::Imm64;
use crate::ir::{AbiParam, ArgumentExtension, Signature};
use crate::ir::{Function, Type};
use crate::isa::scry::abi::{
    arg_area_base, stack_area_size, stack_locals_base, stack_values_layout,
};
use crate::isa::scry::inst::{
    BinaryAluOp, DoubleAluOp, EmitInfo, IssueKind, MInst, QUEUE_CAPACITY, ResizeVariant,
    UnaryAluOp, delivery_group,
};
use crate::isa::scry::lower::isle::scaled_index;
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
use std::cmp::{max, min};
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

/// Makes every cross-block value use explicit as block parameters.
///
/// CLIF allows values to be implicitly used without being part of a block's
/// parameters as long as the defining block dominates the use.
/// This pass makes every such use explicit by having such values be passed
/// as block parameters instead.
fn make_live_ins_explicit(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("make_live_ins_explicit");
    let vertices: Vec<usize> = cfg.graph.all_vertices().collect();

    // Per block: the registers it defines (parameters and instruction defs)
    let mut bb_defs = HashMap::<usize, HashSet<Reg>>::new();
    // Per block: register uses that are not defined in the block (must be define in another block)
    let mut bb_undef_uses = HashMap::<usize, HashSet<Reg>>::new();
    for &v in &vertices {
        let bb = cfg.graph.vertex_weight(v).unwrap();
        let mut defs: HashSet<Reg> = bb.params.iter().copied().collect();
        let mut undef_uses = HashSet::new();
        for inst in &bb.inst {
            for r in inst.get_uses() {
                if !defs.contains(r) {
                    undef_uses.insert(*r);
                }
            }
            defs.extend(inst.get_defs());
        }
        bb_defs.insert(v, defs);
        bb_undef_uses.insert(v, undef_uses);
    }

    // Calculate live-ins
    let mut live_in = bb_undef_uses;
    let mut changed = true;
    while changed {
        changed = false;
        for &v in &vertices {
            let mut new = live_in[&v].clone();
            for (succ_v, _) in cfg.graph.edges_sourced_in(v) {
                new.extend(live_in[&succ_v].iter().filter(|r| !bb_defs[&v].contains(r)));
            }
            if new.len() != live_in[&v].len() {
                live_in.insert(v, new);
                changed = true;
            }
        }
    }

    // One fresh parameter per block and live-in register, in a fixed order per
    // block: the order the arguments are appended in below.
    let mut param_of: HashMap<(usize, Reg), Reg> = HashMap::new();
    for &v in &vertices {
        let bb = cfg.graph.vertex_weight_mut(v).unwrap();
        for &r in &live_in[&v] {
            let p = new_vreg();
            bb.params.push(p);
            param_of.insert((v, r), p);
        }
    }
    if param_of.is_empty() {
        return;
    }
    log::trace!("Live-in parameters: {param_of:?}");

    // The entry block has no predecessors to receive from: a live-in there is
    // a use without a definition.
    let entry = cfg.graph.root();
    assert!(
        live_in[&entry].is_empty(),
        "values used in the entry block without a definition: {:?}",
        live_in[&entry]
    );

    // Rename every use in a block to the block's own name for the register,
    // then pass the live-ins of every successor along the edge under the same
    // naming (`prepare_block_params` derives the jump trigger arguments from
    // `branch_params`).
    let name_in = |v: usize, r: Reg| param_of.get(&(v, r)).copied().unwrap_or(r);
    for &v in &vertices {
        let succs = cfg
            .graph
            .edges_sourced_in(v)
            .map(|(s, _)| (s, cfg.graph.vertex_weight(s).unwrap().vcode_bb))
            .collect::<Vec<_>>();
        let bb = cfg.graph.vertex_weight_mut(v).unwrap();
        for inst in &mut bb.inst {
            for r in inst.get_uses_mut() {
                *r = name_in(v, *r);
            }
        }
        for (succ, succ_block) in succs {
            let args = bb
                .branch_params
                .get_mut(&succ_block)
                .expect("successor without branch parameters");
            // A pre-existing branch argument may itself be a live-in (a value
            // defined elsewhere and forwarded by this block's branch, e.g. a
            // loop-invariant carried over a latch): rename it to this block's
            // own name, just like the instruction uses above.
            for r in args.iter_mut() {
                *r = name_in(v, *r);
            }
            args.extend(live_in[&succ].iter().map(|&r| name_in(v, r)));
        }
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
            // that will be discarded below. Only the first QUEUE_CAPACITY parameters
            // are delivered as operands; the rest are passed on the stack (defined
            // by stack loads, not by Args) and take no wire positions.
            let mut params: Vec<Option<Reg>> =
                vec![None; min(QUEUE_CAPACITY, func_sig.params.len())];
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
                let succ_r = match others.as_slice() {
                    // A successor's own parameter forwarded unchanged over a
                    // back-edge (loop-invariant): both sides of the edge use the
                    // same register, so the edge's register set collapses to just
                    // `out_r`.
                    [] => *out_r,
                    [r] => *r,
                    _ => panic!("Edge ({bb_v}->{succ_v}, {out_idx}) has more than two registers"),
                };
                reg_map.insert((out_idx, *out_r), succ_r);
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
            bb.inst.insert(
                2,
                MInst::Discard {
                    rss: chunk.to_vec(),
                },
            );
        }
    }

    log::trace!("VCodeCFG: {cfg:?}");
}

/// A stack access being resolved by [`stack_access_insts`].
enum StackAccess {
    /// Store the register's value.
    Store(Reg),
    /// Load into the register. The load's type is left for type resolution to
    /// fill in from the destination's uses.
    Load(WritableReg),
}

/// Resolves a stack access of the given scale at the given frame offset into
/// instructions, by reach:
/// 1. a stack access indexed by the accessed scale,
/// 2. a stack address of some other scale plus a plain memory access,
/// 3. full address materialization, frame base + offset (cf.
///    `emit_stack_addr`).
fn stack_access_insts(
    frame_offset: u32,
    scale_pow2: u16,
    access: StackAccess,
    new_vreg: &mut impl FnMut() -> Reg,
) -> Vec<MInst> {
    let scale = 1u32 << scale_pow2;
    if frame_offset % scale == 0 && frame_offset / scale <= 31 {
        let idx = (frame_offset / scale) as u16;
        vec![match access {
            StackAccess::Store(rs) => MInst::StoreStack { rs, idx },
            StackAccess::Load(rd) => MInst::LoadStack {
                ty: IsaType::Invalid,
                rd,
                idx,
            },
        }]
    } else {
        // Materialize the address, then access through it
        let mut insts = vec![];
        let addr = if let Some((sp, idx)) = scaled_index(frame_offset as i64) {
            let addr = new_vreg();
            insts.push(MInst::SAddr {
                rd: WritableReg::from_reg(addr),
                scale_pow2: sp,
                idx,
            });
            addr
        } else {
            let base = new_vreg();
            let off = new_vreg();
            let addr = new_vreg();
            insts.push(MInst::SAddr {
                rd: WritableReg::from_reg(base),
                scale_pow2: 0,
                idx: 0,
            });
            insts.push(MInst::Const {
                ty: IsaType::Invalid,
                rd: WritableReg::from_reg(off),
                imm: Imm64::new(frame_offset as i64),
            });
            insts.push(MInst::BinaryAlu {
                op: BinaryAluOp::IntAddWrap,
                rd: WritableReg::from_reg(addr),
                rs1: base,
                rs2: off,
            });
            addr
        };
        insts.push(match access {
            StackAccess::Store(rs) => MInst::Store { rd: addr, rs },
            StackAccess::Load(rd) => MInst::Load {
                ty: IsaType::Invalid,
                rd,
                rs: addr,
                out: 0,
            },
        });
        insts
    }
}

/// Resolves all stack-passed call arguments and return values (see the ABI's
/// calling convention and `MInst::StoreStackArg`).
///
/// On the callee side, this
/// 1. shifts the entry block's stack-argument loads up by the function's own
///    argument-area base (the loads are emitted with return-area-ignorant
///    offsets by `gen_load_stack`, which cannot see the signature), and
/// 2. splits every `Rets` down to its operand return values, storing the
///    stack-passed ones into the return area (offset 0 onward) before the
///    return issue.
///
/// On the caller side, for each call whose callee has a stack area, this
/// 1. reserves the callee's argument-and-return area on the shared frame
///    (which the call turns into the callee's private frame),
/// 2. resolves each `StoreStackArg` into a concrete stack access at frame
///    offset `private_size + arg_area_base + offset` (the shared frame starts
///    where the private frame ends, and stack indexing is not bounded by the
///    private frame),
/// 3. splits the `CallArgs` down to its operand return values and inserts the
///    echo receiving them -- the machine delivers them to the first
///    instruction executed after the call trigger, so the instruction there
///    must be their consumer,
/// 4. loads the used stack-passed return values from the returned area, and
/// 5. frees the area (the callee releases its own reservations before
///    returning, handing back exactly the area the caller reserved).
///
/// Must run before [`resolve_instruction_types`] so the instructions it
/// inserts get their types resolved, and before [`insert_ref_distances`].
fn lower_stack_args(
    cfg: &mut VCodeCFG<MInst>,
    func_sig: &Signature,
    private_size: u32,
    mut new_vreg: impl FnMut() -> Reg,
) {
    log::debug!("lower_stack_args");

    // Callee side: shift the entry block's argument loads by the function's
    // own argument-area base. Only stack-argument loads carry an `Integer`
    // type from lowering (see `gen_load_stack`), which identifies them.
    let own_arg_base = arg_area_base(func_sig);
    if own_arg_base > 0 {
        for inst in cfg.graph.root_weight_mut().inst.iter_mut() {
            if let MInst::LoadStack {
                ty: ty @ IsaType::Integer(_),
                idx,
                ..
            } = inst
            {
                // The base is 16-aligned, so scale-aligned for any scale
                let shifted = (own_arg_base >> ty.size_pow2()) + *idx as u32;
                assert!(
                    shifted <= 31,
                    "far stack arguments (beyond the stack-load index range) not yet supported"
                );
                *idx = shifted as u16;
            }
        }
    }

    let (own_ret_offsets, _) = stack_values_layout(&func_sig.returns);

    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        // Callee side: store the stack-passed return values into the return
        // area, before the return issue, and keep only the operand return
        // values on the Rets trigger.
        if let Some(rets_idx) = bb.inst.iter().position(|i| matches!(i, MInst::Rets { .. })) {
            let MInst::Rets { rets } = &mut bb.inst[rets_idx] else {
                unreachable!()
            };
            if rets.len() > QUEUE_CAPACITY {
                let stack_rets = rets.split_off(QUEUE_CAPACITY);
                let ret_idx = rets_idx - 1;
                assert!(
                    matches!(bb.inst[ret_idx], MInst::Ret { .. }),
                    "Rets without a directly preceding Ret"
                );
                let mut stores = vec![];
                for (k, pair) in stack_rets.iter().enumerate() {
                    let ty = func_sig.returns[QUEUE_CAPACITY + k].value_type;
                    stores.extend(stack_access_insts(
                        own_ret_offsets[k],
                        ty.bytes().ilog2() as u16,
                        StackAccess::Store(pair.vreg),
                        &mut new_vreg,
                    ));
                }
                bb.inst.splice(ret_idx..ret_idx, stores);
            }
        }

        // Caller side: handle every call whose callee has a stack area.
        let mut idx = 0;
        while idx < bb.inst.len() {
            let Some(trigger_idx) = bb.inst[idx..]
                .iter()
                .position(|i| matches!(i, MInst::CallArgs { .. }))
                .map(|p| p + idx)
            else {
                break;
            };
            idx = trigger_idx + 1;

            let MInst::CallArgs { rets, sig, .. } = &mut bb.inst[trigger_idx] else {
                unreachable!()
            };
            let callee_area = stack_area_size(sig);
            if callee_area == 0 {
                continue;
            }
            let callee_arg_base = arg_area_base(sig);
            let (callee_ret_offsets, _) = stack_values_layout(&sig.returns);
            let ret_types: Vec<Type> = sig.returns.iter().map(|r| r.value_type).collect();
            let num_stack_args = sig.params.len() - min(QUEUE_CAPACITY, sig.params.len());

            // Keep only the operand return values on the trigger; the rest
            // are loaded from the return area below.
            let stack_rets = if rets.len() > QUEUE_CAPACITY {
                rets.drain(QUEUE_CAPACITY..).collect::<Vec<_>>()
            } else {
                vec![]
            };
            let operand_rets = rets.iter().map(|r| r.vreg.to_reg()).collect::<Vec<_>>();

            // The argument stores of the call site are emitted contiguously,
            // directly before the call issue (there are none if only the
            // return values overflow into the stack area).
            let call_idx = trigger_idx - 1;
            assert!(
                matches!(bb.inst[call_idx], MInst::Call { .. }),
                "CallArgs without a directly preceding Call"
            );
            let first_store = call_idx - num_stack_args;
            assert!(
                bb.inst[first_store..call_idx]
                    .iter()
                    .all(|i| matches!(i, MInst::StoreStackArg { .. })),
                "stack-argument stores are not contiguous before their call"
            );

            // The reserve/free instructions can only encode power-of-two
            // amounts, so one instruction per set bit is emitted.
            let adjust_chunks = |reserve: bool| {
                (0..16u16)
                    .filter(move |k| callee_area & (1 << k) != 0)
                    .map(move |k| MInst::StackAdjust {
                        reserve,
                        private: false,
                        amount_pow2: k,
                    })
            };

            // Reserve the area, then store the stack-passed arguments into
            // its argument part.
            let mut resolved: Vec<MInst> = adjust_chunks(true).collect();
            for inst in &bb.inst[first_store..call_idx] {
                let MInst::StoreStackArg {
                    rs,
                    offset,
                    scale_pow2,
                } = inst
                else {
                    unreachable!()
                };
                resolved.extend(stack_access_insts(
                    private_size + callee_arg_base + *offset as u32,
                    *scale_pow2,
                    StackAccess::Store(*rs),
                    &mut new_vreg,
                ));
            }
            resolved.push(bb.inst[call_idx].clone());
            resolved.push(bb.inst[trigger_idx].clone());
            let resolved_len = resolved.len();
            bb.inst.splice(first_store..trigger_idx + 1, resolved);
            let mut pos = first_store + resolved_len;

            // The operand return values arrive at the first instruction
            // executed after the call trigger, which must therefore consume
            // them - insert their receiving echo.
            if !operand_rets.is_empty() {
                let rds = operand_rets
                    .iter()
                    .map(|r| {
                        let fresh = new_vreg();
                        replace_all_uses(bb, *r, fresh);
                        WritableReg::from_reg(fresh)
                    })
                    .collect();
                bb.inst.insert(
                    pos,
                    MInst::Echo {
                        rds,
                        rss: operand_rets,
                    },
                );
                pos += 1;
            }

            // Load the used stack-passed return values from the returned
            // area; unused ones are simply left there.
            for (k, pair) in stack_rets.iter().enumerate() {
                let rd = pair.vreg;
                if bb.reg_uses(rd.to_reg()).next().is_none() {
                    continue;
                }
                let ty = ret_types[QUEUE_CAPACITY + k];
                let loads = stack_access_insts(
                    private_size + callee_ret_offsets[k],
                    ty.bytes().ilog2() as u16,
                    StackAccess::Load(rd),
                    &mut new_vreg,
                );
                for inst in loads {
                    bb.inst.insert(pos, inst);
                    pos += 1;
                }
            }

            // Free the returned area
            for inst in adjust_chunks(false) {
                bb.inst.insert(pos, inst);
                pos += 1;
            }
            idx = pos;
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
                private: true,
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
                        private: false,
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
                                    Alu2OutputVariant::LowOnly
                                } else {
                                    Alu2OutputVariant::HighOnly
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
                                ref_dist - use_idx.1 - (inst.reference_length() as u16) + use_idx.2,
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
                        | MInst::JumpOffset { rd, .. }
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
                    }
                    // An output reference beyond its field's range (5 bits,
                    // except EchoLong's 10) is likewise bridged: the echo's
                    // longer reference carries the distance, chaining further
                    // echoes if even that is exceeded.
                    .or_else(|| {
                        let max_ref: u16 = match inst {
                            MInst::EchoLong { .. } => (1 << 10) - 1,
                            _ => (1 << 5) - 1,
                        };
                        inst.get_defs()
                            .enumerate()
                            .find_map(|(i, d)| (ref_dists[&i] > max_ref).then_some(d))
                    });
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
                        | MInst::UnaryAlu { out, .. }
                        | MInst::Pick { out, .. }
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
                        (0..uses.len() - 1).find(|i| {
                            let pos = |r: &Reg| {
                                *def_pos.get(r).unwrap_or_else(|| {
                                    panic!(
                                        "use of {r:?} not defined in its block (missed by \n                                         make_live_ins_explicit)"
                                    )
                                })
                            };
                            pos(&uses[i + 1]) < pos(&uses[*i])
                        });

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

pub(crate) fn type_to_isatype(t: Type) -> IsaType {
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

    // Map function parameters to entry block registers (the zip truncates to
    // the operand-delivered parameters; stack-passed ones are typed below)
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

    // Map the stack-passed parameters (see the ABI's calling convention) to
    // the entry block's argument-area stack loads. A load's index scaled by
    // its type identifies the argument-area offset (past the argument-area
    // base, `lower_stack_args` having shifted the indices) and thereby the
    // parameter (unused stack parameters simply have no load). Only argument
    // loads carry an `Integer` type from lowering (see `gen_load_stack`);
    // stack accesses fused from `stack_addr` start as `Invalid` instead.
    let (stack_offsets, _) = stack_values_layout(&func_sig.params);
    let own_arg_base = arg_area_base(func_sig);
    for inst in cfg.graph.root_weight().inst.iter() {
        if let MInst::LoadStack {
            ty: ty @ IsaType::Integer(_),
            rd,
            idx,
        } = inst
        {
            let offset = ((*idx as u32) << ty.size_pow2()) - own_arg_base;
            let param_idx = stack_offsets
                .iter()
                .position(|off| *off == offset)
                .expect("entry stack load does not match any stack parameter's offset");
            let ty = abi_param_to_isatype(&func_sig.params[QUEUE_CAPACITY + param_idx]);
            let refined = type_map.get(rd.to_reg()).refine(ty).unwrap();
            type_map.update(rd.to_reg(), refined);
        }
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
        type_analysis_phase(
            cfg,
            func_sig,
            &reg_blocks,
            &mut type_map,
            &mut demands,
            false,
        );
        type_analysis_phase(
            cfg,
            func_sig,
            &reg_blocks,
            &mut type_map,
            &mut demands,
            true,
        );

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
                        .expect("Demanded def slot out of range") = WritableReg::from_reg(fresh);
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
                                let known_unsigned =
                                    |t: IsaType| t.get_known().is_some_and(|k| k.is_unsigned_int());
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
                                        None => {
                                            push_demand(demands, bb_v, inst_idx, 0, rd.to_reg(), td)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                DoubleAlu {
                    op,
                    rdl,
                    rdh,
                    rs1,
                    rs2,
                } => {
                    use crate::isa::scry::inst::DoubleAluOp::*;
                    // Each of these operations is signedness-steered: the
                    // machine picks the operation (and flag meaning) from the
                    // operands' runtime types, so a pinned operand's
                    // signedness is hard: a conflicting operand must be
                    // re-tagged. The shifts read each operand's own tag
                    // (no effective type): the value's picks arithmetic vs
                    // logical, the amount's the direction (negative signed
                    // amounts shift right), so a left shift pins nothing.
                    let (pin1, pin2): (Option<bool>, Option<bool>) = match op {
                        SaddOverflow | SsubOverflow | SmulHi | SdivRem => (Some(true), Some(true)),
                        UaddOverflow | UsubOverflow | UmulHi | UdivRem => {
                            (Some(false), Some(false))
                        }
                        Shl => (None, None),
                        Ushr => (Some(false), Some(true)),
                        Sshr => (Some(true), Some(true)),
                    };
                    let signed = pin1.unwrap_or(false);
                    for (slot, (r, pin)) in [(rs1, pin1), (rs2, pin2)].into_iter().enumerate() {
                        let Some(signed) = pin else { continue };
                        let t = type_map.get(*r);
                        if t.is_int() {
                            let target = IsaType::new_known_int(t.size_pow2(), signed);
                            match t.refine(target) {
                                Some(refined) => update_changed(r, refined, type_map),
                                None => push_demand(demands, bb_v, inst_idx, slot, *r, target),
                            }
                        }
                    }

                    // A helper pinning an output to a known signedness.
                    let mut pin_output =
                        |rd: &WritableReg, sign: bool, type_map: &mut TypeMap<F>| {
                            let td = type_map.get(rd.to_reg());
                            if td.is_int() {
                                update_changed(
                                    &rd.to_reg(),
                                    td.refine(IsaType::new_known_int(td.size_pow2(), sign))
                                        .unwrap(),
                                    type_map,
                                );
                            }
                        };

                    match op {
                        // The value result carries the effective type; the
                        // flag (the machine's high output) is a u8 at
                        // runtime, but its signedness is left to its
                        // consumers: nothing emits the flag's compile-time
                        // type, and forcing it unsigned would conflict with
                        // signed consumers (the 0/1 value widens identically
                        // either way).
                        SaddOverflow | UaddOverflow | UsubOverflow | SsubOverflow => {
                            pin_output(rdl, signed, type_map);
                            let _ = rdh;
                        }
                        // The used high half carries the effective type; the
                        // low output is dead (dropped later) and needs no
                        // type.
                        UmulHi | SmulHi => {
                            pin_output(rdh, signed, type_map);
                        }
                        // The quotient carries the effective type; the
                        // machine types the remainder unsigned (Euclidean
                        // remainders are non-negative).
                        UdivRem | SdivRem => {
                            pin_output(rdl, signed, type_map);
                            pin_output(rdh, false, type_map);
                        }
                        // The result carries the value operand's tag (the
                        // shift-out high output is dead).
                        Ushr | Sshr => {
                            pin_output(rdl, signed, type_map);
                        }
                        // The result carries the value's tag, whatever it
                        // is: unify the two (hard), re-tagging the value on
                        // a conflict so the result keeps its established type.
                        Shl => {
                            let t1 = type_map.get(*rs1);
                            let td = type_map.get(rdl.to_reg());
                            if t1.is_int() && td.is_int() {
                                match t1.refine(td) {
                                    Some(refined) => {
                                        update_changed(rs1, refined, type_map);
                                        update_changed(&rdl.to_reg(), refined, type_map);
                                    }
                                    None => push_demand(
                                        demands,
                                        bb_v,
                                        inst_idx,
                                        0,
                                        *rs1,
                                        IsaType::new_known_int(t1.size_pow2(), td.is_signed_int()),
                                    ),
                                }
                            }
                        }
                    }
                }
                UnaryAlu {
                    op: UnaryAluOp::LogNeg,
                    rd,
                    ..
                } => {
                    // The negation result is a u8 boolean; the input is
                    // compared against 0 by bits, so its type is
                    // unconstrained.
                    let td = type_map
                        .get(rd.to_reg())
                        .refine(IsaType::Known(scry_isa::Type::Uint(0)))
                        .unwrap();
                    update_changed(&rd.to_reg(), td, type_map);
                }
                UnaryAlu {
                    op: UnaryAluOp::BitNeg,
                    rd,
                    rs,
                    ..
                } => {
                    // The result has the input's type (the machine's xor
                    // computes with the effective input type, which for the
                    // single-operand form is simply the input's). Unifying is
                    // only a preference: soft phase, skipped on conflict.
                    if enable_soft {
                        let t = type_map.get(*rs);
                        let td = type_map.get(rd.to_reg());
                        if let Some(refined) = t.refine(td) {
                            update_changed(rs, refined, type_map);
                            update_changed(&rd.to_reg(), refined, type_map);
                        }
                    }
                }
                Pick {
                    rd,
                    if_zero,
                    if_nonzero,
                    ..
                } => {
                    // The machine forwards the chosen value with its own tag,
                    // so the two values and the result should agree; the
                    // condition is tested against 0 by logical value and is
                    // unconstrained. Unifying is only a preference: soft
                    // phase, skipped on conflict.
                    if enable_soft {
                        let t1 = type_map.get(*if_zero);
                        let t2 = type_map.get(*if_nonzero);
                        let td = type_map.get(rd.to_reg());
                        if let Some(refined) = t1.refine(t2).and_then(|t12| t12.refine(td)) {
                            update_changed(if_zero, refined, type_map);
                            update_changed(if_nonzero, refined, type_map);
                            update_changed(&rd.to_reg(), refined, type_map);
                        }
                    }
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
                                    let target = IsaType::new_known_int(t.size_pow2(), sign);
                                    match t.refine(target) {
                                        Some(refined) => update_changed(r, refined, type_map),
                                        None => {
                                            push_demand(demands, bb_v, inst_idx, slot, *r, target)
                                        }
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
                            None => push_demand(demands, bb_v, inst_idx, 0, *rs, target),
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
                SAddr { rd, .. } => {
                    // The machine types a stack address as a full-width
                    // unsigned integer of its pointer size, regardless of any
                    // consumer (the encoding has no type field). This also
                    // grounds the types of address arithmetic on registers
                    // created after lowering, which have no CLIF fallback
                    // type (see `lower_stack_args`).
                    update_changed(
                        &rd.to_reg(),
                        type_map
                            .get(rd.to_reg())
                            .refine(IsaType::Known(scry_isa::Type::Uint(2)))
                            .expect("Stack address refine fail"),
                        type_map,
                    );
                }
                CallArgs {
                    rets, args, sig, ..
                } => {
                    // Only the first QUEUE_CAPACITY values are passed as
                    // trigger operands; the rest go on the stack (the zips
                    // below truncate to the operand values).
                    assert_eq!(rets.len(), min(QUEUE_CAPACITY, sig.returns.len()));
                    assert_eq!(args.len(), min(QUEUE_CAPACITY, sig.params.len()));

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
                    // Only the first QUEUE_CAPACITY return values are passed
                    // as trigger operands; the rest go on the stack (the zip
                    // below truncates to the operand values).
                    assert_eq!(rets.len(), min(QUEUE_CAPACITY, func_sig.returns.len()));
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
                            None => push_demand(demands, bb_v, inst_idx, slot, r, ty),
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
                    // Wrapping add/sub/mul are the low output of the
                    // two-output machine operations; the bitwise operations
                    // are single-output.
                    let alu2_var = match op {
                        BinaryAluOp::IntAddWrap => Some(Alu2Variant::Add),
                        BinaryAluOp::IntSubWrap => Some(Alu2Variant::Sub),
                        BinaryAluOp::IntMulWrap => Some(Alu2Variant::Multiply),
                        BinaryAluOp::BitAnd | BinaryAluOp::BitOr | BinaryAluOp::BitXor => None,
                    };

                    *inst = match alu2_var {
                        Some(var) => MInst::Alu2 {
                            var,
                            out_var: Alu2OutputVariant::LowOnly,
                            rds: vec![*rd],
                            rss: vec![*rs1, *rs2],
                            outs: vec![0],
                        },
                        None => MInst::Alu1 {
                            var: match op {
                                BinaryAluOp::BitAnd => AluVariant::BitAnd,
                                BinaryAluOp::BitOr => AluVariant::BitOr,
                                BinaryAluOp::BitXor => AluVariant::BitXor,
                                _ => unreachable!(),
                            },
                            rd: *rd,
                            rss: vec![*rs1, *rs2],
                            out: 0,
                        },
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
                        DoubleAluOp::SaddOverflow | DoubleAluOp::UaddOverflow => Alu2Variant::Add,
                        DoubleAluOp::UsubOverflow | DoubleAluOp::SsubOverflow => Alu2Variant::Sub,
                        DoubleAluOp::UmulHi | DoubleAluOp::SmulHi => Alu2Variant::Multiply,
                        DoubleAluOp::UdivRem | DoubleAluOp::SdivRem => Alu2Variant::Division,
                        DoubleAluOp::Shl | DoubleAluOp::Ushr | DoubleAluOp::Sshr => {
                            Alu2Variant::Shift
                        }
                    };

                    *inst = MInst::Alu2 {
                        var,
                        // Placeholder for the two-output form: the final
                        // output variant is chosen at emission from the
                        // output references (see emit.rs).
                        out_var: Alu2OutputVariant::LowFirst,
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
                        // The remaining conditions are synthesized during
                        // lowering as LogicalNot of the opposite comparison.
                        _ => unreachable!("icmp condition {:?} reached resolution", cc),
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

                    *ty = match rd_t {
                        t if t.is_known() => t,
                        // A load whose destination no consumer constrains
                        // beyond its scale, e.g. a value that is only stored
                        // back to memory (a stack-passed call argument).
                        // Memory is untagged, so the signedness is a
                        // don't-care; default to unsigned.
                        IsaType::Integer(k) => IsaType::new_known_int(k, false),
                        t => panic!("Invalid load type: {t:?}"),
                    };
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
            MInst::JumpIssue { .. } => {
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
        return issues.iter().find_map(|issue| match &bb.inst[*issue] {
            MInst::JumpIssue { dst, kind, .. } if !kind.is_conditional() => {
                Some(BlockExit::Jump(Block::new(dst.index())))
            }
            _ => None,
        });
    }

    match &bb.inst[issues[0]] {
        MInst::JumpIssue { dst, kind, .. } if kind.is_conditional() => {
            Some(BlockExit::Branch(Block::new(dst.index())))
        }
        MInst::JumpIssue { dst, .. } => Some(BlockExit::Jump(Block::new(dst.index()))),
        _ => None,
    }
}

/// Fixes the branch conditions to match the final block layout.
///
/// The layout is not stored anywhere: [`VCodeCFG::build_vcode`] deterministically
/// recomputes it (see [`VCodeCFG::compute_layout`]) when the blocks are emitted.
/// This pass computes the same layout to learn each branch's direction: lowering
/// emits every branch with backward-jump semantics (taken if the condition
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
    let issuing_blocks = cfg
        .graph
        .all_vertices_weighted()
        .filter_map(|(bb_v, bb)| get_jmp_issues(bb).map(|(issues, _)| (bb_v, bb, issues)));
    for (bb_v, bb, issues) in issuing_blocks {
        let forward: Vec<usize> = issues
            .into_iter()
            .filter(|issue| match bb.inst[*issue] {
                MInst::JumpIssue {
                    dst,
                    kind: IssueKind::Branch { .. },
                    ..
                } => {
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
            let negation = match &mut bb_mut.inst[issue] {
                MInst::JumpIssue {
                    kind: IssueKind::Branch { dir, cond },
                    ..
                } => {
                    assert!(!*dir, "Lowering must emit backward-jump conditions");
                    let neg_cond_r = new_vreg();
                    let negation = MInst::UnaryAlu {
                        op: UnaryAluOp::LogNeg,
                        rd: WritableReg::from_reg(neg_cond_r),
                        rs: *cond,
                        out: 0,
                    };
                    *dir = true;
                    *cond = neg_cond_r;
                    negation
                }
                _ => unreachable!(),
            };
            bb_mut.inst.insert(issue, negation);
        }
        // Backward (or self-loop) jumps are taken if the condition is non-zero,
        // which is what lowering emitted.
    }
}

/// Turns every jump issue whose target the short `jmp`'s 7-bit immediate cannot
/// reach into a far issue (the ISA's two-operand `jmp`, which takes its target
/// offset from an operand materialized by a preceding `JumpOffset` chain), and
/// widens the offset of any far issue that has outgrown its width.
///
/// Distances are measured on the final block layout (the one
/// [`VCodeCFG::build_vcode`] recomputes) with the exact emitted lengths, so this
/// must run after [`insert_ref_distances`] and [`fix_branch_conditions`]. As
/// widening lengthens the code, which may push other jumps out of range, and
/// the inserted instructions need reference distances of their own, the caller
/// alternates [`insert_ref_distances`] and this pass until nothing changes; the
/// fixpoint exists because a jump is only ever widened, never shortened.
///
/// A far branch keeps its condition, which [`fix_branch_conditions`] already
/// gave the direction's sense (taken on zero forward, on non-zero backward); a
/// far unconditional jump gets a constant condition of that sense.
///
/// Returns whether anything was changed.
fn widen_far_jumps(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) -> bool {
    log::debug!("widen_far_jumps");

    // The instruction address of every block's start in the layout.
    let mut block_addr = HashMap::<usize, i64>::new();
    let mut addr = 0i64;
    for bb_v in cfg.compute_layout(block_exit) {
        block_addr.insert(bb_v, addr);
        let bb = cfg.graph.vertex_weight(bb_v).unwrap();
        addr += bb
            .inst
            .iter()
            .map(|i| i.emitted_length() as i64)
            .sum::<i64>();
    }
    let target_addr = |cfg: &VCodeCFG<MInst>, dst: MachLabel| {
        let v = cfg
            .vertex_of_block(Block::new(dst.index()))
            .expect("Jump target is not a block");
        block_addr[&v]
    };
    // The reaches of the short and far forms (independent of the trigger offset
    // and chain placement, which `jump_offset` accounts for).
    let short_range = inst::LabelUse::JmpLoc7.offset_range();
    let far_range = |size_pow2: u16| {
        inst::LabelUse::JmpFar {
            size_pow2,
            gap: 0,
            trig: 0,
        }
        .offset_range()
    };

    let mut changed = false;
    let issuing_blocks: Vec<_> = cfg
        .graph
        .all_vertices()
        .filter_map(|bb_v| {
            get_jmp_issues(cfg.graph.vertex_weight(bb_v).unwrap())
                .map(|(issues, trigger)| (bb_v, issues, trigger))
        })
        .collect();
    for (bb_v, issues, trigger) in issuing_blocks {
        // Decide on every issue from the same snapshot of the addresses: the
        // edits only lengthen the code, and any jump they push out of range
        // is caught by the next round.
        let bb = cfg.graph.vertex_weight(bb_v).unwrap();
        let mut inst_addr = Vec::with_capacity(bb.inst.len() + 1);
        inst_addr.push(block_addr[&bb_v]);
        for inst in bb.inst.iter() {
            inst_addr.push(inst_addr.last().unwrap() + inst.emitted_length() as i64);
        }
        let offset_of = |cfg: &VCodeCFG<MInst>, issue: usize, dst: MachLabel| {
            let jump = inst_addr[issue];
            let trig = inst_addr[trigger] - inst_addr[issue + 1];
            inst::LabelUse::jump_offset(jump, trig, target_addr(cfg, dst))
        };
        // The width for an offset, leaving 4 bits of slack for the widening
        // itself.
        let width_for = |offset: i64| {
            let slack = far_range(1);
            if (slack.start() >> 4..=slack.end() >> 4).contains(&offset) {
                1u16
            } else {
                2u16
            }
        };

        enum Edit {
            /// Turn the short issue into a far one of the given width.
            Widen { offset: i64, size_pow2: u16 },
            /// Widen the far issue's chain to 32 bits.
            Grow,
        }
        let mut edits = Vec::new();
        for issue in issues {
            match &bb.inst[issue] {
                MInst::JumpIssue {
                    dst,
                    kind: IssueKind::Jump | IssueKind::Branch { .. },
                    ..
                } => {
                    // No offset means a fall-through, which is emitted as a
                    // NoOp and never far.
                    if let Some(offset) = offset_of(cfg, issue, *dst) {
                        if !short_range.contains(&offset) {
                            edits.push((
                                issue,
                                Edit::Widen {
                                    offset,
                                    size_pow2: width_for(offset),
                                },
                            ));
                        }
                    }
                }
                MInst::JumpIssue {
                    dst,
                    kind: IssueKind::Far { .. },
                    ..
                } => {
                    let off = offset_of(cfg, issue, *dst).expect("Far jump to a fall-through");
                    // Its chain is the nearest preceding one (see set_trigger_offsets).
                    let MInst::JumpOffset { size_pow2, .. } = bb.inst[bb.inst[..issue]
                        .iter()
                        .rposition(|i| matches!(i, MInst::JumpOffset { .. }))
                        .expect("Far jump issue without a preceding JumpOffset")]
                    else {
                        unreachable!()
                    };
                    if !far_range(size_pow2).contains(&off) {
                        assert!(size_pow2 < 2, "Far jump offset {off} exceeds 32 bits");
                        edits.push((issue, Edit::Grow));
                    }
                }
                _ => unreachable!(),
            }
        }

        // Later issues first, so that inserting does not shift the indices
        // still to be edited.
        let bb = cfg.graph.vertex_weight_mut(bb_v).unwrap();
        for (issue, edit) in edits.into_iter().rev() {
            changed = true;
            match edit {
                Edit::Grow => {
                    let chain = bb.inst[..issue]
                        .iter()
                        .rposition(|i| matches!(i, MInst::JumpOffset { .. }))
                        .unwrap();
                    match &mut bb.inst[chain] {
                        MInst::JumpOffset { size_pow2, .. } => *size_pow2 = 2,
                        _ => unreachable!(),
                    }
                }
                Edit::Widen { offset, size_pow2 } => {
                    let dir = offset > 0;
                    let offset_r = new_vreg();
                    let MInst::JumpIssue {
                        link,
                        dst,
                        kind,
                        trig,
                    } = &bb.inst[issue]
                    else {
                        unreachable!()
                    };
                    let (link, dst, trig) = (*link, *dst, *trig);
                    let (kind, mut inserts) = match kind.clone() {
                        IssueKind::Jump => {
                            // The condition of the direction's sense: zero
                            // takes a forward jump, non-zero a backward one.
                            let cond = new_vreg();
                            (
                                IssueKind::Far {
                                    conditional: false,
                                    dir,
                                    cond,
                                    offset: offset_r,
                                },
                                vec![MInst::Const {
                                    rd: WritableReg::from_reg(cond),
                                    ty: IsaType::Known(scry_isa::Type::Uint(0)),
                                    imm: Imm64::new(if dir { 0 } else { 1 }),
                                }],
                            )
                        }
                        IssueKind::Branch {
                            dir: branch_dir,
                            cond,
                        } => {
                            assert_eq!(branch_dir, dir, "Branch condition sense mismatch");
                            (
                                IssueKind::Far {
                                    conditional: true,
                                    dir,
                                    cond,
                                    offset: offset_r,
                                },
                                vec![],
                            )
                        }
                        IssueKind::Far { .. } => unreachable!(),
                    };
                    let far = MInst::JumpIssue {
                        link,
                        dst,
                        kind,
                        trig,
                    };
                    // The chain comes last, right before its issue, so that the
                    // condition (produced earlier) arrives before the offset.
                    inserts.push(MInst::JumpOffset {
                        rd: WritableReg::from_reg(offset_r),
                        dst,
                        size_pow2,
                        gap: 0,
                        trig: 0,
                    });
                    log::debug!(
                        "Far jump (offset {offset}): {:?} -> {far:?}",
                        bb.inst[issue]
                    );
                    bb.inst[issue] = far;
                    bb.inst.splice(issue..issue, inserts);
                }
            }
        }
    }
    log::trace!("VCodeCFG: {cfg:?}");
    changed
}

/// Fills in each branch issue's trigger offset.
///
/// Must run last: any pass that adds or removes instructions after an issue
/// invalidates the offsets.
fn set_trigger_offsets(cfg: &mut VCodeCFG<MInst>) {
    log::debug!("set_trigger_offsets");
    let issuing_blocks = cfg
        .graph
        .all_vertices_weighted_mut()
        .filter_map(|(_, bb)| get_jmp_issues(bb).map(|(issues, trigger)| (bb, issues, trigger)));
    for (bb, issues, trigger) in issuing_blocks {
        // An `ImmJump` is its own issue and trigger and needs no offset.
        for issue in issues.into_iter().filter(|issue| *issue != trigger) {
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
                MInst::JumpIssue { trig, .. } => *trig = offset,
                _ => unreachable!(),
            }
        }

        // Each far issue's offset chain needs to know where its jump is: the
        // first far issue after it (the chain is inserted right before its
        // issue, and only non-issue instructions, such as a bridging echo, can
        // end up between them).
        let chains: Vec<usize> = (0..bb.inst.len())
            .filter(|chain| matches!(bb.inst[*chain], MInst::JumpOffset { .. }))
            .collect();
        for chain in chains {
            let issue = (chain + 1..bb.inst.len())
                .find(|i| {
                    matches!(
                        bb.inst[*i],
                        MInst::JumpIssue {
                            kind: IssueKind::Far { .. },
                            ..
                        }
                    )
                })
                .expect("JumpOffset without a following far jump issue");
            let issue_trig = match bb.inst[issue] {
                MInst::JumpIssue { trig, .. } => trig,
                _ => unreachable!(),
            };
            let between: usize = bb.inst[chain + 1..issue]
                .iter()
                .map(MInst::emitted_length)
                .sum();
            match &mut bb.inst[chain] {
                MInst::JumpOffset { gap, trig, .. } => {
                    *gap = between as u16;
                    *trig = issue_trig;
                }
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
                    kind: IssueKind::Jump,
                    trig: 0,
                },
                MInst::JumpTrigger { link, args },
            ]
        };

        let mut cfg = VCodeCFG::from_vcode(&vcode, &mut new_vreg, replace_jump);
        make_live_ins_explicit(&mut cfg, &mut new_vreg);

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

        // The private frame layout: the incoming stack-argument area (if any)
        // sits at the frame base (offset 0, per the ABI's calling convention),
        // the function's own stack slots above it starting at the 16-aligned
        // locals base (slot offsets are assigned by `Callee::new`, which the
        // stack access lowerings also use, and shifted by the locals base in
        // `stack_frame_offset`), rounded up so the frame stays 16-aligned.
        // The frame base is 16-aligned by ABI guarantee, and keeping every
        // frame's total size aligned preserves that guarantee for callees (a
        // callee's frame base is this frame's split point).
        let incoming_area = stack_area_size(vcode.abi.signature());
        let locals_base = stack_locals_base(vcode.abi.signature());
        let locals_bytes = func
            .sized_stack_slots
            .iter()
            .zip(vcode.abi.sized_stackslot_offsets().iter())
            .map(|((_, data), (_, offset))| offset + data.size)
            .max()
            .unwrap_or(0)
            .next_multiple_of(16);
        let private_bytes = locals_base + locals_bytes;

        // Resolve the stack-passed arguments and return values of this
        // function and of the calls it makes; the outgoing areas start where
        // the private frame ends.
        lower_stack_args(
            &mut cfg,
            vcode.abi.signature(),
            private_bytes,
            &mut new_vreg,
        );

        // Reserve the part of the private frame the incoming caller-reserved
        // area doesn't already occupy.
        insert_frame_limits(&mut cfg, private_bytes - incoming_area);

        resolve_instruction_types(&mut cfg, reg_type, vcode.abi.signature(), &mut new_vreg);
        insert_duplicates(&mut cfg, &mut new_vreg);
        expand_echoes(&mut cfg, &mut new_vreg);
        fix_orderings(&mut cfg, &mut new_vreg);
        fix_branch_conditions(&mut cfg, &mut new_vreg);
        // Widening a jump inserts instructions, which changes the reference
        // distances and can push other jumps out of range, so repeat until
        // nothing changes.
        loop {
            insert_ref_distances(&mut cfg, &mut new_vreg);
            if !widen_far_jumps(&mut cfg, &mut new_vreg) {
                break;
            }
        }
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
            MInst::JumpIssue {
                link,
                dst,
                kind,
                trig,
            } => MInst::JumpIssue {
                link,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
                kind,
                trig,
            },
            MInst::JumpOffset {
                rd,
                dst,
                size_pow2,
                gap,
                trig,
            } => MInst::JumpOffset {
                rd,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
                size_pow2,
                gap,
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
