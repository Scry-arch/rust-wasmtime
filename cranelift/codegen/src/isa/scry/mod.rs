//! Scry Instruction Set Architecture.

use crate::dominator_tree::DominatorTree;
use crate::ir::{AbiParam, ArgumentExtension, Signature, pcc};
use crate::ir::{Function, Type};
use crate::isa::scry::inst::{BinaryAluOp, EmitInfo, MInst, ResizeVariant, UnaryAluOp};
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
use crate::{CodegenError, MachLabel};
use crate::{VCodeConstants, ir};
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use core::fmt::{Debug, Formatter};
use cranelift_control::ControlPlane;
use cranelift_entity::EntityRef;
use graphene::core::GraphMut;
use graphene::core::property::{AddEdge, Rooted};
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

    /// Power of 2 size of the type in bytes
    fn is_int(&self) -> bool {
        match self {
            IsaType::Integer(_) => true,
            IsaType::Known(t) => t.is_signed_int() || t.is_unsigned_int(),
            _ => false,
        }
    }

    fn is_known(&self) -> bool {
        match self {
            IsaType::Known(_) => true,
            _ => false,
        }
    }

    fn is_signed_int(&self) -> bool {
        match self {
            IsaType::Known(t) => t.is_signed_int(),
            _ => false,
        }
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
            .cloned()
            .unwrap_or((self.vreg_type)(reg).map_or(IsaType::Invalid, type_to_isatype))
    }

    /// Updates the type assigned to the register.
    ///
    /// Returns whether the new value is different from the existing value or default if none
    fn update(&mut self, reg: Reg, ty: IsaType) -> bool {
        let ty_old = self.get(reg);
        if ty != ty_old {
            log::trace!("New type assignment: {:?}({:?}) <- {:?}", reg, ty_old, ty);
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
        let mut vcode = {
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

        // Perform validation of proof-carrying-code facts, if requested.
        if self.flags().enable_pcc() {
            pcc::check_vcode_facts(func, &mut vcode, self).map_err(CodegenError::Pcc)?;
        }
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

fn prepare_block_params(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    // Handle entry block first, moving MInst:Args to the params
    let entry_bb = cfg.graph.root_weight_mut();
    match &entry_bb.inst[0] {
        MInst::Args { args } => {
            entry_bb.params = args.into_iter().map(|r| r.vreg.to_reg()).collect();
        }
        inst => unreachable!("Entry did not include MInst::Args at the start: {:?}", inst),
    }

    let dfg = cfg.dataflow_graph();

    // Resolve parameter orders
    let mut worklist = HashSet::new();
    worklist.insert(cfg.graph.root());

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
    log::trace!("CFG: {:?}", cfg);
    while let Some(bb_v) = worklist.iter().cloned().next() {
        worklist.remove(&bb_v);

        log::trace!("Resolving block: {}", bb_v);
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

            log::trace!("Block {bb_v} branch params: {:?}", br_params);
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

            log::trace!("Parameter register mapping: {:?}", reg_map);

            // Unify existing param order
            let mut bb_branch_param_order = bb.branch_param_order.clone();
            let mut succ_param_order = succ_bb.param_order.clone();
            for idx in 0..max(succ_param_order.len(), bb_branch_param_order.len()) {
                if succ_param_order.len() <= idx {
                    assert!(bb_branch_param_order.len() > idx);
                    // This block has more parameters than the successor.
                    // Add the parameter to the sucessor

                    // succ_param_order.push(bb_branch_param_order[idx].clone().map(|(param_idx, in_reg)| (param_idx, reg_map[&(param_idx, in_reg)])))
                    succ_param_order.push(None)
                } else if bb_branch_param_order.len() <= idx {
                    // Successor block has more parameters than this.
                    // Add the parameter to this
                    assert!(succ_param_order.len() > idx);

                    if let Some(((_, succ_p), _)) = succ_param_order[idx]
                        .map_or(None, |succ_p| reg_map.iter().find(|(_, r)| **r == succ_p))
                    {
                        // Successor already has a parameter in the mapping, assign the corresponding one from this block
                        bb_branch_param_order.push(Some(*succ_p))
                    } else {
                        bb_branch_param_order.push(None)
                    }
                } else {
                    // They both have something in this position, check correct mapping
                    if let Some(bb_p) = bb_branch_param_order[idx] {
                        if let Some(succ_p) = succ_param_order[idx] {
                            assert!(
                                reg_map
                                    .iter()
                                    .find(|((_, p1), p2)| bb_p == *p1 && **p2 == succ_p)
                                    .is_some()
                            );
                        } else {
                            // Succ has nothing assigned
                            // succ_param_order[idx] = Some((p.0, succ_r));
                        }
                    } else if let Some(p) = succ_param_order[idx] {
                        // Succ has an assignment but this block doesn't, assign one if necessary
                        if let Some(p2) = reg_map.iter().find(|((_, _), r)| **r == p) {
                            bb_branch_param_order[idx] = Some(p2.0.1);
                        }
                    }
                }
            }

            assert_eq!(bb_branch_param_order.len(), succ_param_order.len());

            // Check/add any params not in order
            for (param_idx, param) in br_params.iter().enumerate() {
                if bb_branch_param_order
                    .iter()
                    .find(|p| p.map_or(false, |p| p == *param))
                    .is_none()
                {
                    // This block has a missing parameter in the order

                    // Check if succ has its param
                    let succ_param_reg = reg_map[&(param_idx, *param)];
                    let succ_param = succ_param_order
                        .iter()
                        .enumerate()
                        .find(|(_, p)| p.map_or(false, |r| r == *param));
                    assert!(
                        succ_param.map_or(true, |(_, p)| p.map_or(true, |r| r == succ_param_reg))
                    );

                    if let Some((order_idx, Some(succ_r))) = succ_param {
                        // Successor already has a position for this parameter.
                        assert!(
                            bb_branch_param_order[order_idx].is_none(),
                            "Incongruence in this block and succ"
                        );
                        assert_eq!(*succ_r, succ_param_reg);
                        bb_branch_param_order[order_idx] = Some(*param);
                    } else {
                        // Successor also does not have a position for this parameter. Add to both
                        bb_branch_param_order.push(Some(*param));
                        succ_param_order.push(Some(succ_param_reg));
                    }
                }
            }

            for (param_idx, param_reg) in succ_bb.params.iter().enumerate() {
                if succ_param_order
                    .iter()
                    .find(|p| p.map_or(false, |r| r == *param_reg))
                    .is_none()
                {
                    // The successor is missing a parameter in the order

                    // Check if this block has its param
                    let bb_param = reg_map.keys().find(|(i, _)| *i == param_idx).unwrap();
                    let bb_param_order = bb_branch_param_order
                        .iter()
                        .enumerate()
                        .find(|(_, p)| p.map_or(false, |p2| p2 == bb_param.1));
                    assert!(bb_param_order.map_or(true, |(_, p)| p.map_or(true, |p| {
                        reg_map
                            .iter()
                            .find(|((_, p2), p3)| *p2 == p && *p3 == param_reg)
                            .is_some()
                    })));

                    if let Some((order_idx, _)) = bb_param_order {
                        // this block already has a position for this parameter.
                        assert!(
                            succ_param_order[order_idx].is_none(),
                            "Incongruence in this block and succ"
                        );
                        succ_param_order[order_idx] = Some(*param_reg);
                    } else {
                        unreachable!(
                            "Both blocks cannot have nothing in this parameter order position."
                        )
                    }
                }
            }

            if bb.branch_param_order.len() < bb_branch_param_order.len() {
                //Parameters were added to this block's order. Reevaluate all successors
                for (succ_v, _) in cfg.graph.edges_sourced_in(bb_v) {
                    worklist.insert(succ_v);
                }
            }

            if succ_bb.param_order.len() < succ_param_order.len() {
                //Parameters were added to the successor order. Reevaluate all its predecessors
                for (pred_v, _) in cfg.graph.edges_sinked_in(bb_v) {
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

        // Assert all parameters are present in the order
        assert!(bb.params.iter().all(|p| {
            bb.param_order
                .iter()
                .filter(|po| po.map_or(false, |po| po == *p))
                .count()
                == 1
        }));

        // Assert all branch parameters are present in the order
        assert!(bb.branch_params.iter().all(|(_, params)| {
            params.iter().all(|p| {
                bb.branch_param_order
                    .iter()
                    .filter(|po| po.map_or(false, |po| po == *p))
                    .count()
                    == 1
            })
        }));
    }

    let entry_v = cfg.graph.root();
    for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
        let mut to_drop = Vec::new();

        let params = bb
            .param_order
            .iter()
            .map(|r| {
                r.unwrap_or_else(|| {
                    // This parameter order position is not used by this block, drop the value
                    let v = new_vreg();
                    to_drop.push(v);
                    v
                })
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

        if !to_drop.is_empty() {
            // Insert discard instruction after args
            bb.inst.insert(1, MInst::Discard { rss: to_drop });
        }

        // Insert echos for handling params
        if bb.param_order.len() >= 1 {
            let echo_regs = params.iter().map(|p| {
                let new_vr = new_vreg();
                replace_all_uses(bb, *p, new_vr);
                new_vr
            });

            let rds = echo_regs
                .into_iter()
                .map(|r| WritableReg::from_reg(r))
                .collect();

            bb.inst.insert(
                1, // Insert after Args
                MInst::Echo { rss: params, rds },
            );
        }

        // Update jump trigger inputs
        if let Some((_, trigger)) = get_jmp_issue_trigger(bb) {
            match &mut bb.inst[trigger] {
                MInst::JumpTrigger { args, .. } => {
                    *args = bb.branch_param_order.iter().map(|r| r.unwrap()).collect();
                }
                _ => unreachable!(),
            }
        }
    }

    log::trace!("VCodeCFG: {:?}", cfg);
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
            log::trace!("bb {bb_v}: {:?}", bb);
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
    log::trace!("VCodeCFG: {:?}", cfg);
}

fn insert_ref_distances(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("insert_ref_distances");
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        'a: loop {
            log::trace!("BB: {:?}", bb);

            let mut use_pos = HashMap::<Reg, (usize, u16)>::new(); // reg -> (instruction index, reference distance)
            let mut ref_dist = 0;
            for (inst_idx, inst) in bb.inst.iter_mut().rev().enumerate() {
                log::trace!("inst: {:?}", inst);

                // Expand any echo into echo chains and start over
                match &inst {
                    MInst::Echo { rds, rss } => {
                        for i in MInst::echo_chain(
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
                inst.get_uses().for_each(|r| {
                    use_pos.entry(*r).or_insert((inst_idx, ref_dist));
                });

                if inst.get_defs().count() >= 1 {
                    let ref_dists = inst
                        .get_defs()
                        .enumerate()
                        .map(|(i, def)| {
                            let use_idx = use_pos[&def];
                            (i, ref_dist - use_idx.1 - (inst.reference_length() as u16))
                        })
                        .collect::<HashMap<_, _>>();

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
                        MInst::EchoChain { out1, out2, .. }
                        | MInst::EchoSplit { out1, out2, .. }
                        | MInst::Duplicate { out1, out2, .. } => {
                            *out1 = ref_dists[&0];
                            *out2 = ref_dists[&1];
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

                            bb.inst.insert(inst_idx, MInst::Echo { rds, rss });
                            continue 'a;
                        }
                        _ => (),
                    };
                }
            }
            break;
        }
        log::trace!("BB: {:?}", bb);
    }
    log::trace!("VCodeCFG: {:?}", cfg);
}

fn fix_orderings(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    log::debug!("fix_orderings");
    for (bb_v, bb) in cfg.graph.all_vertices_weighted_mut() {
        log::trace!("bb {bb_v}: {:?}", bb);
        'a: loop {
            let mut def_pos = HashMap::<Reg, usize>::new();
            for (inst_idx, inst) in bb.inst.iter_mut().enumerate() {
                log::trace!("Inst {inst_idx}: {:?}", inst);
                // Record def positions
                for def in inst.get_defs() {
                    def_pos.insert(def, inst_idx);
                }

                if inst.use_order_meaningful() && inst.get_uses().count() > 1 {
                    let mut uses = inst.get_uses_mut().collect::<Vec<_>>();
                    if def_pos[uses[1]] < def_pos[uses[0]] {
                        // Create new vregs for the reorder
                        let first_new = new_vreg();
                        let second_new = new_vreg();

                        let first_old = *uses[1];
                        let second_old = *uses[0];

                        // assign reordered vregs to store
                        *uses[0] = second_new;
                        *uses[1] = first_new;

                        // Insert reorder instruction before store
                        bb.inst.insert(
                            inst_idx,
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
    log::trace!("VCodeCFG: {:?}", cfg);
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

fn type_analysis<F: Fn(Reg) -> Option<Type>>(
    cfg: &mut VCodeCFG<MInst>,
    vreg_type: F,
    func_sig: &Signature,
) -> TypeMap<F> {
    log::trace!("Type Analysis");

    let bb_dfg = cfg.dataflow_graph();
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

    // Resolve instruction types in each BB
    let mut bb_worklist: VecDeque<usize> = VecDeque::new();

    // All blocks analyzed at least once
    bb_worklist.extend(cfg.graph.all_vertices());

    while let Some((bb_v, bb)) = bb_worklist
        .pop_front()
        .map(|bb_v| (bb_v, cfg.graph.vertex_weight(bb_v).unwrap()))
    {
        log::trace!("BB idx: {}", bb_v);
        let mut changed_regs = HashSet::<Reg>::new();

        let inst_dfg = bb.dataflow_graph();
        log::trace!("Inst DFG: {:?}", inst_dfg);

        let mut inst_worklist = HashSet::new();

        // All instructions are analyzed at least once
        inst_worklist.extend(0..bb.inst.len());

        while let Some(inst_idx) = inst_worklist.iter().cloned().next() {
            inst_worklist.remove(&inst_idx);
            let inst = &bb.inst[inst_idx];
            log::trace!("Inst {}: {:?}", inst_idx, inst);

            // Checks if the given type is different from the given registers existing type.
            // If so, assigns it and updates worklists for instructions and BBs
            let mut update_changed = |r: &Reg, new_type: IsaType, map: &mut TypeMap<_>| {
                let refined = map.get(*r).refine(new_type).unwrap();
                if map.update(*r, refined) {
                    changed_regs.insert(*r);
                    inst_dfg
                        .edges_incident_on(inst_idx)
                        .filter(|(_, dep_r)| **dep_r == *r)
                        .for_each(|(dep_i, _)| {
                            log::trace!("To worklist: {}({:?})", dep_i, bb.inst[dep_i]);
                            inst_worklist.insert(dep_i);
                        })
                }
            };

            use MInst::*;
            match inst {
                BinaryAlu { rd, rs1, rs2, .. } => {
                    let t1 = type_map.get(*rs1);
                    let t2 = type_map.get(*rs2);
                    let td = type_map.get(rd.to_reg());

                    let refined = t1.refine(t2).unwrap().refine(td).unwrap();

                    update_changed(rs1, refined, &mut type_map);
                    update_changed(rs2, refined, &mut type_map);
                    update_changed(&rd.to_reg(), refined, &mut type_map);
                }
                IntCmp { rd, rs1, rs2, cc } => {
                    let t1 = type_map.get(*rs1);
                    let t2 = type_map.get(*rs2);
                    let td = type_map
                        .get(rd.to_reg())
                        .refine(IsaType::Known(scry_isa::Type::Uint(0)))
                        .unwrap();

                    let refined = t1.refine(t2).unwrap();

                    if refined.is_int() {
                        let in_refined = match cc {
                            IntCC::UnsignedGreaterThan
                            | IntCC::UnsignedLessThan
                            | IntCC::UnsignedGreaterThanOrEqual
                            | IntCC::UnsignedLessThanOrEqual => refined
                                .refine(IsaType::new_known_int(refined.size_pow2(), false))
                                .unwrap(),
                            IntCC::SignedGreaterThan
                            | IntCC::SignedLessThan
                            | IntCC::SignedGreaterThanOrEqual
                            | IntCC::SignedLessThanOrEqual => refined
                                .refine(IsaType::new_known_int(refined.size_pow2(), true))
                                .unwrap(),
                            IntCC::Equal | IntCC::NotEqual => {
                                // Equality put no constraint on operand signedness
                                refined
                            }
                        };

                        update_changed(rs1, in_refined, &mut type_map);
                        update_changed(rs2, in_refined, &mut type_map);
                        update_changed(&rd.to_reg(), td, &mut type_map);
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

                    // Both operands need to be set to the specified signedness
                    if rs_t.is_int() {
                        update_changed(
                            rs,
                            rs_t.refine(IsaType::new_known_int(rs_t.size_pow2(), sign))
                                .unwrap(),
                            &mut type_map,
                        );
                    }
                    if rd_t.is_int() {
                        update_changed(
                            &rd.to_reg(),
                            rd_t.refine(IsaType::new_known_int(rd_t.size_pow2(), sign))
                                .unwrap(),
                            &mut type_map,
                        );
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
                        update_changed(&rd.to_reg(), rd_t, &mut type_map);
                    } else if rs_t.is_known() && rd_t.is_known() {
                        panic!(
                            "Incompatible type requirements: rs_t {:?}, rd_t {:?}",
                            rs_t, rd_t
                        );
                    } else if rs_t.is_known() || rd_t.is_known() {
                        // One is known, so use its signedness
                        let sign = rs_t.is_signed_int() || rd_t.is_signed_int();

                        if rs_t.is_int() {
                            update_changed(
                                rs,
                                rs_t.refine(IsaType::new_known_int(rs_t.size_pow2(), sign))
                                    .unwrap(),
                                &mut type_map,
                            );
                        }
                        if rd_t.is_int() {
                            update_changed(
                                &rd.to_reg(),
                                rd_t.refine(IsaType::new_known_int(rd_t.size_pow2(), sign))
                                    .unwrap(),
                                &mut type_map,
                            );
                        }
                    }
                }
                Const { rd, .. } => {
                    let rd_t = type_map.get(rd.to_reg());
                    update_changed(&rd.to_reg(), rd_t, &mut type_map);
                }
                Echo { rds, rss, .. } => {
                    if rss.len() != 0 {
                        assert_eq!(rds.len(), rss.len());
                        rss.iter().zip(rds.iter()).for_each(|(rs, rd)| {
                            update_changed(&rd.to_reg(), type_map.get(*rs), &mut type_map);
                        });
                    } else {
                        assert!(!rds.is_empty());
                        // This echo handles block parameters. Do nothing.
                    }
                }
                Reorder {
                    rd1, rd2, rs1, rs2, ..
                } => {
                    update_changed(&rd1.to_reg(), type_map.get(*rs1), &mut type_map);
                    update_changed(&rd2.to_reg(), type_map.get(*rs2), &mut type_map);
                }
                Duplicate { rd1, rd2, rs, .. } => {
                    let rs_t = type_map.get(*rs);
                    let rd1_t = type_map.get(rd1.to_reg());
                    let rd2_t = type_map.get(rd2.to_reg());

                    let merged_t = rs_t.refine(rd1_t).unwrap().refine(rd2_t).unwrap();

                    update_changed(rs, merged_t, &mut type_map);
                    update_changed(&rd1.to_reg(), merged_t, &mut type_map);
                    update_changed(&rd2.to_reg(), merged_t, &mut type_map);
                }
                Load { rs, .. } => {
                    update_changed(
                        rs,
                        type_map
                            .get(*rs)
                            .refine(IsaType::Known(scry_isa::Type::Uint(2)))
                            .expect("Load source refine fail"),
                        &mut type_map,
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
                            &mut type_map,
                        );
                    }

                    for (r, p) in args.iter().map(|p| p.vreg).zip(sig.params.iter()) {
                        let ty = abi_param_to_isatype(p);
                        update_changed(&r, type_map.get(r).refine(ty).unwrap(), &mut type_map);
                    }
                }
                Rets { rets } => {
                    assert_eq!(rets.len(), func_sig.returns.len());
                    for (r, p) in rets.iter().map(|p| p.vreg).zip(func_sig.returns.iter()) {
                        let ty = abi_param_to_isatype(p);
                        update_changed(&r, type_map.get(r).refine(ty).unwrap(), &mut type_map);
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
                            .for_each(|r| update_changed(&r, shared_ty, &mut type_map));
                    }
                }
                _ => (),
            }
        }

        // Add any BBs to the worklist if they depend on a register that was updated by this BB
        for changed_r in changed_regs.into_iter() {
            for (other_v, weight) in bb_dfg.edges_incident_on(bb_v) {
                let dep_regs = &weight.1;
                if dep_regs.contains(&changed_r) {
                    if bb_worklist.iter().all(|v| other_v != *v) {
                        bb_worklist.push_back(other_v);
                    }
                }
            }
        }
    }
    log::trace!("Resolved types: {:?}", type_map);

    type_map
}

fn resolve_instruction_types(
    cfg: &mut VCodeCFG<MInst>,
    vreg_type: impl Fn(Reg) -> Option<Type>,
    func_sig: &Signature,
) {
    let type_map = type_analysis(cfg, vreg_type, func_sig);

    log::trace!("Resolve instruction types");

    // Resolve type specific instructions into Scry instructions
    for (_, bb) in cfg.graph.all_vertices_weighted_mut() {
        for (_, inst) in bb.inst.iter_mut().enumerate() {
            match inst {
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
                    assert!(*var == ResizeVariant::Reduce || rd_t.is_same_signedness(&rs_t));
                    assert!(*var != ResizeVariant::Reduce || (rd_t.is_int() && rs_t.is_int()));

                    *inst = MInst::Cast {
                        rd: *rd,
                        ty: rd_t,
                        rs: *rs,
                        out: 0,
                    };
                }
                MInst::Const { ty, rd, .. } => {
                    let rd_t = type_map.get(*&rd.to_reg());

                    assert!(rd_t.is_int());

                    *ty = rd_t;
                }
                MInst::Load { ty, rd, .. } => {
                    let rd_t = type_map.get(*&rd.to_reg());

                    if rd_t.is_known() {
                        *ty = rd_t;
                    } else {
                        panic!("Invalid load type: {:?}", rd_t);
                    }
                }
                _ => (),
            }
        }
    }
}

pub fn get_jmp_issue_trigger(block: &VCodeBB<MInst>) -> Option<(usize, usize)> {
    let mut issue = None;
    let mut trigger = None;

    for idx in 0..block.inst.len() {
        match block.inst[idx] {
            MInst::BranchIssue { .. } | MInst::JumpIssue { .. } => {
                assert!(issue.is_none(), "Multiple jump issues");
                issue = Some(idx);
            }
            MInst::JumpTrigger { .. } => {
                assert!(trigger.is_none(), "Multiple jump triggers");
                trigger = Some(idx);
            }
            MInst::ImmJump { .. } => {
                assert!(issue.is_none(), "Multiple jump issues");
                assert!(trigger.is_none(), "Multiple jump triggers");
                trigger = Some(idx);
                issue = Some(idx);
            }
            _ => (),
        }
    }

    match (issue, trigger) {
        (Some(issue), Some(trigger)) => Some((issue, trigger)),
        (None, None) => None,
        _ => panic!(),
    }
}

fn block_ordering(cfg: &mut VCodeCFG<MInst>, mut new_vreg: impl FnMut() -> Reg) {
    for bb_v in cfg.graph.all_vertices().collect::<Vec<_>>().into_iter() {
        // Add order dependencies based on branches
        if let Some((issue, _)) = get_jmp_issue_trigger(cfg.graph.vertex_weight(bb_v).unwrap()) {
            // Add dependencies of branch targets
            match cfg.graph.vertex_weight(bb_v).unwrap().inst[issue] {
                MInst::BranchIssue {
                    dir,
                    dst,
                    cond,
                    link,
                } => {
                    let target_bb_v = cfg
                        .graph
                        .all_vertices_weighted()
                        .find(|(_, b)| b.vcode_bb == Block::new(dst.index()))
                        .map(|(v, _)| v)
                        .unwrap();

                    if target_bb_v != bb_v {
                        if dir {
                            // Forward jump
                            if cfg.block_order.edges_between(bb_v, target_bb_v).count() == 0 {
                                // Not assigned in the order, assign it
                                cfg.block_order
                                    .add_edge_weighted(bb_v, target_bb_v, Ordering::Before)
                                    .expect("Found conflicting block order requirement");
                            }
                        } else {
                            // Assumes jump backwards
                            if cfg.block_order.edges_between(target_bb_v, bb_v).count() == 0 {
                                if cfg.block_order.edges_between(bb_v, target_bb_v).count() > 0 {
                                    // Ordering requires a forward jump, flip the branch direction
                                    let neg_cond_r = new_vreg();
                                    let pos_cond_r = cond;

                                    let negation = MInst::UnaryAlu {
                                        op: UnaryAluOp::LogNeg,
                                        rd: WritableReg::from_reg(neg_cond_r),
                                        rs: pos_cond_r,
                                    };

                                    let bb_mut = cfg.graph.vertex_weight_mut(bb_v).unwrap();
                                    bb_mut.inst[issue] = MInst::BranchIssue {
                                        dir: true,
                                        dst,
                                        cond: neg_cond_r,
                                        link,
                                    };
                                    bb_mut.inst.insert(issue, negation);
                                } else {
                                    // Not assigned in the order, assign it
                                    cfg.block_order
                                        .add_edge_weighted(target_bb_v, bb_v, Ordering::Before)
                                        .expect("Found conflicting block order requirement");
                                }
                            }
                        }
                    } else {
                        // The branch loops back to the same block
                        // Must use backwards jump
                        if dir {
                            // Existing code assumes forward jump, negate incoming condition
                            let neg_cond_r = new_vreg();
                            let pos_cond_r = cond;

                            let negation = MInst::UnaryAlu {
                                op: UnaryAluOp::LogNeg,
                                rd: WritableReg::from_reg(neg_cond_r),
                                rs: pos_cond_r,
                            };

                            let bb_mut = cfg.graph.vertex_weight_mut(bb_v).unwrap();
                            bb_mut.inst[issue] = MInst::BranchIssue {
                                dir: false,
                                dst,
                                cond: neg_cond_r,
                                link,
                            };
                            bb_mut.inst.insert(issue, negation);
                        } else {
                            // Existing code assumes correct backwards jump, do nothing.
                        }
                    }

                    // Add dependency of successor without branch
                    let mut non_branch_target_iter = cfg
                        .graph
                        .edges_sourced_in(bb_v)
                        .filter(|(succ_v, _)| *succ_v != target_bb_v);

                    let (succ_v, _) = non_branch_target_iter
                        .next()
                        .expect("No other conditional branch target");

                    if cfg
                        .block_order
                        .edges_sourced_in(bb_v)
                        .filter(|(_, o)| **o == Ordering::Precede)
                        .count()
                        == 0
                    {
                        if let Some(o) = cfg
                            .block_order
                            .edges_between_mut(bb_v, succ_v)
                            .find(|o| **o == Ordering::Before)
                        {
                            *o = Ordering::Precede;
                        } else {
                            cfg.block_order
                                .add_edge_weighted(bb_v, succ_v, Ordering::Precede)
                                .unwrap()
                        }
                    } else {
                        unimplemented!("Branch already has a Ordering::Precede dependending on it")
                    }
                    assert!(
                        non_branch_target_iter.next().is_none(),
                        "Block has more than 2 successors"
                    );
                }
                _ => (),
            }
        }
    }
}

impl TargetIsa for ScryBackend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        log::debug!("Beginning Scry compile");
        log::trace!("func: {:?}", func);
        let (vcode, mut new_vregs) = self.compile_vcode(func, domtree, ctrl_plane)?;

        log::trace!("func signature: {:?}", vcode.abi.signature());

        let mut new_vreg = || {
            new_vregs
                .alloc_with_deferred_error(Type::int(32).unwrap())
                .only_reg()
                .unwrap()
        };
        let reg_type = |r: Reg| vcode.vreg_type_maybe(r.to_virtual_reg().unwrap().into());

        let update_branch_target = |bb: &mut VCodeBB<_>, new_target: MachLabel| {
            let (issue_idx, _) = get_jmp_issue_trigger(bb).unwrap();
            let issue_inst = bb.inst.get_mut(issue_idx).unwrap();
            match issue_inst {
                MInst::BranchIssue { dst, .. } => {
                    *dst = new_target;
                }
                _ => unimplemented!(),
            }
        };

        let mut cfg = VCodeCFG::from_vcode(&vcode, update_branch_target);

        log::trace!("VCodeCFG: {:?}", cfg);

        prepare_block_params(&mut cfg, &mut new_vreg);

        // Insert `ret` instruction as movable trigger
        cfg.graph
            .all_vertices_weighted_mut()
            .find(|(_, bb)| {
                bb.inst
                    .iter()
                    .find(|i| {
                        if let MInst::Rets { .. } = i {
                            true
                        } else {
                            false
                        }
                    })
                    .is_some()
            })
            .map(|(_, bb)| {
                let inst_count = bb.inst.len();
                bb.inst.insert(inst_count - 1, MInst::Ret { trig: 0 })
            });

        log::trace!("VCodeCFG: {:?}", cfg);

        resolve_instruction_types(&mut cfg, reg_type, vcode.abi.signature());
        insert_duplicates(&mut cfg, &mut new_vreg);
        fix_orderings(&mut cfg, &mut new_vreg);
        block_ordering(&mut cfg, &mut new_vreg);
        insert_ref_distances(&mut cfg, &mut new_vreg);

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

        cfg.build_vcode(&mut builder, |inst, label_idx_map| match inst {
            MInst::BranchIssue {
                link,
                dst,
                dir,
                cond,
            } => MInst::BranchIssue {
                link,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
                dir,
                cond,
            },
            MInst::JumpIssue { link, dst } => MInst::JumpIssue {
                link,
                dst: MachLabel::new(label_idx_map[&dst.index()]),
            },
            i => i,
        });

        let vreg_alloc = VRegAllocator::with_capacity(vcode.num_vregs());
        let vcode2 = builder.build(vreg_alloc);

        log::trace!("VCode2: {:?}", vcode2);

        let want_disasm = want_disasm || log::log_enabled!(log::Level::Debug);
        let emit_result = vcode2.emit(
            &regalloc2::Output::default(),
            want_disasm,
            &self.flags,
            ctrl_plane,
        );
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
