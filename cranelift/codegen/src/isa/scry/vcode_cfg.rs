use crate::ir::RelSourceLoc;
use crate::machinst::{BlockIndex, VCode, VCodeBuilder};
use crate::{Reg, VCodeInst};
use core::fmt::Debug;
use graphene::common::{AdjListGraph, VertexMapGraph};
use graphene::core::property::*;
use graphene::core::{Directed, Ensure, Graph, GraphMut};
use hashbrown::HashMap;
use regalloc2::{Block, Function, OperandKind};
use std::collections::{HashSet, VecDeque};
use std::ops::Index;
use std::vec::Vec;

#[derive(Debug)]
pub struct VCodeBB<I: VCodeInst> {
    /// The [`Block`] in the vcode corresponding to this block
    pub vcode_bb: Block,

    /// Instructions in the block
    pub inst: Vec<I>,

    /// The block's parameters (inputs)
    pub params: Vec<Reg>,

    pub branch_params: HashMap<Block, Vec<Reg>>,

    /// Ordered list of inputs to the block
    ///
    /// May differ from [`self.params`] if the caller outputs more parameters than this block needs.
    /// E.g., on a conditional branch where one block takes more inputs than the other.
    /// This order should be identical for all successor blocks of a conditional/switch branch.
    pub param_order: Vec<Option<Reg>>,

    /// Ordered list of output of this block
    ///
    /// May differ from `params` of successor blocks. See [`self.param_order`].
    pub branch_param_order: Vec<Option<Reg>>,
}

impl<I: VCodeInst> VCodeBB<I> {
    pub fn reg_op_kind(&self, reg: Reg, kind: OperandKind) -> impl Iterator<Item = usize> {
        self.inst.iter().enumerate().flat_map(move |(idx, inst)| {
            let mut found = 0;
            inst.clone().get_operands(&mut |r: &mut Reg, _, k, _| {
                if *r == reg && k == kind {
                    found += 1;
                }
            });
            (0..found).into_iter().map(move |_| idx)
        })
    }

    /// Gets the index of instructions that use the given register.
    ///
    /// Instructions that use the register multiple times will appear multiple time (equal to the number of defs)
    pub fn reg_uses(&self, reg: Reg) -> impl Iterator<Item = usize> {
        self.reg_op_kind(reg, OperandKind::Use)
    }

    /// Gets the indices of instructions that define the given register.
    ///
    /// Instructions that def the register multiple times will appear multiple time (equal to the number of defs)
    pub fn reg_defs(&self, reg: Reg) -> impl Iterator<Item = usize> {
        self.reg_op_kind(reg, OperandKind::Def)
    }

    pub fn inst_defs(&self) -> impl Iterator<Item = (usize, Reg)> {
        self.inst.iter().enumerate().flat_map(|(idx, inst)| {
            let mut defs = vec![];
            inst.clone().get_operands(&mut |r: &mut Reg, _, k, _| {
                if k == OperandKind::Def {
                    defs.push(*r);
                }
            });
            defs.into_iter().map(move |inst| (idx, inst))
        })
    }
}

/// How a block transfers control to a successor, as far as block layout is
/// concerned. Returned by the machine-dependent `block_exit` callback of
/// [`VCodeCFG::compute_layout`] and [`VCodeCFG::build_vcode`].
pub enum BlockExit {
    /// The block ends in a conditional branch to the given vcode block; the
    /// block's other successor is its fall-through.
    Branch(Block),
    /// The block ends in an unconditional jump to the given vcode block.
    Jump(Block),
}

#[derive(Debug)]
pub struct VCodeCFG<I: VCodeInst> {
    pub graph: UniqueGraph<WeakGraph<RootedGraph<AdjListGraph<VCodeBB<I>, ()>>>>,
}

impl<I: VCodeInst> VCodeCFG<I> {
    pub fn find_unnecessary_block(
        cfg: &impl Graph<Vertex = usize, VertexWeight = VCodeBB<I>>,
    ) -> Option<usize> {
        for (bb_v, bb) in cfg.all_vertices_weighted() {
            if bb.inst.len() == 1
                && bb.inst[0].is_jmp()
                && cfg.edges_sinked_in(bb_v).count() == 1
                && cfg.edges_sourced_in(bb_v).count() == 1
            {
                return Some(bb_v);
            }
        }
        None
    }

    /// `new_vreg`: allocates a fresh virtual register.
    ///
    /// `replace_jump`: the machine-dependent part of edge-block promotion — given a
    /// promoted block's single jump instruction and the block's fresh parameter
    /// registers, returns the instruction sequence that jumps to the same target
    /// while passing those registers on (allocating any scratch registers it needs
    /// from the given allocator).
    pub fn from_vcode(
        vcode: &VCode<I>,
        mut new_vreg: impl FnMut() -> Reg,
        mut replace_jump: impl FnMut(&I, Vec<Reg>, &mut dyn FnMut() -> Reg) -> Vec<I>,
    ) -> Self {
        let mut graph = AdjListGraph::<VCodeBB<I>, ()>::new();

        let mut bb_map = HashMap::new();
        let mut worklist = VecDeque::new();
        let mut donelist = HashSet::new();

        macro_rules! maybe_new_bb {
			($($bb:tt)+) => {
				if !bb_map.contains_key(&($($bb)+)) {
					bb_map.insert(
                        ($($bb)+),
                        graph.new_vertex_weighted(
                            VCodeBB{
                                vcode_bb: ($($bb)+),
                                inst: vec![],
                                params: vec![],
                                branch_params: HashMap::new(),
                                param_order: vec![],
                                branch_param_order: vec![]
                            }
                        ).unwrap()
                    );
				}
			};
		}

        let entry_bb = vcode.entry_block();
        worklist.push_back(entry_bb);

        while let Some(bb) = worklist.pop_front() {
            maybe_new_bb!(bb);
            let cfg_bb_v = bb_map[&bb];

            // Copy parameters
            graph.vertex_weight_mut(cfg_bb_v).unwrap().params.extend(
                vcode
                    .block_params(bb)
                    .iter()
                    .map(|vreg| Reg::from_virtual_reg(*vreg)),
            );

            vcode.block_insns(bb).iter().for_each(|inst| {
                // Copy instructions
                graph
                    .vertex_weight_mut(cfg_bb_v)
                    .unwrap()
                    .inst
                    .push(vcode.index(inst).clone());

                for (i, succ_bb) in vcode.block_succs(bb).iter().enumerate() {
                    maybe_new_bb!(*succ_bb);
                    let branch_params = vcode
                        .branch_blockparams(bb, inst, i)
                        .into_iter()
                        .map(|vreg| Reg::from_virtual_reg(*vreg))
                        .collect();

                    if let Some(p) = graph
                        .vertex_weight(cfg_bb_v)
                        .unwrap()
                        .branch_params
                        .get(succ_bb)
                    {
                        assert_eq!(*p, branch_params);
                        // panic!("Multiple instruction in block targetting same successor block");
                    } else {
                        graph
                            .vertex_weight_mut(cfg_bb_v)
                            .unwrap()
                            .branch_params
                            .insert(*succ_bb, branch_params);
                    }
                }
            });

            // Set edges
            for succ in vcode.block_succs(bb) {
                maybe_new_bb!(*succ);
                graph.add_edge(cfg_bb_v, bb_map[succ]).unwrap();
                if !worklist.contains(succ) && !donelist.contains(succ) {
                    worklist.push_back(*succ);
                }
            }

            donelist.insert(bb);
        }

        // Promote blocks consisting of just a jump into well-formed blocks.
        //
        // These are the edge blocks inserted before instruction selection to split
        // critical edges. They don't adhere to the parameter rules of the "real"
        // blocks: their single jump passes registers owned by the predecessor.
        //
        // They must be *kept* (not folded back into their edge): with every
        // critical edge split, every block with multiple predecessors only has
        // single-successor predecessors and every block with multiple successors
        // only has single-predecessor successors. This guarantees the parameter
        // order unification below always succeeds, that no two conditional blocks
        // ever share a fall-through successor, and that a conditional branch whose
        // arms name the same block keeps two distinguishable edges.
        while let Some(bb_v) = VCodeCFG::find_unnecessary_block(&graph) {
            let bb = graph.vertex_weight_mut(bb_v).unwrap();

            log::trace!("Promoting edge block {bb_v}: {bb:?}");

            let bb_block = bb.vcode_bb;
            assert!(bb.params.is_empty());

            // The single successor, and the registers the block's jump passes to
            // it (owned by the predecessor).
            let (succ, old_args) = {
                let mut iter = bb.branch_params.iter();
                let (succ, args) = iter.next().expect("Jump block without successor params");
                assert!(iter.next().is_none());
                (*succ, args.clone())
            };

            // Fresh registers become the block's parameters and are passed onward
            // in place of the predecessor-owned ones.
            let fresh: Vec<Reg> = old_args.iter().map(|_| new_vreg()).collect();
            bb.params = fresh.clone();
            bb.branch_params.insert(succ, fresh.clone());
            let new_inst = replace_jump(&bb.inst[0], fresh, &mut new_vreg);
            bb.inst = new_inst;

            // The predecessor's branch parameters for this edge were recorded on
            // the edge block itself; the predecessor now passes the registers the
            // edge block's jump used to pass.
            let pred_v = graph.edges_sinked_in(bb_v).next().unwrap().0;
            let pred_bb = graph.vertex_weight_mut(pred_v).unwrap();
            let old = pred_bb.branch_params.insert(bb_block, old_args);
            assert!(old.is_some_and(|p| p.is_empty()));
        }

        let entry_v = bb_map.get(&entry_bb).unwrap();

        Self {
            graph: Ensure::ensure_all(graph, (*(entry_v), ())).unwrap(),
        }
    }

    /// Returns the vertex of the block with the given vcode block, if any.
    pub fn vertex_of_block(&self, block: Block) -> Option<usize> {
        self.graph
            .all_vertices_weighted()
            .find(|(_, bb)| bb.vcode_bb == block)
            .map(|(v, _)| v)
    }

    /// Computes the final linear order of the blocks in the binary, as vertices of
    /// [`Self::graph`]. The entry block is always first. Every conditional block is
    /// directly followed by its fall-through successor.
    ///
    /// The only hard layout constraint is that a conditional block's fall-through
    /// successor (the successor its branch does *not* name, see [`BlockExit`]) must
    /// be placed immediately after it. The layout is built as a greedy trace from
    /// the entry block that always satisfies this. A block reserved as some
    /// conditional's fall-through is never placed by any other means, so it is
    /// guaranteed to be free when its conditional is placed.
    ///
    /// `block_exit` is the machine-dependent query for how a block ends. The result
    /// is deterministic given the graph and the exits, so the layout is not stored
    /// anywhere: callers needing it at different stages (fixing branch conditions,
    /// then emitting in [`Self::build_vcode`]) simply recompute it, which stays
    /// consistent as long as the block structure and exits do not change in
    /// between.
    pub fn compute_layout(
        &self,
        block_exit: impl Fn(&VCodeBB<I>) -> Option<BlockExit>,
    ) -> Vec<usize> {
        // Find each conditional block's fall-through successor.
        let mut fall_through_of = HashMap::new(); // conditional bb -> fall-through vertex
        for (bb_v, bb) in self.graph.all_vertices_weighted() {
            if let Some(BlockExit::Branch(target)) = block_exit(bb) {
                let target_bb_v = self
                    .vertex_of_block(target)
                    .expect("Branch target is not a block");

                let mut non_target_iter = self
                    .graph
                    .edges_sourced_in(bb_v)
                    .map(|(succ_v, _)| succ_v)
                    .filter(|succ_v| *succ_v != target_bb_v);
                let fall_through_v = non_target_iter
                    .next()
                    .expect("No other conditional branch target");
                assert!(
                    non_target_iter.next().is_none(),
                    "Block has more than 2 successors"
                );

                fall_through_of.insert(bb_v, fall_through_v);
            }
        }

        let reserved: HashSet<usize> = fall_through_of.values().copied().collect();
        assert_eq!(
            reserved.len(),
            fall_through_of.len(),
            "Two conditional blocks share a fall-through successor; an intermediate jump block would be needed"
        );
        assert!(
            !reserved.contains(&self.graph.root()),
            "The entry block cannot be a fall-through successor"
        );

        // Greedy trace construction.
        let num_blocks = self.graph.all_vertices().count();
        let mut layout = Vec::with_capacity(num_blocks);
        let mut placed = HashSet::new();
        let mut cursor = Some(self.graph.root());
        while layout.len() < num_blocks {
            let bb_v = cursor.unwrap_or_else(|| {
                // No forced or preferred continuation: pick the smallest unplaced,
                // unreserved vertex (reserved blocks are placed when their conditional
                // is). Deterministic.
                self.graph
                    .all_vertices()
                    .filter(|v| !placed.contains(v) && !reserved.contains(v))
                    .min()
                    .expect("Only reserved blocks left to place")
            });
            layout.push(bb_v);
            placed.insert(bb_v);

            cursor = if let Some(&ft_v) = fall_through_of.get(&bb_v) {
                // Hard constraint: the fall-through successor comes next.
                assert!(!placed.contains(&ft_v));
                Some(ft_v)
            } else {
                // Soft preference: continue with an unconditional jump's target so it
                // becomes a fall-through (the emitter turns a jump-to-next into a NoOp).
                let bb = self.graph.vertex_weight(bb_v).unwrap();
                match block_exit(bb) {
                    Some(BlockExit::Jump(dst)) => self.vertex_of_block(dst),
                    _ => None,
                }
                .filter(|v| !placed.contains(v) && !reserved.contains(v))
            };
        }

        layout
    }

    /// correct_machlabel: correct any machlabel by looking up the old index in the map (old->new)
    ///
    /// Emits the blocks in [`Self::compute_layout`] order (computed here with the
    /// given `block_exit`). Branch conditions must already have been fixed to match
    /// that layout.
    pub fn build_vcode(
        &self,
        builder: &mut VCodeBuilder<I>,
        block_exit: impl Fn(&VCodeBB<I>) -> Option<BlockExit>,
        correct_machlabel: impl Fn(I, &HashMap<usize, usize>) -> I,
    ) {
        log::trace!("building vcode2: {self:?}");

        let layout = self.compute_layout(block_exit);

        // The new block order (and number) no longer fits with the original vcode.
        // MachLabels in instructions depend on the block order and number, so they must be corrected
        // to match the new order and number of the blocks.
        let mut label_idx_map = HashMap::new(); // Old idx to new idx
        for (new_idx, v) in layout.iter().enumerate() {
            let bb = self.graph.vertex_weight(v).unwrap();
            label_idx_map.insert(bb.vcode_bb.index(), new_idx);
        }

        // The builder builds backward, so push the blocks in reverse layout order,
        // correcting MachLabels in instructions on the fly.
        for v in layout.iter().rev().filter(|v| **v != self.graph.root()) {
            let bb = self.graph.vertex_weight(v).unwrap();
            for inst in bb.inst.iter().rev() {
                builder.push(
                    correct_machlabel(inst.clone(), &label_idx_map),
                    RelSourceLoc::default(),
                );
            }
            builder.end_bb();
        }

        // Output root (entry BB) last
        builder.set_entry(BlockIndex::new(0));
        for inst in self.graph.root_weight().inst.iter().rev() {
            builder.push(
                correct_machlabel(inst.clone(), &label_idx_map),
                RelSourceLoc::default(),
            );
        }
        builder.end_bb();
    }

    /// Returns the dataflow graph over the basic blocks of the given CFG.
    ///
    /// The vertices in the graph are the same as the vertices of each BB in the CFG.
    /// The directed edges go from a registers producer BB to a consumer BB.
    /// A value may have a different register in the producer than the consumer (e.g., producer branches using r1, but consumer calls it r2)
    /// All registers referring to the same value will be put on the same edge.
    ///
    /// Each edge has an index (referring to the operand position) a set of registers that are the dependency between the blocks.
    ///
    pub fn dataflow_graph(
        &self,
    ) -> impl Graph<
        Vertex = usize,
        VertexWeight = (),
        EdgeWeight = (usize, HashSet<Reg>),
        Directedness = Directed,
    > + Debug
    + use<I> {
        let mut dfg = VertexMapGraph::<usize, AdjListGraph<_, _, _>>::new();

        // Add every vertex before any edge: an edge is added while processing its
        // sink, and its source may have a higher vertex index (e.g. a join block
        // numbered below one of its predecessors).
        for v in self.graph.all_vertices() {
            dfg.add_vertex(v).unwrap();
        }

        for (v, bb) in self.graph.all_vertices_weighted() {
            for (pred, _) in self.graph.edges_sinked_in(v) {
                let pred_bb = self.graph.vertex_weight(pred).unwrap();

                assert_eq!(pred_bb.branch_params[&bb.vcode_bb].len(), bb.params.len());

                for (idx, (out_r, in_r)) in pred_bb.branch_params[&bb.vcode_bb]
                    .iter()
                    .zip(bb.params.iter())
                    .enumerate()
                {
                    let edge = if let Some(edge) =
                        dfg.edges_between_mut(pred, v).find(|(i, _)| *i == idx)
                    {
                        edge
                    } else {
                        dfg.add_edge_weighted(pred, v, (idx, HashSet::new()))
                            .unwrap();
                        dfg.edges_between_mut(pred, v)
                            .find(|(i, _)| *i == idx)
                            .unwrap()
                    };

                    edge.1.insert(*out_r);
                    edge.1.insert(*in_r);
                }
            }
        }

        dfg
    }
}
