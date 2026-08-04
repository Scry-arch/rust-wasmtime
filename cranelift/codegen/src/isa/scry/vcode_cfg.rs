use crate::ir::RelSourceLoc;
use crate::machinst::{BlockIndex, VCode, VCodeBuilder};
use crate::{MachLabel, Reg, VCodeInst};
use core::fmt::Debug;
use cranelift_entity::EntityRef;
use graphene::algo::Retainable;
use graphene::algo::search::Topo;
use graphene::common::{AdjListGraph, VertexMapGraph};
use graphene::core::property::*;
use graphene::core::{BaseGraphGuard, Directed, Ensure, Graph, GraphMut};
use hashbrown::HashMap;
use regalloc2::{Block, Function, OperandKind};
use std::collections::{HashSet, VecDeque};
use std::iter::once;
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

    pub fn dataflow_graph(
        &self,
    ) -> impl Graph<Vertex = usize, VertexWeight = (), EdgeWeight = Reg, Directedness = Directed> + Debug
    {
        let mut dfg = VertexMapGraph::<usize, AdjListGraph<_, _, _>>::new();
        for (inst_idx, inst) in self.inst.iter().enumerate() {
            dfg.add_vertex(inst_idx).unwrap();

            inst.clone().get_operands(&mut |r: &mut Reg, _, k, _| {
                if k == OperandKind::Use {
                    if let Some(def_inst_idx) = self.reg_defs(*r).next() {
                        dfg.add_edge_weighted(def_inst_idx, inst_idx, *r).unwrap();
                    }
                }
            })
        }

        dfg
    }
}

/// Specifies ordering dependency requirements between blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Source block must come before the sink block in the binary
    Before,

    /// The source block must come immediately before the sink block in the binary, with no other blocks between.
    Precede,
}

#[derive(Debug)]
pub struct VCodeCFG<I: VCodeInst> {
    pub graph: UniqueGraph<WeakGraph<RootedGraph<AdjListGraph<VCodeBB<I>, ()>>>>,

    // A DAG of which blocks must come before other blocks in the function (i.e. at earlier addresses)
    // An edge from v1 to v2 means v1 must come before v2.
    pub block_order: AcyclicGraph<UniqueGraph<VertexMapGraph<usize, AdjListGraph<(), Ordering>>>>,
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

    pub fn from_vcode(
        vcode: &VCode<I>,
        update_branch_target: impl Fn(&mut VCodeBB<I>, MachLabel),
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

        // Eliminate blocks consisting of just a jump.
        // These are inserted before instruction selection to be the "not taken" target of conditional branches
        // They don't adhere to the parameter rules of the "real" blocks and have no purpose.
        // Find them and move all control flow back to the original target
        while let Some(bb_v) = VCodeCFG::find_unnecessary_block(&graph) {
            let bb = graph.vertex_weight(bb_v).unwrap();

            log::trace!("Unnecessary block {}: {:?}", bb_v, bb);

            let bb_block = bb.vcode_bb;
            let pred_v = graph.edges_sinked_in(bb_v).next().unwrap().0;
            let succ_v = graph.edges_sourced_in(bb_v).next().unwrap().0;
            let succ_bb_block = graph.vertex_weight(succ_v).unwrap().vcode_bb;
            let branch_params_to_succ = bb.branch_params.get(&succ_bb_block).unwrap().clone();
            let pred_bb = graph.vertex_weight_mut(pred_v).unwrap();

            // Retarget any predecessors to the successor
            assert!(pred_bb.branch_params.get(&succ_bb_block).is_none());
            let pred_branch_params = pred_bb.branch_params.remove(&bb_block).unwrap();
            assert!(pred_branch_params.is_empty()); // The correct branch parameters are in the unnecessary block, ignore those in the predecessor
            pred_bb
                .branch_params
                .insert(succ_bb_block, branch_params_to_succ);

            update_branch_target(pred_bb, MachLabel::new(succ_bb_block.index()));

            assert_eq!(graph.edges_between(pred_v, succ_v).count(), 0);
            graph.add_edge(pred_v, succ_v).unwrap();

            // Remove block and its edges
            graph.remove_vertex(bb_v).unwrap();
        }

        let entry_v = bb_map.get(&entry_bb).unwrap();

        // Initial block order just requires the entry block to be before all others
        let mut block_order = VertexMapGraph::new();
        graph.all_vertices().for_each(|v| {
            block_order.add_vertex(v).unwrap();
            if v != *entry_v {
                block_order
                    .add_edge_weighted(entry_v, v, Ordering::Before)
                    .unwrap()
            }
        });

        Self {
            graph: Ensure::ensure_all(graph, (*(entry_v), ())).unwrap(),
            block_order: block_order.guard_all().unwrap(),
        }
    }

    /// correct_machlabel: correct any machlabel by looking up the old index in the map (old->new)
    pub fn build_vcode(
        &self,
        builder: &mut VCodeBuilder<I>,
        correct_machlabel: impl Fn(I, &HashMap<usize, usize>) -> I,
    ) {
        log::trace!("building vcode2: {:?}", self);

        let blocks = self.graph.all_vertices_weighted().collect::<Vec<_>>();

        //TODO: Must sort the blocks in topological order while accounting for Ordering::Precede

        // Create a graph where all nodes that have Order::Precede relations are merged into 1 node
        // This can then be sorted and expanded again

        let mut merged = AdjListGraph::<HashSet<usize>, ()>::new();

        // Add nodes in their groups
        for v in self.graph.all_vertices() {
            if merged.all_vertices_weighted().all(|(_, w)| !w.contains(&v)) {
                // v is not in the graph
                if let Some((v2, _)) = self
                    .block_order
                    .edges_incident_on(v)
                    .find(|(_, o)| **o == Ordering::Precede)
                {
                    if let Some((_, set)) = merged
                        .all_vertices_weighted_mut()
                        .find(|(_, w)| w.contains(&v2))
                    {
                        set.insert(v);
                        continue;
                    }
                }
                // v does not need to be merged or none of the others in its merge have already been inserted.
                merged
                    .new_vertex_weighted([v].into_iter().collect())
                    .unwrap();
            }
        }
        // dbg!(&self.graph);
        // dbg!(&self
        //     .block_order);
        // dbg!(&merged);

        // add edges between groups
        for (so, si, _) in self
            .block_order
            .all_edges()
            .filter(|(_, _, o)| **o != Ordering::Precede)
        {
            let so_g = merged
                .all_vertices_weighted()
                .find(|(_, w)| w.contains(&so))
                .unwrap()
                .0;
            let si_g = merged
                .all_vertices_weighted()
                .find(|(_, w)| w.contains(&si))
                .unwrap()
                .0;

            merged.add_edge(so_g, si_g).unwrap(); // Could have multiple edges, but that doesn't matter
        }

        // Find the root vertex
        let merge_root_v = merged
            .all_vertices_weighted()
            .find(|(_, w)| w.contains(&self.graph.root()))
            .unwrap()
            .0;
        let merged = VertexInGraph::<AcyclicGraph<AdjListGraph<_, _>>>::ensure_all(
            merged,
            ([merge_root_v], ()),
        )
        .unwrap();

        let topo_sort = once(merge_root_v)
            .chain(Topo::new(&merged).retain(&merged))
            .flat_map(|mv| {
                let merge_group = merged.vertex_weight(mv).unwrap();

                let mut sorted_group = VecDeque::with_capacity(merge_group.len());
                sorted_group.push_front(merge_group.iter().cloned().next().unwrap());

                // Continuously add the previous and the next in the group
                let mut found_more = true;
                while found_more {
                    found_more = false;

                    if let Some((prec, _)) = self
                        .block_order
                        .edges_sinked_in(sorted_group[0])
                        .find(|(_, o)| **o == Ordering::Precede)
                    {
                        sorted_group.push_front(prec);
                        found_more |= true;
                    }
                    if let Some((succ, _)) = self
                        .block_order
                        .edges_sourced_in(sorted_group.back().cloned().unwrap())
                        .find(|(_, o)| **o == Ordering::Precede)
                    {
                        sorted_group.push_back(succ);
                        found_more |= true;
                    }
                }
                sorted_group.into_iter()
            })
            .collect::<Vec<_>>();

        // The new block order (and number) no longer fits with the original vcode.
        // MachLabels in instructions depend on the block order and number, so they must be corrected
        // to match the new order and number of the blocks.

        let mut label_idx_map = HashMap::new(); // Old idx to new idx
        let mut new_block_idx = blocks.len();
        for v in topo_sort.iter().rev() {
            let bb = self.graph.vertex_weight(v).unwrap();
            new_block_idx -= 1;
            label_idx_map.insert(bb.vcode_bb.index(), new_block_idx);
        }

        // Now that the order is settled, we begin building the new vcode and correcting MachLabels in instructions on the fly.
        for v in topo_sort
            .into_iter()
            .rev()
            .filter(|v| *v != self.graph.root())
        {
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

        for (v, bb) in self.graph.all_vertices_weighted() {
            dfg.add_vertex(v).unwrap();

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
