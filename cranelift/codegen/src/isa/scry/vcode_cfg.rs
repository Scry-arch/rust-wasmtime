use crate::ir::RelSourceLoc;
use crate::machinst::{BlockIndex, VCode, VCodeBuilder};
use crate::{Reg, VCodeInst};
use core::fmt::Debug;
use graphene::common::{AdjListGraph, VertexMapGraph};
use graphene::core::property::*;
use graphene::core::{Directed, Ensure, Graph, GraphMut};
use hashbrown::HashMap;
use regalloc2::{Function, OperandKind};
use std::collections::{HashSet, VecDeque};
use std::ops::Index;
use std::vec::Vec;

#[derive(Debug)]
pub struct VCodeBB<I: VCodeInst> {
    pub inst: Vec<I>,
    pub params: Vec<Reg>,
    pub branch_params: Vec<Reg>,
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

#[derive(Debug)]
pub struct VCodeCFG<I: VCodeInst>(
    pub UniqueGraph<WeakGraph<RootedGraph<AdjListGraph<VCodeBB<I>, ()>>>>,
);

impl<I: VCodeInst> VCodeCFG<I> {
    pub fn from_vcode(vcode: &VCode<I>) -> Self {
        let mut graph = AdjListGraph::<VCodeBB<I>, ()>::new();

        let mut bb_map = HashMap::new();
        let mut worklist = VecDeque::new();
        let mut donelist = HashSet::new();

        macro_rules! maybe_new_bb {
			($($bb:tt)+) => {
				if !bb_map.contains_key(&($($bb)+)) {
					bb_map.insert(($($bb)+), graph.new_vertex_weighted(VCodeBB{inst: vec![], params: vec![], branch_params: vec![]}).unwrap());
				}
			};
		}

        let entry_bb = vcode.entry_block();
        worklist.push_back(entry_bb);

        while let Some(bb) = worklist.pop_front() {
            maybe_new_bb!(bb);
            let cfg_bb = graph.vertex_weight_mut(bb_map[&bb]).unwrap();

            // Copy parameters
            cfg_bb.params.extend(
                vcode
                    .block_params(bb)
                    .iter()
                    .map(|vreg| Reg::from_virtual_reg(*vreg)),
            );

            vcode.block_insns(bb).iter().for_each(|inst| {
                // Copy instructions
                cfg_bb.inst.push(vcode.index(inst).clone());

                for (i, _) in vcode.block_succs(bb).iter().enumerate() {
                    let branch_params = vcode.branch_blockparams(bb, inst, i);
                    let uneven_br_params = cfg_bb.branch_params.len() != branch_params.len();
                    if uneven_br_params && cfg_bb.branch_params.len() == 0 {
                        cfg_bb.branch_params = branch_params
                            .iter()
                            .map(|vreg| Reg::from_virtual_reg(*vreg))
                            .collect();
                    } else if uneven_br_params && branch_params.len() != 0 {
                        unreachable!("Uneven branch params lengths")
                    }
                }
            });

            // Set edges
            for succ in vcode.block_succs(bb) {
                maybe_new_bb!(*succ);
                if !worklist.contains(succ)
                    && !donelist.contains(succ)
                    && graph.add_edge(bb_map[&bb], bb_map[succ]).is_ok()
                {
                    worklist.push_back(*succ);
                }
            }

            donelist.insert(bb);
        }

        Self(Ensure::ensure_all(graph, (*(bb_map.get(&entry_bb).unwrap()), ())).unwrap())
    }

    pub fn build_vcode(&self, builder: &mut VCodeBuilder<I>) {
        for (_, bb) in self
            .0
            .all_vertices_weighted()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .filter(|(v, _)| *v != self.0.root())
        {
            for inst in bb.inst.iter().rev() {
                builder.push(inst.clone(), RelSourceLoc::default());
            }
            builder.end_bb();
        }

        // Output root (entry BB) last
        builder.set_entry(BlockIndex::new(0));
        for inst in self.0.root_weight().inst.iter().rev() {
            builder.push(inst.clone(), RelSourceLoc::default());
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
    > + Debug {
        let mut dfg = VertexMapGraph::<usize, AdjListGraph<_, _, _>>::new();

        for (v, bb) in self.0.all_vertices_weighted() {
            dfg.add_vertex(v).unwrap();

            for (pred, _) in self.0.edges_sinked_in(v) {
                let pred_bb = self.0.vertex_weight(pred).unwrap();

                assert_eq!(pred_bb.branch_params.len(), bb.params.len());

                for (idx, (out_r, in_r)) in pred_bb
                    .branch_params
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
