use std::collections::{HashSet, VecDeque};
use std::ops::Index;
use crate::{Reg, VCodeInst};
use std::vec::Vec;
use graphene::core::property::*;
use graphene::common::AdjListGraph;
use graphene::core::{Ensure, GraphMut};
use hashbrown::HashMap;
use regalloc2::{Function, OperandKind};
use crate::ir::RelSourceLoc;
use crate::machinst::{BlockIndex, VCode, VCodeBuilder};

#[derive(Debug)]
pub struct VCodeBB<I: VCodeInst> {
	pub inst: Vec<I>,
}

impl<I: VCodeInst> VCodeBB<I> {
	
	pub fn reg_op_kind(&self, reg: Reg, kind: OperandKind) -> impl Iterator<Item = usize> {
		self.inst.iter().enumerate().flat_map(move |(idx, inst)| {
			let mut found = 0;
			inst.clone().get_operands(&mut |r: &mut Reg, _, k, _| if *r == reg && k == kind { found += 1; });
			(0..found).into_iter().map(move|_| idx)
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
	
	/// Gets the index of the instruction that defines the given register.
	///
	/// May panic if the register is defined multiple times
	pub fn reg_def(&self, reg: Reg) -> usize {
		let mut defs = self.reg_defs(reg);
		let def = defs.next().unwrap();
		assert!(defs.next().is_none());
		def
	}
	
	pub fn inst_defs(&self) -> impl Iterator<Item = (usize, Reg)> {
		self.inst.iter().enumerate().flat_map(|(idx, inst)| {
			let mut defs = vec![];
			inst.clone().get_operands(&mut |r: &mut Reg, _, k, _| if k == OperandKind::Def { defs.push(*r); });
			defs.into_iter().map(move|inst| (idx, inst))
		})
	}
}

#[derive(Debug)]
pub struct VCodeCFG<I: VCodeInst>(pub WeakGraph<RootedGraph<AdjListGraph<VCodeBB<I>, ()>>>);

impl<I: VCodeInst> VCodeCFG<I> {

	pub fn from_vcode(vcode: &VCode<I>) -> Self {
		let mut graph = AdjListGraph::<VCodeBB<I>, ()>::new();
		
		let mut bb_map = HashMap::new();
		let mut worklist = VecDeque::new();
		let mut donelist = HashSet::new();
		
		macro_rules! maybe_new_bb {
			($($bb:tt)+) => {
				if !bb_map.contains_key(&($($bb)+)) {
					bb_map.insert(($($bb)+), graph.new_vertex_weighted(VCodeBB{inst: vec![]}).unwrap());
				}
			};
		}
		
		let entry_bb = vcode.entry_block();
		worklist.push_back(entry_bb);
		
		while let Some(bb) = worklist.pop_front() {
			maybe_new_bb!(bb);
			let cfg_bb = graph.vertex_weight_mut(bb_map[&bb]).unwrap();
			
			// Copy instructions
			vcode.block_insns(bb).iter().for_each(|inst| {
				cfg_bb.inst.push(vcode.index(inst).clone());
			});
			
			// Set edges
			for succ in vcode.block_succs(bb) {
				maybe_new_bb!(*succ);
				graph.add_edge(bb_map[&bb], bb_map[succ]).unwrap();
			}
			
			donelist.insert(bb);
		}
		
		Self(Ensure::ensure_all(graph, (*(bb_map.get(&entry_bb).unwrap()), ())).unwrap())
	}
	
	pub fn build_vcode(&self, builder: &mut VCodeBuilder<I>) {
		
		builder.set_entry(BlockIndex::new(0));
		for inst in self.0.root_weight().inst.iter().rev() {
			builder.push(inst.clone(), RelSourceLoc::default());
		}
		builder.end_bb();
	}
}