use crate::isa::scry::inst::MInst;
use crate::machinst::{BlockIndex, VCode};
use regalloc2::{Function, Inst};
use std::boxed::Box;
use std::collections::HashMap;
use std::ops::Index;
use std::vec::Vec;

#[derive(Debug)]
pub(crate) enum Patch {
    Replace(MInst),
    Before(MInst),
    After(MInst),
    Delete,
}

impl Patch {
    pub fn is_delete(&self) -> bool {
        if let Patch::Delete = self {
            true
        } else {
            false
        }
    }
    pub fn is_replace(&self) -> bool {
        if let Patch::Replace(_) = self {
            true
        } else {
            false
        }
    }
    pub fn is_before(&self) -> bool {
        if let Patch::Before(_) = self {
            true
        } else {
            false
        }
    }
    pub fn is_after(&self) -> bool {
        if let Patch::After(_) = self {
            true
        } else {
            false
        }
    }
    pub fn extract(&self) -> &MInst {
        match self {
            Patch::Replace(m) | Patch::Before(m) | Patch::After(m) => m,
            _ => unreachable!(),
        }
    }
    pub fn extract_mut(&mut self) -> &mut MInst {
        match self {
            Patch::Replace(m) | Patch::Before(m) | Patch::After(m) => m,
            _ => unreachable!(),
        }
    }
}

pub(crate) type VCodePatches = HashMap<Inst, Vec<Patch>>;

pub(crate) fn verify_patches(patches: &VCodePatches) -> bool {
    patches
        .iter()
        .all(|(_, p)| p.iter().filter(|p| p.is_delete() || p.is_replace()).count() <= 1)
}

pub(crate) struct PatchIterator<'a> {
    vcode: &'a VCode<MInst>,
    patches: &'a VCodePatches,
    range: Box<dyn DoubleEndedIterator<Item = Inst>>,
    next_inst: Inst,
    next_inst_done: bool,
    next_before: Option<usize>,
    next_after: Option<usize>,
}

impl<'a> PatchIterator<'a> {
    pub(crate) fn new(
        vcode: &'a VCode<MInst>,
        patches: &'a VCodePatches,
        block: BlockIndex,
    ) -> Self {
        assert!(verify_patches(patches), "Patches: {:?}", patches);
        let mut x = Self {
            vcode,
            patches,
            range: Box::new(vcode.block_insns(block).iter()),
            next_inst: Inst(0), // placeholder
            next_inst_done: true,
            next_before: None,
            next_after: None,
        };
        if let Some(inst) = x.range.next() {
            x.next_inst = inst;
            x.next_inst_done = false;
            x.next_before = Some(0);
            x.next_after = Some(0);
        }
        x
    }
}

impl<'a> Iterator for PatchIterator<'a> {
    type Item = (&'a MInst, Inst, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = &mut self.next_before {
            if let Some(m) = self.patches.get(&self.next_inst).map_or(None, |v| {
                v.iter().filter(|p| p.is_before()).skip(*idx).next()
            }) {
                *idx += 1;
                Some((m.extract(), self.next_inst, Some(*idx - 1)))
            } else {
                self.next_before = None;
                self.next()
            }
        } else if !self.next_inst_done {
            self.next_inst_done = true;
            if let Some((idx, replacement)) = self
                .patches
                .get(&self.next_inst)
                .into_iter()
                .filter_map(|ps| ps.iter().enumerate().find(|(_, p)| p.is_replace()))
                .next()
            {
                Some((replacement.extract(), self.next_inst, Some(idx)))
            } else if !self
                .patches
                .get(&self.next_inst)
                .map_or(false, |ps| ps.iter().any(Patch::is_delete))
            {
                Some((self.vcode.index(self.next_inst), self.next_inst, None))
            } else {
                self.next()
            }
        } else if let Some(idx) = &mut self.next_after {
            if let Some(m) = self.patches.get(&self.next_inst).map_or(None, |v| {
                v.iter().filter(|p| p.is_after()).skip(*idx).next()
            }) {
                *idx += 1;
                Some((m.extract(), self.next_inst, Some(*idx - 1)))
            } else {
                self.next_after = None;
                self.next()
            }
        } else {
            if let Some(inst) = self.range.next() {
                self.next_inst = inst;
                self.next_inst_done = false;
                self.next_before = Some(0);
                self.next_after = Some(0);
                self.next()
            } else {
                return None;
            }
        }
    }
}
