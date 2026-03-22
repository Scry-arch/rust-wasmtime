use crate::isa::scry::inst::MInst;
use crate::machinst::{BlockIndex, VCode};
use regalloc2::{Function, Inst};
use std::boxed::Box;
use std::collections::HashMap;
use std::ops::Index;
use std::vec::Vec;

#[derive(Debug)]
pub(crate) enum Patch {
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
}

pub(crate) type VCodePatches = HashMap<Inst, Vec<Patch>>;

pub(crate) fn verify_patches(patches: &VCodePatches) -> bool {
    patches.iter().all(|(_, p)| {
        p.iter()
            .filter(|p| if let Patch::Delete = **p { true } else { false })
            .count()
            <= 1
    })
}

pub(crate) struct PatchIterator<'a> {
    vcode: &'a VCode<MInst>,
    patches: &'a VCodePatches,
    range: Box<dyn DoubleEndedIterator<Item = Inst>>,
    next_inst: Option<Inst>,
    next_before: Box<dyn 'a + Iterator<Item = &'a MInst>>,
    next_after: Box<dyn 'a + Iterator<Item = &'a MInst>>,
}

impl<'a> PatchIterator<'a> {
    pub(crate) fn new(
        vcode: &'a VCode<MInst>,
        patches: &'a VCodePatches,
        block: BlockIndex,
    ) -> Self {
        verify_patches(patches);
        Self {
            vcode,
            patches,
            range: Box::new(vcode.block_insns(block).iter()),
            next_inst: None,
            next_before: Box::new(std::iter::empty()),
            next_after: Box::new(std::iter::empty()),
        }
    }
}

impl<'a> Iterator for PatchIterator<'a> {
    type Item = (&'a MInst, Option<Inst>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(m) = self.next_before.next() {
            Some((m, None))
        } else if let Some(next_inst) = self.next_inst.take() {
            if !self
                .patches
                .get(&next_inst)
                .map_or(false, |ps| ps.iter().any(Patch::is_delete))
            {
                Some((self.vcode.index(next_inst), Some(next_inst)))
            } else {
                self.next()
            }
        } else if let Some(m) = self.next_after.next() {
            Some((m, None))
        } else {
            self.next_inst = self.range.next();
            if let Some(next_inst) = self.next_inst {
                self.next_before = Box::new(
                    self.patches
                        .get(&next_inst)
                        .map(|v| {
                            v.iter().filter_map(|p| {
                                if let Patch::Before(m) = p {
                                    Some(m)
                                } else {
                                    None
                                }
                            })
                        })
                        .into_iter()
                        .flatten(),
                );
                self.next_after = Box::new(
                    self.patches
                        .get(&next_inst)
                        .map(|v| {
                            v.iter().filter_map(|p| {
                                if let Patch::After(m) = p {
                                    Some(m)
                                } else {
                                    None
                                }
                            })
                        })
                        .into_iter()
                        .flatten(),
                );
                self.next()
            } else {
                None
            }
        }
    }
}
