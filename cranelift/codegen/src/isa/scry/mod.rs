//! Scry Instruction Set Architecture.

use crate::dominator_tree::DominatorTree;
use crate::ir::{Function, Type};
use crate::isa::scry::settings as scry_settings;
use crate::isa::{
    Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey, OwnedTargetIsa, TargetIsa,
};
use crate::machinst::{
    CompiledCode, CompiledCodeStencil, MachInst, MachTextSectionBuilder, Reg,
    TextSectionBuilder,
};
use crate::result::CodegenResult;
use crate::settings::{self as shared_settings, Flags};
use crate::ir;
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use cranelift_control::ControlPlane;
use target_lexicon::{Architecture, Triple};
use crate::isa::unwind::systemv;

mod abi;
pub(crate) mod inst;
mod lower;
mod settings;


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

}

impl TargetIsa for ScryBackend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        unimplemented!()
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
        result: &CompiledCode,
        kind: crate::isa::unwind::UnwindInfoKind,
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
    fn map_regalloc_reg_to_dwarf(&self, reg: Reg) -> Result<u16, systemv::RegisterMappingError> {
        unimplemented!()
    }

    fn function_alignment(&self) -> FunctionAlignment {
        inst::MInst::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        unimplemented!()
    }

    fn pretty_print_reg(&self, reg: Reg, _size: u8) -> String {
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
        ir::ArgumentExtension::Sext
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
        Architecture::Scry => {}
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
