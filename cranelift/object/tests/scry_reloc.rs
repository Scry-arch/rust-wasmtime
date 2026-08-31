//! Object-emission tests for the Scry backend: exercise the translation of
//! Cranelift relocations (ScryAbs32, ...) into ELF relocation records — the
//! layer between MachBuffer relocs and the `object` crate's writer
//! (EM_SCRY = 264). Filetests stop before this layer; these tests run it and
//! parse the emitted ELF back to assert the records.
//!
//! Functions are written as textual CLIF (parsed with cranelift-reader, like
//! the filetests). One wrinkle: cranelift-module resolves relocations through
//! module-registered user names (`u0:N` = FuncId N, `u1:N` = DataId N), while
//! parsed text carries free `%testcase` names — `bind_symbols` rewrites the
//! latter into the former, which is what `declare_*_in_func` would have done.

use cranelift_codegen::ir::{ExternalName, Function, GlobalValueData, UserExternalName};
use cranelift_codegen::{Context, settings};
use cranelift_module::{DataId, FuncId, Linkage, Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::read::{Object, ObjectSection, ObjectSymbol};
use object::{RelocationFlags, RelocationTarget};

fn scry_module(name: &str) -> ObjectModule {
    let isa = cranelift_codegen::isa::lookup_by_name("scry32-unknown-none-elf")
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();
    ObjectModule::new(ObjectBuilder::new(isa, name, default_libcall_names()).unwrap())
}

fn parse(text: &str) -> Function {
    let mut funcs = cranelift_reader::parse_functions(text).unwrap();
    assert_eq!(funcs.len(), 1);
    funcs.pop().unwrap()
}

fn testcase_is(name: &ExternalName, want: &str) -> bool {
    match name {
        ExternalName::TestCase(tc) => tc.to_string().trim_start_matches('%') == want,
        _ => false,
    }
}

/// Rewrite `%name` references in a parsed function to the module's user
/// external names: namespace 0 for functions, namespace 1 for data.
fn bind_symbols(func: &mut Function, funcs: &[(&str, FuncId)], data: &[(&str, DataId)]) {
    let ext_funcs: Vec<_> = func.dfg.ext_funcs.keys().collect();
    for fr in ext_funcs {
        for (name, id) in funcs {
            if testcase_is(&func.dfg.ext_funcs[fr].name, name) {
                let r = func.declare_imported_user_function(UserExternalName {
                    namespace: 0,
                    index: id.as_u32(),
                });
                func.dfg.ext_funcs[fr].name = ExternalName::User(r);
            }
        }
    }
    let gvs: Vec<_> = func.global_values.keys().collect();
    for gv in gvs {
        for (name, id) in data {
            let matches = matches!(
                &func.global_values[gv],
                GlobalValueData::Symbol { name: n, .. } if testcase_is(n, name)
            );
            if matches {
                let r = func.declare_imported_user_function(UserExternalName {
                    namespace: 1,
                    index: id.as_u32(),
                });
                match &mut func.global_values[gv] {
                    GlobalValueData::Symbol { name: n, .. } => *n = ExternalName::User(r),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Every relocation record in the emitted object, as
/// (offset-in-section, flags, addend, target symbol name).
fn collect_relocs(bytes: &[u8]) -> Vec<(u64, RelocationFlags, i64, String)> {
    let file = object::read::File::parse(bytes).unwrap();
    let mut out = vec![];
    for section in file.sections() {
        for (offset, reloc) in section.relocations() {
            let name = match reloc.target() {
                RelocationTarget::Symbol(idx) => file
                    .symbol_by_index(idx)
                    .unwrap()
                    .name()
                    .unwrap()
                    .to_string(),
                other => panic!("unexpected relocation target {other:?}"),
            };
            out.push((offset, reloc.flags(), reloc.addend(), name));
        }
    }
    out
}

fn assert_elf_reloc(flags: &RelocationFlags, r_type: object::elf::RelocationType) {
    assert_eq!(
        *flags,
        RelocationFlags::Elf { r_type },
        "unexpected ELF relocation type"
    );
}

/// A call to an imported function must produce a relocation against the
/// callee symbol.
#[test]
fn call_to_import_emits_reloc() {
    let mut module = scry_module("call_reloc");
    let mut func = parse(
        r#"
        function %caller() system_v {
            sig0 = () system_v
            fn0 = %callee sig0
        block0:
            call fn0()
            return
        }"#,
    );

    let callee = module
        .declare_function(
            "callee",
            Linkage::Import,
            &func.dfg.signatures.values().next().unwrap().clone(),
        )
        .unwrap();
    let caller = module
        .declare_function("caller", Linkage::Export, &func.signature)
        .unwrap();
    bind_symbols(&mut func, &[("callee", callee)], &[]);

    let mut ctx = Context::for_function(func);
    module.define_function(caller, &mut ctx).unwrap();

    let bytes = module.finish().emit().unwrap();
    let relocs = collect_relocs(&bytes);
    let (_, flags, addend, _) = relocs
        .iter()
        .find(|r| r.3 == "callee")
        .expect("a relocation against `callee`");
    assert_elf_reloc(flags, object::elf::R_SCRY_ABS32);
    assert_eq!(*addend, 0);
}

/// symbol_value of an imported static must produce a relocation against the
/// data symbol.
#[test]
fn symbol_value_emits_reloc() {
    let mut module = scry_module("symval_reloc");
    let mut func = parse(
        r#"
        function %takes_addr() -> i32 system_v {
            gv0 = symbol %my_static
        block0:
            v0 = symbol_value.i32 gv0
            return v0
        }"#,
    );

    let data = module
        .declare_data("my_static", Linkage::Import, false, false)
        .unwrap();
    let func_id = module
        .declare_function("takes_addr", Linkage::Export, &func.signature)
        .unwrap();
    bind_symbols(&mut func, &[], &[("my_static", data)]);

    let mut ctx = Context::for_function(func);
    module.define_function(func_id, &mut ctx).unwrap();

    let bytes = module.finish().emit().unwrap();
    let relocs = collect_relocs(&bytes);
    let (_, flags, addend, _) = relocs
        .iter()
        .find(|r| r.3 == "my_static")
        .expect("a relocation against `my_static`");
    assert_elf_reloc(flags, object::elf::R_SCRY_ABS32);
    assert_eq!(*addend, 0);
}

/// symbol+offset (as rustc emits for fields of statics) must carry the offset
/// as the relocation addend. This pins the REL-vs-RELA decision to RELA:
/// with the address immediate scattered over const+grow 8-bit fields,
/// storing addends in-place would be miserable for the linker, keep them in
/// the relocation record.
#[test]
fn symbol_value_offset_becomes_addend() {
    let mut module = scry_module("symval_addend");
    let mut func = parse(
        r#"
        function %takes_addr4() -> i32 system_v {
            gv0 = symbol %my_static+4
        block0:
            v0 = symbol_value.i32 gv0
            return v0
        }"#,
    );

    let data = module
        .declare_data("my_static", Linkage::Import, false, false)
        .unwrap();
    let func_id = module
        .declare_function("takes_addr4", Linkage::Export, &func.signature)
        .unwrap();
    bind_symbols(&mut func, &[], &[("my_static", data)]);

    let mut ctx = Context::for_function(func);
    module.define_function(func_id, &mut ctx).unwrap();

    let bytes = module.finish().emit().unwrap();
    let relocs = collect_relocs(&bytes);
    let (_, flags, addend, _) = relocs
        .iter()
        .find(|r| r.3 == "my_static")
        .expect("a relocation against `my_static`");
    assert_elf_reloc(flags, object::elf::R_SCRY_ABS32);
    assert_eq!(*addend, 4, "symbol+4 must land in the RELA addend");
}

use cranelift_module::DataDescription;

/// A data object containing the address of another data symbol (as rustc
/// emits for the `&str` file-name field inside #[track_caller] Location
/// statics). This exercises the *data-object* relocation path
/// (`define_data` -> backend.rs data route), which is separate from the
/// function-section path above. The +4 offset also pins RELA on this path.
#[test]
fn data_holding_data_addr_emits_reloc() {
    let mut module = scry_module("data_reloc");
    let target = module
        .declare_data("target", Linkage::Import, false, false)
        .unwrap();
    let holder = module
        .declare_data("holder", Linkage::Export, true, false)
        .unwrap();

    let mut desc = DataDescription::new();
    desc.define(Box::new([0u8; 8]));
    let gv = module.declare_data_in_data(target, &mut desc);
    desc.write_data_addr(4, gv, 4);
    module.define_data(holder, &desc).unwrap();

    let bytes = module.finish().emit().unwrap();
    let relocs = collect_relocs(&bytes);
    let (_, flags, addend, _) = relocs
        .iter()
        .find(|r| r.3 == "target")
        .expect("a relocation against `target`");
    assert_elf_reloc(flags, object::elf::R_SCRY_32);
    assert_eq!(*addend, 4);
}

/// A data object containing a function address (vtables, fn-pointer statics).
#[test]
fn data_holding_func_addr_emits_reloc() {
    let mut module = scry_module("data_fn_reloc");
    let mut func = parse(
        r#"
        function %pointee() system_v {
        block0:
            return
        }"#,
    );
    let pointee = module
        .declare_function("pointee", Linkage::Export, &func.signature)
        .unwrap();
    let holder = module
        .declare_data("holder", Linkage::Export, true, false)
        .unwrap();

    let mut ctx = Context::for_function(func);
    module.define_function(pointee, &mut ctx).unwrap();

    let mut desc = DataDescription::new();
    desc.define(Box::new([0u8; 4]));
    let fref = module.declare_func_in_data(pointee, &mut desc);
    desc.write_function_addr(0, fref);
    module.define_data(holder, &desc).unwrap();

    let bytes = module.finish().emit().unwrap();
    let relocs = collect_relocs(&bytes);
    let (_, flags, addend, _) = relocs
        .iter()
        .find(|r| r.3 == "pointee")
        .expect("a relocation against `pointee`");
    assert_elf_reloc(flags, object::elf::R_SCRY_32);
    assert_eq!(*addend, 0);
}
