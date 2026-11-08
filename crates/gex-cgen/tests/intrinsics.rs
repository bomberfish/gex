//! Every SIMD operation the C side names must lower to the very instruction
//! tier-1 emits for it: compile one tiny function per (op, lane) with clang
//! and look for the opcode in its body.

use gex_ir::{VLane, VLaneBinOp, VLaneUnOp};
use std::process::Command;

const LANES: [VLane; 6] = [VLane::I8, VLane::I16, VLane::I32, VLane::I64, VLane::F32, VLane::F64];

fn leb(mut v: u32) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            return out;
        }
    }
}

#[test]
fn lane_ops_lower_to_tier1_opcodes() {
    use VLaneBinOp::*;
    use VLaneUnOp::*;
    let bins = [
        Add, Sub, Mul, Div, MinS, MinU, MaxS, MaxU, AddSatS, AddSatU, SubSatS, SubSatU, AvgrU, Q15MulrSatS, ExtMulLowS, ExtMulHighS, Eq,
        Ne, LtS, LtU, GtS, GtU, LeS, LeU, GeS, GeU,
    ];
    let uns = [Neg, Abs, Sqrt, Ceil, Floor, Trunc, Nearest];
    let mut src = String::from("#include <wasm_simd128.h>\n");
    let mut want: Vec<(String, u32)> = Vec::new();
    for lane in LANES {
        for op in bins {
            if let Some(code) = op.wasm_opcode(lane) {
                let f = gex_cgen::emit::lane_bin_name(op, lane).unwrap();
                let name = format!("t{}", want.len());
                src += &format!("v128_t {name}(v128_t a, v128_t b) {{ return {f}(a, b); }}\n");
                want.push((format!("{op:?} {lane:?} via {f}"), code));
            }
        }
        for op in uns {
            if let Some(code) = op.wasm_opcode(lane) {
                let f = gex_cgen::emit::lane_un_name(op, lane).unwrap();
                let name = format!("t{}", want.len());
                src += &format!("v128_t {name}(v128_t a) {{ return {f}(a); }}\n");
                want.push((format!("{op:?} {lane:?} via {f}"), code));
            }
        }
    }
    let dir = std::env::temp_dir().join(format!("gex-cgen-intrinsics-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("t.c");
    let o = dir.join("t.o");
    std::fs::write(&c, src).unwrap();
    let st = Command::new(gex_cgen::toolchain::clang())
        .args(gex_cgen::toolchain::CFLAGS)
        .arg("-c")
        .arg(&c)
        .arg("-o")
        .arg(&o)
        .status()
        .expect("clang");
    assert!(st.success());
    let wasm = std::fs::read(&o).unwrap();
    let mut bodies = Vec::new();
    for p in wasmparser::Parser::new(0).parse_all(&wasm) {
        if let wasmparser::Payload::CodeSectionEntry(b) = p.unwrap() {
            bodies.push(b.as_bytes().to_vec());
        }
    }
    assert_eq!(bodies.len(), want.len());
    let mut bad = Vec::new();
    for ((what, code), body) in want.iter().zip(&bodies) {
        let mut pat = vec![0xfd];
        pat.extend(leb(*code));
        if !body.windows(pat.len()).any(|w| w == pat.as_slice()) {
            bad.push(format!("{what}: want 0xfd {code:#x}, body {body:02x?}"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The scalar floating-point builtins the C side uses, and the one wasm
/// instruction each must become (no libm call, no expansion).
#[test]
fn scalar_fp_builtins_are_single_instructions() {
    let cases: &[(&str, &str, &[u8])] = &[
        ("float", "__builtin_roundevenf(a)", &[0x90]),
        ("double", "__builtin_roundeven(a)", &[0x9e]),
        ("float", "__builtin_sqrtf(a)", &[0x91]),
        ("double", "__builtin_sqrt(a)", &[0x9f]),
        ("float", "__builtin_ceilf(a)", &[0x8d]),
        ("double", "__builtin_floor(a)", &[0x9c]),
        ("double", "__builtin_trunc(a)", &[0x9d]),
        ("float", "__builtin_wasm_min_f32(a, b)", &[0x96]),
        ("double", "__builtin_wasm_max_f64(a, b)", &[0xa5]),
        ("double", "__builtin_copysign(a, b)", &[0xa6]),
        ("float", "__builtin_wasm_trunc_saturate_s_i32_f32(a)", &[0xfc, 0x00]),
        ("double", "__builtin_wasm_trunc_saturate_u_i64_f64(a)", &[0xfc, 0x07]),
    ];
    let mut src = String::new();
    for (i, (ty, e, _)) in cases.iter().enumerate() {
        let rt = if e.contains("_i32_") {
            "int"
        } else if e.contains("_i64_") {
            "long long"
        } else {
            ty
        };
        src += &format!("{rt} t{i}({ty} a, {ty} b) {{ return {e}; }}\n");
    }
    let dir = std::env::temp_dir().join(format!("gex-cgen-fp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (c, o) = (dir.join("t.c"), dir.join("t.o"));
    std::fs::write(&c, src).unwrap();
    let st = Command::new(gex_cgen::toolchain::clang())
        .args(gex_cgen::toolchain::CFLAGS)
        .arg("-c")
        .arg(&c)
        .arg("-o")
        .arg(&o)
        .status()
        .expect("clang");
    assert!(st.success());
    let wasm = std::fs::read(&o).unwrap();
    let mut bodies = Vec::new();
    for p in wasmparser::Parser::new(0).parse_all(&wasm) {
        if let wasmparser::Payload::CodeSectionEntry(b) = p.unwrap() {
            let mut ops = b.get_operators_reader().unwrap();
            let mut n = 0;
            while !ops.eof() {
                let op = ops.read().unwrap();
                assert!(!matches!(op, wasmparser::Operator::Call { .. }), "a libcall");
                n += 1;
            }
            bodies.push((b.as_bytes().to_vec(), n));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    for ((_, e, code), (body, n)) in cases.iter().zip(&bodies) {
        assert!(
            body.windows(code.len()).any(|w| w == *code),
            "{e}: want {code:02x?}, body {body:02x?}"
        );
        // local.get(s), the instruction, end.
        assert!(*n <= 4, "{e} expanded to {n} instructions: {body:02x?}");
    }
}
