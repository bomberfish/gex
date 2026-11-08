//! Assembling decoded instructions into a GIR basic block.

use crate::{FlagSource, Flow, Lifter, INST_BYTES};
use gex_ir::{pass, Block, Builder, Term};
use gex_isa::decode;

/// Bounds on how much a single block may cover.
#[derive(Copy, Clone, Debug)]
pub struct BlockLimits {
    /// Stop and fall through after this many instructions, so no single wasm
    /// function grows pathologically large.
    pub max_insts: usize,
}

impl Default for BlockLimits {
    fn default() -> Self {
        BlockLimits { max_insts: 256 }
    }
}

/// Lift the basic block starting at `start`.
///
/// `fetch` returns the instruction word at a guest address, or `None` when it
/// is not backed by translatable memory; an unfetchable address ends the block
/// with [`Term::Illegal`] rather than failing the build.
pub fn lift_block(fetch: impl Fn(u64) -> Option<u32>, start: u64, limits: BlockLimits) -> Block {
    let mut builder = Builder::new(start);
    let mut pc = start;
    let mut term = None;
    // A block's predecessor is unknown here, so the flags start unknown.
    let mut flags = FlagSource::Unknown;

    for _ in 0..limits.max_insts {
        let Some(raw) = fetch(pc) else {
            term = Some(Term::Illegal { raw: 0, pc });
            break;
        };
        let inst = decode(raw, pc);
        let mark = builder.mark();
        let mut lifter = Lifter::new(&mut builder).with_raw(raw).with_flags(flags);
        let flow = lifter.lift(&inst, pc);
        flags = lifter.flags();
        // An instruction with no wasm form is undone and becomes a SIGILL,
        // rather than panicking the backend: exhaustive discovery lifts
        // padding and literal pools a program would never execute.
        if builder.has_inexpressible(mark) {
            builder.rewind(mark);
            term = Some(Term::Illegal { raw, pc });
            break;
        }
        pc += INST_BYTES;
        builder.end_at(pc);
        match flow {
            Flow::Next => {}
            Flow::End(t) => {
                term = Some(t);
                break;
            }
        }
    }

    // Running out of budget is not an error: fall through to the next
    // instruction, which becomes its own block.
    let term = term.unwrap_or(Term::Goto { target: pc });
    let mut block = builder.finish(term);
    pass::optimize(&mut block);
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use gex_ir::{print, GuestReg, Stmt};

    /// `cmp x0, x1` then `b.<cc>`: nothing should read N/Z/C/V afterwards,
    /// since the branch asks about x0 and x1 directly. MI, PL, VS and VC are
    /// absent on purpose -- they have no direct form (see `cond_direct`).
    #[test]
    fn compare_and_branch_does_not_read_the_flags() {
        for cc in [
            0b0000u32, 0b0001, 0b0010, 0b0011, 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101,
        ] {
            let code = vec![0xeb01001f_u32, 0x54000040 | cc];
            let base = 0x1000u64;
            let fetch = |pc: u64| -> Option<u32> {
                if pc < base {
                    return None;
                }
                code.get(((pc - base) / 4) as usize).copied()
            };
            let block = lift_block(fetch, base, BlockLimits::default());
            let reads_flag = block.stmts.iter().any(|s| {
                matches!(
                    s,
                    Stmt::Def {
                        expr: gex_ir::Expr::ReadReg(
                            GuestReg::N | GuestReg::Z | GuestReg::C | GuestReg::V
                        ),
                        ..
                    }
                )
            });
            assert!(!reads_flag, "condition {cc:#06b} still reads a flag");
        }
    }

    /// The same, for the logical-op rule: `ands x0, x0, x1` sets N and Z from
    /// the result and clears C and V, so `b.eq` is `result == 0`.
    #[test]
    fn logical_and_branch_does_not_read_the_flags() {
        let code = vec![0xea010000_u32, 0x54000040];
        let base = 0x1000u64;
        let fetch = |pc: u64| -> Option<u32> {
            if pc < base {
                return None;
            }
            code.get(((pc - base) / 4) as usize).copied()
        };
        let block = lift_block(fetch, base, BlockLimits::default());
        assert!(!block.stmts.iter().any(|s| {
            matches!(
                s,
                Stmt::Def {
                    expr: gex_ir::Expr::ReadReg(
                        GuestReg::N | GuestReg::Z | GuestReg::C | GuestReg::V
                    ),
                    ..
                }
            )
        }));
    }

    /// Build a fetch closure over a slice of instruction words at `base`.
    fn words<'a>(base: u64, code: &'a [u32]) -> impl Fn(u64) -> Option<u32> + 'a {
        move |pc| {
            if pc < base {
                return None;
            }
            let idx = ((pc - base) / 4) as usize;
            code.get(idx).copied()
        }
    }

    #[test]
    fn straight_line_ends_at_branch() {
        // mov x0, #1 ; mov x1, #2 ; ret
        let b = lift_block(
            words(0x1000, &[0xd2800020, 0xd2800041, 0xd65f03c0]),
            0x1000,
            BlockLimits::default(),
        );
        assert_eq!(b.start, 0x1000);
        assert_eq!(b.end, 0x100c);
        assert!(matches!(b.term, Term::Indirect { is_ret: true, .. }));
    }

    #[test]
    fn conditional_branch_records_both_targets() {
        // cmp x0, #0 ; b.eq .+8
        let b = lift_block(
            words(0x2000, &[0xf100001f, 0x54000040]),
            0x2000,
            BlockLimits::default(),
        );
        match b.term {
            Term::Branch {
                taken, not_taken, ..
            } => {
                // The branch is the second instruction, at 0x2004.
                assert_eq!(taken, 0x200c);
                assert_eq!(not_taken, 0x2008);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bl_writes_the_return_address() {
        // bl .+16 from 0x3000
        let b = lift_block(words(0x3000, &[0x94000004]), 0x3000, BlockLimits::default());
        assert_eq!(b.term, Term::Goto { target: 0x3010 });
        let wrote_lr = b.stmts.iter().any(|s| {
            matches!(
                s,
                Stmt::WriteReg {
                    reg: GuestReg::X(30),
                    ..
                }
            )
        });
        assert!(wrote_lr, "bl must set x30:\n{}", print::block(&b));
    }

    #[test]
    fn unknown_instruction_becomes_illegal() {
        // A SIMD encoding: decoded but not yet lifted.
        let b = lift_block(words(0x4000, &[0x4ea11c20]), 0x4000, BlockLimits::default());
        assert!(matches!(b.term, Term::Illegal { .. }));
    }

    #[test]
    fn unfetchable_address_becomes_illegal() {
        let b = lift_block(|_| None, 0x5000, BlockLimits::default());
        assert_eq!(b.term, Term::Illegal { raw: 0, pc: 0x5000 });
    }

    /// Run a lifted block as a one-block memory-form function in gex-ir's
    /// reference interpreter, starting from `regs` (indexed by
    /// [`reg_index`]); returns the run's register file.
    fn run_block(b: &gex_ir::Block, regs: Vec<u64>) -> Vec<u64> {
        run_block_on(b, regs, &std::collections::HashMap::new()).regs
    }

    /// [`run_block`] on a memory that starts with `init`, returning the whole
    /// outcome (registers and the bytes written).
    fn run_block_on(
        b: &gex_ir::Block,
        regs: Vec<u64>,
        init: &std::collections::HashMap<u64, u8>,
    ) -> gex_ir::interp::Outcome {
        use gex_ir::func::{FBlock, FTerm, Form, Function};
        use gex_ir::{interp, ssa::CallEffect, GuestReg, Term};
        let f = Function {
            entry: b.start,
            name: None,
            entries: vec![],
            blocks: vec![FBlock {
                start: b.start,
                end: b.end,
                synthetic: false,
                params: vec![],
                stmts: b.stmts.clone(),
                pcs: vec![b.start; b.stmts.len()],
                insts: (b.start..b.end).step_by(4).collect(),
                term: match &b.term {
                    Term::Indirect { target, is_ret: true } => FTerm::Return { target: *target },
                    other => panic!("the test block must end in ret, got {other:?}"),
                },
                term_pc: b.end,
            }],
            n_values: b.n_values,
            form: Form::Memory,
        };
        let m = interp::Model {
            call: &|_| CallEffect::UNKNOWN,
            sp: GuestReg::Sp,
            max_steps: 100,
            syscall: None,
            narrow: false,
        };
        let out = interp::run_with_memory(&f, 0, regs, &m, init);
        assert!(
            matches!(out.stop, interp::Stop::Return(_)),
            "the run stopped at {:?}",
            out.stop
        );
        out
    }

    /// `casp` swaps the pair whole or not at all, and hands back the pair it
    /// read: both widths, with the pair matching, its first element only, and
    /// its second only. (Two independent cmpxchgs wrote the first element of
    /// a pair whose second did not match.)
    #[test]
    fn compare_swap_pair_is_all_or_nothing() {
        use gex_ir::func::reg_index;
        use gex_ir::{interp, GuestReg};
        use std::collections::HashMap;
        const AT: u64 = 0x10_0000;
        // casp x2, x3, x4, x5, [x1] and caspal w2, w3, w4, w5, [x1].
        for (word, bytes) in [(0x48227c24u32, 8u32), (0x0862fc24, 4)] {
            let mask = if bytes == 8 { u64::MAX } else { 0xffff_ffff };
            let (m0, m1) = (0x1111_2222_3333_4444 & mask, 0x5555_6666_7777_8888 & mask);
            let (n0, n1) = (0x9999_aaaa_bbbb_cccc & mask, 0xdddd_eeee_ffff_0001 & mask);
            for (e0, e1, swaps) in [(m0, m1, true), (m0, m1 ^ 1, false), (m0 ^ 1, m1, false)] {
                let b = lift_block(words(0x6000, &[word, 0xd65f03c0]), 0x6000, BlockLimits::default());
                let mut init = HashMap::new();
                for i in 0..bytes as u64 {
                    init.insert(AT + i, (m0 >> (8 * i)) as u8);
                    init.insert(AT + bytes as u64 + i, (m1 >> (8 * i)) as u8);
                }
                let mut regs = interp::initial_regs(0, GuestReg::Sp);
                regs[reg_index(GuestReg::X(1))] = AT;
                regs[reg_index(GuestReg::X(2))] = e0;
                regs[reg_index(GuestReg::X(3))] = e1;
                regs[reg_index(GuestReg::X(4))] = n0;
                regs[reg_index(GuestReg::X(5))] = n1;
                let out = run_block_on(&b, regs, &init);
                let mut mem = init.clone();
                mem.extend(out.memory.iter().map(|(&a, &v)| (a, v)));
                let elem = |k: u64| {
                    (0..bytes as u64).fold(0u64, |x, i| x | (mem[&(AT + k * bytes as u64 + i)] as u64) << (8 * i))
                };
                let want = if swaps { (n0, n1) } else { (m0, m1) };
                assert_eq!((elem(0), elem(1)), want, "{word:#010x}, expecting ({e0:#x}, {e1:#x})");
                assert_eq!(
                    (out.regs[reg_index(GuestReg::X(2))], out.regs[reg_index(GuestReg::X(3))]),
                    (m0, m1),
                    "{word:#010x}: the compare pair receives what memory held"
                );
            }
        }
    }

    /// `uaddlv h0, v0.8b` lifts (it raised SIGILL).
    #[test]
    fn add_long_across_lanes_lifts() {
        let code = vec![0x2e303800_u32, 0xd65f03c0];
        let base = 0x1000u64;
        let block = lift_block(words(base, &code), base, BlockLimits::default());
        assert!(!matches!(block.term, Term::Illegal { .. }), "uaddlv lifted as illegal");
    }

    /// One word of each kind that had no lowering goes on to the next
    /// instruction: lowered, or as a call to the host.
    #[test]
    fn surveyed_words_lift() {
        for raw in [
            0x4f80d021_u32, 0x0ea00c20, 0x0e60b484, 0x5f809020, 0x0f0e9c5c, 0x5f801822, 0x4f0e9cdc, 0x4e284801,
            0x2e20f400, 0x4e286800, 0x5e280820, 0x4e287800, 0x4e285801, 0x5e063128, 0x5e2818c7, 0x5e0442b6,
            0x5e045255, 0x5f805844, 0x5e006061, 0x5e282803, 0x2ea0f400, 0x5e021014, 0x0f0b961c, 0x5e020014,
            0x5e022014, 0x0e609062, 0x4ea05c21, 0x0ea12c42, 0x0ee1e000, 0x4ee1e000, 0x4f287400, 0x1e23c000,
            0x4f0b969c, 0x6f186695, 0x4ea154a3, 0x2ea5ee91, 0x0ea02842, 0x6ef8f802, 0x4f403041, 0x5f119f5a,
            0x5ea10c00, 0x5ea12c00,
        ] {
            let code = vec![raw, 0xd65f03c0];
            let base = 0x1000u64;
            let block = lift_block(words(base, &code), base, BlockLimits::default());
            assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
        }
    }

    /// The vector values of a lifted block, evaluated plainly (gex-ir's
    /// interpreter leaves V128 alone): only the expressions the saturating
    /// shifts lower to. Returns the V128 registers the block writes.
    fn eval_vectors(b: &gex_ir::Block, v: &[u128; 32]) -> std::collections::HashMap<u8, u128> {
        use gex_ir::{Expr, VBinOp, VLane, VLaneBinOp, VShiftOp};
        let mut val = std::collections::HashMap::<u32, u128>::new();
        let mut out = std::collections::HashMap::new();
        let w = |lane: VLane| match lane {
            VLane::I8 => 8u32,
            VLane::I16 => 16,
            VLane::I32 => 32,
            VLane::I64 => 64,
            l => panic!("{l:?} lanes"),
        };
        let lanes = |x: u128, bits: u32, f: &dyn Fn(u64) -> u64| -> u128 {
            let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let mut r = 0u128;
            for i in 0..128 / bits {
                let l = (x >> (i * bits)) as u64 & m;
                r |= ((f(l) & m) as u128) << (i * bits);
            }
            r
        };
        for st in &b.stmts {
            match st {
                Stmt::Def { dst, expr } => {
                    let g = |x: &gex_ir::Value| val[&x.0];
                    let r = match expr {
                        Expr::ReadReg(GuestReg::V128(n)) => v[*n as usize],
                        Expr::VConst(c) => u128::from_le_bytes(**c),
                        Expr::ConstI32(c) => *c as u128,
                        Expr::VShift { op, lane, a, amount } => {
                            let bits = w(*lane);
                            let n = g(amount) as u32 % bits;
                            let sx = |l: u64| ((l << (64 - bits)) as i64 >> (64 - bits)) as u64;
                            lanes(g(a), bits, &|l| match op {
                                VShiftOp::Shl => l << n,
                                VShiftOp::ShrU => l >> n,
                                VShiftOp::ShrS => (sx(l) as i64 >> n) as u64,
                            })
                        }
                        Expr::VLaneBin { op, lane, a, b } => {
                            let bits = w(*lane);
                            let (x, y) = (g(a), g(b));
                            let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                            let ul = |v: u128, i: u32| (v >> (i * bits)) as u64 & m;
                            let sl = |v: u128, i: u32| ((ul(v, i) << (64 - bits)) as i64 >> (64 - bits)) as i128;
                            let mut r = 0u128;
                            if let VLaneBinOp::ExtMulLowS | VLaneBinOp::ExtMulHighS = op {
                                // Half the lanes, each twice as wide.
                                let base = if *op == VLaneBinOp::ExtMulLowS { 0 } else { 64 / bits };
                                let wm = if bits == 32 { u128::from(u64::MAX) } else { (1u128 << (2 * bits)) - 1 };
                                for i in 0..64 / bits {
                                    let p = sl(x, base + i) * sl(y, base + i);
                                    r |= (p as u128 & wm) << (i * 2 * bits);
                                }
                                r
                            } else {
                                for i in 0..128 / bits {
                                    let (ux, uy, sx, sy) = (ul(x, i), ul(y, i), sl(x, i), sl(y, i));
                                    let all = |c: bool| if c { m } else { 0 };
                                    let out = match op {
                                        VLaneBinOp::Add => ux.wrapping_add(uy),
                                        VLaneBinOp::Sub => ux.wrapping_sub(uy),
                                        VLaneBinOp::Eq => all(ux == uy),
                                        VLaneBinOp::Ne => all(ux != uy),
                                        VLaneBinOp::LtU => all(ux < uy),
                                        VLaneBinOp::LtS => all(sx < sy),
                                        VLaneBinOp::GtS => all(sx > sy),
                                        VLaneBinOp::MinU => ux.min(uy),
                                        VLaneBinOp::MaxS => sx.max(sy) as u64,
                                        VLaneBinOp::Q15MulrSatS => {
                                            assert_eq!(bits, 16);
                                            ((sx * sy + 0x4000) >> 15).clamp(-0x8000, 0x7fff) as u64
                                        }
                                        o => panic!("not evaluated here: {o:?}"),
                                    };
                                    r |= ((out & m) as u128) << (i * bits);
                                }
                                r
                            }
                        }
                        Expr::VShuffle { a, b, lanes } => {
                            let (x, y) = (g(a).to_le_bytes(), g(b).to_le_bytes());
                            let mut o = [0u8; 16];
                            for (i, &l) in lanes.iter().enumerate() {
                                o[i] = if l < 16 { x[l as usize] } else { y[l as usize - 16] };
                            }
                            u128::from_le_bytes(o)
                        }
                        // wasm's narrows read signed lanes, `a`'s then `b`'s, and
                        // clamp them to the signed or unsigned narrow range.
                        Expr::VCvt { op: gex_ir::VCvtOp::NarrowSat { signed }, from, a, b } => {
                            let bits = w(*from);
                            let nb = bits / 2;
                            let (lo, hi) = if *signed { (-(1i128 << (nb - 1)), (1i128 << (nb - 1)) - 1) } else { (0, (1i128 << nb) - 1) };
                            let src = [g(a), g(&b.expect("two operands"))];
                            let mut r = 0u128;
                            for (k, v) in src.iter().enumerate() {
                                for i in 0..128 / bits {
                                    let l = (v >> (i * bits)) as u64 & ((1u64 << bits) - 1);
                                    let s = ((l << (64 - bits)) as i64 >> (64 - bits)) as i128;
                                    let n = s.clamp(lo, hi) as u128 & ((1u128 << nb) - 1);
                                    r |= n << ((k as u32 * (128 / bits) + i) * nb);
                                }
                            }
                            r
                        }
                        Expr::VBin { op, a, b } => match op {
                            VBinOp::And => g(a) & g(b),
                            VBinOp::Or => g(a) | g(b),
                            VBinOp::Xor => g(a) ^ g(b),
                            VBinOp::AndNot => g(a) & !g(b),
                        },
                        Expr::VBitSelect { mask, a, b } => (g(a) & g(mask)) | (g(b) & !g(mask)),
                        e => panic!("not evaluated here: {e:?}"),
                    };
                    val.insert(dst.0, r);
                }
                Stmt::WriteReg { reg: GuestReg::V128(n), val: x } => {
                    out.insert(*n, val[&x.0]);
                }
                Stmt::WriteReg { .. } => {}
                s => panic!("not evaluated here: {s:?}"),
            }
        }
        out
    }

    /// `sqshl`, `uqshl` and `sqshlu` by immediate, every lane width, both
    /// vector sizes and the scalar forms, against the instructions' reference
    /// semantics (`gex_isa::slow`) on edge and pseudo-random lanes.
    #[test]
    fn saturating_shifts_left_by_immediate_match_the_reference() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let reference = |raw: u32, v: &[u128; 32]| -> u128 {
            let mut bytes = [0u8; 512];
            for (i, r) in v.iter().enumerate() {
                bytes[i * 16..i * 16 + 16].copy_from_slice(&r.to_le_bytes());
            }
            let mut fpsr = 0;
            assert!(gex_isa::slow::execute(raw, &mut gex_isa::slow::VRegs(&mut bytes), &mut fpsr));
            u128::from_le_bytes(bytes[16..32].try_into().unwrap())
        };
        for (u, opcode) in [(0u32, 0b01110u32), (1, 0b01110), (1, 0b01100)] {
            for esize in [8u32, 16, 32, 64] {
                for shift in 0..esize {
                    for (q, scalar) in [(false, false), (true, false), (true, true)] {
                        // There is no `.1d` vector form.
                        if esize == 64 && !q {
                            continue;
                        }
                        // sqshl v1, v2, #shift (or its scalar form): 0 Q U 011110 immh:immb opcode 1 Rn Rd
                        let fields = (u << 29) | ((esize + shift) << 16) | (opcode << 11) | (1 << 10) | (2 << 5) | 1;
                        let raw = if scalar {
                            (1 << 30) | (0b111110 << 23) | fields
                        } else {
                            ((q as u32) << 30) | (0b011110 << 23) | fields
                        };
                        let block = lift_block(words(0x1000, &[raw]), 0x1000, BlockLimits { max_insts: 1 });
                        assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
                        for round in 0..24 {
                            let mut v = [0u128; 32];
                            v[1] = (rnd() as u128) << 64 | rnd() as u128;
                            let mut x = 0u128;
                            for i in 0..128 / esize {
                                let m = if esize == 64 { u64::MAX } else { (1u64 << esize) - 1 };
                                // Edges first: zero, one, all-ones, the signed limits and the
                                // values on either side of where the shift saturates.
                                let edge = [0u64, 1, m, m >> 1, (m >> 1) + 1, (m >> 1) >> shift, ((m >> 1) >> shift) + 1, !((m >> 1) >> shift)];
                                let l = if round < 4 { edge[(i as usize + 2 * round) % edge.len()] } else { rnd() >> (rnd() % 64) };
                                x |= ((l & m) as u128) << (i * esize);
                            }
                            v[2] = x;
                            // The reference runs the vector form; a scalar form is its lowest lane.
                            let vraw = if scalar { (raw & !(0b111110 << 23) & !(1 << 28)) | (0b011110 << 23) | (1 << 30) } else { raw };
                            let mut want = reference(vraw, &v);
                            if scalar {
                                want &= if esize == 64 { u64::MAX as u128 } else { (1u128 << esize) - 1 };
                            }
                            let got = eval_vectors(&block, &v)[&1];
                            assert_eq!(got, want, "{} {raw:08x} on {x:032x}", gex_isa::fmt::disasm(&decode::decode(raw, 0x1000), 0x1000));
                        }
                    }
                }
            }
        }
    }

    /// The slow path's own answer (`gex_isa::slow`, the instructions' reference
    /// semantics) for vector word `raw` on `v`: what it leaves in `v1`.
    fn slow_reference(raw: u32, v: &[u128; 32]) -> u128 {
        let mut bytes = [0u8; 512];
        for (i, r) in v.iter().enumerate() {
            bytes[i * 16..i * 16 + 16].copy_from_slice(&r.to_le_bytes());
        }
        let mut fpsr = 0;
        assert!(gex_isa::slow::execute(raw, &mut gex_isa::slow::VRegs(&mut bytes), &mut fpsr), "{raw:08x}");
        u128::from_le_bytes(bytes[16..32].try_into().unwrap())
    }

    /// A pseudo-random source of lanes, edges first: zero, one, all-ones, the
    /// signed limits and their neighbours, then `extra` (values the caller cares
    /// about, such as rounding ties), then random values of random magnitude.
    fn lane_values(esize: u32, extra: &[u64], rnd: &mut impl FnMut() -> u64) -> Vec<u64> {
        let m = if esize == 64 { u64::MAX } else { (1u64 << esize) - 1 };
        let max = m >> 1;
        let mut v = vec![0, 1, m, max, max - 1, max + 1, max + 2, 2, m - 1];
        v.extend(extra.iter().map(|x| x & m));
        for _ in 0..24 {
            v.push(rnd() >> (rnd() % 64) & m);
            v.push(!(rnd() >> (rnd() % 64)) & m);
        }
        v
    }

    /// Fills a register's lanes from `vals`, starting at `at` and wrapping.
    fn fill(esize: u32, vals: &[u64], at: usize) -> u128 {
        (0..128 / esize).fold(0u128, |x, i| x | (vals[(at + i as usize * 7) % vals.len()] as u128) << (i * esize))
    }

    fn xorshift(mut seed: u64) -> impl FnMut() -> u64 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        }
    }

    /// Lifts `raw` alone and checks it against the slow path's answer for `vraw`
    /// (the same instruction's vector form; for a scalar form only the low
    /// `scalar_bits` are kept) on every pair of `vals` lanes in `v2`, `v3` and `v1`.
    fn check_against_slow(raw: u32, vraw: u32, scalar_bits: Option<u32>, esize: u32, vals: &[u64], rnd: &mut impl FnMut() -> u64) {
        let block = lift_block(words(0x1000, &[raw]), 0x1000, BlockLimits { max_insts: 1 });
        assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
        // No host call: the word is lowered, not sent to the slow path.
        assert!(
            !block.stmts.iter().any(|s| matches!(s, Stmt::HostCall { .. })),
            "{raw:08x} still goes to the host"
        );
        let text = gex_isa::fmt::disasm(&decode::decode(raw, 0x1000), 0x1000);
        for round in 0..vals.len() {
            let mut v = [0u128; 32];
            v[1] = (rnd() as u128) << 64 | rnd() as u128;
            v[2] = fill(esize, vals, round);
            v[3] = fill(esize, vals, round * 5 + 3);
            if round % 3 == 0 {
                v[3] = v[2]; // squares: MIN * MIN among them
            }
            let mut want = slow_reference(vraw, &v);
            if let Some(b) = scalar_bits {
                want &= (1u128 << b) - 1;
            }
            let got = eval_vectors(&block, &v)[&1];
            assert_eq!(got, want, "{text} ({raw:08x}) on n={:032x} m={:032x} d={:032x}", v[2], v[3], v[1]);
        }
    }

    /// `sqadd`, `uqadd`, `sqsub`, `uqsub` on word and doubleword lanes, vector and
    /// scalar, against the slow path's reference: wasm saturates only narrower lanes.
    #[test]
    fn wide_saturating_add_and_subtract_match_the_reference() {
        let mut rnd = xorshift(0x2545_f491_4f6c_dd1d);
        for (u, opcode) in [(0u32, 0b00001u32), (1, 0b00001), (0, 0b00101), (1, 0b00101)] {
            for (size, esize) in [(2u32, 32u32), (3, 64)] {
                let vals = lane_values(esize, &[], &mut rnd);
                for (q, scalar) in [(false, false), (true, false), (true, true)] {
                    if esize == 64 && !q {
                        continue;
                    }
                    // sqadd v1, v2, v3: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd; scalar 01 U 11110 ...
                    let vraw = (1 << 30) | (u << 29) | (0b01110 << 24) | (size << 22) | (1 << 21) | (3 << 16) | (opcode << 11) | (1 << 10) | (2 << 5) | 1;
                    let raw = if scalar { vraw | (1 << 28) } else if q { vraw } else { vraw & !(1 << 30) };
                    check_against_slow(raw, raw & !(1 << 28), scalar.then_some(esize), esize, &vals, &mut rnd);
                }
            }
        }
    }

    /// `sqdmulh` and `sqrdmulh`, halfword and word lanes, vector, scalar and by
    /// element (every index, and the high register bit only words can use).
    #[test]
    fn doubling_multiplies_match_the_reference() {
        let mut rnd = xorshift(0x9e37_79b9_7f4a_7c15);
        for (size, esize) in [(1u32, 16u32), (2, 32)] {
            let min = 1u64 << (esize - 1);
            let vals = lane_values(esize, &[min, min + 1, min - 1, min >> 1, (min >> 1) + 1], &mut rnd);
            for u in [0u32, 1] {
                for (q, scalar) in [(false, false), (true, false), (true, true)] {
                    // sq(r)dmulh v1, v2, v3: 0 Q U 01110 size 1 Rm 10110 1 Rn Rd
                    let vraw = (1 << 30) | (u << 29) | (0b01110 << 24) | (size << 22) | (1 << 21) | (3 << 16) | (0b10110 << 11) | (1 << 10) | (2 << 5) | 1;
                    let raw = if scalar { vraw | (1 << 28) } else if q { vraw } else { vraw & !(1 << 30) };
                    check_against_slow(raw, raw & !(1 << 28), scalar.then_some(esize), esize, &vals, &mut rnd);
                }
                // By element: 0 Q 0 01111 size L M Rm 110U H 0 Rn Rd, the index in H:L(:M).
                for q in [false, true] {
                    for index in 0..128 / esize {
                        for rm in [3u32, 19] {
                            let (h, l, m, rm_field) = if esize == 16 {
                                if rm >= 16 {
                                    continue;
                                }
                                (index >> 2, index >> 1 & 1, index & 1, rm)
                            } else {
                                (index >> 1, index & 1, rm >> 4, rm & 15)
                            };
                            let raw = ((q as u32) << 30) | (0b01111 << 24) | (size << 22) | (l << 21) | (m << 20) | (rm_field << 16) | ((0b1100 | u) << 12) | (h << 11) | (2 << 5) | 1;
                            let block = lift_block(words(0x1000, &[raw]), 0x1000, BlockLimits { max_insts: 1 });
                            assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
                            for round in 0..vals.len() {
                                let mut v = [0u128; 32];
                                v[1] = rnd() as u128;
                                v[2] = fill(esize, &vals, round);
                                v[rm as usize] = fill(esize, &vals, round * 3 + 1);
                                let want = slow_reference(raw, &v);
                                let got = eval_vectors(&block, &v)[&1];
                                assert_eq!(got, want, "{raw:08x} index {index} on n={:032x} m={:032x}", v[2], v[rm as usize]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// `sqshrn`, `sqrshrn`, `uqshrn`, `uqrshrn`, `sqshrun` and `sqrshrun`: every
    /// width, every shift, the lower and upper (`2`) halves and the scalar forms,
    /// with rounding ties and the values either side of saturation among the lanes.
    #[test]
    fn saturating_narrowing_shifts_match_the_reference() {
        let mut rnd = xorshift(0xd1b5_4a32_d192_ed03);
        for (u, opcode) in [(0u32, 0b10010u32), (0, 0b10011), (1, 0b10010), (1, 0b10011), (1, 0b10000), (1, 0b10001)] {
            for esize in [8u32, 16, 32] {
                for shift in 1..=esize {
                    let wide = 2 * esize;
                    let wm = (1u128 << wide) - 1;
                    let mut extra = Vec::new();
                    for k in [0u64, 1, 2, 3] {
                        // Ties, either side of them, and the saturation edges after the shift.
                        let half = 1u128 << (shift - 1);
                        for base in [(k as u128) << shift, (((1u128 << (esize - 1)) - k as u128) << shift), (((1u128 << esize) - k as u128) << shift)] {
                            for d in [half, half - 1, half + 1, 0] {
                                extra.push(((base + d) & wm) as u64);
                                extra.push(((base.wrapping_neg().wrapping_add(d)) & wm) as u64);
                            }
                        }
                    }
                    let vals = lane_values(wide, &extra, &mut rnd);
                    for (q, scalar) in [(false, false), (true, false), (true, true)] {
                        // sqshrn v1, v2, #shift: 0 Q U 011110 immh:immb opcode 1 Rn Rd; scalar 01 U 111110 ...
                        let imm = 2 * esize - shift;
                        let fields = (u << 29) | (imm << 16) | (opcode << 11) | (1 << 10) | (2 << 5) | 1;
                        let vform = |q: bool| ((q as u32) << 30) | (0b011110 << 23) | fields;
                        let raw = if scalar { (1 << 30) | (0b111110 << 23) | fields } else { vform(q) };
                        // The reference runs the low-half vector form for a scalar one.
                        let vraw = if scalar { vform(false) } else { raw };
                        let block = lift_block(words(0x1000, &[raw]), 0x1000, BlockLimits { max_insts: 1 });
                        assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
                        let text = gex_isa::fmt::disasm(&decode::decode(raw, 0x1000), 0x1000);
                        for round in 0..vals.len() {
                            let mut v = [0u128; 32];
                            v[1] = (rnd() as u128) << 64 | rnd() as u128;
                            v[2] = fill(wide, &vals, round);
                            let mut want = slow_reference(vraw, &v);
                            if scalar {
                                want &= (1u128 << esize) - 1;
                            }
                            let got = eval_vectors(&block, &v)[&1];
                            assert_eq!(got, want, "{text} ({raw:08x}) on n={:032x} d={:032x}", v[2], v[1]);
                        }
                    }
                }
            }
        }
    }

    /// The saturating fixed-point words lift to wasm SIMD with no host call.
    #[test]
    fn hot_saturating_words_are_lowered() {
        for raw in [
            0x4ea00c00_u32, // sqadd v0.4s, v0.4s, v0.4s
            0x0ea00c00,     // sqadd v0.2s
            0x0ea02c00,     // sqsub v0.2s
            0x4f80d000,     // sqrdmulh v0.4s, v0.4s, v0.s[0]
            0x6ea0b400,     // sqrdmulh v0.4s, v0.4s, v0.4s
            0x0f209c00,     // sqrshrn v0.2s, v0.2d, #32
            0x4f209c00,     // sqrshrn2 v0.4s, v0.2d, #32
            0x0f209400,     // sqshrn v0.2s, v0.2d, #32
        ] {
            let block = lift_block(words(0x1000, &[raw]), 0x1000, BlockLimits { max_insts: 1 });
            assert!(!matches!(block.term, Term::Illegal { .. }), "{raw:08x} lifted as illegal");
            assert!(
                !block.stmts.iter().any(|s| matches!(s, Stmt::HostCall { .. })),
                "{raw:08x} still goes to the host"
            );
        }
    }

    /// `udf` and data stay refusals: only understood words reach the host.
    #[test]
    fn what_nothing_understands_stays_illegal() {
        let block = lift_block(words(0x1000, &[0, 0xd65f03c0]), 0x1000, BlockLimits::default());
        assert!(matches!(block.term, Term::Illegal { raw: 0, .. }));
    }

    /// The CRC32 / CRC32C expansion: known-answer vectors ("123456789") and
    /// a plain reference bit loop over a few inputs.
    #[test]
    fn crc32_expansion_matches_the_architecture() {
        use gex_ir::func::{reg_index, N_REGS};
        use gex_ir::{interp, GuestReg};

        // The reference: the same reflected bit loop the expansion is
        // written from, but plainly.
        fn crc_ref(mut acc: u32, src: u64, nbytes: u32, poly: u32) -> u32 {
            for i in 0..nbytes {
                acc ^= ((src >> (8 * i)) & 0xff) as u32;
                for _ in 0..8 {
                    acc = (acc >> 1) ^ (poly & 0u32.wrapping_sub(acc & 1));
                }
            }
            acc
        }
        const IEEE: u32 = 0xedb8_8320; // 0x04C11DB7 reversed
        const CASTAGNOLI: u32 = 0x82f6_3b78; // 0x1EDC6F41 reversed

        // crc32x w0, w0, x1 ; crc32b w0, w0, w2 ; ret -- the nine bytes of
        // "123456789" through the widest form and then the last byte.
        let run = |word_x: u32, word_b: u32, acc: u32, src: u64, last: u32| -> u32 {            let b = lift_block(
                words(0x6000, &[word_x, word_b, 0xd65f03c0]),
                0x6000,
                BlockLimits::default(),
            );
            let mut regs = interp::initial_regs(0, GuestReg::Sp);
            assert_eq!(regs.len(), N_REGS);
            regs[reg_index(GuestReg::X(0))] = acc as u64;
            regs[reg_index(GuestReg::X(1))] = src;
            regs[reg_index(GuestReg::X(2))] = last as u64;
            run_block(&b, regs)[reg_index(GuestReg::X(0))] as u32
        };
        // "12345678" little-endian in x1, '9' = 0x39 in x2.
        let s = u64::from_le_bytes(*b"12345678");
        // CRC-32/ISO-HDLC of "123456789" is 0xCBF43926 with init and xorout
        // 0xFFFFFFFF: from a 0xFFFFFFFF running CRC, the instruction's form
        // ends at the check value xored with 0xFFFFFFFF.
        assert_eq!(
            run(0x9ac14c00, 0x1ac24000, 0xffff_ffff, s, 0x39),
            0xcbf4_3926 ^ 0xffff_ffff,
            "crc32x + crc32b"
        );
        // CRC-32/ISCSI (Castagnoli) of "123456789" is 0xE3069283.
        assert_eq!(
            run(0x9ac15c00, 0x1ac25000, 0xffff_ffff, s, 0x39),
            0xe306_9283 ^ 0xffff_ffff,
            "crc32cx + crc32cb"
        );

        // The other widths against the reference: halfwords and words take
        // the low 2/4 bytes of their source only, and each form's byte count
        // is its own.
        for (word, nbytes, poly) in [
            (0x1ac14000u32, 1, IEEE),        // crc32b w0, w0, w1
            (0x1ac14400, 2, IEEE),           // crc32h
            (0x1ac14800, 4, IEEE),           // crc32w
            (0x9ac14c00, 8, IEEE),           // crc32x
            (0x1ac15000, 1, CASTAGNOLI),     // crc32cb
            (0x1ac15400, 2, CASTAGNOLI),     // crc32ch
            (0x1ac15800, 4, CASTAGNOLI),     // crc32cw
            (0x9ac15c00, 8, CASTAGNOLI),     // crc32cx
        ] {
            for (acc, src) in [(0u32, 0u64), (0xffff_ffff, 0x0123_4567_89ab_cdef), (0x1234_5678, u64::MAX)] {
                let b = lift_block(
                    words(0x6000, &[word, 0xd65f03c0]),
                    0x6000,
                    BlockLimits::default(),
                );
                let mut regs = interp::initial_regs(0, GuestReg::Sp);
                regs[reg_index(GuestReg::X(0))] = acc as u64;
                regs[reg_index(GuestReg::X(1))] = src;
                let got = run_block(&b, regs)[reg_index(GuestReg::X(0))] as u32;
                assert_eq!(
                    got,
                    crc_ref(acc, src, nbytes, poly),
                    "word {word:#010x} of {nbytes} byte(s), acc {acc:#010x}, src {src:#018x}"
                );
            }
        }
    }

    #[test]
    fn block_splits_at_the_instruction_budget() {
        // 300 nops: longer than the default budget, no terminator in sight.
        static NOPS: [u32; 300] = [0xd503201f; 300];
        let b = lift_block(
            |pc| {
                let idx = ((pc - 0x6000) / 4) as usize;
                NOPS.get(idx).copied()
            },
            0x6000,
            BlockLimits { max_insts: 64 },
        );
        assert_eq!(b.end, 0x6000 + 64 * 4);
        assert_eq!(
            b.term,
            Term::Goto {
                target: 0x6000 + 64 * 4
            }
        );
    }

    #[test]
    fn dead_flags_are_eliminated_across_compares() {
        // Two compares in a row, then a conditional branch: only the second
        // compare's flags can be observed.
        // cmp x0,#1 ; cmp x1,#2 ; b.eq .+8
        let b = lift_block(
            words(0x7000, &[0xf100041f, 0xf100085f, 0x54000040]),
            0x7000,
            BlockLimits::default(),
        );
        let flag_writes = b
            .stmts
            .iter()
            .filter(|s| matches!(s, Stmt::WriteReg { reg, .. } if reg.is_flag()))
            .count();
        // Four flags written once, not twice.
        assert_eq!(flag_writes, 4, "{}", print::block(&b));
    }
}
