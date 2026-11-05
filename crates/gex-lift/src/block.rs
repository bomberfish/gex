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
        let out = interp::run_from(&f, 0, regs, &m);
        assert!(
            matches!(out.stop, interp::Stop::Return(_)),
            "the run stopped at {:?}",
            out.stop
        );
        out.regs
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
