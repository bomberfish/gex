//! Assembling decoded x86-64 instructions into a GIR basic block.

use crate::{FlagSource, Flow, Lifter};
use gex_ir::{pass, Block, Builder, Term};
use gex_isa_x86::{decode, MAX_INST_BYTES};

/// Bounds on how much a single block may cover.
#[derive(Copy, Clone, Debug)]
pub struct BlockLimits {
    /// Stop and fall through after this many instructions.
    pub max_insts: usize,
}

impl Default for BlockLimits {
    fn default() -> Self {
        BlockLimits { max_insts: 256 }
    }
}

/// Lift the basic block starting at `start`.
///
/// `fetch(pc, buf)` copies the bytes of executable memory at `pc` into `buf`
/// and returns how many it could provide (up to 16; an x86 instruction is at
/// most 15). Zero means the address is not backed by translatable memory, and
/// the block ends there with [`Term::Illegal`].
pub fn lift_block(
    fetch: impl Fn(u64, &mut [u8; 16]) -> usize,
    start: u64,
    limits: BlockLimits,
) -> Block {
    let mut builder = Builder::new(start);
    let mut pc = start;
    let mut term = None;
    let mut flags = FlagSource::Unknown;

    for _ in 0..limits.max_insts {
        let mut buf = [0u8; 16];
        let n = fetch(pc, &mut buf).min(MAX_INST_BYTES);
        if n == 0 {
            term = Some(Term::Illegal { raw: 0, pc });
            break;
        }
        let inst = decode(&buf[..n], pc);
        let raw = u32::from_le_bytes(buf[..4].try_into().unwrap());
        if trace_range().is_some_and(|(lo, hi)| pc >= lo && pc < hi) {
            // GEX_TRACE_INSNS: the state before this instruction, to the
            // host (sysreg::TRACE), for comparing against a reference trace.
            let pcv = builder.const_i64(pc);
            let id = builder.const_i32(crate::sysreg::TRACE as u32);
            builder.stmt(gex_ir::Stmt::HostCall { func: gex_ir::HostFunc::SysRegWrite, args: vec![pcv, id, pcv] });
            flags = FlagSource::Unknown;
        }
        let mark = builder.mark();
        let mut lifter = Lifter::new(&mut builder, flags, raw);
        let flow = match (pc == start).then(|| crate::copy_loop(&fetch, start)).flatten() {
            Some(r) => lifter.lift_copy_loop(r),
            None => lifter.lift(&inst, pc),
        };
        flags = lifter.flags();
        if builder.has_inexpressible(mark) {
            builder.rewind(mark);
            term = Some(Term::Illegal { raw, pc });
            break;
        }
        pc += inst.len as u64;
        builder.end_at(pc);
        match flow {
            Flow::Next => {}
            Flow::End(t) => {
                term = Some(t);
                break;
            }
        }
    }

    let term = term.unwrap_or(Term::Goto { target: pc });
    let mut block = builder.finish(term);
    pass::optimize(&mut block);
    block
}

/// `GEX_TRACE_INSNS=LO-HI` (hex): tier-1 x86 instructions in [LO, HI)
/// report the state before them to the host, which logs it (the native
/// shim's `lockstep::log_trace`): a trace to compare with a reference one of
/// the same program, whose first difference is the instruction gex gets
/// wrong.
pub fn trace_range() -> Option<(u64, u64)> {
    static R: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let v = std::env::var("GEX_TRACE_INSNS").ok()?;
        let (a, b) = v.split_once('-')?;
        let p = |x: &str| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok();
        Some((p(a)?, p(b)?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gex_ir::{print, Expr, GuestReg, Stmt};

    fn bytes(base: u64, code: &'static [u8]) -> impl Fn(u64, &mut [u8; 16]) -> usize {
        move |pc, buf| {
            if pc < base || pc >= base + code.len() as u64 {
                return 0;
            }
            let off = (pc - base) as usize;
            let n = (code.len() - off).min(16);
            buf[..n].copy_from_slice(&code[off..off + n]);
            n
        }
    }

    /// The byte-copy loop `l: movsb; cmp edi, eax; jne l` (a4 39 c7 75 fb)
    /// is one `StrOp` copy, then the compare and the branch; the same three
    /// instructions entered anywhere but at `l`, or with `esi` compared,
    /// are lifted one by one.
    #[test]
    fn byte_copy_loop_is_one_string_copy() {
        let strops = |b: &Block| b.stmts.iter().filter(|s| matches!(s, Stmt::StrOp { .. })).count();
        let code = &[0xa4, 0x39, 0xc7, 0x75, 0xfb, 0xc3];
        let b = lift_block(bytes(0x1000, code), 0x1000, BlockLimits::default());
        assert_eq!(strops(&b), 1, "{}", print::block(&b));
        assert!(matches!(b.term, Term::Branch { .. }), "{}", print::block(&b));
        // cmp edi, esi: not the idiom.
        let b = lift_block(bytes(0x1000, &[0xa4, 0x39, 0xf7, 0x75, 0xfb]), 0x1000, BlockLimits::default());
        assert_eq!(strops(&b), 0, "{}", print::block(&b));
        // jne elsewhere: not the idiom.
        let b = lift_block(bytes(0x1000, &[0xa4, 0x39, 0xc7, 0x75, 0xfa]), 0x1000, BlockLimits::default());
        assert_eq!(strops(&b), 0, "{}", print::block(&b));
    }

    /// SSE is lifted under MXCSR only for a program that asks
    /// (`gex_ir::mxcsr_exact`): without it, `addps`, `mulsd`, `sqrtps` and
    /// `cvtpd2ps` are their operations alone, with no `Expr::Mx` and no
    /// read of MXCSR, as if MXCSR did not exist; with it, each is one.
    #[test]
    fn sse_is_under_mxcsr_only_when_asked() {
        let mxs = |b: &Block| {
            b.stmts.iter().filter(|s| matches!(s, Stmt::Def { expr: Expr::Mx { .. }, .. })).count()
        };
        let reads = |b: &Block| {
            b.stmts.iter().filter(|s| matches!(s, Stmt::Def { expr: Expr::ReadReg(GuestReg::Mxcsr), .. })).count()
        };
        // addps xmm0, xmm1 ; mulsd xmm2, xmm1 ; sqrtps xmm3, xmm1 ; cvtpd2ps xmm4, xmm1 ; ret
        let code = &[0x0f, 0x58, 0xc1, 0xf2, 0x0f, 0x59, 0xd1, 0x0f, 0x51, 0xd9, 0x66, 0x0f, 0x5a, 0xe1, 0xc3];
        if !gex_ir::mxcsr_exact() {
            let b = lift_block(bytes(0x1000, code), 0x1000, BlockLimits::default());
            assert_eq!((mxs(&b), reads(&b)), (0, 0), "{}", print::block(&b));
        }
        gex_ir::set_mxcsr_exact(true);
        let b = lift_block(bytes(0x1000, code), 0x1000, BlockLimits::default());
        gex_ir::set_mxcsr_exact(false);
        assert_eq!(mxs(&b), 4, "{}", print::block(&b));
    }

    /// `fpatan` (d9 f3) is handed to the host rather than raising SIGILL:
    /// i386 musl's `atan2` is this one instruction.
    #[test]
    fn x87_transcendentals_go_to_the_host() {
        // fpatan ; ret
        let b = lift_block(bytes(0x1000, &[0xd9, 0xf3, 0xc3]), 0x1000, BlockLimits::default());
        assert!(matches!(b.term, Term::Indirect { is_ret: true, .. }), "{}", print::block(&b));
        assert!(b.stmts.iter().any(|s| matches!(
            s,
            Stmt::HostCall { func: gex_ir::HostFunc::SysRegWrite, .. }
        )));
    }

    /// The x87's arithmetic and its rounding stores are pure `Stmt::X87`s,
    /// never a state-flushing host call: `fmulp st(1), st` (de c9) is one
    /// `Arith(Mul)`, `fstp qword [rax]` (dd 18) one `Narrow` and an 8-byte
    /// store.
    #[test]
    fn x87_arithmetic_and_stores_are_pure_operations() {
        use gex_ir::{X87Arith, X87Op};
        // fmulp st(1), st ; fstp qword [rax] ; ret
        let b = lift_block(bytes(0x1000, &[0xde, 0xc9, 0xdd, 0x18, 0xc3]), 0x1000, BlockLimits::default());
        let ops: Vec<X87Op> = b
            .stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::X87 { op, .. } => Some(*op),
                _ => None,
            })
            .collect();
        assert_eq!(ops, vec![X87Op::Arith(X87Arith::Mul), X87Op::Narrow { single: false }], "{}", print::block(&b));
        assert!(
            !b.stmts.iter().any(|s| matches!(s, Stmt::HostCall { .. })),
            "{}",
            print::block(&b)
        );
        assert!(
            b.stmts.iter().any(|s| matches!(s, Stmt::Store { sz: gex_ir::MemSz::B8, .. })),
            "{}",
            print::block(&b)
        );
    }

    /// `div`/`idiv` fault before writing anything: a Trap for a zero
    /// divisor, then one for an overflowing quotient, then the writes.
    #[test]
    fn division_traps_before_it_writes() {
        // div ecx ; ret
        let b = lift_block(bytes(0x1000, &[0xf7, 0xf1, 0xc3]), 0x1000, BlockLimits::default());
        let traps: Vec<(usize, i32)> = b
            .stmts
            .iter()
            .enumerate()
            .filter_map(|(n, s)| match s {
                Stmt::Trap { code, pc, .. } if *pc == 0x1000 => Some((n, *code)),
                _ => None,
            })
            .collect();
        let codes: Vec<i32> = traps.iter().map(|t| t.1).collect();
        assert_eq!(
            codes,
            vec![gex_abi::imports::BRK_DIVIDE_BY_ZERO, gex_abi::imports::BRK_DIVIDE_OVERFLOW],
            "{}",
            print::block(&b)
        );
        let first_write = b
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::WriteReg { reg: gex_ir::GuestReg::X(0 | 2), .. }))
            .unwrap();
        assert!(traps.iter().all(|t| t.0 < first_write), "{}", print::block(&b));
    }

    /// `hlt` in user mode is a general-protection fault (SIGSEGV on Linux),
    /// raised unconditionally; so are cli/sti and in/out.
    #[test]
    fn privileged_instructions_fault() {
        for code in [&[0xf4u8][..], &[0xfa], &[0xfb], &[0xe4, 0x60], &[0xee], &[0x6c]] {
            let b = lift_block(bytes(0x1000, code), 0x1000, BlockLimits::default());
            assert!(
                b.stmts.iter().any(|s| matches!(
                    s,
                    Stmt::Trap { code: gex_abi::imports::BRK_GENERAL_PROTECTION, pc: 0x1000, .. }
                )),
                "{code:x?}: {}",
                print::block(&b)
            );
            assert_eq!(b.end, 0x1000 + code.len() as u64);
        }
    }

    #[test]
    fn straight_line_ends_at_ret() {
        // mov eax, 1 ; mov ecx, 2 ; ret
        let b = lift_block(
            bytes(0x1000, &[0xb8, 1, 0, 0, 0, 0xb9, 2, 0, 0, 0, 0xc3]),
            0x1000,
            BlockLimits::default(),
        );
        assert_eq!(b.start, 0x1000);
        assert_eq!(b.end, 0x100b);
        assert!(matches!(b.term, Term::Indirect { is_ret: true, .. }));
    }

    #[test]
    fn compare_and_branch_does_not_read_the_flags() {
        // cmp rdi, rsi ; jb +5
        let b = lift_block(
            bytes(0x2000, &[0x48, 0x39, 0xf7, 0x72, 0x05]),
            0x2000,
            BlockLimits::default(),
        );
        match b.term {
            Term::Branch {
                taken, not_taken, ..
            } => {
                assert_eq!(taken, 0x200a);
                assert_eq!(not_taken, 0x2005);
            }
            other => panic!("{other:?}"),
        }
        let reads_flag = b.stmts.iter().any(|s| {
            matches!(
                s,
                Stmt::Def {
                    expr: Expr::ReadReg(GuestReg::N | GuestReg::Z | GuestReg::C | GuestReg::V),
                    ..
                }
            )
        });
        assert!(!reads_flag, "{}", print::block(&b));
    }

    #[test]
    fn call_writes_the_return_address_and_records_it() {
        // call +0x100 from 0x3000 (5 bytes): return 0x3005, target 0x3105
        let b = lift_block(
            bytes(0x3000, &[0xe8, 0x00, 0x01, 0x00, 0x00]),
            0x3000,
            BlockLimits::default(),
        );
        assert_eq!(b.term, Term::Goto { target: 0x3105 });
        assert_eq!(b.call_return, Some(0x3005));
        let wrote_rsp = b
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::WriteReg { reg: GuestReg::X(4), .. }));
        assert!(wrote_rsp, "call must move rsp:\n{}", print::block(&b));
    }

    #[test]
    fn syscall_terminates() {
        let b = lift_block(bytes(0x4000, &[0x0f, 0x05]), 0x4000, BlockLimits::default());
        assert_eq!(b.term, Term::Syscall { next: 0x4002, imm: 0 });
    }

    #[test]
    fn unknown_becomes_illegal() {
        // ud2
        let b = lift_block(bytes(0x5000, &[0x0f, 0x0b]), 0x5000, BlockLimits::default());
        assert!(matches!(b.term, Term::Illegal { .. }));
    }
}
