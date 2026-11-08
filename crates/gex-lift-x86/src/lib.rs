//! `gex-lift-x86` — x86-64 instruction semantics, expressed in GIR.
//!
//! The contract is the A64 lifter's: total (every decoded instruction lifts
//! or ends the block with a precise SIGILL), exact where wasm and the
//! architecture disagree, and flags written as separate one-bit registers so
//! dead-flag elimination can drop the ones nobody reads.
//!
//! Register model: the sixteen GPRs are `GuestReg::X(0..16)` in hardware
//! order, `rsp` included (`X(4)`); `GuestReg::Sp` is unused. The six
//! arithmetic flags are `N` (SF), `Z`, `C`, `V` (OF), `P` and `A`, with `D`
//! for the direction flag. `Tpidr` holds the FS base and `TpidrRo` the GS
//! base. `xmm0`–`xmm15` are `V128(0..16)`. The x87 stack lives in the
//! context as eight doubles reached through `GuestReg::CtxAddr`. `mm0`–`mm7`
//! are the low halves of `V128(16..24)` (upper halves kept zero), apart from
//! the x87 registers they alias in hardware: see `sse.rs`.

#![forbid(unsafe_code)]

use gex_ir::*;
use gex_isa_x86::{Cond, Inst, Mem, Op, Operand, Reg, Rep, Seg, Width};

mod block;

/// A byte-copy loop at `start`, `l: movsb; cmp r32, edi (either order);
/// jne l` (compilers' and hand-written memcpy tails; 40% of the loop
/// iterations in one game's image loading under Wine): the register `r`
/// it compares `edi` with, for [`Lifter::lift_copy_loop`]. Drivers try it
/// at a block's start, where the loop's own branch lands.
pub fn copy_loop(fetch: &dyn Fn(u64, &mut [u8; 16]) -> usize, start: u64) -> Option<u8> {
    let dec = |pc: u64| {
        let mut buf = [0u8; 16];
        let n = fetch(pc, &mut buf).min(gex_isa_x86::MAX_INST_BYTES);
        (n > 0).then(|| gex_isa_x86::decode(&buf[..n], pc))
    };
    let m = dec(start)?;
    if m.op != Op::Movs || m.width != Width::B || m.rep != Rep::None || m.len != 1 {
        return None;
    }
    let c = dec(start + 1)?;
    let r = match (c.op, c.width, c.ops[0], c.ops[1], c.ops[2]) {
        (Op::Cmp, Width::D, Some(Operand::Reg(Reg::Gp(a, Width::D))), Some(Operand::Reg(Reg::Gp(b, Width::D))), None) => {
            if a == RDI && b != RDI && b != RSI {
                b
            } else if b == RDI && a != RDI && a != RSI {
                a
            } else {
                return None;
            }
        }
        _ => return None,
    };
    let j = dec(start + 1 + c.len as u64)?;
    match (j.op, j.ops[0]) {
        (Op::Jcc(Cond(5)), Some(Operand::Imm(t))) if t as u64 == start => Some(r),
        _ => None,
    }
}
mod sse;
mod x87;
mod x87conv;
pub use block::{lift_block, BlockLimits};

/// Pseudo system-register codes passed to the `sysreg_read`/`sysreg_write`
/// host imports. Negative, like the A64 lifter's, so they never collide with
/// an encoded A64 register.
pub mod sysreg {
    /// `rdtsc`: the host's monotonic counter (the A64 `CNTVCT_EL0` code).
    pub const TSC: i32 = -3;
    /// `cpuid`: written with `eax | ecx << 32`; the host fills
    /// `eax`, `ebx`, `ecx`, `edx` in the context.
    pub const CPUID: i32 = -100;
    /// An x87 instruction the host computes: written with an [`super::x87op`]
    /// code; the host reads and updates the register stack, `TOP`, the tags
    /// and the condition codes in the context.
    pub const X87: i32 = -101;
    /// `cmpxchg16b`: written with the effective address; the host compares
    /// and swaps under its lock and sets `rax`, `rdx` and ZF in the context.
    pub const CMPXCHG16B: i32 = -102;
    /// `GEX_TRACE_INSNS`: written with the pc before an instruction; the
    /// host logs the context.
    pub const TRACE: i32 = -103;
    /// A far `jmp`/`call`/`ret` or `iret`: written with `offset | selector
    /// << 32`; the host checks the selector and sets `ctx::CS`.
    pub const FAR: i32 = -104;
    /// A far transfer whose block then reads [`FAR_NEXT`] for the address
    /// to dispatch: written as [`FAR`] is.
    pub const FAR_IN_PLACE: i32 = -105;
    /// Read after [`FAR_IN_PLACE`]: the address the block dispatches.
    pub const FAR_NEXT: i32 = -106;
}

/// The x87 operations [`sysreg::X87`] carries: an operation code in the low
/// byte and its operands in the fields below. `gex_kernel::x86::x87op`
/// serves them and must agree.
pub mod x87op {
    pub const F2XM1: u64 = 1;
    pub const FYL2X: u64 = 2;
    pub const FYL2XP1: u64 = 3;
    pub const FPTAN: u64 = 4;
    pub const FPATAN: u64 = 5;
    pub const FXTRACT: u64 = 6;
    pub const FPREM: u64 = 7;
    pub const FPREM1: u64 = 8;
    pub const FSIN: u64 = 9;
    pub const FCOS: u64 = 10;
    pub const FSINCOS: u64 = 11;
    /// Arithmetic: operands and destination from the form, reversed by `REV`.
    pub const FADD: u64 = 20;
    pub const FMUL: u64 = 21;
    pub const FSUB: u64 = 22;
    pub const FDIV: u64 = 23;
    /// Compare `st0` with the form's other operand; C0/C2/C3, or ZF/PF/CF
    /// in `X87_TMP` with `TO_EFLAGS`.
    pub const FCOM: u64 = 24;
    pub const FTST: u64 = 25;
    pub const FSQRT: u64 = 26;
    pub const FRNDINT: u64 = 27;
    pub const FSCALE: u64 = 28;
    /// Push the memory operand in `X87_TMP`, converted from its format.
    pub const FLD: u64 = 29;
    /// Convert `st0` to the memory format into `X87_TMP` (no pop).
    pub const FST: u64 = 30;
    pub const FXAM: u64 = 31;
    /// Push constant `K` (`fld1`, `fldl2t`, `fldl2e`, `fldpi`, `fldlg2`,
    /// `fldln2`, `fldz`).
    pub const FLDCONST: u64 = 32;

    /// `st(i)`, 3 bits.
    pub const I_SHIFT: u32 = 8;
    /// Operand form, 2 bits.
    pub const FORM_SHIFT: u32 = 11;
    /// `st0 op= memory operand (X87_TMP)`.
    pub const FORM_MEM: u64 = 0;
    /// `st0 op= st(i)`.
    pub const FORM_ST0_STI: u64 = 1;
    /// `st(i) op= st0`.
    pub const FORM_STI_ST0: u64 = 2;
    /// Pops afterwards, 2 bits.
    pub const POP_SHIFT: u32 = 13;
    /// Swap the operands (`fsubr`, `fdivr`).
    pub const REV: u64 = 1 << 15;
    /// Memory format, 3 bits.
    pub const MEM_SHIFT: u32 = 16;
    pub const MEM_F32: u64 = 0;
    pub const MEM_F64: u64 = 1;
    pub const MEM_F80: u64 = 2;
    pub const MEM_I16: u64 = 3;
    pub const MEM_I32: u64 = 4;
    pub const MEM_I64: u64 = 5;
    /// 18-digit packed BCD (`fbld`, `fbstp`): ten bytes, the sign in
    /// bit 7 of the last.
    pub const MEM_BCD: u64 = 6;
    /// `fisttp`: convert toward zero.
    pub const TRUNC: u64 = 1 << 19;
    /// `fucom`: only an SNaN raises IE.
    pub const QUIET: u64 = 1 << 20;
    /// `fcomi`: the result goes to EFLAGS, not the condition codes.
    pub const TO_EFLAGS: u64 = 1 << 21;
    /// Constant index, 3 bits.
    pub const K_SHIFT: u32 = 24;
}

/// What the condition flags hold, when a branch can ask about the operands
/// directly instead of reading the flag registers. See the A64 lifter for the
/// rationale; the x86 twist is that 8- and 16-bit operands travel
/// zero-extended in an `I32`, so a signed comparison first sign-extends.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FlagSource {
    #[default]
    Unknown,
    /// Flags of `a - b` at `bits` wide (`cmp`, `sub`).
    Sub { a: Value, b: Value, ty: Ty, bits: u32 },
    /// Flags of a logical result `v` (`test`, `and`, `or`, `xor`): C and O
    /// are zero.
    Logic { v: Value, ty: Ty, bits: u32 },
}

/// What the lifter decided about control flow after one instruction.
pub enum Flow {
    Next,
    End(Term),
}

pub struct Lifter<'a> {
    b: &'a mut Builder,
    flags: FlagSource,
    raw: u32,
    /// The instruction being lifted and where it is.
    pc: u64,
    next: u64,
    /// The current instruction's memory operand, for the x87 forms whose
    /// decoded `X87Src` names memory without carrying the operand.
    cur_mem: Operand,
    /// Lifting an MMX instruction: `mm` operands and 64-bit memory (sse.rs).
    mmx: bool,
}

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RSP: u8 = 4;
const RBP: u8 = 5;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R11: u8 = 11;

fn ty_of(w: Width) -> Ty {
    if w == Width::Q {
        Ty::I64
    } else {
        Ty::I32
    }
}

fn mem_sz(w: Width) -> MemSz {
    match w {
        Width::B => MemSz::B1,
        Width::W => MemSz::B2,
        Width::D => MemSz::B4,
        Width::Q => MemSz::B8,
    }
}

impl<'a> Lifter<'a> {
    pub fn new(b: &'a mut Builder, flags: FlagSource, raw: u32) -> Lifter<'a> {
        Lifter {
            b,
            flags,
            raw,
            pc: 0,
            next: 0,
            cur_mem: Operand::Imm(0),
            mmx: false,
        }
    }

    /// The address width of the code being lifted: the size of a pushed
    /// return address, of what `ret` pops, of an indirect branch target.
    fn wa(&self) -> Width {
        if gex_isa_x86::mode() == gex_isa_x86::Mode::Compat {
            Width::D
        } else {
            Width::Q
        }
    }

    /// An indirect branch target: read at the address width, as an `I64`.
    fn rd_target(&mut self, o: Operand) -> Value {
        let w = self.wa();
        let v = self.rd(o, w);
        if w == Width::Q {
            v
        } else {
            self.b.extend(v, false)
        }
    }

    pub fn flags(&self) -> FlagSource {
        self.flags
    }

    // -- constants -------------------------------------------------------------

    fn k(&mut self, ty: Ty, v: u64) -> Value {
        self.b.const_ty(ty, v)
    }
    fn k32(&mut self, v: u32) -> Value {
        self.b.const_i32(v)
    }
    fn k64(&mut self, v: u64) -> Value {
        self.b.const_i64(v)
    }

    // -- registers ---------------------------------------------------------------

    /// Read GPR `n` in full.
    fn rd64(&mut self, n: u8) -> Value {
        self.b.read(GuestReg::X(n))
    }
    fn wr64(&mut self, n: u8, v: Value) {
        self.b.write(GuestReg::X(n), v);
    }

    /// Read a register at its width: `I64` for 64-bit, otherwise an `I32`
    /// holding the zero-extended low bits.
    fn rd_reg(&mut self, r: Reg) -> Value {
        match r {
            Reg::Gp(n, Width::Q) => self.rd64(n),
            Reg::Gp(n, Width::D) => {
                let v = self.rd64(n);
                self.b.wrap(v)
            }
            Reg::Gp(n, w) => {
                let v = self.rd64(n);
                let v = self.b.wrap(v);
                let m = self.k32(w.mask() as u32);
                self.b.bin(BinOp::And, Ty::I32, v, m)
            }
            Reg::High(n) => {
                let v = self.rd64(n);
                let v = self.b.wrap(v);
                let eight = self.k32(8);
                let v = self.b.bin(BinOp::ShrU, Ty::I32, v, eight);
                let m = self.k32(0xff);
                self.b.bin(BinOp::And, Ty::I32, v, m)
            }
            Reg::Seg(seg) => {
                let sel = self.sel_of(seg);
                self.k32(sel)
            }
            // Vector and x87 registers are read by their own paths.
            Reg::Xmm(_) | Reg::Mm(_) | Reg::St(_) => self.k64(0),
        }
    }

    /// Write a register at its width. 32-bit writes zero-extend into the
    /// full register; 16- and 8-bit writes merge into it.
    fn wr_reg(&mut self, r: Reg, val: Value) {
        match r {
            Reg::Gp(n, Width::Q) => self.wr64(n, val),
            Reg::Gp(n, Width::D) => {
                let v = self.b.extend(val, false);
                self.wr64(n, v);
            }
            Reg::Gp(n, w) => {
                let old = self.rd64(n);
                let keep = self.k64(!w.mask());
                let old = self.b.bin(BinOp::And, Ty::I64, old, keep);
                let m = self.k32(w.mask() as u32);
                let val = self.b.bin(BinOp::And, Ty::I32, val, m);
                let val = self.b.extend(val, false);
                let v = self.b.bin(BinOp::Or, Ty::I64, old, val);
                self.wr64(n, v);
            }
            Reg::High(n) => {
                let old = self.rd64(n);
                let keep = self.k64(!0xff00u64);
                let old = self.b.bin(BinOp::And, Ty::I64, old, keep);
                let m = self.k32(0xff);
                let val = self.b.bin(BinOp::And, Ty::I32, val, m);
                let eight = self.k32(8);
                let val = self.b.bin(BinOp::Shl, Ty::I32, val, eight);
                let val = self.b.extend(val, false);
                let v = self.b.bin(BinOp::Or, Ty::I64, old, val);
                self.wr64(n, v);
            }
            Reg::Seg(_) | Reg::Xmm(_) | Reg::Mm(_) | Reg::St(_) => {}
        }
    }

    // -- memory --------------------------------------------------------------------

    /// The effective address of a memory operand, as an `I64`, and a static
    /// offset to carry on the access.
    fn addr(&mut self, m: &Mem) -> (Value, i64) {
        self.addr_with(m, true)
    }

    /// [`Self::addr`], folding a KUSER_SHARED_DATA displacement (`fold`) or
    /// not: `lea` is arithmetic, and its sum may be anything.
    fn addr_with(&mut self, m: &Mem, fold: bool) -> (Value, i64) {
        let mut parts: Vec<Value> = Vec::new();
        if let Some(b) = m.base {
            parts.push(self.rd64(b));
        }
        if let Some(i) = m.index {
            let mut v = self.rd64(i);
            if m.scale > 1 {
                let sh = self.k64(m.scale.trailing_zeros() as u64);
                v = self.b.bin(BinOp::Shl, Ty::I64, v, sh);
            }
            parts.push(v);
        }
        if let Some(seg) = m.seg {
            let base = match seg {
                Seg::Fs => self.b.read(GuestReg::Tpidr),
                _ => self.b.read(GuestReg::TpidrRo),
            };
            parts.push(base);
        }
        let mut sum: Option<Value> = None;
        for p in parts {
            sum = Some(match sum {
                None => p,
                Some(s) => self.b.bin(BinOp::Add, Ty::I64, s, p),
            });
        }
        // KUSER_SHARED_DATA folded (`gex_abi::fold_kuser`): a displacement into
        // Windows' page names the folded page instead, in every form that has
        // one (absolute, with a base, an index or both), 32- and 64-bit
        // addressing alike; not through a segment, and not for `lea`. An
        // address the emitter sees as a constant (a pointer loaded in the same
        // block) is folded there; one computed at run time faults by name
        // (`BRK_KUSER_UNFOLDED`).
        let named = if m.addr32 { m.disp as u64 & 0xffff_ffff } else { m.disp as u64 };
        let disp = match gex_abi::fold_kuser(named) {
            Some(f) if fold && m.seg.is_none() => {
                gex_ir::note_kuser_fold();
                f as i64
            }
            _ => m.disp,
        };
        match sum {
            // A bare absolute address: in 32-bit code the sign-extended
            // disp32 names a 32-bit address.
            None if m.addr32 => (self.k64(disp as u64 & 0xffff_ffff), 0),
            None => (self.k64(disp as u64), 0),
            Some(s) if m.addr32 => {
                // 32-bit address arithmetic wraps before the access.
                let d = self.k64(disp as u64);
                let s = self.b.bin(BinOp::Add, Ty::I64, s, d);
                let s = self.b.wrap(s);
                (self.b.extend(s, false), 0)
            }
            Some(s) => (s, disp),
        }
    }

    fn load(&mut self, m: &Mem, w: Width) -> Value {
        let (addr, offset) = self.addr(m);
        self.b.def(Expr::Load {
            addr,
            offset,
            sz: mem_sz(w),
            signed: false,
            ty: ty_of(w),
            order: Order::Relaxed,
        })
    }

    fn store(&mut self, m: &Mem, w: Width, val: Value) {
        let (addr, offset) = self.addr(m);
        self.b.stmt(Stmt::Store {
            addr,
            offset,
            val,
            sz: mem_sz(w),
            order: Order::Relaxed,
        });
    }

    /// Read an operand at width `w`.
    fn rd(&mut self, o: Operand, w: Width) -> Value {
        match o {
            Operand::Reg(r) => self.rd_reg(r),
            Operand::Mem(m) => self.load(&m, w),
            Operand::Imm(v) => self.k(ty_of(w), (v as u64) & w.mask()),
        }
    }

    fn wr(&mut self, o: Operand, w: Width, val: Value) {
        match o {
            Operand::Reg(r) => self.wr_reg(r, val),
            Operand::Mem(m) => self.store(&m, w, val),
            Operand::Imm(_) => {}
        }
    }

    // -- stack ------------------------------------------------------------------------

    fn push(&mut self, w: Width, val: Value) {
        let sp = self.rd64(RSP);
        let n = self.k64((w.bytes() as u64).wrapping_neg());
        let sp = self.b.bin(BinOp::Add, Ty::I64, sp, n);
        self.b.stmt(Stmt::Store {
            addr: sp,
            offset: 0,
            val,
            sz: mem_sz(w),
            order: Order::Relaxed,
        });
        self.wr64(RSP, sp);
    }

    fn pop(&mut self, w: Width) -> Value {
        let sp = self.rd64(RSP);
        let v = self.b.def(Expr::Load {
            addr: sp,
            offset: 0,
            sz: mem_sz(w),
            signed: false,
            ty: ty_of(w),
            order: Order::Relaxed,
        });
        let n = self.k64(w.bytes() as u64);
        let sp = self.b.bin(BinOp::Add, Ty::I64, sp, n);
        self.wr64(RSP, sp);
        v
    }

    // -- flags -------------------------------------------------------------------

    /// Mask an `I32` value to `w` when narrower than 32 bits.
    fn narrow(&mut self, v: Value, w: Width) -> Value {
        match w {
            Width::B | Width::W => {
                let m = self.k32(w.mask() as u32);
                self.b.bin(BinOp::And, Ty::I32, v, m)
            }
            _ => v,
        }
    }

    /// Sign-extend the low `w` bits of a value to its type.
    fn sext(&mut self, v: Value, w: Width) -> Value {
        match w {
            Width::B | Width::W => self.b.def(Expr::ExtendBits {
                a: v,
                ty: Ty::I32,
                bits: w.bits(),
                signed: true,
            }),
            _ => v,
        }
    }

    /// Bit `bits-1` of `v` (already masked), as an `I32` 0/1.
    fn sign_bit(&mut self, v: Value, w: Width) -> Value {
        let ty = ty_of(w);
        let sh = self.k(ty, (w.bits() - 1) as u64);
        let s = self.b.bin(BinOp::ShrU, ty, v, sh);
        let s = if ty == Ty::I64 { self.b.wrap(s) } else { s };
        let one = self.k32(1);
        self.b.bin(BinOp::And, Ty::I32, s, one)
    }

    fn is_zero(&mut self, v: Value, w: Width) -> Value {
        let ty = ty_of(w);
        let z = self.k(ty, 0);
        self.b.cmp(CmpOp::Eq, ty, v, z)
    }

    /// PF: 1 when the low byte has an even number of set bits.
    fn parity(&mut self, v: Value, w: Width) -> Value {
        let v = if ty_of(w) == Ty::I64 { self.b.wrap(v) } else { v };
        let m = self.k32(0xff);
        let v = self.b.bin(BinOp::And, Ty::I32, v, m);
        let p = self.b.un(UnOp::Popcnt, Ty::I32, v);
        let one = self.k32(1);
        let p = self.b.bin(BinOp::And, Ty::I32, p, one);
        self.b.bin(BinOp::Xor, Ty::I32, p, one)
    }

    /// AF: carry out of bit 3, `((a ^ b ^ r) >> 4) & 1`.
    fn aux(&mut self, a: Value, b: Value, r: Value, w: Width) -> Value {
        let ty = ty_of(w);
        let x = self.b.bin(BinOp::Xor, ty, a, b);
        let x = self.b.bin(BinOp::Xor, ty, x, r);
        let x = if ty == Ty::I64 { self.b.wrap(x) } else { x };
        let four = self.k32(4);
        let x = self.b.bin(BinOp::ShrU, Ty::I32, x, four);
        let one = self.k32(1);
        self.b.bin(BinOp::And, Ty::I32, x, one)
    }

    fn write_flag(&mut self, r: GuestReg, v: Value) {
        self.b.write(r, v);
    }

    /// The flags of a logical operation: C and O clear, A cleared too (it is
    /// undefined, and zero is a value), S/Z/P from the masked result.
    fn flags_logic(&mut self, r: Value, w: Width) {
        let s = self.sign_bit(r, w);
        let z = self.is_zero(r, w);
        let p = self.parity(r, w);
        let zero = self.k32(0);
        self.write_flag(GuestReg::N, s);
        self.write_flag(GuestReg::Z, z);
        self.write_flag(GuestReg::C, zero);
        self.write_flag(GuestReg::V, zero);
        self.write_flag(GuestReg::P, p);
        self.write_flag(GuestReg::A, zero);
        self.flags = FlagSource::Logic {
            v: r,
            ty: ty_of(w),
            bits: w.bits(),
        };
    }

    /// The flags of `r = a + b (+ cin)`, `a`/`b`/`r` masked to `w`.
    /// `cin` is an `I32` 0/1 carry-in, or `None`.
    fn flags_add(&mut self, a: Value, b: Value, r: Value, w: Width, cin: Option<Value>, keep_c: bool) {
        let ty = ty_of(w);
        let s = self.sign_bit(r, w);
        let z = self.is_zero(r, w);
        let p = self.parity(r, w);
        let af = self.aux(a, b, r, w);
        // OF: operands same sign, result different.
        let ar = self.b.bin(BinOp::Xor, ty, a, r);
        let br = self.b.bin(BinOp::Xor, ty, b, r);
        let o = self.b.bin(BinOp::And, ty, ar, br);
        let o = self.sign_bit(o, w);
        if !keep_c {
            let c = match (w, cin) {
                (Width::Q, None) => self.b.cmp(CmpOp::LtU, ty, r, a),
                (Width::Q, Some(cin)) => {
                    // carry = cin ? r <= a : r < a
                    let lt = self.b.cmp(CmpOp::LtU, ty, r, a);
                    let le = self.b.cmp(CmpOp::LeU, ty, r, a);
                    self.b.select(cin, le, lt, Ty::I32)
                }
                (Width::D, None) => self.b.cmp(CmpOp::LtU, ty, r, a),
                (Width::D, Some(cin)) => {
                    let lt = self.b.cmp(CmpOp::LtU, ty, r, a);
                    let le = self.b.cmp(CmpOp::LeU, ty, r, a);
                    self.b.select(cin, le, lt, Ty::I32)
                }
                (_, cin) => {
                    // Narrow: the unmasked sum's bit `bits` is the carry.
                    let sum = self.b.bin(BinOp::Add, Ty::I32, a, b);
                    let sum = match cin {
                        Some(c) => self.b.bin(BinOp::Add, Ty::I32, sum, c),
                        None => sum,
                    };
                    let sh = self.k32(w.bits());
                    let c = self.b.bin(BinOp::ShrU, Ty::I32, sum, sh);
                    let one = self.k32(1);
                    self.b.bin(BinOp::And, Ty::I32, c, one)
                }
            };
            self.write_flag(GuestReg::C, c);
        }
        self.write_flag(GuestReg::N, s);
        self.write_flag(GuestReg::Z, z);
        self.write_flag(GuestReg::V, o);
        self.write_flag(GuestReg::P, p);
        self.write_flag(GuestReg::A, af);
        self.flags = FlagSource::Unknown;
    }

    /// The flags of `r = a - b (- bin)`.
    fn flags_sub(&mut self, a: Value, b: Value, r: Value, w: Width, bin: Option<Value>, keep_c: bool) {
        let ty = ty_of(w);
        let s = self.sign_bit(r, w);
        let z = self.is_zero(r, w);
        let p = self.parity(r, w);
        let af = self.aux(a, b, r, w);
        // OF: operands differ in sign, result's sign differs from a's.
        let ab = self.b.bin(BinOp::Xor, ty, a, b);
        let ar = self.b.bin(BinOp::Xor, ty, a, r);
        let o = self.b.bin(BinOp::And, ty, ab, ar);
        let o = self.sign_bit(o, w);
        if !keep_c {
            let c = match bin {
                None => self.b.cmp(CmpOp::LtU, ty, a, b),
                Some(bin) => {
                    let lt = self.b.cmp(CmpOp::LtU, ty, a, b);
                    let le = self.b.cmp(CmpOp::LeU, ty, a, b);
                    self.b.select(bin, le, lt, Ty::I32)
                }
            };
            self.write_flag(GuestReg::C, c);
        }
        self.write_flag(GuestReg::N, s);
        self.write_flag(GuestReg::Z, z);
        self.write_flag(GuestReg::V, o);
        self.write_flag(GuestReg::P, p);
        self.write_flag(GuestReg::A, af);
        self.flags = match bin {
            None if !keep_c => FlagSource::Sub {
                a,
                b,
                ty,
                bits: w.bits(),
            },
            _ => FlagSource::Unknown,
        };
    }

    fn read_flag(&mut self, r: GuestReg) -> Value {
        self.b.read(r)
    }

    /// The `I32` truth value of a condition code.
    fn cond(&mut self, c: Cond) -> Value {
        let base = c.0 >> 1;
        let inv = c.0 & 1 == 1;
        let direct = match self.flags {
            FlagSource::Sub { a, b, ty, bits } => {
                let w = Width::from_bytes(bits / 8);
                match base {
                    1 => Some(self.b.cmp(CmpOp::LtU, ty, a, b)),
                    2 => Some(self.b.cmp(CmpOp::Eq, ty, a, b)),
                    3 => Some(self.b.cmp(CmpOp::LeU, ty, a, b)),
                    6 => {
                        let sa = self.sext(a, w);
                        let sb = self.sext(b, w);
                        Some(self.b.cmp(CmpOp::LtS, ty, sa, sb))
                    }
                    7 => {
                        let sa = self.sext(a, w);
                        let sb = self.sext(b, w);
                        Some(self.b.cmp(CmpOp::LeS, ty, sa, sb))
                    }
                    _ => None,
                }
            }
            FlagSource::Logic { v, ty, bits } => {
                let w = Width::from_bytes(bits / 8);
                match base {
                    0 | 1 => Some(self.k32(0)),
                    2 | 3 => {
                        let z = self.k(ty, 0);
                        Some(self.b.cmp(CmpOp::Eq, ty, v, z))
                    }
                    4 | 6 => {
                        let sv = self.sext(v, w);
                        let z = self.k(ty, 0);
                        Some(self.b.cmp(CmpOp::LtS, ty, sv, z))
                    }
                    7 => {
                        let sv = self.sext(v, w);
                        let z = self.k(ty, 0);
                        Some(self.b.cmp(CmpOp::LeS, ty, sv, z))
                    }
                    _ => None,
                }
            }
            FlagSource::Unknown => None,
        };
        let v = match direct {
            Some(v) => v,
            None => match base {
                0 => self.read_flag(GuestReg::V),
                1 => self.read_flag(GuestReg::C),
                2 => self.read_flag(GuestReg::Z),
                3 => {
                    let c = self.read_flag(GuestReg::C);
                    let z = self.read_flag(GuestReg::Z);
                    self.b.bin(BinOp::Or, Ty::I32, c, z)
                }
                4 => self.read_flag(GuestReg::N),
                5 => self.read_flag(GuestReg::P),
                6 => {
                    let s = self.read_flag(GuestReg::N);
                    let o = self.read_flag(GuestReg::V);
                    self.b.bin(BinOp::Xor, Ty::I32, s, o)
                }
                _ => {
                    let s = self.read_flag(GuestReg::N);
                    let o = self.read_flag(GuestReg::V);
                    let so = self.b.bin(BinOp::Xor, Ty::I32, s, o);
                    let z = self.read_flag(GuestReg::Z);
                    self.b.bin(BinOp::Or, Ty::I32, so, z)
                }
            },
        };
        if inv {
            let one = self.k32(1);
            self.b.bin(BinOp::Xor, Ty::I32, v, one)
        } else {
            v
        }
    }

    // -- the entry point ----------------------------------------------------------------

    pub fn lift(&mut self, i: &Inst, pc: u64) -> Flow {
        self.pc = pc;
        self.next = pc + i.len as u64;
        let next = self.next;
        let w = i.width;
        let ty = ty_of(w);
        match i.op {
            // `int 0x80` is the i386 system-call gate; the kernel reads the
            // number and arguments from the 32-bit registers when the process
            // is a 32-bit one. Any other software interrupt is a fault.
            Op::Int if i.ops[0] == Some(Operand::Imm(0x80)) && self.wa() == Width::D => {
                Flow::End(Term::Syscall { next, imm: 0 })
            }
            // A privileged instruction in user mode (hlt, cli/sti, in/out,
            // ins/outs), and a software interrupt through a gate user code
            // may not use: #GP, which Linux delivers as SIGSEGV. (musl's
            // `a_crash` is `hlt`.)
            Op::Hlt | Op::Privileged(_) | Op::Int
                if i.op != Op::Int
                    || matches!(i.ops[0], Some(Operand::Imm(n)) if n != 3 && n != 0x80) =>
            {
                let yes = self.k32(1);
                self.trap_if(yes, gex_abi::imports::BRK_GENERAL_PROTECTION);
                Flow::End(Term::Illegal { raw: self.raw, pc })
            }
            // `int1` (f1), and `int 0x80` from 64-bit code (Linux's ia32
            // system-call emulation), are not implemented.
            Op::Int if i.ops[0] != Some(Operand::Imm(3)) => Flow::End(Term::Illegal { raw: self.raw, pc }),
            Op::Unknown | Op::Undefined => Flow::End(Term::Illegal { raw: self.raw, pc }),
            Op::Xlat => {
                // al = [seg:(e/r)bx + zx(al)]: the table index is added
                // before 32-bit address arithmetic wraps.
                let Operand::Mem(mut m) = i.op0() else { unreachable!() };
                let al = self.rd_reg(Reg::Gp(0, Width::B));
                let al = self.b.extend(al, false);
                let rbx = self.rd64(3);
                let base = self.b.bin(BinOp::Add, Ty::I64, rbx, al);
                // Reuse the operand's segment and address-size handling with
                // the sum as a base: a scratch register would clobber the
                // guest's, so the address is formed here instead.
                m.base = None;
                let (seg_part, off) = self.addr(&m);
                let sum = self.b.bin(BinOp::Add, Ty::I64, base, seg_part);
                let sum = if m.addr32 {
                    let w32 = self.b.wrap(sum);
                    self.b.extend(w32, false)
                } else {
                    sum
                };
                let v = self.b.def(Expr::Load {
                    addr: sum,
                    offset: off,
                    sz: MemSz::B1,
                    signed: false,
                    ty: Ty::I32,
                    order: Order::Relaxed,
                });
                self.wr_reg(Reg::Gp(0, Width::B), v);
                Flow::Next
            }
            Op::Nop | Op::Endbr | Op::Pause | Op::Prefetch | Op::Fwait | Op::Clflush => Flow::Next,
            Op::Fence => {
                self.b.stmt(Stmt::Fence);
                Flow::Next
            }

            // -- moves --------------------------------------------------------------
            Op::Mov | Op::Movabs => {
                let v = self.rd(i.op1(), w);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Movzx(sw) => {
                let v = self.rd(i.op1(), sw);
                let v = self.widen(v, sw, w, false);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Movsx(sw) => {
                let v = self.rd(i.op1(), sw);
                let v = self.widen(v, sw, w, true);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Movsxd => {
                let v = self.rd(i.op1(), Width::D);
                let v = self.widen(v, Width::D, w, true);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Lea => {
                let Operand::Mem(m) = i.op1() else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                // lea has no segment base and no access: pure arithmetic.
                let mut m = m;
                m.seg = None;
                let (a, off) = self.addr_with(&m, false);
                let a = if off != 0 {
                    let d = self.k64(off as u64);
                    self.b.bin(BinOp::Add, Ty::I64, a, d)
                } else {
                    a
                };
                let v = match w {
                    Width::Q => a,
                    _ => {
                        let v = self.b.wrap(a);
                        self.narrow(v, w)
                    }
                };
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Push => {
                let v = match i.op0() {
                    Operand::Imm(imm) => self.k(ty, (imm as u64) & w.mask()),
                    Operand::Reg(Reg::Seg(seg)) => self.sel_val(seg, w),
                    o => self.rd(o, w),
                };
                self.push(w, v);
                Flow::Next
            }
            Op::Pop => {
                let v = self.pop(w);
                if !matches!(i.op0(), Operand::Reg(Reg::Seg(_))) {
                    self.wr(i.op0(), w, v);
                }
                Flow::Next
            }
            Op::Xchg => {
                let (d, s) = (i.op0(), i.op1());
                if i.lock || matches!(d, Operand::Mem(_)) {
                    if let Operand::Mem(m) = d {
                        let sv = self.rd(s, w);
                        let (addr, offset) = self.addr(&m);
                        let addr = self.offset_addr(addr, offset);
                        let old = self.b.def(Expr::Rmw {
                            op: RmwOp::Xchg,
                            addr,
                            val: sv,
                            sz: mem_sz(w),
                            ty,
                        });
                        self.wr(s, w, old);
                        return Flow::Next;
                    }
                }
                let dv = self.rd(d, w);
                let sv = self.rd(s, w);
                self.wr(d, w, sv);
                self.wr(s, w, dv);
                Flow::Next
            }
            Op::Cmovcc(c) => {
                let cond = self.cond(c);
                let dv = self.rd(i.op0(), w);
                let sv = self.rd(i.op1(), w);
                let v = self.b.select(cond, sv, dv, ty);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Setcc(c) => {
                let cond = self.cond(c);
                self.wr(i.op0(), Width::B, cond);
                Flow::Next
            }
            Op::Cbw => {
                // Sign-extend the low half of rax into the width.
                let half = Width::from_bytes(w.bytes() / 2);
                let v = self.rd_reg(Reg::Gp(RAX, half));
                let v = self.widen(v, half, w, true);
                self.wr_reg(Reg::Gp(RAX, w), v);
                Flow::Next
            }
            Op::Cwd => {
                let v = self.rd_reg(Reg::Gp(RAX, w));
                let s = self.sext(v, w);
                let sh = self.k(ty, (w.bits() - 1) as u64);
                let v = self.b.bin(BinOp::ShrS, ty, s, sh);
                let v = self.narrow(v, w);
                self.wr_reg(Reg::Gp(RDX, w), v);
                Flow::Next
            }
            Op::Bswap => {
                let v = self.rd(i.op0(), w);
                let v = self.byte_reverse(v, w);
                self.wr(i.op0(), w, v);
                Flow::Next
            }
            Op::Movbe => {
                let v = self.rd(i.op1(), w);
                let v = self.byte_reverse(v, w);
                self.wr(i.op0(), w, v);
                Flow::Next
            }

            // -- arithmetic --------------------------------------------------------------
            Op::Add | Op::Or | Op::Adc | Op::Sbb | Op::And | Op::Sub | Op::Xor | Op::Cmp | Op::Test => {
                self.alu(i.op, i.op0(), i.op1(), w, i.lock);
                Flow::Next
            }
            Op::Inc | Op::Dec => {
                let d = i.op0();
                let one = self.k(ty, 1);
                let a = if i.lock {
                    if let Operand::Mem(m) = d {
                        let (addr, offset) = self.addr(&m);
                        let addr = self.offset_addr(addr, offset);
                        self.b.def(Expr::Rmw {
                            op: if i.op == Op::Inc { RmwOp::Add } else { RmwOp::Sub },
                            addr,
                            val: one,
                            sz: mem_sz(w),
                            ty,
                        })
                    } else {
                        self.rd(d, w)
                    }
                } else {
                    self.rd(d, w)
                };
                let a = self.narrow(a, w);
                let r = if i.op == Op::Inc {
                    self.b.bin(BinOp::Add, ty, a, one)
                } else {
                    self.b.bin(BinOp::Sub, ty, a, one)
                };
                let r = self.narrow(r, w);
                if !i.lock {
                    self.wr(d, w, r);
                }
                if i.op == Op::Inc {
                    self.flags_add(a, one, r, w, None, true);
                } else {
                    self.flags_sub(a, one, r, w, None, true);
                }
                Flow::Next
            }
            Op::Neg => {
                let a = self.rd(i.op0(), w);
                let zero = self.k(ty, 0);
                let r = self.b.bin(BinOp::Sub, ty, zero, a);
                let r = self.narrow(r, w);
                self.wr(i.op0(), w, r);
                self.flags_sub(zero, a, r, w, None, false);
                // CF is "a != 0", which the sub formula (0 < a) already gives.
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::Not => {
                let a = self.rd(i.op0(), w);
                let r = self.b.un(UnOp::Not, ty, a);
                let r = self.narrow(r, w);
                self.wr(i.op0(), w, r);
                Flow::Next
            }
            Op::Mul => {
                self.mul1(i.op0(), w, false);
                Flow::Next
            }
            Op::Imul1 => {
                self.mul1(i.op0(), w, true);
                Flow::Next
            }
            Op::Imul => {
                let (dst, a, b) = if i.n_ops() == 3 {
                    (i.op0(), i.op1(), i.op2())
                } else {
                    (i.op0(), i.op0(), i.op1())
                };
                let av = self.rd(a, w);
                let bv = self.rd(b, w);
                let av = self.sext(av, w);
                let bv = self.sext(bv, w);
                let r = self.b.bin(BinOp::Mul, ty, av, bv);
                let rn = self.narrow(r, w);
                self.wr(dst, w, rn);
                // CF = OF = the full product does not fit the width.
                let of = match w {
                    Width::Q => {
                        // Compare high half of the 128-bit product against the
                        // sign of the low half. High half via the identity
                        // hi = mulhs(a, b).
                        let hi = self.mulh_s64(av, bv);
                        let sh = self.k64(63);
                        let sign = self.b.bin(BinOp::ShrS, Ty::I64, r, sh);
                        self.b.cmp(CmpOp::Ne, Ty::I64, hi, sign)
                    }
                    Width::D => {
                        let a64 = self.b.extend(av, true);
                        let b64 = self.b.extend(bv, true);
                        let full = self.b.bin(BinOp::Mul, Ty::I64, a64, b64);
                        let r64 = self.b.extend(r, true);
                        self.b.cmp(CmpOp::Ne, Ty::I64, full, r64)
                    }
                    _ => {
                        let s = self.sext(rn, w);
                        self.b.cmp(CmpOp::Ne, Ty::I32, s, r)
                    }
                };
                self.write_flag(GuestReg::C, of);
                self.write_flag(GuestReg::V, of);
                // S/Z/P/A are undefined; give them the result's values.
                let s = self.sign_bit(rn, w);
                let z = self.is_zero(rn, w);
                let p = self.parity(rn, w);
                let zero = self.k32(0);
                self.write_flag(GuestReg::N, s);
                self.write_flag(GuestReg::Z, z);
                self.write_flag(GuestReg::P, p);
                self.write_flag(GuestReg::A, zero);
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::Div | Op::Idiv => self.div(i.op0(), w, i.op == Op::Idiv),
            Op::Shl | Op::Shr | Op::Sar | Op::Rol | Op::Ror => {
                self.shift(i.op, i.op0(), i.op1(), w);
                Flow::Next
            }
            Op::Rcl | Op::Rcr => self.rotate_carry(i.op, i.op0(), i.op1(), w),
            Op::Shld | Op::Shrd => {
                self.double_shift(i.op, i.op0(), i.op1(), i.op2(), w);
                Flow::Next
            }
            Op::Bt | Op::Bts | Op::Btr | Op::Btc => {
                self.bit_test(i.op, i.op0(), i.op1(), w, i.lock);
                Flow::Next
            }
            Op::Bsf | Op::Bsr | Op::Tzcnt | Op::Lzcnt => {
                let src = self.rd(i.op1(), w);
                let src = self.narrow(src, w);
                let zero = self.k(ty, 0);
                let is_zero = self.b.cmp(CmpOp::Eq, ty, src, zero);
                let r = match i.op {
                    Op::Bsf | Op::Tzcnt => {
                        // ctz of a zero narrow value is the type width; cap
                        // to the operand width for tzcnt.
                        let c = self.b.un(UnOp::Ctz, ty, src);
                        if w.bits() < 32 {
                            let cap = self.k32(w.bits());
                            self.b.select(is_zero, cap, c, Ty::I32)
                        } else {
                            c
                        }
                    }
                    Op::Bsr => {
                        // bits-1-clz at the operand width.
                        let c = self.b.un(UnOp::Clz, ty, src);
                        let k = self.k(ty, (ty.bits() - 1) as u64);
                        let r = self.b.bin(BinOp::Sub, ty, k, c);
                        r
                    }
                    _ => {
                        // lzcnt at the operand width: clz minus the type's
                        // extra bits (zero-extended narrow values).
                        let c = self.b.un(UnOp::Clz, ty, src);
                        if w.bits() < ty.bits() {
                            let k = self.k(ty, (ty.bits() - w.bits()) as u64);
                            self.b.bin(BinOp::Sub, ty, c, k)
                        } else {
                            c
                        }
                    }
                };
                match i.op {
                    Op::Bsf | Op::Bsr => {
                        // Destination undefined when the source is zero; keep
                        // it, as hardware does.
                        let old = self.rd(i.op0(), w);
                        let v = self.b.select(is_zero, old, r, ty);
                        let v = self.narrow(v, w);
                        self.wr(i.op0(), w, v);
                        self.write_flag(GuestReg::Z, is_zero);
                    }
                    _ => {
                        let r = self.narrow(r, w);
                        self.wr(i.op0(), w, r);
                        self.write_flag(GuestReg::C, is_zero);
                        let z = self.is_zero(r, w);
                        self.write_flag(GuestReg::Z, z);
                    }
                }
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::Popcnt => {
                let src = self.rd(i.op1(), w);
                let src = self.narrow(src, w);
                let r = self.b.un(UnOp::Popcnt, ty, src);
                self.wr(i.op0(), w, r);
                let z = self.is_zero(src, w);
                let zero = self.k32(0);
                self.write_flag(GuestReg::Z, z);
                for f in [GuestReg::N, GuestReg::C, GuestReg::V, GuestReg::P, GuestReg::A] {
                    self.write_flag(f, zero);
                }
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::Xadd => {
                let (d, s) = (i.op0(), i.op1());
                let sv = self.rd(s, w);
                let old = match d {
                    Operand::Mem(m) => {
                        let (addr, offset) = self.addr(&m);
                        let addr = self.offset_addr(addr, offset);
                        self.b.def(Expr::Rmw {
                            op: RmwOp::Add,
                            addr,
                            val: sv,
                            sz: mem_sz(w),
                            ty,
                        })
                    }
                    _ => {
                        let dv = self.rd(d, w);
                        let r = self.b.bin(BinOp::Add, ty, dv, sv);
                        let r = self.narrow(r, w);
                        self.wr(d, w, r);
                        dv
                    }
                };
                let old = self.narrow(old, w);
                let r = self.b.bin(BinOp::Add, ty, old, sv);
                let r = self.narrow(r, w);
                self.wr(s, w, old);
                self.flags_add(old, sv, r, w, None, false);
                Flow::Next
            }
            Op::Cmpxchg => {
                let (d, s) = (i.op0(), i.op1());
                let acc = self.rd_reg(Reg::Gp(RAX, w));
                let sv = self.rd(s, w);
                let old = match d {
                    Operand::Mem(m) => {
                        let (addr, offset) = self.addr(&m);
                        let addr = self.offset_addr(addr, offset);
                        self.b.def(Expr::CmpXchg {
                            addr,
                            expected: acc,
                            replacement: sv,
                            sz: mem_sz(w),
                            ty,
                        })
                    }
                    _ => {
                        let dv = self.rd(d, w);
                        let eq = self.b.cmp(CmpOp::Eq, ty, dv, acc);
                        let nv = self.b.select(eq, sv, dv, ty);
                        self.wr(d, w, nv);
                        dv
                    }
                };
                let old = self.narrow(old, w);
                // Unequal: the accumulator receives what was there.
                let eq = self.b.cmp(CmpOp::Eq, ty, old, acc);
                let nacc = self.b.select(eq, acc, old, ty);
                self.wr_reg(Reg::Gp(RAX, w), nacc);
                let r = self.b.bin(BinOp::Sub, ty, acc, old);
                let r = self.narrow(r, w);
                self.flags_sub(acc, old, r, w, None, false);
                Flow::Next
            }
            Op::CmpxchgNb if w != Width::Q => {
                // cmpxchg8b: edx:eax against [m], replaced with ecx:ebx -- one
                // 64-bit atomic compare-exchange. It has to be: a lock-free
                // list's head is a pointer and a counter swapped together
                // (Windows' SLIST_HEADER, a game's job queues), and the two
                // halves loaded and stored separately -- stored back even when
                // unequal -- lose other threads' pushes and pops and hand out
                // torn pointers.
                let Operand::Mem(m) = i.op0() else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                let (addr, offset) = self.addr(&m);
                let addr = self.offset_addr(addr, offset);
                let pair = |s: &mut Self, hi: u8, lo: u8| {
                    let h = s.rd_reg(Reg::Gp(hi, Width::D));
                    let l = s.rd_reg(Reg::Gp(lo, Width::D));
                    let h = s.b.extend(h, false);
                    let l = s.b.extend(l, false);
                    let k = s.k64(32);
                    let h = s.b.bin(BinOp::Shl, Ty::I64, h, k);
                    s.b.bin(BinOp::Or, Ty::I64, h, l)
                };
                let expected = pair(self, RDX, RAX);
                let replacement = pair(self, RCX, 3);
                let old = self.b.def(Expr::CmpXchg {
                    addr,
                    expected,
                    replacement,
                    sz: mem_sz(Width::Q),
                    ty: Ty::I64,
                });
                let eq = self.b.cmp(CmpOp::Eq, Ty::I64, old, expected);
                // Equal, edx:eax already hold the old value; unequal, they
                // receive it.
                let lo = self.b.wrap(old);
                let k = self.k64(32);
                let hi = self.b.bin(BinOp::ShrU, Ty::I64, old, k);
                let hi = self.b.wrap(hi);
                self.wr_reg(Reg::Gp(RAX, Width::D), lo);
                self.wr_reg(Reg::Gp(RDX, Width::D), hi);
                self.write_flag(GuestReg::Z, eq);
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::CmpxchgNb => {
                // cmpxchg16b: rdx:rax against [m], replaced with rcx:rbx.
                // wasm has no 16-byte atomic, so the host does it, under one
                // lock every cmpxchg16b takes (gex_kernel::x86::
                // serve_cmpxchg16b): atomic against each other, which is
                // what a lock-free list's head (x64 Windows' SLIST_HEADER)
                // needs, not against a plain access racing it. It reads and
                // writes rax, rdx and ZF in the context.
                let Operand::Mem(m) = i.op0() else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                let (addr, offset) = self.addr(&m);
                let addr = self.offset_addr(addr, offset);
                let pcv = self.k64(pc);
                let id = self.k32(sysreg::CMPXCHG16B as u32);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::SysRegWrite,
                    args: vec![pcv, id, addr],
                });
                self.flags = FlagSource::Unknown;
                Flow::Next
            }

            // -- control flow ------------------------------------------------------------
            Op::Jmp => match i.op0() {
                Operand::Imm(t) => Flow::End(Term::Goto { target: t as u64 }),
                o => {
                    let target = self.rd_target(o);
                    Flow::End(Term::Indirect {
                        target,
                        is_ret: false,
                    })
                }
            },
            Op::Jcc(c) => {
                let Operand::Imm(t) = i.op0() else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                let cond = self.cond(c);
                Flow::End(Term::Branch {
                    cond,
                    taken: t as u64,
                    not_taken: next,
                })
            }
            Op::Jrcxz | Op::Loop | Op::Loope | Op::Loopne => {
                let Operand::Imm(t) = i.op0() else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                let cw = w; // Q, or D with 0x67
                let mut rcx = self.rd_reg(Reg::Gp(RCX, cw));
                if i.op != Op::Jrcxz {
                    let one = self.k(ty_of(cw), 1);
                    rcx = self.b.bin(BinOp::Sub, ty_of(cw), rcx, one);
                    self.wr_reg(Reg::Gp(RCX, cw), rcx);
                }
                let zero = self.k(ty_of(cw), 0);
                let cond = match i.op {
                    Op::Jrcxz => self.b.cmp(CmpOp::Eq, ty_of(cw), rcx, zero),
                    Op::Loop => self.b.cmp(CmpOp::Ne, ty_of(cw), rcx, zero),
                    Op::Loope => {
                        let nz = self.b.cmp(CmpOp::Ne, ty_of(cw), rcx, zero);
                        let z = self.cond(Cond(4));
                        self.b.bin(BinOp::And, Ty::I32, nz, z)
                    }
                    _ => {
                        let nz = self.b.cmp(CmpOp::Ne, ty_of(cw), rcx, zero);
                        let z = self.cond(Cond(5));
                        self.b.bin(BinOp::And, Ty::I32, nz, z)
                    }
                };
                Flow::End(Term::Branch {
                    cond,
                    taken: t as u64,
                    not_taken: next,
                })
            }
            Op::Call => {
                let target = match i.op0() {
                    Operand::Imm(t) => None.or(Some(t as u64)).map(|t| (t, None)),
                    o => {
                        let v = self.rd_target(o);
                        Some((0, Some(v)))
                    }
                };
                let wa = self.wa();
                let ret = if wa == Width::Q { self.k64(next) } else { self.k32(next as u32) };
                self.push(wa, ret);
                self.b.set_call_return(next);
                match target {
                    Some((t, None)) => Flow::End(Term::Goto { target: t }),
                    Some((_, Some(v))) => Flow::End(Term::Indirect {
                        target: v,
                        is_ret: false,
                    }),
                    None => unreachable!(),
                }
            }
            Op::Ret => {
                let wa = self.wa();
                let target = self.pop(wa);
                let target = if wa == Width::Q { target } else { self.b.extend(target, false) };
                if let Some(Operand::Imm(n)) = i.ops[0] {
                    let sp = self.rd64(RSP);
                    let k = self.k64(n as u64);
                    let sp = self.b.bin(BinOp::Add, Ty::I64, sp, k);
                    self.wr64(RSP, sp);
                }
                Flow::End(Term::Indirect {
                    target,
                    is_ret: true,
                })
            }
            Op::JmpFar | Op::CallFar => {
                let (sel, off) = match (i.op0(), i.ops[1]) {
                    (Operand::Imm(off), Some(Operand::Imm(sel))) => (self.k32(sel as u32), self.k64(off as u64)),
                    (Operand::Mem(m), _) => {
                        let off = self.load(&m, w);
                        let off = if w == Width::Q { off } else { self.b.extend(off, false) };
                        let (addr, offset) = self.addr(&m);
                        let sel = self.b.def(Expr::Load {
                            addr,
                            offset: offset + w.bytes() as i64,
                            sz: MemSz::B2,
                            signed: false,
                            ty: Ty::I32,
                            order: Order::Relaxed,
                        });
                        (sel, off)
                    }
                    _ => return Flow::End(Term::Illegal { raw: self.raw, pc }),
                };
                if i.op == Op::CallFar {
                    // The caller's selector, then the return offset, each a
                    // slot of the operand width.
                    let cs = self.sel_val(Seg::Cs, w);
                    self.push(w, cs);
                    let ret = if w == Width::Q { self.k64(next) } else { self.k(ty_of(w), next & 0xffff_ffff) };
                    self.push(w, ret);
                    self.b.set_call_return(next);
                }
                self.far(pc, sel, off)
            }
            Op::RetFar => {
                let off = self.pop(w);
                let off = if w == Width::Q { off } else { self.b.extend(off, false) };
                let sel = self.pop(w);
                let sel = if w == Width::Q { self.b.wrap(sel) } else { sel };
                if let Some(Operand::Imm(n)) = i.ops[0] {
                    let sp = self.rd64(RSP);
                    let k = self.k64(n as u64);
                    let sp = self.b.bin(BinOp::Add, Ty::I64, sp, k);
                    self.wr64(RSP, sp);
                }
                self.far(pc, sel, off)
            }
            Op::Iret => {
                let off = self.pop(w);
                let off = if w == Width::Q { off } else { self.b.extend(off, false) };
                let sel = self.pop(w);
                let sel = if w == Width::Q { self.b.wrap(sel) } else { sel };
                let fl = self.pop(w);
                let fl = if w == Width::Q { self.b.wrap(fl) } else { fl };
                // 64-bit mode's iret pops the stack pointer and segment too,
                // whatever mode it returns to.
                if self.wa() == Width::Q {
                    let sp = self.pop(w);
                    let sp = if w == Width::Q { sp } else { self.b.extend(sp, false) };
                    let _ss = self.pop(w);
                    self.wr64(RSP, sp);
                }
                self.set_rflags(fl);
                self.far(pc, sel, off)
            }
            Op::Leave => {
                let rbp = self.rd64(RBP);
                self.wr64(RSP, rbp);
                let v = self.pop(w);
                self.wr_reg(Reg::Gp(RBP, w), v);
                Flow::Next
            }
            Op::Pusha => {
                // eax, ecx, edx, ebx, the esp from before, ebp, esi, edi.
                let sp = self.rd(Operand::Reg(Reg::Gp(RSP, w)), w);
                for r in 0..8 {
                    let v = if r == RSP { sp } else { self.rd(Operand::Reg(Reg::Gp(r, w)), w) };
                    self.push(w, v);
                }
                Flow::Next
            }
            Op::Popa => {
                // The reverse, skipping the saved esp.
                for r in (0..8).rev() {
                    let v = self.pop(w);
                    if r != RSP {
                        self.wr_reg(Reg::Gp(r, w), v);
                    }
                }
                Flow::Next
            }
            Op::Enter => {
                // enter imm16, 0: push rbp; mov rbp, rsp; sub rsp, imm.
                let Some(Operand::Imm(size)) = i.ops[0] else {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                };
                let wa = self.wa();
                let rbp = self.rd64(RBP);
                let rbp = if wa == Width::Q { rbp } else { self.b.wrap(rbp) };
                self.push(wa, rbp);
                let sp = self.rd64(RSP);
                self.wr64(RBP, sp);
                let k = self.k64((size as u64).wrapping_neg());
                let sp = self.b.bin(BinOp::Add, Ty::I64, sp, k);
                self.wr64(RSP, sp);
                Flow::Next
            }
            Op::Syscall => {
                // The kernel returns with rcx = the return address and r11 =
                // rflags; both are clobbers a guest may not rely on beyond
                // "clobbered", so a plausible value in each is enough.
                let n = self.k64(next);
                self.wr64(RCX, n);
                let fl = self.k64(0x246);
                self.wr64(R11, fl);
                Flow::End(Term::Syscall { next, imm: 0 })
            }
            // `int $3` (cd 03) is a breakpoint like int3; its gate is open
            // to user code.
            Op::Int3 | Op::Int if i.op == Op::Int3 || i.ops[0] == Some(Operand::Imm(3)) => {
                let pcv = self.k64(pc);
                let immv = self.k32(3);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::Breakpoint,
                    args: vec![pcv, immv],
                });
                Flow::End(Term::Goto { target: next })
            }
            Op::Cpuid => {
                let eax = self.rd_reg(Reg::Gp(RAX, Width::D));
                let ecx = self.rd_reg(Reg::Gp(RCX, Width::D));
                let eax = self.b.extend(eax, false);
                let ecx = self.b.extend(ecx, false);
                let sh = self.k64(32);
                let ecx = self.b.bin(BinOp::Shl, Ty::I64, ecx, sh);
                let v = self.b.bin(BinOp::Or, Ty::I64, eax, ecx);
                let pcv = self.k64(pc);
                let id = self.k32(sysreg::CPUID as u32);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::SysRegWrite,
                    args: vec![pcv, id, v],
                });
                Flow::Next
            }
            Op::Rdtsc | Op::Rdtscp => {
                let pcv = self.k64(pc);
                let id = self.k32(sysreg::TSC as u32);
                let v = self.b.def(Expr::HostCall {
                    func: HostFunc::SysRegRead,
                    args: vec![pcv, id],
                    ty: Ty::I64,
                });
                let lo = self.b.wrap(v);
                let sh = self.k64(32);
                let hi = self.b.bin(BinOp::ShrU, Ty::I64, v, sh);
                let hi = self.b.wrap(hi);
                self.wr_reg(Reg::Gp(RAX, Width::D), lo);
                self.wr_reg(Reg::Gp(RDX, Width::D), hi);
                if i.op == Op::Rdtscp {
                    let z = self.k32(0);
                    self.wr_reg(Reg::Gp(RCX, Width::D), z);
                }
                Flow::Next
            }
            Op::Xgetbv => {
                // XCR0 = x87 | SSE: what the advertised CPUID promises.
                let lo = self.k32(3);
                let hi = self.k32(0);
                self.wr_reg(Reg::Gp(RAX, Width::D), lo);
                self.wr_reg(Reg::Gp(RDX, Width::D), hi);
                Flow::Next
            }
            Op::Rdrand | Op::Rdseed => {
                // Not advertised; a guest that probes anyway gets CF=0 (no
                // random value available).
                let zero = self.k32(0);
                self.write_flag(GuestReg::C, zero);
                self.flags = FlagSource::Unknown;
                Flow::Next
            }

            // -- flags -------------------------------------------------------------------
            Op::Clc | Op::Stc | Op::Cmc => {
                let v = match i.op {
                    Op::Clc => self.k32(0),
                    Op::Stc => self.k32(1),
                    _ => {
                        let c = self.read_flag(GuestReg::C);
                        let one = self.k32(1);
                        self.b.bin(BinOp::Xor, Ty::I32, c, one)
                    }
                };
                self.write_flag(GuestReg::C, v);
                self.flags = FlagSource::Unknown;
                Flow::Next
            }
            Op::Cld | Op::Std => {
                let v = self.k32(if i.op == Op::Std { 1 } else { 0 });
                self.write_flag(GuestReg::D, v);
                Flow::Next
            }
            Op::Lahf => {
                let v = self.flags_byte();
                self.wr_reg(Reg::High(0), v);
                Flow::Next
            }
            Op::Sahf => {
                let v = self.rd_reg(Reg::High(0));
                self.unpack_flags_low(v);
                Flow::Next
            }
            Op::Pushf => {
                let v = self.rflags();
                let v = if w == Width::Q { self.b.extend(v, false) } else { v };
                self.push(w, v);
                Flow::Next
            }
            Op::Popf => {
                let v = self.pop(w);
                let v = if w == Width::Q { self.b.wrap(v) } else { v };
                self.set_rflags(v);
                Flow::Next
            }

            // -- strings ------------------------------------------------------------------
            Op::Movs | Op::Stos | Op::Lods | Op::Scas | Op::Cmps => self.string_op(i),

            // -- SSE / x87 ------------------------------------------------------------------
            Op::Fld(_)
            | Op::Fild(_)
            | Op::Fst { .. }
            | Op::Fist { .. }
            | Op::FldSt(_)
            | Op::FstSt { .. }
            | Op::Fxch(_)
            | Op::Fadd(_)
            | Op::Fsub(_)
            | Op::Fsubr(_)
            | Op::Fmul(_)
            | Op::Fdiv(_)
            | Op::Fdivr(_)
            | Op::Fcom { .. }
            | Op::Fucom { .. }
            | Op::Fcomi { .. }
            | Op::Ficom { .. }
            | Op::Fchs
            | Op::Fabs
            | Op::Ftst
            | Op::Fxam
            | Op::Fld1
            | Op::Fldl2t
            | Op::Fldl2e
            | Op::Fldpi
            | Op::Fldlg2
            | Op::Fldln2
            | Op::Fldz
            | Op::F2xm1
            | Op::Fyl2x
            | Op::Fptan
            | Op::Fpatan
            | Op::Fxtract
            | Op::Fprem1
            | Op::Fdecstp
            | Op::Fincstp
            | Op::Fprem
            | Op::Fyl2xp1
            | Op::Fsqrt
            | Op::Fsincos
            | Op::Frndint
            | Op::Fscale
            | Op::Fsin
            | Op::Fcos
            | Op::Fnop
            | Op::Fninit
            | Op::Fnclex
            | Op::Fldcw
            | Op::Fnstcw
            | Op::Fnstsw
            | Op::FnstswAx
            | Op::Fldenv
            | Op::Fnstenv
            | Op::Frstor
            | Op::Fnsave
            | Op::Fbld
            | Op::Fbstp
            | Op::Ffree(_)
            | Op::Ffreep(_)
            | Op::Fcmovcc { .. } => self.x87(i),

            Op::Ldmxcsr => {
                let v = self.rd(i.op0(), Width::D);
                self.b.write(GuestReg::Mxcsr, v);
                Flow::Next
            }
            Op::Stmxcsr => {
                let v = self.b.read(GuestReg::Mxcsr);
                self.wr(i.op0(), Width::D, v);
                Flow::Next
            }
            Op::Fxsave | Op::Fxrstor | Op::Xsave | Op::Xrstor => self.fxsave(i),

            Op::Crc32
            | Op::Andn
            | Op::Bzhi
            | Op::Blsr
            | Op::Blsmsk
            | Op::Blsi
            | Op::Sarx
            | Op::Shlx
            | Op::Shrx
            | Op::Bextr => Flow::End(Term::Illegal { raw: self.raw, pc }),

            _ => self.sse(i),
        }
    }

    // -- helpers for the arms above ----------------------------------------------------

    fn widen(&mut self, v: Value, from: Width, to: Width, signed: bool) -> Value {
        let v = if signed { self.sext(v, from) } else { self.narrow(v, from) };
        match (ty_of(from), ty_of(to)) {
            (Ty::I32, Ty::I64) => self.b.extend(v, signed),
            (Ty::I32, Ty::I32) => self.narrow(v, to),
            _ => v,
        }
    }

    fn offset_addr(&mut self, addr: Value, offset: i64) -> Value {
        if offset == 0 {
            addr
        } else {
            let d = self.k64(offset as u64);
            self.b.bin(BinOp::Add, Ty::I64, addr, d)
        }
    }

    /// `hi64(a * b)` for signed 64-bit operands, from 32-bit pieces.
    fn mulh_s64(&mut self, a: Value, b: Value) -> Value {
        // Unsigned high product, then correct for signs:
        // mulhs(a,b) = mulhu(a,b) - (a<0 ? b : 0) - (b<0 ? a : 0).
        let hu = self.mulh_u64(a, b);
        let sh = self.k64(63);
        let sa = self.b.bin(BinOp::ShrS, Ty::I64, a, sh); // 0 or -1
        let sb = self.b.bin(BinOp::ShrS, Ty::I64, b, sh);
        let ta = self.b.bin(BinOp::And, Ty::I64, sa, b);
        let tb = self.b.bin(BinOp::And, Ty::I64, sb, a);
        let r = self.b.bin(BinOp::Sub, Ty::I64, hu, ta);
        self.b.bin(BinOp::Sub, Ty::I64, r, tb)
    }

    /// `hi64(a * b)` for unsigned 64-bit operands.
    fn mulh_u64(&mut self, a: Value, b: Value) -> Value {
        let m32 = self.k64(0xffff_ffff);
        let sh = self.k64(32);
        let a0 = self.b.bin(BinOp::And, Ty::I64, a, m32);
        let a1 = self.b.bin(BinOp::ShrU, Ty::I64, a, sh);
        let b0 = self.b.bin(BinOp::And, Ty::I64, b, m32);
        let b1 = self.b.bin(BinOp::ShrU, Ty::I64, b, sh);
        let p00 = self.b.bin(BinOp::Mul, Ty::I64, a0, b0);
        let p01 = self.b.bin(BinOp::Mul, Ty::I64, a0, b1);
        let p10 = self.b.bin(BinOp::Mul, Ty::I64, a1, b0);
        let p11 = self.b.bin(BinOp::Mul, Ty::I64, a1, b1);
        // mid = (p00 >> 32) + (p01 & m32) + (p10 & m32)
        let p00h = self.b.bin(BinOp::ShrU, Ty::I64, p00, sh);
        let p01l = self.b.bin(BinOp::And, Ty::I64, p01, m32);
        let p10l = self.b.bin(BinOp::And, Ty::I64, p10, m32);
        let mid = self.b.bin(BinOp::Add, Ty::I64, p00h, p01l);
        let mid = self.b.bin(BinOp::Add, Ty::I64, mid, p10l);
        let midh = self.b.bin(BinOp::ShrU, Ty::I64, mid, sh);
        let p01h = self.b.bin(BinOp::ShrU, Ty::I64, p01, sh);
        let p10h = self.b.bin(BinOp::ShrU, Ty::I64, p10, sh);
        let hi = self.b.bin(BinOp::Add, Ty::I64, p11, p01h);
        let hi = self.b.bin(BinOp::Add, Ty::I64, hi, p10h);
        self.b.bin(BinOp::Add, Ty::I64, hi, midh)
    }

    fn byte_reverse(&mut self, v: Value, w: Width) -> Value {
        let ty = ty_of(w);
        let n = w.bytes();
        let mut acc: Option<Value> = None;
        for i in 0..n {
            let from = i * 8;
            let to = (n - 1 - i) * 8;
            let m = self.k(ty, 0xffu64 << from);
            let piece = self.b.bin(BinOp::And, ty, v, m);
            let piece = if to > from {
                let s = self.k(ty, (to - from) as u64);
                self.b.bin(BinOp::Shl, ty, piece, s)
            } else if from > to {
                let s = self.k(ty, (from - to) as u64);
                self.b.bin(BinOp::ShrU, ty, piece, s)
            } else {
                piece
            };
            acc = Some(match acc {
                None => piece,
                Some(a) => self.b.bin(BinOp::Or, ty, a, piece),
            });
        }
        acc.unwrap()
    }

    /// Two-operand ALU ops and the compares.
    fn alu(&mut self, op: Op, d: Operand, s: Operand, w: Width, lock: bool) {
        let ty = ty_of(w);
        let sv = self.rd(s, w);
        let sv = self.narrow(sv, w);
        // The atomic forms read the old value through the RMW itself.
        let rmw = match (lock, d, op) {
            (true, Operand::Mem(m), Op::Add | Op::Sub | Op::And | Op::Or | Op::Xor | Op::Adc | Op::Sbb) => {
                Some((m, op))
            }
            _ => None,
        };
        let a = match rmw {
            Some((m, rop)) => {
                let (addr, offset) = self.addr(&m);
                let addr = self.offset_addr(addr, offset);
                let (rmw_op, val) = match rop {
                    Op::Add => (RmwOp::Add, sv),
                    Op::Sub => (RmwOp::Sub, sv),
                    Op::And => (RmwOp::And, sv),
                    Op::Or => (RmwOp::Or, sv),
                    Op::Xor => (RmwOp::Xor, sv),
                    Op::Adc => {
                        let c = self.read_flag(GuestReg::C);
                        let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                        (RmwOp::Add, self.b.bin(BinOp::Add, ty, sv, c))
                    }
                    _ => {
                        let c = self.read_flag(GuestReg::C);
                        let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                        (RmwOp::Sub, self.b.bin(BinOp::Add, ty, sv, c))
                    }
                };
                let old = self.b.def(Expr::Rmw {
                    op: rmw_op,
                    addr,
                    val,
                    sz: mem_sz(w),
                    ty,
                });
                self.narrow(old, w)
            }
            None => {
                let a = self.rd(d, w);
                self.narrow(a, w)
            }
        };
        match op {
            Op::Add | Op::Adc => {
                let cin = if op == Op::Adc {
                    let c = self.read_flag(GuestReg::C);
                    Some(c)
                } else {
                    None
                };
                let mut r = self.b.bin(BinOp::Add, ty, a, sv);
                if let Some(c) = cin {
                    let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                    r = self.b.bin(BinOp::Add, ty, r, c);
                }
                let r = self.narrow(r, w);
                if rmw.is_none() {
                    self.wr(d, w, r);
                }
                self.flags_add(a, sv, r, w, cin, false);
            }
            Op::Sub | Op::Sbb | Op::Cmp => {
                let bin = if op == Op::Sbb {
                    let c = self.read_flag(GuestReg::C);
                    Some(c)
                } else {
                    None
                };
                let mut r = self.b.bin(BinOp::Sub, ty, a, sv);
                if let Some(c) = bin {
                    let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                    r = self.b.bin(BinOp::Sub, ty, r, c);
                }
                let r = self.narrow(r, w);
                if op != Op::Cmp && rmw.is_none() {
                    self.wr(d, w, r);
                }
                self.flags_sub(a, sv, r, w, bin, false);
            }
            Op::And | Op::Or | Op::Xor | Op::Test => {
                let bop = match op {
                    Op::And | Op::Test => BinOp::And,
                    Op::Or => BinOp::Or,
                    _ => BinOp::Xor,
                };
                let r = self.b.bin(bop, ty, a, sv);
                if op != Op::Test && rmw.is_none() {
                    self.wr(d, w, r);
                }
                self.flags_logic(r, w);
            }
            _ => unreachable!(),
        }
    }

    /// One-operand multiply into rdx:rax (or ax for bytes).
    fn mul1(&mut self, s: Operand, w: Width, signed: bool) {
        let ty = ty_of(w);
        let sv = self.rd(s, w);
        let sv = self.narrow(sv, w);
        let av = self.rd_reg(Reg::Gp(RAX, w));
        match w {
            Width::B | Width::W => {
                let (a, b) = if signed {
                    (self.sext(av, w), self.sext(sv, w))
                } else {
                    (av, sv)
                };
                let full = self.b.bin(BinOp::Mul, Ty::I32, a, b);
                let bits = w.bits();
                let lo = self.narrow(full, w);
                let sh = self.k32(bits);
                let hi = self.b.bin(BinOp::ShrU, Ty::I32, full, sh);
                let hi = self.narrow(hi, w);
                if w == Width::B {
                    // ax = full product.
                    let m = self.k32(0xffff);
                    let ax = self.b.bin(BinOp::And, Ty::I32, full, m);
                    self.wr_reg(Reg::Gp(RAX, Width::W), ax);
                } else {
                    self.wr_reg(Reg::Gp(RAX, w), lo);
                    self.wr_reg(Reg::Gp(RDX, w), hi);
                }
                let of = if signed {
                    let s = self.sext(lo, w);
                    self.b.cmp(CmpOp::Ne, Ty::I32, s, full)
                } else {
                    let z = self.k32(0);
                    self.b.cmp(CmpOp::Ne, Ty::I32, hi, z)
                };
                self.write_flag(GuestReg::C, of);
                self.write_flag(GuestReg::V, of);
            }
            Width::D => {
                let (a, b) = (self.b.extend(av, signed), self.b.extend(sv, signed));
                let full = self.b.bin(BinOp::Mul, Ty::I64, a, b);
                let lo = self.b.wrap(full);
                let sh = self.k64(32);
                let hi = self.b.bin(BinOp::ShrU, Ty::I64, full, sh);
                let hi = self.b.wrap(hi);
                self.wr_reg(Reg::Gp(RAX, Width::D), lo);
                self.wr_reg(Reg::Gp(RDX, Width::D), hi);
                let of = if signed {
                    let s = self.b.extend(lo, true);
                    self.b.cmp(CmpOp::Ne, Ty::I64, s, full)
                } else {
                    let z = self.k32(0);
                    self.b.cmp(CmpOp::Ne, Ty::I32, hi, z)
                };
                self.write_flag(GuestReg::C, of);
                self.write_flag(GuestReg::V, of);
            }
            Width::Q => {
                let lo = self.b.bin(BinOp::Mul, Ty::I64, av, sv);
                let hi = if signed {
                    self.mulh_s64(av, sv)
                } else {
                    self.mulh_u64(av, sv)
                };
                self.wr64(RAX, lo);
                self.wr64(RDX, hi);
                let of = if signed {
                    let sh = self.k64(63);
                    let sign = self.b.bin(BinOp::ShrS, Ty::I64, lo, sh);
                    self.b.cmp(CmpOp::Ne, Ty::I64, hi, sign)
                } else {
                    let z = self.k64(0);
                    self.b.cmp(CmpOp::Ne, Ty::I64, hi, z)
                };
                self.write_flag(GuestReg::C, of);
                self.write_flag(GuestReg::V, of);
            }
        }
        let _ = ty;
        self.flags = FlagSource::Unknown;
    }

    /// `div`/`idiv`: rdx:rax by the operand. Division by zero and quotient
    /// overflow raise #DE, which Linux delivers as SIGFPE; here that is an
    /// `Illegal` terminator guarded at run time, so the block is split: the
    /// division happens only when safe.
    /// Fault the current instruction with `code` when `cond` holds.
    fn trap_if(&mut self, cond: Value, code: i32) {
        self.b.stmt(Stmt::Trap {
            cond,
            pc: self.pc,
            code,
        });
    }

    /// `quotient` (signed, in `ty`) does not fit a `w`-bit signed result.
    fn quotient_overflows_signed(&mut self, q: Value, ty: Ty, w: Width) -> Value {
        let half = 1u64 << (w.bits() - 1);
        let bias = self.k(ty, half);
        let biased = self.b.bin(BinOp::Add, ty, q, bias);
        let span = self.k(ty, 2 * half - 1);
        self.b.cmp(CmpOp::GtU, ty, biased, span)
    }

    fn div(&mut self, s: Operand, w: Width, signed: bool) -> Flow {
        let ty = ty_of(w);
        let sv = self.rd(s, w);
        let sv = self.narrow(sv, w);
        // x86 raises #DE for a zero divisor and for a quotient too wide for
        // the destination; both fault before any register changes. The
        // divisions themselves still guard the divisor, since wasm traps on
        // /0 and INT_MIN/-1: the values they compute on the faulting path
        // are never written.
        {
            let zero = self.k(ty, 0);
            let is_zero = self.b.cmp(CmpOp::Eq, ty, sv, zero);
            self.trap_if(is_zero, gex_abi::imports::BRK_DIVIDE_BY_ZERO);
        }
        match w {
            Width::B => {
                let ax = self.rd_reg(Reg::Gp(RAX, Width::W));
                let (n, d) = if signed {
                    (self.sext(ax, Width::W), self.sext(sv, Width::B))
                } else {
                    (ax, sv)
                };
                let d = self.safe_divisor(d, Ty::I32, signed.then_some(n));
                let q = self.b.bin(if signed { BinOp::DivS } else { BinOp::DivU }, Ty::I32, n, d);
                let r = self.b.bin(if signed { BinOp::RemS } else { BinOp::RemU }, Ty::I32, n, d);
                let over = if signed {
                    self.quotient_overflows_signed(q, Ty::I32, Width::B)
                } else {
                    let ff = self.k32(0xff);
                    self.b.cmp(CmpOp::GtU, Ty::I32, q, ff)
                };
                self.trap_if(over, gex_abi::imports::BRK_DIVIDE_OVERFLOW);
                let q = self.narrow(q, Width::B);
                let r = self.narrow(r, Width::B);
                let eight = self.k32(8);
                let r = self.b.bin(BinOp::Shl, Ty::I32, r, eight);
                let ax = self.b.bin(BinOp::Or, Ty::I32, q, r);
                self.wr_reg(Reg::Gp(RAX, Width::W), ax);
            }
            Width::W | Width::D => {
                let lo = self.rd_reg(Reg::Gp(RAX, w));
                let hi = self.rd_reg(Reg::Gp(RDX, w));
                let lo = self.b.extend(lo, false);
                let hi = self.b.extend(hi, false);
                let sh = self.k64(w.bits() as u64);
                let hi = self.b.bin(BinOp::Shl, Ty::I64, hi, sh);
                let n = self.b.bin(BinOp::Or, Ty::I64, lo, hi);
                let n = if w == Width::W && signed {
                    self.b.def(Expr::ExtendBits {
                        a: n,
                        ty: Ty::I64,
                        bits: 32,
                        signed: true,
                    })
                } else {
                    n
                };
                let d = if signed {
                    let s = self.sext(sv, w);
                    self.b.extend(s, true)
                } else {
                    self.b.extend(sv, false)
                };
                let d = self.safe_divisor(d, Ty::I64, signed.then_some(n));
                let q = self.b.bin(if signed { BinOp::DivS } else { BinOp::DivU }, Ty::I64, n, d);
                let r = self.b.bin(if signed { BinOp::RemS } else { BinOp::RemU }, Ty::I64, n, d);
                // The 64-bit quotient is exact (a signed MIN / -1 comes out
                // as MIN, itself out of range), so range-check it.
                let over = if signed {
                    self.quotient_overflows_signed(q, Ty::I64, w)
                } else {
                    let max = self.k64((1u64 << w.bits()) - 1);
                    self.b.cmp(CmpOp::GtU, Ty::I64, q, max)
                };
                self.trap_if(over, gex_abi::imports::BRK_DIVIDE_OVERFLOW);
                let q = self.b.wrap(q);
                let r = self.b.wrap(r);
                let q = self.narrow(q, w);
                let r = self.narrow(r, w);
                self.wr_reg(Reg::Gp(RAX, w), q);
                self.wr_reg(Reg::Gp(RDX, w), r);
            }
            Width::Q => {
                // 128-by-64 division of rdx:rax. glibc's printf and strtod
                // (`__mpn_divrem`) depend on the full-width form.
                let lo = self.rd64(RAX);
                let hi = self.rd64(RDX);
                let d = self.safe_divisor(sv, Ty::I64, None);
                let (q, r) = if signed {
                    // Divide magnitudes, then fix the signs: the quotient
                    // takes the operands' sign product, the remainder the
                    // dividend's.
                    let sh = self.k64(63);
                    let nsign = self.b.bin(BinOp::ShrS, Ty::I64, hi, sh); // 0 or -1
                    let dsign = self.b.bin(BinOp::ShrS, Ty::I64, d, sh);
                    let (ahi, alo) = self.neg128_if(hi, lo, nsign);
                    let ad = self.b.bin(BinOp::Xor, Ty::I64, d, dsign);
                    let ad = self.b.bin(BinOp::Sub, Ty::I64, ad, dsign);
                    let (q, r) = self.divlu(ahi, alo, ad);
                    let qsign = self.b.bin(BinOp::Xor, Ty::I64, nsign, dsign);
                    // The magnitude must fit: below 2^64 (|hi| < |d|), and
                    // at most 2^63 - 1, or 2^63 for a negative quotient.
                    let wide = self.b.cmp(CmpOp::GeU, Ty::I64, ahi, ad);
                    let one = self.k64(1);
                    let neg = self.b.bin(BinOp::And, Ty::I64, qsign, one);
                    let lim = self.k64((1u64 << 63) - 1);
                    let lim = self.b.bin(BinOp::Add, Ty::I64, lim, neg);
                    let big = self.b.cmp(CmpOp::GtU, Ty::I64, q, lim);
                    let over = self.b.bin(BinOp::Or, Ty::I32, wide, big);
                    self.trap_if(over, gex_abi::imports::BRK_DIVIDE_OVERFLOW);
                    let q = self.b.bin(BinOp::Xor, Ty::I64, q, qsign);
                    let q = self.b.bin(BinOp::Sub, Ty::I64, q, qsign);
                    let r = self.b.bin(BinOp::Xor, Ty::I64, r, nsign);
                    let r = self.b.bin(BinOp::Sub, Ty::I64, r, nsign);
                    (q, r)
                } else {
                    let over = self.b.cmp(CmpOp::GeU, Ty::I64, hi, d);
                    self.trap_if(over, gex_abi::imports::BRK_DIVIDE_OVERFLOW);
                    self.divlu(hi, lo, d)
                };
                self.wr64(RAX, q);
                self.wr64(RDX, r);
            }
        }
        let _ = ty;
        Flow::Next
    }

    /// Two's-complement negate the 128-bit `hi:lo` when `neg` is -1 (leave it
    /// when 0).
    fn neg128_if(&mut self, hi: Value, lo: Value, neg: Value) -> (Value, Value) {
        let lo_x = self.b.bin(BinOp::Xor, Ty::I64, lo, neg);
        let hi_x = self.b.bin(BinOp::Xor, Ty::I64, hi, neg);
        // (hi_x:lo_x) - neg  ==  (hi_x:lo_x) + 1 when neg == -1
        let one = self.k64(1);
        let inc = self.b.bin(BinOp::And, Ty::I64, neg, one); // 0 or 1
        let nlo = self.b.bin(BinOp::Add, Ty::I64, lo_x, inc);
        let carry = self.b.cmp(CmpOp::LtU, Ty::I64, nlo, lo_x);
        let carry = self.b.extend(carry, false);
        let nhi = self.b.bin(BinOp::Add, Ty::I64, hi_x, carry);
        (nhi, nlo)
    }

    /// Unsigned 128-by-64 division `(u1:u0) / v` with `v != 0`, returning
    /// `(quotient, remainder)`. Hacker's Delight's `divlu` (two 32-bit digit
    /// steps on a normalised divisor, each with its at-most-two corrections
    /// unrolled), so it is straight-line code the IR can express. When
    /// `u1 >= v` the hardware raises #DE, which the caller's `Stmt::Trap`
    /// does before anything this computes is written.
    fn divlu(&mut self, u1: Value, u0: Value, v: Value) -> (Value, Value) {
        let ty = Ty::I64;
        let k32 = self.k64(32);
        let k64 = self.k64(64);
        let m32 = self.k64(0xffff_ffff);
        let b = self.k64(1u64 << 32);
        let zero = self.k64(0);
        let one = self.k64(1);

        let s = self.b.un(UnOp::Clz, ty, v);
        let vn = self.b.bin(BinOp::Shl, ty, v, s);
        let vn1 = self.b.bin(BinOp::ShrU, ty, vn, k32);
        let vn0 = self.b.bin(BinOp::And, ty, vn, m32);

        // un32 = (u1 << s) | (s ? u0 >> (64 - s) : 0)
        let u1s = self.b.bin(BinOp::Shl, ty, u1, s);
        let rs = self.b.bin(BinOp::Sub, ty, k64, s);
        let u0r = self.b.bin(BinOp::ShrU, ty, u0, rs);
        let s_zero = self.b.cmp(CmpOp::Eq, ty, s, zero);
        let u0r = self.b.select(s_zero, zero, u0r, ty);
        let un32 = self.b.bin(BinOp::Or, ty, u1s, u0r);
        let un10 = self.b.bin(BinOp::Shl, ty, u0, s);
        let un1 = self.b.bin(BinOp::ShrU, ty, un10, k32);
        let un0 = self.b.bin(BinOp::And, ty, un10, m32);

        // First digit.
        let (q1, un21) = {
            let q = self.b.bin(BinOp::DivU, ty, un32, vn1);
            let qv = self.b.bin(BinOp::Mul, ty, q, vn1);
            let rhat = self.b.bin(BinOp::Sub, ty, un32, qv);
            let (q, _rhat) = self.divlu_correct(q, rhat, vn1, vn0, un1, b, one);
            // un21 = un32*b + un1 - q*vn
            let t = self.b.bin(BinOp::Shl, ty, un32, k32);
            let t = self.b.bin(BinOp::Add, ty, t, un1);
            let qvn = self.b.bin(BinOp::Mul, ty, q, vn);
            let un21 = self.b.bin(BinOp::Sub, ty, t, qvn);
            (q, un21)
        };
        // Second digit.
        let q0 = {
            let q = self.b.bin(BinOp::DivU, ty, un21, vn1);
            let qv = self.b.bin(BinOp::Mul, ty, q, vn1);
            let rhat = self.b.bin(BinOp::Sub, ty, un21, qv);
            let (q, _rhat) = self.divlu_correct(q, rhat, vn1, vn0, un0, b, one);
            q
        };
        let q1b = self.b.bin(BinOp::Shl, ty, q1, k32);
        let q = self.b.bin(BinOp::Or, ty, q1b, q0);
        // r = (un21*b + un0 - q0*vn) >> s
        let t = self.b.bin(BinOp::Shl, ty, un21, k32);
        let t = self.b.bin(BinOp::Add, ty, t, un0);
        let q0vn = self.b.bin(BinOp::Mul, ty, q0, vn);
        let r = self.b.bin(BinOp::Sub, ty, t, q0vn);
        let r = self.b.bin(BinOp::ShrU, ty, r, s);
        (q, r)
    }

    /// The `divlu` digit correction: while `q >= b || q*vn0 > b*rhat + un`,
    /// decrement `q` and add `vn1` to `rhat`, stopping once `rhat >= b`.
    /// At most two rounds are ever needed, so both are unrolled.
    #[allow(clippy::too_many_arguments)]
    fn divlu_correct(
        &mut self,
        q: Value,
        rhat: Value,
        vn1: Value,
        vn0: Value,
        un: Value,
        b: Value,
        one: Value,
    ) -> (Value, Value) {
        let ty = Ty::I64;
        let k32 = self.k64(32);
        let mut q = q;
        let mut rhat = rhat;
        let mut go = None;
        for _ in 0..2 {
            let qb = self.b.cmp(CmpOp::GeU, ty, q, b);
            let lhs = self.b.bin(BinOp::Mul, ty, q, vn0);
            let rb = self.b.bin(BinOp::Shl, ty, rhat, k32);
            let rhs = self.b.bin(BinOp::Add, ty, rb, un);
            let gt = self.b.cmp(CmpOp::GtU, ty, lhs, rhs);
            let mut c = self.b.bin(BinOp::Or, Ty::I32, qb, gt);
            if let Some(g) = go {
                // A second round only if the first fired and rhat stayed < b.
                let small = self.b.cmp(CmpOp::LtU, ty, rhat, b);
                c = self.b.bin(BinOp::And, Ty::I32, c, small);
                c = self.b.bin(BinOp::And, Ty::I32, c, g);
            }
            let c64 = self.b.extend(c, false);
            let qm = self.b.bin(BinOp::Sub, ty, q, c64);
            let add = self.b.bin(BinOp::Mul, ty, c64, vn1);
            let rp = self.b.bin(BinOp::Add, ty, rhat, add);
            q = qm;
            rhat = rp;
            go = Some(c);
        }
        let _ = one;
        (q, rhat)
    }

    /// Replace a divisor of zero (and, for signed division, the -1 that
    /// would overflow INT_MIN) with 1, so wasm cannot trap.
    fn safe_divisor(&mut self, d: Value, ty: Ty, signed_n: Option<Value>) -> Value {
        let zero = self.k(ty, 0);
        let one = self.k(ty, 1);
        let bad = self.b.cmp(CmpOp::Eq, ty, d, zero);
        // A signed division traps in wasm only for MIN / -1; any other
        // dividend over -1 is an ordinary negation and must keep its sign.
        let bad = if let Some(n) = signed_n {
            let m1 = self.k(ty, u64::MAX);
            let is_m1 = self.b.cmp(CmpOp::Eq, ty, d, m1);
            let min = self.k(ty, if ty == Ty::I64 { 1u64 << 63 } else { 1u64 << 31 });
            let is_min = self.b.cmp(CmpOp::Eq, ty, n, min);
            let ovf = self.b.bin(BinOp::And, Ty::I32, is_m1, is_min);
            self.b.bin(BinOp::Or, Ty::I32, bad, ovf)
        } else {
            bad
        };
        self.b.select(bad, one, d, ty)
    }

    /// Shifts and rotates by an immediate or `cl`.
    fn shift(&mut self, op: Op, d: Operand, amt: Operand, w: Width) {
        let ty = ty_of(w);
        let bits = w.bits();
        let a = self.rd(d, w);
        let a = self.narrow(a, w);
        let (count, count_static) = match amt {
            Operand::Imm(n) => {
                let n = (n as u32) & if w == Width::Q { 63 } else { 31 };
                (self.k(ty, n as u64), Some(n))
            }
            _ => {
                let c = self.rd_reg(Reg::Gp(RCX, Width::B));
                let m = self.k32(if w == Width::Q { 63 } else { 31 });
                let c = self.b.bin(BinOp::And, Ty::I32, c, m);
                let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                (c, None)
            }
        };
        if count_static == Some(0) {
            // Nothing moves and no flag changes, but the destination is still
            // written: a 32-bit destination zero-extends into its register.
            self.wr(d, w, a);
            return;
        }
        let one = self.k(ty, 1);
        let cm1 = self.b.bin(BinOp::Sub, ty, count, one);
        let (r, cf, of) = match op {
            Op::Shl => {
                let r = self.b.bin(BinOp::Shl, ty, a, count);
                let r = self.narrow(r, w);
                // CF: the last bit shifted out = bit (bits - count) of a.
                let t = self.b.bin(BinOp::Shl, ty, a, cm1);
                let cf = self.sign_bit(t, w);
                let msb = self.sign_bit(r, w);
                let of = self.b.bin(BinOp::Xor, Ty::I32, msb, cf);
                (r, cf, of)
            }
            Op::Shr => {
                let r = self.b.bin(BinOp::ShrU, ty, a, count);
                let t = self.b.bin(BinOp::ShrU, ty, a, cm1);
                let t = if ty == Ty::I64 { self.b.wrap(t) } else { t };
                let one32 = self.k32(1);
                let cf = self.b.bin(BinOp::And, Ty::I32, t, one32);
                let of = self.sign_bit(a, w);
                (r, cf, of)
            }
            Op::Sar => {
                let s = self.sext(a, w);
                let r = self.b.bin(BinOp::ShrS, ty, s, count);
                let r = self.narrow(r, w);
                let t = self.b.bin(BinOp::ShrS, ty, s, cm1);
                let t = if ty == Ty::I64 { self.b.wrap(t) } else { t };
                let one32 = self.k32(1);
                let cf = self.b.bin(BinOp::And, Ty::I32, t, one32);
                let of = self.k32(0);
                (r, cf, of)
            }
            Op::Rol | Op::Ror => {
                // Rotation amount is the count modulo the width.
                let modmask = self.k(ty, (bits - 1) as u64);
                let n = self.b.bin(BinOp::And, ty, count, modmask);
                let bitsv = self.k(ty, bits as u64);
                let nn = self.b.bin(BinOp::Sub, ty, bitsv, n);
                let (l, rr) = if op == Op::Rol { (n, nn) } else { (nn, n) };
                // Shift amounts of `bits` produce 0 through the mask below
                // for narrow widths; for full widths wasm masks the amount,
                // so rotate by 0 needs care: (a << 0) | (a >> bits) where
                // `>> bits` becomes `>> 0`. Use Rotr for full widths.
                let r = if bits == ty.bits() {
                    self.b.bin(BinOp::Rotr, ty, a, rr)
                } else {
                    let hi = self.b.bin(BinOp::Shl, ty, a, l);
                    let lo = self.b.bin(BinOp::ShrU, ty, a, rr);
                    let r = self.b.bin(BinOp::Or, ty, hi, lo);
                    self.narrow(r, w)
                };
                let one32 = self.k32(1);
                let (cf, of) = if op == Op::Rol {
                    let r32 = if ty == Ty::I64 { self.b.wrap(r) } else { r };
                    let cf = self.b.bin(BinOp::And, Ty::I32, r32, one32);
                    let msb = self.sign_bit(r, w);
                    (cf, self.b.bin(BinOp::Xor, Ty::I32, msb, cf))
                } else {
                    let cf = self.sign_bit(r, w);
                    let sh = self.k(ty, (bits - 2) as u64);
                    let m1 = self.b.bin(BinOp::ShrU, ty, r, sh);
                    let m1 = if ty == Ty::I64 { self.b.wrap(m1) } else { m1 };
                    let m1 = self.b.bin(BinOp::And, Ty::I32, m1, one32);
                    (cf, self.b.bin(BinOp::Xor, Ty::I32, cf, m1))
                };
                (r, cf, of)
            }
            _ => unreachable!(),
        };
        let rotate = matches!(op, Op::Rol | Op::Ror);
        if count_static.is_some() {
            self.wr(d, w, r);
            self.write_flag(GuestReg::C, cf);
            self.write_flag(GuestReg::V, of);
            if !rotate {
                let s = self.sign_bit(r, w);
                let z = self.is_zero(r, w);
                let p = self.parity(r, w);
                let zero = self.k32(0);
                self.write_flag(GuestReg::N, s);
                self.write_flag(GuestReg::Z, z);
                self.write_flag(GuestReg::P, p);
                self.write_flag(GuestReg::A, zero);
            }
        } else {
            // A zero count leaves everything unchanged.
            let zero = self.k(ty, 0);
            let is_zero = self.b.cmp(CmpOp::Eq, ty, count, zero);
            let v = self.b.select(is_zero, a, r, ty);
            self.wr(d, w, v);
            for (reg, nv) in [(GuestReg::C, cf), (GuestReg::V, of)] {
                let old = self.read_flag(reg);
                let f = self.b.select(is_zero, old, nv, Ty::I32);
                self.write_flag(reg, f);
            }
            if !rotate {
                let s = self.sign_bit(r, w);
                let z = self.is_zero(r, w);
                let p = self.parity(r, w);
                let zero32 = self.k32(0);
                for (reg, nv) in [(GuestReg::N, s), (GuestReg::Z, z), (GuestReg::P, p), (GuestReg::A, zero32)] {
                    let old = self.read_flag(reg);
                    let f = self.b.select(is_zero, old, nv, Ty::I32);
                    self.write_flag(reg, f);
                }
            }
        }
        self.flags = FlagSource::Unknown;
    }

    /// `rcl`/`rcr` by an immediate; the `cl` forms are rare enough to refuse.
    fn rotate_carry(&mut self, op: Op, d: Operand, amt: Operand, w: Width) -> Flow {
        let Operand::Imm(n) = amt else {
            return Flow::End(Term::Illegal {
                raw: self.raw,
                pc: self.pc,
            });
        };
        let bits = w.bits();
        let n = ((n as u32) & if w == Width::Q { 63 } else { 31 }) % (bits + 1);
        let ty = ty_of(w);
        let a = self.rd(d, w);
        let a = self.narrow(a, w);
        if n == 0 {
            self.wr(d, w, a);
            return Flow::Next;
        }
        let c = self.read_flag(GuestReg::C);
        let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
        // Work in a (bits+1)-bit register: value | carry << bits.
        let (r, cf) = if op == Op::Rcl {
            // r = (a << n) | (c << (n-1)) | (a >> (bits + 1 - n))
            let sh = self.k(ty, n as u64);
            let hi = self.b.bin(BinOp::Shl, ty, a, sh);
            let sh2 = self.k(ty, (n - 1) as u64);
            let cm = self.b.bin(BinOp::Shl, ty, c, sh2);
            let r = self.b.bin(BinOp::Or, ty, hi, cm);
            // The wrapped-around low part, unless its shift is the whole
            // type width (which wasm would mask to zero).
            let r = if bits + 1 - n < ty.bits() {
                let sh3 = self.k(ty, (bits + 1 - n) as u64);
                let lo = self.b.bin(BinOp::ShrU, ty, a, sh3);
                self.b.bin(BinOp::Or, ty, r, lo)
            } else {
                r
            };
            let r = self.narrow(r, w);
            // new CF = bit (bits - n) of a
            let sh4 = self.k(ty, (bits - n) as u64);
            let cf = self.b.bin(BinOp::ShrU, ty, a, sh4);
            let cf = if ty == Ty::I64 { self.b.wrap(cf) } else { cf };
            let one = self.k32(1);
            (r, self.b.bin(BinOp::And, Ty::I32, cf, one))
        } else {
            // r = (a >> n) | (c << (bits - n)) | (a << (bits + 1 - n))
            let sh = self.k(ty, n as u64);
            let lo = self.b.bin(BinOp::ShrU, ty, a, sh);
            let sh2 = self.k(ty, (bits - n) as u64);
            let cm = self.b.bin(BinOp::Shl, ty, c, sh2);
            let r = self.b.bin(BinOp::Or, ty, lo, cm);
            let r = if bits + 1 - n < ty.bits() {
                let sh3 = self.k(ty, (bits + 1 - n) as u64);
                let hi = self.b.bin(BinOp::Shl, ty, a, sh3);
                self.b.bin(BinOp::Or, ty, r, hi)
            } else {
                r
            };
            let r = self.narrow(r, w);
            let sh4 = self.k(ty, (n - 1) as u64);
            let cf = self.b.bin(BinOp::ShrU, ty, a, sh4);
            let cf = if ty == Ty::I64 { self.b.wrap(cf) } else { cf };
            let one = self.k32(1);
            (r, self.b.bin(BinOp::And, Ty::I32, cf, one))
        };
        self.wr(d, w, r);
        self.write_flag(GuestReg::C, cf);
        // OF (architecturally defined for a count of one): for `rcl` the
        // result's MSB xor the new CF; for `rcr` the xor of the result's two
        // most significant bits.
        let msb = self.sign_bit(r, w);
        let other = if op == Op::Rcl {
            cf
        } else {
            let sh = self.k(ty, (bits - 2) as u64);
            let b = self.b.bin(BinOp::ShrU, ty, r, sh);
            let b = if ty == Ty::I64 { self.b.wrap(b) } else { b };
            let one = self.k32(1);
            self.b.bin(BinOp::And, Ty::I32, b, one)
        };
        let of = self.b.bin(BinOp::Xor, Ty::I32, msb, other);
        self.write_flag(GuestReg::V, of);
        self.flags = FlagSource::Unknown;
        Flow::Next
    }

    fn double_shift(&mut self, op: Op, d: Operand, s: Operand, amt: Operand, w: Width) {
        let ty = ty_of(w);
        let bits = w.bits();
        let a = self.rd(d, w);
        let a = self.narrow(a, w);
        let b = self.rd(s, w);
        let b = self.narrow(b, w);
        let (count, count_static) = match amt {
            Operand::Imm(n) => {
                let n = (n as u32) & if w == Width::Q { 63 } else { 31 };
                (self.k(ty, n as u64), Some(n))
            }
            _ => {
                let c = self.rd_reg(Reg::Gp(RCX, Width::B));
                let m = self.k32(if w == Width::Q { 63 } else { 31 });
                let c = self.b.bin(BinOp::And, Ty::I32, c, m);
                let c = if ty == Ty::I64 { self.b.extend(c, false) } else { c };
                (c, None)
            }
        };
        if count_static == Some(0) {
            self.wr(d, w, a);
            return;
        }
        let bitsv = self.k(ty, bits as u64);
        let nn = self.b.bin(BinOp::Sub, ty, bitsv, count);
        let one = self.k(ty, 1);
        let cm1 = self.b.bin(BinOp::Sub, ty, count, one);
        let (r, cf) = if op == Op::Shld {
            let hi = self.b.bin(BinOp::Shl, ty, a, count);
            let lo = self.b.bin(BinOp::ShrU, ty, b, nn);
            let r = self.b.bin(BinOp::Or, ty, hi, lo);
            let r = self.narrow(r, w);
            let t = self.b.bin(BinOp::Shl, ty, a, cm1);
            (r, self.sign_bit(t, w))
        } else {
            let lo = self.b.bin(BinOp::ShrU, ty, a, count);
            let hi = self.b.bin(BinOp::Shl, ty, b, nn);
            let r = self.b.bin(BinOp::Or, ty, lo, hi);
            let r = self.narrow(r, w);
            let t = self.b.bin(BinOp::ShrU, ty, a, cm1);
            let t = if ty == Ty::I64 { self.b.wrap(t) } else { t };
            let one32 = self.k32(1);
            (r, self.b.bin(BinOp::And, Ty::I32, t, one32))
        };
        let sa = self.sign_bit(a, w);
        let sr = self.sign_bit(r, w);
        let of = self.b.bin(BinOp::Xor, Ty::I32, sa, sr);
        let s_ = self.sign_bit(r, w);
        let z = self.is_zero(r, w);
        let p = self.parity(r, w);
        let zero32 = self.k32(0);
        if count_static.is_some() {
            self.wr(d, w, r);
            for (reg, v) in [
                (GuestReg::C, cf),
                (GuestReg::V, of),
                (GuestReg::N, s_),
                (GuestReg::Z, z),
                (GuestReg::P, p),
                (GuestReg::A, zero32),
            ] {
                self.write_flag(reg, v);
            }
        } else {
            let zero = self.k(ty, 0);
            let is_zero = self.b.cmp(CmpOp::Eq, ty, count, zero);
            let v = self.b.select(is_zero, a, r, ty);
            self.wr(d, w, v);
            for (reg, nv) in [
                (GuestReg::C, cf),
                (GuestReg::V, of),
                (GuestReg::N, s_),
                (GuestReg::Z, z),
                (GuestReg::P, p),
                (GuestReg::A, zero32),
            ] {
                let old = self.read_flag(reg);
                let f = self.b.select(is_zero, old, nv, Ty::I32);
                self.write_flag(reg, f);
            }
        }
        self.flags = FlagSource::Unknown;
    }

    fn bit_test(&mut self, op: Op, d: Operand, s: Operand, w: Width, lock: bool) {
        let ty = ty_of(w);
        let bits = w.bits();
        // The bit index. For a register destination it is modulo the width;
        // for memory it addresses bits beyond the operand, so the address
        // is adjusted by the index's quotient.
        let idx = match s {
            Operand::Imm(n) => self.k(ty, (n as u64) & (bits as u64 - 1)),
            o => {
                let v = self.rd(o, w);
                let v = self.sext(v, w);
                v
            }
        };
        let (a, mem_addr) = match d {
            Operand::Mem(m) => {
                let (addr, offset) = self.addr(&m);
                let addr = self.offset_addr(addr, offset);
                // byte address += (idx >> 3) sign-extended, then bit = idx & 7
                // within the operand-sized word at the aligned address.
                let idx64 = if ty == Ty::I64 { idx } else { self.b.extend(idx, true) };
                let sh = self.k64(bits.trailing_zeros() as u64);
                let word = self.b.bin(BinOp::ShrS, Ty::I64, idx64, sh);
                let scale = self.k64(w.bytes().trailing_zeros() as u64);
                let byte_off = self.b.bin(BinOp::Shl, Ty::I64, word, scale);
                let addr = self.b.bin(BinOp::Add, Ty::I64, addr, byte_off);
                let a = self.b.def(Expr::Load {
                    addr,
                    offset: 0,
                    sz: mem_sz(w),
                    signed: false,
                    ty,
                    order: Order::Relaxed,
                });
                (a, Some(addr))
            }
            o => {
                let a = self.rd(o, w);
                (self.narrow(a, w), None)
            }
        };
        let modmask = self.k(ty, (bits - 1) as u64);
        let bit = self.b.bin(BinOp::And, ty, idx, modmask);
        let one = self.k(ty, 1);
        let mask = self.b.bin(BinOp::Shl, ty, one, bit);
        let (r, old) = match op {
            Op::Bt => (None, a),
            Op::Bts => (Some((BinOp::Or, RmwOp::Or)), a),
            Op::Btr => (Some((BinOp::And, RmwOp::And)), a),
            _ => (Some((BinOp::Xor, RmwOp::Xor)), a),
        };
        let old = match (r, mem_addr, lock) {
            (Some((_, rop)), Some(addr), true) => {
                let val = if op == Op::Btr {
                    self.b.un(UnOp::Not, ty, mask)
                } else {
                    mask
                };
                self.b.def(Expr::Rmw {
                    op: rop,
                    addr,
                    val,
                    sz: mem_sz(w),
                    ty,
                })
            }
            (Some((bop, _)), _, _) => {
                let val = if op == Op::Btr {
                    self.b.un(UnOp::Not, ty, mask)
                } else {
                    mask
                };
                let nv = self.b.bin(bop, ty, old, val);
                let nv = self.narrow(nv, w);
                match mem_addr {
                    Some(addr) => self.b.stmt(Stmt::Store {
                        addr,
                        offset: 0,
                        val: nv,
                        sz: mem_sz(w),
                        order: Order::Relaxed,
                    }),
                    None => self.wr(d, w, nv),
                }
                old
            }
            (None, _, _) => old,
        };
        let t = self.b.bin(BinOp::ShrU, ty, old, bit);
        let t = if ty == Ty::I64 { self.b.wrap(t) } else { t };
        let one32 = self.k32(1);
        let cf = self.b.bin(BinOp::And, Ty::I32, t, one32);
        self.write_flag(GuestReg::C, cf);
        self.flags = FlagSource::Unknown;
    }

    /// The low byte of RFLAGS: CF, 1, PF, 0, AF, 0, ZF, SF.
    fn flags_byte(&mut self) -> Value {
        let c = self.read_flag(GuestReg::C);
        let p = self.read_flag(GuestReg::P);
        let a = self.read_flag(GuestReg::A);
        let z = self.read_flag(GuestReg::Z);
        let s = self.read_flag(GuestReg::N);
        let two = self.k32(2);
        let mut v = self.b.bin(BinOp::Or, Ty::I32, c, two);
        for (f, sh) in [(p, 2), (a, 4), (z, 6), (s, 7)] {
            let k = self.k32(sh);
            let t = self.b.bin(BinOp::Shl, Ty::I32, f, k);
            v = self.b.bin(BinOp::Or, Ty::I32, v, t);
        }
        v
    }

    /// The selector of the code being lifted, as a far call pushes it:
    /// 0x23 in 32-bit code, 0x33 in 64-bit code.
    /// [`Self::sel_of`] at width `w`.
    fn sel_val(&mut self, seg: Seg, w: Width) -> Value {
        let sel = self.sel_of(seg) as u64;
        if w == Width::Q {
            self.k64(sel)
        } else {
            self.k(ty_of(w), sel)
        }
    }

    /// What a segment register reads in this block, as under Linux: %cs
    /// names the block's mode (`ctx::CS` is always it), %ss the user data
    /// segment, as do %ds and %es in 32-bit code (64-bit code leaves them
    /// null); %fs and %gs read null, their bases being set apart from them.
    /// Loads are not kept, so each reads the same everywhere in a mode.
    fn sel_of(&self, seg: Seg) -> u32 {
        const USER_DS: u32 = 0x2b;
        let compat = self.wa() == Width::D;
        match seg {
            Seg::Cs if compat => gex_abi::ctx::CS_COMPAT,
            Seg::Cs => gex_abi::ctx::CS_LONG,
            Seg::Ss => USER_DS,
            Seg::Ds | Seg::Es if compat => USER_DS,
            _ => 0,
        }
    }

    /// A far transfer to `sel:off` (`sel` an `I32`, `off` an `I64`): the
    /// host checks the selector and sets the thread's mode
    /// ([`sysreg::FAR_IN_PLACE`]; any selector but a code segment's is #GP,
    /// an access violation), then the block dispatches what the host names
    /// ([`sysreg::FAR_NEXT`]): the offset, or for a change of mode an
    /// address this mode's directory never holds, whose miss the host
    /// answers with the target's translation in the new mode. The thread
    /// stays in wasm across the change.
    fn far(&mut self, pc: u64, sel: Value, off: Value) -> Flow {
        let sel = self.b.extend(sel, false);
        let sh = self.k64(32);
        let sel = self.b.bin(BinOp::Shl, Ty::I64, sel, sh);
        let v = self.b.bin(BinOp::Or, Ty::I64, off, sel);
        let pcv = self.k64(pc);
        let id = self.k32(sysreg::FAR_IN_PLACE as u32);
        self.b.stmt(Stmt::HostCall {
            func: HostFunc::SysRegWrite,
            args: vec![pcv, id, v],
        });
        let next = self.k32(sysreg::FAR_NEXT as u32);
        let target = self.b.def(Expr::HostCall {
            func: HostFunc::SysRegRead,
            args: vec![pcv, next],
            ty: Ty::I64,
        });
        Flow::End(Term::Indirect { target, is_ret: false })
    }

    /// The user-writable flags from a popped `rflags` (an `I32`): what
    /// `popf` and `iret` restore.
    fn set_rflags(&mut self, v: Value) {
        self.unpack_flags_low(v);
        let ten = self.k32(10);
        let d = self.b.bin(BinOp::ShrU, Ty::I32, v, ten);
        let one = self.k32(1);
        let d = self.b.bin(BinOp::And, Ty::I32, d, one);
        self.write_flag(GuestReg::D, d);
        let eleven = self.k32(11);
        let o = self.b.bin(BinOp::ShrU, Ty::I32, v, eleven);
        let o = self.b.bin(BinOp::And, Ty::I32, o, one);
        self.write_flag(GuestReg::V, o);
        // AC (18) and ID (21) are kept for the next pushf; the other
        // system flags (IF, IOPL, ...) cannot be changed in user mode.
        let mask = self.k32(0x24_0000);
        let sys = self.b.bin(BinOp::And, Ty::I32, v, mask);
        let base = self.b.read(GuestReg::CtxAddr);
        self.b.stmt(Stmt::Store {
            addr: base,
            offset: gex_abi::ctx::EFLAGS_SYS as i64,
            val: sys,
            sz: MemSz::B4,
            order: Order::Relaxed,
        });
    }

    /// Full RFLAGS as an `I32`: the low byte plus IF (bit 9, always set for
    /// user code), DF and OF.
    fn rflags(&mut self) -> Value {
        let lo = self.flags_byte();
        let d = self.read_flag(GuestReg::D);
        let o = self.read_flag(GuestReg::V);
        let ten = self.k32(10);
        let d = self.b.bin(BinOp::Shl, Ty::I32, d, ten);
        let eleven = self.k32(11);
        let o = self.b.bin(BinOp::Shl, Ty::I32, o, eleven);
        let v = self.b.bin(BinOp::Or, Ty::I32, lo, d);
        let v = self.b.bin(BinOp::Or, Ty::I32, v, o);
        let ifl = self.k32(0x200);
        let v = self.b.bin(BinOp::Or, Ty::I32, v, ifl);
        // AC and ID, as the last popf left them (ctx::EFLAGS_SYS).
        let base = self.b.read(GuestReg::CtxAddr);
        let sys = self.b.def(Expr::Load {
            addr: base,
            offset: gex_abi::ctx::EFLAGS_SYS as i64,
            sz: MemSz::B4,
            signed: false,
            ty: Ty::I32,
            order: Order::Relaxed,
        });
        self.b.bin(BinOp::Or, Ty::I32, v, sys)
    }

    /// Set CF, PF, AF, ZF, SF from the low byte of `v`.
    fn unpack_flags_low(&mut self, v: Value) {
        let one = self.k32(1);
        for (reg, sh) in [
            (GuestReg::C, 0u32),
            (GuestReg::P, 2),
            (GuestReg::A, 4),
            (GuestReg::Z, 6),
            (GuestReg::N, 7),
        ] {
            let k = self.k32(sh);
            let t = self.b.bin(BinOp::ShrU, Ty::I32, v, k);
            let t = self.b.bin(BinOp::And, Ty::I32, t, one);
            self.write_flag(reg, t);
        }
        self.flags = FlagSource::Unknown;
    }

    // -- string instructions ---------------------------------------------------------------

    /// The first instruction of a [`copy_loop`] (`movsb`, comparing `edi`
    /// with register `r`), as all its iterations: one `StrOp` copying
    /// (r - edi) bytes forward, or (edi - r) backward, mod 2^32 (0 is 2^32,
    /// as the loop runs until the 32-bit compare wraps round). The loop's
    /// `cmp` and `jne` are lifted after it as usual: equal, so the branch
    /// falls through, with the flags the last compare leaves.
    pub fn lift_copy_loop(&mut self, r: u8) -> Flow {
        let df = self.read_flag(GuestReg::D);
        let edi = self.rd_reg(Reg::Gp(RDI, Width::D));
        let end = self.rd_reg(Reg::Gp(r, Width::D));
        let fwd = self.b.bin(BinOp::Sub, Ty::I32, end, edi);
        let back = self.b.bin(BinOp::Sub, Ty::I32, edi, end);
        let zero32 = self.k32(0);
        let df_clear = self.b.cmp(CmpOp::Eq, Ty::I32, df, zero32);
        let n32 = self.b.select(df_clear, fwd, back, Ty::I32);
        let n = self.b.extend(n32, false);
        let wrap = self.b.cmp(CmpOp::Eq, Ty::I32, n32, zero32);
        let full = self.k64(1 << 32);
        let count = self.b.select(wrap, full, n, Ty::I64);
        let dst = self.rd64(RDI);
        let src = self.rd64(RSI);
        let val = self.rd64(RAX);
        let (out_count, out_dst, out_src, out_a, out_b) =
            (self.b.fresh(), self.b.fresh(), self.b.fresh(), self.b.fresh(), self.b.fresh());
        self.b.stmt(Stmt::StrOp {
            kind: StrKind::Movs,
            sz: MemSz::B1,
            until: StrUntil::Count,
            count,
            dst,
            src,
            val,
            df,
            out_count,
            out_dst,
            out_src,
            out_a,
            out_b,
        });
        self.wr64(RDI, out_dst);
        self.wr64(RSI, out_src);
        Flow::Next
    }

    fn string_op(&mut self, i: &Inst) -> Flow {
        let w = i.width;
        let ty = ty_of(w);
        let bytes = w.bytes() as u64;
        let df = self.read_flag(GuestReg::D);
        // step = df ? -bytes : bytes
        let neg = self.k64(bytes.wrapping_neg());
        let pos = self.k64(bytes);
        let zero32 = self.k32(0);
        let df_clear = self.b.cmp(CmpOp::Eq, Ty::I32, df, zero32);
        let step = self.b.select(df_clear, pos, neg, Ty::I64);

        if i.rep != Rep::None && i.op != Op::Lods {
            let kind = match i.op {
                Op::Movs => StrKind::Movs,
                Op::Stos => StrKind::Stos,
                Op::Scas => StrKind::Scas,
                _ => StrKind::Cmps,
            };
            let until = match (i.op, i.rep) {
                (Op::Scas | Op::Cmps, Rep::Rep) => StrUntil::NotEqual,
                (Op::Scas | Op::Cmps, Rep::RepNe) => StrUntil::Equal,
                _ => StrUntil::Count,
            };
            let count = self.rd64(RCX);
            let dst = self.rd64(RDI);
            let src = self.rd64(RSI);
            let val = self.rd64(RAX);
            let out_count = self.b.fresh();
            let out_dst = self.b.fresh();
            let out_src = self.b.fresh();
            let out_a = self.b.fresh();
            let out_b = self.b.fresh();
            self.b.stmt(Stmt::StrOp {
                kind,
                sz: mem_sz(w),
                until,
                count,
                dst,
                src,
                val,
                df,
                out_count,
                out_dst,
                out_src,
                out_a,
                out_b,
            });
            self.wr64(RCX, out_count);
            self.wr64(RDI, out_dst);
            if matches!(kind, StrKind::Movs | StrKind::Cmps) {
                self.wr64(RSI, out_src);
            }
            if matches!(kind, StrKind::Scas | StrKind::Cmps) {
                // Flags of the last compare, unless nothing ran.
                let (a, b) = if ty == Ty::I64 {
                    (out_a, out_b)
                } else {
                    (self.b.wrap(out_a), self.b.wrap(out_b))
                };
                let r = self.b.bin(BinOp::Sub, ty, a, b);
                let r = self.narrow(r, w);
                let olds: Vec<(GuestReg, Value)> = [GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V, GuestReg::P, GuestReg::A]
                    .into_iter()
                    .map(|f| (f, self.read_flag(f)))
                    .collect();
                self.flags_sub(a, b, r, w, None, false);
                let zero = self.k64(0);
                let none = self.b.cmp(CmpOp::Eq, Ty::I64, count, zero);
                for (f, old) in olds {
                    let nv = self.read_flag(f);
                    let v = self.b.select(none, old, nv, Ty::I32);
                    self.write_flag(f, v);
                }
                self.flags = FlagSource::Unknown;
            }
            return Flow::Next;
        }

        // A single element.
        let rdi = self.rd64(RDI);
        let rsi = self.rd64(RSI);
        let load = |l: &mut Self, addr: Value| -> Value {
            l.b.def(Expr::Load {
                addr,
                offset: 0,
                sz: mem_sz(w),
                signed: false,
                ty,
                order: Order::Relaxed,
            })
        };
        match i.op {
            Op::Movs => {
                let v = load(self, rsi);
                self.b.stmt(Stmt::Store {
                    addr: rdi,
                    offset: 0,
                    val: v,
                    sz: mem_sz(w),
                    order: Order::Relaxed,
                });
                let ndi = self.b.bin(BinOp::Add, Ty::I64, rdi, step);
                let nsi = self.b.bin(BinOp::Add, Ty::I64, rsi, step);
                self.wr64(RDI, ndi);
                self.wr64(RSI, nsi);
            }
            Op::Stos => {
                let v = self.rd_reg(Reg::Gp(RAX, w));
                self.b.stmt(Stmt::Store {
                    addr: rdi,
                    offset: 0,
                    val: v,
                    sz: mem_sz(w),
                    order: Order::Relaxed,
                });
                let ndi = self.b.bin(BinOp::Add, Ty::I64, rdi, step);
                self.wr64(RDI, ndi);
            }
            Op::Lods => {
                let v = load(self, rsi);
                self.wr_reg(Reg::Gp(RAX, w), v);
                let nsi = self.b.bin(BinOp::Add, Ty::I64, rsi, step);
                self.wr64(RSI, nsi);
            }
            Op::Scas => {
                let a = self.rd_reg(Reg::Gp(RAX, w));
                let b = load(self, rdi);
                let r = self.b.bin(BinOp::Sub, ty, a, b);
                let r = self.narrow(r, w);
                self.flags_sub(a, b, r, w, None, false);
                let ndi = self.b.bin(BinOp::Add, Ty::I64, rdi, step);
                self.wr64(RDI, ndi);
            }
            _ => {
                let a = load(self, rsi);
                let b = load(self, rdi);
                let r = self.b.bin(BinOp::Sub, ty, a, b);
                let r = self.narrow(r, w);
                self.flags_sub(a, b, r, w, None, false);
                let ndi = self.b.bin(BinOp::Add, Ty::I64, rdi, step);
                let nsi = self.b.bin(BinOp::Add, Ty::I64, rsi, step);
                self.wr64(RDI, ndi);
                self.wr64(RSI, nsi);
            }
        }
        Flow::Next
    }

    /// `fxsave`/`fxrstor` and the `xsave` family: save/restore the x87 and
    /// SSE state (control and status words, tags, the eight x87 registers,
    /// MXCSR and the sixteen xmm registers) in the architectural layout.
    fn fxsave(&mut self, i: &Inst) -> Flow {
        let Operand::Mem(m) = i.op0() else {
            return Flow::End(Term::Illegal {
                raw: self.raw,
                pc: self.pc,
            });
        };
        let (addr, offset) = self.addr(&m);
        let addr = self.offset_addr(addr, offset);
        let save = matches!(i.op, Op::Fxsave | Op::Xsave);
        if save {
            // FCW, FSW, the abridged FTW and the eight x87 registers, then
            // MXCSR at 24 and MXCSR_MASK at 28.
            let fcw = self.b.read(GuestReg::X87Cw);
            self.b.stmt(Stmt::Store {
                addr,
                offset: 0,
                val: fcw,
                sz: MemSz::B2,
                order: Order::Relaxed,
            });
            self.x87_fxsave(addr, true);
            let mxcsr = self.b.read(GuestReg::Mxcsr);
            self.b.stmt(Stmt::Store {
                addr,
                offset: 24,
                val: mxcsr,
                sz: MemSz::B4,
                order: Order::Relaxed,
            });
            let mask = self.k32(0xffff);
            self.b.stmt(Stmt::Store {
                addr,
                offset: 28,
                val: mask,
                sz: MemSz::B4,
                order: Order::Relaxed,
            });
            for n in 0..16u8 {
                let v = self.b.read(GuestReg::V128(n));
                self.b.stmt(Stmt::VStore {
                    addr,
                    offset: 160 + 16 * n as i64,
                    val: v,
                    bytes: 16,
                });
            }
            if matches!(i.op, Op::Xsave) {
                // XSTATE_BV at 512: x87 | SSE present.
                let bv = self.k64(3);
                self.b.stmt(Stmt::Store {
                    addr,
                    offset: 512,
                    val: bv,
                    sz: MemSz::B8,
                    order: Order::Relaxed,
                });
            }
        } else {
            let fcw = self.b.def(Expr::Load {
                addr,
                offset: 0,
                sz: MemSz::B2,
                signed: false,
                ty: Ty::I32,
                order: Order::Relaxed,
            });
            self.x87_set_cw(fcw);
            self.x87_fxsave(addr, false);
            let mxcsr = self.b.def(Expr::Load {
                addr,
                offset: 24,
                sz: MemSz::B4,
                signed: false,
                ty: Ty::I32,
                order: Order::Relaxed,
            });
            self.b.write(GuestReg::Mxcsr, mxcsr);
            for n in 0..16u8 {
                let v = self.b.def(Expr::VLoad {
                    addr,
                    offset: 160 + 16 * n as i64,
                    bytes: 16,
                });
                self.b.write(GuestReg::V128(n), v);
            }
        }
        Flow::Next
    }
}
