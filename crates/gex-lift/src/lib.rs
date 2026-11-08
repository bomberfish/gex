//! `gex-lift` — AArch64 instruction semantics, expressed in GIR.
//!
//! One function per instruction family, hand-written against the Arm ARM
//! pseudocode. The lifter's contract:
//!
//! * **Total.** Every [`Inst`] either lifts to GIR or ends the block with
//!   [`Term::Illegal`]. Nothing is silently skipped, so an instruction gex
//!   cannot execute becomes a precise SIGILL at run time rather than a wrong
//!   answer.
//! * **Exact.** Where the architecture and wasm disagree — division by zero,
//!   `INT_MIN / -1`, shift-amount masking — the lifter inserts the guard, so
//!   the backend never has to know.
//! * **Flags on demand.** NZCV is written as four separate one-bit registers;
//!   dead-flag elimination in [`gex_ir::pass`] removes the ones nobody reads.

#![forbid(unsafe_code)]

use gex_ir::*;
use gex_isa::{
    AddOp, AddrMode, BarrierKind, BfOp, Cond, Dp1Op, Dp2Op, Dp3Op, Extend, FpOp1, FpOp2, FpRound,
    FpSz, Inst, LogOp, MemSz, MoveOp, Ordering as IsaOrder, RegSel, RmwOp as IsaRmw, ShiftType,
    SysMaintOp, SysReg, Sz, VAcrossOp, VArr, VLogOp, VOp2, VOp3, VPermOp, VSelKind, VShiftImmOp,
    VNarrowOp,
    VWidenOp,
};

mod block;
pub use block::{lift_block, BlockLimits};

/// Whether to emit a fused multiply-add as an unfused multiply and add.
///
/// A measurement aid, never a shipping configuration: fusing rounds once and
/// the pair rounds twice, so the answers differ.
fn inline_fma() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEX_INLINE_FMA").is_some())
}

/// Guest instruction width. A64 is fixed-width; the whole design leans on it.
pub const INST_BYTES: u64 = 4;

/// What the condition flags hold, when it is something a branch can ask for
/// directly instead of reading N/Z/C/V. Lets `cmp`+`b.cc` compare the operands
/// directly; the flag writes still happen for successors, but flag liveness
/// then deletes the ones nobody reads.
///
/// Invariant: every write to N, Z, C or V must update this, and any it cannot
/// describe sets [`FlagSource::Unknown`]. A stale source compiles a branch on
/// the wrong question.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FlagSource {
    #[default]
    Unknown,
    /// NZCV are those of `a - b` at width `ty`, which is what `cmp`, `subs`
    /// and `cmn`-with-negation leave behind.
    Sub { a: Value, b: Value, ty: Ty },
    /// N and Z are those of `v`, and C and V are zero: the logical-op rule.
    Logic { v: Value, ty: Ty },
}

pub struct Lifter<'a> {
    b: &'a mut Builder,
    /// What the flags mean right now. Carried in and out across the
    /// instructions of one block.
    flags: FlagSource,
    /// The instruction word being lifted, so a SIGILL for an inexpressible
    /// instruction can name the actual word rather than `0x00000000`.
    raw: u32,
}

/// The comparison that is true exactly when this one is false.
fn invert_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Ne,
        CmpOp::Ne => CmpOp::Eq,
        CmpOp::LtU => CmpOp::GeU,
        CmpOp::GeU => CmpOp::LtU,
        CmpOp::LeU => CmpOp::GtU,
        CmpOp::GtU => CmpOp::LeU,
        CmpOp::LtS => CmpOp::GeS,
        CmpOp::GeS => CmpOp::LtS,
        CmpOp::LeS => CmpOp::GtS,
        CmpOp::GtS => CmpOp::LeS,
    }
}

/// What the lifter decided about control flow after one instruction.
pub enum Flow {
    /// Continue with the next instruction.
    Next,
    /// End the block with this terminator.
    End(Term),
}

fn ty_of(sz: Sz) -> Ty {
    match sz {
        Sz::W => Ty::I32,
        Sz::X => Ty::I64,
    }
}

fn mem_sz(sz: MemSz) -> gex_ir::MemSz {
    match sz {
        MemSz::B => gex_ir::MemSz::B1,
        MemSz::H => gex_ir::MemSz::B2,
        MemSz::W => gex_ir::MemSz::B4,
        // A 16-byte access only appears in SIMD load/store, which the decoder
        // still rejects; mapping it keeps this function total.
        MemSz::D | MemSz::Q => gex_ir::MemSz::B8,
    }
}

fn order_of(o: IsaOrder) -> Order {
    match o {
        IsaOrder::None => Order::Relaxed,
        // Acquire and release both map to seq-cst, which is stronger than
        // either and therefore always sound.
        IsaOrder::Acquire | IsaOrder::Release => Order::SeqCst,
    }
}

/// The `i`th register of a compare-swap pair whose base is `r`: `r + i`.
/// The encoding keeps both base registers even, so only `30 + 1` reaches
/// 31 -- the zero register, which [`Lifter::rd_gp`] reads as 0 and
/// [`Lifter::wr_gp`] drops.
fn pair_next(r: RegSel, i: u8) -> RegSel {
    match r {
        RegSel::Gp(n) => RegSel::gp(n as u32 + i as u32),
        // A pair base is always a plain register; the decoder says so.
        other => other,
    }
}

/// Carry input to [`Lifter::add_with_carry`].
enum Carry {
    Zero,
    One,
    Flag(Value),
}

/// Result of an add-with-carry: the sum plus the flags it would set.
struct AddResult {
    value: Value,
    n: Value,
    z: Value,
    c: Value,
    v: Value,
}

/// A resolved addressing mode.
struct Address {
    base: Value,
    offset: i64,
    /// Post-index writeback applied after the access. A `Value`, not a
    /// constant, because the register-offset form's amount is only known at
    /// run time.
    post: Option<(RegSel, Value)>,
}

impl<'a> Lifter<'a> {
    pub fn new(b: &'a mut Builder) -> Lifter<'a> {
        Lifter {
            b,
            raw: 0,
            flags: FlagSource::Unknown,
        }
    }

    /// What the previous instruction in this block left the flags meaning.
    pub fn with_flags(mut self, flags: FlagSource) -> Lifter<'a> {
        self.flags = flags;
        self
    }

    /// What this instruction leaves them meaning, for the next one.
    pub fn flags(&self) -> FlagSource {
        self.flags
    }

    /// The word `lift` is about to be given, so a SIGILL can name it.
    pub fn with_raw(mut self, raw: u32) -> Lifter<'a> {
        self.raw = raw;
        self
    }

    // -- register access ---------------------------------------------------

    /// Read a general-purpose register at width `sz`.
    fn rd_gp(&mut self, r: RegSel, sz: Sz) -> Value {
        let full = match r {
            RegSel::Zr => return self.b.const_ty(ty_of(sz), 0),
            RegSel::Sp => self.b.read(GuestReg::Sp),
            RegSel::Gp(n) => self.b.read(GuestReg::X(n)),
        };
        match sz {
            Sz::X => full,
            Sz::W => self.b.wrap(full),
        }
    }

    /// Write a general-purpose register at width `sz`. A 32-bit write
    /// zero-extends into the 64-bit register, as the architecture requires.
    fn wr_gp(&mut self, r: RegSel, sz: Sz, val: Value) {
        let full = match sz {
            Sz::X => val,
            Sz::W => self.b.extend(val, false),
        };
        match r {
            RegSel::Zr => {}
            RegSel::Sp => self.b.write(GuestReg::Sp, full),
            RegSel::Gp(n) => self.b.write(GuestReg::X(n), full),
        }
    }

    /// Read a 64-bit address-typed register (never narrowed).
    fn rd_addr(&mut self, r: RegSel) -> Value {
        match r {
            RegSel::Zr => self.b.const_i64(0),
            RegSel::Sp => self.b.read(GuestReg::Sp),
            RegSel::Gp(n) => self.b.read(GuestReg::X(n)),
        }
    }

    fn wr_addr(&mut self, r: RegSel, val: Value) {
        match r {
            RegSel::Zr => {}
            RegSel::Sp => self.b.write(GuestReg::Sp, val),
            RegSel::Gp(n) => self.b.write(GuestReg::X(n), val),
        }
    }

    fn konst(&mut self, ty: Ty, v: u64) -> Value {
        self.b.const_ty(ty, v)
    }

    // -- shared helpers ----------------------------------------------------

    /// Apply a shift-by-constant to a register value.
    fn shift(&mut self, v: Value, ty: Ty, kind: ShiftType, amount: u32) -> Value {
        if amount == 0 {
            return v;
        }
        let amt = self.konst(ty, amount as u64);
        let op = match kind {
            ShiftType::Lsl => BinOp::Shl,
            ShiftType::Lsr => BinOp::ShrU,
            ShiftType::Asr => BinOp::ShrS,
            ShiftType::Ror => BinOp::Rotr,
        };
        self.b.bin(op, ty, v, amt)
    }

    /// `ExtendReg`: take the low bits of `rm`, extend them, then shift left.
    /// The result is `ty`-typed, matching the enclosing operation's width.
    fn extend_reg(&mut self, rm: RegSel, ext: Extend, amount: u32, ty: Ty) -> Value {
        let raw = match ty {
            Ty::I32 => self.rd_gp(rm, Sz::W),
            _ => self.rd_addr(rm),
        };
        let bits = ext.src_bits().min(ty.bits());
        let v = if bits >= ty.bits() {
            raw
        } else {
            self.b.def(Expr::ExtendBits {
                a: raw,
                ty,
                bits,
                signed: ext.is_signed(),
            })
        };
        if amount == 0 {
            v
        } else {
            let amt = self.konst(ty, amount as u64);
            self.b.bin(BinOp::Shl, ty, v, amt)
        }
    }

    /// `AddWithCarry` from the Arm ARM: the sum and the four flags it sets.
    fn add_with_carry(&mut self, ty: Ty, a: Value, b: Value, carry: Carry) -> AddResult {
        let s1 = self.b.bin(BinOp::Add, ty, a, b);
        let c1 = self.b.cmp(CmpOp::LtU, ty, s1, a);
        let (sum, carry_out) = match carry {
            Carry::Zero => (s1, c1),
            Carry::One => {
                let one = self.konst(ty, 1);
                let s = self.b.bin(BinOp::Add, ty, s1, one);
                let c2 = self.b.cmp(CmpOp::LtU, ty, s, s1);
                let c = self.b.bin(BinOp::Or, Ty::I32, c1, c2);
                (s, c)
            }
            Carry::Flag(cin) => {
                // The carry flag is an i32 0/1; widen it to the operand width.
                let cin_ty = match ty {
                    Ty::I32 => cin,
                    _ => self.b.extend(cin, false),
                };
                let s = self.b.bin(BinOp::Add, ty, s1, cin_ty);
                let c2 = self.b.cmp(CmpOp::LtU, ty, s, s1);
                let c = self.b.bin(BinOp::Or, Ty::I32, c1, c2);
                (s, c)
            }
        };
        let zero = self.konst(ty, 0);
        let n = self.b.cmp(CmpOp::LtS, ty, sum, zero);
        let z = self.b.cmp(CmpOp::Eq, ty, sum, zero);
        // Signed overflow: both operands agree in sign and the result differs.
        let xa = self.b.bin(BinOp::Xor, ty, a, sum);
        let xb = self.b.bin(BinOp::Xor, ty, b, sum);
        let both = self.b.bin(BinOp::And, ty, xa, xb);
        let v = self.b.cmp(CmpOp::LtS, ty, both, zero);
        AddResult {
            value: sum,
            n,
            z,
            c: carry_out,
            v,
        }
    }

    fn set_nzcv(&mut self, r: &AddResult) {
        // The default. `add_sub` knows the operands and says so afterwards.
        self.flags = FlagSource::Unknown;
        self.b.write(GuestReg::N, r.n);
        self.b.write(GuestReg::Z, r.z);
        self.b.write(GuestReg::C, r.c);
        self.b.write(GuestReg::V, r.v);
    }

    /// Set N and Z from a result, clearing C and V — the logical-op rule.
    fn set_nz_clear_cv(&mut self, ty: Ty, value: Value) {
        let zero = self.konst(ty, 0);
        let n = self.b.cmp(CmpOp::LtS, ty, value, zero);
        let z = self.b.cmp(CmpOp::Eq, ty, value, zero);
        let zero32 = self.b.const_i32(0);
        self.b.write(GuestReg::N, n);
        self.b.write(GuestReg::Z, z);
        self.b.write(GuestReg::C, zero32);
        self.b.write(GuestReg::V, zero32);
        self.flags = FlagSource::Logic { v: value, ty };
    }

    /// Evaluate a condition code against the current flags, as an `I32` bool.
    fn eval_cond(&mut self, cond: Cond) -> Value {
        if cond.is_always() {
            return self.b.const_i32(1);
        }
        if let Some(v) = self.cond_direct(cond) {
            return v;
        }
        let base = cond.0 >> 1;
        let invert = cond.0 & 1 == 1;
        let one = self.b.const_i32(1);
        let v = match base {
            0b000 => self.b.read(GuestReg::Z),
            0b001 => self.b.read(GuestReg::C),
            0b010 => self.b.read(GuestReg::N),
            0b011 => self.b.read(GuestReg::V),
            0b100 => {
                // HI: C == 1 && Z == 0
                let c = self.b.read(GuestReg::C);
                let z = self.b.read(GuestReg::Z);
                let nz = self.b.bin(BinOp::Xor, Ty::I32, z, one);
                self.b.bin(BinOp::And, Ty::I32, c, nz)
            }
            0b101 => {
                // GE: N == V
                let n = self.b.read(GuestReg::N);
                let vf = self.b.read(GuestReg::V);
                self.b.cmp(CmpOp::Eq, Ty::I32, n, vf)
            }
            _ => {
                // GT: Z == 0 && N == V
                let n = self.b.read(GuestReg::N);
                let vf = self.b.read(GuestReg::V);
                let z = self.b.read(GuestReg::Z);
                let nv = self.b.cmp(CmpOp::Eq, Ty::I32, n, vf);
                let nz = self.b.bin(BinOp::Xor, Ty::I32, z, one);
                self.b.bin(BinOp::And, Ty::I32, nv, nz)
            }
        };
        if invert {
            self.b.bin(BinOp::Xor, Ty::I32, v, one)
        } else {
            v
        }
    }

    /// The condition as a question about the values the flags came from.
    ///
    /// `None` when the caller must fall back to reading N/Z/C/V.
    ///
    /// Gotcha: after `subs` AArch64's carry is *not borrow*, so C set means
    /// `a >= b` unsigned and HI (C set, Z clear) is `a > b`.
    fn cond_direct(&mut self, cond: Cond) -> Option<Value> {
        match self.flags {
            FlagSource::Unknown => None,
            FlagSource::Sub { a, b, ty } => {
                let op = match cond.0 >> 1 {
                    0b000 => CmpOp::Eq,  // EQ
                    0b001 => CmpOp::GeU, // CS/HS
                    0b100 => CmpOp::GtU, // HI
                    0b101 => CmpOp::GeS, // GE
                    0b110 => CmpOp::GtS, // GT
                    // MI, PL, VS and VC are about the difference itself
                    // rather than an ordering, and have no such form.
                    _ => return None,
                };
                let op = if cond.0 & 1 == 1 { invert_cmp(op) } else { op };
                Some(self.b.cmp(op, ty, a, b))
            }
            FlagSource::Logic { v, ty } => {
                // C and V are zero, so four of the eight are constants.
                let op = match cond.0 >> 1 {
                    0b000 => CmpOp::Eq,  // EQ: v == 0
                    0b010 => CmpOp::LtS, // MI: v < 0
                    0b101 => CmpOp::GeS, // GE: N == V, and V is 0
                    0b110 => CmpOp::GtS, // GT: v > 0
                    0b001 | 0b011 | 0b100 => {
                        // CS is false, VS is false, HI is false -- and each
                        // inverse is true.
                        return Some(self.b.const_i32(u32::from(cond.0 & 1 == 1)));
                    }
                    _ => return None,
                };
                let op = if cond.0 & 1 == 1 { invert_cmp(op) } else { op };
                let zero = self.konst(ty, 0);
                Some(self.b.cmp(op, ty, v, zero))
            }
        }
    }

    /// Byte-reverse `v` within each `group_bytes`-sized group.
    fn byte_reverse(&mut self, v: Value, ty: Ty, group_bytes: u32) -> Value {
        let bits = ty.bits();
        let n_groups = bits / (group_bytes * 8);
        let mut acc: Option<Value> = None;
        for g in 0..n_groups {
            let base = g * group_bytes * 8;
            for byte in 0..group_bytes {
                let from = base + byte * 8;
                let to = base + (group_bytes - 1 - byte) * 8;
                let mask = self.konst(ty, 0xffu64 << from);
                let masked = self.b.bin(BinOp::And, ty, v, mask);
                let shifted = match to.cmp(&from) {
                    std::cmp::Ordering::Greater => {
                        let amt = self.konst(ty, (to - from) as u64);
                        self.b.bin(BinOp::Shl, ty, masked, amt)
                    }
                    std::cmp::Ordering::Less => {
                        let amt = self.konst(ty, (from - to) as u64);
                        self.b.bin(BinOp::ShrU, ty, masked, amt)
                    }
                    std::cmp::Ordering::Equal => masked,
                };
                acc = Some(match acc {
                    None => shifted,
                    Some(a) => self.b.bin(BinOp::Or, ty, a, shifted),
                });
            }
        }
        acc.unwrap_or(v)
    }

    /// Reverse bit order using the standard swap network.
    fn bit_reverse(&mut self, v: Value, ty: Ty) -> Value {
        const MASKS: [(u64, u32); 6] = [
            (0x5555_5555_5555_5555, 1),
            (0x3333_3333_3333_3333, 2),
            (0x0f0f_0f0f_0f0f_0f0f, 4),
            (0x00ff_00ff_00ff_00ff, 8),
            (0x0000_ffff_0000_ffff, 16),
            (0x0000_0000_ffff_ffff, 32),
        ];
        let bits = ty.bits();
        let mut cur = v;
        for (mask, shift) in MASKS {
            if shift >= bits {
                break;
            }
            let m = self.konst(ty, mask & ty.mask());
            let s = self.konst(ty, shift as u64);
            let lo = self.b.bin(BinOp::And, ty, cur, m);
            let lo_sh = self.b.bin(BinOp::Shl, ty, lo, s);
            let hi_sh = self.b.bin(BinOp::ShrU, ty, cur, s);
            let hi = self.b.bin(BinOp::And, ty, hi_sh, m);
            cur = self.b.bin(BinOp::Or, ty, lo_sh, hi);
        }
        cur
    }

    // -- addressing --------------------------------------------------------

    /// Resolve an addressing mode into a base value plus a constant offset,
    /// performing any pre-index base update. Post-index writeback is deferred
    /// to [`Self::finish_address`] so it happens *after* the access.
    fn address(&mut self, addr: &AddrMode) -> Address {
        match *addr {
            AddrMode::Offset { rn, off } => {
                let base = self.rd_addr(rn);
                Address {
                    base,
                    offset: off,
                    post: None,
                }
            }
            AddrMode::PreIndex { rn, off } => {
                let base = self.rd_addr(rn);
                let delta = self.b.const_i64(off as u64);
                let updated = self.b.bin(BinOp::Add, Ty::I64, base, delta);
                self.wr_addr(rn, updated);
                Address {
                    base: updated,
                    offset: 0,
                    post: None,
                }
            }
            AddrMode::PostIndex { rn, off } => {
                let base = self.rd_addr(rn);
                let delta = self.b.const_i64(off as u64);
                Address {
                    base,
                    offset: 0,
                    post: Some((rn, delta)),
                }
            }
            AddrMode::PostIndexReg { rn, rm } => {
                let base = self.rd_addr(rn);
                // Read before the access, because the access may write the
                // very register holding the stride.
                let delta = self.rd_addr(rm);
                Address {
                    base,
                    offset: 0,
                    post: Some((rn, delta)),
                }
            }
            AddrMode::RegOffset {
                rn,
                rm,
                ext,
                amount,
            } => {
                let base = self.rd_addr(rn);
                let index = self.extend_reg(rm, ext, amount, Ty::I64);
                let sum = self.b.bin(BinOp::Add, Ty::I64, base, index);
                Address {
                    base: sum,
                    offset: 0,
                    post: None,
                }
            }
            AddrMode::Literal { addr } => {
                let base = self.b.const_i64(addr);
                Address {
                    base,
                    offset: 0,
                    post: None,
                }
            }
        }
    }

    fn finish_address(&mut self, a: &Address) {
        if let Some((rn, delta)) = a.post {
            let base = self.rd_addr(rn);
            let updated = self.b.bin(BinOp::Add, Ty::I64, base, delta);
            self.wr_addr(rn, updated);
        }
    }

    // -- the dispatcher ----------------------------------------------------

    /// Lift one instruction located at `pc`. Returns how the block continues.
    /// One instruction. A word with no lowering that [`gex_isa::slow`] understands is carried
    /// out by the host; the rest of the refusals stay refusals, for the caller to make a SIGILL.
    pub fn lift(&mut self, inst: &Inst, pc: u64) -> Flow {
        let mark = self.b.mark();
        let flags = self.flags;
        let flow = self.lift_inst(inst, pc);
        let raw = self.raw;
        let refused = self.b.has_inexpressible(mark)
            || matches!(flow, Flow::End(Term::Illegal { raw: r, pc: p }) if r == raw && p == pc);
        if !refused || !gex_isa::slow::known(raw) {
            return flow;
        }
        self.b.rewind(mark);
        self.flags = flags;
        let pcv = self.b.const_i64(pc);
        let id = self.b.const_i32(gex_isa::slow::SYSREG as u32);
        let word = self.b.const_i64(raw as u64);
        self.b.stmt(Stmt::HostCall { func: HostFunc::SysRegWrite, args: vec![pcv, id, word] });
        Flow::Next
    }

    fn lift_inst(&mut self, inst: &Inst, pc: u64) -> Flow {
        let next = pc + INST_BYTES;
        match *inst {
            Inst::Unknown { raw } | Inst::Undefined { raw } => Flow::End(Term::Illegal { raw, pc }),

            // Hints are no-ops for us, including the PAC hints: with PAC
            // unimplemented, sign and authenticate are both the identity.
            Inst::Hint { .. } => Flow::Next,

            // Clearing the monitor makes the next store-exclusive fail, which
            // is exactly what CLREX must guarantee.
            Inst::Clrex => {
                let zero = self.b.const_i64(0);
                self.b.write(GuestReg::MonSize, zero);
                Flow::Next
            }

            Inst::Adr { rd, target } | Inst::Adrp { rd, target } => {
                let v = self.b.const_i64(target);
                self.wr_addr(rd, v);
                Flow::Next
            }

            Inst::AddSubImm {
                sz,
                op,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let b = self.konst(ty, imm & ty.mask());
                self.arith(ty, op, set_flags, rd, sz, a, b);
                Flow::Next
            }

            Inst::AddSubShifted {
                sz,
                op,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let raw_b = self.rd_gp(rm, sz);
                let b = self.shift(raw_b, ty, shift, amount);
                self.arith(ty, op, set_flags, rd, sz, a, b);
                Flow::Next
            }

            Inst::AddSubExtended {
                sz,
                op,
                set_flags,
                rd,
                rn,
                rm,
                ext,
                amount,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let b = self.extend_reg(rm, ext, amount, ty);
                self.arith(ty, op, set_flags, rd, sz, a, b);
                Flow::Next
            }

            Inst::AddSubCarry {
                sz,
                op,
                set_flags,
                rd,
                rn,
                rm,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let b0 = self.rd_gp(rm, sz);
                let b = if op == AddOp::Sub {
                    self.b.un(UnOp::Not, ty, b0)
                } else {
                    b0
                };
                let cf = self.b.read(GuestReg::C);
                let r = self.add_with_carry(ty, a, b, Carry::Flag(cf));
                self.wr_gp(rd, sz, r.value);
                if set_flags {
                    self.set_nzcv(&r);
                }
                Flow::Next
            }

            Inst::LogicalImm {
                sz,
                op,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let b = self.konst(ty, imm & ty.mask());
                self.logical(ty, op, false, set_flags, rd, sz, a, b);
                Flow::Next
            }

            Inst::LogicalShifted {
                sz,
                op,
                set_flags,
                invert,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let raw_b = self.rd_gp(rm, sz);
                let b = self.shift(raw_b, ty, shift, amount);
                self.logical(ty, op, invert, set_flags, rd, sz, a, b);
                Flow::Next
            }

            Inst::MoveWide {
                sz,
                op,
                rd,
                imm16,
                shift,
            } => {
                let ty = ty_of(sz);
                let field = (imm16 << shift) & ty.mask();
                match op {
                    MoveOp::Movz => {
                        let v = self.konst(ty, field);
                        self.wr_gp(rd, sz, v);
                    }
                    MoveOp::Movn => {
                        let v = self.konst(ty, !field & ty.mask());
                        self.wr_gp(rd, sz, v);
                    }
                    MoveOp::Movk => {
                        let old = self.rd_gp(rd, sz);
                        let keep = self.konst(ty, !(0xffffu64 << shift) & ty.mask());
                        let cleared = self.b.bin(BinOp::And, ty, old, keep);
                        let ins = self.konst(ty, field);
                        let v = self.b.bin(BinOp::Or, ty, cleared, ins);
                        self.wr_gp(rd, sz, v);
                    }
                }
                Flow::Next
            }

            Inst::Bitfield {
                sz,
                op,
                rd,
                rn,
                immr,
                imms,
            } => {
                self.bitfield(sz, op, rd, rn, immr, imms);
                Flow::Next
            }

            Inst::Extract {
                sz,
                rd,
                rn,
                rm,
                lsb,
            } => {
                let ty = ty_of(sz);
                let bits = ty.bits();
                let low = self.rd_gp(rm, sz);
                let v = if lsb == 0 {
                    low
                } else {
                    let high = self.rd_gp(rn, sz);
                    let sh_lo = self.konst(ty, lsb as u64);
                    let lo_part = self.b.bin(BinOp::ShrU, ty, low, sh_lo);
                    let sh_hi = self.konst(ty, (bits - lsb) as u64);
                    let hi_part = self.b.bin(BinOp::Shl, ty, high, sh_hi);
                    self.b.bin(BinOp::Or, ty, lo_part, hi_part)
                };
                self.wr_gp(rd, sz, v);
                Flow::Next
            }

            Inst::CondSelect {
                sz,
                rd,
                rn,
                rm,
                cond,
                else_inc,
                else_inv,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let raw_b = self.rd_gp(rm, sz);
                let mut alt = raw_b;
                if else_inv {
                    alt = self.b.un(UnOp::Not, ty, alt);
                }
                if else_inc {
                    let one = self.konst(ty, 1);
                    alt = self.b.bin(BinOp::Add, ty, alt, one);
                }
                let c = self.eval_cond(cond);
                let v = self.b.select(c, a, alt, ty);
                self.wr_gp(rd, sz, v);
                Flow::Next
            }

            Inst::CondCompare {
                sz,
                op,
                rn,
                rm,
                imm,
                nzcv,
                cond,
            } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let operand = match rm {
                    Some(r) => self.rd_gp(r, sz),
                    None => self.konst(ty, imm),
                };
                let (b, carry) = if op == AddOp::Sub {
                    let neg = self.b.un(UnOp::Not, ty, operand);
                    (neg, Carry::One)
                } else {
                    (operand, Carry::Zero)
                };
                let r = self.add_with_carry(ty, a, b, carry);
                // When the condition fails, the immediate NZCV is installed
                // verbatim instead of the comparison's flags.
                let c = self.eval_cond(cond);
                let alt_n = self.b.const_i32((nzcv >> 3) as u32 & 1);
                let alt_z = self.b.const_i32((nzcv >> 2) as u32 & 1);
                let alt_c = self.b.const_i32((nzcv >> 1) as u32 & 1);
                let alt_v = self.b.const_i32(nzcv as u32 & 1);
                let n = self.b.select(c, r.n, alt_n, Ty::I32);
                let z = self.b.select(c, r.z, alt_z, Ty::I32);
                let cf = self.b.select(c, r.c, alt_c, Ty::I32);
                let vf = self.b.select(c, r.v, alt_v, Ty::I32);
                // A select between a comparison and a literal NZCV is not a
                // comparison, so the next branch reads the flags.
                self.flags = FlagSource::Unknown;
                self.b.write(GuestReg::N, n);
                self.b.write(GuestReg::Z, z);
                self.b.write(GuestReg::C, cf);
                self.b.write(GuestReg::V, vf);
                Flow::Next
            }

            Inst::DataProc1 { sz, op, rd, rn } => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let v = match op {
                    Dp1Op::Rbit => self.bit_reverse(a, ty),
                    Dp1Op::Rev16 => self.byte_reverse(a, ty, 2),
                    Dp1Op::Rev32 => self.byte_reverse(a, ty, 4),
                    Dp1Op::Rev => self.byte_reverse(a, ty, ty.bits() / 8),
                    Dp1Op::Clz => self.b.un(UnOp::Clz, ty, a),
                    Dp1Op::Cls => {
                        // Leading sign bits: clz(x ^ (x >>s (bits-1))) - 1.
                        let shift = self.konst(ty, (ty.bits() - 1) as u64);
                        let sign = self.b.bin(BinOp::ShrS, ty, a, shift);
                        let folded = self.b.bin(BinOp::Xor, ty, a, sign);
                        let clz = self.b.un(UnOp::Clz, ty, folded);
                        let one = self.konst(ty, 1);
                        self.b.bin(BinOp::Sub, ty, clz, one)
                    }
                };
                self.wr_gp(rd, sz, v);
                Flow::Next
            }

            Inst::DataProc2 { sz, op, rd, rn, rm } => self.dataproc2(sz, op, rd, rn, rm, pc),

            Inst::DataProc3 {
                sz,
                op,
                rd,
                rn,
                rm,
                ra,
            } => {
                self.dataproc3(sz, op, rd, rn, rm, ra);
                Flow::Next
            }

            // -- control flow --------------------------------------------------
            Inst::Branch { target, link } => {
                if link {
                    let ret = self.b.const_i64(next);
                    self.b.write(GuestReg::X(30), ret);
                    self.b.set_call_return(next);
                }
                Flow::End(Term::Goto { target })
            }

            Inst::BranchCond { cond, target } => {
                if cond.is_always() {
                    return Flow::End(Term::Goto { target });
                }
                let c = self.eval_cond(cond);
                Flow::End(Term::Branch {
                    cond: c,
                    taken: target,
                    not_taken: next,
                })
            }

            Inst::CompareBranch {
                sz,
                rt,
                target,
                nonzero,
            } => {
                let ty = ty_of(sz);
                let v = self.rd_gp(rt, sz);
                let zero = self.konst(ty, 0);
                let c = self
                    .b
                    .cmp(if nonzero { CmpOp::Ne } else { CmpOp::Eq }, ty, v, zero);
                Flow::End(Term::Branch {
                    cond: c,
                    taken: target,
                    not_taken: next,
                })
            }

            Inst::TestBranch {
                rt,
                bit,
                target,
                nonzero,
            } => {
                // The tested bit index selects the operand width: bits 32-63
                // are only reachable through the 64-bit view.
                let sz = if bit >= 32 { Sz::X } else { Sz::W };
                let ty = ty_of(sz);
                let v = self.rd_gp(rt, sz);
                let mask = self.konst(ty, 1u64 << (bit % ty.bits()));
                let masked = self.b.bin(BinOp::And, ty, v, mask);
                let zero = self.konst(ty, 0);
                let c = self.b.cmp(
                    if nonzero { CmpOp::Ne } else { CmpOp::Eq },
                    ty,
                    masked,
                    zero,
                );
                Flow::End(Term::Branch {
                    cond: c,
                    taken: target,
                    not_taken: next,
                })
            }

            Inst::BranchReg { rn, link, is_ret } => {
                // Read the target *before* clobbering x30: `blr x30` is legal
                // and not uncommon.
                let target = self.rd_addr(rn);
                if link {
                    let ret = self.b.const_i64(next);
                    self.b.write(GuestReg::X(30), ret);
                    self.b.set_call_return(next);
                }
                Flow::End(Term::Indirect { target, is_ret })
            }

            Inst::Svc { imm } => Flow::End(Term::Syscall { next, imm }),

            Inst::Brk { imm } => {
                let pcv = self.b.const_i64(pc);
                let immv = self.b.const_i32(imm);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::Breakpoint,
                    args: vec![pcv, immv],
                });
                Flow::End(Term::Goto { target: next })
            }

            // -- system ----------------------------------------------------
            Inst::SysRegRead { rt, sysreg } => {
                let v = self.sysreg_read(sysreg, pc);
                self.wr_addr(rt, v);
                Flow::Next
            }

            Inst::SysRegWrite { rt, sysreg } => {
                let v = self.rd_addr(rt);
                self.sysreg_write(sysreg, v, pc);
                Flow::Next
            }

            Inst::SysMaintenance { op, rt } => {
                // A data-cache clean is a no-op on a coherent host and is not
                // worth a host call.
                if op == SysMaintOp::DcClean {
                    return Flow::Next;
                }
                let addr = self.rd_addr(rt);
                let code = match op {
                    SysMaintOp::DcZva => gex_abi::cache_op::DC_ZVA,
                    SysMaintOp::IcInvalidate => gex_abi::cache_op::IC_INVALIDATE,
                    SysMaintOp::DcClean => gex_abi::cache_op::DC_CLEAN,
                    SysMaintOp::Other => gex_abi::cache_op::OTHER,
                };
                let pcv = self.b.const_i64(pc);
                let codev = self.b.const_i32(code as u32);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::CacheOp,
                    args: vec![pcv, codev, addr],
                });
                Flow::Next
            }

            Inst::Barrier { kind, .. } => {
                match kind {
                    // ISB is the architecture's promise that previously
                    // written code is now visible: our translation
                    // invalidation hook.
                    BarrierKind::Isb => {
                        let pcv = self.b.const_i64(pc);
                        let code = self.b.const_i32(gex_abi::cache_op::ISB as u32);
                        let zero = self.b.const_i64(0);
                        self.b.stmt(Stmt::HostCall {
                            func: HostFunc::CacheOp,
                            args: vec![pcv, code, zero],
                        });
                    }
                    _ => self.b.stmt(Stmt::Fence),
                }
                Flow::Next
            }

            // -- memory ----------------------------------------------------
            Inst::Load {
                sz,
                dest,
                signed,
                rt,
                ref addr,
                order,
            } => {
                let a = self.address(addr);
                let ty = ty_of(dest);
                let v = self.b.def(Expr::Load {
                    addr: a.base,
                    offset: a.offset,
                    sz: mem_sz(sz),
                    signed,
                    ty,
                    order: order_of(order),
                });
                self.wr_gp(rt, dest, v);
                self.finish_address(&a);
                Flow::Next
            }

            Inst::Store {
                sz,
                rt,
                ref addr,
                order,
            } => {
                let width = if sz == MemSz::D { Sz::X } else { Sz::W };
                let val = self.rd_gp(rt, width);
                let a = self.address(addr);
                self.b.stmt(Stmt::Store {
                    addr: a.base,
                    offset: a.offset,
                    val,
                    sz: mem_sz(sz),
                    order: order_of(order),
                });
                self.finish_address(&a);
                Flow::Next
            }

            Inst::LoadPair {
                sz,
                signed,
                rt,
                rt2,
                ref addr,
            } => {
                let a = self.address(addr);
                let (ty, dest) = if sz == MemSz::D || signed {
                    (Ty::I64, Sz::X)
                } else {
                    (Ty::I32, Sz::W)
                };
                let step = sz.bytes() as i64;
                let v1 = self.b.def(Expr::Load {
                    addr: a.base,
                    offset: a.offset,
                    sz: mem_sz(sz),
                    signed,
                    ty,
                    order: Order::Relaxed,
                });
                let v2 = self.b.def(Expr::Load {
                    addr: a.base,
                    offset: a.offset + step,
                    sz: mem_sz(sz),
                    signed,
                    ty,
                    order: Order::Relaxed,
                });
                self.wr_gp(rt, dest, v1);
                self.wr_gp(rt2, dest, v2);
                self.finish_address(&a);
                Flow::Next
            }

            Inst::StorePair {
                sz,
                rt,
                rt2,
                ref addr,
            } => {
                let width = if sz == MemSz::D { Sz::X } else { Sz::W };
                let v1 = self.rd_gp(rt, width);
                let v2 = self.rd_gp(rt2, width);
                let a = self.address(addr);
                let step = sz.bytes() as i64;
                self.b.stmt(Stmt::Store {
                    addr: a.base,
                    offset: a.offset,
                    val: v1,
                    sz: mem_sz(sz),
                    order: Order::Relaxed,
                });
                self.b.stmt(Stmt::Store {
                    addr: a.base,
                    offset: a.offset + step,
                    val: v2,
                    sz: mem_sz(sz),
                    order: Order::Relaxed,
                });
                self.finish_address(&a);
                Flow::Next
            }

            // The tag is not kept: zero the granules (STZG / STZ2G) and
            // write the base back.
            Inst::TagStore { ref addr, zero } => {
                let a = self.address(addr);
                let z = self.b.const_i64(0);
                for i in 0..(zero as i64 / 8) {
                    self.b.stmt(Stmt::Store {
                        addr: a.base,
                        offset: a.offset + 8 * i,
                        val: z,
                        sz: gex_ir::MemSz::B8,
                        order: Order::Relaxed,
                    });
                }
                self.finish_address(&a);
                Flow::Next
            }

            // Prefetch is advisory and has no writeback forms, so the whole
            // instruction is a no-op.
            Inst::Prefetch { .. } => Flow::Next,

            Inst::LoadExclusive { sz, rt, rn, order } => {
                let addr = self.rd_addr(rn);
                let ty = if sz == MemSz::D { Ty::I64 } else { Ty::I32 };
                let v = self.b.def(Expr::Load {
                    addr,
                    offset: 0,
                    sz: mem_sz(sz),
                    signed: false,
                    ty,
                    order: order_of(order),
                });
                let v64 = match ty {
                    Ty::I32 => self.b.extend(v, false),
                    _ => v,
                };
                self.b.write(GuestReg::MonAddr, addr);
                self.b.write(GuestReg::MonValue, v64);
                let size = self.b.const_i64(sz.bytes() as u64);
                self.b.write(GuestReg::MonSize, size);
                let dest = if sz == MemSz::D { Sz::X } else { Sz::W };
                self.wr_gp(rt, dest, v);
                Flow::Next
            }

            Inst::StoreExclusive { sz, rs, rt, rn, .. } => {
                self.store_exclusive(sz, rs, rt, rn);
                Flow::Next
            }

            Inst::CompareSwap { sz, rs, rt, rn, .. } => {
                let ty = if sz == MemSz::D { Ty::I64 } else { Ty::I32 };
                let width = if sz == MemSz::D { Sz::X } else { Sz::W };
                let addr = self.rd_addr(rn);
                let expected = self.rd_gp(rs, width);
                let replacement = self.rd_gp(rt, width);
                let old = self.b.def(Expr::CmpXchg {
                    addr,
                    expected,
                    replacement,
                    sz: mem_sz(sz),
                    ty,
                });
                // CAS writes the value actually read back into Rs.
                self.wr_gp(rs, width, old);
                Flow::Next
            }

            Inst::CompareSwapPair { sz, rs, rt, rn, .. } => {
                // The compare pair Rs:Rs+1 receives the pair read from memory;
                // the swap pair Rt:Rt+1 is written when *both* elements match,
                // and neither is written otherwise. All four registers are
                // read before either compare register is written.
                let addr = self.rd_addr(rn);
                let (rs1, rt1) = (pair_next(rs, 1), pair_next(rt, 1));
                if sz != MemSz::D {
                    // A pair of words is one aligned doubleword (CASP requires
                    // the pair's alignment): one 64-bit cmpxchg, exact, the
                    // first element in the low half.
                    let pack = |l: &mut Self, lo: RegSel, hi: RegSel| {
                        let lo = l.rd_gp(lo, Sz::W);
                        let hi = l.rd_gp(hi, Sz::W);
                        let lo = l.b.extend(lo, false);
                        let hi = l.b.extend(hi, false);
                        let k = l.konst(Ty::I64, 32);
                        let hi = l.b.bin(BinOp::Shl, Ty::I64, hi, k);
                        l.b.bin(BinOp::Or, Ty::I64, lo, hi)
                    };
                    let expected = pack(self, rs, rs1);
                    let replacement = pack(self, rt, rt1);
                    let old = self.b.def(Expr::CmpXchg {
                        addr,
                        expected,
                        replacement,
                        sz: gex_ir::MemSz::B8,
                        ty: Ty::I64,
                    });
                    let lo = self.b.wrap(old);
                    let k = self.konst(Ty::I64, 32);
                    let hi = self.b.bin(BinOp::ShrU, Ty::I64, old, k);
                    let hi = self.b.wrap(hi);
                    self.wr_gp(rs, Sz::W, lo);
                    self.wr_gp(rs1, Sz::W, hi);
                    return Flow::Next;
                }
                // A pair of doublewords: wasm has no 128-bit atomic, so the
                // pair is checked whole first and then swapped one element at
                // a time, the first undone if the second no longer matches.
                // With one thread this is exact (no half-written pair, ever);
                // with several, another thread can still see the first
                // element swapped for the instant before the second's
                // cmpxchg, or change the pair between the check and the swap
                // -- what only a real 128-bit atomic would close.
                let e0 = self.rd_gp(rs, Sz::X);
                let e1 = self.rd_gp(rs1, Sz::X);
                let n0 = self.rd_gp(rt, Sz::X);
                let n1 = self.rd_gp(rt1, Sz::X);
                let eight = self.konst(Ty::I64, 8);
                let a1 = self.b.bin(BinOp::Add, Ty::I64, addr, eight);
                let load = |l: &mut Self, a: Value| {
                    l.b.def(Expr::Load {
                        addr: a,
                        offset: 0,
                        sz: gex_ir::MemSz::B8,
                        signed: false,
                        ty: Ty::I64,
                        order: Order::SeqCst,
                    })
                };
                let cas = |l: &mut Self, a: Value, expected: Value, replacement: Value| {
                    l.b.def(Expr::CmpXchg {
                        addr: a,
                        expected,
                        replacement,
                        sz: gex_ir::MemSz::B8,
                        ty: Ty::I64,
                    })
                };
                let l0 = load(self, addr);
                let l1 = load(self, a1);
                let m0 = self.b.cmp(CmpOp::Eq, Ty::I64, l0, e0);
                let m1 = self.b.cmp(CmpOp::Eq, Ty::I64, l1, e1);
                let matched = self.b.bin(BinOp::And, Ty::I32, m0, m1);
                // Element 0: swapped only when the whole pair matched (else
                // a cmpxchg of the expected value with itself, which never
                // changes memory).
                let r0_new = self.b.select(matched, n0, e0, Ty::I64);
                let r0 = cas(self, addr, e0, r0_new);
                let same0 = self.b.cmp(CmpOp::Eq, Ty::I64, r0, e0);
                let ok0 = self.b.bin(BinOp::And, Ty::I32, matched, same0);
                // Element 1: swapped only when element 0 was.
                let r1_new = self.b.select(ok0, n1, e1, Ty::I64);
                let r1 = cas(self, a1, e1, r1_new);
                let differ1 = self.b.cmp(CmpOp::Ne, Ty::I64, r1, e1);
                // Element 0 swapped but element 1 did not: put element 0
                // back (else, again, a cmpxchg that cannot change memory).
                let undo = self.b.bin(BinOp::And, Ty::I32, ok0, differ1);
                let back = self.b.select(undo, e0, n0, Ty::I64);
                cas(self, addr, n0, back);
                // What the guest compares: the pair the decision was made
                // on. A failed check returns the loads (one of which differs
                // from Rs:Rs+1); after it, the cmpxchgs' reads (both equal to
                // Rs:Rs+1 exactly when the pair was swapped).
                let out0 = self.b.select(matched, r0, l0, Ty::I64);
                let out1 = self.b.select(matched, r1, l1, Ty::I64);
                self.wr_gp(rs, Sz::X, out0);
                self.wr_gp(rs1, Sz::X, out1);
                Flow::Next
            }

            Inst::AtomicRmw {
                sz, op, rs, rt, rn, ..
            } => self.atomic_rmw(sz, op, rs, rt, rn, pc),

            // -- SIMD ------------------------------------------------------
            Inst::VLoad {
                bytes,
                vt,
                ref addr,
            } => {
                let a = self.address(addr);
                let v = self.b.def(Expr::VLoad {
                    addr: a.base,
                    offset: a.offset,
                    bytes,
                });
                self.b.write(GuestReg::V128(vt.0), v);
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VStore {
                bytes,
                vt,
                ref addr,
            } => {
                let val = self.b.read(GuestReg::V128(vt.0));
                let a = self.address(addr);
                self.b.stmt(Stmt::VStore {
                    addr: a.base,
                    offset: a.offset,
                    val,
                    bytes,
                });
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VLoadPair {
                bytes,
                vt,
                vt2,
                ref addr,
            } => {
                let a = self.address(addr);
                let v1 = self.b.def(Expr::VLoad {
                    addr: a.base,
                    offset: a.offset,
                    bytes,
                });
                let v2 = self.b.def(Expr::VLoad {
                    addr: a.base,
                    offset: a.offset + bytes as i64,
                    bytes,
                });
                self.b.write(GuestReg::V128(vt.0), v1);
                self.b.write(GuestReg::V128(vt2.0), v2);
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VStorePair {
                bytes,
                vt,
                vt2,
                ref addr,
            } => {
                let v1 = self.b.read(GuestReg::V128(vt.0));
                let v2 = self.b.read(GuestReg::V128(vt2.0));
                let a = self.address(addr);
                self.b.stmt(Stmt::VStore {
                    addr: a.base,
                    offset: a.offset,
                    val: v1,
                    bytes,
                });
                self.b.stmt(Stmt::VStore {
                    addr: a.base,
                    offset: a.offset + bytes as i64,
                    val: v2,
                    bytes,
                });
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VDupGeneral {
                vd,
                rn,
                lane_bytes,
                q,
            } => {
                let src = self.rd_gp(rn, if lane_bytes == 8 { Sz::X } else { Sz::W });
                let splat = self.b.def(Expr::VSplat { a: src, lane_bytes });
                let v = self.narrow_to_64(splat, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            Inst::VMoveImm { vd, value, .. } => {
                let v = self.b.def(Expr::VConst(Box::new(value.to_le_bytes())));
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            // -- scalar floating point -------------------------------------
            Inst::FpDataProc1 { sz, op, vd, vn } => {
                let a = self.rd_fp(vn, sz);
                let double = sz.is_double();
                let v = match op {
                    FpOp1::Mov => a,
                    FpOp1::Abs => self.b.def(Expr::FpUn {
                        op: FpUnOp::Abs,
                        double,
                        a,
                    }),
                    FpOp1::Neg => self.b.def(Expr::FpUn {
                        op: FpUnOp::Neg,
                        double,
                        a,
                    }),
                    FpOp1::Sqrt => self.b.def(Expr::FpUn {
                        op: FpUnOp::Sqrt,
                        double,
                        a,
                    }),
                    // FRINTA breaks ties away from zero; wasm's `nearest`
                    // breaks them to even, so it is built (`round_away`).
                    FpOp1::RoundAway => self.round_away(double, a),
                    // FRINTX/FRINTI follow the FPCR mode, which gex does not
                    // track in fast mode: to nearest, ties to even, as FRINTN.
                    FpOp1::RoundNearest
                    | FpOp1::RoundExact
                    | FpOp1::RoundCurrent => self.b.def(Expr::FpUn {
                        op: FpUnOp::RoundNearest,
                        double,
                        a,
                    }),
                    FpOp1::RoundPlus => self.b.def(Expr::FpUn {
                        op: FpUnOp::Ceil,
                        double,
                        a,
                    }),
                    FpOp1::RoundMinus => self.b.def(Expr::FpUn {
                        op: FpUnOp::Floor,
                        double,
                        a,
                    }),
                    FpOp1::RoundZero => self.b.def(Expr::FpUn {
                        op: FpUnOp::Trunc,
                        double,
                        a,
                    }),
                };
                self.wr_fp(vd, sz, v);
                Flow::Next
            }

            Inst::FpDataProc2 { sz, op, vd, vn, vm } => {
                let double = sz.is_double();
                let a = self.rd_fp(vn, sz);
                let b = self.rd_fp(vm, sz);
                let v = match op {
                    FpOp2::Add => self.fp_bin(FpBinOp::Add, double, a, b),
                    FpOp2::Sub => self.fp_bin(FpBinOp::Sub, double, a, b),
                    FpOp2::Mul => self.fp_bin(FpBinOp::Mul, double, a, b),
                    FpOp2::Div => self.fp_bin(FpBinOp::Div, double, a, b),
                    // Wasm's min/max propagate NaN, matching FMIN/FMAX. The
                    // FMINNM/FMAXNM "number" variants instead return the
                    // non-NaN operand; that difference only shows up with a
                    // NaN input, and modelling it needs a select chain.
                    FpOp2::Min | FpOp2::MinNum => self.fp_bin(FpBinOp::Min, double, a, b),
                    FpOp2::Max | FpOp2::MaxNum => self.fp_bin(FpBinOp::Max, double, a, b),
                    FpOp2::NMul => {
                        let p = self.fp_bin(FpBinOp::Mul, double, a, b);
                        self.b.def(Expr::FpUn {
                            op: FpUnOp::Neg,
                            double,
                            a: p,
                        })
                    }
                };
                self.wr_fp(vd, sz, v);
                Flow::Next
            }

            Inst::FpDataProc3 {
                sz,
                vd,
                vn,
                vm,
                va,
                neg_product,
                neg_addend,
            } => {
                let a = self.rd_fp(vn, sz);
                let b = self.rd_fp(vm, sz);
                let c = self.rd_fp(va, sz);
                // Negating an operand of the product is exact, so it can be
                // done before the fused step without changing the result.
                self.fused(vd, sz, a, b, c, neg_product, neg_addend)
            }

            Inst::VByElement {
                op,
                arr,
                vd,
                vn,
                vm,
                index,
            } => {
                // Scalar `fmla`/`fmls` are fused, as `fmadd` is. (The vector forms round the
                // product first: wasm has no vector FMA.)
                if arr.scalar && matches!(op, VOp3::FMla | VOp3::FMls) {
                    let sz = if arr.lane_bytes == 8 { FpSz::D } else { FpSz::S };
                    let a = self.rd_fp(vn, sz);
                    let m = self.b.read(GuestReg::V128(vm.0));
                    let b = self.b.def(Expr::VExtract {
                        a: m,
                        lane: index,
                        lane_bytes: arr.lane_bytes,
                        signed: false,
                        ty: if sz.is_double() { Ty::I64 } else { Ty::I32 },
                    });
                    let c = self.rd_fp(vd, sz);
                    return self.fused(vd, sz, a, b, c, op == VOp3::FMls, false);
                }
                // Broadcast the chosen lane, then this is an ordinary
                // three-same operation.
                let src = self.b.read(GuestReg::V128(vm.0));
                let mut lanes = [0u8; 16];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let byte = i as u32 % arr.lane_bytes;
                    *slot = (index * arr.lane_bytes + byte) as u8;
                }
                let b = self.b.def(Expr::VShuffle {
                    lanes: Box::new(lanes),
                    a: src,
                    b: src,
                });
                self.three_same_with(op, arr, vd, vn, b, pc)
            }

            Inst::VLogicalImm {
                vd,
                value,
                clear,
                q,
            } => {
                let bytes = if clear {
                    (!value).to_le_bytes()
                } else {
                    value.to_le_bytes()
                };
                let imm = self.b.def(Expr::VConst(Box::new(bytes)));
                let old = self.b.read(GuestReg::V128(vd.0));
                let r = self.b.def(Expr::VBin {
                    op: if clear { VBinOp::And } else { VBinOp::Or },
                    a: old,
                    b: imm,
                });
                // The 64-bit form clears the top half, as every write to a
                // `.8b`/`.4h`/`.2s` arrangement does.
                let r = self.narrow_to_64(r, q);
                self.b.write(GuestReg::V128(vd.0), r);
                Flow::Next
            }
            Inst::VNarrow {
                op,
                dst_bytes,
                upper,
                vd,
                vn,
                vm,
            } => {
                // Add/subtract at source width, then keep each result's high
                // half via a byte shuffle: wasm's `narrow` saturates rather
                // than truncating.
                let src = VArr {
                    lane_bytes: dst_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                let lane = Self::lane_of(src);
                let a = self.b.read(GuestReg::V128(vn.0));
                let b = self.b.read(GuestReg::V128(vm.0));
                let wide = match op {
                    VNarrowOp::AddHn | VNarrowOp::RAddHn => self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane,
                        a,
                        b,
                    }),
                    VNarrowOp::SubHn | VNarrowOp::RSubHn => self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Sub,
                        lane,
                        a,
                        b,
                    }),
                };
                // The rounding forms add half a result-lane before narrowing,
                // so the kept half is rounded rather than truncated.
                let wide = if matches!(op, VNarrowOp::RAddHn | VNarrowOp::RSubHn) {
                    let mut half = [0u8; 16];
                    let sb = (dst_bytes * 2) as usize;
                    for lane_i in 0..16 / sb {
                        // Half of 2^(8*dst_bytes) is a one in the top bit of
                        // the low half, which is byte dst_bytes-1.
                        half[lane_i * sb + dst_bytes as usize - 1] = 0x80;
                    }
                    let h = self.b.def(Expr::VConst(Box::new(half)));
                    self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane,
                        a: wide,
                        b: h,
                    })
                } else {
                    wide
                };
                // Pick the high `dst_bytes` of each source lane. The result
                // is 64 bits wide; where it lands in the register depends on
                // whether this is the `2` form.
                let mut pick = [0u8; 16];
                let sb = (dst_bytes * 2) as usize;
                let db = dst_bytes as usize;
                for out in 0..16 / sb {
                    for byte in 0..db {
                        pick[out * db + byte] = (out * sb + db + byte) as u8;
                    }
                }
                let narrowed = self.b.def(Expr::VShuffle {
                    a: wide,
                    b: wide,
                    lanes: Box::new(pick),
                });
                let out_arr = VArr {
                    lane_bytes: dst_bytes,
                    q: upper,
                    float: false,
                    scalar: false,
                };
                if upper {
                    // `addhn2` writes the top half and leaves the bottom as
                    // it was, so the eight bytes move up and the old value
                    // supplies the rest.
                    let mut up = [0u8; 16];
                    for i in 0..8 {
                        up[i] = i as u8;
                        up[8 + i] = i as u8;
                    }
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let moved = self.b.def(Expr::VShuffle {
                        a: narrowed,
                        b: narrowed,
                        lanes: Box::new(up),
                    });
                    let mut keep = [0u8; 16];
                    for i in 0..8 {
                        keep[i] = i as u8;
                        keep[8 + i] = (16 + 8 + i) as u8;
                    }
                    let r = self.b.def(Expr::VShuffle {
                        a: old,
                        b: moved,
                        lanes: Box::new(keep),
                    });
                    self.b.write(GuestReg::V128(vd.0), r);
                } else {
                    self.commit_vector(vd, out_arr, narrowed);
                }
                Flow::Next
            }
            Inst::VWiden {
                op,
                arr,
                upper,
                signed,
                vd,
                vn,
                vm,
                index,
            } => {
                // Everything here is "widen the narrow operands, then do an
                // ordinary lane-wise operation at the wider size". The `w`
                // forms simply skip widening their first operand.
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                let lane = Self::lane_of(wide);
                let widen = |s: &mut Self, v: Value| {
                    s.b.def(Expr::VCvt {
                        op: if upper {
                            VCvtOp::ExtendHigh { signed }
                        } else {
                            VCvtOp::ExtendLow { signed }
                        },
                        from: Self::lane_of(arr),
                        a: v,
                        b: None,
                    })
                };
                let raw_n = self.b.read(GuestReg::V128(vn.0));
                let mut raw_m = self.b.read(GuestReg::V128(vm.0));
                // The by-element forms multiply every lane of `vn` by one lane
                // of `vm`: broadcast that lane across the register first, then
                // the widen and multiply below are the ordinary vector shape.
                if let Some(idx) = index {
                    let lb = arr.lane_bytes as usize;
                    let base = idx as usize * lb;
                    let mut sel = [0u8; 16];
                    for (i, s) in sel.iter_mut().enumerate() {
                        *s = (base + (i % lb)) as u8;
                    }
                    raw_m = self.b.def(Expr::VShuffle {
                        lanes: Box::new(sel),
                        a: raw_m,
                        b: raw_m,
                    });
                }
                let a = match op {
                    VWidenOp::AddW | VWidenOp::SubW => raw_n,
                    _ => widen(self, raw_n),
                };
                let b = widen(self, raw_m);
                let r = match op {
                    VWidenOp::AddL | VWidenOp::AddW => self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane,
                        a,
                        b,
                    }),
                    VWidenOp::SubL | VWidenOp::SubW => self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Sub,
                        lane,
                        a,
                        b,
                    }),
                    VWidenOp::MulL => self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Mul,
                        lane,
                        a,
                        b,
                    }),
                    VWidenOp::MlaL | VWidenOp::MlsL => {
                        let prod = self.b.def(Expr::VLaneBin {
                            op: VLaneBinOp::Mul,
                            lane,
                            a,
                            b,
                        });
                        let acc = self.b.read(GuestReg::V128(vd.0));
                        self.b.def(Expr::VLaneBin {
                            op: if op == VWidenOp::MlaL {
                                VLaneBinOp::Add
                            } else {
                                VLaneBinOp::Sub
                            },
                            lane,
                            a: acc,
                            b: prod,
                        })
                    }
                    VWidenOp::AbdL | VWidenOp::AbaL => {
                        // Widening first means max-minus-min cannot overflow,
                        // so the absolute difference needs no saturation.
                        let (mx, mn) = if signed {
                            (VLaneBinOp::MaxS, VLaneBinOp::MinS)
                        } else {
                            (VLaneBinOp::MaxU, VLaneBinOp::MinU)
                        };
                        let hi = self.b.def(Expr::VLaneBin { op: mx, lane, a, b });
                        let lo = self.b.def(Expr::VLaneBin { op: mn, lane, a, b });
                        let d = self.b.def(Expr::VLaneBin {
                            op: VLaneBinOp::Sub,
                            lane,
                            a: hi,
                            b: lo,
                        });
                        if op == VWidenOp::AbdL {
                            d
                        } else {
                            let acc = self.b.read(GuestReg::V128(vd.0));
                            self.b.def(Expr::VLaneBin {
                                op: VLaneBinOp::Add,
                                lane,
                                a: acc,
                                b: d,
                            })
                        }
                    }
                };
                self.b.write(GuestReg::V128(vd.0), r);
                Flow::Next
            }

            Inst::VFpCvtPrecision {
                widen,
                narrow_bytes,
                upper,
                vd,
                vn,
            } => {
                // Only single <-> double is reachable; half precision has no
                // wasm lane type and stays undecoded.
                if narrow_bytes != 4 {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                }
                let src = self.b.read(GuestReg::V128(vn.0));
                if widen {
                    // `fcvtl2` promotes the upper two singles, and wasm can
                    // only promote the lower pair, so move them down first.
                    let src = if upper {
                        let mut sel = [0u8; 16];
                        for (i, slot) in sel.iter_mut().enumerate() {
                            *slot = (8 + i % 8) as u8;
                        }
                        self.b.def(Expr::VShuffle {
                            lanes: Box::new(sel),
                            a: src,
                            b: src,
                        })
                    } else {
                        src
                    };
                    let r = self.b.def(Expr::VCvt {
                        op: VCvtOp::PromoteF32,
                        from: VLane::F32,
                        a: src,
                        b: None,
                    });
                    self.b.write(GuestReg::V128(vd.0), r);
                } else {
                    let demoted = self.b.def(Expr::VCvt {
                        op: VCvtOp::DemoteF64,
                        from: VLane::F64,
                        a: src,
                        b: None,
                    });
                    if upper {
                        // `fcvtn2` writes the upper half and leaves the lower
                        // half of the destination alone.
                        let old = self.b.read(GuestReg::V128(vd.0));
                        let mut sel = [0u8; 16];
                        for (i, slot) in sel.iter_mut().enumerate() {
                            *slot = if i < 8 { i as u8 } else { (16 + i - 8) as u8 };
                        }
                        let r = self.b.def(Expr::VShuffle {
                            lanes: Box::new(sel),
                            a: old,
                            b: demoted,
                        });
                        self.b.write(GuestReg::V128(vd.0), r);
                    } else {
                        let r = self.narrow_to_64(demoted, false);
                        self.b.write(GuestReg::V128(vd.0), r);
                    }
                }
                Flow::Next
            }

            Inst::FpCompare { sz, vn, vm } => {
                let double = sz.is_double();
                let a = self.rd_fp(vn, sz);
                let b = match vm {
                    Some(vm) => self.rd_fp(vm, sz),
                    None => self.konst(if double { Ty::I64 } else { Ty::I32 }, 0),
                };
                // The architecture's mapping: equal -> 0110, less -> 1000,
                // greater -> 0010, unordered -> 0011. That is exactly
                // N = lt, Z = eq, C = NOT lt, V = unordered.
                let (n, z, c, v) = self.fp_compare_flags(double, a, b);
                // Floating-point NZCV does not follow the integer rule --
                // unordered sets C and V -- so it gets no shortcut.
                self.flags = FlagSource::Unknown;
                self.b.write(GuestReg::N, n);
                self.b.write(GuestReg::Z, z);
                self.b.write(GuestReg::C, c);
                self.b.write(GuestReg::V, v);
                Flow::Next
            }

            Inst::FpCondSelect {
                sz,
                vd,
                vn,
                vm,
                cond,
            } => {
                let a = self.rd_fp(vn, sz);
                let b = self.rd_fp(vm, sz);
                let c = self.eval_cond(cond);
                let ty = if sz.is_double() { Ty::I64 } else { Ty::I32 };
                let v = self.b.select(c, a, b, ty);
                self.wr_fp(vd, sz, v);
                Flow::Next
            }

            Inst::FpMoveImm { sz, vd, bits } => {
                let ty = if sz.is_double() { Ty::I64 } else { Ty::I32 };
                let v = self.konst(ty, bits);
                self.wr_fp(vd, sz, v);
                Flow::Next
            }

            Inst::FpConvertPrecision { from, to, vd, vn } => {
                let a = self.rd_fp(vn, from);
                let v = self.b.def(Expr::FpCvt {
                    kind: FpConvert::FpToFp,
                    src_wide: from.is_double(),
                    dst_wide: to.is_double(),
                    a,
                });
                self.wr_fp(vd, to, v);
                Flow::Next
            }

            Inst::FpFromInt {
                sz,
                int_sz,
                signed,
                vd,
                rn,
                fbits,
            } => {
                let a = self.rd_gp(rn, int_sz);
                let v = self.b.def(Expr::FpCvt {
                    kind: FpConvert::IntToFp { signed },
                    src_wide: int_sz.is64(),
                    dst_wide: sz.is_double(),
                    a,
                });
                // Fixed-point: the integer is scaled by 2^fbits, so divide it
                // back out. Scaling by a power of two is exact, so the result
                // stays correctly rounded.
                let v = self.scale_by_power_of_two(v, sz.is_double(), -(fbits as i32));
                self.wr_fp(vd, sz, v);
                Flow::Next
            }

            Inst::FpToInt {
                sz,
                int_sz,
                signed,
                round,
                rd,
                vn,
                fbits,
            } => {
                let double = sz.is_double();
                let a = self.rd_fp(vn, sz);
                // Scale up first, so the truncation keeps `fbits` of fraction.
                let a = self.scale_by_power_of_two(a, double, fbits as i32);
                // Only rounding toward zero is a single wasm instruction; the
                // others round first and then truncate, which is exact
                // because the rounded value is already integral.
                let rounded = match round {
                    FpRound::Zero => a,
                    FpRound::Nearest | FpRound::Away => self.b.def(Expr::FpUn {
                        op: FpUnOp::RoundNearest,
                        double,
                        a,
                    }),
                    FpRound::Plus => self.b.def(Expr::FpUn {
                        op: FpUnOp::Ceil,
                        double,
                        a,
                    }),
                    FpRound::Minus => self.b.def(Expr::FpUn {
                        op: FpUnOp::Floor,
                        double,
                        a,
                    }),
                };
                let v = self.b.def(Expr::FpCvt {
                    kind: FpConvert::FpToInt { signed },
                    src_wide: double,
                    dst_wide: int_sz.is64(),
                    a: rounded,
                });
                self.wr_gp(rd, int_sz, v);
                Flow::Next
            }

            Inst::FpMoveGeneral {
                sz,
                to_vector,
                rd,
                vd,
            } => {
                let int_sz = if sz.is_double() { Sz::X } else { Sz::W };
                if to_vector {
                    let a = self.rd_gp(rd, int_sz);
                    self.wr_fp(vd, sz, a);
                } else {
                    let a = self.rd_fp(vd, sz);
                    self.wr_gp(rd, int_sz, a);
                }
                Flow::Next
            }

            Inst::FpMoveTopHalf { to_vector, rd, vd } => {
                if to_vector {
                    let a = self.rd_addr(rd);
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let out = self.b.def(Expr::VReplace {
                        a: old,
                        lane: 1,
                        lane_bytes: 8,
                        val: a,
                    });
                    self.b.write(GuestReg::V128(vd.0), out);
                } else {
                    let vec = self.b.read(GuestReg::V128(vd.0));
                    let hi = self.b.def(Expr::VExtract {
                        a: vec,
                        lane: 1,
                        lane_bytes: 8,
                        signed: false,
                        ty: Ty::I64,
                    });
                    self.wr_addr(rd, hi);
                }
                Flow::Next
            }

            Inst::VMoveLane {
                to_vector,
                signed,
                lane_bytes,
                lane,
                int_sz,
                rd,
                vd,
            } => {
                if to_vector {
                    let val = self.rd_gp(rd, if lane_bytes == 8 { Sz::X } else { Sz::W });
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let out = self.b.def(Expr::VReplace {
                        a: old,
                        lane,
                        lane_bytes,
                        val,
                    });
                    self.b.write(GuestReg::V128(vd.0), out);
                } else {
                    let vec = self.b.read(GuestReg::V128(vd.0));
                    let ty = ty_of(int_sz);
                    let v = self.b.def(Expr::VExtract {
                        a: vec,
                        lane,
                        lane_bytes,
                        signed,
                        ty,
                    });
                    self.wr_gp(rd, int_sz, v);
                }
                Flow::Next
            }

            Inst::VLoadStoreLane {
                load,
                replicate,
                regs,
                lane_bytes,
                index,
                q,
                vt,
                ref addr,
            } => {
                let msz = match lane_bytes {
                    1 => gex_ir::MemSz::B1,
                    2 => gex_ir::MemSz::B2,
                    4 => gex_ir::MemSz::B4,
                    _ => gex_ir::MemSz::B8,
                };
                let ty = if lane_bytes == 8 { Ty::I64 } else { Ty::I32 };
                let a = self.address(addr);
                // The interleaved forms touch `regs` consecutive registers, each
                // one lane at consecutive `lane_bytes` offsets: `st4 {v2..v5}[0]`
                // writes v2[0], v3[0], v4[0], v5[0] to four consecutive words.
                // v31 wraps to v0.
                for i in 0..regs as u32 {
                    let reg = (vt.0 as u32 + i) % 32;
                    let off = a.offset + (i as i64) * lane_bytes as i64;
                    if load {
                        let val = self.b.def(Expr::Load {
                            addr: a.base,
                            offset: off,
                            sz: msz,
                            signed: false,
                            ty,
                            order: Order::Relaxed,
                        });
                        let v = if replicate {
                            let full = self.b.def(Expr::VSplat { a: val, lane_bytes });
                            // The 64-bit arrangement leaves the upper half zero.
                            self.narrow_to_64(full, q)
                        } else {
                            let old = self.b.read(GuestReg::V128(reg as u8));
                            self.b.def(Expr::VReplace {
                                a: old,
                                lane: index,
                                lane_bytes,
                                val,
                            })
                        };
                        self.b.write(GuestReg::V128(reg as u8), v);
                    } else {
                        let old = self.b.read(GuestReg::V128(reg as u8));
                        let val = self.b.def(Expr::VExtract {
                            a: old,
                            lane: index,
                            lane_bytes,
                            signed: false,
                            ty,
                        });
                        self.b.stmt(Stmt::Store {
                            addr: a.base,
                            offset: off,
                            val,
                            sz: msz,
                            order: Order::Relaxed,
                        });
                    }
                }
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VLoadStoreN {
                load,
                regs,
                lane_bytes,
                bytes,
                vt,
                ref addr,
            } => self.load_store_n(load, regs, lane_bytes, bytes, vt, addr),

            Inst::VLoadStore2 {
                load,
                lane_bytes,
                bytes,
                vt,
                ref addr,
            } => {
                // Two registers' worth of memory holds the lanes interleaved.
                // Byte `2*i*L + k` of the block is byte `k` of lane `i` of the
                // first register; the odd positions belong to the second. Both
                // directions are therefore a pair of shuffles.
                let lanes_each = bytes / lane_bytes;
                let lb = lane_bytes;
                let a = self.address(addr);
                if load {
                    let lo = self.b.def(Expr::VLoad {
                        addr: a.base,
                        offset: a.offset,
                        bytes: 16.min(bytes * 2),
                    });
                    let hi = if bytes == 16 {
                        self.b.def(Expr::VLoad {
                            addr: a.base,
                            offset: a.offset + 16,
                            bytes: 16,
                        })
                    } else {
                        lo
                    };
                    for half in 0..2u32 {
                        let mut sel = [0u8; 16];
                        for i in 0..lanes_each {
                            for k in 0..lb {
                                sel[(i * lb + k) as usize] = ((2 * i + half) * lb + k) as u8;
                            }
                        }
                        let v = self.b.def(Expr::VShuffle {
                            lanes: Box::new(sel),
                            a: lo,
                            b: hi,
                        });
                        let v = self.narrow_to_64(v, bytes == 16);
                        self.b.write(GuestReg::V128((vt.0 + half as u8) % 32), v);
                    }
                } else {
                    let v0 = self.b.read(GuestReg::V128(vt.0));
                    let v1 = self.b.read(GuestReg::V128((vt.0 + 1) % 32));
                    // Each output block of 16 bytes alternates lanes taken
                    // from v0 and v1, starting at lane `block * 8 / lb`.
                    let blocks = bytes * 2 / 16;
                    for blk in 0..blocks.max(1) {
                        let mut sel = [0u8; 16];
                        let out_bytes = if bytes == 16 { 16 } else { bytes * 2 };
                        for pos in 0..out_bytes {
                            let byte = blk * 16 + pos;
                            let lane = byte / (2 * lb);
                            let from_second = (byte / lb) % 2 == 1;
                            let k = byte % lb;
                            sel[pos as usize] =
                                (lane * lb + k + if from_second { 16 } else { 0 }) as u8;
                        }
                        let v = self.b.def(Expr::VShuffle {
                            lanes: Box::new(sel),
                            a: v0,
                            b: v1,
                        });
                        self.b.stmt(Stmt::VStore {
                            addr: a.base,
                            offset: a.offset + (blk as i64) * 16,
                            val: v,
                            bytes: if bytes == 16 { 16 } else { bytes * 2 },
                        });
                    }
                }
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VLoadMulti {
                regs,
                vt,
                bytes,
                ref addr,
            } => {
                let a = self.address(addr);
                for i in 0..regs {
                    let v = self.b.def(Expr::VLoad {
                        addr: a.base,
                        offset: a.offset + (i as i64) * bytes as i64,
                        bytes,
                    });
                    self.b.write(GuestReg::V128((vt.0 + i) % 32), v);
                }
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VStoreMulti {
                regs,
                vt,
                bytes,
                ref addr,
            } => {
                let vals: Vec<Value> = (0..regs)
                    .map(|i| self.b.read(GuestReg::V128((vt.0 + i) % 32)))
                    .collect();
                let a = self.address(addr);
                for (i, val) in vals.into_iter().enumerate() {
                    self.b.stmt(Stmt::VStore {
                        addr: a.base,
                        offset: a.offset + (i as i64) * bytes as i64,
                        val,
                        bytes,
                    });
                }
                self.finish_address(&a);
                Flow::Next
            }

            Inst::VCnt { vd, vn, q } => {
                let a = self.b.read(GuestReg::V128(vn.0));
                let counted = self.b.def(Expr::VUn {
                    op: VUnOp::PopcntBytes,
                    a,
                });
                let v = self.narrow_to_64(counted, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            // `saddlv`/`uaddlv`: wasm's pairwise add widens each pair in place, and the sum of
            // those cannot overflow the wider lane (16 bytes into 16 bits, 8 halfwords into 32),
            // so the plain fold below finishes it on them. The upper half of a 64-bit form is
            // not an input, so it is cleared first. Word lanes would take a 64-bit pairwise add
            // wasm lacks, and have not been seen: refused, rather than guessed.
            Inst::VAcross { op: op @ (VAcrossOp::SAddLv | VAcrossOp::UAddLv), arr, vd, vn } => {
                let signed = op == VAcrossOp::SAddLv;
                let pairwise = match (arr.lane_bytes, signed) {
                    (1, true) => VUnOp::ExtAddPairwiseS8,
                    (1, false) => VUnOp::ExtAddPairwiseU8,
                    (2, true) => VUnOp::ExtAddPairwiseS16,
                    (2, false) => VUnOp::ExtAddPairwiseU16,
                    _ => return Flow::End(Term::Illegal { raw: self.raw, pc }),
                };
                let a = self.b.read(GuestReg::V128(vn.0));
                let a = self.narrow_to_64(a, arr.q);
                let mut acc = self.b.def(Expr::VUn { op: pairwise, a });
                let lb = arr.lane_bytes * 2;
                let lane = if lb == 2 { VLane::I16 } else { VLane::I32 };
                let mut span = arr.lanes() / 2;
                while span > 1 {
                    let half = span / 2;
                    let mut lanes = [0u8; 16];
                    for (i, slot) in lanes.iter_mut().enumerate() {
                        let i = i as u32;
                        let l = i / lb;
                        *slot = if l < half { ((l + half) * lb + i % lb) as u8 } else { i as u8 };
                    }
                    let shifted = self.b.def(Expr::VShuffle { a: acc, b: acc, lanes: Box::new(lanes) });
                    acc = self.b.def(Expr::VLaneBin { op: VLaneBinOp::Add, lane, a: acc, b: shifted });
                    span = half;
                }
                let value = self.b.def(Expr::VExtract { a: acc, lane: 0, lane_bytes: lb, signed: false, ty: Ty::I32 });
                let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                let out = self.b.def(Expr::VReplace { a: zero, lane: 0, lane_bytes: lb, val: value });
                self.b.write(GuestReg::V128(vd.0), out);
                Flow::Next
            }

            Inst::VAcross { op, arr, vd, vn } => {
                // Wasm has no horizontal reduction, so fold the vector in
                // half repeatedly: log2(lanes) shuffles and lane-wise ops
                // leave the answer in lane 0.
                let lane = Self::lane_of(arr);
                let bin = match op {
                    VAcrossOp::Add => VLaneBinOp::Add,
                    // As with the three-same forms, the `nm` variants share a
                    // lowering with the plain ones: wasm's max/min already
                    // propagate NaN like FMAX, and differ from FMAXNM only when
                    // an input is NaN. See README's fidelity notes.
                    VAcrossOp::SMax | VAcrossOp::FMax | VAcrossOp::FMaxNm => VLaneBinOp::MaxS,
                    VAcrossOp::SMin | VAcrossOp::FMin | VAcrossOp::FMinNm => VLaneBinOp::MinS,
                    VAcrossOp::UMax => VLaneBinOp::MaxU,
                    VAcrossOp::UMin => VLaneBinOp::MinU,
                    VAcrossOp::SAddLv | VAcrossOp::UAddLv => unreachable!("taken above"),
                };
                let mut acc = self.b.read(GuestReg::V128(vn.0));
                let lb = arr.lane_bytes;
                // The scalar pairwise forms reduce the two lanes of their
                // source, not the one lane of their destination.
                let mut span = if arr.scalar { 2 } else { arr.lanes() };
                while span > 1 {
                    let half = span / 2;
                    // Bring the upper half of the live range down.
                    let mut lanes = [0u8; 16];
                    for (i, slot) in lanes.iter_mut().enumerate() {
                        let i = i as u32;
                        let l = i / lb;
                        *slot = if l < half {
                            ((l + half) * lb + i % lb) as u8
                        } else {
                            i as u8
                        };
                    }
                    let shifted = self.b.def(Expr::VShuffle {
                        a: acc,
                        b: acc,
                        lanes: Box::new(lanes),
                    });
                    acc = self.b.def(Expr::VLaneBin {
                        op: bin,
                        lane,
                        a: acc,
                        b: shifted,
                    });
                    span = half;
                }
                // Only lane 0 is architecturally defined; the rest is zero.
                let value = self.b.def(Expr::VExtract {
                    a: acc,
                    lane: 0,
                    lane_bytes: lb,
                    signed: false,
                    ty: if lb == 8 { Ty::I64 } else { Ty::I32 },
                });
                let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                let out = self.b.def(Expr::VReplace {
                    a: zero,
                    lane: 0,
                    lane_bytes: lb,
                    val: value,
                });
                self.b.write(GuestReg::V128(vd.0), out);
                Flow::Next
            }

            Inst::VSelect {
                kind,
                vd,
                vn,
                vm,
                q,
            } => {
                // All three are one bitselect; they differ only in which
                // operand supplies the mask and which the alternatives.
                let d = self.b.read(GuestReg::V128(vd.0));
                let n = self.b.read(GuestReg::V128(vn.0));
                let m = self.b.read(GuestReg::V128(vm.0));
                let (mask, a, b) = match kind {
                    VSelKind::Bsl => (d, n, m),
                    VSelKind::Bit => (m, n, d),
                    VSelKind::Bif => (m, d, n),
                };
                let r = self.b.def(Expr::VBitSelect { mask, a, b });
                let v = self.narrow_to_64(r, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            Inst::VThreeSame {
                op,
                arr,
                vd,
                vn,
                vm,
            } => self.three_same(op, arr, vd, vn, vm, pc),

            Inst::VTwoMisc { op, arr, vd, vn } => self.two_misc(op, arr, vd, vn, pc),

            Inst::VPermute {
                op,
                arr,
                vd,
                vn,
                vm,
            } => {
                let a = self.b.read(GuestReg::V128(vn.0));
                let b = self.b.read(GuestReg::V128(vm.0));
                let lanes = permute_pattern(op, arr);
                let r = self.b.def(Expr::VShuffle {
                    a,
                    b,
                    lanes: Box::new(lanes),
                });
                let v = self.narrow_to_64(r, arr.q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            Inst::VExt {
                vd,
                vn,
                vm,
                index,
                q,
            } => {
                let a = self.b.read(GuestReg::V128(vn.0));
                let b = self.b.read(GuestReg::V128(vm.0));
                // The 64-bit form windows over the low halves of both
                // registers, so the second source's bytes start at 8.
                let width = if q { 16u32 } else { 8 };
                let mut lanes = [0u8; 16];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let pos = index + i as u32;
                    *slot = if pos < width {
                        pos as u8
                    } else {
                        (16 + (pos - width)) as u8
                    };
                }
                let r = self.b.def(Expr::VShuffle {
                    a,
                    b,
                    lanes: Box::new(lanes),
                });
                let v = self.narrow_to_64(r, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            Inst::VTableLookup {
                vd,
                vn,
                regs,
                vm,
                q,
                is_tbx,
            } => {
                // A single-register table is exactly wasm's swizzle: indices
                // past the end produce zero. Wider tables need one swizzle
                // per register, biased so each sees 0..15, then merged.
                let idx = self.b.read(GuestReg::V128(vm.0));
                let mut acc: Option<Value> = None;
                for i in 0..regs {
                    let table = self.b.read(GuestReg::V128((vn.0 + i) % 32));
                    let bias = self.b.def(Expr::VConst(Box::new([i * 16; 16])));
                    let local = self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Sub,
                        lane: VLane::I8,
                        a: idx,
                        b: bias,
                    });
                    let part = self.b.def(Expr::VSwizzle {
                        a: table,
                        idx: local,
                    });
                    acc = Some(match acc {
                        None => part,
                        Some(prev) => self.b.def(Expr::VBin {
                            op: VBinOp::Or,
                            a: prev,
                            b: part,
                        }),
                    });
                }
                let mut r = acc.expect("at least one table register");
                if is_tbx {
                    // `tbx` keeps the destination's existing bytes where the
                    // index was out of range — exactly where swizzle gave 0.
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let limit = self.b.def(Expr::VConst(Box::new([regs * 16; 16])));
                    let in_range = self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::LtU,
                        lane: VLane::I8,
                        a: idx,
                        b: limit,
                    });
                    r = self.b.def(Expr::VBitSelect {
                        mask: in_range,
                        a: r,
                        b: old,
                    });
                }
                let v = self.narrow_to_64(r, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }

            Inst::VDupElement {
                vd,
                vn,
                lane_bytes,
                index,
                q,
                scalar,
            } => {
                let a = self.b.read(GuestReg::V128(vn.0));
                let mut lanes = [0u8; 16];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let byte = i as u32 % lane_bytes;
                    *slot = (index * lane_bytes + byte) as u8;
                }
                let r = self.b.def(Expr::VShuffle {
                    a,
                    b: a,
                    lanes: Box::new(lanes),
                });
                // The scalar form keeps only lane 0; `commit_vector` already
                // knows how to say that.
                self.commit_vector(
                    vd,
                    VArr {
                        lane_bytes,
                        q,
                        float: false,
                        scalar,
                    },
                    r,
                );
                Flow::Next
            }

            Inst::VInsElement {
                vd,
                vn,
                lane_bytes,
                dst_index,
                src_index,
            } => {
                // Insertion leaves every other lane of the destination alone,
                // so it is one shuffle drawing from both registers.
                let dst = self.b.read(GuestReg::V128(vd.0));
                let src = self.b.read(GuestReg::V128(vn.0));
                let mut lanes = [0u8; 16];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let i = i as u32;
                    let lane = i / lane_bytes;
                    *slot = if lane == dst_index {
                        (16 + src_index * lane_bytes + (i % lane_bytes)) as u8
                    } else {
                        i as u8
                    };
                }
                let r = self.b.def(Expr::VShuffle {
                    a: dst,
                    b: src,
                    lanes: Box::new(lanes),
                });
                self.b.write(GuestReg::V128(vd.0), r);
                Flow::Next
            }

            Inst::VShiftImm {
                op,
                arr,
                vd,
                vn,
                shift,
            } => self.shift_imm(op, arr, vd, vn, shift, pc),

            Inst::FpCondCompare {
                sz,
                vn,
                vm,
                nzcv,
                cond,
                signalling: _,
            } => {
                let double = sz.is_double();
                let a = self.rd_fp(vn, sz);
                let b = self.rd_fp(vm, sz);
                let (n, z, c, v) = self.fp_compare_flags(double, a, b);
                let taken = self.eval_cond(cond);
                // Read the condition first, then forget: this writes all four
                // flags from a floating-point comparison or a literal.
                self.flags = FlagSource::Unknown;
                let alt = [
                    (nzcv >> 3) as u32 & 1,
                    (nzcv >> 2) as u32 & 1,
                    (nzcv >> 1) as u32 & 1,
                    nzcv as u32 & 1,
                ];
                for (i, (flag, reg)) in [
                    (n, GuestReg::N),
                    (z, GuestReg::Z),
                    (c, GuestReg::C),
                    (v, GuestReg::V),
                ]
                .into_iter()
                .enumerate()
                {
                    let fallback = self.b.const_i32(alt[i]);
                    let sel = self.b.select(taken, flag, fallback, Ty::I32);
                    self.b.write(reg, sel);
                }
                Flow::Next
            }
            Inst::VLogical { op, vd, vn, vm, q } => {
                let a = self.b.read(GuestReg::V128(vn.0));
                let b = self.b.read(GuestReg::V128(vm.0));
                let (vop, lhs, rhs) = match op {
                    VLogOp::And => (VBinOp::And, a, b),
                    VLogOp::Orr => (VBinOp::Or, a, b),
                    VLogOp::Eor => (VBinOp::Xor, a, b),
                    VLogOp::Bic => (VBinOp::AndNot, a, b),
                    // `orn a, b` is `not(andnot(not(a), ... ))`; expressed
                    // directly it is `a | !b`, which wasm has no single
                    // instruction for, so build it from andnot and xor.
                    VLogOp::Orn => {
                        let ones = self.b.def(Expr::VConst(Box::new([0xff; 16])));
                        let notb = self.b.def(Expr::VBin {
                            op: VBinOp::Xor,
                            a: b,
                            b: ones,
                        });
                        (VBinOp::Or, a, notb)
                    }
                };
                let r = self.b.def(Expr::VBin {
                    op: vop,
                    a: lhs,
                    b: rhs,
                });
                let v = self.narrow_to_64(r, q);
                self.b.write(GuestReg::V128(vd.0), v);
                Flow::Next
            }
        }
    }

    // -- scalar floating point ---------------------------------------------

    /// The lane type GIR should use for an arrangement.
    fn lane_of(arr: VArr) -> VLane {
        match (arr.lane_bytes, arr.float) {
            (1, _) => VLane::I8,
            (2, _) => VLane::I16,
            (4, true) => VLane::F32,
            (4, _) => VLane::I32,
            (8, true) => VLane::F64,
            _ => VLane::I64,
        }
    }

    /// Lane-wise multiply keeping the low half of each product, as `mul`
    /// and `mla` want.
    ///
    /// wasm has no `i8x16.mul`, so byte lanes are multiplied as halfwords:
    /// the low byte of a 16-bit product depends only on the low bytes of its
    /// factors, which makes `i16x8.mul(a, b)` right in the even bytes. The odd
    /// bytes come from `(a >> 8) * (b & 0xff00)`, whose low byte is zero and
    /// whose high byte is the odd product mod 256.
    fn vmul(&mut self, lane: VLane, a: Value, b: Value) -> Value {
        if lane != VLane::I8 {
            return self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Mul,
                lane,
                a,
                b,
            });
        }
        let even = self.b.def(Expr::VLaneBin {
            op: VLaneBinOp::Mul,
            lane: VLane::I16,
            a,
            b,
        });
        let eight = self.b.const_i32(8);
        let a_hi = self.b.def(Expr::VShift {
            op: VShiftOp::ShrU,
            lane: VLane::I16,
            a,
            amount: eight,
        });
        let mut odd_bytes = [0u8; 16];
        for i in (1..16).step_by(2) {
            odd_bytes[i] = 0xff;
        }
        let hi_mask = self.b.def(Expr::VConst(Box::new(odd_bytes)));
        let b_hi = self.b.def(Expr::VBin {
            op: VBinOp::And,
            a: b,
            b: hi_mask,
        });
        let odd = self.b.def(Expr::VLaneBin {
            op: VLaneBinOp::Mul,
            lane: VLane::I16,
            a: a_hi,
            b: b_hi,
        });
        // `even & ~0xff00`, then merge in the odd bytes.
        let even_lo = self.b.def(Expr::VBin {
            op: VBinOp::AndNot,
            a: even,
            b: hi_mask,
        });
        self.b.def(Expr::VBin {
            op: VBinOp::Or,
            a: even_lo,
            b: odd,
        })
    }

    /// Restrict a result to the lanes the arrangement actually writes.
    ///
    /// A 64-bit form zeroes the register's upper half; a scalar form zeroes
    /// everything above its single element. Both are a mask, which is why the
    /// scalar instructions can share the vector lowerings outright.
    fn commit_vector(&mut self, vd: gex_isa::VReg, arr: VArr, value: Value) {
        let v = if arr.scalar {
            let mut mask = [0u8; 16];
            mask[..arr.lane_bytes as usize].fill(0xff);
            let m = self.b.def(Expr::VConst(Box::new(mask)));
            self.b.def(Expr::VBin {
                op: VBinOp::And,
                a: value,
                b: m,
            })
        } else {
            self.narrow_to_64(value, arr.q)
        };
        self.b.write(GuestReg::V128(vd.0), v);
    }

    fn three_same(
        &mut self,
        op: VOp3,
        arr: VArr,
        vd: gex_isa::VReg,
        vn: gex_isa::VReg,
        vm: gex_isa::VReg,
        pc: u64,
    ) -> Flow {
        let b = self.b.read(GuestReg::V128(vm.0));
        self.three_same_with(op, arr, vd, vn, b, pc)
    }

    /// `three_same`, but with the second operand already in hand -- which is
    /// what lets the by-element forms share every lowering below.
    fn three_same_with(
        &mut self,
        op: VOp3,
        arr: VArr,
        vd: gex_isa::VReg,
        vn: gex_isa::VReg,
        b: Value,
        pc: u64,
    ) -> Flow {
        let lane = Self::lane_of(arr);
        let a = self.b.read(GuestReg::V128(vn.0));

        // wasm saturates byte and halfword lanes only; wider ones are done here.
        if matches!(op, VOp3::SqAdd | VOp3::UqAdd | VOp3::SqSub | VOp3::UqSub)
            && matches!(lane, VLane::I32 | VLane::I64)
        {
            let r = self.sat_add_sub_wide(op, lane, a, b);
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }
        if matches!(op, VOp3::SqDmulh | VOp3::SqRdmulh) {
            let r = self.sat_doubling_mulh(op == VOp3::SqRdmulh, arr, a, b);
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        let simple = |op| Some(op);
        let bin: Option<VLaneBinOp> = match op {
            VOp3::Add | VOp3::FAdd => simple(VLaneBinOp::Add),
            VOp3::Sub | VOp3::FSub => simple(VLaneBinOp::Sub),
            // `fmulx` shares this lowering: it differs only for zero times
            // infinity, which gex does not model. See README's fidelity notes.
            // Byte lanes have no wasm multiply; see `vmul`.
            VOp3::Mul if lane == VLane::I8 => None,
            VOp3::Mul | VOp3::FMul | VOp3::FMulx => simple(VLaneBinOp::Mul),
            VOp3::FDiv => simple(VLaneBinOp::Div),
            VOp3::SMax => simple(VLaneBinOp::MaxS),
            VOp3::SMin => simple(VLaneBinOp::MinS),
            VOp3::UMax => simple(VLaneBinOp::MaxU),
            VOp3::UMin => simple(VLaneBinOp::MinU),
            // NEON's FMAX/FMIN propagate NaN, which is what wasm's
            // f32x4.max/min do; the "number" forms differ only on NaN.
            VOp3::FMax | VOp3::FMaxNm => simple(VLaneBinOp::MaxS),
            VOp3::FMin | VOp3::FMinNm => simple(VLaneBinOp::MinS),
            VOp3::SqAdd => simple(VLaneBinOp::AddSatS),
            VOp3::UqAdd => simple(VLaneBinOp::AddSatU),
            VOp3::SqSub => simple(VLaneBinOp::SubSatS),
            VOp3::UqSub => simple(VLaneBinOp::SubSatU),
            VOp3::UrHAdd => simple(VLaneBinOp::AvgrU),
            VOp3::CmEq | VOp3::FCmEq => simple(VLaneBinOp::Eq),
            VOp3::CmGt | VOp3::FCmGt => simple(VLaneBinOp::GtS),
            VOp3::CmGe | VOp3::FCmGe => simple(VLaneBinOp::GeS),
            // Only up to 32-bit lanes: wasm has no `i64x2.gt_u`. The
            // 64-bit cases are rewritten below rather than refused.
            VOp3::CmHi if lane != VLane::I64 => simple(VLaneBinOp::GtU),
            VOp3::CmHs if lane != VLane::I64 => simple(VLaneBinOp::GeU),
            _ => None,
        };
        if let Some(bin) = bin {
            // The floating-point max/min share an enum entry with the signed
            // integer ones; the lane type disambiguates them in the backend.
            let r = self.b.def(Expr::VLaneBin {
                op: bin,
                lane,
                a,
                b,
            });
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        // The Newton refinement steps: `frecps(a,b) = 2 - a*b` and
        // `frsqrts(a,b) = (3 - a*b)/2`, per lane. gex's `frecpe`/`frsqrte`
        // already return exact values so these steps just reproduce them, but
        // implementing them faithfully keeps any independent use correct.
        if matches!(op, VOp3::FRecps | VOp3::FRSqrts) {
            let splat = |val: f64| -> Box<[u8; 16]> {
                let mut c = [0u8; 16];
                if arr.lane_bytes == 8 {
                    let bytes = val.to_le_bytes();
                    for i in [0usize, 8] {
                        c[i..i + 8].copy_from_slice(&bytes);
                    }
                } else {
                    let bytes = (val as f32).to_le_bytes();
                    for i in [0usize, 4, 8, 12] {
                        c[i..i + 4].copy_from_slice(&bytes);
                    }
                }
                Box::new(c)
            };
            let prod = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Mul,
                lane,
                a,
                b,
            });
            let top = self
                .b
                .def(Expr::VConst(splat(if matches!(op, VOp3::FRSqrts) { 3.0 } else { 2.0 })));
            let diff = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Sub,
                lane,
                a: top,
                b: prod,
            });
            let r = if matches!(op, VOp3::FRSqrts) {
                let half = self.b.def(Expr::VConst(splat(0.5)));
                self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Mul,
                    lane,
                    a: diff,
                    b: half,
                })
            } else {
                diff
            };
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        // The halving add/sub have no single wasm op (only the rounding
        // *unsigned* add, `avgr_u` = `UrHAdd` above). The SWAR identities hold
        // per lane, verified exhaustively over all 8-bit inputs both signs:
        //   shadd/uhadd  = (a & b) + ((a ^ b) >> 1)
        //   srhadd       = (a | b) - ((a ^ b) >> 1)
        //   shsub/uhsub  = ((a ^ b) >> 1) - (~a & b)   [~a & b = (a & b) ^ b]
        // with an arithmetic shift for the signed forms, logical for unsigned.
        if matches!(
            op,
            VOp3::SHAdd | VOp3::UHAdd | VOp3::SrHAdd | VOp3::SHSub | VOp3::UHSub
        ) {
            let signed = matches!(op, VOp3::SHAdd | VOp3::SrHAdd | VOp3::SHSub);
            let rounding = matches!(op, VOp3::SrHAdd);
            let sub = matches!(op, VOp3::SHSub | VOp3::UHSub);
            let xor = self.b.def(Expr::VBin {
                op: VBinOp::Xor,
                a,
                b,
            });
            let one = self.b.const_i32(1);
            let half = self.b.def(Expr::VShift {
                op: if signed { VShiftOp::ShrS } else { VShiftOp::ShrU },
                lane,
                a: xor,
                amount: one,
            });
            let r = if sub {
                // `~a & b`, then `((a ^ b) >> 1) - (~a & b)`.
                let ab = self.b.def(Expr::VBin {
                    op: VBinOp::And,
                    a,
                    b,
                });
                let nab = self.b.def(Expr::VBin {
                    op: VBinOp::Xor,
                    a: ab,
                    b,
                });
                self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Sub,
                    lane,
                    a: half,
                    b: nab,
                })
            } else {
                let base = self.b.def(Expr::VBin {
                    op: if rounding { VBinOp::Or } else { VBinOp::And },
                    a,
                    b,
                });
                self.b.def(Expr::VLaneBin {
                    op: if rounding { VLaneBinOp::Sub } else { VLaneBinOp::Add },
                    lane,
                    a: base,
                    b: half,
                })
            };
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        // `cmhi`/`cmhs` on 64-bit lanes. wasm's i64x2 compares are signed
        // only, so bias into signed order: `a >u b` == `(a^min) >s (b^min)`.
        if matches!(op, VOp3::CmHi | VOp3::CmHs) && lane == VLane::I64 {
            let mut bytes = [0u8; 16];
            bytes[7] = 0x80;
            bytes[15] = 0x80;
            let sign = self.b.def(Expr::VConst(Box::new(bytes)));
            let a = self.b.def(Expr::VBin {
                op: VBinOp::Xor,
                a,
                b: sign,
            });
            let b = self.b.def(Expr::VBin {
                op: VBinOp::Xor,
                a: b,
                b: sign,
            });
            let r = self.b.def(Expr::VLaneBin {
                op: if op == VOp3::CmHi {
                    VLaneBinOp::GtS
                } else {
                    VLaneBinOp::GeS
                },
                lane,
                a,
                b,
            });
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        match op {
            VOp3::Mul => {
                let r = self.vmul(lane, a, b);
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // Multiply-accumulate reads the destination as a third operand.
            VOp3::Mla | VOp3::Mls | VOp3::FMla | VOp3::FMls => {
                let acc = self.b.read(GuestReg::V128(vd.0));
                let prod = self.vmul(lane, a, b);
                let add = matches!(op, VOp3::Mla | VOp3::FMla);
                // The floating-point forms are fused on hardware; wasm has no
                // vector FMA, so the product is rounded first.
                let r = self.b.def(Expr::VLaneBin {
                    op: if add {
                        VLaneBinOp::Add
                    } else {
                        VLaneBinOp::Sub
                    },
                    lane,
                    a: acc,
                    b: prod,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // `cmtst` sets lanes where the bitwise AND is non-zero.
            VOp3::CmTst => {
                let and = self.b.def(Expr::VBin {
                    op: VBinOp::And,
                    a,
                    b,
                });
                let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                let r = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Ne,
                    lane,
                    a: and,
                    b: zero,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // `|a - b|`: for floating point, a subtract and an absolute value.
            VOp3::FAcGe | VOp3::FAcGt => {
                let x = self.b.def(Expr::VLaneUn { op: VLaneUnOp::Abs, lane, a });
                let y = self.b.def(Expr::VLaneUn { op: VLaneUnOp::Abs, lane, a: b });
                let cmp = if op == VOp3::FAcGe { VLaneBinOp::GeS } else { VLaneBinOp::GtS };
                let r = self.b.def(Expr::VLaneBin { op: cmp, lane, a: x, b: y });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            VOp3::FAbd => {
                let diff = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Sub,
                    lane,
                    a,
                    b,
                });
                let r = self.b.def(Expr::VLaneUn {
                    op: VLaneUnOp::Abs,
                    lane,
                    a: diff,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // max - min, which cannot overflow the way subtract-then-abs can.
            VOp3::SAbd | VOp3::UAbd => {
                let signed = op == VOp3::SAbd;
                let hi = self.b.def(Expr::VLaneBin {
                    op: if signed {
                        VLaneBinOp::MaxS
                    } else {
                        VLaneBinOp::MaxU
                    },
                    lane,
                    a,
                    b,
                });
                let lo = self.b.def(Expr::VLaneBin {
                    op: if signed {
                        VLaneBinOp::MinS
                    } else {
                        VLaneBinOp::MinU
                    },
                    lane,
                    a,
                    b,
                });
                let r = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Sub,
                    lane,
                    a: hi,
                    b: lo,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // Pairwise: deinterleave the concatenation, then combine.
            VOp3::AddP
            | VOp3::FAddP
            | VOp3::FMaxP
            | VOp3::FMinP
            | VOp3::SMaxP
            | VOp3::UMaxP
            | VOp3::SMinP
            | VOp3::UMinP => {
                let (evens, odds) = pairwise_patterns(arr);
                let e = self.b.def(Expr::VShuffle {
                    a,
                    b,
                    lanes: Box::new(evens),
                });
                let o = self.b.def(Expr::VShuffle {
                    a,
                    b,
                    lanes: Box::new(odds),
                });
                let pair = match op {
                    // On float lanes these are wasm's f32x4.max/min, as `FMax` above.
                    VOp3::SMaxP | VOp3::FMaxP => VLaneBinOp::MaxS,
                    VOp3::UMaxP => VLaneBinOp::MaxU,
                    VOp3::SMinP | VOp3::FMinP => VLaneBinOp::MinS,
                    VOp3::UMinP => VLaneBinOp::MinU,
                    _ => VLaneBinOp::Add,
                };
                let r = self.b.def(Expr::VLaneBin {
                    op: pair,
                    lane,
                    a: e,
                    b: o,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // Register-variable shifts, done per lane since wasm has no
            // per-lane count. The count is the low byte of the second
            // operand's lane, signed: negative shifts right, and a count at or
            // past the lane width shifts everything out (zero, or the
            // replicated sign bit) rather than wrapping as wasm would.
            VOp3::UShl | VOp3::SShl => {
                let signed = op == VOp3::SShl;
                let bits = arr.lane_bytes * 8;
                let ty = if arr.lane_bytes == 8 { Ty::I64 } else { Ty::I32 };
                let lanes = if arr.scalar {
                    1
                } else {
                    (if arr.q { 16 } else { 8 }) / arr.lane_bytes
                };
                let zero = self.konst(ty, 0);
                let limit = self.konst(ty, bits as u64);
                let mut out = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                for i in 0..lanes {
                    let val = self.b.def(Expr::VExtract {
                        a,
                        lane: i,
                        lane_bytes: arr.lane_bytes,
                        signed,
                        ty,
                    });
                    // The count is the low byte of the lane, signed. Taking
                    // it as a byte is the whole rule: the rest of the lane is
                    // ignored. A byte extract yields an i32 whatever the lane
                    // width, so a 64-bit lane needs it widened.
                    let n = self.b.def(Expr::VExtract {
                        a: b,
                        lane: i * arr.lane_bytes,
                        lane_bytes: 1,
                        signed: true,
                        ty: Ty::I32,
                    });
                    let n = if ty == Ty::I64 {
                        self.b.def(Expr::Extend { a: n, signed: true })
                    } else {
                        n
                    };
                    let neg = self.b.bin(BinOp::Sub, ty, zero, n);
                    let left = self.b.bin(BinOp::Shl, ty, val, n);
                    let right = self.b.bin(
                        if signed { BinOp::ShrS } else { BinOp::ShrU },
                        ty,
                        val,
                        neg,
                    );
                    // Out of range: everything is gone. Left and unsigned
                    // right give zero; a signed right gives the sign bit,
                    // which is what shifting by width-1 leaves behind.
                    let all = self.konst(ty, (bits - 1) as u64);
                    let saturated = if signed {
                        self.b.bin(BinOp::ShrS, ty, val, all)
                    } else {
                        zero
                    };
                    let far_left = self.b.cmp(CmpOp::GeS, ty, n, limit);
                    let left = self.b.def(Expr::Select {
                        cond: far_left,
                        a: zero,
                        b: left,
                        ty,
                    });
                    let far_right = self.b.cmp(CmpOp::GeS, ty, neg, limit);
                    let right = self.b.def(Expr::Select {
                        cond: far_right,
                        a: saturated,
                        b: right,
                        ty,
                    });
                    let is_left = self.b.cmp(CmpOp::GeS, ty, n, zero);
                    let lane_out = self.b.def(Expr::Select {
                        cond: is_left,
                        a: left,
                        b: right,
                        ty,
                    });
                    out = self.b.def(Expr::VReplace {
                        a: out,
                        lane: i,
                        lane_bytes: arr.lane_bytes,
                        val: lane_out,
                    });
                }
                self.commit_vector(vd, arr, out);
                Flow::Next
            }
            _ => Flow::End(Term::Illegal { raw: self.raw, pc }),
        }
    }

    fn two_misc(
        &mut self,
        op: VOp2,
        arr: VArr,
        vd: gex_isa::VReg,
        vn: gex_isa::VReg,
        pc: u64,
    ) -> Flow {
        let lane = Self::lane_of(arr);
        let a = self.b.read(GuestReg::V128(vn.0));

        let un: Option<VLaneUnOp> = match op {
            VOp2::Neg | VOp2::FNeg => Some(VLaneUnOp::Neg),
            VOp2::Abs | VOp2::FAbs => Some(VLaneUnOp::Abs),
            VOp2::FSqrt => Some(VLaneUnOp::Sqrt),
            VOp2::FrintP => Some(VLaneUnOp::Ceil),
            VOp2::FrintM => Some(VLaneUnOp::Floor),
            VOp2::FrintZ => Some(VLaneUnOp::Trunc),
            // Ties to even for FRINTN, and for FRINTX/FRINTI (the FPCR mode,
            // not tracked). FRINTA is built below.
            VOp2::FrintN | VOp2::FrintX | VOp2::FrintI => Some(VLaneUnOp::Nearest),
            _ => None,
        };
        if op == VOp2::FrintA && matches!(lane, VLane::F32 | VLane::F64) {
            let r = self.round_away_lanes(lane, a);
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }
        if let Some(un) = un {
            let r = self.b.def(Expr::VLaneUn { op: un, lane, a });
            self.commit_vector(vd, arr, r);
            return Flow::Next;
        }

        match op {
            // `frecpe`/`frsqrte` are estimates; gex computes the exact value.
            // `frecpe x = 1/x`, `frsqrte x = 1/sqrt(x)`. Code that refines with
            // `frecps`/`frsqrts` converges the same, and code that uses the
            // estimate directly gets better precision than the hardware.
            VOp2::FRecpe | VOp2::FRSqrtE => {
                let mut one = [0u8; 16];
                if arr.lane_bytes == 8 {
                    // 1.0f64 = 0x3ff0000000000000, little-endian, per 8-byte lane.
                    for i in [0usize, 8] {
                        one[i + 6] = 0xf0;
                        one[i + 7] = 0x3f;
                    }
                } else {
                    // 1.0f32 = 0x3f800000, little-endian, per 4-byte lane.
                    for i in [0usize, 4, 8, 12] {
                        one[i + 2] = 0x80;
                        one[i + 3] = 0x3f;
                    }
                }
                let ones = self.b.def(Expr::VConst(Box::new(one)));
                let denom = if matches!(op, VOp2::FRSqrtE) {
                    self.b.def(Expr::VLaneUn { op: VLaneUnOp::Sqrt, lane, a })
                } else {
                    a
                };
                let r = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Div,
                    lane,
                    a: ones,
                    b: denom,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // Reverse the bits of every byte: look each nibble up reversed
            // and swap the halves. Each `swizzle` index is below 16, so no
            // lane is zeroed by the lookup.
            VOp2::Rbit => {
                let rev4 = |n: u8| (n & 1) << 3 | (n & 2) << 1 | (n & 4) >> 1 | (n & 8) >> 3;
                let mut low_table = [0u8; 16];
                let mut high_table = [0u8; 16];
                for n in 0..16u8 {
                    // The low nibble, reversed, becomes the high one.
                    low_table[n as usize] = rev4(n) << 4;
                    high_table[n as usize] = rev4(n);
                }
                let nibble = self.b.def(Expr::VConst(Box::new([0x0f; 16])));
                let lo = self.b.def(Expr::VBin {
                    op: VBinOp::And,
                    a,
                    b: nibble,
                });
                let four = self.b.const_i32(4);
                let hi = self.b.def(Expr::VShift {
                    op: VShiftOp::ShrU,
                    lane: VLane::I8,
                    a,
                    amount: four,
                });
                let lt = self.b.def(Expr::VConst(Box::new(low_table)));
                let ht = self.b.def(Expr::VConst(Box::new(high_table)));
                let from_lo = self.b.def(Expr::VSwizzle { a: lt, idx: lo });
                let from_hi = self.b.def(Expr::VSwizzle { a: ht, idx: hi });
                let r = self.b.def(Expr::VBin {
                    op: VBinOp::Or,
                    a: from_lo,
                    b: from_hi,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            VOp2::Not => {
                let ones = self.b.def(Expr::VConst(Box::new([0xff; 16])));
                let r = self.b.def(Expr::VBin {
                    op: VBinOp::Xor,
                    a,
                    b: ones,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // `saddlp`/`uaddlp` and accumulating `sadalp`/`uadalp`: pairwise
            // add into lanes of twice the width. Byte and halfword steps are
            // wasm builtins; the word step is two shuffles and an add.
            VOp2::Saddlp | VOp2::Uaddlp | VOp2::Sadalp | VOp2::Uadalp => {
                let signed = matches!(op, VOp2::Saddlp | VOp2::Sadalp);
                let accumulate = matches!(op, VOp2::Sadalp | VOp2::Uadalp);
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    ..arr
                };
                let summed = match arr.lane_bytes {
                    1 => self.b.def(Expr::VUn {
                        op: if signed {
                            VUnOp::ExtAddPairwiseS8
                        } else {
                            VUnOp::ExtAddPairwiseU8
                        },
                        a,
                    }),
                    2 => self.b.def(Expr::VUn {
                        op: if signed {
                            VUnOp::ExtAddPairwiseS16
                        } else {
                            VUnOp::ExtAddPairwiseU16
                        },
                        a,
                    }),
                    4 => {
                        // Signed would need the sign replicated into the
                        // upper half rather than zeroes, which this shuffle
                        // cannot do. `saddlp` at this width has not been
                        // seen; leaving it precise is better than guessing.
                        if signed {
                            return Flow::End(Term::Illegal { raw: self.raw, pc });
                        }
                        let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                        let mut ev = [16u8; 16];
                        let mut od = [16u8; 16];
                        for half in 0..2usize {
                            for k in 0..4usize {
                                ev[half * 8 + k] = (half * 8 + k) as u8;
                                od[half * 8 + k] = (half * 8 + 4 + k) as u8;
                            }
                        }
                        let evens = self.b.def(Expr::VShuffle {
                            lanes: Box::new(ev),
                            a,
                            b: zero,
                        });
                        let odds = self.b.def(Expr::VShuffle {
                            lanes: Box::new(od),
                            a,
                            b: zero,
                        });
                        self.b.def(Expr::VLaneBin {
                            op: VLaneBinOp::Add,
                            lane: Self::lane_of(wide),
                            a: evens,
                            b: odds,
                        })
                    }
                    _ => return Flow::End(Term::Illegal { raw: self.raw, pc }),
                };
                let r = if accumulate {
                    let old = self.b.read(GuestReg::V128(vd.0));
                    self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane: Self::lane_of(wide),
                        a: old,
                        b: summed,
                    })
                } else {
                    summed
                };
                self.commit_vector(vd, wide, r);
                Flow::Next
            }
            VOp2::Cnt => {
                let r = self.b.def(Expr::VUn {
                    op: VUnOp::PopcntBytes,
                    a,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // No vector CLZ in wasm. Smear every set bit down its lane, then
            // popcount: `w - popcount` is the leading-zero count. The byte
            // popcount is summed up to the lane width with pairwise adds.
            VOp2::Clz => {
                let bits = arr.lane_bytes * 8;
                let lane = Self::lane_of(arr);
                let mut x = a;
                let mut sh = 1;
                while sh < bits {
                    let amount = self.b.const_i32(sh);
                    let shifted = self.b.def(Expr::VShift {
                        op: VShiftOp::ShrU,
                        lane,
                        a: x,
                        amount,
                    });
                    x = self.b.def(Expr::VBin {
                        op: VBinOp::Or,
                        a: x,
                        b: shifted,
                    });
                    sh *= 2;
                }
                let mut counted = self.b.def(Expr::VUn {
                    op: VUnOp::PopcntBytes,
                    a: x,
                });
                if arr.lane_bytes >= 2 {
                    counted = self.b.def(Expr::VUn {
                        op: VUnOp::ExtAddPairwiseU8,
                        a: counted,
                    });
                }
                if arr.lane_bytes >= 4 {
                    counted = self.b.def(Expr::VUn {
                        op: VUnOp::ExtAddPairwiseU16,
                        a: counted,
                    });
                }
                let mut width = [0u8; 16];
                for i in 0..(16 / arr.lane_bytes) {
                    // Little-endian, so the count goes in the lane's low byte
                    // and the width is never more than 64.
                    width[(i * arr.lane_bytes) as usize] = bits as u8;
                }
                let w = self.b.def(Expr::VConst(Box::new(width)));
                let r = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Sub,
                    lane,
                    a: w,
                    b: counted,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // The reverse family is a fixed byte permutation.
            VOp2::Rev16 | VOp2::Rev32 | VOp2::Rev64 => {
                let group = match op {
                    VOp2::Rev16 => 2u32,
                    VOp2::Rev32 => 4,
                    _ => 8,
                };
                // An element as wide as its group would underflow the
                // arithmetic below; the decoder rejects it, rechecked here
                // because a wrapped shuffle index yields a module that will
                // not load.
                if arr.lane_bytes >= group {
                    return Flow::End(Term::Illegal { raw: self.raw, pc });
                }
                let mut lanes = [0u8; 16];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let i = i as u32;
                    let base = i - (i % group);
                    let within = i % group;
                    // Reverse whole elements inside each group.
                    let elem = within / arr.lane_bytes;
                    let n_elems = group / arr.lane_bytes;
                    let src_elem = n_elems - 1 - elem;
                    *slot = (base + src_elem * arr.lane_bytes + (within % arr.lane_bytes)) as u8;
                }
                let r = self.b.def(Expr::VShuffle {
                    a,
                    b: a,
                    lanes: Box::new(lanes),
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            VOp2::CmEqZero
            | VOp2::CmGtZero
            | VOp2::CmGeZero
            | VOp2::CmLtZero
            | VOp2::CmLeZero
            | VOp2::FCmEqZero
            | VOp2::FCmGtZero
            | VOp2::FCmGeZero
            | VOp2::FCmLtZero
            | VOp2::FCmLeZero => {
                let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                let cmp = match op {
                    VOp2::CmEqZero | VOp2::FCmEqZero => VLaneBinOp::Eq,
                    VOp2::CmGtZero | VOp2::FCmGtZero => VLaneBinOp::GtS,
                    VOp2::CmGeZero | VOp2::FCmGeZero => VLaneBinOp::GeS,
                    VOp2::CmLtZero | VOp2::FCmLtZero => VLaneBinOp::LtS,
                    _ => VLaneBinOp::LeS,
                };
                let r = self.b.def(Expr::VLaneBin {
                    op: cmp,
                    lane,
                    a,
                    b: zero,
                });
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            VOp2::FcvtZs | VOp2::FcvtZu => {
                let signed = op == VOp2::FcvtZs;
                let r = self.lane_convert(arr, a, FpConvert::FpToInt { signed }, signed);
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // The other rounding modes round first, then truncate; the
            // rounded value is already integral so nothing is lost.
            VOp2::FcvtNs
            | VOp2::FcvtNu
            | VOp2::FcvtAs
            | VOp2::FcvtAu
            | VOp2::FcvtPs
            | VOp2::FcvtPu
            | VOp2::FcvtMs
            | VOp2::FcvtMu => {
                let round = match op {
                    VOp2::FcvtPs | VOp2::FcvtPu => VLaneUnOp::Ceil,
                    VOp2::FcvtMs | VOp2::FcvtMu => VLaneUnOp::Floor,
                    _ => VLaneUnOp::Nearest,
                };
                let signed = matches!(
                    op,
                    VOp2::FcvtNs | VOp2::FcvtAs | VOp2::FcvtPs | VOp2::FcvtMs
                );
                let rounded = self.b.def(Expr::VLaneUn { op: round, lane, a });
                let r = self.lane_convert(arr, rounded, FpConvert::FpToInt { signed }, signed);
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            VOp2::Scvtf | VOp2::Ucvtf => {
                let signed = op == VOp2::Scvtf;
                let r = self.lane_convert(arr, a, FpConvert::IntToFp { signed }, signed);
                self.commit_vector(vd, arr, r);
                Flow::Next
            }
            // Widen (either extension: the extended bits are shifted out)
            // and shift left by the source lane width; `shll2` reads the
            // upper half.
            VOp2::Shll => {
                let widened = self.b.def(Expr::VCvt {
                    op: if arr.q {
                        VCvtOp::ExtendHigh { signed: false }
                    } else {
                        VCvtOp::ExtendLow { signed: false }
                    },
                    from: lane,
                    a,
                    b: None,
                });
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                let amount = self.b.const_i32(arr.lane_bytes * 8);
                let r = self.b.def(Expr::VShift {
                    op: VShiftOp::Shl,
                    lane: Self::lane_of(wide),
                    a: widened,
                    amount,
                });
                self.b.write(GuestReg::V128(vd.0), r);
                Flow::Next
            }
            // Narrowing writes the low half of the destination, or the high
            // half for the `2` forms; both are byte permutations of the
            // truncated lanes.
            VOp2::Xtn => {
                let mut lanes = [0u8; 16];
                let half = arr.lane_bytes;
                for i in 0..8u32 {
                    let elem = i / half;
                    let within = i % half;
                    lanes[i as usize] = (elem * half * 2 + within) as u8;
                }
                let packed = self.b.def(Expr::VShuffle {
                    a,
                    b: a,
                    lanes: Box::new(lanes),
                });
                if arr.q {
                    // `xtn2` keeps the destination's low half.
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let mut merge = [0u8; 16];
                    for i in 0..8 {
                        merge[i] = i as u8;
                        merge[i + 8] = (16 + i) as u8;
                    }
                    let r = self.b.def(Expr::VShuffle {
                        a: old,
                        b: packed,
                        lanes: Box::new(merge),
                    });
                    self.b.write(GuestReg::V128(vd.0), r);
                } else {
                    let v = self.narrow_to_64(packed, false);
                    self.b.write(GuestReg::V128(vd.0), v);
                }
                Flow::Next
            }
            VOp2::SqXtn | VOp2::UqXtn | VOp2::SqXtun => {
                self.sat_narrow(op, arr, vd, a)
            }
            _ => Flow::End(Term::Illegal { raw: self.raw, pc }),
        }
    }

    /// `sqxtn`, `uqxtn` and `sqxtun` on `a`, whose lanes are twice `arr`'s: the
    /// saturating narrow into `vd`'s low half, its upper half for the `2` forms, or
    /// its single element for the scalar ones. The saturating narrowing shifts
    /// share it after their shift.
    fn sat_narrow(&mut self, op: VOp2, arr: VArr, vd: gex_isa::VReg, a: Value) -> Flow {
        let wide = VArr {
            lane_bytes: arr.lane_bytes * 2,
            q: true,
            float: false,
            scalar: false,
        };
        // WebAssembly has signed->signed and unsigned->unsigned
        // saturating narrows. SQXTUN's signed->unsigned spelling is
        // the latter after signed-clamping negative lanes to zero.
        let source = if op == VOp2::SqXtun && arr.lane_bytes != 4 {
            let zero = self.b.def(Expr::VConst(Box::new([0; 16])));
            self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::MaxS,
                lane: Self::lane_of(wide),
                a,
                b: zero,
            })
        } else if op == VOp2::UqXtn && arr.lane_bytes != 4 {
            // wasm's unsigned narrow reads its source as signed, so
            // an unsigned source is clamped to the narrow maximum
            // first; after that both readings agree.
            let max = if arr.lane_bytes == 1 { [0xff, 0] } else { [0xff, 0xff] };
            let mut bytes = [0u8; 16];
            let w = (arr.lane_bytes * 2) as usize;
            for l in 0..16 / w {
                bytes[l * w] = max[0];
                bytes[l * w + 1] = max[1];
            }
            let limit = self.b.def(Expr::VConst(Box::new(bytes)));
            self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::MinU,
                lane: Self::lane_of(wide),
                a,
                b: limit,
            })
        } else {
            a
        };
        let packed = if arr.lane_bytes == 4 {
            self.narrow_sat_i64(op, a)
        } else {
            self.b.def(Expr::VCvt {
                op: VCvtOp::NarrowSat {
                    signed: op == VOp2::SqXtn,
                },
                from: Self::lane_of(wide),
                a: source,
                b: Some(source),
            })
        };
        if arr.scalar {
            // The scalar forms write one element and zero the rest.
            let mut mask = [0u8; 16];
            mask[..arr.lane_bytes as usize].fill(0xff);
            let m = self.b.def(Expr::VConst(Box::new(mask)));
            let r = self.b.def(Expr::VBin {
                op: VBinOp::And,
                a: packed,
                b: m,
            });
            self.b.write(GuestReg::V128(vd.0), r);
            return Flow::Next;
        }
        if arr.q {
            let old = self.b.read(GuestReg::V128(vd.0));
            let mut merge = [0u8; 16];
            for i in 0..8 {
                merge[i] = i as u8;
                merge[i + 8] = (16 + i) as u8;
            }
            let r = self.b.def(Expr::VShuffle {
                a: old,
                b: packed,
                lanes: Box::new(merge),
            });
            self.b.write(GuestReg::V128(vd.0), r);
        } else {
            let v = self.narrow_to_64(packed, false);
            self.b.write(GuestReg::V128(vd.0), v);
        }
        Flow::Next
    }

    fn shift_imm(
        &mut self,
        op: VShiftImmOp,
        arr: VArr,
        vd: gex_isa::VReg,
        vn: gex_isa::VReg,
        shift: u32,
        _pc: u64,
    ) -> Flow {
        // The fixed-point conversions share this encoding but are not
        // shifts, so they leave before any of the shift reasoning below.
        if matches!(
            op,
            VShiftImmOp::Scvtf
                | VShiftImmOp::Ucvtf
                | VShiftImmOp::Fcvtzs
                | VShiftImmOp::Fcvtzu
        ) {
            return self.fixed_convert(op, arr, vd, vn, shift);
        }
        let lane = Self::lane_of(arr);
        let a = self.b.read(GuestReg::V128(vn.0));
        // A right shift by the whole element width is legal on AArch64 (every
        // bit out: zero, or the replicated sign bit) but wasm takes the count
        // modulo the lane width, which would instead return the input.
        let bits = arr.lane_bytes * 8;
        let shift = if matches!(op, VShiftImmOp::SShr | VShiftImmOp::UShr) && shift >= bits {
            if op == VShiftImmOp::UShr {
                // Nothing survives. Shifting by one less and then once more
                // would work too; a zero is clearer and cheaper.
                let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
                self.commit_vector(vd, arr, zero);
                return Flow::Next;
            }
            bits - 1
        } else {
            shift
        };
        let amount = self.b.const_i32(shift);
        let sop = match op {
            VShiftImmOp::Shl => VShiftOp::Shl,
            // Shift-and-insert: the vacated bits come from the destination.
            // Shift the source, then splice in the part of the destination the
            // shift did not reach.
            VShiftImmOp::Sli | VShiftImmOp::Sri => {
                let old = self.b.read(GuestReg::V128(vd.0));
                let left = op == VShiftImmOp::Sli;
                // `sri` may shift by the whole element width, which leaves
                // nothing of the source and the destination untouched. wasm's
                // modulo count would hand back the source unchanged instead.
                // `sli` cannot encode that count, so only one side needs it.
                let shifted = if !left && shift >= bits {
                    self.b.def(Expr::VConst(Box::new([0u8; 16])))
                } else {
                    self.b.def(Expr::VShift {
                        op: if left { VShiftOp::Shl } else { VShiftOp::ShrU },
                        lane,
                        a,
                        amount,
                    })
                };
                // Bits of the destination that survive: the low `shift` of
                // each lane for `sli`, the high `shift` for `sri`.
                let mut keep = [0u8; 16];
                let width = arr.lane_bytes as usize;
                for lane_i in 0..16 / width {
                    for bit in 0..bits {
                        let alive = if left { bit < shift } else { bit >= bits - shift };
                        if alive {
                            let byte = lane_i * width + (bit / 8) as usize;
                            keep[byte] |= 1 << (bit % 8);
                        }
                    }
                }
                let mask = self.b.def(Expr::VConst(Box::new(keep)));
                let kept = self.b.def(Expr::VBin {
                    op: VBinOp::And,
                    a: old,
                    b: mask,
                });
                let r = self.b.def(Expr::VBin {
                    op: VBinOp::Or,
                    a: shifted,
                    b: kept,
                });
                self.commit_vector(vd, arr, r);
                return Flow::Next;
            }
            // Saturating shifts left. A lane fits when shifting the result
            // back gives the lane again (arithmetically for `sqshl`,
            // logically for the unsigned forms); one that does not takes the
            // limit on its side: all-ones for the unsigned forms, and for
            // `sqshl` the signed maximum, or the minimum for a negative lane
            // (`MAX ^ (a >> (bits-1))`). `sqshlu` clamps negative lanes to
            // zero, and a lane that is not negative saturates as `uqshl`'s.
            // FPSR.QC is not kept: it reads back as zero, as for the other
            // saturating lowerings.
            VShiftImmOp::SqShl | VShiftImmOp::UqShl | VShiftImmOp::SqShlU => {
                let signed = op == VShiftImmOp::SqShl;
                let shifted = self.b.def(Expr::VShift {
                    op: VShiftOp::Shl,
                    lane,
                    a,
                    amount,
                });
                let back = self.b.def(Expr::VShift {
                    op: if signed { VShiftOp::ShrS } else { VShiftOp::ShrU },
                    lane,
                    a: shifted,
                    amount,
                });
                let fits = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Eq,
                    lane,
                    a: back,
                    b: a,
                });
                let top = self.b.const_i32(bits - 1);
                let sign = self.b.def(Expr::VShift {
                    op: VShiftOp::ShrS,
                    lane,
                    a,
                    amount: top,
                });
                let limit = if signed {
                    let mut max = [0xffu8; 16];
                    let width = arr.lane_bytes as usize;
                    for l in 0..16 / width {
                        max[l * width + width - 1] = 0x7f;
                    }
                    let max = self.b.def(Expr::VConst(Box::new(max)));
                    self.b.def(Expr::VBin {
                        op: VBinOp::Xor,
                        a: max,
                        b: sign,
                    })
                } else {
                    self.b.def(Expr::VConst(Box::new([0xffu8; 16])))
                };
                let mut r = self.b.def(Expr::VBitSelect {
                    mask: fits,
                    a: shifted,
                    b: limit,
                });
                if op == VShiftImmOp::SqShlU {
                    r = self.b.def(Expr::VBin {
                        op: VBinOp::AndNot,
                        a: r,
                        b: sign,
                    });
                }
                self.commit_vector(vd, arr, r);
                return Flow::Next;
            }
            // Shift right, then add to what the destination already holds.
            VShiftImmOp::SSra | VShiftImmOp::USra => {
                let signed = op == VShiftImmOp::SSra;
                let old = self.b.read(GuestReg::V128(vd.0));
                let shifted = if !signed && shift >= bits {
                    self.b.def(Expr::VConst(Box::new([0u8; 16])))
                } else {
                    let n = self.b.const_i32(if shift >= bits { bits - 1 } else { shift });
                    self.b.def(Expr::VShift {
                        op: if signed { VShiftOp::ShrS } else { VShiftOp::ShrU },
                        lane,
                        a,
                        amount: n,
                    })
                };
                let r = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Add,
                    lane,
                    a: old,
                    b: shifted,
                });
                self.commit_vector(vd, arr, r);
                return Flow::Next;
            }
            // Rounding shift right, and its accumulating form. AArch64 adds
            // `1 << (shift-1)` before the shift and keeps the extra carry bit;
            // done that way in wasm the add would overflow a lane. The identical
            // result without overflow is the truncated shift plus the bit the
            // shift rounded off: `(a >> shift) + ((a >> (shift-1)) & 1)`. The
            // truncated part is all-zero once the shift reaches the lane width.
            VShiftImmOp::Srshr
            | VShiftImmOp::Urshr
            | VShiftImmOp::Srsra
            | VShiftImmOp::Ursra => {
                let signed = matches!(op, VShiftImmOp::Srshr | VShiftImmOp::Srsra);
                let accumulate = matches!(op, VShiftImmOp::Srsra | VShiftImmOp::Ursra);
                let trunc = if !signed && shift >= bits {
                    self.b.def(Expr::VConst(Box::new([0u8; 16])))
                } else {
                    let n = self.b.const_i32(if shift >= bits { bits - 1 } else { shift });
                    self.b.def(Expr::VShift {
                        op: if signed { VShiftOp::ShrS } else { VShiftOp::ShrU },
                        lane,
                        a,
                        amount: n,
                    })
                };
                // The bit rounded off: bit `shift-1` of each lane, kept as bit 0.
                let rb = self.b.const_i32(shift - 1);
                let dropped = self.b.def(Expr::VShift {
                    op: VShiftOp::ShrU,
                    lane,
                    a,
                    amount: rb,
                });
                let mut low1 = [0u8; 16];
                let width = arr.lane_bytes as usize;
                for l in 0..16 / width {
                    low1[l * width] = 1;
                }
                let mask = self.b.def(Expr::VConst(Box::new(low1)));
                let round = self.b.def(Expr::VBin {
                    op: VBinOp::And,
                    a: dropped,
                    b: mask,
                });
                let rounded = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::Add,
                    lane,
                    a: trunc,
                    b: round,
                });
                let r = if accumulate {
                    let old = self.b.read(GuestReg::V128(vd.0));
                    self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane,
                        a: old,
                        b: rounded,
                    })
                } else {
                    rounded
                };
                self.commit_vector(vd, arr, r);
                return Flow::Next;
            }
            VShiftImmOp::SShr => VShiftOp::ShrS,
            VShiftImmOp::UShr => VShiftOp::ShrU,
            // The width-changing forms convert first and then shift at the
            // other width, so they do not fit the shape below.
            VShiftImmOp::SShll | VShiftImmOp::UShll => {
                let signed = op == VShiftImmOp::SShll;
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                // `sshll2` reads the source's upper half; `sshll` the lower.
                let widened = self.b.def(Expr::VCvt {
                    op: if arr.q {
                        VCvtOp::ExtendHigh { signed }
                    } else {
                        VCvtOp::ExtendLow { signed }
                    },
                    from: lane,
                    a,
                    b: None,
                });
                let r = if shift == 0 {
                    widened
                } else {
                    self.b.def(Expr::VShift {
                        op: VShiftOp::Shl,
                        lane: Self::lane_of(wide),
                        a: widened,
                        amount,
                    })
                };
                self.b.write(GuestReg::V128(vd.0), r);
                return Flow::Next;
            }
            // The saturating narrowing shifts: the shift at the wide width
            // (arithmetic but for the `uq` forms, whose source is unsigned),
            // rounded as `srshr` is, without overflow; then the saturating
            // narrow `sqxtn` / `uqxtn` / `sqxtun` does. The shift is 1 up to
            // the narrow width, always inside the wide lane. FPSR.QC is not
            // kept, as for the other saturating lowerings.
            VShiftImmOp::SqShrn
            | VShiftImmOp::SqRshrn
            | VShiftImmOp::UqShrn
            | VShiftImmOp::UqRshrn
            | VShiftImmOp::SqShrun
            | VShiftImmOp::SqRshrun => {
                let wlane = Self::lane_of(VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                });
                let unsigned = matches!(op, VShiftImmOp::UqShrn | VShiftImmOp::UqRshrn);
                let rounding = matches!(
                    op,
                    VShiftImmOp::SqRshrn | VShiftImmOp::UqRshrn | VShiftImmOp::SqRshrun
                );
                let mut shifted = self.b.def(Expr::VShift {
                    op: if unsigned { VShiftOp::ShrU } else { VShiftOp::ShrS },
                    lane: wlane,
                    a,
                    amount,
                });
                if rounding {
                    let rb = self.b.const_i32(shift - 1);
                    let dropped = self.b.def(Expr::VShift {
                        op: VShiftOp::ShrU,
                        lane: wlane,
                        a,
                        amount: rb,
                    });
                    let one = self.lane_splat(wlane, 1);
                    let round = self.b.def(Expr::VBin {
                        op: VBinOp::And,
                        a: dropped,
                        b: one,
                    });
                    shifted = self.b.def(Expr::VLaneBin {
                        op: VLaneBinOp::Add,
                        lane: wlane,
                        a: shifted,
                        b: round,
                    });
                }
                let narrow = match op {
                    VShiftImmOp::SqShrn | VShiftImmOp::SqRshrn => VOp2::SqXtn,
                    VShiftImmOp::UqShrn | VShiftImmOp::UqRshrn => VOp2::UqXtn,
                    _ => VOp2::SqXtun,
                };
                return self.sat_narrow(narrow, arr, vd, shifted);
            }
            VShiftImmOp::Shrn | VShiftImmOp::Rshrn => {
                // The source lanes are twice as wide as `arr` describes.
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                let wlane = Self::lane_of(wide);
                let shifted = if shift == 0 {
                    a
                } else {
                    let base = self.b.def(Expr::VShift {
                        op: VShiftOp::ShrU,
                        lane: wlane,
                        a,
                        amount,
                    });
                    if op == VShiftImmOp::Rshrn {
                        // Rounding: add the bit the shift dropped (bit `shift-1`
                        // of each wide lane), overflow-free -- see the rounding
                        // shift-right case above.
                        let rb = self.b.const_i32(shift - 1);
                        let dropped = self.b.def(Expr::VShift {
                            op: VShiftOp::ShrU,
                            lane: wlane,
                            a,
                            amount: rb,
                        });
                        let mut low1 = [0u8; 16];
                        let width = (arr.lane_bytes * 2) as usize;
                        for l in 0..16 / width {
                            low1[l * width] = 1;
                        }
                        let mask1 = self.b.def(Expr::VConst(Box::new(low1)));
                        let round = self.b.def(Expr::VBin {
                            op: VBinOp::And,
                            a: dropped,
                            b: mask1,
                        });
                        self.b.def(Expr::VLaneBin {
                            op: VLaneBinOp::Add,
                            lane: wlane,
                            a: base,
                            b: round,
                        })
                    } else {
                        base
                    }
                };
                // `shrn` truncates: keep the low half of each wide lane. A
                // byte shuffle does that at every width, including the
                // doubleword sources wasm has no narrowing instruction for.
                let mut lanes = [0u8; 16];
                let half = arr.lane_bytes;
                for i in 0..8u32 {
                    lanes[i as usize] = ((i / half) * half * 2 + i % half) as u8;
                }
                let packed = self.b.def(Expr::VShuffle {
                    a: shifted,
                    b: shifted,
                    lanes: Box::new(lanes),
                });
                if arr.q {
                    // `shrn2` fills the destination's upper half.
                    let old = self.b.read(GuestReg::V128(vd.0));
                    let mut sel = [0u8; 16];
                    for (i, slot) in sel.iter_mut().enumerate() {
                        *slot = if i < 8 { i as u8 } else { (16 + i - 8) as u8 };
                    }
                    let r = self.b.def(Expr::VShuffle {
                        lanes: Box::new(sel),
                        a: old,
                        b: packed,
                    });
                    self.b.write(GuestReg::V128(vd.0), r);
                } else {
                    let r = self.narrow_to_64(packed, false);
                    self.b.write(GuestReg::V128(vd.0), r);
                }
                return Flow::Next;
            }
            // Handled above, before any of the shift reasoning.
            VShiftImmOp::Scvtf
            | VShiftImmOp::Ucvtf
            | VShiftImmOp::Fcvtzs
            | VShiftImmOp::Fcvtzu => unreachable!("fixed-point converts leave earlier"),
        };
        let r = self.b.def(Expr::VShift {
            op: sop,
            lane,
            a,
            amount,
        });
        self.commit_vector(vd, arr, r);
        Flow::Next
    }

    /// `ld3`/`ld4` and `st3`/`st4`: three or four registers interleaved in
    /// memory (lane 0 of each, then lane 1, ...), so both directions are a
    /// byte gather over whole vectors plus a trailing 8-byte access when the
    /// `regs * bytes` block is not a multiple of 16. Three or more sources
    /// need a gather tree; see [`Self::gather_bytes`].
    fn load_store_n(
        &mut self,
        load: bool,
        regs: u8,
        lane_bytes: u32,
        bytes: u32,
        vt: gex_isa::VReg,
        addr: &AddrMode,
    ) -> Flow {
        let lb = lane_bytes;
        let lanes_each = bytes / lb;
        let total = regs as u32 * bytes;
        // 16 bytes at a time, with a final 8 when the block is not a whole
        // number of vectors.
        let chunks: Vec<(i64, u32)> = {
            let mut v = Vec::new();
            let mut at = 0;
            while at < total {
                let n = if total - at >= 16 { 16 } else { total - at };
                v.push((at as i64, n));
                at += n;
            }
            v
        };

        if load {
            let a = self.address(addr);
            let srcs: Vec<Value> = chunks
                .iter()
                .map(|&(off, n)| {
                    self.b.def(Expr::VLoad {
                        addr: a.base,
                        offset: a.offset + off,
                        bytes: n,
                    })
                })
                .collect();
            for r in 0..regs {
                // Output byte `lane * lb + k` of register `r` is block byte
                // `(lane * regs + r) * lb + k`.
                let mut want = [0u8; 16];
                for lane in 0..lanes_each {
                    for k in 0..lb {
                        let src = (lane * regs as u32 + r as u32) * lb + k;
                        want[(lane * lb + k) as usize] = src as u8;
                    }
                }
                let v = self.gather_bytes(&srcs, &want);
                let v = self.narrow_to_64(v, bytes == 16);
                self.b.write(GuestReg::V128((vt.0 + r) % 32), v);
            }
            self.finish_address(&a);
        } else {
            let srcs: Vec<Value> = (0..regs)
                .map(|i| self.b.read(GuestReg::V128((vt.0 + i) % 32)))
                .collect();
            let a = self.address(addr);
            for &(off, n) in &chunks {
                // Block byte `B` is byte `lane * lb + k` of register
                // `elem % regs`, where `elem = B / lb`.
                let mut want = [0u8; 16];
                for pos in 0..n {
                    let b_off = off as u32 + pos;
                    let elem = b_off / lb;
                    let k = b_off % lb;
                    let r = elem % regs as u32;
                    let lane = elem / regs as u32;
                    // Registers are 16 bytes apart in the gather's numbering,
                    // whatever the arrangement's width.
                    want[pos as usize] = (r * 16 + lane * lb + k) as u8;
                }
                let v = self.gather_bytes(&srcs, &want);
                self.b.stmt(Stmt::VStore {
                    addr: a.base,
                    offset: a.offset + off,
                    bytes: n,
                    val: v,
                });
            }
            self.finish_address(&a);
        }
        Flow::Next
    }

    /// Gather each byte from `want[i]`, a position in the concatenation of up
    /// to four sources (16 bytes each). wasm shuffle reaches two vectors, so
    /// three or four sources take a tree: one shuffle per pair, then one to
    /// choose between them.
    fn gather_bytes(&mut self, srcs: &[Value], want: &[u8; 16]) -> Value {
        debug_assert!(!srcs.is_empty() && srcs.len() <= 4);
        let pair = |lift: &mut Self, base: usize| -> Value {
            let a = srcs[base];
            let b = *srcs.get(base + 1).unwrap_or(&a);
            let mut sel = [0u8; 16];
            for i in 0..16 {
                let w = want[i] as usize;
                sel[i] = if w >= base * 16 && w < base * 16 + 32 {
                    (w - base * 16) as u8
                } else {
                    0
                };
            }
            lift.b.def(Expr::VShuffle {
                lanes: Box::new(sel),
                a,
                b,
            })
        };
        let lo = pair(self, 0);
        if srcs.len() <= 2 {
            return lo;
        }
        let hi = pair(self, 2);
        let mut sel = [0u8; 16];
        for i in 0..16 {
            sel[i] = if (want[i] as usize) < 32 {
                i as u8
            } else {
                (16 + i) as u8
            };
        }
        self.b.def(Expr::VShuffle {
            lanes: Box::new(sel),
            a: lo,
            b: hi,
        })
    }

    /// `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu` with a fractional-bit count.
    ///
    /// The lane holds a fixed-point number with `fbits` fractional bits, so
    /// the conversion is the ordinary one with a scale either side of it:
    /// dividing by `2^fbits` on the way out of an integer, multiplying by it
    /// on the way in. Scaling by a power of two is exact, so nothing is lost.
    fn fixed_convert(
        &mut self,
        op: VShiftImmOp,
        arr: VArr,
        vd: gex_isa::VReg,
        vn: gex_isa::VReg,
        fbits: u32,
    ) -> Flow {
        let signed = matches!(op, VShiftImmOp::Scvtf | VShiftImmOp::Fcvtzs);
        let to_fp = matches!(op, VShiftImmOp::Scvtf | VShiftImmOp::Ucvtf);
        // The same lanes, read as floats: the arrangement the encoding gives
        // describes the integer side.
        let fa = VArr {
            float: true,
            ..arr
        };
        let lane = Self::lane_of(fa);
        let a = self.b.read(GuestReg::V128(vn.0));

        let scale = if to_fp { -(fbits as i32) } else { fbits as i32 };
        let scale = exp2_exact(scale as i32);
        let mut bytes = [0u8; 16];
        if arr.lane_bytes == 4 {
            let f = (scale as f32).to_le_bytes();
            for i in 0..4 {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&f);
            }
        } else {
            let d = scale.to_le_bytes();
            for i in 0..2 {
                bytes[i * 8..i * 8 + 8].copy_from_slice(&d);
            }
        }
        let k = self.b.def(Expr::VConst(Box::new(bytes)));

        let r = if to_fp {
            let f = self.lane_convert(fa, a, FpConvert::IntToFp { signed }, signed);
            self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Mul,
                lane,
                a: f,
                b: k,
            })
        } else {
            let scaled = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Mul,
                lane,
                a,
                b: k,
            });
            self.lane_convert(fa, scaled, FpConvert::FpToInt { signed }, signed)
        };
        self.commit_vector(vd, arr, r);
        Flow::Next
    }

    /// Convert every lane between integer and floating point. wasm's SIMD
    /// conversions only cover 32-bit lanes, so 64-bit arrangements convert
    /// their two lanes one at a time with the scalar instructions.
    fn lane_convert(&mut self, arr: VArr, a: Value, kind: FpConvert, signed: bool) -> Value {
        if arr.lane_bytes != 8 {
            let to_fp = matches!(kind, FpConvert::IntToFp { .. });
            let lane = Self::lane_of(arr);
            return self.b.def(Expr::VCvt {
                op: if to_fp {
                    VCvtOp::IntToFp { signed }
                } else {
                    VCvtOp::FpToIntSat { signed }
                },
                from: if to_fp { lane.as_int() } else { lane },
                a,
                b: None,
            });
        }
        let mut out = self.b.def(Expr::VConst(Box::new([0u8; 16])));
        for lane in 0..2 {
            let x = self.b.def(Expr::VExtract {
                a,
                lane,
                lane_bytes: 8,
                signed: false,
                ty: Ty::I64,
            });
            let c = self.b.def(Expr::FpCvt {
                kind,
                src_wide: true,
                dst_wide: true,
                a: x,
            });
            out = self.b.def(Expr::VReplace {
                a: out,
                lane,
                lane_bytes: 8,
                val: c,
            });
        }
        out
    }

    /// Multiply a float by `2^exp`, or return it unchanged when `exp` is zero.
    ///
    /// The multiplier is built as a bit pattern rather than parsed from a
    /// literal, because `exp` reaches +/-64 and the exponent field is the only
    /// part that moves.
    fn scale_by_power_of_two(&mut self, v: Value, double: bool, exp: i32) -> Value {
        if exp == 0 {
            return v;
        }
        let k = if double {
            let biased = (1023 + exp) as u64;
            self.konst(Ty::I64, biased << 52)
        } else {
            let biased = (127 + exp) as u64;
            self.konst(Ty::I32, biased << 23)
        };
        self.b.def(Expr::FpBin {
            op: FpBinOp::Mul,
            double,
            a: v,
            b: k,
        })
    }

    /// The four flags an `fcmp` installs, shared with `fccmp`.
    fn fp_compare_flags(
        &mut self,
        double: bool,
        a: Value,
        b: Value,
    ) -> (Value, Value, Value, Value) {
        let lt = self.b.def(Expr::FpCmp {
            op: FpCmpOp::Lt,
            double,
            a,
            b,
        });
        let eq = self.b.def(Expr::FpCmp {
            op: FpCmpOp::Eq,
            double,
            a,
            b,
        });
        let a_nan = self.b.def(Expr::FpCmp {
            op: FpCmpOp::Ne,
            double,
            a,
            b: a,
        });
        let b_nan = self.b.def(Expr::FpCmp {
            op: FpCmpOp::Ne,
            double,
            a: b,
            b,
        });
        let unordered = self.b.bin(BinOp::Or, Ty::I32, a_nan, b_nan);
        let one = self.b.const_i32(1);
        let not_lt = self.b.bin(BinOp::Xor, Ty::I32, lt, one);
        (lt, eq, not_lt, unordered)
    }

    /// FRINTA: to nearest, ties away from zero. `t = trunc(a)`, and one
    /// more step away from zero when `|a - t| >= 0.5`. `a - t` is exact (the
    /// fraction of `a`), so no value near a half is misjudged, unlike
    /// `trunc(a + copysign(0.5, a))`, which rounds 0.49999999999999994 up.
    /// A NaN or an infinity compares false and stays itself; -0.3 stays -0.
    fn round_away(&mut self, double: bool, a: Value) -> Value {
        let (half, one, ty) = if double {
            (self.b.const_i64(0.5f64.to_bits()), self.b.const_i64(1.0f64.to_bits()), Ty::I64)
        } else {
            (self.b.const_i32(0.5f32.to_bits()), self.b.const_i32(1.0f32.to_bits()), Ty::I32)
        };
        let t = self.b.def(Expr::FpUn { op: FpUnOp::Trunc, double, a });
        let frac = self.fp_bin(FpBinOp::Sub, double, a, t);
        let mag = self.b.def(Expr::FpUn { op: FpUnOp::Abs, double, a: frac });
        let up = self.b.def(Expr::FpCmp { op: FpCmpOp::Ge, double, a: mag, b: half });
        let step = self.fp_bin(FpBinOp::Copysign, double, one, a);
        let away = self.fp_bin(FpBinOp::Add, double, t, step);
        self.b.def(Expr::Select { cond: up, a: away, b: t, ty })
    }

    /// [`Self::round_away`] lane-wise, for the vector FRINTA.
    fn round_away_lanes(&mut self, lane: VLane, a: Value) -> Value {
        let splat = |bits: u64| -> Box<[u8; 16]> {
            let mut v = [0u8; 16];
            if lane == VLane::F64 {
                for c in v.chunks_exact_mut(8) {
                    c.copy_from_slice(&bits.to_le_bytes());
                }
            } else {
                for c in v.chunks_exact_mut(4) {
                    c.copy_from_slice(&(bits as u32).to_le_bytes());
                }
            }
            Box::new(v)
        };
        let (half, one, sign) = if lane == VLane::F64 {
            (0.5f64.to_bits(), 1.0f64.to_bits(), 1u64 << 63)
        } else {
            (0.5f32.to_bits() as u64, 1.0f32.to_bits() as u64, 1u64 << 31)
        };
        let half = self.b.def(Expr::VConst(splat(half)));
        let one = self.b.def(Expr::VConst(splat(one)));
        let sign = self.b.def(Expr::VConst(splat(sign)));
        let t = self.b.def(Expr::VLaneUn { op: VLaneUnOp::Trunc, lane, a });
        let frac = self.b.def(Expr::VLaneBin { op: VLaneBinOp::Sub, lane, a, b: t });
        let mag = self.b.def(Expr::VLaneUn { op: VLaneUnOp::Abs, lane, a: frac });
        let up = self.b.def(Expr::VLaneBin { op: VLaneBinOp::GeS, lane, a: mag, b: half });
        let step = self.b.def(Expr::VBitSelect { mask: sign, a, b: one });
        let away = self.b.def(Expr::VLaneBin { op: VLaneBinOp::Add, lane, a: t, b: step });
        self.b.def(Expr::VBitSelect { mask: up, a: away, b: t })
    }

    fn fp_bin(&mut self, op: FpBinOp, double: bool, a: Value, b: Value) -> Value {
        self.b.def(Expr::FpBin { op, double, a, b })
    }

    /// Read a scalar float out of a `v` register as its bit pattern.
    /// `vd = (+/-a) * b + (+/-c)`, with one rounding. AArch64 fuses this; wasm has no FMA, and
    /// rounding the product first is enough to break musl's `pow`, which is built on it. So the
    /// host's FMA is called.
    #[allow(clippy::too_many_arguments)]
    fn fused(&mut self, vd: gex_isa::VReg, sz: FpSz, a: Value, b: Value, c: Value, neg_product: bool, neg_addend: bool) -> Flow {
        let double = sz.is_double();
        let a = if neg_product {
            self.b.def(Expr::FpUn {
                op: FpUnOp::Neg,
                double,
                a,
            })
        } else {
            a
        };
        let c = if neg_addend {
            self.b.def(Expr::FpUn {
                op: FpUnOp::Neg,
                double,
                a: c,
            })
        } else {
            c
        };
        // GEX_INLINE_FMA emits a multiply and an add instead. Not
        // correct -- it rounds twice -- and exists only to measure
        // what the fused host call costs.
        if inline_fma() {
            let prod = self.b.def(Expr::FpBin {
                op: FpBinOp::Mul,
                double,
                a,
                b,
            });
            let v = self.b.def(Expr::FpBin {
                op: FpBinOp::Add,
                double,
                a: prod,
                b: c,
            });
            self.wr_fp(vd, sz, v);
            return Flow::Next;
        }
        let (a64, b64, c64) = if double {
            (a, b, c)
        } else {
            // The import speaks in 64-bit words; a single-precision
            // pattern rides in the low half.
            (
                self.b.def(Expr::Extend { a, signed: false }),
                self.b.def(Expr::Extend {
                    a: b,
                    signed: false,
                }),
                self.b.def(Expr::Extend {
                    a: c,
                    signed: false,
                }),
            )
        };
        let is_double = self.konst(Ty::I32, double as u64);
        let r = self.b.def(Expr::HostCall {
            func: HostFunc::Fma,
            args: vec![a64, b64, c64, is_double],
            ty: Ty::I64,
        });
        let v = if double { r } else { self.b.def(Expr::Wrap(r)) };
        self.wr_fp(vd, sz, v);
        Flow::Next
    }

    fn rd_fp(&mut self, v: gex_isa::VReg, sz: FpSz) -> Value {
        let vec = self.b.read(GuestReg::V128(v.0));
        self.b.def(Expr::VExtract {
            a: vec,
            lane: 0,
            lane_bytes: sz.bytes(),
            signed: false,
            ty: if sz.is_double() { Ty::I64 } else { Ty::I32 },
        })
    }

    /// Write a scalar float bit pattern into a `v` register.
    ///
    /// Writing any scalar view of a SIMD&FP register zeroes every bit above
    /// it, so the destination is built from an all-zero vector rather than
    /// from the register's previous contents.
    fn wr_fp(&mut self, v: gex_isa::VReg, sz: FpSz, bits: Value) {
        let zero = self.b.def(Expr::VConst(Box::new([0u8; 16])));
        let out = self.b.def(Expr::VReplace {
            a: zero,
            lane: 0,
            lane_bytes: sz.bytes(),
            val: bits,
        });
        self.b.write(GuestReg::V128(v.0), out);
    }

    /// A 64-bit-form SIMD instruction zeroes the upper half of its
    /// destination; the 128-bit form leaves the value alone.
    /// `sqxtn`, `uqxtn` and `sqxtun` from doubleword lanes, which wasm has
    /// no narrowing instruction for: clamp each lane with 64-bit compares,
    /// then keep the low word of each. The words land in bytes 0..8, as
    /// wasm's narrowing instructions leave theirs (the rest is don't-care).
    fn narrow_sat_i64(&mut self, op: VOp2, a: Value) -> Value {
        let splat = |b: &mut Builder, v: u64| -> Value {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&v.to_le_bytes());
            bytes[8..].copy_from_slice(&v.to_le_bytes());
            b.def(Expr::VConst(Box::new(bytes)))
        };
        let clamped = if op == VOp2::SqXtn {
            let hi = splat(&mut *self.b, i32::MAX as i64 as u64);
            let lo = splat(&mut *self.b, i32::MIN as i64 as u64);
            let over = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::GtS,
                lane: VLane::I64,
                a,
                b: hi,
            });
            let x = self.b.def(Expr::VBitSelect {
                mask: over,
                a: hi,
                b: a,
            });
            let under = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::LtS,
                lane: VLane::I64,
                a: x,
                b: lo,
            });
            self.b.def(Expr::VBitSelect {
                mask: under,
                a: lo,
                b: x,
            })
        } else {
            // Unsigned saturation to 0xffff_ffff: any bit above the low word
            // saturates. `sqxtun` first clears negative lanes to zero.
            let zero = splat(&mut *self.b, 0);
            let src = if op == VOp2::SqXtun {
                let neg = self.b.def(Expr::VLaneBin {
                    op: VLaneBinOp::LtS,
                    lane: VLane::I64,
                    a,
                    b: zero,
                });
                self.b.def(Expr::VBin {
                    op: VBinOp::AndNot,
                    a,
                    b: neg,
                })
            } else {
                a
            };
            let n32 = self.b.const_i32(32);
            let high = self.b.def(Expr::VShift {
                op: VShiftOp::ShrU,
                lane: VLane::I64,
                a: src,
                amount: n32,
            });
            let over = self.b.def(Expr::VLaneBin {
                op: VLaneBinOp::Ne,
                lane: VLane::I64,
                a: high,
                b: zero,
            });
            let ones = splat(&mut *self.b, u64::MAX);
            self.b.def(Expr::VBitSelect {
                mask: over,
                a: ones,
                b: src,
            })
        };
        let mut lanes = [0u8; 16];
        for i in 0..8u8 {
            lanes[i as usize] = (i / 4) * 8 + i % 4;
        }
        for i in 8..16u8 {
            lanes[i as usize] = lanes[i as usize - 8];
        }
        self.b.def(Expr::VShuffle {
            a: clamped,
            b: clamped,
            lanes: Box::new(lanes),
        })
    }

    /// A vector with `x` in every lane of `lane`'s width.
    fn lane_splat(&mut self, lane: VLane, x: u64) -> Value {
        let w = lane.bytes() as usize;
        let mut bytes = [0u8; 16];
        for l in 0..16 / w {
            bytes[l * w..l * w + w].copy_from_slice(&x.to_le_bytes()[..w]);
        }
        self.b.def(Expr::VConst(Box::new(bytes)))
    }

    /// `sqadd`, `uqadd`, `sqsub` and `uqsub` on word and doubleword lanes, which wasm
    /// has no saturating add or subtract for. The wrapping result is right unless it
    /// overflowed: for the signed forms when the operands' signs agree (add) or differ
    /// (subtract) and the result's sign is not the first operand's, which
    /// `(r ^ a) & (r ^ b)` / `(a ^ b) & (a ^ r)` shows in the sign bit; such a lane
    /// takes the limit on `a`'s side, `MAX ^ (a >> (bits-1))`. Unsigned, the sum
    /// overflowed when it is below `a` (all-ones then), the difference when `a < b`
    /// (zero then). i64x2 has no unsigned compare, so those lanes compare biased into
    /// signed order. FPSR.QC is not kept, as for the other saturating lowerings.
    fn sat_add_sub_wide(&mut self, op: VOp3, lane: VLane, a: Value, b: Value) -> Value {
        let add = matches!(op, VOp3::SqAdd | VOp3::UqAdd);
        let r = self.b.def(Expr::VLaneBin {
            op: if add { VLaneBinOp::Add } else { VLaneBinOp::Sub },
            lane,
            a,
            b,
        });
        let bits = lane.bytes() * 8;
        if matches!(op, VOp3::SqAdd | VOp3::SqSub) {
            let (x, y, z, t) = if add { (r, a, r, b) } else { (a, b, a, r) };
            let p = self.b.def(Expr::VBin { op: VBinOp::Xor, a: x, b: y });
            let q = self.b.def(Expr::VBin { op: VBinOp::Xor, a: z, b: t });
            let ov = self.b.def(Expr::VBin { op: VBinOp::And, a: p, b: q });
            let top = self.b.const_i32(bits - 1);
            let mask = self.b.def(Expr::VShift { op: VShiftOp::ShrS, lane, a: ov, amount: top });
            let sign = self.b.def(Expr::VShift { op: VShiftOp::ShrS, lane, a, amount: top });
            let max = self.lane_splat(lane, u64::MAX >> (65 - bits));
            let limit = self.b.def(Expr::VBin { op: VBinOp::Xor, a: max, b: sign });
            return self.b.def(Expr::VBitSelect { mask, a: limit, b: r });
        }
        // Unsigned: `lt_u(x, y)`, the sum below `a` or `a` below `b`.
        let (x, y) = if add { (r, a) } else { (a, b) };
        let below = if lane == VLane::I64 {
            let bias = self.lane_splat(lane, 1 << 63);
            let x = self.b.def(Expr::VBin { op: VBinOp::Xor, a: x, b: bias });
            let y = self.b.def(Expr::VBin { op: VBinOp::Xor, a: y, b: bias });
            self.b.def(Expr::VLaneBin { op: VLaneBinOp::LtS, lane, a: x, b: y })
        } else {
            self.b.def(Expr::VLaneBin { op: VLaneBinOp::LtU, lane, a: x, b: y })
        };
        if add {
            self.b.def(Expr::VBin { op: VBinOp::Or, a: r, b: below })
        } else {
            self.b.def(Expr::VBin { op: VBinOp::AndNot, a: r, b: below })
        }
    }

    /// `sqdmulh` and `sqrdmulh`: `(2*a*b [+ 1 << (bits-1)]) >> bits` per lane, which
    /// overflows only for `MIN * MIN`. Halfwords: the rounding form is wasm's
    /// `i16x8.q15mulr_sat_s` exactly, and the plain one `(a*b) >> 15` from the widened
    /// products, whose one overflow the saturating narrow takes. Words: the products
    /// widened to i64 (`extmul`), `(p [+ 1 << 30]) >> 31` -- the same value with both
    /// sides halved -- and the low words of that. Nothing but `MIN * MIN` gives
    /// `0x8000_0000` there (the extreme products are `-2^62 + 2^31` and
    /// `2^62 - 2^31`), so turning that one value into `MAX` is the saturation.
    /// FPSR.QC is not kept, as for the other saturating lowerings.
    fn sat_doubling_mulh(&mut self, round: bool, arr: VArr, a: Value, b: Value) -> Value {
        if arr.lane_bytes == 2 {
            if round {
                return self.b.def(Expr::VLaneBin { op: VLaneBinOp::Q15MulrSatS, lane: VLane::I16, a, b });
            }
            let fifteen = self.b.const_i32(15);
            let mut half = [None; 2];
            for (i, op) in [VLaneBinOp::ExtMulLowS, VLaneBinOp::ExtMulHighS].into_iter().enumerate() {
                let p = self.b.def(Expr::VLaneBin { op, lane: VLane::I16, a, b });
                half[i] = Some(self.b.def(Expr::VShift { op: VShiftOp::ShrS, lane: VLane::I32, a: p, amount: fifteen }));
            }
            return self.b.def(Expr::VCvt {
                op: VCvtOp::NarrowSat { signed: true },
                from: VLane::I32,
                a: half[0].unwrap(),
                b: half[1],
            });
        }
        // The 64-bit and scalar forms want only the low half.
        let both = arr.q && !arr.scalar;
        let bias = if round { Some(self.lane_splat(VLane::I64, 1 << 30)) } else { None };
        let thirty_one = self.b.const_i32(31);
        let mut half = Vec::new();
        for op in [VLaneBinOp::ExtMulLowS, VLaneBinOp::ExtMulHighS].into_iter().take(if both { 2 } else { 1 }) {
            let mut p = self.b.def(Expr::VLaneBin { op, lane: VLane::I32, a, b });
            if let Some(bias) = bias {
                p = self.b.def(Expr::VLaneBin { op: VLaneBinOp::Add, lane: VLane::I64, a: p, b: bias });
            }
            half.push(self.b.def(Expr::VShift { op: VShiftOp::ShrS, lane: VLane::I64, a: p, amount: thirty_one }));
        }
        let (lo, hi) = (half[0], *half.last().unwrap());
        // Words 0 and 2 of `lo`, then of `hi` (bytes 16.. of the pair).
        let mut lanes = [0u8; 16];
        for (i, slot) in lanes.iter_mut().enumerate() {
            *slot = [0u8, 8, 16, 24][i / 4] + (i % 4) as u8;
        }
        let packed = self.b.def(Expr::VShuffle { lanes: Box::new(lanes), a: lo, b: hi });
        let min = self.lane_splat(VLane::I32, 0x8000_0000);
        let over = self.b.def(Expr::VLaneBin { op: VLaneBinOp::Eq, lane: VLane::I32, a: packed, b: min });
        self.b.def(Expr::VBin { op: VBinOp::Xor, a: packed, b: over })
    }

    fn narrow_to_64(&mut self, v: Value, q: bool) -> Value {
        if q {
            return v;
        }
        let mut mask = [0u8; 16];
        mask[..8].fill(0xff);
        let m = self.b.def(Expr::VConst(Box::new(mask)));
        self.b.def(Expr::VBin {
            op: VBinOp::And,
            a: v,
            b: m,
        })
    }

    // -- instruction families ----------------------------------------------

    /// `add`/`sub` with optional flag setting, shared by all operand forms.
    fn arith(
        &mut self,
        ty: Ty,
        op: AddOp,
        set_flags: bool,
        rd: RegSel,
        sz: Sz,
        a: Value,
        b: Value,
    ) {
        if !set_flags {
            let v = self.b.bin(
                if op == AddOp::Add {
                    BinOp::Add
                } else {
                    BinOp::Sub
                },
                ty,
                a,
                b,
            );
            self.wr_gp(rd, sz, v);
            return;
        }
        // AArch64 defines subtraction as `a + NOT(b) + 1`, which is also what
        // its carry flag follows.
        let (operand, carry) = if op == AddOp::Sub {
            let neg = self.b.un(UnOp::Not, ty, b);
            (neg, Carry::One)
        } else {
            (b, Carry::Zero)
        };
        let r = self.add_with_carry(ty, a, operand, carry);
        self.wr_gp(rd, sz, r.value);
        self.set_nzcv(&r);
        // After the writes, since `set_nzcv` clears it. `b` is the original
        // second operand, not the negated `operand` the carry rule used.
        if op == AddOp::Sub {
            self.flags = FlagSource::Sub { a, b, ty };
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn logical(
        &mut self,
        ty: Ty,
        op: LogOp,
        invert: bool,
        set_flags: bool,
        rd: RegSel,
        sz: Sz,
        a: Value,
        b0: Value,
    ) {
        let b = if invert {
            self.b.un(UnOp::Not, ty, b0)
        } else {
            b0
        };
        let bin = match op {
            LogOp::And => BinOp::And,
            LogOp::Orr => BinOp::Or,
            LogOp::Eor => BinOp::Xor,
        };
        let v = self.b.bin(bin, ty, a, b);
        self.wr_gp(rd, sz, v);
        if set_flags {
            self.set_nz_clear_cv(ty, v);
        }
    }

    fn bitfield(&mut self, sz: Sz, op: BfOp, rd: RegSel, rn: RegSel, immr: u32, imms: u32) {
        let ty = ty_of(sz);
        let bits = ty.bits();
        // For the bitfield encodings the element size always equals the
        // register width (the decoder enforces `N == sf`), so `DecodeBitMasks`
        // reduces to two constants with no replication.
        let ones = |n: u32| -> u64 {
            if n >= 64 {
                u64::MAX
            } else {
                (1u64 << n) - 1
            }
        };
        let esize_mask = ones(bits);
        let welem = ones(imms + 1) & esize_mask;
        let wmask = if immr == 0 {
            welem
        } else {
            ((welem >> immr) | (welem << (bits - immr))) & esize_mask
        };
        let diff = imms.wrapping_sub(immr) & (bits - 1);
        let tmask = ones(diff + 1) & esize_mask;

        let src = self.rd_gp(rn, sz);
        let rotated = if immr == 0 {
            src
        } else {
            let amt = self.konst(ty, immr as u64);
            self.b.bin(BinOp::Rotr, ty, src, amt)
        };
        let wm = self.konst(ty, wmask);
        let bot = self.b.bin(BinOp::And, ty, rotated, wm);

        let result = match op {
            BfOp::Ubfm => {
                let tm = self.konst(ty, tmask);
                self.b.bin(BinOp::And, ty, bot, tm)
            }
            BfOp::Sbfm => {
                // Replicate the extracted field's top bit into the bits above.
                let shift = self.konst(ty, (bits - 1 - imms) as u64);
                let up = self.b.bin(BinOp::Shl, ty, src, shift);
                let down = self.konst(ty, (bits - 1) as u64);
                let top = self.b.bin(BinOp::ShrS, ty, up, down);
                let tm = self.konst(ty, tmask);
                let not_tm = self.konst(ty, !tmask & esize_mask);
                let hi = self.b.bin(BinOp::And, ty, top, not_tm);
                let lo = self.b.bin(BinOp::And, ty, bot, tm);
                self.b.bin(BinOp::Or, ty, hi, lo)
            }
            BfOp::Bfm => {
                let keep_mask = wmask & tmask;
                let old = self.rd_gp(rd, sz);
                let inv = self.konst(ty, !keep_mask & esize_mask);
                let kept = self.b.bin(BinOp::And, ty, old, inv);
                let km = self.konst(ty, keep_mask);
                let ins = self.b.bin(BinOp::And, ty, rotated, km);
                self.b.bin(BinOp::Or, ty, kept, ins)
            }
        };
        self.wr_gp(rd, sz, result);
    }

    fn dataproc2(
        &mut self,
        sz: Sz,
        op: Dp2Op,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        pc: u64,
    ) -> Flow {
        let ty = ty_of(sz);
        let a = self.rd_gp(rn, sz);
        let b = self.rd_gp(rm, sz);
        let v = match op {
            Dp2Op::Udiv | Dp2Op::Sdiv => {
                let signed = op == Dp2Op::Sdiv;
                let zero = self.konst(ty, 0);
                let is_zero = self.b.cmp(CmpOp::Eq, ty, b, zero);
                // Wasm traps on division by zero and on `INT_MIN / -1`;
                // AArch64 produces 0 and INT_MIN. Steering the divisor to 1
                // already yields INT_MIN for the overflow case, so only the
                // divide-by-zero result needs a select.
                let one = self.konst(ty, 1);
                let mut unsafe_divisor = is_zero;
                if signed {
                    let min = self.konst(ty, 1u64 << (ty.bits() - 1));
                    let neg1 = self.konst(ty, ty.mask());
                    let a_min = self.b.cmp(CmpOp::Eq, ty, a, min);
                    let b_neg1 = self.b.cmp(CmpOp::Eq, ty, b, neg1);
                    let ovf = self.b.bin(BinOp::And, Ty::I32, a_min, b_neg1);
                    unsafe_divisor = self.b.bin(BinOp::Or, Ty::I32, is_zero, ovf);
                }
                let divisor = self.b.select(unsafe_divisor, one, b, ty);
                let q = self.b.bin(
                    if signed { BinOp::DivS } else { BinOp::DivU },
                    ty,
                    a,
                    divisor,
                );
                self.b.select(is_zero, zero, q, ty)
            }
            Dp2Op::Lslv => self.b.bin(BinOp::Shl, ty, a, b),
            Dp2Op::Lsrv => self.b.bin(BinOp::ShrU, ty, a, b),
            Dp2Op::Asrv => self.b.bin(BinOp::ShrS, ty, a, b),
            Dp2Op::Rorv => self.b.bin(BinOp::Rotr, ty, a, b),
            Dp2Op::Crc32b
            | Dp2Op::Crc32h
            | Dp2Op::Crc32w
            | Dp2Op::Crc32x
            | Dp2Op::Crc32cb
            | Dp2Op::Crc32ch
            | Dp2Op::Crc32cw
            | Dp2Op::Crc32cx => {
                // CRC32 / CRC32C: Wn is the running CRC, updated with the low
                // 1/2/4/8 bytes of Xm (least significant byte first) and
                // written to Wd. The reflected bitwise form: XOR the byte in,
                // then shift 8 times against the reversed polynomial. wasm has
                // no carry-less multiply and no table to index, so this
                // expands inline -- a few hundred IR nodes on the widest form.
                let nbytes = match op {
                    Dp2Op::Crc32b | Dp2Op::Crc32cb => 1,
                    Dp2Op::Crc32h | Dp2Op::Crc32ch => 2,
                    Dp2Op::Crc32w | Dp2Op::Crc32cw => 4,
                    _ => 8,
                };
                // The polynomials in their reversed form, as the bit loop
                // shifts right: 0x04C11DB7 (IEEE) / 0x1EDC6F41 (Castagnoli).
                let poly = match op {
                    Dp2Op::Crc32cb | Dp2Op::Crc32ch | Dp2Op::Crc32cw | Dp2Op::Crc32cx => {
                        0x82f6_3b78u64
                    }
                    _ => 0xedb8_8320u64,
                };
                // Wn is 32-bit whatever sf says (on CRC32X only the source is
                // 64-bit), and Wd is written 32-bit, zero-extending into Xd.
                let mut acc = if ty == Ty::I64 { self.b.wrap(a) } else { a };
                let one = self.konst(Ty::I32, 1);
                let fmask = self.konst(Ty::I32, 0xff);
                let polyv = self.konst(Ty::I32, poly);
                for i in 0..nbytes {
                    let sh = self.konst(ty, 8 * i);
                    let shifted = self.b.bin(BinOp::ShrU, ty, b, sh);
                    let byte = if ty == Ty::I64 {
                        self.b.wrap(shifted)
                    } else {
                        shifted
                    };
                    let byte = self.b.bin(BinOp::And, Ty::I32, byte, fmask);
                    acc = self.b.bin(BinOp::Xor, Ty::I32, acc, byte);
                    for _ in 0..8 {
                        let lsb = self.b.bin(BinOp::And, Ty::I32, acc, one);
                        let mask = self.b.un(UnOp::Neg, Ty::I32, lsb);
                        let hi = self.b.bin(BinOp::ShrU, Ty::I32, acc, one);
                        let p = self.b.bin(BinOp::And, Ty::I32, polyv, mask);
                        acc = self.b.bin(BinOp::Xor, Ty::I32, hi, p);
                    }
                }
                self.wr_gp(rd, Sz::W, acc);
                return Flow::Next;
            }
            _ => return Flow::End(Term::Illegal { raw: self.raw, pc }),
        };
        self.wr_gp(rd, sz, v);
        Flow::Next
    }

    fn dataproc3(&mut self, sz: Sz, op: Dp3Op, rd: RegSel, rn: RegSel, rm: RegSel, ra: RegSel) {
        match op {
            Dp3Op::Madd | Dp3Op::Msub => {
                let ty = ty_of(sz);
                let a = self.rd_gp(rn, sz);
                let b = self.rd_gp(rm, sz);
                let acc = self.rd_gp(ra, sz);
                let prod = self.b.bin(BinOp::Mul, ty, a, b);
                let v = self.b.bin(
                    if op == Dp3Op::Madd {
                        BinOp::Add
                    } else {
                        BinOp::Sub
                    },
                    ty,
                    acc,
                    prod,
                );
                self.wr_gp(rd, sz, v);
            }
            Dp3Op::Smaddl | Dp3Op::Smsubl | Dp3Op::Umaddl | Dp3Op::Umsubl => {
                let signed = matches!(op, Dp3Op::Smaddl | Dp3Op::Smsubl);
                let sub = matches!(op, Dp3Op::Smsubl | Dp3Op::Umsubl);
                let a32 = self.rd_gp(rn, Sz::W);
                let b32 = self.rd_gp(rm, Sz::W);
                let a = self.b.extend(a32, signed);
                let b = self.b.extend(b32, signed);
                let acc = self.rd_gp(ra, Sz::X);
                let prod = self.b.bin(BinOp::Mul, Ty::I64, a, b);
                let v = self.b.bin(
                    if sub { BinOp::Sub } else { BinOp::Add },
                    Ty::I64,
                    acc,
                    prod,
                );
                self.wr_gp(rd, Sz::X, v);
            }
            Dp3Op::Smulh | Dp3Op::Umulh => {
                let signed = op == Dp3Op::Smulh;
                let a = self.rd_gp(rn, Sz::X);
                let b = self.rd_gp(rm, Sz::X);
                let v = self.mul_high(a, b, signed);
                self.wr_gp(rd, Sz::X, v);
            }
        }
    }

    /// High 64 bits of a 64x64 product, built from four 32-bit partial
    /// products — wasm has no widening multiply.
    fn mul_high(&mut self, a: Value, b: Value, signed: bool) -> Value {
        let ty = Ty::I64;
        let sh32 = self.konst(ty, 32);
        let lo_mask = self.konst(ty, 0xffff_ffff);

        let a_lo = self.b.bin(BinOp::And, ty, a, lo_mask);
        let b_lo = self.b.bin(BinOp::And, ty, b, lo_mask);
        // Unsigned high halves; the signed correction comes at the end.
        let a_hi = self.b.bin(BinOp::ShrU, ty, a, sh32);
        let b_hi = self.b.bin(BinOp::ShrU, ty, b, sh32);

        let ll = self.b.bin(BinOp::Mul, ty, a_lo, b_lo);
        let lh = self.b.bin(BinOp::Mul, ty, a_lo, b_hi);
        let hl = self.b.bin(BinOp::Mul, ty, a_hi, b_lo);
        let hh = self.b.bin(BinOp::Mul, ty, a_hi, b_hi);

        // mid = (ll >> 32) + (lh & 0xffffffff) + (hl & 0xffffffff)
        let ll_hi = self.b.bin(BinOp::ShrU, ty, ll, sh32);
        let lh_lo = self.b.bin(BinOp::And, ty, lh, lo_mask);
        let hl_lo = self.b.bin(BinOp::And, ty, hl, lo_mask);
        let m1 = self.b.bin(BinOp::Add, ty, ll_hi, lh_lo);
        let mid = self.b.bin(BinOp::Add, ty, m1, hl_lo);

        // high = hh + (lh >> 32) + (hl >> 32) + (mid >> 32)
        let lh_hi = self.b.bin(BinOp::ShrU, ty, lh, sh32);
        let hl_hi = self.b.bin(BinOp::ShrU, ty, hl, sh32);
        let mid_hi = self.b.bin(BinOp::ShrU, ty, mid, sh32);
        let h1 = self.b.bin(BinOp::Add, ty, hh, lh_hi);
        let h2 = self.b.bin(BinOp::Add, ty, h1, hl_hi);
        let mut high = self.b.bin(BinOp::Add, ty, h2, mid_hi);

        if signed {
            // Convert the unsigned high word to the signed one:
            // signed = unsigned - (a < 0 ? b : 0) - (b < 0 ? a : 0).
            let zero = self.konst(ty, 0);
            let a_neg = self.b.cmp(CmpOp::LtS, ty, a, zero);
            let b_neg = self.b.cmp(CmpOp::LtS, ty, b, zero);
            let ca = self.b.select(a_neg, b, zero, ty);
            let cb = self.b.select(b_neg, a, zero, ty);
            let t = self.b.bin(BinOp::Sub, ty, high, ca);
            high = self.b.bin(BinOp::Sub, ty, t, cb);
        }
        high
    }

    /// `stxr`/`stlxr`: succeed only if the local monitor still matches.
    ///
    /// Wasm has no exclusive-access primitive, so the monitor is emulated with
    /// a compare-and-swap against the value the matching load-exclusive
    /// observed. When the monitor does not match, the compare-and-swap is
    /// redirected to a scratch location so it cannot touch guest memory, and
    /// the instruction reports failure. Spurious *failure* is architecturally
    /// permitted; spurious success is not, which is why the redirect exists.
    fn store_exclusive(&mut self, sz: MemSz, rs: RegSel, rt: RegSel, rn: RegSel) {
        let ty = if sz == MemSz::D { Ty::I64 } else { Ty::I32 };
        let width = if sz == MemSz::D { Sz::X } else { Sz::W };
        let addr = self.rd_addr(rn);
        let value = self.rd_gp(rt, width);

        let mon_addr = self.b.read(GuestReg::MonAddr);
        let mon_size = self.b.read(GuestReg::MonSize);
        let want_size = self.b.const_i64(sz.bytes() as u64);
        let addr_ok = self.b.cmp(CmpOp::Eq, Ty::I64, mon_addr, addr);
        let size_ok = self.b.cmp(CmpOp::Eq, Ty::I64, mon_size, want_size);
        let armed = self.b.bin(BinOp::And, Ty::I32, addr_ok, size_ok);

        let scratch = self.b.const_i64(gex_abi::EXCLUSIVE_SCRATCH);
        let eff_addr = self.b.select(armed, addr, scratch, Ty::I64);

        let mon_value = self.b.read(GuestReg::MonValue);
        let expected = match ty {
            Ty::I32 => self.b.wrap(mon_value),
            _ => mon_value,
        };
        let old = self.b.def(Expr::CmpXchg {
            addr: eff_addr,
            expected,
            replacement: value,
            sz: mem_sz(sz),
            ty,
        });
        let swapped = self.b.cmp(CmpOp::Eq, ty, old, expected);
        let success = self.b.bin(BinOp::And, Ty::I32, armed, swapped);

        // Rs receives 0 on success and 1 on failure.
        let one = self.b.const_i32(1);
        let status = self.b.bin(BinOp::Xor, Ty::I32, success, one);
        self.wr_gp(rs, Sz::W, status);

        // The monitor is cleared either way.
        let zero = self.b.const_i64(0);
        self.b.write(GuestReg::MonSize, zero);
    }

    fn atomic_rmw(
        &mut self,
        sz: MemSz,
        op: IsaRmw,
        rs: RegSel,
        rt: RegSel,
        rn: RegSel,
        pc: u64,
    ) -> Flow {
        let ty = if sz == MemSz::D { Ty::I64 } else { Ty::I32 };
        let width = if sz == MemSz::D { Sz::X } else { Sz::W };
        let addr = self.rd_addr(rn);
        let operand = self.rd_gp(rs, width);
        let (rmw_op, val) = match op {
            IsaRmw::Add => (RmwOp::Add, operand),
            IsaRmw::Eor => (RmwOp::Xor, operand),
            IsaRmw::Set => (RmwOp::Or, operand),
            IsaRmw::Clr => {
                // LDCLR clears exactly the bits set in the operand.
                let inv = self.b.un(UnOp::Not, ty, operand);
                (RmwOp::And, inv)
            }
            IsaRmw::Swp => (RmwOp::Xchg, operand),
            // The min/max variants need a compare-and-swap retry loop, which
            // block-local GIR cannot express. They are vanishingly rare in
            // real userspace, so trap precisely rather than guess.
            IsaRmw::Smax | IsaRmw::Smin | IsaRmw::Umax | IsaRmw::Umin => {
                return Flow::End(Term::Illegal { raw: self.raw, pc })
            }
        };
        let old = self.b.def(Expr::Rmw {
            op: rmw_op,
            addr,
            val,
            sz: mem_sz(sz),
            ty,
        });
        self.wr_gp(rt, width, old);
        Flow::Next
    }

    fn sysreg_read(&mut self, sysreg: SysReg, pc: u64) -> Value {
        match sysreg {
            SysReg::TpidrEl0 => self.b.read(GuestReg::Tpidr),
            SysReg::TpidrroEl0 => self.b.read(GuestReg::TpidrRo),
            SysReg::Ctr => self.b.const_i64(gex_abi::CTR_EL0),
            SysReg::Cntfrq => self.b.const_i64(gex_abi::CNTFRQ),
            SysReg::Dczid => {
                // log2 of the block size in words. The "zeroing prohibited"
                // bit stays clear because DC ZVA is implemented.
                let words = gex_abi::DCZID_BLOCK_BYTES / 4;
                self.b.const_i64(words.trailing_zeros() as u64)
            }
            SysReg::Nzcv => {
                let n = self.b.read(GuestReg::N);
                let z = self.b.read(GuestReg::Z);
                let c = self.b.read(GuestReg::C);
                let v = self.b.read(GuestReg::V);
                let mut acc: Option<Value> = None;
                for (flag, shift) in [(n, 31u64), (z, 30), (c, 29), (v, 28)] {
                    let wide = self.b.extend(flag, false);
                    let s = self.b.const_i64(shift);
                    let placed = self.b.bin(BinOp::Shl, Ty::I64, wide, s);
                    acc = Some(match acc {
                        None => placed,
                        Some(a) => self.b.bin(BinOp::Or, Ty::I64, a, placed),
                    });
                }
                acc.expect("four flags")
            }
            // Everything else — including the virtual counter, which the
            // runtime serves from the shared time page — goes to the host.
            other => {
                let pcv = self.b.const_i64(pc);
                let id = self.b.const_i32(sysreg_code(other) as u32);
                self.b.def(Expr::HostCall {
                    func: HostFunc::SysRegRead,
                    args: vec![pcv, id],
                    ty: Ty::I64,
                })
            }
        }
    }

    fn sysreg_write(&mut self, sysreg: SysReg, v: Value, pc: u64) {
        match sysreg {
            SysReg::TpidrEl0 => self.b.write(GuestReg::Tpidr, v),
            SysReg::Nzcv => {
                self.flags = FlagSource::Unknown;
                let one = self.b.const_i64(1);
                for (shift, reg) in [
                    (31u64, GuestReg::N),
                    (30, GuestReg::Z),
                    (29, GuestReg::C),
                    (28, GuestReg::V),
                ] {
                    let s = self.b.const_i64(shift);
                    let sh = self.b.bin(BinOp::ShrU, Ty::I64, v, s);
                    let bit = self.b.bin(BinOp::And, Ty::I64, sh, one);
                    let b32 = self.b.wrap(bit);
                    self.b.write(reg, b32);
                }
            }
            // FPCR/FPSR writes go to the host so the fidelity policy
            // can notice a non-default rounding mode.
            other => {
                let pcv = self.b.const_i64(pc);
                let id = self.b.const_i32(sysreg_code(other) as u32);
                self.b.stmt(Stmt::HostCall {
                    func: HostFunc::SysRegWrite,
                    args: vec![pcv, id, v],
                });
            }
        }
    }
}

/// Byte pattern implementing `trn`/`uzp`/`zip` for an arrangement.
///
/// All six are fixed selections from the concatenation of the two sources,
/// so each becomes one shuffle. Indices 0-15 name the first operand and
/// 16-31 the second.
fn permute_pattern(op: VPermOp, arr: VArr) -> [u8; 16] {
    let lb = arr.lane_bytes;
    let n = arr.lanes();
    let mut lanes = [0u8; 16];
    // The 64-bit forms only fill the low half; the rest is masked away.
    for out in 0..n {
        let src_lane: u32 = match op {
            VPermOp::Zip1 => out / 2 + if out % 2 == 1 { n } else { 0 },
            VPermOp::Zip2 => n / 2 + out / 2 + if out % 2 == 1 { n } else { 0 },
            VPermOp::Uzp1 => out * 2,
            VPermOp::Uzp2 => out * 2 + 1,
            VPermOp::Trn1 => (out & !1) + if out % 2 == 1 { n } else { 0 },
            VPermOp::Trn2 => (out & !1) + 1 + if out % 2 == 1 { n } else { 0 },
        };
        // A lane index at or past `n` refers to the second operand, whose
        // bytes start at 16 regardless of the arrangement's width.
        let (reg_base, lane_in_reg) = if src_lane >= n {
            (16u32, src_lane - n)
        } else {
            (0, src_lane)
        };
        for byte in 0..lb {
            let idx = (out * lb + byte) as usize;
            if idx < 16 {
                lanes[idx] = (reg_base + lane_in_reg * lb + byte) as u8;
            }
        }
    }
    lanes
}

/// Even- and odd-lane gather patterns for pairwise addition.
///
/// `addp` treats the two sources as one 2n-lane vector and adds adjacent
/// pairs, which is a deinterleave followed by an ordinary lane-wise add.
fn pairwise_patterns(arr: VArr) -> ([u8; 16], [u8; 16]) {
    let lb = arr.lane_bytes;
    let n = arr.lanes();
    let mut evens = [0u8; 16];
    let mut odds = [0u8; 16];
    for out in 0..n {
        let src = out * 2;
        let (base, lane) = if src >= n { (16u32, src - n) } else { (0, src) };
        let (base2, lane2) = if src + 1 >= n {
            (16u32, src + 1 - n)
        } else {
            (0, src + 1)
        };
        for byte in 0..lb {
            let idx = (out * lb + byte) as usize;
            if idx < 16 {
                evens[idx] = (base + lane * lb + byte) as u8;
                odds[idx] = (base2 + lane2 * lb + byte) as u8;
            }
        }
    }
    (evens, odds)
}

/// Encode a system register identifier for the host interface. Modelled
/// registers get small negative codes; everything else passes through its
/// `op0:op1:CRn:CRm:op2` packing.
pub fn sysreg_code(r: SysReg) -> i32 {
    match r {
        SysReg::Other(v) => v as i32,
        SysReg::TpidrEl0 => -1,
        SysReg::TpidrroEl0 => -2,
        SysReg::Cntvct => -3,
        SysReg::Cntfrq => -4,
        SysReg::Ctr => -5,
        SysReg::Dczid => -6,
        SysReg::Midr => -7,
        SysReg::Fpcr => -8,
        SysReg::Fpsr => -9,
        SysReg::Nzcv => -10,
    }
}


/// `2^n` for the fixed-point scales, built from the exponent field rather
/// than computed.
///
/// `powi` would do, but it is not `const`-friendly and pulls in a libm call
/// for something that is one exact bit pattern. The range here is bounded by
/// the element width, so the exponent always fits.
fn exp2_exact(n: i32) -> f64 {
    // Bias 1023; the significand stays zero, which is what makes it exact.
    let biased = (n + 1023) as u64;
    f64::from_bits(biased << 52)
}
