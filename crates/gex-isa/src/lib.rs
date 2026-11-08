//! `gex-isa` — AArch64 A64 instruction decoder.
//!
//! Turns a 32-bit little-endian A64 encoding into a structured [`Inst`].
//! Invariants the rest of the compiler depends on:
//!
//! * **Total.** Every encoding maps to some `Inst`; the unrecognised become
//!   [`Inst::Unknown`] or [`Inst::Undefined`] (both → SIGILL). Never panics,
//!   never misdecodes into a wrong opcode.
//! * **Register roles resolved at decode time** into [`RegSel`], so the lifter
//!   never has to re-derive whether `Rn = 31` is the zero register or SP.
//! * **Immediates fully decoded** to final values, not raw fields.

#![forbid(unsafe_code)]

pub mod decode;
pub mod fmt;
pub mod slow;

pub use decode::decode;

/// A general-purpose register operand, with the meaning of encoding value 31
/// already resolved.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RegSel {
    /// `x0`–`x30` / `w0`–`w30`.
    Gp(u8),
    /// The zero register: reads as 0, writes discarded.
    Zr,
    /// The stack pointer.
    Sp,
}

impl RegSel {
    /// Decode a register field where 31 means the zero register.
    #[inline]
    pub fn gp(n: u32) -> RegSel {
        let n = (n & 31) as u8;
        if n == 31 {
            RegSel::Zr
        } else {
            RegSel::Gp(n)
        }
    }

    /// Decode a register field where 31 means the stack pointer.
    #[inline]
    pub fn sp(n: u32) -> RegSel {
        let n = (n & 31) as u8;
        if n == 31 {
            RegSel::Sp
        } else {
            RegSel::Gp(n)
        }
    }
}

/// A SIMD&FP register `v0`–`v31` (also `b/h/s/d/q` views of the same file).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct VReg(pub u8);

/// Operand width for integer data processing.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Sz {
    /// 32-bit (`w`) — results are zero-extended into the 64-bit register.
    W,
    /// 64-bit (`x`).
    X,
}

impl Sz {
    #[inline]
    pub fn from_sf(sf: u32) -> Sz {
        if sf & 1 != 0 {
            Sz::X
        } else {
            Sz::W
        }
    }
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            Sz::W => 32,
            Sz::X => 64,
        }
    }
    #[inline]
    pub fn is64(self) -> bool {
        matches!(self, Sz::X)
    }
    /// Mask of valid bits for this width.
    #[inline]
    pub fn mask(self) -> u64 {
        match self {
            Sz::W => 0xffff_ffff,
            Sz::X => u64::MAX,
        }
    }
}

/// Condition code (`b.<cond>`, `csel`, `ccmp`, …).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Cond(pub u8);

impl Cond {
    pub const EQ: Cond = Cond(0x0);
    pub const NE: Cond = Cond(0x1);
    pub const AL: Cond = Cond(0xe);

    /// The condition with its low bit flipped — i.e. the logical inverse,
    /// except that `AL`/`NV` (0b111x) both mean "always" and are not inverses.
    #[inline]
    pub fn invert(self) -> Cond {
        Cond(self.0 ^ 1)
    }
    /// True for `AL` and `NV`, which the architecture defines as always-true.
    #[inline]
    pub fn is_always(self) -> bool {
        self.0 >= 0b1110
    }
    pub fn name(self) -> &'static str {
        const N: [&str; 16] = [
            "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
            "al", "nv",
        ];
        N[(self.0 & 15) as usize]
    }
}

/// Shift applied to the second source of a data-processing-register op.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

impl ShiftType {
    #[inline]
    pub fn from_bits(b: u32) -> ShiftType {
        match b & 3 {
            0 => ShiftType::Lsl,
            1 => ShiftType::Lsr,
            2 => ShiftType::Asr,
            _ => ShiftType::Ror,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ShiftType::Lsl => "lsl",
            ShiftType::Lsr => "lsr",
            ShiftType::Asr => "asr",
            ShiftType::Ror => "ror",
        }
    }
}

/// Register extension applied by `add/sub (extended)` and register-offset
/// addressing modes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

impl Extend {
    #[inline]
    pub fn from_bits(b: u32) -> Extend {
        match b & 7 {
            0 => Extend::Uxtb,
            1 => Extend::Uxth,
            2 => Extend::Uxtw,
            3 => Extend::Uxtx,
            4 => Extend::Sxtb,
            5 => Extend::Sxth,
            6 => Extend::Sxtw,
            _ => Extend::Sxtx,
        }
    }
    /// Number of source bits consumed before extension.
    #[inline]
    pub fn src_bits(self) -> u32 {
        match self {
            Extend::Uxtb | Extend::Sxtb => 8,
            Extend::Uxth | Extend::Sxth => 16,
            Extend::Uxtw | Extend::Sxtw => 32,
            Extend::Uxtx | Extend::Sxtx => 64,
        }
    }
    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            Extend::Sxtb | Extend::Sxth | Extend::Sxtw | Extend::Sxtx
        )
    }
    pub fn name(self) -> &'static str {
        match self {
            Extend::Uxtb => "uxtb",
            Extend::Uxth => "uxth",
            Extend::Uxtw => "uxtw",
            Extend::Uxtx => "uxtx",
            Extend::Sxtb => "sxtb",
            Extend::Sxth => "sxth",
            Extend::Sxtw => "sxtw",
            Extend::Sxtx => "sxtx",
        }
    }
}

/// Add or subtract.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum AddOp {
    Add,
    Sub,
}

/// Bitwise operation family.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum LogOp {
    And,
    Orr,
    Eor,
}

/// `movz` / `movn` / `movk`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MoveOp {
    /// Zero the register, then insert the immediate.
    Movz,
    /// Insert the immediate, then invert every bit.
    Movn,
    /// Insert the immediate, keeping the other bits.
    Movk,
}

/// Bitfield move variant.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BfOp {
    /// Signed: replicate the top copied bit leftwards.
    Sbfm,
    /// Unsigned: zero outside the copied field.
    Ubfm,
    /// Leave bits outside the field untouched.
    Bfm,
}

/// One-source integer data processing.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dp1Op {
    Rbit,
    Rev16,
    Rev32,
    Rev,
    Clz,
    Cls,
}

/// Two-source integer data processing.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dp2Op {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
    Crc32b,
    Crc32h,
    Crc32w,
    Crc32x,
    Crc32cb,
    Crc32ch,
    Crc32cw,
    Crc32cx,
}

/// Three-source integer data processing (multiply-accumulate family).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dp3Op {
    /// `rd = ra + rn * rm`
    Madd,
    /// `rd = ra - rn * rm`
    Msub,
    /// `rd = ra + sext(wn) * sext(wm)`
    Smaddl,
    Smsubl,
    Umaddl,
    Umsubl,
    /// High 64 bits of the signed 64x64 product.
    Smulh,
    Umulh,
}

/// Memory access size in bytes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MemSz {
    B = 1,
    H = 2,
    W = 4,
    D = 8,
    Q = 16,
}

impl MemSz {
    #[inline]
    pub fn bytes(self) -> u32 {
        self as u32
    }
    #[inline]
    pub fn log2(self) -> u32 {
        match self {
            MemSz::B => 0,
            MemSz::H => 1,
            MemSz::W => 2,
            MemSz::D => 3,
            MemSz::Q => 4,
        }
    }
    #[inline]
    pub fn from_log2(l: u32) -> MemSz {
        match l & 7 {
            0 => MemSz::B,
            1 => MemSz::H,
            2 => MemSz::W,
            3 => MemSz::D,
            _ => MemSz::Q,
        }
    }
}

/// How a load/store computes its address and whether it writes the base back.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum AddrMode {
    /// `[rn, #off]` — offset already scaled by the access size.
    Offset { rn: RegSel, off: i64 },
    /// `[rn], #off` — access at `rn`, then `rn += off`.
    PostIndex { rn: RegSel, off: i64 },
    /// `[rn], rm` — access at `rn`, then `rn += rm`. Only the SIMD structure
    /// loads and stores use it, where the stride is not a constant.
    PostIndexReg { rn: RegSel, rm: RegSel },
    /// `[rn, #off]!` — `rn += off`, then access at `rn`.
    PreIndex { rn: RegSel, off: i64 },
    /// `[rn, rm, <ext> #amount]`.
    RegOffset {
        rn: RegSel,
        rm: RegSel,
        ext: Extend,
        amount: u32,
    },
    /// PC-relative literal: the final absolute guest address.
    Literal { addr: u64 },
}

/// Memory ordering requested by a load/store.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ordering {
    /// Plain access, no ordering.
    None,
    /// Load-acquire.
    Acquire,
    /// Store-release.
    Release,
}

/// Barrier flavour.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BarrierKind {
    /// Data memory barrier.
    Dmb,
    /// Data synchronization barrier.
    Dsb,
    /// Instruction synchronization barrier — also our translation-invalidation
    /// hook for self-modifying guest code.
    Isb,
    /// Speculation barrier.
    Sb,
}

/// Atomic read-modify-write operation (FEAT_LSE).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RmwOp {
    Add,
    Clr,
    Eor,
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
    /// Plain swap.
    Swp,
}

/// A decoded A64 instruction.
///
/// Variants carry fully-resolved operands: the lifter should never need to
/// re-examine raw encoding bits.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Inst {
    /// Not recognised by this decoder — a translator gap, not necessarily an
    /// invalid encoding. Translating one is a hard error; executing one at
    /// runtime raises SIGILL.
    Unknown { raw: u32 },
    /// Architecturally UNDEFINED, or `udf`. Executing raises SIGILL.
    Undefined { raw: u32 },

    /// `nop`, `yield`, `wfe`, … — architecturally a hint, semantically a no-op
    /// for us.
    Hint { imm: u32 },

    /// `adr rd, label` — `target` is the resolved absolute guest address.
    Adr { rd: RegSel, target: u64 },
    /// `adrp rd, label` — `target` is the resolved absolute *page* address
    /// (`(pc & !0xfff) + imm`), already 4 KiB aligned.
    Adrp { rd: RegSel, target: u64 },

    /// `add`/`sub`/`adds`/`subs` (immediate). `rd`/`rn` are SP-form unless
    /// flags are set.
    AddSubImm {
        sz: Sz,
        op: AddOp,
        set_flags: bool,
        rd: RegSel,
        rn: RegSel,
        imm: u64,
    },
    /// `and`/`orr`/`eor`/`ands` (bitmask immediate), immediate already expanded.
    LogicalImm {
        sz: Sz,
        op: LogOp,
        set_flags: bool,
        rd: RegSel,
        rn: RegSel,
        imm: u64,
    },
    /// `movz`/`movn`/`movk`.
    MoveWide {
        sz: Sz,
        op: MoveOp,
        rd: RegSel,
        imm16: u64,
        shift: u32,
    },
    /// `sbfm`/`ubfm`/`bfm` — covers `lsl`/`lsr`/`asr`/`ubfx`/`sbfx`/`uxtb`/…
    Bitfield {
        sz: Sz,
        op: BfOp,
        rd: RegSel,
        rn: RegSel,
        immr: u32,
        imms: u32,
    },
    /// `extr rd, rn, rm, #lsb` (and its `ror` alias).
    Extract {
        sz: Sz,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        lsb: u32,
    },

    /// `add`/`sub`/`adds`/`subs` (shifted register).
    AddSubShifted {
        sz: Sz,
        op: AddOp,
        set_flags: bool,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        shift: ShiftType,
        amount: u32,
    },
    /// `add`/`sub`/`adds`/`subs` (extended register). `rn` is SP-form.
    AddSubExtended {
        sz: Sz,
        op: AddOp,
        set_flags: bool,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        ext: Extend,
        amount: u32,
    },
    /// `adc`/`sbc`/`adcs`/`sbcs`.
    AddSubCarry {
        sz: Sz,
        op: AddOp,
        set_flags: bool,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
    },
    /// `and`/`bic`/`orr`/`orn`/`eor`/`eon`/`ands`/`bics` (shifted register).
    LogicalShifted {
        sz: Sz,
        op: LogOp,
        set_flags: bool,
        /// `bic`/`orn`/`eon`: invert the shifted second operand.
        invert: bool,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        shift: ShiftType,
        amount: u32,
    },
    /// `csel`/`csinc`/`csinv`/`csneg` — `rd = cond ? rn : f(rm)`.
    CondSelect {
        sz: Sz,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        cond: Cond,
        /// Add 1 to the else-value (`csinc`, `cset`, `cinc`).
        else_inc: bool,
        /// Bitwise-invert the else-value (`csinv`/`csneg`, `csetm`).
        else_inv: bool,
    },
    /// `ccmp`/`ccmn` (register or immediate second operand).
    CondCompare {
        sz: Sz,
        /// `ccmp` subtracts; `ccmn` adds.
        op: AddOp,
        rn: RegSel,
        /// Register operand, or `None` when `imm` is used.
        rm: Option<RegSel>,
        imm: u64,
        /// Flags to install when the condition is false.
        nzcv: u8,
        cond: Cond,
    },
    /// One-source data processing.
    DataProc1 {
        sz: Sz,
        op: Dp1Op,
        rd: RegSel,
        rn: RegSel,
    },
    /// Two-source data processing.
    DataProc2 {
        sz: Sz,
        op: Dp2Op,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
    },
    /// Three-source data processing.
    DataProc3 {
        sz: Sz,
        op: Dp3Op,
        rd: RegSel,
        rn: RegSel,
        rm: RegSel,
        ra: RegSel,
    },

    /// `b` / `bl` to an absolute guest address.
    Branch { target: u64, link: bool },
    /// `b.<cond>` to an absolute guest address.
    BranchCond { cond: Cond, target: u64 },
    /// `cbz` / `cbnz`.
    CompareBranch {
        sz: Sz,
        rt: RegSel,
        target: u64,
        /// `cbnz` when true.
        nonzero: bool,
    },
    /// `tbz` / `tbnz`.
    TestBranch {
        rt: RegSel,
        bit: u32,
        target: u64,
        /// `tbnz` when true.
        nonzero: bool,
    },
    /// `br` / `blr` / `ret` — indirect branch through a register.
    BranchReg {
        rn: RegSel,
        link: bool,
        /// `ret`: hint that this pairs with an earlier `bl`.
        is_ret: bool,
    },

    /// `svc #imm` — the syscall entry point.
    Svc { imm: u32 },
    /// `brk #imm` — debug breakpoint, raises SIGTRAP.
    Brk { imm: u32 },

    /// `mrs rt, <sysreg>` — read a system register.
    SysRegRead { rt: RegSel, sysreg: SysReg },
    /// `msr <sysreg>, rt` — write a system register.
    SysRegWrite { rt: RegSel, sysreg: SysReg },
    /// `dc`/`ic`/`at`/`tlbi` — cache and TLB maintenance.
    SysMaintenance { op: SysMaintOp, rt: RegSel },
    /// `dmb`/`dsb`/`isb`/`sb`.
    Barrier { kind: BarrierKind, option: u32 },
    /// `clrex` — clear the local exclusive monitor.
    Clrex,

    /// A plain (non-atomic) load or store of an integer register.
    Load {
        sz: MemSz,
        /// Destination register width; `LDRSW` loads 4 bytes into an X.
        dest: Sz,
        /// Sign-extend the loaded value to `dest`.
        signed: bool,
        rt: RegSel,
        addr: AddrMode,
        order: Ordering,
    },
    Store {
        sz: MemSz,
        rt: RegSel,
        addr: AddrMode,
        order: Ordering,
    },
    /// A SIMD&FP load of `bytes` bytes into the low lanes of `vt`, zeroing
    /// the rest of the register.
    VLoad {
        bytes: u32,
        vt: VReg,
        addr: AddrMode,
    },
    /// A SIMD&FP store of the low `bytes` bytes of `vt`.
    VStore {
        bytes: u32,
        vt: VReg,
        addr: AddrMode,
    },
    /// `ldp`/`stp` of two SIMD&FP registers.
    VLoadPair {
        bytes: u32,
        vt: VReg,
        vt2: VReg,
        addr: AddrMode,
    },
    VStorePair {
        bytes: u32,
        vt: VReg,
        vt2: VReg,
        addr: AddrMode,
    },
    /// `dup vd.<T>, rn` — replicate a general register across lanes.
    VDupGeneral {
        vd: VReg,
        rn: RegSel,
        lane_bytes: u32,
        /// False selects the 64-bit form, which zeroes the upper half.
        q: bool,
    },
    /// `movi`/`mvni`, immediate already expanded to the full 128-bit value.
    /// `inverted` records which it was — needed only so the disassembler can
    /// tell them apart.
    VMoveImm {
        vd: VReg,
        value: u128,
        inverted: bool,
        /// Set for the `fmov` form (`cmode` 1111), which is spelled as a
        /// floating-point immediate even though it moves the same bits.
        fp_lane_bytes: Option<u32>,
        /// The encoding fields, kept so the disassembler can reproduce the
        /// arrangement and shift the assembler wrote rather than the expanded
        /// register value.
        cmode: u8,
        imm8: u8,
        q: bool,
        /// The `op` bit. With `cmode` 1110 it picks the 64-bit form, whose
        /// immediate is a byte mask rather than a lane value.
        op: bool,
    },
    /// Scalar floating-point, one source.
    FpDataProc1 {
        sz: FpSz,
        op: FpOp1,
        vd: VReg,
        vn: VReg,
    },
    /// Scalar floating-point, two sources.
    FpDataProc2 {
        sz: FpSz,
        op: FpOp2,
        vd: VReg,
        vn: VReg,
        vm: VReg,
    },
    /// Fused multiply-add family: `vd = ±(vn * vm) ± va`.
    FpDataProc3 {
        sz: FpSz,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        va: VReg,
        /// Negate the product.
        neg_product: bool,
        /// Negate the addend.
        neg_addend: bool,
    },
    /// `fcmp`/`fcmpe`, optionally against zero.
    FpCompare {
        sz: FpSz,
        vn: VReg,
        /// `None` compares against +0.0.
        vm: Option<VReg>,
    },
    /// `fcsel`.
    FpCondSelect {
        sz: FpSz,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        cond: Cond,
    },
    /// `fmov vd, #imm`, with the immediate already expanded to its bit
    /// pattern at the given precision.
    FpMoveImm { sz: FpSz, vd: VReg, bits: u64 },
    /// `fcvt` between single and double.
    FpConvertPrecision {
        from: FpSz,
        to: FpSz,
        vd: VReg,
        vn: VReg,
    },
    /// `scvtf`/`ucvtf`: integer register to float register.
    /// `scvtf`/`ucvtf`. With `fbits` non-zero this is the fixed-point form:
    /// the integer is read as a value scaled by `2^fbits`.
    FpFromInt {
        sz: FpSz,
        /// Width of the source integer register.
        int_sz: Sz,
        signed: bool,
        vd: VReg,
        rn: RegSel,
        /// Fractional bits for the fixed-point form; zero for the plain one.
        fbits: u32,
    },
    /// `fcvtzs`/`fcvtzu`/`fcvtns`/…: float register to integer register.
    FpToInt {
        sz: FpSz,
        int_sz: Sz,
        signed: bool,
        round: FpRound,
        rd: RegSel,
        vn: VReg,
        /// Fractional bits for the fixed-point form; zero for the plain one.
        fbits: u32,
    },
    /// `fmov` between a general register and a float register.
    FpMoveGeneral {
        sz: FpSz,
        /// True moves general -> float; false moves float -> general.
        to_vector: bool,
        rd: RegSel,
        vd: VReg,
    },

    /// `fmov` between a general register and the *upper* half of a 128-bit
    /// vector register — how a 128-bit long double is assembled.
    FpMoveTopHalf {
        /// True moves general -> vector lane 1.
        to_vector: bool,
        rd: RegSel,
        vd: VReg,
    },

    /// `umov`/`smov`/`ins (general)`: move one lane between a `v` register
    /// and a general register.
    VMoveLane {
        /// True inserts into the vector; false extracts from it.
        to_vector: bool,
        /// Sign-extend on extraction (`smov`).
        signed: bool,
        lane_bytes: u32,
        lane: u32,
        /// Width of the general register.
        int_sz: Sz,
        rd: RegSel,
        vd: VReg,
    },

    /// `ld1`/`st1` with one to four consecutive registers: `regs` back-to-back
    /// vector accesses, since element size does not change which bytes move.
    VLoadMulti {
        regs: u8,
        vt: VReg,
        /// Bytes per register: 8 for the 64-bit form, 16 for the 128-bit one.
        bytes: u32,
        addr: AddrMode,
    },
    VStoreMulti {
        regs: u8,
        vt: VReg,
        bytes: u32,
        addr: AddrMode,
    },
    /// `ld1`/`st1` on a single lane, and `ld1r`, which broadcasts instead.
    /// Also the interleaved single-lane forms `ld2`/`st2`/`ld3`/`st3`/`ld4`/
    /// `st4` and the replicating `ld2r`/`ld3r`/`ld4r`: `regs` consecutive
    /// registers, each contributing (or receiving) one lane at consecutive
    /// `lane_bytes` offsets from the address.
    VLoadStoreLane {
        load: bool,
        /// `ld1r`/`ld2r`/…: fill every lane rather than writing one.
        replicate: bool,
        /// How many consecutive registers the transfer touches: 1 for `ld1`/
        /// `st1`/`ld1r`, up to 4 for `ld4`/`st4`/`ld4r`.
        regs: u8,
        lane_bytes: u32,
        /// Which lane; meaningless when `replicate`.
        index: u32,
        /// Only affects the replicating forms, which must zero the upper half
        /// when clear.
        q: bool,
        vt: VReg,
        addr: AddrMode,
    },
    /// `ld2`/`st2`: two registers' worth of memory, de-interleaved on the way
    /// in and re-interleaved on the way out.
    VLoadStore2 {
        load: bool,
        lane_bytes: u32,
        /// Bytes per register: 8 or 16.
        bytes: u32,
        vt: VReg,
        addr: AddrMode,
    },
    /// `ld3`/`st3`/`ld4`/`st4` — three or four registers interleaved in
    /// memory. Like [`Inst::VLoadStore2`] but lowered separately, since more
    /// than two registers do not fit a single shuffle.
    VLoadStoreN {
        load: bool,
        /// How many registers are interleaved: 3 or 4.
        regs: u8,
        lane_bytes: u32,
        /// Bytes per register: 8 or 16.
        bytes: u32,
        vt: VReg,
        addr: AddrMode,
    },
    /// `cnt` — per-byte population count.
    VCnt { vd: VReg, vn: VReg, q: bool },
    /// `bsl`/`bit`/`bif`.
    VSelect {
        kind: VSelKind,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        q: bool,
    },
    /// Lane-wise operation on two vector sources.
    /// `orr` and `bic` with a modified immediate, which read the destination
    /// rather than replacing it.
    VLogicalImm {
        vd: VReg,
        value: u128,
        /// `bic`: clear the bits in `value` rather than setting them.
        clear: bool,
        q: bool,
    },
    /// Two vectors of one lane width producing one of half that width:
    /// `addhn`, `subhn` and their rounding forms.
    VNarrow {
        op: VNarrowOp,
        /// The width of a *result* lane, in bytes. The sources are twice
        /// this.
        dst_bytes: u32,
        /// Write the top half of the destination, leaving the bottom alone --
        /// the `addhn2` spelling.
        upper: bool,
        vd: VReg,
        vn: VReg,
        vm: VReg,
    },
    VThreeSame {
        op: VOp3,
        arr: VArr,
        vd: VReg,
        vn: VReg,
        vm: VReg,
    },
    /// Lane-wise operation on one vector source.
    VTwoMisc {
        op: VOp2,
        arr: VArr,
        vd: VReg,
        vn: VReg,
    },
    /// `trn`/`uzp`/`zip`.
    VPermute {
        op: VPermOp,
        arr: VArr,
        vd: VReg,
        vn: VReg,
        vm: VReg,
    },
    /// `ext vd, vn, vm, #index` — a byte-granular concatenation window.
    VExt {
        vd: VReg,
        vn: VReg,
        vm: VReg,
        index: u32,
        q: bool,
    },
    /// `tbl`/`tbx` with one to four consecutive table registers.
    VTableLookup {
        vd: VReg,
        vn: VReg,
        regs: u8,
        vm: VReg,
        q: bool,
        /// `tbx` leaves out-of-range lanes unchanged instead of zeroing them.
        is_tbx: bool,
    },
    /// `dup vd, vn.<T>[index]`, and the scalar form binutils spells `mov`,
    /// which copies the one lane and zeroes the rest of the register.
    VDupElement {
        vd: VReg,
        vn: VReg,
        lane_bytes: u32,
        index: u32,
        q: bool,
        scalar: bool,
    },
    /// `ins vd.<T>[dst], vn.<T>[src]`.
    VInsElement {
        vd: VReg,
        vn: VReg,
        lane_bytes: u32,
        dst_index: u32,
        src_index: u32,
    },
    /// Shift by an immediate.
    VShiftImm {
        op: VShiftImmOp,
        arr: VArr,
        vd: VReg,
        vn: VReg,
        shift: u32,
    },
    /// The by-element forms: `fmla v0.4s, v1.4s, v2.s[3]` and friends, where
    /// one lane of `vm` is broadcast across the multiply. Encoded separately
    /// from the three-same group but semantically identical once the lane is
    /// splatted, so it reuses `VOp3`.
    VByElement {
        op: VOp3,
        arr: VArr,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        index: u32,
    },
    /// The three-different group: `smull`, `uaddw`, `saddl` and friends,
    /// whose destination lanes are twice the width of their sources'.
    VWiden {
        op: VWidenOp,
        /// The *source* arrangement. The destination has half as many lanes,
        /// each twice as wide.
        arr: VArr,
        /// Read the upper half of the narrow sources (the `2` suffix).
        upper: bool,
        signed: bool,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        /// The by-element forms (`umlal v0.4s, v1.4h, v2.h[3]`) broadcast one
        /// lane of `vm` across the multiply; `Some(lane)` selects it, `None` is
        /// the ordinary vector form.
        index: Option<u32>,
    },
    /// `fcvtl`/`fcvtn`: change floating-point precision across a vector.
    VFpCvtPrecision {
        /// Widen (`fcvtl`) rather than narrow (`fcvtn`).
        widen: bool,
        /// The narrow side's lane width: 2 for half, 4 for single.
        narrow_bytes: u32,
        /// Operate on the upper half (the `2` suffix).
        upper: bool,
        vd: VReg,
        vn: VReg,
    },
    /// `fccmp`/`fccmpe`: conditional floating-point compare.
    FpCondCompare {
        sz: FpSz,
        vn: VReg,
        vm: VReg,
        nzcv: u8,
        cond: Cond,
        /// The `e` form, which signals on a quiet NaN too. gex does not model
        /// floating-point exceptions, so this only changes the mnemonic.
        signalling: bool,
    },
    /// Reduce every lane into the destination's lowest lane.
    VAcross {
        op: VAcrossOp,
        arr: VArr,
        vd: VReg,
        vn: VReg,
    },

    /// Bitwise operation on two vectors. `mov vd, vn` is `orr vd, vn, vn`.
    VLogical {
        op: VLogOp,
        vd: VReg,
        vn: VReg,
        vm: VReg,
        q: bool,
    },

    /// `ldp`/`ldpsw`.
    LoadPair {
        sz: MemSz,
        signed: bool,
        rt: RegSel,
        rt2: RegSel,
        addr: AddrMode,
    },
    /// `stp`.
    StorePair {
        sz: MemSz,
        rt: RegSel,
        rt2: RegSel,
        addr: AddrMode,
    },
    /// `stg`/`stzg`/`st2g`/`stz2g` — store allocation tags. Tags are not
    /// kept, so what remains is the base writeback of the pre- and
    /// post-index forms, and for the Z forms zeroing `zero` bytes (16 or
    /// 32, one or two granules) at the address.
    TagStore { addr: AddrMode, zero: u32 },
    /// `prfm` — prefetch; a no-op for us but must not decode as Unknown.
    /// `hint` is the `Rt` field: type, cache level and policy.
    Prefetch { addr: AddrMode, hint: u8 },

    /// `ldxr`/`ldaxr` — load-exclusive, arms the local monitor.
    LoadExclusive {
        sz: MemSz,
        rt: RegSel,
        rn: RegSel,
        order: Ordering,
    },
    /// `stxr`/`stlxr` — store-exclusive; `rs` receives 0 on success.
    StoreExclusive {
        sz: MemSz,
        rs: RegSel,
        rt: RegSel,
        rn: RegSel,
        order: Ordering,
    },
    /// `cas`/`casa`/`casl`/`casal` — compare and swap (FEAT_LSE).
    CompareSwap {
        sz: MemSz,
        rs: RegSel,
        rt: RegSel,
        rn: RegSel,
        acquire: bool,
        release: bool,
    },
    /// `casp`/`caspa`/`caspal`/`caspl` — compare and swap a *pair* of
    /// registers against a pair in memory (FEAT_LSE). `rs` is the compare
    /// pair `rs:rs+1`, which receives the pair read from memory; `rt` is
    /// the swap pair `rt:rt+1`, written when the pair in memory matches.
    /// `sz` is the size of each element. The encoding keeps both pair base
    /// registers even, so `+1` never wraps past 31.
    CompareSwapPair {
        sz: MemSz,
        rs: RegSel,
        rt: RegSel,
        rn: RegSel,
        acquire: bool,
        release: bool,
    },
    /// `ldadd`/`ldset`/`swp`/… — atomic read-modify-write (FEAT_LSE).
    AtomicRmw {
        sz: MemSz,
        op: RmwOp,
        rs: RegSel,
        rt: RegSel,
        rn: RegSel,
        acquire: bool,
        release: bool,
    },
}

/// Scalar floating-point precision.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpSz {
    /// Single precision (`s` registers).
    S,
    /// Double precision (`d` registers).
    D,
}

impl FpSz {
    pub fn is_double(self) -> bool {
        matches!(self, FpSz::D)
    }
    pub fn bytes(self) -> u32 {
        match self {
            FpSz::S => 4,
            FpSz::D => 8,
        }
    }
}

/// Scalar floating-point one-source operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpOp1 {
    Mov,
    Abs,
    Neg,
    Sqrt,
    RoundNearest,
    RoundPlus,
    RoundMinus,
    RoundZero,
    /// Round to nearest, ties away from zero.
    RoundAway,
    /// Round using the current FPCR mode, raising inexact.
    RoundExact,
    /// Round using the current FPCR mode.
    RoundCurrent,
}

/// Scalar floating-point two-source operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpOp2 {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min,
    MaxNum,
    MinNum,
    /// Negated multiply.
    NMul,
}

/// How a float-to-integer conversion rounds.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpRound {
    /// Round to nearest, ties to even.
    Nearest,
    /// Round toward +infinity.
    Plus,
    /// Round toward -infinity.
    Minus,
    /// Round toward zero.
    Zero,
    /// Round to nearest, ties away from zero.
    Away,
}

/// A NEON operand arrangement: lane width plus whether the operation is on
/// the full 128-bit register or only its low half.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct VArr {
    /// Lane width in bytes.
    pub lane_bytes: u32,
    /// False selects the 64-bit form, which zeroes the register's upper half.
    pub q: bool,
    /// True when the lanes hold floating-point values.
    pub float: bool,
    /// True for the scalar forms, which operate on lane 0 alone and zero
    /// everything above it.
    pub scalar: bool,
}

impl VArr {
    pub fn lanes(self) -> u32 {
        if self.scalar {
            return 1;
        }
        (if self.q { 16 } else { 8 }) / self.lane_bytes
    }

    /// The element letter: `b`, `h`, `s` or `d`.
    pub fn letter(self) -> char {
        match self.lane_bytes {
            1 => 'b',
            2 => 'h',
            4 => 's',
            _ => 'd',
        }
    }

    /// The arrangement suffix binutils prints, e.g. `4s`. A scalar form has
    /// no arrangement, only an element width.
    pub fn suffix(self) -> String {
        if self.scalar {
            return self.letter().to_string();
        }
        format!("{}{}", self.lanes(), self.letter())
    }
}

/// Lane-wise NEON operations with two vector sources.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VNarrowOp {
    AddHn,
    RAddHn,
    SubHn,
    RSubHn,
}

/// Three-operand lane-wise operations on equal-width vectors.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VOp3 {
    Add,
    Sub,
    Mul,
    /// Multiply-accumulate into the destination.
    Mla,
    Mls,
    SMax,
    SMin,
    UMax,
    UMin,
    SqAdd,
    UqAdd,
    SqSub,
    UqSub,
    /// Saturating doubling multiply returning the high half: `(2 * a * b) >> esize`,
    /// and `sqrdmulh`'s rounding form, `(2 * a * b + (1 << (esize - 1))) >> esize`.
    /// Halfword and word lanes only; only `MIN * MIN` saturates.
    SqDmulh,
    SqRdmulh,
    SHAdd,
    UHAdd,
    SrHAdd,
    UrHAdd,
    SHSub,
    UHSub,
    CmEq,
    CmGt,
    CmGe,
    CmHi,
    CmHs,
    /// Bitwise test: lanes where `a & b` is non-zero become all-ones.
    CmTst,
    /// Pairwise add of the concatenation of both operands.
    AddP,
    /// Pairwise maximum and minimum.
    SMaxP,
    UMaxP,
    SMinP,
    UMinP,
    /// Shift left by a signed per-lane amount; a negative amount shifts
    /// right. The rounding and saturating variants are not here.
    SShl,
    UShl,
    FAdd,
    FSub,
    FMul,
    /// `fmulx`: like `fmul` except that zero times infinity is two.
    FMulx,
    /// `|a - b|` per lane.
    FAbd,
    FDiv,
    /// `frecps`: a Newton step for the reciprocal estimate, `2 - a*b`.
    FRecps,
    /// `frsqrts`: a Newton step for the reciprocal-sqrt estimate, `(3 - a*b)/2`.
    FRSqrts,
    FMax,
    FMin,
    FMaxNm,
    FMinNm,
    FAddP,
    /// `fmaxp`/`fminp`: NaN-propagating maximum and minimum of adjacent lanes.
    FMaxP,
    FMinP,
    /// `facge`/`facgt`: compare absolute values.
    FAcGe,
    FAcGt,
    FMla,
    FMls,
    FCmEq,
    FCmGe,
    FCmGt,
    /// Absolute difference.
    SAbd,
    UAbd,
}

/// Lane-wise NEON operations with one vector source.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VOp2 {
    Neg,
    Abs,
    Not,
    Cnt,
    Clz,
    Cls,
    /// `saddlp`/`uaddlp` — add adjacent lanes into lanes of twice the width.
    /// The arrangement names the *destination*, so the source lanes are half
    /// as wide and there are twice as many.
    Saddlp,
    Uaddlp,
    /// `sadalp`/`uadalp` — the same, accumulated into the destination.
    Sadalp,
    Uadalp,
    Rbit,
    Rev16,
    Rev32,
    Rev64,
    FNeg,
    FAbs,
    FSqrt,
    /// Reciprocal estimate, `~1/x`. gex lifts it as the exact `1/x`: code that
    /// refines it with `frecps` converges the same, and code that uses it raw
    /// gets better precision than the hardware estimate.
    FRecpe,
    /// Reciprocal square-root estimate, `~1/sqrt(x)`; lifted as exact
    /// `1/sqrt(x)` for the same reason. libpixman/Qt geometry math reach it.
    FRSqrtE,
    FrintN,
    FrintP,
    FrintM,
    FrintZ,
    /// Round to nearest, ties away from zero.
    FrintA,
    /// Round using the current FPCR mode, raising inexact.
    FrintX,
    /// Round using the current FPCR mode.
    FrintI,
    /// Compare against zero.
    CmEqZero,
    CmGtZero,
    CmGeZero,
    CmLtZero,
    CmLeZero,
    FCmEqZero,
    FCmGtZero,
    FCmGeZero,
    FCmLtZero,
    FCmLeZero,
    /// Float to signed/unsigned integer. The suffix names the rounding mode:
    /// Z toward zero, N to nearest, P toward +inf, M toward -inf, A to
    /// nearest with ties away.
    FcvtZs,
    FcvtZu,
    FcvtNs,
    FcvtNu,
    FcvtPs,
    FcvtPu,
    FcvtMs,
    FcvtMu,
    FcvtAs,
    FcvtAu,
    /// Integer to float.
    Scvtf,
    Ucvtf,
    /// Narrow each lane to half its width, taking the low half.
    Xtn,
    /// Narrow to half width, saturating.
    SqXtn,
    UqXtn,
    /// Narrow signed lanes to unsigned lanes, saturating negative values to 0.
    SqXtun,
    /// Widen each lane to twice its width and shift left by the original
    /// width (`shll`/`shll2`); the arrangement names the source lanes.
    Shll,
}

/// The three-different group's operations, named for what they do to the
/// widened lanes rather than for each signed/unsigned spelling.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VWidenOp {
    /// `saddl`: widen both operands, then add.
    AddL,
    /// `saddw`: widen only the second operand.
    AddW,
    SubL,
    SubW,
    /// `smull`: widen both, then multiply.
    MulL,
    /// `smlal`: widen both, multiply, accumulate into the destination.
    MlaL,
    MlsL,
    /// `sabdl`: widen both, then absolute difference.
    AbdL,
    /// `sabal`: as `abdl`, accumulating.
    AbaL,
}

/// Across-lane reductions.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VAcrossOp {
    Add,
    /// `saddlv`/`uaddlv`: every lane added into one scalar of twice the lane's width.
    SAddLv,
    UAddLv,
    SMax,
    SMin,
    UMax,
    UMin,
    FMax,
    FMin,
    /// `fmaxnmv`/`fmaxnmp`: a NaN loses to a number, rather than propagating.
    FMaxNm,
    FMinNm,
}

/// The three-operand bitwise selects.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VSelKind {
    /// `bsl`: the destination holds the mask.
    Bsl,
    /// `bit`: insert `vn` where `vm` has ones.
    Bit,
    /// `bif`: insert `vn` where `vm` has zeros.
    Bif,
}

/// Fixed byte permutations expressible as a single shuffle.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VPermOp {
    Trn1,
    Trn2,
    Uzp1,
    Uzp2,
    Zip1,
    Zip2,
}

/// Shift-by-immediate operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VShiftImmOp {
    Shl,
    SShr,
    UShr,
    /// Widen the low (or high, when `q`) half and shift left.
    SShll,
    UShll,
    /// Narrow each lane to half width after shifting right.
    Shrn,
    /// Narrow after a *rounding* shift right: `1 << (shift-1)` is added before
    /// the shift. `rshrn2` writes the destination's upper half.
    Rshrn,
    /// Shift right and accumulate into the destination.
    SSra,
    USra,
    /// Rounding shift right: add half the divisor (`1 << (shift-1)`) before the
    /// shift, so the result is rounded to nearest rather than truncated. Pixman
    /// leans on the unsigned form for alpha blending.
    Srshr,
    Urshr,
    /// Rounding shift right, then accumulate into the destination.
    Srsra,
    Ursra,
    /// Shift and insert: the bits shifted in come from the destination rather
    /// than being zero, so the shift is a splice of the two registers.
    Sli,
    Sri,
    /// Saturating shift left: `sqshl` and `uqshl` clamp a lane whose bits
    /// would leave it to the lane's signed or unsigned limit; `sqshlu` reads
    /// a signed lane and clamps to the unsigned range (negative lanes are 0).
    SqShl,
    UqShl,
    SqShlU,
    /// Saturating narrowing shifts right: like `shrn`/`rshrn`, but the shifted
    /// wide lane is clamped into the narrow one -- signed to signed (`sqshrn`),
    /// unsigned to unsigned (`uqshrn`), signed to unsigned (`sqshrun`). The `R`
    /// forms round. The `2` forms (`q`) write the destination's upper half.
    SqShrn,
    SqRshrn,
    UqShrn,
    UqRshrn,
    SqShrun,
    SqRshrun,
    /// Fixed-point to floating point: the lane is an integer with `shift`
    /// fractional bits, so the result is it divided by `2^shift`. Shares the
    /// shift-by-immediate encoding because the scale sits in the same field.
    Scvtf,
    Ucvtf,
    /// Floating point to fixed point, rounding toward zero: multiply by
    /// `2^shift` and truncate.
    Fcvtzs,
    Fcvtzu,
}

/// Bitwise vector operations on the `v` register file.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VLogOp {
    And,
    /// `vd = vn & !vm`.
    Bic,
    Orr,
    /// `vd = vn | !vm`.
    Orn,
    Eor,
}

/// System registers we model. Anything else decodes to [`SysReg::Other`] and
/// traps to the runtime.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SysReg {
    /// Thread pointer — the guest's TLS base.
    TpidrEl0,
    /// Read-only thread pointer.
    TpidrroEl0,
    /// Virtual counter, served from the shared time page.
    Cntvct,
    /// Counter frequency.
    Cntfrq,
    /// Cache type register.
    Ctr,
    /// Data cache zero ID.
    Dczid,
    /// Main ID register.
    Midr,
    /// FP control register.
    Fpcr,
    /// FP status register.
    Fpsr,
    /// NZCV as a system register.
    Nzcv,
    /// Encoded `op0:op1:crn:crm:op2`, unmodelled.
    Other(u16),
}

/// Cache/TLB maintenance operation.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SysMaintOp {
    /// `dc zva` — zero a cache line; the guest memset fast path.
    DcZva,
    /// `dc cvau`/`civac`/… — data cache clean, a no-op for a coherent host.
    DcClean,
    /// `ic ivau`/`iallu` — instruction cache invalidate. Our translation
    /// invalidation hook.
    IcInvalidate,
    /// Everything else (`at`, `tlbi`, …) — no-op.
    Other,
}

impl Inst {
    /// True when this instruction ends a basic block.
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Inst::Branch { .. }
                | Inst::BranchCond { .. }
                | Inst::CompareBranch { .. }
                | Inst::TestBranch { .. }
                | Inst::BranchReg { .. }
                | Inst::Svc { .. }
                | Inst::Brk { .. }
                | Inst::Unknown { .. }
                | Inst::Undefined { .. }
        )
    }

    /// For a direct branch, the statically known target.
    pub fn direct_target(&self) -> Option<u64> {
        match *self {
            Inst::Branch { target, .. }
            | Inst::BranchCond { target, .. }
            | Inst::CompareBranch { target, .. }
            | Inst::TestBranch { target, .. } => Some(target),
            _ => None,
        }
    }

    /// True when control may continue to the next instruction.
    pub fn falls_through(&self) -> bool {
        match self {
            Inst::Branch { link, .. } => *link,
            Inst::BranchReg { link, .. } => *link,
            Inst::BranchCond { .. } | Inst::CompareBranch { .. } | Inst::TestBranch { .. } => true,
            Inst::Svc { .. } | Inst::Brk { .. } => true,
            Inst::Unknown { .. } | Inst::Undefined { .. } => false,
            _ => true,
        }
    }
}
