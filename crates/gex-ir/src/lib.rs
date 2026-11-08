//! GIR — gex's small IR for binary translation.
//!
//! A [`Block`] is a straight-line list of [`Stmt`]s ending in one [`Term`].
//! Values are SSA: each [`Value`] is defined exactly once, by the `Stmt::Def`
//! that introduces it. Terminators name *guest* addresses, not backend labels.

#![forbid(unsafe_code)]

pub mod func;
pub mod interp;
pub mod pass;
pub mod relayout;
pub mod print;
pub mod ssa;
pub mod text;

use std::fmt;

thread_local! {
    static KUSER_FOLDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count one access moved from Windows' KUSER_SHARED_DATA address to 32-bit Wine's
/// (`gex_abi::fold_kuser`), by the lifter or an emitter, on this thread.
pub fn note_kuser_fold() {
    KUSER_FOLDS.with(|c| c.set(c.get() + 1));
    KUSER_FOLDS_ALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

static KUSER_FOLDS_ALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Every fold this process counted, on any thread (a build's summary).
pub fn kuser_folds_total() -> u64 {
    KUSER_FOLDS_ALL.load(std::sync::atomic::Ordering::Relaxed)
}

/// The folds this thread counted since the last call: a host takes them
/// after each translation and books them to the module it translated.
pub fn take_kuser_folds() -> u64 {
    KUSER_FOLDS.with(|c| c.replace(0))
}

static MXCSR_EXACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// SSE under the guest's MXCSR rounding control, a per-program opt-in: set
/// before anything is translated (the page's `?mxcsr=1`, GEX_MXCSR=1
/// natively). On, an SSE operation MXCSR rounds is lifted as [`Expr::Mx`]
/// and rounds as the rounding control says in every tier. Off, it is the
/// operation alone, rounded to nearest, and every tier's output and every
/// key are as if MXCSR did not exist. FTZ and DAZ are honoured in neither.
pub fn set_mxcsr_exact(on: bool) {
    MXCSR_EXACT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether SSE is lifted under MXCSR ([`set_mxcsr_exact`]).
pub fn mxcsr_exact() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_MXCSR").is_some_and(|v| v != "0"))
        || MXCSR_EXACT.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Guest 8- and 16-bit values live zero- or sign-extended inside an `I32`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    I32,
    I64,
    /// A 128-bit vector — one AArch64 `v` register, one wasm `v128`.
    V128,
}

impl Ty {
    pub fn bits(self) -> u32 {
        match self {
            Ty::I32 => 32,
            Ty::I64 => 64,
            Ty::V128 => 128,
        }
    }

    /// All-ones mask for this width, as a `u64`. Scalar types only.
    pub fn mask(self) -> u64 {
        match self {
            Ty::I32 => 0xffff_ffff,
            Ty::I64 | Ty::V128 => u64::MAX,
        }
    }
}

/// A piece of architectural guest state addressable as a register.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum GuestReg {
    /// `x0`–`x30`.
    X(u8),
    /// `v0`–`v31`, the SIMD and floating-point register file.
    V128(u8),
    /// Stack pointer.
    Sp,
    /// Program counter. Only written at block boundaries and safepoints.
    Pc,
    /// Negative flag.
    N,
    /// Zero flag.
    Z,
    /// Carry flag.
    C,
    /// Overflow flag.
    V,
    /// `TPIDR_EL0`, the guest thread pointer.
    Tpidr,
    /// `TPIDRRO_EL0`.
    TpidrRo,
    /// Exclusive monitor: tagged address.
    MonAddr,
    /// Exclusive monitor: value observed by the load-exclusive.
    MonValue,
    /// Exclusive monitor: access size in bytes, 0 when clear.
    MonSize,

    // -- x86-64 only ---------------------------------------------------------
    /// Parity flag.
    P,
    /// Auxiliary-carry flag.
    A,
    /// Direction flag.
    D,
    /// `MXCSR`.
    Mxcsr,
    /// x87 `TOP` (0..=7).
    X87Top,
    /// x87 control word.
    X87Cw,
    /// x87 status word, less `TOP` and the condition codes.
    X87Sw,
    /// x87 register tags: one valid bit per physical register.
    X87Tags,
    /// The guest address of this thread's own context structure. Read-only;
    /// how the x87 register stack (a computed slot of the context) is reached
    /// through ordinary loads and stores.
    CtxAddr,
}

impl GuestReg {
    pub fn ty(self) -> Ty {
        match self {
            GuestReg::N
            | GuestReg::Z
            | GuestReg::C
            | GuestReg::V
            | GuestReg::P
            | GuestReg::A
            | GuestReg::D
            | GuestReg::Mxcsr
            | GuestReg::X87Top
            | GuestReg::X87Cw
            | GuestReg::X87Sw
            | GuestReg::X87Tags
            => Ty::I32,
            GuestReg::V128(_) => Ty::V128,
            _ => Ty::I64,
        }
    }

    /// True for the condition flags: AArch64's four, x86-64's parity,
    /// auxiliary-carry and direction flags. Each lives in one byte of the context and holds 0 or 1.
    pub fn is_flag(self) -> bool {
        matches!(
            self,
            GuestReg::N
                | GuestReg::Z
                | GuestReg::C
                | GuestReg::V
                | GuestReg::P
                | GuestReg::A
                | GuestReg::D
        )
    }
}

/// Binary operations. Both operands and the result share one [`Ty`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Shift left; the amount is masked to the operand width (as wasm does).
    Shl,
    /// Logical shift right.
    ShrU,
    /// Arithmetic shift right.
    ShrS,
    /// Rotate right.
    Rotr,
    /// Unsigned division. AArch64 returns 0 on divide-by-zero where wasm
    /// traps, so the lifter guards the divisor.
    DivU,
    /// Signed division. Also needs guarding for `INT_MIN / -1`.
    DivS,
    RemU,
    RemS,
}

/// Unary operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    /// Count leading zeros.
    Clz,
    /// Count trailing zeros.
    Ctz,
    /// Population count.
    Popcnt,
    /// Bitwise complement.
    Not,
    /// Two's-complement negation.
    Neg,
    /// Reverse the bytes of the whole value.
    Bswap,
    /// Swap the low two bytes; the result is zero-extended (the bits above
    /// the low sixteen of the operand are ignored).
    Bswap16,
}

/// Integer comparisons. The result is an `I32` that is 0 or 1.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    LtU,
    LtS,
    LeU,
    LeS,
    GtU,
    GtS,
    GeU,
    GeS,
}

/// Memory access width in bytes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MemSz {
    B1,
    B2,
    B4,
    B8,
}

impl MemSz {
    pub fn bytes(self) -> u32 {
        match self {
            MemSz::B1 => 1,
            MemSz::B2 => 2,
            MemSz::B4 => 4,
            MemSz::B8 => 8,
        }
    }
}

/// Memory ordering attached to a load, store or read-modify-write.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Order {
    /// Plain access. Guest data races stay data races.
    Relaxed,
    /// Used for every *ordered* guest access: seq-cst is stronger than
    /// AArch64 acquire/release, so it is always a correct over-approximation.
    SeqCst,
}

/// Atomic read-modify-write operations that map to single wasm instructions.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

/// A value-producing expression.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Expr {
    ConstI32(u32),
    ConstI64(u64),
    /// Read architectural state.
    ReadReg(GuestReg),
    Bin {
        op: BinOp,
        ty: Ty,
        a: Value,
        b: Value,
    },
    Un {
        op: UnOp,
        ty: Ty,
        a: Value,
    },
    /// Compare two `ty`-typed operands, producing `I32` 0 or 1.
    Cmp {
        op: CmpOp,
        ty: Ty,
        a: Value,
        b: Value,
    },
    /// `cond ? a : b`, where `cond` is an `I32` boolean.
    Select {
        cond: Value,
        a: Value,
        b: Value,
        ty: Ty,
    },
    /// Narrow an `I64` to an `I32`.
    Wrap(Value),
    /// Widen an `I32` to an `I64`.
    Extend {
        a: Value,
        signed: bool,
    },
    /// Sign- or zero-extend the low `bits` of a `ty`-typed value, result also
    /// `ty`-typed.
    ExtendBits {
        a: Value,
        ty: Ty,
        bits: u32,
        signed: bool,
    },
    /// Read guest memory at `addr + offset`, where `addr` is a full 64-bit
    /// guest address. The same holds for every other memory access.
    Load {
        addr: Value,
        offset: i64,
        sz: MemSz,
        signed: bool,
        ty: Ty,
        order: Order,
    },
    /// Atomic read-modify-write; produces the value read.
    Rmw {
        op: RmwOp,
        addr: Value,
        val: Value,
        sz: MemSz,
        ty: Ty,
    },
    /// Compare-and-swap; produces the value read.
    CmpXchg {
        addr: Value,
        expected: Value,
        replacement: Value,
        sz: MemSz,
        ty: Ty,
    },
    /// Call a host import that returns a value.
    HostCall {
        func: HostFunc,
        args: Vec<Value>,
        ty: Ty,
    },

    /// Load `bytes` bytes into the low lanes of a vector, zeroing the rest.
    VLoad {
        addr: Value,
        offset: i64,
        bytes: u32,
    },
    /// A vector literal.
    VConst(Box<[u8; 16]>),
    /// Replicate a scalar into every `lane_bytes`-sized lane.
    VSplat {
        a: Value,
        lane_bytes: u32,
    },
    /// Bitwise operation on two vectors.
    VBin {
        op: VBinOp,
        a: Value,
        b: Value,
    },
    /// Lane-wise unary operation on a vector.
    VUn {
        op: VUnOp,
        a: Value,
    },
    /// Lane-wise binary operation.
    VLaneBin {
        op: VLaneBinOp,
        lane: VLane,
        a: Value,
        b: Value,
    },
    /// Lane-wise unary operation.
    VLaneUn {
        op: VLaneUnOp,
        lane: VLane,
        a: Value,
    },
    /// Lane-wise shift by an `I32` amount shared across lanes.
    VShift {
        op: VShiftOp,
        lane: VLane,
        a: Value,
        amount: Value,
    },
    /// Byte-level permutation of two vectors by a fixed index. Covers `trn`,
    /// `uzp`, `zip`, `ext`, the `rev` family and element `dup`/`ins`.
    VShuffle {
        a: Value,
        b: Value,
        lanes: Box<[u8; 16]>,
    },
    /// Byte-level permutation by a runtime index vector; out-of-range indices
    /// produce zero, which is what `tbl` with a single table register does.
    VSwizzle {
        a: Value,
        idx: Value,
    },
    /// `(a & mask) | (b & !mask)`.
    VBitSelect {
        mask: Value,
        a: Value,
        b: Value,
    },
    /// Lane conversion.
    VCvt {
        op: VCvtOp,
        /// Lane type of the source.
        from: VLane,
        a: Value,
        /// Second operand, only for the narrowing conversions.
        b: Option<Value>,
    },
    /// Read one lane out as a scalar.
    VExtract {
        a: Value,
        lane: u32,
        lane_bytes: u32,
        signed: bool,
        /// Type of the extracted scalar.
        ty: Ty,
    },
    /// Write one lane, producing a new vector.
    VReplace {
        a: Value,
        lane: u32,
        lane_bytes: u32,
        val: Value,
    },
    /// The sign bit of every lane, packed into an `I32` with lane 0 at bit 0
    /// (`pmovmskb`, `movmskps`; wasm's `bitmask`).
    VBitmask {
        a: Value,
        lane: VLane,
    },

    /// Scalar floating-point arithmetic on bit patterns.
    FpBin {
        op: FpBinOp,
        /// True for double precision, false for single.
        double: bool,
        a: Value,
        b: Value,
    },
    FpUn {
        op: FpUnOp,
        double: bool,
        a: Value,
    },
    /// Compare two floats, producing an `I32` boolean. Any comparison
    /// involving a NaN is false, which is IEEE-754's rule and wasm's.
    FpCmp {
        op: FpCmpOp,
        double: bool,
        a: Value,
        b: Value,
    },
    /// Convert between floating-point and integer representations.
    FpCvt {
        kind: FpConvert,
        /// Width of the source: true for 64-bit.
        src_wide: bool,
        /// Width of the destination: true for 64-bit.
        dst_wide: bool,
        a: Value,
    },
    /// `e` computed as x86's SSE computes it under the `I32` MXCSR value
    /// `mx`'s rounding control (bits 13-14, [`MX_SPECIAL`]) instead of to
    /// nearest. FTZ and DAZ are not honoured: denormals are kept, as wasm
    /// keeps them. With the rounding to nearest it is `e` exactly, so a
    /// backend lowers it as `e` behind that one test (for a run of them,
    /// one test for the run).
    ///
    /// `e` is one of [`Expr::mx_able`]'s: a float add, subtract, multiply,
    /// divide or square root, a round to nearest (an integer value, the
    /// rounding of a conversion to integer), a conversion to a float, as
    /// scalars (`FpBin`, `FpUn`, `FpCvt`) or lanes (`VLaneBin`, `VLaneUn`,
    /// `VCvt`). Its operands are this expression's.
    Mx { mx: Value, e: Box<Expr> },
}

/// The MXCSR bits [`Expr::Mx`] honours: the rounding control. The exact
/// import (`gex_abi::imports::MX`) is given only these.
pub const MX_SPECIAL: u32 = 0x6000;

/// How an [`Expr::Mx`]'s operation is computed off its fast path: by
/// `gex_f80::sse::op` with `code`, on raw bits, once for a scalar or once
/// per lane.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MxPlan {
    /// `a` (and `b`), each an integer of `in_ty` holding a float's bits or
    /// a conversion's integer; the result is the expression's type.
    Scalar { code: u32, a: Value, b: Option<Value>, in_ty: Ty },
    /// Lanes `0..n` of `a` (and `b`), `in_bytes` wide, each to lane `k`,
    /// `out_bytes` wide, of a zero vector.
    Lanes { code: u32, a: Value, b: Option<Value>, n: u32, in_bytes: u32, out_bytes: u32 },
}

impl MxPlan {
    /// The plan of [`Expr::mx_able`] `e`.
    pub fn of(e: &Expr) -> MxPlan {
        use gex_f80::sse::*;
        let bin = |op: u32| match op {
            0 => ADD,
            1 => SUB,
            2 => MUL,
            _ => DIV,
        };
        let wide = |d: bool| if d { DOUBLE } else { 0 };
        let int_ty = |d: bool| if d { Ty::I64 } else { Ty::I32 };
        match *e {
            Expr::FpBin { op, double, a, b } => MxPlan::Scalar {
                code: bin(op as u32) | wide(double),
                a,
                b: Some(b),
                in_ty: int_ty(double),
            },
            Expr::FpUn { op, double, a } => MxPlan::Scalar {
                code: if op == FpUnOp::Sqrt { SQRT } else { RINT } | wide(double),
                a,
                b: None,
                in_ty: int_ty(double),
            },
            Expr::FpCvt { kind: FpConvert::IntToFp { .. }, src_wide, dst_wide, a } => MxPlan::Scalar {
                code: if src_wide { FROM_I64 } else { FROM_I32 } | wide(dst_wide),
                a,
                b: None,
                in_ty: int_ty(src_wide),
            },
            Expr::FpCvt { src_wide, a, .. } => MxPlan::Scalar {
                code: if src_wide { NARROW } else { WIDEN | DOUBLE },
                a,
                b: None,
                in_ty: int_ty(src_wide),
            },
            Expr::VLaneBin { op, lane, a, b } => {
                let d = lane == VLane::F64;
                let k = match op {
                    VLaneBinOp::Add => ADD,
                    VLaneBinOp::Sub => SUB,
                    VLaneBinOp::Mul => MUL,
                    _ => DIV,
                };
                let lb = lane.bytes();
                MxPlan::Lanes { code: k | wide(d), a, b: Some(b), n: 16 / lb, in_bytes: lb, out_bytes: lb }
            }
            Expr::VLaneUn { op, lane, a } => {
                let d = lane == VLane::F64;
                let lb = lane.bytes();
                let k = if op == VLaneUnOp::Sqrt { SQRT } else { RINT };
                MxPlan::Lanes { code: k | wide(d), a, b: None, n: 16 / lb, in_bytes: lb, out_bytes: lb }
            }
            Expr::VCvt { op: VCvtOp::DemoteF64, a, .. } => {
                MxPlan::Lanes { code: NARROW, a, b: None, n: 2, in_bytes: 8, out_bytes: 4 }
            }
            Expr::VCvt { op: VCvtOp::PromoteF32, a, .. } => {
                MxPlan::Lanes { code: WIDEN | DOUBLE, a, b: None, n: 2, in_bytes: 4, out_bytes: 8 }
            }
            Expr::VCvt { a, .. } => MxPlan::Lanes { code: FROM_I32, a, b: None, n: 4, in_bytes: 4, out_bytes: 4 },
            _ => panic!("MxPlan::of: {e:?} is not an operation MXCSR rounds"),
        }
    }
}

/// Scalar floating-point operations.
///
/// FP values travel through GIR as their integer *bit patterns* (`I32` for
/// single, `I64` for double); the backend reinterprets around each operation.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpBinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// IEEE-754 minimum, propagating NaN — AArch64's `FMIN`.
    Min,
    Max,
    Copysign,
}

/// Scalar floating-point unary operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpUnOp {
    Abs,
    Neg,
    Sqrt,
    /// Round to nearest, ties to even.
    RoundNearest,
    /// Round toward +infinity.
    Ceil,
    /// Round toward -infinity.
    Floor,
    /// Round toward zero.
    Trunc,
}

/// Scalar floating-point comparisons, producing an `I32` boolean.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A conversion between a floating-point value and an integer, or between
/// the two floating-point widths.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FpConvert {
    /// Integer to float. `signed` selects the interpretation of the source.
    IntToFp { signed: bool },
    /// Float to integer, rounding toward zero and *saturating*, matching
    /// AArch64's `FCVTZ*` (hence wasm's non-trapping conversions).
    FpToInt { signed: bool },
    /// Single to double, or double to single.
    FpToFp,
}

/// The lane type a vector operation works on. Lane count follows from the
/// 128-bit register; 64-bit forms are handled by masking the result.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VLane {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl VLane {
    pub fn bytes(self) -> u32 {
        match self {
            VLane::I8 => 1,
            VLane::I16 => 2,
            VLane::I32 | VLane::F32 => 4,
            VLane::I64 | VLane::F64 => 8,
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, VLane::F32 | VLane::F64)
    }
    /// The integer lane type of the same width.
    pub fn as_int(self) -> VLane {
        match self {
            VLane::F32 => VLane::I32,
            VLane::F64 => VLane::I64,
            other => other,
        }
    }
}

/// Lane-wise binary operations.
///
/// Comparisons write all-ones or all-zeros per lane, which is what both NEON
/// and wasm SIMD do, so they compose directly with `bitselect`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VLaneBinOp {
    Add,
    Sub,
    Mul,
    Div,
    MinS,
    MinU,
    MaxS,
    MaxU,
    AddSatS,
    AddSatU,
    SubSatS,
    SubSatU,
    AvgrU,
    /// `i16x8.q15mulr_sat_s`: `(a * b + 0x4000) >> 15` per lane, saturated; the only
    /// overflow is `-32768 * -32768`. AArch64's `sqrdmulh` on halfwords, exactly.
    Q15MulrSatS,
    /// Signed widening multiply of the low (or high) half of the source lanes: the
    /// result has half as many lanes, each twice as wide as `lane`, which names the
    /// *source* lanes (`i32x4.extmul_low_i16x8_s`, `i64x2.extmul_low_i32x4_s`).
    ExtMulLowS,
    ExtMulHighS,
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

/// Lane-wise unary operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VLaneUnOp {
    Neg,
    Abs,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

/// Lane-wise shift by a scalar amount shared across lanes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VShiftOp {
    Shl,
    ShrS,
    ShrU,
}

/// Lane-width and representation conversions.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VCvtOp {
    /// Float lanes to integer lanes, rounding toward zero and saturating.
    FpToIntSat { signed: bool },
    /// Integer lanes to float lanes.
    IntToFp { signed: bool },
    /// `f64x2` to the low two lanes of an `f32x4`.
    DemoteF64,
    /// The low two lanes of an `f32x4` to `f64x2`.
    PromoteF32,
    /// Widen the low half of the source's lanes.
    ExtendLow { signed: bool },
    /// Widen the high half.
    ExtendHigh { signed: bool },
    /// Halve the lane width of two vectors into one, saturating.
    NarrowSat { signed: bool },
}

/// Vector unary operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VUnOp {
    /// Per-byte population count.
    PopcntBytes,
    /// Add adjacent lanes pairwise into lanes of twice the width, signed or
    /// unsigned. Used to sum a popcount back up, and to lift `saddlp`/`uaddlp`.
    ExtAddPairwiseU8,
    ExtAddPairwiseU16,
    ExtAddPairwiseS8,
    ExtAddPairwiseS16,
}

/// Bitwise vector operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum VBinOp {
    And,
    Or,
    Xor,
    /// `a & !b`.
    AndNot,
}

impl Expr {
    /// True when the expression can be reordered or deleted freely. Loads are
    /// *not* pure: they must keep their position relative to stores.
    pub fn is_pure(&self) -> bool {
        !matches!(
            self,
            Expr::Load { .. }
                | Expr::Rmw { .. }
                | Expr::CmpXchg { .. }
                | Expr::HostCall { .. }
                | Expr::VLoad { .. }
        )
    }

    /// The type of the value this expression produces.
    pub fn ty(&self) -> Ty {
        match *self {
            Expr::ConstI32(_) | Expr::Wrap(_) | Expr::Cmp { .. } => Ty::I32,
            Expr::ConstI64(_) | Expr::Extend { .. } => Ty::I64,
            Expr::ReadReg(r) => r.ty(),
            Expr::Bin { ty, .. }
            | Expr::Un { ty, .. }
            | Expr::Select { ty, .. }
            | Expr::ExtendBits { ty, .. }
            | Expr::Load { ty, .. }
            | Expr::Rmw { ty, .. }
            | Expr::CmpXchg { ty, .. }
            | Expr::HostCall { ty, .. }
            | Expr::VExtract { ty, .. } => ty,
            Expr::VLoad { .. }
            | Expr::VConst(_)
            | Expr::VSplat { .. }
            | Expr::VBin { .. }
            | Expr::VUn { .. }
            | Expr::VLaneBin { .. }
            | Expr::VLaneUn { .. }
            | Expr::VShift { .. }
            | Expr::VShuffle { .. }
            | Expr::VSwizzle { .. }
            | Expr::VBitSelect { .. }
            | Expr::VCvt { .. }
            | Expr::VReplace { .. } => Ty::V128,
            Expr::VBitmask { .. } => Ty::I32,
            // FP results are bit patterns, so width follows precision.
            Expr::FpBin { double, .. } | Expr::FpUn { double, .. } => {
                if double {
                    Ty::I64
                } else {
                    Ty::I32
                }
            }
            Expr::FpCmp { .. } => Ty::I32,
            Expr::FpCvt { dst_wide, .. } => {
                if dst_wide {
                    Ty::I64
                } else {
                    Ty::I32
                }
            }
            Expr::Mx { ref e, .. } => e.ty(),
        }
    }

    /// Whether `self` may be the operation of an [`Expr::Mx`].
    pub fn mx_able(&self) -> bool {
        match self {
            Expr::FpBin { op, .. } => matches!(op, FpBinOp::Add | FpBinOp::Sub | FpBinOp::Mul | FpBinOp::Div),
            Expr::FpUn { op, .. } => matches!(op, FpUnOp::Sqrt | FpUnOp::RoundNearest),
            Expr::FpCvt { kind, .. } => matches!(kind, FpConvert::IntToFp { signed: true } | FpConvert::FpToFp),
            Expr::VLaneBin { op, lane, .. } => {
                lane.is_float() && matches!(op, VLaneBinOp::Add | VLaneBinOp::Sub | VLaneBinOp::Mul | VLaneBinOp::Div)
            }
            Expr::VLaneUn { op, lane, .. } => lane.is_float() && matches!(op, VLaneUnOp::Sqrt | VLaneUnOp::Nearest),
            Expr::VCvt { op, from, .. } => matches!(
                (op, from),
                (VCvtOp::DemoteF64, VLane::F64) | (VCvtOp::PromoteF32, VLane::F32) | (VCvtOp::IntToFp { signed: true }, VLane::I32)
            ),
            _ => false,
        }
    }

    /// Visit every value this expression reads.
    pub fn for_each_use(&self, mut f: impl FnMut(Value)) {
        match self {
            Expr::ConstI32(_) | Expr::ConstI64(_) | Expr::ReadReg(_) => {}
            Expr::Bin { a, b, .. } | Expr::Cmp { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Expr::Un { a, .. } | Expr::Wrap(a) | Expr::Extend { a, .. } => f(*a),
            Expr::ExtendBits { a, .. } => f(*a),
            Expr::Select { cond, a, b, .. } => {
                f(*cond);
                f(*a);
                f(*b);
            }
            Expr::Load { addr, .. } => f(*addr),
            Expr::Rmw { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            Expr::CmpXchg {
                addr,
                expected,
                replacement,
                ..
            } => {
                f(*addr);
                f(*expected);
                f(*replacement);
            }
            Expr::HostCall { args, .. } => args.iter().copied().for_each(f),
            Expr::VLoad { addr, .. } => f(*addr),
            Expr::VConst(_) => {}
            Expr::VSplat { a, .. }
            | Expr::VExtract { a, .. }
            | Expr::VUn { a, .. }
            | Expr::VBitmask { a, .. } => f(*a),
            Expr::VLaneUn { a, .. } => f(*a),
            Expr::VLaneBin { a, b, .. } | Expr::VShuffle { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Expr::VShift { a, amount, .. } => {
                f(*a);
                f(*amount);
            }
            Expr::VSwizzle { a, idx } => {
                f(*a);
                f(*idx);
            }
            Expr::VBitSelect { mask, a, b } => {
                f(*mask);
                f(*a);
                f(*b);
            }
            Expr::VCvt { a, b, .. } => {
                f(*a);
                if let Some(b) = b {
                    f(*b);
                }
            }
            Expr::VBin { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Expr::VReplace { a, val, .. } => {
                f(*a);
                f(*val);
            }
            Expr::FpBin { a, b, .. } | Expr::FpCmp { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Expr::FpUn { a, .. } | Expr::FpCvt { a, .. } => f(*a),
            Expr::Mx { mx, e } => {
                f(*mx);
                e.for_each_use(f);
            }
        }
    }
}

/// Host imports callable from translated code.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum HostFunc {
    /// `svc` — enter the kernel.
    Syscall,
    /// Report an undecodable or undefined instruction; raises SIGILL.
    Illegal,
    /// `brk`; raises SIGTRAP.
    Breakpoint,
    /// Read an unmodelled system register.
    SysRegRead,
    /// Write an unmodelled system register.
    SysRegWrite,
    /// Cache maintenance, including the translation-invalidation hook.
    CacheOp,
    /// Safepoint poll for signal delivery.
    Safepoint,
    /// Fused multiply-add, which wasm cannot express.
    Fma,
}

impl HostFunc {
    /// True when the call may change guest architectural state behind the
    /// translator's back, invalidating every cached register.
    pub fn clobbers_guest_state(self) -> bool {
        matches!(
            self,
            HostFunc::Syscall
                | HostFunc::Illegal
                | HostFunc::Breakpoint
                | HostFunc::SysRegWrite
                | HostFunc::Safepoint
        )
    }
}

/// One statement in a block. Statements execute in order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Stmt {
    /// Define an SSA value.
    Def { dst: Value, expr: Expr },
    /// Write architectural state.
    WriteReg { reg: GuestReg, val: Value },
    /// Store to guest memory at `addr + offset`.
    Store {
        addr: Value,
        offset: i64,
        val: Value,
        sz: MemSz,
        order: Order,
    },
    /// A host call whose result is discarded.
    HostCall { func: HostFunc, args: Vec<Value> },
    /// An x87 operation, computed exactly as the hardware computes it
    /// (`gex_f80` is the reference, tested against the hardware):
    /// `outs` are defined from `args`, both as [`X87Op`] lists them.
    ///
    /// Pure: it reads only its arguments and writes only its outputs -- no
    /// guest register, no memory, no fault -- so an analysis treats it as a
    /// definition of several values, and it may be deleted when none of them
    /// is used. The lifter keeps what surrounds it (the register stack, the
    /// status word, the unmasked-exception check) outside it.
    ///
    /// A backend lowers it to a fast path for the common operands and falls
    /// back, for the rest, to the same exact algorithm: tier-1 calls the
    /// `x87` host import, which is pure too (it neither reads nor writes
    /// cached guest state, so no flush); gex-cgen calls a C helper.
    X87 {
        op: X87Op,
        args: Vec<Value>,
        outs: Vec<Value>,
    },
    /// A conditional guest fault: when the `I32` `cond` is nonzero, the
    /// instruction at `pc` faults with reason `code` (a negative
    /// `gex_abi::imports::BRK_*` code, such as `BRK_DIVIDE_BY_ZERO`) and
    /// control does not come back to this block. When `cond` is zero it does
    /// nothing.
    ///
    /// At the fault the architectural state is exact: every guest register
    /// written earlier in the block (and nothing written later) is in the
    /// context, and `PC` holds `pc`, because the host delivers a signal or
    /// an exception from that state. So for an analysis a `Trap` reads every
    /// register: it is a may-unwind point (a function summary must assume
    /// control can leave here), and a backend that keeps registers in locals
    /// must write them all back on the faulting path. Tier-1 lowers it to
    /// `if cond { write back; store pc; breakpoint(ctx, pc, code) }`, the
    /// host raising the fault from `breakpoint`.
    Trap { cond: Value, pc: u64, code: i32 },
    /// Store the low `bytes` bytes of a vector to guest memory.
    VStore {
        addr: Value,
        offset: i64,
        val: Value,
        bytes: u32,
    },
    /// Copy `len` bytes of guest memory from `src` to `dst`, overlap-safe.
    ///
    /// Lowers to wasm's `memory.copy` (which behaves as if through a
    /// temporary, so it serves both `memcpy` and `memmove`). Produced only by
    /// recognising a library function by name, not by the lifter.
    MemCopy { dst: Value, src: Value, len: Value },
    /// Set `len` bytes of guest memory at `dst` to the low byte of `val`.
    MemFill { dst: Value, val: Value, len: Value },
    /// `dst = memcmp(a, b, len)`, as C defines it: negative, zero or positive
    /// according to the first byte that differs, compared as `unsigned char`.
    ///
    /// A statement rather than an `Expr` because it lowers to a wasm *loop*;
    /// there is no single wasm instruction behind it.
    MemCmp {
        dst: Value,
        a: Value,
        b: Value,
        len: Value,
    },
    /// A memory fence.
    Fence,
    /// An x86 `rep`-prefixed string instruction, as one statement: a loop of
    /// `count` elements of `sz` bytes, each step moving `dst`/`src` by `+sz`
    /// when `df` (an `I32`) is 0 and `-sz` when it is 1. Lowers to a wasm
    /// loop (with `memory.copy`/`memory.fill` fast paths), which is why it is
    /// a statement and not an expression.
    ///
    /// Defines `out_count`, `out_dst`, `out_src` (all `I64`) with the final
    /// register values, and for `Scas`/`Cmps` also `out_a`/`out_b` (`I64`,
    /// zero-extended) with the last two elements compared, from which the
    /// lifter derives the flags. When no iteration runs, `out_a` and
    /// `out_b` are both zero.
    StrOp {
        kind: StrKind,
        sz: MemSz,
        until: StrUntil,
        count: Value,
        dst: Value,
        src: Value,
        /// The accumulator: what `stos` stores and `scas` compares against.
        val: Value,
        df: Value,
        out_count: Value,
        out_dst: Value,
        out_src: Value,
        out_a: Value,
        out_b: Value,
    },
}

/// What a [`Stmt::X87`] computes. Every operation's last output is an
/// `I32` of status-word bits: the exception flags it raised (IE, DE, ZE, OE,
/// UE, PE in bits 0-5) and C1 in bit 9 (always clear for `Compare`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum X87Op {
    /// `fst`/`fstp` to single (`single`) or double: the register's value
    /// rounded by the control word's RC to the format.
    /// Args: significand `I64`, sign and exponent `I32`, control word `I32`.
    /// Outs: the format's raw bits `I64` (zero-extended), flags.
    Narrow { single: bool },
    /// `fist`/`fistp` (by RC) or `fisttp` (`trunc`) to a `bits`-bit integer
    /// (16, 32 or 64); out of range, the integer indefinite with IE.
    /// Args: as `Narrow`. Outs: the integer `I64` (the low `bits` matter),
    /// flags.
    ToInt { bits: u8, trunc: bool },
    /// `fadd`/`fsub`/`fmul`/`fdiv` (and their reversed and popping forms,
    /// which only choose the operands), rounded by the control word's
    /// precision and rounding control.
    /// Args: a's significand `I64` and sign/exponent `I32`, b's likewise,
    /// the control word `I32`. Outs: the result's significand `I64` and
    /// sign/exponent `I32`, flags.
    Arith(X87Arith),
    /// `fcom` (any NaN raises IE) or `fucom` (`quiet`: only an SNaN does).
    /// Args: a's significand and sign/exponent, b's likewise. Outs: the
    /// condition codes as status-word bits `I32` (C0 for less, C3 for
    /// equal, C0|C2|C3 for unordered, none for greater), flags.
    Compare { quiet: bool },
}

/// The operation of an [`X87Op::Arith`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum X87Arith {
    Add,
    Sub,
    Mul,
    Div,
}

impl X87Op {
    /// The types of the operation's arguments.
    pub fn arg_types(self) -> &'static [Ty] {
        match self {
            X87Op::Narrow { .. } | X87Op::ToInt { .. } => &[Ty::I64, Ty::I32, Ty::I32],
            X87Op::Arith(_) => &[Ty::I64, Ty::I32, Ty::I64, Ty::I32, Ty::I32],
            X87Op::Compare { .. } => &[Ty::I64, Ty::I32, Ty::I64, Ty::I32],
        }
    }

    /// The types of the operation's outputs.
    pub fn out_types(self) -> &'static [Ty] {
        match self {
            X87Op::Narrow { .. } | X87Op::ToInt { .. } => &[Ty::I64, Ty::I32],
            X87Op::Arith(_) => &[Ty::I64, Ty::I32, Ty::I32],
            X87Op::Compare { .. } => &[Ty::I32, Ty::I32],
        }
    }

    /// The code the `x87` host import takes for this operation: 1 and 2 for
    /// single and double `Narrow`; `0x100 | bits << 1 | trunc` for `ToInt`;
    /// `0x200 | k` for `Arith` (add, sub, mul, div = 0..3); `0x300 | quiet`
    /// for `Compare`.
    pub fn code(self) -> i32 {
        match self {
            X87Op::Narrow { single: true } => 1,
            X87Op::Narrow { single: false } => 2,
            X87Op::ToInt { bits, trunc } => 0x100 | (bits as i32) << 1 | trunc as i32,
            X87Op::Arith(k) => 0x200 | k as i32,
            X87Op::Compare { quiet } => 0x300 | quiet as i32,
        }
    }
}

/// Which string instruction a [`Stmt::StrOp`] repeats.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StrKind {
    /// `[dst] = [src]`.
    Movs,
    /// `[dst] = val`.
    Stos,
    /// compare `val` with `[dst]`.
    Scas,
    /// compare `[src]` with `[dst]`.
    Cmps,
}

/// When a [`Stmt::StrOp`] stops, besides the count reaching zero.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StrUntil {
    /// Only the count (`rep movs`/`rep stos`).
    Count,
    /// Also when the elements compared differ (`repe`).
    NotEqual,
    /// Also when the elements compared match (`repne`).
    Equal,
}

impl Stmt {
    /// Visit every value this statement reads.
    pub fn for_each_use(&self, mut f: impl FnMut(Value)) {
        match self {
            Stmt::Def { expr, .. } => expr.for_each_use(f),
            Stmt::WriteReg { val, .. } => f(*val),
            Stmt::Store { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            Stmt::HostCall { args, .. } => args.iter().copied().for_each(f),
            Stmt::X87 { args, .. } => args.iter().copied().for_each(f),
            Stmt::Trap { cond, .. } => f(*cond),
            Stmt::VStore { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            Stmt::MemCopy { dst, src, len } => {
                f(*dst);
                f(*src);
                f(*len);
            }
            Stmt::MemFill { dst, val, len } => {
                f(*dst);
                f(*val);
                f(*len);
            }
            // `dst` is defined here, not used, so it is not visited.
            Stmt::StrOp {
                count,
                dst,
                src,
                val,
                df,
                ..
            } => {
                f(*count);
                f(*dst);
                f(*src);
                f(*val);
                f(*df);
            }
            Stmt::MemCmp { a, b, len, .. } => {
                f(*a);
                f(*b);
                f(*len);
            }
            Stmt::Fence => {}
        }
    }

    /// True when removing this statement would change observable behaviour.
    pub fn has_side_effects(&self) -> bool {
        match self {
            Stmt::Def { expr, .. } => !expr.is_pure(),
            Stmt::X87 { .. } => false,
            Stmt::WriteReg { .. } => true,
            Stmt::Store { .. }
            | Stmt::HostCall { .. }
            | Stmt::Trap { .. }
            | Stmt::VStore { .. }
            | Stmt::MemCopy { .. }
            | Stmt::MemFill { .. }
            | Stmt::MemCmp { .. }
            | Stmt::StrOp { .. }
            | Stmt::Fence => true,
        }
    }
}

/// How a block transfers control. Terminators name guest addresses, never
/// backend labels.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Term {
    /// Continue at a statically known guest address.
    Goto { target: u64 },
    /// Two-way branch on an `I32` boolean.
    Branch {
        cond: Value,
        taken: u64,
        not_taken: u64,
    },
    /// Branch to a computed guest address (`br`, `blr`, `ret`).
    Indirect {
        target: Value,
        /// True for `ret`, which the backend may serve from a return cache.
        is_ret: bool,
    },
    /// Enter the kernel, then continue at `next`. `imm` is the system-call
    /// number when the ISA encodes it in the instruction itself (an AArch64
    /// `svc #imm`); 0 when the number comes from a register
    /// ([`crate::ctx::SYSCALL_NR`]).
    Syscall { next: u64, imm: u32 },
    /// An instruction we cannot execute. Raises SIGILL and does not return.
    Illegal { raw: u32, pc: u64 },
}

/// A translated basic block: guest instructions from `start` up to `end`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Block {
    /// Guest address of the first instruction.
    pub start: u64,
    /// Guest address one past the last instruction.
    pub end: u64,
    /// Statements in program order.
    pub stmts: Vec<Stmt>,
    /// How the block ends.
    pub term: Term,
    /// Number of SSA values defined; value indices are `0..n_values`.
    pub n_values: u32,
    /// Where a call in this block is expected to return to (set for `bl`,
    /// `blr`). Not a control-flow successor, but an entry point the callee's
    /// `ret` branches to; without it code discovery misses every return site.
    pub call_return: Option<u64>,
    /// The code bytes this block was lifted from, when its page is writable as
    /// well as executable (a JIT's). The emitter compares them at every entry
    /// and re-lifts the page when they have changed: x86-64 has no cache-flush
    /// instruction to announce self-modifying code, so the block checks itself.
    pub guard: Option<Vec<u8>>,
}

impl Block {
    pub fn new(start: u64) -> Block {
        Block {
            start,
            end: start,
            stmts: Vec::new(),
            term: Term::Goto { target: start },
            n_values: 0,
            call_return: None,
            guard: None,
        }
    }

    /// Visit every value read by the terminator.
    pub fn term_uses(&self, mut f: impl FnMut(Value)) {
        match self.term {
            Term::Branch { cond, .. } => f(cond),
            Term::Indirect { target, .. } => f(target),
            _ => {}
        }
    }

    /// Statically known successors, for CFG construction. An indirect branch
    /// contributes none.
    pub fn successors(&self) -> impl Iterator<Item = u64> {
        let mut out = Vec::new();
        match self.term {
            Term::Goto { target } => out.push(target),
            Term::Branch {
                taken, not_taken, ..
            } => {
                out.push(taken);
                out.push(not_taken);
            }
            Term::Syscall { next, .. } => out.push(next),
            Term::Indirect { .. } | Term::Illegal { .. } => {}
        }
        out.into_iter()
    }
}

/// Incrementally builds a [`Block`]. The lifter's only interface to GIR.
pub struct Builder {
    block: Block,
}

impl Builder {
    pub fn new(start: u64) -> Builder {
        Builder {
            block: Block::new(start),
        }
    }

    /// Append a value-producing expression and return its handle.
    pub fn def(&mut self, expr: Expr) -> Value {
        let v = Value(self.block.n_values);
        self.block.n_values += 1;
        self.block.stmts.push(Stmt::Def { dst: v, expr });
        v
    }

    /// Allocate a value that a statement will define itself, e.g. `MemCmp`,
    /// which produces a value without being an `Expr`.
    pub fn fresh(&mut self) -> Value {
        let v = Value(self.block.n_values);
        self.block.n_values += 1;
        v
    }

    pub fn stmt(&mut self, s: Stmt) {
        self.block.stmts.push(s);
    }

    /// Emit a [`Stmt::X87`] and return its outputs.
    pub fn x87(&mut self, op: X87Op, args: Vec<Value>) -> Vec<Value> {
        debug_assert_eq!(args.len(), op.arg_types().len());
        let outs: Vec<Value> = op.out_types().iter().map(|_| self.fresh()).collect();
        self.block.stmts.push(Stmt::X87 {
            op,
            args,
            outs: outs.clone(),
        });
        outs
    }

    pub fn const_i32(&mut self, v: u32) -> Value {
        self.def(Expr::ConstI32(v))
    }

    pub fn const_i64(&mut self, v: u64) -> Value {
        self.def(Expr::ConstI64(v))
    }

    /// A `ty`-typed constant; for `V128`, `v` fills the low 8 bytes.
    pub fn const_ty(&mut self, ty: Ty, v: u64) -> Value {
        match ty {
            Ty::I32 => self.const_i32(v as u32),
            Ty::I64 => self.const_i64(v),
            Ty::V128 => {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&v.to_le_bytes());
                self.def(Expr::VConst(Box::new(bytes)))
            }
        }
    }

    pub fn read(&mut self, reg: GuestReg) -> Value {
        self.def(Expr::ReadReg(reg))
    }

    pub fn write(&mut self, reg: GuestReg, val: Value) {
        self.stmt(Stmt::WriteReg { reg, val });
    }

    pub fn bin(&mut self, op: BinOp, ty: Ty, a: Value, b: Value) -> Value {
        self.def(Expr::Bin { op, ty, a, b })
    }

    pub fn un(&mut self, op: UnOp, ty: Ty, a: Value) -> Value {
        self.def(Expr::Un { op, ty, a })
    }

    pub fn cmp(&mut self, op: CmpOp, ty: Ty, a: Value, b: Value) -> Value {
        self.def(Expr::Cmp { op, ty, a, b })
    }

    pub fn select(&mut self, cond: Value, a: Value, b: Value, ty: Ty) -> Value {
        self.def(Expr::Select { cond, a, b, ty })
    }

    pub fn wrap(&mut self, a: Value) -> Value {
        self.def(Expr::Wrap(a))
    }

    pub fn extend(&mut self, a: Value, signed: bool) -> Value {
        self.def(Expr::Extend { a, signed })
    }

    /// Record that this block calls out and expects to resume at `pc`.
    pub fn set_call_return(&mut self, pc: u64) {
        self.block.call_return = Some(pc);
    }

    /// Mark the block as covering guest instructions up to (excluding) `pc`.
    pub fn end_at(&mut self, pc: u64) {
        self.block.end = pc;
    }

    /// How many statements have been appended so far.
    pub fn mark(&self) -> usize {
        self.block.stmts.len()
    }

    /// Drop everything appended since `mark`. Values defined by the dropped
    /// statements keep their numbers: the counter only ever goes up, so no
    /// live handle is reused.
    pub fn rewind(&mut self, mark: usize) {
        self.block.stmts.truncate(mark);
    }

    /// Whether any statement since `mark` uses a lane operation wasm cannot
    /// express. Checking here lets the lifter turn it into a precise SIGILL at
    /// the instruction's address rather than a backend panic later.
    pub fn has_inexpressible(&self, mark: usize) -> bool {
        self.block.stmts[mark..].iter().any(|s| match s {
            Stmt::Def {
                expr: Expr::VLaneBin { op, lane, .. },
                ..
            } => op.wasm_opcode(*lane).is_none(),
            Stmt::Def {
                expr: Expr::VLaneUn { op, lane, .. },
                ..
            } => op.wasm_opcode(*lane).is_none(),
            _ => false,
        })
    }

    /// The address one past the last instruction lifted so far.
    pub fn end(&self) -> u64 {
        self.block.end
    }

    pub fn finish(mut self, term: Term) -> Block {
        self.block.term = term;
        self.block
    }
}

impl VLaneBinOp {
    /// The wasm SIMD opcode for this operation at this lane width.
    ///
    /// `None` means wasm cannot express it at all (e.g. no `i64x2` saturating
    /// add, no `i8x16.mul`), so the lifter must refuse the instruction. Lifter
    /// and backend both consult this one table so they never disagree.
    pub fn wasm_opcode(self, lane: VLane) -> Option<u32> {
        use VLane::*;
        use VLaneBinOp::*;
        let op = self;
        Some(match (op, lane) {
            (Add, I8) => 0x6e,
            (Add, I16) => 0x8e,
            (Add, I32) => 0xae,
            (Add, I64) => 0xce,
            (Add, F32) => 0xe4,
            (Add, F64) => 0xf0,
            (Sub, I8) => 0x71,
            (Sub, I16) => 0x91,
            (Sub, I32) => 0xb1,
            (Sub, I64) => 0xd1,
            (Sub, F32) => 0xe5,
            (Sub, F64) => 0xf1,
            // There is no `i8x16.mul`; byte multiplies would need widening.
            (Mul, I16) => 0x95,
            (Mul, I32) => 0xb5,
            (Mul, I64) => 0xd5,
            (Mul, F32) => 0xe6,
            (Mul, F64) => 0xf2,
            (Div, F32) => 0xe7,
            (Div, F64) => 0xf3,
            // The signed min/max entries double as the floating-point ones,
            // which the lane type disambiguates.
            (MinS, I8) => 0x76,
            (MinS, I16) => 0x96,
            (MinS, I32) => 0xb6,
            (MinS, F32) => 0xe8,
            (MinS, F64) => 0xf4,
            (MinU, I8) => 0x77,
            (MinU, I16) => 0x97,
            (MinU, I32) => 0xb7,
            (MaxS, I8) => 0x78,
            (MaxS, I16) => 0x98,
            (MaxS, I32) => 0xb8,
            (MaxS, F32) => 0xe9,
            (MaxS, F64) => 0xf5,
            (MaxU, I8) => 0x79,
            (MaxU, I16) => 0x99,
            (MaxU, I32) => 0xb9,
            (AddSatS, I8) => 0x6f,
            (AddSatS, I16) => 0x8f,
            (AddSatU, I8) => 0x70,
            (AddSatU, I16) => 0x90,
            (SubSatS, I8) => 0x72,
            (SubSatS, I16) => 0x92,
            (SubSatU, I8) => 0x73,
            (SubSatU, I16) => 0x93,
            (AvgrU, I8) => 0x7b,
            (AvgrU, I16) => 0x9b,
            (Q15MulrSatS, I16) => 0x82,
            (ExtMulLowS, I8) => 0x9c,
            (ExtMulHighS, I8) => 0x9d,
            (ExtMulLowS, I16) => 0xbc,
            (ExtMulHighS, I16) => 0xbd,
            (ExtMulLowS, I32) => 0xdc,
            (ExtMulHighS, I32) => 0xdd,
            (Eq, I8) => 0x23,
            (Eq, I16) => 0x2d,
            (Eq, I32) => 0x37,
            (Eq, I64) => 0xd6,
            (Eq, F32) => 0x41,
            (Eq, F64) => 0x47,
            (Ne, I8) => 0x24,
            (Ne, I16) => 0x2e,
            (Ne, I32) => 0x38,
            (Ne, I64) => 0xd7,
            (Ne, F32) => 0x42,
            (Ne, F64) => 0x48,
            (LtS, I8) => 0x25,
            (LtS, I16) => 0x2f,
            (LtS, I32) => 0x39,
            (LtS, I64) => 0xd8,
            (LtS, F32) => 0x43,
            (LtS, F64) => 0x49,
            (LtU, I8) => 0x26,
            (LtU, I16) => 0x30,
            (LtU, I32) => 0x3a,
            (GtS, I8) => 0x27,
            (GtS, I16) => 0x31,
            (GtS, I32) => 0x3b,
            (GtS, I64) => 0xd9,
            (GtS, F32) => 0x44,
            (GtS, F64) => 0x4a,
            (GtU, I8) => 0x28,
            (GtU, I16) => 0x32,
            (GtU, I32) => 0x3c,
            (LeS, I8) => 0x29,
            (LeS, I16) => 0x33,
            (LeS, I32) => 0x3d,
            (LeS, I64) => 0xda,
            (LeS, F32) => 0x45,
            (LeS, F64) => 0x4b,
            (LeU, I8) => 0x2a,
            (LeU, I16) => 0x34,
            (LeU, I32) => 0x3e,
            (GeS, I8) => 0x2b,
            (GeS, I16) => 0x35,
            (GeS, I32) => 0x3f,
            (GeS, I64) => 0xdb,
            (GeS, F32) => 0x46,
            (GeS, F64) => 0x4c,
            (GeU, I8) => 0x2c,
            (GeU, I16) => 0x36,
            (GeU, I32) => 0x40,
            _ => return None,
        })
    }
}

impl VLaneUnOp {
    /// The wasm SIMD opcode for this operation at this lane width.
    ///
    /// `None` means wasm cannot express it at all, so the lifter must refuse
    /// the instruction. Lifter and backend both consult this one table.
    pub fn wasm_opcode(self, lane: VLane) -> Option<u32> {
        use VLane::*;
        use VLaneUnOp::*;
        let op = self;
        Some(match (op, lane) {
            (Abs, I8) => 0x60,
            (Abs, I16) => 0x80,
            (Abs, I32) => 0xa0,
            (Abs, I64) => 0xc0,
            (Abs, F32) => 0xe0,
            (Abs, F64) => 0xec,
            (Neg, I8) => 0x61,
            (Neg, I16) => 0x81,
            (Neg, I32) => 0xa1,
            (Neg, I64) => 0xc1,
            (Neg, F32) => 0xe1,
            (Neg, F64) => 0xed,
            (Sqrt, F32) => 0xe3,
            (Sqrt, F64) => 0xef,
            (Ceil, F32) => 0x67,
            (Ceil, F64) => 0x74,
            (Floor, F32) => 0x68,
            (Floor, F64) => 0x75,
            (Trunc, F32) => 0x69,
            (Trunc, F64) => 0x7a,
            (Nearest, F32) => 0x6a,
            (Nearest, F64) => 0x94,
            _ => return None,
        })
    }
}
