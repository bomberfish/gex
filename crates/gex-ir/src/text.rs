//! A textual form of function-level GIR that prints and parses back.
//!
//! `print(f)` then `parse(..)` gives back a [`Function`] equal to `f`, which
//! the tests check on every function recovered from real binaries. The
//! syntax is line-based and regular, one statement per line, so it is also
//! meant for reading: `gexc funcs --ir` prints it.
//!
//! ```text
//! func 0x4001f0 "main" entries [] form ssa values 12
//! b0 0x4001f0..0x4001fc:
//!   @4001f0 v0 = read sp
//!   @4001f0 v1 = bin sub i64 v0 v2
//!   @4001f8 write sp v1
//!   @4001f8 call 0x400300 ret 0x4001fc -> b1(v1)
//! b1 0x4001fc..0x400204 (sp v3):
//!   @400200 return v4
//! ```
//!
//! Operators print as their `Debug` names (lowercased where that is
//! unambiguous) with the spaces taken out, so each is one token.

use crate::func::*;
use crate::*;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

/// The printed name of a register.
pub fn reg_name(r: GuestReg) -> String {
    match r {
        GuestReg::X(n) => format!("x{n}"),
        GuestReg::V128(n) => format!("q{n}"),
        GuestReg::Sp => "sp".into(),
        GuestReg::Pc => "pc".into(),
        GuestReg::N => "nf".into(),
        GuestReg::Z => "zf".into(),
        GuestReg::C => "cf".into(),
        GuestReg::V => "vf".into(),
        GuestReg::P => "pf".into(),
        GuestReg::A => "af".into(),
        GuestReg::D => "df".into(),
        GuestReg::Tpidr => "tpidr".into(),
        GuestReg::TpidrRo => "tpidrro".into(),
        GuestReg::MonAddr => "mon.addr".into(),
        GuestReg::MonValue => "mon.val".into(),
        GuestReg::MonSize => "mon.size".into(),
        GuestReg::Mxcsr => "mxcsr".into(),
        GuestReg::X87Top => "x87.top".into(),
        GuestReg::X87Cw => "x87.cw".into(),
        GuestReg::X87Sw => "x87.sw".into(),
        GuestReg::X87Tags => "x87.tags".into(),
        GuestReg::CtxAddr => "ctx".into(),
    }
}

fn parse_reg(s: &str) -> Result<GuestReg, String> {
    (0..N_REGS)
        .map(reg_of_index)
        .find(|&r| reg_name(r) == s)
        .ok_or_else(|| format!("unknown register `{s}`"))
}

/// One token for an enum value: its `Debug` form without spaces.
fn tok<T: std::fmt::Debug>(t: T) -> String {
    format!("{t:?}").replace(' ', "")
}

fn parse_tok<T: std::fmt::Debug + Copy>(all: &[T], s: &str) -> Result<T, String> {
    all.iter()
        .copied()
        .find(|&t| tok(t) == s)
        .ok_or_else(|| format!("unexpected `{s}`"))
}

const TYS: [Ty; 3] = [Ty::I32, Ty::I64, Ty::V128];
const BINOPS: [BinOp; 14] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::ShrU,
    BinOp::ShrS,
    BinOp::Rotr,
    BinOp::DivU,
    BinOp::DivS,
    BinOp::RemU,
    BinOp::RemS,
];
const UNOPS: [UnOp; 7] = [
    UnOp::Clz,
    UnOp::Ctz,
    UnOp::Popcnt,
    UnOp::Not,
    UnOp::Neg,
    UnOp::Bswap,
    UnOp::Bswap16,
];
const CMPOPS: [CmpOp; 10] = [
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::LtU,
    CmpOp::LtS,
    CmpOp::LeU,
    CmpOp::LeS,
    CmpOp::GtU,
    CmpOp::GtS,
    CmpOp::GeU,
    CmpOp::GeS,
];
const MEMSZ: [MemSz; 4] = [MemSz::B1, MemSz::B2, MemSz::B4, MemSz::B8];
const ORDERS: [Order; 2] = [Order::Relaxed, Order::SeqCst];
const RMWOPS: [RmwOp; 6] = [
    RmwOp::Add,
    RmwOp::Sub,
    RmwOp::And,
    RmwOp::Or,
    RmwOp::Xor,
    RmwOp::Xchg,
];
const HOSTS: [HostFunc; 8] = [
    HostFunc::Syscall,
    HostFunc::Illegal,
    HostFunc::Breakpoint,
    HostFunc::SysRegRead,
    HostFunc::SysRegWrite,
    HostFunc::CacheOp,
    HostFunc::Safepoint,
    HostFunc::Fma,
];
const VBINOPS: [VBinOp; 4] = [VBinOp::And, VBinOp::Or, VBinOp::Xor, VBinOp::AndNot];
const VUNOPS: [VUnOp; 5] = [
    VUnOp::PopcntBytes,
    VUnOp::ExtAddPairwiseU8,
    VUnOp::ExtAddPairwiseU16,
    VUnOp::ExtAddPairwiseS8,
    VUnOp::ExtAddPairwiseS16,
];
const VLANEBIN: [VLaneBinOp; 26] = [
    VLaneBinOp::Add,
    VLaneBinOp::Sub,
    VLaneBinOp::Mul,
    VLaneBinOp::Div,
    VLaneBinOp::MinS,
    VLaneBinOp::MinU,
    VLaneBinOp::MaxS,
    VLaneBinOp::MaxU,
    VLaneBinOp::AddSatS,
    VLaneBinOp::AddSatU,
    VLaneBinOp::SubSatS,
    VLaneBinOp::SubSatU,
    VLaneBinOp::AvgrU,
    VLaneBinOp::Q15MulrSatS,
    VLaneBinOp::ExtMulLowS,
    VLaneBinOp::ExtMulHighS,
    VLaneBinOp::Eq,
    VLaneBinOp::Ne,
    VLaneBinOp::LtS,
    VLaneBinOp::LtU,
    VLaneBinOp::GtS,
    VLaneBinOp::GtU,
    VLaneBinOp::LeS,
    VLaneBinOp::LeU,
    VLaneBinOp::GeS,
    VLaneBinOp::GeU,
];
const VLANEUN: [VLaneUnOp; 7] = [
    VLaneUnOp::Neg,
    VLaneUnOp::Abs,
    VLaneUnOp::Sqrt,
    VLaneUnOp::Ceil,
    VLaneUnOp::Floor,
    VLaneUnOp::Trunc,
    VLaneUnOp::Nearest,
];
const VSHIFT: [VShiftOp; 3] = [VShiftOp::Shl, VShiftOp::ShrS, VShiftOp::ShrU];
const VCVT: [VCvtOp; 12] = [
    VCvtOp::FpToIntSat { signed: false },
    VCvtOp::FpToIntSat { signed: true },
    VCvtOp::IntToFp { signed: false },
    VCvtOp::IntToFp { signed: true },
    VCvtOp::DemoteF64,
    VCvtOp::PromoteF32,
    VCvtOp::ExtendLow { signed: false },
    VCvtOp::ExtendLow { signed: true },
    VCvtOp::ExtendHigh { signed: false },
    VCvtOp::ExtendHigh { signed: true },
    VCvtOp::NarrowSat { signed: false },
    VCvtOp::NarrowSat { signed: true },
];
const VLANES: [VLane; 6] = [
    VLane::I8,
    VLane::I16,
    VLane::I32,
    VLane::I64,
    VLane::F32,
    VLane::F64,
];
const FPBIN: [FpBinOp; 7] = [
    FpBinOp::Add,
    FpBinOp::Sub,
    FpBinOp::Mul,
    FpBinOp::Div,
    FpBinOp::Min,
    FpBinOp::Max,
    FpBinOp::Copysign,
];
const FPUN: [FpUnOp; 7] = [
    FpUnOp::Abs,
    FpUnOp::Neg,
    FpUnOp::Sqrt,
    FpUnOp::RoundNearest,
    FpUnOp::Ceil,
    FpUnOp::Floor,
    FpUnOp::Trunc,
];
const FPCMP: [FpCmpOp; 6] = [
    FpCmpOp::Eq,
    FpCmpOp::Ne,
    FpCmpOp::Lt,
    FpCmpOp::Le,
    FpCmpOp::Gt,
    FpCmpOp::Ge,
];
const FPCVT: [FpConvert; 5] = [
    FpConvert::IntToFp { signed: false },
    FpConvert::IntToFp { signed: true },
    FpConvert::FpToInt { signed: false },
    FpConvert::FpToInt { signed: true },
    FpConvert::FpToFp,
];
const STRKIND: [StrKind; 4] = [StrKind::Movs, StrKind::Stos, StrKind::Scas, StrKind::Cmps];
const STRUNTIL: [StrUntil; 3] = [StrUntil::Count, StrUntil::NotEqual, StrUntil::Equal];

fn ty(t: Ty) -> &'static str {
    match t {
        Ty::I32 => "i32",
        Ty::I64 => "i64",
        Ty::V128 => "v128",
    }
}

fn parse_ty(s: &str) -> Result<Ty, String> {
    TYS.iter()
        .copied()
        .find(|&t| ty(t) == s)
        .ok_or_else(|| format!("unknown type `{s}`"))
}

fn fw(double: bool) -> &'static str {
    if double {
        "f64"
    } else {
        "f32"
    }
}

fn su(signed: bool) -> &'static str {
    if signed {
        "s"
    } else {
        "u"
    }
}

fn hex16(b: &[u8; 16]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn v(x: Value) -> String {
    format!("v{}", x.0)
}

fn vs(xs: &[Value]) -> String {
    let parts: Vec<String> = xs.iter().map(|&x| v(x)).collect();
    format!("[{}]", parts.join(","))
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

/// Print one expression.
pub fn expr(e: &Expr) -> String {
    match e {
        Expr::ConstI32(c) => format!("i32 {c:#x}"),
        Expr::ConstI64(c) => format!("i64 {c:#x}"),
        Expr::ReadReg(r) => format!("read {}", reg_name(*r)),
        Expr::Bin { op, ty: t, a, b } => format!("bin {} {} {} {}", tok(op), ty(*t), v(*a), v(*b)),
        Expr::Un { op, ty: t, a } => format!("un {} {} {}", tok(op), ty(*t), v(*a)),
        Expr::Cmp { op, ty: t, a, b } => format!("cmp {} {} {} {}", tok(op), ty(*t), v(*a), v(*b)),
        Expr::Select { cond, a, b, ty: t } => {
            format!("select {} {} {} {}", ty(*t), v(*cond), v(*a), v(*b))
        }
        Expr::Wrap(a) => format!("wrap {}", v(*a)),
        Expr::Extend { a, signed } => format!("extend {} {}", su(*signed), v(*a)),
        Expr::ExtendBits {
            a,
            ty: t,
            bits,
            signed,
        } => format!("extbits {} {bits} {} {}", ty(*t), su(*signed), v(*a)),
        Expr::Load {
            addr,
            offset,
            sz,
            signed,
            ty: t,
            order,
        } => format!(
            "load {} {} {} {} {} {offset}",
            tok(sz),
            su(*signed),
            ty(*t),
            tok(order),
            v(*addr)
        ),
        Expr::Rmw {
            op,
            addr,
            val,
            sz,
            ty: t,
        } => format!("rmw {} {} {} {} {}", tok(op), tok(sz), ty(*t), v(*addr), v(*val)),
        Expr::CmpXchg {
            addr,
            expected,
            replacement,
            sz,
            ty: t,
        } => format!(
            "cmpxchg {} {} {} {} {}",
            tok(sz),
            ty(*t),
            v(*addr),
            v(*expected),
            v(*replacement)
        ),
        Expr::HostCall { func, args, ty: t } => {
            format!("host {} {} {}", tok(func), ty(*t), vs(args))
        }
        Expr::VLoad {
            addr,
            offset,
            bytes,
        } => format!("vload {bytes} {} {offset}", v(*addr)),
        Expr::VConst(b) => format!("vconst {}", hex16(b)),
        Expr::VSplat { a, lane_bytes } => format!("vsplat {lane_bytes} {}", v(*a)),
        Expr::VBin { op, a, b } => format!("vbin {} {} {}", tok(op), v(*a), v(*b)),
        Expr::VUn { op, a } => format!("vun {} {}", tok(op), v(*a)),
        Expr::VLaneBin { op, lane, a, b } => {
            format!("vlanebin {} {} {} {}", tok(op), tok(lane), v(*a), v(*b))
        }
        Expr::VLaneUn { op, lane, a } => format!("vlaneun {} {} {}", tok(op), tok(lane), v(*a)),
        Expr::VShift {
            op,
            lane,
            a,
            amount,
        } => format!("vshift {} {} {} {}", tok(op), tok(lane), v(*a), v(*amount)),
        Expr::VShuffle { a, b, lanes } => {
            format!("vshuffle {} {} {}", v(*a), v(*b), hex16(lanes))
        }
        Expr::VSwizzle { a, idx } => format!("vswizzle {} {}", v(*a), v(*idx)),
        Expr::VBitSelect { mask, a, b } => {
            format!("vbitselect {} {} {}", v(*mask), v(*a), v(*b))
        }
        Expr::VCvt { op, from, a, b } => format!(
            "vcvt {} {} {} {}",
            tok(op),
            tok(from),
            v(*a),
            b.map_or("-".to_string(), v)
        ),
        Expr::VExtract {
            a,
            lane,
            lane_bytes,
            signed,
            ty: t,
        } => format!(
            "vextract {lane} {lane_bytes} {} {} {}",
            su(*signed),
            ty(*t),
            v(*a)
        ),
        Expr::VReplace {
            a,
            lane,
            lane_bytes,
            val,
        } => format!("vreplace {lane} {lane_bytes} {} {}", v(*a), v(*val)),
        Expr::VBitmask { a, lane } => format!("vbitmask {} {}", tok(lane), v(*a)),
        Expr::FpBin { op, double, a, b } => {
            format!("fbin {} {} {} {}", tok(op), fw(*double), v(*a), v(*b))
        }
        Expr::FpUn { op, double, a } => format!("fun {} {} {}", tok(op), fw(*double), v(*a)),
        Expr::FpCmp { op, double, a, b } => {
            format!("fcmp {} {} {} {}", tok(op), fw(*double), v(*a), v(*b))
        }
        Expr::FpCvt {
            kind,
            src_wide,
            dst_wide,
            a,
        } => format!(
            "fcvt {} {} {} {}",
            tok(kind),
            fw(*src_wide),
            fw(*dst_wide),
            v(*a)
        ),
        Expr::Mx { mx, e } => format!("mx {} {}", v(*mx), expr(e)),
    }
}

/// Print one statement (without its pc).
pub fn stmt(s: &Stmt) -> String {
    match s {
        Stmt::Def { dst, expr: e } => format!("{} = {}", v(*dst), expr(e)),
        Stmt::WriteReg { reg, val } => format!("write {} {}", reg_name(*reg), v(*val)),
        Stmt::Store {
            addr,
            offset,
            val,
            sz,
            order,
        } => format!(
            "store {} {} {} {offset} {}",
            tok(sz),
            tok(order),
            v(*addr),
            v(*val)
        ),
        Stmt::HostCall { func, args } => format!("host {} {}", tok(func), vs(args)),
        Stmt::X87 { op, args, outs } => format!("{} = x87 {} {}", vs(outs), x87_tok(*op), vs(args)),
        Stmt::Trap { cond, pc, code } => format!("trap {} {:#x} {}", v(*cond), pc, code),
        Stmt::VStore {
            addr,
            offset,
            val,
            bytes,
        } => format!("vstore {bytes} {} {offset} {}", v(*addr), v(*val)),
        Stmt::MemCopy { dst, src, len } => {
            format!("memcopy {} {} {}", v(*dst), v(*src), v(*len))
        }
        Stmt::MemFill { dst, val, len } => {
            format!("memfill {} {} {}", v(*dst), v(*val), v(*len))
        }
        Stmt::MemCmp { dst, a, b, len } => {
            format!("{} = memcmp {} {} {}", v(*dst), v(*a), v(*b), v(*len))
        }
        Stmt::Fence => "fence".into(),
        Stmt::StrOp {
            kind,
            sz,
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
        } => format!(
            "{} = strop {} {} {} {} {} {} {} {}",
            vs(&[*out_count, *out_dst, *out_src, *out_a, *out_b]),
            tok(kind),
            tok(sz),
            tok(until),
            v(*count),
            v(*dst),
            v(*src),
            v(*val),
            v(*df)
        ),
    }
}

fn edge(e: &Edge) -> String {
    format!("{:?}{}", e.to, vs(&e.args).replacen('[', "(", 1).replacen(']', ")", 1))
}

fn callee(c: &Callee) -> String {
    match c {
        Callee::Direct(a) => format!("{a:#x}"),
        Callee::Indirect(x) => v(*x),
    }
}

/// Print a terminator.
pub fn term(t: &FTerm) -> String {
    match t {
        FTerm::Jump(e) => format!("jump {}", edge(e)),
        FTerm::Branch {
            cond,
            taken,
            not_taken,
        } => format!("br {} {} {}", v(*cond), edge(taken), edge(not_taken)),
        FTerm::Switch {
            target,
            cases,
            table,
        } => {
            let mut s = format!(
                "switch {} table {:#x} {} {} {} {}",
                v(*target),
                table.base,
                table.entry_bytes,
                table.count,
                table.index.map_or("-".to_string(), reg_name),
                if table.complete { "complete" } else { "partial" }
            );
            for (a, e) in cases {
                let _ = write!(s, " {a:#x}:{}", edge(e));
            }
            s
        }
        FTerm::Call { callee: c, ret_pc, ret } => format!(
            "call {} ret {ret_pc:#x} -> {}",
            callee(c),
            ret.as_ref().map_or("noreturn".to_string(), edge)
        ),
        FTerm::TailCall { callee: c } => format!("tailcall {}", callee(c)),
        FTerm::Return { target } => format!("return {}", v(*target)),
        FTerm::Unresolved { target } => format!("unresolved {}", v(*target)),
        FTerm::Syscall { next_pc, next, imm } => format!(
            "syscall{} {next_pc:#x} -> {}",
            if *imm != 0 { format!(" #{imm}") } else { String::new() },
            next.as_ref().map_or("noreturn".to_string(), edge)
        ),
        FTerm::Illegal { raw, pc } => format!("illegal {raw:#x} {pc:#x}"),
    }
}

/// Print a whole function.
pub fn print(f: &Function) -> String {
    let mut s = String::new();
    let entries: Vec<String> = f.entries.iter().map(|e| format!("{e:#x}")).collect();
    let _ = writeln!(
        s,
        "func {:#x} {} entries [{}] form {} values {}",
        f.entry,
        match &f.name {
            // Names are C/C++ symbols; quote them so spaces (demangled or
            // not) cannot break the token stream.
            Some(n) => format!("{n:?}"),
            None => "-".into(),
        },
        entries.join(","),
        match f.form {
            Form::Memory => "memory",
            Form::Ssa => "ssa",
        },
        f.n_values
    );
    for (i, b) in f.blocks.iter().enumerate() {
        let _ = write!(s, "b{i} {:#x}..{:#x}", b.start, b.end);
        if b.synthetic {
            let _ = write!(s, " synthetic");
        }
        if !b.params.is_empty() {
            let ps: Vec<String> = b
                .params
                .iter()
                .map(|p| format!("{} {}", reg_name(p.reg), v(p.val)))
                .collect();
            let _ = write!(s, " ({})", ps.join(", "));
        }
        let insts: Vec<String> = b.insts.iter().map(|a| format!("{a:x}")).collect();
        let _ = writeln!(s, " insts [{}]:", insts.join(","));
        for (st, pc) in b.stmts.iter().zip(&b.pcs) {
            let _ = writeln!(s, "  @{pc:x} {}", stmt(st));
        }
        let _ = writeln!(s, "  @{:x} {}", b.term_pc, term(&b.term));
    }
    s
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

struct Toks<'a> {
    t: Vec<&'a str>,
    i: usize,
    line: usize,
}

impl<'a> Toks<'a> {
    fn new(s: &'a str, line: usize) -> Toks<'a> {
        Toks {
            t: s.split_whitespace().collect(),
            i: 0,
            line,
        }
    }
    fn err(&self, m: impl std::fmt::Display) -> String {
        format!("line {}: {m}", self.line + 1)
    }
    fn next(&mut self) -> Result<&'a str, String> {
        let t = self
            .t
            .get(self.i)
            .copied()
            .ok_or_else(|| self.err("unexpected end of line"))?;
        self.i += 1;
        Ok(t)
    }
    fn peek(&self) -> Option<&'a str> {
        self.t.get(self.i).copied()
    }
    fn expect(&mut self, want: &str) -> Result<(), String> {
        let t = self.next()?;
        if t != want {
            return Err(self.err(format!("expected `{want}`, found `{t}`")));
        }
        Ok(())
    }
    fn done(&self) -> Result<(), String> {
        if self.i != self.t.len() {
            return Err(self.err(format!("trailing `{}`", self.t[self.i..].join(" "))));
        }
        Ok(())
    }
    fn wrap<T>(&self, r: Result<T, String>) -> Result<T, String> {
        r.map_err(|e| self.err(e))
    }
    fn val(&mut self) -> Result<Value, String> {
        let t = self.next()?;
        self.wrap(parse_val(t))
    }
    fn vals(&mut self) -> Result<Vec<Value>, String> {
        let t = self.next()?;
        self.wrap(parse_vals(t, '[', ']'))
    }
    fn num(&mut self) -> Result<u64, String> {
        let t = self.next()?;
        self.wrap(parse_num(t))
    }
    fn int(&mut self) -> Result<i64, String> {
        let t = self.next()?;
        t.parse::<i64>().map_err(|_| self.err(format!("bad integer `{t}`")))
    }
    fn u32(&mut self) -> Result<u32, String> {
        let t = self.next()?;
        t.parse::<u32>().map_err(|_| self.err(format!("bad integer `{t}`")))
    }
    fn ty(&mut self) -> Result<Ty, String> {
        let t = self.next()?;
        self.wrap(parse_ty(t))
    }
    fn reg(&mut self) -> Result<GuestReg, String> {
        let t = self.next()?;
        self.wrap(parse_reg(t))
    }
    fn signed(&mut self) -> Result<bool, String> {
        match self.next()? {
            "s" => Ok(true),
            "u" => Ok(false),
            t => Err(self.err(format!("expected s or u, found `{t}`"))),
        }
    }
    fn fw(&mut self) -> Result<bool, String> {
        match self.next()? {
            "f64" => Ok(true),
            "f32" => Ok(false),
            t => Err(self.err(format!("expected f32 or f64, found `{t}`"))),
        }
    }
    fn e<T: std::fmt::Debug + Copy>(&mut self, all: &[T]) -> Result<T, String> {
        let t = self.next()?;
        self.wrap(parse_tok(all, t))
    }
    fn hex16(&mut self) -> Result<Box<[u8; 16]>, String> {
        let t = self.next()?;
        if t.len() != 32 {
            return Err(self.err(format!("expected 32 hex digits, found `{t}`")));
        }
        let mut b = [0u8; 16];
        for (i, x) in b.iter_mut().enumerate() {
            *x = u8::from_str_radix(&t[2 * i..2 * i + 2], 16)
                .map_err(|_| self.err(format!("bad hex `{t}`")))?;
        }
        Ok(Box::new(b))
    }
}

fn parse_num(t: &str) -> Result<u64, String> {
    match t.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => t.parse::<u64>(),
    }
    .map_err(|_| format!("bad number `{t}`"))
}

fn parse_val(t: &str) -> Result<Value, String> {
    t.strip_prefix('v')
        .and_then(|n| n.parse::<u32>().ok())
        .map(Value)
        .ok_or_else(|| format!("expected a value, found `{t}`"))
}

fn parse_vals(t: &str, open: char, close: char) -> Result<Vec<Value>, String> {
    let inner = t
        .strip_prefix(open)
        .and_then(|x| x.strip_suffix(close))
        .ok_or_else(|| format!("expected {open}..{close}, found `{t}`"))?;
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner.split(',').map(parse_val).collect()
}

fn parse_edge(t: &str) -> Result<Edge, String> {
    let open = t.find('(').ok_or_else(|| format!("bad edge `{t}`"))?;
    let to = t[..open]
        .strip_prefix('b')
        .and_then(|n| n.parse::<u32>().ok())
        .ok_or_else(|| format!("bad block `{t}`"))?;
    Ok(Edge {
        to: BlockId(to),
        args: parse_vals(&t[open..], '(', ')')?,
    })
}

fn parse_callee(t: &str) -> Result<Callee, String> {
    if t.starts_with('v') {
        parse_val(t).map(Callee::Indirect)
    } else {
        parse_num(t).map(Callee::Direct)
    }
}

fn parse_expr(k: &mut Toks) -> Result<Expr, String> {
    let op = k.next()?;
    Ok(match op {
        "i32" => Expr::ConstI32(k.num()? as u32),
        "i64" => Expr::ConstI64(k.num()?),
        "read" => Expr::ReadReg(k.reg()?),
        "bin" => {
            let op = k.e(&BINOPS)?;
            let ty = k.ty()?;
            Expr::Bin {
                op,
                ty,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "un" => {
            let op = k.e(&UNOPS)?;
            let ty = k.ty()?;
            Expr::Un { op, ty, a: k.val()? }
        }
        "cmp" => {
            let op = k.e(&CMPOPS)?;
            let ty = k.ty()?;
            Expr::Cmp {
                op,
                ty,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "select" => {
            let ty = k.ty()?;
            Expr::Select {
                ty,
                cond: k.val()?,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "wrap" => Expr::Wrap(k.val()?),
        "extend" => {
            let signed = k.signed()?;
            Expr::Extend { signed, a: k.val()? }
        }
        "extbits" => {
            let ty = k.ty()?;
            let bits = k.u32()?;
            let signed = k.signed()?;
            Expr::ExtendBits {
                ty,
                bits,
                signed,
                a: k.val()?,
            }
        }
        "load" => {
            let sz = k.e(&MEMSZ)?;
            let signed = k.signed()?;
            let ty = k.ty()?;
            let order = k.e(&ORDERS)?;
            let addr = k.val()?;
            Expr::Load {
                sz,
                signed,
                ty,
                order,
                addr,
                offset: k.int()?,
            }
        }
        "rmw" => {
            let op = k.e(&RMWOPS)?;
            let sz = k.e(&MEMSZ)?;
            let ty = k.ty()?;
            Expr::Rmw {
                op,
                sz,
                ty,
                addr: k.val()?,
                val: k.val()?,
            }
        }
        "cmpxchg" => {
            let sz = k.e(&MEMSZ)?;
            let ty = k.ty()?;
            Expr::CmpXchg {
                sz,
                ty,
                addr: k.val()?,
                expected: k.val()?,
                replacement: k.val()?,
            }
        }
        "host" => {
            let func = k.e(&HOSTS)?;
            let ty = k.ty()?;
            Expr::HostCall {
                func,
                ty,
                args: k.vals()?,
            }
        }
        "vload" => {
            let bytes = k.u32()?;
            let addr = k.val()?;
            Expr::VLoad {
                bytes,
                addr,
                offset: k.int()?,
            }
        }
        "vconst" => Expr::VConst(k.hex16()?),
        "vsplat" => {
            let lane_bytes = k.u32()?;
            Expr::VSplat {
                lane_bytes,
                a: k.val()?,
            }
        }
        "vbin" => {
            let op = k.e(&VBINOPS)?;
            Expr::VBin {
                op,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "vun" => {
            let op = k.e(&VUNOPS)?;
            Expr::VUn { op, a: k.val()? }
        }
        "vlanebin" => {
            let op = k.e(&VLANEBIN)?;
            let lane = k.e(&VLANES)?;
            Expr::VLaneBin {
                op,
                lane,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "vlaneun" => {
            let op = k.e(&VLANEUN)?;
            let lane = k.e(&VLANES)?;
            Expr::VLaneUn {
                op,
                lane,
                a: k.val()?,
            }
        }
        "vshift" => {
            let op = k.e(&VSHIFT)?;
            let lane = k.e(&VLANES)?;
            Expr::VShift {
                op,
                lane,
                a: k.val()?,
                amount: k.val()?,
            }
        }
        "vshuffle" => {
            let a = k.val()?;
            let b = k.val()?;
            Expr::VShuffle {
                a,
                b,
                lanes: k.hex16()?,
            }
        }
        "vswizzle" => Expr::VSwizzle {
            a: k.val()?,
            idx: k.val()?,
        },
        "vbitselect" => Expr::VBitSelect {
            mask: k.val()?,
            a: k.val()?,
            b: k.val()?,
        },
        "vcvt" => {
            let op = k.e(&VCVT)?;
            let from = k.e(&VLANES)?;
            let a = k.val()?;
            let b = match k.next()? {
                "-" => None,
                t => Some(k.wrap(parse_val(t))?),
            };
            Expr::VCvt { op, from, a, b }
        }
        "vextract" => {
            let lane = k.u32()?;
            let lane_bytes = k.u32()?;
            let signed = k.signed()?;
            let ty = k.ty()?;
            Expr::VExtract {
                lane,
                lane_bytes,
                signed,
                ty,
                a: k.val()?,
            }
        }
        "vreplace" => {
            let lane = k.u32()?;
            let lane_bytes = k.u32()?;
            Expr::VReplace {
                lane,
                lane_bytes,
                a: k.val()?,
                val: k.val()?,
            }
        }
        "vbitmask" => {
            let lane = k.e(&VLANES)?;
            Expr::VBitmask { lane, a: k.val()? }
        }
        "fbin" => {
            let op = k.e(&FPBIN)?;
            let double = k.fw()?;
            Expr::FpBin {
                op,
                double,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "fun" => {
            let op = k.e(&FPUN)?;
            let double = k.fw()?;
            Expr::FpUn {
                op,
                double,
                a: k.val()?,
            }
        }
        "fcmp" => {
            let op = k.e(&FPCMP)?;
            let double = k.fw()?;
            Expr::FpCmp {
                op,
                double,
                a: k.val()?,
                b: k.val()?,
            }
        }
        "fcvt" => {
            let kind = k.e(&FPCVT)?;
            let src_wide = k.fw()?;
            let dst_wide = k.fw()?;
            Expr::FpCvt {
                kind,
                src_wide,
                dst_wide,
                a: k.val()?,
            }
        }
        "mx" => {
            let mx = k.val()?;
            let e = parse_expr(k)?;
            if !e.mx_able() {
                return Err(k.err("mx: not an operation MXCSR rounds".to_string()));
            }
            Expr::Mx { mx, e: Box::new(e) }
        }
        other => return Err(k.err(format!("unknown expression `{other}`"))),
    })
}

/// The text form of an [`X87Op`]: `narrow32`, `narrow64`, `toint16` ..
/// `toint64`, `trunc16` .. `trunc64`, `add`, `sub`, `mul`, `div`, `com`,
/// `ucom`.
fn x87_tok(op: X87Op) -> String {
    match op {
        X87Op::Narrow { single } => format!("narrow{}", if single { 32 } else { 64 }),
        X87Op::ToInt { bits, trunc } => format!("{}{bits}", if trunc { "trunc" } else { "toint" }),
        X87Op::Arith(k) => match k {
            X87Arith::Add => "add",
            X87Arith::Sub => "sub",
            X87Arith::Mul => "mul",
            X87Arith::Div => "div",
        }
        .into(),
        X87Op::Compare { quiet } => if quiet { "ucom" } else { "com" }.into(),
    }
}

fn parse_x87(t: &str) -> Option<X87Op> {
    Some(match t {
        "narrow32" => X87Op::Narrow { single: true },
        "narrow64" => X87Op::Narrow { single: false },
        "add" => X87Op::Arith(X87Arith::Add),
        "sub" => X87Op::Arith(X87Arith::Sub),
        "mul" => X87Op::Arith(X87Arith::Mul),
        "div" => X87Op::Arith(X87Arith::Div),
        "com" => X87Op::Compare { quiet: false },
        "ucom" => X87Op::Compare { quiet: true },
        _ => {
            let (trunc, n) = if let Some(n) = t.strip_prefix("trunc") {
                (true, n)
            } else {
                (false, t.strip_prefix("toint")?)
            };
            let bits: u8 = n.parse().ok().filter(|b| matches!(b, 16 | 32 | 64))?;
            X87Op::ToInt { bits, trunc }
        }
    })
}

fn parse_stmt(k: &mut Toks) -> Result<Stmt, String> {
    let first = k.peek().ok_or_else(|| k.err("empty statement"))?;
    if first.starts_with('[') {
        // [out...] = strop ... | [out...] = x87 ...
        let outs = k.vals()?;
        k.expect("=")?;
        if k.peek() == Some("x87") {
            k.next()?;
            let t = k.next()?;
            let op = parse_x87(t).ok_or_else(|| k.err(format!("unknown x87 operation `{t}`")))?;
            let args = k.vals()?;
            if args.len() != op.arg_types().len() || outs.len() != op.out_types().len() {
                return Err(k.err(format!("x87 {t} takes {} values and defines {}", op.arg_types().len(), op.out_types().len())));
            }
            return Ok(Stmt::X87 { op, args, outs });
        }
        k.expect("strop")?;
        if outs.len() != 5 {
            return Err(k.err("strop defines five values"));
        }
        let kind = k.e(&STRKIND)?;
        let sz = k.e(&MEMSZ)?;
        let until = k.e(&STRUNTIL)?;
        return Ok(Stmt::StrOp {
            kind,
            sz,
            until,
            count: k.val()?,
            dst: k.val()?,
            src: k.val()?,
            val: k.val()?,
            df: k.val()?,
            out_count: outs[0],
            out_dst: outs[1],
            out_src: outs[2],
            out_a: outs[3],
            out_b: outs[4],
        });
    }
    if first.starts_with('v') && first[1..].chars().all(|c| c.is_ascii_digit()) {
        let dst = k.val()?;
        k.expect("=")?;
        if k.peek() == Some("memcmp") {
            k.next()?;
            return Ok(Stmt::MemCmp {
                dst,
                a: k.val()?,
                b: k.val()?,
                len: k.val()?,
            });
        }
        return Ok(Stmt::Def {
            dst,
            expr: parse_expr(k)?,
        });
    }
    Ok(match k.next()? {
        "write" => Stmt::WriteReg {
            reg: k.reg()?,
            val: k.val()?,
        },
        "store" => {
            let sz = k.e(&MEMSZ)?;
            let order = k.e(&ORDERS)?;
            let addr = k.val()?;
            let offset = k.int()?;
            Stmt::Store {
                sz,
                order,
                addr,
                offset,
                val: k.val()?,
            }
        }
        "host" => Stmt::HostCall {
            func: k.e(&HOSTS)?,
            args: k.vals()?,
        },
        "trap" => {
            let cond = k.val()?;
            let pc = k.num()?;
            let code = k.int()?;
            Stmt::Trap {
                cond,
                pc,
                code: i32::try_from(code).map_err(|_| k.err(format!("trap code {code} out of range")))?,
            }
        }
        "vstore" => {
            let bytes = k.u32()?;
            let addr = k.val()?;
            let offset = k.int()?;
            Stmt::VStore {
                bytes,
                addr,
                offset,
                val: k.val()?,
            }
        }
        "memcopy" => Stmt::MemCopy {
            dst: k.val()?,
            src: k.val()?,
            len: k.val()?,
        },
        "memfill" => Stmt::MemFill {
            dst: k.val()?,
            val: k.val()?,
            len: k.val()?,
        },
        "fence" => Stmt::Fence,
        other => return Err(k.err(format!("unknown statement `{other}`"))),
    })
}

fn parse_term(k: &mut Toks) -> Result<FTerm, String> {
    let edge = |k: &mut Toks| -> Result<Edge, String> {
        let t = k.next()?;
        k.wrap(parse_edge(t))
    };
    let opt_edge = |k: &mut Toks| -> Result<Option<Edge>, String> {
        let t = k.next()?;
        if t == "noreturn" {
            Ok(None)
        } else {
            k.wrap(parse_edge(t)).map(Some)
        }
    };
    Ok(match k.next()? {
        "jump" => FTerm::Jump(edge(k)?),
        "br" => FTerm::Branch {
            cond: k.val()?,
            taken: edge(k)?,
            not_taken: edge(k)?,
        },
        "switch" => {
            let target = k.val()?;
            k.expect("table")?;
            let base = k.num()?;
            let entry_bytes = k.u32()?;
            let count = k.u32()?;
            let index = match k.next()? {
                "-" => None,
                t => Some(k.wrap(parse_reg(t))?),
            };
            let complete = match k.next()? {
                "complete" => true,
                "partial" => false,
                t => return Err(k.err(format!("expected complete or partial, found `{t}`"))),
            };
            let mut cases = Vec::new();
            while let Some(t) = k.peek() {
                k.next()?;
                let (a, e) = t
                    .split_once(':')
                    .ok_or_else(|| k.err(format!("bad case `{t}`")))?;
                cases.push((k.wrap(parse_num(a))?, k.wrap(parse_edge(e))?));
            }
            FTerm::Switch {
                target,
                cases,
                table: JumpTable {
                    base,
                    entry_bytes,
                    count,
                    index,
                    complete,
                },
            }
        }
        "call" => {
            let t = k.next()?;
            let callee = k.wrap(parse_callee(t))?;
            k.expect("ret")?;
            let ret_pc = k.num()?;
            k.expect("->")?;
            FTerm::Call {
                callee,
                ret_pc,
                ret: opt_edge(k)?,
            }
        }
        "tailcall" => {
            let t = k.next()?;
            FTerm::TailCall {
                callee: k.wrap(parse_callee(t))?,
            }
        }
        "return" => FTerm::Return { target: k.val()? },
        "unresolved" => FTerm::Unresolved { target: k.val()? },
        "syscall" => {
            // An immediate-numbered system call prints `syscall #<imm> ...`
            // (see the printer above); the bare form is a register-numbered
            // one and reads 0 here.
            let imm = match k.peek() {
                Some(t) if t.starts_with('#') => {
                    let t = k.next()?;
                    k.wrap(parse_num(&t[1..]))? as u32
                }
                _ => 0,
            };
            let next_pc = k.num()?;
            k.expect("->")?;
            FTerm::Syscall {
                next_pc,
                next: opt_edge(k)?,
                imm,
            }
        }
        "illegal" => FTerm::Illegal {
            raw: k.num()? as u32,
            pc: k.num()?,
        },
        other => return Err(k.err(format!("unknown terminator `{other}`"))),
    })
}

fn parse_hex_list(t: &str) -> Result<Vec<u64>, String> {
    let inner = t
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .ok_or_else(|| format!("expected [..], found `{t}`"))?;
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|x| {
            u64::from_str_radix(x.strip_prefix("0x").unwrap_or(x), 16)
                .map_err(|_| format!("bad address `{x}`"))
        })
        .collect()
}

/// Parse the output of [`print`].
pub fn parse(src: &str) -> Result<Function, String> {
    let mut lines = src.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
    let (ln, header) = lines.next().ok_or("empty input")?;
    // The name is a Rust-quoted string and may contain spaces: take it out
    // before splitting the rest.
    let (name, rest) = {
        let after = header
            .strip_prefix("func ")
            .ok_or_else(|| format!("line {}: expected `func`", ln + 1))?;
        let (entry_tok, tail) = after
            .split_once(' ')
            .ok_or_else(|| format!("line {}: truncated header", ln + 1))?;
        let (name, tail) = if let Some(q) = tail.strip_prefix('"') {
            let (lit, tail) = split_quoted(q).map_err(|e| format!("line {}: {e}", ln + 1))?;
            (Some(lit), tail)
        } else {
            let (dash, tail) = tail
                .split_once(' ')
                .ok_or_else(|| format!("line {}: truncated header", ln + 1))?;
            if dash != "-" {
                return Err(format!("line {}: expected a quoted name or `-`", ln + 1));
            }
            (None, tail.to_string())
        };
        (name, format!("{entry_tok} {tail}"))
    };
    let mut k = Toks::new(&rest, ln);
    let entry = k.num()?;
    k.expect("entries")?;
    let t = k.next()?;
    let entries = k.wrap(parse_hex_list(t))?;
    k.expect("form")?;
    let form = match k.next()? {
        "memory" => Form::Memory,
        "ssa" => Form::Ssa,
        t => return Err(k.err(format!("unknown form `{t}`"))),
    };
    k.expect("values")?;
    let n_values = k.u32()?;
    k.done()?;

    let mut blocks: Vec<FBlock> = Vec::new();
    let mut cur: Option<FBlock> = None;
    for (ln, line) in lines {
        if let Some(body) = line.strip_prefix("  @") {
            let b = cur
                .as_mut()
                .ok_or_else(|| format!("line {}: statement outside a block", ln + 1))?;
            let (pc, rest) = body
                .split_once(' ')
                .ok_or_else(|| format!("line {}: missing statement", ln + 1))?;
            let pc = u64::from_str_radix(pc, 16).map_err(|_| format!("line {}: bad pc", ln + 1))?;
            let mut k = Toks::new(rest, ln);
            // A terminator is the last line of a block; anything that parses
            // as one is taken as one (statement and terminator keywords are
            // disjoint).
            let is_term = matches!(
                k.peek(),
                Some(
                    "jump" | "br" | "switch" | "call" | "tailcall" | "return" | "unresolved"
                        | "syscall" | "illegal"
                )
            );
            if is_term {
                b.term = parse_term(&mut k)?;
                b.term_pc = pc;
            } else {
                b.stmts.push(parse_stmt(&mut k)?);
                b.pcs.push(pc);
            }
            k.done()?;
            continue;
        }
        // A block header: `bN start..end [synthetic] [(params)] insts [..]:`
        if let Some(b) = cur.take() {
            blocks.push(b);
        }
        let line = line
            .strip_suffix(':')
            .ok_or_else(|| format!("line {}: expected a block header", ln + 1))?;
        let (params, rest) = match (line.find('('), line.find(')')) {
            (Some(o), Some(c)) if o < c => {
                let mut ps = Vec::new();
                for p in line[o + 1..c].split(", ") {
                    let (r, x) = p
                        .split_once(' ')
                        .ok_or_else(|| format!("line {}: bad parameter `{p}`", ln + 1))?;
                    ps.push(Param {
                        reg: parse_reg(r).map_err(|e| format!("line {}: {e}", ln + 1))?,
                        val: parse_val(x).map_err(|e| format!("line {}: {e}", ln + 1))?,
                    });
                }
                (ps, format!("{} {}", &line[..o], &line[c + 1..]))
            }
            _ => (Vec::new(), line.to_string()),
        };
        let mut k = Toks::new(&rest, ln);
        let id = k.next()?;
        if id != format!("b{}", blocks.len()) {
            return Err(k.err(format!("expected b{}, found `{id}`", blocks.len())));
        }
        let range = k.next()?;
        let (s, e) = range
            .split_once("..")
            .ok_or_else(|| k.err(format!("bad range `{range}`")))?;
        let (start, end) = (k.wrap(parse_num(s))?, k.wrap(parse_num(e))?);
        let synthetic = k.peek() == Some("synthetic");
        if synthetic {
            k.next()?;
        }
        k.expect("insts")?;
        let t = k.next()?;
        let insts = k.wrap(parse_hex_list(t))?;
        k.done()?;
        cur = Some(FBlock {
            start,
            end,
            synthetic,
            params,
            stmts: Vec::new(),
            pcs: Vec::new(),
            insts,
            term: FTerm::Illegal { raw: 0, pc: start },
            term_pc: start,
        });
    }
    if let Some(b) = cur.take() {
        blocks.push(b);
    }
    Ok(Function {
        entry,
        name,
        entries,
        blocks,
        n_values,
        form,
    })
}

/// Split a Rust-escaped string literal (after its opening quote) from what
/// follows its closing quote.
fn split_quoted(s: &str) -> Result<(String, String), String> {
    let mut out = String::new();
    let mut chars = s.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => {
                let rest = s[i + 1..].trim_start().to_string();
                return Ok((out, rest));
            }
            '\\' => match chars.next() {
                Some((_, 'n')) => out.push('\n'),
                Some((_, 't')) => out.push('\t'),
                Some((_, 'r')) => out.push('\r'),
                Some((_, '0')) => out.push('\0'),
                Some((_, '\\')) => out.push('\\'),
                Some((_, '"')) => out.push('"'),
                Some((_, '\'')) => out.push('\''),
                Some((_, 'u')) => {
                    // \u{XXXX}
                    let rest = &s[i + 2..];
                    let close = rest.find('}').ok_or("bad \\u escape")?;
                    let code = u32::from_str_radix(&rest[1..close], 16)
                        .map_err(|_| "bad \\u escape")?;
                    out.push(char::from_u32(code).ok_or("bad \\u escape")?);
                    for _ in 0..close + 1 {
                        chars.next();
                    }
                }
                _ => return Err("bad escape in name".into()),
            },
            c => out.push(c),
        }
    }
    Err("unterminated name".into())
}
