//! A reference interpreter for function-level GIR, for testing.
//!
//! Runs one [`Function`] from its entry on a concrete guest state until
//! control leaves it, in either form, so that a transformation (SSA
//! construction, later the function tier's passes) can be checked by
//! running the function before and after on the same inputs and comparing
//! what leaves it: the exit, every register, and every memory write.
//!
//! Only scalar integer semantics are implemented. A function that reaches
//! anything else (vector or floating-point operations, bulk memory
//! statements, host calls) stops with [`Stop::Unsupported`], which callers
//! treat as "no verdict" rather than a failure.
//!
//! Calls are modelled, not followed: the callee changes every register its
//! [`CallEffect`] does not preserve to a value derived from the call site,
//! the register and a per-run counter, and moves the stack pointer by the
//! effect's `sp_delta` (or clobbers it). Both runs see the same model, so a
//! form that relies on a preserved register surviving a call is checked
//! against exactly that assumption. Syscalls are modelled as calls that do
//! the model's `syscall` says (by default, preserve nothing). A modelled
//! callee reads and writes no memory.
//!
//! Memory nobody wrote reads as a hash of its address and the seed, so it is
//! the same in every run with that seed whatever the registers hold. Only
//! lifted blocks count against the step budget, so the synthetic blocks SSA
//! construction adds do not make the two forms stop at different points.

use crate::func::*;
use crate::ssa::CallEffect;
use crate::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Why a run stopped.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Stop {
    /// Returned, to this target.
    Return(u64),
    /// Left by a tail call or an unresolved jump, to this address.
    Leave(u64),
    /// Reached an illegal instruction.
    Illegal(u64),
    /// A [`Stmt::Trap`] was taken: the instruction at this pc faulted with
    /// this code.
    Fault(u64, i32),
    /// Ran for the step budget without leaving.
    Budget,
    /// Hit something the interpreter does not model, at this guest pc.
    Unsupported(u64),
}

/// The observable result of a run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub stop: Stop,
    /// Every register's final value (scalars zero-extended), by
    /// [`reg_index`].
    pub regs: Vec<u64>,
    /// Final contents of every byte written.
    pub memory: BTreeMap<u64, u8>,
    /// Bytes loaded before this run wrote them: what the run read of the
    /// memory it started with.
    pub read_before_write: BTreeSet<u64>,
}

/// How a run models what lies outside the function.
pub struct Model<'a> {
    /// What each call does to the registers.
    pub call: &'a dyn Fn(Callee) -> CallEffect,
    /// The guest's stack-pointer register (`sp`, or `X(4)` on x86-64): the
    /// one a call's `sp_delta` applies to, started at a plausible stack
    /// address.
    pub sp: GuestReg,
    /// Lifted blocks executed before stopping with [`Stop::Budget`].
    pub max_steps: usize,
    /// What a syscall does to the registers, given the registers at the
    /// syscall; `None` is [`CallEffect::UNKNOWN`].
    pub syscall: Option<&'a dyn Fn(&[u64]) -> CallEffect>,
    /// i386 code: every general register holds a zero-extended 32-bit
    /// value, including what a modelled call leaves in one.
    pub narrow: bool,
}

/// Deterministic pseudo-random bits from a key.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// Keep flags booleans and narrow registers in range, as the lifters
/// maintain them.
fn in_range(r: GuestReg, v: u64) -> u64 {
    match r.ty() {
        Ty::I32 if r.is_flag() => v & 1,
        Ty::I32 => v & 0xffff_ffff,
        _ => v,
    }
}

/// [`in_range`], and for i386 code (`narrow`) general registers are
/// zero-extended 32-bit values.
fn in_range_for(r: GuestReg, v: u64, narrow: bool) -> u64 {
    match r {
        GuestReg::X(_) if narrow => v & 0xffff_ffff,
        _ => in_range(r, v),
    }
}

/// [`initial_regs`] for i386 code: 32-bit general registers and a stack
/// pointer below 3 GiB.
pub fn initial_regs32(seed: u64, sp: GuestReg) -> Vec<u64> {
    let mut regs: Vec<u64> = initial_regs(seed, sp)
        .into_iter()
        .enumerate()
        .map(|(i, v)| in_range_for(reg_of_index(i), v, true))
        .collect();
    regs[reg_index(sp)] = 0xbf00_0000 + (mix(seed) & 0xfff0);
    regs
}

/// [`perturb`] for i386 code when `narrow`.
pub fn perturb_for(r: GuestReg, v: u64, narrow: bool) -> u64 {
    in_range_for(r, perturb(r, v), narrow)
}

/// The registers a run with `seed` starts with: pseudo-random, with the
/// stack pointer at a plausible, 16-byte aligned stack address.
pub fn initial_regs(seed: u64, sp: GuestReg) -> Vec<u64> {
    let mut regs: Vec<u64> = (0..N_REGS)
        .map(|i| in_range(reg_of_index(i), mix(seed ^ (i as u64 + 1) * 0x9e37_79b9)))
        .collect();
    regs[reg_index(sp)] = 0x7fff_0000_0000 + (mix(seed) & 0xfff0);
    regs
}

/// A different value for register `r`, still in range.
pub fn perturb(r: GuestReg, v: u64) -> u64 {
    in_range(r, if r.is_flag() { v ^ 1 } else { !v ^ 0x5a5a })
}

/// Run `f` from its entry with [`initial_regs`].
pub fn run(f: &Function, seed: u64, m: &Model) -> Outcome {
    run_from(f, seed, initial_regs(seed, m.sp), m)
}

/// Run `f` from its entry with the given registers; `seed` fixes the
/// contents of memory and what modelled calls do.
pub fn run_from(f: &Function, seed: u64, regs: Vec<u64>, m: &Model) -> Outcome {
    run_with_memory(f, seed, regs, m, &HashMap::new())
}

/// [`run_from`] on a memory that starts with `init` (anything else reads as
/// the seed's hash). Bytes read from `init` are not
/// [`Outcome::read_before_write`]s.
pub fn run_with_memory(f: &Function, seed: u64, mut regs: Vec<u64>, m: &Model, init: &HashMap<u64, u8>) -> Outcome {
    assert_eq!(regs.len(), N_REGS);
    let spi = reg_index(m.sp);
    let mut mem: HashMap<u64, u8> = init.clone();
    // Hash containers while running; ordered ones in the outcome.
    let mut written: HashMap<u64, u8> = HashMap::new();
    let mut rbw: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut read_byte = |mem: &HashMap<u64, u8>, a: u64| -> u8 {
        match mem.get(&a) {
            Some(&b) => b,
            None => {
                rbw.insert(a);
                (mix(seed ^ a.wrapping_mul(31)) & 0xff) as u8
            }
        }
    };
    let mut vals = Vals(vec![None; f.n_values as usize]);
    let mut calls = 0u64;
    let mut block = BlockId(0);
    // The entry block's parameters hold the context's values.
    let mut args: Vec<u64> = f.blocks[0].params.iter().map(|p| regs[reg_index(p.reg)]).collect();
    let mut steps = 0usize;

    macro_rules! finish {
        ($stop:expr) => {{
            drop(read_byte);
            return Outcome {
                stop: $stop,
                regs,
                memory: written.into_iter().collect(),
                read_before_write: rbw.into_iter().collect(),
            };
        }};
    }

    loop {
        let b = f.block(block);
        if !b.synthetic {
            steps += 1;
            if steps > m.max_steps {
                finish!(Stop::Budget);
            }
        }
        for (p, a) in b.params.iter().zip(&args) {
            vals.insert(p.val, *a);
        }
        let get = |vals: &Vals, v: Value| -> Option<u64> { vals.get(&v).copied() };
        for (s, &pc) in b.stmts.iter().zip(&b.pcs) {
            let unsupported = || Stop::Unsupported(pc);
            match s {
                Stmt::Def { dst, expr } => {
                    let v = match expr {
                        Expr::ReadReg(r) => regs[reg_index(*r)],
                        Expr::Load {
                            addr,
                            offset,
                            sz,
                            signed,
                            ty,
                            ..
                        } => {
                            let Some(a) = get(&vals, *addr) else {
                                finish!(unsupported());
                            };
                            let a = a.wrapping_add(*offset as u64);
                            let n = sz.bytes();
                            let mut x = 0u64;
                            for i in 0..n {
                                x |= (read_byte(&mem, a.wrapping_add(i as u64)) as u64) << (8 * i);
                            }
                            if *signed && n < 8 {
                                let sh = 64 - 8 * n;
                                x = (((x << sh) as i64) >> sh) as u64;
                            }
                            x & ty.mask()
                        }
                        // One thread: read, and write the replacement when the
                        // value read equals the expected one, both taken to the
                        // access size (wasm's `atomic.rmw.cmpxchg`).
                        Expr::CmpXchg {
                            addr,
                            expected,
                            replacement,
                            sz,
                            ty,
                        } => {
                            let (Some(a), Some(e), Some(r)) =
                                (get(&vals, *addr), get(&vals, *expected), get(&vals, *replacement))
                            else {
                                finish!(unsupported());
                            };
                            let n = sz.bytes();
                            let mask = if n == 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 };
                            let mut x = 0u64;
                            for i in 0..n {
                                x |= (read_byte(&mem, a.wrapping_add(i as u64)) as u64) << (8 * i);
                            }
                            if x == e & mask {
                                for i in 0..n {
                                    let byte = (r >> (8 * i)) as u8;
                                    mem.insert(a.wrapping_add(i as u64), byte);
                                    written.insert(a.wrapping_add(i as u64), byte);
                                }
                            }
                            x & ty.mask()
                        }
                        e if e.ty() != Ty::V128
                            && !matches!(
                                e,
                                Expr::FpBin { .. }
                                    | Expr::FpUn { .. }
                                    | Expr::FpCmp { .. }
                                    | Expr::FpCvt { .. }
                                    | Expr::Mx { .. }
                                    | Expr::HostCall { .. }
                                    | Expr::Rmw { .. }
                                    | Expr::CmpXchg { .. }
                                    | Expr::VExtract { .. }
                                    | Expr::VBitmask { .. }
                            ) =>
                        {
                            // Operands renumbered 0.. so the folder's
                            // table stays small.
                            let mut local = Stmt::Def {
                                dst: Value(u32::MAX),
                                expr: e.clone(),
                            };
                            let mut known: Vec<Option<u64>> = Vec::new();
                            let mut ok = true;
                            crate::func::for_each_use_mut(&mut local, |u| {
                                let x = get(&vals, *u);
                                ok &= x.is_some();
                                known.push(x);
                                *u = Value(known.len() as u32 - 1);
                            });
                            let Stmt::Def { expr: le, .. } = &local else { unreachable!() };
                            let r = if ok { pass::eval_expr(le, &known).or_else(|| eval_div(le, &known)) } else { None };
                            match r {
                                Some(x) => x,
                                None => finish!(unsupported()),
                            }
                        }
                        _ => finish!(unsupported()),
                    };
                    vals.insert(*dst, v);
                }
                Stmt::WriteReg { reg, val } => {
                    let Some(v) = get(&vals, *val) else {
                        finish!(unsupported());
                    };
                    regs[reg_index(*reg)] = v;
                }
                Stmt::Store {
                    addr,
                    offset,
                    val,
                    sz,
                    ..
                } => {
                    let (Some(a), Some(v)) = (get(&vals, *addr), get(&vals, *val)) else {
                        finish!(unsupported());
                    };
                    let a = a.wrapping_add(*offset as u64);
                    for i in 0..sz.bytes() {
                        let byte = (v >> (8 * i)) as u8;
                        mem.insert(a.wrapping_add(i as u64), byte);
                        written.insert(a.wrapping_add(i as u64), byte);
                    }
                }
                Stmt::Fence => {}
                Stmt::X87 { op, args, outs } => {
                    let a: Option<Vec<u64>> = args.iter().map(|v| get(&vals, *v)).collect();
                    let Some(a) = a else {
                        finish!(unsupported());
                    };
                    let f = |i: usize| gex_f80::F80 {
                        sig: a[i],
                        se: a[i + 1] as u16,
                    };
                    let r: Vec<u64> = match *op {
                        X87Op::Narrow { single } => {
                            let (r, sw) = gex_f80::narrow_op(a[0], a[1] as u16, a[2] as u16, single);
                            vec![r, sw as u64]
                        }
                        X87Op::ToInt { bits, trunc } => {
                            let (r, sw) = gex_f80::to_int_op(a[0], a[1] as u16, a[2] as u16, bits, trunc);
                            vec![r, sw as u64]
                        }
                        X87Op::Arith(k) => {
                            let (r, sw) = gex_f80::arith_op(k as u8, f(0), f(2), a[4] as u16);
                            vec![r.sig, r.se as u64, sw as u64]
                        }
                        X87Op::Compare { quiet } => {
                            let (cc, sw) = gex_f80::compare_op(f(0), f(2), quiet);
                            vec![cc as u64, sw as u64]
                        }
                    };
                    for (o, v) in outs.iter().zip(r) {
                        vals.insert(*o, v);
                    }
                }
                Stmt::Trap { cond, pc, code } => match get(&vals, *cond) {
                    Some(0) => {}
                    Some(_) => finish!(Stop::Fault(*pc, *code)),
                    None => finish!(unsupported()),
                },
                _ => finish!(unsupported()),
            }
        }
        let tv = |v: Value| vals.get(&v).copied();
        let pc = b.term_pc;
        let edge_args =
            |e: &Edge, vals: &Vals| -> Option<Vec<u64>> {
                e.args.iter().map(|a| vals.get(a).copied()).collect()
            };
        let (next, next_args) = match &b.term {
            FTerm::Jump(e) => (e.to, edge_args(e, &vals)),
            FTerm::Branch {
                cond,
                taken,
                not_taken,
            } => {
                let Some(c) = tv(*cond) else {
                    finish!(Stop::Unsupported(pc));
                };
                let e = if c != 0 { taken } else { not_taken };
                (e.to, edge_args(e, &vals))
            }
            FTerm::Switch { target, cases, .. } => {
                let Some(t) = tv(*target) else {
                    finish!(Stop::Unsupported(pc));
                };
                match cases.iter().find(|(a, _)| *a == t) {
                    Some((_, e)) => (e.to, edge_args(e, &vals)),
                    None => finish!(Stop::Leave(t)),
                }
            }
            FTerm::Call { callee, ret, .. } => {
                calls += 1;
                havoc(&mut regs, (m.call)(*callee), seed ^ callee_key(*callee, &vals), calls, spi, m.narrow);
                match ret {
                    Some(e) => (e.to, edge_args(e, &vals)),
                    None => finish!(Stop::Leave(pc)),
                }
            }
            FTerm::Syscall { next, .. } => {
                calls += 1;
                let eff = m.syscall.map_or(CallEffect::UNKNOWN, |s| s(&regs));
                havoc(&mut regs, eff, seed ^ 0x5ca11, calls, spi, m.narrow);
                match next {
                    Some(e) => (e.to, edge_args(e, &vals)),
                    None => finish!(Stop::Leave(pc)),
                }
            }
            FTerm::TailCall { callee } => {
                let t = match callee {
                    Callee::Direct(a) => *a,
                    Callee::Indirect(v) => tv(*v).unwrap_or(0),
                };
                finish!(Stop::Leave(t));
            }
            FTerm::Return { target } => finish!(Stop::Return(tv(*target).unwrap_or(0))),
            FTerm::Unresolved { target } => finish!(Stop::Leave(tv(*target).unwrap_or(0))),
            FTerm::Illegal { pc, .. } => finish!(Stop::Illegal(*pc)),
        };
        let Some(a) = next_args else {
            finish!(Stop::Unsupported(pc));
        };
        block = next;
        args = a;
    }
}

/// Every value's number when known, indexed densely by [`Value`].
struct Vals(Vec<Option<u64>>);

impl Vals {
    fn get(&self, v: &Value) -> Option<&u64> {
        self.0.get(v.0 as usize).and_then(|x| x.as_ref())
    }
    fn insert(&mut self, v: Value, x: u64) {
        let i = v.0 as usize;
        if i >= self.0.len() {
            self.0.resize(i + 1, None);
        }
        self.0[i] = Some(x);
    }
}

/// What a modelled call does to the registers.
fn havoc(regs: &mut [u64], eff: CallEffect, site: u64, calls: u64, spi: usize, narrow: bool) {
    for i in 0..N_REGS {
        let r = reg_of_index(i);
        if i == spi || eff.preserved.contains(r) {
            continue;
        }
        regs[i] = in_range_for(r, mix(site ^ (calls << 20) ^ ((i as u64) << 8)), narrow);
    }
    let sp = reg_of_index(spi);
    regs[spi] = match eff.sp_delta {
        Some(d) => regs[spi].wrapping_add(d as u64),
        None if eff.preserved.contains(sp) => regs[spi],
        None => in_range_for(sp, mix(site ^ calls ^ 0x5b), narrow),
    };
}

fn callee_key(c: Callee, vals: &Vals) -> u64 {
    match c {
        Callee::Direct(a) => a,
        Callee::Indirect(v) => vals.get(&v).copied().unwrap_or(0),
    }
}

/// Division and remainder, which the constant folder leaves alone.
/// AArch64 and the lifters' guards give division by zero a defined result
/// before it reaches here; an unguarded one is unsupported.
fn eval_div(e: &Expr, known: &[Option<u64>]) -> Option<u64> {
    let Expr::Bin { op, ty, a, b } = *e else {
        return None;
    };
    let x = (*known.get(a.0 as usize)?)? & ty.mask();
    let y = (*known.get(b.0 as usize)?)? & ty.mask();
    if y == 0 {
        return None;
    }
    let (sx, sy) = match ty {
        Ty::I32 => (x as u32 as i32 as i64, y as u32 as i32 as i64),
        _ => (x as i64, y as i64),
    };
    let r = match op {
        BinOp::DivU => x / y,
        BinOp::RemU => x % y,
        BinOp::DivS => sx.checked_div(sy)? as u64,
        BinOp::RemS => sx.checked_rem(sy)? as u64,
        _ => return None,
    };
    Some(r & ty.mask())
}
