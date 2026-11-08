//! Function mode: tier-1's statement lowering for whole functions, for the
//! function JIT (`gex-fjit`).
//!
//! The JIT owns the control flow -- structured `block`/`loop`/`br` over the
//! function's blocks, calls, the `F_body` ABI. Everything inside a block
//! (addressing and the bounds guard, atomics, flags, FP, x87, SIMD) is
//! tier-1's own code, so memory semantics, flags and FP are tier-1's bit for
//! bit. What differs is the register cache: every register the function uses
//! has one local for the whole body, loaded on entry and again wherever guest
//! state may have changed, and the JIT says which registers may differ from
//! the context ([`FnBody::set_dirty`]) and which to write back where
//! ([`FnBody::flush`]), as gex-cgen's function mode does in C.

use super::*;
use gex_ir::func::RegSet;
use std::collections::HashMap;

/// A block's register writes left to its edges ([`FnBody::block`]).
pub struct Deferred {
    writes: Vec<(GuestReg, Value)>,
    sunk: Vec<Stmt>,
    /// The value tables right after the block: what the deferred code reads.
    local: Vec<Option<u32>>,
    cnst: Vec<Option<u64>>,
    ty: Vec<Ty>,
    fp: Vec<Option<bool>>,
    /// The values kept alive for it, released by [`FnBody::end_block`].
    inputs: Vec<u32>,
    borrowed: Vec<u32>,
}

/// A terminator's value, past its block (see [`FnBody::operand`]).
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Local(u32, Ty),
    /// A register's local, borrowed: freed with nothing.
    Reg(u32, Ty),
    Const(u64, Ty),
}

/// One function body under construction.
pub struct FnBody<'a> {
    e: Emitter<'a>,
    /// Per value: scratch for [`Self::block`]'s last-use pass.
    last: Vec<usize>,
    /// The block's values that live in a register's local.
    borrowed: Vec<u32>,
    /// Each statement of the next [`Self::block`]: its instruction's pc, and
    /// where it carries on after a fault delivered in place
    /// ([`Self::stmt_pcs`]).
    pcs: Vec<(u64, Option<u64>)>,
}

/// A last use past the block's statements (the terminator's).
const KEPT: usize = usize::MAX - 1;

/// The precision of the float `e` computes, when it computes one a float
/// consumer could take as is (`Emitter::fp_result`).
fn fp_out(e: &Expr) -> Option<bool> {
    match *e {
        Expr::FpBin { double, .. } | Expr::FpUn { double, .. } => Some(double),
        Expr::FpCvt { kind: FpConvert::FpToFp | FpConvert::IntToFp { .. }, dst_wide, .. } => Some(dst_wide),
        Expr::VExtract { lane_bytes: 8, ty: Ty::I64, .. } => Some(true),
        Expr::VExtract { lane_bytes: 4, ty: Ty::I32, .. } => Some(false),
        _ => None,
    }
}

/// Whether `s` reads `v` only as a float of that precision.
fn fp_in(s: &Stmt, v: Value, double: bool) -> bool {
    let Stmt::Def { expr, .. } = s else { return false };
    match *expr {
        Expr::FpBin { double: d, .. } | Expr::FpUn { double: d, .. } | Expr::FpCmp { double: d, .. } => d == double,
        Expr::FpCvt { kind: FpConvert::FpToFp | FpConvert::FpToInt { .. }, src_wide, a, .. } => src_wide == double && a == v,
        Expr::VReplace { lane_bytes, val, a, .. } => val == v && a != v && lane_bytes == if double { 8 } else { 4 },
        _ => false,
    }
}

/// The values a statement defines or uses.
fn s_defs_and_uses(s: &Stmt) -> Vec<u32> {
    let mut out = Vec::new();
    if let Stmt::Def { dst, .. } = s {
        out.push(dst.0);
    }
    s.for_each_use(|v| out.push(v.0));
    out
}

impl<'a> FnBody<'a> {
    /// A function of `n_params` parameters (`ctx` first) whose blocks use
    /// the registers `used` and number their values `0..n_values` across
    /// the function. No code yet: the caller loads the registers
    /// ([`Self::load`]) where it wants them loaded.
    pub fn new(cx: &'a Context<'a>, start: u64, n_params: u32, used: &[GuestReg], n_values: u32) -> FnBody<'a> {
        let mut e = Emitter::with_params(cx, start, n_params);
        let mut used: Vec<GuestReg> = used.to_vec();
        used.sort_by_key(|r| Emitter::reg_offset(*r));
        used.dedup();
        for &r in &used {
            let local = e.f.add_local(val_type(r.ty()));
            e.regs.insert(r, Cached { local, dirty: false });
        }
        e.func_regs = Some(used);
        e.f.fold_tees();
        let n = n_values as usize;
        e.value_local = vec![None; n];
        e.value_const = vec![None; n];
        e.value_ctx = vec![false; n];
        e.value_ty = vec![Ty::I64; n];
        e.value_fp = vec![None; n];
        FnBody { e, last: vec![usize::MAX; n], borrowed: Vec::new(), pcs: Vec::new() }
    }

    /// Faults the function goes on after: their import's `pc` carries
    /// `bit`, and the function leaves (returns 0) when the import returns.
    pub fn resumable_faults(&mut self, bit: u64) {
        self.e.resumable_pc = bit;
    }

    /// For the next [`Self::block`], by statement: the pc of its
    /// instruction, where a fault it raises is delivered in place (with
    /// [`Self::resumable_faults`]), and for a fault statement (`int3`'s or
    /// `ud2`'s `HostCall`) the pc it carries on at, its next instruction's,
    /// which makes it one the host may deliver in place.
    pub fn stmt_pcs(&mut self, pcs: Vec<(u64, Option<u64>)>) {
        self.pcs = pcs;
    }

    /// The code, for the JIT's own control flow.
    pub fn f(&mut self) -> &mut FuncBody {
        &mut self.e.f
    }

    /// The `ctx` parameter.
    pub fn ctx(&self) -> u32 {
        self.e.ctx_local
    }

    /// A fresh local of the JIT's own.
    pub fn local(&mut self, ty: ValType) -> u32 {
        self.e.f.add_local(ty)
    }

    /// The local holding `r` (which must be in the used set).
    pub fn reg(&self, r: GuestReg) -> u32 {
        self.e.regs[&r].local
    }

    pub fn is_used(&self, r: GuestReg) -> bool {
        self.e.regs.contains_key(&r)
    }

    /// Load `regs` from the context into their locals; they are clean.
    pub fn load(&mut self, regs: &[GuestReg]) {
        self.e.reload(regs);
    }

    /// Reload every used register outside `keep` (whose locals hold what
    /// the context does, e.g. a callee's preserved registers); afterwards
    /// none is dirty.
    pub fn reload_except(&mut self, keep: RegSet) {
        let used = self.e.func_regs.clone().unwrap_or_default();
        let regs: Vec<GuestReg> = used.into_iter().filter(|r| !keep.contains(*r)).collect();
        self.e.reload(&regs);
        self.set_dirty(RegSet::EMPTY);
    }

    /// The registers that may differ from the context from here on.
    pub fn set_dirty(&mut self, dirty: RegSet) {
        for (r, c) in self.e.regs.iter_mut() {
            c.dirty = dirty.contains(*r);
        }
    }

    pub fn dirty(&self) -> RegSet {
        let mut d = RegSet::EMPTY;
        for (r, c) in &self.e.regs {
            if c.dirty {
                d.insert(*r);
            }
        }
        d
    }

    /// Write back the dirty registers in `mask` (tier-1's stores, flags
    /// packed as tier-1 packs them). Unless `keep_dirty`, every register is
    /// clean afterwards: the ones outside `mask` are dead (nothing reads
    /// them before they are written again), as gex-cgen's function mode
    /// treats them.
    pub fn flush(&mut self, mask: RegSet, keep_dirty: bool) {
        let before = self.dirty();
        self.set_dirty(before.inter(mask));
        self.e.flush_inner(true);
        self.set_dirty(if keep_dirty { before } else { RegSet::EMPTY });
    }

    /// Emit one block's statements. `start` is the block's guest address
    /// (what a fault inside reports); values in `keep` stay live after the
    /// statements, for the terminator, until [`Self::release`].
    /// `defer` are indices of `WriteReg`s left to the caller, which does
    /// them on the edges that need them ([`Self::emit_deferred`]); the pure
    /// definitions only they use are left with them (sunk), so a flag
    /// computed for an exit taken once is computed there.
    pub fn block(&mut self, start: u64, stmts: &[Stmt], keep: &[Value], defer: &[usize]) -> Deferred {
        self.e.block_start = start;
        let mut u32s = vec![false; self.e.value_local.len()];
        // Which values a load or a register read defines: a pointer, likely
        // chased (`p = p->next`), whose null guard goes beside the access
        // rather than on its address (`emit_address`).
        let mut fetched = vec![false; self.e.value_local.len()];
        let mut chase = vec![false; self.e.value_local.len()];
        for st in stmts {
            if let Stmt::Def { dst, expr } = st {
                match expr {
                    Expr::Load { .. } | Expr::ReadReg(_) => {
                        if let Some(u) = fetched.get_mut(dst.0 as usize) {
                            *u = true;
                        }
                    }
                    Expr::Extend { signed: false, a } => {
                        if let Some(u) = u32s.get_mut(dst.0 as usize) {
                            *u = true;
                        }
                        if fetched.get(a.0 as usize).copied().unwrap_or(false) {
                            if let Some(u) = chase.get_mut(dst.0 as usize) {
                                *u = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        self.e.value_u32 = u32s;
        self.e.value_chase = chase;
        for c in self.e.value_ctx.iter_mut() {
            *c = false;
        }
        // Values are numbered per block: forget the last block's (a value
        // nobody read keeps its local, which is only a local lost).
        for v in self.e.value_local.iter_mut() {
            *v = None;
        }
        for v in self.e.value_const.iter_mut() {
            *v = None;
        }
        for v in self.e.value_fp.iter_mut() {
            *v = None;
        }
        // Who uses each value, and which definitions sink: pure, not a
        // register read (its register may change before the edge), used by
        // deferred writes and sunk definitions only.
        let mut users: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, s) in stmts.iter().enumerate() {
            s.for_each_use(|v| users.entry(v.0).or_default().push(i));
        }
        let mut sunk = vec![false; stmts.len()];
        for (i, s) in stmts.iter().enumerate().rev() {
            if let Stmt::Def { dst, expr } = s {
                let sinks = expr.is_pure()
                    && !matches!(expr, Expr::ReadReg(_))
                    && !keep.contains(dst)
                    && users.get(&dst.0).is_some_and(|u| u.iter().all(|&j| defer.contains(&j) || sunk[j]));
                sunk[i] = sinks;
            }
        }
        // The values the deferred code reads from the block: kept.
        let mut inputs: Vec<Value> = Vec::new();
        for (i, s) in stmts.iter().enumerate() {
            if sunk[i] || defer.contains(&i) {
                s.for_each_use(|v| {
                    let from_sunk = stmts.iter().enumerate().any(|(j, d)| sunk[j] && matches!(d, Stmt::Def { dst, .. } if *dst == v));
                    if !from_sunk && !inputs.contains(&v) {
                        inputs.push(v);
                    }
                });
            }
        }
        // Last uses, over the values this block touches only.
        let mut touched: Vec<u32> = Vec::new();
        for (i, s) in stmts.iter().enumerate() {
            if sunk[i] || defer.contains(&i) {
                continue;
            }
            let last = &mut self.last;
            let mut note = |v: Value| {
                let k = v.0 as usize;
                if last[k] == usize::MAX {
                    touched.push(v.0);
                }
                last[k] = i;
            };
            if let Stmt::Def { dst, .. } = s {
                note(*dst);
            }
            s.for_each_use(&mut note);
        }
        for k in keep.iter().chain(&inputs) {
            if (k.0 as usize) < self.last.len() {
                if self.last[k.0 as usize] == usize::MAX {
                    touched.push(k.0);
                }
                self.last[k.0 as usize] = KEPT;
            }
        }
        let mut last_use: HashMap<u32, usize> = touched.iter().map(|&v| (v, self.last[v as usize])).collect();
        for &v in &touched {
            self.last[v as usize] = usize::MAX;
        }
        // Whether register `r`'s local changes in stmts[from..=to]: a write,
        // or a clobbering host call (which reloads every register).
        let changes = |r: GuestReg, from: usize, to: usize| {
            stmts[from..stmts.len().min(to.saturating_add(1))]
                .iter()
                .any(|s| matches!(s, Stmt::WriteReg { reg, .. } if *reg == r) || clobbers_guest_state(s))
        };
        // Values living in a register's local rather than their own: a
        // register read whose register does not change before the value's
        // last use, and a definition written to a register at once.
        let mut borrowed: Vec<u32> = Vec::new();
        let dummy = Block::new(start);
        let mut skip_write = usize::MAX;
        let pcs = std::mem::take(&mut self.pcs);
        for (i, s) in stmts.iter().enumerate() {
            if sunk[i] || defer.contains(&i) {
                continue;
            }
            self.e.stmt_pc = pcs.get(i).map(|p| p.0);
            self.e.fault_next = pcs.get(i).and_then(|p| p.1);
            if i == skip_write {
                // The definition before left its value in the register.
                if let Stmt::WriteReg { reg, .. } = s {
                    self.e.regs.get_mut(reg).expect("a used register").dirty = true;
                }
            } else {
                match s {
                    Stmt::Def { dst, expr: Expr::ReadReg(r) }
                        if self.e.regs.contains_key(r) && *r != GuestReg::CtxAddr && !changes(*r, i + 1, last_use[&dst.0]) =>
                    {
                        let k = dst.0 as usize;
                        self.e.value_ty[k] = r.ty();
                        self.e.value_fp[k] = None;
                        self.e.value_local[k] = Some(self.e.regs[r].local);
                        borrowed.push(dst.0);
                    }
                    Stmt::Def { dst, expr } => {
                        // A float only floats read: kept as one.
                        self.e.fp_result = fp_out(expr).is_some_and(|d| {
                            !keep.contains(dst)
                                && users.get(&dst.0).is_some_and(|u| {
                                    !u.is_empty() && u.iter().all(|&j| !sunk[j] && !defer.contains(&j) && fp_in(&stmts[j], *dst, d))
                                })
                        });
                        let into = match stmts.get(i + 1) {
                            Some(Stmt::WriteReg { reg, val })
                                if val == dst && last_use[&dst.0] == i + 1 && expr.ty() == reg.ty() && !defer.contains(&(i + 1)) =>
                            {
                                self.e.regs.get(reg).map(|c| c.local)
                            }
                            _ => None,
                        };
                        self.e.def_into = if self.e.fp_result { None } else { into };
                        let into = self.e.def_into;
                        self.e.stmt(s, &dummy);
                        self.e.fp_result = false;
                        if into.is_some() && self.e.def_into.take().is_none() {
                            borrowed.push(dst.0);
                            skip_write = i + 1;
                        }
                    }
                    _ => self.e.stmt(s, &dummy),
                }
            }
            for v in s_defs_and_uses(s) {
                if last_use.get(&v) == Some(&i) {
                    last_use.remove(&v);
                    if borrowed.contains(&v) {
                        self.e.value_local[v as usize] = None;
                    } else {
                        self.e.release(Value(v));
                    }
                }
            }
        }
        self.e.fault_next = None;
        self.e.stmt_pc = None;
        self.borrowed = borrowed.clone();
        Deferred {
            writes: defer
                .iter()
                .filter_map(|&i| match stmts[i] {
                    Stmt::WriteReg { reg, val } => Some((reg, val)),
                    _ => None,
                })
                .collect(),
            sunk: stmts.iter().enumerate().filter(|(i, _)| sunk[*i]).map(|(_, s)| s.clone()).collect(),
            local: self.e.value_local.clone(),
            cnst: self.e.value_const.clone(),
            ty: self.e.value_ty.clone(),
            fp: self.e.value_fp.clone(),
            inputs: inputs.iter().map(|v| v.0).filter(|v| !keep.iter().any(|k| k.0 == *v)).collect(),
            borrowed,
        }
    }

    /// Do `d`'s writes of the registers `want` admits (and the sunk
    /// definitions they need): on an edge of the block `d` came from.
    pub fn emit_deferred(&mut self, d: &Deferred, want: &dyn Fn(GuestReg) -> bool) {
        let sel: Vec<(GuestReg, Value)> = d.writes.iter().copied().filter(|(r, _)| want(*r)).collect();
        if sel.is_empty() {
            return;
        }
        // Sunk code is pure: no address to guard; nothing claimed for it.
        self.e.value_u32.clear();
        self.e.value_chase.clear();
        self.e.value_local.clone_from(&d.local);
        self.e.value_const.clone_from(&d.cnst);
        self.e.value_ty.clone_from(&d.ty);
        self.e.value_fp.clone_from(&d.fp);
        let mut need: Vec<u32> = sel.iter().map(|(_, v)| v.0).collect();
        let mut emit = vec![false; d.sunk.len()];
        for (k, s) in d.sunk.iter().enumerate().rev() {
            if let Stmt::Def { dst, .. } = s {
                if need.contains(&dst.0) {
                    emit[k] = true;
                    s.for_each_use(|v| need.push(v.0));
                }
            }
        }
        let dummy = Block::new(self.e.block_start);
        let mut made = Vec::new();
        for (k, s) in d.sunk.iter().enumerate() {
            if emit[k] {
                self.e.stmt(s, &dummy);
                if let Stmt::Def { dst, .. } = s {
                    made.push(*dst);
                }
            }
        }
        for (r, v) in sel {
            self.e.push(v);
            let c = self.e.regs.get_mut(&r).expect("a used register");
            c.dirty = true;
            let l = c.local;
            self.e.f.local_set(l);
        }
        for v in made {
            self.e.release(v);
        }
    }

    /// The registers `d` writes.
    pub fn deferred_regs(d: &Deferred) -> impl Iterator<Item = GuestReg> + '_ {
        d.writes.iter().map(|(r, _)| *r)
    }

    /// Free what `d` kept alive, once the block's terminator is done.
    pub fn end_block(&mut self, d: Deferred) {
        for v in d.inputs {
            if d.borrowed.contains(&v) {
                continue;
            }
            if let Some(l) = d.local[v as usize] {
                match (d.fp[v as usize], d.ty[v as usize]) {
                    (Some(true), _) => self.e.free_f64.push(l),
                    (Some(false), _) => self.e.free_f32.push(l),
                    (None, Ty::I32) => self.e.free_i32.push(l),
                    (None, Ty::I64) => self.e.free_i64.push(l),
                    (None, Ty::V128) => self.e.free_v128.push(l),
                }
            }
        }
    }

    /// A value kept past [`Self::block`], as something that outlives the
    /// value numbering (the next block reuses the numbers): its local, or
    /// its constant. Free it with [`Self::release_operand`].
    pub fn operand(&mut self, v: Value) -> Operand {
        let k = v.0 as usize;
        let ty = self.e.value_ty[k];
        match self.e.value_local[k].take() {
            Some(l) if self.borrowed.contains(&v.0) => Operand::Reg(l, ty),
            Some(l) => Operand::Local(l, ty),
            None => Operand::Const(self.e.value_const[k].expect("a kept value has a local or a constant"), ty),
        }
    }

    pub fn push_operand(&mut self, o: Operand) {
        match o {
            Operand::Local(l, _) | Operand::Reg(l, _) => self.e.f.local_get(l),
            Operand::Const(c, Ty::I32) => self.e.f.i32_const(c as i32),
            Operand::Const(c, _) => self.e.f.i64_const(c as i64),
        }
    }

    pub fn release_operand(&mut self, o: Operand) {
        if let Operand::Local(l, ty) = o {
            match ty {
                Ty::I32 => self.e.free_i32.push(l),
                Ty::I64 => self.e.free_i64.push(l),
                Ty::V128 => self.e.free_v128.push(l),
            }
        }
    }


    pub fn store_pc(&mut self, pc: u64) {
        self.e.store_pc(pc);
    }

    /// [`Context::store_syscall_imm`](super::Context::store_syscall_imm).
    pub fn store_syscall_imm(&mut self, imm: u32) {
        self.e.store_syscall_imm(imm);
    }

    /// tier-1's safepoint countdown (the context's counter), polling with
    /// `pc`. Writes nothing back: the caller flushes first if it must.
    pub fn safepoint(&mut self, pc: u64) {
        self.e.emit_safepoint(pc);
    }

    /// The same, with the pc in the i64 local `pc`.
    pub fn safepoint_dynamic(&mut self, pc: u32) {
        let ctx = self.e.ctx_local;
        let sc = self.e.scratch_i32;
        let f = &mut self.e.f;
        f.local_get(ctx);
        f.addr_i32();
        f.local_get(ctx);
        f.addr_i32();
        f.load(0x28, 2, ctx::SAFEPOINT_COUNTER as u32);
        f.i32_const(1);
        f.byte(0x6b); // i32.sub
        f.local_tee(sc);
        f.store(0x36, 2, ctx::SAFEPOINT_COUNTER as u32);
        f.local_get(sc);
        f.i32_const(0);
        f.byte(0x4c); // i32.le_s
        f.if_void();
        f.local_get(ctx);
        f.local_get(pc);
        f.call(self.e.cx.safepoint);
        f.end();
    }

    pub fn finish(mut self) -> Vec<u8> {
        if std::env::var("GEX_FJIT_TEES").as_deref() != Ok("0") {
            let temps: std::collections::HashSet<u32> = self.e.temps.iter().copied().collect();
            self.e.f.strip_dead_tees(&|l| temps.contains(&l), false);
        }
        self.e.f.finish()
    }
}

/// A register's context offset, as tier-1 stores it.
pub fn reg_offset(r: GuestReg) -> u32 {
    Emitter::reg_offset(r)
}

impl Emitter<'_> {
    /// Function mode: load `regs` into their locals, marking them clean.
    pub(super) fn reload(&mut self, regs: &[GuestReg]) {
        for &r in regs {
            let Some(c) = self.regs.get_mut(&r) else { continue };
            c.dirty = false;
            let local = c.local;
            self.f.local_get(self.ctx_local);
            if r == GuestReg::CtxAddr {
                self.f.byte(code::I64_EXTEND_I32_U);
            } else if Self::is_flag(r) {
                self.f.addr_i32();
                self.f.load(0x2d, 0, Self::reg_offset(r)); // i32.load8_u
            } else {
                self.f.addr_i32();
                match r.ty() {
                    Ty::I32 => self.f.load(0x28, 2, Self::reg_offset(r)),
                    Ty::I64 => self.f.load(0x29, 3, Self::reg_offset(r)),
                    Ty::V128 => self.f.simd_mem(simd::V128_LOAD, 4, Self::reg_offset(r)),
                }
            }
            self.f.local_set(local);
        }
    }
}
