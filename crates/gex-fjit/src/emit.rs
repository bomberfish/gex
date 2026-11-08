//! One recovered guest function → one wasm function with the `F_body` ABI,
//! plus a function per outermost loop.
//!
//! The decisions are gex-cgen's (`func.rs`), made by the same analyses
//! (liveness, dirty-in, trap-free, outer loops); this emits wasm where that
//! emits C. Control flow is structured (`cfg`): a branch is a `br`, and
//! guest registers stay in locals across the whole body.

use crate::cfg::{self, Graph, Shape};
use gex_abi::ctx;
use gex_cgen::func::{self as cf, Loop};
use gex_emit::tier1::fmode::{Deferred, FnBody, Operand};
use gex_emit::tier1::Context;
use gex_emit::wasm::ValType;
use gex_ir::func::{Callee, FTerm, Function, RegSet, Summary};
use gex_ir::{Expr, GuestReg, HostFunc, Stmt, Value};
use std::collections::HashMap;


/// `need`: the caller's continuation reads a flag.
pub const NEED_FLAGS: i32 = 1;
pub const NEED_ALL: i32 = -1;
/// A loop function's results beyond the body's 0 and 1.
pub const LOOP_CHUNK: i32 = 2;
pub const LOOP_EXIT: i32 = 3;

/// What a function's code refers to outside itself.
pub struct ModuleCx<'a> {
    /// tier-1's view of the module's imports.
    pub cx: &'a Context<'a>,
    /// The body of every function in this module, by entry.
    pub bodies: &'a HashMap<u64, u32>,
    /// Loop function `k` of the function at an entry.
    pub loops: &'a HashMap<(u64, usize), u32>,
    pub sums: &'a HashMap<u64, Summary>,
    /// The trap-free functions (`gex_cgen::func::trap_free`).
    pub trapfree: &'a HashMap<u64, u32>,
    /// How far a `ret` moves SP (`gex_cgen::func::ret_sp_delta`).
    pub ret_sp_delta: u64,
    /// How far the `ret` at a terminator's pc moves SP, where it pops more
    /// than the return address (i386 `ret N`: stdcall, struct returns).
    pub ret_pops: &'a HashMap<u64, u64>,
    /// How the module meets code outside it.
    pub linkage: Linkage<'a>,
    /// The `F_body` type's index.
    pub body_ty: u32,
}

/// Where a call to a function the module does not hold goes.
pub enum Linkage<'a> {
    /// Spliced into the tier-1 module: an indirect call goes through `ic`,
    /// `(ctx, target, depth, need, ret) -> i32`, -1 when no function here
    /// starts at the target; a direct one leaves for the root.
    Image { ic: u32 },
    /// Installed in the process's function table:
    /// anything outside the module is a probe of the cache and a call
    /// through table `table`; a function checks, on entry, that each page
    /// it covers still holds the `PAGE_GEN` value in `pages` (by entry).
    Runtime { table: u32, pages: &'a HashMap<u64, Vec<(u64, u32)>> },
}

/// What a function's exits did, for `GEX_FJIT_STATS`.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub functions: usize,
    pub blocks: usize,
    pub calls_known: usize,
    pub calls_trapfree: usize,
    pub calls_indirect: usize,
    pub calls_leaving: usize,
    /// Calls and tail calls through the function table (runtime linkage).
    pub calls_table: usize,
    pub tail_table: usize,
    pub tail_known: usize,
    pub switches: usize,
    pub loops_outlined: usize,
    pub returns: usize,
    pub irreducible: usize,
    /// Left to tier-1 (a loop function that would not structure).
    pub dropped: usize,
    pub code_bytes: usize,
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.functions += o.functions;
        self.blocks += o.blocks;
        self.calls_known += o.calls_known;
        self.calls_trapfree += o.calls_trapfree;
        self.calls_indirect += o.calls_indirect;
        self.calls_leaving += o.calls_leaving;
        self.calls_table += o.calls_table;
        self.tail_table += o.tail_table;
        self.tail_known += o.tail_known;
        self.switches += o.switches;
        self.loops_outlined += o.loops_outlined;
        self.returns += o.returns;
        self.irreducible += o.irreducible;
        self.dropped += o.dropped;
        self.code_bytes += o.code_bytes;
    }
}

/// A function's wasm: its body, then its loop functions in order.
pub struct Emitted {
    pub body: Vec<u8>,
    pub loops: Vec<Vec<u8>>,
}

fn check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEX_FJIT_CHECK").is_some())
}

fn trace_leaves() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEX_FJIT_TRACE_LEAVES").is_some())
}

fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEX_FJIT_TRACE").is_some())
}

/// The guest stack pointer as a GIR register.
fn sp_reg() -> GuestReg {
    if gex_abi::X86 {
        GuestReg::X(4)
    } else {
        GuestReg::Sp
    }
}

fn flags() -> RegSet {
    RegSet::of(cf::FLAG_REGS)
}

/// V8's limit on one wasm function's body (`kV8MaxWasmFunctionSize`): a
/// module holding a bigger one does not compile at all. amd64 python3.13's
/// eval loop came to 8.3 MB as one function, and the browser crashed on it.
pub const V8_MAX_FUNCTION_BYTES: usize = 7_654_321;

/// The largest body [`emit`] keeps: V8's limit less room for what the
/// splice and the merger may add to a body's immediates.
/// `GEX_FJIT_MAX_BODY=<bytes>` lowers it natively, to exercise the fallback.
pub fn max_body_bytes() -> usize {
    let cap = V8_MAX_FUNCTION_BYTES - (64 << 10);
    std::env::var("GEX_FJIT_MAX_BODY").ok().and_then(|v| v.parse().ok()).map_or(cap, |v: usize| v.min(cap))
}

/// Whether an emitted function and its outlined loops all fit
/// ([`max_body_bytes`]).
pub fn fits(body: usize, loops: &[usize], max: usize) -> bool {
    body <= max && loops.iter().all(|&l| l <= max)
}

/// Emit `f` (memory form, verified, `loops` = `outer_loops(f)` when
/// outlining). `None` when a loop function's CFG is irreducible, or when
/// the function or one of its loops is over V8's size limit: the caller
/// leaves it to tier-1, block by block.
/// Blocks above which a function is laid out flat in the browser (see `emit`).
const FLAT_ABOVE: usize = 1500;

/// Nesting (`nest_bound`) above which a function is laid out flat in the
/// browser: each level is a `tree`/`within`/`node` frame on the Worker's
/// stack, a few hundred bytes to a few KB each.
const NEST_MAX: usize = 200;

/// An upper bound on the structured emission's recursion depth: `tree(x)`
/// nests one `within` per merge child, and `x`'s code goes on into its
/// dominator-tree children (inline or after a merge's block).
fn nest_bound(s: &cfg::Shape) -> usize {
    let n = s.idom.len();
    let mut d = vec![0usize; n];
    // The deepest non-merge dominator-tree child (emitted inline from `node`).
    let mut inline = vec![0usize; n];
    let mut top = 0;
    for &x in s.rpo.iter().rev() {
        let ys = &s.merge_children[x];
        let k = ys.len();
        // `within` nests once per merge child; `tree(ys[i])` runs at
        // `k - 1 - i` of them, `node(x)` under all `k`.
        let mut v = k + 1 + inline[x];
        for (i, &y) in ys.iter().enumerate() {
            v = v.max(k - i + d[y]);
        }
        d[x] = 1 + v;
        let p = s.idom[x];
        if p == x || p >= n {
            top = top.max(d[x]);
        } else if !s.merge[x] {
            inline[p] = inline[p].max(d[x]);
        }
    }
    top.max(d.iter().copied().max().unwrap_or(0))
}

pub fn emit(m: &ModuleCx<'_>, f: &Function, loops: &[Loop], stats: &mut Stats) -> Option<Emitted> {
    let n = f.blocks.len();
    // The body's graph: the blocks, then one handover node per outlined
    // loop, reached from the loop's back-edges and leading to its exits.
    let mut succs: Vec<Vec<usize>> = f.blocks.iter().map(|b| b.succs().iter().map(|s| s.index()).collect()).collect();
    for (k, lp) in loops.iter().enumerate() {
        for &i in &lp.body {
            if handover_edge(f, lp, i, lp.header) {
                succs[i].push(n + k);
            }
        }
        succs.push(lp.exits.iter().copied().collect());
    }
    let graph = Graph { succs: succs.clone(), entry: 0 };
    // In the browser the structured layout's emission recurses about as deep
    // as the function's structure, on a Worker's stack (about 1 MB): a function
    // of thousands of blocks (large game engines have them) overflows it
    // ("Maximum call stack size exceeded" in `tree`/`within`/`node`). Such a
    // function is laid out flat there.
    // The block count alone misses some deeply nested functions, so the
    // nesting is bounded too (`nest_bound`).
    // `GEX_CGEN_EXP=WASMFLAT`: the same cuts natively, to test the browser's layouts there.
    let browser = cfg!(target_family = "wasm") || cf::exp("WASMFLAT");
    let too_deep = browser && succs.len() > FLAT_ABOVE;
    let fixed = (!too_deep).then(|| cfg::make_reducible(graph)).flatten();
    let fixed = fixed.filter(|fx| !browser || nest_bound(&fx.shape) <= NEST_MAX);
    let mut layout = match fixed {
        Some(fx) => {
            if !fx.dispatch.is_empty() {
                stats.irreducible += 1;
            }
            Layout::Structured(Box::new(fx))
        }
        None => {
            // Not even that: one `loop` over a `br_table` of every block, a
            // branch setting the arm (tier-1's region form).
            stats.dropped += 1;
            if std::env::var_os("GEX_FJIT_STATS").is_some() {
                eprintln!("[fjit] {:#x}{}: irreducible or too big, laid out flat", f.entry, f.name.as_deref().map(|n| format!(" ({n})")).unwrap_or_default());
            }
            let graph = Graph { succs, entry: 0 };
            let mut arm = vec![u32::MAX; graph.succs.len()];
            let order = cfg::rpo(&graph);
            for (k, &x) in order.iter().enumerate() {
                arm[x] = k as u32;
            }
            Layout::Flat { arm, order, idx: 0 }
        }
    };
    let regs = cf::registers(f);
    let live = if cf::live_writebacks() { crate::prep::live_in(f, m.sums) } else { vec![RegSet::ALL; n] };
    let dirty: Vec<RegSet> = if cf::clean_spills() {
        cf::dirty_in(f, &Default::default()).iter().map(|d| RegSet::of(&d.iter().copied().collect::<Vec<_>>())).collect()
    } else {
        vec![RegSet::of(&regs.written.iter().copied().collect::<Vec<_>>()); n]
    };
    let weight = cf::frame_weight(&regs);
    let used: Vec<GuestReg> = regs.used.iter().copied().collect();
    let used_set = RegSet::of(&used);
    let rlive = read_live(m, f, &|_| true);
    // What the entry reads, and what a back-edge into it may leave dirty
    // (the analysis marks those dirty on the first entry too).
    let entry_set = always(used_set).union(rlive[0]).union(dirty[0]).inter(used_set);
    let vin = valid_in(m, f, &|_| true, 0, entry_set, &rlive, &dirty, used_set);

    let mut lfns = Vec::new();
    for lp in loops {
        lfns.push(emit_loop(m, f, lp, &live, stats)?);
        stats.loops_outlined += 1;
    }

    let mut b = FnBody::new(m.cx, f.entry, 4, &used, f.n_values);
    b.resumable_faults(gex_cgen::RESUMABLE_PC);
    let sp0 = b.local(ValType::I64);
    let poll = b.local(ValType::I32);
    let hot = b.local(ValType::I32);
    let lr = b.local(ValType::I32);
    let rk = b.local(ValType::I32);
    let e64 = b.local(ValType::I64);
    // Its pages unchanged since translation, or the table's newer function
    // for this address (`gex.lookup` retires this one and
    // finds or makes it), before anything else.
    if let Linkage::Runtime { table, pages } = &m.linkage {
        let none: &[(u64, u32)] = &[];
        let pages = if cf::exp("NOPAGECHECK") { none } else { pages.get(&f.entry).map(|v| v.as_slice()).unwrap_or(none) };
        for &(word, gen) in pages {
            let fb = b.f();
            fb.addr_const(0);
            fb.load64(0x28, 2, word); // i32.load
            fb.i32_const(gen as i32);
            fb.byte(0x47); // i32.ne
            fb.if_void();
            fb.local_get(0);
            fb.addr_i32();
            fb.i64_const(f.entry as i64);
            fb.store(0x37, 3, ctx::LAST_TARGET as u32);
            fb.local_get(0);
            fb.local_get(1);
            fb.local_get(2);
            fb.local_get(3);
            // Straight to `resolve`, not through the probe cache: when only
            // a later page changed, the entry's cache line still names this
            // function, and probing would come back here forever.
            fb.i32_const(gex_abi::ftab::RESOLVE as i32);
            fb.ftab_index();
            fb.return_call_indirect(m.body_ty, *table);
            fb.end();
        }
    }
    // The recursion-depth guard, before anything is loaded.
    if !m.trapfree.contains_key(&f.entry) {
        let limit = cf::stack_budget().saturating_sub(weight + cf::TRAPFREE_RESERVE);
        let fb = b.f();
        fb.local_get(1);
        fb.i32_const(limit as i32);
        fb.byte(0x4a); // i32.gt_s
        fb.if_void();
        b.store_pc(f.entry);
        b.f().i32_const(0);
        b.f().ret();
        b.f().end();
    }
    b.load(&regs_of(entry_set));
    let spl = b.reg(sp_reg());
    b.f().local_get(spl);
    b.f().local_set(sp0);
    b.f().i32_const(cf::LOOP_POLL);
    b.f().local_set(poll);
    if let Layout::Flat { idx, .. } = &mut layout {
        *idx = b.local(ValType::I32);
    }
    let sel = b.local(ValType::I32);
    let mut e = E {
        m,
        f,
        b,
        layout,
        labels: Vec::new(),
        live: &live,
        dirty,
        weight,
        locals: Locals { sp0, poll, hot, lr, rk, sel, e64 },
        mode: Mode::Body { loops },
        n_blocks: n,
        stats,
        pending: None,
        rlive,
        vin,
        valid: entry_set,
        used: used_set,
    };
    e.stats.functions += 1;
    e.stats.blocks += f.blocks.iter().filter(|b| !b.synthetic).count();
    e.top(0);
    e.b.f().unreachable();
    let body = e.b.finish();
    let loop_lens: Vec<usize> = lfns.iter().map(|l| l.len()).collect();
    if !fits(body.len(), &loop_lens, max_body_bytes()) {
        if std::env::var_os("GEX_FJIT_STATS").is_some() {
            eprintln!(
                "[fjit] {:#x}{}: {} bytes (loops {:?}), over V8's function size limit; left to tier-1",
                f.entry,
                f.name.as_deref().map(|n| format!(" ({n})")).unwrap_or_default(),
                body.len(),
                loop_lens
            );
        }
        return None;
    }
    e.stats.code_bytes += body.len() + loop_lens.iter().sum::<usize>();
    Some(Emitted { body, loops: lfns })
}

/// The last write of each flag in `stmts` that nothing after it in the
/// block reads, traps on or reloads: such a write matters only on edges
/// whose target reads the flag, and is done there (in V8's baseline tier a
/// `cmp` before a loop's branch would otherwise compute all four flags on
/// every trip for an exit taken once). `GEX_FJIT_EXP=NOSINK` keeps them.
fn trailing_flag_writes(stmts: &[Stmt]) -> Vec<usize> {
    if cf::exp("NOSINK") {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen = RegSet::EMPTY;
    for (i, s) in stmts.iter().enumerate().rev() {
        match s {
            Stmt::WriteReg { reg, .. } => {
                if reg.is_flag() && cf::FLAG_REGS.contains(reg) && !seen.contains(*reg) {
                    out.push(i);
                }
                seen.insert(*reg);
            }
            Stmt::Def { expr: Expr::ReadReg(r), .. } => seen.insert(*r),
            Stmt::Trap { .. } => break,
            _ if gex_cgen::emit::clobbers(s) => break,
            _ => {}
        }
    }
    out
}

fn regs_of(s: RegSet) -> Vec<GuestReg> {
    (0..gex_ir::func::N_REGS).filter(|&k| s.0 >> k & 1 != 0).map(gex_ir::func::reg_of_index).collect()
}

/// Registers whose locals must be loaded wherever any are: the stack
/// pointer (the return check reads its local, not a `ReadReg`) and the
/// flags (a write-back packs all four).
/// What a call's continuation reloads whether or not it reads it: SP, and
/// the flags unless calls are taken to clobber them (`prep::flags_abi`), in
/// which case only a continuation that reads one reloads it.
fn after_call(used: RegSet) -> RegSet {
    if crate::prep::flags_abi() {
        RegSet::of(&[sp_reg()]).inter(used)
    } else {
        always(used)
    }
}

fn always(used: RegSet) -> RegSet {
    flags().union(RegSet::of(&[sp_reg()])).inter(used)
}

/// The registers a call leaves in their locals: a summarised callee's
/// preserved ones (not reloaded after it).
fn kept(m: &ModuleCx<'_>, callee: &Callee) -> RegSet {
    match callee {
        Callee::Direct(t) if m.bodies.contains_key(t) && cf::live_writebacks() => {
            m.sums.get(t).map_or(RegSet::EMPTY, |sm| sm.preserved.minus(RegSet::of(&[sp_reg()])))
        }
        _ => RegSet::EMPTY,
    }
}

/// The registers some path from each block's entry reads (a `ReadReg`)
/// before writing, over the blocks `keep` admits. A call reloads what its
/// continuation reads but the callee's preserved registers (`kept`), which
/// must hold their values before it; a syscall and a clobbering host call
/// reload everything.
fn read_live(m: &ModuleCx<'_>, f: &Function, keep: &dyn Fn(usize) -> bool) -> Vec<RegSet> {
    let n = f.blocks.len();
    if cf::exp("FULLRELOAD") {
        return vec![RegSet::ALL; n];
    }
    let mut live = vec![RegSet::EMPTY; n];
    let order: Vec<usize> = f.rpo().iter().rev().map(|b| b.index()).filter(|&i| keep(i)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &i in &order {
            let b = &f.blocks[i];
            let mut l = match &b.term {
                FTerm::Call { callee, ret: Some(r), .. } if keep(r.to.index()) => live[r.to.index()].inter(kept(m, callee)),
                FTerm::Call { .. } | FTerm::Syscall { .. } => RegSet::EMPTY,
                t => t.edges().iter().map(|e| e.to.index()).filter(|&s| keep(s)).fold(RegSet::EMPTY, |a, s| a.union(live[s])),
            };
            for st in b.stmts.iter().rev() {
                match st {
                    Stmt::WriteReg { reg, .. } => l.remove(*reg),
                    Stmt::Def { expr: Expr::ReadReg(r), .. } => l.insert(*r),
                    _ if gex_cgen::emit::clobbers(st) => l = RegSet::EMPTY,
                    _ => {}
                }
            }
            if l != live[i] {
                live[i] = l;
                changed = true;
            }
        }
    }
    live
}

/// For each block of those `keep` admits, the registers whose locals hold
/// the guest's values on entry: a must-analysis from `entry` (which loads
/// `entry_set`). A write makes a register valid; a call leaves its
/// preserved registers as they were and reloads only what its continuation
/// reads (`read_live`); a syscall or clobbering host call reloads all. An
/// edge loads what its target may find dirty and is not valid yet (see
/// `E::edge_inner`), so every dirty register is a valid one.
#[allow(clippy::too_many_arguments)]
fn valid_in(
    m: &ModuleCx<'_>,
    f: &Function,
    keep: &dyn Fn(usize) -> bool,
    entry: usize,
    entry_set: RegSet,
    rlive: &[RegSet],
    dirty: &[RegSet],
    used: RegSet,
) -> Vec<RegSet> {
    let n = f.blocks.len();
    let mut vin = vec![used; n];
    let mut seen = vec![false; n];
    vin[entry] = entry_set;
    seen[entry] = true;
    let order: Vec<usize> = f.rpo().iter().map(|b| b.index()).filter(|&i| keep(i)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &i in &order {
            if !seen[i] {
                continue;
            }
            let b = &f.blocks[i];
            let mut v = vin[i];
            for st in &b.stmts {
                match st {
                    Stmt::WriteReg { reg, .. } => v.insert(*reg),
                    _ if gex_cgen::emit::clobbers(st) => v = used,
                    _ => {}
                }
            }
            let after = match &b.term {
                FTerm::Call { callee, ret: Some(r), .. } => {
                    let kept = kept(m, callee);
                    let reload = after_call(used).union(rlive[r.to.index()]).inter(used).minus(kept);
                    v.inter(kept).union(reload)
                }
                FTerm::Syscall { .. } => used,
                _ => v,
            };
            for e in b.term.edges() {
                let s = e.to.index();
                if !keep(s) {
                    continue;
                }
                let new = after.union(dirty[s].inter(used));
                let merged = if seen[s] { vin[s].inter(new) } else { new };
                if !seen[s] || merged != vin[s] {
                    vin[s] = merged;
                    seen[s] = true;
                    changed = true;
                }
            }
        }
    }
    vin
}

/// Whether `i -> h` is a back-edge of outlined loop `lp` where the body
/// hands the loop over (gex-cgen's rule: a backward branch by address).
fn handover_edge(f: &Function, lp: &Loop, i: usize, t: usize) -> bool {
    t == lp.header && lp.body.contains(&i) && backward_addr(f, i, t)
}

/// tier-1's rule for a backward branch, which polls: by guest address.
fn backward_addr(f: &Function, i: usize, t: usize) -> bool {
    let to = &f.blocks[t];
    !to.synthetic && to.start <= f.blocks[i].start
}

/// Loop function `k` of `f`: `(ctx, depth, need, ret, sp0) -> i32`.
fn emit_loop(m: &ModuleCx<'_>, f: &Function, lp: &Loop, live: &[RegSet], stats: &mut Stats) -> Option<Vec<u8>> {
    let n = f.blocks.len();
    let in_loop = |i: usize| lp.body.contains(&i);
    let succs: Vec<Vec<usize>> = (0..n)
        .map(|i| if in_loop(i) { f.blocks[i].succs().iter().map(|s| s.index()).filter(|&s| in_loop(s)).collect() } else { Vec::new() })
        .collect();
    let shape = Layout::Structured(Box::new(cfg::make_reducible(Graph { succs, entry: lp.header })?));
    let regs = cf::registers_in(f, &in_loop);
    let dirty: Vec<RegSet> = if cf::clean_spills() {
        cf::dirty_in_from(f, lp.header, &in_loop, &Default::default())
            .iter()
            .map(|d| RegSet::of(&d.iter().copied().collect::<Vec<_>>()))
            .collect()
    } else {
        vec![RegSet::of(&regs.written.iter().copied().collect::<Vec<_>>()); n]
    };
    let weight = cf::frame_weight(&regs);
    let used: Vec<GuestReg> = regs.used.iter().copied().collect();
    let used_set = RegSet::of(&used);
    let rlive = read_live(m, f, &in_loop);
    let entry_set = always(used_set).union(rlive[lp.header]).union(dirty[lp.header]).inter(used_set);
    let vin = valid_in(m, f, &in_loop, lp.header, entry_set, &rlive, &dirty, used_set);
    let mut b = FnBody::new(m.cx, f.blocks[lp.header].start, 5, &used, f.n_values);
    b.resumable_faults(gex_cgen::RESUMABLE_PC);
    let poll = b.local(ValType::I32);
    let hot = b.local(ValType::I32);
    let lr = b.local(ValType::I32);
    let rk = b.local(ValType::I32);
    let sel = b.local(ValType::I32);
    let e64 = b.local(ValType::I64);
    b.load(&regs_of(entry_set));
    b.f().i32_const(cf::LOOP_POLL);
    b.f().local_set(poll);
    let mut e = E {
        m,
        f,
        b,
        layout: shape,
        labels: Vec::new(),
        live,
        dirty,
        weight,
        locals: Locals { sp0: 4, poll, hot, lr, rk, sel, e64 },
        mode: Mode::Loop { lp },
        n_blocks: n,
        stats,
        pending: None,
        rlive,
        vin,
        valid: entry_set,
        used: used_set,
    };
    e.top(lp.header);
    e.b.f().unreachable();
    Some(e.b.finish())
}

struct Locals {
    sp0: u32,
    poll: u32,
    hot: u32,
    lr: u32,
    rk: u32,
    /// A dispatch node's selector (`cfg::make_reducible`).
    sel: u32,
    /// Scratch for a probe of the function table's cache.
    e64: u32,
}

/// Push the table index the probe cache holds for constant `t`
/// (`gex_abi::fcache_entry`), 0 (resolve) on a miss. `e` is i64 scratch.
fn probe_const(fb: &mut gex_emit::wasm::FuncBody, t: u64, e: u32) {
    fb.addr_const(0);
    fb.load(0x29, 3, gex_abi::fcache_entry(t) as u32); // i64.load
    fb.local_tee(e);
    fb.i64_const(gex_abi::FCACHE_TAG_BITS as i64);
    fb.byte(0x88); // i64.shr_u
    fb.byte(0xa7); // i32.wrap_i64
    fb.i32_const(gex_abi::ftab::RESOLVE as i32);
    fb.local_get(e);
    fb.i64_const(gex_abi::FCACHE_TAG_MASK as i64);
    fb.byte(0x83); // i64.and
    fb.i64_const(t as i64);
    fb.byte(0x51); // i64.eq
    fb.select();
}

enum Mode<'f> {
    Body { loops: &'f [Loop] },
    Loop { lp: &'f Loop },
}

/// A call's target, for the table.
#[derive(Clone, Copy)]
enum Target {
    Const(u64),
    Op(Operand),
}

/// An enclosing label, for branch depths.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Label {
    /// A `loop` headed by this node.
    Loop(usize),
    /// A `block` whose end is followed by this node.
    Follow(usize),
    /// The `loop` around a flat layout's `br_table`.
    Dispatch,
    /// Anything else (`if`, a switch's case blocks, a do-while).
    Other,
}

/// How a function's blocks are laid out.
enum Layout {
    /// Ramsey's structure (`cfg`): every branch a `br`, but into an
    /// irreducible region's entries, which go through its dispatch node.
    Structured(Box<cfg::Fixed>),
    /// An irreducible CFG: a `loop` whose `br_table` picks the next block,
    /// `arm[x]` being block `x`'s arm and `idx` the local holding it.
    Flat { arm: Vec<u32>, order: Vec<usize>, idx: u32 },
}

struct E<'m, 'a> {
    m: &'m ModuleCx<'a>,
    f: &'m Function,
    b: FnBody<'a>,
    layout: Layout,
    labels: Vec<Label>,
    live: &'m [RegSet],
    dirty: Vec<RegSet>,
    weight: u32,
    locals: Locals,
    mode: Mode<'m>,
    n_blocks: usize,
    stats: &'m mut Stats,
    /// The current block's trailing flag writes, done on the edges that
    /// need them (`trailing_flag_writes`).
    pending: Option<(Deferred, bool)>,
    /// Per block, the registers some path from its entry reads before
    /// writing (`read_live`): what a reload after a call loads.
    rlive: Vec<RegSet>,
    /// Per block, the registers whose locals hold the guest's values on
    /// entry (`valid_in`).
    vin: Vec<RegSet>,
    /// The same, here in the emission.
    valid: RegSet,
    used: RegSet,
}

impl E<'_, '_> {
    /// Write back the dirty registers in `mask` (`FnBody::flush`), which
    /// must all hold their values (`GEX_FJIT_CHECK=1` checks it).
    fn flush(&mut self, mask: RegSet, keep_dirty: bool) {
        if check() {
            let bad = self.b.dirty().inter(mask).inter(self.used).minus(self.valid);
            assert!(
                bad == RegSet::EMPTY,
                "fjit {:#x}: writing back registers whose locals are stale: {:?}",
                self.f.entry,
                regs_of(bad)
            );
        }
        self.b.flush(mask, keep_dirty);
    }

    // -- structure ---------------------------------------------------------

    fn shape(&self) -> &Shape {
        match &self.layout {
            Layout::Structured(s) => &s.shape,
            Layout::Flat { .. } => unreachable!("a flat layout has no shape"),
        }
    }

    /// The whole function from `entry`.
    fn top(&mut self, entry: usize) {
        let Layout::Flat { arm, order, idx } = &self.layout else {
            let Layout::Structured(fx) = &self.layout else { unreachable!() };
            let (start, root) = (fx.start, fx.graph.entry);
            if let Some(k) = start {
                // The entry is an irreducible region's: through its dispatch.
                let sel = self.locals.sel;
                self.b.f().i32_const(k as i32);
                self.b.f().local_set(sel);
            }
            let _ = entry;
            self.tree(root);
            return;
        };
        let (order, idx) = (order.clone(), *idx);
        let n = order.len() as u32;
        let fb = self.b.f();
        fb.i32_const(arm[entry] as i32);
        fb.local_set(idx);
        fb.loop_void();
        self.labels.push(Label::Dispatch);
        for _ in 0..n {
            self.b.f().block_void();
            self.labels.push(Label::Other);
        }
        let fb = self.b.f();
        fb.local_get(idx);
        let targets: Vec<u32> = (0..n).collect();
        fb.br_table(&targets, 0);
        // Arm k's code follows the end of the k-th block out.
        for &x in &order {
            self.close();
            self.node(x);
        }
        self.labels.pop();
        self.b.f().end();
    }

    fn tree(&mut self, x: usize) {
        let ys = self.shape().merge_children[x].clone();
        if self.shape().loop_header[x] {
            self.b.f().loop_void();
            self.labels.push(Label::Loop(x));
            self.within(x, &ys);
            self.labels.pop();
            self.b.f().end();
        } else {
            self.within(x, &ys);
        }
    }

    fn within(&mut self, x: usize, ys: &[usize]) {
        match ys.split_last() {
            Some((&y, rest)) => {
                self.b.f().block_void();
                self.labels.push(Label::Follow(y));
                self.within(x, rest);
                self.labels.pop();
                self.b.f().end();
                self.tree(y);
            }
            None => self.node(x),
        }
    }

    fn depth(&self, l: Label) -> u32 {
        let at = self.labels.iter().rposition(|&k| k == l).expect("structured branch to a label not in scope");
        (self.labels.len() - 1 - at) as u32
    }

    /// Transfer from node `x` to node `y`: a `br`, or `y`'s code inline.
    fn go(&mut self, x: usize, y: usize) {
        if let Layout::Flat { arm, idx, .. } = &self.layout {
            let (a, idx) = (arm[y], *idx);
            let d = self.depth(Label::Dispatch);
            let fb = self.b.f();
            fb.i32_const(a as i32);
            fb.local_set(idx);
            fb.br(d);
            return;
        }
        if let Layout::Structured(fx) = &self.layout {
            if let Some(&(d, k)) = fx.redirect.get(&(x, y)) {
                let sel = self.locals.sel;
                self.b.f().i32_const(k as i32);
                self.b.f().local_set(sel);
                self.go(x, d);
                return;
            }
        }
        if self.shape().backward(x, y) {
            let d = self.depth(Label::Loop(y));
            self.b.f().br(d);
        } else if self.shape().merge[y] {
            let d = self.depth(Label::Follow(y));
            self.b.f().br(d);
        } else {
            self.tree(y);
        }
    }

    fn node(&mut self, x: usize) {
        if x >= self.n_blocks {
            let entries = match &self.layout {
                Layout::Structured(fx) => fx.dispatch.iter().find(|d| d.node == x).map(|d| d.entries.clone()),
                Layout::Flat { .. } => None,
            };
            match entries {
                Some(es) => self.dispatch(x, &es),
                None => self.handover(x - self.n_blocks),
            }
            return;
        }
        let f = self.f;
        let blk = &f.blocks[x];
        self.b.set_dirty(self.dirty[x]);
        if trace() && !blk.synthetic {
            // `GEX_FJIT_TRACE=1`: the host prints each block as it is
            // entered, with the registers written back first.
            self.flush(RegSet::ALL, true);
            let bp = self.m.cx.breakpoint;
            let fb = self.b.f();
            fb.local_get(0);
            fb.i64_const(blk.start as i64);
            fb.i32_const(gex_abi::imports::BRK_TRACE);
            fb.call(bp);
        }
        self.valid = self.vin[x];
        for st in &blk.stmts {
            match st {
                Stmt::WriteReg { reg, .. } => self.valid.insert(*reg),
                Stmt::Def { expr: Expr::ReadReg(r), .. } if check() && !self.valid.contains(*r) && *r != GuestReg::CtxAddr => {
                    panic!("fjit {:#x}: block {:#x} reads {r:?}, whose local is stale", f.entry, blk.start)
                }
                _ if gex_cgen::emit::clobbers(st) => self.valid = self.used,
                _ => {}
            }
        }
        let mut keep: Vec<Value> = Vec::new();
        blk.term.for_each_use(|v| keep.push(v));
        let defer = trailing_flag_writes(&blk.stmts);
        // Faults are delivered in place, at their instruction's pc; `int3`
        // and `ud2` carry on at the next instruction (as gex-cgen's
        // `block_stmts` gives a unit's).
        let pcs = blk
            .stmts
            .iter()
            .enumerate()
            .map(|(k, st)| {
                let pc = blk.pcs.get(k).copied().unwrap_or(blk.start);
                let next = matches!(st, Stmt::HostCall { func: HostFunc::Breakpoint | HostFunc::Illegal, .. })
                    .then(|| blk.insts.iter().position(|&a| a == pc).and_then(|j| blk.insts.get(j + 1).copied()).unwrap_or(blk.end));
                (pc, next)
            })
            .collect();
        self.b.stmt_pcs(pcs);
        let d = self.b.block(blk.start, &blk.stmts, &keep, &defer);
        // A terminator has at most one value; it outlives the block's
        // numbering, which code inlined into the terminator reuses.
        let op = keep.first().map(|&v| self.b.operand(v));
        let saved = std::mem::replace(&mut self.pending, Some((d, false)));
        self.term(x, op);
        let (d, _) = std::mem::replace(&mut self.pending, saved).expect("this block's deferred writes");
        if let Some(o) = op {
            self.b.release_operand(o);
        }
        self.b.end_block(d);
    }

    // -- edges ---------------------------------------------------------------

    fn in_nested(&self, i: usize) -> bool {
        match self.mode {
            Mode::Loop { lp, .. } => lp.nested,
            Mode::Body { loops } => loops.iter().any(|lp| lp.nested && lp.body.contains(&i)),
        }
    }

    /// `if (--poll <= 0)`, opened.
    fn poll_expired(&mut self) {
        let p = self.locals.poll;
        let fb = self.b.f();
        fb.local_get(p);
        fb.i32_const(1);
        fb.byte(0x6b); // i32.sub
        fb.local_tee(p);
        fb.i32_const(0);
        fb.byte(0x4c); // i32.le_s
        fb.if_void();
        self.labels.push(Label::Other);
    }

    fn close(&mut self) {
        self.labels.pop();
        self.b.f().end();
    }

    /// The host's safepoint at `pc`, called outright (the countdown expired).
    fn call_safepoint(&mut self, pc: u64) {
        self.b.store_pc(pc);
        let sp = self.m.cx.safepoint;
        let fb = self.b.f();
        fb.local_get(0);
        fb.i64_const(pc as i64);
        fb.call(sp);
    }

    /// Whether edge `i -> t` is a bare `br` (no code on the edge).
    fn plain_br(&self, i: usize, t: usize) -> bool {
        let pend = match &self.pending {
            Some((d, false)) => FnBody::deferred_regs(d).any(|r| self.live[t].contains(r)),
            _ => false,
        };
        let fix = t < self.dirty.len() && self.dirty[t].inter(self.used).minus(self.valid) != RegSet::EMPTY;
        if backward_addr(self.f, i, t) || pend || fix {
            return false;
        }
        match (&self.layout, &self.mode) {
            (Layout::Flat { .. }, _) => false,
            (Layout::Structured(fx), _) if fx.redirect.contains_key(&(i, t)) => false,
            (_, Mode::Loop { lp, .. }) if !lp.body.contains(&t) => false,
            _ => self.shape().branched(i, t),
        }
    }

    /// The block's flag writes left to its edges, for those whose target
    /// reads the flag.
    fn pending_for(&mut self, t: usize) {
        let live = self.live[t];
        if let Some((d, false)) = &self.pending {
            self.b.emit_deferred(d, &|r| live.contains(r));
        }
    }

    /// All of them, before a terminator that leaves (or calls): from here on
    /// none is pending.
    fn pending_all(&mut self) {
        if let Some((d, done)) = &mut self.pending {
            if !*done {
                self.b.emit_deferred(d, &|_| true);
                *done = true;
            }
        }
    }

    /// The code on edge `i -> t` (block indices), then the transfer. The
    /// registers' dirty state is the block's again afterwards: the target's
    /// code may have been emitted inline here, and the terminator's next
    /// edge (or its default) starts from the block's state, not the end of
    /// that code.
    fn edge(&mut self, i: usize, t: usize) {
        let (dirty, valid) = (self.b.dirty(), self.valid);
        self.edge_inner(i, t);
        self.b.set_dirty(dirty);
        self.valid = valid;
    }

    fn edge_inner(&mut self, i: usize, t: usize) {
        // A register `t` may find dirty must hold its value here too: one
        // this path never loaded is loaded now (clean on this path).
        if t < self.dirty.len() {
            let fix = self.dirty[t].inter(self.used).minus(self.valid);
            if fix != RegSet::EMPTY {
                self.b.load(&regs_of(fix));
                self.valid = self.valid.union(fix);
            }
        }
        self.pending_for(t);
        let f = self.f;
        let backward = backward_addr(f, i, t);
        let to_start = f.blocks[t].start;
        match self.mode {
            Mode::Body { loops } => {
                if let Some(k) = loops.iter().position(|lp| handover_edge(f, lp, i, t)) {
                    // An outermost loop's back-edge in the body: the first
                    // 1024 trips run here, then the loop moves to its own
                    // function, a fresh call per chunk.
                    let nested = loops[k].nested;
                    self.poll_expired();
                    let p = self.locals.poll;
                    self.b.f().i32_const(1024);
                    self.b.f().local_set(p);
                    self.flush(RegSet::ALL, true);
                    self.call_safepoint(to_start);
                    if nested {
                        let h = self.locals.hot;
                        self.b.f().i32_const(1);
                        self.b.f().local_set(h);
                        self.close();
                        self.b.f().local_get(h);
                        self.b.f().if_void();
                        self.labels.push(Label::Other);
                        self.flush(RegSet::ALL, true);
                    }
                    self.go(i, self.n_blocks + k);
                    self.close();
                    self.go(i, t);
                    return;
                }
            }
            Mode::Loop { lp, .. } => {
                if !lp.body.contains(&t) {
                    // Leaving the loop: everything back, and the body
                    // carries on at `t`.
                    self.flush(RegSet::ALL, true);
                    self.b.f().i32_const(LOOP_EXIT + t as i32);
                    self.b.f().ret();
                    return;
                }
                if t == lp.header && backward {
                    // The loop's own back-edge: a chunk ends here when the
                    // countdown expires.
                    self.poll_expired();
                    self.flush(RegSet::ALL, true);
                    self.call_safepoint(to_start);
                    self.b.f().i32_const(LOOP_CHUNK);
                    self.b.f().ret();
                    self.close();
                    if lp.nested {
                        let h = self.locals.hot;
                        self.b.f().local_get(h);
                        self.b.f().if_void();
                        self.labels.push(Label::Other);
                        self.flush(RegSet::ALL, true);
                        self.b.f().i32_const(LOOP_CHUNK);
                        self.b.f().ret();
                        self.close();
                    }
                    self.go(i, t);
                    return;
                }
            }
        }
        if backward && !cf::exp("NOPOLL") {
            // tier-1's rule for a backward branch: poll, with the countdown
            // in a local; the slow path writes back what the host may see.
            self.poll_expired();
            let p = self.locals.poll;
            self.b.f().i32_const(1024);
            self.b.f().local_set(p);
            self.flush(cf::non_flags().union(self.live[t]), true);
            self.call_safepoint(to_start);
            if self.in_nested(i) {
                let h = self.locals.hot;
                self.b.f().i32_const(1);
                self.b.f().local_set(h);
            }
            self.close();
        }
        self.go(i, t);
    }

    /// A dispatch node: `br_table` on the selector to the region's entries.
    fn dispatch(&mut self, x: usize, entries: &[usize]) {
        let k = entries.len() as u32;
        for _ in 0..k {
            self.b.f().block_void();
            self.labels.push(Label::Other);
        }
        let sel = self.locals.sel;
        let fb = self.b.f();
        fb.local_get(sel);
        let targets: Vec<u32> = (0..k).collect();
        fb.br_table(&targets, k - 1);
        for &e in entries {
            self.close();
            self.go(x, e);
        }
    }

    /// Loop `k`'s handover node in the body: run the loop function chunk by
    /// chunk, then carry on where it left, or return as it did.
    fn handover(&mut self, k: usize) {
        let entry = self.f.entry;
        let Mode::Body { loops } = self.mode else { unreachable!("a handover node outside a body") };
        let lf = self.m.loops[&(entry, k)];
        let (lr, sp0, w) = (self.locals.lr, self.locals.sp0, self.weight);
        let fb = self.b.f();
        fb.loop_void();
        fb.local_get(0);
        fb.local_get(1);
        fb.i32_const(w as i32);
        fb.byte(0x6a); // i32.add
        fb.local_get(2);
        fb.local_get(3);
        fb.local_get(sp0);
        fb.call(lf);
        fb.local_tee(lr);
        fb.i32_const(LOOP_CHUNK);
        fb.byte(0x46); // i32.eq
        fb.br_if(0);
        fb.end();
        fb.local_get(lr);
        fb.i32_const(1);
        fb.byte(0x4c); // i32.le_s
        fb.if_void();
        fb.local_get(lr);
        fb.ret();
        fb.end();
        self.b.reload_except(RegSet::EMPTY);
        self.valid = self.used;
        let me = self.n_blocks + k;
        for &x in &loops[k].exits {
            let fb = self.b.f();
            fb.local_get(lr);
            fb.i32_const(LOOP_EXIT + x as i32);
            fb.byte(0x46); // i32.eq
            fb.if_void();
            self.labels.push(Label::Other);
            self.b.set_dirty(RegSet::EMPTY);
            self.go(me, x);
            self.close();
        }
        self.b.f().unreachable();
    }

    // -- exits -------------------------------------------------------------

    /// Write everything back, set PC, return 0: the root takes it from here.
    fn leave_const(&mut self, pc: u64) {
        self.flush(RegSet::ALL, false);
        self.b.store_pc(pc);
        self.b.f().i32_const(0);
        self.b.f().ret();
    }

    fn leave_value(&mut self, v: Operand) {
        self.flush(RegSet::ALL, false);
        self.store_pc_value(v);
        self.b.f().i32_const(0);
        self.b.f().ret();
    }

    fn store_pc_value(&mut self, v: Operand) {
        self.b.f().local_get(0);
        self.b.f().addr_i32();
        self.b.push_operand(v);
        self.b.f().store(0x37, 3, ctx::PC as u32);
    }

    /// Push the table index for the target `t` (a constant or the
    /// operand), after storing it in `ctx::LAST_TARGET` -- every call
    /// through the table does, for the resolve stub.
    /// The caller pushes the arguments in between: `before` does.
    fn probe_call(&mut self, t: Target, before: &mut dyn FnMut(&mut Self), tail: bool) {
        let Linkage::Runtime { table, .. } = self.m.linkage else { unreachable!("a probe outside the runtime linkage") };
        let e = self.locals.e64;
        self.b.f().local_get(0);
        self.b.f().addr_i32();
        match t {
            Target::Const(a) => self.b.f().i64_const(a as i64),
            Target::Op(o) => self.b.push_operand(o),
        }
        self.b.f().store(0x37, 3, ctx::LAST_TARGET as u32);
        before(self);
        match t {
            Target::Const(a) => probe_const(self.b.f(), a, e),
            Target::Op(o) => {
                // e = cache[(t32 * MUL) >> (32 - BITS)]
                self.b.push_operand(o);
                let fb = self.b.f();
                fb.byte(0xa7); // i32.wrap_i64
                fb.i32_const(gex_abi::FCACHE_MUL as i32);
                fb.byte(0x6c); // i32.mul
                fb.i32_const(32 - gex_abi::FCACHE_BITS as i32);
                fb.byte(0x76); // i32.shr_u
                fb.i32_const(3);
                fb.byte(0x74); // i32.shl
                fb.addr_i32();
                fb.load(0x29, 3, gex_abi::FCACHE as u32); // i64.load
                fb.local_tee(e);
                fb.i64_const(gex_abi::FCACHE_TAG_BITS as i64);
                fb.byte(0x88); // i64.shr_u
                fb.byte(0xa7); // i32.wrap_i64
                fb.i32_const(gex_abi::ftab::RESOLVE as i32);
                fb.local_get(e);
                fb.i64_const(gex_abi::FCACHE_TAG_MASK as i64);
                fb.byte(0x83); // i64.and
                self.b.push_operand(o);
                let fb = self.b.f();
                fb.byte(0x51); // i64.eq
                fb.select();
            }
        }
        let ty = self.m.body_ty;
        let fb = self.b.f();
        fb.ftab_index();
        if tail {
            fb.return_call_indirect(ty, table);
        } else {
            fb.byte(gex_emit::wasm::op::CALL_INDIRECT);
            fb.u32(ty);
            fb.u32(table);
        }
    }

    /// A tail call through the table (runtime linkage): everything written
    /// back, a poll, and the callee returns to this function's caller.
    fn tail_through_table(&mut self, t: Target) {
        self.flush(RegSet::ALL, false);
        match t {
            Target::Const(a) => self.b.safepoint(a),
            Target::Op(o) => {
                let e = self.locals.e64;
                self.b.push_operand(o);
                self.b.f().local_set(e);
                self.b.safepoint_dynamic(e);
            }
        }
        self.probe_call(
            t,
            &mut |me: &mut Self| {
                let fb = me.b.f();
                fb.local_get(0);
                fb.local_get(1);
                fb.local_get(2);
                fb.local_get(3);
            },
            true,
        );
    }

    /// Leaves the i32 "`entry`'s code was rewritten" (`GEX_PATCHED`, one
    /// byte at a constant address per page) on the stack. A run-time
    /// translation checks its own pages on entry instead: never.
    fn patched(&mut self, entry: u64) {
        let fb = self.b.f();
        if cf::mutate("NOPATCHCHECK") || matches!(self.m.linkage, Linkage::Runtime { .. }) {
            fb.i32_const(0);
            return;
        }
        let one = |fb: &mut gex_emit::wasm::FuncBody, a: u64| {
            fb.addr_const(0);
            fb.load64(0x2d, 0, gex_abi::patched_pages() + (a >> 15)); // i32.load8_u
            fb.i32_const(((a >> 12) & 7) as i32);
            fb.byte(0x76); // i32.shr_u
            fb.i32_const(1);
            fb.byte(0x71); // i32.and
        };
        one(fb, entry);
        if entry % gex_abi::DISPATCH_PAGE > gex_abi::DISPATCH_PAGE - 5 {
            one(fb, entry + 5);
            fb.byte(0x72); // i32.or
        }
    }

    /// `depth + w` on the stack.
    fn depth_plus(&mut self, w: u32) {
        let fb = self.b.f();
        fb.local_get(1);
        fb.i32_const(w as i32);
        fb.byte(0x6a); // i32.add
    }

    fn term(&mut self, i: usize, op: Option<Operand>) {
        let f = self.f;
        let blk = &f.blocks[i];
        let val = || op.expect("the terminator's value");
        if !matches!(blk.term, FTerm::Jump(_) | FTerm::Branch { .. } | FTerm::Switch { .. }) {
            self.pending_all();
        }
        match &blk.term {
            FTerm::Jump(e) => self.edge(i, e.to.index()),
            FTerm::Branch { taken, not_taken, .. } => {
                let (t, nt) = (taken.to.index(), not_taken.to.index());
                let cond = val();
                if self.plain_br(i, t) {
                    self.b.push_operand(cond);
                    let d = self.br_depth(i, t);
                    self.b.f().br_if(d);
                    self.edge(i, nt);
                } else if self.plain_br(i, nt) {
                    self.b.push_operand(cond);
                    self.b.f().byte(0x45); // i32.eqz
                    let d = self.br_depth(i, nt);
                    self.b.f().br_if(d);
                    self.edge(i, t);
                } else {
                    self.b.push_operand(cond);
                    self.b.f().if_void();
                    self.labels.push(Label::Other);
                    self.edge(i, t);
                    self.b.f().else_();
                    self.edge(i, nt);
                    self.close();
                }
            }
            FTerm::Switch { cases, .. } => {
                self.stats.switches += 1;
                self.switch(i, val(), cases.iter().map(|(a, e)| (*a, e.to.index())).collect());
                self.pending_all();
                self.leave_value(val());
            }
            FTerm::Call { callee, ret_pc, ret } => self.call(i, *callee, op, *ret_pc, ret.as_ref().map(|e| e.to.index())),
            FTerm::TailCall { callee } => match callee {
                Callee::Direct(t) if self.m.bodies.contains_key(t) => {
                    self.stats.tail_known += 1;
                    self.flush(RegSet::ALL, false);
                    // A tail call passes through no back-edge or root, so it
                    // polls.
                    self.b.safepoint(*t);
                    self.patched(*t);
                    self.b.f().if_void();
                    self.b.store_pc(*t);
                    self.b.f().i32_const(0);
                    self.b.f().ret();
                    self.b.f().end();
                    // The frame is replaced (a loop function's included:
                    // its result is the body's), so the depth stays.
                    let body = self.m.bodies[t];
                    let fb = self.b.f();
                    fb.local_get(0);
                    fb.local_get(1);
                    fb.local_get(2);
                    fb.local_get(3);
                    fb.return_call(body);
                }
                Callee::Direct(t) if matches!(self.m.linkage, Linkage::Runtime { .. }) => {
                    self.stats.tail_table += 1;
                    self.tail_through_table(Target::Const(*t));
                }
                Callee::Indirect(_) if matches!(self.m.linkage, Linkage::Runtime { .. }) => {
                    self.stats.tail_table += 1;
                    self.tail_through_table(Target::Op(val()));
                }
                Callee::Direct(t) => {
                    self.stats.calls_leaving += 1;
                    self.leave_const(*t)
                }
                Callee::Indirect(_) => self.leave_value(val()),
            },
            FTerm::Return { .. } => {
                self.stats.returns += 1;
                // The return the caller expects: write back what its
                // continuation may read and report success; anything else
                // leaves with everything written back.
                // SP back where the caller had it, plus what this `ret`
                // pops: the return address, and on i386 `ret N`'s N (a
                // stdcall callee's arguments, a struct return's pointer).
                // The caller reloads SP.
                let delta = match self.m.ret_pops.get(&blk.term_pc) {
                    Some(&d) => d,
                    None => self.m.ret_sp_delta,
                };
                let (sp, sp0) = (self.b.reg(sp_reg()), self.locals.sp0);
                if cf::mutate("NORETCHECK") {
                    self.b.f().i32_const(1);
                } else {
                    self.b.push_operand(val());
                    let fb = self.b.f();
                    fb.local_get(3);
                    fb.byte(0x51); // i64.eq
                    fb.local_get(sp);
                    fb.local_get(sp0);
                    fb.i64_const(delta as i64);
                    fb.byte(0x7c); // i64.add
                    fb.byte(0x51); // i64.eq
                    fb.byte(0x71); // i32.and
                }
                self.b.f().if_void();
                self.flush(RegSet::ALL.minus(flags()), true);
                let fb = self.b.f();
                fb.local_get(2);
                fb.i32_const(NEED_FLAGS);
                fb.byte(0x71); // i32.and
                fb.if_void();
                self.flush(flags(), true);
                self.b.f().end();
                self.b.f().i32_const(1);
                self.b.f().ret();
                self.b.f().end();
                if trace_leaves() {
                    // `GEX_FJIT_TRACE_LEAVES=1`: a failed return check
                    // prints the return address expected, the entry SP,
                    // then the registers at the `ret`.
                    self.flush(RegSet::ALL, false);
                    let bp = self.m.cx.breakpoint;
                    let fb = self.b.f();
                    fb.local_get(0);
                    fb.local_get(3);
                    fb.i32_const(gex_abi::imports::BRK_TRACE);
                    fb.call(bp);
                    fb.local_get(0);
                    fb.local_get(sp0);
                    fb.i32_const(gex_abi::imports::BRK_TRACE);
                    fb.call(bp);
                    fb.local_get(0);
                    fb.i64_const(blk.term_pc as i64);
                    fb.i32_const(gex_abi::imports::BRK_TRACE);
                    fb.call(bp);
                }
                self.leave_value(val());
            }
            // Through the table, the caller's return contract passed on:
            // an indirect jump is a tail call to whatever serves the target
            // (`jmp [iat]` into another function, the i386 syscall stubs'
            // `jmp *__wine_syscall` among them); a jump back into this
            // function reaches a region whose own return check sorts it
            // out. Leaving instead would unwind every frame above.
            FTerm::Unresolved { .. } if matches!(self.m.linkage, Linkage::Runtime { .. }) && !cf::mutate("UNRESOLVEDLEAVES") => {
                self.stats.tail_table += 1;
                self.tail_through_table(Target::Op(val()));
            }
            FTerm::Unresolved { .. } => self.leave_value(val()),
            FTerm::Syscall { next_pc, next, imm } => {
                self.flush(RegSet::ALL, false);
                self.b.store_pc(*next_pc);
                self.b.store_syscall_imm(*imm);
                let sc = self.m.cx.syscall;
                let fb = self.b.f();
                fb.local_get(0);
                fb.i64_const(blk.term_pc as i64);
                fb.call(sc);
                self.b.reload_except(RegSet::EMPTY);
                self.valid = self.used;
                match next {
                    Some(e) => self.edge(i, e.to.index()),
                    None => {
                        self.b.f().i32_const(0);
                        self.b.f().ret();
                    }
                }
            }
            FTerm::Illegal { raw, pc } => {
                self.flush(RegSet::ALL, false);
                self.b.store_pc(*pc);
                let il = self.m.cx.illegal;
                let fb = self.b.f();
                fb.local_get(0);
                fb.i64_const((*pc | gex_cgen::RESUMABLE_PC) as i64);
                fb.i32_const(*raw as i32);
                fb.call(il);
                // The host raised the fault and returned with the handlers'
                // state in the context: leave for it.
                fb.i32_const(0);
                fb.ret();
            }
        }
    }

    fn br_depth(&self, i: usize, t: usize) -> u32 {
        if self.shape().backward(i, t) {
            self.depth(Label::Loop(t))
        } else {
            self.depth(Label::Follow(t))
        }
    }

    /// A jump table's cases; the default (falling out) is the caller's.
    /// Dense tables become a `br_table`, others a chain of compares.
    fn switch(&mut self, i: usize, target: Operand, cases: Vec<(u64, usize)>) {
        let shift = gex_abi::INST_ALIGN_SHIFT;
        let (lo, hi) = (cases.first().map(|c| c.0).unwrap_or(0), cases.last().map(|c| c.0).unwrap_or(0));
        let span = ((hi - lo) >> shift) + 1;
        if cases.len() < 3 || span > (64 * cases.len() as u64).max(256).min(4096) {
            for (a, t) in cases {
                self.b.push_operand(target);
                let fb = self.b.f();
                fb.i64_const(a as i64);
                fb.byte(0x51); // i64.eq
                fb.if_void();
                self.labels.push(Label::Other);
                self.edge(i, t);
                self.close();
            }
            return;
        }
        // Distinct targets, each a block; slot -> its depth inside them.
        let mut groups: Vec<usize> = Vec::new();
        for &(_, t) in &cases {
            if !groups.contains(&t) {
                groups.push(t);
            }
        }
        let g = groups.len() as u32;
        let tmp = self.b.local(ValType::I64);
        self.b.f().block_void(); // the default
        self.labels.push(Label::Other);
        for _ in 0..g {
            self.b.f().block_void();
            self.labels.push(Label::Other);
        }
        // Inside the innermost block group k is at depth k, the default at g.
        self.b.push_operand(target);
        let fb = self.b.f();
        fb.i64_const(lo as i64);
        fb.byte(0x7d); // i64.sub
        fb.local_tee(tmp);
        fb.i64_const(((span << shift) - 1) as i64 | ((1i64 << shift) - 1));
        fb.byte(0x56); // i64.gt_u
        fb.br_if(g);
        if shift != 0 {
            fb.local_get(tmp);
            fb.i64_const((1i64 << shift) - 1);
            fb.byte(0x83); // i64.and
            fb.byte(0xa7); // i32.wrap_i64
            fb.br_if(g);
        }
        let mut slots = vec![g; span as usize];
        for &(a, t) in &cases {
            slots[((a - lo) >> shift) as usize] = groups.iter().position(|&x| x == t).unwrap() as u32;
        }
        fb.local_get(tmp);
        if shift != 0 {
            fb.i64_const(shift as i64);
            fb.byte(0x88); // i64.shr_u
        }
        fb.byte(0xa7); // i32.wrap_i64
        fb.br_table(&slots, g);
        for &t in &groups {
            self.close();
            self.edge(i, t);
        }
        self.close();
    }

    fn call(&mut self, i: usize, callee: Callee, op: Option<Operand>, ret_pc: u64, ret: Option<usize>) {
        let m = self.m;
        let live = self.live;
        // A known callee with a summary needs its inputs, the continuation
        // what is live there; non-flag registers are limited only for a
        // trap-free callee. Anything else may read
        // everything.
        let need = match (callee, ret) {
            (Callee::Direct(t), Some(r)) if m.bodies.contains_key(&t) => match m.sums.get(&t) {
                Some(sm) => {
                    let n = sm.inputs.union(live[r]).union(RegSet::of(&[sp_reg()]));
                    if m.trapfree.contains_key(&t) || cf::mutate("IGNOREUNWIND") {
                        self.stats.calls_trapfree += 1;
                        n
                    } else {
                        n.union(cf::non_flags())
                    }
                }
                None => RegSet::ALL,
            },
            // No callee reads the flags it is entered with (`flags_abi`).
            _ if crate::prep::flags_abi() => cf::non_flags(),
            _ => RegSet::ALL,
        };
        self.flush(need, false);
        // What the callee must hand back: the flags only if the
        // continuation reads one.
        let need_back = match ret {
            Some(r) if cf::live_writebacks() => {
                if live[r].inter(flags()).is_empty() {
                    0
                } else {
                    NEED_ALL
                }
            }
            _ => NEED_ALL,
        };
        let w = self.weight;
        match callee {
            Callee::Direct(t) if m.bodies.contains_key(&t) => {
                self.stats.calls_known += 1;
                // Its code rewritten since (Detours): leave for it.
                self.patched(t);
                self.b.f().if_void();
                self.b.store_pc(t);
                self.b.f().i32_const(0);
                self.b.f().ret();
                self.b.f().end();
                let body = m.bodies[&t];
                self.b.f().local_get(0);
                self.depth_plus(w);
                let fb = self.b.f();
                fb.i32_const(need_back);
                fb.i64_const(ret_pc as i64);
                fb.call(body);
                fb.byte(0x45); // i32.eqz
                fb.if_void();
                fb.i32_const(0);
                fb.ret();
                fb.end();
            }
            Callee::Direct(_) | Callee::Indirect(_) if matches!(m.linkage, Linkage::Runtime { .. }) => {
                // Through the table: whatever serves the target, the JIT's
                // or a unit's function, or the fallback.
                self.stats.calls_table += 1;
                let t = match callee {
                    Callee::Direct(t) => Target::Const(t),
                    Callee::Indirect(_) => Target::Op(op.expect("an indirect call's target")),
                };
                self.probe_call(
                    t,
                    &mut |me: &mut Self| {
                        me.b.f().local_get(0);
                        me.depth_plus(w);
                        let fb = me.b.f();
                        fb.i32_const(need_back);
                        fb.i64_const(ret_pc as i64);
                    },
                    false,
                );
                let fb = self.b.f();
                fb.byte(0x45); // i32.eqz
                fb.if_void();
                fb.i32_const(0);
                fb.ret();
                fb.end();
            }
            Callee::Direct(t) => {
                // Not ours: the root runs it, and the rest of this function
                // runs where the guest returns to.
                self.stats.calls_leaving += 1;
                self.b.store_pc(t);
                self.b.f().i32_const(0);
                self.b.f().ret();
                return;
            }
            Callee::Indirect(_) => {
                self.stats.calls_indirect += 1;
                let Linkage::Image { ic } = m.linkage else { unreachable!() };
                let v = op.expect("an indirect call's target");
                let rk = self.locals.rk;
                self.b.f().local_get(0);
                self.b.push_operand(v);
                self.depth_plus(w + cf::CALL_KNOWN_WEIGHT);
                let fb = self.b.f();
                fb.i32_const(need_back);
                fb.i64_const(ret_pc as i64);
                fb.call(ic);
                fb.local_tee(rk);
                fb.i32_const(0);
                fb.byte(0x48); // i32.lt_s
                fb.if_void();
                self.store_pc_value(v);
                self.b.f().i32_const(0);
                self.b.f().ret();
                self.b.f().end();
                let fb = self.b.f();
                fb.local_get(rk);
                fb.byte(0x45); // i32.eqz
                fb.if_void();
                fb.i32_const(0);
                fb.ret();
                fb.end();
            }
        }
        // A summarised callee that returned as expected left its preserved
        // registers as they were: no reload.
        let keep = match callee {
            Callee::Direct(t) if m.bodies.contains_key(&t) && cf::live_writebacks() => match m.sums.get(&t) {
                Some(sm) => sm.preserved.minus(RegSet::of(&[sp_reg()])),
                None => RegSet::EMPTY,
            },
            _ => RegSet::EMPTY,
        };
        // Reload what the continuation may read; the rest keep stale
        // locals, clean, until written (`valid_in` keeps them out of any
        // write-back).
        let reload = match ret {
            Some(r) => after_call(self.used).union(self.rlive[r]).inter(self.used).minus(keep),
            None => RegSet::EMPTY,
        };
        self.b.load(&regs_of(reload));
        self.b.set_dirty(RegSet::EMPTY);
        self.valid = self.valid.inter(keep).union(reload);
        match ret {
            Some(r) => self.edge(i, r),
            None => {
                // Believed not to return, and it did.
                self.b.store_pc(ret_pc);
                self.b.f().i32_const(0);
                self.b.f().ret();
            }
        }
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;

    #[test]
    fn a_function_over_v8s_size_limit_is_left_to_tier1() {
        let max = V8_MAX_FUNCTION_BYTES - (64 << 10);
        assert!(fits(1000, &[], max));
        assert!(fits(max, &[max], max));
        // amd64 python3.13's eval loop: 8.3 MB as one function.
        assert!(!fits(8_300_000, &[], max));
        // A small body whose outlined loop is too big goes too.
        assert!(!fits(1000, &[10, V8_MAX_FUNCTION_BYTES], max));
        assert!(max_body_bytes() <= max);
    }
}
