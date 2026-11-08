//! Phase B: one C function per recovered guest function.
//!
//! Registers and flags are C locals across the whole function: loaded on entry, written back wherever control can leave, reloaded
//! after anything that can change guest state. Nothing stays in a local
//! across a call, so the host's trap-out is a correct unwind with no landing
//! pads. Each function `F` becomes
//!
//! * `f_<F>(ctx)`, its body: a guest `ret`, or any exit to code the function
//!   tier does not own, writes back, sets `ctx::PC` and *returns*;
//! * `e_<F>(ctx)`, its dispatch entry and nest root: `f_<F>(ctx)` and then a
//!   tail call into dispatch on `ctx::PC`.
//!
//! C callers call `f_<G>` and apply the return check: the context's PC must be the return address and SP back where it
//! was; anything else leaves by returning too.

use crate::emit::{self, reg_name, Emitter, FnRegs};
use anyhow::{bail, Result};
use gex_abi::ctx;
use gex_ir::func::{BlockId, Callee, Edge, FBlock, FTerm, Form, Function, RegSet, Summary};
use gex_ir::{Block, Expr, GuestReg, HostFunc, Stmt};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;


/// The compiler barrier at a loop back-edge (`GEX_BACKEDGE`, gex_abi.h):
/// another thread may change what the loop reads. `GEX_CGEN_EXP=NOBARRIER`
/// leaves it out (unsound: pricing only).
fn backedge(e: &mut Emitter<'_>) {
    if !exp("NOBARRIER") {
        e.line("GEX_BACKEDGE();");
    }
}

/// The C test for "`entry`'s code was rewritten" (`GEX_PATCHED`): its page,
/// and the next one when a 5-byte patch at the entry would reach into it.
/// `GEX_CGEN_MUTATE=NOPATCHCHECK`: never.
fn patched_test(entry: u64) -> String {
    if mutate("NOPATCHCHECK") {
        return "0".into();
    }
    if entry % gex_abi::DISPATCH_PAGE > gex_abi::DISPATCH_PAGE - 5 {
        format!("GEX_PATCHED(0x{entry:x}ull) | GEX_PATCHED(0x{:x}ull)", entry + 5)
    } else {
        format!("GEX_PATCHED(0x{entry:x}ull)")
    }
}

/// The export the body table holds for `entry`:
/// its body, checked against `ctx::PC`, since an x86 slot covers four bytes.
pub fn checked_body_name(entry: u64) -> String {
    format!("ib_{entry:x}")
}

pub fn body_name(entry: u64) -> String {
    format!("f_{entry:x}")
}

/// What a call to `t` names: the unit-private copy when `t` is a small
/// leaf (`FnUnit::leaves`), else its body. On the unit's fields, so it is
/// callable while others are borrowed.
fn callee_name(leaves: &HashMap<u64, Function>, copying: bool, in_loop: bool, copies: &mut BTreeSet<u64>, stats: &mut FnStats, t: u64) -> String {
    // A small caller's copy only at a call in an outlined loop function
    // (`in_loop`), where V8 runs out of inlining budget; everywhere
    // it grew one game's unit by 42% (165 -> 234 MiB) for a gain only the loop showed
    // (`GEX_CGEN_EXP=SMALLCALLSALL` restores that; both are opt-in, see
    // `is_small_copyable`). Inside a copy, only a
    // leaf's copy (no chains of calling copies): a small caller's copy keeps
    // its own leaf callee inlinable, as the body it replaces had it
    // (0x70b110's 0x4ac0d0).
    if leaves.get(&t).is_some_and(|f| makes_no_calls(f) || (!copying && (in_loop || exp("SMALLCALLSALL")))) {
        copies.insert(t);
        stats.calls_leaf_copy += 1;
        leaf_name(t)
    } else {
        body_name(t)
    }
}

/// A unit-private copy of a small leaf (`FnUnit::leaves`).
pub fn leaf_name(entry: u64) -> String {
    format!("fl_{entry:x}")
}

/// Whether `f` may be copied into its callers' translation units: a leaf (no calls, no transfers out other than its
/// returns) of at most `LEAF_INSTS` guest instructions. Lifted statements
/// still hold every flag (a compare is about 30), so they are
/// no measure: 0x72b390, 18 instructions with four compares, is 288. The
/// flags go when the copy is inlined, the caller's `need` folding the
/// return's flag store away.
/// A function none of whose blocks ends in a call.
fn makes_no_calls(f: &Function) -> bool {
    !f.blocks.iter().any(|b| matches!(b.term, FTerm::Call { .. }))
}

pub fn is_small_copyable(f: &Function) -> bool {
    const LEAF_INSTS: usize = 32;
    // Also a small function making at most two calls that return to it
    // (a load and a virtual call, say): V8 never inlines those into an
    // outlined loop (726 wire bytes each, and a 26 KB loop leaves it no
    // inlining budget), where TurboFan
    // would. Opt-in (`GEX_CGEN_EXP=SMALLCALLS`: at calls in
    // outlined loops; `SMALLCALLSALL`: everywhere; not in the unit key): the
    // page A/B runs found 2-3% less
    // time in 0x724580's family and no world-fps change beyond noise, so
    // the default copies leaves only.
    let calls_ok = exp("SMALLCALLS") || exp("SMALLCALLSALL");
    let mut n = 0;
    let mut calls = 0;
    for b in &f.blocks {
        match b.term {
            FTerm::Jump(_) | FTerm::Branch { .. } | FTerm::Return { .. } => {}
            FTerm::Call { ret: Some(_), .. } if calls_ok && calls < 2 => calls += 1,
            _ => return false,
        }
        n += b.insts.len();
        if n > LEAF_INSTS {
            return false;
        }
    }
    f.form == Form::Memory
}

pub fn entry_name(entry: u64) -> String {
    format!("e_{entry:x}")
}

/// The name of the helper that calls a function-tier function by address
/// (one per module): returns 1 if it did, 0 if the address is not one.
pub const CALL_KNOWN: &str = "gex_call_known";

/// The guest's stack pointer as a GIR register.
fn sp_reg() -> GuestReg {
    if gex_abi::X86 {
        GuestReg::X(4)
    } else {
        GuestReg::Sp
    }
}

/// How far a `ret` moves SP relative to the call site: x86-64 pops the
/// return address the call pushed.
/// (x86-64: 8; i386 compatibility mode: 4, see `FnUnit::ret_sp_delta`.)
pub fn ret_sp_delta(class32: bool) -> u64 {
    match (gex_abi::X86, class32) {
        (false, _) => 0,
        (true, false) => 8,
        (true, true) => 4,
    }
}

/// Where a function's `ret` pops more than the return address (i386
/// `ret N`: stdcall callees, struct returns): the terminator's pc and how
/// far it moves SP, for the return check of AOT units and the function JIT,
/// which otherwise sees SP past where the caller expects and leaves. `read`
/// gives a guest byte. Empty off x86.
pub fn ret_pops(functions: &[Function], class32: bool, read: &dyn Fn(u64) -> Option<u8>) -> HashMap<u64, u64> {
    let mut out = HashMap::new();
    if !gex_abi::X86 {
        return out;
    }
    let ptr = if class32 { 4 } else { 8 };
    for f in functions {
        for b in &f.blocks {
            if !matches!(b.term, gex_ir::func::FTerm::Return { .. }) || b.synthetic {
                continue;
            }
            let mut a = b.term_pc;
            // `rep ret` / `bnd ret`; an operand-size prefix (a 16-bit
            // pop) gets no entry, and its check leaves.
            while matches!(read(a), Some(0xf2 | 0xf3)) && a < b.term_pc + 4 {
                a += 1;
            }
            if read(a) == Some(0xc2) {
                if let (Some(lo), Some(hi)) = (read(a + 1), read(a + 2)) {
                    let n = u16::from_le_bytes([lo, hi]) as u64;
                    if n != 0 {
                        out.insert(b.term_pc, ptr + n);
                    }
                }
            }
        }
    }
    out
}

/// The calls in `functions` that read the ZF their caller set: MSVC's
/// stack checks, `cmp esi, esp` (3B F4 or 39 E6) right before `call
/// __chkesp` / `call [__imp___chkesp]` / `call _RTC_CheckEsp`, whose `jne`
/// reports a corrupt stack on a stale ZF (debug builds of games).
/// No ABI passes anything in the flags otherwise (`flags_abi`). By the
/// call's pc. Empty off x86.
pub fn zf_calls(functions: &[Function], read: &dyn Fn(u64) -> Option<u8>) -> HashSet<u64> {
    let mut out = HashSet::new();
    if !gex_abi::X86 {
        return out;
    }
    for f in functions {
        for b in &f.blocks {
            if !matches!(b.term, FTerm::Call { .. }) || b.synthetic || b.insts.len() < 2 {
                continue;
            }
            let prev = b.insts[b.insts.len() - 2];
            if b.insts[b.insts.len() - 1] != b.term_pc || prev + 2 != b.term_pc {
                continue;
            }
            if matches!((read(prev), read(prev + 1)), (Some(0x3b), Some(0xf4)) | (Some(0x39), Some(0xe6))) {
                out.insert(b.term_pc);
            }
        }
    }
    out
}

/// The recursion-depth guard. `depth` is an
/// estimate, in bytes, of the engine stack the C frames below this one in
/// the nest use; a function whose own frame would take it past this budget
/// leaves for its own entry instead of running, which unwinds the nest and
/// restarts it at zero from the dispatch root. `GEX_CGEN_STACK_BUDGET`
/// overrides it, for measurement. Engines: wasmtime's `max_wasm_stack` is
/// 512 KiB, V8's whole stack about 984 KiB.
pub const STACK_BUDGET: u32 = 384 << 10;

pub fn stack_budget() -> u32 {
    std::env::var("GEX_CGEN_STACK_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(STACK_BUDGET)
}

/// A function's frame, estimated from the registers it keeps in locals: the
/// wasm locals clang ends up with measured up to ~2.5x the register bytes
/// (coremark, x86-64), plus the engine's spill slots and frame header.
pub fn frame_weight(regs: &FnRegs) -> u32 {
    let bytes: u32 = regs
        .used
        .iter()
        .map(|r| if r.ty() == gex_ir::Ty::V128 { 16 } else { 8 })
        .sum();
    64 + 3 * bytes
}

/// The `need` bits a caller passes: what the callee's expected return must
/// write back beyond every dirty non-flag register.
const NEED_DEFS: &str = "#define NEED_FLAGS 1u\n#define NEED_ALL 0xffffffffu\n\
    #define LOOP_CHUNK 2\n#define LOOP_EXIT 3\n\n";

/// The flag registers of the guest (AArch64 has no P and A: a liveness set
/// that says they are live, because an exit reads "everything", must not make
/// a caller ask for flags).
pub const FLAG_REGS: &[GuestReg] = if gex_abi::X86 {
    &[GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V, GuestReg::P, GuestReg::A]
} else {
    &[GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V]
};

/// What the indirect-call helper's own frame adds.
pub const CALL_KNOWN_WEIGHT: u32 = 32;

/// Pricing experiments, `GEX_CGEN_EXP=a,b`: variants that are measured
/// before they are built properly. Some are unsound (`NOFLAGWB`) and exist
/// only to price what a sound version could at most win.
pub fn exp(name: &str) -> bool {
    std::env::var("GEX_CGEN_EXP")
        .map(|v| v.split(',').any(|x| x == name))
        .unwrap_or(false)
}

/// Deliberate miscompilations, `GEX_CGEN_MUTATE=a,b`, each disabling one
/// safety condition so a test can show it is needed.
pub fn mutate(name: &str) -> bool {
    std::env::var("GEX_CGEN_MUTATE")
        .map(|v| v.split(',').any(|x| x == name))
        .unwrap_or(false)
}

/// Local loop iterations between safepoint polls (tier-1's `loop_poll`).
pub const LOOP_POLL: i32 = 1024;

/// What the emitted functions do at their exits (`GEX_CGEN_STATS=1`).
#[derive(Default, Debug)]
pub struct FnStats {
    pub functions: usize,
    pub blocks: usize,
    pub calls_known: usize,
    pub calls_indirect: usize,
    /// Known calls whose callee is trap-free (liveness-limited write-back).
    pub calls_trapfree: usize,
    /// Calls with the minimal write-back (`minimal_writeback`).
    pub calls_minimal: usize,
    /// Indirect calls with a guarded direct call to the profile's dominant
    /// target (`FnUnit::icall_hints`).
    pub calls_devirt: usize,
    /// Calls that go to a unit-private copy of a small leaf
    /// (`FnUnit::leaves`).
    pub calls_leaf_copy: usize,
    /// Calls whose small leaf callee was inlined into the caller's GIR
    /// (`FnUnit::inline_leaf_calls`).
    pub calls_inlined: usize,
    pub tail_known: usize,
    pub switches: usize,
    pub loops_outlined: usize,
    pub unresolved: usize,
    pub returns: usize,
    /// Direct calls and tail calls that leave the function tier, by target.
    pub leaves: std::collections::BTreeMap<u64, usize>,
}

impl FnStats {
    pub fn merge(&mut self, o: &FnStats) {
        self.functions += o.functions;
        self.blocks += o.blocks;
        self.calls_known += o.calls_known;
        self.calls_indirect += o.calls_indirect;
        self.calls_trapfree += o.calls_trapfree;
        self.calls_minimal += o.calls_minimal;
        self.calls_devirt += o.calls_devirt;
        self.calls_leaf_copy += o.calls_leaf_copy;
        self.calls_inlined += o.calls_inlined;
        self.tail_known += o.tail_known;
        self.switches += o.switches;
        self.loops_outlined += o.loops_outlined;
        self.unresolved += o.unresolved;
        self.returns += o.returns;
        for (a, n) in &o.leaves {
            *self.leaves.entry(*a).or_default() += n;
        }
    }
}

/// C for one translation unit of functions.
pub struct FnUnit {
    pub stats: FnStats,
    protos: BTreeSet<u64>,
    bodies: String,
    /// Whether any function here calls through `CALL_KNOWN`.
    pub uses_call_known: bool,
    /// The frame weight of the function being emitted (`frame_weight`).
    weight: u32,
    /// How far a `ret` moves SP: 8 on x86-64, 4 for an i386 program.
    pub ret_sp_delta: u64,
    /// Where a `ret` moves it further ([`ret_pops`], by the terminator's
    /// pc): i386 `ret N`. Whoever adds a function fills in its entries.
    pub ret_pops: HashMap<u64, u64>,
    /// Registers live on entry to each block of the function being emitted
    /// (`live_in`): write-backs on internal edges and at calls to functions
    /// with a summary store only these.
    live: Vec<RegSet>,
    /// A unit for `standalone::standalone`: everything `static`, and no
    /// dispatch roots (`e_<F>`), so nothing imports `gexc.dispatch`.
    standalone: bool,
    /// A whole-image unit's: calls to anything
    /// the unit does not hold go through the process's function table, and
    /// there are no dispatch roots or body-table entries.
    pub ftab: bool,
    /// The functions with a body here, in order (a whole-image unit's).
    defined: Vec<u64>,
    /// The body being emitted has a `goto gex_resume` (`resume_block`).
    resumes: bool,
    /// An i386 program (`set_class32`).
    class32: bool,
    /// The registers each function's call tree names (`touched`), from the
    /// recovery; a callee missing here touches everything.
    pub touched: HashMap<u64, RegSet>,
    /// For the function being emitted: per block ending in a call with the
    /// minimal write-back, the live registers its callee never touches,
    /// which stay dirty in locals across it (`carried`).
    carry: HashMap<usize, RegSet>,
    /// The functions that need no depth guard (`guard_free`), from the
    /// recovery.
    pub guardfree: HashSet<u64>,
    /// Profile-guided devirtualisation: an indirect call whose return address is
    /// here gets a guarded direct call to the target, which skips the probe
    /// and lets clang inline it. By return address.
    pub icall_hints: std::sync::Arc<HashMap<u64, u64>>,
    /// Small leaf functions of the image:
    /// a call to one goes to a private `static inline` copy in this
    /// translation unit, which clang can inline across the image's units.
    pub leaves: std::sync::Arc<HashMap<u64, Function>>,
    /// The copies this unit's calls need (`leaves`), emitted by
    /// `add_leaf_copies`.
    leaf_copies: BTreeSet<u64>,
    /// Emitting a copy: `static inline`, under `leaf_name`.
    copying: bool,
    /// Calls that read the ZF their caller set ([`zf_calls`]), by the
    /// call's pc. Whoever adds a function fills in its entries.
    pub zf_calls: HashSet<u64>,
    /// For the function being emitted (`inline_leaf_calls`): each block
    /// an inlined call made (the leaf's copied blocks and the call's
    /// fallback), mapped to the block of the call it replaced.
    inlined: std::sync::Arc<HashMap<usize, usize>>,
    /// For the function being emitted: the fallback calls of inlined
    /// indirect calls (the target was not the hinted one), which are not
    /// devirtualised again.
    no_devirt: HashSet<usize>,
    /// Inline calls to small call-free leaves (`inline_leaves`).
    pub inline_leaves: bool,
}

impl Default for FnUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl FnUnit {
    /// For an i386 (compatibility-mode) program.
    pub fn set_class32(&mut self, class32: bool) {
        self.ret_sp_delta = ret_sp_delta(class32);
        self.class32 = class32;
    }

    /// The calls of `f` that get the minimal write-back, by block: a call to
    /// a function of this unit whose summary is complete and that cannot
    /// reach the kernel, unwind or leave pending, and is not trap-free
    /// (those already have exact write-backs). The value is what the
    /// caller keeps dirty across it: the continuation's live registers its
    /// call tree never names, PC and SP excepted.
    fn carried(&self, f: &Function, known: &HashSet<u64>, sums: &HashMap<u64, Summary>, trapfree: &HashMap<u64, u32>) -> HashMap<usize, RegSet> {
        let mut out = HashMap::new();
        if !self.ftab || !minimal_writeback() {
            return out;
        }
        for (i, b) in f.blocks.iter().enumerate() {
            // A direct call to a function of this unit, or an indirect one
            // devirtualised to one (`icall_hints`: its fallback writes the
            // carried registers back and reloads them). A trap-free callee
            // too: what it never touches need not be stored for it either.
            let (t, r) = match &b.term {
                FTerm::Call { callee: Callee::Direct(t), ret: Some(r), .. } => (t, r),
                FTerm::Call { callee: Callee::Indirect(_), ret: Some(r), ret_pc } if devirt_carry() && !self.no_devirt.contains(&i) => {
                    match self.icall_hints.get(ret_pc).filter(|k| known.contains(*k)) {
                        Some(k) => (k, r),
                        None => continue,
                    }
                }
                _ => continue,
            };
            if !known.contains(t) || (trapfree.contains_key(t) && !trapfree_carry()) || mutate("IGNOREUNWIND") {
                continue;
            }
            let Some(sm) = sums.get(t) else { continue };
            let Some(tt) = self.touched.get(t) else { continue };
            if sm.may_unwind || sm.may_leave_pending || sm.unknown_callee || !wb_bisect() {
                continue;
            }
            let untouched = RegSet::ALL.minus(*tt);
            let kept = untouched.inter(sm.preserved).minus(RegSet::of(&[sp_reg(), GuestReg::Pc, GuestReg::CtxAddr]));
            let k = WB_K.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if std::env::var("GEX_CGEN_WB_SHOW").ok().and_then(|v| v.parse::<u64>().ok()) == Some(k) {
                eprintln!("[wb] site {k}: {:#x} calls {t:#x}: keeps {:?}", f.entry, self.live[r.to.index()].inter(kept));
            }
            out.insert(i, self.live[r.to.index()].inter(kept));
        }
        out
    }

    pub fn new() -> FnUnit {
        FnUnit {
            stats: FnStats::default(),
            protos: BTreeSet::new(),
            bodies: String::new(),
            uses_call_known: false,
            weight: 0,
            ret_sp_delta: ret_sp_delta(false),
            ret_pops: HashMap::new(),
            live: Vec::new(),
            standalone: false,
            ftab: false,
            defined: Vec::new(),
            resumes: false,
            class32: false,
            touched: HashMap::new(),
            carry: HashMap::new(),
            guardfree: HashSet::new(),
            icall_hints: Default::default(),
            leaves: Default::default(),
            leaf_copies: BTreeSet::new(),
            copying: false,
            zf_calls: HashSet::new(),
            inlined: Default::default(),
            no_devirt: HashSet::new(),
            inline_leaves: inline_leaves(),
        }
    }

    /// The copies of small leaves this unit's calls named ([`callee_name`]),
    /// as `static inline` functions after its bodies.
    pub fn add_leaf_copies(&mut self, known: &HashSet<u64>, sums: &HashMap<u64, Summary>, trapfree: &HashMap<u64, u32>) -> Result<()> {
        let leaves = std::sync::Arc::clone(&self.leaves);
        let saved = std::mem::take(&mut self.stats);
        self.copying = true;
        let mut r = Ok(());
        // A worklist: a calling copy may want a leaf's copy (`callee_name`).
        let mut done = BTreeSet::new();
        'all: loop {
            let pending: Vec<u64> = self.leaf_copies.difference(&done).copied().collect();
            if pending.is_empty() {
                break;
            }
            for a in pending {
                done.insert(a);
                if let Some(f) = leaves.get(&a) {
                    if let Err(err) = self.add(f, known, sums, trapfree) {
                        r = Err(err);
                        break 'all;
                    }
                }
            }
        }
        self.copying = false;
        self.stats = saved;
        r
    }

    /// A unit for a standalone translation unit (`crate::standalone`).
    pub fn new_standalone() -> FnUnit {
        FnUnit {
            standalone: true,
            ..FnUnit::new()
        }
    }

    fn storage(&self) -> &'static str {
        if self.copying {
            "static inline "
        } else if self.standalone {
            "static "
        } else {
            ""
        }
    }

    /// Add `f`. `known` is every function entry with a C body in this module;
    /// `sums` are S2's call-graph summaries, by entry (may be empty).
    ///
    /// `trapfree` maps the trap-free functions (`trap_free`) to their
    /// subtree's stack estimate.
    /// A body for `entry` that only leaves for it (PC = entry, return 0), so
    /// the nest unwinds and tier-1 runs the function: for a function whose C
    /// cannot be spliced (clang gave it a shadow stack). Its callers wrote
    /// everything back before the call, as for any function that is not
    /// trap-free, so it must not be one.
    pub fn add_leave_stub(&mut self, entry: u64) {
        let st = self.storage();
        if self.ftab {
            // Re-entered through the table: the host serves the address
            // with something other than this unit (a provider, or tier-1
            // for a body clang could not compile). The probe cache first, so
            // a served address costs one indirect call, not a host lookup.
            let _ = writeln!(
                self.bodies,
                "{st}int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret) {{\n  \
                 GEX_C64({}u) = 0x{entry:x}ull;\n  \
                 GEX_MUSTTAIL return GEX_FTAB(gex_ftab_idx(0x{entry:x}ull))(ctx, depth, need, ret);\n}}\n",
                body_name(entry),
                ctx::LAST_TARGET
            );
            self.protos.insert(entry);
            return;
        }
        let _ = writeln!(
            self.bodies,
            "{st}int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret) {{\n  \
             (void)depth; (void)need; (void)ret;\n  GEX_C64({}u) = 0x{entry:x}ull;\n  return 0;\n}}\n",
            body_name(entry),
            ctx::PC
        );
        self.protos.insert(entry);
    }

    /// A body for `entry` that an export provider compiled into the same
    /// module serves (a link group's provider): a
    /// tail call of the provider's C function `sym`, an F_body. In the
    /// element segment as any body, so direct calls and the table reach it.
    pub fn add_provider_body(&mut self, entry: u64, sym: &str) {
        let st = self.storage();
        let _ = writeln!(
            self.bodies,
            "int {sym}(int ctx, int32_t depth, uint32_t need, uint64_t ret);\n\
             {st}int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret) {{\n  \
             GEX_MUSTTAIL return {sym}(ctx, depth, need, ret);\n}}\n",
            body_name(entry)
        );
        self.protos.insert(entry);
        if self.ftab {
            self.defined.push(entry);
        }
    }

    /// `f` with its calls to small call-free leaves (`inlinable`) inlined in
    /// its GIR: the call's block branches on "the leaf is the target (a
    /// devirtualised indirect call's hint) and its code was not rewritten"
    /// (`GEX_PATCHED`, as `patched_test`) to a copy of the leaf's blocks,
    /// else to a synthetic block with the original call. The copy's `ret`
    /// becomes a one-case switch: the return address the call pushed goes
    /// on at the call's continuation, anything else through the function
    /// table, as an unresolved jump would. The registers stay in the
    /// caller's locals across the leaf: no write-back, no reload, no call.
    ///
    /// A leaf copy called through `fl_` (`callee_name`) is a separate C
    /// function to clang: the caller writes back what the leaf reads, the
    /// leaf what the caller reloads, and clang cannot forward any of it
    /// past a guest store (both are linear memory to it). Worker
    /// functions call such leaves (seven virtual getters, say).
    ///
    /// The copies are synthetic blocks whose `start` is the call's
    /// instruction, so the address heuristics (a back-edge polls) see them
    /// as part of it; `inlined` maps them to the call's block. i386 only:
    /// the fallback of an indirect call keeps its target in `MonAddr`,
    /// which x86 code never names, a local that is never written back
    /// (`emit::is_scratch`).
    fn inline_leaf_calls(&mut self, f: &Function) -> Function {
        use gex_ir::func::JumpTable;
        use gex_ir::{BinOp, CmpOp, MemSz, Order, Ty, Value};
        let mut g = f.clone();
        let mut inlined = HashMap::new();
        // The shared mismatch exits of inlined `ret`s, by how far they pop.
        let mut mismatch: HashMap<u64, usize> = HashMap::new();
        self.no_devirt.clear();
        let n0 = f.blocks.len();
        for i in 0..n0 {
            let (callee, ret_pc, cont) = match &g.blocks[i].term {
                FTerm::Call { callee, ret_pc, ret: Some(r) } => (*callee, *ret_pc, r.clone()),
                _ => continue,
            };
            let (t, target) = match callee {
                Callee::Direct(t) => (t, None),
                Callee::Indirect(v) => match self.icall_hints.get(&ret_pc) {
                    Some(&k) => (k, Some(v)),
                    None => continue,
                },
            };
            let leaves = std::sync::Arc::clone(&self.leaves);
            let Some(leaf) = leaves.get(&t).filter(|l| inlinable(l)) else { continue };
            self.stats.calls_inlined += 1;
            let pc = g.blocks[i].term_pc;
            let mut nv = g.n_values;
            let mut def = |b: &mut FBlock, expr: Expr| {
                let v = Value(nv);
                nv += 1;
                b.push(pc, Stmt::Def { dst: v, expr });
                v
            };
            // `patched_test`: the page's bit, and the next page's when a
            // 5-byte patch at the entry reaches into it.
            let mut pages = vec![t];
            if t % gex_abi::DISPATCH_PAGE > gex_abi::DISPATCH_PAGE - 5 {
                pages.push(t + 5);
            }
            let cb = &mut g.blocks[i];
            let mut patched = None;
            for a in pages {
                let at = def(cb, Expr::ConstI64(gex_abi::patched_pages() + (a >> 15)));
                let byte = def(cb, Expr::Load { addr: at, offset: 0, sz: MemSz::B1, signed: false, ty: Ty::I32, order: Order::Relaxed });
                let sh = def(cb, Expr::ConstI32(((a >> 12) & 7) as u32));
                let bits = def(cb, Expr::Bin { op: BinOp::ShrU, ty: Ty::I32, a: byte, b: sh });
                let one = def(cb, Expr::ConstI32(1));
                let bit = def(cb, Expr::Bin { op: BinOp::And, ty: Ty::I32, a: bits, b: one });
                patched = Some(match patched {
                    None => bit,
                    Some(p) => def(cb, Expr::Bin { op: BinOp::Or, ty: Ty::I32, a: p, b: bit }),
                });
            }
            let patched = patched.expect("one page at least");
            let zero = def(cb, Expr::ConstI32(0));
            let mut cond = def(cb, Expr::Cmp { op: CmpOp::Eq, ty: Ty::I32, a: patched, b: zero });
            if let Some(v) = target {
                let k = def(cb, Expr::ConstI64(t));
                let hit = def(cb, Expr::Cmp { op: CmpOp::Eq, ty: Ty::I64, a: v, b: k });
                cond = def(cb, Expr::Bin { op: BinOp::And, ty: Ty::I32, a: hit, b: cond });
                cb.push(pc, Stmt::WriteReg { reg: GuestReg::MonAddr, val: v });
            }
            // The fallback: the call as it was.
            let fb = g.blocks.len();
            let mut slow = FBlock::synthetic(pc, FTerm::Illegal { raw: 0, pc });
            let callee = match target {
                Some(_) => {
                    let v = def(&mut slow, Expr::ReadReg(GuestReg::MonAddr));
                    self.no_devirt.insert(fb);
                    Callee::Indirect(v)
                }
                None => Callee::Direct(t),
            };
            slow.term = FTerm::Call { callee, ret_pc, ret: Some(cont.clone()) };
            inlined.insert(fb, i);
            g.blocks.push(slow);
            // The leaf's blocks, its values renumbered after ours.
            let base = g.blocks.len();
            let off = nv;
            let mut rets = Vec::new();
            for lb in &leaf.blocks {
                let mut b = lb.clone();
                b.synthetic = true;
                b.start = pc;
                for s in &mut b.stmts {
                    gex_ir::func::for_each_def_mut(s, |v| v.0 += off);
                    gex_ir::func::for_each_use_mut(s, |v| v.0 += off);
                }
                b.term.for_each_use_mut(|v| v.0 += off);
                for e in b.term.edges_mut() {
                    e.to = BlockId(e.to.0 + base as u32);
                }
                if let FTerm::Return { target } = b.term {
                    rets.push((g.blocks.len(), target, ret_delta(lb, &self.ret_pops)));
                    b.term = FTerm::Switch {
                        target,
                        cases: vec![(ret_pc, cont.clone())],
                        table: JumpTable { base: 0, entry_bytes: 4, count: 1, index: None, complete: false },
                    };
                }
                inlined.insert(g.blocks.len(), i);
                g.blocks.push(b);
            }
            nv += leaf.n_values;
            // A `ret` whose pop is known (`ret_delta`): its mismatch (the
            // leaf rewrote its return address) goes to the function's one
            // exit for that pop, which reads the address back from the
            // stack, instead of a write-back and table call of its own
            // (some 90 wasm instructions each: 70% of what inlining added
            // to 0x70bdc0). Guest state only, so a loop function leaving
            // for it loses nothing.
            if shared_ret_exits() {
                for (bi, target, delta) in rets {
                    let Some(delta) = delta else { continue };
                    let mm = *mismatch.entry(delta).or_insert_with(|| {
                        let mut m = FBlock::synthetic(pc, FTerm::Illegal { raw: 0, pc });
                        let mut d = |m: &mut FBlock, expr: Expr| {
                            let v = Value(nv);
                            nv += 1;
                            m.push(pc, Stmt::Def { dst: v, expr });
                            v
                        };
                        let sp = d(&mut m, Expr::ReadReg(sp_reg()));
                        let k = d(&mut m, Expr::ConstI64(delta));
                        let at = d(&mut m, Expr::Bin { op: BinOp::Sub, ty: Ty::I64, a: sp, b: k });
                        let w = d(&mut m, Expr::Load { addr: at, offset: 0, sz: MemSz::B4, signed: false, ty: Ty::I32, order: Order::Relaxed });
                        let t = d(&mut m, Expr::Extend { a: w, signed: false });
                        m.term = FTerm::Unresolved { target: t };
                        inlined.insert(g.blocks.len(), i);
                        g.blocks.push(m);
                        g.blocks.len() - 1
                    });
                    let b = &mut g.blocks[bi];
                    let tpc = b.term_pc;
                    let k = Value(nv);
                    b.push(tpc, Stmt::Def { dst: k, expr: Expr::ConstI64(ret_pc) });
                    let hit = Value(nv + 1);
                    b.push(tpc, Stmt::Def { dst: hit, expr: Expr::Cmp { op: CmpOp::Eq, ty: Ty::I64, a: target, b: k } });
                    nv += 2;
                    b.term = FTerm::Branch { cond: hit, taken: cont.clone(), not_taken: Edge::to(BlockId(mm as u32)) };
                }
            }
            g.n_values = nv;
            g.blocks[i].term = FTerm::Branch {
                cond,
                taken: Edge::to(BlockId(base as u32)),
                not_taken: Edge::to(BlockId(fb as u32)),
            };
        }
        self.inlined = std::sync::Arc::new(inlined);
        g
    }

    pub fn add(
        &mut self,
        f: &Function,
        known: &HashSet<u64>,
        sums: &HashMap<u64, Summary>,
        trapfree: &HashMap<u64, u32>,
    ) -> Result<()> {
        if f.form != Form::Memory {
            bail!("function {:#x}: only memory form is supported", f.entry);
        }
        let with_inlined;
        let f = if !self.copying && self.ftab && gex_abi::X86 && self.class32 && self.inline_leaves && !self.leaves.is_empty() {
            with_inlined = self.inline_leaf_calls(f);
            &with_inlined
        } else {
            self.inlined = Default::default();
            self.no_devirt.clear();
            f
        };
        let dummy = Block::new(f.entry);
        let owned = HashSet::new();
        let mut e = Emitter::new(&dummy, &owned);
        e.class32 = self.class32;
        if self.ftab {
            e.fault_exit = Some("goto gex_resume;");
        }
        e.resize(f.n_values as usize);
        let mut regs = registers(f);
        regs.clean_spills = clean_spills();
        self.live = if live_writebacks() { live_in_with(f, sums, &self.zf_calls) } else { vec![RegSet::ALL; f.blocks.len()] };
        self.carry = if regs.clean_spills { self.carried(f, known, sums, trapfree) } else { HashMap::new() };
        let dirty = if regs.clean_spills { dirty_in(f, &self.carry) } else { Vec::new() };
        let used: Vec<GuestReg> = regs.used.iter().copied().collect();
        self.weight = frame_weight(&regs);
        e.func_regs = Some(regs);
        let mut calls: BTreeSet<u64> = BTreeSet::new();

        self.stats.functions += 1;
        self.stats.blocks += f.blocks.iter().filter(|b| !b.synthetic).count();
        let name = if self.copying { leaf_name(f.entry) } else { body_name(f.entry) };
        // The body's head, kept aside: its loop functions go first.
        let mut fhead = String::new();
        // A calling copy is too costly for clang's threshold at a call site
        // of many (0x70b110 in 0x724580's loop): forced, as V8 would not
        // inline it either.
        let force = if self.copying && !makes_no_calls(f) { "__attribute__((always_inline)) " } else { "" };
        let _ = writeln!(fhead, "{force}{}int {name}(int ctx, int32_t depth, uint32_t need, uint64_t ret) {{", self.storage());
        for r in &used {
            let _ = writeln!(fhead, "  {} {};", emit::reg_cty(*r, self.class32), reg_name(*r));
        }
        let _ = writeln!(fhead, "  int32_t poll = {LOOP_POLL};");
        // Set when a poll expires anywhere: the next outermost back-edge
        // hands the loop over, wherever the hot inner loop is.
        let _ = writeln!(fhead, "  int hot = 0; (void)hot;");
        // Its code rewritten since the unit was installed (Detours): what
        // the table now has for the entry runs instead, under the same
        // contract (a newer translation, which the probe finds once the host
        // has made it). Every caller of a function that is not trap-free
        // wrote everything back first, so nothing is lost; trap-free ones
        // are checked by their callers.
        if self.ftab && !trapfree.contains_key(&f.entry) {
            let _ = writeln!(
                fhead,
                "  if (__builtin_expect({}, 0)) {{ GEX_C64({}u) = 0x{:x}ull; \
                 GEX_MUSTTAIL return GEX_FTAB(gex_ftab_idx(0x{:x}ull))(ctx, depth, need, ret); }}",
                patched_test(f.entry),
                ctx::LAST_TARGET,
                f.entry,
                f.entry
            );
        }
        // The recursion-depth guard, before anything is loaded: every caller
        // wrote everything back before calling, so leaving for this
        // function's own entry here is exactly the caller leaving for it.
        // A trap-free function has none (its caller may not have written
        // everything back, and it must not leave): its whole subtree fits in
        // the reserve every other function keeps below the budget.
        if !trapfree.contains_key(&f.entry) && !(self.guardfree.contains(&f.entry) && !exp("ALLGUARDS")) {
            let _ = writeln!(
                fhead,
                "  if (__builtin_expect(depth > {}, 0)) {{ GEX_C64({}u) = 0x{:x}ull; return 0; }}",
                stack_budget().saturating_sub(self.weight + TRAPFREE_RESERVE),
                ctx::PC,
                f.entry
            );
        }
        e.load_regs(&used);
        // The stack pointer on entry: a return is the one the caller
        // expects when it goes to `ret` with SP back here (plus the popped
        // return address on x86-64).
        e.line(format!("const uint64_t sp0 = {};", reg_name(sp_reg())));
        // Outermost loops become functions of their own (`outer_loops`),
        // emitted before this one.
        let loops = if outline_loops() { outer_loops(f) } else { Vec::new() };
        for (k, lp) in loops.iter().enumerate() {
            self.stats.loops_outlined += 1;
            let lname = loop_name(f.entry, k, self.copying);
            let mut le = Emitter::new(&dummy, &owned);
            le.class32 = self.class32;
            if self.ftab {
                le.fault_exit = Some("return 0;");
            }
            le.resize(f.n_values as usize);
            // Only the registers the loop itself touches are locals here:
            // everything else is already in the context (the body wrote
            // everything back before the call) and stays there, which
            // keeps the loop's register pressure to its own.
            let in_loop = |i: usize| lp.body.contains(&i);
            let mut lregs = registers_in(f, &in_loop);
            lregs.clean_spills = clean_spills();
            let lused: Vec<GuestReg> = lregs.used.iter().copied().collect();
            let ldirty = if lregs.clean_spills {
                dirty_in_from(f, lp.header, &in_loop, &self.carry)
            } else {
                Vec::new()
            };
            le.func_regs = Some(lregs);
            let mut head = String::new();
            let _ = writeln!(
                head,
                "static __attribute__((noinline)) int {lname}(int ctx, int32_t depth, uint32_t need, uint64_t ret, uint64_t sp0) {{"
            );
            for r in &lused {
                let _ = writeln!(head, "  {} {};", emit::reg_cty(*r, self.class32), reg_name(*r));
            }
            let _ = writeln!(head, "  int32_t poll = {LOOP_POLL};");
            let _ = writeln!(head, "  int hot = 0; (void)hot;");
            le.load_regs(&lused);
            le.line(format!("goto L{};", lp.header));
            let scope = Scope { body: &lp.body, header: lp.header, nested: lp.nested };
            for (i, bl) in f.blocks.iter().enumerate() {
                if !lp.body.contains(&i) {
                    continue;
                }
                le.line(format!("L{i}: {{"));
                if let Some(d) = ldirty.get(i) {
                    le.set_dirty(d);
                }
                block_stmts(&mut le, bl)?;
                self.term(&mut le, f, i, known, sums, trapfree, &mut calls, Some(&scope), None)?;
                le.line("}");
            }
            // Every path above ends in a goto or a return.
            le.line("__builtin_unreachable();");
            self.bodies.push_str(&head);
            self.bodies.push_str(&le.take_output());
            self.bodies.push_str("}\n\n");
        }

        self.bodies.push_str(&fhead);
        // Block 0 is the entry and comes first, so control falls into it.
        for (i, b) in f.blocks.iter().enumerate() {
            e.line(format!("L{i}: {{"));
            if let Some(d) = dirty.get(i) {
                e.set_dirty(d);
            }
            block_stmts(&mut e, b)?;
            let heads = Heads { loops: &loops, fname: f.entry, copy: self.copying, used: &used };
            self.term(&mut e, f, i, known, sums, trapfree, &mut calls, None, Some(&heads))?;
            e.line("}");
        }
        // Always there in a whole-image unit's body (clang drops it when
        // nothing goes there).
        self.resumes = false;
        if self.ftab {
            resume_block(&mut e, f, &self.inlined);
        }
        self.bodies.push_str(&e.take_output());
        self.bodies.push_str("}\n\n");

        // The entry and nest root. A standalone unit has none: its caller
        // dispatches on `ctx::PC` itself; nor has a whole-image unit, whose
        // functions are reached through the function table.
        if self.copying {
            // A copy that calls (`is_small_copyable`) needs its callees
            // declared like any body here.
            self.protos.extend(calls);
            return Ok(());
        }
        if self.ftab {
            self.defined.push(f.entry);
        }
        if self.standalone || self.ftab {
            self.protos.insert(f.entry);
            self.protos.extend(calls);
            return Ok(());
        }
        let _ = writeln!(self.bodies, "void {}(int ctx) {{", entry_name(f.entry));
        let root_block = Block::new(f.entry);
        let mut r = Emitter::new(&root_block, &owned);
        r.class32 = self.class32;
        if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
            r.target_check();
        }
        // No caller to predict: `ret` is an address no code has, so a
        // return writes everything back and sets PC; should the guest really
        // return there, PC says so and dispatch faults as tier-1's would.
        r.line(format!("if ({name}(ctx, 0, NEED_ALL, ~0ull)) GEX_C64({}u) = ~0ull;", ctx::PC));
        r.line("if (__builtin_expect(((*(volatile uint64_t *)GEX_P(GEX_NEST_EXITS))++ & 4095u) == 0, 0))");
        r.line(format!(
            "  gex_cache_op(ctx, 0x{:x}ll, GEX_CACHE_OP_NEST_EXIT_SAMPLE, (int64_t)GEX_C64({}u));",
            f.entry,
            ctx::PC
        ));
        r.line(format!("const uint64_t tgt = GEX_C64({}u);", ctx::PC));
        r.safepoint("tgt");
        r.dispatch("tgt");
        self.bodies.push_str(&r.take_output());
        self.bodies.push_str("}\n\n");
        // The body-table entry: `gex_call_known` set PC to the target, and a
        // slot covers four bytes on x86, so a neighbour's call leaves (-1).
        let _ = writeln!(
            self.bodies,
            "int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret) {{\n  \
             if (GEX_C64({}u) != 0x{:x}ull) return -1;\n  \
             GEX_MUSTTAIL return {name}(ctx, depth, need, ret);\n}}\n",
            checked_body_name(f.entry),
            ctx::PC,
            f.entry
        );
        self.protos.insert(f.entry);
        self.protos.extend(calls);
        Ok(())
    }

    /// Emit the terminator of block `i`.
    fn term(
        &mut self,
        e: &mut Emitter<'_>,
        f: &Function,
        i: usize,
        known: &HashSet<u64>,
        sums: &HashMap<u64, Summary>,
        trapfree: &HashMap<u64, u32>,
        calls: &mut BTreeSet<u64>,
        scope: Option<&Scope<'_>>,
        heads: Option<&Heads<'_>>,
    ) -> Result<()> {
        let b = &f.blocks[i];
        let live = &self.live;
        let self_weight = self.weight;
        // A whole-image unit's body (not a loop function) catches what its
        // callees leave for (`resume_block`): where they fail, it goes there.
        let ftab = self.ftab;
        let resumed = std::cell::Cell::new(false);
        let fail = if ftab && scope.is_none() { "goto gex_resume;" } else { "return 0;" };
        let inlined = std::sync::Arc::clone(&self.inlined);
        let goto = |e: &mut Emitter<'_>, edge: &Edge| {
            if !edge.args.is_empty() {
                unreachable!("memory form has no edge arguments");
            }
            let to = &f.blocks[edge.to.index()];
            if let Some(h) = heads {
                let t = edge.to.index();
                if let Some((k, lp)) = h.loops.iter().enumerate().find(|(_, lp)| lp.header == t) {
                    if lp.body.contains(&i) && !to.synthetic && to.start <= b.start {
                        // An outermost loop's back-edge in the body itself: the
                        // first 1024 trips run here; a loop that keeps going
                        // moves to its loop function, a fresh call per chunk,
                        // and the body resumes where it leaves.
                        // A poll that expired anywhere in the loop (an inner
                        // loop's, usually) set `hot`; this edge's own counts
                        // too.
                        e.line("if (--poll <= 0) {");
                        e.line("  poll = 1024;");
                        e.flush_keep();
                        e.store_pc(&format!("0x{:x}ull", to.start));
                        e.line(format!("  gex_safepoint(ctx, 0x{:x}ll);", to.start));
                        if lp.nested {
                            e.line("  hot = 1;");
                            e.line("}");
                            e.line("if (hot) {");
                            e.flush_keep();
                        }
                        e.line("  int lr;");
                        e.line(format!(
                            "  do lr = {}(ctx, depth + {}, need, ret, sp0); while (lr == LOOP_CHUNK);",
                            loop_name(h.fname, k, h.copy),
                            self_weight
                        ));
                        if ftab {
                            resumed.set(true);
                            e.line("  if (lr == 0) goto gex_resume;");
                            e.line("  if (lr == 1) return 1;");
                        } else {
                            e.line("  if (lr <= 1) return lr;");
                        }
                        e.load_regs(h.used);
                        e.line("  switch (lr) {");
                        for x in &lp.exits {
                            e.line(format!("  case LOOP_EXIT + {x}: goto L{x};"));
                        }
                        e.line("  }");
                        e.line("  __builtin_unreachable();");
                        e.line("}");
                        backedge(e);
                        e.line(format!("goto L{t};"));
                        return;
                    }
                }
            }
            if let Some(sc) = scope {
                let t = edge.to.index();
                if !sc.body.contains(&t) {
                    // Leaving the loop: everything back to the context, and
                    // the function carries on at `t`.
                    e.flush_keep();
                    e.line(format!("return LOOP_EXIT + {t};"));
                    return;
                }
                if t == sc.header && !to.synthetic && to.start <= b.start {
                    // The loop's own back-edge: a chunk ends here when the
                    // countdown expires (after the safepoint, as ever).
                    e.line("if (--poll <= 0) {");
                    e.flush_keep();
                    e.store_pc(&format!("0x{:x}ull", to.start));
                    e.line(format!("  gex_safepoint(ctx, 0x{:x}ll);", to.start));
                    e.line("  return LOOP_CHUNK;");
                    e.line("}");
                    if sc.nested {
                        // An inner loop's poll expired during this trip: end
                        // the chunk here too, so a loop whose trips are long
                        // still gets fresh calls.
                        e.line("if (hot) {");
                        e.flush_keep();
                        e.line("  return LOOP_CHUNK;");
                        e.line("}");
                    }
                    backedge(e);
                    e.line(format!("goto L{t};"));
                    return;
                }
            }
            // tier-1's rule for a backward branch: poll. Here the countdown is
            // a local (tier-1's fused loops do the same) and the slow path
            // writes everything back, since the host may deliver a signal.
            // (`GEX_CGEN_EXP=NOPOLL` drops inner polls: unsound, pricing only.)
            if !to.synthetic && to.start <= b.start && !exp("NOPOLL") {
                e.line("if (--poll <= 0) {");
                e.line("  poll = 1024;");
                // Every dirty register, as tier-1 has them at a safepoint;
                // the flags only when the loop head may read one (tier-1's
                // dead-flag elimination leaves dead flags stale too).
                e.flush_keep_live(non_flags().union(live[edge.to.index()]));
                e.store_pc(&format!("0x{:x}ull", to.start));
                e.line(format!("  gex_safepoint(ctx, 0x{:x}ll);", to.start));
                // An inner loop of a nested outermost loop marks it hot.
                let in_nested = scope.is_some_and(|sc| sc.nested)
                    || heads.is_some_and(|h| h.loops.iter().any(|lp| lp.nested && lp.body.contains(&i)));
                if in_nested {
                    e.line("  hot = 1;");
                }
                e.line("}");
            }
            // An inlined leaf's blocks (`inline_leaf_calls`) count as its
            // call's: no barrier inside the copy (it has no loop) or on the
            // way in, and on the way out one where the call's edge to its
            // continuation would have one.
            let backward = match (inlined.get(&i), inlined.contains_key(&edge.to.index())) {
                (_, true) => false,
                (Some(&c), false) => edge.to.index() <= c || to.start <= f.blocks[c].start,
                (None, false) => edge.to.index() <= i || to.start <= b.start,
            };
            if backward {
                backedge(e);
            }
            e.line(format!("goto L{};", edge.to.index()));
        };
        // Write back, set PC, return: the nest root dispatches from there.
        let leave = |e: &mut Emitter<'_>, pc: &str| {
            e.flush();
            e.store_pc(pc);
            e.line("return 0;");
        };
        match &b.term {
            FTerm::Jump(edge) => goto(e, edge),
            FTerm::Branch {
                cond,
                taken,
                not_taken,
            } => {
                e.line(format!("if ({}) {{", e.v(*cond)));
                goto(e, taken);
                e.line("}");
                goto(e, not_taken);
            }
            FTerm::Switch { target, cases, .. } => {
                self.stats.switches += 1;
                let t = e.v(*target);
                e.line(format!("switch ({t}) {{"));
                for (addr, edge) in cases {
                    e.line(format!("case 0x{addr:x}ull:"));
                    goto(e, edge);
                }
                e.line("default:;");
                e.line("}");
                if self.ftab && local_resume() {
                    // Anywhere else: on through the function table under
                    // this function's return contract (as an unresolved
                    // jump), not a leave.
                    e.flush();
                    e.safepoint(&t);
                    ftab_tail(e, &t, scope.is_some(), self.weight);
                } else if self.ftab {
                    // Anywhere else: leave for it, and this body itself
                    // catches it first when it is one of its own blocks.
                    e.flush();
                    e.store_pc(&t);
                    e.safepoint(&t);
                    if scope.is_none() {
                        resumed.set(true);
                    }
                    e.line(fail);
                } else {
                    leave(e, &t);
                }
            }
            FTerm::Call { callee, ret_pc, ret } => {
                // A callee that left: resume right here when the guest is
                // back at this call's continuation (`local_resume`), else
                // leave as before.
                let local = ftab && scope.is_none() && local_resume() && ret.is_some();
                let fail_call = if local { format!("goto gex_lc_{i};") } else { fail.to_string() };
                let fail = fail_call.as_str();
                let spn = reg_name(sp_reg());
                if exp("NOFLAGWB") {
                    e.forget_flags();
                }
                // A known callee with a summary needs its inputs; the
                // continuation needs what is live there. Anything else may
                // read everything.
                //
                // Registers other than flags are limited by liveness only for
                // a trap-free callee (`trap_free`): nothing the host observes
                // or resumes from can happen before it returns. Otherwise
                // every dirty register is written back, so the context is
                // what tier-1's would be wherever the host looks.
                // `minimal_writeback`: a callee in this unit whose summary is
                // complete and that cannot reach the kernel or unwind gets
                // only its inputs and the continuation's live registers it
                // may change; the live registers it never writes (preserved
                // and not read: saving one reads it) stay dirty in locals,
                // and are written back on the leave path (`leave_wb`).
                let mut leave_wb = RegSet::EMPTY;
                let need = match (callee, ret) {
                    (Callee::Direct(t), Some(r)) if known.contains(t) => match sums.get(t) {
                        Some(sm) => {
                            let n = sm.inputs.union(live[r.to.index()]).union(RegSet::of(&[sp_reg()]));
                            if trapfree.contains_key(t) || mutate("IGNOREUNWIND") {
                                self.stats.calls_trapfree += 1;
                                match self.carry.get(&i) {
                                    Some(c) => {
                                        leave_wb = *c;
                                        n.minus(leave_wb)
                                    }
                                    None => n,
                                }
                            } else if let Some(c) = self.carry.get(&i) {
                                self.stats.calls_minimal += 1;
                                leave_wb = *c;
                                n.minus(leave_wb)
                            } else {
                                n.union(non_flags())
                            }
                        }
                        None => RegSet::ALL,
                    },
                    // Devirtualised (`carried`): the hinted callee keeps what
                    // it never touches in locals; the fallback stores those
                    // itself.
                    (Callee::Indirect(_), Some(_)) if self.carry.contains_key(&i) => {
                        leave_wb = self.carry[&i];
                        let all = if flags_abi() { call_reads(self.zf_calls.contains(&b.term_pc)) } else { RegSet::ALL };
                        all.minus(leave_wb)
                    }
                    // No callee reads the flags it is entered with
                    // (`flags_abi`).
                    _ if flags_abi() => call_reads(self.zf_calls.contains(&b.term_pc)),
                    _ => RegSet::ALL,
                };
                // Dirty registers the callee never writes stay dirty across
                // it (they are not reloaded: preserved).
                let carried = e.dirty_set().inter(leave_wb);
                e.flush_live(need);
                // What the callee must hand back: the flags only if the
                // continuation reads one; everything when there is no
                // continuation (believed not to return).
                let (need_back, ret_arg) = match ret {
                    Some(r) if live_writebacks() => {
                        let fl = !self.live[r.to.index()].inter(RegSet::of(FLAG_REGS)).is_empty();
                        (if fl { "NEED_ALL" } else { "0" }, format!("0x{ret_pc:x}ull"))
                    }
                    Some(_) | None => ("NEED_ALL", format!("0x{ret_pc:x}ull")),
                };
                e.line(format!(
                    "{{ const uint64_t sp_want = {spn} + {}ull;", self.ret_sp_delta
                ));
                match callee {
                    Callee::Direct(t) if known.contains(t) && self.ftab => {
                        calls.insert(*t);
                        self.stats.calls_known += 1;
                        // A callee that is not trap-free checks its own pages
                        // on entry; a trap-free one is checked here, where
                        // everything can be written back first. Rewritten:
                        // call what the table has for it instead.
                        if trapfree.contains_key(t) {
                            e.line(format!("  if (__builtin_expect({}, 0)) {{", patched_test(*t)));
                            // The carried registers too: whatever runs
                            // instead may read them.
                            e.mark_dirty(carried);
                            e.flush_keep();
                            ftab_call(e, &format!("0x{t:x}ull"), &format!("depth + {}", self.weight), need_back, &ret_arg, fail);
                            reload_all(e);
                            e.line("  } else");
                        }
                        let callee_c = callee_name(&self.leaves, self.copying, scope.is_some(), &mut self.leaf_copies, &mut self.stats, *t);
                        if leave_wb.is_empty() {
                            e.line(format!(
                                "  if (!{callee_c}(ctx, depth + {}, {need_back}, {ret_arg})) {fail}",
                                self.weight
                            ));
                        } else {
                            // It left: the registers it never writes are
                            // ours, still in locals; tier-1 (or this body's
                            // resume) needs them in the context.
                            e.line(format!("  if (!{callee_c}(ctx, depth + {}, {need_back}, {ret_arg})) {{", self.weight));
                            // (The write-back before the call marked
                            // everything clean.)
                            e.mark_dirty(carried);
                            e.flush_keep_live(leave_wb);
                            e.line(format!("    {fail}"));
                            e.line("  }");
                        }
                    }
                    Callee::Direct(t) if self.ftab => {
                        *self.stats.leaves.entry(*t).or_default() += 1;
                        ftab_call(e, &format!("0x{t:x}ull"), &format!("depth + {}", self.weight), need_back, &ret_arg, fail);
                    }
                    Callee::Indirect(v) if self.ftab => {
                        self.stats.calls_indirect += 1;
                        let t = e.v(*v);
                        let depth = format!("depth + {}", self.weight);
                        match self.icall_hints.get(ret_pc).copied().filter(|k| known.contains(k) && !self.no_devirt.contains(&i)) {
                            Some(k) => {
                                // `if (x == K) K(); else indirect(x)`. A trap-free K
                                // checks no pages itself: a rewritten one
                                // goes through the table.
                                self.stats.calls_devirt += 1;
                                calls.insert(k);
                                let guard = if trapfree.contains_key(&k) {
                                    format!(" && !({})", patched_test(k))
                                } else {
                                    String::new()
                                };
                                let callee_c = callee_name(&self.leaves, self.copying, scope.is_some(), &mut self.leaf_copies, &mut self.stats, k);
                                e.line(format!("  {{ const uint64_t ft = {t};"));
                                if carried.is_empty() {
                                    e.line(format!(
                                        "    if (__builtin_expect(ft == 0x{k:x}ull{guard}, 1)) {{ if (!{callee_c}(ctx, {depth}, {need_back}, {ret_arg})) {fail} }}"
                                    ));
                                    e.line(format!(
                                        "    else {{ GEX_C64({}u) = ft; if (!GEX_FTAB(gex_ftab_idx(ft))(ctx, {depth}, {need_back}, {ret_arg})) {fail} }} }}",
                                        ctx::LAST_TARGET
                                    ));
                                } else {
                                    // The hinted callee never touches the
                                    // carried registers: they stay in locals,
                                    // and are stored only if it leaves. Any
                                    // other target gets them stored first,
                                    // and may change them: reloaded after.
                                    e.line(format!("    if (__builtin_expect(ft == 0x{k:x}ull{guard}, 1)) {{"));
                                    e.line(format!("    if (!{callee_c}(ctx, {depth}, {need_back}, {ret_arg})) {{"));
                                    e.mark_dirty(carried);
                                    e.flush_keep_live(leave_wb);
                                    e.line(format!("      {fail}"));
                                    e.line("    }");
                                    e.line("    } else {");
                                    e.flush_keep_live(leave_wb);
                                    e.line(format!(
                                        "    GEX_C64({}u) = ft; if (!GEX_FTAB(gex_ftab_idx(ft))(ctx, {depth}, {need_back}, {ret_arg})) {fail}",
                                        ctx::LAST_TARGET
                                    ));
                                    let back: Vec<GuestReg> = e
                                        .func_regs
                                        .as_ref()
                                        .map(|fr| fr.used.iter().copied().filter(|r| leave_wb.contains(*r)).collect())
                                        .unwrap_or_default();
                                    e.load_regs(&back);
                                    e.line("    } }");
                                }
                            }
                            None => {
                                if crate::image::icall_count() {
                                    // The counting flavor: the profile the
                                    // hints come from (`image::icall_count`).
                                    e.line(format!("  gex_icall_note(ctx, 0x{ret_pc:x}ull, {t});"));
                                }
                                ftab_call(e, &t, &depth, need_back, &ret_arg, fail)
                            }
                        }
                    }
                    Callee::Direct(t) if known.contains(t) => {
                        calls.insert(*t);
                        self.stats.calls_known += 1;
                        // Its code rewritten since the unit was installed
                        // (Detours): leave for it, as for a callee not ours.
                        e.line(format!("  if (__builtin_expect({}, 0)) {{", patched_test(*t)));
                        e.flush_keep();
                        e.store_pc(&format!("0x{t:x}ull"));
                        e.line("    return 0;");
                        e.line("  }");
                        e.line(format!(
                            "  if (!{}(ctx, depth + {}, {need_back}, {ret_arg})) return 0;",
                            body_name(*t),
                            self.weight
                        ));
                    }
                    Callee::Direct(t) => {
                        *self.stats.leaves.entry(*t).or_default() += 1;
                        // Not ours: the host runs it and comes back at the
                        // return; if it cannot, the rest of this
                        // function runs in tier-1.
                        e.store_pc(&format!("0x{t:x}ull"));
                        e.line("  return 0;");
                    }
                    Callee::Indirect(v) => {
                        self.uses_call_known = true;
                        self.stats.calls_indirect += 1;
                        let t = e.v(*v);
                        e.line(format!(
                            "  int rk = {CALL_KNOWN}(ctx, {t}, depth + {}, {need_back}, {ret_arg});",
                            self.weight + CALL_KNOWN_WEIGHT
                        ));
                        e.line("  if (rk < 0) {");
                        e.store_pc(&t);
                        e.line("    return 0;");
                        e.line("  }");
                        e.line("  if (!rk) return 0;");
                    }
                }
                // The return check is the callee's result: 1 when it returned
                // to `ret_pc` with SP back (it wrote back what the
                // continuation reads), 0 when control went anywhere else
                // (everything written back, PC set): then leave too.
                e.line("  (void)sp_want;");
                e.line("}");
                // A summarised callee that returned as expected left its
                // preserved registers as they were: no reload. With
                // clean-spill elision, only what the continuation may read
                // is reloaded at all (the callee wrote back every register
                // but the flags, and the flags too when `need_back` asked).
                let preserved = match callee {
                    Callee::Direct(t) if known.contains(t) && live_writebacks() => {
                        sums.get(t).map(|sm| sm.preserved.minus(RegSet::of(&[sp_reg()])))
                    }
                    // Devirtualised with carried registers: the hinted
                    // callee kept them, the fallback reloaded them.
                    Callee::Indirect(_) if !leave_wb.is_empty() => Some(leave_wb),
                    _ => None,
                };
                match (ret, preserved) {
                    (Some(r), p) if live_writebacks() && clean_spills() && !exp("FULLRELOAD") => {
                        let load = self.live[r.to.index()].union(RegSet::of(&[sp_reg()]));
                        e.invalidate_only(load.minus(p.unwrap_or(RegSet::EMPTY)));
                    }
                    (_, Some(p)) => e.invalidate_except(p),
                    _ => e.invalidate(),
                }
                e.mark_dirty(carried);
                match ret {
                    Some(edge) => {
                        goto(e, edge);
                        if local {
                            // Reached only from the call's failure: the
                            // continuation is the call's own successor, so
                            // the control flow stays reducible.
                            e.line(format!(
                                "gex_lc_{i}: if (GEX_C64({}u) == 0x{ret_pc:x}ull && GEX_C64({}u) <= sp0) {{",
                                ctx::PC,
                                ctx::SP
                            ));
                            reload_all(e);
                            e.line(format!("  goto L{};", edge.to.index()));
                            e.line("}");
                            resumed.set(true);
                            e.line("goto gex_resume;");
                        }
                    }
                    // Believed not to return, and it did: leave to the
                    // return address, with everything written back.
                    None => {
                        e.store_pc(&format!("0x{ret_pc:x}ull"));
                        e.line("return 0;");
                    }
                }
            }
            FTerm::TailCall { callee } if self.ftab => {
                if exp("NOFLAGWB") {
                    e.forget_flags();
                }
                let t = match callee {
                    Callee::Direct(t) => format!("0x{t:x}ull"),
                    Callee::Indirect(v) => e.v(*v),
                };
                e.flush();
                // A tail call does not pass through a loop back-edge or the
                // root, so it polls (returns no longer do).
                e.safepoint(&t);
                match callee {
                    Callee::Direct(c) if known.contains(c) => {
                        calls.insert(*c);
                        self.stats.tail_known += 1;
                        if trapfree.contains_key(c) {
                            e.line(format!("if (__builtin_expect({}, 0)) {{", patched_test(*c)));
                            ftab_tail(e, &t, scope.is_some(), self.weight);
                            e.line("}");
                        }
                        if scope.is_some() {
                            e.line(format!("return {}(ctx, depth + {}, need, ret);", body_name(*c), self.weight));
                        } else {
                            e.line(format!("GEX_MUSTTAIL return {}(ctx, depth, need, ret);", body_name(*c)));
                        }
                    }
                    _ => {
                        if let Callee::Direct(c) = callee {
                            *self.stats.leaves.entry(*c).or_default() += 1;
                        }
                        ftab_tail(e, &t, scope.is_some(), self.weight);
                    }
                }
            }
            FTerm::TailCall { callee } => match callee {
                Callee::Direct(t) if known.contains(t) => {
                    calls.insert(*t);
                    self.stats.tail_known += 1;
                    if exp("NOFLAGWB") {
                        e.forget_flags();
                    }
                    e.flush();
                    if exp("NORETPOLL") {
                        e.safepoint(&format!("0x{t:x}ull"));
                    }
                    // A tail call does not pass through a loop back-edge or
                    // the root, so it polls (returns no longer do).
                    e.safepoint(&format!("0x{t:x}ull"));
                    e.line(format!("if (__builtin_expect({}, 0)) {{", patched_test(*t)));
                    e.store_pc(&format!("0x{t:x}ull"));
                    e.line("  return 0;");
                    e.line("}");
                    if scope.is_some() {
                        // A loop function's signature is not the bodies':
                        // an ordinary call (one more frame) instead.
                        e.line(format!("return {}(ctx, depth + {}, need, ret);", body_name(*t), self.weight));
                    } else {
                        e.line(format!("GEX_MUSTTAIL return {}(ctx, depth, need, ret);", body_name(*t)));
                    }
                }
                Callee::Direct(t) => {
                    *self.stats.leaves.entry(*t).or_default() += 1;
                    leave(e, &format!("0x{t:x}ull"))
                }
                Callee::Indirect(v) => {
                    let t = e.v(*v);
                    leave(e, &t)
                }
            },
            FTerm::Return { target } | FTerm::Unresolved { target } => {
                if matches!(b.term, FTerm::Return { .. }) {
                    self.stats.returns += 1;
                } else {
                    self.stats.unresolved += 1;
                }
                // tier-1 polls on every indirect branch; so does this, with
                // everything already written back and PC set.
                let t = e.v(*target);
                if matches!(b.term, FTerm::Return { .. }) {
                    // The return the caller expects: write back what its
                    // continuation may read (everything but the flags, and
                    // the flags when `need` says so) and report success;
                    // the caller carries on without looking at the context.
                    // `GEX_CGEN_MUTATE=NORETCHECK` skips the check.
                    // SP back where the caller had it, plus what this
                    // `ret` pops: the return address, and on i386 `ret N`'s
                    // N (a stdcall callee's arguments, a struct return's
                    // pointer). The caller reloads SP after every call.
                    let spn = reg_name(sp_reg());
                    let delta = self.ret_pops.get(&b.term_pc).copied().unwrap_or(self.ret_sp_delta);
                    if mutate("NORETCHECK") {
                        e.line("if (1) {");
                    } else {
                        e.line(format!("if (__builtin_expect({t} == ret && {spn} == sp0 + {delta}ull, 1)) {{"));
                    }
                    let flags: Vec<GuestReg> = FLAG_REGS.to_vec();
                    e.flush_keep_live(RegSet::ALL.minus(RegSet::of(&flags)));
                    e.line("  if (need & NEED_FLAGS) {");
                    e.flush_keep_live(RegSet::of(&flags));
                    e.line("  }");
                    e.line("  return 1;");
                    e.line("}");
                }
                // Anywhere else: everything written back, PC set; the callers
                // unwind and the root polls and dispatches. A jump nobody
                // resolved goes on through the function table instead, under
                // this function's own return contract.
                e.flush();
                e.store_pc(&t);
                if self.ftab && matches!(b.term, FTerm::Unresolved { .. }) && local_resume() {
                    // An indirect jump nobody resolved (`jmp [iat]`, a
                    // computed tail call) goes on through the function
                    // table under this function's own return contract, as
                    // the function JIT does: leaving would unwind every
                    // frame above. A target inside this function runs on
                    // the fallback (the call-out) and returns the same way.
                    e.safepoint(&t);
                    ftab_tail(e, &t, scope.is_some(), self.weight);
                } else if self.ftab {
                    // This body catches it first when it is one of its own
                    // blocks (a jump table recovery did not resolve, a
                    // return used as a jump), at no cost to anyone else.
                    if matches!(b.term, FTerm::Unresolved { .. }) {
                        e.safepoint(&t);
                    }
                    if scope.is_none() {
                        resumed.set(true);
                    }
                    e.line(fail);
                } else {
                    e.line("return 0;");
                }
            }
            FTerm::Syscall { next_pc, next, imm } => {
                e.flush();
                e.store_pc(&format!("0x{next_pc:x}ull"));
                if *imm != 0 {
                    e.line(format!("GEX_C64(GEX_CTX_SYSCALL_IMM) = {imm}ull;"));
                }
                e.line(format!("gex_syscall(ctx, 0x{:x}ll);", b.term_pc));
                e.invalidate();
                match next {
                    Some(edge) => goto(e, edge),
                    None => e.line("return 0;"),
                }
            }
            FTerm::Illegal { raw, pc } => {
                e.flush();
                e.store_pc(&format!("0x{pc:x}ull"));
                match e.fault_exit {
                    Some(exit) => {
                        e.line(format!("gex_illegal(ctx, (int64_t)(0x{pc:x}ull | GEX_RESUMABLE_PC), (int32_t)0x{raw:x}u);"));
                        e.line(exit);
                        if scope.is_none() {
                            resumed.set(true);
                        }
                    }
                    None => {
                        e.line(format!("gex_illegal(ctx, 0x{pc:x}ll, (int32_t)0x{raw:x}u);"));
                        e.line("__builtin_trap();");
                    }
                }
            }
        }
        if resumed.get() {
            self.resumes = true;
        }
        Ok(())
    }

    /// Bytes of C bodies emitted so far.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    pub fn source(&self) -> String {
        let mut s = String::from("#include \"gex_abi.h\"\n\n");
        s.push_str(NEED_DEFS);
        let st = self.storage();
        for a in &self.protos {
            let _ = writeln!(s, "{st}int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret);", body_name(*a));
        }
        for a in &self.leaf_copies {
            let _ = writeln!(s, "static inline int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret);", leaf_name(*a));
        }
        if !self.ftab {
            let _ = writeln!(s, "{st}int {CALL_KNOWN}(int ctx, uint64_t target, int32_t depth, uint32_t need, uint64_t ret);\n");
        }
        s.push_str(&self.bodies);
        if self.ftab && !self.defined.is_empty() {
            // Every body's address taken, so the linker puts it in the
            // element segment that fills the function table from
            // `__table_base` (`image::funcs_section` reads where). Exported
            // so it is kept; nothing calls it.
            let first = self.defined[0];
            let _ = writeln!(s, "__attribute__((export_name(\"gex_refs_{first:x}\"))) uint32_t gex_refs_{first:x}(uint32_t k) {{\n  switch (k) {{");
            for (k, a) in self.defined.iter().enumerate() {
                let _ = writeln!(s, "  case {k}u: return (uint32_t)(uintptr_t)&{};", body_name(*a));
            }
            s.push_str("  default: return 0;\n  }\n}\n");
        }
        s
    }
}

/// A call through the function table to `t` (a C expression): the target in
/// `ctx::LAST_TARGET`, the index from the probe cache (the resolve stub on a
/// miss), the callee's result tested as for any call.
fn ftab_call(e: &mut Emitter<'_>, t: &str, depth: &str, need: &str, ret: &str, fail: &str) {
    e.line(format!("  {{ const uint64_t ft = {t}; GEX_C64({}u) = ft;", ctx::LAST_TARGET));
    e.line(format!("    if (!GEX_FTAB(gex_ftab_idx(ft))(ctx, {depth}, {need}, {ret})) {fail} }}"));
}

/// Where a whole-image unit's body goes when a callee left (returned 0):
/// everything is in the context and `ctx::PC` says where the guest goes.
/// If that is the continuation of one of this body's calls and the stack is
/// at or below this activation's (a callee whose return check failed, a
/// `longjmp` back to a `setjmp`, a call-out that came back), the body
/// reloads every register and carries on there, and nothing further out
/// unwinds; otherwise it leaves too. Correct for any activation of the
/// function -- the guest's own stack holds the real return addresses -- so
/// the stack test only keeps a deeper activation from taking what an outer
/// one is waiting for. Only call continuations: they reload after the call
/// anyway, while making every block a join point with all registers
/// reloaded nearly doubled the largest bodies (one from 414 to
/// 720 KB of wasm) and put them past Cranelift's limits.
/// Emit a block's statements, telling the emitter where each resumable
/// fault statement (`int3`'s `HostCall`) carries on.
fn block_stmts(e: &mut Emitter<'_>, b: &FBlock) -> Result<()> {
    e.stmts(&b.stmts, |e, k| {
        let s = &b.stmts[k];
        e.fault_next = match s {
            Stmt::HostCall { func: HostFunc::Breakpoint | HostFunc::Illegal, .. } if e.fault_exit.is_some() => {
                let pc = b.pcs[k];
                Some(b.insts.iter().position(|&a| a == pc).and_then(|j| b.insts.get(j + 1).copied()).unwrap_or(b.end))
            }
            _ => None,
        };
        e.stmt(s)
    })?;
    e.fault_next = None;
    Ok(())
}

fn resume_block(e: &mut Emitter<'_>, f: &Function, inlined: &HashMap<usize, usize>) {
    if local_resume() {
        // Each call takes its own continuation (`gex_lc_*`). Anything else
        // inside this function's code with the stack at or below this
        // activation is a continuation this frame owns but no call of it
        // returned to: a `longjmp` to a `setjmp` here, a `catch` or
        // `__except` continuation, a callee's leave for one of our blocks.
        // The rest of the function then runs through the table (the
        // fallback's call-out, or a region) under this frame's own return
        // contract, so the frames further out keep running their bodies;
        // leaving would hand all of them to the fallback.
        // Not an inlined leaf's blocks (`inline_leaf_calls`): their code is
        // the leaf's, elsewhere.
        let own = || f.blocks.iter().enumerate().filter(|(i, _)| !inlined.contains_key(i)).map(|(_, b)| b);
        let lo = own().map(|b| b.start).min().unwrap_or(f.entry);
        let hi = own().map(|b| b.end.max(b.start + 1)).max().unwrap_or(f.entry + 1);
        e.line("gex_resume: {");
        e.line(format!("  const uint64_t rpc = GEX_C64({}u);", ctx::PC));
        e.line(format!(
            "  if (rpc - 0x{lo:x}ull < 0x{:x}ull && GEX_C64({}u) <= sp0) {{",
            hi.saturating_sub(lo),
            ctx::SP
        ));
        ftab_tail(e, "rpc", false, 0);
        e.line("  }");
        e.line("  return 0;");
        e.line("}");
        return;
    }
    let mut starts: Vec<(u64, usize)> = f
        .blocks
        .iter()
        .filter_map(|b| match &b.term {
            FTerm::Call { ret: Some(edge), .. } => Some(edge.to.index()),
            _ => None,
        })
        .filter(|&i| !f.blocks[i].synthetic)
        .map(|i| (f.blocks[i].start, i))
        .collect();
    starts.sort_unstable();
    starts.dedup_by_key(|x| x.0);
    if starts.is_empty() {
        e.line("gex_resume: return 0;");
        return;
    }
    let lo = starts[0].0;
    let hi = starts[starts.len() - 1].0 + 1;
    e.line("gex_resume: {");
    e.line(format!(
        "  const uint64_t rpc = GEX_C64({}u);",
        ctx::PC
    ));
    e.line(format!(
        "  if (rpc - 0x{lo:x}ull < 0x{:x}ull && GEX_C64({}u) <= sp0) {{",
        hi.saturating_sub(lo),
        ctx::SP
    ));
    reload_all(e);
    e.line("    switch (rpc) {");
    for (a, i) in &starts {
        e.line(format!("    case 0x{a:x}ull: goto L{i};"));
    }
    e.line("    }");
    e.line("  }");
    e.line("  return 0;");
    e.line("}");
}

/// A tail call through the function table to `t`, with this function's own
/// `need` and `ret`: a guest jump hands its caller's return over. A loop
/// function's signature is not the bodies', so there it is an ordinary call.
fn ftab_tail(e: &mut Emitter<'_>, t: &str, in_loop: bool, weight: u32) {
    e.line(format!("{{ const uint64_t ft = {t}; GEX_C64({}u) = ft;", ctx::LAST_TARGET));
    if in_loop {
        e.line(format!("  return GEX_FTAB(gex_ftab_idx(ft))(ctx, depth + {weight}, need, ret); }}"));
    } else {
        e.line("  GEX_MUSTTAIL return GEX_FTAB(gex_ftab_idx(ft))(ctx, depth, need, ret); }");
    }
}

/// Load every register the function (or loop function) keeps in locals
/// from the context, leaving the emitter's view as it was: after a call on
/// a conditional path, joining one that reloads lazily.
fn reload_all(e: &mut Emitter<'_>) {
    let used: Vec<GuestReg> = e.func_regs.as_ref().map(|r| r.used.iter().copied().collect()).unwrap_or_default();
    e.load_regs(&used);
}

/// The helper indirect calls go through: a `switch` over every function
/// entry with a C body: the slow path, which the dispatch-cache probe and
/// devirtualization speed up.
pub fn call_known_source(entries: &[u64]) -> String {
    let mut s = String::from("#include \"gex_abi.h\"\n\n");
    for a in entries {
        let _ = writeln!(s, "int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret);", body_name(*a));
    }
    let _ = writeln!(s, "\nint {CALL_KNOWN}(int ctx, uint64_t target, int32_t depth, uint32_t need, uint64_t ret) {{");
    s.push_str("  switch (target) {\n");
    for a in entries {
        let _ = writeln!(s, "  case 0x{a:x}ull: return {}(ctx, depth, need, ret);", body_name(*a));
    }
    s.push_str("  default: return -1;\n  }\n}\n");
    s
}

/// [`call_known_source`] for many entries: clang takes minutes over one
/// `switch` of a hundred thousand cases, so the entries are cut into chunks
/// of [`CALL_KNOWN_CHUNK`], each a small function with its own `switch`, and
/// the helper picks the chunk by a balanced tree of comparisons.
pub const CALL_KNOWN_CHUNK: usize = 1024;

pub fn call_known_source_chunked(entries: &[u64]) -> String {
    if entries.len() <= CALL_KNOWN_CHUNK {
        return call_known_source(entries);
    }
    let mut sorted = entries.to_vec();
    sorted.sort_unstable();
    let chunks: Vec<&[u64]> = sorted.chunks(CALL_KNOWN_CHUNK).collect();
    let mut s = String::from("#include \"gex_abi.h\"\n\n");
    for a in &sorted {
        let _ = writeln!(s, "int {}(int ctx, int32_t depth, uint32_t need, uint64_t ret);", body_name(*a));
    }
    for (k, c) in chunks.iter().enumerate() {
        let _ = writeln!(s, "\nstatic int ck_{k}(int ctx, uint64_t target, int32_t depth, uint32_t need, uint64_t ret) {{");
        s.push_str("  switch (target) {\n");
        for a in c.iter() {
            let _ = writeln!(s, "  case 0x{a:x}ull: return {}(ctx, depth, need, ret);", body_name(*a));
        }
        s.push_str("  default: return -1;\n  }\n}\n");
    }
    // A chunk is picked by its first entry: chunk k holds [first[k], first[k+1]).
    fn tree(s: &mut String, chunks: &[&[u64]], lo: usize, hi: usize, indent: usize) {
        let pad = "  ".repeat(indent);
        if hi - lo == 1 {
            let _ = writeln!(s, "{pad}return ck_{lo}(ctx, target, depth, need, ret);");
            return;
        }
        let mid = (lo + hi) / 2;
        let _ = writeln!(s, "{pad}if (target < 0x{:x}ull) {{", chunks[mid][0]);
        tree(s, chunks, lo, mid, indent + 1);
        let _ = writeln!(s, "{pad}}}");
        tree(s, chunks, mid, hi, indent);
    }
    let _ = writeln!(s, "\nint {CALL_KNOWN}(int ctx, uint64_t target, int32_t depth, uint32_t need, uint64_t ret) {{");
    tree(&mut s, &chunks, 0, chunks.len(), 1);
    s.push_str("}\n");
    s
}

/// Every register the function reads or writes, and every one it writes.
/// The stack pointer is always used: the return check reads it.
pub fn registers(f: &Function) -> FnRegs {
    registers_in(f, &|_| true)
}

/// [`registers`] over the blocks `keep` admits (a loop function's).
pub fn registers_in(f: &Function, keep: &dyn Fn(usize) -> bool) -> FnRegs {
    let mut used = BTreeSet::new();
    let mut written = BTreeSet::new();
    used.insert(sp_reg());
    for (i, b) in f.blocks.iter().enumerate() {
        if !keep(i) {
            continue;
        }
        for s in &b.stmts {
            match s {
                Stmt::Def {
                    expr: Expr::ReadReg(r),
                    ..
                } => {
                    used.insert(*r);
                }
                Stmt::WriteReg { reg, .. } => {
                    used.insert(*reg);
                    written.insert(*reg);
                }
                _ => {}
            }
        }
    }
    FnRegs {
        used,
        written,
        clean_spills: false,
    }
}

/// Clean-spill elision is on
/// unless `GEX_CGEN_CLEAN=0`.
pub fn clean_spills() -> bool {
    std::env::var("GEX_CGEN_CLEAN").as_deref() != Ok("0")
}

/// For each block, the registers that may differ from the context on entry:
/// a forward may-analysis. A write makes a register dirty; a write-back
/// followed by a reload (a call, a syscall, a clobbering host call) makes
/// everything clean; the entry starts clean, having just loaded everything.
pub fn dirty_in(f: &Function, carry: &HashMap<usize, RegSet>) -> Vec<BTreeSet<GuestReg>> {
    dirty_in_from(f, 0, &|_| true, carry)
}

/// [`dirty_in`] for the blocks `keep` admits, entered (clean) at `root`:
/// a loop function, whose caller wrote everything back before calling it.
/// `carry`: per block ending in a call with the minimal write-back
/// (`FnUnit::carried`), the registers that stay dirty across it.
pub fn dirty_in_from(f: &Function, root: usize, keep: &dyn Fn(usize) -> bool, carry: &HashMap<usize, RegSet>) -> Vec<BTreeSet<GuestReg>> {
    let n = f.blocks.len();
    let mut inn: Vec<Option<BTreeSet<GuestReg>>> = vec![None; n];
    inn[root] = Some(BTreeSet::new());
    let order: Vec<BlockId> = f.rpo().into_iter().filter(|b| keep(b.index())).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &order {
            let Some(mut d) = inn[id.index()].clone() else { continue };
            let b = &f.blocks[id.index()];
            for s in &b.stmts {
                match s {
                    Stmt::WriteReg { reg, .. } if !emit::is_scratch(*reg) => {
                        d.insert(*reg);
                    }
                    _ if emit::clobbers(s) => d.clear(),
                    _ => {}
                }
            }
            if matches!(b.term, FTerm::Call { .. } | FTerm::Syscall { .. }) {
                match carry.get(&id.index()) {
                    Some(c) if matches!(b.term, FTerm::Call { .. }) => d.retain(|r| c.contains(*r)),
                    _ => d.clear(),
                }
            }
            for s in b.succs() {
                if !keep(s.index()) {
                    continue;
                }
                let slot = &mut inn[s.index()];
                let merged = match slot {
                    Some(old) => {
                        let u: BTreeSet<GuestReg> = old.union(&d).copied().collect();
                        if u.len() == old.len() {
                            continue;
                        }
                        u
                    }
                    None => d.clone(),
                };
                *slot = Some(merged);
                changed = true;
            }
        }
    }
    inn.into_iter().map(|d| d.unwrap_or_default()).collect()
}

/// Liveness-limited write-backs are on unless `GEX_CGEN_LIVE=0`.
pub fn live_writebacks() -> bool {
    std::env::var("GEX_CGEN_LIVE").as_deref() != Ok("0")
}

/// For each block, the registers (flags included) that may be read before
/// being written on some path from its entry, where every exit the function
/// does not see past reads everything: returns, tail calls, unresolved and
/// switch-default jumps, syscalls, calls with no summary or no
/// continuation, and clobbering host calls. A call to a summarised
/// function reads its inputs and passes on what its continuation reads.
///
/// Write-backs guarded by this are exact for anything that resumes at the
/// point the liveness describes (a safepoint's loop head, a callee trapping
/// out and tier-1 returning into the continuation). Registers the ABI does
/// not preserve across calls can be stale in the context when a callee
/// `longjmp`s to an earlier call's continuation in the same frame; C and
/// the ABIs leave them undefined there.
pub fn live_in(f: &Function, sums: &HashMap<u64, Summary>) -> Vec<RegSet> {
    live_in_with(f, sums, &HashSet::new())
}

/// [`live_in`], with ZF read by the calls at `zf` (their `term_pc`s:
/// [`zf_calls`]).
pub fn live_in_with(f: &Function, sums: &HashMap<u64, Summary>, zf: &HashSet<u64>) -> Vec<RegSet> {
    let n = f.blocks.len();
    let mut live = vec![RegSet::EMPTY; n];
    let order: Vec<usize> = f.rpo().iter().rev().map(|b| b.index()).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &i in &order {
            let b = &f.blocks[i];
            let mut l = match &b.term {
                FTerm::Jump(e) => live[e.to.index()],
                FTerm::Branch {
                    taken, not_taken, ..
                } => live[taken.to.index()].union(live[not_taken.to.index()]),
                // The default case leaves.
                FTerm::Switch { .. } => RegSet::ALL,
                FTerm::Call {
                    callee: Callee::Direct(t),
                    ret: Some(r),
                    ..
                } => match sums.get(t) {
                    Some(sm) => sm.inputs.union(live[r.to.index()]),
                    None if flags_abi() => call_reads(zf.contains(&b.term_pc)).union(live[r.to.index()]),
                    None => RegSet::ALL,
                },
                // `flags_abi`: no callee reads the flags it is entered with.
                FTerm::Call { ret, .. } if flags_abi() => {
                    call_reads(zf.contains(&b.term_pc)).union(ret.as_ref().map_or(RegSet::EMPTY, |r| live[r.to.index()]))
                }
                FTerm::TailCall { .. } if flags_abi() => non_flags(),
                _ => RegSet::ALL,
            };
            for s in b.stmts.iter().rev() {
                match s {
                    Stmt::WriteReg { reg, .. } => l.remove(*reg),
                    Stmt::Def {
                        expr: Expr::ReadReg(r),
                        ..
                    } => l.insert(*r),
                    // A trap reads everything (on its faulting path).
                    Stmt::Trap { .. } => l = RegSet::ALL,
                    _ if emit::clobbers(s) => l = RegSet::ALL,
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

/// Every register but the flags.
/// Whether calls are taken not to read the arithmetic flags: no x86 ABI
/// passes anything in them, and compilers treat every call as clobbering
/// them. A flag computation whose only reader would be the callee is then
/// dead, and a call writes back no flags. Returns, switch defaults and unresolved jumps still read them
/// all. `GEX_CGEN_FLAGS_ABI=0` turns it off; `GEX_CGEN_LIVE=0` implies off.
pub fn flags_abi() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| gex_abi::X86 && live_writebacks() && std::env::var("GEX_CGEN_FLAGS_ABI").as_deref() != Ok("0"))
}

/// The whole-image units' write-back at calls to a summarised callee that
/// cannot reach the kernel, unwind or leave pending (`FnUnit::add`): what
/// it reads, and what the continuation reads that it may change. A trap inside it (a memory fault: SEH) sees the caller's
/// other registers as they were at the last write-back; the compilers'
/// handlers reload callee-saved registers from their frames, not from the
/// faulting context. `GEX_CGEN_WB=all` writes back every dirty register.
/// The registers each function's call tree names (reads or writes, saving
/// included): a least fixpoint over direct calls; anything
/// that reaches code outside `funcs`, an indirect transfer or a clobbering
/// host call names everything.
pub fn touched(funcs: &[Function]) -> HashMap<u64, RegSet> {
    let idx: HashMap<u64, usize> = funcs.iter().enumerate().map(|(k, f)| (f.entry, k)).collect();
    let mut local = vec![RegSet::EMPTY; funcs.len()];
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); funcs.len()];
    for (k, f) in funcs.iter().enumerate() {
        let mut l = RegSet::EMPTY;
        for b in &f.blocks {
            for s in &b.stmts {
                match s {
                    Stmt::WriteReg { reg, .. } => l.insert(*reg),
                    Stmt::Def { expr: Expr::ReadReg(r), .. } => l.insert(*r),
                    // A host helper may read or write any register in the
                    // context (x87, fxsave, cpuid), and a trap reads them all.
                    Stmt::HostCall { .. } | Stmt::Def { expr: Expr::HostCall { .. }, .. } | Stmt::Trap { .. } => l = RegSet::ALL,
                    _ if emit::clobbers(s) => l = RegSet::ALL,
                    _ => {}
                }
            }
            match &b.term {
                FTerm::Call { callee: Callee::Direct(t), .. } | FTerm::TailCall { callee: Callee::Direct(t) } => match idx.get(t) {
                    Some(&c) => callees[k].push(c),
                    None => l = RegSet::ALL,
                },
                FTerm::Call { .. } | FTerm::TailCall { .. } | FTerm::Unresolved { .. } | FTerm::Syscall { .. } => l = RegSet::ALL,
                FTerm::Switch { table, .. } if !table.complete => l = RegSet::ALL,
                _ => {}
            }
        }
        local[k] = l;
    }
    let mut t = local.clone();
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 64 {
        changed = false;
        rounds += 1;
        for k in 0..funcs.len() {
            let mut v = t[k];
            for &c in &callees[k] {
                v = v.union(t[c]);
            }
            if v != t[k] {
                t[k] = v;
                changed = true;
            }
        }
    }
    if changed {
        return HashMap::new();
    }
    funcs.iter().enumerate().map(|(k, f)| (f.entry, t[k])).collect()
}

/// `GEX_CGEN_WB_BISECT=N`: the minimal write-back at the first N call
/// sites a process emits only (bisecting a miscompile).
fn wb_bisect() -> bool {
    static N: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    match *N.get_or_init(|| std::env::var("GEX_CGEN_WB_BISECT").ok().and_then(|v| v.parse().ok())) {
        Some(n) => WB_K.load(std::sync::atomic::Ordering::Relaxed) < n,
        None => true,
    }
}
static WB_K: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The functions that need no recursion-depth guard: not recursive, their whole call tree known (no indirect call, no
/// unknown code, so nothing can re-enter them), and the tree's stack fits in
/// the reserve every guarded function keeps below the budget, as a
/// trap-free tree must. The guard stays where a call chain can grow
/// without bound.
pub fn guard_free(funcs: &[Function], sums: &HashMap<u64, Summary>) -> HashSet<u64> {
    let idx: HashMap<u64, usize> = funcs.iter().enumerate().map(|(k, f)| (f.entry, k)).collect();
    let mut cand: Vec<Option<(u32, Vec<usize>)>> = Vec::with_capacity(funcs.len());
    for f in funcs {
        let ok = sums.get(&f.entry).is_some_and(|s| !s.unknown_callee && !s.may_unwind);
        let mut callees = Vec::new();
        let mut bad = !ok;
        for b in &f.blocks {
            match &b.term {
                FTerm::Call { callee: Callee::Direct(t), .. } | FTerm::TailCall { callee: Callee::Direct(t) } => match idx.get(t) {
                    Some(&c) => callees.push(c),
                    None => bad = true,
                },
                FTerm::Call { .. } | FTerm::TailCall { .. } | FTerm::Unresolved { .. } => bad = true,
                _ => {}
            }
        }
        cand.push((!bad).then(|| (frame_weight(&registers(f)), callees)));
    }
    // Depth-first, memoised; a cycle (recursion) fails every function on it.
    fn visit(k: usize, cand: &[Option<(u32, Vec<usize>)>], memo: &mut [Option<Option<u32>>], onstack: &mut [bool]) -> Option<u32> {
        if let Some(m) = memo[k] {
            return m;
        }
        if onstack[k] {
            return None;
        }
        onstack[k] = true;
        let r = cand[k].as_ref().and_then(|(w, callees)| {
            let mut deepest = 0u32;
            for &c in callees {
                deepest = deepest.max(visit(c, cand, memo, onstack)? + CALL_KNOWN_WEIGHT);
            }
            Some(w + deepest).filter(|n| *n <= TRAPFREE_RESERVE)
        });
        onstack[k] = false;
        memo[k] = Some(r);
        r
    }
    let mut memo = vec![None; funcs.len()];
    let mut onstack = vec![false; funcs.len()];
    let mut out = HashSet::new();
    for k in 0..funcs.len() {
        if visit(k, &cand, &mut memo, &mut onstack).is_some() {
            out.insert(funcs[k].entry);
        }
    }
    out
}

/// Whole-image units resume after a callee that left only at that call's
/// own continuation, checked at the call site (`gex_lc_<block>`), not
/// through one switch over every continuation of the body: the switch
/// jumps into loops, which made the control flow irreducible, and clang's
/// wasm backend then dispatched every iteration of those loops through a
/// label variable (one function: 10,845 instructions with the
/// switch, 9,186 without). Lost: resuming at a different continuation of
/// the same frame, which now leaves (correct, slower).
/// `GEX_CGEN_EXP=CENTRALRESUME` restores the switch (not in the unit key:
/// for pricing only).
/// Calls to small leaves go to unit-private copies (`FnUnit::leaves`);
/// `GEX_CGEN_EXP=NOLEAFCOPY` turns it off (not in the unit key: use a
/// private cache).
pub fn leaf_copies() -> bool {
    !exp("NOLEAFCOPY")
}

/// Calls to small call-free leaves are inlined into the caller's GIR
/// (`FnUnit::inline_leaf_calls`) in i386 units, instead of calling the
/// leaf's copy. On by default (faster in the page);
/// `GEX_CGEN_EXP=NOINLINELEAVES` turns it off (not in the unit key: use a
/// private cache).
pub fn inline_leaves() -> bool {
    !exp("NOINLINELEAVES")
}

/// How far an inlined leaf's `ret` (block `b`, a `Return`) moved SP past
/// the return address it popped: 4 for a plain one-byte `ret`, `ret N`'s
/// from `pops` (`ret_pops`); `None` when that is not known from the block
/// alone (prefixes: `66 c3` pops two bytes), and its mismatch keeps an
/// exit of its own.
fn ret_delta(b: &FBlock, pops: &HashMap<u64, u64>) -> Option<u64> {
    match (b.end.checked_sub(b.term_pc), pops.get(&b.term_pc)) {
        (Some(1), None) => Some(4),
        (Some(3), Some(&d)) => Some(d),
        _ => None,
    }
}

/// Whether inlined leaves' `ret` mismatches share one exit per pop
/// (`inline_leaf_calls`); `GEX_CGEN_EXP=NOSHAREDRETEXITS` gives each its
/// own (not in the unit key: use a private cache).
fn shared_ret_exits() -> bool {
    !exp("NOSHAREDRETEXITS")
}

/// Whether `leaf` can be inlined at a call (`FnUnit::inline_leaf_calls`): a
/// copyable leaf (`is_small_copyable`) that makes no calls, has no loop, and
/// leaves only by `ret`.
fn inlinable(leaf: &Function) -> bool {
    if leaf.form != Form::Memory || !is_small_copyable(leaf) || !makes_no_calls(leaf) {
        return false;
    }
    if !leaf.blocks.iter().all(|b| matches!(b.term, FTerm::Jump(_) | FTerm::Branch { .. } | FTerm::Return { .. })) {
        return false;
    }
    // Acyclic: every block reachable from the entry is finished (DFS colours).
    let n = leaf.blocks.len();
    let mut state = vec![0u8; n];
    let mut stack = vec![(0usize, 0usize)];
    state[0] = 1;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        let succs = leaf.blocks[b].succs();
        if *k < succs.len() {
            let s = succs[*k].index();
            *k += 1;
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => return false,
                _ => {}
            }
        } else {
            state[b] = 2;
            stack.pop();
        }
    }
    true
}

pub fn local_resume() -> bool {
    !exp("CENTRALRESUME")
}

pub fn minimal_writeback() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| live_writebacks() && std::env::var("GEX_CGEN_WB").as_deref() != Ok("all"))
}

/// `minimal_writeback` at calls to trap-free callees as well: the live
/// registers they never touch stay in locals across them instead of being
/// stored before and trusted after. Off unless `GEX_CGEN_WB_TRAPFREE=1`:
/// with both carries on, a game measured 8% slower than with them off: the
/// carried registers stay dirty, so every later exit (cold breakpoints and
/// leaves included) stores them and they stay live in locals across the
/// function, while the hot call they were for did not get faster.
pub fn trapfree_carry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GEX_CGEN_WB_TRAPFREE").as_deref() == Ok("1"))
}

/// `minimal_writeback` at devirtualised indirect calls, for the hinted
/// callee; the fallback stores the carried registers and reloads them.
/// Off unless `GEX_CGEN_WB_DEVIRT=1` (see `trapfree_carry`).
pub fn devirt_carry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GEX_CGEN_WB_DEVIRT").as_deref() == Ok("1"))
}

/// What a call to a callee without a summary (indirect, outside the unit,
/// or unsummarised) reads of the caller's registers under `flags_abi`:
/// everything but the flags, and ZF at a stack check (`zf`, [`zf_calls`]).
/// `GEX_CGEN_EXP=ZFCALLS` gives every such call ZF (2.7% slower in the
/// page).
pub fn call_reads(zf: bool) -> RegSet {
    if zf || exp("ZFCALLS") {
        non_flags().union(RegSet::of(&[GuestReg::Z]))
    } else {
        non_flags()
    }
}

pub fn non_flags() -> RegSet {
    RegSet::ALL.minus(RegSet::of(FLAG_REGS))
}

/// Stack every non-trap-free function keeps free below the budget for a
/// trap-free subtree, which has no depth guard of its own.
pub const TRAPFREE_RESERVE: u32 = 64 << 10;

/// The guest instructions `f` was lifted from (its blocks' own, not the
/// synthetic ones').
pub fn guest_insns(f: &Function) -> usize {
    f.blocks.iter().filter(|b| !b.synthetic).map(|b| b.insts.len()).sum()
}

/// The trap-free functions, with the stack estimate of their whole subtree.
///
/// Trap-free: once called, the function returns to its caller without
/// anything the host could observe or resume from: no syscall, no safepoint
/// (no backward edge, no tail call), no clobbering host call, no exit that
/// leaves the function tier (unresolved or incomplete-switch jumps,
/// indirect calls, calls to anything but trap-free functions), no illegal
/// instruction, and not recursive. S2's summary must also say it cannot
/// unwind, leave pending or reach unknown code. Its callers then write back
/// only what it reads and what their continuation reads: registers they
/// hold dead in locals are never seen by anyone. Its subtree must fit in
/// `TRAPFREE_RESERVE`, since it has no depth guard.
pub fn trap_free(funcs: &[Function], sums: &HashMap<u64, Summary>) -> HashMap<u64, u32> {
    let by_entry: HashMap<u64, &Function> = funcs.iter().map(|f| (f.entry, f)).collect();
    // Local conditions, and the direct callees.
    let mut cand: HashMap<u64, (u32, Vec<u64>)> = HashMap::new();
    'f: for f in funcs {
        let Some(sm) = sums.get(&f.entry) else { continue };
        if sm.may_unwind || sm.may_leave_pending || sm.unknown_callee {
            continue;
        }
        let mut callees = Vec::new();
        for b in &f.blocks {
            if b.stmts.iter().any(|s| emit::clobbers(s) || matches!(s, Stmt::Trap { .. })) {
                continue 'f;
            }
            let backward = |e: &Edge| {
                let to = &f.blocks[e.to.index()];
                !to.synthetic && to.start <= b.start
            };
            match &b.term {
                FTerm::Jump(e) if !backward(e) => {}
                FTerm::Branch {
                    taken, not_taken, ..
                } if !backward(taken) && !backward(not_taken) => {}
                FTerm::Switch { cases, table, .. }
                    if table.complete && !cases.iter().any(|(_, e)| backward(e)) => {}
                FTerm::Call {
                    callee: Callee::Direct(t),
                    ret: Some(r),
                    ..
                } if by_entry.contains_key(t) && !backward(r) => callees.push(*t),
                FTerm::Return { .. } => {}
                _ => continue 'f,
            }
        }
        cand.insert(f.entry, (frame_weight(&registers(f)), callees));
    }
    // Callees trap-free too, no recursion, bounded stack: a depth-first
    // walk that memoises the subtree's stack and treats a cycle as failure.
    fn visit(
        a: u64,
        cand: &HashMap<u64, (u32, Vec<u64>)>,
        memo: &mut HashMap<u64, Option<u32>>,
        onstack: &mut HashSet<u64>,
    ) -> Option<u32> {
        if let Some(m) = memo.get(&a) {
            return *m;
        }
        if !onstack.insert(a) {
            return None; // recursion
        }
        let r = cand.get(&a).and_then(|(w, callees)| {
            let mut deepest = 0u32;
            for c in callees {
                deepest = deepest.max(visit(*c, cand, memo, onstack)? + CALL_KNOWN_WEIGHT);
            }
            Some(w + deepest).filter(|n| *n <= TRAPFREE_RESERVE)
        });
        onstack.remove(&a);
        memo.insert(a, r);
        r
    }
    let mut memo = HashMap::new();
    let mut onstack = HashSet::new();
    let keys: Vec<u64> = cand.keys().copied().collect();
    for a in keys {
        visit(a, &cand, &mut memo, &mut onstack);
    }
    memo.into_iter().filter_map(|(a, n)| n.map(|n| (a, n))).collect()
}

/// Outlining outermost loops is on unless `GEX_CGEN_OUTLINE=0`.
pub fn outline_loops() -> bool {
    match std::env::var("GEX_CGEN_OUTLINE").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        // The native flavor (`image::unguarded`): no tiering engine to
        // escape, so no loop functions.
        _ => !crate::image::unguarded(),
    }
}

/// An outlined loop's function. A leaf copy's (`FnUnit::leaves`) are
/// named apart: the original and its copy can share a translation unit,
/// and loop functions are `static`.
fn loop_name(entry: u64, k: usize, copy: bool) -> String {
    if copy {
        format!("ll_{entry:x}_{k}")
    } else {
        format!("l_{entry:x}_{k}")
    }
}

/// The body's view of its outlined loops.
pub(crate) struct Heads<'a> {
    loops: &'a [Loop],
    fname: u64,
    /// Emitting a leaf copy (`loop_name`).
    copy: bool,
    used: &'a [GuestReg],
}

/// The blocks a loop function emits.
pub(crate) struct Scope<'a> {
    body: &'a BTreeSet<usize>,
    header: usize,
    nested: bool,
}

/// An outermost natural loop: its header, its blocks, and the blocks outside
/// it that its edges reach.
pub struct Loop {
    pub header: usize,
    pub body: BTreeSet<usize>,
    pub exits: BTreeSet<usize>,
    /// It contains an inner loop: a poll that expires there marks the
    /// outer loop hot, and its back-edge hands over.
    pub nested: bool,
}

/// The outermost natural loops of `f`, for outlining: V8
/// never replaces a running frame's baseline code, so a loop that
/// runs for the whole of one call of a function called once (a `main`)
/// would never reach optimised code. Outlined, each chunk of iterations is a
/// new call. None when the CFG has a backward edge that is not a natural
/// loop's (irreducible): outlining needs the header to be the only way in.
pub fn outer_loops(f: &Function) -> Vec<Loop> {
    let n = f.blocks.len();
    let idom = f.idoms();
    let dominates = |h: usize, mut b: usize| -> bool {
        loop {
            if b == h {
                return true;
            }
            match idom[b] {
                Some(d) => b = d.index(),
                None => return false,
            }
        }
    };
    let preds = f.preds();
    let reachable: HashSet<usize> = f.rpo().iter().map(|b| b.index()).collect();
    let mut by_header: std::collections::BTreeMap<usize, BTreeSet<usize>> = Default::default();
    for (i, b) in f.blocks.iter().enumerate() {
        if !reachable.contains(&i) {
            continue;
        }
        for s in b.succs() {
            let h = s.index();
            if !dominates(h, i) {
                // Not a natural back-edge; a retreating edge in RPO would be
                // irreducible. Checked below by containment instead.
                continue;
            }
            // Natural loop of i -> h.
            let body = by_header.entry(h).or_default();
            body.insert(h);
            // Everything that reaches the latch without passing the header.
            let mut work = vec![i];
            while let Some(x) = work.pop() {
                if x == h {
                    continue; // a self-loop's latch is its header
                }
                body.insert(x);
                for p in &preds[x] {
                    if !body.contains(&p.index()) && reachable.contains(&p.index()) {
                        work.push(p.index());
                    }
                }
            }
        }
    }
    // Irreducible control flow (a cycle with no dominating header): leave
    // the function alone.
    let order: HashMap<usize, usize> = f.rpo().iter().enumerate().map(|(k, b)| (b.index(), k)).collect();
    for (i, b) in f.blocks.iter().enumerate() {
        let Some(&oi) = order.get(&i) else { continue };
        for s in b.succs() {
            if let Some(&os) = order.get(&s.index()) {
                if os <= oi && !dominates(s.index(), i) {
                    return Vec::new();
                }
            }
        }
    }
    // Outermost: a header no other loop's body contains.
    let headers: Vec<usize> = by_header.keys().copied().collect();
    let mut out = Vec::new();
    for &h in &headers {
        if headers.iter().any(|&o| o != h && by_header[&o].contains(&h)) {
            continue;
        }
        let body = by_header[&h].clone();
        let mut exits = BTreeSet::new();
        for &bi in &body {
            for s in f.blocks[bi].succs() {
                if !body.contains(&s.index()) {
                    exits.insert(s.index());
                }
            }
        }
        let nested = body.iter().any(|&bi| {
            let b = &f.blocks[bi];
            b.succs().iter().any(|s| {
                let t = s.index();
                t != h && body.contains(&t) && !f.blocks[t].synthetic && f.blocks[t].start <= b.start
            })
        });
        // `GEX_CGEN_HOT=0`: hand over only on the loop's own back-edge, for
        // pricing.
        let nested = nested && std::env::var("GEX_CGEN_HOT").as_deref() != Ok("0");
        out.push(Loop { header: h, body, exits, nested });
    }
    let _ = n;
    out
}

/// The guest function a C function belongs to, from its name: `f_<entry>`,
/// `e_<entry>` or a loop function `l_<entry>_<k>`.
pub fn guest_entry_of(name: &str) -> Option<u64> {
    let rest = name
        .strip_prefix("f_")
        .or_else(|| name.strip_prefix("e_"))
        .or_else(|| name.strip_prefix("l_").map(|r| r.split('_').next().unwrap_or(r)))?;
    u64::from_str_radix(rest, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0x1000`: a leaf with a loop (`x0 += 1` while below `x1`), then
    /// `ret`; `0x2000` calls it. The leaf is copied into the caller's unit
    /// (`FnUnit::leaves`), and with loop outlining on (the browser's guarded
    /// flavor) both the leaf and its copy outline the loop: their loop
    /// functions must be named apart (a C++ program's unit failed with
    /// "redefinition of 'l_419a70_0'").
    const LEAF: &str = "func 0x1000 - entries [] form memory values 12
b0 0x1000..0x1003 insts [1000]:
  @1000 jump b1()
b1 0x1003..0x1008 insts [1003,1005]:
  @1003 v0 = read x1
  @1003 v1 = read x0
  @1003 v2 = i64 0x1
  @1003 v3 = bin Add i64 v1 v2
  @1003 write x0 v3
  @1005 v4 = cmp LtU i64 v3 v0
  @1005 br v4 b1() b2()
b2 0x1008..0x1009 insts [1008]:
  @1008 v5 = read x4
  @1008 v6 = load B4 u i32 Relaxed v5 0
  @1008 v7 = i64 0x4
  @1008 v8 = bin Add i64 v5 v7
  @1008 write x4 v8
  @1008 v9 = extend u v6
  @1008 return v9
";
    const CALLER: &str = "func 0x2000 - entries [] form memory values 12
b0 0x2000..0x2005 insts [2000]:
  @2000 v0 = read x4
  @2000 v1 = i64 0x4
  @2000 v2 = bin Sub i64 v0 v1
  @2000 v3 = i32 0x2005
  @2000 store B4 Relaxed v2 0 v3
  @2000 write x4 v2
  @2000 call 0x1000 ret 0x2005 -> b1()
b1 0x2005..0x2006 insts [2005]:
  @2005 v4 = read x4
  @2005 v5 = load B4 u i32 Relaxed v4 0
  @2005 v6 = i64 0x4
  @2005 v7 = bin Add i64 v4 v6
  @2005 write x4 v7
  @2005 v8 = extend u v5
  @2005 return v8
";

    /// `0x3000`: a call-free leaf without a loop (`eax = [ecx + 20]`, `ret`).
    const GETTER: &str = "func 0x3000 - entries [] form memory values 12
b0 0x3000..0x3004 insts [3000,3003]:
  @3000 v0 = read x1
  @3000 v1 = load B4 u i32 Relaxed v0 20
  @3000 v2 = extend u v1
  @3000 write x0 v2
  @3003 v3 = read x4
  @3003 v4 = load B4 u i32 Relaxed v3 0
  @3003 v5 = i64 0x4
  @3003 v6 = bin Add i64 v3 v5
  @3003 write x4 v6
  @3003 v7 = extend u v4
  @3003 return v7
";
    /// `0x4000` calls it directly, then through a pointer in `edx`, hinted
    /// to 0x3000.
    const GETTER_CALLER: &str = "func 0x4000 - entries [] form memory values 20
b0 0x4000..0x4005 insts [4000]:
  @4000 v0 = read x4
  @4000 v1 = i64 0x4
  @4000 v2 = bin Sub i64 v0 v1
  @4000 v3 = i32 0x4005
  @4000 store B4 Relaxed v2 0 v3
  @4000 write x4 v2
  @4000 call 0x3000 ret 0x4005 -> b1()
b1 0x4005..0x4007 insts [4005]:
  @4005 v4 = read x2
  @4005 v5 = read x4
  @4005 v6 = i64 0x4
  @4005 v7 = bin Sub i64 v5 v6
  @4005 v8 = i32 0x4007
  @4005 store B4 Relaxed v7 0 v8
  @4005 write x4 v7
  @4005 call v4 ret 0x4007 -> b2()
b2 0x4007..0x4008 insts [4007]:
  @4007 v9 = read x4
  @4007 v10 = load B4 u i32 Relaxed v9 0
  @4007 v11 = i64 0x4
  @4007 v12 = bin Add i64 v9 v11
  @4007 write x4 v12
  @4007 v13 = extend u v10
  @4007 return v13
";

    #[test]
    fn a_call_free_leaf_is_inlined_with_a_fallback() {
        let leaf = gex_ir::text::parse(GETTER).unwrap();
        let caller = gex_ir::text::parse(GETTER_CALLER).unwrap();
        assert!(inlinable(&leaf));
        assert!(!inlinable(&gex_ir::text::parse(LEAF).unwrap()), "a loop is not inlined");
        if !gex_abi::X86 {
            return; // i386 only
        }
        let known: HashSet<u64> = [0x3000, 0x4000].into_iter().collect();
        let mut unit = FnUnit::new();
        unit.set_class32(true);
        unit.ftab = true;
        unit.inline_leaves = true;
        unit.leaves = std::sync::Arc::new([(0x3000, leaf.clone())].into_iter().collect());
        unit.icall_hints = std::sync::Arc::new([(0x4007, 0x3000)].into_iter().collect());
        unit.add(&leaf, &known, &HashMap::new(), &HashMap::new()).unwrap();
        unit.add(&caller, &known, &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(unit.stats.calls_inlined, 2);
        unit.add_leaf_copies(&known, &HashMap::new(), &HashMap::new()).unwrap();
        let c = unit.source();
        let body = &c[c.find("int f_4000(").unwrap()..];
        // The fallbacks: the direct call, and the indirect one through the
        // table (its target from MonAddr), not devirtualised again.
        assert!(body.contains("r_monaddr"), "the indirect fallback keeps its target in MonAddr:\n{body}");
        assert!(!body.contains("ft == 0x3000ull"), "the fallback is not devirtualised again:\n{body}");
        // The copies' returns: each compares the popped address with its own
        // continuation and goes on there; anything else leaves through the
        // shared exit, a table call on the popped address.
        assert!(body.contains("== v") && body.contains("= 0x4005ull;") && body.contains("= 0x4007ull;"), "{body}");
        assert!(body.contains("GEX_MUSTTAIL return GEX_FTAB(gex_ftab_idx(ft))(ctx, depth, need, ret);"), "{body}");
        assert!(!body.contains("case 0x4005ull:"), "no per-ret switch any more:\n{body}");
        let dir = std::env::temp_dir().join(format!("gex-cgen-inline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gex_abi.h"), crate::header::header()).unwrap();
        std::fs::write(dir.join("u.c"), &c).unwrap();
        crate::toolchain::compile_one(&dir, &dir.join("u.c"), &dir.join("u.o"), None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_leaf_copy_names_its_outlined_loops_apart() {
        let leaf = gex_ir::text::parse(LEAF).unwrap();
        let caller = gex_ir::text::parse(CALLER).unwrap();
        assert!(is_small_copyable(&leaf));
        // One call that returns to it: copyable too with SMALLCALLS.
        assert_eq!(is_small_copyable(&caller), exp("SMALLCALLS") || exp("SMALLCALLSALL"));
        if !outline_loops() {
            return;
        }
        let known: HashSet<u64> = [0x1000, 0x2000].into_iter().collect();
        let mut unit = FnUnit::new();
        unit.set_class32(true);
        unit.ftab = true;
        unit.leaves = std::sync::Arc::new([(0x1000, leaf.clone())].into_iter().collect());
        unit.add(&leaf, &known, &HashMap::new(), &HashMap::new()).unwrap();
        unit.add(&caller, &known, &HashMap::new(), &HashMap::new()).unwrap();
        unit.add_leaf_copies(&known, &HashMap::new(), &HashMap::new()).unwrap();
        let c = unit.source();
        assert!(c.contains("fl_1000(ctx"), "the call goes to the copy:\n{c}");
        assert!(c.contains("int l_1000_0(") && c.contains("int ll_1000_0("), "both outline the loop, apart:\n{c}");
        // And clang takes it (a redefinition is an error).
        let dir = std::env::temp_dir().join(format!("gex-cgen-leafloop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gex_abi.h"), crate::header::header()).unwrap();
        std::fs::write(dir.join("u.c"), &c).unwrap();
        crate::toolchain::compile_one(&dir, &dir.join("u.c"), &dir.join("u.o"), None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
    /// `0x5000`: one block of `n` SSE operations, vectors of each lane
    /// shape and scalars, each on the last's result; under MXCSR
    /// (`Expr::Mx`, as the lifter makes them under `gex_ir::mxcsr_exact`)
    /// when `mx`, else the operations alone (as it makes them without).
    fn sse_heavy(n: usize, mx: bool) -> String {
        let mut b = String::from("  @5000 v0 = read mxcsr\n  @5000 v1 = read q0\n  @5000 v2 = read q1\n  @5000 v3 = read x0\n  @5000 v4 = read x1\n");
        let (mut q, mut x, mut v) = (1, 3, 5);
        for k in 0..n {
            let e = match k % 6 {
                0 => format!("vlanebin Add F32 v{q} v2"),
                1 => format!("vlanebin Mul F64 v{q} v2"),
                2 => format!("vlaneun Sqrt F32 v{q}"),
                3 => format!("vcvt DemoteF64 F64 v{q} -"),
                4 => format!("fbin Div f64 v{x} v4"),
                _ => format!("fun Sqrt f64 v{x}"),
            };
            let e = if mx { format!("mx v0 {e}") } else { e };
            b += &format!("  @5000 v{v} = {e}\n");
            if k % 6 < 4 {
                q = v;
            } else {
                x = v;
            }
            v += 1;
        }
        b += &format!("  @5000 write q0 v{q}\n  @5000 write x0 v{x}\n  @5000 return v{x}\n");
        format!("func 0x5000 - entries [] form memory values {v}\nb0 0x5000..0x5001 insts [5000]:\n{b}")
    }

    /// The function tier's C of `sse_heavy(n, mx)`'s function.
    fn sse_heavy_c(n: usize, mx: bool) -> (String, String) {
        let f = gex_ir::text::parse(&sse_heavy(n, mx)).unwrap();
        let known: HashSet<u64> = [0x5000].into_iter().collect();
        let mut unit = FnUnit::new();
        unit.add(&f, &known, &HashMap::new(), &HashMap::new()).unwrap();
        let c = unit.source();
        let body = c[c.find(&format!("{}(", body_name(0x5000))).unwrap()..].to_string();
        (c, body)
    }

    /// Without MXCSR, an SSE operation's C is the operation's own and
    /// nothing else. Under it, a block's run of up to 32 of them tests the
    /// rounding control once and is the operations' own C or their exact calls, out
    /// of line in `runtime/mx.h` (an operation is one call however many
    /// lanes): no more than twice the C. A test and an inline slow path
    /// per operation was twice the C as well, but as thousands of branches
    /// in one function, which took clang over 32 GB.
    #[test]
    fn sse_operations_under_mxcsr_test_once_a_block() {
        if !gex_abi::X86 {
            return; // MXCSR is x86's
        }
        let n = 600;
        let (_, plain) = sse_heavy_c(n, false);
        assert!(!plain.contains("gex_mx"), "no MXCSR without it");
        let (c, body) = sse_heavy_c(n, true);
        let exact = body.matches("gex_mx_s(").count() + body.matches("gex_mx_v(").count();
        assert_eq!(exact, n, "one out-of-line exact call per operation");
        for inline in ["gex_mx(", "replace_lane", "extract_lane"] {
            assert!(!body.contains(inline), "`{inline}` is out of line, not in the function");
        }
        let tests = |c: &str| c.matches("__builtin_expect").count();
        assert_eq!(tests(&body), tests(&plain) + n.div_ceil(32), "one rounding-control test for each run of 32");
        let (with, without) = (body.len(), plain.len());
        assert!(with < 2 * without + 64 * n, "{with} bytes of C under MXCSR, {without} without");
        eprintln!("{with} bytes of C under MXCSR, {without} without ({n} operations)");
        let dir = std::env::temp_dir().join(format!("gex-cgen-mx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gex_abi.h"), crate::header::header()).unwrap();
        std::fs::write(dir.join("u.c"), &c).unwrap();
        crate::toolchain::compile_one(&dir, &dir.join("u.c"), &dir.join("u.o"), None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
