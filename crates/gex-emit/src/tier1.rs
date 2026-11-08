//! Tier-1 code generation: one wasm function per guest basic block.
//!
//! The shape: every block is a function with the uniform
//! signature `(ctx: i32) -> ()`; direct successors are reached with
//! `return_call`, computed ones with `return_call_indirect` through the
//! PC-indexed dispatch table. Guest stack depth therefore never shows up as
//! wasm stack depth, and any 4-byte-aligned address can become an entry point
//! later without retranslating anything.
//!
//! Two things make tier-1 more than a naive transliteration:
//!
//! * **Guest registers are cached in wasm locals** for the length of the
//!   block. A register is loaded from the context at most once and stored
//!   back at most once, at the block's exit.
//! * **Value locals are recycled** using last-use information, so a long
//!   block needs a handful of locals rather than one per SSA value.

use crate::wasm::{FuncBody, ValType};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use gex_abi::{ctx, DISPATCH_PAGE, GUEST_LIMIT, POISON_ADDR};
use gex_ir::*;
use std::collections::HashMap;

/// A multiply-rotate hasher for small keys (registers, addresses): the
/// standard SipHash was a few percent of translating at run time, and
/// nothing here hashes keys an adversary chooses.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl std::hash::Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u8(&mut self, n: u8) {
        self.write_u64(n as u64);
    }
    fn write_u16(&mut self, n: u16) {
        self.write_u64(n as u64);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64);
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7cc1_b727_220a_95);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    fn write_isize(&mut self, n: isize) {
        self.write_u64(n as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A `HashMap` with [`FastHasher`].
pub type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;

pub mod fmode;

/// SIMD sub-opcodes (the 0xFD prefix space).
mod simd {
    pub const V128_LOAD: u32 = 0x00;
    pub const V128_STORE: u32 = 0x0b;
    pub const I8X16_EXTRACT_LANE_S: u32 = 0x15;
    pub const I8X16_EXTRACT_LANE_U: u32 = 0x16;
    pub const I8X16_REPLACE_LANE: u32 = 0x17;
    pub const I16X8_EXTRACT_LANE_S: u32 = 0x18;
    pub const I16X8_EXTRACT_LANE_U: u32 = 0x19;
    pub const I16X8_REPLACE_LANE: u32 = 0x1a;
    pub const I32X4_EXTRACT_LANE: u32 = 0x1b;
    pub const I32X4_REPLACE_LANE: u32 = 0x1c;
    pub const I64X2_EXTRACT_LANE: u32 = 0x1d;
    pub const I64X2_REPLACE_LANE: u32 = 0x1e;
    pub const I8X16_SPLAT: u32 = 0x0f;
    pub const I16X8_SPLAT: u32 = 0x10;
    pub const I32X4_SPLAT: u32 = 0x11;
    pub const I64X2_SPLAT: u32 = 0x12;
    pub const I8X16_SWIZZLE: u32 = 0x0e;
    pub const V128_BITSELECT: u32 = 0x52;
    pub const I8X16_POPCNT: u32 = 0x62;
    pub const I8X16_BITMASK: u32 = 0x64;
    pub const I16X8_BITMASK: u32 = 0x84;
    pub const I32X4_BITMASK: u32 = 0xa4;
    pub const I64X2_BITMASK: u32 = 0xc4;
    pub const I16X8_EXTADD_PAIRWISE_I8X16_S: u32 = 0x7c;
    pub const I16X8_EXTADD_PAIRWISE_I8X16_U: u32 = 0x7d;
    pub const I32X4_EXTADD_PAIRWISE_I16X8_S: u32 = 0x7e;
    pub const I32X4_EXTADD_PAIRWISE_I16X8_U: u32 = 0x7f;
    pub const V128_AND: u32 = 0x4e;
    pub const V128_ANDNOT: u32 = 0x4f;
    pub const V128_OR: u32 = 0x50;
    pub const V128_XOR: u32 = 0x51;
    // Only the narrow lane loads are used: 4- and 8-byte loads have
    // dedicated zeroing forms, which are one instruction instead of two.
    pub const V128_LOAD8_LANE: u32 = 0x54;
    pub const V128_LOAD16_LANE: u32 = 0x55;
    pub const V128_STORE8_LANE: u32 = 0x58;
    pub const V128_STORE16_LANE: u32 = 0x59;
    pub const V128_STORE32_LANE: u32 = 0x5a;
    pub const V128_STORE64_LANE: u32 = 0x5b;
    pub const V128_LOAD32_ZERO: u32 = 0x5c;
    pub const V128_LOAD64_ZERO: u32 = 0x5d;
}

/// The wasm value type backing a GIR type.
fn val_type(ty: Ty) -> ValType {
    match ty {
        Ty::I32 => ValType::I32,
        Ty::I64 => ValType::I64,
        Ty::V128 => ValType::V128,
    }
}

// Numeric opcodes, grouped so the i32/i64 pairs stay visibly adjacent.
mod code {
    pub const I32_WRAP_I64: u8 = 0xa7;
    pub const I64_EXTEND_I32_S: u8 = 0xac;
    pub const I64_EXTEND_I32_U: u8 = 0xad;
    pub const I32_EXTEND8_S: u8 = 0xc0;
    pub const I32_EXTEND16_S: u8 = 0xc1;
    pub const I64_EXTEND8_S: u8 = 0xc2;
    pub const I64_EXTEND16_S: u8 = 0xc3;
    pub const I64_EXTEND32_S: u8 = 0xc4;
    pub const I64_LT_U: u8 = 0x54;
    pub const I64_GE_U: u8 = 0x5a;
    pub const I64_SUB: u8 = 0x7d;
    pub const I32_EQZ: u8 = 0x45;
    pub const I32_EQ: u8 = 0x46;
    pub const I32_LE_U: u8 = 0x4d;
    pub const I32_SUB: u8 = 0x6b;
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_SHR_U: u8 = 0x76;
    pub const I64_EQZ: u8 = 0x50;
    pub const I64_EQ: u8 = 0x51;
    pub const I64_NE: u8 = 0x52;
    pub const I64_LT_S: u8 = 0x53;
    pub const I64_GT_U: u8 = 0x56;
    pub const I64_ADD: u8 = 0x7c;
    pub const I64_AND: u8 = 0x83;
    pub const I64_OR: u8 = 0x84;
    pub const I64_SHL: u8 = 0x86;
    pub const I64_SHR_U: u8 = 0x88;
    pub const I64_XOR: u8 = 0x85;
    /// Second byte of the `0xfc` bulk-memory pair, after the prefix.
    pub const MEMORY_COPY: u32 = 10;
    pub const MEMORY_FILL: u32 = 11;
}

/// The wasm type of a pointer into guest memory -- the memory's index type.
pub fn ptr_ty() -> crate::wasm::ValType {
    crate::wasm::ValType::I32
}

/// Everything tier-1 needs to know about the module it is emitting into.
pub struct Context<'a> {
    /// Import index of `gex.syscall`.
    pub syscall: u32,
    /// Import index of `gex.illegal`.
    pub illegal: u32,
    /// Import index of `gex.breakpoint`.
    pub breakpoint: u32,
    /// Import index of `gex.sysreg_read`.
    pub sysreg_read: u32,
    /// Import index of `gex.fma`.
    pub fma: u32,
    /// Import index of `gex.x87`.
    pub x87: u32,
    /// Import index of `gex.mx`.
    pub mx: u32,
    /// Import index of `gex.sysreg_write`.
    pub sysreg_write: u32,
    /// Import index of `gex.cache_op`.
    pub cache_op: u32,
    /// Import index of `gex.dispatch_miss`.
    pub dispatch_miss: u32,
    /// Import index of `gex.safepoint`.
    pub safepoint: u32,
    /// Type index of the uniform block signature `(i32) -> ()`.
    pub block_ty: u32,
    /// Type index of `(ctx, arm) -> ()`, the type of `blocks`' entries
    /// (`crate::dense`); unused where dense dispatch is off.
    pub entry_ty: u32,
    /// Guest address -> wasm function index, for blocks already assigned an
    /// index. A miss falls back to the dispatch table.
    pub blocks: &'a HashMap<u64, u32>,
    /// Guest address -> the block's GIR, for growing a region around a loop.
    pub gir: &'a HashMap<u64, &'a Block>,
    /// Profiling instrumentation for this module, when it is on (see
    /// `crate::profile`). `None` emits exactly what an unprofiled build does.
    pub profile: Option<&'a crate::profile::Emit>,
    /// Each page's directory entry as this module numbered it. A block
    /// checks its page's entry on entry ([`Emitter::emit_page_check`]): one
    /// that was retired (zeroed) or translated again (a fresh base -- bases
    /// are never reused) since means the block is stale, however it was
    /// reached. `None`: no check (a module that is never retired).
    pub page_bases: Option<&'a HashMap<u64, u32>>,
    /// `TranslatedModule::module_id` of the module being built.
    pub module_id: u64,
    /// The global holding it instead, set by the host at instantiation
    /// (`crate::module::with_module_id_global`): the bytes then do not
    /// depend on which id the module was given.
    pub module_id_global: Option<u32>,
    /// The module's imported dense base (`crate::dense::GLOBAL`), when it
    /// has one: `PAGE_STALE` passes its value, where this module's dense
    /// blocks really are in the process running it.
    pub dense_global: Option<u32>,
    /// The module's imported other directory ([`gex_abi::imports::ALT_DIR`])
    /// when its code is in the process's other mode: every directory lookup
    /// reads that one.
    pub alt_dir: Option<u32>,
    /// Every address some block of `gir` branches back to (a successor at or
    /// below the branching block's start). A region needs a back edge to its
    /// entry, so a block not in here grows none and the walk is skipped.
    /// `None`: no filter.
    pub loop_heads: Option<&'a std::collections::HashSet<u64>>,
    /// The module's `dispatch(ctx, pc)` (a local function, relocated: `enter`
    /// without its context stores), when it has one: an indirect dispatch
    /// tail-calls it instead of carrying the whole table selection inline.
    /// See [`set_inline_dispatch`].
    pub enter: Option<u32>,
}

/// `GEX_INLINE_DISPATCH=1` natively, `set_inline_dispatch(true)` in the
/// browser: every dispatch carries the directory lookup and the selection over
/// the eight tables inline, as before, instead of tail-calling the module's
/// `enter`. Inline, dispatch was 40% of a module's bytes (`gexc build` of
/// Debian's xz: 687,595 of 1,228 KiB were dispatch and boundary code), on a
/// path that a direct tail call reaches for the price of one call.
static INLINE_DISPATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_inline_dispatch(on: bool) {
    INLINE_DISPATCH.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn inline_dispatch() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_INLINE_DISPATCH").is_some())
        || INLINE_DISPATCH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fold `local.set x; local.get x` into `local.tee x` and strip the tees
/// of temporaries nothing reads (the function JIT's peephole, `fmode`) in
/// tier-1 blocks too: `set_t1_fold` / GEX_T1_FOLD.
static T1_FOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_t1_fold(on: bool) {
    T1_FOLD.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn t1_fold() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_T1_FOLD").is_some()) || T1_FOLD.load(std::sync::atomic::Ordering::Relaxed)
}

/// The emitter switches that change tier-1's bytes without showing in the
/// translation options, for a cache key (GEX_INLINE_DISPATCH, GEX_T1_FOLD,
/// GEX_NULL_GUARD, GEX_MXCSR):
/// natively they are read from the environment, which the module cache's key
/// did not see, so a run with one set served its modules to runs without.
pub fn knobs() -> u8 {
    inline_dispatch() as u8 | (t1_fold() as u8) << 1 | (gex_abi::kuser_fold() as u8) << 2 | (null_guard_on() as u8) << 3
        | (gex_ir::mxcsr_exact() as u8) << 4
}

/// Whether the translators put the null guard (`gex_abi::NULL_GUARD`) on
/// guest accesses: a debugging opt-in, `gex_abi::set_null_guard` (the
/// page's `?nullguard=1`) or GEX_NULL_GUARD=1 natively, and never with
/// GEX_UNSAFE_NO_GUARD. Off, tier-1's and the C units' output is
/// byte-identical to a build without it, and their keys do not name it.
pub fn null_guard_on() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| std::env::var_os("GEX_NULL_GUARD").is_some_and(|v| v != "0"));
    (env || gex_abi::null_guard()) && !Emitter::no_guard()
}

/// The end of a tier-1 block's function with [`t1_fold`]: its dead tees
/// of temporaries removed.
fn strip_tees(e: &mut Emitter) {
    if !t1_fold() {
        return;
    }
    let n = e.temps.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut temp = vec![false; n];
    for &t in &e.temps {
        temp[t as usize] = true;
    }
    e.f.strip_dead_tees(&|l| temp.get(l as usize).copied().unwrap_or(false), true);
}

/// The back-edge targets of `gir`, for [`Context::loop_heads`].
pub fn loop_heads<'b>(blocks: impl Iterator<Item = &'b Block>) -> std::collections::HashSet<u64> {
    let mut out = std::collections::HashSet::new();
    for b in blocks {
        for s in b.successors() {
            if s <= b.start {
                out.insert(s);
            }
        }
    }
    out
}

/// The region grown at `block`, if any (and regions are on).
/// The region `block` would head, if any (diagnostics: t1sweep).
pub fn region_members(block: &Block, cx: &Context<'_>) -> Option<Vec<u64>> {
    region_at(block, cx)
}

fn region_at(block: &Block, cx: &Context<'_>) -> Option<crate::region::Region> {
    if region_limit() <= 1 || cx.loop_heads.is_some_and(|h| !h.contains(&block.start)) {
        return None;
    }
    crate::region::grow(block.start, cx.gir, region_limit())
}

/// `GEX_NO_PAGE_CHECK=1`: blocks do not check their page's directory entry
/// on entry (for pricing, and as a mutation the page-check tests must catch).
fn page_check() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| std::env::var_os("GEX_NO_PAGE_CHECK").is_some())
}

/// `GEX_PAGE_CHECK_NO_RETRANSLATE=1`: a block failing its page check only
/// dispatches (a mutation the page-span tests must catch).
fn page_check_no_retranslate() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEX_PAGE_CHECK_NO_RETRANSLATE").is_some())
}

/// `GEX_PAGE_CHECK_BALLAST=N`: the page check emitted N extra times, for
/// pricing it (the slope over N is its cost).
fn page_check_ballast() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("GEX_PAGE_CHECK_BALLAST").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// The pages `[start, end)` touches that the module numbered, with their
/// entries.
fn checked_pages(bases: &HashMap<u64, u32>, start: u64, end: u64, alt: bool) -> Vec<(u64, u32)> {
    let first = gex_abi::page_key(start, alt);
    let last = gex_abi::page_key(end.max(start + 1).saturating_sub(1), alt);
    (first..=last).filter_map(|p| bases.get(&p).map(|&b| (p, b))).collect()
}


/// What a misaligned locked access does (`Emitter::emit_misaligned_atomic`),
/// with the locals holding its operands.
#[derive(Clone, Copy)]
enum Misaligned {
    Cas { expected: u32, replacement: u32 },
    Rmw { op: RmwOp, val: u32 },
}

/// How many blocks may go into one region. One turns regions off.
fn region_limit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("GEX_REGION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Whether every block should check the guest's stack pointer on entry.
///
/// Off unless asked for: `GEX_CHECK_SP=1` natively, or `gex_set_check_sp(1)` in
/// the browser. Costs a load, a compare and a never-taken branch per block. It
/// exists because a corrupt `sp` is invisible until something dereferences it,
/// by which point the crash names the victim rather than the writer.
static CHECK_SP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the stack-pointer check on or off for blocks compiled after this.
pub fn set_check_sp(on: bool) {
    CHECK_SP.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn check_sp() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_CHECK_SP").is_some())
        || CHECK_SP.load(std::sync::atomic::Ordering::Relaxed)
}

/// `GEX_COPY_LOOP=1` natively, or `set_copy_loop(true)` in the browser: emit
/// `MemCopy` as an explicit forward byte loop instead of `memory.copy`, to tell
/// a `memory.copy`/clamp fault from a bad-argument one. Diagnostic only --
/// forward copy is not memmove-safe.
static COPY_LOOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_copy_loop(on: bool) {
    COPY_LOOP.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn copy_loop() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_COPY_LOOP").is_some())
        || COPY_LOOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Experiment (default OFF): emit guest scalar loads/stores as *atomic* wasm
/// accesses when the effective address is naturally aligned, plain otherwise.
/// A plain wasm load from shared memory can be hoisted/CSE'd by V8 across the
/// atomic fences gex emits for guest acquire/release/DMB, so a correctly
/// synchronised guest (GLib's lock-free fast paths) can still read a stale
/// pointer/count and store wild. An atomic access is a real, un-reorderable
/// memory op -- this makes shared reads coherent. Aligned-only because an
/// unaligned wasm atomic traps; the runtime alignment check picks the plain
/// form for the (rare, usually thread-private) unaligned case. Gated so the
/// default hot path is byte-identical -- zero regression risk when off.
static COHERENT_LOADS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_coherent_loads(on: bool) {
    COHERENT_LOADS.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn coherent_loads() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("GEX_COHERENT_LOADS").is_some())
        || COHERENT_LOADS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A single guest pc (block start) to plant a breakpoint call at, or 0.
///
/// Diagnostic: when the block heading `pc` is entered, translated code calls the
/// `breakpoint` import with `BRK_TRAP_PC`, which dumps the live registers (x30
/// names the caller) and RETURNS so execution carries on. Pointed at
/// `__stack_chk_fail`, it names the function whose canary was smashed -- the one
/// thing a downstream abort cannot recover. `GEX_TRAP_PC=0x50fbde0` natively or
/// `gex_set_trap_pc` in the browser.
static TRAP_PC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_trap_pc(pc: u64) {
    TRAP_PC.store(pc, std::sync::atomic::Ordering::Relaxed);
}

/// Where the bounds guard puts the end of the guest's memory, for blocks
/// translated from now on: `GUEST_LIMIT` unless a host lowers it. A Windows
/// process in the browser owns only the low 2 GiB -- the shim's heap is
/// above it -- so there it is 2 GiB, and a guest access above is a fault
/// rather than a write into the runtime.
static GUARD_LIMIT: AtomicU64 = AtomicU64::new(GUEST_LIMIT);

pub fn set_guard_limit(limit: u64) {
    GUARD_LIMIT.store(limit.min(GUEST_LIMIT), Ordering::Relaxed);
}

fn guard_limit() -> u64 {
    let g = GUARD_LIMIT.load(Ordering::Relaxed);
    // 32-bit Wine with KUSER_SHARED_DATA folded: the old page and above fault by name.
    if gex_abi::kuser_fold() {
        g.min(gex_abi::KUSER_GUARD)
    } else {
        g
    }
}

/// More pcs for the same breakpoint, beside [`set_trap_pc`]'s: to watch a
/// call and its return together.
static TRAP_PCS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

pub fn add_trap_pc(pc: u64) {
    for s in &TRAP_PCS {
        if s.compare_exchange(0, pc, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
            return;
        }
    }
}

fn is_trap_pc(pc: u64) -> bool {
    trap_pc() == pc || TRAP_PCS.iter().any(|s| s.load(std::sync::atomic::Ordering::Relaxed) == pc)
}

/// `GEX_TRACE_BLOCKS`: plant a trace point at every block entry, so the host
/// can print the path a guest takes. A diagnostic for a run that ends
/// somewhere it should not, with no other record of how it got there. Costs
/// a host call per block, so never on by default.
/// Set by a host without an environment (the browser shim, via
/// `gex_set_trace_blocks`); the native host sets `GEX_TRACE_BLOCKS`.
static TRACE_BLOCKS_SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the per-block register trace on for everything translated from now
/// on. Blocks already emitted are unaffected.
pub fn set_trace_blocks(on: bool) {
    TRACE_BLOCKS_SET.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn trace_blocks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    TRACE_BLOCKS_SET.load(std::sync::atomic::Ordering::Relaxed)
        || *ON.get_or_init(|| std::env::var_os("GEX_TRACE_BLOCKS").is_some())
}

/// `GEX_TRACE_RANGE=LO-HI` (hex): trace every block starting in the range
/// (its translation, and each execution through `BRK_TRACE`).
fn in_trace_range(pc: u64) -> bool {
    static R: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    let r = R.get_or_init(|| {
        let v = std::env::var("GEX_TRACE_RANGE").ok()?;
        let (a, b) = v.split_once('-')?;
        let h = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        Some((h(a)?, h(b)?))
    });
    r.is_some_and(|(a, b)| pc >= a && pc < b)
}

fn trap_pc() -> u64 {
    static ENV: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let e = *ENV.get_or_init(|| {
        std::env::var("GEX_TRAP_PC")
            .ok()
            .and_then(|v| {
                let v = v.trim();
                let v = v.strip_prefix("0x").unwrap_or(v);
                u64::from_str_radix(v, 16).ok()
            })
            .unwrap_or(0)
    });
    if e != 0 {
        e
    } else {
        TRAP_PC.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Compile one GIR block into a wasm function body, plus the offsets of every
/// immediate naming a local function.
///
/// When the block heads a loop, the loop's other blocks come with it: see
/// [`Emitter::run_region`]. The reloc offsets let `gex_emit::merge` patch those
/// immediates without decoding the stream; they are written at a fixed width so
/// a patch cannot change a body's length.
pub fn compile_with_relocs(block: &Block, cx: &Context<'_>) -> (Vec<u8>, Vec<u32>) {
    let mut e = Emitter::new(cx, block.start);
    if t1_fold() {
        e.f.fold_tees();
    }
    let region = region_at(block, cx);
    // First, before the page check: a block entered for a neighbour in its
    // dispatch slot (x86-64's 4-byte slots) must hand over to that address.
    // The page check's stale path (and the guard's) dispatch to this block's
    // own start, which for a neighbour's branch is the wrong address: a call
    // to a JIT trampoline at E ran a stale block for E+2 there, mid-
    // instruction (crashes in Mono's JIT trampolines).
    if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
        e.emit_target_check(block.start);
    }
    if let Some(bases) = cx.page_bases.filter(|_| page_check()) {
        let mut pages = checked_pages(bases, block.start, block.end, cx.alt_dir.is_some());
        // A region carries its members: their pages too.
        if let Some(region) = &region {
            for a in region.iter() {
                if let Some(b) = cx.gir.get(a) {
                    pages.extend(checked_pages(bases, b.start, b.end, cx.alt_dir.is_some()));
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        for _ in 0..=page_check_ballast() {
            e.emit_page_check(block.start, &pages);
        }
    }
    if check_sp() {
        e.emit_sp_check(block.start);
    }
    if is_trap_pc(block.start) {
        e.emit_trap_pc(block.start);
    }
    if trace_blocks() || in_trace_range(block.start) {
        e.emit_trace_point(block.start);
    }
    if in_trace_range(block.start) {
        eprintln!(
            "[t1] translate {:#x}..{:#x} guard {} term {:?} bytes {}",
            block.start,
            block.end,
            block.guard.as_ref().map_or(0, |g| g.len()),
            block.term,
            block.guard.as_ref().map_or(String::new(), |g| g.iter().take(64).map(|b| format!("{b:02x}")).collect::<String>())
        );
    }
    if let Some(g) = &block.guard {
        e.emit_guard(block.start, g);
    }
    run_one(&mut e, block, region);
    strip_tees(&mut e);
    e.close_kuser_fault();
    e.f.finish_with_relocs()
}

pub fn compile(block: &Block, cx: &Context<'_>) -> Vec<u8> {
    let mut e = Emitter::new(cx, block.start);
    if t1_fold() {
        e.f.fold_tees();
    }
    let region = region_at(block, cx);
    // First, before the page check: a block entered for a neighbour in its
    // dispatch slot (x86-64's 4-byte slots) must hand over to that address.
    // The page check's stale path (and the guard's) dispatch to this block's
    // own start, which for a neighbour's branch is the wrong address: a call
    // to a JIT trampoline at E ran a stale block for E+2 there, mid-
    // instruction (crashes in Mono's JIT trampolines).
    if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
        e.emit_target_check(block.start);
    }
    if let Some(bases) = cx.page_bases.filter(|_| page_check()) {
        let mut pages = checked_pages(bases, block.start, block.end, cx.alt_dir.is_some());
        // A region carries its members: their pages too.
        if let Some(region) = &region {
            for a in region.iter() {
                if let Some(b) = cx.gir.get(a) {
                    pages.extend(checked_pages(bases, b.start, b.end, cx.alt_dir.is_some()));
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        for _ in 0..=page_check_ballast() {
            e.emit_page_check(block.start, &pages);
        }
    }
    if check_sp() {
        e.emit_sp_check(block.start);
    }
    if is_trap_pc(block.start) {
        e.emit_trap_pc(block.start);
    }
    if trace_blocks() || in_trace_range(block.start) {
        e.emit_trace_point(block.start);
    }
    if in_trace_range(block.start) {
        eprintln!(
            "[t1] translate {:#x}..{:#x} guard {} term {:?} bytes {}",
            block.start,
            block.end,
            block.guard.as_ref().map_or(0, |g| g.len()),
            block.term,
            block.guard.as_ref().map_or(String::new(), |g| g.iter().take(64).map(|b| format!("{b:02x}")).collect::<String>())
        );
    }
    if let Some(g) = &block.guard {
        e.emit_guard(block.start, g);
    }
    run_one(&mut e, block, region);
    strip_tees(&mut e);
    e.close_kuser_fault();
    e.f.finish()
}

/// The body of both entry points: a region if one can be grown here,
/// otherwise the block on its own.
fn run_one(e: &mut Emitter<'_>, block: &Block, region: Option<crate::region::Region>) {
    match region {
        Some(region) => {
            if std::env::var_os("GEX_REGION_DEBUG").is_some() {
                let members: Vec<String> = region.iter().map(|a| format!("{a:#x}")).collect();
                eprintln!("[region] {}", members.join(" "));
            }
            {
                let mut r = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
                r.0 += 1;
                r.1 += region.len();
            }
            e.run_region(&region)
        }
        _ => e.run(block),
    }
}

/// Where a guest register currently lives.
struct Cached {
    local: u32,
    dirty: bool,
}

struct Emitter<'a> {
    f: FuncBody,
    cx: &'a Context<'a>,
    /// The `ctx` parameter.
    ctx_local: u32,
    /// The guest address this block starts at, for `ctx::LAST_SOURCE`.
    block_start: u64,
    /// Scratch for address computation.
    scratch_i64: u32,
    /// Two addresses for a bulk memory op under a memory64 guest memory,
    /// whose addresses are i64 and so cannot borrow the
    /// i32 scratches the wasm32 path uses. Zero, and never declared, unless
    /// the memory is 64-bit: a wasm32 function keeps exactly its locals.
    scratch_addr: (u32, u32),
    /// Scratch for the safepoint counter.
    scratch_i32: u32,
    /// Scratch for the dispatch page index.
    scratch_page: u32,
    /// Scratch for a directory entry, and then for the table it names.
    ///
    /// Only used when there is more than one dispatch table; with one, the
    /// entry is consumed on the stack and never needs a name.
    scratch_entry: u32,
    /// Scratch for the computed slot, held across the `br_table`.
    ///
    /// A value pushed outside a `block` cannot be consumed inside one, so the
    /// index has to be in a local before the selection starts.
    scratch_index: u32,
    /// Locals holding live GIR values.
    value_local: Vec<Option<u32>>,
    /// Constant value of each GIR value, when known.
    value_const: Vec<Option<u64>>,
    /// Function mode only: the values defined by a zero-extension from 32
    /// bits, as addresses below 2^32 that need no bounds guard
    /// (`emit_address`). Empty elsewhere.
    pub(crate) value_u32: Vec<bool>,
    /// Of those, the ones a load or register read feeds (fmode's `block`).
    pub(crate) value_chase: Vec<bool>,
    /// The values that are the thread's context address plus offsets
    /// (`GuestReg::CtxAddr`: the x87 stack, `EFLAGS_SYS`). The context is
    /// below the null guard (`gex_abi::NULL_GUARD`), so these accesses skip
    /// it. Indexed like `value_const`; every `def` sets its entry.
    pub(crate) value_ctx: Vec<bool>,
    /// The locals `alloc` made: block-local temporaries, never holding a
    /// value from one block to another (fmode's `strip_dead_tees`).
    pub(crate) temps: Vec<u32>,
    /// Type of each GIR value, needed to pick the right local pool.
    value_ty: Vec<Ty>,
    /// Free locals available for reuse, by type.
    free_i32: Vec<u32>,
    free_i64: Vec<u32>,
    free_v128: Vec<u32>,
    /// Guest register cache.
    regs: FastMap<GuestReg, Cached>,
    /// Set while emitting a block fused into a `wasm` loop: how many nested
    /// `if`s sit between here and the loop label, so a back-edge knows what
    /// depth to branch to. `None` when the block is not fused.
    loop_depth: Option<u32>,
    /// Counts down to the next re-entry through the dispatcher, inside a
    /// fused loop. See [`Self::emit_loop_safepoint`].
    reenter_local: Option<u32>,
    /// Set while emitting a member of a region.
    region: Option<RegionState>,
    /// Which registers the region writes, and so has to store on the way out.
    /// The ones it only reads are already correct in the context.
    region_writes: std::collections::HashSet<GuestReg, std::hash::BuildHasherDefault<FastHasher>>,
    /// Function mode (`fmode`, the function JIT): every register the
    /// function uses has a local for its whole body, bound up front, and a
    /// clobbering host call reloads them all rather than forgetting them.
    func_regs: Option<Vec<GuestReg>>,
    /// Function mode: OR-ed into a trap's pc (the fault import's `pc`) when
    /// the code goes on after the import returns, by leaving the function
    /// (returning 0: the caller goes on at the context's PC); 0 when the
    /// host never returns from a fault.
    resumable_pc: u64,
    /// Function mode, with `resumable_pc`: where the statement being
    /// emitted carries on when it is a fault the host may deliver in place
    /// (`int3`'s or `ud2`'s `HostCall`): the next instruction's pc.
    fault_next: Option<u64>,
    /// Function mode, with `resumable_pc`: the pc of the instruction the
    /// statement being emitted is of, where a fault it raises in place is
    /// (with the registers written back there).
    stmt_pc: Option<u64>,
    /// Function mode: the local the next `def` leaves its value in (a
    /// register's, when a `WriteReg` of the value follows at once). Taken
    /// by `def`'s common tail; left set by a definition that has no local.
    def_into: Option<u32>,
    /// Function mode: the next `def` (an FP operation or a lane extract)
    /// keeps its value as a float, for float consumers only (`value_fp`).
    fp_result: bool,
    /// Function mode: values held as floats rather than their bit pattern:
    /// `Some(double)`. Read as integers they are reinterpreted.
    value_fp: Vec<Option<bool>>,
    free_f32: Vec<u32>,
    free_f64: Vec<u32>,
    /// A bounds guard branched to the shared KUSER fault: the body still
    /// needs [`Self::close_kuser_fault`].
    kuser_fault: bool,
}

/// What a region member needs to know to branch to its neighbours.
struct RegionState {
    /// Guest address -> arm of the dispatch table.
    index: HashMap<u64, u32>,
    /// The local holding which arm to run next.
    idx_local: u32,
    /// How many labels lie between here and the region's `loop`.
    depth: u32,
}

impl<'a> Emitter<'a> {
    fn new(cx: &'a Context<'a>, block_start: u64) -> Emitter<'a> {
        Self::with_params(cx, block_start, 1)
    }

    /// An emitter for a function of `n_params` parameters, `ctx` first.
    fn with_params(cx: &'a Context<'a>, block_start: u64, n_params: u32) -> Emitter<'a> {
        let mut f = FuncBody::new(n_params);
        let scratch_i64 = f.add_local(ValType::I64);
        let scratch_i32 = f.add_local(ValType::I32);
        let scratch_page = f.add_local(ValType::I64);
        let scratch_entry = f.add_local(ValType::I32);
        let scratch_index = f.add_local(ValType::I32);
        let scratch_addr = if gex_abi::mem64() {
            (f.add_local(ValType::I64), f.add_local(ValType::I64))
        } else {
            (0, 0)
        };
        Emitter {
            scratch_addr,
            f,
            cx,
            ctx_local: 0,
            block_start,
            scratch_i64,
            scratch_i32,
            scratch_page,
            scratch_entry,
            scratch_index,
            value_local: Vec::new(),
            value_const: Vec::new(),
            value_u32: Vec::new(),
            value_chase: Vec::new(),
            value_ctx: Vec::new(),
            temps: Vec::new(),
            value_ty: Vec::new(),
            free_i32: Vec::new(),
            free_i64: Vec::new(),
            free_v128: Vec::new(),
            regs: FastMap::default(),
            loop_depth: None,
            reenter_local: None,
            region: None,
            region_writes: Default::default(),
            func_regs: None,
            resumable_pc: 0,
            fault_next: None,
            stmt_pc: None,
            def_into: None,
            fp_result: false,
            value_fp: Vec::new(),
            free_f32: Vec::new(),
            free_f64: Vec::new(),
            kuser_fault: false,
        }
    }

    /// Emit a whole loop as one function.
    ///
    /// The members become the arms of a `br_table` inside a `loop`, so an edge
    /// between them is a branch and an assignment rather than a tail call, and
    /// every guest register the loop touches stays in a local from the top
    /// until control leaves. Laid out as the usual nest: the table is in the
    /// innermost block, and breaking out of the *k*th lands on the *k*th body.
    ///
    /// The emitter walks the members in one order and the guest reaches them in
    /// another, so which registers are *dirty* is not knowable here: every exit
    /// writes all of them, and so does the safepoint -- which deliberately does
    /// not read them back, since the host only changes a register on a path
    /// that does not return here (reading them back made a loop with a
    /// `continue` and a `break` never finish). Only leaf loops qualify: a call
    /// in the body ends the region.
    fn run_region(&mut self, region: &[u64]) {
        let n = region.len();
        let index: HashMap<u64, u32> = region
            .iter()
            .enumerate()
            .map(|(i, a)| (*a, i as u32))
            .collect();

        // Every register the loop touches starts in a local holding the
        // context's value -- even one the loop only writes, since an exit
        // reached before the write still stores the local. The two sets are
        // for the way out: a register only read needs no store on exit.
        let (mut reads, mut writes): (Vec<GuestReg>, Vec<GuestReg>) = (Vec::new(), Vec::new());
        for a in region {
            let (r, w) = registers_read_and_written(self.cx.gir[a]);
            for x in r {
                if !reads.contains(&x) {
                    reads.push(x);
                }
            }
            for x in w {
                if !writes.contains(&x) {
                    writes.push(x);
                }
            }
        }
        self.region_writes = writes.iter().copied().collect();
        let mut touched = reads;
        for w in writes {
            if !touched.contains(&w) {
                touched.push(w);
            }
        }
        touched.sort_by_key(|r| Self::reg_offset(*r));
        if std::env::var_os("GEX_REGION_DEBUG").is_some() {
            let insts: u64 = region
                .iter()
                .map(|a| (self.cx.gir[a].end - self.cx.gir[a].start) / 4)
                .sum();
            eprintln!(
                "[region] {} blocks, {} instructions, {} registers held",
                n,
                insts,
                touched.len()
            );
        }
        for r in touched {
            self.reg_local(r);
        }

        let idx = self.alloc(Ty::I32);
        self.f.i32_const(0);
        self.f.local_set(idx);
        let counter = self.alloc(Ty::I32);
        self.f.i32_const(loop_poll());
        self.f.local_set(counter);
        self.reenter_local = Some(counter);

        self.f.loop_void();
        for _ in 0..n {
            self.f.block_void();
        }
        self.f.local_get(idx);
        // Inside the innermost block, member k's label is n-1-k deep.
        let targets: Vec<u32> = (0..n).map(|k| (n - 1 - k) as u32).collect();
        self.f.br_table(&targets, targets[0]);
        self.f.end();

        // Bodies come out innermost first: the last member's body sits just
        // inside the second-innermost block.
        for k in (0..n).rev() {
            let block = self.cx.gir[&region[k]];
            // Depth from here back to the loop: k blocks still open.
            self.region = Some(RegionState {
                index: index.clone(),
                idx_local: idx,
                depth: k as u32,
            });
            self.run_member(block);
            self.region = None;
            if k > 0 {
                self.f.end();
            }
        }
        self.f.end(); // loop
                      // Every member either branches back to the table or leaves the
                      // function, so falling out of the loop cannot happen.
        self.f.unreachable();
    }

    /// Write every register the region writes back to the context -- all of
    /// them, not the dirty ones, since dirtiness depends on emission order and
    /// the guest reaches the members in another. Costs stores only on exit.
    fn flush_region_exit(&mut self) {
        let mut all: Vec<(GuestReg, u32)> = self
            .regs
            .iter()
            .filter(|(r, _)| self.region_writes.contains(*r))
            .map(|(r, c)| (*r, c.local))
            .collect();
        all.sort_by_key(|(r, _)| Self::reg_offset(*r));
        // Same packing as `flush_inner`: a cached flag holds the current value
        // whether or not this region wrote it. The unpacked case stores a byte,
        // since a 32-bit store would clobber the three flags sharing its slot.
        let flags = [GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V];
        let packed = flags.iter().all(|r| self.regs.contains_key(r))
            && flags.iter().any(|r| self.region_writes.contains(r));
        if packed {
            let locals: Vec<u32> = flags.iter().map(|r| self.regs[r].local).collect();
            for r in flags {
                note_traffic(r);
            }
            self.ctx_base();
            self.f.local_get(locals[0]);
            for (i, &l) in locals.iter().enumerate().skip(1) {
                self.f.local_get(l);
                self.f.i32_const((i * 8) as i32);
                self.f.byte(0x74); // i32.shl
                self.f.byte(0x72); // i32.or
            }
            self.f.store(0x36, 2, ctx::NZCV as u32);
        }
        let pa = [GuestReg::P, GuestReg::A];
        let pa_packed = pa.iter().all(|r| self.regs.contains_key(r))
            && pa.iter().any(|r| self.region_writes.contains(r));
        if pa_packed {
            let lp = self.regs[&GuestReg::P].local;
            let la = self.regs[&GuestReg::A].local;
            note_traffic(GuestReg::P);
            note_traffic(GuestReg::A);
            self.ctx_base();
            self.f.local_get(lp);
            self.f.local_get(la);
            self.f.i32_const(8);
            self.f.byte(0x74); // i32.shl
            self.f.byte(0x72); // i32.or
            self.f.store(0x3b, 1, ctx::PF as u32); // i32.store16
        }
        for (r, local) in all {
            if packed && matches!(r, GuestReg::N | GuestReg::Z | GuestReg::C | GuestReg::V) {
                continue;
            }
            if pa_packed && matches!(r, GuestReg::P | GuestReg::A) {
                continue;
            }
            note_traffic(r);
            self.ctx_base();
            self.f.local_get(local);
            if Self::is_flag(r) {
                self.f.store(0x3a, 0, Self::reg_offset(r)); // i32.store8
            } else {
                match r.ty() {
                    Ty::I32 => self.f.store(0x36, 2, Self::reg_offset(r)),
                    Ty::I64 => self.f.store(0x37, 3, Self::reg_offset(r)),
                    Ty::V128 => self.f.simd_mem(simd::V128_STORE, 4, Self::reg_offset(r)),
                }
            }
        }
    }

    /// One member of a region: its body, then its terminator translated into
    /// region terms.
    fn run_member(&mut self, block: &Block) {
        let n = block.n_values as usize;
        self.value_local = vec![None; n];
        self.value_const = vec![None; n];
        self.value_ctx = vec![false; n];
        self.value_ty = vec![Ty::I64; n];
        let dying = dying_after(block);
        self.prof_entry(block);
        for (i, stmt) in block.stmts.iter().enumerate() {
            self.stmt(stmt, block);
            for &v in dying[i].iter() {
                self.release(Value(v));
            }
        }
        self.region_terminator(block);
    }

    /// Go to `target`: a branch back to the table when it is in the region,
    /// and the ordinary flush-and-tail-call when it is not.
    fn region_goto(&mut self, target: u64, from: u64) {
        let st = self.region.as_ref().expect("not in a region");
        match st.index.get(&target).copied() {
            Some(k) => {
                let (idx_local, depth) = (st.idx_local, st.depth);
                // Every back edge inside the region polls, not only the one to
                // the entry. Changes nothing under the default growth rule (a
                // region holds one loop, through the entry); it is what makes
                // `GEX_REGION_NESTED` safe, since a nested inner loop would
                // otherwise reach no poll and be unpreemptable.
                if target <= from {
                    self.emit_safepoint(target);
                }
                self.f.i32_const(k as i32);
                self.f.local_set(idx_local);
                self.f.br(depth);
            }
            None => {
                self.flush_region_exit();
                if target <= from {
                    self.emit_safepoint(target);
                }
                self.jump_to_constant(target);
            }
        }
    }

    fn region_terminator(&mut self, block: &Block) {
        self.prof_direct_call(block);
        match block.term {
            Term::Goto { target } => self.region_goto(target, block.start),
            Term::Branch {
                cond,
                taken,
                not_taken,
            } => {
                self.push(cond);
                self.f.if_void();
                self.prof_taken(block);
                // One more block is open inside the `if`.
                if let Some(st) = self.region.as_mut() {
                    st.depth += 1;
                }
                self.region_goto(taken, block.start);
                if let Some(st) = self.region.as_mut() {
                    st.depth -= 1;
                }
                self.f.end();
                self.region_goto(not_taken, block.start);
            }
            // `region::grow` admits nothing else.
            _ => unreachable!("region member with a non-branch terminator"),
        }
    }

    fn run(&mut self, block: &Block) {
        let n = block.n_values as usize;
        self.value_local = vec![None; n];
        self.value_const = vec![None; n];
        self.value_ctx = vec![false; n];
        self.value_ty = vec![Ty::I64; n];

        let dying = dying_after(block);

        // A block whose only successor is itself becomes a real wasm loop, so
        // its guest registers stay in locals across the back-edge. Sound only
        // if every register the body reads is loaded *before* the loop, or a
        // second iteration reads a context the first did not write back.
        if self.can_fuse_self_loop(block) {
            let mut reads = live_in_registers(block);
            reads.sort_by_key(|r| Self::reg_offset(*r));
            for r in reads {
                self.reg_local(r);
            }
            let counter = self.alloc(Ty::I32);
            self.f.i32_const(loop_poll());
            self.f.local_set(counter);
            self.reenter_local = Some(counter);
            self.f.loop_void();
            self.loop_depth = Some(0);
        }
        self.prof_entry(block);

        for (i, stmt) in block.stmts.iter().enumerate() {
            self.stmt(stmt, block);
            // Recycle locals whose values are dead after this statement.
            for &v in dying[i].iter() {
                self.release(Value(v));
            }
        }
        self.terminator(block);
        if self.loop_depth.is_some() {
            self.f.end();
            self.loop_depth = None;
            self.reenter_local = None;
        }
    }

    /// Whether this block can be emitted as a self-contained wasm loop.
    ///
    /// The body must not contain anything that invalidates the register cache
    /// -- a host call can rewrite guest state, and after one the cached locals
    /// no longer describe the machine.
    fn can_fuse_self_loop(&self, block: &Block) -> bool {
        let back_edge = match block.term {
            Term::Goto { target } => target == block.start,
            Term::Branch {
                taken, not_taken, ..
            } => taken == block.start || not_taken == block.start,
            _ => false,
        };
        // A kill switch, so the fused and unfused forms can be compared and a
        // suspected miscompilation bisected.
        if std::env::var_os("GEX_NO_LOOP_FUSION").is_some() {
            return false;
        }
        // And a finer one: `GEX_FUSE_SKIP=0xaaa,0xbbb` leaves the named
        // blocks unfused, to bisect a fusion bug down to one loop;
        // `GEX_FUSE_LIST=1` names every loop that is fused.
        if back_edge {
            static SKIP: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
            let skip = SKIP.get_or_init(|| {
                std::env::var("GEX_FUSE_SKIP")
                    .ok()
                    .map(|v| {
                        v.split(',')
                            .filter_map(|x| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok())
                            .collect()
                    })
                    .unwrap_or_default()
            });
            if skip.contains(&block.start) {
                return false;
            }
            if std::env::var_os("GEX_FUSE_LIST").is_some() {
                eprintln!("[fuse] self-loop at {:#x}", block.start);
            }
        }
        back_edge && !block.stmts.iter().any(clobbers_guest_state)
    }

    // -- local management --------------------------------------------------

    fn alloc(&mut self, ty: Ty) -> u32 {
        let pool = match ty {
            Ty::I32 => &mut self.free_i32,
            Ty::I64 => &mut self.free_i64,
            Ty::V128 => &mut self.free_v128,
        };
        if let Some(l) = pool.pop() {
            return l;
        }
        let l = self.f.add_local(val_type(ty));
        self.temps.push(l);
        l
    }

    fn release(&mut self, v: Value) {
        let idx = v.0 as usize;
        if idx >= self.value_local.len() {
            return;
        }
        if let Some(local) = self.value_local[idx].take() {
            match (self.value_fp.get(idx).copied().flatten(), self.value_ty[idx]) {
                (Some(true), _) => self.free_f64.push(local),
                (Some(false), _) => self.free_f32.push(local),
                (None, Ty::I32) => self.free_i32.push(local),
                (None, Ty::I64) => self.free_i64.push(local),
                (None, Ty::V128) => self.free_v128.push(local),
            }
        }
    }

    /// Push a value onto the wasm operand stack.
    fn push(&mut self, v: Value) {
        let idx = v.0 as usize;
        match self.value_local.get(idx).copied().flatten() {
            Some(l) => {
                self.f.local_get(l);
                if let Some(d) = self.value_fp.get(idx).copied().flatten() {
                    self.reinterpret_from_fp(d);
                }
            }
            None => {
                // A value with no local is either a folded constant or a bug
                // in liveness; constants are re-materialised inline, which is
                // cheaper than a local anyway.
                match self.value_const.get(idx).copied().flatten() {
                    Some(c) if self.value_ty[idx] == Ty::I32 => self.f.i32_const(c as i32),
                    Some(c) => self.f.i64_const(c as i64),
                    None => panic!("value {v:?} used after release"),
                }
            }
        }
    }

    // -- guest register cache ---------------------------------------------

    /// The four condition flags share one 32-bit context slot, a byte each.
    ///
    /// So they load and store as bytes, not words -- `reg_offset` gives the
    /// byte, and `flush_inner` writes all four at once when it can.
    fn is_flag(r: GuestReg) -> bool {
        r.is_flag()
    }

    fn reg_offset(r: GuestReg) -> u32 {
        (match r {
            GuestReg::X(n) => ctx::x(n),
            GuestReg::V128(n) => ctx::v(n),
            GuestReg::Sp => ctx::SP,
            GuestReg::Pc => ctx::PC,
            GuestReg::N => ctx::NF,
            GuestReg::Z => ctx::ZF,
            GuestReg::C => ctx::CF,
            GuestReg::V => ctx::VF,
            GuestReg::P => ctx::PF,
            GuestReg::A => ctx::AF,
            GuestReg::D => ctx::DF,
            GuestReg::Tpidr => ctx::TPIDR,
            GuestReg::TpidrRo => ctx::TPIDRRO,
            GuestReg::MonAddr => ctx::MON_ADDR,
            GuestReg::MonValue => ctx::MON_VALUE,
            GuestReg::MonSize => ctx::MON_SIZE,
            GuestReg::Mxcsr => ctx::MXCSR,
            GuestReg::X87Top => ctx::X87_TOP,
            GuestReg::X87Cw => ctx::X87_CW,
            GuestReg::X87Sw => ctx::X87_SW,
            GuestReg::X87Tags => ctx::X87_TAGS,
            // Not a slot: read as the context's own address. Never stored.
            GuestReg::CtxAddr => 0,
        }) as u32
    }

    /// Trap into the host if the guest's stack pointer is not a stack address.
    ///
    /// Emitted before anything else in the body, so the value tested is what
    /// the *previous* block left behind. One bound, not two: above `STACK_TOP`
    /// is the shim's own world where no guest `sp` ever is, and a lower bound
    /// would need to know where this thread's stack is mapped. Reads the
    /// context directly, not through `reg_local`, so it does not make the body
    /// that follows believe `sp` is already in a local.
    fn emit_sp_check(&mut self, pc: u64) {
        self.ctx_base();
        self.f.load(0x29, 3, ctx::SP as u32);
        self.f.i64_const(gex_abi::STACK_TOP as i64);
        self.f.byte(0x5a); // i64.ge_u
        self.f.if_void();
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.i32_const(gex_abi::imports::BRK_BAD_SP);
        self.f.call(self.cx.breakpoint);
        self.f.end();
    }

    /// Unconditional breakpoint at a block's entry: call the host with
    /// `BRK_TRAP_PC`, which dumps the live registers and returns so the block
    /// runs on. The registers are still in the context here (nothing in this
    /// block has cached them yet), so x30 names the caller. See `trap_pc`.
    fn emit_trap_pc(&mut self, pc: u64) {
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.i32_const(gex_abi::imports::BRK_TRAP_PC);
        self.f.call(self.cx.breakpoint);
    }

    /// The `GEX_TRACE_BLOCKS` trace point. Registers are flushed first so the
    /// host sees them; nothing is invalidated, since the host only reads.
    fn emit_trace_point(&mut self, pc: u64) {
        self.flush();
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.i32_const(gex_abi::imports::BRK_TRACE);
        self.f.call(self.cx.breakpoint);
    }

    /// The local holding `r`, loading it from the context on first use.
    fn reg_local(&mut self, r: GuestReg) -> u32 {
        if let Some(c) = self.regs.get(&r) {
            return c.local;
        }
        assert!(self.func_regs.is_none(), "function mode: {r:?} has no local (not in the used set)");
        let ty = r.ty();
        let local = self.f.add_local(val_type(ty));
        let before = self.f.code.len();
        self.f.local_get(self.ctx_local);
        if r == GuestReg::CtxAddr {
            // The context's guest address is the `ctx` parameter itself.
            self.f.byte(code::I64_EXTEND_I32_U);
        } else if Self::is_flag(r) {
            // i32.load8_u: the flag is one byte of the packed slot, and it is
            // 0 or 1, so there is nothing to sign-extend or mask.
            self.f.addr_i32();
            self.f.load(0x2d, 0, Self::reg_offset(r));
        } else {
            self.f.addr_i32();
            match ty {
                Ty::I32 => self.f.load(0x28, 2, Self::reg_offset(r)),
                Ty::I64 => self.f.load(0x29, 3, Self::reg_offset(r)),
                Ty::V128 => self.f.simd_mem(simd::V128_LOAD, 4, Self::reg_offset(r)),
            }
        }
        self.f.local_set(local);
        CTX_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
        note_traffic(r);
        self.regs.insert(
            r,
            Cached {
                local,
                dirty: false,
            },
        );
        local
    }

    /// Mark `r` as holding the value currently on top of the stack.
    fn reg_store_from_stack(&mut self, r: GuestReg) {
        let ty = r.ty();
        let local = match self.regs.get(&r) {
            Some(c) => c.local,
            None => self.f.add_local(val_type(ty)),
        };
        self.f.local_set(local);
        self.regs.insert(r, Cached { local, dirty: true });
    }

    /// Write every dirty cached register back to the context.
    ///
    /// Carrying the hot registers in block *parameters* instead was tried and
    /// reverted: the context never leaves L1, so it held more values live
    /// across every function and cost more in spills than it saved in traffic.
    fn flush(&mut self) {
        self.flush_inner(false);
    }

    /// Flush, but leave the registers marked dirty.
    ///
    /// Used where the flush is emitted inside a conditional: the stores only
    /// happen on some executions, so the emitter must keep believing the
    /// context is stale and write them back again on the paths that leave.
    fn flush_conditional(&mut self) {
        self.flush_inner(true);
    }

    fn flush_inner(&mut self, keep_dirty: bool) {
        let mut dirty: Vec<(GuestReg, u32)> = self
            .regs
            .iter()
            .filter(|(_, c)| c.dirty)
            .map(|(r, c)| (*r, c.local))
            .collect();
        // Deterministic output makes translated modules reproducible, which
        // the content-addressed cache depends on.
        dirty.sort_by_key(|(r, _)| Self::reg_offset(*r));
        let before = self.f.code.len();

        // The four flags in one store, when all four are cached (the usual
        // case, since almost every flag-setting instruction writes all four).
        // Safe because a cached-but-not-dirty flag still holds the current
        // value. A flag with no local falls through to the per-flag byte store.
        let flags = [GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V];
        let all_cached = flags.iter().all(|r| self.regs.contains_key(r));
        let any_dirty = flags
            .iter()
            .any(|r| self.regs.get(r).is_some_and(|c| c.dirty));
        let packed = all_cached && any_dirty;
        if packed {
            let locals: Vec<u32> = flags.iter().map(|r| self.regs[r].local).collect();
            for r in flags {
                note_traffic(r);
            }
            self.ctx_base();
            self.f.local_get(locals[0]);
            for (i, &l) in locals.iter().enumerate().skip(1) {
                self.f.local_get(l);
                self.f.i32_const((i * 8) as i32);
                self.f.byte(0x74); // i32.shl
                self.f.byte(0x72); // i32.or
            }
            self.f.store(0x36, 2, ctx::NZCV as u32);
            if !keep_dirty {
                for r in flags {
                    if let Some(c) = self.regs.get_mut(&r) {
                        c.dirty = false;
                    }
                }
            }
        }
        // x86-64's parity and auxiliary-carry flags, adjacent bytes, as one
        // 16-bit store when both are cached and either is dirty.
        let pa = [GuestReg::P, GuestReg::A];
        let pa_packed = pa.iter().all(|r| self.regs.contains_key(r))
            && pa.iter().any(|r| self.regs.get(r).is_some_and(|c| c.dirty));
        if pa_packed {
            let lp = self.regs[&GuestReg::P].local;
            let la = self.regs[&GuestReg::A].local;
            note_traffic(GuestReg::P);
            note_traffic(GuestReg::A);
            self.ctx_base();
            self.f.local_get(lp);
            self.f.local_get(la);
            self.f.i32_const(8);
            self.f.byte(0x74); // i32.shl
            self.f.byte(0x72); // i32.or
            self.f.store(0x3b, 1, ctx::PF as u32); // i32.store16
            if !keep_dirty {
                for r in pa {
                    if let Some(c) = self.regs.get_mut(&r) {
                        c.dirty = false;
                    }
                }
            }
        }

        for (r, local) in dirty {
            if packed && matches!(r, GuestReg::N | GuestReg::Z | GuestReg::C | GuestReg::V) {
                continue;
            }
            if pa_packed && matches!(r, GuestReg::P | GuestReg::A) {
                continue;
            }
            note_traffic(r);
            self.ctx_base();
            self.f.local_get(local);
            if Self::is_flag(r) {
                self.f.store(0x3a, 0, Self::reg_offset(r)); // i32.store8
            } else {
                match r.ty() {
                    Ty::I32 => self.f.store(0x36, 2, Self::reg_offset(r)),
                    Ty::I64 => self.f.store(0x37, 3, Self::reg_offset(r)),
                    Ty::V128 => self.f.simd_mem(simd::V128_STORE, 4, Self::reg_offset(r)),
                }
            }
            if !keep_dirty {
                if let Some(c) = self.regs.get_mut(&r) {
                    c.dirty = false;
                }
            }
        }
        CTX_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    /// Forget every cached register, forcing later reads to reload. Used
    /// after a host call that can rewrite guest state.
    fn invalidate(&mut self) {
        if let Some(used) = self.func_regs.take() {
            // Function mode: the locals are the function's, so reload them.
            self.reload(&used);
            self.func_regs = Some(used);
            return;
        }
        self.regs.clear();
    }

    /// Record the guest PC in the context, so a host call or dispatch miss
    /// sees a precise architectural state.
    fn store_pc(&mut self, pc: u64) {
        self.ctx_base();
        self.f.i64_const(pc as i64);
        self.f.store(0x37, 3, ctx::PC as u32);
    }

    /// Write the system-call immediate for the kernel to read at the call
    /// that follows: the number of a system call whose ISA encodes it in the
    /// instruction itself (see [`gex_ir::Term::Syscall`]). A register-numbered
    /// call stores nothing (0 leaves the kernel reading the number register).
    fn store_syscall_imm(&mut self, imm: u32) {
        if imm != 0 {
            self.ctx_base();
            self.f.i64_const(imm as i64);
            self.f.store(0x37, 3, ctx::SYSCALL_IMM as u32);
        }
    }

    // -- addressing --------------------------------------------------------

    /// Emit the 32-bit linear-memory address for `addr + offset`, returning
    /// the static offset the access should carry.
    ///
    /// Guest VA equals linear-memory offset, so this is a
    /// truncation plus one guard: an effective address that does not fit in
    /// 32 bits is steered to an address inside the unmapped top page, which
    /// makes the access trap and become SIGSEGV rather than silently aliasing
    /// low memory.
    fn emit_address(&mut self, addr: Value, offset: i64) -> u32 {
        self.emit_address_in(addr, offset, true)
    }

    /// [`Self::emit_address`] for an operand with no memarg (a bulk
    /// operation's): the address itself, the static offset always 0.
    fn emit_address_flat(&mut self, addr: Value) {
        let off = self.emit_address_in(addr, 0, false);
        debug_assert_eq!(off, 0);
    }

    /// Whether `v` is the context address plus offsets (`value_ctx`).
    fn is_ctx(&self, v: Value) -> bool {
        self.value_ctx.get(v.0 as usize).copied().unwrap_or(false)
    }

    /// `biased`: the caller adds the returned static offset in a memarg, so
    /// the null guard may ride on it (`gex_abi::NULL_GUARD`): the address
    /// less the guard, the guard added back in the memarg, and an address
    /// below the guard wraps past 2^32 and traps (or, where tier-1 guards
    /// the bounds anyway, the bounds check's range starts at the guard).
    /// Otherwise there is no null guard.
    fn emit_address_in(&mut self, addr: Value, offset: i64, biased: bool) -> u32 {
        // Not for the thread's own context, which is below the guard, nor a
        // bulk operand: a zero-length copy from a null pointer touches
        // nothing, and the engine traps any bulk address past the end.
        let on = null_guard_on();
        let null_guard = on && biased && !self.is_ctx(addr);
        // A memory64 guest memory takes the guest address
        // as it stands: past the end of the memory, the engine's own check
        // traps, which is the SIGSEGV the guard below exists to produce. The
        // static offset follows the same rule as the guarded path.
        if gex_abi::mem64() {
            if let Some(c) = self.value_const.get(addr.0 as usize).copied().flatten() {
                let ea = c.wrapping_add(offset as u64);
                // Below the null guard: an address past every memory's end.
                let ea = if null_guard && ea < gex_abi::NULL_GUARD { u64::MAX - 15 } else { ea };
                self.f.i64_const(ea as i64);
                return 0;
            }
            self.push(addr);
            let max_static: i64 = if gex_abi::X86 { 0xffff } else if on { u32::MAX as i64 - gex_abi::NULL_GUARD as i64 } else { u32::MAX as i64 };
            if null_guard {
                // memory64's effective address does not wrap: (a - G) + G
                // past 2^64 is out of bounds. One add, the offset's if any.
                let g = gex_abi::NULL_GUARD as i64;
                if (0..=max_static).contains(&offset) {
                    self.f.i64_const(-g);
                    self.f.byte(0x7c); // i64.add
                    return (offset + g) as u32;
                }
                self.f.i64_const(offset.wrapping_sub(g));
                self.f.byte(0x7c); // i64.add
                return g as u32;
            }
            if biased && (0..=max_static).contains(&offset) {
                return offset as u32;
            }
            if offset != 0 {
                self.f.i64_const(offset);
                self.f.byte(0x7c); // i64.add
            }
            return 0;
        }
        // Constant addresses are extremely common (literal pools, absolute
        // references) and need no guard at all.
        if let Some(c) = self.value_const.get(addr.0 as usize).copied().flatten() {
            let mut ea = c.wrapping_add(offset as u64);
            // 32-bit Wine: Windows' KUSER_SHARED_DATA page is 32-bit Wine's (`gex_abi::WINE32_KUSER`).
            if let Some(f) = gex_abi::fold_kuser(ea) {
                gex_ir::note_kuser_fold();
                ea = f;
            }
            // Below the null guard: the poisoned address, as past the end
            // (32-bit Wine with folding on: the named fault, as for a pointer).
            if null_guard && ea < gex_abi::NULL_GUARD {
                if gex_abi::kuser_fold() {
                    self.f.i64_const(ea as i64);
                    self.guard_to_i32();
                    return 0;
                }
                self.f.i32_const(POISON_ADDR as i32);
                return 0;
            }
            if ea < guard_limit() {
                self.f.i32_const(ea as i32);
                return 0;
            }
            // Past the guard with folding on: the named fault, as for a
            // pointer (`guard_to_i32`).
            if gex_abi::kuser_fold() {
                self.f.i64_const(ea as i64);
                self.guard_to_i32();
                return 0;
            }
            self.f.i32_const(POISON_ADDR as i32);
            return 0;
        }
        // An address zero-extended from 32 bits is below 2^32: past the
        // memory, it traps by itself, as the guard would make it -- when
        // nothing but guest memory lies below the guard limit. The static
        // offset is no different from the guarded case.
        if self.value_u32.get(addr.0 as usize).copied().unwrap_or(false) && guard_limit() >= GUEST_LIMIT && !Self::no_guard() {
            let max_static: i64 = if gex_abi::X86 { 0xffff } else if on { u32::MAX as i64 - gex_abi::NULL_GUARD as i64 } else { u32::MAX as i64 };
            if (0..=max_static).contains(&offset) {
                self.push(addr);
                if !null_guard {
                    self.f.byte(code::I32_WRAP_I64);
                    return offset as u32;
                }
                if !self.value_chase.get(addr.0 as usize).copied().unwrap_or(false) {
                    // A computed address (an index, a sum): the guard rides
                    // on the memarg, (a - G) mod 2^32 plus G in the static
                    // offset past 2^32 for an `a` below G. One add, off any
                    // load's latency, where a branch costs throughput.
                    self.f.byte(code::I32_WRAP_I64);
                    self.f.i32_const(-(gex_abi::NULL_GUARD as i32));
                    self.f.byte(0x6a); // i32.add
                    return offset as u32 + gex_abi::NULL_GUARD as u32;
                }
                // The null guard as a branch beside the access, not on the
                // address: an `a` below G takes an access at POISON_ADDR,
                // which traps as the guard's would. Kept off the address's
                // dependency chain, where a bias (one more add before every
                // load) slowed pointer chasing by a fifth.
                let t = self.scratch_i64;
                self.f.local_tee(t);
                self.f.byte(code::I32_WRAP_I64);
                self.f.local_get(t);
                self.f.i64_const(gex_abi::NULL_GUARD as i64);
                self.f.byte(code::I64_LT_U);
                self.f.if_void();
                self.f.i32_const(POISON_ADDR as i32);
                self.f.load(0x2d, 0, 0); // i32.load8_u
                self.f.byte(0x1a); // drop
                self.f.end();
                return offset as u32;
            }
        }
        // Caching a repeated base's guard in a local was tried and bought
        // nothing: the guard is pure arithmetic on an unchanging value, which
        // V8's own CSE already collapses, so the emitted bytes it removes cost
        // nothing while the local it adds is real.
        // A static offset rides on the guarded base, so it must stay small:
        // the guard only promises the base is below `GUEST_LIMIT`, and the
        // engine's guard region past the memory is what turns base+offset
        // into a trap rather than a wild read. 64 KiB is inside every
        // engine's guard region; x86-64's displacements can reach 2 GiB and
        // go through the add below instead. A64's immediate offsets are
        // bounded by the encoding, so it keeps the full u32 range.
        let max_static: i64 = if gex_abi::X86 { 0xffff } else { u32::MAX as i64 };
        let static_off = if biased && (0..=max_static).contains(&offset) {
            self.push(addr);
            offset as u32
        } else {
            self.push(addr);
            if offset != 0 {
                self.f.i64_const(offset);
                self.f.byte(0x7c); // i64.add
            }
            0
        };
        if null_guard {
            self.guard_to_i32();
        } else {
            self.guard_to_i32_above(0);
        }
        static_off
    }

    /// [`Self::addr_ty`] as a GIR type, for [`Self::alloc`].
    fn addr_ir_ty() -> Ty {
        if gex_abi::mem64() {
            Ty::I64
        } else {
            Ty::I32
        }
    }

    /// The type of a local holding a memory address: the memory's index type.
    fn addr_ty() -> ValType {
        if gex_abi::mem64() {
            ValType::I64
        } else {
            ValType::I32
        }
    }

    /// The i64 length on the stack as a bulk memory op's length: the
    /// memory's index type.
    fn len_to_index(&mut self) {
        if !gex_abi::mem64() {
            self.f.byte(code::I32_WRAP_I64);
        }
    }

    /// Push the context as a memory address: the `ctx` parameter (an i32 in
    /// every ABI), widened for a memory64 guest memory. Pass
    /// `ctx_local` itself to calls.
    fn ctx_base(&mut self) {
        self.f.local_get(self.ctx_local);
        self.f.addr_i32();
    }

    /// Whether to emit the bounds guard at all.
    ///
    /// `GEX_UNSAFE_NO_GUARD=1` drops it, which is **not shippable**: a guest
    /// address above `GUEST_LIMIT` then aliases low memory silently instead of
    /// taking SIGSEGV. It exists only to price the guard.
    fn no_guard() -> bool {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GEX_UNSAFE_NO_GUARD").is_some())
    }

    /// Consume an `i64` effective address and leave a bounds-safe `i32`; for
    /// a memory64 guest memory, leave the `i64` (the engine checks it).
    ///
    /// An address below the null guard (`gex_abi::NULL_GUARD`) is steered
    /// the same way: one subtraction turns the bound into a range.
    fn guard_to_i32(&mut self) {
        self.guard_to_i32_above(if null_guard_on() { gex_abi::NULL_GUARD } else { 0 });
    }

    /// [`Self::guard_to_i32`] with the range's floor given: 0 for an access
    /// the null guard leaves out (the thread's context).
    fn guard_to_i32_above(&mut self, floor: u64) {
        if gex_abi::mem64() {
            return;
        }
        if Self::no_guard() {
            self.f.byte(code::I32_WRAP_I64);
            return;
        }
        let before = self.f.code.len();
        self.guard_to_i32_inner(floor);
        GUARD_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    /// `t - floor <u limit - floor`, with `t` (the i64 address) in the
    /// scratch local and the comparison's operands pushed.
    fn guard_range(&mut self, floor: u64) {
        let t = self.scratch_i64;
        self.f.local_get(t);
        if floor != 0 {
            self.f.i64_const(floor as i64);
            self.f.byte(0x7d); // i64.sub
        }
        self.f.i64_const(guard_limit().saturating_sub(floor) as i64);
    }

    fn guard_to_i32_inner(&mut self, floor: u64) {
        let t = self.scratch_i64;
        // 32-bit Wine with KUSER_SHARED_DATA folded: an address at or above the old
        // page is a pointer the translator could not fold. A branch to the
        // host's named fault (`BRK_KUSER_UNFOLDED`: an access violation at
        // that address), where the select would steer to a bare trap. The
        // context holds what the block last wrote back, at its start's pc, as
        // natively for an out-of-bounds access.
        if gex_abi::kuser_fold() {
            if floor == 0 {
                self.f.local_tee(t);
                self.f.i64_const(guard_limit() as i64);
            } else {
                self.f.local_set(t);
                self.guard_range(floor);
            }
            self.f.byte(code::I64_GE_U);
            if self.func_regs.is_some() {
                // The function JIT's bodies have results and their own exits:
                // the fault stays in place.
                self.f.if_void();
                self.kuser_fault_call();
                self.f.end();
            } else {
                // One fault per body (`close_kuser_fault`), `t` still the
                // address there.
                self.f.br_if(self.f.open());
                self.kuser_fault = true;
            }
            self.f.local_get(t);
            self.f.byte(code::I32_WRAP_I64);
            return;
        }
        self.f.local_set(t);
        self.f.local_get(t);
        self.f.byte(code::I32_WRAP_I64);
        self.f.i32_const(POISON_ADDR as i32);
        self.guard_range(floor);
        self.f.byte(code::I64_LT_U);
        self.f.select();
    }

    /// `BRK_KUSER_UNFOLDED` for the address in the scratch local, at the
    /// block's start (the host writes that pc back); it does not return.
    /// In function mode with resumable faults, the registers written back
    /// first, at the faulting instruction's pc, and
    /// `BRK_KUSER_UNFOLDED_RESUMABLE`: the host delivers the access
    /// violation in place and returns, and the function leaves for the
    /// context's PC.
    fn kuser_fault_call(&mut self) {
        let resumable = self.func_regs.is_some() && self.resumable_pc != 0;
        let pc = if resumable { self.stmt_pc.unwrap_or(self.block_start) } else { self.block_start };
        if resumable {
            self.flush_conditional();
        }
        let t = self.scratch_i64;
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.local_get(t);
        self.f.i64_const(32);
        self.f.byte(code::I64_SHL);
        self.f.byte(code::I64_OR);
        self.f.i32_const(if resumable { gex_abi::imports::BRK_KUSER_UNFOLDED_RESUMABLE } else { gex_abi::imports::BRK_KUSER_UNFOLDED });
        self.f.call(self.cx.breakpoint);
        if resumable {
            self.f.i32_const(0);
            self.f.ret();
        } else {
            self.f.unreachable();
        }
    }

    /// Function mode: `int3`'s or `ud2`'s host call as a fault the host may
    /// deliver in place (`resumable_pc` on its pc), as gex-cgen's units make
    /// it. The registers and PC (the next instruction, `next`) are written
    /// back first, so a host that returns without delivering anything (a
    /// trace point) carries on; any other PC is a handler's target, which the
    /// function leaves for (returning 0). Delivered by trapping out instead,
    /// such a fault unwinds every wasm frame down to the nearest run loop:
    /// those of the native code that called the function too, a native
    /// `__TRY` whose handler then catches the exception among them.
    fn resumable_fault(&mut self, func: HostFunc, args: &[Value], next: u64) {
        self.flush();
        self.store_pc(next);
        self.f.local_get(self.ctx_local);
        self.push(args[0]);
        self.f.i64_const(self.resumable_pc as i64);
        self.f.byte(code::I64_OR);
        self.push(args[1]);
        self.f.call(self.import_index(func));
        self.invalidate();
        self.ctx_base();
        self.f.load(0x29, 3, ctx::PC as u32);
        self.f.i64_const(next as i64);
        self.f.byte(code::I64_NE);
        self.f.if_void();
        self.f.i32_const(0);
        self.f.ret();
        self.f.end();
    }

    /// The fault the body's KUSER guards branch to: the body becomes
    /// `block block <body> br 1 end <fault> end`, so it runs as before and
    /// a guard at any depth reaches the fault with one `br_if`. The body is
    /// a block function's (no results), complete.
    fn close_kuser_fault(&mut self) {
        if !std::mem::take(&mut self.kuser_fault) {
            return;
        }
        let before = self.f.code.len();
        debug_assert_eq!(self.f.open(), 0);
        self.f.prepend_block();
        self.f.prepend_block();
        self.f.br(1);
        self.f.end();
        self.kuser_fault_call();
        self.f.end();
        GUARD_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    // -- statements --------------------------------------------------------

    fn stmt(&mut self, s: &Stmt, block: &Block) {
        match s {
            Stmt::Def { dst, expr } => self.def(*dst, expr),
            Stmt::WriteReg { reg, val } => {
                self.push(*val);
                self.reg_store_from_stack(*reg);
            }
            Stmt::Store {
                addr,
                offset,
                val,
                sz,
                order,
            } => {
                let static_off = self.emit_address(*addr, *offset);
                self.push(*val);
                let ty = self.type_of(*val);
                if *order == Order::SeqCst {
                    let sub = atomic_store_op(ty, *sz);
                    self.f.atomic(sub, natural_align(*sz), static_off);
                } else if coherent_loads() {
                    self.emit_coherent_store(ty, *sz, static_off);
                } else {
                    let opcode = store_op(ty, *sz);
                    self.f.store(opcode, natural_align(*sz), static_off);
                }
            }
            Stmt::HostCall { func: func @ (HostFunc::Breakpoint | HostFunc::Illegal), args }
                if self.resumable_pc != 0 && self.fault_next.is_some() && args.len() == 2 =>
            {
                self.resumable_fault(*func, args, self.fault_next.unwrap());
            }
            Stmt::HostCall { func, args } => {
                self.host_call(*func, args, None, block);
            }
            Stmt::X87 { op, args, outs } => self.emit_x87(*op, args, outs),
            Stmt::Trap { cond, pc, code } => {
                // Only the faulting path writes the registers back, so they
                // stay dirty for the path that carries on.
                self.push(*cond);
                self.f.if_void();
                self.flush_conditional();
                self.store_pc(*pc);
                self.f.local_get(self.ctx_local);
                self.f.i64_const((*pc | self.resumable_pc) as i64);
                self.f.i32_const(*code);
                self.f.call(self.cx.breakpoint);
                if self.resumable_pc != 0 {
                    // The host raised the fault and returned with the
                    // handlers' state in the context: leave for it.
                    self.f.i32_const(0);
                    self.f.ret();
                } else {
                    // The host raises the fault and never returns.
                    self.f.unreachable();
                }
                self.f.end();
            }
            Stmt::VStore {
                addr,
                offset,
                val,
                bytes,
            } => {
                let static_off = self.emit_address(*addr, *offset);
                self.push(*val);
                let align = bytes.trailing_zeros();
                if *bytes == 16 {
                    self.f.simd_mem(simd::V128_STORE, align, static_off);
                } else {
                    let sub = match bytes {
                        1 => simd::V128_STORE8_LANE,
                        2 => simd::V128_STORE16_LANE,
                        4 => simd::V128_STORE32_LANE,
                        _ => simd::V128_STORE64_LANE,
                    };
                    self.f.simd_mem_lane(sub, align, static_off, 0);
                }
            }
            Stmt::MemCopy { dst, src, len } if copy_loop() => {
                self.emit_memcopy_loop(*dst, *src, *len);
            }
            Stmt::MemCopy { dst, src, len } => {
                // Both addresses first: `guard_to_i32` borrows the i64
                // scratch, and the length wants it afterwards.
                let (d, s) = if gex_abi::mem64() { self.scratch_addr } else { (self.scratch_i32, self.scratch_entry) };
                self.emit_address_flat(*dst);
                self.f.local_set(d);
                self.emit_address_flat(*src);
                self.f.local_set(s);
                self.clamp_len(*len, &[d, s]);
                self.f.local_get(d);
                self.f.local_get(s);
                self.f.local_get(self.scratch_i64);
                self.len_to_index();
                self.f.byte(crate::wasm::op::MISC_PREFIX);
                self.f.u32(code::MEMORY_COPY);
                self.f.byte(0); // destination memory
                self.f.byte(0); // source memory
            }
            Stmt::MemFill { dst, val, len } => {
                let d = if gex_abi::mem64() { self.scratch_addr.0 } else { self.scratch_i32 };
                self.emit_address_flat(*dst);
                self.f.local_set(d);
                self.clamp_len(*len, &[d]);
                self.f.local_get(d);
                self.push(*val);
                self.f.byte(code::I32_WRAP_I64);
                self.f.local_get(self.scratch_i64);
                self.len_to_index();
                self.f.byte(crate::wasm::op::MISC_PREFIX);
                self.f.u32(code::MEMORY_FILL);
                self.f.byte(0); // memory
            }
            Stmt::MemCmp { dst, a, b, len } => self.emit_memcmp(*dst, *a, *b, *len),
            Stmt::Fence => self.f.atomic_fence(),
            Stmt::StrOp { .. } => self.emit_strop(s),
        }
    }

    /// Clamp the length in the i64 scratch so a bulk op from any of `bases`
    /// cannot run off the end of guest memory. See [`Self::clamp_len`].
    fn clamp_scratch_len(&mut self, bases: &[u32]) {
        // A memory64 guest memory needs no clamp: an op running past its
        // end traps, which the host reports as the guest's SIGSEGV.
        if gex_abi::mem64() {
            return;
        }
        let n = self.scratch_i64;
        let room = self.scratch_page;
        for &b in bases {
            self.f.i64_const(GUEST_LIMIT as i64);
            self.f.local_get(b);
            self.f.byte(code::I64_EXTEND_I32_U);
            self.f.byte(code::I64_SUB);
            self.f.local_set(room);
            self.f.local_get(n);
            self.f.local_get(room);
            self.f.local_get(n);
            self.f.local_get(room);
            self.f.byte(code::I64_LT_U);
            self.f.select();
            self.f.local_set(n);
        }
    }

    /// An x86 `rep` string instruction: see [`Stmt::StrOp`].
    ///
    /// `rep movs` with the direction flag clear and a forward-safe overlap
    /// becomes one `memory.copy`; `rep stos` with a repeating byte pattern
    /// becomes one `memory.fill`. Everything else is an element loop whose
    /// accesses are guarded like any other guest access.
    fn emit_strop(&mut self, s: &Stmt) {
        let Stmt::StrOp {
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
        } = s
        else {
            unreachable!()
        };
        use gex_ir::{StrKind, StrUntil};
        let (kind, sz, until) = (*kind, *sz, *until);
        let bytes = sz.bytes() as i64;

        let lc = self.alloc(Ty::I64);
        let ld = self.alloc(Ty::I64);
        let ls = self.alloc(Ty::I64);
        let lv = self.alloc(Ty::I64);
        let lstep = self.alloc(Ty::I64);
        let la = self.alloc(Ty::I64);
        let lb = self.alloc(Ty::I64);

        self.push(*count);
        self.f.local_set(lc);
        self.push(*dst);
        self.f.local_set(ld);
        self.push(*src);
        self.f.local_set(ls);
        self.push(*val);
        self.f.local_set(lv);
        // step = df ? -bytes : +bytes  (`select` yields its first operand
        // when the condition is non-zero)
        self.f.i64_const(bytes);
        self.f.i64_const(-bytes);
        self.push(*df);
        self.f.byte(0x45); // i32.eqz
        self.f.select();
        self.f.local_set(lstep);
        self.f.i64_const(0);
        self.f.local_set(la);
        self.f.i64_const(0);
        self.f.local_set(lb);

        let (d32, s32) = if gex_abi::mem64() { self.scratch_addr } else { (self.scratch_i32, self.scratch_entry) };

        self.f.block_void(); // $end
        if matches!(kind, StrKind::Movs | StrKind::Stos) {
            self.f.block_void(); // $slow
            // df set: slow path.
            self.push(*df);
            self.f.br_if(0);
            // count == 0: nothing to do, but the loop handles it too.
            self.f.local_get(lc);
            self.f.byte(0x50); // i64.eqz
            self.f.br_if(0);
            match kind {
                StrKind::Movs => {
                    // Forward-safe unless dst is inside (src, src+len).
                    // rep movs copies element by element ascending, which
                    // equals memmove exactly when dst <= src or dst >= src+len.
                    self.f.local_get(ld);
                    self.f.local_get(ls);
                    self.f.byte(0x58); // i64.le_u
                    self.f.local_get(ld);
                    self.f.local_get(ls);
                    self.f.local_get(lc);
                    self.f.i64_const(bytes);
                    self.f.byte(0x7e); // i64.mul
                    self.f.byte(0x7c); // i64.add
                    self.f.byte(0x5a); // i64.ge_u
                    self.f.byte(0x72); // i32.or
                    self.f.byte(0x45); // i32.eqz
                    self.f.br_if(0);
                    // memory.copy(dst, src, count*bytes)
                    self.f.local_get(ld);
                    self.guard_to_i32_above(0);
                    self.f.local_set(d32);
                    self.f.local_get(ls);
                    self.guard_to_i32_above(0);
                    self.f.local_set(s32);
                    self.f.local_get(lc);
                    self.f.i64_const(bytes);
                    self.f.byte(0x7e);
                    self.f.local_set(self.scratch_i64);
                    self.clamp_scratch_len(&[d32, s32]);
                    self.f.local_get(d32);
                    self.f.local_get(s32);
                    self.f.local_get(self.scratch_i64);
                    self.len_to_index();
                    self.f.byte(crate::wasm::op::MISC_PREFIX);
                    self.f.u32(code::MEMORY_COPY);
                    self.f.byte(0);
                    self.f.byte(0);
                }
                _ => {
                    // Every byte of the element equal, or a single byte.
                    if bytes > 1 {
                        self.f.local_get(lv);
                        self.f.local_get(lv);
                        self.f.i64_const(0xff);
                        self.f.byte(0x83); // i64.and
                        let rep: i64 = match bytes {
                            2 => 0x0101,
                            4 => 0x0101_0101,
                            _ => 0x0101_0101_0101_0101,
                        };
                        self.f.i64_const(rep);
                        self.f.byte(0x7e); // i64.mul
                        if bytes < 8 {
                            // Compare only the element's own bytes.
                            self.f.local_set(self.scratch_i64);
                            self.f.i64_const((1i64 << (bytes * 8)) - 1);
                            self.f.byte(0x83);
                            self.f.local_get(self.scratch_i64);
                            self.f.byte(0x52); // i64.ne
                        } else {
                            self.f.byte(0x52); // i64.ne
                        }
                        self.f.br_if(0);
                    }
                    self.f.local_get(ld);
                    self.guard_to_i32_above(0);
                    self.f.local_set(d32);
                    self.f.local_get(lc);
                    self.f.i64_const(bytes);
                    self.f.byte(0x7e);
                    self.f.local_set(self.scratch_i64);
                    self.clamp_scratch_len(&[d32]);
                    self.f.local_get(d32);
                    self.f.local_get(lv);
                    self.f.byte(code::I32_WRAP_I64);
                    self.f.local_get(self.scratch_i64);
                    self.len_to_index();
                    self.f.byte(crate::wasm::op::MISC_PREFIX);
                    self.f.u32(code::MEMORY_FILL);
                    self.f.byte(0);
                }
            }
            // dst += count*bytes; src likewise; count = 0.
            for l in [ld, ls] {
                self.f.local_get(l);
                self.f.local_get(lc);
                self.f.i64_const(bytes);
                self.f.byte(0x7e);
                self.f.byte(0x7c);
                self.f.local_set(l);
            }
            self.f.i64_const(0);
            self.f.local_set(lc);
            self.f.br(1); // $end
            self.f.end(); // $slow
        }

        // The element loop.
        let (load_op, store_op) = match bytes {
            1 => (0x31u8, 0x3cu8), // i64.load8_u / i64.store8
            2 => (0x33, 0x3d),
            4 => (0x35, 0x3e),
            _ => (0x29, 0x37),
        };
        self.f.block_void();
        self.f.loop_void();
        self.f.local_get(lc);
        self.f.byte(0x50); // i64.eqz
        self.f.br_if(1);
        match kind {
            StrKind::Movs => {
                self.f.local_get(ld);
                self.guard_to_i32_above(0);
                self.f.local_set(d32);
                self.f.local_get(ls);
                self.guard_to_i32_above(0);
                self.f.load(load_op, 0, 0);
                self.f.local_set(la);
                self.f.local_get(d32);
                self.f.local_get(la);
                self.f.store(store_op, 0, 0);
            }
            StrKind::Stos => {
                self.f.local_get(ld);
                self.guard_to_i32_above(0);
                self.f.local_get(lv);
                self.f.store(store_op, 0, 0);
            }
            StrKind::Scas => {
                self.f.local_get(lv);
                if bytes < 8 {
                    self.f.i64_const((1i64 << (bytes * 8)) - 1);
                    self.f.byte(0x83);
                }
                self.f.local_set(la);
                self.f.local_get(ld);
                self.guard_to_i32_above(0);
                self.f.load(load_op, 0, 0);
                self.f.local_set(lb);
            }
            StrKind::Cmps => {
                self.f.local_get(ls);
                self.guard_to_i32_above(0);
                self.f.load(load_op, 0, 0);
                self.f.local_set(la);
                self.f.local_get(ld);
                self.guard_to_i32_above(0);
                self.f.load(load_op, 0, 0);
                self.f.local_set(lb);
            }
        }
        for l in [ld, ls] {
            self.f.local_get(l);
            self.f.local_get(lstep);
            self.f.byte(0x7c);
            self.f.local_set(l);
        }
        self.f.local_get(lc);
        self.f.i64_const(1);
        self.f.byte(0x7d); // i64.sub
        self.f.local_set(lc);
        match until {
            StrUntil::Count => {}
            StrUntil::NotEqual => {
                self.f.local_get(la);
                self.f.local_get(lb);
                self.f.byte(0x52); // i64.ne
                self.f.br_if(1);
            }
            StrUntil::Equal => {
                self.f.local_get(la);
                self.f.local_get(lb);
                self.f.byte(0x51); // i64.eq
                self.f.br_if(1);
            }
        }
        self.f.br(0);
        self.f.end(); // loop
        self.f.end(); // block
        self.f.end(); // $end

        for (v, l) in [(out_count, lc), (out_dst, ld), (out_src, ls), (out_a, la), (out_b, lb)] {
            let i = v.0 as usize;
            self.value_ty[i] = Ty::I64;
            self.value_local[i] = Some(l);
        }
        self.free_i64.push(lv);
        self.free_i64.push(lstep);
    }

    /// Leave in the i64 scratch a length that cannot run off the end of guest
    /// memory from any of `bases`.
    ///
    /// `memory.copy`/`memory.fill` *trap* past the end of the wasm memory,
    /// which would take the whole process rather than fault the guest. Each
    /// base is already below `GUEST_LIMIT` (address-guarded), so clamping the
    /// length to the smallest room above them makes the bulk op unable to trap.
    fn clamp_len(&mut self, len: Value, bases: &[u32]) {
        let n = self.scratch_i64;
        let room = self.scratch_page;
        self.push(len);
        self.f.local_set(n);
        // See `clamp_scratch_len`.
        if gex_abi::mem64() {
            return;
        }
        for &b in bases {
            self.f.i64_const(GUEST_LIMIT as i64);
            self.f.local_get(b);
            self.f.byte(code::I64_EXTEND_I32_U);
            self.f.byte(code::I64_SUB);
            self.f.local_set(room);
            self.f.local_get(n);
            self.f.local_get(room);
            self.f.local_get(n);
            self.f.local_get(room);
            self.f.byte(code::I64_LT_U);
            self.f.select();
            self.f.local_set(n);
        }
    }

    /// `memcpy(dst, src, len)` as an explicit forward byte loop (diagnostic;
    /// see `copy_loop`). Each access is guarded like any other, so a wild
    /// pointer faults where the guest's would rather than trapping the process.
    fn emit_memcopy_loop(&mut self, dst: Value, src: Value, len: Value) {
        let ld = self.alloc(Ty::I64);
        let ls = self.alloc(Ty::I64);
        let llen = self.alloc(Ty::I64);
        let idx = self.alloc(Ty::I64);

        self.push(dst);
        self.f.local_set(ld);
        self.push(src);
        self.f.local_set(ls);
        self.push(len);
        self.f.local_set(llen);
        self.f.i64_const(0);
        self.f.local_set(idx);

        self.f.block_void();
        self.f.loop_void();

        // Done once every byte has been copied.
        self.f.local_get(idx);
        self.f.local_get(llen);
        self.f.byte(0x5a); // i64.ge_u
        self.f.br_if(1);

        // store8(dst + idx, load8(src + idx)). The destination address is
        // guarded and left on the stack, then the source byte on top of it.
        self.f.local_get(ld);
        self.f.local_get(idx);
        self.f.byte(0x7c); // i64.add
        self.guard_to_i32_above(0);
        self.f.local_get(ls);
        self.f.local_get(idx);
        self.f.byte(0x7c); // i64.add
        self.guard_to_i32_above(0);
        self.f.load(0x2d, 0, 0); // i32.load8_u
        self.f.byte(0x3a); // i32.store8
        self.f.byte(0); // align
        self.f.byte(0); // offset

        self.f.local_get(idx);
        self.f.i64_const(1);
        self.f.byte(0x7c);
        self.f.local_set(idx);
        self.f.br(0);

        self.f.end(); // loop
        self.f.end(); // block
    }

    /// Sign-extend the value an atomic (unsigned) load left on the stack from a
    /// sub-width access up to its register type. A no-op for a full-width load.
    /// Atomic loads have no signed form, so a signed guest load becomes an
    /// unsigned atomic load followed by this.
    fn sign_extend_after_atomic(&mut self, ty: Ty, sz: MemSz) {
        let op = match (ty, sz) {
            (Ty::I32, MemSz::B1) => Some(0xc0u8), // i32.extend8_s
            (Ty::I32, MemSz::B2) => Some(0xc1),   // i32.extend16_s
            (Ty::I64, MemSz::B1) => Some(0xc2),   // i64.extend8_s
            (Ty::I64, MemSz::B2) => Some(0xc3),   // i64.extend16_s
            (Ty::I64, MemSz::B4) => Some(0xc4),   // i64.extend32_s
            _ => None,                            // full width: already correct
        };
        if let Some(op) = op {
            self.f.byte(op);
        }
    }

    /// An x86 `lock cmpxchg` (or `cmpxchg8b`) at any alignment: x86 makes a
    /// misaligned locked access atomic (a split lock), wasm's atomics trap on
    /// one, and ConcRT (msvcr120) has 64-bit interlocked fields that are only
    /// 4-aligned. Aligned, the atomic instruction; misaligned, see
    /// [`Self::emit_misaligned_atomic`]. `addr` (i32) is on the stack; leaves
    /// the old value.
    fn emit_cmpxchg_any_alignment(&mut self, expected: Value, replacement: Value, ty: Ty, sz: MemSz, static_off: u32) {
        let a = self.alloc(Self::addr_ir_ty());
        self.f.local_set(a);
        let e = self.alloc(ty);
        let r = self.alloc(ty);
        self.push(expected);
        self.f.local_set(e);
        self.push(replacement);
        self.f.local_set(r);
        self.emit_aligned_or_not(a, ty, sz, static_off, |me| {
            me.f.local_get(a);
            me.f.local_get(e);
            me.f.local_get(r);
            me.f.atomic(atomic_cmpxchg_op(ty, sz), natural_align(sz), static_off);
        }, Misaligned::Cas { expected: e, replacement: r });
    }

    /// An x86 locked read-modify-write (`xadd`, `xchg`, `lock add`/`or`/...)
    /// at any alignment, as [`Self::emit_cmpxchg_any_alignment`]. `addr`
    /// (i32) is on the stack; leaves the old value.
    fn emit_rmw_any_alignment(&mut self, op: RmwOp, val: Value, ty: Ty, sz: MemSz, static_off: u32) {
        let a = self.alloc(Self::addr_ir_ty());
        self.f.local_set(a);
        let v = self.alloc(ty);
        self.push(val);
        self.f.local_set(v);
        self.emit_aligned_or_not(a, ty, sz, static_off, |me| {
            me.f.local_get(a);
            me.f.local_get(v);
            me.f.atomic(atomic_rmw_op(op, ty, sz), natural_align(sz), static_off);
        }, Misaligned::Rmw { op, val: v });
    }

    /// `if ((a + static_off) & (size - 1)) { misaligned } else { aligned }`,
    /// leaving the old value either way.
    ///
    /// First, an address below the guest's floor: the runtime's own memory
    /// (thread contexts, scratch words, the dispatch directory), so a null
    /// pointer's field, which hardware would fault on and gex has no page
    /// protection to fault with. The host raises the access violation
    /// (`cache_op::LOW_ACCESS`), from the block's start, with the registers
    /// as they are here. One compare beside an atomic read-modify-write.
    fn emit_aligned_or_not(
        &mut self,
        a: u32,
        ty: Ty,
        sz: MemSz,
        static_off: u32,
        aligned: impl FnOnce(&mut Self),
        misaligned: Misaligned,
    ) {
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.addr_const(gex_abi::GUEST_FLOOR);
        self.f.addr_lt_u();
        self.f.if_void();
        self.flush_conditional();
        self.store_pc(self.block_start);
        self.f.local_get(self.ctx_local);
        self.f.i64_const(self.block_start as i64);
        self.f.i32_const(gex_abi::cache_op::LOW_ACCESS);
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.addr_to_i64();
        self.f.call(self.cx.cache_op);
        self.f.unreachable();
        self.f.end();
        let mask = (1i32 << natural_align(sz)) - 1;
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.addr_low32();
        self.f.i32_const(mask);
        self.f.byte(0x71); // i32.and
        self.f.if_result(val_type(ty));
        self.emit_misaligned_atomic(a, ty, sz, static_off, misaligned);
        self.f.else_();
        aligned(self);
        self.f.end();
    }

    /// A misaligned locked access, atomically. Inside one aligned 8-byte
    /// word, a compare-exchange loop on that word, which is exactly atomic.
    /// Across two (a 4-byte value at 5..7 mod 8, any misaligned 8-byte one),
    /// under one global lock word ([`gex_abi::SPLIT_LOCK`]): atomic against
    /// every other locked access that spans words, not against a plain
    /// access racing it, which a real split lock also excludes -- so the
    /// first one says so (`cache_op::SPLIT_LOCK`). Leaves the old value.
    fn emit_misaligned_atomic(&mut self, a: u32, ty: Ty, sz: MemSz, static_off: u32, kind: Misaligned) {
        let bytes = 1u32 << natural_align(sz);
        let ea = self.alloc(Self::addr_ir_ty());
        let old = self.alloc(ty);
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.local_set(ea);
        // (ea & 7) + bytes <= 8: within one word.
        self.f.local_get(ea);
        self.f.addr_low32();
        self.f.i32_const(7);
        self.f.byte(0x71); // i32.and
        self.f.i32_const(bytes as i32);
        self.f.byte(0x6a); // i32.add
        self.f.i32_const(8);
        self.f.byte(0x4d); // i32.le_u
        self.f.if_void();
        {
            let wa = self.alloc(Self::addr_ir_ty());
            let sh = self.alloc(Ty::I64);
            let w = self.alloc(Ty::I64);
            let f = self.alloc(Ty::I64);
            let field: i64 = if bytes >= 8 { -1 } else { ((1u64 << (bytes * 8)) - 1) as i64 };
            self.f.local_get(ea);
            self.f.addr_and_const(!7);
            self.f.local_set(wa);
            self.f.local_get(ea);
            self.f.addr_low32();
            self.f.i32_const(7);
            self.f.byte(0x71); // i32.and
            self.f.i32_const(3);
            self.f.byte(0x74); // i32.shl
            self.f.byte(code::I64_EXTEND_I32_U);
            self.f.local_set(sh);
            self.f.block_void();
            self.f.loop_void();
            // w = the word; f = the field in it.
            self.f.local_get(wa);
            self.f.atomic(atomic_load_op(Ty::I64, MemSz::B8), 3, 0);
            self.f.local_tee(w);
            self.f.local_get(sh);
            self.f.byte(0x88); // i64.shr_u
            self.f.i64_const(field);
            self.f.byte(0x83); // i64.and
            self.f.local_set(f);
            // The field's new value, on the stack.
            match kind {
                Misaligned::Cas { expected, replacement } => {
                    // Unequal: done, nothing written.
                    self.f.local_get(f);
                    self.f.local_get(expected);
                    if ty == Ty::I32 {
                        self.f.byte(code::I64_EXTEND_I32_U);
                    }
                    self.f.i64_const(field);
                    self.f.byte(0x83); // i64.and
                    self.f.byte(0x52); // i64.ne
                    self.f.br_if(1);
                    self.f.local_get(replacement);
                    if ty == Ty::I32 {
                        self.f.byte(code::I64_EXTEND_I32_U);
                    }
                }
                Misaligned::Rmw { op, val } => match op {
                    RmwOp::Xchg => {
                        self.f.local_get(val);
                        if ty == Ty::I32 {
                            self.f.byte(code::I64_EXTEND_I32_U);
                        }
                    }
                    _ => {
                        self.f.local_get(f);
                        self.f.local_get(val);
                        if ty == Ty::I32 {
                            self.f.byte(code::I64_EXTEND_I32_U);
                        }
                        let b = match op {
                            RmwOp::Add => BinOp::Add,
                            RmwOp::Sub => BinOp::Sub,
                            RmwOp::And => BinOp::And,
                            RmwOp::Or => BinOp::Or,
                            _ => BinOp::Xor,
                        };
                        self.f.byte(bin_op(b, Ty::I64));
                    }
                },
            }
            self.f.i64_const(field);
            self.f.byte(0x83); // i64.and
            self.f.local_get(sh);
            self.f.byte(0x86); // i64.shl
            // | (w & !(field << sh))
            self.f.local_get(w);
            self.f.i64_const(field);
            self.f.local_get(sh);
            self.f.byte(0x86); // i64.shl
            self.f.i64_const(-1);
            self.f.byte(0x85); // i64.xor
            self.f.byte(0x83); // i64.and
            self.f.byte(0x84); // i64.or
            // cmpxchg(wa, w, that) != w: someone else wrote; again.
            let nw = self.alloc(Ty::I64);
            self.f.local_set(nw);
            self.f.local_get(wa);
            self.f.local_get(w);
            self.f.local_get(nw);
            self.f.atomic(atomic_cmpxchg_op(Ty::I64, MemSz::B8), 3, 0);
            self.f.local_get(w);
            self.f.byte(0x52); // i64.ne
            self.f.br_if(0);
            self.f.end(); // loop
            self.f.end(); // block
            self.f.local_get(f);
            if ty == Ty::I32 {
                self.f.byte(code::I32_WRAP_I64);
            }
            self.f.local_set(old);
        }
        self.f.else_();
        {
            // The first one, out loud.
            self.f.addr_const(gex_abi::SPLIT_LOCK + 4);
            self.f.load(0x28, 2, 0); // i32.load
            self.f.byte(0x45); // i32.eqz
            self.f.if_void();
            self.f.addr_const(gex_abi::SPLIT_LOCK + 4);
            self.f.i32_const(1);
            self.f.store(0x36, 2, 0); // i32.store
            self.f.local_get(self.ctx_local);
            self.f.i64_const(self.block_start as i64);
            self.f.i32_const(gex_abi::cache_op::SPLIT_LOCK);
            self.f.local_get(ea);
            self.f.addr_to_i64();
            self.f.call(self.cx.cache_op);
            self.f.end();
            // Take the lock.
            self.f.loop_void();
            self.f.addr_const(gex_abi::SPLIT_LOCK);
            self.f.i32_const(0);
            self.f.i32_const(1);
            self.f.atomic(atomic_cmpxchg_op(Ty::I32, MemSz::B4), 2, 0);
            self.f.br_if(0);
            self.f.end();
            self.f.local_get(ea);
            self.f.load(load_op(ty, sz, false), 0, 0);
            self.f.local_set(old);
            match kind {
                Misaligned::Cas { expected, replacement } => {
                    self.f.local_get(old);
                    self.f.local_get(expected);
                    self.f.byte(if ty == Ty::I64 { 0x51 } else { 0x46 }); // eq
                    self.f.if_void();
                    self.f.local_get(ea);
                    self.f.local_get(replacement);
                    self.f.store(store_op(ty, sz), 0, 0);
                    self.f.end();
                }
                Misaligned::Rmw { op, val } => {
                    self.f.local_get(ea);
                    match op {
                        RmwOp::Xchg => self.f.local_get(val),
                        _ => {
                            self.f.local_get(old);
                            self.f.local_get(val);
                            let b = match op {
                                RmwOp::Add => BinOp::Add,
                                RmwOp::Sub => BinOp::Sub,
                                RmwOp::And => BinOp::And,
                                RmwOp::Or => BinOp::Or,
                                _ => BinOp::Xor,
                            };
                            self.f.byte(bin_op(b, ty));
                        }
                    }
                    self.f.store(store_op(ty, sz), 0, 0);
                }
            }
            // Let it go.
            self.f.addr_const(gex_abi::SPLIT_LOCK);
            self.f.i32_const(0);
            self.f.atomic(atomic_store_op(Ty::I32, MemSz::B4), 2, 0);
        }
        self.f.end();
        self.f.local_get(old);
    }

    /// Coherent (atomic-when-aligned) load. `addr` (i32) is on the stack; the
    /// effective address is `addr + static_off`. Leaves the loaded value.
    /// Signed loads use an unsigned atomic load plus a sign-extend, so signed
    /// `gint` fields (GObject refcounts/counts/flags) are covered too.
    fn emit_coherent_load(&mut self, ty: Ty, sz: MemSz, signed: bool, static_off: u32) {
        let alogp = natural_align(sz);
        if alogp == 0 {
            // A byte access is always aligned: atomic unconditionally.
            self.f.atomic(atomic_load_op(ty, sz), 0, static_off);
            if signed {
                self.sign_extend_after_atomic(ty, sz);
            }
            return;
        }
        let mask = (1i32 << alogp) - 1;
        let a = self.alloc(Self::addr_ir_ty());
        self.f.local_set(a);
        // (addr + static_off) & mask  -- nonzero iff unaligned.
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.addr_low32();
        self.f.i32_const(mask);
        self.f.byte(0x71); // i32.and
        self.f.if_result(val_type(ty));
        // unaligned: plain load (handles signed natively)
        self.f.local_get(a);
        self.f.load(load_op(ty, sz, signed), alogp, static_off);
        self.f.else_();
        // aligned: atomic (unsigned) load, then sign-extend if needed
        self.f.local_get(a);
        self.f.atomic(atomic_load_op(ty, sz), alogp, static_off);
        if signed {
            self.sign_extend_after_atomic(ty, sz);
        }
        self.f.end();
    }

    /// Coherent (atomic-when-aligned) store. Stack holds `[addr(i32), val(ty)]`
    /// (val on top), matching a plain store's operands. Leaves nothing.
    fn emit_coherent_store(&mut self, ty: Ty, sz: MemSz, static_off: u32) {
        let v = self.alloc(ty);
        self.f.local_set(v);
        let alogp = natural_align(sz);
        if alogp == 0 {
            // addr is on the stack; byte access is always aligned.
            self.f.local_get(v);
            self.f.atomic(atomic_store_op(ty, sz), 0, static_off);
            return;
        }
        let a = self.alloc(Self::addr_ir_ty());
        self.f.local_set(a);
        let mask = (1i32 << alogp) - 1;
        self.f.local_get(a);
        self.f.addr_add_const(static_off);
        self.f.addr_low32();
        self.f.i32_const(mask);
        self.f.byte(0x71); // i32.and
        self.f.if_void();
        // unaligned: plain store
        self.f.local_get(a);
        self.f.local_get(v);
        self.f.store(store_op(ty, sz), alogp, static_off);
        self.f.else_();
        // aligned: atomic store
        self.f.local_get(a);
        self.f.local_get(v);
        self.f.atomic(atomic_store_op(ty, sz), alogp, static_off);
        self.f.end();
    }

    /// `dst = memcmp(a, b, len)` as an inline loop.
    ///
    /// wasm has no `memory.compare`, so unlike `MemCopy` this cannot collapse
    /// to one instruction. A byte at a time, deliberately: eight-at-a-time
    /// needs a byte-swap to find which byte differed and a tail loop, not worth
    /// writing before the simple version is measured. Addresses are guarded per
    /// access, so a wild pointer faults where the guest's would.
    fn emit_memcmp(&mut self, dst: Value, a: Value, b: Value, len: Value) {
        let la = self.alloc(Ty::I64);
        let lb = self.alloc(Ty::I64);
        let llen = self.alloc(Ty::I64);
        let idx = self.alloc(Ty::I64);
        let res = self.alloc(Ty::I32);
        let tb = self.alloc(Ty::I32);

        self.push(a);
        self.f.local_set(la);
        self.push(b);
        self.f.local_set(lb);
        self.push(len);
        self.f.local_set(llen);
        self.f.i64_const(0);
        self.f.local_set(idx);
        self.f.i32_const(0);
        self.f.local_set(res);

        self.f.block_void();
        self.f.loop_void();

        // Done when the whole length has been walked without a difference.
        self.f.local_get(idx);
        self.f.local_get(llen);
        self.f.byte(0x5a); // i64.ge_u
        self.f.br_if(1);

        // The two bytes. Each address is `base + idx`, guarded like any other.
        self.f.local_get(la);
        self.f.local_get(idx);
        self.f.byte(0x7c); // i64.add
        self.guard_to_i32_above(0);
        self.f.load(0x2d, 0, 0); // i32.load8_u
        self.f.local_get(lb);
        self.f.local_get(idx);
        self.f.byte(0x7c);
        self.guard_to_i32_above(0);
        self.f.load(0x2d, 0, 0);
        self.f.local_set(tb);

        // C's answer is the difference of the two as `unsigned char`, which
        // is what `load8_u` already produced. Non-zero means done.
        self.f.local_get(tb);
        self.f.byte(0x6b); // i32.sub
        self.f.local_tee(res);
        self.f.br_if(1);

        self.f.local_get(idx);
        self.f.i64_const(1);
        self.f.byte(0x7c);
        self.f.local_set(idx);
        self.f.br(0);

        self.f.end(); // loop
        self.f.end(); // block

        let local = self.alloc(Ty::I32);
        self.f.local_get(res);
        self.f.local_set(local);
        let i = dst.0 as usize;
        self.value_ty[i] = Ty::I32;
        self.value_local[i] = Some(local);
    }

    fn def(&mut self, dst: Value, expr: &Expr) {
        let ty = expr.ty();
        let idx = dst.0 as usize;
        self.value_ty[idx] = ty;
        let is_ctx = match expr {
            Expr::ReadReg(GuestReg::CtxAddr) => true,
            Expr::Bin { op: BinOp::Add, a, b, .. } => self.is_ctx(*a) || self.is_ctx(*b),
            _ => false,
        };
        if self.value_ctx.len() <= idx {
            self.value_ctx.resize(idx + 1, false);
        }
        self.value_ctx[idx] = is_ctx;
        if let Some(f) = self.value_fp.get_mut(idx) {
            *f = None;
        }

        match expr {
            Expr::ConstI32(c) => {
                self.value_const[idx] = Some(*c as u64);
                // Constants are re-materialised at each use rather than
                // occupying a local; `push` handles that.
                return;
            }
            Expr::ConstI64(c) => {
                self.value_const[idx] = Some(*c);
                return;
            }
            Expr::ReadReg(r) => {
                let local = self.reg_local(*r);
                self.f.local_get(local);
            }
            Expr::Bin { op, ty, a, b } => {
                self.push(*a);
                self.push(*b);
                self.f.byte(bin_op(*op, *ty));
            }
            Expr::Un { op, ty, a } => match op {
                UnOp::Not => {
                    self.push(*a);
                    self.const_of(*ty, -1i64 as u64);
                    self.f.byte(bin_op(BinOp::Xor, *ty));
                }
                UnOp::Neg => {
                    self.const_of(*ty, 0);
                    self.push(*a);
                    self.f.byte(bin_op(BinOp::Sub, *ty));
                }
                UnOp::Clz => {
                    self.push(*a);
                    self.f.byte(if *ty == Ty::I32 { 0x67 } else { 0x79 });
                }
                UnOp::Ctz => {
                    self.push(*a);
                    self.f.byte(if *ty == Ty::I32 { 0x68 } else { 0x7a });
                }
                UnOp::Popcnt => {
                    self.push(*a);
                    self.f.byte(if *ty == Ty::I32 { 0x69 } else { 0x7b });
                }
                UnOp::Bswap | UnOp::Bswap16 => {
                    self.push(*a);
                    self.emit_bswap(*op, *ty);
                }
            },
            Expr::Cmp { op, ty, a, b } => {
                self.push(*a);
                self.push(*b);
                self.f.byte(cmp_op(*op, *ty));
            }
            Expr::Select { cond, a, b, .. } => {
                self.push(*a);
                self.push(*b);
                self.push(*cond);
                self.f.select();
            }
            Expr::Wrap(a) => {
                self.push(*a);
                self.f.byte(code::I32_WRAP_I64);
            }
            Expr::Extend { a, signed } => {
                self.push(*a);
                self.f.byte(if *signed {
                    code::I64_EXTEND_I32_S
                } else {
                    code::I64_EXTEND_I32_U
                });
            }
            Expr::ExtendBits {
                a,
                ty,
                bits,
                signed,
            } => {
                self.push(*a);
                if *signed {
                    match (*ty, *bits) {
                        (Ty::I32, 8) => self.f.byte(code::I32_EXTEND8_S),
                        (Ty::I32, 16) => self.f.byte(code::I32_EXTEND16_S),
                        (Ty::I64, 8) => self.f.byte(code::I64_EXTEND8_S),
                        (Ty::I64, 16) => self.f.byte(code::I64_EXTEND16_S),
                        (Ty::I64, 32) => self.f.byte(code::I64_EXTEND32_S),
                        _ => {
                            // Widths the sign-extension opcodes do not cover:
                            // shift up and back down arithmetically.
                            let sh = ty.bits() - bits;
                            self.const_of(*ty, sh as u64);
                            self.f.byte(bin_op(BinOp::Shl, *ty));
                            self.const_of(*ty, sh as u64);
                            self.f.byte(bin_op(BinOp::ShrS, *ty));
                        }
                    }
                } else {
                    let mask = if *bits >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << bits) - 1
                    };
                    self.const_of(*ty, mask);
                    self.f.byte(bin_op(BinOp::And, *ty));
                }
            }
            Expr::Load {
                addr,
                offset,
                sz,
                signed,
                ty,
                order,
            } => {
                let static_off = self.emit_address(*addr, *offset);
                if *order == Order::SeqCst {
                    // Atomic loads have no signed forms; the architecture has
                    // no signed load-acquire either.
                    let sub = atomic_load_op(*ty, *sz);
                    self.f.atomic(sub, natural_align(*sz), static_off);
                } else if coherent_loads() {
                    self.emit_coherent_load(*ty, *sz, *signed, static_off);
                } else {
                    let opcode = load_op(*ty, *sz, *signed);
                    self.f.load(opcode, natural_align(*sz), static_off);
                }
            }
            Expr::Rmw {
                op,
                addr,
                val,
                sz,
                ty,
            } => {
                let static_off = self.emit_address(*addr, 0);
                if gex_abi::X86 && natural_align(*sz) > 0 {
                    self.emit_rmw_any_alignment(*op, *val, *ty, *sz, static_off);
                } else {
                    self.push(*val);
                    let sub = atomic_rmw_op(*op, *ty, *sz);
                    self.f.atomic(sub, natural_align(*sz), static_off);
                }
            }
            Expr::CmpXchg {
                addr,
                expected,
                replacement,
                sz,
                ty,
            } => {
                let static_off = self.emit_address(*addr, 0);
                if gex_abi::X86 && natural_align(*sz) > 0 {
                    self.emit_cmpxchg_any_alignment(*expected, *replacement, *ty, *sz, static_off);
                } else {
                    self.push(*expected);
                    self.push(*replacement);
                    let sub = atomic_cmpxchg_op(*ty, *sz);
                    self.f.atomic(sub, natural_align(*sz), static_off);
                }
            }
            Expr::HostCall { func, args, .. } => {
                // Handled here so the result lands in a local like any other
                // definition.
                self.host_call_inline(*func, args);
            }

            Expr::VLoad {
                addr,
                offset,
                bytes,
            } => self.emit_vload(*addr, *offset, *bytes),
            Expr::VConst(b) => self.f.v128_const(b),
            Expr::VSplat { a, lane_bytes } => {
                self.push(*a);
                self.f.simd(match lane_bytes {
                    1 => simd::I8X16_SPLAT,
                    2 => simd::I16X8_SPLAT,
                    4 => simd::I32X4_SPLAT,
                    _ => simd::I64X2_SPLAT,
                });
            }
            Expr::VBin { op, a, b } => {
                self.push(*a);
                self.push(*b);
                self.f.simd(match op {
                    VBinOp::And => simd::V128_AND,
                    VBinOp::Or => simd::V128_OR,
                    VBinOp::Xor => simd::V128_XOR,
                    VBinOp::AndNot => simd::V128_ANDNOT,
                });
            }
            Expr::VLaneBin { op, lane, a, b } => {
                self.push(*a);
                self.push(*b);
                // The lifter refuses anything wasm cannot express, so this is
                // unreachable rather than a gap.
                match op.wasm_opcode(*lane) {
                    Some(sub) => self.f.simd(sub),
                    None => panic!("no wasm SIMD instruction for {op:?} on {lane:?}"),
                }
            }
            Expr::VLaneUn { op, lane, a } => {
                self.push(*a);
                match op.wasm_opcode(*lane) {
                    Some(sub) => self.f.simd(sub),
                    None => panic!("no wasm SIMD instruction for {op:?} on {lane:?}"),
                }
            }
            Expr::VShift {
                op,
                lane,
                a,
                amount,
            } => {
                self.push(*a);
                self.push(*amount);
                self.f.simd(lane_shift_op(*op, *lane));
            }
            Expr::VShuffle { a, b, lanes } => {
                // A lane index above 31 is not merely wrong, it makes the
                // module unloadable: V8 validates shuffle masks when it
                // compiles, so one bad mask anywhere rejects everything.
                debug_assert!(
                    lanes.iter().all(|l| *l < 32),
                    "shuffle mask out of range: {lanes:?}"
                );
                self.push(*a);
                self.push(*b);
                self.f.simd_shuffle(lanes);
            }
            Expr::VSwizzle { a, idx } => {
                self.push(*a);
                self.push(*idx);
                self.f.simd(simd::I8X16_SWIZZLE);
            }
            Expr::VBitmask { a, lane } => {
                self.push(*a);
                self.f.simd(match lane.bytes() {
                    1 => simd::I8X16_BITMASK,
                    2 => simd::I16X8_BITMASK,
                    4 => simd::I32X4_BITMASK,
                    _ => simd::I64X2_BITMASK,
                });
            }
            Expr::VBitSelect { mask, a, b } => {
                // wasm's bitselect takes (a, b, mask) and yields
                // (a & mask) | (b & !mask).
                self.push(*a);
                self.push(*b);
                self.push(*mask);
                self.f.simd(simd::V128_BITSELECT);
            }
            Expr::VCvt { op, from, a, b } => {
                self.push(*a);
                if let Some(b) = b {
                    self.push(*b);
                }
                self.f.simd(lane_cvt_op(*op, *from));
            }
            Expr::VUn { op, a } => {
                self.push(*a);
                match op {
                    VUnOp::PopcntBytes => self.f.simd(simd::I8X16_POPCNT),
                    VUnOp::ExtAddPairwiseU8 => {
                        self.f.simd(simd::I16X8_EXTADD_PAIRWISE_I8X16_U)
                    }
                    VUnOp::ExtAddPairwiseU16 => {
                        self.f.simd(simd::I32X4_EXTADD_PAIRWISE_I16X8_U)
                    }
                    VUnOp::ExtAddPairwiseS8 => {
                        self.f.simd(simd::I16X8_EXTADD_PAIRWISE_I8X16_S)
                    }
                    VUnOp::ExtAddPairwiseS16 => {
                        self.f.simd(simd::I32X4_EXTADD_PAIRWISE_I16X8_S)
                    }
                }
            }
            Expr::VExtract {
                a,
                lane,
                lane_bytes,
                signed,
                ty,
            } => {
                self.push(*a);
                if self.fp_result && matches!((lane_bytes, ty), (8, Ty::I64) | (4, Ty::I32)) {
                    // f64x2.extract_lane / f32x4.extract_lane: the float
                    // never goes through an integer register.
                    let double = *lane_bytes == 8;
                    self.f.simd_lane(if double { 0x21 } else { 0x1f }, *lane as u8);
                    self.keep_fp(dst, double);
                    return;
                }
                let sub = match (lane_bytes, signed) {
                    (1, true) => simd::I8X16_EXTRACT_LANE_S,
                    (1, false) => simd::I8X16_EXTRACT_LANE_U,
                    (2, true) => simd::I16X8_EXTRACT_LANE_S,
                    (2, false) => simd::I16X8_EXTRACT_LANE_U,
                    (4, _) => simd::I32X4_EXTRACT_LANE,
                    _ => simd::I64X2_EXTRACT_LANE,
                };
                self.f.simd_lane(sub, *lane as u8);
                // Every lane extract narrower than 64 bits produces an i32
                // whatever the node's type says, so an `I64` result (e.g.
                // `smov x0, v0.h[2]`) must be widened or the module fails
                // validation.
                if *ty == Ty::I64 && *lane_bytes < 8 {
                    self.f.byte(if *signed {
                        code::I64_EXTEND_I32_S
                    } else {
                        code::I64_EXTEND_I32_U
                    });
                }
            }
            Expr::VReplace {
                a,
                lane,
                lane_bytes,
                val,
            } => {
                self.push(*a);
                let fp = self.value_fp.get(val.0 as usize).copied().flatten();
                if matches!((lane_bytes, fp), (8, Some(true)) | (4, Some(false))) {
                    // f64x2.replace_lane / f32x4.replace_lane from a float.
                    self.push_fp(*val, *lane_bytes == 8);
                    self.f.simd_lane(if *lane_bytes == 8 { 0x22 } else { 0x20 }, *lane as u8);
                    let local = self.alloc(ty);
                    self.f.local_set(local);
                    self.value_local[idx] = Some(local);
                    return;
                }
                self.push(*val);
                let sub = match lane_bytes {
                    1 => simd::I8X16_REPLACE_LANE,
                    2 => simd::I16X8_REPLACE_LANE,
                    4 => simd::I32X4_REPLACE_LANE,
                    _ => simd::I64X2_REPLACE_LANE,
                };
                self.f.simd_lane(sub, *lane as u8);
            }

            Expr::Mx { mx, e } => {
                self.emit_mx(dst, *mx, e);
                return;
            }
            Expr::FpBin { op, double, a, b } => {
                self.push_fp(*a, *double);
                self.push_fp(*b, *double);
                let base: u8 = match op {
                    FpBinOp::Add => 0x92,
                    FpBinOp::Sub => 0x93,
                    FpBinOp::Mul => 0x94,
                    FpBinOp::Div => 0x95,
                    FpBinOp::Min => 0x96,
                    FpBinOp::Max => 0x97,
                    FpBinOp::Copysign => 0x98,
                };
                // The f64 arithmetic block sits 14 opcodes after the f32 one.
                self.f.byte(if *double { base + 0x0e } else { base });
                if self.keep_fp(dst, *double) {
                    return;
                }
                self.reinterpret_from_fp(*double);
            }
            Expr::FpUn { op, double, a } => {
                self.push_fp(*a, *double);
                let base: u8 = match op {
                    FpUnOp::Abs => 0x8b,
                    FpUnOp::Neg => 0x8c,
                    FpUnOp::Ceil => 0x8d,
                    FpUnOp::Floor => 0x8e,
                    FpUnOp::Trunc => 0x8f,
                    FpUnOp::RoundNearest => 0x90,
                    FpUnOp::Sqrt => 0x91,
                };
                self.f.byte(if *double { base + 0x0e } else { base });
                if self.keep_fp(dst, *double) {
                    return;
                }
                self.reinterpret_from_fp(*double);
            }
            Expr::FpCmp { op, double, a, b } => {
                self.push_fp(*a, *double);
                self.push_fp(*b, *double);
                let base: u8 = match op {
                    FpCmpOp::Eq => 0x5b,
                    FpCmpOp::Ne => 0x5c,
                    FpCmpOp::Lt => 0x5d,
                    FpCmpOp::Gt => 0x5e,
                    FpCmpOp::Le => 0x5f,
                    FpCmpOp::Ge => 0x60,
                };
                self.f.byte(if *double { base + 0x06 } else { base });
            }
            Expr::FpCvt {
                kind,
                src_wide,
                dst_wide,
                a,
            } => {
                if self.emit_fp_convert(*kind, *src_wide, *dst_wide, *a) && self.keep_fp(dst, *dst_wide) {
                    return;
                }
            }
        }

        // Everything that reaches here left one value on the stack.
        let local = self.def_into.take().unwrap_or_else(|| self.alloc(ty));
        self.f.local_set(local);
        self.value_local[idx] = Some(local);
    }

    /// An [`Expr::Mx`]: `e`'s own lowering, kept when `mx`'s rounding
    /// control ([`MX_SPECIAL`]) is to nearest; otherwise the pure `mx`
    /// import, once for a scalar and once per lane ([`MxPlan`]). Either
    /// way the result is left in one integer-typed local.
    fn emit_mx(&mut self, dst: Value, mx: Value, e: &Expr) {
        use code::*;
        let idx = dst.0 as usize;
        let ty = e.ty();
        let res = self.def_into.take().unwrap_or_else(|| self.alloc(ty));
        self.def(dst, e);
        self.push(dst);
        self.f.local_set(res);
        let plan = MxPlan::of(e);
        self.push(mx);
        self.f.i32_const(MX_SPECIAL as i32);
        self.f.byte(I32_AND);
        self.f.if_void();
        // The import's operand: an integer of `in_bytes` (from lane `lane`
        // of a vector), as an i64.
        let operand = |this: &mut Self, v: Value, lane: Option<(u32, u32)>, in_ty: Ty| {
            this.push(v);
            match lane {
                Some((k, 4)) => {
                    this.f.simd_lane(simd::I32X4_EXTRACT_LANE, k as u8);
                    this.f.byte(I64_EXTEND_I32_U);
                }
                Some((k, _)) => this.f.simd_lane(simd::I64X2_EXTRACT_LANE, k as u8),
                None if in_ty == Ty::I32 => this.f.byte(I64_EXTEND_I32_U),
                None => {}
            }
        };
        match plan {
            MxPlan::Scalar { code, a, b, in_ty } => {
                self.f.local_get(self.ctx_local);
                self.f.i32_const(code as i32);
                operand(self, a, None, in_ty);
                match b {
                    Some(b) => operand(self, b, None, in_ty),
                    None => self.f.i64_const(0),
                }
                self.push(mx);
                self.f.call(self.cx.mx);
                if ty == Ty::I32 {
                    self.f.byte(I32_WRAP_I64);
                }
                self.f.local_set(res);
            }
            MxPlan::Lanes { code, a, b, n, in_bytes, out_bytes } => {
                self.f.v128_const(&[0; 16]);
                self.f.local_set(res);
                for k in 0..n {
                    self.f.local_get(res);
                    self.f.local_get(self.ctx_local);
                    self.f.i32_const(code as i32);
                    operand(self, a, Some((k, in_bytes)), Ty::I64);
                    match b {
                        Some(b) => operand(self, b, Some((k, in_bytes)), Ty::I64),
                        None => self.f.i64_const(0),
                    }
                    self.push(mx);
                    self.f.call(self.cx.mx);
                    if out_bytes == 4 {
                        self.f.byte(I32_WRAP_I64);
                        self.f.simd_lane(simd::I32X4_REPLACE_LANE, k as u8);
                    } else {
                        self.f.simd_lane(simd::I64X2_REPLACE_LANE, k as u8);
                    }
                    self.f.local_set(res);
                }
            }
        }
        self.f.end();
        self.value_ty[idx] = ty;
        if let Some(f) = self.value_fp.get_mut(idx) {
            *f = None;
        }
        self.value_const[idx] = None;
        self.value_local[idx] = Some(res);
    }

    /// Reinterpret the integer bit pattern on top of the stack as a float.
    /// Reverse the bytes of the value on the stack. Wasm
    /// has no byte swap: 32 bits is two rotates, two masks and an or; 16 bits
    /// two shifts, two masks and an or; 64 bits a `pshufb`-shaped swizzle,
    /// which beats the eleven-op scalar sequence and its two 64-bit masks.
    fn emit_bswap(&mut self, op: UnOp, ty: Ty) {
        const I32_AND: u8 = 0x71;
        const I32_OR: u8 = 0x72;
        const I32_SHL: u8 = 0x74;
        const I32_SHR_U: u8 = 0x76;
        const I32_ROTL: u8 = 0x77;
        const I32_ROTR: u8 = 0x78;
        match (op, ty) {
            (UnOp::Bswap, Ty::I32) => {
                let t = self.scratch_i32;
                self.f.local_tee(t);
                self.f.i32_const(8);
                self.f.byte(I32_ROTL);
                self.f.i32_const(0x00ff_00ff);
                self.f.byte(I32_AND);
                self.f.local_get(t);
                self.f.i32_const(8);
                self.f.byte(I32_ROTR);
                self.f.i32_const(0xff00_ff00u32 as i32);
                self.f.byte(I32_AND);
                self.f.byte(I32_OR);
            }
            (UnOp::Bswap16, Ty::I32) => {
                let t = self.scratch_i32;
                self.f.local_tee(t);
                self.f.i32_const(0xff);
                self.f.byte(I32_AND);
                self.f.i32_const(8);
                self.f.byte(I32_SHL);
                self.f.local_get(t);
                self.f.i32_const(8);
                self.f.byte(I32_SHR_U);
                self.f.i32_const(0xff);
                self.f.byte(I32_AND);
                self.f.byte(I32_OR);
            }
            (UnOp::Bswap, Ty::I64) => {
                self.f.simd(simd::I64X2_SPLAT);
                let mut idx = [0u8; 16];
                for (i, b) in idx.iter_mut().enumerate() {
                    *b = if i < 8 { 7 - i as u8 } else { i as u8 };
                }
                self.f.v128_const(&idx);
                self.f.simd(simd::I8X16_SWIZZLE);
                self.f.simd_lane(simd::I64X2_EXTRACT_LANE, 0);
            }
            (UnOp::Bswap16, Ty::I64) => {
                self.f.byte(code::I32_WRAP_I64);
                self.emit_bswap(UnOp::Bswap16, Ty::I32);
                self.f.byte(code::I64_EXTEND_I32_U);
            }
            _ => unreachable!("byte swap of a {ty:?}"),
        }
    }

    fn reinterpret_to_fp(&mut self, double: bool) {
        // f64.reinterpret_i64 / f32.reinterpret_i32
        self.f.byte(if double { 0xbf } else { 0xbe });
    }

    /// Reinterpret the float on top of the stack back to its bit pattern.
    fn reinterpret_from_fp(&mut self, double: bool) {
        // i64.reinterpret_f64 / i32.reinterpret_f32
        self.f.byte(if double { 0xbd } else { 0xbc });
    }

    /// The conversion, its result as an integer (bit pattern) unless it
    /// is a float `def` keeps as one (`fp_result`): then `true`, the float
    /// on the stack.
    fn emit_fp_convert(&mut self, kind: FpConvert, src_wide: bool, dst_wide: bool, a: Value) -> bool {
        match kind {
            FpConvert::FpToFp => {
                self.push_fp(a, src_wide);
                // f32.demote_f64 / f64.promote_f32
                self.f.byte(if dst_wide { 0xbb } else { 0xb6 });
                if self.fp_result {
                    return true;
                }
                self.reinterpret_from_fp(dst_wide);
            }
            FpConvert::IntToFp { signed } => {
                self.push(a);
                let op: u8 = match (dst_wide, src_wide, signed) {
                    (false, false, true) => 0xb2,  // f32.convert_i32_s
                    (false, false, false) => 0xb3, // f32.convert_i32_u
                    (false, true, true) => 0xb4,   // f32.convert_i64_s
                    (false, true, false) => 0xb5,  // f32.convert_i64_u
                    (true, false, true) => 0xb7,   // f64.convert_i32_s
                    (true, false, false) => 0xb8,  // f64.convert_i32_u
                    (true, true, true) => 0xb9,    // f64.convert_i64_s
                    (true, true, false) => 0xba,   // f64.convert_i64_u
                };
                self.f.byte(op);
                if self.fp_result {
                    return true;
                }
                self.reinterpret_from_fp(dst_wide);
            }
            FpConvert::FpToInt { signed } => {
                self.push_fp(a, src_wide);
                // The *saturating* conversions, because AArch64's FCVTZ*
                // clamps out-of-range values and maps NaN to zero, while the
                // trapping forms would abort the guest.
                let sub: u32 = match (dst_wide, src_wide, signed) {
                    (false, false, true) => 0x00,
                    (false, false, false) => 0x01,
                    (false, true, true) => 0x02,
                    (false, true, false) => 0x03,
                    (true, false, true) => 0x04,
                    (true, false, false) => 0x05,
                    (true, true, true) => 0x06,
                    (true, true, false) => 0x07,
                };
                self.f.byte(0xfc);
                self.f.u32(sub);
            }
        }
        false
    }

    /// Push `v` as a float (`f64` when `double`).
    fn push_fp(&mut self, v: Value, double: bool) {
        let idx = v.0 as usize;
        if self.value_fp.get(idx).copied().flatten() == Some(double) {
            if let Some(l) = self.value_local[idx] {
                self.f.local_get(l);
                return;
            }
        }
        self.push(v);
        self.reinterpret_to_fp(double);
    }

    /// Function mode: when `fp_result` is set, take the float on the stack
    /// as `dst`'s value, kept as a float, and say so.
    fn keep_fp(&mut self, dst: Value, double: bool) -> bool {
        if !std::mem::take(&mut self.fp_result) {
            return false;
        }
        let idx = dst.0 as usize;
        let pool = if double { &mut self.free_f64 } else { &mut self.free_f32 };
        let local = match pool.pop() {
            Some(l) => l,
            None => self.f.add_local(if double { ValType::F64 } else { ValType::F32 }),
        };
        self.f.local_set(local);
        self.value_local[idx] = Some(local);
        self.value_fp[idx] = Some(double);
        true
    }

    /// Push a `ty`-typed constant.
    fn const_of(&mut self, ty: Ty, v: u64) {
        match ty {
            Ty::I32 => self.f.i32_const(v as i32),
            Ty::I64 => self.f.i64_const(v as i64),
            Ty::V128 => {
                let mut b = [0u8; 16];
                b[..8].copy_from_slice(&v.to_le_bytes());
                self.f.v128_const(&b);
            }
        }
    }

    /// A SIMD&FP load. Anything narrower than 16 bytes must zero the rest of
    /// the destination register, which is exactly what the `load*_zero` and
    /// `load*_lane`-into-zero forms give us.
    fn emit_vload(&mut self, addr: Value, offset: i64, bytes: u32) {
        let align = bytes.trailing_zeros();
        match bytes {
            16 => {
                let off = self.emit_address(addr, offset);
                self.f.simd_mem(simd::V128_LOAD, align, off);
            }
            8 => {
                let off = self.emit_address(addr, offset);
                self.f.simd_mem(simd::V128_LOAD64_ZERO, align, off);
            }
            4 => {
                let off = self.emit_address(addr, offset);
                self.f.simd_mem(simd::V128_LOAD32_ZERO, align, off);
            }
            // There is no `load8_zero`/`load16_zero`, so load into lane 0 of
            // an all-zero vector.
            _ => {
                let off = self.emit_address(addr, offset);
                self.f.v128_const(&[0u8; 16]);
                let sub = if bytes == 1 {
                    simd::V128_LOAD8_LANE
                } else {
                    simd::V128_LOAD16_LANE
                };
                self.f.simd_mem_lane(sub, align, off, 0);
            }
        }
    }

    fn type_of(&self, v: Value) -> Ty {
        self.value_ty.get(v.0 as usize).copied().unwrap_or(Ty::I64)
    }

    /// A [`Stmt::X87`]: the common operands inline, the rest through the
    /// pure `x87` import (which computes them with `gex_f80`, the same
    /// algorithm, so either way the bits are the hardware's).
    ///
    /// The fast paths: a double or single whose result is normal and cannot
    /// overflow even when rounding carries -- rounded by RC in a dozen
    /// instructions -- or a zero, narrowed to its sign; and an integer from
    /// a value of magnitude at least one that fits (out of range gives the
    /// indefinite with IE), or zero.
    fn emit_x87(&mut self, op: X87Op, args: &[Value], outs: &[Value]) {
        use code::*;
        if matches!(op, X87Op::Arith(_) | X87Op::Compare { .. }) {
            return self.emit_x87_binary(op, args, outs);
        }
        let sig = self.alloc(Ty::I64);
        let se = self.alloc(Ty::I32);
        let cw = self.alloc(Ty::I32);
        let res = self.alloc(Ty::I64);
        let sw = self.alloc(Ty::I32);
        let e = self.alloc(Ty::I32);
        let sign = self.alloc(Ty::I32);
        let m = self.alloc(Ty::I64);
        let rem = self.alloc(Ty::I64);
        let half = self.alloc(Ty::I64);
        let up = self.alloc(Ty::I32);
        self.push(args[0]);
        self.f.local_set(sig);
        self.push(args[1]);
        self.f.local_set(se);
        self.push(args[2]);
        self.f.local_set(cw);
        // The biased exponent and the sign.
        self.f.local_get(se);
        self.f.i32_const(0x7fff);
        self.f.byte(I32_AND);
        self.f.local_set(e);
        self.f.local_get(se);
        self.f.i32_const(15);
        self.f.byte(I32_SHR_U);
        self.f.i32_const(1);
        self.f.byte(I32_AND);
        self.f.local_set(sign);

        match op {
            X87Op::Narrow { single } => {
                // Dropped bits, the normal range that cannot overflow on a
                // carry, and the bias adjustment that folds the significand's
                // integer bit into the exponent field.
                let (drop, frac, total, bias) = if single { (40, 23, 32, 127) } else { (11, 52, 64, 1023) };
                let lo = 16383 - (bias - 1); // smallest normal
                let hi = 16383 + (bias - 1); // a carry stays finite
                // J set and lo <= e <= hi.
                self.f.local_get(sig);
                self.f.i64_const(0);
                self.f.byte(I64_LT_S);
                self.f.local_get(e);
                self.f.i32_const(lo);
                self.f.byte(I32_SUB);
                self.f.i32_const(hi - lo);
                self.f.byte(I32_LE_U);
                self.f.byte(I32_AND);
                self.f.if_void();
                self.f.local_get(sig);
                self.f.i64_const(drop);
                self.f.byte(I64_SHR_U);
                self.f.local_set(m);
                self.f.local_get(sig);
                self.f.i64_const((1i64 << drop) - 1);
                self.f.byte(I64_AND);
                self.f.local_set(rem);
                self.f.i64_const(1i64 << (drop - 1));
                self.f.local_set(half);
                self.emit_x87_round_up(m, rem, half, sign, cw);
                self.f.local_set(up);
                // bits = ((e - (16383 - bias) - 1) << frac) + m + up, signed.
                self.f.local_get(e);
                self.f.i32_const(16383 - bias + 1);
                self.f.byte(I32_SUB);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.i64_const(frac);
                self.f.byte(I64_SHL);
                self.f.local_get(m);
                self.f.byte(I64_ADD);
                self.f.local_get(up);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.byte(I64_ADD);
                self.f.local_get(sign);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.i64_const(total - 1);
                self.f.byte(I64_SHL);
                self.f.byte(I64_OR);
                self.f.local_set(res);
                self.emit_x87_inexact_sw(rem, up);
                self.f.local_set(sw);
                self.f.else_();
                // A zero: its sign alone, exact.
                self.f.local_get(sig);
                self.f.byte(I64_EQZ);
                self.f.local_get(e);
                self.f.byte(I32_EQZ);
                self.f.byte(I32_AND);
                self.f.if_void();
                self.f.local_get(sign);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.i64_const(total - 1);
                self.f.byte(I64_SHL);
                self.f.local_set(res);
                self.f.i32_const(0);
                self.f.local_set(sw);
                self.f.else_();
                self.emit_x87_call(op, sig, se, cw, res, sw);
                self.f.end();
                self.f.end();
            }
            X87Op::ToInt { bits, trunc } => {
                let bits = bits as i64;
                // Zero converts to zero, raising nothing.
                self.f.local_get(sig);
                self.f.byte(I64_EQZ);
                self.f.local_get(e);
                self.f.byte(I32_EQZ);
                self.f.byte(I32_AND);
                self.f.if_void();
                self.f.i64_const(0);
                self.f.local_set(res);
                self.f.i32_const(0);
                self.f.local_set(sw);
                self.f.else_();
                // J set and 1 <= |v| < 2^(bits - 1): 16383 <= e <= 16383 + bits - 2.
                self.f.local_get(sig);
                self.f.i64_const(0);
                self.f.byte(I64_LT_S);
                self.f.local_get(e);
                self.f.i32_const(16383);
                self.f.byte(I32_SUB);
                self.f.i32_const(bits as i32 - 2);
                self.f.byte(I32_LE_U);
                self.f.byte(I32_AND);
                self.f.if_void();
                // Bits below the binary point: 63 - (e - 16383), 1..=63.
                self.f.i32_const(16383 + 63);
                self.f.local_get(e);
                self.f.byte(I32_SUB);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.local_set(half); // the shift, for now
                self.f.local_get(sig);
                self.f.local_get(half);
                self.f.byte(I64_SHR_U);
                self.f.local_set(m);
                self.f.local_get(sig);
                self.f.i64_const(1);
                self.f.local_get(half);
                self.f.byte(I64_SHL);
                self.f.i64_const(1);
                self.f.byte(I64_SUB);
                self.f.byte(I64_AND);
                self.f.local_set(rem);
                self.f.i64_const(1);
                self.f.local_get(half);
                self.f.i64_const(1);
                self.f.byte(I64_SUB);
                self.f.byte(I64_SHL);
                self.f.local_set(half);
                if trunc {
                    self.f.i32_const(0);
                } else {
                    self.emit_x87_round_up(m, rem, half, sign, cw);
                }
                self.f.local_set(up);
                // k = m + up, at most 2^(bits-1): over the limit for a
                // positive value is out of range.
                self.f.local_get(m);
                self.f.local_get(up);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.byte(I64_ADD);
                self.f.local_set(m);
                // Over: k > 2^(bits-1) - 1 + sign.
                self.f.local_get(m);
                self.f.i64_const((1i64 << (bits - 1)).wrapping_sub(1));
                self.f.local_get(sign);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.byte(I64_ADD);
                self.f.byte(I64_GT_U);
                self.f.if_void();
                self.f.i64_const(1i64 << (bits - 1));
                self.f.local_set(res);
                self.f.i32_const(1); // IE
                self.f.local_set(sw);
                self.f.else_();
                // sign ? -k : k, in `bits` bits.
                self.f.i64_const(0);
                self.f.local_get(m);
                self.f.byte(I64_SUB);
                self.f.local_get(m);
                self.f.local_get(sign);
                self.f.select();
                if bits < 64 {
                    self.f.i64_const((1i64 << bits) - 1);
                    self.f.byte(I64_AND);
                }
                self.f.local_set(res);
                self.emit_x87_inexact_sw(rem, up);
                self.f.local_set(sw);
                self.f.end();
                self.f.else_();
                self.emit_x87_call(op, sig, se, cw, res, sw);
                self.f.end();
                self.f.end();
            }
            X87Op::Arith(_) | X87Op::Compare { .. } => unreachable!("lowered by emit_x87_binary"),
        }
        for (i, (o, l)) in outs.iter().zip([res, sw]).enumerate() {
            let idx = o.0 as usize;
            self.value_ty[idx] = op.out_types()[i];
            self.value_local[idx] = Some(l);
        }
        // The scratch locals go back to the pools; `res` and `sw` now hold
        // the outputs.
        self.free_i64.extend([sig, m, rem, half]);
        self.free_i32.extend([se, cw, e, sign, up]);
    }

    /// Push whether to round the kept magnitude `m` up, given the dropped
    /// bits `rem` and half an ulp `half` (I64 locals), the sign and the
    /// control word: to nearest when over half, or at half with `m` odd;
    /// away from zero (RC down for a negative value, up for a positive one)
    /// whenever anything was dropped; toward zero never.
    fn emit_x87_round_up(&mut self, m: u32, rem: u32, half: u32, sign: u32, cw: u32) {
        use code::*;
        // RC == 0 && (rem > half || (rem == half && m odd))
        self.f.local_get(rem);
        self.f.local_get(half);
        self.f.byte(I64_GT_U);
        self.f.local_get(rem);
        self.f.local_get(half);
        self.f.byte(I64_EQ);
        self.f.local_get(m);
        self.f.byte(I32_WRAP_I64);
        self.f.i32_const(1);
        self.f.byte(I32_AND);
        self.f.byte(I32_AND);
        self.f.byte(I32_OR);
        self.f.local_get(cw);
        self.f.i32_const(0xc00);
        self.f.byte(I32_AND);
        self.f.byte(I32_EQZ);
        self.f.byte(I32_AND);
        // RC == 2 - sign && rem != 0
        self.f.local_get(cw);
        self.f.i32_const(10);
        self.f.byte(I32_SHR_U);
        self.f.i32_const(3);
        self.f.byte(I32_AND);
        self.f.i32_const(2);
        self.f.local_get(sign);
        self.f.byte(I32_SUB);
        self.f.byte(I32_EQ);
        self.f.local_get(rem);
        self.f.i64_const(0);
        self.f.byte(I64_NE);
        self.f.byte(I32_AND);
        self.f.byte(I32_OR);
    }

    /// Push the status bits of a rounded result: PE when anything was
    /// dropped, C1 when it rounded up.
    fn emit_x87_inexact_sw(&mut self, rem: u32, up: u32) {
        use code::*;
        self.f.local_get(rem);
        self.f.i64_const(0);
        self.f.byte(I64_NE);
        self.f.i32_const(5);
        self.f.byte(I32_SHL);
        self.f.local_get(up);
        self.f.i32_const(9);
        self.f.byte(I32_SHL);
        self.f.byte(I32_OR);
    }

    /// The slow path of a one-operand operation: the pure `x87` import,
    /// whose status bits come back in the context.
    fn emit_x87_call(&mut self, op: X87Op, sig: u32, se: u32, cw: u32, res: u32, sw: u32) {
        self.f.local_get(self.ctx_local);
        self.f.i32_const(op.code());
        self.f.local_get(sig);
        self.f.local_get(se);
        self.f.i64_const(0);
        self.f.i32_const(0);
        self.f.local_get(cw);
        self.f.call(self.cx.x87);
        self.f.local_set(res);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 12) as u32); // i32.load
        self.f.local_set(sw);
    }

    /// The two-operand operations: `Arith` (outputs: significand,
    /// sign/exponent, status bits) and `Compare` (condition codes, status
    /// bits), through the pure import.
    fn emit_x87_binary(&mut self, op: X87Op, args: &[Value], outs: &[Value]) {
        if let X87Op::Compare { .. } = op {
            return self.emit_x87_compare(op, args, outs);
        }
        if let X87Op::Arith(k @ (X87Arith::Add | X87Arith::Sub | X87Arith::Mul)) = op {
            return self.emit_x87_arith_fast(op, k, args, outs);
        }
        let res = self.alloc(Ty::I64);
        let se = self.alloc(Ty::I32);
        let sw = self.alloc(Ty::I32);
        self.f.local_get(self.ctx_local);
        self.f.i32_const(op.code());
        for a in &args[..4] {
            self.push(*a);
        }
        match args.get(4) {
            Some(cw) => self.push(*cw),
            None => self.f.i32_const(0x37f),
        }
        self.f.call(self.cx.x87);
        self.f.local_set(res);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 8) as u32); // i32.load
        self.f.local_set(se);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 12) as u32);
        self.f.local_set(sw);
        let locals: Vec<(u32, Ty)> = match op {
            X87Op::Compare { .. } => {
                // The condition codes come back as the returned value.
                self.f.local_get(res);
                self.f.byte(code::I32_WRAP_I64);
                self.f.local_set(se);
                self.free_i64.push(res);
                vec![(se, Ty::I32), (sw, Ty::I32)]
            }
            _ => vec![(res, Ty::I64), (se, Ty::I32), (sw, Ty::I32)],
        };
        for (o, (l, ty)) in outs.iter().zip(locals) {
            let idx = o.0 as usize;
            self.value_ty[idx] = ty;
            self.value_local[idx] = Some(l);
        }
    }

    /// `Arith` add, subtract and multiply: inline for normal operands whose
    /// exact result (a 128-bit significand) is normal before rounding and
    /// stays finite after -- for addition and subtraction also exponents
    /// at most 63 apart, and for an effective subtraction at least 2 apart,
    /// so at most one bit cancels -- then rounded by the control word's
    /// precision and rounding control exactly as gex_f80 rounds. A zero
    /// operand with a zero or normal one is inline too, as in x87.h: a sum
    /// is the normal operand, rounded the same way. Everything else (and
    /// every division) goes to the import.
    fn emit_x87_arith_fast(&mut self, op: X87Op, k: X87Arith, args: &[Value], outs: &[Value]) {
        use code::*;
        const I64_MUL: u8 = 0x7e;
        const I64_GE_U: u8 = 0x5a;
        const I32_XOR: u8 = 0x73;
        const I32_LT_U: u8 = 0x49;
        const I32_GE_U: u8 = 0x4f;
        const I32_ADD: u8 = 0x6a;
        const I64_SUB_: u8 = 0x7d;
        let l64 = |t: &mut Self| t.alloc(Ty::I64);
        let l32 = |t: &mut Self| t.alloc(Ty::I32);
        let (asig, bsig, hi, lo, t1, t2, t3, t4, dd, kept) =
            (l64(self), l64(self), l64(self), l64(self), l64(self), l64(self), l64(self), l64(self), l64(self), l64(self));
        let (ase, bse, cw, ae, be, sign, e, up, rb, st) =
            (l32(self), l32(self), l32(self), l32(self), l32(self), l32(self), l32(self), l32(self), l32(self), l32(self));
        let res = self.alloc(Ty::I64);
        let se = self.alloc(Ty::I32);
        let sw = self.alloc(Ty::I32);
        for (v, l) in [(args[0], asig), (args[1], ase), (args[2], bsig), (args[3], bse), (args[4], cw)] {
            self.push(v);
            self.f.local_set(l);
        }
        for (x, ex) in [(ase, ae), (bse, be)] {
            self.f.local_get(x);
            self.f.i32_const(0x7fff);
            self.f.byte(I32_AND);
            self.f.local_set(ex);
        }
        self.f.block_void(); // $out
        self.f.block_void(); // $slow
        self.f.block_void(); // $round: hi, lo, e and sign set
        // Zeros (rb = a is zero, st = b is zero, for now), as x87.h's
        // gex_x87_arith: two zeros give a zero; a product with a zero is a
        // zero of the signs' exclusive or; a sum with a zero is the other,
        // normal operand, rounded to the precision control below.
        for (sig, ex, z) in [(asig, ae, rb), (bsig, be, st)] {
            self.f.local_get(sig);
            self.f.byte(I64_EQZ);
            self.f.local_get(ex);
            self.f.byte(I32_EQZ);
            self.f.byte(I32_AND);
            self.f.local_set(z);
        }
        // up = b's sign as the operation sees it (flipped for 0 - b).
        self.f.local_get(bse);
        if k == X87Arith::Sub {
            self.f.i32_const(0x8000);
            self.f.byte(I32_XOR);
        }
        self.f.i32_const(15);
        self.f.byte(I32_SHR_U);
        self.f.i32_const(1);
        self.f.byte(I32_AND);
        self.f.local_set(up);
        self.f.local_get(rb);
        self.f.local_get(st);
        self.f.byte(I32_OR);
        self.f.if_void();
        {
            self.f.local_get(rb);
            self.f.local_get(st);
            self.f.byte(I32_AND);
            self.f.if_void();
            {
                // Two zeros: the product's sign is the exclusive or; a sum
                // is -0 only if both are (or, rounding down, either is).
                self.f.local_get(ase);
                self.f.i32_const(15);
                self.f.byte(I32_SHR_U);
                self.f.i32_const(1);
                self.f.byte(I32_AND);
                self.f.local_get(up);
                if k == X87Arith::Mul {
                    self.f.byte(I32_XOR);
                } else {
                    self.f.local_set(sign);
                    self.f.local_set(e);
                    self.f.local_get(e);
                    self.f.local_get(sign);
                    self.f.byte(code::I32_OR);
                    self.f.local_get(e);
                    self.f.local_get(sign);
                    self.f.byte(I32_AND);
                    self.f.local_get(cw);
                    self.f.i32_const(0xc00);
                    self.f.byte(I32_AND);
                    self.f.i32_const(0x400);
                    self.f.byte(I32_EQ);
                    self.f.select();
                }
                self.f.i32_const(15);
                self.f.byte(code::I32_SHL);
                self.f.local_set(se);
                self.f.i64_const(0);
                self.f.local_set(res);
                self.f.i32_const(0);
                self.f.local_set(sw);
                self.f.br(4); // $out (past both ifs, $round and $slow)
            }
            self.f.end();
            // One zero; the other must be normal.
            self.f.local_get(bsig);
            self.f.local_get(asig);
            self.f.local_get(rb);
            self.f.select();
            self.f.local_set(hi); // the other operand
            self.f.local_get(be);
            self.f.local_get(ae);
            self.f.local_get(rb);
            self.f.select();
            self.f.local_set(e);
            self.f.local_get(hi);
            self.f.i64_const(0);
            self.f.byte(I64_LT_S);
            self.f.local_get(e);
            self.f.i32_const(1);
            self.f.byte(I32_SUB);
            self.f.i32_const(0x7ffd);
            self.f.byte(I32_LE_U);
            self.f.byte(I32_AND);
            self.f.byte(I32_EQZ);
            self.f.br_if(2); // $slow
            if k == X87Arith::Mul {
                self.f.local_get(ase);
                self.f.local_get(bse);
                self.f.byte(I32_XOR);
                self.f.i32_const(0x8000);
                self.f.byte(I32_AND);
                self.f.local_set(se);
                self.f.i64_const(0);
                self.f.local_set(res);
                self.f.i32_const(0);
                self.f.local_set(sw);
                self.f.br(3); // $out
            } else {
                // sign = a zero ? b's (as seen) : a's.
                self.f.local_get(up);
                self.f.local_get(ase);
                self.f.i32_const(15);
                self.f.byte(I32_SHR_U);
                self.f.i32_const(1);
                self.f.byte(I32_AND);
                self.f.local_get(rb);
                self.f.select();
                self.f.local_set(sign);
                self.f.i64_const(0);
                self.f.local_set(lo);
                self.f.br(1); // $round
            }
        }
        self.f.end();
        // Both normal: J set and 1 <= e <= 0x7ffe.
        for (sig, ex) in [(asig, ae), (bsig, be)] {
            self.f.local_get(sig);
            self.f.i64_const(0);
            self.f.byte(I64_LT_S);
            self.f.local_get(ex);
            self.f.i32_const(1);
            self.f.byte(I32_SUB);
            self.f.i32_const(0x7ffd);
            self.f.byte(I32_LE_U);
            self.f.byte(I32_AND);
        }
        self.f.byte(I32_AND);
        self.f.byte(I32_EQZ);
        self.f.br_if(1); // $slow
        if k == X87Arith::Mul {
            // sign = sa ^ sb.
            self.f.local_get(ase);
            self.f.local_get(bse);
            self.f.byte(I32_XOR);
            self.f.i32_const(15);
            self.f.byte(I32_SHR_U);
            self.f.local_set(sign);
            // The 128-bit product from 32-bit halves: t1 = al*bl,
            // t2 = al*bh, t3 = ah*bl, t4 = ah*bh.
            let half = |t: &mut Self, sig: u32, high: bool| {
                t.f.local_get(sig);
                if high {
                    t.f.i64_const(32);
                    t.f.byte(I64_SHR_U);
                } else {
                    t.f.i64_const(0xffff_ffff);
                    t.f.byte(I64_AND);
                }
            };
            for (dst, ah, bh) in [(t1, false, false), (t2, false, true), (t3, true, false), (t4, true, true)] {
                half(self, asig, ah);
                half(self, bsig, bh);
                self.f.byte(I64_MUL);
                self.f.local_set(dst);
            }
            // mid = (t1 >> 32) + (t2 & M) + (t3 & M), in lo for now.
            self.f.local_get(t1);
            self.f.i64_const(32);
            self.f.byte(I64_SHR_U);
            self.f.local_get(t2);
            self.f.i64_const(0xffff_ffff);
            self.f.byte(I64_AND);
            self.f.byte(I64_ADD);
            self.f.local_get(t3);
            self.f.i64_const(0xffff_ffff);
            self.f.byte(I64_AND);
            self.f.byte(I64_ADD);
            self.f.local_set(lo);
            // hi = t4 + (t2 >> 32) + (t3 >> 32) + (mid >> 32)
            self.f.local_get(t4);
            self.f.local_get(t2);
            self.f.i64_const(32);
            self.f.byte(I64_SHR_U);
            self.f.byte(I64_ADD);
            self.f.local_get(t3);
            self.f.i64_const(32);
            self.f.byte(I64_SHR_U);
            self.f.byte(I64_ADD);
            self.f.local_get(lo);
            self.f.i64_const(32);
            self.f.byte(I64_SHR_U);
            self.f.byte(I64_ADD);
            self.f.local_set(hi);
            // lo = (t1 & M) | (mid << 32)
            self.f.local_get(t1);
            self.f.i64_const(0xffff_ffff);
            self.f.byte(I64_AND);
            self.f.local_get(lo);
            self.f.i64_const(32);
            self.f.byte(I64_SHL);
            self.f.byte(I64_OR);
            self.f.local_set(lo);
            // e = ae + be - 16383 + top; normalise by n = 1 - top.
            self.f.local_get(hi);
            self.f.i64_const(63);
            self.f.byte(I64_SHR_U);
            self.f.local_set(t1); // top
            self.f.local_get(ae);
            self.f.local_get(be);
            self.f.byte(I32_ADD);
            self.f.i32_const(16383);
            self.f.byte(I32_SUB);
            self.f.local_get(t1);
            self.f.byte(I32_WRAP_I64);
            self.f.byte(I32_ADD);
            self.f.local_set(e);
            self.f.i64_const(1);
            self.f.local_get(t1);
            self.f.byte(I64_SUB_);
            self.f.local_set(t2); // n
            self.f.local_get(hi);
            self.f.local_get(t2);
            self.f.byte(I64_SHL);
            self.f.local_get(lo);
            self.f.i64_const(63);
            self.f.byte(I64_SHR_U);
            self.f.local_get(t2);
            self.f.byte(I64_AND);
            self.f.byte(I64_OR);
            self.f.local_set(hi);
            self.f.local_get(lo);
            self.f.local_get(t2);
            self.f.byte(I64_SHL);
            self.f.local_set(lo);
        } else {
            // b's sign, flipped for a subtraction.
            self.f.local_get(bse);
            if k == X87Arith::Sub {
                self.f.i32_const(0x8000);
                self.f.byte(I32_XOR);
            }
            self.f.local_set(bse);
            // swap = |a| < |b|: then X = b, Y = a.
            self.f.local_get(ae);
            self.f.local_get(be);
            self.f.byte(I32_LT_U);
            self.f.local_get(ae);
            self.f.local_get(be);
            self.f.byte(I32_EQ);
            self.f.local_get(asig);
            self.f.local_get(bsig);
            self.f.byte(I64_LT_U);
            self.f.byte(I32_AND);
            self.f.byte(I32_OR);
            self.f.local_set(up); // swap, for now
            // t1 = sigX, t2 = sigY, e = eX, rb = eY, sign = sX.
            for (dst, ifswap, ifnot, wide) in [(t1, bsig, asig, true), (t2, asig, bsig, true)] {
                let _ = wide;
                self.f.local_get(ifswap);
                self.f.local_get(ifnot);
                self.f.local_get(up);
                self.f.select();
                self.f.local_set(dst);
            }
            for (dst, ifswap, ifnot) in [(e, be, ae), (rb, ae, be), (sign, bse, ase)] {
                self.f.local_get(ifswap);
                self.f.local_get(ifnot);
                self.f.local_get(up);
                self.f.select();
                self.f.local_set(dst);
            }
            // st = effective subtraction (signs differ).
            self.f.local_get(ase);
            self.f.local_get(bse);
            self.f.byte(I32_XOR);
            self.f.i32_const(15);
            self.f.byte(I32_SHR_U);
            self.f.local_set(st);
            self.f.local_get(sign);
            self.f.i32_const(15);
            self.f.byte(I32_SHR_U);
            self.f.local_set(sign);
            // d = eX - eY, in dd; require d <= 63, and d >= 2 to subtract.
            self.f.local_get(e);
            self.f.local_get(rb);
            self.f.byte(I32_SUB);
            self.f.byte(I64_EXTEND_I32_U);
            self.f.local_set(dd);
            self.f.local_get(dd);
            self.f.i64_const(63);
            self.f.byte(I64_GT_U);
            self.f.local_get(st);
            self.f.local_get(dd);
            self.f.i64_const(2);
            self.f.byte(I64_LT_U);
            self.f.byte(I32_AND);
            self.f.byte(I32_OR);
            self.f.br_if(1); // $slow
            // yhi = sigY >> d; ylo = d ? sigY << (64 - d) : 0.
            self.f.local_get(t2);
            self.f.local_get(dd);
            self.f.byte(I64_SHR_U);
            self.f.local_set(t3);
            self.f.local_get(t2);
            self.f.i64_const(64);
            self.f.local_get(dd);
            self.f.byte(I64_SUB_);
            self.f.byte(I64_SHL);
            self.f.i64_const(0);
            self.f.local_get(dd);
            self.f.byte(I64_EQZ);
            self.f.byte(I32_EQZ);
            self.f.select();
            self.f.local_set(t4);
            self.f.local_get(st);
            self.f.if_void();
            {
                // Subtract: lo = -ylo, hi = sigX - yhi - (ylo != 0); the top
                // bit is at 63 or 62.
                self.f.i64_const(0);
                self.f.local_get(t4);
                self.f.byte(I64_SUB_);
                self.f.local_set(lo);
                self.f.local_get(t1);
                self.f.local_get(t3);
                self.f.byte(I64_SUB_);
                self.f.local_get(t4);
                self.f.i64_const(0);
                self.f.byte(I64_NE);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.byte(I64_SUB_);
                self.f.local_set(hi);
                // n = top clear.
                self.f.local_get(hi);
                self.f.i64_const(63);
                self.f.byte(I64_SHR_U);
                self.f.i64_const(1);
                self.f.byte(code::I64_XOR);
                self.f.local_set(t2);
                self.f.local_get(e);
                self.f.local_get(t2);
                self.f.byte(I32_WRAP_I64);
                self.f.byte(I32_SUB);
                self.f.local_set(e);
                self.f.local_get(hi);
                self.f.local_get(t2);
                self.f.byte(I64_SHL);
                self.f.local_get(lo);
                self.f.i64_const(63);
                self.f.byte(I64_SHR_U);
                self.f.local_get(t2);
                self.f.byte(I64_AND);
                self.f.byte(I64_OR);
                self.f.local_set(hi);
                self.f.local_get(lo);
                self.f.local_get(t2);
                self.f.byte(I64_SHL);
                self.f.local_set(lo);
            }
            self.f.else_();
            {
                // Add: hi = sigX + yhi with carry c; lo = ylo. On a carry,
                // shift right one, keeping the dropped bit sticky.
                self.f.local_get(t1);
                self.f.local_get(t3);
                self.f.byte(I64_ADD);
                self.f.local_set(hi);
                self.f.local_get(hi);
                self.f.local_get(t1);
                self.f.byte(I64_LT_U);
                self.f.byte(I64_EXTEND_I32_U);
                self.f.local_set(t2); // c
                self.f.local_get(t4);
                self.f.local_get(t2);
                self.f.byte(I64_SHR_U);
                self.f.local_get(hi);
                self.f.i64_const(63);
                self.f.byte(I64_SHL);
                self.f.local_get(t2);
                self.f.byte(I64_MUL); // only when c
                self.f.byte(I64_OR);
                self.f.local_get(t4);
                self.f.local_get(t2);
                self.f.byte(I64_AND);
                self.f.byte(I64_OR); // sticky
                self.f.local_set(lo);
                self.f.local_get(hi);
                self.f.local_get(t2);
                self.f.byte(I64_SHR_U);
                self.f.local_get(t2);
                self.f.i64_const(63);
                self.f.byte(I64_SHL);
                self.f.byte(I64_OR);
                self.f.local_set(hi);
                self.f.local_get(e);
                self.f.local_get(t2);
                self.f.byte(I32_WRAP_I64);
                self.f.byte(I32_ADD);
                self.f.local_set(e);
            }
            self.f.end();
        }
        self.f.end(); // $round
        // Normal before rounding and finite after: 1 <= e <= 0x7ffd.
        self.f.local_get(e);
        self.f.i32_const(1);
        self.f.byte(I32_SUB);
        self.f.i32_const(0x7ffc);
        self.f.byte(0x4b); // i32.gt_u
        self.f.br_if(0);
        // dd = 64 - p: 40, 11 or 0 by the precision control.
        self.f.i64_const(11);
        self.f.i64_const(0);
        self.f.local_get(cw);
        self.f.i32_const(0x300);
        self.f.byte(I32_AND);
        self.f.i32_const(0x200);
        self.f.byte(I32_EQ);
        self.f.select(); // PC 10 -> 11, 01 and 11 -> 0
        self.f.i64_const(40);
        self.f.local_get(cw);
        self.f.i32_const(0x300);
        self.f.byte(I32_AND);
        self.f.select(); // PC 00 -> 40
        self.f.local_set(dd);
        // kept, round bit, sticky.
        self.f.local_get(hi);
        self.f.local_get(dd);
        self.f.byte(I64_SHR_U);
        self.f.local_set(kept);
        // d > 0: rb = (hi >> (d - 1)) & 1; st = hi & ((1 << (d - 1)) - 1) | lo.
        // d = 0: rb = lo >> 63; st = lo << 1.
        self.f.local_get(hi);
        self.f.local_get(dd);
        self.f.i64_const(1);
        self.f.byte(I64_SUB_);
        self.f.byte(I64_SHR_U);
        self.f.local_get(lo);
        self.f.i64_const(63);
        self.f.byte(I64_SHR_U);
        self.f.local_get(dd);
        self.f.byte(I64_EQZ);
        self.f.byte(I32_EQZ);
        self.f.select();
        self.f.i64_const(1);
        self.f.byte(I64_AND);
        self.f.byte(I32_WRAP_I64);
        self.f.local_set(rb);
        self.f.local_get(hi);
        self.f.i64_const(1);
        self.f.local_get(dd);
        self.f.i64_const(1);
        self.f.byte(I64_SUB_);
        self.f.byte(I64_SHL);
        self.f.i64_const(1);
        self.f.byte(I64_SUB_);
        self.f.byte(I64_AND);
        self.f.local_get(lo);
        self.f.byte(I64_OR);
        self.f.local_get(lo);
        self.f.i64_const(1);
        self.f.byte(I64_SHL);
        self.f.local_get(dd);
        self.f.byte(I64_EQZ);
        self.f.byte(I32_EQZ);
        self.f.select();
        self.f.i64_const(0);
        self.f.byte(I64_NE);
        self.f.local_set(st);
        // up = near & rb & (st | odd)  |  away & (rb | st).
        self.f.local_get(rb);
        self.f.local_get(st);
        self.f.local_get(kept);
        self.f.byte(I32_WRAP_I64);
        self.f.i32_const(1);
        self.f.byte(I32_AND);
        self.f.byte(I32_OR);
        self.f.byte(I32_AND);
        self.f.local_get(cw);
        self.f.i32_const(0xc00);
        self.f.byte(I32_AND);
        self.f.byte(I32_EQZ);
        self.f.byte(I32_AND);
        self.f.local_get(rb);
        self.f.local_get(st);
        self.f.byte(I32_OR);
        self.f.local_get(cw);
        self.f.i32_const(10);
        self.f.byte(I32_SHR_U);
        self.f.i32_const(3);
        self.f.byte(I32_AND);
        self.f.i32_const(2);
        self.f.local_get(sign);
        self.f.byte(I32_SUB);
        self.f.byte(I32_EQ);
        self.f.byte(I32_AND);
        self.f.byte(I32_OR);
        self.f.local_set(up);
        // A carry out of the kept bits: kept was all ones.
        self.f.local_get(up);
        self.f.local_get(kept);
        self.f.i64_const(-1);
        self.f.local_get(dd);
        self.f.byte(I64_SHR_U);
        self.f.byte(I64_EQ);
        self.f.byte(I32_AND);
        self.f.local_set(ae); // carry
        // res = carry ? 1 << 63 : (kept + up) << d
        self.f.i64_const(i64::MIN);
        self.f.local_get(kept);
        self.f.local_get(up);
        self.f.byte(I64_EXTEND_I32_U);
        self.f.byte(I64_ADD);
        self.f.local_get(dd);
        self.f.byte(I64_SHL);
        self.f.local_get(ae);
        self.f.select();
        self.f.local_set(res);
        // se = (e + carry) | sign << 15
        self.f.local_get(e);
        self.f.local_get(ae);
        self.f.byte(I32_ADD);
        self.f.local_get(sign);
        self.f.i32_const(15);
        self.f.byte(I32_SHL);
        self.f.byte(I32_OR);
        self.f.local_set(se);
        // sw = PE if inexact | C1 if up.
        self.f.local_get(rb);
        self.f.local_get(st);
        self.f.byte(I32_OR);
        self.f.i32_const(5);
        self.f.byte(I32_SHL);
        self.f.local_get(up);
        self.f.i32_const(9);
        self.f.byte(I32_SHL);
        self.f.byte(I32_OR);
        self.f.local_set(sw);
        self.f.br(1); // $out
        self.f.end(); // $slow
        self.f.local_get(self.ctx_local);
        self.f.i32_const(op.code());
        for a in &args[..5] {
            self.push(*a);
        }
        self.f.call(self.cx.x87);
        self.f.local_set(res);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 8) as u32);
        self.f.local_set(se);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 12) as u32);
        self.f.local_set(sw);
        self.f.end(); // $out
        let _ = (I64_GE_U, I32_GE_U);
        for (o, (l, ty)) in outs.iter().zip([(res, Ty::I64), (se, Ty::I32), (sw, Ty::I32)]) {
            let idx = o.0 as usize;
            self.value_ty[idx] = ty;
            self.value_local[idx] = Some(l);
        }
        self.free_i64.extend([asig, bsig, hi, lo, t1, t2, t3, t4, dd, kept]);
        self.free_i32.extend([ase, bse, cw, ae, be, sign, e, up, rb, st]);
    }

    /// `Compare`: inline when both operands are zeros or normal numbers
    /// (no NaN, infinity, denormal or unsupported encoding, so no flag can
    /// be raised); otherwise the import.
    fn emit_x87_compare(&mut self, op: X87Op, args: &[Value], outs: &[Value]) {
        use code::*;
        let asig = self.alloc(Ty::I64);
        let bsig = self.alloc(Ty::I64);
        let ase = self.alloc(Ty::I32);
        let bse = self.alloc(Ty::I32);
        let ae = self.alloc(Ty::I32);
        let be = self.alloc(Ty::I32);
        let an = self.alloc(Ty::I32);
        let bn = self.alloc(Ty::I32);
        let cc = self.alloc(Ty::I32);
        let sw = self.alloc(Ty::I32);
        let lt = self.alloc(Ty::I32);
        let eq = self.alloc(Ty::I32);
        for (v, l) in [(args[0], asig), (args[1], ase), (args[2], bsig), (args[3], bse)] {
            self.push(v);
            self.f.local_set(l);
        }
        for (se, e) in [(ase, ae), (bse, be)] {
            self.f.local_get(se);
            self.f.i32_const(0x7fff);
            self.f.byte(I32_AND);
            self.f.local_set(e);
        }
        // Eligible: (sig == 0 && e == 0) || (J set && 1 <= e <= 0x7ffe),
        // for both operands.
        for (sig, e) in [(asig, ae), (bsig, be)] {
            self.f.local_get(sig);
            self.f.byte(I64_EQZ);
            self.f.local_get(e);
            self.f.byte(I32_EQZ);
            self.f.byte(I32_AND);
            self.f.local_get(sig);
            self.f.i64_const(0);
            self.f.byte(I64_LT_S);
            self.f.local_get(e);
            self.f.i32_const(1);
            self.f.byte(I32_SUB);
            self.f.i32_const(0x7ffd);
            self.f.byte(I32_LE_U);
            self.f.byte(I32_AND);
            self.f.byte(I32_OR);
        }
        self.f.byte(I32_AND);
        self.f.if_void();
        // Negative and nonzero, for each.
        for (sig, se, n) in [(asig, ase, an), (bsig, bse, bn)] {
            self.f.local_get(se);
            self.f.i32_const(15);
            self.f.byte(I32_SHR_U);
            self.f.local_get(sig);
            self.f.i64_const(0);
            self.f.byte(I64_NE);
            self.f.byte(I32_AND);
            self.f.local_set(n);
        }
        // Magnitudes: eq, and lt by (exponent, significand).
        self.f.local_get(ae);
        self.f.local_get(be);
        self.f.byte(I32_EQ);
        self.f.local_get(asig);
        self.f.local_get(bsig);
        self.f.byte(I64_EQ);
        self.f.byte(I32_AND);
        self.f.local_set(eq);
        self.f.local_get(ae);
        self.f.local_get(be);
        self.f.byte(0x49); // i32.lt_u
        self.f.local_get(ae);
        self.f.local_get(be);
        self.f.byte(I32_EQ);
        self.f.local_get(asig);
        self.f.local_get(bsig);
        self.f.byte(I64_LT_U);
        self.f.byte(I32_AND);
        self.f.byte(I32_OR);
        self.f.local_set(lt);
        // Same sign: lt as is for positives, reversed (and not equal) for
        // negatives. Different signs: a < b exactly when a is negative.
        self.f.local_get(lt);
        self.f.byte(I32_EQZ);
        self.f.local_get(eq);
        self.f.byte(I32_EQZ);
        self.f.byte(I32_AND);
        self.f.local_get(lt);
        self.f.local_get(an);
        self.f.select();
        self.f.local_get(an);
        self.f.local_get(an);
        self.f.local_get(bn);
        self.f.byte(I32_EQ);
        self.f.select();
        self.f.local_set(lt);
        self.f.local_get(eq);
        self.f.local_get(an);
        self.f.local_get(bn);
        self.f.byte(I32_EQ);
        self.f.byte(I32_AND);
        self.f.local_set(eq);
        // cc = lt ? C0 : eq ? C3 : 0.
        self.f.i32_const(1 << 8);
        self.f.i32_const(1 << 14);
        self.f.i32_const(0);
        self.f.local_get(eq);
        self.f.select();
        self.f.local_get(lt);
        self.f.select();
        self.f.local_set(cc);
        self.f.i32_const(0);
        self.f.local_set(sw);
        self.f.else_();
        self.f.local_get(self.ctx_local);
        self.f.i32_const(op.code());
        for l in [asig, ase, bsig, bse] {
            self.f.local_get(l);
        }
        self.f.i32_const(0x37f);
        self.f.call(self.cx.x87);
        self.f.byte(I32_WRAP_I64);
        self.f.local_set(cc);
        self.ctx_base();
        self.f.load(0x28, 2, (ctx::X87_TMP + 12) as u32);
        self.f.local_set(sw);
        self.f.end();
        for (o, l) in outs.iter().zip([cc, sw]) {
            let idx = o.0 as usize;
            self.value_ty[idx] = Ty::I32;
            self.value_local[idx] = Some(l);
        }
        self.free_i64.extend([asig, bsig]);
        self.free_i32.extend([ase, bse, ae, be, an, bn, lt, eq]);
    }

    /// A fused multiply-add, without leaving the module.
    ///
    /// wasm has no FMA, so this would otherwise call the shim's. The arithmetic
    /// and its guard are in [`crate::fma`], with the fuzzing that establishes
    /// them; this emits the same steps inline (the sites are few even when the
    /// executions are not). Returns false when the caller should emit the call.
    fn emit_fma(&mut self, args: &[Value]) -> bool {
        if args.len() != 4 {
            return false;
        }
        // The precision is a constant the lifter just made, so the shape is
        // decided here rather than at run time.
        let double = match self.value_const.get(args[3].0 as usize).copied().flatten() {
            Some(v) => v != 0,
            None => return false,
        };
        if double {
            self.emit_fma64(args);
        } else {
            self.emit_fmaf(args);
        }
        true
    }

    /// Push an f64 constant.
    fn f64_const(&mut self, v: f64) {
        self.f.byte(0x44);
        for b in v.to_bits().to_le_bytes() {
            self.f.byte(b);
        }
    }

    /// Single precision: the product of two f32 is exact in f64, so only the
    /// sum can round, and round-to-odd stops the narrowing from breaking a
    /// tie the wrong way. No guard is needed.
    fn emit_fmaf(&mut self, args: &[Value]) {
        let (pl, sl, tl) = (
            self.f.add_local(ValType::F64),
            self.f.add_local(ValType::F64),
            self.f.add_local(ValType::F64),
        );
        // Each operand arrives as f32 bits in the low half of an i64.
        let widen = |me: &mut Self, v: Value| {
            me.push(v);
            me.f.byte(0xa7); // i32.wrap_i64
            me.f.byte(0xbe); // f32.reinterpret_i32
            me.f.byte(0xbb); // f64.promote_f32
        };
        widen(self, args[0]);
        widen(self, args[1]);
        self.f.byte(0xa2); // f64.mul -- exact
        self.f.local_set(pl);
        self.f.local_get(pl);
        widen(self, args[2]);
        self.f.byte(0xa0); // f64.add
        self.f.local_set(sl);

        // t = (p - (s - bb)) + (c - bb), where bb = s - p.
        self.f.local_get(sl);
        self.f.local_get(pl);
        self.f.byte(0xa1); // bb = s - p
        let bb = self.f.add_local(ValType::F64);
        self.f.local_set(bb);

        self.f.local_get(pl);
        self.f.local_get(sl);
        self.f.local_get(bb);
        self.f.byte(0xa1); // s - bb
        self.f.byte(0xa1); // p - (s - bb)
        widen(self, args[2]);
        self.f.local_get(bb);
        self.f.byte(0xa1); // c - bb
        self.f.byte(0xa0);
        self.f.local_set(tl);

        // if t != 0 and s is even, step s one ulp toward t.
        self.f.local_get(tl);
        self.f64_const(0.0);
        self.f.byte(0x61); // f64.eq
        self.f.byte(0x45); // i32.eqz -> t != 0
        self.f.local_get(sl);
        self.f.byte(0xbd); // i64.reinterpret_f64
        self.f.i64_const(1);
        self.f.byte(0x83); // i64.and
        self.f.byte(0x50); // i64.eqz -> low bit clear
        self.f.byte(0x71); // i32.and
        self.f.if_result(ValType::F64);
        {
            // Toward the error's sign, not simply upward: raising the
            // magnitude is backwards when the error is negative.
            self.f.local_get(sl);
            self.f.byte(0xbd);
            self.f.local_get(tl);
            self.f64_const(0.0);
            self.f.byte(0x64); // f64.gt -- t > 0
            self.f.local_get(sl);
            self.f64_const(0.0);
            self.f.byte(0x64); // s > 0
            self.f.byte(0x46); // i32.eq -- same sign
            self.f.if_result(ValType::I64);
            self.f.i64_const(1);
            self.f.else_();
            self.f.i64_const(-1);
            self.f.end();
            self.f.byte(0x7c); // i64.add
            self.f.byte(0xbf); // f64.reinterpret_i64
        }
        self.f.else_();
        self.f.local_get(sl);
        self.f.end();

        self.f.byte(0xb6); // f32.demote_f64
        self.f.byte(0xbc); // i32.reinterpret_f32
        self.f.byte(0xad); // i64.extend_i32_u -- the import's result shape
    }

    /// Double precision: Dekker's method behind an exponent guard, with the
    /// call it replaces in the else arm.
    fn emit_fma64(&mut self, args: &[Value]) {
        let l = |me: &mut Self| me.f.add_local(ValType::F64);
        let (al, bl, cl) = (l(self), l(self), l(self));
        self.push(args[0]);
        self.f.byte(0xbf);
        self.f.local_set(al);
        self.push(args[1]);
        self.f.byte(0xbf);
        self.f.local_set(bl);
        self.push(args[2]);
        self.f.byte(0xbf);
        self.f.local_set(cl);

        // The guard: every operand either zero, or with a biased exponent in
        // the band the fuzzing showed to be exact. The band cannot simply be
        // "finite" -- a*b doubles the exponent, so operands that are
        // individually fine overflow their product.
        for (i, v) in [al, bl, cl].iter().enumerate() {
            self.emit_in_fast_range(*v);
            if i > 0 {
                self.f.byte(0x71); // i32.and
            }
        }
        self.f.if_result(ValType::I64);
        self.emit_dekker(al, bl, cl);
        self.f.byte(0xbd); // i64.reinterpret_f64
        self.f.else_();
        // Everything the guard rejects goes where it used to.
        self.f.local_get(self.ctx_local);
        for a in args {
            self.push(*a);
        }
        self.f.call(self.cx.fma);
        self.f.end();
    }

    /// Leave 1 on the stack when the f64 in `v` is one the fast path handles.
    fn emit_in_fast_range(&mut self, v: u32) {
        let e = self.f.add_local(ValType::I64);
        self.f.local_get(v);
        self.f.byte(0xbd); // i64.reinterpret_f64
        self.f.i64_const(52);
        self.f.byte(0x87); // i64.shr_u
        self.f.i64_const(0x7ff);
        self.f.byte(0x83); // i64.and
        self.f.local_tee(e);
        self.f.i64_const(crate::fma::SAFE_LO as i64);
        self.f.byte(0x56); // i64.ge_u
        self.f.local_get(e);
        self.f.i64_const(crate::fma::SAFE_HI as i64);
        self.f.byte(0x58); // i64.le_u
        self.f.byte(0x71); // i32.and
        // Zero is admitted too, and is carried through exactly.
        self.f.local_get(v);
        self.f64_const(0.0);
        self.f.byte(0x61); // f64.eq
        self.f.byte(0x72); // i32.or
    }

    /// `p + e == a*b` exactly, `s + t == p + c` exactly, then one rounding.
    fn emit_dekker(&mut self, al: u32, bl: u32, cl: u32) {
        let lf = |me: &mut Self| me.f.add_local(ValType::F64);
        let (p, e, s, t, tmp) = (lf(self), lf(self), lf(self), lf(self), lf(self));
        let (ah, al2, bh, bl2) = (lf(self), lf(self), lf(self), lf(self));

        // p = a * b
        self.f.local_get(al);
        self.f.local_get(bl);
        self.f.byte(0xa2);
        self.f.local_set(p);

        // Split a and b into halves that multiply without error.
        for (x, hi, lo) in [(al, ah, al2), (bl, bh, bl2)] {
            self.f.local_get(x);
            self.f64_const(crate::fma::SPLIT);
            self.f.byte(0xa2);
            self.f.local_tee(tmp);
            self.f.local_get(tmp);
            self.f.local_get(x);
            self.f.byte(0xa1); // tmp - x
            self.f.byte(0xa1); // hi = tmp - (tmp - x)
            self.f.local_set(hi);
            self.f.local_get(x);
            self.f.local_get(hi);
            self.f.byte(0xa1); // lo = x - hi
            self.f.local_set(lo);
        }

        // e = ((ah*bh - p) + ah*bl + al*bh) + al*bl
        self.f.local_get(ah);
        self.f.local_get(bh);
        self.f.byte(0xa2);
        self.f.local_get(p);
        self.f.byte(0xa1);
        self.f.local_get(ah);
        self.f.local_get(bl2);
        self.f.byte(0xa2);
        self.f.byte(0xa0);
        self.f.local_get(al2);
        self.f.local_get(bh);
        self.f.byte(0xa2);
        self.f.byte(0xa0);
        self.f.local_get(al2);
        self.f.local_get(bl2);
        self.f.byte(0xa2);
        self.f.byte(0xa0);
        self.f.local_set(e);

        // s = p + c, and t its exact error.
        self.f.local_get(p);
        self.f.local_get(cl);
        self.f.byte(0xa0);
        self.f.local_set(s);
        self.f.local_get(s);
        self.f.local_get(p);
        self.f.byte(0xa1);
        self.f.local_set(tmp); // bb
        self.f.local_get(p);
        self.f.local_get(s);
        self.f.local_get(tmp);
        self.f.byte(0xa1);
        self.f.byte(0xa1);
        self.f.local_get(cl);
        self.f.local_get(tmp);
        self.f.byte(0xa1);
        self.f.byte(0xa0);
        self.f.local_set(t);

        // The one rounding, `s + (t + e)` -- or `s` itself when the
        // correction is zero, which keeps the sign of an exact zero (see
        // `fma::fma_soft`).
        self.f.local_get(t);
        self.f.local_get(e);
        self.f.byte(0xa0);
        self.f.local_set(tmp); // t + e
        self.f.local_get(s);
        self.f.local_get(s);
        self.f.local_get(tmp);
        self.f.byte(0xa0);
        self.f.local_get(tmp);
        self.f64_const(0.0);
        self.f.byte(0x61); // f64.eq
        self.f.select();
    }

    fn host_call_inline(&mut self, func: HostFunc, args: &[Value]) {
        if func == HostFunc::Fma && self.emit_fma(args) {
            return;
        }
        if func.clobbers_guest_state() {
            self.flush();
        }
        self.f.local_get(self.ctx_local);
        for a in args {
            self.push(*a);
        }
        self.f.call(self.import_index(func));
        if func.clobbers_guest_state() {
            self.invalidate();
        }
    }

    fn host_call(&mut self, func: HostFunc, args: &[Value], _dst: Option<Value>, _b: &Block) {
        if func.clobbers_guest_state() {
            self.flush();
        }
        self.f.local_get(self.ctx_local);
        for a in args {
            self.push(*a);
        }
        self.f.call(self.import_index(func));
        if func.clobbers_guest_state() {
            self.invalidate();
        }
    }

    fn import_index(&self, func: HostFunc) -> u32 {
        match func {
            HostFunc::Syscall => self.cx.syscall,
            HostFunc::Illegal => self.cx.illegal,
            HostFunc::Breakpoint => self.cx.breakpoint,
            HostFunc::SysRegRead => self.cx.sysreg_read,
            HostFunc::SysRegWrite => self.cx.sysreg_write,
            HostFunc::CacheOp => self.cx.cache_op,
            HostFunc::Safepoint => self.cx.safepoint,
            HostFunc::Fma => self.cx.fma,
        }
    }

    // -- terminators -------------------------------------------------------

    // -- profiling (crate::profile; all no-ops unless it is on) --------------

    /// The profile record of `block`, when this module is profiled.
    fn prof_rec(&self, block: &Block) -> Option<(&'a crate::profile::Emit, u32)> {
        let e = self.cx.profile?;
        e.records.get(&block.start).map(|&r| (e, r))
    }

    /// At the top of a block's body: count the entry, and its kind.
    fn prof_entry(&mut self, block: &Block) {
        if let Some((e, rec)) = self.prof_rec(block) {
            crate::profile::emit_block_entry(&mut self.f, e, block.start, rec);
        }
    }

    /// A direct call (`bl`, `call rel`): push the shadow frame and announce
    /// the callee. Indirect calls do the same in [`Self::prof_indirect`].
    fn prof_direct_call(&mut self, block: &Block) {
        let (Some((e, _)), Some(ret), Term::Goto { target }) =
            (self.prof_rec(block), block.call_return, &block.term)
        else {
            return;
        };
        self.f.i64_const(ret as i64);
        self.f.i64_const(*target as i64);
        self.f.call(e.f_call);
    }

    /// Inside a conditional branch's taken arm.
    fn prof_taken(&mut self, block: &Block) {
        if let Some((e, rec)) = self.prof_rec(block) {
            crate::profile::emit_count(&mut self.f, e.base_global, rec + crate::profile::REC_TAKEN);
        }
    }

    /// An indirect branch, its target in `scratch_i64`: a return is checked
    /// against the shadow stack, anything else counts its target.
    fn prof_indirect(&mut self, block: &Block) {
        let Some((e, rec)) = self.prof_rec(block) else {
            return;
        };
        let Term::Indirect { is_ret, .. } = block.term else {
            return;
        };
        self.f.i32_const(rec as i32);
        self.f.local_get(self.scratch_i64);
        if is_ret {
            self.f.call(e.f_ret);
            return;
        }
        let kind = if block.call_return.is_some() {
            crate::profile::KIND_CALL
        } else {
            crate::profile::KIND_JUMP
        };
        self.f.i32_const(kind);
        self.f.call(e.f_indirect);
        if let Some(ret) = block.call_return {
            self.f.i64_const(ret as i64);
            self.f.local_get(self.scratch_i64);
            self.f.call(e.f_call);
        }
    }

    fn terminator(&mut self, block: &Block) {
        self.prof_direct_call(block);
        match block.term {
            Term::Goto { target } => {
                if self.loop_depth.is_some() {
                    self.emit_loop_safepoint(target);
                    let d = self.loop_depth.unwrap();
                    self.f.br(d);
                    return;
                }
                self.flush();
                if target <= block.start {
                    self.emit_safepoint(target);
                }
                self.jump_to_constant(target);
            }
            Term::Branch {
                cond,
                taken,
                not_taken,
            } => {
                if let Some(depth) = self.loop_depth {
                    // One arm is the back-edge, which stays inside the loop
                    // and keeps every register in its local. Only the exit arm
                    // pays for a flush.
                    self.push(cond);
                    self.f.if_void();
                    self.prof_taken(block);
                    if taken == block.start {
                        self.emit_loop_safepoint(taken);
                        self.f.br(depth + 1);
                    } else {
                        self.flush();
                        if taken <= block.start {
                            self.emit_safepoint(taken);
                        }
                        self.jump_to_constant(taken);
                    }
                    self.f.end();
                    if not_taken == block.start {
                        self.emit_loop_safepoint(not_taken);
                        self.f.br(depth);
                    } else {
                        self.flush();
                        if not_taken <= block.start {
                            self.emit_safepoint(not_taken);
                        }
                        self.jump_to_constant(not_taken);
                    }
                    return;
                }
                // Flush before the branch so both arms see the stores; doing
                // it inside the `if` would only cover the taken path.
                self.flush();
                // A conditional back-edge polls only on the taken path, so a
                // loop's exit test costs nothing extra.
                self.push(cond);
                self.f.if_void();
                self.prof_taken(block);
                if taken <= block.start {
                    self.emit_safepoint(taken);
                }
                self.jump_to_constant(taken);
                self.f.end();
                if not_taken <= block.start {
                    self.emit_safepoint(not_taken);
                }
                self.jump_to_constant(not_taken);
            }
            Term::Indirect { target, .. } => {
                self.push(target);
                self.f.local_set(self.scratch_i64);
                self.prof_indirect(block);
                // The context PC must be exact before a possible dispatch
                // miss, since that is the only place the target is recorded.
                self.ctx_base();
                self.f.local_get(self.scratch_i64);
                self.f.store(0x37, 3, ctx::PC as u32);
                self.flush();
                // Poll on indirect branches too, not only backward direct ones:
                // a computed back edge (a bytecode interpreter's `br <reg>`
                // dispatch, anything jump-threaded) is exactly the loop a
                // guest can spin in forever, reaching no poll and so unable to
                // be preempted or take a signal. Safe here because the PC is
                // already the target (stored above) and the registers are
                // flushed, so the slow path may deliver a signal and unwind.
                // The pc handed to the poll is what the host writes back as
                // the context PC, and what a dispatch miss right after this
                // resumes from. On A64 the branch's own address serves: the
                // instruction is re-executed and re-reads its register. On
                // x86-64 `end - 1` is inside the branch instruction, so the
                // poll gets the target itself -- the value the PC already
                // holds, and the only address that is safe to resume at.
                if gex_abi::X86 {
                    self.emit_safepoint_no_flush_dynamic();
                } else {
                    self.emit_safepoint_no_flush(block.end - gex_abi::INST_ALIGN);
                }
                self.emit_dispatch_tail_call();
            }
            Term::Syscall { next, imm } => {
                self.flush();
                // The recorded PC is the instruction *after* the `svc`: it is
                // where a signal frame must resume, and resuming at the `svc`
                // itself would re-issue the syscall. The call still receives
                // the syscall's own address, which is what a trace wants.
                self.store_pc(next);
                // An ISA that numbers its system calls in the instruction
                // (`svc #imm`) leaves the number here, so the
                // kernel does not read the number register the guest's ABI
                // keeps live across the call (`x8` is AArch64's
                // result-location register).
                self.store_syscall_imm(imm);
                self.f.local_get(self.ctx_local);
                self.f.i64_const((block.end - gex_abi::SYSCALL_INST_BYTES) as i64);
                self.f.call(self.cx.syscall);
                self.invalidate();
                self.jump_to_constant(next);
            }
            Term::Illegal { raw, pc } => {
                self.flush();
                self.store_pc(pc);
                self.f.local_get(self.ctx_local);
                self.f.i64_const((pc | gex_abi::ILLEGAL_RETURNS_PC) as i64);
                self.f.i32_const(raw as i32);
                self.f.call(self.cx.illegal);
                // `illegal` raises SIGILL and does not return, except where
                // the host ran the instruction as a call (the browser:
                // 64-bit native Wine's system call entry, a `ud2`;
                // `ILLEGAL_RETURNS_PC` says this code allows it): the
                // thread goes on at the context's PC, through a safepoint,
                // so a loop of nothing but system calls still reaches one.
                self.invalidate();
                self.ctx_base();
                self.f.load(0x29, 3, ctx::PC as u32);
                self.f.local_set(self.scratch_i64);
                if gex_abi::X86 {
                    self.emit_safepoint_no_flush_dynamic();
                }
                self.emit_dispatch_tail_call();
            }
        }
    }

    /// The safepoint countdown, for a caller that has already flushed and
    /// carries on rather than jumping.
    ///
    /// Unlike [`Self::emit_safepoint`] this never flushes -- an indirect branch
    /// has flushed already and its target is in a local -- so it emits only the
    /// counter and the call.
    fn emit_safepoint_no_flush(&mut self, pc: u64) {
        let before = self.f.code.len();
        self.emit_safepoint_no_flush_inner(pc);
        SAFEPOINT_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    /// The poll with the pc taken from `scratch_i64` (the branch target an
    /// indirect terminator has just stored) rather than a constant.
    fn emit_safepoint_no_flush_dynamic(&mut self) {
        let before = self.f.code.len();
        self.ctx_base();
        self.ctx_base();
        self.f.load(0x28, 2, ctx::SAFEPOINT_COUNTER as u32);
        self.f.i32_const(1);
        self.f.byte(0x6b); // i32.sub
        self.f.local_tee(self.scratch_i32);
        self.f.store(0x36, 2, ctx::SAFEPOINT_COUNTER as u32);
        self.f.local_get(self.scratch_i32);
        self.f.i32_const(0);
        self.f.byte(0x4c); // i32.le_s
        self.f.if_void();
        self.f.local_get(self.ctx_local);
        self.f.local_get(self.scratch_i64);
        self.f.call(self.cx.safepoint);
        self.f.end();
        SAFEPOINT_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    fn emit_safepoint_no_flush_inner(&mut self, pc: u64) {
        self.ctx_base();
        self.ctx_base();
        self.f.load(0x28, 2, ctx::SAFEPOINT_COUNTER as u32);
        self.f.i32_const(1);
        self.f.byte(0x6b); // i32.sub
        self.f.local_tee(self.scratch_i32);
        self.f.store(0x36, 2, ctx::SAFEPOINT_COUNTER as u32);
        self.f.local_get(self.scratch_i32);
        self.f.i32_const(0);
        self.f.byte(0x4c); // i32.le_s
        self.f.if_void();
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.call(self.cx.safepoint);
        self.f.end();
    }

    /// Poll for a pending signal before a backward jump.
    ///
    /// A backward direct branch is the cheap, sound approximation of a loop:
    /// no program can spin without reaching a poll, while straight-line code,
    /// forward branches and calls pay nothing (syscalls are delivery points
    /// already).
    fn emit_safepoint(&mut self, pc: u64) {
        let before = self.f.code.len();
        self.emit_safepoint_inner(pc);
        SAFEPOINT_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    fn emit_safepoint_inner(&mut self, pc: u64) {
        // Count down, and only call out when the counter runs out, so the
        // common case is a load, a subtract, a store and a not-taken branch.
        self.ctx_base();
        self.ctx_base();
        self.f.load(0x28, 2, ctx::SAFEPOINT_COUNTER as u32);
        self.f.i32_const(1);
        self.f.byte(0x6b); // i32.sub
        self.f.local_tee(self.scratch_i32);
        self.f.store(0x36, 2, ctx::SAFEPOINT_COUNTER as u32);

        self.f.local_get(self.scratch_i32);
        self.f.i32_const(0);
        self.f.byte(0x4c); // i32.le_s
        self.f.if_void();
        // Inside a fused loop the registers live only in locals, so the slow
        // path -- which may deliver a signal and unwind -- must write them back.
        if self.loop_depth.is_some() {
            self.flush_conditional();
        }
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.call(self.cx.safepoint);
        // Written out above but deliberately not read back: the host only
        // changes a register on a path that does not return here, and every
        // one resumes through the context. Reading them back made a loop with
        // a `continue` and a `break` never finish.
        self.f.end();
    }

    /// The safepoint on a fused loop's own back-edge.
    ///
    /// Two differences from [`Self::emit_safepoint`]. The countdown is a local,
    /// not the context counter, so the hot path is a subtract and a branch with
    /// no memory traffic. And on expiry the loop does not resume: it goes back
    /// through the dispatcher, which re-enters this block. That is for V8,
    /// whose baseline compiler promotes on a budget counting *calls* -- so a
    /// loop entered once and iterated forever would never be promoted.
    fn emit_loop_safepoint(&mut self, pc: u64) {
        let before = self.f.code.len();
        self.emit_loop_safepoint_inner(pc);
        SAFEPOINT_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    fn emit_loop_safepoint_inner(&mut self, pc: u64) {
        let Some(counter) = self.reenter_local else {
            // The inner form, so the wrapper above counts these bytes once.
            self.emit_safepoint_inner(pc);
            return;
        };
        self.f.local_get(counter);
        self.f.i32_const(1);
        self.f.byte(0x6b); // i32.sub
        self.f.local_tee(counter);
        self.f.i32_const(0);
        self.f.byte(0x4c); // i32.le_s
        self.f.if_void();
        self.flush_conditional();
        // The host rewrites the context's counter, so the next entry starts a
        // fresh interval without this having to say so.
        self.f.local_get(self.ctx_local);
        self.f.i64_const(pc as i64);
        self.f.call(self.cx.safepoint);
        self.jump_to_constant(pc);
        self.f.end();
    }

    /// Transfer control to a statically known guest address.
    fn jump_to_constant(&mut self, target: u64) {
        // TEST(GEX_NO_DIRECT_JUMP): force every branch through the dispatch
        // table, so a same-module target is looked up (and its PC stored) the
        // way a cross-module one is. Native only -- `env` always fails on wasm,
        // so the shim keeps direct calls; set it for the closure build to tell a
        // direct-jump fault from a dispatch one.
        static NO_DIRECT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_direct = *NO_DIRECT.get_or_init(|| std::env::var_os("GEX_NO_DIRECT_JUMP").is_some());
        if !no_direct {
            if let Some(&func) = self.cx.blocks.get(&target) {
                // The callee checks `LAST_TARGET` against its own address
                // (x86-64), so a direct jump has to record it as a dispatch would.
                if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
                    self.ctx_base();
                    self.f.i64_const(target as i64);
                    self.f.store(0x37, 3, ctx::LAST_TARGET as u32);
                }
                self.f.local_get(self.ctx_local);
                self.f.return_call_local(func);
                return;
            }
        }
        // Not in this module: record the PC and go through the table. The
        // page's directory entry decides whether that reaches real code or
        // the miss stub, and it can change as translation proceeds, so even a
        // constant target has to be looked up at run time.
        self.store_pc(target);
        self.f.i64_const(target as i64);
        self.f.local_set(self.scratch_i64);
        // The exact target, for a slot shared by several blocks (x86-64): the
        // trampoline there reads `LAST_TARGET` to pick one.
        if gex_abi::X86 {
            self.ctx_base();
            self.f.local_get(self.scratch_i64);
            self.f.store(0x37, 3, ctx::LAST_TARGET as u32);
        }
        self.emit_dispatch_tail_call();
    }

    /// Consume the target address in `scratch_i64` and leave a dispatch-table
    /// index on the stack.
    ///
    /// Two levels, and deliberately branchless. A page with no translation
    /// has a directory entry of zero, so the index lands in the reserved
    /// miss-stub span at the bottom of the table and the stub asks the host
    /// to translate. An address beyond the directory's reach selects zero the
    /// same way, so it needs no separate test either.
    fn emit_dispatch_index(&mut self) {
        self.emit_dispatch_entry();
        self.mask_slot_bits();
        self.add_page_offset();
    }

    /// Load the directory entry for the address in `scratch_i64`.
    ///
    /// Split out from the index arithmetic because with several tables the
    /// entry is wanted twice -- once for its table index and once for its
    /// base slot -- and loading it twice would mean two loads on every
    /// indirect branch.
    fn emit_dispatch_entry(&mut self) {
        // page = pc >> 12
        self.f.local_get(self.scratch_i64);
        self.f.i64_const(DISPATCH_PAGE.trailing_zeros() as i64);
        self.f.byte(0x88); // i64.shr_u
        self.f.local_set(self.scratch_page);

        // entry = page < MAX_PAGES ? load(DISPATCH_DIR + page*4) : 0
        self.f.local_get(self.scratch_page);
        self.f.byte(0xa7); // i32.wrap_i64
        self.f.i32_const(4);
        self.f.byte(0x6c); // i32.mul
        self.load_dir_entry();
        self.f.i32_const(0);
        self.f.local_get(self.scratch_page);
        self.f.i64_const(gex_abi::dispatch_max_pages() as i64);
        self.f.byte(code::I64_LT_U);
        self.f.select();
    }

    /// The directory entry at the byte offset (an `i32`, page * 4) on the
    /// stack: in the process's directory, or for a module of code in its
    /// other mode the other one.
    fn load_dir_entry(&mut self) {
        match self.cx.alt_dir {
            Some(g) => {
                self.f.addr_i32();
                self.f.global_get(g);
                self.f.addr_add();
                self.f.load64(0x28, 2, 0);
            }
            None => {
                self.f.addr_i32();
                self.f.load64(0x28, 2, gex_abi::dispatch_dir());
            }
        }
    }

    /// Drop the table index from an entry on the stack, leaving the base slot.
    ///
    /// Skipped entirely with one table, where the top bits are always zero
    /// and the mask would be an instruction that can never change anything.
    fn mask_slot_bits(&mut self) {
        if gex_abi::DISPATCH_TABLES > 1 {
            self.f.i32_const(gex_abi::DISPATCH_SLOT_MASK as i32);
            self.f.byte(0x71); // i32.and
        }
    }

    /// index += (pc & 0xfff) >> DISPATCH_SHIFT
    fn add_page_offset(&mut self) {
        self.f.local_get(self.scratch_i64);
        self.f.i64_const((DISPATCH_PAGE - 1) as i64);
        self.f.byte(0x83); // i64.and
        if gex_abi::DISPATCH_SHIFT != 0 {
            self.f.i64_const(gex_abi::DISPATCH_SHIFT as i64);
            self.f.byte(0x88); // i64.shr_u
        }
        self.f.byte(0xa7); // i32.wrap_i64
        self.f.byte(0x6a); // i32.add
    }

    /// Tail-call the block at the address in `scratch_i64`, through whichever
    /// dispatch table its page lives in.
    ///
    /// With one table: compute the index, then a single
    /// `return_call_indirect`. With several (see `DISPATCH_TABLES`), the entry
    /// also carries the table index in the bits above the slot, and the choice
    /// is a `br_table` over one tail call per table. Entry zero still means "no
    /// translation" -- table 0, a slot in the reserved miss span -- so an
    /// untranslated page, an out-of-range address and a real table-0 block all
    /// take the same path with no test between them. Index and table go into
    /// locals first: a value pushed outside a `block` cannot be consumed inside.
    /// A slot covers several guest addresses (x86-64: four bytes), so a block
    /// can be entered for a neighbour's address. Every dispatch records the
    /// exact target in `ctx::LAST_TARGET`; a block that is not it asks the
    /// host to translate the real target (the slot then gets a trampoline
    /// covering both) and dispatches again.
    fn emit_target_check(&mut self, start: u64) {
        self.ctx_base();
        self.f.load(0x29, 3, ctx::LAST_TARGET as u32);
        self.f.i64_const(start as i64);
        self.f.byte(0x52); // i64.ne
        self.f.if_void();
        // First make the host forget the neighbour: a translated address is
        // otherwise answered without translating, and the plain block here
        // would be entered for it again, for ever.
        self.f.local_get(self.ctx_local);
        self.f.i64_const(start as i64);
        self.f.i32_const(gex_abi::cache_op::SLOT_CONFLICT);
        self.ctx_base();
        self.f.load(0x29, 3, ctx::LAST_TARGET as u32);
        self.f.call(self.cx.cache_op);
        self.f.local_get(self.ctx_local);
        self.ctx_base();
        self.f.load(0x29, 3, ctx::LAST_TARGET as u32);
        self.f.call(self.cx.dispatch_miss);
        self.f.byte(0x1a); // drop the slot it answered with
        self.ctx_base();
        self.f.load(0x29, 3, ctx::LAST_TARGET as u32);
        self.f.local_set(self.scratch_i64);
        self.emit_dispatch_tail_call();
        self.f.end();
    }

    /// The self-modifying-code guard (`Block::guard`): compare the block's
    /// code bytes with what was lifted; on a difference tell the host to
    /// re-lift the page (`cache_op` IC_INVALIDATE, the same path `ic ivau`
    /// takes on AArch64) and dispatch to this address again, which now misses.
    fn emit_guard(&mut self, start: u64, bytes: &[u8]) {
        self.f.i64_const(0);
        self.f.local_set(self.scratch_i64);
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let at = start + (i * 8) as u64;
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let want = u64::from_le_bytes(buf);
            let mask: u64 = if chunk.len() == 8 { !0 } else { (1u64 << (8 * chunk.len())) - 1 };
            self.f.addr_const(at);
            self.f.load(0x29, 0, 0); // i64.load, unaligned
            if mask != !0 {
                self.f.i64_const(mask as i64);
                self.f.byte(0x83); // i64.and
            }
            self.f.i64_const(want as i64);
            self.f.byte(0x85); // i64.xor
            self.f.local_get(self.scratch_i64);
            self.f.byte(0x84); // i64.or
            self.f.local_set(self.scratch_i64);
        }
        self.f.local_get(self.scratch_i64);
        self.f.byte(0x50); // i64.eqz
        self.f.byte(0x45); // i32.eqz: changed?
        self.f.if_void();
        self.f.local_get(self.ctx_local);
        self.f.i64_const(start as i64);
        self.f.i32_const(gex_abi::cache_op::IC_INVALIDATE);
        self.f.i64_const(start as i64);
        self.f.call(self.cx.cache_op);
        // The dispatch below now misses, and a miss translates the address in
        // the context's PC: this one, not the last one stored (a direct jump
        // into rewritten code arrives without storing it, and the miss went
        // to wherever the last indirect branch had been).
        self.store_pc(start);
        self.f.i64_const(start as i64);
        self.f.local_set(self.scratch_i64);
        if gex_abi::X86 {
            self.ctx_base();
            self.f.local_get(self.scratch_i64);
            self.f.store(0x37, 3, ctx::LAST_TARGET as u32);
        }
        self.emit_dispatch_tail_call();
        self.f.end();
    }

    /// The block's pages must still have the directory entries it was
    /// translated for. Otherwise the code was retired (unmapped, rewritten:
    /// Detours patching a prologue) or translated again after it, and a
    /// direct jump from a block of the same module -- which never looks at
    /// the directory -- must not run it: store the PC and dispatch, which
    /// misses and translates the bytes as they are now. Fused loops and
    /// regions re-enter through the dispatcher every `loop_poll` trips, so a
    /// thread already inside a stale loop leaves too.
    fn emit_page_check(&mut self, start: u64, pages: &[(u64, u32)]) {
        if pages.is_empty() {
            return;
        }
        for (k, (key, base)) in pages.iter().enumerate() {
            let (page, _) = gex_abi::key_page(*key);
            match self.cx.alt_dir {
                Some(_) => {
                    self.f.i32_const((page * 4) as i32);
                    self.load_dir_entry();
                }
                None => {
                    self.f.addr_const(gex_abi::dispatch_dir() + page * 4);
                    self.f.load(0x28, 2, 0); // i32.load
                }
            }
            self.f.i32_const(*base as i32);
            self.f.byte(0x47); // i32.ne
            if k > 0 {
                self.f.byte(0x72); // i32.or
            }
        }
        self.f.if_void();
        // Dispatch alone could come straight back here: a block that runs
        // into a second page fails when only that page changed, while the
        // directory entry of the page it starts on -- the one dispatch reads
        // -- still names this block's slot. So the host is told first
        // (`PAGE_STALE`, with this module's id): if this module still provides
        // the address it translates it again, which writes the new block into
        // the slot; if a newer translation provides it already (this block was
        // reached by a direct jump from a sibling), it does nothing.
        if !page_check_no_retranslate() {
            self.f.local_get(self.ctx_local);
            self.f.i64_const(start as i64);
            self.f.i32_const(gex_abi::cache_op::PAGE_STALE);
            // The argument: this module's dense base (high half, read at run
            // time: where its blocks are in the process running it, unique
            // there by construction) and its id (low half). The id is
            // compiled in (natively, a global set at instantiation) and code
            // outlives its process (units adopted from
            // another, modules restored from the translation cache), so ids
            // repeat; it only decides whether to translate again, never
            // whether to resume from JavaScript (`gex_cache_op`).
            match self.cx.module_id_global {
                Some(g) => {
                    self.f.global_get(g);
                    self.f.byte(0xad); // i64.extend_i32_u
                }
                None => self.f.i64_const((self.cx.module_id & 0xffff_ffff) as i64),
            }
            if let Some(g) = self.cx.dense_global {
                self.f.global_get(g);
                self.f.byte(0xad); // i64.extend_i32_u
                self.f.i64_const(32);
                self.f.byte(0x86); // i64.shl
                self.f.byte(0x84); // i64.or
            }
            self.f.call(self.cx.cache_op);
        }
        self.store_pc(start);
        self.f.i64_const(start as i64);
        self.f.local_set(self.scratch_i64);
        if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
            self.ctx_base();
            self.f.local_get(self.scratch_i64);
            self.f.store(0x37, 3, ctx::LAST_TARGET as u32);
        }
        self.emit_dispatch_tail_call();
        self.f.end();
    }

    fn emit_dispatch_tail_call(&mut self) {
        let before = self.f.code.len();
        self.emit_ballast();
        self.emit_dispatch_tail_call_inner();
        DISPATCH_BYTES.fetch_add(self.f.code.len() - before, Ordering::Relaxed);
    }

    /// How many load-and-store pairs `GEX_DISPATCH_BALLAST` asks for.
    fn ballast() -> usize {
        use std::sync::OnceLock;
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("GEX_DISPATCH_BALLAST")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        })
    }

    /// Work on the indirect-dispatch path that does nothing, for measurement:
    /// does this program care about the dispatch path at all?
    ///
    /// Each pair is a load from the dispatch directory near the entry this path
    /// reads (same cache line, so the cheapest possible extra load and thus a
    /// lower bound) and a store into a context slot nothing reads. The store is
    /// what makes it survive the engine's dead-code elimination. The slope of
    /// run time against `GEX_DISPATCH_BALLAST` is the answer.
    fn emit_ballast(&mut self) {
        let n = Self::ballast();
        if n == 0 {
            return;
        }
        for i in 0..n {
            self.ctx_base();
            self.f.local_get(self.scratch_i64);
            self.f.i64_const(DISPATCH_PAGE.trailing_zeros() as i64);
            self.f.byte(0x88); // i64.shr_u
            self.f.byte(0xa7); // i32.wrap_i64
            self.f.i32_const(4);
            self.f.byte(0x6c); // i32.mul
            self.f.addr_i32();
            self.f.load64(0x28, 2, gex_abi::dispatch_dir() + 4 * (i as u64 + 1));
            self.f.store(0x36, 2, ctx::BALLAST as u32);
        }
    }

    fn emit_dispatch_tail_call_inner(&mut self) {
        // Outlined: the module's `dispatch(ctx, target)` stores `LAST_TARGET`
        // and does the same lookup and selection, so the context is left
        // exactly as the inline form leaves it, `ctx::PC` as the caller left
        // it. (Re-storing the PC as `enter` does is not the
        // same: after an indirect branch's safepoint the PC is the branch
        // itself, to be re-executed, and a provider's IFUNC address then
        // reached the miss path, lost its bridge in a loop and panicked
        // tier-1 in the native-debian page.)
        if let Some(enter) = self.cx.enter.filter(|_| !inline_dispatch() && gex_abi::DISPATCH_TABLES > 1) {
            if gex_abi::X86 {
                self.ctx_base();
                self.f.i64_const(self.block_start as i64);
                self.f.store(0x37, 3, ctx::LAST_SOURCE as u32);
            }
            self.f.local_get(self.ctx_local);
            self.f.local_get(self.scratch_i64);
            self.f.return_call_local(enter);
            return;
        }
        if gex_abi::DISPATCH_TABLES <= 1 {
            self.f.local_get(self.ctx_local);
            self.emit_dispatch_index();
            self.f.return_call_indirect(self.cx.block_ty, 0);
            return;
        }
        let n = gex_abi::DISPATCH_TABLES as u32;
        let entry = self.scratch_entry;
        let index = self.scratch_index;

        // Where this branch is going, before it goes. See `ctx::LAST_TARGET`:
        // the tail call below replaces this frame, so afterwards there is no
        // way left to ask.
        self.ctx_base();
        self.f.local_get(self.scratch_i64);
        self.f.store(0x37, 3, ctx::LAST_TARGET as u32);
        // And where it was made: the block start, a constant per block
        // (x86-64, whose wild-branch reports name it).
        if gex_abi::X86 {
            self.ctx_base();
            self.f.i64_const(self.block_start as i64);
            self.f.store(0x37, 3, ctx::LAST_SOURCE as u32);
        }

        self.emit_dispatch_entry();
        self.f.local_tee(entry);
        self.mask_slot_bits();
        self.add_page_offset();
        self.f.local_set(index);

        // tbl = entry >> 24, reusing the entry local now its base is spent.
        self.f.local_get(entry);
        self.f.i32_const(gex_abi::DISPATCH_SLOT_BITS as i32);
        self.f.byte(0x76); // i32.shr_u
        self.f.local_set(entry);

        // `n` blocks, not `n-1`: the region *inside* the innermost one is
        // unreachable, because `br_table` always branches. Each `end` opens a
        // reachable region and there must be one per table.
        for _ in 0..n {
            self.f.block_void();
        }
        self.f.local_get(entry);
        // Depth 0 is the innermost block, so its `end` is the first region
        // emitted and belongs to the last table. Table k breaks to depth
        // `n-1-k`; the default is table 0's depth, which is where an entry
        // naming a table that does not exist ends up -- the miss stub, for a
        // directory that has been scribbled on.
        let targets: Vec<u32> = (0..n).map(|k| n - 1 - k).collect();
        self.f.br_table(&targets, n - 1);
        for k in (0..n).rev() {
            self.f.end();
            if crate::dense::table(k as usize) {
                crate::dense::emit_arm(&mut self.f, k as usize, self.ctx_local, index, entry, self.cx.block_ty, self.cx.entry_ty);
                continue;
            }
            self.f.local_get(self.ctx_local);
            self.f.local_get(index);
            self.f.return_call_indirect(self.cx.block_ty, k);
        }
    }
}

/// For each statement, the values whose last use it is, in value order (the
/// order the releases always ran in). Was a scan of every value after every
/// statement: quadratic, and most of a Unity start-up's time in Mono's large
/// JIT-generated blocks.
fn dying_after(block: &Block) -> Dying {
    let last = compute_last_use(block);
    // Counted, then filled in place: one allocation for every statement's
    // list, where a `Vec` per statement was the emitter's largest single
    // source of allocator traffic.
    let mut start = vec![0u32; block.stmts.len() + 1];
    for &i in last.iter().flatten() {
        if let Some(c) = start.get_mut(i + 1) {
            *c += 1;
        }
    }
    for i in 1..start.len() {
        start[i] += start[i - 1];
    }
    let mut fill = start.clone();
    let mut vals = vec![0u32; *start.last().unwrap_or(&0) as usize];
    // Only the statements counted above (`i + 1` within `start`): a block
    // with values but no statements defaults them to statement 0, which it
    // does not have -- filling it anyway wrote past `vals` (a tier-1
    // panic_bounds_check here).
    for (v, lu) in last.into_iter().enumerate() {
        if let Some(i) = lu.filter(|&i| i + 1 < start.len()) {
            if let Some(at) = fill.get_mut(i) {
                vals[*at as usize] = v as u32;
                *at += 1;
            }
        }
    }
    Dying { start, vals }
}

/// The values whose last use is each statement (see [`dying_after`]), in
/// value order per statement.
struct Dying {
    start: Vec<u32>,
    vals: Vec<u32>,
}

impl std::ops::Index<usize> for Dying {
    type Output = [u32];
    fn index(&self, i: usize) -> &[u32] {
        &self.vals[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// Index of the last statement that reads each value, or `None` when the
/// value is read by the terminator (and so must survive the whole block).
fn compute_last_use(block: &Block) -> Vec<Option<usize>> {
    let n = block.n_values as usize;
    // A value with no reader at all is freed immediately after its own
    // definition, which is why the default is the defining statement.
    let mut last: Vec<Option<usize>> = vec![Some(0); n];
    for (i, stmt) in block.stmts.iter().enumerate() {
        if let Stmt::Def { dst, .. } = stmt {
            if let Some(slot) = last.get_mut(dst.0 as usize) {
                *slot = Some(i);
            }
        }
    }
    for (i, stmt) in block.stmts.iter().enumerate() {
        stmt.for_each_use(|v| {
            if let Some(slot) = last.get_mut(v.0 as usize) {
                *slot = Some(i);
            }
        });
    }
    // Values the terminator reads must survive to the end of the block.
    block.term_uses(|v| {
        if let Some(slot) = last.get_mut(v.0 as usize) {
            *slot = None;
        }
    });
    last
}

// ---------------------------------------------------------------------------
// Opcode tables
// ---------------------------------------------------------------------------

fn bin_op(op: BinOp, ty: Ty) -> u8 {
    let base32: u8 = match op {
        BinOp::Add => 0x6a,
        BinOp::Sub => 0x6b,
        BinOp::Mul => 0x6c,
        BinOp::DivS => 0x6d,
        BinOp::DivU => 0x6e,
        BinOp::RemS => 0x6f,
        BinOp::RemU => 0x70,
        BinOp::And => 0x71,
        BinOp::Or => 0x72,
        BinOp::Xor => 0x73,
        BinOp::Shl => 0x74,
        BinOp::ShrS => 0x75,
        BinOp::ShrU => 0x76,
        BinOp::Rotr => 0x78,
    };
    match ty {
        Ty::I32 => base32,
        // The i64 arithmetic block starts 18 opcodes after the i32 one and
        // holds the same operations in the same order. Vectors never reach
        // here: they use the dedicated `VBin` expression.
        Ty::I64 | Ty::V128 => base32 + 0x12,
    }
}

fn cmp_op(op: CmpOp, ty: Ty) -> u8 {
    let base32: u8 = match op {
        CmpOp::Eq => 0x46,
        CmpOp::Ne => 0x47,
        CmpOp::LtS => 0x48,
        CmpOp::LtU => 0x49,
        CmpOp::GtS => 0x4a,
        CmpOp::GtU => 0x4b,
        CmpOp::LeS => 0x4c,
        CmpOp::LeU => 0x4d,
        CmpOp::GeS => 0x4e,
        CmpOp::GeU => 0x4f,
    };
    match ty {
        Ty::I32 => base32,
        // The i64 comparison block mirrors the i32 one, 11 opcodes later.
        Ty::I64 | Ty::V128 => base32 + 0x0b,
    }
}

fn lane_shift_op(op: VShiftOp, lane: VLane) -> u32 {
    let base = match lane {
        VLane::I8 => 0x6b,
        VLane::I16 => 0x8b,
        VLane::I32 | VLane::F32 => 0xab,
        _ => 0xcb,
    };
    base + match op {
        VShiftOp::Shl => 0,
        VShiftOp::ShrS => 1,
        VShiftOp::ShrU => 2,
    }
}

// What a block boundary costs, split three ways, and process-wide. These were
// thread-locals, so every count landed on a compile-pool worker and `gexc
// build` read its own near-zero copy. Atomics are ample against the time each
// block spends being lifted and encoded.
/// Bytes emitted moving guest registers between the context and locals.
pub static CTX_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Bytes emitted on the safepoint countdown and its call.
pub static SAFEPOINT_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Bytes emitted proving a guest address is inside guest memory, on every
/// load and store whose address is not a literal constant.
pub static GUARD_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Bytes emitted choosing where to go next: the directory load, the slot
/// arithmetic, the table selection and the indirect tail call.
pub static DISPATCH_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Regions formed, and how many blocks went into them.
pub static REGIONS: std::sync::Mutex<(usize, usize)> = std::sync::Mutex::new((0, 0));
/// How often each guest register crosses a boundary, under
/// `GEX_REG_HISTOGRAM`. Which registers those are is the first question
/// anyone asks about that percentage.
pub static REG_TRAFFIC: std::sync::Mutex<Option<HashMap<GuestReg, usize>>> =
    std::sync::Mutex::new(None);

fn note_traffic(r: GuestReg) {
    // Only `gexc` reads it, under the variable: a lock per register load and
    // store is otherwise a translation cost for nothing.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("GEX_REG_HISTOGRAM").is_some()) {
        return;
    }
    let mut t = REG_TRAFFIC.lock().unwrap_or_else(|e| e.into_inner());
    *t.get_or_insert_with(HashMap::new).entry(r).or_default() += 1;
}

/// Every guest register a block reads, and every one it writes.
///
/// Unlike [`live_in_registers`] this ignores order: a region holds its
/// registers in locals from entry to exit, so only whether a register is read
/// (needs a load in) and whether it is written (needs a store out) matters.
/// Keeping the two apart saves the store on a read-only register and the load
/// on a write-only one, paid on every entry and exit.
pub(crate) fn registers_read_and_written(block: &Block) -> (Vec<GuestReg>, Vec<GuestReg>) {
    let (mut read, mut written) = (Vec::new(), Vec::new());
    for s in &block.stmts {
        match s {
            Stmt::Def {
                expr: Expr::ReadReg(r),
                ..
            } => {
                if !read.contains(r) {
                    read.push(*r);
                }
            }
            Stmt::WriteReg { reg, .. } => {
                if !written.contains(reg) {
                    written.push(*reg);
                }
            }
            _ => {}
        }
    }
    (read, written)
}

/// Iterations of a fused loop between trips through the dispatcher.
///
/// The same cadence the context's safepoint counter uses, so the time between
/// signal polls does not depend on the loop's shape. Overridable to bisect a
/// scheduling problem; no measurable difference between 1024 and a million.
fn loop_poll() -> i32 {
    static POLL: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *POLL.get_or_init(|| {
        std::env::var("GEX_LOOP_POLL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

/// Whether a statement can leave the register cache stale.
pub fn clobbers_guest_state(s: &Stmt) -> bool {
    match s {
        Stmt::HostCall { func, .. } => func.clobbers_guest_state(),
        Stmt::Def {
            expr: Expr::HostCall { func, .. },
            ..
        } => func.clobbers_guest_state(),
        _ => false,
    }
}

/// The architectural registers a block reads before writing.
///
/// Exactly the ones the fused loop form must load before the loop, since the
/// context is left stale inside it. A register written before reading needs no
/// preload, and leaving it out matters: every preloaded register stays live
/// across the whole loop, costing the host's allocator a machine register.
fn live_in_registers(block: &Block) -> Vec<GuestReg> {
    let mut live_in = Vec::new();
    let mut written = Vec::new();
    for s in &block.stmts {
        match s {
            Stmt::Def {
                expr: Expr::ReadReg(r),
                ..
            } => {
                if !written.contains(r) && !live_in.contains(r) {
                    live_in.push(*r);
                }
            }
            Stmt::WriteReg { reg, .. } => {
                if !written.contains(reg) {
                    written.push(*reg);
                }
            }
            _ => {}
        }
    }
    live_in
}

fn lane_cvt_op(op: VCvtOp, from: VLane) -> u32 {
    match (op, from) {
        (VCvtOp::FpToIntSat { signed: true }, VLane::F32) => 0xf8,
        (VCvtOp::FpToIntSat { signed: false }, VLane::F32) => 0xf9,
        // The f64 forms produce two lanes and zero the upper two.
        (VCvtOp::FpToIntSat { signed: true }, VLane::F64) => 0xfc,
        (VCvtOp::FpToIntSat { signed: false }, VLane::F64) => 0xfd,
        (VCvtOp::IntToFp { signed: true }, VLane::I32) => 0xfa,
        (VCvtOp::IntToFp { signed: false }, VLane::I32) => 0xfb,
        // wasm has no i64x2 <-> f64x2 conversion, so the lifter must never
        // ask for one; `f64x2.convert_low_i32x4_s` is not a substitute.
        (VCvtOp::IntToFp { signed: true }, VLane::I64)
        | (VCvtOp::IntToFp { signed: false }, VLane::I64)
        | (VCvtOp::FpToIntSat { .. }, _) => {
            panic!("no wasm SIMD conversion for {op:?} from {from:?}")
        }
        (VCvtOp::IntToFp { signed: true }, _) => 0xfe,
        (VCvtOp::IntToFp { signed: false }, _) => 0xff,
        (VCvtOp::DemoteF64, _) => 0x5e,
        (VCvtOp::PromoteF32, _) => 0x5f,
        (VCvtOp::ExtendLow { signed: true }, VLane::I8) => 0x87,
        (VCvtOp::ExtendHigh { signed: true }, VLane::I8) => 0x88,
        (VCvtOp::ExtendLow { signed: false }, VLane::I8) => 0x89,
        (VCvtOp::ExtendHigh { signed: false }, VLane::I8) => 0x8a,
        (VCvtOp::ExtendLow { signed: true }, VLane::I16) => 0xa7,
        (VCvtOp::ExtendHigh { signed: true }, VLane::I16) => 0xa8,
        (VCvtOp::ExtendLow { signed: false }, VLane::I16) => 0xa9,
        (VCvtOp::ExtendHigh { signed: false }, VLane::I16) => 0xaa,
        (VCvtOp::ExtendLow { signed: true }, _) => 0xc7,
        (VCvtOp::ExtendHigh { signed: true }, _) => 0xc8,
        (VCvtOp::ExtendLow { signed: false }, _) => 0xc9,
        (VCvtOp::ExtendHigh { signed: false }, _) => 0xca,
        (VCvtOp::NarrowSat { signed: true }, VLane::I16) => 0x65,
        (VCvtOp::NarrowSat { signed: false }, VLane::I16) => 0x66,
        (VCvtOp::NarrowSat { signed: true }, VLane::I32) => 0x85,
        (VCvtOp::NarrowSat { signed: false }, VLane::I32) => 0x86,
        // wasm narrows only i16 and i32 lanes; the lifter clamps doublewords
        // itself.
        (VCvtOp::NarrowSat { .. }, _) => {
            panic!("no wasm SIMD conversion for {op:?} from {from:?}")
        }
    }
}

fn natural_align(sz: MemSz) -> u32 {
    match sz {
        MemSz::B1 => 0,
        MemSz::B2 => 1,
        MemSz::B4 => 2,
        MemSz::B8 => 3,
    }
}

fn load_op(ty: Ty, sz: MemSz, signed: bool) -> u8 {
    match (ty, sz, signed) {
        (Ty::I32, MemSz::B1, false) => 0x2d,
        (Ty::I32, MemSz::B1, true) => 0x2c,
        (Ty::I32, MemSz::B2, false) => 0x2f,
        (Ty::I32, MemSz::B2, true) => 0x2e,
        (Ty::I32, _, _) => 0x28,
        (Ty::I64, MemSz::B1, false) => 0x31,
        (Ty::I64, MemSz::B1, true) => 0x30,
        (Ty::I64, MemSz::B2, false) => 0x33,
        (Ty::I64, MemSz::B2, true) => 0x32,
        (Ty::I64, MemSz::B4, false) => 0x35,
        (Ty::I64, MemSz::B4, true) => 0x34,
        (Ty::I64, MemSz::B8, _) => 0x29,
        // Vector accesses go through `VLoad`/`VStore`, never this table.
        (Ty::V128, _, _) => unreachable!("vector load reached the scalar table"),
    }
}

fn store_op(ty: Ty, sz: MemSz) -> u8 {
    match (ty, sz) {
        (Ty::I32, MemSz::B1) => 0x3a,
        (Ty::I32, MemSz::B2) => 0x3b,
        (Ty::I32, _) => 0x36,
        (Ty::I64, MemSz::B1) => 0x3c,
        (Ty::I64, MemSz::B2) => 0x3d,
        (Ty::I64, MemSz::B4) => 0x3e,
        (Ty::I64, MemSz::B8) => 0x37,
        (Ty::V128, _) => unreachable!("vector store reached the scalar table"),
    }
}

/// Position of a `(type, size)` pair within an atomic opcode group.
///
/// Every atomic group in the threads proposal lists its seven variants in the
/// same order: `i32`, `i64`, `i32.8`, `i32.16`, `i64.8`, `i64.16`, `i64.32`.
fn atomic_slot(ty: Ty, sz: MemSz) -> u32 {
    match (ty, sz) {
        (Ty::I32, MemSz::B4) => 0,
        (Ty::I64, MemSz::B8) => 1,
        (Ty::I32, MemSz::B1) => 2,
        (Ty::I32, MemSz::B2) => 3,
        (Ty::I64, MemSz::B1) => 4,
        (Ty::I64, MemSz::B2) => 5,
        (Ty::I64, MemSz::B4) => 6,
        // A 32-bit value with an 8-byte access cannot occur; treat it as the
        // widest i32 form so the mapping stays total.
        (Ty::I32, MemSz::B8) => 0,
        // There are no atomic vector accesses in the architecture.
        (Ty::V128, _) => unreachable!("atomic vector access"),
    }
}

fn atomic_load_op(ty: Ty, sz: MemSz) -> u32 {
    0x10 + atomic_slot(ty, sz)
}

fn atomic_store_op(ty: Ty, sz: MemSz) -> u32 {
    0x17 + atomic_slot(ty, sz)
}

fn atomic_rmw_op(op: RmwOp, ty: Ty, sz: MemSz) -> u32 {
    let base = match op {
        RmwOp::Add => 0x1e,
        RmwOp::Sub => 0x25,
        RmwOp::And => 0x2c,
        RmwOp::Or => 0x33,
        RmwOp::Xor => 0x3a,
        RmwOp::Xchg => 0x41,
    };
    base + atomic_slot(ty, sz)
}

fn atomic_cmpxchg_op(ty: Ty, sz: MemSz) -> u32 {
    0x48 + atomic_slot(ty, sz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dying_after_a_block_with_values_and_no_statements() {
        let mut b = Block::new(0x1000);
        b.n_values = 3;
        let d = dying_after(&b);
        assert_eq!(d.start, vec![0]);
        assert!(d.vals.is_empty());
    }

    #[test]
    fn atomic_groups_are_seven_wide() {
        assert_eq!(atomic_rmw_op(RmwOp::Add, Ty::I32, MemSz::B4), 0x1e);
        assert_eq!(atomic_rmw_op(RmwOp::Add, Ty::I64, MemSz::B4), 0x24);
        assert_eq!(atomic_rmw_op(RmwOp::Sub, Ty::I32, MemSz::B4), 0x25);
        assert_eq!(atomic_cmpxchg_op(Ty::I64, MemSz::B8), 0x49);
    }

    #[test]
    fn integer_opcodes_match_the_spec() {
        assert_eq!(bin_op(BinOp::Add, Ty::I32), 0x6a);
        assert_eq!(bin_op(BinOp::Add, Ty::I64), 0x7c);
        assert_eq!(bin_op(BinOp::Rotr, Ty::I64), 0x8a);
        assert_eq!(cmp_op(CmpOp::Eq, Ty::I64), 0x51);
        assert_eq!(cmp_op(CmpOp::GeU, Ty::I64), 0x5a);
    }

    #[test]
    fn dispatch_indices_follow_the_page_directory() {
        // A page whose directory entry is `base` maps its instructions to
        // consecutive slots from there.
        assert_eq!(gex_abi::dispatch_index(0x400000, 2048), 2048);
        assert_eq!(gex_abi::dispatch_index(0x400004, 2048), 2049);
        assert_eq!(gex_abi::dispatch_index(0x400ffc, 2048), 2048 + 1023);
        // The next page starts over from its own entry.
        assert_eq!(gex_abi::dispatch_index(0x401000, 4096), 4096);
        // A page with no translation lands in the reserved miss span.
        assert!(gex_abi::dispatch_index(0x400004, 0) < gex_abi::MISS_SLOTS);
    }
}
