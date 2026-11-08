//! Profiling instrumentation for tier-1 code (`GEX_PROFILE`).
//!
//! Off by default. Off means **byte-identical** modules: nothing in this file
//! is consulted by the emitter unless [`set_enabled`] turned it on before the
//! module was built.
//!
//! # Where the counters live
//!
//! Not in guest memory. A profiled module imports a second linear memory,
//! `gex.profile`, which becomes memory index 1 (the guest's memory is always
//! the first memory import, index 0). The host creates one such memory **per
//! store**, which is one per guest thread, so every counter is written by
//! exactly one thread and plain loads and stores are exact without atomics.
//! The guest cannot address memory 1 at all, and its own address-space layout
//! is unchanged, so profiling cannot change what a program observes.
//!
//! Memory 1 is laid out as follows (all little-endian):
//!
//! ```text
//!   0   u64  THREAD_MAGIC "GEXPROF1", written by the host
//!   8   u32  THREAD_VERSION
//!  16   u64  ENTRY_PC      target of the last call/jump/ret, 0 once consumed
//!  24   u32  ENTRY_KIND    1 call, 2 indirect jump, 3 return
//!  32   u32  SS_TOP        shadow-stack pushes minus pops (monotonic ring index)
//!  36   u32  SS_FLOOR      lowest index still held by the ring
//!  40   u32  REGIONS_END   host-maintained bump pointer over the module regions
//!  64   ...  SHADOW STACK  SS_ENTRIES x { u64 return address, u64 callee }
//!  REGIONS_START ...       one region per instantiated module, back to back
//! ```
//!
//! # A module's region
//!
//! Each profiled module instance owns one region. The module defines a
//! mutable `i32` global holding the region's base, and exports two helpers:
//! `gexprofsize() -> i32` (the region's size in bytes) and
//! `gexprofinit(base: i32)`, which sets the global and copies the region's
//! header and block descriptors in from a passive data segment. The host
//! allocates `size` bytes (aligned to [`REGION_ALIGN`]) at `REGIONS_END`,
//! calls `gexprofinit`, and advances `REGIONS_END`. Nothing runs in the
//! instance before that, because its table slots belong to this store alone.
//! The base is not baked into code, so module caches stay valid.
//!
//! ```text
//! header (REGION_HEADER bytes)
//!    0 u32 REGION_MAGIC "GXPM"   4 u32 REGION_VERSION
//!    8 u32 n_blocks             12 u32 size (whole region)
//!   16 u32 desc_off             20 u32 rec_off
//! descriptors, n_blocks x DESC_BYTES
//!    0 u64 start   8 u64 end   16 u64 target   24 u64 aux
//!   32 u32 rec (offset of this block's record, from the region base)
//!   36 u32 flags (see `flags`)
//! records, per block:
//!    +0  u64 entries      every time the block's code began to run
//!    +8  u64 call         entries that were the target of a call
//!   +16  u64 jump         entries that were the target of an indirect jump
//!   +24  u64 ret          entries that were the target of a return
//!   then, by terminator:
//!   Branch:        +32 u64 taken
//!   Indirect jump
//!   or call:       +32 u64 total  +40 u64 replaced  +48 4 x {u64 target, u64 count}
//!   Return:        +32 u64 expected  +40 u64 unexpected  +48 u64 unchecked
//! ```
//!
//! `target` is the static successor (Goto target, Branch taken, Syscall
//! next); `aux` is the Branch not-taken successor or a call's return address.
//!
//! The indirect-target table is Metwally's space-saving algorithm over four
//! slots: exact when a site has at most four targets (`replaced == 0`),
//! otherwise every target with more than a quarter of the site's traffic is
//! guaranteed to be present and counts are over-estimates by at most the
//! smallest slot's count.
//!
//! Returns are checked against a shadow stack that every call pushes: a return
//! to the address the matching call pushed is `expected`; one elsewhere is
//! `unexpected` (longjmp, coroutine switches, a `ret` used as a jump, a signal
//! handler returning through the trampoline); one with nothing to compare
//! against (the ring is empty or wrapped, as in a forked child or a thread's
//! first frame) is `unchecked`. On a mismatch the ring resynchronises if the
//! target is within [`RESYNC_DEPTH`] frames below the top.

use crate::wasm::{FuncBody, ValType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn profiling instrumentation on or off for every module built after this.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether modules built now are instrumented.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// The import name of the profile memory (module `gex`).
pub const MEMORY_IMPORT: &str = "profile";
/// The memory index profiled code addresses the profile memory as.
pub const MEM: u32 = 1;
/// Export: `() -> i32`, the region size this module needs.
pub const EXPORT_SIZE: &str = "gexprofsize";
/// Export: `(base: i32) -> ()`, place the region at `base` and initialise it.
pub const EXPORT_INIT: &str = "gexprofinit";

pub const THREAD_MAGIC: u64 = u64::from_le_bytes(*b"GEXPROF1");
pub const THREAD_VERSION: u32 = 1;
pub const ENTRY_PC: u32 = 16;
pub const ENTRY_KIND: u32 = 24;
pub const SS_TOP: u32 = 32;
pub const SS_FLOOR: u32 = 36;
pub const REGIONS_END: u32 = 40;
pub const SS_BASE: u32 = 64;
/// Shadow-stack capacity. A ring: deeper recursion than this makes the
/// oldest frames' returns `unchecked`, never wrong.
pub const SS_ENTRIES: u32 = 1 << 18;
pub const SS_ENTRY_BYTES: u32 = 16;
/// How far below the top a mismatched return looks for its target.
pub const RESYNC_DEPTH: u32 = 64;
/// Where the first module region starts.
pub const REGIONS_START: u32 = (SS_BASE + SS_ENTRIES * SS_ENTRY_BYTES + 0xffff) & !0xffff;
pub const REGION_ALIGN: u32 = 64;

pub const REGION_MAGIC: u32 = u32::from_le_bytes(*b"GXPM");
pub const REGION_VERSION: u32 = 1;
pub const REGION_HEADER: u32 = 32;
pub const DESC_BYTES: u32 = 40;

/// Entry kinds, as stored in `ENTRY_KIND` and used to index the record.
pub const KIND_CALL: i32 = 1;
pub const KIND_JUMP: i32 = 2;
pub const KIND_RET: i32 = 3;

/// Record field offsets.
pub const REC_ENTRIES: u32 = 0;
pub const REC_BASE_BYTES: u32 = 32;
pub const REC_TAKEN: u32 = 32;
pub const REC_IND_TOTAL: u32 = 32;
pub const REC_IND_REPLACED: u32 = 40;
pub const REC_IND_SLOTS: u32 = 48;
pub const IND_SLOTS: u32 = 4;
pub const REC_RET_EXPECTED: u32 = 32;
pub const REC_RET_UNEXPECTED: u32 = 40;
pub const REC_RET_UNCHECKED: u32 = 48;

/// Descriptor flags.
pub mod flags {
    /// Low three bits: the terminator.
    pub const TERM_MASK: u32 = 7;
    pub const TERM_GOTO: u32 = 0;
    pub const TERM_BRANCH: u32 = 1;
    pub const TERM_INDIRECT: u32 = 2;
    pub const TERM_SYSCALL: u32 = 3;
    pub const TERM_ILLEGAL: u32 = 4;
    /// The indirect branch is a return.
    pub const RET: u32 = 1 << 3;
    /// The block ends in a call (`call_return` is set).
    pub const CALL: u32 = 1 << 4;
    /// The block is code in the process's other mode (a WoW64 process's
    /// 32-bit code), translated into that mode's module.
    pub const OTHER_MODE: u32 = 1 << 5;
}

/// Size of a block's record, by its flags.
pub fn record_bytes(fl: u32) -> u32 {
    REC_BASE_BYTES
        + match fl & flags::TERM_MASK {
            flags::TERM_BRANCH => 8,
            flags::TERM_INDIRECT if fl & flags::RET != 0 => 24,
            flags::TERM_INDIRECT => 16 + IND_SLOTS * 16,
            _ => 0,
        }
}

/// Descriptor flags and static successors of a block.
pub fn describe(b: &gex_ir::Block) -> (u32, u64, u64) {
    use gex_ir::Term;
    let call = if b.call_return.is_some() { flags::CALL } else { 0 };
    let aux_call = b.call_return.unwrap_or(0);
    match b.term {
        Term::Goto { target } => (flags::TERM_GOTO | call, target, aux_call),
        Term::Branch {
            taken, not_taken, ..
        } => (flags::TERM_BRANCH | call, taken, not_taken),
        Term::Indirect { is_ret, .. } => (
            flags::TERM_INDIRECT | call | if is_ret { flags::RET } else { 0 },
            0,
            aux_call,
        ),
        Term::Syscall { next, .. } => (flags::TERM_SYSCALL | call, next, aux_call),
        Term::Illegal { .. } => (flags::TERM_ILLEGAL | call, 0, aux_call),
    }
}

/// What the tier-1 emitter needs to instrument a module's blocks.
#[derive(Debug, Default)]
pub struct Emit {
    /// The mutable global holding the region base.
    pub base_global: u32,
    /// `(rec: i32) -> ()`: an entry whose `ENTRY_PC` matched; counts it by kind.
    pub f_entry: u32,
    /// `(rec: i32, target: i64, kind: i32) -> ()`: an indirect jump or call.
    pub f_indirect: u32,
    /// `(rec: i32, target: i64) -> ()`: a return.
    pub f_ret: u32,
    /// `(ret_addr: i64, callee: i64) -> ()`: a call.
    pub f_call: u32,
    /// Block start -> offset of its record from the region base.
    pub records: HashMap<u64, u32>,
}

/// A module's region: the records, and the bytes `gexprofinit` copies in.
pub struct Region {
    pub records: HashMap<u64, u32>,
    /// Header plus descriptors.
    pub init: Vec<u8>,
    /// Whole region size, header and records included.
    pub size: u32,
}

/// Lay out the region for these blocks, in the order given (`other_mode`:
/// a module of the process's other mode's code). A start that appears twice
/// keeps its first record.
pub fn layout<'a>(blocks: impl Iterator<Item = &'a gex_ir::Block>, other_mode: bool) -> Region {
    let mut seen: HashMap<u64, u32> = HashMap::new();
    let mut descs: Vec<(u64, u64, u64, u64, u32)> = Vec::new();
    for b in blocks {
        if seen.contains_key(&b.start) {
            continue;
        }
        let (fl, target, aux) = describe(b);
        let fl = if other_mode { fl | flags::OTHER_MODE } else { fl };
        seen.insert(b.start, descs.len() as u32);
        descs.push((b.start, b.end, target, aux, fl));
    }
    let n = descs.len() as u32;
    let desc_off = REGION_HEADER;
    let rec_off = (desc_off + n * DESC_BYTES + 7) & !7;
    let mut records = HashMap::with_capacity(descs.len());
    let mut at = rec_off;
    let mut init = Vec::with_capacity(rec_off as usize);
    init.extend_from_slice(&REGION_MAGIC.to_le_bytes());
    init.extend_from_slice(&REGION_VERSION.to_le_bytes());
    init.extend_from_slice(&n.to_le_bytes());
    init.extend_from_slice(&0u32.to_le_bytes()); // size, patched below
    init.extend_from_slice(&desc_off.to_le_bytes());
    init.extend_from_slice(&rec_off.to_le_bytes());
    init.resize(REGION_HEADER as usize, 0);
    for (start, end, target, aux, fl) in &descs {
        records.insert(*start, at);
        init.extend_from_slice(&start.to_le_bytes());
        init.extend_from_slice(&end.to_le_bytes());
        init.extend_from_slice(&target.to_le_bytes());
        init.extend_from_slice(&aux.to_le_bytes());
        init.extend_from_slice(&at.to_le_bytes());
        init.extend_from_slice(&fl.to_le_bytes());
        at += record_bytes(*fl);
    }
    init.resize(rec_off as usize, 0);
    let size = (at + REGION_ALIGN - 1) & !(REGION_ALIGN - 1);
    init[12..16].copy_from_slice(&size.to_le_bytes());
    Region {
        records,
        init,
        size,
    }
}

// ---------------------------------------------------------------------------
// Helper function bodies. Each is a finished code-section entry.
// ---------------------------------------------------------------------------

const I64_LOAD: u8 = 0x29;
const I64_STORE: u8 = 0x37;
const I32_LOAD: u8 = 0x28;
const I32_STORE: u8 = 0x36;

/// `gexprofsize() -> i32`.
pub fn body_size(size: u32) -> Vec<u8> {
    let mut f = FuncBody::new(0);
    f.i32_const(size as i32);
    f.finish()
}

/// `gexprofinit(base: i32)`: set the base global and copy the header and
/// descriptors (passive data segment `seg`) to `base` in memory 1.
pub fn body_init(global: u32, seg: u32, len: u32) -> Vec<u8> {
    let mut f = FuncBody::new(1);
    f.local_get(0);
    f.global_set(global);
    f.local_get(0); // dest
    f.i32_const(0); // src
    f.i32_const(len as i32);
    f.memory_init(seg, MEM);
    f.finish()
}

/// `(rec: i32)`: the block's `ENTRY_PC` matched, so count the entry under its
/// kind and consume it.
pub fn body_entry(global: u32) -> Vec<u8> {
    let mut f = FuncBody::new(1);
    let addr = f.add_local(ValType::I32);
    // addr = base + rec + kind * 8
    f.global_get(global);
    f.local_get(0);
    f.byte(0x6a); // i32.add
    f.i32_const(0);
    f.load_m(I32_LOAD, 2, MEM, ENTRY_KIND);
    f.i32_const(3);
    f.byte(0x74); // i32.shl
    f.byte(0x6a); // i32.add
    f.local_tee(addr);
    f.local_get(addr);
    f.load_m(I64_LOAD, 3, MEM, 0);
    f.i64_const(1);
    f.byte(0x7c); // i64.add
    f.store_m(I64_STORE, 3, MEM, 0);
    f.i32_const(0);
    f.i64_const(0);
    f.store_m(I64_STORE, 3, MEM, ENTRY_PC);
    f.finish()
}

/// Record the next block to be entered: `ENTRY_PC = target; ENTRY_KIND = kind`.
fn set_entry(f: &mut FuncBody, target_local: u32, kind: Option<i32>, kind_local: u32) {
    f.i32_const(0);
    f.local_get(target_local);
    f.store_m(I64_STORE, 3, MEM, ENTRY_PC);
    f.i32_const(0);
    match kind {
        Some(k) => f.i32_const(k),
        None => f.local_get(kind_local),
    }
    f.store_m(I32_STORE, 2, MEM, ENTRY_KIND);
}

/// `(rec: i32, target: i64, kind: i32)`: space-saving top-4 over the site's
/// targets, then record the entry.
pub fn body_indirect(global: u32) -> Vec<u8> {
    let mut f = FuncBody::new(3);
    let (rec, target, kind) = (0, 1, 2);
    let base = f.add_local(ValType::I32);
    let i = f.add_local(ValType::I32);
    let min_at = f.add_local(ValType::I32);
    let min = f.add_local(ValType::I64);
    let slot = f.add_local(ValType::I32);

    set_entry(&mut f, target, None, kind);
    // base = global + rec
    f.global_get(global);
    f.local_get(rec);
    f.byte(0x6a);
    f.local_set(base);
    // total += 1
    incr(&mut f, base, REC_IND_TOTAL);

    // Pass 1: a slot already holding the target, or the first empty one
    // (count zero), whichever comes first -- targets fill slots in order and
    // are never removed, so an empty slot means no later slot holds it.
    f.block_void(); // done
    f.i32_const(0);
    f.local_set(i);
    f.i64_const(-1);
    f.local_set(min);
    f.i32_const(0);
    f.local_set(min_at);
    f.loop_void();
    // slot = base + REC_IND_SLOTS + i*16
    f.local_get(base);
    f.local_get(i);
    f.i32_const(4);
    f.byte(0x74); // shl
    f.byte(0x6a);
    f.local_set(slot);
    // if count == 0: claim
    f.local_get(slot);
    f.load_m(I64_LOAD, 3, MEM, REC_IND_SLOTS + 8);
    f.byte(0x50); // i64.eqz
    f.if_void();
    f.local_get(slot);
    f.local_get(target);
    f.store_m(I64_STORE, 3, MEM, REC_IND_SLOTS);
    f.local_get(slot);
    f.i64_const(1);
    f.store_m(I64_STORE, 3, MEM, REC_IND_SLOTS + 8);
    f.br(2); // out of loop and `done`
    f.end();
    // if target matches: count
    f.local_get(slot);
    f.load_m(I64_LOAD, 3, MEM, REC_IND_SLOTS);
    f.local_get(target);
    f.byte(0x51); // i64.eq
    f.if_void();
    incr(&mut f, slot, REC_IND_SLOTS + 8);
    f.br(2);
    f.end();
    // track the minimum
    f.local_get(slot);
    f.load_m(I64_LOAD, 3, MEM, REC_IND_SLOTS + 8);
    f.local_get(min);
    f.byte(0x54); // i64.lt_u
    f.if_void();
    f.local_get(slot);
    f.load_m(I64_LOAD, 3, MEM, REC_IND_SLOTS + 8);
    f.local_set(min);
    f.local_get(slot);
    f.local_set(min_at);
    f.end();
    // i += 1; continue while i < 4
    f.local_get(i);
    f.i32_const(1);
    f.byte(0x6a);
    f.local_tee(i);
    f.i32_const(IND_SLOTS as i32);
    f.byte(0x48); // i32.lt_s
    f.br_if(0);
    f.end(); // loop
    // Full and absent: replace the minimum, inheriting its count.
    f.local_get(min_at);
    f.local_get(target);
    f.store_m(I64_STORE, 3, MEM, REC_IND_SLOTS);
    f.local_get(min_at);
    f.local_get(min);
    f.i64_const(1);
    f.byte(0x7c);
    f.store_m(I64_STORE, 3, MEM, REC_IND_SLOTS + 8);
    incr(&mut f, base, REC_IND_REPLACED);
    f.end(); // done
    f.finish()
}

/// `(ret_addr: i64, callee: i64)`: push a frame on the shadow stack, and
/// record that the next block entered is a call target.
pub fn body_call() -> Vec<u8> {
    let mut f = FuncBody::new(2);
    let (ret, callee) = (0, 1);
    let top = f.add_local(ValType::I32);
    let at = f.add_local(ValType::I32);
    let kind_unused = f.add_local(ValType::I32);
    set_entry(&mut f, callee, Some(KIND_CALL), kind_unused);
    f.i32_const(0);
    f.load_m(I32_LOAD, 2, MEM, SS_TOP);
    f.local_set(top);
    ring_slot(&mut f, top, at);
    f.local_get(at);
    f.local_get(ret);
    f.store_m(I64_STORE, 3, MEM, SS_BASE);
    f.local_get(at);
    f.local_get(callee);
    f.store_m(I64_STORE, 3, MEM, SS_BASE + 8);
    // top += 1; if top - floor > SS_ENTRIES: floor = top - SS_ENTRIES
    f.i32_const(0);
    f.local_get(top);
    f.i32_const(1);
    f.byte(0x6a);
    f.local_tee(top);
    f.store_m(I32_STORE, 2, MEM, SS_TOP);
    f.local_get(top);
    f.i32_const(0);
    f.load_m(I32_LOAD, 2, MEM, SS_FLOOR);
    f.byte(0x6b); // sub
    f.i32_const(SS_ENTRIES as i32);
    f.byte(0x4b); // i32.gt_u
    f.if_void();
    f.i32_const(0);
    f.local_get(top);
    f.i32_const(SS_ENTRIES as i32);
    f.byte(0x6b);
    f.store_m(I32_STORE, 2, MEM, SS_FLOOR);
    f.end();
    f.finish()
}

/// `(rec: i32, target: i64)`: check a return against the shadow stack.
pub fn body_ret(global: u32) -> Vec<u8> {
    let mut f = FuncBody::new(2);
    let (rec, target) = (0, 1);
    let base = f.add_local(ValType::I32);
    let top = f.add_local(ValType::I32);
    let floor = f.add_local(ValType::I32);
    let at = f.add_local(ValType::I32);
    let depth = f.add_local(ValType::I32);
    let unused = f.add_local(ValType::I32);
    set_entry(&mut f, target, Some(KIND_RET), unused);
    f.global_get(global);
    f.local_get(rec);
    f.byte(0x6a);
    f.local_set(base);
    f.i32_const(0);
    f.load_m(I32_LOAD, 2, MEM, SS_TOP);
    f.local_set(top);
    f.i32_const(0);
    f.load_m(I32_LOAD, 2, MEM, SS_FLOOR);
    f.local_set(floor);
    f.block_void(); // done
    // Empty (or wrapped past): nothing to check against.
    f.local_get(top);
    f.local_get(floor);
    f.byte(0x4d); // i32.le_u
    f.if_void();
    incr(&mut f, base, REC_RET_UNCHECKED);
    f.br(1);
    f.end();
    // top -= 1; the frame the matching call pushed
    f.local_get(top);
    f.i32_const(1);
    f.byte(0x6b);
    f.local_tee(top);
    f.local_set(top);
    ring_slot(&mut f, top, at);
    f.local_get(at);
    f.load_m(I64_LOAD, 3, MEM, SS_BASE);
    f.local_get(target);
    f.byte(0x51); // i64.eq
    f.if_void();
    f.i32_const(0);
    f.local_get(top);
    f.store_m(I32_STORE, 2, MEM, SS_TOP);
    incr(&mut f, base, REC_RET_EXPECTED);
    f.br(1);
    f.end();
    incr(&mut f, base, REC_RET_UNEXPECTED);
    // Resynchronise: a longjmp that skips frames returns to one further down.
    f.i32_const(RESYNC_DEPTH as i32);
    f.local_set(depth);
    f.loop_void();
    // stop at the floor or after RESYNC_DEPTH frames
    f.local_get(top);
    f.local_get(floor);
    f.byte(0x4d); // le_u
    f.br_if(1);
    f.local_get(depth);
    f.byte(0x45); // i32.eqz
    f.br_if(1);
    f.local_get(depth);
    f.i32_const(1);
    f.byte(0x6b);
    f.local_set(depth);
    f.local_get(top);
    f.i32_const(1);
    f.byte(0x6b);
    f.local_set(top);
    ring_slot(&mut f, top, at);
    f.local_get(at);
    f.load_m(I64_LOAD, 3, MEM, SS_BASE);
    f.local_get(target);
    f.byte(0x51);
    f.if_void();
    f.i32_const(0);
    f.local_get(top);
    f.store_m(I32_STORE, 2, MEM, SS_TOP);
    f.br(2);
    f.end();
    f.br(0);
    f.end(); // loop
    f.end(); // done
    f.finish()
}

/// `at = (top & (SS_ENTRIES-1)) * SS_ENTRY_BYTES`, an address relative to
/// `SS_BASE` (which the caller adds as a static offset).
fn ring_slot(f: &mut FuncBody, top: u32, at: u32) {
    f.local_get(top);
    f.i32_const((SS_ENTRIES - 1) as i32);
    f.byte(0x71); // and
    f.i32_const(SS_ENTRY_BYTES.trailing_zeros() as i32);
    f.byte(0x74); // shl
    f.local_set(at);
}

/// `mem1[addr + off] += 1` for an i64 counter, `addr` in a local.
fn incr(f: &mut FuncBody, addr: u32, off: u32) {
    f.local_get(addr);
    f.local_get(addr);
    f.load_m(I64_LOAD, 3, MEM, off);
    f.i64_const(1);
    f.byte(0x7c);
    f.store_m(I64_STORE, 3, MEM, off);
}

/// Inline: `mem1[base + off] += 1`, the base read from `global`.
pub fn emit_count(f: &mut FuncBody, global: u32, off: u32) {
    f.global_get(global);
    f.global_get(global);
    f.load_m(I64_LOAD, 3, MEM, off);
    f.i64_const(1);
    f.byte(0x7c);
    f.store_m(I64_STORE, 3, MEM, off);
}

/// Inline, at a block's start: count the entry, and if the transfer that got
/// here announced this block (`ENTRY_PC == pc`), count it under its kind.
pub fn emit_block_entry(f: &mut FuncBody, e: &Emit, pc: u64, rec: u32) {
    emit_count(f, e.base_global, rec + REC_ENTRIES);
    f.i32_const(0);
    f.load_m(I64_LOAD, 3, MEM, ENTRY_PC);
    f.i64_const(pc as i64);
    f.byte(0x51); // i64.eq
    f.if_void();
    f.i32_const(rec as i32);
    f.call(e.f_entry);
    f.end();
}
