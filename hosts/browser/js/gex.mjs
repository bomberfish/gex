// The browser runtime: instantiate the shim (gex-kernel compiled to wasm) and
// a guest module against one memory, and run. The shim imports the guest's
// memory, so a syscall reads guest buffers in place. `run-node.mjs` is the
// Node entry.

// How far a process's memory may grow. One memory per process. See
// `gex_abi::GUEST_MEM_INITIAL_BYTES`.
const GUEST_BYTES = 0xffff0000;

// A 64-bit guest memory: the wasm64 shim and seed, a
// memory64 of `gex_abi::LIMIT64` (16 GiB). The shim's own addresses (its
// stack, data and the low runtime words) stay below 4 GiB either way; only
// the memory's type and the i64 globals of a wasm64 module differ.
import { kernelThread, lazyFetch, lazyReady, processMemories, wasmMemory, wasmMemoryGone } from "./lazy-fetch.mjs";
import { blockIo } from './blocks.mjs';
import { SP_CODE } from './mode.mjs';
import { pageFs, pageFsSuspending } from './page-fs.mjs';
import { xcEnter, xcLeave, xcRead, xcSleep, xcWrite } from "./xcache.mjs";
import { single, sched } from "./single.mjs";
import { logNamed, logOn, logRoutine } from "./log.mjs";
import { frameOf } from './dgram-frames.mjs';

/** The longest an exec waits for the other memory flavour's runtime. */
const OTHER_WAIT_MS = 300_000;

/** Function-table room past the indices handed out (`Gex#ftabSpare`): for a
 *  process with one Worker, and for one with threads. Room is only how soon
 *  a Worker must grow again: an index past a running Worker's table never
 *  enters the probe cache (cgen.rs `fcache_publish`), so a short table costs
 *  a `lookup`, never a trap. Every entry is held per Worker in the
 *  renderer's trusted range, which all its isolates share (1 GiB), so the
 *  room is kept small. */
const FTAB_SPARE_ALONE = 4096;
const FTAB_SPARE_THREADED = 8192;
// GEX_TEST_KEEP_PROBE (a test-only switch): a retirement leaves the probe
// cache's entries (cgen.rs `gex_test_keep_probe`).
const TEST_KEEP_PROBE = !!(globalThis.process?.env?.GEX_TEST_KEEP_PROBE);
/** A test-only switch: both rooms this small, so one unit's
 *  install outruns a sleeping thread's table. */
const ftabSpareTest = () => Number(globalThis.GEX_TEST_FTAB_SPARE ?? globalThis.process?.env?.GEX_TEST_FTAB_SPARE ?? 0);
const LIMIT64_BYTES = 16 * 2 ** 30;
/** The smallest module a Worker hands its siblings compiled (`Gex#offerModule`).
 *  Below it a sibling compiles the bytes itself: at most a few milliseconds. */
const RELAY_MIN_BYTES = 256 * 1024;
// Run switches a Worker takes at the top of its message as well as in its
// run options (worker.mjs's `run`), which an exec it adopts takes too.
const EXEC_SWITCHES = ['link', 'dense', 'fjit', 'fjitMax', 'fjitOnly', 'liftMax', 'inlineDispatch', 'xlateReport', 'discoverSkip', 't1Fold', 'providers', 'xcache'];
/** A value for an address-typed global: i64 (BigInt) in a wasm64 module. */
const addrVal = (w, v) => (w ? BigInt(v >>> 0) : v | 0);
/** An address a wasm program passed to a `gex` import: a BigInt from a
 *  memory64 program's (64-bit Wine's memory64 flavour, GEX_MEMORY64, i64 at its
 *  imports), a Number from a 32-bit or lowered one's. As a Number, and for
 *  a shim export taking a 64-bit address, as a BigInt. */
const addrNum = (v) => (typeof v === 'bigint' ? Number(v) : v >>> 0);
const addrBig = (v) => (typeof v === 'bigint' ? v : BigInt(v >>> 0));

/** A wasm32 shared library or program (cgen/dylink.rs) rewritten so its
 *  instance carries no function exports the page does not call by name:
 *  each one costs every Worker an export wrapper, its name and a slot in
 *  the exports object (KDE's libraries export about 150,000 functions and
 *  call one in eight by name), and its constant globals likewise. A table
 *  slot past the element segment the loader filled with an export (`X`)
 *  is filled by an added element segment instead, and an import of a
 *  dropped export comes from its slot or value (dylink.rs `E`). Kept: the
 *  `__` and `asyncify_` names, `_start` and `main`, and anything not a
 *  function or an immutable constant i32 global. `extra` is the binding list's
 *  [slot, export], `base` the module's `__table_base`. Null when the
 *  module is not of that shape (more than one table, fpcast thunks, no
 *  i32 `env.__table_base`, a construct this reader does not know). */
export function packExports(bytes, extra, base) {
  let p = 8;
  const leb = () => {
    let r = 0, sh = 0, b;
    do { b = bytes[p++]; r += (b & 0x7f) * 2 ** sh; sh += 7; } while (b & 0x80);
    return r;
  };
  const sleb = () => { let b; do { b = bytes[p++]; } while (b & 0x80); };
  // An ASCII name made here: TextDecoder takes each view as a garbage-collected
  // object outside the JS heap, which a Worker parked in Atomics.wait never
  // collects (a library's hundred thousand names were megabytes per Worker).
  const str = () => {
    const n = leb(), end = p + n;
    let v = '';
    for (let i = p; i < end; i++) {
      if (bytes[i] >= 0x80) { v = DECODER.decode(bytes.slice(p, end)); break; }
      v += String.fromCharCode(bytes[i]);
    }
    p = end;
    return v;
  };
  const VAL = new Set([0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f]);
  let globalImports = 0, tables = 0, tb = -1;
  const constGlobal = [];
  let exp = null, elem = null, after = -1;
  const secs = [];
  while (p < bytes.length) {
    const start = p;
    const id = bytes[p++];
    const size = leb();
    const body = p, end = p + size;
    secs.push([id, start, end]);
    if (id === 2) {
      const n = leb();
      for (let i = 0; i < n; i++) {
        const m = str(), f = str();
        const kind = bytes[p++];
        if (kind === 0) leb();
        else if (kind === 1) { p++; const fl = leb(); leb(); if (fl & 1) leb(); tables++; }
        else if (kind === 2) { const fl = leb(); leb(); if (fl & 1) leb(); }
        else if (kind === 3) {
          const vt = bytes[p++];
          p++;
          if (m === 'env' && f === '__table_base') { if (vt !== 0x7f) return null; tb = globalImports; }
          if (!VAL.has(vt)) return null;
          globalImports++;
        } else if (kind === 4) { p++; leb(); }
        else return null;
      }
    } else if (id === 4) {
      tables += leb();
    } else if (id === 6) {
      const n = leb();
      for (let i = 0; i < n; i++) {
        if (!VAL.has(bytes[p++])) return null;
        const mut = bytes[p++];
        const e0 = p;
        let ops = 0, plain = bytes[p] === 0x41;
        for (;;) {
          const op = bytes[p++];
          if (op === 0x0b) break;
          ops++;
          if (op === 0x41 || op === 0x42) sleb();
          else if (op === 0x23 || op === 0xd2) leb();
          else if (op === 0x43) p += 4;
          else if (op === 0x44) p += 8;
          else if (op === 0xd0) p++;
          else if (op === 0xfd && bytes[p] === 0x0c) p += 17;
          else if (!(op >= 0x6a && op <= 0x6c) && !(op >= 0x7c && op <= 0x7e)) return null;
        }
        constGlobal.push(!mut && plain && ops === 1 && p > e0);
      }
    } else if (id === 7) {
      exp = [];
      const n = leb();
      for (let i = 0; i < n; i++) {
        const a = p;
        const name = str();
        const kind = bytes[p++];
        const idx = leb();
        exp.push({ name, kind, idx, a, b: p });
      }
    } else if (id === 9) {
      elem = [body, end];
    }
    if (id === 7 || id === 8) after = secs.length;
    p = end;
  }
  if (tables !== 1 || tb < 0 || !exp || after < 0) return null;
  if (exp.some((x) => x.name.startsWith('__fpcast$'))) return null;
  const funcOf = new Map(exp.filter((x) => x.kind === 0).map((x) => [x.name, x.idx]));
  const keep = (x) => {
    if (x.kind === 0) return x.name.startsWith('__') || x.name.startsWith('asyncify_') || x.name === '_start' || x.name === 'main';
    if (x.kind === 3) return x.name.startsWith('__') || x.idx < globalImports || !constGlobal[x.idx - globalImports];
    return true;
  };
  // The X slots as runs of consecutive slots: one segment each.
  const runs = [];
  for (const [slot, name] of [...extra].sort((a, b) => a[0] - b[0])) {
    const f = funcOf.get(name);
    const rel = slot - base;
    if (f === undefined || rel < 0) return null;
    const last = runs.at(-1);
    if (last && last.rel + last.fs.length === rel) last.fs.push(f);
    else runs.push({ rel, fs: [f] });
  }
  const out = [];
  const u = (v) => { do { let b = v % 128; v = Math.floor(v / 128); if (v) b |= 0x80; out.push(b); } while (v); };
  const s32 = (v) => { for (;;) { const b = v & 0x7f; v >>= 7; if ((v === 0 && !(b & 0x40)) || (v === -1 && (b & 0x40))) { out.push(b); return; } out.push(b | 0x80); } };
  // Every section copied as it is, but for the exports (filtered) and
  // the element section (the runs added after the module's own segments).
  const parts = [bytes.subarray(0, 8)];
  const flush = () => { if (out.length) { parts.push(Uint8Array.from(out)); out.length = 0; } };
  const segs = () => {
    for (const r of runs) {
      out.push(0x00, 0x23);
      u(tb);
      if (r.rel) { out.push(0x41); s32(r.rel); out.push(0x6a); }
      out.push(0x0b);
      u(r.fs.length);
      for (const f of r.fs) u(f);
    }
  };
  const ours = () => { const at = out.length; segs(); return out.splice(at); };
  secs.forEach(([id, start, end], i) => {
    if (id === 7) {
      const kept = exp.filter(keep);
      const len = lebLen(kept.length) + kept.reduce((n, x) => n + x.b - x.a, 0);
      out.push(7); u(len); u(kept.length);
      flush();
      for (const x of kept) parts.push(bytes.subarray(x.a, x.b));
    } else if (id === 9 && runs.length) {
      p = elem[0];
      const n = leb();
      const rest = bytes.subarray(p, elem[1]);
      const add = ours();
      out.push(9); u(lebLen(n + runs.length) + rest.length + add.length); u(n + runs.length);
      flush();
      parts.push(rest, Uint8Array.from(add));
    } else {
      parts.push(bytes.subarray(start, end));
    }
    if (i + 1 === after && !elem && runs.length) {
      const add = ours();
      out.push(9); u(lebLen(runs.length) + add.length); u(runs.length);
      flush();
      parts.push(Uint8Array.from(add));
    }
  });
  flush();
  const total = parts.reduce((a, b) => a + b.length, 0);
  const r = new Uint8Array(total);
  let o = 0;
  for (const b of parts) { r.set(b, o); o += b.length; }
  return r;
}
const lebLen = (v) => { let n = 1; while (v >= 128) { v = Math.floor(v / 128); n++; } return n; };

/** An index into the function table: a table64 (BigInt indices) beside a 64-bit memory. */
const tIdx = (w, v) => (w ? BigInt(v) : v);

/**
 * Decode UTF-8 out of guest memory.
 *
 * `TextDecoder.decode` refuses a `SharedArrayBuffer`-backed view, so a shared
 * memory needs the `slice` copy; it is only paid when the memory is shared.
 */
const DECODER = new TextDecoder();
/** Byte, word and DataView views of all of `buf`: what a door call reads a
 *  client's request through. The word view's length is computed without
 *  `>>> 2`, which wraps for a 64-bit memory past 4 GiB (a memory of 4 GiB +
 *  64 KiB gave a view of 16384 words). */
export function memoryViews(buf) {
  return { u8: new Uint8Array(buf), i32: new Int32Array(buf, 0, Math.floor(buf.byteLength / 4)), dv: new DataView(buf) };
}

export function utf8(view) {
  return DECODER.decode(isShared(view.buffer) ? view.slice() : view);
}

/** Whether `buf` is a SharedArrayBuffer -- asked without naming the global,
 *  which a page that is not cross-origin isolated does not have. */
const SAB = globalThis.SharedArrayBuffer;
function isShared(buf) {
  return typeof SAB === 'function' && buf instanceof SAB;
}

// Registry indices describe a shared queue, not module identity: two Workers
// can drain and publish that queue in different turns. A compiled Module may
// only be reused for byte-identical input, or its element segments install
// unrelated functions into otherwise-correct dispatch slots. Two independent
// 32-bit accumulators plus the exact length make a cheap content key; this is
// validation against an index race, not a persistent content-addressed cache.
/** The most the module store (`Gex#moduleStore`) may grow to, tried in turn:
 *  the first a runtime accepts. 4 GiB filled up in 22 minutes of a game
 *  on Unity's own Mono (the translations of Mono's JIT output), after which
 *  the registry kept its bytes in the shim's heap again and that ran out.
 *  A growable SharedArrayBuffer only commits what it has grown to. */
const MODULE_STORE_MAX = [32, 16, 8, 4].map((g) => g * 1024 * 1024 * 1024);
/** How many wasm32 modules a program installs before its first thread is
 *  given a Worker ahead (`#offerSpare`): below it, instantiating them again
 *  in the thread's Worker is short. */
const SPARE_MODULES = 20;
/** How often a waiting spare looks for new installs: at first, and at most
 *  apart once nothing has come for a while. */
const SPARE_POLL_MS = 20;
const SPARE_POLL_MAX_MS = 1000;
/** How long a spare waits for a sibling's relay of a native module before
 *  compiling it itself (`Gex#relayAwaited`). */
const RELAY_WAIT_MS = 250;
/** A test-only switch: the threshold itself, so a small program gets one. */
const spareModules = () => Number(globalThis.GEX_TEST_SPARE_MODULES ?? globalThis.process?.env?.GEX_TEST_SPARE_MODULES ?? 0) || SPARE_MODULES;

/** Where a module store's first module goes: after the bump cursor (eight
 *  bytes) and the registry epoch it was made at (four, then padding). */
const STORE_HEAD = 16;
/** A new module store for the registry of `epoch` (`gex_registry_epoch`):
 *  a growable SharedArrayBuffer whose first eight bytes are the bump cursor
 *  and the next four the epoch. Null where SharedArrayBuffer cannot grow. */
function makeModuleStore(epoch = 0) {
  if (typeof SharedArrayBuffer === 'undefined' || typeof SharedArrayBuffer.prototype.grow !== 'function') return null;
  for (const max of MODULE_STORE_MAX) {
    try {
      const sab = new SharedArrayBuffer(1 << 20, { maxByteLength: max });
      new Int32Array(sab, 8, 1)[0] = epoch;
      Atomics.store(new BigInt64Array(sab, 0, 1), 0, BigInt(STORE_HEAD));
      if (globalThis.GEX_REGISTRY_TRACE || globalThis.process?.env?.GEX_REGISTRY_TRACE) {
        globalThis.process?.getBuiltinModule?.('fs')?.writeSync(2, `[gex] module store max ${max / 2 ** 30} GiB\n`);
      }
      return sab;
    } catch {
      // Too big for this runtime: the next.
    }
  }
  return null;
}
let storeFullSaid = false;
/** Append `bytes` to a module store: their offset, or -1 when it is full.
 *  Any Worker of the process may append at once: the cursor is atomic and a
 *  grow that loses a race to a bigger one is simply retried. */
export function storeAppend(sab, bytes) {
  const len = bytes.length;
  const off = Number(Atomics.add(new BigInt64Array(sab, 0, 1), 0, BigInt((len + 7) & ~7)));
  const end = off + len;
  if (end > sab.maxByteLength) {
    // Said once: from here the registry keeps bytes in the shim's heap, which
    // shares the guest's 2 GiB arena and will run out in a long run.
    if (!storeFullSaid) {
      storeFullSaid = true;
      console.error(`[gex] module store full at ${(sab.maxByteLength / 2 ** 30).toFixed(0)} GiB: the translated-module registry keeps its bytes in the shim heap from here`);
    }
    return -1;
  }
  // A grow throws both when a sibling grew past it meanwhile (a lost race, so
  // retried) and when the engine will give no more (out of memory). The second
  // used to spin here for good, every Worker of the process that appended
  // after it with it. Doubling first, then exactly what this needs, then the
  // registry keeps the bytes itself (-1), as it does when the store is full.
  for (let tries = 0; sab.byteLength < end; tries++) {
    const to = tries === 0 ? Math.min(sab.maxByteLength, Math.max(end, sab.byteLength * 2)) : end;
    try {
      sab.grow(to);
    } catch (e) {
      if (sab.byteLength >= end) break;
      if (tries >= 1) {
        if (!storeFullSaid) {
          storeFullSaid = true;
          console.error(`[gex] module store would not grow to ${(end / 1048576).toFixed(0)} MiB (${e?.message ?? e}): the translated-module registry keeps its bytes in the shim heap from here`);
        }
        return -1;
      }
    }
  }
  new Uint8Array(sab, off, len).set(bytes);
  return off;
}
function moduleContentKey(bytes) {
  if (!(bytes instanceof Uint8Array)) return null;
  // A little-endian word at a time, then the tail bytes (the shim's
  // `content_key` is the same function): a byte at a time was 0.65 s of a
  // node start-up per Worker. Bytes in a shared memory too: a registry
  // module's are written before the row naming them and never again, and
  // read a byte at a time they held a dispatch waiting on its module 20 ms
  // per 500 KB.
  let a = 0x811c9dc5;
  let b = 0x9e3779b9;
  const n = bytes.length >>> 2;
  const words = (bytes.byteOffset & 3) === 0
    ? new Uint32Array(bytes.buffer, bytes.byteOffset, n)
    : null;
  for (let i = 0; i < n; i++) {
    const x = words ? words[i]
      : (bytes[4 * i] | (bytes[4 * i + 1] << 8) | (bytes[4 * i + 2] << 16) | (bytes[4 * i + 3] << 24)) >>> 0;
    a = Math.imul(a ^ x, 0x01000193);
    b = Math.imul(b ^ ((x + i) | 0), 0x85ebca6b);
    b ^= b >>> 13;
  }
  for (let i = 4 * n; i < bytes.length; i++) {
    const x = bytes[i];
    a = Math.imul(a ^ x, 0x01000193);
    b = Math.imul(b ^ (x + (n + i - 4 * n)), 0x85ebca6b);
    b ^= b >>> 13;
  }
  return `${bytes.length}:${a >>> 0}:${b >>> 0}`;
}
const GUEST_INITIAL_BYTES = 0x07000000;
// The shim's stack sits at the bottom of its region, below its data.
const SHIM_STACK_TOP = 0x06000000;
/** `gex_emit::dense::FIRST_TABLE`: dispatch tables from here on may be dense. */
const DENSE_FIRST_TABLE = 4;
/** The most a dispatch table grows past what is needed, in entries. */
const TABLE_STEP = 1 << 20;
/** `gex_abi::ctx::LAST_TARGET`. */
const CTX_LAST_TARGET = 44 * 8;
/** `gex_abi::ctx::PC`. */
const CTX_PC = 32 * 8;
/** x86's `rsp` (`ctx::x(4)`): the stack pointer's place in the context
 *  where the shim does not say (`gex_ctx_sp`); aarch64's is `ctx::x(31)`. */
const CTX_SP = 4 * 8;
/** cgen.rs `CALLOUT_UNWOUND`: 64 KiB. */
const CALLOUT_UNWOUND = 64n << 10n;
/** gex-kernel `tramp::NATIVE_RETURN_PAD` and the native host's
 *  `NATIVE_UNWOUND`. */
const NATIVE_RETURN_PAD = 0x3ffff1n;
/** The door area's user shared data state (+144, the server's door.c):
 *  "USD0", no process has filled the page; "USD1", one has. */
const USD_UNPUBLISHED = 0x30445355;
const USD_PUBLISHED = 0x31445355;
const NATIVE_UNWOUND = 512n;
/** Where a native caller's `run_until` goes after a stray return to its
 *  pad (cgen.rs `STRAY_PC`): the pad's filler, which no code has. */
const STRAY_PC = NATIVE_RETURN_PAD + 1n;
/** gex-kernel `tramp::WOW64_STOP_SYSCALL`, `WOW64_STOP_UNIX` and
 *  `WOW64_RETURN_MARK`: where WoW64's 32-bit code reaches 64-bit code. A
 *  call-out waiting for the first returns at either of the first two
 *  (`tramp::returns_at`); one waiting for a stop or the mark takes it with
 *  the stack anywhere (`tramp::any_stack`). */
const WOW64_STOP_SYSCALL = 0x3ffff4n;
const WOW64_STOP_UNIX = 0x3ffff7n;
const WOW64_RETURN_MARK = 0x3ffffan;
const returnsAt = (ret, pc) => pc === ret || (ret === WOW64_STOP_SYSCALL && pc === WOW64_STOP_UNIX);
/** A thread `clone` made that could not be handed to the host. */
function threadNotStarted(who, err) {
  console.error(`[${who}] starting a thread: ${err?.message ?? err}`);
}

/**
 * `table.grow`, said out loud once a table passes a million entries: each
 * Worker's dispatch tables are V8 heap, and their size is the renderer's.
 */
/** `gex_set_dense`'s argument (hosts/browser/src/dense.rs): 0 off, 1 on,
 *  2 on without catching up. */
function denseArg(o, on) {
  return on ? (o?.dense === 2 ? 2 : 1) : 0;
}

// `blocks` holds entries (gex_emit::dense): each entry
// is `(ctx, arm) -> ()`. A function of the block type `(ctx) -> ()` that
// must sit in `blocks` (the miss stub, a bridge) gets a wrapper that drops
// the arm.
// (module (type (func (param i32))) (type (func (param i32 i32)))
//   (import "w" "f" (func (type 0)))
//   (func (export "e") (type 1) local.get 0 return_call 0))
const ENTRY_WRAP = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 10, 2, 96, 1, 127, 0, 96, 2, 127, 127, 0,
  2, 7, 1, 1, 119, 1, 102, 0, 0,
  3, 2, 1, 1,
  7, 5, 1, 1, 101, 0, 1,
  10, 8, 1, 6, 0, 32, 0, 18, 0, 11,
]);
// `GEX_LOG=bridges`: a unit bridge that tells `count` the context, then goes
// on to the bridge.
// (module (type (func (param i32)))
//   (import "w" "count" (func (type 0))) (import "w" "real" (func (type 0)))
//   (func (export "c") (type 0) local.get 0 call 0 local.get 0 return_call 1))
const BRIDGE_COUNT = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 96, 1, 127, 0,
  2, 20, 2, 1, 119, 5, 99, 111, 117, 110, 116, 0, 0, 1, 119, 4, 114, 101, 97, 108, 0, 0,
  3, 2, 1, 0,
  7, 5, 1, 1, 99, 0, 2,
  10, 12, 1, 10, 0, 32, 0, 16, 0, 32, 0, 18, 1, 11,
]);
// A watched arm of an entry: arm `arm` calls `hit` as the block-type watch
// stub does (returns when it says so, else traps), every other arm goes on
// to `old`.
// (module (type (func (param i32 i32))) (type (func (param i32) (result i32)))
//   (import "w" "old" (func (type 0))) (import "w" "hit" (func (type 1)))
//   (import "w" "arm" (global i32))
//   (func (export "c") (type 0)
//     local.get 1 global.get 0 i32.eq
//     if local.get 0 call 1 if return end unreachable end
//     local.get 0 local.get 1 return_call 0))
const ENTRY_WATCH = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 11, 2, 96, 2, 127, 127, 0, 96, 1, 127, 1, 127,
  2, 26, 3, 1, 119, 3, 111, 108, 100, 0, 0, 1, 119, 3, 104, 105, 116, 0, 1, 1, 119, 3, 97, 114, 109, 3, 127, 0,
  3, 2, 1, 0,
  7, 5, 1, 1, 99, 0, 2,
  10, 27, 1, 25, 0, 32, 1, 35, 0, 70, 4, 64, 32, 0, 16, 1, 4, 64, 15, 11, 0, 11, 32, 0, 32, 1, 18, 0, 11,
]);
let entryWrapMod = null, entryWatchMod = null;

// 64-bit Wine's direct entry in wasm, so a unit's system or Unix call into Wine
// (cgen.rs `wine64_entry`) costs no JS. The shim imports `entry_call` from the
// trampoline, made with the shim; it tail-calls slot 0 of its `hook`
// table, the entry function once the Worker has made it (`#entryFn`, when
// both the function table and Wine's stack pointer exist) and the
// trampoline's own `slow` (to JS, which makes it) until then.
// (module (type $e (func (param i32 i32 i64) (result i32)))
//   (import "e" "hook" (table 1 funcref)) (import "e" "slow" (func $slow (type $e)))
//   (elem (i32.const 0) func $s)
//   (func $s (export "slow") (type $e) local.get 0 local.get 1 local.get 2 return_call $slow)
//   (func (export "entry_call") (type $e)
//     local.get 0 local.get 1 local.get 2 i32.const 0 return_call_indirect (type $e) 0))
const ENTRY_TRAMP = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 8, 1, 96, 3, 127, 127, 126, 1, 127,
  2, 21, 2, 1, 101, 4, 104, 111, 111, 107, 1, 112, 0, 1, 1, 101, 4, 115, 108, 111, 119, 0, 0,
  3, 3, 2, 0, 0,
  7, 21, 2, 4, 115, 108, 111, 119, 0, 1, 10, 101, 110, 116, 114, 121, 95, 99, 97, 108, 108, 0, 2,
  9, 7, 1, 0, 65, 0, 11, 1, 1,
  10, 26, 2, 10, 0, 32, 0, 32, 1, 32, 2, 18, 0, 11, 13, 0, 32, 0, 32, 1, 32, 2, 65, 0, 19, 0, 0, 11,
]);
/** Whether this thread's stack has room for one more fault's dispatch, its
 *  shim, hook and guest frames. A call that pushes STACK_ROOM's entries as
 *  arguments needs that much stack below the caller's frame, and throws a
 *  RangeError when it is not there, which is caught here: nothing the guest
 *  or the shim was doing is unwound. */
const STACK_ROOM = new Array(32768).fill(0);
/** Call-outs open past which a new one asks `stackRoom` first (each asks
 *  at the cost of a call that large), and the room it asks for: a level's
 *  frames and what the next check's way there may take, less than a fault's
 *  dispatch (a delivery in place asks for STACK_ROOM's itself). */
const CALLOUT_ROOM_DEPTH = 32;
const CALLOUT_ROOM = new Array(12288).fill(0);
const stackSink = function () { return 1; };
function stackRoom(room = STACK_ROOM) {
  try {
    return Reflect.apply(stackSink, null, room);
  } catch (err) {
    if (err instanceof RangeError) return 0;
    throw err;
  }
}
let entryTrampMod = null;
/** A JS call into wasm takes the engine's generic wrapper until that export
 *  has been called often enough to get one made for its signature (V8: a
 *  thousand calls), and the generic one holds about 700 bytes of the host's
 *  stack under every call: each call-out level keeps one open, under the
 *  guest code it runs. The run loops' `enter` goes through a front of its own
 *  (`enterFront`), which a call with a null context leaves at once:
 *  (module (type $e (func (param i32 i64))) (import "e" "f" (func $f (type $e)))
 *    (func (export "enter") (type $e)
 *      local.get 0 i32.eqz if return end
 *      local.get 0 local.get 1 return_call $f)) */
const ENTER_FRONT = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 6, 1, 0x60, 2, 0x7f, 0x7e, 0,
  2, 7, 1, 1, 101, 1, 102, 0, 0,
  3, 2, 1, 0,
  7, 9, 1, 5, 101, 110, 116, 101, 114, 0, 1,
  10, 17, 1, 15, 0, 0x20, 0, 0x45, 0x04, 0x40, 0x0f, 0x0b, 0x20, 0, 0x20, 1, 0x12, 0, 0x0b,
]);
/** The other way, a wasm call into JS takes a generic wrapper of the same
 *  kind until that import has been called as often, about 250 bytes under
 *  every call-out level: the shim's and every module's `call_out` goes
 *  through a front too (`callOutFront`), whose import a call with a null
 *  context reaches, which `fn` answers with 0 at once:
 *  (module (type $c (func (param i32 i64 i64 i64) (result i32)))
 *    (import "e" "f" (func $f (type $c)))
 *    (func (export "call_out") (type $c)
 *      local.get 0 local.get 1 local.get 2 local.get 3 return_call $f)) */
const CALLOUT_FRONT = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 9, 1, 0x60, 4, 0x7f, 0x7e, 0x7e, 0x7e, 1, 0x7f,
  2, 7, 1, 1, 101, 1, 102, 0, 0,
  3, 2, 1, 0,
  7, 12, 1, 8, 99, 97, 108, 108, 95, 111, 117, 116, 0, 1,
  10, 14, 1, 12, 0, 0x20, 0, 0x20, 1, 0x20, 2, 0x20, 3, 0x12, 0, 0x0b,
]);
/** The fronts are warmed (called that often with a null context, about a
 *  millisecond for the pair) only once call-outs nest this deep, which few
 *  processes do: the levels below keep the generic wrappers.
 *  `warmFronts` warms those not yet warmed. */
const FRONT_WARM_DEPTH = 4;
const FRONT_WARM_CALLS = 1100;
// Held weakly: a front holds its process's instance (and so its memory),
// and in the single-thread build every process's is in this one realm. A
// front whose process is gone has nothing left to warm.
const coldFronts = [];
function warmFronts() {
  for (const [ref, args] of coldFronts.splice(0)) {
    const f = ref.deref();
    if (f) for (let i = 0; i < FRONT_WARM_CALLS; i++) f(...args);
  }
}
let enterFrontMod = null;
const enterFronts = new WeakMap();
function enterFront(fn) {
  let w = enterFronts.get(fn);
  if (w) return w;
  try {
    enterFrontMod ??= new WebAssembly.Module(ENTER_FRONT);
    const f = new WebAssembly.Instance(enterFrontMod, { e: { f: fn } }).exports.enter;
    coldFronts.push([new WeakRef(f), [0, 0n]]);
    w = f;
  } catch (err) {
    // An `enter` of another signature: called as it is.
    if (!(err instanceof WebAssembly.LinkError)) throw err;
    w = fn;
  }
  enterFronts.set(fn, w);
  return w;
}
let callOutFrontMod = null;
function callOutFront(fn) {
  callOutFrontMod ??= new WebAssembly.Module(CALLOUT_FRONT);
  const f = new WebAssembly.Instance(callOutFrontMod, { e: { f: fn } }).exports.call_out;
  coldFronts.push([new WeakRef(f), [0, 0n, 0n, 0n]]);
  return f;
}
/** The entry function's module, for a 64-bit function table (`t64`) and an
 *  i64 stack pointer (`s64`), compiled once each.
 *  (module (type $vi (func (param i32))) (type $e (func (param i32 i32 i64) (result i32)))
 *    (import "e" "t" (table T 0 funcref)) (import "e" "sp" (global $sp (mut S)))
 *    (func (export "entry") (type $e) (local $cur S)
 *      ;; Wine's C frames below the guest's stack pointer, as `fault_call`'s:
 *      ;; sp = (min(cur, rsp) - 4096) & ~15 (cur 0: rsp, for an i64 one)
 *      global.get $sp local.set $cur  ...  global.set $sp
 *      try (call_indirect (type $vi) (local.get 1) (local.get 0))
 *      catch_all (global.set $sp (local.get $cur)) rethrow 0 end
 *      (global.set $sp (local.get $cur)) i32.const 1))
 *  The saved stack pointer is this call's local, so each nesting level
 *  restores its own. A trap (`unreachable`: the shim's trap out) is not
 *  caught here: the JS loop that catches it puts the pointer back
 *  (`#progSpAt`). */
/** A binding list (cgen/dylink.rs's format) split into the lines that are
 *  the same in every process binding the module alike, and its numbers
 *  (`V`, strings), which are not. */
function bindSplit(text) {
  const i = text.startsWith('V\t') ? 0 : text.lastIndexOf('\nV\t') + 1;
  if (!i && !text.startsWith('V\t')) return [text, []];
  return [text.slice(0, i), text.slice(i + 2, text.endsWith('\n') ? -1 : undefined).split(',')];
}
/** A binding list's lines, parsed: a plan of records of `PLAN_REC` numbers
 *  each (`ops`: kind; module and name as offsets and lengths into the plan's
 *  names, `names`, UTF-8; then the kind's own: the index of its number(s) in
 *  the list's `V`, flags, a further name), in the lines' order. Kinds:
 *  imports to bind (BIND_ASYNC .. BIND_MISSING), late calls (BIND_LATE), the
 *  table slots to fill after instantiation (BIND_EXTRA) and the module's
 *  thread-local block (BIND_TLS).
 *
 *  The lines are the same in every process binding a module alike, and a
 *  boot binds the same few hundred shapes ten thousand times (KDE: 492 of
 *  them, 22 MB, parsed 10351 times, 6.1 s of its Workers' time). So a plan
 *  is made once for the page: its two arrays are on shared buffers, kept
 *  by the shape's key (`plan:`, cgen/dylink.rs `gex_wasmprog_bind_key`) and
 *  handed to Workers as compiled modules are; a Worker that has it reads
 *  only the list's numbers. The single-thread build's processes all run in
 *  one realm: there plans are kept by their lines (`BIND_PLANS`, up to
 *  `BIND_PLAN_BYTES` of them), and a late module is made once for the same
 *  slots (`LATE_MODS`, up to `LATE_MODS_MAX`). */
const BIND_ASYNC = 0, BIND_GLOBAL = 1, BIND_EXPORT = 2, BIND_SLOT = 3, BIND_MISSING = 4, BIND_LATE = 5, BIND_EXTRA = 6, BIND_TLS = 7;
const PLAN_REC = 10;
const BIND_PLANS = new Map(), BIND_PLAN_BYTES = 64 << 20, LATE_MODS = new Map(), LATE_MODS_MAX = 4096;
let bindPlanBytes = 0;
const planEncoder = new TextEncoder(), planDecoder = new TextDecoder();
/** `n` bytes (a multiple of 4) for a plan: from a shared arena where the
 *  page can share one (a typed array on it crosses `postMessage` by
 *  reference), carved so that a plan costs no buffer of its own (one each
 *  was most of a plan's making). [buffer, offset]. */
const PLAN_ARENA = 4 << 20;
let planArena = null, planArenaAt = 0;
function planAlloc(n) {
  const share = typeof SharedArrayBuffer === 'function' && globalThis.crossOriginIsolated !== false;
  if (!share || n > PLAN_ARENA / 4) return [share ? new SharedArrayBuffer(n) : new ArrayBuffer(n), 0];
  if (!planArena || planArenaAt + n > PLAN_ARENA) {
    planArena = new SharedArrayBuffer(PLAN_ARENA);
    planArenaAt = 0;
  }
  const at = planArenaAt;
  planArenaAt += n;
  return [planArena, at];
}
/** Whether `x` is a plan (`bindPlan`), as the page relays them. */
const isPlan = (x) => x?.ops instanceof Int32Array && x.names instanceof Uint8Array;
/** `shape`'s plan, and its names as a string: `{ plan, str }`. */
function bindPlan(shape) {
  const shared = SP_CODE && single.on;
  const had = shared ? BIND_PLANS.get(shape) : undefined;
  if (had) return had;
  const lines = shape.split('\n');
  // records made here, copied to the plan's buffer once their count is known
  const ops = new Int32Array(lines.length * PLAN_REC);
  const parts = [];
  let len = 0, n = 0;
  const at = (str) => {
    const o = len;
    parts.push(str);
    len += str.length;
    return o;
  };
  const rec = (kind, m, name, a = 0, b = 0, c = 0, d = 0, e = 0) => {
    ops[n] = kind; ops[n + 1] = at(m); ops[n + 2] = m.length; ops[n + 3] = at(name); ops[n + 4] = name.length;
    ops[n + 5] = a; ops[n + 6] = b; ops[n + 7] = c; ops[n + 8] = d; ops[n + 9] = e;
    n += PLAN_REC;
  };
  let v = 0;
  for (const line of lines) {
    if (!line) continue;
    const f = line.split('\t');
    switch (f[0]) {
      case 'S':
        if (f[1] === 'env' && (f[2] === '__asyncify_state' || f[2] === '__asyncify_data')) rec(BIND_ASYNC, 'env', f[2], f[2] === '__asyncify_state' ? 0 : 1);
        break;
      case 'G':
        rec(BIND_GLOBAL, f[1], f[2], f[5] === '64' ? 1 : 0, f[4] === '1' ? 1 : 0, v++);
        break;
      case 'E': {
        const mod = v++;
        // a packed module's export: from its slot (a number) or a value
        const [pk, pv] = f[5] === 's' ? [1, v++] : f[5] && f[5] !== '-' ? [2, Number(f[5].slice(1)) | 0] : [0, 0];
        rec(BIND_EXPORT, f[1], f[2], mod, at(f[4]), f[4].length, pk, pv);
        break;
      }
      case 'L':
        if (f[4]) rec(BIND_LATE, f[1], f[2], v++, at(f[4]), f[4].length);
        else rec(BIND_SLOT, f[1], f[2], v++);
        break;
      case 'M': {
        const msg = f.slice(3).join('\t');
        rec(BIND_MISSING, f[1], f[2], 0, at(msg), msg.length);
        break;
      }
      case 'X':
        rec(BIND_EXTRA, '', f[2], v++);
        break;
      case 'T':
        rec(BIND_TLS, '', '', Number(f[1]), Number(f[2]));
        break;
    }
  }
  const str = parts.join('');
  const bytes = planEncoder.encode(str);
  const [ob, oo] = planAlloc(n * 4), [nb, no] = planAlloc((bytes.length + 3) & ~3);
  const plan = { ops: new Int32Array(ob, oo, n), names: new Uint8Array(nb, no, bytes.length) };
  plan.ops.set(ops.subarray(0, n));
  plan.names.set(bytes);
  const out = { plan, str };
  if (shared && bindPlanBytes + shape.length <= BIND_PLAN_BYTES) {
    BIND_PLANS.set(shape, out);
    bindPlanBytes += shape.length;
  }
  return out;
}
/** A plan's names, as a string (`names` is shared: copied out first). */
const planNames = (plan) => planDecoder.decode(plan.names.slice());
/** A plan's `X` slots: [slot, export]. */
function bindExtra({ plan, str, vals }) {
  const out = [];
  const o = plan.ops;
  for (let i = 0; i < o.length; i += PLAN_REC) {
    if (o[i] === BIND_EXTRA) out.push([Number(vals[o[i + 5]]), str.slice(o[i + 3], o[i + 3] + o[i + 4])]);
  }
  return out;
}

/** A wasm32 module's late-bound functions (`L` in a binding list, each
 *  [mod, name, slot, sig]): a module of one function per binding, exported
 *  by its index, that calls table `e.t`'s slot with its own arguments.
 *  (func (export "N") (type sig) local.get 0 .. local.get n-1
 *    (i32|i64).const slot call_indirect (type sig) 0)
 *  `t64`: a 64-bit table. */
function lateCalls(late, t64) {
  const uleb = (v, o) => { do { let b = v % 128; v = Math.floor(v / 128); if (v) b |= 128; o.push(b); } while (v); return o; };
  const sleb = (v, o) => { for (;;) { const b = v & 127; v = Math.floor(v / 128); if ((v === 0 && !(b & 64)) || (v === -1 && (b & 64))) { o.push(b); return o; } o.push(b | 128); } };
  const vt = { i: 0x7f, j: 0x7e, f: 0x7d, d: 0x7c };
  const sec = (id, b) => [id, ...uleb(b.length, []), ...b];
  const vec = (items) => [...uleb(items.length, []), ...items.flat()];
  const sigs = [...new Set(late.map((l) => l[3]))], at = new Map(sigs.map((g, i) => [g, i]));
  const types = sigs.map((g) => { const [p, r] = g.split('_'); return [0x60, ...vec([...p].map((c) => [vt[c]])), ...vec([...r].map((c) => [vt[c]]))]; });
  const enc = new TextEncoder();
  const exps = late.map((_, i) => { const n = enc.encode(String(i)); return [...uleb(n.length, []), ...n, 0, ...uleb(i, [])]; });
  const bodies = late.map(([, , slot, g]) => {
    const n = g.split('_')[0].length, b = [0];
    for (let k = 0; k < n; k++) b.push(0x20, ...uleb(k, []));
    b.push(t64 ? 0x42 : 0x41, ...sleb(t64 ? slot : slot | 0, []), 0x11, ...uleb(at.get(g), []), 0, 0x0b);
    return [...uleb(b.length, []), ...b];
  });
  return new WebAssembly.Module(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...sec(1, vec(types)),
    ...sec(2, [1, 1, 101, 1, 116, 1, 0x70, t64 ? 4 : 0, 0]),
    ...sec(3, vec(late.map((l) => uleb(at.get(l[3]), [])))),
    ...sec(7, vec(exps)),
    ...sec(10, vec(bodies)),
  ]));
}

const entryFnMods = new Map();
function entryFnModule(t64, s64) {
  const k = (t64 ? 2 : 0) | (s64 ? 1 : 0);
  let m = entryFnMods.get(k);
  if (m) return m;
  const S = s64 ? 0x7e : 0x7f;
  const head = s64
    // local.get cur, local.get rsp, cur != 0 && cur <u rsp, select; - 4096; & -16
    ? [0x20, 3, 0x20, 2, 0x20, 3, 0x50, 0x45, 0x20, 3, 0x20, 2, 0x54, 0x71, 0x1b, 0x42, 0x80, 0x20, 0x7d, 0x42, 0x70, 0x83]
    // min_u(cur, esp) - 4096 & -16, esp the low half of rsp
    : [0x20, 3, 0x20, 2, 0xa7, 0x20, 3, 0x20, 2, 0xa7, 0x49, 0x1b, 0x41, 0x80, 0x20, 0x6b, 0x41, 0x70, 0x71];
  const body = [
    1, 1, S,
    0x23, 0, 0x21, 3,
    ...head, 0x24, 0,
    0x06, 0x40, 0x20, 1, 0x20, 0, ...(t64 ? [0xad] : []), 0x11, 0, 0,
    0x19, 0x20, 3, 0x24, 0, 0x09, 0, 0x0b,
    0x20, 3, 0x24, 0, 0x41, 1, 0x0b,
  ];
  const sec = (id, b) => [id, b.length, ...b];
  const bytes = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...sec(1, [2, 96, 1, 127, 0, 96, 3, 127, 127, 126, 1, 127]),
    ...sec(2, [2, 1, 101, 1, 116, 1, 112, t64 ? 4 : 0, 0, 1, 101, 2, 115, 112, 3, S, 1]),
    ...sec(3, [1, 1]),
    ...sec(7, [1, 5, 101, 110, 116, 114, 121, 0, 0]),
    ...sec(10, [1, body.length, ...body]),
  ]);
  m = new WebAssembly.Module(bytes);
  entryFnMods.set(k, m);
  return m;
}

/** The snapshot flavour's entry function (a program built against the
 *  snapshot sysroot, libgex-snap/entry64.c): `entryFnModule`'s, its depth
 *  counted in `d` (a session snapshot stops a thread only in the outermost
 *  entry), and the program's asyncify state looked at after the call: an
 *  unwind the snapshot began (1) ends here, `af(idx, ctx, rsp)` keeps the record,
 *  parks and starts the rewind, and the call is made again (rewound).
 *  (module (type $vi (func (param i32))) (type $e (func (param i32 i32 i64) (result i32)))
 *    (type $st (func (result i32))) (type $af (func (param i32 i32 i64)))
 *    (import "e" "t" (table T 0 funcref)) (import "e" "sp" (global $sp (mut S)))
 *    (import "e" "d" (global $d (mut i32)))
 *    (import "e" "st" (func $st (type $st))) (import "e" "af" (func $af (type $af)))
 *    (func (export "entry") (type $e) (local $cur S)
 *      ... sp as entryFnModule's ...  d += 1
 *      loop $again
 *        try (call_indirect (type $vi) (local.get 1) (local.get 0))
 *        catch_all d -= 1 (global.set $sp (local.get $cur)) rethrow 0 end
 *        (if (i32.eq (call $st) (i32.const 1)) (then (call $af (local.get 0) (local.get 1) (local.get 2)) (br $again)))
 *      end
 *      d -= 1 (global.set $sp (local.get $cur)) i32.const 1)) */
const entrySnapMods = new Map();
/** GEX_SNAP_TRACE: each thread a session snapshot cannot stop says why, once. */
const SNAP_TRACE = !!globalThis.process?.env?.GEX_SNAP_TRACE;
/** A snapshot record's levels (libgex-snap/snap.h `nlv`, `lv`): the
 *  count's offset, the first level's and a level's size; and how many. */
const SNAP_NLV = 24, SNAP_LV = 32, SNAP_LVS = 40, SNAP_LEVELS = 16;
// A record's upper half where Wine's PE module of the snapshot flavour runs
// in the thread (`#snapPeWin`): the PE levels' calls and the per-thread state
// at their call-outs (a slot each), then its asyncify window, up to the
// restored thread's stack (snap.h SNAP_SIZE, SNAP_STACK).
const SNAP_HALF = 0x20000, SNAP_PE_SLOT = 128, SNAP_PE_HDR = SNAP_LEVELS * SNAP_PE_SLOT, SNAP_PE_END = 0x40000 - 16384;
/** Thrown by a snapshot's stop inside guest code a call-out runs, to that
 *  call-out (`#snapCallOut`). */
const SNAP_NEST = Object.freeze({ snapNest: true });
function entryFnSnapModule(t64, s64) {
  const k = (t64 ? 2 : 0) | (s64 ? 1 : 0);
  let m = entrySnapMods.get(k);
  if (m) return m;
  const S = s64 ? 0x7e : 0x7f;
  const head = s64
    ? [0x20, 3, 0x20, 2, 0x20, 3, 0x50, 0x45, 0x20, 3, 0x20, 2, 0x54, 0x71, 0x1b, 0x42, 0x80, 0x20, 0x7d, 0x42, 0x70, 0x83]
    : [0x20, 3, 0x20, 2, 0xa7, 0x20, 3, 0x20, 2, 0xa7, 0x49, 0x1b, 0x41, 0x80, 0x20, 0x6b, 0x41, 0x70, 0x71];
  const dDec = [0x23, 1, 0x41, 1, 0x6b, 0x24, 1];
  const body = [
    1, 1, S,
    0x23, 0, 0x21, 3,
    ...head, 0x24, 0,
    0x23, 1, 0x41, 1, 0x6a, 0x24, 1,
    0x03, 0x40,
    0x06, 0x40, 0x20, 1, 0x20, 0, ...(t64 ? [0xad] : []), 0x11, 0, 0,
    0x19, ...dDec, 0x20, 3, 0x24, 0, 0x09, 0, 0x0b,
    0x10, 0, 0x41, 1, 0x46, 0x04, 0x40, 0x20, 0, 0x20, 1, 0x20, 2, 0x10, 1, 0x0c, 1, 0x0b,
    0x0b,
    ...dDec, 0x20, 3, 0x24, 0, 0x41, 1, 0x0b,
  ];
  const sec = (id, b) => [id, b.length, ...b];
  const imps = [5,
    1, 101, 1, 116, 1, 112, t64 ? 4 : 0, 0,
    1, 101, 2, 115, 112, 3, S, 1,
    1, 101, 1, 100, 3, 0x7f, 1,
    1, 101, 2, 115, 116, 0, 2,
    1, 101, 2, 97, 102, 0, 3];
  const bytes = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...sec(1, [4, 96, 1, 127, 0, 96, 3, 127, 127, 126, 1, 127, 96, 0, 1, 127, 96, 3, 127, 127, 126, 0]),
    ...sec(2, imps),
    ...sec(3, [1, 1]),
    ...sec(7, [1, 5, 101, 110, 116, 114, 121, 0, 2]),
    ...sec(10, [1, body.length, ...body]),
  ]);
  m = new WebAssembly.Module(bytes);
  entrySnapMods.set(k, m);
  return m;
}

function growTable(table, delta, init, which) {
  const to = table.length + delta;
  if (to > (1 << 20)) console.warn(`[gex] dispatch table ${which} grows ${table.length} -> ${to} entries`);
  table.grow(delta, init);
}

/** `gex_abi::BOOT_STACK_COUNTER`: which bootstrap stack slice is next. */
const BOOT_STACK_COUNTER = 0x00100000 - 192;
function adoptAfterTraps() {
  // Traps a forked child takes before adopting its parent's modules in bulk.
  // Default 1: deferred -- it picks up what it branches into rather than
  // installing all inherited modules up front (which almost every child throws
  // away on an immediate execve).
  return Number(globalThis.GEX_ADOPT_AFTER ?? 1);
}

// The module export that writes its passive element segments into the imported
// tables. See `Module::add_fill`.
const FILL = 'gexfill';
// The one a miss can ask for: fill whichever segment holds this slot.
const FILL_AT = 'gexfillat';
// The shim's debug channel. A heartbeat line gets this isolate's own memory
// figures appended at node (GEX_MEMSTATS): a Worker's wasm memories and heap,
// which the main thread's `process.memoryUsage()` cannot see.
function debugLine(text, traced = false) {
  if (!traced && !logOn('debug')) return;
  if (typeof process !== 'undefined' && process.env?.GEX_MEMSTATS && text.startsWith('[beat]')) {
    try {
      const m = process.memoryUsage();
      const mib = (n) => (n / 1048576) | 0;
      text += ` | isolate heap=${mib(m.heapUsed)}MiB arrayBuffers=${mib(m.arrayBuffers)}MiB external=${mib(m.external)}MiB`;
      if (process.env.GEX_MEMSTATS === 'spaces') {
        const v8 = process.getBuiltinModule?.('v8') ?? null;
        if (v8) text += ' | ' + v8.getHeapSpaceStatistics().filter((sp) => sp.space_used_size > 8 << 20).map((sp) => `${sp.space_name}=${mib(sp.space_used_size)}`).join(' ');
      }
    } catch {}
  }
  narrateAlways(text);
}

// Narration under node goes to stderr: node's console.debug is stdout, which
// is the guest's own output there (an ejected program's, a test's compared
// output). In a page it stays on the debug console.
export function narrate(text) {
  // Debug output, off unless GEX_LOG (`?log=1`) asks for it (log.mjs).
  if (!logOn('debug')) return;
  narrateAlways(text);
}

/** `narrate` whatever GEX_LOG says: a `--strace` the run asked for. */
function narrateAlways(text) {
  // Synchronously where there is a file descriptor to write to. A node
  // Worker's `process.stderr` is flushed by its event loop. A Worker parked
  // in wasm on its wait slot until its process ends never turns that loop, so
  // the lines were lost when it was terminated: native-libs's "libsqlite3.so.0
  // replaced by" lines were all lost after the first child process.
  const fs = typeof process !== 'undefined' ? process.getBuiltinModule?.('fs') : null;
  if (fs?.writeSync) {
    try {
      fs.writeSync(2, `${text}\n`);
      return;
    } catch {}
  }
  const stderr = typeof process !== 'undefined' ? process.stderr : null;
  if (stderr?.write) stderr.write(`${text}\n`);
  else console.debug(text);
}

// Dispatch slots one 4 KiB guest page covers (`gex_abi::SLOTS_PER_PAGE`) and
// the `log2` of the bytes a slot spans (`gex_abi::DISPATCH_SHIFT`): 1024 and
// 2 on both guests today. Asked of the shim once it exists (see
// `syncSlotShift`), so the glue never has to assume; 2 until then.
let SLOT_SHIFT = 2;
let SLOTS_PER_PAGE = 4096 >> SLOT_SHIFT;
function syncSlotShift(exports) {
  const s = exports?.gex_inst_align_shift?.();
  if (typeof s === 'number' && s >= 0 && s <= 2) {
    SLOT_SHIFT = s;
    SLOTS_PER_PAGE = 4096 >> s;
  }
}
const FILL_ALL = 'gexfillall';

// Above this, a module's slots are written when a branch asks rather than at
// instantiation: small units are wanted immediately, closure libraries mostly
// never reached. Provisional.
const FILL_ON_DEMAND_BYTES = 4 << 20;

/// Write `src` at `at`, leaving the pages of it that are entirely zero alone.
///
/// A page a guest process never made resident costs nothing, and a page written
/// with zeros over zeros is resident for no reason. A fresh
/// `WebAssembly.Memory` is zero-filled and so is every page `grow` adds, so any
/// all-zero page of a parent's image is *already* what the child should have --
/// writing it is the only thing that makes it cost memory.
///
/// This is not a guess about what the child needs, unlike `?forkskip`: the
/// bytes end up identical either way. It pays a read of the source (already
/// resident here -- it was just transferred in) to avoid a write to the
/// destination. Guests are sparse in the ordinary case: `.bss`, a fresh
/// anonymous mapping, the untouched tail of a heap or a stack.
///
/// Returns `[written, skipped]`.
function writeSparse(dst, src, at) {
  const PAGE = 4096;
  // Word-at-a-time when the source is aligned, which it is: the regions are
  // `slice`d, so each owns its buffer from offset zero.
  const words = src.byteOffset % 4 === 0 && src.byteLength % 4 === 0
    ? new Uint32Array(src.buffer, src.byteOffset, src.byteLength / 4)
    : null;
  // Division, not `>> 2`: a region can pass 2 GiB (see `#processImage`).
  const blank = (from, to) => {
    if (words) {
      for (let i = from / 4; i < to / 4; i++) if (words[i] !== 0) return false;
      return true;
    }
    for (let i = from; i < to; i++) if (src[i] !== 0) return false;
    return true;
  };
  let written = 0;
  let skipped = 0;
  let i = 0;
  while (i < src.length) {
    const end = Math.min(i + PAGE, src.length);
    if (blank(i, end)) {
      skipped += end - i;
      i = end;
      continue;
    }
    // Coalesce the run of non-blank pages into one `set`, so a dense region is
    // still one copy and not one per page.
    let run = end;
    while (run < src.length) {
      const next = Math.min(run + PAGE, src.length);
      if (blank(run, next)) break;
      run = next;
    }
    dst.set(src.subarray(i, run), at + i);
    written += run - i;
    i = run;
  }
  return [written, skipped];
}

const DISPATCH_DIR = 0x00100000;
/** gex_abi::ftab::NATIVE_FN_TOP: native PE code's function pointers (table
 *  indices) stay below it. */
const NATIVE_FN_TOP = 0x00300000;
const LOW_END = DISPATCH_DIR + 0x80000 * 4;
// An M64 memory's dense directory, one word per page of all 16 GiB, and its
// patched-page bits (gex_abi::DISPATCH_DIR64 .. HIGH_TABLES_END), as the
// shim names them (`gex_m64_tables`, MiB), read at its first instantiation.
let DISPATCH_DIR64 = 0;
let HIGH_TABLES_END = 0;
const noteM64Tables = (e) => {
  const t = e.gex_m64_tables?.();
  if (t === undefined || DISPATCH_DIR64) return;
  DISPATCH_DIR64 = (t & 0xffff) * 2 ** 20;
  HIGH_TABLES_END = (t >>> 16) * 2 ** 20;
};
const dispatchDir = (w) => (w ? DISPATCH_DIR64 : DISPATCH_DIR);
const dispatchPages = (w) => (w ? LIMIT64_BYTES / 4096 : 0x80000);
const PAGES = GUEST_BYTES / 65536;
/** GEX_MEM_MAX_MIB (a test switch): a new wasm32 process memory's maximum, in
 *  MiB, so a grow past it fails the way an engine out of memory fails one
 *  (`memory.grow` -1 in wasm, a RangeError in JavaScript). Undefined when off. */
function memCapPages() {
  const v = globalThis.GEX_MEM_MAX_MIB ?? globalThis.process?.env?.GEX_MEM_MAX_MIB;
  if (!v) return undefined;
  return Math.max(INITIAL_PAGES, Math.min(PAGES, Math.floor(Number(v) * 16)));
}
const INITIAL_PAGES = GUEST_INITIAL_BYTES / 65536;

/** The spans of a map blob (`vma_blob` in the shim: u32 count, then per
 *  region u64 start, u64 end, u32 prot, a name and a code identity). */
function vmaRegions(blob) {
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const out = [];
  let p = 4;
  for (let i = dv.getUint32(0, true); i > 0 && p + 24 <= blob.length; i--) {
    const start = Number(dv.getBigUint64(p, true));
    const end = Number(dv.getBigUint64(p + 8, true));
    const prot = dv.getUint32(p + 16, true);
    p += 20;
    p += 4 + dv.getUint32(p, true);
    const code = dv.getUint32(p, true);
    p += 4 + code + 8;
    out.push({ start, end, prot });
  }
  return out;
}

/**
 * A cheap content hash for whether two copies of a region agree. FNV-1a over
 * words, not bytes: only ever answers same-or-different.
 */
/// How many bytes a fork image entry carries: its `bytes`, or, staged in the
/// global kernel's memory (`#processImage`), its `len`.
function imageLen(r) {
  return r.len ?? r.bytes?.byteLength ?? 0;
}

function sum32(buf) {
  const u = new Uint32Array(buf.buffer ?? buf, buf.byteOffset ?? 0, Math.floor(buf.byteLength / 4));
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193);
  return h >>> 0;
}

/**
 * How far into the slot space a module's element segments reach, from its
 * `gex.meta` (`next_slot` at offset 24).
 *
 * `need` cannot be trusted: a closure library passes zero deliberately, so a
 * forked child adopting it would grow nothing and instantiation would write
 * past the end of the table.
 */
function slotsNeeded(meta) {
  if (!meta || meta.length < 32) return 0;
  const v = new DataView(meta.buffer, meta.byteOffset, meta.byteLength);
  return Number(v.getBigUint64(24, true));
}


/**
 * The signal Linux raises for the fault a trap in guest code stands for, by
 * the engine's message (V8's, then SpiderMonkey's and JavaScriptCore's
 * wording), or 0 for a trap that is no guest fault: SIGSEGV (11) for an
 * access out of bounds, a call through a null or mismatched function pointer
 * or past the table, and a stack overflow; SIGFPE (8) for an integer
 * division by zero or overflow; SIGILL (4) for `unreachable`; SIGBUS (7) for
 * a misaligned atomic. Whether the trap was the guest's at all is the shim's
 * call (`gex_guest_trap`).
 */
function guestTrapSignal(err) {
  const m = String(err?.message ?? '');
  if (err instanceof WebAssembly.RuntimeError) {
    if (/memory access out of bounds|index out of bounds|null function|signature mismatch|indirect call|table index/i.test(m)) return 11;
    if (/divide by zero|division by zero|integer overflow|divide result unrepresentable/i.test(m)) return 8;
    if (/unaligned|misaligned/i.test(m)) return 7;
    if (/unreachable/i.test(m)) return 4;
    return 0;
  }
  if (/maximum call stack size exceeded|too much recursion/i.test(m)) return 11;
  return 0;
}

/**
 * Whether `elf` is a wasm32 program flagged as running i386 code (native
 * Wine's loader, 32-bit Wine): its GEXWASM header sits 0x40 bytes before the entry
 * point (cgen/wasmprog.rs), with the flags at offset 36.
 */
/** clang's names for a C main, and WinMain, which Wine's crt_main also
 * references weakly (see the PE module's instantiation). */
const MAIN_NAMES = ['main', '__main_argc_argv', '__original_main', 'WinMain'];

function wine32ProgramFlags(elf) {
  return (wasmProgramFlags(elf) ?? 0) & 3;
}

/** A native wasm program's image, `[lowest, highest)` of what its loaded
 *  segments cover, or null when `elf` is not one (see `wasmProgramFlags`). */
function wasmProgramImage(elf) {
  if (wasmProgramFlags(elf) === null) return null;
  const e64 = elf[4] === 2;
  const dv = new DataView(elf.buffer, elf.byteOffset, elf.byteLength);
  const w = (o) => (e64 ? Number(dv.getBigUint64(o, true)) : dv.getUint32(o, true));
  const phoff = w(e64 ? 32 : 28), phent = dv.getUint16(e64 ? 54 : 42, true), phnum = dv.getUint16(e64 ? 56 : 44, true);
  let lo = Infinity, hi = 0;
  for (let i = 0; i < phnum; i++) {
    const p = phoff + i * phent;
    if (p + (e64 ? 56 : 32) > elf.length || dv.getUint32(p, true) !== 1) continue;
    const vaddr = w(p + (e64 ? 16 : 8)), memsz = w(p + (e64 ? 40 : 20));
    lo = Math.min(lo, vaddr);
    hi = Math.max(hi, vaddr + memsz);
  }
  return hi > 0 ? [lo, hi] : null;
}

/** A native wasm program's header flags (see `wine32ProgramFlags`), or null
 *  when `elf` is not one: a translated program, or not an ELF. */
function wasmProgramFlags(elf) {
  // ELF32 (32-bit Wine, i386 Wine) or ELF64 (64-bit Wine, x86-64 Wine: wasm-elf.py --elf64)
  if (!(elf instanceof Uint8Array) || elf.length < 64 || elf[0] !== 0x7f || (elf[4] !== 1 && elf[4] !== 2)) return null;
  const e64 = elf[4] === 2;
  const dv = new DataView(elf.buffer, elf.byteOffset, elf.byteLength);
  const w = (o) => (e64 ? Number(dv.getBigUint64(o, true)) : dv.getUint32(o, true));
  const at = w(24) - 0x40;
  const phoff = w(e64 ? 32 : 28), phent = dv.getUint16(e64 ? 54 : 42, true), phnum = dv.getUint16(e64 ? 56 : 44, true);
  for (let i = 0; i < phnum; i++) {
    const p = phoff + i * phent;
    if (p + (e64 ? 56 : 32) > elf.length || dv.getUint32(p, true) !== 1) continue;
    const off = w(p + (e64 ? 8 : 4)), vaddr = w(p + (e64 ? 16 : 8)), filesz = w(p + (e64 ? 32 : 16));
    if (at < vaddr || at + 40 > vaddr + filesz) continue;
    const h = off + (at - vaddr);
    const magic = String.fromCharCode(...elf.subarray(h, h + 7));
    // FLAG_GUEST_I386 (1) or FLAG_GUEST_X86_64 (2): Windows' address space
    return magic === 'GEXWASM' ? dv.getUint32(h + 36, true) : null;
  }
  return null;
}

/** The epoch every Worker's high-resolution clock shares, in nanoseconds:
 * `performance.timeOrigin` is when this realm started, on the wall clock, so
 * `timeOrigin + now()` reads the same in every Worker. Whole microseconds,
 * exact in a double. */
const ORIGIN_NS = BigInt(Math.round(performance.timeOrigin * 1000)) * 1000n;

// The host's clock object, looked up once in this realm: `performance` is a
// Blink attribute of the global scope, and reading it costs more than `now()`
// (a guest clock read is one such call).
const perf = globalThis.performance;

/**
 * Nanoseconds since the Unix epoch, as the guest's clocks and deadlines read
 * them: monotonic and fine-grained (about 5 us with cross-origin isolation),
 * where `Date.now()` steps by a millisecond and follows the wall clock
 * backwards. A delta-time loop saw 0 ms frames, and Wine's
 * QueryPerformanceCounter, which sits on CLOCK_MONOTONIC, ticked by 1 ms.
 * The same base in every Worker: park deadlines, futex timeouts and the
 * lock-free clock calls compare one Worker's reading with another's.
 */
export function hostNowNs() {
  return ORIGIN_NS + BigInt(Math.round(perf.now() * 1e6));
}

export class Env {
  /** Called with the bytes of a guest write to fd 1 or 2. */
  write(fd, bytes) {
    const text = utf8(bytes);
    if (fd === 2) console.error(text);
    else console.log(text);
  }
  /** Nanoseconds since an arbitrary epoch. */
  now() { return hostNowNs(); }
  /**
   * Block until everything `write` has been handed is delivered. Called just
   * before the process's exit is published. Synchronous output needs nothing.
   */
  drainOutput() {}

  /**
   * Called when every guest thread is waiting for input. Resolve once some has
   * been delivered with `pushStdin`, or return false for end-of-file. The
   * default has no terminal, so it returns false and a read returns zero.
   */
  async readStdin() { return false; }

  /**
   * Start a guest process the guest itself forked.
   *
   * `child` is `{ pid, pc, ctx, sp, tls, image }`, where `image` is the
   * parent's mapped memory as `{ at, bytes }` records. A host makes a
   * `WebAssembly.Memory`, writes every record into it, instantiates the shim
   * and guest against it, calls `gex_adopt_child(pid, ctx, pc, sp, tls, brk)`
   * and enters at what it returns. No default: the answer differs by host.
   */
  async spawnProcess(child) {
    throw new Error(
      `this environment cannot start guest process ${child.pid}: no spawnProcess`,
    );
  }

  /**
   * Start a guest thread the guest itself cloned.
   *
   * `thread` is `{ tid, ctx, pc, memory }`. A host makes a Worker, constructs a
   * `Gex` on the *same* memory, and calls `adoptThread`. Nothing is copied --
   * unlike `spawnProcess`, which is why a thread needs a shared memory.
   *
   * Declared and deliberately not defined: a class field would shadow a
   * subclass's method of the same name. `Gex` asks whether the method exists.
   */
}

/**
 * What each dispatch table must hold, for this module, read from its
 * `dir_entries` -- each a `(page, base)` with the table in the base's top
 * eight bits. `next_slot` alone is one running total and cannot say which
 * table wants what. Returns a sparse array of lengths, indexed by table.
 */
export function metaTableNeeds(meta) {
  if (!meta || meta.byteLength < 48) return [];
  const dv = new DataView(meta.buffer, meta.byteOffset, meta.byteLength);
  const count = Number(dv.getBigUint64(40, true));
  const needs = [];
  let at = 48;
  for (let i = 0; i < count && at + 12 <= meta.byteLength; i++, at += 12) {
    const base = dv.getUint32(at + 8, true);
    const t = base >>> 24;
    const end = (base & 0x00ffffff) + SLOTS_PER_PAGE;
    if (!(needs[t] >= end)) needs[t] = end;
  }
  return needs;
}

/**
 * What each dispatch table must hold, read from the module's active element
 * segments -- the only authority, since `Metadata` cannot give per-table needs.
 * Returns a sparse array indexed by table; passive segments are skipped.
 */
export function elementTableNeeds(wasm) {
  if (!(wasm instanceof Uint8Array) || wasm.length < 8) return [];
  const dv = new DataView(wasm.buffer, wasm.byteOffset, wasm.byteLength);
  let i = 8;
  const leb = () => {
    let n = 0, shift = 0, b;
    do { b = dv.getUint8(i++); n |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return n >>> 0;
  };
  const needs = [];
  try {
    while (i < wasm.length) {
      const id = dv.getUint8(i++);
      const size = leb();
      const end = i + size;
      if (id !== 9) { i = end; continue; }
      const count = leb();
      for (let seg = 0; seg < count; seg++) {
        const flags = leb();
        let table = 0;
        if (flags === 2 || flags === 6) table = leb();
        let off = -1;
        if ((flags & 3) === 0 || flags === 2 || flags === 6) {
          // The offset expression: only a constant is understood.
          if (dv.getUint8(i) === 0x41) { i++; off = leb(); }
          while (dv.getUint8(i) !== 0x0b) i++;
          i++;
        }
        if (flags === 1 || flags === 2 || flags === 3) leb();
        else if (flags >= 5) leb();
        const n = leb();
        for (let k = 0; k < n; k++) {
          if (flags & 4) { while (dv.getUint8(i) !== 0x0b) i++; i++; } else leb();
        }
        // Passive and declared segments write nothing now.
        const active = flags === 0 || flags === 2 || flags === 4 || flags === 6;
        if (active && off >= 0 && !(needs[table] >= off + n)) needs[table] = off + n;
      }
      i = end;
    }
  } catch {
    return needs;
  }
  return needs;
}

/** Where a module's `gex.dense` section payload (its base word) starts, or -1. */
function denseBaseAt(wasm) {
  let i = 8;
  const leb = () => {
    let n = 0, shift = 0, b;
    do { b = wasm[i++]; n |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return n >>> 0;
  };
  while (i < wasm.length) {
    const id = wasm[i++];
    const size = leb();
    const end = i + size;
    if (id === 0) {
      const n = leb();
      if (n === 9 && utf8(wasm.subarray(i, i + n)) === 'gex.dense') return i + n;
    }
    i = end;
  }
  return -1;
}

/** A module's `gex.dense` section (gex_emit::dense::SECTION), or null. */
export function denseSection(wasm) {
  if (wasm instanceof WebAssembly.Module) {
    const secs = WebAssembly.Module.customSections(wasm, 'gex.dense');
    return secs.length ? new Uint8Array(secs[0]) : null;
  }
  if (!(wasm instanceof Uint8Array)) return null;
  const dv = new DataView(wasm.buffer, wasm.byteOffset, wasm.byteLength);
  let i = 8;
  const leb = () => {
    let n = 0, shift = 0, b;
    do { b = dv.getUint8(i++); n |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return n >>> 0;
  };
  while (i < wasm.length) {
    const id = dv.getUint8(i++);
    const size = leb();
    const end = i + size;
    if (id === 0) {
      const nameLen = leb();
      if (utf8(wasm.subarray(i, i + nameLen)) === 'gex.dense') return wasm.slice(i + nameLen, end);
    }
    i = end;
  }
  return null;
}

export function metadataSection(wasm) {
  // A compiled `WebAssembly.Module` -- which `attach` accepts, so the guest
  // image and the closure's libraries cross `postMessage` by reference rather
  // than re-cloning and re-compiling hundreds of megabytes per forked process
  // -- has no bytes to scan. Its custom sections are read through the
  // reflection API instead. Without this, a forked process in the browser threw
  // "First argument to DataView constructor must be an ArrayBuffer".
  if (wasm instanceof WebAssembly.Module) {
    const secs = WebAssembly.Module.customSections(wasm, 'gex.meta');
    return secs.length ? new Uint8Array(secs[0]) : null;
  }
  // A bare ArrayBuffer (or SharedArrayBuffer) has no `.buffer`, so the DataView
  // below would throw "First argument to DataView constructor must be an
  // ArrayBuffer"; wrap it. Anything else is not module bytes and has no
  // metadata to give -- a guard rather than a crash, matching
  // `elementTableNeeds`.
  if (wasm instanceof ArrayBuffer || (typeof SharedArrayBuffer !== 'undefined' && wasm instanceof SharedArrayBuffer)) {
    wasm = new Uint8Array(wasm);
  }
  if (!(wasm instanceof Uint8Array)) return null;
  const dv = new DataView(wasm.buffer, wasm.byteOffset, wasm.byteLength);
  let i = 8;
  const leb = () => {
    let n = 0, shift = 0, b;
    do { b = dv.getUint8(i++); n |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return n >>> 0;
  };
  while (i < wasm.length) {
    const id = dv.getUint8(i++);
    const size = leb();
    const end = i + size;
    if (id === 0) {
      const nameLen = leb();
      const name = utf8(wasm.subarray(i, i + nameLen));
      // A copy, not a view: `subarray` shares the module's ArrayBuffer, and
      // postMessage would then clone every compiled module behind it.
      if (name === 'gex.meta') return wasm.slice(i + nameLen, end);
    }
    i = end;
  }
  return null;
}

// The dispatch tables a translated module imports and exports, in index order.
// More than one because V8 caps a table at ten million entries (38 MiB of guest
// text) and the graphics rootfs exceeds that. Table 0 keeps its bare name for
// modules built when there was only one.
const TABLE_NAMES = [
  'dispatch',
  'dispatch1',
  'dispatch2',
  'dispatch3',
  'dispatch4',
  'dispatch5',
  'dispatch6',
  'dispatch7',
];

/** The tag on a bridge's table (cgen.rs `ALT_BRIDGE`): the other mode's. */
const ALT_BRIDGE = 0x80000000;

/** A stateful provider's `provider_throw`, caught by its `provider_try`. */
class ProviderThrow extends Error {
  constructor(code) {
    super(`gex.provider_throw(${code}) with no gex.provider_try to catch it`);
    this.code = code;
  }
}

/** posix_spawn children run in their parent's Worker, narrated
 *  (GEX_VFORK_TRACE, `?vforktrace=1`): to the console, or appended to the
 *  file it names under Node (a Worker's console output is lost when the run
 *  ends first). */
export function vforkTrace(line) {
  const to = globalThis.GEX_VFORK_TRACE || globalThis.process?.env?.GEX_VFORK_TRACE;
  if (!to) return;
  if (typeof to === 'string' && to.includes('/') && globalThis.process?.getBuiltinModule) {
    globalThis.process.getBuiltinModule('fs').appendFileSync(to, `${line}\n`);
  } else {
    console.error(line);
  }
}

export class Gex {
  /** Whether this process's memory is a 64-bit one (its addresses, table
   *  indices and stack pointer i64): the start message says, and the
   *  constructor is told. Per runtime, not a global: the single-thread
   *  build's processes of both flavours share one realm. */
  #wide = false;
  #memory; #shim; #guest; #env; #u8; #imports; #net = null; #netShared = false; #gl = null; #d3d = false; #wl = null;
  // The channel `#turn` posts to itself on, made once and kept.
  #tick = null;
  // Translated modules this Worker has put into its own dispatch tables. Per
  // Worker, not per process: tables belong to an instance, so every thread
  // starts empty however much the process has translated.
  #installed = 0;
  // This process narrates its syscalls (`?strace=1`): the shim's debug lines are wanted.
  #traced = false;
  // What compiling and instantiating shared modules costs. See
  // `#installModuleInner`.
  #installMs = 0;
  #faBaseMs = 0;
  #faInitMs = 0;
  #fillWidthTold = false;
  #imageMs = 0;
  #imageBytes = 0;
  #imageRegions = 0;
  #nowFills = 0;
  #nowMs = 0;
  #bigGrows = [];
  /// `gex_table_need(0)` at named moments, so the step that raises it is
  /// visible rather than guessed at.
  #tn = [];
  #tnAt(tag) {
    const e = this.#shim?.exports;
    if (!e?.gex_table_need || this.#tn.length > 24) return;
    const v = e.gex_table_need(0) >>> 0;
    const last = this.#tn[this.#tn.length - 1];
    if (!last || !last.endsWith(`:${v}`)) this.#tn.push(`${tag}:${v}`);
  }
  #iGrows = 0;
  #iAdded = 0;
  // How many "table index out of bounds" dispatch traps were repaired by
  // catching this Worker's tables up to the shared claimed high-water. Capped
  // in the run loop so a genuine over-run still fails.
  #tableRetries = 0;
  #iCopied = 0;
  #compileMs = 0;
  /** `#fjitModule`'s compile time, and when it next reports (`[gex] fjit so far`). */
  #fjitCompileMs = 0;
  #fjitNextReport = 0;
  #instantiateMs = 0;
  #growMs = 0;
  /// What every Worker pays instantiating the shim and guest before the guest
  /// runs -- the floor under every process, which `installMs` does not cover.
  #shimCompileMs = 0;
  #shimInstantiateMs = 0;
  #guestInstantiateMs = 0;
  #installs = 0;
  /// Compiled `WebAssembly.Module`s for the process's registry, by index,
  /// shared across the sibling Workers of one process. A thread `#catchUp`s the
  /// same modules its siblings already compiled; recompiling each from bytes
  /// per Worker cost seconds (16.9s of a 45s gtk3-demo run, thousands of
  /// modules re-`compile`d). Populated from `sharedModule` messages the host
  /// relays; a `WebAssembly.Module` crosses `postMessage` by reference, so this
  /// carries no byte copies. Only the compile is shared -- each Worker still
  /// instantiates against its own tables.
  #sharedModules = new Map();
  /** Whether this process numbers its blocks densely (gex_emit::dense): a
   *  dense table's slots live in the process's slot maps, and this Worker's
   *  only table for them is `blocks`, one entry per block. */
  #dense = false;
  /** `blocks` holds entries (gex_emit::dense), as it does wherever dispatch
   *  is dense: an entry is `(ctx, arm) -> ()`, and a slot map word
   *  `entry << 8 | arm`. */
  #entries = false;
  /** Entry wrappers of block-type functions (`#entryOf`). */
  #entryWraps = new WeakMap();
  /** Entries with watched arms (`#refSet`): entry -> { orig, arms, fn }. */
  #entryWatch = new Map();
  // Instances whose slots have not been written, newest first: a miss asks
  // each in turn whether it holds the slot. See `fillSlot`.
  #unfilled = [];
  /** Closure libraries that arrived after this process started. */
  #streamed = 0;
  /** Closure libraries registered but not yet instantiated, in shim order. */
  #lazyClosure = [];
  /** Inherited modules registered but not yet instantiated, in shim order. */
  #lazyInherited = [];
  // Fill requests answered, and requests nobody could answer.
  #filled = 0;
  #unanswered = 0;
  #slowFills = 0;
  #slowMs = 0;
  #publishMs = 0;
  #publishes = 0;
  #pubGrowMs = 0;
  #pubGrewSlots = 0;
  #wantMs = 0;
  #wants = 0;
  #instMs = 0;
  #tailMs = 0;
  /// Modules that only instantiated after growing every table the running
  /// total implies. If common, the loose reading is the wrong default.
  #strictGrows = 0;
  #strictSlots = 0;
  /// Next free slot in each table, for pages this process claims.
  #claimed = [];
  /// Bytes of the parent's image this child wrote into its own memory, and
  /// bytes it was told to leave alone by `?forkskip`. Reported on exit: with no
  /// copy-on-write, the first number is what a fork costs the tab.
  #forkCopied = 0;
  #forkSkipped = 0;
  #forkZeros = 0;
  /** A 32-bit Wine process's Windows address-space line: a forked child's from its
   *  parent, -1 for a fresh process to split at once (see `#workerStack`). */
  #windowsSpace = 0;
  /** A 64-bit Wine program being loaded into a 64-bit memory: the split. */
  #wine64Split = false;
  /** A process that runs no translated code (a native wasm program, a fork
   *  child of one, an exec of one): no placement window, its arena from
   *  `MMAP_BOTTOM` (`gex_set_no_place_window`, before the first claim).
   *  Null for one with the window, else its program's image span, which the
   *  arena steps over (`[0, 0]`: a fork child's, inherited). */
  #noPlace = null;
  get forkCopiedBytes() { return this.#forkCopied; }
  get forkSkippedBytes() { return this.#forkSkipped; }
  get forkZeroBytes() { return this.#forkZeros; }
  // The parent's modules, held rather than installed. See `#maybeAdopt`.
  #inherited = [];
  /// The exec generation last seen, so `#maybeAdopt` can tell when this process
  /// has become a different program.
  #execGen = 0;
  #traps = 0;
  // What `load` was told, so a forked child can be told the same: a wrong tty
  // or identity changes how libc buffers and whether a shell prints a prompt.
  #opts = {};
  // Every module this process has installed, compiled. Kept because a forked
  // child needs the same code in the same slots, and a `WebAssembly.Module`
  // crosses to a Worker without recompiling.
  #compiled = [];
  // `{ module, memory }` for the global kernel and its instance. The global
  // kernel is a second wasm module holding state that spans processes (process
  // table, file table, pipes), instantiated by every Worker against the same
  // shared kernel memory. Null is a run of one process, which needs none of it.
  #kernelSpec = null;
  /** This instance made its memory (a process's first Worker): `releaseWorker`
   *  takes it off the tab's count. */
  #ownsMemory = false;
  /** The single-thread build: the fiber this runtime runs on (single.mjs
   *  `FiberWorker`), whose waits and turns end with it. */
  #fiber = null;
  #census = null;
  /** The single-thread build: where this fiber's JS last was, for the census
   *  (a fiber that waits in JS is otherwise invisible to it). */
  #where = '';
  #kernel = null;
  // This Worker's shim stack, claimed on the way in. See `#workerStack`.
  #stackTop = null;
  /** The slot went back (`releaseWorker`): nothing may call in again. */
  #released = false;
  /** A thread's Worker whose `clone` promised it a slot it has not claimed. */
  #promise = false;
  /** `gex_worker_init` found no slot: the instance has no thread-locals. */
  #noSlot = false;
  /** This instance is a forked child's first: its slot bits are its parent's. */
  #forkChild = false;
  /** A forked child's inherited regions' ends, for its first arena claim. */
  #inheritedEnds = [];

  /**
   * One turn of the event loop, to deliver messages that arrived while this
   * Worker was parked. A message task, not a timer: nothing to wait for.
   */
  #turn() {
    if (!this.#tick) {
      this.#tick = new MessageChannel();
      this.#tick.port1.start();
    }
    return new Promise((r) => {
      this.#tick.port1.onmessage = () => r();
      this.#tick.port2.postMessage(0);
    });
  }

  /** Consecutive turns taken for a wait that is only on the network. */
  #netTurns = 0;

  /**
   * How the run loop waited, by kind. `park` is a sleep on the shared word
   * another Worker ends; `turn` is a MessageChannel task; `nap` is a timer.
   */
  waits = { park: 0, turn: 0, nap: 0, net: 0 };

  /**
   * Traps after which the guest went straight on (`gex_after_trap` took the
   * resume), and those that went through each step, by loop: the run loop,
   * a call-out's and a native caller's system call.
   */
  traps = { loop: 0, loopFull: 0, callout: 0, calloutFull: 0, native: 0, nativeFull: 0 };

  /**
   * One question for a trap that left this side nothing to do: the resume
   * address the shim took (its context then at `#afterTrapCtx`), or 0n when
   * something waits (a fill, an install, a unit, a fork, a thread, a fault,
   * a vfork child here, a parent's modules to adopt), nothing taken.
   */
  #afterTrap() {
    const f = this.#shim.exports.gex_after_trap;
    if (!f || this.#inherited.length || this.#vfork.length) return 0n;
    return f(this.#installed, this.#wasmdllHave, this.#cgenMode & 3 ? 1 : 0);
  }
  #afterTrapAt = 0;
  #afterTrapCtx() {
    this.#afterTrapAt ||= this.#shim.exports.gex_after_trap_ctx() >>> 0;
    return this.#dv().getUint32(this.#afterTrapAt, true);
  }

  /** Resolves the network wait in progress (`#netWait`) on this Worker's own
   *  network's activity. */
  #netWake = null;
  #kernelWords = null;

  /** What ends a network wait, sampled before the guest is given another
   *  chance (`gex_wake`): the page network's event count. (This Worker's wait
   *  slot is sampled by the shim itself, at the start of each scheduler turn:
   *  `gex_wait_seq`.) */
  #netSnap() {
    return { n: this.#net?.activitySeq?.() ?? null };
  }

  /**
   * Sleep, without blocking this Worker's event loop, until the network has
   * something (this Worker's own network's `onActivity`, or the page's
   * through the bridge's activity word), something the thread waits for is
   * notified (`Atomics.waitAsync` on this Worker's wait slot), or the waiting
   * thread's own deadline is due (`gex_wait_bound_ms`, at most 50 ms). It
   * replaces a MessageChannel turn loop that spun every network-waiting
   * Worker at full speed, each turn taking the host lock four times (node's
   * idle Workers in the page: ~2.2 s of KLock each over a 13 s run).
   */
  #netWait(e, snap) {
    // A waiting thread installs nothing, so it lets go of dead modules here.
    this.#reapDead();
    const ms = e.gex_wait_bound_ms ? e.gex_wait_bound_ms(50) : 1;
    const waits = [new Promise((r) => { this.#netWake = r; })];
    const word = e.gex_wait_word?.() >>> 0;
    if (word) {
      this.#kernelWords ??= new Int32Array(this.#kernelSpec.memory.buffer);
      const w = Atomics.waitAsync(this.#kernelWords, word >>> 2, e.gex_wait_seq(), ms);
      if (!w.async) return Promise.resolve();
      waits.push(w.value);
    } else {
      waits.push(new Promise((r) => setTimeout(r, Math.min(ms, 4))));
    }
    if (snap.n !== null) waits.push(this.#net.activityWait(snap.n, ms));
    return Promise.race(waits).finally(() => { this.#netWake = null; });
  }

  // A MessageChannel first, then `setTimeout(_, 1)`: a bare timer nests and the
  // HTML spec clamps a nested timer to 4ms, but the level resets after a
  // message. Not a bare message task either -- that spins.
  #nap() {
    if (!this.#tick) {
      this.#tick = new MessageChannel();
      this.#tick.port1.start();
    }
    return new Promise((r) => {
      this.#tick.port1.onmessage = () => setTimeout(r, 1);
      this.#tick.port2.postMessage(0);
    });
  }

  constructor(env = new Env(), { shared = false, memory = null, kernel = null, moduleStore = null, modules = null, fiber = null, mem64 = !!globalThis.GEX_MEM64 } = {}) {
    this.#wide = mem64;
    this.#fiber = fiber;
    this.#env = env;
    // Compiled modules this Worker was given before it had a process, kept
    // as they are (a Map of key to Module), not copied one by one.
    if (modules) this.#sharedModules = modules;
    // A thread joins its process's module store (`#moduleStore`) too.
    this.#moduleStore = moduleStore;
    // `memory` adopts a process's existing address space rather than making a
    // new one -- which is what a *thread* is: a second Worker on the same
    // memory, seeing the same guest and kernel. A new memory is a new process.
    // (counted in the tab's memories, `wasmMemory`, until this instance's
    // Worker is released)
    this.#ownsMemory = !memory;
    this.#memory =
      memory ??
      (this.#wide
        ? wasmMemory({ address: 'i64', initial: BigInt(INITIAL_PAGES), maximum: BigInt(LIMIT64_BYTES / 65536), shared }, 'a process memory64', kernel?.spare)
        : wasmMemory({ initial: INITIAL_PAGES, maximum: memCapPages() ?? PAGES, shared }, 'a process memory', kernel?.spare));
    this.#kernelSpec = kernel;
  }

  /**
   * Where this Worker's shim stack goes. Each Worker needs its own, since two
   * sharing a memory (a thread) would clash; `gex_worker_init` claims a stack
   * and thread-local block per Worker. The plain build has one of each.
   */
  #workerStack() {
    // Claimed once and remembered: asked again after every trap (the stack
    // pointer resets to the top, since a trap abandons the stack), and a fresh
    // slot each time would exhaust the sixty-four available in a second.
    if (this.#stackTop === null) {
      // `gex_worker_init` runs on a stack too, and until it answers this
      // instance's is wasm-ld's: the top of the guest's first 8 MiB, which is
      // a Windows program's image (a game's .text at 0x7ff000..0x800000
      // was overwritten by every thread that started). So it runs on a
      // bootstrap slice of the shim's own stack band below SHIM_STACK_TOP,
      // one of 32 handed out round-robin through a counter in the reserved
      // low region -- Workers starting together get different ones.
      // The single-thread build runs the shared build's shim on a plain
      // memory: one memory, several instances (threads), so the same claim.
      if (typeof this.#shim.exports.gex_worker_init?.call === 'function' && (isShared(this.#memory.buffer) || (SP_CODE && single.on))) {
        const k = Atomics.add(new Int32Array(this.#memory.buffer), BOOT_STACK_COUNTER >> 2, 1) & 31;
        this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, SHIM_STACK_TOP - k * 0x40000);
      }
      // A 32-bit Wine process says so before anything claims arena -- this stack is
      // the first claim: the 2 GiB split at its parent's line, or 32-bit Wine's
      // compact space (nothing grown; inherited regions become islands).
      // `?winesplit=1` before the space is taken: the split, not the compact one.
      if (globalThis.GEX_WINE32_SPLIT || this.#wine64Split) this.#shim.exports.gex_set_wine_split?.(1);
      // `?nullguard=1` before anything is translated (gex_abi::null_guard).
      if (globalThis.GEX_NULL_GUARD) this.#shim.exports.gex_set_null_guard?.(1);
      // `?mxcsr=1` before anything is translated: SSE under MXCSR (gex_ir::mxcsr_exact).
      if (globalThis.GEX_MXCSR) this.#shim.exports.gex_set_mxcsr_exact?.(1);
      // `?pinregion=BYTES`: a project's libraries' region above the placement
      // window, before the first claim of arena (gex_abi::set_pin_region).
      if (globalThis.GEX_PIN_REGION) this.#shim.exports.gex_set_pin_region?.(BigInt(globalThis.GEX_PIN_REGION));
      if (this.#noPlace) this.#shim.exports.gex_set_no_place_window?.(BigInt(this.#noPlace[0]), BigInt(this.#noPlace[1]));
      // `?sehreport=1`: crash reports name the SEH chain and a longjmp's jmp_buf.
      if (globalThis.GEX_SEH_REPORT) this.#shim.exports.gex_set_seh_report?.(1);
      // Which memory flavours the page runs: what an exec
      // may ask for (vfork.rs `gex_set_page_flavours`).
      if (globalThis.GEX_PAGE_FLAVOURS !== undefined) this.#shim.exports.gex_set_page_flavours?.(globalThis.GEX_PAGE_FLAVOURS | 0);
      if (this.#windowsSpace > 0) this.#shim.exports.gex_adopt_windows_space?.(this.#windowsSpace);
      else if (this.#windowsSpace < 0) this.#shim.exports.gex_split_windows_space?.();
      // A forked child's memory carries its parent's slot bits (`adoptProcess`).
      if (this.#forkChild) {
        this.#forkChild = false;
        this.#shim.exports.gex_worker_slots_reset?.();
        // Where its inherited mappings end, before the claim below picks its
        // arena floor (`gex_note_inherited`).
        // With its start where the shim takes it: a 32-bit Wine child keeps the
        // regions as islands its arena steps over (`gex_note_inherited_range`).
        const e = this.#shim.exports;
        for (const [at, end] of this.#inheritedEnds) {
          if (e.gex_note_inherited_range) e.gex_note_inherited_range(BigInt(at), BigInt(end));
          else e.gex_note_inherited?.(BigInt(end));
        }
        this.#inheritedEnds = [];
      }
      // The slot's span may be one another Worker took and grew the memory
      // for (a `clone`'s `spans_ready`) since this instance was made, and the
      // engine updates this Worker's bound only at its next interrupt: the
      // thread-locals' first write would trap out of bounds (Wine's sync.exe,
      // about one start in ten of `wine/backtoback`'s). Growing by nothing
      // takes the memory's present length (worker.mjs does so again before
      // the thread runs).
      if (isShared(this.#memory.buffer)) {
        try {
          this.#memory.grow(0);
        } catch {
          this.#memory.grow(0n);
        }
      }
      const at = this.#shim.exports.gex_worker_init?.();
      // A thread's slot was promised by its `clone`: claimed now, or lost.
      this.keepPromise();
      // No slot (every one held by a live instance, or no arena for its
      // stack). Running on, or calling into this instance at all, would
      // put its thread-locals at address zero (`__tls_base` unset), over
      // the thread contexts at the bottom of the memory: the instance is
      // done (`noSlot`) and nothing calls in again.
      if (at === 0 && (isShared(this.#memory.buffer) || (SP_CODE && single.on))) {
        this.#noSlot = true;
        throw new Error('gex: no shim stack slot left for this Worker (every slot of this process is held)');
      }
      this.#stackTop = at ? at | 0 : SHIM_STACK_TOP | 0;
      // A test's: this Worker's stack guard broken, as an overrun leaves it,
      // for the tripwire at its next syscall.
      if (globalThis.GEX_TEST_BREAK_STACK_GUARD) this.#shim.exports.gex_test_break_stack_guard?.();
    }
    return this.#stackTop;
  }

  /**
   * Make this process's memory cover `end`, growing it if it does not. For the
   * places JavaScript writes guest memory directly -- a forked child's image
   * and nothing else; the shim grows for itself from inside.
   */
  #growTo(end) {
    const have = this.#memory.buffer.byteLength;
    if (end <= have) return;
    // A 64-bit memory counts its pages in BigInt, here as everywhere else.
    const pages = Math.ceil((end - have) / 65536);
    try {
      this.#memory.grow(this.#wide ? BigInt(pages) : pages);
    } catch (e) {
      // Said for what it is: the engine's RangeError names no process and no size.
      throw new Error(`gex: out of memory: this process's memory would not grow from ${(have / 1048576) | 0} to ${Math.ceil(end / 1048576)} MiB (${e?.message ?? e})`);
    }
    this.#u8 = null;
  }

  /** Closure library modules waiting for the guest instance to exist. */
  #closureModules = [];
  /// Ask whoever holds the closure for library `i`, compiled. Null when the
  /// caller handed the modules over up front, which the page still does for
  /// pid 1 -- it compiled them itself and has nowhere to ask.
  #wantLibrary = null;
  /// The dispatch pages to narrate, by number. Empty in a normal run; the gate
  /// is the set itself, since `opts.strace` never reaches a Worker. `?trace=1`
  /// prints for these only.
  static watchPages = new Set();
  /// Fill a whole image page on first ask. See `fillSlotNow`.
  static wholePage = false;
  /// How many times the fill has said why it gave up. See `fillSlotNow`.
  static #whyNo = 0;
  /// And how many times it has said what the fill itself answered.
  static #whyDid = 0;
  /// Report the exact slot a missing pc dispatches through. See `fillSlotNow`.
  static watchSlot = false;
  /// Narrate a fill request no module answered. Off by default: a closure that
  /// deliberately leaves a library out (`GEX_SKIP`, the crypto libs) hands
  /// every one of that library's pages to the run-time translator, and each is
  /// a fill nobody holds followed by a clean lift -- expected, not a fault, and
  /// a flood of `console.error` that fails a harness counting them. On to debug
  /// a genuinely unfillable page. See `#fillSlotTail`.
  static debugFill = false;
  /// What each ask saw, keyed `page:from`. See `fillSlotNow` and `liftSlots`.
  static askState = new Map();
  /// A dispatch page whose directory writes should be narrated, or -1: every
  /// writer of `DISPATCH_DIR` reports itself, naming the one that moves it.
  static watchDir = -1;
  /** `soname=path,...` (the page's `?wasmdlls=`): libraries replaced by wasm64 providers. */
  static wasmDlls = '';
  /** `name,...` (the page's `?elfunits=`): shared objects that get function-tier units. */
  static elfUnits = '';
  /** `export,...` (the page's `?provskip=`): exports no provider binds (`GEX_PROVIDER_SKIP`). */
  static providerSkip = '';
  /** Provider sources from outside gex, `name\ntext` entries separated by NUL (`GEX_PROVIDER_DIRS` natively). */
  static providerSources = '';
  /** `dll=file,...`: stateful providers replacing Windows libraries, files under the unit cache (`libprov` natively). */
  static stateful = '';
  /** `+name,-name` (the page's `?fjitlibs=`): objects the function JIT is limited to or kept from. */
  static fjitLibs = '';
  /** `base:key:identity:path;...` (cgen.rs `gex_cgen_identities`). */
  static identities = '';
  static #whySlot = 0;

  /** A DataView of the memory, rebuilt as `#bytes` is: for `#hookStack`,
   *  which every direct system call entry runs (a new view each call was
   *  4.7% of a game's render thread samples). */
  #dv() {
    const buf = this.#memory.buffer;
    if (!this.#dview || this.#dview.buffer !== buf || this.#dview.byteLength !== buf.byteLength) {
      this.#dview = new DataView(buf);
    }
    return this.#dview;
  }

  #dview = null;

  #bytes() {
    // Rebuilt when the memory has grown, not only when detached. A shared
    // memory never detaches -- the old SharedArrayBuffer stays valid at its
    // old length -- so a length check is needed: the shim grows the memory
    // from inside on every guest `mmap`, and a stale view reads nothing.
    const buf = this.#memory.buffer;
    if (!this.#u8 || this.#u8.buffer !== buf || this.#u8.byteLength !== buf.byteLength) {
      this.#u8 = new Uint8Array(buf);
    }
    return this.#u8;
  }

  get memory() { return this.#memory; }

  /** The global kernel's memory, this process's wait queue's address in it
   *  and where the wait slots' words start, or null: what a compositor walks
   *  to wake this process's parked threads after an input event
   *  (`wl-bridge.mjs` `sendEvent`). */
  /** The global kernel's memory and the address in it of `/dev/dsp`'s
   *  ring (hosts/kernel/src/audio.rs), or null: what the page plays
   *  (page/src/audio.js). */
  audioRing() {
    const e = this.#shim?.exports;
    if (!this.#kernel || !e?.gex_audio_ring_addr || !this.#kernelSpec?.memory) return null;
    const at = e.gex_audio_ring_addr() >>> 0;
    return at ? { memory: this.#kernelSpec.memory, at } : null;
  }

  kernelWake() {
    const e = this.#shim?.exports;
    if (!this.#kernel || !e?.gex_proc_queue_addr || !this.#kernelSpec?.memory) return null;
    return { memory: this.#kernelSpec.memory, addr: e.gex_proc_queue_addr() >>> 0, slots: e.gex_slots_addr() >>> 0 };
  }

  /**
   * Every guest block PC translated in this process so far, as a BigUint64Array
   * (a fresh copy, safe to keep). The profile a sandbox run dumps to seed a
   * reachability-trimmed closure build. Null if the shim lacks the export.
   */
  coverage() {
    const e = this.#shim?.exports;
    if (!e?.gex_coverage_snapshot) return null;
    const n = e.gex_coverage_snapshot() >>> 0;
    const ptr = e.gex_coverage_ptr() >>> 0;
    // Copied out: the snapshot lives in linear memory that a later host call
    // may reallocate, and the caller writes it to disk asynchronously.
    return new BigUint64Array(this.#memory.buffer.slice(ptr, ptr + n * 8));
  }

  /**
   * Let go of everything that would keep a host's event loop alive. `#tick` is
   * a started `MessageChannel`, and in node a started port is an active handle
   * that keeps the process alive on its own -- so a run that ever yielded never
   * exited. Closed here rather than `unref`'d so it holds the run while in use.
   */
  close() {
    // The census timer too: a fiber (the single-thread build) is not a
    // Worker whose end takes its timers with it.
    if (this.#census) {
      clearInterval(this.#census);
      this.#census = null;
    }
    if (this.#tick) {
      this.#tick.port1.close();
      this.#tick.port2.close();
      this.#tick = null;
    }
  }

  get shim() { return this.#shim; }

  /** The slot went back (`releaseWorker`): call nothing in this instance. */
  get released() { return this.#released; }
  /** No slot was had (`#workerStack`): call nothing in this instance. */
  get noSlot() { return this.#noSlot; }
  /**
   * A promised slot claimed, or never to be: told once, so the process's
   * next `clone` can count on it. Pairs with `thread_runner_take` (the
   * shim's), which promised it.
   */
  keepPromise() {
    if (!this.#promise) return;
    this.#promise = false;
    try { this.#shim?.exports?.gex_worker_promise_done?.(); } catch {}
  }

  /**
   * Give this instance's stack and thread-local slot back to the process
   * (`gex_worker_release`). The last call into this instance: the slot's next
   * holder runs on the same stack.
   */
  /** This process's memory, every non-zero page zeroed, for another
   *  process to start in; null where that cannot be done (the memory is not
   *  shared, mem64, no global kernel). Only after the process is gone and
   *  nothing will run in this instance again. The non-zero runs are found by
   *  the shim's scan (`gex_image_runs`) and cleared by the kernel's
   *  `memory.fill` (`k_proc_zero`), so only pages the process wrote are
   *  written again. */
  recycleMemory() {
    const e = this.#shim?.exports;
    const k = this.#kernel;
    if (!e?.gex_image_runs || !k?.k_proc_zero || this.#wide || (SP_CODE && single.on)) return null;
    if (!isShared(this.#memory.buffer)) return null;
    const size = this.#memory.buffer.byteLength;
    let cap = 4096;
    let at = e.gex_alloc(cap * 8) >>> 0;
    const runs = [];
    const CHUNK = 64 << 20;
    for (let s = 0; s < size; ) {
      const end = Math.min(size, s + CHUNK);
      const n = e.gex_image_runs(s, end, at, cap);
      if (n < 0) {
        e.gex_free?.(at, cap * 8);
        cap *= 4;
        at = e.gex_alloc(cap * 8) >>> 0;
        continue;
      }
      const pairs = new Uint32Array(this.#memory.buffer, at, n * 2);
      for (let j = 0; j < n; j++) runs.push(pairs[j * 2], pairs[j * 2 + 1]);
      s = end;
    }
    for (let j = 0; j < runs.length; j += 2) {
      if (k.k_proc_zero(BigInt(runs[j]), runs[j + 1]) !== 0) return null;
    }
    return this.#memory;
  }

  /** `threw`: after a throw (worker.mjs), which may have left the shim's
   *  heap mid-call, so nothing more is asked of it. */
  releaseWorker(threw = false) {
    if (this.#stackTop === null) return;
    // The census first: its timer calls into this instance every 3 s
    // (gex_alloc, gex_wait_census). An ordinary exit has `close` stop it
    // first, but a run that threw (worker.mjs) goes straight here, and its
    // Worker, alive until the host ends it, would tick on the stack and
    // thread-locals the slot's next holder already runs on. Pairs with the
    // slot claim in `gex_worker_init`.
    if (this.#census) {
      clearInterval(this.#census);
      this.#census = null;
    }
    this.#released = true;
    // this Worker's late-call flags (`#lateFlagsAt`): no thunk reads them now
    const chunks = threw ? null : this.#cgen?.lateChunks;
    if (chunks) {
      this.#cgen.lateChunks = null;
      try { for (const [at, n] of chunks) this.#shim.exports.gex_free?.(at, n); } catch {}
    }
    try { this.#shim?.exports?.gex_worker_release?.(); } catch {}
    this.#stackTop = null;
    if (this.#ownsMemory) {
      this.#ownsMemory = false;
      wasmMemoryGone(this.#wide ? 'a process memory64' : 'a process memory', this.#kernelSpec?.spare);
    }
  }

  /**
   * Translated modules this Worker has installed into its own dispatch tables.
   * A late-joining thread must reach the process's count or its tables miss
   * code the shared page directory already points at.
   */
  get installedModules() { return this.#installed; }
  /** Milliseconds this process spent compiling and instantiating modules. */
  get installMillis() { return Math.round(this.#installMs); }
  get compileMillis() { return Math.round(this.#compileMs); }
  get instantiateMillis() { return Math.round(this.#instantiateMs); }
  get growMillis() { return Math.round(this.#growMs); }
  /// Syscalls the guest's dynamic linker made, and when it made its last. See
  /// `gex_loader_calls`.
  get loader() {
    const e = this.#shim?.exports;
    if (!e?.gex_loader_calls) return null;
    return {
      calls: Number(e.gex_loader_calls()),
      doneMs: Number(e.gex_loader_done_ms()),
      // What the linked-image store did for this process, if anything.
      link: ['off', 'no-kernel', 'armed', 'kept', 'restored', 'asked'][
        Number(e.gex_link_state?.() ?? 0)
      ],
      // What restoring the address space cost, when one was restored: regions,
      // bytes re-read from files, bytes copied out of the image, ms in each.
      cost: e.gex_link_cost ? (() => {
        const p = e.gex_alloc(40) >>> 0;
        // Read at a Worker's end, the process possibly torn down under it:
        // an allocation this Worker's view of the memory does not hold is
        // no cost to report, not a reason for the report to throw.
        const b = this.#bytes();
        if (!p || p % 8 || p + 40 > b.byteLength) return null;
        e.gex_link_cost(p);
        const v = new BigUint64Array(b.buffer, p, 5);
        return [Number(v[0]), Number(v[1]), Number(v[2]), Number(v[3]), Number(v[4])];
      })() : null,
    };
  }
  /// The program's argv as the shim has it (after any `execve`).
  get cmdline() {
    const e = this.#shim?.exports;
    if (!e?.gex_cmdline) return '';
    const max = 160;
    const p = e.gex_alloc(max) >>> 0;
    const n = e.gex_cmdline(p, max) >>> 0;
    const s = utf8(this.#bytes().subarray(p, p + n));
    e.gex_free?.(p, max);
    return s;
  }
  /// The guest thread this realm runs: its process, its thread (0 for the
  /// process's first) and the program's argv as it now is.
  get who() {
    return { pid: this.#shim?.exports?.gex_my_pid?.() ?? null, tid: this.#threadTid, cmdline: this.cmdline };
  }
  /// Which objects this process lifted blocks from, `name=count` separated.
  get blocksByObject() {
    const e = this.#shim?.exports;
    if (!e?.gex_jit_by_object) return '';
    const max = 512;
    const p = e.gex_alloc(max) >>> 0;
    const n = e.gex_jit_by_object(p, max) >>> 0;
    const s = utf8(this.#bytes().subarray(p, p + n));
    e.gex_free?.(p, max);
    return s;
  }
  get shimCompileMillis() { return Math.round(this.#shimCompileMs); }
  get shimInstantiateMillis() { return Math.round(this.#shimInstantiateMs); }
  get guestInstantiateMillis() { return Math.round(this.#guestInstantiateMs); }
  /** How many modules that was. */
  get installCount() { return this.#installs; }

  /** Fill requests answered, and requests nobody could answer. */
  get fillStats() {
    return {
      filled: this.#filled,
      unanswered: this.#unanswered,
      // `fillSlotNow` (a directory write and table grow) vs `fillSlot` (an
      // await, which is a whole event-loop turn), separated because neither
      // showed in the other counters.
      now: this.#nowFills,
      // Dense modules this Worker's catch-up left alone, and the ones a miss
      // then instantiated (`#leaveDense`).
      denseLeft: this.#denseLeft,
      denseWanted: this.#denseCatchUps,
      denseAsks: this.#denseAsks,
      denseFillMs: this.#denseFillMs,
      denseFillWorst: this.#denseFillWorst,
      blocksLen: this.#guest?.exports?.blocks?.length ?? 0,
      ftabLen: Number(this.#cgen?.ftab?.length ?? 0),
      nowMs: Math.round(this.#nowMs),
      sysMs: Math.round(Number(this.#shim.exports.gex_sys_us?.() ?? 0n) / 1000),
      sysN: Number(this.#shim.exports.gex_sys_calls?.() ?? 0n),
      faBase: Math.round(this.#faBaseMs),
      faInit: Math.round(this.#faInitMs),
      slow: this.#slowFills,
      slowMs: Math.round(this.#slowMs),
      pub: this.#publishes,
      pubMs: Math.round(this.#publishMs),
      growMs: Math.round(this.#pubGrowMs),
      grew: this.#pubGrewSlots,
      wants: this.#wants,
      wantMs: Math.round(this.#wantMs),
      instMs: Math.round(this.#instMs),
      // What `instMs` is made of: `iGrow` is `Table.grow`, `iComp` is
      // `WebAssembly.compile` (should be ~0 -- a `Module` clones by reference),
      // `iInst` is `new WebAssembly.Instance`.
      iGrow: Math.round(this.#growMs),
      iComp: Math.round(this.#compileMs),
      iInst: Math.round(this.#instantiateMs),
      iGrows: this.#iGrows,
      iAdded: this.#iAdded,
      iCopied: this.#iCopied,
      bigGrows: this.#bigGrows.slice(0, 6).join(' | '),
      tn: this.#tn.join(' '),
      tlen: this.#shim ? this.#dispatchTables().map((t) => t.length).join('/') : '',
      blen: this.#guest?.exports?.blocks?.length ?? -1,
      tailMs: Math.round(this.#tailMs),
      strictGrows: this.#strictGrows,
      strictSlots: this.#strictSlots,
    };
  }

  /**
   * Read a file back out of the guest's filesystem, or null if absent. The
   * other half of bundling: a page that runs a compiler wants the object file.
   */
  readFile(path) {
    const p = new TextEncoder().encode(path);
    const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
    this.#bytes().set(p, pp);
    const len = this.#shim.exports.gex_get_file(pp, p.length) >>> 0;
    if (len === 0) return null;
    const at = this.#shim.exports.gex_file_data() >>> 0;
    return this.#bytes().slice(at, at + len);
  }


  /**
   * Everything the shim asks of JS: console, clock, network, two graphics
   * paths. A method, not an inline object, because `attach` needs the same set.
   */
  #hostImports() {
    return {
        // Whether the other memory flavour's runtime is here (worker.mjs
        // `otherGate`), for an exec of a program of that flavour while it can
        // still fail (vfork.rs `exec_unrunnable`): asked for on first use, and
        // after a failed try, and waited for, then 1, or a negated errno. 1
        // where nothing says.
        // The single-thread build has no gate: the exec waits for the host's
        // promise on its own stack (`GEX_LOAD_OTHER`, host.mjs `#otherLoadFn`).
        other_runtime: (SP_CODE && single.on) ? new WebAssembly.Suspending(() => {
          const load = globalThis.GEX_LOAD_OTHER;
          if (!load || this.#hardNested()) return 1;
          return Promise.race([load(), new Promise((r) => setTimeout(() => r(-110), OTHER_WAIT_MS))]);
        }) : () => {
          const g = globalThis.GEX_OTHER_GATE;
          if (!g || !globalThis.GEX_ASK_OTHER) return 1;
          // A last try that failed is asked again (the host says at once if
          // its backoff has not passed).
          const was = Atomics.load(g, 0);
          if (was === 0 || (was < 0 && Atomics.compareExchange(g, 0, was, 0) === was)) globalThis.GEX_ASK_OTHER();
          // A slow link's few MB, but never for ever.
          for (const until = perf.now() + OTHER_WAIT_MS; perf.now() < until;) {
            const s = Atomics.load(g, 0);
            if (s === 1 || s < 0) return s;
            Atomics.wait(g, 0, s, 1000);
          }
          return -110; // ETIMEDOUT
        },
        // `ptr` is an i64 (a BigInt) from the wasm64 shim, whose thread
        // stacks sit above 4 GiB: cut to 32 bits, a thread's stderr printed
        // as NUL bytes from the wrong place.
        write: (fd, ptr, len) => {
          const at = typeof ptr === 'bigint' ? Number(ptr) : ptr >>> 0;
          this.#env.write(fd, this.#bytes().subarray(at, at + (len >>> 0)));
        },
        // The shim's own narration, to the console -- never to the guest's
        // output, which shares fd 2 with a terminal a person is reading.
        // Off unless GEX_LOG asks, or this process is traced (`--strace`).
        debug: (ptr, len) => (this.#traced || logOn('debug')) && debugLine(
          utf8(
            this.#bytes().subarray(ptr >>> 0, (ptr >>> 0) + (len >>> 0)),
          ).replace(/\n$/, ''),
          this.#traced,
        ),
        warn: (ptr, len) => {
          const text = utf8(
            this.#bytes().subarray(ptr >>> 0, (ptr >>> 0) + (len >>> 0)),
          ).replace(/\n$/, '');
          // `[gex-note]`: worth a line, but nothing went wrong (a large
          // allocation that succeeded): a warning, not an error.
          if (text.startsWith('[gex-note] ')) {
            console.warn(text.slice(11));
            return;
          }
          // `[gex-info] `: one of the few things said by default (a process
          // starting, log.mjs): information, not a warning.
          if (text.startsWith('[gex-info] ')) {
            if (logRoutine()) console.info(`[gex] ${text.slice(11)}`);
            // `?straceexe=NAME`: narrate the process that execs a program
            // of that name, whatever its pid
            const ex = this.#opts?.straceExe && /^pid \d+: exec (\S+)/.exec(text.slice(11));
            if (ex && ex[1].split('/').pop() === this.#opts.straceExe) {
              this.#shim.exports.gex_set_trace(1);
              this.#traced = true;
            }
            return;
          }
          // The shim's own per-library line, said once per session instead
          // (the host's `loaded` line, from `noteStatus`).
          // Said to the host, which says it the first time in the session.
          const dll = /^gex-pe: native wasm32 module for (\S+): /.exec(text);
          if (dll) {
            this.#env.noteStatus?.({ kind: 'module', path: dll[1] });
            if (!logOn('debug')) return;
          }
          // `[gex-stack]`: the shim wants to be told who called it (the wasm
          // frames are named).
          if (text.startsWith('[gex-stack]')) {
            // The run loop turns stack traces off around guest code.
            const limit = Error.stackTraceLimit;
            Error.stackTraceLimit = 40;
            const stack = new Error().stack;
            Error.stackTraceLimit = limit;
            console.error(`${text}\n${stack}`);
          } else {
            console.error(text);
          }
        },
        now: () => this.#env.now(),
        // The clock's hot path (lib.rs `host_now_ns`): an f64 per call, the
        // BigInt origin once per thread. `hostNowNs` is the same sum.
        perf_now: () => perf.now(),
        clock_origin: () => ORIGIN_NS,
        // The function table's fallback (cgen.rs `gex_ftab_fallback`): tier-1
        // runs nested until the guest returns to the function body that
        // called it.
        call_out: (SP_CODE && single.on)
          ? new WebAssembly.Suspending((ctx, target, ret, spWant) => (this.#hardNested()
            ? this.#callOut(ctx >>> 0, target, ret, spWant)
            : this.#callOutAsync(ctx >>> 0, target, ret, spWant)))
          : this.#callOutFront(),
        // An AOT-only fallback whose target this Worker has just installed
        // (cgen.rs `gex_ftab_fallback`): called as a function body.
        // In the single build, on a stack of its own, so the body (and guest
        // code it calls) can block.
        ftab_call: (SP_CODE && single.on)
          ? new WebAssembly.Suspending((idx, ctx, depth, need, ret) => {
            if (this.#unitsDeferred.length) this.#unitWant(idx >>> 0);
            const f = this.#cgen.ftab.get(tIdx(this.#wide, idx >>> 0));
            return this.#hardNested() ? f(ctx, depth, need, ret) : this.#promisingOf(f)(ctx, depth, need, ret);
          })
          : (idx, ctx, depth, need, ret) => {
            if (this.#unitsDeferred.length) this.#unitWant(idx >>> 0);
            return this.#cgen.ftab.get(tIdx(this.#wide, idx >>> 0))(ctx, depth, need, ret);
          },
        // 32-bit Wine: ntdll.so's handler for a guest's Unix call (cgen.rs).
        ftab_call_vi: (idx, arg) => {
          if (this.#unitsDeferred.length) this.#unitWant(idx >>> 0);
          this.#cgen.ftab.get(tIdx(this.#wide, idx >>> 0))(arg);
        },
        // 64-bit Wine: a unit's call to Wine's system or Unix call entry, run by
        // Wine's direct entry as the fault hook runs the `ud2` (cgen.rs
        // `wine64_entry`): its C frames below the guest's, then the units a
        // hook's return serves (a thread the call made: `thread_start`,
        // below, which the shim calls). 0: not here (the
        // single-thread build: a blocking call needs a promising frame).
        // `sp` (an i64) is the guest's stack pointer, which the shim read:
        // the hook's frames go below it as `#hookStack` puts them, without
        // touching the memory here (its buffer and a DataView per call
        // were most of this glue's cost).
        // In wasm (`entryFnModule`): the shim calls the trampoline's export,
        // which goes on to this Worker's entry function, made on the first
        // call. In the single-thread build too: the trampoline and the
        // entry function are wasm, so a call that blocks suspends the
        // fiber's stack through them; only the first call, made from JS
        // (`#entrySlow`), goes by the fault hook instead.
        entry_call: this.#entryTramp(),
        // A session snapshot's question (`snap_stop`), asked only while one
        // is pending: whether this thread's frames below its wait can be
        // unwound to its program's snapshot root (`#snapOk`).
        snap_ok: () => this.#snapOk(),
        // The units and bindings the shim says are waiting, after a direct
        // entry's call (cgen.rs `wine64_entry`, only when there are some).
        cgen_serve: () => this.#cgenServe(),
        // A thread a direct entry's call staged (cgen.rs `wine64_entry` asks
        // only when there is one).
        thread_start: () => this.#startThread(),
        // A guest fault given to the process's fault hook in place (cgen.rs
        // `fault_in_place`), its C frames below the guest's stack pointer
        // as `#hookStack` puts them. A native __EXCEPT's longjmp out of the
        // hook's dispatch passes through here to its own frame. 0: not here
        // (the single-thread build: the staged fault then).
        fault_call: (idx, ctx, code, addr, kind, sp) => {
          if ((SP_CODE && single.on)) return 0;
          const f = this.#cgen.ftab.get(tIdx(this.#wide, idx >>> 0));
          const spG = this.#wasmprogSp ?? this.#wasmprogStack();
          const cur = this.#belowGuest(spG, sp);
          try {
            const a = this.#shim.exports.gex_guest_x86_64?.() === 1 ? BigInt.asUintN(64, addr) : Number(BigInt.asUintN(32, addr)) | 0;
            if (this.#snapHooks) this.#snapHookCall(f, idx >>> 0, ctx, code, a, kind, BigInt.asUintN(64, sp));
            else f(ctx, code, a, kind);
          } finally {
            spG.value = cur;
          }
          return 1;
        },
        // A signal frame the kernel built at a safepoint, run through the
        // process's signal hook (cgen.rs `signal_hook`: native Wine's
        // handlers are wasm functions), its C frames below the guest's stack
        // pointer `sp` as `fault_call`'s are. 0: not here (the single-thread
        // build: the shim stages the frame then, `#signalTake`).
        signal_call: (idx, ctx, pc, sp) => {
          if ((SP_CODE && single.on)) return 0;
          const f = this.#cgen.ftab.get(tIdx(this.#wide, idx >>> 0));
          const spG = this.#wasmprogSp ?? this.#wasmprogStack();
          const cur = this.#belowGuest(spG, sp);
          try {
            f(ctx, this.#signalPc(pc));
          } finally {
            spG.value = cur;
          }
          return 1;
        },
        // Room on this thread's stack for one more fault's dispatch (cgen.rs
        // `fault_in_place`, inside another's).
        stack_room: () => stackRoom(),
        // A dispatch miss installs in place (lib.rs `in_place`).
        jit_install: (_ctx) => this.#jitInstall(),
        // This Worker's function table, short of what another Worker
        // allocated, or this Worker not yet counted (cgen.rs `ftab_resume`).
        ftab_grow: () => this.#cgenGrowIfShort(),
        // An AOT-only run serves units and new threads inside the syscall
        // (lib.rs `serve_now`).
        serve_in_place: () => {
          this.#cgenServe();
          this.#startThread();
        },
        drain_output: () => this.#env.drainOutput?.(),
        // What a system call made through the shim's own export left for JS
        // (lib.rs `syscall_after`), and a native PE DLL's marked call.
        syscall_after: (flags) => this.#syscallAfter(flags),
        native_syscall_marked: (ctx, pc) => this.#nativeSyscall(ctx, pc),
        // The network. Every one of these must return without waiting: there is
        // one wasm stack, and the guest thread is parked by the kernel, not JS.
        // A buffer's address is a full one (a BigInt: the guest's buffer, in
        // place, can be above 4 GiB in a memory64 process); addrNum takes either.
        net_open: () => (this.#net ? this.#net.open() : 0),
        net_connect: (id, host, len, port) => {
          const name = utf8(
            this.#bytes().subarray(host >>> 0, (host >>> 0) + (len >>> 0)),
          );
          this.#net?.connect(id, name, port);
        },
        net_status: (id) => (this.#net ? this.#net.status(id) : 2),
        net_send: (id, ptr, len) =>
          this.#net
            ? this.#net.send(id, this.#bytes().subarray(addrNum(ptr), addrNum(ptr) + (len >>> 0)))
            : -1,
        net_recv: (id, ptr, len) =>
          this.#net
            ? this.#net.recv(id, this.#bytes().subarray(addrNum(ptr), addrNum(ptr) + (len >>> 0)))
            : 0,
        // `recv(MSG_PEEK)`: -2 (an error) from a network without it.
        net_peek: (id, ptr, len) =>
          this.#net?.peek
            ? this.#net.peek(id, this.#bytes().subarray(addrNum(ptr), addrNum(ptr) + (len >>> 0)))
            : this.#net ? -2 : 0,
        // A UDP flow's next datagram, framed, whole or nothing; taken unless
        // `take` is 0. A network without `frame` (one Worker's own, as the
        // Node lane's) is read by peeks, nothing else reading it.
        net_frame: (id, ptr, len, take) => {
          if (!this.#net) return 0;
          const out = this.#bytes().subarray(addrNum(ptr), addrNum(ptr) + (len >>> 0));
          if (this.#net.frame) return this.#net.frame(id, out, take !== 0);
          if (!this.#net.peek) return -2;
          return frameOf((o) => this.#net.peek(id, o), (o) => this.#net.recv(id, o), out, take !== 0);
        },
        net_avail: (id) => (this.#net ? (this.#net.available(id) ? 1 : 0) : 1),
        net_close: (id) => this.#net?.close(id),
        // Another holder of a stream (a socket passed to another process).
        net_dup: (id) => (this.#netShared && this.#net?.dup ? this.#net.dup(id) : 0),
        // Graphics: the guest's GL library batched commands in its own memory;
        // this hands over the address and the page runs them on Mesa.
        // The token as well as the ring: the host keeps a GL context per
        // process, because the object names in the stream are this process's
        // own and another process hands out the same ones.
        // In the single build a promise (the compositor's Mesa runs in this
        // realm), so the import suspends; never from a nested call-out.
        // `ring` is an f64, whole in a 64-bit memory (gl.rs).
        gl_submit: (SP_CODE && single.on) ? new WebAssembly.Suspending((ring) =>
          (this.#gl
            ? this.#gl.submit(this.#memory, ring, this.wlToken(), false, this.#glWaitNote)
            : -1))
          : (ring) =>
            (this.#gl
              ? this.#gl.submit(this.#memory, ring, this.wlToken(), false, this.#glWaitNote)
              : -1),
        // The same, not waited for (a frame's last batch): the copy is taken
        // and the guest draws on; the next submit waits for this one, and
        // says how long through `gex_gl_note_wait_ns`.
        gl_submit_async: (SP_CODE && single.on) ? new WebAssembly.Suspending((ring) =>
          (this.#gl
            ? this.#gl.submit(this.#memory, ring, this.wlToken(), true, this.#glWaitNote)
            : -1))
          : (ring) =>
            (this.#gl
              ? this.#gl.submit(this.#memory, ring, this.wlToken(), true, this.#glWaitNote)
              : -1),
        // This pump's scene: `count` surface records at `ptr` in this memory,
        // saying where each surface's pixels are rather than carrying them. The
        // compositor reads those addresses out of this same memory, so a frame
        // is never copied on its way to the screen.
        wl_publish: (ptr, count) =>
          this.#wl?.publish(this.#memory, ptr >>> 0, count >>> 0),
        // The reverse: drain this process's input and control ring into `ptr`,
        // so the compositor pump can deliver each event to the client and
        // surface the compositor's own hit test picked.
        wl_poll_input: (ptr, max) =>
          this.#wl?.pollInput(this.#memory, ptr >>> 0, max >>> 0) ?? 0,
        // Whether that ring holds anything, without draining it.
        wl_input_pending: () => (this.#wl?.inputPending() ? 1 : 0),
        // The output's size now, packed (`WlProxy.outputSize`), 0 for none:
        // the pump asks every turn and tells its clients when it changed.
        wl_output_size: () => this.#wl?.outputSize?.() ?? 0,
        // The page's drag of host files: its data at the drop, and what the
        // client says of it, for the page (wl-bridge.mjs `waitDrag`).
        wl_drag_data: (ptr, max) => this.#wl?.dragData(this.#memory, ptr >>> 0, max >>> 0) ?? 0,
        wl_drag_status: (serial, flags) => this.#wl?.dragStatus(serial >>> 0, flags >>> 0),
        wl_clip_serial: () => this.#wl?.clipSerial?.() ?? 0,
        wl_clip_get: (ptr, max) => this.#wl?.clipGet?.(this.#memory, ptr >>> 0, max >>> 0) ?? 0,
        wl_clip_set: (ptr, len, from) => this.#wl?.clipSet?.(this.#memory, ptr >>> 0, len >>> 0, from >>> 0) ?? 0,
        wl_clip_from: () => this.#wl?.clipFrom?.() ?? 0,
    };
  }

  /**
   * `shimWasm` is hosts/browser's build output, `guestWasm` a module from
   * `gexc build --exhaustive --no-shared-memory`, `elf` the program's bytes.
   *
   * `files` is a map of guest path to contents. A browser has no filesystem to
   * pass through to, so whatever the program needs to read is bundled here.
   */
  async load(shimWasm, guestWasm, elf, files = {}, args = [], opts = {}) {
    // A 32-bit Wine program (native Wine's loader) splits the memory with Windows
    // before the runtime's startup heap is taken, or that heap sits where
    // Windows DLLs load (0x10000000). A 64-bit one in a 32-bit memory takes no space here unless
    // the site asks for the split: its load picks the compact space or the
    // split (wasmprog.rs `publish`), and a space taken here first would be
    // 32-bit Wine's.
    const wine = wine32ProgramFlags(elf);
    if (wine && ((wine & 2) === 0 || this.#wide || globalThis.GEX_WINE32_SPLIT)) this.#windowsSpace = -1;
    this.#wine64Split = (wine & 2) !== 0 && this.#wide;
    // Not one whose header asks for Windows space: it runs translated code,
    // which the placement window serves.
    this.#noPlace = (wasmProgramFlags(elf) & 3) === 0 ? wasmProgramImage(elf) : null;
    // A compiled `WebAssembly.Module` if the host has one, bytes otherwise:
    // a module is structured-cloneable and the engine shares the code behind
    // it, so passing one saves recompiling the 8.7 MB shim per Worker.
    const shimMod =
      shimWasm instanceof WebAssembly.Module ? shimWasm : await WebAssembly.compile(shimWasm);
    await this.#startKernel();
    this.#shim = await WebAssembly.instantiate(shimMod, {
      env: { memory: this.#memory },
      gex_env: this.#hostImports(),
      gex_k: this.#kernelImports(shimMod),
    });
    noteM64Tables(this.#shim.exports);
    this.#u8 = null;
    // First, before any call into the shim: `gex_worker_init` gives this Worker
    // its thread-local block, and until it runs `__tls_base` is zero, so a shim
    // function touching a thread-local writes through a null base.
    this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
    // An older shim loads but fails on the first call to something it lacks,
    // far from the mistake. Named here so the skew is a sentence, not a mystery.
    const required = [
      'gex_alloc', 'gex_load', 'gex_add_file', 'gex_push_stdin', 'gex_wake',
      // Each of these is called through `?.` at its use site so a shim that
      // predates it still loads -- which is how a stale runtime turns into a
      // silent wrong answer (no `gex_chmod` leaves every file 0644).
      'gex_chmod', 'gex_add_dir', 'gex_add_env', 'gex_add_symlink',
      'gex_paused_for_input', 'gex_paused_for_net', 'gex_paused_for_host',
      'gex_park', 'gex_resume_pc',
      'gex_resume_ctx', 'gex_exit_code', 'gex_jit_pending_len',
    ];
    const missing = required.filter((n) => typeof this.#shim.exports[n] !== 'function');
    if (missing.length) {
      throw new Error(
        `the gex runtime is out of date: it has no ${missing.join(', ')}. ` +
          'Rebuild it with page/build-assets.sh and reload.',
      );
    }
    // Whatever `useGl` was told before the shim existed.
    this.#shim.exports.gex_set_gl?.(this.#gl ? 1 : 0);
    this.#shim.exports.gex_set_d3d?.(this.#d3d ? 1 : 0);
    // And `useCgen`, before the process boots: the exe and ntdll are mapped
    // there, and their units are looked up then.
    if (this.#cgenMode) this.#shim.exports.gex_cgen_set_mode?.(this.#cgenMode);
    // `GEX_LOG=serveall`: units served after every direct entry's call, as
    // before cgen.rs `serve_wanted` (A/B runs).
    if (logOn('serveall')) this.#shim.exports.gex_cgen_serve_always?.(1);
    // For gex-eject's 32-bit Wine recorder: code a unit's image ran translated.
    if (logOn('record')) this.#shim.exports.gex_cgen_set_record?.(1);
    if (this.#noElfProviders) this.#shim.exports.gex_elfprov_off?.();
    this.#libprovConfig();
    this.#shim.exports.gex_set_wl?.(this.#wl ? 1 : 0);
    if (this.#wlOut) this.#shim.exports.gex_set_wl_output?.(...this.#wlOut);
    if (this.#wlOut) this.#shim.exports.gex_set_wl_scale?.(this.#wlScale);
    // `kernel.net` is decided inside `gex_load` from this flag, so it must be
    // set before it (not merely in `useNetwork`, before the shim exists).
    this.#shim.exports.gex_set_network?.(this.#net ? 1 : 0);
    // The processors the guest sees: each of its threads is a Worker, so the
    // host's count (`opts.cpus` overrides it).
    this.#shim.exports.gex_set_cpus?.(opts.cpus || globalThis.navigator?.hardwareConcurrency || 1);
    // Narrate one page's directory writes, in every process.
    if (Gex.watchDir >= 0) this.#shim.exports.gex_watch_dir?.(BigInt(Gex.watchDir));
    // wasm-ld puts the shim's stack at address zero (the guest's image); move
    // it into the shim's own region, above everything the guest maps.
    this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
    // Carried into every fork child's Worker (see the fork record's `opts`), so
    // a debugging switch set for the run holds for the processes it spawns --
    // the block trace of a child that crashes is otherwise silent.
    this.#opts = {
      strace: opts.strace,
      // Which single process to narrate, if any. Inherited like `strace` --
      // the interesting process is usually a child, and a whole session's
      // syscalls is more output than a page survives.
      stracePid: opts.stracePid,
      straceExe: opts.straceExe,
      parkReport: opts.parkReport,
      tty: opts.tty, uid: opts.uid, gid: opts.gid,
      traceBlocks: opts.traceBlocks, traceMisses: opts.traceMisses, checkSp: opts.checkSp,
      dlChecked: opts.dlChecked,
      jitDiag: opts.jitDiag,
      watchMem: opts.watchMem,
      noIntrinsics: opts.noIntrinsics, intrinsicMask: opts.intrinsicMask,
      // The translator's knobs, for every process of the run: set only here,
      // a forked child (node under `bash -c`) ran with the defaults.
      liftMax: opts.liftMax, inlineDispatch: opts.inlineDispatch,
      xlateReport: opts.xlateReport, discoverSkip: opts.discoverSkip, t1Fold: opts.t1Fold,
    };
    syncSlotShift(this.#shim.exports);
    // `?stracepid=N` for a process booted here, not forked or exec'd into
    // (`#childOpts` sees those): pid 1, or what the kernel says it is.
    const bootPid = this.#shim.exports.gex_my_pid?.() || 1;
    if (opts.parkReport) this.#shim.exports.gex_set_park_report?.(opts.parkReport);
    if (opts.stracePid ? opts.stracePid === bootPid : opts.strace) { this.#shim.exports.gex_set_trace(1); this.#traced = true; }
    // An ejected project without a JIT fallback: a miss on code its build
    // never saw fails loudly instead of translating.
    if (opts.jit === false) {
      if (!this.#shim.exports.gex_set_jit) throw new Error('this gex runtime cannot turn translation off');
      this.#shim.exports.gex_set_jit(0);
    }
    // How many blocks one dispatch miss lifts (default 2048).
    if (opts.liftMax) this.#shim.exports.gex_set_lift_max?.(opts.liftMax);
    if (opts.inlineDispatch) this.#shim.exports.gex_set_inline_dispatch?.(1);
    if (opts.xlateReport) this.#shim.exports.gex_set_xlate_report?.(opts.xlateReport);
    if (opts.discoverSkip) this.#shim.exports.gex_set_discover_skip?.(opts.discoverSkip);
    if (opts.t1Fold) this.#shim.exports.gex_set_t1_fold?.(1);
    // Keep what the guest's dynamic linker produces, and start the next process
    // of the same program from it instead of linking again.
    if (opts.link) this.#shim.exports.gex_set_snapshot?.(Number(opts.link));
    // A terminal changes how libc buffers and whether a shell prints a prompt.
    if (opts.tty) this.#shim.exports.gex_set_tty(1);
    // A page has no user; an embedder that has one says so.
    if (opts.uid !== undefined) this.#shim.exports.gex_set_identity(opts.uid, opts.gid ?? opts.uid);

    // The module describes its own dispatch in a custom section; runtime
    // translation needs it to carry on allocating slots where the build stopped.
    this.#setMetadata(guestWasm);
    // Where the guest starts, for a program given relative paths.
    if (opts.cwd) {
      const c = new TextEncoder().encode(opts.cwd);
      const cp = this.#shim.exports.gex_alloc(c.length) >>> 0;
      this.#bytes().set(c, cp);
      this.#shim.exports.gex_set_cwd(cp, c.length);
    }
    // Standard input, if the embedder has any: a page has nothing to block
    // on, so the guest is given all of it up front.
    if (opts.stdin && opts.stdin.length) {
      const sp = this.#shim.exports.gex_alloc(opts.stdin.length) >>> 0;
      this.#bytes().set(opts.stdin, sp);
      this.#shim.exports.gex_set_stdin(sp, opts.stdin.length);
    }
    // The bundle goes over before the program does: a dynamic executable names
    // an interpreter the loader reads out of this filesystem. Freed one file at
    // a time -- `gex_add_file` copies, so the buffer is dead once it returns,
    // and keeping it doubled a rootfs's cost in this process's memory.
    const enc = new TextEncoder();
    const free = this.#shim.exports.gex_free;
    // Empty directories first, so a file arriving under one finds it.
    // `gex_add_file` makes directories on the way to each file, so an empty one
    // would otherwise arrive not at all; `page/src/tar.js` collects them.
    for (const path of opts.dirs ?? []) {
      const p = enc.encode(path);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#shim.exports.gex_add_dir?.(pp, p.length);
      free?.(pp, p.length);
    }
    // Which of them are executable: the VFS makes every file 0644 and the tar's
    // mode did not travel with its bytes, so programs read as not executable.
    const modes = opts.modes ?? {};
    // The environment, when the page has one in mind. Empty leaves the shim's
    // default (Debian's PATH for root plus HOME).
    for (const kv of opts.env ?? []) {
      const e = enc.encode(kv);
      const ep = this.#shim.exports.gex_alloc(e.length) >>> 0;
      this.#bytes().set(e, ep);
      this.#shim.exports.gex_add_env?.(ep, e.length);
      free?.(ep, e.length);
    }
    // A large file goes over in pieces through one buffer: the shim's heap
    // never gives a span back, so a file-sized buffer would be held for the
    // process's life (and a Windows process loses that address space).
    const PART = 16 << 20;
    let partBuf = 0;
    for (const [path, contents] of Object.entries(files)) {
      const p = enc.encode(path);
      const d = contents instanceof Uint8Array ? contents : enc.encode(contents);
      if (d.length > PART && this.#shim.exports.gex_add_file_part) {
        const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
        this.#bytes().set(p, pp);
        if (!partBuf) partBuf = this.#shim.exports.gex_alloc(PART) >>> 0;
        for (let off = 0; off < d.length; off += PART) {
          const piece = d.subarray(off, Math.min(off + PART, d.length));
          this.#bytes().set(piece, partBuf);
          this.#shim.exports.gex_add_file_part(pp, p.length, BigInt(off), partBuf, piece.length);
        }
        if (modes[path] !== undefined) this.#shim.exports.gex_chmod?.(pp, p.length, modes[path]);
        free?.(pp, p.length);
        continue;
      }
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      const dp = this.#shim.exports.gex_alloc(d.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#bytes().set(d, dp);
      this.#shim.exports.gex_add_file(pp, p.length, dp, d.length);
      // Optional so an older shim still loads.
      if (modes[path] !== undefined) {
        this.#shim.exports.gex_chmod?.(pp, p.length, modes[path]);
      }
      free?.(pp, p.length);
      free?.(dp, d.length);
    }
    if (partBuf) free?.(partBuf, PART);
    // Directories' modes last, once their files are in (a read-only one
    // keeps its mode: Wine reports it read only).
    for (const path of opts.dirs ?? []) {
      if (modes[path] === undefined) continue;
      const p = enc.encode(path);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#shim.exports.gex_chmod?.(pp, p.length, modes[path]);
      free?.(pp, p.length);
    }
    // Files fetched as they are read (`gex_add_lazy`): [path, size, url,
    // head], after the bundle's, which win -- unless the bundle's is the
    // lazy file's head (`head` bytes of it: its first chunks, maybe its last
    // one too), which the kernel then holds as those chunks.
    for (const [path, size, url, head] of opts.lazy ?? []) {
      if (!this.#shim.exports.gex_add_lazy) break;
      if (files[path] !== undefined && !(head && files[path].length === head)) continue;
      const p = enc.encode(path);
      const u = enc.encode(url);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      const up = this.#shim.exports.gex_alloc(u.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#bytes().set(u, up);
      this.#shim.exports.gex_add_lazy(pp, p.length, up, u.length, BigInt(size));
      if (modes[path] !== undefined) this.#shim.exports.gex_chmod?.(pp, p.length, modes[path]);
      free?.(pp, p.length);
      free?.(up, u.length);
    }
    // Hard links (`page/src/tar.js`), after their targets: one inode with two
    // names, as the host had it. A shim that cannot link gets a copy.
    for (const [path, target] of Object.entries(opts.hard ?? {})) {
      const p = enc.encode(path);
      const t = enc.encode(target);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      const tp = this.#shim.exports.gex_alloc(t.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#bytes().set(t, tp);
      const linked = this.#shim.exports.gex_add_hardlink?.(tp, t.length, pp, p.length) ?? 0;
      const d = files[target];
      if (!linked && d) {
        const dp = this.#shim.exports.gex_alloc(d.length) >>> 0;
        this.#bytes().set(d, dp);
        this.#shim.exports.gex_add_file(pp, p.length, dp, d.length);
        free?.(dp, d.length);
      }
      if (modes[target] !== undefined) this.#shim.exports.gex_chmod?.(pp, p.length, modes[target]);
      free?.(pp, p.length);
      free?.(tp, t.length);
    }
    // Symbolic links (`page/src/tar.js`'s `links`), after files and hard links:
    // path -> target text. Only with the global kernel (a page or run-node):
    // a process that is the whole system has no filesystem yet to link in,
    // so a caller without one passes copies as files instead.
    for (const [path, target] of Object.entries(opts.symlinks ?? {})) {
      const p = enc.encode(path);
      const t = enc.encode(target);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      const tp = this.#shim.exports.gex_alloc(t.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#bytes().set(t, tp);
      if (!this.#shim.exports.gex_add_symlink(tp, t.length, pp, p.length)) {
        console.warn(`gex: symlink ${path} -> ${target} not made (no global kernel)`);
      }
      free?.(pp, p.length);
      free?.(tp, t.length);
    }
    // The bundle's file times (seconds since the epoch), as the host had them:
    // a program that records one -- gzip's header, make's rebuild test --
    // sees what it would natively rather than 1970.
    for (const [path, secs] of Object.entries(opts.mtimes ?? {})) {
      if (!this.#shim.exports.gex_set_mtime) break;
      const p = enc.encode(path);
      const pp = this.#shim.exports.gex_alloc(p.length) >>> 0;
      this.#bytes().set(p, pp);
      this.#shim.exports.gex_set_mtime(pp, p.length, BigInt(Math.trunc(secs)));
      free?.(pp, p.length);
    }
    // The bundle's extended attributes (`page/src/tar.js`'s `xattrs`, path to
    // { name: bytes }): Wine's DOS attributes, hidden and system, on the
    // prefix's files and folders.
    for (const [path, attrs] of Object.entries(opts.xattrs ?? {})) {
      if (!this.#shim.exports.gex_set_xattr) break;
      for (const [name, value] of Object.entries(attrs)) {
        const p = enc.encode(path);
        const n = enc.encode(name);
        const v = value instanceof Uint8Array ? value : enc.encode(String(value));
        const pp = this.#shim.exports.gex_alloc(p.length + n.length + v.length) >>> 0;
        this.#bytes().set(p, pp);
        this.#bytes().set(n, pp + p.length);
        this.#bytes().set(v, pp + p.length + n.length);
        this.#shim.exports.gex_set_xattr(pp, p.length, pp + p.length, n.length, pp + p.length + n.length, v.length);
        free?.(pp, p.length + n.length + v.length);
      }
    }

    // Which program the image module was built from, so the shim can refuse to
    // hand its pages to a process running something else. See
    // `image_pages_are_ours`.
    if (opts.program && this.#shim.exports.gex_set_image_program) {
      const b = enc.encode(opts.program);
      const bp = this.#shim.exports.gex_alloc(b.length) >>> 0;
      this.#bytes().set(b, bp);
      this.#shim.exports.gex_set_image_program(bp, b.length);
      free?.(bp, b.length);
    }
    // The program's own path, when the embedder knows it (an ejected
    // project): what `/proc/self/exe` says, whatever argv[0] is.
    if (opts.exe) {
      if (!this.#shim.exports.gex_set_exe) throw new Error('this gex runtime cannot be told the program path');
      const b = enc.encode(opts.exe);
      const bp = this.#shim.exports.gex_alloc(b.length) >>> 0;
      this.#bytes().set(b, bp);
      this.#shim.exports.gex_set_exe(bp, b.length);
      free?.(bp, b.length);
    }
    const pin = this.#shim.exports.gex_pin_object;
    for (const [path, where] of Object.entries(opts.pins ?? {})) {
      if (!pin) break;
      const b = enc.encode(path);
      const bp = this.#shim.exports.gex_alloc(b.length) >>> 0;
      this.#bytes().set(b, bp);
      // The span as well as the base: the placement table needs to know how
      // much of the address space a pinned object owns, or it hands the next
      // one an address inside it.
      pin(bp, b.length, BigInt(where.base ?? where), BigInt(where.span ?? 0));
      free?.(bp, b.length);
    }

    const ptr = this.#shim.exports.gex_alloc(elf.length) >>> 0;
    this.#bytes().set(elf, ptr);
    const argvBlob = new TextEncoder().encode(args.join('\0'));
    let ap = 0;
    if (argvBlob.length) {
      ap = this.#shim.exports.gex_alloc(argvBlob.length) >>> 0;
      this.#bytes().set(argvBlob, ap);
    }
    // Phase markers, so a boot that never finishes says where it stopped.
    if (globalThis.GEX_DEBUG_EXIT) {
      console.error(`[gex] load: ${Object.keys(opts.pins ?? {}).length} pins, `
        + `${Object.keys(files).length} files staged, loading the program`);
    }
    const entry = this.#shim.exports.gex_load(ptr, elf.length, ap, argvBlob.length);
    if (entry === 0n) throw new Error('gex: could not load the program');
    // `?xcache=`: the store the last load saved, laid in before anything runs.
    // After `gex_load`, which unpacked the bundle the
    // saved paths are checked against.
    if (opts.xcache) await this.#xcacheRestore(String(opts.xcache));
    if (globalThis.GEX_DEBUG_EXIT) {
      console.error(`[gex] load: program at ${entry}, `
        + `${(opts.closure ?? []).length} closure modules to install`);
    }

    // The rest of the closure: one module per shared library, translated ahead
    // of time by the same `gexc` run as the image. Held until now because the
    // image must be placed first (its metadata says how far the tables reach)
    // and `#installClosure` below builds the tables the modules import.
    this.#closureModules = opts.closure ?? [];
    this.#wantLibrary = opts.wantLibrary ?? this.#wantLibrary;


    const e = this.#shim.exports;
    this.#imports = {
      memory: this.#memory,
      // An embedder may vet each system call first (an ejected project's
      // allow-list); it throws to stop the run.
      syscall: opts.syscallGuard
        ? (ctx, pc) => {
          opts.syscallGuard(ctx >>> 0, pc, this.#memory);
          return e.gex_syscall(ctx, pc);
        }
        : this.#guestSyscall(e),
      dispatch_miss: e.gex_dispatch_miss,
      illegal: e.gex_illegal,
      breakpoint: e.gex_breakpoint,
      sysreg_read: e.gex_sysreg_read,
      // An ejected project's library replacements (`Intrinsic::Replace`,
      // sysreg -200) are served by the embedder; everything else by the shim.
      sysreg_write: opts.replace
        ? (ctx, pc, reg, v) => (reg === -200
          ? opts.replace(ctx >>> 0, pc, Number(v), this.#memory, this.#imports)
          : e.gex_sysreg_write(ctx, pc, reg, v))
        : e.gex_sysreg_write,
      cache_op: e.gex_cache_op,
      safepoint: (SP_CODE && single.on) ? this.#singleSafepoint(e) : e.gex_safepoint,
      fma: e.gex_fma,
      x87: e.gex_x87,
      mx: e.gex_mx,
      // A counting unit's table calls (cgen.rs `gex_icall_note`).
      icall_note: e.gex_icall_note,
    };
    const tGuest = performance.now();
    this.#guest = await WebAssembly.instantiate(
      guestWasm instanceof WebAssembly.Module ? guestWasm : await WebAssembly.compile(guestWasm),
      { gex: this.#imports },
    );
    this.#guestInstantiateMs = performance.now() - tGuest;
    this.#tnAt('boot');
    // Dense block tables when the image module has `blocks` (the seed does;
    // a closure image, whose libraries dispatch classically, does not). A
    // native wasm program numbers pages too, for the units its libraries
    // call through, and those numbers are shared system-wide: classic, its
    // table would reach the highest any process numbered.
    const denseAsked = opts.dense !== false && !!this.#guest.exports.blocks && !!e.gex_set_dense;
    this.#dense = denseAsked;
    this.#opts.dense = denseAsked ? (opts.dense === 2 ? 2 : 1) : false;
    e.gex_set_dense?.(denseArg(this.#opts, this.#dense));
    if (TEST_KEEP_PROBE) e.gex_test_keep_probe?.(1);
    this.#entriesInit();
    // A test-only switch: a retired page
    // left numbered with no directory entry.
    if (globalThis.GEX_TEST_UNPUBLISHED_RETIRE || globalThis.process?.env?.GEX_TEST_UNPUBLISHED_RETIRE) e.gex_test_unpublished_retire?.(1);
    if (globalThis.GEX_TEST_FORK_REBASE || globalThis.process?.env?.GEX_TEST_FORK_REBASE) e.gex_test_fork_rebase?.(1);
    // Another test-only switch: stale words past a
    // module's last block.
    if (globalThis.process?.env?.GEX_TEST_DENSE_HOLES) e.gex_test_dense_holes?.(1);
    // Another test-only switch: served modules are
    // installed one at a time, as when the merger declines them.
    if (globalThis.GEX_TEST_SERVE_UNMERGED || globalThis.process?.env?.GEX_TEST_SERVE_UNMERGED) e.gex_test_serve_unmerged?.(1);
    // The function JIT (fjit.rs): its translations reach the function table,
    // which the function tier's plumbing drives (units where any exist).
    // Carried into every child (the fork and exec records' `opts`, `#childOpts`):
    // a Wine program runs in an exec'd process, never in pid 1.
    Object.assign(this.#opts, { fjit: opts.fjit, fjitMax: opts.fjitMax, fjitOnly: opts.fjitOnly });
    this.#fjitOn(e, opts);
    this.#offerToFill(this.#guest);
    if (globalThis.GEX_DEBUG_EXIT) console.error('[gex] load: instantiated, installing');
    await this.#installClosure();
    if (globalThis.GEX_DEBUG_EXIT) console.error('[gex] load: done');
    return entry;
  }

  /**
   * Take one module a fork handed over, without instantiating it yet. A child
   * branches into almost none of what it inherits, so the directory entries are
   * published now but the instance waits for a branch, as a closure library
   * does. Without metadata there is nothing to key a later fill on, so those
   * install the old way (as does the image).
   */
  async #adoptModule(m) {
    const e = this.#shim.exports;
    // Its dense blocks keep the parent's numbering: point this process's
    // slots at them where nothing newer does.
    // `m.dense`: the parent's own numbering (`#compiled`); the Module's
    // section may be another process's (the unit cache).
    const ds = this.#dense ? (m.dense ?? denseSection(m.mod)) : null;
    if (globalThis.GEX_DENSE_DEBUG && ds && m.dense) {
      const own = denseSection(m.mod);
      if (own && new DataView(own.buffer, own.byteOffset).getUint32(0, true) !== new DataView(ds.buffer, ds.byteOffset).getUint32(0, true)) {
        console.error(`[dense] adopt: a module compiled elsewhere (base ${new DataView(own.buffer, own.byteOffset).getUint32(0, true)}) at this process's base ${new DataView(ds.buffer, ds.byteOffset).getUint32(0, true)}`);
      }
    }
    if (ds && e.gex_dense_adopt) {
      const dp = e.gex_alloc(ds.length) >>> 0;
      this.#bytes().set(ds, dp);
      e.gex_dense_adopt(dp, ds.length);
      e.gex_free?.(dp, ds.length);
    }
    // What this module actually reaches: a closure library claims zero here.
    // See `slotsNeeded`.
    const truly = Math.max(m.need >>> 0, slotsNeeded(m.meta));
    if (!m.meta || !m.meta.length || !e.gex_add_lazy_module) {
      await this.#installModule(m.mod, truly, 64, m.meta, 'adopt-nometa', null, false, m.dense);
      return;
    }
    const gp = e.gex_alloc(m.meta.length) >>> 0;
    this.#bytes().set(m.meta, gp);
    const need = e.gex_add_lazy_module(gp, m.meta.length) >>> 0;
    e.gex_free?.(gp, m.meta.length);
    if (!need) {
      await this.#installModule(m.mod, truly, 64, m.meta, 'adopt-noneed', null, false, m.dense);
      return;
    }
    this.#growTables(Math.max(need, truly), 'adopt');
    this.#lazyInherited.push({ mod: m.mod, need: truly, inst: null, meta: m.meta, dense: m.dense });
  }

  /**
   * Install one closure library, arriving after the process is already up.
   *
   * The whole closure cannot be handed to a Worker at once (694 MB of
   * `WebAssembly.Module` overflows `postMessage`), so they come one at a time
   * and the guest runs while they arrive -- safe because the miss path forgives
   * a page whose module has not landed and asks again when it does.
   */
  async installClosureModule(entry) {
    if (!this.#shim || !entry) return false;
    const meta = entry instanceof Uint8Array ? metadataSection(entry) : entry.meta;
    if (!meta || !meta.length) return false;
    const add = this.#shim.exports.gex_add_closure_module;
    if (!add) return false;
    const gp = this.#shim.exports.gex_alloc(meta.length) >>> 0;
    this.#bytes().set(meta, gp);
    const need = add(gp, meta.length) >>> 0;
    this.#shim.exports.gex_free?.(gp, meta.length);
    if (!need) return false;
    const mod = entry instanceof Uint8Array ? entry : entry.mod;
    // Registered and reserved, not instantiated: the directory points at this
    // library's pages and the tables are grown, so a branch resolves and
    // faults. `fillSlot` builds the instance when a branch actually wants one
    // -- most of a closure's libraries are never reached.
    this.#growTables(need, 'closure');
    this.#lazyClosure.push({ mod, need, inst: null });
    this.#streamed++;
    return true;
  }

  async #installClosure() {
    const add = this.#shim.exports.gex_add_closure_module;
    if (!add) return;
    // The highest slot total any of them needs, so the tables grow once.
    let most = 0;
    let other = 0;
    for (const [i, entry] of this.#closureModules.entries()) {
      // A module of the process's other mode (an ejected project's `.other`
      // module, sent with its bytes): installed now, as a translation is,
      // into the other directory (`gex_queue_other_mode_module`).
      if (entry?.otherMode && entry.bytes && this.#queueOtherMode(entry)) {
        other++;
        continue;
      }
      // Either the raw wasm, or the pair a page should send: the metadata
      // section alone plus the already-compiled module. A `Module` crosses
      // `postMessage` by reference, so the pair avoids cloning and recompiling
      // hundreds of megabytes of closure per process.
      const paired = !(entry instanceof Uint8Array);
      const wasm = paired ? null : entry;
      const meta = paired ? entry.meta : metadataSection(entry);
      let need = 0;
      if (meta && meta.length) {
        const t0 = performance.now();
        const gp = this.#shim.exports.gex_alloc(meta.length) >>> 0;
        this.#bytes().set(meta, gp);
        const tCopied = performance.now();
        // With the program this library is the image of, when it is one: every
        // root of a closure claims `PIE_BASE`, so they are told apart by program.
        const prog = new TextEncoder().encode(paired ? (entry.program ?? '') : '');
        let pp = 0;
        if (prog.length) {
          pp = this.#shim.exports.gex_alloc(prog.length) >>> 0;
          this.#bytes().set(prog, pp);
        }
        need = add(gp, meta.length, pp, prog.length) >>> 0;
        if (pp) this.#shim.exports.gex_free?.(pp, prog.length);
        const tAdded = performance.now();
        this.#shim.exports.gex_free?.(gp, meta.length);
        if (globalThis.GEX_DEBUG_EXIT) {
          console.error(
            `[closure] library ${i}: ${(meta.length / 1024) | 0} KiB of metadata, `
            + `${Math.round(tCopied - t0)}ms copying, `
            + `${Math.round(tAdded - tCopied)}ms registering, ${need} slots`,
          );
        }
      }
      // `need` is 0 now: registration reserves and publishes nothing. The
      // metadata is kept because `fillSlot` publishes from it. Registered and
      // reserved, not instantiated -- `fillSlot` builds an instance when a
      // branch wants a page it covers, and most libraries are never reached.
      void need;
      most = Math.max(most, need);
      this.#lazyClosure.push({
        mod: paired ? entry.mod : wasm,
        // From the metadata, not `need` (which registration returns as zero).
        // `#publish` grows the tables when a page is first wanted.
        need: Math.max(need, slotsNeeded(meta)),
        inst: null,
        meta,
        published: false,
      });
    }
    // Nothing is grown here: `#publish` grows for a library when a branch asks
    // for one of its pages, instead of growing to the whole closure up front.
    void most;
    this.#closureModules = [];
    if (other) {
      const e = this.#shim.exports;
      const need = (e.gex_jit_pending_slots?.() ?? 0) >>> 0;
      if (need) this.#growTables(need, 'closure');
      e.gex_jit_pending_refresh?.();
      while (e.gex_jit_pending_len() >>> 0) this.#installTranslationSync();
    }
  }

  /** Queue an other-mode closure module ({ meta, bytes }) in the shim: true
   *  when it was one. */
  #queueOtherMode(entry) {
    const e = this.#shim.exports;
    if (!e.gex_queue_other_mode_module || !entry.meta?.length) return false;
    const { meta, bytes } = entry;
    const mp = e.gex_alloc(meta.length) >>> 0;
    this.#bytes().set(meta, mp);
    const wp = e.gex_alloc(bytes.length) >>> 0;
    this.#bytes().set(bytes, wp);
    const queued = e.gex_queue_other_mode_module(mp, meta.length, wp, bytes.length) >>> 0;
    e.gex_free?.(wp, bytes.length);
    e.gex_free?.(mp, meta.length);
    return queued !== 0;
  }

  /** The kernel, shim and guest instances on this Worker's memory: what
   *  `attach` starts with, and what `preattach` does ahead of a process. */
  async #instantiate(shimWasm, guestWasm, _at = () => {}, { stack = true } = {}) {
    await this.#startKernel();
    _at('kernel started');
    // A compiled `Module` if the host has one, bytes otherwise: passing a
    // module saves recompiling the shim per Worker. See `load`.
    const tShimC = performance.now();
    const shimMod =
      shimWasm instanceof WebAssembly.Module ? shimWasm : await WebAssembly.compile(shimWasm);
    this.#shimCompileMs = performance.now() - tShimC;
    const tShimI = performance.now();
    this.#shim = await WebAssembly.instantiate(shimMod, {
      env: { memory: this.#memory },
      gex_env: this.#hostImports(),
      gex_k: this.#kernelImports(shimMod),
    });
    this.#shimInstantiateMs = performance.now() - tShimI;
    noteM64Tables(this.#shim.exports);
    _at('shim instantiated');
    this.#u8 = null;
    // The stack pointer is a per-instance global, so every Worker has its own
    // and must be told where it goes. Not yet for a runtime made ahead of
    // its process (`preattach`): the stack is the arena's first claim, and
    // how the arena is laid out is the process's (its start message's
    // `?winesplit=1`, which arrives after, and `#noPlace`).
    if (stack) this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
    const e = this.#shim.exports;
    this.#imports = {
      memory: this.#memory,
      syscall: this.#guestSyscall(e),
      dispatch_miss: e.gex_dispatch_miss,
      illegal: e.gex_illegal,
      breakpoint: e.gex_breakpoint,
      sysreg_read: e.gex_sysreg_read,
      sysreg_write: e.gex_sysreg_write,
      cache_op: e.gex_cache_op,
      safepoint: (SP_CODE && single.on) ? this.#singleSafepoint(e) : e.gex_safepoint,
      fma: e.gex_fma,
      x87: e.gex_x87,
      mx: e.gex_mx,
      // A counting unit's table calls (cgen.rs `gex_icall_note`).
      icall_note: e.gex_icall_note,
    };
    const tGuest = performance.now();
    this.#guest = await WebAssembly.instantiate(
      guestWasm instanceof WebAssembly.Module ? guestWasm : await WebAssembly.compile(guestWasm),
      { gex: this.#imports },
    );
    this.#guestInstantiateMs = performance.now() - tGuest;
    _at('guest instantiated');
  }

  /**
   * `attach`'s instances, made while this Worker waits in the host's pool for
   * a process to run on a memory of its own (an exec'd posix_spawn child,
   * `#adoptExec`): the kernel's, the shim's and the guest's, on this Gex's
   * fresh memory. Nothing of a process is in them yet -- its pid, descriptors,
   * environment and devices are its kernel's, built at adopt -- so the Worker
   * can still be given any process; one that brings its own memory (a thread,
   * a forked child) leaves this Gex unused.
   */
  async preattach(shimWasm, guestWasm) {
    await this.#instantiate(shimWasm, guestWasm, undefined, { stack: false });
    this.#preattached = true;
  }

  /** Whether `preattach` has run and no `attach` has used it yet. */
  get preattached() {
    return this.#preattached;
  }
  #preattached = false;

  /** The process's environment, for a Gex made before its process was known
   *  (`preattach`). */
  setEnv(env) {
    this.#env = env;
  }

  /**
   * Join a process that is already loaded, as one of its threads. Same memory,
   * shim and guest as `load`, but no program is placed and no kernel built --
   * both already exist in the adopted memory.
   *
   * Instantiating a second time is safe only because the shared build is linked
   * with `--shared-memory`: wasm-ld emits passive data segments and an
   * atomic-guarded `__wasm_init_memory`, so the first instance initialises the
   * data and later ones leave it alone. A plain memory would wipe the kernel.
   */
  async attach(shimWasm, guestWasm, { devices = true } = {}) {
    const _at = (s) => { if (globalThis.GEX_ATTACH_TRACE) console.error(`[attach] ${s}`); };
    _at('start');
    if (this.#preattached) {
      this.#preattached = false;
      this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
    } else await this.#instantiate(shimWasm, guestWasm, _at);
    // The devices this process has -- process-wide flags in shared memory, so
    // only a *process* sets them. A thread answering for itself would turn a
    // GPU or network off underneath the threads using it. Before the kernel is
    // built, which is when the device nodes are made.
    if (devices) {
      this.#shim.exports.gex_set_gl?.(this.#gl ? 1 : 0);
      this.#shim.exports.gex_set_d3d?.(this.#d3d ? 1 : 0);
      this.#shim.exports.gex_set_wl?.(this.#wl ? 1 : 0);
      if (this.#wlOut) this.#shim.exports.gex_set_wl_output?.(...this.#wlOut);
      if (this.#wlOut) this.#shim.exports.gex_set_wl_scale?.(this.#wlScale);
      this.#shim.exports.gex_set_network?.(this.#net ? 1 : 0);
      // The unit cache and ELF export providers, as `load` gives a process its
      // own: a forked child's table starts empty while its GOT holds what
      // its parent's providers made code (cgen/elfprov.rs).
      if (this.#cgenMode) this.#shim.exports.gex_cgen_set_mode?.(this.#cgenMode);
      if (logOn('serveall')) this.#shim.exports.gex_cgen_serve_always?.(1);
      if (logOn('record')) this.#shim.exports.gex_cgen_set_record?.(1);
      if (this.#noElfProviders) this.#shim.exports.gex_elfprov_off?.();
      this.#libprovConfig();
    // Narrate one page's directory writes, in every process.
    if (Gex.watchDir >= 0) this.#shim.exports.gex_watch_dir?.(BigInt(Gex.watchDir));
    }
    const e = this.#shim.exports;
    // A thread joins its process's numbering; a forked child decides once its
    // kernel exists (`adoptProcess`).
    this.#dense = !!e.gex_dense_on?.() && !!this.#guest.exports.blocks;
    this.#entriesInit();
    this.#offerToFill(this.#guest);
    _at('offerToFill done, catchUp');
    // Everything the process translated before this thread existed.
    await this.#catchUp();
    _at('attach catchUp done');
  }

  /**
   * Everything this process has mapped, as `{ at, bytes }` -- what a forked
   * child needs and only that: its image, stack and heap (the parent's after
   * `fork`), plus the low region, where the register contexts and dispatch
   * directory live below the image and outside the memory map.
   *
   * Copied rather than shared: a copy once at `fork`, not a copy on every
   * switch for the life of both, which is what the switcher was built to cope
   * with.
   */
  /** Copy the runs `#processImage` left as `len` into one staging buffer,
   *  noting each one's offset (`stage`) and the buffer (`#imageStage`, sent
   *  as the record's `imageStage`); the child copies them out and frees it.
   *  Sliced as before when the kernel cannot hold them. */
  #stageImage(out) {
    const k = this.#kernel;
    const staged = out.filter((r) => r.len !== undefined);
    const total = staged.reduce((n, r) => n + r.len, 0);
    let id = total ? k.k_stage_take(total) : 0;
    let off = 0;
    for (const r of id > 0 ? staged : []) {
      if (k.k_stage_pull(id, off, BigInt(r.at), r.len) !== 0) {
        k.k_stage_drop(id);
        id = 0;
        break;
      }
      r.stage = off;
      off += r.len;
    }
    if (id > 0) {
      this.#imageStage = id;
      return;
    }
    const mem = this.#bytes();
    for (const r of staged) {
      r.bytes = mem.slice(r.at, r.at + r.len);
      delete r.len;
      delete r.stage;
    }
  }

  /// The staging buffer of the image `#processImage` built last (0: none).
  #imageStage = 0;

  #processImage() {
    const t0 = performance.now();
    const e = this.#shim.exports;
    // Only the runs of pages that are not all zero, scanned in place: a
    // child's memory starts zeroed, so a zero page needs no bytes, and
    // slicing it whole first (637 MiB of zeros for a node child) was most of
    // the cost of a fork, paid again by the child scanning the copy for
    // zeros (`writeSparse`). Each mapped region also goes as an entry of its
    // own with no bytes (`end`, `prot`, and `zeros`, how much of it was
    // left out); runs are `dense`, written without a scan.
    const u8 = this.#bytes();
    // Division, not `>> 2` or `& ~3`: those are int32 operations, and a 32-bit Wine
    // process's memory reaches past 2 GiB (Wine's Windows address space), where
    // they turn negative -- `b & ~3` then started the tail loop near -2^31, and
    // a fork (Wine starting wineserver) scanned for hours.
    const words = new Int32Array(this.#memory.buffer, 0, Math.floor(u8.length / 4));
    const blank = (a, b) => {
      const tail = b - (b % 4);
      for (let i = Math.floor(a / 4); i < tail / 4; i++) if (words[i] !== 0) return false;
      for (let i = tail; i < b; i++) if (u8[i] !== 0) return false;
      return true;
    };
    const out = [];
    // The runs go through a staging buffer in the global kernel's memory where
    // there is one to share (`k_stage_*`): `memory.copy` both ways, into pages
    // a previous fork already touched, where a fresh ArrayBuffer per run cost
    // ~50 ms here and ~100 ms in the child for apt-get's 62.6 MiB. Not with
    // `GEX_DEBUG_EXIT`, whose checksums read the bytes; not in a build whose
    // memories are not shared (the single-thread page), which transfers them.
    const k = this.#kernel;
    const staging = !globalThis.GEX_DEBUG_EXIT && !this.#wide && !!k?.k_stage_take
      && isShared(this.#memory.buffer)
      && isShared(this.#kernelSpec?.memory?.buffer);
    this.#imageStage = 0;
    // The scan in the shim (`gex_image_runs`, 16 bytes a load) when it has
    // one; the pairs it writes are the runs of pages that are not all zero.
    // Not in a 64-bit memory: its addresses are u32, and a region above 4 GiB
    // (a thread's stack in the high arena) was scanned at its address cut to
    // 32 bits -- thread-churn's forked child got the parent's low pages over
    // its own (module registry rows naming a module store it does not have).
    let runsCap = 0;
    let runsAt = 0;
    const runsWasm = e.gex_image_runs && !this.#wide
      ? (start, end) => {
        for (;;) {
          if (!runsCap) {
            runsCap = 1024;
            runsAt = e.gex_alloc(runsCap * 8) >>> 0;
          }
          const n = e.gex_image_runs(start, end, runsAt, runsCap);
          if (n >= 0) {
            // Views made now: the allocation above may have grown memory.
            const pairs = new Uint32Array(this.#memory.buffer, runsAt, n * 2);
            const mem = this.#bytes();
            let dense = 0;
            for (let j = 0; j < n; j++) {
              const at = pairs[j * 2];
              const len = pairs[j * 2 + 1];
              out.push(staging ? { at, len, dense: true } : { at, bytes: mem.slice(at, at + len), dense: true });
              dense += len;
            }
            return end - start - dense;
          }
          e.gex_free?.(runsAt, runsCap * 8);
          runsCap *= 4;
          runsAt = e.gex_alloc(runsCap * 8) >>> 0;
        }
      }
      : null;
    const runs = runsWasm ?? ((start, end) => {
      const PAGE = 4096;
      let zeros = 0;
      let i = start;
      while (i < end) {
        const pe = Math.min(i + PAGE, end);
        if (blank(i, pe)) {
          zeros += pe - i;
          i = pe;
          continue;
        }
        let run = pe;
        while (run < end) {
          const next = Math.min(run + PAGE, end);
          if (blank(run, next)) break;
          run = next;
        }
        out.push({ at: i, bytes: u8.slice(i, run), dense: true });
        i = run;
      }
      return zeros;
    });
    out.push({ at: 0, bytes: new Uint8Array(0), zeros: runs(0, LOW_END) });
    if (this.#wide) out.push({ at: DISPATCH_DIR64, bytes: new Uint8Array(0), zeros: runs(DISPATCH_DIR64, HIGH_TABLES_END) });
    // The map read once, as the blob the child rebuilds its map from
    // (`#vmaBlob`), and the regions copied are that blob's: read a region at
    // a time (`gex_vma_count`, then `gex_vma_at`), other threads mapping and
    // unmapping meanwhile shifted the indexes, and the image missed regions
    // and named others twice. A missed thread stack above 4 GiB left the
    // child's memory short of it, and musl's fork in the child faulted
    // walking the thread list there (fork-maps-threads).
    const blob = this.#vmaBlob();
    this.#forkVmaBlob = blob;
    const regions = blob ? vmaRegions(blob) : this.#vmaRegionsEach();
    for (const { start, end, prot } of regions) {
      // `prot` travels with the bytes: the child rebuilds its memory *map* from
      // these, and a map is not recoverable from the memory it describes.
      if (end > start) {
        const r = { at: start, end, prot, bytes: new Uint8Array(0), zeros: 0 };
        out.push(r);
        r.zeros = runs(start, end);
      }
    }
    if (runsCap) e.gex_free?.(runsAt, runsCap * 8);
    if (staging) this.#stageImage(out);
    // A checksum per region, so the child can say whether what it runs on is
    // what was sent: a corrupt copy and a good copy under a disagreeing kernel
    // look identical otherwise.
    if (globalThis.GEX_DEBUG_EXIT) for (const r of out) r.sum = sum32(r.bytes);
    if (globalThis.GEX_DEBUG_EXIT) {
      const bytes = out.reduce((n, r) => n + r.bytes.byteLength, 0);
      // How much of it the child could have re-read instead. See
      // `gex_vma_reloadable`: read-only library text is identical to its file.
      let again = 0;
      if (e.gex_vma_reloadable) {
        const n = e.gex_vma_count();
        const scratch = e.gex_alloc(24) >>> 0;
        const dv = new DataView(this.#memory.buffer);
        for (let i = 0; i < n; i++) {
          if (e.gex_vma_at(i, scratch) !== 0) continue;
          if (!e.gex_vma_reloadable(i)) continue;
          const a = Number(dv.getBigUint64(scratch, true));
          const b = Number(dv.getBigUint64(scratch + 8, true));
          if (b > a) again += b - a;
        }
        e.gex_free?.(scratch, 24);
      }
      console.error(
        `[gex] fork image: ${out.length} regions, ${(bytes / 1048576).toFixed(1)} MiB, `
        + `${(again / 1048576).toFixed(1)} MiB of it re-readable, `
        + `${Math.round(performance.now() - t0)}ms to copy`,
      );
    }
    return out;
  }

  /// The map `#processImage` read for the fork being built: the child gets
  /// these bytes, so its map is the one its image was copied by.
  #forkVmaBlob = null;

  /** The map a region at a time, for a shim with no blob. */
  #vmaRegionsEach() {
    const e = this.#shim.exports;
    const scratch = e.gex_alloc(24) >>> 0;
    const out = [];
    const n = e.gex_vma_count();
    for (let i = 0; i < n; i++) {
      if (e.gex_vma_at(i, scratch) !== 0) continue;
      const dv = new DataView(this.#memory.buffer);
      out.push({
        start: Number(dv.getBigUint64(scratch, true)),
        end: Number(dv.getBigUint64(scratch + 8, true)),
        prot: Number(dv.getBigUint64(scratch + 16, true)),
      });
    }
    e.gex_free?.(scratch, 24);
    return out;
  }

  /**
   * The memory map whole: each region's name and the file its code came from,
   * which the three numbers of `gex_vma_at` cannot carry. A child given only
   * spans identifies no region, finds nothing in the shared module store, and
   * lifts its parent's libraries again.
   */
  #vmaBlob() {
    const e = this.#shim.exports;
    if (!e.gex_vma_blob_len || !e.gex_vma_blob_read) return null;
    // One read of the map, into room to spare: other threads map and unmap
    // between a length asked and a read, and a blob read against a stale
    // length was refused (the image then fell back to reading the map a
    // region at a time). Asked again with more room while it does not fit.
    let cap = (e.gex_vma_blob_len() >>> 0) * 2 + 4096;
    let blob = null;
    for (let tries = 0; tries < 8 && !blob; tries++, cap *= 2) {
      const buf = e.gex_alloc(cap) >>> 0;
      const got = e.gex_vma_blob_read(buf, cap) >>> 0;
      if (got) blob = new Uint8Array(this.#memory.buffer, buf, got).slice();
      e.gex_free?.(buf, cap);
    }
    return blob;
  }

  /**
   * Instantiate the global kernel against this process's memory. Every Worker
   * does this with the same module and the same *kernel* memory (where the
   * module's Rust state lives, so all see one process table) and a different
   * process memory (memory 1, how it copies bytes in and out). Nothing without
   * a `kernel` in the constructor options: one process needs no such table.
   */
  async #startKernel() {
    const _k = (s) => { if (globalThis.GEX_ATTACH_TRACE) console.error(`[kernel] ${s}`); };
    if (!this.#kernelSpec || this.#kernel) { _k(`skip (spec=${!!this.#kernelSpec} kernel=${!!this.#kernel})`); return; }
    const { module, memory, stackTop, spare } = this.#kernelSpec;
    _k(`compile (isModule=${module instanceof WebAssembly.Module} stackTop=0x${(stackTop>>>0).toString(16)})`);
    const mod = module instanceof WebAssembly.Module
      ? module
      : await WebAssembly.compile(module);
    _k('instantiate');
    await lazyReady();
    const inst = await WebAssembly.instantiate(mod, {
      env: { memory, ...processMemories(this.#memory, this.#wide, spare), lazy_fetch: lazyFetch(memory), block_io: blockIo(memory), page_fs: (SP_CODE && single.on) ? pageFsSuspending(memory) : pageFs(memory) },
    });
    _k('instantiated');
    // The stack, before anything is called. Each Worker's `__stack_pointer` is
    // a per-instance global that wasm-ld inits to the same address in shared
    // memory, so two Workers in the kernel would overwrite each other's locals.
    // `stackTop` is a per-Worker slice the kernel reserves; the first Worker
    // has none and keeps the linked stack.
    // Its thread-locals too (`kernelThread`), at the top of that slice.
    if (stackTop) kernelThread(inst.exports, stackTop);
    this.#kernel = inst.exports;
    // `?timescale=N&fixedtime=S`: the benchmark clocks (gex-kernel
    // `set_guest_clock`). The
    // first process sets them for all.
    const scale = Number(globalThis.GEX_TIME_SCALE) || 0, fixed = Number(globalThis.GEX_FIXED_TIME) || 0;
    if ((scale > 0 && scale !== 1) || fixed) {
      const shift = fixed ? Math.floor(Number(this.#env.now() / 1000000000n)) - fixed : 0;
      inst.exports.k_guest_clock?.(Math.round((scale || 1) * 1000), shift | 0);
    }
    this.#guestClock();
  }

  /** The benchmark clocks the kernel holds: monotonic speed, wall shift (s). */
  #guestClock() {
    if (this.#clockSet) return this.#clockSet;
    const gc = BigInt.asUintN(64, BigInt(this.#kernel?.k_guest_clock?.(0, 0) ?? 0));
    this.#clockSet = { scale: Number(gc >> 32n) / 1000 || 1, shift: BigInt(BigInt.asIntN(32, gc & 0xffffffffn)) };
    return this.#clockSet;
  }
  #clockSet = null;

  /** How long the monotonic clocks are held back (a restored session
   *  snapshot's time saved, gex-kernel `MONOTONIC_HELD`): set before this
   *  Worker's first guest instruction, so read once. */
  #monoHeld() {
    return (this.#monoHeldNs ??= BigInt(this.#kernel?.k_mono_held?.(0n) ?? 0n));
  }
  #monoHeldNs = null;

  /** Monotonic ns since boot as the guest reads it. */
  #guestMono(ns) {
    const c = this.#guestClock(), m = ns - BigInt(this.#kernel?.k_boot_secs?.(0) ?? 0) * 1000000000n - this.#monoHeld();
    return c.scale === 1 ? m : BigInt(Math.floor(Number(m) * c.scale));
  }

  /**
   * Where the global kernel's per-Worker stacks are, for the host to hand out.
   * Null when there is no global kernel or it is too old to say. Read from this
   * Worker, which must be the first: see `#startKernel`.
   */
  kernelStacks() {
    const k = this.#kernel;
    if (!k?.k_stack_area) return null;
    return {
      area: k.k_stack_area() >>> 0, size: k.k_stack_size() >>> 0, count: k.k_stack_count() >>> 0,
      // A session snapshot's word, and the wait slots it bumps
      // (`ProcessHost.snapshot`).
      ...(k.k_snap_word ? { snapWord: k.k_snap_word() >>> 0, wsAddr: k.k_ws_addr(0) >>> 0, wsCount: k.k_ws_count() >>> 0 } : {}),
      ...(k.k_mono_held_at ? { monoHeld: k.k_mono_held_at() >>> 0 } : {}),
      ...(k.k_snap_extents_at ? { extents: k.k_snap_extents_at() >>> 0 } : {}),
    };
  }

  /// Armed once: every few seconds, say what this process is waiting for.
  /// A deadlock reaches neither census the shim already had (idle spins, wakes),
  /// so a wedged run printed nothing at all. Around the whole run loop, since a
  /// process parked on a pipe another process fills exits the loop differently.
  #armCensus() {
    // What it finds is debug output (`narrate`), so it runs only when that is on.
    if (this.#census || !logOn('debug')) return;
    const e = this.#shim?.exports;
    if (!e?.gex_wait_census) return;
    let last = '';
    this.#census = setInterval(() => {
      // Never after the slot went back (`releaseWorker`).
      if (this.#released) return;
      const p = e.gex_alloc(512) >>> 0;
      const n = e.gex_wait_census(p, 512) >>> 0;
      if (n) {
        const text = utf8(this.#bytes().slice(p, p + n));
        // Only when it has stopped changing: a wedged process repeats the same
        // wait forever, which is the signal; a busy one churns through them.
        // Debug output: an idle process (a shell at its prompt, Wine's
        // services) is as quiet as a wedged one.
        if (text === last) {
          narrate(`[gex] quiet: ${text}${(SP_CODE && single.on) ? ` js=${this.#where}/installs=${this.#installs}/pending=${e.gex_jit_pending_len() >>> 0}` : ''}`);
        }
        last = text;
      }
      e.gex_free?.(p, 512);
    }, 3000);
    // Node keeps a process alive for a pending timer; this must not be the
    // reason a Worker outlives its guest.
    this.#census.unref?.();
  }

  /** The global kernel's exports, or null. */
  get kernel() { return this.#kernel; }

  /** The saved store, into the kernel and checked. */
  async #xcacheRestore(name) {
    const e = this.#shim.exports;
    const k = this.#kernel;
    if (!k?.k_store_restore_buf || !e.gex_xcache_restore) return;
    k.k_xcache_enable();
    const t0 = performance.now();
    const bytes = await xcRead(name);
    if (!bytes) {
      console.log(`[gex] xcache ${name}: nothing saved`);
      return;
    }
    const t1 = performance.now();
    const at = k.k_store_restore_buf(bytes.length) >>> 0;
    new Uint8Array(this.#kernelSpec.memory.buffer).set(bytes, at);
    const r = e.gex_xcache_restore();
    const t2 = performance.now();
    console.log(`[gex] xcache ${name}: ${(bytes.length / 1048576).toFixed(1)} MiB read in ${Math.round(t1 - t0)} ms, `
      + `restored ${r} modules in ${Math.round(t2 - t1)} ms`);
  }

  /**
   * Save the kernel's store for the next load if it grew by `minBytes` since
   * the last save. The kernel lets one save run at a time. `gate`: the page's
   * save gate (xcache.mjs), which every process of a page shares; `pid`, for
   * the line it prints.
   */
  async xcacheSave(name, minBytes = 1 << 20, gate = undefined, delayMs = 0, pid = undefined) {
    const k = this.#kernel;
    if (!name || !k?.k_store_save) return;
    // The page's save gate (xcache.mjs): the host does not end Workers while
    // this is inside it, and refuses entry once it is stopping.
    if (gate && !xcEnter(gate)) return;
    try {
      for (let again = false; ; again = true) {
        const t0 = performance.now();
        if (gate) Atomics.add(gate, 1, 1);
        let n = 0;
        try {
          n = k.k_store_save(minBytes) >>> 0;
          if (n) await this.#xcacheWrite(name, n, t0, delayMs, pid);
        } finally {
          if (gate) Atomics.sub(gate, 1, 1);
        }
        // Nothing taken: nothing new, or another process's save under way
        // (the kernel takes one at a time). Wait for that one, then ask once
        // more for what it did not cover.
        if (n || again || !gate || !Atomics.load(gate, 1)) return;
        while (Atomics.load(gate, 1)) await xcSleep(5);
      }
    } finally {
      if (gate) xcLeave(gate);
    }
  }

  /** The `n` bytes `k_store_save` took, out to the store `name`. */
  async #xcacheWrite(name, n, t0, delayMs, pid) {
    const k = this.#kernel;
    try {
      const at = k.k_store_save_ptr() >>> 0;
      const bytes = new Uint8Array(this.#kernelSpec.memory.buffer, at, n).slice();
      k.k_store_save_done();
      // A test's stand-in for a slow disk (run-node GEX_XCACHE_DELAY_MS).
      if (delayMs) await xcSleep(delayMs);
      await xcWrite(String(name), bytes);
      console.log(`[gex] xcache ${name}: saved ${(n / 1048576).toFixed(1)} MiB in ${Math.round(performance.now() - t0)} ms${pid ? `, pid ${pid}` : ''}`);
    } catch (err) {
      k.k_store_save_done();
      console.log(`[gex] xcache ${name}: not saved: ${err?.message ?? err}`);
    }
  }

  /**
   * What the shim calls to reach the process table every process shares.
   * `k_present` is the whole negotiation: zero means no global kernel, and the
   * shim keeps the table in its own memory. The stubs below then go unused --
   * they exist only because a wasm import cannot be absent.
   */
  #kernelImports(shimMod) {
    const k = this.#kernel;
    const absent = () => 0;
    // What the shim actually imports, from `WebAssembly.Module.imports` -- the
    // only authority. A hand-written list was a third place to keep in step and
    // `check-glue.mjs` did not catch drift, because these arrive via the kernel.
    const NAMES = WebAssembly.Module.imports(shimMod)
      .filter((i) => i.module === 'gex_k' && i.kind === 'function')
      .map((i) => i.name);
    // Some of these return an i64, which JS has to answer with a BigInt.
    const BIG = new Set(['k_inet_op', 'k_wait', 'k_take_signals', 'k_take_kill_signals', 'k_take_thread_signal', 'k_tty_winsize']);
    const out = { k_present: () => (k ? 1 : 0) };
    for (const n of NAMES) {
      // Some `gex_k` imports are answered here, not by the kernel (`k_present`
      // is the negotiation itself), so skip names this file already owns.
      if (n in out) continue;
      if (k) {
        if (typeof k[n] !== 'function') {
          throw new Error(
            `the global kernel has no ${n}: it is older than this glue. ` +
              'Rebuild it with hosts/kernel/build.sh.',
          );
        }
        out[n] = k[n];
      } else {
        out[n] = BIG.has(n) ? () => 0n : absent;
      }
    }
    // Nobody has asked for a halt, rather than "halt with status 0".
    if (!k) out.k_halt_code = () => -1;
    // The single-thread build: the wait words are promises (single.mjs
    // `Sched`). A park that need not wait answers at once, as
    // `memory.atomic.wait32` does when the value has moved; one that must
    // suspends this fiber's stack. Every wake also wakes the fibers.
    if (k && (SP_CODE && single.on)) {
      const nested = () => {
        throw new Error('gex (single-thread build): a blocking call inside a nested guest call '
          + `(${this.#nest.join(' > ')}) cannot suspend`);
      };
      // A wait slot's word is in the kernel's (plain) memory, which can
      // grow: the view is asked for each time.
      const words = () => new Int32Array(this.#kernelSpec.memory.buffer);
      out.k_ws_park = new WebAssembly.Suspending((slot, seq, ns) => {
        if (k.k_ws_seq(slot) !== seq) return 1;
        if (this.#hardNested()) nested();
        this.#where = `park(${ns})`;
        return sched.parkSlot(words, (k.k_ws_addr(slot) >>> 0) >>> 2, seq, ns, this.#fiber);
      });
      // A notify moved slot words nobody `Atomics.wait`s on here; the fibers
      // parked on them look (and every park and yield looks too, for the
      // notifies inside the kernel's own calls).
      out.k_wq_notify = (id) => {
        k.k_wq_notify(id);
        sched.checkSlots();
      };
      out.k_futex_wake = (addr) => {
        k.k_futex_wake(addr);
        sched.checkSlots();
      };
      out.k_wq_release = () => {
        k.k_wq_release();
        sched.checkSlots();
      };
    }
    return out;
  }

  /**
   * Hand the shim the module's `gex.meta` section -- the record of where the
   * ahead-of-time build stopped, which runtime translation carries on from.
   * Without it a kernel reports "nothing to translate" at the first uncovered
   * address.
   */
  #setMetadata(guestWasm) {
    const meta = metadataSection(guestWasm);
    if (!meta || !meta.length) return;
    const mp = this.#shim.exports.gex_alloc(meta.length) >>> 0;
    this.#bytes().set(meta, mp);
    this.#shim.exports.gex_set_metadata(mp, meta.length);
  }

  /**
   * Which dispatch slots this process's translated pages are in. Read as one
   * block, not page by page: a real process has tens of thousands of pages.
   */
  #pageMap() {
    const e = this.#shim.exports;
    if (!e.gex_page_count) return [];
    const n = e.gex_page_count() >>> 0;
    if (!n) return [];
    const p = e.gex_alloc(n * 12) >>> 0;
    const wrote = e.gex_page_dump(p, n * 12);
    const out = wrote > 0 ? this.#bytes().slice(p, p + wrote * 12) : new Uint8Array(0);
    e.gex_free?.(p, n * 12);
    return out;
  }

  /**
   * Which object each translated page belongs to, as identities. The companion
   * to `#pageMap` and the one that outlives an `execve`: the page map describes
   * an address space about to be replaced, this describes the libraries in it.
   */
  #sharedMap() {
    const e = this.#shim.exports;
    if (!e.gex_shared_count) return [];
    const n = e.gex_shared_count() >>> 0;
    if (!n) return [];
    const p = e.gex_alloc(n * 20) >>> 0;
    const wrote = e.gex_shared_dump(p, n * 20);
    const out = wrote > 0 ? this.#bytes().slice(p, p + wrote * 20) : new Uint8Array(0);
    e.gex_free?.(p, n * 20);
    return out;
  }

  /**
   * Become the child a `fork` in another Worker produced.
   *
   * Everything a process is, in dependency order: its memory (the parent's
   * mapped regions, copied), a shim and guest instantiated against it, a kernel
   * (pid, break, memory map), its dispatch slots and the code in them. The
   * child inherits the parent's dispatch *directory* with the memory, so its
   * code branches into slots the parent allocated -- and without the same map
   * and modules those slots are empty or past the end of an ungrown table.
   */
  /** `bytes` into the shim's heap for the length of `f(ptr, len)`. */
  #putBytes(bytes, f) {
    const e = this.#shim.exports;
    const n = Math.max(1, bytes.length);
    const p = e.gex_alloc(n) >>> 0;
    this.#bytes().set(bytes, p);
    try {
      return f(p, bytes.length);
    } finally {
      e.gex_free?.(p, n);
    }
  }

  /**
   * posix_spawn's child, whose exec was staged while it ran in its parent's
   * Worker (`#vforkChild`): a fresh memory, the kernel a forked child gets
   * with its descriptors and signal state, and the exec run here. The
   * entry, or 0n when the program could not be loaded (the process has
   * then exited).
   */
  async #adoptExec(shimWasm, guestWasm, child, opts) {
    this.#windowsSpace = 0;
    this.#noPlace = child.wasmImage ?? null;
    const pooled = this.#preattached;
    await this.attach(shimWasm, guestWasm);
    // Whether the exec started on the runtime its Worker made in the pool
    // (a test counts these).
    vforkTrace(`[pool] pid ${child.pid}: ${pooled ? 'pooled runtime' : 'new runtime'}`);
    const e = this.#shim.exports;
    this.#setMetadata(guestWasm);
    const o = child.opts ?? {};
    // The switches the host hands every Worker (its `extra`, at the top of
    // `opts` here) that the record's options leave unsaid: a page's spawn
    // (ProcessHost.spawnExec) carries the run options alone, where a
    // posix_spawn child's carry its parent's.
    for (const k of EXEC_SWITCHES) if (o[k] === undefined && opts?.[k] !== undefined) o[k] = opts[k];
    const uid = o.uid ?? child.uid;
    if (uid !== undefined) e.gex_set_identity(uid, o.gid ?? child.gid ?? uid);
    if (child.cwd) this.#putBytes(new TextEncoder().encode(child.cwd), e.gex_set_cwd);
    if (child.state?.length) this.#putBytes(child.state, e.gex_set_child_state);
    e.gex_adopt_child(child.pid, child.ctx, 0n, 0n, 0n, 0n);
    // Dense unless asked not to (`dense: false`), as for a process booted
    // fresh: an exec whose options never said is not a classic one.
    const denseAsked = o.dense !== false && !!this.#guest.exports.blocks && !!e.gex_set_dense;
    this.#dense = denseAsked;
    o.dense = denseAsked ? (o.dense === 2 ? 2 : 1) : false;
    e.gex_set_dense?.(denseArg(o, this.#dense));
    if (TEST_KEEP_PROBE) e.gex_test_keep_probe?.(1);
    this.#entriesInit();
    this.#opts = o;
    syncSlotShift(e);
    this.#childOpts(e, o, child.pid);
    if (child.fds?.length) this.#putBytes(child.fds, e.gex_adopt_fds);
    this.#closureModules = opts?.closure ?? this.#closureModules;
    this.#wantLibrary = opts?.wantLibrary ?? this.#wantLibrary;
    await this.#installClosure();
    // The single-thread build: the exec reads the program's files through
    // the kernel, which may wait, so it runs on a stack that can suspend.
    if (SP_CODE && single.on) return this.#putBytesP(child.exec, e.gex_adopted_exec);
    return this.#putBytes(child.exec, e.gex_adopted_exec);
  }

  /** `#putBytes` for the single-thread build: `f` on a JSPI stack of its own
   *  (`#promisingOf`), whose waits suspend it. */
  async #putBytesP(bytes, f) {
    const e = this.#shim.exports;
    const n = Math.max(1, bytes.length);
    const p = e.gex_alloc(n) >>> 0;
    this.#bytes().set(bytes, p);
    try {
      return await this.#promisingOf(f)(p, bytes.length);
    } finally {
      e.gex_free?.(p, n);
    }
  }

  /** A child's run options, said to its kernel (`adoptProcess`). */
  #fjitOn(e, o) {
    if (!o.fjit || !e.gex_set_fjit) return;
    e.gex_set_fjit(o.fjit === 2 ? 2 : 1);
    if (o.fjitMax != null || o.fjitOnly) e.gex_fjit_limit?.(o.fjitMax ?? 0xffffffff, BigInt(o.fjitOnly ?? 0));
    // Units for the JIT's sake alone: an image without one is said only
    // under GEX_LOG=unit (bit 5 quiets it otherwise).
    if (!(this.#cgenMode & 3)) this.useCgen(this.#cgenMode | 1 | (logOn('unit') ? 0 : 32));
  }

  #childOpts(e, o, pid) {
    this.#fjitOn(e, o);
    // `?strace=1` narrates every process; `?stracepid=N` only that one, which
    // is what a run that must stay alive can afford -- with `strace` set or
    // not (`start({ strace: N })` sets both, as the KDE page's
    // `?strace=1&stracepid=N` does).
    if (o.stracePid ? o.stracePid === pid : o.strace) { e.gex_set_trace(1); this.#traced = true; }
    if (o.parkReport) e.gex_set_park_report?.(o.parkReport);
    if (o.traceBlocks) e.gex_set_trace_blocks?.(1);
    if (o.watchMem) e.gex_set_watch_mem?.(BigInt(o.watchMem));
    if (o.traceMisses) e.gex_set_trace_misses?.(1);
    if (o.dlChecked) e.gex_set_dl_checked?.(1);
    if (o.jitDiag) e.gex_set_jit_diag?.(1);
    if (o.checkSp) e.gex_set_check_sp?.(1);
    if (o.noIntrinsics) e.gex_set_intrinsics?.(0);
    if (o.intrinsicMask !== undefined) e.gex_set_intrinsic_mask?.(o.intrinsicMask);
    if (o.liftMax) e.gex_set_lift_max?.(o.liftMax);
    if (o.inlineDispatch) e.gex_set_inline_dispatch?.(1);
    if (o.xlateReport) e.gex_set_xlate_report?.(o.xlateReport);
    if (o.discoverSkip) e.gex_set_discover_skip?.(o.discoverSkip);
    if (o.t1Fold) e.gex_set_t1_fold?.(1);
    if (o.tty) e.gex_set_tty(1);
  }

  async adoptProcess(shimWasm, guestWasm, child, opts = {}) {
    // posix_spawn's child, run in its parent's Worker until its exec: no image.
    if (child.exec) return this.#adoptExec(shimWasm, guestWasm, child, opts);
    // Narrated and timed when asked: every step is a place a child can stop
    // dead, and a hang is indistinguishable from a slow start otherwise.
    const began = performance.now();
    let last = began;
    // `GEX_ADOPT_TRACE` narrates adopt phases WITHOUT the heavy GEX_DEBUG_EXIT
    // apparatus (per-guest-write console.error flood + image sum-checks), so a
    // near-native-speed run can reach a deep fork (e.g. Xvfb's xkbcomp helper)
    // and show where its adopt hangs. The sum-checks below stay GEX_DEBUG_EXIT-
    // only, and their sums are only computed then (fork side), so this path
    // computes none.
    const step = globalThis.GEX_DEBUG_EXIT || globalThis.GEX_ADOPT_TRACE
      ? (s) => {
          const now = performance.now();
          console.error(
            `[gex] adopt ${child.pid}: ${s} (+${Math.round(now - last)}ms, `
            + `${Math.round(now - began)}ms in)`,
          );
          last = now;
        }
      : () => {};
    step('copying the image');
    // Grow to cover the parent's mapped regions before writing a byte, or the
    // copy throws `RangeError: offset is out of bounds` above the initial size.
    this.#growTo(child.image.reduce((n, r) => Math.max(n, r.end ?? 0, r.at + imageLen(r)), 0));
    // A staged image (`#stageImage`) is copied out by the global kernel, here
    // before `attach` as the transferred one is: its instance only needs this
    // memory, and starting it again there is a no-op.
    const stage = child.imageStage > 0 ? child.imageStage : 0;
    if (stage) await this.#startKernel();
    const u8 = this.#bytes();
    // `?forkskip=N`: leave any region larger than N MiB uncopied.
    //
    // There is no copy-on-write here -- one guest process is one linear memory,
    // so a fork writes the parent's mappings into the child and every byte of
    // that is a byte the tab pays for. Whether that copy is *needed* is a
    // question about the child's future, and this is how to ask it: skip the
    // big regions, leave them zero, and see whether the run still passes. A
    // child that execs immediately (nearly all of them) never looks at them.
    //
    // Destructive on purpose -- if the guest reads a skipped page it reads
    // zeros -- so it is a measurement, not a mode.
    const skipOver = typeof globalThis.GEX_FORK_SKIP_MIB === 'number'
      ? globalThis.GEX_FORK_SKIP_MIB * (1 << 20)
      : Infinity;
    let copied = 0;
    let skipped = 0;
    let zeros = 0;
    for (const r of child.image) {
      const n = imageLen(r);
      if (n > skipOver) {
        skipped += n;
        continue;
      }
      zeros += r.zeros ?? 0;
      if (r.stage !== undefined) {
        if (this.#kernel.k_stage_push(stage, r.stage, BigInt(r.at), n) !== 0) {
          throw new Error(`fork image: staged run at 0x${r.at.toString(16)} (${n} bytes) did not copy`);
        }
        copied += n;
        continue;
      }
      if (r.dense) {
        u8.set(new Uint8Array(r.bytes), r.at);
        copied += n;
        continue;
      }
      const [wrote, blank] = writeSparse(u8, new Uint8Array(r.bytes), r.at);
      copied += wrote;
      zeros += blank;
    }
    if (stage) this.#kernel.k_stage_drop(stage);
    this.#forkCopied = copied;
    this.#forkSkipped = skipped;
    this.#forkZeros = zeros;
    // Checked here, before anything else writes to this memory: the check
    // further down runs after the child has rewritten the low region itself,
    // so it cannot tell a damaged copy from a deliberate rewrite. This one can.
    if (globalThis.GEX_DEBUG_EXIT && child.image?.[0]?.sum !== undefined) {
      let bad = 0;
      for (const r of child.image) {
        const end = r.at + r.bytes.byteLength;
        if (sum32(u8.subarray(r.at, end)) === r.sum) continue;
        bad++;
        console.error(
          `[adopt] on arrival, region ${r.at.toString(16)}..${end.toString(16)} `
            + `(${r.prot ?? '-'}) is not what the parent sent`,
        );
      }
      step(bad ? `${bad} regions arrived wrong` : 'image arrived intact');
    }
    // How much of the context survives each step, to find which step empties it
    // when a child enters at zero.
    const live = () => {
      let n = 0;
      const at = Number(child.ctx);
      for (let i = at; i < at + 512; i++) if (u8[i]) n++;
      return n;
    };
    step(`image copied, ${live()} non-zero bytes in the context`);
    this.#windowsSpace = child.windowsSpace ?? 0;
    this.#noPlace = child.placeWindow === 0 ? [0, 0] : null;
    this.#forkChild = true;
    this.#inheritedEnds = child.image.map((r) => [r.at, Math.max(r.end ?? 0, r.at + imageLen(r))]);
    await this.attach(shimWasm, guestWasm);
    step(`instantiated, ${live()} non-zero bytes in the context`);
    const e = this.#shim.exports;
    // The test switch `load` sets, for a child: it is this instance's.
    if (globalThis.GEX_TEST_FORK_REBASE || globalThis.process?.env?.GEX_TEST_FORK_REBASE) e.gex_test_fork_rebase?.(1);
    // Before the kernel is built, not after: the metadata is what the child's
    // dispatch table starts from, and `gex_adopt_child` reads it as it builds.
    this.#setMetadata(guestWasm);
    // A deferred child must not claim its parent's slots: the inherited
    // directory names one per page the parent translated, none of which this
    // child has filled. Rebuilt rather than erased, keeping the guest module's
    // own passive-segment entries (the copied memory marks them applied, so
    // they are never rewritten). Not conditional on there being modules -- a
    // child handed none needs it more. See `gex_reset_directory`.
    if (adoptAfterTraps() > 0 && e.gex_reset_directory) {
      step(`directory rebuilt: ${e.gex_reset_directory() >>> 0} entries kept`);
    }
    // Before the kernel is built: `Kernel::new` fixes the identity at
    // construction, so a uid set afterwards leaves every child running as the
    // default. `gex_adopt_child` reads it as it builds.
    {
      const o0 = child.opts ?? {};
      const uid = o0.uid ?? child.uid;
      if (uid !== undefined) e.gex_set_identity(uid, o0.gid ?? child.gid ?? uid);
    }
    // The parent's working directory, so a child inherits it as it does on
    // Linux. `gex_adopt_child` reads the CWD global as it builds its kernel,
    // so this has to be set first -- same ordering as the identity above.
    // The parent's program and argv (`/proc/self/exe`, `cmdline`), likewise.
    if (child.exe && e.gex_set_child_exe) {
      const c = new TextEncoder().encode(child.exe);
      const cp = e.gex_alloc(Math.max(1, c.length)) >>> 0;
      this.#bytes().set(c, cp);
      e.gex_set_child_exe(cp, c.length);
      e.gex_free?.(cp, Math.max(1, c.length));
    }
    if (child.cwd && e.gex_set_cwd) {
      const c = new TextEncoder().encode(child.cwd);
      const cp = e.gex_alloc(c.length) >>> 0;
      this.#bytes().set(c, cp);
      e.gex_set_cwd(cp, c.length);
      e.gex_free?.(cp, c.length);
    }
    // And the signal state, umask and limits, which `gex_adopt_child` builds
    // the kernel with.
    if (child.state?.length && e.gex_set_child_state) this.#putBytes(child.state, e.gex_set_child_state);
    const entry = e.gex_adopt_child(
      child.pid,
      child.ctx,
      child.pc,
      child.sp,
      child.tls,
      child.brk ?? 0n,
    );
    if (child.compat) e.gex_set_compat?.(1);
    // Its program is its parent's, served again here: no spare for that one.
    this.#spareGen = e.gex_exec_gen?.() ?? 0;
    if (child.wasm32) {
      if (child.wasm32Link && e.gex_wasmprog_adopt_link) {
        // Before the link: adopting it may instantiate the modules.
        if (child.dlTls?.length) this.#dlTlsKept = new Map(child.dlTls);
        const b = typeof child.wasm32Link === 'string' ? new TextEncoder().encode(child.wasm32Link) : child.wasm32Link;
        const p = e.gex_alloc(b.length) >>> 0;
        this.#bytes().set(b, p);
        e.gex_wasmprog_adopt_link(p, b.length);
        e.gex_free?.(p, b.length);
      }
      if (child.wasm32Base && e.gex_wasmprog_adopt_at) e.gex_wasmprog_adopt_at(BigInt(child.wasm32), child.wasm32Base);
      else e.gex_wasmprog_adopt?.(BigInt(child.wasm32));
    }
    // The parent's block numbering carries on in the child (its low region),
    // over slot maps of its own (`dense::child`).
    this.#dense = !!(child.opts ?? {}).dense && !!this.#guest.exports.blocks && !!e.gex_set_dense;
    e.gex_set_dense?.(denseArg(child.opts, this.#dense));
    if (TEST_KEEP_PROBE) e.gex_test_keep_probe?.(1);
    this.#entriesInit();
    step(`kernel built, entry 0x${entry.toString(16)}, ${live()} non-zero bytes in the context`);
    // The kernel exists now, so everything that configures one can be said.
    const o = child.opts ?? {};
    this.#opts = o;
    syncSlotShift(e);
    this.#childOpts(e, o, child.pid);
    // Identity was already said above, before the kernel was built.
    // The map, which the bytes do not carry: without it the child's first
    // `mmap` hands out an address its own image is already at.
    // Preferring the blob, which carries each region's name and the file its
    // code came from as well as its span. The three-number form stays for a
    // shim built before the blob existed.
    const blob = child.vmaBlob;
    if (blob && e.gex_adopt_vma_blob) {
      const p = e.gex_alloc(blob.length) >>> 0;
      new Uint8Array(this.#memory.buffer, p, blob.length).set(blob);
      e.gex_adopt_vma_blob(p, blob.length);
      e.gex_free?.(p, blob.length);
    } else {
      for (const r of child.image) {
        if (r.end !== undefined) e.gex_adopt_vma(BigInt(r.at), BigInt(r.end), r.prot);
      }
    }
    step(`${child.image.length} regions mapped`);
    // Before the modules, because installing one asks the shim which slot
    // each of its blocks belongs in and the answer is in this map.
    if (child.pages?.length && e.gex_adopt_page) {
      const dv = new DataView(child.pages.buffer, child.pages.byteOffset, child.pages.byteLength);
      for (let i = 0; i + 12 <= child.pages.byteLength; i += 12) {
        e.gex_adopt_page(dv.getBigUint64(i, true), dv.getUint32(i + 8, true));
      }
    }
    // And the identity map, what is left of all this once the child execs.
    // Nothing reads it until a translation misses, so seeding early is free.
    if (child.shared?.length && e.gex_shared_seed) {
      const n = (child.shared.byteLength / 20) | 0;
      const p = e.gex_alloc(n * 20) >>> 0;
      this.#bytes().set(child.shared.subarray(0, n * 20), p);
      e.gex_shared_seed(p, n);
      e.gex_free?.(p, n * 20);
    }
    step('pages adopted');
    // What this child's isolate is holding, for the OOM in #145: measured
    // rather than guessed. `measureUserAgentSpecificMemory` needs cross-origin
    // isolation (already on for `SharedArrayBuffer`) and reports a breakdown by
    // type. Fired from the child right after adopt, awaited in the background
    // since the measurement waits for a GC.
    // One line per forked child, so a hundred-process `apt-get update` prints a
    // hundred: narration for the #145 OOM, behind the same `debugexit` flag as
    // the rest of the fork story, so an ordinary run does not fail a harness
    // counting `console.error`.
    if (globalThis.GEX_DEBUG_EXIT) {
      console.error(
        `[heap] debug=${!!globalThis.GEX_DEBUG_EXIT}`
          + ` api=${typeof performance.measureUserAgentSpecificMemory}`
          + ` isolated=${globalThis.crossOriginIsolated}`
          // Standard API absent in this Worker even when isolated, so fall back
          // to Chrome's own coarse counter.
          + ` used=${performance.memory?.usedJSHeapSize ?? 'n/a'}`
          + ` limit=${performance.memory?.jsHeapSizeLimit ?? 'n/a'}`
          + ` image=${(child.image ?? []).reduce((n, r) => n + imageLen(r), 0)}`
          + ` modules=${(child.modules ?? []).length}`,
      );
    }
    // Only then: the measurement forces a GC in every child, which an
    // ordinary run (apt's hundred forks) pays for and never reads.
    if (globalThis.GEX_DEBUG_EXIT && performance.measureUserAgentSpecificMemory) {
      const imageBytes = (child.image ?? []).reduce(
        (n, r) => n + (r.bytes?.byteLength ?? 0), 0);
      performance
        .measureUserAgentSpecificMemory()
        .then((m) => {
          const by = {};
          for (const b of m.breakdown ?? []) {
            if (!b.bytes) continue;
            const k = (b.types ?? []).join('+') || 'other';
            by[k] = (by[k] ?? 0) + b.bytes;
          }
          const parts = Object.entries(by)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 6)
            .map(([k, v]) => `${k}=${v}`)
            .join(' ');
          console.error(
            `[heap] pid ${this.#env.pid ?? '?'}: total ${m.bytes}`
              + ` image ${imageBytes} modules ${(child.modules ?? []).length}`
              + ` | ${parts}`,
          );
        })
        .catch((e) => console.error(`[heap] measure failed: ${e.message}`));
    }
    // After the kernel exists and before the guest runs: the first thing a
    // forked child does is read or write a descriptor it expects to have.
    if (child.fds?.length && e.gex_adopt_fds) {
      const p = e.gex_alloc(child.fds.length) >>> 0;
      this.#bytes().set(child.fds, p);
      e.gex_adopt_fds(p, child.fds.length);
      e.gex_free?.(p, child.fds.length);
    }
    // The closure first: its slots sit below anything the parent translated,
    // and a child's dispatch tables start empty. The directory came with the
    // adopted memory, so without this the child's branches point at unfilled
    // slots.
    this.#closureModules = opts?.closure ?? this.#closureModules;
    this.#wantLibrary = opts?.wantLibrary ?? this.#wantLibrary;
    await this.#installClosure();
    // Reachable, not filled: a child mostly execs and throws the inherited
    // image away, so instantiating every parent module up front was wasted.
    // Growing the tables cannot be skipped -- the inherited directory names
    // slots by number, and a branch past a table's end is a trap not a miss;
    // grown slots hold the miss stub, which fetches code from the shared store.
    //
    // Only as far as this child will actually dispatch through: a deferred
    // child rebuilt the directory (`gex_reset_directory` above), so the
    // parent's slot total is not a bound and growing to it wasted 272ms of
    // stub-filling per process.
    //
    // Growing to the parent's total came from #114, on the theory that a branch
    // was running off the end of a table. It was not: `enter` was failing to
    // mask the table index out of a directory entry, and the tables were never
    // too small. An eager child installs its parent's modules here, each
    // growing the tables to its own `need`, which the maximum below is for.
    let need = 0;
    if (adoptAfterTraps() <= 0) {
      for (const m of child.modules ?? []) need = Math.max(need, m.need);
      need = Math.max(need, (child.slots ?? 0) >>> 0);
    }
    if (need) this.#growTables(need, 'a1806');
    // Kept, not dropped: a child that stays wants them after all, and by then
    // its parent's Worker may be gone. See `#maybeAdopt`.
    this.#inherited = child.modules ?? [];
    // Eager (adoptAfterTraps <= 0): install all inherited modules before the
    // child runs. Kept for comparison against the deferred path, which starts
    // with empty tables and meets the parent's code through the miss path.
    if (adoptAfterTraps() <= 0) {
      step(`installing ${this.#inherited.length} modules`);
      const mine = this.#inherited;
      this.#inherited = [];
      for (const m of mine) await this.#adoptModule(m);
    } else {
      step(`${this.#inherited.length} modules reachable, ${need} slots`);
    }
    // Last of all: everything above writes to this memory, and any of it can
    // land on the parent's bytes, which the guest would meet later elsewhere.
    if (globalThis.GEX_DEBUG_EXIT && child.image?.[0]?.sum !== undefined) {
      const now = this.#bytes();
      let bad = 0;
      for (const r of child.image) {
        const end = r.at + r.bytes.byteLength;
        if (sum32(now.subarray(r.at, end)) === r.sum) continue;
        if (bad++ < 8) {
          console.error(
            `[adopt] region ${r.at.toString(16)}..${end.toString(16)} ` +
              `(${r.prot ?? '-'}) is not what the parent sent`,
          );
        }
      }
      step(bad ? `${bad} of ${child.image.length} regions differ` : 'image verified');
    }
    step('running');
    return entry;
  }

  /**
   * Hand the host any thread `clone` has just made. A thread is not a process:
   * nothing is copied, because the memory, kernel, file table and thread table
   * are already shared. All that crosses is which thread-table entry to run.
   */
  async #drainThread() {
    const thread = this.#takeThread();
    if (thread) await this.#postThread(thread);
  }

  /**
   * `#drainThread` from a loop that may never reach the event loop again (a
   * call-out, a system call served in place: a Wine thread's
   * whole life can be one call-out). Nothing asynchronous unless a thread is
   * there: a promise per trap, its rejection handler holding the trap's scope
   * (and so the trap's error), is a microtask that never runs, and a 64-bit game's
   * page queued 4 GB of them before the renderer died.
   */
  #startThread(who = 'gex') {
    let thread;
    // A thread that will not start gives back the slot its `clone` was
    // promised (`gex_worker_promise_done`).
    const lost = (err) => {
      threadNotStarted(who, err);
      try { this.#shim?.exports?.gex_worker_promise_done?.(); } catch {}
    };
    try {
      thread = this.#takeThread();
    } catch (err) {
      lost(err);
      return;
    }
    if (thread) this.#postThread(thread).catch(lost);
  }

  /** The thread `clone` left for the host, claimed: null with none. */
  #takeThread() {
    const e = this.#shim.exports;
    if (!e.gex_thread_pending) return null;
    const tid = e.gex_thread_pending();
    if (!tid) return null;
    // Made before the first thread exists, so every Worker of the process
    // holds the same one.
    this.#programStore(true);
    const thread = {
      tid,
      ctx: e.gex_thread_ctx(),
      pc: e.gex_thread_pc(),
      memory: this.#memory,
      moduleStore: this.#moduleStore ?? undefined,
      // A thread of a 32-bit process decodes in compatibility mode too.
      compat: e.gex_compat?.() ?? 0,
      // The process's run options, so a fork from this thread hands its
      // child the same ones (the fork record's `opts`): a thread Worker had
      // none, and a child of a thread lost dense numbering and the rest.
      procOpts: this.#opts,
      // Which program it belongs to: a spare caught up with another one is not used.
      execGen: e.gex_exec_gen?.() ?? 0,
    };
    if (globalThis.GEX_ATTACH_TRACE) console.error(`[spawn] #drainThread posting tid ${tid} (t=${Date.now()})`);
    e.gex_thread_taken();
    // A thread needs a memory two Workers can both hold. Posting a non-shared
    // memory does not fail -- the structured clone copies it, and the thread
    // runs on a private image nothing else can see -- so it is refused here.
    // In the single-thread build a thread is a fiber of this realm, handed
    // the memory object itself.
    if (!isShared(this.#memory?.buffer) && !(SP_CODE && single.on)) {
      throw new Error(
        `the guest started thread ${tid} on a memory that is not shared. ` +
          'A guest thread is a Worker and a Worker needs a SharedArrayBuffer: ' +
          'serve the page with COOP/COEP and load the shared build.',
      );
    }
    if (globalThis.GEX_DEBUG_EXIT) {
      console.error(
        `[gex] thread -> tid ${thread.tid} pc=0x${thread.pc.toString(16)} ` +
          `ctx=0x${thread.ctx.toString(16)}`,
      );
    }
    if (typeof this.#env.spawnThread !== 'function') {
      throw new Error(
        `the guest started thread ${tid} and this environment cannot run one. ` +
          'Give the Env a `spawnThread`, or the thread will never run.',
      );
    }
    return thread;
  }

  /** Start a claimed thread's Worker (`#takeThread`). */
  async #postThread(thread) {
    // Before the thread exists: from its first install it may publish
    // indices past what this Worker has replayed.
    this.#threaded();
    await this.#env.spawnThread(thread);
  }

  /**
   * Join a process as one of its threads, and say where to enter. The
   * counterpart of `adoptProcess` and much smaller: a process is rebuilt from a
   * copy, a thread only pointed at what is already there.
   */
  async adoptThread(shimWasm, guestWasm, thread) {
    this.#threaded();
    // A spare (`#offerSpare`) claims its slot unpromised, and is promised
    // one when it is given its thread.
    this.#promise = !thread.spare;
    // A Worker whose last thread ended (`parkThread`) is attached already.
    if (!thread.parked) await this.attach(shimWasm, guestWasm, { devices: false });
    if (thread.spare) {
      // Everything the process has installed so far, before it has a thread;
      // what it installs meanwhile is caught up with below, as any thread's.
      const gen = this.#shim.exports.gex_exec_gen?.() ?? 0;
      this.#spareWait = true;
      await this.#catchUp();
      this.#cgenCatchUp(true);
      const said = logNamed('spare');
      if (said) console.warn(`[spare] pid ${this.#shim.exports.gex_my_pid?.() ?? '?'}: caught up with ${this.#cgen?.dyl.size ?? 0} modules, waiting`);
      // And what the process installs while it waits (its program's modules
      // as its first Worker installs them, the libraries it dlopens),
      // looked for less often while there is nothing new.
      let go;
      const got = thread.spareGo().then((g) => { go = g ?? false; });
      for (let delay = SPARE_POLL_MS; go === undefined;) {
        await Promise.race([got, new Promise((r) => setTimeout(r, delay))]);
        if (go !== undefined) break;
        const was = this.#cgen?.have ?? 0;
        // Not another program's: the host lets this spare go when it hears of it.
        const same = (this.#shim.exports.gex_exec_gen?.() ?? 0) === gen && !this.#shim.exports.gex_exited?.();
        if (same && (this.#shim.exports.gex_ftab_log_len?.() >>> 0) > was) this.#cgenCatchUp(true);
        const now = this.#cgen?.have ?? 0;
        if (now !== was && said) console.warn(`[spare] pid ${this.#shim.exports.gex_my_pid?.() ?? '?'}: caught up with ${this.#cgen?.dyl.size ?? 0} modules while waiting`);
        // Waiting on a sibling's module (`#relayWait`): looked at again soon.
        delay = now !== was || this.#relayWait.size ? SPARE_POLL_MS : Math.min(2 * delay, SPARE_POLL_MAX_MS);
      }
      this.#spareWait = false;
      if (!go?.tid) {
        // Let go while the program it caught up with still runs: its blocks go.
        if ((this.#shim.exports.gex_exec_gen?.() ?? 0) === gen && !this.#shim.exports.gex_exited?.()) this.freeThreadTls();
        return { entry: 0n, ctx: 0, gone: true };
      }
      Object.assign(thread, go);
      // The `clone` that made this thread promised it a slot, and this
      // Worker has held one since it attached: kept now, or every thread a
      // spare takes keeps a slot promised for good and the process's
      // `clone`s fail once they add up to every slot.
      this.#promise = true;
      this.keepPromise();
      const had = this.#cgen?.dyl.size ?? 0;
      if (said) {
        const ta = performance.now();
        thread.spareSaid = () => console.warn(`[spare] tid ${thread.tid}: adopted, ${(this.#cgen?.dyl.size ?? 0) - had} modules more, in ${(performance.now() - ta).toFixed(1)} ms`);
      }
    }
    this.#threadTid = thread.tid;
    const _t = (s) => { if (globalThis.GEX_ATTACH_TRACE) console.error(`[thr ${thread.tid}] ${s}`); };
    _t('attached, adopting');
    const entry = this.#shim.exports.gex_adopt_thread(thread.tid);
    this.adoptGen = this.#shim.exports.gex_exec_gen?.() ?? 0;
    // Gone before its Worker came up: the process exec'd (the other threads go
    // with the old program) or ended. Nothing to run, and nothing wrong.
    if (!entry) return { entry: 0n, ctx: Number(thread.ctx), gone: true };
    if (thread.procOpts) this.#opts = thread.procOpts;
    // Nothing of the process's to set: its ABI, start time and decode mode
    // (`gex_set_compat`) are process constants in the shared memory, set
    // once by the process's own Worker before its first guest instruction.
    _t(`adopted, catchUp (${(this.#shim.exports.gex_module_count?.()>>>0)} modules)`);
    // Install every already-translated module into this Worker's own tables
    // before it runs an instruction. The funcref tables are per-thread (a wasm
    // table is not shareable) while the directory naming their slots is shared
    // and already written, so a fresh thread branches into a slot its own table
    // has never filled. Catching up on the next trap raced the branch and
    // deadlocked; doing it here once makes the first run deterministic.
    await this.#catchUp();
    // And the function table's installs (a wasm32 program's module, units):
    // its entry is a bridge into one.
    this.#cgenCatchUp();
    thread.spareSaid?.();
    return { entry, ctx: Number(thread.ctx) };
  }

  /** A thread of a restored session snapshot (`ProcessHost.restore`), in
   *  this Worker: as a started thread (`adoptThread`), on the memory the
   *  snapshot held, entering at the start its libc left in its record
   *  (`snap.pc`, `snap.sp`). */
  async snapThread(shimWasm, guestWasm, thread) {
    this.#threaded();
    this.#threadTid = thread.tid;
    this.#promise = false;
    // The first of a process's Workers: the memory's shim slots are the
    // snapshot's Workers', none of which runs now (as a fork child's).
    if (thread.snapReset) this.#forkChild = true;
    await this.attach(shimWasm, guestWasm, { devices: false });
    const e = this.#shim.exports;
    const entry = e.gex_snap_thread(thread.tid, BigInt(thread.snap.pc), BigInt(thread.snap.sp));
    if (!entry) throw new Error(`snapshot: process has no thread ${thread.tid}`);
    if (thread.procOpts) this.#opts = thread.procOpts;
    await this.#catchUp();
    this.#cgenCatchUp();
    // The wineserver's: its door, published again for the restored
    // session's Workers (the snapshot's area, in its memory).
    if (thread.snap.door !== undefined) this.#doorPublish(thread.snap.door >>> 0);
    return { entry, ctx: Number(e.gex_snap_ctx()) };
  }

  /** A restored thread, attached (`snapThread`), waits here until the host
   *  lets the whole session go on (the kernel's snapshot word cleared);
   *  its clocks then skip the time the session spent saved, which the host
   *  has added to the kernel's (`k_mono_held`). */
  snapRelease() {
    const kern = this.#kernelSpec?.memory, k = this.#kernel;
    if (kern && k?.k_snap_word) {
      const words = new Int32Array(kern.buffer), at = (k.k_snap_word() >>> 0) >>> 2;
      while (Atomics.load(words, at) !== 0) Atomics.wait(words, at, Atomics.load(words, at), 1000);
    }
    this.#monoHeldNs = null;
    this.#shim.exports.gex_snap_clock?.();
  }

  /**
   * posix_spawn's child (a wasm32 program's CLONE_VM | CLONE_VFORK clone),
   * run here before the parent's `clone` returns, as Linux runs it: in the
   * parent's memory, the parent suspended (hosts/browser/src/vfork.rs). The
   * child's frames end when it exits or execs, an exec going on in a new
   * Worker; the parent's process state comes back, and so do the program's
   * globals, which the child's entry set. Whether a child ran.
   */
  #vforkChild() {
    const e = this.#shim.exports;
    const saved = {
      shimSp: e.__stack_pointer.value,
      progSp: this.#wasmprogSp?.value,
      async: this.#wasmprogAsync?.map((g) => g.value),
      muts: this.#wasmprogMuts.map((g) => g.value),
    };
    const t0 = performance.now();
    const entry = e.gex_vfork_begin();
    if (!entry) return false;
    const pid = e.gex_vfork_pid();
    const ctx = Number(e.gex_fork_ctx());
    let why = 'exit';
    let crash = null;
    // The child is not stopped for a snapshot: the record is its parent's.
    this.#vforkDepth++;
    e.gex_snap_enable?.(0);
    try {
      this.#enterOf(ctx, entry)(ctx, entry);
    } catch (err) {
      if (!e.gex_handoff?.()) {
        why = 'crash';
        crash = err;
      } else {
        const w = e.gex_vfork_why();
        why = w === 2 ? 'exec' : w === 3 ? 'crash' : 'exit';
      }
    }
    if (--this.#vforkDepth === 0 && this.#snapRec) e.gex_snap_enable?.(1);
    // A trap abandons the shim's frames without restoring its stack pointer.
    e.__stack_pointer.value = saved.shimSp;
    // Frames that ended with no exit said: nothing will tell the parent.
    if (why === 'exit' && !e.gex_exited?.()) {
      why = 'crash';
      crash = new Error('its frames ended without an exit');
    }
    let exec = null;
    if (why === 'exec') {
      exec = this.#vforkExecRecord(pid, ctx, performance.now() - t0);
      e.gex_vfork_exec_done();
    } else if (why === 'crash') {
      if (crash) console.error(`[gex] pid ${pid} (posix_spawn's child) crashed: ${crash?.message ?? crash}`);
      // As a trapped Worker is reported.
      try { e.gex_process_gone?.(137); } catch {}
    }
    e.gex_vfork_end();
    if (this.#wasmprogSp && saved.progSp !== undefined) this.#wasmprogSp.value = saved.progSp;
    saved.async?.forEach((v, i) => { this.#wasmprogAsync[i].value = v; });
    saved.muts.forEach((v, i) => { this.#wasmprogMuts[i].value = v; });
    if (exec) {
      this.#env.spawnProcess(exec).catch((err) => console.error(`[gex] starting pid ${pid} after its exec: ${err.message ?? err}`));
    }
    this.#vforks++;
    vforkTrace(`[vfork] pid ${pid}: ${why} after ${(performance.now() - t0).toFixed(1)}ms`);
    return true;
  }

  /** A translated program's posix_spawn children, running in this
   *  Worker's run loop in place of the thread that cloned them (vfork.rs):
   *  the innermost last, a child that spawned waiting under its own. */
  #vfork = [];

  /** Start the staged vfork child here: where to enter it, or null when
   *  there was nothing to start. */
  #vforkEnter() {
    const e = this.#shim.exports;
    const t0 = performance.now();
    const entry = e.gex_vfork_begin();
    if (!entry) return null;
    const ctx = Number(e.gex_fork_ctx());
    this.#vfork.push({ t0, pid: e.gex_vfork_pid(), ctx });
    return { pc: entry, ctx };
  }

  /** The vfork child is done (its exit, its exec, or `crash`): hand an exec
   *  to a new Worker, and go back to the thread that cloned it, after its
   *  `clone`. */
  #vforkLeave(crash) {
    const e = this.#shim.exports;
    const { t0, pid, ctx } = this.#vfork.at(-1);
    const w = e.gex_vfork_why();
    let why = crash ? 'crash' : w === 2 ? 'exec' : w === 3 ? 'crash' : 'exit';
    let exec = null;
    if (why === 'exec') {
      exec = this.#vforkExecRecord(pid, ctx, performance.now() - t0);
      e.gex_vfork_exec_done();
    } else if (why === 'crash') {
      if (crash) console.error(`[gex] pid ${pid} (posix_spawn's child) crashed: ${crash?.message ?? crash}`);
      try { e.gex_process_gone?.(137); } catch {}
    }
    const resume = e.gex_vfork_end();
    this.#vfork.pop();
    if (exec) {
      this.#env.spawnProcess(exec).catch((err) => console.error(`[gex] starting pid ${pid} after its exec: ${err.message ?? err}`));
    }
    this.#vforks++;
    vforkTrace(`[vfork] pid ${pid}: ${why} after ${(performance.now() - t0).toFixed(1)}ms`);
    return { pc: resume, ctx };
  }

  /** The record of an exec a vfork child staged (`vfork::stage_exec`): a
   *  fork record with no image, which the host starts as any child, and
   *  whose Worker runs the exec (`#adoptExec`). */
  /** The record of the exec this process handed to another Worker (lib.rs
   *  `exec_away`: a program of the other memory flavour), null when it
   *  handed none: a posix_spawn child's exec record, for the same pid. */
  execAwayRecord() {
    const e = this.#shim.exports;
    if (!e.gex_exec_away?.()) return null;
    const r = this.#vforkExecRecord(e.gex_exec_away_of(0), e.gex_exec_away_of(1), 0);
    r.vforkMs = undefined;
    e.gex_vfork_exec_done();
    return r;
  }

  #vforkExecRecord(pid, ctx, ms) {
    const e = this.#shim.exports;
    const part = (i) => {
      const p = e.gex_vfork_exec_ptr(i) >>> 0;
      return this.#bytes().slice(p, p + (e.gex_vfork_exec_len(i) >>> 0));
    };
    return {
      pid,
      ctx: BigInt(ctx),
      pc: 0n,
      sp: 0n,
      tls: 0n,
      compat: 0,
      wasm32: 0n,
      brk: 0n,
      windowsSpace: 0,
      image: [],
      vmaBlob: null,
      slots: 0,
      pages: null,
      shared: null,
      modules: [],
      opts: this.#opts,
      exec: part(0),
      // A native wasm program's image: its fresh memory has no placement
      // window (`#noPlace`).
      wasmImage: (() => {
        const hi = Number(e.gex_vfork_exec_wasm?.(1) ?? 0n);
        return hi ? [Number(e.gex_vfork_exec_wasm(0)), hi] : null;
      })(),
      fds: part(1),
      cwd: new TextDecoder().decode(part(2)),
      state: part(3),
      exe: null,
      // The memory its program needs, in a 64-bit memory's page (vfork.rs
      // `exec_memory`): true or false, or undefined for the parent's.
      mem64: (() => { const m = e.gex_vfork_exec_mem64?.() ?? -1; return m < 0 ? undefined : m === 1; })(),
      // How long the child ran in its parent's Worker before this exec, for
      // the process's exit line: a child with this ran in place, nothing copied.
      vforkMs: ms,
    };
  }

  /** A `CLONE_VM | CLONE_VFORK` clone this Worker copies instead of
   *  running in place (`gex_vfork_ok` said why not): the child still runs,
   *  as a forked one, so nothing fails, only a spawn costs a copy of the
   *  memory. Said once per Worker, so a lost in-place path shows. */
  #vforkCopiedSaid = false;
  #vforkCopied(why) {
    vforkTrace(`[vfork] not here: ${why}`);
    if (this.#vforkCopiedSaid) return;
    this.#vforkCopiedSaid = true;
    const reason = { [-1]: 'vfork children nest too deep here', [-4]: "inside a provider's call", [-5]: 'a Windows process', [-6]: 'no global kernel', [-7]: 'no fork staged on this Worker' }[why] ?? `code ${why}`;
    console.error(`[gex] pid ${this.wlToken()}: a CLONE_VM|CLONE_VFORK child is copied, not run in place (${reason}); said once per Worker`);
  }

  /** How many posix_spawn children ran in this Worker (`#vforkChild`). */
  #vforks = 0;
  get vforks() {
    return this.#vforks;
  }

  /** The fork this thread staged, started (`#startFork`): its promise, or
   *  null with none staged. No promise otherwise, since a call-out's loop
   *  asks at every trap and may never reach the event loop. */
  #drainFork() {
    const pid = this.#shim.exports.gex_fork_pending?.();
    return pid ? this.#startFork(pid) : null;
  }

  async #startFork(pid) {
    const e = this.#shim.exports;
    let wasm32Link = null;
    const child = {
      pid,
      pc: e.gex_fork_pc(),
      ctx: e.gex_fork_ctx(),
      sp: e.gex_fork_sp(),
      tls: e.gex_fork_tls(),
      // A 32-bit process forks a 32-bit child; its Worker's shim must decode
      // in compatibility mode from the first block.
      compat: e.gex_compat?.() ?? 0,
      // A wasm32 program's entry: the child installs the same module.
      wasm32: e.gex_wasmprog_entry?.() ?? 0n,
      // And where its module sits in the function table: the child's copy
      // of the data holds function pointers for that base.
      wasm32Base: e.gex_wasmprog_base?.() ?? 0,
      // And its shared libraries: where each module's data and table range
      // are, and how the parent bound them. Said with the memory's copy,
      // below, under the link map's lock (`gex_fork_hold`).
      wasm32Link: null,
      // And this thread's blocks for its `dlopen`ed modules' thread-local
      // data, which the child's copy of memory holds (`#dlTlsInit`).
      dlTls: this.#dlTlsFork(),
      // Where the parent's heap had reached. Without the number the child's
      // kernel thinks it has no heap and grows one from zero, over its contexts.
      brk: e.gex_fork_brk?.() ?? 0n,
      // A 32-bit Wine process's Windows address space starts here (0: none); the
      // child splits its memory at the same line before its heap takes any.
      windowsSpace: e.gex_windows_space?.() ?? 0,
      // Whether its memory has the placement window: the child's is laid out
      // as its parent's (`#noPlace`).
      placeWindow: e.gex_place_window?.() ?? 1,
      image: (() => {
        const t0 = performance.now();
        // The process's other threads out of guest code while the memory is
        // copied, so the child's is one moment's (`gex_fork_hold`). Not in
        // the single-thread build: its threads are fibers on this thread, so
        // none runs during the copy, and its memory has no wait to hold one
        // with (one parked at a safepoint is not "out", and was waited for).
        if (!(SP_CODE && single.on)) e.gex_fork_hold?.();
        let img;
        try {
          wasm32Link = this.#wasmprogLink();
          img = this.#processImage();
        } finally {
          e.gex_fork_release?.();
        }
        this.#imageMs = performance.now() - t0;
        this.#imageBytes = img.reduce((n, r) => n + imageLen(r), 0);
        this.#imageRegions = img.length;
        return img;
      })(),
      // Where the image's runs are staged, in the global kernel's memory (0:
      // they travel as `bytes`). The child copies them out and frees it.
      imageStage: this.#imageStage,
      // The map, separately from the bytes: names and code identities the child
      // needs to recognise a page as belonging to an already-translated library.
      vmaBlob: this.#forkVmaBlob,
      // Every slot this process has claimed -- not the highest any module needed.
      // The directory is written when a page is *claimed*, not when its module
      // installs, so the max over inherited modules misses a page claimed after
      // the last install, and a branch there traps rather than misses.
      slots: e.gex_jit_slots_total?.() ?? 0,
      pages: this.#pageMap(),
      // Which object each page belongs to -- the only part that outlives the
      // child's first `execve`. See `gex_shared_dump`.
      shared: this.#sharedMap(),
      // What this process *holds*, not just what it has instantiated. Three
      // places a module can be: `#inherited`, `#lazyInherited` (not yet
      // instantiated -- an instantiated one is already in `#compiled`), and
      // `#compiled`. Carrying all of them keeps a grandchild from inheriting a
      // directory whose modules it lacks. `WebAssembly.Module` clones for free.
      modules: [
        ...this.#inherited,
        ...this.#lazyInherited.filter((l) => !l.inst),
        ...this.#compiled,
      ],
      opts: this.#opts,
      // The descriptors the child inherits, described rather than cloned: an
      // open file is an `Arc` in the parent's heap the child never sees.
      fds: e.gex_fork_fds_len
        ? this.#bytes().slice(
            e.gex_fork_fds_ptr() >>> 0,
            (e.gex_fork_fds_ptr() >>> 0) + (e.gex_fork_fds_len() >>> 0),
          )
        : null,
      // The dispositions, mask, umask and limits the child inherits.
      state: e.gex_fork_state_len?.()
        ? this.#bytes().slice(
            e.gex_fork_state_ptr() >>> 0,
            (e.gex_fork_state_ptr() >>> 0) + (e.gex_fork_state_len() >>> 0),
          )
        : null,
      // The parent's exe path and argv, NUL-separated.
      exe:
        e.gex_fork_exe_len && e.gex_fork_exe_len() > 0
          ? new TextDecoder().decode(
              this.#bytes().slice(
                e.gex_fork_exe_ptr() >>> 0,
                (e.gex_fork_exe_ptr() >>> 0) + (e.gex_fork_exe_len() >>> 0),
              ),
            )
          : null,
      // The parent's working directory, so the child resolves relative paths
      // where the parent stood. Decoded to a string; the child restages it.
      cwd:
        e.gex_fork_cwd_len && e.gex_fork_cwd_len() > 0
          ? new TextDecoder().decode(
              this.#bytes().slice(
                e.gex_fork_cwd_ptr() >>> 0,
                (e.gex_fork_cwd_ptr() >>> 0) + (e.gex_fork_cwd_len() >>> 0),
              ),
            )
          : null,
    };
    child.wasm32Link = wasm32Link;
    child.forkCost = {
      buildMs: Math.round(this.#imageMs),
      bytes: this.#imageBytes,
      regions: this.#imageRegions,
      // Absolute, not `performance.now()`: a Worker and the page have different
      // time origins, so a raw reading across them is skew, not a duration.
      sentAt: performance.timeOrigin + performance.now(),
    };
    // What a fork costs to hand over, per fork and in bytes: this record is the
    // largest thing crossing per fork and is copied three times. See task 145.
    if (globalThis.GEX_DEBUG_EXIT) console.error(
      `[fork] pid ${child.pid} image ${child.forkCost.bytes} bytes in ` +
        `${child.forkCost.regions} regions, built in ${child.forkCost.buildMs}ms, ` +
        // Every child inherits its parent's modules and adds its own, so this
        // list grows down the fork tree. The isolate that OOMs has been alive
        // about four seconds -- a freshly spawned child -- which points here
        // rather than at the forking parent. See task 145.
        `${child.modules.length} modules (${this.#inherited.length} inherited ` +
        `+ ${this.#compiled.length} own)`,
    );
    e.gex_fork_taken();
    if (globalThis.GEX_DEBUG_EXIT) {
      // The record as it leaves, not as the child reports it. A child that
      // enters at zero was either handed a zero or lost it on the way, and
      // those are different bugs in different files.
      console.error(
        `[gex] fork -> pid ${child.pid} pc=0x${child.pc.toString(16)} ` +
          `ctx=0x${child.ctx.toString(16)} sp=0x${child.sp.toString(16)} ` +
          `${child.image.length} regions, ${child.modules.length} modules`,
      );
    }
    if (!this.#env.spawnProcess) {
      throw new Error(
        `the guest forked pid ${pid} and this environment cannot start a process. ` +
          'Give the Env a `spawnProcess`, or the child will never run.',
      );
    }
    await this.#env.spawnProcess(child);
  }

  async #catchUp() {
    const e = this.#shim.exports;
    if (!e.gex_module_count) return;
    // A Worker that has attached but not yet been given a process has no
    // kernel to ask, and asking anyway is a panic rather than a zero. That is
    // the order a forked child arrives in: attach, then adopt.
    if (e.gex_has_host && !e.gex_has_host()) return;
    const have = e.gex_module_count() >>> 0;
    // Every slot the *process* has claimed, not the pending module's -- this
    // Worker is catching up on all of them at once.
    const need = (e.gex_jit_slots_total?.() ?? 0) >>> 0;
    const t0 = globalThis.GEX_ATTACH_TRACE ? performance.now() : 0;
    const comp0 = this.#compileMs;
    let n = 0, reused = 0;
    for (; this.#installed < have; this.#installed++) {
      const i = this.#installed;
      if (this.#leaveDense(i)) continue;
      // Copied out before compiling: the registry is in the guest's memory,
      // and another thread allocating can move it.
      const bytes = this.#registryBytes(i);
      if (!bytes) continue;
      const ptr = this.#registryRow(i).ptr, len = bytes.length;
      // A sibling Worker may already have compiled this module: reuse its
      // `WebAssembly.Module` (same registry bytes -> same module) rather than
      // paying `WebAssembly.compile` again. The bytes are still needed for the
      // metadata and table-need computation the install does regardless.
      const key = this.#registryKey(i, bytes);
      // Or one the page's unit cache has (another process's).
      const cached = (key ? this.#sharedModules.get(key) : undefined) ?? this.#unitGet(this.#unitKey(bytes)) ?? undefined;
      let who = 'jit2049';
      if (globalThis.GEX_DUMP_WHERE ?? true) who = `jit2049-${i}-at-${ptr.toString(16)}`;
      try {
        await this.#installModule(bytes, need, 64, null, who, cached);
      } catch (err) {
        if (err instanceof WebAssembly.CompileError) {
          // And whether the registry's copy is still what was read.
          const now = this.#bytes().slice(ptr, ptr + len);
          let diff = 0;
          for (let j = 0; j < len; j++) if (now[j] !== bytes[j]) diff++;
          console.error(`[gex] module ${i} at 0x${ptr.toString(16)}+0x${len.toString(16)}: ${diff} bytes differ from the copy compiled`);
        }
        throw err;
      }
      if (cached) { reused++; continue; }
      // Newly compiled here: hand it to the host to relay to siblings, and keep
      // it so this Worker does not compile it twice either.
      const mod = this.#compiled[this.#compiled.length - 1]?.mod;
      if (mod && key) {
        this.#sharedModules.set(key, mod);
        this.#deadKeys.delete(key);
        this.#shareRegistry(i, key, mod);
      }
      n++;
    }
    if (globalThis.GEX_ATTACH_TRACE && (n || reused)) {
      console.error(`[catchup] installed ${n + reused} modules in ${Math.round(performance.now() - t0)}ms`
        + ` (${Math.round(this.#compileMs - comp0)}ms compile, ${reused} reused from siblings)`);
    }
    if (globalThis.GEX_FILL_TIME) {
      // The first catch-up ends a Worker's start: from its runtime's
      // construction to holding what the process had translated.
      const started = this.#startedMs < 0 ? (this.#startedMs = performance.now() - this.#bornAt) : -1;
      console.warn(`[gex] ${this.#whoFills()}: caught up, ${this.#fillCounts()}${started >= 0 ? `, started in ${started.toFixed(1)} ms` : ''}`);
    }
  }
  /** When this runtime was made, and how long until its first catch-up
   *  (`?filltime=1`). */
  #bornAt = performance.now();
  #startedMs = -1;

  /**
   * A `WebAssembly.Module` a sibling Worker compiled, relayed by the host. Kept
   * by registry index so this Worker's `#catchUp` reuses it instead of
   * recompiling the same bytes. See `#sharedModules`.
   */
  /** A sibling Worker's dead modules (`#reapDead` there, relayed by the
   *  host): this Worker lets go of them too, so their code can go. */
  dropSharedModules(keys) {
    for (const k of keys ?? []) {
      if (typeof k !== 'string') continue;
      this.#sharedModules.delete(k);
      this.#deadKeys.add(k);
    }
  }

  addSharedModule(key, mod) {
    if (typeof key === 'string' && (mod instanceof WebAssembly.Module || isPlan(mod)) && !this.#sharedModules.has(key) && !this.#deadKeys.has(key)) {
      this.#sharedModules.set(key, mod);
    }
  }

  /** Hand the guest some standard input. Safe to call at any time. */
  pushStdin(bytes) {
    const p = this.#shim.exports.gex_alloc(bytes.length) >>> 0;
    this.#bytes().set(bytes, p);
    this.#shim.exports.gex_push_stdin(p, bytes.length);
    this.#shim.exports.gex_free(p, bytes.length);
  }

  /** Say there will be no more input. */
  endStdin() { this.#shim.exports.gex_stdin_eof(); }

  /**
   * Give the guest a network. `net` has the seven calls in `Net` below. Off by
   * default: a guest with no proxy should see a machine with no network, which
   * fails cleanly, not one whose connections hang.
   */
  useNetwork(net, { shared = false } = {}) {
    this.#net = net;
    // One network object every process of the machine reaches (the page's
    // sockets over net-bridge, or the single-thread build's one
    // object): its stream ids mean the same in every process, so a socket can
    // be handed to another process on the same stream (`net_dup`). A network
    // of this Worker's own (run-node's sockets, a Worker's own page-served hosts)
    // has ids no other process knows.
    this.#netShared = !!(net && shared);
    // Its events end a network wait (`#netWait`). Chained: the single-thread
    // build's scheduler wake is set before this.
    if (net && 'onActivity' in net) {
      const prev = net.onActivity;
      net.onActivity = (...a) => {
        prev?.apply(net, a);
        const r = this.#netWake;
        this.#netWake = null;
        r?.();
      };
    }
    // Guarded, because this runs before the shim exists (a Worker is given its
    // devices between construction and `adoptProcess`); `load`/`attach` set it
    // from `#net` too.
    this.#shim?.exports?.gex_set_network?.(net ? 1 : 0);
  }

  /**
   * Give the guest a graphics device. `gl` is a `MesaGlDevice` from page/src/mesa-gl.js.
   * Off by default, like the network. Best set before `load`, since the device
   * node is made while the filesystem is built; a later call still works.
   */
  /** How long a GL submit waited for the unwaited batch before it, to the
   *  shim's per-thread time split. */
  #glWaitNote = (ms) => this.#shim?.exports?.gex_gl_note_wait_ns?.(ms * 1e6);

  /** The function tier's whole-image units (`cgen/hosts/browser/src/cgen.rs`):
   *  1 use one where it exists, 2 also require one for a program image. The
   *  Env's `loadUnit(key)` returns `{ wasm, entries }` (bytes, or a compiled
   *  `WebAssembly.Module` for `wasm`) or null. */
  useCgen(mode) {
    this.#cgenMode = mode >>> 0;
    if (this.#shim) this.#shim.exports.gex_cgen_set_mode?.(this.#cgenMode);
  }

  #cgenMode = 0;
  /** No ELF export providers (cgen/elfprov.rs): A/B runs, every process of
   *  the run (a forked child's Worker is told too). Before the shim exists. */
  /** `?wasmdlls=soname=path,...`: libraries this process replaces with wasm64
   *  providers (cgen/libprov.rs). */
  #libprovConfig() {
    const e = this.#shim.exports;
    // `?straytrap=1`: a guest return to native Wine's pad past open
    // call-outs traps as "breakpoint" (cgen.rs `STRAY_TRAP`).
    if (globalThis.GEX_STRAY_TRAP) e.gex_set_stray_trap?.(1);
    // Images' keys and identities taken from the page, not hashed at each map
    // (cgen.rs `gex_cgen_identities`).
    if (Gex.identities && e.gex_cgen_identities) {
      const b = new TextEncoder().encode(Gex.identities);
      const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
      this.#bytes().set(b, p);
      e.gex_cgen_identities(p, b.length);
      e.gex_free?.(p, Math.max(8, b.length));
    }
    if (Gex.fjitLibs && e.gex_fjit_libs) {
      const b = new TextEncoder().encode(Gex.fjitLibs);
      const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
      this.#bytes().set(b, p);
      e.gex_fjit_libs(p, b.length);
      e.gex_free?.(p, Math.max(8, b.length));
    }
    for (const [v, f] of [[Gex.providerSources, e.gex_provider_sources], [Gex.stateful, e.gex_cgen_stateful]]) {
      if (!v || !f) continue;
      const b = new TextEncoder().encode(v);
      const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
      this.#bytes().set(b, p);
      f(p, b.length);
      e.gex_free?.(p, Math.max(8, b.length));
    }
    if (Gex.providerSkip && e.gex_provider_skip) {
      const b = new TextEncoder().encode(Gex.providerSkip);
      const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
      this.#bytes().set(b, p);
      e.gex_provider_skip(p, b.length);
      e.gex_free?.(p, Math.max(8, b.length));
    }
    if (Gex.elfUnits && e.gex_cgen_elf_units) {
      const b = new TextEncoder().encode(Gex.elfUnits);
      const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
      this.#bytes().set(b, p);
      e.gex_cgen_elf_units(p, b.length);
      e.gex_free?.(p, Math.max(8, b.length));
    }
    if (!Gex.wasmDlls || !e.gex_libprov_config) return;
    const b = new TextEncoder().encode(Gex.wasmDlls);
    const p = e.gex_alloc(Math.max(8, b.length)) >>> 0;
    this.#bytes().set(b, p);
    e.gex_libprov_config(p, b.length);
    e.gex_free?.(p, Math.max(8, b.length));
  }

  noElfProviders() {
    this.#noElfProviders = true;
    this.#shim?.exports.gex_elfprov_off?.();
  }
  #noElfProviders = false;
  /** This Worker's half of the function table:
   *  its `gex.ftab`, the runtime module's `resolve` and `bridge`, and how far
   *  through the table's log it is. Made at the first unit. */
  #cgen = null;

  /** A string the shim writes for unit `u` through `fn(u, buf, cap)`. */
  #cgenStr(fn, u) {
    const e = this.#shim.exports;
    const cap = 1024;
    const buf = e.gex_alloc(cap) >>> 0;
    try {
      const n = Math.min(fn(u, buf, cap) >>> 0, cap);
      return new TextDecoder().decode(this.#bytes().slice(buf, buf + n));
    } finally {
      e.gex_free?.(buf, cap);
    }
  }

  /** The unit module for `key`: a sibling's, or loaded and compiled here and
   *  offered to the siblings. Synchronous, so the nested callback loop can
   *  install too. */
  #cgenModule(key, got = null) {
    const name = `cgen:${key}`;
    let mod = this.#sharedModules.get(name);
    if (mod) {
      this.#unitGot.delete(key);
      return mod;
    }
    const tl = performance.now();
    // The bytes alone: the entries were published with the unit.
    got ??= this.#unitGot.get(key) ?? { wasm: this.#env.loadFile?.(`${key}/c.wasm`) ?? null };
    this.#unitGot.delete(key);
    if (!got?.wasm) throw new Error(`gex-cgen: unit ${key} was published but cannot be loaded here`);
    const t0 = performance.now();
    mod = got.wasm instanceof WebAssembly.Module ? got.wasm : new WebAssembly.Module(got.wasm);
    this.#compileMs += performance.now() - t0;
    // Start-up's largest costs, one line per unit (a page loads about 40):
    // `GEX_LOG=unit`.
    if (logOn('unit')) console.warn(`[unit] ${key.slice(0, 16)} ${((got.wasm.byteLength ?? 0) / 1048576).toFixed(1)} MiB: `
      + `load ${Math.round(t0 - tl)} ms, compile ${Math.round(performance.now() - t0)} ms, at ${Math.round(performance.timeOrigin + performance.now()) % 1e7}`);
    this.#sharedModules.set(name, mod);
    this.#offerModule(name, mod, got.wasm.byteLength ?? Infinity);
    return mod;
  }

  /** This Worker's function table and runtime instance, made at the first
   *  unit: every slot holds `resolve` until something is installed there,
   *  index 1 the shim's fallback. `gex.lookup` catches this Worker up on
   *  the table's log before asking the shim. */
  #cgenStore() {
    if (this.#cgen) return this.#cgen;
    const e = this.#shim.exports;
    const tables = this.#dispatchTables();
    // By its bytes: the map outlives the process in a pooled Worker, and
    // another process's runtime module (dense or not) imports other tables.
    const rptr = e.gex_ftab_runtime_ptr() >>> 0;
    const rbytes = this.#bytes().slice(rptr, rptr + (e.gex_ftab_runtime_len() >>> 0));
    const rkey = `ftab-runtime:${moduleContentKey(rbytes)}`;
    let rm = this.#sharedModules.get(rkey);
    if (!rm) {
      rm = new WebAssembly.Module(rbytes);
      this.#sharedModules.set(rkey, rm);
    }
    // A table64 beside a 64-bit memory: every module of such a process (units, translations,
    // providers) addresses its function table with i64.
    const ftab = new WebAssembly.Table(this.#wide ? { element: 'anyfunc', address: 'i64', initial: 0n } : { element: 'anyfunc', initial: 0 });
    const lookup = (ctx, addr) => {
      if (e.gex_ftab_behind()) this.#cgenCatchUp();
      const idx = e.gex_ftab_lookup(ctx, addr);
      // A translation the function JIT just published: in this Worker's
      // table before `resolve` calls it.
      if (e.gex_ftab_behind()) this.#cgenCatchUp();
      // A unit this Worker has deferred: instantiated now, before `resolve`
      // tail-calls into its slot.
      if (this.#unitsDeferred.length) this.#unitWant(Number(idx));
      return idx;
    };
    const gex = {
      memory: this.#memory,
      ftab,
      lookup,
      safepoint: this.#imports.safepoint,
      cache_op: this.#imports.cache_op,
      dispatch_miss: this.#imports.dispatch_miss,
    };
    for (let i = 0; i < TABLE_NAMES.length; i++) {
      gex[TABLE_NAMES[i]] = tables[i] ?? new WebAssembly.Table({ element: 'anyfunc', initial: 0 });
    }
    // Dense: the shim's runtime module dispatches through `blocks` too.
    if (this.#dense) gex.blocks = this.#guest.exports.blocks;
    const ri = new WebAssembly.Instance(rm, { gex });
    const bridge = logNamed('bridges') ? this.#countBridges(ri.exports.bridge) : ri.exports.bridge;
    this.#cgen = { ftab, resolve: ri.exports.resolve, bridge, lookup, have: 0, instances: [ri], dyl: new Map(), dlTls: new Map(), gex, alt: null };
    this.#entryReset();
    this.#cgenGrow();
    ftab.set(tIdx(this.#wide, 1), e.gex_ftab_fallback);
    return this.#cgen;
  }

  /** This Worker's runtime instance of the process's other mode
   *  (`gex_cgen::runtime::module_alt`, a WoW64 process's 32-bit code): its
   *  `bridge` serves a function bound in that mode's directory and goes on
   *  through that directory, and its `enter` starts a thread there. Made at
   *  the first use, against the directory the process has then. */
  #cgenAlt() {
    const st = this.#cgenStore();
    const e = this.#shim.exports;
    const dir = BigInt(e.gex_alt_dir?.() ?? 0);
    if (!dir) throw new Error('gex: a bridge of the other mode before the process has its directory');
    if (st.alt?.dir === dir) return st.alt;
    let rm = this.#sharedModules.get('ftab-runtime-alt');
    if (!rm) {
      const ptr = e.gex_ftab_runtime_alt_ptr() >>> 0;
      rm = new WebAssembly.Module(this.#bytes().slice(ptr, ptr + (e.gex_ftab_runtime_alt_len() >>> 0)));
      this.#sharedModules.set('ftab-runtime-alt', rm);
    }
    const alt_dir = this.#wide ? new WebAssembly.Global({ value: 'i64' }, dir) : new WebAssembly.Global({ value: 'i32' }, Number(dir) | 0);
    const ri = new WebAssembly.Instance(rm, { gex: { ...st.gex, alt_dir } });
    const bridge = logNamed('bridges') ? this.#countBridges(ri.exports.bridge) : ri.exports.bridge;
    st.instances.push(ri);
    st.alt = { dir, bridge, enter: ri.exports.enter };
    return st.alt;
  }

  /** Grow this Worker's function table past every index handed out, with
   *  room over (`#ftabSpare`), so an index another Worker published to the
   *  probe cache is in range here (holding `resolve`) before this one
   *  replays it. */
  #cgenGrow() {
    const st = this.#cgen;
    const e = this.#shim.exports;
    let size = e.gex_ftab_size() >>> 0;
    for (;;) {
      const want = size + this.#ftabSpare;
      const have = Number(st.ftab.length);
      if (have < want) {
        st.ftab.grow(tIdx(this.#wide, want - have), st.resolve);
        this.#tallyTables('ftab grew');
      }
      // Counted for the probe cache's indices at this length (cgen.rs
      // `gex_ftab_enter`); again if another Worker allocated meanwhile.
      if (!e.gex_ftab_enter) return;
      size = e.gex_ftab_enter(Number(st.ftab.length)) >>> 0;
      if (size <= Number(st.ftab.length)) return;
    }
  }

  /** Before guest code runs again from here: this Worker counted for the
   *  probe cache's indices, and its function table grown if another Worker
   *  allocated past its end meanwhile (cgen.rs `ftab_resume`). */
  #cgenGrowIfShort() {
    const st = this.#cgen;
    const e = this.#shim.exports;
    if (!st || !e.gex_ftab_enter) return;
    if ((e.gex_ftab_enter(Number(st.ftab.length)) >>> 0) > Number(st.ftab.length)) this.#cgenGrow();
  }

  /** The function table's room past the indices handed out. Only another
   *  Worker of the process (a thread) publishes an index this one has not
   *  replayed, so a process with one Worker needs little; growing every
   *  process's table by 65536 `resolve`s was 4.5 ms of each fork's child
   *  and 1.5 ms of each exec (the grow copies the whole table). A process
   *  takes the full room when it gets its second Worker (`#threaded`). */
  #ftabSpare = ftabSpareTest() || FTAB_SPARE_ALONE;

  /** This Worker's process has more than one Worker from now on: grow to the
   *  full room before any other Worker can publish an index. */
  #threaded() {
    const want = ftabSpareTest() || FTAB_SPARE_THREADED;
    if (this.#ftabSpare === want) return;
    this.#ftabSpare = want;
    if (this.#cgen) this.#cgenGrow();
  }

  /** Install unit `u` into this Worker: instantiate it with `__table_base` =
   *  `base` (its element segment fills that range of the function table),
   *  then point tier-1's dispatch slots for its entries at the bridge. */
  #cgenInstall(u, base, deferred = false) {
    const e = this.#shim.exports;
    const st = this.#cgenStore();
    // An export provider's module (`PROVIDER | number`, cgen.rs) installs as
    // a unit does, from its file in the unit cache.
    const prov = u >= 0x80000000 ? u - 0x80000000 : -1;
    // A function JIT translation (`FJIT | n`, fjit.rs): its bytes are the shim's.
    const jit = prov < 0 && u >= 0x40000000 ? u - 0x40000000 : -1;
    // Wine's PE module (native wasm32 Wine, cgen/wasmdll.rs): its bytes are
    // in the shim's heap, read from the prefix.
    const wasmdll = prov >= 0 && (e.gex_cgen_pmod_wasm_len?.(prov) >>> 0) > 0;
    // A native wasm32 program (cgen/wasmprog.rs): its bytes come the same way.
    const wasmprog = wasmdll && !!e.gex_cgen_pmod_wasmprog?.(prov);
    // A replaced Linux library's wasm64 provider (cgen/libprov.rs): the 64-bit provider_* imports.
    const libprov = wasmdll && !!e.gex_cgen_pmod_libprov?.(prov);
    // One built for wasm64 and lowered to this 32-bit memory (an LP64 wasm
    // library under an aarch64 program): its globals
    // stay i64, with an i32 copy of the table base for its element segment.
    // A wasm32 program built LP64 for wasm64 and lowered the same way (64-bit Wine),
    // told apart by its __memory_base import's type.
    const progLowered = wasmprog && !!e.gex_cgen_pmod_lowered?.(prov);
    // Wine's PE module in a 64-bit Wine process (the native PE core,
    // winepe.wasm): built LP64 and lowered as its program was, so i64 globals and
    // stack pointer. A Worker installs its program before any PE module.
    // Not WoW64's i386 one (the process's other mode): wasm32 as built.
    const dllLowered = wasmdll && !wasmprog && !libprov && this.#wasmprogWide && !this.#wide && !e.gex_cgen_pmod_alt?.(prov);
    const lowered = (libprov || progLowered || dllLowered) && !this.#wide;
    const w64 = this.#wide || lowered;
    if (wasmprog) {
      this.#wasmprogWide = this.#wide || progLowered;
      // `entry_call` takes the cached global as it is: one of the other
      // width is made again here, as `#wasmprogStack` would.
      if (this.#wasmprogSp && (typeof this.#wasmprogSp.value === 'bigint') !== this.#wasmprogWide) {
        this.#wasmprogSp = null;
        this.#entryReset();
      }
    }
    // A wasm32 program's module, and Wine's PE module, is keyed by its bytes
    // (`wasm:`), not its path: the page keeps those for every process
    // (host.mjs), so a program run again, by any process, is not compiled
    // again, and a file apt replaced is not mistaken for the old one. Keyed
    // by path, winepe.wasm (8.9 MB) was compiled once per Wine process, ten
    // times in one demo, each copy of its code resident.
    // The key from the shim (`gex_cgen_pmod_wasm_key`) where it makes one,
    // and the bytes copied out only when the page has no module for it.
    const plen = wasmdll ? e.gex_cgen_pmod_wasm_len(prov) >>> 0 : 0;
    // A wasm32 program's bytes as a copy the shim made for this Worker
    // (`gex_wasmprog_owned`): another Worker of the process may release
    // the shim's own while this one copies.
    const pcopy = () => {
      const own = wasmprog && e.gex_wasmprog_owned ? e.gex_wasmprog_owned(prov) >>> 0 : 0;
      if (wasmprog && e.gex_wasmprog_owned && !own) throw new Error(`wasm32 module ${prov}: its bytes are gone`);
      const at = own || (e.gex_cgen_pmod_wasm_ptr(prov) >>> 0);
      const b = new Uint8Array(this.#memory.buffer, at, plen).slice();
      if (own) e.gex_free(own, plen);
      return b;
    };
    let pbytes = null, pkey = null;
    // A wasm32 program's or shared library's binding list (cgen/dylink.rs),
    // read before its module is compiled: its exports are packed
    // (`packExports`, a separate module key) unless GEX_LOG=nodlpack.
    // Split once, here, for `#wasmprogBind` too.
    const bind = wasmprog ? this.#bindGet(prov) : null;
    let packed = false;
    // Wine's PE module too: copied and hashed here, its 9 MB were 28 copies
    // and hashes in one demo's boot.
    if ((wasmprog || (wasmdll && !libprov)) && e.gex_cgen_pmod_wasm_key) {
      const lane = (n) => BigInt.asUintN(64, BigInt(e.gex_cgen_pmod_wasm_key(prov, n)));
      pkey = `wasm:${plen}:${lane(0)}:${lane(1)}`;
      if (wasmprog && !w64 && !logNamed('nodlpack')) {
        if (this.#sharedModules.has(`${pkey}:p`)) {
          pkey = `${pkey}:p`;
          packed = true;
        } else if (!this.#sharedModules.has(pkey)) {
          const raw = pcopy();
          const t0 = performance.now();
          // The list's `X` slots, for packing only: made here, where a module
          // is packed, and nowhere else (an install whose packed module the
          // page has, a fork's child or a thread, has no use for them; they
          // were 65 ms of a Plasma fork child's installs).
          const xs = bind ? bindExtra(bind) : [];
          pbytes = packExports(raw, xs, base);
          if (logNamed('dlpack')) console.warn(`[dlpack] ${this.#cgenStr(e.gex_cgen_pmod_file, prov)}: ${pbytes ? `${raw.length} -> ${pbytes.length} bytes, ${xs.length} slots` : 'not packed'} in ${(performance.now() - t0).toFixed(1)} ms`);
          if (pbytes) { pkey = `${pkey}:p`; packed = true; } else pbytes = raw;
        }
      }
      if (!packed && !this.#sharedModules.has(pkey)) pbytes ??= pcopy();
    } else if (wasmdll) {
      pbytes = pcopy();
      pkey = `wasm:${moduleContentKey(pbytes)}`;
    }
    // What gex-eject's 32-bit Wine recorder (site/record-wine32.mjs) collects: the
    // native wasm32 modules a session ran.
    if (wasmdll) {
      const file = this.#cgenStr(e.gex_cgen_pmod_file, prov);
      if (logOn('record')) console.warn(`[gex-record] module ${file} ${pkey.slice(5)}`);
      // The same, structured: a native module (a Wine builtin), a process's
      // program when it is an .exe (fetch-note.mjs).
      const path = file.replace(/^wasm(dll|prog):/, '');
      this.#env.noteStatus?.({ kind: 'module', path, program: wasmprog ? undefined : (/\.exe$/i.test(path) ? this.cmdline : undefined) });
    }
    const mod = wasmdll
      ? this.#cgenFileModule(this.#cgenStr(e.gex_cgen_pmod_file, prov), pbytes, pkey)
      : prov >= 0
      ? this.#cgenFileModule(this.#cgenStr(e.gex_cgen_pmod_file, prov))
      : jit >= 0 ? this.#fjitModule(jit)
      : this.#cgenModule(this.#cgenStr(e.gex_cgen_key, u));
    // A stateful provider: its data in the guest's
    // memory, this Worker's provider stack, and the provider_* imports.
    // (above 4 GiB for Wine's PE module in a 64-bit memory: the upper half)
    const memBase = prov >= 0 ? (e.gex_cgen_pmod_memory_base?.(prov) >>> 0) + (e.gex_cgen_pmod_memory_base_hi?.(prov) >>> 0) * 2 ** 32 : 0;
    const regsOf = prov >= 0 ? (b, c) => e.gex_cgen_pmod_regs(prov, b, c)
      : jit >= 0 ? (b, c) => e.gex_fjit_regs(jit, b, c)
      : (b, c) => e.gex_cgen_regs(u, b, c);
    const tables = this.#growTables((prov >= 0 ? e.gex_cgen_pmod_slots(prov) : jit >= 0 ? e.gex_fjit_slots(jit) : e.gex_cgen_slots(u)) >>> 0, 'cgen');
    const tInst = performance.now();
    let inst;
    const imports = {
      env: {
        memory: this.#memory,
        __indirect_function_table: st.ftab,
        // i64 in a 64-bit memory's process (wasm64 PIC modules).
        __table_base: new WebAssembly.Global({ value: w64 ? 'i64' : 'i32', mutable: false }, w64 ? BigInt(base >>> 0) : base | 0),
        __table_base32: new WebAssembly.Global({ value: 'i32', mutable: false }, base | 0),
        __memory_base: new WebAssembly.Global({ value: w64 ? 'i64' : 'i32', mutable: false }, w64 ? BigInt(memBase) : memBase),
        // A unit's functions never use clang's shadow stack (those are stubs),
        // but a position-independent module imports its pointer all the same.
        // A stateful provider's is this Worker's (one C stack per thread).
        // A wasm32 program sets its own at each thread's entry.
        // and its shared libraries share it (one per thread, as natively).
        // (Wine's PE module of a twin install in a 64-bit memory's process,
        // winepe.wasm.m64: built for wasm64 and not lowered, so i64 too.)
        __stack_pointer: memBase && !wasmprog ? (libprov || dllLowered || (wasmdll && this.#wide) ? this.#providerStack64() : this.#providerStack())
          : wasmprog ? this.#wasmprogStack()
          : new WebAssembly.Global(this.#wide ? { value: 'i64', mutable: true } : { value: 'i32', mutable: true }, this.#wide ? 0n : 0),
        // clang's three names for a C main, and WinMain: Wine's crt_main
        // references all of them weakly and a position-independent module (32-bit Wine's PE module)
        // keeps the undefined ones as imports. Null through `GOT.func`, as a
        // static link leaves them, and a trap behind the null check.
        ...(wasmdll ? Object.fromEntries(MAIN_NAMES.map((n) => [n, () => { throw new Error(`${n}: a main the program does not define was called`); }])) : {}),
      },
      // (i64 for a module built for wasm64, lowered or not: 64-bit Wine's
      // wineboot.exe, a module of its own, names __original_main)
      ...(wasmdll ? { 'GOT.func': Object.fromEntries(MAIN_NAMES.map((n) => [n, w64
        ? new WebAssembly.Global({ value: 'i64', mutable: true }, 0n)
        : new WebAssembly.Global({ value: 'i32', mutable: true }, 0)])) } : {}),
      gex: {
        ...this.#imports, lookup: st.lookup, ftab: st.ftab,
        // native PE code's ICALL bound: this Worker's (`#raiseIcallLimit`)
        icall_limit64: this.#icallLimit[0], icall_limit: this.#icallLimit[1],
        // Built-in providers' host functions (gex-cgen/providers): nt_qpc's
        // clock, crt_heap's pages.
        ...(prov >= 0 ? {
          // The shim's exports themselves: a wasm-to-wasm call, no JS.
          nt_qpc: e.gex_nt_qpc,
          clock_ns: e.gex_clock_ns,
          // A provider's fallback to the lifted body (provider.h gp_lifted).
          ftab_twin: e.gex_ftab_twin,
          provider_alloc_top: e.gex_provider_alloc_top,
          // Wine's Unix side : its system-call dispatcher and fault
          // hook, process-wide in the shim (the page's delivery of faults
          // is still to come).
          // (a test's, GEX_TEST_UD2_ENTRY: 64-bit Wine's direct entries,
          // ops 0 and 4, never registered, so its every system and Unix
          // call takes the ud2 route)
          unix_dispatcher: (op, idx) => (globalThis.GEX_TEST_UD2_ENTRY && (op === 0 || op === 4)) ? 0 :
            e.gex_unix_dispatcher?.(op, addrNum(idx)) ?? 0,
          // 64-bit Wine's guarded call (the debugging switch GEX_NATIVE_TRY):
          // fn(arg), and a memory access out of bounds inside it (a bad pointer
          // Wine's code wrote or read through) caught here, every wasm frame
          // since gone: 0xc0000005 (STATUS_ACCESS_VIOLATION), else what fn
          // returned. Index 0 asks whether it is on: 2, else -1 (always in the
          // single-thread build, where a blocking call under it would need a
          // promise each). The high bit of `idx`: fn takes an i64. Its function
          // is cached by index: the callers' (Wine's system call entry, wow64's
          // thunks) stay put.
          native_try: single.on || !globalThis.GEX_NATIVE_TRY ? () => -1 : (idx, arg) => this.#nativeTry(idx, arg),
          fault_hook: (idx) => e.gex_fault_hook_set?.(addrNum(idx)),
          signal_hook: (idx) => e.gex_signal_hook_set?.(addrNum(idx)),
          code_changed: (addr, len) => e.gex_code_changed?.(addrBig(addr), len),
          wine_pe_slot: (ptr, len) => e.gex_wine_pe_slot?.(addrNum(ptr), len) ?? 0,
          // A memory64 program's context address is an i64 (Wine's
          // gex_addr_t); the shim's export takes the i32 every context is.
          ...(wasmprog && this.#wide ? { sysreg_write: (ctx, pc, reg, v) => this.#imports.sysreg_write(addrNum(ctx), pc, reg, v) } : {}),
          // Wine's Unix side runs the program's code nested
          // (signal_wasm32.c's `gex_call_out(ctx, pc, 0, 0)`, a call-out
          // that never returns for a thread's loop), and reports each image
          // it maps (natively for unit placement; the page's units path
          // takes it once `gex_image_mapped` exists).
          // In the single-thread build on a stack that may suspend, as the
          // function tier's call_out: 64-bit Wine's system calls are faults served
          // inside this loop (`#callOutSteps`), and they block.
          call_out: (SP_CODE && single.on)
            ? new WebAssembly.Suspending((ctx, target, ret, spWant) => (this.#hardNested()
              ? this.#callOut(ctx >>> 0, target, ret, spWant)
              : this.#callOutAsync(ctx >>> 0, target, ret, spWant)))
            : this.#callOutFront(),
          // The memory gex mapped for native modules, for Wine's views (cgen.rs `gex_host_data`).
          host_data: (from, out, max) => e.gex_host_data?.(from >>> 0, addrNum(out), max >>> 0) ?? 0,
          image_mapped: (base, fd) => {
            const number = e.gex_image_mapped?.(addrBig(base), fd) ?? -1;
            if (number === -1) return;
            // An i386 image with a function-tier unit to fetch (cgen.rs
            // `wine32_mapped`): served here, before Wine's loader goes on. (-2
            // is a later skeleton bound into the install.)
            if (number === -3) {
              this.#cgenServe();
              return;
            }
            this.#cgenCatchUp();
            if (number >= 0) {
              e.gex_cgen_pmod_installed(number);
              e.gex_wasmdll_installed();
            }
            this.#wasmdllCatchUp();
          },
          // libgex's dlopen past its static registry (Wine's Unix side links
          // it): no shared libraries here
          dlopen: () => 0,
          dlsym: () => 0,
        } : {}),
        ...(memBase && !wasmprog ? (libprov ? this.#providerImports64() : this.#providerImports()) : {}),
        // its runtime's heap: fresh guest pages
        ...(lowered ? { provider_alloc: (size) => e.gex_provider_alloc(size) } : {}),
        ...(wasmprog ? this.#wasmprogImports(() => inst, mod) : libprov ? { syscall: this.#providerSyscall } : wasmdll ? {
          syscall: (SP_CODE && single.on) ? this.#nativeSyscall : (e.gex_native_syscall ?? this.#nativeSyscall),
          // Where this process has KUSER_SHARED_DATA (64-bit Wine's PE side reads it there).
          kuser_address: () => (e.gex_kuser_address ? e.gex_kuser_address() : 0x7ffe0000) | 0,
          // The native DLL a function pointer or static variable is in (ntdll's
          // LdrFindEntryForAddress, when no image holds the address).
          image_of: (addr) => e.gex_wasmdll_image_of ? e.gex_wasmdll_image_of(BigInt.asUintN(64, BigInt(addr))) : 0n,
        } : {}),
      },
    };
    // A unit split by function (gex_split): its cold clusters' loader,
    // or the recorder's hit map of an instrumented unit.
    if (prov < 0 && jit < 0) this.#splitImports(mod, this.#cgenStr(e.gex_cgen_key, u), imports);
    // Wine's PE module of the snapshot flavour (pelink.py): its entries from
    // the guest are levels of the thread's (`#snapPeStop`), its call-outs
    // the flavour's (`#snapPeCallOut`), and its system and Unix calls go by
    // the host's route, a call-out to the guest's entry, not the Unix
    // side's direct entries (ops 1 and 5), which a stop could not unwind.
    const peSnap = dllLowered && !(SP_CODE && single.on) && WebAssembly.Module.exports(mod).some((x) => x.name === '__gexw_snap_globals');
    if (peSnap) {
      const g = imports.gex, ud = g.unix_dispatcher;
      g.unix_dispatcher = (op, idx) => (op === 1 || op === 5 ? 0 : ud(op, idx));
      g.snap_pe_stop = (ctx, depth, need, ret, body) => this.#snapPeStop(ctx, depth, need, ret, body);
      if (this.#snapHooks) g.call_out = (ctx, target, ret, spWant) => this.#snapPeCallOut(ctx >>> 0, target, ret, spWant);
      this.#snapDepth ??= new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
      Object.assign(imports.env, {
        __asyncify_state: new WebAssembly.Global({ value: 'i32', mutable: true }, 0),
        __asyncify_data: new WebAssembly.Global({ value: 'i32', mutable: true }, 0),
        __gexw_snap_depth: this.#snapDepth,
      });
    }
    // A wasm32 program's or shared library's binding list (cgen/dylink.rs).
    const extra = wasmprog ? this.#wasmprogBind(prov, imports, st, bind) : [];
    // One of Wine's PE modules with imports of the others (a DLL linked as
    // a module of its own, or the core calling into one): its binding list.
    const peMod = wasmdll && !wasmprog && !libprov;
    if (peMod) this.#wasmdllBind(prov, imports, st);
    // What a door instance needs to run this program again on its memory
    // (a wineserver: `#doorPublish`).
    if (wasmprog) this.#wasmprogInfo = { module: mod, base, memBase, prov, binds: bind?.text ?? null, packed };
    inst = new WebAssembly.Instance(mod, imports);
    if (peMod) {
      st.dyl.set(prov, inst);
      this.#wasmdllLate(st, prov);
    }
    if (peSnap) {
      this.#snapPe = { inst, state: imports.env.__asyncify_state };
      this.#snapSplit();
    }
    // Not a spare's to apply: it may install a module before the Worker that
    // published it, which then runs it at once (`#spareWait`).
    let relocate = 0;
    if (!this.#spareWait && wasmprog) {
      // Another Worker applying them is waited for (the shim's side).
      if (e.gex_wasmprog_relocate(prov)) {
        try {
          inst.exports.__wasm_apply_data_relocs?.();
        } finally {
          e.gex_wasmprog_relocated?.(prov);
        }
      }
    } else if (!this.#spareWait && memBase && (relocate = e.gex_provider_relocate(memBase))) {
      // Another Worker applying them is waited for (the shim's side), and
      // one that never finished ends this one.
      if (relocate < 0) throw new Error(`provider module ${prov} (${this.#cgenStr(e.gex_cgen_pmod_file, prov)}): another Worker took its data relocations and never applied them`);
      try {
        inst.exports.__wasm_apply_data_relocs?.();
      } finally {
        e.gex_provider_relocated?.(memBase);
      }
    }
    if (wasmprog) {
      // (a packed module's element segments filled them)
      if (!packed) for (const [slot, name] of extra) {
        const f = inst.exports[name];
        if (!f) throw new Error(`wasm32 module ${prov}: ${name} is not exported`);
        st.ftab.set(tIdx(this.#wide, slot), f);
      }
      st.dyl.set(prov, inst);
      // Instantiated: the shim's copy of a library's bytes can go
      // (`gex_wasmprog_release`); a Worker that needs them again reads them.
      e.gex_wasmprog_release?.(prov);
      if (this.#wasmprogTls) this.#dlTlsInit(prov, inst, st, this.#wasmprogTls);
    }
    // A Wine PE module with thread-local data (a gexcc-win program's C++
    // runtime): its start function gives a block only to the first
    // instance to run it, so each of this Worker's gets one of its own.
    else if (wasmdll && inst.exports.__wasm_init_tls && inst.exports.__tls_size) {
      const size = Number(inst.exports.__tls_size.value);
      const align = Number(inst.exports.__tls_align?.value ?? 1);
      const at = this.#dlTlsAlloc(prov, size, align);
      inst.exports.__wasm_init_tls(typeof inst.exports.__tls_size.value === 'bigint' ? BigInt(at) : at);
      this.#tlsOwners.push({ prov, inst, size, align });
    }
    // A deferred unit's bridges were written when it was deferred.
    const bridges = deferred ? 0 : this.#cgenBridges(regsOf, tables);
    st.instances.push(inst);
    // The first Worker to install a JIT module writes the directory for the
    // pages it numbered (idempotent).
    if (jit >= 0) e.gex_fjit_installed(jit);
    // What each Worker pays to install a unit (instantiation fills its slots).
    const ms = performance.now() - tInst;
    // GEX_LOG=wasmlib: each native wasm32 module a process installs, what its
    // compile and its instantiation took.
    if (wasmdll && logNamed('wasmlib')) {
      const c = this.#fileModuleMs;
      console.warn(`[wasmlib] pid ${e.gex_my_pid?.() ?? '?'} w${Math.round(performance.timeOrigin) % 1e6}: ${this.#cgenStr(e.gex_cgen_pmod_file, prov).split('/').pop()} ${c < 0 ? 'had' : `compiled ${c.toFixed(1)} ms`}${this.#spareWait ? ' (spare)' : ''}, instantiated ${ms.toFixed(1)} ms, at ${Math.round(performance.timeOrigin + performance.now()) % 1e7}`);
    }
    if (prov < 0 && jit < 0 && ms >= 20) console.error(`[unit] installed at ${base} in ${Math.round(ms)} ms (${bridges} bridges)`);
    return bridges;
  }

  /** Point tier-1's dispatch slots for a module's entries at the bridge
   *  (`regsOf`: its `[table, slot]` pairs, cgen.rs `bridges`); how many. */
  #cgenBridges(regsOf, tables) {
    const e = this.#shim.exports;
    const st = this.#cgen;
    const n = regsOf(0, 0) >>> 0;
    const buf = e.gex_alloc(4 * Math.max(1, n)) >>> 0;
    let regs;
    try {
      regsOf(buf, n);
      regs = new Uint32Array(this.#memory.buffer.slice(buf, buf + 4 * n));
    } finally {
      e.gex_free?.(buf, 4 * Math.max(1, n));
    }
    // Table 8 (`gex_emit::dense::TABLE_INDEX`) is `blocks`: bridged slots
    // share entries there (cgen.rs `bridges`).
    const blocks = this.#guest.exports.blocks;
    if (blocks) this.#growBlocks();
    this.#writeBridges(tables, blocks, regs, st.bridge);
    if (this.#bridgeCount) this.#bridgeCount.bridged += regs.length >>> 1;
    return regs.length >>> 1;
  }

  /** The units this Worker's catch-up deferred (`#cgenDefer`), by base: each
   *  `{ u, base, end, retired }`, its install's range `[base, end)` of the
   *  function table still holding `resolve`, and the slots retired
   *  meanwhile. */
  #unitsDeferred = [];

  /** Units are instantiated in a Worker when it first calls one, not when
   *  it catches up: a process with units for a desktop's 400 libraries
   *  instantiated every one in each of its threads' Workers, ~320 instances
   *  and their tables per thread, in the one renderer every Worker shares.
   *  A deferred unit's slots hold `resolve`, which asks `lookup` for the
   *  index (`#unitWant` instantiates it there) and calls it; tier-1 reaches
   *  it through its bridges, written at once. Not in the single-thread
   *  build, nor for providers or function-JIT modules. */
  #cgenDefer(u, base, len) {
    const e = this.#shim.exports;
    const tables = this.#growTables(e.gex_cgen_slots(u) >>> 0, 'cgen');
    this.#cgenBridges((b, c) => e.gex_cgen_regs(u, b, c), tables);
    const end = base + len;
    // Slots another unit had and gave up: that unit is gone from here.
    const d = this.#unitsDeferred.filter((x) => x.end <= base || x.base >= end);
    let i = d.length;
    while (i > 0 && d[i - 1].base > base) i--;
    d.splice(i, 0, { u, base, end, retired: null });
    this.#unitsDeferred = d;
  }

  /** Instantiate the deferred unit holding function-table index `idx`, if
   *  one does: off the list first, so a call it makes while it starts finds
   *  it in or instantiates another. */
  #unitWant(idx) {
    const d = this.#unitsDeferred;
    let lo = 0, hi = d.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (d[mid].end <= idx) lo = mid + 1;
      else hi = mid;
    }
    const x = d[lo];
    if (!x || idx < x.base) return;
    d.splice(lo, 1);
    if (logOn('unit')) console.error(`[unit] ${this.#whoFills()}: unit at ${x.base} instantiated at its first call (${d.length} still deferred)`);
    const timed = logNamed('unitstall');
    const t0 = timed ? performance.now() : 0;
    const had = timed && this.#sharedModules.has(`cgen:${this.#cgenStr(this.#shim.exports.gex_cgen_key, x.u)}`);
    this.#cgenInstall(x.u, x.base, true);
    if (timed) console.warn(`[unitstall] ${this.#whoFills()}: ${(performance.now() - t0).toFixed(1)} ms ${had ? 'had' : 'compiled'}`);
    // Its element segment filled every slot; those retired since go back.
    if (x.retired) for (const r of x.retired) this.#cgen.ftab.set(tIdx(this.#wide, r), this.#cgen.resolve);
  }

  /** A retired function-table slot of a deferred unit: kept so, when the
   *  unit is instantiated. */
  #unitRetired(idx) {
    for (const x of this.#unitsDeferred) {
      if (idx >= x.base && idx < x.end) (x.retired ??= new Set()).add(idx);
    }
  }

  /** Each module's imports, as `module.name`. */
  #modImports = new WeakMap();

  /** The imports a split unit's primary takes beyond a unit's: `gex.cold(c)`,
   *  which instantiates cold cluster `c` here (its start function fills its
   *  slots in this Worker's table, and the stub that called goes on to the
   *  real body); a pruned unit's `gex.pruned(slot)`, which traps naming the
   *  pruned function's address; and an instrumented unit's `gexcov.hit(slot)`. */
  #splitImports(mod, key, imports) {
    let names = this.#modImports.get(mod);
    if (!names) {
      names = new Set(WebAssembly.Module.imports(mod).map((i) => `${i.module}.${i.name}`));
      this.#modImports.set(mod, names);
    }
    if (names.has('gex.cold')) {
      const loaded = new Set();
      imports.gex = { ...imports.gex, cold: (c) => {
        c >>>= 0;
        if (loaded.has(c)) throw new Error(`gex-cgen: ${key}: cold cluster ${c} is in, yet its stub ran`);
        loaded.add(c);
        const t0 = performance.now();
        const m = this.#coldModule(key, c);
        new WebAssembly.Instance(m, { env: imports.env, gex: imports.gex });
        this.#coldMs += performance.now() - t0;
      } };
    }
    if (names.has('gex.pruned')) {
      // A pruned unit (gex_split::prune): a slot static reachability left
      // out ran. Never recovered from: name its guest address and trap.
      const sec = WebAssembly.Module.customSections(mod, 'gex.funcs')[0];
      const dv = sec ? new DataView(sec) : null;
      imports.gex = { ...imports.gex, pruned: (k) => {
        k >>>= 0;
        const pc = dv && 12 + 8 * k <= dv.byteLength ? dv.getBigUint64(4 + 8 * k, true) : null;
        const msg = `gex-cgen: ${key}: pruned function ${pc === null ? '?' : '0x' + pc.toString(16)} (slot ${k}) ran`;
        console.error(`[pruned] ${msg}`);
        throw new Error(msg);
      } };
    }
    if (names.has('gexcov.hit')) {
      // A byte per slot: the slot count is the first word of `gex.funcs`.
      const sec = WebAssembly.Module.customSections(mod, 'gex.funcs')[0];
      const n = sec ? new DataView(sec).getUint32(0, true) : 65536;
      imports.gexcov = { hit: this.#covHit(key, n) };
    }
  }

  /** Time this Worker spent bringing in cold clusters, and how many. */
  #coldMs = 0;
  #coldIndex = new Map();

  /** Cold cluster `c` of the unit `key`, compiled: a sibling's, or fetched
   *  (a range of `<key>/cold.bin`, by `<key>/cold.json`) and compiled here. */
  #coldModule(key, c) {
    const name = `cold:${key}:${c}`;
    let mod = this.#sharedModules.get(name);
    if (mod) return mod;
    let ix = this.#coldIndex.get(key);
    if (!ix) {
      // `cold.idx`: u32 unpacked length, then a zstd frame of u32 n, the n
      // clusters' compressed lengths, then their unpacked lengths.
      const b = this.#env.loadFile?.(`${key}/cold.idx`) ?? null;
      const z = this.#env.zstd;
      if (!b) throw new Error(`gex-cgen: ${key}: a split unit with no cold.idx`);
      if (!z) throw new Error(`gex-cgen: ${key}: a split unit, and this Worker has no zstd decoder`);
      const raw = z.decode(b.subarray(4), new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true));
      const r = raw.slice();
      const w = new Uint32Array(r.buffer, 0, r.byteLength >> 2);
      const n = w[0];
      const offs = new Float64Array(n + 1);
      for (let i = 0; i < n; i++) offs[i + 1] = offs[i] + w[1 + i];
      ix = { n, w, offs };
      this.#coldIndex.set(key, ix);
    }
    if (!(c >= 0 && c < ix.n)) throw new Error(`gex-cgen: ${key}: no cold cluster ${c}`);
    const off = ix.offs[c], zlen = ix.w[1 + c], len = ix.w[1 + ix.n + c];
    const t0 = performance.now();
    let bytes = this.#env.loadRange?.(`${key}/cold.bin`, off, zlen) ?? null;
    if (!bytes) throw new Error(`gex-cgen: ${key}: cold cluster ${c} (${zlen} bytes at ${off}) could not be read`);
    const t1 = performance.now();
    bytes = this.#env.zstd.decode(bytes, len);
    const t2 = performance.now();
    mod = new WebAssembly.Module(bytes);
    const t3 = performance.now();
    if (logOn('unit')) console.warn(`[cold] ${key.slice(0, 16)} ${c}: ${zlen} bytes (${len} unpacked), fetch ${(t1 - t0).toFixed(1)} ms, `
      + `decode ${(t2 - t1).toFixed(1)} ms, compile ${(t3 - t2).toFixed(1)} ms, at ${Math.round(performance.timeOrigin + performance.now()) % 1e7}`);
    this.#sharedModules.set(name, mod);
    this.#offerModule(name, mod, bytes.byteLength);
    return mod;
  }

  /** An instrumented unit's hit map in this Worker (a byte per slot), and the
   *  slots already said. */
  #cov = new Map();
  #covTimer = null;
  #covHit(key, slots) {
    let h = this.#cov.get(key);
    if (!h) {
      // One memory per Worker for every unit's counts (each Memory reserves
      // address space: one per unit ran a page out of it), a u32 per slot at
      // an offset per unit. The first entry also calls `f` with the slot,
      // which notes the time (the profile's first use). The count saturates.
      // (module (import "m" "m" (memory 1)) (import "m" "o" (global i32))
      //   (import "m" "f" (func $f (param i32)))
      //   (func (export "hit") (param $s i32) (local $a i32) (local $c i32)
      //     local.get $s global.get 0 i32.add i32.const 2 i32.shl local.tee $a
      //     i32.load local.tee $c i32.eqz if local.get $s call $f end
      //     local.get $a local.get $c i32.const 1 i32.add i32.const -1
      //     local.get $c i32.const -1 i32.ne select i32.store))
      const wasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 1, 127, 0, 2, 21, 3, 1, 109, 1, 109, 2, 0, 1,
        1, 109, 1, 111, 3, 127, 0, 1, 109, 1, 102, 0, 0, 3, 2, 1, 0, 7, 7, 1, 3, 104, 105, 116, 0, 1, 10, 47, 1, 45, 1, 2,
        127, 32, 0, 35, 0, 106, 65, 2, 116, 34, 1, 40, 2, 0, 34, 2, 69, 4, 64, 32, 0, 16, 0, 11, 32, 1, 32, 2, 65, 1, 106,
        65, 127, 32, 2, 65, 127, 71, 27, 54, 2, 0, 11]);
      this.#covMod ??= new WebAssembly.Module(wasm);
      this.#covMem ??= wasmMemory({ initial: 1 }, 'the coverage counters', this.#kernelSpec?.spare);
      const off = this.#covUsed;
      this.#covUsed += slots;
      const need = Math.ceil((4 * this.#covUsed) / 65536) - this.#covMem.buffer.byteLength / 65536;
      if (need > 0) this.#covMem.grow(need);
      // First use, in ms of the epoch (performance.timeOrigin + now(), so
      // every Worker's and the page's times compare).
      const first = new Map();
      const f = (slot) => { first.set(slot, Math.round(performance.timeOrigin + performance.now())); };
      const inst = new WebAssembly.Instance(this.#covMod, { m: { m: this.#covMem, o: new WebAssembly.Global({ value: 'i32' }, off), f } });
      h = { off, n: slots, hit: inst.exports.hit, said: new Uint8Array(slots), sent: new Uint32Array(slots), first };
      this.#cov.set(key, h);
      this.#covTimer ??= setInterval(() => this.covFlush(), 2000);
    }
    return h.hit;
  }
  #covMod = null;
  #covAt = 0;
  #covMem = null;
  #covUsed = 0;

  /** Say the slots instrumented units were entered at since the last time:
   *  `[gex-cov] <key> <slot or a-b>,...` (site/record-wine32.mjs), and
   *  the profile, `[gex-prof] <key> <slot>:<first>:<calls>,...`: each slot
   *  entered since, its first entry (ms of the epoch; empty once said) and
   *  its entries since the last line (gexc units profile adds them up). */
  covFlush() {
    for (const [key, h] of this.#cov) {
      const m = new Uint32Array(this.#covMem.buffer, 4 * h.off, h.n);
      const runs = [];
      const prof = [];
      let a = -1;
      for (let k = 0; k <= h.n; k++) {
        const fresh = k < h.n && m[k] && !h.said[k];
        if (fresh && a < 0) a = k;
        if (!fresh && a >= 0) {
          runs.push(a === k - 1 ? `${a}` : `${a}-${k - 1}`);
          h.said.fill(1, a, k);
          a = -1;
        }
        if (k < h.n && m[k] !== h.sent[k]) {
          const t = h.first.get(k);
          if (t !== undefined) h.first.delete(k);
          prof.push(`${k}:${t ?? ''}:${(m[k] - h.sent[k]) >>> 0}`);
          h.sent[k] = m[k];
        }
      }
      if (runs.length) console.warn(`[gex-cov] ${key} ${runs.join(',')}`);
      if (prof.length) console.warn(`[gex-prof] ${key} ${prof.join(',')}`);
    }
  }

  /** This Worker's stack for stateful providers (one per guest thread, as
   *  natively), made at the first one's instantiation. */
  #provStack = null;
  #providerStack() {
    if (!this.#provStack) {
      const size = 1 << 20;
      // Top down unless an ejected project's providers are set (cgen.rs).
      const e = this.#shim.exports;
      const base = (e.gex_provider_stack ? e.gex_provider_stack(size) : e.gex_provider_alloc(size)) >>> 0;
      if (!base) {
        // Where the arena went, as a shim trap says it (guest maps, heap, stacks).
        const a = e.gex_arena_spent_mib
          ? ` (arena taken: maps ${e.gex_arena_spent_mib(0)}, heap ${e.gex_arena_spent_mib(1)}, ` +
            `stacks ${e.gex_arena_spent_mib(2)} MiB, memory ${Math.round(this.#memory.buffer.byteLength / 1048576)} MiB)`
          : '';
        throw new Error(`gex-pe: no memory for a provider stack${a}`);
      }
      this.#provStack = new WebAssembly.Global({ value: 'i32', mutable: true }, base + size - 16);
    }
    return this.#provStack;
  }

  /** This Worker's stack for a wasm64 provider (i64 `__stack_pointer`), and its lowest address. */
  #provStack64 = null;
  #provStackLo = 0n;
  #providerStack64() {
    if (!this.#provStack64) {
      const size = 1 << 20;
      const e = this.#shim.exports;
      const base = e.gex_provider_stack(size) >>> 0;
      if (!base) throw new Error('gex: no memory for a provider stack');
      this.#provStackLo = BigInt(base);
      this.#provStack64 = new WebAssembly.Global({ value: 'i64', mutable: true }, BigInt(base + size - 16));
    }
    return this.#provStack64;
  }

  /** A native `__TRY`'s longjmp that reached the outermost run loop, its
   *  frame gone (cgen.rs `gex_handler_lost`): true when the shim raised the
   *  exception again with no handler, a fault the loop delivers. */
  #handlerLost(err, ctx) {
    const e = this.#shim.exports;
    if (!e.gex_handler_lost || typeof WebAssembly.Exception !== 'function' || !(err instanceof WebAssembly.Exception)) return false;
    for (const inst of [this.#guest, ...(this.#cgen?.instances ?? [])]) {
      const tag = inst?.exports?.__c_longjmp;
      if (!(tag instanceof WebAssembly.Tag) || !err.is(tag)) continue;
      const p = Number(err.getArg(tag, 0)) >>> 0;
      const jmp = new DataView(this.#memory.buffer).getUint32(p, true);
      return e.gex_handler_lost(ctx, jmp) === 1;
    }
    return false;
  }

  /** What reached the run loop, for a crash report: an Error's message, a
   *  wasm exception by the export name of its tag in this Worker's modules
   *  (a C++ throw or a longjmp whose catching frame is not on this stack),
   *  anything else as itself. Asked only when reporting. */
  #thrownName(err) {
    if (err instanceof Error) return err.message;
    if (typeof WebAssembly.Exception === 'function' && err instanceof WebAssembly.Exception) {
      const mods = [['shim', this.#shim], ['guest', this.#guest], ...(this.#cgen?.instances ?? []).map((m, i) => [`unit ${i}`, m])];
      for (const [who, inst] of mods) {
        for (const [name, v] of Object.entries(inst?.exports ?? {})) {
          if (v instanceof WebAssembly.Tag && err.is(v)) {
            // A longjmp's payload (libgex/winecrt0 setjmp: `&jb->arg`, the
            // jmp_buf and the value), for finding whose frame it wanted.
            let arg = '';
            if (name === '__c_longjmp' && globalThis.GEX_SEH_REPORT) {
              try {
                const p = Number(err.getArg(v, 0)) >>> 0;
                const dv = new DataView(this.#memory.buffer);
                arg = ` (jmp_buf 0x${dv.getUint32(p, true).toString(16)}, value ${dv.getInt32(p + 4, true)})`;
              } catch {}
            }
            return `a wasm exception, tag ${name} of the ${who} module${arg}`;
          }
        }
      }
      return 'a wasm exception whose tag no module of this Worker exports';
    }
    return `a thrown ${typeof err}: ${String(err)}`;
  }

  /** Traps seen abandoning provider guest calls (reported up to 16). */
  #provLeaks = 0;

  /** What a replaced Linux library's wasm64 provider imports from `gex` (the native
   *  `libprov::register64`). Pointers and table indices
   *  are i64 (BigInt). */
  #providerImports64() {
    const e = this.#shim.exports;
    return {
      // The shim's export itself: a wasm-to-wasm call on every guest call.
      provider_guest_call: e.gex_libprov_guest_call,
      provider_missing: (addr) => e.gex_libprov_missing(addr),
      provider_thread_entry: () => e.gex_libprov_thread_entry(),
      provider_code_page: () => e.gex_libprov_thread_entry(),
      provider_stack: () => this.#provStackLo,
      provider_abi: (which) => e.gex_libprov_abi(which),
      provider_table_size: (base) => e.gex_libprov_table_size(base),
    };
  }

  /** What a stateful provider imports from `gex` (the native `libprov`). */
  #providerImports() {
    const e = this.#shim.exports;
    const ftab = this.#cgen.ftab;
    return {
      // A protected call: `fn(arg)` from the function table, 0, or the code
      // a provider_throw inside it threw (every frame between discarded, as
      // a C longjmp).
      provider_try: (f, arg) => {
        try {
          ftab.get(this.#wide ? BigInt(f) : f >>> 0)(arg);
          e.gex_provider_note(0);
          return 0;
        } catch (err) {
          if (err instanceof ProviderThrow) {
            e.gex_provider_note(1);
            return err.code;
          }
          throw err;
        }
      },
      provider_throw: (code) => {
        throw new ProviderThrow(code);
      },
      provider_ctx: () => e.gex_provider_ctx(),
      ftab_twin: (addr) => e.gex_ftab_twin(addr),
      provider_time: (which) => (which === 0 ? Date.now() / 1000 : performance.now() / 1000),
      provider_missing: (addr) => e.gex_provider_missing(addr),
      provider_alloc: (size) => e.gex_provider_alloc(size),
      provider_alloc_top: (size) => e.gex_provider_alloc_top(size),
      provider_log: (ptr, len) => {
        console.error(new TextDecoder().decode(this.#bytes().slice(ptr >>> 0, (ptr >>> 0) + (len >>> 0))));
      },
    };
  }

  /** A module file from the unit cache (an export provider's), compiled
   *  once per process and offered to the siblings. Synchronous. */
  /** The last `#cgenFileModule`'s compile, ms; -1 when the Worker had it. */
  #fileModuleMs = -1;
  #cgenFileModule(file, bytes = null, key = null) {
    const name = key ?? `cgenfile:${file}`;
    let mod = this.#sharedModules.get(name);
    this.#fileModuleMs = -1;
    if (mod) return mod;
    bytes ??= this.#env.loadFile?.(file) ?? null;
    if (!bytes) throw new Error(`gex-cgen: ${file} was published but cannot be loaded here`);
    const t0 = performance.now();
    mod = new WebAssembly.Module(bytes);
    this.#fileModuleMs = performance.now() - t0;
    this.#sharedModules.set(name, mod);
    this.#offerModule(name, mod, bytes.byteLength);
    return mod;
  }

  /** Bind unit `u`'s export providers, after the unit, whether or
   *  not it had one: each provider's module from the unit cache, published
   *  by the shim at its exports' addresses. */
  #cgenProviders(u) {
    const e = this.#shim.exports;
    const n = e.gex_cgen_provider_count?.(u) >>> 0;
    for (let i = 0; i < n; i++) {
      // A replaced Linux library (cgen/libprov.rs): its wasm64 provider, installed as a module.
      if (e.gex_cgen_provider_libprov?.(u, i)) {
        const number = e.gex_libprov_publish(u, i);
        this.#cgenCatchUp();
        if (number >= 0) {
          e.gex_cgen_pmod_installed(number);
          // its GexA64Icall entry thunks, bound as guest addresses
          e.gex_libprov_installed?.(number);
          this.#wasmdllCatchUp();
        }
        continue;
      }
      if (e.gex_cgen_provider_wasmdll?.(u, i)) {
        const number = e.gex_wasmdll_publish(u, i);
        this.#cgenCatchUp();
        if (number >= 0) {
          e.gex_cgen_pmod_installed(number);
          e.gex_wasmdll_installed();
        }
        this.#wasmdllCatchUp();
        continue;
      }
      const file = this.#cgenStr((uu, b, c) => e.gex_cgen_provider_file(uu, i, b, c), u);
      const bytes = this.#env.loadFile?.(file) ?? null;
      if (!bytes) {
        console.error(`gex-pe: missing unit: no provider module ${file} in the unit cache, so ${this.#cgenStr(e.gex_cgen_path, u)} runs without its providers; build them with GEX_CGEN_CACHE=<the unit cache> gexc cgen-image --providers (or natively: a run of the program)`);
        continue;
      }
      const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      this.#cgenFileModule(file, b);
      const gp = e.gex_alloc(Math.max(8, b.byteLength)) >>> 0;
      this.#bytes().set(b, gp);
      const number = e.gex_cgen_provider_publish(u, i, gp, b.byteLength);
      e.gex_free?.(gp, Math.max(8, b.byteLength));
      if (number < 0) continue;
      this.#cgenCatchUp();
      e.gex_cgen_pmod_installed(number);
    }
  }

  /** An ejected project's region entries for unit `u`:
   *  `<cache>/regions/<key>/`, published after the unit as a unit of
   *  their own, when the build made them. */
  #cgenRegions(u, path) {
    const e = this.#shim.exports;
    if (!e.gex_cgen_region) return;
    const r = e.gex_cgen_region(u);
    if (r < 0) return;
    const key = this.#cgenStr(e.gex_cgen_key, r);
    const got = this.#env.loadUnit?.(key) ?? null;
    if (!got) return;
    // Entries with no module this page can load (a cache the site was built
    // from holding region entries its files lack): left out, the unit's
    // code reaching those addresses through tier-1 as without them.
    if (!this.#sharedModules.has(`cgen:${key}`) && !got.wasm) {
      console.error(`gex-cgen: ${path}: region entries ${key} have no module here; left out`);
      return;
    }
    this.#cgenModule(key, got);
    const ent = got.entries instanceof Uint8Array ? got.entries : new Uint8Array(got.entries);
    const gp = e.gex_alloc(Math.max(8, ent.byteLength)) >>> 0;
    this.#bytes().set(ent, gp);
    const res = e.gex_cgen_publish(r, gp, ent.byteLength / 8);
    e.gex_free?.(gp, Math.max(8, ent.byteLength));
    if (res < 0) throw new Error(`gex-cgen: publishing ${path}'s region entries failed`);
    this.#cgenCatchUp();
    e.gex_cgen_installed(r);
    let n = 0;
    for (let i = 0; i + 8 <= ent.byteLength; i += 8) if (ent[i] | ent[i + 1] | ent[i + 2] | ent[i + 3] | ent[i + 4] | ent[i + 5] | ent[i + 6] | ent[i + 7]) n++;
    console.error(`gex-pe: region entries for ${path}: ${n}`);
  }

  /** Function JIT module `n` (fjit.rs), compiled once per process from the
   *  shim's bytes and offered to the siblings. Synchronous (a Worker). */
  #fjitModule(n) {
    const e = this.#shim.exports;
    const ptr = e.gex_fjit_ptr(n) >>> 0;
    const bytes = this.#bytes().slice(ptr, ptr + (e.gex_fjit_len(n) >>> 0));
    // By its bytes, not its number: the map outlives the process in a
    // pooled Worker, where another process's module `n` is other code.
    const name = `fjit:${moduleContentKey(bytes)}`;
    let mod = this.#sharedModules.get(name);
    if (mod) return mod;
    const dump = globalThis.process?.env?.GEX_FJIT_DUMP;
    if (dump) globalThis.process.getBuiltinModule('fs').writeFileSync(`${dump}/fjit-${n}.wasm`, bytes);
    const t0 = performance.now();
    mod = new WebAssembly.Module(bytes);
    const dt = performance.now() - t0;
    this.#compileMs += dt;
    this.#fjitCompileMs += dt;
    // Every 60 s while the JIT makes modules: what it has cost so far (the
    // census at exit comes too late for a game that never exits).
    if (t0 >= this.#fjitNextReport && e.gex_fjit_translate_ms) {
      if (this.#fjitNextReport) {
        console.error(`[gex] fjit so far: ${e.gex_fjit_modules()} modules, ${e.gex_fjit_wasm_kib()} KiB of wasm, `
          + `${e.gex_fjit_translate_ms()} ms translating (the process), ${this.#fjitCompileMs.toFixed(0)} ms compiling (this Worker)`);
      }
      this.#fjitNextReport = t0 + 60000;
    }
    this.#sharedModules.set(name, mod);
    this.#offerModule(name, mod, bytes.byteLength);
    return mod;
  }

  /** Replay the function table's log this Worker has not seen: installs,
   *  and retirements (back to `resolve`). */
  /** Table slot `slot`'s function, for an import of a packed module's
   *  export (`#wasmprogBind`): read once per catch-up (`#slotMemo`). A Plasma
   *  fork child's installs read 52,922 such slots, 13,221 of them distinct. */
  #slotFn(st, slot) {
    const memo = this.#slotMemo;
    let f = memo?.get(slot);
    if (f === undefined) {
      f = st.ftab.get(tIdx(this.#wide, slot));
      memo?.set(slot, f);
    }
    return f;
  }
  /** The slots `#slotFn` read during the outermost catch-up running, and the
   *  end of the highest install it replayed: an install below that end
   *  (slots used again) or a slot's retirement drops what was read. */
  #slotMemo = null;
  #slotMemoEnd = 0;

  /** Native modules a spare's catch-up left for the sibling that compiled
   *  them to relay (`#relayAwaited`): key -> when it was first left. */
  #relayWait = new Map();
  /** Whether a spare's catch-up (`#cgenCatchUp(true)`) leaves provider `prov`
   *  for later: a native wasm32 module this Worker has no compiled module for,
   *  which the Worker that published it compiled and is relaying (its record
   *  follows its install) -- for at most RELAY_WAIT_MS, then compiled here. A
   *  spare runs no thread, so nothing calls into what it has not installed,
   *  and it turns its event loop while it waits, which the relay needs; a
   *  running thread's Worker catches up at once, as before. */
  #relayAwaited(prov) {
    const e = this.#shim.exports;
    if (!e.gex_cgen_pmod_wasm_key || !((e.gex_cgen_pmod_wasm_len?.(prov) >>> 0) > 0)) return false;
    if (!e.gex_cgen_pmod_wasmprog?.(prov) && e.gex_cgen_pmod_libprov?.(prov)) return false;
    const lane = (n) => BigInt.asUintN(64, BigInt(e.gex_cgen_pmod_wasm_key(prov, n)));
    const key = `wasm:${e.gex_cgen_pmod_wasm_len(prov) >>> 0}:${lane(0)}:${lane(1)}`;
    if (this.#sharedModules.has(key) || this.#sharedModules.has(`${key}:p`)) {
      this.#relayWait.delete(key);
      return false;
    }
    const now = performance.now(), since = this.#relayWait.get(key);
    if (since === undefined) {
      this.#relayWait.set(key, now);
      return true;
    }
    if (now - since < RELAY_WAIT_MS) return true;
    this.#relayWait.delete(key);
    return false;
  }

  #cgenCatchUp(spare = false) {
    const e = this.#shim.exports;
    if (!e.gex_ftab_log_len) return;
    if (!spare && this.#relayWait.size) this.#relayWait.clear();
    // before the log's length: every slot below it is published by then
    const size = e.gex_ftab_size() >>> 0;
    const len = e.gex_ftab_log_len() >>> 0;
    if (!len && !this.#cgen) return;
    const st = this.#cgenStore();
    if (st.have >= len) {
      // The shim's count is per thread and starts again at 0 with a thread's
      // fresh TLS (an execve from another thread): this Worker's table still
      // holds what it replayed.
      if (e.gex_ftab_behind()) e.gex_cgen_have(st.have);
      // (and every slot below the map's size is in it: a module this Worker
      // installed itself has no event left for it here)
      this.#raiseIcallLimit(size);
      this.#wasmdllCatchUp();
      return;
    }
    this.#cgenGrow();
    const timed = logNamed('unitstall');
    const t0 = timed ? performance.now() : 0, from = st.have;
    const outer = !this.#slotMemo;
    if (outer) {
      this.#slotMemo = new Map();
      this.#slotMemoEnd = 0;
    }
    try {
    while (st.have < len) {
      const ev = BigInt.asUintN(64, BigInt(e.gex_ftab_event(st.have)));
      const kind = Number(ev >> 32n), v = Number(ev & 0xffffffffn);
      // A provider of a program an execve replaced (`FuncMap::kill_module`) is skipped.
      if (kind === 1) {
        const u = e.gex_ftab_install_unit(v) >>> 0;
        // A provider another Worker is publishing: its record follows its
        // install, and the next catch-up takes it from here.
        if (u >= 0x80000000 && u !== 0xffffffff && e.gex_cgen_pmod_ready && !e.gex_cgen_pmod_ready(u - 0x80000000)) break;
        const ib = e.gex_ftab_install_base(v) >>> 0, iend = ib + (e.gex_ftab_install_len ? e.gex_ftab_install_len(v) >>> 0 : 0);
        if (ib < this.#slotMemoEnd) this.#slotMemo.clear();
        this.#slotMemoEnd = Math.max(this.#slotMemoEnd, iend);
        // Plain units deferred (`#cgenDefer`); providers and JIT modules now.
        if (u !== 0xffffffff) {
          if (u < 0x40000000 && !(SP_CODE && single.on) && e.gex_ftab_install_len) {
            this.#cgenDefer(u, e.gex_ftab_install_base(v) >>> 0, e.gex_ftab_install_len(v) >>> 0);
          }
          else if (u >= 0x80000000) {
            if (spare && this.#relayAwaited(u - 0x80000000)) break;
            try {
              this.#cgenInstall(u, e.gex_ftab_install_base(v) >>> 0);
            } catch (err) {
              // A provider an execve is replacing: its install is killed
              // before its record and bytes go (cgen.rs `on_exec`), so one
              // whose bytes are gone is dead by now, and skipped.
              if ((e.gex_ftab_install_unit(v) >>> 0) !== 0xffffffff) throw err;
            }
          }
          else this.#cgenInstall(u, e.gex_ftab_install_base(v) >>> 0);
        }
        // installs come in index order: every slot below this one's end is
        // in the table (or a deferred unit's, which native code never calls)
        if (e.gex_ftab_install_len) this.#raiseIcallLimit((e.gex_ftab_install_base(v) >>> 0) + (e.gex_ftab_install_len(v) >>> 0));
      }
      else if (kind === 2) {
        st.ftab.set(tIdx(this.#wide, v), st.resolve);
        this.#slotMemo.delete(v);
        if (this.#unitsDeferred.length) this.#unitRetired(v);
      }
      st.have++;
      e.gex_cgen_have(st.have);
    }
    } finally {
      if (outer) this.#slotMemo = null;
    }
    this.#dlSize.value = size;
    if (st.have >= len) this.#raiseIcallLimit(size);
    if (timed && st.have - from > 16) console.warn(`[unitcatch] ${this.#whoFills()}${this.#forkChild ? ' (fork child)' : ''}: ${st.have - from} events in ${(performance.now() - t0).toFixed(1)} ms`);
    this.#wasmdllCatchUp();
  }

  /** The bridges of exports bound into Wine's PE module after its install
   *  (a later skeleton, the entry thunks: cgen/wasmdll.rs), each once. */
  #wasmdllHave = 0;
  #wasmdllCatchUp() {
    const e = this.#shim.exports;
    const len = (e.gex_wasmdll_binds_len?.() ?? 0) >>> 0;
    if (this.#wasmdllHave >= len) return;
    const blocks = this.#guest.exports.blocks;
    while (this.#wasmdllHave < len) {
      const i = this.#wasmdllHave;
      const tables = this.#growTables(e.gex_wasmdll_bind_slots(i) >>> 0, 'cgen');
      if (blocks) this.#growBlocks();
      const n = e.gex_wasmdll_bind_regs(i, 0, 0) >>> 0;
      const buf = e.gex_alloc(4 * Math.max(1, n)) >>> 0;
      let regs;
      try {
        e.gex_wasmdll_bind_regs(i, buf, n);
        regs = new Uint32Array(this.#memory.buffer.slice(buf, buf + 4 * n));
      } finally {
        e.gex_free?.(buf, 4 * Math.max(1, n));
      }
      const st = this.#cgenStore();
      this.#writeBridges(tables, blocks, regs, st.bridge);
      if (this.#bridgeCount) this.#bridgeCount.bridged += regs.length >>> 1;
      e.gex_wasmdll_bind_done(i);
      this.#wasmdllHave++;
    }
  }

  /** The process's wasm32 link map, for a fork's child (cgen/dylink.rs), or
   *  null: its bytes, as the child's shim takes them (with every module's
   *  binding list, megabytes for Plasma's shell: not made a string). */
  #wasmprogLink() {
    const e = this.#shim.exports;
    const n = (e.gex_wasmprog_link?.(0, 0) ?? 0) >>> 0;
    if (!n) return null;
    const buf = e.gex_alloc(n) >>> 0;
    try {
      e.gex_wasmprog_link(buf, n);
      return this.#bytes().slice(buf, buf + n);
    } finally {
      e.gex_free?.(buf, n);
    }
  }

  /** This Worker's `__stack_pointer` for wasm32 programs: one global the
   *  program and its shared libraries share (one C stack per thread). */
  #wasmprogSp = null;
  /** 64-bit Wine's direct entry (`entry_call`): the trampoline's hook table, its
   *  slot 0 the entry function or the trampoline's `slow` export. */
  #entryHook = null;
  #entrySlowFn = null;
  #entryTramp() {
    entryTrampMod ??= new WebAssembly.Module(ENTRY_TRAMP);
    const hook = new WebAssembly.Table({ element: 'anyfunc', initial: 1 });
    const t = new WebAssembly.Instance(entryTrampMod, { e: { hook, slow: (idx, ctx, sp) => this.#entrySlow(idx, ctx, sp) } });
    this.#entryHook = hook;
    this.#entrySlowFn = t.exports.slow;
    this.#entryCallFn = t.exports.entry_call;
    return t.exports.entry_call;
  }
  #entryCallFn = null;
  /** The first call since the function table or Wine's stack pointer
   *  changed: the entry function over them, in the hook, then called. */
  #entrySlow(idx, ctx, sp) {
    const spG = this.#wasmprogSp ?? this.#wasmprogStack();
    const prog = this.#snapProg?.()?.exports;
    const f = prog?.__gex_snap_resume64
      ? new WebAssembly.Instance(entryFnSnapModule(this.#wide, typeof spG.value === 'bigint'), { e: {
        t: this.#cgen.ftab, sp: spG, d: (this.#snapDepth ??= new WebAssembly.Global({ value: 'i32', mutable: true }, 0)), st: prog.asyncify_get_state,
        af: (idx, ctx, rsp) => this.#snapStop(0, idx, ctx, rsp),
      } }).exports.entry
      : new WebAssembly.Instance(entryFnModule(this.#wide, typeof spG.value === 'bigint'), { e: { t: this.#cgen.ftab, sp: spG } }).exports.entry;
    this.#entryHook.set(0, f);
    // (a JS frame here, which a blocking call could not suspend through)
    if (SP_CODE && single.on) return 0;
    return f(idx, ctx, sp);
  }
  /** The entry function is made again at its next call. */
  #entryReset() {
    if (this.#entryHook) this.#entryHook.set(0, this.#entrySlowFn);
  }
  /** Wine's stack pointer as a loop that catches traps found it, and put back
   *  there: the entry function's own restore does not run for a trap. */
  #progSpAt() {
    // 0: the program had not set it yet (its start sets it, inside the
    // loop's entry). Put back, that would undo the program's own setting,
    // and a fault hook's frames would then go below address 0.
    const v = this.#wasmprogSp?.value;
    return v ? v : undefined;
  }
  #progSpBack(v) {
    const g = this.#wasmprogSp;
    if (g && v !== undefined && typeof g.value === typeof v) g.value = v;
  }

  #wasmprogStack() {
    // i64 in a 64-bit memory's process and for a lowered (LP64) program
    const wide = this.#wide || this.#wasmprogWide;
    if (this.#wasmprogSp && (typeof this.#wasmprogSp.value === 'bigint') !== wide) this.#wasmprogSp = null;
    if (!this.#wasmprogSp) this.#entryReset();
    return (this.#wasmprogSp ??= new WebAssembly.Global(wide ? { value: 'i64', mutable: true } : { value: 'i32', mutable: true }, wide ? 0n : 0));
  }
  /** This Worker's program has an i64 stack pointer (`#cgenInstall`). */
  #wasmprogWide = false;

  /** This Worker's asyncify state and data globals, which a program's
   *  modules built with relocatable asyncify share. */
  #wasmprogAsync = null;
  /** The mutable globals the program's modules were bound to (`G`), which
   *  a vfork child's run may change (`#vforkChild`). */
  #wasmprogMuts = [];

  /** Apply provider module `prov`'s binding list (cgen/dylink.rs's format)
   *  to `imports`; the table slots to fill with exports after
   *  instantiation, as [slot, name]. */
  #wasmprogBind(prov, imports, st, bind) {
    this.#wasmprogTls = null;
    if (!bind) return [];
    const { plan, str, vals } = bind;
    const o = plan.ops;
    const name = (at) => str.slice(o[at], o[at] + o[at + 1]);
    const set = (m, n, v) => { (imports[m] ??= {})[n] = v; };
    const extra = [];
    let late = null;
    for (let i = 0; i < o.length; i += PLAN_REC) {
      switch (o[i]) {
        case BIND_ASYNC:
          this.#wasmprogAsync ??= [0, 1].map(() => new WebAssembly.Global({ value: 'i32', mutable: true }, 0));
          set('env', name(i + 3), this.#wasmprogAsync[o[i + 5]]);
          break;
        case BIND_GLOBAL: {
          const mutable = o[i + 6] === 1, v = vals[o[i + 7]];
          const g = o[i + 5]
            ? new WebAssembly.Global({ value: 'i64', mutable }, BigInt(v))
            : new WebAssembly.Global({ value: 'i32', mutable }, Number(v) | 0);
          if (mutable) this.#wasmprogMuts.push(g);
          set(name(i + 1), name(i + 3), g);
          break;
        }
        case BIND_EXPORT: {
          const mod = Number(vals[o[i + 5]]), exp = name(i + 6);
          const from = st.dyl.get(mod);
          // a packed module's (`packExports`): from its slot or value
          let x = from?.exports[exp];
          if (x === undefined && from && o[i + 8]) {
            x = o[i + 8] === 2
              ? new WebAssembly.Global({ value: 'i32', mutable: false }, o[i + 9])
              : this.#slotFn(st, Number(vals[o[i + 9]]));
          }
          if (x === undefined) throw new Error(`wasm32 module ${prov}: ${name(i + 1)}.${name(i + 3)}: ${exp} of module ${mod} is not instantiated in this Worker`);
          set(name(i + 1), name(i + 3), x);
          break;
        }
        case BIND_SLOT: {
          const slot = tIdx(this.#wide, Number(vals[o[i + 5]])), t = st.ftab;
          set(name(i + 1), name(i + 3), (...a) => t.get(slot)(...a));
          break;
        }
        case BIND_MISSING: {
          const msg = name(i + 6);
          set(name(i + 1), name(i + 3), () => { throw new Error(msg); });
          break;
        }
        case BIND_LATE:
          (late ??= []).push([name(i + 1), name(i + 3), Number(vals[o[i + 5]]), name(i + 6)]);
          break;
        case BIND_EXTRA:
          extra.push([Number(vals[o[i + 5]]), name(i + 3)]);
          break;
        case BIND_TLS:
          this.#wasmprogTls = [o[i + 5], o[i + 6]];
          break;
      }
    }
    if (late) {
      // In wasm, for a type the list gives: the call stays a wasm frame,
      // which the single-thread build's fiber can suspend through, as it
      // cannot through a JS one.
      // Its bytes are a function of these slots and types alone, so it is
      // made once per key: in the single-thread build in this realm
      // (`LATE_MODS`), in the default build on the page (`late:`, relayed
      // as a program's module is, so a fork's child or a thread has it).
      const shared = SP_CODE && single.on, key = `${this.#wide ? 'w' : ''}${late.map((l) => `${l[2]}${l[3]}`).join()}`;
      let mod = shared ? LATE_MODS.get(key) : this.#sharedModules.get(`late:${key}`);
      if (!mod) {
        mod = lateCalls(late, this.#wide);
        if (shared && LATE_MODS.size < LATE_MODS_MAX) LATE_MODS.set(key, mod);
        else if (!shared) {
          this.#sharedModules.set(`late:${key}`, mod);
          this.#offerModule(`late:${key}`, mod, 0);
        }
      }
      const x = new WebAssembly.Instance(mod, { e: { t: st.ftab } }).exports;
      late.forEach(([m, n], k) => set(m, n, x[k]));
    }
    return extra;
  }

  /** Provider module `prov`'s binding list, parsed (`bindPlan`): the page's
   *  plan for its shape when the page has one (`plan:`), and then only its
   *  numbers are read; else the list read whole, parsed, and its plan
   *  offered to the page. `{ plan, str, vals, text }` (`str` the plan's
   *  names, `text` the list when it was read), or null for none. */
  #bindGet(prov) {
    const e = this.#shim.exports;
    const len = !(SP_CODE && single.on) && e.gex_wasmprog_bind_key ? e.gex_wasmprog_bind_key(prov, 0) >>> 0 : 0;
    // the shape's length and two independent 64-bit hashes of it
    const w = (lane) => (e.gex_wasmprog_bind_key(prov, lane) >>> 0).toString(16);
    const key = len ? `plan:${len - 1}:${w(2)}:${w(1)}:${w(4)}:${w(3)}` : null;
    const had = key ? this.#sharedModules.get(key) : null;
    if (had && isPlan(had)) return { plan: had, str: planNames(had), vals: this.#bindVals(prov), text: null };
    if (key) {
      // only the shape's lines read: the numbers come from the shim
      const { plan, str } = bindPlan(this.#wasmprogStrRead(e.gex_wasmprog_binds, prov, len - 1));
      this.#sharedModules.set(key, plan);
      this.#offerModule(key, plan, 0);
      return { plan, str, vals: this.#bindVals(prov), text: null };
    }
    const text = this.#wasmprogBindRead(prov);
    if (!text) return null;
    const [shape, vals] = bindSplit(text);
    return { ...bindPlan(shape), vals, text };
  }

  /** Provider module `prov`'s binding list's numbers (`V`), as the shim
   *  keeps them: no text to split. */
  #bindVals(prov) {
    const e = this.#shim.exports;
    const n = e.gex_wasmprog_bind_vals(prov, 0, 0) >>> 0;
    if (!n) return new Uint32Array(0);
    const buf = e.gex_alloc(n) >>> 0;
    try {
      e.gex_wasmprog_bind_vals(prov, buf, n);
      return new Uint32Array(this.#bytes().slice(buf, buf + n).buffer);
    } finally {
      e.gex_free?.(buf, n);
    }
  }

  /** Wine's PE module `prov`'s binding list (cgen/wasmdll.rs, the format of
   *  cgen/dylink.rs: `G`, `E`, `L`) applied to `imports`. A call through a
   *  slot (`L`) whose module this Worker has not instantiated yet catches
   *  up first. */
  #wasmdllBind(prov, imports, st) {
    const e = this.#shim.exports;
    const n = e.gex_wasmdll_binds ? e.gex_wasmdll_binds(prov, 0, 0) >>> 0 : 0;
    if (!n) return;
    const buf = e.gex_alloc(n) >>> 0;
    let text;
    try {
      e.gex_wasmdll_binds(prov, buf, n);
      text = new TextDecoder().decode(this.#bytes().slice(buf, buf + n));
    } finally {
      e.gex_free?.(buf, n);
    }
    const set = (m, name, v) => { (imports[m] ??= {})[name] = v; };
    const s = {}, lines = [];
    const t = st.ftab, empty = t.get(tIdx(this.#wide, 0));
    // the module's ready flags, a byte each from `base`: this Worker's, in
    // the process memory (`#lateFlagsAt`), which its thunks read
    let base = -1;
    for (const line of text.split('\n')) {
      if (!line) continue;
      const f = line.split('\t');
      switch (f[0]) {
        case 'G':
          set(f[1], f[2], f[5] === '64'
            ? new WebAssembly.Global({ value: 'i64', mutable: f[4] === '1' }, BigInt(f[3]))
            : new WebAssembly.Global({ value: 'i32', mutable: f[4] === '1' }, Number(f[3]) | 0));
          break;
        case 'E': {
          const x = st.dyl.get(Number(f[3]))?.exports[f[4]];
          if (x === undefined) throw new Error(`Wine's PE module ${prov}: ${f[2]}: module ${f[3]} is not instantiated in this Worker`);
          set(f[1], f[2], x);
          break;
        }
        case 'L': {
          // a call through a slot filled once the module it calls into is
          // instantiated here (#wasmdllLate): a wasm thunk (cgen/wasmdll.rs
          // `late_thunks`) calls the slot once its flag is set, and before
          // that this, which catches up first
          if (base < 0) base = this.#lateFlagsAt(st, text);
          const n = Number(f[3]), slot = tIdx(this.#wide, n), flag = base + Number(f[4]);
          const ready = st.lateReady ??= new Map();
          const had = ready.get(n);
          if (had === undefined) ready.set(n, flag);
          else if (typeof had === 'number') ready.set(n, [had, flag]);
          else had.push(flag);
          const fn = t.get(slot);
          if (fn && fn !== empty) this.#bytes()[flag] = 1;
          s[f[4]] = (...a) => this.#lateMiss(st, slot, flag, f[2], a);
          lines.push([f[1], f[2], f[4]]);
          break;
        }
      }
    }
    if (lines.length) {
      let mod = this.#wasmdllThunkMods.get(prov);
      if (!mod) {
        const n = e.gex_wasmdll_thunks(prov, 0, 0) >>> 0;
        const tb = e.gex_alloc(n) >>> 0;
        try {
          e.gex_wasmdll_thunks(prov, tb, n);
          mod = new WebAssembly.Module(this.#bytes().slice(tb, tb + n));
        } finally {
          e.gex_free?.(tb, n);
        }
        this.#wasmdllThunkMods.set(prov, mod);
      }
      const b = this.#wide ? new WebAssembly.Global({ value: 'i64', mutable: false }, BigInt(base)) : new WebAssembly.Global({ value: 'i32', mutable: false }, base);
      const x = new WebAssembly.Instance(mod, { t: { t: st.ftab }, s, r: { m: this.#memory, b } }).exports;
      for (const [m, name, k] of lines) set(m, name, x[k]);
    }
  }

  /** Where this Worker's ready flags for a module with binding list `text`
   *  go: a byte per `L` line, zeroed, from the shim's heap in the process
   *  memory (given back when the Worker is: `releaseWorker`). Not a memory
   *  of their own: an engine reserves the same address space for any
   *  memory however small (8 GiB a 32-bit one), a tab has room for about
   *  124, and a memory per thread ran a page out of them. */
  #lateFlagsAt(st, text) {
    let n = 0;
    for (const line of text.split('\n')) if (line.startsWith('L\t')) n++;
    const at = this.#shim.exports.gex_alloc(n) >>> 0;
    (st.lateChunks ??= []).push([at, n]);
    return at;
  }

  /** A late call before its slot's flag is set: the Worker catches up, and
   *  the slot then holds the callee (or the call fails). */
  #lateMiss(st, slot, flag, name, a) {
    const t = st.ftab, empty = t.get(tIdx(this.#wide, 0));
    let fn = t.get(slot);
    if (!fn || fn === empty) {
      this.#cgenCatchUp();
      this.#wasmdllLate(st);
      fn = t.get(slot);
      if (!fn || fn === empty) throw new Error(`${name}: its module is not installed`);
    }
    this.#bytes()[flag] = 1;
    return fn(...a);
  }

  /** Fill the slots between Wine's PE modules (cgen/wasmdll.rs `Late`)
   *  whose two sides this Worker has instantiated: those of module `prov`
   *  (just instantiated), or every one still open without it. The list is
   *  read again only when it changed (`gex_wasmdll_lates_gen`), and each
   *  entry is kept under its two modules until filled. */
  #wasmdllLate(st, prov) {
    const e = this.#shim.exports;
    if (!e.gex_wasmdll_lates) return;
    const gen = e.gex_wasmdll_lates_gen ? e.gex_wasmdll_lates_gen() >>> 0 : -1;
    const fresh = [];
    if (gen < 0 || gen !== st.lateGen) {
      st.lateGen = gen;
      const n = e.gex_wasmdll_lates(0, 0) >>> 0;
      if (n) {
        const buf = e.gex_alloc(n) >>> 0;
        let text;
        try {
          e.gex_wasmdll_lates(buf, n);
          text = new TextDecoder().decode(this.#bytes().slice(buf, buf + n));
        } finally {
          e.gex_free?.(buf, n);
        }
        const seen = st.lateSeen ??= new Set(), open = st.lateOpen ??= new Map();
        const under = (m, l) => { const a = open.get(m); if (a) a.push(l); else open.set(m, [l]); };
        for (const line of text.split('\n')) {
          if (!line) continue;
          const f = line.split('\t'), slot = Number(f[0]);
          if (seen.has(slot)) continue;
          seen.add(slot);
          const l = { slot, importer: Number(f[1]), provider: Number(f[2]), name: f[3], done: false };
          fresh.push(l);
          under(l.importer, l);
          under(l.provider, l);
        }
      }
    }
    const open = st.lateOpen;
    if (!open) return;
    const fill = (l) => {
      if (l.done) return;
      const p = st.dyl.get(l.provider);
      if (!p || !st.dyl.has(l.importer)) return;
      l.done = true;
      const i = tIdx(this.#wide, l.slot);
      const had = st.ftab.get(i);
      if (had && had !== st.ftab.get(tIdx(this.#wide, 0))) return;
      const f = p.exports[l.name];
      if (typeof f !== 'function') return;
      st.ftab.set(i, f);
      const r = st.lateReady?.get(l.slot);
      if (typeof r === 'number') this.#bytes()[r] = 1;
      else if (r) for (const k of r) this.#bytes()[k] = 1;
    };
    for (const l of fresh) fill(l);
    if (prov !== undefined) {
      for (const l of open.get(prov) ?? []) fill(l);
      // (an entry left open waits for its other module, under whose number it is too)
      open.delete(prov);
    } else {
      for (const [m, a] of open) {
        for (const l of a) fill(l);
        if (a.every((l) => l.done)) open.delete(m);
      }
    }
  }

  /** Provider module `prov`'s binding list, '' for none. */
  #wasmprogBindRead(prov) {
    return this.#wasmprogStrRead(this.#shim.exports.gex_wasmprog_binds, prov);
  }

  /** A string the shim's `fn(prov, buf, cap)` puts (its length for a
   *  `cap` of 0), or '' for none. */
  #wasmprogStrRead(fn, prov, upto = Infinity) {
    const e = this.#shim.exports;
    const n = fn ? Math.min(fn(prov, 0, 0) >>> 0, upto) : 0;
    if (!n) return '';
    const buf = e.gex_alloc(n) >>> 0;
    try {
      fn(prov, buf, n);
      return new TextDecoder().decode(this.#bytes().slice(buf, buf + n));
    } finally {
      e.gex_free?.(buf, n);
    }
  }

  /** The last binding list's `T size align` (a `dlopen`ed module's
   *  thread-local block), or null. */
  #wasmprogTls = null;
  /** A fork's child: its forking thread's blocks, by link-map index
   *  (`#dlTlsFork`), which this, its first thread, keeps. */
  #dlTlsKept = null;
  /** This Worker's chunks for those blocks: [address, length], the last
   *  one's next free byte. */
  #dlTlsChunks = [];
  /** The last `gex_wasmprog_dl_gen` this Worker caught up with. */
  #dlGen = 0;
  /** `gex.dl_size`: the function-table slots this Worker has caught up
   *  with (`#cgenCatchUp`), for modules built with GEX_DL_CHECKED_CALLS. */
  #dlSize = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
  /** `gex.icall_limit64` and `gex.icall_limit` (gex_abi::ftab::ICALL_LIMIT64):
   *  native PE code's indirect calls take an index below it straight to this
   *  Worker's table, so it is the slots that table holds, capped at
   *  NATIVE_FN_TOP (`#raiseIcallLimit`). */
  #icallLimit = [new WebAssembly.Global({ value: 'i64', mutable: true }, 0n), new WebAssembly.Global({ value: 'i32', mutable: true }, 0)];
  #raiseIcallLimit(held) {
    const v = Math.min(held, NATIVE_FN_TOP);
    if (v <= Number(this.#icallLimit[1].value >>> 0)) return;
    this.#icallLimit[0].value = BigInt(v);
    this.#icallLimit[1].value = v;
  }
  #dlTlsNext = 0;

  /** Module `prov`'s (just instantiated as `inst`) thread-local block in
   *  this Worker's thread (cgen/dylink.rs `T`): the forking thread's in a
   *  fork's child, otherwise a fresh one with the module's initial image. */
  #dlTlsInit(prov, inst, st, [size, align]) {
    const e = this.#shim.exports;
    const idx = this.#dlTlsKept ? e.gex_wasmprog_link_index(prov) : -1;
    const kept = idx >= 0 ? this.#dlTlsKept.get(idx) : undefined;
    this.#tlsOwners.push({ prov, inst, size, align, st });
    if (kept !== undefined) {
      this.#dlTlsKept.delete(idx);
      inst.exports.__gex_set_tls_base(kept);
      st.dlTls.set(prov, kept);
      return;
    }
    const at = this.#dlTlsAlloc(prov, size, align);
    inst.exports.__wasm_init_tls(at);
    st.dlTls.set(prov, at);
  }

  /** The modules of this Worker's that have a thread-local block of its
   *  thread's (`#dlTlsInit`, a Wine PE module's): `{ prov, inst, size,
   *  align, st }`, `st` for a `dlopen`ed one's. */
  #tlsOwners = [];

  /** This Worker's thread has ended and its process runs on as the same
   *  program (`worker.mjs`): whether it waits for one of the process's next
   *  threads, as a spare does (`adoptThread` with `parked`). The shim lets
   *  go of the thread (`gex_thread_park`), and every module's thread-local
   *  block is made again, as a new Worker's are when it installs them: the
   *  next thread finds its thread-locals as their module starts them, none
   *  of the last one's. False: the Worker ends as before. */
  parkThread() {
    const e = this.#shim?.exports;
    if (!e?.gex_thread_park || this.#released || this.#noSlot || this.#vfork.length || this.#dlTlsKept?.size) return false;
    const why = e.gex_thread_park();
    if (why !== 1) {
      if (logNamed('spare')) console.warn(`[spare] tid ${this.#threadTid}: its Worker ends, not parked (${why})`);
      return false;
    }
    this.freeThreadTls();
    this.#dlTlsNext = 0;
    for (const o of this.#tlsOwners) {
      const at = this.#dlTlsAlloc(o.prov, o.size, o.align);
      o.inst.exports.__wasm_init_tls(typeof o.inst.exports.__tls_size?.value === 'bigint' ? BigInt(at) : at);
      o.st?.dlTls.set(o.prov, at);
    }
    this.#threadTid = 0;
    this.adoptGen = undefined;
    return true;
  }

  /** `size` bytes at `align` for a module's thread-local block in this
   *  Worker's chunks. */
  #dlTlsAlloc(prov, size, align) {
    const e = this.#shim.exports;
    align = Math.max(align, 8);
    const last = this.#dlTlsChunks.at(-1);
    let at = (this.#dlTlsNext + align - 1) & -align;
    if (!last || at + size > last[0] + last[1]) {
      const len = Math.max(0x4000, size + align);
      const chunk = e.gex_wasmprog_tls_chunk(len) >>> 0;
      if (!chunk) throw new Error(`wasm module ${prov}: no memory for its thread-local block`);
      this.#dlTlsChunks.push([chunk, len]);
      at = (chunk + align - 1) & -align;
    }
    this.#dlTlsNext = at + size;
    return at;
  }

  /** This thread's blocks for a fork's child: [link-map index, address]. */
  #dlTlsFork() {
    const e = this.#shim.exports, m = this.#cgen?.dlTls;
    if (!m?.size || !e.gex_wasmprog_link_index) return [];
    return [...m].map(([prov, at]) => [e.gex_wasmprog_link_index(prov), at]).filter(([i]) => i >= 0);
  }

  /** This Worker's thread ended, its process goes on: its blocks go. */
  freeThreadTls() {
    const e = this.#shim?.exports;
    for (const [at, len] of this.#dlTlsChunks) e?.gex_wasmprog_tls_free?.(at, len);
    this.#dlTlsChunks = [];
  }

  /** A session snapshot's imports, for a program built with the snapshot
   *  flavour (libgex-snap/). Each thread's driver registers its record
   *  (`snap_register`). The program's system calls are the default build's
   *  (`syscall`); a recording module's (it exports `__gex_snap_blocks`) are
   *  wrapped: while the kernel's snapshot word asks threads to stop, a call
   *  that can block unwinds the thread's frames to its driver instead of
   *  being made, as it does when the shim left the call's wait for it
   *  (`gex_snap_stopped`). The driver parks in `snap_park` until the word is
   *  cleared, and rewinds: the rewound `syscall` makes the call. The record
   *  holds the program's stack pointer at the call (+16), which the unwind
   *  does not keep: the rewind puts it back. */
  #snapImports(asyncify, call, inst, mod = null) {
    this.#snapProg = inst;
    const exp = mod ? WebAssembly.Module.exports(mod).map((x) => x.name) : [];
    // A wasm64 program of the flavour: its hooks and call-outs stop too.
    const flavour = !(SP_CODE && single.on) && exp.includes('__gex_snap_resume64');
    this.#snapHooks = flavour;
    if (flavour) this.#snapDepth ??= new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
    const recording = exp.includes('__gex_snap_blocks');
    const e = this.#shim.exports;
    const k = () => this.#kernel;
    const dv = () => new DataView(this.#memory.buffer);
    const unwind = () => {
      const rec = this.#snapRec;
      const sp = this.#wasmprogSp;
      if (sp) dv().setUint32(rec + 16, Number(sp.value) >>> 0, true);
      this.#snapUnwinding = true;
      asyncify(1, rec);
    };
    // (the context's words are 64-bit; the call's number is a long, an i64
    // in a wasm64 program)
    const w64 = exp.includes('__gex_snap_resume64');
    const blocks = (ctx) => {
      const n = dv().getBigUint64(Number(ctx) >>> 0, true);
      return !!inst().exports.__gex_snap_blocks(w64 ? n : Number(BigInt.asIntN(32, n)));
    };
    const stops = (ctx) => {
      const P = inst().exports;
      if (!blocks(ctx)) return false;
      if (!this.#snapRec && P.__gex_snap_ready) {
        // The record's making makes calls of its own through the thread's
        // context, which holds this call's number and arguments (words 0
        // to 32): kept, and put back for the call the unwind leaves undone.
        const at = Number(ctx) >>> 0, keep = new Uint8Array(this.#memory.buffer, at, 33 * 8).slice();
        P.__gex_snap_ready();
        new Uint8Array(this.#memory.buffer, at, 33 * 8).set(keep);
      }
      return !!this.#snapRec && this.#snapOk();
    };
    return {
      ...(recording ? { syscall: (ctx, pc) => {
        const st = asyncify(0, 0);
        if (st === 2) {
          asyncify(4, 0);
          const sp = this.#wasmprogSp;
          if (sp) {
            const v = dv().getUint32(this.#snapRec + 16, true);
            sp.value = typeof sp.value === 'bigint' ? BigInt(v) : v | 0;
          }
        } else if (st !== 0) {
          throw new Error(`gex.syscall in asyncify state ${st}`);
        } else if (!this.#vforkDepth && k()?.k_snap_pending?.() && stops(ctx)) {
          unwind();
          return;
        }
        // (only a call that can block is one whose wait the shim leaves
        // for the snapshot, undone: its number is read before the call)
        const b = this.#snapRec && blocks(ctx);
        call(ctx, pc);
        if (b && e.gex_snap_stopped?.()) unwind();
      } } : {}),
      // No record for a posix_spawn child (`#vforkChild`): its parent's
      // memory map is not its to change.
      snap_wanted: () => (this.#vforkDepth ? 0 : 1),
      // (a wasm64 program's pointers are i64s)
      snap_register: (rec, root) => {
        this.#snapRec = Number(rec) >>> 0;
        this.#snapRoot = this.#snapRec ? root : 0;
        this.#snapSplit();
        e.gex_snap_enable?.(this.#snapRec && !this.#vforkDepth ? 1 : 0);
      },
      // A restored thread's call into the program's handler, which its
      // snapshot stopped inside the host's entry (entry64.c): rewound under
      // this Worker's entry function, then finished as `w64_entry` finishes
      // one (`gex_snap_entry_tail`).
      snap_entry_rewind: (rec) => {
        this.#snapLevelEnter(Number(rec) >>> 0, 0, Number(e.gex_snap_ctx()));
      },
      ...(flavour ? { call_out: (ctx, target, ret, spWant) => this.#snapCallOut(ctx >>> 0, target, ret, spWant) } : {}),
      snap_unwinding: () => {
        const was = this.#snapUnwinding;
        this.#snapUnwinding = false;
        return was ? this.#snapRec : 0;
      },
      snap_park: (rec) => this.#snapPark(rec),
    };
  }
  /** Park this Worker's thread for the session snapshot, its record `rec`
   *  filled in, until the kernel's snapshot word is cleared. */
  #snapPark(rec) {
    const e = this.#shim.exports;
    const k = () => this.#kernel;
    const dv = () => new DataView(this.#memory.buffer);
    const kern = this.#kernelSpec?.memory;
    if (!kern || !k()) throw new Error('gex.snap_park: no global kernel');
    const words = new Int32Array(kern.buffer);
    const at = (k().k_snap_word() >>> 0) >>> 2;
    const r = Number(rec) >>> 0, d = dv();
    // The first thread to stop lists where the kernel's file bytes lie
    // (`k_snap_extents`), for the host's references.
    if (k().k_snap_extents_at) {
      const h = (k().k_snap_extents_at() >>> 0) >>> 2;
      if (Atomics.compareExchange(words, h, 0, 1) === 0) k().k_snap_extents();
    }
    this.#env.snapParked?.({
      pid: e.gex_my_pid?.() ?? 0,
      tid: e.gex_snap_tid?.() ?? this.#threadTid,
      rec: r,
      entry: d.getUint32(r + 8, true),
      start: d.getUint32(r + 12, true),
      maxPages: memCapPages() ?? PAGES,
      moduleStore: this.#moduleStore ?? undefined,
    });
    while (Atomics.load(words, at) !== 0) Atomics.wait(words, at, Atomics.load(words, at), 1000);
  }
  /** A snapshot's stop where the program's handler came back unwinding,
   *  called by the snapshot entry function (`entryFnSnapModule`, `route`
   *  0) or the hook's call (`#snapHookCall`, `route` 1): `idx` the
   *  function, `rsp` the guest's stack pointer at the entry, and a hook's
   *  arguments. The unwind ends here; the level is kept in the record
   *  (`SNAP_LV`). The outermost level keeps the rest (the registers, a
   *  restored thread's start), parks, and begins the rewind, which its
   *  caller's call makes. A level inside guest code that an outer level's
   *  call-out runs leaves by `SNAP_NEST` to that call-out
   *  (`#snapCallOut`), which unwinds the outer level's frames in turn. */
  #snapStop(route, idx, ctx, rsp, code = 0, addr = 0n, kind = 0) {
    const P = this.#snapProg().exports;
    const rec = this.#snapRec;
    if (!this.#snapUnwinding || !rec) throw new Error('snapshot: an unwind not the snapshot\'s reached the entry');
    this.#snapUnwinding = false;
    P.asyncify_stop_unwind();
    const k = this.#snapDepth.value - 1;
    const v = new DataView(this.#memory.buffer), b = rec + SNAP_LV + k * SNAP_LVS;
    if (v.getUint32(rec + SNAP_NLV, true) === 0) v.setUint32(rec + SNAP_NLV, k + 1, true);
    v.setUint32(b, route, true);
    v.setUint32(b + 4, idx >>> 0, true);
    v.setUint32(b + 8, code >>> 0, true);
    v.setUint32(b + 12, kind >>> 0, true);
    v.setBigUint64(b + 24, BigInt.asUintN(64, BigInt(addr)), true);
    v.setBigUint64(b + 32, BigInt.asUintN(64, BigInt(rsp)), true);
    P.__gex_snap_fill(BigInt(rec));
    if (k > 0) {
      // (the entry function's own count is left to this: its call of the
      // stop is outside its catch; the hook's call counts its own)
      if (route === 0) this.#snapDepth.value--;
      throw SNAP_NEST;
    }
    v.setUint32(rec + 20, idx >>> 0, true);
    if (SNAP_TRACE) this.#snapTraceLevels(rec);
    this.#snapPark(rec);
    new DataView(this.#memory.buffer).setUint32(rec + SNAP_NLV, 0, true);
    P.asyncify_start_rewind(rec);
  }
  /** A fault hook's call (`f`, function `idx`) in the snapshot flavour:
   *  counted as an entry, and a snapshot's unwind that comes back through
   *  it stops there (`#snapStop`) and the hook is called again, rewound.
   *  `rsp`: the guest's stack pointer as the fault left it. */
  #snapHookCall(f, idx, ctx, code, addr, kind, rsp) {
    if (!this.#snapHooks) return f(ctx, code, addr, kind);
    const P = this.#snapProg().exports, d = this.#snapDepth;
    d.value++;
    try {
      for (;;) {
        f(ctx, code, addr, kind);
        if (P.asyncify_get_state() !== 1) return undefined;
        this.#snapStop(1, idx, ctx, rsp, code, addr, kind);
      }
    } finally {
      d.value--;
    }
  }
  /** A level of a restored or rewound thread's record entered again
   *  (`#snapStop` kept it): the handler's call made as the level's route
   *  made it, rewound, then finished as `w64_entry` finishes one. Where
   *  the guest goes on. */
  #snapLevelEnter(rec, k, ctx) {
    const e = this.#shim.exports;
    const v = new DataView(this.#memory.buffer), b = rec + SNAP_LV + k * SNAP_LVS;
    const route = v.getUint32(b, true), idx = v.getUint32(b + 4, true);
    const rsp = v.getBigUint64(b + 32, true);
    const P = this.#snapProg().exports;
    if (route === 2) {
      // An entry into Wine's PE module, the guest's call of its export: its
      // root called again, rewound, and the guest goes on where the
      // export's return left it.
      if (!this.#snapPe) this.#cgenCatchUp();
      if (!this.#snapPe) throw new Error('snapshot: a level in Wine\'s PE module, which this thread has not installed');
      const W = this.#snapPe.inst.exports, sl = rec + SNAP_HALF + k * SNAP_PE_SLOT;
      W.asyncify_start_rewind(this.#snapPeWin(rec));
      W.__gexw_entry(v.getInt32(sl, true), v.getInt32(sl + 4, true), v.getInt32(sl + 8, true), v.getBigUint64(sl + 16, true), v.getBigUint64(sl + 24, true));
      return this.#faultPc(ctx);
    }
    if (route === 0) {
      P.asyncify_start_rewind(rec);
      this.#entryCallFn(idx, ctx, rsp);
    } else {
      const f = this.#cgenStore().ftab.get(tIdx(this.#wide, idx));
      const wide = e.gex_guest_x86_64?.() === 1;
      const addr = v.getBigUint64(b + 24, true);
      const spG = this.#wasmprogStack();
      const cur = spG.value;
      if (typeof cur === 'bigint') spG.value = ((cur !== 0n && cur < rsp ? cur : rsp) - 4096n) & ~15n;
      else {
        const esp = Number(rsp & 0xffffffffn), c32 = Number(cur) >>> 0;
        spG.value = (((c32 !== 0 && c32 < esp ? c32 : esp) - 4096) & ~15) | 0;
      }
      try {
        P.asyncify_start_rewind(rec);
        this.#snapHookCall(f, idx, ctx, v.getUint32(b + 8, true) | 0, wide ? addr : Number(BigInt.asUintN(32, addr)) | 0, v.getUint32(b + 12, true) | 0, rsp);
      } finally {
        spG.value = cur;
      }
    }
    if (!e.gex_snap_entry_tail(ctx)) throw new Error('snapshot: a restored thread\'s call left another context to resume');
    return this.#faultPc(ctx);
  }
  /** The program's call-out in the snapshot flavour (an import its
   *  asyncify unwinds through): a stop inside the guest code it runs
   *  (`SNAP_NEST`) unwinds the program's frames from here, the program's
   *  stack pointer kept in the inner level; called again by their rewind,
   *  it enters the inner level again (`#snapLevelEnter`) and runs the
   *  guest on from there, as the call-out it was. */
  #snapCallOut(ctx, target, ret, spWant) {
    const P = this.#snapProg().exports;
    const rec = this.#snapRec;
    if (P.asyncify_get_state() === 2) {
      P.asyncify_stop_rewind();
      const j = this.#snapDepth.value;
      const sp = new DataView(this.#memory.buffer).getUint32(rec + SNAP_LV + j * SNAP_LVS + 16, true);
      const spG = this.#wasmprogStack();
      spG.value = typeof spG.value === 'bigint' ? BigInt(sp) : sp | 0;
      return this.#drive(this.#callOutSteps(ctx, target, ret, spWant, false, () => this.#snapLevelEnter(rec, j, ctx)));
    }
    try {
      return this.#callOut(ctx, target, ret, spWant);
    } catch (err) {
      if (err !== SNAP_NEST) throw err;
      const j = this.#snapDepth.value;
      new DataView(this.#memory.buffer).setUint32(rec + SNAP_LV + j * SNAP_LVS + 16, Number(this.#wasmprogStack().value) >>> 0, true);
      this.#snapUnwinding = true;
      P.asyncify_start_unwind(rec);
      return 0;
    }
  }
  /** A snapshot's stop at the root of an entry into Wine's PE module of the
   *  snapshot flavour (gexrt's `__gexw_entry`), whose call came back
   *  unwinding: the level kept (route 2, the root's arguments in its slot),
   *  as `#snapStop` keeps one of the program's, and the root makes the call
   *  again, rewound, when this returns. */
  #snapPeStop(ctx, depth, need, ret, body) {
    const W = this.#snapPe.inst.exports;
    const rec = this.#snapRec;
    if (!this.#snapPeUnwinding || !rec) throw new Error('snapshot: an unwind not the snapshot\'s reached an entry of Wine\'s PE module');
    this.#snapPeUnwinding = false;
    W.asyncify_stop_unwind();
    const k = this.#snapDepth.value - 1;
    const v = new DataView(this.#memory.buffer), b = rec + SNAP_LV + k * SNAP_LVS, sl = rec + SNAP_HALF + k * SNAP_PE_SLOT;
    if (v.getUint32(rec + SNAP_NLV, true) === 0) v.setUint32(rec + SNAP_NLV, k + 1, true);
    v.setUint32(b, 2, true);
    v.setUint32(b + 4, 0, true);
    v.setInt32(sl, ctx, true);
    v.setInt32(sl + 4, depth, true);
    v.setInt32(sl + 8, need, true);
    v.setBigUint64(sl + 16, BigInt.asUintN(64, ret), true);
    v.setBigUint64(sl + 24, BigInt.asUintN(64, body), true);
    // (the root's count goes down as this leaves it)
    if (k > 0) throw SNAP_NEST;
    this.#snapProg().exports.__gex_snap_fill(BigInt(rec));
    v.setUint32(rec + 20, 0, true);
    if (SNAP_TRACE) this.#snapTraceLevels(rec);
    this.#snapPark(rec);
    new DataView(this.#memory.buffer).setUint32(rec + SNAP_NLV, 0, true);
    W.asyncify_start_rewind(this.#snapPeWin(rec));
  }
  /** Wine's PE module's call-out in the snapshot flavour, as the program's
   *  (`#snapCallOut`): the module's per-thread state and stack pointer at
   *  the call kept in the inner level's slot, which its unwind does not
   *  keep and its rewind puts back. */
  #snapPeCallOut(ctx, target, ret, spWant) {
    const pe = this.#snapPe, W = pe.inst.exports;
    const rec = this.#snapRec, j = this.#snapDepth.value;
    const sp = this.#providerStack64();
    const sl = rec + SNAP_HALF + j * SNAP_PE_SLOT;
    if (pe.state.value === 2) {
      W.asyncify_stop_rewind();
      W.__gexw_snap_globals(1, BigInt(sl + 40));
      sp.value = new DataView(this.#memory.buffer).getBigUint64(sl + 80, true);
      return this.#drive(this.#callOutSteps(ctx, target, ret, spWant, false, () => this.#snapLevelEnter(rec, j, ctx)));
    }
    if (rec && j < SNAP_LEVELS) {
      W.__gexw_snap_globals(0, BigInt(sl + 40));
      new DataView(this.#memory.buffer).setBigUint64(sl + 80, sp.value, true);
    }
    try {
      return this.#callOut(ctx, target, ret, spWant);
    } catch (err) {
      if (err !== SNAP_NEST || !rec || j >= SNAP_LEVELS) throw err;
      this.#snapPeUnwinding = true;
      W.asyncify_start_unwind(this.#snapPeWin(rec));
      return 0;
    }
  }
  /** (GEX_SNAP_TRACE) a stopped thread's levels: each one's route. */
  #snapTraceLevels(rec) {
    const v = new DataView(this.#memory.buffer), n = v.getUint32(rec + SNAP_NLV, true);
    const routes = Array.from({ length: n }, (_, k) => v.getUint32(rec + SNAP_LV + k * SNAP_LVS, true));
    console.error(`[gex-snap] tid ${this.#threadTid}: stopped, levels by route [${routes.join(' ')}]`);
  }
  /** The asyncify window of Wine's PE module in this thread's record, made
   *  at the first use (a restored record keeps its own). */
  #snapPeWin(rec) {
    const v = new DataView(this.#memory.buffer), h = rec + SNAP_HALF + SNAP_PE_HDR;
    if (v.getUint32(h + 4, true) === 0) {
      v.setUint32(h, h + 8, true);
      v.setUint32(h + 4, rec + SNAP_PE_END, true);
    }
    return h;
  }
  /** With Wine's PE module of the flavour on this Worker, the program's
   *  asyncify window ends at the record's upper half (`SNAP_HALF`). */
  #snapSplit() {
    const rec = this.#snapRec;
    if (!this.#snapPe || !rec) return;
    const v = new DataView(this.#memory.buffer);
    if (v.getUint32(rec + 4, true) > rec + SNAP_HALF && v.getUint32(rec, true) < rec + SNAP_HALF) v.setUint32(rec + 4, rec + SNAP_HALF, true);
  }
  /** Wine's PE module of the snapshot flavour on this Worker (its instance
   *  and asyncify state), and whether the unwind under way is the
   *  snapshot's, through that module. */
  #snapPe = null;
  #snapPeUnwinding = false;
  /** Whether the program's hooks and call-outs are the snapshot flavour's
   *  (`#snapHookCall`, `#snapCallOut`). */
  #snapHooks = false;
  /** Whether this Worker's thread can be unwound to its program's snapshot
   *  root where it waits now (`snap_ok`): its driver (root 1) with nothing
   *  of the host's above it, or (root 2) the outermost entry from the
   *  thread's outermost call-out, Wine's own loop (the shim opened none for
   *  it). */
  #snapOk() {
    if (!this.#snapRec || this.#vforkDepth) return 0;
    const depth = this.#snapDepth?.value ?? 0;
    if ((this.#snapRoot & 1) && depth === 0 && this.#callouts.length === 0 && this.#nest.length === 0) return 1;
    // (root 2) every level an entry from guest code that the call-out
    // below it runs, each call-out the program's own (Wine's)
    const n = this.#callouts.length;
    if ((this.#snapRoot & 2) && depth >= 1 && depth === n && n <= SNAP_LEVELS && this.#nest.length === n
      && (n === 1 || this.#snapHooks) && this.#callouts.every((c) => c.ri === -1)) return 1;
    if (SNAP_TRACE && !this.#snapTraced) {
      // Once per thread: why it cannot be stopped where it waits.
      this.#snapTraced = true;
      let pc = '?';
      try {
        const v = new DataView(this.#memory.buffer), c = Number(this.#shim.exports.gex_snap_ctx());
        const rsp = v.getBigUint64(c + 4 * 8, true);
        pc = `0x${v.getBigUint64(c + 32 * 8, true).toString(16)} rax 0x${v.getBigUint64(c, true).toString(16)} ret 0x${v.getBigUint64(Number(rsp), true).toString(16)}`;
      } catch {}
      console.error(`[gex-snap] tid ${this.#threadTid}: not at a root (rec ${this.#snapRec}, root ${this.#snapRoot}, vfork ${this.#vforkDepth}, entry depth ${depth}, `
        + `call-outs ${n}${n > SNAP_LEVELS ? `, past the record's ${SNAP_LEVELS} levels` : ''} [${this.#callouts.map((c) => c.ri).join(' ')}], nest ${this.#nest.length}, pc ${pc})`);
    }
    return 0;
  }
  #snapTraced = false;
  /** The program built with the snapshot flavour (its instance), its
   *  thread's roots (bits: 1 a driver, 2 the host's entry), and the
   *  snapshot entry function's depth on this Worker. */
  #snapProg = null;
  #snapRoot = 0;
  #snapDepth = null;
  /** This Worker's thread's snapshot record (`snap_register`), and whether
   *  the unwind under way is the snapshot's. */
  #snapRec = 0;
  #snapUnwinding = false;
  /** posix_spawn children running on this Worker (`#vforkChild`). */
  #vforkDepth = 0;

  /** What a native wasm32 program imports from `gex` (cgen/wasmprog.rs):
   *  its system calls, which return unless the program's frames end (a
   *  thread or a child they staged is handed to the host here), and fork's
   *  asyncify operations on its own instance (the musl fork's
   *  src/process/wasm32/__gex_fork.c). */
  #wasmprogImports(inst, mod = null) {
    const e = this.#shim.exports;
    const ops = ['asyncify_get_state', 'asyncify_start_unwind', 'asyncify_stop_unwind', 'asyncify_start_rewind', 'asyncify_stop_rewind'];
    const asyncify = (op, arg) => {
      const f = inst().exports[ops[op]];
      if (!f) {
        if (op === 0) return 0;
        throw new Error(`gex.asyncify: the program has no ${ops[op]} (fork needs wcc's asyncify)`);
      }
      if (op === 0) return f();
      if (op === 1 || op === 3) f(arg);
      else f();
      return 0;
    };
    const staged = () => {
      if (e.gex_wasmprog_staged()) {
        // posix_spawn's child first, here, when it can be (`#vforkChild`);
        // any other child is started below.
        if (!(SP_CODE && single.on)) {
          const ok = e.gex_vfork_ok?.() ?? 0;
          if (ok > 0) this.#vforkChild();
          else if (ok !== -2) this.#vforkCopied(ok);
        }
        this.#drainFork()?.catch((err) => console.error(`[gex] starting a process: ${err.message ?? err}`));
        this.#startThread();
      }
    };
    // What a system call left: a child or thread to start, and a library
    // another thread has `dlopen`ed since, in this Worker's table before the
    // program goes on (cgen/dylink.rs `DL_GEN`).
    const after = () => {
      staged();
      const g = e.gex_wasmprog_dl_gen?.();
      if (g !== undefined && g !== this.#dlGen) {
        this.#dlGen = g;
        this.#cgenCatchUp();
      }
    };
    const sync = (ctx, pc) => {
      e.gex_syscall(ctx, pc);
      after();
    };
    return {
      // The default build: the shim's export, with no JS frame
      // (`#syscallAfter` gets what the call left).
      syscall: (SP_CODE && single.on) ? this.#suspendingSyscall(sync, after) : (e.gex_wasmprog_syscall ?? sync),
      // a module built with GEX_DL_CHECKED_CALLS (cgen/dylink.rs `checked`):
      // the slots this Worker's table has caught up with, and the catch-up
      // its guard calls at an index past them
      dl_size: this.#dlSize,
      dl_stale: () => {
        this.#dlGen = e.gex_wasmprog_dl_gen?.() ?? this.#dlGen;
        e.gex_wasmprog_dl_seen?.();
        this.#cgenCatchUp();
      },
      asyncify,
      // The snapshot flavour's (libgex-snap/snapshot.c).
      ...this.#snapImports(asyncify, (SP_CODE && single.on) ? this.#suspendingSyscall(sync, after) : sync, inst, mod),
      // -fgex-guest-i386 (gexcc's rt/i386.c): i386 code calling `addr` runs
      // the program's function at table index `idx` (cgen/libprov.rs)
      i386_bind: (addr, idx) => e.gex_libprov_i386_bind(addrBig(addr), idx >>> 0),
      fork_point: (rec) => {
        const st = asyncify(0, 0);
        if (st === 0) asyncify(1, rec);
        else if (st === 2) asyncify(4, 0);
        else throw new Error(`gex.fork_point in asyncify state ${st}`);
      },
      // libgex's dlopen/dlsym past its static registry
      // (cgen/dylink.rs): the shim loads and publishes, this Worker
      // instantiates the new modules and runs their constructors; other
      // Workers instantiate them as they next catch up
      dlopen: (SP_CODE && single.on) ? new WebAssembly.Suspending((path, err, errlen) => this.#dlopen(path, err, errlen, true)) : (path, err, errlen) => this.#dlopen(path, err, errlen, false),
      dlsym: (h, name) => e.gex_wasmprog_dlsym(h, addrBig(name)) | 0,
      // Door calls into the wineserver.
      door_publish: (area) => this.#doorPublish(area >>> 0),
      door_call: (req, pid, tid) => this.#doorCall(addrNum(req), pid, tid, false),
      // A 64-bit client's (64-bit Wine's lowered LP64 ntdll.so): the request's
      // gather list at LP64 offsets (`#doorRun`, hosts/native's door::req_layout).
      door_call_lp64: (req, pid, tid) => this.#doorCall(addrNum(req), pid, tid, true),
      // A Worker the door was broadcast to while it ran (it takes no
      // messages): Wine runs its program again in its place, in a new Worker
      // that starts with the door (loader.c `wineserver_relaunch`,
      // `exec_fresh`). None in the SP build.
      door_missing: () => ((this.#door && this.#doorLive(this.#door)) || (SP_CODE && single.on) ? 0 : 1),
      // The next exec goes on in a new Worker and memory, the process's pid
      // kept (shim `gex_exec_fresh`).
      exec_fresh: () => { e.gex_exec_fresh?.(); },
      // The end of 64-bit Wine's space in a 64-bit memory, in 64 KiB units
      // (shim `gex_space_top`): ntdll's top instead of its probe; 0 to probe.
      space_top: () => (e.gex_space_top?.() ?? 0) | 0,
      // The session mapping, a block of the server's memory (area +128,
      // +132), copied from directly by win32u: the
      // bytes copied, or -1 with no door session; the session's
      // size for a size of 0.
      session_copy: (dst, off, size) => this.#sessionCopy(addrNum(dst), off >>> 0, size >>> 0),
      session_seq: (off) => this.#sessionSeq(off >>> 0),
      // The server's user shared data page (area +140), copied at a
      // process's start and written by the first boot's fill.
      usd_copy: (dst, size) => this.#usdMove(addrNum(dst), size >>> 0, false),
      usd_publish: (src, size) => this.#usdMove(addrNum(src), size >>> 0, true),
      // Where i386 code reads KUSER_SHARED_DATA (32-bit Wine checks its own
      // address against it at start): 0x7ffe0000 from a shim without the fold.
      kuser_address: () => (e.gex_kuser_address ? e.gex_kuser_address() : 0x7ffe0000) | 0,
      // The native DLL a function pointer or static variable is in (32-bit
      // Wine's ntdll, when no image holds the address).
      image_of: (addr) => e.gex_wasmdll_image_of ? e.gex_wasmdll_image_of(BigInt.asUintN(64, BigInt(addr))) : 0n,
      // The server's own (the pump): a thread's wait pipe, which door calls
      // write in place. Global pipe ids.
      wake_pipe: (fd) => this.#shim.exports.gex_fd_pipe_id?.(fd) ?? -1,
      wake_write: (id, buf, len) => this.#kernel?.k_pipe_write?.(id, addrBig(buf), len >>> 0) ?? -38,
      wake_close: () => {},
    };
  }

  /** libgex's dlopen past its static registry (cgen/dylink.rs): the shim
   *  loads and publishes, this Worker instantiates the new modules and runs
   *  their constructors; other Workers instantiate them as they next catch
   *  up. In the single-thread build a constructor may block (a system call
   *  that waits), so each runs on a stack of its own and the program's
   *  waits for it (`suspend`), unless a JS frame below cannot suspend. */
  #dlopen(path, err, errlen, suspend) {
    const e = this.#shim.exports;
    const h = e.gex_wasmprog_dlopen(addrBig(path), addrBig(err), errlen) >>> 0;
    if (!h) return 0;
    this.#cgenCatchUp();
    const st = this.#cgenStore();
    // Taken whole before any constructor runs: one that calls dlopen
    // itself replaces the list.
    const fresh = [];
    for (let k = 0, n; (n = e.gex_wasmprog_dl_new(k)) >= 0; k++) fresh.push(n);
    const ctors = fresh.map((n) => st.dyl.get(n)?.exports.__wasm_call_ctors).filter(Boolean);
    if (!suspend || this.#hardNested()) {
      for (const f of ctors) f();
      return h | 0;
    }
    return (async () => {
      for (const f of ctors) await this.#promisingOf(f)();
      return h | 0;
    })();
  }

  // -- door calls ------------------------------
  //
  // The wineserver is its own process. When it is up it publishes its door
  // area; the host hands the server's memory and module to every Worker.
  // A client thread then makes a Wine request by running it itself, on its
  // Worker's own instance of the server's module on the server's memory,
  // under the server's door lock: no pipe, no wake. A system call the server
  // code makes inside a door is answered here when it needs no process (the
  // clock, getpid), and otherwise proxied to the server's pump.

  /** This process's wasm32 program as installed (`#cgenInstall`). */
  #wasmprogInfo = null;
  /** Wine's PE modules' `L` thunk modules, by provider number (#wasmdllBind). */
  #wasmdllThunkMods = new Map();
  /** The session's door server, and this Worker's instance on it. */
  #door = null;
  /** The session's door word (host.mjs `#doorWord`): the pid of the server
   *  whose door is current, written by that server as it publishes and
   *  cleared when it ends. Read where a door is used, so a Worker never has
   *  to have heard a message to know that its door is another server's. */
  #doorWord = null;
  #doorInst = null;
  #doorViews = null;
  /** Census: door calls, fallbacks, proxies. */
  #doorStats = { calls: 0, fallbacks: 0, proxies: 0, t0: 0, nodoor: 0, hi: 0, tPre: 0, tDisp: 0, tPost: 0, T2: 0, tSys: 0, sys: 0, tProxy: 0, byNr: {} };

  /** The server: `area` is its door area, in this process's memory. */
  #doorPublish(area) {
    const info = this.#wasmprogInfo;
    if (!info || (SP_CODE && single.on)) return;
    const e = this.#shim.exports;
    const dv = new DataView(this.#memory.buffer);
    const kickFd = dv.getInt32(area + 68, true);
    const kickId = e.gex_fd_pipe_id?.(kickFd) ?? -1;
    if (kickId < 0) {
      console.error(`[gex-door] the server's kick fd ${kickFd} is not a global pipe: no doors`);
      return;
    }
    // The door context must hold gex's whole register context (gex_abi
    // ctx::SIZE, 1120 bytes for i386): the kernel writes past the
    // system-call slots, and a short one corrupted the server's heap.
    const ctxSize = dv.getUint32(area + 136, true);
    if (ctxSize < 1120) {
      console.error(`[gex-door] the server's door context is ${ctxSize} bytes, under gex's 1120: no doors`);
      return;
    }
    const door = {
      memory: this.#memory, module: info.module, base: info.base, memBase: info.memBase, binds: info.binds ?? this.#wasmprogBindRead(info.prov), packed: !!info.packed,
      area, kickId, pid: dv.getInt32(area + 112, true), id: `${Date.now()}:${area}`,
    };
    if (globalThis.GEX_DOOR_TRACE) console.error(`[gex-door] wineserver pid ${door.pid} published its door (area ${area})`);
    // Current from now, for every Worker of the session at once: before the
    // server says it is ready, so the process that started it finds its own
    // door (if any) not this one, and runs again with this one
    // (ntdll wineserver_relaunch, `door_missing`).
    if (this.#doorWord) Atomics.store(this.#doorWord, 0, door.pid);
    this.#env.publishDoor?.(door);
  }

  /** `gex.usd_copy` / `gex.usd_publish`: the server's user shared data
   *  page (area +140) to or from this process; the bytes copied, -1 with no
   *  door, and for a copy -2 while no process has published a fill there
   *  (area +144 USD_UNPUBLISHED): the bytes are copied, but only the
   *  server's own fields are good, and the fill is in the caller's mapping.
   *  A publish marks it USD_PUBLISHED. A server without the field has
   *  neither value there. */
  #usdMove(at, size, publish) {
    const d = this.#door;
    if (!d || (SP_CODE && single.on) || !this.#doorLive(d)) return -1;
    const { u8, dv, i32 } = this.#doorView();
    const usd = dv.getUint32(d.area + 140, true);
    if (!usd) return -1;
    const state = Atomics.load(i32, (d.area + 144) >>> 2);
    const n = Math.min(size, 4096);
    if (publish) {
      u8.set(this.#bytes().subarray(at, at + n), usd);
      if (state === USD_UNPUBLISHED) Atomics.store(i32, (d.area + 144) >>> 2, USD_PUBLISHED);
      return n;
    }
    this.#bytes().set(u8.subarray(usd, usd + n), at);
    return state === USD_UNPUBLISHED ? -2 : n;
  }

  #sessTr = 0;
  /** The door session (the server's session block, `#sessionCopy`): its
   *  place and size, which never change once the server has published it,
   *  and views of the server's memory taken when it was first read. A
   *  shared memory's old buffer stays valid at its old length, so the
   *  views serve every later read; win32u reads the session several times
   *  per call (a seqlock's copy and its release, per shared object: four
   *  for GetKeyState), and each read went through `memory.buffer` twice. */
  #sess = null;
  #session() {
    const d = this.#door;
    if (!d || !this.#doorLive(d)) return null;
    let s = this.#sess;
    if (s && s.door === d && s.u8.length >= s.end) return s;
    const v = this.#doorView();
    const base = v.dv.getUint32(d.area + 128, true), len = v.dv.getUint32(d.area + 132, true);
    if (!base || !len) return null;
    s = { door: d, base, len, end: base + len, u8: v.u8, i64: v.i64 };
    // not yet inside the views (a server that grew since): read once more next time
    if (v.u8.length < s.end) return null;
    return this.#sess = s;
  }

  #sessionCopy(dst, off, size) {
    if (globalThis.GEX_SESSION_TRACE && !(++this.#sessTr % 20000)) console.error(`[session] copy ${this.#sessTr} off ${off} size ${size}`);
    const s = this.#session();
    if (!s) return -1;
    if (!size) return s.len;
    if (off >= s.len) return -1;
    const n = Math.min(size, s.len - off);
    this.#doorGuest(dst + n).u8.set(s.u8.subarray(s.base + off, s.base + off + n), dst);
    return n;
  }

  #sessionSeq(off) {
    if (globalThis.GEX_SESSION_TRACE && !(++this.#sessTr % 20000)) console.error(`[session] seq ${this.#sessTr} off ${off}`);
    const s = this.#session();
    if (!s) return 0n;
    const at = s.base + off;
    // A seqlock's word: whole, as the server stores it (8-aligned).
    if (!(at & 7)) return Atomics.load(s.i64, at / 8);
    return new DataView(s.u8.buffer).getBigInt64(at, true);
  }

  /** The session's door word, from the start message. */
  setDoorWord(word) {
    this.#doorWord = word ?? null;
  }

  /** Whether `d` is the current server's door: one whose server ended, or
   *  that a later server's replaced, is not (a program that started while
   *  a server was ending, and connected to the next). */
  #doorLive(d) {
    const w = this.#doorWord;
    return !w || Atomics.load(w, 0) === d.pid;
  }

  /** The door server, from the host (all Workers of the session). */
  setDoor(door) {
    if (this.#door?.id === door?.id) return;
    this.#door = door;
    this.#doorInst = null;
    this.#doorViews = null;
  }

  #doorView() {
    const d = this.#door;
    const buf = d.memory.buffer;
    let v = this.#doorViews;
    if (!v || v.buf !== buf || v.len !== buf.byteLength) {
      v = this.#doorViews = { buf, len: buf.byteLength, u8: new Uint8Array(buf), i32: new Int32Array(buf), i64: new BigInt64Array(buf, 0, Math.floor(buf.byteLength / 8)), dv: new DataView(buf) };
    }
    return v;
  }

  #doorInstantiate(d) {
    const v = this.#doorView();
    const a = d.area;
    const stackTop = v.dv.getUint32(a + 52, true);
    const sp = new WebAssembly.Global({ value: 'i32', mutable: true }, stackTop);
    // `slots` is the program's table reservation (tens of millions, past
    // an engine's table limit): the door's table needs only the module's
    // own elements, its dylink table_size.
    // The binding list (dylink.rs): a static program's are its exports past
    // the element segment (`X`) and its GOT (`G`); a server with shared
    // libraries is not a door server.
    const extra = [];
    const got = [];
    let size = d.base + dylinkTableSize(d.module);
    const [lines, nums] = bindSplit(d.binds);
    let at = 0;
    for (const line of lines.split('\n')) {
      if (!line) continue;
      const f = line.split('\t');
      if (f[0] === 'X') { const slot = Number(nums[at++]); extra.push([slot, f[2]]); size = Math.max(size, slot + 1); }
      else if (f[0] === 'G') { f[3] = nums[at++]; got.push(f); }
      else if (f[0] === 'S') { if (f[1] === 'env' && f[2].startsWith('__asyncify_')) got.push(['G', 'env', f[2], '0', '1']); }
      else throw new Error(`door: the server program's binding ${f[0]} ${f[1]}.${f[2]} is not supported`);
    }
    const table = new WebAssembly.Table({ element: 'anyfunc', initial: size });
    class DoorExit extends Error {}
    const st = this.#doorStats;
    const syscall = globalThis.GEX_DOOR_STATS
      ? (ctx) => { const t = performance.now(); try { this.#doorSyscall(ctx >>> 0, DoorExit); } finally { st.sys++; st.tSys += performance.now() - t; } }
      : (ctx) => this.#doorSyscall(ctx >>> 0, DoorExit);
    const imports = {
      env: {
        memory: d.memory, __indirect_function_table: table, __stack_pointer: sp,
        __memory_base: new WebAssembly.Global({ value: 'i32', mutable: false }, d.memBase | 0),
        __table_base: new WebAssembly.Global({ value: 'i32', mutable: false }, d.base | 0),
      },
      gex: {
        syscall,
        asyncify: (op) => { if (op === 0) return 0; throw new Error('gex.asyncify: no fork inside a door call'); },
        // A snapshot-flavour server's: a door call is never a snapshot's
        // stop (its thread has no record), so its calls are plain ones.
        snap_wanted: () => 0,
        snap_register: () => {},
        snap_unwinding: () => 0,
        // a thread's wait pipe is resolved by the pump (init_thread); written in place
        wake_pipe: () => -1,
        wake_write: (id, buf, len) => this.#doorWake(id, buf >>> 0, len >>> 0),
        wake_close: () => {},
        // The lazy clock of a DOOR_VERSION 2 server (door_time_refresh): the
        // shim's export, a wasm-to-wasm call.
        clock_ns: this.#shim.exports.gex_clock_ns,
      },
    };
    for (const f of got) (imports[f[1]] ??= {})[f[2]] = new WebAssembly.Global({ value: 'i32', mutable: f[4] === '1' }, Number(f[3]) | 0);
    for (const imp of WebAssembly.Module.imports(d.module)) {
      if (imp.kind === 'function' && !imports[imp.module]?.[imp.name]) {
        (imports[imp.module] ??= {})[imp.name] = () => { throw new Error(`${imp.module}.${imp.name}: not available inside a door call`); };
      }
    }
    const inst = new WebAssembly.Instance(d.module, imports);
    if (!d.packed) for (const [slot, name] of extra) table.set(slot, inst.exports[name]);
    // Its first call sets its context, thread pointer and TLS
    // (`__gex_thread_start` on the area's frame) and ends at the exit call.
    const ctx = v.dv.getUint32(a + 72, true);
    const frame = v.dv.getUint32(a + 76, true);
    v.dv.setUint32(ctx + 32, frame, true);
    v.dv.setUint32(ctx + 36, 0, true);
    try {
      table.get(v.dv.getUint32(a + 108, true))(ctx, 0, 0, 0n);
      throw new Error('door: __gex_thread_start returned');
    } catch (err) {
      if (!(err instanceof DoorExit)) throw err;
    }
    const dispatch = table.get(v.dv.getUint32(a + 60, true));
    // DOOR_VERSION 2 (area +40): the server reads the clock itself, lazily
    // (`dispatch(pid, tid, 0, 1)`), so a request that never looks at the
    // time costs no clock read (Chrome: performance.now() alone is ~0.5 us).
    const lazyTime = v.dv.getInt32(a + 40, true) >= 2;
    return { door: d, dispatch, sp, stackTop, lazyTime, bootNs: BigInt(this.#kernel?.k_boot_secs?.(0) ?? 0) * 1000000000n + this.#monoHeld() };
  }

  /** A system call from server code inside a door (i386 numbers, the
   *  context in the server memory). */
  #doorSyscall(ctx, DoorExit) {
    const d = this.#door;
    const { dv, i32 } = this.#doorView();
    const nr = dv.getUint32(ctx, true);
    const arg = (slot) => dv.getUint32(ctx + slot * 8, true);
    const ret = (r) => { dv.setUint32(ctx, r >>> 0, true); dv.setUint32(ctx + 4, 0, true); };
    switch (nr) {
      case 1: case 252: throw new DoorExit();
      case 20: case 224: ret(d.pid); return;
      case 265: case 403: case 78: {
        // clock_gettime, clock_gettime64, gettimeofday: as the shim's
        // browser_clock and gex-kernel's monotonic_from (one boot base).
        const clk = arg(3), buf = nr === 78 ? arg(3) : arg(1);
        if (nr === 78 || clk === 0 || clk === 5 || clk === 8 || [1, 4, 6, 7, 9].includes(clk)) {
          if (!buf) { ret(nr === 78 ? 0 : -14); return; }
          const now = this.#env.now();
          const ns = nr !== 78 && [1, 4, 6, 7, 9].includes(clk) ? this.#guestMono(now) : now - this.#guestClock().shift * 1000000000n;
          const secs = Number(ns / 1000000000n), nanos = Number(ns % 1000000000n);
          if (nr === 403) { dv.setBigUint64(buf, BigInt(secs), true); dv.setBigUint64(buf + 8, BigInt(nanos), true); }
          else { dv.setUint32(buf, secs >>> 0, true); dv.setUint32(buf + 4, nr === 78 ? Math.floor(nanos / 1000) : nanos, true); }
          ret(0);
          return;
        }
        break;
      }
    }
    // Proxied: the pump makes it as the server process. ebx, ecx, edx,
    // esi, edi, ebp.
    const a = d.area;
    i32[(a + 8) >> 2] = nr;
    const regs = [3, 1, 2, 6, 7, 5];
    for (let k = 0; k < 6; k++) i32[(a + 12 + 4 * k) >> 2] = arg(regs[k]);
    const stats = dv.getUint32(a + 96, true);
    if (stats && nr < 512) Atomics.add(i32, (stats >> 2) + nr, 1);
    Atomics.add(i32, (a + 100) >> 2, 1);
    const st = this.#doorStats;
    st.proxies++;
    const tp = globalThis.GEX_DOOR_STATS ? performance.now() : 0;
    Atomics.store(i32, (a + 4) >> 2, 1);
    Atomics.notify(i32, a >> 2);
    this.#doorKick();
    while (Atomics.load(i32, (a + 4) >> 2) !== 2) Atomics.wait(i32, (a + 4) >> 2, 1, 5);
    const r = i32[(a + 36) >> 2];
    Atomics.store(i32, (a + 4) >> 2, 0);
    if (tp) { st.tProxy += performance.now() - tp; (st.byNr[nr] = (st.byNr[nr] ?? 0) + 1); }
    ret(r);
  }

  #doorScratch = 0;

  /** `gex.wake_write` inside a door: a client's wait pipe written from this
   *  Worker (the pipes are the global kernel's), not proxied to the pump. */
  #doorWake(id, buf, len) {
    if (len > 64 || !this.#doorScratch) return -22;
    const { u8 } = this.#doorView();
    this.#bytes().set(u8.subarray(buf, buf + len), this.#doorScratch);
    return this.#kernel?.k_pipe_write?.(id, BigInt(this.#doorScratch), len) ?? -38;
  }

  #doorKick() {
    // one byte, from anywhere in this process's memory: the pump only drains
    this.#kernel?.k_pipe_write?.(this.#door.kickId, 0n, 1);
  }

  /** `gex.door_call(req, unix_pid, unix_tid)`: 0 done (the reply in `req`),
   *  1 use the pipes, 2 the thread died in the request. The hot path is kept
   *  small and free of the statistics (`#doorCallStats`, GEX_DOOR_STATS), so
   *  it stays optimised: every server request of a 32-bit Wine process comes here. */
  #doorCall(req, pid, tid, lp64) {
    // A 64-bit memory's client (64-bit Wine's memory64 flavour) takes the door too:
    // the server's instance here touches only the server's own wasm32
    // memory, and the request is copied in and out at full 64-bit
    // addresses (`#doorRun`'s `wide`).
    if (this.#doorStatsOn) return this.#doorCallStats(req, pid, tid, lp64);
    const d = this.#door;
    if (!d || (SP_CODE && single.on)) return 1;
    return this.#doorRun(d, req, pid, tid, null, lp64);
  }

  /** GEX_DOOR_STATS, read once per Worker (a page sets it before it starts). */
  #doorStatsOn = !!globalThis.GEX_DOOR_STATS;

  /** Views of the guest's memory for door calls: { u8, i32, dv } (word
   *  indices with `>>> 2`: the memory passes 2 GiB). A shared
   *  memory never detaches (an old SharedArrayBuffer stays valid at its old
   *  length), so a view is good for any range inside it and is rebuilt only
   *  when a range reaches past it. Reading `memory.buffer` for every call,
   *  as `#bytes` does, was about 15% of a door call in Chrome. */
  #doorGv = null;
  #doorGuest(end) {
    let g = this.#doorGv;
    if (!g || end > g.u8.length) {
      const buf = this.#memory.buffer;
      g = this.#doorGv = memoryViews(buf);
    }
    return g;
  }

  /** The door memory's views (`#doorView`), checked the same way. */
  #doorMem(end) {
    const v = this.#doorViews;
    return v && end <= v.len ? v : this.#doorView();
  }

  /** One door call under the door lock (`st`: the statistics, or null).
   *  The client's `struct __server_request_info`, past its 64-byte union:
   *  data_count, reply_data and data[] (pointer, size) at 64, 68 and 72 in
   *  8-byte entries (ILP32), or at 64, 72 and 80 in 16-byte ones (`lp64`).
   *  A lowered LP64 client's pointers are below 4 GiB; a 64-bit memory's
   *  (`wide`) are read whole, and its addresses, which pass 4 GiB, are
   *  never put through 32-bit operators. */
  #doorRun(d, req, pid, tid, st, lp64) {
    const wide = this.#wide;
    const T = st ? performance.now() : 0;
    // The request's header is in the door buffer once copied in: its 64
    // bytes are the scratch an in-place wakeup is written from (#doorWake).
    this.#doorScratch = req;
    let g = this.#doorGuest(req + (lp64 ? 160 : 136));
    const cdv = g.dv;
    const reqSize = cdv.getUint32(req + 4, true);
    const replyMax = cdv.getUint32(req + 8, true);
    const count = Math.min(cdv.getUint32(req + 64, true), 16);
    // reply_data is set only for a request with a reply buffer: with
    // reply_size 0 it is whatever the client's stack held, so it is read
    // only then (a stale one made #doorGuest rebuild its views per call).
    const replyData = !replyMax ? 0 : wide ? Number(cdv.getBigUint64(req + 72, true)) : cdv.getUint32(req + (lp64 ? 72 : 68), true);
    if (st && replyData + replyMax > st.hi) st.hi = replyData + replyMax;
    const data0 = lp64 ? 80 : 72, entry = lp64 ? 16 : 8;
    const a = d.area;
    // The door buffer's place and size never change: read once per door.
    let buf = d.kBuf, bufSize = d.kBufSize;
    if (buf === undefined) {
      const v0 = this.#doorMem(a + 144);
      buf = d.kBuf = v0.dv.getUint32(a + 44, true);
      bufSize = d.kBufSize = v0.dv.getUint32(a + 48, true);
    }
    const { u8, i32, dv } = this.#doorMem(buf + bufSize);
    const replyOff = (64 + reqSize + 7) & ~7;
    if (replyOff + 64 + replyMax > bufSize) { if (st) st.fallbacks++; return 1; }
    // the lock: 0 free, 1 held, 2 held with waiters
    const L = a >> 2;
    // Poisoned (a door call died midway, or the server ended): no more
    // doors; the pipes say the rest when the server goes. A plain read,
    // before the lock as the atomic load was: the flag only ever goes from
    // 0 to set. Nor another server's door than the current one.
    if (i32[(a + 124) >> 2] !== 0 || !this.#doorLive(d)) return 1;
    if (Atomics.compareExchange(i32, L, 0, 1) !== 0 && !this.#doorLockSlow(i32, a, L, pid, tid)) return 1;
    i32[(a + 116) >> 2] = tid;
    i32[(a + 120) >> 2] = pid;
    let r;
    try {
      // This Worker's instance, started under the lock: its start runs on
      // the one door stack and context.
      let di = this.#doorInst;
      if (!di || di.door !== d) di = this.#doorInst = this.#doorInstantiate(d);
      // The 64-byte header as 16 words (no subarray views), then the data.
      if (!wide && ((req | buf) & 3) === 0) {
        const gi = g.i32, s0 = req >>> 2, d0 = buf >>> 2;
        for (let k = 0; k < 16; k++) i32[d0 + k] = gi[s0 + k];
      } else u8.set(g.u8.subarray(req, req + 64), buf);
      let at = buf + 64;
      for (let k = 0; k < count; k++) {
        const e0 = req + data0 + entry * k;
        if (e0 + entry > g.u8.length) g = this.#doorGuest(e0 + entry);
        const ptr = wide ? Number(g.dv.getBigUint64(e0, true)) : g.dv.getUint32(e0, true), size = g.dv.getUint32(e0 + entry / 2, true);
        if (st && ptr + size > st.hi) st.hi = ptr + size;
        if (ptr + size > g.u8.length) g = this.#doorGuest(ptr + size);
        u8.set(g.u8.subarray(ptr, ptr + size), at);
        at += size;
      }
      di.sp.value = di.stackTop;
      const T1 = st ? performance.now() : 0;
      if (di.lazyTime) r = di.dispatch(pid, tid, 0n, 1n) >>> 0;
      else {
        // The time, as set_current_time would read it (gettimeofday,
        // CLOCK_BOOTTIME, both `env.now()`), in 100 ns: two clock calls out
        // of the door saved.
        const ns = this.#env.now(), c = this.#guestClock();
        const wall = ns - c.shift * 1000000000n, mono = c.scale === 1 ? ns - di.bootNs : this.#guestMono(ns);
        r = di.dispatch(pid, tid, wall / 100n + 116444736000000000n, mono / 100n) >>> 0;
      }
      if (st) { const T2 = performance.now(); st.tPre += T1 - T; st.tDisp += T2 - T1; st.T2 = T2; }
      if (r === 0xffffffff) { if (st) st.fallbacks++; r = 1; }
      else if (r === 0xfffffffe) r = 2;
      else {
        const ro = buf + r;
        // the dispatch may have grown the guest's memory (a request that maps)
        g = this.#doorGuest(Math.max(req + 64, replyData + replyMax));
        if (!wide && ((req | ro) & 3) === 0) {
          const gi = g.i32, s0 = ro >>> 2, d0 = req >>> 2;
          for (let k = 0; k < 16; k++) gi[d0 + k] = i32[s0 + k];
        } else g.u8.set(u8.subarray(ro, ro + 64), req);
        const size = Math.min(dv.getUint32(ro + 4, true), replyMax);
        if (size && replyData) g.u8.set(u8.subarray(ro + 64, ro + 64 + size), replyData);
        if (st) st.calls++;
        r = 0;
      }
    } catch (err) {
      this.#doorInst = null;
      // A handler cut off midway: the server is poisoned (area +124, the
      // caller's tid), and its pump ends it at its next turn, as when
      // wineserver dies. The caller takes the pipes, which then close.
      Atomics.store(i32, (a + 124) >> 2, tid || -1);
      console.error(`[gex-door] the wineserver trapped in a door call (pid ${pid} tid ${tid}); the server is poisoned: ${err?.stack ?? err}`);
      r = 1;
    }
    // kick_needed is written only under the lock (by this call's dispatch,
    // or the pump): plain accesses, ordered by the lock's own atomics.
    const K = (a + 80) >> 2, kick = i32[K] !== 0;
    if (kick) i32[K] = 0;
    if (Atomics.exchange(i32, L, 0) === 2) Atomics.notify(i32, L);
    if (kick) this.#doorKick();
    return r;
  }

  /** The door lock, contended: spin, then wait (the lock word 2: waiters).
   *  False, the lock not taken, when a session snapshot asks this thread to
   *  stop meanwhile: the holder may be stopped holding it (the server's
   *  pump stops at any call that can block), and the request goes by the
   *  pipe, whose wait stops. */
  #doorLockSlow(i32, a, L, pid, tid) {
    let waitedMs = 0;
    for (let spins = 0; ; spins++) {
      const v = Atomics.compareExchange(i32, L, 0, 1);
      if (v === 0) return true;
      if (spins < 200) continue;
      // Poisoned meanwhile (the server ended, or a call died in it), or no
      // longer the current server's: the lock may never come free; the pipe
      // says the rest.
      if (Atomics.load(i32, (a + 124) >> 2) !== 0 || !this.#doorLive(this.#door ?? {})) return false;
      if (this.#snapRec && this.#kernel?.k_snap_pending?.()) return false;
      if (v === 1 && Atomics.compareExchange(i32, L, 1, 2) !== 1) continue;
      Atomics.wait(i32, L, 2, 1);
      // A lock held this long is a stuck door (a Worker ended inside one,
      // or a deadlock): say who holds it, every 5 s of waiting.
      if (++waitedMs % 5000 === 0) {
        console.error(`[gex-door] pid ${pid} tid ${tid}: ${waitedMs / 1000} s waiting for the door lock, held by tid ${i32[(a + 116) >> 2]} (proxy state ${i32[(a + 4) >> 2]})`);
      }
    }
  }

  /** `#doorCall` with GEX_DOOR_STATS: the same call, timed, and a report
   *  every GEX_DOOR_STATS ms (default 10 s) or 50,000 calls. */
  #doorCallStats(req, pid, tid, lp64) {
    const d = this.#door;
    const st = this.#doorStats;
    if (!d || (SP_CODE && single.on)) {
      if (!(st.nodoor++ & 0x3fff)) console.error(`[gex-door] pid ${pid} tid ${tid}: no door here (${st.nodoor} calls)`);
      return 1;
    }
    st.hi = Math.max(st.hi, req);
    const r = this.#doorRun(d, req, pid, tid, st, lp64);
    const { i32, dv } = this.#doorView();
    const a = d.area;
    const now = performance.now();
    if (st.T2) { st.tPost += now - st.T2; st.T2 = 0; }
    if (!st.t0) st.t0 = now;
    else if (now - st.t0 >= (globalThis.GEX_DOOR_STATS > 1 ? globalThis.GEX_DOOR_STATS : 10000) || st.calls + st.fallbacks >= 50000) {
      const sa = dv.getUint32(a + 96, true) >> 2;
      const top = [];
      if (sa) for (let n = 0; n < 512; n++) if (i32[sa + n]) top.push([i32[sa + n], n]);
      top.sort((x, y) => y[0] - x[0]);
      console.error(`[gex-door] pid ${pid} tid ${tid}: ${st.calls} door calls, ${st.fallbacks} fallbacks, ${st.proxies} proxies in ${Math.round(now - st.t0)} ms;`
        + ` server: ${i32[(a + 104) >> 2]} calls, ${i32[(a + 100) >> 2]} proxies (${top.slice(0, 16).map(([c, n]) => `${n}:${c}`).join(' ')});`
        + ` per call us: in ${(st.tPre / st.calls * 1000).toFixed(2)} dispatch ${(st.tDisp / st.calls * 1000).toFixed(2)} out ${(st.tPost / st.calls * 1000).toFixed(2)};`
        + ` in-door syscalls ${st.sys} (${(st.tSys / Math.max(1, st.sys) * 1000).toFixed(2)} us each);`
        + ` proxied ${st.proxies} waited ${st.tProxy.toFixed(1)} ms [${Object.entries(st.byNr).sort((x, y) => y[1] - x[1]).slice(0, 6).map(([n, c]) => `${n}:${c}`).join(' ')}];`
        + ` in doors ${(st.tPre + st.tDisp + st.tPost).toFixed(1)} ms; highest client address 0x${st.hi.toString(16)}`);
      st.tPre = st.tDisp = st.tPost = st.tSys = st.sys = st.tProxy = 0;
      st.byNr = {};
      st.t0 = now; st.calls = st.fallbacks = st.proxies = 0;
    }
    return r;
  }

  /** A native wasm32 program the shim staged (a load, an exec, a fork's
   *  child): published and installed here before the guest runs. */
  #wasmprogServe() {
    const e = this.#shim.exports;
    const t0 = logNamed('forklink') ? performance.now() : 0;
    const n = e.gex_wasmprog_serve?.() ?? -1;
    if (n < 0) return;
    // `GEX_LOG=forklink`: a fork's child took its parent's link (cgen/dylink.rs
    // `adopt_inherited`), or bound its libraries again.
    if (t0) {
      const how = e.gex_wasmprog_inherited?.() ? "its parent's link" : 'its libraries bound here';
      console.warn(`[forklink] pid ${e.gex_my_pid?.() ?? '?'}: served with ${how} in ${(performance.now() - t0).toFixed(1)} ms`);
    }
    // Published, not yet installed here: the spare installs them meanwhile.
    this.#offerSpare((e.gex_ftab_log_len?.() >>> 0) - (this.#cgen?.have ?? 0));
    this.#cgenCatchUp();
    e.gex_cgen_pmod_installed(n);
    // libc's registry of the program's shared libraries (cgen/dylink.rs)
    e.gex_wasmprog_ready?.();
  }

  /** A spare waiting for its thread (`adoptThread`): what it installs, the
   *  Worker that published it installs too, and applies the module's data
   *  relocations there -- once, by whichever instance is first to ask -- so
   *  they are done before that Worker runs the module. */
  #spareWait = false;
  /** The `gex_exec_gen` a spare was last offered for (or, in a fork's
   *  child, of the program it was forked running). */
  #spareGen = -1;

  /**
   * A program just published with many modules (`n` installs to come): a
   * Worker for its first thread, made now and caught up with them while
   * this Worker installs them too and the program starts
   * (host.mjs `#startSpare`), so the thread's `clone` does not wait for
   * every module to be instantiated again. Once per program; none in the
   * single-thread build, whose threads are fibers of this realm.
   */
  #offerSpare(n) {
    const e = this.#shim.exports;
    const gen = e.gex_exec_gen?.() ?? 0;
    const why = gen === this.#spareGen ? 'its parent\'s program' : (SP_CODE && single.on) ? 'single-thread build'
      : typeof this.#env.offerSpare !== 'function' ? 'no offerSpare' : n < spareModules() ? 'few modules'
      : !isShared(this.#memory?.buffer) ? 'memory not shared' : '';
    if (logNamed('spare')) console.warn(`[spare] pid ${e.gex_my_pid?.() ?? '?'}: program ${gen} with ${n} modules, ${why ? `no spare (${why})` : 'offered'}`);
    this.#spareGen = gen;
    if (why) return;
    this.#programStore(true);
    this.#env.offerSpare({
      memory: this.#memory,
      moduleStore: this.#moduleStore,
      compat: e.gex_compat?.() ?? 0,
      procOpts: this.#opts,
      execGen: gen,
    });
  }

  /** Wine's PE module's `gex.syscall` (cgen/wasmdll.rs): a system call from
   *  native code (gexrt marks its pc, bit 61) that traps to JS -- a unit to
   *  install, a thread to start -- is served here and returns, so the native
   *  frames below it survive, as natively. Anything else (the process or
   *  thread ending, a callback's return) goes on up. */
  /** Tier-1's `gex.syscall`: the shim's export itself, except in a process with a
   *  stateful provider, where guest code a provider called (its frames below) stages a
   *  thread or a child rather than trapping, and the host starts it here. */
  #guestSyscall(e) {
    if (!Gex.wasmDlls) return e.gex_syscall;
    const staged = () => {
      if (e.gex_wasmprog_staged()) {
        this.#drainFork()?.catch((err) => console.error(`[gex] starting a process: ${err.message ?? err}`));
        this.#startThread();
      }
    };
    const sync = (ctx, pc) => {
      e.gex_syscall(ctx, pc);
      staged();
    };
    return (SP_CODE && single.on) ? this.#suspendingSyscall(sync, staged) : (e.gex_staged_syscall ?? sync);
  }

  /** A wasm64 provider's `gex.syscall(ctx, pc)` (cgen/libprov.rs): marked, so the shim
   *  serves it without leaving the provider's frames (a blocking call waits in place, a
   *  thread is staged); the host then starts what it staged, and the call returns. */
  #providerSyscallSync = (ctx, pc) => {
    const e = this.#shim.exports;
    e.gex_syscall(ctx, BigInt.asIntN(64, BigInt(pc) | (1n << 61n)));
    this.#providerStaged();
  };
  #providerStaged = () => {
    const e = this.#shim.exports;
    if (e.gex_wasmprog_staged()) {
      this.#drainFork()?.catch((err) => console.error(`[gex] starting a process: ${err.message ?? err}`));
      this.#startThread();
    }
  };
  #providerSyscallS = null;
  /** The single-thread build's: see `#suspendingSyscall`. */
  get #providerSyscall() {
    if (!(SP_CODE && single.on)) return this.#shim.exports.gex_provider_syscall ?? this.#providerSyscallSync;
    return (this.#providerSyscallS ??= this.#suspendingSyscall(this.#providerSyscallSync, this.#providerStaged, 1n << 61n));
  }

  /** What a system call through the shim's export (lib.rs
   *  `gex_wasmprog_syscall` and the rest) left: bit 0 a thread or a child
   *  it staged, bit 1 a wasm32 program's (whose posix_spawn child may run in
   *  this Worker), bit 2 a library another thread `dlopen`ed since this
   *  Worker last caught up. */
  #syscallAfter(flags) {
    if (flags & 1) {
      if (flags & 2) {
        const e = this.#shim.exports;
        const ok = e.gex_vfork_ok?.() ?? 0;
        if (ok > 0) this.#vforkChild();
        else if (ok !== -2) this.#vforkCopied(ok);
      }
      this.#drainFork()?.catch((err) => console.error(`[gex] starting a process: ${err.message ?? err}`));
      this.#startThread();
    }
    if (flags & 4) this.#cgenCatchUp();
  }

  // gex.native_try's functions, by table index: the last one called (the
  // system call entry, every 64-bit call), then the rest (wow64's thunks).
  // Not an array: the indices are in the thousands, and a sparse array's
  // lookups hashed (a third of the import's cost).
  #tryTab = null;
  #tryIdx = -1;
  #tryFn = null;
  #tryFns = new Map();

  #nativeTry(idx, arg) {
    let f;
    if (idx === this.#tryIdx && this.#cgen.ftab === this.#tryTab) f = this.#tryFn;
    else {
      if (idx === 0) return 2;
      const tab = this.#cgen.ftab;
      if (tab !== this.#tryTab) {
        this.#tryTab = tab;
        this.#tryFns = new Map();
      }
      const i = idx & 0x7fffffff;
      f = this.#tryFns.get(i);
      if (!f) this.#tryFns.set(i, (f = tab.get(tIdx(this.#wide, i))));
      this.#tryIdx = idx;
      this.#tryFn = f;
    }
    try {
      return f(idx < 0 ? BigInt(arg >>> 0) : arg) ?? 0;
    } catch (err) {
      if (err?.gexHandoff || !(err instanceof WebAssembly.RuntimeError) || !/memory access out of bounds/.test(err.message)) throw err;
      const e = this.#shim.exports;
      if (e.gex_handoff()) {
        err.gexHandoff = true;
        throw err;
      }
      // the shim's own (in a syscall, translating): a crash
      if (e.gex_native_oob?.() !== 1) throw err;
      return -1073741819;
    }
  }

  #nativeSyscall = (ctx, pc) => {
    const e = this.#shim.exports;
    const p = BigInt.asUintN(64, BigInt(pc));
    if (!(p & (1n << 61n))) return e.gex_syscall(ctx, pc);
    const sp0 = e.__stack_pointer.value;
    const psp0 = this.#progSpAt();
    try {
      return e.gex_syscall(ctx, pc);
    } catch (err) {
      if (err?.gexHandoff || !e.gex_handoff()) throw err;
      err.gexHandoff = true;
      if (e.gex_exited?.()) throw err;
      // win::nr::NATIVE_LEFT: the guest re-entered native code from above
      // the innermost native entry. Only when the innermost loop is a native
      // caller's call-out that the guest has left does it leave (its caller
      // drops its frames, `#callOutSteps`). Otherwise gexrt's mark is stale
      // or the frames it left were reached without a call-out (a function
      // body called straight from native code): the call returns and the
      // guest goes on nested, and gexrt's own checks drop the dead frames
      // when the guest returns past them. Trapping on would drop every
      // native frame down to the innermost loop, live ones included.
      if (e.gex_native_left?.()) {
        const c = this.#callouts.at(-1);
        const sp = this.#liveSp(Number(ctx), c);
        if (c && this.#nest.at(-1)?.startsWith('callout') && c.ret === NATIVE_RETURN_PAD && sp > c.spWant + c.unwound) {
          err.gexNativeLeft = true;
          throw err;
        }
        e.__stack_pointer.value = sp0;
        this.#progSpBack(psp0);
        e.gex_resume_pc();
        e.gex_resume_ctx();
        return;
      }
      const stop = p & ~(1n << 61n);
      // Nothing waiting: the resume the shim took, and the call returns.
      const quick = this.#afterTrap();
      const next = BigInt.asUintN(64, quick !== 0n ? quick : BigInt(e.gex_resume_pc()));
      const rc = quick !== 0n ? this.#afterTrapCtx() : e.gex_resume_ctx() >>> 0;
      if (next !== stop || (rc && rc !== (ctx >>> 0))) throw err;
      this.#progSpBack(psp0);
      if (quick !== 0n) {
        this.traps.native++;
        return;
      }
      this.traps.nativeFull++;
      if (e.gex_jit_pending_len() >>> 0) {
        const need = (e.gex_jit_pending_slots?.() ?? 0) >>> 0;
        if (need) this.#growTables(need, 'call-out');
      }
      while (e.gex_jit_pending_len() >>> 0) this.#installTranslationSync();
      this.#catchUpSync();
      this.#cgenServe();
      this.#startThread();
    }
  };

  /** Load and publish the units the shim placed (images mapped since the last
   *  trap), then catch up. A unit's bytes never pass through the shim: only
   *  its entry list does. */
  #cgenServe() {
    const e = this.#shim.exports;
    // A thread that went with its old program when another called `execve`
    // (out of guest code, so the exec did not wait for it): what is staged
    // and published now is the new program's, for that thread's Worker.
    if (e.gex_thread_gone?.()) return;
    this.#wasmprogServe();
    // ELF libraries' export providers (cgen/elfprov.rs): provider-only units,
    // bound before the guest goes on (and, in a forked child, before it
    // first runs: its table starts empty while its GOT is its parent's).
    if (e.gex_elfprov_wanted) {
      let any = false;
      for (let u = e.gex_elfprov_wanted(); u >= 0; u = e.gex_elfprov_wanted()) {
        this.#cgenProviders(u);
        any = true;
      }
      if (any) this.#cgenCatchUp();
      // Slots of provided addresses that lost their bridge (a forked child
      // instantiating a module it inherited): the bridge again.
      const n = e.gex_elfprov_rebridge_len() >>> 0;
      if (n) {
        this.#cgenCatchUp();
        const st = this.#cgenStore();
        const buf = e.gex_alloc(4 * n) >>> 0;
        let regs;
        try {
          const k = e.gex_elfprov_rebridge(buf, n) >>> 0;
          regs = new Uint32Array(this.#memory.buffer.slice(buf, buf + 4 * k));
        } finally {
          e.gex_free?.(buf, 4 * n);
        }
        const tables = this.#growTables(e.gex_elfprov_rebridge_need() >>> 0, 'elfprov');
        const blocks = this.#guest.exports.blocks;
        if (blocks) this.#growBlocks();
        this.#writeBridges(tables, blocks, regs, st.bridge);
        if (this.#bridgeCount) this.#bridgeCount.bridged += regs.length >>> 1;
      }
    }
    // A Worker with the function JIT on but no units of its own still
    // replays the function table's log.
    if (!(this.#cgenMode & 3) && e.gex_fjit_on?.()) {
      this.#cgenCatchUp();
      return;
    }
    // A miss the shim called a catch-up (`gex_ftab_behind`) traps here with
    // units off too: the log carries every installed provider.
    if (!(this.#cgenMode & 3) || !e.gex_cgen_wanted) {
      if (e.gex_ftab_behind?.()) this.#cgenCatchUp();
      return;
    }
    for (let u = e.gex_cgen_wanted(); u >= 0; u = e.gex_cgen_wanted()) {
      const t0 = performance.now();
      // The image's indirect-call profile is part of its unit's key, as
      // natively: read it from the unit cache first.
      if (e.gex_cgen_icall_path) {
        const prof = this.#env.loadFile?.(this.#cgenStr(e.gex_cgen_icall_path, u)) ?? null;
        const b = prof ? (prof instanceof Uint8Array ? prof : new Uint8Array(prof)) : null;
        const gp = b ? e.gex_alloc(Math.max(8, b.byteLength)) >>> 0 : 0;
        if (b) this.#bytes().set(b, gp);
        e.gex_cgen_set_icall(u, gp, b ? b.byteLength : 0);
        if (b) e.gex_free?.(gp, Math.max(8, b.byteLength));
      }
      const key = this.#cgenStr(e.gex_cgen_key, u);
      const path = this.#cgenStr(e.gex_cgen_path, u);
      // A link group's member: held until every
      // member is placed, then the group's module stands for them all.
      const grp = e.gex_cgen_group ? this.#cgenGroupOf(key) : null;
      if (grp) {
        e.gex_cgen_hold(u);
        grp.held.set(key, { u, path });
        if (logOn('unit')) console.warn(`gex-pe: ${path}: held for its link group`);
        if (grp.held.size === grp.members.length) this.#cgenGroup(grp, t0);
        continue;
      }
      this.#cgenUnit(u, key, path, t0);
    }
    this.#cgenCatchUp();
  }

  /** Load and publish unit `u` (`key`), then its providers and region
   *  entries. */
  #cgenUnit(u, key, path, t0) {
    const e = this.#shim.exports;
    // How long after its image was mapped (and its unit wanted) this unit's
    // serve starts, and how long its load takes (cgen.rs
    // `gex_cgen_mapped_age_us`): whether a unit arriving mid-run is placed
    // promptly.
    const lat = logOn('placelat') ? [Number(e.gex_cgen_mapped_age_us?.(u, 0) ?? -1n), Number(e.gex_cgen_mapped_age_us?.(u, 1) ?? -1n), performance.now()] : null;
    const tp = logNamed('unitstall') ? performance.now() : -1;
    const got = this.#env.loadUnit?.(key) ?? null;
    const tl = tp >= 0 ? performance.now() : -1;
    if (lat && lat[0] >= 0) {
      console.warn(`[placelat] ${path} ${(lat[0] / 1000).toFixed(1)} ms mapped, ${(lat[1] / 1000).toFixed(1)} ms wanted, load ${(performance.now() - lat[2]).toFixed(1)} ms`);
    }
    // For gex-eject's 32-bit Wine recorder: which image each unit is, and which images
    // had none; and which program this process is, at its main image.
    if (logOn('record')) console.warn(`[gex-record] unit ${key} ${got ? 'hit' : 'miss'} ${path}`);
    if (logOn('record') && e.gex_cgen_base) console.warn(`[gex-record] base ${key} 0x${BigInt.asUintN(64, BigInt(e.gex_cgen_base(u))).toString(16)}`);
    const program = /\.exe$/i.test(path) ? this.cmdline : undefined;
    if (program !== undefined && logOn('record')) console.warn(`[gex-record] process ${path} :: ${program}`);
    // The same, structured, for the page's start-up status (fetch-note.mjs).
    this.#env.noteStatus?.({ kind: 'unit', key, path, hit: !!got, program });
    if (!got) {
      if (e.gex_cgen_missing(u)) throw new Error(`gex-cgen: --cgen: ${path} has no function-tier unit (${key})`);
      this.#cgenProviders(u);
      return;
    }
    if (this.#cgenPublish(u, key, path, got) === 1) {
      // Another thread of this process took the same wanted unit first and
      // publishes it (and its providers and region entries); installed here.
      if (logOn('unit')) console.error(`gex-pe: function tier for ${path}: placed by another thread`);
      return;
    }
    this.#cgenProviders(u);
    if (logOn('unit')) console.error(`gex-pe: pid ${e.gex_my_pid?.() ?? '?'}: function tier for ${path}: ${got.entries.byteLength / 8} entries, placed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    this.#cgenRegions(u, path);
    if (tp >= 0) console.warn(`[unitpub] ${this.#whoFills()}${this.#forkChild ? ' (fork child)' : ''}: ${path.split('/').pop()} ${(performance.now() - tp).toFixed(1)} ms (load ${(tl - tp).toFixed(1)})`);
  }

  /** A unit this Worker published and left to its first call (`#cgenPublish`),
   *  by key: its files as loaded, the module's bytes fetched when wanted. */
  #unitGot = new Map();

  /** Publish unit `u`'s entries (`got`, under `key`) and install it here. A
   *  unit catch-up defers (`#cgenDefer`) is compiled at its first call, or
   *  before, by the host (`prefetchUnit`), for every Worker of the page;
   *  the rest now. */
  #cgenPublish(u, key, path, got) {
    const e = this.#shim.exports;
    if (u < 0x40000000 && !(SP_CODE && single.on) && e.gex_ftab_install_len) {
      if (!this.#sharedModules.has(`cgen:${key}`)) {
        this.#unitGot.set(key, got);
        this.#env.prefetchUnit?.(key);
      }
    } else this.#cgenModule(key, got);
    const ent = got.entries instanceof Uint8Array ? got.entries : new Uint8Array(got.entries);
    const gp = e.gex_alloc(Math.max(8, ent.byteLength)) >>> 0;
    this.#bytes().set(ent, gp);
    const r = e.gex_cgen_publish(u, gp, ent.byteLength / 8);
    e.gex_free?.(gp, Math.max(8, ent.byteLength));
    if (r < 0) throw new Error(`gex-cgen: publishing ${path}'s unit failed`);
    this.#cgenCatchUp();
    e.gex_cgen_installed(u);
    return r;
  }

  /** The unit cache's link groups (`groups/index.txt`: a group key, then its
   *  members' unit keys, per line), by member key; read once. */
  #cgenGroups = null;
  #cgenGroupOf(key) {
    if (!this.#cgenGroups) {
      this.#cgenGroups = new Map();
      const b = this.#env.loadFile?.('groups/index.txt') ?? null;
      const text = b ? new TextDecoder().decode(b) : '';
      for (const line of text.split('\n')) {
        const [gkey, ...members] = line.trim().split(/\s+/);
        if (!gkey || !members.length) continue;
        const g = { key: gkey, members, held: new Map(), done: false };
        for (const m of members) this.#cgenGroups.set(m, g);
      }
    }
    const g = this.#cgenGroups.get(key);
    return g && !g.done ? g : null;
  }

  /** Every member of `grp` is placed: publish the group's module in place
   *  of their units (or, without one, each member's own), then each
   *  member's providers and region entries. */
  #cgenGroup(grp, t0) {
    const e = this.#shim.exports;
    grp.done = true;
    const ms = grp.members.map((k) => ({ key: k, ...grp.held.get(k) }));
    const got = this.#env.loadUnit?.(`groups/${grp.key}`) ?? null;
    if (!got) {
      console.error(`gex-pe: link group ${grp.key} is not in the unit cache; its members' own units`);
      for (const m of ms) this.#cgenUnit(m.u, m.key, m.path, t0);
      return;
    }
    for (const m of ms) this.#env.noteStatus?.({ kind: 'unit', key: m.key, path: m.path, hit: true, program: /\.exe$/i.test(m.path) ? this.cmdline : undefined });
    const us = new Uint32Array(ms.map((m) => m.u));
    const kb = new TextEncoder().encode(grp.key);
    const up = e.gex_alloc(Math.max(8, us.byteLength)) >>> 0;
    const kp = e.gex_alloc(Math.max(8, kb.byteLength)) >>> 0;
    this.#bytes().set(new Uint8Array(us.buffer), up);
    this.#bytes().set(kb, kp);
    const g = e.gex_cgen_group(up, us.length, kp, kb.byteLength);
    e.gex_free?.(up, Math.max(8, us.byteLength));
    e.gex_free?.(kp, Math.max(8, kb.byteLength));
    if (g < 0) throw new Error(`gex-cgen: link group ${grp.key}: no unit`);
    const path = this.#cgenStr(e.gex_cgen_path, g);
    this.#cgenPublish(g, `groups/${grp.key}`, path, got);
    console.error(`gex-pe: ${path} (${grp.key}): ${got.entries.byteLength / 8} entries, placed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    for (const m of ms) {
      this.#cgenProviders(m.u);
      this.#cgenRegions(m.u, m.path);
    }
  }

  useGl(gl) {
    this.#gl = gl;
    this.#shim?.exports?.gex_set_gl?.(gl ? 1 : 0);
  }

  /** Say that the GL device runs D3D too (`/dev/gex-d3d`). After `useGl`,
   *  before `load`. */
  useD3d(on) {
    this.#d3d = !!on;
    this.#shim?.exports?.gex_set_d3d?.(on ? 1 : 0);
  }

  /**
   * Give the guest a Wayland compositor. `wl` is a `WlProxy` (wl-bridge.mjs),
   * this process's window into the shared scene arena the compositor worker
   * draws from. Like `useGl`, best set before `load` -- the socket is bound
   * while the filesystem is built.
   */
  useWl(wl) {
    this.#wl = wl;
    this.#shim?.exports?.gex_set_wl?.(wl ? 1 : 0);
  }

  /** Say how large the compositor's output is, before the guest runs. */
  wlOutput(w, h, scale = 1) {
    // Kept, and given again to every shim instance this process makes: a
    // fresh instance (a fork child's, an exec's) starts at the default
    // 640x480, and 32-bit Wine starts every Windows program by fork and exec.
    this.#wlOut = [w | 0, h | 0];
    this.#wlScale = Math.max(1, scale | 0);
    this.#shim?.exports?.gex_set_wl_output?.(w | 0, h | 0);
    this.#shim?.exports?.gex_set_wl_scale?.(this.#wlScale);
  }

  #wlOut = null;
  #wlScale = 1;

  /**
   * This process's name in the compositor's scene arena: its pid, which every
   * Worker of the process agrees on and no other process shares. Zero before
   * the process has one, so callers ask again rather than caching it.
   */
  wlToken() {
    return this.#shim?.exports?.gex_wl_token?.() ?? 0;
  }

  async run(entry, startCtx = 0x1000) {
    // Every switch, signal and miss leaves wasm by trapping, and V8 captures a
    // stack trace per trap that nothing here reads -- hundreds of thousands of
    // them for a program with contended locks.
    const limit = Error.stackTraceLimit;
    if (!globalThis.__gexKeepTrace && !globalThis.GEX_TRAP_STACK) Error.stackTraceLimit = 0;
    try {
      return await this.#runLoop(entry, startCtx);
    } finally {
      // An instrumented unit's hits, before a short-lived process is gone.
      if (this.#cov.size) this.covFlush();
      Error.stackTraceLimit = limit;
    }
  }

  async #runLoop(entry, startCtx = 0x1000) {
    let pc = entry;
    // Which thread's registers to enter with. The first thread unless a thread
    // in its own Worker says otherwise, or it runs the main thread's registers.
    let ctx = startCtx;
    // When the last trap came out, for GEX_WIN_STATS' time split: what the
    // glue costs between a trap and the guest again.
    // A fault for the process's hook, taken at the last trap (`#faultTake`):
    // called at the next entry.
    let hook = null;
    // Units for what booting mapped (the exe, ntdll), before the guest runs.
    this.#cgenServe();
    // A program the shim refused there (a 32-bit Wine built for another probe
    // layout, cgen/wasmprog.rs): the process has ended, and its ELF container
    // is not guest code to enter.
    if (this.#shim.exports.gex_exited?.()) return this.#shim.exports.gex_exit_code();
    // Wine's stack pointer as each entry found it (`#progSpBack`).
    let psp;
    for (;;) {
      if (this.#cov.size && performance.now() - this.#covAt > 250) {
        this.#covAt = performance.now();
        this.covFlush();
      }
      try {
        // Address zero is never a guest instruction: entering there is how a
        // lost resume address shows up, far from whatever forgot to set it.
        // An exec handed over from posix_spawn's child whose program could
        // not be loaded: the process has already exited.
        if (!pc && this.#shim.exports.gex_exited?.()) return this.#shim.exports.gex_exit_code();
        if (!pc) {
          throw new Error(
            `gex: nothing said where to resume (ctx=0x${ctx.toString(16)}). ` +
              'A trap gave the stack back without setting a resume address.',
          );
        }
        this.#shim.exports.gex_callout_leave_ok?.(0);
        psp = this.#progSpAt();
        // The fault hook first (staged at the last trap): the guest goes on
        // where it leaves the context.
        if (hook) {
          const h = hook;
          hook = null;
          try {
            if ((SP_CODE && single.on)) await this.#promisingOf(h.f)(ctx, ...h.args);
            else h.f(ctx, ...h.args);
          } finally {
            h.spG.value = h.cur;
          }
          pc = this.#faultPc(ctx);
        }
        // The single-thread build enters on a stack of the guest's own
        // (JSPI), which a blocking call suspends; see single.mjs.
        this.#cgenGrowIfShort();
        if ((SP_CODE && single.on)) {
          this.#where = 'enter';
          await this.#enterP(ctx, pc);
          this.#where = 'trap';
        }
        else this.#enterOf(ctx, pc)(ctx, pc);
        // `enter` returning means the guest gave the stack back for good,
        // normally because the last process exited.
        if (globalThis.GEX_DEBUG_EXIT) {
          console.error(`[gex] run ended: enter returned, code ${this.#shim.exports.gex_exit_code()}`);
        }
        return this.#shim.exports.gex_exit_code();
      } catch (err) {
        this.#where = 'trap';
        // First, before asking the shim anything: was this a handoff or a
        // crash? Every way out is the same `unreachable` to the engine, and
        // treating a crash as a handoff reported a crashed runtime as a clean
        // exit 0. `trap()` sets a flag; nothing else does. First because the
        // recovery below calls into the shim, whose kernel a crash may leave
        // borrowed -- a second call would bury the first error.
        // Asked once: the answer is taken with it.
        const crashed = !err?.gexHandoff && this.#shim.exports.gex_handoff && !this.#shim.exports.gex_handoff();
        // The guest's own access out of bounds (a null pointer's, at the
        // null guard, or a wild one past the end): the shim makes it the
        // guest's fault -- the fault hook, a SIGSEGV handler, or the
        // process's death by SIGSEGV -- and the trap is a handoff after all.
        // Not when the shim says the trap was its own (in a syscall, or
        // translating): that stays a crash.
        // Any other trap in guest code (a wasm32 program's call through a null
        // or wrong function pointer, a division by zero, `__builtin_trap`, a
        // stack overflow) is the signal Linux raises for it, the same way
        // (`guestTrapSignal`); a stale dispatch table only once the retry
        // below has failed.
        const tsig = crashed ? guestTrapSignal(err) : 0;
        let guestFault = crashed && ((/memory access out of bounds/.test(err?.message ?? '') &&
          (this.#shim.exports.gex_guest_oob?.(ctx) ?? 0) === 1) || this.#handlerLost(err, ctx) ||
          (tsig !== 0 && !/table index is out of bounds/.test(err?.message ?? '') &&
            (this.#shim.exports.gex_guest_trap?.(ctx, tsig) ?? 0) === 1));
        if (crashed && !guestFault && this.#vfork.length && !/table index is out of bounds/.test(err?.message ?? '')) {
          // posix_spawn's child crashed in this Worker: it ends, and the
          // thread that cloned it goes on.
          this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
          this.#progSpBack(psp);
          ({ pc, ctx } = this.#vforkLeave(err));
          continue;
        }
        crash: if (crashed && !guestFault) {
          // RECOVERABLE first: a "table index is out of bounds" during dispatch
          // is usually just this Worker's funcref table lagging the shared
          // directory. Another Worker numbered a page (bumping the process-wide
          // CLAIM_NEXT and growing ITS table) while ours had not caught up, so
          // the directory names a slot past the end of our table and
          // `return_call_indirect` trapped. Grow to the claimed high-water and
          // retry the same dispatch: the slot then holds the block (if we filled
          // it) or the miss stub (a normal miss that fills it). Only retry when
          // the grow made progress and only so many times, so a genuine over-run
          // still fails loudly. This is the concurrent-JIT table-growth race that
          // stranded threads under heavy multithread translation (gtk3-demo, and
          // the jit-storm repro). See gex_claimed_slots.
          const eR = this.#shim.exports;
          if (/table index is out of bounds/.test(err.message) && eR.gex_last_target) {
            // This thread's last target, read from its own context: the
            // shim's answer was once another thread's.
            let tgt = 0n;
            try { tgt = new DataView(this.#memory.buffer).getBigUint64(ctx + CTX_LAST_TARGET, true); } catch { tgt = 0n; }
            const grew = this.#growTablesToClaimed();
            if (grew && (globalThis.GEX_TABLE_DEBUG || (typeof process !== 'undefined' && process.env?.GEX_TABLE_DEBUG)))
              console.error(`[tblrec] caught up tables for a sibling-numbered page (retry ${this.#tableRetries})`);
            if (tgt && this.#tableRetries < 100000 && grew) {
              this.#tableRetries++;
              pc = tgt < 0n ? tgt + (1n << 64n) : tgt;
              continue;
            }
          }
          // Not a lagging table: a wasm32 program's call through a garbage
          // function pointer, the guest's SIGSEGV.
          if (tsig !== 0 && /table index is out of bounds/.test(err.message) &&
            (eR.gex_guest_trap?.(ctx, tsig) ?? 0) === 1) {
            guestFault = true;
            break crash;
          }
          // Each question in its own try: after a crash the kernel may be
          // borrowed and a second call would trap on top of the first.
          const ask = (f, how) => {
            try {
              return f ? how(f()) : '';
            } catch {
              return '';
            }
          };
          const e = this.#shim.exports;
          const at = ask(e.gex_current_pc, (pc) =>
            `, guest pc=0x${(pc < 0n ? pc + (1n << 64n) : pc).toString(16)}`);
          const lr = ask(e.gex_current_lr, (lr) =>
            `, lr=0x${(lr < 0n ? lr + (1n << 64n) : lr).toString(16)}`);
          const code = ask(e.gex_exit_code, (c) => (c ? `, exit code ${c}` : ''));
          const heap = ask(e.gex_heap_used_kib, (k) =>
            k >= 0 ? `, ${(k / 1024) | 0} MiB of shim heap in use` : '') +
            // Where the process's arena went (guest mappings, the shim's heap,
            // Worker stacks, MiB taken), and what its memory grew to: a trap
            // after the guest ran out of address space says which side spent it.
            ask(e.gex_arena_spent_mib, () =>
              `, arena taken: maps ${e.gex_arena_spent_mib(0)}, heap ${e.gex_arena_spent_mib(1)}, ` +
              `stacks ${e.gex_arena_spent_mib(2)} MiB, memory ${Math.round(this.#memory.buffer.byteLength / 1048576)} MiB`);
          // "inside" only when the syscall is still on the stack: a plain
          // static, so it answers even with the kernel borrowed.
          const within = ask(e.gex_in_syscall, (v) => v) === 1;
          // `!0` reaches JS as -1, not 0xffffffff (u32 in Rust, i32 in wasm),
          // so both sentinels are checked.
          const nr = ask(e.gex_last_syscall, (n) =>
            n === -1 || n === 0xffffffff
              ? ''
              : `, ${within ? 'inside' : 'after'} syscall ${n}`);
          if (globalThis.GEX_TRAP_STACK) console.error(err.stack ?? err);
          // Physical size of the linear memory at the trap: a "memory access out
          // of bounds" fires only past this, so it is the ceiling the faulting
          // address crossed -- a stack whose mmap did not grow the memory, or a
          // wild pointer beyond it.
          try {
            console.error(`[trap] linear memory grown to ${(this.#memory.buffer.byteLength / (1024 * 1024)) | 0} MiB (0x${this.#memory.buffer.byteLength.toString(16)})`);
          } catch {}
          // Dump every thread's register context: two with the same ctx is #126
          // (two Workers on one context). Prints via the shim; safe if borrowed.
          ask(e.gex_dump_threads, () => '');
          // The target of the last indirect branch, stored by the dispatch
          // itself. `enter` is long gone by now (dispatch ends in
          // `return_call_indirect`, which replaces frames), and the argument it
          // was called with and the saved PC both name where the guest already
          // left -- so a table-index trap can only be pinned from this.
          const target = ask(e.gex_last_target, (v) => v) || 0n;
          const where = ask(() => (target ? target : pc), (entered) => {
            if (!/table index/.test(err.message)) return '';
            const raw = BigInt(entered);
            const guest = raw < 0n ? raw + (1n << 64n) : raw;
            const page = Number(guest >> 12n);
            const dir = new DataView(this.#memory.buffer).getUint32(dispatchDir(this.#wide) + page * 4, true);
            const tables = this.#dispatchTables();
            const packed = dir >>> 0;
            const idx = packed >>> 24;
            const slot = packed & 0xffffff;
            // The page's base plus the block's offset within the page: the
            // base alone showed a slot inside the table while the branch was
            // past its end.
            const at = slot + ((Number(guest) & 0xfff) >> SLOT_SHIFT);
            const len = tables[idx]?.length ?? 0;
            // Asked, not computed: `get` settles whether the table really is
            // short (RangeError) or the trap belongs to some other operation.
            let probe;
            try {
              probe = tables[idx].get(at) ? 'holds a function' : 'is empty';
            } catch (e) {
              probe = `cannot be read: ${e.message}`;
            }
            // And when that entry is fine, the whole directory: a dispatch is
            // for the branch's target, not the pc, so the trapping entry is
            // usually not the one at `dispatch_page(pc)`. Every entry is checked
            // against the tables; a high-table entry (tables 2-7 are usually
            // length 0) is a corrupt write, a real table past its end was not
            // grown.
            let bad = 0;
            let highTable = 0;
            const sample = [];
            const dv = new DataView(this.#memory.buffer);
            for (let p = 0, n = dispatchPages(this.#wide); p < n; p++) {
              const v = dv.getUint32(dispatchDir(this.#wide) + p * 4, true) >>> 0;
              if (v === 0) continue;
              const t = v >>> 24;
              const b = v & 0xffffff;
              const cap = tables[t]?.length ?? 0;
              // The whole page, because any block in it may be branched to.
              if (t < tables.length && b + SLOTS_PER_PAGE <= cap) continue;
              bad++;
              if (t >= 2) highTable++;
              if (sample.length < 6) {
                sample.push(`page ${p} -> ${t}[${b}] (table is ${cap})`);
              }
            }
            const survey = bad
              ? `. ${bad} directory entries name a slot this process cannot ` +
                `reach, ${highTable} of them in a table above 1: ${sample.join('; ')}`
              : '. every other directory entry is reachable';
            return (
              `. the last indirect branch was to 0x${guest.toString(16)}` +
              `${target ? '' : ' (no target recorded, so this is where enter was called)'}` +
              `, page ${page} of the directory says table ${idx} slot ${slot}` +
              `, so this branch wants ${idx}[${at}] and that table is ${len} long` +
              ` (${at < len ? 'in range' : 'past the end'}, and reading it says it ${probe})` +
              survey +
              `, and this process has ${tables.length} tables of ` +
              `[${tables.map((t) => t.length).join(', ')}]` +
              `, ${this.#installs} modules installed`
            );
          });
          throw new Error(
            `the gex runtime trapped inside the shim (${this.#thrownName(err)}${at}${lr}${nr}` +
              `${code}${heap}${where}` +
              `${this.#shim.exports.gex_panic_reporting?.() ? '; it panicked again inside the panic report, whose message is lost (the first is above)' : ''}` +
              `). That is a crash in the runtime, not a guest exit; ` +
              'anything printed above is all the guest managed to do first.'
              + (globalThis.GEX_TRAP_STACK ? `\n${err.stack ?? ''}` : ''),
          );
        }
        // A trap abandons the engine's frames without running the epilogues
        // that restore the shim's shadow-stack global, leaking a few dozen
        // bytes each. Nothing is live below here, so reset it exactly -- left
        // alone it slid into the guest's stack and overwrote saved registers.
        this.#shim.exports.__stack_pointer.value = addrVal(this.#wide, this.#workerStack());
        this.#progSpBack(psp);
        const e = this.#shim.exports;
        // A trap that reached the run loop with a provider's guest call
        // still open abandoned the provider's frames.
        if ((e.gex_libprov_guest_calls?.() ?? 0) > 0) {
          if (++this.#provLeaks <= 16) {
            console.error(`[gex] a trap reached the run loop with ${e.gex_libprov_guest_calls()} provider guest call(s) open: ${err?.message ?? err}; ` +
              `resume ${BigInt.asUintN(64, BigInt(e.gex_resume_pc())).toString(16)}\n${err?.stack ?? ''}`);
          }
          e.gex_libprov_guest_calls_set(0);
        }
        const quick = this.#afterTrap();
        if (quick !== 0n) {
          this.traps.loop++;
          pc = quick;
          const nextCtx = this.#afterTrapCtx();
          if (nextCtx !== 0) ctx = nextCtx;
          this.#armCensus();
          continue;
        }
        this.traps.loopFull++;
        // A slot the shim wants written before the guest goes back to it: its
        // module is instantiated with passive segments, so the code is present
        // and simply not in the table. Filling here costs one segment; at
        // instantiation it cost every block of the object, in every process.
        const want = e.gex_fill_wanted ? Number(e.gex_fill_wanted()) : -1;
        if (want >= 0) {
          // Whether anybody answered matters: a page nobody holds yet must be
          // askable again, a page nobody ever will must stop trapping.
          const q0 = performance.now();
          const quick = this.fillSlotNow(want >>> 0);
          this.#nowFills++;
          this.#nowMs += performance.now() - q0;
          let answer = quick;
          if (answer === null || answer === undefined) {
            const t0 = performance.now();
            this.#where = `fillSlot(${want >>> 0})`;
            answer = await this.fillSlot(want >>> 0);
            this.#where = 'trap';
            this.#slowFills++;
            this.#slowMs += performance.now() - t0;
          }
          e.gex_fill_done(answer ? 1 : 0);
        }
        this.#denseCatchUp();
        // One grow for the whole queue before any install: `#installModule`
        // grows per module and `Table.grow` reallocates, so a batch of hundreds
        // with a rising need copies an ever-larger table hundreds of times.
        if (e.gex_jit_pending_len() >>> 0) {
          const need = (e.gex_jit_pending_slots?.() ?? 0) >>> 0;
          if (need) this.#growTables(need, 'a2358');
        }
        this.#where = 'install';
        while (e.gex_jit_pending_len() >>> 0) await this.#installTranslation();
        this.#where = 'adopt';
        // The parent's modules go in after this Worker's own queue, never
        // between a translation and its install: a module's page assignments
        // are the next one's starting point, so interleaving two orderings
        // corrupts the assignments.
        await this.#maybeAdopt();
        // And anything another Worker of this process translated: the pending
        // queue is one queue for the process, so whichever Worker traps next
        // drains it into its own tables and the asker never sees it. `#catchUp`
        // is a no-op comparison when there is nothing to do.
        this.#armCensus();
        this.#where = 'catchUp';
        await this.#catchUp();
        this.#cgenServe();
        // posix_spawn's child in this Worker, in place of the thread that
        // cloned it (hosts/browser/src/vfork.rs): over when it exits or
        // execs, and then that thread goes on after its `clone`.
        if (this.#vfork.length && e.gex_vfork_over()) {
          ({ pc, ctx } = this.#vforkLeave(null));
          continue;
        }
        // One a running child makes nests (vfork.rs says how deep).
        const vforkOk = e.gex_vfork_ok?.() ?? 0;
        if (vforkOk > 0) {
          const at = this.#vforkEnter();
          if (at) {
            ({ pc, ctx } = at);
            continue;
          }
        } else if (vforkOk !== -2) this.#vforkCopied(vforkOk);
        this.#where = 'drainFork';
        // A `fork` leaves a child here for the host to start. Drained before
        // the parent runs on, so the two are only ever one process apart --
        // and before the memory is touched again, because what the child gets
        // is a copy of it as `fork` left it.
        await this.#drainFork();
        this.#where = 'drainThread';
        await this.#drainThread();
        this.#where = 'resume';
        // A guest fault the shim staged for the process's hook (32-bit Wine): run
        // at the next entry, inside the trap handling below, since the hook
        // can end the thread or the process as any guest code can (64-bit Wine:
        // every system call is one, signal_wasm64.c's `ud2` entry, and a
        // thread the server stops dies inside it).
        const ft = this.#faultTake(ctx);
        // A process that has exited is not resumed, whatever the shim left in
        // the resume slot. The shim clears it in `exit_process` now, and this
        // is the same rule stated on the side that acts on it: re-entering a
        // guest whose descriptors have been released produces failures that
        // read as bugs anywhere but here.
        const next = e.gex_exited?.() ? 0n : e.gex_resume_pc();
        if (next !== 0n) {
          pc = next;
          const nextCtx = e.gex_resume_ctx() >>> 0;
          if (nextCtx !== 0) ctx = nextCtx;
          hook = ft;
          continue;
        }
        if (ft) ft.spG.value = ft.cur;
        // `exit_group` marks every guest thread exited and deliberately leaves
        // no resume PC. Do not inspect the wait state that thread had before
        // the exit: a network/futex flag can still be set, and waiting on it
        // here strands an already-dead process so its parent can never reap it.
        if (e.gex_exited?.()) return e.gex_exit_code();
        // The guest gave the stack back unfinished: every thread waits on
        // something only the page can produce. Waiting happens out here because
        // a guest holding the one wasm stack is a guest nothing can reach. The
        // loop matters: one event-loop turn is no promise the wait is over.
        if (e.gex_paused_for_net() || e.gex_paused_for_input()) {
          let woke = 0;
          let snap = this.#netSnap();
          while (e.gex_paused_for_net() || e.gex_paused_for_input()) {
            // Another Worker can execute exit_group while this one is parked.
            // The global-kernel wake it generates is not a resume request for
            // this dead thread, so observe the process flag before consuming it.
            if (e.gex_exited?.()) return e.gex_exit_code();
            // Input first, and the two are not exclusive: a guest can wait on
            // the keyboard and a socket at once.
            if (e.gex_paused_for_input()) {
              // A keystroke is an indefinite wait, but when a clock or socket
              // waits too it must end on its own so that thread can run. Losing
              // the race is free -- a late key is still delivered.
              const key = this.#env.readStdin();
              const more = e.gex_paused_for_net()
                ? await Promise.race([key, new Promise((r) => setTimeout(() => r(true), 5))])
                : await key;
              if (more === false) {
                this.endStdin();
                // An environment with no input answers instantly, so one tick
                // keeps this loop from spinning.
                await new Promise((r) => setTimeout(r, 1));
              }
            } else {
              // `gex_park` sleeps this Worker in the global kernel's memory
              // until another Worker bumps the wake counter (every syscall
              // does) or the bound passes -- a wake, not a poll. The bound
              // keeps it answerable: messages the host posts are queued while
              // asleep and the turn below runs that queue. -1 is "no global
              // kernel to sleep on", and a network wait cannot sleep at all --
              // its bytes arrive on this Worker's own event loop.
              // The single-thread build parks for the network as well: the
              // park is a promise, so the event loop delivers the bytes, and
              // the network's activity wakes it (`onActivity`). A turn per
              // waiting fiber instead spins the one thread every fiber shares.
              if ((SP_CODE && single.on) ? (await this.#parkP(20)) < 0 : (e.gex_paused_for_host() || e.gex_park(20) < 0)) {
                // A network wait runs on this Worker's event loop, so it cannot
                // sleep -- but it needs a turn, not a timer: a MessageChannel
                // turn returns to the loop in microseconds where a nested
                // `setTimeout(_, 1)` is clamped to 4ms, which was the whole
                // speed of a download. `gex_paused_for_host` is the network
                // case, so it is the condition for turning.
                const onLoop = !!e.gex_paused_for_host();
                this.#netTurns = onLoop ? this.#netTurns + 1 : 0;
                if (onLoop && e.gex_wait_bound_ms && e.gex_wait_word?.()) {
                  // Asleep until the network, another Worker or the clock.
                  this.waits.net++;
                  await this.#netWait(e, snap);
                } else
                // A nap every 1024th turn keeps an empty socket from spinning
                // the Worker hot and lets timer-driven work run. Raising it
                // from 64 did not change throughput -- the naps overlap with
                // waiting for bytes rather than serialising it.
                if (onLoop && (this.#netTurns & 1023) !== 0) {
                  this.waits.turn++;
                  await this.#turn();
                } else {
                  this.waits.nap++;
                  // See `#nap` for why this is not a bare `setTimeout`.
                  await this.#nap();
                }
              } else {
                this.waits.park++;
                await this.#turn();
              }
            }
            if (e.gex_exited?.()) return e.gex_exit_code();
            snap = this.#netSnap();
            woke = e.gex_wake();
            if (woke === 1) break;
          }
          if (e.gex_exited?.()) return e.gex_exit_code();
          if (woke !== 1) {
            // Every thread is waiting and nothing woke them. Two cases, told
            // apart by whether the guest actually exited: a process whose last
            // thread `exit`ed leaves its Worker here with nothing to wake, which
            // is the ordinary end of every forked child -- it reports the guest's
            // own code and stays quiet. A give-up (nothing exited, nothing woke)
            // is the runtime stuck; that is 125 -- outside a program's range so
            // it is not mistaken for a clean exit -- and it is said out loud.
            // (A thread that ended in a process that goes on -- a joined
            // worker thread -- is this Worker's end too, not a stall.)
            const stalled = (e.gex_exited ? !e.gex_exited() : false) && !(e.gex_ctx_exited?.(ctx) > 0);
            if (stalled) {
              // Dump every thread's wait-state + pc so a give-up says WHICH
              // thread is stuck on WHAT (futex vs socket read vs exited), not
              // just that nothing moved. Prints via the shim; safe if borrowed.
              try { e.gex_dump_threads?.(); } catch {}
              const why =
                `[gex] the run stopped and nothing could wake it ` +
                `(wake=${woke}, exited=${!stalled})`;
              this.#env.write(2, new TextEncoder().encode(`${why}\n`));
              console.error(why);
              return 125;
            }
            return e.gex_exit_code();
          }
          pc = e.gex_resume_pc();
          const wakeCtx = e.gex_resume_ctx() >>> 0;
          if (wakeCtx !== 0) ctx = wakeCtx;
          continue;
        }
        // Anything this far is an unexpected trap whose message ("memory access
        // out of bounds") says nothing about where. So attach the guest pc and
        // which library it is in.
        if (e.gex_current_pc && !String(err).includes('unreachable')) {
          const pc = e.gex_current_pc();
          let where = '';
          if (e.gex_object_at) {
            const at = e.gex_alloc(160) >>> 0;
            const n = e.gex_object_at(pc, at, 160) >>> 0;
            if (n) where = ' in ' + utf8(this.#bytes().subarray(at, at + n));
          }
          const ctx = e.gex_current_ctx ? e.gex_current_ctx() : 0n;
          // And how full the shim's heap was: it is a fixed region that cannot
          // grow, so running out surfaces only as a fault somewhere in the
          // guest. Present only with `--features heap-counters`.
          const used = e.gex_heap_used_kib?.() ?? -1;
          const total = e.gex_heap_total_kib?.() ?? -1;
          const heap =
            used >= 0 && total > 0
              ? `, heap ${(used / 1024).toFixed(0)}/${(total / 1024).toFixed(0)} MiB`
              : '';
          // And where dispatch thought that address lived: for "table index is
          // out of bounds", the slot wanted and how long that table is, which
          // the engine's message omits.
          let dispatch = '';
          if (e.gex_dispatch_slot) {
            const packed = e.gex_dispatch_slot(pc);
            const tables = this.#dispatchTables();
            const lens = tables.map((t) => t.length).join(',');
            dispatch =
              packed < 0
                ? `, no slots for this page, tables ${lens}`
                : `, table ${packed >>> 24} slot ${packed & 0xffffff}, tables ${lens}`;
          }
          err.message +=
            ` (guest pc=0x${(pc < 0n ? pc + (1n << 64n) : pc).toString(16)}${where}` +
            `, ctx=0x${ctx.toString(16)}${heap}${dispatch})`;
        }
        if (String(err).includes("unreachable")) {
          // A handoff can race a sibling wake: the first scheduling pass sees
          // no owned thread ready, then a sibling makes this Worker's thread
          // runnable before JS observes the pause mask. Do one final scheduler
          // pass before declaring the process abandoned. Without this, a
          // perfectly valid runnable thread is discarded with pc=0.
          if (!e.gex_exited?.() && e.gex_wake && e.gex_resume_pc) {
            const scheduled = e.gex_wake();
            const resume = e.gex_resume_pc();
            if (scheduled === 1 && resume !== 0n) {
              pc = resume < 0n ? resume + (1n << 64n) : resume;
              const resumeCtx = e.gex_resume_ctx?.() >>> 0;
              if (resumeCtx) ctx = resumeCtx;
              continue;
            }
            if (globalThis.GEX_REPORT_ABANDONED) {
              console.error(
                `[gex] final handoff recovery found scheduler=${scheduled}, resume=0x${
                  (resume < 0n ? resume + (1n << 64n) : resume).toString(16)
                }`,
              );
            }
          }
          // The ordinary end: the shim set an exit status and abandoned the
          // stack. `enter` never returns then, so this is the path a finished
          // program takes.
          // An abandoned translated stack is always a crash and must explain
          // itself. Keep successful exits quiet unless full debug was asked
          // for, but never hide the only useful PC/wasm-stack diagnostics.
          if ((!e.gex_exited?.() && globalThis.GEX_REPORT_ABANDONED) || globalThis.GEX_DEBUG_EXIT) {
            // With where it was, because a crash with no status set looks
            // identical to a clean finish otherwise.
            const say = (fn, f) => {
              try {
                return fn ? f(fn()) : '';
              } catch {
                return '';
              }
            };
            const where =
              say(e.gex_current_pc, (pc) => `, pc=0x${(pc < 0n ? pc + (1n << 64n) : pc).toString(16)}`) +
              say(e.gex_last_syscall, (n) => (n === 0xffffffff ? '' : `, in syscall ${n}`));
            // How much of this process's memory the shim used, which decides
            // how many processes a browser can hold.
            const heap = say(e.gex_heap_used_kib, (k) =>
              k >= 0 ? `, shim heap ${(k / 1024).toFixed(1)} MiB` : '');
            console.error(
              // `exited` or `ABANDONED`: both a finished program and a crashed
              // shim produce `unreachable` and code 0; `gex_exited` tells them
              // apart.
              `[gex] run ended: ${e.gex_exited?.() ? 'exited' : 'ABANDONED'}`
              + `, code ${e.gex_exit_code()}${where}${heap}`,
            );
            // What the global kernel has to say: it has no console, so its panic
            // hook writes the message into its own memory. `worker.mjs` reads it
            // only when the error propagates, which an abandoned stack does not.
            if (!e.gex_exited?.()) {
              try {
                const k = this.#kernel;
                const n = k?.k_panic_len?.() >>> 0;
                if (n) {
                  const at = k.k_panic_ptr() >>> 0;
                  const mem = this.#kernelSpec.memory;
                  console.error(
                    '[gex] the global kernel panicked: '
                    + utf8(new Uint8Array(mem.buffer, at, n)),
                  );
                }
              } catch {
                // A crash report must not crash.
              }
            }
            // And the file it was asking for, when it died inside `openat`.
            if (!e.gex_exited?.() && e.gex_last_open && e.gex_alloc) {
              try {
                const max = 256;
                const p = e.gex_alloc(max) >>> 0;
                const n = e.gex_last_open(p, max) >>> 0;
                const path = utf8(this.#bytes().subarray(p, p + n));
                e.gex_free?.(p, max);
                if (path) console.error(`[gex] the last path it opened: ${path}`);
              } catch {
                // A crash report must not crash.
              }
            }
            // And the wasm stack, when nobody set a status: a finished program's
            // says nothing, but a faulted shim's names the function that did it.
            if (!e.gex_exited?.()) {
              try { e.gex_dump_threads?.(); } catch {}
              try { e.gex_dump_state?.(); } catch {}
              console.error(`[gex] the stack under it:\n${err.stack ?? String(err)}`);
            }
          }
          return e.gex_exit_code();
        }
        throw err;
      }
    }
  }

  /**
   * Make every dispatch table long enough to hold `need` slots, and return
   * them. Separate from installing a module: a forked child must *reach* every
   * slot its parent's directory names (or a branch traps rather than missing)
   * without instantiating the modules that fill them. An unfilled slot holds
   * the miss stub, which is what makes growing without filling safe.
   */
  #growTables(need, who = '?') {
    this.#tnAt(`grow-${who}`);
    const e = this.#shim.exports;
    const tables = this.#dispatchTables();
    // The shim reports a running total across one slot space that spans every
    // table; each table's share is that total clamped to its window. Finite,
    // not `Infinity`: the window arithmetic multiplies this by the table index
    // and `0 * Infinity` is `NaN`. A pre-export shim has one table, so any
    // unreachable number is right.
    const per = (e.gex_slots_per_table?.() ?? 0) >>> 0 || Number.MAX_SAFE_INTEGER;
    // Which table the module's highest slot lands in, and how far into it.
    // Slot indices are global (table `slot / per`, offset `slot % per`), so a
    // slot in table 4 does not require tables 0-3 to be full.
    const top = need > 0 ? need - 1 : -1;
    const topTable = top >= 0 ? Math.floor(top / per) : -1;
    const topWants = top >= 0 ? (top % per) + 1 : 0;
    // The shim's per-table high-water *and* what the module reaches -- neither
    // alone is enough. `gex_table_need` counts only slots this shim allocated,
    // so a module handed in from JS (closure or inherited) is invisible to it;
    // the running-total fallback would instead grow lower tables to their max.
    const need_of = e.gex_table_need
      ? (i) => Math.max(e.gex_table_need(i) >>> 0, i === topTable ? topWants : 0)
      : (i) => Math.min(Math.max(need - i * per, 0), per);
    // Slot 0 of table 0 is the miss stub, what every unfilled slot must hold so
    // a branch to untranslated code reaches the host. Tables past the first
    // start empty, with no element 0 of their own.
    const stub = tables[0].get(0);
    for (let i = 0; i < tables.length; i++) {
      if (this.#denseTable(i)) continue;
      const need_i = need_of(i);
      if (tables[i].length >= need_i) continue;
      // Geometrically, not exactly: `Table.grow` reallocates and copies, so
      // growing by exactly what each module wants copies an ever-larger table
      // once per module. Doubling makes total copying linear in the final size.
      // Bounded: doubling a table of millions asks V8 for one FixedArray of
      // tens of megabytes per table per Worker, and at one game's world load
      // that request killed the renderer (V8 OOM with an 87 MB heap). A
      // million entries (8 MB) at most beyond what is needed.
      const len = tables[i].length;
      const want = Math.min(Math.max(need_i, Math.min(len * 2, len + TABLE_STEP)), per);
      this.#iGrows++;
      this.#iAdded += want - tables[i].length;
      this.#iCopied += want;
      // Big grows tallied and shipped with the fill stats -- a Worker's console
      // does not reach the page.
      if (want - tables[i].length > 100000) {
        this.#bigGrows.push(
          `${who}:t${i}:${tables[i].length}->${want}:need${need}`
            + `:ni${need_i}:tn${e.gex_table_need ? e.gex_table_need(i) >>> 0 : -1}`,
        );
      }
      try {
        growTable(tables[i], want - tables[i].length, stub, i);
      } catch (err) {
        throw new Error(
          `dispatch table ${i} would not grow from ${tables[i].length} to ${want}: ${err.message}`,
        );
      }
      this.#tallyTables(`dispatch${i} grew`);
    }
    return tables;
  }

  /**
   * Grow this Worker's funcref tables to the process-wide claimed high-water,
   * so a page a sibling Worker numbered (and grew ITS table for) can be
   * dispatched here without `return_call_indirect` running off the end. Returns
   * true if anything grew -- the run loop only retries a "table index out of
   * bounds" when this made progress, so a genuine over-run still fails loudly.
   * Cheap when nothing changed: a size read and a compare per table.
   */
  #growTablesToClaimed() {
    const e = this.#shim.exports;
    if (!e.gex_claimed_slots) return false;
    const tables = this.#dispatchTables();
    const stub = tables[0].get(0);
    let grew = this.#growBlocks();
    for (let i = 0; i < tables.length; i++) {
      if (this.#denseTable(i)) continue;
      const need = e.gex_claimed_slots(i) >>> 0;
      if (need <= tables[i].length) continue;
      try {
        growTable(tables[i], need - tables[i].length, stub, i);
        grew = true;
        this.#tallyTables(`dispatch${i} grew to the claimed`);
      } catch {
        /* a table that will not grow is reported by the caller's retry cap */
      }
    }
    return grew;
  }

  /**
   * Take the parent's modules after all, if this child turns out to be staying.
   * A child arrives with its tables grown but nothing filled, so one that execs
   * (almost all of them) pays nothing. One that keeps running its parent's code
   * meets it a miss at a time; past a few dozen misses it is plainly not about
   * to exec, and adopting the modules in one go is cheaper than faulting on.
   */
  async #maybeAdopt() {
    if (!this.#inherited.length) return;
    // An exec makes the parent's modules worthless -- their directory would
    // point at code belonging to a program that is gone -- so let go of them.
    // A generation, compared: two execs between checks still differ.
    const gen = this.#shim.exports.gex_exec_gen?.() ?? 0;
    if (gen !== this.#execGen) {
      this.#execGen = gen;
      this.#inherited = [];
      return;
    }
    if (++this.#traps < adoptAfterTraps()) return;
    const mine = this.#inherited;
    this.#inherited = [];
    for (const m of mine) await this.#adoptModule(m);
  }

  /** This Worker's call-outs in progress, innermost last: the return each
   *  waits for, the stack it comes back at, and the table entry watching
   *  for a translated return (`{ table, index, old }`). */
  #callouts = [];
  #callOutFn = null;
  /** The shim's and every module's `call_out` outside the single-thread
   *  build: one front per process (`callOutFront`). */
  #callOutFront() {
    return this.#callOutFn ??= callOutFront((ctx, target, ret, spWant) => (ctx === 0 ? 0 : this.#drive(this.#callOutSteps(ctx >>> 0, target, ret, spWant, false))));
  }
  /** What the watch stub found (`{ kind: 'return', i }` or `{ kind:
   *  'bypass', f }`), for the nested loop to act on. */
  #calloutSignal = null;
  /** Which nested loops this Worker is in, innermost last ('callout', or
   *  'callout~' for one that may suspend). */
  #nest = [];

  // --- The single-thread build (single.mjs) ---------
  //
  // Every guest thread is a fiber of one realm, its guest on a JSPI stack. A
  // blocking call suspends that stack (`k_ws_park`, below) where the default
  // build waits in `Atomics.wait`, so wherever JS calls into wasm that may
  // block it calls through `WebAssembly.promising`, and wherever wasm calls
  // JS that calls back into wasm that may block, that JS is a
  // `WebAssembly.Suspending` returning the inner call's promise.
  #enterPf = null;
  #enterPOf = null;
  /** `enter`, on a stack of its own. */
  #enterP(ctx, pc) {
    const f = this.#enterOf(ctx, pc);
    if (this.#enterPOf !== f) {
      this.#enterPf = WebAssembly.promising(f);
      this.#enterPOf = f;
    }
    return this.#enterPf(ctx, pc);
  }

  // The `enter` of a module of the process's other mode (gex_abi::ALT_PAGE: a
  // WoW64 thread's 32-bit code), which dispatches through that mode's
  // directory, and the directory it was installed against.
  #altEnter = null;
  #altEnterDir = 0n;
  /** The `enter` to run `ctx`'s thread at `pc` by: the program's, or, while
   *  its CS names the process's other mode, a module of that mode's. With
   *  none yet, the miss there translates and installs one, and traps to
   *  come back (the caller's loop resumes at `pc`). */
  #enterOf(ctx, pc) {
    const e = this.#shim.exports;
    if (!e.gex_ctx_alt?.(ctx)) return enterFront(this.#guest.exports.enter);
    const dir = BigInt(e.gex_alt_dir());
    if (this.#altEnter && this.#altEnterDir === dir) return enterFront(this.#altEnter);
    this.#altEnter = null;
    // Before any of that mode's code is translated (its first code native
    // i386 Wine's): the runtime's, which dispatches through its directory.
    if (dir && this.#cgen) return enterFront(this.#cgenAlt().enter);
    e.gex_dispatch_miss(ctx, BigInt.asIntN(64, BigInt(pc)));
    return enterFront(this.#altEnter ?? this.#guest.exports.enter);
  }
  #parkPf = null;
  /** `gex_park`, on a stack of its own (it suspends in `k_ws_park`). */
  #parkP(ms) {
    this.#parkPf ??= WebAssembly.promising(this.#shim.exports.gex_park);
    return this.#parkPf(ms);
  }
  #sysPf = null;
  /**
   * A system call made from JS that wasm called (a wasm32 program's, a
   * provider's): the shim runs on a stack of its own, so a wait inside it
   * suspends that stack, and the returned promise suspends the caller's.
   * `after` runs when it is done, before the caller goes on (a fork's image
   * is taken there). Nested inside a call-out (JS frames below), nothing can
   * suspend, so it runs as the default build's does.
   */
  #suspendingSyscall(sync, after, mark = 0n) {
    const e = this.#shim.exports;
    return new WebAssembly.Suspending((ctx, pc) => {
      if (this.#hardNested()) return sync(ctx, pc);
      this.#sysPf ??= WebAssembly.promising(e.gex_syscall);
      this.#where = 'syscall';
      const p = this.#sysPf(ctx, mark ? BigInt.asIntN(64, BigInt(pc) | mark) : pc);
      return p.then(() => {
        this.#where = 'syscall-done';
        after?.();
        // A preemption point as well: a program that only makes system
        // calls gives the other fibers a turn here, and `sched_yield` gives
        // the woken one at once (single.mjs `Sched#settle`).
        if (e.gex_last_syscall() === (globalThis.GEX_ARCH === 'x86' ? 24 : 124)) return sched.yieldNow(this.#fiber);
        return sched.maybeYield(this.#fiber);
      });
    });
  }
  /** A safepoint (every 1024 back edges of translated code): the shim's,
   *  then a turn for the other fibers if the slice is over. */
  #safepointS = null;
  #singleSafepoint(e) {
    return (this.#safepointS ??= new WebAssembly.Suspending((ctx, pc) => {
      e.gex_safepoint(ctx, pc);
      this.#where = 'safepoint';
      return this.#hardNested() ? 0 : sched.maybeYield(this.#fiber);
    }));
  }
  #watchStub = null;

  /** The function a watched table entry holds: it tells JS (`hit`) and
   *  traps, so the nested loop in `#callOut` sees why -- or, when `hit`
   *  says so (the awaited return of the innermost call-out, only tail calls
   *  since its `enter`), returns, which leaves that `enter` without a trap
   *  (each NT user-mode callback's end: a trap cost more than the callback). */
  #watch() {
    if (this.#watchStub) return this.#watchStub;
    // (module (type (func (param i32))) (type (func (param i32) (result i32)))
    //   (import "w" "hit" (func (type 1)))
    //   (func (export "c") (type 0) local.get 0 call 0 if return end unreachable))
    const bytes = new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0,
      1, 10, 2, 96, 1, 127, 0, 96, 1, 127, 1, 127,
      2, 9, 1, 1, 119, 3, 104, 105, 116, 0, 1,
      3, 2, 1, 0,
      7, 5, 1, 1, 99, 0, 1,
      10, 13, 1, 11, 0, 32, 0, 16, 0, 4, 64, 15, 11, 0, 11,
    ]);
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { w: { hit: (ctx) => this.#watchHit(ctx >>> 0) } });
    this.#watchStub = inst.exports.c;
    return this.#watchStub;
  }

  /** The innermost call-out whose return the guest is at (`ctx`'s PC and
   *  stack, by `#watchHit`'s rule), or -1. */
  #returnedAt(ctx) {
    const dv = new DataView(this.#memory.buffer);
    const pc = dv.getBigUint64(ctx + CTX_PC, true);
    const sp = dv.getBigUint64(ctx + this.#ctxSp(), true);
    for (let k = this.#callouts.length - 1; k >= 0; k--) {
      const c = this.#callouts[k];
      if (returnsAt(c.ret, pc) && sp >= c.spWant && sp <= c.spWant + c.span) return k;
    }
    return -1;
  }

  /** A dispatch reached a watched entry: the awaited return of a call-out
   *  (the innermost one waiting for it, as natively), or something else
   *  that shares the entry (another address of an x86 slot, a deeper frame
   *  returning to the same place), which runs what the entry held. */
  #watchHit(ctx) {
    const dv = new DataView(this.#memory.buffer);
    const target = dv.getBigUint64(ctx + CTX_LAST_TARGET, true);
    const sp = dv.getBigUint64(ctx + this.#ctxSp(), true);
    let i = -1;
    for (let k = this.#callouts.length - 1; k >= 0; k--) {
      const c = this.#callouts[k];
      if (returnsAt(c.ret, target) && sp >= c.spWant && sp <= c.spWant + c.span) { i = k; break; }
    }
    dv.setBigUint64(ctx + CTX_PC, target, true);
    if (i >= 0) {
      this.#calloutSignal = { kind: 'return', i };
      // The innermost call-out's return with only tail calls since its
      // loop's `enter`: the stub returns, and that `enter` with it.
      return i === this.#callouts.length - 1 && this.#shim.exports.gex_callout_may_return?.(this.#callouts[i].tok) ? 1 : 0;
    }
    const ref = this.#shim.exports.gex_dispatch_ref(target);
    const table = Number(BigInt.asIntN(64, ref) >> 32n), index = Number(BigInt.asUintN(64, ref) & 0xffffffffn);
    const w = this.#callouts.find((c) => c.w && c.w.table === table && c.w.index === index)?.w;
    this.#calloutSignal = { kind: 'bypass', f: w?.old ?? null };
    return 0;
  }

  /** The table `table` of `gex_dispatch_ref` names. */
  #refTable(table) {
    return table === TABLE_NAMES.length ? this.#guest.exports.blocks : this.#dispatchTables()[table];
  }

  /** `old` with arm `arm` watched (ENTRY_WATCH), made once per pair: a
   *  call-out arms and disarms the same return over and over. */
  #watchArms = new WeakMap();
  #watchArm(old, arm) {
    let m = this.#watchArms.get(old);
    if (!m) this.#watchArms.set(old, (m = new Map()));
    let fn = m.get(arm);
    if (!fn) {
      entryWatchMod ??= new WebAssembly.Module(ENTRY_WATCH);
      this.#watchHitFn ??= (ctx) => this.#watchHit(ctx >>> 0);
      fn = new WebAssembly.Instance(entryWatchMod, { w: { old, hit: this.#watchHitFn, arm: new WebAssembly.Global({ value: 'i32' }, arm) } }).exports.c;
      m.set(arm, fn);
    }
    return fn;
  }
  #watchHitFn = null;

  /** Whether a `gex_dispatch_ref` names an entry of this Worker's tables. A ref
   *  into `blocks` is a slot map word: its entry
   *  `index >>> 8`, its arm `index & 255`. */
  #refHas(table, index) {
    const t = this.#refTable(table);
    if (!t) return false;
    return (this.#entries && table === TABLE_NAMES.length ? index >>> 8 : index) < t.length;
  }

  /** What dispatch to a ref runs: the table's entry; in `blocks`
   *  that arm of the entry (the watch stub for a watched arm), as a function
   *  of `ctx` alone. */
  #refGet(table, index) {
    const t = this.#refTable(table);
    if (!this.#entries || table !== TABLE_NAMES.length) return t.get(index);
    const at = index >>> 8, arm = index & 255;
    const now = t.get(at);
    const w = this.#entryWatch.get(at);
    if (w && w.fn === now) {
      if (w.arms.has(arm)) return this.#watch();
      return (ctx) => w.orig(ctx, arm);
    }
    return (ctx) => now(ctx, arm);
  }

  /** Point a ref at `f`. In `blocks` only the watch stub can be
   *  put on an arm (anything else takes the watch off it): the entry is
   *  wrapped, one wrapper per watched arm, around what it held. An install
   *  that writes the entry since drops the wrappers, and `#armCallouts`
   *  puts them back. */
  #refSet(table, index, f) {
    const t = this.#refTable(table);
    if (!this.#entries || table !== TABLE_NAMES.length) {
      t.set(index, f);
      return;
    }
    const at = index >>> 8, arm = index & 255;
    const now = t.get(at);
    let w = this.#entryWatch.get(at);
    if (!w || w.fn !== now) w = { orig: now, arms: new Set(), fn: now };
    if (f === this.#watch()) w.arms.add(arm);
    else w.arms.delete(arm);
    let fn = w.orig;
    for (const a of w.arms) fn = this.#watchArm(fn, a);
    w.fn = fn;
    t.set(at, fn);
    if (w.arms.size) this.#entryWatch.set(at, w);
    else this.#entryWatch.delete(at);
  }

  /** Point every call-out's return entry at the watch stub, wherever
   *  dispatch now reaches its return (an install or a retirement may have
   *  moved or overwritten it); an untranslated return is caught by the miss
   *  path (cgen.rs `returned`). */
  #armCallouts() {
    const e = this.#shim.exports;
    const stub = this.#watch();
    for (const c of this.#callouts) {
      // In the mode the call-out was made in: the thread may be in the
      // other, and a change of mode in wasm (no trap) comes back to it with
      // no arming in between.
      const ref = e.gex_dispatch_ref_mode ? e.gex_dispatch_ref_mode(c.ret, c.alt ? 1 : 0) : e.gex_dispatch_ref(c.ret);
      const table = ref < 0n ? -1 : Number(ref >> 32n), index = ref < 0n ? -1 : Number(ref & 0xffffffffn);
      if (c.w && (c.w.table !== table || c.w.index !== index)) this.#unwatch(c);
      if (table < 0) continue;
      if (!this.#refHas(table, index)) continue;
      if (c.w) {
        const now = this.#refGet(table, index);
        if (now !== stub) c.w.old = now;
        if (now !== stub) this.#refSet(table, index, stub);
        continue;
      }
      const shared = this.#callouts.find((o) => o !== c && o.w && o.w.table === table && o.w.index === index);
      const old = shared ? shared.w.old : this.#refGet(table, index);
      c.w = { table, index, old };
      this.#refSet(table, index, stub);
    }
  }

  /** Put a call-out's watched entry back, unless another still watches it. */
  #unwatch(c) {
    const w = c.w;
    c.w = null;
    if (!w || this.#callouts.some((o) => o !== c && o.w && o.w.table === w.table && o.w.index === w.index)) return;
    if (this.#refHas(w.table, w.index) && this.#refGet(w.table, w.index) === this.#watchStub) this.#refSet(w.table, w.index, w.old);
  }

  /**
   * The call-out (cgen.rs `gex_ftab_fallback`): run the guest from `target`
   * in a nested loop on this Worker until it returns to `ret` with the stack
   * at `spWant` (or what the callee popped above it). 1 then, and the
   * function body that called the fallback carries on; 0 when it cannot
   * (the process ended, the stack unwound past its caller), with `ctx::PC`
   * where the guest is, and the body leaves as before. Synchronous: a shim
   * frame is below this one.
   */
  #callOut(ctx, target, ret, spWant) {
    return this.#drive(this.#callOutSteps(ctx, target, ret, spWant, false));
  }

  /**
   * The single-thread build's call-out: the same loop, with the guest on a
   * stack of its own (`WebAssembly.promising`), so guest code the call-out
   * runs can block -- the import that asked is a `Suspending`, and the
   * whole fiber waits.
   */
  #callOutAsync(ctx, target, ret, spWant) {
    return this.#driveAsync(this.#callOutSteps(ctx, target, ret, spWant, true));
  }

  /** Run a nested loop's steps (`['enter', ctx, pc]` / `['call', f, ...args]`)
   *  synchronously: an exception goes back into the loop at its step. */
  #drive(g) {
    let r = g.next();
    while (!r.done) {
      let v;
      try {
        const [op, a, b, c, d, x] = r.value;
        v = op === 'enter' ? this.#enterOf(a, b)(a, b) : a(b, c, d, x);
      } catch (err) {
        // A nested loop never yields to the event loop (a Wine thread's
        // whole life can be one call-out): an instrumented unit's hits go
        // from here too.
        if (this.#cov.size && performance.now() - this.#covAt > 250) {
          this.#covAt = performance.now();
          this.covFlush();
        }
        r = g.throw(err);
        continue;
      }
      r = g.next(v);
    }
    return r.value;
  }

  /** The same, each step on a JSPI stack of its own that may suspend. */
  async #driveAsync(g) {
    let r = g.next();
    while (!r.done) {
      let v;
      try {
        const [op, a, b, c, d, x] = r.value;
        v = op === 'enter' ? await this.#enterP(a, b) : await this.#promisingOf(a)(b, c, d, x);
      } catch (err) {
        r = g.throw(err);
        continue;
      }
      r = g.next(v);
    }
    return r.value;
  }

  /** `WebAssembly.promising` of a wasm function, made once per function. */
  #promising = new WeakMap();
  #promisingOf(f) {
    let p = this.#promising.get(f);
    if (!p) this.#promising.set(f, (p = WebAssembly.promising(f)));
    return p;
  }

  /** Whether a JS frame that cannot suspend is open (a nested loop run
   *  synchronously); `~` marks one that can (single.mjs). */
  #hardNested() {
    for (const n of this.#nest) if (!n.endsWith('~')) return true;
    return false;
  }

  *#callOutSteps(ctx, target, ret, spWant, suspendable, resume = null) {
    const e = this.#shim.exports;
    // A table index native code took for a guest address: at or past this
    // Worker's ICALL bound, a module another Worker installed since. In this
    // Worker's table before the entry thunk bound at that index runs.
    if (!resume && Number(target) < NATIVE_FN_TOP) this.#cgenCatchUp();
    const sp0 = e.__stack_pointer.value;
    const psp0 = this.#progSpAt();
    const me = this.#callouts.length;
    // The snapshot flavour's count of entries from the guest: an exception
    // that reaches this loop has ended every one made inside it (a trap
    // passes their own catches by), and the count goes back to this.
    const d0 = this.#snapDepth?.value ?? 0;
    // `ri`: this call-out's index in the shim's list (cgen.rs `CALLOUTS`),
    // which `gex_callout_hit`, `gex_callout_leave_ok` and
    // `gex_callout_may_return` speak in; -1 for one the shim did not open
    // (native Wine's own for a thread's loop, `gex_call_out(ctx, pc, 0, 0)`).
    // This list's index is not it.
    const ri = e.gex_callout_opened?.() ?? me;
    // `alt`: made from code in the mode its process's class is not (a WoW64
    // process's 32-bit code), whose frames are on the 32-bit stack.
    // `tok`: what this loop tells `gex_callout_leave_ok` as it calls `enter`,
    // and `#watchHit` asks `gex_callout_may_return` with: ri + 1 for one the
    // shim opened; for one it did not (an NT user-mode callback's, a guest
    // call from native Wine), this list's index + 1 with the top bit set,
    // which no shim index (a miss's `may_leave`) matches. Its watched return
    // then leaves `enter` by returning too, not by a trap through JS.
    const tok = ri >= 0 ? ri + 1 : (0x80000000 | (me + 1));
    const c = { ret: BigInt.asUintN(64, ret), spWant: BigInt.asUintN(64, spWant), w: null, ri, tok, alt: (e.gex_ctx_alt?.(ctx) ?? 0) === 1 };
    // How far above spWant its return comes back (`span`), and past which
    // the stack has left its caller (`unwound`). A guest call from native
    // Wine (its return is gex-kernel's NATIVE_RETURN_PAD) that names its
    // callee's argument bytes (cgen.rs `callout_span`): exactly those, as
    // its native caller's frames start just above them. One that names
    // none: NATIVE_UNWOUND past the arguments.
    const span = e.gex_callout_span?.() ?? -1n;
    c.span = span >= 0n ? span : CALLOUT_UNWOUND;
    c.unwound = span >= 0n ? span : c.ret === NATIVE_RETURN_PAD ? NATIVE_UNWOUND : CALLOUT_UNWOUND;
    // WoW64's stops: reached with the stack anywhere, never unwound past
    // (a spWant of 0 is never read as unwound). The shim counts them for
    // its stray-arrival rule (`gex_wow64_waits`).
    const anyStack = c.ret === WOW64_STOP_SYSCALL || c.ret === WOW64_RETURN_MARK;
    if (anyStack) {
      c.spWant = 0n;
      c.span = c.unwound = (1n << 64n) - 1n;
      e.gex_wow64_waits?.(1);
    }
    // Deep inside others, one more needs the host's stack to have room for
    // it: native Wine's guest calls hold their native frames, which no trap
    // drops, so the shim's count cannot bound them (the shim ends the
    // process, as a spent stack does).
    if (me >= CALLOUT_ROOM_DEPTH && stackRoom(CALLOUT_ROOM) === 0) e.gex_callout_stack_spent?.();
    if (me >= FRONT_WARM_DEPTH && coldFronts.length) warmFronts();
    this.#callouts.push(c);
    this.#calloutGuard();
    this.#nest.push(suspendable ? 'callout~' : 'callout');
    this.#calloutsMade++;
    if (globalThis.GEX_CALLOUT_TRACE || (globalThis.GEX_DENSE_DEBUG && ((this.#calloutsMade & (this.#calloutsMade - 1)) === 0 || this.#calloutsMade % 4096 === 0))) {
      console.error(`[callout] #${this.#calloutsMade} depth ${me}: ${BigInt.asUintN(64, target).toString(16)} returning to ${c.ret.toString(16)}`);
    }
    let pc = BigInt.asUintN(64, target);
    let next = null;
    // Whether this call-out ends at its own return (the loop gives 1): the
    // frames of the one around it, between its `enter` and the call that
    // made this one, are as they were, so its watched return may again
    // leave by returning (the finally gives its token back).
    let own = false;
    const leave = (v) => { own = v === 1; return v; };
    try {
      // A snapshot's level entered again inside it (`#snapCallOut`): the
      // guest goes on where that leaves it.
      if (resume) pc = resume();
      this.#armCallouts();
      for (;;) {
        try {
          if (next) {
            const f = next;
            next = null;
            e.gex_callout_leave_ok?.(tok);
            yield ['call', f, ctx];
          } else {
            e.gex_callout_leave_ok?.(tok);
            yield ['enter', ctx, pc];
          }
          // A watched return with only tail calls since this `enter`
          // returns instead of trapping (`#watchHit`).
          const ret = this.#calloutSignal;
          if (ret?.kind === 'return') {
            // An outer call-out's: it stays for that one's loop, which the
            // frames between (function-tier bodies, which trap on a call-out
            // that did not come back) reach through the loops between.
            this.#calloutSignal = ret.i < me ? ret : null;
            if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: watched return ${ret.i} (no trap)`);
            return leave(ret.i === me ? 1 : 0);
          }
          // A miss at a call-out's return with only tail calls since this
          // `enter` returns instead of trapping (DISPATCH_MISS_LEAVE).
          const hit = e.gex_callout_hit?.() ?? -1;
          if (hit >= 0) { if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: hit ${hit} (no trap)`); return leave(hit === ri ? 1 : 0); }
          // A return the shim left at with no call-out of its own waiting
          // there: one only this list has (native Wine's guest call, its
          // return gex-kernel's NATIVE_RETURN_PAD), whose return has no
          // translation to watch when nothing is translated at run time
          // (AOT-only: the shim's miss at the pad leaves). The watch's
          // rule (`#watchHit`) on where the guest is.
          const at = this.#returnedAt(ctx);
          if (at >= 0) {
            this.#calloutSignal = at < me ? { kind: 'return', i: at } : null;
            if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: its return ${at} at the pad (no watch)`);
            return leave(at === me ? 1 : 0);
          }
          console.error('[gex] guest code returned out of a call-out');
          return 0;
        } catch (err) {
          e.__stack_pointer.value = sp0;
          this.#progSpBack(psp0);
          if (this.#snapDepth) this.#snapDepth.value = d0;
          if (err === SNAP_NEST) throw err;
          const sig = this.#calloutSignal;
          this.#calloutSignal = null;
          // A call-out ended here was the trap's whole business: the resume
          // the miss path set at its return (`cgen::returned`) is spent, and
          // left in place it is where the run loop next goes -- the next
          // time this thread blocks with nothing else to run, it "resumed"
          // at the provider's return address (under ?fjit).
          // (Only this one's: an outer call-out's hit returns through the
          // frames between, which leave as before.)
          const ended = (v) => {
            if (v === 1) {
              e.gex_resume_pc();
              e.gex_resume_ctx();
            }
            return leave(v);
          };
          // An outer call-out's return (the stack unwound to a native caller's
          // frame, a 64-bit handler's target under a WoW64 thread's nested
          // 32-bit calls): it stays for that one's loop, as above. Taken here,
          // the loops between saw a plain trap at the return pad and left at
          // STRAY_PC, which the outer loop then ran (its int3).
          if (sig?.kind === 'return') {
            if (sig.i < me) this.#calloutSignal = sig;
            return ended(sig.i === me ? 1 : 0);
          }
          if (sig?.kind === 'bypass') {
            if (!sig.f) throw new Error('gex: a watched dispatch entry with nothing to run');
            next = sig.f;
            continue;
          }
          if (!err?.gexHandoff && !e.gex_handoff()) {
            if (/table index is out of bounds/.test(err.message)) {
              const tgt = new DataView(this.#memory.buffer).getBigUint64(Number(ctx) + CTX_LAST_TARGET, true);
              if (tgt && this.#growTablesToClaimed() && this.#tableRetries++ < 100000) {
                pc = BigInt.asUintN(64, tgt);
                continue;
              }
            }
            // The guest's own access out of bounds (a null pointer's): its
            // fault, staged for the hook below (32-bit Wine) or its SIGSEGV, as the
            // run loop does and natively's call-out loop.
            // Any other trap in guest code likewise (`guestTrapSignal`).
            const tsig = guestTrapSignal(err);
            if (!(/memory access out of bounds/.test(err?.message ?? '') && (e.gex_guest_oob?.(ctx) ?? 0) === 1) &&
              !(tsig !== 0 && (e.gex_guest_trap?.(ctx, tsig) ?? 0) === 1)) throw err;
          }
          const hit = e.gex_callout_hit();
          if (hit >= 0) { if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: hit ${hit}`); return ended(hit === ri ? 1 : 0); }
          // win::nr::NATIVE_LEFT (gexrt saw the guest leave a native caller's
          // frames): a native caller's call-out leaves, at the context's PC
          // (the entry the guest makes again); anything else carries on.
          if (err?.gexNativeLeft && c.ret === NATIVE_RETURN_PAD
              && this.#liveSp(ctx, c) > c.spWant + c.unwound) { if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: native left`); return 0; }
          // The process ended: down through every frame to the run loop
          // (returning would run the guest on).
          if (e.gex_exited?.()) {
            err.gexHandoff = true;
            throw err;
          }
          // Nothing waiting: the resume the shim took. Otherwise each step.
          const quick = this.#afterTrap();
          let faulted = null;
          let resumed = quick;
          if (quick !== 0n) this.traps.callout++;
          else {
            this.traps.calloutFull++;
            const want = e.gex_fill_wanted ? Number(e.gex_fill_wanted()) : -1;
            if (want >= 0) e.gex_fill_done(this.fillSlotNow(want >>> 0) ? 1 : 0);
            this.#denseCatchUp();
            if (e.gex_jit_pending_len() >>> 0) {
              const need = (e.gex_jit_pending_slots?.() ?? 0) >>> 0;
              if (need) this.#growTables(need, 'call-out');
            }
            while (e.gex_jit_pending_len() >>> 0) this.#installTranslationSync();
            this.#catchUpSync();
            this.#cgenServe();
            // A fork from the guest code this call-out runs (a function-tier
            // unit's libc calling `fork`): its child is started from here, as
            // the run loop starts one, since the guest goes on in this loop and
            // may wait for it. The child's image is taken before this returns.
            this.#drainFork()?.catch((err) => console.error(`[gex] starting a process: ${err.message ?? err}`));
            this.#startThread();
            // The fault hook as one of this loop's steps: on a stack that may
            // suspend in the single-thread build (`#driveAsync`). (Only here:
            // a staged fault keeps the trap off the quick path.)
            const ft = this.#faultTake(ctx);
            // Inside a fault's dispatch, which this call-out may lie within:
            // with no room on the stack for another, the process ends, as
            // its frames below cannot be left (`gex_fault_overflow`).
            if (ft && !ft.sig && e.gex_fault_nesting?.() > 0 && !stackRoom()) {
              ft.spG.value = ft.cur;
              e.gex_fault_overflow(ft.code);
              err.gexHandoff = true;
              throw err;
            }
            if (ft) {
              try {
                if (this.#snapHooks && !ft.sig) yield ['call', (c, co, ad, ki) => this.#snapHookCall(ft.f, ft.idx, c, co, ad, ki, ft.rsp), ctx, ft.code, ft.addr, ft.kind];
                else yield ['call', ft.f, ctx, ...ft.args];
              } finally {
                ft.spG.value = ft.cur;
              }
              faulted = this.#faultPc(ctx);
            }
            resumed = e.gex_resume_pc();
          }
          const nextPc = faulted ?? resumed;
          // Nowhere to resume (the thread ended, or waits): the run loop's
          // business, down through every frame.
          if (nextPc === 0n) {
            err.gexHandoff = true;
            throw err;
          }
          pc = BigInt.asUintN(64, nextPc);
          const rc = quick !== 0n ? this.#afterTrapCtx() : e.gex_resume_ctx() >>> 0;
          if (rc && rc !== ctx) {
            // A resume on another thread's context is not this call-out's.
            if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: resume on ctx ${rc.toString(16)} not ${ctx.toString(16)}`);
            return 0;
          }
          const sp = this.#liveSp(ctx, c);
          // Native Wine's return pad, reached with nobody's watch or miss
          // taking it (cgen.rs `native_return_stray`): the guest went on
          // past this call-out to a native caller's return (an exception
          // handled in the function that made it, which then returned on
          // tier-1). It leaves, as every call-out down to that caller does,
          // with the guest at STRAY_PC: the caller's `run_until` goes there,
          // and its fallback puts back the registers the guest had at the
          // pad (the bodies leaving between wrote stale ones) and ends the
          // call (a body between two call-outs that leaves sends tier-1 to
          // STRAY_PC inside the outer one, which leaves the same way). A
          // call-out waiting for the pad itself with the stack where it
          // wants it is simply done.
          if ((pc === NATIVE_RETURN_PAD || pc === STRAY_PC) && e.gex_callout_stray?.(0)) {
            const mine = pc === NATIVE_RETURN_PAD && c.ret === NATIVE_RETURN_PAD && sp >= c.spWant && sp <= c.spWant + c.unwound;
            if (mine) e.gex_callout_stray(1);
            new DataView(this.#memory.buffer).setBigUint64(ctx + CTX_PC, mine ? pc : STRAY_PC, true);
            if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: native return pad reached (sp ${sp.toString(16)}${mine ? ', its own' : ''}; waited for ${c.ret.toString(16)}, dispatch ${e.gex_dispatch_ref(c.ret).toString(16)}, watching ${c.w ? `${c.w.table}:${c.w.index}` : 'nothing'})`);
            return leave(mine ? 1 : 0);
          }
          // The stack unwound past the caller (longjmp, an exception
          // handled further out): its frame is gone, so it leaves. A call-out
          // that wants no stack (native Wine's thread loop) has no caller's
          // frame to unwind past.
          if (c.spWant !== 0n && sp > c.spWant + c.unwound) {
            if (globalThis.GEX_CALLOUT_TRACE) console.error(`[callout] depth ${me} leaves: unwound (sp ${sp.toString(16)}, resume ${pc.toString(16)})`);
            new DataView(this.#memory.buffer).setBigUint64(ctx + CTX_PC, pc, true);
            return 0;
          }
          this.#armCallouts();
        }
      }
    } finally {
      // A return signal for this call-out or a deeper one is spent with it.
      if (this.#calloutSignal?.kind === 'return' && this.#calloutSignal.i >= me) this.#calloutSignal = null;
      if (anyStack) e.gex_wow64_waits?.(-1);
      e.gex_callout_leave_ok?.(own && me > 0 ? this.#callouts[me - 1].tok : 0);
      this.#unwatch(c);
      this.#callouts.pop();
      this.#calloutGuard();
      this.#nest.pop();
      // Left by an exception, the shim's half has not popped its entry.
      e.gex_callout_unwind?.(me);
      if (globalThis.GEX_CALLOUT_TRACE) {
        const dv = new DataView(this.#memory.buffer);
        console.error(`[callout] end depth ${me}: pc ${dv.getBigUint64(ctx + CTX_PC, true).toString(16)} sp ${dv.getBigUint64(ctx + this.#ctxSp(), true).toString(16)} want ${c.spWant.toString(16)}`);
      }
    }
  }

  #calloutsMade = 0;

  /** Tell the shim where the innermost call-out's caller is unwound past
   *  (`gex_callout_guard`): a change of mode served in wasm, which never
   *  comes back to this loop, leaves it when the stack is past there, as
   *  this loop's own check would after a trap. */
  #calloutGuard() {
    const g = this.#shim.exports.gex_callout_guard;
    if (!g) return;
    const c = this.#callouts.at(-1);
    if (!c || c.spWant === 0n) g(0n, 0);
    else g(BigInt.asIntN(64, c.spWant + c.unwound), c.alt ? 1 : 0);
  }

  /** The guest stack pointer's byte offset in a thread's context, which
   *  differs by guest architecture. */
  #ctxSpAt = -1;
  #ctxSp() {
    if (this.#ctxSpAt < 0) this.#ctxSpAt = this.#shim.exports.gex_ctx_sp?.() ?? CTX_SP;
    return this.#ctxSpAt;
  }

  /** Where the hook left the context. */
  #faultPc(ctx) {
    return BigInt.asUintN(64, new DataView(this.#memory.buffer).getBigUint64(ctx + CTX_PC, true));
  }

  /** The staged fault, taken, and the stack made ready for the hook: { f,
   *  code, addr, kind, args, spG, cur } (the caller calls `f(ctx, ...args)`,
   *  args being `code, addr, kind`, and puts `spG` back to `cur`), or a
   *  staged signal frame (`#signalTake`), or null with neither staged. */
  #faultTake(ctx) {
    const e = this.#shim.exports;
    if (!e.gex_fault_take) return null;
    const buf = e.gex_alloc(16) >>> 0;
    let hook, code, addr, kind;
    try {
      hook = e.gex_fault_take(buf) >>> 0;
      if (!hook) return this.#signalTake(buf);
      const dv = new DataView(this.#memory.buffer);
      code = dv.getUint32(buf, true);
      addr = dv.getBigUint64(buf + 4, true);
      kind = dv.getUint32(buf + 12, true);
    } finally {
      e.gex_free?.(buf, 16);
    }
    const f = this.#cgenStore().ftab.get(tIdx(this.#wide, hook));
    if (!f) throw new Error(`gex: the guest fault hook (function ${hook}) is not in this Worker's table`);
    const rsp = this.#snapHooks ? this.#dv().getBigUint64(ctx + this.#ctxSp(), true) : 0n;
    const { spG, cur } = this.#hookStack(ctx);
    // 64-bit Wine's hook (signal_wasm64.c) takes the address as a 64-bit integer
    // (above 4 GiB in a 64-bit memory), 32-bit Wine's (signal_wasm32.c) as 32 bits.
    const wide = e.gex_guest_x86_64?.() === 1;
    const a = wide ? BigInt.asUintN(64, addr) : Number(BigInt.asUintN(32, addr)) | 0;
    return { f, idx: hook, rsp, code: code | 0, addr: a, kind: kind | 0, args: [code | 0, a, kind | 0], spG, cur };
  }

  /** A signal frame the shim staged for the process's signal hook (cgen.rs
   *  `signal_hook`, in the single-thread build), taken into `buf` (16
   *  bytes, `#faultTake`'s): { f, sig, args, spG, cur } as `#faultTake`'s,
   *  the hook's C frames below the stack pointer the shim named (as
   *  `signal_call` puts them), or null with none staged. */
  #signalTake(buf) {
    const e = this.#shim.exports;
    const hook = (e.gex_signal_take?.(buf) ?? 0) >>> 0;
    if (!hook) return null;
    const dv = new DataView(this.#memory.buffer);
    const pc = dv.getBigUint64(buf, true);
    const sp = dv.getBigUint64(buf + 8, true);
    const f = this.#cgenStore().ftab.get(tIdx(this.#wide, hook));
    if (!f) throw new Error(`gex: the guest signal hook (function ${hook}) is not in this Worker's table`);
    const spG = this.#wasmprogSp ?? this.#wasmprogStack();
    const cur = this.#belowGuest(spG, sp);
    return { f, idx: hook, sig: true, args: [this.#signalPc(pc)], spG, cur };
  }

  /** The interrupted pc as the signal hook takes it: 64-bit Wine's whole
   *  (a guest above 4 GiB in a 64-bit memory: cut to its low half, it never
   *  matched the pc the frames return to, and the hook ran that pc as a
   *  handler), 32-bit Wine's low half. */
  #signalPc(pc) {
    return this.#shim.exports.gex_guest_x86_64?.() === 1 ? BigInt.asUintN(64, BigInt(pc)) : Number(BigInt.asUintN(32, BigInt(pc))) | 0;
  }

  /** The stack pointer of the stack call-outs are made on, which a call-out's
   *  caller's frame is or is not unwound past: RSP, but in a WoW64 process
   *  the 64-bit stack's wherever the WoW64 CPU keeps it (R14 in 32-bit code
   *  and around its transitions: the shim's `gex_live_sp`). For call-out
   *  `c` made from 32-bit code, ESP, and -1n while the thread runs 64-bit
   *  code, which no comparison takes as unwound (`gex_callout_live_sp`). */
  #liveSp(ctx, c = null) {
    const e = this.#shim.exports;
    if (c?.alt && e.gex_callout_live_sp) return BigInt(e.gex_callout_live_sp(ctx, 1));
    if (e.gex_live_sp) return BigInt.asUintN(64, BigInt(e.gex_live_sp(ctx)));
    return new DataView(this.#memory.buffer).getBigUint64(ctx + this.#ctxSp(), true);
  }

  /** Move native Wine's C stack pointer `spG` 4 KiB below the guest's
   *  stack pointer `sp` (an i64), unless it is already lower, for a hook
   *  called in place (`fault_call`, `signal_call`): the value it had. */
  #belowGuest(spG, sp) {
    const cur = spG.value;
    if (typeof cur === 'bigint') {
      const rsp = BigInt.asUintN(64, sp);
      spG.value = ((cur !== 0n && cur < rsp ? cur : rsp) - 4096n) & ~15n;
    } else {
      const esp = Number(sp & 0xffffffffn);
      const c32 = Number(cur) >>> 0;
      spG.value = (((c32 !== 0 && c32 < esp ? c32 : esp) - 4096) & ~15) | 0;
    }
    return cur;
  }

  /** The program's stack pointer moved below the guest's for a hook's C
   *  frames: { spG, cur } (the caller puts `spG` back to `cur`). */
  #hookStack(ctx) {
    // The hook's C frames go below the guest's stack pointer, as a signal
    // frame does, with room for the exception frame it builds there.
    const spG = this.#wasmprogStack();
    const cur = spG.value;
    // In code of the process's other mode (a WoW64 process's 32-bit code)
    // the 64-bit stack, which R14 holds while the 32-bit one is in RSP (the
    // WoW64 CPU's convention): the 32-bit stack may be nearly full, and a
    // fault there is dispatched on the 64-bit one, a little below R14 (the
    // native shim's WOW64_DISPATCH_ROOM, 16 KiB, keeps these frames under it).
    const alt = !!this.#shim.exports.gex_ctx_alt?.(ctx);
    const at = ctx + this.#ctxSp() + (alt ? 10 * 8 : 0);
    const room = alt ? 0x4000 : 0;
    // An i64 stack pointer (a 64-bit memory, or a lowered LP64 program)
    // takes the guest's whole 64-bit one: a thread stack above 4 GiB cut to
    // its low 32 bits put the hook's frames, and the exception record it
    // writes, over whatever lives there, the runtime's own heap included.
    if (typeof cur === 'bigint') {
      const rsp = this.#dv().getBigUint64(at, true) - BigInt(room);
      const low = cur !== 0n && cur < rsp ? cur : rsp;
      spG.value = (low - 4096n) & ~15n;
    } else {
      const esp = (this.#dv().getUint32(at, true) - room) >>> 0;
      const c32 = Number(cur) >>> 0;
      // 0: the program has not set it yet, as for an i64 one.
      const low = c32 !== 0 && c32 < esp ? c32 : esp;
      spG.value = ((low - 4096) & ~15) | 0;
    }
    return { spG, cur };
  }

  /** What a dispatch miss needs, installed synchronously on this Worker
   *  (lib.rs `in_place`): 1, or 0 when it cannot be done here (a page's
   *  main thread may not compile synchronously) and the miss traps out. */
  #jitInstall() {
    if (typeof window !== 'undefined') return 0;
    const e = this.#shim.exports;
    try {
      this.#denseCatchUp();
      if (e.gex_jit_pending_len() >>> 0) {
        const need = (e.gex_jit_pending_slots?.() ?? 0) >>> 0;
        if (need) this.#growTables(need, 'in-place');
      }
      while (e.gex_jit_pending_len() >>> 0) this.#installTranslationSync();
      this.#catchUpSync();
      this.#armCallouts();
      return 1;
    } catch (err) {
      console.error(`[gex] an in-place install failed, trapping instead: ${err.message ?? err}`);
      return 0;
    }
  }

  /**
   * The process's JS-side module store: a growable SharedArrayBuffer every
   * Worker of the process holds, where the registry keeps a joining Worker's
   * catch-up bytes instead of in the shim's heap (which shares a fixed 2 GiB
   * arena; the registry alone reached 980 MiB in a long fjit run). Null where
   * there is no growable SharedArrayBuffer; the registry then keeps the bytes
   * itself, as before.
   */
  #moduleStore = null;

  /** The module store of the program running now, made if `make` and there
   *  is none; null otherwise. A store outlives an exec in its Worker, and the
   *  registry does not (`reset_registry`): the old program's modules in it
   *  are no one's, and appending the new program's after them kept both, so
   *  every program of a threaded exec chain paid for all before it. A store
   *  of an earlier registry epoch is let go of here, and a new one made for
   *  the threads the new program starts. No row names the old one: the
   *  registry started over without any, and only threads add them. */
  #programStore(make) {
    const e = this.#shim.exports;
    const epoch = e.gex_registry_epoch?.() ?? 0;
    if (this.#moduleStore && new Int32Array(this.#moduleStore, 8, 1)[0] !== epoch) {
      this.#moduleStore = null;
      // This Worker's place in the registry is the old one's: what is there
      // now this Worker installed itself.
      this.#installed = e.gex_module_count?.() >>> 0;
      this.#rowMemo = null;
    }
    if (make) this.#moduleStore ??= makeModuleStore(epoch);
    return this.#moduleStore;
  }

  /** Registry module `i`'s bytes (a copy), wherever they are kept; null if none. */
  /** Registry module `i`'s content key: as the store's writer recorded it
   *  when the module is in the store, so a catching-up Worker does not hash
   *  bytes a sibling already hashed; computed from `bytes` otherwise. */
  #registryKey(i, bytes) {
    const r = this.#registryRow(i);
    // The shim keys every row as it publishes it (`content_key`, the same
    // function); hashing here is for a shim that keys only the store's.
    if (r.ext || r.keyHi || r.keyLo) return `${bytes.length}:${r.keyHi}:${r.keyLo}`;
    return moduleContentKey(bytes);
  }

  /** Registry module `i`'s row: where its bytes are (`ptr`, or `ext`, the
   *  module store's offset plus one), how many, and the bytes' content key.
   *  One call and a read of the row (`gex_module_row`), which the shim
   *  publishes for readers holding no lock; the getters one field at a time
   *  for a shim without it. Remembered for the last `i` asked, since a
   *  catch-up asks about each module several times running. */
  #registryRow(i) {
    if (this.#rowMemo?.i === i) return this.#rowMemo;
    const e = this.#shim.exports;
    let r;
    const at = e.gex_module_row ? e.gex_module_row(i) >>> 0 : -1;
    if (at > 0) {
      const v = new DataView(this.#bytes().buffer, at, 24);
      r = { i, ptr: v.getUint32(0, true), len: v.getUint32(4, true), ext: Number(v.getBigUint64(8, true)),
        keyHi: v.getUint32(20, true), keyLo: v.getUint32(16, true) };
    } else if (at === 0) {
      // Not published yet: not remembered either.
      return { i, ptr: 0, len: 0, ext: 0, keyHi: 0, keyLo: 0 };
    } else {
      const ext = e.gex_module_ext ? Number(e.gex_module_ext(i)) : 0;
      r = { i, ptr: e.gex_module_ptr(i) >>> 0, len: e.gex_module_len(i) >>> 0, ext,
        keyHi: e.gex_module_key ? e.gex_module_key(i, 0) >>> 0 : 0,
        keyLo: e.gex_module_key ? e.gex_module_key(i, 1) >>> 0 : 0 };
    }
    this.#rowMemo = r;
    return r;
  }
  #rowMemo = null;

  /** With `scratch`, the copy is a view of this Worker's one scratch buffer,
   * good until the next such call: for a caller that
   *  compiles and installs it without yielding. */
  #registryBytes(i, scratch = false) {
    const { ptr, len, ext } = this.#registryRow(i);
    if (!len) return null;
    let from;
    if (ext) {
      if (!this.#moduleStore) throw new Error(`gex: registry module ${i} is in the module store, and this Worker has none`);
      from = new Uint8Array(this.#moduleStore, ext - 1, len);
    } else {
      from = this.#bytes().subarray(ptr, ptr + len);
    }
    if (!scratch) return from.slice();
    if (!this.#scratch || this.#scratch.byteLength < len) {
      this.#scratch = new ArrayBuffer(Math.max(1 << 20, 2 ** Math.ceil(Math.log2(len))));
    }
    const view = new Uint8Array(this.#scratch, 0, len);
    view.set(from);
    return view;
  }
  /** Where a synchronous install's module bytes are copied out of the
   *  registry. A copy per module would be garbage that a Worker running
   *  translated code may not collect for a long time (it allocates little
   *  JavaScript), one copy of every module its thread ran; this holds the
   *  largest one. */
  #scratch = null;

  /** Before `gex_jit_installed`: when the registry will keep this module, put
   *  its bytes in the module store and tell the shim where, so it keeps the
   *  place instead of the bytes. */
  #noteExternal(bytes, key) {
    const e = this.#shim.exports;
    if (!key || !e.gex_jit_note_external || !e.gex_jit_keeps?.() || !this.#programStore(false)) return;
    const off = storeAppend(this.#moduleStore, bytes);
    if (off < 0) return;
    // Its size now and then (every 64 MiB; every MiB under
    // GEX_REGISTRY_TRACE), beside what the registry still holds in the heap.
    const trace = globalThis.GEX_REGISTRY_TRACE || globalThis.process?.env?.GEX_REGISTRY_TRACE;
    const step = trace ? 20 : 26;
    if (Math.floor((off + bytes.length) / 2 ** step) !== Math.floor(off / 2 ** step)) {
      const line = `[gex] module store at ${((off + bytes.length) / 1048576).toFixed(0)} MiB; registry bytes in the shim heap: ${e.gex_registry_mib?.(0) ?? '?'} MiB`;
      // Traced under Node, straight to fd 2: a Worker's console output can be
      // lost when the process exits soon after.
      const fs = trace ? globalThis.process?.getBuiltinModule?.('fs') : null;
      if (fs) fs.writeSync(2, `${line}\n`);
      // A size report, not an error (a console error fails a page test).
      else narrate(line);
    }
    const [, a, b] = key.split(':');
    e.gex_jit_note_external(BigInt(off), bytes.length, Number(a) | 0, Number(b) | 0);
  }

  /** `#installTranslation` without yielding, for the nested loops. */
  #installTranslationSync() {
    const e = this.#shim.exports;
    // The queue's front now, not whenever this Worker last pushed or popped:
    // a module a sibling installed since may have been freed.
    e.gex_jit_pending_refresh?.();
    const ptr = e.gex_jit_pending_ptr() >>> 0;
    const len = e.gex_jit_pending_len() >>> 0;
    if (len === 0) return;
    const bytes = this.#bytes().slice(ptr, ptr + len);
    const [key, ukey] = this.#pendingKeys(bytes);
    this.#installModule(bytes, e.gex_jit_slots() >>> 0, 64, null, 'nt-callback', this.#unitGet(ukey), true);
    this.#unitShare(ukey);
    this.#noteExternal(bytes, key);
    e.gex_jit_installed();
    const count = e.gex_module_count ? e.gex_module_count() >>> 0 : this.#installed;
    this.#installed = count;
    if (count) {
      const mod = this.#compiled[this.#compiled.length - 1]?.mod;
      if (mod && key && !this.#sharedModules.has(key)) {
        this.#sharedModules.set(key, mod);
        this.#deadKeys.delete(key);
        this.#shareRegistry(count - 1, key, mod);
      }
    }
  }

  /**
   * With the page's unit cache (`?unitcache=1`, GEX_UNIT_CACHE): a
   * translation's key in it, `u:` and its content key with the dense base
   * word zeroed -- the one part of a unit's bytes that differs between the
   * processes it is installed in (gex_emit::dense::GLOBAL) -- else null.
   * `bytes` is a private copy; the word is put back.
   */
  #unitKey(bytes, always = false) {
    if (!(always || globalThis.GEX_UNIT_CACHE) || !(bytes instanceof Uint8Array)) return null;
    const at = denseBaseAt(bytes);
    if (at < 0) return `u:${moduleContentKey(bytes)}`;
    const w = bytes.slice(at, at + 4);
    bytes.fill(0, at, at + 4);
    const k = moduleContentKey(bytes);
    bytes.set(w, at);
    return `u:${k}`;
  }

  /** The pending module's content key and `#unitKey`, for its `bytes`: as
   *  the shim keyed it when it queued the module (`gex_jit_pending_key`), so
   *  no Worker hashes the bytes. GEX_CHECK_INSTALL hashes them too and says
   *  where the two differ. */
  #pendingKeys(bytes) {
    const k = this.#shim.exports.gex_jit_pending_key;
    const len = bytes.length;
    const key = `${len}:${k(0) >>> 0}:${k(1) >>> 0}`;
    const ukey = `u:${len}:${k(2) >>> 0}:${k(3) >>> 0}`;
    if (globalThis.GEX_CHECK_INSTALL || globalThis.process?.env?.GEX_CHECK_INSTALL) {
      const want = moduleContentKey(bytes);
      const uwant = this.#unitKey(bytes, true);
      const pid = this.#shim.exports.gex_my_pid?.() ?? '?';
      if (key !== want || ukey !== uwant) console.error(`[install-check] pid ${pid}: pending keys ${key} ${ukey}, the bytes' ${want} ${uwant}`);
      else if (!this.#keysChecked++) console.error(`[install-check] pid ${pid}: the pending keys match the bytes' (${ukey === `u:${key}` ? 'no dense section' : 'dense base word zeroed'})`);
    }
    return [key, globalThis.GEX_UNIT_CACHE ? ukey : null];
  }
  #keysChecked = 0;

  /** The compiled module for `ukey`, counted (`unitHits`), or null. */
  #unitGet(ukey) {
    const m = (ukey && this.#sharedModules.get(ukey)) || null;
    if (m) this.#unitHits++;
    return m;
  }
  #unitHits = 0;
  get unitHits() { return this.#unitHits; }
  /** Modules this Worker compiled from bytes rather than took compiled: the
   *  engine may still find the code of the same bytes compiled elsewhere. */
  #compiles = 0;
  get compileCount() { return this.#compiles; }

  /** The module just installed, to the page's unit cache under `ukey`. */
  #unitShare(ukey) {
    if (!ukey || this.#sharedModules.has(ukey)) return;
    const mod = this.#compiled[this.#compiled.length - 1]?.mod;
    if (!mod) return;
    this.#sharedModules.set(ukey, mod);
    this.#deadKeys.delete(ukey);
    this.#offerModule(ukey, mod, Number(ukey.split(':')[1]) || 0);
  }

  /** `#catchUp` (eager) without yielding, for the nested loops. */
  #catchUpSync() {
    const e = this.#shim.exports;
    if (!e.gex_module_count) return;
    const have = e.gex_module_count() >>> 0;
    const need = (e.gex_jit_slots_total?.() ?? 0) >>> 0;
    for (; this.#installed < have; this.#installed++) {
      const i = this.#installed;
      if (this.#leaveDense(i)) continue;
      const bytes = this.#registryBytes(i, true);
      if (!bytes) continue;
      const key = this.#registryKey(i, bytes);
      const cached = key ? this.#sharedModules.get(key) : undefined;
      this.#installModule(bytes, need, 64, null, 'nt-catchup', cached, true);
      if (cached) continue;
      const mod = this.#compiled[this.#compiled.length - 1]?.mod;
      if (mod && key) {
        this.#sharedModules.set(key, mod);
        this.#deadKeys.delete(key);
        this.#shareRegistry(i, key, mod);
      }
    }
  }

  async #installTranslation() {
    const e = this.#shim.exports;
    // The queue's front now, not whenever this Worker last pushed or popped:
    // a module a sibling installed since may have been freed.
    e.gex_jit_pending_refresh?.();
    const ptr = e.gex_jit_pending_ptr() >>> 0;
    const len = e.gex_jit_pending_len() >>> 0;
    // A zero-length published view means the shared pending queue was drained
    // (by a sibling Worker) between the caller's `gex_jit_pending_len()` guard
    // and this read -- the queue is process-wide, the view is per-Worker.
    // Compiling a zero-length slice traps ("BufferSource argument is empty" --
    // Qt's heavy multi-thread JIT hit exactly this); there is nothing to install,
    // so resync the view and stop rather than crash.
    if (len === 0) {
      if (globalThis.GEX_JIT_TRACE) console.error('[jit] empty pending view (sibling drained the queue), skipping compile');
      return;
    }
    // GEX_CHECK_INSTALL: once the pending queue is drained, is the slot the
    // missed pc dispatches through written? A unit whose blocks land elsewhere
    // makes the resume run whatever was there before -- a wild branch with no
    // trace. Checked only when nothing is left to install: the queue is FIFO
    // and a served module ahead of the unit legitimately lacks the block.
    const check = globalThis.GEX_CHECK_INSTALL || (typeof process !== 'undefined' && process.env?.GEX_CHECK_INSTALL);
    const bytes = this.#bytes().slice(ptr, ptr + len);
    // A module another process compiled (a served unit) comes from the
    // page's unit cache: instantiated, not compiled again.
    const [key, ukey] = this.#pendingKeys(bytes);
    await this.#installModule(bytes, e.gex_jit_slots() >>> 0, 64, null, 'jit2878', this.#unitGet(ukey));
    this.#unitShare(ukey);
    this.#noteExternal(bytes, key);
    e.gex_jit_installed();
    if (check && e.gex_miss_pc && e.gex_dispatch_slot && (e.gex_jit_pending_len() >>> 0) === 0) {
      try {
        const raw = e.gex_miss_pc();
        const mpc = raw < 0n ? raw + (1n << 64n) : raw;
        const packed = e.gex_dispatch_slot(mpc);
        const tabs = this.#dispatchTables();
        const stub = tabs[0].get(0);
        let what;
        if (packed < 0) what = 'no page-table entry';
        else {
          const t = packed >>> 24, sl = packed & 0xffffff;
          const f = sl < (tabs[t]?.length ?? 0) ? tabs[t].get(sl) : 'past-end';
          what = f === stub ? `STUB at table ${t} slot ${sl} (len ${tabs[t]?.length})` : f === 'past-end' ? `PAST END table ${t} slot ${sl} len ${tabs[t]?.length}` : null;
        }
        const dv = new DataView(this.#memory.buffer);
        const dir = dv.getUint32(dispatchDir(this.#wide) + Number(mpc >> 12n) * 4, true) >>> 0;
        if (what) console.error(`[install-check] pid ${e.gex_my_pid?.() ?? '?'} miss pc 0x${mpc.toString(16)}: ${what}; dir entry t${dir >>> 24} s${dir & 0xffffff}, pt ${packed >= 0 ? `t${packed >>> 24} s${(packed & 0xffffff) - Number(mpc & 0xfffn)}` : '-'}`);
      } catch (err) { console.error(`[install-check] failed: ${err}`); }
    }
    // This Worker translated it, so the entry `gex_jit_installed` just added is
    // the newest here and must not be replayed on top of itself.
    const count = e.gex_module_count ? e.gex_module_count() >>> 0 : this.#installed;
    this.#installed = count;
    // Hand the module this Worker just translated to the sibling cache -- the
    // main thread translates almost everything, and a thread that spawns later
    // then reuses it (from its start message) instead of recompiling the whole
    // registry. Its index is the newest entry.
    if (count) {
      const mod = this.#compiled[this.#compiled.length - 1]?.mod;
      if (mod && key && !this.#sharedModules.has(key)) {
        this.#sharedModules.set(key, mod);
        this.#deadKeys.delete(key);
        this.#shareRegistry(count - 1, key, mod);
      }
    }
  }

  /**
   * Put one translated module's blocks into this Worker's dispatch tables.
   * One function for the two ways a module arrives (this Worker translated it,
   * or another did and this one catches up): a thread that installed them
   * differently would dispatch to different code.
   */
  #installModule(bytes, need, segments = 64, meta = null, who = '?', precompiled = null, sync = false, dense = null) {
    if (!sync) return this.#installModuleAsync(bytes, need, segments, meta, who, precompiled, dense);
    const t0 = performance.now();
    this.#installs++;
    // Timed here: the inner call has done all its work when it returns,
    // and a `finally` after an `await` would run only once the guest
    // next yields, counting everything it ran meanwhile (a node start-up
    // reported hours of "installing").
    // Not an async function: one returning the inner promise queues a
    // microtask to adopt it, which a nested loop (that never reaches the
    // event loop) left queued per install, holding the install's result.
    const done = this.#installModuleInner(bytes, need, segments, meta, who, precompiled, sync, dense);
    this.#installMs += performance.now() - t0;
    return done;
  }

  async #installModuleAsync(bytes, need, segments, meta, who, precompiled, dense) {
    const t0 = performance.now();
    this.#installs++;
    try {
      return await this.#installModuleInner(bytes, need, segments, meta, who, precompiled, false, dense);
    } finally {
      this.#installMs += performance.now() - t0;
    }
  }

  /**
   * The install itself, timed by the wrapper above -- because the counter the
   * project steers by (`millis translating`) measures only lifting, which
   * turned out not to be what a run costs: each process installs thousands of
   * shared modules, and compiling and instantiating those was uncounted.
   */
  async #installModuleInner(bytes, need, segments = 64, meta = null, who = '?', precompiled = null, sync = false, dense = null) {
    const t0 = performance.now();
    const e = this.#shim.exports;
    const P = [performance.now()];
    this.#reapDead();
    P.push(performance.now());
    const tg = performance.now();
    const tables = this.#growTables(need, 'install');
    this.#growMs += performance.now() - tg;
    this.#tnAt('pre-inst');
    const tc = performance.now();
    // Nothing to install: a zero-length buffer is not a module, and
    // `WebAssembly.compile` on one traps ("BufferSource argument is empty").
    // It reaches here when a Worker's view of the shared pending queue, or a
    // registry slot, is drained/unwritten under concurrency -- an empty module
    // has no blocks, so skipping it is exactly correct. Logged (with `who`) so
    // the racing producer can be found.
    if (!(precompiled instanceof WebAssembly.Module)
        && !(bytes instanceof WebAssembly.Module)
        && (bytes?.byteLength ?? 0) === 0) {
      if (globalThis.GEX_JIT_TRACE) console.error(`[jit] empty module bytes, skipping install (who=${who})`);
      return null;
    }
    // A `precompiled` Module (a sibling Worker's, relayed by the host) skips the
    // compile entirely; otherwise compile the bytes (or use them if already a
    // Module). Same registry bytes either way, so the Module is interchangeable.
    let mod;
    try {
      mod = precompiled instanceof WebAssembly.Module
        ? precompiled
        : bytes instanceof WebAssembly.Module ? bytes
          : sync ? new WebAssembly.Module(bytes) : await WebAssembly.compile(bytes);
    } catch (err) {
      // The engine refused a translation: hand its bytes to whoever serves
      // the page (a test harness may keep them), synchronously, since the
      // Worker may be about to die. Best effort.
      if (bytes instanceof Uint8Array && err instanceof WebAssembly.CompileError) {
        // Under Node, `GEX_DUMP_BAD_WASM=DIR` keeps it, as gexc does natively.
        const dir = globalThis.process?.env?.GEX_DUMP_BAD_WASM;
        if (dir) try { globalThis.process.getBuiltinModule('fs').writeFileSync(`${dir}/bad-${who}-${Date.now()}.wasm`, bytes); } catch {}
        try {
          const x = new XMLHttpRequest();
          x.open('POST', new URL(`/__gex_dump?name=${encodeURIComponent(who)}`, globalThis.GEX_BASE ?? globalThis.location?.href).href, false);
          x.send(bytes.slice());
        } catch {}
      }
      throw err;
    }
    this.#compileMs += performance.now() - tc;
    if (mod !== precompiled && mod !== bytes) {
      this.#compiles++;
      // `?filltime=1`: what a big module cost to compile here, not handed one.
      if (globalThis.GEX_FILL_TIME && bytes?.byteLength >= 1 << 20) {
        console.warn(`[gex] ${this.#whoFills()}: compiled ${(bytes.byteLength / 1048576).toFixed(1)} MiB here in ${(performance.now() - tc).toFixed(1)} ms`);
      }
    }
    // Under Node, `GEX_DUMP_WASM=DIR` keeps every unit compiled, for
    // looking at what a translation is made of.
    if (bytes instanceof Uint8Array && globalThis.process?.env?.GEX_DUMP_WASM) {
      try { globalThis.process.getBuiltinModule('fs').writeFileSync(`${globalThis.process.env.GEX_DUMP_WASM}/u-${this.#shim.exports.gex_my_pid?.() ?? 0}-${moduleContentKey(bytes).replaceAll(':', '-')}.wasm`, bytes); } catch {}
    }
    // With its metadata: a fork hands the child `#compiled`, and without the
    // metadata the child cannot tell which module covers which page and must
    // instantiate all of them. The section is in the bytes; keeping it is a
    // parse.
    // And its dense base as THIS process numbered it (`gex.dense` from its
    // bytes, or what it was handed): with the page's unit cache, `mod` may be
    // another process's compile, whose own section names that process's base,
    // and a fork child instantiating from `mod` would put the blocks there.
    const ds = dense ?? denseSection(bytes instanceof Uint8Array ? bytes : mod);
    this.#compiled.push({
      mod,
      need,
      meta: bytes instanceof Uint8Array ? metadataSection(bytes) : undefined,
      dense: ds ?? undefined,
    });
    const imports = { ...this.#imports };
    for (let i = 0; i < tables.length; i++) imports[TABLE_NAMES[i]] = tables[i];
    // A runtime module imports its dense base (gex_emit::dense::GLOBAL)
    // whether or not it has dense blocks; set below when it has.
    imports.dense_base = 0;
    // The other mode's directory (gex_abi::ALT_PAGE: a WoW64 thread's 32-bit
    // code), which only a module of that mode imports; 0 until the process
    // has one, and then no module needs it.
    // Above 4 GiB in a wasm64 process: the global is address-typed.
    const altDir = BigInt(e.gex_alt_dir?.() ?? 0);
    if (altDir) imports.alt_dir = this.#wide ? new WebAssembly.Global({ value: 'i64' }, altDir) : new WebAssembly.Global({ value: 'i32' }, Number(altDir) | 0);
    // Dense: the module's blocks go to `blocks` at the base its bytes name,
    // within what the process has numbered.
    if (this.#guest?.exports?.blocks) {
      imports.blocks = this.#guest.exports.blocks;
      this.#growBlocks();
      if (ds) {
        const dv = new DataView(ds.buffer, ds.byteOffset, ds.byteLength);
        const base = dv.getUint32(0, true);
        if (globalThis.GEX_DENSE_DEBUG) {
          const theirs = bytes instanceof Uint8Array ? denseSection(mod) : null;
          const tb = theirs ? new DataView(theirs.buffer, theirs.byteOffset).getUint32(0, true) : base;
          console.error(`[dense] pid ${e.gex_my_pid?.() ?? '?'} ${who} base=${base} n=${dv.getUint32(4, true) & 0x7fffffff} blocks=${imports.blocks.length} size=${e.gex_dense_size()}`
            + (tb !== base ? ` (a module compiled elsewhere, at base ${tb})` : ''));
        }
        // Base 0 is a module nobody numbered: its blocks would land on the
        // miss stub and its neighbours.
        if (base === 0) throw new Error(`gex: a translation with unnumbered dense blocks reached the install (${who})`);
        // Where a module places its dense segment by import rather than by a
        // constant in its bytes (gex_emit::dense::GLOBAL).
        imports.dense_base = base;
      }
    }
    // What this module's own entries need per table, where known. `need` alone
    // is one running total that cannot say which tables absolute segments fall
    // in; the segments (or `Metadata` for an already-compiled module) can, which
    // avoids the strict-grow retry below.
    P.push(performance.now());
    const own = meta ?? (bytes instanceof Uint8Array ? metadataSection(bytes) : null);
    P.push(performance.now());
    const seg = elementTableNeeds(bytes);
    P.push(performance.now());
    const needs = seg.length ? seg : own ? metaTableNeeds(own) : [];
    for (let t = 0; t < needs.length; t++) if (needs[t]) this.#reach(t, needs[t]);
    let inst;
    const ti = performance.now();
    try {
      // Synchronously: `mod` is already compiled, so this is legal and does not
      // yield -- an `await` here goes to the back of the Worker's task queue.
      inst = new WebAssembly.Instance(mod, { gex: imports });
      this.#instantiateMs += performance.now() - ti;
    } catch (first) {
      // Grow to everything the module could need and retry once. `need` read
      // loosely (only the table the total lands in) is right for a module whose
      // segments live in one table, nearly all of them; the strict reading
      // (lower tables full) is expensive, so pay it only when the engine says
      // "table index is out of bounds".
      const per = (e.gex_slots_per_table?.() ?? 0) >>> 0 || Number.MAX_SAFE_INTEGER;
      const stub = tables[0].get(0);
      let grew = 0;
      for (let i = 0; i < tables.length; i++) {
        if (this.#denseTable(i)) continue;
        const want = Math.min(Math.max(need - i * per, 0), per);
        if (tables[i].length < want) {
          grew += want - tables[i].length;
          growTable(tables[i], want - tables[i].length, stub, i);
        }
      }
      try {
        inst = new WebAssembly.Instance(mod, { gex: imports });
        this.#instantiateMs += performance.now() - ti;
        this.#strictGrows++;
        this.#strictSlots += grew;
        this.#bigGrows.push(`RETRY:${who}:need${need}:own${own ? own.length : -1}:needs[${needs.join(',')}]:+${grew}`);
      } catch (err) {
        throw new Error(
          `installing a module of ${need} slots failed: ${err.message}. ` +
            `tables are [${tables.map((t) => t.length).join(', ')}], ` +
            `${(e.gex_slots_per_table?.() ?? 0) >>> 0} slots per table, ` +
            `${this.#installs} modules in so far, ` +
            `${grew} slots grown on retry after: ${first.message}`,
        );
      }
    }
    // A module of the other mode: its `enter` is where this thread's loops go
    // in while its CS names that mode (`#enterOf`).
    if (altDir && this.#altEnterDir !== altDir && WebAssembly.Module.imports(mod).some((i) => i.name === 'alt_dir')) {
      this.#altEnter = inst.exports.enter;
      this.#altEnterDir = altDir;
    }
    // Write the module's blocks into the tables. Its element segments are
    // passive, so instantiating wrote nothing -- deliberate, because a module
    // is instantiated once per process and an active segment would make every
    // process write a slot for every block it holds (1.2-2.2s a process for a
    // closure library, mostly code never reached). Filling here moves that work
    // under the host's control, so a segment can be filled only when a branch
    // reaches its page. `gexfill` is absent from a pre-fill module and the image.
    if (globalThis.GEX_DEBUG_EXIT && this.#installs <= 2) {
      console.error(`[fill] module ${this.#installs}: gexfill ${typeof inst.exports[FILL]}, exports ${Object.keys(inst.exports).slice(0, 4).join(',')}`);
    }
    // Written now, or when a branch asks. The module decides: only one built
    // passive exports `gexfillat` and is deferred. A runtime unit has active
    // segments and is filled here -- it exists because a branch missed on it,
    // so deferring would only guarantee a second miss.
    if (typeof inst.exports[FILL_AT] === 'function') {
      this.#unfilled.unshift(inst);
      // Tell the shim there is now something worth asking for: until a module
      // is deferred, the miss path must not spend a trap per page on a host
      // with nothing to give.
      e.gex_fill_possible?.(1);
    } else if (typeof inst.exports[FILL_ALL] === 'function') {
      // One crossing, not one per segment -- a module has up to 355.
      inst.exports[FILL_ALL]();
    } else if (typeof inst.exports[FILL] === 'function') {
      // Until it says there are no more: the count is the module's own (1 for
      // the image, 355 for libc), and guessing left segments unwritten.
      for (let seg = 0; inst.exports[FILL](seg); seg++);
    }
    P.push(performance.now());
    this.#installSplit = P;
    // The instance, so the caller can remember which library it belongs to.
    return inst;
  }

  /**
   * Let an already-instantiated module answer fills. `#installModule` does this
   * for the modules it installs; the image module does not go through it (it is
   * the guest module itself, instantiated by `load`), so without this its
   * passive segments are never written and its pages stay on the miss stub.
   */
  #offerToFill(inst) {
    if (typeof inst?.exports?.[FILL_AT] !== 'function') return;
    this.#unfilled.unshift(inst);
    this.#shim.exports.gex_fill_possible?.(1);
  }

  /**
   * Put a registered closure library into the directory, and grow for it. Doing
   * this per library at registration cost each process the whole closure's
   * slots; here it costs a process only the libraries it enters.
   */
  #publish(lazy, i) {
    this.#tnAt('pre-pub');
    const e = this.#shim.exports;
    if (!lazy.meta || !lazy.meta.length || !e.gex_publish_closure_module) return;
    const t0 = performance.now();
    const gp = e.gex_alloc(lazy.meta.length) >>> 0;
    this.#bytes().set(lazy.meta, gp);
    e.gex_publish_closure_module(gp, lazy.meta.length, i);
    e.gex_free?.(gp, lazy.meta.length);
    // No grow here: a library's pages reach the table one at a time as
    // `#pageBase` claims them. Counted unconditionally because publishing is a
    // real per-process cost that no other counter accounts for.
    this.#publishes++;
    this.#publishMs += performance.now() - t0;
    if (globalThis.GEX_DEBUG_EXIT) {
      console.error(
        `[closure] library ${i} published in ${Math.round(performance.now() - t0)}ms`,
      );
    }
  }

  /**
   * Where this process's dispatch directory says a page's slots live.
   * `gexfillat` is told this rather than using the slot numbers baked in at
   * build time, which are one sequence across the whole closure. The directory
   * is per process, so the numbering can be too.
   */
  #pageBase(page) {
    const e = this.#shim.exports;
    if (!e.gex_claim_page || !e.gex_page_table) return -1;
    const t = e.gex_page_table(BigInt(page >>> 0));
    if (t < 0) return 0;
    // The shim numbers the page from a PROCESS-WIDE atomic allocator and writes
    // the directory with a compare-and-swap, so every thread of this process
    // agrees on the base (a per-Worker counter here used to let two threads
    // number one page differently and dispatch it through the wrong slot -- the
    // multithreaded-GTK stack smash). It returns the base, whether this thread
    // won the claim or adopted another's. The table is per instance, so reach
    // this Worker's to cover the base or a branch traps past its end.
    const base = e.gex_claim_page(BigInt(page >>> 0)) >>> 0;
    if (base === 0) return 0;
    this.#reach(t, base + SLOTS_PER_PAGE);
    return base >>> 0;
  }

  /**
   * Make dispatch table `t` at least `end` slots long. In fixed chunks, not
   * doubling: the watermark advances a page at a time, so doubling overshoots
   * enormously. A chunk amortises the reallocation over sixteen pages.
   */
  #reach(t, end) {
    const e = this.#shim.exports;
    const tables = this.#dispatchTables();
    if (t < 0 || t >= tables.length || tables[t].length >= end || this.#denseTable(t)) return;
    const per = (e.gex_slots_per_table?.() ?? 0) >>> 0 || Number.MAX_SAFE_INTEGER;
    const CHUNK = SLOTS_PER_PAGE * 16;
    const want = Math.min(Math.ceil(end / CHUNK) * CHUNK, per);
    const stub = tables[0].get(0);
    const delta = want - tables[t].length;
    if (delta <= 0) return;
    const g0 = performance.now();
    growTable(tables[t], delta, stub, t);
    this.#tallyTables(`dispatch${t} grew`);
    this.#pubGrewSlots += delta;
    this.#pubGrowMs += performance.now() - g0;
  }

  /**
   * Where in the page the fill should start, in slots -- recorded by the shim
   * from the pc that missed, rounded down to a multiple of `FILL_SPAN`. A stale
   * shim answers zero, which fills from the start of the page: wrong, not unsafe.
   */
  #fillFrom() {
    return this.#shim.exports.gex_fill_from ? this.#shim.exports.gex_fill_from() >>> 0 : 0;
  }

  /**
   * For each address that translated although its span had been filled, whether
   * its dispatch slot is written (a diagnostic). A written slot means
   * dispatch reads elsewhere than the fill wrote; a stub means the module holds
   * no block there. The shim collects the addresses; the table is read here.
   */
  liftSlots() {
    const e = this.#shim.exports;
    if (!e.gex_lift_pc || !e.gex_page_base) return undefined;
    const out = [];
    const tabs = this.#dispatchTables();
    if (!tabs.length) return undefined;
    const stub = tabs[0].get(0);
    const seen = new Set();
    for (let i = 0; i < 16; i++) {
      const pc = Number(e.gex_lift_pc(i));
      if (!pc) break;
      const page = Math.floor(pc / 4096);
      // The base the shim recorded at lift, not the one this page has now: a
      // forked child lifts its parent's code then execs a different program.
      const then = e.gex_lift_base ? e.gex_lift_base(i) >>> 0 : 0;
      const base = then || (e.gex_page_base(page >>> 0) >>> 0);
      const tn = e.gex_page_table ? e.gex_page_table(BigInt(page >>> 0)) : -1;
      let what;
      if (!base) {
        what = 'no base';
        // And why there is none: which images claim this page, which one
        // `image_is_ours` picks, and whether that one has a table for it.
        // Only the shim can see any of that.
        if (seen.size < 3 && !seen.has(page)) {
          seen.add(page);
          e.gex_explain_page?.(BigInt(page));
        }
      }
      else if (tn < 0 || tn >= tabs.length) what = 'no table';
      else {
        const slot = base + ((pc & 0xfff) >>> SLOT_SHIFT);
        what = slot >= tabs[tn].length
          ? `past end ${tabs[tn].length}`
          : tabs[tn].get(slot) !== stub ? 'written' : 'stub';
        // A stub, asked again for exactly this address's span: `FILL_ASKED`
        // keys a span once, so if the fill later runs at a different base the
        // gap is lost. `stub->written` means the span was mismatched,
        // `stub->stub` means there is nothing to write.
        if (what === 'stub') {
          const owner = e.gex_closure_owner
            ? e.gex_closure_owner(BigInt(page >>> 0)) : -1;
          const inst = owner >= 0 ? this.#lazyClosure[owner]?.instObj : null;
          const fill = inst?.exports?.[FILL_AT];
          if (typeof fill === 'function') {
            const mine = ((pc & 0xfff) >>> SLOT_SHIFT) & ~31;
            const said = fill(page, base, mine);
            const then = Gex.askState.get(`${page}:${mine}`);
            const now = `at=${base} len=${tabs[tn].length}`;
            const was = then
              ? `at=${then.at} len=${then.len} pub=${then.published ? 1 : 0}`
                + `${then.inst === inst ? '' : ' DIFFERENT-INSTANCE'}`
              : 'never asked for this span';
            // When the fill claims it wrote (`said=1`) but the slot is still a
            // stub, which table it actually wrote: a module's segments carry
            // their build table, which need not be the page's per-process table.
            // See task 148.
            let elsewhere = '';
            if (said && tabs[tn].get(slot) === stub) {
              for (let t = 0; t < tabs.length; t++) {
                if (t === tn) continue;
                if (slot < tabs[t].length && tabs[t].get(slot) !== stub) {
                  elsewhere += ` WROTE-TABLE-${t}`;
                }
              }
              if (!elsewhere) elsewhere = ' WROTE-NOWHERE';
            }
            what = tabs[tn].get(slot) !== stub
              ? `stub->written(was[${was}] now[${now}] said=${said ? 1 : 0})`
              : `stub->stub(was[${was}] now[${now}] said=${said ? 1 : 0}`
                + `${elsewhere} tn=${tn} slot=${slot})`;
          } else {
            what = `stub, owner=${owner} no fill`;
          }
        }
      }
      out.push(`${pc.toString(16)}:${what}`);
    }
    return out.length ? out.join(' ') : undefined;
  }

  // Whether a fill actually wrote the slot that *missed*, not merely something
  // in its span. `gexfillat` reports success when it wrote any of the page's
  // covered blocks, but on a page the closure covers only partly the missing
  // slot is a gap it leaves on the miss stub. Reporting that filled sends the
  // guest back to re-dispatch the same stub instead of demand-translating the
  // block, and the churn on a partial page corrupted execution -- the
  // closure-only fault `apt-get update` hit verifying an EdDSA signature and in
  // its forked methods. `gex_fill_pc` is the address that missed; with none
  // (a bulk fill) there is no single slot to insist on, so trust the report.
  #wroteWanted(page, at) {
    const mpc = Number(this.#shim.exports.gex_fill_pc?.() ?? 0n);
    if (!mpc || at <= 0) return true;
    const tn = this.#shim.exports.gex_page_table
      ? this.#shim.exports.gex_page_table(BigInt(page >>> 0)) : -1;
    const tabs = this.#dispatchTables();
    if (tn < 0 || tn >= tabs.length) return true;
    const want = at + ((mpc & 0xfff) >>> SLOT_SHIFT);
    return want < tabs[tn].length && tabs[tn].get(want) !== tabs[0].get(0);
  }

  // Fill a page without waiting, or null if impossible. The common case: the
  // page belongs to a library this process already built, and filling is a call
  // into wasm rather than the `await` of `fillSlot` (a whole event-loop turn).
  fillSlotNow(page) {
    const e = this.#shim.exports;
    // Every object that covers this page, not just the first: translation is
    // batched by block, so one image page can span several objects. See
    // `gex_closure_owner_at`.
    let owned = false;
    let filled = false;
    for (let k = 0; ; k++) {
      const owner = e.gex_closure_owner_at
        ? e.gex_closure_owner_at(BigInt(page), k)
        : (k === 0 && e.gex_closure_owner ? e.gex_closure_owner(BigInt(page)) : -1);
      if (owner < 0) break;
      owned = true;
      {
      const lazy = this.#lazyClosure[owner];
      // Which of the three ways this gives up, for the watched pages: an owned,
      // claimed image page should be answered here and after an execve was not.
      if (Gex.watchPages.has(page) && Gex.#whyNo < 24) {
        Gex.#whyNo++;
        const why = !lazy ? 'no lazy entry'
          : !lazy.instObj ? 'module not instantiated'
          : typeof lazy.instObj.exports[FILL_AT] !== 'function' ? 'no fill export'
          : 'reached the fill';
        console.error(`[closure] fill page ${page} owner ${owner}: ${why}`);
      }
      if (!lazy) continue;
      // Publishing is synchronous -- a directory write and a table grow --
      // so it belongs on this side of the fence.
      if (!lazy.published) {
        lazy.published = true;
        this.#publish(lazy, owner);
      }
      if (!lazy.instObj) continue;
      // A module below the block gate has *active* segments and no fill
      // helper: instantiating it wrote its slots, so there is nothing to ask
      // for and the page is already answered.
      if (typeof lazy.instObj.exports[FILL_AT] !== 'function') { filled = true; continue; }
      // An old closure exports `gexfillat(page, base)` and fills whole pages;
      // JS drops the extra `from` argument silently, so the arity is the honest
      // signal that the shim must ask once per page rather than per span.
      if (!this.#fillWidthTold) {
        this.#fillWidthTold = true;
        const wide = lazy.instObj.exports[FILL_AT].length < 3;
        if (wide) this.#shim.exports.gex_set_fill_width?.(SLOTS_PER_PAGE);
      }
      const b0 = performance.now();
      const at = this.#pageBase(page);
      const b1 = performance.now();
      this.#faBaseMs += b1 - b0;
      // `> 0`, not `>= 0`: zero is `#pageBase`'s "no translation" sentinel, and
      // filling at base 0 writes funcrefs over the bottom of table 0 and reports
      // success, wedging the address for good.
      // askState keeps what the module and table looked like here, so the retry
      // in `liftSlots` can be compared against it.
      const askFrom = this.#fillFrom();
      if (Gex.askState.size < 4096) {
        const tn0 = e.gex_page_table ? e.gex_page_table(BigInt(page >>> 0)) : -1;
        const tabs0 = this.#dispatchTables();
        Gex.askState.set(`${page}:${askFrom}`, {
          at,
          len: tn0 >= 0 && tn0 < tabs0.length ? tabs0[tn0].length : -1,
          published: !!lazy.published,
          inst: lazy.instObj,
        });
      }
      let did = at > 0 && lazy.instObj.exports[FILL_AT](page, at, askFrom);
      this.#faInitMs += performance.now() - b1;
      // Read back the one slot the guest will dispatch through: a span read-back
      // says the fill wrote something, this says it wrote the slot that missed.
      if (Gex.watchSlot && at > 0) {
        const mpc = Number(this.#shim.exports.gex_fill_pc?.() ?? 0n);
        if (mpc && Gex.#whySlot < 12) {
          Gex.#whySlot++;
          const want = at + ((mpc & 0xfff) >>> SLOT_SHIFT);
          const tn2 = this.#shim.exports.gex_page_table
            ? this.#shim.exports.gex_page_table(BigInt(page >>> 0)) : -1;
          const tabs2 = this.#dispatchTables();
          let set = 'no table';
          if (tn2 >= 0 && tn2 < tabs2.length) {
            set = want < tabs2[tn2].length
              ? (tabs2[tn2].get(want) !== tabs2[0].get(0) ? 'WRITTEN' : 'still stub')
              : `past end (${tabs2[tn2].length})`;
          }
          console.error(`[closure] pc ${mpc.toString(16)} page ${page} slot ${want}`
            + ` (base ${at} + ${(mpc & 0xfff) >>> SLOT_SHIFT}) t=${tn2}: ${set}`);
        }
      }
      if (Gex.watchPages.has(page) && Gex.#whyDid < 24) {
        Gex.#whyDid++;
        // `did` only says the module wrote something in the span; count the
        // span rather than trusting it.
        const from = this.#fillFrom();
        const tn = this.#shim.exports.gex_page_table
          ? this.#shim.exports.gex_page_table(BigInt(page >>> 0)) : -1;
        const tabs = this.#dispatchTables();
        let wrote = -1, len = -1;
        if (tn >= 0 && tn < tabs.length) {
          const stub = tabs[0].get(0);
          len = tabs[tn].length;
          wrote = 0;
          for (let i = from; i < from + 128 && at + i < len; i++) {
            if (tabs[tn].get(at + i) !== stub) wrote++;
          }
        }
        console.error(`[closure] fill page ${page} owner ${owner}: at=${at} did=${did} `
          + `t=${tn} from=${from} wrote=${wrote}/128 tlen=${len}`);
      }
      // `did` says the module wrote *somewhere* in the span, not that it wrote
      // the slot that actually missed -- see `#wroteWanted`.
      if (did && this.#wroteWanted(page, at)) {
        this.#filled++;
        filled = true;
      }
      }
    }
    if (filled) return true;
    // Owned but unfillable: the caller's fallback scan cannot help, because
    // the objects that claim this page are the only ones that hold its code.
    if (owned) return null;
    if (page * 4096 < 0x04a0_0000) return null;
    const where = this.#pageBase(page);
    // Same sentinel as above: zero is "no translation", not slot zero.
    if (where <= 0) return null;
    for (const inst of this.#unfilled) {
      if (inst.exports[FILL_AT](page, where, this.#fillFrom())
          && this.#wroteWanted(page, where)) {
        this.#filled++;
        return true;
      }
    }
    return null;
  }

  async fillSlot(slot) {
    // The library that covers this page, instantiated now if not yet. The
    // shim's ordinals are the registration order, the same as `#lazyClosure`.
    const owner = this.#shim.exports.gex_closure_owner
      ? this.#shim.exports.gex_closure_owner(BigInt(slot))
      : -1;
    if (owner >= 0) {
      const lazy = this.#lazyClosure[owner];
      if (lazy && !lazy.published) {
        lazy.published = true;
        this.#publish(lazy, owner);
      }
      if (lazy && !lazy.inst) {
        lazy.inst = true;
        // The compiled module, fetched now if registration only brought its
        // metadata: a Worker gets the metadata blobs up front and asks for a
        // module when a branch lands in it. Timed apart, since it is the host
        // round-trip, not the install, that costs the slow fill its second.
        const w0 = performance.now();
        const mod = lazy.mod ?? (this.#wantLibrary ? await this.#wantLibrary(owner) : null);
        this.#wants++;
        this.#wantMs += performance.now() - w0;
        if (mod) {
          lazy.mod = mod;
          // For a demand-filled (claimed) library, need 0 and no metadata: it
          // numbers its own pages one at a time, so the build's `dir_entries`
          // would grow the tables to a layout nobody uses. Only a module that
          // writes absolute slots at instantiation wants the metadata.
          const claimed = this.#shim.exports.gex_lib_lazy?.(owner >>> 0);
          const i0 = performance.now();
          lazy.instObj = await this.#installModule(
            mod,
            claimed ? 0 : lazy.need,
            64,
            claimed ? null : lazy.meta,
            'closure',
          );
          this.#instMs += performance.now() - i0;
        } else {
          // Nothing to install; let the miss fall through and be lifted rather
          // than wedging on a library that never arrives.
          lazy.inst = false;
        }
      }
    }
    const tail0 = performance.now();
    try {
      return await this.#fillSlotTail(slot, owner);
    } finally {
      this.#tailMs += performance.now() - tail0;
    }
  }

  async #fillSlotTail(slot, owner) {
    // The same for a module inherited from a parent: instantiating writes its
    // active segments, so there is nothing further to fill, but asking beats
    // the miss path re-serving code this process already holds.
    const heir = this.#shim.exports.gex_lazy_owner
      ? this.#shim.exports.gex_lazy_owner(BigInt(slot))
      : -1;
    if (heir >= 0) {
      const lazy = this.#lazyInherited[heir];
      if (lazy && !lazy.inst) {
        lazy.inst = true;
        await this.#installModule(lazy.mod, lazy.need, 64, lazy.meta, 'inherit', null, false, lazy.dense);
        return true;
      }
    }
    // The owner first, and below `INTERP_BASE` the owner only: those pages
    // belong to every image at once and only one of them is ours.
    const mine = owner >= 0 ? this.#lazyClosure[owner]?.instObj : null;
    if (mine && typeof mine.exports[FILL_AT] !== 'function') return true;
    const at = this.#pageBase(slot);
    // Only "filled" when the slot that missed is off the stub, not merely when
    // the module wrote its span. See `#wroteWanted`.
    if (mine && at > 0 && mine.exports[FILL_AT](slot, at, this.#fillFrom())
        && this.#wroteWanted(slot, at)) {
      this.#filled++;
      return true;
    }
    if (slot * 4096 >= 0x04a0_0000) {
      for (const inst of this.#unfilled) {
        if (at > 0 && inst.exports[FILL_AT](slot, at, this.#fillFrom())
            && this.#wroteWanted(slot, at)) {
          this.#filled++;
          return true;
        }
      }
    }
    // A request nobody can answer, then a clean run-time lift of the block: for
    // a skipped library (its pages have a run-time base but no deferred module)
    // this is every page and expected, so it narrates only under `debugFill`.
    // Capped even then, since a flood would be the failure, not the diagnosis.
    if (Gex.debugFill && this.#unanswered++ < 24) {
      console.error(
        `[fill] nothing holds entry ${slot >>> 0}: ` +
          `${this.#unfilled.length} deferred modules, ${this.#filled} filled`,
      );
    }
    return false;
  }

  /** A dispatch missed on a block of this process this Worker never
   *  instantiated (`gex_dense_want`): instantiate the registry module that
   *  holds it, whatever order this Worker caught up in. Synchronous. */
  #denseCatchUp() {
    const e = this.#shim.exports;
    const d = e.gex_dense_want ? e.gex_dense_want() >>> 0 : 0;
    if (!d) return;
    this.#denseAsks++;
    const i = e.gex_dense_owner(d);
    if (i < 0) return;
    const bytes = this.#registryBytes(i, true);
    if (!bytes) return;
    const t0 = performance.now();
    const c0 = this.#compileMs, i0 = this.#instantiateMs, g0 = this.#growMs;
    const key = this.#registryKey(i, bytes);
    const cached = key ? this.#sharedModules.get(key) : undefined;
    this.#installModule(bytes, 0, 64, null, `dense-${i}`, cached, true);
    this.#denseCatchUps++;
    if (!cached) {
      const mod = this.#compiled[this.#compiled.length - 1]?.mod;
      if (mod && key) {
        this.#sharedModules.set(key, mod);
        this.#deadKeys.delete(key);
        this.#shareRegistry(i, key, mod);
      }
    }
    const ms = performance.now() - t0;
    this.#denseFillMs += ms;
    if (ms > this.#denseFillWorst) {
      this.#denseFillWorst = ms;
      if (globalThis.GEX_FILL_TIME && ms >= 1) {
        console.warn(`[gex] ${this.#whoFills()}: dense module ${i} (${bytes.length} bytes${cached ? ', compiled elsewhere' : ''}) took ${ms.toFixed(2)} ms, ` +
          `${(this.#compileMs - c0).toFixed(1)} compiling, ${(this.#instantiateMs - i0).toFixed(1)} instantiating, ${(this.#growMs - g0).toFixed(1)} growing (${this.#splitText()}), the longest here; ` +
          `${this.#fillCounts()}, ${this.#denseFillMs.toFixed(1)} ms filling`);
      }
    }
  }

  /** The last install's times (`#installModuleInner`): its start, then the
   *  end of reaping, of growing and compiling, of reading its metadata, of
   *  reading its segments, and of instantiating and filling. */
  #installSplit = null;
  #splitText() {
    const P = this.#installSplit;
    if (!P || P.length < 6) return '';
    const d = (k) => (P[k] - P[k - 1]).toFixed(1);
    return `reap ${d(1)}, grow+compile ${d(2)}, meta ${d(3)}, segments ${d(4)}, instantiate+fill ${d(5)}`;
  }

  /** Dense modules a miss instantiated here, and the misses that asked. */
  #denseCatchUps = 0;
  #denseAsks = 0;
  /** Their instantiate time in this Worker: the total, and the longest one
   *  (a dispatch waits for it, whichever thread it is). */
  #denseFillMs = 0;
  #denseFillWorst = 0;
  /** The thread this Worker runs, when it was spawned for one (`adoptThread`). */
  #threadTid = 0;
  /** A thread's: how many `execve`s its process had made when it was
   *  adopted (`adoptThread`). */
  adoptGen = undefined;
  #whoFills() {
    return `pid ${this.#shim.exports.gex_my_pid?.() ?? '?'} ${this.#threadTid ? `tid ${this.#threadTid}` : 'main thread'}`;
  }
  /** What this Worker holds, for `?filltime=1`. */
  #fillCounts() {
    return `${this.#installs} modules installed (${this.#compiles} compiled here), ` +
      `dense ${this.#denseCatchUps} filled of ${this.#denseLeft} left (${this.#denseAsks} asks), blocks ${this.#guest?.exports?.blocks?.length ?? 0}, ` +
      `${this.#sharedModules.size} modules held, ${this.#relaySkipped} not relayed`;
  }
  /** Dense registry modules this Worker's catch-up left to a miss. */
  #denseLeft = 0;

  /** Registry module `i`, compiled here, to the host's relay for this
   *  process's other Workers -- unless it is a dense module those Workers
   *  instantiate on a miss (`#leaveDense`): relayed, every module would be
   *  posted through the page's main thread to every Worker and kept there,
   *  for Workers most of which never run it. A miss compiling the same bytes
   *  again gets the engine's code for them while any Worker holds it. */
  #shareRegistry(i, key, mod) {
    const e = this.#shim.exports;
    if (this.#dense && e.gex_registry_dense_n && (e.gex_registry_dense_n(i) >>> 0)) return;
    this.#offerModule(key, mod, this.#registryRow(i).len);
  }

  /** A module this Worker compiled from `bytes` bytes, to the host's relay
   *  for the process's other Workers (and, by its key, the page's caches) --
   *  unless it is smaller than RELAY_MIN_BYTES. Every relayed module is held
   *  by the page's main thread as well as by each Worker it reaches, and
   *  each collection on the main thread takes longer for every module alive
   *  in the renderer, holding up compiles on every Worker meanwhile; a small
   *  module costs a sibling little to compile from its bytes, less while any
   *  Worker holds the same bytes compiled. A native program or library
   *  (`wasm:`) is relayed at any size: the page keeps those for the
   *  processes that run it next. */
  #offerModule(key, mod, bytes) {
    // Units go to the page whatever their size: the host keeps them for every
    // process, and a Worker without one fetches its bytes as well.
    if (!(bytes >= RELAY_MIN_BYTES) && !/^(wasm|cgen|cold|late|plan):/.test(String(key))) {
      this.#relaySkipped++;
      return;
    }
    this.#env.shareModule?.(key, mod);
  }
  #relaySkipped = 0;

  /** Whether catching up skips registry module `i`: a dense module, which
   *  the first dispatch that wants one of its blocks instantiates here
   *  (`#denseCatchUp`), so a Worker holds only what its thread runs.
   *  `blocks` is grown for it all the same, its entries the miss stub. */
  #leaveDense(i) {
    const e = this.#shim.exports;
    if (!this.#dense || !e.gex_registry_dense_n) return false;
    if (!(e.gex_registry_dense_n(i) >>> 0)) return false;
    this.#denseLeft++;
    this.#growBlocks();
    return true;
  }

  /** Dead modules of the process (`gex_dense_dead_*`) this Worker has let go of. */
  #deadSeen = 0;
  /** Their content keys, so a sibling's relay does not bring one back. */
  #deadKeys = new Set();

  /** Runtime modules of this process no dispatch can reach any more (all
   *  their pages retired: an unloaded image's: a plugin scan loads
   *  and unloads dozens), let go of here: their `blocks` entries back to the
   *  miss stub, which kept each one's instance (and with it its code) alive in
   *  every Worker, and this Worker's own references dropped, so V8 frees them.
   *  Cheap when nothing died: one call. */
  #reapDead() {
    const e = this.#shim.exports;
    if (!this.#dense || !e.gex_dense_dead_len) return;
    const n = e.gex_dense_dead_len() >>> 0;
    // A new program (execve) numbers its dead from 0 again.
    if (n < this.#deadSeen) this.#deadSeen = 0;
    if (n <= this.#deadSeen) return;
    const b = this.#guest?.exports?.blocks;
    const stub = b?.get(0);
    const dead = new Set();
    for (; this.#deadSeen < n; this.#deadSeen++) {
      const base = e.gex_dense_dead_at(this.#deadSeen, 0) >>> 0;
      const count = e.gex_dense_dead_at(this.#deadSeen, 1) >>> 0;
      dead.add(base);
      if (b) for (let i = base, end = Math.min(base + count, b.length); i < end; i++) b.set(i, stub);
    }
    const mods = new Set();
    this.#compiled = this.#compiled.filter((c) => {
      const ds = c.dense;
      if (!ds || ds.byteLength < 4) return true;
      const base = new DataView(ds.buffer, ds.byteOffset, ds.byteLength).getUint32(0, true);
      if (!dead.has(base)) return true;
      mods.add(c.mod);
      return false;
    });
    if (!mods.size) return;
    const keys = [];
    for (const [k, m] of this.#sharedModules) {
      if (mods.has(m)) {
        this.#sharedModules.delete(k);
        if (typeof k === 'string') { keys.push(k); this.#deadKeys.add(k); }
      }
    }
    if (keys.length) this.#env.dropModules?.(keys);
  }

  /** Whether dispatch table `t` is dense here (gex_emit::dense::FIRST_TABLE
   *  and up): its slots are the process's slot maps, never this table. */
  #denseTable(t) {
    return this.#dense && t >= DENSE_FIRST_TABLE;
  }

  /** Once this process's numbering is set (`gex_set_dense`, or the
   *  process's for a thread), where dispatch is dense: `blocks[0]`, the miss
   *  stub every entry not yet instantiated here holds, becomes its entry
   *  wrapper. */
  #entriesInit() {
    const e = this.#shim.exports;
    this.#entries = this.#dense;
    const b = this.#guest?.exports?.blocks;
    if (!this.#entries || !b) return;
    const miss = b.get(0);
    if (this.#entryWrapped.has(miss)) return;
    const w = this.#entryOf(miss);
    for (let i = 0; i < b.length; i++) if (b.get(i) === miss) b.set(i, w);
  }

  /** `GEX_LOG=bridges`: the unit-entry addresses this Worker entered
   *  through a bridge, and the bridge entries it wrote into its tables (a
   *  `blocks` entry serves many bridged slots). */
  #bridgeCount = null;

  /** `real` (the runtime module's unit bridge), counting the distinct
   *  addresses entered through it. A line names the new ones: the first
   *  eight at once, then at most one line a second or per 64 -- a Worker
   *  parked in a wait says nothing later. */
  #countBridges(real) {
    const seen = new Set();
    let fresh = [];
    let said = 0;
    const c = { bridged: 0 };
    this.#bridgeCount = c;
    const say = () => {
      console.error(`[bridges] ${this.#whoFills()}: entered ${seen.size} addresses, ${c.bridged} bridge entries written; new ${fresh.map((pc) => pc.toString(16)).join(',')}`);
      fresh = [];
      said = performance.now();
    };
    const count = (ctx) => {
      const pc = new DataView(this.#memory.buffer).getBigUint64((ctx >>> 0) + CTX_LAST_TARGET, true);
      if (seen.has(pc)) return;
      seen.add(pc);
      fresh.push(pc);
      if (seen.size <= 8 || fresh.length >= 64 || performance.now() - said >= 1000) say();
    };
    return new WebAssembly.Instance(new WebAssembly.Module(BRIDGE_COUNT), { w: { count, real } }).exports.c;
  }

  /** The wrappers `#entryOf` made. */
  #entryWrapped = new WeakSet();

  /** The `(ctx, arm)` entry wrapper of a block-type function, where
   *  dispatch is dense (else `f` itself). */
  #entryOf(f) {
    if (!this.#entries || !f) return f;
    let w = this.#entryWraps.get(f);
    if (!w) {
      entryWrapMod ??= new WebAssembly.Module(ENTRY_WRAP);
      w = new WebAssembly.Instance(entryWrapMod, { w: { f } }).exports.e;
      this.#entryWraps.set(f, w);
      this.#entryWrapped.add(w);
    }
    return w;
  }

  /** Write the bridge at `blocks` entry `i`. Bridged slots share entries
   *  (`gex_emit::dense::bridge_words`), and a publish names every entry its
   *  slots take arms of, ones this Worker already holds the bridge in among
   *  them. Rewriting such an entry dropped the watch a call-out keeps on a
   *  sibling arm (`#refSet`'s wrappers): the guest returning there before
   *  the next `#armCallouts` ran the bridge rather than the watch stub, and
   *  the call-out never saw its return. So an entry holding the bridge is
   *  left as it is, and a watched one keeps its watches around the bridge. */
  /** Write the bridge at each (table, index) pair of `regs`, table
   *  `TABLE_NAMES.length` being `blocks`; a table tagged `ALT_BRIDGE` (the
   *  other mode's, cgen.rs `bridges_mode`) gets that mode's bridge. */
  #writeBridges(tables, blocks, regs, bridge) {
    for (let j = 0; j + 1 < regs.length; j += 2) {
      let t = regs[j], b = bridge;
      if (t & ALT_BRIDGE) {
        t &= ~ALT_BRIDGE;
        b = this.#cgenAlt().bridge;
      }
      if (t === TABLE_NAMES.length) this.#setBridgeEntry(blocks, regs[j + 1], b);
      else tables[t].set(regs[j + 1], b);
    }
  }

  #setBridgeEntry(blocks, i, bridge) {
    const be = this.#entryOf(bridge);
    const now = blocks.get(i);
    if (now === be) return;
    const w = this.#entryWatch.get(i);
    if (w && w.fn === now) {
      if (w.orig === be) return;
      w.orig = be;
      let fn = be;
      for (const a of w.arms) fn = this.#watchArm(fn, a);
      w.fn = fn;
      blocks.set(i, fn);
      return;
    }
    blocks.set(i, be);
  }

  /** Under GEX_TEST_BRIDGE_WATCH (read once, at load), `Gex.testBridgeWatch()`:
   *  a call-out's watch on one arm of a `blocks` entry, then publishes that
   *  write the entry its sibling arms bridge through, the same bridge and a
   *  new one. The watch must outlive both, and the sibling arms run the
   *  bridge last written. Returns the checks that failed (a test's). */
  static {
    if (globalThis.process?.env?.GEX_TEST_BRIDGE_WATCH) {
      Gex.testBridgeWatch = () => {
        const g = new Gex();
        const blocks = new WebAssembly.Table({ element: 'anyfunc', initial: 4 });
        g.#entries = true;
        g.#guest = { exports: { blocks } };
        const ran = [];
        // (module (type (func (param i32))) (import "w" "f" (func (type 0)))
        //   (func (export "b") (type 0) local.get 0 call 0))
        const mod = new WebAssembly.Module(new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 1, 127, 0, 2, 7, 1, 1, 119, 1, 102, 0, 0,
          3, 2, 1, 0, 7, 5, 1, 1, 98, 0, 1, 10, 8, 1, 6, 0, 32, 0, 16, 0, 11,
        ]));
        const bridge = (tag) => new WebAssembly.Instance(mod, { w: { f: (ctx) => ran.push(`${tag}:${ctx}`) } }).exports.b;
        const one = bridge('one'), two = bridge('two');
        const at = 2, T = TABLE_NAMES.length;
        const fails = [];
        const check = (what, ok) => { if (!ok) fails.push(what); };
        const sibling = (want) => {
          ran.length = 0;
          blocks.get(at)(7, 2);
          check(`a sibling arm runs ${want} (ran ${ran.join(',') || 'nothing'})`, ran.join() === `${want}:7`);
        };
        g.#writeBridges([], blocks, [T, at], one);
        g.#refSet(T, (at << 8) | 1, g.#watch());
        check('the arm is watched', g.#refGet(T, (at << 8) | 1) === g.#watch());
        g.#writeBridges([], blocks, [T, at, T, at], one);
        check('the watch outlives a publish of the same bridge', g.#refGet(T, (at << 8) | 1) === g.#watch());
        sibling('one');
        g.#writeBridges([], blocks, [T, at], two);
        check('the watch outlives a publish of a new bridge', g.#refGet(T, (at << 8) | 1) === g.#watch());
        sibling('two');
        g.#refSet(T, (at << 8) | 1, g.#entryOf(two));
        check('the watch comes off', !g.#entryWatch.has(at) && blocks.get(at) === g.#entryOf(two));
        return fails;
      };
    }
  }

  /** Grow this Worker's `blocks` to every index the process has numbered
   *  (`gex_dense_size`), the miss stub (its entry 0) in the new entries.
   *  Whether it grew. */
  #growBlocks() {
    const b = this.#guest?.exports?.blocks;
    if (!this.#dense || !b) return false;
    const need = this.#shim.exports.gex_dense_size() >>> 0;
    const len = b.length;
    if (len >= need) return false;
    // An eighth over what is numbered (4096 entries at least), not double:
    // each entry is per Worker in the renderer's trusted range, shared by
    // all its isolates, and a doubled `blocks` was mostly never filled.
    const want = need + Math.max(need >>> 3, 4096);
    growTable(b, want - len, b.get(0), 'blocks');
    this.#tallyTables('blocks grew');
    return true;
  }

  /**
   * `GEX_LOG=tables`: this Worker's funcref tables, each as length/entries
   * used (an entry other than the table's fill)/entries the process has
   * numbered (where it says), one line, said when one of
   * them grows -- a Worker parked in a wait says
   * nothing later, so its last line is what it holds. V8 keeps each entry
   * in the renderer's trusted range, which all its isolates share. Off: one
   * call and a compare.
   */
  #tallyTables(why) {
    if (!logOn('tables')) return;
    const used = (t, fill) => {
      let n = 0;
      const len = Number(t.length);
      for (let j = 0; j < len; j++) if (t.get(typeof t.length === 'bigint' ? BigInt(j) : j) !== fill) n++;
      return `${len}/${n}`;
    };
    const seen = new Set();
    const out = [];
    // `need`: the entries the process has numbered, which the table must cover.
    const note = (name, t, fill, need) => {
      if (!t || seen.has(t)) return;
      seen.add(t);
      out.push(`${name}=${used(t, fill)}${need === undefined ? '' : `/${need >>> 0}`}`);
    };
    try {
      const tables = this.#guest ? this.#dispatchTables() : [];
      const stub = tables[0]?.get(0);
      tables.forEach((t, i) => note(`dispatch${i}`, t, stub));
      const b = this.#guest?.exports?.blocks;
      const e = this.#shim.exports;
      if (b) note('blocks', b, b.get(0), e.gex_dense_size?.());
      if (this.#cgen) note('ftab', this.#cgen.ftab, this.#cgen.resolve, e.gex_ftab_size?.());
      for (const [where, ex] of [['guest', this.#guest?.exports], ['shim', this.#shim?.exports]]) {
        for (const [k, v] of Object.entries(ex ?? {})) if (v instanceof WebAssembly.Table) note(`${where}.${k}`, v, null);
      }
    } catch (err) {
      out.push(`(tally failed: ${err?.message ?? err})`);
    }
    console.error(`[tables] ${this.#whoFills()} ${why}: ${out.join(' ')}`);
  }
  /** Every dispatch table the guest module exports, in index order. */
  #dispatchTables() {
    const out = [];
    for (const n of TABLE_NAMES) {
      const t = this.#guest.exports[n];
      if (!t) break;
      out.push(t);
    }
    return out;
  }

}

/** A PIC module's `dylink.0` table_size (WASM_DYLINK_MEM_INFO: mem_size,
 *  mem_align, table_size, table_align, LEB128). */
function dylinkTableSize(mod) {
  const sec = WebAssembly.Module.customSections(mod, 'dylink.0')[0];
  if (!sec) throw new Error('door: the server program has no dylink.0 section');
  const b = new Uint8Array(sec);
  let i = 0;
  const leb = () => { let r = 0, sh = 0, c; do { c = b[i++]; r += (c & 0x7f) * 2 ** sh; sh += 7; } while (c & 0x80); return r; };
  while (i < b.length) {
    const kind = b[i++], len = leb(), end = i + len;
    if (kind === 1) { leb(); leb(); return leb(); }
    i = end;
  }
  throw new Error('door: no WASM_DYLINK_MEM_INFO in the server program');
}
