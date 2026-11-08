// The page's half of a guest system: one Worker per guest process.
//
// Each Worker has its own `WebAssembly.Memory`, so nothing collides and
// nothing is snapshotted. Shared state -- the process table, filesystem, pipes
// -- lives in the global kernel, a second wasm module every Worker
// instantiates against one shared memory, so a syscall stays a call rather
// than a message back through here. This file starts pid 1, makes a Worker per
// fork, relays I/O, and holds the run open until pid 1 exits.

// Narration under node goes to stderr: node's console.debug is stdout, the
// guest's own output there. In a page it stays on the debug console.
function narrate(text) {
  // Debug output, off unless GEX_LOG (`?log=1`) asks for it (log.mjs).
  if (!logOn('debug')) return;
  const stderr = typeof process !== 'undefined' ? process.stderr : null;
  if (stderr?.write) stderr.write(`${text}\n`);
  else console.debug(text);
}

/**
 * Announce a crash to an automated watcher. These failures are caught (the
 * spawn promise is rejected and handled), so no `error` event fires and a test
 * that already matched its output has stopped reading. A hook, not prose
 * scraping, which fails open. See task 146.
 */
import { single } from './single.mjs';
import { SP_CODE } from './mode.mjs';
import { forkTransfer } from './fork-transfer.mjs';
import { HELPER_STACK_COUNT, wasmMemory } from './lazy-fetch.mjs';
import { logNamed, logOn, logQuiet, logRoutine } from './log.mjs';
import { XC_STOP, xcBusy, xcGate, xcSleep } from './xcache.mjs';
// The host's clock object, looked up once in this realm: `performance` is a
// Blink attribute of the global scope, and reading it costs more than `now()`.
const perf = globalThis.performance;


/** A line gex says by default (log.mjs): to stderr under Node, else the
 *  console's info level. */
function inform(text) {
  if (logQuiet()) return;
  const fs = typeof process !== 'undefined' ? process.getBuiltinModule?.('fs') : null;
  if (fs?.writeSync) {
    try { fs.writeSync(2, `${text}\n`); return; } catch {}
  }
  console.info(text);
}

// `?memories=1` (or GEX_MEMORIES), read once at load: a test harness's view
// of each threaded process's memory by pid, `globalThis.__gexMemories`, what
// it reads a guest word from while the process's own Workers are busy in
// wasm (a game's frame counter). Held until the process ends; off, nothing
// is registered and nothing is kept.
const MEMORIES = !!(globalThis.GEX_MEMORIES || globalThis.process?.env?.GEX_MEMORIES || (typeof location !== 'undefined' && new URLSearchParams(location.search).get('memories') === '1'));
function forgetMemory(pid) {
  if (MEMORIES) globalThis.__gexMemories?.delete(Number(String(pid).split(':')[0]));
}

// A library or image loaded for the first time this session, said once
// rather than once per process (each Wine process maps the same dozen DLLs).
// Every Worker says what it installs (fetch-note.mjs `noteStatus`): to its
// host (`relayNote`), or on the 'gex-fetch' channel from a realm without
// one; this side keeps the set.
const libsSeen = new Set();
let libChannel = null;
function seeLibrary(st) {
  if (!st || (st.kind !== 'module' && st.kind !== 'unit') || !st.path) return;
  const path = String(st.path).replace(/^wasm(dll|prog):/, '');
  if (libsSeen.has(path)) return;
  libsSeen.add(path);
  if (logRoutine()) inform(`[gex] loaded ${path}${st.kind === 'unit' ? (st.hit ? ' (with its unit)' : ' (no unit)') : ''}`);
}
function watchLibraries() {
  if (libChannel || typeof BroadcastChannel !== 'function') return;
  libChannel = new BroadcastChannel('gex-fetch');
  // Under Node an open channel keeps the event loop alive.
  libChannel.unref?.();
  libChannel.onmessage = (ev) => seeLibrary(ev.data?.status);
}
/** A Worker's note (fetch-note.mjs): seen here, and said on the channel for
 *  the page (app.js `watchFetches`), which this side's own channel is not
 *  handed back. */
function relayNote(note) {
  seeLibrary(note?.status);
  watchLibraries();
  try { libChannel?.postMessage(note); } catch {}
}

function crashed(what) {
  try {
    globalThis.GEX_ON_CRASH?.(what);
  } catch {
    // A reporter that throws must not add a failure to the one it reports.
  }
}

/// Where a pooled Worker's early messages are kept until somebody owns it.
const EARLY = Symbol('gex.early');
/// The keys of the page's modules a pooled Worker was sent while it waited
/// (`#poolModules`), so its start message carries only the rest.
const POOLED = Symbol('gex.pooled');
/// A pooled Worker's global-kernel stack slice, taken when it was sent
/// `preattach` (keyed `pool:N` until a process or thread takes the Worker).
const STACK_KEY = Symbol('gex.stackKey');
/// A pooled Worker has said `ready` (its handler is on): it can be sent
/// `preattach`.
const READY = Symbol('gex.ready');
/// A process Worker's listeners (`#spawn`), taken off when it is given back.
const LISTENING = Symbol('gex.listening');

/**
 * Which processes and threads (`pid` or `pid:tid`) a named Worker has run,
 * in order, as `globalThis.__gexWorkers` (Worker name -> keys): a pooled
 * Worker is named before it has a process, and a profiler or trace lists its
 * thread only by that name. A Worker without a name is not recorded.
 */
function workerRan(w, key) {
  const name = w?.gexName;
  if (!name) return;
  const m = (globalThis.__gexWorkers ??= new Map());
  const ran = m.get(name);
  if (ran) ran.push(String(key));
  else m.set(name, [String(key)]);
}
/// How many processes a Worker has run after its first (`#giveBack`).
const REUSES = Symbol('gex.reuses');
/** How many processes one Worker runs before it is ended anyway, so a leak
 *  in a realm that is given back cannot grow without end; and how many
 *  given-back Workers the pool keeps beyond the new ones it is topped up to. */
const REUSE_MAX = 256;
const GIVEN_BACK_MAX = 2;
/** The other flavour's runtime after a failed fetch: when it is tried again
 *  (`#loadOther`), the wait doubling from the first to the last. */
const OTHER_RETRY_MS = 1000, OTHER_RETRY_MAX_MS = 60_000;
/** How long `stop` waits for store saves under way. */
const SAVE_WAIT_MS = 60_000;
/** How many exited processes' memories are kept for fork children, and how
 *  many bytes they may hold together: memory the page held anyway while
 *  those processes ran, not more. */
const RECYCLE_COUNT = 4;
/** How many spare thread Workers (`#startSpare`) the page keeps waiting at
 *  once; past it the oldest is let go. */
const SPARE_MAX = 8;
/** How long output waits for a write numbered before it (`#inOrder`). */
const OUTPUT_GAP_MS = 200;
/** How long a spare waits for its thread before it is let go. A process
 *  starts its threads in bursts, which find their spares within seconds;
 *  one that waits longer holds a whole thread's instantiation for a start
 *  that may never come. */
const SPARE_IDLE_MS = 5000;
/** How many Workers whose thread ended one process keeps waiting for its
 *  next threads (`park`), within SPARE_MAX. A program that starts short
 *  threads one after another needs one. */
const PARKED_MAX = 2;
/// How many threads a Worker has run (`park`), the last included.
const THREADS_RUN = Symbol('gex.threadsRun');
/** A test-only switch: whether a running fork child's
 *  record, its parent's modules with it, can be collected once the child's
 *  Worker has it. Node only, with `--expose-gc`. */
const TEST_FORK_RECORDS = !!globalThis.process?.env?.GEX_TEST_FORK_RECORDS;
const RECYCLE_BYTES = 768 << 20;
/** How long kept memories wait for a fork child after the last exit: then
 *  they go. A page that forks keeps forking within seconds (a shell, a
 *  build); one that does not (a Wine game: wineboot exits, then nothing
 *  forks) would otherwise hold up to RECYCLE_BYTES of touched pages for good. */
const RECYCLE_KEEP_MS = 15_000;

/** How far a fork record's image reaches: what its child's memory grows to. */
function imageExtent(child) {
  // Said by a record whose image travels apart (worker.mjs `FORK_BULK`).
  if (child?.extent !== undefined) return child.extent;
  let n = 0;
  for (const r of child?.image ?? []) {
    n = Math.max(n, r.end ?? 0, r.at + (r.len ?? r.bytes?.byteLength ?? 0));
  }
  return n;
}
/** How many content-keyed modules (`wasm:`) the page keeps for every
 *  process, and how many bytes of wasm they may add up to (their compiled
 *  code is larger still; python3.13 alone is 24 MiB of wasm). A program's
 *  shared libraries are keyed so too: Konsole alone maps 152 of them, about
 *  210 MiB, and a process exec'd again recompiles every one the page let go. */
const PAGE_MODULES = 1024;
const PAGE_MODULE_BYTES = 1024 << 20;
/** The most late-call modules (`late:`, gex.mjs `lateCalls`) the page keeps
 *  compiled for its Workers (`#lateModules`): each is a few hundred bytes. */
const LATE_PAGE_MODULES = 4096;
/** The most parsed binding-list plans (`plan:`, gex.mjs `bindPlan`) the
 *  page keeps for its Workers (`#planModules`): a few KiB to 100 KiB each,
 *  on shared buffers (KDE's boot makes 492, 22 MB of lines). */
const PLAN_PAGE_MODULES = 4096;
/** The most unit modules the page keeps compiled for its Workers (`#cgenModules`). */
const UNIT_PAGE_MODULES = 4096;
/** The most unit-cache files the page keeps for its Workers (`#unitFiles`). */
const UNIT_FILES = 16384;
/** The most wasm the page-wide unit cache (`#unitModules`) keeps compiled. */
const UNIT_MODULE_BYTES = 256 << 20;
/** The most wasm one process's relay cache (`#moduleCache`) keeps: a pc
 *  translated again and again (a dispatch that does not
 *  converge) made a fresh key each time, and an
 *  unbounded cache held every one, 10 GiB of renderer within minutes. Past
 *  it the oldest are dropped from the cache (Workers that installed them
 *  keep theirs; a thread started later compiles what it needs). */
const PROCESS_MODULE_BYTES = 256 << 20;
/** And the most modules it, or the page's unit cache, keeps: each module the
 *  main thread holds lengthens every collection there, and those hold up
 *  compiles on every Worker of the page. */
const RELAY_MODULES = 512;
/** A registry key's module size: `LENGTH:HASH:HASH` (or `u:LENGTH:...`). */
const regKeyBytes = (k) => Number(String(k).replace(/^u:/, '').split(':')[0]) || 0;
/** A `wasm:` key's module size: the key is `wasm:LENGTH:HASH:HASH`. */
const keyBytes = (k) => Number(String(k).split(':')[1]) || 0;

/** A memory's bytes in a snapshot (`ProcessHost.restore`): a Uint8Array of
 *  all of them, or `{ length, runs }`, its nonzero runs as `[offset, bytes]`
 *  (the rest zero, as a new memory is). */
function snapLength(m) {
  return m.length;
}
/** A memory's nonzero 64 KiB pages, copied, as `{ length, runs }`, less
 *  the pages in `skip` (by offset). */
/** A restored session's answer to a snapshot request. */
const SNAP_RESTORED = 'a restored session takes no snapshot: a new one comes from a fresh boot that records';
function snapTake(buffer, skip = null) {
  const P = 65536, length = buffer.byteLength;
  const words = new Int32Array(buffer, 0, length >>> 2);
  const spans = [];
  for (let p = 0; p < length; p += P) {
    if (skip?.has(p)) continue;
    let z = true;
    for (let i = p >>> 2, e = (p + P) >>> 2; i < e; i++) if (words[i] !== 0) { z = false; break; }
    if (z) continue;
    const last = spans.at(-1);
    if (last && last[0] + last[1] === p) last[1] += P;
    else spans.push([p, P]);
  }
  return { length, runs: spans.map(([at, len]) => [at, new Uint8Array(buffer, at, len).slice()]) };
}
/** Nanoseconds since the epoch, as every Worker's clock reads it (gex.mjs
 *  `hostNowNs`), as a decimal string. */
function snapNow() {
  return String(BigInt(Math.round(performance.timeOrigin * 1000)) * 1000n + BigInt(Math.round(performance.now() * 1e6)));
}
function snapFill(memory, m, sources = null) {
  const to = new Uint8Array(memory.buffer);
  if (!m.runs) to.set(m);
  else for (const [at, bytes] of m.runs) to.set(bytes, at);
  for (const [at, len, si, off] of m.refs ?? []) {
    const [kind, name] = m.sources[si];
    const b = sources?.read(kind, name, off, len);
    if (!b || b.length !== len) throw new Error(`restore: ${name} has no ${len} bytes at ${off} (the snapshot's source)`);
    to.set(b, at);
  }
}
/** Whether two byte ranges hold the same bytes. */
const snapSame = globalThis.Buffer
  ? (a, b) => globalThis.Buffer.from(a.buffer, a.byteOffset, a.length).equals(globalThis.Buffer.from(b.buffer, b.byteOffset, b.length))
  : (a, b) => {
    if (a.length !== b.length) return false;
    const n = a.length >>> 2;
    if ((a.byteOffset | b.byteOffset) & 3) { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
    const x = new Int32Array(a.buffer, a.byteOffset, n), y = new Int32Array(b.buffer, b.byteOffset, n);
    for (let i = 0; i < n; i++) if (x[i] !== y[i]) return false;
    for (let i = n << 2; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };
/** The kernel's list of where file bytes lie (hosts/kernel
 *  `k_snap_extents`), at its header `h` in `buffer`: `[kind, address,
 *  length, offset, name]`, empty when no thread made one. */
function snapExtents(buffer, h) {
  if (!h) return [];
  const w = new Int32Array(buffer), i = h >>> 2;
  if (Atomics.load(w, i + 1) !== 1) return [];
  const at = w[i + 2] >>> 0, n = w[i + 3] >>> 0;
  const text = new TextDecoder().decode(new Uint8Array(buffer, at, n).slice());
  return text.split('\n').filter(Boolean).map((l) => {
    const [k, a, len, off, ...name] = l.split(' ');
    return [Number(k), Number(a), Number(len), Number(off), name.join(' ')];
  });
}
/** The kernel memory's pages that hold a source's bytes unchanged (a
 *  bundled file's, a lazy file's chunk), by reference: `{ skip, refs,
 *  sources }`, `refs` as runs `[offset, length, source, source offset]`.
 *  `sources.read(kind, name, offset, length)` gives a source's bytes or
 *  null. */
function snapRefs(buffer, extents, sources) {
  const P = 65536, skip = new Set(), table = [], ids = new Map(), pages = [], seen = new Set();
  if (!sources) return { skip, refs: [], sources: table };
  for (const [kind, at, len, off, name] of extents) {
    if (seen.has(at)) continue;
    const first = Math.ceil(at / P) * P, end = Math.floor((at + len) / P) * P;
    if (first >= end) continue;
    let src = null;
    try { src = sources.read(kind, name, off + (first - at), end - first); } catch { src = null; }
    if (!src || src.length !== end - first) continue;
    seen.add(at);
    let si = ids.get(`${kind} ${name}`);
    for (let p = first; p < end; p += P) {
      if (skip.has(p) || !snapSame(new Uint8Array(buffer, p, P), src.subarray(p - first, p - first + P))) continue;
      if (si === undefined) { si = table.length; table.push([kind, name]); ids.set(`${kind} ${name}`, si); }
      skip.add(p);
      pages.push([p, si, off + (p - at)]);
    }
  }
  pages.sort((x, y) => x[0] - y[0]);
  const refs = [];
  for (const [p, si, o] of pages) {
    const r = refs.at(-1);
    if (r && r[0] + r[1] === p && r[2] === si && r[3] + r[1] === o) r[1] += P;
    else refs.push([p, P, si, o]);
  }
  return { skip, refs, sources: table };
}

export class ProcessHost {
  #makeWorker; #shim; #guest; #kernel; #extra;
  /// The other memory flavour's runtime, `{ shim, guest, kernel }` (a
  /// kernel module, on the same kernel memory), for a page whose processes
  /// do not all have one kind of memory: in a 64-bit memory's page
  /// (`?mem64=1`), a program that is a wasm32 module (64-bit Wine's
  /// wineserver) or a 32-bit ELF runs in a 32-bit memory.
  /// What a process gets: the exec record says (`mem64`, vfork.rs
  /// `exec_memory`), a fork child has its parent's, and anything else the
  /// page's. As the X server's (runtime.js), which takes the wasm32 kernel.
  #other; #otherShim = null; #otherLoading = null;
  /// Whether the other flavour's runtime is here, for an exec that asks for
  /// it to wait on before it is past failing (vfork.rs `exec_unrunnable`):
  /// 0 not asked (or asked again), 2 loading, 1 loaded, -2 (`-ENOENT`) the last try failed, which
  /// an exec sets back to 0 to ask again (`#loadOther` retries after a
  /// backoff). Only for one fetched on first use, and only with
  /// SharedArrayBuffer.
  #otherGate = null; #otherError = null; #otherRetryAt = 0; #otherBackoff = 0; #otherSaid = false;
  /// Each process's memory flavour, by pid, where it is not the page's.
  #flavourOf = new Map();
  /// Per Worker, a shared counter of the output messages this host has
  /// delivered. A Worker about to publish its process's exit waits until the
  /// count reaches what it sent (see `drainOutput` in worker.mjs): output from
  /// different Workers crosses different ports, which nothing orders, so a
  /// parent that waited for its child could otherwise print first.
  #outputAcks = new WeakMap();
  #outputAck(w) {
    // Fibers (the single-thread build) deliver in order through one queue,
    // and could not wait for an ack on the thread that delivers it anyway.
    if ((SP_CODE && single.on) || typeof SharedArrayBuffer === 'undefined') return undefined;
    const a = new Int32Array(new SharedArrayBuffer(4));
    this.#outputAcks.set(w, a);
    return a.buffer;
  }
  /// One counter for every Worker's output (`outputSeq` in worker.mjs): a
  /// write takes the next number as it happens, and output is delivered in
  /// that order. Two processes' writes to the page's stdout cross different
  /// ports, which nothing orders: a child that printed and then wrote to a
  /// socket could be overtaken by its parent, which read the socket and then
  /// printed. A number taken whose output never comes (its Worker gone in
  /// between) holds the rest for OUTPUT_GAP_MS at most.
  #outputSeq = null;
  #outputNext = 0;
  #outputHeld = new Map();
  #outputGap = null;
  #outputSeqBuf() {
    if ((SP_CODE && single.on) || typeof SharedArrayBuffer === 'undefined') return undefined;
    this.#outputSeq ??= new Int32Array(new SharedArrayBuffer(4));
    return this.#outputSeq.buffer;
  }
  /** `deliver` for output numbered `seq` (none: at once), once all before it are. */
  #inOrder(seq, deliver) {
    if (typeof seq !== 'number') {
      deliver();
      return;
    }
    this.#outputHeld.set(seq, deliver);
    this.#outputFlush();
  }
  #outputFlush() {
    let moved = false;
    for (let d; (d = this.#outputHeld.get(this.#outputNext));) {
      this.#outputHeld.delete(this.#outputNext);
      this.#outputNext = (this.#outputNext + 1) | 0;
      d();
      moved = true;
    }
    // The wait is for the number now first missing: one that came starts it again.
    if (moved && this.#outputGap) {
      clearTimeout(this.#outputGap);
      this.#outputGap = null;
    }
    if (!this.#outputHeld.size || this.#outputGap) return;
    // Held output is what arrived within OUTPUT_GAP_MS of the gap: a Worker
    // writing fast meanwhile is held no longer than that.
    this.#outputGap = setTimeout(() => {
      this.#outputGap = null;
      if (!this.#outputHeld.size) return;
      // Past the numbers that never came: on from the first held one.
      let first = null;
      for (const s of this.#outputHeld.keys()) if (first === null || ((s - this.#outputNext) | 0) < ((first - this.#outputNext) | 0)) first = s;
      const lost = (first - this.#outputNext) | 0;
      if (logNamed('output')) console.warn(`[output] writes ${this.#outputNext}..${(first - 1) | 0} never came in ${OUTPUT_GAP_MS} ms (${lost} lost); ${this.#outputHeld.size} held go on`);
      if (lost > 0) this.#outputNext = first;
      this.#outputFlush();
    }, OUTPUT_GAP_MS);
    this.#outputGap.unref?.();
  }
  #onOutput; #onFork; #onExit; #readStdin; #onOther;
  #workers = new Map();
  /** The wineserver's door, once published (`doorPublish`). */
  #door = null;
  /** The pid of the server whose door is current, shared with every Worker
   *  (gex.mjs `#doorWord`): the server writes it as it publishes, before it
   *  says it is ready; cleared here when it ends. Without shared memory
   *  (the single-thread build) there are no doors. */
  #doorWord = typeof SharedArrayBuffer === 'function' ? new Int32Array(new SharedArrayBuffer(4)) : null;
  #pending = new Map();
  #first = null;
  #forked = 0;
  // Per-process compiled-module cache: `procPid -> Map<registry index,
  // WebAssembly.Module>`. A process's threads share one guest memory and so one
  // registry, so a module a sibling compiled is reusable by the rest. Relayed
  // here between the Workers of a process, and handed to a thread when it starts
  // so it need not recompile what its siblings already did. Keyed by the process
  // (the first segment of a `pid:tid` worker key); forks copy the registry into
  // a separate memory, so they do NOT share this -- the fork path carries the
  // parent's compiled modules itself. See `Gex#sharedModules`.
  #moduleCache = new Map();
  // Modules whose compiled code depends on their bytes alone (`wasm:` keys:
  // native wasm32 programs, by content), for every process of the page: an
  // exec'd program another process already ran is not compiled again. Kept
  // to the most recent PAGE_MODULES and PAGE_MODULE_BYTES. A hit is by
  // content (length plus two 32-bit hashes of every byte), never by path, so
  // a file apt replaced in place is compiled afresh.
  #pageModules = new Map();
  // Compiled translation units (content keys `len:a:b`) for every process of
  // the page, with `extra.unitCache`: a process served units another one
  // built instantiates the other's compiled code instead of compiling the
  // same bytes again. Most recent first out, to UNIT_MODULE_BYTES of wasm.
  #unitModules = new Map();
  #unitBytes = 0;
  // Function-tier units (`cgen:KEY`) and their cold clusters (`cold:KEY:C`),
  // compiled, for every process of the page: a unit's key is its content's,
  // so the Module one process compiled is the one every other would. Handed
  // to the pool as it comes in and to each Worker that starts; the oldest
  // go past UNIT_PAGE_MODULES.
  #cgenModules = new Map();
  // A wasm32 module's late-bound calls (`late:` keys, gex.mjs `lateCalls`):
  // their bytes are a function of the key alone, so one compiled is every
  // Worker's, a fork's child's above all (its process has no cache of its
  // own yet). Apart from `#pageModules`, whose count they would take; the
  // oldest go past LATE_PAGE_MODULES.
  #lateModules = new Map();
  // Parsed binding lists (`plan:` keys, gex.mjs `bindPlan`): a module's
  // lines are the same in every process binding it alike, so the plan one
  // Worker made is every Worker's. On shared buffers: handed out by
  // reference. The oldest go past PLAN_PAGE_MODULES.
  #planModules = new Map();
  // Units a Worker published and nobody has compiled yet: fetched and
  // compiled here, off the guest's threads (`prefetchUnit`), by key, with
  // the processes that asked for it meanwhile.
  #cgenFetching = new Map();
  #unitModule = null;
  // The unit cache's small files (a unit's entries, its indirect-call
  // profile, a provider's metadata) and the ones it does not have (null),
  // by path, as some Worker of the page read them: handed to every Worker
  // that starts, which then asks the server for none of them. Bytes in
  // shared memory, so a start message carries references.
  #unitFiles = new Map();
  // `#unitFiles` in one shared buffer and an index (path, offset, length;
  // -1 for a file the cache lacks), made again when a file has come since:
  // what a start message carries, one buffer rather than one per file.
  #unitPack = null;
  #unitFilePack() {
    if (!this.#unitFiles.size || typeof SharedArrayBuffer !== 'function') return undefined;
    if (this.#unitPack?.n === this.#unitFiles.size) return this.#unitPack.pack;
    let total = 0;
    for (const b of this.#unitFiles.values()) total += b?.byteLength ?? 0;
    const buf = new SharedArrayBuffer(Math.max(1, total));
    const u8 = new Uint8Array(buf);
    const index = [];
    let at = 0;
    for (const [rel, b] of this.#unitFiles) {
      if (!b) { index.push([rel, 0, -1]); continue; }
      u8.set(b, at);
      index.push([rel, at, b.byteLength]);
      at += b.byteLength;
    }
    this.#unitPack = { n: this.#unitFiles.size, pack: { buf, index } };
    return this.#unitPack.pack;
  }
  /** `?filltime=1`: at most once a minute, how many modules the relay
   *  holds on this thread, and how many it has been handed. */
  #relayCensus() {
    this.#relayed++;
    const now = perf.now();
    if (now < this.#relayCensusAt) return;
    this.#relayCensusAt = now + 60000;
    let n = 0, bytes = 0;
    for (const c of this.#moduleCache.values()) { n += c.size; bytes += c.bytes ?? 0; }
    console.warn(`[host] relay holds ${n} process modules (${(bytes / 1048576).toFixed(0)} MiB), ${this.#pageModules.size} programs, `
      + `${this.#unitModules.size} units (${(this.#unitBytes / 1048576).toFixed(0)} MiB); ${this.#relayed} handed to it so far`);
  }
  #relayed = 0;
  #relayCensusAt = 0;

  /** What a starting Worker gets: its process's modules and the page's. */
  #sharedFor(pid, withProcess, w = null) {
    const own = withProcess ? [...(this.#moduleCache.get(String(pid).split(':')[0])?.entries() ?? [])] : [];
    // And the page's translation units (empty unless `extra.unitCache`),
    // less those the Worker was given while it waited in the pool.
    const page = [...this.#pageModules.entries(), ...this.#unitModules.entries(), ...this.#cgenModules.entries(), ...this.#lateModules.entries(), ...this.#planModules.entries()];
    const had = w?.[POOLED];
    return [...own, ...(had ? page.filter(([k]) => !had.has(k)) : page)];
  }

  /** Give a pooled Worker the page's modules now, while it waits: in the
   *  start message they were most of a fork's hand-off (about 860 on the
   *  demo after a few commands; 2.5 ms to clone in, 1.1 ms more in transit
   *  and 0.7 ms to register, of every child's start). */
  #poolModules(w) {
    const entries = [...this.#pageModules.entries(), ...this.#unitModules.entries(), ...this.#cgenModules.entries(), ...this.#lateModules.entries(), ...this.#planModules.entries()];
    w[POOLED] = new Set(entries.map(([k]) => k));
    if (entries.length) w.postMessage({ kind: 'pooledModules', entries });
  }

  /**
   * Have a pooled Worker make its runtime's instances (kernel, shim, guest)
   * on a fresh memory while it waits (worker.mjs `preattach`), for an exec'd
   * posix_spawn child to start on. On the kernel stack slice its start
   * message will name, taken now. Only with a global kernel whose stacks are
   * known, and shared memory (the build every Worker of a page is); nothing
   * of a process is in it, so a Worker taken for anything else drops it.
   * About what a fresh process's memory holds before it runs: the shim's
   * data and the Worker's shim stack.
   *
   * One pooled Worker at a time has it (`#preattachOne`), and only an exec'd
   * child is given that one (`#takeWorker`): its memory is grown to 177 MiB
   * at once, and a Worker taken for a thread or a fork dropped it unfreed --
   * the Worker then sits in wasm or a wait, its isolate never collects, and
   * the dropped memory stays in the renderer's VmData (one program: 24 of them
   * alive at its window, 4.15 GiB; a desktop shell: 58, 10 GiB).
   */
  #preattach(w) {
    if (!this.#kernel || !this.#stacks || !this.#extra?.shared || w[STACK_KEY]) return;
    const key = `pool:${++this.#poolSeq}`;
    let stack;
    try {
      stack = this.#nextKernelStack(key);
    } catch {
      return;
    }
    w[STACK_KEY] = key;
    this.#shimModule().then((shim) => w.postMessage({
      kind: 'preattach',
      shim,
      guest: this.#guest,
      kernel: this.#kernel,
      kernelStack: stack,
      shared: true,
      // The process memory's index type: the page's (`?mem64=1`, run-node's
      // GEX_MEM64), which every pooled Worker has (`#other`).
      mem64: !!this.#extra?.mem64,
      pageFlavours: this.#pageFlavours(),
      otherGate: this.#otherGate?.buffer,
      otherLoad: this.#otherLoadFn(),
    }));
  }
  #poolSeq = 0;

  /** A runtime made ahead in one pooled Worker, unless one already has it. */
  #preattachOne() {
    if (this.#warm.some((w) => w[STACK_KEY] !== undefined)) return;
    const w = this.#warm.find((x) => x[READY]);
    if (w) this.#preattach(w);
  }

  /** The kernel stack slice for the Worker `w` starting `key`: the one it
   *  was given in the pool (`#preattach`), now `key`'s, or a new one. */
  #stackFor(w, key) {
    const pooled = w[STACK_KEY];
    if (pooled === undefined) return this.#nextKernelStack(key);
    delete w[STACK_KEY];
    const slice = this.#stackOf.get(pooled);
    this.#stackOf.delete(pooled);
    this.#stackOf.set(key, slice);
    return this.#stacks.area + slice * this.#stacks.size;
  }

  /** Unit module `key`, compiled for process `proc` (by its Worker `from`,
   *  or here): kept for the page, handed to that process's other Workers
   *  and the pool now, and to every Worker that starts. Not to the rest:
   *  a post per module per Worker of the page held up this thread, which
   *  every fork's hand-off waits on, for a second at a time. */
  #cgenShare(key, mod, from, proc) {
    if (this.#cgenModules.has(key)) return;
    this.#cgenModules.set(key, mod);
    this.#cgenToProc(key, mod, proc, from);
    this.#poolModule(key, mod);
    while (this.#cgenModules.size > UNIT_PAGE_MODULES) this.#cgenModules.delete(this.#cgenModules.keys().next().value);
  }
  /** Unit module `key` to process `proc`'s live Workers but `from`. */
  #cgenToProc(key, mod, proc, from = null) {
    for (const [k, other] of this.#workers) {
      if (other !== from && String(k).split(':')[0] === proc) other.postMessage({ kind: 'sharedModule', key, mod });
    }
  }
  /** A module the page just kept, to the Workers waiting in the pool. */
  #poolModule(key, mod) {
    for (const w of this.#warm) {
      if (!w[POOLED] || w[POOLED].has(key)) continue;
      w[POOLED].add(key);
      w.postMessage({ kind: 'pooledModules', entries: [[key, mod]] });
    }
  }
  /// When each process was asked for. See `#spawn`.
  #asked = new Map();
  /** A process whose first Worker's thread went with the old program when
   *  another thread called `execve` (Linux's de_thread): the process runs on,
   *  and ends, in that thread's Worker. Per pid, whichever of the two exits
   *  arrives first waits here for the other: `{ resolve }` from the first
   *  Worker, `{ status }` from the thread Worker that carried it. */
  #carried = new Map();
  #threads = 0;
  /** Each thread Worker's `ref` (`#threadWorker`): the key its messages
   *  come under, which `#park` renames. */
  #refOf = new WeakMap();
  /** Spare thread Workers waiting for their process's next thread, oldest
   *  first: `{ proc, key, w, ref, gen }` (`#startSpare`). */
  #spares = [];
  #spareSeq = 0;
  /** What a process offered for its spares (worker.mjs `offerSpare`), by pid. */
  #spareRec = new Map();
  /** How many threads each process with a spare record has started. */
  #threadsOf = new Map();
  // The global kernel's per-Worker stacks, as the first Worker reported them.
  // Each Worker's kernel instance has its own `__stack_pointer` global, all
  // pointing into the shared memory, so each needs its own slice or two Workers
  // in the kernel overwrite each other's locals.
  #stacks = null;
  #stacksTaken = 0;
  // Slices handed back by ended Workers, and which slice each live Worker
  // holds: a run of many short-lived threads uses as many slices as run at
  // once, not as many as it ever made.
  #stacksFree = [];
  /// The shim compiling, once, so no Worker waits on a second compile.
  #shimCompiling = null;
  /// Workers made before anybody asks, so a fork's ~170ms of Worker startup is
  /// off the path. One spare suits a shell; a pipeline wants more, and an empty
  /// pool falls back to making a Worker inline.
  #warm = [];
  #warmWanted = 3;
  /// Exited processes' memories, zeroed (`Gex#recycleMemory`), for fork
  /// children to adopt into: their pages are touched, where a new memory's
  /// first touch was most of the cost of copying a fork's image in. Kept
  /// newest last, up to RECYCLE_COUNT and RECYCLE_BYTES, for RECYCLE_KEEP_MS
  /// after the last one came.
  #recycled = [];
  #recycleTimer = null;
  /** TEST_FORK_RECORDS: each fork child's record, weakly, by pid. */
  #forkRecords = new Map();
  /** TEST_FORK_RECORDS: said a moment after the child's Worker has its
   *  record. A scope of its own, so the timer holds nothing of the caller's. */
  #checkForkRecord(pid) {
    const record = this.#forkRecords.get(pid);
    if (!record) return;
    this.#forkRecords.delete(pid);
    setTimeout(() => {
      globalThis.gc?.();
      console.error(`[test] pid ${pid}: fork record ${record.deref() ? 'kept' : 'released'} while it runs`);
    }, 500);
  }
  #warmMade = 0;
  #warmUsed = 0;
  #stackOf = new Map();

  constructor({
    makeWorker,
    shim,
    guest,
    kernel = null,
    // Merged into every Worker's start message: everything a process needs
    // that is not the program -- GPU channel, proxy address, terminal size.
    extra = {},
    onOutput = () => {},
    onFork = () => {},
    // Every process's exit message, whole. `start` resolves with pid 1's
    // status; this hands a host the rest of what a Worker reports on exit.
    // A process that threw comes here too, with `{ threw: true, message }`
    // and none of an exit's numbers.
    onExit = () => {},
    readStdin = null,
    // Anything a Worker posts that this file has no meaning for.
    onMessage = () => {},
    // The other memory flavour's `{ shim, guest, kernel }` (`#other`), or
    // an async function that fetches it.
    other = null,
    // `(url) => Promise<WebAssembly.Module>`: how a unit `prefetchUnit` names
    // is fetched and compiled, when not with fetch and WebAssembly.compile
    // here (a compressed one, decoded elsewhere).
    unitModule = null,
  }) {
    this.#other = other;
    if (typeof other === 'function' && typeof SharedArrayBuffer === 'function') this.#otherGate = new Int32Array(new SharedArrayBuffer(4));
    this.#unitModule = unitModule;
    this.#makeWorker = makeWorker;
    this.#shim = shim;
    this.#guest = guest;
    this.#kernel = kernel;
    // Only a global kernel has a store to save.
    this.#saveGate = kernel ? xcGate() : undefined;
    this.#onOutput = onOutput;
    this.#onFork = onFork;
    this.#onExit = onExit;
    this.#readStdin = readStdin;
    this.#extra = extra;
    this.#onOther = onMessage;
  }

  /**
   * The shim, compiled once and shared to every Worker. `postMessage` clones a
   * `Uint8Array` (8.7 MB each) but clones a `WebAssembly.Module` by reference.
   */
  #shimModule() {
    if (this.#shim instanceof WebAssembly.Module) return Promise.resolve(this.#shim);
    // Async, not a preference: Chromium refuses a synchronous compile over 8 MB
    // on the main thread and the shim is 8.7. Node allows it, so the sync form
    // passed every test here and killed the page at boot.
    this.#shimCompiling ??= WebAssembly.compile(this.#shim).then((m) => (this.#shim = m));
    return this.#shimCompiling;
  }

  /** Whether process `pid` (or a thread's `pid:tid`) has a 64-bit memory. */
  #mem64(pid) {
    const f = this.#flavourOf.get(Number(String(pid).split(':')[0]));
    return f ?? !!this.#extra?.mem64;
  }

  /** The page's memory flavours, for the shim (vfork.rs
   *  `gex_set_page_flavours`): bit 0 its own is a 64-bit memory, bit 1 it
   *  runs the other too (`#other`), so that an exec may ask for that one. */
  #pageFlavours() {
    return (this.#extra?.mem64 ? 1 : 0) | (this.#other && this.#kernel ? 2 : 0);
  }

  /** Whether `pid` runs on the other flavour's runtime (`#other`). */
  #foreign(pid) {
    return this.#mem64(pid) !== !!this.#extra?.mem64;
  }

  /** What a Worker of process `pid` starts on: `{ shim, guest, kernel,
   *  mem64 }`, the shim compiled. */
  async #runtimeFor(pid) {
    const mem64 = this.#mem64(pid);
    if (!this.#foreign(pid)) {
      return { shim: await this.#shimModule(), guest: this.#guest, kernel: this.#kernel ?? undefined, mem64 };
    }
    const o = await this.#loadOther();
    if (!o) throw new Error(`process ${pid} needs a ${mem64 ? '64' : '32'}-bit memory, which this page does not run`);
    return {
      shim: await this.#otherShim,
      guest: o.guest,
      kernel: this.#kernel ? { ...this.#kernel, module: o.kernel } : undefined,
      mem64,
    };
  }

  /** The other flavour's runtime, its shim compiled: fetched on first use
   *  when given as a function (few pages run one). A fetch that fails is
   *  tried again on the next request once a backoff has passed (1 s,
   *  doubling to a minute), and until then a request fails at once: a
   *  dropped connection does not cost the page the flavour for good. The
   *  gate says how the last try went, and the first failure is said on the
   *  console. */
  #loadOther() {
    if (!this.#otherLoading && this.#otherError) {
      if (performance.now() < this.#otherRetryAt) {
        this.#answerOther(-2);
        return Promise.reject(this.#otherError);
      }
      this.#otherError = null;
    }
    if (this.#otherLoading) {
      // Asked again after it settled (a page with no other runtime answers
      // -2 for good): the asker waits on the gate.
      this.#otherLoading.then((o) => this.#answerOther(o ? 1 : -2), () => {});
      return this.#otherLoading;
    }
    const p = (async () => {
      const o = typeof this.#other === 'function' ? await this.#other() : this.#other;
      if (o) this.#otherShim ??= o.shim instanceof WebAssembly.Module ? Promise.resolve(o.shim) : WebAssembly.compile(o.shim);
      if (o) await this.#otherShim;
      return o;
    })();
    this.#otherLoading = p;
    if (this.#otherGate) Atomics.store(this.#otherGate, 0, 2);
    p.then((o) => {
      this.#otherBackoff = 0;
      this.#answerOther(o ? 1 : -2);
    }, (e) => {
      this.#otherLoading = null;
      this.#otherShim = null;
      this.#otherError = e;
      this.#otherBackoff = Math.min(Math.max(this.#otherBackoff * 2, OTHER_RETRY_MS), OTHER_RETRY_MAX_MS);
      this.#otherRetryAt = performance.now() + this.#otherBackoff;
      if (!this.#otherSaid) {
        this.#otherSaid = true;
        console.error(`gex: the other memory flavour's runtime could not be loaded, so no program that needs it runs until a later try gets it: ${e?.message ?? e}`);
      }
      this.#answerOther(-2);
    });
    return p;
  }

  /** The single-thread build's word on the other flavour's runtime, where
   *  no shared gate can be: its fibers share this realm, so an exec that
   *  needs the runtime calls this and waits for its promise (gex.mjs
   *  `other_runtime`): 1 once it is here, -2 when it cannot be had now. */
  #otherLoadFn() {
    if (!(SP_CODE && single.on) || !this.#other || !this.#kernel) return undefined;
    return (this.#otherLoadF ??= () => this.#loadOther().then((o) => (o ? 1 : -2), () => -2));
  }
  #otherLoadF = null;

  /** The gate's answer for the execs waiting on it (gex.mjs `other_runtime`). */
  #answerOther(s) {
    if (!this.#otherGate) return;
    Atomics.store(this.#otherGate, 0, s);
    Atomics.notify(this.#otherGate, 0);
  }

  /** How many children the guest has forked. */
  get forked() { return this.#forked; }

  /** How many threads its processes have started in Workers of their own. */
  get threads() { return this.#threads; }

  /** Every live process, by pid. Pid 1 is the one `start` was given. */
  get pids() { return [...this.#workers.keys()]; }

  /**
   * Where the global kernel's per-Worker stacks are (`Gex#kernelStacks`'s
   * shape), said by whatever instance of it came first when no process of
   * this host did: a page with no pid 1 of its own. Before this no Worker
   * is given a stack, so none may start.
   */
  useKernelStacks(stacks) {
    if (this.#stacks) return;
    this.#stacks = stacks;
    this.#preattachOne();
  }

  /**
   * Start process `pid` (the global kernel's `k_spawn` made it) on a
   * program, from no parent Worker: the record a posix_spawn child's exec
   * hands its new Worker (`Gex#vforkExecRecord`, run by `#adoptExec`), with
   * the page's own parts. `exec` is the command (vfork.rs
   * `stage_exec_parts`), `fds` its descriptors (`adopt_fds_text` lines, each
   * end already held for `pid`), `opts` its run options with the identity.
   * Resolves with its exit status.
   */
  spawnExec({ pid, exec, fds, cwd = '/', wasmImage = null, opts = {} }) {
    this.#flavourOf.delete(pid);
    const child = {
      pid,
      // Its first thread's registers, at the first context
      // (`gex_abi::ctx_addr(0)`), as a loaded program's are.
      ctx: 0x1000n,
      pc: 0n, sp: 0n, tls: 0n, compat: 0, wasm32: 0n, brk: 0n, windowsSpace: 0,
      image: [], vmaBlob: null, slots: 0, pages: null, shared: null, modules: [],
      opts, exec, wasmImage, fds, cwd, state: new Uint8Array(0), exe: null,
    };
    return this.#spawn(pid, { kind: 'adopt', child });
  }

  /**
   * Run `elf` as pid 1 and resolve with its exit status. Children are stopped
   * when it exits -- a page done with a program is done with its daemons.
   */
  async start({ elf, files = {}, args = [], opts = {}, transfer = [] }) {
    watchLibraries();
    // One of the few default lines: what pid 1 runs. Its execs say their own
    // (the shim's `do_exec`).
    const what = [opts.argv0, ...args].filter((a) => typeof a === 'string' && a).join(' ');
    if (logRoutine()) inform(`[gex] pid 1: start ${what.length > 200 ? `${what.slice(0, 200)}...` : what || '(program)'}`);
    const status = await this.#spawn(1, {
      kind: 'run',
      elf,
      files,
      args,
      opts,
      transfer,
    });
    await this.stop();
    return status;
  }

  /** Deliver a line of standard input to pid 1. */
  pushStdin(bytes) {
    this.#first?.postMessage({ kind: 'stdin', bytes });
  }

  /** Say there will be no more input. */
  endStdin() {
    this.#first?.postMessage({ kind: 'eof' });
  }

  /** Tell every process the terminal changed size. */
  resize(rows, cols) {
    for (const w of this.#workers.values()) w.postMessage({ kind: 'winsize', rows, cols });
  }

  /**
   * A session snapshot: every thread of every process stopped where it waits
   * (the snapshot flavour's programs, libgex-snap), the global
   * kernel's memory and each process's memory and module store copied, and
   * the threads let go on (`resume`) or stopped. Each thread is one of this
   * host's Workers; a Worker whose thread does not stop within `timeout` ms
   * (one in a call that is not left so, computing, or not built so) refuses
   * the snapshot, and the threads go on. `{ kernel, memories, stores, meta,
   * stopMs, copyMs }` (a memory as its nonzero runs, `snapTake`), or
   * `{ refused, stopMs }`.
   */
  async snapshot({ timeout = 10000, resume = true, sources = null } = {}) {
    if (this.#restored) return { refused: SNAP_RESTORED, stopMs: 0 };
    const st = this.#stacks;
    if (!this.#kernel?.memory || !st?.snapWord) throw new Error('snapshot: no global kernel with a snapshot word');
    if (this.#pending.size) return { refused: `still starting: ${[...this.#pending.keys()].join(' ')}`, stopMs: 0 };
    const want = [...this.#workers.keys()].map(String);
    const parked = new Map();
    const words = new Int32Array(this.#kernel.memory.buffer);
    const wake = () => {
      const base = st.wsAddr >>> 2;
      for (let i = 0; i < st.wsCount; i++) {
        Atomics.add(words, base + i, 1);
        Atomics.notify(words, base + i);
      }
    };
    const t0 = performance.now();
    let timer;
    const all = new Promise((res) => {
      this.#snapWait = { parked, check: () => { if (want.every((k) => parked.has(k))) res(true); } };
      timer = setTimeout(() => res(false), timeout);
    });
    Atomics.store(words, st.snapWord >>> 2, 1);
    wake();
    this.#snapWait.check();
    const ok = await all;
    // when the session stopped, on the clock every Worker shares
    // (`hostNowNs`): a restored one skips the time since
    const at = snapNow();
    clearTimeout(timer);
    this.#snapWait = null;
    const stopMs = performance.now() - t0;
    const release = () => {
      if (st.extents) for (let i = 0; i < 4; i++) Atomics.store(words, (st.extents >>> 2) + i, 0);
      Atomics.store(words, st.snapWord >>> 2, 0);
      Atomics.notify(words, st.snapWord >>> 2);
      wake();
    };
    if (!ok) {
      release();
      return { refused: `not stopped: ${want.filter((k) => !parked.has(k)).join(' ')}`, stopMs };
    }
    // The kernel's pages that still hold a source's bytes (the page's
    // bundle, its lazy files' chunks), by reference; the rest copied.
    const kb = this.#kernel.memory.buffer;
    const byRef = snapRefs(kb, snapExtents(kb, st.extents), sources);
    const kernel = snapTake(kb, byRef.skip);
    kernel.refs = byRef.refs;
    kernel.sources = byRef.sources;
    for (const [at, bytes] of kernel.runs) {
      const dv = new DataView(bytes.buffer);
      if (st.snapWord >= at && st.snapWord < at + bytes.length) dv.setInt32(st.snapWord - at, 0, true);
      if (st.extents) for (let i = 0; i < 16; i += 4) if (st.extents + i >= at && st.extents + i < at + bytes.length) dv.setInt32(st.extents + i - at, 0, true);
    }
    const memories = new Map(), stores = new Map(), maxPages = {};
    const threads = [];
    for (const key of want) {
      const m = parked.get(key);
      const pid = Number(key.split(':')[0]);
      if (!memories.has(pid)) {
        memories.set(pid, snapTake(m.memory.buffer));
        maxPages[pid] = m.maxPages;
        if (m.moduleStore) {
          const used = Number(Atomics.load(new BigInt64Array(m.moduleStore, 0, 1), 0));
          stores.set(pid, new Uint8Array(m.moduleStore, 0, used).slice());
        }
      }
      threads.push({ key, pid, tid: m.tid, pc: m.entry, sp: m.start, main: !key.includes(':') });
    }
    const meta = {
      version: 1,
      stacks: { ...st },
      threads,
      maxPages,
      flavourOf: [...this.#flavourOf],
      at,
      // what the references are to (`restore` takes only the same)
      bundle: kernel.refs.length ? sources.id ?? null : null,
      door: this.#door ? { ...this.#door, memory: undefined, module: undefined } : null,
    };
    const copyMs = performance.now() - t0 - stopMs;
    if (resume) release();
    else this.stop();
    return { kernel, memories, stores, meta, stopMs, copyMs };
  }
  #snapWait = null;

  /**
   * Run a session from a snapshot (`snapshot`) in place of `start`: the global
   * kernel's memory as it held it (this host's kernel memory, before any
   * Worker attaches), each process's memory and module store, and every
   * thread in a Worker of its own, entering where its libc's record says
   * (`Gex#snapThread`). A process's first Worker comes up before its others
   * (it frees the snapshot's shim slots). Resolves with pid 1's status;
   * `onRunning(ms)` once every thread's Worker is up.
   */
  /** Whether this session can take a snapshot: not once it is a restored
   *  one (its programs run without the recording's code), so a page shows
   *  no snapshot control then. */
  get snapshotAllowed() { return !this.#restored; }
  #restored = false;
  async restore({ kernel, memories, stores = new Map(), meta }, { onRunning, sources = null } = {}) {
    const t0 = performance.now();
    this.#restored = true;
    watchLibraries();
    if (!this.#kernel?.memory) throw new Error('restore: no global kernel');
    if (kernel.refs?.length && (!sources || (sources.id ?? null) !== meta.bundle)) {
      throw new Error(`restore: the snapshot's pages are by reference to bundle ${meta.bundle}, and this session's is ${sources?.id ?? 'none'}`);
    }
    const km = this.#kernel.memory;
    const grow = Math.ceil((snapLength(kernel) - km.buffer.byteLength) / 65536);
    if (grow > 0) km.grow(grow);
    snapFill(km, kernel, sources);
    // Every restored thread waits, attached, until all are (`snapRelease`).
    const words = new Int32Array(km.buffer), word = meta.stacks.snapWord >>> 2;
    Atomics.store(words, word, 1);
    this.#stacks = meta.stacks;
    for (const [p, f] of meta.flavourOf ?? []) this.#flavourOf.set(p, f);
    inform(`[gex] restore: ${meta.threads.length} threads of ${memories.size} processes`);
    const byPid = new Map();
    for (const t of meta.threads) {
      if (!byPid.has(t.pid)) byPid.set(t.pid, []);
      byPid.get(t.pid).push(t);
    }
    let first = null;
    const ups = [];
    // The wineserver's process first: its Worker publishes the door
    // (`Gex#snapThread`), which the others' start messages then carry.
    const order = [...byPid].sort(([a], [b]) => (b === meta.door?.pid) - (a === meta.door?.pid));
    for (const [pid, ts] of order) {
      const bytes = memories.get(pid);
      const memory = wasmMemory({ initial: snapLength(bytes) / 65536, maximum: meta.maxPages[pid], shared: true }, 'a restored process memory', this.#kernel?.spare);
      snapFill(memory, bytes);
      let moduleStore;
      const sb = stores.get(pid);
      if (sb) {
        moduleStore = new SharedArrayBuffer(Math.max(1 << 20, sb.length), { maxByteLength: 4 * 1024 * 1024 * 1024 });
        new Uint8Array(moduleStore).set(sb);
      }
      if (MEMORIES) (globalThis.__gexMemories ??= new Map()).set(pid, memory);
      const main = ts.find((t) => t.main);
      if (!main) throw new Error(`restore: pid ${pid} has no first thread`);
      const one = (t, reset, door) => {
        let up;
        const attached = new Promise((r) => { up = r; });
        this.#snapUp.set(t.key, up);
        const done = this.#spawn(t.main ? pid : t.key, {
          kind: 'thread', tid: t.tid, memory, moduleStore,
          ctx: 0n, pc: BigInt(t.pc),
          snap: { pc: t.pc, sp: t.sp, door }, snapMain: t.main || undefined, snapThreads: ts.length, snapReset: reset || undefined,
        });
        return { attached, done };
      };
      const m = one(main, true, pid === meta.door?.pid ? meta.door.area : undefined);
      if (pid === 1) first = m.done;
      else m.done.catch((e) => console.error(`[gex] restore: pid ${pid}: ${e?.message ?? e}`));
      await m.attached;
      for (const t of ts) if (t !== main) ups.push(one(t, false).attached);
    }
    if (!first) throw new Error('restore: no pid 1');
    // All up: the session goes on, its monotonic clocks held back the time
    // since it stopped (every Worker's, as each is let go).
    Promise.all(ups).then(() => {
      const held = meta.at ? BigInt(snapNow()) - BigInt(meta.at) : 0n;
      if (held > 0n && meta.stacks.monoHeld) Atomics.add(new BigUint64Array(km.buffer), meta.stacks.monoHeld >>> 3, held);
      Atomics.store(words, word, 0);
      Atomics.notify(words, word);
      onRunning?.(performance.now() - t0);
    });
    const status = await first;
    this.stop();
    return status;
  }
  /** A restored thread's Worker is up (`snap-attached`): by key. */
  #snapUp = new Map();

  /** Stop every Worker, whatever it is doing -- once no process is inside a
   *  save of the translation store (xcache.mjs's gate): a Worker ended
   *  during its write loses the save. No save starts once this is called,
   *  so the wait ends. Resolves when they are stopped. */
  stop() {
    const g = this.#saveGate;
    if (g) Atomics.or(g, 0, XC_STOP);
    if (!xcBusy(g)) {
      this.#end();
      return Promise.resolve();
    }
    return (async () => {
      const t0 = performance.now();
      while (xcBusy(g) && performance.now() - t0 < SAVE_WAIT_MS) await xcSleep(5);
      if (xcBusy(g)) console.log(`[gex] xcache: a save still under way after ${SAVE_WAIT_MS / 1000} s; stopping anyway`);
      this.#end();
    })();
  }

  #saveGate;

  #end() {
    clearTimeout(this.#recycleTimer);
    this.#recycled.length = 0;
    for (const w of this.#workers.values()) w.terminate();
    this.#workers.clear();
    // Including the warm ones nobody claimed.
    for (const w of this.#warm) w.terminate();
    this.#warm = [];
    // A later `start` saves again (a Worker cut off inside a save left its count).
    if (this.#saveGate) this.#saveGate.fill(0);
  }

  /**
   * One process in one Worker. `msg` is `run` (load a program) or `adopt` (a
   * forked child, from its parent's fork record).
   */
  #spawn(pid, msg) {
    // When this process was asked for, to measure the gap before it runs.
    this.#asked.set(pid, performance.now());
    return new Promise((resolve, reject) => {
      // A pooled Worker holds the page flavour's runtime and modules: one of
      // the other flavour starts in a Worker of its own.
      const w = this.#foreign(pid) ? this.#makeWorker() : this.#takeWorker(msg.kind === 'adopt' && !!msg.child?.exec);
      this.#workers.set(pid, w);
      workerRan(w, pid);
      if (pid === 1) this.#first = w;
      const onEvent = (e) => this.#onMessage(pid, w, e.data, resolve, reject);
      const onMsg = (m) => this.#onMessage(pid, w, m, resolve, reject);
      w.addEventListener?.('message', onEvent);
      w.on?.('message', onMsg);
      const fail = (e) => {
        this.#workers.delete(pid);
        reject(e instanceof Error ? e : new Error(e?.message ?? String(e)));
      };
      w.addEventListener?.('error', fail);
      w.on?.('error', fail);
      w[LISTENING] = { onEvent, onMsg, fail };
      this.#pending.set(pid, { ...msg, pid });
      const kind = msg.kind;
      // The listeners above live as long as the process and share this
      // scope: holding `msg` here kept a fork's whole record (every module
      // its parent had, with their metadata) on this thread for the child's
      // life, and a child's record names its parent's modules too -- some
      // twenty thousand modules on a desktop's page. `#pending` has it until
      // the Worker is ready.
      msg = null;
      // `#pending` pins the whole fork record (image bytes included) until the
      // child reports ready. A child that never starts leaves ~19 MB pinned.
      // See task 145.
      if (this.#extra?.debugExit) {
        let held = 0;
        for (const r of this.#pending.values()) {
          for (const g of r.image ?? []) held += g.bytes?.byteLength ?? 0;
        }
        console.error(
          `[pending] ${this.#pending.size} records holding ${held} bytes`,
        );
      }
      // One line per process, to `debug`: a process that never starts leaves
      // no other trace -- the fork is logged by the parent, the exit by the
      // child. #137.
      narrate(`[host] pid ${pid}: worker taken, ${kind} held for ready`);
      // And now whatever it said while it was still in the pool.
      const held = w[EARLY];
      if (held) {
        delete w[EARLY];
        w.removeEventListener?.('message', held.buf);
        w.off?.('message', held.buf);
        for (const m of held.early) this.#onMessage(pid, w, m, resolve, reject);
      }
    });
  }

  /**
   * A Worker, from the pool if one is warm. Tops up after the take, so the
   * Worker started now serves the next process, not this one.
   */
  #takeWorker(exec = false) {
    // An exec'd child takes the Worker with a runtime made ahead, anything
    // else one without (`#preattach`), whichever comes first otherwise.
    let i = this.#warm.findIndex((x) => (x[STACK_KEY] !== undefined) === exec);
    if (i < 0 && this.#warm.length) i = 0;
    const w = i < 0 ? null : this.#warm.splice(i, 1)[0];
    if (w) this.#warmUsed++;
    const got = w ?? this.#makeWorker();
    this.#topUp();
    this.#preattachOne();
    return got;
  }

  /// Start Workers up to the pool size, one per call so a burst of forks does
  /// not spend its first moments making Workers it will not reach. Called on
  /// a take and again when a pooled Worker is ready, so the pool fills one
  /// Worker after another: called on a take alone, a take removed one and
  /// added one, and the pool never held more than one Worker. The second
  /// and third processes of a pipeline (`echo a | cat | cat`) then each
  /// waited about 20 ms for a Worker made at their fork (demo page).
  #topUp() {
    if (this.#warm.length >= this.#warmWanted) return;
    try {
      const w = this.#makeWorker();
      this.#pool(w, false);
      this.#warmMade++;
    } catch {
      // Cannot make another Worker: the next `#takeWorker` makes one itself.
    }
  }

  /** Into the pool: a new Worker at the back, a given-back one at the front
   *  (its JS is compiled and warm, so the next process takes it first). */
  #pool(w, givenBack) {
    // Buffer whatever it says before `#spawn` attaches its listener: a
    // pooled Worker announces itself while still sitting in the array, and
    // that message would otherwise be delivered to nobody and dropped. A
    // given-back Worker's last process may still say something on its way
    // out: nothing before its next `ready` is kept.
    const early = [];
    let readied = !givenBack;
    // A given-back Worker is ready again once it has reset.
    w[READY] = false;
    const buf = (e) => {
      const m = e && e.data !== undefined ? e.data : e;
      if (m?.kind === 'ready') readied = true;
      if (!readied) return;
      early.push(m);
      // Listening now (`ready` is posted once its handler is on).
      if (m?.kind === 'ready') {
        w[READY] = true;
        if (!w[POOLED]) this.#poolModules(w);
        if (this.#warm.includes(w)) this.#preattachOne();
      }
      // Still pooled (not taken, the host not stopped): start the next.
      if (m?.kind === 'ready' && !givenBack && this.#warm.includes(w)) this.#topUp();
    };
    w.addEventListener?.('message', buf);
    w.on?.('message', buf);
    w[EARLY] = { early, buf };
    if (givenBack) this.#warm.unshift(w);
    else this.#warm.push(w);
  }

  /**
   * A Worker whose process exited cleanly, back into the pool for the next
   * process rather than ended: a new Worker compiles and warms the runtime's
   * JS on every process's way in and out. The Worker resets its realm
   * (worker.mjs `reset`) and makes a new runtime on a new memory, as a new
   * one does. False when it is not wanted (the pool is full, the Worker has
   * run `REUSE_MAX` processes), and then it is ended as before.
   */
  #giveBack(w) {
    const n = (w[REUSES] ?? 0) + 1;
    const l = w[LISTENING];
    if (!l || n > REUSE_MAX || this.#warm.length >= this.#warmWanted + GIVEN_BACK_MAX) return false;
    delete w[LISTENING];
    w.removeEventListener?.('message', l.onEvent);
    w.off?.('message', l.onMsg);
    w.removeEventListener?.('error', l.fail);
    w.off?.('error', l.fail);
    w[REUSES] = n;
    this.#reused++;
    this.#pool(w, true);
    w.postMessage({ kind: 'reset' });
    return true;
  }
  #reused = 0;

  /// How the pool is doing, for the per-run line.
  get warmStats() {
    return { made: this.#warmMade, used: this.#warmUsed, idle: this.#warm.length, reused: this.#reused };
  }

  /// Send a Worker the closure metadata only -- which library owns which
  /// pages. Sending all compiled modules up front cost 549ms per fork; each
  /// module is fetched later, when a branch lands in it (`#sendLibrary`).
  #streamClosure(w, closure) {
    const metas = (closure ?? []).map((e) => (e instanceof Uint8Array ? e : e.meta));
    // Which program each library images, where it is one: every closure root
    // claims `PIE_BASE`, so this tells `ls`'s pages from `bash`'s.
    const programs = (closure ?? []).map((e) => (e instanceof Uint8Array ? '' : (e.program ?? '')));
    try {
      w.postMessage({ kind: 'closureMeta', metas, programs });
    } catch (e) {
      this.#onOutput(2, `gex: the closure metadata did not reach a worker: ${e.message}\n`, 0);
    }
  }

  /// Hand one compiled closure library to the Worker that asked, fetching and
  /// compiling it now on first ask. A closure entry may arrive with only
  /// metadata and a `url`; the compile is kept and reused by reference.
  async #sendLibrary(w, i) {
    const entry = (this.#extra?.closure ?? [])[i];
    if (!entry) return;
    try {
      if (!(entry instanceof Uint8Array) && !entry.mod && entry.url) {
        entry.fetching ??= (async () => {
          const t0 = performance.now();
          const res = await fetch(entry.url);
          if (!res.ok) throw new Error(`${entry.url}: ${res.status} ${res.statusText}`);
          const bytes = new Uint8Array(await res.arrayBuffer());
          const mod = await WebAssembly.compile(bytes);
          narrate(
            `[gex] closure library ${i}: ${(bytes.length / 1048576).toFixed(1)} MiB `
            + `fetched and compiled in ${Math.round(performance.now() - t0)}ms`,
          );
          return mod;
        })();
        entry.mod = await entry.fetching;
      }
      w.postMessage({
        kind: 'closureModule',
        i,
        mod: entry instanceof Uint8Array ? entry : entry.mod,
      });
    } catch (e) {
      // Loud, not fatal: a missing library costs the guest the translation it
      // would have saved, nothing else.
      console.error(`[gex] closure library ${i} did not reach a worker:`, e);
      this.#onOutput(2, `gex: closure library ${i} did not reach a worker: ${e.message}\n`, 0);
    }
  }

  #nextKernelStack(key) {
    if (!this.#stacks) return undefined;
    const reuse = this.#stacksFree.pop();
    if (reuse !== undefined) {
      this.#stackOf.set(key, reuse);
      return this.#stacks.area + reuse * this.#stacks.size;
    }
    // Slice 0 belongs to nobody: the first Worker uses the linked-in stack.
    const n = ++this.#stacksTaken;
    this.#stackOf.set(key, n);
    // The top slices are the page helpers' that take a stack of their own,
    // one each (lazy-fetch.mjs `HELPER_STACKS`: tty.mjs, hostx.js,
    // persist-worker.js, fs-opened-worker.js). Slice n's stack is the one
    // below the top `area + n * size`, so the highest a Worker may have is
    // the one under the lowest helper's.
    if (n > this.#stacks.count - HELPER_STACK_COUNT) {
      throw new Error(
        `the global kernel has ${this.#stacks.count} stacks and this is Worker ${n + 1}. ` +
          'Raise STACK_COUNT in hosts/kernel.',
      );
    }
    return this.#stacks.area + n * this.#stacks.size;
  }

  /**
   * A kernel stack slice for something that is not a guest Worker but calls
   * into the global kernel from a realm of its own (a page's drop Worker,
   * `page/src/drop.js`). Undefined until the first Worker has said
   * where the stacks are. Give it back with `releaseKernelStack(key)`.
   */
  takeKernelStack(key) {
    return this.#nextKernelStack(`ext:${key}`);
  }

  releaseKernelStack(key) {
    const slice = this.#stackOf.get(`ext:${key}`);
    if (slice === undefined) return;
    this.#stackOf.delete(`ext:${key}`);
    this.#stacksFree.push(slice);
  }

  /** Whether the Worker `key` is inside a wineserver door call: it holds
   *  the door lock. */
  #inDoor(key) {
    const d = this.#door;
    if (!d) return false;
    const i32 = new Int32Array(d.memory.buffer);
    const a = d.area >> 2;
    const parts = String(key).split(':');
    return Atomics.load(i32, a) !== 0 && i32[a + 30] === Number(parts[0]) && i32[a + 29] === Number(parts[parts.length - 1]);
  }

  /** A Worker has ended: stop it, and put its kernel stack back. `reuse`:
   *  its process exited cleanly, and it may go back to the pool instead. */
  #release(key, w, reuse = false) {
    // Not inside a door call: a Worker stopped there would leave the door
    // lock held and the server's pump waiting for it forever (a process the
    // server kills from inside its own request, as `terminate_process`
    // does). It finishes the call first; the guest it then goes back to
    // runs a moment longer, as a signal's target does.
    if (this.#inDoor(key)) {
      const t0 = Date.now();
      if (globalThis.GEX_DOOR_STATS || globalThis.process?.env?.GEX_DOOR_STATS) console.error(`[host] ${key}: ended inside a wineserver door call; stopped when it leaves`);
      const again = () => {
        if (this.#inDoor(key) && Date.now() - t0 < 10000) { setTimeout(again, 1); return; }
        if (this.#inDoor(key)) {
          // Stopped inside a door call: its handler is cut off midway, so
          // the server is poisoned (area +124) and the lock given up; the
          // pump takes it, sees the poison and ends the server, as when
          // wineserver dies.
          const i32 = new Int32Array(this.#door.memory.buffer);
          const a = this.#door.area >> 2;
          console.error(`[host] ${key}: still inside a wineserver door call after 10 s; stopped, and the server is poisoned`);
          Atomics.store(i32, a + 31, i32[a + 29] || -1);
          Atomics.store(i32, a, 0);
          Atomics.notify(i32, a);
        }
        this.#releaseNow(key, w);
      };
      setTimeout(again, 1);
      return;
    }
    this.#releaseNow(key, w, reuse);
  }

  /** Start thread Worker `key` of process `pid` from its start record
   *  `rec`, sent when the Worker is ready. Its messages are delivered under
   *  the returned `ref.key`, which a spare's thread renames. */
  #threadWorker(pid, key, rec) {
    // From the warm pool, as a process's: a fresh Worker was ~100 ms of
    // script load and shim instantiation before the thread could start,
    // and a starting Wine process makes its threads one after another
    // (61 of them before one program's first window, 13.5 s in all).
    const w2 = this.#foreign(pid) ? this.#makeWorker() : this.#takeWorker();
    this.#workers.set(key, w2);
    workerRan(w2, key);
    // Keyed by the thread, so its end stops the right Worker and frees the
    // right kernel stack.
    const ref = { key };
    this.#refOf.set(w2, ref);
    const deliver = (mm) => this.#onMessage(ref.key, w2, mm, () => {}, () => {});
    w2.addEventListener?.('message', (e) => deliver(e.data));
    w2.on?.('message', deliver);
    // A thread that throws, said out loud: a thread Worker had no `error`
    // listener, so an exception was swallowed and read as a hang.
    const threadFailed = (e) => {
      this.#onOutput(2, `gex: thread ${ref.key} failed: ${e?.message ?? String(e)}\n`, pid);
      crashed(`thread ${ref.key}: ${e?.message ?? String(e)}`);
    };
    w2.addEventListener?.('error', threadFailed);
    w2.on?.('error', threadFailed);
    // Sent from the `ready` case, which finds it here.
    this.#pending.set(key, rec);
    // A pooled Worker's 'ready' came while it sat in the pool: deliver it
    // now that `#pending` names this thread.
    const held = w2[EARLY];
    if (held) {
      delete w2[EARLY];
      w2.removeEventListener?.('message', held.buf);
      w2.off?.('message', held.buf);
      for (const mm of held.early) deliver(mm);
    }
    return { w: w2, ref };
  }

  /** A Worker for process `proc`'s next thread, started on what the process
   *  offered (`threadSpare`): it instantiates the process's modules and
   *  waits (`Gex#adoptThread`). The page's oldest spare goes past SPARE_MAX. */
  #startSpare(proc) {
    const rec = this.#spareRec.get(proc);
    if (!rec) return;
    const key = `${proc}:spare${++this.#spareSeq}`;
    const { w, ref } = this.#threadWorker(proc, key, {
      kind: 'thread',
      spare: true,
      memory: rec.memory,
      moduleStore: rec.moduleStore,
      compat: rec.compat,
      procOpts: rec.procOpts,
      park: !(SP_CODE && single.on),
    });
    const sp = { proc, key, w, ref, gen: rec.execGen };
    this.#spares.push(sp);
    this.#spareIdle(sp);
    if (logNamed('spare')) console.warn(`[spare] ${key}: started (${this.#spares.length} waiting on the page)`);
    while (this.#spares.length > SPARE_MAX) this.#letSpareGo(this.#spares.shift());
  }

  /** Spare `sp` let go when SPARE_IDLE_MS passes without a thread for it. */
  #spareIdle(sp) {
    const { key } = sp;
    const idle = setTimeout(() => {
      const i = this.#spares.indexOf(sp);
      if (i < 0) return;
      this.#spares.splice(i, 1);
      if (logNamed('spare')) console.warn(`[spare] ${key}: let go, no thread in ${SPARE_IDLE_MS / 1000} s`);
      this.#letSpareGo(sp);
    }, SPARE_IDLE_MS);
    // Not what keeps a host's event loop alive.
    idle.unref?.();
  }

  /** Thread Worker `key` of process `proc`, its thread ended (`park`):
   *  kept as the process's spare, under a spare's key, for its next thread
   *  of program `gen` (the Worker's, which `#takeSpare` checks). A process
   *  need not have offered spares: a Wine process, whose modules are not a
   *  wasm32 program's, never does. False when it is not kept (the process
   *  has another program now, PARKED_MAX of its Workers wait already, or
   *  this one has run REUSE_MAX threads): it is let go and ends as before. */
  #park(proc, key, w, gen) {
    const rec = this.#spareRec.get(proc);
    const ref = this.#refOf.get(w);
    const ran = (w[THREADS_RUN] ?? 1) + 1;
    const no = !ref ? 'no ref' : rec && rec.execGen !== gen ? 'another program' : this.#foreign(proc) ? 'other flavour'
      : this.#inDoor(key) ? 'in a door call' : ran > REUSE_MAX ? 'REUSE_MAX threads run'
      : this.#spares.filter((x) => x.proc === proc && x.parked).length >= PARKED_MAX ? 'PARKED_MAX waiting' : '';
    if (no) {
      if (logNamed('spare')) console.warn(`[spare] ${key}: ended, its Worker not kept (${no})`);
      return false;
    }
    w[THREADS_RUN] = ran;
    const spareKey = `${proc}:spare${++this.#spareSeq}`;
    this.#workers.delete(key);
    this.#workers.set(spareKey, w);
    workerRan(w, spareKey);
    const slice = this.#stackOf.get(key);
    if (slice !== undefined) {
      this.#stackOf.delete(key);
      this.#stackOf.set(spareKey, slice);
    }
    ref.key = spareKey;
    const sp = { proc, key: spareKey, w, ref, gen, parked: true };
    this.#spares.push(sp);
    this.#spareIdle(sp);
    if (logNamed('spare')) console.warn(`[spare] ${spareKey}: ${key} ended, waits for the next thread (${this.#spares.length} waiting on the page)`);
    while (this.#spares.length > SPARE_MAX) this.#letSpareGo(this.#spares.shift());
    return true;
  }

  /** Process `proc`'s spare for a thread of its program `gen`, taken; null
   *  with none (one caught up with an earlier program is let go). */
  #takeSpare(proc, gen) {
    // A ready one first: a parked Worker, or a spare that has started.
    let i = this.#spares.findIndex((x) => x.proc === proc && (x.parked || !this.#pending.has(x.ref.key)));
    if (i < 0) i = this.#spares.findIndex((x) => x.proc === proc);
    if (i < 0) return null;
    const [sp] = this.#spares.splice(i, 1);
    if (gen === sp.gen) return sp;
    this.#letSpareGo(sp);
    this.#dropSpares(proc);
    return null;
  }

  /** Let a spare go: it gives back what it took of its process and ends,
   *  its exit releasing it (or now, if it never started). */
  #letSpareGo(sp) {
    if (this.#pending.has(sp.ref.key)) {
      this.#pending.delete(sp.ref.key);
      this.#releaseNow(sp.ref.key, sp.w);
      return;
    }
    sp.w.postMessage({ kind: 'spareGo', thread: null });
  }

  /** Process `proc`'s spares and their record, let go. */
  #dropSpares(proc) {
    this.#spareRec.delete(proc);
    this.#threadsOf.delete(proc);
    const mine = this.#spares.filter((x) => x.proc === proc);
    if (!mine.length) return;
    this.#spares = this.#spares.filter((x) => x.proc !== proc);
    for (const sp of mine) this.#letSpareGo(sp);
  }

  #releaseNow(key, w, reuse = false) {
    this.#workers.delete(key);
    // A spare that ended without a thread (let go, or its start failed).
    const at = this.#spares.findIndex((x) => x.w === w);
    if (at >= 0) this.#spares.splice(at, 1);
    // The last Worker of a process gone: drop its shared-module cache so a long
    // page does not accumulate every process's compiled modules forever.
    const proc = String(key).split(':')[0];
    // Only spares left: the process has ended (or gone on in another Worker
    // as another program), and they end with it, unasked -- its memory goes.
    const left = [...this.#workers.keys()].filter((k) => String(k).split(':')[0] === proc);
    if (left.length && left.every((k) => this.#spares.some((x) => x.ref.key === k))) {
      this.#spareRec.delete(proc);
      this.#threadsOf.delete(proc);
      for (const sp of this.#spares.filter((x) => x.proc === proc)) {
        this.#workers.delete(sp.ref.key);
        this.#pending.delete(sp.ref.key);
        const s2 = this.#stackOf.get(sp.ref.key);
        if (s2 !== undefined) {
          this.#stackOf.delete(sp.ref.key);
          this.#stacksFree.push(s2);
        }
        sp.w.terminate();
      }
      this.#spares = this.#spares.filter((x) => x.proc !== proc);
      left.length = 0;
    }
    if (!left.length) {
      this.#moduleCache.delete(proc);
    }
    const slice = this.#stackOf.get(key);
    if (slice !== undefined) {
      this.#stackOf.delete(key);
      this.#stacksFree.push(slice);
    }
    // Not the other flavour's: the pool is the page flavour's (`#other`).
    if (reuse && !this.#foreign(key) && this.#giveBack(w)) return;
    w.terminate();
  }

  /** Keep an exited process's zeroed memory for a fork child, dropping the
   *  oldest past RECYCLE_COUNT or RECYCLE_BYTES, and all of them once
   *  RECYCLE_KEEP_MS passes with no exit. */
  #keepMemory(mem) {
    this.#recycled.push(mem);
    clearTimeout(this.#recycleTimer);
    this.#recycleTimer = setTimeout(() => {
      narrate(`[host] ${this.#recycled.length} kept memories let go: no exit in ${RECYCLE_KEEP_MS / 1000} s`);
      this.#recycled.length = 0;
      this.#recycleTimer = null;
    }, RECYCLE_KEEP_MS);
    // Node: not a reason to stay up.
    this.#recycleTimer.unref?.();
    let bytes = this.#recycled.reduce((n, x) => n + x.buffer.byteLength, 0);
    while (this.#recycled.length > RECYCLE_COUNT || (this.#recycled.length > 1 && bytes > RECYCLE_BYTES)) {
      bytes -= this.#recycled.shift().buffer.byteLength;
    }
    if (bytes > RECYCLE_BYTES) this.#recycled.length = 0;
  }

  /** A kept memory for a child whose image ends at `need` bytes, taken: the
   *  smallest that already covers it, up to twice its size (the slack is
   *  zeroed, already-touched pages the shim's arena grows into without a
   *  `memory.grow`); otherwise the largest smaller one, which `growTo`
   *  extends. A memory never shrinks, so one more than twice the child's is
   *  left for a bigger child. */
  #takeMemory(need) {
    let best = -1;
    let bestN = 0;
    for (let i = 0; i < this.#recycled.length; i++) {
      const n = this.#recycled[i].buffer.byteLength;
      const fits = n >= need && n <= 2 * need;
      const better = best < 0
        || (fits && (bestN < need || n < bestN))
        || (!fits && n < need && bestN < need && n > bestN);
      if ((fits || n < need) && better) {
        best = i;
        bestN = n;
      }
    }
    return best < 0 ? undefined : this.#recycled.splice(best, 1)[0];
  }

  #onMessage(pid, w, m, resolve, reject) {
    switch (m?.kind) {
      case 'ready': {
        // How long the Worker took to exist and load its bundle.
        if (this.#extra?.debugExit && this.#asked.has(pid)) {
          const t = performance.now() - this.#asked.get(pid);
          this.#onOutput(2, `[gex] pid ${pid}: worker ready after ${Math.round(t)}ms\n`, pid);
        }
        const msg = this.#pending.get(pid);
        this.#pending.delete(pid);
        narrate(`[host] pid ${pid}: ready, ${msg ? `sending ${msg.kind}` : 'NOTHING HELD'}`);
        if (msg) {
          // Awaited: compiling the shim cannot be synchronous in a browser.
          this.#runtimeFor(pid).then((rt) => {
          // The closure goes after, one library per message: all at once does
          // not cross (Chromium `DataCloneError, out of memory`). The guest
          // runs while they arrive -- the miss path forgives a module not yet
          // landed and asks again when it shows up.
          const { closure, ...extra } = this.#extra;
          const tPost = logNamed('unitstall') ? performance.now() : -1;
          w.postMessage({
            ...extra,
            // How many to wait for. The Worker installs them before it runs
            // the guest, so it has to know when it has them all.
            closureCount: (closure ?? []).length,
            ...msg,
            outputAck: this.#outputAck(w),
            outputSeq: this.#outputSeqBuf(),
            kernelStack: this.#stackFor(w, pid),
            // Compiled once here, not from bytes in every Worker.
            shim: rt.shim,
            guest: rt.guest,
            kernel: rt.kernel,
            mem64: rt.mem64,
            pageFlavours: this.#pageFlavours(),
            otherGate: this.#otherGate?.buffer,
            otherLoad: this.#otherLoadFn(),
            saveGate: this.#saveGate,
            // A thread joins its process: every module the process's Workers
            // have compiled so far, so it does not fetch and compile them all
            // again (each thread of a game spent ~3 s doing that, one after the
            // other, at start-up). The `start` closure above meant to send
            // these, but this is the path that runs. A fork has its own memory
            // and registry, so only threads (pid `proc:tid`).
            sharedModules: this.#sharedFor(pid, msg.kind === 'thread', w),
            unitFiles: this.#unitFilePack(),
            // the wineserver's door, if one is up (`doorPublish`)
            door: this.#door ?? undefined,
            doorWord: this.#doorWord ?? undefined,
            memories: MEMORIES || undefined,
            // A fork child's memory: an exited process's, zeroed, if one fits.
            recycledMemory: msg.kind === 'adopt' && msg.child && !msg.child.exec && !this.#foreign(pid)
              ? this.#takeMemory(imageExtent(msg.child))
              : undefined,
          // `transfer`: buffers the caller gives up (the page's bundle), moved
          // rather than copied, and freed by the Worker once loaded.
          }, [...forkTransfer(msg.child), ...(msg.child?.bulk ? [msg.child.bulk] : []), ...(msg.bulk ? [msg.bulk] : []), ...(msg.transfer ?? [])]);
          if (tPost >= 0) console.warn(`[host-post] ${pid} ${msg.kind}: ${(performance.now() - tPost).toFixed(1)} ms`);
          // Timed separately: the fork-to-adopt gap is 549ms and unaccounted --
          // this `postMessage` (the 15.4 MiB image copy) or the closure below.
          const tSent = performance.now();
          if (TEST_FORK_RECORDS) this.#checkForkRecord(pid);
          this.#streamClosure(w, closure);
          if (this.#extra?.debugExit) {
            this.#onOutput(
              2,
              `[gex] pid ${pid}: sent ${(closure ?? []).length} closure modules in `
                + `${Math.round(performance.now() - tSent)}ms\n`,
              pid,
            );
          }
          },
          ).catch((e) => {
            // Its runtime could not be had: the process ends, not waits.
            this.#onOutput(2, `gex: process ${pid} could not start: ${e?.message ?? e}\n`, pid);
            reject(e);
          });
        }
        break;
      }
      case 'note':
        relayNote(m.note);
        break;
      case 'loadOther':
        // An exec waits on the gate (`#otherGate`) for the answer.
        this.#loadOther().catch(() => {});
        break;
      case 'doorPublish': {
        // The wineserver's door: every Worker of
        // the session, now and later, gets the server's memory and module.
        this.#door = m.door;
        narrate(`[host] wineserver pid ${m.door?.pid} published its door`);
        for (const [, other] of this.#workers) other.postMessage({ kind: 'door', door: m.door });
        break;
      }
      case 'wantLibrary':
        this.#sendLibrary(w, m.i);
        break;
      case 'snap-parked':
        this.#snapWait?.parked.set(String(pid), m);
        this.#snapWait?.check();
        break;
      case 'snap-attached': {
        const up = this.#snapUp.get(String(pid));
        this.#snapUp.delete(String(pid));
        up?.();
        break;
      }
      case 'kernelStacks':
        this.#stacks = m;
        // A pooled Worker that was ready before the stacks were known.
        this.#preattachOne();
        break;
      case 'output': {
        this.#inOrder(m.seq, () => {
          this.#onOutput(m.fd, m.text, pid, m);
          // Delivered: the Worker may now let its exit be seen.
          const a = this.#outputAcks.get(w);
          if (a) {
            Atomics.add(a, 0, 1);
            Atomics.notify(a, 0);
          }
        });
        break;
      }
      case 'droppedModules': {
        // A Worker let go of modules its process can no longer reach
        // (`Gex#reapDead`): the relay forgets them, so their code can go.
        // Its siblings and the Workers in the pool hold them too (relayed
        // and pooled), and so does the page's unit cache: all of them let go,
        // or the code stays. A unit dropped here is compiled again if the
        // same bytes come back (a DLL loaded again).
        const proc = String(pid).split(':')[0];
        const keys = m.keys ?? [];
        const cache = this.#moduleCache.get(proc);
        const units = [];
        for (const k of keys) {
          if (cache?.has(k)) {
            cache.delete(k);
            cache.bytes = (cache.bytes ?? 0) - regKeyBytes(k);
          }
          if (this.#unitModules.delete(k)) {
            this.#unitBytes -= Number(k.split(':')[1]);
            units.push(k);
          }
        }
        for (const [key, other] of this.#workers) {
          if (other !== w && String(key).split(':')[0] === proc) other.postMessage({ kind: 'droppedModules', keys });
        }
        if (units.length) {
          for (const p of this.#warm) {
            if (!p[POOLED]) continue;
            for (const k of units) p[POOLED].delete(k);
            p.postMessage({ kind: 'droppedModules', keys: units });
          }
        }
        break;
      }
      case 'unitFile': {
        if (typeof m.rel === 'string' && !this.#unitFiles.has(m.rel) && this.#unitFiles.size < UNIT_FILES) {
          this.#unitFiles.set(m.rel, m.bytes ?? null);
        }
        break;
      }
      case 'prefetchUnit': {
        // A unit a Worker published, its Module not yet compiled anywhere:
        // compiled here while the process runs on, to be every Worker's
        // before its first call there (`Gex#unitWant` compiles it itself
        // when this has not arrived).
        const name = `cgen:${m.key}`;
        const proc = String(pid).split(':')[0];
        if (typeof m.key !== 'string' || typeof m.url !== 'string') break;
        // Kept or on its way: to the asking process, which started before
        // the page had it and so was not handed it.
        const kept = this.#cgenModules.get(name);
        if (kept) { this.#cgenToProc(name, kept, proc); break; }
        const asking = this.#cgenFetching.get(m.key);
        if (asking) { asking.add(proc); break; }
        this.#cgenFetching.set(m.key, new Set([proc]));
        const t0 = performance.now();
        const timed = logNamed('unitstall');
        (this.#unitModule ? this.#unitModule(m.url) : fetch(m.url)
          .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.arrayBuffer(); })
          .then((b) => WebAssembly.compile(b)))
          .then((mod) => {
            if (timed) console.warn(`[unit-pf] ${m.key.slice(0, 16)}: compiled ahead in ${Math.round(performance.now() - t0)} ms`);
            // A Worker's own compile may have come in first: that one, then.
            const had = this.#cgenModules.get(name);
            if (!had) this.#cgenShare(name, mod, null, proc);
            for (const other of this.#cgenFetching.get(m.key) ?? []) {
              if (had || other !== proc) this.#cgenToProc(name, had ?? mod, other);
            }
          })
          .catch((err) => { if (timed) console.warn(`[unit-pf] ${m.key.slice(0, 16)}: ${err?.message ?? err}`); })
          .finally(() => this.#cgenFetching.delete(m.key));
        break;
      }
      case 'compiledModule': {
        // A Worker compiled a module with content key `m.key`. Remember it for this process
        // and relay it to the process's other live Workers so none recompiles
        // the same bytes. `WebAssembly.Module` crosses `postMessage` by
        // reference, so this is cheap. See `#moduleCache`.
        const proc = String(pid).split(':')[0];
        let cache = this.#moduleCache.get(proc);
        if (!cache) this.#moduleCache.set(proc, (cache = new Map()));
        if (this.#extra?.fillTime) this.#relayCensus();
        if (typeof m.key === 'string' && (m.key.startsWith('cgen:') || m.key.startsWith('cold:'))) {
          this.#cgenShare(m.key, m.mod, w, String(pid).split(':')[0]);
          break;
        }
        if (typeof m.key === 'string' && m.key.startsWith('plan:')) {
          if (!this.#planModules.has(m.key)) {
            this.#planModules.set(m.key, m.mod);
            this.#poolModule(m.key, m.mod);
            while (this.#planModules.size > PLAN_PAGE_MODULES) this.#planModules.delete(this.#planModules.keys().next().value);
            // and to the process's live Workers, which install the same
            // modules (not into its module cache, whose count it would take)
            for (const [key, other] of this.#workers) {
              if (other !== w && String(key).split(':')[0] === proc) other.postMessage({ kind: 'sharedModule', key: m.key, mod: m.mod });
            }
          }
          break;
        }
        if (typeof m.key === 'string' && m.key.startsWith('late:') && !this.#lateModules.has(m.key)) {
          this.#lateModules.set(m.key, m.mod);
          this.#poolModule(m.key, m.mod);
          while (this.#lateModules.size > LATE_PAGE_MODULES) this.#lateModules.delete(this.#lateModules.keys().next().value);
        }
        if (typeof m.key === 'string' && m.key.startsWith('wasm:')) {
          this.#pageModules.delete(m.key);
          this.#pageModules.set(m.key, m.mod);
          this.#poolModule(m.key, m.mod);
          let bytes = 0;
          for (const k of this.#pageModules.keys()) bytes += keyBytes(k);
          while (this.#pageModules.size > 1 && (this.#pageModules.size > PAGE_MODULES || bytes > PAGE_MODULE_BYTES)) {
            const oldest = this.#pageModules.keys().next().value;
            bytes -= keyBytes(oldest);
            this.#pageModules.delete(oldest);
          }
        }
        // A translation unit for the page's cache (`Gex#unitKey`): kept
        // for processes to come, not relayed -- the process's own Workers
        // share it by its content key.
        if (typeof m.key === 'string' && m.key.startsWith('u:')) {
          if (this.#extra?.unitCache && !this.#unitModules.has(m.key)) {
            this.#unitModules.set(m.key, m.mod);
            this.#poolModule(m.key, m.mod);
            this.#unitBytes += Number(m.key.split(':')[1]);
            while ((this.#unitBytes > UNIT_MODULE_BYTES || this.#unitModules.size > RELAY_MODULES) && this.#unitModules.size > 1) {
              const oldest = this.#unitModules.keys().next().value;
              this.#unitBytes -= Number(oldest.split(':')[1]);
              this.#unitModules.delete(oldest);
            }
          }
          break;
        }
        if (typeof m.key === 'string' && !cache.has(m.key)) {
          cache.set(m.key, m.mod);
          cache.bytes = (cache.bytes ?? 0) + regKeyBytes(m.key);
          while ((cache.bytes > PROCESS_MODULE_BYTES || cache.size > RELAY_MODULES) && cache.size > 1) {
            const oldest = cache.keys().next().value;
            cache.bytes -= regKeyBytes(oldest);
            cache.delete(oldest);
            if (!cache.capped) {
              cache.capped = true;
              console.error(`[host] process ${proc}'s compiled-module relay passed ${PROCESS_MODULE_BYTES >> 20} MiB: dropping the oldest (a pc translated over and over? see the "LIVELOCK?" line)`);
            }
          }
          if (this.#extra?.shareTrace) console.warn(`[share] host cached ${m.key.slice(0, 30)} from ${pid} (${cache.size} for process ${proc})`);
          for (const [key, other] of this.#workers) {
            if (other !== w && String(key).split(':')[0] === proc) {
              other.postMessage({ kind: 'sharedModule', key: m.key, mod: m.mod });
            }
          }
        }
        break;
      }
      case 'fork': {
        const c = m.child?.forkCost;
        if (c) {
          const hop1 = Math.round(performance.timeOrigin + performance.now() - c.sentAt);
          narrate(
            `[gex] fork -> pid ${m.child.pid}: image ${(c.bytes / 1048576).toFixed(1)} MiB`
              + ` in ${c.regions} regions, built ${c.buildMs}ms,`
              + ` ${hop1}ms to reach the host`,
          );
          c.atHost = performance.now();
        }
        this.#forked++;
        {
          // The exec'd program's memory, or the parent's.
          const f = m.child.mem64 ?? this.#mem64(pid);
          if (f !== !!this.#extra?.mem64) this.#flavourOf.set(m.child.pid, f);
          else this.#flavourOf.delete(m.child.pid);
        }
        this.#onFork(m.child, pid);
        if (TEST_FORK_RECORDS) this.#forkRecords.set(m.child.pid, new WeakRef(m.child));
        // Started and let run -- nobody waits for a child. The handler below
        // outlives this call by the child's whole life, so it holds the pid
        // alone, and the message is let go (see `#spawn`).
        const child = m.child.pid;
        this.#spawn(child, { kind: 'adopt', child: m.child }).catch((e) => {
          this.#onOutput(2, `gex: process ${child} failed: ${e.message}\n`, pid);
          // As an event too: this is caught, so no `error` fires and a harness
          // may miss the line above. See task 146.
          crashed(`pid ${child}: ${e.message}`);
        });
        m = null;
        break;
      }
      case 'execAway': {
        // The process's exec of a program of the other memory flavour (a
        // wasm32 one in a 64-bit memory's page: shim lib.rs `exec_away`):
        // its Worker has ended without the process, which goes on in a new
        // one with that flavour's runtime, as a posix_spawn child's exec
        // does, and ends there.
        const child = m.child;
        if (child.mem64 !== undefined && child.mem64 !== !!this.#extra?.mem64) this.#flavourOf.set(Number(pid), child.mem64);
        else this.#flavourOf.delete(Number(pid));
        this.#releaseNow(pid, w);
        this.#spawn(pid, { kind: 'adopt', child }).then(resolve, reject);
        break;
      }
      case 'thread': {
        // A thread of a running process: same memory, same kernel, its own
        // Worker. Nothing waits for it.
        this.#threads++;
        const key = `${pid}:${m.thread.tid}`;
        const proc = String(pid).split(':')[0];
        // A threaded process's memory, for a harness (`MEMORIES`).
        if (MEMORIES && m.thread.memory) (globalThis.__gexMemories ??= new Map()).set(Number(proc), m.thread.memory);
        if (globalThis.GEX_ATTACH_TRACE) console.error(`[spawn] host got 'thread' for ${key}, taking a Worker (t=${Date.now()})`);
        const before = this.#threadsOf.get(proc);
        if (before !== undefined) this.#threadsOf.set(proc, before + 1);
        const spare = this.#takeSpare(proc, m.thread.execGen);
        if (spare) {
          // The thread goes to the Worker already caught up with its
          // process; the memory it was posted with is not needed.
          m.thread.bulk?.close?.();
          this.#workers.delete(spare.key);
          this.#workers.set(key, spare.w);
          workerRan(spare.w, key);
          const slice = this.#stackOf.get(spare.key);
          if (slice !== undefined) {
            this.#stackOf.delete(spare.key);
            this.#stackOf.set(key, slice);
          }
          const held = this.#pending.get(spare.key);
          if (held) {
            this.#pending.delete(spare.key);
            this.#pending.set(key, held);
          }
          spare.ref.key = key;
          if (logNamed('spare')) console.warn(`[spare] ${key}: on ${spare.key}, ${held ? 'still starting' : 'ready'}`);
          const { tid, ctx, pc, procOpts } = m.thread;
          spare.w.postMessage({ kind: 'spareGo', thread: { tid, ctx, pc, procOpts } });
        } else {
          this.#threadWorker(pid, key, { kind: 'thread', ...m.thread, park: !this.#foreign(proc) && !!this.#extra?.shared && !(SP_CODE && single.on) });
        }
        // The next spare, for a process that starts more than one thread.
        if (before >= 1 && !this.#spares.some((x) => x.proc === proc)) this.#startSpare(proc);
        break;
      }
      case 'park': {
        // A thread has ended and its process runs on (`Gex#parkThread`): its
        // Worker waits for the process's next thread, or is let go.
        const proc = String(pid).split(':')[0];
        if (!this.#park(proc, pid, w, m.gen)) w.postMessage({ kind: 'spareGo', thread: null });
        break;
      }
      case 'threadSpare': {
        // A program with many modules has started (`Gex#offerSpare`): a
        // Worker for its first thread, caught up with them meanwhile.
        const proc = String(pid).split(':')[0];
        this.#dropSpares(proc);
        if (this.#foreign(proc) || !this.#extra?.shared || (SP_CODE && single.on)) break;
        this.#spareRec.set(proc, m.spare);
        this.#threadsOf.set(proc, 0);
        this.#startSpare(proc);
        break;
      }
      case 'stdin': {
        // Only pid 1 has a terminal; a child reads the inherited descriptor,
        // which is the kernel's business, not this file's.
        if (!this.#readStdin) {
          w.postMessage({ kind: 'eof' });
          break;
        }
        Promise.resolve(this.#readStdin(m.lflag)).then((bytes) => {
          // `true` means "input is already in the shared queue you read from".
          if (bytes === true) w.postMessage({ kind: 'stdin', bytes: true });
          else if (bytes && bytes.length) w.postMessage({ kind: 'stdin', bytes });
          else w.postMessage({ kind: 'eof' });
        });
        break;
      }
      case 'exit':
        if (m.threadGone) {
          // Not the process's end: it goes on in another Worker.
          const c = this.#carried.get(pid);
          this.#release(pid, w);
          if (c && 'status' in c) {
            this.#carried.delete(pid);
            resolve(c.status);
          } else {
            this.#carried.set(pid, { resolve });
          }
          break;
        }
        if (m.carries) {
          // A thread Worker that took its process over by `execve`: its exit
          // is the process's, owed to whoever waits on the first Worker.
          const proc = Number(String(pid).split(':')[0]);
          const c = this.#carried.get(proc);
          this.#onExit(m, proc);
          forgetMemory(proc);
          this.#release(pid, w);
          if (c?.resolve) {
            this.#carried.delete(proc);
            c.resolve(m.status);
          } else {
            this.#carried.set(proc, { status: m.status });
          }
          break;
        }
        // The wineserver ended: its door goes with it. A program started
        // after it starts a server of its own (Wine's loader asks
        // `door_missing`), and one would otherwise wait for a dead server.
        //
        // Its clients go with it, as on Linux when their pipes close: the
        // door is poisoned (area +124), so a call waiting for the lock, which
        // the server's pump may have held as it ended, or made before the
        // Worker hears of the end, goes by the pipe and finds the server
        // gone.
        if (this.#door && pid === this.#door.pid) {
          const i32 = new Int32Array(this.#door.memory.buffer), a = this.#door.area >> 2;
          if (!Atomics.load(i32, a + 31)) Atomics.store(i32, a + 31, -1);
          if (this.#doorWord) Atomics.compareExchange(this.#doorWord, 0, this.#door.pid, 0);
          Atomics.notify(i32, a);
          this.#door = null;
          for (const [, other] of this.#workers) other.postMessage({ kind: 'door', door: null });
        }
        // Kept for the page flavour's fork children only.
        if (m.recycle instanceof WebAssembly.Memory && !this.#foreign(pid)) this.#keepMemory(m.recycle);
        if (this.#extra?.exitTrace) {
          this.#onOutput(
            2,
            `[gex] worker ${pid} ended with status ${m.status}` +
              `${m.exitPc ? ` at pc 0x${m.exitPc.toString(16)}` : ''}` +
              ` (guest ${m.guestMillis ?? -1}ms)\n`,
            typeof pid === 'number' ? pid : 0,
          );
        }
        // How its Worker's traps went on (`Gex.traps`): straight on, or
        // through each step.
        if (m.traps && logNamed('traps')) {
          const t = m.traps;
          console.error(`[gex] pid ${pid}: traps straight on/through each step: run loop ${t.loop}/${t.loopFull}, call-outs ${t.callout}/${t.calloutFull}, native calls ${t.native}/${t.nativeFull}`);
        }
        // The page passes no `onExit`, so these per-process numbers were
        // computed and thrown away. Printed here, except under run-node
        // (`ownExitLines`), which prints its own.
        if (!this.#extra?.ownExitLines && m.millis !== undefined && logOn('debug')) {
          // How long the process existed, against what it can account for.
          const lived = this.#asked.has(pid)
            ? Math.round(performance.now() - this.#asked.get(pid))
            : 0;
          this.#asked.delete(pid);
          console.log(
            // `guest` is the time inside `gex.run`; `lived` minus `guest` is
            // Worker creation, adoption and the wait for a start message. The
            // two must be separable -- a cost has been blamed on the wrong side.
            `[gex] pid ${pid}: lived ${lived}ms (guest ${m.guestMillis ?? -1}ms)` +
              (m.cmdline ? ` {${m.cmdline}}` : '') +
              // What the runtime still held on exit. -1 without GEX_HEAP_COUNTERS=1.
              (m.heapKib >= 0 ? `, runtime holds ${Math.round(m.heapKib / 1024)} MiB` : '') +
              // Dispatch slots the page has left. A budget nothing refills, so
              // this only falls, and how fast it falls is what says how many
              // more processes the run can translate code in.
              (m.slotsLeftK >= 0 ? `, ${m.slotsLeftK}k dispatch slots left` : '') +
              // What this process's memory grew to. One memory per process, so
              // this times the number live at once is what the tab costs.
              (m.memoryMib !== undefined ? `, memory ${m.memoryMib} MiB` : '') +
              // And what that memory went on, which is the only way to tell a
              // guest that really maps a lot from a runtime spending it.
              (m.arena ? ` (maps ${m.arena[0]}, heap ${m.arena[1]}, stacks ${m.arena[2]})` : '') +
              // Dense modules this Worker left for a miss, and those a miss took.
              (m.fill?.denseLeft ? `, dense modules ${m.fill.denseWanted} wanted of ${m.fill.denseLeft} left (${m.fill.denseAsks} asks, ${m.fill.denseFillMs.toFixed(1)} ms, worst ${m.fill.denseFillWorst.toFixed(2)} ms)` : '') +
              (m.fill ? `, tables blocks ${m.fill.blocksLen} ftab ${m.fill.ftabLen}` : '') +
              // The one memory every process shares: the filesystem and the
              // store of translated modules live in it, and it only grows.
              (this.#kernel?.memory
                ? `, kernel ${Math.round(this.#kernel.memory.buffer.byteLength / 1048576)} MiB`
                : '') +
              // What the fork wrote into this process. There is no
              // copy-on-write, so this is paid in full, per fork.
              // A posix_spawn child that ran in its parent's Worker, in its
              // memory, until its exec: no copy, and this long to the exec.
              (m.vforkMs !== undefined ? `, forked in place (${m.vforkMs.toFixed(1)}ms to exec, nothing copied)` : '') +
              // A Worker given back after an earlier process (`#giveBack`).
              (m.workerRuns > 1 ? `, on a given-back worker (its process ${m.workerRuns})` : '') +
              (m.forkCopied
                ? `, fork copied ${Math.round(m.forkCopied / 1048576)} MiB`
                  + (m.forkZeros ? `, left ${Math.round(m.forkZeros / 1048576)} MiB of zeros alone` : '')
                  + (m.forkSkipped ? ` and skipped ${Math.round(m.forkSkipped / 1048576)}` : '')
                : '') +
              `, ${m.millis}ms translating ${m.blocks} blocks` +
              // Of those, ones on a page the closure already had: if most of
              // `blocks`, the fill is failing to reach the guest, not a
              // recompiler coverage gap.
              (m.liftInClosure ? ` (${m.liftInClosure} on closure pages)` : '') +
              // Why those misses translated rather than being filled.
              (m.liftWhy && m.liftWhy.some((n) => n)
                ? ` why[nofill=${m.liftWhy[0]} unowned=${m.liftWhy[1]} asked=${m.liftWhy[2]} fresh=${m.liftWhy[3]}]`
                : '') +
              (m.liftSlots ? ` slots[${m.liftSlots}]` : '') +
              (m.waits && (m.waits.park || m.waits.turn || m.waits.nap || m.waits.net)
                ? `, waited park=${m.waits.park} turn=${m.waits.turn} nap=${m.waits.nap} net=${m.waits.net ?? 0}`
                : '') +
              (m.byObject ? ` [${m.byObject}]` : '') +
              (m.xlate ? ` xlate[insns=${m.xlate[0]} wasmKiB=${Math.round(m.xlate[1] / 1024)} units=${m.xlate[2]} smc=${m.xlate[3]} retiredPages=${m.xlate[4]} retires=${m.xlate[5]} relift=${m.xlate[6]}blk/${Math.round((m.xlate[8] ?? 0) / 1024)}KiB anon=${m.xlate[7]}blk/${Math.round((m.xlate[9] ?? 0) / 1024)}KiB]` : '') +
              (m.installs ? `, ${m.installMs}ms installing ${m.installs} modules` +
            ` (${m.compileMs}ms compile${m.compiles !== undefined ? ` of ${m.compiles}` : ''}${m.unitHits ? `, ${m.unitHits} from the unit cache` : ''}, ${m.instantiateMs}ms instantiate, ${m.growMs}ms grow)` : '') +
          `, worker start ${(m.shimCompileMs ?? 0) + (m.shimInstMs ?? 0) + (m.guestInstMs ?? 0)}ms` + ` (shim ${m.shimCompileMs}c/${m.shimInstMs}i, guest ${m.guestInstMs}i)` +
              (m.loader && m.loader.calls
                ? `, ld.so ${m.loader.calls} syscalls, last at ${m.loader.doneMs}ms`
                : '') +
              (m.loader && m.loader.link && m.loader.link !== 'off'
                ? `, link ${m.loader.link}` +
                  (m.loader.cost && m.loader.cost[0]
                    ? ` (${m.loader.cost[0]} regions, ` +
                      `${Math.round(m.loader.cost[1] / 1024)} KiB reread in ${m.loader.cost[3]}ms, ` +
                      `${Math.round(m.loader.cost[2] / 1024)} KiB copied in ${m.loader.cost[4]}ms)`
                    : '')
                : '') +
              (m.fill && (m.fill.filled || m.fill.unanswered)
                ? `, fill ${m.fill.filled} answered ${m.fill.unanswered} not`
                  + (m.fill.sysN ? `, sys ${m.fill.sysN}x ${m.fill.sysMs}ms` : '')
                  + ` (${m.fill.now ?? 0} fast, ${m.fill.nowMs ?? 0}ms[base ${m.fill.faBase ?? 0} init ${m.fill.faInit ?? 0}];`
                  + ` ${m.fill.slow ?? 0} slow, ${m.fill.slowMs ?? 0}ms;`
                  + ` ${m.fill.pub ?? 0} published, ${m.fill.pubMs ?? 0}ms,`
                  + ` grew ${m.fill.grew ?? 0} slots in ${m.fill.growMs ?? 0}ms,`
                  + ` want ${m.fill.wantMs ?? 0}ms inst ${m.fill.instMs ?? 0}ms`
                  + ` (grow ${m.fill.iGrow ?? 0} comp ${m.fill.iComp ?? 0}`
                  + ` inst ${m.fill.iInst ?? 0};`
                  + ` ${m.fill.iGrows ?? 0} grows +${m.fill.iAdded ?? 0}`
                  + ` copying ${m.fill.iCopied ?? 0}, tables ${m.fill.tlen ?? ''})`
                  + (m.fill.bigGrows ? `, big [${m.fill.bigGrows}]` : '')
                  + (m.fill.tn ? `, tn [${m.fill.tn}]` : '')
                  + ` tail ${m.fill.tailMs ?? 0}ms`
                  + (m.fill.strictGrows ? ` retry ${m.fill.strictGrows}x/${m.fill.strictSlots}` : '')
                  + `)`
                : '') +
              (m.reuse
                ? `, reuse have=${m.reuse[0]} hit=${m.reuse[1]} miss=${m.reuse[2]}` +
                  ` noident=${m.reuse[3]} served=${m.reuse[4]} stored=${m.reuse[5]} writable=${m.reuse[9]} nocode=${m.reuse[10]}`
                : ''),
          );
        }
        // The guest's own hot spots, when `?hot=1` asked for them.
        for (const line of m.hot ?? []) {
          this.#onOutput(2, `[hot] pid ${pid}: ${line}\n`, pid);
        }
        // Before the release: releasing terminates the Worker.
        this.#onExit(m, pid);
        forgetMemory(pid);
        this.#release(pid, w, !!m.giveBack);
        resolve(m.status);
        break;
      case 'threw':
        if (!String(pid).includes(':')) {
          // A process that fell over has ended as surely as one that exited:
          // its host lets go of it the same way, or its windows stay on the
          // screen, drawn from a block nobody writes again. Before the
          // release, as for an exit.
          this.#onExit({ threw: true, message: m.message }, pid);
          forgetMemory(pid);
        }
        this.#release(pid, w);
        // Here as well as in the Worker: only this side knows which process
        // fell over, and a child's promise often has nobody awaiting it.
        console.error(`[gex] pid ${pid} threw: ${m.message}\n${m.stack ?? ''}`);
        // As an event: the console line is INFO to a watcher and easy to miss.
        // See task 146.
        crashed(`pid ${pid} threw: ${m.message}`);
        reject(
          Object.assign(new Error(`pid ${pid}: ${m.message}`), { guestStack: m.stack }),
        );
        break;
      default:
        this.#onOther(m, pid, w);
        break;
    }
  }
}
