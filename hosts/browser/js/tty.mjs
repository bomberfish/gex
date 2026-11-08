// The page's end of the terminal. The terminal itself -- flags, size,
// foreground group, line discipline -- lives in the global kernel's shared
// memory (see `gex_kernel::tty`); this holds an `xterm.js`, calls `input` with
// what was typed and writes back what `drainEcho` returns.
//
// It reaches the kernel by instantiating it directly, rather than messaging a
// Worker: a `^C` must interrupt a program that may not be reading or running
// and may be one of a dozen Workers, and only the process table (in this
// module's memory) knows the foreground group. So a keystroke becomes a signal
// in the same turn it was typed, waking a parked guest by the same counter
// every cross-process delivery uses.
//
// Without a `SharedArrayBuffer` there is no global kernel: `open` answers null
// and the caller keeps its own line discipline.
import { helperStackTop, kernelThread, lazyFetch, processMemories, wasmMemory, wasmMemoryGone } from './lazy-fetch.mjs';
import { blockIo } from './blocks.mjs';
import { pageFs } from './page-fs.mjs';
import { single, sched } from './single.mjs';
import { SP_CODE, MP_CODE } from './mode.mjs';

/** The single-thread build's fibers park on the scheduler, not on their
 *  wait slots' words: whatever this file changes in the kernel, they look. */
const poke = () => { if ((SP_CODE && single.on)) sched.checkSlots(); };

/// Where the page writes bytes for the kernel to copy out of. The kernel
/// imports the calling process's memory and addresses it with `memory.copy`;
/// the page has none, so it provides a page of scratch and passes offsets. One
/// page suffices -- the largest thing that crosses is a 60-byte `termios`.
const SCRATCH = 0x100;
const SCRATCH_BYTES = 0x1000;

export class PageTty {
  #k;
  #mem;
  #scratch;

  /**
   * Instantiate the global kernel here, for the terminal's sake. `spec` is
   * `{ module, memory }`; returns null when the page has no shared memory or
   * the module predates this file (an absent export is a rebuild, not a boot
   * failure).
   */
  static async open(spec) {
    if (!spec?.module || !spec?.memory) return null;
    // In its own Worker when the kernel's memory is shared: the kernel's locks
    // wait (`memory.atomic.wait32`), which the page's thread may not, so a key
    // typed while a Worker held one -- a persist snapshot, a guest's syscall --
    // threw "Atomics.wait cannot be called in this context" here. The
    // single-thread build has nothing else in the kernel, and stays here.
    if (MP_CODE && !(SP_CODE && single.on) && typeof Worker === 'function' && typeof SharedArrayBuffer !== 'undefined'
        && spec.memory.buffer instanceof SharedArrayBuffer) {
      return RemoteTty.open(spec);
    }
    return PageTty.openHere(spec);
  }

  /** `open`, on this thread: what the single-thread build and tty-worker.mjs use. */
  static async openHere(spec) {
    if (!spec?.module || !spec?.memory) return null;
    const mod = spec.module instanceof WebAssembly.Module
      ? spec.module
      : await WebAssembly.compile(spec.module);
    // A maximum, because a memory import's limits are part of its type: one
    // with no maximum does not satisfy an import that declares one. 65535 is
    // what `hosts/kernel/add-memory.mjs` writes for the process import.
    //
    // Shared or plain to match the build: the page loads the kernel whose
    // process import is shared when its processes' memories are (every
    // multi-threaded shim), and a memory import's `shared` flag is part of
    // its type. Only the plain one was offered, so the page's terminal never
    // opened under the shared shim -- harmless for a translated shell, which
    // gives its stack back and is fed by the fallback below, and fatal for a
    // wasm32 one, which waits inside its own frames for keys that only the
    // kernel's terminal can hand it ("can't even type").
    let inst;
    let scratch;
    try {
      try {
        scratch = wasmMemory({ initial: 1, maximum: 65535, shared: true }, "the terminal's scratch memory", spec.spare);
        inst = await WebAssembly.instantiate(mod, {
          env: { memory: spec.memory, ...processMemories(scratch, false, spec.spare), lazy_fetch: lazyFetch(spec.memory), block_io: blockIo(spec.memory), page_fs: pageFs(spec.memory) },
        });
      } catch {
        wasmMemoryGone("the terminal's scratch memory", spec.spare);
        scratch = wasmMemory({ initial: 1, maximum: 65535 }, "the terminal's scratch memory", spec.spare);
        inst = await WebAssembly.instantiate(mod, {
          env: { memory: spec.memory, ...processMemories(scratch, false, spec.spare), lazy_fetch: lazyFetch(spec.memory), block_io: blockIo(spec.memory), page_fs: pageFs(spec.memory) },
        });
      }
    } catch (e) {
      // A module that will not take a plain process memory is the shared build,
      // not what a page loads. Warn rather than fail silently: the fallback
      // line editing looks almost right, but `^C` is quietly gone.
      console.warn(`[gex] no shared terminal: ${e.message}`);
      return null;
    }
    const k = inst.exports;
    if (typeof k.k_tty_input !== 'function') {
      console.warn('[gex] no shared terminal: the global kernel has no k_tty_input, '
        + 'so it is older than this glue. Rebuild it with hosts/kernel/build.sh.');
      return null;
    }
    // A stack of this instance's own: every instance's `__stack_pointer` starts
    // at the same shared address, so two in the kernel at once overwrite each
    // other's locals. The kernel reserves an array of slices for this.
    //
    // `area + count * size` is above every top the host hands out. `area + size`
    // would collide with the first Worker (which is handed exactly that top),
    // and it failed as stack collisions do -- rarely, and somewhere else.
    // Reading where the array is uses the linked-in stack, safe only because
    // the page instantiates before any Worker exists.
    if (k.k_stack_area && inst.exports.__stack_pointer) {
      // Its own slot (`HELPER_STACKS`), thread-locals at the top of it.
      kernelThread(inst.exports, helperStackTop(k, 'tty'));
    }
    return new PageTty(k, scratch);
  }

  constructor(k, scratch) {
    this.#k = k;
    this.#scratch = scratch;
    this.#mem = new Uint8Array(scratch.buffer);
  }

  #bytes() {
    // A grown memory detaches its buffer. This one never grows, but re-reading
    // is free and a stale view is a silent wrong answer.
    if (this.#mem.buffer !== this.#scratch.buffer) {
      this.#mem = new Uint8Array(this.#scratch.buffer);
    }
    return this.#mem;
  }

  /**
   * Keystrokes, through the line discipline.
   *
   * Returns how many bytes of echo are queued. Everything else a key can mean
   * -- a line completed, a signal raised, the input flushed, end of input --
   * has already happened by the time this returns.
   */
  input(data) {
    const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    let queued = 0;
    // In scratch-sized pieces. A paste is one `onData` of arbitrary length.
    for (let at = 0; at < bytes.length; at += SCRATCH_BYTES - SCRATCH) {
      const chunk = bytes.subarray(at, at + SCRATCH_BYTES - SCRATCH);
      this.#bytes().set(chunk, SCRATCH);
      queued = this.#k.k_tty_input(BigInt(SCRATCH), chunk.length);
    }
    poke();
    return queued;
  }

  /**
   * A keystroke as the page's terminal takes it: `input`, then the echo to
   * show (null for none) and what `ready` says after it, in one call.
   */
  key(data) {
    const echo = this.input(data) ? this.drainEcho() : null;
    return { echo, ready: this.ready() };
  }

  /** What the terminal should display, as bytes. Empty when there is none. */
  drainEcho() {
    const out = [];
    for (;;) {
      const n = this.#k.k_tty_echo(BigInt(SCRATCH), SCRATCH_BYTES - SCRATCH);
      if (n <= 0) break;
      out.push(this.#bytes().slice(SCRATCH, SCRATCH + n));
      if (n < SCRATCH_BYTES - SCRATCH) break;
    }
    if (!out.length) return null;
    if (out.length === 1) return out[0];
    const total = out.reduce((a, b) => a + b.length, 0);
    const all = new Uint8Array(total);
    let at = 0;
    for (const b of out) {
      all.set(b, at);
      at += b.length;
    }
    return all;
  }

  /**
   * Bytes a guest could read now: positive for waiting input, 0 for none,
   * -1 once there will be no more.
   */
  ready() {
    return this.#k.k_stdin_ready();
  }

  /** There will be no more input -- the page has gone, or `^D` at the end. */
  eof() {
    this.#k.k_stdin_eof();
    poke();
  }

  /** `c_lflag`, which says whether the guest has taken over line editing. */
  lflag() {
    const n = this.#k.k_tty_get_termios(BigInt(SCRATCH), 60);
    if (n < 16) return 0;
    const b = this.#bytes();
    return (b[SCRATCH + 12] | (b[SCRATCH + 13] << 8)
      | (b[SCRATCH + 14] << 16) | (b[SCRATCH + 15] << 24)) >>> 0;
  }

  /** The size the guest believes the terminal is, as `[rows, cols]`. */
  winsize() {
    const v = this.#k.k_tty_winsize();
    return [Number(v >> 48n) & 0xffff, Number((v >> 32n) & 0xffffn)];
  }

  /**
   * The terminal changed size. Signals `SIGWINCH` to the foreground group,
   * and only when it really changed -- a page that reports its size on every
   * animation frame would otherwise make `bash` redraw its prompt sixty times
   * a second.
   */
  resize(rows, cols) {
    const packed = (BigInt(rows & 0xffff) << 48n) | (BigInt(cols & 0xffff) << 32n);
    const r = this.#k.k_tty_resize(packed) !== 0;
    poke();
    return r;
  }

  /** The foreground process group, or 0 before a shell has claimed one. */
  fg() {
    return this.#k.k_tty_fg();
  }

  /** Raise a signal on the foreground group, for what no key stands for. */
  signal(signo) {
    const r = this.#k.k_tty_signal(signo >>> 0);
    poke();
    return r;
  }
}

/**
 * `PageTty` in a Worker of its own (tty-worker.mjs), for a page whose kernel
 * memory is shared. Every method answers a promise, in the order called (the
 * Worker takes them one at a time), with what `PageTty`'s returns.
 */
class RemoteTty {
  #w;
  #id = 0;
  #waiting = new Map();

  static async open(spec) {
    const w = new Worker(new URL('./tty-worker.mjs', import.meta.url), { type: 'module', name: 'tty' });
    const t = new RemoteTty(w);
    const ok = await t.#call('open', { module: spec.module, memory: spec.memory }).catch((e) => {
      console.warn(`[gex] no shared terminal: ${e?.message ?? e}`);
      return false;
    });
    if (!ok) {
      w.terminate();
      return null;
    }
    return t;
  }

  constructor(w) {
    this.#w = w;
    w.onmessage = (e) => {
      const { id, value, error } = e.data;
      const p = this.#waiting.get(id);
      this.#waiting.delete(id);
      if (error !== undefined) p?.reject(new Error(error));
      else p?.resolve(value);
    };
    w.onerror = (e) => {
      for (const p of this.#waiting.values()) p.reject(new Error(e.message || 'the terminal Worker failed'));
      this.#waiting.clear();
    };
  }

  #call(op, ...args) {
    const id = ++this.#id;
    return new Promise((resolve, reject) => {
      this.#waiting.set(id, { resolve, reject });
      this.#w.postMessage({ id, op, args });
    });
  }

  key(data) { return this.#call('key', data); }
  input(data) { return this.#call('input', data); }
  drainEcho() { return this.#call('drainEcho'); }
  ready() { return this.#call('ready'); }
  eof() { return this.#call('eof'); }
  lflag() { return this.#call('lflag'); }
  winsize() { return this.#call('winsize'); }
  resize(rows, cols) { return this.#call('resize', rows, cols); }
  fg() { return this.#call('fg'); }
  signal(signo) { return this.#call('signal', signo); }
  close() { this.#w.terminate(); }
}
