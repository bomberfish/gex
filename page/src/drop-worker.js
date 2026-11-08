// Files dropped on the page, copied into the running system's filesystem
// (app.js's handle.drop, drop.js). One Worker for every drop of a page, on a
// kernel instance and a kernel stack of its own (kernel-helper.js); drops
// are copied one at a time, in the order they came.
//
// A drop's files are read from the browser's File objects a chunk at a time
// and written through `k_fs_open`/`k_fs_write`, as the program's own writes
// are (so a kept root keeps them), into one directory: the one
// the page named, or a new /tmp/drops/N. Nothing there is written over: a
// top-level name it has already is numbered, "a (2).txt", as is one the
// drop has twice. Each file is written under a hidden name beside its own
// (".NAME.ID.drop-part") and renamed when whole. Directories are made
// 0755, files 0644, or 0755 when they start as a program does ("\x7fELF" or
// "#!"); ownership is the guest's one user's (the kernel keeps none).
// Each file is stamped with its File's lastModified (the kernel has no
// clock), and each folder the drop made with the newest time of the files
// under it, or the drop's if there are none.
//
// Messages in:  { spec: { module, memory }, stackTop }        once, first;
//               { id, dir, root, entries: [{ path, file? }] } a drop: `dir` the
//                 directory to make or null, `root` where /tmp/drops/N go, and
//                 each entry a relative path, a File, or no file for a directory;
//               { id, move: { from, names, into } }   the drop's top-level names
//                 moved into another directory (beside what is there: "a (2).txt");
//               { abort: id }                                   stop that drop.
// Messages out: { id, progress: { done, total, files, filesDone, name } },
//               { id, done: { path, names, files, bytes, ms } } (`names`: the
//                 top-level names as they went in), { id, placed: { paths } },
//               { id, error, aborted? }.
// An aborted or failed drop takes back what it wrote.

import { helperKernel } from './kernel-helper.js';

const O_WRONLY = 0o1;
const O_CREAT = 0o100;
const O_EXCL = 0o200;
const EEXIST = 17;
/** Where paths go in this Worker's scratch memory, and the bytes after them. */
const PATH_AT = 0;
const PATH_MAX = 1 << 16;
const DATA_AT = PATH_MAX;
/** The most bytes one `k_fs_write` carries. */
const CHUNK = 1 << 20;
/** How often progress is said while bytes move. */
const PROGRESS_MS = 100;

const enc = new TextEncoder();
let k = null;
let mem = null;
const aborted = new Set();
/** The drops sent and not yet answered: an abort for any other is dropped. */
const pending = new Set();

function view(need) {
  const have = mem.buffer.byteLength;
  if (need > have) mem.grow(Math.ceil((need - have) / 65536));
  return new Uint8Array(mem.buffer);
}

function putPath(p) {
  const b = enc.encode(p);
  if (b.length > PATH_MAX) throw new Error(`${p.slice(0, 80)}...: path too long`);
  view(PATH_AT + b.length).set(b, PATH_AT);
  return b.length;
}

const mkdir = (p, mode = 0o755) => k.k_fs_mkdir(BigInt(PATH_AT), putPath(p), mode);
const unlink = (p) => k.k_fs_unlink(BigInt(PATH_AT), putPath(p));
const rmdir = (p) => k.k_fs_rmdir(BigInt(PATH_AT), putPath(p));
const chmod = (p, mode) => k.k_fs_chmod(BigInt(PATH_AT), putPath(p), mode);
const setMtime = (p, ms) => k.k_fs_set_mtime(BigInt(PATH_AT), putPath(p), BigInt(Math.floor(ms / 1000)));
/** `src` renamed `dst`, never over a name already there (RENAME_NOREPLACE). */
function rename(src, dst) {
  const sb = enc.encode(src);
  const db = enc.encode(dst);
  view(PATH_AT + sb.length + db.length).set(sb, PATH_AT);
  new Uint8Array(mem.buffer).set(db, PATH_AT + sb.length);
  return k.k_fs_rename(BigInt(PATH_AT), sb.length, BigInt(PATH_AT + sb.length), db.length, 1);
}
/** An open file's access and modification times, both `ms` (Unix ms). */
function stamp(h, ms, nowMs) {
  const s = BigInt(Math.floor(ms / 1000));
  const ns = Math.round((ms - Math.floor(ms / 1000) * 1000) * 1e6);
  return k.k_fs_set_times_fd(h, s, ns, s, ns, BigInt(Math.floor(nowMs / 1000)));
}
/** A File's time: its lastModified, or `nowMs` when it has none. */
const fileTime = (f, nowMs) => (Number.isFinite(f.lastModified) && f.lastModified > 0 ? f.lastModified : nowMs);
/** Whether anything (a file, a folder, a link) is at `p`. */
const exists = (p) => k.k_fs_path_meta(BigInt(PATH_AT), putPath(p), BigInt(DATA_AT)) >= 0;

/** `p` and every directory above it (made 0755 where missing). */
function mkdirs(p) {
  let at = '';
  for (const part of p.split('/').filter(Boolean)) {
    at += `/${part}`;
    const r = mkdir(at);
    if (r < 0 && r !== -EEXIST) throw new Error(`mkdir ${at}: errno ${-r}`);
  }
}

/** A new directory for a drop: `dir`, or the next free `root`/N. */
function newDir(dir, root) {
  if (dir) {
    mkdirs(dir);
    return dir;
  }
  mkdirs(root);
  for (let n = 1; ; n++) {
    const p = `${root}/${n}`;
    const r = mkdir(p);
    if (r === 0) return p;
    if (r !== -EEXIST) throw new Error(`mkdir ${p}: errno ${-r}`);
  }
}

/** `name` with " (n)" before its extension: "a (2).txt". */
function numbered(name, n) {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? `${name.slice(0, dot)} (${n})${name.slice(dot)}` : `${name} (${n})`;
}

/** A top-level name for `name` free in `top` and not yet taken by this drop. */
function freeName(top, name, taken) {
  for (let n = 1; ; n++) {
    const c = n === 1 ? name : numbered(name, n);
    if (!taken.has(c) && !exists(`${top}/${c}`)) return c;
  }
}

const startsAsProgram = (b) =>
  (b.length >= 4 && b[0] === 0x7f && b[1] === 0x45 && b[2] === 0x4c && b[3] === 0x46) ||
  (b.length >= 2 && b[0] === 0x23 && b[1] === 0x21);

async function copy(m) {
  const t0 = performance.now();
  const nowMs = Date.now();
  // The newest time under each folder the drop made, by path.
  const newest = new Map();
  const touched = (path, ms) => {
    for (let at = path.lastIndexOf('/'); at > 0; at = path.lastIndexOf('/', at - 1)) {
      const dir = path.slice(0, at);
      if (newest.has(dir)) newest.set(dir, Math.max(newest.get(dir), ms));
    }
  };
  const files = m.entries.filter((e) => e.file);
  const total = files.reduce((n, e) => n + e.file.size, 0);
  const top = newDir(m.dir, m.root);
  // Each top-level name as it goes in: one already in `top` (an existing
  // `dir`) or twice in the drop is numbered ("a (2).txt"), and what is
  // under it with it, so nothing in `top` is ever written over.
  const names = new Map();
  const taken = new Set();
  const placed = [];
  const place = (path) => {
    const i = path.indexOf('/');
    const head = i < 0 ? path : path.slice(0, i);
    // A top-level entry, or (a folder comes before what is in it, so only
    // for entries made without one) the first under its name.
    if (i < 0 || !names.has(head)) {
      const n = freeName(top, head, taken);
      taken.add(n);
      names.set(head, n);
      placed.push(n);
    }
    return `${names.get(head)}${i < 0 ? '' : path.slice(i)}`;
  };
  // A top-level name something else made after `freeName` looked: the
  // entry takes the next free one, once.
  const replace = (path) => {
    if (path.includes('/')) return null;
    const n = freeName(top, path, taken);
    taken.add(n);
    names.set(path, n);
    placed[placed.length - 1] = n;
    return `${top}/${n}`;
  };
  // What this drop made, to take back if it does not finish.
  const made = [];
  let done = 0;
  let filesDone = 0;
  let last = 0;
  const progress = (name, force = false) => {
    const now = performance.now();
    if (!force && now - last < PROGRESS_MS) return;
    last = now;
    self.postMessage({ id: m.id, progress: { done, total, files: files.length, filesDone, name } });
  };
  const check = () => {
    if (aborted.has(m.id)) throw Object.assign(new Error('the drop was aborted'), { aborted: true });
  };
  if (!m.dir) newest.set(top, -Infinity);
  try {
    progress('', true);
    for (const e of m.entries) {
      check();
      let path = `${top}/${place(e.path)}`;
      if (!e.file) {
        let r = mkdir(path);
        if (r === -EEXIST) {
          const again = replace(e.path);
          if (again) r = mkdir((path = again));
        }
        if (r < 0 && r !== -EEXIST) throw new Error(`mkdir ${path}: errno ${-r}`);
        if (r === 0) {
          made.push([path, true]);
          newest.set(path, -Infinity);
        }
        continue;
      }
      // Written under a hidden name beside its own and renamed once whole:
      // a program watching the folder (explorer's desktop reads a program's
      // icon as it comes) never sees a file part written. O_EXCL: never a
      // file that is there already.
      const open = (p) => k.k_fs_open(BigInt(PATH_AT), putPath(p), BigInt(O_WRONLY | O_CREAT | O_EXCL), 0o644);
      const slash = path.lastIndexOf('/');
      const part = `${path.slice(0, slash)}/.${path.slice(slash + 1)}.${m.id}.drop-part`;
      const h = open(part);
      if (h < 0) throw new Error(`open ${part}: errno ${-h}`);
      made.push([part, false]);
      let off = 0;
      let program = false;
      const reader = e.file.stream().getReader();
      let finished = false;
      try {
        for (;;) {
          const { done: end, value } = await reader.read();
          if (end) break;
          check();
          if (off === 0) program = startsAsProgram(value);
          for (let at = 0; at < value.length; at += CHUNK) {
            const part = value.subarray(at, Math.min(value.length, at + CHUNK));
            view(DATA_AT + part.length).set(part, DATA_AT);
            const n = k.k_fs_write(h, BigInt(off), BigInt(DATA_AT), part.length);
            if (n !== part.length) throw new Error(`write ${path}: ${n < 0 ? `errno ${-n}` : `${n} of ${part.length} bytes`}`);
            off += n;
            done += n;
          }
          progress(e.path);
        }
        finished = true;
        const ms = fileTime(e.file, nowMs);
        const r = stamp(h, ms, nowMs);
        if (r < 0) throw new Error(`set times ${path}: errno ${-r}`);
        touched(path, ms);
      } finally {
        k.k_fs_close(h);
        // Stopped part way: the browser need not read the rest of the File.
        if (!finished) reader.cancel().catch(() => {});
      }
      // (see `place`: a name something else took since, the next free one)
      let r = rename(part, path);
      if (r === -EEXIST) {
        const again = replace(e.path);
        if (again) r = rename(part, (path = again));
      }
      if (r < 0) throw new Error(`rename ${part} to ${path}: errno ${-r}`);
      made[made.length - 1] = [path, false];
      if (program) chmod(path, 0o755);
      filesDone++;
      progress(e.path);
    }
    // The folders, deepest first, so each one's time is in its parent's
    // (an empty one's, the drop's, is not).
    for (const dir of [...newest.keys()].sort((a, b) => b.split('/').length - a.split('/').length)) {
      const t = newest.get(dir);
      const r = setMtime(dir, t > -Infinity ? t : nowMs);
      if (r < 0) throw new Error(`set mtime ${dir}: errno ${-r}`);
      if (t > -Infinity) touched(dir, t);
    }
  } catch (err) {
    for (const [p, dir] of made.reverse()) (dir ? rmdir : unlink)(p);
    if (!m.dir) rmdir(top);
    throw err;
  }
  progress('', true);
  return { path: top, names: placed, files: files.length, bytes: done, ms: Math.round(performance.now() - t0) };
}

/** The top-level `names` of `from` moved into `into`, beside what is there. */
function move({ from, names, into }) {
  mkdirs(into);
  const paths = [];
  for (const name of names) {
    const src = `${from}/${name}`;
    for (let n = 1; ; n++) {
      const dst = `${into}/${n === 1 ? name : numbered(name, n)}`;
      const r = rename(src, dst);
      if (r === 0) {
        paths.push(dst);
        break;
      }
      if (r !== -EEXIST || n > 999) throw new Error(`rename ${src} to ${dst}: errno ${-r}`);
    }
  }
  rmdir(from);
  return { paths };
}

let queue = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  if (m.spec) {
    queue = queue.then(async () => {
      if (!k) ({ k, mem } = await helperKernel(m.spec, m.stackTop));
    });
    return;
  }
  if ('abort' in m) {
    if (pending.has(m.abort)) aborted.add(m.abort);
    return;
  }
  pending.add(m.id);
  queue = queue
    .then(() => (m.move ? { placed: move(m.move) } : { done: copy(m) }))
    .then(async (r) => self.postMessage({ id: m.id, ...(r.done ? { done: await r.done } : r) }))
    .catch((err) => self.postMessage({ id: m.id, error: String(err?.message ?? err), aborted: !!err?.aborted }))
    .finally(() => {
      pending.delete(m.id);
      aborted.delete(m.id);
    });
};
