//! Syscall entry and the syscall implementations.
//!
//! `svc #0` arrives here with the guest context still in linear memory: the
//! number in `x8`, arguments in `x0`-`x5`. The kernel reads them from the
//! context, dispatches, and writes the result back to `x0` — the same shape
//! the wasm-compiled kernel will use in the browser.
//!
//! Everything filesystem-shaped goes through [`crate::vfs`], so the same
//! implementation serves a host passthrough, an in-memory root and the
//! synthetic `/proc` without the syscall layer knowing which is which.

use crate::errno::{encode, Errno, SysResult};
use crate::file::{FdTable, OpenFile, PipeEnd};
use crate::fs::{oflags, Stat};
use crate::futex::{self, FutexTable};
use crate::mem::{flags as mflags, MemoryMap};
use crate::net;
use crate::proc::ProcTable;
use crate::pipe::Pipe;
use crate::unix;
use crate::signal::{self, Delivery, SigAction, SigAltStack, Signals};
use crate::stdio::{StdStream, Stream};
use crate::vfs::{FileType, Vfs};
use crate::GuestMem;
use gex_abi::{ctx, ctx_addr};
use std::sync::{Arc, Mutex};

/// aarch64 syscall numbers (the asm-generic ABI).
pub mod nr {
    pub const GETCWD: u64 = 17;
    pub const DUP: u64 = 23;
    pub const DUP3: u64 = 24;
    pub const FCNTL: u64 = 25;
    pub const IOCTL: u64 = 29;
    pub const MKDIRAT: u64 = 34;
    pub const UNLINKAT: u64 = 35;
    pub const SYMLINKAT: u64 = 36;
    pub const LINKAT: u64 = 37;
    pub const RENAMEAT: u64 = 38;
    pub const RENAMEAT2: u64 = 276;
    pub const SENDFILE: u64 = 71;
    pub const COPY_FILE_RANGE: u64 = 285;
    pub const SPLICE: u64 = 76;
    pub const TRUNCATE: u64 = 45;
    pub const FTRUNCATE: u64 = 46;
    pub const FACCESSAT: u64 = 48;
    pub const CHDIR: u64 = 49;
    pub const FCHDIR: u64 = 50;
    pub const UNSHARE: u64 = 97;
    pub const FCHMOD: u64 = 52;
    pub const FCHMODAT: u64 = 53;
    pub const SETXATTR: u64 = 5;
    pub const LSETXATTR: u64 = 6;
    pub const FSETXATTR: u64 = 7;
    pub const GETXATTR: u64 = 8;
    pub const LGETXATTR: u64 = 9;
    pub const FGETXATTR: u64 = 10;
    pub const LISTXATTR: u64 = 11;
    pub const LLISTXATTR: u64 = 12;
    pub const FLISTXATTR: u64 = 13;
    pub const REMOVEXATTR: u64 = 14;
    pub const LREMOVEXATTR: u64 = 15;
    pub const FREMOVEXATTR: u64 = 16;
    pub const OPENAT: u64 = 56;
    pub const OPENAT2: u64 = 437;
    pub const CLOSE: u64 = 57;
    pub const PIPE2: u64 = 59;
    pub const GETDENTS64: u64 = 61;
    pub const LSEEK: u64 = 62;
    pub const READ: u64 = 63;
    pub const WRITE: u64 = 64;
    pub const READV: u64 = 65;
    pub const WRITEV: u64 = 66;
    pub const PREAD64: u64 = 67;
    pub const PWRITE64: u64 = 68;
    pub const PPOLL: u64 = 73;
    pub const READLINKAT: u64 = 78;
    pub const NEWFSTATAT: u64 = 79;
    pub const FSTAT: u64 = 80;
    pub const FSYNC: u64 = 82;
    pub const FDATASYNC: u64 = 83;
    pub const SYNC_FILE_RANGE: u64 = 84;
    pub const SYNCFS: u64 = 267;
    pub const EXIT: u64 = 93;
    pub const EXIT_GROUP: u64 = 94;
    pub const SET_TID_ADDRESS: u64 = 96;
    pub const FUTEX: u64 = 98;
    pub const SET_ROBUST_LIST: u64 = 99;
    pub const GET_ROBUST_LIST: u64 = 100;
    pub const TIMERFD_CREATE: u64 = 85;
    pub const TIMERFD_SETTIME: u64 = 86;
    pub const TIMERFD_GETTIME: u64 = 87;
    pub const NANOSLEEP: u64 = 101;
    pub const CLOCK_GETTIME: u64 = 113;
    pub const CLOCK_GETRES: u64 = 114;
    pub const CLOCK_NANOSLEEP: u64 = 115;
    pub const SYNC: u64 = 81;
    pub const STATX: u64 = 291;
    pub const PSELECT6: u64 = 72;
    pub const FCHOWNAT: u64 = 54;
    pub const FCHOWN: u64 = 55;
    pub const UTIMENSAT: u64 = 88;
    pub const STATFS: u64 = 43;
    pub const FSTATFS: u64 = 44;
    pub const UMASK: u64 = 166;
    pub const UMOUNT2: u64 = 39;
    pub const MOUNT: u64 = 40;
    pub const GETITIMER: u64 = 102;
    pub const SETITIMER: u64 = 103;
    pub const SCHED_SETPARAM: u64 = 118;
    pub const SCHED_SETSCHEDULER: u64 = 119;
    pub const SCHED_GETSCHEDULER: u64 = 120;
    pub const SCHED_GETPARAM: u64 = 121;
    pub const SCHED_GETAFFINITY: u64 = 123;
    pub const SCHED_YIELD: u64 = 124;
    pub const SCHED_GET_PRIORITY_MAX: u64 = 125;
    pub const SCHED_GET_PRIORITY_MIN: u64 = 126;
    pub const SETPRIORITY: u64 = 140;
    pub const GETPRIORITY: u64 = 141;
    pub const GETCPU: u64 = 168;
    pub const KILL: u64 = 129;
    pub const TKILL: u64 = 130;
    pub const TGKILL: u64 = 131;
    pub const RT_SIGSUSPEND: u64 = 133;
    pub const RT_SIGACTION: u64 = 134;
    pub const RT_SIGPROCMASK: u64 = 135;
    pub const RT_SIGPENDING: u64 = 136;
    pub const RT_SIGRETURN: u64 = 139;
    pub const SIGALTSTACK: u64 = 132;
    pub const RT_SIGTIMEDWAIT: u64 = 137;
    pub const REBOOT: u64 = 142;
    pub const SETREGID: u64 = 143;
    pub const SETGID: u64 = 144;
    pub const SETREUID: u64 = 145;
    pub const SETUID: u64 = 146;
    pub const SETRESUID: u64 = 147;
    pub const GETRESUID: u64 = 148;
    pub const SETRESGID: u64 = 149;
    pub const GETRESGID: u64 = 150;
    pub const SETFSUID: u64 = 151;
    pub const SETFSGID: u64 = 152;
    pub const CAPGET: u64 = 90;
    pub const CAPSET: u64 = 91;
    pub const CHROOT: u64 = 51;
    pub const TIMES: u64 = 153;
    pub const SETPGID: u64 = 154;
    pub const GETPGID: u64 = 155;
    pub const GETSID: u64 = 156;
    pub const SETSID: u64 = 157;
    pub const GETGROUPS: u64 = 158;
    pub const SETGROUPS: u64 = 159;
    pub const UNAME: u64 = 160;
    pub const GETRUSAGE: u64 = 165;
    pub const GETTIMEOFDAY: u64 = 169;
    pub const GETPID: u64 = 172;
    pub const GETPPID: u64 = 173;
    pub const GETUID: u64 = 174;
    pub const GETEUID: u64 = 175;
    pub const GETGID: u64 = 176;
    pub const GETEGID: u64 = 177;
    pub const GETTID: u64 = 178;
    pub const SYSINFO: u64 = 179;
    pub const PRCTL: u64 = 167;
    pub const SOCKET: u64 = 198;
    pub const SOCKETPAIR: u64 = 199;
    pub const BIND: u64 = 200;
    pub const LISTEN: u64 = 201;
    pub const ACCEPT: u64 = 202;
    pub const ACCEPT4: u64 = 242;
    pub const MEMFD_CREATE: u64 = 279;
    pub const FLOCK: u64 = 32;
    pub const FALLOCATE: u64 = 47;
    pub const EVENTFD2: u64 = 19;
    pub const EPOLL_CREATE1: u64 = 20;
    pub const INOTIFY_INIT1: u64 = 26;
    pub const INOTIFY_ADD_WATCH: u64 = 27;
    pub const INOTIFY_RM_WATCH: u64 = 28;
    pub const EPOLL_CTL: u64 = 21;
    pub const EPOLL_PWAIT: u64 = 22;
    pub const EPOLL_PWAIT2: u64 = 441;
    pub const CLOSE_RANGE: u64 = 436;
    pub const PIDFD_SEND_SIGNAL: u64 = 424;
    pub const PIDFD_OPEN: u64 = 434;
    pub const MEMBARRIER: u64 = 283;
    pub const CONNECT: u64 = 203;
    pub const GETSOCKNAME: u64 = 204;
    pub const GETPEERNAME: u64 = 205;
    pub const SENDTO: u64 = 206;
    pub const RECVFROM: u64 = 207;
    pub const SETSOCKOPT: u64 = 208;
    pub const GETSOCKOPT: u64 = 209;
    pub const SHUTDOWN: u64 = 210;
    pub const SENDMSG: u64 = 211;
    pub const RECVMSG: u64 = 212;
    pub const RECVMMSG: u64 = 243;
    pub const SENDMMSG: u64 = 269;
    pub const BRK: u64 = 214;
    pub const MUNMAP: u64 = 215;
    pub const CLONE: u64 = 220;
    pub const EXECVE: u64 = 221;
    pub const CLONE3: u64 = 435;
    pub const MMAP: u64 = 222;
    pub const FADVISE64: u64 = 223;
    pub const MREMAP: u64 = 216;
    pub const MPROTECT: u64 = 226;
    pub const MADVISE: u64 = 233;
    pub const MSYNC: u64 = 227;
    pub const MLOCK: u64 = 228;
    pub const MUNLOCK: u64 = 229;
    pub const MLOCKALL: u64 = 230;
    pub const MUNLOCKALL: u64 = 231;
    pub const MLOCK2: u64 = 284;
    pub const WAIT4: u64 = 260;
    pub const WAITID: u64 = 95;
    pub const PRLIMIT64: u64 = 261;
    pub const GETRANDOM: u64 = 278;
    pub const PROCESS_VM_READV: u64 = 270;
    pub const PROCESS_VM_WRITEV: u64 = 271;
    pub const RSEQ: u64 = 293;
    pub const FACCESSAT2: u64 = 439;

    // Legacy calls only x86-64 has, given canonical numbers above anything
    // the asm-generic ABI assigns. See `crate::x86::translate`.
    pub const DUP2: u64 = 0x1000;
    pub const TIME: u64 = 0x1001;
    pub const ALARM: u64 = 0x1002;
    pub const ARCH_PRCTL: u64 = 0x1003;
    /// Advisory calls a guest may issue and never checks: `fadvise64`,
    /// `readahead`, `mlock`... Answered with success and nothing done.
    pub const NOOP_OK: u64 = 0x1004;
}

/// Hand a system call's result back to the guest: `x0`, or `rax` (both
/// `x(0)`). `ret` is the kernel's own convention, `-errno` for a failure.
/// Every place a result is written goes through here.
pub fn set_result(m: &mut GuestMem<'_>, ctx_addr: u64, ret: u64) -> Result<(), Errno> {
    m.write_u64(ctx_addr + ctx::x(0) as u64, crate::gpr(ret))
}

/// `*at` calls use this to mean "relative to the current directory".
const AT_FDCWD: i32 = -100;
const AT_EMPTY_PATH: u64 = 0x1000;
const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
const AT_REMOVEDIR: u64 = 0x200;

/// What the host should do after a syscall returns.
#[derive(Debug, Clone)]
pub enum SyscallOutcome {
    /// Resume the guest where it left off.
    Continue,
    /// Re-enter translated code at this address instead.
    ///
    /// Produced by `rt_sigreturn` and by signal delivery. Redirecting is safe
    /// at any host call because the guest's registers are already flushed to
    /// the context, so there is no wasm stack to unwind.
    Resume(u64),
    /// The process is over.
    Exit(i32),
    /// A signal with no handler ended the process.
    Killed { signal: u32, core: bool },
    /// `clone` without `CLONE_THREAD`: a new *process*.
    ///
    /// The host copies the address space, duplicates this kernel, and runs
    /// the child from the same instruction with `x0` zero. Nothing about the
    /// child's memory can be prepared here, because the kernel can only see
    /// the parent's.
    SpawnProcess {
        pid: i32,
        /// The stack the child must start on, or zero to keep the parent's.
        ///
        /// A plain `fork` passes zero. `posix_spawn` does not: musl clones
        /// onto a stack it allocated, and its `__clone` trampoline pops the
        /// function to run off that stack, so a child left on the parent's
        /// stack pointer reads whatever happens to be there.
        child_stack: u64,
        /// `CLONE_SETTLS`: the thread pointer for the child.
        tls: Option<u64>,
        /// `CLONE_VM | CLONE_VFORK` (`posix_spawn`'s clone): the child shares
        /// this memory and the calling thread waits until it execs or exits.
        /// A host that cannot share the memory makes a fork of it.
        vfork: bool,
    },
    /// `clone` set up a new thread's context; the host must run it.
    ///
    /// The kernel cannot spawn a thread itself -- a wasmtime store belongs to
    /// one -- so it prepares the context in shared guest memory and hands the
    /// host what it needs.
    SpawnThread {
        ctx_addr: u64,
        tid: i32,
    },
    /// This thread is finished; the process is not.
    ThreadExit,
    /// `execve`: replace this process's program.
    ///
    /// The kernel has read the strings out of the address space it is about
    /// to lose; everything else -- loading the image, building a new address
    /// space -- is the host's, because only the host can make one.
    Exec {
        path: String,
        argv: Vec<String>,
        envp: Vec<String>,
    },
    /// The syscall has to sleep, and must do so with the kernel unlocked.
    ///
    /// A blocking call that held the shared kernel would stop every other
    /// thread from making the very progress it is waiting for -- a futex wait
    /// could never be woken, because the waker needs the same lock to issue
    /// its wake. The host performs the operation, then hands the result back
    /// through [`Kernel::finish_blocking`].
    Blocking(BlockOp),
}

/// A syscall's blocking half, to be run without the kernel lock.
#[derive(Debug, Clone)]
pub enum BlockOp {
    FutexWait {
        table: Arc<FutexTable>,
        uaddr: u64,
        val: u32,
        deadline: Option<std::time::Duration>,
        /// A queue place snapshotted BEFORE the word was checked, for a host
        /// (the browser) that parks cooperatively instead of blocking under
        /// the queue lock. Taking it before the check is what stops a
        /// `FUTEX_WAKE` that lands between the check and the park from being
        /// lost -- the native host holds the lock across both instead and
        /// ignores this. See [`futex::FutexTable::park`].
        parked: Option<crate::futex::Parked>,
    },
    Sleep(std::time::Duration),
    /// `FUTEX_LOCK_PI(2)`: take the priority-inheritance lock at `uaddr` for
    /// `tid`, waiting as long as it is held. The host loops: an atomic take
    /// when the word is free (keeping the waiters bit, since others may
    /// sleep), else set the waiters bit and wait on the word; `EDEADLK` for
    /// a lock `tid` already owns; `ETIMEDOUT` at `deadline`.
    FutexLockPi {
        table: Arc<FutexTable>,
        uaddr: u64,
        tid: u32,
        deadline: Option<std::time::Duration>,
    },
    /// `rt_sigsuspend`: wait until a signal the caller can see arrives.
    ///
    /// The blocked mask has already been replaced with the one the caller
    /// asked for, so "can see" means whatever is deliverable now.
    /// `restore_mask` is what to put back afterwards, which
    /// [`Kernel::finish_blocking`] does before the call returns `EINTR` --
    /// the only thing it ever returns.
    SigSuspend { restore_mask: u64 },
    /// `rt_sigtimedwait` with none of `set` pending: wait until one of them
    /// is, for at most `timeout`. The host answers `Ok(0)` when one is
    /// pending (the call is then made again, which takes it with its
    /// `siginfo_t`), `EAGAIN` when the time is up, `EINTR` when a signal
    /// outside the set is delivered. The thread's and the process's pending
    /// sets travel with it, so asking takes no kernel lock; a signal another
    /// process sends reaches the process's set when the waiter's host
    /// collects it, as it does to ask whether a signal ends any wait.
    SigWait {
        set: u64,
        timeout: Option<std::time::Duration>,
        thread: Arc<crate::signal::ThreadSig>,
        process: Arc<crate::signal::ProcSig>,
    },
    /// A `read` on a stream with nothing in it yet: a terminal nobody has
    /// typed at, a socket with nothing arrived.
    ///
    /// The handle travels with the operation so the host can both ask whether
    /// it has become readable and perform the read once it has, without
    /// holding the kernel meanwhile. On a host with real threads that is a
    /// poll; in the browser, where there is one wasm stack, it is the only
    /// way a shell can wait for a keystroke at all.
    ReadStream {
        handle: Arc<dyn crate::vfs::FileHandle>,
        buf: u64,
        len: usize,
        /// A `readv` of more than one vector: read once, `len` bytes in all,
        /// and the bytes scattered over these (base, length) vectors in place
        /// of `buf`, so no other reader takes bytes from between them
        /// ([`scatter`]).
        vectors: Option<Arc<[(u64, u64)]>>,
    },
    /// A `connect` on a descriptor that is not non-blocking. Ready once the
    /// connection is up, refused, or gone.
    Connect { socket: Arc<net::Socket> },
    /// An `accept` on a listener with nothing waiting. The host only waits;
    /// the descriptor is made by the kernel afterwards, because only the
    /// kernel has a descriptor table. See `Kernel::retry_accept`.
    ///
    /// A `FileHandle` rather than a `UnixSocket`, because the socket may be
    /// in the global kernel instead of this process -- and what the scheduler
    /// wants of it is the one question both kinds answer the same way:
    /// `poll_readable`, which for a listener means a connection is waiting.
    AcceptWait {
        socket: Arc<dyn crate::vfs::FileHandle>,
    },
    /// A `poll` with nothing ready yet. The descriptors travel with it so the
    /// host can ask again without going back through the kernel.
    Poll {
        fds: Vec<PollFd>,
        deadline: Option<std::time::Duration>,
        /// How the answer is reported, which is the only thing that differs
        /// between `poll` and `select`.
        report: PollReport,
    },
    /// A `write` to a stream with no room: a pipe nobody is draining.
    WriteStream {
        handle: Arc<dyn crate::vfs::FileHandle>,
        buf: u64,
        len: usize,
        /// A vectored write of at most `PIPE_ATOMIC` bytes, gathered: written
        /// whole, in place of `buf`/`len` (see [`gather_small`]).
        gathered: Option<Arc<[u8]>>,
    },
    /// `wait4`. Blocks until one of `children` has a status to collect.
    WaitChild {
        /// Empty when the table is shared: a `ChildHandle` is an `Arc`, and
        /// an `Arc` names a memory the other processes cannot read. A shared
        /// table is asked instead -- `wait_ready` to see whether this thread
        /// can run, `collect` to take the child.
        children: Vec<ChildHandle>,
        /// So the waiter can drop the record once it has collected the child,
        /// and can notice a halt while parked.
        procs: Arc<ProcTable>,
        /// Who is waiting. Implicit in `children` when the table is local,
        /// and needed on every call when it is not.
        ppid: i32,
        /// Which child, or -1 for any.
        want: i32,
        /// Where to write the wait status, or zero.
        wstatus: u64,
        /// `WNOHANG`: report nothing rather than blocking.
        nohang: bool,
        /// `waitid`'s answer instead of `wait4`'s: a `siginfo_t` and zero.
        /// (`si_uid` is left zero.)
        waitid: Option<WaitidOut>,
    },
    /// `fcntl(F_SETLKW)`, `F_OFD_SETLKW` or a blocking `flock` with the
    /// lock held in the way: until `locks` would grant `req`
    /// ([`unix::FileLocks::would_grant`]; every release notifies its
    /// `queue_id`). The host then makes the request (natively,
    /// [`unix::FileLocks::apply`]) or the call again (the page), parking
    /// again if another got there first.
    FileLock { locks: Arc<unix::FileLocks>, req: crate::locks::LockReq },
}

/// Where and how `waitid` reports the child it found ([`waitid_child`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitidOut {
    /// The caller's `siginfo_t`, or zero.
    pub info: u64,
    /// `WNOWAIT`: report the child and leave it to be waited for again.
    pub nowait: bool,
    /// A 32-bit caller's `siginfo_t` (its union at 12, not 16).
    pub compat: bool,
}

impl WaitidOut {
    /// Fill the `siginfo_t` for child `pid` with wait status `word` (`wait4`'s
    /// encoding), or for no child yet (`pid` 0: Linux zeroes it, and callers
    /// read `si_pid` to tell). The call's own answer is zero either way.
    pub fn report(&self, m: &mut GuestMem<'_>, pid: i32, word: u32) -> Result<u64, Errno> {
        if self.info == 0 {
            return Ok(0);
        }
        let u = if self.compat { 12 } else { 16 };
        for off in (0..u + 12).step_by(4) {
            m.write_u32(self.info + off, 0)?;
        }
        if pid <= 0 {
            return Ok(0);
        }
        // CLD_EXITED with the exit code, CLD_KILLED or CLD_DUMPED with the signal.
        let (code, status) = if word & 0x7f == 0 {
            (1, (word >> 8) & 0xff)
        } else if word & 0x80 != 0 {
            (3, word & 0x7f)
        } else {
            (2, word & 0x7f)
        };
        m.write_u32(self.info, crate::signal::sig::SIGCHLD)?;
        m.write_u32(self.info + 8, code)?;
        m.write_u32(self.info + u, pid as u32)?;
        m.write_u32(self.info + u + 8, status)?;
        Ok(0)
    }
}

/// A parked call a signal ended that starts again ([`BlockOp::restart`],
/// [`crate::signal::Signals::restarts`]), set up in its context at
/// `ctx_addr`, its syscall's return at `pc`, before the host delivers the
/// signal. The registers are as the guest made the call, so the host
/// writes no result. The PC goes back to the call's instruction, so a
/// handler's frame returns there; that PC, which the host resumes at when
/// no frame is built. `None` when the context cannot be written (the call
/// answers `EINTR`).
///
/// A wasm32 program's PC is its stub's (the handler, then its own again),
/// so its context is marked instead ([`crate::signal::RESTART_MARK`]): the
/// i386 frame keeps the mark, and the program's sigreturn makes the call
/// again ([`Kernel::syscall`]'s `rt_sigreturn`). With no handler to run, its
/// stub could not be sent back to the call: the host clears the mark
/// ([`restart_undo`]) and answers `EINTR`.
pub fn restart_call(m: &mut GuestMem<'_>, ctx_addr: u64, pc: u64, wasm32: bool) -> Option<u64> {
    if wasm32 {
        m.write_u64(ctx_addr + ctx::x(crate::signal::RESTART_SLOT) as u64, crate::signal::RESTART_MARK).ok()?;
        return Some(pc);
    }
    let back = pc.wrapping_sub(gex_abi::SYSCALL_INST_BYTES);
    m.write_u64(ctx_addr + ctx::PC as u64, back).ok()?;
    Some(back)
}

/// A wasm32 program's restart that no handler frame took ([`restart_call`]).
pub fn restart_undo(m: &mut GuestMem<'_>, ctx_addr: u64) {
    let _ = m.write_u64(ctx_addr + ctx::x(crate::signal::RESTART_SLOT) as u64, 0);
}

/// Whether a wasm32 program's sigreturn, its context just restored, has a
/// call to make again ([`restart_call`]); the mark is taken.
fn restart_taken(m: &mut GuestMem<'_>, ctx_addr: u64) -> bool {
    let at = ctx_addr + ctx::x(crate::signal::RESTART_SLOT) as u64;
    m.read_u64(at).ok() == Some(crate::signal::RESTART_MARK) && m.write_u64(at, 0).is_ok()
}

impl BlockOp {
    /// What a signal that ends this wait makes of the call
    /// ([`crate::signal::Signals::restarts`]; signal(7)).
    pub fn restart(&self) -> crate::signal::Restart {
        use crate::signal::Restart;
        match self {
            BlockOp::FutexWait { deadline: None, .. } => Restart::SaRestart,
            BlockOp::FutexWait { .. } | BlockOp::Sleep(_) | BlockOp::SigSuspend { .. } | BlockOp::Poll { .. } => Restart::Never,
            // signal(7): never restarted, whatever SA_RESTART says.
            BlockOp::SigWait { .. } => Restart::Never,
            BlockOp::FutexLockPi { .. } => Restart::Always,
            // `connect` is not in signal(7)'s list: an interrupted connect
            // goes on in the background and says `EINTR`.
            BlockOp::Connect { .. } => Restart::Never,
            // signal(7) lists `flock` and `F_SETLKW` with the reads and writes.
            BlockOp::ReadStream { .. } | BlockOp::WriteStream { .. } | BlockOp::AcceptWait { .. } | BlockOp::WaitChild { .. } | BlockOp::FileLock { .. } => {
                Restart::SaRestart
            }
        }
    }
}

/// `clone` flags gex understands.
pub mod clone_flags {
    pub const VM: u64 = 0x0000_0100;
    pub const FS: u64 = 0x0000_0200;
    pub const PIDFD: u64 = 0x0000_1000;
    pub const VFORK: u64 = 0x0000_4000;
    pub const THREAD: u64 = 0x0001_0000;
    pub const SETTLS: u64 = 0x0008_0000;
    pub const PARENT_SETTID: u64 = 0x0010_0000;
    pub const CHILD_CLEARTID: u64 = 0x0020_0000;
    pub const CHILD_SETTID: u64 = 0x0100_0000;
}

/// Whether a path names a shared object, by the only thing available at
/// `openat`: its name.
///
/// `libc.so.6`, `ld-linux-aarch64.so.1`, `libfoo.so` -- the component is a
/// shared object when it ends in `.so` or has `.so.` in it. Reading the ELF
/// header would be exact, but an `openat` has no bytes yet and the cost of
/// being wrong is small: the mapping goes somewhere no other process will
/// look, which is where every mapping went before this existed.
fn is_shared_object(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".so") || name.contains(".so.")
}

/// Somewhere that knows where a shared object belongs.
///
/// Implemented by the host, because the answer has to be the same in every
/// process and a process's kernel state is its own. The browser answers out
/// of the global kernel every Worker shares; a host with nothing to share
/// leaves it unset and the arena allocates as it always did.
pub trait ObjectPlace: Send + Sync {
    /// Where `path` goes, given how long the reservation for it is. `None`
    /// is no opinion.
    fn place(&self, path: &str, len: u64) -> Option<u64>;
}

/// One guest process's kernel state.
pub struct Kernel {
    pub fds: Arc<FdTable>,
    pub mem: crate::mem::Mm,
    pub vfs: Vfs,
    /// Address of the running thread's context struct.
    pub ctx_addr: u64,
    pub pid: i32,
    pub tid: i32,
    /// Guest path of the running program.
    pub exe_path: String,
    /// The memory flavour the last `execve` asked for:
    /// `Some(true)` a 64-bit memory, `Some(false)` a 32-bit one, `None`
    /// the page's own. From the call's `GEX_MEM64` entry, which the kernel
    /// takes out of the new program's environment so that it applies to
    /// that exec alone and no child inherits it, else from the image's
    /// `user.gex.mem64` attribute (the learned record). The shim takes it
    /// when it places the exec.
    pub exec_mem64: Option<bool>,
    /// The argument and environment vectors, for `/proc/self/cmdline`.
    pub argv: Vec<String>,
    pub envp: Vec<String>,
    /// Its user and group ids, which its threads share and a fork copies
    /// ([`Creds`]).
    pub creds: Arc<Creds>,
    /// Trace every syscall to stderr (`gexc run --strace`).
    pub trace: bool,
    /// Signal dispositions, masks and pending set.
    pub signals: Signals,
    /// Futex wait queues. A single-threaded guest still uses these: musl's
    /// locks take the futex path whenever a lock is contended, and `EAGAIN`
    /// from an already-changed word is the answer it needs.
    pub futexes: Arc<FutexTable>,
    /// Syscalls seen but not implemented, for ENOSYS telemetry.
    pub unimplemented: Vec<u64>,
    /// The register-context slots ([`CtxSlots`]). Slot 0 is the initial
    /// thread's. Shared with the free path (`Free::slots`), which makes
    /// threads there.
    pub next_ctx_slot: Arc<CtxSlots>,
    /// Threads that have been created and not yet exited, including this one.
    /// Shared with the free path (`Free::live`), which ends a thread there.
    pub live_threads: Arc<std::sync::atomic::AtomicU32>,
    /// The system's processes: pids, parents, groups and sessions. Shared by
    /// every process, because reparenting an orphan and killing a process
    /// group are both questions no single process can answer. See [`crate::proc`].
    pub procs: Arc<ProcTable>,
    /// Where a pipe's buffer goes when the two ends may be in different
    /// address spaces. `None` keeps the local `Pipe`, which is right whenever
    /// every process really is one process. See [`crate::pipe::Remote`].
    pub pipes: Option<crate::pipe::Remote>,
    /// Where a unix socket lives when its two ends may be in different
    /// address spaces. `None` keeps the local `UnixSocket`, which is right
    /// whenever every process really is one process.
    ///
    /// It matters when processes are in separate Workers: an X server
    /// binds a name in its own Worker and its client, another Worker, looks
    /// in a name table that has never heard of it. See [`crate::unix::Remote`].
    pub unix_remote: Option<crate::unix::Remote>,
    /// Where a pseudoterminal pair lives when its master and slave may be in
    /// different address spaces (xterm and its shell). `None` keeps a local
    /// `Pty`, right whenever every process really is one process.
    pub ptys: Option<crate::pty::Remote>,
    /// This process's parent, cached from the table so `getppid` is not a
    /// lock. It never changes for a live process except when the parent dies,
    /// and `getppid` re-reads the table in that case.
    pub ppid: i32,
    /// How to read the wall clock.
    pub clock: ClockFn,
    /// An `accept` that waited, and the arguments to finish it with.
    ///
    /// A blocking operation is performed by the host with the kernel
    /// unlocked, and the host cannot allocate a descriptor. So `accept`
    /// splits: the host waits for a connection to arrive, and the work
    /// happens back here.
    pub retry_accept: Option<(i32, u64, u64, u64)>,
    /// A write parked as a blocking operation: its `EPIPE`, when it comes
    /// back through `finish_blocking`, raises SIGPIPE as a direct one does.
    pub write_blocked: bool,
    /// A `recvmsg` on a blocking socket with nothing to read: the fd and the
    /// `msghdr`, re-run once the socket becomes readable. Like `retry_accept`,
    /// the real work waits for the kernel lock (the msghdr may carry
    /// descriptors, which only the kernel's table can place). Wine's protocol
    /// with wineserver is a blocking `recvmsg` for every reply.
    pub retry_recvmsg: Option<(i32, u64, u64)>,
    /// A `splice`, `sendfile` or `copy_file_range` (number, arguments) from
    /// an empty blocking pipe or socket, re-run once it is readable, as
    /// `retry_recvmsg`.
    pub retry_transfer: Option<(u64, [u64; 6])>,
    /// The blocked mask an `rt_sigsuspend` has to put back when it returns.
    pub restore_sigmask: Option<u64>,
    /// An `rt_sigtimedwait` that waited (`BlockOp::SigWait`): its arguments,
    /// to take the signal with once one of its set is pending.
    pub retry_sigwait: Option<[u64; 6]>,
    /// A lock request on this descriptor parked (`BlockOp::FileLock`): once
    /// the host has granted it, the file's shared mappings are brought up
    /// to date ([`sync_shared_file`]), as the call itself does.
    pub retry_lock: Option<i32>,
    /// A 32-bit (i386) process: its code runs in compatibility mode, it
    /// enters with `int 0x80`, and its numbering and structures are the i386
    /// ones (see `crate::i386`). Set at exec from the image's class,
    /// inherited across fork.
    pub compat: bool,
    /// Set by a syscall that may have made a descriptor readable or
    /// writable for someone else (a write, a read that freed room, a close,
    /// a signal); the host takes it after the call and wakes threads parked
    /// on descriptors ([`Kernel::take_fd_activity`]).
    pub fd_activity: bool,
    /// The wasm32 personality: a native wasm32
    /// program on the i386 ABI. Everything is i386's except how control
    /// moves: a signal frame or a `sigreturn` is left in the thread context
    /// and the syscall returns normally ([`SyscallOutcome::Continue`]), for
    /// the program's own syscall stub to run the handler or carry on -- a
    /// wasm program's frames cannot be resumed at a new guest PC. Set by the
    /// host for each program it starts; kept across fork, cleared by exec.
    pub wasm32: bool,
    /// The i386 process entered through the non-rt `sigreturn` (119), whose
    /// frame is laid out differently from `rt_sigreturn`'s. Set at entry,
    /// consumed by `sigreturn`.
    pub sigreturn_legacy: bool,
    /// When this process began, on the guest's clock, so it can be asked how
    /// much time it has used. See [`Kernel::cpu_nanos`].
    ///
    /// Zero until something asks, rather than set at construction: a host may
    /// replace `clock` after building the kernel -- the browser does, because
    /// `SystemTime` panics there -- and a start time taken from the clock that
    /// was in place beforehand is fifty years out.
    pub started: u128,
    /// Where connections come from, when the host has anywhere to get them.
    /// `None` is a guest with no network, which is every guest until a host
    /// says otherwise.
    pub net: Option<Arc<dyn net::NetStack>>,
    /// Names the guest has resolved. Shared across `fork`, because a child
    /// connecting to an address its parent resolved must reach the same host.
    pub resolver: Arc<net::Resolver>,
    /// The TCP ports guests listen on. Shared across `fork` like the resolver.
    pub ports: Arc<net::Ports>,
    /// Which unix socket answers to which path. Shared system-wide, because
    /// a server and its clients are different processes and the whole point
    /// of a named socket is that they find each other by name.
    pub socket_names: Arc<unix::SocketNames>,
    /// Names *this process's host* listens on, which is not the same table.
    ///
    /// The compositor and the GPU device are in the shim, one per process
    /// (`mount_devices` gives every adopted child its own), and they can only
    /// read the memory of the process they are in -- a `wl_shm` pool is guest
    /// memory. So `wayland-0` must resolve to the compositor of the process
    /// that is connecting, and not to whichever one bound the name last. It is
    /// deliberately NOT shared across a fork: a child gets its own.
    pub host_names: Arc<unix::SocketNames>,
    /// Advisory whole-file locks, shared for the same reason.
    pub file_locks: Arc<unix::FileLocks>,
    /// Where each object of a statically recompiled closure must be placed.
    ///
    /// A translated block has its guest addresses baked into it, so a module
    /// built ahead of time is only correct if the object it was built from
    /// lands where the build assumed. Nothing needs to *predict* that: the
    /// guest's `ld.so` reserves each object with `mmap(NULL, ..)` and takes
    /// whatever address it is given, so the layout is gex's to assign. This
    /// is the assignment, by guest path.
    ///
    /// Empty unless a closure build asked for it, and then honoured only for
    /// the reservation described at `MapLog::armed_pin`.
    pub pins: Arc<std::collections::HashMap<String, u64>>,
    /// Who decides where a shared object goes, when anybody does.
    ///
    /// [`Self::pins`] is the same idea decided ahead of time by a closure
    /// build. This one is decided as the run goes: the first process to map
    /// an object fixes its address and every later one is given the same,
    /// which is what makes a translated module worth sharing between them.
    pub place: Option<Arc<dyn ObjectPlace>>,
    /// Every thread's own signal state but the running one's, which is in
    /// `signals` (`Signals::save_thread`/`load_thread`): masks, thread-directed
    /// pending signals and alternate stacks are per thread, as on Linux.
    /// Behind its own lock, shared with the host's lock-free calls, which
    /// signal a thread through it (`tgkill`).
    pub thread_signals: Arc<crate::signal::ThreadSigs>,
    /// The thread whose part `signals` holds now.
    pub sig_tid: i32,
    /// Every other thread's half-finished blocking call (`PendingCall`),
    /// swapped with `signals`: one thread's `rt_sigsuspend` must not hand
    /// its mask to the next thread that finishes a futex wait (Mono's GC
    /// left the main thread with SIGPWR blocked that way).
    pub thread_calls: std::collections::HashMap<i32, PendingCall>,
    /// The pc of the syscall being handled (a mapping the dynamic loader
    /// asks for stays below 4 GiB, since it will hold code).
    pub syscall_pc: u64,
    /// The restart class of the call that last parked
    /// ([`BlockOp::restart`]) and its syscall's return pc, for
    /// [`Self::finish_blocking`]; per thread (`PendingCall`).
    pub block_restart: crate::signal::Restart,
    pub block_pc: u64,
    /// The canonical number of the syscall being handled (for the host's
    /// `GEX_SYS_WAITS`, which sees only the guest's own numbering).
    pub syscall_nr: u64,
    /// The process's first fs_struct: its working directory and umask,
    /// every thread's that has not made one of its own.
    pub proc_fs: Arc<FsCwd>,
    /// The fs_structs other than the process's first one: a thread's after
    /// `unshare(CLONE_FS)` or a `clone` without `CLONE_FS`, shared with the
    /// threads it creates with it. A thread's directory is entered into
    /// `vfs` for each of its calls made under the kernel lock
    /// (`syscall_cwd`); its calls made without the lock read it here.
    pub thread_fs: Arc<ThreadFs>,
    /// The fs_struct of the thread whose `fork` or `execve` the host is
    /// carrying out: what the child or the new image starts in
    /// ([`Kernel::caller_fs`], [`Kernel::exec_signals`]). Kept from the call
    /// itself: by the time the host acts, another thread's call may have run.
    pub call_fs: Option<Arc<FsCwd>>,
    /// Serial for the unnamed files `O_TMPFILE` asks for.
    pub tmpfiles: Arc<std::sync::atomic::AtomicU64>,
    /// `getrlimit`/`setrlimit` (prlimit64): soft and hard per resource,
    /// behind their own lock ([`Rlimits`]). RLIMIT_NOFILE's soft limit is
    /// the descriptor table's (`FdTable`).
    pub rlimits: Arc<Rlimits>,
    /// The terminal: flags, size, foreground process group, and the line
    /// discipline behind them. See [`crate::tty`].
    ///
    /// Shared, not copied, and that is the point of it. There is one terminal
    /// and its settings outlive any process using it: `stty -echo` in one
    /// command must still be off in the next, a shell that puts the terminal
    /// in raw mode and dies must leave it that way for whoever cleans up, and
    /// `^C` goes to the *foreground group* -- which is a fact about the
    /// system, not about the process that happened to type it.
    pub tty: Arc<crate::tty::TtyRef>,
}

pub use crate::signal::Itimer;

/// The three interval timers, in the order `which` names them.
/// Resource limits (prlimit64): Linux's numbering and its defaults for a
/// login process. RLIM_INFINITY is all ones.
pub const RLIM_NLIMITS: usize = 16;
pub const RLIMIT_STACK: usize = 3;
pub const RLIMIT_CORE: usize = 4;
pub const RLIMIT_NOFILE: usize = 7;
pub const RLIMIT_NPROC: usize = 6;
pub const RLIMIT_MEMLOCK: usize = 8;
pub const RLIMIT_SIGPENDING: usize = 11;
pub const RLIMIT_MSGQUEUE: usize = 12;
pub const RLIM_INFINITY: u64 = u64::MAX;

/// A process's resource limits, behind their own lock: `prlimit64` sets
/// them without the kernel lock.
pub struct Rlimits(std::sync::Mutex<[[u64; 2]; RLIM_NLIMITS]>);

impl Default for Rlimits {
    fn default() -> Rlimits {
        Rlimits(std::sync::Mutex::new(default_rlimits()))
    }
}

impl Rlimits {
    /// Soft and hard, per resource, now.
    pub fn get(&self) -> [[u64; 2]; RLIM_NLIMITS] {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replace them all (a process built from another's state).
    pub fn set(&self, l: [[u64; 2]; RLIM_NLIMITS]) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = l;
    }

    /// A copy of its own, for a fork's child.
    pub fn copy(&self) -> Rlimits {
        Rlimits(std::sync::Mutex::new(self.get()))
    }
}

pub fn default_rlimits() -> [[u64; 2]; RLIM_NLIMITS] {
    let mut l = [[RLIM_INFINITY; 2]; RLIM_NLIMITS];
    l[RLIMIT_STACK] = [8 << 20, RLIM_INFINITY];
    l[RLIMIT_CORE] = [0, RLIM_INFINITY];
    l[RLIMIT_NOFILE] = [crate::file::NOFILE_SOFT as u64, crate::file::NOFILE_HARD as u64];
    l[RLIMIT_NPROC] = [63_000, 63_000];
    l[RLIMIT_MEMLOCK] = [8 << 20, 8 << 20];
    l[RLIMIT_SIGPENDING] = [63_000, 63_000];
    l[RLIMIT_MSGQUEUE] = [819_200, 819_200];
    l
}

pub const ITIMER_REAL: usize = 0;
pub const ITIMER_VIRTUAL: usize = 1;
pub const ITIMER_PROF: usize = 2;

/// How the host tells the time: seconds and nanoseconds since the epoch.
///
/// A function rather than a call to `SystemTime`, because there are hosts
/// without one. `wasm32-unknown-unknown` has no clock at all -- `SystemTime`
/// panics there -- so the browser host passes its own, and everything else
/// gets [`system_clock`].
pub type ClockFn = fn() -> (u64, u32);

/// Whether an `EPIPE` from syscall `n` also raises SIGPIPE, as Linux does
/// for a write to a pipe or stream socket with no reader -- except a `send`
/// with `MSG_NOSIGNAL`. (A shell pipeline's `yes | head` ends with the
/// writer killed by it; without it the writer sees EPIPE and complains.)
fn raises_sigpipe(n: u64, a: &[u64; 6]) -> bool {
    const MSG_NOSIGNAL: u64 = 0x4000;
    match n {
        nr::WRITE | nr::WRITEV | nr::PWRITE64 | nr::SENDFILE | nr::SPLICE => true,
        nr::SENDTO => a[3] & MSG_NOSIGNAL == 0,
        nr::SENDMSG => a[2] & MSG_NOSIGNAL == 0,
        nr::SENDMMSG => a[3] & MSG_NOSIGNAL == 0,
        _ => false,
    }
}

/// Read a `struct timeval` -- seconds then microseconds -- as nanoseconds.
fn read_timeval(m: &GuestMem<'_>, at: u64) -> Result<u128, Errno> {
    let secs = m.read_u64(at)?;
    let usecs = m.read_u64(at + 8)?;
    Ok(secs as u128 * 1_000_000_000 + (usecs as u128).min(999_999) * 1_000)
}

/// Write nanoseconds back as a `struct timeval`.
fn write_timeval(m: &mut GuestMem<'_>, at: u64, nanos: u128) -> Result<(), Errno> {
    m.write_u64(at, (nanos / 1_000_000_000) as u64)?;
    m.write_u64(at + 8, (nanos % 1_000_000_000 / 1_000) as u64)?;
    Ok(())
}

/// The clock of a host that has one.
/// CLOCK_MONOTONIC and its kin (RAW 4, COARSE 6, BOOTTIME 7, BOOTTIME_ALARM 9).
fn is_monotonic_clock(clk: u64) -> bool {
    matches!(clk, 1 | 4 | 6 | 7 | 9)
}

/// Where the monotonic clocks start: the wall clock at the first reading,
/// less [`MONOTONIC_START_SECS`]. One base for every guest process of this
/// host process, as on a machine every process shares the boot time.
static MONOTONIC_BASE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The "uptime" the guest sees at its first monotonic reading.
const MONOTONIC_START_SECS: u64 = 1000;
/// A host whose processes do not share this static (the browser's, a
/// Worker and an instance each) names one base for all of them: given the
/// base this process would pick, the one every process uses.
static MONOTONIC_SHARED: std::sync::OnceLock<fn(u64) -> u64> = std::sync::OnceLock::new();

/// See [`MONOTONIC_SHARED`]. Set before the first clock reading.
pub fn share_monotonic_base(f: fn(u64) -> u64) {
    let _ = MONOTONIC_SHARED.set(f);
}

/// Nanoseconds the monotonic clocks have been held back by: the time a
/// restored session snapshot spent saved, which its clocks do not count
/// (as a machine's do not while it is suspended). 0 for a session never
/// restored.
static MONOTONIC_HELD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// A host whose processes do not share this static: given nanoseconds to
/// add (0 to only ask), how long every process's clocks are held back.
static MONOTONIC_HELD_SHARED: std::sync::OnceLock<fn(u64) -> u64> = std::sync::OnceLock::new();

/// See [`MONOTONIC_HELD_SHARED`]. Set before the first clock reading.
pub fn share_monotonic_held(f: fn(u64) -> u64) {
    let _ = MONOTONIC_HELD_SHARED.set(f);
}

/// Hold this process's monotonic clocks back `ns` (a restored session's,
/// the time it spent saved, as the host shares it).
pub fn set_monotonic_held(ns: u64) {
    MONOTONIC_HELD.store(ns, std::sync::atomic::Ordering::Relaxed);
}

/// How long this process's monotonic clocks are held back (nanoseconds): a
/// counter of the host's time read without them (the x86 time-stamp
/// counter) takes it off too.
pub fn monotonic_held() -> u64 {
    MONOTONIC_HELD.load(std::sync::atomic::Ordering::Relaxed)
}

/// How much faster than real time the monotonic clocks run, as f64 bits
/// (0: real time). For benchmarks only, off unless a host sets it
/// ([`set_guest_clock`]).
static CLOCK_SCALE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Seconds taken off the wall clock (CLOCK_REALTIME, gettimeofday, time) as
/// the guest reads it: 0, or now less a fixed starting instant.
static WALL_SHIFT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// The benchmark clocks: the monotonic clocks -- which Wine's
/// QueryPerformanceCounter, GetTickCount and timeGetTime all come from --
/// run `scale` times faster than real time, so a program that caps its own
/// frame rate against them stops sleeping and runs as fast as it can; the
/// wall clock runs at real time but `wall_shift` seconds behind, so it
/// starts at a fixed date (the fixed date plus the real time since start).
/// Sleeps and timeouts stay real time: Wine's
/// NtDelayExecution counts on the wall clock, and a relative wait on the
/// monotonic one waits its length in real time. Every process of a host
/// must be given the same values (the browser's kernel hands them out).
pub fn set_guest_clock(scale: f64, wall_shift: i64) {
    use std::sync::atomic::Ordering;
    let bits = if scale > 0.0 && scale != 1.0 { scale.to_bits() } else { 0 };
    CLOCK_SCALE.store(bits, Ordering::Relaxed);
    WALL_SHIFT.store(wall_shift, Ordering::Relaxed);
}

/// The wall clock as the guest reads it: see [`set_guest_clock`].
fn guest_wall((secs, nanos): (u64, u32)) -> (u64, u32) {
    let shift = WALL_SHIFT.load(std::sync::atomic::Ordering::Relaxed);
    (secs.wrapping_sub(shift as u64), nanos)
}

/// A monotonic clock reading made from a wall clock reading: seconds since
/// a shared boot instant, not since the epoch. Programs truncate monotonic
/// milliseconds to 32 bits and compare them across processes -- X server
/// event times are `(CARD32)` CLOCK_MONOTONIC milliseconds, and toolkits
/// compare them with their own clock -- which only works while the uptime
/// is under 49.7 days, as it is on real machines; an epoch-based value
/// (1.8e12 ms) never agrees with its own truncation.
/// `clock_gettime(clk)` in nanoseconds from `clock`, as [`Kernel`]'s
/// `sys_clock_gettime` answers it for every clock but the two CPU-time ones
/// (0 for those): no kernel, no lock. For a host import that reads the
/// guest's clock directly (`gex.clock_ns`, Wine's PE ntdll
/// NtQueryPerformanceCounter), so the value is the
/// one a system call would give.
pub fn clock_ns(clock: ClockFn, clk: u64) -> u64 {
    let (secs, nanos) = match clk {
        2 | 3 => return 0,
        c if is_monotonic_clock(c) => monotonic_from(clock()),
        _ => guest_wall(clock()),
    };
    secs * 1_000_000_000 + nanos as u64
}

fn monotonic_from((secs, nanos): (u64, u32)) -> (u64, u32) {
    use std::sync::atomic::Ordering;
    let mut base = MONOTONIC_BASE.load(Ordering::Relaxed);
    if base == 0 {
        let mut want = secs.saturating_sub(MONOTONIC_START_SECS).max(1);
        if let Some(f) = MONOTONIC_SHARED.get() {
            want = f(want);
        }
        if let Some(f) = MONOTONIC_HELD_SHARED.get() {
            MONOTONIC_HELD.store(f(0), Ordering::Relaxed);
        }
        base = match MONOTONIC_BASE.compare_exchange(0, want, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => want,
            Err(other) => other,
        };
    }
    let (mut secs, mut nanos) = (secs.saturating_sub(base), nanos);
    let held = MONOTONIC_HELD.load(Ordering::Relaxed);
    if held != 0 {
        let ns = (secs * 1_000_000_000 + nanos as u64).saturating_sub(held);
        (secs, nanos) = (ns / 1_000_000_000, (ns % 1_000_000_000) as u32);
    }
    let scale = CLOCK_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return (secs, nanos);
    }
    let ns = ((secs as f64 * 1e9 + nanos as f64) * f64::from_bits(scale)) as u64;
    (ns / 1_000_000_000, (ns % 1_000_000_000) as u32)
}

/// `getrusage`'s `who`: the process (`RUSAGE_SELF`, 0), its reaped
/// children (`RUSAGE_CHILDREN`, -1) or the calling thread (`RUSAGE_THREAD`,
/// 1); the time each has used, from `self_ns`, `child_ns` and `thread_ns`.
fn rusage_ns(who: u64, self_ns: impl FnOnce() -> u128, child_ns: impl FnOnce() -> u128, thread_ns: impl FnOnce() -> u128) -> Result<u128, Errno> {
    match who as u32 as i32 {
        0 => Ok(self_ns()),
        -1 => Ok(child_ns()),
        1 => Ok(thread_ns()),
        _ => Err(Errno::EINVAL),
    }
}

/// The time `pid`'s reaped children used, as the process table keeps it.
pub fn child_cpu_of(procs: &ProcTable, pid: i32) -> u128 {
    procs.entries().into_iter().find(|e| e.pid == pid).map_or(0, |e| e.usage.child_cpu_ns as u128)
}

/// A call's answer as the word its result register holds: the value, or
/// the negated errno.
fn result_word(r: SysResult) -> u64 {
    match r {
        Ok(v) => v,
        Err(e) => (e.0 as i64).wrapping_neg() as u64,
    }
}

/// `times`: four `clock_t` -- user, system, and the same for reaped
/// children -- counted in the hundredths of a second that `USER_HZ` means
/// on Linux, and a return value on the same scale; `cpu_ns` the processor
/// time used, `child_ns` its reaped children's. i386's are 32-bit.
fn times_answer(m: &mut GuestMem<'_>, compat: bool, cpu_ns: u128, child_ns: u128, buf: u64) -> SysResult {
    let ticks = (cpu_ns / 10_000_000) as u64;
    let child = (child_ns / 10_000_000) as u64;
    if buf != 0 && compat {
        m.zero(buf, 16)?;
        m.write_u32(buf, ticks as u32)?;
        m.write_u32(buf + 8, child as u32)?;
    } else if buf != 0 {
        m.zero(buf, 32)?;
        m.write_u64(buf, ticks)?;
        m.write_u64(buf + 16, child)?;
    }
    Ok(if compat { ticks as u32 as u64 } else { ticks })
}

/// `time`: the wall clock's seconds, also stored at `tloc` unless it is
/// null -- as a 32-bit `time_t` for i386, whose 64-bit store wrote four
/// bytes past the caller's variable.
fn time_answer(m: &mut GuestMem<'_>, compat: bool, clock: ClockFn, tloc: u64) -> SysResult {
    let (secs, _) = guest_wall(clock());
    if tloc != 0 {
        if compat {
            m.write_u32(tloc, secs as u32)?;
        } else {
            m.write_u64(tloc, secs)?;
        }
    }
    Ok(if compat { secs as u32 as u64 } else { secs })
}

/// Answer one of the calls that only read the clock -- `clock_gettime`,
/// `gettimeofday`, `getrusage`, `times`, `time`, `sched_yield` -- straight from the context at
/// `ctx_addr`, without a [`Kernel`], and say whether it was one.
///
/// A host takes its kernel lock for every syscall, and Wine asks the time
/// several times per sleep: a game's main thread can make thousands of
/// these calls a second, each one queueing for the lock behind the
/// process's other Workers. None of them
/// needs the kernel: the answer is the clock, and `started` (the process's
/// [`Kernel::started`], which the processor-time clocks count from).
///
/// Returns false, writing nothing, for any other call. A clock call is
/// always answered here, `EFAULT` included (a buffer the guest cannot be
/// written at, as the kernel's own calls answer), so a host that routes these
/// numbers here never takes its kernel for them. `compat` is
/// [`Kernel::compat`] (the i386 numbering, number in
/// `eax`, 32-bit layouts); `started` the process's start, which the
/// processor-time clocks count from; `children` the time its reaped children
/// used (`times`, `getrusage(RUSAGE_CHILDREN)`). The result goes where [`set_result`]
/// puts it. Signal delivery, which the kernel does at every syscall
/// boundary, is the host's to arrange (the page's safepoints).
pub fn clock_call(
    m: &mut GuestMem<'_>,
    ctx_addr: u64,
    compat: bool,
    clock: ClockFn,
    started: u128,
    children: &dyn Fn() -> u128,
) -> bool {
    let (raw, a0, a1) = if compat {
        let r = |slot: usize| m.read_u64(ctx_addr + slot as u64).unwrap_or(!0) & 0xffff_ffff;
        (
            r(ctx::x(0)),
            r(crate::i386::ARG_SLOTS[0]),
            r(crate::i386::ARG_SLOTS[1]),
        )
    } else {
        let r = |slot: usize| m.read_u64(ctx_addr + slot as u64).unwrap_or(!0);
        (r(ctx::SYSCALL_NR), r(ctx::SYSCALL_ARGS[0]), r(ctx::SYSCALL_ARGS[1]))
    };
    enum Call {
        // (clock id, buffer, 32-bit timespec)
        Clock(u64, u64, bool),
        // (buffer, 32-bit timespec)
        Res(u64, bool),
        // (buffer, 32-bit timeval)
        Tod(u64, bool),
        // (who, buffer)
        Rusage(u64, u64),
        // (buffer): `times`, in the ABI's `clock_t`.
        Times(u64),
        // (where to store it, or 0): `time`, x86-64's and i386's.
        Time(u64),
        Yield,
    }
    let call = if compat {
        use crate::i386::raw as x;
        match raw {
            x::CLOCK_GETTIME => Call::Clock(a0, a1, true),
            x::CLOCK_GETTIME64 => Call::Clock(a0, a1, false),
            x::CLOCK_GETRES => Call::Res(a1, true),
            x::CLOCK_GETRES_TIME64 => Call::Res(a1, false),
            x::GETTIMEOFDAY => Call::Tod(a0, true),
            x::GETRUSAGE => Call::Rusage(a0, a1),
            x::TIMES => Call::Times(a0),
            x::TIME => Call::Time(a0),
            x::SCHED_YIELD => Call::Yield,
            _ => return false,
        }
    } else if gex_abi::X86 {
        // x86-64's own numbers: these four pass through `x86::translate`
        // unchanged but for the number.
        match raw {
            228 => Call::Clock(a0, a1, false),
            229 => Call::Res(a1, false),
            96 => Call::Tod(a0, false),
            98 => Call::Rusage(a0, a1),
            100 => Call::Times(a0),
            201 => Call::Time(a0),
            24 => Call::Yield,
            _ => return false,
        }
    } else {
        match raw {
            nr::CLOCK_GETTIME => Call::Clock(a0, a1, false),
            nr::CLOCK_GETRES => Call::Res(a1, false),
            nr::GETTIMEOFDAY => Call::Tod(a0, false),
            nr::GETRUSAGE => Call::Rusage(a0, a1),
            nr::TIMES => Call::Times(a0),
            nr::SCHED_YIELD => Call::Yield,
            _ => return false,
        }
    };
    // Before the host has said when the process started (0): no time used.
    // A host that measures processor time says it (`crate::proc::set_cpu_clock`).
    let cpu_of = |thread: bool| {
        if let Some(ns) = crate::proc::cpu_clock(thread) {
            return ns as u128;
        }
        let (s, n) = clock();
        if started == 0 { 0 } else { (s as u128 * 1_000_000_000 + n as u128).saturating_sub(started) }
    };
    let cpu = || cpu_of(false);
    let pair = |m: &mut GuestMem<'_>, at: u64, hi: u64, lo: u64, w32: bool| {
        if w32 {
            m.write_u32(at, hi as u32).and_then(|_| m.write_u32(at + 4, lo as u32))
        } else {
            m.write_u64(at, hi).and_then(|_| m.write_u64(at + 8, lo))
        }
    };
    let ok = match call {
        Call::Clock(clk, buf, w32) => {
            // As `sys_clock_gettime`, which a null buffer faults in.
            let (secs, nanos) = match clk {
                2 | 3 => {
                    let t = cpu_of(clk == 3);
                    ((t / 1_000_000_000) as u64, (t % 1_000_000_000) as u32)
                }
                c if is_monotonic_clock(c) => monotonic_from(clock()),
                _ => guest_wall(clock()),
            };
            buf != 0 && pair(m, buf, secs, nanos as u64, w32).is_ok()
        }
        Call::Res(buf, w32) => buf == 0 || pair(m, buf, 0, 1, w32).is_ok(),
        Call::Tod(buf, w32) => {
            if buf == 0 {
                true
            } else {
                let (secs, nanos) = guest_wall(clock());
                pair(m, buf, secs, (nanos / 1000) as u64, w32).is_ok()
            }
        }
        Call::Rusage(who, buf) => match rusage_ns(who, cpu, &children, || cpu_of(true)) {
            Err(e) => {
                if gex_abi::X86 {
                    let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
                }
                let _ = set_result(m, ctx_addr, e.as_return());
                return true;
            }
            Ok(_) if buf == 0 => true,
            Ok(ns) => {
                let (s, us) = ((ns / 1_000_000_000) as u64, (ns % 1_000_000_000 / 1000) as u64);
                let size = if compat { 72 } else { 144 };
                m.zero(buf, size).is_ok() && pair(m, buf, s, us, compat).is_ok()
            }
        },
        // A pure yield: the caller's Worker is its own OS thread, and the
        // browser schedules those without being asked.
        Call::Yield => true,
        Call::Times(buf) => {
            let r = times_answer(m, compat, cpu(), children(), buf);
            if gex_abi::X86 {
                let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
            }
            let _ = set_result(m, ctx_addr, result_word(r));
            return true;
        }
        Call::Time(tloc) => {
            let r = time_answer(m, compat, clock, tloc);
            if gex_abi::X86 {
                let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
            }
            let _ = set_result(m, ctx_addr, result_word(r));
            return true;
        }
    };
    if gex_abi::X86 {
        let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
    }
    let ret = if ok { 0 } else { (Errno::EFAULT.0 as i64).wrapping_neg() as u64 };
    let _ = set_result(m, ctx_addr, ret);
    true
}

// -- The descriptor half --------------------------------------------------
//
// `read`, `write`, their vectored and positioned forms and `lseek` need the
// descriptor table and nothing else of a kernel, blocking included: the
// full call and a host calling without the kernel lock ([`free_call`]) run
// the same code.

/// The descriptor table, and the ABI its `iovec`s are in.
#[derive(Clone, Copy)]
pub struct FdIo<'a> {
    pub fds: &'a FdTable,
    pub compat: bool,
    /// A wasm32 program ([`Kernel::wasm32`]): i386's numbering, its own
    /// layouts where they differ (a `struct flock`'s 64-bit offsets are
    /// 8-aligned).
    pub wasm32: bool,
    /// The process's namespace, for a descriptor rebuilt from its name
    /// (`SCM_RIGHTS` from another process), and its pid.
    pub ns: &'a crate::vfs::Namespace,
    pub pid: i32,
    /// The process's user and group, which a socket it joins records
    /// (`SO_PEERCRED`).
    pub creds: &'a Creds,
    /// `flock`'s locks (`Kernel::file_locks`, shared with the process's
    /// forks), released when a description goes.
    pub file_locks: &'a Arc<unix::FileLocks>,
    /// The global kernel's pipes, when there is one (`Kernel::pipes`).
    pub pipes: Option<crate::pipe::Remote>,
    /// The process's shared file mappings, which its reads and writes of a
    /// mapped file keep up with (`Mm::shared_maps`).
    pub shared: Option<&'a SharedMapView>,
}

impl FdIo<'_> {
    /// This process's credentials as `struct ucred` has them.
    pub fn cred(&self) -> unix::Cred {
        unix::Cred { pid: self.pid, uid: self.creds.uid(), gid: self.creds.gid() }
    }

    /// `eventfd2`: a counting descriptor an event loop pokes to wake itself.
    ///
    /// One descriptor, readable and writable, over a single `EventFd`. The
    /// `EFD_NONBLOCK`/`EFD_CLOEXEC` flags are `O_NONBLOCK`/`O_CLOEXEC` by
    /// another name, so they go straight into the open flags and the
    /// close-on-exec bit; only `EFD_SEMAPHORE` changes the object.
    pub fn eventfd2(&self, initval: u64, flags: u64) -> SysResult {
        use crate::eventfd::{flags as efd, EventFd};
        // anything else is refused, as Linux does (Chromium's Mojo probes
        // for eventfd2 with every bit set and expects EINVAL)
        if flags & 0xffff_ffff & !(efd::SEMAPHORE | oflags::O_CLOEXEC | oflags::O_NONBLOCK) != 0 {
            return Err(Errno::EINVAL);
        }
        let handle = Arc::new(EventFd::new(initval, flags & efd::SEMAPHORE != 0));
        let cloexec = flags & oflags::O_CLOEXEC != 0;
        let file = OpenFile::new(
            handle,
            "anon_inode:[eventfd]".to_string(),
            (flags & oflags::O_NONBLOCK) | oflags::O_RDWR,
            true,
            true,
        );
        self.fds.insert(file, cloexec)
    }
    /// `timerfd_create`: a clock as a descriptor for an event loop to poll.
    pub fn timerfd_create(&self, clock: ClockFn, clockid: u64, flags: u64) -> SysResult {
        let mut timer = crate::timerfd::TimerFd::new(clock);
        if is_monotonic_clock(clockid) {
            timer = timer.monotonic(monotonic_from);
        }
        let handle = Arc::new(timer);
        let cloexec = flags & oflags::O_CLOEXEC != 0;
        let file = OpenFile::new(
            handle,
            "anon_inode:[timerfd]".to_string(),
            (flags & oflags::O_NONBLOCK) | oflags::O_RDONLY,
            true,
            false,
        );
        self.fds.insert(file, cloexec)
    }
    /// `timerfd_settime`: arm or disarm the timer from an `itimerspec`.
    ///
    /// The struct is two `timespec`s -- `it_interval` then `it_value`, sixteen
    /// bytes each. `it_value` of zero disarms; the `TFD_TIMER_ABSTIME` flag
    /// makes it an absolute time on the clock. The old setting is written back
    /// through `old` when it is non-null, the same shape the kernel returns.
    pub fn timerfd_settime(
        &self,
        m: &mut GuestMem<'_>,
        fd: i32,
        flags: u64,
        new: u64,
        old: u64,
    ) -> SysResult {
        let read_ns = |m: &GuestMem<'_>, at: u64| -> Result<u64, Errno> {
            let secs = m.read_u64(at)?;
            let nanos = m.read_u64(at + 8)?;
            Ok(secs.saturating_mul(1_000_000_000).saturating_add(nanos))
        };
        let interval = read_ns(m, new)?;
        let value = read_ns(m, new + 16)?;
        let handle = self.fds.get(fd)?.handle.clone();
        let (rem, prev_int) = handle
            .timerfd_set(value, interval, flags & crate::timerfd::TFD_TIMER_ABSTIME != 0)
            .ok_or(Errno::EINVAL)?;
        if old != 0 {
            m.write_u64(old, prev_int / 1_000_000_000)?;
            m.write_u64(old + 8, prev_int % 1_000_000_000)?;
            m.write_u64(old + 16, rem / 1_000_000_000)?;
            m.write_u64(old + 24, rem % 1_000_000_000)?;
        }
        Ok(0)
    }
    /// `timerfd_gettime`: write the time to the next expiry and the interval.
    pub fn timerfd_gettime(&self, m: &mut GuestMem<'_>, fd: i32, curr: u64) -> SysResult {
        if curr == 0 {
            return Ok(0);
        }
        let handle = self.fds.get(fd)?.handle.clone();
        let (rem, interval) = handle.timerfd_get().ok_or(Errno::EINVAL)?;
        m.write_u64(curr, interval / 1_000_000_000)?;
        m.write_u64(curr + 8, interval % 1_000_000_000)?;
        m.write_u64(curr + 16, rem / 1_000_000_000)?;
        m.write_u64(curr + 24, rem % 1_000_000_000)?;
        Ok(0)
    }
    /// `inotify_init1`: an instance as a descriptor (`IN_NONBLOCK` and
    /// `IN_CLOEXEC` are `O_NONBLOCK` and `O_CLOEXEC`; any other flag is
    /// `EINVAL`).
    pub fn inotify_init1(&self, flags: u64) -> SysResult {
        if flags & !(oflags::O_NONBLOCK | oflags::O_CLOEXEC) != 0 {
            return Err(Errno::EINVAL);
        }
        let handle: Arc<dyn crate::vfs::FileHandle> = Arc::new(crate::inotify::Inotify::new()?);
        let file = OpenFile::new(handle, "anon_inode:inotify".to_string(), (flags & oflags::O_NONBLOCK) | oflags::O_RDONLY, true, false);
        self.fds.insert(file, flags & oflags::O_CLOEXEC != 0)
    }

    /// The instance descriptor `fd` is: `EBADF` for no descriptor, `EINVAL`
    /// for one that is not an instance.
    fn inotify_of(&self, fd: i32) -> Result<Arc<crate::inotify::Inotify>, Errno> {
        let file = self.fds.get(fd)?;
        Arc::clone(&file.handle).as_inotify().ok_or(Errno::EINVAL)
    }

    /// `inotify_add_watch`: a watch on what `path` names, by the filesystem
    /// holding it ([`crate::vfs::FileSystem::watch_key`]).
    pub fn inotify_add_watch(&self, m: &GuestMem<'_>, fd: i32, path: u64, mask: u64) -> SysResult {
        use crate::inotify::mask as im;
        let inst = self.inotify_of(fd)?;
        let raw = m.read_cstr(path, 4095)?;
        let path = String::from_utf8_lossy(&raw).into_owned();
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        let mask = mask as u32;
        let (fs, rel) = self.ns.watch_target(&path, mask & im::DONT_FOLLOW == 0)?;
        if fs.watched_remotely() {
            let isdir = fs.metadata(&rel)?.file_type == crate::vfs::FileType::Directory;
            return inst.add_remote(&rel, isdir, mask);
        }
        let (key, isdir) = fs.watch_key(&rel)?;
        inst.add_local(key, isdir, mask)
    }

    pub fn inotify_rm_watch(&self, fd: i32, wd: i32) -> SysResult {
        self.inotify_of(fd)?.rm_watch(wd)
    }

    pub fn memfd_create(&self, m: &GuestMem<'_>, name: u64, flags: u64) -> SysResult {
        const MFD_CLOEXEC: u64 = 1;
        // MFD_ALLOW_SEALING, MFD_HUGETLB (with its page size in bits 26-31),
        // MFD_NOEXEC_SEAL and MFD_EXEC: anything else is refused, as Linux
        // does, and callers probe for memfd_create with every bit set
        // (Chromium's Mojo expects EINVAL)
        const MFD_HUGETLB: u64 = 4;
        let known = MFD_CLOEXEC | 2 | MFD_HUGETLB | 8 | 0x10 | if flags & MFD_HUGETLB != 0 { 0x3f << 26 } else { 0 };
        let flags = flags & 0xffff_ffff;
        if flags & !known != 0 {
            return Err(Errno::EINVAL);
        }
        let raw = m.read_cstr(name, 249)?;
        let name = String::from_utf8_lossy(&raw).into_owned();
        let memfd = Arc::new(unix::MemFd::new(&name));
        let handle: Arc<dyn crate::vfs::FileHandle> = memfd;
        let file = OpenFile::new(handle, format!("/memfd:{name}"), 0, true, true);
        self.fds.insert(file, flags & MFD_CLOEXEC != 0)
    }
    /// `fallocate`. Only mode zero, which is "make sure these bytes exist".
    ///
    /// Xwayland calls this through `posix_fallocate` after sealing its shared
    /// memory, to be certain the pages are there before it hands the
    /// descriptor to a compositor. The other modes -- punching holes,
    /// collapsing ranges -- are refused rather than silently treated as
    /// allocation, which would leave a caller believing it had freed space.
    pub fn fallocate(&self, fd: i32, mode: u64, offset: u64, len: u64) -> SysResult {
        if mode != 0 {
            return Err(Errno::EOPNOTSUPP);
        }
        let end = offset.checked_add(len).ok_or(Errno::EINVAL)?;
        if let Ok(memfd) = self.memfd_of(fd) {
            return memfd.reserve(end);
        }
        // An ordinary file grows by being written to its new length.
        let file = self.fds.get(fd)?;
        let size = file.handle.metadata()?.size;
        if end > size {
            file.handle.truncate(end)?;
        }
        Ok(0)
    }
    /// `epoll_create1`.
    pub fn epoll_create(&self, flags: u64) -> SysResult {
        const EPOLL_CLOEXEC: u64 = 0o2000000;
        let handle: Arc<dyn crate::vfs::FileHandle> = Arc::new(unix::EpollSet::new());
        let file = OpenFile::new(handle, "anon_inode:[eventpoll]".into(), 0, true, true);
        self.fds.insert(file, flags & EPOLL_CLOEXEC != 0)
    }
    /// Copy up to `count` bytes from one descriptor to another through a kernel
    /// buffer -- the shared core of `sendfile` and `copy_file_range`. When
    /// `in_off`/`out_off` is `Some` it is an absolute offset used without
    /// touching that descriptor's own position (the offset-pointer forms);
    /// `None` uses and advances the descriptor's position. Returns the bytes
    /// transferred. A single call moves at most one buffer, as Linux does --
    /// callers loop for more. If the output takes fewer bytes than were read
    /// from a positional-less input, the input is rewound so the tail is not
    /// silently dropped.
    pub fn transfer_fds(
        &self,
        in_fd: i32,
        in_off: Option<u64>,
        out_fd: i32,
        out_off: Option<u64>,
        count: u64,
    ) -> SysResult {
        let cap = count.min(1 << 24) as usize;
        if cap == 0 {
            return Ok(0);
        }
        let in_file = self.fds.get(in_fd)?;
        let out_file = self.fds.get(out_fd)?;
        let mut buf = vec![0u8; cap];
        let n = match in_off {
            Some(o) => in_file.pread(o, &mut buf)?,
            None => in_file.read(&mut buf)?,
        } as usize;
        if n == 0 {
            return Ok(0);
        }
        let mut written = 0usize;
        while written < n {
            let w = match out_off {
                Some(o) => out_file.pwrite(o + written as u64, &buf[written..n])?,
                None => out_file.write(&buf[written..n])?,
            } as usize;
            if w == 0 {
                break;
            }
            written += w;
        }
        // A positional-less input already advanced by the full `n`; hand back
        // the tail the output could not take (best effort: a stream cannot seek).
        if in_off.is_none() && written < n {
            let _ = in_file.seek(-((n - written) as i64), 1 /* SEEK_CUR */);
        }
        Ok(written as u64)
    }
    /// `sendfile(out_fd, in_fd, offset_ptr, count)`: copy in the kernel. With a
    /// non-null `offset_ptr` the read starts there and the pointer is advanced
    /// by the count sent, leaving the input's own position untouched -- so a
    /// server can serve the same file to many clients without seeking.
    pub fn sendfile(
        &self,
        m: &mut GuestMem<'_>,
        out_fd: i32,
        in_fd: i32,
        off_ptr: u64,
        count: u64,
    ) -> SysResult {
        let in_off = if off_ptr != 0 {
            Some(m.read_u64(off_ptr)?)
        } else {
            None
        };
        let sent = self.transfer_fds(in_fd, in_off, out_fd, None, count)?;
        if let Some(o) = in_off {
            m.write_u64(off_ptr, o + sent)?;
        }
        Ok(sent)
    }
    /// `copy_file_range(fd_in, off_in_ptr, fd_out, off_out_ptr, len, flags)`.
    /// Both offset pointers are optional; each non-null one is used for the
    /// transfer and then advanced by the bytes copied. `flags` is reserved and
    /// must be zero.
    pub fn copy_file_range(
        &self,
        m: &mut GuestMem<'_>,
        fd_in: i32,
        off_in: u64,
        fd_out: i32,
        off_out: u64,
        len: u64,
        flags: u32,
    ) -> SysResult {
        if flags != 0 {
            return Err(Errno::EINVAL);
        }
        let in_off = if off_in != 0 {
            Some(m.read_u64(off_in)?)
        } else {
            None
        };
        let out_off = if off_out != 0 {
            Some(m.read_u64(off_out)?)
        } else {
            None
        };
        let copied = self.transfer_fds(fd_in, in_off, fd_out, out_off, len)?;
        if let Some(o) = in_off {
            m.write_u64(off_in, o + copied)?;
        }
        if let Some(o) = out_off {
            m.write_u64(off_out, o + copied)?;
        }
        Ok(copied)
    }
    /// A transfer whose input is an empty blocking stream waits for it to be
    /// readable (`AcceptWait`) and is then made again.
    pub fn transfer_park(&self, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        let (fd, off) = match n {
            nr::SENDFILE => (a[1] as i32, a[2]),
            _ => (a[0] as i32, a[1]),
        };
        let file = self.fds.get(fd).ok()?;
        if off != 0 || !file.would_block() || file.handle.poll_readable() {
            return None;
        }
        Some(BlockOp::AcceptWait { socket: Arc::clone(&file.handle) })
    }

    /// sendfile, copy_file_range and splice, by number.
    pub fn transfer_syscall(&self, m: &mut GuestMem<'_>, n: u64, a: &[u64; 6]) -> SysResult {
        match n {
            nr::SENDFILE => self.sendfile(m, a[0] as i32, a[1] as i32, a[2], a[3]),
            // A blocking `recvfrom`/`recv` that waited for a datagram or
            // bytes (the classifier below).
            nr::RECVFROM => recvfrom(self, m, a[0] as i32, a[1], a[2], a[3], a[4], a[5]),
            nr::COPY_FILE_RANGE => self.copy_file_range(m, a[0] as i32, a[1], a[2] as i32, a[3], a[4], a[5] as u32),
            _ => self.splice(m, a[0] as i32, a[1], a[2] as i32, a[3], a[4]),
        }
    }
    /// `splice(fd_in, off_in_ptr, fd_out, off_out_ptr, len, flags)`: as
    /// copy_file_range, with pipes allowed on either side (a pipe's offset
    /// pointer must be null, which the transfer does not need to check: a
    /// pipe has no `pread`). The SPLICE_F_* flags are hints.
    pub fn splice(&self, m: &mut GuestMem<'_>, fd_in: i32, off_in: u64, fd_out: i32, off_out: u64, len: u64) -> SysResult {
        self.copy_file_range(m, fd_in, off_in, fd_out, off_out, len, 0)
    }

    /// `pipe2`.
    pub fn pipe2(&self, m: &mut GuestMem<'_>, fds: u64, flags: u64) -> SysResult {
        // A pipe in the global kernel when there is one, because the whole
        // point of a pipe is that another process holds the other end -- and
        // with a Worker per process that other end is in another memory,
        // where an `Arc<Pipe>` means nothing.
        if let Some(r) = self.pipes {
            return self.shared_pipe2(m, r, fds, flags);
        }
        let pipe = Arc::new(Pipe::new());
        let cloexec = flags & oflags::O_CLOEXEC != 0;
        let read_end = OpenFile::new(
            Arc::new(PipeEnd {
                pipe: Arc::clone(&pipe),
                write: false,
            }),
            "pipe:[0]".to_string(),
            // The access mode belongs in the flags, not only in the two
            // booleans: `fcntl(F_GETFL)` reports it, and glibc's `fdopen`
            // refuses a descriptor whose mode does not match the one asked
            // for. A write end reporting `O_RDONLY` is why apt could not
            // open its dpkg pipe.
            flags | oflags::O_RDONLY,
            true,
            false,
        );
        let write_end = OpenFile::new(
            Arc::new(PipeEnd {
                pipe: Arc::clone(&pipe),
                write: true,
            }),
            "pipe:[0]".to_string(),
            flags | oflags::O_WRONLY,
            false,
            true,
        );
        // Both ends start with one reference each, which `Pipe::new` already
        // accounted for.
        let rfd = self.fds.insert(read_end, cloexec)?;
        let wfd = self.fds.insert(write_end, cloexec)?;
        m.write_u32(fds, rfd as u32)?;
        m.write_u32(fds + 4, wfd as u32)?;
        Ok(0)
    }

    /// `pipe2` with the buffer in the global kernel.
    ///
    /// The two descriptions are the same shape as the local ones -- the same
    /// flags, the same access modes -- and differ only in what the handle is.
    /// The path carries the id and the end, because that is what a `fork` has
    /// to send across to rebuild the descriptor in the child: see
    /// `gex_fork_fds` in the browser shim.
    fn shared_pipe2(
        &self,
        m: &mut GuestMem<'_>,
        r: crate::pipe::Remote,
        fds: u64,
        flags: u64,
    ) -> SysResult {
        let id = (r.new)(self.pid);
        if id < 0 {
            return Err(Errno::ENFILE);
        }
        let cloexec = flags & oflags::O_CLOEXEC != 0;
        let read_end = OpenFile::new(
            Arc::new(crate::pipe::RemotePipeEnd {
                id,
                write: false,
                remote: r,
            }),
            format!("pipe:[{id}]"),
            flags | oflags::O_RDONLY,
            true,
            false,
        );
        let write_end = OpenFile::new(
            Arc::new(crate::pipe::RemotePipeEnd {
                id,
                write: true,
                remote: r,
            }),
            format!("pipe:[{id}]"),
            flags | oflags::O_WRONLY,
            false,
            true,
        );
        let rfd = self.fds.insert(read_end, cloexec)?;
        let wfd = self.fds.insert(write_end, cloexec)?;
        m.write_u32(fds, rfd as u32)?;
        m.write_u32(fds + 4, wfd as u32)?;
        Ok(0)
    }

    /// `ftruncate`.
    pub fn ftruncate(&self, m: &mut GuestMem<'_>, fd: i32, len: u64) -> SysResult {
        let f = self.fds.get(fd)?;
        self.truncate_file(m, &f, len)
    }

    /// `ftruncate` or `truncate` on `f`: the file `len` bytes long.
    pub fn truncate_file(&self, m: &mut GuestMem<'_>, f: &OpenFile, len: u64) -> SysResult {
        if (len as i64) < 0 {
            return Err(Errno::EINVAL);
        }
        f.handle.truncate(len)?;
        // A shared mapping's bytes past the new end are no longer the file's:
        // zero, as the file now reads there once it grows again.
        if any_shared(self.shared) {
            for (at, _, k) in shared_spans(self.shared, f, len, u64::MAX - len) {
                let _ = m.zero(at, k as usize);
            }
        }
        Ok(0)
    }

    /// `fadvise64(fd, offset, len, advice)`: [`crate::vfs::FileHandle::advise`]
    /// (a lazy file fetches what `WILLNEED` or gex's image advice asks for).
    pub fn fadvise(&self, fd: i32, offset: u64, len: u64, advice: u64) -> SysResult {
        self.fds.get(fd).and_then(|f| f.handle.advise(offset, len, advice as u32).map(|_| 0))
    }

    /// `epoll_ctl`. A `struct epoll_event` on aarch64 is a 32-bit event mask,
    /// four bytes of padding, and the caller's 64-bit token -- unlike x86-64,
    /// where the structure is packed and the token starts at offset four.
    pub fn epoll_ctl(&self, m: &GuestMem<'_>, epfd: i32, op: u64, fd: i32, event: u64) -> SysResult {
        let set = self.epoll_of(epfd)?;
        // `EPOLL_CTL_DEL` is allowed a null event pointer.
        let (events, data) = if event == 0 {
            (0, 0)
        } else if gex_abi::X86 || self.compat {
            (m.read_u32(event)?, m.read_u64(event + 4)?)
        } else {
            (m.read_u32(event)?, m.read_u64(event + 8)?)
        };
        // Watching something that is not open is an error, and a program that
        // gets silence instead waits for a readiness that cannot arrive.
        let file = if op != unix::epoll_op::DEL { Some(self.fds.get(fd)?) } else { None };
        let source = file.as_ref().map_or(0, |f| f.handle.wait_queue());
        let r = set.ctl_queue(op, fd, events, data, source)?;
        if let Some(f) = file {
            set.note_file(fd, &f.handle, f.writable);
        }
        Ok(r)
    }

    /// A `readv`'s vectors when there is more than one with room in it,
    /// for a parked read to fill in one go.
    fn readv_vectors(&self, m: &GuestMem<'_>, a: [u64; 6]) -> Option<Arc<[(u64, u64)]>> {
        let vecs = Kernel::read_iovecs(m, a[1], a[2], self.compat).ok()?;
        (vecs.iter().filter(|v| v.1 > 0).count() > 1).then(|| Arc::from(vecs))
    }

    /// A `writev`'s bytes when they are few enough to be written whole
    /// ([`gather_small`]).
    fn small_writev(&self, m: &GuestMem<'_>, a: [u64; 6]) -> Option<Arc<[u8]>> {
        let vecs = Kernel::read_iovecs(m, a[1], a[2], self.compat).ok()?;
        gather_small(m, &vecs).map(Arc::from)
    }

    /// A vectored call's first non-empty vector (`a[1]`, `a[2]`), where a
    /// blocked `readv`/`writev` parks.
    fn first_iovec(&self, m: &GuestMem<'_>, a: [u64; 6]) -> Option<(u64, u64)> {
        Kernel::read_iovecs(m, a[1], a[2], self.compat).ok()?.into_iter().find(|v| v.1 > 0)
    }

    /// The park a stream `read`, `readv`, `write` or `writev` makes before
    /// it runs: nothing to read, or no room for the write.
    fn blocking_op(&self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        match n {
            nr::WRITE => {
                let file = self.fds.get(a[0] as i32).ok()?;
                if !file.write_is_atomic() {
                    return None;
                }
                // A handle that can report its room takes the direct path
                // when the write fits (the common case). A pipe cannot --
                // another Worker can change the answer between the claim and
                // the write, so `write_room` returns `None` -- so when the
                // room is unknown, park only if the descriptor is not writable
                // at all; writable takes the short count, which callers retry.
                // Getting this wrong sent a full-pipe write straight through
                // to `EAGAIN`, truncating a `gzip -dc | sha256sum` stream.
                match file.handle.write_room() {
                    Some(room) if a[2] <= room => return None,
                    Some(_) => {}
                    None if !file.write_would_block() => return None,
                    None => {}
                }
                Some(BlockOp::WriteStream {
                    handle: Arc::clone(&file.handle),
                    buf: a[1],
                    len: a[2].min(1 << 24) as usize,
                    gathered: None,
                })
            }
            // `writev` and `readv` on a stream block as `write` and `read` do,
            // on their first non-empty vector: the parked call is finished
            // with that vector alone, a short count the caller already loops
            // on. Without this a full pipe answered `writev` with `EAGAIN` on
            // a blocking descriptor: coreutils' `head` writes with it and died
            // "Resource temporarily unavailable" once its reader was slower
            // than it, and Wine's `send_request` writes the server's requests
            // with it (a thread that found the request pipe full died with
            // "wine client error: write: Resource temporarily unavailable").
            nr::WRITEV => {
                let file = self.fds.get(a[0] as i32).ok()?;
                if !file.write_is_atomic() {
                    return None;
                }
                // Or, at most PIPE_BUF bytes in all, on the whole of it,
                // gathered: such a write is never split.
                let (buf, len) = self.first_iovec(m, a)?;
                let gathered = self.small_writev(m, a);
                let len = gathered.as_ref().map_or(len, |g| g.len() as u64);
                match file.handle.write_room() {
                    Some(room) if len <= room => return None,
                    Some(_) => {}
                    None if !file.write_would_block() => return None,
                    None => {}
                }
                Some(BlockOp::WriteStream {
                    handle: Arc::clone(&file.handle),
                    buf,
                    len: len.min(1 << 24) as usize,
                    gathered,
                })
            }
            nr::READV => {
                let file = self.fds.get(a[0] as i32).ok()?;
                // (a terminal's polling read answers 0 instead: `polling_read`)
                if !file.would_block() || file.handle.empty_read_returns() {
                    return None;
                }
                let (buf, len) = self.first_iovec(m, a)?;
                let vectors = self.readv_vectors(m, a);
                let len = vectors.as_ref().map_or(len, |v| v.iter().map(|v| v.1).sum::<u64>());
                Some(BlockOp::ReadStream {
                    handle: Arc::clone(&file.handle),
                    buf,
                    len: len.min(1 << 24) as usize,
                    vectors,
                })
            }
            nr::READ => {
                let file = self.fds.get(a[0] as i32).ok()?;
                if !file.would_block() || file.handle.empty_read_returns() {
                    return None;
                }
                Some(BlockOp::ReadStream {
                    handle: Arc::clone(&file.handle),
                    buf: a[1],
                    len: a[2].min(1 << 24) as usize,
                    vectors: None,
                })
            }
            _ => None,
        }
    }

    /// Whether `fd` is a blocking descriptor of a terminal whose empty reads
    /// return 0 ([`crate::vfs::FileHandle::empty_read_returns`]).
    fn polling_read(&self, fd: u64) -> bool {
        self.fds.get(fd as i32).is_ok_and(|f| {
            f.flags() & crate::fs::oflags::O_NONBLOCK == 0 && f.handle.empty_read_returns()
        })
    }

    /// The park for a `read`/`readv` or `write`/`writev` that returned
    /// `EAGAIN` on a stream descriptor that should block.
    fn eagain_park(&self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        if n == nr::FCNTL || n == nr::FLOCK {
            return self.lock_park(m, n, a);
        }
        let file = self.fds.get(a[0] as i32).ok()?;
        let (buf, len) = match n {
            nr::READ | nr::WRITE => (a[1], a[2]),
            nr::READV | nr::WRITEV => self.first_iovec(m, a)?,
            _ => return None,
        };
        let handle = Arc::clone(&file.handle);
        let len = len.min(1 << 24) as usize;
        match n {
            nr::READ | nr::READV
                if file.readable
                    && file.flags() & crate::fs::oflags::O_NONBLOCK == 0
                    && file.handle.is_stream() =>
            {
                let vectors = if n == nr::READV { self.readv_vectors(m, a) } else { None };
                let len = vectors.as_ref().map_or(len, |v| v.iter().map(|v| v.1).sum::<u64>().min(1 << 24) as usize);
                Some(BlockOp::ReadStream { handle, buf, len, vectors })
            }
            nr::WRITEV if file.write_is_atomic() => {
                let gathered = self.small_writev(m, a);
                let len = gathered.as_ref().map_or(len, |g| g.len());
                Some(BlockOp::WriteStream { handle, buf, len, gathered })
            }
            nr::WRITE if file.write_is_atomic() => Some(BlockOp::WriteStream { handle, buf, len, gathered: None }),
            _ => None,
        }
    }

    /// A `struct cmsghdr` is a length (a `size_t`), a level, a type, then
    /// the payload, with the whole header aligned to a pointer: 16 bytes,
    /// or 12 for an i386 process. `msghdr`, `iovec` and `mmsghdr` are
    /// pointer-sized fields too ([`Self::msg_word`]).
    pub(crate) fn cmsg_hdr(&self) -> u64 {
        if self.compat {
            12
        } else {
            16
        }
    }

    pub(crate) fn cmsg_align(&self, n: u64) -> u64 {
        let a = self.msg_word();
        n.div_ceil(a) * a
    }

    /// The pointer and `size_t` width of the message structures.
    pub(crate) fn msg_word(&self) -> u64 {
        if self.compat {
            4
        } else {
            8
        }
    }

    /// The `i`th pointer-sized word of a message structure at `at`.
    pub(crate) fn msg_rd(&self, m: &GuestMem<'_>, at: u64, i: u64) -> Result<u64, Errno> {
        if self.compat {
            Ok(m.read_u32(at + 4 * i)? as u64)
        } else {
            m.read_u64(at + 8 * i)
        }
    }

    pub(crate) fn msg_wr(&self, m: &mut GuestMem<'_>, at: u64, i: u64, v: u64) -> Result<(), Errno> {
        if self.compat {
            m.write_u32(at + 4 * i, v as u32)
        } else {
            m.write_u64(at + 8 * i, v)
        }
    }

    /// Collect the descriptors a `sendmsg` is passing.
    ///
    /// `SCM_RIGHTS` is the whole reason unix sockets are here. A Wayland
    /// client draws into shared memory and sends the *descriptor*; without
    /// this the compositor is handed a number that means nothing to it.
    fn read_scm_rights(
        &self,
        m: &GuestMem<'_>,
        control: u64,
        len: u64,
    ) -> Result<Vec<Arc<OpenFile>>, Errno> {
        const SOL_SOCKET: u32 = 1;
        const SCM_RIGHTS: u32 = 1;
        let (hdr, w) = (self.cmsg_hdr(), self.msg_word());
        let mut out = Vec::new();
        let mut at = 0u64;
        while at + hdr <= len {
            let cmsg_len = self.msg_rd(m, control + at, 0)?;
            if cmsg_len < hdr || at + cmsg_len > len {
                break;
            }
            let level = m.read_u32(control + at + w)?;
            let ty = m.read_u32(control + at + w + 4)?;
            if level == SOL_SOCKET && ty == SCM_RIGHTS {
                let n = (cmsg_len - hdr) / 4;
                for i in 0..n.min(253) {
                    let fd = m.read_u32(control + at + hdr + i * 4)? as i32;
                    out.push(self.fds.get(fd)?);
                }
            }
            at += self.cmsg_align(cmsg_len);
        }
        Ok(out)
    }

    /// Install received descriptors and describe them in the control buffer.
    ///
    /// Returns the bytes of control data written. A buffer too small for all
    /// of them gets what fits; Linux sets `MSG_CTRUNC` and closes the rest,
    /// and the socket layer here simply never handed over more than fit.
    fn write_scm_rights(
        &self,
        m: &mut GuestMem<'_>,
        control: u64,
        len: u64,
        files: Vec<Arc<OpenFile>>,
    ) -> Result<u64, Errno> {
        const SOL_SOCKET: u32 = 1;
        const SCM_RIGHTS: u32 = 1;
        if files.is_empty() || control == 0 {
            return Ok(0);
        }
        let (hdr, w) = (self.cmsg_hdr(), self.msg_word());
        let payload = files.len() as u64 * 4;
        let cmsg_len = hdr + payload;
        if cmsg_len > len {
            return Ok(0);
        }
        self.msg_wr(m, control, 0, cmsg_len)?;
        m.write_u32(control + w, SOL_SOCKET)?;
        m.write_u32(control + w + 4, SCM_RIGHTS)?;
        for (i, f) in files.into_iter().enumerate() {
            let fd = self.fds.insert(f, false)?;
            m.write_u32(control + hdr + i as u64 * 4, fd as u32)?;
        }
        Ok(cmsg_len)
    }

    /// The `SCM_CREDENTIALS` message a socket with `SO_PASSCRED` gets with
    /// what it reads: the sender's `struct ucred`, here the far end's as
    /// the two were joined (`SO_PEERCRED`'s: a process that changes its ids
    /// after connecting is still reported as it was). A peer not known is
    /// Linux's overflow ids, as for a sender in no namespace of the
    /// reader's. The bytes written: none when there is no room.
    fn write_scm_creds(&self, m: &mut GuestMem<'_>, control: u64, len: u64, peer: Option<unix::Cred>) -> Result<u64, Errno> {
        const SOL_SOCKET: u32 = 1;
        const SCM_CREDENTIALS: u32 = 2;
        let (hdr, w) = (self.cmsg_hdr(), self.msg_word());
        let cmsg_len = hdr + 12;
        let used = self.cmsg_align(cmsg_len);
        if control == 0 || used > len {
            return Ok(0);
        }
        let c = peer.unwrap_or(unix::Cred { pid: 0, uid: 65534, gid: 65534 });
        self.msg_wr(m, control, 0, cmsg_len)?;
        m.write_u32(control + w, SOL_SOCKET)?;
        m.write_u32(control + w + 4, SCM_CREDENTIALS)?;
        m.write_u32(control + hdr, c.pid as u32)?;
        m.write_u32(control + hdr + 4, c.uid)?;
        m.write_u32(control + hdr + 8, c.gid)?;
        Ok(used)
    }

    /// `sendmsg`, which is how glibc writes to a socket, and the only way to
    /// pass a descriptor to another process.
    pub fn sendmsg(&self, m: &GuestMem<'_>, fd: i32, msg: u64) -> SysResult {
        // `msg_name` is where this one goes, and a reply has to be reported
        // as having come from there or a resolver drops it as spoofed.
        let name = self.msg_rd(m, msg, 0)?;
        if let Ok(d) = self.dgram_of(fd) {
            // A datagram, to `msg_name` when there is one (resolved as the
            // socket's own names are: an abstract name, or a path from `/`;
            // a relative one is taken from `/`), else to the connected peer.
            let namelen = m.read_u32(msg + if self.compat { 4 } else { 8 })? as u64;
            let to = if name != 0 {
                let p = Kernel::sockaddr_un(m, name, namelen)?;
                Some(if p.starts_with('\0') { p } else { crate::vfs::normalize(&p, "/") })
            } else {
                None
            };
            let iov = self.msg_rd(m, msg, 2)?;
            let iovlen = self.msg_rd(m, msg, 3)?;
            let mut data = Vec::new();
            for i in 0..iovlen.min(1024) {
                let base = self.msg_rd(m, iov, 2 * i)?;
                let len = self.msg_rd(m, iov, 2 * i + 1)? as usize;
                if len > 0 {
                    data.extend_from_slice(m.read(base, len.min(1 << 22))?);
                }
            }
            return d.send_to(&data, to.as_deref());
        }
        if name != 0 {
            if let Ok(sock) = self.socket_of(fd) {
                let bytes = m.read(name, 16)?.to_vec();
                if let Ok((ip, port)) = net::parse_sockaddr(&bytes) {
                    sock.set_peer(ip, port);
                }
            }
        }
        let iov = self.msg_rd(m, msg, 2)?;
        let iovlen = self.msg_rd(m, msg, 3)?;
        let mut data = Vec::new();
        for i in 0..iovlen.min(1024) {
            let base = self.msg_rd(m, iov, 2 * i)?;
            let len = self.msg_rd(m, iov, 2 * i + 1)? as usize;
            if len > 0 {
                data.extend_from_slice(m.read(base, len.min(1 << 22))?);
            }
        }
        // Ancillary data, which for a unix socket may be descriptors. Staged
        // on the socket rather than passed down, so the send attaches them to
        // the first byte it writes and a send that fails attaches nothing.
        let control = self.msg_rd(m, msg, 4)?;
        let controllen = self.msg_rd(m, msg, 5)?;
        if control != 0 && controllen >= self.cmsg_hdr() {
            if let Ok(sock) = self.unix_of(fd) {
                let files = self.read_scm_rights(m, control, controllen)?;
                if !self.fds.get(fd)?.writable {
                    return Err(Errno::EBADF);
                }
                // The bytes and their descriptors in one step (`send_with`).
                return sock.send_with(&data, files);
            } else if let Ok(sock) = self.remote_unix_of(fd) {
                // The far end is in another memory, where an `OpenFile` means
                // nothing: each descriptor goes as a description the host
                // knows how to rebuild there. One that cannot leave this
                // process fails the send, as Linux fails one it cannot pass,
                // rather than arriving as a hole the receiver finds later.
                let files = self.read_scm_rights(m, control, controllen)?;
                if !files.is_empty() {
                    let mut descs = String::new();
                    for f in &files {
                        let d = (sock.remote.describe)(f).ok_or(Errno::EBADF)?;
                        descs.push_str(&d);
                        descs.push('\n');
                    }
                    return sock.send_described(&data, &descs);
                }
            }
        }
        self.fds.get(fd)?.write(&data)
    }

    /// `recvmsg` on a unix datagram socket: one datagram scattered over the
    /// vector, its sender in `msg_name`, `MSG_TRUNC` in `msg_flags` when it
    /// did not fit, and no ancillary data.
    fn recvmsg_dgram(&self, m: &mut GuestMem<'_>, d: &Arc<unix::UnixDgram>, msg: u64, iov: u64, iovlen: u64, peek: bool) -> SysResult {
        let mut parts = Vec::new();
        let mut want = 0usize;
        for i in 0..iovlen.min(1024) {
            let base = self.msg_rd(m, iov, 2 * i)?;
            let len = self.msg_rd(m, iov, 2 * i + 1)? as usize;
            let len = len.min((1 << 22) - want);
            if len > 0 {
                parts.push((base, len));
                want += len;
            }
        }
        let mut tmp = vec![0u8; want];
        let (n, whole, from) = d.recv_from(&mut tmp, peek)?;
        let mut at = 0usize;
        for (base, len) in parts {
            if at >= n {
                break;
            }
            let take = len.min(n - at);
            m.write(base, &tmp[at..at + take])?;
            at += take;
        }
        let name = self.msg_rd(m, msg, 0)?;
        let namelen_at = msg + if self.compat { 4 } else { 8 };
        if name != 0 {
            match from {
                Some(p) => Kernel::write_sockaddr_un(m, name, namelen_at, &p)?,
                None => {
                    Kernel::write_sockaddr_unnamed(m, name, namelen_at)?;
                }
            }
        }
        self.msg_wr(m, msg, 5, 0)?;
        const MSG_TRUNC: u32 = 0x20;
        m.write_u32(msg + if self.compat { 24 } else { 48 }, if whole > n { MSG_TRUNC } else { 0 })?;
        Ok(n as u64)
    }

    /// `recvmsg` on a socket in the global kernel: one read for the whole
    /// vector (so descriptors attach to it as they would to the first byte),
    /// scattered after, and the descriptions that came with it rebuilt here.
    #[allow(clippy::too_many_arguments)]
    fn recvmsg_remote(
        &self,
        m: &mut GuestMem<'_>,
        sock: &Arc<unix::RemoteUnix>,
        msg: u64,
        iov: u64,
        iovlen: u64,
        control: u64,
        controllen: u64,
        room: usize,
        peek: bool,
    ) -> SysResult {
        let mut parts = Vec::new();
        let mut want = 0usize;
        for i in 0..iovlen.min(1024) {
            let base = self.msg_rd(m, iov, 2 * i)?;
            let len = self.msg_rd(m, iov, 2 * i + 1)? as usize;
            let len = len.min((1 << 22) - want);
            if len > 0 {
                parts.push((base, len));
                want += len;
            }
        }
        let mut tmp = vec![0u8; want];
        let (n, descs) = sock.recv_described(&mut tmp, room, peek, self.pid)?;
        let mut at = 0usize;
        for (base, len) in parts {
            if at >= n as usize {
                break;
            }
            let take = len.min(n as usize - at);
            m.write(base, &tmp[at..at + take])?;
            at += take;
        }
        let mut files = Vec::new();
        for d in &descs {
            // One the host cannot rebuild is said there, where it knows why.
            if let Some(f) = (sock.remote.rebuild)(self.ns, d) {
                files.push(f);
            }
        }
        let name = self.msg_rd(m, msg, 0)?;
        if name != 0 {
            m.write_u32(msg + if self.compat { 4 } else { 8 }, 0)?;
        }
        let creds = if sock.passcred() { self.write_scm_creds(m, control, controllen, sock.peer_cred())? } else { 0 };
        let written = creds + self.write_scm_rights(m, control + creds, controllen - creds, files)?;
        self.msg_wr(m, msg, 5, written)?;
        m.write_u32(msg + if self.compat { 24 } else { 48 }, 0)?;
        Ok(n)
    }

    /// The park of a `recvmsg` on a blocking socket: until it is readable,
    /// when the call is made again (the caller's to arrange). A non-blocking
    /// socket, or a call that asked for `MSG_DONTWAIT`, does not wait.
    /// libwayland reads its blocking socket with `MSG_DONTWAIT` and expects
    /// `EAGAIN` once it has drained it; waiting here instead hangs the client
    /// in its first round-trip.
    pub fn recvmsg_park(&self, a: [u64; 6]) -> Option<BlockOp> {
        let file = self.fds.get(a[0] as i32).ok()?;
        if a[2] & Kernel::MSG_DONTWAIT != 0 || !file.would_block() {
            return None;
        }
        Some(BlockOp::AcceptWait { socket: Arc::clone(&file.handle) })
    }

    pub fn recvmsg(&self, m: &mut GuestMem<'_>, fd: i32, msg: u64, flags: u64) -> SysResult {
        let peek = flags & Kernel::MSG_PEEK != 0;
        let iov = self.msg_rd(m, msg, 2)?;
        let iovlen = self.msg_rd(m, msg, 3)?;
        let control = self.msg_rd(m, msg, 4)?;
        let controllen = self.msg_rd(m, msg, 5)?;
        // How many descriptors the caller left room for: the rest of a
        // write's are closed (`MSG_CTRUNC`).
        let room = if control != 0 && controllen >= self.cmsg_hdr() {
            ((controllen - self.cmsg_hdr()) / 4) as usize
        } else {
            0
        };
        if let Ok(d) = self.dgram_of(fd) {
            return self.recvmsg_dgram(m, &d, msg, iov, iovlen, peek);
        }
        let unix_sock = self.unix_of(fd).ok();
        if unix_sock.is_none() {
            if let Ok(sock) = self.remote_unix_of(fd) {
                return self.recvmsg_remote(m, &sock, msg, iov, iovlen, control, controllen, room, peek);
            }
        }
        // The read and the sender address it set, as one (`recv_turn`).
        // Only when it asks who sent it.
        let inet = if unix_sock.is_none() && self.msg_rd(m, msg, 0)? != 0 { self.socket_of(fd).ok() } else { None };
        let _turn = inet.as_ref().map(|s| s.recv_turn());
        // One read for the whole vector, then scattered over it, as the
        // remote path does. Reading each iovec in turn lost data: a first
        // iovec filled exactly, with nothing left behind it, made the second
        // read fail EAGAIN and the call return that error -- the bytes of the
        // first were taken from the socket and never reported. libwayland
        // reads its ring buffer as two iovecs whenever it wraps, so a Wayland
        // client would lose whole events now and then (a window stepping
        // back a pixel during a caption drag). A peek
        // per iovec also peeked the same bytes into each.
        let mut parts = Vec::new();
        let mut want = 0usize;
        for i in 0..iovlen.min(1024) {
            let base = self.msg_rd(m, iov, 2 * i)?;
            let len = self.msg_rd(m, iov, 2 * i + 1)? as usize;
            let len = len.min((1 << 22) - want);
            if len > 0 {
                parts.push((base, len));
                want += len;
            }
        }
        let mut received = Vec::new();
        let mut cut = false;
        let total = if want == 0 {
            0
        } else {
            let mut tmp = vec![0u8; want];
            let n = match &unix_sock {
                // The bytes and their descriptors in one step (`recv_with`).
                Some(sock) => {
                    let (n, fds, c) = sock.recv_msg(&mut tmp, room, peek)?;
                    received = fds;
                    cut = c;
                    n
                }
                None if peek => match self.socket_of(fd) {
                    Ok(sock) => sock.peek(&mut tmp)?,
                    Err(_) => self.fds.get(fd)?.read(&mut tmp)?,
                },
                None => self.fds.get(fd)?.read(&mut tmp)?,
            };
            let mut at = 0usize;
            for (base, len) in parts {
                if at >= n as usize {
                    break;
                }
                let take = len.min(n as usize - at);
                m.write(base, &tmp[at..at + take])?;
                at += take;
            }
            n
        };
        // Who sent it, as recvfrom says: musl's resolver drops a reply whose
        // msg_name is not one of its nameservers. A unix peer has no name.
        let name = self.msg_rd(m, msg, 0)?;
        let namelen_at = msg + if self.compat { 4 } else { 8 };
        if name != 0 {
            if unix_sock.is_some() {
                m.write_u32(namelen_at, 0)?;
            } else {
                let peer = self.socket_of(fd).ok().and_then(|s| s.last_from().or_else(|| s.peer()));
                let (ip, port) = peer.unwrap_or((0, 0));
                self.write_sockaddr(m, ip, port, name, namelen_at)?;
            }
        }
        let given = !received.is_empty();
        let creds = match &unix_sock {
            Some(s) if s.passcred() => self.write_scm_creds(m, control, controllen, s.peer_cred())?,
            _ => 0,
        };
        let written = creds + self.write_scm_rights(m, control + creds, controllen - creds, received)?;
        self.msg_wr(m, msg, 5, written)?;
        // msg_flags (int): MSG_CTRUNC when descriptors were closed for want
        // of room.
        const MSG_CTRUNC: u32 = 0x8;
        let flags = if cut || (given && written == 0) { MSG_CTRUNC } else { 0 };
        m.write_u32(msg + if self.compat { 24 } else { 48 }, flags)?;
        Ok(total)
    }

    fn write_sockaddr(
        &self,
        m: &mut GuestMem<'_>,
        ip: u32,
        port: u16,
        addr: u64,
        addrlen: u64,
    ) -> SysResult {
        if addr == 0 {
            return Ok(0);
        }
        let sa = net::build_sockaddr(ip, port);
        let room = if addrlen != 0 {
            m.read_u32(addrlen)? as usize
        } else {
            sa.len()
        };
        let n = room.min(sa.len());
        m.write(addr, &sa[..n])?;
        if addrlen != 0 {
            m.write_u32(addrlen, sa.len() as u32)?;
        }
        Ok(0)
    }

    /// The socket in the global kernel behind a descriptor.
    pub(crate) fn remote_unix_of(&self, fd: i32) -> Result<Arc<unix::RemoteUnix>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_remote_unix().ok_or(Errno::ENOTSOCK)
    }

    /// The internet socket behind a descriptor.
    pub(crate) fn socket_of(&self, fd: i32) -> Result<Arc<net::Socket>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_inet_socket().ok_or(Errno::ENOTSOCK)
    }

    /// The local unix socket behind a descriptor.
    pub(crate) fn unix_of(&self, fd: i32) -> Result<Arc<unix::UnixSocket>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_unix_socket().ok_or(Errno::ENOTSOCK)
    }

    /// The unix datagram socket behind a descriptor.
    pub(crate) fn dgram_of(&self, fd: i32) -> Result<Arc<unix::UnixDgram>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_unix_dgram().ok_or(Errno::ENOTSOCK)
    }

    /// The epoll set behind a descriptor: `EINVAL` for any other.
    pub(crate) fn epoll_of(&self, fd: i32) -> Result<Arc<unix::EpollSet>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_epoll().ok_or(Errno::EINVAL)
    }

    /// Read the guest's `pollfd` array into something the host can re-check.
    fn collect_pollfds(&self, m: &GuestMem<'_>, fds: u64, nfds: u64) -> Option<Vec<PollFd>> {
        let mut out = Vec::with_capacity(nfds.min(1024) as usize);
        for i in 0..nfds.min(1024) {
            let addr = fds + i * 8;
            let fd = m.read_u32(addr).ok()? as i32;
            let events = (m.read_u32(addr + 4).ok()? & 0xffff) as u16;
            let file = self.fds.get(fd).ok();
            out.push(PollFd {
                addr,
                fd,
                events,
                handle: file.as_ref().map(|f| Arc::clone(&f.handle)),
                // The open mode only. Whether a write would *block* is
                // asked of the handle in `poll_revents`, every time, because
                // a parked poll outlives the state it was built from.
                writable: file.as_ref().is_some_and(|f| f.writable),
                epoll: None,
                edge: false,
            });
        }
        Some(out)
    }

    /// Read `select`'s three bitmaps into the same list `poll` uses.
    ///
    /// The two calls ask the same question. Only the reply differs, so only
    /// the reply is written twice.
    fn collect_selectfds(
        &self,
        m: &GuestMem<'_>,
        nfds: u32,
        readfds: u64,
        writefds: u64,
        exceptfds: u64,
    ) -> Result<Vec<PollFd>, Errno> {
        const POLLIN: u16 = 0x001;
        const POLLOUT: u16 = 0x004;
        if (nfds as i32) < 0 {
            return Err(Errno::EINVAL);
        }
        let nfds = nfds.min(1024);
        let mut want: std::collections::BTreeMap<i32, u16> = Default::default();
        // (exceptfds: watched for nothing, but its descriptors must be open)
        for (set, ev) in [(readfds, POLLIN), (writefds, POLLOUT), (exceptfds, 0)] {
            if set == 0 {
                continue;
            }
            for w in 0..(nfds as usize).div_ceil(64) {
                let word = m.read_bits64(set + (w as u64) * 8).map_err(|_| Errno::EFAULT)?;
                if word == 0 {
                    continue;
                }
                for bit in 0..64u32 {
                    let fd = (w as u32) * 64 + bit;
                    if fd >= nfds {
                        break;
                    }
                    if word & (1 << bit) != 0 {
                        *want.entry(fd as i32).or_default() |= ev;
                    }
                }
            }
        }
        // An exceptional condition never happens here, so a descriptor named
        // only in that set is watched for nothing and simply never fires. A
        // descriptor in any set that is not open is EBADF, as on Linux.
        if want.keys().any(|&fd| self.fds.get(fd).is_err()) {
            return Err(Errno::EBADF);
        }
        Ok(
            want.into_iter()
                .map(|(fd, events)| {
                    let file = self.fds.get(fd).ok();
                    PollFd {
                        addr: 0,
                        fd,
                        events,
                        handle: file.as_ref().map(|f| Arc::clone(&f.handle)),
                        // The open mode only. Whether a write would *block* is
                // asked of the handle in `poll_revents`, every time, because
                // a parked poll outlives the state it was built from.
                writable: file.as_ref().is_some_and(|f| f.writable),
                        epoll: None,
                        edge: false,
                    }
                })
                .collect(),
        )
    }

    /// A set's registrations as the list the poll path understands.
    fn epoll_pollfds(&self, set: &Arc<unix::EpollSet>) -> Vec<PollFd> {
        set.watches()
            .iter()
            .filter_map(|w| {
                let file = self.fds.get(w.fd).ok()?;
                set.note_file(w.fd, &file.handle, file.writable);
                Some(PollFd {
                    addr: 0,
                    fd: w.fd,
                    events: (w.events & 0xffff) as u16,
                    handle: Some(Arc::clone(&file.handle)),
                    writable: file.writable,
                    epoll: Some(Arc::clone(set)),
                    edge: w.events & unix::epoll_flag::ET != 0,
                })
            })
            .collect()
    }

    /// Every epoll set among `fds` (and the sets they watch, `depth` deep)
    /// learns its watched descriptors' open files from this process's
    /// table ([`unix::EpollSet::note_file`]) before it is asked whether it
    /// is readable: a set polled before any wait on it (a fork's copy, one
    /// only ever polled) knows them by its first poll, and a watch whose
    /// descriptor is not open here reports nothing, as Linux drops it.
    fn note_epoll_sets(&self, fds: &[PollFd], depth: u32) {
        if depth == 0 {
            return;
        }
        for p in fds {
            if let Some(set) = p.handle.as_ref().and_then(|h| Arc::clone(h).as_epoll()) {
                let inner = self.epoll_pollfds(&set);
                self.note_epoll_sets(&inner, depth - 1);
            }
        }
    }

    /// The poll family's question (`ppoll`, `pselect6`, `epoll_pwait`,
    /// `epoll_pwait2`: canonical `n`, arguments `a`): the descriptors, how
    /// the answer is written, and how long the call may wait (`None`: not at
    /// all; `Some(None)`: for ever).
    #[allow(clippy::type_complexity)]
    fn poll_question(
        &self,
        m: &GuestMem<'_>,
        n: u64,
        a: [u64; 6],
    ) -> Result<(Vec<PollFd>, PollReport, Option<Option<std::time::Duration>>), Errno> {
        // A `timespec` timeout: null waits for ever, zero not at all.
        let timespec = |p: u64| -> Result<Option<Option<std::time::Duration>>, Errno> {
            if p == 0 {
                return Ok(Some(None));
            }
            let (secs, nanos) = (m.read_u64(p)?, m.read_u64(p + 8)?);
            Ok((secs != 0 || nanos != 0)
                .then(|| Some(std::time::Duration::new(secs, (nanos % 1_000_000_000) as u32))))
        };
        match n {
            nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2 => {
                let set = self.epoll_of(a[0] as i32)?;
                let generation = set.generation();
                let fds = self.epoll_pollfds(&set);
                self.note_epoll_sets(&fds, 4);
                // `epoll_pwait` takes milliseconds, negative for ever;
                // `epoll_pwait2` a `timespec`.
                let wait = if n == nr::EPOLL_PWAIT2 {
                    timespec(a[3])?
                } else {
                    match a[3] as i32 {
                        0 => None,
                        ms if ms < 0 => Some(None),
                        ms => Some(Some(std::time::Duration::from_millis(ms as u64))),
                    }
                };
                let report = PollReport::Epoll {
                    out: a[1],
                    max: a[2] as u32,
                    data: set.watches().iter().map(|w| (w.fd, w.data)).collect(),
                    packed: gex_abi::X86 || self.compat,
                    set,
                    generation,
                };
                Ok((fds, report, wait))
            }
            nr::PPOLL => {
                let fds = self.collect_pollfds(m, a[0], a[1]).ok_or(Errno::EFAULT)?;
                self.note_epoll_sets(&fds, 4);
                Ok((fds, PollReport::Poll, timespec(a[2])?))
            }
            nr::PSELECT6 => {
                let fds = self.collect_selectfds(m, a[0] as u32, a[1], a[2], a[3])?;
                self.note_epoll_sets(&fds, 4);
                Ok((
                    fds,
                    PollReport::Select { nfds: a[0] as u32, readfds: a[1], writefds: a[2], exceptfds: a[3] },
                    timespec(a[4])?,
                ))
            }
            _ => Err(Errno::ENOSYS),
        }
    }

    /// The park of a poll-family call that may wait and has nothing ready.
    fn poll_park(&self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        let (fds, report, wait) = self.poll_question(m, n, a).ok()?;
        let deadline = wait?;
        (!poll_ready(&fds)).then_some(BlockOp::Poll { fds, deadline, report })
    }

    /// A poll-family call answered now, with whatever is ready.
    fn poll_now(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
        let (fds, report, _) = self.poll_question(m, n, a)?;
        poll_finish(&fds, &report, m)
    }

    /// A poll-family call: its answer when something is ready or it may not
    /// wait, otherwise its park; one look at the descriptors either way.
    fn poll(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Result<SysResult, BlockOp> {
        let (fds, report, wait) = match self.poll_question(m, n, a) {
            Ok(q) => q,
            Err(e) => return Ok(Err(e)),
        };
        match wait {
            Some(deadline) if !poll_ready(&fds) => Err(BlockOp::Poll { fds, deadline, report }),
            _ => Ok(poll_finish(&fds, &report, m)),
        }
    }

    /// The call itself, for the calls of this half; `None` for any other.
    fn call(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<SysResult> {
        let fds = self.fds;
        Some(match n {
            nr::READ => io_read(fds, m, a[0] as i32, a[1], a[2], self.shared),
            nr::WRITE => io_write(fds, m, a[0] as i32, a[1], a[2], self.shared),
            nr::READV => io_readv(fds, m, a[0] as i32, a[1], a[2], self.compat, self.shared),
            nr::WRITEV => io_writev(fds, m, a[0] as i32, a[1], a[2], self.compat, self.shared),
            nr::PREAD64 => io_pread(fds, m, a[0] as i32, a[1], a[2], a[3], self.shared),
            nr::PWRITE64 => io_pwrite(fds, m, a[0] as i32, a[1], a[2], a[3], self.shared),
            nr::LSEEK => fds.get(a[0] as i32).and_then(|f| f.seek(a[1] as i64, a[2] as u32)),
            nr::CLOSE => {
                self.closing_writes(m, a[0], a[0]);
                self.close(a[0] as i32)
            }
            nr::CLOSE_RANGE => {
                self.closing_writes(m, a[0], a[1]);
                self.close_range(a[0], a[1], a[2])
            }
            nr::DUP => self.dup(a[0] as i32),
            nr::DUP2 => {
                if a[0] != a[1] {
                    self.closing_writes(m, a[1], a[1]);
                }
                self.dup2(a[0] as i32, a[1] as i32)
            }
            nr::DUP3 => {
                self.closing_writes(m, a[1], a[1]);
                self.dup3(a[0] as i32, a[1] as i32, a[2])
            }
            nr::FCNTL => self.fcntl_call(m, a[0] as i32, a[1], a[2]),
            nr::FLOCK => self.flock(m, a[0] as i32, a[1]),
            _ => return None,
        })
    }
}

/// The signal mask `epoll_pwait`, `epoll_pwait2`, `ppoll` or `pselect6`
/// (`n`, canonical, its arguments `a`) asks to wait with: `Ok(None)` when
/// the call has none (a null pointer, or another call), `EINVAL` for a
/// set size that is not 8 bytes, `EFAULT` for a set that cannot be read.
fn call_sigmask(m: &GuestMem<'_>, n: u64, a: &[u64; 6]) -> Result<Option<u64>, Errno> {
    let (set, size) = match n {
        nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2 => (a[4], a[5]),
        nr::PPOLL => (a[3], a[4]),
        // The sixth argument points at `{ const sigset_t *ss; size_t
        // ss_len; }`, with 64-bit fields here: a 32-bit ABI's are widened
        // by its translation (`i386::translate`).
        nr::PSELECT6 => {
            if a[5] == 0 {
                return Ok(None);
            }
            (m.read_u64(a[5])?, m.read_u64(a[5] + 8)?)
        }
        _ => return Ok(None),
    };
    if set == 0 {
        return Ok(None);
    }
    if size != 8 {
        return Err(Errno::EINVAL);
    }
    Ok(Some(m.read_bits64(set)?))
}

/// `rt_sigprocmask` on `signals`, the calling thread's: the whole call,
/// which needs nothing else of a kernel.
/// `rt_sigaction`: the process's disposition of `signo`, read and set
/// through its own lock (`ProcSig`'s), no kernel lock. The i386 struct
/// sigaction is { u32 handler; u32 flags; u32 restorer; u64 mask }.
pub fn sigaction(signals: &Signals, m: &mut GuestMem<'_>, compat: bool, signo: u32, act: u64, old: u64) -> SysResult {
    if signo == 0 || signo > signal::sig::MAX {
        return Err(Errno::EINVAL);
    }
    // Read before the old one is written: they may be the same struct.
    let new = if act == 0 {
        None
    } else if compat {
        Some(SigAction {
            handler: m.read_u32(act)? as u64,
            flags: m.read_u32(act + 4)? as u64,
            restorer: m.read_u32(act + 8)? as u64,
            mask: m.read_bits64(act + 12)?,
        })
    } else {
        Some(SigAction { handler: m.read_u64(act)?, flags: m.read_u64(act + 8)?, restorer: m.read_u64(act + 16)?, mask: m.read_bits64(act + 24)? })
    };
    if old != 0 {
        let a = signals.action(signo);
        if compat {
            m.write_u32(old, a.handler as u32)?;
            m.write_u32(old + 4, a.flags as u32)?;
            m.write_u32(old + 8, a.restorer as u32)?;
            m.write_bits64(old + 12, a.mask)?;
        } else {
            m.write_u64(old, a.handler)?;
            m.write_u64(old + 8, a.flags)?;
            m.write_u64(old + 16, a.restorer)?;
            m.write_bits64(old + 24, a.mask)?;
        }
    }
    if let Some(a) = new {
        signals.set_action(signo, a)?;
    }
    Ok(0)
}

/// `rt_sigpending`: the signals raised on this thread or its process while
/// blocked and not yet delivered, as a `sigset_t` (bit `signal-1`).
pub fn sigpending(signals: &Signals, m: &mut GuestMem<'_>, set: u64) -> SysResult {
    if set != 0 {
        m.write_bits64(set, signals.pending_raw())?;
    }
    Ok(0)
}

/// `sigaltstack`: the calling thread's own (`ThreadSig`). The 32-bit
/// stack_t is { u32 sp; i32 flags; u32 size }.
pub fn sigaltstack(signals: &Signals, m: &mut GuestMem<'_>, compat: bool, new: u64, old: u64) -> SysResult {
    let want = if new == 0 {
        None
    } else if compat {
        Some(SigAltStack { sp: m.read_u32(new)? as u64, flags: m.read_u32(new + 4)?, size: m.read_u32(new + 8)? as u64 })
    } else {
        Some(SigAltStack { sp: m.read_u64(new)?, flags: m.read_u32(new + 8)?, size: m.read_u64(new + 16)? })
    };
    if old != 0 {
        let a = signals.alt_stack();
        if compat {
            m.write_u32(old, a.sp as u32)?;
            m.write_u32(old + 4, a.flags)?;
            m.write_u32(old + 8, a.size as u32)?;
        } else {
            m.write_u64(old, a.sp)?;
            m.write_u32(old + 8, a.flags)?;
            m.write_u64(old + 16, a.size)?;
        }
    }
    if let Some(a) = want {
        signals.set_alt_stack(a);
    }
    Ok(0)
}

/// `setitimer` and `getitimer` (`set`) at `now` (wall nanoseconds). A
/// `struct itimerval` is two `timeval`s (reload interval, then time
/// remaining), each two 64-bit words; the process's timers are behind
/// their own lock (`ProcSig`), and a host that parks threads publishes the
/// next deadline after an arming (`waitq::publish_timer` in the page).
pub fn itimer(signals: &Signals, now: u128, m: &mut GuestMem<'_>, which: u64, new: u64, old: u64, set: bool) -> SysResult {
    let which = which as usize;
    if which > ITIMER_PROF {
        return Err(Errno::EINVAL);
    }
    let want = if !set {
        None
    } else if new == 0 {
        return Err(Errno::EFAULT);
    } else {
        Some((read_timeval(m, new)?, read_timeval(m, new + 16)?))
    };
    // What is left, which both calls may be asked to report.
    if old != 0 {
        let (interval, remaining) = match signals.process().timer(which) {
            Some(t) => (t.interval, t.next.saturating_sub(now)),
            None => (0, 0),
        };
        write_timeval(m, old, interval)?;
        write_timeval(m, old + 16, remaining)?;
    }
    if let Some((interval, value)) = want {
        // A zero `it_value` disarms, whatever the interval says.
        signals.process().set_timer(which, (value != 0).then(|| Itimer { next: now + value, interval }));
    }
    Ok(0)
}

/// `alarm(secs)`: `ITIMER_REAL` armed for `secs` (0 disarms), answering the
/// whole seconds the old one had left, rounded up.
pub fn alarm(signals: &Signals, now: u128, secs: u64) -> u64 {
    let left = signals.process().timer(0).map_or(0, |t| t.next.saturating_sub(now));
    signals.process().set_timer(0, (secs != 0).then(|| Itimer { next: now + secs as u128 * 1_000_000_000, interval: 0 }));
    // As the timeval it was: whole seconds, plus one for any microseconds.
    (left / 1_000_000_000) as u64 + u64::from(left % 1_000_000_000 >= 1_000)
}

/// `getrandom`: bytes from the namespace's `/dev/urandom` (a host may put
/// real entropy behind it), at most 1 MiB a call.
pub fn getrandom(ns: &crate::vfs::Namespace, m: &mut GuestMem<'_>, buf: u64, len: u64) -> SysResult {
    let len = len.min(1 << 20) as usize;
    let handle = ns.open("/dev/urandom", oflags::O_RDONLY, 0)?;
    let mut tmp = vec![0u8; len];
    handle.read_at(0, &mut tmp)?;
    m.write(buf, &tmp)?;
    Ok(len as u64)
}

pub fn sigprocmask(signals: &Signals, m: &mut GuestMem<'_>, how: u32, set: u64, old: u64) -> SysResult {
        const SIG_BLOCK: u32 = 0;
        const SIG_UNBLOCK: u32 = 1;
        const SIG_SETMASK: u32 = 2;
        if old != 0 {
            m.write_bits64(old, signals.blocked())?;
        }
        if set != 0 {
            let mask = m.read_bits64(set)?;
            let blocked = signals.blocked();
            let blocked = match how {
                SIG_BLOCK => blocked | mask,
                SIG_UNBLOCK => blocked & !mask,
                SIG_SETMASK => mask,
                _ => return Err(Errno::EINVAL),
            };
            // Neither of these can ever be blocked.
            let unblockable =
                (1u64 << (signal::sig::SIGKILL - 1)) | (1u64 << (signal::sig::SIGSTOP - 1));
            signals.set_blocked(blocked & !unblockable);
        }
        Ok(0)
}

/// What a working directory that has been removed resolves as: no name
/// under it exists (`ENOENT`, as Linux answers in a removed directory), and
/// `getcwd` says `ENOENT` for it.
pub const GONE_CWD: &str = "/\u{0}gone";

/// An fs_struct: the working directory and the umask, shared by the threads
/// that share one (every pthread: `CLONE_FS`). The process's first is
/// `Kernel::proc_fs`; `unshare(CLONE_FS)` and a `clone` without it make
/// others (`ThreadFs`).
///
/// The directory is held by identity, as Linux holds a dentry: the open
/// directory, which follows a rename of itself or of a directory above it
/// ([`crate::vfs::FileHandle::moved`]). Its path is kept with the
/// filesystem's rename count it was found at, so while no directory there
/// has been renamed (one lock-free load) the path stands.
pub struct FsCwd {
    at: std::sync::Mutex<CwdAt>,
    umask: std::sync::atomic::AtomicU32,
}

#[derive(Clone)]
struct CwdAt {
    /// The path now, and the path the directory was opened at.
    path: String,
    opened: String,
    dir: Option<Arc<dyn crate::vfs::FileHandle>>,
    /// The filesystem's rename count `path` was found at, and the
    /// directory's own move count then ([`crate::vfs::Moved::epoch`]).
    epoch: Option<u64>,
    moves: Option<u64>,
    /// The filesystem's count of its own renames (gex's, through that
    /// mount) when `path` was last found ([`FsCwd::get_call`]).
    own: Option<u64>,
}

impl FsCwd {
    /// At `cwd`, held by name only (until a `chdir` opens it).
    pub fn new(cwd: String) -> FsCwd {
        FsCwd {
            at: std::sync::Mutex::new(CwdAt { opened: cwd.clone(), path: cwd, dir: None, epoch: None, moves: None, own: None }),
            umask: std::sync::atomic::AtomicU32::new(0o022),
        }
    }

    /// A copy, its own from now (`clone` without `CLONE_FS`, `unshare`).
    pub fn copy(&self) -> FsCwd {
        FsCwd {
            at: std::sync::Mutex::new(self.lock().clone()),
            umask: std::sync::atomic::AtomicU32::new(self.umask()),
        }
    }

    /// Take `other`'s directory and umask (`execve` from a thread with an
    /// fs_struct of its own).
    pub fn copy_from(&self, other: &FsCwd) {
        let at = other.lock().clone();
        *self.lock() = at;
        self.set_umask(other.umask());
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CwdAt> {
        self.at.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The directory's path now ([`GONE_CWD`] once it is removed).
    pub fn get(&self) -> String {
        self.get_seen().0
    }

    /// The directory's path now, and how many times the directory had
    /// moved when that was found ([`crate::vfs::Moved::epoch`]; `None` for
    /// one not held by identity): a lookup relative to it that missed while
    /// that changed raced a rename (`PathIo::call`).
    pub fn get_seen(&self) -> (String, Option<u64>) {
        let mut at = self.lock();
        let Some(dir) = at.dir.clone() else { return (at.path.clone(), None) };
        let epoch = dir.rename_epoch();
        if epoch.is_some() && epoch == at.epoch {
            return (at.path.clone(), at.moves);
        }
        (at.path, at.moves) = match dir.moved() {
            Some(Ok(m)) => (m.path_from(&at.opened).unwrap_or_else(|| at.path.clone()), Some(m.epoch)),
            Some(Err(_)) => (GONE_CWD.to_string(), None),
            None => (at.path.clone(), None),
        };
        at.epoch = epoch;
        (at.path.clone(), at.moves)
    }

    /// The path a call starts and ends with (`Kernel::syscall_cwd`), what
    /// resolves outside [`PathIo`]: [`FsCwd::get`], except on a filesystem
    /// that counts its own renames but cannot count the host's (a host
    /// directory). There the path found stands while gex has renamed no
    /// directory through that mount. Asking the host costs a call each
    /// time, twice for every system call. A host process outside gex that
    /// moves the directory is seen by the next lookup relative to it
    /// ([`PathIo::at_path_str`] asks [`FsCwd::get_seen`]), and by `getcwd`.
    pub fn get_call(&self) -> String {
        let own = {
            let at = self.lock();
            let Some(dir) = &at.dir else { return at.path.clone() };
            let own = dir.own_rename_epoch();
            if own.is_some() && own == at.own {
                return at.path.clone();
            }
            own
        };
        // Read before the path is found: a rename in between leaves the
        // count behind, so the next call asks again.
        let path = self.get_seen().0;
        self.lock().own = own;
        path
    }

    /// The directory, when held by identity.
    fn dir(&self) -> Option<Arc<dyn crate::vfs::FileHandle>> {
        self.lock().dir.clone()
    }

    /// Now at `path`, held through `dir` when the filesystem gave one that
    /// follows its directory.
    pub fn set(&self, path: String, dir: Option<Arc<dyn crate::vfs::FileHandle>>) {
        // Its path checked on first use: a rename between the open and now
        // is seen then.
        let dir = dir.filter(|d| d.moved().is_some());
        *self.lock() = CwdAt { opened: path.clone(), path, dir, epoch: None, moves: None, own: None };
    }

    pub fn umask(&self) -> u32 {
        self.umask.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_umask(&self, mask: u32) -> u32 {
        self.umask.swap(mask & 0o777, std::sync::atomic::Ordering::Relaxed)
    }
}

/// The threads of a process whose fs_struct is not the process's first one,
/// by context address (`Kernel::thread_fs`). Threads made with `CLONE_FS`
/// (every pthread) share their creator's; `unshare(CLONE_FS)` and a
/// `clone` without it make one of the thread's own.
#[derive(Default)]
pub struct ThreadFs {
    /// Whether any thread has one: most processes never do.
    any: std::sync::atomic::AtomicBool,
    by_ctx: std::sync::Mutex<std::collections::HashMap<u64, Arc<FsCwd>>>,
}

impl ThreadFs {
    /// The fs_struct of the thread whose context is at `ctx`, if not the
    /// process's first.
    pub fn of(&self, ctx: u64) -> Option<Arc<FsCwd>> {
        if !self.any.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        self.by_ctx.lock().unwrap_or_else(|e| e.into_inner()).get(&ctx).cloned()
    }

    /// Every thread back on the process's fs_struct (`execve`).
    fn clear(&self) {
        self.by_ctx.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    fn set(&self, ctx: u64, fs: Option<Arc<FsCwd>>) {
        let mut m = self.by_ctx.lock().unwrap_or_else(|e| e.into_inner());
        match fs {
            Some(fs) => {
                m.insert(ctx, fs);
                self.any.store(true, std::sync::atomic::Ordering::Release);
            }
            None => {
                m.remove(&ctx);
            }
        }
    }
}

/// The path lookups and opens: what they
/// read is the namespace (the kernel's own, or its published view), the
/// working directory of the calling thread's fs_struct, the descriptor
/// table and the mapping log (an object's pin or placement, armed by its
/// open). The kernel's calls use the same code (`Kernel::path_io`).
pub struct PathIo<'a> {
    pub ns: &'a crate::vfs::Namespace,
    /// The published namespace, which an `O_TMPFILE`'s name is removed
    /// through when its description goes.
    pub view: &'a Arc<crate::vfs::VfsView>,
    /// The descriptor table, the map and its placement.
    pub map: MapIo<'a>,
    pub compat: bool,
    pub pid: i32,
    /// The process's ids, read at each call: the owner a stat reports.
    pub creds: &'a Creds,
    pub procs: &'a ProcTable,
    pub ptys: Option<crate::pty::Remote>,
    /// `O_TMPFILE`'s names, numbered per process.
    pub tmpfiles: &'a std::sync::atomic::AtomicU64,
    /// The calling thread's fs_struct: `chdir`'s, `umask`'s, and what
    /// `mkdir` leaves out of the mode it is asked for. Borrowed where the
    /// caller holds it (a page Worker's call: no count taken per call).
    pub fs: std::borrow::Cow<'a, Arc<FsCwd>>,
    /// For `utimensat`'s "now".
    pub clock: ClockFn,
    /// The bound socket names an unlink or rename of a socket file takes
    /// with it: this kernel's, and the global kernel's when there is one.
    pub socket_names: &'a Arc<unix::SocketNames>,
    pub unix_remote: Option<unix::Remote>,
}

thread_local! {
    /// The directory descriptors this thread's path call resolved through,
    /// and their filesystems' rename epochs then ([`PathIo::call`]).
    static DIRS_SEEN: std::cell::RefCell<Vec<Seen>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A directory a path call resolved through, and how many times it had
/// moved then: a descriptor's, or the working directory.
#[derive(Clone, Copy)]
enum Seen {
    Fd(i32, u64),
    Cwd(u64),
}

impl PathIo<'_> {
    /// The path call `n` (canonical; [`is_path`]), or `None` for another.
    ///
    /// A descriptor's directory, or the working directory, is found where
    /// it is now and the call then resolves through that path, two steps:
    /// a directory renamed between them leaves the call looking where the
    /// directory was. So a call through a directory descriptor or relative
    /// to the working directory that finds nothing (`ENOENT`, `ENOTDIR`)
    /// while that directory moved is made again, as Linux's lookup by the directory itself would not have
    /// missed. A miss has done nothing to undo. After a few such tries the
    /// call is made with the directories' filesystems' renames held off
    /// (`FileHandle::hold_renames`, the filesystem's own lock, not the
    /// kernel's), so it cannot miss for a rename: Linux's walk, after its
    /// optimistic tries on the rename seqlock, falls back to `rename_lock`
    /// the same way.
    pub fn call(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<SysResult> {
        let mut seen = Vec::new();
        for _ in 0..8 {
            DIRS_SEEN.with(|d| d.borrow_mut().clear());
            let r = self.call_once(m, n, a)?;
            if !matches!(r, Err(Errno::ENOENT) | Err(Errno::ENOTDIR)) {
                return Some(r);
            }
            seen = DIRS_SEEN.with(|d| std::mem::take(&mut *d.borrow_mut()));
            let raced = seen.iter().any(|&s| match s {
                Seen::Fd(fd, epoch) => {
                    self.map.fds.get(fd).ok().and_then(|f| f.dir_at().ok().flatten()).is_some_and(|(_, now)| now != epoch)
                }
                Seen::Cwd(epoch) => self.cwd_moves() != Some(epoch),
            });
            if !raced {
                return Some(r);
            }
        }
        // Still racing: once more with the directories' filesystems'
        // renames held off, so the directory found is where the call looks.
        // Released however the call ends.
        struct Held(Vec<Arc<dyn crate::vfs::FileHandle>>, i32);
        impl Drop for Held {
            fn drop(&mut self) {
                for h in &self.0 {
                    h.hold_renames(self.1, false);
                    crate::vfs::note_own_hold(false);
                }
            }
        }
        let mut held = Held(Vec::new(), self.pid);
        for s in seen {
            let h = match s {
                Seen::Fd(fd, _) => self.map.fds.get(fd).ok().map(|f| Arc::clone(&f.handle)),
                Seen::Cwd(_) => self.fs.dir(),
            };
            if let Some(h) = h {
                if h.hold_renames(self.pid, true) {
                    crate::vfs::note_own_hold(true);
                    held.0.push(h);
                }
            }
        }
        let r = self.call_once(m, n, a);
        drop(held);
        r
    }

    /// How many times the working directory has moved, now.
    fn cwd_moves(&self) -> Option<u64> {
        match self.fs.dir()?.moved()? {
            Ok(m) => Some(m.epoch),
            Err(_) => None,
        }
    }

    fn call_once(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<SysResult> {
        Some(match n {
            nr::OPENAT => self.openat(m, a[0] as i32, a[1], oflags::from_guest(a[2], self.compat), a[3] as u32),
            nr::OPENAT2 => self.openat2(m, a[0] as i32, a[1], a[2], a[3]),
            nr::GETDENTS64 => self.getdents(m, a[0] as i32, a[1], a[2]),
            nr::FSTAT => self.fstat(m, a[0] as i32, a[1]),
            nr::NEWFSTATAT => self.fstatat(m, a[0] as i32, a[1], a[2], a[3]),
            nr::STATX => self.statx(m, a[0] as i32, a[1], a[2], a[4]),
            nr::FACCESSAT | nr::FACCESSAT2 => self.faccessat(m, a[0] as i32, a[1], a[2] as u32),
            nr::READLINKAT => self.readlinkat(m, a[0] as i32, a[1], a[2], a[3]),
            nr::GETCWD => self.getcwd(m, a[0], a[1]),
            nr::MKDIRAT => self.sys_mkdirat(m, a[0] as i32, a[1], a[2] as u32),
            nr::CHDIR => self.sys_chdir(m, a[0]),
            nr::FCHDIR => self.sys_fchdir(a[0] as i32),
            // There is one process's view of the world and nothing consults
            // a umask, but a program that sets one wants the old value back.
            nr::UMASK => Ok(self.fs.set_umask(a[0] as u32) as u64),
            nr::UNLINKAT => self.sys_unlinkat(m, a[0] as i32, a[1], a[2]),
            nr::RENAMEAT => self.sys_renameat(m, a[0] as i32, a[1], a[2] as i32, a[3]),
            nr::RENAMEAT2 => self.sys_renameat2(m, a[0] as i32, a[1], a[2] as i32, a[3], a[4] as u32),
            nr::LINKAT => self.sys_linkat(m, a),
            nr::SYMLINKAT => self.sys_symlinkat(m, a[0], a[1] as i32, a[2]),
            // Modification times are not settable here: tmpfs keeps one it
            // maintains itself and hostfs reports the host's. Accepting the
            // call is still better than refusing it -- apt sets the time on
            // a file it has just downloaded, and treats the failure as a
            // failure of the download.
            nr::UTIMENSAT => self.sys_utimensat(m, a[0] as i32, a[1], a[2], a[3] as u32),
            // Permission bits are not modelled; reporting success keeps
            // installers and build systems moving.
            nr::FCHMOD | nr::FCHMODAT => self.sys_chmod(m, n, a),
            nr::SETXATTR..=nr::FREMOVEXATTR => self.sys_xattr(m, n, a),
            // Ownership is not modelled: there is one user and the guest is
            // it. Refusing would be worse than accepting -- tar treats a
            // failed chown while unpacking as a failure to unpack.
            // Ownership is not kept, but the file must be there.
            nr::FCHOWN => self.map.fds.get(a[0] as i32).map(|_| 0),
            nr::FCHOWNAT => {
                let empty = m.read_cstr(a[1], 4096).map(|r| r.is_empty()).unwrap_or(false);
                if empty && a[4] & AT_EMPTY_PATH != 0 {
                    self.map.fds.get(a[0] as i32).map(|_| 0)
                } else {
                    self.at_path(m, a[0] as i32, a[1]).and_then(|p| {
                        if a[4] & AT_SYMLINK_NOFOLLOW != 0 {
                            self.ns.metadata_nofollow(&p).map(|_| 0)
                        } else {
                            self.ns.metadata(&p).map(|_| 0)
                        }
                    })
                }
            }
            _ => return None,
        })
    }

    /// Open a pseudoterminal end if `p` names one, else `None` so the ordinary
    /// open runs. The pair lives in the global kernel (`self.ptys`); with none
    /// there is no pty support and `/dev/ptmx` falls through to `ENOENT`.
    pub fn open_pty(&self, p: &str, flags: u64) -> Option<SysResult> {
        let r = self.ptys?;
        // `/dev/tty` is the caller's *controlling terminal*. Inside a terminal
        // emulator that is a pty in the global kernel, not the page's own
        // terminal -- a shell opens this for job control and compares its
        // foreground group with its own, and the page's terminal never matches:
        // dash then spins raising SIGTTIN at itself, which shows as a window
        // that echoes what is typed and runs none of it. Only when this
        // session has claimed one; otherwise the name falls through to the
        // console handle, which is right for the page's own shell.
        if p == "/dev/tty" {
            let sid = self.procs.sid(self.pid, 0).unwrap_or(0);
            let id = (r.for_sid)(sid);
            if id >= 0 {
                let handle = Arc::new(crate::pty::RemotePty { id, master: false, remote: r });
                // A descriptor of its own on that end, so the counting stays
                // right, but no claim: this session already owns it.
                (r.open_end)(id, 0);
                let file = OpenFile::new(handle, format!("pty:[{id}:s]"), flags, true, true);
                return Some(self.map.fds.insert(file, flags & oflags::O_CLOEXEC != 0));
            }
            return None;
        }
        let (id, master) = if p == "/dev/ptmx" {
            let packed = (r.open_master)();
            if packed < 0 {
                return Some(Err(Errno::ENOSPC));
            }
            (packed >> 16, true)
        } else if let Some(n) = p.strip_prefix("/dev/pts/") {
            let ptn: i32 = n.parse().ok()?;
            let id = (r.open_slave)(ptn);
            if id == -1 {
                return None; // no such pts -- let the VFS answer ENOENT
            }
            if id < 0 {
                return Some(Err(Errno(-id))); // locked, or otherwise refused
            }
            (id, false)
        } else {
            return None;
        };
        let handle = Arc::new(crate::pty::RemotePty { id, master, remote: r });
        // Opening a terminal is how a session claims it.
        //
        // A session leader with no controlling terminal that opens one without
        // O_NOCTTY acquires it, and its process group becomes the terminal's
        // foreground group -- Linux does this in the open, not in `TIOCSCTTY`,
        // and a terminal emulator's child relies on it: xterm's child calls
        // `setsid` and opens the slave, and never issues the ioctl. Without
        // this the shell it execs is in a *background* group, so the kernel
        // answers every read from the terminal with SIGTTIN: a window that
        // echoes what is typed (the line discipline does that) and runs none
        // of it.
        if !master && flags & oflags::O_NOCTTY == 0 {
            let leader = self.procs.sid(self.pid, 0) == Ok(self.pid);
            if leader {
                use crate::vfs::FileHandle;
                handle.claim_session(self.pid);
            }
        }
        let path = format!("pty:[{id}:{}]", if master { 'm' } else { 's' });
        let file = OpenFile::new(handle, path, flags, true, true);
        Some(self.map.fds.insert(file, flags & oflags::O_CLOEXEC != 0))
    }

    pub fn openat(
        &self,
        m: &GuestMem<'_>,
        dirfd: i32,
        path: u64,
        flags: u64,
        mode: u32,
    ) -> SysResult {
        let p = self.at_path(m, dirfd, path)?;
        // Unix98 pseudoterminals. `/dev/ptmx` makes a fresh pair and hands back
        // the master; `/dev/pts/N` takes the matching slave. Both live in the
        // global kernel, so the master (xterm) and the slave (its shell) can be
        // different Workers. Intercepted before the VFS, which has no such nodes.
        if let Some(res) = self.open_pty(&p, flags) {
            return res;
        }
        // `/proc/self/fd/N` (and `/dev/stdin`, `/dev/fd/N`, which lead there)
        // names the descriptor's file itself, as Linux's magic link does,
        // not the name its link reads back.
        if flags & oflags::O_PATH == 0 {
            if let Some(n) = self.magic_fd(&p) {
                return self.reopen_fd(n, flags, mode);
            }
        }
        // `O_TMPFILE` names a directory and asks for an unnamed file inside
        // it: made, opened, and unlinked at once, so the handle is the only
        // way to it. `O_PATH` opens the name, not the file, so it does not
        // follow a dangling symlink to `ENOENT` -- which is how tar sets the
        // mode on a symlink whose target does not exist yet.
        let mut tmpfile_path: Option<String> = None;
        let handle = if flags & oflags::O_PATH != 0 {
            let md = if flags & oflags::O_NOFOLLOW != 0 {
                self.ns.metadata_nofollow(&p)?
            } else {
                self.ns.metadata(&p)?
            };
            if flags & oflags::O_DIRECTORY != 0 && md.file_type != FileType::Directory {
                return Err(Errno::ENOTDIR);
            }
            Arc::new(crate::vfs::PathHandle(md)) as Arc<dyn crate::vfs::FileHandle>
        } else if flags & oflags::O_TMPFILE == oflags::O_TMPFILE {
            // A name can be taken already: one left by a gex that was
            // killed with the descriptor open, from a process that had this
            // pid (pids repeat from run to run). The next number, then.
            let mut tries = 0;
            let (tmp, h) = loop {
                let n = self.tmpfiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let tmp = format!("{}/.gex-tmpfile-{}-{}", p.trim_end_matches('/'), self.pid, n);
                match self.ns.open_mode(
                    &tmp,
                    oflags::O_RDWR | oflags::O_CREAT | oflags::O_EXCL,
                    mode & 0o7777 & !self.fs.umask(),
                ) {
                    Err(Errno::EEXIST) if tries < 256 => tries += 1,
                    r => break (tmp, r?),
                }
            };
            // Kept named (not unlinked here) so `/proc/self/fd/N` resolves to the
            // file and the `linkat(..., AT_SYMLINK_FOLLOW)` idiom can hardlink it
            // into place; the name is removed when the descriptor closes (see
            // `OpenFile::unlink_on_close`). Recording `tmp` as the fd's path --
            // not the directory `p` -- is what makes `/proc/self/fd/N` point at
            // the file rather than at a directory (linking which is EPERM).
            tmpfile_path = Some(tmp);
            h
        } else {
            // Arm a pin if this is an object of the closure. Before the open
            // rather than after, because `open` moves `p` into the call and
            // the path is what identifies the object.
            if let Some(base) = self.map.pin_for(&p) {
                self.map.mem.lock().log.armed_pin = Some((p.clone(), base));
            }
            // O_CREAT of "name/" that is not there: a slash names a
            // directory, which open does not create (EISDIR, as Linux).
            if flags & oflags::O_CREAT != 0 && Self::path_tail(m, path)?.0 && self.ns.metadata(&p).is_err() {
                return Err(Errno::EISDIR);
            }
            // O_NOFOLLOW on a symlink: ELOOP.
            if flags & oflags::O_NOFOLLOW != 0
                && self.ns.metadata_nofollow(&p).is_ok_and(|md| md.file_type == FileType::Symlink)
            {
                return Err(Errno::ELOOP);
            }
            // A directory opens for reading only: each filesystem's `open`
            // says `EISDIR`, without a lookup of its own here.
            // O_CREAT's mode, less the umask, as Linux makes the file.
            let h = self.ns.open_mode(&p, flags, mode & 0o7777 & !self.fs.umask())?;
            // O_DIRECTORY on what is not one: ENOTDIR (mv's and find's test
            // of whether a name is a directory)
            if flags & oflags::O_DIRECTORY != 0 && h.metadata()?.file_type != FileType::Directory {
                return Err(Errno::ENOTDIR);
            }
            // And arm a placement, for an object nobody pinned. Only for a
            // shared object: the executable's own base is fixed by the
            // loader rather than reserved, and arming on every open would
            // hand the next unrelated reservation somebody else's address.
            //
            // After the open rather than before, so a path that does not
            // exist -- `ld.so` probes several directories for every library
            // it loads -- leaves nothing armed.
            if self.map.place.is_some() && is_shared_object(&p) {
                self.map.mem.lock().log.armed_place = Some(p.clone());
            }
            h
        };
        let acc = flags & oflags::ACCMODE;
        let readable = acc == oflags::O_RDONLY || acc == oflags::O_RDWR;
        let writable = acc == oflags::O_WRONLY || acc == oflags::O_RDWR;
        let file = match tmpfile_path {
            Some(tp) => OpenFile::tmpfile(handle, tp, self.view.get(), flags, readable, writable),
            None => OpenFile::new(handle, p, flags, readable, writable),
        };
        self.map.fds.insert(file, flags & oflags::O_CLOEXEC != 0)
    }

    /// The descriptor a path names as Linux's magic links do: `/proc/self/fd/N`,
    /// `/proc/thread-self/fd/N` and `/proc/<own pid>/fd/N`, `/dev/fd/N`, and
    /// `/dev/stdin`, `/dev/stdout` and `/dev/stderr` (links to the first).
    fn magic_fd(&self, p: &str) -> Option<i32> {
        match p {
            "/dev/stdin" => return Some(0),
            "/dev/stdout" => return Some(1),
            "/dev/stderr" => return Some(2),
            _ => {}
        }
        let own = format!("/proc/{}/fd/", self.pid);
        let n = ["/proc/self/fd/", "/proc/thread-self/fd/", own.as_str(), "/dev/fd/"]
            .iter()
            .find_map(|pre| p.strip_prefix(pre))?;
        if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        n.parse().ok()
    }

    /// `open` of descriptor `n`'s magic link ([`Self::magic_fd`]): a new open
    /// file description of the same file, with its own offset and the flags
    /// of this call. A pipe, a terminal or a device is the same object (a
    /// pipe end only in its own direction); a file or directory is the same
    /// one too, unless the call asks for more access than the descriptor has
    /// or truncates, when it is opened again by its name. A socket cannot be
    /// opened so (`ENXIO`, as Linux).
    fn reopen_fd(&self, n: i32, flags: u64, mode: u32) -> SysResult {
        let of = self.map.fds.get(n).map_err(|_| Errno::ENOENT)?;
        let acc = flags & oflags::ACCMODE;
        let readable = acc == oflags::O_RDONLY || acc == oflags::O_RDWR;
        let writable = acc == oflags::O_WRONLY || acc == oflags::O_RDWR;
        let within = (!readable || of.readable) && (!writable || of.writable);
        let ft = of.handle.metadata()?.file_type;
        let handle = match ft {
            FileType::Socket => return Err(Errno::ENXIO),
            FileType::Fifo if !within => return Err(Errno::ENXIO),
            FileType::Regular | FileType::Directory if !within || flags & oflags::O_TRUNC != 0 => {
                if flags & oflags::O_DIRECTORY != 0 && ft != FileType::Directory {
                    return Err(Errno::ENOTDIR);
                }
                self.ns.open(&of.path, flags & !(oflags::O_CREAT | oflags::O_EXCL), mode)?
            }
            _ => {
                if flags & oflags::O_DIRECTORY != 0 && ft != FileType::Directory {
                    return Err(Errno::ENOTDIR);
                }
                if flags & (oflags::O_CREAT | oflags::O_EXCL) == oflags::O_CREAT | oflags::O_EXCL {
                    return Err(Errno::EEXIST);
                }
                Arc::clone(&of.handle)
            }
        };
        let file = OpenFile::new(handle, of.path.clone(), flags & !(oflags::O_CREAT | oflags::O_EXCL | oflags::O_TRUNC), readable, writable);
        self.map.fds.insert(file, flags & oflags::O_CLOEXEC != 0)
    }

    /// `openat2(dirfd, path, how, size)`: `openat` with an extensible
    /// `open_how` struct { u64 flags; u64 mode; u64 resolve } instead of a bare
    /// flags word.
    ///
    /// `resolve` is the reason a program reaches for this syscall specifically:
    /// RESOLVE_BENEATH / RESOLVE_IN_ROOT / RESOLVE_NO_SYMLINKS pin path lookup
    /// inside a subtree, which is how a sandbox keeps a lookup from escaping.
    /// This VFS cannot enforce that per-open, and the one thing worse than not
    /// offering containment is offering it falsely -- so any non-zero `resolve`
    /// gets EINVAL rather than an open that silently ignored the constraint.
    /// `resolve == 0` is ordinary resolution and routes to the `openat` path.
    pub fn openat2(
        &self,
        m: &GuestMem<'_>,
        dirfd: i32,
        path: u64,
        how: u64,
        size: u64,
    ) -> SysResult {
        // The struct is versioned by length; Linux rejects one shorter than the
        // first defined version (three u64s = 24 bytes).
        if size < 24 {
            return Err(Errno::EINVAL);
        }
        let flags = oflags::from_guest(m.read_u64(how)?, self.compat);
        let mode = m.read_u64(how + 8)? as u32;
        let resolve = m.read_u64(how + 16)?;
        if resolve != 0 {
            return Err(Errno::EINVAL);
        }
        self.openat(m, dirfd, path, flags, mode)
    }

    pub fn getdents(&self, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64) -> SysResult {
        let file = self.map.fds.get(fd)?;
        let bytes = file.read_dirents(len as usize)?;
        m.write(buf, &bytes)?;
        Ok(bytes.len() as u64)
    }

    /// `statx`, how glibc stats a file on a modern kernel.
    ///
    /// Implemented rather than left to answer `ENOSYS`: the `fstatat` fallback
    /// returned the right answer but `ls` then branched into padding, so serve
    /// the path glibc actually wants.
    pub fn statx(
        &self,
        m: &mut GuestMem<'_>,
        dirfd: i32,
        path: u64,
        flags: u64,
        buf: u64,
    ) -> SysResult {
        let raw = m.read_cstr(path, 4096)?;
        let (md, mnt_id) = if raw.is_empty() && flags & AT_EMPTY_PATH != 0 {
            let f = self.map.fds.get(dirfd)?;
            let id = if f.path.starts_with('/') { self.ns.mount_id_of(&f.path) } else { 1 };
            (f.metadata()?, id)
        } else {
            let p = self.at_path(m, dirfd, path)?;
            let md = if flags & AT_SYMLINK_NOFOLLOW != 0 {
                self.ns.metadata_nofollow(&p)?
            } else {
                self.ns.metadata(&p)?
            };
            (md, self.ns.mount_id_of(&p))
        };
        let st = Kernel::stat_from(&md);

        // `struct statx` is 256 bytes and mostly reserved.
        let mut out = [0u8; 256];
        let put32 = |o: &mut [u8; 256], at: usize, v: u32| {
            o[at..at + 4].copy_from_slice(&v.to_le_bytes())
        };
        let put64 = |o: &mut [u8; 256], at: usize, v: u64| {
            o[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        // What is filled in. The caller asks for a mask and the kernel says
        // which parts it actually answered; claiming everything but the
        // birth time is what a Linux filesystem without one reports.
        const STATX_BASIC_STATS: u32 = 0x07ff;
        // And the mount's id, as Linux has since 5.8 whether asked or not,
        // the one /proc/self/mountinfo gives it. systemd's path lookups
        // (chase) compare mount ids, and stop where a statx has none
        // (sysusers found no dbus.conf).
        const STATX_MNT_ID: u32 = 0x1000;
        put32(&mut out, 0, STATX_BASIC_STATS | STATX_MNT_ID);
        put64(&mut out, 144, mnt_id);
        put32(&mut out, 4, st.blksize as u32);
        put32(&mut out, 16, st.nlink as u32);
        put32(&mut out, 20, self.creds.uid());
        put32(&mut out, 24, self.creds.gid());
        out[28..30].copy_from_slice(&(st.mode as u16).to_le_bytes());
        put64(&mut out, 32, st.ino);
        put64(&mut out, 40, st.size as u64);
        put64(&mut out, 48, st.blocks as u64);
        // atime, btime, ctime, mtime: 16 bytes each from offset 64, seconds
        // then nanoseconds; btime is reported as mtime.
        for (at, (t, ns)) in [64usize, 80, 96, 112].into_iter().zip([
            (st.atime, st.atime_ns),
            (st.mtime, st.mtime_ns),
            (st.ctime, st.ctime_ns),
            (st.mtime, st.mtime_ns),
        ]) {
            put64(&mut out, at, t as u64);
            put32(&mut out, at + 8, ns);
        }
        put32(&mut out, 128, (st.rdev >> 8) as u32);
        put32(&mut out, 132, (st.rdev & 0xff) as u32);
        m.write(buf, &out)?;
        Ok(0)
    }

    pub fn fstat(&self, m: &mut GuestMem<'_>, fd: i32, buf: u64) -> SysResult {
        let md = self.map.fds.get(fd)?.metadata()?;
        m.write(buf, &self.stat_owned(&md).encode_for(self.compat))?;
        Ok(0)
    }

    /// [`Kernel::stat_from`], owned by the process's own ids, as `statx`
    /// reports them: a program that stats its files and compares the owner
    /// with `getuid` (Wine's prefix check) sees itself. Before, `stat` and
    /// `fstat` said root, so only 32-bit programs (whose libc stats through
    /// `statx`) passed.
    fn stat_owned(&self, md: &crate::vfs::Metadata) -> Stat {
        Stat { uid: self.creds.uid(), gid: self.creds.gid(), ..Kernel::stat_from(md) }
    }

    pub fn fstatat(
        &self,
        m: &mut GuestMem<'_>,
        dirfd: i32,
        path: u64,
        buf: u64,
        flags: u64,
    ) -> SysResult {
        crate::phase::mark(5);
        // A path that fits is read onto the stack, and a normal absolute one
        // is used as it stands: no allocation on the common path (each took
        // the process allocator's one lock, which threads stat-ing at once
        // queued on).
        let mut small = [0u8; 256];
        let long;
        let raw: &[u8] = match m.read_cstr_into(path, &mut small)? {
            Some(n) => &small[..n],
            None => {
                long = m.read_cstr(path, 4096)?;
                &long
            }
        };
        if raw.is_empty() && flags & AT_EMPTY_PATH != 0 {
            return self.fstat(m, dirfd, buf);
        }
        // The path as read: not read again (`at_path`).
        let p = self.at_path_cow(dirfd, std::str::from_utf8(raw).map_err(|_| Errno::EINVAL)?)?;
        crate::phase::mark(6);
        let md = if flags & AT_SYMLINK_NOFOLLOW != 0 {
            self.ns.metadata_nofollow(&p)
        } else {
            self.ns.metadata(&p)
        };
        crate::phase::mark(7);
        let md = md?;
        m.write(buf, &self.stat_owned(&md).encode_for(self.compat))?;
        crate::phase::mark(9);
        Ok(0)
    }

    pub fn faccessat(&self, m: &GuestMem<'_>, dirfd: i32, path: u64, mode: u32) -> SysResult {
        let p = self.at_path(m, dirfd, path)?;
        // A root guest (`--uid 0`) may read and write anything; execute
        // still needs an execute bit, which the host checks.
        let mode = if self.creds.ruid() == 0 { mode & 1 } else { mode & 7 };
        self.ns.access(&p, mode).map(|_| 0)
    }

    pub fn readlinkat(
        &self,
        m: &mut GuestMem<'_>,
        dirfd: i32,
        path: u64,
        buf: u64,
        size: u64,
    ) -> SysResult {
        let p = self.at_path(m, dirfd, path)?;
        let target = self.ns.read_link(&p)?;
        let bytes = target.as_bytes();
        let n = bytes.len().min(size as usize);
        m.write(buf, &bytes[..n])?;
        Ok(n as u64)
    }

    pub fn getcwd(&self, m: &mut GuestMem<'_>, buf: u64, size: u64) -> SysResult {
        // Removed: Linux's `getcwd` says so.
        let cwd = self.fs.get();
        if cwd == GONE_CWD {
            return Err(Errno::ENOENT);
        }
        let mut bytes = cwd.into_bytes();
        bytes.push(0);
        if bytes.len() as u64 > size {
            return Err(Errno::ERANGE);
        }
        m.write(buf, &bytes)?;
        Ok(bytes.len() as u64)
    }

    /// `chdir`: the calling thread's fs_struct's directory, held by
    /// identity (the open directory), so it follows a rename as Linux's
    /// does.
    pub fn sys_chdir(&self, m: &GuestMem<'_>, path: u64) -> SysResult {
        let raw = m.read_cstr(path, 4096)?;
        let p = String::from_utf8(raw).map_err(|_| Errno::EINVAL)?;
        if p.is_empty() {
            return Err(Errno::ENOENT);
        }
        let abs = self.at_path_str(AT_FDCWD, &p)?;
        self.enter(abs, None)
    }

    /// `fchdir`: the working directory named by a descriptor, where the
    /// directory is now. coreutils' `mkdir -p` needs this: gnulib's
    /// `savewd` returns between components with `fchdir`.
    pub fn sys_fchdir(&self, fd: i32) -> SysResult {
        let file = self.map.fds.get(fd)?;
        let path = file.dir_path()?.unwrap_or_else(|| file.path.clone());
        // A pipe or a socket has a name that is not a path. `ENOTDIR` is
        // what Linux says about a descriptor that is not a directory.
        if !path.starts_with('/') {
            return Err(Errno::ENOTDIR);
        }
        let held = file.handle.moved().is_some().then(|| Arc::clone(&file.handle));
        self.enter(path, held)
    }

    /// Make `abs` the fs_struct's directory: it must be one. Held through
    /// `dir`, or a handle opened on it here.
    fn enter(&self, abs: String, dir: Option<Arc<dyn crate::vfs::FileHandle>>) -> SysResult {
        // Followed through its links: the directory, by the name it has.
        let md = self.ns.metadata(&abs)?;
        if md.file_type != FileType::Directory {
            return Err(Errno::ENOTDIR);
        }
        let dir = dir.or_else(|| self.ns.open(&abs, oflags::O_RDONLY | oflags::O_DIRECTORY, 0).ok());
        self.fs.set(abs, dir);
        Ok(0)
    }

    /// How the path at `ptr` ends, which normalising loses: with a slash
    /// (`name/`), and whether its last component is `.` (1) or `..` (2).
    pub fn path_tail(m: &GuestMem<'_>, ptr: u64) -> Result<(bool, u8), Errno> {
        let raw = m.read_cstr(ptr, 4096)?;
        let slash = raw.len() > 1 && raw.ends_with(b"/");
        let last = raw.split(|&b| b == b'/').filter(|c| !c.is_empty()).next_back();
        let dot = match last {
            Some(b".") => 1,
            Some(b"..") => 2,
            _ => 0,
        };
        Ok((slash, dot))
    }

    /// A new non-directory name (link, symlink) that ends in a slash or in
    /// `.`/`..` names a directory: EEXIST when something is there, else
    /// ENOENT, as Linux answers.
    pub fn new_name_check(&self, m: &GuestMem<'_>, ptr: u64, p: &str) -> Result<(), Errno> {
        let (slash, dot) = Self::path_tail(m, ptr)?;
        if slash || dot != 0 {
            return Err(if self.ns.metadata_nofollow(p).is_ok() { Errno::EEXIST } else { Errno::ENOENT });
        }
        Ok(())
    }

    /// `linkat`: a second name for a file that already exists.
    pub fn sys_linkat(&self, m: &GuestMem<'_>, a: [u64; 6]) -> SysResult {
        const AT_SYMLINK_FOLLOW: u64 = 0x400;
        let from = self.at_path(m, a[0] as i32, a[1])?;
        // With AT_SYMLINK_FOLLOW the link is to what `from` points at, not the
        // symlink itself. This is how the O_TMPFILE atomic-create idiom names an
        // unnamed file: write to an `O_TMPFILE` fd, then
        // `linkat("/proc/self/fd/N", target, AT_SYMLINK_FOLLOW)`. gex's
        // `/proc/self/fd/N` is a symlink to the temp's real path, so following it
        // here turns a would-be cross-mount link (/proc vs the target's fs, an
        // EXDEV) into the same-fs link it really is. QSaveFile (Qt, hence every
        // KDE app) relies on it to save a file.
        let from = if a[4] & AT_SYMLINK_FOLLOW != 0 {
            let mut cur = from;
            for _ in 0..40 {
                match self.ns.read_link(&cur) {
                    Ok(target) => {
                        cur = if target.starts_with('/') {
                            crate::vfs::normalize(&target, "/")
                        } else {
                            let parent = crate::vfs::parent_of(&cur);
                            crate::vfs::normalize(&target, &parent)
                        };
                    }
                    Err(_) => break,
                }
            }
            cur
        } else {
            from
        };
        let to = self.at_path(m, a[2] as i32, a[3])?;
        self.new_name_check(m, a[3], &to)?;
        self.ns.link(&from, &to).map(|_| 0)
    }

    pub fn sys_mkdirat(&self, m: &GuestMem<'_>, dirfd: i32, path: u64, mode: u32) -> SysResult {
        // "link/" where link dangles: the name is taken (EEXIST); resolving
        // the slash would say ENOENT
        if Self::path_tail(m, path)?.0 {
            let raw = String::from_utf8(m.read_cstr(path, 4096)?).map_err(|_| Errno::EINVAL)?;
            let bare = self.at_path_str(dirfd, raw.trim_end_matches('/'))?;
            if self.ns.metadata_nofollow(&bare).is_ok() {
                return Err(Errno::EEXIST);
            }
        }
        let p = self.at_path(m, dirfd, path)?;
        // "dir/." and "dir/.." exist if dir does, else ENOENT
        if Self::path_tail(m, path)?.1 != 0 {
            return Err(if self.ns.metadata(&p).is_ok() { Errno::EEXIST } else { Errno::ENOENT });
        }
        self.ns.mkdir(&p, mode & !self.fs.umask()).map(|_| 0)
    }

    pub fn sys_unlinkat(&self, m: &GuestMem<'_>, dirfd: i32, path: u64, flags: u64) -> SysResult {
        let tail = Self::path_tail(m, path)?;
        let p = self.at_path(m, dirfd, path)?;
        if flags & AT_REMOVEDIR != 0 {
            // rmdir("dir/.") is EINVAL, rmdir("dir/..") ENOTEMPTY
            match tail.1 {
                1 => return Err(Errno::EINVAL),
                2 => return Err(Errno::ENOTEMPTY),
                _ => {}
            }
            self.ns.rmdir(&p).map(|_| 0)
        } else {
            let sock = self.is_socket_file(&p);
            self.ns.unlink(&p)?;
            if sock {
                self.socket_renamed(&p, None);
            }
            Ok(0)
        }
    }

    fn is_socket_file(&self, p: &str) -> bool {
        self.ns.metadata_nofollow(p).is_ok_and(|md| md.file_type == FileType::Socket)
    }

    /// The socket file at `from` unlinked (`to` none) or renamed: its bound
    /// name goes with it ([`unix::SocketNames::rename_name`]), here and in
    /// the global kernel, so the path can be bound again.
    fn socket_renamed(&self, from: &str, to: Option<&str>) {
        self.socket_names.rename_name(from, to);
        if let Some(r) = self.unix_remote {
            let to = to.unwrap_or("");
            (r.rename)(from.as_ptr() as crate::PAddr, from.len() as u32, to.as_ptr() as crate::PAddr, to.len() as u32);
        }
    }

    pub fn sys_renameat(
        &self,
        m: &GuestMem<'_>,
        old_fd: i32,
        old: u64,
        new_fd: i32,
        new: u64,
    ) -> SysResult {
        self.sys_renameat2(m, old_fd, old, new_fd, new, 0)
    }

    /// `renameat2`: `renameat` plus a flags word. The flags matter -- programs
    /// reach for this syscall specifically to get the atomic guarantee, so
    /// silently ignoring them would corrupt the very race they were avoiding.
    ///
    /// - `flags == 0` is plain `renameat`.
    /// - `RENAME_NOREPLACE` (1): fail with `EEXIST` if the destination already
    ///   exists, rather than clobbering it -- the atomic create-or-fail idiom.
    ///   The filesystem makes the check and the move one step
    ///   (`FileSystem::rename2`), so no other thread's call comes between.
    /// - `RENAME_EXCHANGE` (2) and `RENAME_WHITEOUT` (4) are not something this
    ///   VFS can do atomically, so they get `EINVAL` -- the same answer Linux
    ///   gives on a filesystem that does not support them.
    pub fn sys_renameat2(
        &self,
        m: &GuestMem<'_>,
        old_fd: i32,
        old: u64,
        new_fd: i32,
        new: u64,
        flags: u32,
    ) -> SysResult {
        const RENAME_NOREPLACE: u32 = 1;
        const RENAME_EXCHANGE: u32 = 2;
        // "." or ".." as the last component of either name: EINVAL (the
        // normalised paths would name the directory itself)
        for ptr in [old, new] {
            let raw = m.read_cstr(ptr, 4096)?;
            let last = raw.split(|&b| b == b'/').filter(|c| !c.is_empty()).next_back();
            if matches!(last, Some(b".") | Some(b"..")) {
                return Err(Errno::EINVAL);
            }
        }
        let from = self.at_path(m, old_fd, old)?;
        // "x" to "link/" where link is a symlink: the slash asks to follow
        // it, and rename never replaces through a link (ENOTDIR, as GNU)
        if Self::path_tail(m, new)?.0 {
            let raw = String::from_utf8(m.read_cstr(new, 4096)?).map_err(|_| Errno::EINVAL)?;
            let bare = self.at_path_str(new_fd, raw.trim_end_matches('/'))?;
            if self.ns.metadata_nofollow(&bare).is_ok_and(|md| md.file_type == FileType::Symlink) {
                return Err(Errno::ENOTDIR);
            }
        }
        let to = self.at_path(m, new_fd, new)?;
        // "file" to "name/": a slash names a directory, which a file is not
        if Self::path_tail(m, new)?.0
            && self.ns.metadata_nofollow(&from).is_ok_and(|md| md.file_type != FileType::Directory)
        {
            return Err(Errno::ENOTDIR);
        }
        if flags & !RENAME_NOREPLACE != 0 {
            // EXCHANGE, WHITEOUT, or any unknown bit: unsupported here.
            let _ = RENAME_EXCHANGE;
            return Err(Errno::EINVAL);
        }
        let from_moves = self.ns.metadata_nofollow(&from).is_ok_and(|md| matches!(md.file_type, FileType::Socket | FileType::Directory));
        let to_sock = self.is_socket_file(&to);
        self.ns.rename2(&from, &to, flags & RENAME_NOREPLACE != 0)?;
        if from_moves && from != to {
            self.socket_renamed(&from, Some(&to));
        } else if to_sock && from != to {
            self.socket_renamed(&to, None);
        }
        Ok(0)
    }

    /// `utimensat(dirfd, path, times, flags)`: set a file's timestamps.
    ///
    /// gex keeps one time per file, the modification time (`stat` reports
    /// it for all three), so the access time in `times` is checked and
    /// dropped. `times` NULL or `UTIME_NOW` means now; `UTIME_OMIT` leaves
    /// the time alone. A missing path is `ENOENT`, which `touch` relies on
    /// to create the file. A NULL path is `futimens` on `dirfd`, whose
    /// descriptor is validated; its time is not kept (an open file's name
    /// is not tracked).
    pub fn sys_utimensat(&self, m: &GuestMem<'_>, dirfd: i32, path: u64, times: u64, flags: u32) -> SysResult {
        const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
        const UTIME_NOW: u64 = (1 << 30) - 1;
        const UTIME_OMIT: u64 = (1 << 30) - 2;
        // The access and modification times to set (`None`: leave it).
        let now = {
            let (s, ns) = (self.clock)();
            (s as i64, ns as u32)
        };
        let (atime, mtime) = if times == 0 {
            (Some(now), Some(now))
        } else {
            let mut v = [None, None];
            for (k, t) in [times, times + 16].into_iter().enumerate() {
                let ns = m.read_u64(t + 8)?;
                v[k] = match ns {
                    UTIME_OMIT => None,
                    UTIME_NOW => Some(now),
                    _ if ns >= 1_000_000_000 => return Err(Errno::EINVAL),
                    _ => Some((m.read_u64(t)? as i64, ns as u32)),
                };
            }
            (v[0], v[1])
        };
        if path == 0 {
            // futimens: the "path" is really the open descriptor in `dirfd`.
            self.map.fds.get(dirfd)?.handle.set_times(atime, mtime)?;
            return Ok(0);
        }
        let p = self.at_path(m, dirfd, path)?;
        let follow = flags & AT_SYMLINK_NOFOLLOW == 0;
        if atime.is_some() || mtime.is_some() {
            // Which says `ENOENT` itself.
            self.ns.set_times(&p, atime, mtime, follow)?;
        } else if follow {
            self.ns.metadata(&p)?;
        } else {
            self.ns.metadata_nofollow(&p)?;
        }
        Ok(0)
    }

    /// `symlinkat(target, newdirfd, linkpath)`. The link path resolves
    /// against `newdirfd`, not the working directory -- tar works relative to
    /// a descriptor it just created, and against the cwd that is `ENOENT`.
    pub fn sys_symlinkat(
        &self,
        m: &GuestMem<'_>,
        target: u64,
        newdirfd: i32,
        path: u64,
    ) -> SysResult {
        let t = String::from_utf8(m.read_cstr(target, 4096)?).map_err(|_| Errno::EINVAL)?;
        let p = self.at_path(m, newdirfd, path)?;
        self.new_name_check(m, path, &p)?;
        self.ns.symlink(&t, &p).map(|_| 0)
    }

    /// `fchmod` and `fchmodat`. The first names an open file, whose path the
    /// descriptor kept; the second names a path directly.
    /// The extended-attribute calls: `{,l,f}{set,get,list,remove}xattr`.
    /// Only `user.` attributes are kept (`vfs::xattr_name`); a filesystem
    /// that keeps none says `EOPNOTSUPP`.
    pub fn sys_xattr(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
        let kind = (n - nr::SETXATTR) / 3; // 0 set, 1 get, 2 list, 3 remove
        let form = (n - nr::SETXATTR) % 3; // 0 path, 1 lpath, 2 fd
        let name = if kind == 2 {
            String::new()
        } else {
            let raw = m.read_cstr(a[1], crate::vfs::XATTR_NAME_MAX + 1)?;
            if raw.len() > crate::vfs::XATTR_NAME_MAX {
                return Err(Errno::ERANGE);
            }
            String::from_utf8(raw).map_err(|_| Errno::EINVAL)?
        };
        let x = Xattr { kind, name: &name, flags: a[4] as u32, at: if kind == 2 { a[1] } else { a[2] }, size: if kind == 2 { a[2] } else { a[3] } };
        if form == 2 {
            let f = self.map.fds.get(a[0] as i32)?;
            if let Some(r) = x.run(m, |op| f.handle.xattr(op)) {
                return r;
            }
            // No node of its own here: by the name it was opened under, as
            // `fchmod` falls back. A pipe or socket has no name, and no
            // attributes.
            if f.path.is_empty() {
                return Err(Errno::EOPNOTSUPP);
            }
            return x.run(m, |op| Some(self.ns.xattr(&f.path, true, op))).unwrap_or(Err(Errno::EOPNOTSUPP));
        }
        let path = self.at_path(m, AT_FDCWD, a[0])?;
        x.run(m, |op| Some(self.ns.xattr(&path, form == 0, op))).unwrap_or(Err(Errno::EOPNOTSUPP))
    }

    pub fn sys_chmod(&self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
        let (path, mode, flags) = if n == nr::FCHMOD {
            let f = self.map.fds.get(a[0] as i32)?;
            // The handle, when it can: no walk of the name it kept, and
            // what it is open on is not a symlink.
            if let Some(r) = f.handle.chmod(a[1] as u32) {
                return r.map(|_| 0);
            }
            (f.path.clone(), a[1] as u32, 0)
        } else {
            (self.at_path(m, a[0] as i32, a[1])?, a[2] as u32, a[3])
        };
        // `chmod` on a symlink itself is `EOPNOTSUPP`, as on Linux: a symlink
        // has no mode of its own. Following the link would change a different
        // file, and answers `ENOENT` when the target does not exist yet --
        // which tar hits unpacking a package with a dangling symlink.
        if self
            .ns
            .metadata_nofollow(&path)
            .is_ok_and(|m| m.file_type == FileType::Symlink)
            && (flags & AT_SYMLINK_NOFOLLOW != 0 || n == nr::FCHMOD)
        {
            return Err(Errno::EOPNOTSUPP);
        }
        self.ns.chmod(&path, mode).map(|_| 0)
    }

    /// `mount` or `umount2` ([`is_mount`]): the mount table, through its
    /// published view, its one writer; the kernel's copy follows at its
    /// next call ([`crate::vfs::Vfs::refresh`]).
    pub fn mount_call(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
        if n == nr::CHROOT {
            return self.chroot(m, a[0]);
        }
        if n == nr::UMOUNT2 {
            let target = self.at_path(m, AT_FDCWD, a[0])?;
            return if self.view.unmount(&target) { Ok(0) } else { Err(Errno::EINVAL) };
        }
        let target = self.at_path(m, AT_FDCWD, a[1])?;
        let fstype = if a[2] == 0 {
            String::new()
        } else {
            String::from_utf8_lossy(&m.read_cstr(a[2], 64)?).into_owned()
        };
        // A remount or a bind of something already there changes nothing gex
        // models, and reporting failure would stop an init script that is in
        // fact fine.
        const MS_REMOUNT: u64 = 32;
        const MS_BIND: u64 = 4096;
        if a[3] & (MS_REMOUNT | MS_BIND) != 0 {
            return Ok(0);
        }
        match fstype.as_str() {
            "tmpfs" | "ramfs" => {
                self.view.mount(&target, Arc::new(crate::vfs::TmpFs::new()));
                Ok(0)
            }
            "devtmpfs" | "devfs" => {
                self.view.mount(&target, Arc::new(crate::vfs::DevFs::new()));
                Ok(0)
            }
            // `/proc` needs a source of process state that only the host can
            // build, so it cannot be created on demand -- but the host has
            // already mounted one, and a second request for it is satisfied.
            "proc" | "procfs" => {
                if self.view.get().is_mount_point(&target) {
                    Ok(0)
                } else {
                    Err(Errno::ENODEV)
                }
            }
            // An empty `/sys` is what most programs want from sysfs: they
            // probe for a path, do not find it, and take the other branch.
            "sysfs" | "debugfs" | "cgroup" | "cgroup2" => {
                self.view.mount_unless_there(&target, || Arc::new(crate::vfs::TmpFs::new()));
                Ok(0)
            }
            _ => Err(Errno::ENODEV),
        }
    }

    /// `chroot(path)`: the namespace's root, as published, now `path`
    /// ([`crate::vfs::Namespace::chroot`]), for every thread of the process
    /// (Linux's is the calling thread's fs_struct, which its threads
    /// share); the caller's directory moved with it. Needs
    /// `CAP_SYS_CHROOT`, as on Linux.
    fn chroot(&self, m: &GuestMem<'_>, path: u64) -> SysResult {
        const CAP_SYS_CHROOT: u64 = 1 << 18;
        let p = self.at_path(m, AT_FDCWD, path)?;
        let dir = self.ns.canonical(&p);
        if self.ns.metadata(&dir)?.file_type != FileType::Directory {
            return Err(Errno::ENOTDIR);
        }
        if self.creds.caps()[0] & CAP_SYS_CHROOT == 0 {
            return Err(Errno::EPERM);
        }
        self.view.chroot(&dir)?;
        let cwd = self.fs.get();
        let inside = cwd.len() > dir.len() && cwd.starts_with(&dir) && (dir == "/" || cwd.as_bytes()[dir.len()] == b'/');
        if dir != "/" {
            let new = if inside { cwd[dir.len()..].to_string() } else { "/".to_string() };
            self.fs.set(new, None);
        }
        Ok(0)
    }

    /// What `truncate(path, length)` truncates: the file `path` names (its
    /// links followed), opened for writing as `ftruncate`'s descriptor
    /// would be. A directory is `EISDIR`.
    pub fn truncate_target(&self, m: &GuestMem<'_>, path: u64) -> Result<Arc<OpenFile>, Errno> {
        let p = self.at_path(m, AT_FDCWD, path)?;
        if self.ns.metadata(&p)?.file_type == FileType::Directory {
            return Err(Errno::EISDIR);
        }
        let h = self.ns.open(&p, oflags::O_WRONLY, 0)?;
        Ok(OpenFile::new(h, p, oflags::O_WRONLY, false, true))
    }

    pub fn at_path(&self, m: &GuestMem<'_>, dirfd: i32, path: u64) -> Result<String, Errno> {
        let raw = m.read_cstr(path, 4096)?;
        self.at_path_str(dirfd, std::str::from_utf8(&raw).map_err(|_| Errno::EINVAL)?)
    }

    /// [`Self::at_path_str`], borrowing `p` when it is absolute and already
    /// normal: what that would return, since a normal path has no trailing
    /// slash to check and is not relative to any directory.
    pub fn at_path_cow<'a>(&self, dirfd: i32, p: &'a str) -> Result<std::borrow::Cow<'a, str>, Errno> {
        if crate::vfs::is_normal(p) {
            return Ok(std::borrow::Cow::Borrowed(p));
        }
        self.at_path_str(dirfd, p).map(std::borrow::Cow::Owned)
    }

    pub fn at_path_str(&self, dirfd: i32, p: &str) -> Result<String, Errno> {
        // An empty path names nothing (the AT_EMPTY_PATH forms check for it
        // before they get here), as on Linux.
        if p.is_empty() {
            return Err(Errno::ENOENT);
        }
        let abs = if p.starts_with('/') {
            crate::vfs::normalize(&p, "/")
        } else if dirfd == AT_FDCWD {
            // Relative to the working directory, where it is now.
            let (cwd, moves) = self.fs.get_seen();
            if let Some(moves) = moves {
                DIRS_SEEN.with(|d| d.borrow_mut().push(Seen::Cwd(moves)));
            }
            crate::vfs::normalize(&p, &cwd)
        } else {
            self.at_dir_path(dirfd, &p)?
        };
        // A trailing slash names a directory: "file/" is ENOTDIR (the
        // normalised path has lost the slash, so look now).
        if p.len() > 1 && p.ends_with('/') {
            let followed = self.ns.metadata(&abs);
            if let Ok(md) = followed.as_ref() {
                if md.file_type != crate::vfs::FileType::Directory {
                    return Err(Errno::ENOTDIR);
                }
                // "link/" names the directory the link points to, even for
                // lstat and the other calls that do not follow a last link
                let mut cur = abs;
                for _ in 0..40 {
                    match self.ns.read_link(&cur) {
                        Ok(t) if t.starts_with('/') => cur = crate::vfs::normalize(&t, "/"),
                        Ok(t) => cur = crate::vfs::normalize(&t, &crate::vfs::parent_of(&cur)),
                        Err(_) => break,
                    }
                }
                return Ok(cur);
            }
            // a dangling or looping link with a slash names nothing: the
            // error following it gives (ENOENT, ELOOP)
            if self.ns.metadata_nofollow(&abs).is_ok() {
                return Err(followed.err().unwrap_or(Errno::ENOENT));
            }
        }
        Ok(abs)
    }

    pub fn at_dir_path(&self, dirfd: i32, p: &str) -> Result<String, Errno> {
        // Relative to an open directory: where the directory is now (it
        // follows a rename, as on Linux), else its recorded path. A
        // descriptor that is not a directory is ENOTDIR.
        let f = self.map.fds.get(dirfd)?;
        if let Some((base, epoch)) = f.dir_at()? {
            DIRS_SEEN.with(|d| d.borrow_mut().push(Seen::Fd(dirfd, epoch)));
            return Ok(crate::vfs::normalize(p, &base));
        }
        if f.metadata().is_ok_and(|md| md.file_type != crate::vfs::FileType::Directory) {
            return Err(Errno::ENOTDIR);
        }
        Ok(crate::vfs::normalize(p, &f.path))
    }
}

/// The memory map and what a mapping call needs beside it, none of it behind
/// the kernel lock: `mmap`, `munmap`,
/// `mprotect`, `brk`, `mremap`, `madvise` and `msync`, the same code for the
/// full call and [`free_call`]. What they note for the host (code mapped and
/// retired, pins, shared file mappings) is the map's own ([`MapLog`]).
///
/// [`MapLog`]: crate::mem::MapLog
#[derive(Clone, Copy)]
pub struct MapIo<'a> {
    pub mem: &'a crate::mem::Mm,
    /// For a file mapping's descriptor.
    pub fds: &'a FdTable,
    /// `Kernel::pins` and `Kernel::place`: constants once the process runs.
    pub pins: &'a std::collections::HashMap<String, u64>,
    pub place: Option<&'a Arc<dyn ObjectPlace>>,
    /// The call's pc: a mapping the dynamic loader asks for stays below
    /// 4 GiB, since it will hold code.
    pub pc: u64,
}

impl MapIo<'_> {
    /// The mapping call `n` (canonical), or `None` for any other.
    pub fn call(&self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<SysResult> {
        Some(match n {
            nr::BRK => {
                // Zeroed, because Linux zeroes it and libc counts on that.
                let (result, grew) = self.mem.lock().brk_grow(a[0]);
                if let Some((from, to)) = grew {
                    let _ = m.zero_dirty(from, (to - from) as usize);
                }
                Ok(result)
            }
            nr::MMAP => self.mmap(m, a),
            nr::MUNMAP => {
                // An alias's unmap: another mapping still holds the pages;
                // the last holder's unmaps all of them.
                let mut a = a;
                match self.mem.lock().log.shared_maps.unalias(a[0], crate::mem::page_align_up(a[1])) {
                    Unalias::Keep => return Some(Ok(0)),
                    Unalias::Whole(s, l) => (a[0], a[1]) = (s, l),
                    Unalias::Plain => {}
                }
                let flushed = self.flush_shared(m, a[0], a[1]);
                {
                    let mut mm = self.mem.lock();
                    let maps = &mut mm.log.shared_maps;
                    let gone: Vec<SharedMap> = maps.iter().filter(|s| s.start >= a[0] && s.start + s.len <= a[0] + a[1]).cloned().collect();
                    if flushed.is_ok() {
                        for s in &gone {
                            maps.remember(s);
                        }
                    }
                    maps.retain(|s| s.start + s.len <= a[0] || s.start >= a[0] + a[1]);
                }
                self.retire_code_why(a[0], a[1], "munmap", 0);
                flushed.and_then(|_| self.mem.lock().unmap(a[0], a[1])).map(|_| 0)
            }
            // MS_INVALIDATE (2): and what other processes wrote back since
            // read in ([`sync_shared_file`]).
            nr::MSYNC => self.flush_shared(m, a[0], a[1]).and_then(|_| if a[2] & 2 != 0 { self.refresh_shared(m, a[0], a[1]) } else { Ok(()) }).map(|_| 0),
            nr::MREMAP => self.mremap(m, a),
            nr::MPROTECT => self.mprotect(m, a),
            nr::MADVISE => self.madvise(m, a[0], a[1], a[2] as u32),
            _ => return None,
        })
    }

    /// `mprotect`.
    pub fn mprotect(&self, m: &GuestMem<'_>, a: [u64; 6]) -> SysResult {
        // Losing execute retires the translations as surely as
        // unmapping does: `ld.so` applies RELRO by dropping write,
        // but a JIT in the guest flips a page from executable back to
        // writable precisely so it can put different code there.
        // Making code writable is the other half (a patch, a
        // detour), which x86 needs no cache flush to see.
        if (a[2] as u32) & crate::mem::prot::EXEC == 0 || (a[2] as u32) & crate::mem::prot::WRITE != 0 {
            self.retire_code_why(a[0], a[1], "mprotect", a[2] as u32);
        }
        let mut mm = self.mem.lock();
        let r = mm.protect(a[0], a[1], (a[2] & 0x7) as u32).map(|_| 0);
        // Executable again (a loader filling an import table in
        // `.text`, a JIT done writing): what is there now is checked
        // against the function tier's units (`cgen::on_retire`
        // compares the pages; a page made non-executable for a while
        // is not gone: `cgen::in_pe_image`). A check, not a retire
        // (`MapLog::recheck_code`).
        if r.is_ok() && (a[2] as u32) & crate::mem::prot::EXEC != 0 && (a[2] as u32) & crate::mem::prot::WRITE == 0 {
            let (start, end) = (
                crate::mem::page_align_down(a[0]),
                crate::mem::page_align_up(a[0].saturating_add(a[1])),
            );
            if end > start {
                mm.log.recheck_code.push((start, end));
                crate::note_code_change();
                // Code read into anonymous memory (an image whose sections are
                // not page-aligned in its file) is named by its bytes, so the
                // same code at the same address is known as such in another
                // process and on another load.
                // Only when something keeps translations by that name.
                if crate::mem::content_names() {
                    mm.name_by_content(start, end, |lo, hi| {
                        m.read(lo, (hi - lo) as usize).ok().map(|b| content_id(b))
                    });
                }
            }
        }
        if r.is_ok() && (a[2] as u32) & crate::mem::prot::WRITE != 0 {
            mm.forget_code(crate::mem::page_align_down(a[0]), crate::mem::page_align_up(a[0].saturating_add(a[1])));
        }
        r
    }

    /// `mremap`. The map decides where; this moves the bytes.
    pub fn mremap(&self, m: &mut GuestMem<'_>, a: [u64; 6]) -> SysResult {
        const MREMAP_MAYMOVE: u64 = 1;
        let (addr, old_len, new_len, flags) = (a[0], a[1], a[2], a[3]);
        // The map's lock held through the copy: the old range is free in
        // the map once it has moved, and another thread's mapping there
        // must not land before its bytes are out.
        let mut mm = self.mem.lock();
        let (dst, moved) = mm.mremap(addr, old_len, new_len, flags & MREMAP_MAYMOVE != 0)?;
        if moved {
            let keep = old_len.min(new_len) as usize;
            let bytes = m.read(addr, keep)?.to_vec();
            m.write(dst, &bytes)?;
            // The grown tail at the new location is fresh anonymous memory,
            // which Linux exposes as zero -- but `dst` may be a reused hole
            // holding stale bytes, and only `keep` of them were overwritten.
            if new_len > old_len {
                m.zero_dirty(dst + keep as u64, (new_len - keep as u64) as usize)?;
            }
        } else if new_len > old_len {
            // Growing in place must expose zeroes, not whatever the guest
            // left behind the last time this range was mapped.
            m.zero_dirty(addr + old_len, (new_len - old_len) as usize)?;
        }
        Ok(dst)
    }

    /// The pinned base for a path a closure build named, if there is one.
    ///
    /// Exact spelling first, then the file's name alone: `gexc` records the
    /// path it walked in its sysroot while `ld.so` opens what `ld.so.cache`
    /// gives it, and on Debian those differ by a symlinked prefix. The
    /// name-only fallback is safe because a rootfs does not ship two libraries
    /// of one name that a single program loads both of; the exact match still
    /// wins.
    pub fn pin_for(&self, path: &str) -> Option<u64> {
        if self.pins.is_empty() {
            return None;
        }
        if let Some(&base) = self.pins.get(path) {
            return Some(base);
        }
        let want = path.rsplit('/').next()?;
        self.pins
            .iter()
            .find(|(k, _)| k.rsplit('/').next() == Some(want))
            .map(|(_, &v)| v)
    }

    /// `madvise`. Only `MADV_DONTNEED` does anything: on Linux it makes an
    /// anonymous range read as zero on next access, and glibc's malloc frees
    /// arena memory this way and then REUSES it trusting the zero. A no-op left
    /// the reused pages holding stale bytes, so glibc read garbage -- a
    /// deterministic corruption Qt hits (heavy glibc malloc/threads) and musl
    /// does not (it frees via `munmap`, which is re-zeroed on remap). Only
    /// anonymous VMAs are zeroed; a file mapping's `MADV_DONTNEED` reloads from
    /// the file, which the demand path already does, so it is left alone.
    pub fn madvise(&self, m: &mut GuestMem<'_>, addr: u64, len: u64, advice: u32) -> SysResult {
        const MADV_DONTNEED: u32 = 4;
        if advice == MADV_DONTNEED {
            let end = addr.saturating_add(len);
            let anon: Vec<(u64, u64)> = self
                .mem
                .lock()
                .vmas()
                .iter()
                .filter(|v| v.origin.is_none() && v.start < end && addr < v.end)
                .map(|v| (v.start.max(addr), v.end.min(end)))
                .collect();
            for (lo, hi) in anon {
                if hi > lo {
                    m.zero_dirty(lo, (hi - lo) as usize)?;
                }
            }
        }
        Ok(0)
    }

    pub fn mmap(&self, m: &mut GuestMem<'_>, a: [u64; 6]) -> SysResult {
        let (addr, len, prot_bits, flags, fd, offset) =
            (a[0], a[1], a[2] as u32, a[3], a[4] as i32, a[5]);
        // A fixed mapping over code replaces it, as an unmap would (Wine's
        // VirtualFree decommits by mapping over the range).
        if flags & mflags::FIXED != 0 {
            self.retire_code_why(addr, len, "mmap-fixed", prot_bits);
            // And a shared file mapping it replaces ends as an unmap ends
            // it: written back, and no longer handed to the next mapper.
            self.flush_shared(m, addr, len)?;
            self.mem.lock().log.shared_maps.retain(|s| s.start + s.len <= addr || s.start >= addr + len);
        }
        if flags & mflags::ANONYMOUS == 0 && fd >= 0 {
            // A private file mapping is an eager copy, which is exactly
            // correct and needs no page faults.
            let len_aligned = crate::mem::page_align_up(len);
            let file = self.fds.get(fd)?;
            let name = file.path.clone();
            if file.handle.maps_own_memory() {
                let start = self.mem.lock().mmap(addr, len, prot_bits, flags | mflags::ANONYMOUS, &name)?;
                file.handle.mapped(start, len);
                return Ok(start);
            }
            // A loader that maps an object's first segment straight from the
            // file, spanning the whole object (glibc 2.35 and later), rather
            // than reserving anonymous memory first: that mapping takes the
            // pin armed when the object was opened, the same way.
            let mut addr = addr;
            let mut pinned = None;
            if addr == 0 && flags & crate::mem::flags::FIXED == 0 {
                let mut placed = None;
                {
                    let mut mm = self.mem.lock();
                    if let Some((path, base)) = mm.log.armed_pin.take() {
                        if path.rsplit('/').next() == name.rsplit('/').next() {
                            mm.log.armed_place = None;
                            addr = base;
                            pinned = Some((path, base));
                        } else {
                            mm.log.armed_pin = Some((path, base));
                        }
                    }
                    // Or the placement armed at its open, as for a
                    // reservation (below): every process is given the
                    // address the first one got.
                    if pinned.is_none() && mm.log.armed_place.as_deref() == Some(name.as_str()) {
                        placed = mm.log.armed_place.take();
                    }
                }
                if let Some(path) = placed {
                    if let Some(base) = self.place.and_then(|p| p.place(&path, len_aligned)) {
                        addr = base;
                    }
                }
            }
            // A second `MAP_SHARED` of the same open file must see the first
            // one's writes, so hand back the same pages -- a Wayland client
            // maps a buffer and sends the descriptor, and the next mapper must
            // see the same pixels, not a zeroed copy. Counted (`holders`): an
            // alias's `munmap` leaves the pages to the mappings still holding
            // them.
            if flags & crate::mem::flags::SHARED != 0 {
                // (or any part of one: a window into a mapping another
                // view holds whole, as win32u maps the session's blocks).
                // Another `open` of the same file too, as Linux's page cache
                // makes it one set of pages (two sqlite connections to one
                // database map its `-shm` index each). Counted, so the
                // first `munmap` leaves the other's pages.
                let _ = file.ino(); // asked outside the lock (file I/O)
                let mut mm = self.mem.lock();
                if let Some(i) = mm.log.shared_maps.iter().position(|s| {
                    s.file.same_file(&file) && s.offset <= offset && offset + len_aligned <= s.offset + s.len
                }) {
                    let existing = &mm.log.shared_maps[i];
                    let at = existing.start + (offset - existing.offset);
                    // Not for a fixed mapping elsewhere: the caller uses the
                    // address it named (Wine maps every view into its own
                    // reserved range), so that one is a mapping of its own,
                    // read from the file the others are written back to.
                    if flags & crate::mem::flags::FIXED == 0 || addr == at {
                        mm.log.shared_maps.alias(i, at, len_aligned);
                        return Ok(at);
                    }
                }
            }
            // The path and offset travel with the mapping. Two processes
            // that map the same library at the same address then agree that
            // the code is the same, which is what lets them share the
            // dispatch slots it was translated into.
            let start = self.mem.lock().mmap_from(
                addr,
                len_aligned,
                prot_bits,
                flags,
                &name,
                Some((name.clone(), offset)),
            )?;
            if let Some((path, base)) = pinned {
                // Said as it happens: the end-of-run report is lost when the
                // process is killed (a run ended by its timeout).
                if base != start {
                    eprintln!("[pin] {path} WANTED {base:#x} GOT {start:#x}");
                }
                self.mem.lock().log.pin_landed.push((path, base, start));
            }
            // Straight into the mapping, as `read` does: through a host
            // buffer it was a second copy of the file, and the buffer's own
            // first touch, on every private file mapping (apt-get maps its
            // two 44 MB package caches this way on every run).
            // The file as this process's shared mappings of it hold it: their
            // writes reach the file only at `msync` and `munmap`, and a new
            // mapping (a private copy, or a shared one only partly inside
            // theirs) is read from the file. Written back first.
            let behind: Vec<(u64, u64)> = self
                .mem
                .lock()
                .log
                .shared_maps
                .iter()
                .filter(|s| s.file.same_file(&file) && s.offset < offset + len_aligned && offset < s.offset + s.len)
                .map(|s| (s.start, s.len))
                .collect();
            for (at, n) in behind {
                self.flush_shared(m, at, n)?;
            }
            // The file's change count either side of the read: the same
            // nonzero count is the contents it was before (a change is
            // counted before it can be read), and the hashes kept of them
            // stand (`SharedMaps::recall`).
            let shared = flags & crate::mem::flags::SHARED != 0;
            let before = if shared && len_aligned >= RECALL_MIN { file.handle.changes() } else { 0 };
            let n = file.pread(offset, m.read_mut(start, len as usize)?)? as usize;
            m.zero_dirty(start + n as u64, len_aligned as usize - n)?;
            // A shared mapping is written back at `msync` and `munmap` (where
            // a program that never calls `msync` expects its changes to land).
            // An executable file mapping is a shared library arriving: tell the
            // host so the static recompiler gets the whole object at once.
            if prot_bits & crate::mem::prot::EXEC != 0 {
                crate::note_code_change();
                self.mem.lock().log.new_code.push(CodeMap {
                    start,
                    end: start + len_aligned,
                    path: name.clone(),
                    offset,
                });
            }
            if shared {
                let path_of_map = file.path.clone();
                let changes = if before != 0 && file.handle.changes() == before { before } else { 0 };
                let kept = if changes != 0 { self.mem.lock().log.shared_maps.recall(&file, offset, len_aligned, changes) } else { None };
                let pages = match kept {
                    Some(p) => p,
                    None => page_hashes(m.read(start, len_aligned as usize)?),
                };
                self.mem.lock().log.shared_maps.push(SharedMap {
                    start,
                    len: len_aligned,
                    file,
                    offset,
                    holders: vec![(start, len_aligned)],
                    pages,
                    changes: Arc::new(std::sync::atomic::AtomicU64::new(changes)),
                    seen: Arc::new(std::sync::atomic::AtomicU64::new(crate::locks::written(&path_of_map))),
                });
            }
            return Ok(start);
        }
        // The anonymous PROT_NONE reservation an ELF loader makes before it
        // maps an object: hand it the pinned base and every `MAP_FIXED` that
        // follows lands where the closure build assumed. Passed as a hint, so
        // a collision is ignored and recorded in `pin_landed`, not forced.
        let mut addr = addr;
        let mut pinned = None;
        if addr == 0 && prot_bits == 0 && flags & mflags::ANONYMOUS != 0 {
            let (pin, place) = {
                let mut mm = self.mem.lock();
                match mm.log.armed_pin.take() {
                    Some(pin) => {
                        mm.log.armed_place = None;
                        (Some(pin), None)
                    }
                    None => (None, mm.log.armed_place.take()),
                }
            };
            if let Some((path, base)) = pin {
                addr = base;
                pinned = Some((path, base));
            } else if let Some(path) = place {
                // The reservation is the first thing that says how long the
                // object is, which is why the question is asked here and not
                // at the `openat` that armed it.
                if let Some(base) = self
                    .place
                    .and_then(|p| p.place(&path, crate::mem::page_align_up(len)))
                {
                    addr = base;
                }
            }
        }
        // The dynamic loader's reservation for an object (glibc maps an
        // over-aligned library into anonymous PROT_NONE memory) will hold
        // code, so in a 64-bit memory it stays below 4 GiB with the code; the
        // loader runs at INTERP_BASE.
        // What was known to be zero before the mapping (which forgets it).
        let clean_before = self.mem.lock().clean_ranges().to_vec();
        // Only what this mapping claims (another caller's claim may since
        // have been written).
        let _ = self.mem.lock().take_fresh();
        let from_loader = (gex_abi::INTERP_BASE..gex_abi::STACK_TOP - gex_abi::STACK_SIZE).contains(&self.pc);
        let start = if from_loader || (cfg!(feature = "x86") && flags & mflags::MAP_32BIT != 0) {
            self.mem.lock().mmap_low(addr, len, prot_bits, flags & !mflags::MAP_32BIT, "")?
        } else {
            self.mem.lock().mmap(addr, len, prot_bits, flags, "")?
        };
        if let Some((path, base)) = pinned {
            // Said as it happens: the end-of-run report is lost when the
            // process is killed (a run ended by its timeout).
            if base != start {
                eprintln!("[pin] {path} WANTED {base:#x} GOT {start:#x}");
            }
            self.mem.lock().log.pin_landed.push((path, base, start));
        }
        // Anonymous memory is defined to read as zero.
        self.mem.zero_new(m, start, len, prot_bits, clean_before)?;
        Ok(start)
    }


    /// Write every shared mapping back (the changed pages): at `exit` and
    /// `execve`, where Linux's page cache keeps a mapping's stores for the
    /// file though nobody called `msync` or `munmap`.
    pub fn flush_all_shared(&self, m: &GuestMem<'_>) {
        if self.mem.lock().log.shared_maps.is_empty() {
            return;
        }
        let _ = self.flush_shared(m, 0, u64::MAX);
    }

    /// `msync(MS_INVALIDATE)`: the shared mappings over `[addr, addr + len)`
    /// given what their files hold now where another process wrote back.
    pub fn refresh_shared(&self, m: &mut GuestMem<'_>, addr: u64, len: u64) -> Result<(), Errno> {
        let (lo, hi) = (addr, addr.saturating_add(len));
        let maps: Vec<SharedMap> = {
            let mm = self.mem.lock();
            mm.log.shared_maps.iter().filter(|map| map.start.max(lo) < (map.start + map.len).min(hi)).cloned().collect()
        };
        for map in maps {
            refresh(m, &map, true, None, false)?;
        }
        Ok(())
    }

    /// Write a shared mapping's bytes back to the file behind it.
    pub fn flush_shared(&self, m: &GuestMem<'_>, addr: u64, len: u64) -> Result<(), Errno> {
        let (lo, hi) = (addr, addr.saturating_add(len));
        // Taken out of the map's lock: the writes are file I/O.
        let maps: Vec<SharedMap> = {
            let mm = self.mem.lock();
            if mm.log.shared_maps.is_empty() {
                return Ok(());
            }
            mm.log.shared_maps.iter().filter(|map| map.start.max(lo) < (map.start + map.len).min(hi)).cloned().collect()
        };
        for map in maps {
            write_back(m, &map, lo, hi)?;
        }
        Ok(())
    }

    /// Note that `[addr, addr + len)` no longer holds the code it did.
    ///
    /// Only ranges that overlap an executable mapping are worth reporting:
    /// unmapping the heap or a data file happens constantly and has no
    /// translations behind it.
    pub fn retire_code(&self, addr: u64, len: u64) {
        self.retire_code_why(addr, len, "retire", 0)
    }

    /// [`Self::retire_code`], saying which call it was for the host's
    /// retire-storm diagnostic ([`crate::note_retire_why`]).
    pub fn retire_code_why(&self, addr: u64, len: u64, why: &'static str, prot: u32) {
        let (start, end) = (
            crate::mem::page_align_down(addr),
            crate::mem::page_align_up(addr.saturating_add(len)),
        );
        if end <= start {
            return;
        }
        let mut mm = self.mem.lock();
        let executable = mm.vmas().iter().any(|v| v.prot & crate::mem::prot::EXEC != 0 && v.start < end && v.end > start);
        if executable {
            mm.log.retired_code.push((start, end));
            crate::note_retire_why(crate::RetireWhy { start, end, why, prot, pc: self.pc });
            crate::note_code_change();
        }
    }
}

/// The socket calls: `socket`, `socketpair`, `bind`, `listen`,
/// `accept`/`accept4`, `connect`, `sendto`, `recvfrom`, `sendmmsg`,
/// `recvmmsg`, the names (`getsockname`, `getpeername`), the options and
/// `shutdown` (canonical numbers). `sendmsg` and `recvmsg` are [`FdIo`]'s.
pub fn is_socket_call(n: u64) -> bool {
    matches!(
        n,
        nr::SOCKET
            | nr::SOCKETPAIR
            | nr::BIND
            | nr::LISTEN
            | nr::ACCEPT
            | nr::ACCEPT4
            | nr::CONNECT
            | nr::GETSOCKNAME
            | nr::GETPEERNAME
            | nr::SENDTO
            | nr::RECVFROM
            | nr::SETSOCKOPT
            | nr::GETSOCKOPT
            | nr::SHUTDOWN
            | nr::SENDMMSG
            | nr::RECVMMSG
    )
}

/// What the socket calls need of their process beside its descriptor table
/// ([`FdIo`]), none of it behind the kernel lock: the network and the name
/// tables are shared and lock
/// themselves, the global kernel's sockets are another module's.
#[derive(Clone, Copy)]
pub struct NetIo<'a> {
    /// The network stack (`Kernel::net`); none is loopback only.
    pub net: Option<&'a Arc<dyn net::NetStack>>,
    pub resolver: &'a Arc<net::Resolver>,
    pub ports: &'a Arc<net::Ports>,
    /// The global kernel's unix sockets, when there is one.
    pub unix_remote: Option<unix::Remote>,
    /// The system's bound names, and this process's host's own.
    pub socket_names: &'a Arc<unix::SocketNames>,
    pub host_names: &'a Arc<unix::SocketNames>,
}

const SOCK_NONBLOCK: u64 = 0o4000;
const SOCK_CLOEXEC: u64 = 0o2000000;

impl NetIo<'_> {
    /// A socket call ([`is_socket_call`]), with the calling thread's working
    /// directory `cwd` for a unix socket's relative name.
    pub fn call(&self, io: &FdIo<'_>, m: &mut GuestMem<'_>, n: u64, a: [u64; 6], cwd: &str) -> SysResult {
        let fd = a[0] as i32;
        match n {
            nr::SOCKET => self.socket(io, a[0], a[1], a[2]),
            // A unix socket and an internet socket are different concrete
            // types behind the same descriptor, so which `connect` this is
            // depends on what the descriptor turned out to be.
            nr::CONNECT => match self.unix_connect(io, m, fd, a[1], a[2], cwd) {
                Err(Errno::ENOTSOCK) => inet_connect(io, m, fd, a[1], a[2]),
                other => other,
            },
            nr::SENDTO => sendto(io, m, fd, a[1], a[2], a[4], a[5], cwd),
            nr::RECVFROM => recvfrom(io, m, fd, a[1], a[2], a[3], a[4], a[5]),
            nr::SENDMMSG => sendmmsg(io, m, fd, a[1], a[2]),
            nr::RECVMMSG => recvmmsg(io, m, fd, a[1], a[2], a[3]),
            nr::GETPEERNAME if io.dgram_of(fd).is_ok() => match io.dgram_of(fd)?.peer_name()? {
                Some(p) => Kernel::write_sockaddr_un(m, a[1], a[2], &p).map(|_| 0),
                None => Kernel::write_sockaddr_unnamed(m, a[1], a[2]),
            },
            // A unix stream socket: the name it connected to (an accepted
            // end's peer is unnamed: the family alone).
            nr::GETPEERNAME if matches!(socket_family_type(io, fd), Ok((1, _))) => {
                let peer = match io.unix_of(fd) {
                    Ok(sock) => sock.peer_name()?,
                    Err(_) => {
                        let r = io.remote_unix_of(fd)?;
                        if r.listening {
                            return Err(Errno::ENOTCONN);
                        }
                        r.peer.clone()
                    }
                };
                match peer {
                    Some(p) => Kernel::write_sockaddr_un(m, a[1], a[2], &p).map(|_| 0),
                    None => Kernel::write_sockaddr_unnamed(m, a[1], a[2]),
                }
            }
            nr::GETPEERNAME => {
                let (ip, port) = io.socket_of(fd)?.peer().ok_or(Errno::ENOTCONN)?;
                io.write_sockaddr(m, ip, port, a[1], a[2])
            }
            // An unbound, unconnected socket is at 0.0.0.0; a connected one
            // not bound to an address is on the interface its peer is
            // reached by: lo for a loopback peer, else eth0 (crate::netlink,
            // the interfaces the route socket and SIOCGIF* list).
            //
            // A unix socket is AF_UNIX and unnamed: libuv's
            // `uv_guess_handle` decides from the family whether a socket on
            // stdout is a pipe or TCP, and node made the stdout of every
            // child it spawned a black hole when told AF_INET.
            nr::GETSOCKNAME => match socket_family_type(io, fd) {
                // A bound datagram socket has its name.
                Ok((1, 2)) if io.dgram_of(fd)?.name().is_some() => {
                    Kernel::write_sockaddr_un(m, a[1], a[2], &io.dgram_of(fd)?.name().unwrap()).map(|_| 0)
                }
                // A unix stream socket: its name (bound, listening, or its
                // listener's when accepted), else unnamed. A listener in the
                // global kernel has the name it was bound to.
                Ok((1, _)) => {
                    let name = match io.unix_of(fd) {
                        Ok(sock) => sock.local_name(),
                        Err(_) => io.remote_unix_of(fd).ok().filter(|r| r.listening).and_then(|r| r.peer.clone()),
                    };
                    match name {
                        Some(p) => Kernel::write_sockaddr_un(m, a[1], a[2], &p).map(|_| 0),
                        None => Kernel::write_sockaddr_unnamed(m, a[1], a[2]),
                    }
                }
                _ => {
                    let (ip, port) = io
                        .socket_of(fd)
                        .map(|s| match (s.local(), s.peer()) {
                            ((0, port), Some((peer, _))) => (crate::netlink::local_for(peer), port),
                            (local, _) => local,
                        })
                        .unwrap_or((0, 0));
                    io.write_sockaddr(m, ip, port, a[1], a[2])
                }
            },
            // `SO_PASSCRED` on a unix socket (`SCM_CREDENTIALS` on each
            // `recvmsg`, [`FdIo::write_scm_creds`]). Nothing else here has an
            // option worth setting -- no timeouts, no buffer sizes that mean
            // anything -- and refusing makes libraries treat the socket as
            // broken: agreed to without effect, on a descriptor that exists.
            nr::SETSOCKOPT => {
                const SOL_SOCKET: u64 = 1;
                const SO_PASSCRED: u64 = 16;
                io.fds.get(fd)?;
                if a[1] == SOL_SOCKET && a[2] == SO_PASSCRED && a[3] != 0 && a[4] >= 4 {
                    let on = m.read_u32(a[3])? != 0;
                    if let Ok(s) = io.unix_of(fd) {
                        s.set_passcred(on);
                    } else if let Ok(s) = io.remote_unix_of(fd) {
                        s.set_passcred(on);
                    }
                }
                Ok(0)
            }
            nr::GETSOCKOPT => getsockopt(io, m, fd, a[1], a[2], a[3], a[4]),
            // A unix socket's directions close as asked, local or in the
            // global kernel; the proxied network's as it can; anything else
            // just agrees.
            nr::SHUTDOWN => match io.unix_of(fd) {
                Ok(sock) => sock.shutdown_how(a[1]),
                // A global-kernel one too (a socketpair's ends in two
                // Workers): SHUT_WR is how libuv ends a child's stdin.
                Err(_) => match io.remote_unix_of(fd) {
                    Ok(sock) => sock.shutdown_how(a[1]),
                    Err(_) => match io.socket_of(fd) {
                        Ok(s) => s.shutdown(a[1]),
                        Err(_) => io.fds.get(fd).map(|_| 0),
                    },
                },
            },
            // A local address means nothing when there is one interface and
            // it is a proxy, but a resolver binds its socket before it sends
            // and gives up if that fails.
            nr::BIND => match self.unix_bind(io, m, fd, a[1], a[2], cwd) {
                Err(Errno::ENOTSOCK) => match io.socket_of(fd) {
                    // An internet socket remembers where it is (`listen`,
                    // `getsockname`).
                    Ok(sock) => m
                        .read(a[1], (a[2] as usize).min(128))
                        .and_then(|b| net::parse_sockaddr(b))
                        .and_then(|(ip, port)| sock.bind(ip, port).map(|_| 0)),
                    Err(_) => io.fds.get(fd).map(|_| 0),
                },
                other => other,
            },
            // A socket in the global kernel was bound and listening in one
            // call; there is nothing left for this to do but agree.
            nr::LISTEN => match io.remote_unix_of(fd) {
                Ok(_) => Ok(0),
                Err(_) => match io.unix_of(fd) {
                    Ok(s) => {
                        s.set_cred(io.cred());
                        s.listen()
                    }
                    // A TCP listener, for guests on this machine.
                    Err(Errno::ENOTSOCK) => io.socket_of(fd).and_then(|s| s.listen()).map(|_| 0),
                    Err(e) => Err(e),
                },
            },
            nr::ACCEPT => accept(io, m, fd, a[1], a[2], 0),
            nr::ACCEPT4 => accept(io, m, fd, a[1], a[2], a[3]),
            nr::SOCKETPAIR => self.socketpair(io, m, a[0], a[1], a[3]),
            _ => Err(Errno::ENOSYS),
        }
    }

    /// Where a socket call made without the kernel lock waits, as the full
    /// call does: a blocking `connect` until the connection is up or not
    /// (started here; the call made afterwards only reports on it), and a
    /// blocking `accept` or `recvfrom` with nothing there until there is,
    /// then the call made again (`BlockOp::AcceptWait`).
    pub fn park(&self, io: &FdIo<'_>, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        let file = io.fds.get(a[0] as i32).ok()?;
        match n {
            nr::CONNECT => {
                let bytes = m.read(a[1], (a[2] as usize).min(128)).ok()?.to_vec();
                let (ip, port) = net::parse_sockaddr(&bytes).ok()?;
                let socket = io.socket_of(a[0] as i32).ok()?;
                let started = socket.connect(ip, port);
                let waits = file.would_block() && started == Err(Errno::EINPROGRESS);
                waits.then_some(BlockOp::Connect { socket })
            }
            nr::ACCEPT | nr::ACCEPT4 | nr::RECVFROM => {
                if n == nr::RECVFROM && a[3] & Kernel::MSG_DONTWAIT != 0 {
                    return None;
                }
                let socket: Arc<dyn crate::vfs::FileHandle> = Arc::clone(&file.handle);
                if !file.would_block() || socket.poll_readable() {
                    return None;
                }
                Some(BlockOp::AcceptWait { socket })
            }
            _ => None,
        }
    }

    pub fn socket(&self, io: &FdIo<'_>, domain: u64, ty: u64, protocol: u64) -> SysResult {
        const AF_UNIX: u64 = 1;
        const AF_INET: u64 = 2;
        const SOCK_STREAM: u64 = 1;
        const SOCK_DGRAM: u64 = 2;
        // A unix socket needs no network behind it: both ends are in this
        // machine, and one of them may be the host rather than a guest.
        if domain == AF_UNIX {
            let kind = ty & !(SOCK_NONBLOCK | SOCK_CLOEXEC);
            if kind == SOCK_DGRAM {
                let sock = unix::UnixDgram::new(self.socket_names);
                let nonblock = ty & SOCK_NONBLOCK != 0;
                sock.set_nonblocking(nonblock);
                let flags = if nonblock { oflags::O_NONBLOCK } else { 0 };
                let file = OpenFile::new(sock, "socket".to_string(), flags, true, true);
                return io.fds.insert(file, ty & SOCK_CLOEXEC != 0);
            }
            if kind != SOCK_STREAM && kind != 5 {
                // SOCK_SEQPACKET is 5 and behaves closely enough to a stream
                // for what uses it here.
                return Err(Errno::EPROTONOSUPPORT);
            }
            let sock = Arc::new(unix::UnixSocket::new(true));
            let nonblock = ty & SOCK_NONBLOCK != 0;
            sock.set_nonblocking(nonblock);
            let flags = if nonblock { oflags::O_NONBLOCK } else { 0 };
            return install_unix(io, sock, flags, ty & SOCK_CLOEXEC != 0);
        }
        // AF_NETLINK, which libudev's device monitor opens (SDL's joystick
        // and haptic start-up through it; a failure there makes SDL_Init undo
        // its video subsystem too, and the program sees no display). There
        // are no devices to announce, so the socket binds, takes its options
        // and never has a message: an empty eventfd behind the descriptor.
        // NETLINK_ROUTE answers its interface and address dumps
        // (crate::netlink): musl's if_nameindex and getifaddrs, and Wine's
        // nsi through them, and glibc's getaddrinfo (AI_ADDRCONFIG), which
        // an eventfd here once answered with the 8-byte counter its own send
        // wrote, and aborted on (apt's http method). No other family.
        const AF_NETLINK: u64 = 16;
        const NETLINK_ROUTE: u64 = 0;
        const NETLINK_KOBJECT_UEVENT: u64 = 15;
        if domain == AF_NETLINK && protocol == NETLINK_ROUTE {
            let handle: Arc<dyn crate::vfs::FileHandle> = Arc::new(crate::netlink::NetlinkRoute::new());
            let flags = if ty & SOCK_NONBLOCK != 0 { oflags::O_NONBLOCK } else { 0 };
            let file = OpenFile::new(handle, "socket:[netlink]".to_string(), flags, true, true);
            return io.fds.insert(file, ty & SOCK_CLOEXEC != 0);
        }
        if domain == AF_NETLINK && protocol != NETLINK_KOBJECT_UEVENT {
            return Err(Errno::EAFNOSUPPORT);
        }
        if domain == AF_NETLINK {
            let handle: Arc<dyn crate::vfs::FileHandle> = Arc::new(crate::eventfd::EventFd::new(0, false));
            let flags = if ty & SOCK_NONBLOCK != 0 { oflags::O_NONBLOCK } else { 0 };
            let file = OpenFile::new(handle, "socket:[netlink]".to_string(), flags, true, true);
            return io.fds.insert(file, ty & SOCK_CLOEXEC != 0);
        }
        if domain != AF_INET {
            return Err(Errno::EAFNOSUPPORT);
        }
        // Without a network stack (the browser with no relay), loopback still
        // works: Wine's in-process server and Winsock use it.
        let net = self.net.cloned().unwrap_or_else(|| Arc::new(net::NoNet) as Arc<dyn net::NetStack>);
        let kind = match ty & !(SOCK_NONBLOCK | SOCK_CLOEXEC) {
            SOCK_STREAM => net::Kind::Tcp,
            SOCK_DGRAM => net::Kind::Udp,
            _ => return Err(Errno::EPROTONOSUPPORT),
        };
        let sock = Arc::new(net::Socket::new(kind, net, Arc::clone(self.resolver), Arc::clone(self.ports), self.unix_remote));
        sock.set_self();
        sock.set_nonblocking(ty & SOCK_NONBLOCK != 0);
        let flags = if ty & SOCK_NONBLOCK != 0 { oflags::O_NONBLOCK } else { 0 };
        let handle: Arc<dyn crate::vfs::FileHandle> = sock.clone();
        let file = OpenFile::new(handle, "socket".to_string(), flags, true, true);
        io.fds.insert(file, ty & SOCK_CLOEXEC != 0)
    }

    pub fn socketpair(&self, io: &FdIo<'_>, m: &mut GuestMem<'_>, domain: u64, ty: u64, out: u64) -> SysResult {
        const AF_UNIX: u64 = 1;
        if domain != AF_UNIX {
            return Err(Errno::EAFNOSUPPORT);
        }
        let nonblock = ty & SOCK_NONBLOCK != 0;
        // A stream pair in the global kernel when there is one: its ends are
        // often for two processes (Wine makes one per new process, keeps an
        // end and hands the other to the child), and with a Worker per
        // process a local pair cannot cross a fork.
        const SOCK_STREAM: u64 = 1;
        if let (Some(r), SOCK_STREAM) = (self.unix_remote, ty & 0xf) {
            let mut ids = [0u8; 8];
            let me = io.cred();
            let rc = (r.pair)(ids.as_mut_ptr() as crate::PAddr, me.pid, me.uid, me.gid);
            if rc < 0 {
                return Err(Errno(-rc));
            }
            let flags = if nonblock { oflags::O_NONBLOCK } else { 0 };
            let cloexec = ty & SOCK_CLOEXEC != 0;
            let mut fds = [0i32; 2];
            for (i, fd) in fds.iter_mut().enumerate() {
                let id = i32::from_le_bytes(ids[i * 4..i * 4 + 4].try_into().unwrap());
                let sock = Arc::new(unix::RemoteUnix::new(id, false, None, r));
                sock.set_nonblocking(nonblock);
                *fd = insert_remote_unix(io, sock, flags, cloexec)?;
            }
            m.write_u32(out, fds[0] as u32)?;
            m.write_u32(out + 4, fds[1] as u32)?;
            return Ok(0);
        }
        // A datagram pair: this kernel's (in a page, the process's own
        // Worker; see unix::UnixDgram).
        const SOCK_DGRAM: u64 = 2;
        if ty & 0xf == SOCK_DGRAM {
            let (a, b) = unix::UnixDgram::pair(self.socket_names);
            let flags = if nonblock { oflags::O_NONBLOCK } else { 0 };
            let cloexec = ty & SOCK_CLOEXEC != 0;
            let mut fds = [0i32; 2];
            for (fd, s) in fds.iter_mut().zip([a, b]) {
                s.set_nonblocking(nonblock);
                *fd = io.fds.insert(OpenFile::new(s, "socket".to_string(), flags, true, true), cloexec)? as i32;
            }
            m.write_u32(out, fds[0] as u32)?;
            m.write_u32(out + 4, fds[1] as u32)?;
            return Ok(0);
        }
        let (a, b) = unix::UnixSocket::pair();
        unix::UnixSocket::set_pair_cred(&a, &b, io.cred());
        a.set_nonblocking(nonblock);
        b.set_nonblocking(nonblock);
        let flags = if nonblock { oflags::O_NONBLOCK } else { 0 };
        let cloexec = ty & SOCK_CLOEXEC != 0;
        let fa = install_unix(io, a, flags, cloexec)?;
        let fb = install_unix(io, b, flags, cloexec)?;
        m.write_u32(out, fa as u32)?;
        m.write_u32(out + 4, fb as u32)?;
        Ok(0)
    }

    pub fn unix_bind(&self, io: &FdIo<'_>, m: &GuestMem<'_>, fd: i32, addr: u64, len: u64, cwd: &str) -> SysResult {
        if let Ok(d) = io.dgram_of(fd) {
            let path = Kernel::sockaddr_un(m, addr, len)?;
            let path = if path.starts_with('\0') { path } else { crate::vfs::normalize(&path, cwd) };
            // Named in the table every process shares as the namespace the
            // process started with names it (a `chroot`'s root in front).
            let name = io.ns.outer_path(&path).into_owned();
            d.bind(&name)?;
            if !path.starts_with('\0') {
                let _ = io.ns.mksocket(&path);
            }
            return Ok(0);
        }
        let sock = io.unix_of(fd)?;
        let path = Kernel::sockaddr_un(m, addr, len)?;
        // A filesystem socket resolves against the guest's root the way any
        // other path does; an abstract one has no filesystem existence.
        let path = if path.starts_with('\0') { path } else { crate::vfs::normalize(&path, cwd) };
        // Named in the table every process shares as the namespace the
        // process started with names it (a `chroot`'s root in front).
        let name = io.ns.outer_path(&path).into_owned();
        // In the global kernel when there is one, because the client that
        // will connect is another process -- and with a Worker per process
        // that is another memory, where a name bound here means nothing.
        //
        // Bound and listening in one call: what a guest does between `bind`
        // and `listen` is local to itself, and what the far end can see is
        // only whether the name answers.
        if let Some(r) = self.unix_remote {
            let me = io.cred();
            let id = (r.bind)(name.as_ptr() as crate::PAddr, name.len() as u32, me.pid, me.uid, me.gid);
            if id < 0 {
                return Err(Errno(-id));
            }
            let flags = io.fds.get(fd)?.flags();
            let cloexec = io.fds.cloexec(fd).unwrap_or(false);
            let remote = Arc::new(unix::RemoteUnix::new(id, true, Some(path.clone()), r));
            install_remote_unix(io, fd, remote, flags, cloexec)?;
            if !path.starts_with('\0') {
                let _ = io.ns.mksocket(&path);
            }
            return Ok(0);
        }
        sock.bind(&name)?;
        if let Err(e) = self.socket_names.bind(&name, sock.clone()) {
            sock.unbind_failed(&name);
            return Err(e);
        }
        // And a node in the filesystem, so the name can be seen as well as
        // connected to. Xvfb binds /tmp/.X11-unix/X0 and every X client waits
        // for that path to appear before trying it; without the node the
        // server was listening and nothing could tell.
        //
        // A filesystem that will not take one is not an error here: the
        // socket is bound and reachable either way, which is the part that
        // matters.
        if !path.starts_with('\0') {
            let _ = io.ns.mksocket(&path);
        }
        Ok(0)
    }

    pub fn unix_connect(&self, io: &FdIo<'_>, m: &GuestMem<'_>, fd: i32, addr: u64, len: u64, cwd: &str) -> SysResult {
        if let Ok(d) = io.dgram_of(fd) {
            let path = Kernel::sockaddr_un(m, addr, len)?;
            let path = if path.starts_with('\0') { path } else { crate::vfs::normalize(&path, cwd) };
            // Named in the table every process shares as the namespace the
            // process started with names it (a `chroot`'s root in front).
            let name = io.ns.outer_path(&path).into_owned();
            return d.connect(&name);
        }
        let sock = io.unix_of(fd)?;
        let path = Kernel::sockaddr_un(m, addr, len)?;
        let path = if path.starts_with('\0') { path } else { crate::vfs::normalize(&path, cwd) };
        // Named in the table every process shares as the namespace the
        // process started with names it (a `chroot`'s root in front).
        let name = io.ns.outer_path(&path).into_owned();
        // The global kernel first, because that is where a name another
        // process bound will be. A host-bound name -- the virtio-gpu socket --
        // is in *this* process's table and never in the kernel's, so the
        // local path is still tried when the kernel does not know the name.
        if let Some(r) = self.unix_remote {
            let me = io.cred();
            let id = (r.connect)(name.as_ptr() as crate::PAddr, name.len() as u32, me.pid, me.uid, me.gid);
            if id >= 0 {
                let flags = io.fds.get(fd)?.flags();
                let cloexec = io.fds.cloexec(fd).unwrap_or(false);
                let remote = Arc::new(unix::RemoteUnix::new(id, false, Some(path.clone()), r));
                install_remote_unix(io, fd, remote, flags, cloexec)?;
                return Ok(0);
            }
        }
        // This process's own host first -- its compositor, its GPU device --
        // then the system-wide table, which is where a guest server (an X
        // server's socket) lives.
        sock.set_cred(io.cred());
        if let Ok(r) = self.host_names.connect(&sock, &name) {
            return Ok(r);
        }
        self.socket_names.connect(&sock, &name)
    }
}

/// Put a unix socket in the descriptor table.
fn install_unix(io: &FdIo<'_>, sock: Arc<unix::UnixSocket>, flags: u64, cloexec: bool) -> SysResult {
    let file = OpenFile::new(sock, "socket".to_string(), flags, true, true);
    io.fds.insert(file, cloexec)
}

/// Put a socket that lives in the global kernel at `fd`, replacing whatever
/// was there.
///
/// Replacing, because a unix socket becomes remote only at `bind` or
/// `connect` -- `socket()` has no name to decide by, so the descriptor
/// starts as an ordinary local socket and is swapped when its name is
/// known.
fn install_remote_unix(io: &FdIo<'_>, fd: i32, sock: Arc<unix::RemoteUnix>, flags: u64, cloexec: bool) -> Result<(), Errno> {
    // An option set on the local socket it replaces.
    if io.unix_of(fd).is_ok_and(|s| s.passcred()) {
        sock.set_passcred(true);
    }
    let handle: Arc<dyn crate::vfs::FileHandle> = sock;
    let file = OpenFile::new(handle, "socket".to_string(), flags, true, true);
    io.fds.install(fd, file, cloexec)?;
    Ok(())
}

/// Put a socket that lives in the global kernel at the lowest free
/// descriptor.
fn insert_remote_unix(io: &FdIo<'_>, sock: Arc<unix::RemoteUnix>, flags: u64, cloexec: bool) -> Result<i32, Errno> {
    let handle: Arc<dyn crate::vfs::FileHandle> = sock;
    let file = OpenFile::new(handle, "socket".to_string(), flags, true, true);
    Ok(io.fds.insert(file, cloexec)? as i32)
}

fn inet_connect(io: &FdIo<'_>, m: &GuestMem<'_>, fd: i32, addr: u64, len: u64) -> SysResult {
    let bytes = m.read(addr, (len as usize).min(128))?.to_vec();
    let (ip, port) = net::parse_sockaddr(&bytes)?;
    io.socket_of(fd)?.connect(ip, port).map(|_| 0)
}

/// `accept`: a unix listener's, else a TCP listener's (`net::Socket`).
fn accept(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, addr: u64, len_out: u64, flags: u64) -> SysResult {
    match unix_accept(io, m, fd, addr, len_out, flags) {
        Err(Errno::ENOTSOCK) => inet_accept(io, m, fd, addr, len_out, flags),
        other => other,
    }
}

fn inet_accept(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, addr: u64, len_out: u64, flags: u64) -> SysResult {
    let listener = io.socket_of(fd)?;
    let sock = listener.accept()?;
    let (ip, port) = sock.peer().unwrap_or((0, 0));
    let nonblock = flags & SOCK_NONBLOCK != 0;
    sock.set_nonblocking(nonblock);
    let oflag = if nonblock { oflags::O_NONBLOCK } else { 0 };
    let handle: Arc<dyn crate::vfs::FileHandle> = sock.clone();
    let file = OpenFile::new(handle, "socket".to_string(), oflag, true, true);
    let new_fd = io.fds.insert(file, flags & SOCK_CLOEXEC != 0)?;
    io.write_sockaddr(m, ip, port, addr, len_out)?;
    Ok(new_fd)
}

fn unix_accept(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, addr: u64, len_out: u64, flags: u64) -> SysResult {
    // A listener in the global kernel answers with an id, and the
    // descriptor made from it is the same kind.
    if let Ok(listener) = io.remote_unix_of(fd) {
        let session = listener.accept()?;
        // The peer's address: a client that never bound has an unnamed one,
        // as Linux reports it (the family alone), not the listener's path.
        Kernel::write_sockaddr_unnamed(m, addr, len_out)?;
        let nonblock = flags & SOCK_NONBLOCK != 0;
        session.set_nonblocking(nonblock);
        let oflag = if nonblock { oflags::O_NONBLOCK } else { 0 };
        let handle: Arc<dyn crate::vfs::FileHandle> = session.clone();
        let file = OpenFile::new(handle, "socket".to_string(), oflag, true, true);
        return io.fds.insert(file, flags & SOCK_CLOEXEC != 0);
    }
    let listener = io.unix_of(fd)?;
    let session = listener.accept()?;
    Kernel::write_sockaddr_unnamed(m, addr, len_out)?;
    let nonblock = flags & SOCK_NONBLOCK != 0;
    session.set_nonblocking(nonblock);
    let oflag = if nonblock { oflags::O_NONBLOCK } else { 0 };
    install_unix(io, session, oflag, flags & SOCK_CLOEXEC != 0)
}

#[allow(clippy::too_many_arguments)]
fn sendto(io: &FdIo<'_>, m: &GuestMem<'_>, fd: i32, buf: u64, len: u64, dest: u64, dest_len: u64, cwd: &str) -> SysResult {
    if let Ok(d) = io.dgram_of(fd) {
        let data = m.read(buf, (len as usize).min(1 << 22))?.to_vec();
        let to = if dest != 0 {
            let path = Kernel::sockaddr_un(m, dest, dest_len)?;
            Some(if path.starts_with('\0') { path } else { crate::vfs::normalize(&path, cwd) })
        } else {
            None
        };
        return d.send_to(&data, to.as_deref());
    }
    if dest != 0 {
        if let Ok(sock) = io.socket_of(fd) {
            let bytes = m.read(dest, 16)?.to_vec();
            if let Ok((ip, port)) = net::parse_sockaddr(&bytes) {
                sock.set_peer(ip, port);
            }
        }
    }
    let data = m.read(buf, (len as usize).min(1 << 22))?.to_vec();
    io.fds.get(fd)?.write(&data)
}

#[allow(clippy::too_many_arguments)]
fn recvfrom(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, flags: u64, addr: u64, addrlen: u64) -> SysResult {
    let len = (len as usize).min(1 << 22);
    let mut tmp = vec![0u8; len];
    if let Ok(d) = io.dgram_of(fd) {
        let (n, whole, from) = d.recv_from(&mut tmp, flags & Kernel::MSG_PEEK != 0)?;
        m.write(buf, &tmp[..n])?;
        if addr != 0 {
            match from {
                Some(p) => Kernel::write_sockaddr_un(m, addr, addrlen, &p)?,
                None => {
                    Kernel::write_sockaddr_unnamed(m, addr, addrlen)?;
                }
            }
        }
        const MSG_TRUNC: u64 = 0x20;
        return Ok(if flags & MSG_TRUNC != 0 { whole } else { n } as u64);
    }
    // The read and the sender address it set, as one (`recv_turn`).
    let inet = if addr != 0 { io.socket_of(fd).ok() } else { None };
    let _turn = inet.as_ref().map(|s| s.recv_turn());
    let n = match io.socket_of(fd) {
        Ok(sock) if flags & Kernel::MSG_PEEK != 0 => sock.peek(&mut tmp)?,
        _ => io.fds.get(fd)?.read(&mut tmp)?,
    };
    m.write(buf, &tmp[..n as usize])?;
    if addr != 0 {
        // Where the datagram came from (a guest's port, the resolver),
        // else the connected peer.
        let peer = io.socket_of(fd).ok().and_then(|s| s.last_from().or_else(|| s.peer()));
        let (ip, port) = peer.unwrap_or((0, 0));
        io.write_sockaddr(m, ip, port, addr, addrlen)?;
    }
    Ok(n)
}

/// The family and type of the socket behind `fd`.
fn socket_family_type(io: &FdIo<'_>, fd: i32) -> Result<(u64, u32), Errno> {
    const AF_INET: u64 = 2;
    const AF_UNIX: u64 = 1;
    const SOCK_STREAM: u32 = 1;
    const SOCK_DGRAM: u32 = 2;
    if let Ok(s) = io.socket_of(fd) {
        return Ok((AF_INET, if s.kind == net::Kind::Udp { SOCK_DGRAM } else { SOCK_STREAM }));
    }
    if io.dgram_of(fd).is_ok() {
        return Ok((AF_UNIX, SOCK_DGRAM));
    }
    if io.unix_of(fd).is_ok() || io.remote_unix_of(fd).is_ok() {
        return Ok((AF_UNIX, SOCK_STREAM));
    }
    let f = io.fds.get(fd)?;
    match f.handle.metadata() {
        Ok(md) if md.file_type == crate::vfs::FileType::Socket => Ok((AF_UNIX, SOCK_STREAM)),
        _ => Err(Errno::ENOTSOCK),
    }
}

fn getsockopt(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, level: u64, name: u64, val: u64, optlen: u64) -> SysResult {
    const SOL_SOCKET: u64 = 1;
    const SO_TYPE: u64 = 3;
    const SO_ERROR: u64 = 4;
    const SO_PROTOCOL: u64 = 38;
    const SO_DOMAIN: u64 = 39;
    // What kind of socket this is, which libuv asks before it decides
    // how to wrap a descriptor it was handed (`uv_guess_handle`): a type
    // of 0 is no kind at all, and node's stdout became a black hole.
    if level == SOL_SOCKET && matches!(name, SO_TYPE | SO_DOMAIN | SO_PROTOCOL) {
        let (family, ty) = socket_family_type(io, fd)?;
        let v = match name {
            SO_TYPE => ty as u64,
            SO_DOMAIN => family,
            _ => 0,
        };
        if val != 0 {
            m.write_u32(val, v as u32)?;
        }
        return Ok(0);
    }
    // A pidfd for the peer (Linux 6.5) and its security label: gex has
    // neither, and says so as a kernel without them does. A success here
    // hands back descriptor 0 as the peer's pidfd, which dbus-daemon
    // duplicates and later closes -- its own standard input, and then
    // whatever took that number next.
    const SO_PEERSEC: u64 = 31;
    const SO_PEERPIDFD: u64 = 77;
    if level == SOL_SOCKET && matches!(name, SO_PEERSEC | SO_PEERPIDFD) {
        socket_family_type(io, fd)?;
        return Err(Errno::ENOPROTOOPT);
    }
    if level == SOL_SOCKET && name == SO_ERROR {
        // A unix socket (local or the global kernel's) has no connect
        // in progress to report: zero, as for a socketpair end on Linux.
        let err = match io.socket_of(fd) {
            Err(_) => {
                socket_family_type(io, fd)?;
                0
            }
            Ok(sock) => match sock.status() {
                net::Status::Refused => Errno::ECONNREFUSED.0,
                // Not `EINPROGRESS`. `SO_ERROR` reports how a connect
                // *finished*, and Linux leaves it zero until one has; a
                // caller is meant to learn "not yet" from the descriptor not
                // being writable, which is what `poll_writable` now says.
                // Reporting it here made `apt` treat "still connecting" as
                // the error that stopped it.
                _ => 0,
            },
        };
        if val != 0 {
            m.write_u32(val, err as u32)?;
        }
        return Ok(0);
    }
    socket_family_type(io, fd)?;
    // SO_PEERCRED: the far end's `struct ucred` (pid, uid, gid; 12 bytes on
    // every ABI), as it was when the two were joined. Every byte is written:
    // dbus-daemon reads the struct off its stack uninitialised and compares
    // the uid with the one a client claims, so a struct left as the caller
    // had it refused every wasm32 client's authentication. A socket in the
    // global kernel asks it (`unix::Remote::peer_cred`). A peer neither
    // kernel saw made (a host's end) has no pid to give (0) and this
    // process's user and group, the only ones there are.
    const SO_PEERCRED: u64 = 17;
    if level == SOL_SOCKET && name == SO_PEERCRED {
        let me = io.cred();
        let peer = io
            .unix_of(fd)
            .ok()
            .and_then(|s| s.peer_cred())
            .or_else(|| io.remote_unix_of(fd).ok().and_then(|s| s.peer_cred()))
            .unwrap_or(unix::Cred { pid: 0, uid: me.uid, gid: me.gid });
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&peer.pid.to_le_bytes());
        bytes[4..8].copy_from_slice(&peer.uid.to_le_bytes());
        bytes[8..12].copy_from_slice(&peer.gid.to_le_bytes());
        if val != 0 {
            let room = if optlen != 0 { (m.read_u32(optlen)? as usize).min(12) } else { 12 };
            m.write(val, &bytes[..room])?;
            if optlen != 0 {
                m.write_u32(optlen, room as u32)?;
            }
        }
        return Ok(0);
    }
    // `setsockopt` takes every option without effect, so these read back
    // as a Linux socket's defaults: the options programs ask about (Wine's
    // ws2_32 for Windows programs, libuv, Chromium's net stack, dbus).
    // Anything else is an option this kernel does not have, said as Linux
    // says it, rather than a success with a value of 0 (dbus-daemon took
    // that 0 as a descriptor, see above).
    const IPPROTO_IP: u64 = 0;
    const IPPROTO_TCP: u64 = 6;
    const IPPROTO_IPV6: u64 = 41;
    let ptr = if io.compat { 4 } else { 8 };
    let (v, len): (u64, u64) = match (level, name) {
        // SO_DEBUG, SO_REUSEADDR, SO_DONTROUTE, SO_BROADCAST, SO_KEEPALIVE,
        // SO_OOBINLINE, SO_NO_CHECK, SO_PRIORITY, SO_REUSEPORT,
        // SO_TIMESTAMP, SO_PASSSEC, SO_MARK
        (SOL_SOCKET, 1 | 2 | 5 | 6 | 9 | 10 | 11 | 12 | 15 | 29 | 34 | 36) => (0, 4),
        // SO_PASSCRED, as set
        (SOL_SOCKET, 16) => {
            let on = io.unix_of(fd).map(|s| s.passcred()).or_else(|_| io.remote_unix_of(fd).map(|s| s.passcred())).unwrap_or(false);
            (on as u64, 4)
        }
        // SO_SNDBUF, SO_RCVBUF: Linux's defaults, as it reports them
        (SOL_SOCKET, 7) => (212_992, 4),
        (SOL_SOCKET, 8) => (212_992, 4),
        // SO_RCVLOWAT, SO_SNDLOWAT
        (SOL_SOCKET, 18 | 19) => (1, 4),
        // SO_ACCEPTCONN
        (SOL_SOCKET, 30) => (io.socket_of(fd).map(|s| s.is_listening()).unwrap_or(false) as u64, 4),
        // SO_LINGER: off (struct linger, two ints)
        (SOL_SOCKET, 13) => (0, 8),
        // SO_RCVTIMEO, SO_SNDTIMEO (and their 64-bit-time forms): none
        (SOL_SOCKET, 20 | 21) => (0, 2 * ptr),
        (SOL_SOCKET, 66 | 67) => (0, 16),
        // IP_TOS, IP_TTL, IP_HDRINCL, IP_PKTINFO, IP_MTU_DISCOVER,
        // IP_MULTICAST_TTL, IP_MULTICAST_LOOP
        (IPPROTO_IP, 1) => (0, 4),
        (IPPROTO_IP, 2) => (64, 4),
        (IPPROTO_IP, 3 | 8 | 10) => (0, 4),
        (IPPROTO_IP, 33 | 34) => (1, 4),
        // TCP_NODELAY, TCP_MAXSEG, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT
        (IPPROTO_TCP, 1) => (0, 4),
        (IPPROTO_TCP, 2) => (1448, 4),
        (IPPROTO_TCP, 4) => (7200, 4),
        (IPPROTO_TCP, 5) => (75, 4),
        (IPPROTO_TCP, 6) => (9, 4),
        // IPV6_UNICAST_HOPS, IPV6_MULTICAST_HOPS, IPV6_MULTICAST_LOOP,
        // IPV6_V6ONLY
        (IPPROTO_IPV6, 16) => (64, 4),
        (IPPROTO_IPV6, 18 | 19) => (1, 4),
        (IPPROTO_IPV6, 26) => (0, 4),
        _ => return Err(Errno::ENOPROTOOPT),
    };
    if val != 0 {
        if len == 0 {
            m.write_u32(val, 0)?;
        } else {
            let mut bytes = vec![0u8; len as usize];
            bytes[..4].copy_from_slice(&(v as u32).to_le_bytes());
            let room = if optlen != 0 { m.read_u32(optlen)? as u64 } else { len };
            m.write(val, &bytes[..len.min(room) as usize])?;
            if optlen != 0 {
                m.write_u32(optlen, len.min(room) as u32)?;
            }
        }
    }
    Ok(0)
}

/// One `struct mmsghdr` is a `msghdr` and the length it carried: 64
/// bytes with the length at 56, or 32 with it at 28 for i386.
fn mmsghdr(compat: bool) -> (u64, u64) {
    if compat {
        (32, 28)
    } else {
        (64, 56)
    }
}

/// `sendmmsg`: several datagrams in one call. glibc's resolver sends its A
/// and AAAA queries this way and does not fall back to `sendto` on
/// `ENOSYS`, so it must be real or name resolution fails.
fn sendmmsg(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, vec: u64, n: u64) -> SysResult {
    let (size, len_at) = mmsghdr(io.compat);
    let mut sent = 0u64;
    for i in 0..n.min(1024) {
        let hdr = vec + i * size;
        match io.sendmsg(m, fd, hdr) {
            Ok(len) => {
                m.write_u32(hdr + len_at, len as u32)?;
                sent += 1;
            }
            // A failure on the first is the call's failure; after that
            // the count is the answer, as it is for `sendmsg`.
            Err(e) if sent == 0 => return Err(e),
            Err(_) => break,
        }
    }
    Ok(sent)
}

fn recvmmsg(io: &FdIo<'_>, m: &mut GuestMem<'_>, fd: i32, vec: u64, n: u64, flags: u64) -> SysResult {
    let (size, len_at) = mmsghdr(io.compat);
    let mut got = 0u64;
    for i in 0..n.min(1024) {
        let hdr = vec + i * size;
        match io.recvmsg(m, fd, hdr, flags) {
            Ok(len) => {
                m.write_u32(hdr + len_at, len as u32)?;
                got += 1;
                // A datagram socket delivers one message per call; if
                // this one came up short there is nothing else waiting.
                if len == 0 {
                    break;
                }
            }
            Err(e) if got == 0 => return Err(e),
            Err(_) => break,
        }
    }
    Ok(got)
}

/// What a call [`free_call`] answers needs of its process, none of it
/// behind a kernel lock.
pub struct Free<'a> {
    pub io: FdIo<'a>,
    /// The memory map, for the mapping calls.
    pub map: MapIo<'a>,
    /// The namespace and the thread's working directory, for the path calls.
    pub path: PathIo<'a>,
    /// The calling thread's signal state.
    pub signals: &'a Signals,
    pub procs: &'a Arc<ProcTable>,
    /// The process's live thread count (`Kernel::live_threads`), for `exit`.
    pub live: &'a std::sync::atomic::AtomicU32,
    /// Its register-context slots (`Kernel::next_ctx_slot`), for `clone`.
    pub slots: &'a CtxSlots,
    /// The terminal (`Kernel::tty`, the same one for the process's life),
    /// for `ioctl`.
    pub tty: &'a crate::tty::TtyRef,
    pub pid: i32,
    /// The process's ids: the sender a signal names, the owner the
    /// credential calls answer and change.
    pub creds: &'a Creds,
    /// The process's threads' signal states, for a `tgkill` to one.
    pub threads: &'a crate::signal::ThreadSigs,
    /// A wasm32 program, whose own code runs a handler's frame.
    pub wasm32: bool,
    /// The process's futexes, for `futex`.
    pub futex: FutexCx<'a>,
    /// When the process started (`Kernel::started`, a constant once its
    /// first call is made): the processor-time clocks, for a sleep until a
    /// time on one.
    pub started: u128,
    /// Its resource limits, for `prlimit64`.
    pub rlimits: &'a Rlimits,
    /// The network and the socket names, for the socket calls.
    pub net: NetIo<'a>,
    /// The process's threads' fs_structs, for `unshare(CLONE_FS)`.
    pub thread_fs: &'a ThreadFs,
}

/// How a call [`free_call`] answers ended.
pub enum FreeOutcome {
    /// Answered, its result in the context; and what a pending signal does
    /// now (`Delivery::Resume` only where the host must go to the handler).
    Done(Delivery),
    /// It waits for its descriptor: park on this, as for
    /// [`SyscallOutcome::Blocking`]; with the mask to put back when the wait
    /// is over (the poll family's own mask: `Kernel::restore_sigmask`).
    Blocking(BlockOp, Option<u64>),
    /// The machine has been halted, or the call ends the process
    /// (`exit_group`, the last thread's `exit`): it ends with this status.
    Exit(i32),
    /// `exit` from one of several threads: its thread is over, the rest go
    /// on (as [`SyscallOutcome::ThreadExit`]).
    ThreadExit,
    /// `clone` made a thread, its context written (the creator's answer,
    /// the tid, is in the caller's), for the host to run (as
    /// [`SyscallOutcome::SpawnThread`]).
    SpawnThread { ctx_addr: u64, tid: i32 },
    /// A wasm32 program's `rt_sigreturn` put back a call its handler
    /// interrupted, to be made again now from the restored context, as a
    /// syscall of its own (`restart_call`).
    Again,
}

/// One extended-attribute call's arguments ([`PathIo::sys_xattr`]): which
/// kind (0 set, 1 get, 2 list, 3 remove), the name, `setxattr`'s flags,
/// and the guest buffer (the value going in, or room for what comes out).
struct Xattr<'a> {
    kind: u64,
    name: &'a str,
    flags: u32,
    at: u64,
    size: u64,
}

impl Xattr<'_> {
    /// Make the call with `call`, copying the value in or the answer out;
    /// `None` when `call` has nowhere to make it.
    fn run(&self, m: &mut GuestMem<'_>, call: impl FnOnce(crate::vfs::XattrOp<'_>) -> Option<Result<usize, Errno>>) -> Option<SysResult> {
        use crate::vfs::XattrOp;
        // Linux caps both at XATTR_SIZE_MAX (a value one past it is E2BIG).
        let mut bytes = match self.kind {
            0 => match m.read(self.at, (self.size as usize).min(crate::vfs::XATTR_SIZE_MAX + 1)) {
                Ok(b) => b.to_vec(),
                Err(e) => return Some(Err(e)),
            },
            1 | 2 => vec![0u8; (self.size as usize).min(crate::vfs::XATTR_SIZE_MAX)],
            _ => Vec::new(),
        };
        let op = match self.kind {
            0 => XattrOp::Set { name: self.name, value: &bytes, flags: self.flags },
            1 => XattrOp::Get { name: self.name, buf: &mut bytes },
            2 => XattrOp::List { buf: &mut bytes },
            _ => XattrOp::Remove { name: self.name },
        };
        let got = match call(op)? {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        if (self.kind == 1 || self.kind == 2) && self.size > 0 {
            if let Err(e) = m.write(self.at, &bytes[..got]) {
                return Some(Err(e));
            }
        }
        Some(Ok(got as u64))
    }
}

/// `ppoll`, `pselect6`, `epoll_pwait` or `epoll_pwait2` (canonical).
fn is_poll(n: u64) -> bool {
    matches!(n, nr::PPOLL | nr::PSELECT6 | nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2)
}

/// The mapping calls ([`MapIo`]), canonical.
/// The path lookups and opens [`PathIo`] answers.
fn is_path(n: u64) -> bool {
    matches!(
        n,
        nr::OPENAT
            | nr::OPENAT2
            | nr::GETDENTS64
            | nr::FSTAT
            | nr::NEWFSTATAT
            | nr::STATX
            | nr::FACCESSAT
            | nr::FACCESSAT2
            | nr::READLINKAT
            | nr::GETCWD
            | nr::MKDIRAT
            | nr::UNLINKAT
            | nr::RENAMEAT
            | nr::RENAMEAT2
            | nr::LINKAT
            | nr::SYMLINKAT
            | nr::UTIMENSAT
            | nr::FCHMOD
            | nr::FCHMODAT
            | nr::SETXATTR..=nr::FREMOVEXATTR
            | nr::FCHOWN
            | nr::FCHOWNAT
            | nr::CHDIR
            | nr::FCHDIR
            | nr::UMASK
    )
}

/// The calls on the process's and the thread's own signal state, behind
/// their own locks (`ProcSig`, `ThreadSig`), and `getrandom`: canonical.
pub fn is_own_state(n: u64) -> bool {
    matches!(n, nr::RT_SIGACTION | nr::RT_SIGPENDING | nr::SIGALTSTACK | nr::SETITIMER | nr::GETITIMER | nr::ALARM | nr::GETRANDOM)
}

/// `process_vm_readv` (`write` false) and `process_vm_writev`: bytes
/// between this process's buffers and another's, the remote side as that
/// process's own map allows (mapped, and readable or writable as the
/// direction needs). The transfer stops at the first remote byte that is
/// not, or at the first local buffer past this memory's end, and answers
/// what it moved before that, or `EFAULT` if nothing. Bytes written over
/// the target's code retire that code's translations, as a write after
/// `mprotect` would.
pub fn process_vm(procs: &ProcTable, m: &mut GuestMem<'_>, compat: bool, write: bool, a: [u64; 6]) -> SysResult {
    const CHUNK: u64 = 64 << 10;
    let pid = a[0] as i32;
    if a[5] != 0 {
        return Err(Errno::EINVAL);
    }
    let local = Kernel::read_iovecs(m, a[1], a[2], compat)?;
    let remote = Kernel::read_iovecs(m, a[3], a[4], compat)?;
    if pid <= 0 {
        return Err(Errno::ESRCH);
    }
    let peer = procs.vm(pid)?;
    let need = if write { crate::mem::prot::WRITE } else { crate::mem::prot::READ };
    let total: u64 = local.iter().map(|v| v.1).sum::<u64>().min(remote.iter().map(|v| v.1).sum());
    let mut buf = vec![0u8; total.min(CHUNK) as usize];
    let (mut li, mut lo, mut ri, mut ro) = (0usize, 0u64, 0usize, 0u64);
    let mut done = 0u64;
    let mut fault = false;
    loop {
        while li < local.len() && lo == local[li].1 {
            (li, lo) = (li + 1, 0);
        }
        while ri < remote.len() && ro == remote[ri].1 {
            (ri, ro) = (ri + 1, 0);
        }
        if li == local.len() || ri == remote.len() {
            break;
        }
        let (la, ra) = (local[li].0.wrapping_add(lo), remote[ri].0.wrapping_add(ro));
        let want = (local[li].1 - lo).min(remote[ri].1 - ro).min(CHUNK);
        let (n, code) = {
            let mm = peer.mm.lock();
            let mut at = ra;
            let end = ra.saturating_add(want);
            let mut code = false;
            while at < end {
                match mm.vma_at(at) {
                    Some(v) if v.prot & need == need => {
                        code |= v.prot & crate::mem::prot::EXEC != 0;
                        at = v.end;
                    }
                    _ => break,
                }
            }
            (at.min(end) - ra, code)
        };
        // An exec or an exit since the lookup: the memory is no longer
        // that process's.
        if n == 0 || !procs.vm(pid).is_ok_and(|now| Arc::ptr_eq(&now, &peer)) {
            fault = true;
            break;
        }
        let b = &mut buf[..n as usize];
        let moved = if write {
            match m.read(la, n as usize) {
                Ok(src) => {
                    b.copy_from_slice(src);
                    peer.mem.write(ra, b)
                }
                Err(_) => false,
            }
        } else {
            peer.mem.read(ra, b) && m.write(la, b).is_ok()
        };
        if !moved {
            fault = true;
            break;
        }
        if write && code {
            let (start, end) = (crate::mem::page_align_down(ra), crate::mem::page_align_up(ra + n));
            peer.mm.lock().log.retired_code.push((start, end));
            crate::note_code_change();
        }
        done += n;
        lo += n;
        ro += n;
        if n < want {
            fault = true;
            break;
        }
    }
    if done == 0 && fault {
        return Err(Errno::EFAULT);
    }
    Ok(done)
}

/// The small descriptor and process calls that need no more than the
/// descriptor table, the global kernel's pipes, the resource limits'
/// own lock, the process table or nothing at all: `epoll_ctl`, `pipe2`,
/// `ftruncate`, `fadvise64`, `statfs`/`fstatfs`, `getpriority`/`setpriority`,
/// `sysinfo`, `prlimit64` and `process_vm_readv`/`writev`. Canonical.
pub fn is_fd_misc(n: u64) -> bool {
    matches!(
        n,
        nr::EPOLL_CTL | nr::PIPE2 | nr::FTRUNCATE | nr::TRUNCATE | nr::FADVISE64 | nr::STATFS | nr::FSTATFS | nr::GETPRIORITY | nr::SETPRIORITY | nr::SYSINFO | nr::PRLIMIT64
            | nr::PROCESS_VM_READV | nr::PROCESS_VM_WRITEV
    )
}

/// The calls that need nothing of the process but its descriptor table
/// (`fsync`, `fdatasync`: a check that the descriptor is open), the process
/// table (`getppid`), the calling thread's own context (`arch_prctl`) or
/// nothing at all: `uname`, `prctl`, `sync`, the robust list and `rseq`,
/// the scheduling policy calls, `sched_getaffinity`, `getcpu`,
/// `membarrier`, the supplementary groups, and the calls that are
/// accepted and do nothing here. Canonical, for this process's ABI
/// (`compat`): `arch_prctl` is x86's, and the do-nothing number is made
/// only by a translation from x86-64 or i386.
pub fn is_quiet(n: u64, compat: bool) -> bool {
    matches!(
        n,
        nr::UNAME
            | nr::PRCTL
            | nr::SYNC
            | nr::SYNCFS
            | nr::SYNC_FILE_RANGE
            | nr::FSYNC
            | nr::FDATASYNC
            | nr::SET_ROBUST_LIST
            | nr::GET_ROBUST_LIST
            | nr::RSEQ
            | nr::SCHED_GETAFFINITY
            | nr::SCHED_GETSCHEDULER
            | nr::SCHED_GETPARAM
            | nr::SCHED_SETPARAM
            | nr::SCHED_SETSCHEDULER
            | nr::SCHED_GET_PRIORITY_MAX
            | nr::SCHED_GET_PRIORITY_MIN
            | nr::GETCPU
            | nr::MEMBARRIER
            | nr::GETGROUPS
            | nr::SETGROUPS
            | nr::GETPPID
            | nr::MLOCK
            | nr::MUNLOCK
            | nr::MLOCKALL
            | nr::MUNLOCKALL
            | nr::MLOCK2
    ) || (n == nr::ARCH_PRCTL && gex_abi::X86)
        || (n == nr::NOOP_OK && (gex_abi::X86 || compat))
}

/// The calls that make a descriptor's object or work one, needing the
/// descriptor table and nothing else of a kernel (and the clock for a
/// timer): `eventfd2`, `epoll_create1`, `memfd_create`, the timerfd calls,
/// `fallocate`, and the transfers `sendfile`, `splice` and
/// `copy_file_range`. Canonical.
pub fn is_fd_object(n: u64) -> bool {
    matches!(
        n,
        nr::EVENTFD2
            | nr::EPOLL_CREATE1
            | nr::INOTIFY_INIT1
            | nr::INOTIFY_ADD_WATCH
            | nr::INOTIFY_RM_WATCH
            | nr::MEMFD_CREATE
            | nr::TIMERFD_CREATE
            | nr::TIMERFD_SETTIME
            | nr::TIMERFD_GETTIME
            | nr::FALLOCATE
            | nr::SENDFILE
            | nr::SPLICE
            | nr::COPY_FILE_RANGE
    )
}

/// A call [`is_fd_object`] admits, made, for the kernel's arms and
/// [`free_call`] alike. A transfer that must wait is the caller's to park
/// first ([`FdIo::transfer_park`]).
pub fn fd_object_call(io: &FdIo<'_>, clock: ClockFn, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
    match n {
        nr::EVENTFD2 => io.eventfd2(a[0], a[1]),
        nr::EPOLL_CREATE1 => io.epoll_create(a[0]),
        nr::INOTIFY_INIT1 => io.inotify_init1(a[0]),
        nr::INOTIFY_ADD_WATCH => io.inotify_add_watch(m, a[0] as i32, a[1], a[2]),
        nr::INOTIFY_RM_WATCH => io.inotify_rm_watch(a[0] as i32, a[1] as i32),
        nr::MEMFD_CREATE => io.memfd_create(m, a[0], a[1]),
        nr::TIMERFD_CREATE => io.timerfd_create(clock, a[0], a[1]),
        nr::TIMERFD_SETTIME => io.timerfd_settime(m, a[0] as i32, a[1], a[2], a[3]),
        nr::TIMERFD_GETTIME => io.timerfd_gettime(m, a[0] as i32, a[1]),
        nr::FALLOCATE => io.fallocate(a[0] as i32, a[1], a[2], a[3]),
        nr::SENDFILE | nr::SPLICE | nr::COPY_FILE_RANGE => io.transfer_syscall(m, n, &a),
        _ => unreachable!("is_fd_object admitted a call fd_object_call does not make"),
    }
}

/// A process's user and group ids, real, effective and saved, and its
/// capability sets. Its threads share one (a `setuid` on any is every
/// thread's, as Linux's process-wide set), a fork copies it, and an `exec`
/// keeps the ids ([`Creds::exec`] recomputes the capabilities). Read at each
/// use (the owner a stat reports, the sender a signal names, `getuid`), so
/// a change is seen by every thread's next call, with or without a kernel
/// lock. A set call stores word by word: two threads setting at once may
/// interleave, which no program relies on.
///
/// The filesystem ids are the effective ones (no `setfsuid`), and there
/// are no file capabilities, no ambient set and no bounding set narrower
/// than every capability.
#[derive(Debug)]
pub struct Creds {
    /// Real, effective, saved.
    uid: [std::sync::atomic::AtomicU32; 3],
    gid: [std::sync::atomic::AtomicU32; 3],
    /// Effective, permitted, inheritable.
    caps: [std::sync::atomic::AtomicU64; 3],
    /// `PR_SET_KEEPCAPS`: a uid change away from root keeps the permitted
    /// set (it clears at exec).
    keep_caps: std::sync::atomic::AtomicBool,
}

/// Every capability Linux 6.x has (`CAP_LAST_CAP` 40): root's sets.
pub const CAPS_ALL: u64 = (1 << 41) - 1;
const CAP_SETPCAP: u64 = 1 << 8;

const RE: usize = 0;
const EF: usize = 1;
const SV: usize = 2;

impl Creds {
    fn blank(uid: u32, gid: u32) -> Creds {
        Creds {
            uid: [uid.into(), uid.into(), uid.into()],
            gid: [gid.into(), gid.into(), gid.into()],
            caps: [0.into(), 0.into(), 0.into()],
            keep_caps: false.into(),
        }
    }

    pub fn new(uid: u32, gid: u32) -> Arc<Creds> {
        let c = Creds::blank(uid, gid);
        c.exec();
        Arc::new(c)
    }

    /// A fork's copy, every id and set as they are.
    pub fn fork(&self) -> Arc<Creds> {
        let c = Creds::blank(0, 0);
        c.restore(&self.save());
        Arc::new(c)
    }

    fn ld32(a: &std::sync::atomic::AtomicU32) -> u32 {
        a.load(std::sync::atomic::Ordering::Acquire)
    }

    fn st32(a: &std::sync::atomic::AtomicU32, v: u32) {
        a.store(v, std::sync::atomic::Ordering::Release)
    }

    /// The effective uid: what permission, ownership and peer credentials
    /// go by.
    pub fn uid(&self) -> u32 {
        Self::ld32(&self.uid[EF])
    }

    /// The effective gid.
    pub fn gid(&self) -> u32 {
        Self::ld32(&self.gid[EF])
    }

    /// The real uid: `getuid`, and the sender a signal names.
    pub fn ruid(&self) -> u32 {
        Self::ld32(&self.uid[RE])
    }

    pub fn rgid(&self) -> u32 {
        Self::ld32(&self.gid[RE])
    }

    /// Real, effective and saved uids.
    pub fn uids(&self) -> [u32; 3] {
        [self.ruid(), self.uid(), Self::ld32(&self.uid[SV])]
    }

    pub fn gids(&self) -> [u32; 3] {
        [self.rgid(), self.gid(), Self::ld32(&self.gid[SV])]
    }

    /// Effective, permitted and inheritable capabilities.
    pub fn caps(&self) -> [u64; 3] {
        self.caps.each_ref().map(|c| c.load(std::sync::atomic::Ordering::Acquire))
    }

    fn set_caps(&self, v: [u64; 3]) {
        for (c, v) in self.caps.iter().zip(v) {
            c.store(v, std::sync::atomic::Ordering::Release);
        }
    }

    /// Become `uid`, real, effective and saved (a host's identity), with
    /// root's capabilities if it is root and none if not.
    pub fn set_uid(&self, uid: u32) {
        for a in &self.uid {
            Self::st32(a, uid);
        }
        self.exec();
    }

    pub fn set_gid(&self, gid: u32) {
        for a in &self.gid {
            Self::st32(a, gid);
        }
    }

    /// The ids and sets as bytes, for a process another memory goes on in
    /// (the browser's fork and exec): 6 `u32`s, 3 `u64`s and a flag byte.
    pub fn save(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(Self::SAVED);
        for id in self.uids().into_iter().chain(self.gids()) {
            v.extend_from_slice(&id.to_le_bytes());
        }
        for c in self.caps() {
            v.extend_from_slice(&c.to_le_bytes());
        }
        v.push(self.keep_caps.load(std::sync::atomic::Ordering::Acquire) as u8);
        v
    }

    /// The length of [`Creds::save`]'s bytes.
    pub const SAVED: usize = 6 * 4 + 3 * 8 + 1;

    /// [`Creds::save`]'s bytes back; anything else changes nothing.
    pub fn restore(&self, b: &[u8]) {
        if b.len() != Self::SAVED {
            return;
        }
        let w = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        for i in 0..3 {
            Self::st32(&self.uid[i], w(i));
            Self::st32(&self.gid[i], w(3 + i));
        }
        let c = |i: usize| u64::from_le_bytes(b[24 + i * 8..32 + i * 8].try_into().unwrap());
        self.set_caps([c(0), c(1), c(2)]);
        self.keep_caps.store(b[48] != 0, std::sync::atomic::Ordering::Release);
    }

    /// The capabilities an `execve` leaves, as Linux computes them for a
    /// file with none of its own: everything for a program run by root
    /// (real or effective), effective only when the effective uid is;
    /// nothing otherwise. The inheritable set stays; `PR_SET_KEEPCAPS`
    /// clears.
    pub fn exec(&self) {
        let [r, e, _] = self.uids();
        let [_, _, inh] = self.caps();
        let permitted = if r == 0 || e == 0 { CAPS_ALL } else { 0 };
        let effective = if e == 0 { permitted } else { 0 };
        self.set_caps([effective, permitted, inh]);
        self.keep_caps.store(false, std::sync::atomic::Ordering::Release);
    }

    /// `prctl(PR_GET_KEEPCAPS)` / `PR_SET_KEEPCAPS`.
    pub fn keep_caps(&self) -> bool {
        self.keep_caps.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn set_keep_caps(&self, on: bool) {
        self.keep_caps.store(on, std::sync::atomic::Ordering::Release);
    }

    /// The capability changes a uid change makes (Linux's
    /// `cap_emulate_setxuid`): leaving root altogether (no id 0 left)
    /// drops the permitted and effective sets, unless `PR_SET_KEEPCAPS`
    /// keeps the permitted; an effective uid leaving 0 drops the effective
    /// set, and one becoming 0 takes the permitted set as effective.
    fn after_uid_change(&self, old: [u32; 3]) {
        let new = self.uids();
        let [mut eff, mut perm, inh] = self.caps();
        if old.contains(&0) && !new.contains(&0) {
            if !self.keep_caps() {
                perm = 0;
            }
            eff = 0;
        }
        if old[1] == 0 && new[1] != 0 {
            eff = 0;
        }
        if old[1] != 0 && new[1] == 0 {
            eff = perm;
        }
        // A process that was not root at all and is now (every change is
        // granted, [`cred_call`]): root's sets, as a set-user-ID-root
        // program's exec would have given it.
        if !old.contains(&0) && new[1] == 0 {
            (eff, perm) = (CAPS_ALL, CAPS_ALL);
        }
        self.set_caps([eff, perm, inh]);
    }

    /// `setresuid`/`setresgid` and the calls made of it: each id `None`
    /// to keep.
    fn set_res(ids: &[std::sync::atomic::AtomicU32; 3], want: [Option<u32>; 3]) {
        for (a, v) in ids.iter().zip(want) {
            if let Some(v) = v {
                Self::st32(a, v);
            }
        }
    }

    /// `setuid(id)` (`gid`: `setgid`), as root's: every id.
    fn setid(&self, gid: bool, id: u32) {
        Self::set_res(if gid { &self.gid } else { &self.uid }, [Some(id); 3]);
    }

    /// `setreuid(r, e)` (`gid`: `setregid`). When the real id is set, or
    /// the effective one set to other than the old real, the saved id
    /// becomes the new effective.
    fn setreid(&self, gid: bool, r: Option<u32>, e: Option<u32>) {
        let ids = if gid { &self.gid } else { &self.uid };
        let (or, oe) = (Self::ld32(&ids[RE]), Self::ld32(&ids[EF]));
        let saved = (r.is_some() || e.is_some_and(|e| e != or)).then_some(e.unwrap_or(oe));
        Self::set_res(ids, [r, e, saved]);
    }

    /// `capget`'s answer for this process.
    pub fn capget(&self) -> [u64; 3] {
        self.caps()
    }

    /// `capset`: the inheritable set within the old inheritable and
    /// permitted (any, with CAP_SETPCAP), the permitted within the old
    /// permitted, the effective within the new permitted; `EPERM` else.
    pub fn capset(&self, eff: u64, perm: u64, inh: u64) -> Result<(), Errno> {
        let [ceff, cperm, cinh] = self.caps();
        let inh_room = if ceff & CAP_SETPCAP != 0 { CAPS_ALL } else { cinh | cperm };
        if inh & !inh_room != 0 || perm & !cperm != 0 || eff & !perm != 0 {
            return Err(Errno::EPERM);
        }
        self.set_caps([eff, perm, inh]);
        Ok(())
    }
}

/// `capget`/`capset`'s header versions: 1 has one 32-bit data struct, 2
/// and 3 two.
const CAP_V1: u32 = 0x1998_0330;
const CAP_V2: u32 = 0x2007_1026;
const CAP_V3: u32 = 0x2008_0522;

/// `capget(hdr, data)` and `capset(hdr, data)` on this process's sets
/// ([`Creds`]). Another live process's sets are not kept where this
/// process can read them, so `capget` of one answers as Linux does for a
/// process run by root, every capability; `capset` of one is `EPERM`, as
/// on Linux.
fn cap_call(creds: &Creds, procs: &ProcTable, pid: i32, m: &mut GuestMem<'_>, set: bool, a: [u64; 6]) -> SysResult {
    let (hdr, data) = (a[0], a[1]);
    let version = m.read_u32(hdr)?;
    let target = m.read_u32(hdr + 4)? as i32;
    let words = match version {
        CAP_V1 => 1,
        CAP_V2 | CAP_V3 => 2,
        _ => {
            // The version this kernel prefers, written back for libcap to
            // ask again with; a probe with no data is not an error.
            m.write_u32(hdr, CAP_V3)?;
            return if data == 0 && !set { Ok(0) } else { Err(Errno::EINVAL) };
        }
    };
    if target < 0 {
        return Err(Errno::EINVAL);
    }
    let me = target == 0 || target == pid;
    if set {
        if !me {
            return Err(Errno::EPERM);
        }
        let mut v = [0u64; 3];
        for w in 0..words {
            for (i, s) in v.iter_mut().enumerate() {
                *s |= u64::from(m.read_u32(data + w * 12 + i as u64 * 4)?) << (32 * w);
            }
        }
        return creds.capset(v[0], v[1], v[2]).map(|_| 0);
    }
    let sets = if me {
        creds.capget()
    } else if procs.task_owner(target).is_some() {
        [CAPS_ALL, CAPS_ALL, 0]
    } else {
        return Err(Errno::ESRCH);
    };
    if data != 0 {
        for w in 0..words {
            for (i, s) in sets.iter().enumerate() {
                m.write_u32(data + w * 12 + i as u64 * 4, (s >> (32 * w)) as u32)?;
            }
        }
    }
    Ok(0)
}

/// The credential, session and process group calls: `get[e]uid`,
/// `mount` and `umount2`: the namespace's mount table, which its
/// published view ([`crate::vfs::VfsView`]) writes. Canonical.
pub fn is_mount(n: u64) -> bool {
    matches!(n, nr::MOUNT | nr::UMOUNT2 | nr::CHROOT)
}

/// The calls on the calling thread's own state alone:
/// `set_tid_address`, the word in its [`crate::signal::ThreadSig`] its exit
/// clears, and `unshare(CLONE_FS)`, a fs_struct of its own in the
/// process's [`ThreadFs`]. Canonical.
pub fn is_thread_own(n: u64) -> bool {
    matches!(n, nr::SET_TID_ADDRESS | nr::UNSHARE)
}

/// The calls that end a thread or the process, or wait for a child:
/// `exit`, `exit_group`, `wait4` and `reboot`. What they touch locks
/// itself: the live thread count (an atomic), the thread's own signal
/// state and fs_struct, the futexes and the process table. Canonical.
pub fn is_lifecycle_end(n: u64) -> bool {
    matches!(n, nr::EXIT | nr::EXIT_GROUP | nr::WAIT4 | nr::WAITID | nr::REBOOT)
}

/// Count one thread out of `live` if it is not the last: whether `exit`
/// ends only its thread (others go on) rather than the process.
pub fn one_of_several(live: &std::sync::atomic::AtomicU32) -> bool {
    use std::sync::atomic::Ordering;
    let mut n = live.load(Ordering::Acquire);
    while n > 1 {
        match live.compare_exchange_weak(n, n - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => n = now,
        }
    }
    false
}

/// What a thread that ends while others go on lets go of (`exit`), the same
/// for the kernel's arm and the free path.
pub struct ThreadEnd<'a> {
    pub thread: &'a crate::signal::ThreadSig,
    pub threads: &'a crate::signal::ThreadSigs,
    pub thread_fs: &'a ThreadFs,
    pub futexes: &'a crate::futex::FutexTable,
    pub procs: &'a ProcTable,
}

impl ThreadEnd<'_> {
    /// End thread `tid`, its context at `ctx_addr`: its fs_struct, its
    /// signal state, its id, and the word its exit clears (zeroed and woken:
    /// `pthread_join` waits on it). The id goes first, so a thread that has
    /// joined it no longer finds it in `/proc`.
    pub fn end(&self, m: &mut GuestMem<'_>, ctx_addr: u64, tid: i32) {
        self.thread_fs.set(ctx_addr, None);
        self.threads.remove(tid);
        self.procs.end_tid(tid);
        let addr = self.thread.clear_child_tid();
        if addr != 0 {
            let _ = m.write_u32(addr, 0);
            self.futexes.wake(addr, u32::MAX);
        }
    }
}

/// `reboot`'s command, if it halts the machine (its name), or `None` for
/// the Ctrl-Alt-Del settings and the rest, which do nothing and succeed.
/// `reboot` is two syscalls wearing one number. The shutdown commands do
/// not return -- there is one machine here and it is this process, so
/// halting it and exiting are the same act -- but the Ctrl-Alt-Del commands
/// are ordinary settings, and busybox init issues `CAD_OFF` as its first
/// act. Treating that as a shutdown ends the boot on line one, which is
/// exactly what it did.
fn reboot_halts(cmd: u64) -> Option<&'static str> {
    const RESTART: u64 = 0x0123_4567;
    const HALT: u64 = 0xcdef_0123;
    const POWER_OFF: u64 = 0x4321_fedc;
    const RESTART2: u64 = 0xa1b2_c3d4;
    match cmd {
        RESTART | RESTART2 => Some("restart"),
        HALT => Some("halt"),
        POWER_OFF => Some("power off"),
        // CAD_OFF, CAD_ON, SW_SUSPEND, KEXEC.
        _ => None,
    }
}

/// `wait4`'s park for process `pid`. Children come from the table, not from
/// a list this process owns, so a child reparented here when its own parent
/// died is one this `wait4` can collect. That is what makes pid 1 able to do
/// its job.
/// `clone` and `clone3` that make a thread (`CLONE_THREAD`): what
/// they touch locks itself (the context slots and live count are atomics,
/// the thread ids the process table's, the fs_structs and signal states
/// their tables'). Canonical. A `clone` that makes a process is the host's
/// (a second address space), decided by its flags ([`thread_clone_flags`]).
pub fn is_thread_clone(n: u64) -> bool {
    matches!(n, nr::CLONE | nr::CLONE3)
}

/// The flags of the `clone` or `clone3` with arguments `a`, if it makes a
/// thread (`CLONE_THREAD`).
fn thread_clone_flags(m: &GuestMem<'_>, n: u64, a: &[u64; 6]) -> Option<u64> {
    let flags = if n == nr::CLONE3 { m.read_u64(a[0]).ok()? } else { a[0] };
    (flags & clone_flags::THREAD != 0).then_some(flags)
}

/// `clone3`'s `struct clone_args` at `c`, as `clone`'s arguments
/// (`[flags, sp, parent_tid, tls, child_tid, 0]`). Its `stack` is the
/// allocation base and `stack_size` its length, but the thread starts on
/// the top (the SP, which grows down). An i386 caller's `tls` is a `struct
/// user_desc *`, as for its `clone` (`crate::i386`): the base is its second
/// word.
fn clone3_args(m: &GuestMem<'_>, c: u64, compat: bool) -> Result<[u64; 6], Errno> {
    // flags@0, child_tid@16, parent_tid@24, stack@40, stack_size@48,
    // tls@56 (aligned u64s).
    let w = |o: u64| m.read_u64(c + o).map_err(|_| Errno::EFAULT);
    let (flags, child_tid, parent_tid, stack, stack_size, tls) = (w(0)?, w(16)?, w(24)?, w(40)?, w(48)?, w(56)?);
    let tls = if compat && flags & clone_flags::SETTLS != 0 && tls != 0 {
        m.read_u32(tls + 4).map(|b| b as u64).unwrap_or(0)
    } else {
        tls
    };
    // and pidfd@8: where CLONE_PIDFD's descriptor goes
    Ok([flags, stack.wrapping_add(stack_size), parent_tid, tls, child_tid, w(8)?])
}

/// What making a thread needs of its process, the same for the
/// kernel's `clone` and the free path's.
pub struct ThreadMaker<'a> {
    pub procs: &'a ProcTable,
    pub pid: i32,
    pub compat: bool,
    /// The register-context slots (`Kernel::next_ctx_slot`).
    pub slots: &'a CtxSlots,
    /// The live thread count (`Kernel::live_threads`).
    pub live: &'a std::sync::atomic::AtomicU32,
    pub thread_fs: &'a ThreadFs,
    pub threads: &'a crate::signal::ThreadSigs,
    /// The creator's signal mask, which the new thread starts with.
    pub blocked: u64,
    /// The creator's fs_struct.
    pub fs: &'a Arc<FsCwd>,
}

/// A process's register-context slots, in the reserved region below the
/// guest's own memory ([`ctx_addr`]): numbered upwards from `next`, and
/// the slots of threads that have ended, which the host gives back once
/// nothing of the thread runs ([`Self::give`]). Without that a process
/// could make only as many threads in its whole life as there are slots.
#[derive(Debug, Default)]
pub struct CtxSlots {
    next: std::sync::atomic::AtomicU32,
    free: std::sync::Mutex<Vec<u32>>,
}

impl CtxSlots {
    /// Slots from `next` up, none given back.
    pub fn new(next: u32) -> Self {
        Self { next: std::sync::atomic::AtomicU32::new(next), free: Default::default() }
    }

    /// A fork's child's: the same, its address space a copy.
    pub fn fork(&self) -> Self {
        Self { next: std::sync::atomic::AtomicU32::new(self.load()), free: std::sync::Mutex::new(self.free_list().clone()) }
    }

    fn free_list(&self) -> std::sync::MutexGuard<'_, Vec<u32>> {
        self.free.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A slot for a new thread: the one given back last, else the next
    /// number, or `None` when that would end past [`gex_abi::CTX_LIMIT`].
    pub fn take(&self) -> Option<u32> {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.free_list().pop() {
            return Some(slot);
        }
        // Claimed with one atomic add; one past the limit is never handed out.
        let slot = self.next.fetch_add(1, Ordering::AcqRel);
        if ctx_addr(slot) + ctx::SIZE as u64 > gex_abi::CTX_LIMIT {
            self.next.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(slot)
    }

    /// The slot of the context at `ctx`, whose thread has ended and whose
    /// host runs nothing of it any more, for the next thread.
    pub fn give(&self, ctx: u64) {
        let stride = (ctx::SIZE as u64 + 63) & !63;
        let Some(off) = ctx.checked_sub(ctx_addr(0)) else { return };
        let slot = (off / stride) as u32;
        if off % stride == 0 && slot != 0 && slot < self.load() {
            self.free_list().push(slot);
        }
    }

    /// The next number a slot would take.
    pub fn load(&self) -> u32 {
        self.next.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Number from `next` up again, none given back: a new program's
    /// (`execve`).
    pub fn reset(&self, next: u32) {
        let mut free = self.free_list();
        free.clear();
        self.next.store(next, std::sync::atomic::Ordering::Release);
    }

    /// Number from `next` up at least, none at or below it given back:
    /// a process running on a context there.
    pub fn at_least(&self, next: u32) {
        let mut free = self.free_list();
        free.retain(|&s| s >= next);
        self.next.fetch_max(next, std::sync::atomic::Ordering::AcqRel);
    }
}

/// What the host runs a process's threads on, when it has only so many (the
/// page: a shim stack and thread-local block per Worker). `take` promises
/// one to a thread being made, or answers false when none is left; `give`
/// takes back a promise the thread will not use.
pub struct ThreadRunners {
    pub take: fn() -> bool,
    pub give: fn(),
}

static THREAD_RUNNERS: std::sync::OnceLock<ThreadRunners> = std::sync::OnceLock::new();

/// Set once by a host with a limit; without one every thread can run.
pub fn set_thread_runners(r: ThreadRunners) {
    let _ = THREAD_RUNNERS.set(r);
}

impl ThreadMaker<'_> {
    /// Make a thread for `clone(a)` (`CLONE_THREAD`) called from the context
    /// at `parent`: its context a copy of the caller's with the child's
    /// answer (0), stack, tid and TLS, its id, its fs_struct and its own
    /// signal state. A thread the host has nothing to run on is refused
    /// with `EAGAIN` before anything of it exists, as Linux refuses one
    /// past its limits: made, it would never run.
    pub fn make(&self, m: &mut GuestMem<'_>, parent: u64, a: [u64; 6]) -> Result<SpawnRequest, Errno> {
        // a thread has no process descriptor of its own
        if a[0] & clone_flags::PIDFD != 0 {
            return Err(Errno::EINVAL);
        }
        let runners = THREAD_RUNNERS.get();
        if runners.is_some_and(|r| !(r.take)()) {
            return Err(Errno::EAGAIN);
        }
        let made = self.make_promised(m, parent, a);
        if made.is_err() {
            if let Some(r) = runners {
                (r.give)();
            }
        }
        made
    }

    fn make_promised(&self, m: &mut GuestMem<'_>, parent: u64, a: [u64; 6]) -> Result<SpawnRequest, Errno> {
        use std::sync::atomic::Ordering;
        let (flags, child_stack, parent_tid, tls, child_tid) = (a[0], a[1], a[2], a[3], a[4]);
        if flags & clone_flags::THREAD == 0 {
            return Err(Errno::ENOSYS);
        }
        if child_stack == 0 {
            return Err(Errno::EINVAL);
        }
        // Contexts live in the reserved region below the guest's own memory;
        // running out of slots is a hard limit, not something to paper over.
        let Some(slot) = self.slots.take() else {
            return Err(Errno::EAGAIN);
        };
        let child = ctx_addr(slot);
        // Numbered with the processes (`ProcTable::new_tid`).
        let tid = self.procs.new_tid(self.pid);
        for off in (0..ctx::SIZE).step_by(8) {
            let v = m.read_u64(parent + off as u64)?;
            m.write_u64(child + off as u64, v)?;
        }
        set_result(m, child, 0)?;
        m.write_u64(child + ctx::sp_slot(self.compat) as u64, child_stack)?;
        m.write_u64(child + ctx::MON_SIZE as u64, 0)?;
        m.write_u64(child + ctx::TID as u64, tid as u64)?;
        if flags & clone_flags::SETTLS != 0 {
            // A 32-bit thread's TLS descriptor is loaded into %gs, a 64-bit
            // one's base into %fs.
            let slot = if self.compat { ctx::TPIDRRO } else { ctx::TPIDR };
            m.write_u64(child + slot as u64, tls)?;
        }
        if flags & clone_flags::PARENT_SETTID != 0 && parent_tid != 0 {
            m.write_u32(parent_tid, tid as u32)?;
        }
        if flags & clone_flags::CHILD_SETTID != 0 && child_tid != 0 {
            m.write_u32(child_tid, tid as u32)?;
        }
        self.live.fetch_add(1, Ordering::AcqRel);
        // Its fs_struct: its creator's with `CLONE_FS` (every pthread), a
        // copy of it without.
        let fs = if flags & clone_flags::FS != 0 { self.thread_fs.of(parent) } else { Some(Arc::new(self.fs.copy())) };
        self.thread_fs.set(child, fs);
        // The new thread's own state: its creator's mask, nothing pending,
        // and with `CLONE_CHILD_CLEARTID` the word its exit clears.
        let own = crate::signal::ThreadSig::new_thread(self.blocked);
        if flags & clone_flags::CHILD_CLEARTID != 0 {
            own.set_clear_child_tid(child_tid);
        }
        self.threads.insert(tid, Arc::new(own));
        Ok(SpawnRequest { ctx_addr: child, tid })
    }
}

fn wait_child(procs: &Arc<ProcTable>, pid: i32, a: [u64; 6]) -> BlockOp {
    BlockOp::WaitChild {
        // Nothing to hand out when the table is elsewhere; the host asks it
        // directly. See `BlockOp::WaitChild`.
        children: if procs.is_shared() { Vec::new() } else { procs.children(pid) },
        procs: Arc::clone(procs),
        ppid: pid,
        want: a[0] as i32,
        wstatus: a[1],
        nohang: a[2] & 1 != 0,
        waitid: None,
    }
}

/// `waitid(idtype, id, infop, options, rusage)` as `wait4`'s park: `P_ALL`
/// and `P_PID` (and `P_PGID`, as any child: process groups are not told
/// apart, as for `wait4`), `WEXITED` with `WNOHANG` and `WNOWAIT`. No child
/// stops or continues here, so `WSTOPPED`/`WCONTINUED` alone find nothing.
/// `P_PIDFD` waits for the child a process descriptor names (`crate::pidfd`):
/// `EBADF` for no descriptor, `EINVAL` for one that is not a process's,
/// `ECHILD` for a process that is not the caller's child, and `EAGAIN` for
/// an `O_NONBLOCK` one whose child is still running.
fn waitid_child(procs: &Arc<ProcTable>, fds: &FdTable, pid: i32, mut a: [u64; 6], compat: bool) -> Result<BlockOp, Errno> {
    if a[0] as u32 == crate::pidfd::P_PIDFD {
        let f = fds.get(a[1] as i32)?;
        let child = f.handle.pidfd().ok_or(Errno::EINVAL)?;
        const WNOHANG: u64 = 1;
        if f.flags() & oflags::O_NONBLOCK != 0 && a[3] & WNOHANG == 0 && !procs.wait_ready(pid, child) {
            return Err(Errno::EAGAIN);
        }
        a[0] = 1;
        a[1] = child as u64;
    }
    const WNOHANG: u64 = 1;
    const WSTOPPED: u64 = 2;
    const WEXITED: u64 = 4;
    const WCONTINUED: u64 = 8;
    const WNOWAIT: u64 = 0x0100_0000;
    const KNOWN: u64 = WNOHANG | WSTOPPED | WEXITED | WCONTINUED | WNOWAIT | 0x2000_0000 | 0x4000_0000 | 0x8000_0000;
    let (idtype, id, options) = (a[0] as u32, a[1] as i32, a[3]);
    if options & !KNOWN != 0 || options & (WSTOPPED | WEXITED | WCONTINUED) == 0 {
        return Err(Errno::EINVAL);
    }
    let want = match idtype {
        0 | 2 => -1,
        1 if id > 0 => id,
        _ => return Err(Errno::EINVAL),
    };
    let mut op = wait_child(procs, pid, [want as u64, 0, options & WNOHANG, 0, 0, 0]);
    if let BlockOp::WaitChild { waitid, nohang, .. } = &mut op {
        *waitid = Some(WaitidOut { info: a[2], nowait: options & WNOWAIT != 0, compat });
        // Only exits are ever waited for here: without WEXITED a wait would
        // never end, so it reports none (zero, or ECHILD with no children).
        if options & WEXITED == 0 {
            *nohang = true;
        }
    }
    Ok(op)
}

/// [`is_thread_own`]'s calls, for thread `tid` (context at `ctx_addr`,
/// fs_struct `fs` now), whose own state is `thread`.
pub fn thread_own_call(
    thread: &crate::signal::ThreadSig,
    thread_fs: &ThreadFs,
    fs: &Arc<FsCwd>,
    ctx_addr: u64,
    tid: i32,
    n: u64,
    a: [u64; 6],
) -> Result<u64, Errno> {
    match n {
        nr::SET_TID_ADDRESS => {
            thread.set_clear_child_tid(a[0]);
            Ok(tid as u64)
        }
        // Only a thread's own working directory (CLONE_FS) is unshared;
        // a private descriptor table or namespace is not supported.
        nr::UNSHARE => {
            if a[0] & !clone_flags::FS != 0 {
                return Err(Errno::EINVAL);
            }
            if a[0] & clone_flags::FS != 0 {
                // A copy of the directory it had, its own from now.
                thread_fs.set(ctx_addr, Some(Arc::new(fs.copy())));
            }
            Ok(0)
        }
        _ => unreachable!("is_thread_own admitted {n}"),
    }
}

/// `get[e]gid`, the `set*id` family, `getres[ug]id`, `setsid`, `getsid`,
/// `getpgid` (and i386's and x86-64's `getpgrp`, translated to it) and
/// `setpgid`. They need the process's [`Creds`] and the process
/// table, which locks itself. Canonical.
pub fn is_cred(n: u64) -> bool {
    matches!(
        n,
        nr::GETUID
            | nr::GETEUID
            | nr::GETGID
            | nr::GETEGID
            | nr::SETUID
            | nr::SETGID
            | nr::SETREUID
            | nr::SETREGID
            | nr::SETRESUID
            | nr::SETRESGID
            | nr::GETRESUID
            | nr::GETRESGID
            | nr::SETFSUID
            | nr::SETFSGID
            | nr::CAPGET
            | nr::CAPSET
            | nr::SETSID
            | nr::GETSID
            | nr::GETPGID
            | nr::SETPGID
    )
}

/// A call [`is_cred`] admits, made, for the kernel's arms and
/// [`free_call`] alike, as process `pid`.
///
/// There is no privilege check on the id changes: the guest is the whole
/// system, and there are no set-user-ID programs (a file's owner is the
/// process's own), so a request to become another user is recorded and
/// granted, as root's would be. Refusing would be the less honest answer,
/// since nothing gex does is protected by the uid in the first place. The
/// real, effective and saved ids are each kept, set as Linux sets them for
/// root, and the capability sets follow a uid change as Linux's do
/// ([`Creds::after_uid_change`]): what peer credentials, `getuid` and
/// `geteuid` report is what Linux's would. All ones means "leave alone".
pub fn cred_call(creds: &Creds, procs: &ProcTable, pid: i32, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
    let r = cred_change(creds, procs, pid, m, n, a);
    // A change the other processes' `/proc/<pid>/status` shows.
    if r.is_ok() && !matches!(n, nr::GETUID | nr::GETEUID | nr::GETGID | nr::GETEGID | nr::GETRESUID | nr::GETRESGID | nr::CAPGET | nr::SETFSUID | nr::SETFSGID | nr::SETSID | nr::GETSID | nr::GETPGID | nr::SETPGID) {
        procs.set_creds(pid, &creds.save());
    }
    r
}

fn cred_change(creds: &Creds, procs: &ProcTable, pid: i32, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> SysResult {
    // -1 keeps an id (the 16-bit calls' 0xffff arrives as -1 too,
    // `i386::canonical`).
    let id = |v: u64| (v as u32 != u32::MAX).then_some(v as u32);
    let getres = |m: &mut GuestMem<'_>, ids: [u32; 3]| {
        for (slot, id) in [a[0], a[1], a[2]].into_iter().zip(ids) {
            if slot != 0 {
                m.write_u32(slot, id)?;
            }
        }
        Ok(0)
    };
    let uid_change = |f: &dyn Fn()| {
        let old = creds.uids();
        f();
        creds.after_uid_change(old);
        Ok(0)
    };
    match n {
        nr::GETUID => Ok(creds.ruid() as u64),
        nr::GETEUID => Ok(creds.uid() as u64),
        nr::GETGID => Ok(creds.rgid() as u64),
        nr::GETEGID => Ok(creds.gid() as u64),
        nr::SETUID => match id(a[0]) {
            Some(v) => uid_change(&|| creds.setid(false, v)),
            None => Err(Errno::EINVAL),
        },
        nr::SETGID => match id(a[0]) {
            Some(v) => Ok(creds.setid(true, v)).map(|_| 0),
            None => Err(Errno::EINVAL),
        },
        nr::SETREUID => uid_change(&|| creds.setreid(false, id(a[0]), id(a[1]))),
        nr::SETREGID => Ok(creds.setreid(true, id(a[0]), id(a[1]))).map(|_| 0),
        nr::SETRESUID => uid_change(&|| Creds::set_res(&creds.uid, [id(a[0]), id(a[1]), id(a[2])])),
        nr::SETRESGID => Ok(Creds::set_res(&creds.gid, [id(a[0]), id(a[1]), id(a[2])])).map(|_| 0),
        nr::GETRESUID => getres(m, creds.uids()),
        nr::GETRESGID => getres(m, creds.gids()),
        // The filesystem ids are the effective ones: the old one back, and
        // nothing changed (Linux answers the old one whatever it does).
        nr::SETFSUID => Ok(creds.uid() as u64),
        nr::SETFSGID => Ok(creds.gid() as u64),
        nr::CAPGET => cap_call(creds, procs, pid, m, false, a),
        nr::CAPSET => cap_call(creds, procs, pid, m, true, a),
        nr::SETSID => procs.setsid(pid).map(|s| s as u64),
        nr::GETSID => procs.sid(pid, a[0] as i32).map(|s| s as u64),
        nr::GETPGID => procs.pgid(pid, a[0] as i32).map(|g| g as u64),
        nr::SETPGID => procs.setpgid(pid, a[0] as i32, a[1] as i32).map(|_| 0),
        _ => unreachable!("is_cred admitted a call cred_call does not make"),
    }
}

/// The signal calls: those that send one, `kill` to a process, a
/// group or every process and `tkill` and `tgkill` to a thread; and the
/// calling thread's own `rt_sigtimedwait`, `rt_sigsuspend` and
/// `rt_sigreturn`. They need the process table, which locks itself, the
/// process's threads' signal states ([`ThreadSigs`](crate::signal::ThreadSigs),
/// behind their own lock), and the caller's. Canonical.
pub fn is_signal_call(n: u64) -> bool {
    matches!(n, nr::KILL | nr::TKILL | nr::TGKILL | nr::RT_SIGTIMEDWAIT | nr::RT_SIGSUSPEND | nr::RT_SIGRETURN | nr::PIDFD_OPEN | nr::PIDFD_SEND_SIGNAL)
}

/// `pidfd_open` and `pidfd_send_signal` (`crate::pidfd`), for the kernel's
/// arm and [`free_call`] alike, into the caller's descriptor table `fds`.
///
/// `pidfd_open(pid, flags)`: a process descriptor for `pid`, which must be
/// a process (a thread's id that is not its process's is `EINVAL`, as
/// without `PIDFD_THREAD`); `PIDFD_NONBLOCK` (`O_NONBLOCK`) the one flag.
///
/// `pidfd_send_signal(pidfd, sig, info, flags)`: `kill` of the process the
/// descriptor names (a zombie too), `ESRCH` once it is reaped. No flags,
/// and no `siginfo` of the caller's: one is `EINVAL` here (Linux would
/// queue it as `rt_sigqueueinfo` does, which gex has no path for).
pub fn pidfd_call(fds: &FdTable, procs: &Arc<ProcTable>, to: &KillTo<'_>, n: u64, a: [u64; 6]) -> SysResult {
    if n == nr::PIDFD_OPEN {
        let (pid, flags) = (a[0] as i32, a[1]);
        if flags & !oflags::O_NONBLOCK != 0 || pid <= 0 {
            return Err(Errno::EINVAL);
        }
        match procs.task_owner(pid) {
            None => return Err(Errno::ESRCH),
            Some(owner) if owner != pid => return Err(Errno::EINVAL),
            Some(_) => {}
        }
        return fds.insert(crate::pidfd::open_file(pid, procs, flags & oflags::O_NONBLOCK != 0), true);
    }
    let f = fds.get(a[0] as i32)?;
    let pid = f.handle.pidfd().ok_or(Errno::EBADF)?;
    if a[3] != 0 || a[2] != 0 || a[1] > crate::signal::sig::MAX as u64 {
        return Err(Errno::EINVAL);
    }
    if f.handle.poll_hup() {
        return Err(Errno::ESRCH);
    }
    kill_call(to, nr::KILL, [pid as u64, a[1], 0, 0, 0, 0])
}

/// `rt_sigtimedwait`'s answer now: a signal of `set` already pending is
/// taken, with its `siginfo_t`; none is `EAGAIN`. A call with none pending
/// and time to wait parks first ([`sigwait_park`]) and comes here once one
/// is. For the kernel's arm and [`free_call`] alike.
pub fn sigtimedwait(signals: &Signals, m: &mut GuestMem<'_>, compat: bool, a: [u64; 6]) -> SysResult {
    let set = if a[0] == 0 { 0 } else { m.read_bits64(a[0])? };
    // The timeout is read and checked before anything is taken, as Linux
    // does (the time64 layout: i386's arrives widened).
    if a[2] != 0 {
        sigwait_timeout(m, a[2])?;
    }
    let Some((signal, source)) = signals.claim_in(set) else { return Err(Errno::EAGAIN) };
    // A `siginfo_t`: si_signo, si_errno, si_code, then the sender for one a
    // process sent (the union is 8-aligned in the 64-bit layout).
    if a[1] != 0 {
        m.zero(a[1], 128)?;
        m.write_u32(a[1], signal)?;
        match source {
            Some(src) => {
                let fields = a[1] + if compat { 12 } else { 16 };
                m.write_u32(a[1] + 8, src.code as u32)?;
                m.write_u32(fields, src.pid as u32)?;
                m.write_u32(fields + 4, src.uid)?;
            }
            None => m.write_u32(a[1] + 8, 0x80)?, // SI_KERNEL
        }
    }
    Ok(signal as u64)
}

/// `rt_sigtimedwait`'s timeout at `at`: `EFAULT` unreadable, `EINVAL` not a
/// time.
fn sigwait_timeout(m: &GuestMem<'_>, at: u64) -> Result<std::time::Duration, Errno> {
    let secs = m.read_u64(at)? as i64;
    let nanos = m.read_u64(at + 8)?;
    if secs < 0 || nanos >= 1_000_000_000 {
        return Err(Errno::EINVAL);
    }
    Ok(std::time::Duration::new(secs as u64, nanos as u32))
}

/// Whether `rt_sigtimedwait` waits: none of its set pending and time to wait
/// (no timeout, or one that is not zero). `None` answers now
/// ([`sigtimedwait`]: the signal, `EAGAIN`, or the error a bad argument
/// is). For the kernel's blocking path and [`free_call`] alike.
pub fn sigwait_park(signals: &Signals, m: &GuestMem<'_>, a: [u64; 6]) -> Option<BlockOp> {
    let set = if a[0] == 0 { 0 } else { m.read_bits64(a[0]).ok()? };
    let timeout = if a[2] == 0 { None } else { Some(sigwait_timeout(m, a[2]).ok()?) };
    // An empty set waits too: for the time, or for a signal to interrupt it.
    if signals.pending_raw() & set != 0 || timeout == Some(std::time::Duration::ZERO) {
        return None;
    }
    Some(BlockOp::SigWait {
        set,
        timeout,
        thread: Arc::clone(signals.thread()),
        process: Arc::clone(signals.process()),
    })
}

/// `rt_sigsuspend`'s start: the mask the caller asks for, in place, and the
/// one to put back afterwards. For the kernel's blocking path and
/// [`free_call`] alike.
///
/// The swap and the wait have to look atomic to the guest -- that is the
/// entire reason the call exists instead of a `sigprocmask` followed by a
/// `pause` -- so the mask changes here, and is put back when the wait is
/// over (`finish_blocking`, or the host's own finish).
pub fn sigsuspend_begin(signals: &Signals, m: &GuestMem<'_>, a: [u64; 6]) -> Result<u64, Errno> {
    let want = m.read_bits64(a[0])?;
    let restore = signals.blocked();
    // Neither `SIGKILL` nor `SIGSTOP` can be blocked, here or anywhere else.
    const UNBLOCKABLE: u64 = (1 << (9 - 1)) | (1 << (19 - 1));
    signals.set_blocked(want & !UNBLOCKABLE);
    Ok(restore)
}

/// What [`kill_call`] sends from: the calling thread's signal state, its
/// process's other threads', the process table, and who is sending.
pub struct KillTo<'a> {
    pub signals: &'a Signals,
    pub threads: &'a crate::signal::ThreadSigs,
    pub procs: &'a ProcTable,
    pub pid: i32,
    /// The calling thread.
    pub tid: i32,
    pub creds: &'a Creds,
}

/// A send [`is_signal_call`] admits (`kill`, `tkill`, `tgkill`), made, for
/// the kernel's arms and [`free_call`] alike. A host that has threads
/// waiting elsewhere wakes them after (a signal for one is raised on its
/// own pending word, which it sees when it looks).
pub fn kill_call(to: &KillTo<'_>, n: u64, a: [u64; 6]) -> SysResult {
    {
        let (pid, tid, signo) = match n {
            nr::KILL => (a[0] as i32, 0, a[1] as u32),
            nr::TKILL => (to.pid, a[0] as i32, a[1] as u32),
            _ => (a[0] as i32, a[1] as i32, a[2] as u32),
        };
        to.signals.process().note_send(to.tid, pid, tid, signo);
    }
    if n == nr::KILL {
        // `kill` addresses a process, which may be any process on the
        // system. It goes through the table, which leaves the signal
        // pending for the target to collect at its next syscall.
        to.procs.kill(to.pid, a[0] as i32, a[1] as u32)?;
        // If the caller was one of the targets -- it signalled itself, or
        // its own group, or everything -- it must see the signal before
        // this syscall returns, because `raise` is expected to have
        // delivered by then. Who the targets were is the table's question
        // to answer, not a rule to restate here: getting it wrong the other
        // way made `kill(-pgid)` signal a caller that was in a different
        // group entirely. Whether or not anything is still in the inbox:
        // another thread's collect may have taken it and not yet raised it,
        // and collecting waits for that one (`ProcSig::collect`).
        to.signals.process().collect(to.procs, to.pid, to.creds.ruid());
        return Ok(0);
    }
    // `tkill(tid, sig)` and `tgkill(tgid, tid, sig)`: a thread-directed
    // signal goes to that thread. Mono's GC stops the world with
    // `tgkill(SIGPWR)` to each thread, and raising it on the caller instead
    // left the target running and the GC waiting.
    let (tgid, tid, signo) = if n == nr::TKILL { (to.pid, a[0] as i32, a[1] as u32) } else { (a[0] as i32, a[1] as i32, a[2] as u32) };
    if signo > crate::signal::sig::MAX || tid <= 0 || (n == nr::TGKILL && tgid <= 0) {
        return Err(Errno::EINVAL);
    }
    let target = if tid == to.tid { None } else { to.threads.get(tid) };
    // An id names one task in the whole system (`ProcTable::new_tid`). One
    // not of this process is another's thread: that process takes the
    // signal, as `kill` gives it; `tgkill` must name its process.
    if target.is_none() && tid != to.tid {
        let owner = to.procs.task_owner(tid).ok_or(Errno::ESRCH)?;
        if owner == to.pid || (n == nr::TGKILL && owner != tgid) {
            return Err(Errno::ESRCH);
        }
        // That thread alone, collected by its process (`collect_signals`).
        // A probe (`signo` 0) asks after the process only.
        if signo == 0 {
            return to.procs.kill(to.pid, owner, 0).map(|_| 0);
        }
        return to.procs.tkill(to.pid, owner, tid, signo).map(|_| 0);
    }
    if tgid != to.pid {
        return Err(Errno::ESRCH);
    }
    if signo > 0 {
        let source = crate::signal::Source { code: crate::signal::SI_TKILL, pid: to.pid, uid: to.creds.ruid() };
        match target {
            Some(t) => t.raise_from(signo, source),
            None => to.signals.raise_from(signo, source),
        }
    }
    Ok(0)
}

/// A call [`is_quiet`] admits, made: the kernel's arms and [`free_call`]
/// both answer them here. `ctx_addr` is the calling thread's context.
pub fn quiet_call(io: &FdIo<'_>, procs: &ProcTable, tid: i32, m: &mut GuestMem<'_>, ctx_addr: u64, n: u64, a: [u64; 6]) -> SysResult {
    match n {
        nr::UNAME => uname(m, io.compat, a[0]),
        // The capability options ([`Creds`]): `PR_GET_KEEPCAPS` and
        // `PR_SET_KEEPCAPS`, and the bounding set, every capability (a
        // drop is not kept; `PR_CAPBSET_READ` of one past the last is
        // `EINVAL`, which libcap counts them by). Names, death signals,
        // dumpability and the rest: all things a process may set about
        // itself that change nothing observable here. A thread's name
        // (`PR_SET_NAME`, `PR_GET_NAME`: what `pthread_setname_np` and
        // `_getname_np` make) is the table's, for `/proc/<pid>/task/<tid>/comm`.
        nr::PRCTL => match a[0] {
            15 => {
                // at most 15 bytes, cut there when no NUL comes sooner
                let name = match m.read_cstr(a[1], 16) {
                    Err(Errno::ENAMETOOLONG) => m.read(a[1], 15)?.to_vec(),
                    r => r?,
                };
                procs.set_task_name(tid, &name);
                Ok(0)
            }
            16 => {
                let e = procs.entries().into_iter().find(|e| e.pid == io.pid);
                let name = e.map_or_else(|| "gex".to_string(), |e| e.task_comm(tid));
                let mut out = [0u8; 16];
                let b = name.as_bytes();
                out[..b.len().min(15)].copy_from_slice(&b[..b.len().min(15)]);
                m.write(a[1], &out)?;
                Ok(0)
            }
            7 => Ok(io.creds.keep_caps() as u64),
            8 if a[1] <= 1 => {
                io.creds.set_keep_caps(a[1] == 1);
                Ok(0)
            }
            8 => Err(Errno::EINVAL),
            23 if a[1] <= 40 => Ok(1),
            23 => Err(Errno::EINVAL),
            // gex's own: the process has crashed, and says how (the report
            // its exit makes, `ProcTable::set_crash`).
            crate::proc::PR_GEX_CRASH => {
                let note = m.read(a[1], (a[2] as usize).min(1024))?.to_vec();
                procs.set_crash(io.pid, &note);
                Ok(0)
            }
            _ => Ok(0),
        },
        // Everything written made durable: the block store's dirty blocks
        // written out and the store flushed (vfs::sync_all); with the files
        // in memory, nothing. MAP_SHARED pages still in a process's memory
        // reach the file at msync and munmap, not here.
        nr::SYNC => crate::vfs::sync_all().map(|_| 0),
        // One filesystem: the fd's is the root's (a mount's provider keeps
        // its own, and flushes nothing here).
        nr::SYNCFS => {
            io.fds.get(a[0] as i32)?;
            crate::vfs::sync_all().map(|_| 0)
        }
        // A range of the file written out: the file's data, as fdatasync.
        nr::SYNC_FILE_RANGE => io.fds.get(a[0] as i32)?.handle.sync(true).map(|_| 0),
        // `fsync` and `fdatasync`: the file's own (FileHandle::sync); a file
        // in memory has nothing to flush, so it is the fd's validation alone.
        // (Without them sqlite's journal sync was ENOSYS, which it reports
        // as SQLITE_IOERR and refuses to write a file-backed database.)
        // The file's size and names with it: a generation of the kept root.
        nr::FSYNC | nr::FDATASYNC => io.fds.get(a[0] as i32)?.handle.sync(false).map(|_| 0),
        nr::SET_ROBUST_LIST | nr::GET_ROBUST_LIST => Ok(0),
        // Restartable sequences are an optimisation the guest can live
        // without, and both libcs treat ENOSYS as "not available".
        nr::RSEQ => Err(Errno::ENOSYS),
        // Every processor the guest is told about (`crate::cpus`): what
        // musl's sysconf counts, and what dpkg and engines size their worker
        // pools by.
        nr::SCHED_GETAFFINITY => {
            let (len, mask) = ((a[1] as usize).min(1024), a[2]);
            if mask != 0 && len >= 8 {
                m.zero(mask, len)?;
                m.write_u64(mask, crate::cpus::mask())?;
            }
            Ok(8)
        }
        // Scheduling policy: every guest thread runs SCHED_OTHER at
        // priority 0, which is what these report. Setting a normal policy
        // is accepted; a real-time one is refused as for an unprivileged
        // process. Mono treats a failed priority query as fatal.
        nr::SCHED_GETSCHEDULER => Ok(0),
        nr::SCHED_GETPARAM => {
            m.write_u32(a[1], 0)?;
            Ok(0)
        }
        nr::SCHED_SETPARAM => Ok(0),
        nr::SCHED_SETSCHEDULER => match a[1] & 0xff {
            1 | 2 => Err(Errno::EPERM),
            0 | 3 | 5 => Ok(0),
            _ => Err(Errno::EINVAL),
        },
        nr::SCHED_GET_PRIORITY_MAX => Ok(if matches!(a[0], 1 | 2) { 99 } else { 0 }),
        nr::SCHED_GET_PRIORITY_MIN => Ok(if matches!(a[0], 1 | 2) { 1 } else { 0 }),
        // getcpu(cpu, node, tcache): a single scheduling domain, CPU 0.
        nr::GETCPU => {
            if a[0] != 0 {
                m.write_u32(a[0], 0)?;
            }
            if a[1] != 0 {
                m.write_u32(a[1], 0)?;
            }
            Ok(0)
        }
        nr::MEMBARRIER => Ok(0),
        // No supplementary groups. `getgroups(0, _)` asks for the count.
        nr::GETGROUPS | nr::SETGROUPS => Ok(0),
        nr::GETPPID => Ok(procs.ppid(io.pid) as u64),
        nr::ARCH_PRCTL => arch_prctl(m, ctx_addr, a[0], a[1]),
        nr::NOOP_OK => Ok(0),
        // Memory is never paged out, so every page is as resident as a
        // locked one: locking and unlocking have nothing to do.
        // Flags Linux does not know are EINVAL: MCL_CURRENT, MCL_FUTURE and
        // MCL_ONFAULT for mlockall (at least one), MLOCK_ONFAULT for mlock2.
        nr::MLOCKALL if a[0] == 0 || a[0] & !7 != 0 => Err(Errno::EINVAL),
        nr::MLOCK2 if a[2] & !1 != 0 => Err(Errno::EINVAL),
        nr::MLOCK | nr::MUNLOCK | nr::MLOCKALL | nr::MUNLOCKALL | nr::MLOCK2 => Ok(0),
        _ => unreachable!("is_quiet admitted a call quiet_call does not make"),
    }
}

/// `uname`: the kernel gex says it is.
fn uname(m: &mut GuestMem<'_>, compat: bool, buf: u64) -> SysResult {
    let fields = [
        "Linux",
        "gex",
        "6.1.0-gex",
        "#1 SMP gex",
        // An i386 process on the x86 build; a wasm32 program on the
        // AArch64 build is part of an aarch64 system (its dpkg and
        // config.guess should agree with the arm64 programs beside it).
        if compat && gex_abi::X86 { "i686" } else { gex_abi::ARCH_NAME },
        "(none)",
    ];
    let mut out = vec![0u8; 65 * 6];
    for (i, f) in fields.iter().enumerate() {
        let b = f.as_bytes();
        out[i * 65..i * 65 + b.len()].copy_from_slice(b);
    }
    m.write(buf, &out)?;
    Ok(0)
}

/// `arch_prctl`: the calling thread's segment bases, in its context.
fn arch_prctl(m: &mut GuestMem<'_>, ctx_addr: u64, code: u64, addr: u64) -> SysResult {
    const ARCH_SET_GS: u64 = 0x1001;
    const ARCH_SET_FS: u64 = 0x1002;
    const ARCH_GET_FS: u64 = 0x1003;
    const ARCH_GET_GS: u64 = 0x1004;
    match code {
        ARCH_SET_FS => m.write_u64(ctx_addr + ctx::FS_BASE as u64, addr)?,
        ARCH_SET_GS => m.write_u64(ctx_addr + ctx::GS_BASE as u64, addr)?,
        ARCH_GET_FS => {
            let v = m.read_u64(ctx_addr + ctx::FS_BASE as u64)?;
            m.write_u64(addr, v)?;
        }
        ARCH_GET_GS => {
            let v = m.read_u64(ctx_addr + ctx::GS_BASE as u64)?;
            m.write_u64(addr, v)?;
        }
        _ => return Err(Errno::EINVAL),
    }
    Ok(0)
}

fn is_mapping(n: u64) -> bool {
    matches!(n, nr::BRK | nr::MMAP | nr::MUNMAP | nr::MPROTECT | nr::MREMAP | nr::MADVISE | nr::MSYNC)
}

/// SIGKILL and SIGSTOP, which no mask blocks.
const UNBLOCKABLE: u64 = (1 << (signal::sig::SIGKILL - 1)) | (1 << (signal::sig::SIGSTOP - 1));

/// The canonical number of a call [`free_call`] answers, from the number
/// in the context as the caller's ABI has it.
fn free_number(raw: u64, compat: bool) -> Option<u64> {
    let n = if compat {
        use crate::i386::raw as x;
        match raw {
            x::READ => nr::READ,
            x::WRITE => nr::WRITE,
            x::READV => nr::READV,
            x::WRITEV => nr::WRITEV,
            x::PREAD64 => nr::PREAD64,
            x::PWRITE64 => nr::PWRITE64,
            x::LSEEK | x::LLSEEK => nr::LSEEK,
            x::RT_SIGPROCMASK => nr::RT_SIGPROCMASK,
            x::POLL | x::PPOLL | x::PPOLL_TIME64 => nr::PPOLL,
            x::NEWSELECT | x::PSELECT6 | x::PSELECT6_TIME64 => nr::PSELECT6,
            x::EPOLL_WAIT | x::EPOLL_PWAIT => nr::EPOLL_PWAIT,
            x::EPOLL_PWAIT2 => nr::EPOLL_PWAIT2,
            x::FUTEX | x::FUTEX_TIME64 => nr::FUTEX,
            x::NANOSLEEP => nr::NANOSLEEP,
            x::CLOCK_NANOSLEEP | x::CLOCK_NANOSLEEP_TIME64 => nr::CLOCK_NANOSLEEP,
            x::SENDMSG => nr::SENDMSG,
            x::RECVMSG => nr::RECVMSG,
            x::IOCTL => nr::IOCTL,
            x::CLOSE => nr::CLOSE,
            x::CLOSE_RANGE => nr::CLOSE_RANGE,
            x::DUP => nr::DUP,
            x::DUP2 => nr::DUP2,
            x::DUP3 => nr::DUP3,
            x::FCNTL | x::FCNTL64 => nr::FCNTL,
            x::FLOCK => nr::FLOCK,
            x::BRK => nr::BRK,
            x::MMAP | x::MMAP2 => nr::MMAP,
            x::MUNMAP => nr::MUNMAP,
            x::MPROTECT => nr::MPROTECT,
            x::MREMAP => nr::MREMAP,
            x::MADVISE => nr::MADVISE,
            x::MSYNC => nr::MSYNC,
            x::OPEN | x::CREAT | x::OPENAT => nr::OPENAT,
            x::OPENAT2 => nr::OPENAT2,
            x::STAT | x::STAT64 | x::LSTAT | x::LSTAT64 | x::FSTATAT64 => nr::NEWFSTATAT,
            x::FSTAT | x::FSTAT64 => nr::FSTAT,
            x::STATX => nr::STATX,
            x::ACCESS | x::FACCESSAT => nr::FACCESSAT,
            x::FACCESSAT2 => nr::FACCESSAT2,
            x::READLINK | x::READLINKAT => nr::READLINKAT,
            x::GETCWD => nr::GETCWD,
            x::GETDENTS64 => nr::GETDENTS64,
            x::MKDIR | x::MKDIRAT => nr::MKDIRAT,
            x::UNLINK | x::RMDIR | x::UNLINKAT => nr::UNLINKAT,
            x::RENAME | x::RENAMEAT => nr::RENAMEAT,
            x::RENAMEAT2 => nr::RENAMEAT2,
            x::LINK | x::LINKAT => nr::LINKAT,
            x::SYMLINK | x::SYMLINKAT => nr::SYMLINKAT,
            x::UTIMENSAT | x::UTIMENSAT_TIME64 => nr::UTIMENSAT,
            x::CHDIR => nr::CHDIR,
            x::FCHDIR => nr::FCHDIR,
            x::UMASK => nr::UMASK,
            x::CHMOD | x::FCHMODAT => nr::FCHMODAT,
            x::SETXATTR..=x::FREMOVEXATTR => nr::SETXATTR + (raw - x::SETXATTR),
            x::FCHMOD => nr::FCHMOD,
            x::CHOWN | x::LCHOWN | x::LCHOWN16 | x::FCHOWNAT => nr::FCHOWNAT,
            x::FCHOWN | x::FCHOWN16 => nr::FCHOWN,
            x::RT_SIGACTION => nr::RT_SIGACTION,
            x::RT_SIGPENDING => nr::RT_SIGPENDING,
            x::SIGALTSTACK => nr::SIGALTSTACK,
            // Their itimervals widened and narrowed (`i386::translate`).
            x::SETITIMER => nr::SETITIMER,
            x::GETITIMER => nr::GETITIMER,
            x::ALARM => nr::ALARM,
            x::GETRANDOM => nr::GETRANDOM,
            // `i386::translate` widens them (the old `statfs`/`fstatfs`
            // with 32-bit counts are not made: musl does not).
            x::EPOLL_CTL => nr::EPOLL_CTL,
            x::PIPE | x::PIPE2 => nr::PIPE2,
            x::FTRUNCATE | x::FTRUNCATE64 => nr::FTRUNCATE,
            x::TRUNCATE | x::TRUNCATE64 => nr::TRUNCATE,
            x::FADVISE64 | x::FADVISE64_64 => nr::FADVISE64,
            x::STATFS64 => nr::STATFS,
            x::FSTATFS64 => nr::FSTATFS,
            x::GETPRIORITY => nr::GETPRIORITY,
            x::SETPRIORITY => nr::SETPRIORITY,
            x::SYSINFO => nr::SYSINFO,
            x::PROCESS_VM_READV => nr::PROCESS_VM_READV,
            x::PROCESS_VM_WRITEV => nr::PROCESS_VM_WRITEV,
            x::GETRLIMIT | x::UGETRLIMIT | x::PRLIMIT64 => nr::PRLIMIT64,
            x::SOCKET => nr::SOCKET,
            x::SOCKETPAIR => nr::SOCKETPAIR,
            x::BIND => nr::BIND,
            x::CONNECT => nr::CONNECT,
            x::LISTEN => nr::LISTEN,
            x::ACCEPT4 => nr::ACCEPT4,
            x::GETSOCKOPT => nr::GETSOCKOPT,
            x::SETSOCKOPT => nr::SETSOCKOPT,
            x::GETSOCKNAME => nr::GETSOCKNAME,
            x::GETPEERNAME => nr::GETPEERNAME,
            x::SENDTO => nr::SENDTO,
            x::RECVFROM => nr::RECVFROM,
            x::SHUTDOWN => nr::SHUTDOWN,
            x::SENDMMSG => nr::SENDMMSG,
            x::RECVMMSG_TIME64 => nr::RECVMMSG,
            // The quiet calls, as the translation numbers them (several
            // numbers become the do-nothing one).
            r => match crate::i386::canonical_number(r) {
                c if is_quiet(c, true) || is_fd_object(c) || is_cred(c) || is_signal_call(c) || is_thread_own(c) || is_mount(c) || is_lifecycle_end(c) || is_thread_clone(c) => c,
                _ => return None,
            },
        }
    } else if gex_abi::X86 {
        use crate::x86::raw as x;
        match raw {
            x::READ => nr::READ,
            x::WRITE => nr::WRITE,
            x::READV => nr::READV,
            x::WRITEV => nr::WRITEV,
            x::PREAD64 => nr::PREAD64,
            x::PWRITE64 => nr::PWRITE64,
            x::LSEEK => nr::LSEEK,
            x::RT_SIGPROCMASK => nr::RT_SIGPROCMASK,
            x::POLL | x::PPOLL => nr::PPOLL,
            x::SELECT | x::PSELECT6 => nr::PSELECT6,
            x::EPOLL_WAIT | x::EPOLL_PWAIT => nr::EPOLL_PWAIT,
            x::EPOLL_PWAIT2 => nr::EPOLL_PWAIT2,
            x::FUTEX => nr::FUTEX,
            x::NANOSLEEP => nr::NANOSLEEP,
            x::CLOCK_NANOSLEEP => nr::CLOCK_NANOSLEEP,
            x::SENDMSG => nr::SENDMSG,
            x::RECVMSG => nr::RECVMSG,
            x::IOCTL => nr::IOCTL,
            x::CLOSE => nr::CLOSE,
            x::CLOSE_RANGE => nr::CLOSE_RANGE,
            x::DUP => nr::DUP,
            x::DUP2 => nr::DUP2,
            x::DUP3 => nr::DUP3,
            x::FCNTL => nr::FCNTL,
            x::FLOCK => nr::FLOCK,
            x::BRK => nr::BRK,
            x::MMAP => nr::MMAP,
            x::MUNMAP => nr::MUNMAP,
            x::MPROTECT => nr::MPROTECT,
            x::MREMAP => nr::MREMAP,
            x::MADVISE => nr::MADVISE,
            x::MSYNC => nr::MSYNC,
            x::OPEN | x::CREAT | x::OPENAT => nr::OPENAT,
            x::OPENAT2 => nr::OPENAT2,
            x::STAT | x::LSTAT | x::NEWFSTATAT => nr::NEWFSTATAT,
            x::FSTAT => nr::FSTAT,
            x::STATX => nr::STATX,
            x::ACCESS | x::FACCESSAT => nr::FACCESSAT,
            x::FACCESSAT2 => nr::FACCESSAT2,
            x::READLINK | x::READLINKAT => nr::READLINKAT,
            x::GETCWD => nr::GETCWD,
            x::GETDENTS64 => nr::GETDENTS64,
            x::MKDIR | x::MKDIRAT => nr::MKDIRAT,
            x::UNLINK | x::RMDIR | x::UNLINKAT => nr::UNLINKAT,
            x::RENAME | x::RENAMEAT => nr::RENAMEAT,
            x::RENAMEAT2 => nr::RENAMEAT2,
            x::LINK | x::LINKAT => nr::LINKAT,
            x::SYMLINK | x::SYMLINKAT => nr::SYMLINKAT,
            x::UTIMENSAT => nr::UTIMENSAT,
            x::CHDIR => nr::CHDIR,
            x::FCHDIR => nr::FCHDIR,
            x::UMASK => nr::UMASK,
            x::CHMOD | x::FCHMODAT => nr::FCHMODAT,
            x::SETXATTR..=x::FREMOVEXATTR => nr::SETXATTR + (raw - x::SETXATTR),
            x::FCHMOD => nr::FCHMOD,
            x::CHOWN | x::LCHOWN | x::FCHOWNAT => nr::FCHOWNAT,
            x::FCHOWN => nr::FCHOWN,
            x::RT_SIGACTION => nr::RT_SIGACTION,
            x::RT_SIGPENDING => nr::RT_SIGPENDING,
            x::SIGALTSTACK => nr::SIGALTSTACK,
            x::SETITIMER => nr::SETITIMER,
            x::GETITIMER => nr::GETITIMER,
            x::ALARM => nr::ALARM,
            x::GETRANDOM => nr::GETRANDOM,
            x::EPOLL_CTL => nr::EPOLL_CTL,
            x::PIPE | x::PIPE2 => nr::PIPE2,
            x::FTRUNCATE => nr::FTRUNCATE,
            x::TRUNCATE => nr::TRUNCATE,
            x::FADVISE64 => nr::FADVISE64,
            x::STATFS => nr::STATFS,
            x::FSTATFS => nr::FSTATFS,
            x::GETPRIORITY => nr::GETPRIORITY,
            x::SETPRIORITY => nr::SETPRIORITY,
            x::SYSINFO => nr::SYSINFO,
            x::PROCESS_VM_READV => nr::PROCESS_VM_READV,
            x::PROCESS_VM_WRITEV => nr::PROCESS_VM_WRITEV,
            x::GETRLIMIT | x::SETRLIMIT | x::PRLIMIT64 => nr::PRLIMIT64,
            x::SOCKET => nr::SOCKET,
            x::SOCKETPAIR => nr::SOCKETPAIR,
            x::BIND => nr::BIND,
            x::CONNECT => nr::CONNECT,
            x::LISTEN => nr::LISTEN,
            x::ACCEPT => nr::ACCEPT,
            x::ACCEPT4 => nr::ACCEPT4,
            x::GETSOCKOPT => nr::GETSOCKOPT,
            x::SETSOCKOPT => nr::SETSOCKOPT,
            x::GETSOCKNAME => nr::GETSOCKNAME,
            x::GETPEERNAME => nr::GETPEERNAME,
            x::SENDTO => nr::SENDTO,
            x::RECVFROM => nr::RECVFROM,
            x::SHUTDOWN => nr::SHUTDOWN,
            x::SENDMMSG => nr::SENDMMSG,
            x::RECVMMSG => nr::RECVMMSG,
            r => match crate::x86::canonical_number(r) {
                c if is_quiet(c, false) || is_fd_object(c) || is_cred(c) || is_signal_call(c) || is_thread_own(c) || is_mount(c) || is_lifecycle_end(c) || is_thread_clone(c) => c,
                _ => return None,
            },
        }
    } else {
        match raw {
            nr::READ | nr::WRITE | nr::READV | nr::WRITEV | nr::PREAD64 | nr::PWRITE64 | nr::LSEEK
            | nr::RT_SIGPROCMASK | nr::PPOLL | nr::PSELECT6 | nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2 | nr::FUTEX
            | nr::NANOSLEEP | nr::CLOCK_NANOSLEEP | nr::SENDMSG | nr::RECVMSG | nr::IOCTL | nr::CLOSE
            | nr::CLOSE_RANGE | nr::DUP | nr::DUP3 | nr::FCNTL | nr::FLOCK | nr::BRK | nr::MMAP | nr::MUNMAP
            | nr::MPROTECT | nr::MREMAP | nr::MADVISE | nr::MSYNC => raw,
            n if is_path(n) => raw,
            // No `alarm` in this ABI.
            n if is_own_state(n) && n != nr::ALARM => raw,
            n if is_fd_misc(n) => raw,
            n if is_socket_call(n) => raw,
            n if is_quiet(n, false) || is_fd_object(n) || is_cred(n) || is_signal_call(n) || is_thread_own(n) || is_mount(n) || is_lifecycle_end(n) || is_thread_clone(n) => raw,
            _ => return None,
        }
    };
    Some(n)
}

/// A call [`free_call`] answers, as read from its thread's context.
#[derive(Clone, Copy, Debug)]
pub struct FreeCall {
    /// The canonical number, and the number as the caller's ABI has it.
    n: u64,
    raw: u64,
    a: [u64; 6],
}

/// Whether the call in the context at `ctx_addr` is one [`free_call`]
/// answers, decided by its number alone: `read`, `write`, `readv`,
/// `writev`, `pread64`, `pwrite64`, `lseek`, `rt_sigprocmask`, the poll
/// family (`poll`, `ppoll`, `select`, `pselect6`, `epoll_wait`,
/// `epoll_pwait`, `epoll_pwait2`, in each ABI's forms), `futex` (every
/// operation, and i386's `futex_time64`), `nanosleep`, `clock_nanosleep`,
/// `sendmsg`, `recvmsg`, `ioctl`, and the descriptor table's own calls
/// (`close`, `close_range`, `dup`, `dup2`, `dup3`, `fcntl`, `flock`), and
/// the mapping calls (`mmap`, `munmap`, `mprotect`, `brk`, `mremap`,
/// `madvise`, `msync`: [`MapIo`]), the calls on the process's and thread's
/// own signal state (`rt_sigaction`, `rt_sigpending`, `sigaltstack`,
/// `setitimer`, `getitimer`, `alarm`) and `getrandom` ([`is_own_state`]);
/// the 32-bit ABI's `pread64`, `pwrite64`,
/// `_llseek`, `fcntl64`, `mmap2` and old `mmap` too, translated as the full
/// call translates them.
pub fn free_class(m: &GuestMem<'_>, ctx_addr: u64, compat: bool) -> Option<FreeCall> {
    free_class_of(m, call_regs(m, ctx_addr, compat), compat)
}

/// A call's number and arguments as its ABI gives them, read from the
/// context once ([`call_regs`]): what [`free_class_of`] classes, and what a
/// host's own bookkeeping of the call reads instead of the context again.
#[derive(Clone, Copy, Debug)]
pub struct CallRegs {
    /// The number in the ABI's own numbering (an i386 process's from `x(0)`,
    /// 32 bits).
    pub raw: u64,
    /// The six arguments (an i386 process's 32 bits each, unextended).
    pub a: [u64; 6],
}

impl CallRegs {
    /// The canonical number (the kernel's own, AArch64's).
    pub fn canonical(&self, compat: bool) -> u64 {
        if compat {
            crate::i386::canonical_number(self.raw)
        } else if gex_abi::X86 {
            crate::x86::canonical_number(self.raw)
        } else {
            self.raw
        }
    }
}

/// The call in the context at `ctx_addr`: its number and arguments, where
/// [`free_class`] and the full call read them.
pub fn call_regs(m: &GuestMem<'_>, ctx_addr: u64, compat: bool) -> CallRegs {
    let raw = if compat {
        m.read_u64(ctx_addr + ctx::x(0) as u64).unwrap_or(!0) & 0xffff_ffff
    } else {
        m.read_u64(ctx_addr + ctx::SYSCALL_NR as u64).unwrap_or(!0)
    };
    let mut a = [0u64; 6];
    for (i, v) in a.iter_mut().enumerate() {
        *v = if compat {
            m.read_u64(ctx_addr + crate::i386::ARG_SLOTS[i] as u64).unwrap_or(0) & 0xffff_ffff
        } else {
            m.read_u64(ctx_addr + ctx::SYSCALL_ARGS[i] as u64).unwrap_or(0)
        };
    }
    CallRegs { raw, a }
}

/// [`free_class`] of a call already read ([`call_regs`] of the same
/// context and ABI).
pub fn free_class_of(m: &GuestMem<'_>, regs: CallRegs, compat: bool) -> Option<FreeCall> {
    let CallRegs { raw, mut a } = regs;
    // i386's `socketcall` is classed as the call it makes, with that call's
    // arguments; `raw` stays the number the guest used, which a restart
    // makes again.
    let inner = if compat && raw == crate::i386::raw::SOCKETCALL {
        match crate::i386::socketcall_unpacked(m, &a) {
            Some((n, v)) => {
                a = v;
                n
            }
            None => return nosys_call(raw, compat),
        }
    } else {
        raw
    };
    let Some(n) = free_number(inner, compat) else { return nosys_call(raw, compat) };
    // A `clone` that makes a process (no `CLONE_THREAD`) is the host's, and
    // so are `fork` and `vfork`, which are `clone` canonically: only the
    // ABI's own `clone` and `clone3` carry their flags in the first argument.
    if is_thread_clone(n) {
        let (clone, clone3) = if compat {
            (crate::i386::raw::CLONE, crate::i386::raw::CLONE3)
        } else if gex_abi::X86 {
            (crate::x86::raw::CLONE, crate::x86::raw::CLONE3)
        } else {
            (nr::CLONE, nr::CLONE3)
        };
        if (raw != clone && raw != clone3) || thread_clone_flags(m, n, &a).is_none() {
            return None;
        }
    }
    // The poll family's arguments are the ABI's own until `free_call`
    // translates them (into the thread's scratch, which needs the memory
    // writable).
    // A mapping call's first argument is an address (or, for the old
    // `mmap`, a pointer to its arguments), never sign-extended.
    if compat && !matches!(n, nr::RT_SIGPROCMASK | nr::FUTEX | nr::NANOSLEEP | nr::CLOCK_NANOSLEEP | nr::CLOSE_RANGE) && !is_poll(n) && !is_mapping(n) && !is_path(n) && !is_own_state(n) && !is_fd_misc(n) && !is_quiet(n, true) && !is_fd_object(n) && !is_cred(n) && !is_signal_call(n) && !is_thread_own(n) && !is_mount(n) && !is_lifecycle_end(n) && !is_thread_clone(n) {
        a[0] = crate::i386::sext(a[0]);
    }
    Some(FreeCall { n, raw, a })
}

/// A call in none of the free path's classes: one that makes or replaces
/// a process (`execve`, and `clone`, `clone3`, `fork` and `vfork` making a
/// process), which is the host's (`None`), or one the kernel does not make,
/// which [`free_call`] answers `ENOSYS` (its number [`NOSYS`]). Which is
/// which is `only_the_process_calls_take_the_locked_path`'s census.
fn nosys_call(raw: u64, compat: bool) -> Option<FreeCall> {
    let n = if compat {
        crate::i386::canonical_number(raw)
    } else if gex_abi::X86 {
        crate::x86::canonical_number(raw)
    } else {
        raw
    };
    if matches!(n, nr::EXECVE | nr::CLONE | nr::CLONE3) {
        return None;
    }
    // A host answers the clocks and the ids before it classes a call
    // ([`clock_call`], [`ident_call`]); one that reaches here is answered
    // the same way.
    let clock_or_id = matches!(
        n,
        nr::CLOCK_GETTIME | nr::CLOCK_GETRES | nr::GETTIMEOFDAY | nr::GETRUSAGE | nr::SCHED_YIELD | nr::TIMES | nr::TIME | nr::GETPID | nr::GETTID
    );
    Some(FreeCall { n: if clock_or_id { CLOCK_OR_ID } else { NOSYS }, raw, a: [0; 6] })
}

/// The number [`free_class`] gives a call the kernel does not make.
pub const NOSYS: u64 = u64::MAX;

/// The number [`free_class`] gives a clock or id call ([`nosys_call`]).
const CLOCK_OR_ID: u64 = u64::MAX - 1;

impl FreeCall {
    /// Its canonical number.
    pub fn number(&self) -> u64 {
        self.n
    }

    /// Its first three arguments.
    pub fn args(&self) -> [u64; 3] {
        [self.a[0], self.a[1], self.a[2]]
    }
}

/// The call [`free_class`] found, whole, for a host that makes it without a
/// kernel lock: the same code as the full
/// call ([`FdIo`], [`sigprocmask`]), the park included, and the signal that
/// is due at a syscall's end.
pub fn free_call(f: &Free<'_>, m: &mut GuestMem<'_>, ctx_addr: u64, call: FreeCall) -> FreeOutcome {
    let FreeCall { n, raw, mut a } = call;
    if let Some(v) = f.io.shared {
        v.catch_up(m);
    }
    let compat = f.io.compat;
    // The poll family in the canonical form: `poll`, `select` and
    // `epoll_wait` as their `p` forms, and a 32-bit ABI's times and
    // `pselect6` mask argument widened, and a 32-bit `futex`'s timeout
    // (none of these has anything to do after the call: `Post::None`).
    let mut post = crate::i386::Post::None;
    if n == NOSYS || n == CLOCK_OR_ID {
        // Nothing to translate: the answer is ENOSYS, or the clock's or
        // the id's, which read the call as the guest made it.
    } else if is_poll(n)
        || (compat
            && (matches!(
                n,
                nr::FUTEX | nr::NANOSLEEP | nr::CLOCK_NANOSLEEP | nr::PREAD64 | nr::PWRITE64 | nr::LSEEK | nr::DUP2 | nr::DUP3 | nr::FCNTL
            ) || is_mapping(n)))
        // The path calls' older forms (`open`, `stat`, `access`, ...) as the
        // `*at` ones, a 32-bit ABI's descriptor sign-extended.
        || ((is_path(n) || is_own_state(n) || is_fd_misc(n) || is_quiet(n, compat) || is_fd_object(n) || is_cred(n) || is_signal_call(n) || is_thread_own(n) || is_mount(n) || is_lifecycle_end(n) || is_thread_clone(n)) && (compat || gex_abi::X86))
    {
        let t = if compat {
            let (t, p) = crate::i386::translate(raw, &mut a, m, ctx_addr + ctx::COMPAT_SCRATCH as u64, ctx_addr, f.signals.blocked());
            post = p;
            t
        } else if gex_abi::X86 {
            crate::x86::translate(raw, &mut a, m, ctx_addr + ctx::SCRATCH as u64, f.signals.blocked())
        } else {
            raw
        };
        debug_assert_eq!(t, n);
    }
    // The number's register doubles as the result register on x86-64.
    if gex_abi::X86 {
        let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
    }
    if let Some(code) = f.procs.halt_code() {
        return FreeOutcome::Exit(code);
    }
    if n == CLOCK_OR_ID {
        let ids = Idents { pid: f.pid as u32, ppid: 0, tid: f.futex.tid };
        let children = || child_cpu_of(f.procs, f.pid);
        if !clock_call(m, ctx_addr, compat, f.futex.clock, f.started, &children) && !ident_call(m, ctx_addr, compat, &ids, f.creds) {
            let _ = set_result(m, ctx_addr, Errno::ENOSYS.as_return());
        }
        let d = if f.signals.has_pending() {
            signal::dispatch(f.signals, m, ctx_addr, compat).unwrap_or(Delivery::None)
        } else {
            Delivery::None
        };
        return FreeOutcome::Done(match d {
            Delivery::Resume(_) if f.wasm32 => Delivery::None,
            d => d,
        });
    }
    let r = if n == NOSYS {
        Err(Errno::ENOSYS)
    } else if n == nr::RT_SIGPROCMASK {
        sigprocmask(f.signals, m, a[0] as u32, a[1], a[2])
    } else if n == nr::FUTEX {
        // As the full call: a wait parks, an operation answers, and a PI
        // lock found held parks once its waiters bit is set.
        if let Some(op) = futex_block(&f.futex, m, a) {
            return FreeOutcome::Blocking(op, None);
        }
        let r = futex_call(&f.futex, m, a);
        if r == Err(Errno::EAGAIN) {
            if let Some(op) = futex_pi_park(&f.futex, m, a) {
                return FreeOutcome::Blocking(op, None);
            }
        }
        r
    } else if n == nr::IOCTL {
        let r = f.io.ioctl(m, f.procs, f.tty, a[0] as i32, a[1], a[2]);
        // A terminal's request may have signalled this process's own group
        // (`TIOCSIG`, `TIOCSWINSZ`): in now, to be delivered at the call's
        // end as the full call's are.
        f.signals.process().collect(f.procs, f.pid, f.creds.ruid());
        r
    } else if is_mapping(n) {
        let Some(r) = f.map.call(m, n, a) else { unreachable!("free_class admitted a call MapIo does not make") };
        r
    } else if is_own_state(n) {
        let now = || {
            let (secs, nanos) = (f.futex.clock)();
            secs as u128 * 1_000_000_000 + nanos as u128
        };
        match n {
            nr::RT_SIGACTION => sigaction(f.signals, m, compat, a[0] as u32, a[1], a[2]),
            nr::RT_SIGPENDING => sigpending(f.signals, m, a[0]),
            nr::SIGALTSTACK => sigaltstack(f.signals, m, compat, a[0], a[1]),
            nr::SETITIMER => itimer(f.signals, now(), m, a[0], a[1], a[2], true),
            nr::GETITIMER => itimer(f.signals, now(), m, a[0], 0, a[1], false),
            nr::ALARM => Ok(alarm(f.signals, now(), a[0])),
            _ => getrandom(f.path.ns, m, a[0], a[1]),
        }
    } else if is_path(n) {
        let Some(r) = f.path.call(m, n, a) else { unreachable!("free_class admitted a call PathIo does not make") };
        r
    } else if is_quiet(n, compat) {
        quiet_call(&f.io, f.procs, f.futex.tid as i32, m, ctx_addr, n, a)
    } else if is_cred(n) {
        cred_call(f.creds, f.procs, f.pid, m, n, a)
    } else if is_mount(n) {
        f.path.mount_call(m, n, a)
    } else if is_thread_clone(n) {
        let ca = if n == nr::CLONE3 { clone3_args(m, a[0], compat) } else { Ok(a) };
        let maker = ThreadMaker {
            procs: f.procs,
            pid: f.pid,
            compat,
            slots: f.slots,
            live: f.live,
            thread_fs: f.thread_fs,
            threads: f.threads,
            blocked: f.signals.blocked(),
            fs: &f.path.fs,
        };
        match ca.and_then(|ca| maker.make(m, ctx_addr, ca)) {
            Ok(req) => {
                let _ = set_result(m, ctx_addr, req.tid as u64);
                return FreeOutcome::SpawnThread { ctx_addr: req.ctx_addr, tid: req.tid };
            }
            Err(e) => Err(e),
        }
    } else if is_lifecycle_end(n) {
        match n {
            nr::EXIT_GROUP => {
                f.map.flush_all_shared(m);
                return FreeOutcome::Exit((a[0] & 0xff) as i32);
            }
            nr::EXIT if one_of_several(f.live) => {
                let tid = f.futex.tid as i32;
                let end = ThreadEnd { thread: f.signals.thread(), threads: f.threads, thread_fs: f.thread_fs, futexes: f.futex.table, procs: f.procs };
                end.end(m, ctx_addr, tid);
                return FreeOutcome::ThreadExit;
            }
            // The last thread's `exit` is the process's.
            nr::EXIT => {
                f.map.flush_all_shared(m);
                return FreeOutcome::Exit((a[0] & 0xff) as i32);
            }
            nr::WAIT4 => return FreeOutcome::Blocking(wait_child(f.procs, f.pid, a), None),
            nr::WAITID => match waitid_child(f.procs, f.io.fds, f.pid, a, compat) {
                Ok(op) => return FreeOutcome::Blocking(op, None),
                Err(e) => Err(e),
            },
            _ => match reboot_halts(a[2]) {
                Some(_) => {
                    f.procs.request_halt(0);
                    return FreeOutcome::Exit(0);
                }
                None => Ok(0),
            },
        }
    } else if is_thread_own(n) {
        thread_own_call(f.signals.thread(), f.thread_fs, &f.path.fs, ctx_addr, f.futex.tid as i32, n, a)
    } else if is_signal_call(n) {
        match n {
            // Returning from a handler restores a whole context, so it
            // does not go through "write the result and continue".
            nr::RT_SIGRETURN => {
                let legacy = compat && raw & 0xffff_ffff == crate::i386::raw::SIGRETURN;
                return match signal::sigreturn(f.signals, m, ctx_addr, compat, legacy) {
                    // The restored context is in ctx: a wasm32 program goes
                    // on there, or makes the call its handler interrupted.
                    Ok(_) if f.wasm32 => {
                        if restart_taken(m, ctx_addr) {
                            FreeOutcome::Again
                        } else {
                            FreeOutcome::Done(Delivery::None)
                        }
                    }
                    Ok(pc) => FreeOutcome::Done(Delivery::Resume(pc)),
                    Err(_) => FreeOutcome::Done(Delivery::Terminate { signal: signal::sig::SIGSEGV, core: true }),
                };
            }
            nr::RT_SIGSUSPEND => match sigsuspend_begin(f.signals, m, a) {
                // Until a signal it can see arrives, its old mask owed
                // back (`BlockOp::SigSuspend`).
                Ok(restore_mask) => return FreeOutcome::Blocking(BlockOp::SigSuspend { restore_mask }, Some(restore_mask)),
                Err(_) => Err(Errno::EFAULT),
            },
            nr::RT_SIGTIMEDWAIT => match sigwait_park(f.signals, m, a) {
                // The host waits; its `Ok(0)` makes the call again, which
                // takes the signal ([`BlockOp::SigWait`]).
                Some(op) => return FreeOutcome::Blocking(op, None),
                None => sigtimedwait(f.signals, m, compat, a),
            },
            nr::PIDFD_OPEN | nr::PIDFD_SEND_SIGNAL => {
                let to = KillTo { signals: f.signals, threads: f.threads, procs: f.procs, pid: f.pid, tid: f.futex.tid as i32, creds: f.creds };
                pidfd_call(f.io.fds, f.procs, &to, n, a)
            }
            _ => {
                let to = KillTo { signals: f.signals, threads: f.threads, procs: f.procs, pid: f.pid, tid: f.futex.tid as i32, creds: f.creds };
                kill_call(&to, n, a)
            }
        }
    } else if is_fd_object(n) {
        // As the full call: a transfer from an empty blocking stream parks
        // until it is readable and is made again then.
        if matches!(n, nr::SENDFILE | nr::SPLICE | nr::COPY_FILE_RANGE) {
            if let Some(op) = f.io.transfer_park(n, a) {
                return FreeOutcome::Blocking(op, None);
            }
        }
        fd_object_call(&f.io, f.futex.clock, m, n, a)
    } else if is_fd_misc(n) {
        match n {
            nr::EPOLL_CTL => f.io.epoll_ctl(m, a[0] as i32, a[1], a[2] as i32, a[3]),
            nr::PIPE2 => f.io.pipe2(m, a[0], a[1]),
            nr::FTRUNCATE => f.io.ftruncate(m, a[0] as i32, a[1]),
            nr::TRUNCATE => f.path.truncate_target(m, a[0]).and_then(|t| f.io.truncate_file(m, &t, a[1])),
            nr::FADVISE64 => f.io.fadvise(a[0] as i32, a[1], a[2], a[3]),
            nr::STATFS | nr::FSTATFS => statfs(m, compat, a[1]),
            // As the kernel's arms: nice 0, and a change taken and ignored.
            nr::GETPRIORITY => Ok(20),
            nr::SETPRIORITY => Ok(0),
            nr::SYSINFO => sysinfo(m, compat, a[0], f.procs, f.futex.clock),
            nr::PROCESS_VM_READV | nr::PROCESS_VM_WRITEV => process_vm(f.procs, m, compat, n == nr::PROCESS_VM_WRITEV, a),
            _ => prlimit(f.rlimits, f.io.fds, f.creds.uid(), m, a[1], a[2], a[3]),
        }
    } else if is_socket_call(n) {
        // As the full call: a blocking `connect`, `accept` or `recvfrom`
        // parks ([`NetIo::park`]).
        if let Some(op) = f.net.park(&f.io, m, n, a) {
            return FreeOutcome::Blocking(op, None);
        }
        // A unix socket's relative name is the calling thread's.
        let cwd = if matches!(n, nr::BIND | nr::CONNECT) { f.path.fs.get() } else { String::new() };
        f.net.call(&f.io, m, n, a, &cwd)
    } else if n == nr::SENDMSG {
        f.io.sendmsg(m, a[0] as i32, a[1])
    } else if n == nr::RECVMSG {
        // As the full call: a blocking socket parks until it is readable,
        // and the call is made again then (`BlockOp::AcceptWait`).
        if let Some(op) = f.io.recvmsg_park(a) {
            return FreeOutcome::Blocking(op, None);
        }
        f.io.recvmsg(m, a[0] as i32, a[1], a[2])
    } else if n == nr::NANOSLEEP || n == nr::CLOCK_NANOSLEEP {
        // As the full call: a sleep parks; a malformed or empty one answers.
        if let Some(op) = sleep_block(m, n, a, f.futex.clock, f.started) {
            return FreeOutcome::Blocking(op, None);
        }
        nanosleep_answer(m, if n == nr::NANOSLEEP { a[0] } else { a[2] })
    } else if is_poll(n) {
        // The call's mask for its duration (as `Kernel::syscall_inner`):
        // back at once when it answers now, before any signal is looked at;
        // through the handler's frame or when the wait is finished
        // otherwise (the host's `restore_sigmask`).
        let mask = match call_sigmask(m, n, &a) {
            Ok(mask) => mask,
            Err(e) => {
                let _ = set_result(m, ctx_addr, e.as_return());
                return FreeOutcome::Done(Delivery::None);
            }
        };
        let old = f.signals.blocked();
        if let Some(want) = mask {
            f.signals.set_blocked(want & !UNBLOCKABLE);
        }
        match f.io.poll(m, n, a) {
            Err(op) => return FreeOutcome::Blocking(op, mask.map(|_| old)),
            Ok(r) => {
                if mask.is_some() {
                    f.signals.set_blocked(old);
                }
                r
            }
        }
    } else {
        if let Some(op) = f.io.blocking_op(m, n, a) {
            return FreeOutcome::Blocking(op, None);
        }
        let Some(r) = f.io.call(m, n, a) else { unreachable!("free_class admitted a call FdIo does not make") };
        // Lost the room or the bytes to another Worker since the check:
        // park, as the full call does.
        // (a terminal's polling read: 0, as the full call answers)
        let r = if r == Err(Errno::EAGAIN) && matches!(n, nr::READ | nr::READV) && f.io.polling_read(a[0]) { Ok(0) } else { r };
        if r == Err(Errno::EAGAIN) {
            if let Some(op) = f.io.eagain_park(m, n, a) {
                return FreeOutcome::Blocking(op, None);
            }
        }
        r
    };
    if r == Err(Errno::EPIPE) && raises_sigpipe(n, &a) {
        // Linux sends it as the writer itself sending it (SI_USER).
        f.signals.raise_from(signal::sig::SIGPIPE, signal::Source { code: signal::SI_USER, pid: f.pid, uid: f.creds.ruid() });
    }
    // `_llseek` answers through memory (`i386::Post`).
    let ret = match r {
        Ok(v) if post != crate::i386::Post::None => encode(Ok(post.finish(m, v))),
        r => encode(r),
    };
    let _ = set_result(m, ctx_addr, ret);
    let d = if f.signals.has_pending() {
        signal::dispatch(f.signals, m, ctx_addr, compat).unwrap_or(Delivery::None)
    } else {
        Delivery::None
    };
    FreeOutcome::Done(match d {
        Delivery::Resume(_) if f.wasm32 => Delivery::None,
        d => d,
    })
}

/// Now on `clk`, in nanoseconds, as `clock_gettime` answers it, for a
/// process that started at `started` (the processor-time clocks count from
/// it): process constants, so no kernel lock.
pub fn clock_nanos_at(clock: ClockFn, started: u128, clk: u64) -> u128 {
    const CLOCK_PROCESS_CPUTIME_ID: u64 = 2;
    const CLOCK_THREAD_CPUTIME_ID: u64 = 3;
    let wall = || {
        let (secs, nanos) = clock();
        secs as u128 * 1_000_000_000 + nanos as u128
    };
    match clk {
        CLOCK_PROCESS_CPUTIME_ID | CLOCK_THREAD_CPUTIME_ID => match crate::proc::cpu_clock(clk == CLOCK_THREAD_CPUTIME_ID) {
            Some(ns) => ns as u128,
            None => wall().saturating_sub(started),
        },
        c if is_monotonic_clock(c) => {
            let (secs, nanos) = monotonic_from(clock());
            secs as u128 * 1_000_000_000 + nanos as u128
        }
        _ => wall(),
    }
}

/// The park of a `nanosleep` or `clock_nanosleep`; `None` leaves the call to
/// [`nanosleep_answer`], which answers at once: EINVAL for a malformed
/// request, 0 for one with nothing left to wait for. The same code in the
/// kernel and in a host that makes the call without it.
pub fn sleep_block(m: &GuestMem<'_>, n: u64, a: [u64; 6], clock: ClockFn, started: u128) -> Option<BlockOp> {
    // `clock_nanosleep` puts the request in a[2]; `nanosleep` in a[0].
    let req = if n == nr::NANOSLEEP { a[0] } else { a[2] };
    if req == 0 {
        return None;
    }
    let secs = m.read_u64(req).ok()? as i64;
    let nanos = m.read_u64(req + 8).ok()?;
    if secs < 0 || nanos >= 1_000_000_000 {
        return None;
    }
    let mut wait = secs as u128 * 1_000_000_000 + nanos as u128;
    // TIMER_ABSTIME: a deadline on the clock, not a duration. It was slept
    // as a duration, so every absolute sleep lasted the cap there was then.
    const TIMER_ABSTIME: u64 = 1;
    if n == nr::CLOCK_NANOSLEEP && a[1] & TIMER_ABSTIME != 0 {
        wait = wait.checked_sub(clock_nanos_at(clock, started, a[0])).filter(|&w| w > 0)?;
    }
    // The whole request: `sleep 2200` returned 0 after 60 s here (the
    // session's stop timer fired 37 minutes early). Bounded only so a
    // deadline added to a clock reading cannot overflow (a century).
    const LONGEST: u128 = 100 * 365 * 24 * 3600 * 1_000_000_000;
    let wait = wait.min(LONGEST) as u64;
    Some(BlockOp::Sleep(std::time::Duration::from_nanos(wait)))
}

/// A sleep's answer when [`sleep_block`] did not park it.
pub fn nanosleep_answer(m: &GuestMem<'_>, req: u64) -> SysResult {
    if req == 0 {
        return Err(Errno::EFAULT);
    }
    let secs = m.read_u64(req)? as i64;
    let nanos = m.read_u64(req + 8)?;
    if secs < 0 || nanos >= 1_000_000_000 {
        return Err(Errno::EINVAL);
    }
    // The sleep itself is `BlockOp::Sleep`, run by the host with the kernel
    // unlocked; reaching here means nothing to wait for: a zero request, or
    // a deadline already past.
    Ok(0)
}

/// What a process's futex calls need of it:
/// its futex table, its memory map's published view (whether a word is
/// mapped), the calling thread's id (a PI lock's owner) and the clock. No
/// kernel lock: the table and the view synchronise themselves.
pub struct FutexCx<'a> {
    pub table: &'a Arc<futex::FutexTable>,
    /// `None` where the host answers "mapped" itself (`mapped_by`).
    pub map: Option<&'a crate::mem::MapView>,
    /// The host's own answer, where it keeps no view (the single-threaded
    /// page, which reads its map directly).
    pub mapped_by: Option<&'a dyn Fn(u64, u64) -> bool>,
    pub tid: u32,
    pub clock: ClockFn,
}

impl FutexCx<'_> {
    /// Whether the futex word at `addr` is inside the process's memory map.
    /// Checked against the map, not by reading: a guest address is a
    /// linear-memory offset, so reading address zero succeeds and proves
    /// nothing.
    fn mapped(&self, addr: u64) -> bool {
        match (self.map, self.mapped_by) {
            (Some(v), _) => v.mapped(addr, 4),
            (None, Some(f)) => f(addr, 4),
            (None, None) => false,
        }
    }

    /// The wall clock as the guest reads it ([`guest_wall`]): the clock an
    /// absolute CLOCK_REALTIME deadline the guest names is on.
    fn now_nanos(&self) -> u128 {
        let (secs, nanos) = guest_wall((self.clock)());
        secs as u128 * 1_000_000_000 + nanos as u128
    }

    fn monotonic_nanos(&self) -> u128 {
        let (secs, nanos) = monotonic_from((self.clock)());
        secs as u128 * 1_000_000_000 + nanos as u128
    }
}

/// The waiting half of `futex` (`FUTEX_WAIT`, `FUTEX_WAIT_BITSET`): the
/// park, when the word still holds what the caller expects and is mapped;
/// `None` for every other operation and for a wait that answers at once
/// ([`futex_call`] then says how).
pub fn futex_block(cx: &FutexCx<'_>, m: &GuestMem<'_>, a: [u64; 6]) -> Option<BlockOp> {
    let operation =
        (a[1] as u32) & !(futex::op::PRIVATE_FLAG | futex::op::CLOCK_REALTIME);
    // A priority-inheritance lock is tried first (`sys_futex`),
    // which parks only a call that found it held (`lock_pi_op`).
    if operation != futex::op::WAIT && operation != futex::op::WAIT_BITSET {
        return None;
    }
    // A futex word outside the process's memory map is `EFAULT`,
    // not a wait: the wake would have to come from an address
    // nothing can write. Checked against the map, not with
    // `GuestMem::read` -- a guest address is a linear-memory
    // offset, so reading address zero succeeds and proves nothing.
    if !cx.mapped(a[0]) {
        return None;
    }
    // Take the queue place BEFORE reading the word. A cooperative
    // host (the browser) parks by snapshotting the generation and
    // polling it; if it snapshotted AFTER the check, a `FUTEX_WAKE`
    // from another thread's Worker landing between the check and the
    // snapshot would bump the generation first and the park would
    // baseline against the already-bumped value -- the wake lost,
    // the thread asleep forever (Mesa's util_queue deadlocked here).
    // Snapshotting first means such a wake either changed the word
    // (caught just below as EAGAIN) or bumped a generation the park
    // will see. The native host holds the queue lock across its
    // check-and-sleep instead and ignores this.
    let parked = cx.table.park(a[0]);
    // The word must still hold what the caller expects, or the
    // wake that changed it has already been and gone and the wait
    // would be permanent. A mismatch declines the wait (EAGAIN via
    // `sys_futex`).
    if m.read_u32(a[0]).ok()? != a[2] as u32 {
        return None;
    }
    let deadline = if a[3] == 0 {
        None
    } else {
        let secs = m.read_u64(a[3]).ok()?;
        let nanos = m.read_u64(a[3] + 8).ok()?;
        let want = secs as u128 * 1_000_000_000 + (nanos % 1_000_000_000) as u128;
        // `FUTEX_WAIT_BITSET` names an absolute moment: on
        // CLOCK_REALTIME with `FUTEX_CLOCK_REALTIME`, else on
        // CLOCK_MONOTONIC (what pthread_cond_clockwait and C++'s
        // steady-clock waits use). Plain `FUTEX_WAIT` really is
        // relative.
        let left = if operation == futex::op::WAIT_BITSET {
            let now = if (a[1] as u32) & futex::op::CLOCK_REALTIME != 0 {
                cx.now_nanos()
            } else {
                cx.monotonic_nanos()
            };
            want.saturating_sub(now)
        } else {
            want
        };
        let left = left.min(3600 * 1_000_000_000);
        Some(std::time::Duration::new(
            (left / 1_000_000_000) as u64,
            (left % 1_000_000_000) as u32,
        ))
    };
    Some(BlockOp::FutexWait {
        table: Arc::clone(cx.table),
        uaddr: a[0],
        val: a[2] as u32,
        deadline,
        parked: Some(parked),
    })
}

/// The park for a `FUTEX_LOCK_PI` that found the lock held ([`futex_call`]
/// set the waiters bit and answered `EAGAIN`).
pub fn futex_pi_park(cx: &FutexCx<'_>, m: &GuestMem<'_>, a: [u64; 6]) -> Option<BlockOp> {
    let operation = (a[1] as u32) & !(futex::op::PRIVATE_FLAG | futex::op::CLOCK_REALTIME);
    if operation != futex::op::LOCK_PI && operation != futex::op::LOCK_PI2 {
        return None;
    }
    // An absolute time: CLOCK_REALTIME for LOCK_PI, the
    // monotonic clock for LOCK_PI2 (unless FUTEX_CLOCK_REALTIME).
    let deadline = if a[3] == 0 {
        None
    } else {
        let secs = m.read_u64(a[3]).ok()?;
        let nanos = m.read_u64(a[3] + 8).ok()?;
        let want = secs as u128 * 1_000_000_000 + (nanos % 1_000_000_000) as u128;
        let monotonic = operation == futex::op::LOCK_PI2 && (a[1] as u32) & futex::op::CLOCK_REALTIME == 0;
        let now = if monotonic { cx.monotonic_nanos() } else { cx.now_nanos() };
        let left = want.saturating_sub(now).min(3600 * 1_000_000_000);
        Some(std::time::Duration::new((left / 1_000_000_000) as u64, (left % 1_000_000_000) as u32))
    };
    Some(BlockOp::FutexLockPi {
        table: Arc::clone(cx.table),
        uaddr: a[0],
        tid: cx.tid,
        deadline,
    })
}

/// `futex`'s answer, every operation but the waiting half of `FUTEX_WAIT`
/// (which [`futex_block`] parks): the same code in the kernel and in a host
/// that makes the call without the kernel.
pub fn futex_call(cx: &FutexCx<'_>, m: &mut GuestMem<'_>, a: [u64; 6]) -> SysResult {
    let (uaddr, raw_op, val) = (a[0], a[1] as u32, a[2] as u32);
    // The private and clock bits change nothing here: the timeout is
    // treated as relative either way.
    let operation = raw_op & !(futex::op::PRIVATE_FLAG | futex::op::CLOCK_REALTIME);
    match operation {
        // The waiting forms reach here only when `blocking_op` refused
        // them, which it does for a futex word the process cannot read.
        futex::op::WAIT | futex::op::WAIT_BITSET => {
            let mapped = cx.mapped(uaddr);
            // `EAGAIN`, not `EINVAL`, for a wait on memory the process
            // has: glibc aborts on a futex error it does not recognise,
            // and `EAGAIN` (word changed, so re-read and loop) is on its
            // recognised list while `EINVAL` is not. An address the
            // process lacks is still `EFAULT`.
            Err(if mapped { Errno::EAGAIN } else { Errno::EFAULT })
        }
        futex::op::WAKE | futex::op::WAKE_BITSET => Ok(cx.table.wake(uaddr, val)),
        // Take the lock if it is free. Held, the waiters bit goes on
        // *before* the caller parks (`EAGAIN`, which `lock_pi_op` turns
        // into the park), as Linux sets it in the kernel: the owner's
        // unlock then comes here and wakes, where without the bit it
        // released the lock in user space and the waiter slept on.
        futex::op::LOCK_PI | futex::op::LOCK_PI2 => {
            if uaddr % 4 != 0 {
                return Err(Errno::EINVAL);
            }
            if !cx.mapped(uaddr) {
                return Err(Errno::EFAULT);
            }
            let tid = cx.tid;
            // The bit and the count together, against an unlock's reading
            // of the count and its store (`FutexTable::pi_lock`).
            let mut pi = cx.table.pi_lock();
            loop {
                let v = gex_atomic::load_u32(m.bytes, uaddr).ok_or(Errno::EFAULT)?;
                match v & futex::pi::TID_MASK {
                    0 => {
                        if gex_atomic::cas_u32(m.bytes, uaddr, v, tid | (v & futex::pi::WAITERS))
                            .ok_or(Errno::EFAULT)?
                            .is_ok()
                        {
                            return Ok(0);
                        }
                    }
                    owner if owner == tid => return Err(Errno::EDEADLK),
                    // A compare-and-swap even when the bit is already on:
                    // it checks the word is still the one just read.
                    _ => {
                        if gex_atomic::cas_u32(m.bytes, uaddr, v, v | futex::pi::WAITERS)
                            .ok_or(Errno::EFAULT)?
                            .is_ok()
                        {
                            // Counted until its wait ends (the host's
                            // `pi_done`).
                            pi.wait(uaddr);
                            return Err(Errno::EAGAIN);
                        }
                    }
                }
            }
        }
        // Release a PI lock this thread owns, and wake one waiter, which
        // takes it in its own `FUTEX_LOCK_PI` loop. The waiters bit stays
        // while anyone waits (the table's count): a waiter that has set it
        // may not be parked yet, and a thread taking the lock in user space
        // meanwhile (from a bare 0) would release it there too, waking
        // nobody. Count and store under the count's lock, which a waiter
        // takes to set the bit and count itself.
        futex::op::UNLOCK_PI => {
            let tid = cx.tid;
            let pi = cx.table.pi_lock();
            let rest = if pi.waiters(uaddr) { futex::pi::WAITERS } else { 0 };
            loop {
                let v = gex_atomic::load_u32(m.bytes, uaddr).ok_or(Errno::EFAULT)?;
                if v & futex::pi::TID_MASK != tid {
                    return Err(Errno::EPERM);
                }
                if gex_atomic::cas_u32(m.bytes, uaddr, v, rest).ok_or(Errno::EFAULT)?.is_ok() {
                    break;
                }
            }
            drop(pi);
            cx.table.wake(uaddr, 1);
            Ok(0)
        }
        futex::op::TRYLOCK_PI => {
            let tid = cx.tid;
            loop {
                let v = gex_atomic::load_u32(m.bytes, uaddr).ok_or(Errno::EFAULT)?;
                match v & futex::pi::TID_MASK {
                    0 => {
                        let new = tid | (v & futex::pi::WAITERS);
                        if gex_atomic::cas_u32(m.bytes, uaddr, v, new).ok_or(Errno::EFAULT)?.is_ok() {
                            return Ok(0);
                        }
                    }
                    owner if owner == tid => return Err(Errno::EDEADLK),
                    _ => return Err(Errno::EAGAIN),
                }
            }
        }
        // Requeue is an optimisation for condition variables; waking
        // everyone is correct, if less efficient.
        futex::op::REQUEUE | futex::op::CMP_REQUEUE => Ok(cx.table.wake(uaddr, u32::MAX)),
        // `FUTEX_WAKE_OP`: atomically apply an op to a second word, wake
        // `val` waiters on `uaddr`, then -- if the old second word compares
        // true -- wake `a[3]` waiters on `uaddr2` (`a[4]`). glibc's condvar
        // signalling uses it; returning ENOSYS here lost a wake and left
        // Qt's pthread-cond waiters asleep forever. The operation and
        // comparison are packed into `a[5]` (see the kernel's `FUTEX_OP`).
        futex::op::WAKE_OP => {
            let nr_wake2 = a[3] as u32;
            let uaddr2 = a[4];
            let encoded = a[5] as u32;
            let mut oper = (encoded >> 28) & 0xf;
            let cmp = (encoded >> 24) & 0xf;
            // `oparg`/`cmparg` are 12-bit signed fields.
            let mut oparg = ((((encoded >> 12) & 0xfff) as i32) << 20) >> 20;
            let cmparg = (((encoded & 0xfff) as i32) << 20) >> 20;
            // `FUTEX_OP_OPARG_SHIFT`: the arg is a shift count, not a value.
            if oper & 8 != 0 {
                oper &= 7;
                oparg = 1i32.wrapping_shl((oparg & 31) as u32);
            }
            let oldval = m.read_u32(uaddr2)? as i32;
            let newval = match oper {
                0 => oparg,                        // SET
                1 => oldval.wrapping_add(oparg),   // ADD
                2 => oldval | oparg,               // OR
                3 => oldval & !oparg,              // ANDN
                4 => oldval ^ oparg,               // XOR
                _ => return Err(Errno::ENOSYS),
            };
            m.write_u32(uaddr2, newval as u32)?;
            let mut woken = cx.table.wake(uaddr, val);
            let fire = match cmp {
                0 => oldval == cmparg, // EQ
                1 => oldval != cmparg, // NE
                2 => oldval < cmparg,  // LT
                3 => oldval <= cmparg, // LE
                4 => oldval > cmparg,  // GT
                5 => oldval >= cmparg, // GE
                _ => return Err(Errno::ENOSYS),
            };
            if fire {
                woken += cx.table.wake(uaddr2, nr_wake2);
            }
            Ok(woken)
        }
        _ => Err(Errno::ENOSYS),
    }
}


// -- I/O on a descriptor table ----------------------------------------------
//
// The bodies of `read`, `write`, `pread64`, `pwrite64`, `readv` and `writev`:
// they need the descriptor table and nothing else of the kernel, so the full
// call and any caller holding only the table run the same code.

/// Write a shared mapping's changed pages over guest `[lo, hi)` back to its
/// file ([`MapIo::flush_shared`]).
fn write_back(m: &GuestMem<'_>, map: &SharedMap, lo: u64, hi: u64) -> Result<(), Errno> {
    // A mapping of a descriptor opened read-only has nothing to
    // write back (Linux refuses it PROT_WRITE): its stores are
    // the process's own, as Wine's relocations of an image view
    // are. Writing them was EBADF, and failed the next mmap.
    if !map.file.writable {
        return Ok(());
    }
    let from = map.start.max(lo);
    let to = (map.start + map.len).min(hi);
    // The pages' hashes first, and nothing more when none changed: a lock
    // given up over an untouched mapping costs no call to the file.
    const P: u64 = crate::mem::PAGE_SIZE;
    let first = (from.max(map.start) - map.start) / P;
    let mut hashes = Vec::new();
    let mut any = false;
    while (map.start + (first + hashes.len() as u64) * P).max(from) < to {
        let page = first + hashes.len() as u64;
        let whole = map.start + page * P;
        let h = page_hash(m.read(whole, (P as usize).min((map.start + map.len - whole) as usize))?);
        any |= map.pages.get(page as usize).map(|s| s.load(std::sync::atomic::Ordering::Relaxed)) != Some(h);
        hashes.push(h);
    }
    if !any {
        return Ok(());
    }
    // Only what the file holds: a shared mapping's page past the
    // end of the file is not the file's (Linux never extends it).
    let known = map.changes.load(std::sync::atomic::Ordering::Acquire);
    let (size, at_start) = match map.file.handle.metadata_changes() {
        Ok((md, c)) => (md.size, if known != 0 { c } else { 0 }),
        Err(_) => (u64::MAX, 0),
    };
    let to = to.min(map.start + size.saturating_sub(map.offset));
    // Whether `pages` stay the file's at a count (`SharedMap::changes`):
    // only if nothing else changed the file since they were (the count
    // has not moved), and after the writes below only if those were
    // the only changes and every page written had its hash kept.
    use std::sync::atomic::Ordering::AcqRel;
    let mut writes = 0u64;
    let mut exact = known != 0 && at_start == known;
    // Only the pages that changed since the file last had them, in
    // runs: another process's writes to the other pages stand.
    let mut run: Option<u64> = None;
    let mut page = first;
    loop {
        let ps = (map.start + page * P).max(from);
        let changed = ps < to && {
            let pe = (map.start + (page + 1) * P).min(to);
            let whole = map.start + page * P;
            let h = hashes[(page - first) as usize];
            let slot = map.pages.get(page as usize);
            let was = slot.map(|s| s.load(std::sync::atomic::Ordering::Relaxed));
            // A partial range writes back only its part, so the page's
            // hash is the file's only when the whole page went.
            if was != Some(h) {
                if ps == whole && pe == (whole + P).min(map.start + map.len) {
                    if let Some(s) = slot {
                        s.store(h, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    exact = false;
                }
            }
            was != Some(h)
        };
        match (changed, run) {
            (true, None) => run = Some(ps),
            (false, Some(rs)) => {
                let at = map.offset + (rs - map.start);
                map.file.pwrite(at, m.read(rs, (ps.min(to) - rs) as usize)?)?;
                writes += 1;
                run = None;
            }
            _ => {}
        }
        if ps >= to {
            break;
        }
        page += 1;
    }
    if known != 0 {
        let now = if exact && writes > 0 { map.file.handle.changes() } else { at_start };
        let _ = map.changes.compare_exchange(known, if exact && now == at_start + writes { now } else { 0 }, AcqRel, std::sync::atomic::Ordering::Relaxed);
    }
    // Other processes' mappings of the file see that it changed; this one's
    // stays up to date with it unless somebody else wrote back meanwhile.
    if writes > 0 {
        let was = map.seen.load(std::sync::atomic::Ordering::Acquire);
        let now = crate::locks::note_written(&map.file.path);
        if now == was + 1 {
            map.seen.store(now, std::sync::atomic::Ordering::Release);
        }
    }
    Ok(())
}

/// What `map`'s file now holds where its pages differ from what they were
/// last read or written back as (another process wrote them back), into
/// the mapping: a page whose hash still matches is left as it is, this
/// process's own stores in it included. Nothing to read when the file's
/// change count is still the one the pages were the file's at.
///
/// `keep_own`: a page this process changed since it last synced it is left
/// as it is (and the mapping is not marked up to date, so the next lock
/// brings it in): for a refresh at a point that is no synchronisation of
/// the program's ([`SharedMapView::catch_up`]).
fn refresh(m: &mut GuestMem<'_>, map: &SharedMap, force: bool, seen: Option<u64>, keep_own: bool) -> Result<(), Errno> {
    use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
    // The count the lock call brought, when it did: unchanged, nothing to ask.
    // Without one, the count alone (no call where the filesystem keeps it
    // here), or for a filesystem that keeps none the write generation.
    if !force {
        let c = seen.unwrap_or_else(|| map.file.handle.changes());
        if if c != 0 { map.changes.load(Acquire) == c } else { map.seen.load(Acquire) == crate::locks::written(&map.file.path) } {
            return Ok(());
        }
    }
    let (md, now) = map.file.handle.metadata_changes()?;
    let written = if now == 0 { crate::locks::written(&map.file.path) } else { 0 };
    if !force && (if now != 0 { map.changes.load(Acquire) == now } else { map.seen.load(Acquire) == written }) {
        return Ok(());
    }
    const P: u64 = crate::mem::PAGE_SIZE;
    // Only what the file holds (a page past its end is not the file's).
    let held = map.len.min(md.size.saturating_sub(map.offset));
    let mut buf = vec![0u8; P as usize];
    let mut page = 0u64;
    let mut kept = false;
    while page * P < held {
        let in_map = P.min(map.len - page * P) as usize;
        let in_file = P.min(held - page * P) as usize;
        buf[..in_map].fill(0);
        let mut got = 0;
        while got < in_file {
            match map.file.handle.read_at(map.offset + page * P + got as u64, &mut buf[got..in_file])? {
                0 => break,
                n => got += n as usize,
            }
        }
        let h = page_hash(&buf[..in_map]);
        let slot = map.pages.get(page as usize);
        let was = slot.map(|s| s.load(Relaxed));
        if was != Some(h) && keep_own && was != Some(page_hash(m.read(map.start + page * P, in_map)?)) {
            kept = true;
        } else if was != Some(h) {
            m.write(map.start + page * P, &buf[..in_map])?;
            if let Some(s) = slot {
                s.store(h, Relaxed);
            }
        }
        page += 1;
    }
    if !kept {
        map.changes.store(now, Release);
        map.seen.store(written, Release);
    }
    Ok(())
}

/// Make this process's shared mappings of `file` and the file agree, at a
/// point where programs synchronise across processes (a lock taken or
/// given up on it, `msync(MS_INVALIDATE)`): its changed pages written back,
/// then what other processes wrote back read in. A mapping is a copy per
/// process (separate memories), so this, not every access, is where two
/// processes' mappings meet -- which is where SQLite's WAL index, and
/// anything else that locks around shared memory, looks for each other's
/// writes. One empty-list test when the process maps nothing shared.
///
/// `seen`: the file's change count as the lock call that granted the lock
/// answered it (the page's global kernel does), which spares asking again.
pub fn sync_shared_file(shared: Option<&SharedMapView>, m: &mut GuestMem<'_>, file: &OpenFile, sync: MapSync, seen: Option<u64>) {
    let Some(view) = shared else { return };
    let maps = view.get();
    if maps.is_empty() || file.handle.is_stream() {
        return;
    }
    for map in maps.iter().filter(|s| s.file.same_file(file)) {
        if sync != MapSync::Acquire {
            let _ = write_back(m, map, 0, u64::MAX);
        }
        if sync != MapSync::Release {
            let _ = refresh(m, map, false, seen, false);
        }
    }
}

/// Which way a lock call synchronises a process's shared mappings with
/// other processes' ([`sync_shared_file`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapSync {
    /// A lock taken: what others wrote back comes in.
    Acquire,
    /// A lock given up: this process's stores go out.
    Release,
}

// -- coherence with shared file mappings ------------------------------------
//
// A `MAP_SHARED` mapping is an eager copy of the file, written back at
// `msync` and `munmap`. Within the process the two must still agree: a
// `read` of a mapped range returns what the mapping holds (its stores not yet
// written back), and a `write` lands in the mapping as well as the file.
// One empty-list test when the process has no shared mappings.

/// The process's shared mappings over `[off, off + n)` of `file`: for each,
/// the guest address of its first byte there, where that is in the range,
/// and how long.
fn shared_spans(shared: Option<&SharedMapView>, file: &OpenFile, off: u64, n: u64) -> Vec<(u64, u64, u64)> {
    let Some(view) = shared else { return Vec::new() };
    if n == 0 || file.handle.is_stream() {
        return Vec::new();
    }
    let maps = view.get();
    let mut out = Vec::new();
    let end = off.saturating_add(n);
    for s in maps.iter().filter(|s| s.file.same_file(file)) {
        let (lo, hi) = (off.max(s.offset), end.min(s.offset + s.len));
        if lo < hi {
            out.push((s.start + (lo - s.offset), lo - off, hi - lo));
        }
    }
    out
}

/// After a read of `n` bytes at `off` into `buf`: the mapped parts as the
/// mappings hold them.
fn shared_after_read(shared: Option<&SharedMapView>, m: &mut GuestMem<'_>, file: &OpenFile, off: u64, buf: u64, n: u64) {
    for (at, r, k) in shared_spans(shared, file, off, n) {
        let _ = m.copy_within(at, buf + r, k as usize);
    }
}

/// After a write of `n` bytes at `off` from `buf`: the same bytes in the
/// mappings.
fn shared_after_write(shared: Option<&SharedMapView>, m: &mut GuestMem<'_>, file: &OpenFile, off: u64, buf: u64, n: u64) {
    for (at, r, k) in shared_spans(shared, file, off, n) {
        let _ = m.copy_within(buf + r, at, k as usize);
    }
}

/// Whether `shared` has any mapping (the fast path's one test).
fn any_shared(shared: Option<&SharedMapView>) -> bool {
    shared.is_some_and(|v| !v.get().is_empty())
}

fn io_write(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, shared: Option<&SharedMapView>) -> SysResult {
    // Straight out of the guest's memory: a guest address is already a
    // linear-memory offset, so the bytes are a slice and a copy would be
    // pure overhead.
    let data = m.read(buf, len as usize)?;
    let file = fds.get(fd)?;
    if !any_shared(shared) || file.handle.is_stream() {
        return file.write(data);
    }
    let (n, off) = {
        let _pos = file.pos_lock();
        let n = file.write_here(data)?;
        (n, file.offset() - n)
    };
    shared_after_write(shared, m, &file, off, buf, n);
    Ok(n)
}

fn io_read(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, shared: Option<&SharedMapView>) -> SysResult {
    // Straight into the guest's memory, no intermediate buffer: the guest
    // picks the length and a copy would allocate that much on every call.
    // Capped as Linux caps it (MAX_RW_COUNT): a regular file gives every
    // byte asked for below that, and programs rely on it (Wine maps a PE
    // image's sections with one pread each; a 16 MiB cap left the rest of
    // a 24 MiB .text zero).
    let len = rw_len(m, buf, len);
    let file = fds.get(fd)?;
    let dst = m.read_mut(buf, len)?;
    if !any_shared(shared) || file.handle.is_stream() {
        return file.read(dst);
    }
    let (n, off) = {
        let _pos = file.pos_lock();
        let n = file.read_here(dst)?;
        (n, file.offset() - n)
    };
    shared_after_read(shared, m, &file, off, buf, n);
    Ok(n)
}

fn io_pread(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, off: u64, shared: Option<&SharedMapView>) -> SysResult {
    // As `io_read`: the guest's buffer is the destination already.
    let len = rw_len(m, buf, len);
    let file = fds.get(fd)?;
    let dst = m.read_mut(buf, len)?;
    let n = file.pread(off, dst)?;
    shared_after_read(shared, m, &file, off, buf, n);
    Ok(n)
}

fn io_pwrite(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, off: u64, shared: Option<&SharedMapView>) -> SysResult {
    let data = m.read(buf, len as usize)?.to_vec();
    let file = fds.get(fd)?;
    let n = file.pwrite(off, &data)?;
    shared_after_write(shared, m, &file, off, buf, n);
    Ok(n)
}

fn io_writev(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, iov: u64, count: u64, compat: bool, shared: Option<&SharedMapView>) -> SysResult {
    let vecs = Kernel::read_iovecs(m, iov, count, compat)?;
    let file = fds.get(fd)?;
    // At most PIPE_BUF bytes to a stream go as one write, which a pipe
    // takes whole or not at all: Linux's guarantee that such a write is
    // never split by another writer's.
    if file.handle.is_stream() {
        if let Some(bytes) = gather_small(m, &vecs) {
            return file.write_here(&bytes);
        }
    }
    // One position for the whole call, as Linux holds `f_pos_lock`.
    let _pos = file.pos_lock();
    let mut total = 0u64;
    for (base, len) in vecs {
        if len == 0 {
            continue;
        }
        let data = match m.read(base, len as usize) {
            Ok(d) => d,
            // Same rule as below: bytes already accepted have to be
            // reported, or the caller sends them twice.
            Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
        };
        let n = match file.write_here(data) {
            Ok(n) => {
                if !file.handle.is_stream() {
                    shared_after_write(shared, m, &file, file.offset() - n, base, n);
                }
                n
            }
            // A pipe or socket that has filled up answers `EAGAIN`. If
            // earlier vectors were already accepted, the call has
            // succeeded partially and must say so: returning the error
            // instead throws away the count for bytes that are already
            // gone, and the caller -- believing it sent nothing --
            // retransmits them. An X client doing that corrupts the
            // request stream.
            Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
        };
        total += n;
        // A short write ends the call; the caller retries the remainder.
        if n < len {
            break;
        }
    }
    Ok(total)
}

/// A vectored write's bytes, in one buffer, when there are at most
/// [`crate::pipe::PIPE_ATOMIC`] of them in more than one vector (one vector
/// is already one write); `None` otherwise, or when a vector cannot be read
/// (the vector-at-a-time path reports that).
fn gather_small(m: &GuestMem<'_>, vecs: &[(u64, u64)]) -> Option<Vec<u8>> {
    let total: u64 = vecs.iter().map(|v| v.1).sum();
    if total as usize > crate::pipe::PIPE_ATOMIC || vecs.iter().filter(|v| v.1 > 0).count() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(total as usize);
    for &(base, len) in vecs {
        out.extend_from_slice(m.read(base, len as usize).ok()?);
    }
    Some(out)
}

/// Put the first `bytes` read by one call over `vectors` in order (a
/// stream's `readv`, read once: see [`BlockOp::ReadStream`]).
pub fn scatter(m: &mut GuestMem<'_>, vectors: &[(u64, u64)], bytes: &[u8]) -> Result<(), Errno> {
    let mut at = 0usize;
    for &(base, len) in vectors {
        if at >= bytes.len() {
            break;
        }
        let k = (len as usize).min(bytes.len() - at);
        m.write(base, &bytes[at..at + k])?;
        at += k;
    }
    Ok(())
}

fn io_readv(fds: &FdTable, m: &mut GuestMem<'_>, fd: i32, iov: u64, count: u64, compat: bool, shared: Option<&SharedMapView>) -> SysResult {
    let vecs = Kernel::read_iovecs(m, iov, count, compat)?;
    let file = fds.get(fd)?;
    // A stream is read once for all the vectors and the bytes scattered:
    // read a vector at a time, another reader of the same pipe could take
    // the bytes between two vectors (a record torn between two readers),
    // which Linux's one read under the pipe's lock never allows.
    if file.handle.is_stream() && vecs.iter().filter(|v| v.1 > 0).count() > 1 {
        let want = vecs.iter().map(|v| v.1).sum::<u64>().min(1 << 24) as usize;
        let mut tmp = vec![0u8; want];
        let n = file.read_here(&mut tmp)? as usize;
        scatter(m, &vecs, &tmp[..n])?;
        return Ok(n as u64);
    }
    // One position for the whole call, as Linux holds `f_pos_lock`.
    let _pos = file.pos_lock();
    let mut total = 0u64;
    for (base, len) in vecs {
        if len == 0 {
            continue;
        }
        let mut tmp = vec![0u8; len as usize];
        let n = match file.read_here(&mut tmp) {
            Ok(n) => n,
            // The vector filled and the next one found the socket empty,
            // which is `EAGAIN`. Reporting that would discard the count
            // for bytes this call has *already taken out of the socket*
            // and written into the caller's memory -- they are consumed,
            // and nothing can give them back. The caller sees "nothing
            // arrived" and waits for data it is already holding.
            //
            // That is the X protocol deadlocking: the server reads a
            // request header into the first vector, loses it here, and
            // waits forever for a request the client believes it sent.
            Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
        };
        m.write(base, &tmp[..n as usize])?;
        if !file.handle.is_stream() {
            shared_after_read(shared, m, &file, file.offset() - n, base, n);
        }
        total += n;
        if n < len {
            break;
        }
    }
    Ok(total)
}

/// What the calling process and thread are called, as its host knows them
/// (the pid since its load or fork, the tid since the thread started): the
/// answers [`ident_call`] gives.
#[derive(Clone, Copy, Debug, Default)]
pub struct Idents {
    pub pid: u32,
    pub ppid: u32,
    pub tid: u32,
}

/// `getpid`, `getppid`, `gettid` answered from `ids`, and the 32-bit
/// `get[e]uid`/`get[e]gid` from the process's `creds`, with no lock,
/// as [`clock_call`] answers the clocks: a
/// host that knows the calling thread's ids calls this before anything
/// else, and the full syscall runs whenever it returns false. `ids.tid ==
/// 0` means the host has none yet, and `ids.ppid == 0` leaves `getppid` to
/// the full call. The 16-bit i386 forms are left to the full call.
pub fn ident_call(m: &mut GuestMem<'_>, ctx_addr: u64, compat: bool, ids: &Idents, creds: &Creds) -> bool {
    if ids.tid == 0 {
        return false;
    }
    let raw = if compat {
        m.read_u64(ctx_addr + ctx::x(0) as u64).unwrap_or(!0) & 0xffff_ffff
    } else {
        m.read_u64(ctx_addr + ctx::SYSCALL_NR as u64).unwrap_or(!0)
    };
    let v = if compat {
        use crate::i386::raw as x;
        match raw {
            x::GETPID => ids.pid,
            x::GETPPID if ids.ppid != 0 => ids.ppid,
            x::GETTID => ids.tid,
            x::GETUID => creds.ruid(),
            x::GETEUID => creds.uid(),
            x::GETGID => creds.rgid(),
            x::GETEGID => creds.gid(),
            _ => return false,
        }
    } else if gex_abi::X86 {
        use crate::x86::raw as x;
        match raw {
            x::GETPID => ids.pid,
            x::GETPPID if ids.ppid != 0 => ids.ppid,
            x::GETTID => ids.tid,
            x::GETUID => creds.ruid(),
            x::GETEUID => creds.uid(),
            x::GETGID => creds.rgid(),
            x::GETEGID => creds.gid(),
            _ => return false,
        }
    } else {
        match raw {
            nr::GETPID => ids.pid,
            nr::GETPPID if ids.ppid != 0 => ids.ppid,
            nr::GETTID => ids.tid,
            nr::GETUID => creds.ruid(),
            nr::GETEUID => creds.uid(),
            nr::GETGID => creds.rgid(),
            nr::GETEGID => creds.gid(),
            _ => return false,
        }
    };
    if gex_abi::X86 {
        let _ = m.write_u64(ctx_addr + ctx::LAST_SYSCALL as u64, raw);
    }
    set_result(m, ctx_addr, v as u64).is_ok()
}

pub fn system_clock() -> (u64, u32) {
    #[cfg(not(target_family = "wasm"))]
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        (now.as_secs(), now.subsec_nanos())
    }
    // Standing still is wrong but it is not a panic, and a host that cares
    // replaces this.
    #[cfg(target_family = "wasm")]
    {
        (0, 0)
    }
}

/// The path a syscall names, for the trace.
///
/// A pointer is not a useful thing to read in a trace of a package manager:
/// half the diagnosis is which file it could not open. Only the calls that
/// take a path at a known argument are decoded, and a bad pointer prints
/// nothing rather than failing.
fn traced_path(m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> String {
    let at = match n {
        nr::OPENAT
        | nr::OPENAT2
        | nr::NEWFSTATAT
        | nr::FACCESSAT
        | nr::FACCESSAT2
        | nr::UNLINKAT
        | nr::MKDIRAT
        | nr::READLINKAT
        | nr::UTIMENSAT
        | nr::FCHOWNAT
        | nr::FCHMODAT
        | nr::STATX => a[1],
        nr::CHDIR | nr::STATFS | nr::MOUNT | nr::UMOUNT2 | nr::TRUNCATE => a[0],
        nr::RENAMEAT | nr::RENAMEAT2 | nr::LINKAT => a[1],
        // `symlinkat` names the link it creates third, not second.
        nr::SYMLINKAT => a[2],
        _ => return String::new(),
    };
    match m.read_cstr(at, 4096) {
        Ok(raw) => format!(" {:?}", String::from_utf8_lossy(&raw)),
        Err(_) => String::new(),
    }
}

/// The terminal's settings and how much of them the kernel exchanges.
///
/// Both live in [`crate::tty`] now, next to the discipline that obeys them --
/// there is no second definition here to fall out of step with it.
pub use crate::tty::{lflag::ECHO, lflag::ICANON, COOKED as COOKED_TERMIOS, KERNEL_TERMIOS};

/// One entry of a `poll` set, kept so the host can re-check it without going
/// back through the kernel.
#[derive(Clone, Debug)]
pub struct PollFd {
    /// Where the guest's `struct pollfd` sits, so `revents` can be written.
    /// Unused when the waiter is `select`, which reports through bitmaps.
    pub addr: u64,
    /// The descriptor, which `select` needs to find its bit again.
    pub fd: i32,
    pub events: u16,
    /// `None` for a descriptor that does not exist, which is `POLLNVAL`.
    pub handle: Option<Arc<dyn crate::vfs::FileHandle>>,
    pub writable: bool,
    /// The epoll set this entry is a watch of (`epoll_wait`), which says
    /// whether an edge-triggered or one-shot watch may report.
    pub epoll: Option<Arc<unix::EpollSet>>,
    /// An edge-triggered watch (`EPOLLET`): only for one is the handle's
    /// wakeup count asked ([`crate::vfs::FileHandle::wakes`]).
    pub edge: bool,
}

/// How a waiter wants its answer written back.
///
/// `poll` and `select` ask the same question and differ only in the shape of
/// the reply: one writes `revents` next to each descriptor, the other sets
/// bits in three bitmaps. Everything between them is shared.
#[derive(Clone, Debug)]
pub enum PollReport {
    /// `revents` in place, in the caller's `pollfd` array.
    Poll,
    /// `epoll_wait`'s output array: one `struct epoll_event` per ready
    /// descriptor, rather than a slot for every one that was asked about.
    Epoll {
        /// Where the array starts.
        out: u64,
        /// How many entries fit.
        max: u32,
        /// The opaque word to hand back for each descriptor.
        data: Vec<(i32, u64)>,
        /// `struct epoll_event` packed in 12 bytes (x86-64's, and the i386
        /// ABI's in any build), rather than 16.
        packed: bool,
        /// The set, whose wait queue the waiter registers on (it is chained
        /// onto every watched descriptor's), and its interest list's
        /// generation when the descriptors were collected: a change means
        /// the list in hand is out of date and the call must be made again.
        set: Arc<unix::EpollSet>,
        generation: u32,
    },
    /// `select`'s three `fd_set`s. A zero address means the caller did not
    /// pass that set.
    Select {
        nfds: u32,
        readfds: u64,
        writefds: u64,
        exceptfds: u64,
    },
}

/// What a `poll` entry would report right now.
pub fn poll_revents(p: &PollFd) -> u16 {
    poll_revents_at(p, edge_wakes(p)).0
}

/// An edge-triggered watch's handle's wakeup count; `None` for any other
/// entry, which never needs it.
fn edge_wakes(p: &PollFd) -> Option<u64> {
    if p.edge {
        p.handle.as_ref().and_then(|h| h.wakes())
    } else {
        None
    }
}

/// [`poll_revents`], with the handle's wakeup count read once before its
/// readiness (`FileHandle::wakes`), which an edge-triggered watch records
/// when it reports: what it reports and that count.
fn poll_revents_at(p: &PollFd, wakes: Option<u64>) -> (u16, Option<u64>) {
    let r = poll_revents_level(p);
    match &p.epoll {
        Some(set) if r != 0 && !set.may_report(p.fd, wakes) => (0, wakes),
        _ => (r, wakes),
    }
}

/// What a level-triggered `poll` entry reports right now.
fn poll_revents_level(p: &PollFd) -> u16 {
    const POLLIN: u16 = 0x001;
    const POLLOUT: u16 = 0x004;
    const POLLNVAL: u16 = 0x020;
    // A negative descriptor is a slot the caller is not using: its events
    // are ignored and it reports nothing, as on Linux. musl's resolver keeps
    // one such slot per query for a TCP retry; reporting POLLNVAL there made
    // every `ppoll` return at once with nothing to read, and the resolver
    // spun until its 5 s timeout -- on every `getaddrinfo` (Wine's
    // `wineboot` resolving the host name was 5 s of each session boot).
    if p.fd < 0 {
        return 0;
    }
    let Some(h) = &p.handle else {
        return POLLNVAL;
    };
    let mut r = 0;
    if p.events & POLLIN != 0 && h.poll_readable() {
        r |= POLLIN;
    }
    // Whether a write would block is asked of the handle every time, not
    // frozen: a parked poll outlives the state it was built from, and a
    // stored "not writable" never moved once the reader drained the buffer.
    if p.events & POLLOUT != 0 && p.writable && h.poll_writable() {
        r |= POLLOUT;
    }
    // Never masked by `events`, as on Linux.
    const POLLHUP: u16 = 0x010;
    if h.poll_hup() {
        r |= POLLHUP;
    }
    // Asked for, unlike `POLLHUP`: the peer shut its sending side (or
    // closed), so what is left to read ends in EOF.
    const POLLRDHUP: u16 = 0x2000;
    if p.events & POLLRDHUP != 0 && h.poll_rdhup() {
        r |= POLLRDHUP;
    }
    // An exceptional condition: asked for, and with `POLLERR`, which never
    // is (a mount changed, to a reader of `/proc/self/mountinfo`).
    const POLLPRI: u16 = 0x002;
    const POLLERR: u16 = 0x008;
    if p.events & POLLPRI != 0 && h.poll_pri() {
        r |= POLLPRI | POLLERR;
    }
    r
}

pub fn poll_ready(fds: &[PollFd]) -> bool {
    fds.iter().any(|p| poll_revents(p) != 0)
}

/// Read-ready and write-ready counts from the last `select`. See the note
/// where they are set.
pub static SELECT_READY: [core::sync::atomic::AtomicU64; 2] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];

pub fn poll_finish(fds: &[PollFd], report: &PollReport, m: &mut GuestMem<'_>) -> SysResult {
    const POLLIN: u16 = 0x001;
    const POLLOUT: u16 = 0x004;
    match report {
        PollReport::Poll => {
            let mut ready = 0u64;
            for p in fds {
                let r = poll_revents(p);
                if r != 0 {
                    ready += 1;
                    if let Some(h) = &p.handle {
                        h.poll_reported(r);
                    }
                }
                m.write_u32(p.addr + 4, ((r as u32) << 16) | p.events as u32)?;
            }
            Ok(ready)
        }
        PollReport::Epoll { out, max, data, packed, .. } => {
            // Only the ready ones, each with the caller's own token, which is
            // the whole difference between this and `poll`.
            let mut n = 0u64;
            for p in fds {
                if n >= *max as u64 {
                    break;
                }
                let (r, wakes) = poll_revents_at(p, edge_wakes(p));
                if r == 0 {
                    continue;
                }
                if let Some(set) = &p.epoll {
                    set.reported(p.fd, wakes);
                }
                if let Some(h) = &p.handle {
                    h.poll_reported(r);
                }
                let token = data
                    .iter()
                    .find(|(fd, _)| *fd == p.fd)
                    .map(|(_, d)| *d)
                    .unwrap_or(0);
                // The same layout `write_epoll_events` uses: packed 12 bytes
                // on x86-64 (the token at offset four), 16 on aarch64. Xvfb
                // read its `data.ptr` from the wrong offset after every wait
                // that had to block, and dereferenced the mask.
                if *packed {
                    let at = out + n * 12;
                    m.write_u32(at, r as u32)?;
                    m.write_u64(at + 4, token)?;
                } else {
                    let at = out + n * 16;
                    m.write_u32(at, r as u32)?;
                    m.write_u32(at + 4, 0)?;
                    m.write_u64(at + 8, token)?;
                }
                n += 1;
            }
            Ok(n)
        }
        PollReport::Select {
            nfds,
            readfds,
            writefds,
            exceptfds,
        } => {
            // `select` reports by rewriting the caller's sets in place, so
            // every set it was given is cleared first: a descriptor that is
            // not ready must come back clear even though it went in set.
            let words = (*nfds as usize).div_ceil(64).max(1);
            let mut out = [[0u64; 16]; 2];
            for p in fds {
                if p.fd < 0 || p.fd as u32 >= *nfds {
                    continue;
                }
                let r = poll_revents(p);
                let (w, bit) = ((p.fd / 64) as usize, p.fd % 64);
                if w >= 16 {
                    continue;
                }
                if r & POLLIN != 0 {
                    out[0][w] |= 1 << bit;
                }
                if r & POLLOUT != 0 {
                    out[1][w] |= 1 << bit;
                }
            }
            let mut ready = 0u64;
            let mut per = [0u64; 2];
            for (i, set) in [*readfds, *writefds].into_iter().enumerate() {
                if set == 0 {
                    continue;
                }
                for w in 0..words.min(16) {
                    m.write_bits64(set + (w as u64) * 8, out[i][w])?;
                    per[i] += out[i][w].count_ones() as u64;
                    ready += out[i][w].count_ones() as u64;
                }
            }
            // Split the count, because the total hides the answer.
            //
            // apt spins on `pselect6(7, ..) = 7` and never reads: seven
            // descriptors are ready and it believes none of them is readable.
            // Whether those seven bits are readability or writability decides
            // whether the bug is in what a pipe reports or in what apt is
            // waiting for. See task 140.
            SELECT_READY[0].store(per[0], core::sync::atomic::Ordering::Relaxed);
            SELECT_READY[1].store(per[1], core::sync::atomic::Ordering::Relaxed);
            // Nothing here ever reports an exceptional condition, but a
            // caller that passed the set expects it cleared.
            if *exceptfds != 0 {
                for w in 0..words.min(16) {
                    m.write_bits64(*exceptfds + (w as u64) * 8, 0)?;
                }
            }
            Ok(ready)
        }
    }
}

/// An executable mapping the guest has just made.
///
/// The host drains these and statically recompiles what is in them. Only the
/// executable arrives through gex's own loader; every shared library is
/// mapped here, by the interpreter, out of guest code -- so without this the
/// static recompiler never sees any of them.
#[derive(Clone, Debug)]
pub struct CodeMap {
    pub start: u64,
    pub end: u64,
    /// The file it came from, for the trace.
    pub path: String,
    /// The file offset mapped at `start` (where a shared object's load bias
    /// is worked out from: the host's ELF units, pe_units::on_elf_maps).
    pub offset: u64,
}

impl CodeMap {
    /// The executable segments of an ELF object gex's own loader placed at
    /// load bias `bias` (the interpreter, at exec): what its `mmap` would have
    /// reported had `ld.so` mapped it, so a host's ELF units see it too.
    pub fn of_placed(path: &str, bytes: &[u8], bias: u64) -> Vec<CodeMap> {
        let rd = |o: usize, n: usize| -> Option<u64> {
            let b = bytes.get(o..o + n)?;
            let mut v = [0u8; 8];
            v[..n].copy_from_slice(b);
            Some(u64::from_le_bytes(v))
        };
        let mut out = Vec::new();
        if bytes.get(0..4) != Some(b"\x7fELF") || bytes.get(4) != Some(&2) {
            return out;
        }
        let (Some(phoff), Some(phentsize), Some(phnum)) = (rd(0x20, 8), rd(0x36, 2), rd(0x38, 2)) else {
            return out;
        };
        for i in 0..phnum as usize {
            let ph = phoff as usize + i * phentsize as usize;
            let (Some(ty), Some(flags), Some(off), Some(vaddr), Some(filesz)) =
                (rd(ph, 4), rd(ph + 4, 4), rd(ph + 8, 8), rd(ph + 16, 8), rd(ph + 32, 8))
            else {
                break;
            };
            if ty == 1 && flags & 1 != 0 && filesz != 0 {
                out.push(CodeMap {
                    start: bias + (vaddr & !0xfff),
                    end: crate::mem::page_align_up(bias + vaddr + filesz),
                    path: path.to_string(),
                    offset: off & !0xfff,
                });
            }
        }
        out
    }
}

/// A `MAP_SHARED` file mapping, and where its bytes belong.
///
/// A file mapping here is an eager copy rather than a page table pointed at
/// the file, so the two are only the same thing until somebody writes. This
/// is what makes them the same thing again.
#[derive(Debug, Clone)]
pub struct SharedMap {
    pub start: u64,
    pub len: u64,
    pub file: Arc<OpenFile>,
    pub offset: u64,
    /// The `mmap`s handed these pages, as (address, length): a second
    /// mapping of the same file (the same description, or another `open` of
    /// it) aliases the first, and its `munmap` only drops it
    /// ([`SharedMaps::unalias`]); the last one's unmaps them all.
    pub holders: Vec<(u64, u64)>,
    /// A hash of each page as the file last had it (read at `mmap`, written
    /// back since): a write-back writes only the pages that differ, so two
    /// processes' mappings of one file, each written back whole before,
    /// stop undoing each other's pages.
    pub pages: Arc<[std::sync::atomic::AtomicU64]>,
    /// The file's [`crate::vfs::FileHandle::changes`] at which `pages` are
    /// its pages, or 0 when that is not known (it keeps no count, or
    /// something else changed it since). What lets the next mapping of the
    /// same range take `pages` rather than hash it again
    /// ([`SharedMaps::recall`]).
    pub changes: Arc<std::sync::atomic::AtomicU64>,
    /// The file's write generation ([`crate::locks::written`]) its pages
    /// were last brought up to date at, for a file whose filesystem keeps
    /// no count (`changes` 0): what [`sync_shared_file`] compares to know
    /// that no other process has written it back since.
    pub seen: Arc<std::sync::atomic::AtomicU64>,
}

/// Page hashes of a file range a shared mapping held, kept past its
/// `munmap` with the file's change count then ([`SharedMaps::remember`]).
#[derive(Debug)]
struct Recent {
    /// The file as [`OpenFile::same_file`] knows it, by path and inode, not
    /// the description: holding that would keep a closed (or unlinked) file.
    path: String,
    ino: u64,
    offset: u64,
    len: u64,
    changes: u64,
    pages: Arc<[std::sync::atomic::AtomicU64]>,
}

/// Unmapped ranges whose page hashes a process keeps: a program that maps a
/// file, unmaps it and maps it again (Wine's views of an index file, a lock
/// at a time) gets the hashes back when the file has not changed.
const RECENT_MAX: usize = 16;

/// The smallest mapping worth asking the file's change count for: two
/// asks (one either side of the read) cost about what hashing a few pages
/// does.
pub const RECALL_MIN: u64 = 8 * crate::mem::PAGE_SIZE;

/// [`SharedMaps::unalias`]'s answer.
#[derive(Debug, PartialEq, Eq)]
pub enum Unalias {
    Keep,
    Whole(u64, u64),
    Plain,
}

/// The identity of code named by its bytes ([`crate::mem::MemoryMap::name_by_content`]):
/// [`crate::mem::CONTENT_ID`], the length and two 64-bit lanes of a hash over
/// the bytes, so two different images agree only by a 128-bit accident.
pub fn content_id(bytes: &[u8]) -> String {
    let (mut a, mut b): (u64, u64) = (0xcbf2_9ce4_8422_2325, 0x6c62_272e_07bb_0142);
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        let x = u64::from_le_bytes(w.try_into().unwrap());
        a = (a ^ x).wrapping_mul(0x100_0000_01b3);
        b = (b ^ x.rotate_left(29)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        b ^= b >> 31;
    }
    for &c in words.remainder() {
        a = (a ^ c as u64).wrapping_mul(0x100_0000_01b3);
        b = (b ^ c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    let mix = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    };
    format!("{}{:x}:{:016x}{:016x}", crate::mem::CONTENT_ID, bytes.len(), mix(a), mix(b))
}

/// The hash a [`SharedMap`] keeps of a page.
pub fn page_hash(bytes: &[u8]) -> u64 {
    // FNV-style over 8-byte words in four lanes, then a final mix: only ever
    // compared with itself, so any build agrees with itself. Four chains
    // rather than one, because each step waits on the multiply before it
    // and a page is hashed whole at every mapping and every write-back.
    const K: u64 = 0x100_0000_01b3;
    let w = |c: &[u8]| u64::from_le_bytes(c.try_into().unwrap());
    let (mut a, mut b, mut c, mut d) = (0xcbf2_9ce4_8422_2325u64, 0x6c62_272e_07bb_0142u64, 0x9e37_79b9_7f4a_7c15u64, 0xff51_afd7_ed55_8ccdu64);
    let mut quads = bytes.chunks_exact(32);
    for q in &mut quads {
        a = (a ^ w(&q[0..8])).wrapping_mul(K);
        b = (b ^ w(&q[8..16])).wrapping_mul(K);
        c = (c ^ w(&q[16..24])).wrapping_mul(K);
        d = (d ^ w(&q[24..32])).wrapping_mul(K);
    }
    let mut h = a ^ b.rotate_left(16) ^ c.rotate_left(32) ^ d.rotate_left(48);
    for &x in quads.remainder() {
        h = (h ^ x as u64).wrapping_mul(K);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^ (h >> 33)
}

/// The page hashes of `bytes`, one per 4 KiB.
pub fn page_hashes(bytes: &[u8]) -> Arc<[std::sync::atomic::AtomicU64]> {
    bytes.chunks(crate::mem::PAGE_SIZE as usize).map(|p| std::sync::atomic::AtomicU64::new(page_hash(p))).collect()
}

/// A process's shared file mappings ([`SharedMap`]), and their published
/// copy ([`SharedMaps::view`]): what a caller without the kernel lock reads
/// (the page's compositor, finding a client's pool). Read as a slice
/// (`Deref`); the two changes, a mapping added and
/// mappings dropped, are this type's, and publish.
#[derive(Debug)]
pub struct SharedMaps {
    list: Vec<SharedMap>,
    view: Arc<SharedMapView>,
    recent: Vec<Recent>,
}

/// A [`SharedMaps`]'s list as other threads read it: replaced whole on every
/// change, so a reader holds one consistent list for as long as it likes.
pub struct SharedMapView {
    cur: gex_atomic::rcu::Published<Arc<[SharedMap]>>,
    /// Whether the process maps any file shared: all a system call of a
    /// process that maps none looks at ([`SharedMapView::catch_up`]).
    any: std::sync::atomic::AtomicBool,
    /// [`crate::locks::write_seq`] when the mappings were last caught up.
    seq: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for SharedMapView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SharedMapView")
    }
}

impl SharedMapView {
    fn new(list: &[SharedMap]) -> SharedMapView {
        SharedMapView {
            cur: gex_atomic::rcu::Published::new(list.into(), crate::file::FD_READER_SLOTS),
            any: std::sync::atomic::AtomicBool::new(!list.is_empty()),
            seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// At a system call's entry, and at the end of a wait (its wake can be
    /// what orders this process after another's write-back): when any process has written a shared
    /// mapping back since this one last looked, this process's mappings of
    /// files that changed take in what they hold now -- except pages it
    /// changed itself since its last sync, which wait for its next lock.
    /// A program that reads shared memory with no lock call of its own in
    /// between (SQLite's second connection in a process whose first holds
    /// the wal-index read lock: SQLite counts those locks itself) still
    /// sees other processes' writes at its next system call. One load for
    /// a process that maps nothing shared; one more (the global kernel's
    /// count, in the page a call) for one that does.
    pub fn catch_up(&self, m: &mut GuestMem<'_>) {
        use std::sync::atomic::Ordering::{AcqRel, Relaxed};
        if !self.any.load(Relaxed) {
            return;
        }
        let now = crate::locks::write_seq();
        let was = self.seq.load(Relaxed);
        if now == was || self.seq.compare_exchange(was, now, AcqRel, Relaxed).is_err() {
            return;
        }
        for map in self.get().iter() {
            let _ = refresh(m, map, false, None, true);
        }
    }

    /// The mappings as they stand.
    pub fn get(&self) -> Arc<[SharedMap]> {
        self.cur.read(crate::sync::me() as usize, Arc::clone)
    }

    /// The mapping of `file`, if it is mapped shared.
    pub fn of(&self, file: &Arc<OpenFile>) -> Option<SharedMap> {
        self.cur.read(crate::sync::me() as usize, |l| l.iter().find(|m| Arc::ptr_eq(&m.file, file)).cloned())
    }

    /// The mapping that holds `file`'s first `len` bytes ([`pool_map`]).
    pub fn pool(&self, file: &Arc<OpenFile>, len: u64) -> Option<SharedMap> {
        self.cur.read(crate::sync::me() as usize, |l| pool_map(l.iter(), file, len).cloned())
    }
}

/// Of the shared mappings `maps`, the one that holds `file`'s first `len`
/// bytes: where a process that sent `file` to its compositor as a pool
/// (`wl_shm.create_pool(fd, len)`, which is the file from offset 0) has the
/// pool's pixels.
///
/// A program may map one file more than once, and at different lengths: Wine
/// maps a 4 KiB view of the file that later backs a 3 MiB window buffer. The
/// first mapping of the file in the list was then the 4 KiB one, too short
/// for the buffer, which the compositor refused, and the window never showed.
/// Which mapping came first depended on what the process had mapped before.
///
/// So any mapping of the file that holds the whole range will do, and the
/// latest one is preferred, since a client maps its buffer just before
/// sending it. A mapping of this description wins over a mapping of another
/// `open` of the same file (the same pages, [`OpenFile::same_file`]), which
/// is looked for only when this description has none. With none that holds
/// the range, the longest from offset 0, so the caller can refuse it as too
/// short.
pub fn pool_map<'a>(
    maps: impl Iterator<Item = &'a SharedMap> + Clone,
    file: &Arc<OpenFile>,
    len: u64,
) -> Option<&'a SharedMap> {
    let from0 = |m: &&SharedMap| m.offset == 0;
    let holds = |m: &&SharedMap| m.len >= len;
    let mine = maps.clone().filter(|m| Arc::ptr_eq(&m.file, file)).filter(from0);
    if let Some(m) = mine.clone().filter(holds).last() {
        return Some(m);
    }
    let same = maps.filter(|m| !Arc::ptr_eq(&m.file, file) && m.file.same_file(file)).filter(from0);
    if let Some(m) = same.clone().filter(holds).last() {
        return Some(m);
    }
    mine.chain(same).max_by_key(|m| m.len)
}

impl SharedMaps {
    pub fn new() -> SharedMaps {
        SharedMaps { list: Vec::new(), view: Arc::new(SharedMapView::new(&[])), recent: Vec::new() }
    }

    /// The published list, the same view for this process's whole life.
    pub fn view(&self) -> &Arc<SharedMapView> {
        &self.view
    }

    fn publish(&self) {
        let l: Arc<[SharedMap]> = self.list.as_slice().into();
        self.view.any.store(!l.is_empty(), std::sync::atomic::Ordering::Relaxed);
        self.view.cur.write(|_| (Some(l), ()));
    }

    pub fn push(&mut self, map: SharedMap) {
        self.list.push(map);
        self.publish();
    }

    /// None any more: an exec's new image maps nothing of the old one's.
    pub fn clear(&mut self) {
        self.list.clear();
        self.recent.clear();
        self.publish();
    }

    /// A forked child's list: the same mappings, with their own view and
    /// their own copy of the page hashes.
    pub fn fork_copy(&self) -> SharedMaps {
        use std::sync::atomic::{AtomicU64, Ordering};
        let list: Vec<SharedMap> = self
            .list
            .iter()
            .map(|s| SharedMap {
                pages: s.pages.iter().map(|h| AtomicU64::new(h.load(Ordering::Relaxed))).collect(),
                changes: Arc::new(AtomicU64::new(s.changes.load(Ordering::Relaxed))),
                seen: Arc::new(AtomicU64::new(s.seen.load(Ordering::Relaxed))),
                ..s.clone()
            })
            .collect();
        let view = Arc::new(SharedMapView::new(&list));
        SharedMaps { list, view, recent: Vec::new() }
    }

    /// Mapping `s` as text, all but its file, for a fork's child in another
    /// memory (the page's Worker) to rebuild with [`SharedMaps::adopt`]:
    /// `start len offset holders hashes`, holders as `addr:len` and the page
    /// hashes in hex, each comma-separated.
    pub fn carry(s: &SharedMap) -> String {
        let holders: Vec<String> = s.holders.iter().map(|(a, l)| format!("{a}:{l}")).collect();
        let pages: Vec<String> = s.pages.iter().map(|h| format!("{:x}", h.load(std::sync::atomic::Ordering::Relaxed))).collect();
        format!("{} {} {} {} {}", s.start, s.len, s.offset, holders.join(","), pages.join(","))
    }

    /// The other half of [`SharedMaps::carry`]: the mapping `text` names,
    /// of `file`. `None` for text that is not [`SharedMaps::carry`]'s.
    pub fn adopt(&mut self, text: &str, file: Arc<OpenFile>) -> Option<()> {
        let mut p = text.split(' ');
        let start = p.next()?.parse().ok()?;
        let len = p.next()?.parse().ok()?;
        let offset = p.next()?.parse().ok()?;
        let holders = p
            .next()?
            .split(',')
            .map(|h| {
                let (a, l) = h.split_once(':')?;
                Some((a.parse().ok()?, l.parse().ok()?))
            })
            .collect::<Option<Vec<(u64, u64)>>>()?;
        let pages = p
            .next()?
            .split(',')
            .filter(|h| !h.is_empty())
            .map(|h| u64::from_str_radix(h, 16).ok().map(std::sync::atomic::AtomicU64::new))
            .collect::<Option<Arc<[std::sync::atomic::AtomicU64]>>>()?;
        // The count is not carried: the child's hashes are taken as unknown
        // against the file's count, and its next mapping hashes afresh.
        self.push(SharedMap { start, len, file, offset, holders, pages, changes: Default::default(), seen: Default::default() });
        Some(())
    }

    /// Keep mapping `s`'s page hashes past its unmap, when its count says
    /// they are the file's pages ([`SharedMap::changes`]); the oldest kept
    /// goes first.
    pub fn remember(&mut self, s: &SharedMap) {
        let changes = s.changes.load(std::sync::atomic::Ordering::Acquire);
        let ino = s.file.ino();
        if changes == 0 || s.len < RECALL_MIN || ino == 0 {
            return;
        }
        self.recent.retain(|r| !(r.path == s.file.path && r.ino == ino && r.offset == s.offset && r.len == s.len));
        if self.recent.len() == RECENT_MAX {
            self.recent.remove(0);
        }
        self.recent.push(Recent { path: s.file.path.clone(), ino, offset: s.offset, len: s.len, changes, pages: Arc::clone(&s.pages) });
    }

    /// The page hashes [`SharedMaps::remember`] kept of `file`'s
    /// `[offset, offset + len)`, if the file's count is still `changes`
    /// (nonzero): its pages are then the ones they were hashed from.
    pub fn recall(&mut self, file: &OpenFile, offset: u64, len: u64, changes: u64) -> Option<Arc<[std::sync::atomic::AtomicU64]>> {
        let ino = file.ino();
        let i = self.recent.iter().position(|r| ino != 0 && r.path == file.path && r.ino == ino && r.offset == offset && r.len == len)?;
        let r = self.recent.remove(i);
        (changes != 0 && r.changes == changes).then_some(r.pages)
    }

    /// One more holder of mapping `i`'s pages (an aliasing `mmap` of
    /// `[at, at + len)`).
    pub fn alias(&mut self, i: usize, at: u64, len: u64) {
        self.list[i].holders.push((at, len));
        self.publish();
    }

    /// What an `munmap` of `[addr, addr + len)` (page-aligned) is to the
    /// mappings it names exactly: one holder's of several ([`Unalias::Keep`]:
    /// nothing to unmap), the last holder's ([`Unalias::Whole`]: unmap the
    /// whole mapping, which earlier holders' unmaps left), or neither.
    pub fn unalias(&mut self, addr: u64, len: u64) -> Unalias {
        let Some(i) = self.list.iter().position(|s| s.holders.contains(&(addr, len))) else {
            return Unalias::Plain;
        };
        let s = &mut self.list[i];
        if s.holders.len() == 1 {
            return Unalias::Whole(s.start, s.len);
        }
        let h = s.holders.iter().position(|&h| h == (addr, len)).unwrap();
        s.holders.remove(h);
        self.publish();
        Unalias::Keep
    }

    /// Keep the mappings `keep` says to; publishes only on a change.
    pub fn retain(&mut self, keep: impl FnMut(&SharedMap) -> bool) {
        let n = self.list.len();
        self.list.retain(keep);
        if self.list.len() != n {
            self.publish();
        }
    }
}

impl Default for SharedMaps {
    fn default() -> SharedMaps {
        SharedMaps::new()
    }
}

impl std::ops::Deref for SharedMaps {
    type Target = [SharedMap];
    fn deref(&self) -> &[SharedMap] {
        &self.list
    }
}

/// A child process the kernel is tracking.
#[derive(Debug)]
pub struct Child {
    pub pid: i32,
    /// Set when the child exits, with the status `wait4` should report.
    pub status: Arc<Mutex<Option<i32>>>,
    /// Signalled when `status` is filled in.
    pub done: Arc<std::sync::Condvar>,
    /// Set once a `wait4` has collected this child, so a second one reports
    /// `ECHILD` rather than blocking forever.
    pub reaped: Arc<std::sync::atomic::AtomicBool>,
}

impl Child {
    pub fn new(pid: i32) -> Child {
        Child {
            pid,
            status: Arc::new(Mutex::new(None)),
            done: Arc::new(std::sync::Condvar::new()),
            reaped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// The handle the host needs to wait on this child.
    pub fn handle(&self) -> ChildHandle {
        ChildHandle {
            pid: self.pid,
            status: Arc::clone(&self.status),
            done: Arc::clone(&self.done),
            reaped: Arc::clone(&self.reaped),
        }
    }
}

/// A child, as the host sees it while waiting with the kernel unlocked.
#[derive(Debug, Clone)]
pub struct ChildHandle {
    pub pid: i32,
    pub status: Arc<Mutex<Option<i32>>>,
    pub done: Arc<std::sync::Condvar>,
    pub reaped: Arc<std::sync::atomic::AtomicBool>,
}

/// A blocking call's state that waits in the kernel for the host to finish
/// the wait (`finish_blocking`), kept per thread: see `Kernel::thread_calls`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingCall {
    pub retry_accept: Option<(i32, u64, u64, u64)>,
    pub write_blocked: bool,
    pub retry_recvmsg: Option<(i32, u64, u64)>,
    pub retry_transfer: Option<(u64, [u64; 6])>,
    pub restore_sigmask: Option<u64>,
    pub retry_sigwait: Option<[u64; 6]>,
    pub retry_lock: Option<i32>,
    /// The parked call's restart class and its syscall's return pc
    /// ([`Kernel::finish_blocking`]).
    pub block_restart: crate::signal::Restart,
    pub block_pc: u64,
}

/// A thread the kernel has prepared and the host must start.
#[derive(Clone, Copy, Debug)]
pub struct SpawnRequest {
    pub ctx_addr: u64,
    pub tid: i32,
}

/// The parts of a kernel's state that belong to one thread, not the process.
///
/// [`Kernel`] is shared between every guest thread, so the fields describing
/// "the thread making this call" cannot live there permanently. They are
/// installed by [`Kernel::enter_thread`] under the same lock that guards the
/// syscall, so what the kernel holds always describes whichever thread
/// currently owns it. What a thread changes about itself (its mask, its
/// `set_tid_address` word) is in its own [`crate::signal::ThreadSig`].
#[derive(Clone, Copy, Debug)]
pub struct ThreadState {
    /// Address of this thread's register context.
    pub ctx_addr: u64,
    pub tid: i32,
}

impl ThreadState {
    pub fn new(ctx_addr: u64, tid: i32) -> ThreadState {
        ThreadState { ctx_addr, tid }
    }
}

impl Kernel {
    /// Make the shared kernel describe `t` for the duration of a syscall.
    pub fn enter_thread(&mut self, t: &ThreadState) {
        self.ctx_addr = t.ctx_addr;
        self.tid = t.tid;
        self.switch_signals(t.tid);
    }

    /// Make `tid`'s signal state the running one (see `thread_signals`).
    fn switch_signals(&mut self, tid: i32) {
        if tid == self.sig_tid {
            return;
        }
        // A new kernel (and a forked child's) holds the state of the one
        // thread it has, whatever its tid: the first thread in adopts it.
        if self.sig_tid == 0 {
            self.sig_tid = tid;
            self.thread_signals.insert(tid, Arc::clone(self.signals.thread()));
            return;
        }
        let call = PendingCall {
            retry_accept: self.retry_accept.take(),
            write_blocked: std::mem::take(&mut self.write_blocked),
            retry_recvmsg: self.retry_recvmsg.take(),
            retry_transfer: self.retry_transfer.take(),
            restore_sigmask: self.restore_sigmask.take(),
            retry_sigwait: self.retry_sigwait.take(),
            retry_lock: self.retry_lock.take(),
            block_restart: std::mem::take(&mut self.block_restart),
            block_pc: std::mem::take(&mut self.block_pc),
        };
        if call != PendingCall::default() {
            self.thread_calls.insert(self.sig_tid, call);
        }
        let call = self.thread_calls.remove(&tid).unwrap_or_default();
        self.retry_accept = call.retry_accept;
        self.write_blocked = call.write_blocked;
        self.retry_recvmsg = call.retry_recvmsg;
        self.retry_transfer = call.retry_transfer;
        self.restore_sigmask = call.restore_sigmask;
        self.retry_sigwait = call.retry_sigwait;
        self.retry_lock = call.retry_lock;
        self.block_restart = call.block_restart;
        self.block_pc = call.block_pc;
        let blocked = self.signals.blocked();
        let next = self
            .thread_signals
            .get_or_insert_with(tid, || Arc::new(crate::signal::ThreadSig::new_thread(blocked)));
        self.signals.switch_to(next);
        self.sig_tid = tid;
    }

    /// Thread `tid`'s signal state, for a host that asks about it or
    /// delivers to it without the kernel ([`signal::ThreadSig`]). The first
    /// thread to ask of a new kernel adopts the state it holds, as
    /// [`Self::switch_signals`] does.
    pub fn thread_sig(&mut self, tid: i32) -> Arc<signal::ThreadSig> {
        if self.sig_tid == 0 {
            self.sig_tid = tid;
        }
        if tid == self.sig_tid {
            let cur = Arc::clone(self.signals.thread());
            self.thread_signals.get_or_insert_with(tid, || Arc::clone(&cur));
            return cur;
        }
        let blocked = self.signals.blocked();
        self.thread_signals
            .get_or_insert_with(tid, || Arc::new(signal::ThreadSig::new_thread(blocked)))
    }

    /// The signal state after `execve` ([`Signals::exec`]): the other
    /// threads are gone, and their state with them. The calling thread's
    /// (its mask and pending signals, which survive) goes on as thread
    /// `tid`: the process's leader, whichever thread called.
    ///
    /// And the fs_struct: the calling thread's, as Linux keeps it through
    /// `execve`. A thread that had its own (`unshare(CLONE_FS)`) makes its
    /// directory the process's; the other threads' go with them.
    pub fn exec_signals(&mut self, tid: i32) {
        if let Some(fs) = self.call_fs.take() {
            if !Arc::ptr_eq(&fs, &self.proc_fs) {
                self.proc_fs.copy_from(&fs);
            }
        }
        self.thread_fs.clear();
        let cwd = self.proc_fs.get();
        if cwd != self.vfs.cwd {
            self.vfs.put_cwd(cwd);
        }
        self.signals.exec();
        let cur = Arc::clone(self.signals.thread());
        // The other threads are gone, and their ids with them.
        for t in self.thread_signals.tids() {
            if t != tid && t != self.pid {
                self.procs.end_tid(t);
            }
        }
        self.thread_signals.clear();
        self.thread_signals.insert(tid, cur);
        self.sig_tid = tid;
    }

    /// Whether a signal is ready for thread `tid` (the host's wait loops ask
    /// for the thread that is waiting, which need not be the one whose state
    /// is loaded).
    /// Whether a signal would end thread `tid`'s wait: one that does
    /// something when delivered (a handler, or a default that is not
    /// "ignore"), as the browser host asks (`would_deliver`). An ignored
    /// SIGCHLD from a pipeline's writer exiting must not turn the reader's
    /// blocking read or splice into EINTR, which Linux never does.
    pub fn signal_pending_for(&mut self, tid: i32) -> bool {
        self.collect_signals();
        self.check_timers();
        if tid == self.sig_tid {
            return self.signals.would_deliver();
        }
        match self.thread_signals.get(tid) {
            Some(t) => t.ready(self.signals.process()),
            None => self.signals.would_deliver(),
        }
    }

    pub fn new(ctx_addr: u64, brk_start: u64, exe_path: String) -> Kernel {
        let fds = FdTable::new();
        // The three standard streams, reported as terminals when the host's
        // are, because that is what decides libc's buffering mode.
        let tty = std::io::IsTerminal::is_terminal(&std::io::stdin());
        let tty_out = std::io::IsTerminal::is_terminal(&std::io::stdout());
        for (stream, name, readable, writable, is_tty) in [
            (Stream::In, "/dev/stdin", true, false, tty),
            (Stream::Out, "/dev/stdout", false, true, tty_out),
            (Stream::Err, "/dev/stderr", false, true, tty_out),
        ] {
            let handle = Arc::new(StdStream::new(stream, is_tty));
            let _ = fds.insert(
                OpenFile::new(handle, name.to_string(), 0, readable, writable),
                false,
            );
        }
        Kernel {
            fds: Arc::new(fds),
            mem: crate::mem::Mm::new(MemoryMap::new(brk_start)),
            vfs: Vfs::new(),
            ctx_addr,
            pid: 1,
            tid: 1,
            exe_path,
            exec_mem64: None,
            argv: Vec::new(),
            envp: Vec::new(),
            creds: Creds::new(1000, 1000),
            signals: Signals::new(),
            futexes: Arc::new(FutexTable::new()),
            trace: false,
            unimplemented: Vec::new(),
            next_ctx_slot: Arc::new(CtxSlots::new(1)),
            live_threads: Arc::new(std::sync::atomic::AtomicU32::new(1)),
            procs: Arc::new(ProcTable::new()),
            pipes: None,
            unix_remote: None,
            ptys: None,
            ppid: 0,
            clock: system_clock,
            retry_accept: None,
            write_blocked: false,
            retry_recvmsg: None,
            retry_transfer: None,
            restore_sigmask: None,
            retry_sigwait: None,
            retry_lock: None,
            started: 0,
            pins: Default::default(),
            place: None,
            syscall_pc: 0,
            block_restart: crate::signal::Restart::Never,
            block_pc: 0,
            syscall_nr: 0,
            thread_signals: Arc::new(crate::signal::ThreadSigs::default()),
            thread_calls: std::collections::HashMap::new(),
            sig_tid: 0,
            compat: false,
            fd_activity: false,
            wasm32: false,
            sigreturn_legacy: false,
            thread_fs: Arc::new(ThreadFs::default()),
            call_fs: None,
            tmpfiles: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            proc_fs: Arc::new(FsCwd::new("/".to_string())),
            rlimits: Arc::new(Rlimits::default()),
            net: None,
            resolver: Arc::new(net::Resolver::new()),
            ports: Arc::new(net::Ports::new()),
            socket_names: Arc::new(unix::SocketNames::new()),
            host_names: Arc::new(unix::SocketNames::new()),
            file_locks: Arc::new(unix::FileLocks::new()),
            tty: Arc::new(crate::tty::TtyRef::new()),
        }
    }

    /// The memory flavour an exec of `path` asks for (`exec_mem64`): its
    /// `GEX_MEM64=1` or `=0` entry, removed from `envp` (every one, so none
    /// reaches the program or its children), else the image's
    /// `user.gex.mem64` attribute, `1` or `0`. `None`: neither says.
    ///
    /// The image is `path`, unless the exec names another with a
    /// `WINEGEX_IMAGE=<path>` entry: Wine execs its loader for every Windows
    /// program, and names the program's `.exe` there, so the attribute on
    /// the `.exe` applies to that program's process alone. That entry is
    /// removed too.
    ///
    /// Neither saying, the image itself may (Wine's loader named on a DOS
    /// path, its first argument, found through WINEPREFIX's dosdevices,
    /// when no WINEGEX_IMAGE names it): an x86-64 one that cannot be
    /// loaded anywhere but above the lowered memory's code space
    /// ([`pe_needs_mem64`]) asks for a 64-bit memory.
    ///
    /// The exec of a Gecko program ([`pe_is_gecko`]) also gets Gecko's
    /// switches that start its children without the sandbox
    /// ([`gecko_without_sandbox`]).
    pub fn exec_flavour(&self, path: &str, argv: &[String], envp: &mut Vec<String>) -> Option<bool> {
        let mut flag = None;
        let mut image = None;
        // Wine's loader started on a program by hand (`wine C:\app.exe`,
        // not by CreateProcess, so no WINEGEX_IMAGE): the program is its
        // first argument, a DOS path, found through the prefix's dosdevices.
        let wine = path.rsplit('/').next().is_some_and(|n| n == "wine" || n == "wine.m64");
        let dos = argv.get(1).filter(|a| wine && a.len() > 3 && a.as_bytes()[1] == b':' && a.as_bytes()[2] == b'\\');
        let by_hand = dos.and_then(|a| {
            let pfx = envp.iter().find_map(|e| e.strip_prefix("WINEPREFIX="))?;
            Some(format!("{pfx}/dosdevices/{}:/{}", a[..1].to_ascii_lowercase(), a[3..].replace('\\', "/")))
        });
        envp.retain(|e| {
            if let Some(v) = e.strip_prefix("GEX_MEM64=") {
                flag = Some(v == "1");
            } else if let Some(v) = e.strip_prefix("WINEGEX_IMAGE=") {
                image = Some(v.to_string());
            } else {
                return true;
            }
            false
        });
        // A Gecko program (Wine's loader on its .exe): its children start
        // without the sandbox (gecko_without_sandbox).
        if wine || image.is_some() {
            let image = image.as_deref().or(by_hand.as_deref()).unwrap_or(path);
            if let Some(head) = self.read_head(image, 4096) {
                let dir = image.rsplit_once('/').map_or(".", |(d, _)| d);
                if pe_is_gecko(&head, |name: &str| self.vfs.metadata(&format!("{dir}/{name}")).is_ok()) {
                    gecko_without_sandbox(envp);
                }
            }
        }
        flag.or_else(|| {
            let mut b = [0u8; 8];
            let op = crate::vfs::XattrOp::Get { name: "user.gex.mem64", buf: &mut b };
            let n = self.vfs.xattr(image.as_deref().unwrap_or(path), true, op).ok()?;
            match &b[..n] {
                b"1" | b"1\n" => Some(true),
                b"0" | b"0\n" => Some(false),
                _ => None,
            }
        })
        .or_else(|| {
            let image = image.as_deref().or(by_hand.as_deref()).unwrap_or(path);
            let head = self.read_head(image, 4096)?;
            let beside = |name: &str| {
                let dir = image.rsplit_once('/').map_or(".", |(d, _)| d);
                self.vfs.metadata(&format!("{dir}/{name}")).is_ok()
            };
            pe_needs_mem64(&head, beside).then_some(true)
        })
    }

    /// Read one range of a file into `out`, returning bytes filled; the rest
    /// of `out` is left alone. Unlike `read_file`, this serves a caller that
    /// wants only a page of a large file rather than the whole thing.
    pub fn read_file_at(&self, path: &str, offset: u64, out: &mut [u8]) -> Option<usize> {
        let h = self.vfs.open(path, 0, 0).ok()?;
        let mut done = 0usize;
        while done < out.len() {
            let n = h.read_at(offset + done as u64, &mut out[done..]).ok()? as usize;
            if n == 0 {
                break;
            }
            done += n;
        }
        Some(done)
    }

    /// The first `n` bytes of `path` (fewer for a shorter file).
    pub fn read_head(&self, path: &str, n: usize) -> Option<Vec<u8>> {
        let mut out = vec![0u8; n];
        let got = self.read_file_at(path, 0, &mut out)?;
        out.truncate(got);
        Some(out)
    }

    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        let h = self.vfs.open(path, 0, 0).ok()?;
        let mut out = Vec::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = h.read_at(out.len() as u64, &mut buf).ok()? as usize;
            if n == 0 {
                return Some(out);
            }
            out.extend_from_slice(&buf[..n]);
            if out.len() > 512 * 1024 * 1024 {
                return None;
            }
        }
    }

    /// Drop the descriptors marked close-on-exec.
    pub fn close_on_exec(&mut self) {
        self.fds.close_on_exec();
    }

    /// Everything `execve` can fail at before the point of no return.
    ///
    /// The strings have to be read here because the address space holding
    /// them is about to go away, and the file has to be checked here because
    /// a failed `execve` *returns*: the guest sees `-ENOENT` and carries on in
    /// the program it already had. Once the host starts building the new
    /// address space there is nothing to return to.
    fn prepare_exec(
        &mut self,
        m: &GuestMem<'_>,
        a: [u64; 6],
    ) -> Result<(String, Vec<String>, Vec<String>), Errno> {
        let (mut path, argv, envp) = self.read_exec_args(m, a)?;
        // The program as `/proc/self/exe` will name it: absolute, from the
        // caller's working directory, as Linux records it. ld.so asserts that
        // the link starts with `/` when it works out `$ORIGIN` (a launcher
        // that execs `./game.x86_64`).
        if !path.starts_with('/') {
            path = crate::vfs::normalize(&path, &self.caller_cwd());
        }
        // `/proc/self/exe` and its kind name this process's files, which the
        // exec replaces before the loader reads its program's segments: the
        // file they name now, not the link.
        for _ in 0..4 {
            if !path.starts_with("/proc/") {
                break;
            }
            match self.vfs.read_link(&path) {
                Ok(to) if to.starts_with('/') => path = to,
                _ => break,
            }
        }
        let (path, argv) = self.resolve_interpreter(path, argv)?;
        // Its head only: the loader reads the whole file once it is past the
        // point of no return (apt's methods are one 14 MB program, read
        // three times over per exec while every check read all of it).
        let image = self.read_head(&path, 64).ok_or(Errno::ENOENT)?;
        if image.len() < 64 || &image[..4] != b"\x7fELF" {
            return Err(Errno::ENOEXEC);
        }
        Ok((path, argv, envp))
    }

    /// Follow a `#!` line, so a script can be executed like a program.
    ///
    /// Linux does this in the kernel, not the shell: `execve` on a file
    /// beginning `#!` runs the named interpreter with the script's path
    /// inserted as its first argument. Without it every interpreted script in
    /// a distribution fails to start, and what the caller sees is not a clear
    /// error but whatever its shell decides to do next -- Debian's
    /// `x11-common` postinst runs a Perl script through debconf, dash got
    /// ENOENT, fell back to reading it as shell, and reported
    /// `use: not found`.
    ///
    /// The line is `#!` then optional spaces, the interpreter, then at most
    /// *one* argument, which is Linux's rule and not an oversight: everything
    /// after the first space is a single argument even if it contains more
    /// spaces. `#!/usr/bin/perl -w` passes `-w`; `#!/usr/bin/env perl -w`
    /// passes `perl -w` as one word, which is why that form does not work on
    /// Linux either.
    ///
    /// Nesting is bounded. A script naming itself would otherwise loop.
    pub fn resolve_interpreter(
        &mut self,
        mut path: String,
        mut argv: Vec<String>,
    ) -> Result<(String, Vec<String>), Errno> {
        for _ in 0..4 {
            let head = self.read_head(&path, 256).ok_or(Errno::ENOENT)?;
            if head.len() >= 4 && &head[..4] == b"\x7fELF" {
                return Ok((path, argv));
            }
            if head.len() < 2 || &head[..2] != b"#!" {
                return Ok((path, argv));
            }
            // Only the first line, and Linux truncates it at 255 bytes.
            let line_end = head[..head.len().min(255)]
                .iter()
                .position(|&c| c == b'\n')
                .unwrap_or(head.len().min(255));
            let line = String::from_utf8_lossy(&head[2..line_end]);
            let line = line.trim_start();
            let (interp, arg) = match line.find(char::is_whitespace) {
                Some(at) => (&line[..at], line[at..].trim()),
                None => (line, ""),
            };
            if interp.is_empty() {
                return Err(Errno::ENOEXEC);
            }
            // argv[0] as the caller gave it is replaced by the script's path,
            // which is what the interpreter is asked to open.
            let mut next = vec![interp.to_string()];
            if !arg.is_empty() {
                next.push(arg.to_string());
            }
            next.push(path.clone());
            next.extend(argv.into_iter().skip(1));
            argv = next;
            path = self.vfs.absolute(interp);
        }
        Err(Errno::ELOOP)
    }

    /// Read `execve`'s path and its two string vectors out of guest memory,
    /// before the address space holding them goes away.
    fn read_exec_args(
        &mut self,
        m: &GuestMem<'_>,
        a: [u64; 6],
    ) -> Result<(String, Vec<String>, Vec<String>), Errno> {
        let bytes_to_string = |b: Vec<u8>| String::from_utf8(b).map_err(|_| Errno::EINVAL);
        let path = bytes_to_string(m.read_cstr(a[0], 4096)?)?;
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        // A 32-bit process hands over vectors of 32-bit pointers.
        let compat = self.compat;
        let read_vec = |mut p: u64| -> Result<Vec<String>, Errno> {
            let mut out = Vec::new();
            // Linux's limits: one string up to MAX_ARG_STRLEN (32 pages),
            // all of them together a quarter of the 8 MiB stack limit;
            // past either, E2BIG. (A make recipe of several KiB, passed to
            // sh -c, is ordinary: automake's test-suite.log rule.)
            const MAX_ARG_STRLEN: usize = 32 * 4096;
            const ARG_MAX: usize = 2 << 20;
            let mut total = 0usize;
            loop {
                if p == 0 {
                    break;
                }
                let ptr = if compat { m.read_u32(p)? as u64 } else { m.read_u64(p)? };
                if ptr == 0 {
                    break;
                }
                let b = m.read_cstr(ptr, MAX_ARG_STRLEN).map_err(|e| if e == Errno::ENAMETOOLONG { Errno::E2BIG } else { e })?;
                total += b.len() + 1 + if compat { 4 } else { 8 };
                if total > ARG_MAX {
                    return Err(Errno::E2BIG);
                }
                out.push(bytes_to_string(b)?);
                p += if compat { 4 } else { 8 };
            }
            Ok(out)
        };
        let argv = read_vec(a[1])?;
        let envp = read_vec(a[2])?;
        Ok((path, argv, envp))
    }

    /// The kernel a forked child starts with.
    ///
    /// Descriptors carry `Arc` handles, so parent and child share the open
    /// file *descriptions* -- offsets included -- exactly as Linux does, while
    /// the table itself is separate so a `close` in one does not affect the
    /// other. The VMA tree is copied because the child gets its own address
    /// space, and the child list starts empty because a grandchild is not the
    /// parent's to reap.
    /// The fs_struct of the thread whose context is at `ctx`: its own, or
    /// the process's.
    pub fn fs_of(&self, ctx: u64) -> Arc<FsCwd> {
        self.thread_fs.of(ctx).unwrap_or_else(|| Arc::clone(&self.proc_fs))
    }

    /// The fs_struct of the thread making the call: in a call, the calling
    /// thread's; after one (the host carrying out its `fork` or `execve`),
    /// the one that call was made with.
    pub fn caller_fs(&self) -> Arc<FsCwd> {
        if self.vfs.in_thread_cwd() {
            return self.fs_of(self.ctx_addr);
        }
        self.call_fs.clone().unwrap_or_else(|| Arc::clone(&self.proc_fs))
    }

    /// The working directory of the calling thread's fs_struct: what a
    /// `fork` (which copies the caller's) starts in.
    pub fn caller_cwd(&self) -> String {
        self.caller_fs().get()
    }

    /// The process's working directory, where it is now.
    pub fn cwd(&self) -> String {
        self.proc_fs.get()
    }

    /// Make `path` the process's working directory (a host starting a
    /// process there), held by identity as `chdir` holds it.
    pub fn set_cwd(&mut self, path: &str) -> Result<(), Errno> {
        self.vfs.set_cwd(path)?;
        let abs = self.vfs.cwd.clone();
        let dir = self.vfs.open(&abs, oflags::O_RDONLY | oflags::O_DIRECTORY, 0).ok();
        self.proc_fs.set(abs, dir);
        Ok(())
    }

    /// A new process's entry in the table, for `clone`/`clone3` without
    /// `CLONE_THREAD`, and with `CLONE_PIDFD` its process descriptor in this
    /// process (close-on-exec), its number written as an `int` at `at`
    /// (`u64::MAX`: a combination Linux refuses, `EINVAL`). Checked before
    /// the entry is made, so a refused call makes no process.
    fn spawn_with_pidfd(&self, m: &mut GuestMem<'_>, flags: u64, at: u64) -> Result<i32, Errno> {
        if flags & clone_flags::PIDFD == 0 {
            return Ok(self.procs.spawn(self.pid));
        }
        if at == u64::MAX {
            return Err(Errno::EINVAL);
        }
        m.write_u32(at, u32::MAX).map_err(|_| Errno::EFAULT)?;
        let pid = self.procs.spawn(self.pid);
        let fd = self.fds.insert(crate::pidfd::open_file(pid, &self.procs, false), true)?;
        let _ = m.write_u32(at, fd as u32);
        Ok(pid)
    }

    /// A fork's child: the process descriptor its parent made for it in the
    /// same `clone` is not its own (Linux makes it after copying the table);
    /// one for any other process it keeps.
    pub fn forget_own_pidfds(&self) {
        for (fd, file, _) in self.fds.entries() {
            if file.handle.pidfd() == Some(self.pid) {
                let _ = self.fds.close(fd);
            }
        }
    }

    pub fn fork_kernel(&self) -> Kernel {
        let mut child = self.fork_kernel_inner();
        let cwd = child.proc_fs.get();
        if cwd != child.vfs.cwd {
            child.vfs.put_cwd(cwd);
        }
        child
    }

    fn fork_kernel_inner(&self) -> Kernel {
        Kernel {
            fds: Arc::new((*self.fds).clone()),
            mem: self.mem.clone(),
            vfs: self.vfs.clone(),
            ctx_addr: self.ctx_addr,
            pid: self.pid,
            tid: self.tid,
            exe_path: self.exe_path.clone(),
            exec_mem64: None,
            argv: self.argv.clone(),
            envp: self.envp.clone(),
            // The child's own ids from here on: a set in either is not the
            // other's.
            creds: self.creds.fork(),
            trace: self.trace,
            // A fork resets pending signals but keeps the dispositions.
            signals: self.signals.fork(),
            // Futexes live in the address space, and the child has its own.
            futexes: Arc::new(FutexTable::new()),
            unimplemented: Vec::new(),
            next_ctx_slot: Arc::new(self.next_ctx_slot.fork()),
            live_threads: Arc::new(std::sync::atomic::AtomicU32::new(1)),
            // The table is the system's, not the process's.
            procs: Arc::clone(&self.procs),
            pipes: self.pipes,
            unix_remote: self.unix_remote,
            ptys: self.ptys,
            ppid: self.pid,
            clock: self.clock,
            // A fork does not inherit its parent's timers.
            retry_accept: None,
            write_blocked: false,
            retry_recvmsg: None,
            retry_transfer: None,
            restore_sigmask: None,
            retry_sigwait: None,
            retry_lock: None,
            // Nor its accumulated time: the child's clock starts now.
            started: self.now_nanos(),
            // The layout a closure build assumed holds in every process of
            // the run: a child that maps a pinned object gets its base.
            pins: self.pins.clone(),
            // The placement outlives any one process: a child that maps the
            // same library must be given the same address its parent was.
            place: self.place.clone(),
            syscall_pc: 0,
            block_restart: crate::signal::Restart::Never,
            block_pc: 0,
            syscall_nr: 0,
            thread_signals: Arc::new(crate::signal::ThreadSigs::default()),
            thread_calls: std::collections::HashMap::new(),
            sig_tid: 0,
            compat: self.compat,
            fd_activity: false,
            wasm32: self.wasm32,
            sigreturn_legacy: false,
            thread_fs: Arc::new(ThreadFs::default()),
            call_fs: None,
            tmpfiles: Arc::new(std::sync::atomic::AtomicU64::new(self.tmpfiles.load(std::sync::atomic::Ordering::Relaxed))),
            // The calling thread's fs_struct, copied: a fork's directory and
            // umask are its caller's.
            proc_fs: Arc::new(self.caller_fs().copy()),
            rlimits: Arc::new(self.rlimits.copy()),
            net: self.net.clone(),
            resolver: Arc::clone(&self.resolver),
            ports: Arc::clone(&self.ports),
            // Named sockets are system-wide: a forked child must still be
            // able to reach a server its parent could.
            socket_names: Arc::clone(&self.socket_names),
            // Host-served names are not: the child's own shim will bind its own
            // compositor and GPU device over these, and connecting to the
            // parent's would hand a client's surfaces to a process that cannot
            // read them.
            host_names: Arc::new(unix::SocketNames::new()),
            file_locks: Arc::clone(&self.file_locks),
            // The terminal is the system's, not this process's: a child
            // shares it rather than inheriting a copy.
            tty: Arc::clone(&self.tty),
        }
    }

    /// `clone`, for the `CLONE_THREAD` case that `pthread_create` uses.
    ///
    /// The child's context is built in shared guest memory from the parent's,
    /// so it inherits everything the ABI does not say otherwise about, and
    /// then differs in exactly what the architecture and the flags require:
    /// `x0` is zero because that is what the child's `clone` returns, `sp` and
    /// the thread pointer come from the caller, and the exclusive monitor
    /// starts clear because a monitor is per-thread and the parent's says
    /// nothing about the child's.
    ///
    /// A `clone` without `CLONE_THREAD` never reaches here; the new-process
    /// case is handled in `syscall`.
    fn sys_clone(&mut self, m: &mut GuestMem<'_>, a: [u64; 6]) -> Result<SpawnRequest, Errno> {
        let fs = self.fs_of(self.ctx_addr);
        self.thread_maker(&fs).make(m, self.ctx_addr, a)
    }

    /// What making a thread of this process needs ([`ThreadMaker`]), for a
    /// creator whose fs_struct is `fs`.
    fn thread_maker<'a>(&'a self, fs: &'a Arc<FsCwd>) -> ThreadMaker<'a> {
        ThreadMaker {
            procs: &self.procs,
            pid: self.pid,
            compat: self.compat,
            slots: &self.next_ctx_slot,
            live: &self.live_threads,
            thread_fs: &self.thread_fs,
            threads: &self.thread_signals,
            blocked: self.signals.blocked(),
            fs,
        }
    }


    fn arg(&self, m: &GuestMem<'_>, i: u8) -> u64 {
        m.read_u64(self.ctx_addr + ctx::SYSCALL_ARGS[i as usize] as u64)
            .unwrap_or(0)
    }

    /// Handle one system call (`svc #0`, or `syscall`).
    pub fn syscall(&mut self, m: &mut GuestMem<'_>, pc: u64) -> SyscallOutcome {
        // Other processes' write-backs into this one's shared mappings.
        self.mem.shared_maps().catch_up(m);
        let out = self.syscall_cwd(m, pc);
        // What a signal that ends the wait makes of it ([`Self::finish_blocking`]).
        if let SyscallOutcome::Blocking(op) = &out {
            self.block_restart = op.restart();
            self.block_pc = pc;
        }
        out
    }

    fn syscall_cwd(&mut self, m: &mut GuestMem<'_>, pc: u64) -> SyscallOutcome {
        // The calling thread's working directory is in `vfs` while its call
        // runs: what its relative paths resolve against. The fs_struct
        // holds it (a `chdir` changes it there, `PathIo`).
        let me = self.ctx_addr;
        // The mount table as it is now: a thread may have mounted or
        // unmounted without the lock since this copy's was taken.
        self.vfs.refresh();
        self.vfs.enter_cwd(self.fs_of(me).get_call());
        let out = self.syscall_inner(m, pc);
        self.vfs.leave_cwd();
        if matches!(out, SyscallOutcome::SpawnProcess { .. } | SyscallOutcome::Exec { .. }) {
            // An `unshare` in the call may have given it its own.
            self.call_fs = Some(self.fs_of(me));
        }
        // The namespace's own directory follows the process's, for what is
        // resolved outside a call.
        let cwd = self.proc_fs.get_call();
        if cwd != self.vfs.cwd {
            self.vfs.put_cwd(cwd);
        }
        out
    }

    fn syscall_inner(&mut self, m: &mut GuestMem<'_>, pc: u64) -> SyscallOutcome {
        self.syscall_pc = pc;
        // An i386-ABI process has its number in eax (`x(0)`) in every build:
        // on an AArch64 build that is a native wasm32 program.
        let nr_slot = if self.compat { ctx::x(0) } else { ctx::SYSCALL_NR };
        let raw = m
            .read_u64(self.ctx_addr + nr_slot as u64)
            .unwrap_or(!0);
        let mut a: [u64; 6] = if self.compat {
            let mut a = [0u64; 6];
            for (i, slot) in crate::i386::ARG_SLOTS.iter().enumerate() {
                a[i] = m.read_u64(self.ctx_addr + *slot as u64).unwrap_or(0) & 0xffff_ffff;
            }
            a
        } else {
            [
                self.arg(m, 0),
                self.arg(m, 1),
                self.arg(m, 2),
                self.arg(m, 3),
                self.arg(m, 4),
                self.arg(m, 5),
            ]
        };
        // The number's register doubles as the result register on x86-64, so
        // keep the raw number where a host can read it after the fact.
        if gex_abi::X86 {
            let _ = m.write_u64(self.ctx_addr + ctx::LAST_SYSCALL as u64, raw);
        }
        // An x86-64 guest's numbering is translated to the canonical one, and
        // its legacy calls rewritten into the `*at`/extended forms.
        let mut post = crate::i386::Post::None;
        let n = if self.compat {
            let (n, p) = crate::i386::translate(
                raw & 0xffff_ffff,
                &mut a,
                m,
                self.ctx_addr + ctx::COMPAT_SCRATCH as u64,
                self.ctx_addr,
                self.signals.blocked(),
            );
            post = p;
            self.sigreturn_legacy = (raw & 0xffff_ffff) == crate::i386::raw::SIGRETURN;
            n
        } else if gex_abi::X86 {
            crate::x86::translate(
                raw,
                &mut a,
                m,
                self.ctx_addr + ctx::SCRATCH as u64,
                self.signals.blocked(),
            )
        } else {
            raw
        };

        self.fd_activity |= wakes_fd_waiters(n) && self.io_may_wake(n, &a);
        self.syscall_nr = n;

        // Returning from a handler restores a whole context, so it cannot go
        // through the normal "write x0 and continue" path.
        if n == nr::RT_SIGRETURN {
            return match self.sigreturn(m) {
                Ok(pc) => {
                    if self.trace {
                        eprintln!(
                            "[gex:{}.{}] rt_sigreturn -> 0x{pc:x}, blocked {:#x}",
                            self.pid, self.tid, self.signals.blocked()
                        );
                    }
                    // The restored context is in ctx, PC included.
                    if self.wasm32 {
                        // A call its handler interrupted that restarts
                        // (`restart_call`): made again now, from the
                        // registers the frame put back, its answer what the
                        // program's stub returns.
                        if restart_taken(m, self.ctx_addr) {
                            return self.syscall(m, pc);
                        }
                        return SyscallOutcome::Continue;
                    }
                    SyscallOutcome::Resume(pc)
                }
                Err(_) => SyscallOutcome::Killed {
                    signal: signal::sig::SIGSEGV,
                    core: true,
                },
            };
        }

        // Someone has halted the machine. Every process notices here and
        // stops, however deep in its own work it was.
        if let Some(code) = self.procs.halt_code() {
            return SyscallOutcome::Exit(code);
        }
        // Anything another process has sent becomes pending here, at the one
        // point where this process is definitely holding its own signal state
        // and not running guest code.
        self.take_sent_signals();
        // `reboot` is two syscalls wearing one number. The shutdown commands
        // do not return -- there is one machine here and it is this process,
        // so halting it and exiting are the same act -- but the Ctrl-Alt-Del
        // commands are ordinary settings, and busybox init issues `CAD_OFF`
        // as its first act. Treating that as a shutdown ends the boot on line
        // one, which is exactly what it did.
        if n == nr::REBOOT {
            let Some(what) = reboot_halts(a[2]) else {
                let _ = set_result(m, self.ctx_addr, 0);
                return SyscallOutcome::Continue;
            };
            if self.trace {
                eprintln!("[gex:{}] reboot({what})", self.pid);
            }
            self.procs.request_halt(0);
            return SyscallOutcome::Exit(0);
        }
        if n == nr::EXECVE {
            return match self.prepare_exec(m, a) {
                Ok((path, argv, mut envp)) => {
                    self.exec_mem64 = self.exec_flavour(&path, &argv, &mut envp);
                    self.map_io().flush_all_shared(m);
                    if self.trace {
                        eprintln!("[gex:{}] execve({path:?}, {} args)", self.pid, argv.len());
                    }
                    SyscallOutcome::Exec { path, argv, envp }
                }
                Err(e) => {
                    let _ = set_result(m, self.ctx_addr, e.as_return());
                    SyscallOutcome::Continue
                }
            };
        }
        // `clone3` (435): glibc >= 2.34 uses it for `pthread_create` (and
        // `posix_spawn`), so without it every threaded program dies at the first
        // thread with EFAULT. It passes a `struct clone_args *` + size rather
        // than registers; read the fields and route through the same paths as
        // `clone`. clone3's `stack` is the allocation base and `stack_size` its
        // length, but `clone`/our thread setup want the *top* (the SP, which
        // grows down), so hand on `stack + stack_size`.
        if n == nr::CLONE3 {
            let ca = match clone3_args(m, a[0], self.compat) {
                Ok(ca) => ca,
                Err(e) => {
                    let _ = set_result(m, self.ctx_addr, e.as_return());
                    return SyscallOutcome::Continue;
                }
            };
            let (flags, sp, tls) = (ca[0], ca[1], ca[3]);
            if flags & clone_flags::THREAD == 0 {
                let pid = match self.spawn_with_pidfd(m, flags, ca[5]) {
                    Ok(pid) => pid,
                    Err(e) => {
                        let _ = set_result(m, self.ctx_addr, e.as_return());
                        return SyscallOutcome::Continue;
                    }
                };
                let _ = set_result(m, self.ctx_addr, pid as u64);
                return SyscallOutcome::SpawnProcess {
                    pid,
                    child_stack: sp,
                    tls: (flags & clone_flags::SETTLS != 0).then_some(tls),
                    vfork: flags & (clone_flags::VM | clone_flags::VFORK) == clone_flags::VM | clone_flags::VFORK,
                };
            }
            return match self.sys_clone(m, ca) {
                Ok(req) => {
                    let _ = set_result(m, self.ctx_addr, req.tid as u64);
                    SyscallOutcome::SpawnThread {
                        ctx_addr: req.ctx_addr,
                        tid: req.tid,
                    }
                }
                Err(e) => {
                    let _ = set_result(m, self.ctx_addr, e.as_return());
                    SyscallOutcome::Continue
                }
            };
        }
        // A clone without CLONE_THREAD is a new process, which only the host
        // can build: it needs a second address space.
        if n == nr::CLONE && a[0] & clone_flags::THREAD == 0 {
            // clone's CLONE_PIDFD descriptor goes where its parent tid would,
            // so the two together are EINVAL
            let pidfd_at = if a[0] & clone_flags::PIDFD != 0 && a[0] & clone_flags::PARENT_SETTID != 0 { u64::MAX } else { a[2] };
            let pid = match self.spawn_with_pidfd(m, a[0], pidfd_at) {
                Ok(pid) => pid,
                Err(e) => {
                    let _ = set_result(m, self.ctx_addr, e.as_return());
                    return SyscallOutcome::Continue;
                }
            };
            // The parent sees the child's pid; the child sees zero, which the
            // host writes into the copy it makes.
            let _ = set_result(m, self.ctx_addr, pid as u64);
            if self.trace {
                eprintln!(
                    "[gex] clone(flags={:#x}, stack={:#x}) = {pid} (new process)",
                    a[0], a[1]
                );
            }
            return SyscallOutcome::SpawnProcess {
                pid,
                child_stack: a[1],
                tls: (a[0] & clone_flags::SETTLS != 0).then_some(a[3]),
                vfork: a[0] & (clone_flags::VM | clone_flags::VFORK) == clone_flags::VM | clone_flags::VFORK,
            };
        }
        if n == nr::CLONE {
            return match self.sys_clone(m, a) {
                Ok(req) => {
                    let _ = set_result(m, self.ctx_addr, req.tid as u64);
                    if self.trace {
                        eprintln!("[gex:{}] clone(...) = {} (new thread)", self.pid, req.tid);
                    }
                    SyscallOutcome::SpawnThread {
                        ctx_addr: req.ctx_addr,
                        tid: req.tid,
                    }
                }
                Err(e) => {
                    let _ = set_result(m, self.ctx_addr, e.as_return());
                    if self.trace {
                        eprintln!("[gex:{}] clone(...) = {}", self.pid, -(e.0 as i64));
                    }
                    SyscallOutcome::Continue
                }
            };
        }
        // `exit` ends one thread; only `exit_group` ends the process. For the
        // last thread the two are the same thing.
        if n == nr::EXIT && one_of_several(&self.live_threads) {
            if self.trace {
                eprintln!("[gex:{}] exit({}) (thread {})", self.pid, a[0] as i32, self.tid);
            }
            let end = ThreadEnd {
                thread: self.signals.thread(),
                threads: &self.thread_signals,
                thread_fs: &self.thread_fs,
                futexes: &self.futexes,
                procs: &self.procs,
            };
            end.end(m, self.ctx_addr, self.tid);
            self.thread_calls.remove(&self.tid);
            return SyscallOutcome::ThreadExit;
        }

        // `epoll_pwait`, `ppoll` and `pselect6` with a signal mask: it is the
        // thread's mask for the call, as `rt_sigsuspend`'s is. A call that
        // waits takes the old one back through the handler's frame, or when
        // it finishes (`finish_blocking`, the hosts' `restore_sigmask`); one
        // that answers at once, below, puts it back before any signal is
        // looked at. foot blocks SIGTERM and takes it only in `epoll_pwait`:
        // without this it never heard one.
        let call_mask = match call_sigmask(m, n, &a) {
            Ok(set) => set,
            Err(e) => {
                let _ = set_result(m, self.ctx_addr, e.as_return());
                return SyscallOutcome::Continue;
            }
        };
        if let Some(want) = call_mask {
            self.restore_sigmask = Some(self.signals.blocked());
            self.signals.set_blocked(want & !UNBLOCKABLE);
        }
        // A syscall that sleeps must do so with the kernel unlocked, or the
        // thread that would wake it can never get in.
        if let Some(op) = self.blocking_op(m, n, a) {
            if self.trace {
                // A parked `poll` says what it is parked on: which
                // descriptors, and what it wants from each. Finding out that
                // apt was waiting to read from a worker it had nothing
                // queued for was the whole of one afternoon.
                if let BlockOp::Poll { fds, .. } = &op {
                    let want: Vec<String> = fds
                        .iter()
                        .map(|p| format!("{}:{:x}", p.fd, p.events))
                        .collect();
                    eprintln!("[gex:{}] {} blocking on [{}]", self.pid, syscall_name(n), want.join(" "));
                } else {
                    eprintln!("[gex:{}.{}] {} blocking", self.pid, self.tid, syscall_name(n));
                }
            }
            self.write_blocked = matches!(op, BlockOp::WriteStream { .. });
            return SyscallOutcome::Blocking(op);
        }

        if self.trace && n == u64::MAX {
            // The raw number, since the canonical one below says only "unknown".
            eprintln!(
                "[gex:{}] raw syscall {} ({}) has no canonical form",
                self.pid,
                raw & 0xffff_ffff,
                if self.compat { "i386" } else { "native" }
            );
        }
        // An x86-64 number with no mapping reaches `dispatch` as `UNKNOWN`;
        // record the guest's own number, which is the one worth looking up.
        let outcome = if (gex_abi::X86 || self.compat) && n == crate::x86::UNKNOWN {
            let before = self.unimplemented.len();
            let r = self.dispatch(m, n, a);
            if self.unimplemented.len() > before {
                self.unimplemented.pop();
                if !self.unimplemented.contains(&raw) {
                    self.unimplemented.push(raw);
                }
            }
            r
        } else {
            self.dispatch(m, n, a)
        };
        // A call with a mask of its own that answered without waiting: the
        // old mask is back before any signal is delivered (Linux's
        // `restore_saved_sigmask_unless`, for a call that is not `EINTR`).
        if call_mask.is_some() {
            if let Some(old) = self.restore_sigmask.take() {
                self.signals.set_blocked(old);
            }
        }
        // A blocking write that still found no room: the check in
        // `blocking_op` and the write are not one step, and with a pipe in
        // the global kernel another Worker's writer can fill it in between
        // (SP: `yes | head -c 20M | wc -c` once counted 65536, head having
        // been told EAGAIN). `writev` has no check at all. Park instead, as
        // Linux would; a `writev` parks on its first non-empty segment and
        // returns that short count, which its callers loop on.
        // A blocking read can lose the same race to another reader of the
        // pipe (two readers of one pipe).
        // A blocking read of a terminal in non-canonical mode with `VMIN` and
        // `VTIME` 0 that found nothing: 0, at once (termios(3)), not a wait.
        let outcome = match outcome {
            Ok(Err(Errno::EAGAIN)) if matches!(n, nr::READ | nr::READV) && self.fd_io().polling_read(a[0]) => Ok(Ok(0)),
            o => o,
        };
        if let Ok(Err(Errno::EAGAIN)) = &outcome {
            if let Some(op) = self.eagain_park(m, n, a) {
                self.write_blocked = matches!(op, BlockOp::WriteStream { .. });
                return SyscallOutcome::Blocking(op);
            }
        }
        if let Ok(Err(Errno::EPIPE)) = &outcome {
            if raises_sigpipe(n, &a) {
                self.signals.raise_from(
                    crate::signal::sig::SIGPIPE,
                    // Linux sends it as the writer itself sending it (SI_USER).
                    crate::signal::Source { code: crate::signal::SI_USER, pid: self.pid, uid: self.creds.ruid() },
                );
            }
        }
        let ret = match &outcome {
            Ok(r) => match r {
                Ok(v) if post != crate::i386::Post::None => encode(Ok(post.finish(m, *v))),
                _ => encode(*r),
            },
            Err(exit) => {
                if self.trace {
                    eprintln!("[gex:{}] exit_group({exit})", self.pid);
                }
                return SyscallOutcome::Exit(*exit);
            }
        };
        if self.trace {
            eprintln!(
                "[gex:{}{}] pc=0x{pc:x} {}({:#x}, {:#x}, {:#x}){} = {}",
                self.pid,
                // The thread, when it is not the process's first.
                if self.tid != self.pid { format!(".{}", self.tid) } else { String::new() },
                syscall_name(n),
                a[0],
                a[1],
                a[2],
                traced_path(m, n, a),
                ret as i64
            );
            // `GEX_STRACE_STACK=N`: the top N stack words too, which in a
            // libc syscall wrapper start with the caller's return address --
            // enough to tell who made a call without a debugger.
            static STACK_WORDS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let words = *STACK_WORDS.get_or_init(|| {
                std::env::var("GEX_STRACE_STACK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
            });
            if words > 0 {
                if let Ok(sp) = m.read_u64(self.ctx_addr + ctx::SP as u64) {
                    let ws: Vec<String> = (0..words)
                        .filter_map(|i| m.read_u64(sp + 8 * i).ok())
                        .map(|w| format!("{w:#x}"))
                        .collect();
                    eprintln!("[gex:{}]   sp={sp:#x}: {}", self.pid, ws.join(" "));
                }
            }
        }
        let _ = set_result(m, self.ctx_addr, ret);

        // A syscall boundary is a safepoint: the context is complete, so a
        // pending signal can be delivered precisely here.
        match self.deliver_pending(m) {
            Delivery::None => SyscallOutcome::Continue,
            // The frame is in ctx (PC = the handler): the stub runs it.
            Delivery::Resume(_) if self.wasm32 => SyscallOutcome::Continue,
            Delivery::Resume(pc) => SyscallOutcome::Resume(pc),
            Delivery::Terminate { signal, core } => SyscallOutcome::Killed { signal, core },
        }
    }

    /// The park for a `read`/`readv` or `write`/`writev` that returned
    /// `EAGAIN` on a stream descriptor that should block (see the call in
    /// `syscall`): the same [`BlockOp`] `blocking_op` would have chosen, on
    /// the first non-empty vector of a vectored call.
    fn eagain_park(&mut self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        if n == nr::FUTEX {
            return self.lock_pi_op(m, a);
        }
        let op = self.fd_io().eagain_park(m, n, a);
        if matches!(op, Some(BlockOp::FileLock { .. })) {
            self.retry_lock = Some(a[0] as i32);
        }
        op
    }

    /// The park for a `FUTEX_LOCK_PI` that found the lock held (`sys_futex`
    /// set the waiters bit and answered `EAGAIN`).
    fn lock_pi_op(&mut self, m: &GuestMem<'_>, a: [u64; 6]) -> Option<BlockOp> {
        let mapped = |addr: u64, len: u64| self.mem.lock().vma_at(addr).is_some_and(|v| addr + len <= v.end);
        futex_pi_park(&self.futex_cx(&mapped), m, a)
    }

    /// The descriptor half of this kernel ([`FdIo`]).
    pub fn fd_io(&self) -> FdIo<'_> {
        FdIo {
            fds: &self.fds,
            compat: self.compat,
            wasm32: self.wasm32,
            ns: &self.vfs,
            pid: self.pid,
            creds: &self.creds,
            file_locks: &self.file_locks,
            pipes: self.pipes,
            shared: Some(self.mem.shared_maps()),
        }
    }

    /// The socket calls' part of this kernel ([`NetIo`]).
    pub fn net_io(&self) -> NetIo<'_> {
        NetIo {
            net: self.net.as_ref(),
            resolver: &self.resolver,
            ports: &self.ports,
            unix_remote: self.unix_remote,
            socket_names: &self.socket_names,
            host_names: &self.host_names,
        }
    }

    /// The path calls' part of this kernel ([`PathIo`]): the namespace
    /// itself, whose working directory is the calling thread's while its
    /// call runs (`syscall_cwd`).
    pub fn path_io(&self) -> PathIo<'_> {
        PathIo {
            ns: &self.vfs,
            view: self.vfs.view(),
            // In a call, the calling thread's (entered into the namespace
            // for it); outside one, the process's.
            map: self.map_io(),
            compat: self.compat,
            pid: self.pid,
            creds: &self.creds,
            procs: &self.procs,
            ptys: self.ptys,
            tmpfiles: &self.tmpfiles,
            fs: std::borrow::Cow::Owned(self.fs_of(self.ctx_addr)),
            clock: self.clock,
            socket_names: &self.socket_names,
            unix_remote: self.unix_remote,
        }
    }

    /// The mapping calls' part of this kernel ([`MapIo`]).
    pub fn map_io(&self) -> MapIo<'_> {
        MapIo { mem: &self.mem, fds: &self.fds, pins: &self.pins, place: self.place.as_ref(), pc: self.syscall_pc }
    }

    /// Note that `[addr, addr + len)` no longer holds the code it did
    /// ([`MapIo::retire_code`]).
    pub fn retire_code(&self, addr: u64, len: u64) {
        self.map_io().retire_code(addr, len)
    }

    /// The pinned base for a path a closure build named ([`MapIo::pin_for`]).
    pub fn pin_for(&self, path: &str) -> Option<u64> {
        self.map_io().pin_for(path)
    }

    /// The blocking half of a syscall, if it has one.
    ///
    /// Recognised before dispatch, because these are exactly the calls whose
    /// whole purpose is to wait for another thread.
    fn blocking_op(&mut self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        match n {
            // `rt_sigsuspend`: replace the blocked mask, wait for a signal
            // the caller can now see, put the mask back.
            //
            // The swap and the wait have to look atomic to the guest -- that
            // is the entire reason the call exists instead of a
            // `sigprocmask` followed by a `pause` -- so the mask changes
            // here, before the kernel is unlocked, and is restored in
            // `finish_blocking`.
            //
            // Without it, `timeout 3 sleep 30` ran for the full thirty
            // seconds and then reported success: musl saw `ENOSYS` and spun,
            // 1.6 million syscalls' worth, until the child ended by itself.
            nr::RT_SIGSUSPEND => {
                let restore_mask = sigsuspend_begin(&self.signals, m, a).ok()?;
                self.restore_sigmask = Some(restore_mask);
                Some(BlockOp::SigSuspend { restore_mask })
            }
            // `rt_sigtimedwait` with nothing of its set pending waits for one
            // (it had answered `EAGAIN` at once, whatever the timeout); the
            // signal is taken in `finish_blocking`.
            nr::RT_SIGTIMEDWAIT => {
                let op = sigwait_park(&self.signals, m, a)?;
                self.retry_sigwait = Some(a);
                Some(op)
            }
            nr::CONNECT => {
                let bytes = m.read(a[1], (a[2] as usize).min(128)).ok()?.to_vec();
                let (ip, port) = net::parse_sockaddr(&bytes).ok()?;
                let file = self.fds.get(a[0] as i32).ok()?;
                let socket = self.socket_of(a[0] as i32).ok()?;
                // Start it here, whichever way this goes: the non-blocking
                // path reaches `sys_connect`, which finds the connection
                // already under way and only reports on it.
                let started = socket.connect(ip, port);
                let waits = file.would_block() && started == Err(Errno::EINPROGRESS);
                waits.then_some(BlockOp::Connect { socket })
            }
            // `epoll_pwait` waits, and a version of it that always returned
            // at once turned every event loop into a spin. Xwayland's is the
            // one that showed it: it made progress, but only when the timing
            // happened to suit it.
            nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2 | nr::PPOLL | nr::PSELECT6 => self.fd_io().poll_park(m, n, a),
            nr::ACCEPT | nr::ACCEPT4 => {
                let file = self.fds.get(a[0] as i32).ok()?;
                let socket: Arc<dyn crate::vfs::FileHandle> = Arc::clone(&file.handle);
                if !file.would_block() || socket.poll_readable() {
                    return None;
                }
                let flags = if n == nr::ACCEPT4 { a[3] } else { 0 };
                self.retry_accept = Some((a[0] as i32, a[1], a[2], flags));
                Some(BlockOp::AcceptWait { socket })
            }
            nr::READ | nr::READV | nr::WRITE | nr::WRITEV => self.fd_io().blocking_op(m, n, a),
            // A transfer whose input is an empty blocking stream (`cat f |
            // grep x`: GNU grep splices its stdin pipe) waits for it to be
            // readable and then does the real transfer (`retry_transfer`).
            nr::SPLICE | nr::SENDFILE | nr::COPY_FILE_RANGE => {
                let op = self.fd_io().transfer_park(n, a)?;
                self.retry_transfer = Some((n, a));
                Some(op)
            }
            // `recvfrom` (and `recv`) on an empty blocking socket waits for
            // it to be readable, then reads under the lock, as `recvmsg`
            // does (by `retry_transfer`). It had answered EAGAIN.
            nr::RECVFROM => {
                let file = self.fds.get(a[0] as i32).ok()?;
                if a[3] & Self::MSG_DONTWAIT != 0 || !file.would_block() {
                    return None;
                }
                self.retry_transfer = Some((n, a));
                Some(BlockOp::AcceptWait {
                    socket: Arc::clone(&file.handle),
                })
            }
            nr::RECVMSG => {
                // Wait until it is readable, then do the real recvmsg (see
                // `retry_recvmsg`), the way `accept` does.
                let op = self.fd_io().recvmsg_park(a)?;
                self.retry_recvmsg = Some((a[0] as i32, a[1], a[2]));
                Some(op)
            }
            nr::FUTEX => {
                let mapped = |addr: u64, len: u64| self.mem.lock().vma_at(addr).is_some_and(|v| addr + len <= v.end);
                futex_block(&self.futex_cx(&mapped), m, a)
            }
            nr::WAIT4 => Some(wait_child(&self.procs, self.pid, a)),
            // Its argument errors are the dispatch's (below).
            nr::WAITID => waitid_child(&self.procs, &self.fds, self.pid, a, self.compat).ok(),
            nr::NANOSLEEP | nr::CLOCK_NANOSLEEP => {
                // The clocks as of now, the processor-time ones counted from
                // a start this sets if it has none yet.
                let _ = self.cpu_nanos();
                sleep_block(m, n, a, self.clock, self.started)
            }
            _ => None,
        }
    }

    /// Finish a syscall whose blocking half the host has now run.
    /// Whether a syscall since the last call may have changed descriptor
    /// readiness ([`Kernel::fd_activity`]), clearing it.
    pub fn take_fd_activity(&mut self) -> bool {
        std::mem::take(&mut self.fd_activity)
    }

    pub fn finish_blocking(&mut self, m: &mut GuestMem<'_>, r: SysResult) -> SyscallOutcome {
        // What other processes wrote back while the call waited: the wait's
        // end can be what orders this process after their writes (a pipe
        // read of a byte written after a commit), with no system call of
        // its own before it reads the mapping ([`SharedMapView::catch_up`]).
        self.mem.shared_maps().catch_up(m);
        // An `accept` that waited does its real work here, now that the
        // kernel is locked again and there is a descriptor table to put the
        // new socket in.
        let r = match self.retry_accept.take() {
            Some((fd, addr, len, flags)) if r.is_ok() => {
                accept(&self.fd_io(), m, fd, addr, len, flags)
            }
            _ => r,
        };
        let r = match self.retry_recvmsg.take() {
            Some((fd, msg, flags)) if r.is_ok() => self.sys_recvmsg(m, fd, msg, flags),
            _ => r,
        };
        let r = match self.retry_transfer.take() {
            Some((n, a)) if r.is_ok() => self.fd_io().transfer_syscall(m, n, &a),
            _ => r,
        };
        let r = match self.retry_sigwait.take() {
            Some(a) if r.is_ok() => sigtimedwait(&self.signals, m, self.compat, a),
            _ => r,
        };
        if let (Some(fd), Ok(_)) = (self.retry_lock.take(), &r) {
            if let Ok(file) = self.fds.get(fd) {
                sync_shared_file(Some(self.mem.shared_maps()), m, &file, MapSync::Acquire, None);
            }
        }
        // A wait a signal ended that Linux would restart (signal(7)): the
        // handler runs, then the call again ([`restart_call`]).
        let class = std::mem::replace(&mut self.block_restart, crate::signal::Restart::Never);
        let ret_pc = std::mem::take(&mut self.block_pc).wrapping_add(gex_abi::SYSCALL_INST_BYTES);
        if r == Err(Errno::EINTR) {
            self.collect_signals();
            self.check_timers();
            if self.signals.restarts(class) {
                if let Some(back) = restart_call(m, self.ctx_addr, ret_pc, self.wasm32) {
                    self.restore_sigmask = None;
                    self.write_blocked = false;
                    match self.deliver_pending(m) {
                        Delivery::Resume(_) if self.wasm32 => return SyscallOutcome::Continue,
                        Delivery::Resume(target) => return SyscallOutcome::Resume(target),
                        Delivery::Terminate { signal, core } => return SyscallOutcome::Killed { signal, core },
                        Delivery::None if !self.wasm32 => return SyscallOutcome::Resume(back),
                        // See `restart_call`: `EINTR` after all.
                        Delivery::None => restart_undo(m, self.ctx_addr),
                    }
                }
            }
        }
        // `rt_sigsuspend` waited with a mask of the caller's choosing. The
        // old mask goes back *after* the handler runs, which means the frame
        // has to carry it -- restoring it here instead would block the very
        // signal that ended the wait, leaving it pending forever.
        let suspended = self.restore_sigmask.take();
        if std::mem::take(&mut self.write_blocked) && r == Err(Errno::EPIPE) {
            self.signals.raise_from(
                crate::signal::sig::SIGPIPE,
                // Linux sends it as the writer itself sending it (SI_USER).
                crate::signal::Source { code: crate::signal::SI_USER, pid: self.pid, uid: self.creds.ruid() },
            );
        }
        let ret = encode(r);
        let _ = set_result(m, self.ctx_addr, ret);
        if let Some(mask) = suspended {
            self.signals.set_suspend_restore(Some(mask));
        }
        let delivery = self.deliver_pending(m);
        // Nothing was delivered, so no frame took the mask: put it back here.
        if let Some(mask) = self.signals.take_suspend_restore() {
            self.signals.set_blocked(mask);
        }
        if self.trace {
            eprintln!(
                "[gex:{}.{}] after blocking: delivery {:?}, pending {:#x}, blocked {:#x}",
                self.pid,
                self.tid,
                delivery,
                self.signals.pending_raw(),
                self.signals.blocked()
            );
        }
        match delivery {
            Delivery::None => SyscallOutcome::Continue,
            Delivery::Resume(_) if self.wasm32 => SyscallOutcome::Continue,
            Delivery::Resume(pc) => SyscallOutcome::Resume(pc),
            Delivery::Terminate { signal, core } => SyscallOutcome::Killed { signal, core },
        }
    }

    /// The dispatch table. `Err(code)` means the process is exiting.
    fn dispatch(&mut self, m: &mut GuestMem<'_>, n: u64, a: [u64; 6]) -> Result<SysResult, i32> {
        let r: SysResult = match n {
            nr::EXIT | nr::EXIT_GROUP => {
                self.map_io().flush_all_shared(m);
                return Err((a[0] & 0xff) as i32);
            }
            // The calls that need nothing of the process ([`quiet_call`],
            // the free path's code).
            n if is_quiet(n, self.compat) => quiet_call(&self.fd_io(), &self.procs, self.tid, m, self.ctx_addr, n, a),

            // -- x86-64 legacy forms (see `crate::x86`) ----------------------------
            // Only `x86::translate` produces these numbers; an AArch64 guest
            // issuing one raw falls through to ENOSYS below.
            nr::DUP2 if gex_abi::X86 || self.compat => {
                if a[0] != a[1] {
                    self.fd_io().closing_writes(m, a[1], a[1]);
                }
                self.sys_dup2(a[0] as i32, a[1] as i32)
            }
            nr::TIME if gex_abi::X86 || self.compat => self.sys_time(m, a[0]),
            nr::ALARM if gex_abi::X86 || self.compat => self.sys_alarm(a[0]),

            // -- I/O ---------------------------------------------------------
            nr::WRITE => self.sys_write(m, a[0] as i32, a[1], a[2]),
            nr::READ => self.sys_read(m, a[0] as i32, a[1], a[2]),
            nr::WRITEV => self.sys_writev(m, a[0] as i32, a[1], a[2]),
            nr::READV => self.sys_readv(m, a[0] as i32, a[1], a[2]),
            nr::PREAD64 => self.sys_pread(m, a[0] as i32, a[1], a[2], a[3]),
            nr::PWRITE64 => self.sys_pwrite(m, a[0] as i32, a[1], a[2], a[3]),
            nr::CLOSE => {
                self.fd_io().closing_writes(m, a[0], a[0]);
                self.sys_close(a[0] as i32)
            }
            nr::LSEEK => self
                .fds
                .get(a[0] as i32)
                .and_then(|f| f.seek(a[1] as i64, a[2] as u32)),
            nr::DUP => self.sys_dup(a[0] as i32),
            nr::DUP3 => {
                self.fd_io().closing_writes(m, a[1], a[1]);
                self.sys_dup3(a[0] as i32, a[1] as i32, a[2])
            }
            nr::PIPE2 => self.sys_pipe2(m, a[0], a[1]),
            // The calls that make or work a descriptor's object
            // ([`fd_object_call`], the free path's code).
            n if is_fd_object(n) => fd_object_call(&self.fd_io(), self.clock, m, n, a),
            // F_GETLK and F_OFD_GETLK: nobody holds a lock (see
            // `sys_fcntl`), so the answer is l_type = F_UNLCK, which is the
            // first field in every `struct flock` layout (i386's flock and
            // flock64, x86-64's, wasm32's).
            nr::FCNTL => self.fd_io().fcntl_call(m, a[0] as i32, a[1], a[2]),
            nr::IOCTL => self.sys_ioctl(m, a[0] as i32, a[1], a[2]),
            nr::FTRUNCATE => self.fd_io().ftruncate(m, a[0] as i32, a[1]),
            nr::TRUNCATE => self.path_io().truncate_target(m, a[0]).and_then(|f| self.fd_io().truncate_file(m, &f, a[1])),
            nr::FADVISE64 => self.fd_io().fadvise(a[0] as i32, a[1], a[2], a[3]),
            // `set_tid_address` and `unshare`.
            n if is_thread_own(n) => {
                let fs = self.fs_of(self.ctx_addr);
                thread_own_call(self.signals.thread(), &self.thread_fs, &fs, self.ctx_addr, self.tid, n, a)
            }
            nr::PPOLL | nr::PSELECT6 => self.fd_io().poll_now(m, n, a),
            nr::STATFS | nr::FSTATFS => self.sys_statfs(m, a[1]),
            // There is one process's view of the world and nothing consults
            // a umask, but a program that sets one wants the old value back.

            // -- memory ------------------------------------------------------
            nr::BRK | nr::MMAP | nr::MUNMAP | nr::MSYNC | nr::MREMAP | nr::MPROTECT | nr::MADVISE => {
                let Some(r) = self.map_io().call(m, n, a) else { unreachable!() };
                r
            }
            // The path lookups and opens, the free path's code.
            n if is_path(n) => {
                let Some(r) = self.path_io().call(m, n, a) else { unreachable!() };
                r
            }

            // -- process and time --------------------------------------------
            nr::GETPID => Ok(self.pid as u64),
            nr::GETTID => Ok(self.tid as u64),
            // -- credentials, sessions and process groups --------------------
            n if is_cred(n) => cred_call(&self.creds, &self.procs, self.pid, m, n, a),

            // -- mounts ------------------------------------------------------
            n if is_mount(n) => self.path_io().mount_call(m, n, a),

            // -- sockets -----------------------------------------------------
            // The socket calls ([`NetIo`]), against the calling thread's
            // working directory (entered into the namespace for the call).
            nr::SENDMSG => self.fd_io().sendmsg(m, a[0] as i32, a[1]),
            nr::RECVMSG => self.sys_recvmsg(m, a[0] as i32, a[1], a[2]),
            n if is_socket_call(n) => {
                let cwd = self.vfs.cwd.clone();
                self.net_io().call(&self.fd_io(), m, n, a, &cwd)
            }
            nr::FLOCK => self.fd_io().flock(m, a[0] as i32, a[1]),
            nr::EPOLL_CTL => self.sys_epoll_ctl(m, a[0] as i32, a[1], a[2] as i32, a[3]),
            nr::EPOLL_PWAIT | nr::EPOLL_PWAIT2 => self.fd_io().poll_now(m, n, a),
            nr::CLOSE_RANGE => {
                self.fd_io().closing_writes(m, a[0], a[1]);
                self.sys_close_range(a[0], a[1], a[2])
            }
            nr::CLOCK_GETTIME => self.sys_clock_gettime(m, a[0], a[1]),
            nr::CLOCK_GETRES => self.sys_clock_getres(m, a[1]),
            nr::GETTIMEOFDAY => self.sys_gettimeofday(m, a[0]),
            nr::NANOSLEEP => self.sys_nanosleep(m, a[0]),
            nr::CLOCK_NANOSLEEP => self.sys_nanosleep(m, a[2]),
            nr::GETRANDOM => self.sys_getrandom(m, a[0], a[1]),
            nr::PROCESS_VM_READV => self.sys_process_vm(m, false, a),
            nr::PROCESS_VM_WRITEV => self.sys_process_vm(m, true, a),
            // Interval timers raise their signal when due (`check_timers`,
            // and a page Worker's own `sigs::gather`).
            nr::SETITIMER => self.sys_itimer(m, a[0], a[1], a[2], true),
            nr::GETITIMER => self.sys_itimer(m, a[0], 0, a[1], false),
            nr::SCHED_YIELD => {
                std::thread::yield_now();
                Ok(0)
            }
            // Scheduling niceness: we run every guest thread at the default
            // priority, so report nice 0 and accept any set as a no-op. The
            // raw getpriority ABI returns `20 - nice` (kept non-negative so a
            // real nice value is never mistaken for an -errno), i.e. 20 here.
            nr::GETPRIORITY => Ok(20),
            nr::SETPRIORITY => Ok(0),
            nr::TIMES => self.sys_times(m, a[0]),
            nr::GETRUSAGE => self.sys_getrusage(m, a[0], a[1]),
            nr::SYSINFO => self.sys_sysinfo(m, a[0]),
            nr::PRLIMIT64 => self.sys_prlimit(m, a[1], a[2], a[3]),
            // A waitid that parks is `blocking_op`'s; one that gets here had bad arguments.
            nr::WAITID => waitid_child(&self.procs, &self.fds, self.pid, a, self.compat).map(|_| 0),

            // -- signals -----------------------------------------------------
            nr::RT_SIGACTION => self.sys_rt_sigaction(m, a[0] as u32, a[1], a[2]),
            nr::RT_SIGTIMEDWAIT => sigtimedwait(&self.signals, m, self.compat, a),
            // Reached only when the new mask could not be read (the
            // blocking path takes the rest).
            nr::RT_SIGSUSPEND => Err(Errno::EFAULT),
            nr::RT_SIGPROCMASK => self.sys_rt_sigprocmask(m, a[0] as u32, a[1], a[2]),
            nr::RT_SIGPENDING => self.sys_rt_sigpending(m, a[0]),
            nr::SIGALTSTACK => self.sys_sigaltstack(m, a[0], a[1]),
            // `kill`, `tkill` and `tgkill`, as the free path makes
            // them.
            nr::KILL | nr::TKILL | nr::TGKILL => {
                if self.trace && n == nr::KILL {
                    eprintln!("[gex:{}] kill -> pid {} signal {}", self.pid, a[0] as i32, a[1] as u32);
                }
                let to = KillTo { signals: &self.signals, threads: &self.thread_signals, procs: &self.procs, pid: self.pid, tid: self.tid, creds: &self.creds };
                kill_call(&to, n, a)
            }
            nr::PIDFD_OPEN | nr::PIDFD_SEND_SIGNAL => {
                let to = KillTo { signals: &self.signals, threads: &self.thread_signals, procs: &self.procs, pid: self.pid, tid: self.tid, creds: &self.creds };
                pidfd_call(&self.fds, &self.procs, &to, n, a)
            }
            nr::RT_SIGRETURN => {
                // Handled by the caller, which must redirect execution to the
                // restored PC rather than returning here.
                Ok(0)
            }

            nr::FUTEX => self.sys_futex(m, a),

            _ => {
                if !self.unimplemented.contains(&n) {
                    self.unimplemented.push(n);
                }
                if self.trace {
                    eprintln!("[gex:{}] unimplemented syscall {n} -> ENOSYS", self.pid);
                }
                Err(Errno::ENOSYS)
            }
        };
        Ok(r)
    }

    fn sys_futex(&mut self, m: &mut GuestMem<'_>, a: [u64; 6]) -> SysResult {
        let mapped = |addr: u64, len: u64| self.mem.lock().vma_at(addr).is_some_and(|v| addr + len <= v.end);
        futex_call(&self.futex_cx(&mapped), m, a)
    }

    /// What this process's futex calls need ([`FutexCx`]), for the calling
    /// thread, with the map asked directly (`mapped`): this kernel owns it.
    fn futex_cx<'a>(&'a self, mapped: &'a dyn Fn(u64, u64) -> bool) -> FutexCx<'a> {
        FutexCx { table: &self.futexes, map: None, mapped_by: Some(mapped), tid: self.tid as u32, clock: self.clock }
    }

    // -- signals -------------------------------------------------------------

    fn sys_rt_sigaction(&mut self, m: &mut GuestMem<'_>, signo: u32, act: u64, old: u64) -> SysResult {
        sigaction(&self.signals, m, self.compat, signo, act, old)
    }

    fn sys_rt_sigpending(&self, m: &mut GuestMem<'_>, set: u64) -> SysResult {
        sigpending(&self.signals, m, set)
    }


    fn sys_rt_sigprocmask(
        &mut self,
        m: &mut GuestMem<'_>,
        how: u32,
        set: u64,
        old: u64,
    ) -> SysResult {
        sigprocmask(&self.signals, m, how, set, old)
    }

    fn sys_sigaltstack(&mut self, m: &mut GuestMem<'_>, new: u64, old: u64) -> SysResult {
        sigaltstack(&self.signals, m, self.compat, new, old)
    }

    /// Deliver a pending signal, if any is deliverable right now.
    ///
    /// Called after every syscall and at safepoints. A `Resume` result means
    /// the caller must re-enter translated code at the returned address
    /// rather than continuing where it was.
    pub fn deliver_pending(&mut self, m: &mut GuestMem<'_>) -> Delivery {
        self.collect_signals();
        self.check_timers();
        if !self.signals.has_pending() {
            return Delivery::None;
        }
        if self.trace {
            eprintln!(
                "[gex:{}.{}] delivering signal {} (pending {:#x}, blocked {:#x})",
                self.pid,
                self.sig_tid,
                self.signals.peek_deliverable().unwrap_or(0),
                self.signals.pending_raw(),
                self.signals.blocked()
            );
        }
        signal::dispatch(&mut self.signals, m, self.ctx_addr, self.compat).unwrap_or(Delivery::None)
    }

    /// Move signals other processes have sent into this one's pending set.
    ///
    /// `kill` cannot reach into another process's kernel -- each has its own
    /// -- so it leaves the signal on a shared per-process word and the target
    /// picks it up here. This is what makes `timeout`, job control and every
    /// supervisor work.
    pub fn collect_signals(&self) {
        // `tgkill`s from other processes, each for one thread.
        if self.procs.has_signals(self.pid) {
            let mut other = false;
            for (tid, signo, from) in self.procs.take_thread_signals(self.pid, 0) {
                let source = crate::signal::Source { code: crate::signal::SI_TKILL, pid: from, uid: self.creds.ruid() };
                if tid == self.sig_tid || (self.sig_tid == 0 && tid == self.tid) {
                    self.signals.raise_from(signo, source);
                } else if let Some(t) = self.thread_signals.get(tid) {
                    t.raise_from(signo, source);
                    other = true;
                }
            }
            // Another thread's, taken here: it may have woken for the send
            // (`ProcTable::tkill`), found nothing yet and gone back to its
            // wait, so it is woken again to see it (tests/gexcc
            // tgkill-thread.c hung two runs in three in the page).
            if other {
                crate::wait::notify(self.procs.queue(self.pid));
            }
        }
        let inbox = self.signals.process().collect(&self.procs, self.pid, self.creds.ruid());
        if inbox != 0 && self.trace {
            eprintln!("[gex:{}] collected signals {inbox:#x}", self.pid);
        }
    }

    /// Whether anything is waiting to be delivered, counting what other
    /// processes have sent. Asked from the host's wait loops, which have no
    /// other way to notice.
    pub fn signal_pending(&mut self) -> bool {
        self.collect_signals();
        self.check_timers();
        let ready = self.signals.has_pending();
        if self.trace && !ready && self.signals.pending_raw() != 0 {
            // Arrived but not deliverable. Almost always the blocked mask,
            // and worth saying out loud: a process that will not wake looks
            // exactly like one nobody signalled.
            eprintln!(
                "[gex:{}.{}] signal pending {:#x} but blocked {:#x}",
                self.pid,
                self.sig_tid,
                self.signals.pending_raw(),
                self.signals.blocked()
            );
        }
        ready
    }

    /// Raise a synchronous fault against the running instruction.
    ///
    /// Returns where to resume if a handler took it. Unlike an asynchronous
    /// signal these cannot simply be discarded: a program with no handler for
    /// its own SIGSEGV must die.
    pub fn raise_fault(
        &mut self,
        m: &mut GuestMem<'_>,
        signo: u32,
        pc: u64,
        fault_address: u64,
    ) -> Delivery {
        let action = self.signals.action(signo);
        let blocked = self.signals.blocked() & (1u64 << (signo - 1)) != 0;
        // A synchronous fault that is blocked or ignored is fatal, because
        // returning would re-execute the faulting instruction forever.
        if action.handler == signal::SIG_DFL || action.handler == signal::SIG_IGN || blocked {
            return Delivery::Terminate {
                signal: signo,
                core: matches!(signal::default_action(signo), signal::DefaultAction::Core),
            };
        }
        if m.write_u64(self.ctx_addr + ctx::PC as u64, pc).is_err() {
            return Delivery::Terminate {
                signal: signo,
                core: true,
            };
        }
        self.signals.forget_source(signo);
        match signal::setup_frame(
            &mut self.signals,
            m,
            self.ctx_addr,
            signo,
            fault_address,
            self.compat,
        ) {
            Ok(resume) => {
                self.signals.process().note_fault_handled(signo);
                self.fault_note(signo, pc, fault_address);
                Delivery::Resume(resume)
            }
            Err(_) => Delivery::Terminate {
                signal: signo,
                core: true,
            },
        }
    }

    /// A fault that goes to the program's handler, said in a traced process
    /// (`gexc run --strace`; off by default and nothing done then: Wine's
    /// processes take handled faults routinely). A crash handler that
    /// re-raises the signal (KCrash) otherwise leaves nothing of the real
    /// fault. (The page's is `sigs::fault_note`.)
    fn fault_note(&self, signo: u32, pc: u64, addr: u64) {
        if !self.trace {
            return;
        }
        let at = self
            .mem
            .lock()
            .vmas()
            .iter()
            .find(|v| pc >= v.start && pc < v.end)
            .map(|v| {
                let n = v.code.as_ref().map(|(p, _)| p.as_str()).unwrap_or(&v.name);
                format!("{} +{:#x}", n.rsplit('/').next().unwrap_or(n), pc - v.start)
            })
            .unwrap_or_default();
        eprintln!("[gex:{}] {} at pc={pc:#x} {at} address {addr:#x}, to its handler", self.pid, signal::fault_name(signo));
    }

    /// Restore the context a handler frame captured.
    pub fn sigreturn(&mut self, m: &mut GuestMem<'_>) -> Result<u64, Errno> {
        let legacy = std::mem::replace(&mut self.sigreturn_legacy, false);
        signal::sigreturn(&mut self.signals, m, self.ctx_addr, self.compat, legacy)
    }

    // -- path helpers --------------------------------------------------------



    /// `p` relative to the directory open at `dirfd`.
    fn at_dir_path(&self, dirfd: i32, p: &str) -> Result<String, Errno> {
        self.path_io().at_dir_path(dirfd, p)
    }

    // -- I/O -----------------------------------------------------------------

    fn sys_write(&mut self, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64) -> SysResult {
        io_write(&self.fds, m, fd, buf, len, Some(self.mem.shared_maps()))
    }

    fn sys_read(&mut self, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64) -> SysResult {
        io_read(&self.fds, m, fd, buf, len, Some(self.mem.shared_maps()))
    }

    fn sys_pread(
        &mut self,
        m: &mut GuestMem<'_>,
        fd: i32,
        buf: u64,
        len: u64,
        off: u64,
    ) -> SysResult {
        io_pread(&self.fds, m, fd, buf, len, off, Some(self.mem.shared_maps()))
    }

    fn sys_pwrite(&mut self, m: &mut GuestMem<'_>, fd: i32, buf: u64, len: u64, off: u64) -> SysResult {
        io_pwrite(&self.fds, m, fd, buf, len, off, Some(self.mem.shared_maps()))
    }






    fn read_iovecs(
        m: &GuestMem<'_>,
        iov: u64,
        count: u64,
        compat: bool,
    ) -> Result<Vec<(u64, u64)>, Errno> {
        if count > 1024 {
            return Err(Errno::EINVAL);
        }
        let mut v = Vec::with_capacity(count as usize);
        for i in 0..count {
            // A 32-bit process's iovec is two 32-bit words.
            if compat {
                let base = iov + i * 8;
                v.push((m.read_u32(base)? as u64, m.read_u32(base + 4)? as u64));
            } else {
                let base = iov + i * 16;
                v.push((m.read_u64(base)?, m.read_u64(base + 8)?));
            }
        }
        Ok(v)
    }

    fn sys_writev(&mut self, m: &mut GuestMem<'_>, fd: i32, iov: u64, count: u64) -> SysResult {
        io_writev(&self.fds, m, fd, iov, count, self.compat, Some(self.mem.shared_maps()))
    }

    fn sys_readv(&mut self, m: &mut GuestMem<'_>, fd: i32, iov: u64, count: u64) -> SysResult {
        io_readv(&self.fds, m, fd, iov, count, self.compat, Some(self.mem.shared_maps()))
    }




    fn sys_dup(&mut self, fd: i32) -> SysResult {
        self.fd_io().dup(fd)
    }

    fn sys_dup3(&mut self, old: i32, new: i32, flags: u64) -> SysResult {
        self.fd_io().dup3(old, new, flags)
    }






    fn sys_pipe2(&mut self, m: &mut GuestMem<'_>, fds: u64, flags: u64) -> SysResult {
        self.fd_io().pipe2(m, fds, flags)
    }

    fn stat_from(md: &crate::vfs::Metadata) -> Stat {
        Stat {
            ino: md.ino,
            mode: md.file_type.mode_bits() | md.mode,
            nlink: md.nlink.max(1),
            size: md.size as i64,
            blksize: 4096,
            blocks: (md.size as i64 + 511) / 512,
            mtime: md.mtime,
            // A filesystem that keeps one time reports it for all three.
            mtime_ns: md.times.mtime_ns,
            atime: if md.times == Default::default() { md.mtime } else { md.times.atime },
            atime_ns: md.times.atime_ns,
            ctime: if md.times == Default::default() { md.mtime } else { md.times.ctime },
            ctime_ns: md.times.ctime_ns,
            // What the node says, or for a character device that does not,
            // a pseudo-terminal's number (136:0).
            rdev: if md.rdev != 0 {
                md.rdev
            } else if md.file_type == FileType::CharDevice {
                0x8800
            } else {
                0
            },
            ..Default::default()
        }
    }




    fn sys_fcntl(&mut self, fd: i32, cmd: u64, arg: u64) -> SysResult {
        self.fd_io().fcntl(fd, cmd, arg)
    }

}

/// The descriptor table's own calls: `close`, `close_range`, the `dup`
/// family, `fcntl` and `flock`. No kernel lock: the table publishes itself,
/// `flock`'s locks and a memfd's seals
/// lock themselves, and an `O_TMPFILE`'s name is dropped through the
/// process's published namespace.
impl FdIo<'_> {
    pub fn dup(&self, fd: i32) -> SysResult {
        let file = self.fds.get(fd)?;
        self.fds.insert(file, false)
    }

    pub fn dup3(&self, old: i32, new: i32, flags: u64) -> SysResult {
        if old == new {
            return Err(Errno::EINVAL);
        }
        let file = self.fds.get(old)?;
        self.fds.install(new, file, flags & oflags::O_CLOEXEC != 0)
    }

    /// `dup2`: `dup3` without flags, except that `dup2(fd, fd)` is a no-op
    /// that still validates the descriptor, where `dup3` would refuse.
    pub fn dup2(&self, old: i32, new: i32) -> SysResult {
        if old == new {
            self.fds.get(old)?;
            return Ok(new as u64);
        }
        self.dup3(old, new, 0)
    }

    /// Close a descriptor. The description's `flock` lock and an
    /// `O_TMPFILE`'s name go with the description itself (`OpenFile`'s
    /// `Drop`): when its last descriptor closes here, or in another process
    /// it was passed to, and no call still in flight holds it. `dup` and
    /// `fork` share one description, so closing a copy releases nothing.
    pub fn close(&self, fd: i32) -> SysResult {
        let file = self.fds.get(fd)?;
        self.fds.close(fd)?;
        let settle = file.undoes_on_last_close();
        drop(file);
        if settle {
            self.fds.settle();
        }
        Ok(0)
    }

    /// `close_range`: shut every open descriptor in a span at once.
    pub fn close_range(&self, first: u64, last: u64, flags: u64) -> SysResult {
        // `CLOSE_RANGE_CLOEXEC` marks the span close-on-exec instead of closing
        // it now; `CLOSE_RANGE_UNSHARE` asks for a private descriptor table
        // first, which is a no-op here because a process already has its own.
        const CLOSE_RANGE_CLOEXEC: u64 = 4;
        if first > last {
            return Err(Errno::EINVAL);
        }
        self.fds
            .close_range(first as u32, last as u32, flags & CLOSE_RANGE_CLOEXEC != 0);
        self.fds.settle();
        Ok(0)
    }

    /// `fcntl` as the call is made: the record-lock commands read and write
    /// a `struct flock` ([`FdIo::record_lock`]).
    pub fn fcntl_call(&self, m: &mut GuestMem<'_>, fd: i32, cmd: u64, arg: u64) -> SysResult {
        if is_record_lock(cmd) {
            return self.record_lock(m, fd, cmd, arg);
        }
        // The status flags in the guest's own spelling (an AArch64
        // program's `O_DIRECT` is not x86's).
        const F_GETFL: u64 = 3;
        const F_SETFL: u64 = 4;
        match cmd {
            F_SETFL => self.fcntl(fd, cmd, oflags::from_guest(arg, self.compat)),
            F_GETFL => self.fcntl(fd, cmd, arg).map(|f| oflags::to_guest(f, self.compat)),
            _ => self.fcntl(fd, cmd, arg),
        }
    }

    pub fn fcntl(&self, fd: i32, cmd: u64, arg: u64) -> SysResult {
        const F_DUPFD: u64 = 0;
        const F_GETFD: u64 = 1;
        const F_SETFD: u64 = 2;
        const F_GETFL: u64 = 3;
        const F_SETFL: u64 = 4;
        const F_SETLK: u64 = 6;
        const F_SETLKW: u64 = 7;
        // Open file description locks: the same story.
        const F_OFD_SETLK: u64 = 37;
        const F_OFD_SETLKW: u64 = 38;
        const F_ADD_SEALS: u64 = 1033;
        const F_GET_SEALS: u64 = 1034;
        const F_DUPFD_CLOEXEC: u64 = 1030;
        // every command is on an open descriptor
        self.fds.get(fd)?;
        match cmd {
            F_DUPFD | F_DUPFD_CLOEXEC => {
                let file = self.fds.get(fd)?;
                self.fds.insert_at(file, arg as i32, cmd == F_DUPFD_CLOEXEC)
            }
            F_GETFD => Ok(self.fds.cloexec(fd)? as u64),
            F_SETFD => {
                self.fds.set_cloexec(fd, arg & 1 != 0)?;
                Ok(0)
            }
            // Only the access mode and the status flags. Linux does not
            // keep `O_CREAT`, `O_EXCL`, `O_TRUNC`, `O_TMPFILE` or
            // `O_CLOEXEC` on the open file description, and reporting them
            // is how `fdopen` decided the descriptor was invalid.
            F_GETFL => {
                const STATUS: u64 = oflags::ACCMODE
                    | oflags::O_APPEND
                    | oflags::O_NONBLOCK
                    | oflags::O_DIRECT
                    | 0o4010000; // O_SYNC | O_NOATIME
                // The access mode comes from what the description can do,
                // not from the stored flags: inherited stdio, socketpair
                // pipes and `F_SETFL` callers all leave it at 0 there.
                // libuv's `uv_pipe_open` reads it to decide whether a
                // stream is writable, and node's stdout reported EPIPE on
                // every write while a write end said `O_RDONLY`.
                let f = self.fds.get(fd)?;
                let acc = match (f.readable, f.writable) {
                    (true, true) => oflags::O_RDWR,
                    (false, true) => oflags::O_WRONLY,
                    _ => oflags::O_RDONLY,
                };
                Ok(f.flags() & STATUS & !oflags::ACCMODE | acc)
            }
            // Only what Linux lets F_SETFL change (O_APPEND, O_ASYNC,
            // O_DIRECT, O_NOATIME, O_NONBLOCK); the access mode and the rest
            // stay. Taking `arg` whole lost a pipe write end's `O_WRONLY`
            // whenever a caller passed a bare `O_NONBLOCK`, and a forked
            // child then rebuilt that end as a read end.
            F_SETFL => {
                const SETTABLE: u64 = oflags::O_APPEND
                    | oflags::O_NONBLOCK
                    | 0o20000 // O_ASYNC
                    | oflags::O_DIRECT
                    | 0o1000000; // O_NOATIME
                let f = self.fds.get(fd)?;
                f.set_flags(f.flags() & !SETTABLE | arg & SETTABLE);
                Ok(0)
            }
            // Through memory: `fcntl_call` (`record_lock`).
            F_SETLK | F_SETLKW | F_OFD_SETLK | F_OFD_SETLKW => Err(Errno::EINVAL),
            // Sealing an anonymous shared file. Only a memfd has seals, and
            // Linux says EINVAL for anything else -- which is the answer
            // Xwayland was already getting, and it stopped rather than
            // carrying on without the guarantee.
            F_ADD_SEALS => self.memfd_of(fd)?.add_seals(arg as u32),
            F_GET_SEALS => Ok(self.memfd_of(fd)?.seals() as u64),
            _ => Err(Errno::EINVAL),
        }
    }

    /// The anonymous shared file behind a descriptor: `EINVAL` for any other.
    pub fn memfd_of(&self, fd: i32) -> Result<Arc<unix::MemFd>, Errno> {
        Arc::clone(&self.fds.get(fd)?.handle).as_memfd().ok_or(Errno::EINVAL)
    }

    /// `flock`. The open file description owns the lock, and its address is
    /// a stable name for it while it is open. Held in the way, a blocking
    /// request parks ([`FdIo::lock_park`]).
    pub fn flock(&self, m: &mut GuestMem<'_>, fd: i32, op: u64) -> SysResult {
        use crate::locks::lock_op;
        let file = self.fds.get(fd)?;
        // This process's stores go out before anyone else can take the lock:
        // an exclusive lock given up or turned shared (a writer holds one; a
        // shared lock given up has nothing of its own to write).
        let wkey = flock_owner(file.lock_key());
        let gives_up = op & (lock_op::UN | lock_op::SH) != 0;
        if gives_up && self.fds.release_write_lock(&file.path, wkey, 0, u64::MAX, false) {
            sync_shared_file(self.shared, m, &file, MapSync::Release, None);
        }
        let (r, seen) = self.file_locks.flock_seen(&file.path, file.lock_key(), op, file.handle.remote_id());
        if r.is_ok() {
            file.note_locks(self.file_locks);
            if gives_up {
                self.fds.release_write_lock(&file.path, wkey, 0, u64::MAX, true);
            } else if op & lock_op::EX != 0 {
                self.fds.note_write_lock(&file.path, wkey, 0, u64::MAX);
            }
            if op & lock_op::UN == 0 {
                sync_shared_file(self.shared, m, &file, MapSync::Acquire, seen);
            }
        }
        r
    }

    /// A record-lock command's `struct flock` at `arg`, as the request it
    /// makes: the request, and whether it waits. `cmd` is canonical (5-7,
    /// 36-38), or an i386 `struct flock`'s (`LK32` added: its offsets are
    /// 32-bit).
    fn lock_request(&self, m: &GuestMem<'_>, file: &Arc<OpenFile>, cmd: u64, arg: u64) -> Result<(crate::locks::LockReq, bool), Errno> {
        use crate::locks::{lk, LockReq, Owner};
        // (a wasm32 program's numbering is i386's, its struct the 64-bit one)
        let small = cmd & LK32 != 0 && self.compat && !self.wasm32;
        let packed = self.compat && !self.wasm32;
        let cmd = cmd & !LK32;
        // l_type, l_whence, l_start, l_len, l_pid: 64-bit offsets at 8 and 16
        // (x86-64, AArch64, wasm32), i386's flock64 packed at 4 and 12, its
        // flock's 32-bit at 4 and 8.
        let h = m.read(arg, 4)?;
        let (typ, whence) = (u16::from_le_bytes([h[0], h[1]]) as u32, u16::from_le_bytes([h[2], h[3]]));
        let (start, len, pid_at) = if small {
            (m.read_u32(arg + 4)? as i32 as i64, m.read_u32(arg + 8)? as i32 as i64, arg + 12)
        } else if packed {
            (m.read_u64(arg + 4)? as i64, m.read_u64(arg + 12)? as i64, arg + 20)
        } else {
            (m.read_u64(arg + 8)? as i64, m.read_u64(arg + 16)? as i64, arg + 24)
        };
        if typ > lk::UNLCK {
            return Err(Errno::EINVAL);
        }
        let base = match whence {
            0 => 0i64,
            1 => file.offset() as i64,
            2 => file.handle.metadata()?.size as i64,
            _ => return Err(Errno::EINVAL),
        };
        let mut lo = base.checked_add(start).ok_or(Errno::EOVERFLOW)?;
        let hi = if len > 0 {
            lo.checked_add(len).ok_or(Errno::EOVERFLOW)? as u64
        } else if len == 0 {
            u64::MAX
        } else {
            let h = lo;
            lo += len;
            h as u64
        };
        if lo < 0 {
            return Err(Errno::EINVAL);
        }
        let ofd = matches!(cmd, 36..=38);
        if ofd && m.read_u32(pid_at)? != 0 {
            return Err(Errno::EINVAL);
        }
        // A lock of a kind the descriptor cannot be used for (unlocking is
        // always allowed).
        if cmd != 5 && cmd != 36 && (typ == lk::RDLCK && !file.readable || typ == lk::WRLCK && !file.writable) {
            return Err(Errno::EBADF);
        }
        let (owner, pid) = if ofd { (Owner::Desc(file.lock_key()), -1) } else { (Owner::Pid(self.pid), self.pid) };
        let req = LockReq::Set { file: file.path.clone(), owner, pid, typ, start: lo as u64, end: hi };
        Ok((req, matches!(cmd, 7 | 38)))
    }

    /// `F_GETLK`, `F_SETLK`, `F_SETLKW` and their OFD forms. A lock held in
    /// the way answers `EAGAIN`, and a waiting one parks then
    /// ([`FdIo::lock_park`]).
    pub fn record_lock(&self, m: &mut GuestMem<'_>, fd: i32, cmd: u64, arg: u64) -> SysResult {
        use crate::locks::{lk, LockReq};
        let file = self.fds.get(fd)?;
        let (req, wait) = self.lock_request(m, &file, cmd, arg)?;
        let LockReq::Set { file: path, owner, pid, typ, start, end } = &req else { unreachable!() };
        let (small, plain) = (cmd & LK32 != 0 && self.compat && !self.wasm32, cmd & !LK32);
        if plain == 5 || plain == 36 {
            if *typ == lk::UNLCK {
                return Err(Errno::EINVAL);
            }
            let got = self.file_locks.get(path, *owner, *typ, *start, *end);
            let Some(c) = got else {
                return m.write(arg, &(lk::UNLCK as u16).to_le_bytes()).map(|_| 0);
            };
            let len = if c.end == u64::MAX { 0 } else { c.end - c.start };
            m.write(arg, &(c.typ as u16).to_le_bytes())?;
            m.write(arg + 2, &0u16.to_le_bytes())?;
            if small {
                if c.start > i32::MAX as u64 || len > i32::MAX as u64 {
                    return Err(Errno::EOVERFLOW);
                }
                m.write_u32(arg + 4, c.start as u32)?;
                m.write_u32(arg + 8, len as u32)?;
                m.write_u32(arg + 12, c.pid as u32)?;
            } else if self.compat && !self.wasm32 {
                m.write_u64(arg + 4, c.start)?;
                m.write_u64(arg + 12, len)?;
                m.write_u32(arg + 20, c.pid as u32)?;
            } else {
                m.write_u64(arg + 8, c.start)?;
                m.write_u64(arg + 16, len)?;
                m.write_u32(arg + 24, c.pid as u32)?;
            }
            return Ok(0);
        }
        // This process's stores go out before anyone else can take the range:
        // a write lock given up or turned read (writers hold one; giving up
        // a read lock does not look at the mapping).
        let wkey = owner.encode();
        if *typ != lk::WRLCK && self.fds.release_write_lock(path, wkey, *start, *end, false) {
            sync_shared_file(self.shared, m, &file, MapSync::Release, None);
        }
        let (r, seen) = self.file_locks.set_seen(path, *owner, *pid, *typ, *start, *end, wait, file.handle.remote_id());
        r?;
        if *typ == lk::WRLCK {
            self.fds.note_write_lock(path, wkey, *start, *end);
        } else {
            self.fds.release_write_lock(path, wkey, *start, *end, true);
        }
        if *typ != lk::UNLCK {
            match owner {
                crate::locks::Owner::Pid(p) => self.fds.note_record_lock(self.file_locks, *p, path),
                crate::locks::Owner::Desc(_) => file.note_locks(self.file_locks),
            }
            sync_shared_file(self.shared, m, &file, MapSync::Acquire, seen);
        }
        Ok(0)
    }

    /// Descriptors `lo..=hi` about to close: the process's record write
    /// locks on their files go with any close (Linux's rule), so its stores
    /// to shared mappings of those files go out first, as an unlock's do.
    /// One load while it holds no write lock.
    pub fn closing_writes(&self, m: &mut GuestMem<'_>, lo: u64, hi: u64) {
        if !self.fds.holds_write_locks() {
            return;
        }
        let me = crate::locks::Owner::Pid(self.pid).encode();
        let files: Vec<Arc<OpenFile>> = if lo == hi {
            self.fds.get(lo as i32).ok().into_iter().collect()
        } else {
            self.fds.list().into_iter().filter(|(fd, _)| (lo..=hi).contains(&(*fd as u64))).filter_map(|(fd, _)| self.fds.get(fd).ok()).collect()
        };
        for file in files {
            if self.fds.drop_write_locks(&file.path, me, false) {
                sync_shared_file(self.shared, m, &file, MapSync::Release, None);
            }
        }
    }

    /// The park for a lock request that found the lock held and waits
    /// (`F_SETLKW`, `F_OFD_SETLKW`, `flock` without `LOCK_NB`).
    fn lock_park(&self, m: &GuestMem<'_>, n: u64, a: [u64; 6]) -> Option<BlockOp> {
        let file = self.fds.get(a[0] as i32).ok()?;
        let req = match n {
            nr::FLOCK if a[1] & (crate::locks::lock_op::NB | crate::locks::lock_op::UN) == 0 => crate::locks::LockReq::Flock { file: file.path.clone(), key: file.lock_key(), op: a[1] },
            nr::FCNTL if is_record_lock(a[1]) => match self.lock_request(m, &file, a[1], a[2]) {
                Ok((req, true)) => req,
                _ => return None,
            },
            _ => return None,
        };
        // Noted now for when the host grants it: the close rule and the
        // description's release then cover it (nothing to drop if it never is).
        match &req {
            crate::locks::LockReq::Set { owner: crate::locks::Owner::Pid(p), file: path, .. } => self.fds.note_record_lock(self.file_locks, *p, path),
            _ => file.note_locks(self.file_locks),
        }
        // And a write lock as held (an extra look at the mapping at its
        // release, if it never is).
        match &req {
            crate::locks::LockReq::Set { file: path, owner, typ, start, end, .. } if *typ == crate::locks::lk::WRLCK => self.fds.note_write_lock(path, owner.encode(), *start, *end),
            crate::locks::LockReq::Flock { file: path, key, op } if op & crate::locks::lock_op::EX != 0 => self.fds.note_write_lock(path, flock_owner(*key), 0, u64::MAX),
            _ => {}
        }
        Some(BlockOp::FileLock { locks: Arc::clone(self.file_locks), req })
    }
}

/// An i386 `struct flock`'s record-lock commands (32-bit offsets): the
/// canonical number with this added ([`crate::i386`]'s `fcntl`). A wasm32
/// program's calls take i386's numbers too, and its 5-7 are the canonical
/// ones (a 64-bit `struct flock`): the bit means nothing there.
pub const LK32: u64 = 0x8000;

/// A `flock` lock's owner among [`crate::file::FdTable::note_write_lock`]'s
/// (bit 62 keeps it apart from the record-lock owners' encodings).
fn flock_owner(key: u64) -> u64 {
    key | 1 << 62
}

/// `F_GETLK`, `F_SETLK`, `F_SETLKW`, `F_OFD_GETLK`, `F_OFD_SETLK`,
/// `F_OFD_SETLKW`, and the i386 `struct flock` forms ([`LK32`]).
fn is_record_lock(cmd: u64) -> bool {
    matches!(cmd, 5..=7 | 36..=38) || matches!(cmd ^ LK32, 5..=7)
}

impl FdIo<'_> {
    /// `ioctl`: the device's own requests, the descriptor's, and the
    /// terminal's. No kernel lock: the table, the process table, the
    /// terminal and every device synchronise themselves.
    pub fn ioctl(
        &self,
        m: &mut GuestMem<'_>,
        procs: &ProcTable,
        tty: &crate::tty::TtyRef,
        fd: i32,
        req: u64,
        arg: u64,
    ) -> SysResult {
        use crate::tty::{self, lflag};
        // The kernel takes an `unsigned int` command: a request with the top
        // `_IOC` direction bit set (a read ioctl like TIOCGPTN or TCGETS2)
        // reaches glibc/musl as a negative `int` and is sign-extended when
        // widened into the syscall register, arriving here as `0xffffffff_8...`.
        // Masking to 32 bits is what the real kernel does and what makes those
        // read-direction ioctls match their numbers at all.
        let req = req & 0xffff_ffff;
        // The terminal's own requests. `_IOC` encodes direction and size into
        // the number, which is why the four `termios2` calls and the pty ones
        // look nothing like the rest: they are not arbitrary, they are
        // `_IOR('T', 0x2a, struct termios2)` and so on.
        const TCGETS: u64 = 0x5401;
        const TCSETS: u64 = 0x5402;
        const TCSETSW: u64 = 0x5403;
        const TCSETSF: u64 = 0x5404;
        const TCGETA: u64 = 0x5405;
        const TCSETA: u64 = 0x5406;
        const TCSETAW: u64 = 0x5407;
        const TCSETAF: u64 = 0x5408;
        const TCSBRK: u64 = 0x5409;
        const TCXONC: u64 = 0x540a;
        const TCFLSH: u64 = 0x540b;
        const TIOCEXCL: u64 = 0x540c;
        const TIOCNXCL: u64 = 0x540d;
        const TIOCSCTTY: u64 = 0x540e;
        const TIOCGPGRP: u64 = 0x540f;
        const TIOCSPGRP: u64 = 0x5410;
        const TIOCOUTQ: u64 = 0x5411;
        const TIOCSTI: u64 = 0x5412;
        const TIOCGWINSZ: u64 = 0x5413;
        const TIOCSWINSZ: u64 = 0x5414;
        const TIOCMGET: u64 = 0x5415;
        const TIOCMBIS: u64 = 0x5416;
        const TIOCMBIC: u64 = 0x5417;
        const TIOCMSET: u64 = 0x5418;
        const TIOCGSOFTCAR: u64 = 0x5419;
        const TIOCSSOFTCAR: u64 = 0x541a;
        const FIONREAD: u64 = 0x541b; // and TIOCINQ, which is the same number
        const TIOCLINUX: u64 = 0x541c;
        const TIOCGSERIAL: u64 = 0x541e;
        const TIOCSSERIAL: u64 = 0x541f;
        const FIONBIO: u64 = 0x5421;
        const TIOCNOTTY: u64 = 0x5422;
        const TIOCSETD: u64 = 0x5423;
        const TIOCGETD: u64 = 0x5424;
        const TCSBRKP: u64 = 0x5425;
        const TIOCGSID: u64 = 0x5429;
        const TCGETS2: u64 = 0x802c_542a;
        const TCSETS2: u64 = 0x402c_542b;
        const TCSETSW2: u64 = 0x402c_542c;
        const TCSETSF2: u64 = 0x402c_542d;
        const TIOCGPTN: u64 = 0x8004_5430;
        const TIOCSPTLCK: u64 = 0x4004_5431;
        const TIOCGDEV: u64 = 0x8004_5432;
        const TIOCSIG: u64 = 0x4004_5436;
        const TIOCVHANGUP: u64 = 0x5437;
        const TIOCGPKT: u64 = 0x8004_5438;
        const TIOCGPTLCK: u64 = 0x8004_5439;
        const TIOCGEXCL: u64 = 0x8004_5440;
        const TIOCGPTPEER: u64 = 0x5441;
        const FIONCLEX: u64 = 0x5450;
        const FIOCLEX: u64 = 0x5451;
        const FIOASYNC: u64 = 0x5452;
        const TIOCGLCKTRMIOS: u64 = 0x5456;
        const TIOCSLCKTRMIOS: u64 = 0x5457;

        let file = self.fds.get(fd)?;
        // A device the host provides answers its own ioctls, and knows things
        // the kernel does not. Asked first, because the request numbers below
        // are the terminal's and a graphics device's are its own.
        // Claiming a controlling terminal has to be recorded where the terminal
        // lives. For a local tty that is the branch further down (it has the
        // process table); for a pty in the global kernel it is the handle, which
        // is the only thing that can reach the other module. See
        // `FileHandle::claim_session`.
        if req == TIOCSCTTY {
            let sid = procs.sid(self.pid, 0).unwrap_or(self.pid);
            file.handle.claim_session(sid);
        }
        if let Some(r) = file.handle.ioctl(m, req, arg) {
            return r;
        }
        // The close-on-exec pair are properties of the descriptor and say
        // nothing about the device, so they answer for any file at all --
        // which is what `FIOCLEX` on a pipe expects.
        match req {
            FIOCLEX => {
                self.fds.set_cloexec(fd, true)?;
                return Ok(0);
            }
            FIONCLEX => {
                self.fds.set_cloexec(fd, false)?;
                return Ok(0);
            }
            FIONBIO => {
                let on = m.read_u32(arg)? != 0;
                let f = self.fds.get(fd)?;
                let flags = f.flags();
                f.set_flags(if on { flags | oflags::O_NONBLOCK } else { flags & !oflags::O_NONBLOCK });
                return Ok(0);
            }
            _ => {}
        }
        let is_tty = file.is_tty();
        if !is_tty {
            // Everything below is a terminal's. `FIONREAD` is the one that is
            // also asked of pipes and sockets, so it is answered for those
            // before the terminal cases begin.
            return match req {
                FIONREAD => {
                    let n = match file.handle.available() {
                        Some(n) => n,
                        None => {
                            let md = file.metadata()?;
                            md.size.saturating_sub(file.offset())
                        }
                    };
                    m.write_u32(arg, n as u32)?;
                    Ok(0)
                }
                FIOASYNC => Ok(0),
                // An interface's index, name, flags, MTU or addresses
                // (crate::netlink), asked of any socket.
                r if crate::netlink::is_iface_ioctl(r) => {
                    let mut ifr: [u8; 32] = m.read(arg, 32)?.try_into().unwrap();
                    crate::netlink::iface_ioctl(r, &mut ifr)?;
                    m.write(arg, &ifr)?;
                    Ok(0)
                }
                _ => Err(Errno::ENOTTY),
            };
        }
                match req {
            // Only the kernel's part of the structure: glibc's `tcgetattr`
            // passes a 36-byte `struct __kernel_termios` on the stack, and
            // writing the 60-byte userspace shape overruns it onto the stack
            // canary -- `bash -i` reported stack smashing before its prompt.
            TCGETS => {
                m.write(arg, &tty.termios()[..KERNEL_TERMIOS])?;
                Ok(0)
            }
            // Kept, not discarded. A shell that turns off ICANON and ECHO is
            // taking over line editing, and the terminal has to stop doing
            // its own or every keystroke lands twice.
            //
            // The three differ only in when they take effect -- now, after
            // the output drains, and after discarding pending input. There is
            // no output queue to drain here, so the first two are the same
            // call; the third flushes, because a program that asked for a
            // clean slate and did not get one reads the keys typed at the
            // prompt it just replaced.
            TCSETS | TCSETSW | TCSETSF => {
                // The same shape in the other direction: the caller gave a
                // kernel-sized structure, so reading past it would read the
                // caller's stack.
                let src = m.read(arg, KERNEL_TERMIOS)?.to_vec();
                if req == TCSETSF {
                    tty.flush(tty::TCIFLUSH);
                }
                tty.set_termios(&src);
                Ok(0)
            }
            // `termios2`, which adds the two `c_ispeed`/`c_ospeed` words
            // after `c_cc`. Nothing here has a baud rate, so they are the
            // 38400 the flags already say, and the rest of the structure is
            // the one above.
            TCGETS2 => {
                let t = tty.termios();
                let mut out = [0u8; KERNEL_TERMIOS + 8];
                out[..KERNEL_TERMIOS].copy_from_slice(&t[..KERNEL_TERMIOS]);
                out[KERNEL_TERMIOS..KERNEL_TERMIOS + 4].copy_from_slice(&38400u32.to_le_bytes());
                out[KERNEL_TERMIOS + 4..].copy_from_slice(&38400u32.to_le_bytes());
                m.write(arg, &out)?;
                Ok(0)
            }
            TCSETS2 | TCSETSW2 | TCSETSF2 => {
                let src = m.read(arg, KERNEL_TERMIOS)?.to_vec();
                if req == TCSETSF2 {
                    tty.flush(tty::TCIFLUSH);
                }
                tty.set_termios(&src);
                Ok(0)
            }
            // The `termio` of System V, which is `termios` with one-byte
            // flags. Ancient, and still what `busybox stty` falls back to on
            // a kernel that refuses `TCGETS`.
            TCGETA => {
                let t = tty.termios();
                let mut out = [0u8; 8 + 8];
                out[0..2].copy_from_slice(&t[0..2]);
                out[2..4].copy_from_slice(&t[4..6]);
                out[4..6].copy_from_slice(&t[8..10]);
                out[6..8].copy_from_slice(&t[12..14]);
                out[8] = t[16]; // c_line
                out[9..9 + 7].copy_from_slice(&t[17..17 + 7]);
                m.write(arg, &out)?;
                Ok(0)
            }
            TCSETA | TCSETAW | TCSETAF => {
                let src = m.read(arg, 16)?.to_vec();
                let mut t = tty.termios();
                t[0..2].copy_from_slice(&src[0..2]);
                t[2..4].fill(0);
                t[4..6].copy_from_slice(&src[2..4]);
                t[6..8].fill(0);
                t[8..10].copy_from_slice(&src[4..6]);
                t[10..12].fill(0);
                t[12..14].copy_from_slice(&src[6..8]);
                t[14..16].fill(0);
                t[16] = src[8];
                t[17..17 + 7].copy_from_slice(&src[9..9 + 7]);
                tty.set_termios(&t);
                Ok(0)
            }
            // The lock is answered and not enforced. A program that reads it
            // back is checking whether it may change a setting; all-zero says
            // it may, which is true.
            TIOCGLCKTRMIOS => {
                m.write(arg, &tty.locked_termios()[..KERNEL_TERMIOS])?;
                Ok(0)
            }
            TIOCSLCKTRMIOS => {
                let src = m.read(arg, KERNEL_TERMIOS)?.to_vec();
                tty.set_locked_termios(&src);
                Ok(0)
            }
            TIOCGWINSZ => {
                let (rows, cols, x, y) = tty.winsize();
                let mut ws = [0u8; 8];
                ws[0..2].copy_from_slice(&rows.to_le_bytes());
                ws[2..4].copy_from_slice(&cols.to_le_bytes());
                ws[4..6].copy_from_slice(&x.to_le_bytes());
                ws[6..8].copy_from_slice(&y.to_le_bytes());
                m.write(arg, &ws)?;
                Ok(0)
            }
            // A program may resize the terminal too -- `resize` does, and so
            // does a terminal multiplexer telling its children how big their
            // pane is. Everyone in the foreground group hears about it.
            TIOCSWINSZ => {
                let ws = m.read(arg, 8)?.to_vec();
                let g = |n: usize| u16::from_le_bytes([ws[n], ws[n + 1]]);
                if tty.set_winsize((g(0), g(2), g(4), g(6))) {
                    signal_foreground(procs, tty, self.pid, crate::signal::sig::SIGWINCH);
                }
                Ok(0)
            }
            // Who is in the foreground. Zero before any shell has claimed the
            // terminal, and Linux answers with a group that does not exist
            // rather than with an error -- but the caller is a better guess
            // than a lie, and it is what `tcgetpgrp` in a script wants.
            TIOCGPGRP => {
                let fg = tty.fg();
                let g = if fg > 0 {
                    fg
                } else {
                    procs.pgid(self.pid, 0).unwrap_or(self.pid)
                };
                m.write_u32(arg, g as u32)?;
                Ok(0)
            }
            // A shell handing the terminal to a job. This is what makes job
            // control real: the group named here is the one `^C` reaches, so
            // a shell that backgrounds a pipeline stops it being interrupted
            // and the shell itself starts being.
            TIOCSPGRP => {
                let pgrp = m.read_u32(arg)? as i32;
                if pgrp <= 0 {
                    return Err(Errno::EINVAL);
                }
                tty.set_fg(pgrp);
                Ok(0)
            }
            TIOCGSID => {
                let sid = tty.sid();
                let s = if sid > 0 { sid } else { procs.sid(self.pid, 0).unwrap_or(1) };
                m.write_u32(arg, s as u32)?;
                Ok(0)
            }
            // Claiming the terminal. A session leader that has none takes it,
            // and becomes the foreground group by doing so.
            TIOCSCTTY => {
                let sid = procs.sid(self.pid, 0).unwrap_or(self.pid);
                let held = tty.sid();
                // `arg` non-zero is the steal-it-anyway form, which only root
                // may use -- and here everybody is root often enough that
                // refusing it would be the surprising answer.
                if held != 0 && held != sid && arg == 0 {
                    return Err(Errno::EPERM);
                }
                tty.set_sid(sid);
                Ok(0)
            }
            TIOCNOTTY => {
                if tty.sid() == procs.sid(self.pid, 0).unwrap_or(-1) {
                    tty.set_sid(0);
                }
                Ok(0)
            }
            // Throw away what has been typed and not read, what has been
            // written and not displayed, or both. There is no output queue --
            // a write reaches the page before the call returns -- so only the
            // input half does anything.
            TCFLSH => {
                let which = arg as i32;
                if which == tty::TCIFLUSH || which == tty::TCIOFLUSH {
                    tty.flush(tty::TCIFLUSH);
                }
                Ok(0)
            }
            // Send a break, or wait for output to drain. Both are `tcdrain`
            // and `tcsendbreak` underneath, and both are satisfied the moment
            // they are asked: there is nothing in flight.
            TCSBRK | TCSBRKP | TIOCVHANGUP => Ok(0),
            // Flow control by hand: `tcflow(TCOOFF)` and friends. The same
            // switch `^S` throws, which is why it is the terminal's and not
            // this process's.
            TCXONC => {
                const TCOOFF: u64 = 0;
                const TCOON: u64 = 1;
                const TCIOFF: u64 = 2;
                const TCION: u64 = 3;
                match arg {
                    TCOOFF => tty.set_stopped(true),
                    TCOON => tty.set_stopped(false),
                    // Sending the stop and start characters upstream. There
                    // is no upstream: the keyboard is the page.
                    TCIOFF | TCION => {}
                    _ => return Err(Errno::EINVAL),
                }
                Ok(0)
            }
            // Bytes typed and not yet read. The terminal knows, because the
            // queue is its own; the descriptor's own answer is the fallback
            // for a run with no shared terminal.
            FIONREAD => {
                let n = match tty.queued() {
                    Some(n) if n >= 0 => n as u64,
                    _ => file.handle.available().unwrap_or(0),
                };
                m.write_u32(arg, n as u32)?;
                Ok(0)
            }
            // Bytes written and not yet displayed, which is always none.
            TIOCOUTQ => {
                m.write_u32(arg, 0)?;
                Ok(0)
            }
            // Push a byte back as though it had been typed. `TIOCSTI` is how
            // a shell puts a command on the next prompt, and how `su` used to
            // be escaped from -- Linux now wants `CAP_SYS_ADMIN` for it. Here
            // there is one terminal and one person at it.
            TIOCSTI => {
                let b = m.read(arg, 1)?.to_vec();
                tty.push(&b);
                Ok(0)
            }
            // Exclusive use. Nothing enforces it -- there is one page and one
            // terminal -- but a program that sets it reads it back.
            TIOCEXCL => {
                tty.set_exclusive(true);
                Ok(0)
            }
            TIOCNXCL => {
                tty.set_exclusive(false);
                Ok(0)
            }
            TIOCGEXCL => {
                m.write_u32(arg, tty.exclusive() as u32)?;
                Ok(0)
            }
            // The line discipline. `N_TTY` and nothing else; see
            // `TtyRef::set_ldisc` for why the others are refused rather than
            // accepted and ignored.
            TIOCGETD => {
                m.write_u32(arg, tty.ldisc() as u32)?;
                Ok(0)
            }
            TIOCSETD => {
                let n = m.read_u32(arg)? as i32;
                if tty.set_ldisc(n) {
                    Ok(0)
                } else {
                    Err(Errno::EINVAL)
                }
            }
            // Modem lines. There is no modem, and the honest answer for a
            // terminal that is always connected is carrier plus the two
            // ready lines: DTR | RTS | CTS | CAR | DSR.
            TIOCMGET => {
                const TIOCM_DTR: u32 = 0x002;
                const TIOCM_RTS: u32 = 0x004;
                const TIOCM_CTS: u32 = 0x020;
                const TIOCM_CAR: u32 = 0x040;
                const TIOCM_DSR: u32 = 0x100;
                m.write_u32(arg, TIOCM_DTR | TIOCM_RTS | TIOCM_CTS | TIOCM_CAR | TIOCM_DSR)?;
                Ok(0)
            }
            TIOCMSET | TIOCMBIS | TIOCMBIC => Ok(0),
            TIOCGSOFTCAR => {
                m.write_u32(arg, tty.softcar() as u32)?;
                Ok(0)
            }
            TIOCSSOFTCAR => {
                tty.set_softcar(m.read_u32(arg)? != 0);
                Ok(0)
            }
            // Send a signal to the foreground group, which is what the master
            // end of a pty does for `^C`. Reachable here because the page is
            // the master end.
            TIOCSIG => {
                let signo = m.read_u32(arg)?;
                if signo == 0 || signo > crate::signal::sig::MAX {
                    return Err(Errno::EINVAL);
                }
                signal_foreground(procs, tty, self.pid, signo);
                Ok(0)
            }
            // Asynchronous I/O by SIGIO. Accepted and not armed: nothing here
            // has ever sent one, and a program that asks is asking to be told
            // sooner, not to be told differently.
            FIOASYNC => Ok(0),
            // The pty ones. This terminal is not one half of a pair -- it is
            // the console -- so the number and the lock have no meaning, and
            // `EINVAL` is what Linux says for them on a console too.
            TIOCGPTN | TIOCSPTLCK | TIOCGPTLCK | TIOCGPTPEER | TIOCGPKT => Err(Errno::EINVAL),
            // The device number, for a program tracking which terminal it is
            // on. `/dev/console` is 5:1.
            TIOCGDEV => {
                m.write_u32(arg, (5 << 8) | 1)?;
                Ok(0)
            }
            // A serial port's parameters. There is no serial port; refusing
            // is what `setserial` on a console gets.
            TIOCGSERIAL | TIOCSSERIAL => Err(Errno::ENOTTY),
            // The console multiplexer's own call: screen dumps, selection,
            // keyboard mode. Only the "is the console blanked" subcode has an
            // answer here, and it is no.
            TIOCLINUX => Err(Errno::EINVAL),
            _ => {
                // Worth saying rather than swallowing: an unknown terminal
                // request is usually a feature the guest wanted and quietly
                // did without. `stty` asking for something absent looks
                // exactly like `stty` reporting the wrong thing.
                let _ = lflag::ISIG;
                Err(Errno::ENOTTY)
            }
        }
    }
}

/// Raise `signo` on whoever holds the terminal.
///
/// The foreground process group if a shell has set one, and otherwise
/// this process's own group -- which is right for the case that has no
/// shell in it at all, a program started directly on the terminal.
fn signal_foreground(procs: &ProcTable, tty: &crate::tty::TtyRef, pid: i32, signo: u32) {
    let fg = tty.fg();
    let g = if fg > 0 {
        fg
    } else {
        procs.pgid(pid, 0).unwrap_or(pid)
    };
    // `kill_group` and not `kill(-g)`: a negative target of -1 is
    // POSIX's "every process", so group 1 is the one group that cannot
    // be named negatively -- and a shell running as pid 1 is in it.
    let _ = procs.kill_group(pid, g, signo);
}

impl Kernel {
    fn sys_ioctl(&mut self, m: &mut GuestMem<'_>, fd: i32, req: u64, arg: u64) -> SysResult {
        self.fd_io().ioctl(m, &self.procs, &self.tty, fd, req, arg)
    }













    /// Nanoseconds on the guest's clock, for comparing deadlines against.
    pub fn now_nanos(&self) -> u128 {
        let (secs, nanos) = (self.clock)();
        secs as u128 * 1_000_000_000 + nanos as u128
    }

    /// `setitimer` and `getitimer`. A `struct itimerval` is two `timeval`s
    /// (reload interval then time remaining), each two 64-bit words of seconds
    /// and *microseconds*: 32 bytes. These actually fire -- `openssl speed`
    /// runs its loop until SIGALRM and nothing else stops it.
    fn sys_itimer(&mut self, m: &mut GuestMem<'_>, which: u64, new: u64, old: u64, set: bool) -> SysResult {
        itimer(&self.signals, self.now_nanos(), m, which, new, old, set)
    }

    /// Whether this read or write, about to be made, can end anyone's wait
    /// (see [`wakes_fd_waiters`]). Its descriptor must be a stream -- a read
    /// or write of a file, `/proc` or a device never ends a wait -- and a
    /// read must be able to make room (`FileHandle::read_frees_nothing`).
    /// Each "no" spares every parked Worker in the page a wake: a Wine
    /// process's reads of the wineserver's replies can be half of all global
    /// wakes, every one of them going round every idle Worker of every
    /// process.
    /// Anything else is "yes".
    fn io_may_wake(&self, n: u64, a: &[u64; 6]) -> bool {
        let read = matches!(n, nr::READ | nr::READV | nr::PREAD64);
        if !read && !matches!(n, nr::WRITE | nr::WRITEV | nr::PWRITE64) {
            return true;
        }
        let Ok(file) = self.fds.get(a[0] as i32) else { return true };
        if !file.handle.is_stream() {
            return false;
        }
        !(read && file.handle.read_frees_nothing())
    }

    /// Whether any interval timer is armed.
    pub fn timers_armed(&self) -> bool {
        self.signals.process().timers_armed()
    }

    /// How long until the next interval timer fires, if one is armed: a
    /// parked thread's host sleeps no longer than that, since the timer's
    /// signal may be what ends its wait.
    pub fn timer_in_ns(&self) -> Option<u64> {
        self.signals.process().timer_in_ns(self.now_nanos())
    }

    /// Whether an interval timer has expired and has yet to raise its signal.
    /// The host asks this on its way past: a process that never traps into
    /// the kernel (a shell's `timeout` spinning on `wait4`/`sigsuspend`) would
    /// otherwise never have its `alarm` delivered, because delivery is only
    /// attempted once a signal is already pending.
    pub fn timer_due(&self) -> bool {
        self.signals.process().timer_due(self.now_nanos())
    }

    /// Fire any timer whose deadline has passed, and reload it if it repeats
    /// ([`signal::ProcSig::fire_timers`]).
    pub fn check_timers(&self) {
        if self.signals.process().timers_armed() {
            self.signals.process().fire_timers(self.now_nanos());
        }
    }


    /// `statfs`. One filesystem, large and mostly empty, which is what a
    /// program checking for room before it writes wants to hear.
    fn sys_statfs(&mut self, m: &mut GuestMem<'_>, buf: u64) -> SysResult {
        statfs(m, self.compat, buf)
    }

    // -- memory --------------------------------------------------------------

    // -- sessions, credentials and mounts --------------------------------------

    /// Move anything another process has sent into this one's pending set.
    fn take_sent_signals(&mut self) {
        if !self.procs.has_signals(self.pid) {
            return;
        }
        self.signals.process().collect(&self.procs, self.pid, self.creds.ruid());
    }

    /// `mount`, for the filesystems gex has.
    ///
    /// An init script's first act is usually to mount `/proc`, `/sys` and
    /// `/dev`, and failing that is not fatal but it does mean everything
    /// afterwards reads an empty `/proc`. The hosts already mount `/proc` and
    /// `/dev` before the guest starts, so the common case is a request for
    /// something that is there -- which succeeds, because the guest's view
    /// afterwards is the one it asked for.
    // -- sockets ---------------------------------------------------------------





    /// Put a socket that lives in the global kernel at the lowest free
    /// descriptor.
    pub fn insert_remote_unix(&mut self, sock: Arc<unix::RemoteUnix>, flags: u64, cloexec: bool) -> Result<i32, Errno> {
        let handle: Arc<dyn crate::vfs::FileHandle> = sock.clone();
        let file = OpenFile::new(handle, "socket".to_string(), flags, true, true);
        let fd = self.fds.insert(file, cloexec)? as i32;
        Ok(fd)
    }

    /// The global-kernel socket behind an open file, if that is what it is.
    pub fn remote_unix_of_file(file: &OpenFile) -> Option<Arc<unix::RemoteUnix>> {
        Arc::clone(&file.handle).as_remote_unix()
    }


    /// The path out of a `struct sockaddr_un`: a two-byte family then up to
    /// 108 bytes of NUL-terminated path.
    ///
    /// The abstract namespace -- a leading NUL, which is how D-Bus and X11
    /// usually bind -- is kept as a distinct name by prefixing it, so an
    /// abstract socket and a filesystem one of the same spelling do not
    /// collide.
    fn sockaddr_un(m: &GuestMem<'_>, addr: u64, len: u64) -> Result<String, Errno> {
        if addr == 0 || len < 2 {
            return Err(Errno::EINVAL);
        }
        let bytes = m.read(addr, (len as usize).min(110))?;
        let path = &bytes[2..];
        if path.first() == Some(&0) {
            // An abstract name is every byte the length gives, NULs and
            // all: Linux's, which getsockname gives back as it was bound.
            return Ok(format!("\0{}", String::from_utf8_lossy(&path[1..])));
        }
        let end = path.iter().position(|&c| c == 0).unwrap_or(path.len());
        Ok(String::from_utf8_lossy(&path[..end]).into_owned())
    }

    /// Write a `struct sockaddr_un` back, for `getsockname`, `recvfrom` and
    /// `recvmsg`: at most the caller's `*len_out` bytes of it (a
    /// `struct sockaddr` of 16 is a common buffer), its whole length reported.
    fn write_sockaddr_un(
        m: &mut GuestMem<'_>,
        addr: u64,
        len_out: u64,
        path: &str,
    ) -> Result<(), Errno> {
        const AF_UNIX: u16 = 1;
        let abstract_name = path.starts_with('\0');
        // sun_path's 108 bytes: all an abstract name's, or a path and its NUL.
        let max = if abstract_name { 108 } else { 107 };
        if addr != 0 {
            let mut buf = vec![0u8; 110];
            buf[..2].copy_from_slice(&AF_UNIX.to_le_bytes());
            let raw = path.as_bytes();
            let n = raw.len().min(max);
            buf[2..2 + n].copy_from_slice(&raw[..n]);
            let whole = 2 + n + usize::from(!abstract_name);
            let room = if len_out != 0 { m.read_u32(len_out)? as usize } else { whole };
            m.write(addr, &buf[..whole.min(room)])?;
        }
        // An abstract name's length is exactly its bytes (no terminating NUL:
        // the name may hold NULs), as Linux reports it; a path's counts one.
        if len_out != 0 {
            m.write_u32(len_out, (2 + path.len().min(max) + usize::from(!abstract_name)) as u32)?;
        }
        Ok(())
    }

    /// `close` ([`FdIo::close`]).
    fn sys_close(&mut self, fd: i32) -> SysResult {
        self.fd_io().close(fd)
    }

    /// `close_range`: shut every open descriptor in a span at once.
    fn sys_close_range(&mut self, first: u64, last: u64, flags: u64) -> SysResult {
        self.fd_io().close_range(first, last, flags)
    }



    /// The epoll set an open file is, if it is one.

    /// An open epoll set holding `watches`: what a fork's child rebuilds its
    /// parent's set as.
    pub fn adopt_epoll(watches: Vec<unix::Watch>, flags: u64) -> Arc<OpenFile> {
        let handle: Arc<dyn crate::vfs::FileHandle> = Arc::new(unix::EpollSet::with_watches(watches));
        OpenFile::new(handle, "anon_inode:[eventpoll]".into(), flags, true, true)
    }

    /// The epoll set behind a descriptor.
    fn epoll_of(&self, fd: i32) -> Result<Arc<unix::EpollSet>, Errno> {
        self.fd_io().epoll_of(fd)
    }

    fn sys_epoll_ctl(&mut self, m: &GuestMem<'_>, epfd: i32, op: u64, fd: i32, event: u64) -> SysResult {
        self.fd_io().epoll_ctl(m, epfd, op, fd, event)
    }



    /// `flock`. The open file description owns the lock, and its address is
    /// a stable name for it while it is open.








    fn socket_of(&self, fd: i32) -> Result<Arc<net::Socket>, Errno> {
        self.fd_io().socket_of(fd)
    }






    /// `getsockopt`. The only one anybody asks: has the connection come up?
    /// An unnamed `sockaddr_un`: the family alone, two bytes long.
    fn write_sockaddr_unnamed(m: &mut GuestMem<'_>, addr: u64, addrlen: u64) -> SysResult {
        if addr != 0 {
            let room = if addrlen != 0 { m.read_u32(addrlen)? } else { 2 };
            if room >= 2 {
                m.write(addr, &1u16.to_le_bytes())?;
            }
            if addrlen != 0 {
                m.write_u32(addrlen, 2)?;
            }
        }
        Ok(0)
    }













    fn sys_recvmsg(&self, m: &mut GuestMem<'_>, fd: i32, msg: u64, flags: u64) -> SysResult {
        self.fd_io().recvmsg(m, fd, msg, flags)
    }




    /// `recvmsg` flags. `MSG_DONTWAIT` is decided in `blocking_op`; here it
    /// only matters that an empty socket answers `EAGAIN`, which the stream
    /// already does. `MSG_PEEK` leaves the bytes queued.
    const MSG_PEEK: u64 = 0x2;
    const MSG_DONTWAIT: u64 = 0x40;



    // -- process and time ----------------------------------------------------

    /// `dup2` ([`FdIo::dup2`]).
    fn sys_dup2(&mut self, old: i32, new: i32) -> SysResult {
        self.fd_io().dup2(old, new)
    }

    fn sys_time(&mut self, m: &mut GuestMem<'_>, tloc: u64) -> SysResult {
        time_answer(m, self.compat, self.clock, tloc)
    }

    /// `alarm(s)`: `setitimer(ITIMER_REAL, {0, s})`, returning the seconds the
    /// previous timer had left. Both itimervals live in the thread's scratch
    /// area, since the call has none of its own.
    fn sys_alarm(&mut self, secs: u64) -> SysResult {
        Ok(alarm(&self.signals, self.now_nanos(), secs))
    }



    fn sys_clock_gettime(&mut self, m: &mut GuestMem<'_>, clk: u64, buf: u64) -> SysResult {
        // The two processor clocks count from when the process started, not
        // from the epoch. A program timing itself with one of these and
        // getting the wall clock instead sees a plausible-looking number that
        // is forty years too large, which subtracts away for a delta and does
        // not for anything else.
        const CLOCK_PROCESS_CPUTIME_ID: u64 = 2;
        const CLOCK_THREAD_CPUTIME_ID: u64 = 3;
        let (secs, nanos) = match clk {
            CLOCK_PROCESS_CPUTIME_ID | CLOCK_THREAD_CPUTIME_ID => {
                let t = match crate::proc::cpu_clock(clk == CLOCK_THREAD_CPUTIME_ID) {
                    Some(ns) => ns as u128,
                    None => self.cpu_nanos(),
                };
                ((t / 1_000_000_000) as u64, (t % 1_000_000_000) as u32)
            }
            c if is_monotonic_clock(c) => monotonic_from((self.clock)()),
            _ => guest_wall((self.clock)()),
        };
        m.write_u64(buf, secs)?;
        m.write_u64(buf + 8, nanos as u64)?;
        Ok(0)
    }

    fn sys_clock_getres(&mut self, m: &mut GuestMem<'_>, buf: u64) -> SysResult {
        if buf == 0 {
            return Ok(0);
        }
        m.write_u64(buf, 0)?;
        m.write_u64(buf + 8, 1)?;
        Ok(0)
    }

    fn sys_gettimeofday(&mut self, m: &mut GuestMem<'_>, buf: u64) -> SysResult {
        if buf == 0 {
            return Ok(0);
        }
        let (secs, nanos) = guest_wall((self.clock)());
        m.write_u64(buf, secs)?;
        m.write_u64(buf + 8, (nanos / 1000) as u64)?;
        Ok(0)
    }

    fn sys_nanosleep(&mut self, m: &GuestMem<'_>, req: u64) -> SysResult {
        nanosleep_answer(m, req)
    }

    /// The wall clock in nanoseconds as the guest reads it: an absolute
    /// CLOCK_REALTIME deadline the guest names is on this clock.
    fn guest_now_nanos(&mut self) -> u128 {
        let shift = WALL_SHIFT.load(std::sync::atomic::Ordering::Relaxed);
        (self.now_nanos() as i128 - shift as i128 * 1_000_000_000).max(0) as u128
    }

    /// How much time this process has used, in nanoseconds: wall clock since
    /// it started, not true processor time (which there is no portable way to
    /// ask for here). `openssl speed` divides by it, so zero printed every
    /// throughput as `infk`.
    pub fn cpu_nanos(&mut self) -> u128 {
        // a host that measures it (`crate::proc::set_cpu_clock`)
        if let Some(ns) = crate::proc::cpu_clock(false) {
            return ns as u128;
        }
        let now = self.now_nanos();
        if self.started == 0 {
            self.started = now;
        }
        now.saturating_sub(self.started)
    }

    /// `times`. Four `clock_t` -- user, system, and the same for reaped
    /// children -- counted in the hundredths of a second that `USER_HZ` means
    /// on Linux, and a return value on the same scale.
    fn sys_times(&mut self, m: &mut GuestMem<'_>, buf: u64) -> SysResult {
        let cpu = self.cpu_nanos();
        times_answer(m, self.compat, cpu, child_cpu_of(&self.procs, self.pid), buf)
    }

    /// `getrusage`. Only the two times are filled in; the page-fault and
    /// context-switch counters stay zero, which is what a program that asks
    /// for them under emulation should expect to see.
    fn sys_getrusage(&mut self, m: &mut GuestMem<'_>, who: u64, buf: u64) -> SysResult {
        let own = self.cpu_nanos();
        let ns = rusage_ns(
            who,
            || own,
            || child_cpu_of(&self.procs, self.pid),
            || crate::proc::cpu_clock(true).map_or(own, u128::from),
        )?;
        if buf != 0 && self.compat {
            // i386: 32-bit timevals and longs (72 bytes)
            m.zero(buf, 72)?;
            m.write_u32(buf, (ns / 1_000_000_000) as u32)?;
            m.write_u32(buf + 4, (ns % 1_000_000_000 / 1000) as u32)?;
        } else if buf != 0 {
            m.zero(buf, 144)?;
            write_timeval(m, buf, ns)?;
        }
        Ok(0)
    }

    fn sys_sysinfo(&mut self, m: &mut GuestMem<'_>, buf: u64) -> SysResult {
        sysinfo(m, self.compat, buf, &self.procs, self.clock)
    }

    fn sys_process_vm(&mut self, m: &mut GuestMem<'_>, write: bool, a: [u64; 6]) -> SysResult {
        process_vm(&self.procs, m, self.compat, write, a)
    }

    fn sys_getrandom(&mut self, m: &mut GuestMem<'_>, buf: u64, len: u64) -> SysResult {
        getrandom(&self.vfs, m, buf, len)
    }

    fn sys_prlimit(&mut self, m: &mut GuestMem<'_>, resource: u64, new: u64, old: u64) -> SysResult {
        prlimit(&self.rlimits, &self.fds, self.creds.uid(), m, resource, new, old)
    }
}

/// `statfs` and `fstatfs`: one answer for every filesystem (tmpfs's
/// magic, a 4 GiB volume half free). `f_flags` has ST_VALID, as Linux's
/// always does: glibc's `statvfs` otherwise looks the mount flags up in
/// /proc/mounts.
pub fn statfs(m: &mut GuestMem<'_>, compat: bool, buf: u64) -> SysResult {
    if buf == 0 {
        return Err(Errno::EFAULT);
    }
    let mut out = [0u8; 120];
    let put = |o: &mut [u8; 120], at: usize, v: u64| {
        o[at..at + 8].copy_from_slice(&v.to_le_bytes())
    };
    put(&mut out, 0, 0x0102_1994); // f_type: TMPFS_MAGIC
    put(&mut out, 8, 4096); // f_bsize
    put(&mut out, 16, 1 << 20); // f_blocks
    put(&mut out, 24, 1 << 19); // f_bfree
    put(&mut out, 32, 1 << 19); // f_bavail
    put(&mut out, 40, 1 << 18); // f_files
    put(&mut out, 48, 1 << 17); // f_ffree
    // f_fsid (8 bytes) at 56
    put(&mut out, 64, 255); // f_namelen
    put(&mut out, 72, 4096); // f_frsize
    put(&mut out, 80, ST_VALID); // f_flags
    if compat {
        // i386's statfs64 (and musl's wasm32 `struct statfs`): 32-bit
        // f_type, f_bsize, f_namelen, f_frsize, 64-bit counts
        let mut c = [0u8; 84];
        let w32 = |o: &mut [u8; 84], at: usize, v: u32| o[at..at + 4].copy_from_slice(&v.to_le_bytes());
        w32(&mut c, 0, 0x0102_1994);
        w32(&mut c, 4, 4096);
        c[8..48].copy_from_slice(&out[16..56]); // blocks, bfree, bavail, files, ffree
        w32(&mut c, 56, 255);
        w32(&mut c, 60, 4096);
        w32(&mut c, 64, ST_VALID as u32);
        m.write(buf, &c)?;
        return Ok(0);
    }
    m.write(buf, &out)?;
    Ok(0)
}

/// `struct statfs`'s `f_flags` bit saying the flags are filled in.
const ST_VALID: u64 = 0x20;

/// `sysinfo`: a constant machine (4 GiB, half free, one process).
pub fn sysinfo(m: &mut GuestMem<'_>, compat: bool, buf: u64, procs: &ProcTable, clock: ClockFn) -> SysResult {
    if buf == 0 {
        return Err(Errno::EFAULT);
    }
    // As `/proc` has them (`vfs::procfs`): the monotonic clock's uptime, the
    // process table's load averages (fixed point, 1 << 16 is 1.0), free
    // memory a fixed 4 GiB less what the processes have resident, and their
    // count.
    let up_ns = clock_ns(clock, 1);
    let loads = procs.loadavg(up_ns).map(|h| h as u64 * 65536 / 100);
    let (rss, count) = procs.memory();
    let free = (4u64 << 30).saturating_sub(rss);
    let count = count.clamp(1, u16::MAX as usize) as u16;
    // struct sysinfo: uptime, loads[3], totalram, freeram, sharedram,
    // bufferram, totalswap, freeswap, procs (u16), totalhigh, freehigh,
    // mem_unit. The 64-bit layout has 8-byte longs (112 bytes); the i386
    // one 4-byte longs (64 bytes), where 4 GiB does not fit and memory
    // is counted in pages through mem_unit. glibc's qsort asks sysconf
    // for the physical pages once a sort has 256 elements, and its
    // caller's frame is exactly this struct's size: the wide layout in a
    // 32-bit process overwrote the stack canary.
    if compat {
        m.zero(buf, 64)?;
        m.write_u32(buf, (up_ns / 1_000_000_000) as u32)?; // uptime
        for (i, l) in loads.iter().enumerate() {
            m.write_u32(buf + 4 + 4 * i as u64, *l as u32)?;
        }
        m.write_u32(buf + 16, 1 << 20)?; // totalram: 4 GiB in pages
        m.write_u32(buf + 20, (free >> 12) as u32)?; // freeram
        m.write(buf + 40, &count.to_le_bytes())?; // procs
        m.write_u32(buf + 52, 4096)?; // mem_unit
    } else {
        m.zero(buf, 112)?;
        m.write_u64(buf, up_ns / 1_000_000_000)?; // uptime
        for (i, l) in loads.iter().enumerate() {
            m.write_u64(buf + 8 + 8 * i as u64, *l)?;
        }
        m.write_u64(buf + 32, 4 * 1024 * 1024 * 1024)?; // totalram
        m.write_u64(buf + 40, free)?; // freeram
        m.write(buf + 80, &count.to_le_bytes())?; // procs
        m.write_u32(buf + 104, 1)?; // mem_unit
    }
    Ok(0)
}

/// `prlimit64(pid, resource, new, old)`, for the calling process (gex's
/// processes set only their own). Limits are recorded and reported;
/// RLIMIT_NOFILE is enforced by the descriptor table, the rest are not.
/// Under the limits' own lock, read, checked and set at once.
pub fn prlimit(rl: &Rlimits, fds: &FdTable, uid: u32, m: &mut GuestMem<'_>, resource: u64, new: u64, old: u64) -> SysResult {
    let r = resource as usize;
    if r >= RLIM_NLIMITS {
        return Err(Errno::EINVAL);
    }
    let mut limits = rl.0.lock().unwrap_or_else(|e| e.into_inner());
    if r == RLIMIT_NOFILE {
        limits[r][0] = fds.limit() as u64;
    }
    let cur = limits[r];
    let want = if new != 0 { Some([m.read_u64(new)?, m.read_u64(new + 8)?]) } else { None };
    if let Some([soft, hard]) = want {
        if soft > hard {
            return Err(Errno::EINVAL);
        }
        // raising the hard limit takes privilege; NOFILE cannot pass
        // what the table holds
        if (hard > cur[1] && uid != 0)
            || (r == RLIMIT_NOFILE && hard > crate::file::NOFILE_HARD as u64)
        {
            return Err(Errno::EPERM);
        }
    }
    if old != 0 {
        m.write_u64(old, cur[0])?;
        m.write_u64(old + 8, cur[1])?;
    }
    if let Some(lim) = want {
        limits[r] = lim;
        if r == RLIMIT_NOFILE {
            fds.set_limit(lim[0] as usize);
        }
    }
    Ok(0)
}

/// Whether canonical syscall `n` can change another thread's descriptor
/// readiness (or signal state): what a host parking threads on descriptors
/// must wake them for. Over-approximate (any read or write, of any file);
/// futexes, clocks, memory and the like are left out because they are the
/// hot calls and never make a descriptor ready.
pub fn wakes_fd_waiters(n: u64) -> bool {
    matches!(
        n,
        nr::WRITE
            | nr::WRITEV
            | nr::PWRITE64
            | nr::READ
            | nr::READV
            | nr::PREAD64
            | nr::SENDTO
            | nr::SENDMSG
            | nr::SENDMMSG
            | nr::SENDFILE
            | nr::SPLICE
            | nr::RECVFROM
            | nr::RECVMSG
            | nr::RECVMMSG
            | nr::CLOSE
            | nr::CLOSE_RANGE
            | nr::DUP3
            | nr::SHUTDOWN
            | nr::CONNECT
            | nr::ACCEPT
            | nr::ACCEPT4
            | nr::LISTEN
            | nr::EXIT
            | nr::EXIT_GROUP
            | nr::KILL
            | nr::TKILL
            | nr::TGKILL
            | nr::TIMERFD_SETTIME
            // A lock released: a blocked `F_SETLKW` or `flock` looks again.
            | nr::FCNTL
            | nr::FLOCK
            | nr::DUP2
    )
}

/// Name a syscall for trace output.
pub fn syscall_name(n: u64) -> &'static str {
    match n {
        nr::DUP2 => "dup2",
        nr::TIME => "time",
        nr::ALARM => "alarm",
        nr::ARCH_PRCTL => "arch_prctl",
        nr::NOOP_OK => "advisory-noop",
        nr::MLOCK => "mlock",
        nr::MUNLOCK => "munlock",
        nr::MLOCKALL => "mlockall",
        nr::MUNLOCKALL => "munlockall",
        nr::MLOCK2 => "mlock2",
        nr::GETCWD => "getcwd",
        nr::DUP => "dup",
        nr::DUP3 => "dup3",
        nr::FCNTL => "fcntl",
        nr::IOCTL => "ioctl",
        nr::MKDIRAT => "mkdirat",
        nr::UNLINKAT => "unlinkat",
        nr::LINKAT => "linkat",
        nr::SYMLINKAT => "symlinkat",
        nr::RENAMEAT => "renameat",
        nr::RENAMEAT2 => "renameat2",
        nr::FTRUNCATE => "ftruncate",
        nr::TRUNCATE => "truncate",
        nr::FADVISE64 => "fadvise64",
        nr::FACCESSAT | nr::FACCESSAT2 => "faccessat",
        nr::CHDIR => "chdir",
        nr::FCHDIR => "fchdir",
        nr::UNSHARE => "unshare",
        nr::FCHMOD => "fchmod",
        nr::FCHMODAT => "fchmodat",
        nr::SETXATTR => "setxattr",
        nr::LSETXATTR => "lsetxattr",
        nr::FSETXATTR => "fsetxattr",
        nr::GETXATTR => "getxattr",
        nr::LGETXATTR => "lgetxattr",
        nr::FGETXATTR => "fgetxattr",
        nr::LISTXATTR => "listxattr",
        nr::LLISTXATTR => "llistxattr",
        nr::FLISTXATTR => "flistxattr",
        nr::REMOVEXATTR => "removexattr",
        nr::LREMOVEXATTR => "lremovexattr",
        nr::FREMOVEXATTR => "fremovexattr",
        nr::OPENAT => "openat",
        nr::OPENAT2 => "openat2",
        nr::CLOSE => "close",
        nr::PIPE2 => "pipe2",
        nr::EVENTFD2 => "eventfd2",
        nr::INOTIFY_INIT1 => "inotify_init1",
        nr::INOTIFY_ADD_WATCH => "inotify_add_watch",
        nr::INOTIFY_RM_WATCH => "inotify_rm_watch",
        nr::TIMERFD_CREATE => "timerfd_create",
        nr::TIMERFD_SETTIME => "timerfd_settime",
        nr::TIMERFD_GETTIME => "timerfd_gettime",
        nr::GETDENTS64 => "getdents64",
        nr::LSEEK => "lseek",
        nr::READ => "read",
        nr::WRITE => "write",
        nr::READV => "readv",
        nr::WRITEV => "writev",
        nr::PREAD64 => "pread64",
        nr::PWRITE64 => "pwrite64",
        nr::SENDFILE => "sendfile",
        nr::COPY_FILE_RANGE => "copy_file_range",
        nr::SPLICE => "splice",
        nr::PPOLL => "ppoll",
        nr::PSELECT6 => "pselect6",
        nr::UTIMENSAT => "utimensat",
        nr::FCHOWN => "fchown",
        nr::FCHOWNAT => "fchownat",
        nr::STATFS => "statfs",
        nr::FSTATFS => "fstatfs",
        nr::UMASK => "umask",
        nr::READLINKAT => "readlinkat",
        nr::NEWFSTATAT => "newfstatat",
        nr::FSTAT => "fstat",
        nr::FSYNC => "fsync",
        nr::FDATASYNC => "fdatasync",
        nr::SYNC_FILE_RANGE => "sync_file_range",
        nr::SYNCFS => "syncfs",
        nr::EXIT => "exit",
        nr::EXIT_GROUP => "exit_group",
        nr::SET_TID_ADDRESS => "set_tid_address",
        nr::FUTEX => "futex",
        nr::SET_ROBUST_LIST => "set_robust_list",
        nr::NANOSLEEP => "nanosleep",
        nr::CLOCK_GETTIME => "clock_gettime",
        nr::CLOCK_GETRES => "clock_getres",
        nr::CLOCK_NANOSLEEP => "clock_nanosleep",
        nr::SETITIMER => "setitimer",
        nr::GETITIMER => "getitimer",
        nr::SCHED_GETAFFINITY => "sched_getaffinity",
        nr::SCHED_YIELD => "sched_yield",
        nr::SCHED_SETPARAM => "sched_setparam",
        nr::SCHED_SETSCHEDULER => "sched_setscheduler",
        nr::SCHED_GETSCHEDULER => "sched_getscheduler",
        nr::SCHED_GETPARAM => "sched_getparam",
        nr::SCHED_GET_PRIORITY_MAX => "sched_get_priority_max",
        nr::SCHED_GET_PRIORITY_MIN => "sched_get_priority_min",
        nr::SETPRIORITY => "setpriority",
        nr::GETPRIORITY => "getpriority",
        nr::GETCPU => "getcpu",
        nr::KILL => "kill",
        nr::TKILL => "tkill",
        nr::TGKILL => "tgkill",
        nr::RT_SIGACTION => "rt_sigaction",
        nr::RT_SIGPROCMASK => "rt_sigprocmask",
        nr::RT_SIGPENDING => "rt_sigpending",
        nr::RT_SIGRETURN => "rt_sigreturn",
        nr::SIGALTSTACK => "sigaltstack",
        nr::TIMES => "times",
        nr::UNAME => "uname",
        nr::GETRUSAGE => "getrusage",
        nr::GETTIMEOFDAY => "gettimeofday",
        nr::GETPID => "getpid",
        nr::GETPPID => "getppid",
        nr::SETSID => "setsid",
        nr::GETSID => "getsid",
        nr::SETPGID => "setpgid",
        nr::GETPGID => "getpgid",
        nr::SETUID => "setuid",
        nr::SETGID => "setgid",
        nr::SETREUID => "setreuid",
        nr::SETREGID => "setregid",
        nr::SETRESUID => "setresuid",
        nr::SETRESGID => "setresgid",
        nr::GETRESUID => "getresuid",
        nr::GETRESGID => "getresgid",
        nr::GETGROUPS => "getgroups",
        nr::SETGROUPS => "setgroups",
        nr::PRCTL => "prctl",
        nr::CHROOT => "chroot",
        nr::CAPGET => "capget",
        nr::CAPSET => "capset",
        nr::SETFSUID => "setfsuid",
        nr::SETFSGID => "setfsgid",
        nr::SYNC => "sync",
        nr::STATX => "statx",
        nr::MOUNT => "mount",
        nr::UMOUNT2 => "umount2",
        nr::REBOOT => "reboot",
        nr::RT_SIGTIMEDWAIT => "rt_sigtimedwait",
        nr::SOCKET => "socket",
        nr::BIND => "bind",
        nr::LISTEN => "listen",
        nr::ACCEPT4 => "accept4",
        nr::MEMFD_CREATE => "memfd_create",
        nr::FLOCK => "flock",
        nr::FALLOCATE => "fallocate",
        nr::EPOLL_CREATE1 => "epoll_create1",
        nr::EPOLL_CTL => "epoll_ctl",
        nr::EPOLL_PWAIT => "epoll_pwait",
        nr::EPOLL_PWAIT2 => "epoll_pwait2",
        nr::CLOSE_RANGE => "close_range",
        nr::MEMBARRIER => "membarrier",
        nr::ACCEPT => "accept",
        nr::GETSOCKNAME => "getsockname",
        nr::GETPEERNAME => "getpeername",
        nr::RECVFROM => "recvfrom",
        nr::SETSOCKOPT => "setsockopt",
        nr::GETSOCKOPT => "getsockopt",
        nr::SHUTDOWN => "shutdown",
        nr::SENDMSG => "sendmsg",
        nr::RECVMSG => "recvmsg",
        nr::SENDMMSG => "sendmmsg",
        nr::RECVMMSG => "recvmmsg",
        nr::SOCKETPAIR => "socketpair",
        nr::CONNECT => "connect",
        nr::SENDTO => "sendto",
        nr::GETUID => "getuid",
        nr::GETEUID => "geteuid",
        nr::GETGID => "getgid",
        nr::GETEGID => "getegid",
        nr::GETTID => "gettid",
        nr::SYSINFO => "sysinfo",
        nr::BRK => "brk",
        nr::MUNMAP => "munmap",
        nr::MREMAP => "mremap",
        nr::CLONE => "clone",
        nr::CLONE3 => "clone3",
        nr::EXECVE => "execve",
        nr::MMAP => "mmap",
        nr::MPROTECT => "mprotect",
        nr::MADVISE => "madvise",
        nr::MSYNC => "msync",
        nr::WAIT4 => "wait4",
        nr::WAITID => "waitid",
        nr::PRLIMIT64 => "prlimit64",
        nr::PIDFD_OPEN => "pidfd_open",
        nr::PIDFD_SEND_SIGNAL => "pidfd_send_signal",
        nr::GETRANDOM => "getrandom",
        nr::PROCESS_VM_READV => "process_vm_readv",
        nr::PROCESS_VM_WRITEV => "process_vm_writev",
        nr::RSEQ => "rseq",
        _ => "unknown",
    }
}

/// Linux's per-call limit on read and write sizes (`INT_MAX & PAGE_MASK`).
const MAX_RW_COUNT: u64 = 0x7fff_f000;

/// A read's length: at most MAX_RW_COUNT, and no further than the guest's
/// memory goes from `buf` (Linux faults only on a page it touches, so a
/// length past the end with a short file still reads; a `buf` outside the
/// memory keeps its length and faults).
fn rw_len(m: &GuestMem<'_>, buf: u64, len: u64) -> usize {
    let room = (m.bytes.len() as u64).saturating_sub(buf);
    (if room == 0 { len.min(MAX_RW_COUNT) } else { len.min(MAX_RW_COUNT).min(room) }) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::TmpFs;

    /// Linux's 64-bit `struct statfs`: seven 8-byte counts and sizes, the
    /// 8-byte fsid, then f_namelen, f_frsize, f_flags. The name length was
    /// one field late (in f_frsize's place), so programs saw a 4096-byte
    /// name limit and a fragment size of 0.
    #[test]
    fn statfs_fields_are_where_linux_puts_them() {
        let mut buf = vec![0u8; 0x1000];
        statfs(&mut GuestMem::new(&mut buf), false, 0x100).unwrap();
        let q = |at: usize| u64::from_le_bytes(buf[0x100 + at..0x108 + at].try_into().unwrap());
        assert_eq!(q(8), 4096, "f_bsize");
        assert_eq!(q(56), 0, "f_fsid");
        assert_eq!(q(64), 255, "f_namelen");
        assert_eq!(q(72), 4096, "f_frsize");
        assert_eq!(q(80) & ST_VALID, ST_VALID, "f_flags");
        let mut buf = vec![0u8; 0x1000];
        statfs(&mut GuestMem::new(&mut buf), true, 0x100).unwrap();
        let d = |at: usize| u32::from_le_bytes(buf[0x100 + at..0x104 + at].try_into().unwrap());
        assert_eq!((d(56), d(60), d(64) & 0x20), (255, 4096, 0x20), "i386 statfs64: f_namelen, f_frsize, f_flags");
    }

    #[test]
    fn poll_skips_negative_descriptors() {
        let slot = |fd| PollFd { addr: 0, fd, events: 1, handle: None, writable: false, epoll: None, edge: false };
        // An unused slot reports nothing; a closed descriptor is POLLNVAL.
        assert_eq!(poll_revents(&slot(-1)), 0);
        assert_eq!(poll_revents(&slot(7)), 0x020);
        assert!(!poll_ready(&[slot(-1), slot(-1)]));
    }

    /// A process's memory for another's `process_vm_*`: a plain buffer.
    struct BufPeer(std::sync::Mutex<Vec<u8>>);

    impl crate::proc::PeerMem for BufPeer {
        fn read(&self, addr: u64, out: &mut [u8]) -> bool {
            let b = self.0.lock().unwrap();
            let a = addr as usize;
            a + out.len() <= b.len() && {
                out.copy_from_slice(&b[a..a + out.len()]);
                true
            }
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            let mut b = self.0.lock().unwrap();
            let a = addr as usize;
            a + data.len() <= b.len() && {
                b[a..a + data.len()].copy_from_slice(data);
                true
            }
        }
    }

    /// `process_vm_readv`/`writev` against a second process whose map has
    /// one read-write page and one read-only page after it: a
    /// transfer stops at the first page the direction may not touch, with
    /// what it moved before; `EFAULT` when that is nothing; `EINVAL` for
    /// flags; `ESRCH` for no such process and for one that has exited.
    #[test]
    fn process_vm_moves_what_the_target_map_allows() {
        use crate::mem::{flags as mf, prot as mp};
        let (mut mem, mut k) = setup();
        let mut peer = k.fork_kernel();
        peer.pid = k.procs.spawn(k.pid);
        let b = gex_abi::MMAP_BOTTOM;
        let fl = mf::PRIVATE | mf::ANONYMOUS | mf::FIXED;
        peer.mem.lock().mmap(b, 0x1000, mp::READ | mp::WRITE, fl, "rw").unwrap();
        peer.mem.lock().mmap(b + 0x1000, 0x1000, mp::READ, fl, "ro").unwrap();
        let mut bytes = vec![0u8; (b + 0x3000) as usize];
        bytes[b as usize..(b + 0x2000) as usize].iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
        let pm = Arc::new(BufPeer(std::sync::Mutex::new(bytes)));
        let vm = crate::proc::PeerVm { mm: peer.mem.shared(), mem: pm.clone() };
        k.procs.set_vm(peer.pid, Some(Arc::new(vm)));
        let mut m = GuestMem::new(&mut mem);
        // Local iovec at 0x100, remote at 0x200: (base, len) pairs.
        let iov = |m: &mut GuestMem<'_>, at: u64, base: u64, len: u64| {
            m.write_u64(at, base).unwrap();
            m.write_u64(at + 8, len).unwrap();
        };
        let call = |k: &mut Kernel, m: &mut GuestMem<'_>, write: bool, pid: i32, flags: u64| {
            k.sys_process_vm(m, write, [pid as u64, 0x100, 1, 0x200, 1, flags])
        };
        let pid = peer.pid;
        iov(&mut m, 0x100, 0x2000, 0x3000);
        iov(&mut m, 0x200, b, 0x3000);
        assert_eq!(call(&mut k, &mut m, false, pid, 0), Ok(0x2000), "a read stops past the mapped pages");
        assert_eq!(m.read(0x2000 + 0x1fff, 1).unwrap()[0], 0xff);
        iov(&mut m, 0x200, b + 0x2000, 0x10);
        assert_eq!(call(&mut k, &mut m, false, pid, 0), Err(Errno::EFAULT), "nothing mapped there");
        m.write(0x3000, &[9, 8, 7, 6]).unwrap();
        iov(&mut m, 0x100, 0x3000, 4);
        iov(&mut m, 0x200, b + 0xffe, 4);
        assert_eq!(call(&mut k, &mut m, true, pid, 0), Ok(2), "a write stops at the read-only page");
        assert_eq!(&pm.0.lock().unwrap()[(b + 0xffe) as usize..(b + 0x1002) as usize], &[9, 8, 0, 1]);
        iov(&mut m, 0x200, b + 0x1000, 4);
        assert_eq!(call(&mut k, &mut m, true, pid, 0), Err(Errno::EFAULT));
        assert_eq!(call(&mut k, &mut m, false, pid, 1), Err(Errno::EINVAL));
        assert_eq!(call(&mut k, &mut m, false, 9999, 0), Err(Errno::ESRCH));
        k.procs.exit(pid, 0);
        assert_eq!(call(&mut k, &mut m, false, pid, 0), Err(Errno::ESRCH), "an exited process");
    }

    fn setup() -> (Vec<u8>, Kernel) {
        let mem = vec![0u8; 0x40000];
        let mut k = Kernel::new(0x1000, 0x10000, "/prog".into());
        k.vfs.mount("/", Arc::new(TmpFs::new()));
        k.vfs.mount("/dev", Arc::new(crate::vfs::DevFs::new()));
        (mem, k)
    }

    /// `F_GETFL` reports the access mode a descriptor can actually do, even
    /// when it was installed with flags of 0 (inherited stdio), and `F_SETFL`
    /// changes only the status flags. libuv's `uv_pipe_open` reads the mode to
    /// decide if a stream is writable; node's stdout said EPIPE on every write.
    #[test]
    fn getfl_reports_the_access_mode_and_setfl_keeps_it() {
        let (_mem, mut k) = setup();
        const F_GETFL: u64 = 3;
        const F_SETFL: u64 = 4;
        let r = Arc::new(Pipe::new());
        let wfd = k
            .fds
            .insert(
                OpenFile::new(Arc::new(crate::file::PipeEnd { pipe: r, write: true }), "pipe".into(), 0, false, true),
                false,
            )
            .unwrap() as i32;
        assert_eq!(k.sys_fcntl(wfd, F_GETFL, 0), Ok(oflags::O_WRONLY));
        assert_eq!(k.sys_fcntl(wfd, F_SETFL, oflags::O_NONBLOCK), Ok(0));
        assert_eq!(k.sys_fcntl(wfd, F_GETFL, 0), Ok(oflags::O_WRONLY | oflags::O_NONBLOCK));
        let (a, _b) = unix::UnixSocket::pair();
        let sfd = install_unix(&k.fd_io(), a, 0, false).unwrap() as i32;
        assert_eq!(k.sys_fcntl(sfd, F_GETFL, 0), Ok(oflags::O_RDWR));
        // And the socket says what it is, which is how libuv tells a pipe
        // from TCP (`uv_guess_handle`).
        assert_eq!(socket_family_type(&k.fd_io(), sfd), Ok((1, 1)));
        assert_eq!(socket_family_type(&k.fd_io(), wfd), Err(Errno::ENOTSOCK));
    }

    /// A unix socket that arrived over SCM_RIGHTS is placed straight in the
    /// descriptor table, never through `install_unix`; it must still be one
    /// to the calls that pass descriptors. (A per-process side table made it
    /// ENOTSOCK there, and `sendmsg`/`recvmsg` then carried no descriptors.)
    #[test]
    fn a_received_unix_socket_is_still_a_unix_socket() {
        let (_mem, mut k) = setup();
        let (a, _b) = unix::UnixSocket::pair();
        let fd = k
            .fds
            .insert(OpenFile::new(a.clone(), "socket".into(), 0, true, true), false)
            .unwrap() as i32;
        assert!(Arc::ptr_eq(&k.fd_io().unix_of(fd).unwrap(), &a));
        // Anything else is not.
        let r = Arc::new(Pipe::new());
        let pfd = k
            .fds
            .insert(
                OpenFile::new(Arc::new(crate::file::PipeEnd { pipe: r, write: false }), "pipe".into(), 0, true, false),
                false,
            )
            .unwrap() as i32;
        assert_eq!(k.fd_io().unix_of(pfd).err(), Some(Errno::ENOTSOCK));
    }

    /// `times` and `getrusage(RUSAGE_CHILDREN)` count the reaped children's
    /// time, the kernel's and the free path alike; a `who` past the three
    /// is `EINVAL`.
    #[test]
    fn reaped_children_time_in_times_and_getrusage() {
        if gex_abi::X86 {
            return;
        }
        let (mut mem, mut k) = setup();
        let at = k.ctx_addr;
        let buf = 0x3000u64;
        let child = k.procs.spawn(k.pid);
        k.procs.set_usage(child, 2_500_000_000, 0, 0, 0);
        k.procs.exit(child, 0);
        k.procs.reap(child);
        let procs = Arc::clone(&k.procs);
        let pid = k.pid;
        let children = move || child_cpu_of(&procs, pid);
        for free in [false, true] {
            let mut m = GuestMem::new(&mut mem);
            let mut call = |m: &mut GuestMem<'_>, n: u64, args: &[u64]| {
                m.zero(buf, 160).unwrap();
                set_args(m, at, n, args);
                if free {
                    assert!(clock_call(m, at, false, k.clock, k.started, &children));
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            assert_eq!(call(&mut m, nr::GETRUSAGE, &[(-1i64) as u64, buf]), 0, "free {free}");
            assert_eq!((m.read_u64(buf).unwrap(), m.read_u64(buf + 8).unwrap()), (2, 500_000), "free {free}");
            call(&mut m, nr::TIMES, &[buf]);
            assert_eq!(m.read_u64(buf + 16).unwrap(), 250, "cutime, free {free}");
            assert_eq!(call(&mut m, nr::GETRUSAGE, &[7, buf]), -(Errno::EINVAL.0 as i64), "free {free}");
            assert_eq!(call(&mut m, nr::GETRUSAGE, &[1, 0]), 0, "RUSAGE_THREAD, free {free}");
        }
    }

    /// `ident_call` answers the id calls as the kernel does, and declines
    /// what it was not given.
    #[test]
    fn ident_call_answers_as_the_kernel_does() {
        if gex_abi::X86 {
            return;
        }
        let (mut mem, mut k) = setup();
        let at = k.ctx_addr;
        let ids = Idents { pid: k.pid as u32, ppid: 0, tid: k.tid as u32 };
        for n in [nr::GETPID, nr::GETTID, nr::GETUID, nr::GETEUID, nr::GETGID, nr::GETEGID] {
            let mut m = GuestMem::new(&mut mem);
            set_args(&mut m, at, n, &[]);
            k.syscall(&mut m, 0);
            let want = ret(&m, at);
            m.write_u64(at + ctx::x(0) as u64, 0x5555).unwrap();
            set_args(&mut m, at, n, &[]);
            assert!(ident_call(&mut m, at, false, &ids, &k.creds), "call {n}");
            assert_eq!(ret(&m, at), want, "call {n}");
        }
        let mut m = GuestMem::new(&mut mem);
        set_args(&mut m, at, nr::GETPPID, &[]);
        assert!(!ident_call(&mut m, at, false, &ids, &k.creds), "no parent given");
        set_args(&mut m, at, nr::GETPID, &[]);
        assert!(!ident_call(&mut m, at, false, &Idents { tid: 0, ..ids }, &k.creds), "no thread yet");
        set_args(&mut m, at, nr::CLOCK_GETTIME, &[0, 0x3000]);
        assert!(!ident_call(&mut m, at, false, &ids, &k.creds), "not an id call");
    }

    /// `clock_call`, the lock-free answer, writes what the full call writes,
    /// in both numberings and layouts, and declines what it cannot answer.
    #[test]
    fn clock_call_answers_as_the_kernel_does() {
        fn fixed() -> (u64, u32) {
            (1_700_000_000, 123_456_789)
        }
        // The monotonic clocks' base is one per host process, set by the
        // first monotonic reading. Set it from the real clock before `fixed`
        // can: from 2023's `fixed` it made every later reading in this test
        // binary an uptime of years (the_monotonic_clock_counts_from_boot_
        // not_the_epoch failed when this test ran first).
        monotonic_from(system_clock());
        let (mut mem, mut k) = setup();
        k.clock = fixed;
        let buf = 0x3000u64;
        let at = k.ctx_addr;
        // Native numbering: each call made both ways into a cleared buffer.
        for (n, args) in [
            (nr::CLOCK_GETTIME, [0u64, buf]),
            (nr::CLOCK_GETTIME, [1, buf]),
            (nr::CLOCK_GETTIME, [3, buf]),
            (nr::GETTIMEOFDAY, [buf, 0]),
            (nr::GETRUSAGE, [0, buf]),
            (nr::TIMES, [buf, 0]),
            (nr::TIMES, [0, 0]),
            (if gex_abi::X86 { nr::TIME } else { nr::TIMES }, [buf, 0]),
            (nr::SCHED_YIELD, [0, 0]),
        ] {
            let mut m = GuestMem::new(&mut mem);
            m.zero(buf, 160).unwrap();
            set_args(&mut m, at, n, &args);
            k.syscall(&mut m, 0);
            let want = m.read(buf, 160).unwrap().to_vec();
            let want_ret = ret(&m, at);
            m.zero(buf, 160).unwrap();
            m.write_u64(at + ctx::x(0) as u64, 0x5555).unwrap();
            set_args(&mut m, at, n, &args);
            assert!(clock_call(&mut m, at, false, fixed, k.started, &|| 0), "call {n}");
            assert_eq!(m.read(buf, 160).unwrap(), &want[..], "call {n} {args:?}");
            assert_eq!(ret(&m, at), want_ret, "call {n}");
        }
        // Declined: another call. Answered in place: a null clock buffer
        // (EFAULT, as the kernel's), the processor clock before the
        // process's start is known (no time used), and the resolution.
        let mut m = GuestMem::new(&mut mem);
        set_args(&mut m, at, nr::GETPID, &[]);
        assert!(!clock_call(&mut m, at, false, fixed, k.started, &|| 0));
        set_args(&mut m, at, nr::CLOCK_GETTIME, &[0, 0]);
        assert!(clock_call(&mut m, at, false, fixed, k.started, &|| 0));
        assert_eq!(ret(&m, at) as i64, -(Errno::EFAULT.0 as i64));
        m.write_u64(buf, !0).unwrap();
        set_args(&mut m, at, nr::GETRUSAGE, &[0, buf]);
        assert!(clock_call(&mut m, at, false, fixed, 0, &|| 0));
        assert_eq!(m.read_u64(buf).unwrap(), 0);
        assert_eq!(ret(&m, at), 0);
        set_args(&mut m, at, nr::CLOCK_GETRES, &[1, buf]);
        assert!(clock_call(&mut m, at, false, fixed, k.started, &|| 0));
        assert_eq!((m.read_u64(buf).unwrap(), m.read_u64(buf + 8).unwrap()), (0, 1));
        // i386: the number in eax, arguments in ebx/ecx, 32-bit layouts.
        let set32 = |m: &mut GuestMem<'_>, n: u64, a: [u64; 2]| {
            m.write_u64(at + ctx::x(0) as u64, n).unwrap();
            m.write_u64(at + crate::i386::ARG_SLOTS[0] as u64, a[0]).unwrap();
            m.write_u64(at + crate::i386::ARG_SLOTS[1] as u64, a[1]).unwrap();
        };
        let started = 1_699_999_990u128 * 1_000_000_000;
        set32(&mut m, crate::i386::raw::CLOCK_GETTIME, [0, buf]);
        assert!(clock_call(&mut m, at, true, fixed, started, &|| 0));
        assert_eq!(m.read_u32(buf).unwrap(), 1_700_000_000);
        assert_eq!(m.read_u32(buf + 4).unwrap(), 123_456_789);
        set32(&mut m, crate::i386::raw::GETTIMEOFDAY, [buf, 0]);
        assert!(clock_call(&mut m, at, true, fixed, started, &|| 0));
        assert_eq!(m.read_u32(buf + 4).unwrap(), 123_456);
        m.write_u64(buf + 64, !0).unwrap();
        m.write_u64(buf + 72, !0).unwrap();
        set32(&mut m, crate::i386::raw::GETRUSAGE, [0, buf]);
        assert!(clock_call(&mut m, at, true, fixed, started, &|| 0));
        assert_eq!(m.read_u32(buf).unwrap(), 10);
        assert_eq!(m.read_u32(buf + 4).unwrap(), 123_456);
        assert_eq!(m.read_u64(buf + 64).unwrap(), 0, "the 72-byte i386 rusage is cleared");
        assert_eq!(m.read_u64(buf + 72).unwrap(), !0, "and nothing past it");
        set32(&mut m, crate::i386::raw::SCHED_YIELD, [0, 0]);
        assert!(clock_call(&mut m, at, true, fixed, started, &|| 0));
        assert_eq!(m.read_u64(at + ctx::x(0) as u64).unwrap(), 0);
        set32(&mut m, crate::i386::raw::GETPID, [0, 0]);
        assert!(!clock_call(&mut m, at, true, fixed, started, &|| 0));
    }

    /// What `clock_nanosleep` turns into: a sleep of the time left for an
    /// absolute deadline (none if it has passed), EINVAL for a malformed
    /// request instead of a sleep.
    #[test]
    fn the_monotonic_clock_counts_from_boot_not_the_epoch() {
        let (mut mem, mut k) = setup();
        let at = 0x2000u64;
        let mut read = |k: &mut Kernel, mem: &mut Vec<u8>, clk: u64| {
            let mut m = GuestMem::new(mem);
            k.sys_clock_gettime(&mut m, clk, at).unwrap();
            m.read_u64(at).unwrap()
        };
        let wall = read(&mut k, &mut mem, 0);
        let mono = read(&mut k, &mut mem, 1);
        // An uptime: milliseconds fit the 32 bits X server times have.
        assert!(mono >= 1 && mono < 49 * 86400, "monotonic {mono}");
        assert!(wall > 1_000_000_000, "realtime {wall}");
        // BOOTTIME and COARSE agree with it.
        for clk in [6, 7] {
            let other = read(&mut k, &mut mem, clk);
            assert!(other.abs_diff(mono) <= 1, "clock {clk}: {other} vs {mono}");
        }
    }

    /// `writev` to a full blocking pipe parks on its first non-empty vector,
    /// as `write` does, and `readv` on an empty one parks as `read` does --
    /// they had answered `EAGAIN` on a blocking descriptor.
    #[test]
    fn vectored_io_on_a_blocked_pipe_parks() {
        let (mut mem, mut k) = setup();
        let p = Arc::new(Pipe::new());
        let end = |write| OpenFile::new(Arc::new(crate::file::PipeEnd { pipe: Arc::clone(&p), write }), "pipe".into(), 0, !write, write);
        let rfd = k.fds.insert(end(false), false).unwrap() as u64;
        let wfd = k.fds.insert(end(true), false).unwrap() as u64;
        // Two vectors, the first empty: 0x3000 (len 0) and 0x3100 (len 100).
        let iov = 0x2000usize;
        for (i, (b, l)) in [(0x3000u64, 0u64), (0x3100, 100)].into_iter().enumerate() {
            mem[iov + 16 * i..iov + 16 * i + 8].copy_from_slice(&b.to_le_bytes());
            mem[iov + 16 * i + 8..iov + 16 * i + 16].copy_from_slice(&l.to_le_bytes());
        }
        let op = |k: &mut Kernel, mem: &mut Vec<u8>, n| k.blocking_op(&GuestMem::new(mem), n, [if n == nr::READV { rfd } else { wfd }, iov as u64, 2, 0, 0, 0]);
        match op(&mut k, &mut mem, nr::READV) {
            Some(BlockOp::ReadStream { buf: 0x3100, len: 100, .. }) => {}
            other => panic!("readv on an empty pipe: {other:?}"),
        }
        assert!(op(&mut k, &mut mem, nr::WRITEV).is_none(), "writev to an empty pipe goes straight through");
        p.write(&vec![0u8; crate::pipe::PIPE_BUF]).unwrap();
        match op(&mut k, &mut mem, nr::WRITEV) {
            Some(BlockOp::WriteStream { buf: 0x3100, len: 100, .. }) => {}
            other => panic!("writev to a full pipe: {other:?}"),
        }
        assert!(op(&mut k, &mut mem, nr::READV).is_none(), "readv on a full pipe goes straight through");
    }

    /// A read or write that met `EAGAIN` on a blocking stream after all
    /// (the room check and the operation are not one step across Workers)
    /// parks as `blocking_op` would have; on a non-blocking descriptor it
    /// does not, and a vectored call parks on its first non-empty vector.
    #[test]
    fn eagain_on_a_blocking_pipe_parks_instead() {
        let (mut mem, mut k) = setup();
        let p = Arc::new(Pipe::new());
        let end = |write, nb: bool| {
            let f = OpenFile::new(Arc::new(crate::file::PipeEnd { pipe: Arc::clone(&p), write }), "pipe".into(), 0, !write, write);
            if nb {
                f.set_flags(crate::fs::oflags::O_NONBLOCK);
            }
            f
        };
        let rfd = k.fds.insert(end(false, false), false).unwrap() as u64;
        let wfd = k.fds.insert(end(true, false), false).unwrap() as u64;
        let nbr = k.fds.insert(end(false, true), false).unwrap() as u64;
        let iov = 0x2000usize;
        for (i, (b, l)) in [(0x3000u64, 0u64), (0x3100, 100)].into_iter().enumerate() {
            mem[iov + 16 * i..iov + 16 * i + 8].copy_from_slice(&b.to_le_bytes());
            mem[iov + 16 * i + 8..iov + 16 * i + 16].copy_from_slice(&l.to_le_bytes());
        }
        let park = |k: &mut Kernel, mem: &mut Vec<u8>, n, a| k.eagain_park(&GuestMem::new(mem), n, a);
        match park(&mut k, &mut mem, nr::READ, [rfd, 0x4000, 7, 0, 0, 0]) {
            Some(BlockOp::ReadStream { buf: 0x4000, len: 7, .. }) => {}
            other => panic!("read: {other:?}"),
        }
        match park(&mut k, &mut mem, nr::WRITE, [wfd, 0x4000, 9, 0, 0, 0]) {
            Some(BlockOp::WriteStream { buf: 0x4000, len: 9, .. }) => {}
            other => panic!("write: {other:?}"),
        }
        match park(&mut k, &mut mem, nr::WRITEV, [wfd, iov as u64, 2, 0, 0, 0]) {
            Some(BlockOp::WriteStream { buf: 0x3100, len: 100, .. }) => {}
            other => panic!("writev: {other:?}"),
        }
        assert!(park(&mut k, &mut mem, nr::READ, [nbr, 0x4000, 7, 0, 0, 0]).is_none(), "O_NONBLOCK keeps its EAGAIN");
        assert!(park(&mut k, &mut mem, nr::WRITE, [rfd, 0x4000, 7, 0, 0, 0]).is_none(), "not a write end");
    }

    #[test]
    fn clock_nanosleep_honours_absolute_deadlines() {
        let (mut mem, mut k) = setup();
        let at = 0x2000u64;
        let ask = |k: &mut Kernel, mem: &mut Vec<u8>, flags: u64, secs: i64, nanos: u64| {
            mem[at as usize..at as usize + 8].copy_from_slice(&secs.to_le_bytes());
            mem[at as usize + 8..at as usize + 16].copy_from_slice(&nanos.to_le_bytes());
            let m = GuestMem::new(mem);
            let op = k.blocking_op(&m, nr::CLOCK_NANOSLEEP, [1, flags, at, 0, 0, 0]);
            let now = match op {
                None => Some(k.sys_nanosleep(&m, at)),
                Some(_) => None,
            };
            (op, now)
        };
        // The request names clock 1, CLOCK_MONOTONIC.
        let now = clock_nanos_at(k.clock, k.started, 1) as i64 / 1_000_000_000;
        // Relative: slept as given.
        match ask(&mut k, &mut mem, 0, 0, 5_000_000) {
            (Some(BlockOp::Sleep(d)), _) => assert_eq!(d, std::time::Duration::from_millis(5)),
            other => panic!("{other:?}"),
        }
        // Absolute, two seconds ahead: about two seconds, not the whole clock reading.
        match ask(&mut k, &mut mem, 1, now + 2, 0) {
            (Some(BlockOp::Sleep(d)), _) => assert!(d <= std::time::Duration::from_secs(2), "{d:?}"),
            other => panic!("{other:?}"),
        }
        // Longer than a minute: all of it (it was cut to 60 s, answered 0).
        match ask(&mut k, &mut mem, 0, 2200, 0) {
            (Some(BlockOp::Sleep(d)), _) => assert_eq!(d, std::time::Duration::from_secs(2200)),
            other => panic!("{other:?}"),
        }
        // Absolute and already past: returns at once.
        assert!(matches!(ask(&mut k, &mut mem, 1, now - 5, 0), (None, Some(Ok(0)))));
        // Malformed: EINVAL, not a sleep.
        assert!(matches!(ask(&mut k, &mut mem, 0, -1, 0), (None, Some(Err(Errno::EINVAL)))));
        assert!(matches!(ask(&mut k, &mut mem, 0, 0, 1_000_000_000), (None, Some(Err(Errno::EINVAL)))));
    }

    /// The terminal, as a program sees it through `ioctl`.
    ///
    /// These are the calls every interactive program makes before it draws
    /// anything, and the ones a shell makes to run a job. They were three --
    /// `TCGETS`, `TIOCGWINSZ`, `TIOCGPGRP` -- and everything else answered
    /// `ENOTTY`, which is how `stty`, `tcflush` and job control quietly did
    /// nothing.
    mod terminal {
        use super::*;
        use crate::tty;

        const TCGETS: u64 = 0x5401;
        const TCSETS: u64 = 0x5402;
        const TIOCGWINSZ: u64 = 0x5413;
        const TIOCSWINSZ: u64 = 0x5414;
        const TIOCGPGRP: u64 = 0x540f;
        const TIOCSPGRP: u64 = 0x5410;
        const TIOCGSID: u64 = 0x5429;
        const TIOCOUTQ: u64 = 0x5411;
        const TIOCGETD: u64 = 0x5424;
        const TIOCSETD: u64 = 0x5423;
        const TIOCEXCL: u64 = 0x540c;
        const TIOCGEXCL: u64 = 0x8004_5440;
        const TIOCMGET: u64 = 0x5415;
        const TCGETS2: u64 = 0x802c_542a;
        const TIOCGPTN: u64 = 0x8004_5430;
        const AT: u64 = 0x2000;

        /// A kernel whose standard input is a terminal, which is what the
        /// browser and the native host both build for an interactive run.
        ///
        /// Reinstalled rather than asked for: `Kernel::new` copies whether
        /// the *host's* stdin is a terminal, and a test runner's is a pipe.
        fn tty_kernel() -> (Vec<u8>, Kernel) {
            let mem = vec![0u8; 0x40000];
            let mut k = Kernel::new(0x1000, 0x10000, "/prog".into());
            k.vfs.mount("/", Arc::new(TmpFs::new()));
            k.fds = Arc::new(FdTable::new());
            for (stream, name, r, w) in [
                (Stream::In, "/dev/stdin", true, false),
                (Stream::Out, "/dev/stdout", false, true),
                (Stream::Err, "/dev/stderr", false, true),
            ] {
                let h = Arc::new(StdStream::new(stream, true));
                let _ = k.fds.insert(OpenFile::new(h, name.to_string(), 0, r, w), false);
            }
            (mem, k)
        }

        fn ioctl(k: &mut Kernel, mem: &mut Vec<u8>, req: u64, arg: u64) -> Result<u64, Errno> {
            let mut m = GuestMem::new(mem);
            k.sys_ioctl(&mut m, 0, req, arg)
        }

        #[test]
        fn termios_round_trips_through_the_kernel_shape() {
            let (mut mem, mut k) = tty_kernel();
            assert_eq!(ioctl(&mut k, &mut mem, TCGETS, AT), Ok(0));
            // 36 bytes and not one more: glibc's `tcgetattr` passes a
            // `struct __kernel_termios` on the stack, and the 24 bytes past
            // it are the caller's, ending at the stack canary.
            assert_eq!(&mem[AT as usize..AT as usize + 4], &0x6502u32.to_le_bytes());
            assert_eq!(mem[AT as usize + tty::KERNEL_TERMIOS], 0, "wrote past the kernel struct");
            // `intr` is `^C`, which is where a shell reads it from.
            assert_eq!(mem[AT as usize + 17], 3);

            // Raw mode, as a line editor sets it, and read back.
            let mut t = tty::COOKED;
            let l = u32::from_le_bytes([t[12], t[13], t[14], t[15]])
                & !(tty::lflag::ICANON | tty::lflag::ECHO);
            t[12..16].copy_from_slice(&l.to_le_bytes());
            mem[AT as usize..AT as usize + 60].copy_from_slice(&t);
            assert_eq!(ioctl(&mut k, &mut mem, TCSETS, AT), Ok(0));
            assert_eq!(k.tty.termios()[12..16], l.to_le_bytes());
        }

        #[test]
        fn the_window_size_is_read_written_and_shared() {
            let (mut mem, mut k) = tty_kernel();
            let ws = [40u16, 100, 0, 0];
            for (i, v) in ws.iter().enumerate() {
                mem[AT as usize + i * 2..AT as usize + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
            assert_eq!(ioctl(&mut k, &mut mem, TIOCSWINSZ, AT), Ok(0));
            mem[AT as usize..AT as usize + 8].fill(0);
            assert_eq!(ioctl(&mut k, &mut mem, TIOCGWINSZ, AT), Ok(0));
            assert_eq!(u16::from_le_bytes([mem[AT as usize], mem[AT as usize + 1]]), 40);
            assert_eq!(u16::from_le_bytes([mem[AT as usize + 2], mem[AT as usize + 3]]), 100);
        }

        #[test]
        fn the_foreground_group_is_settable_and_is_what_a_shell_reads_back() {
            let (mut mem, mut k) = tty_kernel();
            let mut m = GuestMem::new(&mut mem);
            m.write_u32(AT, 42).unwrap();
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCSPGRP, AT), Ok(0));
            m.write_u32(AT, 0).unwrap();
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCGPGRP, AT), Ok(0));
            assert_eq!(m.read_u32(AT).unwrap(), 42);
            // Zero is not a process group, and accepting it would silently
            // point `^C` at nobody.
            m.write_u32(AT, 0).unwrap();
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCSPGRP, AT), Err(Errno::EINVAL));
        }

        #[test]
        fn the_rest_of_the_surface_answers_rather_than_saying_enotty() {
            let (mut mem, mut k) = tty_kernel();
            // Every one of these was ENOTTY before, and every one of them is
            // asked by something in a Debian rootfs.
            assert_eq!(ioctl(&mut k, &mut mem, TIOCGSID, AT), Ok(0), "TIOCGSID");
            assert_eq!(ioctl(&mut k, &mut mem, TIOCOUTQ, AT), Ok(0), "TIOCOUTQ");
            assert_eq!(ioctl(&mut k, &mut mem, TIOCGETD, AT), Ok(0), "TIOCGETD");
            assert_eq!(ioctl(&mut k, &mut mem, TCGETS2, AT), Ok(0), "TCGETS2");
            assert_eq!(ioctl(&mut k, &mut mem, TIOCMGET, AT), Ok(0), "TIOCMGET");
            assert_eq!(ioctl(&mut k, &mut mem, TIOCEXCL, AT), Ok(0), "TIOCEXCL");
            assert_eq!(ioctl(&mut k, &mut mem, TIOCGEXCL, AT), Ok(0), "TIOCGEXCL");
            let mut m = GuestMem::new(&mut mem);
            assert_eq!(m.read_u32(AT).unwrap(), 1, "TIOCEXCL is read back");
            // A line discipline that is not N_TTY is refused rather than
            // accepted and ignored: a program told yes would then write SLIP
            // frames into a terminal.
            m.write_u32(AT, 1).unwrap();
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCSETD, AT), Err(Errno::EINVAL));
            m.write_u32(AT, 0).unwrap();
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCSETD, AT), Ok(0));
            // And this terminal is a console, not half of a pty pair. Saying
            // so is the honest answer and the one Linux gives.
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCGPTN, AT), Err(Errno::EINVAL));
        }

        #[test]
        fn a_pipe_is_still_not_a_terminal() {
            let (mut mem, mut k) = setup();
            let mut m = GuestMem::new(&mut mem);
            // fd 0 here is a plain stream, not a tty.
            assert_eq!(k.sys_ioctl(&mut m, 0, TCGETS, AT), Err(Errno::ENOTTY));
            assert_eq!(k.sys_ioctl(&mut m, 0, TIOCGWINSZ, AT), Err(Errno::ENOTTY));
        }
    }

    /// Load a system call the way the guest would: `n` is the canonical
    /// number, which an x86-64 guest spells as its own (the highest raw number
    /// `x86::translate` maps to it, so the modern form rather than a legacy
    /// one whose arguments it rewrites; unmapped numbers pass through as-is).
    fn set_args(m: &mut GuestMem<'_>, ctx_addr: u64, n: u64, args: &[u64]) {
        let raw = if gex_abi::X86 {
            (0..512).rev().find(|&r| crate::x86::canonical_number(r) == n).unwrap_or(n)
        } else {
            n
        };
        m.write_u64(ctx_addr + ctx::SYSCALL_NR as u64, raw).unwrap();
        for i in 0..6 {
            let v = args.get(i).copied().unwrap_or(0);
            m.write_u64(ctx_addr + ctx::SYSCALL_ARGS[i] as u64, v).unwrap();
        }
    }

    fn ret(m: &GuestMem<'_>, ctx_addr: u64) -> i64 {
        m.read_u64(ctx_addr + ctx::SYSCALL_RET as u64).unwrap() as i64
    }

    fn put_str(m: &mut GuestMem<'_>, addr: u64, s: &str) {
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        m.write(addr, &b).unwrap();
    }

    /// An epoll descriptor in a `poll` set is readable only while a wait on
    /// it would report something, and never writable. Always readable, GLib
    /// main loops that poll one (kded6's, plasmashell's) went round
    /// `ppoll` and an empty `epoll_wait` at a core each, for good.
    #[test]
    fn an_epoll_descriptor_polls_readable_only_with_something_to_report() {
        const EPOLLIN: u32 = 0x001;
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        let at = k.ctx_addr;
        let call = |k: &mut Kernel, m: &mut GuestMem<'_>, n: u64, a: &[u64]| {
            set_args(m, at, n, a);
            k.syscall(m, 0);
            ret(m, at)
        };
        // `ppoll` of the one descriptor for POLLIN|POLLOUT, not waiting: its revents.
        let poll = |k: &mut Kernel, m: &mut GuestMem<'_>, fd: u64| -> u32 {
            m.write_u64(0x20500, 0).unwrap();
            m.write_u64(0x20508, 0).unwrap();
            m.write_u32(0x20600, fd as u32).unwrap();
            m.write_u32(0x20604, 0x5).unwrap();
            let n = call(k, m, nr::PPOLL, &[0x20600, 1, 0x20500, 0, 8]);
            assert!(n >= 0, "ppoll: {n}");
            m.read_u32(0x20604).unwrap() >> 16
        };
        assert_eq!(call(&mut k, &mut m, nr::PIPE2, &[0x20010, 0]), 0);
        let (rfd, wfd) = (m.read_u32(0x20010).unwrap() as u64, m.read_u32(0x20014).unwrap() as u64);
        let ep = call(&mut k, &mut m, nr::EPOLL_CREATE1, &[0]) as u64;
        assert_eq!(poll(&mut k, &mut m, ep), 0, "an empty set");
        m.write_u32(0x20300, EPOLLIN).unwrap();
        m.write_u64(if gex_abi::X86 { 0x20304 } else { 0x20308 }, 7).unwrap();
        assert_eq!(call(&mut k, &mut m, nr::EPOLL_CTL, &[ep, 1, rfd, 0x20300]), 0);
        assert_eq!(poll(&mut k, &mut m, ep), 0, "an empty pipe");
        // A set watching that set (GLib's loop over a library's own epoll).
        let outer = call(&mut k, &mut m, nr::EPOLL_CREATE1, &[0]) as u64;
        assert_eq!(call(&mut k, &mut m, nr::EPOLL_CTL, &[outer, 1, ep, 0x20300]), 0);
        assert_eq!(poll(&mut k, &mut m, ep), 0, "an empty pipe");
        assert_eq!(poll(&mut k, &mut m, outer), 0, "an empty pipe, two sets deep");
        m.write(0x21000, b"a").unwrap();
        assert_eq!(call(&mut k, &mut m, nr::WRITE, &[wfd, 0x21000, 1]), 1);
        assert_eq!(poll(&mut k, &mut m, ep), 1, "the pipe written: POLLIN only");
        assert_eq!(poll(&mut k, &mut m, outer), 1, "the pipe written, two sets deep");
        assert_eq!(call(&mut k, &mut m, nr::READ, &[rfd, 0x21100, 8]), 1);
        assert_eq!(poll(&mut k, &mut m, ep), 0, "the pipe read again");
        assert_eq!(poll(&mut k, &mut m, outer), 0, "the pipe read again, two sets deep");
        // The watched descriptor closed: its watch reports nothing.
        assert_eq!(call(&mut k, &mut m, nr::CLOSE, &[rfd]), 0);
        assert_eq!(poll(&mut k, &mut m, ep), 0, "the pipe closed");
    }

    /// `EPOLLET` as Linux has it: an edge-triggered watch reports at each
    /// new wakeup of its file, not only when it goes from not ready to
    /// ready -- a pipe written twice without a read reports twice -- and a
    /// file that is always ready reports once. `EPOLLONESHOT` reports once
    /// until `EPOLL_CTL_MOD` arms it again. libmount's mount monitor
    /// watches `/proc/self/mountinfo` with `EPOLLET` and loops until a wait
    /// reports nothing; as level-triggered, it never ended.
    #[test]
    fn epoll_edge_triggered_and_one_shot() {
        const EPOLLIN: u64 = 0x001;
        const ET: u64 = 1 << 31;
        const ONESHOT: u64 = 1 << 30;
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        let at = k.ctx_addr;
        let call = |k: &mut Kernel, m: &mut GuestMem<'_>, n: u64, a: &[u64]| {
            set_args(m, at, n, a);
            k.syscall(m, 0);
            ret(m, at)
        };
        let packed = gex_abi::X86;
        let ctl = |k: &mut Kernel, m: &mut GuestMem<'_>, ep: u64, op: u64, fd: u64, events: u64| {
            m.write_u32(0x20300, events as u32).unwrap();
            m.write_u64(if packed { 0x20304 } else { 0x20308 }, fd + 0x100).unwrap();
            assert_eq!(call(k, m, nr::EPOLL_CTL, &[ep, op, fd, 0x20300]), 0);
        };
        // What one wait that does not block reports: (events, token) each.
        let wait = |k: &mut Kernel, m: &mut GuestMem<'_>, ep: u64| -> Vec<(u32, u64)> {
            let n = call(k, m, nr::EPOLL_PWAIT, &[ep, 0x20400, 8, 0, 0, 8]);
            assert!(n >= 0, "epoll_pwait: {n}");
            let size = if packed { 12 } else { 16 };
            (0..n as u64)
                .map(|i| {
                    let e = 0x20400 + i * size;
                    (m.read_u32(e).unwrap(), m.read_u64(e + if packed { 4 } else { 8 }).unwrap())
                })
                .collect()
        };
        m.write(0x21000, b"ab").unwrap();

        // A pipe written twice without a read: two reports, then none.
        assert_eq!(call(&mut k, &mut m, nr::PIPE2, &[0x20010, 0]), 0);
        let (rfd, wfd) = (m.read_u32(0x20010).unwrap() as u64, m.read_u32(0x20014).unwrap() as u64);
        let ep = call(&mut k, &mut m, nr::EPOLL_CREATE1, &[0]) as u64;
        ctl(&mut k, &mut m, ep, 1, rfd, EPOLLIN | ET);
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "an empty pipe");
        assert_eq!(call(&mut k, &mut m, nr::WRITE, &[wfd, 0x21000, 1]), 1);
        assert_eq!(wait(&mut k, &mut m, ep), vec![(EPOLLIN as u32, rfd + 0x100)], "the first write");
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "nothing new");
        assert_eq!(call(&mut k, &mut m, nr::WRITE, &[wfd, 0x21000, 1]), 1);
        assert_eq!(wait(&mut k, &mut m, ep), vec![(EPOLLIN as u32, rfd + 0x100)], "the second write, unread");
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "nothing new");
        // Level-triggered, the same pipe reports at every wait.
        ctl(&mut k, &mut m, ep, 3, rfd, EPOLLIN);
        for _ in 0..2 {
            assert_eq!(wait(&mut k, &mut m, ep).len(), 1, "level-triggered");
        }
        ctl(&mut k, &mut m, ep, 2, rfd, 0);

        // A file always ready, edge-triggered: reported once, when added.
        let f = open_at(&mut k, &mut m, "/f", 0o100 | 2) as u64;
        ctl(&mut k, &mut m, ep, 1, f, EPOLLIN | ET);
        assert_eq!(wait(&mut k, &mut m, ep), vec![(EPOLLIN as u32, f + 0x100)], "a file, added");
        for _ in 0..3 {
            assert_eq!(wait(&mut k, &mut m, ep), vec![], "a file, again");
        }
        ctl(&mut k, &mut m, ep, 2, f, 0);

        // One-shot: reported once, written again and still not, until
        // `EPOLL_CTL_MOD` arms it.
        ctl(&mut k, &mut m, ep, 1, rfd, EPOLLIN | ONESHOT);
        assert_eq!(wait(&mut k, &mut m, ep).len(), 1, "one-shot, armed");
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "one-shot, spent");
        assert_eq!(call(&mut k, &mut m, nr::WRITE, &[wfd, 0x21000, 1]), 1);
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "one-shot, spent, written");
        ctl(&mut k, &mut m, ep, 3, rfd, EPOLLIN | ONESHOT);
        assert_eq!(wait(&mut k, &mut m, ep).len(), 1, "one-shot, armed again");
        assert_eq!(wait(&mut k, &mut m, ep), vec![], "one-shot, spent again");
    }

    /// `/proc/self/mountinfo` under epoll, as libmount's monitor watches it
    /// (`EPOLLIN | EPOLLPRI | EPOLLET`): readable, reported once when added;
    /// after a mount, reported again with `EPOLLPRI | EPOLLERR`, once; and
    /// a level-triggered `poll` has the change reported to it once too.
    #[test]
    fn mountinfo_reports_a_mount_change() {
        const POLLIN: u32 = 0x001;
        const POLLPRI: u32 = 0x002;
        const POLLERR: u32 = 0x008;
        const ET: u64 = 1 << 31;
        let (mut buf, mut k) = setup();
        for d in ["/proc", "/mnt", "/mnt2"] {
            k.vfs.mkdir(d, 0o755).unwrap();
        }
        let cell = Arc::new(crate::proclive::ProcCell::default());
        let src = crate::proclive::KernelProcSource(Arc::clone(&cell));
        k.vfs.mount("/proc", Arc::new(crate::vfs::ProcFs::new(Arc::new(src))));
        k.publish_proc(&cell);
        let mut m = GuestMem::new(&mut buf);
        let at = k.ctx_addr;
        let call = |k: &mut Kernel, m: &mut GuestMem<'_>, n: u64, a: &[u64]| {
            set_args(m, at, n, a);
            k.syscall(m, 0);
            ret(m, at)
        };
        let packed = gex_abi::X86;
        let fd = open_at(&mut k, &mut m, "/proc/self/mountinfo", 0) as u64;
        let ep = call(&mut k, &mut m, nr::EPOLL_CREATE1, &[0]) as u64;
        m.write_u32(0x20300, (POLLIN | POLLPRI) | ET as u32).unwrap();
        m.write_u64(if packed { 0x20304 } else { 0x20308 }, 0x55).unwrap();
        assert_eq!(call(&mut k, &mut m, nr::EPOLL_CTL, &[ep, 1, fd, 0x20300]), 0);
        let wait = |k: &mut Kernel, m: &mut GuestMem<'_>| -> Option<u32> {
            match call(k, m, nr::EPOLL_PWAIT, &[ep, 0x20400, 4, 0, 0, 8]) {
                0 => None,
                1 => Some(m.read_u32(0x20400).unwrap()),
                n => panic!("epoll_pwait: {n}"),
            }
        };
        assert_eq!(wait(&mut k, &mut m), Some(POLLIN), "added");
        assert_eq!(wait(&mut k, &mut m), None, "no change");
        k.vfs.mount("/mnt", Arc::new(TmpFs::new()));
        assert_eq!(wait(&mut k, &mut m), Some(POLLIN | POLLPRI | POLLERR), "a mount");
        assert_eq!(wait(&mut k, &mut m), None, "the mount, reported");

        // `poll`, level-triggered: the change once, then only readable.
        let pollfd = 0x20500u64;
        m.write_u32(pollfd, fd as u32).unwrap();
        m.write_u32(pollfd + 4, POLLIN | POLLPRI).unwrap();
        let ppoll = |k: &mut Kernel, m: &mut GuestMem<'_>| {
            m.write_u64(0x20600, 0).unwrap();
            m.write_u64(0x20608, 0).unwrap();
            assert_eq!(call(k, m, nr::PPOLL, &[pollfd, 1, 0x20600, 0, 8]), 1);
            m.read_u32(pollfd + 4).unwrap() >> 16
        };
        assert_eq!(ppoll(&mut k, &mut m), POLLIN, "the mount, reported to epoll already");
        k.vfs.mount("/mnt2", Arc::new(TmpFs::new()));
        assert_eq!(ppoll(&mut k, &mut m), POLLIN | POLLPRI | POLLERR, "another mount");
        assert_eq!(ppoll(&mut k, &mut m), POLLIN, "another mount, reported");
        assert!(k.vfs.unmount("/mnt"));
        assert_eq!(wait(&mut k, &mut m), Some(POLLIN | POLLPRI | POLLERR), "an unmount");
    }

    /// Open `path` (`openat` from the working directory) with `flags`.
    fn open_at(k: &mut Kernel, m: &mut GuestMem<'_>, path: &str, flags: u64) -> i64 {
        put_str(m, 0x20000, path);
        set_args(m, k.ctx_addr, nr::OPENAT, &[(-100i64) as u64, 0x20000, flags, 0o644]);
        k.syscall(m, 0);
        ret(m, k.ctx_addr)
    }

    /// `/proc/self/fd/N` opens descriptor N's file, as Linux's magic link
    /// does: a file is the same file with an offset of its own; a pipe the
    /// same pipe; a socket not at all; and `/dev/stderr`, which links there,
    /// the stdio stream itself, whatever name its link reads back.
    #[test]
    fn proc_fd_opens_the_descriptors_file() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        // A file: written through one descriptor, read from the start
        // through the reopened one, whose offset is its own.
        let fd = open_at(&mut k, &mut m, "/f.txt", oflags::O_CREAT | oflags::O_RDWR);
        assert!(fd >= 0);
        m.write(0x21000, b"payload").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[fd as u64, 0x21000, 7]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);
        let again = open_at(&mut k, &mut m, &format!("/proc/self/fd/{fd}"), oflags::O_RDONLY);
        assert!(again >= 0 && again != fd, "{again}");
        set_args(&mut m, k.ctx_addr, nr::READ, &[again as u64, 0x22000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);
        assert_eq!(m.read(0x22000, 7).unwrap(), b"payload");
        // More access than the descriptor has: opened again by name.
        let ro = open_at(&mut k, &mut m, "/f.txt", oflags::O_RDONLY);
        let own = format!("/proc/{}/fd/{ro}", k.pid);
        let rw = open_at(&mut k, &mut m, &own, oflags::O_WRONLY | oflags::O_APPEND);
        assert!(rw >= 0, "{rw}");
        m.write(0x21000, b"!").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[rw as u64, 0x21000, 1]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 1);

        // A pipe: what goes into the reopened write end comes out of the
        // read end.
        set_args(&mut m, k.ctx_addr, nr::PIPE2, &[0x23000, 0]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);
        let (r, w) = (m.read_u32(0x23000).unwrap() as i32, m.read_u32(0x23004).unwrap() as i32);
        let w2 = open_at(&mut k, &mut m, &format!("/dev/fd/{w}"), oflags::O_WRONLY);
        assert!(w2 >= 0, "{w2}");
        set_args(&mut m, k.ctx_addr, nr::CLOSE, &[w as u64]);
        k.syscall(&mut m, 0);
        m.write(0x21000, b"via").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[w2 as u64, 0x21000, 3]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 3);
        set_args(&mut m, k.ctx_addr, nr::READ, &[r as u64, 0x22000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 3);
        assert_eq!(m.read(0x22000, 3).unwrap(), b"via");
        // The other direction of a pipe end: not here.
        assert_eq!(open_at(&mut k, &mut m, &format!("/proc/self/fd/{r}"), oflags::O_WRONLY), -6);

        // A socket: ENXIO, as Linux.
        set_args(&mut m, k.ctx_addr, nr::SOCKETPAIR, &[1, 1, 0, 0x24000]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);
        let sock = m.read_u32(0x24000).unwrap();
        assert_eq!(open_at(&mut k, &mut m, &format!("/proc/self/fd/{sock}"), oflags::O_RDWR), -6);

        // stdio, whose link reads back as `/dev/stderr`: the stream itself.
        let err = OpenFile::new(Arc::new(crate::stdio::StdStream::new(crate::stdio::Stream::Err, false)), "/dev/stderr".into(), 0, false, true);
        k.fds.install(2, err, false).unwrap();
        let e2 = open_at(&mut k, &mut m, "/dev/stderr", oflags::O_WRONLY | oflags::O_APPEND);
        assert!(e2 >= 0, "{e2}");
        assert!(Arc::ptr_eq(&k.fds.get(e2 as i32).unwrap().handle, &k.fds.get(2).unwrap().handle));
        // No such descriptor.
        assert_eq!(open_at(&mut k, &mut m, "/proc/self/fd/99", oflags::O_RDONLY), -2);
    }

    #[test]
    fn openat_write_read_roundtrip() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/hello.txt");

        set_args(
            &mut m,
            k.ctx_addr,
            nr::OPENAT,
            &[
                (-100i64) as u64,
                0x20000,
                oflags::O_CREAT | oflags::O_RDWR,
                0o644,
            ],
        );
        k.syscall(&mut m, 0);
        let fd = ret(&m, k.ctx_addr);
        assert!(fd >= 3, "expected a fresh descriptor, got {fd}");

        m.write(0x21000, b"payload").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[fd as u64, 0x21000, 7]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);

        set_args(&mut m, k.ctx_addr, nr::LSEEK, &[fd as u64, 0, 0]);
        k.syscall(&mut m, 0);
        set_args(&mut m, k.ctx_addr, nr::READ, &[fd as u64, 0x22000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);
        assert_eq!(m.read(0x22000, 7).unwrap(), b"payload");
    }

    /// A regular file gives a read every byte asked for, past 16 MiB too
    /// (Wine reads a PE section with one pread and trusts the count).
    #[test]
    fn a_large_pread_of_a_file_is_not_short() {
        let mut buf = vec![0u8; 0x40000 + (40 << 20)];
        let mut k = Kernel::new(0x1000, 0x10000, "/prog".into());
        k.vfs.mount("/", Arc::new(TmpFs::new()));
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/big");
        set_args(&mut m, k.ctx_addr, nr::OPENAT, &[(-100i64) as u64, 0x20000, oflags::O_CREAT | oflags::O_RDWR, 0o644]);
        k.syscall(&mut m, 0);
        let fd = ret(&m, k.ctx_addr) as u64;
        let n: u64 = 20 << 20;
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8 + 1).collect();
        m.write(0x40000, &data).unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[fd, 0x40000, n]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), n as i64);
        let dst = 0x40000 + (20 << 20);
        set_args(&mut m, k.ctx_addr, nr::PREAD64, &[fd, dst, n, 0]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), n as i64);
        assert!(m.read(dst, n as usize).unwrap() == &data[..]);
    }

    #[test]
    fn dup_shares_the_file_offset() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/f");
        set_args(
            &mut m,
            k.ctx_addr,
            nr::OPENAT,
            &[
                (-100i64) as u64,
                0x20000,
                oflags::O_CREAT | oflags::O_RDWR,
                0,
            ],
        );
        k.syscall(&mut m, 0);
        let fd = ret(&m, k.ctx_addr) as u64;
        set_args(&mut m, k.ctx_addr, nr::DUP, &[fd]);
        k.syscall(&mut m, 0);
        let dup = ret(&m, k.ctx_addr) as u64;

        m.write(0x21000, b"abcd").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[fd, 0x21000, 4]);
        k.syscall(&mut m, 0);
        // The duplicate must see the advanced offset, not zero.
        set_args(&mut m, k.ctx_addr, nr::LSEEK, &[dup, 0, 1]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 4);
    }

    /// A unix socket's name is its file's: unlinked, the path binds again
    /// (the old socket open or closed, its connections still carrying);
    /// renamed, the name moves; while the file is there a bind is
    /// EADDRINUSE and leaves the socket free to bind later.
    #[test]
    fn a_socket_files_unlink_or_rename_takes_its_name() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        const AF_UNIX: u64 = 1;
        const SOCK_STREAM: u64 = 1;
        const AT_FDCWD: u64 = -100i64 as u64;
        let mut call = |m: &mut GuestMem<'_>, n: u64, a: &[u64]| {
            set_args(m, at, n, a);
            k.syscall(m, 0);
            ret(m, at)
        };
        // sockaddr_un for `name` at `addr`, its length; the bare path after it.
        let addr = |m: &mut GuestMem<'_>, addr: u64, name: &str| {
            let mut sa = vec![1u8, 0];
            sa.extend_from_slice(name.as_bytes());
            sa.push(0);
            m.write(addr, &sa).unwrap();
            m.write(addr + 0x80, name.as_bytes()).unwrap();
            m.write(addr + 0x80 + name.len() as u64, &[0]).unwrap();
            sa.len() as u64
        };
        let la = addr(&mut m, 0x20000, "/srv-sock");
        let sock = |m: &mut GuestMem<'_>, call: &mut dyn FnMut(&mut GuestMem<'_>, u64, &[u64]) -> i64| call(m, nr::SOCKET, &[AF_UNIX, SOCK_STREAM, 0]) as u64;
        let l = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l, 0x20000, la]), 0);
        assert_eq!(call(&mut m, nr::LISTEN, &[l, 4]), 0);
        // A connection made before the unlink.
        let c = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::CONNECT, &[c, 0x20000, la]), 0);
        let s = call(&mut m, nr::ACCEPT4, &[l, 0, 0, 0]);
        assert!(s > 0, "accept: {s}");
        // The name is taken while the file is there, the listener open or not.
        let l2 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l2, 0x20000, la]), -98, "bind over a bound name");
        // Unlinked: bound again, by a new socket, with the old one still open.
        assert_eq!(call(&mut m, nr::UNLINKAT, &[AT_FDCWD, 0x20080, 0]), 0);
        assert_eq!(call(&mut m, nr::BIND, &[l2, 0x20000, la]), 0, "bind after the unlink");
        assert_eq!(call(&mut m, nr::LISTEN, &[l2, 4]), 0);
        // The old connection still carries bytes; a new one reaches the new
        // listener.
        m.write(0x21000, b"old").unwrap();
        assert_eq!(call(&mut m, nr::SENDTO, &[c, 0x21000, 3, 0, 0, 0]), 3);
        assert_eq!(call(&mut m, nr::RECVFROM, &[s as u64, 0x21100, 16, 0, 0, 0]), 3);
        let c2 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::CONNECT, &[c2, 0x20000, la]), 0);
        assert!(call(&mut m, nr::ACCEPT4, &[l2, 0, 0, oflags::O_NONBLOCK]) > 0, "the new listener's connection");
        // Closed, then unlinked: free again.
        assert_eq!(call(&mut m, nr::CLOSE, &[l2]), 0);
        let l3 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l3, 0x20000, la]), -98, "bind over a closed socket's file");
        assert_eq!(call(&mut m, nr::UNLINKAT, &[AT_FDCWD, 0x20080, 0]), 0);
        assert_eq!(call(&mut m, nr::BIND, &[l3, 0x20000, la]), 0, "bind after closing and unlinking");
        assert_eq!(call(&mut m, nr::LISTEN, &[l3, 4]), 0);
        // Renamed: the name moves with the file; the old path is free.
        let ra = addr(&mut m, 0x22000, "/srv-sock2");
        assert_eq!(call(&mut m, nr::RENAMEAT2, &[AT_FDCWD, 0x20080, AT_FDCWD, 0x22080, 0]), 0);
        let c3 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::CONNECT, &[c3, 0x22000, ra]), 0, "connect to the new name");
        let c4 = sock(&mut m, &mut call);
        assert_ne!(call(&mut m, nr::CONNECT, &[c4, 0x20000, la]), 0, "connect to the old name");
        let l4 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l4, 0x20000, la]), 0, "bind the old name after the rename");
        // Renamed over: the file replaced takes its name with it.
        assert_eq!(call(&mut m, nr::RENAMEAT2, &[AT_FDCWD, 0x20080, AT_FDCWD, 0x22080, 0]), 0);
        let c5 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::LISTEN, &[l4, 4]), 0);
        assert_eq!(call(&mut m, nr::CONNECT, &[c5, 0x22000, ra]), 0, "the renamed-over name is the new file's");
        assert!(call(&mut m, nr::ACCEPT4, &[l4, 0, 0, oflags::O_NONBLOCK]) > 0, "reached the socket renamed over it");
        // A directory renamed: the socket under it is bound at its new path,
        // and its old path is free.
        addr(&mut m, 0x23000, "/sockdir");
        assert_eq!(call(&mut m, nr::MKDIRAT, &[AT_FDCWD, 0x23080, 0o755]), 0);
        let ia = addr(&mut m, 0x24000, "/sockdir/s");
        let l5 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l5, 0x24000, ia]), 0);
        assert_eq!(call(&mut m, nr::LISTEN, &[l5, 4]), 0);
        addr(&mut m, 0x25000, "/sockdir2");
        assert_eq!(call(&mut m, nr::RENAMEAT2, &[AT_FDCWD, 0x23080, AT_FDCWD, 0x25080, 0]), 0);
        let na = addr(&mut m, 0x26000, "/sockdir2/s");
        let c6 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::CONNECT, &[c6, 0x26000, na]), 0, "connect under the renamed directory");
        assert!(call(&mut m, nr::ACCEPT4, &[l5, 0, 0, oflags::O_NONBLOCK]) > 0, "reached the socket under the renamed directory");
        assert_eq!(call(&mut m, nr::MKDIRAT, &[AT_FDCWD, 0x23080, 0o755]), 0);
        let l6 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l6, 0x24000, ia]), 0, "bind the old path under the directory's old name");
        // A name that only shares a prefix stays where it is.
        addr(&mut m, 0x27000, "/sockdir2x");
        assert_eq!(call(&mut m, nr::MKDIRAT, &[AT_FDCWD, 0x27080, 0o755]), 0);
        let xa = addr(&mut m, 0x28000, "/sockdir2x/s");
        let l7 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::BIND, &[l7, 0x28000, xa]), 0);
        assert_eq!(call(&mut m, nr::LISTEN, &[l7, 4]), 0);
        addr(&mut m, 0x29000, "/sockdir3");
        assert_eq!(call(&mut m, nr::RENAMEAT2, &[AT_FDCWD, 0x25080, AT_FDCWD, 0x29080, 0]), 0);
        let c7 = sock(&mut m, &mut call);
        assert_eq!(call(&mut m, nr::CONNECT, &[c7, 0x28000, xa]), 0, "a prefix-sharing sibling keeps its name");
    }

    /// `free_call`, the lock-free class, answers as the full call does:
    /// the same results and bytes, the same park on an empty pipe, the same
    /// mask, the same `SIGPIPE`; and it declines every other call by number.
    /// The calls on a process's and thread's own signal state, and
    /// `getrandom`, made lock-free: what one path sets the other
    /// reads back.
    /// The small descriptor and process calls, lock-free and in the
    /// kernel, give the same answers and change the same state.
    /// The socket calls answer on the free path as the kernel does:
    /// a pair carries bytes, a named listener takes a connection and hands
    /// it out, the names and options are the same, a blocking `accept` with
    /// nothing waiting parks, and the errors are the kernel's.
    #[test]
    fn free_socket_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        const AF_UNIX: u64 = 1;
        const SOCK_STREAM: u64 = 1;
        for lock_free in [true, false] {
            let call = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(done(free(k, m, at)), "lock_free {lock_free}");
                } else {
                    k.syscall(m, 0);
                }
            };
            // A pair: bytes one way, then the other end's name is AF_UNIX.
            set_args(&mut m, at, nr::SOCKETPAIR, &[AF_UNIX, SOCK_STREAM, 0, 0x20000]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0, "lock_free {lock_free}");
            let (p0, p1) = (m.read_u32(0x20000).unwrap() as u64, m.read_u32(0x20004).unwrap() as u64);
            m.write(0x20100, b"pair").unwrap();
            set_args(&mut m, at, nr::SENDTO, &[p0, 0x20100, 4, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 4);
            set_args(&mut m, at, nr::RECVFROM, &[p1, 0x20200, 16, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 4);
            assert_eq!(m.read(0x20200, 4).unwrap(), b"pair");
            m.write_u32(0x20300, 16).unwrap();
            set_args(&mut m, at, nr::GETSOCKNAME, &[p1, 0x20310, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!((ret(&m, at), m.read_u32(0x20300).unwrap(), m.read(0x20310, 2).unwrap()), (0, 2, &[1u8, 0][..]));

            // A named listener: bind, listen, connect, accept.
            let name = if lock_free { "/sock-a" } else { "/sock-b" };
            let mut sa = vec![1u8, 0];
            sa.extend_from_slice(name.as_bytes());
            sa.push(0);
            m.write(0x20400, &sa).unwrap();
            set_args(&mut m, at, nr::SOCKET, &[AF_UNIX, SOCK_STREAM, 0]);
            call(&mut m, &mut k);
            let l = ret(&m, at) as u64;
            set_args(&mut m, at, nr::BIND, &[l, 0x20400, sa.len() as u64]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0, "bind, lock_free {lock_free}");
            set_args(&mut m, at, nr::LISTEN, &[l, 4]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            if lock_free {
                // Nothing waiting on a blocking listener: it parks.
                set_args(&mut m, at, nr::ACCEPT4, &[l, 0, 0, 0]);
                assert!(matches!(free(&k, &mut m, at), Some(FreeOutcome::Blocking(BlockOp::AcceptWait { .. }, None))));
            }
            set_args(&mut m, at, nr::SOCKET, &[AF_UNIX, SOCK_STREAM, 0]);
            call(&mut m, &mut k);
            let c = ret(&m, at) as u64;
            set_args(&mut m, at, nr::CONNECT, &[c, 0x20400, sa.len() as u64]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0, "connect, lock_free {lock_free}");
            set_args(&mut m, at, nr::ACCEPT4, &[l, 0, 0, oflags::O_CLOEXEC]);
            call(&mut m, &mut k);
            let s = ret(&m, at);
            assert!(s > 0, "accept, lock_free {lock_free}: {s}");
            assert!(k.fds.cloexec(s as i32).unwrap());
            // A second client, accepted into a `struct sockaddr` (16 bytes):
            // its peer never bound, so the address is the family alone, and
            // nothing past the 16 bytes is written.
            set_args(&mut m, at, nr::SOCKET, &[AF_UNIX, SOCK_STREAM, 0]);
            call(&mut m, &mut k);
            let c2 = ret(&m, at) as u64;
            set_args(&mut m, at, nr::CONNECT, &[c2, 0x20400, sa.len() as u64]);
            call(&mut m, &mut k);
            m.write(0x20600, &[0xee; 32]).unwrap();
            m.write_u32(0x205f0, 16).unwrap();
            set_args(&mut m, at, nr::ACCEPT4, &[l, 0x20600, 0x205f0, 0]);
            call(&mut m, &mut k);
            assert!(ret(&m, at) > 0, "second accept, lock_free {lock_free}");
            assert_eq!((m.read_u32(0x205f0).unwrap(), m.read(0x20600, 2).unwrap()), (2, &[1u8, 0][..]));
            assert!(m.read(0x20610, 16).unwrap().iter().all(|&b| b == 0xee), "accept wrote past the caller's 16 bytes");
            // The options: a stream, no error.
            set_args(&mut m, at, nr::GETSOCKOPT, &[c, 1, 3, 0x20500, 0]);
            call(&mut m, &mut k);
            assert_eq!((ret(&m, at), m.read_u32(0x20500).unwrap()), (0, 1));
            set_args(&mut m, at, nr::GETSOCKOPT, &[c, 1, 4, 0x20500, 0]);
            call(&mut m, &mut k);
            assert_eq!((ret(&m, at), m.read_u32(0x20500).unwrap()), (0, 0));
            // No peer pidfd and no security label, as on a kernel without
            // them: an error, not descriptor 0 or an empty label.
            for opt in [31u64, 77] {
                m.write_u32(0x205f0, 4).unwrap();
                set_args(&mut m, at, nr::GETSOCKOPT, &[s as u64, 1, opt, 0x20500, 0x205f0]);
                call(&mut m, &mut k);
                assert_eq!(ret(&m, at), -92, "getsockopt option {opt}");
            }
            // A buffer size reads back as Linux's default, its length
            // written; an option this kernel does not have is refused.
            m.write_u32(0x205f0, 4).unwrap();
            set_args(&mut m, at, nr::GETSOCKOPT, &[c, 1, 8, 0x20500, 0x205f0]);
            call(&mut m, &mut k);
            assert_eq!((ret(&m, at), m.read_u32(0x20500).unwrap(), m.read_u32(0x205f0).unwrap()), (0, 212_992, 4));
            set_args(&mut m, at, nr::GETSOCKOPT, &[c, 1, 1000, 0x20500, 0x205f0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -92);
            set_args(&mut m, at, nr::SETSOCKOPT, &[c, 1, 2, 0x20500, 4]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            // Shut for writing: the accepted end reads its end.
            set_args(&mut m, at, nr::SHUTDOWN, &[c, 1]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            set_args(&mut m, at, nr::RECVFROM, &[s as u64, 0x20200, 16, 0x40, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0, "EOF after shutdown, lock_free {lock_free}");

            // The errors.
            set_args(&mut m, at, nr::SOCKET, &[10, SOCK_STREAM, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EAFNOSUPPORT.0 as i64));
            // A socketpair's peer is unnamed: the family alone, as Linux has it.
            m.write_u32(0x20300, 110).unwrap();
            set_args(&mut m, at, nr::GETPEERNAME, &[p0, 0x20310, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!((ret(&m, at), m.read_u32(0x20300).unwrap()), (0, 2));
            set_args(&mut m, at, nr::GETPEERNAME, &[0x20000, 0x20310, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EBADF.0 as i64));
            set_args(&mut m, at, nr::SETSOCKOPT, &[999, 1, 2, 0x20500, 4]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EBADF.0 as i64));
            for f in [p0, p1, l, c, s as u64] {
                set_args(&mut m, at, nr::CLOSE, &[f]);
                k.syscall(&mut m, 0);
            }
        }
    }

    /// `mlock` and its kin succeed and change nothing: KSharedDataCache
    /// checks a file-backed cache's mapping with `mlock` and gives the file
    /// up when it fails.
    #[test]
    fn locking_memory_succeeds() {
        for n in [nr::MLOCK, nr::MUNLOCK, nr::MLOCKALL, nr::MUNLOCKALL, nr::MLOCK2] {
            let (mut buf, mut k) = setup();
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let first = if n == nr::MLOCKALL { 1 } else { 0x10000 };
            set_args(&mut m, at, n, &[first, 4096, 0, 0, 0, 0]);
            assert!(is_quiet(n, false), "{n}");
            assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
            assert_eq!(ret(&m, at), 0, "{n}");
        }
        // flags it does not know
        for (n, args) in [(nr::MLOCKALL, [0u64, 0, 0, 0, 0, 0]), (nr::MLOCKALL, [8, 0, 0, 0, 0, 0]), (nr::MLOCK2, [0x10000, 4096, 2, 0, 0, 0])] {
            let (mut buf, mut k) = setup();
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            set_args(&mut m, at, n, &args);
            assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
            assert_eq!(ret(&m, at), -(Errno::EINVAL.0 as i64), "{n} {args:?}");
        }
    }

    /// What a page still makes under the kernel lock: every call but `execve`
    /// and a process's `clone` is in one of
    /// the free path's classes, and one it answers `ENOSYS` the kernel does
    /// not make either. Each canonical number, with zero arguments.
    #[test]
    fn only_the_process_calls_take_the_locked_path() {
        let (mut locked, mut wrong) = (Vec::new(), Vec::new());
        for n in (0..=470u64).chain(nr::DUP2..=nr::NOOP_OK) {
            // The ABI's own spelling of `n`; a number it has none for is
            // not one of its calls.
            let spelled = if gex_abi::X86 {
                (0..512).any(|r| crate::x86::canonical_number(r) == n)
            } else {
                true
            };
            if !spelled {
                continue;
            }
            let (mut buf, mut k) = setup();
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            set_args(&mut m, at, n, &[0; 6]);
            match free_class(&m, at, false) {
                None => locked.push(n),
                // Answered ENOSYS without the kernel: the kernel must say
                // the same.
                Some(c) if c.number() == NOSYS => {
                    let out = k.syscall(&mut m, 0);
                    if !(matches!(out, SyscallOutcome::Continue) && ret(&m, at) == -(Errno::ENOSYS.0 as i64)) {
                        wrong.push(n);
                    }
                }
                Some(_) => {}
            }
        }
        let names = |v: &[u64]| v.iter().map(|&n| syscall_name(n)).collect::<Vec<_>>();
        assert_eq!(wrong, Vec::<u64>::new(), "answered ENOSYS, but the kernel makes {:?}", names(&wrong));
        assert_eq!(locked, vec![nr::CLONE, nr::EXECVE, nr::CLONE3], "{:?}", names(&locked));
    }

    /// i386's `socketcall`, as Debian's i386 glibc makes every socket call,
    /// on the lock-free path: classed as the call it makes, which is made.
    #[test]
    fn an_i386_socketcall_is_made_without_the_lock() {
        if !gex_abi::X86 {
            return;
        }
        let (mut buf, mut k) = setup();
        k.compat = true;
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        // socketcall(SYS_SOCKETPAIR, {AF_UNIX, SOCK_STREAM, 0, sv}).
        let (args_at, sv_at) = (0x31000u64, 0x31100u64);
        for (i, w) in [1u32, 1, 0, sv_at as u32].iter().enumerate() {
            m.write_u32(args_at + 4 * i as u64, *w).unwrap();
        }
        m.write_u64(sv_at, !0).unwrap();
        m.write_u64(at + ctx::x(0) as u64, crate::i386::raw::SOCKETCALL).unwrap();
        for (slot, v) in crate::i386::ARG_SLOTS.into_iter().zip([8, args_at, 0, 0, 0, 0]) {
            m.write_u64(at + slot as u64, v).unwrap();
        }
        let call = free_class(&m, at, true).expect("in the class");
        assert_eq!(call.number(), nr::SOCKETPAIR);
        assert_eq!(call.args(), [1, 1, 0]);
        let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
        let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
        assert!(matches!(free_call(&f, &mut m, at, call), FreeOutcome::Done(_)));
        assert_eq!(m.read_u64(at + ctx::x(0) as u64).unwrap() as u32, 0);
        let (a, b) = (m.read_u32(sv_at).unwrap(), m.read_u32(sv_at + 4).unwrap());
        assert!(a < 1024 && b < 1024 && a != b, "socketpair gave {a} and {b}");
        // An unknown call is ENOSYS, as the full call answers it.
        m.write_u64(at + crate::i386::ARG_SLOTS[0] as u64, 99).unwrap();
        assert_eq!(free_class(&m, at, true).map(|c| c.number()), Some(NOSYS));
    }

    /// [`only_the_process_calls_take_the_locked_path`] for an i386 process
    /// (32-bit Wine), in its own numbering: each raw number with zero
    /// arguments.
    #[test]
    fn only_the_i386_process_calls_take_the_locked_path() {
        if !gex_abi::X86 {
            return;
        }
        let (mut locked, mut wrong) = (Vec::new(), Vec::new());
        for raw in 0..=460u64 {
            let (mut buf, mut k) = setup();
            k.compat = true;
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            m.write_u64(at + ctx::x(0) as u64, raw).unwrap();
            for slot in crate::i386::ARG_SLOTS {
                m.write_u64(at + slot as u64, 0).unwrap();
            }
            match free_class(&m, at, true) {
                None => locked.push(raw),
                Some(c) if c.number() == NOSYS => {
                    let out = k.syscall(&mut m, 0);
                    let r = m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32;
                    if !(matches!(out, SyscallOutcome::Continue) && r == -(Errno::ENOSYS.0)) {
                        wrong.push(raw);
                    }
                }
                Some(_) => {}
            }
        }
        use crate::i386::raw as x;
        assert_eq!(wrong, Vec::<u64>::new(), "answered ENOSYS, but the kernel makes these i386 calls");
        assert_eq!(locked, vec![x::FORK, x::EXECVE, x::CLONE, x::VFORK, x::CLONE3], "{locked:?}");
    }


    #[test]
    fn free_quiet_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        let buf_at = 0x30000u64;
        let mut cases: Vec<(u64, [u64; 3])> = vec![
            (nr::UNAME, [buf_at, 0, 0]),
            (nr::PRCTL, [15, buf_at, 0]),
            (nr::SYNC, [0, 0, 0]),
            (nr::FSYNC, [1, 0, 0]),
            (nr::FSYNC, [999, 0, 0]),
            (nr::FDATASYNC, [2, 0, 0]),
            (nr::SET_ROBUST_LIST, [buf_at, 24, 0]),
            (nr::GET_ROBUST_LIST, [0, buf_at, buf_at + 8]),
            (nr::RSEQ, [buf_at, 32, 0]),
            (nr::SCHED_GETAFFINITY, [0, 128, buf_at]),
            (nr::SCHED_GETAFFINITY, [0, 4, buf_at]),
            (nr::SCHED_GETSCHEDULER, [0, 0, 0]),
            (nr::SCHED_GETPARAM, [0, buf_at, 0]),
            (nr::SCHED_SETPARAM, [0, buf_at, 0]),
            (nr::SCHED_SETSCHEDULER, [0, 1, buf_at]),
            (nr::SCHED_SETSCHEDULER, [0, 3, buf_at]),
            (nr::SCHED_SETSCHEDULER, [0, 9, buf_at]),
            (nr::SCHED_GET_PRIORITY_MAX, [1, 0, 0]),
            (nr::SCHED_GET_PRIORITY_MIN, [0, 0, 0]),
            (nr::GETCPU, [buf_at, buf_at + 4, 0]),
            (nr::MEMBARRIER, [0, 0, 0]),
            (nr::GETGROUPS, [0, 0, 0]),
            (nr::SETGROUPS, [0, 0, 0]),
            (nr::GETPPID, [0, 0, 0]),
        ];
        if gex_abi::X86 {
            // arch_prctl: set the fs base, read it back.
            cases.push((nr::ARCH_PRCTL, [0x1002, 0x1234_5000, 0]));
            cases.push((nr::ARCH_PRCTL, [0x1003, buf_at, 0]));
            cases.push((nr::ARCH_PRCTL, [0x9999, buf_at, 0]));
        }
        for (n, args) in cases {
            m.zero(buf_at, 256).unwrap();
            m.write_u64(buf_at + 200, 0x5a5a).unwrap();
            set_args(&mut m, at, n, &args);
            k.syscall(&mut m, 0);
            let (want, want_ret) = (m.read(buf_at, 256).unwrap().to_vec(), ret(&m, at));
            let fs = m.read_u64(at + ctx::FS_BASE as u64).unwrap();
            m.zero(buf_at, 256).unwrap();
            m.write_u64(buf_at + 200, 0x5a5a).unwrap();
            set_args(&mut m, at, n, &args);
            assert!(done(free(&k, &mut m, at)), "call {n} in the class");
            assert_eq!(m.read(buf_at, 256).unwrap(), &want[..], "call {n} {args:?}");
            assert_eq!(ret(&m, at), want_ret, "call {n} {args:?}");
            assert_eq!(m.read_u64(at + ctx::FS_BASE as u64).unwrap(), fs, "call {n}");
        }
        // A process call: declined, for the kernel.
        set_args(&mut m, at, nr::EXECVE, &[0, 0, 0]);
        assert!(free_class(&m, at, false).is_none());
    }

    /// The credential calls answer and change the ids as the
    /// kernel's arms do, the change is what the rest of the process reads
    /// (a stat's owner), and a fork's ids are its own.
    #[test]
    fn free_cred_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        let buf_at = 0x30000u64;
        let keep = u32::MAX as u64;
        let cases: Vec<(u64, [u64; 3])> = vec![
            (nr::GETUID, [0, 0, 0]),
            (nr::GETEUID, [0, 0, 0]),
            (nr::GETGID, [0, 0, 0]),
            (nr::GETEGID, [0, 0, 0]),
            (nr::SETUID, [0, 0, 0]),
            (nr::SETUID, [keep, 0, 0]),
            (nr::SETGID, [5, 0, 0]),
            (nr::SETREUID, [keep, 7, 0]),
            (nr::SETREGID, [keep, 8, 0]),
            (nr::SETRESUID, [keep, 9, keep]),
            (nr::SETRESUID, [keep, keep, keep]),
            (nr::SETRESGID, [keep, 10, keep]),
            (nr::GETRESUID, [buf_at, buf_at + 4, buf_at + 8]),
            (nr::GETRESGID, [buf_at, 0, buf_at + 8]),
            (nr::GETRESUID, [1, 0, 0]),
            (nr::GETSID, [0, 0, 0]),
            (nr::GETPGID, [0, 0, 0]),
            (nr::GETPGID, [99999, 0, 0]),
            (nr::SETPGID, [0, 0, 0]),
            (nr::SETPGID, [99999, 0, 0]),
        ];
        for (n, args) in cases {
            let run = |m: &mut GuestMem<'_>, k: &mut Kernel, kernel: bool| {
                k.creds.set_uid(1000);
                k.creds.set_gid(1000);
                m.zero(buf_at, 64).unwrap();
                set_args(m, at, n, &args);
                if kernel {
                    k.syscall(m, 0);
                } else {
                    assert!(done(free(k, m, at)), "call {n} in the class");
                }
                (ret(m, at), m.read(buf_at, 64).unwrap().to_vec(), k.creds.uid(), k.creds.gid())
            };
            let want = run(&mut m, &mut k, true);
            assert_eq!(run(&mut m, &mut k, false), want, "call {n} {args:?}");
        }
        // setsid: the same answer both ways (EPERM here, for a process
        // that already leads its group), and the session the kernel's
        // getsid reports unchanged.
        set_args(&mut m, at, nr::GETSID, &[0]);
        k.syscall(&mut m, 0);
        let sid = ret(&m, at);
        set_args(&mut m, at, nr::SETSID, &[]);
        assert!(done(free(&k, &mut m, at)));
        let made = ret(&m, at);
        set_args(&mut m, at, nr::SETSID, &[]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), made);
        set_args(&mut m, at, nr::GETSID, &[0]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), if made < 0 { sid } else { made });
        // A uid set without the kernel is the owner the kernel's stat reports.
        set_args(&mut m, at, nr::SETUID, &[4321]);
        assert!(done(free(&k, &mut m, at)));
        set_args(&mut m, at, nr::FSTAT, &[1, buf_at]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 0);
        let uid_at = if gex_abi::X86 { 28 } else { 24 };
        assert_eq!(m.read_u32(buf_at + uid_at).unwrap(), 4321);
        // A fork's ids are its own from then on.
        let child = k.fork_kernel();
        assert_eq!(child.creds.uid(), 4321);
        child.creds.set_uid(5);
        assert_eq!(k.creds.uid(), 4321);
    }

    /// The real, effective and saved ids each call sets, as Linux sets
    /// them for root, and the capability sets that follow (`Creds`).
    #[test]
    fn set_ids_and_capabilities_as_linux_for_root() {
        let all = CAPS_ALL;
        let c = Creds::new(0, 0);
        assert_eq!(c.caps(), [all, all, 0]);
        let run = |c: &Creds, n: u64, a: [u64; 3]| {
            let mut buf = [0u8; 64];
            let mut m = GuestMem::new(&mut buf);
            cred_call(c, &ProcTable::new(), 1, &mut m, n, [a[0], a[1], a[2], 0, 0, 0]).unwrap();
        };
        let keep = u32::MAX as u64;
        run(&c, nr::SETRESUID, [keep, 997, keep]);
        assert_eq!(c.uids(), [0, 997, 0]);
        assert_eq!(c.caps(), [0, all, 0]);
        run(&c, nr::SETRESUID, [keep, 0, keep]);
        assert_eq!(c.caps(), [all, all, 0]);
        run(&c, nr::SETREUID, [keep, 998, 0]);
        assert_eq!(c.uids(), [0, 998, 998]);
        // The real id alone: no root id left, so no capabilities.
        run(&c, nr::SETREUID, [995, keep, 0]);
        assert_eq!(c.uids(), [995, 998, 998]);
        assert_eq!(c.caps(), [0, 0, 0]);
        // From root, the real id alone: the effective root keeps them.
        let d = Creds::new(0, 0);
        run(&d, nr::SETREUID, [995, keep, 0]);
        assert_eq!((d.uids(), d.caps()), ([995, 0, 0], [all, all, 0]));
        run(&c, nr::SETUID, [996, 0, 0]);
        assert_eq!(c.uids(), [996, 996, 996]);
        assert_eq!(c.caps(), [0, 0, 0]);
        // A fork's copy, and an exec's sets.
        let f = c.fork();
        assert_eq!((f.uids(), f.caps()), ([996; 3], [0; 3]));
        f.exec();
        assert_eq!(f.caps(), [0; 3]);
        // Granted, as root's would be, with root's sets.
        run(&f, nr::SETUID, [0, 0, 0]);
        assert_eq!((f.uids(), f.caps()), ([0; 3], [all, all, 0]));
        // PR_SET_KEEPCAPS keeps the permitted set across leaving root.
        let k = Creds::new(0, 0);
        k.set_keep_caps(true);
        run(&k, nr::SETUID, [996, 0, 0]);
        assert_eq!(k.caps(), [0, all, 0]);
        // capset within the sets only.
        let r = Creds::new(0, 0);
        assert_eq!(r.capset(all & !(1 << 13), all, 0), Ok(()));
        assert_eq!(r.capset(all, all & !(1 << 13), 0), Err(Errno::EPERM));
        assert_eq!(r.capset(all & !(1 << 13), all & !(1 << 13), 0), Ok(()));
        assert_eq!(r.capset(all, all, 0), Err(Errno::EPERM));
        // The bytes a page's other Worker restores from.
        let b = Creds::new(0, 0);
        b.restore(&c.save());
        assert_eq!((b.uids(), b.gids(), b.caps()), (c.uids(), c.gids(), c.caps()));
    }

    /// `kill`, `tkill` and `tgkill` answer as the kernel's arms do,
    /// and one for another thread of the process is raised on that
    /// thread's own state, found through the shared table.
    #[test]
    fn free_kill_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        let (pid, tid) = (k.pid as u64, k.tid as u64);
        let cases: Vec<(u64, [u64; 3])> = vec![
            (nr::KILL, [pid, 0, 0]),
            (nr::KILL, [99999, 0, 0]),
            (nr::TKILL, [tid, 0, 0]),
            (nr::TKILL, [0, 0, 0]),
            (nr::TKILL, [tid, 99, 0]),
            (nr::TGKILL, [pid, tid, 0]),
            (nr::TGKILL, [pid + 999, tid, 0]),
            (nr::TGKILL, [pid, 99999, 0]),
            (nr::TGKILL, [0, tid, 0]),
        ];
        for (n, args) in cases {
            set_args(&mut m, at, n, &args);
            k.syscall(&mut m, 0);
            let want = ret(&m, at);
            set_args(&mut m, at, n, &args);
            assert!(done(free(&k, &mut m, at)), "call {n} in the class");
            assert_eq!(ret(&m, at), want, "call {n} {args:?}");
        }
        // Another thread's: on its own pending word, with the sender named,
        // and none on the caller's.
        const SIGUSR2: u64 = 12;
        let other = Arc::new(signal::ThreadSig::new_thread(0));
        k.thread_signals.insert(4242, Arc::clone(&other));
        set_args(&mut m, at, nr::TGKILL, &[pid, 4242, SIGUSR2]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 0);
        assert_eq!(other.pending(), 1 << (SIGUSR2 - 1));
        assert_eq!(k.signals.thread().pending() & (1 << (SIGUSR2 - 1)), 0);

        // rt_sigtimedwait: nothing pending with a zero timeout is EAGAIN
        // both ways; a blocked signal the process sent itself is taken,
        // with its sender, the same both ways.
        let set_at = 0x31000u64;
        let info_at = 0x31100u64;
        let zero_at = 0x31200u64;
        m.write_u64(set_at, 1 << (SIGUSR2 - 1)).unwrap();
        m.zero(zero_at, 16).unwrap();
        k.signals.set_blocked(1 << (SIGUSR2 - 1));
        let mut answers = Vec::new();
        for kernel in [true, false] {
            set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, zero_at, 8]);
            if kernel {
                k.syscall(&mut m, 0);
            } else {
                assert!(done(free(&k, &mut m, at)));
            }
            assert_eq!(ret(&m, at), -(Errno::EAGAIN.0 as i64));
            set_args(&mut m, at, nr::KILL, &[pid, SIGUSR2]);
            k.syscall(&mut m, 0);
            m.zero(info_at, 128).unwrap();
            set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, 0, 8]);
            if kernel {
                k.syscall(&mut m, 0);
            } else {
                assert!(done(free(&k, &mut m, at)));
            }
            answers.push((ret(&m, at), m.read(info_at, 32).unwrap().to_vec()));
        }
        assert_eq!(answers[0].0, SIGUSR2 as i64);
        assert_eq!(answers[0], answers[1]);

        // rt_sigsuspend: the asked-for mask in place (SIGKILL never
        // blocked), a park on SigSuspend, the old mask owed back.
        k.signals.set_blocked(0x5);
        m.write_u64(set_at, !0).unwrap();
        set_args(&mut m, at, nr::RT_SIGSUSPEND, &[set_at, 8]);
        match free(&k, &mut m, at) {
            Some(FreeOutcome::Blocking(BlockOp::SigSuspend { restore_mask: 0x5 }, Some(0x5))) => {}
            _ => panic!("rt_sigsuspend did not park"),
        }
        assert_eq!(k.signals.blocked(), !0 & !((1 << 8) | (1 << 18)));
        k.signals.set_blocked(0);
        set_args(&mut m, at, nr::RT_SIGSUSPEND, &[0x7fff_ffff_0000, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), -(Errno::EFAULT.0 as i64));
    }

    /// `set_tid_address` and `unshare` answer the same with and
    /// without the kernel, on the calling thread's own state: the word its
    /// exit clears is in its `ThreadSig` (`CLONE_CHILD_CLEARTID`'s from its
    /// start), so a thread that names one without the lock has it cleared
    /// and woken by its exit under the lock, and its siblings' are their
    /// own. `unshare(CLONE_FS)` gives the caller a fs_struct of its own.
    #[test]
    fn thread_own_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let main = ThreadState::new(k.ctx_addr, k.tid);
        k.enter_thread(&main);
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let (word, word2, ctid) = (0x31000u64, 0x31010u64, 0x31020u64);

        // A thread made with CLONE_CHILD_CLEARTID has the word from its start.
        let flags = clone_flags::VM | clone_flags::FS | clone_flags::THREAD | clone_flags::CHILD_CLEARTID;
        let req = k.sys_clone(&mut m, [flags, 0x30000, 0, 0, ctid, 0]).expect("clone");
        let kid = k.thread_signals.get(req.tid).expect("its own state");
        assert_eq!(kid.clear_child_tid(), ctid);
        assert_eq!(k.signals.thread().clear_child_tid(), 0, "not its creator's");

        // set_tid_address: the tid, and the caller's word, both ways.
        for (kernel, addr) in [(true, word), (false, word2)] {
            set_args(&mut m, main.ctx_addr, nr::SET_TID_ADDRESS, &[addr]);
            if kernel {
                k.syscall(&mut m, 0);
            } else {
                assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Done(Delivery::None)));
            }
            assert_eq!(ret(&m, main.ctx_addr), k.tid as i64);
            assert_eq!(k.signals.thread().clear_child_tid(), addr);
        }
        assert_eq!(kid.clear_child_tid(), ctid, "a sibling's is its own");

        // The child names a word without the lock; its exit, under it,
        // clears that one and leaves the clone's alone.
        k.enter_thread(&ThreadState::new(req.ctx_addr, req.tid));
        set_args(&mut m, req.ctx_addr, nr::SET_TID_ADDRESS, &[word]);
        assert!(matches!(free(&k, &mut m, req.ctx_addr), FreeOutcome::Done(Delivery::None)));
        assert_eq!(ret(&m, req.ctx_addr), req.tid as i64);
        m.write_u32(word, 7).unwrap();
        m.write_u32(ctid, 7).unwrap();
        set_args(&mut m, req.ctx_addr, nr::EXIT, &[0]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::ThreadExit));
        assert_eq!(m.read_u32(word).unwrap(), 0);
        assert_eq!(m.read_u32(ctid).unwrap(), 7);
        k.enter_thread(&main);
        assert_eq!(k.signals.thread().clear_child_tid(), word2, "main's own still");

        // unshare: CLONE_FS gives the caller its own fs_struct; anything
        // else is EINVAL, both ways.
        let shared = k.fs_of(main.ctx_addr);
        for kernel in [true, false] {
            for (flags, want) in [(0x400u64, -(Errno::EINVAL.0 as i64)), (0, 0), (clone_flags::FS, 0)] {
                set_args(&mut m, main.ctx_addr, nr::UNSHARE, &[flags]);
                if kernel {
                    k.syscall(&mut m, 0);
                } else {
                    assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Done(Delivery::None)));
                }
                assert_eq!(ret(&m, main.ctx_addr), want, "unshare({flags:#x})");
            }
            let own = k.fs_of(main.ctx_addr);
            assert!(!Arc::ptr_eq(&own, &shared), "its own fs_struct");
            assert_eq!(own.get(), shared.get());
        }

        // An exec forgets the word (it was in the old image).
        k.exec_signals(k.tid);
        assert_eq!(k.signals.thread().clear_child_tid(), 0);
    }

    /// `exit`, `exit_group`, `wait4` and `reboot` answer the same
    /// with and without the kernel: an exit of one of several threads ends
    /// only it (its joiner's word cleared, its signal state and id gone,
    /// the count down by one); the last thread's exit and `exit_group` end
    /// the process with their code; `wait4` parks on the child table; and
    /// `reboot` halts the machine only when told to.
    #[test]
    fn lifecycle_end_calls_answer_as_the_kernel_does() {
        use std::sync::atomic::Ordering::SeqCst;
        let (mut buf, mut k) = setup();
        let main = ThreadState::new(k.ctx_addr, k.tid);
        k.enter_thread(&main);
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let ctid = 0x31020u64;
        let flags = clone_flags::VM | clone_flags::FS | clone_flags::THREAD | clone_flags::CHILD_CLEARTID;
        for kernel in [true, false] {
            let req = k.sys_clone(&mut m, [flags, 0x30000, 0, 0, ctid, 0]).expect("clone");
            assert_eq!(k.live_threads.load(SeqCst), 2);
            m.write_u32(ctid, 7).unwrap();
            k.enter_thread(&ThreadState::new(req.ctx_addr, req.tid));
            set_args(&mut m, req.ctx_addr, nr::EXIT, &[3]);
            if kernel {
                assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::ThreadExit));
            } else {
                assert!(matches!(free(&k, &mut m, req.ctx_addr), FreeOutcome::ThreadExit));
            }
            assert_eq!(m.read_u32(ctid).unwrap(), 0, "its joiner's word cleared");
            assert!(k.thread_signals.get(req.tid).is_none(), "its signal state gone");
            assert_eq!(k.live_threads.load(SeqCst), 1);
            k.enter_thread(&main);
        }
        // The last thread's exit and exit_group: the process ends.
        for (n, code) in [(nr::EXIT, 0x105u64), (nr::EXIT_GROUP, 9)] {
            set_args(&mut m, main.ctx_addr, n, &[code]);
            let want = (code & 0xff) as i32;
            assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Exit(c) if c == want));
            set_args(&mut m, main.ctx_addr, n, &[code]);
            assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Exit(c) if c == want));
        }
        // wait4 parks on the child table.
        let pid = k.pid;
        set_args(&mut m, main.ctx_addr, nr::WAIT4, &[-1i64 as u64, 0x32000, 1, 0]);
        assert!(matches!(
            free(&k, &mut m, main.ctx_addr),
            FreeOutcome::Blocking(BlockOp::WaitChild { ppid, want: -1, wstatus: 0x32000, nohang: true, .. }, None) if ppid == pid
        ));
        set_args(&mut m, main.ctx_addr, nr::WAIT4, &[-1i64 as u64, 0x32000, 0, 0]);
        assert!(matches!(
            k.syscall(&mut m, 0),
            SyscallOutcome::Blocking(BlockOp::WaitChild { ppid, want: -1, nohang: false, .. }) if ppid == pid
        ));
        // waitid parks the same, with its siginfo and WNOWAIT; no wait kind
        // is EINVAL, with and without the kernel.
        set_args(&mut m, main.ctx_addr, nr::WAITID, &[1, 7, 0x32000, 4 | 1 | 0x0100_0000]);
        let info = WaitidOut { info: 0x32000, nowait: true, compat: false };
        assert!(matches!(
            free(&k, &mut m, main.ctx_addr),
            FreeOutcome::Blocking(BlockOp::WaitChild { ppid, want: 7, nohang: true, waitid: Some(w), .. }, None) if ppid == pid && w == info
        ));
        set_args(&mut m, main.ctx_addr, nr::WAITID, &[0, 0, 0x32000, 4]);
        assert!(matches!(
            k.syscall(&mut m, 0),
            SyscallOutcome::Blocking(BlockOp::WaitChild { want: -1, nohang: false, waitid: Some(WaitidOut { nowait: false, .. }), .. })
        ));
        // (P_PIDFD of a descriptor that is not open: EBADF)
        for (args, e) in [([3u64, 5, 0x32000, 4 | 1], Errno::EBADF), ([0, 0, 0x32000, 1], Errno::EINVAL)] {
            set_args(&mut m, main.ctx_addr, nr::WAITID, &args);
            assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Done(_)));
            assert_eq!(ret(&m, main.ctx_addr), e.as_return() as i64);
            set_args(&mut m, main.ctx_addr, nr::WAITID, &args);
            assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
            assert_eq!(ret(&m, main.ctx_addr), e.as_return() as i64);
        }
        // reboot: CAD_OFF answers 0 and halts nothing; a halt ends the
        // process with 0 and halts the machine.
        for kernel in [true, false] {
            set_args(&mut m, main.ctx_addr, nr::REBOOT, &[0xfee1_dead, 0x2812_1969, 0]);
            if kernel {
                assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
            } else {
                assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Done(Delivery::None)));
            }
            assert_eq!(ret(&m, main.ctx_addr), 0);
        }
        assert_eq!(k.procs.halt_code(), None);
        set_args(&mut m, main.ctx_addr, nr::REBOOT, &[0xfee1_dead, 0x2812_1969, 0x4321_fedc]);
        assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Exit(0)));
        assert_eq!(k.procs.halt_code(), Some(0));
    }

    /// `waitid`'s `siginfo_t`: SIGCHLD, the CLD_* code, the pid and the exit
    /// code or signal at the 64-bit and the 32-bit offsets; nothing ready is
    /// all zero.
    #[test]
    fn waitid_reports_a_siginfo() {
        let mut buf = vec![0xaau8; 0x100];
        let mut m = GuestMem::new(&mut buf);
        let w = WaitidOut { info: 0x10, nowait: false, compat: false };
        assert_eq!(w.report(&mut m, 42, 3 << 8), Ok(0));
        let rd = |m: &GuestMem<'_>, o: u64| m.read_u32(0x10 + o).unwrap();
        assert_eq!([rd(&m, 0), rd(&m, 4), rd(&m, 8), rd(&m, 16), rd(&m, 20), rd(&m, 24)], [17, 0, 1, 42, 0, 3]);
        assert_eq!(w.report(&mut m, 42, 9), Ok(0));
        assert_eq!([rd(&m, 8), rd(&m, 24)], [2, 9]);
        assert_eq!(w.report(&mut m, 42, 0x80 | 11), Ok(0));
        assert_eq!([rd(&m, 8), rd(&m, 24)], [3, 11]);
        assert_eq!(w.report(&mut m, 0, 0), Ok(0));
        assert!((0..28).step_by(4).all(|o| rd(&m, o) == 0));
        let c = WaitidOut { compat: true, ..w };
        assert_eq!(c.report(&mut m, 5, 1 << 8), Ok(0));
        assert_eq!([rd(&m, 0), rd(&m, 8), rd(&m, 12), rd(&m, 20)], [17, 1, 5, 1]);
    }

    /// `clone` and `clone3` that make a thread answer the same with
    /// and without the kernel: the creator gets the tid, the child a copy of
    /// its context answering 0 on its own stack, its own slot, id and signal
    /// state (with the word its exit clears), and the count goes up. A
    /// `clone` that makes a process is not the free path's; a missing stack
    /// is EINVAL both ways.
    #[test]
    fn thread_clone_calls_answer_as_the_kernel_does() {
        use std::sync::atomic::Ordering::SeqCst;
        let (mut buf, mut k) = setup();
        let main = ThreadState::new(k.ctx_addr, k.tid);
        k.enter_thread(&main);
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let flags = clone_flags::VM | clone_flags::FS | clone_flags::THREAD | clone_flags::CHILD_CLEARTID | clone_flags::PARENT_SETTID;
        let (ptid, ctid, args3) = (0x31000u64, 0x31010u64, 0x31100u64);
        let mut seen = Vec::new();
        for (i, (kernel, three)) in [(true, false), (false, false), (true, true), (false, true)].into_iter().enumerate() {
            let stack = 0x40000 + 0x1000 * i as u64;
            if three {
                // struct clone_args: flags, pidfd, child_tid, parent_tid,
                // exit_signal, stack (the base), stack_size, tls.
                for (o, v) in [(0u64, flags), (8, 0), (16, ctid), (24, ptid), (32, 0), (40, stack - 0x800), (48, 0x800), (56, 0)] {
                    m.write_u64(args3 + o, v).unwrap();
                }
                set_args(&mut m, main.ctx_addr, nr::CLONE3, &[args3, 88]);
            } else {
                // x86-64 has the child's tid word before the TLS.
                let args = if gex_abi::X86 { [flags, stack, ptid, ctid, 0] } else { [flags, stack, ptid, 0, ctid] };
                set_args(&mut m, main.ctx_addr, nr::CLONE, &args);
                // x86-64's `fork` and `vfork` are `clone` canonically too,
                // and `set_args` picks the highest: its own `clone` here.
                if gex_abi::X86 {
                    m.write_u64(main.ctx_addr + ctx::SYSCALL_NR as u64, crate::x86::raw::CLONE).unwrap();
                }
            }
            let made = if kernel {
                match k.syscall(&mut m, 0) {
                    SyscallOutcome::SpawnThread { ctx_addr, tid } => (ctx_addr, tid),
                    _ => panic!("clone {i}: no thread"),
                }
            } else {
                match free(&k, &mut m, main.ctx_addr) {
                    FreeOutcome::SpawnThread { ctx_addr, tid } => (ctx_addr, tid),
                    _ => panic!("clone {i}: no thread"),
                }
            };
            let (child, tid) = made;
            assert_eq!(ret(&m, main.ctx_addr), tid as i64, "clone {i}: the creator's answer");
            assert_eq!(ret(&m, child), 0, "clone {i}: the child's answer");
            assert_eq!(m.read_u64(child + ctx::sp_slot(false) as u64).unwrap(), stack, "clone {i}: its stack");
            assert_eq!(m.read_u32(ptid).unwrap(), tid as u32, "clone {i}: PARENT_SETTID");
            assert_eq!(k.thread_signals.get(tid).expect("its own state").clear_child_tid(), ctid);
            assert!(!seen.iter().any(|&(c, t)| c == child || t == tid), "clone {i}: a slot and id of its own");
            seen.push((child, tid));
            assert_eq!(k.live_threads.load(SeqCst), 2 + i as u32);
        }
        // A missing stack, both ways.
        for kernel in [true, false] {
            set_args(&mut m, main.ctx_addr, nr::CLONE, &[flags, 0, 0, 0, 0]);
            if gex_abi::X86 {
                m.write_u64(main.ctx_addr + ctx::SYSCALL_NR as u64, crate::x86::raw::CLONE).unwrap();
            }
            if kernel {
                k.syscall(&mut m, 0);
            } else {
                assert!(matches!(free(&k, &mut m, main.ctx_addr), FreeOutcome::Done(Delivery::None)));
            }
            assert_eq!(ret(&m, main.ctx_addr), -(Errno::EINVAL.0 as i64));
        }
        // A process is the host's, and so is a fork whose argument
        // registers happen to hold CLONE_THREAD.
        set_args(&mut m, main.ctx_addr, nr::CLONE, &[17, 0, 0, 0, 0]);
        assert!(free_class(&m, main.ctx_addr, false).is_none());
        if gex_abi::X86 {
            m.write_u64(main.ctx_addr + ctx::SYSCALL_NR as u64, crate::x86::raw::FORK).unwrap();
            m.write_u64(main.ctx_addr + ctx::SYSCALL_ARGS[0] as u64, flags).unwrap();
            assert!(free_class(&m, main.ctx_addr, false).is_none());
        }
        m.write_u64(args3, 0).unwrap();
        set_args(&mut m, main.ctx_addr, nr::CLONE3, &[args3, 88]);
        assert!(free_class(&m, main.ctx_addr, false).is_none());
    }

    /// `mount` and `umount2` answer the same with and without the
    /// kernel. The mount table's one writer is its published view: a mount
    /// made without the lock is the kernel's at its next call, and mounts
    /// made from several threads at once, beside the kernel's own and its
    /// `chdir`s, are all kept.
    #[test]
    fn mounts_are_made_through_the_view() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let (mnt, tmpfs, proc_, sysfs, none) = (0x31000u64, 0x31100u64, 0x31200u64, 0x31300u64, 0x31400u64);
        m.write(mnt, b"/mnt\0").unwrap();
        m.write(tmpfs, b"tmpfs\0").unwrap();
        m.write(proc_, b"proc\0").unwrap();
        m.write(sysfs, b"sysfs\0").unwrap();
        m.write(none, b"none\0").unwrap();
        let call = |k: &mut Kernel, m: &mut GuestMem<'_>, kernel: bool, n: u64, args: &[u64]| -> i64 {
            set_args(m, at, n, args);
            if kernel {
                k.syscall(m, 0);
            } else {
                assert!(matches!(free(k, m, at), FreeOutcome::Done(Delivery::None)), "call {n} in the class");
            }
            ret(m, at)
        };
        for kernel in [false, true] {
            // proc only where something is mounted already; an unknown type
            // is ENODEV.
            assert_eq!(call(&mut k, &mut m, kernel, nr::MOUNT, &[none, mnt, proc_, 0, 0]), -(Errno::ENODEV.0 as i64));
            assert_eq!(call(&mut k, &mut m, kernel, nr::MOUNT, &[none, mnt, none, 0, 0]), -(Errno::ENODEV.0 as i64));
            // A tmpfs: the view's at once, the kernel's at its next call.
            assert_eq!(call(&mut k, &mut m, kernel, nr::MOUNT, &[none, mnt, tmpfs, 0, 0]), 0);
            assert!(k.vfs.view().get().is_mount_point("/mnt"));
            call(&mut k, &mut m, true, nr::GETPID, &[]);
            assert!(k.vfs.is_mount_point("/mnt"), "the kernel's copy follows");
            // And now proc there is satisfied, and sysfs keeps what is there.
            assert_eq!(call(&mut k, &mut m, kernel, nr::MOUNT, &[none, mnt, proc_, 0, 0]), 0);
            assert_eq!(call(&mut k, &mut m, kernel, nr::MOUNT, &[none, mnt, sysfs, 0, 0]), 0);
            // Gone, then EINVAL: nothing is mounted there any more.
            assert_eq!(call(&mut k, &mut m, kernel, nr::UMOUNT2, &[mnt, 0]), 0);
            assert!(!k.vfs.view().get().is_mount_point("/mnt"));
            assert_eq!(call(&mut k, &mut m, kernel, nr::UMOUNT2, &[mnt, 0]), -(Errno::EINVAL.0 as i64));
            call(&mut k, &mut m, true, nr::GETPID, &[]);
            assert!(!k.vfs.is_mount_point("/mnt"), "the kernel's copy follows");
        }

        // Four threads mount and unmount beside the kernel's own mounts
        // and directory changes: nothing is lost.
        let view = Arc::clone(k.vfs.view());
        std::thread::scope(|s| {
            for t in 0..4 {
                let view = &view;
                s.spawn(move || {
                    for r in 0..50 {
                        view.mount(&format!("/c{t}/{r}"), Arc::new(TmpFs::new()));
                        if r % 2 == 1 {
                            assert!(view.unmount(&format!("/c{t}/{r}")));
                        }
                    }
                });
            }
            for r in 0..50 {
                k.vfs.mount(&format!("/k{r}"), Arc::new(TmpFs::new()));
                k.vfs.put_cwd(format!("/k{r}"));
            }
        });
        k.vfs.refresh();
        for ns in [&*view.get(), &*k.vfs] {
            for t in 0..4 {
                for r in 0..50 {
                    assert_eq!(ns.is_mount_point(&format!("/c{t}/{r}")), r % 2 == 0, "/c{t}/{r}");
                }
            }
            for r in 0..50 {
                assert!(ns.is_mount_point(&format!("/k{r}")), "/k{r}");
            }
        }
        assert_eq!(view.get().cwd, "/k49");
    }

    /// `rt_sigtimedwait` with none of its set pending waits (it had answered
    /// `EAGAIN` at once, whatever the timeout): it parks on `SigWait`, with
    /// and without the lock, and the kernel's finish takes the signal that
    /// ended the wait, or passes on the host's `EAGAIN` (time up) and
    /// `EINTR`. A bad timeout is `EINVAL` before anything is taken.
    #[test]
    fn sigtimedwait_waits_for_its_set() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        const SIGUSR1: u64 = 10;
        let bit = 1u64 << (SIGUSR1 - 1);
        let (set_at, info_at, ts_at, bad_at) = (0x31000u64, 0x31100u64, 0x31200u64, 0x31300u64);
        m.write_u64(set_at, bit).unwrap();
        m.write_u64(ts_at, 2).unwrap();
        m.write_u64(ts_at + 8, 500).unwrap();
        m.write_u64(bad_at, 0).unwrap();
        m.write_u64(bad_at + 8, 1_000_000_000).unwrap();
        k.signals.set_blocked(bit);
        let pid = k.pid as u64;

        // Without the lock: a park on its set, with the timeout.
        set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, ts_at, 8]);
        let call = free_class(&m, at, false).expect("in the class");
        let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
        let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
        match free_call(&f, &mut m, at, call) {
            FreeOutcome::Blocking(BlockOp::SigWait { set, timeout, .. }, None) => {
                assert_eq!(set, bit);
                assert_eq!(timeout, Some(std::time::Duration::new(2, 500)));
            }
            _ => panic!("rt_sigtimedwait did not park"),
        }

        // With it: the same park; the signal raised meanwhile is what the
        // host saw, and the finish takes it with its siginfo_t.
        set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, 0, 8]);
        let op = match k.syscall(&mut m, 0) {
            SyscallOutcome::Blocking(op) => op,
            _ => panic!("rt_sigtimedwait did not park under the lock"),
        };
        let BlockOp::SigWait { set, timeout: None, thread, process } = op else { panic!("not SigWait: {op:?}") };
        assert_eq!((thread.pending() | process.shared()) & set, 0);
        k.signals.raise(SIGUSR1 as u32);
        assert_ne!((thread.pending() | process.shared()) & set, 0, "the waiter sees it without the kernel");
        m.zero(info_at, 128).unwrap();
        k.finish_blocking(&mut m, Ok(0));
        assert_eq!(ret(&m, at), SIGUSR1 as i64);
        assert_eq!(m.read_u32(info_at).unwrap(), SIGUSR1 as u32);
        assert_eq!(k.signals.pending_raw() & bit, 0, "taken");

        // Time up and interrupted: the host's answer is the call's.
        for (host, want) in [(Errno::EAGAIN, Errno::EAGAIN), (Errno::EINTR, Errno::EINTR)] {
            set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, ts_at, 8]);
            assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Blocking(BlockOp::SigWait { .. })));
            k.finish_blocking(&mut m, Err(host));
            assert_eq!(ret(&m, at), -(want.0 as i64));
        }

        // A tgkill to the waiting thread itself: taken with its sender
        // (SI_TKILL), not SI_KERNEL (its sender was looked for on the
        // process's set once the thread's pending bit was cleared).
        set_args(&mut m, at, nr::TGKILL, &[pid, k.tid as u64, SIGUSR1]);
        k.syscall(&mut m, 0);
        m.zero(info_at, 128).unwrap();
        set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, ts_at, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), SIGUSR1 as i64);
        assert_eq!(m.read_u32(info_at + 8).unwrap() as i32, signal::SI_TKILL);
        assert_eq!(m.read_u32(info_at + 16).unwrap(), k.pid as u32);

        // Pending already: answered at once, no park. A bad timeout: EINVAL.
        set_args(&mut m, at, nr::KILL, &[pid, SIGUSR1]);
        k.syscall(&mut m, 0);
        set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, ts_at, 8]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue | SyscallOutcome::Resume(_)));
        assert_eq!(ret(&m, at), SIGUSR1 as i64);
        set_args(&mut m, at, nr::RT_SIGTIMEDWAIT, &[set_at, info_at, bad_at, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), -(Errno::EINVAL.0 as i64));
    }

    #[test]
    fn free_i386_quiet_calls_answer_as_the_kernel_does() {
        if !gex_abi::X86 {
            return;
        }
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        k.compat = true;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, true).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let set32 = |m: &mut GuestMem<'_>, n: u64, a: [u64; 3]| {
            m.write_u64(at + ctx::x(0) as u64, n).unwrap();
            for (i, v) in a.iter().enumerate() {
                m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
            }
        };
        let r32 = |m: &GuestMem<'_>| m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32;
        use crate::i386::raw;
        let buf_at = 0x30000u64;
        for (n, args) in [
            (raw::UNAME, [buf_at, 0, 0]),
            (raw::GETCPU, [buf_at, buf_at + 4, 0]),
            (raw::SCHED_GETAFFINITY, [0, 64, buf_at]),
            (raw::FSYNC, [1, 0, 0]),
            (raw::FSYNC, [0xffff_ffff, 0, 0]),
            (raw::GETGROUPS, [0, 0, 0]),
            (raw::PERSONALITY, [0, 0, 0]),
            (raw::GETPPID, [0, 0, 0]),
        ] {
            m.zero(buf_at, 512).unwrap();
            set32(&mut m, n, args);
            k.syscall(&mut m, 0);
            let (want, want_ret) = (m.read(buf_at, 512).unwrap().to_vec(), r32(&m));
            m.zero(buf_at, 512).unwrap();
            set32(&mut m, n, args);
            assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)), "raw {n}");
            assert_eq!(m.read(buf_at, 512).unwrap(), &want[..], "raw {n}");
            assert_eq!(r32(&m), want_ret, "raw {n}");
        }
        // A timerfd armed and re-armed through the 32-bit itimerspec, its
        // old value narrowed back, either way.
        for lock_free in [true, false] {
            let mut go = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(matches!(free(k, m, at), FreeOutcome::Done(_)));
                } else {
                    k.syscall(m, 0);
                }
            };
            set32(&mut m, raw::TIMERFD_CREATE, [1, 0, 0]);
            go(&mut m, &mut k);
            let tfd = r32(&m) as u64;
            m.zero(0x31000, 64).unwrap();
            m.write_u32(0x31008, 100).unwrap(); // it_value.tv_sec
            m.write_u32(0x31000, 7).unwrap(); // it_interval.tv_sec
            set32(&mut m, raw::TIMERFD_SETTIME, [tfd, 0, 0x31000]);
            m.write_u64(at + crate::i386::ARG_SLOTS[3] as u64, 0).unwrap();
            go(&mut m, &mut k);
            assert_eq!(r32(&m), 0, "lock_free {lock_free}");
            set32(&mut m, raw::TIMERFD_SETTIME, [tfd, 0, 0x31000]);
            m.write_u64(at + crate::i386::ARG_SLOTS[3] as u64, 0x31100).unwrap();
            m.write_u64(0x31110, !0).unwrap();
            go(&mut m, &mut k);
            assert_eq!(r32(&m), 0);
            assert_eq!(m.read_u32(0x31100).unwrap(), 7, "old interval, 32-bit");
            let left = m.read_u32(0x31108).unwrap();
            assert!(left == 99 || left == 100, "{left}");
            assert_eq!(m.read_u64(0x31110).unwrap(), !0, "nothing past the 16 bytes");
            set32(&mut m, raw::CLOSE, [tfd, 0, 0]);
            k.syscall(&mut m, 0);
        }
        // uname's machine is i686 for an i386 process.
        set32(&mut m, raw::UNAME, [buf_at, 0, 0]);
        let _ = free(&k, &mut m, at);
        assert_eq!(&m.read(buf_at + 4 * 65, 4).unwrap(), b"i686");
        // set_thread_area: the entry number chosen and the base in %gs, as
        // the full call's translation does.
        m.write_u32(buf_at + 0x100, u32::MAX).unwrap();
        m.write_u32(buf_at + 0x104, 0x0abc_d000).unwrap();
        set32(&mut m, raw::SET_THREAD_AREA, [buf_at + 0x100, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(r32(&m), 0);
        assert_eq!(m.read_u32(buf_at + 0x100).unwrap(), 6);
        assert_eq!(m.read_u64(at + ctx::GS_BASE as u64).unwrap(), 0x0abc_d000);
        // time(&t): a 32-bit time_t stored, nothing past it (the clocks'
        // own path answers it).
        m.write_u64(buf_at, !0).unwrap();
        set32(&mut m, raw::TIME, [buf_at, 0, 0]);
        assert!(clock_call(&mut m, at, true, k.clock, k.started, &|| 0));
        assert_eq!(m.read_u32(buf_at + 4).unwrap(), u32::MAX, "only four bytes stored");
        assert_eq!(m.read_u32(buf_at).unwrap() as i32, r32(&m));
    }

    #[test]
    fn free_fd_object_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        for lock_free in [true, false] {
            let call = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(done(free(k, m, at)));
                } else {
                    k.syscall(m, 0);
                }
            };
            let full = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, args: &[u64]| {
                set_args(m, at, n, args);
                k.syscall(m, 0);
                ret(m, at) as u64
            };
            // eventfd2: a counter, written and read back.
            set_args(&mut m, at, nr::EVENTFD2, &[3, 0]);
            call(&mut m, &mut k);
            let efd = ret(&m, at) as u64;
            assert!(efd < 100, "lock_free {lock_free}");
            assert_eq!(k.fds.get(efd as i32).unwrap().path, "anon_inode:[eventfd]");
            assert_eq!(full(&mut m, &mut k, nr::READ, &[efd, 0x20000, 8]), 8);
            assert_eq!(m.read_u64(0x20000).unwrap(), 3);
            // epoll_create1 with CLOEXEC.
            set_args(&mut m, at, nr::EPOLL_CREATE1, &[0o2000000]);
            call(&mut m, &mut k);
            let ep = ret(&m, at) as u64;
            assert_eq!(k.fds.get(ep as i32).unwrap().path, "anon_inode:[eventpoll]");
            // memfd_create, then fallocate to a size the descriptor reports.
            m.write(0x20100, b"buf\0").unwrap();
            set_args(&mut m, at, nr::MEMFD_CREATE, &[0x20100, 1]);
            call(&mut m, &mut k);
            let mfd = ret(&m, at) as u64;
            assert_eq!(k.fds.get(mfd as i32).unwrap().path, "/memfd:buf");
            set_args(&mut m, at, nr::FALLOCATE, &[mfd, 0, 0, 8192]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            assert_eq!(k.fds.get(mfd as i32).unwrap().handle.metadata().unwrap().size, 8192);
            set_args(&mut m, at, nr::FALLOCATE, &[mfd, 1, 0, 8192]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EOPNOTSUPP.0 as i64));
            // A timerfd armed for ten seconds says about ten seconds left.
            set_args(&mut m, at, nr::TIMERFD_CREATE, &[1, 0]);
            call(&mut m, &mut k);
            let tfd = ret(&m, at) as u64;
            m.zero(0x20200, 64).unwrap();
            m.write_u64(0x20210, 10).unwrap();
            set_args(&mut m, at, nr::TIMERFD_SETTIME, &[tfd, 0, 0x20200, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            set_args(&mut m, at, nr::TIMERFD_GETTIME, &[tfd, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            let left = m.read_u64(0x20310).unwrap();
            assert!(left == 9 || left == 10, "{left}");
            // The transfers: a file into a pipe, then a pipe into a file.
            m.write(0x20400, b"/src\0").unwrap();
            let src = full(&mut m, &mut k, nr::OPENAT, &[(-100i64) as u64, 0x20400, (oflags::O_RDWR | oflags::O_CREAT | oflags::O_TRUNC) as u64, 0o644]);
            m.write(0x20500, b"transferred").unwrap();
            assert_eq!(full(&mut m, &mut k, nr::WRITE, &[src, 0x20500, 11]), 11);
            set_args(&mut m, at, nr::PIPE2, &[0x20600, 0]);
            k.syscall(&mut m, 0);
            let (r, w) = (m.read_u32(0x20600).unwrap() as u64, m.read_u32(0x20604).unwrap() as u64);
            m.write_u64(0x20700, 0).unwrap();
            set_args(&mut m, at, nr::SENDFILE, &[w, src, 0x20700, 64]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 11);
            assert_eq!(m.read_u64(0x20700).unwrap(), 11, "the offset advanced");
            m.write(0x20800, b"/dst\0").unwrap();
            let dst = full(&mut m, &mut k, nr::OPENAT, &[(-100i64) as u64, 0x20800, (oflags::O_RDWR | oflags::O_CREAT | oflags::O_TRUNC) as u64, 0o644]);
            set_args(&mut m, at, nr::SPLICE, &[r, 0, dst, 0, 64, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 11);
            m.write_u64(0x20900, 0).unwrap();
            m.write_u64(0x20908, 11).unwrap();
            set_args(&mut m, at, nr::COPY_FILE_RANGE, &[src, 0x20900, dst, 0x20908, 64, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 11);
            assert_eq!(full(&mut m, &mut k, nr::PREAD64, &[dst, 0x20a00, 64, 0]), 22);
            assert_eq!(m.read(0x20a00, 22).unwrap(), b"transferredtransferred");
            // An empty blocking pipe: the transfer waits, both ways.
            set_args(&mut m, at, nr::SPLICE, &[r, 0, dst, 0, 64, 0]);
            if lock_free {
                assert!(matches!(free(&k, &mut m, at), Some(FreeOutcome::Blocking(BlockOp::AcceptWait { .. }, None))));
            } else {
                assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Blocking(BlockOp::AcceptWait { .. })));
                k.retry_transfer = None;
            }
            for f in [efd, ep, mfd, tfd, src, r, w, dst] {
                set_args(&mut m, at, nr::CLOSE, &[f]);
                k.syscall(&mut m, 0);
            }
        }
    }

    #[test]
    fn free_fd_misc_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        for lock_free in [true, false] {
            let call = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(done(free(k, m, at)));
                } else {
                    k.syscall(m, 0);
                }
            };
            // A pipe made one way carries bytes the other way.
            set_args(&mut m, at, nr::PIPE2, &[0x20000, oflags::O_CLOEXEC]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0, "lock_free {lock_free}");
            let (r, w) = (m.read_u32(0x20000).unwrap() as u64, m.read_u32(0x20004).unwrap() as u64);
            assert_eq!(k.fds.get(r as i32).unwrap().flags() & 3, oflags::O_RDONLY);
            m.write(0x20100, b"pipe").unwrap();
            set_args(&mut m, at, nr::WRITE, &[w, 0x20100, 4]);
            k.syscall(&mut m, 0);
            set_args(&mut m, at, nr::READ, &[r, 0x20200, 4]);
            k.syscall(&mut m, 0);
            assert_eq!(m.read(0x20200, 4).unwrap(), b"pipe");

            // An epoll set watches what either side added: a second add is
            // EEXIST, a watch of a closed descriptor EBADF.
            set_args(&mut m, at, nr::EPOLL_CREATE1, &[0]);
            k.syscall(&mut m, 0);
            let ep = ret(&m, at) as u64;
            m.write_u32(0x20300, 1).unwrap(); // EPOLLIN
            m.write_u64(if gex_abi::X86 { 0x20304 } else { 0x20308 }, 0x77).unwrap();
            set_args(&mut m, at, nr::EPOLL_CTL, &[ep, 1, r, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            set_args(&mut m, at, nr::EPOLL_CTL, &[ep, 1, r, 0x20300]);
            if lock_free { let _ = k.syscall(&mut m, 0); } else { assert!(done(free(&k, &mut m, at))); }
            assert_eq!(ret(&m, at), -(Errno::EEXIST.0 as i64));
            set_args(&mut m, at, nr::EPOLL_CTL, &[ep, 1, 999, 0x20300]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EBADF.0 as i64));

            // ftruncate: the size the kernel then reports.
            m.write(0x20400, b"/t\0").unwrap();
            set_args(&mut m, at, nr::OPENAT, &[(-100i64) as u64, 0x20400, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            k.syscall(&mut m, 0);
            let fd = ret(&m, at) as u64;
            set_args(&mut m, at, nr::FTRUNCATE, &[fd, 1234]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            assert_eq!(k.fds.get(fd as i32).unwrap().handle.metadata().unwrap().size, 1234);
            set_args(&mut m, at, nr::FTRUNCATE, &[999, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EBADF.0 as i64));
            // truncate: by name, shorter and longer; no such file, a
            // directory, a negative length.
            for len in [10u64, 5000] {
                set_args(&mut m, at, nr::TRUNCATE, &[0x20400, len]);
                k.syscall(&mut m, 0);
                assert_eq!(ret(&m, at), 0);
                assert_eq!(k.fds.get(fd as i32).unwrap().handle.metadata().unwrap().size, len);
            }
            m.write(0x20410, b"/nope\0").unwrap();
            set_args(&mut m, at, nr::TRUNCATE, &[0x20410, 0]);
            k.syscall(&mut m, 0);
            assert_eq!(ret(&m, at), -(Errno::ENOENT.0 as i64));
            m.write(0x20420, b"/\0").unwrap();
            set_args(&mut m, at, nr::TRUNCATE, &[0x20420, 0]);
            k.syscall(&mut m, 0);
            assert_eq!(ret(&m, at), -(Errno::EISDIR.0 as i64));
            set_args(&mut m, at, nr::TRUNCATE, &[0x20400, u64::MAX]);
            k.syscall(&mut m, 0);
            assert_eq!(ret(&m, at), -(Errno::EINVAL.0 as i64));
            for f in [r, w, ep, fd] {
                set_args(&mut m, at, nr::CLOSE, &[f]);
                k.syscall(&mut m, 0);
            }

            // statfs, fstatfs and sysinfo: the same bytes either way.
            m.zero(0x21000, 0x200).unwrap();
            set_args(&mut m, at, nr::FSTATFS, &[0, 0x21000]);
            call(&mut m, &mut k);
            assert_eq!(m.read_u64(0x21000).unwrap(), 0x0102_1994);
            m.write(0x21300, b"/\0").unwrap();
            set_args(&mut m, at, nr::STATFS, &[0x21300, 0x21100]);
            call(&mut m, &mut k);
            assert_eq!(m.read(0x21000, 120).unwrap(), m.read(0x21100, 120).unwrap());
            set_args(&mut m, at, nr::SYSINFO, &[0x21400]);
            call(&mut m, &mut k);
            assert_eq!(m.read_u64(0x21400 + 32).unwrap(), 4 << 30);
            // getpriority's nice 0 (20 - nice, as the raw call answers it).
            set_args(&mut m, at, nr::GETPRIORITY, &[0, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 20);
            set_args(&mut m, at, nr::SETPRIORITY, &[0, 0, 5]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);

            // prlimit64: a soft NOFILE set one way is read back the other
            // and is the descriptor table's; raising a hard limit unprivileged
            // is EPERM, soft over hard EINVAL.
            let soft = if lock_free { 300u64 } else { 400 };
            m.write_u64(0x22000, soft).unwrap();
            m.write_u64(0x22008, crate::file::NOFILE_HARD as u64).unwrap();
            set_args(&mut m, at, nr::PRLIMIT64, &[0, RLIMIT_NOFILE as u64, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            assert_eq!(k.fds.limit(), soft as usize);
            set_args(&mut m, at, nr::PRLIMIT64, &[0, RLIMIT_NOFILE as u64, 0, 0x22100]);
            if lock_free { let _ = k.syscall(&mut m, 0); } else { assert!(done(free(&k, &mut m, at))); }
            assert_eq!(m.read_u64(0x22100).unwrap(), soft);
            k.creds.set_uid(1000);
            m.write_u64(0x22000, 10).unwrap();
            m.write_u64(0x22008, 20).unwrap();
            set_args(&mut m, at, nr::PRLIMIT64, &[0, RLIMIT_CORE as u64, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), 0);
            m.write_u64(0x22008, 30).unwrap();
            set_args(&mut m, at, nr::PRLIMIT64, &[0, RLIMIT_CORE as u64, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EPERM.0 as i64));
            m.write_u64(0x22000, 25).unwrap();
            m.write_u64(0x22008, 20).unwrap();
            set_args(&mut m, at, nr::PRLIMIT64, &[0, RLIMIT_CORE as u64, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(ret(&m, at), -(Errno::EINVAL.0 as i64));
            k.creds.set_uid(0);
            k.rlimits.set(default_rlimits());
        }
    }

    #[test]
    fn free_own_state_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));
        const SIGUSR1: u64 = 10;

        // A handler set lock-free is the kernel's, and back.
        m.write_u64(0x20000, 0x1234).unwrap();
        m.write_u64(0x20008, 0x4000_0000).unwrap(); // SA_RESTART
        m.write_u64(0x20010, 0).unwrap();
        m.write_bits64(0x20018, 1 << 3).unwrap();
        set_args(&mut m, at, nr::RT_SIGACTION, &[SIGUSR1, 0x20000, 0, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 0);
        set_args(&mut m, at, nr::RT_SIGACTION, &[SIGUSR1, 0, 0x20100, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(m.read_u64(0x20100).unwrap(), 0x1234);
        assert_eq!(m.read_bits64(0x20118).unwrap(), 1 << 3);
        // SIGKILL's cannot be set, either way; the old one is read first
        // when both are the same struct.
        set_args(&mut m, at, nr::RT_SIGACTION, &[9, 0x20000, 0, 8]);
        free(&k, &mut m, at);
        assert_eq!(ret(&m, at), -(Errno::EINVAL.0 as i64));
        m.write_u64(0x20100, 0x5678).unwrap();
        set_args(&mut m, at, nr::RT_SIGACTION, &[SIGUSR1, 0x20100, 0x20100, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(m.read_u64(0x20100).unwrap(), 0x1234);
        assert_eq!(k.signals.action(SIGUSR1 as u32).handler, 0x5678);

        // The thread's alternate stack.
        m.write_u64(0x21000, 0x7000_0000).unwrap();
        m.write_u32(0x21008, 0).unwrap();
        m.write_u64(0x21010, 0x8000).unwrap();
        set_args(&mut m, at, nr::SIGALTSTACK, &[0x21000, 0]);
        assert!(done(free(&k, &mut m, at)));
        set_args(&mut m, at, nr::SIGALTSTACK, &[0, 0x21100]);
        k.syscall(&mut m, 0);
        assert_eq!(m.read_u64(0x21100).unwrap(), 0x7000_0000);
        assert_eq!(m.read_u64(0x21110).unwrap(), 0x8000);

        // A blocked signal raised is pending, read lock-free.
        k.signals.set_blocked(1 << (SIGUSR1 - 1));
        k.signals.raise_from(SIGUSR1 as u32, signal::Source { code: signal::SI_USER, pid: k.pid, uid: k.creds.uid() });
        set_args(&mut m, at, nr::RT_SIGPENDING, &[0x22000, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(m.read_bits64(0x22000).unwrap() & (1 << (SIGUSR1 - 1)), 1 << (SIGUSR1 - 1));

        // An interval timer armed lock-free is the kernel's to report, and
        // a kernel-armed one the free path's.
        for (i, v) in [0u64, 0, 100, 0].into_iter().enumerate() {
            m.write_u64(0x23000 + i as u64 * 8, v).unwrap();
        }
        set_args(&mut m, at, nr::SETITIMER, &[0, 0x23000, 0]);
        assert!(done(free(&k, &mut m, at)));
        set_args(&mut m, at, nr::GETITIMER, &[0, 0x23100]);
        k.syscall(&mut m, 0);
        let left = m.read_u64(0x23110).unwrap();
        assert!((98..=100).contains(&left), "{left} s left");
        assert!(k.timers_armed());
        for (i, v) in [0u64, 0, 0, 0].into_iter().enumerate() {
            m.write_u64(0x23000 + i as u64 * 8, v).unwrap();
        }
        set_args(&mut m, at, nr::SETITIMER, &[0, 0x23000, 0x23200]);
        k.syscall(&mut m, 0);
        assert!(!k.timers_armed());
        if gex_abi::X86 {
            // alarm: the seconds the old one had left, rounded up.
            set_args(&mut m, at, nr::ALARM, &[30]);
            assert!(done(free(&k, &mut m, at)));
            assert_eq!(ret(&m, at), 0);
            set_args(&mut m, at, nr::ALARM, &[0]);
            assert!(done(free(&k, &mut m, at)));
            assert_eq!(ret(&m, at), 30);
            assert!(!k.timers_armed());
        }

        // getrandom: the bytes /dev/urandom gives.
        set_args(&mut m, at, nr::GETRANDOM, &[0x24000, 64, 0]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 64);
        assert!(m.read(0x24000, 64).unwrap().iter().any(|&b| b != 0));
    }

    #[test]
    fn free_call_answers_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, at, nr::PIPE2, &[0x20000, 0]);
        k.syscall(&mut m, 0);
        let rfd = m.read_u32(0x20000).unwrap() as u64;
        let wfd = m.read_u32(0x20004).unwrap() as u64;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> Option<FreeOutcome> {
            let call = free_class(m, at, false)?;
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            Some(free_call(&f, m, at, call))
        }
        let done = |o: Option<FreeOutcome>| matches!(o, Some(FreeOutcome::Done(Delivery::None)));

        // A process call: declined before anything is done. An id call
        // is answered (`ident_call`), and an unknown one is ENOSYS.
        set_args(&mut m, at, nr::EXECVE, &[0, 0, 0]);
        assert!(free(&k, &mut m, at).is_none());
        set_args(&mut m, at, nr::GETPID, &[]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), k.pid as i64);
        set_args(&mut m, at, 999, &[]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), -(Errno::ENOSYS.0 as i64));

        // An empty pipe parks both ways, on the same stream.
        set_args(&mut m, at, nr::READ, &[rfd, 0x22000, 16]);
        assert!(matches!(free(&k, &mut m, at), Some(FreeOutcome::Blocking(BlockOp::ReadStream { len: 16, .. }, _))));
        set_args(&mut m, at, nr::READ, &[rfd, 0x22000, 16]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Blocking(BlockOp::ReadStream { len: 16, .. })));

        // Written lock-free, read by the kernel; and the other way round.
        m.write(0x21000, b"through").unwrap();
        set_args(&mut m, at, nr::WRITE, &[wfd, 0x21000, 7]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 7);
        set_args(&mut m, at, nr::READ, &[rfd, 0x22000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 7);
        assert_eq!(m.read(0x22000, 7).unwrap(), b"through");
        set_args(&mut m, at, nr::WRITE, &[wfd, 0x21000, 7]);
        k.syscall(&mut m, 0);
        set_args(&mut m, at, nr::READ, &[rfd, 0x23000, 16]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 7);
        assert_eq!(m.read(0x23000, 7).unwrap(), b"through");

        // Vectored, both ways: two vectors out, two in.
        m.write(0x24000, b"gather").unwrap();
        for (i, (base, len)) in [(0x24000u64, 3u64), (0x24003, 3)].into_iter().enumerate() {
            m.write_u64(0x25000 + i as u64 * 16, base).unwrap();
            m.write_u64(0x25008 + i as u64 * 16, len).unwrap();
        }
        for (i, (base, len)) in [(0x26000u64, 2u64), (0x26100, 8)].into_iter().enumerate() {
            m.write_u64(0x25100 + i as u64 * 16, base).unwrap();
            m.write_u64(0x25108 + i as u64 * 16, len).unwrap();
        }
        set_args(&mut m, at, nr::WRITEV, &[wfd, 0x25000, 2]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 6);
        set_args(&mut m, at, nr::READV, &[rfd, 0x25100, 2]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 6);
        assert_eq!(m.read(0x26000, 2).unwrap(), b"ga");
        assert_eq!(m.read(0x26100, 4).unwrap(), b"ther");

        // The poll family. An empty pipe parks, the call's mask in force
        // and the caller's to put back; a zero timeout answers at once, and
        // so does a ready pipe, with its events; either leaves the caller's
        // mask in force.
        let pollfd = 0x29000u64;
        m.write_u32(pollfd, rfd as u32).unwrap();
        m.write_u32(pollfd + 4, 1).unwrap(); // POLLIN
        m.write_bits64(0x29200, 0).unwrap(); // a mask that blocks nothing
        let ppoll = |m: &mut GuestMem<'_>, secs: u64| {
            m.write_u64(0x29100, secs).unwrap();
            m.write_u64(0x29108, 0).unwrap();
            set_args(m, at, nr::PPOLL, &[pollfd, 1, 0x29100, 0x29200, 8]);
        };
        let caller = 1u64 << 14;
        k.signals.set_blocked(caller);
        ppoll(&mut m, 5);
        match free(&k, &mut m, at) {
            Some(FreeOutcome::Blocking(BlockOp::Poll { deadline, .. }, restore)) => {
                assert_eq!(deadline, Some(std::time::Duration::from_secs(5)));
                assert_eq!(restore, Some(caller));
                assert_eq!(k.signals.blocked(), 0);
            }
            _ => panic!("an empty pipe's ppoll did not park"),
        }
        k.signals.set_blocked(caller);
        ppoll(&mut m, 0);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 0);
        assert_eq!(k.signals.blocked(), caller);
        set_args(&mut m, at, nr::WRITE, &[wfd, 0x21000, 1]);
        assert!(done(free(&k, &mut m, at)));
        ppoll(&mut m, 5);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 1);
        assert_eq!(m.read_u32(pollfd + 4).unwrap() >> 16, 1);
        assert_eq!(k.signals.blocked(), caller);
        k.signals.set_blocked(0);

        // epoll_pwait on a set holding the pipe: the caller's token back
        // when it is ready, a park when it is not.
        set_args(&mut m, at, nr::EPOLL_CREATE1, &[0]);
        k.syscall(&mut m, 0);
        let ep = ret(&m, at) as u64;
        let packed = gex_abi::X86;
        m.write_u32(0x29300, 1).unwrap();
        m.write_u64(if packed { 0x29304 } else { 0x29308 }, 0x77).unwrap();
        set_args(&mut m, at, nr::EPOLL_CTL, &[ep, 1, rfd, 0x29300]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 0);
        set_args(&mut m, at, nr::EPOLL_PWAIT, &[ep, 0x29400, 4, (-1i64) as u64, 0, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 1);
        assert_eq!(m.read_u64(if packed { 0x29404 } else { 0x29408 }).unwrap(), 0x77);
        set_args(&mut m, at, nr::READ, &[rfd, 0x22000, 16]);
        assert!(done(free(&k, &mut m, at)));
        set_args(&mut m, at, nr::EPOLL_PWAIT, &[ep, 0x29400, 4, 0, 0, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 0);
        set_args(&mut m, at, nr::EPOLL_PWAIT, &[ep, 0x29400, 4, 1000, 0, 8]);
        assert!(matches!(
            free(&k, &mut m, at),
            Some(FreeOutcome::Blocking(BlockOp::Poll { deadline: Some(_), .. }, None))
        ));

        // A file: positioned and seeking calls agree with the kernel's.
        let tmp = TmpFs::new();
        tmp.add_file("f", b"0123456789".to_vec());
        k.vfs.mount("/d", Arc::new(tmp));
        put_str(&mut m, 0x27000, "/d/f");
        set_args(&mut m, at, nr::OPENAT, &[(-100i64) as u64, 0x27000, 0, 0]);
        k.syscall(&mut m, 0);
        let fd = ret(&m, at) as u64;
        set_args(&mut m, at, nr::PREAD64, &[fd, 0x28000, 3, 4]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 3);
        assert_eq!(m.read(0x28000, 3).unwrap(), b"456");
        set_args(&mut m, at, nr::LSEEK, &[fd, 7, 0]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(ret(&m, at), 7);
        set_args(&mut m, at, nr::READ, &[fd, 0x28100, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 3);
        assert_eq!(m.read(0x28100, 3).unwrap(), b"789");

        // The mask: set lock-free, read back by the kernel's own call.
        m.write_u64(0x29000, 1 << (signal::sig::SIGPIPE - 1)).unwrap();
        set_args(&mut m, at, nr::RT_SIGPROCMASK, &[0, 0x29000, 0, 8]);
        assert!(done(free(&k, &mut m, at)));
        set_args(&mut m, at, nr::RT_SIGPROCMASK, &[0, 0, 0x29100, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(m.read_u64(0x29100).unwrap(), 1 << (signal::sig::SIGPIPE - 1));
        // SIGKILL cannot be blocked, here as there.
        m.write_u64(0x29000, 1 << (signal::sig::SIGKILL - 1)).unwrap();
        set_args(&mut m, at, nr::RT_SIGPROCMASK, &[2, 0x29000, 0x29100, 8]);
        assert!(done(free(&k, &mut m, at)));
        assert_eq!(m.read_u64(0x29100).unwrap(), 1 << (signal::sig::SIGPIPE - 1));
        assert_eq!(k.signals.blocked(), 0);

        // A write with no reader: EPIPE, and SIGPIPE's default ends the
        // process, as the full call's does.
        set_args(&mut m, at, nr::CLOSE, &[rfd]);
        k.syscall(&mut m, 0);
        set_args(&mut m, at, nr::WRITE, &[wfd, 0x21000, 7]);
        assert!(matches!(
            free(&k, &mut m, at),
            Some(FreeOutcome::Done(Delivery::Terminate { signal: signal::sig::SIGPIPE, .. }))
        ));
        set_args(&mut m, at, nr::WRITE, &[wfd, 0x21000, 7]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Killed { signal: signal::sig::SIGPIPE, .. }));
    }


    /// `nanosleep` and `clock_nanosleep` without the kernel: the same park
    /// or answer as the full call.
    #[test]
    fn free_sleep_answers_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        let req = 0x3000u64;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("sleeps are in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let r = |m: &GuestMem<'_>| ret(m, at) as i64;
        let ts = |m: &mut GuestMem<'_>, secs: u64, nanos: u64| {
            m.write_u64(req, secs).unwrap();
            m.write_u64(req + 8, nanos).unwrap();
        };
        // A plain sleep parks for its length, both ways.
        ts(&mut m, 0, 5_000_000);
        set_args(&mut m, at, nr::NANOSLEEP, &[req, 0]);
        let FreeOutcome::Blocking(BlockOp::Sleep(d), None) = free(&k, &mut m, at) else { panic!("a sleep parks") };
        assert_eq!(d, std::time::Duration::from_millis(5));
        assert!(matches!(k.blocking_op(&m, nr::NANOSLEEP, [req, 0, 0, 0, 0, 0]), Some(BlockOp::Sleep(e)) if e == d));
        // Zero: a park of nothing, both ways (a turn of the scheduler).
        ts(&mut m, 0, 0);
        set_args(&mut m, at, nr::NANOSLEEP, &[req, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Blocking(BlockOp::Sleep(d), None) if d.is_zero()));
        assert!(matches!(k.blocking_op(&m, nr::NANOSLEEP, [req, 0, 0, 0, 0, 0]), Some(BlockOp::Sleep(d)) if d.is_zero()));
        // Malformed: EINVAL; none: EFAULT.
        for (secs, nanos, want) in [(0, 1_000_000_000, Errno::EINVAL.as_return() as i64)] {
            ts(&mut m, secs, nanos);
            set_args(&mut m, at, nr::NANOSLEEP, &[req, 0]);
            assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
            assert_eq!(r(&m), want);
            set_args(&mut m, at, nr::NANOSLEEP, &[req, 0]);
            k.syscall(&mut m, 0);
            assert_eq!(r(&m), want);
        }
        set_args(&mut m, at, nr::NANOSLEEP, &[0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(r(&m), Errno::EFAULT.as_return() as i64);
        // An absolute deadline already past on the monotonic clock: 0 at
        // once; one ahead of it: a park.
        let now = clock_nanos_at(k.clock, k.started, 1);
        ts(&mut m, (now / 1_000_000_000) as u64 - 1, 0);
        set_args(&mut m, at, nr::CLOCK_NANOSLEEP, &[1, 1, req, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(r(&m), 0);
        ts(&mut m, (now / 1_000_000_000) as u64 + 2, 0);
        set_args(&mut m, at, nr::CLOCK_NANOSLEEP, &[1, 1, req, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Blocking(BlockOp::Sleep(_), None)));
    }

    /// `futex` in the lock-free class
    /// answers as the full call does: a wait on a changed word, on an
    /// unmapped word, a wait that parks, a wake, and a PI lock taken, found
    /// held (waiters bit set, then a park) and released. The map is asked
    /// through its published view, which follows `mmap` and `munmap`.
    #[test]
    fn free_futex_answers_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        let word = 0x30000u64;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("futex is in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let errno = |m: &GuestMem<'_>| ret(m, at) as i64;
        let wait = futex::op::WAIT | futex::op::PRIVATE_FLAG;
        // Not mapped yet: EFAULT, both ways.
        set_args(&mut m, at, nr::FUTEX, &[word, wait as u64, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(errno(&m), Errno::EFAULT.as_return() as i64);
        set_args(&mut m, at, nr::FUTEX, &[word, wait as u64, 0, 0]);
        k.syscall(&mut m, 0);
        assert_eq!(errno(&m), Errno::EFAULT.as_return() as i64);
        // Mapped: the view follows the map.
        k.mem.lock().insert(word, word + 0x1000, crate::mem::prot::READ | crate::mem::prot::WRITE, "futex");
        assert!(k.mem.view().mapped(word, 4));
        assert!(!k.mem.view().mapped(word + 0xffe, 4));
        m.write_u32(word, 7).unwrap();
        // A changed word: EAGAIN.
        set_args(&mut m, at, nr::FUTEX, &[word, wait as u64, 6, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(errno(&m), Errno::EAGAIN.as_return() as i64);
        // The word as expected: a park on it, which a wake ends.
        set_args(&mut m, at, nr::FUTEX, &[word, wait as u64, 7, 0]);
        let FreeOutcome::Blocking(BlockOp::FutexWait { parked: Some(parked), uaddr, .. }, None) = free(&k, &mut m, at) else {
            panic!("a wait on the expected word parks");
        };
        assert_eq!(uaddr, word);
        assert!(!parked.woken());
        set_args(&mut m, at, nr::FUTEX, &[word, futex::op::WAKE as u64, 1]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert!(parked.woken());
        assert_eq!(k.futexes.take_woken(), vec![word]);
        // PI: free, so taken; held by another, so the waiters bit and a park.
        let pi = word + 0x100;
        set_args(&mut m, at, nr::FUTEX, &[pi, futex::op::LOCK_PI as u64, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(m.read_u32(pi).unwrap(), k.tid as u32);
        set_args(&mut m, at, nr::FUTEX, &[pi, futex::op::UNLOCK_PI as u64, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(m.read_u32(pi).unwrap(), 0);
        m.write_u32(pi, 4242).unwrap();
        set_args(&mut m, at, nr::FUTEX, &[pi, futex::op::LOCK_PI as u64, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Blocking(BlockOp::FutexLockPi { .. }, None)));
        assert_eq!(m.read_u32(pi).unwrap(), 4242 | futex::pi::WAITERS);
        set_args(&mut m, at, nr::FUTEX, &[pi, futex::op::TRYLOCK_PI as u64, 0, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(errno(&m), Errno::EAGAIN.as_return() as i64);
        // Unmapped again: EFAULT.
        k.mem.lock().unmap(word, 0x1000).unwrap();
        assert!(!k.mem.view().mapped(word, 4));
        set_args(&mut m, at, nr::FUTEX, &[word, wait as u64, 7, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(errno(&m), Errno::EFAULT.as_return() as i64);
    }

    /// A map's view under writers and readers at once: readers on other
    /// threads, each in a reader slot of its own, never see a mapping that
    /// was never made, and always see the one that is never removed, while
    /// the owner maps and unmaps around it.
    #[test]
    fn a_map_view_is_read_while_the_map_changes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut map = crate::mem::MemoryMap::new(0x40_1000);
        let rw = crate::mem::prot::READ | crate::mem::prot::WRITE;
        map.insert(0x1000_0000, 0x1000_1000, rw, "fixed");
        let view = Arc::clone(map.view());
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|i| {
                let (view, stop) = (Arc::clone(&view), Arc::clone(&stop));
                std::thread::spawn(move || {
                    crate::sync::set_thread_id(i + 1);
                    let mut looks = 0u64;
                    while !stop.load(Ordering::Relaxed) || looks < 1000 {
                        assert!(view.mapped(0x1000_0000, 4), "the fixed mapping went missing");
                        assert!(!view.mapped(0x2000_0000, 4), "a mapping never made appeared");
                        looks += 1;
                    }
                })
            })
            .collect();
        for i in 0..2000u64 {
            let a = map.mmap(0, 0x10000 + (i % 4) * 0x1000, rw, crate::mem::flags::PRIVATE | crate::mem::flags::ANONYMOUS, "churn").unwrap();
            assert!(view.mapped(a, 4));
            map.protect(a, 0x1000, crate::mem::prot::READ).unwrap();
            map.unmap(a, 0x10000 + (i % 4) * 0x1000).unwrap();
            assert!(!view.mapped(a, 4));
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        // A copy (fork) has a view of its own.
        let mut child = map.clone();
        child.insert(0x3000_0000, 0x3000_1000, rw, "child");
        assert!(child.view().mapped(0x3000_0000, 4));
        assert!(!map.view().mapped(0x3000_0000, 4));
    }

    /// Futex calls from many threads at once, with no kernel lock between
    /// them: a mutex built on `FUTEX_WAIT`/`FUTEX_WAKE` over a shared word,
    /// each thread making the calls through `futex_block`/`futex_call` as
    /// a lock-free host does, keeps its count exact.
    #[test]
    fn futex_calls_from_many_threads_keep_a_mutex() {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
        let table = Arc::new(FutexTable::new());
        let word = Arc::new(AtomicU32::new(0));
        let count = Arc::new(AtomicU64::new(0));
        const N: u64 = 4;
        const ITERS: u64 = 3000;
        let threads: Vec<_> = (0..N)
            .map(|i| {
                let (table, word, count) = (Arc::clone(&table), Arc::clone(&word), Arc::clone(&count));
                std::thread::spawn(move || {
                    crate::sync::set_thread_id(i as u32 + 1);
                    let mapped = |_: u64, _: u64| true;
                    let cx = FutexCx { table: &table, map: None, mapped_by: Some(&mapped), tid: i as u32 + 1, clock: system_clock };
                    // The word lives in an AtomicU32 every thread shares; each
                    // call sees it through a copy of its own (the crate has no
                    // unsafe code to alias it), and a waiter also re-reads it,
                    // which is what a wake between the copy and the park
                    // would otherwise cost.
                    for _ in 0..ITERS {
                        // Drepper's mutex: 0 free, 1 held, 2 held with waiters.
                        let mut c = word.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).unwrap_or_else(|v| v);
                        if c != 0 {
                            if c != 2 {
                                c = word.swap(2, Ordering::Acquire);
                            }
                            while c != 0 {
                                let mut copy = word.load(Ordering::Relaxed).to_le_bytes().to_vec();
                                let mut m = GuestMem::new(&mut copy);
                                if let Some(BlockOp::FutexWait { parked: Some(p), .. }) =
                                    futex_block(&cx, &m, [0, futex::op::WAIT as u64, 2, 0, 0, 0])
                                {
                                    while !p.woken() && word.load(Ordering::Relaxed) == 2 {
                                        std::thread::yield_now();
                                    }
                                } else {
                                    let _ = futex_call(&cx, &mut m, [0, futex::op::WAIT as u64, 2, 0, 0, 0]);
                                }
                                c = word.swap(2, Ordering::Acquire);
                            }
                        }
                        let v = count.load(Ordering::Relaxed);
                        count.store(v + 1, Ordering::Relaxed);
                        if word.fetch_sub(1, Ordering::Release) != 1 {
                            word.store(0, Ordering::Release);
                            let mut copy = vec![0u8; 4];
                            let mut m = GuestMem::new(&mut copy);
                            futex_call(&cx, &mut m, [0, futex::op::WAKE as u64, 1, 0, 0, 0]).unwrap();
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), N * ITERS);
    }

    #[test]
    fn pipes_carry_data_between_descriptors() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::PIPE2, &[0x20000, 0]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);
        let rfd = m.read_u32(0x20000).unwrap() as u64;
        let wfd = m.read_u32(0x20004).unwrap() as u64;

        m.write(0x21000, b"through").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[wfd, 0x21000, 7]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);
        set_args(&mut m, k.ctx_addr, nr::READ, &[rfd, 0x22000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 7);
        assert_eq!(m.read(0x22000, 7).unwrap(), b"through");
    }

    #[test]
    fn getdents_lists_a_directory() {
        let (mut buf, mut k) = setup();
        let tmp = TmpFs::new();
        tmp.add_file("one", b"1".to_vec());
        tmp.add_file("two", b"2".to_vec());
        k.vfs.mount("/d", Arc::new(tmp));
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/d");
        set_args(
            &mut m,
            k.ctx_addr,
            nr::OPENAT,
            &[(-100i64) as u64, 0x20000, 0, 0],
        );
        k.syscall(&mut m, 0);
        let fd = ret(&m, k.ctx_addr) as u64;
        set_args(&mut m, k.ctx_addr, nr::GETDENTS64, &[fd, 0x21000, 512]);
        k.syscall(&mut m, 0);
        let n = ret(&m, k.ctx_addr);
        assert!(n > 0);
        let bytes = m.read(0x21000, n as usize).unwrap();
        let text = String::from_utf8_lossy(bytes);
        assert!(text.contains("one") && text.contains("two"));
        // A second call at the end of the directory returns zero.
        set_args(&mut m, k.ctx_addr, nr::GETDENTS64, &[fd, 0x21000, 512]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);
    }

    #[test]
    fn chdir_makes_relative_paths_resolve() {
        let (mut buf, mut k) = setup();
        k.vfs.mkdir("/work", 0o755).unwrap();
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/work");
        set_args(&mut m, k.ctx_addr, nr::CHDIR, &[0x20000]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);

        put_str(&mut m, 0x20100, "rel.txt");
        set_args(
            &mut m,
            k.ctx_addr,
            nr::OPENAT,
            &[
                (-100i64) as u64,
                0x20100,
                oflags::O_CREAT | oflags::O_RDWR,
                0,
            ],
        );
        k.syscall(&mut m, 0);
        assert!(ret(&m, k.ctx_addr) >= 3);
        assert!(k.vfs.metadata("/work/rel.txt").is_ok());
    }

    #[test]
    fn dev_null_and_zero_behave() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        put_str(&mut m, 0x20000, "/dev/zero");
        set_args(
            &mut m,
            k.ctx_addr,
            nr::OPENAT,
            &[(-100i64) as u64, 0x20000, 0, 0],
        );
        k.syscall(&mut m, 0);
        let fd = ret(&m, k.ctx_addr) as u64;
        m.write(0x21000, &[0xff; 8]).unwrap();
        set_args(&mut m, k.ctx_addr, nr::READ, &[fd, 0x21000, 8]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 8);
        assert_eq!(m.read(0x21000, 8).unwrap(), &[0u8; 8]);
    }

    #[test]
    fn unknown_syscalls_return_enosys_and_are_recorded() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, 9999, &[]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
        assert_eq!(ret(&m, k.ctx_addr), -38);
        assert_eq!(k.unimplemented, vec![9999]);
    }

    #[test]
    fn exit_group_reports_the_status() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::EXIT_GROUP, &[7]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Exit(7)));
    }

    #[test]
    fn faults_are_reported_rather_than_panicking() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[1, 0xffff_0000, 16]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), -14);
    }

    #[test]
    fn brk_round_trips() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::BRK, &[0]);
        k.syscall(&mut m, 0);
        let base = ret(&m, k.ctx_addr) as u64;
        set_args(&mut m, k.ctx_addr, nr::BRK, &[base + 0x2000]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr) as u64, base + 0x2000);
    }

    /// `recvmsg` honours its flags: `MSG_DONTWAIT` on an empty blocking
    /// socket is `EAGAIN`, not a wait (libwayland drains its socket this way
    /// and hung in its first round-trip when the flag was dropped), and
    /// `MSG_PEEK` leaves the bytes for the next read.
    #[test]
    fn recvmsg_flags() {
        const MSG_PEEK: u64 = 0x2;
        const MSG_DONTWAIT: u64 = 0x40;
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::SOCKETPAIR, &[1, 1, 0, 0x20000]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 0);
        let a = m.read_u32(0x20000).unwrap() as u64;
        let b = m.read_u32(0x20004).unwrap() as u64;
        // msghdr at 0x21000 with one iovec at 0x21100 -> 64 bytes at 0x22000.
        m.write_u64(0x21000 + 16, 0x21100).unwrap();
        m.write_u64(0x21000 + 24, 1).unwrap();
        m.write_u64(0x21100, 0x22000).unwrap();
        m.write_u64(0x21108, 64).unwrap();

        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_DONTWAIT]);
        assert!(matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue));
        assert_eq!(ret(&m, k.ctx_addr), -(Errno::EAGAIN.0 as i64));

        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, 0]);
        assert!(
            !matches!(k.syscall(&mut m, 0), SyscallOutcome::Continue),
            "a plain recvmsg on an empty blocking socket must wait"
        );
        k.retry_recvmsg = None;

        m.write(0x23000, b"hello").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[b, 0x23000, 5]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 5);
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_PEEK]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 5);
        assert_eq!(m.read(0x22000, 5).unwrap(), b"hello");
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_DONTWAIT]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 5, "peek must leave the bytes queued");
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_DONTWAIT]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), -(Errno::EAGAIN.0 as i64));
    }

    /// A vector whose first iovec the waiting bytes fill exactly: the call
    /// returns them, not EAGAIN from the empty second iovec (libwayland's
    /// wrapped ring buffer, which lost Wayland events); and a peek over two
    /// iovecs fills them in order rather than peeking the same bytes twice.
    #[test]
    fn recvmsg_iovecs() {
        const MSG_PEEK: u64 = 0x2;
        const MSG_DONTWAIT: u64 = 0x40;
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::SOCKETPAIR, &[1, 1, 0, 0x20000]);
        k.syscall(&mut m, 0);
        let a = m.read_u32(0x20000).unwrap() as u64;
        let b = m.read_u32(0x20004).unwrap() as u64;
        // Two iovecs: 4 bytes at 0x22000, 64 at 0x22100.
        m.write_u64(0x21000 + 16, 0x21100).unwrap();
        m.write_u64(0x21000 + 24, 2).unwrap();
        m.write_u64(0x21100, 0x22000).unwrap();
        m.write_u64(0x21108, 4).unwrap();
        m.write_u64(0x21110, 0x22100).unwrap();
        m.write_u64(0x21118, 64).unwrap();
        m.write(0x23000, b"abcdefgh").unwrap();
        set_args(&mut m, k.ctx_addr, nr::WRITE, &[b, 0x23000, 4]);
        k.syscall(&mut m, 0);
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_DONTWAIT]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 4, "the bytes that filled the first iovec");
        assert_eq!(m.read(0x22000, 4).unwrap(), b"abcd");

        set_args(&mut m, k.ctx_addr, nr::WRITE, &[b, 0x23000, 8]);
        k.syscall(&mut m, 0);
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_PEEK]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 8);
        assert_eq!(m.read(0x22000, 4).unwrap(), b"abcd");
        assert_eq!(m.read(0x22100, 4).unwrap(), b"efgh");
        set_args(&mut m, k.ctx_addr, nr::RECVMSG, &[a, 0x21000, MSG_DONTWAIT]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, k.ctx_addr), 8, "peek must leave the bytes queued");
        assert_eq!(m.read(0x22100, 4).unwrap(), b"efgh");
    }

    /// `sendmsg` and `recvmsg` in the lock-free class: bytes and a descriptor
    /// over a socketpair, a blocking
    /// socket's park, and `MSG_DONTWAIT`'s EAGAIN, as the full call answers.
    #[test]
    fn free_socket_messages_answer_as_the_kernel_does() {
        const MSG_DONTWAIT: u64 = 0x40;
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("socket messages are in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        set_args(&mut m, at, nr::SOCKETPAIR, &[1, 1, 0, 0x20000]);
        k.syscall(&mut m, 0);
        let a = m.read_u32(0x20000).unwrap() as u64;
        let b = m.read_u32(0x20004).unwrap() as u64;
        set_args(&mut m, at, nr::PIPE2, &[0x20010, 0]);
        k.syscall(&mut m, 0);
        let pipe_r = m.read_u32(0x20010).unwrap() as i32;
        // sendmsg(b): one iovec of 5 bytes and the pipe's read end.
        let (msg, iov, ctl) = (0x21000u64, 0x21100u64, 0x21200u64);
        m.write(0x23000, b"hello").unwrap();
        m.zero(msg, 56).unwrap();
        m.write_u64(msg + 16, iov).unwrap();
        m.write_u64(msg + 24, 1).unwrap();
        m.write_u64(iov, 0x23000).unwrap();
        m.write_u64(iov + 8, 5).unwrap();
        m.write_u64(msg + 32, ctl).unwrap();
        m.write_u64(msg + 40, 20).unwrap();
        m.write_u64(ctl, 20).unwrap();
        m.write_u32(ctl + 8, 1).unwrap();
        m.write_u32(ctl + 12, 1).unwrap();
        m.write_u32(ctl + 16, pipe_r as u32).unwrap();
        set_args(&mut m, at, nr::SENDMSG, &[b, msg, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(ret(&m, at), 5);
        // recvmsg(a): the bytes, and the descriptor installed here.
        let (rmsg, riov, rctl) = (0x24000u64, 0x24100u64, 0x24200u64);
        m.zero(rmsg, 56).unwrap();
        m.write_u64(rmsg + 16, riov).unwrap();
        m.write_u64(rmsg + 24, 1).unwrap();
        m.write_u64(riov, 0x25000).unwrap();
        m.write_u64(riov + 8, 64).unwrap();
        m.write_u64(rmsg + 32, rctl).unwrap();
        m.write_u64(rmsg + 40, 24).unwrap();
        set_args(&mut m, at, nr::RECVMSG, &[a, rmsg, MSG_DONTWAIT]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(ret(&m, at), 5);
        assert_eq!(m.read(0x25000, 5).unwrap(), b"hello");
        assert_eq!(m.read_u64(rmsg + 40).unwrap(), 20, "one descriptor's control message");
        let got = m.read_u32(rctl + 16).unwrap() as i32;
        assert_ne!(got, pipe_r);
        assert!(Arc::ptr_eq(&k.fds.get(got).unwrap().handle, &k.fds.get(pipe_r).unwrap().handle));
        // Empty: EAGAIN with MSG_DONTWAIT, both ways; a park without.
        set_args(&mut m, at, nr::RECVMSG, &[a, rmsg, MSG_DONTWAIT]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(ret(&m, at) as i64, Errno::EAGAIN.as_return() as i64);
        set_args(&mut m, at, nr::RECVMSG, &[a, rmsg, MSG_DONTWAIT]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at) as i64, Errno::EAGAIN.as_return() as i64);
        set_args(&mut m, at, nr::RECVMSG, &[a, rmsg, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Blocking(BlockOp::AcceptWait { .. }, None)));
    }

    /// `ioctl` in the lock-free class
    /// answers as the full call does: a pipe's bytes waiting, close-on-exec
    /// and non-blocking set from either path, a terminal request on a pipe,
    /// the terminal's own (whatever stdin is here) and a bad descriptor.
    #[test]
    fn free_ioctl_answers_as_the_kernel_does() {
        const FIONREAD: u64 = 0x541b;
        const FIONBIO: u64 = 0x5421;
        const FIOCLEX: u64 = 0x5451;
        const FIONCLEX: u64 = 0x5450;
        const TCGETS: u64 = 0x5401;
        const TIOCGWINSZ: u64 = 0x5413;
        const TIOCGPGRP: u64 = 0x540f;
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("ioctl is in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        set_args(&mut m, at, nr::PIPE2, &[0x20010, 0]);
        k.syscall(&mut m, 0);
        let pr = m.read_u32(0x20010).unwrap() as u64;
        let pw = m.read_u32(0x20014).unwrap() as u64;
        m.write(0x23000, b"abc").unwrap();
        set_args(&mut m, at, nr::WRITE, &[pw, 0x23000, 3]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 3);
        // The same request each way: the same result, the same bytes out.
        let both = |m: &mut GuestMem<'_>, k: &mut Kernel, fd: u64, req: u64, arg: u64| {
            m.zero(0x26000, 64).unwrap();
            m.zero(0x27000, 64).unwrap();
            m.write_u32(0x26000, 1).unwrap();
            m.write_u32(0x27000, 1).unwrap();
            set_args(m, at, nr::IOCTL, &[fd, req, if arg == 0 { 0x26000 } else { arg }]);
            assert!(matches!(free(k, m, at), FreeOutcome::Done(_)));
            let a = ret(m, at);
            set_args(m, at, nr::IOCTL, &[fd, req, if arg == 0 { 0x27000 } else { arg }]);
            k.syscall(m, 0);
            let b = ret(m, at);
            assert_eq!(a, b, "ioctl({fd}, {req:#x})");
            assert_eq!(m.read(0x26000, 64).unwrap().to_vec(), m.read(0x27000, 64).unwrap().to_vec());
            a
        };
        assert_eq!(both(&mut m, &mut k, pr, FIONREAD, 0), 0);
        assert_eq!(m.read_u32(0x26000).unwrap(), 3);
        assert_eq!(both(&mut m, &mut k, pr, TCGETS, 0) as i64, Errno::ENOTTY.as_return() as i64);
        assert_eq!(both(&mut m, &mut k, 99, FIONREAD, 0) as i64, Errno::EBADF.as_return() as i64);
        for req in [TCGETS, TIOCGWINSZ, TIOCGPGRP] {
            both(&mut m, &mut k, 0, req, 0);
        }
        // Flags set without the kernel are the descriptor's.
        set_args(&mut m, at, nr::IOCTL, &[pr, FIOCLEX, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_eq!(ret(&m, at), 0);
        set_args(&mut m, at, nr::FCNTL, &[pr, 1 /* F_GETFD */]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 1);
        set_args(&mut m, at, nr::IOCTL, &[pr, FIONCLEX, 0]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        set_args(&mut m, at, nr::FCNTL, &[pr, 1]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 0);
        m.write_u32(0x26000, 1).unwrap();
        set_args(&mut m, at, nr::IOCTL, &[pr, FIONBIO, 0x26000]);
        assert!(matches!(free(&k, &mut m, at), FreeOutcome::Done(_)));
        assert_ne!(k.fds.get(pr as i32).unwrap().flags() & oflags::O_NONBLOCK, 0);
    }

    /// The shared mappings' published view follows the list: what the page's
    /// compositor reads without the
    /// kernel to find a client's pool. A mapping added is seen, one dropped
    /// is gone, a reader's copy stays whole, and a forked copy's view is
    /// its own.
    #[test]
    fn the_shared_map_view_follows_the_list() {
        let file = |name: &str| OpenFile::new(Arc::new(crate::unix::MemFd::new(name)), name.into(), 0, true, true);
        let (a, b) = (file("a"), file("b"));
        let mut maps = SharedMaps::new();
        let view = Arc::clone(maps.view());
        assert!(view.of(&a).is_none());
        maps.push(SharedMap { start: 0x10000, len: 0x4000, file: Arc::clone(&a), offset: 0, holders: Vec::new(), pages: page_hashes(&[0; 0x4000]), changes: Default::default(), seen: Default::default() });
        maps.push(SharedMap { start: 0x20000, len: 0x1000, file: Arc::clone(&b), offset: 0, holders: Vec::new(), pages: page_hashes(&[0; 0x1000]), changes: Default::default(), seen: Default::default() });
        let held = view.get();
        assert_eq!(view.of(&a).map(|m| (m.start, m.len)), Some((0x10000, 0x4000)));
        assert_eq!(view.of(&b).map(|m| m.start), Some(0x20000));
        maps.retain(|m| m.start != 0x10000);
        assert!(view.of(&a).is_none(), "gone when dropped");
        assert!(view.of(&b).is_some());
        assert_eq!(held.len(), 2, "a reader's copy is whole");
        assert_eq!(maps.len(), 1);
    }

    /// A pool is where a mapping of its file holds all of it: a 4 KiB view
    /// of the file mapped before the pool's 3 MiB one is not it (Wine maps
    /// both, and the compositor took the first and refused the window's
    /// buffer as longer than its pool).
    #[test]
    fn a_pool_is_the_mapping_that_holds_it() {
        let file = |name: &str| OpenFile::new(Arc::new(crate::unix::MemFd::new(name)), name.into(), 0, true, true);
        let a = file("pool");
        let map = |start, len, offset| SharedMap { start, len, file: Arc::clone(&a), offset, holders: Vec::new(), pages: page_hashes(&[0; 0x1000]), changes: Default::default(), seen: Default::default() };
        let mut maps = SharedMaps::new();
        let view = Arc::clone(maps.view());
        maps.push(map(0xde56000, 0x1000, 0));
        assert_eq!(view.pool(&a, 0x30_0000).map(|m| m.start), Some(0xde56000), "too short, but the only one: the caller refuses it");
        maps.push(map(0x40000000, 0x30_0000, 0x1000));
        maps.push(map(0x22890000, 0x30_0000, 0));
        assert_eq!(view.pool(&a, 0x30_0000).map(|m| m.start), Some(0x22890000));
        assert_eq!(view.pool(&a, 0x1000).map(|m| m.start), Some(0x22890000), "the latest that holds it");
        assert!(view.pool(&file("other"), 0x1000).is_none());
    }

    /// i386's `_llseek`, `pread64` and `pwrite64` in the lock-free class
    /// answer as the full call does: the
    /// 64-bit offset from two registers, `_llseek`'s result through memory.
    /// The small descriptor and process calls in their i386 forms:
    /// `pipe`, `ftruncate64`'s two-word length, `fstatfs64`'s size argument,
    /// `ugetrlimit`'s 32-bit struct and the narrow `sysinfo`.
    #[test]
    fn free_i386_fd_misc_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/f\0").unwrap();
        set_args(&mut m, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        k.syscall(&mut m, 0);
        let fd = ret(&m, at) as u64;
        k.compat = true;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, true).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let set32 = |m: &mut GuestMem<'_>, n: u64, a: [u64; 5]| {
            m.write_u64(at + ctx::x(0) as u64, n).unwrap();
            for (i, v) in a.iter().enumerate() {
                m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
            }
        };
        let r32 = |m: &GuestMem<'_>| m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32;
        use crate::i386::raw;
        for lock_free in [true, false] {
            let call = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(matches!(free(k, m, at), FreeOutcome::Done(_)));
                } else {
                    k.syscall(m, 0);
                }
            };
            // pipe(fds): two descriptors.
            m.write_u64(0x21000, !0).unwrap();
            set32(&mut m, raw::PIPE, [0x21000, 0, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0, "lock_free {lock_free}");
            let (r, w) = (m.read_u32(0x21000).unwrap(), m.read_u32(0x21004).unwrap());
            assert!(r != w && r < 100 && w < 100);
            // ftruncate64(fd, lo, hi): the length from its two words (a high
            // word would make a 4 GiB file in memory, so it is zero here).
            set32(&mut m, raw::FTRUNCATE64, [fd, 0x12345, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0);
            assert_eq!(k.fds.get(fd as i32).unwrap().handle.metadata().unwrap().size, 0x12345);
            // fstatfs64(fd, 84, buf): the 32-bit f_type first.
            m.zero(0x22000, 84).unwrap();
            set32(&mut m, raw::FSTATFS64, [fd, 84, 0x22000, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0);
            assert_eq!(m.read_u32(0x22000).unwrap(), 0x0102_1994);
            // ugetrlimit(RLIMIT_STACK, &rlim32): 8 MiB soft, the hard limit
            // (infinite) as the 32-bit all-ones.
            m.write_u64(0x23000, 0).unwrap();
            set32(&mut m, raw::UGETRLIMIT, [RLIMIT_STACK as u64, 0x23000, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0);
            assert_eq!(m.read_u32(0x23000).unwrap(), 8 << 20);
            assert_eq!(m.read_u32(0x23004).unwrap(), u32::MAX);
            // sysinfo: the 64-byte layout, memory in pages.
            m.zero(0x24000, 128).unwrap();
            m.write_u64(0x24040, 0x1122_3344).unwrap();
            set32(&mut m, raw::SYSINFO, [0x24000, 0, 0, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0);
            assert_eq!(m.read_u32(0x24010).unwrap(), 1 << 20);
            assert_eq!(m.read_u32(0x24034).unwrap(), 4096);
            assert_eq!(m.read_u64(0x24040).unwrap(), 0x1122_3344, "past the 32-bit struct");
            for f in [r as u64, w as u64] {
                set32(&mut m, raw::CLOSE, [f, 0, 0, 0, 0]);
                k.syscall(&mut m, 0);
            }
        }
    }

    #[test]
    fn free_i386_positioned_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/f\0").unwrap();
        set_args(&mut m, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        k.syscall(&mut m, 0);
        let fd = ret(&m, at) as u64;
        m.write(0x21000, b"hello world").unwrap();
        set_args(&mut m, at, nr::WRITE, &[fd, 0x21000, 11]);
        k.syscall(&mut m, 0);
        assert_eq!(ret(&m, at), 11);
        k.compat = true;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, true).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        let set32 = |m: &mut GuestMem<'_>, n: u64, a: [u64; 5]| {
            m.write_u64(at + ctx::x(0) as u64, n).unwrap();
            for (i, v) in a.iter().enumerate() {
                m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
            }
        };
        let r32 = |m: &GuestMem<'_>| m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32;
        use crate::i386::raw;
        for lock_free in [true, false] {
            let call = |m: &mut GuestMem<'_>, k: &mut Kernel| {
                if lock_free {
                    assert!(matches!(free(k, m, at), FreeOutcome::Done(_)));
                } else {
                    k.syscall(m, 0);
                }
            };
            // _llseek(fd, 0, 6, &result, SEEK_SET): 0, and 6 at result.
            m.write_u64(0x22000, !0).unwrap();
            set32(&mut m, raw::LLSEEK, [fd, 0, 6, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 0, "lock_free {lock_free}");
            assert_eq!(m.read_u64(0x22000).unwrap(), 6);
            // pread64(fd, buf, 5, lo 6, hi 0): "world", the offset kept.
            set32(&mut m, raw::PREAD64, [fd, 0x23000, 5, 6, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 5);
            assert_eq!(m.read(0x23000, 5).unwrap(), b"world");
            // pwrite64 at 0, then the offset still 6 (_llseek SEEK_CUR 0).
            m.write(0x24000, b"J").unwrap();
            set32(&mut m, raw::PWRITE64, [fd, 0x24000, 1, 0, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), 1);
            set32(&mut m, raw::LLSEEK, [fd, 0, 0, 0x22000, 1]);
            call(&mut m, &mut k);
            assert_eq!(m.read_u64(0x22000).unwrap(), 6);
            // A bad descriptor: EBADF, with nothing written.
            m.write_u64(0x22000, 77).unwrap();
            set32(&mut m, raw::LLSEEK, [99, 0, 0, 0x22000, 0]);
            call(&mut m, &mut k);
            assert_eq!(r32(&m), -(Errno::EBADF.0 as i32));
            assert_eq!(m.read_u64(0x22000).unwrap(), 77);
        }
    }

    #[test]
    fn record_locks_contend_park_and_go_with_any_descriptor() {
        use crate::locks::{lk, LockReq, Owner};
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/lk\0").unwrap();
        let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
            set_args(m, at, n, a);
            let out = k.syscall(m, 0);
            (out, ret(m, at))
        };
        let open = |m: &mut GuestMem<'_>, k: &mut Kernel| sys(m, k, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]).1 as u64;
        let (fd, fd2) = (open(&mut m, &mut k), open(&mut m, &mut k));
        // struct flock at 0x25000: type, whence, start, len, pid.
        let flock = |m: &mut GuestMem<'_>, typ: u16, start: u64, len: u64| {
            m.write(0x25000, &[0u8; 32]).unwrap();
            m.write(0x25000, &typ.to_le_bytes()).unwrap();
            m.write_u64(0x25008, start).unwrap();
            m.write_u64(0x25010, len).unwrap();
        };
        let me = Owner::Pid(k.pid);
        flock(&mut m, lk::WRLCK as u16, 10, 10);
        assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[fd, 6, 0x25000]).1, 0, "F_SETLK");
        let locks = Arc::clone(&k.file_locks);
        assert_eq!(locks.set("/lk", Owner::Pid(999), 999, lk::RDLCK, 15, 16, false), Err(Errno::EAGAIN), "another process is kept out");
        assert_eq!(locks.set("/lk", Owner::Pid(999), 999, lk::WRLCK, 20, 30, false), Ok(0), "beside it is free");
        // F_GETLK names the other process's lock and pid.
        flock(&mut m, lk::WRLCK as u16, 0, 0);
        assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[fd, 5, 0x25000]).1, 0);
        assert_eq!(m.read(0x25000, 2).unwrap(), &(lk::WRLCK as u16).to_le_bytes());
        assert_eq!((m.read_u64(0x25008).unwrap(), m.read_u64(0x25010).unwrap(), m.read_u32(0x25018).unwrap()), (20, 10, 999));
        // F_SETLK in its way: EAGAIN; F_SETLKW: parks on the lock.
        flock(&mut m, lk::RDLCK as u16, 25, 1);
        assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[fd, 6, 0x25000]).1, -(Errno::EAGAIN.0 as i64));
        let (out, _) = sys(&mut m, &mut k, nr::FCNTL, &[fd, 7, 0x25000]);
        let SyscallOutcome::Blocking(BlockOp::FileLock { req, .. }) = out else { panic!("F_SETLKW parks: {out:?}") };
        assert_eq!(req, LockReq::Set { file: "/lk".into(), owner: me, pid: k.pid, typ: lk::RDLCK, start: 25, end: 26 });
        assert!(!locks.would_grant(&req));
        locks.set("/lk", Owner::Pid(999), 999, lk::UNLCK, 0, u64::MAX, false).unwrap();
        assert!(locks.would_grant(&req));
        assert_eq!(locks.apply(&req), Ok(0));
        // Closing the other descriptor of the file drops every lock the
        // process has on it.
        assert_eq!(locks.set("/lk", Owner::Pid(999), 999, lk::WRLCK, 0, u64::MAX, false), Err(Errno::EAGAIN));
        sys(&mut m, &mut k, nr::CLOSE, &[fd2]);
        assert_eq!(locks.set("/lk", Owner::Pid(999), 999, lk::WRLCK, 0, u64::MAX, false), Ok(0));
        // An OFD lock is the description's, pid -1, and needs l_pid 0.
        locks.set("/lk", Owner::Pid(999), 999, lk::UNLCK, 0, u64::MAX, false).unwrap();
        flock(&mut m, lk::WRLCK as u16, 0, 5);
        m.write_u32(0x25018, 1).unwrap();
        assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[fd, 37, 0x25000]).1, -(Errno::EINVAL.0 as i64));
        flock(&mut m, lk::WRLCK as u16, 0, 5);
        assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[fd, 37, 0x25000]).1, 0);
        assert_eq!(locks.get("/lk", Owner::Pid(999), lk::RDLCK, 0, 1).map(|c| c.pid), Some(-1));
        sys(&mut m, &mut k, nr::CLOSE, &[fd]);
        assert_eq!(locks.get("/lk", Owner::Pid(999), lk::WRLCK, 0, u64::MAX), None, "all gone with the last descriptor");
    }

    #[test]
    fn free_fd_table_calls_answer_as_the_kernel_does() {
        use crate::unix::lock_op;
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/f\0").unwrap();
        let open = |m: &mut GuestMem<'_>, k: &mut Kernel| {
            set_args(m, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            k.syscall(m, 0);
            ret(m, at) as u64
        };
        let fd = open(&mut m, &mut k);
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, false).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        const F_DUPFD_CLOEXEC: u64 = 1030;
        const F_GETFD: u64 = 1;
        const F_GETFL: u64 = 3;
        const F_SETFL: u64 = 4;
        const F_GETLK: u64 = 5;
        const F_ADD_SEALS: u64 = 1033;
        const F_GET_SEALS: u64 = 1034;
        let ebadf = -(Errno::EBADF.0 as i64);
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    assert!(matches!(free(k, m, at), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let mut got = Vec::new();
            let d1 = sys(&mut m, &mut k, nr::DUP, &[fd]);
            got.push(d1);
            got.push(sys(&mut m, &mut k, nr::DUP3, &[fd, 21, oflags::O_CLOEXEC]));
            got.push(sys(&mut m, &mut k, nr::DUP3, &[fd, fd, 0]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[21, F_GETFD]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[fd, F_DUPFD_CLOEXEC, 30]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[30, F_GETFD]));
            m.write(0x25000, &[0u8; 32]).unwrap();
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[fd, F_GETLK, 0x25000]));
            got.push(m.read(0x25000, 2).unwrap()[0] as i64);
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[fd, F_SETFL, oflags::O_NONBLOCK]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[d1 as u64, F_GETFL]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[fd, F_SETFL, 0]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[fd, F_GET_SEALS]));
            // A second description of the file contends for its lock until
            // the first's last descriptor closes; a copy's close is not that.
            let own = open(&mut m, &mut k);
            got.push(sys(&mut m, &mut k, nr::FLOCK, &[own, lock_op::EX]));
            let other = open(&mut m, &mut k);
            got.push(sys(&mut m, &mut k, nr::FLOCK, &[other, lock_op::EX | lock_op::NB]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[d1 as u64]));
            let copy = sys(&mut m, &mut k, nr::DUP, &[own]);
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[copy as u64]));
            got.push(sys(&mut m, &mut k, nr::FLOCK, &[other, lock_op::EX | lock_op::NB]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[21]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[30]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[30]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[own]));
            got.push(sys(&mut m, &mut k, nr::FLOCK, &[other, lock_op::EX | lock_op::NB]));
            got.push(sys(&mut m, &mut k, nr::FLOCK, &[other, lock_op::UN]));
            got.push(sys(&mut m, &mut k, nr::CLOSE_RANGE, &[40, 50, 0]));
            got.push(sys(&mut m, &mut k, nr::CLOSE_RANGE, &[5, 4, 0]));
            // `other` closed by its range.
            got.push(sys(&mut m, &mut k, nr::CLOSE_RANGE, &[other, other, 0]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[other, F_GETFD]));
            // A memfd's seals, found from its descriptor.
            m.write(0x26000, b"mem\0").unwrap();
            let mfd = sys_locked(&mut m, &mut k, at, nr::MEMFD_CREATE, &[0x26000, 0]);
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[mfd, F_ADD_SEALS, 2]));
            got.push(sys(&mut m, &mut k, nr::FCNTL, &[mfd, F_GET_SEALS]));
            got.push(sys(&mut m, &mut k, nr::CLOSE, &[mfd]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let g = &seen[0];
        assert_eq!(g[1], 21);
        assert_eq!(g[2], -(Errno::EINVAL.0 as i64));
        assert_eq!((g[3], g[4], g[5]), (1, 30, 1));
        assert_eq!((g[6], g[7]), (0, 2), "F_GETLK: F_UNLCK");
        assert_eq!(g[9] as u64 & oflags::O_NONBLOCK, oflags::O_NONBLOCK, "a copy shares the flags");
        assert_eq!(g[11], -(Errno::EINVAL.0 as i64), "no seals on a file");
        assert_eq!((g[12], g[13]), (0, -(Errno::EWOULDBLOCK.0 as i64)));
        assert_eq!(g[16], -(Errno::EWOULDBLOCK.0 as i64), "a copy's close keeps the lock");
        assert_eq!((g[17], g[18], g[19], g[20]), (0, 0, ebadf, 0));
        assert_eq!(g[21], 0, "the last close released it");
        assert_eq!((g[22], g[23], g[24]), (0, 0, -(Errno::EINVAL.0 as i64)));
        assert_eq!((g[25], g[26]), (0, ebadf), "closed by its range");
        assert_eq!((g[27], g[28], g[29]), (0, 2, 0));
    }

    #[test]
    fn a_tmpfile_name_goes_with_its_last_descriptor() {
        // Closed through the lock-free path, which has no kernel: the name
        // goes with the description (`OpenFile`'s `Drop`).
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, oflags::O_TMPFILE | oflags::O_RDWR, 0o600]);
        assert!((fd as i64) >= 0, "O_TMPFILE: {}", fd as i64);
        let name = k.fds.get(fd as i32).unwrap().path.clone();
        let copy = sys_locked(&mut m, &mut k, at, nr::DUP, &[fd]);
        let close = |m: &mut GuestMem<'_>, k: &Kernel, fd: u64| {
            set_args(m, at, nr::CLOSE, &[fd]);
            let call = free_class(m, at, false).expect("close is in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)));
            ret(m, at)
        };
        assert_eq!(close(&mut m, &k, fd), 0);
        assert!(k.vfs.metadata(&name).is_ok(), "a copy is still open");
        assert_eq!(close(&mut m, &k, copy), 0);
        assert!(k.vfs.metadata(&name).is_err(), "{name} outlived its last descriptor");
    }

    /// A leftover `.gex-tmpfile-PID-N` (a gex killed with the descriptor
    /// open; pids repeat) does not make the next `O_TMPFILE` fail with
    /// EEXIST: QTemporaryFile then gave up, and ksmserver with it.
    #[test]
    fn a_tmpfile_skips_a_name_left_behind() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        for n in 0..3 {
            let left = format!("/.gex-tmpfile-{}-{}\0", k.pid, n);
            m.write(0x20100, left.as_bytes()).unwrap();
            let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20100, oflags::O_CREAT | oflags::O_EXCL | oflags::O_RDWR, 0o600]);
            assert!((fd as i64) >= 0, "the leftover {left}");
        }
        m.write(0x20000, b"/\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, oflags::O_TMPFILE | oflags::O_RDWR, 0o600]);
        assert!((fd as i64) >= 0, "O_TMPFILE over leftovers: {}", fd as i64);
        assert!(k.fds.get(fd as i32).unwrap().path.ends_with("-3"), "the first free number");
    }

    /// A call made through the kernel, its result.
    fn sys_locked(m: &mut GuestMem<'_>, k: &mut Kernel, at: u64, n: u64, a: &[u64]) -> u64 {
        set_args(m, at, n, a);
        k.syscall(m, 0);
        ret(m, at) as u64
    }

    /// The path lookups and opens through the lock-free path ([`PathIo`])
    /// answer as the full call does: two kernels, the same calls, one
    /// through each path. Relative to the directory and to a descriptor,
    /// through a symlink, a stat of each kind, a listing, a missing name.
    #[test]
    fn free_path_calls_answer_as_the_kernel_does() {
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    let call = free_class(m, at, false).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let cwd = (-100i64) as u64;
            for (a, t) in [(0x20000u64, &b"/d\0"[..]), (0x20100, b"f\0"), (0x20200, b"/d/f\0"), (0x20300, b"/d/l\0"), (0x20400, b"/d/none\0"), (0x20500, b"l\0")] {
                m.write(a, t).unwrap();
            }
            sys_locked(&mut m, &mut k, at, nr::MKDIRAT, &[cwd, 0x20000, 0o755]);
            sys_locked(&mut m, &mut k, at, nr::SYMLINKAT, &[0x20200, cwd, 0x20300]);
            let mut got = Vec::new();
            let dir = sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20000, (oflags::O_RDONLY | oflags::O_DIRECTORY) as u64, 0]);
            got.push(dir);
            let f = sys(&mut m, &mut k, nr::OPENAT, &[dir as u64, 0x20100, (oflags::O_RDWR | oflags::O_CREAT | oflags::O_CLOEXEC) as u64, 0o640]);
            got.push(f);
            m.write(0x21000, b"twelve bytes").unwrap();
            got.push(sys(&mut m, &mut k, nr::WRITE, &[f as u64, 0x21000, 12]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[dir as u64, 0x20500, 0x22000, 0]));
            got.extend(m.read(0x22000, 128).unwrap().iter().map(|&b| b as i64));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[dir as u64, 0x20500, 0x22000, AT_SYMLINK_NOFOLLOW]));
            got.extend(m.read(0x22000, 128).unwrap().iter().map(|&b| b as i64));
            got.push(sys(&mut m, &mut k, nr::FSTAT, &[f as u64, 0x22000]));
            got.extend(m.read(0x22000, 128).unwrap().iter().map(|&b| b as i64));
            got.push(sys(&mut m, &mut k, nr::STATX, &[cwd, 0x20300, 0, 0x7ff, 0x23000]));
            got.extend(m.read(0x23000, 256).unwrap().iter().map(|&b| b as i64));
            // the mount's id, the same for the file and for the root
            // (systemd's chase compares them)
            let sx = m.read(0x23000, 256).unwrap();
            let mask = u32::from_le_bytes(sx[0..4].try_into().unwrap());
            let mnt = u64::from_le_bytes(sx[144..152].try_into().unwrap());
            assert!(mask & 0x1000 != 0 && mnt != 0, "statx mask {mask:#x}, mount {mnt}");
            m.write(0x20600, b"/\0").unwrap();
            assert_eq!(sys(&mut m, &mut k, nr::STATX, &[cwd, 0x20600, 0, 0x1000, 0x23000]), 0);
            let root = m.read(0x23000, 256).unwrap();
            assert_eq!(u64::from_le_bytes(root[144..152].try_into().unwrap()), mnt, "the root's mount");
            got.push(sys(&mut m, &mut k, nr::FACCESSAT, &[cwd, 0x20200, 4]));
            got.push(sys(&mut m, &mut k, nr::FACCESSAT, &[cwd, 0x20400, 0]));
            got.push(sys(&mut m, &mut k, nr::READLINKAT, &[dir as u64, 0x20500, 0x24000, 64]));
            got.extend(m.read(0x24000, 8).unwrap().iter().map(|&b| b as i64));
            got.push(sys(&mut m, &mut k, nr::GETCWD, &[0x25000, 64]));
            got.push(sys(&mut m, &mut k, nr::GETDENTS64, &[dir as u64, 0x26000, 4096]));
            got.push(sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20400, 0, 0]));
            got.push(k.fds.cloexec(f as i32).unwrap() as i64);
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let g = &seen[0];
        assert!(g[0] >= 3 && g[1] > g[0] && g[2] == 12, "{:?}", &g[..3]);
        assert_eq!(*g.last().unwrap(), 1, "O_CLOEXEC");
        assert_eq!(g[g.len() - 2], -(Errno::ENOENT.0 as i64));
    }

    /// Open flags are read in the guest's own spelling. An AArch64 program's
    /// `O_LARGEFILE` is x86's `O_NOFOLLOW`, and glibc passes it on every
    /// open: taken as x86's, opening anything through a symlink was
    /// `ELOOP`. And `fcntl(F_GETFL)` answers in that spelling too.
    #[test]
    fn open_flags_are_the_guests_spelling() {
        let g = |f: u64| oflags::to_guest(f, false);
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                match free_class(m, at, false) {
                    Some(call) if lock_free => {
                        let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                        let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                        assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                    }
                    _ => {
                        k.syscall(m, 0);
                    }
                }
                ret(m, at)
            };
            let cwd = (-100i64) as u64;
            for (a, t) in [(0x20000u64, &b"/d\0"[..]), (0x20200, b"/d/f\0"), (0x20300, b"/d/l\0")] {
                m.write(a, t).unwrap();
            }
            sys_locked(&mut m, &mut k, at, nr::MKDIRAT, &[cwd, 0x20000, 0o755]);
            let f = sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20200, g(oflags::O_RDWR | oflags::O_CREAT | oflags::O_LARGEFILE), 0o644]);
            assert!(f >= 3, "create: {f}");
            sys_locked(&mut m, &mut k, at, nr::SYMLINKAT, &[0x20200, cwd, 0x20300]);
            let via = sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20300, g(oflags::O_RDONLY | oflags::O_LARGEFILE), 0]);
            assert!(via >= 3, "open through the link with O_LARGEFILE (lock_free {lock_free}): {via}");
            let nofollow = sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20300, g(oflags::O_RDONLY | oflags::O_NOFOLLOW), 0]);
            assert_eq!(nofollow, -(Errno::ELOOP.0 as i64), "O_NOFOLLOW on the link");
            let dir = sys(&mut m, &mut k, nr::OPENAT, &[cwd, 0x20000, g(oflags::O_RDONLY | oflags::O_DIRECTORY), 0]);
            assert!(dir >= 3, "O_DIRECTORY: {dir}");
            for (name, what) in [(0x20200u64, "a file"), (0x20300, "a link to a file")] {
                let notdir = sys(&mut m, &mut k, nr::OPENAT, &[cwd, name, g(oflags::O_RDONLY | oflags::O_DIRECTORY), 0]);
                assert_eq!(notdir, -(Errno::ENOTDIR.0 as i64), "O_DIRECTORY on {what}");
                let path = sys(&mut m, &mut k, nr::OPENAT, &[cwd, name, g(oflags::O_PATH | oflags::O_DIRECTORY), 0]);
                assert_eq!(path, -(Errno::ENOTDIR.0 as i64), "O_PATH | O_DIRECTORY on {what}");
            }
            // O_DIRECT set and read back, in the guest's spelling.
            const F_GETFL: u64 = 3;
            const F_SETFL: u64 = 4;
            assert_eq!(sys(&mut m, &mut k, nr::FCNTL, &[f as u64, F_SETFL, g(oflags::O_DIRECT | oflags::O_NONBLOCK)]), 0);
            let fl = sys(&mut m, &mut k, nr::FCNTL, &[f as u64, F_GETFL]) as u64;
            assert_eq!(fl & g(oflags::O_DIRECT | oflags::O_NONBLOCK), g(oflags::O_DIRECT | oflags::O_NONBLOCK), "F_GETFL {fl:#o}");
            assert_eq!(fl & oflags::ACCMODE, oflags::O_RDWR);
            assert_eq!(fl & g(oflags::O_DIRECTORY | oflags::O_NOFOLLOW), 0, "F_GETFL {fl:#o}");
        }
    }

    /// Extended attributes (`user.` only) on the root tmpfs, through the
    /// lock-free path and the full call alike: kept by inode (through a
    /// rename and a hard link, and on a descriptor), the `l` forms on a
    /// symlink itself, and Linux's errors.
    #[test]
    fn xattrs_are_kept_by_inode() {
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    let call = free_class(m, at, false).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let cwd = (-100i64) as u64;
            let strs: [&[u8]; 9] = [b"/a\0", b"/b\0", b"/c\0", b"/l\0", b"/dir\0", b"user.DOSATTRIB\0", b"0x22", b"trusted.x\0", b"user.none\0"];
            let ptr = |i: usize| 0x20000u64 + 0x100 * i as u64;
            for (i, t) in strs.iter().enumerate() {
                m.write(ptr(i), t).unwrap();
            }
            let (a, b, c, l, dir, name, value, trusted, none) = (ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), ptr(7), ptr(8));
            let out = 0x24000u64;
            let mut got = Vec::new();
            let f = sys(&mut m, &mut k, nr::OPENAT, &[cwd, a, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, dir, 0o755]));
            got.push(sys(&mut m, &mut k, nr::SYMLINKAT, &[a, cwd, l]));
            got.push(sys(&mut m, &mut k, nr::SETXATTR, &[a, name, value, 4, 0]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[a, name, out, 0]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[a, name, out, 64]));
            got.extend(m.read(out, 4).unwrap().iter().map(|&x| x as i64));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[a, name, out, 2]));
            got.push(sys(&mut m, &mut k, nr::SETXATTR, &[a, name, value, 4, 1]));
            got.push(sys(&mut m, &mut k, nr::SETXATTR, &[a, none, value, 4, 2]));
            got.push(sys(&mut m, &mut k, nr::SETXATTR, &[a, trusted, value, 4, 0]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[a, none, out, 64]));
            // Through the symlink, and on the link itself.
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[l, name, out, 64]));
            got.push(sys(&mut m, &mut k, nr::LGETXATTR, &[l, name, out, 64]));
            got.push(sys(&mut m, &mut k, nr::LSETXATTR, &[l, name, value, 4, 0]));
            // The inode keeps them: renamed, linked, on a descriptor.
            got.push(sys(&mut m, &mut k, nr::RENAMEAT, &[cwd, a, cwd, b]));
            got.push(sys(&mut m, &mut k, nr::LINKAT, &[cwd, b, cwd, c, 0]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[c, name, out, 64]));
            got.push(sys(&mut m, &mut k, nr::FGETXATTR, &[f as u64, name, out, 64]));
            got.push(sys(&mut m, &mut k, nr::LISTXATTR, &[b, out, 64]));
            got.extend(m.read(out, 15).unwrap().iter().map(|&x| x as i64));
            let d = sys(&mut m, &mut k, nr::OPENAT, &[cwd, dir, (oflags::O_RDONLY | oflags::O_DIRECTORY) as u64, 0]);
            got.push(sys(&mut m, &mut k, nr::FSETXATTR, &[d as u64, name, value, 2, 0]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[dir, name, out, 0]));
            got.push(sys(&mut m, &mut k, nr::FREMOVEXATTR, &[f as u64, name]));
            got.push(sys(&mut m, &mut k, nr::GETXATTR, &[b, name, out, 64]));
            got.push(sys(&mut m, &mut k, nr::REMOVEXATTR, &[b, name]));
            got.push(sys(&mut m, &mut k, nr::LISTXATTR, &[c, out, 64]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let e = |x: Errno| -(x.0 as i64);
        let mut want = vec![0, 0, 0, 4, 4];
        want.extend(b"0x22".iter().map(|&x| x as i64));
        want.extend([e(Errno::ERANGE), e(Errno::EEXIST), e(Errno::ENODATA), e(Errno::EOPNOTSUPP), e(Errno::ENODATA)]);
        want.extend([4, e(Errno::ENODATA), e(Errno::EPERM)]);
        want.extend([0, 0, 4, 4, 15]);
        want.extend(b"user.DOSATTRIB\0".iter().map(|&x| x as i64));
        want.extend([0, 2, 0, e(Errno::ENODATA), e(Errno::ENODATA), 0]);
        assert_eq!(seen[0], want);
    }

    /// The path writers through the lock-free path ([`PathIo`])
    /// answer as the full call does. A directory renamed takes what is under
    /// it, a descriptor on a directory below follows it there (Linux
    /// resolves it by the directory, not its name), `RENAME_NOREPLACE`
    /// refuses a name that is taken, and a descriptor on a removed
    /// directory finds nothing.
    #[test]
    fn free_path_writers_answer_as_the_kernel_does() {
        const NOREPLACE: u64 = 1;
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    let call = free_class(m, at, false).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let cwd = (-100i64) as u64;
            let names: [&[u8]; 9] = [b"/d\0", b"/d/sub\0", b"/d/sub/f\0", b"/e\0", b"f\0", b"/e/sub/f\0", b"/d/l\0", b"/d/h\0", b"/x\0"];
            let ptr = |i: usize| 0x20000u64 + 0x100 * i as u64;
            for (i, t) in names.iter().enumerate() {
                m.write(ptr(i), t).unwrap();
            }
            let (d, sub, sub_f, e, f, e_sub_f, l, h, x) = (ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), ptr(7), ptr(8));
            let mut got = Vec::new();
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, d, 0o777]));
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, sub, 0o777]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, d, 0x22000, 0]));
            got.extend(m.read(0x22000, 128).unwrap().iter().map(|&b| b as i64));
            let file = sys(&mut m, &mut k, nr::OPENAT, &[cwd, sub_f, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            got.push(file);
            let dir = sys(&mut m, &mut k, nr::OPENAT, &[cwd, sub, (oflags::O_RDONLY | oflags::O_DIRECTORY) as u64, 0]);
            got.push(dir);
            // The directory and everything under it.
            got.push(sys(&mut m, &mut k, nr::RENAMEAT, &[cwd, d, cwd, e]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, sub_f, 0x22000, 0]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, e_sub_f, 0x22000, 0]));
            // The descriptor follows its directory.
            got.push(sys(&mut m, &mut k, nr::OPENAT, &[dir as u64, f, 0, 0]));
            got.push(sys(&mut m, &mut k, nr::RENAMEAT2, &[cwd, e, cwd, x, NOREPLACE]));
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, d, 0o755]));
            got.push(sys(&mut m, &mut k, nr::RENAMEAT2, &[cwd, x, cwd, d, NOREPLACE]));
            // Over an empty directory.
            got.push(sys(&mut m, &mut k, nr::RENAMEAT, &[cwd, x, cwd, d]));
            got.push(sys(&mut m, &mut k, nr::SYMLINKAT, &[sub_f, cwd, l]));
            got.push(sys(&mut m, &mut k, nr::LINKAT, &[cwd, sub_f, cwd, h, 0]));
            got.push(sys(&mut m, &mut k, nr::FCHMODAT, &[cwd, h, 0o600, 0]));
            got.push(sys(&mut m, &mut k, nr::FCHMODAT, &[cwd, l, 0o600, AT_SYMLINK_NOFOLLOW]));
            got.push(sys(&mut m, &mut k, nr::FCHMOD, &[file as u64, 0o640]));
            got.push(sys(&mut m, &mut k, nr::UTIMENSAT, &[cwd, h, 0, 0]));
            got.push(sys(&mut m, &mut k, nr::UTIMENSAT, &[cwd, x, 0, 0]));
            got.push(sys(&mut m, &mut k, nr::FCHOWNAT, &[cwd, l, 0, 0, AT_SYMLINK_NOFOLLOW]));
            got.push(sys(&mut m, &mut k, nr::FCHOWN, &[file as u64, 0, 0]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, l, 0]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, h, 0]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, sub, AT_REMOVEDIR]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[dir as u64, f, 0]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, sub, AT_REMOVEDIR]));
            // Its directory gone, the descriptor finds nothing.
            got.push(sys(&mut m, &mut k, nr::OPENAT, &[dir as u64, f, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]));
            got.push(sys(&mut m, &mut k, nr::GETDENTS64, &[dir as u64, 0x26000, 4096]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let g = &seen[0];
        let e = |x: Errno| -(x.0 as i64);
        assert_eq!(&g[..2], &[0, 0]);
        let tail = &g[2 + 1 + 128..];
        assert!(tail[0] >= 3 && tail[1] > tail[0], "{tail:?}");
        assert_eq!(
            &tail[2..],
            &[
                0,
                e(Errno::ENOENT),
                0,
                tail[5],
                0,
                0,
                e(Errno::EEXIST),
                0,
                0,
                0,
                0,
                e(Errno::EOPNOTSUPP),
                0,
                0,
                e(Errno::ENOENT),
                0,
                0,
                0,
                0,
                e(Errno::ENOTEMPTY),
                0,
                0,
                e(Errno::ENOENT),
                0,
            ]
        );
        assert!(tail[5] > tail[1], "an open through the moved directory: {}", tail[5]);
    }

    /// `chdir`, `fchdir` and `umask` through the lock-free path answer as
    /// the full call does, and a working directory is held by identity: it
    /// follows a rename of itself or of a directory above it, and once
    /// removed `getcwd` and relative lookups say `ENOENT`, as on Linux.
    #[test]
    fn a_working_directory_follows_its_renames() {
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    let call = free_class(m, at, false).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let cwd = (-100i64) as u64;
            let names: [&[u8]; 8] = [b"/a\0", b"/a/b\0", b"/a/b/s\0", b"/c\0", b"f\0", b"/c/b/f\0", b"/c/b\0", b"s\0"];
            let ptr = |i: usize| 0x20000u64 + 0x100 * i as u64;
            for (i, t) in names.iter().enumerate() {
                m.write(ptr(i), t).unwrap();
            }
            let (a, ab, abs_, c, f, cbf, cb, s) = (ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), ptr(7));
            let mut got = Vec::new();
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, a, 0o755]));
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, ab, 0o755]));
            got.push(sys(&mut m, &mut k, nr::UMASK, &[0o077]));
            got.push(sys(&mut m, &mut k, nr::MKDIRAT, &[cwd, abs_, 0o777]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, abs_, 0x22000, 0]));
            got.push(m.read_u32(0x22000 + if gex_abi::X86 { 24 } else { 16 }).unwrap() as i64 & 0o777);
            got.push(sys(&mut m, &mut k, nr::CHDIR, &[ab]));
            got.push(sys(&mut m, &mut k, nr::RENAMEAT, &[cwd, a, cwd, c]));
            got.push(sys(&mut m, &mut k, nr::GETCWD, &[0x25000, 64]));
            got.extend(m.read(0x25000, 6).unwrap().iter().map(|&b| b as i64));
            let file = sys(&mut m, &mut k, nr::OPENAT, &[cwd, f, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            got.push((file >= 3) as i64);
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, cbf, 0x22000, 0]));
            // fchdir to the root and back through a descriptor on /c/b.
            let dir = sys(&mut m, &mut k, nr::OPENAT, &[cwd, cb, (oflags::O_RDONLY | oflags::O_DIRECTORY) as u64, 0]);
            m.write(0x26000, b"/\0").unwrap();
            got.push(sys(&mut m, &mut k, nr::CHDIR, &[0x26000]));
            got.push(sys(&mut m, &mut k, nr::FCHDIR, &[dir as u64]));
            got.push(sys(&mut m, &mut k, nr::NEWFSTATAT, &[cwd, f, 0x22000, 0]));
            // Removed under it.
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, cbf, 0]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, s, AT_REMOVEDIR]));
            got.push(sys(&mut m, &mut k, nr::UNLINKAT, &[cwd, cb, AT_REMOVEDIR]));
            got.push(sys(&mut m, &mut k, nr::GETCWD, &[0x25000, 64]));
            got.push(sys(&mut m, &mut k, nr::OPENAT, &[cwd, f, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]));
            got.push(sys(&mut m, &mut k, nr::UMASK, &[0o022]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let e = |x: Errno| -(x.0 as i64);
        let mut want = vec![0, 0, 0o022, 0, 0, 0o700, 0, 0, 5];
        want.extend(b"/c/b\0\0".iter().map(|&b| b as i64));
        want.extend([1, 0, 0, 0, 0, 0, 0, 0, e(Errno::ENOENT), e(Errno::ENOENT), 0o077]);
        assert_eq!(seen[0], want);
    }

    /// The 32-bit ABI's path calls through the lock-free path, in their
    /// older forms too (`open`, `creat`, `stat64`, `lstat64`, `fstat64`,
    /// `access`, `readlink`): translated as the full call translates them,
    /// `AT_FDCWD` sign-extended.
    #[test]
    fn free_i386_path_calls_answer_as_the_kernel_does() {
        use crate::i386::raw;
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            k.compat = true;
            let mut buf = vec![0u8; 0x40000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: [u64; 6]| {
                m.write_u64(at + ctx::x(0) as u64, n).unwrap();
                for (i, v) in a.iter().enumerate() {
                    m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
                }
                if lock_free {
                    let call = free_class(m, at, true).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32 as i64
            };
            let cwd32 = 0xffff_ff9cu64;
            for (a, t) in [(0x20000u64, &b"/f\0"[..]), (0x20100, b"/l\0"), (0x20200, b"/none\0"), (0x20300, b"/g\0")] {
                m.write(a, t).unwrap();
            }
            let mut got = Vec::new();
            let f = sys(&mut m, &mut k, raw::OPEN, [0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o600, 0, 0, 0]);
            got.push(f);
            got.push(sys(&mut m, &mut k, raw::CREAT, [0x20300, 0o600, 0, 0, 0, 0]));
            m.write(0x21000, b"abc").unwrap();
            got.push(sys(&mut m, &mut k, raw::WRITE, [f as u64, 0x21000, 3, 0, 0, 0]));
            sys_locked(&mut m, &mut k, at, nr::SYMLINKAT, &[0x20000, (-100i64) as u64, 0x20100]);
            for (n, a) in [
                (raw::STAT64, [0x20100u64, 0x22000, 0, 0, 0, 0]),
                (raw::LSTAT64, [0x20100, 0x22000, 0, 0, 0, 0]),
                (raw::FSTAT64, [f as u64, 0x22000, 0, 0, 0, 0]),
                (raw::FSTATAT64, [cwd32, 0x20000, 0x22000, 0, 0, 0]),
            ] {
                got.push(sys(&mut m, &mut k, n, a));
                got.extend(m.read(0x22000, 96).unwrap().iter().map(|&b| b as i64));
            }
            got.push(sys(&mut m, &mut k, raw::ACCESS, [0x20000, 4, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::ACCESS, [0x20200, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::FACCESSAT, [cwd32, 0x20000, 4, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::READLINK, [0x20100, 0x24000, 64, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::READLINKAT, [cwd32, 0x20100, 0x24000, 64, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::GETCWD, [0x25000, 64, 0, 0, 0, 0]));
            // The writers' older forms.
            for (p, t) in [(0x20400u64, &b"/dd\0"[..]), (0x20500, b"/dd/f\0"), (0x20600, b"/ee\0"), (0x20700, b"/ee/f\0"), (0x20800, b"/ee/h\0"), (0x20900, b"/ee/s\0")] {
                m.write(p, t).unwrap();
            }
            got.push(sys(&mut m, &mut k, raw::MKDIR, [0x20400, 0o755, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::RENAME, [0x20000, 0x20500, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::RENAME, [0x20400, 0x20600, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::LINK, [0x20700, 0x20800, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::SYMLINK, [0x20700, 0x20900, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::CHMOD, [0x20800, 0o600, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::FCHMOD, [f as u64, 0o600, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::CHOWN, [0x20800, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::LCHOWN, [0x20900, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::FCHOWN, [f as u64, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::UTIMENSAT, [cwd32, 0x20800, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::UNLINK, [0x20900, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::UNLINK, [0x20800, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::RMDIR, [0x20600, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::UNLINK, [0x20700, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::RMDIR, [0x20600, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::OPENAT, [cwd32, 0x20200, 0, 0, 0, 0]));
            // The own-state calls in their 32-bit layouts: a handler
            // set and read back, an interval timer armed and its old value
            // narrowed back (musl's `alarm`), `getitimer`, `alarm`.
            for (o, v) in [(0u64, 0x1234u32), (4, 0), (8, 0)] {
                m.write_u32(0x26000 + o, v).unwrap();
            }
            m.write_bits64(0x2600c, 1 << 3).unwrap();
            got.push(sys(&mut m, &mut k, raw::RT_SIGACTION, [10, 0x26000, 0, 8, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::RT_SIGACTION, [10, 0, 0x26100, 8, 0, 0]));
            got.push(m.read_u32(0x26100).unwrap() as i64);
            for (o, v) in [(0u64, 0u32), (4, 0), (8, 50), (12, 0)] {
                m.write_u32(0x26200 + o, v).unwrap();
            }
            got.push(sys(&mut m, &mut k, raw::SETITIMER, [0, 0x26200, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::GETITIMER, [0, 0x26300, 0, 0, 0, 0]));
            got.push((48..=50).contains(&m.read_u32(0x26308).unwrap()) as i64);
            m.write(0x26400, &[0u8; 16]).unwrap();
            got.push(sys(&mut m, &mut k, raw::SETITIMER, [0, 0x26400, 0x26500, 0, 0, 0]));
            got.push((48..=50).contains(&m.read_u32(0x26508).unwrap()) as i64);
            got.push(sys(&mut m, &mut k, raw::ALARM, [7, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::ALARM, [0, 0, 0, 0, 0, 0]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let g = &seen[0];
        assert!(g[0] >= 3 && g[1] > g[0] && g[2] == 3, "{:?}", &g[..3]);
        let n = g.len();
        assert_eq!(&g[n - 27..n - 10], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -(Errno::ENOTEMPTY.0 as i64), 0, 0, -(Errno::ENOENT.0 as i64)]);
        assert_eq!(&g[n - 10..], &[0, 0, 0x1234, 0, 0, 1, 0, 1, 0, 7]);
    }

    /// A file mapped shared and written, then mapped again: a second shared
    /// mapping is the first's pages, and a private one (Wine's write-copy
    /// and read-only section views) copies what the shared one holds, not
    /// the file as it was before the writes.
    #[test]
    fn a_new_mapping_reads_what_a_shared_one_wrote() {
        use crate::mem::{flags as mf, prot};
        let (_, mut k) = setup();
        let mut buf = vec![0u8; 0x0708_0000];
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        let rw = (prot::READ | prot::WRITE) as u64;
        m.write(0x20000, b"/s\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        m.write(0x21000, &[1u8; 0x2000]).unwrap();
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::PWRITE64, &[fd, 0x21000, 0x2000, 0]), 0x2000);
        let a = sys_locked(&mut m, &mut k, at, nr::MMAP, &[0x07010000, 0x2000, rw, mf::SHARED | mf::FIXED, fd, 0]);
        assert_eq!(a, 0x07010000);
        m.write(0x07011004, b"new").unwrap();
        let b = sys_locked(&mut m, &mut k, at, nr::MMAP, &[0, 0x1000, prot::READ as u64, mf::SHARED, fd, 0x1000]);
        assert_eq!(b, 0x07011000, "a second shared mapping is the first's pages");
        let c = sys_locked(&mut m, &mut k, at, nr::MMAP, &[0x07020000, 0x2000, rw, mf::PRIVATE | mf::FIXED, fd, 0]);
        assert_eq!(c, 0x07020000);
        assert_eq!(m.read(0x07021004, 3).unwrap(), b"new", "the private copy has the shared mapping's writes");
        m.write(0x07021004, b"own").unwrap();
        assert_eq!(m.read(0x07011004, 3).unwrap(), b"new", "and its own writes stay its own");
    }

    /// An exec leaves no shared mapping behind. One that stayed on the list
    /// was the old program's: the new one's mapping of the same file was
    /// handed the old address (and saw whatever the new image had there), and
    /// its exit wrote that memory into the file.
    #[test]
    fn an_exec_leaves_no_shared_mapping_behind() {
        use crate::mem::{flags as mf, prot};
        let (_, mut k) = setup();
        let mut buf = vec![0u8; 0x0708_0000];
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        let rw = (prot::READ | prot::WRITE) as u64;
        m.write(0x20000, b"/x\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        m.write(0x21000, b"data").unwrap();
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::PWRITE64, &[fd, 0x21000, 4, 0]), 4);
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::MMAP, &[0x07010000, 0x1000, rw, mf::SHARED | mf::FIXED, fd, 0]), 0x07010000);
        assert_eq!(k.mem.lock().log.shared_maps.len(), 1);
        k.mem.lock().renew(0);
        assert_eq!(k.mem.lock().log.shared_maps.len(), 0);
        // The new image's memory there, and its exit: the file keeps its bytes.
        m.write(0x07010000, &[0u8; 4]).unwrap();
        k.map_io().flush_all_shared(&m);
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::PREAD64, &[fd, 0x22000, 4, 0]), 4);
        assert_eq!(m.read(0x22000, 4).unwrap(), b"data");
    }

    /// A shared mapping unmapped and mapped again: the file unchanged, the
    /// page hashes come back rather than being taken again; changed by a
    /// write through another descriptor while unmapped, they do not, and a
    /// store that puts a page back as it was before that write still
    /// reaches the file (kept hashes would call it clean).
    #[test]
    fn a_remapped_file_takes_its_kept_hashes_only_while_unchanged() {
        use crate::mem::{flags as mf, prot};
        let (_, mut k) = setup();
        let mut buf = vec![0u8; 0x0708_0000];
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        let rw = (prot::READ | prot::WRITE) as u64;
        let len = RECALL_MIN;
        m.write(0x20000, b"/r\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        let other = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, oflags::O_RDWR as u64, 0]);
        m.write(0x21000, &vec![b'A'; len as usize]).unwrap();
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::PWRITE64, &[fd, 0x21000, len, 0]), len);
        let map = |m: &mut GuestMem<'_>, k: &mut Kernel| sys_locked(m, k, at, nr::MMAP, &[0x07010000, len, rw, mf::SHARED | mf::FIXED, fd, 0]);
        let unmap = |m: &mut GuestMem<'_>, k: &mut Kernel| sys_locked(m, k, at, nr::MUNMAP, &[0x07010000, len]);
        let at_file = |m: &mut GuestMem<'_>, k: &mut Kernel, off: u64| {
            assert_eq!(sys_locked(m, k, at, nr::PREAD64, &[fd, 0x22000, 1, off]), 1);
            m.read(0x22000, 1).unwrap()[0]
        };
        assert_eq!(map(&mut m, &mut k), 0x07010000);
        m.write(0x07010000, b"P").unwrap();
        assert_eq!(unmap(&mut m, &mut k), 0);
        assert_eq!(at_file(&mut m, &mut k, 0), b'P');
        assert_eq!(k.mem.lock().log.shared_maps.recent.len(), 1, "the unmapped range's hashes are kept");
        assert_eq!(map(&mut m, &mut k), 0x07010000);
        assert!(k.mem.lock().log.shared_maps.recent.is_empty(), "and taken back by the next mapping");
        assert_ne!(k.mem.lock().log.shared_maps[0].changes.load(std::sync::atomic::Ordering::Relaxed), 0);
        m.write(0x07011000, b"Q").unwrap();
        assert_eq!(unmap(&mut m, &mut k), 0);
        assert_eq!(at_file(&mut m, &mut k, 0x1000), b'Q', "a store after taking them back is written back");
        m.write(0x23000, b"B").unwrap();
        assert_eq!(sys_locked(&mut m, &mut k, at, nr::PWRITE64, &[other, 0x23000, 1, 0x2000]), 1);
        assert_eq!(map(&mut m, &mut k), 0x07010000);
        assert_eq!(m.read(0x07012000, 1).unwrap(), b"B", "the other descriptor's write is seen");
        m.write(0x07012000, b"A").unwrap();
        assert_eq!(unmap(&mut m, &mut k), 0);
        assert_eq!(at_file(&mut m, &mut k, 0x2000), b'A', "the page put back as it was is written back");
    }

    /// The mapping calls through the lock-free path ([`MapIo`]) answer as
    /// the full call does, and leave the same notes for the host: two
    /// kernels, the same calls, one through each path.
    #[test]
    fn free_mapping_calls_answer_as_the_kernel_does() {
        use crate::mem::{flags as mf, prot};
        let anon = mf::PRIVATE | mf::ANONYMOUS | mf::FIXED;
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            // Above the runtime's regions, where a guest's mappings go.
            let mut buf = vec![0u8; 0x0708_0000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: &[u64]| {
                set_args(m, at, n, a);
                if lock_free {
                    let call = free_class(m, at, false).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                ret(m, at)
            };
            let rw = (prot::READ | prot::WRITE) as u64;
            let mut got = Vec::new();
            got.push(sys(&mut m, &mut k, nr::MMAP, &[0x07000000, 0x4000, rw, anon, !0, 0]));
            m.write(0x07000000, b"dirty").unwrap();
            got.push(sys(&mut m, &mut k, nr::MADVISE, &[0x07000000, 0x1000, 4]));
            got.push(m.read(0x07000000, 1).unwrap()[0] as i64);
            got.push(sys(&mut m, &mut k, nr::MPROTECT, &[0x07001000, 0x1000, (prot::READ | prot::EXEC) as u64]));
            got.push(sys(&mut m, &mut k, nr::MPROTECT, &[0x07001000, 0x1000, rw]));
            got.push(sys(&mut m, &mut k, nr::MREMAP, &[0x07003000, 0x1000, 0x3000, 0]));
            // A shared file mapping reaches the file at msync; an executable
            // one is code for the host.
            m.write(0x20000, b"/m\0").unwrap();
            let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
            m.write(0x21000, &[7u8; 0x2000]).unwrap();
            got.push(sys(&mut m, &mut k, nr::PWRITE64, &[fd, 0x21000, 0x2000, 0]));
            got.push(sys(&mut m, &mut k, nr::MMAP, &[0x07010000, 0x2000, rw, mf::SHARED | mf::FIXED, fd, 0]));
            m.write(0x07010010, b"shared").unwrap();
            got.push(sys(&mut m, &mut k, nr::MSYNC, &[0x07010000, 0x2000, 4]));
            got.push(sys(&mut m, &mut k, nr::PREAD64, &[fd, 0x22000, 6, 0x10]));
            got.push((m.read(0x22000, 6).unwrap() == b"shared") as i64);
            got.push(sys(&mut m, &mut k, nr::MMAP, &[0x07014000, 0x1000, (prot::READ | prot::EXEC) as u64, mf::PRIVATE | mf::FIXED, fd, 0]));
            got.push(sys(&mut m, &mut k, nr::MUNMAP, &[0x07010000, 0x2000]));
            got.push(sys(&mut m, &mut k, nr::MUNMAP, &[0x07014000, 0x1000]));
            let base = sys(&mut m, &mut k, nr::BRK, &[0]);
            got.push(sys(&mut m, &mut k, nr::BRK, &[base as u64 + 0x2000]) - base);
            let mm = k.mem.lock();
            let log = &mm.log;
            got.push(log.recheck_code.len() as i64);
            got.extend(log.retired_code.iter().flat_map(|&(a, b)| [a as i64, b as i64]));
            got.extend(log.new_code.iter().map(|c| c.start as i64));
            got.push(log.shared_maps.len() as i64);
            got.push(mm.vmas().len() as i64);
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let g = &seen[0];
        assert_eq!(g[0], 0x07000000);
        assert_eq!((g[1], g[2]), (0, 0), "MADV_DONTNEED zeroes");
        assert_eq!((g[3], g[4]), (0, 0));
        assert_eq!((g[7], g[8], g[9], g[10]), (0x07010000, 0, 6, 1), "msync wrote the file");
        assert_eq!((g[11], g[12], g[13]), (0x07014000, 0, 0));
        assert_eq!(g[14], 0x2000);
        assert_eq!(g[15], 1, "made executable again: a recheck");
        assert_eq!(&g[16..20], &[0x07001000, 0x07002000, 0x07014000, 0x07015000], "retired: made writable, unmapped");
        assert_eq!((g[20], g[21]), (0x07014000, 0), "code mapped; no shared mapping left");
    }

    /// The 32-bit ABI's mapping calls through the lock-free path: `mmap2`'s
    /// page offset and `-1` descriptor, the old `mmap`'s argument block,
    /// addresses above 2 GiB not sign-extended.
    #[test]
    fn free_i386_mapping_calls_answer_as_the_kernel_does() {
        use crate::i386::raw;
        use crate::mem::{flags as mf, prot};
        let anon = mf::PRIVATE | mf::ANONYMOUS | mf::FIXED;
        let rw = (prot::READ | prot::WRITE) as u64;
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let (_, mut k) = setup();
            k.compat = true;
            let mut buf = vec![0u8; 0x0708_0000];
            let at = k.ctx_addr;
            let mut m = GuestMem::new(&mut buf);
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: [u64; 6]| {
                m.write_u64(at + ctx::x(0) as u64, n).unwrap();
                for (i, v) in a.iter().enumerate() {
                    m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
                }
                if lock_free {
                    let call = free_class(m, at, true).expect("in the class");
                    let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
                    let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
                    assert!(matches!(free_call(&f, m, at, call), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32 as i64
            };
            let mut got = Vec::new();
            got.push(sys(&mut m, &mut k, raw::MMAP2, [0x0700_0000, 0x2000, rw, anon, 0xffff_ffff, 0]));
            // The old form: six words at a pointer.
            for (i, v) in [0x0700_4000u32, 0x1000, rw as u32, anon as u32, 0xffff_ffff, 0].iter().enumerate() {
                m.write_u32(0x30000 + 4 * i as u64, *v).unwrap();
            }
            got.push(sys(&mut m, &mut k, raw::MMAP, [0x30000, 0, 0, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::MPROTECT, [0x0700_1000, 0x1000, prot::READ as u64, 0, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::MUNMAP, [0x0700_0000, 0x1000, 0, 0, 0, 0]));
            let base = sys(&mut m, &mut k, raw::BRK, [0; 6]);
            got.push(sys(&mut m, &mut k, raw::BRK, [base as u64 + 0x1000, 0, 0, 0, 0, 0]) - base);
            got.push(k.mem.lock().vmas().len() as i64);
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        assert_eq!(&seen[0][..5], &[0x0700_0000, 0x0700_4000, 0, 0, 0x1000]);
    }

    #[test]
    fn free_i386_fd_table_calls_answer_as_the_kernel_does() {
        let (mut buf, mut k) = setup();
        let at = k.ctx_addr;
        let mut m = GuestMem::new(&mut buf);
        m.write(0x20000, b"/f\0").unwrap();
        let fd = sys_locked(&mut m, &mut k, at, nr::OPENAT, &[(-100i64) as u64, 0x20000, (oflags::O_RDWR | oflags::O_CREAT) as u64, 0o644]);
        k.compat = true;
        fn free(k: &Kernel, m: &mut GuestMem<'_>, at: u64) -> FreeOutcome {
            let call = free_class(m, at, true).expect("in the class");
            let futex = FutexCx { table: &k.futexes, map: Some(k.mem.view()), mapped_by: None, tid: k.tid as u32, clock: k.clock };
            let f = Free { io: k.fd_io(), map: k.map_io(), path: k.path_io(), signals: &k.signals, procs: &k.procs, live: &k.live_threads, slots: &k.next_ctx_slot, tty: &k.tty, pid: k.pid, creds: &k.creds, threads: &k.thread_signals, wasm32: false, futex, started: k.started, rlimits: &k.rlimits, net: k.net_io(), thread_fs: &k.thread_fs };
            free_call(&f, m, at, call)
        }
        use crate::i386::raw;
        let mut seen = Vec::new();
        for lock_free in [true, false] {
            let sys = |m: &mut GuestMem<'_>, k: &mut Kernel, n: u64, a: [u64; 3]| {
                m.write_u64(at + ctx::x(0) as u64, n).unwrap();
                for (i, v) in a.iter().enumerate() {
                    m.write_u64(at + crate::i386::ARG_SLOTS[i] as u64, *v).unwrap();
                }
                if lock_free {
                    assert!(matches!(free(k, m, at), FreeOutcome::Done(_)), "{n} in the class");
                } else {
                    k.syscall(m, 0);
                }
                m.read_u64(at + ctx::x(0) as u64).unwrap() as u32 as i32
            };
            let mut got = Vec::new();
            got.push(sys(&mut m, &mut k, raw::DUP2, [fd, 20, 0]));
            got.push(sys(&mut m, &mut k, raw::DUP2, [fd, fd, 0]));
            got.push(sys(&mut m, &mut k, raw::DUP3, [fd, 21, oflags::O_CLOEXEC]));
            got.push(sys(&mut m, &mut k, raw::FCNTL64, [21, 1, 0]));
            // F_GETLK64 (12) is the 32-bit ABI's F_GETLK.
            m.write(0x25000, &[0u8; 32]).unwrap();
            got.push(sys(&mut m, &mut k, raw::FCNTL64, [fd, 12, 0x25000]));
            got.push(m.read(0x25000, 2).unwrap()[0] as i32);
            got.push(sys(&mut m, &mut k, raw::FLOCK, [fd, 2, 0]));
            got.push(sys(&mut m, &mut k, raw::FLOCK, [fd, 8, 0]));
            got.push(sys(&mut m, &mut k, raw::CLOSE, [20, 0, 0]));
            got.push(sys(&mut m, &mut k, raw::CLOSE, [20, 0, 0]));
            // close_range's bounds are unsigned: ~0 is "to the end".
            got.push(sys(&mut m, &mut k, raw::CLOSE_RANGE, [21, 0xffff_ffff, 0]));
            got.push(sys(&mut m, &mut k, raw::FCNTL, [21, 1, 0]));
            got.push(sys(&mut m, &mut k, raw::DUP, [0xffff_ffff, 0, 0]));
            seen.push(got);
        }
        assert_eq!(seen[0], seen[1]);
        let ebadf = -(Errno::EBADF.0 as i32);
        assert_eq!(seen[0], vec![20, fd as i32, 21, 1, 0, 2, 0, 0, 0, ebadf, 0, ebadf, ebadf]);
    }

    #[test]
    fn uname_reports_the_guest_arch() {
        let (mut buf, mut k) = setup();
        let mut m = GuestMem::new(&mut buf);
        set_args(&mut m, k.ctx_addr, nr::UNAME, &[0x20000]);
        k.syscall(&mut m, 0);
        let arch = gex_abi::ARCH_NAME.as_bytes();
        assert_eq!(m.read(0x20000 + 65 * 4, arch.len()).unwrap(), arch);
    }
}

/// The top of the code space of 64-bit Wine lowered to a wasm32 memory: its
/// images, executable allocations and mappings go below it (Wine's
/// virtual.c `w64_code_space_limit`).
pub const W64_CODE_TOP: u64 = 0x7fff_0000;

/// Whether `head` starts an x86-64 PE32+ image's headers.
fn pe_x64(head: &[u8]) -> bool {
    let u16_at = |o: usize| head.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let nt = head.get(0x3c..0x40).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    head.get(..2) == Some(b"MZ")
        && nt.is_some_and(|nt| head.get(nt..nt + 4) == Some(b"PE\0\0") && u16_at(nt + 4) == Some(0x8664) && u16_at(nt + 24) == Some(0x20b))
}

/// Whether the image whose headers start `head` is an x86-64 Gecko
/// program's (Firefox, Thunderbird): `beside` says its directory has
/// mozglue.dll.
pub fn pe_is_gecko(head: &[u8], beside: impl Fn(&str) -> bool) -> bool {
    pe_x64(head) && beside("mozglue.dll")
}

/// Gecko's switches that start its child processes without Chromium's
/// sandbox, each added to `envp` unless it already sets it (so a user
/// can still ask for the sandbox). Wine enforces no token or job, so the
/// sandbox protects nothing here; and to set it up, the parent patches
/// each child's ntdll through Chromium's interceptions, which accept only
/// the Windows system call stub at an Nt export, not the PE skeleton's
/// stub (`skel`), so it killed every child it started.
pub fn gecko_without_sandbox(envp: &mut Vec<String>) {
    for kind in ["CONTENT", "GPU", "RDD", "SOCKET_PROCESS", "GMP", "UTILITY"] {
        let name = format!("MOZ_DISABLE_{kind}_SANDBOX=");
        if !envp.iter().any(|e| e.starts_with(&name)) {
            envp.push(format!("{name}1"));
        }
    }
}

/// Whether the PE image whose headers start `head` is an x86-64 one that
/// only a 64-bit memory can load: it has no relocations (stripped, or no
/// relocation directory), so it goes at its preferred base alone, and that
/// base puts it above [`W64_CODE_TOP`] (a linker's default x86-64 base,
/// 0x140000000, is). Or it is a Gecko program's (Firefox, Thunderbird:
/// `beside` says its directory has mozglue.dll), whose JavaScript engine
/// reserves 2 GiB of executable memory at start, more than the lowered
/// memory's whole code space. Or it is a Java launcher's (java.exe and
/// its kin: jvm.dll in its directory's `server` or `client`, or beside
/// it), whose VM reserves its heap, a 1 GiB compressed class space and its
/// code cache at start, more than the lowered memory has. Anything else,
/// or headers that do not parse: false.
pub fn pe_needs_mem64(head: &[u8], beside: impl Fn(&str) -> bool) -> bool {
    let u16_at = |o: usize| head.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |o: usize| head.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let u64_at = |o: usize| head.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let parse = || -> Option<bool> {
        if head.get(..2)? != b"MZ" {
            return None;
        }
        let nt = u32_at(0x3c)? as usize;
        if head.get(nt..nt + 4)? != b"PE\0\0" || u16_at(nt + 4)? != 0x8664 {
            return None;
        }
        let chars = u16_at(nt + 22)?;
        let opt = nt + 24;
        if u16_at(opt)? != 0x20b {
            return None;
        }
        let (base, size) = (u64_at(opt + 24)?, u32_at(opt + 56)? as u64);
        let dirs = u32_at(opt + 108)?;
        let relocs = if dirs > 5 { u32_at(opt + 112 + 5 * 8 + 4)? } else { 0 };
        let fixed = chars & 1 != 0 || relocs == 0;
        let gecko = pe_is_gecko(head, &beside);
        let java = ["server/jvm.dll", "client/jvm.dll", "jvm.dll"].into_iter().any(&beside);
        Some((fixed && base.saturating_add(size) > W64_CODE_TOP) || gecko || java)
    };
    parse().unwrap_or(false)
}

#[cfg(test)]
mod pe_mem64_tests {
    use super::pe_needs_mem64;

    /// A minimal PE32+ header: `base`, `chars`, the relocation directory's size.
    fn pe(base: u64, chars: u16, relocs: u32, machine: u16) -> Vec<u8> {
        let mut b = vec![0u8; 512];
        b[..2].copy_from_slice(b"MZ");
        b[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        b[0x80..0x84].copy_from_slice(b"PE\0\0");
        b[0x84..0x86].copy_from_slice(&machine.to_le_bytes());
        b[0x96..0x98].copy_from_slice(&chars.to_le_bytes());
        let opt = 0x98;
        b[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        b[opt + 24..opt + 32].copy_from_slice(&base.to_le_bytes());
        b[opt + 56..opt + 60].copy_from_slice(&0x10000u32.to_le_bytes());
        b[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes());
        b[opt + 112 + 44..opt + 112 + 48].copy_from_slice(&relocs.to_le_bytes());
        b
    }

    #[test]
    fn fixed_high_base_needs_mem64() {
        assert!(pe_needs_mem64(&pe(0x1_4000_0000, 0x23, 0, 0x8664), |_| false));
        assert!(pe_needs_mem64(&pe(0x1_4000_0000, 0x22, 0, 0x8664), |_| false));
        assert!(pe_needs_mem64(&pe(0x1_4000_0000, 0x22, 0x200, 0x8664), |n| n == "mozglue.dll"));
        // A Java launcher: the VM in bin\server\ (a JDK's or a JRE's), or beside it.
        assert!(pe_needs_mem64(&pe(0x1_4000_0000, 0x22, 0x200, 0x8664), |n| n == "server/jvm.dll"));
        assert!(pe_needs_mem64(&pe(0x1_4000_0000, 0x22, 0x200, 0x8664), |n| n == "jvm.dll"));
    }

    #[test]
    fn gecko_children_start_without_the_sandbox() {
        use super::{gecko_without_sandbox, pe_is_gecko};
        let exe = pe(0x1_4000_0000, 0x22, 0x200, 0x8664);
        assert!(pe_is_gecko(&exe, |n| n == "mozglue.dll"));
        assert!(!pe_is_gecko(&exe, |_| false));
        assert!(!pe_is_gecko(&pe(0x40_0000, 0x22, 0x200, 0x14c), |n| n == "mozglue.dll"));
        let mut env = vec!["PATH=/bin".to_string(), "MOZ_DISABLE_GPU_SANDBOX=0".to_string()];
        gecko_without_sandbox(&mut env);
        assert!(env.contains(&"MOZ_DISABLE_CONTENT_SANDBOX=1".to_string()));
        assert!(env.contains(&"MOZ_DISABLE_UTILITY_SANDBOX=1".to_string()));
        // the user's own setting stays
        assert_eq!(env.iter().filter(|e| e.starts_with("MOZ_DISABLE_GPU_SANDBOX=")).collect::<Vec<_>>(), ["MOZ_DISABLE_GPU_SANDBOX=0"]);
        assert_eq!(env.len(), 7);
    }

    #[test]
    fn relocatable_or_low_does_not() {
        let none = |_: &str| false;
        assert!(!pe_needs_mem64(&pe(0x1_4000_0000, 0x22, 0x200, 0x8664), none));
        assert!(!pe_needs_mem64(&pe(0x40_0000, 0x23, 0, 0x8664), none));
        assert!(!pe_needs_mem64(&pe(0x1_4000_0000, 0x23, 0, 0x14c), none));
        assert!(!pe_needs_mem64(&pe(0x40_0000, 0x22, 0x200, 0x14c), |n| n == "mozglue.dll"));
        assert!(!pe_needs_mem64(&pe(0x40_0000, 0x22, 0x200, 0x14c), |n| n == "server/jvm.dll"));
        assert!(!pe_needs_mem64(b"MZ", none));
        assert!(!pe_needs_mem64(b"\x7fELF", none));
    }
}
