//! The contract every gex component agrees on: the layout of the thread
//! context, the guest address space and the host import surface, in one place.
//! Nothing in this crate depends on anything else in gex.

#![forbid(unsafe_code)]
#![no_std]

// ---------------------------------------------------------------------------
// Guest architecture
// ---------------------------------------------------------------------------

/// True when this build targets an x86-64 guest, false for AArch64. Chosen by
/// the `x86` cargo feature: the two guests differ in constants the emitter,
/// kernel and shims bake into code (instruction alignment, syscall registers),
/// so one binary serves one guest.
pub const X86: bool = cfg!(feature = "x86");

/// The guest architecture's name as Linux spells it (`uname -m`, `AT_PLATFORM`).
pub const ARCH_NAME: &str = if X86 { "x86_64" } else { "aarch64" };

/// The ELF `e_machine` of a guest object.
pub const ELF_MACHINE: u16 = if X86 { 62 } else { 183 };


/// Log2 of the guest's instruction alignment: A64 instructions are 4-byte
/// aligned words, x86-64 instructions start on any byte.
pub const INST_ALIGN_SHIFT: u32 = if X86 { 0 } else { 2 };

/// The guest's instruction alignment in bytes.
pub const INST_ALIGN: u64 = 1 << INST_ALIGN_SHIFT;

/// `log2` of the dispatch slot granularity: a slot covers this many bytes of
/// guest code. AArch64 instructions are 4-byte aligned, so a slot per
/// instruction. x86-64 instructions start at any byte, but a slot per byte made
/// the dispatch tables (one copy per thread Worker) four times the size and
/// broke the browser's memory budget; a slot per 4 bytes instead, with an
/// in-module trampoline wherever two blocks start in one slot (see
/// `ctx::LAST_TARGET`, which every dispatch records so the trampoline can tell).
pub const DISPATCH_SHIFT: u32 = 2;

/// Length in bytes of the guest's system-call instruction (`svc #0` is a
/// 4-byte word; `syscall` is `0f 05`). Where a syscall resumes is the
/// instruction after it, and a host that restarts or skips one needs this.
pub const SYSCALL_INST_BYTES: u64 = if X86 { 2 } else { 4 };

// ---------------------------------------------------------------------------
// Guest address space
// ---------------------------------------------------------------------------

/// Wasm page size.
pub const WASM_PAGE: u64 = 65536;

/// Size of the guest's linear memory, in bytes: one page short of 4 GiB. The
/// missing top page makes wild pointers trappable — 32-bit-overflowing
/// addresses are steered to [`POISON_ADDR`] there, which faults as SIGSEGV.
pub const GUEST_MEM_BYTES: u64 = 0xFFFF_0000;

/// Number of wasm pages in [`GUEST_MEM_BYTES`].
pub const GUEST_MEM_PAGES: u64 = GUEST_MEM_BYTES / WASM_PAGE;

/// What a process's memory is *allocated* with, as opposed to what it may grow
/// to. Everything below [`MMAP_BOTTOM`] is the fixed layout a process needs
/// before it runs; above it, memory is grown on demand. Paid per process, so
/// this number decides whether a demo fits on someone else's machine.
pub const GUEST_MEM_INITIAL_BYTES: u64 = MMAP_BOTTOM;

/// Number of wasm pages in [`GUEST_MEM_INITIAL_BYTES`].
pub const GUEST_MEM_INITIAL_PAGES: u64 = GUEST_MEM_INITIAL_BYTES / WASM_PAGE;

/// Highest guest address that can ever be valid.
pub const GUEST_LIMIT: u64 = GUEST_MEM_BYTES;

/// Which guest memory a process has. `M32` is the default
/// and every program that fits keeps it: a 4 GiB wasm32 memory with gex's
/// bounds guard. `M64` is a 64-bit (memory64) memory of [`LIMIT64`] bytes for
/// a 64-bit guest that needs more address space: the same layout below
/// 4 GiB, and non-executable data in the high arena above it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemFlavor {
    #[default]
    M32,
    M64,
}

impl MemFlavor {
    /// The highest address a guest of this flavour can use, exclusive.
    pub const fn limit(self) -> u64 {
        match self {
            MemFlavor::M32 => GUEST_LIMIT,
            MemFlavor::M64 => LIMIT64,
        }
    }

    pub const fn is_64(self) -> bool {
        matches!(self, MemFlavor::M64)
    }
}

static MEM64: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Set the guest memory every translation in this host process targets.
/// Natively a gexc process's guests share one flavour; a
/// browser shim's is fixed by its build. Set before anything is translated.
pub fn set_mem_flavor(f: MemFlavor) {
    MEM64.store(f.is_64() || cfg!(target_arch = "wasm64"), core::sync::atomic::Ordering::Relaxed);
}

/// The guest memory translations target: [`MemFlavor::M32`] unless
/// [`set_mem_flavor`] said otherwise, or this is a wasm64 build (whose own
/// memory is the guest's, so it can only emit for a 64-bit one).
pub fn mem_flavor() -> MemFlavor {
    if cfg!(target_arch = "wasm64") || MEM64.load(core::sync::atomic::Ordering::Relaxed) {
        MemFlavor::M64
    } else {
        MemFlavor::M32
    }
}

/// `mem_flavor().is_64()`.
pub fn mem64() -> bool {
    mem_flavor().is_64()
}

/// The size of an [`MemFlavor::M64`] memory: 16 GiB, V8's memory64 cap
/// (262144 pages), used natively too so that every module declares the same
/// maximum and runs on either host.
pub const LIMIT64: u64 = 16 << 30;

/// Wasm pages in [`LIMIT64`].
pub const LIMIT64_PAGES: u64 = LIMIT64 / WASM_PAGE;

/// An M64 process's dense per-page tables: its dispatch
/// directory, one `u32` for every page of `[0, LIMIT64)`, so code can live
/// anywhere in the memory (a W^X JIT's high pages made executable), and its
/// [`PATCHED_PAGES`] bits after it. At 8 GiB, so that `[4 GiB, 8 GiB)`
/// holds nothing of gex's when a process starts (the high arena's data
/// comes later, and in 64-bit Wine from the top down), and a Windows
/// image keeps the base it was linked at there -- Free Pascal's x86-64
/// exe at 0x100000000 (no relocations: at 4 GiB, where these tables were,
/// such an exe could not start), MSVC's and MinGW's at 0x140000000, a
/// DLL's at 0x180000000. An M32 process uses the low region's directory
/// instead; see [`dispatch_dir`].
pub const DISPATCH_DIR64: u64 = 8 << 30;

/// The bottom of an M64 memory's high arena: the guest's data above
/// everything a wasm32 process has, filled bottom-up to [`LIMIT64`] around
/// the M64 tables, and in 64-bit Wine top-down from its top
/// ([`DISPATCH_DIR64`] by default), so that its images at their own low
/// bases (0x100000000, 0x140000000) are reached by the data last.
pub const IMAGE_ZONE64: u64 = 1 << 32;

/// Pages [`DISPATCH_DIR64`] covers: all of [`LIMIT64`].
pub const DISPATCH_MAX_PAGES64: u64 = LIMIT64 / DISPATCH_PAGE;

/// The M64 [`PATCHED_PAGES`]: one bit per page of [`DISPATCH_MAX_PAGES64`].
pub const PATCHED_PAGES64: u64 = DISPATCH_DIR64 + DISPATCH_MAX_PAGES64 * 4;

/// End of the M64 tables, rounded to 1 MiB.
pub const HIGH_TABLES_END: u64 = (PATCHED_PAGES64 + DISPATCH_MAX_PAGES64 / 8 + 0xf_ffff) & !0xf_ffff;

/// The bottom of the part of an M64 process's high arena ([`IMAGE_ZONE64`])
/// above the M64 tables: what the memory grows over as that part is used.
pub const HIGH_ARENA_BASE: u64 = HIGH_TABLES_END;

/// The dispatch directory of this process's memory flavour: the low region's
/// [`DISPATCH_DIR`] for M32, [`DISPATCH_DIR64`] for M64. Also [`page_gen`].
#[inline]
pub fn dispatch_dir() -> u64 {
    if mem64() { DISPATCH_DIR64 } else { DISPATCH_DIR }
}

/// Pages [`dispatch_dir`] covers.
#[inline]
pub fn dispatch_max_pages() -> u64 {
    if mem64() { DISPATCH_MAX_PAGES64 } else { DISPATCH_MAX_PAGES }
}

/// The [`PAGE_GEN`] words of this flavour (the directory, as for M32).
#[inline]
pub fn page_gen() -> u64 {
    dispatch_dir()
}

/// The [`PATCHED_PAGES`] bits of this flavour.
#[inline]
pub fn patched_pages() -> u64 {
    if mem64() { PATCHED_PAGES64 } else { PATCHED_PAGES }
}

/// The directory's byte range `[dispatch_dir(), end)`.
#[inline]
pub fn dispatch_dir_range() -> (u64, u64) {
    (dispatch_dir(), dispatch_dir() + dispatch_max_pages() * 4)
}

/// The patched-pages bitmap's byte range.
#[inline]
pub fn patched_pages_range() -> (u64, u64) {
    (patched_pages(), patched_pages() + dispatch_max_pages() / 8)
}

/// Address used for effective addresses that do not fit in 32 bits. Chosen so
/// that even a 16-byte access starting here stays out of bounds.
pub const POISON_ADDR: u32 = 0xFFFF_FFF0;

/// Start of the runtime-reserved low region. Page 0 is left unmapped so guest
/// null-pointer dereferences land somewhere recognisable in a debugger.
pub const RESERVED_BASE: u64 = 0x1000;

/// End of the runtime-reserved low region; guest mappings start above here.
///
/// Below the 0x400000 non-PIE link address (see [`EXEC_BASE`]) so a
/// fixed-address image never collides with runtime state. 1 MiB holds ~950
/// thread contexts.
pub const RESERVED_END: u64 = 0x0010_0000;

/// Default load bias for position-independent executables.
///
/// Low, because memory up to the highest touched address must exist per
/// process from creation. 16 MiB leaves room under it for a non-PIE image at
/// [`EXEC_BASE`] and the runtime's tables.
pub const PIE_BASE: u64 = 0x0100_0000;

/// Where a non-PIE image's own `p_vaddr` values are honoured from. Non-PIE
/// aarch64 links at 0x400000, which is above both the reserved region and the
/// dispatch directory, so a fixed-address image never collides with them.
pub const EXEC_BASE: u64 = 0x0040_0000;

/// Where a dynamic program's interpreter is placed: above the executable and
/// its break, below the stack. It defines the break's ceiling ([`BRK_TOP`]),
/// since a break growing past it would run over `ld.so`.
pub const INTERP_BASE: u64 = 0x04a0_0000;

/// Highest address the break may reach. The break grows up from the end of
/// the image at [`PIE_BASE`]; a program that wants more gets a failing `brk`
/// and libc falls back to `mmap`.
pub const BRK_TOP: u64 = INTERP_BASE;

/// Top of the guest stack. Grows down from here.
///
/// Kept just above the image and its break, not at the top of the address
/// space: linear memory must be allocated up to its highest used address, so
/// a stack at the top would make every process cost the whole 4 GiB.
pub const STACK_TOP: u64 = 0x0580_0000;

/// Default guest stack size (8 MiB, matching a typical `RLIMIT_STACK`).
pub const STACK_SIZE: u64 = 8 * 1024 * 1024;

/// Ceiling of the region a guest's mappings and the runtime's heap grow into.
///
/// Exactly the dispatch directory's reach, and that is the constraint, not a
/// coincidence: `ld.so` maps libraries into this arena and calls into them, so
/// code lives here, and code the directory cannot name can never be
/// translated. The assertion below keeps the two numbers together.
pub const MMAP_TOP: u64 = DISPATCH_MAX_PAGES * DISPATCH_PAGE;

/// Lowest address the mmap allocator will hand out, and the bottom of the
/// region that grows.
pub const MMAP_BOTTOM: u64 = 0x0700_0000;

/// The bottom of the region shared objects are placed in.
///
/// A translated block has its guest addresses baked in, so processes can share
/// a library's translation only if both map it at the same address. This fixed
/// window plus a system-wide ownership table (`ObjectPlace`) gives every
/// process the same placement, which a per-process bump allocator cannot.
pub const PLACE_BOTTOM: u64 = MMAP_BOTTOM;

/// How much of the address space that window is.
///
/// A fixed constant reserved from the start: the heap and arena share one
/// monotonic cursor, so a placement is only safe if nothing could ever have
/// been put there, and the size cannot be learned during a run or reclaimed
/// after. 64 MiB is apt's whole closure with room over; objects that overflow
/// it fall back to per-process translation.
pub const PLACE_WINDOW: u64 = 64 << 20;

/// One past the top of the placement window.
pub const PLACE_TOP: u64 = PLACE_BOTTOM + PLACE_WINDOW;

/// Lowest address the growing arena -- guest mappings and the runtime's own
/// heap alike -- will hand out, when no pin region is reserved.
///
/// Above the placement window, because the two must never overlap and only
/// this one can be made to move.
pub const ARENA_BOTTOM: u64 = PLACE_TOP;

/// The size of the pin region: address space reserved right above the
/// placement window for one project's libraries, each at the fixed base its
/// build gave it (an ejected Linux program whose libraries do not fit the
/// window). Zero, the default, reserves nothing.
static PIN_REGION: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Reserve the pin region, rounded up to 64 KiB. Set once per process before
/// anything is taken from the arena (the host does it before its first
/// allocation); it only grows.
pub fn set_pin_region(bytes: u64) {
    let bytes = bytes.min(MMAP_TOP - PLACE_TOP).next_multiple_of(0x1_0000);
    PIN_REGION.fetch_max(bytes, core::sync::atomic::Ordering::Relaxed);
}

/// One past the top of the pin region (`PLACE_TOP` when there is none).
pub fn pin_region_top() -> u64 {
    PLACE_TOP + PIN_REGION.load(core::sync::atomic::Ordering::Relaxed)
}

/// Lowest address the growing arena hands out: above the placement window
/// and the pin region.
pub fn arena_bottom() -> u64 {
    pin_region_top()
}

// ---------------------------------------------------------------------------
// Thread context
// ---------------------------------------------------------------------------

/// Byte offsets within the per-thread guest context structure, which lives in
/// guest linear memory in the reserved low region.
pub mod ctx {
    /// `x0`–`x30`, 8 bytes each.
    pub const X: usize = 0;
    /// Stack pointer. On x86-64 the stack pointer is an ordinary GPR (`rsp`,
    /// `x(4)`), so this names that slot and the kernel's every use of `SP`
    /// reads the right register.
    pub const SP: usize = if super::X86 { x(4) } else { 31 * 8 };
    /// Program counter. Only meaningful at block boundaries and safepoints.
    pub const PC: usize = 32 * 8;

    /// The stack pointer of a process on the i386 ABI (`compat`): `esp`,
    /// `x(4)`, in every build. On x86-64 that is `SP` itself; an AArch64
    /// build has i386-ABI processes too -- native wasm32 programs, whose
    /// musl port reads and writes `x(4)` -- and their stack pointer is not
    /// its `SP`. Everything else on the i386 ABI (the number in `x(0)`, the
    /// arguments, `PC`) is at the same offsets in both layouts.
    pub const fn sp_slot(compat: bool) -> usize {
        if compat { x(4) } else { SP }
    }

    /// Condition flags: one **byte** each, inside a single 32-bit slot. One
    /// flag stores as one `i32.store8`, all four as one `i32.store`; a byte
    /// (not a bit) each keeps the single-flag store off a read-modify-write.
    /// Every flag holds 0 or 1 (the lifter masks `& 1`; `adc` reads C as an
    /// addend).
    pub const NZCV: usize = 33 * 8;
    pub const NF: usize = NZCV;
    pub const ZF: usize = NZCV + 1;
    pub const CF: usize = NZCV + 2;
    pub const VF: usize = NZCV + 3;

    /// x86-64 only: the parity, auxiliary-carry and direction flags, bytes in
    /// the doubleword after NZCV so all six arithmetic flags (N=SF, Z, C,
    /// V=OF, P, A) store as one 64-bit word. `DF` is not an arithmetic flag
    /// (only `cld`/`std` write it) but keeps a byte here for one packed frame.
    pub const PF: usize = NZCV + 4;
    pub const AF: usize = NZCV + 5;
    pub const DF: usize = NZCV + 6;

    /// x86 only: the EFLAGS system bits user code can toggle, in place as a
    /// 32-bit value: AC (bit 18) and ID (bit 21). `popf` stores them and
    /// `pushf` gives them back, which is how a program tests for `cpuid`
    /// (it flips ID) or for a 486 (it flips AC). Stored only: gex does not
    /// check alignment when AC is set.
    pub const EFLAGS_SYS: usize = 34 * 8;

    /// Pack four 0/1 flags into the slot (and [`unpack_nzcv`] to reverse), so
    /// the packing lives in one place.
    #[inline]
    pub const fn pack_nzcv(n: u32, z: u32, c: u32, v: u32) -> u32 {
        (n & 1) | ((z & 1) << 8) | ((c & 1) << 16) | ((v & 1) << 24)
    }

    /// `(n, z, c, v)`, each 0 or 1.
    #[inline]
    pub const fn unpack_nzcv(packed: u32) -> (u32, u32, u32, u32) {
        (
            packed & 1,
            (packed >> 8) & 1,
            (packed >> 16) & 1,
            (packed >> 24) & 1,
        )
    }

    /// `TPIDR_EL0` — the guest thread pointer. On x86-64 this slot is the
    /// `FS` segment base, which is how that ABI reaches thread-local storage.
    pub const TPIDR: usize = 35 * 8;
    /// `TPIDRRO_EL0`; the `GS` base on x86-64.
    pub const TPIDRRO: usize = 36 * 8;
    /// The x86-64 names for the same two slots.
    pub const FS_BASE: usize = TPIDR;
    pub const GS_BASE: usize = TPIDRRO;

    /// Local exclusive monitor: address, value and access size. `SIZE == 0`
    /// means the monitor is clear.
    pub const MON_ADDR: usize = 37 * 8;
    pub const MON_VALUE: usize = 38 * 8;
    pub const MON_SIZE: usize = 39 * 8;

    /// FP control and status registers. On x86-64 `FPCR` holds `MXCSR`.
    pub const FPCR: usize = 40 * 8;
    pub const FPSR: usize = FPCR + 4;
    pub const MXCSR: usize = FPCR;

    /// Per-thread interrupt flag, polled at safepoints for signal delivery.
    pub const INTERRUPT: usize = 41 * 8;
    /// Countdown to the next safepoint poll in tier-1 code.
    pub const SAFEPOINT_COUNTER: usize = 41 * 8 + 4;

    /// Guest thread id, as reported by `gettid`.
    pub const TID: usize = 42 * 8;

    /// A slot nothing reads, for measurement only. `GEX_DISPATCH_BALLAST`
    /// writes load/store pairs here on the indirect dispatch path; a store
    /// cannot be optimised away, so it measures work on that path. See
    /// `emit_dispatch_tail_call`.
    pub const BALLAST: usize = 43 * 8;

    /// The target of the last indirect branch, written just before the
    /// `br_table` that dispatches it. Dispatch is a tail call that leaves no
    /// wasm frames, so this is the only record of where a crash on that path
    /// was headed. One store per indirect branch; direct branches use
    /// `return_call` and never come here.
    pub const LAST_TARGET: usize = 44 * 8;
    /// The start of the block that made the last indirect branch, stored
    /// beside `LAST_TARGET`: a wild target is only half the story, the code
    /// that computed it is the other half.
    pub const LAST_SOURCE: usize = 46 * 8;

    /// The raw system-call number the guest last issued, saved by the kernel
    /// at entry. On x86-64 the number's register (`rax`) is also the result
    /// register, so a host that wants to know which call just ran after the
    /// fact cannot read it back from there.
    pub const LAST_SYSCALL: usize = 45 * 8;

    /// x86 only: the thread's code selector, a `u32`. 0 is the process's
    /// own class (an x86-64 process's code is 64-bit, an i386 one's
    /// 32-bit); a WoW64 process's threads switch with far transfers and
    /// `iret` to [`CS_COMPAT`] (32-bit) and [`CS_LONG`] (64-bit), as
    /// Windows' selectors name them. Code is translated in the mode it names.
    pub const CS: usize = 47 * 8;
    /// 32-bit user code (Windows' WoW64 selector; i386 Windows' 0x1b is
    /// taken as the same).
    pub const CS_COMPAT: u32 = 0x23;
    /// 64-bit user code.
    pub const CS_LONG: u32 = 0x33;

    /// Whether code under selector `cs` is 32-bit, `default` being the
    /// process's own class (`cs` 0).
    pub const fn cs_compat(cs: u32, default: bool) -> bool {
        match cs {
            0 => default,
            0x23 | 0x1b => true,
            _ => false,
        }
    }

    /// `v0`–`v31`, the SIMD/FP register file, 16 bytes each. On x86-64 the
    /// first sixteen are `xmm0`–`xmm15`.
    pub const V: usize = 48 * 8;

    /// x86-64 only: the x87 register stack, eight 16-byte slots holding each
    /// physical register `R(i)` (not stack-relative) in the 80-bit extended
    /// format, laid out as `fxsave` does: the 64-bit significand with its
    /// explicit integer bit at +0, the sign and exponent word at +8, six
    /// bytes of padding.
    pub const X87: usize = V + 32 * 16;
    /// x87 `TOP` (0..7), the physical register `ST(0)` names.
    pub const X87_TOP: usize = X87 + 8 * 16;
    /// x87 control word, as `fldcw`/`fnstcw` see it. Each of these x87
    /// words has a 4-byte slot of its own: translated code moves them as
    /// 32-bit values.
    pub const X87_CW: usize = X87_TOP + 4;
    /// x87 status word, less `TOP` (which lives in `X87_TOP`).
    pub const X87_SW: usize = X87_CW + 4;
    /// One bit per physical register: 1 means valid (tag word derived).
    pub const X87_TAGS: usize = X87_SW + 4;
    /// 16 bytes through which an x87 memory operand passes to and from the
    /// host, which computes the x87's arithmetic (`gex_f80`): raw
    /// bits at +0, and for the 80-bit format the sign and exponent at +8.
    pub const X87_TMP: usize = X87_TAGS + 4;

    /// A 64-byte per-thread scratch area the kernel may use for structures it
    /// must materialise in guest memory on the guest's behalf (an x86-64
    /// `poll` becomes `ppoll` with a timespec that has to live somewhere).
    pub const SCRATCH: usize = X87_TMP + 16;

    /// Scratch for the i386 ABI's rewritten calls (`gex-kernel`'s `i386`):
    /// `SCRATCH` where the context has one; an AArch64 context ends at
    /// `v31`, and its only i386-ABI processes are native wasm32 programs,
    /// which never use the vector registers, so their first 64 bytes.
    pub const COMPAT_SCRATCH: usize = if super::X86 { SCRATCH } else { V };

    /// The context structure's own registers, before the trailing
    /// system-call immediate. The x87 stack and the scratch area are
    /// x86-64's; an AArch64 context ends with `v31`.
    pub const SIZE_REGS: usize = if super::X86 {
        SCRATCH + 64
    } else {
        V + 32 * 16
    };

    /// The immediate of the `svc` the guest issued, for ISAs that encode the
    /// system-call number in the instruction itself: an AArch64 ABI that
    /// numbers its calls `svc #imm`, with the arguments in `x0`–`x5` and no
    /// number register to read. The emitter stores the immediate here before
    /// it calls `gex.syscall`; an ISA whose number lives in a register
    /// (Linux `svc #0`, x86's `syscall`) writes nothing and leaves 0, which
    /// is what the kernel then reads as "take [`Self::SYSCALL_NR`]". A
    /// system call's own store leaves the register file alone: `x8` is the
    /// AArch64 result-location register and clobbering it would break the
    /// guest's ABI across the call.
    pub const SYSCALL_IMM: usize = SIZE_REGS;

    /// Total size of the context structure, rounded up.
    pub const SIZE: usize = SIZE_REGS + 16;

    /// Where the guest puts a system-call number: `x8`, or `rax` (`x(0)` in
    /// gex's numbering of x86-64 registers, which is the hardware's). An
    /// immediate-numbered call carries its number in
    /// [`Self::SYSCALL_IMM`] instead.
    pub const SYSCALL_NR: usize = if super::X86 { x(0) } else { x(8) };
    /// Where the six system-call arguments arrive: `x0`–`x5`, or
    /// `rdi, rsi, rdx, r10, r8, r9`.
    pub const SYSCALL_ARGS: [usize; 6] = if super::X86 {
        [x(7), x(6), x(2), x(10), x(8), x(9)]
    } else {
        [x(0), x(1), x(2), x(3), x(4), x(5)]
    };
    /// Where a system call's result goes: `x0` (`rax` is `x(0)`).
    pub const SYSCALL_RET: usize = x(0);

    /// The first argument register of the guest's C calling convention
    /// (`x0`, or `rdi`), and the next two. What a library-function
    /// intrinsic reads its arguments from.
    pub const CALL_ARGS: [usize; 3] = if super::X86 {
        [x(7), x(6), x(2)]
    } else {
        [x(0), x(1), x(2)]
    };

    /// Byte offset of general-purpose register `n` (0..=30).
    #[inline]
    pub const fn x(n: u8) -> usize {
        X + (n as usize) * 8
    }

    /// Byte offset of vector register `n` (0..=31).
    #[inline]
    pub const fn v(n: u8) -> usize {
        V + (n as usize) * 16
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Bytes of guest text one dispatch-directory entry covers.
pub const DISPATCH_PAGE: u64 = 4096;

/// Dispatch-table slots per covered page: one per possible instruction start,
/// so 1,024 for A64's 4-byte words and 4,096 for byte-aligned x86-64.
pub const SLOTS_PER_PAGE: u64 = DISPATCH_PAGE >> DISPATCH_SHIFT;

/// Table slots reserved at index 0, all holding the miss stub. An untranslated
/// page has directory entry zero, so its index lands in this span and calls
/// the stub with no sentinel comparison — this is what makes lookup branchless.
pub const MISS_SLOTS: u64 = SLOTS_PER_PAGE;

/// Guest address of the dispatch directory: one `u32` per page of guest text,
/// holding the base table slot for that page, or zero when it has none.
///
/// Indirecting through a page directory keeps the table proportional to the
/// code actually mapped, rather than to the span between far-apart libraries,
/// which is what makes dynamic linking affordable.
pub const DISPATCH_DIR: u64 = RESERVED_END;

/// Pages the directory covers, and so the highest address code may occupy.
pub const DISPATCH_MAX_PAGES: u64 = 0x8_0000;

/// Size of the directory in bytes.
pub const DISPATCH_DIR_BYTES: u64 = DISPATCH_MAX_PAGES * 4;

/// First guest address above the dispatch directory.
pub const DISPATCH_DIR_END: u64 = DISPATCH_DIR + DISPATCH_DIR_BYTES;

/// Ceiling on one dispatch table (10M slots ≈ 38 MiB of guest text). The
/// table grows from the image's size, so the ceiling costs nothing until used.
///
/// The number is V8's, not a choice: `WebAssembly.Table.grow` refuses past ten
/// million however the table is declared, so a larger ceiling would not hold.
pub const DISPATCH_TABLE_MAX_SLOTS: u64 = 10_000_000;

/// How many dispatch tables there are.
///
/// One table (V8's 10M-slot cap, ≈38 MiB of text) cannot hold a real
/// application. wasm allows many tables and `call_indirect` names one, so the
/// cap is per table: eight of them is 305 MiB of guest text.
///
/// The cost is a `br_table` on the indirect-branch path only; direct branches
/// use `return_call`. Set this to 1 and the emitter goes back to a single
/// `return_call_indirect` with no selection at all.
pub const DISPATCH_TABLES: usize = 8;

/// Bits of a directory entry that hold the slot; the top 8 hold the table
/// index. Ten million needs 24 bits, so the table index is free in the `u32`.
pub const DISPATCH_SLOT_BITS: u32 = 24;

/// Mask of the slot half of a directory entry.
pub const DISPATCH_SLOT_MASK: u32 = (1u32 << DISPATCH_SLOT_BITS) - 1;

/// Blocks above which a module's element segments are passive.
///
/// A module past this size gets passive segments and a fill helper, because
/// writing every block into every process's table is most of what a process
/// costs; below it, segments are active and instantiation does the writing.
/// Both emitter and host consult this: active-segment modules keep the
/// closure's numbering, only demand-filled ones move.
pub const BIG_BLOCKS: usize = 4096;

/// Slots a dispatch fill writes around the one that missed, rather than the
/// whole 1,024-entry page (`table.init` materialises a `funcref` per entry,
/// costly on the first fill). Trades entries for misses; wins unless a page is
/// entered at more than `1024 / FILL_SPAN` scattered points.
pub const FILL_SPAN: u32 = 32;

/// Pack a table index and a base slot into a directory entry. Entry zero must
/// keep meaning "no translation, table 0, miss span" — what an unwritten
/// directory reads as, and what makes the lookup branchless.
#[inline]
pub const fn dispatch_entry(table: usize, base: u32) -> u32 {
    ((table as u32) << DISPATCH_SLOT_BITS) | (base & DISPATCH_SLOT_MASK)
}

/// The table a directory entry names.
#[inline]
pub const fn dispatch_entry_table(entry: u32) -> usize {
    (entry >> DISPATCH_SLOT_BITS) as usize
}

/// The base slot a directory entry names, within its own table.
#[inline]
pub const fn dispatch_entry_base(entry: u32) -> u32 {
    entry & DISPATCH_SLOT_MASK
}

/// The bit a page number carries in a page key (gex-emit's `PageTable`,
/// a module's directory entries) for code translated in the mode the
/// process's own class is not: a WoW64 process's 32-bit code (or a 32-bit
/// process's 64-bit code). Such a page's entry is in a directory of its
/// own, allocated when the process first runs code in that mode and named
/// by [`imports::ALT_DIR`], so the same address reached in both modes never
/// shares a block, a slot or an entry.
pub const ALT_PAGE: u64 = 1 << 40;

/// The bit an address carries in a set of translated addresses for code in
/// the process's other mode ([`ALT_PAGE`]'s, for addresses).
pub const ALT_ADDR: u64 = 1 << 63;

/// `pc` as such a set keys it.
#[inline]
pub const fn addr_key(pc: u64, alt: bool) -> u64 {
    pc | if alt { ALT_ADDR } else { 0 }
}

/// Where a page key's directory entry is: in the process's directory, or
/// in the other one at `alt_dir` (`None` while there is none).
#[inline]
pub fn dir_entry_addr(key: u64, alt_dir: u64) -> Option<u64> {
    let (page, alt) = key_page(key);
    if page >= dispatch_max_pages() {
        return None;
    }
    match (alt, alt_dir) {
        (false, _) => Some(dispatch_dir() + page * 4),
        (true, 0) => None,
        (true, d) => Some(d + page * 4),
    }
}

/// Bytes of the other directory: as many pages as the process's own.
#[inline]
pub fn alt_dir_bytes() -> u64 {
    dispatch_max_pages() * 4
}

/// The page key for `pc` in its mode (`alt`: the process's other one).
#[inline]
pub const fn page_key(pc: u64, alt: bool) -> u64 {
    dispatch_page(pc) | if alt { ALT_PAGE } else { 0 }
}

/// A page key's page number and whether it is the other mode's.
#[inline]
pub const fn key_page(key: u64) -> (u64, bool) {
    (key & !ALT_PAGE, key & ALT_PAGE != 0)
}

/// Directory index for a guest address.
#[inline]
pub const fn dispatch_page(pc: u64) -> u64 {
    pc / DISPATCH_PAGE
}

/// Scratch doubleword used to neutralise a store-exclusive whose local monitor
/// no longer matches (see `gex_lift::Lifter::store_exclusive`). In the reserved
/// low region, 8-byte aligned, never guest-observable.
pub const EXCLUSIVE_SCRATCH: u64 = RESERVED_END - 64;

/// The lock word an x86 locked access spanning two aligned 8-byte words is
/// done under (wasm has no atomic wider than eight bytes), and beside it a
/// word set once the first such access has been reported. In the reserved
/// low region, never guest-observable.
pub const SPLIT_LOCK: u64 = RESERVED_END - 128;

/// One bit per dispatch-directory page (bit `page & 7` of byte `page >> 3`),
/// set by the host once a page's code changed after a function-tier unit was
/// installed over it (Detours patching a prologue, a JIT rewriting code).
/// Compiled code checks a callee's entry page before a direct call and leaves
/// for it when set, so a direct C call never runs a function whose bytes are
/// no longer the ones it was compiled from. Sticky: tier-1 retranslates the
/// page, the C stays out. In the reserved low region, above the contexts.
pub const PATCHED_PAGES: u64 = RESERVED_END - 0x1000 - PATCHED_PAGES_BYTES;
pub const PATCHED_PAGES_BYTES: u64 = DISPATCH_MAX_PAGES / 8;

/// Function-tier nest exits (a nest root handing control to dispatch), a
/// plain `u64` counter bumped without atomics: a statistic, for
/// a statistic of leaves per second.
pub const NEST_EXITS: u64 = RESERVED_END - 256;

/// How many slots each of the function tier's body tables has in every store
/// (`[u32; DISPATCH_TABLES]`: body table `t` mirrors dispatch table `t`; each
/// word the smallest over the process's stores): a unit's indirect call
/// leaves for a slot at or above it rather than index past a table a lagging
/// store has not grown yet.
pub const BODY_LIMITS: u64 = RESERVED_END - 320;

/// Body table 0's limit (`BODY_LIMITS[0]`), the name the browser shim used
/// first.
pub const BODY_LIMIT: u64 = BODY_LIMITS;

/// The function table's probe cache: guest address
/// → [`ftab`] index, direct-mapped, shared by every thread of the process.
/// Entry `fcache_slot(a)` is one `u64`: the address's low
/// [`FCACHE_TAG_BITS`] bits (all of it, for code below 64 GiB: a memory64
/// process's images sit near the top of its 16 GiB) and the index above
/// them. Generated code probes it inline on an indirect transfer; a miss (or
/// a stale entry) is index 0, the resolve stub, which asks the host
/// (`gex.lookup`) and writes the entry. All zero at start: address 0 maps to
/// the resolve stub.
pub const FCACHE: u64 = PATCHED_PAGES - FCACHE_BYTES;
/// `log2` of the probe cache's entries.
pub const FCACHE_BITS: u32 = 15;
pub const FCACHE_ENTRIES: u64 = 1 << FCACHE_BITS;
pub const FCACHE_BYTES: u64 = FCACHE_ENTRIES * 8;

/// The probe cache slot of guest address `a`: a multiplicative hash of its
/// low 32 bits, so code that is 16-byte aligned
/// still spreads over every slot. Every emitter computes exactly this.
#[inline]
pub const fn fcache_slot(a: u64) -> u64 {
    ((a as u32).wrapping_mul(FCACHE_MUL) >> (32 - FCACHE_BITS)) as u64
}
/// The multiplier of [`fcache_slot`].
pub const FCACHE_MUL: u32 = 0x9e37_79b1;

/// The address of `a`'s probe cache entry.
#[inline]
pub const fn fcache_entry(a: u64) -> u64 {
    FCACHE + 8 * fcache_slot(a)
}

/// The address bits a probe cache entry holds, below its index: the whole
/// address of code below 64 GiB. A target at or past that never hits (the
/// entry cannot name it), and a probe compares the whole target, so an
/// entry never stands for another address with the same low bits.
pub const FCACHE_TAG_BITS: u32 = 36;
/// The probe cache entry layout a native module's inline probe reads
/// (`gex_ftab_idx` in gex_abi.h): 1 the index above 32 address bits, 2 the
/// index above [`FCACHE_TAG_BITS`]. The header records it in every module
/// built with it (a `gex.abi` custom section, one little-endian u32 per
/// object), and the browser host refuses a 32-bit Wine module built for another
/// (a module without the section is a version-1 one).
pub const NATIVE_ABI: u32 = 2;
/// The address part of a probe cache entry.
pub const FCACHE_TAG_MASK: u64 = (1 << FCACHE_TAG_BITS) - 1;
/// One past the largest index an entry holds (its other 28 bits).
pub const FCACHE_IDX_LIMIT: u64 = 1 << (64 - FCACHE_TAG_BITS);

/// A probe cache entry for `a` at index `idx`; `None` for an address or
/// index an entry cannot hold (the slot is then left to miss).
#[inline]
pub const fn fcache_pack(a: u64, idx: u32) -> Option<u64> {
    if a > FCACHE_TAG_MASK || idx as u64 >= FCACHE_IDX_LIMIT {
        return None;
    }
    Some(a | ((idx as u64) << FCACHE_TAG_BITS))
}

/// The index probe cache entry `e` holds for target `t`, or the resolve stub
/// ([`ftab::RESOLVE`]) when it holds another address: what every emitter's
/// inline probe computes.
#[inline]
pub const fn fcache_probe(e: u64, t: u64) -> u32 {
    if e & FCACHE_TAG_MASK == t { (e >> FCACHE_TAG_BITS) as u32 } else { ftab::RESOLVE }
}

/// Where the thread contexts must end: the runtime's fixed words
/// ([`FCACHE`] up to [`RESERVED_END`]) start here.
pub const CTX_LIMIT: u64 = FCACHE;

/// One `u32` per code page that changes whenever the page's code may have
/// (per-function page checks): a function translated at run
/// time records the value of every page it covers and compares them on
/// entry; on a mismatch it re-enters through the resolve stub with
/// `ctx::LAST_TARGET` = its entry, and the host translates it again. Until
/// tier-1's dispatch slots go, this *is* the dispatch directory:
/// a page retired is zeroed there and numbered afresh, never with an old
/// value, and the host numbers every page it hands a translator
/// (`page_gen`), so an entry of zero never belongs to a translation. Units
/// built ahead of time check [`PATCHED_PAGES`] instead (their bytes are
/// fixed at build time, so "changed since install" is the question).
pub const PAGE_GEN: u64 = DISPATCH_DIR;

/// The function table: one `funcref` table per
/// process -- per store natively, per Worker in the browser, numbered alike
/// in all of them -- holding only `F_body` functions:
///
/// ```text
/// F_body(ctx: i32, depth: i32, need: i32, ret: i64) -> i32
/// ```
///
/// 1: the predicted return happened (the guest returned to `ret` with the
/// stack pointer back); 0: the function left with all state written back
/// and `ctx::PC` set. The callee does the return check; `depth` is the
/// recursion guard's estimate.
///
/// Every call through the table first stores the target in
/// `ctx::LAST_TARGET`, so any slot a thread's table has not filled yet --
/// which holds the resolve stub -- and any retired slot -- which is
/// pointed back at it -- can still find out where the call was going.
pub mod ftab {
    /// The table, imported by every module that calls through it. Units
    /// linked by wasm-ld (`--experimental-pic -shared`) import it as
    /// [`PIC_TABLE`] instead; the host defines both names to the same table.
    pub const TABLE: &str = "ftab";
    /// What wasm-ld's position-independent modules call the table, the base
    /// of their slots and their memory base, all in module [`PIC_MODULE`].
    pub const PIC_MODULE: &str = "env";
    pub const PIC_TABLE: &str = "__indirect_function_table";
    pub const PIC_TABLE_BASE: &str = "__table_base";
    pub const PIC_MEMORY_BASE: &str = "__memory_base";
    pub const PIC_STACK_POINTER: &str = "__stack_pointer";

    /// `gex.lookup(ctx: i32, addr: i64) -> i32`: the index of a function
    /// for guest address `addr`, translating it on demand (a unit's body,
    /// else the function JIT, else a region). Never 0. Also writes the
    /// probe cache entry, so the next probe hits.
    pub const LOOKUP: &str = "lookup";

    /// Index 0: the resolve stub. Reads `ctx::LAST_TARGET`, calls
    /// [`LOOKUP`], tail-calls the result with its own arguments.
    pub const RESOLVE: u32 = 0;
    /// Index 1: runs code no function body holds yet (tier-1 until the
    /// region tier replaces it), entered at `ctx::LAST_TARGET`, under the
    /// same contract.
    pub const FALLBACK: u32 = 1;
    /// Indices below this are the runtime's; modules' functions start here.
    pub const RESERVED: u32 = 16;

    /// Native PE code's function pointers are table indices, and they stay
    /// below this: the end of the dispatch directory ([`super::DISPATCH_DIR_END`]),
    /// below which no Windows view lies (Wine's virtual memory floor on gex), so
    /// a value below it is a function, never data. A Wine PE module whose slots
    /// would reach it is refused. `gex.image_of` finds the DLL of either.
    pub const NATIVE_FN_TOP: u64 = super::DISPATCH_DIR_END;
    /// `gex.image_of(addr: i64) -> i64`: the base of the skeleton of the
    /// native DLL whose function (a table index below [`NATIVE_FN_TOP`]) or
    /// static data `addr` is, or 0. Indices are one space for both modes of a
    /// WoW64 process, and each module's data is its own allocation.
    pub const IMAGE_OF: &str = "image_of";
    /// `gex.icall_limit64` (i64) and `gex.icall_limit` (i32): mutable globals,
    /// one pair per thread, that native PE code's indirect calls compare a
    /// function pointer with (the GexIcall pass): below it, a table index
    /// the thread's table holds, called directly; at or above it, the exit
    /// stub's guest call. The host keeps them at the slots the thread's
    /// table holds (capped at [`NATIVE_FN_TOP`]), raised as it catches up,
    /// so an index into a module another thread installed since reaches the
    /// call-out, which catches the thread up first.
    pub const ICALL_LIMIT64: &str = "icall_limit64";
    pub const ICALL_LIMIT: &str = "icall_limit";

    /// `F_body`'s results.
    pub const RET_PREDICTED: i32 = 1;
    pub const RET_LEFT: i32 = 0;

    /// `need` bits: the caller's continuation reads a flag, so an expected
    /// return writes the flags back too. A root passes [`NEED_ALL`].
    pub const NEED_FLAGS: u32 = 1;
    pub const NEED_ALL: u32 = u32::MAX;
    /// `need` bits of a guest call from native code (its `ret` the native
    /// return pad): the bytes of arguments the callee may pop. Its return
    /// is the call's only with the stack pointer at most that far above the
    /// return address's slot. Bodies read only [`NEED_FLAGS`] and pass
    /// `need` on unchanged.
    pub const NEED_ARGS: u32 = !3;

    /// The argument bytes a native caller's `need` bounds its callee's
    /// return by ([`NEED_ARGS`]); `None` for [`NEED_ALL`], which names none.
    pub const fn need_args(need: u32) -> Option<u64> {
        if need == NEED_ALL {
            None
        } else {
            Some((need & NEED_ARGS) as u64)
        }
    }

    /// The `ret` a root passes: no code is there, so no return is predicted.
    pub const NO_RETURN: u64 = u64::MAX;

    /// A module's install descriptor: a custom section listing, for each
    /// slot its one active element segment fills from `__table_base`, the
    /// guest address of the function there (0: not reachable by address,
    /// e.g. a helper). `u32` count, then that many little-endian `u64`s.
    pub const FUNCS_SECTION: &str = "gex.funcs";

    /// Marks a slot's address in [`FUNCS_SECTION`] as a lifted twin: the
    /// image's own body for an export a `lifted` provider serves
    /// (gex-cgen/providers/provider.h `gp_lifted`), kept beside the
    /// provider. Hosts map it apart from the address itself -- never in the
    /// probe cache, never bridged from tier-1 -- and answer
    /// [`TWIN_IMPORT`] with it.
    pub const TWIN: u64 = 1 << 63;

    /// `(addr: i64) -> i32`: the function-table index of `addr`'s lifted
    /// twin, or 0 when there is none. A provider's fallback only.
    pub const TWIN_IMPORT: &str = "ftab_twin";
}

/// A counter the browser glue takes bootstrap stacks by: a Worker calls
/// `gex_worker_init` before it has a stack of its own, and wasm-ld's default
/// one is the guest's low memory (hosts/browser/js/gex.mjs `#workerStack`).
pub const BOOT_STACK_COUNTER: u64 = RESERVED_END - 192;

/// Address of the context for thread slot `n`.
///
/// Contexts are packed into the reserved low region, one per guest thread.
#[inline]
pub const fn ctx_addr(slot: u32) -> u64 {
    RESERVED_BASE + (slot as u64) * ((ctx::SIZE as u64 + 63) & !63)
}

// ---------------------------------------------------------------------------
// Host import surface
// ---------------------------------------------------------------------------

/// Wasm import module name for every host-provided function.
pub const IMPORT_MODULE: &str = "gex";

/// Names of the functions translated code imports from the host.
pub mod imports {
    /// `svc #0`. Signature `(ctx: i32, pc: i64) -> ()`. Reads the number and
    /// args from the context, writes the result to `x0`. Guest exit is
    /// signalled by trapping, not a return value.
    pub const SYSCALL: &str = "syscall";

    /// An indirect branch to an address with no translation.
    /// Signature `(ctx: i32, pc: i64) -> i32`, returning the dispatch-table
    /// index to re-enter. With no translator this traps; with one it drives
    /// gex-jit.
    pub const DISPATCH_MISS: &str = "dispatch_miss";

    /// An immutable global in the memory's index type (`i32`, `i64` under
    /// memory64): the guest address of the process's other
    /// directory ([`crate::ALT_PAGE`]). Imported only by a module of code
    /// translated in the mode the process's own class is not (a WoW64
    /// process's 32-bit code); every other module is as it was.
    pub const ALT_DIR: &str = "alt_dir";

    /// The function tier calling code it does not hold. Signature `(ctx: i32, target: i64, ret: i64, sp_want: i64) ->
    /// i32`: everything is written back and `ctx::PC = target`; the host runs
    /// the guest from there until it returns to `ret` with the stack pointer
    /// back at `sp_want` (or above it by what a callee popped) and returns 1,
    /// the caller then carrying on; or returns 0 when it cannot (nested too
    /// deep, the stack unwound past the caller), and the caller leaves.
    pub const CALL_OUT: &str = "call_out";

    /// A decoded-but-unimplemented or architecturally undefined instruction.
    /// Signature `(ctx: i32, pc: i64, raw: i32) -> ()`. Raises SIGILL.
    pub const ILLEGAL: &str = "illegal";

    /// `brk #imm`. Signature `(ctx: i32, pc: i64, imm: i32) -> ()`.
    pub const BREAKPOINT: &str = "breakpoint";

    /// Not a `brk`: the stack-pointer check reporting a wild `sp`, reusing the
    /// breakpoint import. A guest `brk` immediate is 16 bits unsigned, so a
    /// negative value cannot be one and the host tells them apart. See
    /// `tier1::emit_sp_check`.
    pub const BRK_BAD_SP: i32 = -1;

    /// Not a `brk`: a diagnostic breakpoint planted at a chosen guest pc (see
    /// `tier1::emit_trap_pc`). The host dumps the live registers and RETURNS, so
    /// the block runs on -- unlike `BRK_BAD_SP`, which exits.
    pub const BRK_TRAP_PC: i32 = -2;

    /// Not a `brk`: a block-entry trace point (`GEX_TRACE_BLOCKS`), planted at
    /// the start of every block so a run can be followed block by block. The
    /// host prints the pc and a few registers and RETURNS.
    pub const BRK_TRACE: i32 = -3;

    /// Not a `brk`: a `Stmt::Trap` for an integer division by zero (x86
    /// `div`/`idiv`, #DE). The host raises it and does not return: SIGFPE
    /// with `FPE_INTDIV` for a Linux guest, EXCEPTION_INT_DIVIDE_BY_ZERO for
    /// a Windows one.
    pub const BRK_DIVIDE_BY_ZERO: i32 = -4;

    /// Not a `brk`: a `Stmt::Trap` for an integer division whose quotient
    /// does not fit (x86 #DE: `INT_MIN / -1`, or any quotient too wide for
    /// the destination). Linux reports it as SIGFPE with `FPE_INTDIV`, the
    /// same as a zero divisor; Windows as EXCEPTION_INT_OVERFLOW.
    pub const BRK_DIVIDE_OVERFLOW: i32 = -5;

    /// Not a `brk`: a `Stmt::Trap` for a general-protection fault, such as
    /// a privileged instruction (x86 `hlt`) in user mode. Linux reports it
    /// as SIGSEGV at the instruction.
    pub const BRK_GENERAL_PROTECTION: i32 = -6;

    /// Not a `brk`: a `Stmt::Trap` for an x87 exception the control word
    /// leaves unmasked. gex does not deliver x87 exceptions (a real x87
    /// would fault at the next x87 instruction), so the host fails loudly.
    pub const BRK_X87_EXCEPTION: i32 = -7;

    /// Not a `brk`: an access at or above [`super::KUSER_GUARD`] while
    /// KUSER_SHARED_DATA folding is on (32-bit Wine): a run-time pointer into
    /// Windows' old KUSER page, which the translator could not fold. `pc`
    /// is [`super::kuser_miss_pack`]'s (block pc and address). The host says
    /// "KUSER access not folded at pc X (module+off)" and raises an access
    /// violation (Wine's handler, through the fault hook); it does not
    /// return. The fix is a fold in the translator, not an emulation.
    pub const BRK_KUSER_UNFOLDED: i32 = -8;

    /// [`BRK_KUSER_UNFOLDED`] from code that goes on after the import
    /// returns (the function JIT's bodies, which leave for the context's PC):
    /// the access violation is delivered in place where the host can, under
    /// the native frames the code was called from, and the import returns.
    pub const BRK_KUSER_UNFOLDED_RESUMABLE: i32 = -9;

    /// A system register read the compiler could not resolve statically.
    /// Signature `(ctx: i32, pc: i64, sysreg: i32) -> i64`.
    pub const SYSREG_READ: &str = "sysreg_read";

    /// A system register write. Signature `(ctx: i32, pc: i64, sysreg: i32,
    /// value: i64) -> ()`.
    pub const SYSREG_WRITE: &str = "sysreg_write";

    /// Cache maintenance, including the `IC IVAU` translation-invalidation
    /// hook. Signature `(ctx: i32, pc: i64, op: i32, addr: i64) -> ()`.
    pub const CACHE_OP: &str = "cache_op";

    /// Fused multiply-add: `round(a * b + c)` with one rounding.
    ///
    /// Signature `(ctx: i32, a: i64, b: i64, c: i64, double: i32) -> i64`,
    /// operands and result as bit patterns (single precision in the low 32
    /// bits). Wasm has no FMA and a separate multiply-then-add rounds twice,
    /// enough to break musl's `pow`, so this asks the host rather than
    /// approximating.
    pub const FMA: &str = "fma";

    /// An x87 operation (`gex_ir::Stmt::X87`) the inline fast path did not
    /// cover, computed exactly (`gex_f80`). Pure: it reads only its
    /// arguments and writes only its results, so a block calls it without
    /// flushing its cached registers. Signature `(ctx: i32, op: i32,
    /// a_sig: i64, a_se: i32, b_sig: i64, b_se: i32, cw: i32) -> i64`, `op`
    /// being `X87Op::code` and `b` unused by the one-operand operations:
    /// the first result is returned; an extended result's sign and
    /// exponent are left as an i32 at `ctx::X87_TMP + 8`, and the
    /// status-word bits (exception flags, C1 at bit 9) as an i32 at
    /// `ctx::X87_TMP + 12`.
    pub const X87: &str = "x87";

    /// An SSE operation under a rounding control other than to nearest
    /// (`gex_ir::Expr::Mx`'s slow path), computed exactly
    /// (`gex_f80::sse::op`). Pure, as [`X87`] is. Its `mxcsr` holds only
    /// the rounding control (`gex_ir::MX_SPECIAL`).
    /// Signature `(ctx: i32, code: i32, a: i64, b: i64, mxcsr: i32) -> i64`:
    /// `code` is `gex_f80::sse`'s, the operands and the result raw bits
    /// (a single's or an `i32`'s zero-extended).
    pub const MX: &str = "mx";

    /// Safepoint poll taken when the interrupt flag is set.
    /// Signature `(ctx: i32, pc: i64) -> ()`.
    pub const SAFEPOINT: &str = "safepoint";

    /// An indirect call about to be made, in a unit of the counting flavor
    /// (`gex_cgen::image::icall_count`): its return address and target,
    /// which the host tallies into an indirect-call profile. Signature
    /// `(ctx: i32, ret: i64, target: i64) -> ()`.
    pub const ICALL_NOTE: &str = "icall_note";
}

/// Names translated modules export back to the host.
pub mod exports {
    /// The guest's linear memory.
    pub const MEMORY: &str = "memory";
    /// The PC-indexed dispatch table of translated block entry points.
    pub const DISPATCH_TABLE: &str = "dispatch";
    /// Writes one of the module's passive element segments into the imported
    /// tables, chosen by index. See `Module::add_fill`: passive segments let a
    /// module's blocks reach a table without every process paying for every
    /// block at instantiation.
    pub const FILL: &str = "gexfill";
    /// Writes whichever passive segment holds a given dispatch slot, which is
    /// the question a miss can actually ask. See `Module::add_fill_at`.
    pub const FILL_AT: &str = "gexfillat";
    /// Writes every passive segment in one call, which is what an eager fill
    /// wants: a crossing per segment costs more than the writes.
    pub const FILL_ALL: &str = "gexfillall";

    /// The dispatch tables, several because V8 caps one at ten million
    /// entries; see [`super::DISPATCH_TABLES`]. Table 0 keeps the bare name
    /// `dispatch` for compatibility and holds the miss span an unwritten
    /// directory entry of zero points at.
    pub const DISPATCH_TABLE_N: [&str; 8] = [
        "dispatch",
        "dispatch1",
        "dispatch2",
        "dispatch3",
        "dispatch4",
        "dispatch5",
        "dispatch6",
        "dispatch7",
    ];

    /// The name of dispatch table `i`.
    pub const fn dispatch_table(i: usize) -> &'static str {
        DISPATCH_TABLE_N[i]
    }

    /// `(ctx: i32, pc: i64) -> ()`: enter translated code at `pc`.
    pub const ENTER: &str = "enter";
}

/// Cache-maintenance operation codes passed to [`imports::CACHE_OP`].
pub mod cache_op {
    /// Not a cache operation: a block was entered for a neighbouring address
    /// that shares its dispatch slot (x86-64, see `DISPATCH_SHIFT`). The host
    /// forgets that address so the miss that follows re-translates it and the
    /// slot gets a trampoline covering both.
    pub const SLOT_CONFLICT: i32 = 9;
    /// `dc zva` — zero a cache line's worth of memory.
    pub const DC_ZVA: i32 = 0;
    /// `dc cvau`/`civac`/… — clean; a no-op on a coherent host.
    pub const DC_CLEAN: i32 = 1;
    /// `ic ivau`/`iallu` — invalidate translations covering the address.
    pub const IC_INVALIDATE: i32 = 2;
    /// `isb` — translation barrier.
    pub const ISB: i32 = 3;
    /// Anything else.
    pub const OTHER: i32 = 4;
    /// Not a cache operation: the first x86 locked access that spans two
    /// aligned 8-byte words, at the address given, which runs under
    /// [`super::SPLIT_LOCK`] -- atomic against other such accesses only.
    pub const SPLIT_LOCK: i32 = 10;
    /// Not a cache operation: a locked access below [`super::GUEST_FLOOR`],
    /// in the runtime's own memory, at the address given. The host raises
    /// an access violation (NT) or SIGSEGV (Linux) and does not return.
    pub const LOW_ACCESS: i32 = 11;
    /// Not a cache operation: the block at `pc` failed its page check (its
    /// pages are no longer numbered as when it was translated), and `addr` is
    /// the id of the module it belongs to (`TranslatedModule::module_id`).
    /// The host translates `pc` again if that module still provides it --
    /// otherwise dispatch already reaches a newer translation -- and the
    /// block then dispatches.
    pub const PAGE_STALE: i32 = 12;
    /// Not a cache operation: a function-tier nest root handed control to
    /// dispatch at `addr` (one exit in 4096, a statistic of exit sites).
    pub const NEST_EXIT_SAMPLE: i32 = 13;
    /// Not a cache operation: the function table's bridge in tier-1's
    /// dispatch slot for `addr` called a function that left for `addr`
    /// itself without running anything (no body serves it after all). The
    /// host lets tier-1 have the address: the miss that follows translates
    /// it and its block takes the slot.
    pub const FTAB_TIER1: i32 = 14;
}

/// The lowest address a guest access may name: below it are the runtime's
/// thread contexts, words and dispatch directory, which a null pointer's
/// field would otherwise write into silently (gex has no page protection).
/// Checked where an access already goes through host code or a slow path.
pub const GUEST_FLOOR: u64 = DISPATCH_DIR_END;

/// The null guard, a debugging opt-in ([`null_guard`]; GEX_NULL_GUARD=1
/// natively, `?nullguard=1` in the page): every translated guest access
/// below this address traps, as a read or write of an unmapped page does on
/// Linux (`vm.mmap_min_addr` is 64 KiB) and of the 64 KiB Windows reserves
/// at 0. Nothing a guest may map lies below [`GUEST_FLOOR`], so the guard
/// never refuses a real access. Off, the translators' output is what it was
/// without it, and no key names it.
///
/// The translators fold it into the access at no more than one add: the
/// address less `NULL_GUARD` wraps past 2^32 when it is below, and the
/// access's static offset adds `NULL_GUARD` back, so the engine's own bounds
/// check (the sum is past the end of memory) is the trap. Accesses to the
/// thread's own context (the x87 stack, `EFLAGS_SYS`), which lives below,
/// are left out by the value they are made through (`GuestReg::CtxAddr`).
pub const NULL_GUARD: u64 = 0x1_0000;
const _: () = assert!(NULL_GUARD <= GUEST_FLOOR && NULL_GUARD % 16 == 0);

static NULL_GUARD_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Turn the null guard ([`NULL_GUARD`]) on or off, before anything is
/// translated. A debugging switch: off by default, since it costs 2-8% in
/// tier 1 and a guest that reads through a null
/// pointer is broken anyway.
pub fn set_null_guard(on: bool) {
    NULL_GUARD_ON.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether the null guard is on ([`set_null_guard`]).
pub fn null_guard() -> bool {
    NULL_GUARD_ON.load(core::sync::atomic::Ordering::Relaxed)
}

/// KUSER_SHARED_DATA where Windows has it, a page every process maps
/// read-only and much i386 code reads by constant (tick counts, the
/// interrupt time, the OS version).
pub const KUSER_WINDOWS: u64 = 0x7ffe_0000;

/// One past the structure's page.
pub const KUSER_WINDOWS_END: u64 = KUSER_WINDOWS + 0x1000;

/// Set in the pc an `illegal` import is given by code that dispatches to the
/// context's PC if the call returns (tier 1): the host may then run the
/// instruction as a call (64-bit native Wine's system call entry, a `ud2`)
/// instead of raising the fault. Every other caller's `illegal` never
/// returns. The host masks it off.
pub const ILLEGAL_RETURNS_PC: u64 = 1 << 61;

/// Where 32-bit Wine maps KUSER_SHARED_DATA instead (wasm32): the top of the
/// low space it reserves below the wasm32 program's stack (`STACK_TOP -
/// STACK_SIZE`), so that a 32-bit Wine process's memory need not reach 2 GiB. Wine's
/// `include/wine/kuser.h` says the same address (its ntdll checks it against
/// `GEX_WINE32_KUSER`). i386 code's constant accesses to [`KUSER_WINDOWS`] are
/// folded to it ([`fold_kuser`]); any other access to the old page faults.
pub const WINE32_KUSER: u64 = 0x04ff_0000;

static KUSER_FOLD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Fold KUSER_SHARED_DATA's constant addresses in everything translated from
/// now on (a 32-bit Wine host, or `gexc cgen-image` for wine32 units). In the unit key.
pub fn set_kuser_fold(on: bool) {
    KUSER_FOLD.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether constant KUSER_SHARED_DATA addresses are folded ([`set_kuser_fold`]).
pub fn kuser_fold() -> bool {
    KUSER_FOLD.load(core::sync::atomic::Ordering::Relaxed)
}

/// Whether `exe` (Wine's loader's first argument, a Windows or Unix path)
/// is one of Wine's session programs: the services, device and RPC hosts,
/// the desktop and the prefix's boot, into which no program's code is
/// loaded. Not rundll32 or regsvr32, which load a program's DLLs.
pub fn wine_session_program(exe: &str) -> bool {
    const SESSION: &[&str] = &["explorer", "services", "plugplay", "winedevice", "rpcss", "svchost", "wineboot", "winemenubuilder"];
    let name = exe.rsplit(['/', '\\']).next().unwrap_or(exe);
    let name = match name.len().checked_sub(4) {
        Some(n) if name.is_char_boundary(n) && name[n..].eq_ignore_ascii_case(".exe") => &name[..n],
        _ => name,
    };
    SESSION.iter().any(|s| s.eq_ignore_ascii_case(name))
}

static KUSER_FOLD_ENTRY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// With [`set_kuser_fold`]: fold the page after KUSER_SHARED_DATA too (64-bit
/// Wine's entry page, which x86-64 system call stubs call through by
/// constant), to the page after [`WINE32_KUSER`]. In the unit key.
pub fn set_kuser_fold_entry(on: bool) {
    KUSER_FOLD_ENTRY.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether [`fold_kuser`] covers the entry page ([`set_kuser_fold_entry`]).
pub fn kuser_fold_entry() -> bool {
    KUSER_FOLD_ENTRY.load(core::sync::atomic::Ordering::Relaxed)
}

/// One past the pages [`fold_kuser`] moves.
pub fn kuser_fold_end() -> u64 {
    if kuser_fold_entry() {
        KUSER_WINDOWS_END + 0x1000
    } else {
        KUSER_WINDOWS_END
    }
}

/// Where this process has KUSER_SHARED_DATA: [`WINE32_KUSER`] when folding is on, else Windows' address.
pub fn kuser_address() -> u64 {
    if kuser_fold() {
        WINE32_KUSER
    } else {
        KUSER_WINDOWS
    }
}

/// A constant effective address in Windows' KUSER_SHARED_DATA page (and,
/// with [`set_kuser_fold_entry`], the next), moved to 32-bit Wine's
/// ([`WINE32_KUSER`]) when folding is on; `None` for any other address.
pub fn fold_kuser(ea: u64) -> Option<u64> {
    (kuser_fold() && (KUSER_WINDOWS..kuser_fold_end()).contains(&ea)).then(|| ea - KUSER_WINDOWS + WINE32_KUSER)
}

/// With folding on, the bounds guard's limit (tier-1 and gex-cgen): an
/// access at or above it is one the translator did not fold -- a pointer
/// into the old page computed at run time -- and becomes
/// [`imports::BRK_KUSER_UNFOLDED`]: a named diagnostic and an access
/// violation, never a read of whatever is mapped there and never a bare
/// trap. Nothing of a 32-bit Wine process is mapped this high (its `mmap` top).
pub const KUSER_GUARD: u64 = KUSER_WINDOWS;

/// [`imports::BRK_KUSER_UNFOLDED`]'s `pc` argument: the block's guest pc in
/// the low half, the effective address's low 32 bits in the high half (an
/// i386 guest's pcs and addresses are both 32-bit).
pub fn kuser_miss_pack(pc: u64, ea: u64) -> i64 {
    ((pc & 0xffff_ffff) | (ea << 32)) as i64
}

/// [`kuser_miss_pack`] undone: `(pc, address)`.
pub fn kuser_miss_unpack(arg: i64) -> (u64, u64) {
    let a = arg as u64;
    (a & 0xffff_ffff, a >> 32)
}

const _: () = assert!(WINE32_KUSER + 0x1_0000 <= STACK_TOP - STACK_SIZE);
const _: () = assert!(WINE32_KUSER % 0x1_0000 == 0);

/// Size in bytes zeroed by one `dc zva`, and the value reported by
/// `DCZID_EL0`. 64 bytes is what every mainstream aarch64 core reports.
pub const DCZID_BLOCK_BYTES: u64 = 64;

/// `CTR_EL0` value reported to the guest: 64-byte I- and D-cache lines,
/// with the "instruction cache is not coherent" behaviour that makes guests
/// emit the `IC IVAU` sequences gex relies on for SMC detection.
pub const CTR_EL0: u64 = 0x8444_c004;

/// `CNTFRQ_EL0`: virtual counter frequency in Hz. 1 GHz keeps the guest's
/// arithmetic exact against a nanosecond host clock.
pub const CNTFRQ: u64 = 1_000_000_000;

/// Table slot for a guest address, given its page's directory entry. Masks off
/// the table index, so a caller holding only the base (table 0) can pass it.
#[inline]
pub const fn dispatch_index(pc: u64, page_entry: u32) -> u64 {
    dispatch_entry_base(page_entry) as u64 + ((pc % DISPATCH_PAGE) >> DISPATCH_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The context layout is a compile-time contract between codegen, the
    // kernel and the host, so it is checked at compile time.
    const _: () = {
        assert!(ctx::x(0) == 0);
        assert!(ctx::x(30) == 240);
        assert!(ctx::PC > ctx::x(30));
        assert!(ctx::NF > ctx::PC);
        assert!(ctx::TPIDR > ctx::VF);
        assert!(ctx::V >= ctx::LAST_SYSCALL + 8);
        assert!(ctx::X87 == ctx::v(31) + 16);
        assert!(ctx::X87 % 16 == 0 && ctx::X87_TMP % 16 == 0);
        assert!(!X86 || ctx::SIZE >= ctx::SCRATCH + 64);
        // An AArch64 context's register file ends with `v31`, and the
        // structure with the system-call immediate slot after it (every ISA's
        // does; see [`ctx::SYSCALL_IMM`]).
        assert!(X86 || ctx::SIZE_REGS == ctx::v(31) + 16);
        assert!(ctx::SYSCALL_IMM == ctx::SIZE_REGS && ctx::SIZE == ctx::SYSCALL_IMM + 16);
        assert!(ctx::SIZE % 8 == 0);
        // The vector slots must be 16-byte aligned for `v128` accesses.
        assert!(ctx::V % 16 == 0);
    };

    #[test]
    fn guest_memory_leaves_a_trap_page() {
        assert!(GUEST_MEM_BYTES.is_multiple_of(WASM_PAGE));
        assert!((POISON_ADDR as u64) >= GUEST_MEM_BYTES);
        assert!((POISON_ADDR as u64) + 16 <= u32::MAX as u64 + 1);
    }

    #[test]
    fn layout_regions_are_ordered() {
        assert!(RESERVED_BASE < RESERVED_END);
        // Runtime state and the dispatch directory must both stay clear of
        // the fixed non-PIE link address.
        assert!(RESERVED_END <= DISPATCH_DIR);
        assert!(DISPATCH_DIR_END <= EXEC_BASE);
        // The fixed runtime words above the contexts do not overlap.
        assert!(PATCHED_PAGES + PATCHED_PAGES_BYTES <= NEST_EXITS);
        assert!(BODY_LIMITS + 4 * DISPATCH_TABLES as u64 <= NEST_EXITS);
        assert!(PATCHED_PAGES + PATCHED_PAGES_BYTES <= BODY_LIMITS);
        assert!(NEST_EXITS + 8 <= SPLIT_LOCK);
        assert!(NEST_EXITS + 8 <= BOOT_STACK_COUNTER && BOOT_STACK_COUNTER + 8 <= SPLIT_LOCK);
        assert!(PATCHED_PAGES + PATCHED_PAGES_BYTES <= BODY_LIMIT);
        assert!(FCACHE + FCACHE_BYTES <= PATCHED_PAGES);
        assert!(FCACHE % 8 == 0 && FCACHE > RESERVED_BASE + 256 * ((ctx::SIZE as u64 + 63) & !63));
        assert!(SPLIT_LOCK + 8 <= EXCLUSIVE_SCRATCH);
        // Every address code may occupy needs a directory entry, including the
        // mmap arena where `ld.so` maps and calls into libraries.
        assert!(DISPATCH_MAX_PAGES * DISPATCH_PAGE >= STACK_TOP);
        assert!(DISPATCH_MAX_PAGES * DISPATCH_PAGE >= MMAP_TOP);
        assert!(EXEC_BASE < PIE_BASE);
        assert!(MMAP_BOTTOM < MMAP_TOP);
        // The stack sits at the top of the fixed low region, and the arena
        // starts above it and grows upward.
        assert!(PIE_BASE < INTERP_BASE);
        assert!(INTERP_BASE + 0x0010_0000 <= STACK_TOP - STACK_SIZE);
        assert!(INTERP_BASE < STACK_TOP);
        assert!(STACK_TOP <= MMAP_BOTTOM);
        assert!(MMAP_TOP <= GUEST_LIMIT);
        // The M64 tables sit above every M32 address and the image zone
        // `[4 GiB, 8 GiB)`, below the arena.
        assert!(DISPATCH_DIR64 >= GUEST_LIMIT && DISPATCH_DIR64 >= 8 << 30);
        assert!(DISPATCH_MAX_PAGES64 * DISPATCH_PAGE == LIMIT64);
        assert!(PATCHED_PAGES64 + DISPATCH_MAX_PAGES64 / 8 <= HIGH_TABLES_END);
        assert!(HIGH_ARENA_BASE < LIMIT64 && HIGH_ARENA_BASE % WASM_PAGE == 0);
    }
}
