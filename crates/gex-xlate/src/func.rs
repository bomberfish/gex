//! Function recovery: from a loaded image to function-level GIR.
//!
//! Where functions start ([`Source`]):
//!
//! * strong starts, which bound functions: the ELF entry point, `STT_FUNC`
//!   symbols, `.eh_frame` FDEs, direct call targets, `DT_INIT_ARRAY`,
//!   relocation-typed code pointers (`RELATIVE`/`IRELATIVE` addends and
//!   absolute relocations landing in code);
//! * weak starts, which only become functions when no function already
//!   covers them, and otherwise become interior entries
//!   ([`Function::entries`]): untyped data words that point into code,
//!   `adr`/`adrp+add`/`lea rip` in code, and C++ landing pads.
//!
//! A function's body is everything reachable from its start by intra-
//! procedural control flow that does not reach another strong start: a
//! branch to another strong start is a tail call, and running off the end
//! into one is a tail call too. Code two functions both reach without it
//! being a start (a shared tail, a `.cold` part's jump back) is duplicated
//! into each. Blocks are split at every leader, so a function's CFG has no
//! overlapping blocks (except x86 code that jumps into the middle of an
//! instruction, which gets a block of its own).
//!
//! Calls to functions that never return have no continuation; whether a
//! function returns is a least fixpoint over the call graph. Jump tables are
//! resolved on the SSA form of the function ([`crate::jumptable`]); what
//! cannot be resolved is left [`FTerm::Unresolved`].
//!
//! Recovery never decides correctness: every address the function tier does
//! not own stays tier-1's. What it decides is how much of
//! the program the function tier sees whole.

use crate::arch::{self, BlockLimits, FetchBuf, Lifted};
use crate::ehframe;
use crate::CodeView;
use gex_abi::INST_ALIGN;
use gex_ir::func::*;
use gex_ir::{pass, Block, Expr, GuestReg, Stmt, Term, Value};
use gex_ld::Image;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Where a function start came from. A start can have several.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Source {
    Entry,
    Symbol,
    EhFrame,
    Call,
    InitArray,
    Reloc,
    /// An untyped data word pointing into code (weak).
    DataPtr,
    /// A code address materialised by an instruction (weak).
    CodePtr,
    /// A C++ landing pad (weak; an interior entry of its function).
    LandingPad,
    /// A weak start no recovered function covers, promoted to a function
    /// of its own.
    Orphan,
    /// A PE import thunk: `jmp [IAT slot]`.
    ImportThunk,
    /// A PE base relocation whose target is code (weak: in a PE32 every
    /// absolute address is relocated, jump-table cases and labels inside
    /// functions included).
    PeReloc,
}

impl Source {
    /// Strong sources bound functions; weak ones only fill gaps.
    pub fn strong(self) -> bool {
        !matches!(
            self,
            Source::DataPtr | Source::CodePtr | Source::LandingPad | Source::PeReloc
        )
    }
    fn bit(self) -> u16 {
        1 << self as u16
    }
    pub const ALL: [Source; 12] = [
        Source::Entry,
        Source::Symbol,
        Source::EhFrame,
        Source::Call,
        Source::InitArray,
        Source::Reloc,
        Source::DataPtr,
        Source::CodePtr,
        Source::LandingPad,
        Source::Orphan,
        Source::ImportThunk,
        Source::PeReloc,
    ];
}

/// What is known about a start address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StartInfo {
    sources: u16,
    /// A symbol naming it.
    pub name: Option<String>,
    /// The extent the symbol table or an FDE gives, when either does.
    pub end: Option<u64>,
    /// For a landing pad: the function whose FDE names it.
    pub pad_of: Option<u64>,
}

impl StartInfo {
    pub fn has(&self, s: Source) -> bool {
        self.sources & s.bit() != 0
    }
    pub fn add(&mut self, s: Source) {
        self.sources |= s.bit();
    }
    pub fn remove(&mut self, s: Source) {
        self.sources &= !s.bit();
    }
    pub fn sources(&self) -> Vec<Source> {
        Source::ALL.iter().copied().filter(|&s| self.has(s)).collect()
    }
    pub fn strong(&self) -> bool {
        Source::ALL.iter().any(|&s| s.strong() && self.has(s))
    }
}

/// Knobs for [`recover`].
#[derive(Clone, Debug)]
pub struct RecoverOptions {
    pub limits: BlockLimits,
    /// Rounds of the noreturn / jump-table / interior-entry loop.
    pub max_rounds: usize,
    /// Largest jump table resolved.
    pub max_table: u32,
    /// Scan data for untyped code pointers (weak starts).
    pub scan_data: bool,
    /// Times to retry unresolved jump tables with call-graph summaries.
    pub summary_passes: usize,
}

impl Default for RecoverOptions {
    fn default() -> Self {
        RecoverOptions {
            limits: BlockLimits::default(),
            max_rounds: 8,
            max_table: 4096,
            scan_data: true,
            summary_passes: 3,
        }
    }
}

/// How well recovery did, for `gexc funcs` and the tests.
#[derive(Clone, Debug, Default)]
pub struct RecoverStats {
    pub functions: usize,
    pub blocks: usize,
    /// Distinct `STT_FUNC` symbol addresses in code.
    pub symbols: usize,
    /// ... of which a recovered function starts at.
    pub symbols_found: usize,
    /// FDEs, and how many a function starts at.
    pub fdes: usize,
    pub fdes_found: usize,
    /// Bytes inside symbol extents, and how many some block covers.
    pub symbol_bytes: u64,
    pub symbol_bytes_covered: u64,
    /// Bytes inside FDE ranges, and how many some block covers: the same
    /// for a stripped binary, whose unwind tables still bound every
    /// compiled function.
    pub fde_bytes: u64,
    pub fde_bytes_covered: u64,
    /// Switches recovered: proved complete, and best effort.
    pub switches_complete: usize,
    pub switches_partial: usize,
    /// Indirect jumps left unresolved.
    pub unresolved: usize,
    /// Calls known not to return.
    pub noreturn_calls: usize,
    /// Interior entries.
    pub interior_entries: usize,
    /// Rounds the fixpoint took.
    pub rounds: usize,
}

/// Everything recovered from one image.
pub struct Program {
    /// Recovered functions, by entry address, in memory form.
    pub functions: BTreeMap<u64, Function>,
    /// Every start considered, with where it came from.
    pub starts: BTreeMap<u64, StartInfo>,
    /// Functions proved never to return.
    pub noreturn: BTreeSet<u64>,
    pub stats: RecoverStats,
}

// ---------------------------------------------------------------------------
// Code access
// ---------------------------------------------------------------------------

/// Read access to the image for recovery: instruction bytes gated on the
/// code sections, and data reads that say whether the memory is writable.
pub struct Code<'a> {
    src: Src<'a>,
    /// Executable sections read from the file when the loader did not
    /// (ELF32): they bound code more tightly than the segments.
    ranges: Vec<(u64, u64)>,
}

/// Where [`Code`] reads from. Tagged (`repr(u8)`), not niche-encoded: the
/// browser shim's guest memory starts at linear address 0, so `mem`'s
/// pointer is null there, and with the niche a `Memory` read as an `Image`
/// (every fetch empty, so every function JIT translation one `illegal`).
#[repr(u8)]
enum Src<'a> {
    Image { image: &'a Image, view: CodeView<'a> },
    /// Guest memory at run time (the function JIT): what
    /// is executable now, read as it is now.
    Memory { mem: &'a [u8], exec: &'a dyn Fn(u64) -> bool, readonly: &'a dyn Fn(u64, u64) -> bool },
}

impl<'a> Code<'a> {
    pub fn new(image: &'a Image) -> Code<'a> {
        Code {
            src: Src::Image { image, view: CodeView::new(image) },
            ranges: Vec::new(),
        }
    }

    /// Guest memory as it is at run time; `exec` says what the guest may
    /// execute, `readonly(addr, len)` which data cannot change: only that
    /// is read (a jump table, a constant pool).
    pub fn memory(mem: &'a [u8], exec: &'a dyn Fn(u64) -> bool, readonly: &'a dyn Fn(u64, u64) -> bool) -> Code<'a> {
        Code { src: Src::Memory { mem, exec, readonly }, ranges: Vec::new() }
    }

    /// The image, for an image's code.
    pub fn image(&self) -> Option<&'a Image> {
        match self.src {
            Src::Image { image, .. } => Some(image),
            Src::Memory { .. } => None,
        }
    }
    /// [`Code::new`], taking the executable sections from the raw file when
    /// the image has none.
    pub fn with_elf(image: &'a Image, elf: Option<&[u8]>) -> Code<'a> {
        let mut c = Code::new(image);
        if image.code.is_empty() {
            if let Some(d) = elf {
                c.ranges = ehframe::code_ranges(d, image.bias);
            }
        }
        c
    }
    pub fn fetch(&self, pc: u64, buf: &mut FetchBuf) -> usize {
        match &self.src {
            Src::Image { view, .. } => view.fetch_code(pc, buf),
            Src::Memory { mem, exec, .. } => {
                if pc % INST_ALIGN != 0 || !exec(pc) || pc as usize + INST_ALIGN as usize > mem.len() {
                    return 0;
                }
                // Up to the first byte that is not executable.
                let off = pc as usize;
                let mut n = (mem.len() - off).min(16);
                for k in INST_ALIGN as usize..n {
                    if !exec(pc + k as u64) {
                        n = k;
                        break;
                    }
                }
                buf[..n].copy_from_slice(&mem[off..off + n]);
                n
            }
        }
    }
    pub fn is_code(&self, a: u64) -> bool {
        a % INST_ALIGN == 0
            && match &self.src {
                Src::Memory { mem, exec, .. } => (a as usize) < mem.len() && exec(a),
                Src::Image { view, .. } => {
                    if self.ranges.is_empty() {
                        view.is_executable(a)
                    } else {
                        self.ranges.iter().any(|&(lo, hi)| a >= lo && a < hi)
                    }
                }
            }
    }
    /// `bytes` little-endian bytes at `addr`, and whether the segment holding
    /// them is writable (its contents could change at run time).
    pub fn read(&self, addr: u64, bytes: u32) -> Option<(u64, bool)> {
        let n = bytes as u64;
        let image = match self.src {
            Src::Image { image, .. } => image,
            Src::Memory { mem, readonly, .. } => {
                // Only what cannot change: data the program wrote by now
                // (a bytecode array, say) would fold into a constant target
                // that is not one; ahead of time it reads as the file has it.
                if !readonly(addr, n) {
                    return None;
                }
                let o = addr as usize;
                let b = mem.get(o..o.checked_add(bytes as usize)?)?;
                let mut buf = [0u8; 8];
                buf[..bytes as usize].copy_from_slice(b);
                return Some((u64::from_le_bytes(buf), false));
            }
        };
        for s in &image.segments {
            if addr >= s.vaddr && addr.checked_add(n)? <= s.vaddr + s.bytes.len() as u64 {
                let o = (addr - s.vaddr) as usize;
                let mut buf = [0u8; 8];
                buf[..bytes as usize].copy_from_slice(&s.bytes[o..o + bytes as usize]);
                return Some((u64::from_le_bytes(buf), s.writable));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------

/// Collect start candidates. `elf` is the raw file, for `.eh_frame`; `None`
/// skips it.
pub fn seeds(
    code: &Code,
    elf: Option<&[u8]>,
    scan_data: bool,
) -> (BTreeMap<u64, StartInfo>, Vec<ehframe::Fde>) {
    let image = code.image().expect("seeds come from an image");
    let mut starts: BTreeMap<u64, StartInfo> = BTreeMap::new();
    let add = |starts: &mut BTreeMap<u64, StartInfo>, a: u64, s: Source| {
        if code.is_code(a) {
            starts.entry(a).or_default().add(s);
        }
    };
    if image.entry != image.bias {
        add(&mut starts, image.entry, Source::Entry);
    }
    // gex-ld reads ELF64 symbol tables only.
    let syms32 = match elf {
        Some(d) if image.class32 && image.symbols.is_empty() => ehframe::symbols32(d, image.bias),
        _ => Vec::new(),
    };
    for sym in image.symbols.iter().chain(&syms32) {
        if !code.is_code(sym.value) {
            continue;
        }
        let e = starts.entry(sym.value).or_default();
        e.add(Source::Symbol);
        // Prefer a name without a leading underscore-dot (compiler locals).
        let better = match &e.name {
            None => true,
            Some(n) => n.contains('.') && !sym.name.contains('.'),
        };
        if better {
            e.name = Some(sym.name.clone());
        }
        if sym.size > 0 {
            e.end = Some(e.end.unwrap_or(0).max(sym.value + sym.size));
        }
    }
    let fdes = elf.map(|d| ehframe::fdes(d, image.bias)).unwrap_or_default();
    for f in &fdes {
        if code.is_code(f.start) {
            let e = starts.entry(f.start).or_default();
            e.add(Source::EhFrame);
            e.end = Some(e.end.unwrap_or(0).max(f.end));
        }
        if let Some(l) = f.lsda {
            for lp in ehframe::landing_pads(image, l, f.start) {
                add(&mut starts, lp, Source::LandingPad);
                if code.is_code(lp) {
                    starts.get_mut(&lp).unwrap().pad_of = Some(f.start);
                }
            }
        }
    }
    for &a in &image.init_array {
        add(&mut starts, a, Source::InitArray);
    }
    for r in &image.relocs {
        use gex_ld::elf::reloc;
        let target = match r.kind {
            k if k == reloc::RELATIVE || k == reloc::IRELATIVE => {
                image.bias.wrapping_add(r.addend as u64)
            }
            _ => continue,
        };
        add(&mut starts, target, Source::Reloc);
    }
    for &(_, resolver) in &image.irelative {
        add(&mut starts, resolver, Source::Reloc);
    }
    if scan_data {
        // Untyped pointers in data (not in code: literal pools are data too,
        // but a code word that happens to look like an address is noise).
        for seg in &image.segments {
            if seg.executable {
                continue;
            }
            let w = arch::ptr_bytes() as usize;
            for (i, chunk) in seg.bytes.chunks_exact(w).enumerate() {
                let at = seg.vaddr + (i * w) as u64;
                if at % w as u64 != 0 {
                    continue;
                }
                let mut b = [0u8; 8];
                b[..w].copy_from_slice(chunk);
                add(&mut starts, u64::from_le_bytes(b), Source::DataPtr);
            }
        }
        for seg in &image.segments {
            if !seg.executable {
                continue;
            }
            let mut found = Vec::new();
            arch::scan_code_pointers(&seg.bytes, seg.vaddr, &|a| code.is_code(a), &mut |t| {
                found.push(t)
            });
            for t in found {
                add(&mut starts, t, Source::CodePtr);
            }
        }
    }
    (starts, fdes)
}

// ---------------------------------------------------------------------------
// Lifting
// ---------------------------------------------------------------------------

/// Lifted blocks, each lifted once at its natural length.
struct Lifter<'a> {
    code: &'a Code<'a>,
    limits: BlockLimits,
    cache: HashMap<u64, std::rc::Rc<Lifted>>,
}

impl<'a> Lifter<'a> {
    fn get(&mut self, pc: u64) -> std::rc::Rc<Lifted> {
        if let Some(l) = self.cache.get(&pc) {
            return l.clone();
        }
        let code = self.code;
        let l = std::rc::Rc::new(arch::lift_insts(
            &|p, b| code.fetch(p, b),
            pc,
            self.limits,
            &|_| false,
        ));
        self.cache.insert(pc, l.clone());
        l
    }
}

/// Cut a lifted block short so it ends just before instruction `at`, with a
/// fall-through to it. Exact because nothing was optimised across
/// instructions: the statements of the instructions kept are what they
/// were, and they wrote every register and flag they set to the context.
fn cut(l: &Lifted, at: u64) -> Lifted {
    let keep = l.pcs.iter().take_while(|&&p| p < at).count();
    let mut block = Block {
        start: l.block.start,
        end: at,
        stmts: l.block.stmts[..keep].to_vec(),
        term: Term::Goto { target: at },
        n_values: l.block.n_values,
        call_return: None,
        guard: None,
    };
    block.stmts.truncate(keep);
    Lifted {
        block,
        pcs: l.pcs[..keep].to_vec(),
        insts: l.insts.iter().copied().take_while(|&p| p < at).collect(),
        term_pc: at,
    }
}

// ---------------------------------------------------------------------------
// Body discovery
// ---------------------------------------------------------------------------

/// Where an edge goes: a block of this function, or out of it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Tgt {
    In(u64),
    Out(u64),
}

#[derive(Clone, Debug)]
enum DTerm {
    Jump(Tgt),
    Branch {
        cond: Value,
        taken: Tgt,
        not_taken: Tgt,
    },
    Call {
        callee: Callee,
        ret_pc: u64,
        ret: Option<Tgt>,
    },
    TailCall(Callee),
    Return(Value),
    Indirect(Value),
    Switch {
        target: Value,
        cases: Vec<(u64, Tgt)>,
        table: JumpTable,
    },
    Syscall {
        next_pc: u64,
        next: Option<Tgt>,
        imm: u32,
    },
    Illegal {
        raw: u32,
        pc: u64,
    },
}

struct Draft {
    l: Lifted,
    term: DTerm,
}

/// A resolved jump table, keyed by the address of the block ending in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub cases: Vec<u64>,
    pub table: JumpTable,
}

/// Whether a jump table resolved again replaces the one a site has: only
/// when it has more cases, or the same ones now proven complete. A table
/// resolved on the CFG it added to can come out smaller (SDL2.dll's
/// 0x68450: 84 cases, then 1, then 84 again), and letting it shrink made the
/// fixpoint alternate until the round limit, before weak starts were ever
/// promoted -- SDL2 kept 754 of its functions. Only growing, it converges.
/// Keeping the larger table is safe: a unit's switch sends any other
/// target through the function table.
fn replaces(old: Option<&Table>, new: &Table) -> bool {
    match old {
        None => true,
        Some(o) if o == new => false,
        Some(o) => new.cases.len() > o.cases.len() || (new.cases.len() == o.cases.len() && new.table.complete && !o.table.complete),
    }
}

/// Inputs to body discovery that the outer fixpoint refines.
struct Facts<'a> {
    starts: &'a BTreeMap<u64, StartInfo>,
    noreturn: &'a BTreeSet<u64>,
    /// Resolved tables by (function, jump instruction): a table is proved
    /// in the context of one function (its bound check dominates there),
    /// and code two functions share may be entered differently in each.
    tables: &'a HashMap<(u64, u64), Table>,
    /// Extra block leaders per function (interior entries).
    leaders: &'a HashMap<u64, BTreeSet<u64>>,
}

impl Facts<'_> {
    fn strong(&self, a: u64) -> bool {
        self.starts.get(&a).is_some_and(|s| s.strong())
    }
}

/// The constant the syscall-number register holds at the end of a block,
/// if the block sets it to one.
fn syscall_number(l: &Lifted) -> Option<u64> {
    let mut b = l.block.clone();
    pass::const_fold(&mut b);
    let v = b.stmts.iter().rev().find_map(|s| match s {
        Stmt::WriteReg { reg, val } if *reg == arch::SYSCALL_NR => Some(*val),
        _ => None,
    })?;
    pass::const_value(&b, v)
}

/// Discover the body of the function starting at `start`. Direct call
/// targets found on the way are returned for the caller to add as starts.
fn discover(
    start: u64,
    facts: &Facts,
    lifter: &mut Lifter,
    calls: &mut Vec<u64>,
) -> BTreeMap<u64, Draft> {
    let code = lifter.code;
    let extra = facts.leaders.get(&start);
    let mut leaders: BTreeSet<u64> = BTreeSet::new();
    leaders.insert(start);
    if let Some(e) = extra {
        leaders.extend(e.iter().copied());
    }
    let mut blocks: BTreeMap<u64, Draft> = BTreeMap::new();
    let mut work: Vec<u64> = vec![start];
    // Interior entries are blocks too, reached from outside.
    if let Some(e) = extra {
        work.extend(e.iter().copied());
    }
    // A target is inside the function unless it is another strong start.
    let classify = |t: u64| -> Tgt {
        if t == start || (!facts.strong(t) && code.is_code(t)) {
            Tgt::In(t)
        } else {
            Tgt::Out(t)
        }
    };
    while let Some(pc) = work.pop() {
        if blocks.contains_key(&pc) {
            continue;
        }
        // A new leader inside an existing block splits it.
        if let Some((&s, d)) = blocks.range(..pc).next_back() {
            if pc < d.l.block.end && d.l.insts.contains(&pc) {
                let cut_l = cut(&d.l, pc);
                blocks.insert(
                    s,
                    Draft {
                        l: cut_l,
                        term: DTerm::Jump(Tgt::In(pc)),
                    },
                );
            }
        }
        let full = lifter.get(pc);
        // Cut at the first later leader or strong start.
        let stop = full
            .insts
            .iter()
            .skip(1)
            .copied()
            .find(|&a| leaders.contains(&a) || facts.strong(a));
        let l = match stop {
            Some(a) => cut(&full, a),
            None => (*full).clone(),
        };
        let b = &l.block;
        let mut succ: Vec<u64> = Vec::new();
        let internal = |t: Tgt, succ: &mut Vec<u64>| {
            if let Tgt::In(a) = t {
                succ.push(a);
            }
            t
        };
        let term = match (b.term.clone(), b.call_return) {
            // `call 1f; 1: pop %ebx`, i386 PIC code's way to read its own
            // address: the call pushed the return address, and control
            // just continues. Not a call, and not a function start.
            (Term::Goto { target }, Some(r)) if target == r => DTerm::Jump(internal(classify(r), &mut succ)),
            (Term::Goto { target }, Some(r)) => {
                calls.push(target);
                let returns = !facts.noreturn.contains(&target);
                DTerm::Call {
                    callee: Callee::Direct(target),
                    ret_pc: r,
                    ret: if returns {
                        Some(internal(classify(r), &mut succ))
                    } else {
                        None
                    },
                }
            }
            (Term::Goto { target }, None) => {
                let t = classify(target);
                match t {
                    Tgt::In(_) => DTerm::Jump(internal(t, &mut succ)),
                    Tgt::Out(a) => DTerm::TailCall(Callee::Direct(a)),
                }
            }
            (Term::Branch {
                cond,
                taken,
                not_taken,
            }, _) => DTerm::Branch {
                cond: cond,
                taken: internal(classify(taken), &mut succ),
                not_taken: internal(classify(not_taken), &mut succ),
            },
            (Term::Indirect { target, .. }, Some(r)) => DTerm::Call {
                callee: Callee::Indirect(target),
                ret_pc: r,
                ret: Some(internal(classify(r), &mut succ)),
            },
            (Term::Indirect { target, is_ret: true }, None) => DTerm::Return(target),
            (Term::Indirect { target, is_ret: false }, None) => match facts.tables.get(&(start, l.term_pc)) {
                Some(t) => {
                    let mut cases: Vec<(u64, Tgt)> = t
                        .cases
                        .iter()
                        .map(|&c| (c, internal(classify(c), &mut succ)))
                        .collect();
                    cases.sort_by_key(|c| c.0);
                    cases.dedup_by_key(|c| c.0);
                    DTerm::Switch {
                        target: target,
                        cases,
                        table: t.table.clone(),
                    }
                }
                None => DTerm::Indirect(target),
            },
            (Term::Syscall { next, imm }, _) => {
                // The never-returns analysis reads a *register* number (the
                // Linux shape); an immediate-numbered call (an
                // `svc #imm`) leaves `x8` free to hold anything, so a fold of
                // it must not decide this call does not return.
                let never = imm == 0 && syscall_number(&l).is_some_and(arch::syscall_never_returns);
                DTerm::Syscall {
                    next_pc: next,
                    next: if never {
                        None
                    } else {
                        Some(internal(classify(next), &mut succ))
                    },
                    imm,
                }
            }
            (Term::Illegal { raw, pc }, _) => DTerm::Illegal { raw: raw, pc: pc },
        };
        for s in succ {
            if blocks.contains_key(&s) {
                continue;
            }
            leaders.insert(s);
            // Split an existing block containing it (checked when popped).
            work.push(s);
        }
        blocks.insert(pc, Draft { l, term });
        // Blocks lifted before this leader existed may span it.
        let spanning: Vec<u64> = blocks
            .iter()
            .filter(|(&s, d)| s < pc && pc < d.l.block.end && d.l.insts.contains(&pc))
            .map(|(&s, _)| s)
            .collect();
        for s in spanning {
            let d = blocks.get(&s).unwrap();
            let c = cut(&d.l, pc);
            blocks.insert(
                s,
                Draft {
                    l: c,
                    term: DTerm::Jump(Tgt::In(pc)),
                },
            );
        }
    }
    // Leaders found late may sit inside blocks lifted earlier: cut until
    // stable (each cut only shortens).
    loop {
        let starts: BTreeSet<u64> = blocks.keys().copied().collect();
        let mut fix: Option<(u64, u64)> = None;
        for (&s, d) in &blocks {
            if let Some(&a) = d.l.insts.iter().skip(1).find(|a| starts.contains(a)) {
                fix = Some((s, a));
                break;
            }
        }
        let Some((s, a)) = fix else { break };
        let d = blocks.get(&s).unwrap();
        let c = cut(&d.l, a);
        blocks.insert(
            s,
            Draft {
                l: c,
                term: DTerm::Jump(Tgt::In(a)),
            },
        );
    }
    blocks
}

/// Build the memory-form [`Function`] from discovered drafts.
fn assemble(start: u64, name: Option<String>, drafts: &BTreeMap<u64, Draft>, entries: &[u64]) -> Function {
    // Block order: the entry, then the rest by address; synthetic exits
    // after, in the order they are needed.
    let mut order: Vec<u64> = vec![start];
    order.extend(drafts.keys().copied().filter(|&a| a != start));
    let id: HashMap<u64, BlockId> = order
        .iter()
        .enumerate()
        .map(|(i, &a)| (a, BlockId(i as u32)))
        .collect();
    let mut f = Function {
        entry: start,
        name,
        entries: entries.to_vec(),
        blocks: Vec::with_capacity(order.len()),
        n_values: 0,
        form: Form::Memory,
    };
    let mut exits: Vec<FBlock> = Vec::new();
    let base_exit = order.len() as u32;
    for &a in &order {
        let d = &drafts[&a];
        let base = f.n_values;
        let rn = |v: Value| Value(v.0 + base);
        f.n_values += d.l.block.n_values;
        let mut stmts = d.l.block.stmts.clone();
        for s in stmts.iter_mut() {
            for_each_use_mut(s, |v| *v = rn(*v));
            for_each_def_mut(s, |v| *v = rn(*v));
        }
        let term_pc = d.l.term_pc;
        let edge = |t: Tgt, exits: &mut Vec<FBlock>| -> Edge {
            match t {
                Tgt::In(x) => Edge::to(id[&x]),
                Tgt::Out(x) => {
                    let e = BlockId(base_exit + exits.len() as u32);
                    exits.push(FBlock::synthetic(
                        term_pc,
                        FTerm::TailCall {
                            callee: Callee::Direct(x),
                        },
                    ));
                    Edge::to(e)
                }
            }
        };
        let callee = |c: Callee| match c {
            Callee::Direct(x) => Callee::Direct(x),
            Callee::Indirect(v) => Callee::Indirect(rn(v)),
        };
        let term = match &d.term {
            DTerm::Jump(t) => match t {
                Tgt::In(_) => FTerm::Jump(edge(*t, &mut exits)),
                // A fall-through into another function: leave directly.
                Tgt::Out(x) => FTerm::TailCall {
                    callee: Callee::Direct(*x),
                },
            },
            DTerm::Branch {
                cond,
                taken,
                not_taken,
            } => FTerm::Branch {
                cond: rn(*cond),
                taken: edge(*taken, &mut exits),
                not_taken: edge(*not_taken, &mut exits),
            },
            DTerm::Call { callee: c, ret_pc, ret } => FTerm::Call {
                callee: callee(*c),
                ret_pc: *ret_pc,
                ret: ret.map(|t| edge(t, &mut exits)),
            },
            DTerm::TailCall(c) => FTerm::TailCall { callee: callee(*c) },
            DTerm::Return(v) => FTerm::Return { target: rn(*v) },
            DTerm::Indirect(v) => FTerm::Unresolved { target: rn(*v) },
            DTerm::Switch {
                target,
                cases,
                table,
            } => FTerm::Switch {
                target: rn(*target),
                cases: cases.iter().map(|&(a, t)| (a, edge(t, &mut exits))).collect(),
                table: table.clone(),
            },
            DTerm::Syscall { next_pc, next, imm } => FTerm::Syscall {
                next_pc: *next_pc,
                next: next.map(|t| edge(t, &mut exits)),
                imm: *imm,
            },
            DTerm::Illegal { raw, pc } => FTerm::Illegal { raw: *raw, pc: *pc },
        };
        f.blocks.push(FBlock {
            start: a,
            end: d.l.block.end,
            synthetic: false,
            params: Vec::new(),
            stmts,
            pcs: d.l.pcs.clone(),
            insts: d.l.insts.clone(),
            term,
            term_pc,
        });
    }
    f.blocks.extend(exits);
    f
}

// ---------------------------------------------------------------------------
// Noreturn
// ---------------------------------------------------------------------------

/// Whether each function may return: a least fixpoint. A function returns
/// if a `ret` is reachable from its entry, or a jump to code that may
/// return (an unknown target, an unresolved jump, a tail call to a function
/// that may). A call's continuation is followed only if the callee may
/// return. Unrecovered callees are assumed to return.
pub(crate) fn may_return(functions: &BTreeMap<u64, Function>) -> HashMap<u64, bool> {
    let mut ret: HashMap<u64, bool> = functions.keys().map(|&a| (a, false)).collect();
    let returns = |ret: &HashMap<u64, bool>, a: u64| ret.get(&a).copied().unwrap_or(true);
    loop {
        let mut changed = false;
        for (&a, f) in functions {
            if ret[&a] {
                continue;
            }
            let mut seen = vec![false; f.blocks.len()];
            let mut work = vec![BlockId(0)];
            let mut r = false;
            while let Some(b) = work.pop() {
                if seen[b.index()] {
                    continue;
                }
                seen[b.index()] = true;
                let blk = f.block(b);
                match &blk.term {
                    FTerm::Return { .. } | FTerm::Unresolved { .. } => r = true,
                    FTerm::TailCall { callee } => match callee {
                        Callee::Direct(t) => r |= returns(&ret, *t),
                        Callee::Indirect(_) => r = true,
                    },
                    FTerm::Switch { table, .. } if !table.complete => r = true,
                    FTerm::Call { callee, ret: Some(e), .. } => {
                        let goes_on = match callee {
                            Callee::Direct(t) => returns(&ret, *t),
                            Callee::Indirect(_) => true,
                        };
                        if goes_on {
                            work.push(e.to);
                        }
                        continue;
                    }
                    _ => {}
                }
                if r {
                    break;
                }
                for s in blk.succs() {
                    work.push(s);
                }
            }
            if r {
                ret.insert(a, true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    ret
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Recover every function of `image`. `elf` is the raw file (for
/// `.eh_frame`); `summary` gives callee effects to the jump-table resolver
/// when summaries exist (the second pass, see [`crate::summary::analyze`]).
pub fn recover(image: &Image, elf: Option<&[u8]>, opts: &RecoverOptions) -> Program {
    // i386 code decodes in compatibility mode: this thread's setting until
    // its next image, which summaries read back ([`arch::compat`]).
    crate::set_thread_compat(image.class32);
    if image.class32 && !gex_abi::X86 {
        eprintln!("gex-xlate: an i386 image needs the x86 build; recovering nothing");
        return Program {
            functions: BTreeMap::new(),
            starts: BTreeMap::new(),
            noreturn: BTreeSet::new(),
            stats: RecoverStats::default(),
        };
    }
    let code = Code::with_elf(image, elf);
    let (starts, fdes) = seeds(&code, elf, opts.scan_data);
    recover_seeded(&code, starts, fdes, opts)
}

/// Recover the function at `start` alone, from code read as it is now (the
/// function JIT at run time): discovery, then its
/// jump tables, rediscovered until both stand. `starts` are the function
/// entries known so far, which bound it (a jump to one is a tail call); the
/// call targets it makes are added. `noreturn` are callees known not to
/// return. No summaries, no interior entries, no symbols: code the
/// discovery reaches that belongs to a function nobody has called yet is
/// part of this one, which only duplicates it.
pub fn recover_one(
    code: &Code,
    start: u64,
    starts: &mut BTreeMap<u64, StartInfo>,
    noreturn: &BTreeSet<u64>,
    opts: &RecoverOptions,
) -> Function {
    starts.entry(start).or_default().add(Source::Call);
    let mut lifter = Lifter { code, limits: opts.limits, cache: HashMap::new() };
    let mut tables: HashMap<(u64, u64), Table> = HashMap::new();
    let mut failed: HashSet<(u64, u64)> = HashSet::new();
    let leaders: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    let mut f;
    let mut rounds = 0;
    loop {
        rounds += 1;
        let mut calls = Vec::new();
        let drafts = {
            let facts = Facts { starts, noreturn, tables: &tables, leaders: &leaders };
            discover(start, &facts, &mut lifter, &mut calls)
        };
        // A call target inside what was discovered splits it there.
        let mut changed = false;
        for c in calls {
            if code.is_code(c) && !starts.get(&c).is_some_and(|s| s.has(Source::Call)) {
                starts.entry(c).or_default().add(Source::Call);
                if c != start && drafts.values().any(|d| d.l.insts.contains(&c)) {
                    changed = true;
                }
            }
        }
        f = assemble(start, None, &drafts, &[]);
        let todo: Vec<(u64, u64)> = f
            .blocks
            .iter()
            .filter(|b| !b.synthetic)
            .filter(|b| match b.term {
                FTerm::Unresolved { .. } => !tables.contains_key(&(start, b.term_pc)) && !failed.contains(&(start, b.term_pc)),
                FTerm::Switch { .. } => tables.contains_key(&(start, b.term_pc)),
                _ => false,
            })
            .map(|b| (b.start, b.term_pc))
            .collect();
        if !todo.is_empty() {
            let at: Vec<u64> = todo.iter().map(|t| t.0).collect();
            let extent = starts.get(&start).and_then(|i| i.end).map(|e| (start, e));
            let found = crate::jumptable::resolve_all(&f, &at, code, opts.max_table, None, extent);
            for ((_, t), (_, a)) in found.into_iter().zip(todo) {
                let key = (start, a);
                match t {
                    Some(t) => {
                        if replaces(tables.get(&key), &t) {
                            tables.insert(key, t);
                            changed = true;
                        }
                    }
                    None => {
                        if tables.remove(&key).is_some() {
                            changed = true;
                        }
                        failed.insert(key);
                    }
                }
            }
        }
        if !changed || rounds >= opts.max_rounds {
            return f;
        }
    }
}

/// Seeds of a PE module beyond what [`seeds`] finds in any image, and its
/// `.pdata` ranges, which play the part of `.eh_frame` FDEs.
pub struct PeSeeds {
    pub starts: Vec<(u64, Source)>,
    pub fdes: Vec<ehframe::Fde>,
}

/// A PE image as recovery input: placed at `base` (relocated if it moves),
/// as an [`Image`] with one segment per section, its executable sections as
/// the code, its exports as symbols, and PE32 marked i386
/// ([`Image::class32`]); plus the seeds only a PE has: the entry point, TLS
/// callbacks, base relocations whose target is code, and `.pdata`.
pub fn pe_input(pe: &gex_ld::pe::PeImage, base: u64) -> Result<(Image, PeSeeds), String> {
    let mut img = pe.map(base).map_err(|e| e.to_string())?;
    // The import address tables hold, in the file, hint/name RVAs that the
    // loader overwrites with the imports' addresses. Read as constant data
    // (they usually sit in a read-only section) they made every `jmp [slot]`
    // thunk a one-entry jump table into the middle of the hint table, which
    // decodes as garbage and never returns -- so every call through a thunk
    // (a game's memset, memcpy...) was taken not to return, and its
    // continuation was left out of the function. Recovered with the slots
    // zeroed, a thunk's target is unknown: an indirect tail call.
    {
        let ptr = if pe.is64 { 8 } else { 4 };
        let mut slots: Vec<usize> = pe.imports.iter().flat_map(|d| d.symbols.iter().map(|s| s.iat_rva as usize)).collect();
        slots.extend(pe.delay_imports.iter().flat_map(|d| d.symbols.iter().map(|s| s.iat_rva as usize)));
        for o in slots {
            if let Some(b) = img.get_mut(o..o + ptr) {
                b.fill(0);
            }
        }
    }
    let img = img;
    let segments = pe.segments(base, &img);
    let code: Vec<(u64, u64)> = pe
        .sections
        .iter()
        .filter(|s| s.executable() && s.vsize.max(s.raw_size) > 0)
        .map(|s| (base + s.rva as u64, base + s.rva as u64 + s.vsize.max(s.raw_size) as u64))
        .collect();
    let text_range = (
        code.iter().map(|c| c.0).min().unwrap_or(base),
        code.iter().map(|c| c.1).max().unwrap_or(base),
    );
    let mut symbols = Vec::new();
    for e in pe.exports.entries.iter().filter(|e| e.rva != 0 && e.forwarder.is_none()) {
        let name = e.names.first().cloned().unwrap_or_else(|| format!("#{}", e.ordinal));
        symbols.push(gex_ld::Symbol {
            name,
            value: base + e.rva as u64,
            size: 0,
            ifunc: false,
        });
    }
    let entry = if pe.entry_rva != 0 { base + pe.entry_rva as u64 } else { base };
    let image = Image {
        soname: None,
        needed: Vec::new(),
        runpath: Vec::new(),
        bias: base,
        entry,
        code,
        segments,
        phdr_vaddr: base,
        phentsize: 0,
        phnum: 0,
        brk: base + pe.image_size as u64,
        text_range,
        tls: None,
        symbols,
        named_relocs: Vec::new(),
        relocs: Vec::new(),
        init_array: Vec::new(),
        dynsyms: Vec::new(),
        irelative: Vec::new(),
        pie: false,
        interp: None,
        class32: !pe.is64,
    };
    let ptr = if pe.is64 { 8 } else { 4 };
    let word = |va: u64| -> Option<u64> {
        let o = va.checked_sub(base)? as usize;
        let b = img.get(o..o + ptr)?;
        let mut w = [0u8; 8];
        w[..ptr].copy_from_slice(b);
        Some(u64::from_le_bytes(w))
    };
    let mut starts = Vec::new();
    if pe.entry_rva != 0 {
        starts.push((entry, Source::Entry));
    }
    // TLS callbacks: a null-terminated array of pointers (relocated).
    if let Some(t) = &pe.tls {
        if t.callbacks_va != 0 {
            let mut at = t.callbacks_va.wrapping_sub(pe.preferred_base).wrapping_add(base);
            for _ in 0..256 {
                match word(at) {
                    Some(0) | None => break,
                    Some(cb) => starts.push((cb, Source::InitArray)),
                }
                at += ptr as u64;
            }
        }
    }
    // Every relocated word that points into code is a code pointer
    // something takes: a function, a jump-table case, a handler -- unless
    // what it points at is a pointer itself. The operand of `call [slot]`
    // is relocated as well, and Microsoft's linker keeps such slots in
    // `.text` (the IAT, and lazily bound wrappers laid out `[slot][nops]
    // [stub]`). An address holding a relocated word, or inside an import
    // slot, is data: taken for a start, the slot's bytes decoded straight
    // into the stub after it and made that an interior block of a function
    // at the slot, so the stub had no function of its own and every call
    // through the slot left the unit.
    let held: std::collections::HashSet<u64> = pe
        .relocs
        .iter()
        .filter(|r| r.kind != gex_ld::pe::rel::ABSOLUTE)
        .map(|r| base + r.rva as u64)
        .collect();
    let mut slots: Vec<(u64, u64)> = pe
        .imports
        .iter()
        .flat_map(|d| d.symbols.iter())
        .chain(pe.delay_imports.iter().flat_map(|d| d.symbols.iter()))
        .map(|s| (base + s.iat_rva as u64, base + s.iat_rva as u64 + ptr as u64))
        .collect();
    slots.sort_unstable();
    let in_slot = |a: u64| {
        let i = slots.partition_point(|r| r.1 <= a);
        slots.get(i).is_some_and(|r| r.0 <= a)
    };
    for r in &pe.relocs {
        if r.kind == gex_ld::pe::rel::ABSOLUTE {
            continue;
        }
        if let Some(t) = word(base + r.rva as u64) {
            if held.contains(&t) || in_slot(t) {
                continue;
            }
            starts.push((t, Source::PeReloc));
        }
    }
    // Import thunks: `jmp [slot]` (PE32: `ff 25` and the slot's address;
    // PE32+: `ff 25` and its displacement from the next instruction) for a
    // slot of an IAT. Code jumps to them from everywhere, and a start keeps
    // them from being copied into every function that does.
    let mut iat: Vec<(u64, u64)> = Vec::new();
    for d in &pe.imports {
        for sym in &d.symbols {
            iat.push((base + sym.iat_rva as u64, base + sym.iat_rva as u64 + ptr as u64));
        }
    }
    for d in &pe.delay_imports {
        for sym in &d.symbols {
            iat.push((base + sym.iat_rva as u64, base + sym.iat_rva as u64 + ptr as u64));
        }
    }
    iat.sort_unstable();
    let in_iat = |a: u64| {
        let i = iat.partition_point(|r| r.1 <= a);
        iat.get(i).is_some_and(|r| r.0 <= a)
    };
    for sec in pe.sections.iter().filter(|s| s.executable()) {
        let lo = sec.rva as usize;
        let hi = (lo + sec.vsize.max(sec.raw_size) as usize).min(img.len());
        let mut o = lo;
        while o + 6 <= hi {
            if img[o] == 0xff && img[o + 1] == 0x25 {
                let d = u32::from_le_bytes(img[o + 2..o + 6].try_into().unwrap());
                let slot = if pe.is64 {
                    (base + o as u64 + 6).wrapping_add(d as i32 as i64 as u64)
                } else {
                    d as u64
                };
                if in_iat(slot) {
                    starts.push((base + o as u64, Source::ImportThunk));
                }
            }
            o += 1;
        }
    }
    let mut fdes: Vec<ehframe::Fde> = pe
        .pdata
        .iter()
        .filter(|f| f.end_rva > f.begin_rva)
        .map(|f| ehframe::Fde {
            start: base + f.begin_rva as u64,
            end: base + f.end_rva as u64,
            lsda: None,
        })
        .collect();
    // MinGW keeps DWARF CFI in an `.eh_frame` section.
    if let Some(s) = pe.sections.iter().find(|s| pe_section_name(pe, &s.name) == ".eh_frame") {
        let lo = s.rva as usize;
        let hi = (lo + s.vsize.max(s.raw_size) as usize).min(img.len());
        if lo < hi {
            fdes.extend(ehframe::parse_eh_frame(&img[lo..hi], base + s.rva as u64, pe.is64));
        }
    }
    Ok((image, PeSeeds { starts, fdes }))
}

/// A PE section's name, with a long name (`/N`: an offset into the COFF
/// string table, which MinGW uses for `.eh_frame` and the debug sections)
/// looked up.
fn pe_section_name(pe: &gex_ld::pe::PeImage, raw: &str) -> String {
    let Some(off) = raw.strip_prefix('/').and_then(|n| n.parse::<usize>().ok()) else {
        return raw.to_string();
    };
    let d = &pe.data;
    let rd32 = |o: usize| d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    let Some(lfanew) = rd32(0x3c) else { return raw.to_string() };
    let (Some(symtab), Some(nsyms)) = (rd32(lfanew + 12), rd32(lfanew + 16)) else {
        return raw.to_string();
    };
    let at = symtab + nsyms * 18 + off;
    match d.get(at..) {
        Some(rest) if symtab != 0 => {
            let n = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
            String::from_utf8_lossy(&rest[..n]).into_owned()
        }
        _ => raw.to_string(),
    }
}

/// [`recover`] for a PE module from [`pe_input`].
pub fn recover_pe(image: &Image, extra: &PeSeeds, opts: &RecoverOptions) -> Program {
    crate::set_thread_compat(image.class32);
    let code = Code::with_elf(image, None);
    let (mut starts, mut fdes) = seeds(&code, None, opts.scan_data);
    for &(a, src) in &extra.starts {
        if code.is_code(a) {
            starts.entry(a).or_default().add(src);
        }
    }
    for f in &extra.fdes {
        if code.is_code(f.start) {
            let e = starts.entry(f.start).or_default();
            e.add(Source::EhFrame);
            e.end = Some(e.end.unwrap_or(0).max(f.end));
        }
    }
    fdes.extend(extra.fdes.iter().cloned());
    recover_seeded(&code, starts, fdes, opts)
}

/// Recovery from seeds: the fixpoint of discovery, noreturn, jump tables
/// and interior entries.
fn recover_seeded(
    code: &Code,
    mut starts: BTreeMap<u64, StartInfo>,
    fdes: Vec<ehframe::Fde>,
    opts: &RecoverOptions,
) -> Program {
    let mut lifter = Lifter {
        code: &code,
        limits: opts.limits,
        cache: HashMap::new(),
    };
    let mut noreturn: BTreeSet<u64> = BTreeSet::new();
    let mut tables: HashMap<(u64, u64), Table> = HashMap::new();
    let mut leaders: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    let mut entries: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut failed_tables: HashSet<(u64, u64)> = HashSet::new();
    // Orphans a table claimed (made weak again), and those then promoted
    // again, which stay functions.
    let mut demoted: HashSet<u64> = HashSet::new();
    let mut pinned: HashSet<u64> = HashSet::new();
    let mut functions: BTreeMap<u64, Function>;
    let mut rounds = 0;
    // Callee summaries, once there are any: a jump table whose base sits in
    // a callee-saved register across calls resolves only when the calls are
    // known to preserve it. Summaries only get more precise as tables
    // resolve, so a table resolved with earlier ones stays valid.
    let mut sums: Option<crate::summary::Summaries> = None;
    let mut pass = 0;
    'passes: loop {
    let pass_rounds = rounds;
    loop {
        rounds += 1;
        // Discover every strong start, adding call targets as they appear.
        functions = BTreeMap::new();
        let mut drafts_of: BTreeMap<u64, BTreeMap<u64, Draft>> = BTreeMap::new();
        let mut work: Vec<u64> = starts
            .iter()
            .filter(|(_, s)| s.strong())
            .map(|(&a, _)| a)
            .collect();
        let mut done: HashSet<u64> = HashSet::new();
        while let Some(s) = work.pop() {
            if !done.insert(s) {
                continue;
            }
            let mut calls = Vec::new();
            let drafts = {
                let facts = Facts {
                    starts: &starts,
                    noreturn: &noreturn,
                    tables: &tables,
                    leaders: &leaders,
                };
                discover(s, &facts, &mut lifter, &mut calls)
            };
            for c in calls {
                if code.is_code(c) {
                    let e = starts.entry(c).or_default();
                    if !e.has(Source::Call) {
                        e.add(Source::Call);
                        if !done.contains(&c) {
                            work.push(c);
                        }
                    }
                }
            }
            drafts_of.insert(s, drafts);
        }
        // A call target found late may split functions discovered before it
        // was a start: rediscover those (cheap, lifting is cached).
        let strong: Vec<u64> = starts.iter().filter(|(_, s)| s.strong()).map(|(&a, _)| a).collect();
        let strong_set: HashSet<u64> = strong.iter().copied().collect();
        for (&s, drafts) in drafts_of.iter_mut() {
            let stale = drafts.iter().any(|(&a, d)| {
                a != s
                    && (strong_set.contains(&a)
                        || d.l.insts.iter().skip(1).any(|i| strong_set.contains(i)))
            });
            if stale {
                let mut calls = Vec::new();
                let facts = Facts {
                    starts: &starts,
                    noreturn: &noreturn,
                    tables: &tables,
                    leaders: &leaders,
                };
                *drafts = discover(s, &facts, &mut lifter, &mut calls);
            }
        }
        for (&s, drafts) in &drafts_of {
            let name = starts.get(&s).and_then(|i| i.name.clone());
            let ents = entries.get(&s).cloned().unwrap_or_default();
            functions.insert(s, assemble(s, name, drafts, &ents));
        }

        let mut changed = false;

        // Noreturn.
        let mr = may_return(&functions);
        let nr: BTreeSet<u64> = mr.iter().filter(|(_, &r)| !r).map(|(&a, _)| a).collect();
        if nr != noreturn {
            noreturn = nr;
            changed = true;
        }

        // Jump tables, on SSA.
        for f in functions.values() {
            // Keyed by (function, jump instruction): the jump survives block
            // splits.
            // A table is resolved on a CFG without its own cases, which can
            // loop back to it (a state machine): resolved, it is checked
            // again each round on the CFG with them, until it stands.
            let todo: Vec<(u64, u64)> = f
                .blocks
                .iter()
                .filter(|b| !b.synthetic)
                .filter(|b| match b.term {
                    FTerm::Unresolved { .. } => {
                        !tables.contains_key(&(f.entry, b.term_pc)) && !failed_tables.contains(&(f.entry, b.term_pc))
                    }
                    FTerm::Switch { .. } => tables.contains_key(&(f.entry, b.term_pc)),
                    _ => false,
                })
                .map(|b| (b.start, b.term_pc))
                .collect();
            if todo.is_empty() {
                continue;
            }
            let starts_todo: Vec<u64> = todo.iter().map(|t| t.0).collect();
            let extent = starts.get(&f.entry).and_then(|i| i.end).map(|e| (f.entry, e));
            let found = crate::jumptable::resolve_all(f, &starts_todo, &code, opts.max_table, sums.as_ref(), extent);
            for ((_, t), (_, a)) in found.into_iter().zip(todo) {
                let key = (f.entry, a);
                match t {
                    Some(t) => {
                        if replaces(tables.get(&key), &t) {
                            tables.insert(key, t);
                            changed = true;
                        }
                    }
                    None => {
                        // A table that no longer stands is dropped for good.
                        if tables.remove(&key).is_some() {
                            changed = true;
                        }
                        failed_tables.insert(key);
                    }
                }
            }
        }

        // A weak start promoted to a function because nothing covered it
        // goes back to weak once a jump table claims it as a case (an
        // absolute table's entries are data pointers).
        {
            let cases = switch_cases(&functions);
            for (a, st) in starts.iter_mut() {
                if st.has(Source::Orphan) && cases.contains(a) && !pinned.contains(a) {
                    st.remove(Source::Orphan);
                    demoted.insert(*a);
                    changed = true;
                }
            }
        }

        // Weak starts: interior entries of the functions covering them, or
        // functions of their own when nothing does. Once the rest has
        // settled (a jump-table case is uncovered until its table resolves,
        // and an absolute table's entries are data pointers to it), or in
        // the pass's last round when it does not settle: a large image's
        // tables can keep changing round after round, and the weak starts
        // (an x86-64 image's vtable entries among them) would otherwise
        // never be placed. What that round promotes gets one more round,
        // to be discovered.
        let last_round = rounds - pass_rounds == opts.max_rounds;
        let mut promoted = false;
        if !changed || last_round {
            let covered = coverage_index(&functions);
            let case_targets = switch_cases(&functions);
            let guessed = guessed_cases(&functions);
            // The code inside a switch's own block: `adr x10, .` anchors a
            // relative table there, and is not an entry anyone takes.
            let switch_blocks: Vec<(u64, u64)> = functions
                .values()
                .flat_map(|f| f.blocks.iter())
                .filter(|b| matches!(b.term, FTerm::Switch { .. }) && !b.synthetic)
                .map(|b| (b.start, b.end))
                .collect();
            let in_switch = |a: u64| switch_blocks.iter().any(|&(s, e)| a > s && a < e);
            let weak: Vec<(u64, bool, Option<u64>)> = starts
                .iter()
                .filter(|(_, s)| !s.strong())
                .map(|(&a, s)| {
                    let typed = s.has(Source::CodePtr) || s.has(Source::LandingPad) || s.has(Source::PeReloc);
                    (a, typed, s.pad_of)
                })
                .collect();
            for (w, typed, pad_of) in weak {
                if case_targets.contains(&w) || in_switch(w) {
                    continue;
                }
                // A pointer to it is relocated and a virtual call's tail
                // reaches it (`one_case_guess`): a function, whoever's
                // blocks it also is. A PE32+ relocated pointer into code
                // that another function's blocks cover (a direct tail jump
                // to it puts it in the jumper's) stays that function's
                // interior entry: as a function of its own, the jumper's
                // flow into it would become a call through the table. Calls
                // through the pointer reach it as a region entry, which the
                // unit build makes for a PE32+ image's interior entries.
                if typed && guessed.contains(&w) && !starts[&w].has(Source::Orphan) {
                    if plausible_start(&code, w, typed) {
                        starts.get_mut(&w).unwrap().add(Source::Orphan);
                        promoted = true;
                    }
                    continue;
                }
                let mut owners = covered.owners(w);
                // A landing pad belongs to the function whose FDE names it,
                // even though no branch there reaches it.
                if let Some(o) = pad_of.filter(|o| functions.contains_key(o)) {
                    if !owners.contains(&o) {
                        owners.push(o);
                    }
                }
                if owners.is_empty() {
                    if plausible_start(&code, w, typed) {
                        starts.get_mut(&w).unwrap().add(Source::Orphan);
                        // Promoted again after a demotion: the table that
                        // claims it only resolves while it is a function
                        // (as one, it splits a loop off the switch). Keep it
                        // one, or the two would alternate for ever.
                        if demoted.contains(&w) {
                            pinned.insert(w);
                        }
                        promoted = true;
                    }
                    continue;
                }
                for o in owners {
                    if leaders.entry(o).or_default().insert(w) {
                        entries.entry(o).or_default().push(w);
                        promoted = true;
                    }
                }
            }
            for v in entries.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
        }
        changed |= promoted;

        if !changed || (rounds - pass_rounds >= opts.max_rounds && !(last_round && promoted)) {
            break;
        }
    }
    // Retry what is still unresolved with callee summaries.
    let unresolved = functions
        .values()
        .flat_map(|f| f.blocks.iter())
        .any(|b| matches!(b.term, FTerm::Unresolved { .. }));
    if pass >= opts.summary_passes || !unresolved {
        break 'passes;
    }
    sums = Some(crate::summary::analyze_functions(&functions));
    failed_tables.clear();
    pass += 1;
    // Only another pass if the retry finds something: run the resolver
    // alone first.
    let mut found_any = false;
    for f in functions.values() {
        let todo: Vec<(u64, u64)> = f
            .blocks
            .iter()
            .filter(|b| matches!(b.term, FTerm::Unresolved { .. }))
            .map(|b| (b.start, b.term_pc))
            .collect();
        if todo.is_empty() {
            continue;
        }
        let starts_todo: Vec<u64> = todo.iter().map(|t| t.0).collect();
        let s = sums.as_ref().unwrap();
        let extent = starts.get(&f.entry).and_then(|i| i.end).map(|e| (f.entry, e));
        let found = crate::jumptable::resolve_all(f, &starts_todo, &code, opts.max_table, Some(s), extent);
        for ((_, t), (_, a)) in found.into_iter().zip(todo) {
            match t {
                Some(t) => {
                    tables.insert((f.entry, a), t);
                    found_any = true;
                }
                None => {
                    failed_tables.insert((f.entry, a));
                }
            }
        }
    }
    if !found_any {
        break 'passes;
    }
    }

    let stats = stats(&functions, &starts, &fdes, rounds);
    Program {
        functions,
        starts,
        noreturn,
        stats,
    }
}

/// Every case address of the switches the functions have now. (Not the
/// table cache: a table of a function that has since changed shape or gone
/// claims nothing.)
fn switch_cases(functions: &BTreeMap<u64, Function>) -> HashSet<u64> {
    functions
        .values()
        .flat_map(|f| f.blocks.iter())
        .filter_map(|b| match &b.term {
            FTerm::Switch { cases, table, .. } if !one_case_guess(table) => Some(cases.iter().map(|c| c.0)),
            _ => None,
        })
        .flatten()
        .collect()
}

/// The cases of [`one_case_guess`] switches.
fn guessed_cases(functions: &BTreeMap<u64, Function>) -> HashSet<u64> {
    functions
        .values()
        .flat_map(|f| f.blocks.iter())
        .filter_map(|b| match &b.term {
            FTerm::Switch { cases, table, .. } if one_case_guess(table) => Some(cases.iter().map(|c| c.0)),
            _ => None,
        })
        .flatten()
        .collect()
}

/// A "switch" of one case read through writable memory: a virtual call's
/// tail (`mov eax, [obj]; jmp [eax + 16]` with `obj` in `.data`) resolved
/// from the initial bytes, not a jump table. Its case is the callee, a
/// function of its own; it does not stop that start from being promoted
/// (otherwise the vtable functions such tails reach are blocks of the tail
/// and nobody's entry, and every call to them goes through the function
/// table's fallback). Read-only memory too: a `jmp [slot]` with no index
/// through a pointer kept in `.text` is a tail call all the same (a lazily
/// bound wrapper: the stub the slot holds would be a block of every
/// function that tail-jumps through it, and every call through the slot
/// would leave the unit).
fn one_case_guess(t: &gex_ir::func::JumpTable) -> bool {
    t.count == 1 && t.entry_bytes == 0
}

/// Whether an uncovered weak start is worth a function of its own: its
/// first instruction decodes, and on x86-64 (where any byte could start an
/// instruction) it is either materialised by code (`typed`) or 16-byte
/// aligned, as compilers align functions.
fn plausible_start(code: &Code, a: u64, typed: bool) -> bool {
    let mut buf = [0u8; 16];
    let n = code.fetch(a, &mut buf);
    if n == 0 || !arch::decodable(&buf[..n], a) {
        return false;
    }
    // A syscall stub (`int $0x80; ret`, `syscall; ret`): i386 musl's
    // `__vsyscall`, which only its `__sysinfo` pointer names.
    let stub = n >= 3 && (buf[..3] == [0xcd, 0x80, 0xc3] || buf[..3] == [0x0f, 0x05, 0xc3]);
    !gex_abi::X86 || typed || a % 16 == 0 || stub
}

/// Which functions cover an address at an instruction boundary.
struct Coverage {
    /// (start, end, function) for every lifted block, sorted.
    ranges: Vec<(u64, u64, u64)>,
    insts: HashMap<u64, Vec<u64>>,
}

impl Coverage {
    fn owners(&self, a: u64) -> Vec<u64> {
        self.insts.get(&a).cloned().unwrap_or_default()
    }
}

fn coverage_index(functions: &BTreeMap<u64, Function>) -> Coverage {
    let mut ranges = Vec::new();
    let mut insts: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&fa, f) in functions {
        for b in &f.blocks {
            if b.synthetic {
                continue;
            }
            ranges.push((b.start, b.end, fa));
            for &i in &b.insts {
                let v = insts.entry(i).or_default();
                if !v.contains(&fa) {
                    v.push(fa);
                }
            }
        }
    }
    ranges.sort_unstable();
    Coverage { ranges, insts }
}

fn stats(
    functions: &BTreeMap<u64, Function>,
    starts: &BTreeMap<u64, StartInfo>,
    fdes: &[ehframe::Fde],
    rounds: usize,
) -> RecoverStats {
    let mut s = RecoverStats {
        functions: functions.len(),
        rounds,
        ..Default::default()
    };
    for f in functions.values() {
        s.blocks += f.blocks.iter().filter(|b| !b.synthetic).count();
        s.interior_entries += f.entries.len();
        for b in &f.blocks {
            match &b.term {
                FTerm::Switch { table, .. } => {
                    if table.complete {
                        s.switches_complete += 1
                    } else {
                        s.switches_partial += 1
                    }
                }
                FTerm::Unresolved { .. } => s.unresolved += 1,
                FTerm::Call { ret: None, .. } => s.noreturn_calls += 1,
                _ => {}
            }
        }
    }
    let sym: BTreeMap<u64, u64> = starts
        .iter()
        .filter(|(_, i)| i.has(Source::Symbol))
        .map(|(&a, i)| (a, i.end.unwrap_or(a)))
        .collect();
    s.symbols = sym.len();
    s.symbols_found = sym.keys().filter(|a| functions.contains_key(a)).count();
    s.fdes = fdes.len();
    s.fdes_found = fdes.iter().filter(|f| functions.contains_key(&f.start)).count();
    // Byte coverage of symbol extents by lifted blocks.
    let cov = coverage_index(functions);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for &(a, b, _) in &cov.ranges {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let ext: Vec<(u64, u64)> = sym.iter().filter(|(a, e)| e > a).map(|(&a, &e)| (a, e)).collect();
    (s.symbol_bytes, s.symbol_bytes_covered) = covered_bytes(ext, &merged);
    let ext: Vec<(u64, u64)> = fdes.iter().filter(|f| f.end > f.start).map(|f| (f.start, f.end)).collect();
    (s.fde_bytes, s.fde_bytes_covered) = covered_bytes(ext, &merged);
    s
}

/// Bytes in the union of `ext`, and how many of them the sorted, disjoint
/// ranges `merged` cover.
fn covered_bytes(mut ext: Vec<(u64, u64)>, merged: &[(u64, u64)]) -> (u64, u64) {
    ext.sort_unstable();
    let mut ext_m: Vec<(u64, u64)> = Vec::new();
    for (a, b) in ext {
        match ext_m.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => ext_m.push((a, b)),
        }
    }
    let (mut total, mut covered) = (0, 0);
    for &(a, b) in &ext_m {
        total += b - a;
        let i = merged.partition_point(|m| m.1 <= a);
        for m in &merged[i..] {
            if m.0 >= b {
                break;
            }
            covered += m.1.min(b) - m.0.max(a);
        }
    }
    (total, covered)
}

/// SSA construction for this guest: [`gex_ir::ssa::to_ssa_sp`] with the
/// guest's stack pointer ([`arch::SP_REG`]), so a callee's
/// [`Summary::sp_delta`] carries the stack pointer across its calls.
pub fn to_ssa(f: &Function, call: &dyn Fn(Callee) -> gex_ir::ssa::CallEffect) -> Function {
    gex_ir::ssa::to_ssa_sp(f, call, arch::SP_REG)
}

/// The value a register holds according to a block's writes, for tests.
#[doc(hidden)]
pub fn last_write(b: &FBlock, r: GuestReg) -> Option<Value> {
    b.stmts.iter().rev().find_map(|s| match s {
        Stmt::WriteReg { reg, val } if *reg == r => Some(*val),
        _ => None,
    })
}

#[doc(hidden)]
pub fn is_const(f: &Function, v: Value) -> Option<u64> {
    for b in &f.blocks {
        for s in &b.stmts {
            if let Stmt::Def { dst, expr } = s {
                if *dst == v {
                    return match expr {
                        Expr::ConstI64(c) => Some(*c),
                        Expr::ConstI32(c) => Some(*c as u64),
                        _ => None,
                    };
                }
            }
        }
    }
    None
}

/// Coverage against an outside list of function starts (another tool's, a
/// symbol table recovery could not read): how many of `truth` are starts of
/// recovered functions, how many are covered by a block at all, and bytes
/// of the extents `[start, next start)` (within one code range) covered.
/// The last start of each code range has no known end and is left out.
/// Returns `(starts, functions, inside a block, bytes, bytes covered)`.
pub fn truth_coverage(prog: &Program, code: &[(u64, u64)], truth: &[u64]) -> (usize, usize, usize, u64, u64) {
    let mut t: Vec<u64> = truth.to_vec();
    t.sort_unstable();
    t.dedup();
    let cov = coverage_index(&prog.functions);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for &(a, b, _) in &cov.ranges {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let inside = |a: u64| {
        let i = merged.partition_point(|m| m.1 <= a);
        merged.get(i).is_some_and(|m| m.0 <= a)
    };
    let funcs = t.iter().filter(|a| prog.functions.contains_key(a)).count();
    let within = t.iter().filter(|&&a| inside(a)).count();
    let mut ext = Vec::new();
    for (i, &a) in t.iter().enumerate() {
        let Some(&(_, hi)) = code.iter().find(|&&(lo, hi)| a >= lo && a < hi) else { continue };
        match t.get(i + 1) {
            Some(&e) if e <= hi && e > a => ext.push((a, e)),
            _ => {}
        }
    }
    let (bytes, covered) = covered_bytes(ext, &merged);
    (t.len(), funcs, within, bytes, covered)
}

/// Bytes of the code ranges, and how many recovered blocks cover.
pub fn code_coverage(prog: &Program, code: &[(u64, u64)]) -> (u64, u64) {
    let cov = coverage_index(&prog.functions);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for &(a, b, _) in &cov.ranges {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    covered_bytes(code.to_vec(), &merged)
}

/// Byte ranges inside symbol extents that no recovered block covers, with
/// the symbol's start, for finding what recovery missed.
pub fn uncovered(prog: &Program) -> Vec<(u64, u64, u64)> {
    let cov = coverage_index(&prog.functions);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for &(a, b, _) in &cov.ranges {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut out = Vec::new();
    for (&a, s) in &prog.starts {
        if !s.has(Source::Symbol) {
            continue;
        }
        let Some(e) = s.end else { continue };
        let mut at = a;
        let i = merged.partition_point(|m| m.1 <= a);
        for m in &merged[i..] {
            if m.0 >= e {
                break;
            }
            if m.0 > at {
                out.push((at, m.0, a));
            }
            at = at.max(m.1);
        }
        if at < e {
            out.push((at, e, a));
        }
    }
    out
}
