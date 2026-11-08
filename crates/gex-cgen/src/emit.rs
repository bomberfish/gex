//! GIR block → C function, with tier-1's semantics statement for statement.
//!
//! Phase A of the function tier: every block is one
//! C function `void b_<addr>(int ctx)`, the uniform signature. The register
//! cache, the write-backs, the safepoints and the dispatch follow
//! `gex_emit::tier1` exactly; only the code generator differs. Where tier-1
//! emits a wasm instruction, this emits the C that clang lowers to it.

use anyhow::{bail, Result};
use gex_abi::{ctx, GUEST_LIMIT};
use gex_ir::func::RegSet;
use gex_ir::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write;

/// The C name of the function for a block at `addr`.
pub fn fn_name(addr: u64) -> String {
    format!("b_{addr:x}")
}

/// Refuse the tier-1 diagnostics the function tier does not reproduce, so a
/// run that asked for one does not silently go without it.
pub fn check_environment() -> Result<()> {
    for var in [
        "GEX_CHECK_SP",
        "GEX_TRAP_PC",
        "GEX_TRACE_BLOCKS",
        "GEX_COHERENT_LOADS",
        "GEX_COPY_LOOP",
        "GEX_NO_DIRECT_JUMP",
        "GEX_DISPATCH_BALLAST",
    ] {
        if std::env::var_os(var).is_some() {
            bail!("{var} is set, and the function tier (--cgen) does not implement it");
        }
    }
    Ok(())
}

fn ballast() -> usize {
    std::env::var("GEX_CGEN_BALLAST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// `GEX_CGEN_DISP=0`: guard every i386 address after its displacement, as
/// before (`Emitter::note_disp`).
fn disp_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GEX_CGEN_DISP").as_deref() == Ok("0"))
}

fn no_guard() -> bool {
    std::env::var_os("GEX_UNSAFE_NO_GUARD").is_some()
}

/// The null guard, a debugging opt-in (gex-emit `null_guard_on`).
fn null_on() -> bool {
    gex_emit::tier1::null_guard_on()
}

pub(crate) fn cty(ty: Ty) -> &'static str {
    match ty {
        Ty::I32 => "uint32_t",
        Ty::I64 => "uint64_t",
        Ty::V128 => "v128_t",
    }
}

/// Whether `r` lives in a 32-bit C variable: an i386 program's
/// general-purpose registers (the context slot is 64 bits and its low half
/// is the register, so loads read only that). Stores are 32-bit too, as
/// every writer zero-extends (`image::narrow_stores`). `GEX_CGEN_NARROW=0`
/// keeps them 64-bit. Arithmetic, addresses and compares then stay in i32, where
/// wasm needs no wraps and extends between them.
pub(crate) fn narrow_reg(r: GuestReg, class32: bool) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    class32
        && gex_abi::X86
        && matches!(r, GuestReg::X(n) if n < 8)
        && *ON.get_or_init(|| std::env::var("GEX_CGEN_NARROW").as_deref() != Ok("0"))
}

/// The C type of `r`'s variable.
pub(crate) fn reg_cty(r: GuestReg, class32: bool) -> &'static str {
    if narrow_reg(r, class32) {
        "uint32_t"
    } else {
        cty(r.ty())
    }
}

/// A register x86 code never names, which function mode uses as a local
/// only (`FnUnit::inline_leaf_calls` keeps an indirect call's target in it
/// for the fallback): never dirty, so never written back.
pub(crate) fn is_scratch(r: GuestReg) -> bool {
    gex_abi::X86 && r == GuestReg::MonAddr
}

pub(crate) fn reg_name(r: GuestReg) -> String {
    match r {
        GuestReg::X(n) => format!("r_x{n}"),
        GuestReg::V128(n) => format!("r_v{n}"),
        GuestReg::Sp => "r_sp".into(),
        GuestReg::Pc => "r_pc".into(),
        GuestReg::N => "f_n".into(),
        GuestReg::Z => "f_z".into(),
        GuestReg::C => "f_c".into(),
        GuestReg::V => "f_v".into(),
        GuestReg::P => "f_p".into(),
        GuestReg::A => "f_a".into(),
        GuestReg::D => "f_d".into(),
        GuestReg::Tpidr => "r_tpidr".into(),
        GuestReg::TpidrRo => "r_tpidrro".into(),
        GuestReg::MonAddr => "r_monaddr".into(),
        GuestReg::MonValue => "r_monvalue".into(),
        GuestReg::MonSize => "r_monsize".into(),
        GuestReg::Mxcsr => "r_mxcsr".into(),
        GuestReg::X87Top => "r_x87top".into(),
        GuestReg::X87Cw => "r_x87cw".into(),
        GuestReg::X87Sw => "r_x87sw".into(),
        GuestReg::X87Tags => "r_x87tags".into(),
        GuestReg::CtxAddr => "r_ctxaddr".into(),
    }
}

/// `tier1::Emitter::reg_offset`.
pub(crate) fn reg_offset(r: GuestReg) -> usize {
    match r {
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
        GuestReg::CtxAddr => 0,
    }
}

/// C for one translation unit: the header include, prototypes for every block
/// function referenced, and the bodies.
pub struct Unit {
    protos: BTreeSet<u64>,
    bodies: String,
}

impl Default for Unit {
    fn default() -> Self {
        Self::new()
    }
}

impl Unit {
    pub fn new() -> Unit {
        Unit {
            protos: BTreeSet::new(),
            bodies: String::new(),
        }
    }

    /// Add the C function for `block`. `owned` is every block address with a
    /// C body in this module, for direct `musttail` successors.
    pub fn add_block(&mut self, block: &Block, owned: &HashSet<u64>) -> Result<()> {
        let mut e = Emitter::new(block, owned);
        e.run()?;
        self.protos.insert(block.start);
        self.protos.extend(e.called.iter().copied());
        let _ = writeln!(self.bodies, "void {}(int ctx) {{", fn_name(block.start));
        for (r, _) in e.declared.iter() {
            let _ = writeln!(self.bodies, "  {} {};", reg_cty(*r, e.class32), reg_name(*r));
        }
        self.bodies.push_str(&e.out);
        self.bodies.push_str("}\n\n");
        Ok(())
    }

    /// The finished C source.
    pub fn source(&self) -> String {
        let mut s = String::from("#include \"gex_abi.h\"\n\n");
        for a in &self.protos {
            let _ = writeln!(s, "void {}(int ctx);", fn_name(*a));
        }
        s.push('\n');
        s.push_str(&self.bodies);
        s
    }
}

/// Function mode (Phase B): every register the function touches is a C
/// local for the whole function, loaded at entry and after anything that may
/// change guest state, and written back at every point control can leave.
pub struct FnRegs {
    /// Read or written anywhere: loaded on entry and reloaded after calls.
    pub used: BTreeSet<GuestReg>,
    /// Written anywhere: stored back before control can leave.
    pub written: BTreeSet<GuestReg>,
    /// Store only the registers that may be dirty at each write-back.
    pub clean_spills: bool,
}

pub(crate) struct Emitter<'a> {
    block: &'a Block,
    /// Set in function mode; `None` for tier-1-style blocks.
    pub(crate) func_regs: Option<FnRegs>,
    owned: &'a HashSet<u64>,
    out: String,
    /// Cached registers and whether each is dirty (tier-1's `regs`).
    regs: BTreeMap<GuestReg, bool>,
    /// Every register that has a C variable (declared at the top).
    declared: BTreeMap<GuestReg, ()>,
    value_ty: Vec<Ty>,
    value_const: Vec<Option<u64>>,
    /// Values known to fit in 32 bits because they are `x & 0xffffffff` or
    /// `x >> 32` -- the halves of a lifted `umulh`/`smulh`. See `bin`.
    value_half: Vec<bool>,
    /// Blocks tail-called directly, for prototypes.
    called: BTreeSet<u64>,
    /// Counter for block-scoped temporaries.
    tmp: u32,
    /// An i386 program: its GPRs are 32-bit variables (`narrow_reg`).
    pub(crate) class32: bool,
    /// What a fault import's caller does when the host returns from it (the
    /// fault was dispatched nested and `ctx::PC` says where to go on):
    /// `goto gex_resume;` in a whole-image body, `return 0;` in its loop
    /// functions. `None`: the import never returns (a trap follows).
    pub(crate) fault_exit: Option<&'static str>,
    /// With `fault_exit`: the address after the instruction of the
    /// statement being emitted, set by the caller for a `HostCall` to
    /// `Breakpoint` or `Illegal` (`int3`), which carries on there when the
    /// fault's handlers continue at it.
    pub(crate) fault_next: Option<u64>,
    /// Values known to be below 2^32 (a zero-extension, a narrow register).
    value_zext: Vec<bool>,
    /// `base + c` (stage 0), its wrap to 32 bits (1) and that zero-extended
    /// (2), for a `base` below 2^32 and 0 <= c <= 0xffff: an i386 effective
    /// address with a displacement, which `address` turns into the guarded
    /// base plus a static offset (tier-1's shape, and the guard of one base
    /// shared by every displacement from it).
    value_disp: Vec<Option<(Value, u64, u8)>>,
    /// The values that are the context address plus offsets
    /// (`GuestReg::CtxAddr`: the x87 stack, `EFLAGS_SYS`), whose accesses
    /// the null guard leaves out: the context is below it.
    value_ctx: Vec<bool>,
    /// How an `Expr::Mx` is lowered here ([`Emitter::stmts`]).
    mx_mode: MxMode,
    /// Values are declared before their definitions: a definition assigns.
    predeclared: bool,
}

/// The most `Expr::Mx`s under one test of the rounding control
/// ([`Emitter::stmts`]).
const MX_RUN: usize = 32;

/// How [`Emitter::def`] lowers an `Expr::Mx`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum MxMode {
    /// The operation's own C, then the exact call when the rounding control
    /// is not to nearest: one test per operation.
    Each,
    /// The operation's own C: the run's test found rounding to nearest.
    Own,
    /// The exact call: the run's test found another rounding.
    Exact,
}

impl<'a> Emitter<'a> {
    pub(crate) fn new(block: &'a Block, owned: &'a HashSet<u64>) -> Emitter<'a> {
        let n = block.n_values as usize;
        Emitter {
            block,
            func_regs: None,
            owned,
            out: String::new(),
            regs: BTreeMap::new(),
            declared: BTreeMap::new(),
            value_ty: vec![Ty::I64; n],
            value_const: vec![None; n],
            value_half: vec![false; n],
            called: BTreeSet::new(),
            tmp: 0,
            class32: false,
            fault_exit: None,
            fault_next: None,
            value_zext: vec![false; n],
            value_disp: vec![None; n],
            value_ctx: vec![false; n],
            mx_mode: MxMode::Each,
            predeclared: false,
        }
    }

    /// Size the per-value tables for `n` values (function mode numbers
    /// values across the whole function).
    pub(crate) fn resize(&mut self, n: usize) {
        self.value_ty = vec![Ty::I64; n];
        self.value_const = vec![None; n];
        self.value_half = vec![false; n];
        self.value_zext = vec![false; n];
        self.value_disp = vec![None; n];
        self.value_ctx = vec![false; n];
    }

    /// The C emitted so far, leaving the emitter empty.
    pub(crate) fn take_output(&mut self) -> String {
        std::mem::take(&mut self.out)
    }

    pub(crate) fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    pub(crate) fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("t{}", self.tmp)
    }

    pub(crate) fn v(&self, v: Value) -> String {
        format!("v{}", v.0)
    }

    fn run(&mut self) -> Result<()> {
        let block = self.block;
        if block.guard.is_some() {
            bail!("block {:#x} carries a self-modification guard; it stays tier-1", block.start);
        }
        if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
            self.target_check();
        }
        self.stmts(&block.stmts, |e, k| e.stmt(&block.stmts[k]))?;
        self.terminator()
    }

    // -- register cache (tier-1's `reg_local`, `flush`, `invalidate`) -------

    fn reg(&mut self, r: GuestReg) -> String {
        let name = reg_name(r);
        if self.func_regs.is_some() {
            return name;
        }
        if self.regs.contains_key(&r) {
            return name;
        }
        self.declared.insert(r, ());
        let off = reg_offset(r);
        let load = if r == GuestReg::CtxAddr {
            "(uint64_t)(uint32_t)ctx".to_string()
        } else if r.is_flag() {
            format!("GEX_C8({off}u)")
        } else if narrow_reg(r, self.class32) {
            format!("GEX_C32({off}u)")
        } else {
            match r.ty() {
                Ty::I32 => format!("GEX_C32({off}u)"),
                Ty::I64 => format!("GEX_C64({off}u)"),
                Ty::V128 => format!("GEX_CV({off}u)"),
            }
        };
        self.line(format!("{name} = {load};"));
        self.regs.insert(r, false);
        name
    }

    fn write_reg(&mut self, r: GuestReg, val: &str) {
        self.declared.insert(r, ());
        self.line(format!("{} = {val};", reg_name(r)));
        self.regs.insert(r, !is_scratch(r));
    }

    /// The registers currently dirty (differ from the context).
    pub(crate) fn dirty_set(&self) -> RegSet {
        let v: Vec<GuestReg> = self.regs.iter().filter(|(_, d)| **d).map(|(r, _)| *r).collect();
        RegSet::of(&v)
    }

    /// Mark `set` dirty again (registers a call left untouched in locals).
    pub(crate) fn mark_dirty(&mut self, set: RegSet) {
        if let Some(fr) = &self.func_regs {
            for r in fr.used.iter() {
                if set.contains(*r) {
                    self.regs.insert(*r, true);
                }
            }
        }
    }

    /// Function mode: the registers that may differ from the context here.
    pub(crate) fn set_dirty(&mut self, dirty: &BTreeSet<GuestReg>) {
        self.regs = dirty.iter().map(|r| (*r, true)).collect();
    }

    /// Pricing only (`GEX_CGEN_EXP=NOFLAGWB`): treat the flags as clean, so
    /// the next write-back skips them.
    pub(crate) fn forget_flags(&mut self) {
        self.regs.retain(|r, _| !r.is_flag());
    }

    /// A write-back on a conditional path: the stores happen, but the
    /// emitter keeps believing the registers are dirty.
    pub(crate) fn flush_keep(&mut self) {
        self.flush_inner(true, None);
    }

    /// Function mode: `flush_keep`, storing only registers in `live`.
    pub(crate) fn flush_keep_live(&mut self, live: RegSet) {
        self.flush_inner(true, Some(live));
    }

    pub(crate) fn flush(&mut self) {
        self.flush_inner(false, None);
    }

    /// Function mode: `flush`, storing only registers in `live`. The others
    /// are forgotten (treated as clean from here on): nothing reads them
    /// before they are written again.
    pub(crate) fn flush_live(&mut self, live: RegSet) {
        self.flush_inner(false, Some(live));
    }

    fn flush_inner(&mut self, keep_dirty: bool, live: Option<RegSet>) {
        if let Some(fr) = &self.func_regs {
            // Clean-spill elision: only what may differ from the context on
            // some path to here (`func::dirty_in`); otherwise everything the
            // function writes anywhere.
            let mut w: Vec<GuestReg> = if fr.clean_spills {
                self.regs.iter().filter(|(_, d)| **d).map(|(r, _)| *r).collect()
            } else {
                fr.written.iter().copied().collect()
            };
            if let Some(l) = live {
                w.retain(|r| l.contains(*r));
            }
            let loaded = fr.used.clone();
            self.store_regs(&w, &loaded);
            if !keep_dirty {
                self.regs.clear();
            }
            return;
        }
        let mut dirty: Vec<GuestReg> = self
            .regs
            .iter()
            .filter(|(_, d)| **d)
            .map(|(r, _)| *r)
            .collect();
        dirty.sort_by_key(|r| reg_offset(*r));
        let flags = [GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V];
        let packed = flags.iter().all(|r| self.regs.contains_key(r))
            && flags.iter().any(|r| self.regs.get(r) == Some(&true));
        if packed {
            self.line(format!(
                "GEX_C32({}u) = f_n | (f_z << 8) | (f_c << 16) | (f_v << 24);",
                ctx::NZCV
            ));
            if !keep_dirty {
                for r in flags {
                    self.regs.insert(r, false);
                }
            }
        }
        let pa = [GuestReg::P, GuestReg::A];
        let pa_packed = pa.iter().all(|r| self.regs.contains_key(r))
            && pa.iter().any(|r| self.regs.get(r) == Some(&true));
        if pa_packed {
            self.line(format!("GEX_C16({}u) = (uint16_t)(f_p | (f_a << 8));", ctx::PF));
            if !keep_dirty {
                for r in pa {
                    self.regs.insert(r, false);
                }
            }
        }
        for r in dirty {
            if packed && flags.contains(&r) {
                continue;
            }
            if pa_packed && pa.contains(&r) {
                continue;
            }
            let off = reg_offset(r);
            let name = reg_name(r);
            if r.is_flag() {
                self.line(format!("GEX_C8({off}u) = (uint8_t){name};"));
            } else if narrow_reg(r, self.class32) && crate::image::narrow_stores() {
                // 32-bit stores: exact, every writer of
                // the slot zero-extends (`image::narrow_stores`).
                self.line(format!("GEX_C32({off}u) = {name};"));
            } else {
                match r.ty() {
                    Ty::I32 => self.line(format!("GEX_C32({off}u) = {name};")),
                    Ty::I64 => self.line(format!("GEX_C64({off}u) = {name};")),
                    Ty::V128 => self.line(format!("GEX_CV_SET({off}u, {name});")),
                }
            }
            if !keep_dirty {
                self.regs.insert(r, false);
            }
        }
    }

    /// Function mode: `invalidate`, except for registers known to hold what
    /// their locals already hold (a summarised callee's preserved set).
    pub(crate) fn invalidate_except(&mut self, keep: RegSet) {
        if let Some(fr) = &self.func_regs {
            let u: Vec<GuestReg> = fr.used.iter().copied().filter(|r| !keep.contains(*r)).collect();
            self.load_regs(&u);
            self.regs.clear();
            return;
        }
        self.regs.clear();
    }

    /// Function mode, after a call that returned as predicted: load only the
    /// used registers in `load` (what the continuation may read). The others
    /// keep stale locals, marked clean like everything else, so no
    /// write-back stores them before they are written again (which needs
    /// clean-spill elision: `FnRegs::clean_spills`).
    pub(crate) fn invalidate_only(&mut self, load: RegSet) {
        if let Some(fr) = &self.func_regs {
            let u: Vec<GuestReg> = fr.used.iter().copied().filter(|r| load.contains(*r)).collect();
            self.load_regs(&u);
        }
        self.regs.clear();
    }

    pub(crate) fn invalidate(&mut self) {
        if let Some(fr) = &self.func_regs {
            let u: Vec<GuestReg> = fr.used.iter().copied().collect();
            self.load_regs(&u);
            self.regs.clear();
            return;
        }
        self.regs.clear();
    }

    /// Function mode: load `regs` from the context into their locals.
    pub(crate) fn load_regs(&mut self, regs: &[GuestReg]) {
        for &r in regs {
            let off = reg_offset(r);
            let name = reg_name(r);
            let load = if r == GuestReg::CtxAddr {
                "(uint64_t)(uint32_t)ctx".to_string()
            } else if r.is_flag() {
                format!("GEX_C8({off}u)")
            } else if narrow_reg(r, self.class32) {
                // The low half is the register (`narrow_reg`), as the lazy
                // load reads it.
                format!("GEX_C32({off}u)")
            } else {
                match r.ty() {
                    Ty::I32 => format!("GEX_C32({off}u)"),
                    Ty::I64 => format!("GEX_C64({off}u)"),
                    Ty::V128 => format!("GEX_CV({off}u)"),
                }
            };
            self.line(format!("{name} = {load};"));
        }
    }

    /// Function mode: store `regs` from their locals, packing the flags the
    /// way tier-1 does when all four are loaded.
    fn store_regs(&mut self, regs: &[GuestReg], loaded: &BTreeSet<GuestReg>) {
        // `GEX_CGEN_BALLAST=N`: N more stores at every write-back, into a slot
        // nothing reads, to price the write-back traffic before optimising it
        // (the OPTIMIZATIONS-MISSING.md method: a store cannot be deleted).
        for i in 0..ballast() {
            self.line(format!(
                "*(volatile uint64_t *)(GEX_CTXP(ctx) + {}u) = {i}u;",
                ctx::BALLAST
            ));
        }
        let mut regs: Vec<GuestReg> = regs.to_vec();
        regs.sort_by_key(|r| reg_offset(*r));
        let flags = [GuestReg::N, GuestReg::Z, GuestReg::C, GuestReg::V];
        let packed = flags.iter().all(|r| loaded.contains(r)) && flags.iter().any(|r| regs.contains(r));
        if packed {
            self.line(format!(
                "GEX_C32({}u) = f_n | (f_z << 8) | (f_c << 16) | (f_v << 24);",
                ctx::NZCV
            ));
        }
        // `GEX_CGEN_MUTATE_NOWB=<n>`: never write x<n> back. A deliberate
        // miscompilation, to check that the lockstep check catches one.
        let skip = std::env::var("GEX_CGEN_MUTATE_NOWB")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .map(GuestReg::X);
        for r in regs {
            if r == GuestReg::CtxAddr || (packed && flags.contains(&r)) || Some(r) == skip {
                continue;
            }
            let off = reg_offset(r);
            let name = reg_name(r);
            if r.is_flag() {
                self.line(format!("GEX_C8({off}u) = (uint8_t){name};"));
            } else if narrow_reg(r, self.class32) && crate::image::narrow_stores() {
                // 32-bit stores: exact, every writer of
                // the slot zero-extends (`image::narrow_stores`).
                self.line(format!("GEX_C32({off}u) = {name};"));
            } else {
                match r.ty() {
                    Ty::I32 => self.line(format!("GEX_C32({off}u) = {name};")),
                    Ty::I64 => self.line(format!("GEX_C64({off}u) = {name};")),
                    Ty::V128 => self.line(format!("GEX_CV_SET({off}u, {name});")),
                }
            }
        }
    }

    pub(crate) fn store_pc(&mut self, pc: &str) {
        self.line(format!("GEX_C64({}u) = {pc};", ctx::PC));
    }

    // -- addressing (tier-1's `emit_address`) --

    /// A `char *` expression for a scalar or vector access at `addr + offset`.
    /// An x86 locked access at any alignment (tier-1's
    /// `emit_aligned_or_not`): the atomic `aligned` expression when the
    /// address is naturally aligned, else `gex_locked_misaligned`. Other
    /// architectures, and byte accesses, are `aligned` as before.
    ///
    /// First (a line before the expression), tier-1's check for an address
    /// below the guest's floor -- a null pointer's field, in the runtime's
    /// own memory -- which the host raises as an access violation.
    #[allow(clippy::too_many_arguments)]
    fn any_alignment(&mut self, sz: MemSz, ty: Ty, p: &str, aligned: String, op: &str, a: &str, b: &str) -> String {
        let bytes = match sz {
            MemSz::B1 => 1u32,
            MemSz::B2 => 2,
            MemSz::B4 => 4,
            MemSz::B8 => 8,
        };
        // GEX_CGEN_MUTATE=aligned-atomics: natural-alignment atomics only, as
        // before; a misaligned locked access then traps.
        if !gex_abi::X86 || bytes == 1 || crate::func::mutate("aligned-atomics") {
            return aligned;
        }
        let ea = if gex_abi::mem64() {
            format!("((uint64_t)(uintptr_t)({p}))")
        } else {
            format!("((uint32_t)(uintptr_t)({p}))")
        };
        let start = self.block.start;
        self.line(format!("if (__builtin_expect({ea} < GEX_GUEST_FLOOR, 0)) {{"));
        self.flush_keep();
        self.store_pc(&format!("0x{start:x}ull"));
        self.line(format!("  gex_cache_op(ctx, 0x{start:x}ll, GEX_CACHE_OP_LOW_ACCESS, (int64_t){ea});"));
        self.line("  __builtin_unreachable();");
        self.line("}");
        format!(
            "(__builtin_expect({ea} & {}u, 0) ? ({})gex_locked_misaligned(ctx, 0x{:x}ll, {ea}, {bytes}u, {op}, {a}, {b}) : {aligned})",
            bytes - 1,
            cty(ty),
            self.block.start
        )
    }

    /// The unguarded flavor (`image::unguarded`) for an i386 unit: every
    /// address is reduced mod 2^32 and used as is. Past the guest memory
    /// (whose end, `GUEST_LIMIT`, is the linear memory's maximum) an access
    /// traps on its own, exactly as the guard would make it. Not for a
    /// 64-bit guest, whose addresses a truncation would alias.
    fn unguarded(&self) -> bool {
        self.class32 && crate::image::unguarded()
    }

    /// Track `value_zext` and `value_disp` for the definition of value `idx`.
    fn note_disp(&mut self, idx: usize, expr: &Expr) {
        if idx >= self.value_zext.len() || !gex_abi::X86 || disp_off() {
            return;
        }
        let z = |me: &Self, v: &Value| me.value_zext.get(v.0 as usize).copied().unwrap_or(false);
        let k = |me: &Self, v: &Value| me.value_const.get(v.0 as usize).copied().flatten();
        let disp = |me: &Self, v: &Value| me.value_disp.get(v.0 as usize).copied().flatten();
        match expr {
            Expr::ReadReg(r) => self.value_zext[idx] = narrow_reg(*r, self.class32),
            Expr::Extend { a, signed: false } => {
                self.value_zext[idx] = true;
                if let Some((b, c, 1)) = disp(self, a) {
                    self.value_disp[idx] = Some((b, c, 2));
                }
            }
            Expr::Wrap(a) => {
                if let Some((b, c, 0)) = disp(self, a) {
                    self.value_disp[idx] = Some((b, c, 1));
                }
            }
            Expr::Bin { op: BinOp::Add, ty: Ty::I64, a, b } => {
                if let Some(c) = k(self, b).filter(|c| *c <= 0xffff).filter(|_| z(self, a)) {
                    self.value_disp[idx] = Some((*a, c, 0));
                } else if let Some(c) = k(self, a).filter(|c| *c <= 0xffff).filter(|_| z(self, b)) {
                    self.value_disp[idx] = Some((*b, c, 0));
                }
            }
            _ => {}
        }
    }

    /// Whether `v` is the context address plus offsets (`value_ctx`).
    fn is_ctx(&self, v: Value) -> bool {
        self.value_ctx.get(v.0 as usize).copied().unwrap_or(false)
    }

    fn address(&self, addr: Value, offset: i64) -> String {
        // The null guard (`gex_abi::NULL_GUARD`, tier-1's `emit_address`):
        // every access below it traps. Unguarded, the address less the guard
        // (`gex_n`) with the guard added back in the access's constant
        // offset, which clang folds into the memarg: one add. Guarded, the
        // bounds check's range starts at the guard. The thread's own
        // context, below the guard, is reached as before.
        let g = gex_abi::NULL_GUARD;
        let on = null_on();
        if on && self.is_ctx(addr) && (0..=0xffff).contains(&offset) {
            let a = self.v(addr);
            return if gex_abi::mem64() {
                format!("(GEX_P({a}) + {offset}u)")
            } else {
                format!("(GEX_P((uint32_t)({a})) + {offset}u)")
            };
        }
        // A memory64 guest memory: the guest address as it
        // stands, as tier-1 does; the engine's check traps past the end.
        if gex_abi::mem64() {
            if let Some(c) = self.value_const[addr.0 as usize] {
                let ea = c.wrapping_add(offset as u64);
                let ea = if ea < g && on { u64::MAX - 15 } else { ea };
                return format!("GEX_P(0x{ea:x}ull)");
            }
            let a = self.v(addr);
            let max_static: i64 = if gex_abi::X86 { 0xffff } else if on { u32::MAX as i64 - g as i64 } else { u32::MAX as i64 };
            if on {
                return if (0..=max_static).contains(&offset) {
                    format!("(gex_n({a}) + {}u)", offset as u64 + g)
                } else {
                    format!("(gex_n({a} + (uint64_t){offset}ll) + {g}u)")
                };
            }
            return if offset == 0 {
                format!("GEX_P({a})")
            } else if (0..=max_static).contains(&offset) {
                format!("(GEX_P({a}) + {offset}u)")
            } else {
                format!("GEX_P({a} + (uint64_t){offset}ll)")
            };
        }
        // 32-bit Wine with KUSER_SHARED_DATA folded (tier-1's `emit_address`):
        // a constant in Windows' page names 32-bit Wine's. A guarded unit checks every
        // other address against the old page (`gex_k`), whose failure is the
        // named fault, an access violation. An unguarded one does not: that
        // check on every access made units half as big again (advapi32
        // 1.31 -> 2.01 MB) and a program's start slower, and in 32-bit Wine's compact
        // space an address there is past the memory's end, so the engine's
        // own bounds check traps it (unnamed) as it would any other.
        if gex_abi::kuser_fold() && (self.value_const[addr.0 as usize].is_some() || !(no_guard() || self.unguarded())) {
            let pc = self.block.start;
            if let Some(c) = self.value_const[addr.0 as usize] {
                let ea = c.wrapping_add(offset as u64);
                if let Some(f) = gex_abi::fold_kuser(ea) {
                    gex_ir::note_kuser_fold();
                    return format!("GEX_P(0x{f:x}u)");
                }
                if (ea >= g || !on) && ea < gex_abi::KUSER_GUARD {
                    return format!("GEX_P(0x{ea:x}u)");
                }
                return format!("GEX_P({})", self.kuser_guard(pc, &format!("0x{ea:x}ull")));
            }
            let a = self.v(addr);
            return if offset == 0 {
                format!("GEX_P({})", self.kuser_guard(pc, &format!("(uint64_t)({a})")))
            } else {
                format!("GEX_P({})", self.kuser_guard(pc, &format!("(uint64_t)({a}) + (uint64_t){offset}ll")))
            };
        }
        // An i386 base-plus-displacement address (`value_disp`): the
        // displacement rides on the guarded base, as tier-1's static offset
        // does. Only a sum past 2^32 (which the guest would wrap to low
        // memory) differs: it traps.
        if offset == 0 && (!no_guard() || self.unguarded()) {
            if let Some((b, c, 2)) = self.value_disp.get(addr.0 as usize).copied().flatten() {
                return if self.unguarded() && !on {
                    format!("(GEX_P((uint32_t){}) + {c}u)", self.v(b))
                } else if self.unguarded() {
                    format!("(gex_n((uint32_t){}) + {}u)", self.v(b), c + g)
                } else {
                    format!("(gex_gb({}) + {c}u)", self.v(b))
                };
            }
        }
        if let Some(c) = self.value_const[addr.0 as usize] {
            let ea = c.wrapping_add(offset as u64);
            return if ea < GUEST_LIMIT && (ea >= g || !on) {
                format!("GEX_P(0x{ea:x}u)")
            } else {
                "GEX_P(GEX_POISON_ADDR)".into()
            };
        }
        let a = self.v(addr);
        let max_static: i64 = if gex_abi::X86 { 0xffff } else { u32::MAX as i64 };
        if no_guard() || (self.unguarded() && !on) {
            if (0..=max_static).contains(&offset) {
                return format!("(GEX_P((uint32_t){a}) + {offset}u)");
            }
            return format!("GEX_P((uint32_t)({a} + (uint64_t){offset}ll))");
        }
        if self.unguarded() {
            if (0..=max_static).contains(&offset) {
                return format!("(gex_n((uint32_t){a}) + {}u)", offset as u64 + g);
            }
            return format!("(gex_n((uint32_t)({a} + (uint64_t){offset}ll)) + {g}u)");
        }
        if (0..=0xffff).contains(&offset) {
            if offset == 0 && self.class32 && !disp_off() {
                // i386: every effective address is mod 2^32, so the guard
                // works on 32 bits (a push's `sp - 4` included).
                format!("gex_gb32((uint32_t)({a}))")
            } else if offset == 0 {
                format!("gex_gb({a})")
            } else {
                format!("(gex_gb({a}) + {offset}u)")
            }
        } else if (0..=max_static).contains(&offset) {
            format!("gex_g33({a}, {offset}ull)")
        } else {
            format!("GEX_P(gex_g({a} + (uint64_t){offset}ll))")
        }
    }

    /// The KUSER guard on the 64-bit address expression `a`, in the block at
    /// `pc` (`gex_k`): in a body with a fault exit, the form whose slow path
    /// has the fault delivered in place and leaves by that exit (`gex_k_r`),
    /// as a resumable `int3` does.
    fn kuser_guard(&self, pc: u64, a: &str) -> String {
        match self.fault_exit {
            Some(exit) => format!("gex_k_r(ctx, 0x{pc:x}ull, {a}, {exit})"),
            None => format!("gex_k(ctx, 0x{pc:x}ull, {a})"),
        }
    }

    /// The guarded 32-bit address of a bulk operation's operand (no offset).
    fn bulk_address(&self, v: &str) -> String {
        if gex_abi::mem64() {
            return format!("(uint64_t)({v})");
        }
        if gex_abi::kuser_fold() {
            return self.kuser_guard(self.block.start, &format!("(uint64_t)({v})"));
        }
        if no_guard() || self.unguarded() {
            format!("(uint32_t)({v})")
        } else {
            format!("gex_g({v})")
        }
    }

    // -- statements -------------------------------------------------------

    pub(crate) fn stmt(&mut self, s: &Stmt) -> Result<()> {
        // A bulk access is not volatile: no guest access moves across it
        // (header.rs's GEX_ORDER).
        let bulk = matches!(s, Stmt::MemCopy { .. } | Stmt::MemFill { .. } | Stmt::MemCmp { .. } | Stmt::StrOp { .. });
        if bulk {
            self.line("GEX_ORDER();");
        }
        self.stmt_inner(s)?;
        if bulk {
            self.line("GEX_ORDER();");
        }
        Ok(())
    }

    fn stmt_inner(&mut self, s: &Stmt) -> Result<()> {
        match s {
            Stmt::Def { dst, expr } => self.def(*dst, expr)?,
            Stmt::WriteReg { reg, val } => {
                let v = self.v(*val);
                self.write_reg(*reg, &v);
            }
            Stmt::Store {
                addr,
                offset,
                val,
                sz,
                order,
            } => {
                let p = self.address(*addr, *offset);
                let v = self.v(*val);
                let (ut, at) = match sz {
                    MemSz::B1 => ("gex_u8", "gex_a8"),
                    MemSz::B2 => ("gex_u16", "gex_a16"),
                    MemSz::B4 => ("gex_u32", "gex_a32"),
                    MemSz::B8 => ("gex_u64", "gex_a64"),
                };
                let nt = match sz {
                    MemSz::B1 => "uint8_t",
                    MemSz::B2 => "uint16_t",
                    MemSz::B4 => "uint32_t",
                    MemSz::B8 => "uint64_t",
                };
                if *order == Order::SeqCst {
                    self.line(format!(
                        "__atomic_store_n(({at} *)({p}), ({nt}){v}, __ATOMIC_SEQ_CST);"
                    ));
                } else {
                    self.line(format!("*(volatile {ut} *)({p}) = ({nt}){v};"));
                }
            }
            Stmt::Trap { cond, pc, code } => {
                // tier-1's lowering: only the faulting path writes back
                // (everything, flags too: the host raises a signal from this
                // state), so the registers stay dirty for the path that
                // carries on.
                let c = self.v(*cond);
                self.line(format!("if (__builtin_expect({c} != 0, 0)) {{"));
                self.flush_keep();
                self.store_pc(&format!("0x{pc:x}ull"));
                match self.fault_exit {
                    Some(exit) => {
                        self.line(format!("  gex_breakpoint(ctx, (int64_t)(0x{pc:x}ull | GEX_RESUMABLE_PC), (int32_t){code});"));
                        self.line(format!("  {exit}"));
                    }
                    None => {
                        self.line(format!("  gex_breakpoint(ctx, 0x{pc:x}ll, (int32_t){code});"));
                        self.line("  __builtin_unreachable();");
                    }
                }
                self.line("}");
            }
            Stmt::HostCall { func, args }
                if matches!(func, HostFunc::Breakpoint | HostFunc::Illegal)
                    && self.fault_exit.is_some()
                    && self.fault_next.is_some() =>
            {
                // A resumable fault (`int3`): the host dispatches it nested
                // and returns with the handlers' state in the context. PC is
                // set to the next instruction first, so a host that returns
                // without delivering anything (a trace point) carries on
                // too; any other PC is a handler's target, which the body's
                // `gex_resume` (or the loop function's caller) takes.
                let (exit, next) = (self.fault_exit.unwrap(), self.fault_next.unwrap());
                let name = if *func == HostFunc::Breakpoint { "gex_breakpoint" } else { "gex_illegal" };
                let [pc, imm] = args.as_slice() else {
                    bail!("{func:?} with {} arguments, the import takes 2", args.len());
                };
                let (pc, imm) = (self.v(*pc), self.v(*imm));
                self.flush();
                self.store_pc(&format!("0x{next:x}ull"));
                self.line(format!("{name}(ctx, (int64_t)((uint64_t)({pc}) | GEX_RESUMABLE_PC), {imm});"));
                self.line(format!("if (GEX_C64({}u) != 0x{next:x}ull) {exit}", ctx::PC));
                self.invalidate();
            }
            Stmt::HostCall { func, args } => {
                let call = self.host_call_expr(*func, args)?;
                if func.clobbers_guest_state() {
                    self.flush();
                }
                self.line(format!("{call};"));
                if func.clobbers_guest_state() {
                    self.invalidate();
                }
            }
            Stmt::VStore {
                addr,
                offset,
                val,
                bytes,
            } => {
                let p = self.address(*addr, *offset);
                let v = self.v(*val);
                self.line("GEX_ORDER();");
                match bytes {
                    16 => self.line(format!("wasm_v128_store({p}, {v});")),
                    1 => self.line(format!("wasm_v128_store8_lane({p}, {v}, 0);")),
                    2 => self.line(format!("wasm_v128_store16_lane({p}, {v}, 0);")),
                    4 => self.line(format!("wasm_v128_store32_lane({p}, {v}, 0);")),
                    8 => self.line(format!("wasm_v128_store64_lane({p}, {v}, 0);")),
                    b => bail!("vector store of {b} bytes"),
                }
                self.line("GEX_ORDER();");
            }
            Stmt::MemCopy { dst, src, len } => {
                let d = self.bulk_address(&self.v(*dst));
                let s = self.bulk_address(&self.v(*src));
                let n = self.v(*len);
                self.line("{");
                let (at, nc) = addr_c();
                self.line(format!("  {at} d = {d}, s = {s}; uint64_t n = {n}, room;"));
                self.clamp("d");
                self.clamp("s");
                self.line(format!("  __builtin_memmove(GEX_P(d), GEX_P(s), ({nc})n);"));
                self.line("}");
            }
            Stmt::MemFill { dst, val, len } => {
                let d = self.bulk_address(&self.v(*dst));
                let v = self.v(*val);
                let n = self.v(*len);
                self.line("{");
                let (at, nc) = addr_c();
                self.line(format!("  {at} d = {d}; uint64_t n = {n}, room;"));
                self.clamp("d");
                self.line(format!("  __builtin_memset(GEX_P(d), (int)(uint32_t){v}, ({nc})n);"));
                self.line("}");
            }
            Stmt::MemCmp { dst, a, b, len } => {
                self.value_ty[dst.0 as usize] = Ty::I32;
                let (a, b, n, d) = (self.v(*a), self.v(*b), self.v(*len), self.v(*dst));
                let ga = self.bulk_address(&format!("{a} + i"));
                let gb = self.bulk_address(&format!("{b} + i"));
                self.line(format!("uint32_t {d} = 0;"));
                self.line(format!("for (uint64_t i = 0; i < {n}; i++) {{"));
                self.line(format!(
                    "  {d} = (uint32_t)*(gex_u8 *)GEX_P({ga}) - (uint32_t)*(gex_u8 *)GEX_P({gb});"
                ));
                self.line(format!("  if ({d}) break;"));
                self.line("}");
            }
            Stmt::Fence => self.line("__atomic_thread_fence(__ATOMIC_SEQ_CST);"),
            Stmt::X87 { op, args, outs } => self.x87(*op, args, outs),
            Stmt::StrOp { .. } => self.strop(s)?,
        }
        Ok(())
    }

    /// A `Stmt::X87`: a call to its `runtime/x87.h` helper, which returns
    /// the first output and leaves the others in the context, where the
    /// import leaves them.
    fn x87(&mut self, op: X87Op, args: &[Value], outs: &[Value]) {
        let a: Vec<String> = args.iter().map(|v| self.v(*v)).collect();
        for (o, ty) in outs.iter().zip(op.out_types()) {
            self.value_ty[o.0 as usize] = *ty;
        }
        let o: Vec<String> = outs.iter().map(|v| self.v(*v)).collect();
        let first = match op {
            X87Op::Narrow { single } => format!(
                "gex_x87_narrow(ctx, {}, (uint32_t){}, (uint32_t){}, {})",
                a[0], a[1], a[2], single as u32
            ),
            X87Op::ToInt { bits, trunc } => format!(
                "gex_x87_toint(ctx, {}, (uint32_t){}, (uint32_t){}, {bits}, {})",
                a[0], a[1], a[2], trunc as u32
            ),
            X87Op::Arith(k) => format!(
                "gex_x87_arith(ctx, {}, {}, (uint32_t){}, {}, (uint32_t){}, (uint32_t){})",
                k as u32, a[0], a[1], a[2], a[3], a[4]
            ),
            X87Op::Compare { quiet } => format!(
                "gex_x87_compare(ctx, {}, {}, (uint32_t){}, {}, (uint32_t){})",
                quiet as u32, a[0], a[1], a[2], a[3]
            ),
        };
        let ty0 = cty(op.out_types()[0]);
        self.line(format!("{ty0} {} = ({ty0}){first};", o[0]));
        match op {
            X87Op::Arith(_) => {
                self.line(format!("uint32_t {} = GEX_C32(GEX_X87_SE);", o[1]));
                self.line(format!("uint32_t {} = GEX_C32(GEX_X87_SW);", o[2]));
            }
            _ => self.line(format!("uint32_t {} = GEX_C32(GEX_X87_SW);", o[1])),
        }
    }

    /// `tier1::clamp_len` for one base: `n = min(n, GUEST_LIMIT - base)`.
    /// Nothing for a memory64 guest memory.
    fn clamp(&mut self, base: &str) {
        // A memory64 guest memory needs none: an op past its end traps.
        if gex_abi::mem64() {
            return;
        }
        self.line(format!(
            "  room = 0x{GUEST_LIMIT:x}ull - (uint64_t){base}; n = n < room ? n : room;"
        ));
    }

    /// `tier1::emit_strop`, the same loop and fast paths.
    fn strop(&mut self, s: &Stmt) -> Result<()> {
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
        let bytes = sz.bytes() as u64;
        let (count, dst, src, val, df) =
            (self.v(*count), self.v(*dst), self.v(*src), self.v(*val), self.v(*df));
        let ut = match bytes {
            1 => "gex_u8",
            2 => "gex_u16",
            4 => "gex_u32",
            _ => "gex_u64",
        };
        let mask: u64 = if bytes < 8 { (1u64 << (bytes * 8)) - 1 } else { u64::MAX };
        let id = self.fresh();
        for o in [out_count, out_dst, out_src, out_a, out_b] {
            self.value_ty[o.0 as usize] = Ty::I64;
        }
        let (oc, od, os, oa, ob) = (
            self.v(*out_count),
            self.v(*out_dst),
            self.v(*out_src),
            self.v(*out_a),
            self.v(*out_b),
        );
        self.line(format!("uint64_t {oc}, {od}, {os}, {oa}, {ob};"));
        self.line("{");
        self.line(format!(
            "  uint64_t lc = {count}, ld = {dst}, ls = {src}, lv = {val}, la = 0, lb = 0;"
        ));
        self.line(format!(
            "  uint64_t lstep = {df} ? (uint64_t)-{bytes}ll : {bytes}ull;"
        ));
        let g = |e: &str| {
            if gex_abi::mem64() {
                format!("(uint64_t)({e})")
            } else if no_guard() {
                format!("(uint32_t)({e})")
            } else {
                format!("gex_g({e})")
            }
        };
        let (at, nc) = addr_c();
        if matches!(kind, StrKind::Movs | StrKind::Stos) {
            self.line(format!("  if (!{df} && lc != 0) {{"));
            match kind {
                StrKind::Movs => {
                    self.line(format!(
                        "    if ((ld <= ls) | (ld >= ls + lc * {bytes}ull)) {{"
                    ));
                    self.line(format!(
                        "      {at} d = {}, s = {}; uint64_t n = lc * {bytes}ull, room;",
                        g("ld"),
                        g("ls")
                    ));
                    self.clamp("  d");
                    self.clamp("  s");
                    self.line(format!("      __builtin_memmove(GEX_P(d), GEX_P(s), ({nc})n);"));
                }
                _ => {
                    if bytes > 1 {
                        let rep: u64 = match bytes {
                            2 => 0x0101,
                            4 => 0x0101_0101,
                            _ => 0x0101_0101_0101_0101,
                        };
                        self.line(format!(
                            "    if ((lv & 0x{mask:x}ull) == (lv & 0xff) * 0x{rep:x}ull) {{"
                        ));
                    } else {
                        self.line("    if (1) {");
                    }
                    self.line(format!(
                        "      {at} d = {}; uint64_t n = lc * {bytes}ull, room;",
                        g("ld")
                    ));
                    self.clamp("  d");
                    self.line(format!("      __builtin_memset(GEX_P(d), (int)(uint32_t)lv, ({nc})n);"));
                }
            }
            self.line(format!(
                "      ld += lc * {bytes}ull; ls += lc * {bytes}ull; lc = 0; goto {id};"
            ));
            self.line("    }");
            self.line("  }");
        }
        self.line("  while (lc != 0) {");
        match kind {
            StrKind::Movs => {
                self.line(format!("    la = *({ut} *)GEX_P({});", g("ls")));
                self.line(format!("    *({ut} *)GEX_P({}) = ({ut})la;", g("ld")));
            }
            StrKind::Stos => {
                self.line(format!("    *({ut} *)GEX_P({}) = ({ut})lv;", g("ld")));
            }
            StrKind::Scas => {
                self.line(format!("    la = lv & 0x{mask:x}ull;"));
                self.line(format!("    lb = *({ut} *)GEX_P({});", g("ld")));
            }
            StrKind::Cmps => {
                self.line(format!("    la = *({ut} *)GEX_P({});", g("ls")));
                self.line(format!("    lb = *({ut} *)GEX_P({});", g("ld")));
            }
        }
        self.line("    ld += lstep; ls += lstep; lc -= 1;");
        match until {
            StrUntil::Count => {}
            StrUntil::NotEqual => self.line("    if (la != lb) break;"),
            StrUntil::Equal => self.line("    if (la == lb) break;"),
        }
        self.line("  }");
        self.line(format!("  {id}:;"));
        self.line(format!("  {oc} = lc; {od} = ld; {os} = ls; {oa} = la; {ob} = lb;"));
        self.line("}");
        Ok(())
    }

    // -- expressions --------------------------------------------------------

    fn host_call_expr(&self, func: HostFunc, args: &[Value]) -> Result<String> {
        let name = match func {
            HostFunc::Syscall => "gex_syscall",
            HostFunc::Illegal => "gex_illegal",
            HostFunc::Breakpoint => "gex_breakpoint",
            HostFunc::SysRegRead => "gex_sysreg_read",
            HostFunc::SysRegWrite => "gex_sysreg_write",
            HostFunc::CacheOp => "gex_cache_op",
            HostFunc::Safepoint => "gex_safepoint",
            HostFunc::Fma => "gex_fma",
        };
        let want = match func {
            HostFunc::Syscall | HostFunc::Safepoint => 1,
            HostFunc::Illegal | HostFunc::Breakpoint | HostFunc::SysRegRead => 2,
            HostFunc::SysRegWrite | HostFunc::CacheOp => 3,
            HostFunc::Fma => 4,
        };
        if args.len() != want {
            bail!("{func:?} with {} arguments, the import takes {want}", args.len());
        }
        let mut s = format!("{name}(ctx");
        for a in args {
            let _ = write!(s, ", {}", self.v(*a));
        }
        s.push(')');
        Ok(s)
    }

    fn def(&mut self, dst: Value, expr: &Expr) -> Result<()> {
        let ty = expr.ty();
        let idx = dst.0 as usize;
        self.value_ty[idx] = ty;
        let d = self.v(dst);
        if let Some(z) = self.value_zext.get_mut(idx) {
            *z = false;
        }
        if let Some(p) = self.value_disp.get_mut(idx) {
            *p = None;
        }
        let is_ctx = match expr {
            Expr::ReadReg(GuestReg::CtxAddr) => true,
            Expr::Bin { op: BinOp::Add, a, b, .. } => self.is_ctx(*a) || self.is_ctx(*b),
            _ => false,
        };
        if let Some(c) = self.value_ctx.get_mut(idx) {
            *c = is_ctx;
        }
        self.note_disp(idx, expr);
        if let Expr::HostCall { func, args, ty } = expr {
            if *func == HostFunc::Fma && args.len() == 4 {
                if let Some(dbl) = self.value_const[args[3].0 as usize] {
                    let (a, b, c) = (self.v(args[0]), self.v(args[1]), self.v(args[2]));
                    let e = if dbl != 0 {
                        format!("gex_fma64(ctx, {a}, {b}, {c}, 1)")
                    } else {
                        format!("gex_fmaf({a}, {b}, {c})")
                    };
                    self.line(format!("{} = ({}){e};", self.decl(*ty, &d), cty(*ty)));
                    return Ok(());
                }
            }
            let call = self.host_call_expr(*func, args)?;
            if func.clobbers_guest_state() {
                self.flush();
            }
            self.line(format!("{} = ({}){call};", self.decl(*ty, &d), cty(*ty)));
            if func.clobbers_guest_state() {
                self.invalidate();
            }
            return Ok(());
        }
        if let Expr::Mx { mx, e } = expr {
            match self.mx_mode {
                MxMode::Own => {
                    let own = self.rhs(idx, e)?;
                    self.line(format!("{} = {own};", self.decl(ty, &d)));
                }
                MxMode::Exact => {
                    let exact = self.mx_exact(*mx, e);
                    self.line(format!("{} = {exact};", self.decl(ty, &d)));
                }
                MxMode::Each => {
                    let own = self.rhs(idx, e)?;
                    let m = self.v(*mx);
                    let exact = self.mx_exact(*mx, e);
                    let decl = if self.predeclared { String::new() } else { format!("{} ", cty(ty)) };
                    self.line(format!("{decl}{d} = {own};"));
                    self.line(format!(
                        "if (__builtin_expect(((uint32_t){m} & 0x{:x}u) != 0, 0)) {d} = {exact};",
                        MX_SPECIAL
                    ));
                }
            }
            return Ok(());
        }
        let rhs = self.rhs(idx, expr)?;
        self.line(format!("{} = {rhs};", self.decl(ty, &d)));
        Ok(())
    }

    /// The left of a definition of `d`: a declaration, or the name alone
    /// when it was declared ahead ([`Emitter::stmts`]).
    fn decl(&self, ty: Ty, d: &str) -> String {
        if self.predeclared {
            d.to_string()
        } else {
            format!("const {} {d}", cty(ty))
        }
    }

    /// A block's statements, each through `one`. Its `Expr::Mx`s go in runs
    /// of up to [`MX_RUN`], each from its first to its last: a run tests
    /// MXCSR's rounding control once and is emitted twice, with the
    /// operations' own C (taken when it is to nearest) and with their exact
    /// calls. A run's values are declared ahead, so both copies define them
    /// for what follows (the registers' locals are the function's). Short
    /// runs keep clang's work per basic block near what the block alone
    /// costs. A block that writes MXCSR or makes a host call (which may
    /// change it) tests each operation instead.
    pub(crate) fn stmts(&mut self, stmts: &[Stmt], mut one: impl FnMut(&mut Self, usize) -> Result<()>) -> Result<()> {
        let mxs: Vec<usize> =
            stmts.iter().enumerate().filter(|(_, s)| matches!(s, Stmt::Def { expr: Expr::Mx { .. }, .. })).map(|(k, _)| k).collect();
        let once = !mxs.is_empty()
            && !stmts.iter().any(|s| {
                matches!(
                    s,
                    Stmt::WriteReg { reg: GuestReg::Mxcsr, .. } | Stmt::HostCall { .. } | Stmt::Def { expr: Expr::HostCall { .. }, .. }
                )
            });
        if !once {
            return (0..stmts.len()).try_for_each(|k| one(self, k));
        }
        let m = self.reg(GuestReg::Mxcsr);
        let mut next = 0;
        for chunk in mxs.chunks(MX_RUN) {
            let run = chunk[0]..chunk[chunk.len() - 1] + 1;
            (next..run.start).try_for_each(|k| one(self, k))?;
            for s in &stmts[run.clone()] {
                if let Stmt::Def { dst, expr } = s {
                    self.line(format!("{} {};", cty(expr.ty()), self.v(*dst)));
                }
            }
            let regs = self.regs.clone();
            self.predeclared = true;
            self.line(format!("if (__builtin_expect(((uint32_t){m} & 0x{:x}u) != 0, 0)) {{", MX_SPECIAL));
            self.mx_mode = MxMode::Exact;
            run.clone().try_for_each(|k| one(self, k))?;
            self.line("} else {");
            self.regs = regs;
            self.mx_mode = MxMode::Own;
            run.clone().try_for_each(|k| one(self, k))?;
            self.line("}");
            self.mx_mode = MxMode::Each;
            self.predeclared = false;
            next = run.end;
        }
        (next..stmts.len()).try_for_each(|k| one(self, k))
    }

    /// The C expression of `expr`, defining value `idx`.
    fn rhs(&mut self, idx: usize, expr: &Expr) -> Result<String> {
        Ok(match expr {
            Expr::ConstI32(c) => {
                self.value_const[idx] = Some(*c as u64);
                format!("0x{c:x}u")
            }
            Expr::ConstI64(c) => {
                self.value_const[idx] = Some(*c);
                format!("0x{c:x}ull")
            }
            Expr::ReadReg(r) => self.reg(*r),
            Expr::Bin { op, ty, a, b } => {
                let k = |v: &Value| self.value_const[v.0 as usize];
                self.value_half[idx] = *ty == Ty::I64
                    && match op {
                        BinOp::And => k(a) == Some(0xffff_ffff) || k(b) == Some(0xffff_ffff),
                        BinOp::ShrU => k(b).is_some_and(|s| (32..64).contains(&s)),
                        _ => false,
                    };
                self.bin(*op, *ty, *a, *b)
            }
            Expr::Un { op, ty, a } => {
                let a = self.v(*a);
                let w = *ty == Ty::I64;
                match op {
                    UnOp::Not => format!("~{a}"),
                    UnOp::Neg => format!("({})0 - {a}", cty(*ty)),
                    UnOp::Clz => format!(
                        "({})__builtin_clzg({a}, {})",
                        cty(*ty),
                        if w { 64 } else { 32 }
                    ),
                    UnOp::Ctz => format!(
                        "({})__builtin_ctzg({a}, {})",
                        cty(*ty),
                        if w { 64 } else { 32 }
                    ),
                    UnOp::Popcnt => format!("({})__builtin_popcountg({a})", cty(*ty)),
                    UnOp::Bswap if w => format!("__builtin_bswap64({a})"),
                    UnOp::Bswap => format!("__builtin_bswap32({a})"),
                    UnOp::Bswap16 => format!("({})__builtin_bswap16((uint16_t){a})", cty(*ty)),
                }
            }
            Expr::Cmp { op, ty, a, b } => {
                let (a, b) = (self.v(*a), self.v(*b));
                let st = if *ty == Ty::I32 { "int32_t" } else { "int64_t" };
                let (sa, sb) = (format!("({st}){a}"), format!("({st}){b}"));
                let e = match op {
                    CmpOp::Eq => format!("{a} == {b}"),
                    CmpOp::Ne => format!("{a} != {b}"),
                    CmpOp::LtU => format!("{a} < {b}"),
                    CmpOp::LeU => format!("{a} <= {b}"),
                    CmpOp::GtU => format!("{a} > {b}"),
                    CmpOp::GeU => format!("{a} >= {b}"),
                    CmpOp::LtS => format!("{sa} < {sb}"),
                    CmpOp::LeS => format!("{sa} <= {sb}"),
                    CmpOp::GtS => format!("{sa} > {sb}"),
                    CmpOp::GeS => format!("{sa} >= {sb}"),
                };
                format!("(uint32_t)({e})")
            }
            Expr::Select { cond, a, b, .. } => {
                format!("{} ? {} : {}", self.v(*cond), self.v(*a), self.v(*b))
            }
            Expr::Wrap(a) => format!("(uint32_t){}", self.v(*a)),
            Expr::Extend { a, signed } => {
                if *signed {
                    format!("(uint64_t)(int64_t)(int32_t){}", self.v(*a))
                } else {
                    format!("(uint64_t){}", self.v(*a))
                }
            }
            Expr::ExtendBits {
                a,
                ty,
                bits,
                signed,
            } => {
                let a = self.v(*a);
                let width = ty.bits();
                if *signed {
                    let sh = width - bits;
                    let (u, s) = if *ty == Ty::I32 {
                        ("uint32_t", "int32_t")
                    } else {
                        ("uint64_t", "int64_t")
                    };
                    if sh == 0 {
                        a
                    } else {
                        format!("({u})(({s})({a} << {sh}) >> {sh})")
                    }
                } else {
                    let mask = if *bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
                    format!("{a} & 0x{mask:x}ull")
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
                let p = self.address(*addr, *offset);
                let t = cty(*ty);
                if *order == Order::SeqCst {
                    // Atomic loads have no signed form (tier-1 ignores `signed`
                    // here too: the architecture has no signed load-acquire).
                    let at = match sz {
                        MemSz::B1 => "gex_a8",
                        MemSz::B2 => "gex_a16",
                        MemSz::B4 => "gex_a32",
                        MemSz::B8 => "gex_a64",
                    };
                    format!("({t})__atomic_load_n(({at} *)({p}), __ATOMIC_SEQ_CST)")
                } else {
                    let vt = match (sz, signed) {
                        (MemSz::B1, false) => "gex_u8",
                        (MemSz::B1, true) => "gex_i8",
                        (MemSz::B2, false) => "gex_u16",
                        (MemSz::B2, true) => "gex_i16",
                        (MemSz::B4, false) => "gex_u32",
                        (MemSz::B4, true) => "gex_i32",
                        (MemSz::B8, _) => "gex_u64",
                    };
                    if *signed && *ty == Ty::I64 {
                        format!("(uint64_t)(int64_t)*(volatile {vt} *)({p})")
                    } else if *signed {
                        format!("(uint32_t)(int32_t)*(volatile {vt} *)({p})")
                    } else {
                        format!("({t})*(volatile {vt} *)({p})")
                    }
                }
            }
            Expr::Rmw {
                op,
                addr,
                val,
                sz,
                ty,
            } => {
                let p = self.address(*addr, 0);
                let (at, nt) = atomic_types(*sz);
                let v = self.v(*val);
                let f = match op {
                    RmwOp::Add => "__atomic_fetch_add",
                    RmwOp::Sub => "__atomic_fetch_sub",
                    RmwOp::And => "__atomic_fetch_and",
                    RmwOp::Or => "__atomic_fetch_or",
                    RmwOp::Xor => "__atomic_fetch_xor",
                    RmwOp::Xchg => "__atomic_exchange_n",
                };
                let aligned = format!("({}){f}(({at} *)({p}), ({nt}){v}, __ATOMIC_SEQ_CST)", cty(*ty));
                let lk = match op {
                    RmwOp::Add => "GEX_LK_ADD",
                    RmwOp::Sub => "GEX_LK_SUB",
                    RmwOp::And => "GEX_LK_AND",
                    RmwOp::Or => "GEX_LK_OR",
                    RmwOp::Xor => "GEX_LK_XOR",
                    RmwOp::Xchg => "GEX_LK_XCHG",
                };
                self.any_alignment(*sz, *ty, &p, aligned, lk, &format!("(uint64_t)({nt}){v}"), "0")
            }
            Expr::CmpXchg {
                addr,
                expected,
                replacement,
                sz,
                ty,
            } => {
                let p = self.address(*addr, 0);
                let (at, nt) = atomic_types(*sz);
                let (e, r) = (self.v(*expected), self.v(*replacement));
                let aligned = format!("({})GEX_CAS({at}, {p}, ({nt}){e}, ({nt}){r})", cty(*ty));
                self.any_alignment(
                    *sz,
                    *ty,
                    &p,
                    aligned,
                    "GEX_LK_CAS",
                    &format!("(uint64_t)({nt}){e}"),
                    &format!("(uint64_t)({nt}){r}"),
                )
            }
            Expr::HostCall { .. } => unreachable!("a host call is emitted by `def`"),
            Expr::VLoad {
                addr,
                offset,
                bytes,
            } => {
                let p = self.address(*addr, *offset);
                let load = match bytes {
                    16 => format!("wasm_v128_load({p})"),
                    8 => format!("wasm_v128_load64_zero({p})"),
                    4 => format!("wasm_v128_load32_zero({p})"),
                    1 => format!("wasm_v128_load8_lane({p}, wasm_i64x2_const(0, 0), 0)"),
                    2 => format!("wasm_v128_load16_lane({p}, wasm_i64x2_const(0, 0), 0)"),
                    b => bail!("vector load of {b} bytes"),
                };
                format!("GEX_VLOAD({load})")
            }
            Expr::VConst(b) => {
                let lo = u64::from_le_bytes(b[..8].try_into().unwrap());
                let hi = u64::from_le_bytes(b[8..].try_into().unwrap());
                format!("wasm_i64x2_const((int64_t)0x{lo:x}ull, (int64_t)0x{hi:x}ull)")
            }
            Expr::VSplat { a, lane_bytes } => {
                let a = self.v(*a);
                match lane_bytes {
                    1 => format!("wasm_i8x16_splat((int8_t){a})"),
                    2 => format!("wasm_i16x8_splat((int16_t){a})"),
                    4 => format!("wasm_i32x4_splat((int32_t){a})"),
                    _ => format!("wasm_i64x2_splat((int64_t){a})"),
                }
            }
            Expr::VBin { op, a, b } => {
                let f = match op {
                    VBinOp::And => "wasm_v128_and",
                    VBinOp::Or => "wasm_v128_or",
                    VBinOp::Xor => "wasm_v128_xor",
                    VBinOp::AndNot => "wasm_v128_andnot",
                };
                format!("{f}({}, {})", self.v(*a), self.v(*b))
            }
            Expr::VLaneBin { op, lane, a, b } => {
                format!("{}({}, {})", lane_bin(*op, *lane)?, self.v(*a), self.v(*b))
            }
            Expr::VLaneUn { op, lane, a } => {
                format!("{}({})", lane_un(*op, *lane)?, self.v(*a))
            }
            Expr::VShift {
                op,
                lane,
                a,
                amount,
            } => {
                let p = match lane {
                    VLane::I8 => ("i8x16", "u8x16"),
                    VLane::I16 => ("i16x8", "u16x8"),
                    VLane::I32 | VLane::F32 => ("i32x4", "u32x4"),
                    _ => ("i64x2", "u64x2"),
                };
                let f = match op {
                    VShiftOp::Shl => format!("wasm_{}_shl", p.0),
                    VShiftOp::ShrS => format!("wasm_{}_shr", p.0),
                    VShiftOp::ShrU => format!("wasm_{}_shr", p.1),
                };
                format!("{f}({}, (uint32_t){})", self.v(*a), self.v(*amount))
            }
            Expr::VShuffle { a, b, lanes } => {
                if lanes.iter().any(|l| *l >= 32) {
                    bail!("shuffle mask out of range: {lanes:?}");
                }
                let l: Vec<String> = lanes.iter().map(|x| x.to_string()).collect();
                format!(
                    "wasm_i8x16_shuffle({}, {}, {})",
                    self.v(*a),
                    self.v(*b),
                    l.join(", ")
                )
            }
            Expr::VSwizzle { a, idx } => {
                format!("wasm_i8x16_swizzle({}, {})", self.v(*a), self.v(*idx))
            }
            Expr::VBitSelect { mask, a, b } => format!(
                "wasm_v128_bitselect({}, {}, {})",
                self.v(*a),
                self.v(*b),
                self.v(*mask)
            ),
            Expr::VCvt { op, from, a, b } => {
                let f = lane_cvt(*op, *from)?;
                match b {
                    Some(b) => format!("{f}({}, {})", self.v(*a), self.v(*b)),
                    None => format!("{f}({})", self.v(*a)),
                }
            }
            Expr::VUn { op, a } => {
                let f = match op {
                    VUnOp::PopcntBytes => "wasm_i8x16_popcnt",
                    VUnOp::ExtAddPairwiseU8 => "wasm_u16x8_extadd_pairwise_u8x16",
                    VUnOp::ExtAddPairwiseU16 => "wasm_u32x4_extadd_pairwise_u16x8",
                    VUnOp::ExtAddPairwiseS8 => "wasm_i16x8_extadd_pairwise_i8x16",
                    VUnOp::ExtAddPairwiseS16 => "wasm_i32x4_extadd_pairwise_i16x8",
                };
                format!("{f}({})", self.v(*a))
            }
            Expr::VExtract {
                a,
                lane,
                lane_bytes,
                signed,
                ty,
            } => {
                let a = self.v(*a);
                // As an i32 first, the way the wasm instruction produces it.
                let as_i32 = match (lane_bytes, signed) {
                    (1, true) => format!("(uint32_t)(int32_t)wasm_i8x16_extract_lane({a}, {lane})"),
                    (1, false) => format!("(uint32_t)wasm_u8x16_extract_lane({a}, {lane})"),
                    (2, true) => format!("(uint32_t)(int32_t)wasm_i16x8_extract_lane({a}, {lane})"),
                    (2, false) => format!("(uint32_t)wasm_u16x8_extract_lane({a}, {lane})"),
                    (4, _) => format!("(uint32_t)wasm_i32x4_extract_lane({a}, {lane})"),
                    _ => format!("(uint64_t)wasm_i64x2_extract_lane({a}, {lane})"),
                };
                if *lane_bytes == 8 {
                    if *ty != Ty::I64 {
                        bail!("64-bit lane extracted into {ty:?}");
                    }
                    as_i32
                } else if *ty == Ty::I64 {
                    if *signed {
                        format!("(uint64_t)(int64_t)(int32_t)({as_i32})")
                    } else {
                        format!("(uint64_t)({as_i32})")
                    }
                } else {
                    as_i32
                }
            }
            Expr::VReplace {
                a,
                lane,
                lane_bytes,
                val,
            } => {
                let (a, v) = (self.v(*a), self.v(*val));
                match lane_bytes {
                    1 => format!("wasm_i8x16_replace_lane({a}, {lane}, (int8_t){v})"),
                    2 => format!("wasm_i16x8_replace_lane({a}, {lane}, (int16_t){v})"),
                    4 => format!("wasm_i32x4_replace_lane({a}, {lane}, (int32_t){v})"),
                    _ => format!("wasm_i64x2_replace_lane({a}, {lane}, (int64_t){v})"),
                }
            }
            Expr::VBitmask { a, lane } => {
                let f = match lane.bytes() {
                    1 => "wasm_i8x16_bitmask",
                    2 => "wasm_i16x8_bitmask",
                    4 => "wasm_i32x4_bitmask",
                    _ => "wasm_i64x2_bitmask",
                };
                format!("(uint32_t){f}({})", self.v(*a))
            }
            Expr::FpBin { op, double, a, b } => {
                let (fa, fb) = (self.fp(*a, *double), self.fp(*b, *double));
                let (sfx, back) = if *double { ("f64", "gex_b64") } else { ("f32", "gex_b32") };
                let e = match op {
                    FpBinOp::Add => format!("{fa} + {fb}"),
                    FpBinOp::Sub => format!("{fa} - {fb}"),
                    FpBinOp::Mul => format!("{fa} * {fb}"),
                    FpBinOp::Div => format!("{fa} / {fb}"),
                    FpBinOp::Min => format!("__builtin_wasm_min_{sfx}({fa}, {fb})"),
                    FpBinOp::Max => format!("__builtin_wasm_max_{sfx}({fa}, {fb})"),
                    FpBinOp::Copysign => {
                        if *double {
                            format!("__builtin_copysign({fa}, {fb})")
                        } else {
                            format!("__builtin_copysignf({fa}, {fb})")
                        }
                    }
                };
                format!("{back}({e})")
            }
            Expr::FpUn { op, double, a } => {
                let fa = self.fp(*a, *double);
                let (f, back) = if *double { ("", "gex_b64") } else { ("f", "gex_b32") };
                let e = match op {
                    FpUnOp::Abs => format!("__builtin_fabs{f}({fa})"),
                    FpUnOp::Neg => format!("-{fa}"),
                    FpUnOp::Sqrt => format!("__builtin_sqrt{f}({fa})"),
                    FpUnOp::RoundNearest => format!("__builtin_roundeven{f}({fa})"),
                    FpUnOp::Ceil => format!("__builtin_ceil{f}({fa})"),
                    FpUnOp::Floor => format!("__builtin_floor{f}({fa})"),
                    FpUnOp::Trunc => format!("__builtin_trunc{f}({fa})"),
                };
                format!("{back}({e})")
            }
            Expr::FpCmp { op, double, a, b } => {
                let (fa, fb) = (self.fp(*a, *double), self.fp(*b, *double));
                let o = match op {
                    FpCmpOp::Eq => "==",
                    FpCmpOp::Ne => "!=",
                    FpCmpOp::Lt => "<",
                    FpCmpOp::Le => "<=",
                    FpCmpOp::Gt => ">",
                    FpCmpOp::Ge => ">=",
                };
                format!("(uint32_t)({fa} {o} {fb})")
            }
            Expr::FpCvt {
                kind,
                src_wide,
                dst_wide,
                a,
            } => self.fp_convert(*kind, *src_wide, *dst_wide, *a),
            Expr::Mx { .. } => unreachable!("def lowers Mx"),
        })
    }

    /// [`Expr::Mx`] computed exactly: `gex_mx_s` or `gex_mx_v`
    /// (`runtime/mx.h`).
    fn mx_exact(&self, mx: Value, e: &Expr) -> String {
        let m = self.v(mx);
        match MxPlan::of(e) {
            MxPlan::Scalar { code, a, b, .. } => {
                let b = b.map_or("0".to_string(), |b| format!("(uint64_t){}", self.v(b)));
                let r = format!("gex_mx_s(ctx, {code}, (uint64_t){}, {b}, (uint32_t){m})", self.v(a));
                if e.ty() == Ty::I32 {
                    format!("(uint32_t){r}")
                } else {
                    r
                }
            }
            MxPlan::Lanes { code, a, b, n, in_bytes, out_bytes } => {
                let f = n | if in_bytes == 8 { 16 } else { 0 } | if out_bytes == 8 { 32 } else { 0 };
                let b = b.map_or("wasm_i64x2_const(0, 0)".to_string(), |b| self.v(b));
                format!("gex_mx_v(ctx, {code}, {}, {b}, {f}u, (uint32_t){m})", self.v(a))
            }
        }
    }

    /// A value's bit pattern as a C float of the given precision.
    fn fp(&self, v: Value, double: bool) -> String {
        if double {
            format!("gex_f64({})", self.v(v))
        } else {
            format!("gex_f32((uint32_t){})", self.v(v))
        }
    }

    fn fp_convert(&self, kind: FpConvert, src_wide: bool, dst_wide: bool, a: Value) -> String {
        let av = self.v(a);
        match kind {
            FpConvert::FpToFp => {
                let f = self.fp(a, src_wide);
                if dst_wide {
                    format!("gex_b64((double){f})")
                } else {
                    format!("gex_b32((float){f})")
                }
            }
            FpConvert::IntToFp { signed } => {
                let src = match (src_wide, signed) {
                    (false, true) => format!("(int32_t){av}"),
                    (false, false) => format!("(uint32_t){av}"),
                    (true, true) => format!("(int64_t){av}"),
                    (true, false) => format!("(uint64_t){av}"),
                };
                if dst_wide {
                    format!("gex_b64((double){src})")
                } else {
                    format!("gex_b32((float){src})")
                }
            }
            FpConvert::FpToInt { signed } => {
                let f = self.fp(a, src_wide);
                let b = format!(
                    "__builtin_wasm_trunc_saturate_{}_i{}_f{}",
                    if signed { "s" } else { "u" },
                    if dst_wide { 64 } else { 32 },
                    if src_wide { 64 } else { 32 }
                );
                if dst_wide {
                    format!("(uint64_t){b}({f})")
                } else {
                    format!("(uint32_t){b}({f})")
                }
            }
        }
    }

    fn bin(&self, op: BinOp, ty: Ty, av: Value, bv: Value) -> String {
        let (a, b) = (self.v(av), self.v(bv));
        let w = ty == Ty::I64;
        let (u, s, m) = if w {
            ("uint64_t", "int64_t", 63)
        } else {
            ("uint32_t", "int32_t", 31)
        };
        match op {
            BinOp::Add => format!("{a} + {b}"),
            BinOp::Sub => format!("{a} - {b}"),
            // Four 32x32 products of the halves are how the lifter spells
            // `umulh`; LLVM recognises them and forms an i128 multiply, which
            // wasm32 lowers to `__multi3` returning through the shadow stack
            // (which generated code must never touch). An empty
            // asm on one operand hides the idiom, at the cost of nothing else.
            BinOp::Mul
                if w && self.value_half[av.0 as usize] && self.value_half[bv.0 as usize] =>
            {
                format!("gex_opaque64({a}) * {b}")
            }
            BinOp::Mul => format!("{a} * {b}"),
            BinOp::And => format!("{a} & {b}"),
            BinOp::Or => format!("{a} | {b}"),
            BinOp::Xor => format!("{a} ^ {b}"),
            BinOp::Shl => format!("{a} << ({b} & {m})"),
            BinOp::ShrU => format!("{a} >> ({b} & {m})"),
            BinOp::ShrS => format!("({u})(({s}){a} >> ({b} & {m}))"),
            BinOp::Rotr => {
                if w {
                    format!("__builtin_rotateright64({a}, {b})")
                } else {
                    format!("__builtin_rotateright32({a}, {b})")
                }
            }
            BinOp::DivU => format!("{a} / {b}"),
            BinOp::RemU => format!("{a} % {b}"),
            BinOp::DivS => format!("({u})(({s}){a} / ({s}){b})"),
            BinOp::RemS => format!("({u})(({s}){a} % ({s}){b})"),
        }
    }

    // -- control flow (tier-1's `terminator`, `jump_to_constant`) -----------

    pub(crate) fn safepoint(&mut self, pc: &str) {
        let c = ctx::SAFEPOINT_COUNTER;
        self.line(format!(
            "{{ int32_t sc = (int32_t)GEX_C32({c}u) - 1; GEX_C32({c}u) = (uint32_t)sc; \
             if (sc <= 0) gex_safepoint(ctx, (int64_t)({pc})); }}"
        ));
    }

    /// Tail-call dispatch on `target` (tier-1's `emit_dispatch_tail_call`,
    /// whose stores this makes before handing over).
    pub(crate) fn dispatch(&mut self, target: &str) {
        self.line(format!("GEX_C64({}u) = {target};", ctx::LAST_TARGET));
        if gex_abi::X86 {
            self.line(format!(
                "GEX_C64({}u) = 0x{:x}ull;",
                ctx::LAST_SOURCE,
                self.block.start
            ));
        }
        self.line("GEX_MUSTTAIL return gexc_dispatch(ctx);");
    }

    fn jump(&mut self, target: u64) {
        if self.owned.contains(&target) {
            if gex_abi::DISPATCH_SHIFT > gex_abi::INST_ALIGN_SHIFT {
                self.line(format!("GEX_C64({}u) = 0x{target:x}ull;", ctx::LAST_TARGET));
            }
            self.called.insert(target);
            self.line(format!("GEX_MUSTTAIL return {}(ctx);", fn_name(target)));
            return;
        }
        self.store_pc(&format!("0x{target:x}ull"));
        self.dispatch(&format!("0x{target:x}ull"));
    }

    /// x86-64: a slot covers four bytes, so a block can be entered for a
    /// neighbour's address (tier-1's `emit_target_check`).
    pub(crate) fn target_check(&mut self) {
        let start = self.block.start;
        let lt = ctx::LAST_TARGET;
        self.line(format!("if (GEX_C64({lt}u) != 0x{start:x}ull) {{"));
        self.line(format!(
            "  gex_cache_op(ctx, 0x{start:x}ll, GEX_CACHE_OP_SLOT_CONFLICT, (int64_t)GEX_C64({lt}u));"
        ));
        self.line(format!("  (void)gex_dispatch_miss(ctx, (int64_t)GEX_C64({lt}u));"));
        self.line(format!("  uint64_t t0 = GEX_C64({lt}u);"));
        self.out.push_str("  ");
        self.dispatch("t0");
        self.line("}");
    }

    fn terminator(&mut self) -> Result<()> {
        let block = self.block;
        match block.term {
            Term::Goto { target } => {
                self.flush();
                if target <= block.start {
                    self.safepoint(&format!("0x{target:x}ull"));
                }
                self.jump(target);
            }
            Term::Branch {
                cond,
                taken,
                not_taken,
            } => {
                self.flush();
                let c = self.v(cond);
                self.line(format!("if ({c}) {{"));
                if taken <= block.start {
                    self.safepoint(&format!("0x{taken:x}ull"));
                }
                self.jump(taken);
                self.line("}");
                if not_taken <= block.start {
                    self.safepoint(&format!("0x{not_taken:x}ull"));
                }
                self.jump(not_taken);
            }
            Term::Indirect { target, .. } => {
                let t = self.v(target);
                self.line(format!("const uint64_t tgt = {t};"));
                self.store_pc("tgt");
                self.flush();
                if gex_abi::X86 {
                    self.safepoint("tgt");
                } else {
                    let pc = block.end - gex_abi::INST_ALIGN;
                    self.safepoint(&format!("0x{pc:x}ull"));
                }
                self.dispatch("tgt");
            }
            Term::Syscall { next, imm } => {
                self.flush();
                self.store_pc(&format!("0x{next:x}ull"));
                if imm != 0 {
                    self.line(format!("GEX_C64(GEX_CTX_SYSCALL_IMM) = {imm}ull;"));
                }
                let at = block.end - gex_abi::SYSCALL_INST_BYTES;
                self.line(format!("gex_syscall(ctx, 0x{at:x}ll);"));
                self.invalidate();
                self.jump(next);
            }
            Term::Illegal { raw, pc } => {
                self.flush();
                self.store_pc(&format!("0x{pc:x}ull"));
                self.line(format!("gex_illegal(ctx, 0x{pc:x}ll, (int32_t)0x{raw:x}u);"));
                self.line("__builtin_trap();");
            }
        }
        Ok(())
    }
}

fn atomic_types(sz: MemSz) -> (&'static str, &'static str) {
    match sz {
        MemSz::B1 => ("gex_a8", "uint8_t"),
        MemSz::B2 => ("gex_a16", "uint16_t"),
        MemSz::B4 => ("gex_a32", "uint32_t"),
        MemSz::B8 => ("gex_a64", "uint64_t"),
    }
}

fn lane_prefix(lane: VLane, unsigned: bool) -> &'static str {
    match (lane, unsigned) {
        (VLane::I8, false) => "i8x16",
        (VLane::I8, true) => "u8x16",
        (VLane::I16, false) => "i16x8",
        (VLane::I16, true) => "u16x8",
        (VLane::I32, false) => "i32x4",
        (VLane::I32, true) => "u32x4",
        (VLane::I64, false) => "i64x2",
        (VLane::I64, true) => "u64x2",
        (VLane::F32, _) => "f32x4",
        (VLane::F64, _) => "f64x2",
    }
}

/// The `wasm_simd128.h` intrinsic for a lane-wise binary operation, exactly
/// where `VLaneBinOp::wasm_opcode` has one (the test checks the opcodes).
pub(crate) fn lane_bin(op: VLaneBinOp, lane: VLane) -> Result<String> {
    use VLaneBinOp::*;
    if op.wasm_opcode(lane).is_none() {
        bail!("no wasm SIMD instruction for {op:?} on {lane:?}");
    }
    let unsigned = matches!(
        op,
        MinU | MaxU | AddSatU | SubSatU | AvgrU | LtU | GtU | LeU | GeU
    );
    let name = match op {
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        Div => "div",
        MinS | MinU => "min",
        MaxS | MaxU => "max",
        AddSatS | AddSatU => "add_sat",
        SubSatS | SubSatU => "sub_sat",
        AvgrU => "avgr",
        Eq => "eq",
        Ne => "ne",
        LtS | LtU => "lt",
        GtS | GtU => "gt",
        LeS | LeU => "le",
        GeS | GeU => "ge",
    };
    Ok(format!("wasm_{}_{name}", lane_prefix(lane, unsigned)))
}

pub(crate) fn lane_un(op: VLaneUnOp, lane: VLane) -> Result<String> {
    use VLaneUnOp::*;
    if op.wasm_opcode(lane).is_none() {
        bail!("no wasm SIMD instruction for {op:?} on {lane:?}");
    }
    let name = match op {
        Neg => "neg",
        Abs => "abs",
        Sqrt => "sqrt",
        Ceil => "ceil",
        Floor => "floor",
        Trunc => "trunc",
        Nearest => "nearest",
    };
    Ok(format!("wasm_{}_{name}", lane_prefix(lane, false)))
}

/// `tier1::lane_cvt_op`, as intrinsics, including its fallthrough arms.
pub(crate) fn lane_cvt(op: VCvtOp, from: VLane) -> Result<&'static str> {
    use VCvtOp::*;
    Ok(match (op, from) {
        (FpToIntSat { signed: true }, VLane::F32) => "wasm_i32x4_trunc_sat_f32x4",
        (FpToIntSat { signed: false }, VLane::F32) => "wasm_u32x4_trunc_sat_f32x4",
        (FpToIntSat { signed: true }, VLane::F64) => "wasm_i32x4_trunc_sat_f64x2_zero",
        (FpToIntSat { signed: false }, VLane::F64) => "wasm_u32x4_trunc_sat_f64x2_zero",
        (IntToFp { signed: true }, VLane::I32) => "wasm_f32x4_convert_i32x4",
        (IntToFp { signed: false }, VLane::I32) => "wasm_f32x4_convert_u32x4",
        (IntToFp { .. }, VLane::I64) | (FpToIntSat { .. }, _) => {
            bail!("no wasm SIMD conversion for {op:?} from {from:?}")
        }
        (IntToFp { signed: true }, _) => "wasm_f64x2_convert_low_i32x4",
        (IntToFp { signed: false }, _) => "wasm_f64x2_convert_low_u32x4",
        (DemoteF64, _) => "wasm_f32x4_demote_f64x2_zero",
        (PromoteF32, _) => "wasm_f64x2_promote_low_f32x4",
        (ExtendLow { signed: true }, VLane::I8) => "wasm_i16x8_extend_low_i8x16",
        (ExtendHigh { signed: true }, VLane::I8) => "wasm_i16x8_extend_high_i8x16",
        (ExtendLow { signed: false }, VLane::I8) => "wasm_u16x8_extend_low_u8x16",
        (ExtendHigh { signed: false }, VLane::I8) => "wasm_u16x8_extend_high_u8x16",
        (ExtendLow { signed: true }, VLane::I16) => "wasm_i32x4_extend_low_i16x8",
        (ExtendHigh { signed: true }, VLane::I16) => "wasm_i32x4_extend_high_i16x8",
        (ExtendLow { signed: false }, VLane::I16) => "wasm_u32x4_extend_low_u16x8",
        (ExtendHigh { signed: false }, VLane::I16) => "wasm_u32x4_extend_high_u16x8",
        (ExtendLow { signed: true }, _) => "wasm_i64x2_extend_low_i32x4",
        (ExtendHigh { signed: true }, _) => "wasm_i64x2_extend_high_i32x4",
        (ExtendLow { signed: false }, _) => "wasm_u64x2_extend_low_u32x4",
        (ExtendHigh { signed: false }, _) => "wasm_u64x2_extend_high_u32x4",
        (NarrowSat { signed: true }, VLane::I16) => "wasm_i8x16_narrow_i16x8",
        (NarrowSat { signed: false }, VLane::I16) => "wasm_u8x16_narrow_i16x8",
        (NarrowSat { signed: true }, _) => "wasm_i16x8_narrow_i32x4",
        (NarrowSat { signed: false }, _) => "wasm_u16x8_narrow_i32x4",
    })
}

/// The intrinsic `lane_bin` names, for the opcode test.
pub fn lane_bin_name(op: VLaneBinOp, lane: VLane) -> Result<String> {
    lane_bin(op, lane)
}

/// The intrinsic `lane_un` names, for the opcode test.
pub fn lane_un_name(op: VLaneUnOp, lane: VLane) -> Result<String> {
    lane_un(op, lane)
}

/// Whether a statement is a host call that can change guest state, so the
/// function tier writes back before it and reloads after.
pub fn clobbers(s: &Stmt) -> bool {
    match s {
        Stmt::HostCall { func, .. } => func.clobbers_guest_state(),
        Stmt::Def {
            expr: Expr::HostCall { func, .. },
            ..
        } => func.clobbers_guest_state(),
        _ => false,
    }
}

/// The C types of a bulk operation's addresses and length: 32-bit for a
/// wasm32 guest memory, 64-bit for memory64.
fn addr_c() -> (&'static str, &'static str) {
    if gex_abi::mem64() {
        ("uint64_t", "uint64_t")
    } else {
        ("uint32_t", "uint32_t")
    }
}
