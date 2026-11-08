//! Native wasm32 Wine DLLs in the browser: the native host's `wasmdll`, shared mode. Wine's PE side is a
//! core module, winepe.wasm, and DLLs linked as modules of their own
//! (pelink.py `--split`), behind a skeleton per DLL (a `.wasmref` section
//! names its module). The first skeleton naming a module installs it as a
//! stateful provider (its data from the NT allocator, `PROVIDER | number`
//! in the function table), the core and the modules it imports from first;
//! each skeleton binds its exports to slots of the install
//! (`FuncMap::bind`), whose tier-1 bridges every Worker writes as it
//! catches up (`gex_wasmdll_binds_len` and friends, `BINDS`). A module's
//! imports of the others are a binding list JS applies at each
//! instantiation (`gex_wasmdll_binds`), the slots between them that wait on
//! a later module filled per Worker (`gex_wasmdll_lates`).
//!
//! The protocol with `gex.mjs` (`#cgenProviders`, `#wasmdllCatchUp`):
//!   - a unit's provider `i` with `gex_cgen_provider_wasmdll(u, i)` set is
//!     published by `gex_wasmdll_publish(u, i)`: a provider module number
//!     (the first skeleton: JS catches up, which instantiates the module
//!     from `gex_cgen_pmod_wasm_ptr/len`, then calls
//!     `gex_cgen_pmod_installed` and `gex_wasmdll_installed`), or -2 (bound
//!     into the install: only bridges to write);
//!   - every Worker writes each `BINDS` entry's bridges once.
//!   - The module's `gex.syscall` is JS's (`#nativeSyscall`), so a system
//!     call from native code that traps to JS (a unit or thread to start)
//!     is served there and returns, keeping the native frames below it.

use super::*;
use std::collections::HashMap;

/// One of Wine's PE modules, once its first skeleton mapped: the core
/// (winepe.wasm), or a DLL linked as a module of its own (`pelink.py
/// --split`).
struct WinePe {
    /// Its provider module number (`PROVIDERS`).
    number: u32,
    /// Its function-table install, and that install's first slot.
    install: u32,
    base: u32,
    /// Its data (`__memory_base`).
    data: u64,
    slots: HashMap<String, u32>,
    /// `gexp_<export>` slots by export name, whichever DLL owns them (a
    /// DLL's `-import` re-exports are served by the owner's adapter).
    any_dll: HashMap<String, u32>,
    /// Data symbols' offsets in its data (exported globals).
    data_exports: HashMap<String, u32>,
    /// Writes into its data that follow its initialisation (ntdll's boot
    /// variables).
    pending: Vec<(u64, Vec<u8>)>,
    wasm: Vec<u8>,
    /// The file `wasm` is from, and its content key, made on first asking
    /// (`gex_cgen_pmod_wasm_key`): from the system-wide cache, so a process
    /// neither hashes the 9 MB nor has JS copy them out when the page
    /// already has the module.
    file: String,
    key: Option<(u64, u64)>,
    installed: bool,
    /// The DLLs whose skeleton has been mapped. A later mapping of one (an
    /// image view: wineboot's install maps DLLs so) binds its exports but
    /// writes none of the variables, which are the loaded DLL's -- under 32-bit Wine
    /// rewriting them reset ntdll's `__wine_unixlib_handle`, which Wine's
    /// own loader had set (as the native host found).
    /// Each with the base of that first mapping, the loaded DLL's (`image_of`).
    mapped: HashMap<String, u64>,
    /// Its data bytes from `data`, and each DLL's table slots and data
    /// (pelink's `gexw.dllslots`).
    data_size: u64,
    dll_slots: Vec<DllRange>,
    /// Each DLL's data as it was before its first load ran anything
    /// ([`snap_data`]), to start it afresh when it is loaded again; and the
    /// DLLs whose snapshot waits for the module's data initialisation.
    snaps: HashMap<String, std::sync::Arc<Vec<DataSnap>>>,
    snap_todo: Vec<String>,
    /// An LLP64 module (64-bit Wine, its skeletons PE32+): pointers in its data are
    /// 8 bytes, the entry pairs 16 (as natively).
    ptr64: bool,
    /// A module of the process's other mode (an ILP32 module in an x86-64
    /// process: WoW64's i386 Wine): its addresses are bound in the other
    /// directory (`ALT_INSTALLS`), and it binds only to modules of its own
    /// mode, whose names are the 64-bit side's too (as natively).
    alt: bool,
    /// The name its skeletons' `.wasmref` gives it.
    name: String,
    /// Its exports, and which of them are functions.
    exports: std::collections::HashSet<String>,
    funcs: std::collections::HashSet<String>,
    /// Its imports of the other modules, as a binding list (cgen/dylink.rs's
    /// format: `G`, `E`, `L`), which JS applies at each instantiation.
    binds: String,
    /// Its `L` binds' thunks: a module whose function `k` calls the `k`th
    /// `L` line's slot (`late_thunks`), empty for none.
    thunks: Vec<u8>,
}

/// A call or function address from one module into a function another
/// defines that was not installed when it was: the function-table slot
/// `slot`, past the importer's own, which each Worker fills once both are
/// instantiated there (`gex_wasmdll_lates`).
struct Late {
    importer: u32,
    slot: u32,
    name: String,
    provider: Option<u32>,
    /// The importer's mode (`WinePe::ptr64`): only a module of that mode provides it.
    ptr64: bool,
}

/// A range of one DLL's table slots (past the install's base) or data
/// bytes (past the module's data), end exclusive.
struct DllRange {
    dll: String,
    func: bool,
    lo: u64,
    hi: u64,
}

/// One piece of a DLL's data range: its bytes, or (`None`) `len` zeros.
struct DataSnap {
    at: u64,
    len: u64,
    bytes: Option<Vec<u8>>,
}

/// A DLL's data (its `gexw.dllslots` data ranges, past the module's data
/// at `data`) in 4 KiB pieces, the zero ones not kept (as natively).
fn snap_data(data: u64, ranges: &[DllRange], dll: &str) -> Vec<DataSnap> {
    let memory = crate::guest_bytes();
    let mut out = Vec::new();
    for r in ranges.iter().filter(|r| !r.func && r.dll == dll) {
        let mut a = data + r.lo;
        while a < data + r.hi {
            let len = (data + r.hi - a).min(4096);
            let Some(b) = memory.get(a as usize..(a + len) as usize) else { break };
            out.push(DataSnap { at: a, len, bytes: b.iter().any(|&x| x != 0).then(|| b.to_vec()) });
            a += len;
        }
    }
    out
}

/// pelink's `gexw.dllslots` custom section: lines `dll f|d LO HI` (hex).
fn dll_slots(wasm: &[u8]) -> Vec<DllRange> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let Ok(wasmparser::Payload::CustomSection(c)) = payload {
            if c.name() == "gexw.dllslots" {
                for line in String::from_utf8_lossy(c.data()).lines() {
                    let t: Vec<&str> = line.split(' ').collect();
                    if let [dll, kind, lo, hi] = t[..] {
                        if let (Ok(lo), Ok(hi)) = (u64::from_str_radix(lo, 16), u64::from_str_radix(hi, 16)) {
                            out.push(DllRange { dll: dll.to_string(), func: kind == "f", lo, hi });
                        }
                    }
                }
            }
        }
    }
    out
}

/// `gex.image_of(addr)`: the base of the loaded skeleton of the native DLL
/// whose function `addr` is (a table index below `NATIVE_FN_TOP`, in a Wine
/// PE module's slots) or whose static data it is (in a module's data, which
/// no image holds), else 0, as natively. ntdll asks when no image contains
/// `addr` (GetModuleHandleExW's FROM_ADDRESS, RtlPcToFileHeader).
#[no_mangle]
pub extern "C" fn gex_wasmdll_image_of(addr: u64) -> u64 {
    let g = lock(&WINE_PE);
    let func = addr < gex_abi::ftab::NATIVE_FN_TOP;
    for wp in &g.mods {
        let (lo, n) = if func { (wp.base as u64, ftab(|m| m.install(wp.install).len) as u64) } else { (wp.data, wp.data_size) };
        let Some(off) = addr.checked_sub(lo).filter(|&o| o < n) else { continue };
        let dll = wp.dll_slots.iter().find(|r| r.func == func && off >= r.lo && off < r.hi);
        return dll.and_then(|r| wp.mapped.get(&r.dll).copied()).unwrap_or(0);
    }
    0
}

struct WinePes {
    /// In install order: the core first when DLLs are split.
    mods: Vec<WinePe>,
    late: Vec<Late>,
    /// A function's one address across the modules (`GOT.func`) when its
    /// defining module's element segment has no slot for it: the slot of
    /// the first module that named it. By that module's mode and the name.
    canon: std::collections::BTreeMap<(bool, String), u32>,
    /// The lowest of the PE32+ data window handed out so far.
    data_low: u64,
}

/// Bumped each time the late list changes (an entry added, or its provider
/// installed): a Worker reads the list again only when it moved
/// (`gex_wasmdll_lates_gen`).
static LATE_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static WINE_PE: Mutex<WinePes> = Mutex::new(WinePes { mods: Vec::new(), late: Vec::new(), canon: std::collections::BTreeMap::new(), data_low: 0 });

/// The core module's file name: the module a skeleton of a DLL linked as
/// a module of its own finds beside itself.
const CORE_MODULE: &str = "winepe.wasm";
/// clang's names for a C main, and WinMain: Wine's crt_main references them
/// weakly (gex.mjs MAIN_NAMES gives the undefined ones a trapping stub).
const MAIN_NAMES: [&str; 4] = ["main", "__main_argc_argv", "__original_main", "WinMain"];

/// The installs of the process's other mode (`WinePe::alt`), whose bridges
/// go into the other directory (`bind`).
static ALT_INSTALLS: Mutex<std::collections::BTreeSet<u32>> = Mutex::new(std::collections::BTreeSet::new());

/// Whether a function of the process's other mode serves `addr`: an i386
/// Wine module's export or entry thunk bound there in an x86-64 process.
pub(crate) fn alt_serves(addr: u64) -> bool {
    let Some(install) = ftab(|m| m.get(addr).map(|f| f.install)) else { return false };
    lock(&ALT_INSTALLS).contains(&install)
}

/// Whether a Wine module of this pointer size is of the process's other
/// mode: an ILP32 one (32-bit Wine's, PE32 skeletons) in an x86-64 process.
fn module_alt(is64: bool) -> bool {
    gex_abi::X86 && !is64 && !gex_xlate::default_compat()
}

/// Bridges for addresses bound into the install after it was published.
struct Bind {
    regs: Vec<u32>,
    dirs: Vec<(u64, u32)>,
    slots: u64,
}

// `BINDS`: every late binding, in order, read by every Worker with no lock.
// A binding is boxed once and never changes; its index is reserved with a
// fetch_add, its pointer stored in a segment that never moves (segment k
// holds 64 << k), and the published length moves over every stored one in
// order (any writer moves it), so a Worker never sees a gap.
const BIND_SEG0: usize = 64;
static BIND_SEGS: [std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicPtr<Bind>>; 32] =
    [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; 32];
static BINDS_RESERVED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static BINDS_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn bind_at(i: usize) -> (usize, usize) {
    let k = (usize::BITS - 1 - (i / BIND_SEG0 + 1).leading_zeros()) as usize;
    (k, i - BIND_SEG0 * ((1 << k) - 1))
}

/// Binding `i`'s slot, its segment made if `make` (a writer's).
fn bind_slot(i: usize, make: bool) -> Option<&'static std::sync::atomic::AtomicPtr<Bind>> {
    use std::sync::atomic::{AtomicPtr, Ordering::AcqRel, Ordering::Acquire};
    let (k, at) = bind_at(i);
    let mut seg = BIND_SEGS.get(k)?.load(Acquire);
    if seg.is_null() {
        if !make {
            return None;
        }
        let b: Box<[AtomicPtr<Bind>]> = (0..BIND_SEG0 << k).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect();
        let fresh = Box::into_raw(b) as *mut AtomicPtr<Bind>;
        seg = match BIND_SEGS[k].compare_exchange(std::ptr::null_mut(), fresh, AcqRel, Acquire) {
            Ok(_) => fresh,
            Err(other) => {
                // SAFETY: ours, never shared: the other writer's is used.
                drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(fresh, BIND_SEG0 << k)) });
                other
            }
        };
    }
    // SAFETY: segment k holds `BIND_SEG0 << k` slots, `at` is below that,
    // and segments are never freed.
    Some(unsafe { &*seg.add(at) })
}

fn binds_push(b: Bind) {
    use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
    let i = BINDS_RESERVED.fetch_add(1, AcqRel);
    let Some(slot) = bind_slot(i, true) else { return };
    slot.store(Box::into_raw(Box::new(b)), Release);
    // Publish every stored binding in order, this one or another writer's.
    loop {
        let l = BINDS_LEN.load(Acquire);
        if l >= BINDS_RESERVED.load(Acquire) || bind_slot(l, false).is_none_or(|s| s.load(Acquire).is_null()) {
            break;
        }
        let _ = BINDS_LEN.compare_exchange(l, l + 1, AcqRel, Acquire);
    }
}

thread_local! {
    /// How many bindings this Worker's JS has read (`gex_wasmdll_bind_regs`).
    static SEEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Bindings this Worker has still to write the bridges of.
pub(crate) fn behind() -> bool {
    SEEN.with(|c| c.get()) < BINDS_LEN.load(std::sync::atomic::Ordering::Acquire)
}

/// Binding `i`, published.
fn binds_get(i: usize) -> Option<&'static Bind> {
    use std::sync::atomic::Ordering::Acquire;
    if i >= BINDS_LEN.load(Acquire) {
        return None;
    }
    let p = bind_slot(i, false)?.load(Acquire);
    // SAFETY: a published binding is boxed once and never freed or changed.
    (!p.is_null()).then(|| unsafe { &*p })
}

pub(super) fn is_skeleton(pe: &gex_ld::pe::PeImage) -> bool {
    gex_cgen::image::is_skeleton(pe)
}

/// Whether provider `i` of unit `u` is a native Wine DLL's skeleton.
#[no_mangle]
pub extern "C" fn gex_cgen_provider_wasmdll(u: u32, i: u32) -> i32 {
    lock(&UNITS).get(u as usize).and_then(|x| x.providers.get(i as usize)).is_some_and(|p| p.skeleton.is_some()) as i32
}

/// Publish provider `i` of unit `u`, a skeleton: the module's number for
/// the first (JS installs it), -2 when bound into the install, -1 on an
/// error (said).
#[no_mangle]
pub extern "C" fn gex_wasmdll_publish(u: u32, i: u32) -> i32 {
    let (path, base, pe) = {
        let units = lock(&UNITS);
        let Some(x) = units.get(u as usize) else { return -1 };
        let Some(pe) = x.providers.get(i as usize).and_then(|p| p.skeleton.clone()) else { return -1 };
        (x.path.clone(), x.base, pe)
    };
    match publish(&path, base, &pe) {
        Ok(n) => {
            let mut r = lock(&SKELETON_CODE);
            for sec in pe.sections.iter().filter(|s| s.executable() && s.vsize.max(s.raw_size) > 0) {
                let lo = base + sec.rva as u64;
                r.push((lo, lo + sec.vsize.max(sec.raw_size) as u64));
            }
            n
        }
        Err(e) => {
            crate::warn_line(&format!("gex-pe: {path}: native wasm32 DLL: {e}\n"));
            crate::end_process(0xc000_0135);
        }
    }
}

fn publish(path: &str, base: u64, pe: &gex_ld::pe::PeImage) -> Result<i32, String> {
    let sec = pe.sections.iter().find(|s| s.name == ".wasmref").ok_or("no .wasmref")?;
    let (module_name, dll, data_list) = wasmref(&crate::guest_bytes()[..], base, sec)?;
    // beside the image's file itself: a prefix's system32 and syswow64 name
    // the dist's skeletons by links (soju's bundles)
    let real = crate::sigs::proc_home().ns().canonical(path);
    let dir = real.rsplit_once('/').map(|(d, _)| d.to_string());
    // A DLL freed and loaded again maps anew, often at another base (the
    // Start menu's Control Panel applets: setupapi, cfgmgr32, rpcrt4): its
    // first mapping is gone, released to a PROT_NONE reservation or with
    // the new one at its place, and this one is the loaded DLL's (as
    // natively): winecrt0's base of it and `image_of` follow it. Asked
    // before the modules' lock, which `forget_gone` takes apart from the
    // map's.
    let old = lock(&WINE_PE).mods.iter().filter(|w| w.ptr64 == pe.is64).find_map(|w| w.mapped.get(&dll).copied());
    // Released is also another mapping at the old base since, not this
    // DLL's (a program's own data there): still the DLL's, its .wasmref is.
    let still_there = |o: u64| {
        let bytes = crate::guest_bytes();
        let (at, from, n) = ((o + sec.rva as u64) as usize, (base + sec.rva as u64) as usize, sec.vsize as usize);
        matches!((bytes.get(at..at + n), bytes.get(from..from + n)), (Some(a), Some(b)) if a == b)
    };
    let released = old.is_some_and(|o| {
        o == base || crate::sigs::proc_home().mm().lock().vma_at(o).is_none_or(|v| v.prot == 0) || !still_there(o)
    });
    let mut g = lock(&WINE_PE);
    let before = g.mods.len();
    // The module: installed once, at the first skeleton that names it. A
    // DLL linked as a module of its own (`pelink.py --split`) binds its
    // imports to the core's exports, the core (winepe.wasm beside it)
    // installed first if no skeleton of its has been (a program's image
    // maps before ntdll's).
    let find = |g: &WinePes, name: &str| g.mods.iter().position(|w| w.ptr64 == pe.is64 && w.name == name);
    let core = match find(&g, CORE_MODULE) {
        Some(c) => Some(c),
        None if module_name == CORE_MODULE => None,
        None => match read_module(dir.as_deref(), CORE_MODULE) {
            Ok(m) => Some(install_module(&mut g, dir.as_deref(), CORE_MODULE, m, None, pe.is64, &mut Vec::new())?),
            // no core beside it: this module is Wine's whole PE side
            Err(_) => None,
        },
    };
    let idx = match find(&g, &module_name) {
        Some(i) => i,
        None => {
            let m = read_module(dir.as_deref(), &module_name)?;
            let c = core.filter(|_| module_name != CORE_MODULE);
            install_module(&mut g, dir.as_deref(), &module_name, m, c, pe.is64, &mut Vec::new())?
        }
    };
    let fresh = g.mods.len() > before;
    if fresh {
        super::dylink::published();
    }
    // the core first, then the other modules of their own
    let others: Vec<usize> = core.into_iter().chain(0..g.mods.len()).filter(|&c| c != idx && g.mods[c].ptr64 == pe.is64).collect();

    // This DLL's bindings: its own module's adapters, and for a `-import`
    // re-export the owner's, in whichever install has it (the core's, or
    // another module of its own: advapi32's OpenSCManagerW is sechost's).
    let wp = &g.mods[idx];
    let slot = |n: &str| wp.slots.get(n).copied();
    let missing = slot(&format!("{dll}!gexp__missing"))
        .or_else(|| wp.slots.iter().find(|(k, _)| k.ends_with("!gexp__missing")).map(|(_, v)| *v))
        .ok_or("Wine's PE module has no gexp__missing")?;
    let mut pairs = Vec::new();
    let mut other_pairs: Vec<Vec<(u64, u32)>> = vec![Vec::new(); others.len()];
    for e in pe.exports.entries.iter().filter(|e| e.rva != 0 && e.forwarder.is_none()) {
        let own = e
            .names
            .iter()
            .find_map(|n| slot(&format!("{dll}!gexp_{n}")))
            .or_else(|| slot(&format!("{dll}!gexp_#{}", e.ordinal)))
            .or_else(|| e.names.iter().find_map(|n| wp.any_dll.get(n).copied()));
        match own {
            Some(k) => pairs.push((base + e.rva as u64, k)),
            None => match others.iter().enumerate().find_map(|(i, &c)| e.names.iter().find_map(|n| g.mods[c].any_dll.get(n).copied()).map(|k| (i, k))) {
                Some((i, k)) => other_pairs[i].push((base + e.rva as u64, k)),
                None => pairs.push((base + e.rva as u64, missing)),
            },
        }
    }
    if let (Some(k), true) = (slot(&format!("{dll}!gexp__entry")), pe.entry_rva != 0) {
        pairs.push((base + pe.entry_rva as u64, k));
    }
    pairs.sort_unstable();
    pairs.dedup_by_key(|&mut (a, _)| a);

    // Data exports: the skeleton's export address table pointed at the
    // module's data (32-bit RVAs wrap to any address).
    let edir = pe.dirs.first().map(|d| d.0).unwrap_or(0);
    if edir != 0 && !data_list.is_empty() {
        let mut gm = GuestMem::new(crate::guest_bytes());
        let functions = gm.read_u32(base + edir as u64 + 28).map_err(|_| "export directory outside memory")? as u64;
        for (i, link) in &data_list {
            let Some(off) = wp.data_exports.get(&format!("{dll}!{link}")) else {
                crate::warn_line(&format!("gex-pe: {path}: data export {link} is not in Wine's PE module\n"));
                continue;
            };
            let target = (wp.data + *off as u64).wrapping_sub(base) as u32;
            let _ = gm.write_u32(base + functions + 4 * *i as u64, target);
        }
    }
    // winecrt0's base of this DLL: the skeleton's (after the module's data
    // initialisation: at its first instantiation, or now once it had one)
    let wp = &mut g.mods[idx];
    let first_mapping = !wp.mapped.contains_key(&dll) || released;
    if first_mapping {
        wp.mapped.insert(dll.clone(), base);
    }
    // Loaded again, the DLL starts from its data as it was before its
    // first load: Windows maps a fresh copy, and Wine's DLLs count on it (a
    // one-time initialisation's flag set and what it made freed by the
    // detach). First loaded, that is what is kept (as natively).
    if released {
        if let Some(snap) = wp.snaps.get(&dll).cloned() {
            for p in snap.iter() {
                let bytes = p.bytes.clone().unwrap_or_else(|| vec![0; p.len as usize]);
                if !wp.installed {
                    wp.pending.push((p.at, bytes));
                } else if let Some(dst) = crate::guest_bytes().get_mut(p.at as usize..(p.at + p.len) as usize) {
                    dst.copy_from_slice(&bytes);
                }
            }
        }
    } else if first_mapping {
        if wp.installed {
            let snap = std::sync::Arc::new(snap_data(wp.data, &wp.dll_slots, &dll));
            wp.snaps.insert(dll.clone(), snap);
        } else {
            wp.snap_todo.push(dll.clone());
        }
    }
    if let (true, Some(off)) = (first_mapping, wp.data_exports.get(&format!("{dll}!__gexw_image_base"))) {
        let at = wp.data + *off as u64;
        let bytes = if wp.ptr64 { base.to_le_bytes().to_vec() } else { (base as u32).to_le_bytes().to_vec() };
        if !wp.installed {
            wp.pending.push((at, bytes));
        } else if let Some(dst) = crate::guest_bytes().get_mut(at as usize..at as usize + bytes.len()) {
            dst.copy_from_slice(&bytes);
        }
    }
    // (64-bit Wine's ntdll has Wine's own Unix side, which sets them: these are
    // gex-kernel's i386 trampolines)
    if first_mapping && dll == "ntdll.dll" && !wp.ptr64 {
        for (name, bytes) in ntdll_variables() {
            if let Some(off) = wp.data_exports.get(&format!("ntdll.dll!{name}")) {
                let at = wp.data + *off as u64;
                if !wp.installed {
                    wp.pending.push((at, bytes));
                } else if let Some(dst) = crate::guest_bytes().get_mut(at as usize..at as usize + bytes.len()) {
                    dst.copy_from_slice(&bytes);
                }
            }
        }
    }
    crate::warn_line(&format!("gex-pe: native wasm32 module for {path}: {} entries\n", pairs.len() + other_pairs.iter().map(Vec::len).sum::<usize>()));
    let install = wp.install;
    let other_installs: Vec<u32> = others.iter().map(|&c| g.mods[c].install).collect();
    let last = g.mods.last().map(|w| w.number).unwrap_or(0);
    drop(g);
    bind(install, &pairs);
    for (oi, p) in other_installs.iter().zip(&other_pairs).filter(|(_, p)| !p.is_empty()) {
        bind(*oi, p);
    }
    // A module installed here: JS catches up (which instantiates it, and
    // the modules installed before it), then `gex_cgen_pmod_installed` and
    // `gex_wasmdll_installed`.
    Ok(if fresh { last as i32 } else { -2 })
}

/// A module's bytes, beside the skeleton: a twin install's
/// (wine64-dist.sh WINE64_TWIN=1) memory64 build `<module>.m64` in a
/// process with a 64-bit memory. The file it was read from too.
fn read_module(dir: Option<&str>, name: &str) -> Result<(String, Vec<u8>), String> {
    let file = match dir {
        Some(d) => format!("{d}/{name}"),
        None => name.to_string(),
    };
    let ns = crate::sigs::proc_home().ns();
    let twin = format!("{file}.m64");
    Ok(match gex_abi::mem64().then(|| gex_kernel::win::read_guest_file(&ns, &twin).ok()).flatten() {
        Some(w) => (twin, w),
        None => {
            let w = gex_kernel::win::read_guest_file(&ns, &file).map_err(|e| format!("reading Wine's PE module {file}: {e:?}"))?;
            (file, w)
        }
    })
}

/// What install_module reads of a module besides its slots: its imports
/// (module, name, a function's, a global's i64 and mutability), exports
/// (name, a function's), and `gexw.modules` (DLL -> module file).
struct Shape {
    imports: Vec<(String, String, bool, Option<(bool, bool)>)>,
    /// Each import's function type, beside `imports`.
    import_types: Vec<Option<wasmparser::FuncType>>,
    exports: Vec<(String, bool)>,
    modules: HashMap<String, String>,
}

fn shape(wasm: &[u8]) -> Result<Shape, String> {
    let e = |e: wasmparser::BinaryReaderError| e.to_string();
    let mut s = Shape { imports: Vec::new(), import_types: Vec::new(), exports: Vec::new(), modules: HashMap::new() };
    let mut types = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(e)? {
            wasmparser::Payload::TypeSection(r) => {
                for t in r.into_iter_err_on_gc_types() {
                    types.push(t.map_err(e)?);
                }
            }
            wasmparser::Payload::ImportSection(r) => {
                for imp in r {
                    let imp = imp.map_err(e)?;
                    let (func, global) = match imp.ty {
                        wasmparser::TypeRef::Func(_) => (true, None),
                        wasmparser::TypeRef::Global(g) => (false, Some((g.content_type == wasmparser::ValType::I64, g.mutable))),
                        _ => (false, None),
                    };
                    s.imports.push((imp.module.to_string(), imp.name.to_string(), func, global));
                    s.import_types.push(match imp.ty {
                        wasmparser::TypeRef::Func(t) => types.get(t as usize).cloned(),
                        _ => None,
                    });
                }
            }
            wasmparser::Payload::ExportSection(r) => {
                for x in r {
                    let x = x.map_err(e)?;
                    s.exports.push((x.name.to_string(), x.kind == wasmparser::ExternalKind::Func));
                }
            }
            wasmparser::Payload::CustomSection(c) if c.name() == "gexw.modules" => {
                for line in String::from_utf8_lossy(c.data()).lines() {
                    if let Some((d, f)) = line.split_once(' ') {
                        s.modules.insert(d.to_string(), f.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(s)
}

/// The C functions a module may import from a module of the process's
/// other mode (wasm32 i386 Wine calling the lowered wasm64 side, or back):
/// an export of the exact wasm type, never a slot. WoW64's native i386
/// ntdll makes its system calls and Unix calls through them.
pub(crate) const CROSS_MODE: &[&str] = &["wow64cpu.dll!__wine_gex_syscall32", "wow64cpu.dll!__wine_gex_unix_call32"];

/// The type of the function `wasm` exports as `name`.
fn export_func_type(wasm: &[u8], name: &str) -> Option<wasmparser::FuncType> {
    let mut types = Vec::new();
    let mut funcs: Vec<u32> = Vec::new();
    let mut index = None;
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.ok()? {
            wasmparser::Payload::TypeSection(r) => {
                for t in r.into_iter_err_on_gc_types() {
                    types.push(t.ok()?);
                }
            }
            wasmparser::Payload::ImportSection(r) => {
                for imp in r {
                    if let wasmparser::TypeRef::Func(t) = imp.ok()?.ty {
                        funcs.push(t);
                    }
                }
            }
            wasmparser::Payload::FunctionSection(r) => {
                for t in r {
                    funcs.push(t.ok()?);
                }
            }
            wasmparser::Payload::ExportSection(r) => {
                for x in r {
                    let x = x.ok()?;
                    if x.name == name && x.kind == wasmparser::ExternalKind::Func {
                        index = Some(x.index);
                    }
                }
            }
            _ => {}
        }
    }
    types.get(*funcs.get(index? as usize)? as usize).cloned()
}

/// A cross-mode import `name` of type `want` bound to `provider`'s export:
/// the types must match exactly.
fn cross_mode_check(name: &str, want: Option<&wasmparser::FuncType>, provider: &WinePe) -> Result<(), String> {
    let have = export_func_type(&provider.wasm, name);
    if want.is_some() && want == have.as_ref() {
        return Ok(());
    }
    Err(format!("{name}: imported as {want:?}, {} exports it as {have:?}", provider.file))
}

/// Install one of Wine's PE modules (`file`, `wasm` read by
/// [`read_module`]) as a stateful provider: its data allocated, and with
/// `core` (a module of its own) its imports of the other modules bound, as
/// natively (hosts/native wasmdll.rs `install_module`): the modules it calls
/// into first (eagerly, `busy` the ones being installed), then functions
/// it calls to their exports, a function's address (`GOT.func`) to the
/// defining module's own slot, or to one slot per function in the process
/// that each Worker fills (`Late`). Its index in `g.mods`.
fn install_module(
    g: &mut WinePes,
    dir: Option<&str>,
    name: &str,
    (file, wasm): (String, Vec<u8>),
    core: Option<usize>,
    is64: bool,
    busy: &mut Vec<String>,
) -> Result<usize, String> {
    // The module's memory must be this process's: a wasm32 one in a
    // 64-bit memory fails to instantiate (a LinkError, deep in the
    // runtime), so it is said here, with what is missing.
    if let Some(m64) = crate::vfork::wasm_memory64(&wasm[..wasm.len().min(4 << 20)]) {
        if m64 != gex_abi::mem64() {
            return Err(format!(
                "Wine's PE module {file} is for a {}-bit memory and this process has a {}-bit one{}",
                if m64 { 64 } else { 32 },
                if m64 { 32 } else { 64 },
                if m64 { "" } else { " (a twin install's memory64 module, winepe.wasm.m64 beside it, is missing: wine64-dist.sh WINE64_TWIN=1)" }
            ));
        }
    }
    // 32-bit Wine's PE module: refused when its inline probe reads another layout
    // (gexrt's `gex_ftab_idx`; 64-bit Wine's wasm64 gexrt has no probe).
    if !is64 {
        if let Some(e) = super::wasmprog::abi_refusal(&wasm, &format!("Wine's PE module {file}")) {
            return Err(e);
        }
    }
    let sh = shape(&wasm)?;
    let alt = module_alt(is64);
    if alt {
        crate::with_px(|px| crate::ensure_alt_dir(px));
    }
    // The modules it calls into or takes function addresses of, installed
    // first (no code of theirs runs until Wine loads their DLLs): its
    // imports then bind to their functions directly. Only a cycle's far
    // side is left to `Late` slots.
    if core.is_some() {
        let mut deps = std::collections::BTreeSet::new();
        // (not a DLL it only delay-imports, which it marks `gexw-delay!DLL`:
        // its calls load the DLL first, and then are bound, through `Late`
        // slots, when Wine loads it)
        let delayed = |d: &str| sh.exports.iter().any(|(e, _)| e.strip_prefix("gexw-delay!") == Some(d));
        for (m, n, _, _) in &sh.imports {
            if let (true, Some((d, _))) = (m == "env" || m == "GOT.func", n.split_once('!')) {
                if delayed(d) {
                    continue;
                }
                if let Some(f) = sh.modules.get(d) {
                    deps.insert(f.clone());
                }
            }
        }
        busy.push(name.to_string());
        for f in deps {
            if g.mods.iter().any(|w| w.ptr64 == is64 && w.name == f) || busy.contains(&f) {
                continue;
            }
            let Ok(m) = read_module(dir, &f) else { continue };
            if let Err(e) = install_module(g, dir, &f, m, core, is64, busy) {
                busy.pop();
                return Err(e);
            }
        }
        busy.pop();
    }

    let (size, align) = dylink_memory(&wasm)?;
    let a = 1u64 << align;
    // (an LLP64 module's above the images: see `crate::wine64_pe_data`;
    // and an i386 module's in a 64-bit process, whose 64-bit side finds
    // ntdll's data exports so, wow64's Wow64Transition)
    let data = if is64 || alt {
        alloc_pe_data_window(&mut g.data_low, (size as u64).max(16) + a, "[wine PE data]")?
    } else {
        provider_alloc_top_bytes((size as u64).max(16) + a, "[wine PE data]")?
    };
    let data = (data + a - 1) & !(a - 1);
    let names = gex_cgen::provider::slot_names(&wasm).map_err(|e| e.to_string())?;
    let mut slots = HashMap::new();
    let mut any_dll = HashMap::new();
    for (k, n) in names.iter().enumerate() {
        if let Some(n) = n {
            if let Some((_, export)) = n.split_once("!gexp_") {
                any_dll.entry(export.to_string()).or_insert(k as u32);
            }
            slots.insert(n.clone(), k as u32);
        }
    }
    for (name, target) in forwards(&wasm) {
        if let Some((d, f)) = target.split_once('!') {
            if let Some(&k) = slots.get(&format!("{d}!gexp_{f}")) {
                any_dll.entry(name).or_insert(k);
            }
        }
    }
    let data_exports = exported_offsets(&wasm)?;
    let exports: std::collections::HashSet<String> = sh.exports.iter().map(|(n, _)| n.clone()).collect();
    let funcs: std::collections::HashSet<String> = sh.exports.iter().filter(|(_, f)| *f).map(|(n, _)| n.clone()).collect();
    let number = lock(&PROVIDERS).len() as u32;

    // Its imports of the other modules, the binding list's lines but for
    // slots of its own, known once it is published.
    enum B {
        Export(u32),
        Late(u32),
        Own(u32),
        Abs(u32),
    }
    let mut len = names.len() as u32;
    let mut binds: Vec<(usize, B)> = Vec::new();
    let mut lates: Vec<(u32, String, Option<u32>)> = Vec::new();
    let mut canon_new: Vec<(String, u32)> = Vec::new();
    {
        let others = |n: &str| g.mods.iter().find(|w| w.ptr64 == is64 && w.funcs.contains(n));
        for (i, (m, n, func, global)) in sh.imports.iter().enumerate() {
            let b = if m == "GOT.func" && global.is_some() {
                let own = if funcs.contains(n) { slots.get(n).copied() } else { None };
                let theirs = others(n).and_then(|w| Some(w.base + *w.slots.get(n)?));
                match (own, theirs, g.canon.get(&(is64, n.clone())).copied()) {
                    (Some(k), _, _) => B::Own(k),
                    (None, Some(v), _) => B::Abs(v),
                    (None, None, Some(v)) => B::Abs(v),
                    // a weak name nothing defines (the C mains Wine's crt_main
                    // names): JS's null
                    (None, None, None) if !n.contains('!') && !funcs.contains(n) && others(n).is_none() => continue,
                    (None, None, None) => {
                        let off = len;
                        len += 1;
                        canon_new.push((n.clone(), off));
                        let provider = if funcs.contains(n) { Some(number) } else { others(n).map(|w| w.number) };
                        lates.push((off, n.clone(), provider));
                        B::Own(off)
                    }
                }
            } else if m == "env" && core.is_some_and(|c| g.mods[c].exports.contains(n)) {
                B::Export(g.mods[core.unwrap()].number)
            } else if m == "env" && *func && (n.contains('!') || !MAIN_NAMES.contains(&n.as_str())) {
                // another module's: bound to it if it is installed, else a
                // call through a slot filled when it is (the core's calls
                // into user32, say; a program with no core beside it, whose
                // image maps before ntdll's, calls the core's runtime,
                // __gexw_entry, as natively). Not the C mains Wine's crt_main
                // names weakly, which JS's stubs answer.
                let cross = CROSS_MODE.contains(&n.as_str());
                match others(n).or_else(|| if cross { g.mods.iter().find(|w| w.ptr64 != is64 && w.funcs.contains(n)) } else { None }) {
                    Some(w) => {
                        if w.ptr64 != is64 {
                            cross_mode_check(n, sh.import_types[i].as_ref(), w)?;
                        }
                        B::Export(w.number)
                    }
                    None => {
                        lates.push((len, n.clone(), None));
                        len += 1;
                        B::Late(len - 1)
                    }
                }
            } else {
                continue;
            };
            binds.push((i, b));
        }
    }

    let dll_slots = dll_slots(&wasm);
    g.mods.push(WinePe {
        number,
        install: 0,
        base: 0,
        data,
        slots,
        any_dll,
        data_exports,
        pending: Vec::new(),
        wasm,
        file,
        key: None,
        installed: false,
        mapped: Default::default(),
        data_size: size as u64,
        dll_slots,
        snaps: HashMap::new(),
        snap_todo: Vec::new(),
        ptr64: is64,
        alt,
        name: name.to_string(),
        exports,
        funcs,
        binds: String::new(),
        thunks: Vec::new(),
    });
    let idx = g.mods.len() - 1;
    // bound at each DLL's skeleton, freed and loaded again: its slots are
    // its functions however the addresses come and go (as natively)
    let install = ftab(|m| {
        let i = m.publish(len, PROVIDER | number, Tier::Unit, &[]);
        m.keep(i);
        i
    });
    if alt {
        lock(&ALT_INSTALLS).insert(install);
    }
    let base = ftab(|m| m.install(install).base);
    // its function pointers are its slots: below NATIVE_FN_TOP, or they
    // would read as data addresses
    let end = base as u64 + len as u64;
    if end > gex_abi::ftab::NATIVE_FN_TOP {
        return Err(format!(
            "{}: its {len} table slots, from index {base:#x} ({base} indices taken before it), end at {end:#x}, past {:#x}: \
             its function pointers would read as data addresses",
            g.mods[idx].file, gex_abi::ftab::NATIVE_FN_TOP
        ));
    }
    let (regs, dirs, nslots) = bridges(&[]);
    lock(&PROVIDERS).push(ProviderModule { file: format!("wasmdll:{}", g.mods[idx].file), regs, dirs, slots: nslots, memory_base: data });

    let mut text = String::new();
    let mut thunks: Vec<(wasmparser::FuncType, u32)> = Vec::new();
    for (i, b) in binds {
        let (m, n, _, global) = &sh.imports[i];
        let g64 = |v: u32| {
            let (w, mutable) = global.unwrap_or((true, true));
            format!("G\t{m}\t{n}\t{v}\t{}\t{}\n", mutable as u8, if w { 64 } else { 32 })
        };
        text += &match b {
            B::Export(p) => format!("E\t{m}\t{n}\t{p}\t{n}\t-\n"),
            B::Late(off) => {
                let Some(ty) = sh.import_types[i].clone() else { return Err(format!("{m}.{n}: an import of another module's function without a type")) };
                thunks.push((ty, base + off));
                format!("L\t{m}\t{n}\t{}\t{}\n", base + off, thunks.len() - 1)
            }
            B::Own(off) => g64(base + off),
            B::Abs(v) => g64(v),
        };
    }
    for (n, off) in canon_new {
        g.canon.insert((is64, n), base + off);
    }
    let funcs_now = g.mods[idx].funcs.clone();
    for k in 0..g.late.len() {
        let l = &g.late[k];
        if l.provider.is_some() || !funcs_now.contains(&l.name) {
            continue;
        }
        if l.ptr64 != is64 {
            if !CROSS_MODE.contains(&l.name.as_str()) {
                continue;
            }
            let importer = g.mods.iter().find(|w| w.number == l.importer).ok_or("a late bind's importer is gone")?;
            let sh = shape(&importer.wasm)?;
            let want = sh.imports.iter().zip(sh.import_types).find(|((m, n, _, _), _)| m == "env" && *n == l.name).and_then(|(_, t)| t);
            cross_mode_check(&l.name, want.as_ref(), &g.mods[idx])?;
        }
        g.late[k].provider = Some(number);
    }
    g.late.extend(lates.into_iter().map(|(off, name, provider)| Late { importer: number, slot: base + off, name, provider, ptr64: is64 }));
    LATE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    let wp = &mut g.mods[idx];
    wp.install = install;
    wp.base = base;
    wp.binds = text;
    wp.thunks = if thunks.is_empty() { Vec::new() } else { late_thunks(&thunks) };
    crate::trace_line(&format!("gex: wine PE: {} installed as module {number} at slot {base}, {len} slots\n", wp.file));
    Ok(idx)
}

/// The executable sections of every skeleton mapped: `ud2` stubs, never
/// x86 code -- Wine's PE module serves their exports through the table.
/// The function JIT reads no code there (`fjit::translate_root`): an
/// import thunk's `jmp [iat]`, its slot read-only once Wine's loader filled
/// it, resolves to a one-case jump table whose target is the stub, and the
/// thunk's translation then ran the `ud2` (imm32's call of user32's
/// `User32InitializeImmEntryTable`: SIGILL in every 32-bit Wine page process with
/// `?fjit=1`).
static SKELETON_CODE: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Unmapped ranges (`super::on_retire`'s `gone`): a skeleton mapped there
/// is gone, and so are its exports' bindings. Kept, they sent the code
/// that next landed at those addresses to Wine's PE module: 64-bit Wine's wineboot
/// maps DLLs as image views and unmaps them, and two pages of anonymous
/// code later at 0x7fc11000 called a native export until the rebridge
/// guard gave up.
pub(super) fn forget_gone(gone: &[(u64, u64)]) {
    if gone.is_empty() {
        return;
    }
    let installs = installs();
    if installs.is_empty() {
        return;
    }
    let bound: Vec<u64> = ftab(|m| {
        gone.iter().flat_map(|&(s, e)| m.range(s, e).filter(|(_, f)| installs.contains(&f.install)).map(|(a, _)| a).collect::<Vec<_>>()).collect()
    });
    let skeletons = lock(&SKELETON_CODE).clone();
    // Only where nothing is mapped now (Wine releases a view as a PROT_NONE
    // reservation): a live skeleton's page made writable for a while (a
    // hook's VirtualProtect) keeps its exports.
    let (dead, dead_code): (Vec<u64>, Vec<(u64, u64)>) = {
        let mm = crate::sigs::proc_home().mm().lock();
        let released = |a: u64| mm.vma_at(a).is_none_or(|v| v.prot == 0);
        (
            bound.into_iter().filter(|&a| released(a)).collect(),
            skeletons.into_iter().filter(|&(lo, hi)| gone.iter().any(|&(s, e)| s <= lo && hi <= e) && released(lo)).collect(),
        )
    };
    if !dead.is_empty() {
        let memory = crate::guest_bytes();
        ftab(|m| {
            for &a in &dead {
                m.retire(a);
                gex_kernel::ftab::fcache_clear(memory, a);
            }
        });
    }
    if !dead_code.is_empty() {
        lock(&SKELETON_CODE).retain(|r| !dead_code.contains(r));
    }
}

/// An ordinary PE image mapped at `[start, end)`: the bindings of Wine's PE
/// modules there belong to a skeleton that was mapped there before and is
/// gone, so they go too. Kept, the new image's own code at a freed stub's
/// address called the native export: DXSETUP frees advpack (an i386
/// skeleton), then dsetup.dll maps over it and its DllMain ran advpack's
/// DelNodeA. Keyed on the remap, not on the bytes, so a hook's rewrite of
/// a live skeleton's stub keeps its binding. Every arch and every Wine.
pub(super) fn take_back(start: u64, end: u64) {
    let installs = installs();
    if installs.is_empty() {
        return;
    }
    let memory = crate::guest_bytes();
    let n = ftab(|m| {
        let gone: Vec<u64> = m.range(start, end).filter(|(_, f)| installs.contains(&f.install)).map(|(a, _)| a).collect();
        for &a in &gone {
            m.retire(a);
            gex_kernel::ftab::fcache_clear(memory, a);
        }
        gone.len()
    });
    lock(&SKELETON_CODE).retain(|&(lo, hi)| hi <= start || lo >= end);
    if n > 0 {
        crate::trace_line(&format!("gex: wine PE: {n} bindings at {start:#x}..{end:#x} taken back by an image mapped over them\n"));
    }
}

/// Whether the binding of Wine's PE module at `pc` outlived its skeleton:
/// the bytes there are no longer the skeleton's stub
/// (`gex_kernel::win::skel`), so the code is the guest's and the native
/// export has nothing to do with it. Every binding of the install in that
/// mapping whose stub is gone too is forgotten here, and the caller lets
/// tier-1 translate `pc`. A hook rewrote a 64-bit stub: the native export is
/// bound at the stub's alias first, where the hook's trampoline goes.
///
/// [`forget_gone`] catches a view unmapped, but not every way new code
/// lands where a skeleton was: one program's page start had two pages of
/// anonymous code at 0x7fc11000 and 0x7fc31000, where a skeleton had been
/// mapped as an image view, rebridged to the native export at every call
/// (`elfprov::rebridge`, "provider re-bridge", 254 misses running) while
/// the thread held the loader lock. (Not the mapping's file: Wine reads an
/// image whose sections are not page-aligned into anonymous memory, so a
/// live skeleton is often anonymous too.)
pub(crate) fn binding_is_stale(pc: u64) -> bool {
    let installs = installs();
    if installs.is_empty() || !ftab(|m| m.get(pc).is_some_and(|f| installs.contains(&f.install))) {
        return false;
    }
    use gex_kernel::win::skel;
    let bytes: &[u8] = crate::guest_bytes();
    let stub = |a: u64| skel::intact(bytes, a);
    // Unreadable or the stub still there: a live skeleton.
    if crate::GuestMem::new(crate::guest_bytes()).read_u32(pc).is_err() || stub(pc) {
        return false;
    }
    let (lo, hi) = {
        let mm = crate::sigs::proc_home().mm().lock();
        match mm.vma_at(pc) {
            Some(v) => (v.start, v.end),
            None => (pc & !0xfff, (pc & !0xfff) + 0x1000),
        }
    };
    let memory = crate::guest_bytes();
    let gone: Vec<u64> = ftab(|m| m.range(lo, hi).filter(|(a, f)| installs.contains(&f.install) && !stub(*a)).map(|(a, _)| a).collect());
    ftab(|m| {
        for &a in &gone {
            bind_alias_of(m, bytes, a);
            m.retire(a);
            gex_kernel::ftab::fcache_clear(memory, a);
        }
    });
    crate::trace_line(&format!("gex: {} native bindings in {lo:#x}..{hi:#x} outlived their skeleton; forgotten\n", gone.len()));
    true
}

/// Bind the native export at `stub`'s alias too (`gex_kernel::win::skel`),
/// when it is still the alias and nothing is bound there.
fn bind_alias_of(m: &mut gex_kernel::ftab::FuncMap, bytes: &[u8], stub: u64) -> bool {
    let alias = stub + gex_kernel::win::skel::ALIAS;
    if !gex_kernel::win::skel::alias(bytes, alias) || m.get(alias).is_some() {
        return false;
    }
    let Some(f) = m.get(stub) else { return false };
    let base = m.install(f.install).base;
    m.bind(f.install, &[(alias, f.idx - base)]);
    true
}

/// A miss at a 64-bit stub's alias whose stub still has its binding: a
/// hook's trampoline reached it before a call through the rewritten stub
/// did. The native export is bound there, and the caller bridges it.
pub(crate) fn bind_alias(pc: u64) -> bool {
    use gex_kernel::win::skel;
    if pc % 16 != skel::ALIAS || !in_skeleton(pc) {
        return false;
    }
    let bytes: &[u8] = crate::guest_bytes();
    if !skel::alias(bytes, pc) {
        return false;
    }
    let installs = installs();
    ftab(|m| m.get(pc - skel::ALIAS).is_some_and(|f| installs.contains(&f.install)) && bind_alias_of(m, bytes, pc - skel::ALIAS))
}

/// The function-table installs of Wine's PE modules.
fn installs() -> Vec<u32> {
    lock(&WINE_PE).mods.iter().map(|w| w.install).collect()
}

/// Whether `pc` is in a mapped skeleton's executable sections.
pub(crate) fn in_skeleton(pc: u64) -> bool {
    lock(&SKELETON_CODE).iter().any(|&(lo, hi)| lo <= pc && pc < hi)
}

pub(crate) fn skeleton_code() -> Vec<(u64, u64)> {
    lock(&SKELETON_CODE).clone()
}

/// From a PE file's first bytes: its size mapped (`SizeOfImage`) and
/// whether it is a native Wine DLL's skeleton (a `.wasmref` section, as
/// [`is_skeleton`]); `None` when the headers are not all there.
fn pe_head(b: &[u8]) -> Option<(u64, bool)> {
    let u16_at = |o: usize| b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]));
    let u32_at = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    if b.get(..2)? != b"MZ" {
        return None;
    }
    let pe = u32_at(0x3c)? as usize;
    if b.get(pe..pe + 4)? != b"PE\0\0" {
        return None;
    }
    let (sections, opt_size) = (u16_at(pe + 6)? as usize, u16_at(pe + 20)? as usize);
    let opt = pe + 24;
    // `SizeOfImage` is at 56 in both PE32 and PE32+ optional headers.
    let size = u32_at(opt + 56)? as u64;
    let mut skeleton = false;
    for i in 0..sections {
        let name = b.get(opt + opt_size + i * 40..opt + opt_size + i * 40 + 8)?;
        skeleton |= name.split(|&c| c == 0).next() == Some(b".wasmref".as_slice());
    }
    Some((size, skeleton))
}

/// `gex.image_mapped(base, fd)` (32-bit Wine): Wine's
/// own PE loader mapped the image open at `fd` at `base`, relocated. A
/// native DLL's skeleton gets Wine's PE module bound at its exports, as the
/// native host's `image_mapped` does. The module number when this is the
/// first skeleton (the host instantiates it, then `gex_cgen_pmod_installed`
/// and `gex_wasmdll_installed`), -2 when bound into the install (bridges
/// to write), -3 for an i386 image with a function-tier unit to serve
/// (`super::wine32_mapped`), -1 for anything else (an i386 image stays on
/// tier-1). `base` is 64-bit for a memory64 program's (JS widens a 32-bit
/// one's).
#[no_mangle]
pub extern "C" fn gex_image_mapped(base: u64, fd: i32) -> i32 {
    // The table needs no kernel lock (step 3), and the read is the file's.
    let Some((path, file)) = crate::fds_free().and_then(|fds| {
        let f = fds.get(fd).ok()?;
        Some((f.path.clone(), Arc::clone(&f.handle)))
    }) else {
        return -1;
    };
    // An x86 image (not a skeleton): left unread for its first miss, which
    // a resource-only map or an import scan never has. Its headers say
    // which.
    if super::wine32_may_defer(&path, base) {
        let mut head = vec![0u8; 4096];
        let n = file.read_at(0, &mut head).unwrap_or(0) as usize;
        head.truncate(n);
        if let Some((size, false)) = pe_head(&head) {
            take_back(base, base + size);
            crate::note_wine32_image(base, size, &path);
            super::wine32_defer_unread(&path, base, size, file);
            return -1;
        }
    }
    let Some((pe, lean)) = super::read_pe(&file) else { return -1 };
    crate::note_wine32_image(base, pe.image_size as u64, &path);
    if !is_skeleton(&pe) {
        take_back(base, base + pe.image_size as u64);
        // An i386 image (the program's, a game's DLL): its function-tier
        // unit, when units are on; -3 tells JS to serve it.
        return if super::wine32_mapped(&path, base, &pe, lean) { -3 } else { -1 };
    }
    match publish(&path, base, &pe) {
        Ok(n) => {
            let mut r = lock(&SKELETON_CODE);
            for sec in pe.sections.iter().filter(|s| s.executable() && s.vsize.max(s.raw_size) > 0) {
                let lo = base + sec.rva as u64;
                r.push((lo, lo + sec.vsize.max(sec.raw_size) as u64));
            }
            n
        }
        Err(e) => {
            crate::warn_line(&format!("gex-wine32: {path}: native wasm32 DLL: {e}\n"));
            // A module built for another probe layout runs nothing: the
            // process ends here, not at its first wrong call.
            if e.contains(super::wasmprog::ABI_REFUSED) {
                crate::exit_process(127);
            }
            -1
        }
    }
}

/// A PE32+ module's data, from the top of [`crate::wine64_pe_data`] (reserved at
/// the x86-64 split) down, below `low` (the last one's, 0 for none), zeroed.
/// Packed by the page, not the 64 KiB granule: a process that loads every
/// builtin (a prefix update's setupapi DefaultInstall registers them all in
/// one rundll32) has 74 modules' data, 5.2 MB, which took 8.4 MiB rounded to
/// granules.
fn alloc_pe_data_window(low: &mut u64, size: u64, name: &str) -> Result<u64, String> {
    use gex_kernel::mem::{flags, prot};
    let (lo, mut hi) = crate::wine64_pe_data();
    if *low != 0 {
        hi = hi.min(*low);
    }
    let len = (size + 0xfff) & !0xfff;
    if hi < lo || len > hi - lo {
        return Err(format!("{name}: {len:#x} bytes do not fit the wine64 PE data window"));
    }
    let at = hi - len;
    *low = at;
    let a = {
        let mut mm = crate::sigs::proc_home().mm().lock();
        let a = mm.mmap(at, len, prot::READ | prot::WRITE, flags::PRIVATE | flags::ANONYMOUS | flags::FIXED, name).map_err(|e| format!("mapping {name} at {at:#x}: errno {}", e.0))?;
        mm.note_host_data(a, a + len);
        a
    };
    if let Some(b) = crate::guest_bytes().get_mut(a as usize..(a + len) as usize) {
        b.fill(0);
    }
    Ok(a)
}

fn clear(addrs: &[u64]) {
    let mem = crate::guest_bytes();
    for &a in addrs {
        gex_kernel::ftab::fcache_clear(mem, a);
    }
}

/// Map `pairs` to slots of `install` and queue their bridges for every
/// Worker.
pub(super) fn bind(install: u32, pairs: &[(u64, u32)]) {
    if pairs.is_empty() {
        return;
    }
    ftab(|m| m.bind(install, pairs));
    let addrs: Vec<u64> = pairs.iter().map(|&(a, _)| a).collect();
    clear(&addrs);
    let alt = lock(&ALT_INSTALLS).contains(&install);
    let (regs, dirs, slots) = super::bridges_mode(&addrs, alt);
    binds_push(Bind { regs, dirs, slots });
}

/// Provider module `number`'s bytes when it is Wine's PE module or a
/// wasm32 program (in the shim's heap), else 0.
#[no_mangle]
pub extern "C" fn gex_cgen_pmod_wasm_ptr(number: u32) -> u32 {
    if let Some((p, _)) = super::wasmprog::wasm(number).or_else(|| super::libprov::wasm(number)) {
        return p;
    }
    lock(&WINE_PE).mods.iter().find(|w| w.number == number).map_or(0, |w| w.wasm.as_ptr() as u32)
}

/// Whether provider module `number` is Wine's PE module of the process's
/// other mode (WoW64's i386 one): built for wasm32 and not lowered, beside a
/// 64-bit Wine whose own is.
#[no_mangle]
pub extern "C" fn gex_cgen_pmod_alt(number: u32) -> i32 {
    lock(&WINE_PE).mods.iter().any(|w| w.number == number && w.alt) as i32
}

#[no_mangle]
pub extern "C" fn gex_cgen_pmod_wasm_len(number: u32) -> u32 {
    if let Some(n) = super::wasmprog::len(number).or_else(|| super::libprov::wasm(number).map(|(_, n)| n)) {
        return n;
    }
    lock(&WINE_PE).mods.iter().find(|w| w.number == number).map_or(0, |w| w.wasm.len() as u32)
}

/// Wine's PE module's content key when it is provider module `number`.
pub(super) fn wasm_key(number: u32) -> Option<(u64, u64)> {
    let mut g = lock(&WINE_PE);
    let w = g.mods.iter_mut().find(|w| w.number == number)?;
    Some(*w.key.get_or_insert_with(|| super::wasmprog::file_key(&w.file, &w.wasm)))
}

/// The publishing Worker instantiated Wine's PE modules it installed (their
/// data is initialised and relocated): ntdll's boot variables, then the
/// entry thunks (once each).
#[no_mangle]
pub extern "C" fn gex_wasmdll_installed() {
    let todo: Vec<_> = {
        let mut g = lock(&WINE_PE);
        g.mods
            .iter_mut()
            .filter(|w| !w.installed)
            .map(|wp| {
                wp.installed = true;
                // (before the pending writes: the DLLs mapped so far have
                // run nothing yet)
                for dll in std::mem::take(&mut wp.snap_todo) {
                    let snap = std::sync::Arc::new(snap_data(wp.data, &wp.dll_slots, &dll));
                    wp.snaps.insert(dll, snap);
                }
                let pending = std::mem::take(&mut wp.pending);
                let entries = wp.data_exports.get("__gexw_entries").copied();
                let decl = wp.data_exports.get("__gexw_entries_decl").copied();
                (wp.install, wp.data, entries.map(|e| (e, decl)), pending, if wp.ptr64 { 8u64 } else { 4 })
            })
            .collect()
    };
    for (install, data, entries, pending, w) in todo {
        installed_one(install, data, entries, pending, w);
    }
}

fn installed_one(install: u32, data: u64, entries: Option<(u32, Option<u32>)>, pending: Vec<(u64, Vec<u8>)>, w: u64) {
    let mem = crate::guest_bytes();
    for (at, bytes) in pending {
        if let Some(dst) = mem.get_mut(at as usize..at as usize + bytes.len()) {
            dst.copy_from_slice(&bytes);
        }
    }
    let Some((off, decl)) = entries else { return };
    // The ICALL pass's {function, thunk} pairs (table indices): each thunk
    // at its function's index as a guest address (as the native
    // `bind_entry_thunks`).
    let gm = GuestMem::new(crate::guest_bytes());
    let base = ftab(|m| m.install(install).base);
    let read = |off: u32| -> Vec<(u64, u32)> {
        let at = data + off as u64;
        // An LLP64 module's pointers are 8 bytes. A lowered one's values are
        // below 4 GiB; the memory64 core's data window is at the top of 16
        // GiB (`wine64_pe_data`), so the whole pointer is read -- its low half
        // pointed elsewhere, no thunk was bound, and a fiber's start
        // (`start_fiber`'s table index as its Rip) faulted as a guest pc.
        let ptr = |a: u64| if w == 8 { gm.read_u64(a).unwrap_or(0) } else { gm.read_u32(a).unwrap_or(0) as u64 };
        let (start, stop) = (ptr(at), ptr(at + w));
        let mut out = Vec::new();
        let mut a = start;
        while start != 0 && a + 2 * w <= stop {
            let (f, t) = (gm.read_u32(a).unwrap_or(0), gm.read_u32(a + w).unwrap_or(0));
            a += 2 * w;
            if f < base || t < base || f as u64 >= 0x3f0000 {
                continue;
            }
            out.push((f as u64, t - base));
        }
        out
    };
    let mut pairs = read(off);
    // Thunks made where a function is handed out but not defined: only for
    // functions no defining object gave one (it knows the convention).
    if let Some(off) = decl {
        let strong: std::collections::HashSet<u64> = pairs.iter().map(|&(f, _)| f).collect();
        let mut seen = std::collections::HashSet::new();
        for (f, t) in read(off) {
            if !strong.contains(&f) && seen.insert(f) {
                pairs.push((f, t));
            }
        }
    }
    bind(install, &pairs);
}

/// Provider module `number`'s binding list when it is one of Wine's PE
/// modules with imports of the others (cgen/dylink.rs's format: `G`, `E`,
/// `L`), copied to `buf` up to `cap`; its length, 0 for none.
#[no_mangle]
pub extern "C" fn gex_wasmdll_binds(number: u32, buf: u32, cap: u32) -> u32 {
    let g = lock(&WINE_PE);
    let Some(w) = g.mods.iter().find(|w| w.number == number) else { return 0 };
    // (a buffer from the shim's heap, above the guest's limit)
    super::put_str(&w.binds, buf, cap)
}

/// Provider module `number`'s `L` thunks (`late_thunks`) into `buf` up to
/// `cap`; their length, 0 for none.
#[no_mangle]
pub extern "C" fn gex_wasmdll_thunks(number: u32, buf: u32, cap: u32) -> u32 {
    let g = lock(&WINE_PE);
    let Some(w) = g.mods.iter().find(|w| w.number == number) else { return 0 };
    let n = w.thunks.len().min(cap as usize);
    // (a buffer from the shim's heap, above the guest's limit)
    crate::shim_slice_mut(buf, n as u32).copy_from_slice(&w.thunks[..n]);
    w.thunks.len() as u32
}

/// A module of thunks for calls into a module not yet installed (`L`): its
/// function `k` (exported as `k`) calls `slots[k]`'s table entry, with
/// `call_indirect` once the Worker has filled it (its flag set: byte
/// `r.b + k` of the Worker's flags for this module, in the process memory
/// `r.m`, from the shim's heap), else JS's import `s.k`, which catches up
/// first. In wasm, so a
/// call through it keeps no JS frame below the callee: the single-thread
/// build suspends the whole stack at a blocking call, which a JS frame
/// would refuse.
fn late_thunks(slots: &[(wasmparser::FuncType, u32)]) -> Vec<u8> {
    use wasm_encoder as we;
    let vt = |v: &wasmparser::ValType| match v {
        wasmparser::ValType::I32 => we::ValType::I32,
        wasmparser::ValType::I64 => we::ValType::I64,
        wasmparser::ValType::F32 => we::ValType::F32,
        wasmparser::ValType::F64 => we::ValType::F64,
        wasmparser::ValType::V128 => we::ValType::V128,
        _ => we::ValType::I32,
    };
    let t64 = gex_abi::mem64();
    let n = slots.len() as u32;
    let mut types = we::TypeSection::new();
    let mut imports = we::ImportSection::new();
    let mut funcs = we::FunctionSection::new();
    let mut exports = we::ExportSection::new();
    let mut code = we::CodeSection::new();
    imports.import("t", "t", we::TableType { element_type: we::RefType::FUNCREF, table64: t64, minimum: 0, maximum: None, shared: false });
    for (k, (ty, _)) in slots.iter().enumerate() {
        types.ty().function(ty.params().iter().map(vt), ty.results().iter().map(vt));
        imports.import("s", &k.to_string(), we::EntityType::Function(k as u32));
    }
    // the Worker's ready flags, a byte each from `r.b`, in the process
    // memory: its type as the page makes it (a shared memory's maximum is
    // part of its type; this one admits the page's, 4 GiB less a page or
    // 16 GiB, and any smaller)
    let shared = cfg!(feature = "shared-memory");
    let maximum = if t64 { Some(gex_abi::LIMIT64_PAGES) } else if shared { Some(1 << 16) } else { None };
    imports.import("r", "m", we::MemoryType { minimum: 0, maximum, memory64: t64, shared, page_size_log2: None });
    imports.import("r", "b", we::GlobalType { val_type: if t64 { we::ValType::I64 } else { we::ValType::I32 }, mutable: false, shared: false });
    for (k, (ty, slot)) in slots.iter().enumerate() {
        let k = k as u32;
        funcs.function(k);
        exports.export(&k.to_string(), we::ExportKind::Func, n + k);
        let mut f = we::Function::new([]);
        let np = ty.params().len() as u32;
        f.instruction(&we::Instruction::GlobalGet(0));
        f.instruction(&we::Instruction::I32Load8U(we::MemArg { offset: k as u64, align: 0, memory_index: 0 }));
        f.instruction(&we::Instruction::I32Eqz);
        f.instruction(&we::Instruction::If(we::BlockType::Empty));
        for p in 0..np {
            f.instruction(&we::Instruction::LocalGet(p));
        }
        f.instruction(&we::Instruction::Call(k));
        f.instruction(&we::Instruction::Return);
        f.instruction(&we::Instruction::End);
        for p in 0..np {
            f.instruction(&we::Instruction::LocalGet(p));
        }
        f.instruction(&if t64 { we::Instruction::I64Const(*slot as i64) } else { we::Instruction::I32Const(*slot as i32) });
        f.instruction(&we::Instruction::CallIndirect { type_index: k, table_index: 0 });
        f.instruction(&we::Instruction::End);
        code.function(&f);
    }
    let mut m = we::Module::new();
    m.section(&types).section(&imports).section(&funcs).section(&exports).section(&code);
    m.finish()
}

/// The `Late` slots whose provider is installed, as lines `slot importer
/// provider name` (provider module numbers), copied to `buf` up to `cap`;
/// their length. A Worker fills each slot once it has both modules.
#[no_mangle]
pub extern "C" fn gex_wasmdll_lates(buf: u32, cap: u32) -> u32 {
    let g = lock(&WINE_PE);
    let mut text = String::new();
    for l in &g.late {
        if let Some(p) = l.provider {
            text += &format!("{}\t{}\t{p}\t{}\n", l.slot, l.importer, l.name);
        }
    }
    super::put_str(&text, buf, cap)
}

/// The late list's generation: it changed when this did.
#[no_mangle]
pub extern "C" fn gex_wasmdll_lates_gen() -> u32 {
    LATE_GEN.load(std::sync::atomic::Ordering::Acquire)
}

/// How many late bindings there are: a Worker writes each one's bridges.
#[no_mangle]
pub extern "C" fn gex_wasmdll_binds_len() -> u32 {
    BINDS_LEN.load(std::sync::atomic::Ordering::Acquire) as u32
}

pub(super) fn binds_len() -> u32 {
    gex_wasmdll_binds_len()
}

/// Binding `i`'s bridges, `[table, slot]` flattened, as `gex_cgen_regs`.
#[no_mangle]
pub extern "C" fn gex_wasmdll_bind_regs(i: u32, buf: u32, cap: u32) -> u32 {
    let Some(x) = binds_get(i as usize) else { return 0 };
    SEEN.with(|c| c.set(c.get().max(i as usize + 1)));
    let n = x.regs.len().min(cap as usize);
    put_u32s(&x.regs[..n], buf);
    x.regs.len() as u32
}

#[no_mangle]
pub extern "C" fn gex_wasmdll_bind_slots(i: u32) -> u32 {
    binds_get(i as usize).map_or(0, |x| x.slots.min(u32::MAX as u64) as u32)
}

/// A Worker wrote binding `i`'s bridges: the directory for the pages it
/// numbered (idempotent).
#[no_mangle]
pub extern "C" fn gex_wasmdll_bind_done(i: u32) {
    let dirs = binds_get(i as usize).map(|x| x.dirs.clone()).unwrap_or_default();
    write_dirs(dirs);
}

/// The skeleton's `.wasmref`: `GEXWREF1 module\0 dll\0 u32 count { u32
/// index, name\0 }` (winebuild `--wasm-shared`).
fn wasmref(mem: &[u8], base: u64, sec: &gex_ld::pe::Section) -> Result<(String, String, Vec<(u32, String)>), String> {
    let at = (base + sec.rva as u64) as usize;
    let raw = mem.get(at..at + sec.vsize as usize).ok_or(".wasmref outside memory")?;
    if raw.get(..8) != Some(&b"GEXWREF1"[..]) {
        return Err(".wasmref is not GEXWREF1".into());
    }
    let mut p = 8;
    let cstr = |p: &mut usize| -> Result<String, String> {
        let end = raw[*p..].iter().position(|&c| c == 0).ok_or(".wasmref truncated")? + *p;
        let s = String::from_utf8_lossy(&raw[*p..end]).into_owned();
        *p = end + 1;
        Ok(s)
    };
    let u32_at = |p: &mut usize| -> Result<u32, String> {
        let v = u32::from_le_bytes(raw.get(*p..*p + 4).ok_or(".wasmref truncated")?.try_into().unwrap());
        *p += 4;
        Ok(v)
    };
    let module = cstr(&mut p)?;
    let dll = cstr(&mut p)?;
    let count = u32_at(&mut p)?;
    let mut data = Vec::new();
    for _ in 0..count {
        let idx = u32_at(&mut p)?;
        data.push((idx, cstr(&mut p)?));
    }
    Ok((module, dll, data))
}

/// As natively (`wasmdll::ntdll_variables`): what gex-kernel's NT boot
/// fills in ntdll, in a native ntdll's own data.
fn ntdll_variables() -> [(&'static str, Vec<u8>); 3] {
    use gex_kernel::win::tramp;
    [
        ("__wine_syscall_dispatcher", (tramp::SYSCALL_DISPATCHER_32 as u32).to_le_bytes().to_vec()),
        ("__wine_unix_call_dispatcher", (tramp::UNIX_CALL_32 as u32).to_le_bytes().to_vec()),
        ("__wine_unixlib_handle", gex_kernel::win::NTDLL_HANDLE.to_le_bytes().to_vec()),
    ]
}

/// pelink's `gexw.forwards` custom section: (export name, `dll!name`).
fn forwards(wasm: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let Ok(wasmparser::Payload::CustomSection(c)) = payload {
            if c.name() == "gexw.forwards" {
                for line in String::from_utf8_lossy(c.data()).lines() {
                    if let Some((a, b)) = line.split_once(' ') {
                        out.push((a.to_string(), b.to_string()));
                    }
                }
            }
        }
    }
    out
}

/// Exported globals with constant initialisers: a PIC module's data
/// symbols, as offsets from `__memory_base` (i64 in a wasm64 module).
pub(super) fn exported_offsets(wasm: &[u8]) -> Result<HashMap<String, u32>, String> {
    let e = |e: wasmparser::BinaryReaderError| e.to_string();
    let mut imported_globals = 0u32;
    let mut inits: Vec<Option<u32>> = Vec::new();
    let mut out = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(e)? {
            wasmparser::Payload::ImportSection(r) => {
                for imp in r {
                    if let wasmparser::TypeRef::Global(_) = imp.map_err(e)?.ty {
                        imported_globals += 1;
                    }
                }
            }
            wasmparser::Payload::GlobalSection(r) => {
                for g in r {
                    let g = g.map_err(e)?;
                    let mut ops = g.init_expr.get_operators_reader();
                    inits.push(match ops.read().map_err(e)? {
                        wasmparser::Operator::I32Const { value } => Some(value as u32),
                        wasmparser::Operator::I64Const { value } => u32::try_from(value).ok(),
                        _ => None,
                    });
                }
            }
            wasmparser::Payload::ExportSection(r) => {
                for x in r {
                    let x = x.map_err(e)?;
                    if x.kind == wasmparser::ExternalKind::Global && x.index >= imported_globals {
                        if let Some(Some(v)) = inits.get((x.index - imported_globals) as usize) {
                            out.insert(x.name.to_string(), *v);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// `gex.wine_pe_slot(name)` (32-bit Wine): the function-table index of a named
/// function of Wine's PE module (win32u.so starts explorer's desktop thread
/// by it), or 0 when the module is not installed or has no such function.
#[no_mangle]
pub extern "C" fn gex_wine_pe_slot(ptr: u32, len: u32) -> u32 {
    let mem = crate::guest_bytes();
    let Some(b) = mem.get(ptr as usize..(ptr as usize).saturating_add(len as usize)) else { return 0 };
    let name = String::from_utf8_lossy(b).into_owned();
    let g = lock(&WINE_PE);
    // (of the process's own mode: the caller is a Unix side's)
    g.mods.iter().filter(|wp| !wp.alt).find_map(|wp| Some(wp.base + *wp.slots.get(&name)?)).unwrap_or(0)
}

/// `gex.code_changed(addr, len)` (32-bit Wine: Wine's NtFlushInstructionCache): the
/// translations of `[addr, addr + len)` are retired, as natively.
#[no_mangle]
pub extern "C" fn gex_code_changed(addr: u64, len: u32) {
    let len = len as u64;
    let (start, end) = (addr & !0xfff, (addr + len + 0xfff) & !0xfff);
    if end > start {
        crate::sigs::proc_home().mm().lock().log.retired_code.push((start, end));
    }
    gex_kernel::note_code_change();
}
