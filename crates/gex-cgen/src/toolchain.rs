//! Driving clang and wasm-ld.
//!
//! Translation units compile in parallel, one clang process each; the objects
//! link into one standalone module that the splice then takes apart. Missing
//! tools are an error, never a quiet fallback to tier-1.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// clang's flags for generated C.
pub const CFLAGS: &[&str] = &[
    "--target=wasm32",
    "-O2",
    // No switch-to-lookup-table conversion: its tables are data segments,
    // which the splice cannot place (a guest-memory address); jump tables
    // (br_table) are unaffected.
    "-mllvm",
    "-switch-to-lookup=false",
    // No SLP vectorization: it packs a run of scalar compares into a vector
    // of booleans and reduces it through a bitcast, which wasm lowers with a
    // shadow-stack slot. A function with a shadow-stack frame cannot be
    // spliced, and a trap-free one cannot be left to tier-1 either. Loop
    // vectorization is unaffected.
    "-fno-slp-vectorize",
    "-mtail-call",
    "-msimd128",
    "-matomics",
    "-mbulk-memory",
    "-mnontrapping-fptoint",
    "-msign-ext",
    "-fno-strict-aliasing",
    "-fno-delete-null-pointer-checks",
    "-ffp-contract=off",
    "-fno-math-errno",
    "-ffreestanding",
    "-nostdlib",
    "-fno-exceptions",
    "-Wno-unused-label",
    "-Wno-unused-variable",
    "-Wno-unused-but-set-variable",
];

/// [`CFLAGS`] for this process's guest memory: `--target=wasm64` for a
/// memory64 one, the flags as they are otherwise.
pub fn cflags() -> Vec<&'static str> {
    CFLAGS
        .iter()
        .map(|f| if *f == "--target=wasm32" && gex_abi::mem64() { "--target=wasm64" } else { *f })
        .collect()
}

/// wasm-ld's memory flags: the wasm32 guest memory's maximum, or a memory64
/// one of `LIMIT64` (`-mwasm64`).
fn ld_memory_flags() -> Vec<String> {
    if gex_abi::mem64() {
        vec!["-mwasm64".into(), format!("--max-memory={}", gex_abi::LIMIT64)]
    } else {
        vec!["--max-memory=4294901760".into()]
    }
}

/// What a whole-image unit adds: it is linked
/// position-independent, so its function pointers are indices into the
/// process's function table from `__table_base`, and calls between its
/// functions stay direct (hidden visibility).
pub const IMAGE_CFLAGS: &[&str] = &["-fPIC", "-fvisibility=hidden"];

fn tool(env: &str, default: &str) -> String {
    std::env::var(env).unwrap_or_else(|_| default.to_string())
}

pub fn clang() -> String {
    tool("GEX_CLANG", "clang")
}

pub fn wasm_ld() -> String {
    tool("GEX_WASM_LD", "wasm-ld")
}

/// A scratch directory for one build: `GEX_CGEN_KEEP` if set (kept), else a
/// fresh temporary directory (removed on drop).
pub struct WorkDir {
    pub path: PathBuf,
    keep: bool,
}

impl WorkDir {
    pub fn new() -> Result<WorkDir> {
        if let Some(d) = std::env::var_os("GEX_CGEN_KEEP") {
            let path = PathBuf::from(d);
            std::fs::create_dir_all(&path)
                .with_context(|| format!("creating {}", path.display()))?;
            return Ok(WorkDir { path, keep: true });
        }
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("gex-cgen-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(WorkDir { path, keep: false })
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// The address space each clang may take, in MiB: past it clang fails, and
/// the unit's largest function is left to tier-1 ([`OutOfMemory`]). Default
/// [`DEFAULT_CLANG_MEM_MB`]; `GEX_CGEN_CLANG_MEM_MB` or [`set_clang_mem_mb`]
/// (gexc's `--clang-mem-mb`); 0, no limit.
pub const DEFAULT_CLANG_MEM_MB: u64 = 8192;

/// `u64::MAX`: not set.
static CLANG_MEM_MB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

pub fn set_clang_mem_mb(mb: u64) {
    CLANG_MEM_MB.store(mb, std::sync::atomic::Ordering::Relaxed);
}

pub fn clang_mem_mb() -> u64 {
    match CLANG_MEM_MB.load(std::sync::atomic::Ordering::Relaxed) {
        u64::MAX => {
            static ENV: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            *ENV.get_or_init(|| {
                std::env::var("GEX_CGEN_CLANG_MEM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_CLANG_MEM_MB)
            })
        }
        n => n,
    }
}

/// A clang that ran out of the memory [`clang_mem_mb`] gives it, on `src`.
#[derive(Debug)]
pub struct OutOfMemory {
    pub src: PathBuf,
}

impl std::fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "clang ran out of its {} MiB on {}", clang_mem_mb(), self.src.display())
    }
}

impl std::error::Error for OutOfMemory {}

/// A clang for one source, under [`clang_mem_mb`]'s limit.
///
/// The limit is installed `unix`ly except on macOS, which has no working
/// address-space ceiling: Darwin's `setrlimit` takes `RLIMIT_AS`,
/// `RLIMIT_DATA` and `RLIMIT_RSS` only at infinity and fails any lower value
/// with EINVAL (the child then never starts, which reads as "is clang
/// installed?"). Clang runs unlimited there; the ceiling is a guard against a
/// runaway clang, not a property of the output.
fn clang_command() -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(clang());
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mb = clang_mem_mb();
        if mb != 0 {
            use std::os::unix::process::CommandExt;
            let bytes = mb.saturating_mul(1 << 20) as libc::rlim_t;
            // SAFETY: setrlimit is async-signal-safe; it is all the child
            // does between fork and exec.
            unsafe {
                cmd.pre_exec(move || {
                    let lim = libc::rlimit { rlim_cur: bytes, rlim_max: bytes };
                    if libc::setrlimit(libc::RLIMIT_AS, &lim) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
    }
    cmd
}

/// [`run`] for a clang on `src`: a failure that is clang running out of
/// memory under its limit is an [`OutOfMemory`].
fn run_clang(cmd: &mut Command, src: &Path) -> Result<()> {
    let what = format!("clang on {}", src.display());
    let out = cmd
        .output()
        .with_context(|| format!("running {what} ({:?}); is it installed?", cmd.get_program()))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    #[cfg(unix)]
    let signalled = std::os::unix::process::ExitStatusExt::signal(&out.status).is_some();
    #[cfg(not(unix))]
    let signalled = false;
    let oom = ["out of memory", "Allocation failed", "bad_alloc", "Cannot allocate memory"]
        .iter()
        .any(|m| err.contains(m));
    if clang_mem_mb() != 0 && (oom || signalled) {
        return Err(anyhow::Error::new(OutOfMemory { src: src.to_path_buf() }));
    }
    let lines: Vec<&str> = err.lines().take(60).collect();
    bail!("{what} failed ({}):\n{}", out.status, lines.join("\n"));
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let out = cmd
        .output()
        .with_context(|| format!("running {what} ({:?}); is it installed?", cmd.get_program()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = err.lines().take(60).collect();
        bail!("{what} failed ({}):\n{}", out.status, lines.join("\n"));
    }
    Ok(())
}

/// How many clang processes at once: `GEX_CGEN_JOBS`, else the CPUs this
/// process may run on.
pub fn jobs() -> usize {
    std::env::var("GEX_CGEN_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .max(1)
}

/// `clang --version`, for object-cache keys.
pub fn clang_version() -> Result<String> {
    let out = Command::new(clang())
        .arg("--version")
        .output()
        .with_context(|| format!("running {} --version; is it installed?", clang()))?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The flags every whole-image unit is compiled with, as one string, for
/// cache keys.
pub fn cflags_key() -> String {
    let mut v: Vec<String> = cflags().into_iter().chain(IMAGE_CFLAGS.iter().copied()).map(|s| s.to_string()).collect();
    v.extend(extra_cflags());
    v.join(" ")
}

/// Precompile `dir/gex_abi.h` with the units' flags; returns the `.pch`.
pub fn precompile_header(dir: &Path) -> Result<PathBuf> {
    let pch = dir.join("gex_abi.h.pch");
    let mut cmd = Command::new(clang());
    cmd.args(cflags())
        .args(IMAGE_CFLAGS)
        .args(extra_cflags())
        .arg("-x")
        .arg("c-header")
        .arg(dir.join("gex_abi.h"))
        .arg("-o")
        .arg(&pch);
    run(&mut cmd, "clang (precompiling gex_abi.h)")?;
    Ok(pch)
}

/// Compile one unit to `obj` (written beside it first, then renamed, so a
/// killed build leaves no truncated object in a cache).
pub fn compile_one(dir: &Path, src: &Path, obj: &Path, pch: Option<&Path>) -> Result<()> {
    let tmp = obj.with_extension(format!("o.tmp-{}", std::process::id()));
    let mut cmd = clang_command();
    cmd.args(cflags()).args(IMAGE_CFLAGS).args(extra_cflags()).arg("-I").arg(dir);
    if let Some(p) = pch {
        cmd.arg("-include-pch").arg(p);
    }
    cmd.arg("-c").arg(src).arg("-o").arg(&tmp);
    if let Err(e) = run_clang(&mut cmd, src) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, obj)?;
    Ok(())
}

/// Compile every `.c` in `sources` to an object beside it, in parallel.
pub fn compile_all(dir: &Path, sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let objs: Vec<PathBuf> = sources.iter().map(|s| s.with_extension("o")).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let errors = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs().min(sources.len().max(1)) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= sources.len() {
                    return;
                }
                let mut cmd = clang_command();
                cmd.args(cflags())
                    .args(extra_cflags())
                    .arg("-I")
                    .arg(dir)
                    .arg("-c")
                    .arg(&sources[i])
                    .arg("-o")
                    .arg(&objs[i]);
                if let Err(e) = run_clang(&mut cmd, &sources[i]) {
                    errors.lock().unwrap().push(e);
                }
            });
        }
    });
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    Ok(objs)
}

/// Link objects into one module exporting `exports`.
pub fn link(dir: &Path, objs: &[PathBuf], exports: &[String], out: &Path) -> Result<()> {
    link_with_table(dir, objs, exports, out, false)
}

/// Link a whole-image unit: position-
/// independent, importing the process's function table and filling its
/// slots from `__table_base` with one element segment.
pub fn link_image(dir: &Path, objs: &[PathBuf], out: &Path) -> Result<()> {
    let rsp = dir.join("link.rsp");
    let mut text = String::new();
    for o in objs {
        text.push_str(&o.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(&rsp, text)?;
    let mut cmd = Command::new(wasm_ld());
    cmd.args(["--experimental-pic", "-shared", "--no-entry", "--import-memory", "--shared-memory"])
        .args(ld_memory_flags())
        .args(["--allow-undefined", "-O2"]);
    if LTO.load(std::sync::atomic::Ordering::Relaxed) {
        // Code generation happens here: the flags clang's own would have had.
        cmd.args(["--lto-O2", "-mllvm", "-switch-to-lookup=false"]).arg(format!("--thinlto-jobs={}", jobs()));
    }
    cmd.arg(format!("@{}", rsp.display()))
        .arg("-o")
        .arg(out);
    run(&mut cmd, "wasm-ld")
}

/// [`link`], importing the function table (`env.__indirect_function_table`)
/// when `import_table`: a whole-image unit's `call_indirect`s go through the
/// process's body table, which the host provides.
pub fn link_with_table(dir: &Path, objs: &[PathBuf], exports: &[String], out: &Path, import_table: bool) -> Result<()> {
    // A response file: a closure can have a hundred thousand exports.
    let rsp = dir.join("link.rsp");
    let mut text = String::new();
    for e in exports {
        text.push_str("--export=");
        text.push_str(e);
        text.push('\n');
    }
    for o in objs {
        text.push_str(&o.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(&rsp, text)?;
    let mut cmd = Command::new(wasm_ld());
    cmd.args([
        "--no-entry",
        "--import-memory",
        "--shared-memory",
    ])
    .args(ld_memory_flags())
    .args(["--allow-undefined", "-O2"])
    .args(import_table.then_some("--import-table"))
    .arg(format!("@{}", rsp.display()))
    .arg("-o")
    .arg(out);
    run(&mut cmd, "wasm-ld")
}

/// `GEX_CGEN_CFLAGS`: extra clang flags, for pricing code-generation choices
/// (not part of any cache key: build with `--no-cache` when using it).
fn extra_cflags() -> Vec<String> {
    let mut v: Vec<String> = std::env::var("GEX_CGEN_CFLAGS")
        .map(|v| v.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    if LTO.load(std::sync::atomic::Ordering::Relaxed) {
        v.push("-flto=thin".to_string());
    }
    v
}

/// ThinLTO for what this process builds next (a link group's module):
/// objects are bitcode (another object key, through
/// [`cflags_key`]), and [`link_image`] optimises and generates code across
/// them.
static LTO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_lto(on: bool) {
    LTO.store(on, std::sync::atomic::Ordering::Relaxed);
}
