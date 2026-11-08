//! The web build of a project: the browser shim (gex's runtime compiled to
//! wasm) running the same program, in Node (`run.mjs`) or a page
//! (the user's `user/index.html`, which calls `app.mjs`'s `start()`).
//!
//! An eject given `--web-shim` copies the shim into `runtime/web/`, beside the
//! `gex.mjs` its gexc was built with. `build.sh` then also translates the
//! program for it into `build/web/` and writes `build/web/manifest.json`,
//! which says everything the launcher (`runtime/web/eject.mjs`) needs: the
//! modules, the files, the arguments, the replacements and the guest ABI they
//! are called with.
//!
//! v1 runs a single process with a single thread (gex's single-Worker path,
//! with a plain memory): a session that forked or started threads gets a
//! manifest that says so, and the launcher refuses it.

use crate::build::{BuiltImage, BuiltReplacement};
use crate::json::{self, obj, Json};
use crate::project::{rootfs_rel, Project};
use anyhow::{anyhow, Context, Result};
use gex_abi::ctx;
use gex_native::aot::{Abi, REPL_BASE, REPL_DATA_BYTES, REPL_END, REPL_STACK_BYTES};
use std::path::Path;

pub const SHIM: &str = "runtime/web/gex_browser.wasm";
pub const MANIFEST: &str = "build/web/manifest.json";

/// The browser runtime's JavaScript, as built into this gexc.
pub const GEX_MJS: &str = include_str!("../../../hosts/browser/js/gex.mjs");
/// The modules `gex.mjs` imports, vendored beside it.
pub const GEX_MJS_DEPS: [(&str, &str); 11] = [
    ("lazy-fetch.mjs", include_str!("../../../hosts/browser/js/lazy-fetch.mjs")),
    ("fetch-note.mjs", include_str!("../../../hosts/browser/js/fetch-note.mjs")),
    ("xcache.mjs", include_str!("../../../hosts/browser/js/xcache.mjs")),
    ("single.mjs", include_str!("../../../hosts/browser/js/single.mjs")),
    ("log.mjs", include_str!("../../../hosts/browser/js/log.mjs")),
    ("lazy-cache.mjs", include_str!("../../../hosts/browser/js/lazy-cache.mjs")),
    ("page-fs.mjs", include_str!("../../../hosts/browser/js/page-fs.mjs")),
    ("dgram-frames.mjs", include_str!("../../../hosts/browser/js/dgram-frames.mjs")),
    ("mode.mjs", include_str!("../../../hosts/browser/js/mode.mjs")),
    ("blocks.mjs", include_str!("../../../hosts/browser/js/blocks.mjs")),
    ("zstd-late.mjs", include_str!("../../../hosts/browser/js/zstd-late.mjs")),
];

/// Why this project cannot have a web build, if it cannot.
pub fn unsupported(project: &Project) -> Option<String> {
    if project.processes > 1 {
        Some(format!(
            "the recorded session ran {} processes; the web build runs one (gex's single-Worker path)",
            project.processes
        ))
    } else if project.threads > 1 {
        Some(format!(
            "the recorded session ran {} threads; the web build runs one (gex's single-Worker path, \
             with a plain memory)",
            project.threads
        ))
    } else {
        None
    }
}

/// The replacement module the web build uses for a native one: its
/// `.web.wasm` sibling (linked for a plain memory, which `build.sh` makes),
/// else the same file.
pub fn web_module(dir: &Path, module: &str) -> String {
    let web = module.strip_suffix(".wasm").map(|m| format!("{m}.web.wasm"));
    match web {
        Some(w) if dir.join(&w).exists() => w,
        _ => module.to_string(),
    }
}

type Files = (Vec<(String, String)>, Vec<(String, u64)>, Vec<String>, Vec<(String, u64)>);

/// The rootfs as the launcher bundles it: `(guest, project-relative)` files,
/// executable modes, empty directories, and every file's mtime.
fn files(dir: &Path) -> Result<Files> {
    let root = dir.join("build/rootfs");
    let (mut files, mut modes, mut dirs, mut mtimes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&d)
            .with_context(|| format!("reading {}", d.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        if entries.is_empty() && d != root {
            dirs.push(format!("/{}", d.strip_prefix(&root)?.display()));
        }
        for e in entries {
            let p = e.path();
            let meta = std::fs::metadata(&p).with_context(|| format!("reading {}", p.display()))?;
            if meta.is_dir() {
                stack.push(p);
                continue;
            }
            let guest = format!("/{}", p.strip_prefix(&root)?.display());
            let rel = p.strip_prefix(dir)?.display().to_string();
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o111 != 0 {
                modes.push((guest.clone(), 0o755));
            }
            // Kept by the eject and the build from the files the session read.
            use std::os::unix::fs::MetadataExt;
            mtimes.push((guest.clone(), meta.mtime().max(0) as u64));
            files.push((guest, rel));
        }
    }
    files.sort();
    dirs.sort();
    mtimes.sort();
    Ok((files, modes, dirs, mtimes))
}

/// Write `build/web/manifest.json`, for a build whose web modules are
/// `images` and whose replacements are `repl`.
pub fn manifest(
    dir: &Path,
    project: &Project,
    images: &[BuiltImage],
    pins: &[(String, u64, u64)],
    repl: &[BuiltReplacement],
    syscalls: Option<&std::collections::HashSet<u64>>,
    jit_fallback: bool,
) -> Result<()> {
    // A static program's image module owns the memory and tables. A dynamic
    // one's objects are all library modules (see `Translation::translate`),
    // started from a seed module that translates nothing.
    let own = images.iter().find(|i| i.kind == "image" && i.path == project.program.exe);
    let (module, closure, image_module): (String, Vec<&String>, Json) = match own {
        Some(i) => (
            i.module.clone(),
            images.iter().filter(|x| x.kind == "shared" && x.identity == i.identity).map(|x| &x.module).collect(),
            Json::Null,
        ),
        None => {
            let seed = "build/web/seed.wasm";
            std::fs::write(dir.join(seed), gex_xlate::seed_module(false).wasm)
                .with_context(|| format!("writing {seed}"))?;
            let img = images
                .iter()
                .find(|x| x.path == project.program.exe)
                .ok_or_else(|| anyhow!("internal: no web module for {}", project.program.exe))?;
            (
                seed.to_string(),
                images.iter().filter(|x| x.kind == "shared").map(|x| &x.module).collect(),
                json::s(&img.module),
            )
        }
    };
    // The libraries' function-tier units (`elf-unit`), the native build's
    // own files, which the page runs unguarded as natively.
    let units: Vec<&BuiltImage> = images.iter().filter(|i| i.kind == "elf-unit").collect();
    let unit_entries = |i: &BuiltImage| i.module.trim_end_matches("c.wasm").to_string() + "c.entries";
    let (files, modes, dirs, mtimes) = files(dir)?;
    let int_args: &[u8] = if gex_abi::X86 { &[7, 6, 2, 1, 8, 9] } else { &[0, 1, 2, 3, 4, 5, 6, 7] };
    let exe_file = format!("build/{}", rootfs_rel(&project.program.exe)?.display());
    // Every file the launcher reads, by its path here, and its size: the
    // page's progress has its whole download from the start.
    let sizes: Vec<(String, u64)> = {
        let mut rels: Vec<String> = vec![SHIM.to_string(), module.clone(), exe_file.clone()];
        rels.extend(closure.iter().map(|c| c.to_string()));
        rels.extend(units.iter().flat_map(|u| [u.module.clone(), unit_entries(u)]));
        rels.extend(files.iter().map(|(_, r)| r.clone()));
        rels.extend(repl.iter().filter(|r| !r.missing).map(|r| web_module(dir, &r.module)));
        rels.sort();
        rels.dedup();
        rels.into_iter()
            .filter_map(|r| std::fs::metadata(dir.join(&r)).ok().map(|md| (r, md.len())))
            .collect()
    };
    let m = obj(vec![
        ("format", json::s("gex-eject-web")),
        ("version", json::n(1)),
        ("arch", json::s(gex_abi::ARCH_NAME)),
        ("jit_fallback", json::b(jit_fallback)),
        ("shim", json::s(SHIM)),
        ("module", json::s(&module)),
        ("closure", json::strs(closure)),
        (
            "units",
            Json::Obj(
                units
                    .iter()
                    .map(|u| (u.identity.clone(), obj(vec![("wasm", json::s(&u.module)), ("entries", json::s(unit_entries(u)))])))
                    .collect(),
            ),
        ),
        (
            "elf_units",
            json::strs(units.iter().map(|u| u.path.rsplit('/').next().unwrap_or(&u.path).to_string()).collect::<Vec<_>>()),
        ),
        ("image_module", image_module),
        ("exe", json::s(&project.program.exe)),
        ("exe_file", json::s(exe_file)),
        ("argv", json::strs(&project.program.argv)),
        ("env", json::strs(&project.program.env)),
        ("cwd", json::opt(project.program.cwd.as_ref())),
        ("dirs", json::strs(&dirs)),
        ("files", Json::Obj(files.into_iter().map(|(g, r)| (g, json::s(r))).collect())),
        ("sizes", Json::Obj(sizes.into_iter().map(|(r, n)| (r, json::n(n))).collect())),
        ("modes", Json::Obj(modes.into_iter().map(|(g, m)| (g, json::n(m))).collect())),
        ("mtimes", Json::Obj(mtimes.into_iter().map(|(g, t)| (g, json::n(t))).collect())),
        (
            "pins",
            Json::Obj(
                pins.iter()
                    .map(|(p, base, span)| (p.clone(), obj(vec![("base", json::n(*base)), ("span", json::n(*span))])))
                    .collect(),
            ),
        ),
        (
            "replacements",
            Json::Arr(
                repl.iter()
                    .map(|r| {
                        obj(vec![
                            ("id", json::n(u64::from(r.id))),
                            ("module", json::s(web_module(dir, &r.module))),
                            ("export", json::s(&r.export)),
                            ("abi", json::s(r.abi.name())),
                            ("sig", json::s(&r.sig)),
                            ("symbol", json::s(&r.symbol)),
                            ("missing", json::b(r.missing)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "guest_abi",
            obj(vec![
                ("int_args", Json::Arr(int_args.iter().map(|r| json::n(ctx::x(*r) as u64)).collect())),
                ("fp_args", Json::Arr((0..8u8).map(|v| json::n(ctx::v(v) as u64)).collect())),
                ("ret", json::n(ctx::x(0) as u64)),
                ("fret", json::n(ctx::v(0) as u64)),
            ]),
        ),
        (
            "syscalls",
            match syscalls {
                None => Json::Null,
                Some(set) => obj(vec![
                    ("nr_offset", json::n(ctx::SYSCALL_NR as u64)),
                    ("allowed", Json::Arr(gex_native::aot::raw_syscalls(set).into_iter().map(json::n).collect())),
                ]),
            },
        ),
        (
            "region",
            obj(vec![
                ("end", json::n(REPL_END)),
                // The single Worker is thread slot 0.
                ("stack_top", json::n(REPL_BASE + REPL_DATA_BYTES + REPL_STACK_BYTES)),
            ]),
        ),
    ]);
    std::fs::write(dir.join(MANIFEST), json::write(&m)).with_context(|| format!("writing {MANIFEST}"))
}

/// The manifest of a project that has no web build, saying why.
pub fn unsupported_manifest(dir: &Path, why: &str) -> Result<()> {
    std::fs::create_dir_all(dir.join("build/web"))?;
    let m = obj(vec![("format", json::s("gex-eject-web")), ("version", json::n(1)), ("unsupported", json::s(why))]);
    std::fs::write(dir.join(MANIFEST), json::write(&m)).with_context(|| format!("writing {MANIFEST}"))
}

/// Check a replacement's web module links to the browser's plain memory.
pub fn check_module(dir: &Path, module: &str, export: &str, abi: Abi) -> Result<()> {
    let w = web_module(dir, module);
    let bytes = std::fs::read(dir.join(&w)).with_context(|| format!("reading {w}"))?;
    gex_native::aot::ReplacementModule::inspect_for(&bytes, false)
        .and_then(|m| m.check(export, abi))
        .with_context(|| format!("{w}, for the web build"))?;
    Ok(())
}

/// `runtime/web/eject.mjs`: the launcher, shared by Node and the page.
pub const EJECT_MJS: &str = include_str!("eject.mjs");

/// `run.mjs`: the Node launcher.
pub const RUN_MJS: &str = include_str!("run.mjs");

/// `serve.mjs`: a static server for the page (`user/index.html` at `/`, the
/// project's files beside it). gex's runtime needs
/// `SharedArrayBuffer`, so the page must be cross-origin isolated (the two
/// headers below); a plain static server is not enough.
pub const SERVE_MJS: &str = include_str!("serve.mjs");

/// `user/index.html`: the example page, the same for every kind of project
/// (a Linux program, a 32-bit Wine page). Written once; the user's
/// from then on. Everything the program needs is the app library's
/// (`app.mjs`, `start()`): the page only gives it an element. Its own file,
/// so a site put together without `gexc eject` can copy it as it is.
pub const SHELL_HTML: &str = include_str!("index.html");

/// Fail if a built web directory has or names `nt.html`: gex's test page is
/// never part of an ejected build (the page is the user's).
pub fn check_no_test_page(dir: &Path) -> Result<()> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                stack.push(p);
                continue;
            }
            let name = e.file_name().to_string_lossy().to_string();
            if name == "nt.html" {
                anyhow::bail!("{} is gex's test page, which an ejected build never ships", p.display());
            }
            if [".js", ".mjs", ".html", ".json"].iter().any(|x| name.ends_with(x)) {
                let b = std::fs::read(&p)?;
                if b.windows(7).any(|w| w == b"nt.html") {
                    anyhow::bail!("{} names nt.html (gex's test page), which an ejected build never ships", p.display());
                }
            }
        }
    }
    Ok(())
}

/// Write the example page to `user/index.html` unless the user has one.
pub fn write_shell(out: &Path) -> Result<()> {
    let p = out.join("user/index.html");
    if !p.exists() {
        crate::eject::write_file(&p, SHELL_HTML.as_bytes(), false)?;
    }
    Ok(())
}

/// `app.mjs` of a Linux program's project: the app library's `start()` on
/// the single-Worker launcher (`runtime/web/eject.mjs`), so the page is the
/// same as for every other kind of project.
pub const APP_MJS: &str = include_str!("app.mjs");

#[cfg(test)]
mod tests {
    use super::{EJECT_MJS, GEX_MJS, GEX_MJS_DEPS};

    /// Every module a Linux program's vendored runtime imports (`'./x.mjs'`)
    /// is vendored too: lazy-fetch.mjs's
    /// import of fetch-note.mjs, missed, failed every Linux page's load.
    #[test]
    fn the_linux_web_runtime_vendors_every_module_it_imports() {
        let mut have: Vec<&str> = GEX_MJS_DEPS.iter().map(|(f, _)| *f).collect();
        have.extend(["gex.mjs", "eject.mjs"]);
        let all = GEX_MJS_DEPS.iter().copied().chain([("gex.mjs", GEX_MJS), ("eject.mjs", EJECT_MJS)]);
        for (f, text) in all {
            for q in ['\'', '"'] {
                for part in text.split(&format!("{q}./")).skip(1) {
                    let Some(end) = part.find(q) else { continue };
                    let m = &part[..end];
                    if m.ends_with(".mjs") {
                        assert!(have.contains(&m), "{f} imports ./{m}, which GEX_MJS_DEPS leaves out");
                    }
                }
            }
        }
    }
}
