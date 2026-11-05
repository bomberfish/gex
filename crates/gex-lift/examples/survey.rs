//! Which instructions of a flat image gex cannot lift, before a run has to reach them:
//! every aligned word of `[start, end)` lifted alone, the distinct words that come out
//! illegal or undecoded printed with how often they appear and where first.
//!
//!     cargo run -p gex-lift --example survey -- <image> <start> <end>
use gex_ir::Term;

fn parse(s: &str) -> u64 {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).expect("hex"),
        None => s.parse().expect("number"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img = std::fs::read(&args[1]).expect("image");
    let (start, end) = (parse(&args[2]), parse(&args[3]).min(img.len() as u64));
    let word = |at: u64| u32::from_le_bytes(img[at as usize..at as usize + 4].try_into().unwrap());
    let mut seen: std::collections::BTreeMap<u32, (u64, u64)> = Default::default();
    let mut at = start;
    while at + 4 <= end {
        let raw = word(at);
        let limits = gex_lift::BlockLimits { max_insts: 1 };
        // One instruction, fetched from a one-word image at its own address.
        let block = gex_lift::lift_block(move |pc| (pc == at).then_some(raw), at, limits);
        if let Term::Illegal { raw: r, .. } = block.term {
            if r == raw {
                let e = seen.entry(raw).or_insert((0, at));
                e.0 += 1;
            }
        }
        at += 4;
    }
    for (raw, (n, first)) in seen {
        println!("{raw:08x} {n} {first:#x}");
    }
}
