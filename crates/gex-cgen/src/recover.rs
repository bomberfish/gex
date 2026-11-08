//! A stand-in for function recovery, kept to compare gex-xlate's against.
//!
//! Forms memory-form [`Function`]s from the tier-1 block list
//! alone: entries are the function symbols and every direct call target; a
//! function owns the blocks its entry reaches through direct jumps, branches
//! and syscall fall-throughs, stopping at other functions' entries. It knows
//! nothing about jump tables (an indirect jump is `Unresolved`, which leaves
//! the function tier) and nothing about interior entries (those blocks keep
//! their tier-1 bodies anyway, because only a function's entry is given a C
//! body). What it produces is always *correct* to compile, only sometimes
//! less complete than real recovery would be.

use gex_ir::func::{BlockId, Callee, Edge, FBlock, FTerm, Form, Function};
use gex_ir::{Block, Stmt, Term, Value};
use std::collections::{BTreeSet, HashMap};

/// Every function the blocks and the symbol addresses describe.
pub fn functions(blocks: &[&Block], symbols: &[u64]) -> Vec<Function> {
    let by_addr: HashMap<u64, &Block> = blocks.iter().map(|b| (b.start, *b)).collect();
    let mut entries: BTreeSet<u64> = symbols
        .iter()
        .copied()
        .filter(|a| by_addr.contains_key(a))
        .collect();
    for b in blocks {
        if let (Some(_), Term::Goto { target }) = (b.call_return, &b.term) {
            if by_addr.contains_key(target) {
                entries.insert(*target);
            }
        }
    }
    entries
        .iter()
        .map(|&e| form(e, &by_addr, &entries))
        .collect()
}

fn form(entry: u64, by_addr: &HashMap<u64, &Block>, entries: &BTreeSet<u64>) -> Function {
    let mut f = Function {
        entry,
        name: None,
        entries: Vec::new(),
        blocks: Vec::new(),
        n_values: 0,
        form: Form::Memory,
    };
    let mut ids: HashMap<u64, BlockId> = HashMap::new();
    let mut work: Vec<u64> = vec![entry];
    // Assign ids in discovery order; block 0 is the entry.
    let mut order: Vec<u64> = Vec::new();
    let id_of = |a: u64, ids: &mut HashMap<u64, BlockId>, order: &mut Vec<u64>, work: &mut Vec<u64>| {
        *ids.entry(a).or_insert_with(|| {
            order.push(a);
            work.push(a);
            BlockId(order.len() as u32 - 1)
        })
    };
    id_of(entry, &mut ids, &mut order, &mut work);
    work.clear();
    work.push(entry);
    let mut pending_synth: Vec<FBlock> = Vec::new();
    let mut lifted: HashMap<u64, FBlock> = HashMap::new();
    while let Some(a) = work.pop() {
        if lifted.contains_key(&a) {
            continue;
        }
        let b = by_addr[&a];
        // Value numbers are function-wide: offset this block's.
        let base = f.n_values;
        f.n_values += b.n_values;
        let rn = |v: Value| Value(v.0 + base);
        let mut stmts = b.stmts.clone();
        for s in &mut stmts {
            renumber(s, base);
        }
        let n = stmts.len();
        // An edge inside the function, or a synthetic tail call out of it.
        let mut edge = |t: u64, synth: &mut Vec<FBlock>, work: &mut Vec<u64>| -> Option<Edge> {
            if (t != entry && entries.contains(&t)) || !by_addr.contains_key(&t) {
                // Leaves: a synthetic block whose only job is the tail call.
                synth.push(FBlock::synthetic(t, FTerm::TailCall { callee: Callee::Direct(t) }));
                return None;
            }
            Some(Edge::to(id_of(t, &mut ids, &mut order, work)))
        };
        let term = match &b.term {
            Term::Goto { target } => match b.call_return {
                Some(r) => {
                    let ret = if by_addr.contains_key(&r) && !(entries.contains(&r) && r != entry) {
                        Some(Edge::to(id_of(r, &mut ids, &mut order, &mut work)))
                    } else {
                        None
                    };
                    FTerm::Call {
                        callee: Callee::Direct(*target),
                        ret_pc: r,
                        ret,
                    }
                }
                None if (*target != entry && entries.contains(target))
                    || !by_addr.contains_key(target) =>
                {
                    FTerm::TailCall {
                        callee: Callee::Direct(*target),
                    }
                }
                None => FTerm::Jump(Edge::to(id_of(*target, &mut ids, &mut order, &mut work))),
            },
            Term::Branch {
                cond,
                taken,
                not_taken,
            } => {
                let t = edge(*taken, &mut pending_synth, &mut work);
                let t = t.unwrap_or_else(|| {
                    Edge::to(BlockId(u32::MAX - pending_synth.len() as u32 + 1))
                });
                let nt = edge(*not_taken, &mut pending_synth, &mut work);
                let nt = nt.unwrap_or_else(|| {
                    Edge::to(BlockId(u32::MAX - pending_synth.len() as u32 + 1))
                });
                FTerm::Branch {
                    cond: rn(*cond),
                    taken: t,
                    not_taken: nt,
                }
            }
            Term::Indirect { target, is_ret } => {
                let t = rn(*target);
                if *is_ret {
                    FTerm::Return { target: t }
                } else if let Some(r) = b.call_return {
                    let ret = if by_addr.contains_key(&r) && !(entries.contains(&r) && r != entry) {
                        Some(Edge::to(id_of(r, &mut ids, &mut order, &mut work)))
                    } else {
                        None
                    };
                    FTerm::Call {
                        callee: Callee::Indirect(t),
                        ret_pc: r,
                        ret,
                    }
                } else {
                    FTerm::Unresolved { target: t }
                }
            }
            Term::Syscall { next, imm } => {
                let e = if by_addr.contains_key(next) && !(entries.contains(next) && *next != entry) {
                    Some(Edge::to(id_of(*next, &mut ids, &mut order, &mut work)))
                } else {
                    None
                };
                FTerm::Syscall {
                    next_pc: *next,
                    next: e,
                    imm: *imm,
                }
            }
            Term::Illegal { raw, pc } => FTerm::Illegal { raw: *raw, pc: *pc },
        };
        let fb = FBlock {
            start: b.start,
            end: b.end,
            synthetic: false,
            params: Vec::new(),
            pcs: vec![b.start; n],
            stmts,
            insts: vec![b.start],
            term_pc: if matches!(b.term, Term::Syscall { .. }) {
                b.end - gex_abi::SYSCALL_INST_BYTES
            } else {
                b.end.saturating_sub(gex_abi::INST_ALIGN)
            },
            term,
        };
        lifted.insert(a, fb);
    }
    // Lifted blocks in id order, then the synthetic exits, patching the
    // placeholder ids (counted down from u32::MAX) to their real positions.
    let n_lifted = order.len() as u32;
    for a in &order {
        f.blocks.push(lifted.remove(a).expect("every ordered block was lifted"));
    }
    let n_synth = pending_synth.len() as u32;
    for fb in &mut f.blocks {
        if let FTerm::Branch {
            taken, not_taken, ..
        } = &mut fb.term
        {
            for e in [taken, not_taken] {
                if e.to.0 > u32::MAX - n_synth {
                    let k = u32::MAX - e.to.0; // 0-based synthetic index
                    e.to = BlockId(n_lifted + k);
                }
            }
        }
    }
    f.blocks.extend(pending_synth);
    f
}

fn renumber(s: &mut Stmt, base: u32) {
    gex_ir::func::for_each_use_mut(s, |v| v.0 += base);
    match s {
        Stmt::Def { dst, .. } | Stmt::MemCmp { dst, .. } => dst.0 += base,
        Stmt::X87 { outs, .. } => {
            for v in outs {
                v.0 += base;
            }
        }
        Stmt::StrOp {
            out_count,
            out_dst,
            out_src,
            out_a,
            out_b,
            ..
        } => {
            for v in [out_count, out_dst, out_src, out_a, out_b] {
                v.0 += base;
            }
        }
        _ => {}
    }
}
