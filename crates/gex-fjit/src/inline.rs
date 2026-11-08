//! Inlining small callees into their callers, in GIR.
//!
//! clang inlines the C bodies of small guest functions into their callers
//! (the guest's own `noinline` binds the guest compiler, not the C), which
//! is most of its lead on call-heavy code. The same in GIR: a direct call to
//! a small, non-recursive function of the same module becomes a jump to a
//! copy of its blocks, and each of its returns a branch:
//!
//! * a `ret` to the call's return address continues at the call's
//!   continuation; any other target leaves for it (the address carried in
//!   `GuestReg::Pc`, which nothing else reads, to the leaving block). No
//!   stack-pointer check: nothing is assumed of the callee once it is
//!   inline -- its registers are the caller's locals, exact on every path --
//!   so the check that guards a summary's "preserved" no longer has
//!   anything to guard;
//! * a tail call to another function becomes a call with the same return
//!   address and continuation (the callee returns where the inlined one
//!   would have), and so does an indirect jump that is not a switch;
//! * everything else (calls, syscalls, leaves) is as it was, in the copy.
//!
//! The inlined code checks no page of its own on entry; the caller's
//! checks cover the pages of every block it holds, inlined ones included
//! (`runtime`'s page lists are computed after inlining).

use gex_ir::func::{BlockId, Callee, Edge, FBlock, FTerm, Function};
use gex_ir::{CmpOp, Expr, GuestReg, Stmt, Ty, Value};
use std::collections::{HashMap, HashSet};

/// Callees up to this many statements are inlined (`GEX_FJIT_INLINE`; 0
/// turns it off).
pub const MAX_CALLEE_STMTS: usize = 48;
/// A caller stops growing at this many blocks.
pub const MAX_CALLER_BLOCKS: usize = 400;

fn limit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("GEX_FJIT_INLINE").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_CALLEE_STMTS))
}

fn size(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.stmts.len() + 1).sum()
}

/// Inline into every function of `funcs` the small callees among them,
/// callees first (so a chain collapses bottom-up).
pub fn inline_all(funcs: &mut [Function]) -> usize {
    let lim = limit();
    if lim == 0 {
        return 0;
    }
    let index: HashMap<u64, usize> = funcs.iter().enumerate().map(|(k, f)| (f.entry, k)).collect();
    // Callees first: a post-order of the direct-call graph, cycles broken
    // (a function in a cycle is never inlined).
    let callees = |f: &Function| -> Vec<usize> {
        f.blocks
            .iter()
            .filter_map(|b| match b.term {
                FTerm::Call { callee: Callee::Direct(t), .. } => index.get(&t).copied(),
                _ => None,
            })
            .collect()
    };
    let n = funcs.len();
    let mut order = Vec::with_capacity(n);
    let mut state = vec![0u8; n]; // 0 new, 1 on the stack, 2 done
    let mut recursive: HashSet<usize> = HashSet::new();
    for root in 0..n {
        if state[root] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(root, callees(&funcs[root]), 0)];
        state[root] = 1;
        while let Some(top) = stack.last_mut() {
            if top.2 < top.1.len() {
                let c = top.1[top.2];
                top.2 += 1;
                match state[c] {
                    0 => {
                        state[c] = 1;
                        let cs = callees(&funcs[c]);
                        stack.push((c, cs, 0));
                    }
                    1 => {
                        // A cycle: everything on the stack from `c` up.
                        let from = stack.iter().position(|e| e.0 == c).unwrap_or(0);
                        recursive.extend(stack[from..].iter().map(|e| e.0));
                    }
                    _ => {}
                }
            } else {
                let (f, _, _) = stack.pop().unwrap();
                state[f] = 2;
                order.push(f);
            }
        }
    }
    let mut done = 0;
    for &k in &order {
        loop {
            let site = funcs[k].blocks.iter().position(|b| match b.term {
                FTerm::Call { callee: Callee::Direct(t), ret: Some(_), .. } => index.get(&t).is_some_and(|&c| {
                    c != k && !recursive.contains(&c) && funcs[c].entries.is_empty() && size(&funcs[c]) <= lim
                }),
                _ => false,
            });
            let Some(site) = site else { break };
            if funcs[k].blocks.len() >= MAX_CALLER_BLOCKS {
                break;
            }
            let FTerm::Call { callee: Callee::Direct(t), .. } = funcs[k].blocks[site].term else { unreachable!() };
            let callee = funcs[index[&t]].clone();
            inline_at(&mut funcs[k], site, &callee);
            done += 1;
        }
    }
    done
}

/// Replace the call ending block `site` of `f` with a copy of `g`.
fn inline_at(f: &mut Function, site: usize, g: &Function) {
    let FTerm::Call { ret_pc, ret: Some(cont), .. } = f.blocks[site].term.clone() else {
        unreachable!("inline_at needs a call with a continuation")
    };
    let base = f.blocks.len() as u32;
    let map = |e: &Edge| Edge { to: BlockId(e.to.0 + base), args: Vec::new() };
    let mut leaves: Vec<FBlock> = Vec::new();
    let mut copies: Vec<FBlock> = Vec::with_capacity(g.blocks.len());
    for b in &g.blocks {
        let mut c = b.clone();
        c.term = match &b.term {
            FTerm::Jump(e) => FTerm::Jump(map(e)),
            FTerm::Branch { cond, taken, not_taken } => FTerm::Branch { cond: *cond, taken: map(taken), not_taken: map(not_taken) },
            FTerm::Switch { target, cases, table } => FTerm::Switch {
                target: *target,
                cases: cases.iter().map(|(a, e)| (*a, map(e))).collect(),
                table: table.clone(),
            },
            FTerm::Call { callee, ret_pc, ret } => FTerm::Call { callee: *callee, ret_pc: *ret_pc, ret: ret.as_ref().map(map) },
            FTerm::Syscall { next_pc, next, imm } => FTerm::Syscall { next_pc: *next_pc, next: next.as_ref().map(map), imm: *imm },
            FTerm::TailCall { callee } => FTerm::Call { callee: *callee, ret_pc, ret: Some(cont.clone()) },
            // An indirect jump out of the callee is its tail call (the
            // emitter's reading of `Unresolved`), so here a call too: left
            // as it was it would pass on the caller's own return contract.
            FTerm::Unresolved { target } => FTerm::Call { callee: Callee::Indirect(*target), ret_pc, ret: Some(cont.clone()) },
            FTerm::Return { target } => {
                // ok = target == ret_pc; Pc = target (for the leaving block).
                let pc = c.term_pc;
                let k = next_value(&c);
                let (rv, ok) = (Value(k), Value(k + 1));
                c.push(pc, Stmt::WriteReg { reg: GuestReg::Pc, val: *target });
                c.push(pc, Stmt::Def { dst: rv, expr: Expr::ConstI64(ret_pc) });
                c.push(pc, Stmt::Def { dst: ok, expr: Expr::Cmp { op: CmpOp::Eq, ty: Ty::I64, a: *target, b: rv } });
                let leave = base + g.blocks.len() as u32 + leaves.len() as u32;
                let mut l = FBlock::synthetic(pc, FTerm::Unresolved { target: Value(0) });
                l.push(pc, Stmt::Def { dst: Value(0), expr: Expr::ReadReg(GuestReg::Pc) });
                leaves.push(l);
                FTerm::Branch { cond: ok, taken: cont.clone(), not_taken: Edge::to(BlockId(leave)) }
            }
            other => other.clone(),
        };
        copies.push(c);
    }
    f.blocks[site].term = FTerm::Jump(Edge::to(BlockId(base)));
    f.blocks.extend(copies);
    f.blocks.extend(leaves);
    f.n_values = f.n_values.max(g.n_values + 2);
}

/// A value number no statement of `b` defines (values are block-local in
/// memory form, renumbered later).
fn next_value(b: &FBlock) -> u32 {
    let mut n = 0;
    for s in &b.stmts {
        gex_ir::func::for_each_def(s, |v| n = n.max(v.0 + 1));
        s.for_each_use(|v| n = n.max(v.0 + 1));
    }
    let mut t = 0;
    b.term.for_each_use(|v| t = t.max(v.0 + 1));
    n.max(t)
}
