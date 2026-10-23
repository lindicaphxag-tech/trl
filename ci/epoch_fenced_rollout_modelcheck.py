"""Finite-state protocol audit: KV epoch, sampling epoch, and commit label.

This is a bounded abstract model, not vLLM execution or a novelty claim.
A safe publisher waits for outstanding requests or aborts them, commits
model weights + caller-chosen label, INVALIDATES reusable prefix KV state,
then resumes. Keeping old requests, or forgetting prefix invalidation, is
shown unsound even when new output tokens have uniform version labels.
"""
from dataclasses import dataclass, replace
import unittest


@dataclass(frozen=True)
class Request:
    id: int
    kv: int
    tokens: tuple[int, ...] = ()
    labels: tuple[int, ...] = ()


@dataclass(frozen=True)
class State:
    weights: int = 0
    label: int = 0
    paused: bool = False
    mode: str = ""
    prepared: bool = False
    prefix_kv: int | None = None
    active: tuple[Request, ...] = ()
    finished: tuple[Request, ...] = ()
    issued: frozenset[int] = frozenset()


def homogeneous(r):
    return (bool(r.tokens) and
            len(set(r.tokens)) == 1 and
            r.kv == r.tokens[0] and
            r.labels == r.tokens)


def edges(s, *, allow_keep=False, leave_prefix=False):
    if not s.paused:
        for rid in range(2):
            if rid not in s.issued:
                context = s.weights if s.prefix_kv is None else s.prefix_kv
                yield "start", replace(
                    s, active=s.active + (Request(rid, context),),
                    issued=s.issued | {rid})
        for i, r in enumerate(s.active):
            if len(r.tokens) < 2:
                rr = replace(r, tokens=r.tokens+(s.weights,),
                             labels=r.labels+(s.label,))
                yield "sample", replace(
                    s, active=s.active[:i]+(rr,)+s.active[i+1:])
            if r.tokens:
                yield "finish", replace(
                    s, active=s.active[:i]+s.active[i+1:],
                    finished=s.finished+(r,), prefix_kv=r.kv)
        if not s.active:
            yield "pause_wait", replace(s, paused=True, mode="wait")
        if s.active:
            yield "pause_abort", replace(s, paused=True, mode="abort",
                                          active=())
            if allow_keep:
                yield "pause_keep", replace(s, paused=True, mode="keep")
    else:
        if s.weights < 2 and not s.prepared:
            yield "prepare", replace(s, prepared=True)
        if s.prepared:
            # An idealized atomic weight+label commit, after a successful
            # transfer acknowledgement. Real server weights are NOT attested.
            yield "commit", replace(
                s, weights=s.weights+1, label=s.weights+1,
                prepared=False,
                prefix_kv=(s.prefix_kv if s.mode == "keep" or
                           leave_prefix else None))
        if not s.prepared:
            yield "resume", replace(s, paused=False, mode="")


def explore(*, allow_keep=False, leave_prefix=False, depth=10):
    initial=State()
    front={initial:()}
    visited={initial}
    checked=0
    witness=None
    for _ in range(depth):
        nxt={}
        for s, history in front.items():
            for action, state in edges(
                    s, allow_keep=allow_keep, leave_prefix=leave_prefix):
                checked+=1
                trace=history+(action,)
                for req in state.active+state.finished:
                    if (req.tokens and
                        (req.kv != req.tokens[-1] or
                         len(set(req.tokens)) > 1) and
                        witness is None):
                        witness=(trace, req)
                if state not in visited:
                    visited.add(state)
                    nxt[state]=trace
        front=nxt
    return len(visited), checked, witness


class ProtocolTests(unittest.TestCase):
    def test_wait_or_abort_with_prefix_invalidation(self):
        states, transitions, witness=explore(depth=9)
        self.assertGreater(states, 100)
        self.assertGreater(transitions, 100)
        self.assertIsNone(witness)
        print("SAFE_MODEL",states,transitions)

    def test_keep_crosses_model_epoch(self):
        _, _, found=explore(allow_keep=True,depth=9)
        self.assertIsNotNone(found)
        trace, req=found
        self.assertIn("pause_keep",trace)
        self.assertIn("commit",trace)
        self.assertNotEqual(req.kv,req.tokens[-1])
        self.assertEqual(req.tokens[-1],req.labels[-1])
        print("KEEP_WITNESS",trace,req)

    def test_wait_without_prefix_reset_is_insufficient(self):
        _, _, found=explore(leave_prefix=True,depth=11)
        self.assertIsNotNone(found)
        trace, req=found
        self.assertIn("pause_wait",trace)
        self.assertIn("commit",trace)
        self.assertNotEqual(req.kv,req.tokens[-1])
        self.assertEqual(req.tokens[-1],req.labels[-1])
        print("PREFIX_REUSE_WITNESS",trace,req)

    def test_indistinguishable_new_labels_can_hide_stale_kv(self):
        clean=Request(0,kv=1,tokens=(1,),labels=(1,))
        stale=Request(0,kv=0,tokens=(1,),labels=(1,))
        self.assertEqual(clean.labels,stale.labels)
        self.assertTrue(homogeneous(clean))
        self.assertFalse(homogeneous(stale))


if __name__=="__main__":
    unittest.main(verbosity=2)
