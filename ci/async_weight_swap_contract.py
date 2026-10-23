"""Finite concurrency model: fencing at async weight-swap completion.

This tests a protocol candidate, NOT actual vLLM behavior. Two writers stage
private snapshots and later try to atomically install them. The strong model
assumes a linearizable gate, trustworthy copy completion, and an engine lock
that covers the fence check and pointer swap. Staging never mutates live data.
"""
from dataclasses import dataclass, field
from itertools import combinations
import unittest


class Fenced(RuntimeError):
    pass


@dataclass
class Gate:
    term: int = 0
    epoch: int = 0
    live_digest: bytes = b"boot"
    open: bool = True

    def begin(self):
        self.term += 1
        self.open = False
        return self.term

    def commit(self, term, epoch, intended, engines):
        if term != self.term or self.open or epoch <= self.epoch:
            raise Fenced("obsolete publication")
        if not engines or any(
            e.term != term or e.live != intended or e.label != epoch
            or not e.quiescent or e.cache_epoch is not None for e in engines
        ):
            raise Fenced("some engine has not installed the snapshot")
        for e in engines:
            e.quiescent = False
        self.live_digest, self.epoch, self.open = intended, epoch, True

    def sound(self, engines):
        return not self.open or all(
            e.term == self.term and e.live == self.live_digest
            and e.label == self.epoch and e.cache_epoch is None
            and not e.quiescent for e in engines
        )


@dataclass
class Engine:
    live: bytes = b"boot"
    label: int = 0
    term: int = 0
    cache_epoch: int | None = None
    quiescent: bool = False
    staged: dict[int, bytes] = field(default_factory=dict)
    installed_term: int = 0

    def quiesce(self, term, gate):
        if term != gate.term or gate.open:
            raise Fenced("cannot quiesce for old term")
        self.quiescent = True
        self.cache_epoch = None
        self.term = term

    def stage(self, term, payload):
        # An old private copy may finish, but it does not affect serving.
        self.staged[term] = payload

    def finish(self, term, gate, *, unsafe=False):
        if term not in self.staged:
            raise Fenced("no completed staging buffer")
        if not unsafe and (term != gate.term or self.term != term
                           or gate.open or not self.quiescent):
            raise Fenced("stale asynchronous completion")
        # In production: atomically recheck fence and swap serving pointer.
        self.live = self.staged.pop(term)
        self.installed_term = term

    def set_label(self, term, epoch, gate):
        if (term != gate.term or self.term != term or gate.open
                or not self.quiescent or self.installed_term != term):
            raise Fenced("no matching completed install")
        self.label = epoch


STEPS = ("begin", "quiesce", "stage0", "stage1",
         "finish0", "finish1", "label", "commit")


def apply(gate, engines, states, who, step):
    state = states.setdefault(who, {})
    if step == "begin":
        state["term"] = gate.begin()
        return
    term = state["term"]
    epoch = 1 if who == "A" else 2
    payload = f"weights-{who}".encode()
    if step == "quiesce":
        for e in engines:
            e.quiesce(term, gate)
    elif step.startswith("stage"):
        engines[int(step[-1])].stage(term, payload)
    elif step.startswith("finish"):
        engines[int(step[-1])].finish(term, gate)
    elif step == "label":
        for e in engines:
            e.set_label(term, epoch, gate)
    elif step == "commit":
        gate.commit(term, epoch, payload, engines)
    else:
        raise ValueError(step)


class ConcurrencyTests(unittest.TestCase):
    def test_unsafe_late_completion_corrupts_published_weights(self):
        gate, es = Gate(), [Engine(), Engine()]
        a = gate.begin()
        for e in es:
            e.quiesce(a, gate)
            e.stage(a, b"weights-A")
        b = gate.begin()
        for e in es:
            e.quiesce(b, gate)
            e.stage(b, b"weights-B")
            e.finish(b, gate)
            e.set_label(b, 2, gate)
        gate.commit(b, 2, b"weights-B", es)
        self.assertTrue(gate.sound(es))
        # Delayed completion: no new version label, but weights change.
        es[0].finish(a, gate, unsafe=True)
        self.assertEqual(es[0].label, 2)
        self.assertFalse(gate.sound(es))

    def test_completion_fence_rejects_delayed_transfer(self):
        gate, es = Gate(), [Engine(), Engine()]
        a = gate.begin()
        for e in es:
            e.quiesce(a, gate)
            e.stage(a, b"weights-A")
        b = gate.begin()
        for e in es:
            e.quiesce(b, gate)
            e.stage(b, b"weights-B")
            e.finish(b, gate)
            e.set_label(b, 2, gate)
        gate.commit(b, 2, b"weights-B", es)
        with self.assertRaises(Fenced):
            es[0].finish(a, gate)
        self.assertTrue(gate.sound(es))

    def test_private_staging_does_not_change_serving_state(self):
        e = Engine()
        for term in (1, 2, 3):
            e.stage(term, f"snapshot-{term}".encode())
        self.assertEqual(e.live, b"boot")
        self.assertEqual(e.label, 0)

    def test_label_requires_matching_installed_snapshot(self):
        g, e = Gate(), Engine()
        t = g.begin()
        e.quiesce(t, g)
        e.stage(t, b"new")
        with self.assertRaises(Fenced):
            e.set_label(t, 1, g)
        e.finish(t, g)
        e.set_label(t, 1, g)
        g.commit(t, 1, b"new", [e])
        self.assertTrue(g.sound([e]))

    def test_partial_install_cannot_reopen_gate(self):
        g, es = Gate(), [Engine(), Engine()]
        t = g.begin()
        for e in es:
            e.quiesce(t, g)
            e.stage(t, b"new")
        es[0].finish(t, g)
        es[0].set_label(t, 1, g)
        with self.assertRaises(Fenced):
            g.commit(t, 1, b"new", es)
        self.assertFalse(g.open)

    def test_all_order_preserving_interleavings(self):
        count, successful_commits = 0, 0
        for places in combinations(range(16), 8):
            a_positions = set(places)
            gate, engines, writers = Gate(), [Engine(), Engine()], {}
            ia = ib = 0
            for position in range(16):
                if position in a_positions:
                    who, step = "A", STEPS[ia]
                    ia += 1
                else:
                    who, step = "B", STEPS[ib]
                    ib += 1
                was_open = gate.open
                try:
                    apply(gate, engines, writers, who, step)
                except Fenced:
                    pass
                if step == "commit" and not was_open and gate.open:
                    successful_commits += 1
                self.assertTrue(gate.sound(engines),
                                (places, position, who, step, gate, engines))
            count += 1
        self.assertEqual(count, 12870)
        self.assertGreater(successful_commits, 0)
        print("SAFE_INTERLEAVINGS", count,
              "SUCCESSFUL_COMMITS", successful_commits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
