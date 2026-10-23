"""Two-writer / two-engine publication model, not a vLLM integration.

Assumptions: a linearizable durable ingress gate; every parameter/label
mutation checks a global fencing term; trusted installed-weight verification;
true request quiescence and cache invalidation. Not a security attestation.
"""
from dataclasses import dataclass
from itertools import combinations
import unittest

class FencedOut(RuntimeError):
    pass

@dataclass
class Engine:
    name: str
    weights: bytes = b"boot"
    label: int = 0
    fence_term: int = 0
    paused: bool = False
    prefix_epoch: int | None = None

    def prepare(self, term, gate):
        if term != gate.term or gate.open:
            raise FencedOut("obsolete prepare")
        self.fence_term, self.paused, self.prefix_epoch = term, True, None

    def install(self, term, epoch, payload, gate, *, enforce_fence=True):
        if enforce_fence and (term != gate.term or term != self.fence_term
                              or gate.open or not self.paused):
            raise FencedOut("obsolete or unsafe mutation")
        # Unfenced mode models legacy raw write endpoint.
        self.weights, self.label = payload, epoch

@dataclass
class Gate:
    term: int = 0
    epoch: int = 0
    digest: bytes = b"boot"
    open: bool = True

    def begin(self):
        self.term += 1
        self.open = False
        return self.term

    def commit(self, term, epoch, payload, engines):
        if term != self.term or self.open or epoch <= self.epoch or not engines:
            raise FencedOut("obsolete publisher")
        if any(e.fence_term != term or not e.paused
               or e.prefix_epoch is not None or e.weights != payload
               or e.label != epoch for e in engines):
            raise FencedOut("unverified publication")
        # Real resume acknowledgements must be durable and reconciled.
        for engine in engines:
            engine.paused = False
        self.epoch, self.digest, self.open = epoch, payload, True

    def admit(self, engine):
        if not self.open or engine.fence_term != self.term or engine.paused:
            raise FencedOut("unpublished generation")
        # Model specification oracle: NOT a real-time serving hash.
        if (engine.label != self.epoch or engine.weights != self.digest
                or engine.prefix_epoch is not None):
            raise FencedOut("source mismatch")
        return self.term, self.epoch

    def sound(self, engines):
        return not self.open or all(
            not e.paused and e.fence_term == self.term
            and e.label == self.epoch and e.weights == self.digest
            and e.prefix_epoch is None for e in engines)

PLAN = ("begin", "fence0", "fence1", "install0", "install1", "commit")

def action(gate, engines, writers, writer, step, *, enforce_fence=True):
    state = writers.setdefault(writer, {})
    if step == "begin":
        state["term"] = gate.begin()
        return
    term = state["term"]
    epoch = 1 if writer == "A" else 2
    payload = f"weights-{writer}".encode()
    if step.startswith("fence"):
        engines[int(step[-1])].prepare(term, gate)
    elif step.startswith("install"):
        engines[int(step[-1])].install(
            term, epoch, payload, gate, enforce_fence=enforce_fence)
    elif step == "commit":
        gate.commit(term, epoch, payload, engines)
    else:
        raise ValueError(step)

class PublicationTests(unittest.TestCase):
    def test_unfenced_stale_writer_corrupts_publication(self):
        g, es, writers = Gate(), [Engine("0"), Engine("1")], {}
        action(g, es, writers, "A", "begin")
        action(g, es, writers, "B", "begin")
        for s in PLAN[1:]:
            action(g, es, writers, "B", s)
        self.assertTrue(g.sound(es))
        action(g, es, writers, "A", "install0", enforce_fence=False)
        self.assertTrue(g.open)
        self.assertFalse(g.sound(es))

    def test_fence_rejects_obsolete_write(self):
        g, es, writers = Gate(), [Engine("0"), Engine("1")], {}
        action(g, es, writers, "A", "begin")
        action(g, es, writers, "B", "begin")
        for s in PLAN[1:]:
            action(g, es, writers, "B", s)
        with self.assertRaises(FencedOut):
            action(g, es, writers, "A", "install0")
        self.assertTrue(g.sound(es))
        self.assertEqual(g.admit(es[0]), (2, 2))

    def test_obsolete_commit_cannot_reopen_gate(self):
        g, es = Gate(), [Engine("0"), Engine("1")]
        old = g.begin()
        g.begin()
        with self.assertRaises(FencedOut):
            g.commit(old, 1, b"weights-A", es)
        self.assertFalse(g.open)

    def test_broken_transfer_or_label_cannot_commit(self):
        for failure in ("weight", "label"):
            g, es = Gate(), [Engine("0"), Engine("1")]
            term = g.begin()
            for engine in es:
                engine.prepare(term, g)
                engine.install(term, 1, b"weights-A", g)
            if failure == "weight":
                es[1].weights = b"old"
            else:
                es[1].label = 8
            with self.assertRaises(FencedOut):
                g.commit(term, 1, b"weights-A", es)
            self.assertFalse(g.open)

    def test_recovery_preempts_abandoned_writer(self):
        g, es = Gate(), [Engine("0"), Engine("1")]
        old = g.begin()
        es[0].prepare(old, g)
        es[0].install(old, 1, b"partial", g)
        self.assertFalse(g.open)
        new = g.begin()
        for e in es:
            e.prepare(new, g)
            e.install(new, 2, b"recovered", g)
        g.commit(new, 2, b"recovered", es)
        with self.assertRaises(FencedOut):
            es[0].install(old, 1, b"partial", g)
        self.assertTrue(g.sound(es))

    def test_exhaustive_order_preserving_interleavings(self):
        checked = 0
        for positions in combinations(range(12), 6):
            a_positions = set(positions)
            g, es, writers = Gate(), [Engine("0"), Engine("1")], {}
            ia = ib = 0
            for position in range(12):
                if position in a_positions:
                    who, step = "A", PLAN[ia]
                    ia += 1
                else:
                    who, step = "B", PLAN[ib]
                    ib += 1
                try:
                    action(g, es, writers, who, step)
                except FencedOut:
                    pass
                self.assertTrue(
                    g.sound(es), (positions, position, who, step, g, es))
            checked += 1
        self.assertEqual(checked, 924)
        print("CHECKED_INTERLEAVINGS", checked)

if __name__ == "__main__":
    unittest.main(verbosity=2)
