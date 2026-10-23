"""Fail-closed multi-engine epoch publication protocol (research prototype).

No real vLLM requests or weight hashes. Assumes exclusive weight writer and
that wait+clear_cache is a quiescence/invalidation barrier on each engine.
On any uncertain update/ack, all further publications are blocked until
manual reconciliation. The complete protocol still needs a real integration.
"""
import unittest


class Uncertain(RuntimeError):
    pass


class Engine:
    def __init__(self, name, fail=None):
        self.name, self.fail = name, fail
        self.label, self.epoch = "v0", 0
        self.prefix_epoch = 0
        self.paused = False
        self.staged = None
        self.calls = []

    def record(self, step):
        self.calls.append(step)
        if self.fail == step:
            raise IOError(f"{self.name}:{step}")

    def pause(self, *, mode, clear_cache):
        assert mode == "wait" and clear_cache
        self.record("pause")
        self.paused = True
        self.prefix_epoch = None

    def start_weight_update(self):
        assert self.paused
        self.record("start")

    def update_weights(self, epoch):
        assert self.paused
        self.record("transfer")
        self.staged = epoch

    def finish_weight_update(self, *, weight_version):
        assert self.paused and self.staged is not None
        self.record("finish")
        assert weight_version == f"v{self.staged}"
        self.epoch, self.label = self.staged, weight_version
        self.staged = None

    def weight_info(self):
        assert self.paused
        self.record("verify")
        return self.label

    def resume(self):
        assert self.label == f"v{self.epoch}"
        self.record("resume")
        self.paused = False


class Publisher:
    def __init__(self, engines):
        self.engines = engines
        self.published, self.poisoned = 0, False

    def publish(self, epoch):
        if self.poisoned or epoch != self.published + 1:
            raise Uncertain("unknown or non-monotonic publication")
        try:
            # Barrier: no instance updates until EVERY instance is paused.
            for e in self.engines:
                e.pause(mode="wait", clear_cache=True)
            for e in self.engines:
                e.start_weight_update()
            for e in self.engines:
                e.update_weights(epoch)
            for e in self.engines:
                e.finish_weight_update(weight_version=f"v{epoch}")
            for e in self.engines:
                if e.weight_info() != f"v{epoch}":
                    raise Uncertain("unconfirmed engine label")
            self.published = epoch
            for e in self.engines:
                e.resume()
        except Exception as exc:
            # Never blindly resume or retry partially transferred weights.
            self.poisoned = True
            raise Uncertain("manual reconciliation required") from exc




class IngressGate:
    """External, exclusive entry gate; a real deployment needs durable, atomic control."""

    def __init__(self):
        self.open = True
        self.epoch = 0

    def close(self):
        self.open = False

    def admit(self, engine):
        if (
            not self.open
            or engine.paused
            or engine.epoch != self.epoch
            or engine.prefix_epoch is not None
        ):
            raise Uncertain("generation not admitted: publication barrier")
        return self.epoch

    def open_at(self, epoch, engines):
        if self.open or any(
            e.paused or e.epoch != epoch or e.label != f"v{epoch}"
            or e.prefix_epoch is not None for e in engines
        ):
            raise Uncertain("not all engines have resumed at a coherent epoch")
        # This memory assignment represents a REQUIRED atomic durable
        # gate+epoch store; it is not a distributed consensus implementation.
        self.epoch, self.open = epoch, True


class GatedPublisher(Publisher):
    """Close ingress before mutation and reopen only after every resume ACK."""

    def __init__(self, engines, ingress):
        super().__init__(engines)
        self.ingress = ingress

    def publish(self, epoch):
        if self.poisoned or epoch != self.published + 1 or not self.ingress.open:
            raise Uncertain("previous publication unconfirmed")
        self.ingress.close()
        try:
            super().publish(epoch)
            self.ingress.open_at(epoch, self.engines)
        except Exception:
            # Even if some engines have already resumed, they MUST remain
            # inaccessible to clients until manual reconciliation.
            self.ingress.close()
            self.poisoned = True
            raise


class GatedPublisherTests(unittest.TestCase):
    def test_all_engine_counts_publish_through_one_epoch(self):
        for n in (1, 2, 4):
            engines = [Engine(str(i)) for i in range(n)]
            ingress = IngressGate()
            pub = GatedPublisher(engines, ingress)
            for epoch in (1, 2):
                pub.publish(epoch)
                self.assertEqual([ingress.admit(e) for e in engines], [epoch] * n)

    def test_failure_at_each_stage_never_reopens_ingress(self):
        checked = 0
        for idx in (0, 1):
            for stage in ("pause", "start", "transfer", "finish", "verify", "resume"):
                engines = [Engine("a"), Engine("b")]
                engines[idx].fail = stage
                ingress = IngressGate()
                pub = GatedPublisher(engines, ingress)
                self.assertRaises(Uncertain, pub.publish, 1)
                self.assertFalse(ingress.open)
                self.assertTrue(pub.poisoned)
                for e in engines:
                    self.assertRaises(Uncertain, ingress.admit, e)
                checked += 1
        self.assertEqual(checked, 12)

    def test_post_resume_ambiguous_failure_keeps_ingress_closed(self):
        class ResumeThenError(Engine):
            def resume(self):
                super().resume()
                raise IOError("lost ACK after successful resume")

        engines = [ResumeThenError("a"), Engine("b")]
        ingress = IngressGate()
        pub = GatedPublisher(engines, ingress)
        self.assertRaises(Uncertain, pub.publish, 1)
        self.assertFalse(ingress.open)
        self.assertFalse(engines[0].paused)  # A resumed engine is not safe ingress.
        self.assertRaises(Uncertain, ingress.admit, engines[0])
        self.assertRaises(Uncertain, ingress.admit, engines[1])

    def test_gate_remains_closed_during_staggered_resume(self):
        ingress = IngressGate()
        checked = []

        class ObservedEngine(Engine):
            def resume(self):
                super().resume()
                with self_subtest_context():
                    try:
                        ingress.admit(self)
                    except Uncertain:
                        checked.append(self.name)

        from contextlib import nullcontext
        def self_subtest_context():
            return nullcontext()

        engines = [ObservedEngine("a"), ObservedEngine("b")]
        GatedPublisher(engines, ingress).publish(1)
        self.assertEqual(checked, ["a", "b"])
        self.assertEqual([ingress.admit(e) for e in engines], [1, 1])


class PublisherTests(unittest.TestCase):
    def test_publication_order_and_cache_reset(self):
        es = [Engine("a"), Engine("b")]
        pub = Publisher(es)
        pub.publish(1)
        self.assertEqual(pub.published, 1)
        for e in es:
            self.assertEqual(e.calls, [
                "pause", "start", "transfer", "finish", "verify", "resume"])
            self.assertEqual((e.epoch, e.label), (1, "v1"))
            self.assertIsNone(e.prefix_epoch)
            self.assertFalse(e.paused)

    def test_all_single_stage_faults_fail_closed(self):
        checked = 0
        for which in range(2):
            for step in ("pause", "start", "transfer", "finish", "verify", "resume"):
                es = [Engine("a"), Engine("b")]
                es[which].fail = step
                pub = Publisher(es)
                with self.subTest(which=which, step=step):
                    self.assertRaises(Uncertain, pub.publish, 1)
                    self.assertTrue(pub.poisoned)
                    self.assertRaises(Uncertain, pub.publish, 2)
                    for e in es:
                        # Any resumed engine has fully committed and verified.
                        if not e.paused and "resume" in e.calls:
                            self.assertEqual((e.epoch, e.label), (1, "v1"))
                checked += 1
        self.assertEqual(checked, 12)

    def test_dishonest_label_ack_does_not_resume(self):
        class Bad(Engine):
            def weight_info(self):
                super().weight_info()
                return "v0"
        es = [Engine("a"), Bad("b")]
        pub = Publisher(es)
        self.assertRaises(Uncertain, pub.publish, 1)
        self.assertTrue(pub.poisoned)
        self.assertTrue(all(e.paused for e in es))

    def test_multi_engine_monotone_epochs(self):
        for n in (1, 2, 4):
            with self.subTest(engines=n):
                es = [Engine(str(i)) for i in range(n)]
                pub = Publisher(es)
                pub.publish(1)
                pub.publish(2)
                self.assertEqual({e.label for e in es}, {"v2"})
                self.assertRaises(Uncertain, pub.publish, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
