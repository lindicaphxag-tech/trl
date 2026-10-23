"""Reference experiment: fail-closed, digest-bound rollout publication.

This is a deterministic MODEL, not a tested TRL/vLLM integration.
Goal: distinguish version-label acknowledgement from verification of the
weights that a non-Byzantine server actually installed.

Assumptions: a single trusted writer, a durable/atomic shared ingress gate,
trustworthy server-side digest computed over installed tensors, and a wait+
cache-invalidate barrier that really drains old requests. No untrusted remote
attestation, scheduler fairness, crash recovery, or live tensor cost is modeled.
"""

from __future__ import annotations

import hashlib
import unittest


def fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class UnsafePublication(RuntimeError):
    pass


class ModelEngine:
    def __init__(self, name: str, initial_weights: bytes = b"epoch-0"):
        self.name = name
        self.weights = initial_weights
        self.label = 0
        self.paused = False
        self.prefix_epoch: int | None = 0
        self.ignore_transfer = False
        self.flip_transfer = False
        self.fail_after_resume = False
        self.skip_prefix_clear = False
        self.calls: list[str] = []

    def pause_wait_clear_cache(self) -> None:
        self.calls.append("pause")
        self.paused = True
        if not self.skip_prefix_clear:
            self.prefix_epoch = None

    def transfer(self, payload: bytes) -> None:
        self.calls.append("transfer")
        if not self.paused:
            raise UnsafePublication("transfer while serving")
        if not self.ignore_transfer:
            self.weights = payload + b"corrupt" if self.flip_transfer else payload

    def publish_label(self, epoch: int) -> None:
        self.calls.append("publish-label")
        if not self.paused:
            raise UnsafePublication("label changed while serving")
        # An acknowledgement of an arbitrary version label is not attestation.
        self.label = epoch

    def installed_digest(self) -> str:
        self.calls.append("digest")
        # Modeled trusted readback of the *installed* tensor bytes.
        return fingerprint(self.weights)

    def resume(self) -> None:
        self.calls.append("resume")
        self.paused = False
        if self.fail_after_resume:
            raise OSError("lost resume acknowledgement")


class Ingress:
    def __init__(self):
        self.open = True
        self.epoch = 0

    def admit(self, engine: ModelEngine) -> int:
        if (not self.open or engine.paused or engine.label != self.epoch
                or engine.prefix_epoch is not None):
            raise UnsafePublication("incoherent or unpublished generation")
        return self.epoch


class DigestBoundPublisher:
    def __init__(self, engines: list[ModelEngine], ingress: Ingress,
                 *, verify_digest: bool = True):
        if not engines:
            raise ValueError("at least one engine required")
        self.engines = engines
        self.ingress = ingress
        self.verify_digest = verify_digest
        self.published = 0
        self.poisoned = False

    def publish(self, epoch: int, payload: bytes) -> None:
        if (self.poisoned or epoch != self.published + 1
                or not self.ingress.open):
            raise UnsafePublication("unconfirmed or nonmonotonic publication")
        # Must be a durable atomic fence in a real distributed deployment.
        self.ingress.open = False
        try:
            expected = fingerprint(payload)  # Trusted trainer manifest.
            for engine in self.engines:
                engine.pause_wait_clear_cache()
                if not engine.paused or engine.prefix_epoch is not None:
                    raise UnsafePublication("incomplete quiescence/cache barrier")
            for engine in self.engines:
                engine.transfer(payload)
            for engine in self.engines:
                engine.publish_label(epoch)
            for engine in self.engines:
                if engine.label != epoch:
                    raise UnsafePublication("label acknowledgement mismatch")
                if self.verify_digest and engine.installed_digest() != expected:
                    raise UnsafePublication("installed weights mismatch manifest")
            for engine in self.engines:
                engine.resume()
            if any(e.paused or e.label != epoch or
                   e.prefix_epoch is not None for e in self.engines):
                raise UnsafePublication("resume incomplete")
            self.published = epoch
            self.ingress.epoch = epoch
            self.ingress.open = True
        except Exception as exc:
            self.poisoned = True
            self.ingress.open = False
            raise UnsafePublication("publication uncertain: ingress stays closed") from exc


class ContractTests(unittest.TestCase):
    def test_multi_engine_two_epochs(self):
        for n in (1, 2, 4):
            engines = [ModelEngine(str(i)) for i in range(n)]
            gate = Ingress()
            publisher = DigestBoundPublisher(engines, gate)
            for epoch in (1, 2):
                publisher.publish(epoch, f"weights-{epoch}".encode())
                self.assertEqual([gate.admit(e) for e in engines], [epoch] * n)
                self.assertEqual({fingerprint(e.weights) for e in engines},
                                 {fingerprint(f"weights-{epoch}".encode())})

    def test_label_only_admits_stale_weights_negative_control(self):
        engines = [ModelEngine("healthy"), ModelEngine("broken")]
        engines[1].ignore_transfer = True
        gate = Ingress()
        label_only = DigestBoundPublisher(engines, gate, verify_digest=False)
        label_only.publish(1, b"weights-1")
        self.assertEqual(gate.admit(engines[1]), 1)  # False provenance claim.
        self.assertNotEqual(engines[1].installed_digest(), fingerprint(b"weights-1"))

    def test_digest_blocks_stale_weight_even_with_correct_label(self):
        engines = [ModelEngine("healthy"), ModelEngine("broken")]
        engines[1].ignore_transfer = True
        gate = Ingress()
        publisher = DigestBoundPublisher(engines, gate)
        with self.assertRaises(UnsafePublication):
            publisher.publish(1, b"weights-1")
        self.assertEqual(engines[1].label, 1)
        self.assertTrue(publisher.poisoned)
        self.assertFalse(gate.open)
        for engine in engines:
            with self.assertRaises(UnsafePublication):
                gate.admit(engine)

    def test_digest_blocks_corrupt_tensor_with_correct_label(self):
        engines = [ModelEngine("healthy"), ModelEngine("corrupt")]
        engines[1].flip_transfer = True
        gate = Ingress()
        publisher = DigestBoundPublisher(engines, gate)
        with self.assertRaises(UnsafePublication):
            publisher.publish(1, b"weights-1")
        self.assertFalse(gate.open)
        self.assertTrue(publisher.poisoned)

    def test_prefix_cache_clear_is_required(self):
        engines = [ModelEngine("cached")]
        engines[0].skip_prefix_clear = True
        gate = Ingress()
        with self.assertRaises(UnsafePublication):
            DigestBoundPublisher(engines, gate).publish(1, b"weights-1")
        self.assertFalse(gate.open)

    def test_resume_ack_loss_is_fail_closed(self):
        engines = [ModelEngine("a"), ModelEngine("b")]
        engines[1].fail_after_resume = True
        gate = Ingress()
        pub = DigestBoundPublisher(engines, gate)
        with self.assertRaises(UnsafePublication):
            pub.publish(1, b"weights-1")
        self.assertFalse(gate.open)
        self.assertTrue(pub.poisoned)
        self.assertTrue(all(not e.paused for e in engines))
        with self.assertRaises(UnsafePublication):
            pub.publish(2, b"weights-2")

    def test_stale_or_replayed_epoch_rejected(self):
        gate = Ingress()
        pub = DigestBoundPublisher([ModelEngine("a")], gate)
        pub.publish(1, b"weights-1")
        for epoch in (0, 1, 3):
            with self.assertRaises(UnsafePublication):
                pub.publish(epoch, b"payload")
        self.assertEqual(gate.epoch, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
