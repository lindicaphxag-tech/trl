"""Prototype: exact output-token provenance vs scalar rollout version.

This is a research *contract checker*, not a patched TRL Trainer or an
authenticated vLLM attestation service. Production evidence is pinned in
research/vllm_token_provenance_contract_20261023.md.
"""
from dataclasses import dataclass
from pathlib import Path
import ast
import re
import unittest

_VERSION = re.compile(r"v(0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class SourceDecision:
    kind: str
    versions: tuple[int, ...]
    reason: str


def classify_spans(spans, token_count, *, committed_labels, lora=False):
    """Validate exact half-open partition and known, committed base labels.

    A label is only *declared* by the server, not a cryptographic attestation.
    LoRA source identity is separate from the base-model weight_version.
    """
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError("invalid completion length")
    if lora:
        return SourceDecision("unattested", (), "base-weight spans do not identify the LoRA adapter")
    if spans is None:
        return SourceDecision("unattested", (), "server did not return token provenance")
    if not isinstance(spans, list):
        return SourceDecision("malformed", (), "weight_versions must be a list")
    if token_count == 0:
        return SourceDecision("empty" if not spans else "malformed", (), "no generated tokens")
    if not spans:
        return SourceDecision("malformed", (), "positive output length has no spans")
    left = 0
    ids = []
    for span in spans:
        if not isinstance(span, dict):
            return SourceDecision("malformed", (), "span is not an object")
        a, b, label = span.get("start"), span.get("end"), span.get("version")
        if (not isinstance(a, int) or isinstance(a, bool)
            or not isinstance(b, int) or isinstance(b, bool)
            or a != left or b <= a or b > token_count):
            return SourceDecision("malformed", (), "hole/overlap/invalid half-open boundary")
        if not isinstance(label, str):
            return SourceDecision("malformed", (), "non-string weight version")
        match = _VERSION.fullmatch(label)
        if match is None or label not in committed_labels:
            return SourceDecision("unattested", (), "unknown or uncommitted user-defined label")
        ids.append(int(match.group(1)))
        left = b
    if left != token_count:
        return SourceDecision("malformed", (), "version spans do not cover last token")
    versions = tuple(ids)
    return SourceDecision(
        "uniform" if min(versions) == max(versions) else "mixed",
        versions,
        "valid declared server-version coverage (not a weights checksum)",
    )


def group_admission(decisions, *, learner_version, max_staleness, cache_epoch_verified=False):
    """Conservative all-or-nothing admission for one GRPO reward group."""
    if max_staleness < 0 or learner_version < 0:
        raise ValueError("invalid version threshold")
    if not decisions:
        return "reject-empty"
    if any(d.kind in {"malformed", "empty"} for d in decisions):
        return "reject-invalid"
    if any(d.kind == "unattested" for d in decisions):
        return "reject-unattested"
    if any(d.kind == "mixed" for d in decisions):
        return "reject-mixed-token-source"
    assert all(d.kind == "uniform" for d in decisions)
    all_versions = [d.versions[0] for d in decisions]
    if min(all_versions) != max(all_versions):
        return "reject-mixed-group-source"
    # Sampling-version spans alone cannot prove that KV-cache state was
    # computed under the same weights. This proof must come from a protocol
    # that resets/pins context or returns a real cache-epoch certificate.
    if not cache_epoch_verified:
        return "reject-cache-unverified"
    if learner_version - all_versions[0] > max_staleness:
        return "reject-stale"
    if all_versions[0] > learner_version:
        return "reject-future"
    return "accept"


class TokenProvenanceContractTests(unittest.TestCase):
    def setUp(self):
        self.committed = {"v0", "v1", "v2", "v3"}
        self.spans = lambda *triples: [
            {"version": ver, "start": a, "end": b} for ver, a, b in triples
        ]

    def check(self, spans, length, *, lora=False):
        return classify_spans(
            spans, length, committed_labels=self.committed, lora=lora)

    def test_midrequest_pause_keep_is_mixed_source(self):
        # vLLM's documented mode=keep can produce tokens from two versions.
        d = self.check(self.spans(("v1", 0, 2), ("v2", 2, 4)), 4)
        self.assertEqual(d.kind, "mixed")
        self.assertEqual(
            group_admission([d], learner_version=2, max_staleness=1),
            "reject-mixed-token-source",
        )
        # A request-only v1 stamp cannot encode the second token source.

    def test_malformed_boundaries_and_uncommitted_labels(self):
        bad = [
            self.spans(("v1", 0, 2), ("v2", 1, 4)),
            self.spans(("v1", 0, 2), ("v2", 3, 4)),
            self.spans(("v1", 0, 2)),
            self.spans(("v1", 0, 0), ("v2", 0, 4)),
            self.spans(("v1", 0, 5)),
        ]
        for spans in bad:
            with self.subTest(spans=spans):
                self.assertEqual(self.check(spans, 4).kind, "malformed")
        self.assertEqual(self.check(self.spans(("default", 0, 4)), 4).kind, "unattested")
        self.assertEqual(self.check(self.spans(("v9", 0, 4)), 4).kind, "unattested")
        self.assertEqual(self.check(None, 4).kind, "unattested")

    def test_lora_adapter_identity_is_a_different_contract(self):
        self.assertEqual(
            self.check(self.spans(("v1", 0, 4)), 4, lora=True).kind,
            "unattested",
        )

    def test_empty_and_malicious_span_metadata(self):
        self.assertEqual(self.check([], 0).kind, "empty")
        self.assertEqual(self.check(self.spans(("v0", 0, 1)), 0).kind, "malformed")
        self.assertEqual(self.check([{"version": "v1", "start": True, "end": 4}], 4).kind, "malformed")
        self.assertEqual(self.check([{"version": "v1", "start": 0, "end": -1}], 4).kind, "malformed")
        self.assertEqual(self.check([{"version": 1, "start": 0, "end": 4}], 4).kind, "malformed")
        self.assertEqual(
            group_admission([self.check(None, 3)], learner_version=2, max_staleness=1),
            "reject-unattested",
        )

    def test_group_atomicity_and_staleness(self):
        v1 = self.check(self.spans(("v1", 0, 4)), 4)
        v2 = self.check(self.spans(("v2", 0, 4)), 4)
        self.assertEqual(group_admission([v1, v2], learner_version=2, max_staleness=2, cache_epoch_verified=True),
                         "reject-mixed-group-source")
        self.assertEqual(group_admission([v1, v1], learner_version=2, max_staleness=1, cache_epoch_verified=True),
                         "accept")
        self.assertEqual(group_admission([v1, v1], learner_version=3, max_staleness=1, cache_epoch_verified=True),
                         "reject-stale")
        self.assertEqual(group_admission([v2, v2], learner_version=1, max_staleness=1, cache_epoch_verified=True),
                         "reject-future")

    def test_uniform_token_labels_do_not_attest_old_cached_context(self):
        # The server may label every newly sampled token v2 while its cached
        # prefix keys/values still originate from v1 under pause(mode=keep).
        d = self.check(self.spans(("v2", 0, 4)), 4)
        self.assertEqual(d.kind, "uniform")
        self.assertEqual(
            group_admission([d], learner_version=2, max_staleness=0),
            "reject-cache-unverified",
        )
        # Only an externally justified cache-consistency proof can enable
        # the strict homogeneous-policy admission contract.
        self.assertEqual(
            group_admission(
                [d], learner_version=2, max_staleness=0,
                cache_epoch_verified=True,
            ),
            "accept",
        )

    def test_gap_exists_in_pinned_current_trl_source(self):
        worker = Path("trl/experimental/async_grpo/async_rollout_worker.py").read_text()
        client = Path("trl/experimental/async_grpo/vllm_client.py").read_text()
        parsed = ast.parse(worker)
        methods = [
            n for n in ast.walk(parsed) if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "_generate_one_turn"
        ]
        self.assertEqual(len(methods), 1)
        source = ast.unparse(methods[0])
        self.assertIn("/v1/completions", source)
        self.assertNotIn("return_weight_versions", source)
        self.assertNotIn("weight_versions", source)
        self.assertIn('"mode": "keep"', client)
        self.assertNotIn("weight_version", client)
        print("SOURCE GAP: no per-token spans and pause=keep with no committed-version label")


if __name__ == "__main__":
    unittest.main(verbosity=2)
