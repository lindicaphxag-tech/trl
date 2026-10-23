# Digest-bound generation admission: reference contract (2026-10-23)

## Status and contribution boundary

**Research candidate, not an upstream TRL/vLLM implementation or a claim of priority.**
This document extends the earlier fork-only epoch and prefix-cache reference
models with an independently testable *installed-weight identity* check. Code:
[`ci/weight_digest_admission_contract.py`](../ci/weight_digest_admission_contract.py).
The additional check addresses a fault **not** excluded by a version-label
ACK: a server can acknowledge `v1` after a failed/partial tensor transfer.

## Explicit model

Let the trusted trainer provide a fixed publication pair `(e, H(W_e))`, where
`H` is a collision-resistant hash of a canonical serialization of the intended
weights. For each inference engine `i`, let `W_i` be *actually installed*
parameters, `l_i` its version label, `p_i` its cache epoch, and `q_i`
its paused/quiescent status. Let global ingress `G` be a fenced, atomic,
durable gate with a published epoch.

Desired safety property at the time admission is reopened:

```
G.open and G.epoch == e
  implies for every reachable engine i:
    not q_i
    and l_i == e
    and H(W_i) == H(W_e)
    and p_i == EMPTY
    and no older request remains active
```

**Important assumptions:** trusted exclusive weight writer; trustworthy
engine-side digest over installed (not staged or transfer-source) parameters;
correct wait/drain, cache invalidation and resume ACK semantics; all client
traffic is forced through the atomic ingress gate. In particular, no Byzantine
server, uncoordinated writer, ingress failover, network partition, or
post-admission external weight mutation is covered.

## Concrete counterexample to label-only admission

1. Close ingress; quiesce engines A and B and clear their prefix caches.
2. Install desired `W_1` on A; fault: B silently keeps `W_0`.
3. Both instances acknowledge label `v1`.
4. Label-only validation passes; resumes and opens ingress at `v1`.
5. B generates from `W_0` while the trajectory carries an apparent `v1` source.

Adding `H(W_B) == H(W_1)` before resuming rejects step 4 and leaves
ingress closed. The negative control and patched mode are run by the same
standalone standard-library test module, without GPU or Ray/vLLM.

## Conditional safety argument

For a successful publication, the gate closes *before* engine mutation.
The quiescence barrier blocks previously admitted work and invalidates prefix
KV. The exclusive writer then updates all engines. A trusted readback hash,
measured *after* transfer and *before* resumption, is compared with the trainer's
fixed manifest. All ACKs must succeed. No client is admitted while the gate
is closed, even if a server has resumed after another engine failed. Only once
all checks and resumptions succeed is the gate opened at epoch `e`.
Therefore the stated property holds under the assumptions above. Every
exception leaves ingress closed; this is a safety claim, **not** liveness.
The test suite explores selected faults, **not** the full reachable state space.

## Nonclaims and measurements required

- SHA-256 over small Python byte strings is a test oracle, not proof that
  actual vLLM tensor storage matches the reference, nor secure remote
  attestation of a potentially malicious server.
- A hash supplied by the publishing client, echoed by the inference server,
  is **not** installed-weight verification. The digest must be computed from
  the serving parameters, in a trusted context, after the transfer.
- GPU readback, canonical ordering, sharded weights, LoRA adapters, quantized
  weights, asynchronous transfers, and mixed-precision representations can
  invalidate a naive byte-for-byte hash scheme.
- Full-model hash computation is O(number of weight bytes) per publication
  and may synchronize GPU streams; no throughput or learning gains are claimed.
  Statistical spot checks are cheaper but do not establish exact identity.
- Exact same-revision vLLM integration, stale KV behavior, failure isolation,
  end-to-end rollout lineage and strong baseline throughput/learning tests
  are needed **before** proposing a production patch.
- **Prior art confirmed (2026-10-23): vLLM already ships a Weight Checker.**
  It exposes `POST /weight_checker` with `checksum`, `reset`, and `compare`
  actions, computing SHA-256 per model parameter across worker ranks. See
  https://docs.vllm.ai/en/latest/training/weight_checker/ and the actual
  server implementation documented at
  https://docs.vllm.ai/en/stable/api/vllm/entrypoints/rl/online/api_router/ .
  **The hash/verification primitive is therefore not our proposed novelty.**
  The possible *integration* contribution is coordinating an already-available
  checker with a multi-API-server admission gate, prefix-cache epoch, and
  per-token trajectory provenance, where baseline checksums truly represent
  the intended new policy.
- vLLM's checker requires development mode and hashes weight tensors by
  copying them to CPU; it excludes buffers, draft models, and LoRA adapters
  and has backend-specific limitations. Accordingly a default-on production
  checksum barrier is **not** a viable first upstream patch. Prefer an
  opt-in diagnostics/strict-source mode only after real latency measurement.
  A baseline taken from an *old* server snapshot is not a trusted manifest
  for intended *new* parameters; this must be solved before any claim of
  exact published-weight identity.

## Promotion criteria

1. Validate a real post-install tensor digest using documented vLLM APIs
   (single server, then multi-server, with a controlled failed transfer).
2. Show a real, end-to-end false-source sample in an unchanged baseline;
   rule out artifacts of the deterministic fixture.
3. Implement one opt-in strict-provenance boundary, preserving existing
   streaming/latency behavior in default mode.
4. Benchmark against the unchanged baseline, quiescent updates without
   digest, and digest-bound updates under fixed GPU budgets.
5. Resolve architecture ownership with upstream maintainers; keep the
   initial official PR minimal and attach unambiguous regression evidence.
