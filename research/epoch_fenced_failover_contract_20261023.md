# Fenced inference-weight publication under publisher failover (2026-10-23)

## Scope / status

**Abstract, CPU-only research prototype. Not a deployed vLLM/Ray/TRL fix and
not a claim to have invented fencing tokens or linearizable gates.**
Code: [ci/epoch_fenced_failover_contract.py](../ci/epoch_fenced_failover_contract.py).

This is an additional failure mode beyond the already modeled
label-versus-weight discrepancy and cached-KV/source discrepancy:
**stale publisher A can overwrite verified policy B after B opens ingress**.
If raw weight or version mutation remains available, comparing the new weights
once during publication is insufficient to preserve published policy identity.

## Counterexample

A obtains publication term 1 but stalls before transferring to an engine.
B obtains term 2, closes ingress, quiesces both engines, installs identical
weights and labels, verifies them, then resumes and opens ingress.
A's delayed `update_weights` or `update_weight_version` executes after
B's successful publication without checking a fencing term. Even if B's
weights and labels were verified at commit time, an old write can invalidate
that result while the gate is open.

The unittest `test_unfenced_stale_writer_corrupts_publication` is a
deterministic **negative control** for this ordering. The fixed candidate
requires a monotonically increasing term on *all* weight and label mutation
operations, verified atomically against the authoritative term at execution.
An older writer is rejected even if its HTTP request was queued earlier.

## Conditional safety invariant

For term `t`, epoch `e`, intended installed weights `W_e`, all inference
engines `i`, and the externally controlled ingress gate `G`:

```
G.open and G.term=t and G.epoch=e
  => each reachable engine i:
     engine.fence_term == t
     and engine.installed_weights == W_e
     and engine.label == e
     and not engine.paused
     and engine.prefix_cache_epoch is EMPTY
```

The reference harness uses a byte-string equality **oracle** for installed
weights. An actual implementation would require a trusted post-install
verification of the serving tensors (or a weaker explicitly stated contract)
and must prevent all out-of-band mutation. Per-token old logprobs and exact
trajectory lineage remain separate requirements.

### Required assumptions

1. `G.begin` and `G.commit` are durable and linearizable; a coordinator
   restart cannot restore an old `term`. **The prototype does not build
   consensus or implement a distributed store.**
2. Every model-weight and version-label mutation endpoint checks that same
   durable term at the *mutation linearization point*; checking at HTTP receipt
   is insufficient if asynchronous transfers can finish later.
3. Every externally routed generation request passes through the fenced
   ingress; old in-flight requests are genuinely drained/aborted and their
   prefix state invalidated before reopening.
4. Installed-weight checks are trustworthy for intended tensors, including
   shard identity. No Byzantine servers, external untracked writers,
   reconfiguration of the engine set, or silent post-commit parameter mutation.
5. Fail-stop semantics: ambiguous ACKs keep ingress closed until a new
   fenced publisher reconciles state; **liveness is not established**.

Under these assumptions, successful `commit(t)` implies the invariant at
admission reopen, and every stale delayed write is rejected by the term
check. This is a proof outline of the *model's* contract, not a proof of a
real vLLM implementation.

## Evidence

Standalone tests cover a precise unfenced counterexample, rejection of
obsolete writes and commits, verification failures, recovery takeover,
and **all 924 order-preserving interleavings** of two six-step publisher
programs (`C(12,6)`). The search does **not** enumerate network partitions,
time-based lease expiry, arbitrary duplicated writes, reconfiguration,
multiple generations or implementation-specific queues.

## Upstream compatibility and originality boundary

- vLLM already has `/start_weight_update`, `/update_weights`,
  `/finish_weight_update`, `/update_weight_version`, `/weight_info`
  and `/pause`/`/resume` endpoints; its development-mode
  `/weight_checker` provides parameter checksums. Those primitives,
  distributed fencing, and epoch protocols **are existing techniques**.
- Official docs warn that a failed `/update_weights` can leave parameters
  mid-reload while health checks still succeed; the actual policy must not
  interpret a healthy HTTP process as coherent weights.
- This candidate has **not** modified vLLM endpoints to enforce terms.
  Adding a term to a TRL-side publisher *alone* cannot stop another holder
  of the unfenced server API. The change would have to live in a trusted
  sidecar/proxy with exclusive mutation access, or in the inference server
  itself, after upstream design agreement.
- Existing RL serving infrastructure, e.g. NVIDIA Dynamo's rollout
  update coordination, must be compared before claiming novelty.
- A lower-risk initial upstream contribution is evidence-backed validation
  or an optional strict mode with clear performance budgets, not an
  unconditional multi-service redesign.

## Next independent promotion evidence

(1) reproduce the stale-writer sequence using actual vLLM HTTP API calls
with multiple clients; (2) prove mutation fencing holds across asynchronous
weight-transfer completion; (3) verify real tensors, all model shards, cache
epochs and request sources; (4) compare operational overhead and learner
outcomes under fixed GPU budgets; (5) obtain maintainer feedback on which
project should own the fence, before proposing production changes.

References:
- https://docs.vllm.ai/en/latest/training/weight_transfer/
- https://docs.vllm.ai/en/v0.32.0/training/weight_checker/
- https://docs.dynamo.nvidia.com/dynamo/reinforcement-learning/rollout-weight-updates
