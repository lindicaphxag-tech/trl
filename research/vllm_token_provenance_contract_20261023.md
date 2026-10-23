# Research contract: token-level policy provenance across asynchronous RL rollouts

Status: **research-only, not an accepted upstream fix or validated algorithmic improvement**. 2026-10-23.
Target issue: [TRL #7861](https://github.com/huggingface/trl/issues/7861).
Unit proof-of-concept: `ci/vllm_token_source_admission_contract.py`.

## Source-verified missing guarantee

Current TRL AsyncGRPO `_generate_one_turn` requests token IDs and log-probs but **does not request `return_weight_versions`**, nor read the returned `weight_versions` half-open spans. Its scalar `request_version` describes the requested / advertised policy, not necessarily every token's source.

Current TRL `vllm_client.pause` uses `?mode=keep&clear_cache=false`; this allows in-flight continuations to resume after weight replacement. Its `finish_weight_update` issues POST without the optional `weight_version` field. Consequently merely setting `return_weight_versions=true` would not produce trustworthy ordered model-version labels: labels may stay `default` unless the trainer explicitly commits one.

vLLM upstream source (as of Oct 23): `vllm/entrypoints/rl/online/api_router.py` accepts `weight_version` in `/finish_weight_update`, `vllm/entrypoints/openai/completion/protocol.py` accepts `return_weight_versions`, and `vllm/outputs.py` defines `WeightVersionSpan(version,start,end)`. Its [Async RL documentation](https://docs.vllm.ai/en/latest/training/async_rl/) states that `mode=keep` can produce a single completion with both old and new weight-version spans; if KV cache isn't cleared, cached computation from old weights may persist.

## Strict, compatible design path — proposal, not implemented in TRL

1. **Do not change legacy behavior by default**. Opt in only for vLLM endpoints that advertise the requisite protocol. Older deployments lack `return_weight_versions`.
2. **At dense weight commit while paused**, bind a monotonic training version to vLLM by committing `weight_version="vN"` in the `finish_weight_update` transaction or a verified post-finish update before resume. Failure to commit must not silently advance the trainer's provenance epoch. The label is still user-assigned, not a cryptographic hash of installed weights.
3. **At each dense generation**, request `return_token_ids=true` and `return_weight_versions=true`. Require a complete, non-overlapping half-open partition of exactly the `n` generated tokens and a trainer-known label for every segment. Treat missing / `default` / uncommitted / structurally invalid spans as **unattested or invalid**, never silently reinterpret them as the request version.
4. **Per-turn and per-group identity**. Tokens in a turn may be mixed, turns in a conversation may differ, and conversations in one GRPO reward group may differ. A conservative **strict homogeneous** policy rejects the entire reward group when the versions cannot all be made equal; this avoids silently fragmenting group-normalized advantages. Alternative per-token off-policy correction is a different algorithm and requires a strong baseline and bias/variance justification; do not present the homogeneous rule as necessary for all valid RL objectives.
5. **Do not conflate LoRA adapter identity with the vLLM base-weight version label.** In TRL's versioned LoRA mode the model/adapter name is the source selector; base-model token spans do not certify the adapter checkpoint. A separate adapter checksum/lease contract would be required.
6. **Do not call labels weight attestation**. For actual weight verification, a server-confirmed digest or equivalent authenticated report is needed; vLLM version strings are caller-controlled and not ordered/validated by the engine.
7. **Do not claim clean KV-state semantics** with `clear_cache=false`: even spans that are correctly labeled by the sampling-time weight may condition on pre-update cached activations. A stricter trajectory guarantee may require `wait/abort`, or cache reset / a pinned weight snapshot, with a measured throughput cost.

## Counterexample and theorem boundary

For an output of length four, the documented legal server reply
`[{version:"v1",start:0,end:2},{version:"v2",start:2,end:4}]`
contains no single behavior-weight identity. The old scalar request label `v1` cannot encode this fact. If a method asserts homogeneous provenance solely from that scalar, the metadata inference is unsound.

**Exact coverage invariant:** with half-open spans `(s_i,e_i,v_i)`, require `s_0=0`, `e_i=s_{i+1}`, `e_i>s_i` and `e_last=n`. This suffices to partition the output token indices, not to prove actual parameter tensors match labels.

**Strict group-admission invariant (an intentionally conservative design):**
Given fully labeled tokens across a reward group, admit only if every label equals the same committed epoch `v` and `0<=V-v<=s` (learner version `V`, staleness bound `s`). This is a proposal for a homogeneous-policy objective; admitting correctly mixed-policy data with per-token log probabilities would need a different loss derivation and direct empirical study.

## Novelty and acceptance limits

This document does **not** claim invention of the token-version span mechanism: it is provided by vLLM. It identifies a **potential cross-library semantic gap** (version commit, turn token-source validation, group-level admission and cache lineage) that is not solved by a scalar request version. Full contribution needs a maintainer-approved backwards-compatible contract, source-level tests in TRL and a real server/learner experiment with mid-request weight sync. At present that integration has **not** been implemented or merged.

Success milestones: maintainer acceptance of a narrow design; native TRL tests on supported vLLM; reproducible actual mixed-version responses; old-vs-strict correctness with drop rate / compute overhead; only then new estimator or on-policy training results.

## Stronger counterexample: all sampling-version spans can be uniform while cached policy state is stale

Consider a single attention head with one previously processed context token `x=(1,0)`. Under old weights `θ0`, its cached value vector is `V_0(x)=(1,0)`. Under new weights `θ1`, a clean new forward pass would compute `V_1(x)=(4,0)`. If `pause(mode=keep,clear_cache=false)` retains the old KV while a newer forward computes the next-token query/logits, the next-token probability differs:

- Fresh θ1 from recomputed context: `softmax([4,0])[0] ≈ 0.982014`.
- θ1 sampling with old cached context: `softmax([1,0])[0] ≈ 0.731059`.

The server can label the only *newly sampled token* with a complete uniform `weight_versions=[{version:"v1",start:0,end:1}]`, yet the conditional distribution does **not** equal fresh `π_θ1(y|x)`. The vLLM token version may correctly name the weights that sampled the token, but it does not certify its **contextual KV lineage**. This is a theoretical toy counterexample based on the documented `clear_cache=false` semantics, not a reproduction of a real vLLM GPU execution.

**Revised admission contract:** Even complete uniform output-token version labels are not sufficient to claim homogeneous-policy likelihood unless cached context is guaranteed to have been rebuilt from the matching weights, a pinned snapshot is used, or the server supplies an independently verified context cache-epoch guarantee. Our standalone prototype's `group_admission` now fails closed with `reject-cache-unverified` by default; `cache_epoch_verified=True` is *only a caller assertion for design testing*, not an attestation obtainable from existing TRL/vLLM APIs.

The research trade-off is no longer just correct version labels. It is **source-consistent trajectory generation under asynchronous weight publication**, balancing `pause(wait/abort)` or cache reset (performance loss) against a verified pinned-context/lease scheme. Demonstrating a genuinely better throughput/quality trade-off and real training benefit is required for a doctoral-level algorithm claim. The exact numerical toy is in `ci/vllm_stale_kv_context_counterexample.py` and cannot establish novelty relative to the literature.


## External ingress activation: previously missing global admission fence

The initial multi-engine publisher marked itself poisoned if a weight-update or
`resume` failed, but it did **not** stop client requests to an engine that had
already resumed successfully. For an externally balanced group of independent
vLLM servers, server-by-server confirmation is **not** equivalent to atomic
client-visible publication. After a lost later resume ACK, earlier engines can
already serve the new version while other engines are unavailable or uncertain.

**Revised contract** in `ci/vllm_multi_engine_epoch_publisher.py`:
an exclusive external admission gate is closed *before the first server pause*;
all servers quiesce, clear reusable KV/prefix state, transfer and label weights,
verify labels, and acknowledge resume **while the gate is still closed**;
only then an atomic durable gate/epoch commit admits new requests. Any uncertain
stage leaves the ingress gate closed and blocks further publication until
reconciliation. A resumed worker after a lost ACK is still inaccessible to
cooperative clients while the gate is closed.

The gate model checks success across 1/2/4 servers, all 12 single-stage
two-server failures, an ambiguous failure after a completed resume, and probes
that client admission remains blocked during a staggered resume. A separate
expanded local reference model adds before/after failure injection at all
six stages across two servers (24 cases). The expanded model and 14 passing local
tests are distributed in the user's research bundle. New GitHub Actions on
this fork branch have **not produced an independent receipt**.

**Required external assumptions:** every client request goes through a gate
that cannot be bypassed; one exclusive writer holds the publishing lease;
pause(wait) really drains requests; clear-cache reliably invalidates old
prefix/KV state; gate epoch and open/closed status commit atomically and
durably; and epoch labels correctly refer to installed weights. The prototype
tests neither durable consensus nor the real vLLM engine implementation.
`weight_info` checks only the chosen version label, not the actual tensor
checksum. During any uncertain publication, safety may require sacrificing
liveness: the gate remains closed until reconciled.

This is the **next hypothesis** in a stronger system research line, not a
claim to have invented an atomic publish protocol. It must be compared against
existing async RL runtimes, including `AsyncOPD`, `VCPO`, and `M2PO`, then
implemented and measured with real cache-aware vLLM requests and GRPO/OPD
learning objectives before PhD-level originality or benefit can be asserted.
