# Async weight completion race: fenced snapshot-swap contract

**2026-10-23 — research-only reference model, not an observed vLLM incident or novel ownership of fencing tokens.**

## New gap beyond epoch fencing

The earlier fork model treated `install(weights)` as one atomic operation. That assumption is **not warranted** for an asynchronous GPU, NCCL, or IPC transfer: the request may pass an initial writer-term check and then complete after another publisher has acquired a newer term. If it writes into the active serving parameters after the newer publisher commits, a version label and even a digest checked *before* this late completion no longer certify the served policy.

The [deterministic regression](../ci/async_weight_swap_contract.py) separates **stage** (an isolated, non-serving snapshot) from **finish** (publication). It includes:

- Negative control: A stages weights under term 1; B installs and publishes term 2; A's unchecked late completion overwrites one engine's live weights while its label remains v2.
- Positive control: the late completion is fenced at the *mutation linearization point* and rejected.
- Additional checks: no live changes during staging, no early label publication, a partial multi-engine install cannot reopen ingress.
- An exhaustive bounded enumeration of all **12,870** order-preserving interleavings of two eight-step publisher programs, checking a safety invariant after every step.

The local standard-library unittest run: 6 tests passed, 12,870 enumerated interleavings with no invariant violation under the model's conditions. This is a finite model check, not an exhaustive distributed-system proof.

## Strong assumptions and actual implementation feasibility

1. Each slow transfer populates **private staging state**; it cannot write into active GPU serving tensors before completion.
2. The fence comparison and active snapshot/pointer swap are **atomic under the same synchronization domain**. Rechecking only when a request is received is insufficient.
3. The writer term and ingress barrier are durable and linearizable; every mutation path enforces that term, including retries and out-of-band callers.
4. All inference engines participate; existing requests and cached prefixes are safely quiesced or invalidated. Tensor sharding, graph captures, optimizer hooks and LoRA/draft weights remain unmodeled.
5. Verifying a hash once does not block subsequent writes. The invariant needs continuous mutation fencing (or exclusive guarded serving access), not merely a digest.
6. The abstract byte-string snapshot is **not** an implemented zero-copy, CUDA, vLLM, or Ray integration. Real backends may mutate parameters in place and cannot adopt this protocol without architectural changes.

## Literature/API comparison and adoption strategy

Existing vLLM exposes start/update/finish-weight-update, pause/resume and optional version labels; the official docs warn that a failed update can leave live parameters mid-reload even with a healthy process. vLLM's pause `keep` explicitly allows mixed-version token spans; strict consistency is a new *optional policy*, not a reason to change its default. See:

- https://docs.vllm.ai/en/latest/training/weight_transfer/
- https://docs.vllm.ai/en/latest/training/async_rl/
- https://docs.vllm.ai/en/latest/training/weight_checker/

**Novelty claim restricted:** not isolated staging, fencing, snapshot swaps, hashing or pause APIs themselves; the research question is whether a *cross-system, token-provenance-aware generation admission contract* can be implemented with demonstrable correctness and acceptable learning/throughput cost.

For highest upstream merge probability, do **not** propose a mandatory inference-engine memory-layout redesign. First reproduce a live async stale-completion hazard (if reachable), verify all existing backend serialization guarantees, seek maintainer guidance on correct ownership (vLLM vs trusted external gate), and land only a minimal, opt-in regression-backed patch. If current vLLM serialization already excludes this race, record the negative finding rather than assert a bug.

## Concrete real-system compatibility blocker found during follow-up

As of 2026-10-23, upstream vLLM [issue #60744](https://github.com/vllm-project/vllm/issues/60744)
documents a genuine CPU scheduler failure: `pause_generation(mode="keep", clear_cache=True)`
raises if a resumable streaming-input trajectory is parked between turns,
because `reset_prefix_cache` does not reclaim block-holding waiting sessions.
This is directly relevant to any proposed trajectory-preserving, clean-cache
strict-provenance protocol. The failure also leaves the engine paused.

Importantly, [vLLM PR #60764](https://github.com/vllm-project/vllm/pull/60764)
is **already open to fix this issue**, with its own native scheduler
regressions and reported CPU validation. Do **not** submit a duplicate patch.
Our research should treat correct cache-clearing pause across parked
sessions as an external prerequisite that must pass the *exact* deployed
version's tests. The adjacent `wait` mode has not been separately proved
immune in our own setup, so no stronger production claim is made here.

A stronger novelty target would be a measured end-to-end TRL/vLLM
generation-source admission contract that composes already existing
native pause, weight-version and checksum APIs with explicit failure
semantics. That must be demonstrated on genuine serving processes before
it can be called an implemented systems contribution.

## Next evidence gate

- Execute genuine 2-client/2-server vLLM requests with a delayed transfer and inspect source-token `weight_versions` plus installed tensors.
- Prove whether any write is possible after `finish_weight_update` and normal `resume` under supported NCCL/IPC backends.
- Compare default async, strict pause/drain, and fenced snapshot publication at matched GPU-seconds, freshness, rollout acceptance and return.
- Maintain the separate JAX/Ray PR merge track; abstract models are not a substitute for the current maintainer's CI and review acceptance.
