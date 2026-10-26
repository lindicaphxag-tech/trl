# AsyncGRPO: behavior-policy lineage and the minimum-version staleness contract

Research note, 2026-10-21. This is **not** a production patch nor a claim that
the existing policy-gradient estimator is biased. It isolates correctness
contracts and unknowns for asynchronous, multi-turn GRPO tool trajectories.

## Source-proven facts

Tested against an unchanged upstream TRL worker at commit
`96a50e271db494a12336cfc60eb17429247c489f`:

- `_generate_loop` creates `RolloutGroup(model_version=self.model_version)`
  before the generation coroutine necessarily constructs any HTTP request.
- Each `_generate_one_turn` creates its own vLLM payload using
  `self._request_model`, which reads the **currently published** shared version.
- `_generate_one` can invoke `_generate_one_turn` repeatedly within one tool
  trajectory, with asynchronous tool execution between those requests.
- Every scored sample ultimately receives `model_version=group.model_version`.
- `RolloutQueueDataset.__iter__` drops when
  `current_version - sample.model_version > max_staleness`.

The source-extracted, deterministic two-turn experiment is in
`ci/async_grpo_multiturn_policy_contract.py`. It executes real
`_generate_one`, `_generate_one_turn`, and queue `__iter__` method bodies,
with scripted HTTP, tools, tokenizer, rewards, and samples. Runs both against
the fork snapshot and the pinned, unchanged official upstream source.

## Counterexample 1: mixed behavior within one trajectory

At initial group construction, published version = 0. First HTTP generation
requests `policy-v0`. An optimizer update publishes version 1 while the tool
executes. Second generation requests `policy-v1`. The scored samples still
carry group metadata `model_version=0`.

This does **not** prove policy-gradient bias: the trainer also records actual
per-token generation log-probabilities, which can support certain off-policy
objectives. It **does** prove that a group has no guaranteed single behavior
policy version, so interpreting `model_version` as a uniform source is invalid.

## Counterexample 2: false stale drop without a mixed trajectory

A group is created at version 0, but version 1 is published while the prompt is
prepared, *before* the first generation HTTP request. Both turns request v1.
At training consumption, current version = 5 and allowed staleness = 4:

- Recorded initial-group version 0: `5 - 0 = 5 > 4` -> the real consumer drops it.
- Actual minimum request version 1: `5 - 1 = 4 <= 4` -> it would be accepted.

The same actual `RolloutQueueDataset.__iter__` method was executed with both
tags; the outcomes differed exactly as above. The v1 request is an **identity
claim to the vLLM endpoint**, not a cryptographic attestation of installed
weights. In-place dense-model transfers have an additional provenance limit.

## Exact algebra: interval summary for two separate questions

Let `v_t` be the (verified or, under a weaker assumption, requested) policy
version for each completed generation turn in a trajectory; `V` is the
current training version, and `s` is the allowed staleness. Define:

```
v_min = min_t v_t
v_max = max_t v_t
```

**Freshness**: all turns are no more than s versions stale exactly when

```
forall t: V - v_t <= s
    iff V - v_min <= s
```

**Single-version homogeneity**: all turns share one version exactly when
`v_min == v_max`.

Consequently, the pair `(v_min, v_max)` is sufficient to decide *both*
properties under the version-indexed model. A single minimum does not establish
homogeneity (compare `[0,0]` vs `[0,1]`); a single maximum does not establish
universal freshness (compare `[1,2]` vs `[2,2]` with V=3 and s=1). This is
an elementary sufficient-statistic result, **not a novel statistical estimator**.

This summary is **not sufficient** for reconstructing complete trajectory
behavior probabilities, identifying a server's actual installed weights,
handling request cancellation and retry outcomes, or capturing arbitrary
nonmonotone parameter changes. A per-turn provenance record (or server-signed
weight identity) is required for those stronger claims.

## Two compatible but different design decisions

1. **Allow mixed versions, report them truthfully**: attach per-turn source
   identity, retain `v_min` and `v_max` for a conservative staleness gate and
   monitoring, and continue using the actual per-token old logprobs in the loss.
   Study whether group-relative normalization and trajectory returns remain
   appropriate for mixed behavior. Avoid changing the objective silently.
2. **Require a homogeneous trajectory**: pin generation to a versioned server
   snapshot for all turns (including retries), or cancel/resample when that
   version becomes unavailable. This requires lease/eviction compatibility
   with PR #7176, and an explicit answer for dense in-place weight updates.

Neither design is implemented here. The right choice must be agreed with
maintainers and tested against actual vLLM weight publication and tool latency.

## Experimental status and originality discipline

- **Passed**: actual source-extracted two-turn counterexample, current upstream
  source pinned at `96a50e2`, consumer false-drop scenario, and finite checks
  of the min/max freshness and coherence identities.
- **Not run**: native transformer training, vLLM server LoRA eviction, multiple
  actual policy checkpoints, FSDP/AMP, reward-trajectory learning outcomes.
- **Not claimed**: SOTA improvements, statistical bias of the current GRPO
  gradient estimator, new importance-sampling theory, upstream acceptance.

The scientific opportunity is a broader *behavior-policy provenance contract*
covering asynchronously updated serving weights, trajectory source identities,
version leases and loss semantics, compared with recorded-logprob off-policy
correction. The counterexamples establish a mechanism worth investigating;
they are not, alone, a doctorate-level new algorithm.
