# AsyncGRPO policy lineage: interval contract and falsifiable evidence

**Date:** 2026-10-22. **Status:** fork-side implementation and source-driven CPU
verification, **not** an upstream-accepted algorithm or training benchmark.
Official design discussion: huggingface/trl#7861.

## 1. Precise estimands and invariants

Each generation request in a completed rollout group `g` has a captured
requested-policy version `v[g,j,t]`, where `j` indexes independent GRPO
completions and `t` indexes turns within a tool trajectory.

Let `V` be the training model's currently published version and `s` the
allowed version gap. Define the version interval:

```
v_min(g) = min_{j,t} v[g,j,t]
v_max(g) = max_{j,t} v[g,j,t]
```

The exact *universal freshness* gate is:

```
for every (j,t): V - v[g,j,t] <= s
    iff V - v_min(g) <= s
```

The exact *single-policy-version* check is:

```
for every (j,t): v[g,j,t] = constant
    iff v_min(g) = v_max(g)
```

These are mathematical identities (not novel algorithms). Their **operational**
preconditions are stronger: model versions must be meaningfully ordered, each
request must bind to its returned version snapshot, and a requested vLLM model
name must actually designate the intended installed weights if the identities
are interpreted as *behavior* rather than *request* lineage.

An interval is a sufficient statistic for these two Boolean questions. It is
**not** sufficient to reconstruct the behavior distribution of every generated
token or to compute trajectory off-policy likelihood ratios.

## 2. Source-level counterexamples from unmodified TRL

The original `_generate_loop` tags a group with `self.model_version` when it
**schedules** the group. Subsequent `_generate_one_turn` calls read the shared
version independently when **constructing** each HTTP request:

**A. Mixed-policy version:** group created at v0, first generation requests
`policy-v0`, an update occurs while a tool executes, and the next generation
requests `policy-v1`. The group still declares v0 although the true request
lineage has range [0,1].

**B. Incorrect stale drop:** group created at v0; a new version is published
while the prompt is prepared; both HTTP requests use v1. At consumption
version V=5 with threshold s=4, the original group tag v0 wrongly fails the
staleness gate (5-0 > 4) while the actual requested minimum v1 passes (5-1=4).

The unmodified TRL production generator and trainer consumer were tested by
AST extraction using a fixed official revision. This validates the exact
relevant method bodies but uses fake tokenizer, network and tool handlers.

## 3. Proposed fork implementation

Production candidate base: upstream TRL main at `7408f20d5715ca4c264404a6b88332fd8d7f0a7d`.

Changed files: `trl/experimental/async_grpo/async_rollout_worker.py`,
`tests/experimental/test_async_grpo_trainer.py`.

- `_generate_one_turn` captures the version **once** while building its
  model-specific request payload and returns that snapshot with token ids,
  logprobs and optional top-k/top-p sampling mask. All retries reuse the same
  payload and model name.
- `_generate_one` accumulates min/max across all turns (including
  tool-mediated interleavings) and returns those bounds with its existing
  sequence, tool and reward results.
- `_generate_loop` aggregates the returned bounds across all
  `num_generations` completions in a group. It **replaces**, on first
  observation, the initial scheduling version rather than erroneously
  including it in the minimum.
- `group.model_version` becomes the **minimum observed requested version**;
  `group.model_version_max` is the maximum; `_score_group` preserves the
  original reward/advantage/logprob path, stamps the minimum into each sample
  and exposes `policy/version_span` for diagnostics.
- The existing `RolloutQueueDataset` requires **no new code**: its gate is
  already written in terms of `sample.model_version`, which now carries the
  intended conservative version lower bound.

Changing a group metadata field does not change the gradient estimator, reward
normalization or sampling procedure. It only changes the source label used by
the existing freshness filter and provides a diagnostic interval.

## 4. Verified source-extracted tests

GitHub Actions uses CPU, NumPy, Ruff, actual checked-out production coroutine
and function definitions, plus a synthetic HTTP/tool engine.

- v0->v1 transition between tool turns yields range [0,1].
- Group created at v0, both turns actually dispatched at v1 yields [1,1].
- No version change yields [0,0].
- Sampling-replay endpoint carries same version semantics as normal
  completion endpoint.
- Dense model names remain the base name; the version label is a request-time
  snapshot and **not** a server weight identity proof.
- HTTP timeout followed by publish v9->v10 does not reroute a retry away from v9.
- Two group completions with request ranges [2,3] and [5,5] aggregate to [2,5].
- Real `_score_group` emits the conservative version and mixed-policy span.
- Real `RolloutQueueDataset.__iter__` drops the original falsely tagged
  sample and retains the fixed one.
- Exhaustive finite examples check the min/max version identities.
- Actual native test sources must compile and pass Ruff check/format.

These tests show correctness **under the mocked scheduling conditions**.
No real server/version eviction or end-to-end policy training was executed.

## 5. Research boundaries that block stronger claims

1. A vLLM HTTP `model` field is not a weight hash or attested model manifest.
   For *dense in-place sync*, two different weight versions can share one name;
   the captured version alone does not prove which was executed.
2. An adapter can be unloaded during asynchronous work, especially while
   retries are pending. A safe *homogeneous-trajectory* design additionally
   needs a version lease, checked server-installed epoch, and explicit
   cancellation/resampling if the lease expires.
3. **Mixed-policy trajectory != proven biased policy gradient.** TRL records
   each token's `old_log_probs`, which is useful for PPO-style corrections,
   but those alone do not prove unbiasedness of trajectory-level reward
   gradients or group-relative baselines under nonstationary behavior.
4. Full framework tests with Transformers, Accelerate, real vLLM, and GPU
   training remain required before the patch can be represented as a
   production-validated systems innovation.
5. The first-time TRL contribution policy requires issue assignment before a
   new upstream PR; fork staging does not count as upstream acceptance or merge.

## 6. Correct next experiment (pre-register before running)

Use two low-compute environments with deterministic seeds: (i) a short
multi-turn lookup/tool task and (ii) a horizon-limited bandit tool task.
Randomize weight publication phase relative to tool latency, with independent
replicates and hidden evaluation instances. Log the *actual installed* serving
model epoch or LoRA checksum, not only HTTP model names. Compare:
- original scheduling-version staleness,
- request-derived interval freshness,
- homogeneous pinned-version trajectories,
- fresh-only version-lease filtering,
- the original off-policy token log-probability objective unchanged.

Primary endpoints: **false-stale-drop rate** against server-attested provenance,
usable trajectories per 1,000 Teacher/tool requests, and held-out task return
under matched compute/rollout budgets. Report confidence intervals and adverse
cases. Without this study, no SOTA, training-efficiency, or PhD-level algorithm
breakthrough is claimed.
