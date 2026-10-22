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


## 7. Strong-prior-art exclusion (literature checked 2026-10-22)

Do not present importance-weight clipping, old-logprob correction,
ESS-dependent stabilisation, or staleness/advantage correction as originating
from this work. Relevant stronger references include:

- Espeholt et al., [IMPALA / V-trace, ICML 2018](https://proceedings.mlr.press/v80/espeholt18a.html):
  asynchronous actor-learner off-policy correction.
- Huang et al., [Stable Asynchrony / VCPO, ICML 2026](https://proceedings.mlr.press/v306/huang26ca.html):
  effective sample size, off-policy variance and minimum-variance baselines.
- Zheng et al., [M2PO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8590b946e4f0c2b1262cb5b047924653-Abstract-Conference.html):
  second-moment constraints under high rollout staleness.
- Hu et al., [COPC, arXiv 2026-10-07](https://arxiv.org/abs/2610.09597):
  nonseparable coupling between policy-ratio and advantage-estimation staleness.
- Guan et al., [Missing Old Logits in Asynchronous Agentic RL, arXiv 2026](https://arxiv.org/abs/2605.12070):
  training-inference discrepancy versus historical policy staleness.
- Lin et al., [ResT, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8730837e119f6b5d1b25f6da1dc2c28a-Abstract-Conference.html):
  tool-task token-level policy-gradient reweighting.

The new work here is a **system-specific correctness repair with
source-code evidence**, not a claim to a new RL objective.

## 8. Indistinguishability limit: request version is not weight attestation

Consider a dense model whose client request is always `model="base"`.
History A: a transfer installs weights from epoch 9 before the request is
executed. History B: the request is executed just before that transfer, with
epoch 8 weights, but the client-visible version counter is already 9.
Both may yield the same token/logprob response on a particular prompt.
Without the server reporting a *served-weight identity*, the client's request
payload, local counter and returned tokens need not distinguish A from B.
Thus no estimator based solely on client-side snapshots can universally
identify which weight epoch actually produced the sample.

A strong end-to-end design needs (a) a server-side installed-weight generation
ID or cryptographic digest, bound to the decode request and echoed in its
response; (b) a version lease that prevents eviction of an actively referenced
adapter throughout multi-turn execution (or explicit cancellation/restart);
(c) a train-side gate using the verified provenance and explicit policy for
mixed-version trajectories; and (d) fault injection for late publishes,
retries, cancellation and LoRA eviction.

This is a **proposal requiring protocol support**, not a feature in current
TRL/vLLM, and the indistinguishability observation is not claimed as a new
distributed-systems theorem.

## 9. Specific publishable experiment required before algorithm novelty claims

Freeze a trace schema with fields `group_id`, `turn_id`,
`requested_version`, `served_weight_digest`, `request_attempt`,
`leased_epoch`, `old_logprob_checksum`, and `outcome`. Test across two
distinct execution substrates (at least LoRA versioned serving and dense
in-place weight transfer), with artificial tool delays, event-order
randomisation and simulated worker crashes.

Use a pre-registered factorial design crossing staleness budget, publish rate,
network retry rate, adapter eviction window and trajectory length. Primary
endpoint: confirmed provenance mismatch / false-stale-drop per 1,000 completed
tool trajectories, with confidence intervals. Secondary: usable trajectories,
wall-time and model utility under a matched token budget. Compare unchanged
TRL, conservative request interval only, and server-attested lease protocol.
Importantly, record cases where the new machinery causes cancellation overhead
or no net learning improvement.


## 10. Exact finite counterexample: group normalisation and mixed behavior

A further subtlety is **not** fixed by accurate version tags alone. Group-relative
advantages are nonlinear functions of the *joint rollout batch*, so correcting
each rollout with its marginal old-policy ratio need not recover the expected
on-policy **group-normalised** gradient. This is a group estimator
identification problem and should not be equated to proven bias in full TRL.

The executable [exact enumerator](./exact_mixed_behavior_group_estimator.py)
has no ML dependencies. It enumerates all 16 action tuples for a four-sample
one-step Bernoulli bandit, with reward `r_i=a_i`, target-policy probability
`p=0.05`, Bernoulli-logit score `a_i-p`, and GRPO-style population-standardised
`A_i=(r_i-mean(r))/(std(r)+1e-8)`. It uses the **true** individual importance
ratios `pi(a_i)/mu_i(a_i)`, with exact (not Monte Carlo) enumeration.

- If every rollout uses the target policy `mu_i=p`, the expected group
  gradient statistic is about **0.0812252822**.
- If heterogeneous rollouts use `mu=[0.01,0.01,0.99,0.99]`, the expected
  **marginal importance-weighted** statistic is about **0.0749968841**;
  that is **7.67% lower** than the on-policy group-statistic expectation,
  despite the ratios being correct.
- Multiplying the *entire* group statistic by the correct **joint**
  likelihood ratio `prod_i pi(a_i)/mu_i(a_i)` recovers the on-policy expected
  group statistic exactly. Its finite-population correctness says nothing
  about practical variance, which can become prohibitive with large groups.
- An unstandardised leave-one-out reward baseline yields the unbiased
  **ordinary expected-reward** policy gradient `p(1-p)=0.0475`, which is
  **a different estimand** and should not be compared as though equal to the
  standardised group objective.

The mechanism is nonseparability: marginally correcting action `a_i` does not
also correct the distribution of `A_i(a_1,...,a_K)` formed from the other
actions. This is an elementary measure-change observation, not a claim that
we invented joint importance sampling or leave-one-out baselines.

**Limits:** the example is unclipped; it is *not* the full TRL GRPO/PPO loss,
does not account for multiple optimizer steps, and does not test any real
policy model. It motivates a separate comparison between homogeneous
policy-epoch grouping, heterogeneous grouping with token ratios, and
an explicitly specified joint/group-corrected objective under matched
variance and rollout budgets.


### Closed-form K=4 identity (not only a numerical experiment)

In the same binary-reward group toy model, set `epsilon=0` for a nonconstant
group and define `A_i=0` for an all-equal group. Let
`S_1=mu_1+mu_2+mu_3+mu_4` and
`S_2=sum_{i<j} mu_i mu_j`. Expanding all 16 action tuples gives the exact
expected **marginal-importance-weighted group-standardised** logit-gradient:

```
E_mu[g_marginal]
  = p(1-p)/12 * [
      (4 sqrt(3)-6) S_2
      + (9-6 sqrt(3)) S_1
      + 12 sqrt(3)
    ].
```

The formula holds for any strictly interior Bernoulli target `p` and
behavior probabilities `mu_i`. The executable finite enumerator separately
verifies this formula over heterogeneous probability tuples to 1e-12. It
shows explicitly why true **marginal** ratios do not generally reconstruct
the **joint** on-policy distribution of the other completions entering
a nonseparable group advantage. With TRL's `1e-8` stabiliser, the exact
numbers differ slightly, while the demonstrated effect persists.

This identifies a concrete source of cross-policy **group-estimator
shift**, not a proof of asymptotic estimator bias in an actual neural GRPO
training run, nor an original derivation of importance sampling.


## 11. OpenEnv compatibility: unknown provenance is not a homogeneous policy

A cross-module audit found that `_HarnessRolloutLoop` in
`openenv_harness.py` overrides the native six-field `_generate_one`
contract and consumes the old three-field `_generate_one_turn`.
The earlier eight-/four-field source fix would have broken this supported
environment-RL path despite passing isolated AsyncGRPO source tests.

The updated fork research branch preserves both six-/eight-field group
producer results and three-/four-field white-box sampling results. When
a producer supplies no request-version provenance, the scheduler keeps its
older scheduling-time version as a **conservative bound**, sets
`policy_source_complete=False`, and reports `policy/version_span=NaN`.
It never labels that unknown source as a verified single-policy trajectory.
A group whose every producer supplies request identity has
`policy_source_complete=True` and an exact request-version interval.

[GitHub Actions verification](https://github.com/lindicaphxag-tech/trl/actions/runs/39789207006)
completed successfully and exercised actual production-source methods:
the concurrent scheduler with reversed completion order, six-field legacy
and mixed producer results, OpenEnv white-box sampling with three-/four-field
return values, scoring, and the trainer's staleness consumer. It also passed
Ruff lint/format on all modified native production/test files and a finite
group-normalisation estimator check. This is CPU source-driven verification;
it is **not** a full integration test with OpenEnv/vLLM or server-side
weight-identity attestation.

## 12. Review boundary

The minimal **fork-side production change** is now cross-module rather than
a worker-only patch: it touches the native AsyncGRPO worker, OpenEnv
compatibility bridge, and original AsyncGRPO tests. Source-driven CPU scripts
remain on the separate fork validation [PR #16](https://github.com/lindicaphxag-tech/trl/pull/16).
Official [issue #7861](https://github.com/huggingface/trl/issues/7861) awaits
maintainer design confirmation/assignment; no official new PR or Merge is
claimed. Further work must verify real OpenEnv/vLLM paths before marking the
patch fully integration-ready.
