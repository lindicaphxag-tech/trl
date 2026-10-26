# Full issue #7861 evidence archive (before triage condensation)

The following is a verbatim copy of the GitHub issue body as read on 2026-10-26. It is retained to avoid losing provenance, reproduction links, or scope limitations when editing the issue for reviewability.

---

## Summary: two reproducible policy-source ambiguities in AsyncGRPO tool trajectories

The current async GRPO worker creates `RolloutGroup(model_version=self.model_version)` when a group is scheduled, but each subsequent `_generate_one_turn` independently reads `self._request_model` when constructing its HTTP payload. Multi-turn tool rollouts can therefore use several different policy versions while the group records only one version.

**Deterministic reproduction from the actual production methods:** [Fork test](https://github.com/lindicaphxag-tech/trl/blob/research/async-grpo-multiturn-policy-contract-20261021/ci/async_grpo_multiturn_policy_contract.py) and [passing Actions](https://github.com/lindicaphxag-tech/trl/actions/runs/39615695942). The same fixture successfully runs against the **unchanged official TRL source pinned to `96a50e271db494a12336cfc60eb17429247c489f`**, not just the fork snapshot. It runs the real `_generate_one`, `_generate_one_turn` and `RolloutQueueDataset.__iter__` methods with deterministic fake HTTP, tokenizer, tools and sample objects; no GPU/vLLM is needed.

### Counterexample A: different policy versions inside one trajectory

Group created at v0 -> first tool turn requests `policy-v0` -> v1 published -> next turn requests `policy-v1`; nevertheless `group.model_version == 0`. A single source-version label cannot assert that this trajectory was sampled from one policy.

### Counterexample B: all turns from v1, but initial group tag causes an incorrect stale drop

Group created at v0 -> v1 published before first HTTP dispatch -> both turns request `policy-v1`, while group still records v0. At current trainer version V=5 and `max_staleness=4`, real queue code drops because `5 - 0 > 4`; all actual requested versions satisfy `5 - 1 <= 4`. The test executes the real queue iterator and confirms the result.

### Proposed semantic contract for discussion

Let `v_t` be each actual generation-turn version. Two different questions require different provenance:
- Conservatively fresh: `V - min(v_t) <= max_staleness` (assuming version-indexed request identity tracks actual serving behavior).
- Uniform one-policy trajectory: `min(v_t) == max(v_t)`.

The minimal interval `[min(v_t), max(v_t)]` answers both of these version questions but does **not** replace per-turn old logprobs or attest installed server weights. [Research note with proof and limits](https://github.com/lindicaphxag-tech/trl/blob/research/async-grpo-multiturn-policy-contract-20261021/research/async_grpo_policy_lineage_contract_20261021.md).

There are two defensible remedies, depending on the intended training semantics:
1. **Allow mixed versions** and propagate per-turn requested/verified versions, retaining the interval for staleness and diagnostic checks; confirm the current group-relative loss semantics remain intended.
2. **Require homogeneous groups** by pinning a versioned snapshot across turns and HTTP retries, with an explicit cancel/resample protocol if the adapter is evicted. Dense in-place transfers need a separate weight identity guarantee.

This is distinct from #7176: cancelling rollouts only after they exceed `max_staleness` does not prevent a v0→v1 change within the permitted window. It is related to but not duplicative of the AsyncDistillation single-generation issue #7857.

### Correctness limits and contribution process

We **do not claim the current GRPO gradient estimator is necessarily biased**: per-token old logprobs are already recorded, and may support valid off-policy correction. The reproduced issue concerns version provenance and one confirmed false stale-drop case. No unverified performance improvement or actual-vLLM installed-weight guarantee is claimed.

Before proposing a production patch, could a maintainer confirm whether mixed-version tool trajectories are intentional, and which of the above contracts is preferred? In accordance with TRL's first-time contributor policy, please **assign this issue to @lindicaphxag-tech** if the change is welcome; I will then submit a focused template-compliant PR. AI tools assisted with the regression harness, and validation claims are restricted to actual observed CI logs.


### Tested implementation candidate and source-level regression (2026-10-22)

I implemented the **allow-mixed-version, truthfully-record-source** option in a fork-only two-file production branch: [minimal source comparison](https://github.com/lindicaphxag-tech/trl/compare/7408f20d5715ca4c264404a6b88332fd8d7f0a7d...research/async-grpo-policy-interval-fix-20261022). It propagates each completed turn's exact *requested* policy version with the response, aggregates `min/max` across tool turns and `num_generations`, uses the observed minimum for the existing staleness gate, and exposes `policy/version_span` without changing rewards, advantages or token logprob computations.

[Successful fork CI](https://github.com/lindicaphxag-tech/trl/actions/runs/39784223997) independently executes actual production methods including `_generate_one`, `_generate_one_turn`, group aggregation, `_score_group`, and the unchanged trainer's `RolloutQueueDataset.__iter__`. It covers cross-turn publish, all-new-version generation after old group scheduling, HTTP retries, normal and sampling-replay endpoints, LoRA and dense paths, multiple group completions, and training-admission correctness. Ruff and Python compilation pass. [Methodology, proof and limits](https://github.com/lindicaphxag-tech/trl/blob/ci/async-grpo-policy-interval-20261022/research/ASYNC_GRPO_POLICY_LINEAGE_20261022.md).

This is **fork research, not a formal upstream PR**. In particular a request's model version is not an attested vLLM installed-weight hash, and this does not establish an unbiased mixed-policy GRPO objective or benchmark performance gain. I'd value maintainer input on whether the intended contract is to **allow mixed-version trajectories with accurate conservative freshness** or to **enforce version-homogeneous tool episodes**; that decision affects the correct upstream change. Please assign to @lindicaphxag-tech if the proposed scope is appropriate, per the first-contributor policy.


### Latest-main two-file candidate and group-admission invariant (2026-10-22)

To avoid relying on an older fork snapshot, I also prepared a **two-file-only** candidate directly from official TRL main `ea8cb1b57bd29da6deb1fc2ee9f1f8ff77c7038e`: [reviewable source comparison](https://github.com/lindicaphxag-tech/trl/compare/ea8cb1b57bd29da6deb1fc2ee9f1f8ff77c7038e...fix/async-grpo-request-lineage-20261022). It carries actual *requested* LoRA versions through each generation turn and retry, collects their min/max interval, and intentionally applies the **minimum across the entire GRPO reward group** to every scored sample. This last detail matters: applying per-conversation stale filtering to a reward-normalized group can admit only some completions, changing its training semantics. Group-wide conservative admission maintains the original complete-group decision. In dense in-place serving, a version label does not attest the installed weight snapshot, so the original conservative group-start version is retained.

[**Completed successful GitHub Actions red/green run**](https://github.com/lindicaphxag-tech/trl/actions/runs/39789076517) ([fork CI PR #17](https://github.com/lindicaphxag-tech/trl/pull/17)) compiles and Ruff-checks the production worker and native test file, then executes actual source-extracted `_generate_one_turn`, `_generate_one`, `_score_group`, and `RolloutQueueDataset.__iter__` against both this patch and the **unchanged official producer/consumer sources from the same pinned commit**. It reproduces original v1-v1 samples falsely discarded under a group-start v0 tag, confirms corrected group-wide admission, and exercises v0→v1 mixed turns, partial provenance rejection, dense fallback and retry provenance. These are deterministic CPU tests, not live-vLLM model-weight attestations or empirical GRPO performance claims.

This candidate and the earlier [broader research branch](https://github.com/lindicaphxag-tech/trl/tree/research/async-grpo-policy-interval-fix-20261022) implement the same *proposed semantics*; I am not asking for duplicate upstream PRs. The official design decision (allow accurately recorded mixed-source tool trajectories versus enforce a homogeneous serving snapshot) and initial-contributor issue assignment are still pending.


### 2026-10-23 source audit: vLLM now exposes per-token weight-version spans

An important higher-level limitation of the scalar/request-version proposal above is confirmed against the current public source:
- TRL's `AsyncGRPO` client pauses vLLM with `mode=keep,clear_cache=false`. vLLM explicitly documents that a request continuing over a weight swap can produce output tokens sampled under **two different weight versions**, and may preserve earlier KV-cache activations.
- Current TRL `_generate_one_turn` requests `return_token_ids` but not `return_weight_versions`. The current TRL `finish_weight_update` call supplies no `weight_version` label. Thus request-version provenance alone cannot certify a **homogeneous token source**, and asking vLLM for spans without registering the version first may yield only its default/uncommitted label.
- [vLLM's current API](https://docs.vllm.ai/en/latest/training/async_rl/) supports `finish_weight_update(weight_version=...)` / `update_weight_version` and token-indexed `weight_versions` response spans. These are **user-defined labels, not verified tensor checksums**. LoRA adapter identity is separate from the base-weight version; spans must not be treated as LoRA attestations.

This shifts the research question from **"Which model version did TRL request?"** to **"Which version sampled each generated token, was that label committed, and can a complete GRPO reward group safely use that data under the chosen objective?"**. Conservative rejection of mixed-token or mixed-group sources is one optional strict policy; a per-token off-policy correction would require a separate learning-theory and benchmark study, and is *not* implemented here.

I have placed the compatibility caveats, exact span-partition contract, and standalone stdlib unit prototype in [fork research PR #18](https://github.com/lindicaphxag-tech/trl/pull/18). **It is research only**: no official TRL production patch, no live vLLM token-version experiment, no claim of actual training regression or original invention of vLLM spans; this new fork workflow has not produced a GitHub CI receipt as of this update.

Could maintainers advise whether a **backward-compatible opt-in dense-mode provenance contract** (committed server epoch + returned spans + explicit mixed-version handling) is in scope, or whether this belongs in a separate upstream design discussion? I will avoid opening a new official PR without assignment or an agreed scope.


### Additional bounded protocol and source-cache counterexample — 2026-10-23

A follow-up correctness boundary is now explicit: **even fully uniform `weight_versions` output spans and a quiescent `pause(mode=wait)` interval do not by themselves prove fresh-policy conditional likelihood if old prefix KV is reused**. There are two distinct ways stale context can appear:

1. An active request crosses an in-place update via `pause(mode=keep,clear_cache=false)`; newly sampled tokens may truthfully be labeled with the new `weight_version` while cached K/V was produced under the old epoch.
2. **No active request at all**, but an old *prefix cache* survives the update: a new request reuses it, with all newly sampled token spans labeled using the new weights. Merely waiting until old requests finish is insufficient without cache invalidation (or correctly epoch-keyed prefix reuse).

A [bounded 2-request exhaustive reference model](https://github.com/lindicaphxag-tech/trl/blob/research/vllm-token-source-admission-20261023/ci/epoch_fenced_rollout_modelcheck.py) checks quiescence, in-flight token epochs, weight+label commits, and shared prefix cache. A local independent run explored 640 safe-protocol states / 820 transitions at depth 9 with no safety counterexample and produced both unsafe-keep and unsafe-wait-without-prefix-clear witnesses. This is **bounded model testing**, not proof that a real vLLM engine implements the abstract assumptions.

A [separate multi-engine fail-closed publisher prototype](https://github.com/lindicaphxag-tech/trl/blob/research/vllm-token-source-admission-20261023/ci/vllm_multi_engine_epoch_publisher.py) expresses the stronger `WAIT+cache invalidation -> all engines pause -> transfer -> finish with epoch label -> confirm label -> resume` barrier, with an exclusive writer assumption and fail-stop behavior if any server is in an uncertain partial-transfer state. A separate local Python reference run tested 12 injected two-server failure points plus successful 1/2/4-engine updates. This is **not yet wired into TRL** and `/weight_info` is label confirmation, **not** tensor checksum attestation.

This does not invent vLLM's `pause`, version spans, or weight update APIs. The potential contribution is a **cross-system coherent generation-context contract**: (sampling version, cache/prefix epoch, adapter identity if LoRA, group admission), with a safe publication boundary across externally balanced vLLM instances. A future strict TRL implementation must be opt-in, backwards compatible, and checked against real vLLM weight-update/long-lived generation, actual reset semantics, performance, and mixed-policy objectives. Its stronger integrity version may use the existing vLLM weight-checker/checksum endpoint.

**Maintainer question refined:** Is an opt-in strict provenance mode based on `pause(mode=wait,clear_cache=true)` with server epoch publication and exact token-span verification acceptable in TRL, or should context-cache lineage be enforced at the inference/server layer? The current `keep` default should not be silently changed because latency/throughput semantics matter.
