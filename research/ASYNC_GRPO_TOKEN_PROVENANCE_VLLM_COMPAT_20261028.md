# AsyncGRPO policy provenance: request labels versus token-attested weight spans

**Date:** 2026-10-28  
**Scope:** research-only compatibility contract for [TRL #7861](https://github.com/huggingface/trl/issues/7861). No production code is changed here.

## Why a scalar request version is not a token-level proof

TRL's current async worker creates a `RolloutGroup.model_version` when a group is scheduled, while each tool turn later builds its own generation request. With asynchronous publishing and vLLM `pause_generation(mode="keep")`, a single output request may resume after a weight update and emit tokens from more than one weight version. A scheduling label, or even a version sampled once when the HTTP payload is constructed, is **request identity**, not proof that every output token used that weight snapshot.

The current fork-only candidate snapshots `request_version` once and uses the same value to construct the LoRA model alias and annotate the completed turn. That fixes an identity race for versioned LoRA model names and can expose mixed request versions across turns. It must not be described as engine-attested token provenance.

## Cross-repository evidence

At the checked vLLM source revision:

- [Async RL docs](https://github.com/vllm-project/vllm/blob/a75277a0a1db82d5929a41b11ead4abd2ac5c9c0/docs/training/async_rl.md) describe `weight_versions` spans on final output, with half-open token ranges that identify the label which sampled each token. The same documentation explicitly describes `pause_generation(mode="keep")` retaining in-flight requests across a weight update.
- [OpenAI completion protocol schema](https://github.com/vllm-project/vllm/blob/a75277a0a1db82d5929a41b11ead4abd2ac5c9c0/vllm/entrypoints/openai/completion/protocol.py) declares `return_weight_versions`.
- However, [the Rust OpenAI completion validator](https://github.com/vllm-project/vllm/blob/e08e35bbe8660cfb9fd3587d777457785650e80c/rust/src/server/src/routes/openai/completions/validate.rs) explicitly rejects `return_weight_versions` as unsupported. This means the feature cannot be assumed available across every vLLM server implementation or deployment.

## Safe staged contract

1. **Request identity:** snapshot the version once per actual request and use the same snapshot for the model alias and the recorded request label. This is a valid narrow correction for versioned LoRA aliases, but it is not token attestation.
2. **Token provenance when supported:** if the selected server endpoint explicitly supports `weight_versions`, request it and validate spans: sorted, contiguous, in-range, cover every generated output token, and use labels whose mapping to trainer versions is explicit. An empty output may legitimately have an empty span list.
3. **Conservative whole-group admission:** derive the group's admission version from the oldest observed token/request source in the group, and assign the same conservative version to every sample derived from that reward-normalized group. Never independently drop rows from one GRPO group after computing group advantages.
4. **Explicit fallback:** if the backend does not expose token spans, preserve a clearly marked request-label fallback and report that it cannot certify homogeneous token provenance. Do not silently claim full correctness.
5. **Separate homogeneous-policy mode:** if a training algorithm requires every token in an episode to use one weight snapshot, enforce a server-side generation boundary (or abort/restart on version change). A scalar client label cannot enforce this property.

## Required test matrix before upstream integration

| Case | Required invariant |
|---|---|
| v0 group scheduled, first request dispatched under v1 | versioned LoRA request records v1, not scheduling v0 |
| turn 1 uses v0, turn 2 uses v1 | source interval `[v0, v1]` is preserved; no false homogeneous claim |
| one response has two `weight_versions` spans | token-level interval is derived from actual spans, not the request snapshot |
| backend rejects `return_weight_versions` | capability fallback is explicit; no request failure unless strict mode is selected |
| group contains a stale and a fresh completion | admission/rejection is atomic at the reward-group level |
| malformed, overlapping, missing, or out-of-range spans | fail closed or mark provenance unavailable; never fabricate a version |
| legacy/manual group without provenance | existing conservative behavior is preserved and labeled as unverified |

## Decision still needed from maintainers

The issue asks whether the initial upstream change should be narrowly limited to the versioned-LoRA false-stale-admission bug, or whether capability-gated token-span provenance belongs in scope. Until that is confirmed, this document remains research-only and no upstream PR is created from it.
