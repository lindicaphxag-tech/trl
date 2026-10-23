# 2026-10-23 originality audit and next genuinely research-level problem

**Status:** exact finite toy proofs + local tests, not a TRL patch, native vLLM reproduction, or validated new estimator.

## Prior-art disqualification

Do **not** claim invention of policy-version prefix-cache isolation:
- [prime-rl](https://www.primeintellect.ai/rl-glossary/kv-cache) already uses group-dispatch policy-version `cache_salt` and permits in-flight KV continuity.
- [PSRL KV cache](https://psrl.readthedocs.io/en/latest/design/kv_cache.html) already has `multi_version_kv` and `clear_on_weight_update`.
- [vLLM](https://docs.vllm.ai/en/latest/usage/security/) already supports authenticated caller salt isolation; in [its actual first-block hashing code](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py), `cache_salt` enters first-block extra keys, and subsequent block hashes chain from their parent.

Our independently executed source-matched model checks that a stale salt permits reuse of old prefix state, changing the salt at each committed policy epoch isolates shared prefix blocks, and merely changing the new request's salt **does not** recompute an already-live request's private KV. An epoch salt must be unpredictable and tenant-/adapter-/model-scoped, not a public predictable integer. Safe no-flush operation is conditional on a complete old-request drain, an all-request gateway, proper non-prefix cache handling, and actual epoch confirmation.

**No measured speedup:** the first new-epoch prompt incurs cold-prefill work under BOTH full cache flush and epoch-salted isolation. Retaining old entries may increase memory pressure. Real vLLM GPU tests are indispensable.

## Next research gap: coupled group advantages with mixed behavior

Current TRL AsyncGRPO stores sampled `old_log_probs` and builds group-normalized advantages. A mixed-context rollout may lack a single policy checkpoint, but its actual history-conditional sampling probability `mu_t(a|h,cache)` can still be logged. Under common transition laws, coverage and **exact** logged behavior probabilities, ordinary trajectory importance sampling `W = product_t pi(a_t|h_t) / mu_t(a_t|h_t,cache_t)` changes the measure for bounded trajectory functions. This fact is **existing off-policy evaluation mathematics**.

A group-coupled advantage changes the target: for two conditionally independent trajectories with a group function `F(tau1,tau2)`, the target expectation is represented by **joint** weight `W1 * W2`, not generally by only `W1` applied to sample1's group-normalized contribution. In an exactly enumerated 2-step binary example:
- target group functional expectation **0.2176**;
- raw mixed behavior **0.18624375**;
- naive row-wise correction **0.5117** (biased because sample2's reward remains behavior-distributed);
- full joint correction **0.2176**;
- joint ratio clipping at 1 **0.17260625** (bias/variance tradeoff).

The example uses `F = reward1 * sign(reward1 - reward2)` and is *not* the full TRL GRPO gradient. Correlated group sampling requires the true **joint** behavior likelihood, and old log probabilities must match the actual action sampling distribution including masks/temperature and old KV context. Importance weights can have extreme variance, so exact group correction may be impractical at long horizons.

**Research-worthy target, not yet achieved:** a provenance-aware **group-coupled** objective that remains calibrated under asynchronous mixed behavior, handles policy/KV lineage, and lowers effective variance relative to full joint importance sampling and a strict reject-all baseline. Require a derivation, a counterexample-resistant implementation in native TRL trainer tests, measured training quality per GPU-hour on held-out prompts/tasks, and comparison with V-trace/sequence-level IS/VCPO/AsyncOPD before originality or superiority is claimed.

The fully runnable reference scripts and SHA-256 manifest are supplied as a conversation download artifact, not yet incorporated into upstream TRL. It is preferable to wait for maintainers' scope decision in [TRL issue #7861](https://github.com/huggingface/trl/issues/7861) before opening a new official pull request.
