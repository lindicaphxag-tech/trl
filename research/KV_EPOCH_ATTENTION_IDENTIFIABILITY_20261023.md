# Numeric attention counterexample for strict async-OPD provenance (2026-10-23)

**Research status:** small mathematical reference, not a live vLLM/TRL production patch, not a new estimator or training-performance result. Complements [TRL issue #7861](https://github.com/huggingface/trl/issues/7861), the existing abstract cache-epoch model, and fork research PR #18.

## Why token weight-version spans do not certify a context

vLLM's [current asynchronous RL API documentation](https://docs.vllm.ai/en/latest/training/async_rl/) distinguishes the version used to sample **output tokens** from earlier context that might have been formed under older weights if an active request is preserved or prefix cache survives a weight swap. The API's `weight_version` values are user-supplied labels, not cryptographic tensor checksums.

A strict output group needs more than uniform emitted token labels:
- current server weight epoch `w`;
- current token's generating epoch `v_t`;
- the epochs of **all reused prefix/context K/V** `c_i` (or reliable evidence the cache was cleared and rebuilt).

Sufficient strict contract under exclusive-writer and complete invalidation assumptions: `w = v_t = c_i` for every token/context contribution. A token label alone omits the context epochs.

## Executable two-position single-head attention example

Take the current-token query `q=1`; its self-attention position has `K=0, V=0`. A visible prefix token has two possible cached representations, depending on which model weights formed its cache:
- old W0 prefix: `K=0,V=-2`;
- new W1 prefix: `K=2,V=+2`.

For a one-dimensional attention head,

```
alpha = exp(q*K_prefix) / (exp(q*K_prefix) + exp(q*K_self))
logit(action A) = alpha * V_prefix
logit(action B) = 0
p(A) = sigmoid(logit(A))
```

The *current committed weights*, visible prompt, requested version and output-token `weight_versions=[(0,1,'1')]` can all be **identical** in two execution histories. Only the cache provenance differs.

Then:
- fresh W1 prefix: `logit_A=1.761594156`; `p(A)=0.8534092046`;
- stale W0 prefix used after W1 swap: `logit_A=-1`; `p(A)=0.2689414214`.
- absolute probability difference **0.5844677832**; clean-current / stale-context behavior probability ratio **3.173215938**.

Thus no function of *only* the request version, committed current epoch, visible token IDs and uniform output span labels can identify the actual behavior conditional in both histories. This is an **observational indistinguishability** counterexample in a toy attention model, not a statement that vLLM in a particular production run necessarily follows the stale path.

If an algorithm naively equates `weight_version=1` with the clean-current policy for its behavior denominator, it may infer an importance ratio 1 where this example's correct ratio for action A is 3.17. **TRL already records token old logprobs** in some paths, so this does not establish bias in TRL's actual estimator if those reflect actual sampling conditionals.

A `pause(mode=wait)` barrier prevents ongoing requests from crossing the weight update, but a previously reusable W0 *prefix cache* still requires invalidation or reliable epoch partitioning for new requests. `pause(mode=keep)` has different active-request semantics and should not be silently changed.

## Evidence and merge gates

- Runnable Python-stdlib test: `ci/kv_epoch_attention_counterexample.py` (four tests; includes explicit fail-closed epoch gate).
- Existing bounded abstract protocol audit: `ci/epoch_fenced_rollout_modelcheck.py`.
- Workflow: `.github/workflows/async_opd_context_epoch_reference.yml`.

This does **not** prove server memory isolation, prefix-cache invalidation implementation details, native vLLM attribution, RL policy-gradient bias, or throughput gains. To justify a high-impact upstream production contribution, we still need a maintainer-selected strict/legacy compatibility contract, a real vLLM mid-update+prefix-reuse test, training-group admission checks, and a performance-vs-correctness comparison with strong async-RL baselines.
