"""Tiny-attention counterexample: output weight-version labels do not identify
the behavior conditional when a cached prefix was formed under old weights.

This is a mathematical, single-head two-position construction using the
softmax attention equations. It is NOT a live vLLM integration or an
empirical claim about actual rollout frequencies. Uses Python stdlib only.
"""
import math
import unittest
from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    committed_weight_epoch: int
    emitted_token_epoch: int
    prefix_kv_epoch: int | None


def attention_prob(prefix_key: float, prefix_value: float) -> float:
    # One attention head: current query q=1, current k=v=0; two context
    # positions are the cached prefix and the current position.
    # Binary action A has logit equal to the attention output; B has logit 0.
    prefix_attention = math.exp(prefix_key) / (math.exp(prefix_key) + 1.0)
    logit_a = prefix_attention * prefix_value
    return 1.0 / (1.0 + math.exp(-logit_a))


def strict_homogeneous(provenance: Provenance) -> bool:
    return (provenance.prefix_kv_epoch is not None
            and provenance.committed_weight_epoch
                == provenance.emitted_token_epoch
                == provenance.prefix_kv_epoch)


class TestCacheEpochNonidentifiability(unittest.TestCase):
    def test_same_visible_version_but_different_conditional_distribution(self):
        # W0 caches (K,V)=(0,-2) for the same visible prefix token.
        # W1 would instead cache (K,V)=(2,+2).
        old_cache = (0.0, -2.0)
        new_cache = (2.0, +2.0)
        stale = attention_prob(*old_cache)
        fresh = attention_prob(*new_cache)

        # The issued request, final output-token weight label, visible
        # prompt and current model version are identical in both worlds.
        shared_observation = {
            "request_model_version": 1,
            "server_committed_version": 1,
            "emitted_token_spans": ((0, 1, "1"),),
            "prompt_tokens": (17,),
        }
        old_context_case = (shared_observation, Provenance(1, 1, 0))
        new_context_case = (shared_observation, Provenance(1, 1, 1))
        self.assertEqual(old_context_case[0], new_context_case[0])
        self.assertFalse(strict_homogeneous(old_context_case[1]))
        self.assertTrue(strict_homogeneous(new_context_case[1]))

        self.assertAlmostEqual(stale, 0.2689414213699951, places=13)
        self.assertAlmostEqual(fresh, 0.8534092045709029, places=13)
        self.assertGreater(fresh - stale, 0.58)
        self.assertGreater(fresh / stale, 3.17)
        print("TOKEN_SOURCE_COUNTEREXAMPLE",
              f"p_fresh={fresh:.12f}", f"p_old_KV={stale:.12f}",
              f"probability_ratio={fresh / stale:.9f}", flush=True)

    def test_wait_without_cache_clear_does_not_certify_a_new_request(self):
        # All old active requests have drained. A v0 prefix remains cached.
        # A new v1 request may reuse that v0 prefix even if all generated
        # tokens carry the user-assigned v1 output weight_version label.
        no_active_requests = True
        label_committed = True
        old_prefix_retained = Provenance(1, 1, 0)
        self.assertTrue(no_active_requests and label_committed)
        self.assertFalse(strict_homogeneous(old_prefix_retained))
        cleared_and_regenerated = Provenance(1, 1, 1)
        self.assertTrue(strict_homogeneous(cleared_and_regenerated))

    def test_token_label_only_importance_ratio_is_not_identified(self):
        # A simplistic label-only assumption of 'current policy generated
        # this action' yields importance ratio 1. That does not identify
        # its true conditional probability when an old prefix is reused.
        p_under_clean_current_policy = attention_prob(2.0, +2.0)
        p_under_current_weights_with_old_cache = attention_prob(0.0, -2.0)
        assumed_ratio = 1.0
        actual_ratio_for_action_a = (
            p_under_clean_current_policy / p_under_current_weights_with_old_cache
        )
        self.assertGreater(actual_ratio_for_action_a - assumed_ratio, 2.17)
        # If actual sampled-token logprobs are available and accurate,
        # legitimate off-policy corrections may still be possible. This
        # test does not claim bias in TRL's existing estimator.

    def test_unknown_cache_epoch_must_fail_closed_for_strict_admission(self):
        self.assertFalse(strict_homogeneous(Provenance(1, 1, None)))


if __name__ == "__main__":
    unittest.main()
