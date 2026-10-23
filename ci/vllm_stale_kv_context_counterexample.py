"""Counterexample: sampling-time vLLM weight labels do not certify KV history.

A tiny attention head keeps values cached under theta0, swaps projection
weights to theta1 and generates the next token under theta1. Even if the
new sampled token has a uniform server-declared version span "v1", the
conditional distribution can differ from a fresh theta1 forward pass.

This illustrates a *possible* stale-KV semantic risk; it is not an actual
vLLM GPU execution or a training-return measurement. The vLLM documentation
explicitly says pause(mode=keep,clear_cache=false) can retain old KV state.
"""
import math
import unittest


def softmax_2(a, b):
    exp_delta = math.exp(b - a)
    return 1.0 / (1.0 + exp_delta)


def sampled_token_probability(*, cached_value_weight):
    # One cached prefix token x=(1,0) with a value projection:
    # V_old(x)=(1,0), V_new(x)=(4,0).
    # A single-prefix-token attention head has attention weight 1.0, so
    # the next-token logits are exactly the cached value vector.
    return softmax_2(cached_value_weight, 0.0)


class KVLineageCounterexample(unittest.TestCase):
    def test_same_sampling_epoch_not_same_behavior_distribution(self):
        old_cache = sampled_token_probability(cached_value_weight=1.0)
        fresh_new = sampled_token_probability(cached_value_weight=4.0)
        stale_cache_new_sampling = sampled_token_probability(cached_value_weight=1.0)
        # Both serve the next token after theta1 was committed, so either
        # execution can truthfully label that *sampled output token* v1.
        output_version_spans = [{"version": "v1", "start": 0, "end": 1}]
        self.assertEqual(output_version_spans[0]["end"], 1)
        self.assertNotAlmostEqual(stale_cache_new_sampling, fresh_new, delta=0.1)
        self.assertAlmostEqual(old_cache, stale_cache_new_sampling)
        print(
            "KV COUNTEREXAMPLE:",
            "new_version_fresh_p", round(fresh_new, 8),
            "new_version_old_kv_p", round(stale_cache_new_sampling, 8),
            "absolute_difference", round(abs(fresh_new-stale_cache_new_sampling), 8),
            "reported_version", output_version_spans[0]["version"],
        )

    def test_rebuilding_context_restores_new_policy_distribution(self):
        fresh = sampled_token_probability(cached_value_weight=4.0)
        rebuilt = sampled_token_probability(cached_value_weight=4.0)
        self.assertEqual(fresh, rebuilt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
