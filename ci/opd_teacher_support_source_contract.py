"""CPU contract checks for TRL's production OPD teacher-support accounting.

The tested functions are extracted from the checked-out production source via AST;
the script does not reimplement the support predicate.
"""
import ast
from pathlib import Path

import torch

SOURCE = Path("trl/experimental/async_distillation/async_distillation_trainer.py")
source = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(SOURCE))
wanted = {"_narrow_top1_actual_support", "_has_teacher_signal_mask"}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
assert {node.name for node in functions} == wanted, "production support helpers not found"
module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
namespace = {"torch": torch}
exec(compile(module, str(SOURCE), "exec"), namespace)

support = namespace["_has_teacher_signal_mask"]
input_ids = torch.tensor([[0, 11, 12, 13, 14]], dtype=torch.long)
candidate_ids = torch.full((1, 5, 4), -1, dtype=torch.long)
candidate_logps = torch.full((1, 5, 4), float("-inf"), dtype=torch.float32)

# Position 0 is a prompt. Position 1 has the realized token and top-1.
candidate_ids[0, 1] = torch.tensor([11, 20, -1, -1])
candidate_logps[0, 1] = torch.tensor([-0.1, -1.0, float("-inf"), float("-inf")])
# Position 2 has teacher top-1 support but does not contain the realized token.
candidate_ids[0, 2] = torch.tensor([9, 8, -1, -1])
candidate_logps[0, 2] = torch.tensor([-0.2, -1.2, float("-inf"), float("-inf")])
# Position 3 has no teacher candidates.
# Position 4 has the realized token in the candidate list, but not at top-1.
candidate_ids[0, 4] = torch.tensor([7, 14, -1, -1])
candidate_logps[0, 4] = torch.tensor([-0.3, -0.7, float("-inf"), float("-inf")])

completion_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)
expected = {
    0.0: torch.tensor([[False, True, True, False, True]]),
    0.5: torch.tensor([[False, True, True, False, True]]),
    1.0: torch.tensor([[False, True, False, False, True]]),
}
for beta, expected_mask in expected.items():
    actual_mask = support(input_ids, candidate_ids, candidate_logps, beta)
    assert torch.equal(actual_mask, expected_mask), (
        f"beta={beta}: expected {expected_mask.tolist()}, got {actual_mask.tolist()}"
    )
    effective_count = int((completion_mask & actual_mask).sum())
    expected_count = 2 if beta == 1.0 else 3
    assert effective_count == expected_count, (beta, effective_count, expected_count)
    print(f"PASS beta={beta}: support={actual_mask.tolist()} effective_completion_tokens={effective_count}")

# Guard the production wiring: the collator denominator, per-sample metrics, and loss
# mask must all consume the same divergence-specific support predicate.
contracts = [
    "has_teacher_signal = _has_teacher_signal_mask(input_ids, teacher_topk_ids, teacher_topk_logprobs, self.beta)",
    "effective_completion_mask = completion_mask.bool() & has_teacher_signal",
    "n_trained_tokens = int(effective_completion_mask.sum())",
    "trained.append(int((completion_mask & has_signal).sum()))",
    "token_mask_1d = (completion_mask.bool() & has_teacher_signal)[0]",
]
for contract in contracts:
    assert contract in source, f"production accounting contract missing: {contract}"
print("PASS production source wiring: collator denominator, metrics, and loss share teacher-support semantics")
print("PASS: five deterministic OPD teacher-support contracts")
