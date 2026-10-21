"""Source-driven CPU regression for TRL async teacher wire contracts.
Extracts the *actual* checked-out production coroutine via AST. No vLLM needed.
"""
import ast
import asyncio
import math
from pathlib import Path

path = Path("trl/experimental/async_distillation/async_rollout_worker.py")
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parse_teacher_logprobs_at_position")
loop = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_AsyncRolloutLoop")
score = next(n for n in loop.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_score_with_teacher")
module = ast.fix_missing_locations(ast.Module(body=[parser, score], type_ignores=[]))
ns = {"math": math}
exec(compile(module, str(path), "exec"), ns)
score = ns["_score_with_teacher"]

class FakeLoop:
    teacher_top_k = 2
    teacher_temperature = 1.0
    request_timeout = 4

    def __init__(self, rows):
        self.rows = rows

    async def _post(self, server, path, payload, timeout):
        assert payload["prompt"] == [1, 2, 7, 8]
        return {"choices": [{"prompt_logprobs": self.rows}]}

    async def _retry(self, thunk, **kwargs):
        return await thunk()

def position(*entries):
    return {str(tid): {"logprob": score, "rank": rank} for tid, score, rank in entries}

rows = [None, position((2, -.1, 1)), position((9, -1., 2), (7, -.5, 1)), position((8, -.2, 1))]
for delta in [-1, 0, 1]:
    case = rows[:-1] if delta < 0 else rows + [position((5, -.1, 1))] if delta > 0 else rows
    try:
        ids, logps = asyncio.run(score(FakeLoop(case), [1, 2], [7, 8], "http://teacher", "teacher"))
        assert delta == 0, f"invalid-length response accepted (delta={delta})"
        assert ids == [[7, 9], [8]] and logps == [[-.5, -1.], [-.2]]
        print("PASS legitimate teacher-forced response (including unordered top-k)")
    except ValueError as exc:
        assert delta != 0 and "length mismatch" in str(exc), str(exc)
        print(f"PASS malformed response rejected (delta={delta})")
print("PASS: three real-production-source teacher response contracts")
