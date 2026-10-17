"""Dependency-free execution of actual TRL async generation methods from source AST.

Fork-side smoke check only; does not replace TRL integration tests.
"""
import ast
import asyncio
import random
from pathlib import Path

path = Path("trl/experimental/async_distillation/async_rollout_worker.py")
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_AsyncRolloutLoop")
names = {"_generate_one_turn", "_request_model_for_version"}
methods = [n for n in worker.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
assert {n.name for n in methods} == names, "Missing production methods"
cls = ast.ClassDef(name="PolicyProbe", bases=[], keywords=[], body=methods, decorator_list=[])
source = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
scope = {"random": random}
exec(compile(source, str(path), "exec"), scope)
probe = scope["PolicyProbe"]()
probe.lora_name = "trl-policy"
probe.model_name = "base"
probe.model_version = 9
probe.vllm_server_url = "http://student:8000"
probe.max_tokens = 8
probe.temperature = probe.top_p = probe.repetition_penalty = 1.0
probe.top_k = 0
probe.request_timeout = 30
probe.min_p = None
requests = []


async def fake_post(base_url, endpoint, payload, timeout):
    requests.append((base_url, endpoint, payload["model"]))
    probe.model_version = 10  # concurrent publication during the HTTP request
    return {"choices": [{"token_ids": [42]}]}


async def fake_retry(fn, **kwargs):
    return await fn()


probe._post, probe._retry = fake_post, fake_retry
ids, version = asyncio.run(probe._generate_one_turn([1, 2]))
assert (ids, version) == ([42], 9)
assert requests == [("http://student:8000", "/v1/completions", "trl-policy-v9")]
assert probe.model_version == 10

# Ensure the caller also uses the version returned by generation.
caller = next(n for n in worker.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_generate_and_score_one")
assignments = [
    n for n in ast.walk(caller)
    if isinstance(n, ast.Assign) and len(n.targets) == 1
    and isinstance(n.targets[0], ast.Tuple)
    and [v.id for v in n.targets[0].elts if isinstance(v, ast.Name)] == ["completion_ids", "model_version"]
    and isinstance(n.value, ast.Await)
    and isinstance(n.value.value, ast.Call)
    and isinstance(n.value.value.func, ast.Attribute)
    and n.value.value.func.attr == "_generate_one_turn"
]
assert len(assignments) == 1, "Caller must carry the same sampled version into RolloutSample"
print("PASS: source-executed policy v9 request carries v9 after concurrent publication v10.")
print("PASS: rollout caller captures the returned policy identity.")
