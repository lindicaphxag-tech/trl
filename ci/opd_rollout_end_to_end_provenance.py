"""End-to-end-in-worker OPD provenance red/green regression using real source AST.

Run against patched file (default) and unmodified base file (--baseline).
The HTTP call and teacher are deterministic fakes; the actual
_generate_and_score_one and _generate_one_turn production methods execute.
"""
import ast
import asyncio
import random
import sys
import time
import queue
import contextvars
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np

baseline = "--baseline" in sys.argv
source_path = next((Path(x) for x in sys.argv[1:] if not x.startswith("--")),
                   Path("trl/experimental/async_distillation/async_rollout_worker.py"))
tree = ast.parse(source_path.read_text(), filename=str(source_path))
clazz = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_AsyncRolloutLoop")
wanted = {"_generate_one_turn", "_generate_and_score_one", "_request_model",
          "_request_model_for_version"}
methods = [n for n in clazz.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name in wanted]
names = {n.name for n in methods}
assert {"_generate_one_turn", "_generate_and_score_one"} <= names
assert ("_request_model" in names) or ("_request_model_for_version" in names)
scope = {"np": np, "random": random, "time": time, "Any": Any,
         "RolloutSample": lambda **kwargs: SimpleNamespace(**kwargs)}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             str(source_path), "exec"), scope)

class Tokenizer:
    eos_token_id = 2
    def __init__(self, owner):
        self.owner = owner
    def apply_chat_template(self, *args, **kwargs):
        # Publication while a queued task is being prepared (before HTTP dispatch).
        assert self.owner.model_version == 0
        self.owner.model_version = 9
        return [1, 2]
    def decode(self, ids, skip_special_tokens=False):
        return "completion"

class Loop:
    lora_name = "policy"
    model_name = "base"
    model_version = 0
    max_tokens = 4
    temperature = 1.
    top_p = 1.
    top_k = 0
    repetition_penalty = 1.
    min_p = None
    vllm_server_url = "http://vllm"
    request_timeout = 2
    chat_template_kwargs = {}
    teacher_top_k = 2
    teacher_model_names = {"default": "teacher"}
    _generate_one_turn = scope["_generate_one_turn"]
    _generate_and_score_one = scope["_generate_and_score_one"]
    def __init__(self, adapter):
        self.lora_name = adapter
        self.model_version = 0
        self.tokenizer = Tokenizer(self)
        self.seen = []
    async def _retry(self, thunk, **kw):
        return await thunk()
    async def _post(self, url, endpoint, payload, timeout):
        assert endpoint == "/v1/completions"
        assert payload["model"] == ("policy-v9" if self.lora_name else "base"), payload
        self.seen.append(("dispatch", payload["model"]))
        self.model_version = 10
        await asyncio.sleep(0)
        return {"choices": [{"token_ids": [7]}]}
    def _resolve_teacher_server_url(self, row):
        return "default", "http://teacher"
    async def _score_with_teacher(self, prompt_ids, completion_ids, url, model):
        self.seen.append(("teacher", tuple(completion_ids), self.model_version))
        assert completion_ids == [7] and model == "teacher"
        return [[7]], [[-0.2]]
    def _push_rollout_metrics(self, **kw):
        pass

if "_request_model" in scope:
    Loop._request_model = scope["_request_model"]
if "_request_model_for_version" in scope:
    Loop._request_model_for_version = scope["_request_model_for_version"]

# Exercise the REAL trainer queue consumer on the worker's returned sample.
trainer_file=Path("trl/experimental/async_distillation/async_distillation_trainer.py")
trainer_ast=ast.parse(trainer_file.read_text(),filename=str(trainer_file))
queue_cls=next(n for n in trainer_ast.body if isinstance(n,ast.ClassDef) and n.name=="RolloutQueueDataset")
real_iter=next(n for n in queue_cls.body if isinstance(n,ast.FunctionDef) and n.name=="__iter__")
consumer_scope={"time":time, "queue":queue, "contextvars":contextvars,
                "logger":SimpleNamespace(info=lambda *args,**kwargs:None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[real_iter],type_ignores=[])),
             str(trainer_file),"exec"),consumer_scope)
class Consumer:
    __iter__=consumer_scope["__iter__"]
    def __init__(self,sample):
        self.queue=queue.Queue()
        sample.enqueued_at=None
        self.queue.put(sample)
        # A truly fresh fallback avoids an infinite wait when baseline drops sample #12.
        fallback=SimpleNamespace(**vars(sample))
        fallback.prompt_id=99
        fallback.model_version=9
        self.queue.put(fallback)
        self.model_version_fn=lambda:10
        self.check_health_fn=lambda timeout:None
        self.poll_interval_s=.01
        self.stale_after_s=5
        self.max_staleness=4
        self.metrics=defaultdict(list)
        self.wait_s=0.
        self._trace_log_interval=8
        self._last_traced_version=-1

for adapter in ("policy", None):
    loop = Loop(adapter)
    sample = asyncio.run(loop._generate_and_score_one(
        12, {"prompt": [{"role": "user", "content": "question"}]}))
    assert loop.seen == [("dispatch", "policy-v9" if adapter else "base"),
                         ("teacher", (7,), 10)]
    assert sample.input_ids.tolist() == [1, 2, 7]
    assert sample.teacher_topk_ids[-1, 0] == 7
    assert sample.teacher_topk_logprobs[-1, 0] == np.float32(-0.2)
    assert loop.model_version == 10
    consumer=Consumer(sample)
    delivered=next(iter(consumer))
    if baseline:
        assert delivered["prompt_id"]==99, delivered["prompt_id"]
        assert consumer.metrics["sample/dropped_stale_total"]==[1.0]
        print("EXPECTED BASELINE TRAINING LOSS:",adapter,
              "v9-generated prompt #12 dropped as mislabeled v0")
    else:
        assert delivered["prompt_id"]==12, delivered["prompt_id"]
        assert consumer.metrics["sample/dropped_stale_total"]==[]
        print("PASS TRAINING RETAINS:",adapter,"v9 prompt #12 at v10")
    if baseline:
        assert sample.model_version == 0, sample.model_version
        print("EXPECTED BASELINE PROVENANCE FLAW:", adapter, "request v9, sample v0")
    else:
        assert sample.model_version == 9, sample.model_version
        print("PASS:", adapter, "request v9, scored sample v9, later publish v10")
print("PASS: source-level end-to-end version provenance contrast")
