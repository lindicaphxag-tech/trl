"""Source-extracted, end-to-end AsyncGRPO provenance regression.

Executes the real checked-out generation, group scoring and trainer consumer
methods with deterministic fake transport/tool outputs. --baseline validates
the expected old behavior on pinned unchanged upstream source. Neither mode
attests installed serving weights or proves PPO/GRPO gradient correctness.
"""
import ast
import asyncio
import contextvars
import queue
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

baseline = "--baseline" in sys.argv
sources = [Path(x) for x in sys.argv[1:] if not x.startswith("--")]
worker_path = sources[0] if sources else Path("trl/experimental/async_grpo/async_rollout_worker.py")
trainer_path = sources[1] if len(sources) > 1 else Path("trl/experimental/async_grpo/async_grpo_trainer.py")
wt = ast.parse(worker_path.read_text(encoding="utf-8"),filename=str(worker_path))
cl = next(n for n in wt.body if isinstance(n,ast.ClassDef) and n.name=="_AsyncRolloutLoop")
required = {"_request_model","_generate_one","_generate_one_turn","_score_group"}
possible = required | {"_request_model_for_version"}
nodes = [n for n in cl.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in possible]
assert required <= {n.name for n in nodes}
gl = next(n for n in cl.body if isinstance(n,ast.AsyncFunctionDef) and n.name=="_generate_loop")
group_calls = [n for n in ast.walk(gl) if isinstance(n,ast.Call) and
    isinstance(n.func,ast.Name) and n.func.id=="RolloutGroup"]
assert any(k.arg=="model_version" and isinstance(k.value,ast.Attribute) and
    k.value.attr=="model_version" for n in group_calls for k in n.keywords)

def parse_response(tok, ids, *, prefix, tools):
    if ids==[7]:
        return {"role":"assistant","content":"call tool","tool_calls":[{"function":{"name":"lookup","arguments":{}}}]}
    if ids==[8]:
        return {"role":"assistant","content":"done"}
    raise AssertionError(ids)

def chain(turns, rollout_id, threshold):
    return [{"turns":turns}], {"reconciled":0}

scope = {
    "random":random,"time":time,"uuid":uuid,"np":np,"Any":Any,"Callable":Callable,
    "Messages":list,"TrainingSequence":list,
    "RolloutGroup":SimpleNamespace,
    "RolloutSample":lambda **kwargs:SimpleNamespace(**kwargs),
    "parse_response":parse_response,"_chain_to_sequences":chain,
    "TurnRecord":lambda *args,**kwargs:SimpleNamespace(args=args,kwargs=kwargs),
    "logger":SimpleNamespace(info=lambda *a,**k:None),
    "is_async_callable":lambda _:False,
}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
             str(worker_path),"exec"),scope)

class Tokenizer:
    def __init__(self,owner):self.owner=owner
    def apply_chat_template(self,messages,**kw):
        if self.owner.publish_before_first and not self.owner.requests:
            self.owner.model_version=1
        return [1,2,7,3] if any(m.get("role")=="tool" for m in messages) else [1,2]

class Worker:
    _request_model=scope["_request_model"]
    _generate_one=scope["_generate_one"]
    _generate_one_turn=scope["_generate_one_turn"]
    _score_group=scope["_score_group"]
    model_name="base"
    lora_name="policy"
    max_tokens=4
    temperature=1.
    top_p=1.
    top_k=0
    min_p=None
    repetition_penalty=1.
    request_timeout=2
    sampling_replay=False
    chat_template=None
    chat_template_kwargs={}
    max_tool_calling_iterations=3
    _fork_threshold_tokens=0
    reward_func_names=["synthetic"]
    _env_reward_types=[]
    _provides_rollout_reward=False
    def __init__(self,publish_before_first=False):
        self.model_version=0
        self.publish_before_first=publish_before_first
        self.requests=[]
        self.tokenizer=Tokenizer(self)
        self.reward_funcs=[lambda completions,**kw:[1.,3.]]
    async def _retry(self,thunk,**kw):return await thunk()
    async def _post(self,path,payload,timeout):
        assert path=="/v1/completions"
        model=payload["model"]
        self.requests.append(model)
        if len(self.requests)==1:
            assert model==("policy-v1" if self.publish_before_first else "policy-v0"), model
            self.model_version=1
            return {"choices":[{"token_ids":[7],"logprobs":{"token_logprobs":[-.1]}}]}
        assert model=="policy-v1"
        return {"choices":[{"token_ids":[8],"logprobs":{"token_logprobs":[-.2]}}]}
    async def _execute_tool_calls(self,calls,tool_dict):
        return [{"role":"tool","name":"lookup","content":"ok"}],1,0
    def _push_rollout_metrics(self,**kw):pass

if "_request_model_for_version" in scope:
    Worker._request_model_for_version=scope["_request_model_for_version"]

def generate(before):
    worker=Worker(before)
    group_start=worker.model_version
    result=asyncio.run(worker._generate_one([{"role":"user","content":"ask"}],
        {"lookup":lambda:None},[{"name":"lookup"}],group_id=42))
    assert len(result)==(6 if baseline else 7),len(result)
    completion,token_ids,rows,ncalls,failures,_=result[:6]
    assert token_ids==[7,8] and ncalls==1 and failures==0
    assert len(rows)==1 and len(rows[0]["turns"])==2
    assert worker.requests==(["policy-v1","policy-v1"] if before else
                             ["policy-v0","policy-v1"])
    bound=None if baseline else result[6]
    if not baseline:assert bound==((1,1) if before else (0,1)),bound
    return worker,group_start,completion,token_ids,rows,bound

worker,start,completion,ids,rows,bounds=generate(False)
fresh_worker,fresh_start,fresh_completion,fresh_ids,fresh_rows,fresh_bounds=generate(True)
assert start==fresh_start==0

def to_group(worker,comp,tokens,rows,bounds):
    seq=SimpleNamespace(input_ids=[1,2,7,8],completion_mask=[0,0,1,1],
        old_log_probs=[0,0,-.1,-.2],sampling_mask=[[],[],[],[]])
    g=SimpleNamespace(prompts=[[{"role":"user","content":"ask"}]]*2,
        completions=[comp,comp],completions_ids=[tokens,tokens],
        completions_sequences=[[seq],[seq]],reward_kwargs={},
        tool_call_counts=[1,1],tool_failure_counts=[0,0],
        group_id=42,model_version=0,env_rewards=[None,None],rollout_rewards=[None,None])
    if not baseline:
        g.source_version_ranges=[bounds,bounds]
    return g

async def score(worker,comp,tokens,rows,bounds):
    return await worker._score_group(to_group(worker,comp,tokens,rows,bounds))

mixed_samples=asyncio.run(score(worker,completion,ids,rows,bounds))
fresh_samples=asyncio.run(score(fresh_worker,fresh_completion,fresh_ids,fresh_rows,fresh_bounds))
assert len(mixed_samples)==len(fresh_samples)==2
if baseline:
    assert [s.model_version for s in fresh_samples]==[0,0]
    print("EXPECTED BASELINE: v1-v1 generation recorded as group version v0")
else:
    assert [s.model_version for s in mixed_samples]==[0,0]
    assert [s.metrics["rollout/request_version_span"] for s in mixed_samples]==[1.,1.]
    assert [s.model_version for s in fresh_samples]==[1,1]
    assert [s.metrics["rollout/request_version_span"] for s in fresh_samples]==[0.,0.]
    dense=Worker()
    dense.lora_name=None
    dense_samples=asyncio.run(score(dense,fresh_completion,fresh_ids,fresh_rows,fresh_bounds))
    assert [s.model_version for s in dense_samples]==[0,0]
    print("PASS mixed v0/v1 interval, fresh v1/v1 interval and conservative dense fallback")

# Actual trainer queue admission, not an invented surrogate.
tt=ast.parse(trainer_path.read_text(encoding="utf-8"),filename=str(trainer_path))
dataset=next(n for n in tt.body if isinstance(n,ast.ClassDef) and n.name=="RolloutQueueDataset")
iteration=next(n for n in dataset.body if isinstance(n,ast.FunctionDef) and n.name=="__iter__")
consumer_scope={"time":time,"queue":queue,"contextvars":contextvars,
    "logger":SimpleNamespace(info=lambda *a,**k:None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[iteration],type_ignores=[])),
             str(trainer_path),"exec"),consumer_scope)
class Consumer:
    __iter__=consumer_scope["__iter__"]
    def __init__(self,sample):
        self.queue=queue.Queue()
        self.queue.put(sample)
        fallback=SimpleNamespace(**vars(sample))
        fallback.group_id=99
        fallback.model_version=1
        self.queue.put(fallback)
        self.model_version_fn=lambda:5
        self.check_health_fn=lambda *_:None
        self.poll_interval_s=.01
        self.stale_after_s=2
        self.max_staleness=4
        self.metrics=defaultdict(list)
        self.wait_s=0.
        self._trace_log_interval=8
        self._last_traced_version=-1
        self._trace_buf=[]
        self._traces_per_log=8
        self._trace_queue=queue.Queue()

consumer=Consumer(fresh_samples[0])
delivered=next(iter(consumer))
assert delivered["group_id"]==(99 if baseline else 42),delivered
assert consumer.metrics["sample/dropped_stale_total"]==([1.] if baseline else [])
print("EXPECTED BASELINE stale-drop" if baseline else "PASS actual producer -> group scoring -> queue delivery")
print("PASS source-extracted AsyncGRPO provenance contract")
