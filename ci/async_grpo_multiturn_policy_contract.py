"""CPU source-level regression: one async GRPO tool trajectory can mix policy versions.

Executes the real _generate_one / _generate_one_turn methods from the checked-out
TRL source, with deterministic fake HTTP, tools, and parsing. No vLLM/GPU.
A counterexample is evidence of an unverified trajectory-policy contract;
it is not itself a proof of incorrect policy gradients.
"""
import ast
import asyncio
import random
import sys
import time
import queue
import contextvars
from collections import defaultdict
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

path=Path(sys.argv[1]) if len(sys.argv)>1 else Path("trl/experimental/async_grpo/async_rollout_worker.py")
tree=ast.parse(path.read_text(),filename=str(path))
clazz=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="_AsyncRolloutLoop")
wanted={"_generate_one","_generate_one_turn","_request_model"}
nodes=[x for x in clazz.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in wanted]
assert {x.name for x in nodes}==wanted

# Verify that the *real* group scheduler attaches just one version at group creation.
gen_loop=next(x for x in clazz.body if isinstance(x,ast.AsyncFunctionDef) and x.name=="_generate_loop")
group_constructors=[
    x for x in ast.walk(gen_loop)
    if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=="RolloutGroup"
]
assert group_constructors
assert any(
    isinstance(kw.value,ast.Attribute) and kw.value.attr=="model_version"
    and isinstance(kw.value.value,ast.Name) and kw.value.value.id=="self"
    for call in group_constructors for kw in call.keywords if kw.arg=="model_version"
), "Source no longer snapshots group policy version at initial group creation"

def parse_response(tokenizer, turn_ids, *, prefix, tools):
    if turn_ids==[7]:
        return {"role":"assistant","content":"invoke lookup","tool_calls":[{"function":{"name":"lookup","arguments":{}}}]}
    if turn_ids==[8]:
        return {"role":"assistant","content":"finished"}
    raise AssertionError(f"Unexpected tokens {turn_ids}")

def chain_to_sequences(turns, rollout_id, threshold):
    return ([{"turns":turns}],{"reconciliations":0})

scope={
  "time":time, "uuid":uuid, "random":random, "Any":Any, "Callable":Callable,
  "Messages":list, "TrainingSequence":list,
  "parse_response":parse_response,
  "TurnRecord":lambda *args,**kwargs:SimpleNamespace(args=args,kwargs=kwargs),
  "_chain_to_sequences":chain_to_sequences,
}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),scope)

class Tokenizer:
    def __init__(self, owner):
        self.owner=owner
    def apply_chat_template(self, messages, **kwargs):
        if self.owner.update_before_first and not self.owner.requests:
            self.owner.model_version=1
        return [1,2,7,3] if any(m.get("role")=="tool" for m in messages) else [1,2]

class Loop:
    _generate_one=scope["_generate_one"]
    _generate_one_turn=scope["_generate_one_turn"]
    _request_model=scope["_request_model"]
    model_version=0
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
    def __init__(self, update_before_first=False):
        self.model_version=0
        self.requests=[]
        self.update_before_first=update_before_first
        self.tokenizer=Tokenizer(self)
    async def _retry(self, thunk, **kwargs):
        return await thunk()
    async def _post(self, endpoint, payload, timeout):
        assert endpoint=="/v1/completions"
        model=payload["model"]
        self.requests.append(model)
        if len(self.requests)==1:
            assert model==("policy-v1" if self.update_before_first else "policy-v0")
            # Simulate a weight publication in preparation or after the first request.
            self.model_version=1
            return {"choices":[{"token_ids":[7],"logprobs":{"token_logprobs":[-0.1]}}]}
        assert model=="policy-v1"
        return {"choices":[{"token_ids":[8],"logprobs":{"token_logprobs":[-0.2]}}]}
    async def _execute_tool_calls(self, calls, tool_dict):
        assert len(calls)==1 and calls[0]["function"]["name"]=="lookup"
        return ([{"role":"tool","name":"lookup","content":"ok"}],1,0)
    def _push_rollout_metrics(self, **kwargs):
        pass

loop=Loop()
group_version=loop.model_version  # Same data source used by real RolloutGroup ctor
completion,token_ids,rows,tool_count,failures,reward=asyncio.run(loop._generate_one(
    [{"role":"user","content":"look it up"}],
    {"lookup":lambda:None},
    [{"name":"lookup"}],group_id=12))
assert loop.requests==["policy-v0","policy-v1"],loop.requests
assert group_version==0
assert token_ids==[7,8] and tool_count==1 and failures==0
assert len(rows)==1 and len(rows[0]["turns"])==2
print("REPRODUCED: group.model_version=v0 while turns requested policy-v0 then policy-v1")
print("CONTRACT GAP: a multi-turn rollout need not correspond to a single behavior-policy version")
print("BOUNDARY: logged per-token old_log_probs can still support correction; training bias is not proven")


# Scenario B: the group was created at v0, but both HTTP turns used v1.
# This is a second effect: the group-level staleness gate may drop fresh data.
loop2=Loop(update_before_first=True)
group_initial_version=loop2.model_version
completion2,tokens2,rows2,_,_,_=asyncio.run(loop2._generate_one(
    [{"role":"user","content":"look it up"}],
    {"lookup":lambda:None},[{"name":"lookup"}],group_id=13))
assert loop2.requests==["policy-v1","policy-v1"],loop2.requests
assert group_initial_version==0 and tokens2==[7,8]
print("REPRODUCED: initial group tag v0, all actual turn requests v1")

# Execute the real AsyncGRPO trainer's RolloutQueueDataset.__iter__ staleness logic
# with two source-tagged samples. No torch/transformers import is needed.
trainer_file=Path("trl/experimental/async_grpo/async_grpo_trainer.py")
trainer_ast=ast.parse(trainer_file.read_text(),filename=str(trainer_file))
queue_class=next(n for n in trainer_ast.body if isinstance(n,ast.ClassDef) and n.name=="RolloutQueueDataset")
iter_node=next(n for n in queue_class.body if isinstance(n,ast.FunctionDef) and n.name=="__iter__")
it_scope={"time":time,"queue":queue,"contextvars":contextvars,
          "logger":SimpleNamespace(info=lambda *args,**kwargs:None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[iter_node],type_ignores=[])),
             str(trainer_file),"exec"),it_scope)

def make_sample(version, group_id):
    return SimpleNamespace(model_version=version,group_id=group_id,
        input_ids=[1,2,7,8], completion_mask=[0,0,1,1],
        old_log_probs=[-.1,-.2],sampling_mask=None,advantage=1.,
        metrics={},enqueued_at=None)

class Consumer:
    __iter__=it_scope["__iter__"]
    def __init__(self, version):
        self.queue=queue.Queue()
        self.queue.put(make_sample(version,13))
        self.queue.put(make_sample(1,99))  # fresh fallback if mislabeled sample dropped
        self.model_version_fn=lambda:5
        self.check_health_fn=lambda _:None
        self.stale_after_s=2
        self.poll_interval_s=.01
        self.max_staleness=4
        self.metrics=defaultdict(list)
        self.wait_s=0.
        self._trace_log_interval=8
        self._last_traced_version=-1

current=5
actual_versions=[1,1]
assert current-min(actual_versions)<=4
bad=Consumer(group_initial_version)
bad_result=next(iter(bad))
assert bad_result["group_id"]==99, bad_result
assert bad.metrics["sample/dropped_stale_total"]==[1.]
good=Consumer(min(actual_versions))
good_result=next(iter(good))
assert good_result["group_id"]==13, good_result
assert good.metrics["sample/dropped_stale_total"]==[]
print("REPRODUCED: actual v1-v1 group rejected at v5 solely by group-start v0 tag")
print("PASS: minimum actual turn version v1 satisfies the same real staleness gate")

# Minimum actual version is sufficient for universal freshness; min/max
# jointly describe policy-version homogeneity, without a full per-turn history.
for versions in ([0,1],[1,1],[0,2],[2,3,2],[1,4,5]):
    for current in range(0,9):
        for limit in range(0,6):
            assert all(current-v<=limit for v in versions)==(current-min(versions)<=limit)
    assert (min(versions)==max(versions))==all(v==versions[0] for v in versions)
print("PASS: finite version interval freshness/coherence invariants")
