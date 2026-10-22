"""Production-source CPU contract for AsyncGRPO version ranges (no live vLLM).

Exercises the edited worker's real two-turn generator, request generator,
policy-range aggregation, score-to-sample construction and trainer queue gate.
The fake components are network IO, tool execution, tokenizer and reward.
"""
import ast
import asyncio
import contextvars
import copy
import queue
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import numpy as np

producer=Path("trl/experimental/async_grpo/async_rollout_worker.py")
tree=ast.parse(producer.read_text(),filename=str(producer))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="_AsyncRolloutLoop")
wanted={"_generate_one","_generate_one_turn","_request_model_for_version",
        "_record_group_policy_versions","_score_group"}
nodes=[]
for n in cls.body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted:
        n=copy.deepcopy(n)
        n.decorator_list=[]
        nodes.append(n)
assert {x.name for x in nodes}==wanted
scope={
    "time":time, "uuid":uuid, "random":random, "np":np,
    "Any":Any, "Callable":Callable,
    "Messages":list, "TrainingSequence":list,
    "RolloutGroup":object,
    "RolloutSample":lambda **kw:SimpleNamespace(**kw),
    "logger":SimpleNamespace(info=lambda *a,**kw:None),
    "is_async_callable":lambda fn:False
}

def parse_response(tokenizer, turn_ids, *, prefix, tools):
    if turn_ids==[7]:
        return {"role":"assistant","content":"tool invocation",
                "tool_calls":[{"function":{"name":"lookup","arguments":{}}}]}
    assert turn_ids==[8],turn_ids
    return {"role":"assistant","content":"finished"}

def chain_to_sequences(turns, rollout_id, threshold):
    result=SimpleNamespace(
        input_ids=[1,2,7,8], completion_mask=[0,0,1,1],
        old_log_probs=[0.,0.,-.1,-.2], sampling_mask=[[1],[2]])
    return [result], SimpleNamespace(reason="no drift")

scope.update({
    "parse_response":parse_response,
    "_chain_to_sequences":chain_to_sequences,
    "TurnRecord":lambda *args,**kwargs:SimpleNamespace(args=args,kwargs=kwargs)
})
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
             str(producer),"exec"),scope)

class Tokenizer:
    def __init__(self, owner):
        self.owner=owner
    def apply_chat_template(self, messages, **kwargs):
        if self.owner.publish_before_first and not self.owner.requests:
            self.owner.model_version=1
        return [1,2,7] if any(m.get("role")=="tool" for m in messages) else [1,2]

class Loop:
    _generate_one=scope["_generate_one"]
    _generate_one_turn=scope["_generate_one_turn"]
    _request_model_for_version=scope["_request_model_for_version"]
    _record_group_policy_versions=staticmethod(scope["_record_group_policy_versions"])
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
    max_tool_calling_iterations=2
    _fork_threshold_tokens=0
    reward_funcs=[lambda **kwargs:[1.,3.]]
    reward_func_names=["reward"]
    _env_reward_types=[]
    _provides_rollout_reward=False
    def __init__(self, *, publish_before_first=False, initial=0, change_after_first=False,
                 lora="policy", replay=False, fail_first=False):
        self.model_version=initial
        self.publish_before_first=publish_before_first
        self.change_after_first=change_after_first
        self.lora_name=lora
        self.sampling_replay=replay
        self.fail_first=fail_first
        self.tokenizer=Tokenizer(self)
        self.requests=[]
        self.attempts=[]
    async def _post(self,path,payload,timeout):
        requested_model=payload["model"]
        self.attempts.append(requested_model)
        if self.fail_first and len(self.attempts)==1:
            self.model_version+=1
            raise TimeoutError("synthetic timeout")
        self.requests.append(requested_model)
        if len(self.requests)==1:
            if self.change_after_first:
                self.model_version+=1
            token=7
        else: token=8
        if self.sampling_replay:
            assert path=="/inference/v1/generate"
            return {"choices":[{"token_ids":[token],"logprobs":{"content":[{"logprob":-.1}]},
                                 "sampling_mask":[[token]]}]}
        assert path=="/v1/completions"
        return {"choices":[{"token_ids":[token],"logprobs":{"token_logprobs":[-.1]}}]}
    async def _retry(self,request,**kwargs):
        try:return await request()
        except TimeoutError:return await request()
    async def _execute_tool_calls(self,calls,tool_dict):
        return ([{"role":"tool","name":"lookup","content":"ok"}],1,0)
    def _push_rollout_metrics(self,**kwargs):
        pass

def run_two_turns(loop,group_id):
    return asyncio.run(loop._generate_one([{"role":"user","content":"question"}],
                 {"lookup":lambda:None},[{"name":"lookup"}],group_id=group_id))

def group_instance(group_id):
    return SimpleNamespace(
        group_id=group_id, model_version=0, model_version_max=None,
        prompts=[],completions=[],completions_ids=[],
        completions_sequences=[],tool_call_counts=[],
        tool_failure_counts=[],reward_kwargs={},
        env_rewards=[],rollout_rewards=[],
    )

def add_generated(group, result):
    completed,ids,seq,calls,failures,reward,v_min,v_max=result
    Loop._record_group_policy_versions(group,v_min,v_max)
    group.prompts.append([{"role":"user","content":"question"}])
    group.completions.append(completed)
    group.completions_ids.append(ids)
    group.completions_sequences.append(seq)
    group.tool_call_counts.append(calls)
    group.tool_failure_counts.append(failures)
    group.rollout_rewards.append(reward)
    return v_min,v_max

def check_case(name, kwargs, expected_requests, expected_range):
    loop=Loop(**kwargs)
    group=group_instance(12)
    result=run_two_turns(loop,12)
    actual_range=add_generated(group,result)
    assert loop.requests==expected_requests,(name,loop.requests,expected_requests)
    assert actual_range==expected_range,(name,actual_range,expected_range)
    assert (group.model_version,group.model_version_max)==expected_range
    assert result[1]==[7,8] and result[3]==1
    print(f"PASS {name}: requests={loop.requests} range={actual_range}")
    return loop,group

mixed,group0=check_case(
    "cross-turn publish",{"change_after_first":True},
    ["policy-v0","policy-v1"],(0,1))
fresh,group1=check_case(
    "publish before first HTTP",{"publish_before_first":True},
    ["policy-v1","policy-v1"],(1,1))
plain,group2=check_case(
    "unchanged source",{},["policy-v0","policy-v0"],(0,0))
replay,group3=check_case(
    "sampling replay",{"replay":True,"change_after_first":True},
    ["policy-v0","policy-v1"],(0,1))
dense,group4=check_case(
    "dense server routing",{"lora":None,"change_after_first":True},
    ["base","base"],(0,1))

retry=Loop(initial=9,fail_first=True,lora="policy")
turn=asyncio.run(retry._generate_one_turn([1,2]))
assert turn[3]==9 and retry.attempts==["policy-v9","policy-v9"],retry.attempts
assert retry.model_version==10
print("PASS HTTP retry does not switch its requested policy after publication")

two=group_instance(99)
add_generated(two,run_two_turns(Loop(initial=2,change_after_first=True),99))
add_generated(two,run_two_turns(Loop(initial=5),99))
assert (two.model_version,two.model_version_max)==(2,5)
try:
    Loop._record_group_policy_versions(two,7,6)
    raise AssertionError("inverted range accepted")
except ValueError: pass
assert (two.model_version,two.model_version_max)==(2,5)
print("PASS concurrent-group interval reduction (2,3) union (5,5) -> (2,5)")

# Use _score_group to verify trainer-facing RolloutSample uses the newly recorded min.
two.reward_kwargs={}
scorer=Loop()
samples=asyncio.run(scorer._score_group(two))
assert len(samples)==2
assert all(s.model_version==2 for s in samples)
assert all(s.metrics["policy/version_span"]==3. for s in samples)
print("PASS actual scoring pathway propagates min version and mixed-policy diagnostic")

# Extract the unmodified trainer's real staleness consumer.
trainer=Path("trl/experimental/async_grpo/async_grpo_trainer.py")
trainer_ast=ast.parse(trainer.read_text(),filename=str(trainer))
dataset=next(x for x in trainer_ast.body if isinstance(x,ast.ClassDef)
             and x.name=="RolloutQueueDataset")
iter_fn=next(x for x in dataset.body if isinstance(x,ast.FunctionDef) and x.name=="__iter__")
consumer_scope={"time":time,"queue":queue,"contextvars":contextvars,
                "logger":SimpleNamespace(info=lambda *a,**kw:None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[iter_fn],type_ignores=[])),
             str(trainer),"exec"),consumer_scope)

class Consumer:
    __iter__=consumer_scope["__iter__"]
    def __init__(self,model_version):
        self.queue=queue.Queue()
        def mk(v,id):
            return SimpleNamespace(
                model_version=v, group_id=id,
                input_ids=[1,2,7,8],completion_mask=[0,0,1,1],
                old_log_probs=[0.,0.,-.1,-.1],
                sampling_mask=None,advantage=1.,metrics={},enqueued_at=None)
        self.queue.put(mk(model_version,12))
        self.queue.put(mk(1,99))
        self.model_version_fn=lambda:5
        self.check_health_fn=lambda _:None
        self.stale_after_s=2
        self.poll_interval_s=.01
        self.max_staleness=4
        self.metrics=defaultdict(list)
        self.wait_s=0.
        self._trace_log_interval=8
        self._last_traced_version=-1

bad=Consumer(0)
assert next(iter(bad))["group_id"]==99
assert bad.metrics["sample/dropped_stale_total"]==[1.]
good=Consumer(group1.model_version)
assert next(iter(good))["group_id"]==12
assert good.metrics["sample/dropped_stale_total"]==[]
print("PASS real training queue: initial-v0 tag drops; observed-min-v1 tag retains")

for versions in ([0,1],[1,1],[0,2],[2,3,2],[1,4,5],[5]):
    for now in range(9):
        for threshold in range(6):
            assert all(now-v<=threshold for v in versions)==(
                now-min(versions)<=threshold)
    assert (min(versions)==max(versions))==all(v==versions[0] for v in versions)
print("PASS conservative freshness and one-policy interval proofs")
