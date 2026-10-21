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
    def apply_chat_template(self, messages, **kwargs):
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
    tokenizer=Tokenizer()
    def __init__(self):
        self.requests=[]
    async def _retry(self, thunk, **kwargs):
        return await thunk()
    async def _post(self, endpoint, payload, timeout):
        assert endpoint=="/v1/completions"
        model=payload["model"]
        self.requests.append(model)
        if len(self.requests)==1:
            assert model=="policy-v0"
            # Simulate a completed weight sync before the next tool-driven turn.
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
