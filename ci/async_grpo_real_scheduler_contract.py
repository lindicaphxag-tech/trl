"""Real _generate_loop scheduler integration: two concurrent completions, one group.

This executes the checked-out actual production scheduler and static version
range reducer. Only unrelated dataset, reward, tool and HTTP I/O is stubbed.
"""
import ast
import asyncio
import copy
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

path=Path("trl/experimental/async_grpo/async_rollout_worker.py")
tree=ast.parse(path.read_text(),filename=str(path))
clazz=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_AsyncRolloutLoop")
names={"_generate_loop","_record_group_policy_versions"}
methods=[]
for n in clazz.body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:
        clone=copy.deepcopy(n);clone.decorator_list=[]
        methods.append(clone)
assert {n.name for n in methods}==names
ns={
    "time":time, "asyncio":asyncio, "Any":Any, "Messages":list,
    "RolloutGroup":lambda **kw:SimpleNamespace(**kw,model_version_max=None,queued_at=0),
    "logger":SimpleNamespace(info=lambda *a,**kw:None),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),
             str(path),"exec"),ns)

class Loop:
    _generate_loop=ns["_generate_loop"]
    _record_group_policy_versions=staticmethod(ns["_record_group_policy_versions"])
    model_version=0
    num_generations=2
    _multi_environment=False
    environment_factories=None
    tools=[]
    _standalone_tools=[]
    _env_reward_types=[]
    _inflight_cap=2
    _inflight=0
    _total_rollouts=0
    _total_completion_tokens=0
    def __init__(self):
        self._heartbeat_value=SimpleNamespace(value=0)
        self._rates=defaultdict(lambda:[0.,0.])
        self._groups_to_score=asyncio.Queue(maxsize=10)
        self._pushed=[]
        self.launched=0
        self.finished=0
        self.event=None
    def _repeat_iterator(self):
        return cycle([(10,{"prompt":[{"role":"user","content":"q"}]})])
    async def _generate_one(self,prompt,tool_dict,tools,group_id):
        self.launched+=1
        index=self.launched
        # Simulate exactly the two completions of one group. The real
        # iterator produces another group_id after two siblings are scheduled;
        # stop admitting new work once both siblings have been dispatched.
        if index==2:
            self.event.set()
        await asyncio.sleep(.02 if index==1 else .005)
        self.finished+=1
        # Reverse completion order tests commutative range aggregation.
        if index==1:
            v_min,v_max=2,3
        else:
            v_min,v_max=5,5
        sequence=SimpleNamespace(input_ids=[1,2,7],completion_mask=[0,0,1],
                                 old_log_probs=[0.,0.,-.1],sampling_mask=None)
        return ([{"role":"assistant","content":"answer"}],[7],[sequence],0,0,None,v_min,v_max)
    def _push_metrics(self,values):
        self._pushed.append(values)

async def main():
    loop=Loop()
    event=asyncio.Event()
    loop.event=event
    await asyncio.wait_for(loop._generate_loop(event),timeout=4)
    rows=[]
    while not loop._groups_to_score.empty():
        rows.append(loop._groups_to_score.get_nowait())
    groups=[x for x in rows if x is not None]
    assert loop.launched==loop.finished==2,(loop.launched,loop.finished)
    assert len(groups)==1,groups
    g=groups[0]
    assert (g.model_version,g.model_version_max)==(2,5),(g.model_version,g.model_version_max)
    assert len(g.completions)==len(g.completions_sequences)==2
    assert g.group_id==10
    print("PASS real async _generate_loop: two reversed-order completions aggregated to [2,5]")
    print("PASS real queued RolloutGroup has exactly two completions and actual min/max versions")

asyncio.run(main())
