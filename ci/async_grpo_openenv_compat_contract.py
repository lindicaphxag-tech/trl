"""Run the real OpenEnv white-box sampling adapter with both generation ABIs.

OpenEnv's session-owning producer must retain its legacy six-field group
return; tool sampling must accept the new four-field native generator response.
No OpenEnv package, GPU, or vLLM server is required to exercise that method.
"""
import ast
import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

path=Path("trl/experimental/async_grpo/openenv_harness.py")
tree=ast.parse(path.read_text(),filename=str(path))
cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_HarnessRolloutLoop")
method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=="_sample_turn")
module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
def parse_response(tokenizer, ids, *, prefix, tools):
    return {"role":"assistant","content":"ok","tool_calls":[]}

ns={
    "time":time,
    "asyncio":asyncio,
    "_tools_to_schema":lambda items:items,
    "parse_response":parse_response,
    "_msg_to_llm_response":lambda x:SimpleNamespace(tool_calls=[]),
    "TurnRecord":lambda *args,**kw:SimpleNamespace(args=args,kw=kw),
    "ModelStepResult":lambda **kw:SimpleNamespace(**kw)
}
exec(compile(module,str(path),"exec"),ns)

class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):
        return [1,2]

class Fake:
    _sample_turn=ns["_sample_turn"]
    tokenizer=Tokenizer()
    chat_template=None
    chat_template_kwargs={}
    def __init__(self,returned,loop):
        self.value=returned
        self._loop=loop
    async def _generate_one_turn(self, prompt_ids):
        assert prompt_ids==[1,2]
        return self.value

loop=asyncio.new_event_loop()
started=threading.Event()
def worker():
    asyncio.set_event_loop(loop)
    loop.call_soon(started.set)
    loop.run_forever()
t=threading.Thread(target=worker,daemon=True)
t.start()
assert started.wait(3)
try:
    for returned in ([7], [-.4], None), ([7],[-.4],None,9):
        fake=Fake(returned,loop)
        turns,timings=[],[]
        result=fake._sample_turn(turns,timings,[{"role":"user","content":"hi"}],[],{})
        assert result.completion_ids==[7]
        assert result.logprobs==[-.4]
        assert len(turns)==1 and turns[0].args[:2]==([1,2],[7])
        assert len(timings)==1
        print(f"PASS actual OpenEnv white-box turn with generation tuple length {len(returned)}")
finally:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    loop.close()

# _HarnessRolloutLoop intentionally remains a six-field producer; the
# scheduler must not reclassify its unverified provenance as a known interval.
empty=next(n.value for n in cl.body if isinstance(n,ast.Assign)
    and any(isinstance(x,ast.Name) and x.id=="_EMPTY_ROLLOUT" for x in n.targets))
assert isinstance(empty,ast.Tuple) and len(empty.elts)==6
print("PASS OpenEnv keeps the six-field producer interface (unknown source version)")
