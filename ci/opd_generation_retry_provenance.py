"""Source-extracted OPD HTTP retry identity contract, CPU-only."""
import ast
import asyncio
import random
from pathlib import Path

path=Path("trl/experimental/async_distillation/async_rollout_worker.py")
tree=ast.parse(path.read_text(), filename=str(path))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_AsyncRolloutLoop")
names={"_generate_one_turn", "_request_model_for_version"}
nodes=[n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
assert {n.name for n in nodes}==names
scope={"random":random}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),scope)

class Loop:
    lora_name="policy"
    model_name="base"
    model_version=9
    max_tokens=4
    temperature=1.
    top_p=1.
    top_k=0
    min_p=None
    repetition_penalty=1.
    vllm_server_url="http://vllm"
    request_timeout=2
    _generate_one_turn=scope["_generate_one_turn"]
    _request_model_for_version=scope["_request_model_for_version"]
    def __init__(self):
        self.models=[]
    async def _post(self,url,path,payload,timeout):
        self.models.append(payload["model"])
        if len(self.models)==1:
            self.model_version=10
            raise TimeoutError("simulated timeout")
        return {"choices":[{"token_ids":[7]}]}
    async def _retry(self,thunk,**kwargs):
        try:
            return await thunk()
        except TimeoutError:
            return await thunk()

loop=Loop()
ids,version=asyncio.run(loop._generate_one_turn([1,2]))
assert ids==[7] and version==9,(ids,version)
assert loop.models==["policy-v9","policy-v9"],loop.models
assert loop.model_version==10
print("PASS in-flight policy publication does not reroute generation retries")
