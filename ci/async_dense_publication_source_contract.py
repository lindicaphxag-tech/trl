"""Run real merged-weight sync methods from both TRL async trainers, without GPU/vLLM.

Source-AST execution is narrower than an integration test. It detects ordering
regressions directly against checked-out production methods.
"""
import ast
import time
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    "AsyncGRPOTrainer": Path("trl/experimental/async_grpo/async_grpo_trainer.py"),
    "AsyncDistillationTrainer": Path("trl/experimental/async_distillation/async_distillation_trainer.py"),
}

for name, path in SOURCES.items():
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    trainer_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    method = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sync_weight_merged")
    isolated = ast.fix_missing_locations(
        ast.Module(body=[ast.ClassDef(name="SyncProbe", bases=[], keywords=[], body=[method], decorator_list=[])],
                   type_ignores=[])
    )
    events = []
    namespace = {
        "time": time,
        "logger": SimpleNamespace(info=lambda message: None),
        "is_peft_model": lambda model: False,
    }
    exec(compile(isolated, str(path), "exec"), namespace)
    probe = namespace["SyncProbe"]()
    probe.model = object()
    probe.model_version = 7
    probe._streaming_iter = lambda: iter(())
    probe.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model,
        is_main_process=True,
        wait_for_everyone=lambda: events.append("barrier"),
    )
    probe.weight_transfer = SimpleNamespace(
        pause=lambda: events.append("pause"),
        send_weights=lambda iterator: (list(iterator), events.append("transfer")),
        resume=lambda: events.append(("resume", probe.model_version)),
    )
    probe.rollout_worker = SimpleNamespace(
        update_model_version=lambda v: events.append(("publish", v))
    )
    probe._sync_weight_merged(0.0)
    expected = ["pause", "barrier", "transfer", "barrier", ("publish", 8), ("resume", 8)]
    assert events == expected, (name, events)
    print(f"PASS {name}: publish(v8) before resume(v8) after transfer/barriers")

    # Non-main ranks join the collectives, but may not publish a policy or resume vLLM.
    events.clear()
    probe.model_version = 7
    probe.accelerator.is_main_process = False
    probe._sync_weight_merged(0.0)
    assert events == ["barrier", "barrier"], (name, events)
    assert probe.model_version == 7
    print(f"PASS {name}: non-main process cannot publish or resume")

    # Failed transfers must not publish the next policy or resume a partial update.
    events.clear()
    probe.accelerator.is_main_process = True
    def failed_send(iterator):
        events.append("partial-transfer")
        raise RuntimeError("transfer failed")
    probe.weight_transfer.send_weights = failed_send
    try:
        probe._sync_weight_merged(0.0)
    except RuntimeError as exc:
        assert str(exc) == "transfer failed"
    else:
        raise AssertionError(f"{name}: failed transfer unexpectedly succeeded")
    assert "partial-transfer" in events
    assert not any(isinstance(event, tuple) for event in events)
    assert probe.model_version == 7
    print(f"PASS {name}: failed transfer does not publish or resume")

print("PASS: Six source-executed merged-weight publication contracts")
