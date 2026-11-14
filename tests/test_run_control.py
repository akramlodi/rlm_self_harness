import rlm.core.run_control as run_control_module
from rlm.core.run_control import RunControl, RunPhase
from rlm.environments.local_repl import LocalREPL
from rlm.utils.prompts import build_user_prompt


def make_control(*, finalization_iterations: int = 3) -> RunControl:
    return RunControl.create(
        max_iterations=10,
        finalization_iterations=finalization_iterations,
        max_timeout=None,
        finalization_timeout=0.0,
    )


def test_subcalls_remain_unlimited_during_work_phase() -> None:
    control = make_control()

    for _ in range(100):
        assert control.subcall_refusal() is None


def test_reserved_turns_switch_to_finalization() -> None:
    control = make_control()

    control.update_iteration(6)
    assert control.phase is RunPhase.WORK
    control.update_iteration(7)

    assert control.phase is RunPhase.FINALIZE
    assert control.subcall_refusal() is not None


def test_time_reserve_switches_to_finalization(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(run_control_module.time, "monotonic", lambda: now)
    control = RunControl.create(
        max_iterations=10,
        finalization_iterations=0,
        max_timeout=60.0,
        finalization_timeout=15.0,
    )
    assert control.phase is RunPhase.WORK

    now = 145.0

    assert control.subcall_refusal() is not None
    assert control.phase is RunPhase.FINALIZE


def test_local_repl_refuses_expansion_but_can_submit() -> None:
    control = make_control()
    control.update_iteration(7)
    repl = LocalREPL(run_control=control)

    refusal = repl.execute_code(
        "results = [llm_query('x'), rlm_query('x'), "
        "llm_query_batched(['x'])[0], rlm_query_batched(['x'])[0]]"
    )
    assert refusal.stderr == ""
    assert all(value.startswith("SUBCALL_REFUSED:") for value in repl.locals["results"])

    submitted = repl.execute_code("submit('best available answer')")
    assert submitted.stderr == ""
    assert submitted.final_answer == "best available answer"


def test_finalization_prompt_requires_existing_state_and_atomic_submit() -> None:
    prompt = build_user_prompt(iteration=7, max_iterations=10, run_phase=RunPhase.FINALIZE)

    assert "FINALIZATION PHASE" in prompt["content"]
    assert "do not start any new" in prompt["content"]
    assert "submit(value)" in prompt["content"]
