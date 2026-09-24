import json

import pytest

from rlm.core.llm_observation import ObservationPersistenceError
from shrlm.optimization.llm_observation_store import observation_recorder, verify_observations


def test_files_survive_failed_call_and_resolve_after_move(tmp_path):
    trace = tmp_path / "runs" / "example.json"
    recorder = observation_recorder(trace, {"run_id": "example"})
    with pytest.raises(ValueError), recorder.call("root_turn") as call:
        call.receive({"availability": "returned", "reasoning": {"reasoning": "full text"}})
        raise ValueError("execution failed")
    payload = {"llm_observations": recorder.references}
    verify_observations(payload, trace.parent)
    moved = tmp_path / "moved"
    trace.parent.rename(moved)
    verify_observations(payload, moved)
    response = next(r for r in recorder.references if r.get("attempt_id"))
    assert json.loads((moved / response["path"]).read_text())["reasoning"] == {
        "reasoning": "full text"
    }
    (moved / response["path"]).write_text("corrupt")
    with pytest.raises(ObservationPersistenceError, match="sha256"):
        verify_observations(payload, moved)


def test_reference_cannot_escape_owner(tmp_path):
    with pytest.raises(ObservationPersistenceError, match="outside"):
        verify_observations(
            {"llm_observations": [{"path": "../secret", "sha256": "abc"}]}, tmp_path
        )


def test_legacy_payload_needs_no_files(tmp_path):
    verify_observations({"response": "old", "metadata": {"iterations": []}}, tmp_path)


def test_started_call_is_durable_without_finished_marker(tmp_path):
    recorder = observation_recorder(tmp_path / "run.json", {})
    with recorder.call("root_turn"):
        files = list(tmp_path.rglob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["event"] == "started"


def test_driver_preserves_response_before_execution_failure(tmp_path, monkeypatch):
    from rlm.core.rlm import RLM
    from shrlm.optimization.driver import build_round_rlm, execute_run
    from tests.clients.test_openai_transport import good_response, make_client
    from tests.optimization.test_driver import make_round_config

    response = good_response()
    response.choices[0].message.reasoning_content = "saved before failed execution"
    client = make_client(response)
    monkeypatch.setattr("rlm.core.rlm.get_client", lambda *args: client)
    original = RLM._completion_turn

    def fail_after_response(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("post-response failure")

    monkeypatch.setattr(RLM, "_completion_turn", fail_after_response)
    harnessed = build_round_rlm(make_round_config(tmp_path))
    trace = tmp_path / "runs" / "test.json"
    outcome = execute_run(
        harnessed, {"id": "test", "prompt": "task"}, model_name="model", trace_path=trace
    )
    assert outcome.completion.execution_failure.cause == "runtime_error"
    verify_observations(outcome.completion.to_dict(), trace.parent)
    assert len([r for r in outcome.completion.llm_observations if r.get("attempt_id")]) == 1


def test_driver_write_failure_is_not_a_task_failure_or_paid_retry(tmp_path, monkeypatch):
    import shrlm.optimization.driver as driver
    from rlm.core.llm_observation import ObservationRecorder
    from tests.clients.test_openai_transport import good_response, make_client
    from tests.optimization.test_driver import make_round_config

    client = make_client(good_response())
    monkeypatch.setattr("rlm.core.rlm.get_client", lambda *args: client)

    def fail_response(record):
        if record["event"] == "response":
            raise OSError("disk full")
        return {"observation": record}

    monkeypatch.setattr(
        driver, "observation_recorder", lambda *args: ObservationRecorder(sink=fail_response)
    )
    harnessed = driver.build_round_rlm(make_round_config(tmp_path))
    with pytest.raises(driver.RoundPersistenceError, match="disk full"):
        driver.execute_run(
            harnessed, {"prompt": "task"}, model_name="model", trace_path=tmp_path / "trace.json"
        )
    assert client.client.chat.completions.create.call_count == 1
    assert harnessed.rlm.last_completion_usage.total_calls == 1
