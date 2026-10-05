"""Offline checks that observation storage cannot become optimization evidence."""

import asyncio
import json

from rlm.core.llm_observation import observation_session
from shrlm.optimization.driver import build_round_rlm, execute_run, persist_run, verify_trace
from shrlm.optimization.llm_observation_store import observation_recorder, read_observation
from shrlm.runner import RLM
from tests.clients.test_openai_transport import good_response, make_client
from tests.optimization.test_driver import make_round_config


def test_subject_requests_answers_usage_and_digests_do_not_change(tmp_path, monkeypatch):
    from shrlm.optimization.digest import build_digest
    from shrlm.optimization.walker import walk

    monkeypatch.setattr("time.perf_counter", lambda: 100.0)
    outcomes, requests, digests = [], [], []
    instance = {"id": "generic", "prompt": "add the numbers", "gold": "5"}
    for enabled in (False, True):
        response = good_response()
        response.choices[
            0
        ].message.content = '```repl\nanswer["content"] = "4"\nanswer["ready"] = True\n```'
        response.choices[0].message.reasoning_content = "SUBJECT_PRIVATE_SENTINEL"
        client = make_client(response)
        monkeypatch.setitem(
            RLM.__init__.__globals__, "get_client", lambda *args, client=client: client
        )
        config = make_round_config(tmp_path / str(enabled), instances=[instance], max_budget=None)
        harnessed = build_round_rlm(config)
        outcome = execute_run(
            harnessed,
            instance,
            model_name="model",
            verifier=config.verifier,
            trace_path=tmp_path / "runs" / "subject.json" if enabled else None,
        )
        outcomes.append(outcome)
        requests.append(client.client.chat.completions.create.call_args.kwargs)
        assert outcome.verdict is not None
        root, stats = walk(outcome.completion)
        digests.append(
            build_digest("generic", "add the numbers", root, stats, outcome.verdict).text
        )
    assert requests[0] == requests[1]
    assert outcomes[0].completion.response == outcomes[1].completion.response == "4"
    assert (
        outcomes[0].completion.usage_summary.to_dict()
        == outcomes[1].completion.usage_summary.to_dict()
    )
    assert outcomes[0].verdict == outcomes[1].verdict
    assert digests[0] == digests[1]
    assert "SUBJECT_PRIVATE_SENTINEL" not in digests[1]
    trace_dir = tmp_path / "runs"
    entry = persist_run(
        tmp_path, "subject", "generic", 1, outcomes[1].completion, outcomes[1].verdict
    )
    before = (trace_dir / "subject.json").read_bytes()
    verify_trace(tmp_path, entry)
    assert (trace_dir / "subject.json").read_bytes() == before


def test_lambda_sync_and_async_calls_use_the_same_durable_contract(tmp_path):
    from shrlm.baselines.lambda_runner import BudgetGuardClient

    response = good_response()
    response.choices[0].message.reasoning_content = "LAMBDA_PRIVATE"
    delegate = make_client(response)

    async def create(**kwargs):
        return response

    delegate.async_client.chat.completions.create = create
    artifact = tmp_path / "run.json"
    recorder = observation_recorder(artifact, {"method": "lambda"})
    with observation_session(recorder):
        client = BudgetGuardClient(delegate, None)
    # Calls may execute in a socket thread after the factory's context exits.
    assert client.completion("task") == "ok"
    assert asyncio.run(client.acompletion("task")) == "ok"
    records = [
        read_observation(r, artifact.parent) for r in recorder.references if r.get("attempt_id")
    ]
    assert len(records) == 2
    assert len({r["call_id"] for r in records}) == 2
    assert all(r["reasoning"]["reasoning_content"] == "LAMBDA_PRIVATE" for r in records)
    assert delegate.get_usage_summary().total_calls == 2


def test_legacy_trace_bytes_and_hash_are_not_rewritten(tmp_path):
    import hashlib

    trace = tmp_path / "runs" / "legacy.json"
    trace.parent.mkdir()
    payload = '{ "response": "old", "metadata": {"iterations": []} }\n'
    trace.write_text(payload)
    entry = {
        "run_id": "legacy",
        "trace_path": "runs/legacy.json",
        "trace_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    verify_trace(tmp_path, entry)
    assert trace.read_bytes() == payload.encode()
    assert "llm_observations" not in json.loads(trace.read_text())
