import json

import pytest

from rlm.core.llm_observation import ObservationPersistenceError, observation_session
from shrlm.optimization.attribution import AttributionCache, AttributorConfig, LLMAttributor
from shrlm.optimization.digest import build_digest
from shrlm.optimization.llm_observation_store import observation_recorder, read_observation
from shrlm.optimization.proposal import ProposalCache
from shrlm.optimization.walker import walk
from tests.clients.test_openai_transport import good_response, make_client
from tests.optimization.fixtures import as_completion, make_verdict, shallow_run
from tests.optimization.test_attribution import UNGROUNDED, canned_attribution


@pytest.mark.parametrize("cache_type", [AttributionCache, ProposalCache])
def test_cache_observations_are_portable_and_legacy_keys_stay_valid(tmp_path, cache_type):
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"key": "old", "response": "old answer"}) + "\n")
    cache = cache_type(path=str(path))
    cache.put("new", "answer", observations=[{"reasoning": {"reasoning": "saved"}}])
    loaded = cache_type(path=str(path))
    assert loaded.get("old") == "old answer"
    assert loaded.get_observations("old") is None
    assert loaded.get_observations("new")[0]["reasoning"]["reasoning"] == "saved"


def test_attribution_repair_and_cache_replay_keep_reasoning_out_of_prompts(tmp_path):
    root, stats = walk(as_completion(shallow_run()))
    verdict = make_verdict()
    digest = build_digest("example", "task", root, stats, verdict)
    bad, good = good_response(), good_response()
    bad.choices[0].message.content = "invalid attribution"
    good.choices[0].message.content = canned_attribution()
    for index, response in enumerate((bad, good)):
        response.choices[0].message.reasoning_content = f"META_PRIVATE_{index}"
    client = make_client()
    client.client.chat.completions.create.side_effect = [bad, good]
    cache = AttributionCache(path=str(tmp_path / "cache.jsonl"))
    attributor = LLMAttributor(client, AttributorConfig(max_attempts=2), cache)
    for directory in ("first", "replay"):
        artifact = tmp_path / directory / "attributions.jsonl"
        recorder = observation_recorder(
            artifact, {"stage": "attribution"}, namespace="llm_calls/attribution"
        )
        with observation_session(recorder):
            result = attributor.attribute(digest, root, verdict, UNGROUNDED)
        assert [a.accepted for a in result.attempts] == [False, True]
        for index, attempt in enumerate(result.attempts):
            assert attempt.llm_observations is not None
            responses = [
                read_observation(r, artifact.parent)
                for r in attempt.llm_observations
                if r.get("attempt_id")
            ]
            assert responses[0]["reasoning"]["reasoning_content"] == f"META_PRIVATE_{index}"
            assert responses[0]["cached"] == (directory == "replay")
    assert client.client.chat.completions.create.call_count == 2
    assert all(
        "META_PRIVATE" not in str(c.kwargs)
        for c in client.client.chat.completions.create.call_args_list
    )


def test_proposal_repair_replay_is_self_contained_and_checks_integrity(tmp_path):
    from shrlm.optimization.proposal import ProposerConfig, propose_round
    from shrlm.rlm_harness import H0
    from tests.optimization.test_proposal import BUNDLE, synthetic_evidence

    bad, good = good_response(), good_response()
    bad.choices[0].message.content = "not JSON"
    good.choices[
        0
    ].message.content = '{"format": "proposal-selection/v2", "selections": [], "candidates": []}'
    for index, response in enumerate((bad, good)):
        response.choices[0].message.reasoning = f"PROPOSAL_PRIVATE_{index}"
    client = make_client()
    client.client.chat.completions.create.side_effect = [bad, good]
    cache = ProposalCache(path=str(tmp_path / "cache.jsonl"))
    config = ProposerConfig(max_attempts=2)
    evidence = synthetic_evidence(BUNDLE["patterns"])
    for directory in ("first", "replay"):
        work = tmp_path / directory / "work"
        result = propose_round(
            BUNDLE,
            H0,
            client,
            tmp_path / directory / "proposals",
            config=config,
            cache=cache,
            workdir=work,
            evidence=evidence,
        )
        for index, attempt in enumerate(result.attempts):
            assert attempt.llm_observations is not None
            responses = [
                read_observation(r, work) for r in attempt.llm_observations if r.get("attempt_id")
            ]
            assert responses[0]["reasoning"]["reasoning"] == f"PROPOSAL_PRIVATE_{index}"
            assert responses[0]["cached"] == (directory == "replay")
    assert client.client.chat.completions.create.call_count == 2
    assert all(
        "PROPOSAL_PRIVATE" not in str(c.kwargs)
        for c in client.client.chat.completions.create.call_args_list
    )
    assert result.attempts[0].llm_observations is not None
    ref = next(r for r in result.attempts[0].llm_observations if r.get("attempt_id"))
    (work / ref["path"]).unlink()
    with pytest.raises(ObservationPersistenceError):
        propose_round(
            BUNDLE,
            H0,
            client,
            tmp_path / "replay" / "proposals",
            config=config,
            cache=cache,
            workdir=work,
            evidence=evidence,
        )
    assert client.client.chat.completions.create.call_count == 2


def test_budget_exhaustion_response_is_cached_without_another_call(tmp_path):
    from rlm.core.llm_observation import capture_response
    from rlm.utils.exceptions import TokenLimitExceededError
    from shrlm.optimization.proposal import ProposalBudgetExhausted, propose_round
    from shrlm.rlm_harness import H0
    from tests.mock_lm import MockLM
    from tests.optimization.test_proposal import BUNDLE, synthetic_evidence

    response = good_response()
    response.choices[0].message.content = ""
    response.choices[0].message.reasoning_content = "BUDGET_PRIVATE"

    def exhausted(prompt):
        capture_response(response, provider="fixture", model="fixture", account_usage=lambda: None)
        raise TokenLimitExceededError(10, 10)

    client = MockLM(response_fn=exhausted)
    cache = ProposalCache(path=str(tmp_path / "cache.jsonl"))
    for directory in ("first", "replay"):
        work = tmp_path / directory / "work"
        with pytest.raises(ProposalBudgetExhausted) as caught:
            propose_round(
                BUNDLE,
                H0,
                client,
                tmp_path / directory / "proposals",
                cache=cache,
                workdir=work,
                evidence=synthetic_evidence(BUNDLE["patterns"]),
            )
        attempt = caught.value.attempts[-1]
        assert attempt.llm_observations is not None
        assert (work / "proposal_failure.json").exists()
        response = next(
            read_observation(r, work) for r in attempt.llm_observations if r.get("attempt_id")
        )
        assert response["reasoning"]["reasoning_content"] == "BUDGET_PRIVATE"
        assert response["cached"] == (directory == "replay")
    assert client._call_count == 1


def test_capture_failure_does_not_enter_meta_transport_retries(tmp_path, monkeypatch):
    from rlm.core.llm_observation import ObservationRecorder
    from shrlm.optimization.proposal import ProposerConfig, _completion_with_retry

    def fail_response(record):
        if record["event"] == "response":
            raise OSError("disk full")
        return {"observation": record}

    client = make_client(good_response())
    recorder = ObservationRecorder(sink=fail_response)
    with pytest.raises(ObservationPersistenceError), recorder.call("proposal"):
        _completion_with_retry(client, [{"role": "user", "content": "task"}], ProposerConfig(), [])
    assert client.client.chat.completions.create.call_count == 1
    assert client.get_usage_summary().total_calls == 1
