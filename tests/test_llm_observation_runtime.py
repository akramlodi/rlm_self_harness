import asyncio
from unittest.mock import patch

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.llm_observation import ObservationRecorder, observation_session
from rlm.core.lm_handler import LMHandler
from rlm.core.rlm import RLM
from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.logger.rlm_logger import RLMLogger
from tests.clients.test_openai_transport import good_response, make_client


def received(references):
    return [r["observation"] for r in references if r["observation"]["event"] == "response"]


def test_batched_socket_results_keep_call_local_reasoning():
    client = make_client()
    arrived = 0

    async def create(**kwargs):
        nonlocal arrived
        slot = arrived
        arrived += 1
        while arrived < 3:
            await asyncio.sleep(0)
        if slot == 1:
            raise ValueError("failed slot")
        await asyncio.sleep((2 - slot) * 0.01)
        response = good_response()
        response.choices[0].message.reasoning = f"slot-{slot}"
        response.choices[0].message.content = f"answer-{slot}"
        return response

    client.async_client.chat.completions.create = create
    recorder = ObservationRecorder()
    with LMHandler(client, observation_recorder=recorder) as handler:
        responses = send_lm_request_batched(handler.address, ["duplicate"] * 3)
    assert arrived == 3
    assert received(responses[1].llm_observations)[0]["availability"] == "no_response"
    assert responses[1].error
    for slot in (0, 2):
        completion = responses[slot].chat_completion
        assert completion.response == f"answer-{slot}"
        records = received(completion.llm_observations)
        assert records[0]["reasoning"]["reasoning"] == f"slot-{slot}"
        assert records[0]["coordinates"]["batch_index"] == slot
    assert len({r["call_id"] for r in received(recorder.references)}) == 3


def test_root_and_plain_child_keep_reasoning_out_of_prompt_history():
    response = good_response()
    response.choices[
        0
    ].message.content = (
        '```repl\nanswer["content"] = llm_query("child task")\nanswer["ready"] = True\n```'
    )
    response.choices[0].message.reasoning = "ROOT_REASONING_SENTINEL"
    child = good_response()
    child.choices[0].message.reasoning = "CHILD_REASONING_SENTINEL"
    client = make_client()
    client.client.chat.completions.create.side_effect = [response, child]
    recorder = ObservationRecorder()
    logger = RLMLogger()
    with patch("rlm.core.rlm.get_client", return_value=client), observation_session(recorder):
        rlm = RLM(logger=logger, max_iterations=1)
        result = rlm.completion("task")
    assert result.response == "ok"
    trajectory = result.metadata
    assert (
        received(trajectory["iterations"][0]["llm_observations"])[0]["reasoning"]["reasoning"]
        == "ROOT_REASONING_SENTINEL"
    )
    child_result = trajectory["iterations"][0]["code_blocks"][0]["result"]["rlm_calls"][0]
    assert (
        received(child_result["llm_observations"])[0]["reasoning"]["reasoning"]
        == "CHILD_REASONING_SENTINEL"
    )
    assert len(received(trajectory["llm_observations"])) == 2
    for request in client.client.chat.completions.create.call_args_list:
        assert "REASONING_SENTINEL" not in str(request.kwargs["messages"])


def test_legacy_completion_serialization_does_not_add_capture_fields():
    original = RLMChatCompletion("model", "task", "answer", UsageSummary({}), 0.1).to_dict()
    assert "llm_observations" not in original
    assert RLMChatCompletion.from_dict(original).to_dict() == original


def test_direct_socket_call_and_logger_reuse_are_isolated():
    client = make_client(good_response())
    first = ObservationRecorder()
    second = ObservationRecorder()
    logger = RLMLogger()
    for recorder in (first, second):
        with LMHandler(client, observation_recorder=recorder) as handler:
            response = send_lm_request(handler.address, LMRequest(prompt="task"))
        assert (
            received(response.chat_completion.llm_observations)[0]["availability"] == "not_returned"
        )
    assert set(r["call_id"] for r in received(first.references)).isdisjoint(
        r["call_id"] for r in received(second.references)
    )
    logger.observation_recorder = first
    logger.clear_iterations()
    assert logger.observation_recorder is None


def test_recursive_child_and_depth_fallback_are_recorded():
    for max_depth, purpose in ((1, "plain_child"), (2, "child_turn")):
        root = good_response()
        root.choices[
            0
        ].message.content = (
            '```repl\nanswer["content"] = rlm_query("child")\nanswer["ready"] = True\n```'
        )
        child = good_response()
        child.choices[0].message.content = (
            '```repl\nanswer["content"] = "child answer"\nanswer["ready"] = True\n```'
            if max_depth == 2
            else "child answer"
        )
        child.choices[0].message.reasoning_content = "recursive reasoning"
        client = make_client()
        client.client.chat.completions.create.side_effect = [root, child]
        recorder = ObservationRecorder()
        with patch("rlm.core.rlm.get_client", return_value=client):
            rlm = RLM(
                logger=RLMLogger(),
                max_iterations=1,
                max_depth=max_depth,
                observation_recorder=recorder,
            )
            result = rlm.completion("task")
        assert result.response == "child answer"
        records = received(result.metadata["llm_observations"])
        assert [r["purpose"] for r in records] == ["root_turn", purpose]
        assert records[1]["parent_call_id"] == records[0]["call_id"]


def test_compaction_and_default_answer_capture_only_observations():
    from unittest.mock import Mock

    response = good_response()
    response.choices[0].message.reasoning_content = "PRIVATE_REASONING"
    client = make_client(response)
    recorder = ObservationRecorder()
    rlm = RLM(logger=RLMLogger(), observation_recorder=recorder)
    with LMHandler(client, observation_recorder=recorder) as handler:
        history = rlm._compact_history(handler, Mock(), [{"role": "user", "content": "task"}])
        assert rlm._default_answer(history, handler) == "ok"
    assert [r["purpose"] for r in received(recorder.references)] == ["compaction", "default_answer"]
    assert "PRIVATE_REASONING" not in str(history)
