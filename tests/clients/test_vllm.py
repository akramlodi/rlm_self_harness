"""Tests for the local/self-hosted vLLM client.

Mocked tier only: zero network, always runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlm.clients.vllm import VLLMClient, rewrite_python_fence_to_repl


def _make_response(content: str) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(finish_reason="stop", message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_client(response=None):
    with (
        patch("rlm.clients.openai.openai.OpenAI") as mock_openai,
        patch("rlm.clients.openai.openai.AsyncOpenAI"),
    ):
        mock_sync = MagicMock()
        if response is not None:
            mock_sync.chat.completions.create.return_value = response
        mock_openai.return_value = mock_sync
        return VLLMClient(
            model_name="qwen3.6:35b-a3b", base_url="http://localhost:11434/v1", api_key="ollama"
        )


class TestRewritePythonFenceToRepl:
    def test_python_fence_rewritten(self):
        text = "```python\nprint(1)\n```"
        assert rewrite_python_fence_to_repl(text) == "```repl\nprint(1)\n```"

    def test_py_fence_rewritten(self):
        text = "```py\nprint(1)\n```"
        assert rewrite_python_fence_to_repl(text) == "```repl\nprint(1)\n```"

    def test_repl_fence_left_unchanged(self):
        text = "```repl\nprint(1)\n```"
        assert rewrite_python_fence_to_repl(text) == text

    def test_no_fence_left_unchanged(self):
        text = "Here is my plan in prose, no code yet."
        assert rewrite_python_fence_to_repl(text) == text

    def test_only_fence_opener_rewritten_not_prose(self):
        text = 'I will write some python code.\n```python\nprint("python")\n```'
        result = rewrite_python_fence_to_repl(text)
        assert result == 'I will write some python code.\n```repl\nprint("python")\n```'
        assert "python code" in result  # prose untouched

    def test_pythonic_like_tag_not_rewritten(self):
        text = "```pythonic\nprint(1)\n```"
        assert rewrite_python_fence_to_repl(text) == text


class TestVLLMClientNormalization:
    def test_completion_rewrites_python_fence(self):
        response = _make_response("```python\nanswer['ready'] = True\n```")
        client = _make_client(response=response)
        result = client.completion("hi")
        assert result == "```repl\nanswer['ready'] = True\n```"

    def test_completion_leaves_repl_fence_unchanged(self):
        response = _make_response("```repl\nanswer['ready'] = True\n```")
        client = _make_client(response=response)
        result = client.completion("hi")
        assert result == "```repl\nanswer['ready'] = True\n```"
