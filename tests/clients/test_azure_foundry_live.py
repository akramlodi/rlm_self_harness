"""Live tier (U4) for the Azure AI Foundry client, split out of
``test_azure_foundry.py`` to keep that file under the file-size guideline.

A handful of tiny paid calls against the REAL configured deployment of the
env-selected config (``SHRLM_EXPERIMENT_CONFIG``, else the shipped smoke
profile), gated by KTD8 (see ``shrlm/experiment/live_gates.py``): it runs
only when that config's runner backend is azure_foundry (this is an
azure-specific tier; with another backend selected the gate would demand
that backend's credentials and then run azure code paths) AND both Azure
credentials are set AND ``SHRLM_RUN_LIVE=1`` AND pricing is attested, and
never in CI (live-marked items are deselected outright without the opt-in;
see ``tests/conftest.py``).

``TestLiveRowSelectionOffline`` at the bottom is offline ($0) and always
runs -- it only exercises the row-selection logic the live class depends on.
"""

import copy
from typing import Any

import pytest

from shrlm.experiment.live_gates import CONFIG_ENV_KEY, live_config_path, live_skip_reason


def _selected_config() -> Any:
    """The env-selected smoke-profile config the live tier runs against
    (KTD7): ``SHRLM_EXPERIMENT_CONFIG`` when set, else the shipped
    ``configs/experiment.toml``."""
    from shrlm.experiment.config import CONFIG_PATH, load_config

    return load_config(profile="smoke", path=live_config_path(CONFIG_PATH))


def _azure_live_skip() -> str | None:
    """Skip reason for the azure-specific live tier, or ``None`` to run it.

    ``live_skip_reason`` gates on the SELECTED config's runner backend, so
    with a non-azure backend selected an open gate would demand that
    backend's credentials and then run this module's azure code paths
    (backend_kwargs without pricing -> constructor error). The azure live
    tier therefore also requires the selected config to run azure_foundry;
    the pricing attestation is checked against that config's rate card.
    """
    try:
        config = _selected_config()
    except Exception as exc:  # bad SHRLM_EXPERIMENT_CONFIG must skip, not error collection
        return f"failed to load config selected via {CONFIG_ENV_KEY}: {exc!r}"
    if config.backends.runner.backend != "azure_foundry":
        return (
            "selected runner backend is not azure_foundry -- azure live tier requires "
            "an azure config (set SHRLM_EXPERIMENT_CONFIG to one)"
        )
    return live_skip_reason(config=config)


_LIVE_SKIP = _azure_live_skip()

# Output caps for the live calls. Trivial prompts under these caps keep the
# whole class in the cents at either configured rate card. Under a reasoning
# config the trivial cap is raised: reasoning tokens bill as completion
# tokens, so 64 would starve the answer by design.
LIVE_TRIVIAL_MAX_TOKENS = 64
LIVE_REASONING_TRIVIAL_MAX_TOKENS = 2048
LIVE_CAP_MAX_TOKENS = 16
# Some gateways report a couple of tokens beyond max_completion_tokens (e.g.
# a stop token); the cap test tolerates that without tolerating a busted cap.
LIVE_CAP_SLACK_TOKENS = 8
# The reasoning-effort comparison needs room for a high-effort chain.
LIVE_EFFORT_MAX_TOKENS = 8192

LIVE_TRIVIAL_PROMPT = "Reply with the single word: ok"
LIVE_EFFORT_PROMPT = (
    "In how many ways can the letters of the word FOUNDRY be arranged so that "
    "no two vowels are adjacent? Reply with just the number."
)


def _probe_expectation() -> Any:
    """What the selected config says a response must look like (KTD6)."""
    from examples.experiment_smoke import probe_expectation

    return probe_expectation(_selected_config())


def _trivial_max_tokens(expectation: Any) -> int:
    return (
        LIVE_REASONING_TRIVIAL_MAX_TOKENS
        if expectation.expects_reasoning
        else LIVE_TRIVIAL_MAX_TOKENS
    )


def _runner_mismatch(provider: Any, config: Any) -> str | None:
    """Why a provider-matrix row must skip under the selected config, or
    ``None`` when the row is the config's runner (KTD7).

    The live class loads ONLY the env-selected config, so a row whose
    backend/model differ from that config's runner has nothing real to run
    against; it skips with a reason naming the mismatch and ``-rs`` shows
    exactly one PASSED row per test. Pure, so the offline tests below
    exercise it without credentials or spend.
    """
    runner = config.backends.runner
    if (provider.backend, provider.model) == (runner.backend, runner.model):
        return None
    return (
        f"provider row {provider.id} ({provider.backend}/{provider.model}) does not "
        f"match the selected config's runner ({runner.backend}/{runner.model})"
    )


def _skip_unmatched_row(provider: Any) -> None:
    reason = _runner_mismatch(provider, _selected_config())
    if reason is not None:
        pytest.skip(reason)


def _live_runner_kwargs(max_tokens: int) -> dict[str, Any]:
    """The REAL configured runner backend_kwargs (the selected config's smoke
    profile) with only the output cap lowered for these tiny calls.

    ``backend_kwargs_for`` builds fresh dicts on every call, and the result is
    deep-copied anyway before the override, so no shared config object is ever
    mutated. Everything else -- temperature, top_p, the extra_body
    (``chat_template_kwargs`` for Kimi), a top-level ``reasoning_effort`` when
    the config sets one, and the nested ``pricing`` -- is exactly what a real
    experiment round sends.
    """
    from shrlm.experiment.config import backend_kwargs_for

    kwargs = copy.deepcopy(backend_kwargs_for(_selected_config(), "runner"))
    kwargs["sampling_args"]["max_tokens"] = max_tokens
    return kwargs


def _raw_live_completion(lm: Any, prompt: str) -> Any:
    """One paid chat completion through the client's own request builders.

    This is ``OpenAIClient.completion`` taken apart into its two halves --
    the exact ``chat.completions.create`` parameter shape, then the client's
    own ``_track_cost`` (the azure_foundry override that validates usage and
    synthesizes the cost) -- so the test can inspect the raw response
    (finish_reason, reasoning fields, usage details) while the cost path
    exercised is byte-for-byte the one every persisted run travels.
    """
    from rlm.clients.openai import _merge_extra_body, _normalize_sampling_args

    response = lm.client.chat.completions.create(
        model=lm.model_name,
        messages=[{"role": "user", "content": prompt}],
        extra_body=_merge_extra_body({}, lm.sampling_args),
        **_normalize_sampling_args(lm.sampling_args),
    )
    lm._track_cost(response, lm.model_name)
    return response


@pytest.mark.live
@pytest.mark.skipif(_LIVE_SKIP is not None, reason=_LIVE_SKIP or "live gates satisfied")
class TestAzureFoundryLive:
    """Provider-parametrized (the ``provider`` fixture from tests/conftest.py),
    but every row loads ONLY the env-selected config: a row whose
    backend/model do not match that config's runner skips with a reason
    naming the mismatch, so ``-rs`` shows exactly one PASSED row per test
    (KTD7). What each test asserts is derived from the config's
    ``probe_expectation``, never from a provider name."""

    def test_trivial_call_synthesizes_cost_and_matches_reasoning_contract(self, provider):
        """One trivial call: non-empty content, positive token counts, a
        positive synthesized cost, and a reasoning signal exactly when the
        config expects one (R10/KTD6 -- the detector is the very one the
        tier-2 smoke probe uses; it raises SmokeError naming each signal)."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        cap = _trivial_max_tokens(expectation)

        lm = get_client("azure_foundry", _live_runner_kwargs(cap))
        response = _raw_live_completion(lm, LIVE_TRIVIAL_PROMPT)
        payload = response.model_dump()

        content = payload["choices"][0]["message"]["content"]
        assert content and content.strip()

        last = lm.get_last_usage()
        assert last.total_input_tokens > 0
        assert last.total_output_tokens > 0
        # A trivial "ok" must come nowhere near the cap; cap-saturating output
        # on this prompt would itself be a runaway-reasoning smell.
        assert last.total_output_tokens < cap
        assert last.total_cost is not None and last.total_cost > 0
        assert last.cost_source == "synthesized"

        from examples.experiment_smoke import check_probe_reasoning

        check_probe_reasoning(payload, expectation)

    def test_output_cap_is_honored_not_merely_tolerated(self, provider):
        """Without reasoning, a verbose prompt under max_tokens=16 must be
        TRUNCATED: finish_reason 'length', completion tokens at the cap --
        the route enforces the cap rather than merely accepting the
        parameter. With reasoning expected, the 16-token budget is consumed
        by reasoning before any content lands, so the client's own
        ``completion()`` must raise TokenLimitExceededError -- while the
        spend-then-validate order still recorded a positive synthesized cost
        for the paid, truncated call."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        lm = get_client("azure_foundry", _live_runner_kwargs(LIVE_CAP_MAX_TOKENS))

        if expectation.expects_reasoning:
            from rlm.utils.exceptions import TokenLimitExceededError

            with pytest.raises(TokenLimitExceededError):
                lm.completion("Count from 1 to 500, one number per line.")
            last = lm.get_last_usage()
            assert last.total_cost is not None and last.total_cost > 0
            assert last.cost_source == "synthesized"
            return

        response = _raw_live_completion(lm, "Count from 1 to 500, one number per line.")
        assert response.choices[0].finish_reason == "length"
        completion_tokens = response.usage.completion_tokens
        assert completion_tokens <= LIVE_CAP_MAX_TOKENS + LIVE_CAP_SLACK_TOKENS

    def test_configured_decoding_args_are_accepted_on_the_client_path(self, provider):
        """The client's own ``completion`` path with the real configured
        sampling args (temperature / top_p / max_completion_tokens /
        extra_body / reasoning_effort all ride this request): a deployment
        that rejects any of them surfaces as an HTTP 400 -> exception here."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        lm = get_client("azure_foundry", _live_runner_kwargs(_trivial_max_tokens(expectation)))
        content = lm.completion(LIVE_TRIVIAL_PROMPT)
        assert content and content.strip()
        assert "<think>" not in content

    def test_reasoning_effort_is_honored_not_merely_accepted(self, provider):
        """R11's stop condition at the client tier: one ``low`` and one
        ``high`` raw call on a reasoning-heavy prompt must be distinguishable
        in completion tokens -- indistinguishable efforts mean the gateway
        ignores the knob. Runs only under a config that sets a real
        reasoning_effort (the trivial and cap tests cover the rest)."""
        from rlm.clients import get_client

        _skip_unmatched_row(provider)
        expectation = _probe_expectation()
        if not expectation.expects_reasoning:
            pytest.skip(
                "selected config sets no reasoning_effort; the low-vs-high "
                "comparison needs a reasoning config"
            )

        tokens: dict[str, int] = {}
        for effort in ("low", "high"):
            kwargs = _live_runner_kwargs(LIVE_EFFORT_MAX_TOKENS)
            kwargs["sampling_args"]["reasoning_effort"] = effort
            lm = get_client("azure_foundry", kwargs)
            response = _raw_live_completion(lm, LIVE_EFFORT_PROMPT)
            tokens[effort] = int(response.usage.completion_tokens)
        assert tokens["high"] > tokens["low"], (
            f"reasoning_effort indistinguishable: completion tokens {tokens} "
            "(stop condition (a) -- surface before any multi-call tier)"
        )


class TestLiveRowSelectionOffline:
    """KTD7 offline ($0): the live class loads only the env-selected config,
    and a mismatching provider row reports a skip reason naming the runner
    mismatch while the matching row runs."""

    def _gptoss_config(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from tests.conftest import CONFIG_DIR

        monkeypatch.setenv(
            "SHRLM_EXPERIMENT_CONFIG", str(CONFIG_DIR / "experiment_oolong_gptoss.toml")
        )
        return _selected_config()

    def test_mismatching_rows_name_the_runner_mismatch(self, monkeypatch):
        from tests.conftest import AZURE_KIMI, OPENROUTER_QWEN

        config = self._gptoss_config(monkeypatch)
        for row in (AZURE_KIMI, OPENROUTER_QWEN):
            reason = _runner_mismatch(row, config)
            assert reason is not None
            assert row.id in reason and row.model in reason
            assert "gpt-oss-120b" in reason

    def test_the_selected_configs_own_row_does_not_skip(self, monkeypatch):
        from tests.conftest import AZURE_GPTOSS

        config = self._gptoss_config(monkeypatch)
        assert _runner_mismatch(AZURE_GPTOSS, config) is None

    def test_env_unset_selects_the_shipped_smoke_profile(self, monkeypatch):
        """With the env unset this module resolves the config it loads today:
        the shipped configs/experiment.toml smoke profile (openrouter runner,
        so every azure row mismatches and the module-level gate skips)."""
        from shrlm.experiment.config import CONFIG_PATH

        monkeypatch.delenv("SHRLM_EXPERIMENT_CONFIG", raising=False)
        assert live_config_path(CONFIG_PATH) == CONFIG_PATH
