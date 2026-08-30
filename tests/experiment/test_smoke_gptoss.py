"""U5: gpt-oss probe/offline tests, split out of test_smoke_mock.py.

Covers --config selection, the config-derived probe expectation, and the
gpt-oss probe sequence -- all offline, network blocked. Moved verbatim from
``test_smoke_mock.py`` to keep that file's size manageable; behavior and
assertions are unchanged.
"""

import json
import re
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from examples import experiment_smoke
from examples import run_experiment as run_experiment_cli
from shrlm.experiment.config import CONFIG_PATH, ExperimentConfig
from tests.experiment.test_smoke_mock import block_network

# ---------------------------------------------------------------------------
# U5: --config selection, the config-derived probe expectation, and the
# gpt-oss probe sequence -- all offline, network blocked
# ---------------------------------------------------------------------------

GPTOSS_CONFIG = CONFIG_PATH.with_name("experiment_oolong_gptoss.toml")

GPTOSS_RATES = "0.15/0.6"


def gptoss_live_config() -> ExperimentConfig:
    return experiment_smoke.live_config(GPTOSS_CONFIG)


def probe_payload(
    content: str = "OK",
    *,
    reasoning_content: str | None = None,
    prompt_tokens: int = 12,
    completion_tokens: int | None = 8,
    reasoning_tokens: int | None = None,
    finish_reason: str = "stop",
    with_usage: bool = True,
) -> dict[str, Any]:
    """A raw chat-completion payload shaped like ``response.model_dump()``."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    payload: dict[str, Any] = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if with_usage:
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if reasoning_tokens is not None:
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        payload["usage"] = usage
    return payload


class TestConfigSelection:
    """--config threads the whole gate chain: env keys, pricing attestation."""

    def test_gptoss_config_demands_the_azure_env_keys(self, capsys, monkeypatch):
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "present-but-unused")
        block_network(monkeypatch)

        assert experiment_smoke.main(["--probe", "--config", str(GPTOSS_CONFIG)]) == 1
        out = capsys.readouterr().out
        assert "AZURE_API_KEY" in out
        assert "AZURE_FOUNDRY_ENDPOINT" in out
        assert "OPENROUTER_API_KEY" not in out
        assert "Nothing was spent" in out

    def test_gptoss_config_attests_the_gptoss_rates(self, capsys, monkeypatch):
        monkeypatch.setenv("AZURE_API_KEY", "sentinel-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")
        monkeypatch.delenv("SHRLM_VERIFIED_PRICING", raising=False)
        block_network(monkeypatch)

        assert experiment_smoke.main(["--probe", "--config", str(GPTOSS_CONFIG)]) == 1
        out = capsys.readouterr().out
        assert "SHRLM_VERIFIED_PRICING" in out
        assert GPTOSS_RATES in out
        assert "Nothing was spent" in out

    def test_a_mismatched_attestation_names_the_configured_rate(self, capsys, monkeypatch):
        monkeypatch.setenv("AZURE_API_KEY", "sentinel-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("SHRLM_VERIFIED_PRICING", "0.6/3.0")  # the Kimi rate
        block_network(monkeypatch)

        assert experiment_smoke.main(["--probe", "--config", str(GPTOSS_CONFIG)]) == 1
        out = capsys.readouterr().out
        assert GPTOSS_RATES in out
        assert "Nothing was spent" in out

    def test_the_default_config_is_the_shipped_path(self):
        """Unset --config keeps the shipped file: the two loads are identical."""
        assert experiment_smoke.live_config() == experiment_smoke.live_config(CONFIG_PATH)


class TestProbeExpectation:
    def test_derives_from_the_config_not_the_provider_name(self):
        shipped = experiment_smoke.probe_expectation(experiment_smoke.live_config())
        assert shipped.expects_reasoning is False
        assert shipped.effort is None

        gptoss = experiment_smoke.probe_expectation(gptoss_live_config())
        assert gptoss.expects_reasoning is True
        assert gptoss.effort == "medium"
        assert gptoss.max_output_tokens == 16384

    def test_effort_none_expects_no_reasoning(self):
        """The derivation rule itself: "none" and None both mean no reasoning."""
        import dataclasses

        base = gptoss_live_config()
        for effort in (None, "none"):
            config = dataclasses.replace(
                base,
                backends=dataclasses.replace(
                    base.backends,
                    azure_foundry=dataclasses.replace(
                        base.backends.azure_foundry, reasoning_effort=effort
                    ),
                ),
            )
            expectation = experiment_smoke.probe_expectation(config)
            assert expectation.expects_reasoning is False


class TestCheckProbeReasoningDefault:
    """The one-argument call (tests/clients/test_azure_foundry.py's site) keeps
    today's behavior byte for byte."""

    def test_a_clean_instant_payload_passes(self):
        experiment_smoke.check_probe_reasoning(probe_payload())

    def test_a_reasoning_field_fails(self):
        with pytest.raises(experiment_smoke.SmokeError, match="reasoning_content"):
            experiment_smoke.check_probe_reasoning(probe_payload(reasoning_content="hmm"))

    def test_think_markup_fails(self):
        with pytest.raises(experiment_smoke.SmokeError, match="<think>"):
            experiment_smoke.check_probe_reasoning(probe_payload("<think>hi</think>OK"))

    def test_the_64_token_ceiling_still_applies(self):
        with pytest.raises(experiment_smoke.SmokeError, match="ceiling 64"):
            experiment_smoke.check_probe_reasoning(probe_payload(completion_tokens=65))


class TestCheckProbeReasoningExpected:
    """With reasoning expected the checks invert (KTD6)."""

    EXPECTATION = experiment_smoke.ProbeExpectation(
        expects_reasoning=True, effort="medium", max_output_tokens=16384
    )

    def test_a_reasoning_content_payload_passes(self):
        payload = probe_payload(reasoning_content="step by step", completion_tokens=5000)
        experiment_smoke.check_probe_reasoning(payload, self.EXPECTATION)

    def test_a_reasoning_token_count_alone_passes(self):
        payload = probe_payload(completion_tokens=500, reasoning_tokens=480)
        experiment_smoke.check_probe_reasoning(payload, self.EXPECTATION)

    def test_a_harmony_marker_in_content_fails_naming_it(self):
        payload = probe_payload(
            "<|channel|>analysis<|message|>x", reasoning_content="r", completion_tokens=10
        )
        with pytest.raises(experiment_smoke.SmokeError, match=re.escape("<|channel|>")):
            experiment_smoke.check_probe_reasoning(payload, self.EXPECTATION)

    def test_no_reasoning_signal_fails(self):
        with pytest.raises(experiment_smoke.SmokeError, match="no reasoning signal"):
            experiment_smoke.check_probe_reasoning(
                probe_payload(completion_tokens=8), self.EXPECTATION
            )

    def test_the_ceiling_is_max_output_tokens_not_64(self):
        # Above 64 but under max_output_tokens: fine when reasoning is expected.
        experiment_smoke.check_probe_reasoning(
            probe_payload(reasoning_content="r", completion_tokens=4096), self.EXPECTATION
        )
        with pytest.raises(experiment_smoke.SmokeError, match="max_output_tokens"):
            experiment_smoke.check_probe_reasoning(
                probe_payload(reasoning_content="r", completion_tokens=16385),
                self.EXPECTATION,
            )


class TestGptossBudgetArithmetic:
    def test_the_ungoverned_per_call_ceiling_reprices_to_gptoss(self):
        config = gptoss_live_config()
        assert experiment_smoke.ungoverned_call_ceiling(config) == pytest.approx(
            (49_152 * 0.15 + 16_384 * 0.60) / 1e6
        )

    def test_the_reasoning_probe_counts_eight_calls(self):
        config = gptoss_live_config()
        assert experiment_smoke.probe_call_count(config) == 8
        # (t=1 x n_in=2 x m=1 x 3 x 3) attribution + (1 x 8 x 3) proposal.
        assert experiment_smoke.stage_call_count(config) == 18 + 24
        assert experiment_smoke.ungoverned_call_count(config) == 8 + 18 + 24

    def test_the_probe_bound_prices_seven_full_calls_and_the_capped_one(self):
        config = gptoss_live_config()
        input_usd = 1_024 * 0.15 / 1e6
        full_call = input_usd + 16_384 * 0.60 / 1e6
        capped_call = input_usd + 16 * 0.60 / 1e6
        assert experiment_smoke.probe_spend_bound_usd(config) == pytest.approx(
            7 * full_call + capped_call
        )
        assert experiment_smoke.standalone_probe_reserve_usd(config) == pytest.approx(
            experiment_smoke.probe_spend_bound_usd(config)
        )

    def test_the_cumulative_proof_holds_for_both_shipped_configs(self):
        gptoss = gptoss_live_config()
        assert experiment_smoke.check_budget_arithmetic(gptoss) < 5.0
        shipped = experiment_smoke.live_config()
        assert experiment_smoke.probe_call_count(shipped) == experiment_smoke.PROBE_CALLS == 2
        assert experiment_smoke.check_budget_arithmetic(shipped) < 5.0


# ---------------------------------------------------------------------------
# The gpt-oss probe sequence, exercised offline with a scripted raw stub
# ---------------------------------------------------------------------------


class FakeRawResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeRunnerLM:
    """The client-path seam: a clean completion and a synthesized cost."""

    def completion(self, prompt: str) -> str:
        return "OK"

    def get_last_usage(self) -> SimpleNamespace:
        return SimpleNamespace(total_cost=0.0001, cost_source="synthesized")


class ScriptedRaw:
    """``raw_completion`` replacement: pops scripted results in call order and
    records every call's prompt and per-call overrides."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        lm: Any,
        config: ExperimentConfig,
        prompt: str,
        *,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "prompt": prompt,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_completion_tokens,
                "extra_body": extra_body,
            }
        )
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeRawResponse(result)


def rejected_request(name: str) -> BadRequestError:
    body = {"error": {"code": "unsupported_value", "param": name}}
    response = httpx.Response(
        400, request=httpx.Request("POST", "http://localhost/openai/v1/chat/completions")
    )
    return BadRequestError(f"400: {name} is not supported", response=response, body=body)


def reasoning_script(
    *,
    low_reasoning: int | None = 40,
    high_reasoning: int | None = 900,
    low_completion: int = 60,
    high_completion: int = 950,
    high_content: str = "419",
    length_finish: str = "length",
    length_content: str = "",
) -> list[Any]:
    """The seven scripted raw results, in the probe's fixed call order."""
    return [
        probe_payload(reasoning_content="checking primes", completion_tokens=30),
        probe_payload(
            "419",
            reasoning_content="quick",
            completion_tokens=low_completion,
            reasoning_tokens=low_reasoning,
        ),
        probe_payload(
            high_content,
            reasoning_content="thorough",
            completion_tokens=high_completion,
            reasoning_tokens=high_reasoning,
        ),
        probe_payload(length_content, completion_tokens=16, finish_reason=length_finish),
        rejected_request("none"),
        rejected_request("xhigh"),
        rejected_request("chat_template_kwargs"),
    ]


def stub_reasoning_probe(monkeypatch: pytest.MonkeyPatch, results: list[Any]) -> ScriptedRaw:
    block_network(monkeypatch)
    stub = ScriptedRaw(results)
    monkeypatch.setattr(experiment_smoke, "raw_completion", stub)
    monkeypatch.setattr(experiment_smoke, "runner_client", lambda config: FakeRunnerLM())
    return stub


class TestGptossProbeSequence:
    def test_the_full_sequence_passes_and_records_everything(self, monkeypatch):
        stub = stub_reasoning_probe(monkeypatch, reasoning_script())

        verdict = experiment_smoke.probe(gptoss_live_config())

        # The seven raw calls, in order, with the plan's per-call overrides.
        assert [call["reasoning_effort"] for call in stub.calls] == [
            None,
            "low",
            "high",
            None,
            "none",
            "xhigh",
            None,
        ]
        assert stub.calls[0]["prompt"] == experiment_smoke.PROBE_PROMPT
        hard = experiment_smoke.HARD_PROBE_PROMPT
        assert stub.calls[1]["prompt"] == stub.calls[2]["prompt"] == hard
        assert stub.calls[3]["prompt"] == hard  # the 16-token call, same HARD prompt
        assert stub.calls[3]["max_completion_tokens"] == 16
        assert stub.calls[6]["extra_body"] == {"chat_template_kwargs": {"thinking": False}}

        # The verdict keeps the report's existing keys and adds the new ones.
        assert verdict["client_cost"] == pytest.approx(0.0001)
        assert verdict["cost_source"] == "synthesized"
        assert verdict["input_tokens"] == 12
        assert verdict["output_tokens"] == 30
        assert verdict["sampling_args"]["reasoning_effort"] == "medium"
        assert verdict["expects_reasoning"] is True
        assert verdict["effort"] == "medium"
        assert verdict["high_low_branch"] == "reasoning_tokens"

        # Seven payload records; the informational rejections are recorded,
        # status codes and all, never asserted.
        records = verdict["payloads"]
        assert [record["name"] for record in records] == [
            "configured",
            "low",
            "high",
            "length_16",
            "none",
            "xhigh",
            "thinking_false",
        ]
        for record in records[4:]:
            assert record["status"] == "error"
            assert record["status_code"] == 400
        assert records[3]["completion_tokens"] == 16

    def test_equal_high_and_low_is_inconclusive_and_says_rerun(self, monkeypatch):
        stub_reasoning_probe(monkeypatch, reasoning_script(low_reasoning=100, high_reasoning=100))
        with pytest.raises(experiment_smoke.SmokeError, match="INCONCLUSIVE") as excinfo:
            experiment_smoke.probe(gptoss_live_config())
        message = str(excinfo.value)
        assert "Rerun the probe once" in message
        assert "reasoning_tokens" in message

    def test_high_below_low_fails_the_gate(self, monkeypatch):
        stub_reasoning_probe(monkeypatch, reasoning_script(low_reasoning=900, high_reasoning=40))
        with pytest.raises(experiment_smoke.SmokeError, match="not honoring"):
            experiment_smoke.probe(gptoss_live_config())

    def test_the_gate_falls_back_to_completion_tokens_and_records_the_branch(self, monkeypatch):
        stub_reasoning_probe(monkeypatch, reasoning_script(low_reasoning=None, high_reasoning=None))
        verdict = experiment_smoke.probe(gptoss_live_config())
        assert verdict["high_low_branch"] == "completion_tokens"

    def test_a_harmony_marker_in_any_content_fails_naming_it(self, monkeypatch):
        stub_reasoning_probe(
            monkeypatch, reasoning_script(high_content="<|channel|>final<|message|>419")
        )
        with pytest.raises(experiment_smoke.SmokeError, match=re.escape("<|channel|>")):
            experiment_smoke.probe(gptoss_live_config())

    def test_a_16_token_call_that_is_not_length_fails(self, monkeypatch):
        stub_reasoning_probe(
            monkeypatch, reasoning_script(length_finish="stop", length_content="419")
        )
        with pytest.raises(experiment_smoke.SmokeError, match="finish_reason"):
            experiment_smoke.probe(gptoss_live_config())

    def test_a_configured_call_with_no_reasoning_signal_fails(self, monkeypatch):
        script = reasoning_script()
        script[0] = probe_payload(completion_tokens=8)  # no reasoning anywhere
        stub_reasoning_probe(monkeypatch, script)
        with pytest.raises(experiment_smoke.SmokeError, match="no reasoning signal"):
            experiment_smoke.probe(gptoss_live_config())

    def test_run_probe_writes_probe_json(self, monkeypatch, tmp_path, capsys):
        stub_reasoning_probe(monkeypatch, reasoning_script())
        out_dir = tmp_path / "smoke_gptoss"

        assert experiment_smoke.run_probe(gptoss_live_config(), out_dir) == 0

        path = out_dir / experiment_smoke.PROBE_JSON_FILENAME
        assert path.exists()
        persisted = json.loads(path.read_text())
        assert persisted["expects_reasoning"] is True
        assert len(persisted["payloads"]) == 7
        out = capsys.readouterr().out
        assert "PROBE PASSED" in out
        assert str(path) in out


class TestDryRunProbeVerdict:
    """run_experiment.py --dry-run: pricing per role plus the probe verdict."""

    @staticmethod
    def gate_env(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_API_KEY", "sentinel-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("SHRLM_VERIFIED_PRICING", GPTOSS_RATES)
        block_network(monkeypatch)

    def test_probe_json_round_trips_through_the_dry_run_printer(
        self, monkeypatch, tmp_path, capsys
    ):
        self.gate_env(monkeypatch)
        probe_path = tmp_path / "probe.json"
        probe_path.write_text(
            json.dumps(
                {
                    "cost_source": "synthesized",
                    "expects_reasoning": True,
                    "high_low_branch": "reasoning_tokens",
                    "payloads": [{"name": "configured", "payload": {"huge": "elided"}}],
                }
            )
        )

        rc = run_experiment_cli.main(
            [
                "--dry-run",
                "--config",
                str(GPTOSS_CONFIG),
                "--out-dir",
                str(tmp_path / "exp"),
                "--probe-json",
                str(probe_path),
            ]
        )

        assert rc == 0
        out = capsys.readouterr().out
        # model @ in/out per azure_foundry role, priced from pricing.list_price.
        assert out.count(f"gpt-oss-120b @ {GPTOSS_RATES}") == 3
        assert "synthesized" in out
        assert "reasoning_tokens" in out
        assert "elided" not in out  # payloads stay in the file, off stdout

    def test_a_missing_probe_json_prints_no_probe_verdict_and_passes(
        self, monkeypatch, tmp_path, capsys
    ):
        self.gate_env(monkeypatch)

        # No --probe-json: the default is <out-dir>/probe.json, absent here.
        rc = run_experiment_cli.main(
            ["--dry-run", "--config", str(GPTOSS_CONFIG), "--out-dir", str(tmp_path / "exp")]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "no probe verdict" in out
        assert str(tmp_path / "exp" / "probe.json") in out
        assert "Dry run: pre-flight passed" in out
