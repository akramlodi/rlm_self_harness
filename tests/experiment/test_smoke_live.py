"""U4 live loop tier: ONE real driver round through the actual experiment machinery.

The round is built exactly the way ``tests/optimization/test_driver.py`` builds
its real (non-mocked) rounds -- baseline harness ``H0`` through
``build_harnessed_rlm`` with the REPL, a ``Verifier``-protocol verifier, and
``run_round``'s persist-first contract -- except the client is the real
azure_foundry Kimi-K2.5 deployment with the REAL configured backend_kwargs
(``backend_kwargs_for`` over the smoke profile, runner role forced to
azure_foundry when the shipped config selects another backend).

Structure, not behavior: the tests assert the persisted round contract (files,
shas, costs, cost provenance, verdict presence) and the resume-as-no-op
contract. They NEVER assert pass/fail outcomes -- model behavior is the
model's business, and a run legitimately terminated at these tiny caps
(RESOURCE_TERMINATED) must satisfy the contract too.

Gating (KTD8, ``shrlm/experiment/live_gates.py``): this is an azure-specific
live tier, so the live class runs only when the SHIPPED smoke profile's runner
backend is azure_foundry (otherwise the gate would demand another backend's
credentials and then run azure code paths) AND both Azure credentials are set
AND ``SHRLM_RUN_LIVE=1``, never in CI. The gate-logic tests at the bottom
always run offline. Worst-case spend: 2 runs x (max_budget $0.10 + one
post-trip call ~$0.013) ~= $0.23, inside the $0.60 U4 reserve
(``examples/experiment_smoke.PYTEST_LIVE_RESERVE_USD``).
"""

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.live_gates import (
    CONFIG_ENV_KEY,
    LIVE_FLAG,
    live_config_path,
    live_skip_reason,
    pricing_attestation_mismatch,
)
from shrlm.harness_identity import harness_hash
from shrlm.optimization.driver import (
    _BACKEND_ENV_KEYS,
    RoundConfig,
    instance_lines,
    round_dir,
    run_round,
    sha256_file,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0
from tests.conftest import AZURE_KIMI, CONFIG_DIR, PROJECT_ROOT, SENTINEL_ENV, ProviderCase
from tests.optimization.test_driver import read_manifest

# The credentials the azure_foundry gate path demands.
AZURE_CREDENTIAL_KEYS = _BACKEND_ENV_KEYS["azure_foundry"]


def _azure_live_skip() -> str | None:
    """Skip reason for the azure-specific live tier, or ``None`` to run it.

    ``live_skip_reason`` gates on the SELECTED config's runner backend
    (``SHRLM_EXPERIMENT_CONFIG``, else the shipped smoke profile; KTD7), so
    with a non-azure backend selected an open gate would demand that
    backend's credentials and then run this module's azure code paths
    (backend_kwargs without pricing -> constructor error). The azure live
    tier therefore also requires the selected config to run azure_foundry;
    the pricing attestation is checked against that config's rate card.
    """
    from shrlm.experiment.config import CONFIG_PATH, load_config

    try:
        config = load_config(profile="smoke", path=live_config_path(CONFIG_PATH))
    except Exception as exc:  # bad SHRLM_EXPERIMENT_CONFIG must skip, not error collection
        return f"failed to load config selected via {CONFIG_ENV_KEY}: {exc!r}"
    if config.backends.runner.backend != "azure_foundry":
        return (
            "selected runner backend is not azure_foundry -- azure live tier requires "
            "an azure config (set SHRLM_EXPERIMENT_CONFIG to one)"
        )
    return live_skip_reason(config=config)


_LIVE_SKIP = _azure_live_skip()

# Per-run RLM limits: tiny, so a misbehaving run is cut off cheaply. A run
# that trips one lands as RESOURCE_TERMINATED -- a recorded, paid run the
# contract assertions below must (and do) tolerate.
LIVE_MAX_ITERATIONS = 4
LIVE_MAX_BUDGET_USD = 0.10
LIVE_MAX_TIMEOUT_SECONDS = 120.0


def live_instances() -> list[dict[str, Any]]:
    """Two tiny SYNTHETIC instances with known answer formats; no dataset
    downloads, no fixtures -- the prompts are the whole environment."""
    return [
        {
            "id": "live-bfs-tiny",
            "question": "one-hop BFS on a two-node graph",
            "prompt": (
                "A directed graph has exactly one edge: A -> B. Starting a BFS at "
                "node A, which single node is at depth 1? Answer with just that "
                "node's name."
            ),
            "gold": "B",
        },
        {
            "id": "live-echo-tiny",
            "question": "echo a fixed word",
            "prompt": "Answer with exactly the word PAPER (uppercase, nothing else).",
            "gold": "PAPER",
        },
    ]


@dataclass
class ContainsGoldVerifier:
    """Verifier-protocol verdict from a trivial substring check.

    Structure, not behavior: it exists so every run gets a real ``Verdict``
    (any cause); no test below asserts which way it went.
    """

    calls: int = 0

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        self.calls += 1
        gold = str(instance["gold"])
        if gold in (produced or ""):
            return Verdict(passed=True, cause=None, gold=gold, produced=produced)
        return Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold=gold,
            produced=produced,
            detail="gold string not found in produced answer",
        )


def smoke_config_for(provider: ProviderCase) -> Any:
    """The shipped smoke profile with the runner role switched to ``provider``.

    A no-op when the shipped config already selects the row's backend and
    model; otherwise the runner endpoint is replaced, the row's ``azure_table``
    overrides (if any) are applied to ``[backends.azure_foundry]``, and the
    shipped provider tables and ``[pricing.list_price]`` supply the rest.
    """
    from dataclasses import replace

    from shrlm.experiment.config import load_config

    config = load_config(profile="smoke")
    runner = config.backends.runner
    if (runner.backend, runner.model) == (provider.backend, provider.model):
        return config
    backends = replace(
        config.backends,
        runner=replace(runner, backend=provider.backend, model=provider.model),
    )
    if provider.azure_table:
        assert backends.azure_foundry is not None
        backends = replace(
            backends, azure_foundry=replace(backends.azure_foundry, **provider.azure_table)
        )
    return replace(config, backends=backends)


def azure_smoke_config() -> Any:
    """The smoke profile the paid azure tier runs (KTD7).

    The env-selected config (``SHRLM_EXPERIMENT_CONFIG``) is used as-is when
    its runner already selects azure_foundry (e.g. the gpt-oss profile). With
    the env unset the shipped smoke profile runs openrouter, so -- exactly as
    before -- the runner role is forced to the Kimi row."""
    from shrlm.experiment.config import CONFIG_PATH, load_config

    config = load_config(profile="smoke", path=live_config_path(CONFIG_PATH))
    if config.backends.runner.backend == "azure_foundry":
        return config
    return smoke_config_for(AZURE_KIMI)


def make_live_round_config(out_dir: Path, config: Any = None) -> RoundConfig:
    """The real construction path: H0 + real configured runner backend_kwargs.

    ``config`` defaults to ``azure_smoke_config()`` (the paid tier); the
    offline matrix passes ``smoke_config_for(provider)``. ``backend_kwargs_for``
    builds fresh dicts per call; the deep copy makes doubly sure no shared
    config object can be mutated through the RoundConfig.
    """
    from shrlm.experiment.config import backend_kwargs_for

    if config is None:
        config = azure_smoke_config()
    backend_kwargs = copy.deepcopy(backend_kwargs_for(config, "runner"))
    return RoundConfig(
        round_index=1,
        harness=H0,
        instances=live_instances(),
        verifier=ContainsGoldVerifier(),
        out_dir=out_dir,
        backend=config.backends.runner.backend,
        backend_kwargs=backend_kwargs,
        attempts=1,
        max_iterations=LIVE_MAX_ITERATIONS,
        max_budget=LIVE_MAX_BUDGET_USD,
        max_timeout=LIVE_MAX_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="module")
def live_round(tmp_path_factory: pytest.TempPathFactory) -> tuple[RoundConfig, Path]:
    """Run the one paid round once for the whole module (only ever invoked
    when the gated class actually executes)."""
    out_dir = tmp_path_factory.mktemp("live_round")
    config = make_live_round_config(out_dir)
    run_round(config)
    return config, round_dir(out_dir, config.round_index)


@pytest.mark.live
@pytest.mark.skipif(_LIVE_SKIP is not None, reason=_LIVE_SKIP or "live gates satisfied")
class TestLiveDriverRound:
    def test_round_directory_contract(self, live_round):
        config, round_path = live_round

        # harness.json: the H0 identity envelope.
        envelope = json.loads((round_path / "harness.json").read_text())
        assert envelope["hash"] == harness_hash(H0)

        # instances.jsonl: byte-identical to the canonical rendering.
        assert (round_path / "instances.jsonl").read_text() == instance_lines(config.instances)

        # runs.jsonl: one line per configured run, each with a verdict (any
        # cause) and a sha-verified trace file.
        entries = read_manifest(round_path)
        assert len(entries) == len(config.instances)
        assert {entry["instance_id"] for entry in entries} == {
            instance["id"] for instance in config.instances
        }
        for entry in entries:
            verdict = entry["verdict"]
            assert isinstance(verdict, dict)
            assert isinstance(verdict["passed"], bool)
            assert entry["cause"] == verdict["cause"]
            trace = round_path / entry["trace_path"]
            assert trace.is_file()
            assert sha256_file(trace) == entry["trace_sha256"]

    def test_usage_and_cost_provenance(self, live_round):
        """Strict for completed runs; tolerant only where termination
        genuinely changes the record (see driver._persist_run and
        _partial_completion): a RESOURCE_TERMINATED run's usage is rebuilt
        from the limit exception, so its tokens may honestly be zero, its
        cost is only present when the exception carried a spent figure
        (BudgetExceededError does; timeout/token/error limits do not), and
        cost_source is always absent because exception-rebuilt usage carries
        no provenance. The line is flagged usage_lower_bound instead."""
        _config, round_path = live_round
        entries = read_manifest(round_path)

        for entry in entries:
            assert entry["execution_time"] > 0.0
            if entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value:
                assert entry["usage_lower_bound"] is True
                assert "cost_source" not in entry
                if entry["cost"] is not None:
                    assert entry["cost"] > 0.0
            else:
                assert entry["usage_lower_bound"] is False
                assert entry["cost"] is not None and entry["cost"] > 0.0
                assert entry["cost_source"] == "synthesized"
                assert entry["input_tokens"] > 0
                assert entry["output_tokens"] > 0

    def test_resume_is_a_persisted_no_op(self, live_round):
        """Persist-first: re-invoking run_round over the completed round must
        reuse the manifest -- byte-identical runs.jsonl, the same trace file
        set, no new spend (a re-spend would rewrite/append manifest bytes)."""
        config, round_path = live_round
        manifest_before = (round_path / "runs.jsonl").read_bytes()
        traces_before = sorted(path.name for path in (round_path / "runs").iterdir())

        entries = run_round(make_live_round_config(config.out_dir))

        assert (round_path / "runs.jsonl").read_bytes() == manifest_before
        assert sorted(path.name for path in (round_path / "runs").iterdir()) == traces_before
        assert len(entries) == len(config.instances)


# ---------------------------------------------------------------------------
# Offline: the exact live construction path over a scripted client ($0)
# ---------------------------------------------------------------------------


class TestLiveRoundConstructionOffline:
    """Prove the round the live tier would pay for is well-formed, offline,
    once per provider-matrix row (KTD5).

    Same ``make_live_round_config`` (H0, the row's backend, the real
    smoke-profile backend_kwargs), run through the scripted-client seam every
    driver test uses -- so a backend_kwargs shape error, a sensitive-kwarg
    refusal, or a broken resume would surface here for free, never on the
    paid tier. The Kimi row is byte-for-byte the round the azure tier pays for.
    """

    def test_live_config_runs_and_resumes_through_run_round(
        self, provider: ProviderCase, tmp_path, monkeypatch
    ):
        import rlm.core.rlm as rlm_module
        from tests.optimization.test_driver import ClientFactory, final

        # The driver refuses to start a round without the backend's credentials.
        assert set(provider.env_keys) == set(_BACKEND_ENV_KEYS[provider.backend])
        for key in provider.env_keys:
            monkeypatch.setenv(key, SENTINEL_ENV[key])
        factory = ClientFactory([final("B"), final("WRONG")], cost_source=provider.cost_source)
        monkeypatch.setattr(rlm_module, "get_client", factory)

        config = make_live_round_config(tmp_path, smoke_config_for(provider))
        assert config.backend == provider.backend
        assert config.backend_kwargs["model_name"] == provider.model
        # The real configured kwargs the live tier sends: nested pricing (which
        # the driver's sensitive-kwarg scan must accept) exactly when the row
        # carries pricing, and the row's extra_body (instant mode for Kimi:
        # chat_template_kwargs == {"thinking": False}).
        assert ("pricing" in config.backend_kwargs) == (provider.pricing is not None)
        if provider.pricing is not None:
            assert set(config.backend_kwargs["pricing"]) == {
                "input_per_million",
                "output_per_million",
            }
        sampling = config.backend_kwargs["sampling_args"]
        assert sampling["extra_body"] == provider.expected_extra_body
        assert ("reasoning_effort" in sampling) == (
            "reasoning_effort" in provider.expected_sampling_keys
        )

        entries = run_round(config)
        assert len(entries) == len(live_instances())
        assert all(entry["cost_source"] == provider.cost_source for entry in entries)
        assert all(isinstance(entry["verdict"], dict) for entry in entries)

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        resumed = run_round(make_live_round_config(tmp_path, smoke_config_for(provider)))
        assert len(resumed) == len(entries)
        assert idle.total_calls == 0

    def test_kimi_row_is_the_round_the_azure_tier_pays_for(self, tmp_path, monkeypatch):
        """The paid azure tier's default construction and the matrix's Kimi
        row build the same round: same backend and byte-identical kwargs.

        Offline and always-run (no live opt-in required), so it must pin the
        default Kimi config regardless of any ``SHRLM_EXPERIMENT_CONFIG``
        left set in the environment.
        """
        monkeypatch.delenv(CONFIG_ENV_KEY, raising=False)
        paid = make_live_round_config(tmp_path)
        matrix = make_live_round_config(tmp_path, smoke_config_for(AZURE_KIMI))
        assert paid.backend == matrix.backend == "azure_foundry"
        assert paid.backend_kwargs == matrix.backend_kwargs
        assert matrix.backend_kwargs["sampling_args"]["extra_body"]["chat_template_kwargs"] == {
            "thinking": False
        }


# ---------------------------------------------------------------------------
# Offline: the gate logic itself (always runs; spends nothing)
# ---------------------------------------------------------------------------


def configured_list_price_attestation() -> str:
    """The attestation string matching the shipped [pricing.list_price] (the
    azure branch checks the attestation against the config even when the
    shipped roles run another backend)."""
    from shrlm.experiment.config import load_config

    pricing = load_config(profile="smoke").pricing.list_price
    return f"{pricing.input_per_million}/{pricing.output_per_million}"


class TestLiveGateOffline:
    ALL_GATES_OPEN = {
        "AZURE_API_KEY": "sentinel-key",
        "AZURE_FOUNDRY_ENDPOINT": "https://sentinel.services.ai.azure.com",
        LIVE_FLAG: "1",
        "SHRLM_VERIFIED_PRICING": configured_list_price_attestation(),
    }

    def test_all_gates_open_returns_none(self):
        assert live_skip_reason(self.ALL_GATES_OPEN, runner_backend="azure_foundry") is None

    def test_ci_wins_over_fully_credentialed_environment(self):
        env = dict(self.ALL_GATES_OPEN, CI="true")
        reason = live_skip_reason(env, runner_backend="azure_foundry")
        assert reason is not None and "CI" in reason

    def test_empty_ci_value_does_not_gate(self):
        env = dict(self.ALL_GATES_OPEN, CI="")
        assert live_skip_reason(env, runner_backend="azure_foundry") is None

    def test_credentials_alone_never_spend(self):
        env = {key: "sentinel" for key in AZURE_CREDENTIAL_KEYS}
        reason = live_skip_reason(env, runner_backend="azure_foundry")
        assert reason is not None and LIVE_FLAG in reason

    def test_flag_value_must_be_exactly_one(self):
        env = dict(self.ALL_GATES_OPEN, **{LIVE_FLAG: "true"})
        reason = live_skip_reason(env, runner_backend="azure_foundry")
        assert reason is not None and LIVE_FLAG in reason

    def test_each_missing_credential_is_named(self):
        for missing in AZURE_CREDENTIAL_KEYS:
            env = {key: value for key, value in self.ALL_GATES_OPEN.items() if key != missing}
            reason = live_skip_reason(env, runner_backend="azure_foundry")
            assert reason is not None and missing in reason

    def test_openrouter_backend_requires_only_its_own_credential(self):
        """Backend-conditionality (R4/KTD9 fallback): an openrouter runner is
        gated on OPENROUTER_API_KEY and the flag, and NO pricing attestation
        is demanded (openrouter reports provider costs directly)."""
        reason = live_skip_reason({}, runner_backend="openrouter")
        assert reason is not None and "OPENROUTER_API_KEY" in reason
        assert "AZURE" not in reason

        env = {"OPENROUTER_API_KEY": "sentinel"}
        reason = live_skip_reason(env, runner_backend="openrouter")
        assert reason is not None and LIVE_FLAG in reason

        env[LIVE_FLAG] = "1"
        assert live_skip_reason(env, runner_backend="openrouter") is None

    def test_ci_wins_before_any_backend_is_considered(self):
        reason = live_skip_reason({"CI": "true"}, runner_backend="openrouter")
        assert reason is not None and "CI" in reason

    def test_unknown_runner_backend_fails_loud(self):
        with pytest.raises(ValueError, match="bogus_backend"):
            live_skip_reason({}, runner_backend="bogus_backend")

    def test_reason_never_echoes_credential_values(self):
        env = {
            "AZURE_API_KEY": "sentinel-value-1f9c",
            "AZURE_FOUNDRY_ENDPOINT": "https://sentinel-value-2e8d.services.ai.azure.com",
        }
        reason = live_skip_reason(env, runner_backend="azure_foundry")
        assert reason is not None
        assert "sentinel-value-1f9c" not in reason
        assert "sentinel-value-2e8d" not in reason


if __name__ == "__main__":
    pytest.main([__file__])


class TestPricingAttestationOffline:
    """The pricing-verification prerequisite: live tiers fail fast when the
    attested rate does not match the configured [pricing.list_price]."""

    def test_absent_attestation_names_the_variable(self) -> None:
        reason = pricing_attestation_mismatch(None, 0.6, 3.0)
        assert reason is not None and "SHRLM_VERIFIED_PRICING" in reason

    def test_malformed_attestation_rejected(self) -> None:
        assert pricing_attestation_mismatch("cheap", 0.6, 3.0) is not None
        assert pricing_attestation_mismatch("0.60", 0.6, 3.0) is not None
        assert pricing_attestation_mismatch("a/b", 0.6, 3.0) is not None

    def test_mismatched_attestation_rejected(self) -> None:
        reason = pricing_attestation_mismatch("0.70/3.00", 0.6, 3.0)
        assert reason is not None and "does not match" in reason

    def test_matching_attestation_passes(self) -> None:
        assert pricing_attestation_mismatch("0.60/3.00", 0.6, 3.0) is None
        assert pricing_attestation_mismatch("0.6/3.0", 0.6, 3.0) is None

    def test_live_gate_requires_attestation(self) -> None:
        env = {
            "AZURE_API_KEY": "sentinel-key",
            "AZURE_FOUNDRY_ENDPOINT": "https://sentinel.services.ai.azure.com",
            "SHRLM_RUN_LIVE": "1",
        }
        reason = live_skip_reason(env, runner_backend="azure_foundry")
        assert reason is not None and "SHRLM_VERIFIED_PRICING" in reason
        env["SHRLM_VERIFIED_PRICING"] = configured_list_price_attestation()
        assert live_skip_reason(env, runner_backend="azure_foundry") is None

    def test_live_gate_checks_the_attestation_against_the_caller_s_config(self) -> None:
        """A live test spending under a different config must pass that config:
        the Kimi rate card (0.6/3.0) is not the shipped smoke card (0.1/0.3)."""
        from dataclasses import replace

        from shrlm.experiment.config import load_config

        config = load_config(profile="smoke")
        other = replace(
            config,
            pricing=replace(
                config.pricing,
                list_price=replace(
                    config.pricing.list_price, input_per_million=0.6, output_per_million=3.0
                ),
            ),
        )
        env = {
            "AZURE_API_KEY": "sentinel-key",
            "AZURE_FOUNDRY_ENDPOINT": "https://sentinel.services.ai.azure.com",
            "SHRLM_RUN_LIVE": "1",
            "SHRLM_VERIFIED_PRICING": "0.6/3.0",
        }
        assert live_skip_reason(env, runner_backend="azure_foundry", config=other) is None
        default = live_skip_reason(env, runner_backend="azure_foundry")
        assert default is not None and "does not match" in default


class TestLiveConfigPathOffline:
    """KTD7: ``SHRLM_EXPERIMENT_CONFIG`` selects the config for every live
    tier; unset or empty, each load site keeps its OWN current default."""

    def test_env_unset_resolves_each_live_module_default(self, monkeypatch):
        """Each live module resolves the config it loads today: the shipped
        smoke path for the client-live and smoke-live tiers (and the smoke
        script), the Kimi and OOLONG TOMLs for the two recursion modules."""
        from shrlm.experiment.config import CONFIG_PATH
        from tests.experiment.test_oolong_recursion_live import OOLONG_CONFIG
        from tests.experiment.test_recursion_live import KIMI_CONFIG

        monkeypatch.delenv(CONFIG_ENV_KEY, raising=False)
        assert live_config_path(CONFIG_PATH) == CONFIG_PATH
        assert (
            live_config_path(KIMI_CONFIG) == KIMI_CONFIG == Path("configs/experiment_kimiK25.toml")
        )
        assert (
            live_config_path(OOLONG_CONFIG)
            == OOLONG_CONFIG
            == Path("configs/experiment_oolong.toml")
        )

    def test_empty_env_resolves_the_default(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_KEY, "")
        default = Path("configs/experiment.toml")
        assert live_config_path(default) == default

    def test_env_set_wins_over_every_default(self, monkeypatch):
        selected = "configs/experiment_oolong_gptoss.toml"
        monkeypatch.setenv(CONFIG_ENV_KEY, selected)
        for default in ("configs/experiment.toml", Path("configs/experiment_kimiK25.toml")):
            assert live_config_path(default) == Path(selected)

    def test_gptoss_attestation_is_checked_against_its_own_rate_card(self):
        """Proof-first (U6): attesting the KIMI rates ('0.60/3.00') against
        the gpt-oss config must be refused with a reason naming the gpt-oss
        card 0.15/0.6 -- the gate compares against the SELECTED config."""
        from shrlm.experiment.config import load_config

        config = load_config("full", path=CONFIG_DIR / "experiment_oolong_gptoss.toml")
        env = {
            "AZURE_API_KEY": "sentinel-key",
            "AZURE_FOUNDRY_ENDPOINT": "https://sentinel.services.ai.azure.com",
            LIVE_FLAG: "1",
            "SHRLM_VERIFIED_PRICING": "0.60/3.00",
        }
        reason = live_skip_reason(env, config=config)
        assert reason is not None and "does not match" in reason
        assert "0.15/0.6" in reason

        env["SHRLM_VERIFIED_PRICING"] = "0.15/0.6"
        assert live_skip_reason(env, config=config) is None


class TestLiveDeselectionOffline:
    """The ``live``-marker deselection hook (tests/conftest.py): live items
    are DESELECTED -- never executed -- unless ``SHRLM_RUN_LIVE == "1"`` and
    ``CI`` is unset; the modules' own skipif gates then name any still-missing
    gate in ``-rs``. Proven through a real pytest subprocess collection."""

    TARGET = "tests/experiment/test_oolong_recursion_live.py"

    def _collect(self, overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        for key in ("SHRLM_RUN_LIVE", "CI", CONFIG_ENV_KEY):
            env.pop(key, None)
        env.update(overrides)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "live",
                "-p",
                "no:cacheprovider",
                self.TARGET,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=300,
        )

    def test_without_the_opt_in_zero_live_items_are_selected(self):
        result = self._collect({})
        assert result.returncode == 5, result.stdout + result.stderr  # nothing selected
        assert "deselected" in result.stdout
        assert "::" not in result.stdout  # no test id survived collection

    def test_ci_deselects_even_with_the_opt_in_flag(self):
        result = self._collect({"SHRLM_RUN_LIVE": "1", "CI": "true"})
        assert result.returncode == 5, result.stdout + result.stderr
        assert "deselected" in result.stdout
        assert "::" not in result.stdout

    def test_the_opt_in_without_ci_selects_the_live_items(self):
        """Selection (not execution: this is --collect-only, $0) resumes with
        the opt-in; the skipif gates still guard actual runs."""
        result = self._collect({"SHRLM_RUN_LIVE": "1"})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "TestOolongRecursesLive" in result.stdout
