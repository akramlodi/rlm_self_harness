"""Tests for the U2 cost governor: caps, tighten-only merge, circuit breaker.

Cost governance is experiment-owned (KTD3): evaluation caps bind at the RLM
constructor through the round config, a candidate's enabled S6 policy may
tighten but never exceed them (with the runner's S6-ownership guard honored by
omitting the constructor value when the policy declares ``max_depth``), and a
per-candidate cumulative-spend circuit breaker stops a candidate's remaining
runs the moment its persisted spend crosses the budget -- completed runs stand,
and the candidate surfaces as over_budget rather than silently vanishing (R3,
R4).

The offline seam is the repo's usual one: patch ``rlm.core.rlm.get_client``
with a factory of scripted clients that stub per-call costs (plain MockLM
reports no cost, and the breaker prices runs from persisted costs alone).
"""

import json
import os
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
import shrlm.optimization.costs as costs_module
import shrlm.runner as runner_module
from rlm.core.types import ModelUsageSummary, UsageSummary
from shrlm.optimization.candidates import GATE_CAPS, CandidateRejection
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    HardDeadlineExceeded,
    SplitClaimedError,
    ValidationCaps,
    breaker_run_cost,
    claim_split,
    governed_limits,
    hard_deadline_seconds,
    run_governed_round,
)
from shrlm.optimization.driver import RoundConfig, round_dir, run_round
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0, build_runtime_policy
from shrlm.runner import build_harnessed_rlm
from tests.mock_lm import MockLM

COST_PER_CALL = 0.001

# Per-run budget 0.0015: one scripted call per run stays under it, a two-call
# run crosses it after its second iteration (BudgetExceededError, spent 0.002).
CAPS = ValidationCaps(
    max_depth=2,
    max_iterations=3,
    max_budget=0.0015,
    max_timeout=60.0,
    candidate_budget=0.01,
)


# ---------------------------------------------------------------------------
# Offline seam: scripted clients with stubbed per-call cost
# ---------------------------------------------------------------------------


class ScriptedLM(MockLM):
    """A ``MockLM`` popping from a shared script, with a stubbed per-call cost.

    Exhausting the script raises rather than improvising: a test whose scripted
    turns run out is a broken test, not a passing run.
    """

    def __init__(self, script: list[str], cost_per_call: float):
        super().__init__(model_name="mock-model")
        self._script = script
        self._cost_per_call = cost_per_call

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self._call_count += 1
        if not self._script:
            raise IndexError("ScriptedLM: script exhausted")
        return self._script.pop(0)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=self._call_count * 10,
                    total_output_tokens=self._call_count * 10,
                    total_cost=self._cost_per_call * self._call_count,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=10,
            total_cost=self._cost_per_call,
        )


# A script turn that makes ``HangingScriptLM`` block forever: the offline
# stand-in for candidate code hanging inside a live call in the host loop.
HANG = "__HANG__"


class HangingScriptLM(ScriptedLM):
    """A ``ScriptedLM`` whose ``HANG`` turns never return.

    The sleep loop is interruptible Python bytecode, exactly like a candidate
    middleware's ``while True``: only a signal-raised exception (the hard
    deadline backstop) can break it -- the runtime's own between-iteration
    timeout check is never reached.
    """

    def completion(self, prompt: str | dict[str, Any]) -> str:
        turn = super().completion(prompt)
        if turn == HANG:
            while True:
                time.sleep(0.01)
        return turn


class UnrecordedHangingScriptLM(ScriptedLM):
    """A client that hangs *before* the call is recorded.

    The counterpart to ``HangingScriptLM``, which counts the request and then
    blocks -- there the runtime has a real figure to publish for the
    terminated run. Here nothing was recorded at all, which is the case that
    keeps today's ceiling pricing: the breaker cannot know what a run it has
    no figure for cost, so it assumes the worst.
    """

    def completion(self, prompt: str | dict[str, Any]) -> str:
        if self._script and self._script[0] == HANG:
            while True:
                time.sleep(0.01)
        return super().completion(prompt)


class ClientFactory:
    """Stands in for ``get_client``: fresh client per call, one shared script."""

    def __init__(
        self,
        script: list[str],
        cost_per_call: float = COST_PER_CALL,
        client_cls: type[ScriptedLM] = ScriptedLM,
    ):
        self.script = list(script)
        self.cost_per_call = cost_per_call
        self.client_cls = client_cls
        self.clients: list[ScriptedLM] = []

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> ScriptedLM:
        client = self.client_cls(self.script, self.cost_per_call)
        self.clients.append(client)
        return client

    @property
    def total_calls(self) -> int:
        return sum(client._call_count for client in self.clients)


def final(content: str) -> str:
    """A ```repl``` block that returns ``content`` from the answer variable."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


@dataclass
class GoldVerifier:
    """Deterministic string-match verifier."""

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold = str(instance["gold"])
        if produced == gold:
            return Verdict(passed=True, cause=None, gold=gold, produced=produced)
        return Verdict(passed=False, cause=VerifierCause.WRONG_VALUE, gold=gold, produced=produced)


def make_instances(n: int = 3) -> list[dict[str, Any]]:
    return [{"id": f"inst-{i}", "prompt": f"context {i}", "gold": "RIGHT"} for i in range(1, n + 1)]


def enabled_policy(**fields: Any) -> dict[str, Any]:
    """H0's disabled policy floor, switched on with the given fields set."""
    return build_runtime_policy() | {"enabled": True} | fields


def make_config(
    tmp_path: Path,
    caps: ValidationCaps = CAPS,
    policy: dict[str, Any] | None = None,
    **overrides: Any,
) -> RoundConfig:
    """A round config whose limits come from the governor, the U3 wiring."""
    harness = H0 if policy is None else replace(H0, runtime_policy=policy)
    limits = governed_limits("cand", harness.runtime_policy, caps)
    assert not isinstance(limits, CandidateRejection)
    values: dict[str, Any] = {
        "round_index": 1,
        "harness": harness,
        "instances": make_instances(),
        "verifier": GoldVerifier(),
        "out_dir": tmp_path,
        "backend": "openai",
        "backend_kwargs": {"model_name": "costs-test"},
        **limits,
    }
    values.update(overrides)
    return RoundConfig(**values)


def read_manifest(round_path: Path) -> list[dict[str, Any]]:
    lines = (round_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def manifest_entry(
    run_id: str = "inst-1__a01",
    cost: float | None = COST_PER_CALL,
    cause: str | None = None,
    passed: bool = True,
) -> dict[str, Any]:
    """A ``runs.jsonl``-shaped line, the breaker's input contract."""
    return {
        "run_id": run_id,
        "instance_id": run_id.split("__")[0],
        "attempt": 1,
        "passed": passed,
        "cause": cause,
        "cost": cost,
    }


def spy_on_rlm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the kwargs every RLM the runner builds actually received."""
    captured: list[dict[str, Any]] = []
    real_rlm = runner_module.RLM

    class SpyRLM(real_rlm):
        def __init__(self, **kwargs: Any):
            captured.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(runner_module, "RLM", SpyRLM)
    return captured


# ---------------------------------------------------------------------------
# ValidationCaps: the experiment-owned config
# ---------------------------------------------------------------------------


class TestValidationCaps:
    @pytest.mark.parametrize(
        "field",
        ["max_depth", "max_iterations", "max_budget", "max_timeout", "candidate_budget"],
    )
    def test_non_positive_cap_is_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            replace(CAPS, **{field: 0})

    def test_s6_caps_names_only_the_policy_expressible_fields(self):
        # The runner already forbids S6 from declaring budget/timeout
        # (EXPERIMENT_OWNED_KEYS), so max_depth is the whole overlap.
        assert CAPS.s6_caps() == {"max_depth": 2}


# ---------------------------------------------------------------------------
# Tighten-only merge and the S6-ownership forwarding rule
# ---------------------------------------------------------------------------


class TestGovernedLimits:
    def test_silent_policy_forwards_every_cap(self):
        limits = governed_limits("cand", build_runtime_policy(), CAPS)
        assert limits == {
            "max_depth": 2,
            "max_iterations": 3,
            "max_budget": 0.0015,
            "max_timeout": 60.0,
        }

    def test_disabled_policy_values_are_inert_and_the_cap_forwards(self):
        # A disabled policy binds nothing at runtime, so even an over-cap value
        # passes trivially (the U1 comparison) and the cap stays in force.
        policy = build_runtime_policy() | {"max_depth": 5}
        limits = governed_limits("cand", policy, CAPS)
        assert not isinstance(limits, CandidateRejection)
        assert limits["max_depth"] == 2

    def test_enabled_policy_above_the_cap_is_a_structured_rejection(self):
        result = governed_limits("cand-deep", enabled_policy(max_depth=3), CAPS)
        assert isinstance(result, CandidateRejection)
        assert result.candidate_id == "cand-deep"
        assert result.gate == GATE_CAPS
        assert "max_depth" in result.reason

    def test_enabled_policy_below_the_cap_omits_the_constructor_value(self):
        limits = governed_limits("cand", enabled_policy(max_depth=1), CAPS)
        assert not isinstance(limits, CandidateRejection)
        assert "max_depth" not in limits
        assert limits == {"max_iterations": 3, "max_budget": 0.0015, "max_timeout": 60.0}

    def test_declaring_candidate_builds_with_its_own_tighter_depth_in_force(self):
        harness = replace(H0, runtime_policy=enabled_policy(max_depth=1))
        limits = governed_limits("cand", harness.runtime_policy, CAPS)
        assert not isinstance(limits, CandidateRejection)
        # No double-declaration ValueError: the policy binds via the runner.
        harnessed = build_harnessed_rlm(
            harness, backend="openai", backend_kwargs={"model_name": "costs-test"}, **limits
        )
        assert harnessed.rlm.max_depth == 1
        assert harnessed.rlm.max_budget == CAPS.max_budget
        assert harnessed.rlm.max_timeout == CAPS.max_timeout


# ---------------------------------------------------------------------------
# Caps bind at the RLM constructor through the round config
# ---------------------------------------------------------------------------


class TestCapsBindAtTheConstructor:
    def test_round_config_forwards_the_caps_to_the_rlm(self, tmp_path, monkeypatch):
        captured = spy_on_rlm(monkeypatch)
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        run_round(make_config(tmp_path))

        assert len(captured) == 1
        kwargs = captured[0]
        assert kwargs["max_depth"] == CAPS.max_depth
        assert kwargs["max_iterations"] == CAPS.max_iterations
        assert kwargs["max_budget"] == CAPS.max_budget
        assert kwargs["max_timeout"] == CAPS.max_timeout

    def test_s6_declaring_candidate_runs_without_the_double_declaration_error(
        self, tmp_path, monkeypatch
    ):
        captured = spy_on_rlm(monkeypatch)
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path, policy=enabled_policy(max_depth=1))
        entries = run_round(config)

        assert len(entries) == 3
        assert captured[0]["max_depth"] == 1  # the candidate's tighter value
        assert captured[0]["max_budget"] == CAPS.max_budget


# ---------------------------------------------------------------------------
# Terminated-run cost policy: what one persisted run charges the breaker
# ---------------------------------------------------------------------------


class TestBreakerRunCost:
    def test_reported_cost_is_charged_verbatim(self):
        assert breaker_run_cost(manifest_entry(cost=0.0007), CAPS) == pytest.approx(0.0007)

    def test_budget_terminated_run_charges_its_persisted_spent_amount(self):
        # The driver persists BudgetExceededError.spent as the run's cost, and
        # the breaker charges that actual figure even above the per-run cap.
        entry = manifest_entry(
            cost=0.002, cause=VerifierCause.RESOURCE_TERMINATED.value, passed=False
        )
        assert breaker_run_cost(entry, CAPS) == pytest.approx(0.002)

    def test_costless_termination_is_charged_at_the_per_run_ceiling_never_zero(self):
        # A timeout on a cost-less backend persists no cost; the breaker must
        # assume the worst a run is allowed to spend, never zero.
        entry = manifest_entry(
            cost=None, cause=VerifierCause.RESOURCE_TERMINATED.value, passed=False
        )
        charged = breaker_run_cost(entry, CAPS)
        assert charged == pytest.approx(CAPS.max_budget)
        assert charged > 0

    def test_missing_cost_on_a_passed_run_fails_loudly(self):
        with pytest.raises(ValueError, match="cost"):
            breaker_run_cost(manifest_entry(cost=None, passed=True), CAPS)

    def test_missing_cost_on_an_ordinary_failure_fails_loudly(self):
        entry = manifest_entry(cost=None, cause=VerifierCause.WRONG_VALUE.value, passed=False)
        with pytest.raises(ValueError, match="cost"):
            breaker_run_cost(entry, CAPS)


# ---------------------------------------------------------------------------
# The breaker: cumulative, idempotent per persisted run, trips strictly above
# ---------------------------------------------------------------------------


class TestCandidateSpendBreaker:
    def test_trips_only_strictly_above_the_candidate_budget(self):
        breaker = CandidateSpendBreaker(replace(CAPS, candidate_budget=0.002))
        breaker.charge(manifest_entry("inst-1__a01"))
        assert not breaker.tripped
        breaker.charge(manifest_entry("inst-2__a01"))
        assert breaker.spent == pytest.approx(0.002)
        assert not breaker.tripped  # spending exactly the budget is within it
        breaker.charge(manifest_entry("inst-3__a01"))
        assert breaker.tripped

    def test_charging_the_same_persisted_run_twice_counts_once(self):
        breaker = CandidateSpendBreaker(CAPS)
        entry = manifest_entry("inst-1__a01")
        breaker.charge(entry, namespace="round_01")
        breaker.charge(entry, namespace="round_01")
        assert breaker.spent == pytest.approx(COST_PER_CALL)

    def test_identical_run_ids_in_different_namespaces_both_charge(self):
        # Held-in and held-out splits may reuse instance ids; the namespace
        # (the round directory) keeps their runs distinct.
        breaker = CandidateSpendBreaker(CAPS)
        entry = manifest_entry("inst-1__a01")
        breaker.charge(entry, namespace="heldin")
        breaker.charge(entry, namespace="heldout")
        assert breaker.spent == pytest.approx(2 * COST_PER_CALL)


# ---------------------------------------------------------------------------
# The governed round: trip mid-split, persist the prefix, resume from disk
# ---------------------------------------------------------------------------


class TestGovernedRound:
    def test_breaker_trips_mid_split_and_skips_the_remaining_runs(self, tmp_path, monkeypatch):
        caps = replace(CAPS, candidate_budget=0.0015)
        factory = ClientFactory([final("RIGHT")] * 3)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, caps=caps)

        result = run_governed_round(config, CandidateSpendBreaker(caps))

        # Two runs completed and persisted (the prefix); the third never ran.
        assert factory.total_calls == 2
        assert [entry["run_id"] for entry in result.entries] == ["inst-1__a01", "inst-2__a01"]
        manifest = read_manifest(round_dir(tmp_path, 1))
        assert [entry["run_id"] for entry in manifest] == ["inst-1__a01", "inst-2__a01"]
        assert result.outcome == OUTCOME_OVER_BUDGET
        assert result.over_budget
        assert result.skipped_run_ids == ["inst-3__a01"]
        assert result.spent == pytest.approx(0.002)

    def test_resume_prices_persisted_runs_and_stays_over_budget_without_calls(
        self, tmp_path, monkeypatch
    ):
        caps = replace(CAPS, candidate_budget=0.0015)
        factory = ClientFactory([final("RIGHT")] * 3)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        run_governed_round(make_config(tmp_path, caps=caps), CandidateSpendBreaker(caps))

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        result = run_governed_round(make_config(tmp_path, caps=caps), CandidateSpendBreaker(caps))

        assert idle.total_calls == 0
        assert result.outcome == OUTCOME_OVER_BUDGET
        assert result.skipped_run_ids == ["inst-3__a01"]
        assert len(result.entries) == 2

    def test_under_budget_round_completes_with_nothing_skipped(self, tmp_path, monkeypatch):
        factory = ClientFactory([final("RIGHT"), final("WRONG"), final("RIGHT")])
        monkeypatch.setattr(rlm_module, "get_client", factory)

        result = run_governed_round(make_config(tmp_path), CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert not result.over_budget
        assert result.skipped_run_ids == []
        assert len(result.entries) == 3
        assert result.spent == pytest.approx(3 * COST_PER_CALL)

    def test_budget_terminated_run_persists_and_charges_its_spent_amount(
        self, tmp_path, monkeypatch
    ):
        # Run 1 burns two iterations into the per-run budget cap: the driver
        # persists the exception's spent figure (0.002) as the run's cost, and
        # the breaker charges it, tripping before run 2.
        caps = replace(CAPS, candidate_budget=0.0015)
        factory = ClientFactory(["Scanning part one.", "Scanning part two."])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, caps=caps, instances=make_instances(2))

        result = run_governed_round(config, CandidateSpendBreaker(caps))

        assert len(result.entries) == 1
        terminated = result.entries[0]
        assert terminated["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        assert terminated["cost"] == pytest.approx(2 * COST_PER_CALL)
        assert result.outcome == OUTCOME_OVER_BUDGET
        assert result.skipped_run_ids == ["inst-2__a01"]
        assert result.spent == pytest.approx(2 * COST_PER_CALL)


# ---------------------------------------------------------------------------
# The hard wall-clock backstop: a hung candidate call can never block the host
# ---------------------------------------------------------------------------


class TestHardDeadlineBackstop:
    def _tight_deadline(self, monkeypatch: pytest.MonkeyPatch) -> ValidationCaps:
        """Caps whose backstop fires in well under a second (0.05 * 1.5 + 0.5)."""
        monkeypatch.setattr(costs_module, "HARD_DEADLINE_GRACE_SECONDS", 0.5)
        return replace(CAPS, max_timeout=0.05)

    def test_deadline_derivation_and_opt_out(self):
        assert hard_deadline_seconds(None) is None
        assert hard_deadline_seconds(60.0) == pytest.approx(
            60.0 * costs_module.HARD_DEADLINE_FACTOR + costs_module.HARD_DEADLINE_GRACE_SECONDS
        )

    def test_hung_run_terminates_persists_charges_and_the_round_continues(
        self, tmp_path, monkeypatch
    ):
        # Run 1 hangs inside its live call; the backstop interrupts it, the
        # driver persists it as an ordinary RESOURCE_TERMINATED run, and runs
        # 2 and 3 proceed normally. The client counted the request before it
        # blocked, so the completion context has a real figure to publish and
        # the breaker charges that instead of the per-run ceiling.
        caps = self._tight_deadline(monkeypatch)
        factory = ClientFactory([HANG, final("RIGHT"), final("RIGHT")], client_cls=HangingScriptLM)
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_config(tmp_path, caps=caps)

        start = time.monotonic()
        result = run_governed_round(config, CandidateSpendBreaker(caps))
        assert time.monotonic() - start < 5.0  # returned instead of hanging

        assert result.outcome == OUTCOME_COMPLETED
        assert [entry["run_id"] for entry in result.entries] == [
            "inst-1__a01",
            "inst-2__a01",
            "inst-3__a01",
        ]
        hung = result.entries[0]
        assert hung["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        assert "HardDeadlineExceeded" in hung["verdict"]["detail"]
        assert "hard wall-clock deadline" in hung["verdict"]["detail"]
        assert hung["cost"] == pytest.approx(COST_PER_CALL)
        assert hung["usage_lower_bound"] is True
        assert not result.entries[1]["cause"] and not result.entries[2]["cause"]
        # The persist-first contract: the terminated run is on disk like any other.
        manifest = read_manifest(round_dir(tmp_path, 1))
        assert [entry["run_id"] for entry in manifest] == [
            entry["run_id"] for entry in result.entries
        ]
        assert result.skipped_run_ids == []
        assert result.spent == pytest.approx(3 * COST_PER_CALL)

    def test_hung_run_charge_trips_the_breaker_and_skips_the_rest(self, tmp_path, monkeypatch):
        # A run that hung before its request was ever recorded leaves the
        # runtime with nothing to publish, so the breaker still prices it at
        # the per-run ceiling (max_budget). That charge exceeds the candidate
        # budget: the remaining runs are skipped, exactly like an ordinary
        # budget termination.
        caps = replace(self._tight_deadline(monkeypatch), candidate_budget=0.001)
        factory = ClientFactory([HANG], client_cls=UnrecordedHangingScriptLM)
        monkeypatch.setattr(rlm_module, "get_client", factory)

        result = run_governed_round(make_config(tmp_path, caps=caps), CandidateSpendBreaker(caps))

        assert result.outcome == OUTCOME_OVER_BUDGET
        assert [entry["run_id"] for entry in result.entries] == ["inst-1__a01"]
        assert result.skipped_run_ids == ["inst-2__a01", "inst-3__a01"]
        assert result.spent == pytest.approx(caps.max_budget)

    def test_escaped_deadline_synthesizes_and_persists_the_in_flight_run(
        self, tmp_path, monkeypatch
    ):
        # Simulate the interrupt escaping run_round's per-run handler (e.g. a
        # hang during harness construction): every execution slice raises, and
        # run_governed_round must synthesize each in-flight run's terminated
        # manifest line itself, charge it, and keep the breaker in control.
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([]))
        real_run_round = costs_module.run_round

        def escaping_run_round(config, *, stop_after=None):
            if stop_after == 0:
                return real_run_round(config, stop_after=0)
            raise HardDeadlineExceeded(1.0)

        monkeypatch.setattr(costs_module, "run_round", escaping_run_round)
        # Two ceiling charges (0.003) cross the candidate budget; run 3 skips.
        caps = replace(CAPS, candidate_budget=0.002)

        result = run_governed_round(make_config(tmp_path, caps=caps), CandidateSpendBreaker(caps))

        assert result.outcome == OUTCOME_OVER_BUDGET
        assert [entry["run_id"] for entry in result.entries] == ["inst-1__a01", "inst-2__a01"]
        manifest = read_manifest(round_dir(tmp_path, 1))
        assert [entry["run_id"] for entry in manifest] == ["inst-1__a01", "inst-2__a01"]
        for entry in manifest:
            assert entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value
            assert "HardDeadlineExceeded" in entry["verdict"]["detail"]
            assert entry["cost"] is None
        assert result.skipped_run_ids == ["inst-3__a01"]
        assert result.spent == pytest.approx(2 * caps.max_budget)

    def test_alarm_state_is_restored_after_a_normal_round(self, tmp_path, monkeypatch):
        factory = ClientFactory([final("RIGHT")] * 3)
        monkeypatch.setattr(rlm_module, "get_client", factory)

        def sentinel(signum: int, frame: Any) -> None:  # pragma: no cover - never fired
            raise AssertionError("sentinel SIGALRM handler should never fire")

        previous = signal.signal(signal.SIGALRM, sentinel)
        try:
            result = run_governed_round(make_config(tmp_path), CandidateSpendBreaker(CAPS))
            assert result.outcome == OUTCOME_COMPLETED
            assert signal.getsignal(signal.SIGALRM) is sentinel
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous)

    def test_without_sigalrm_support_the_round_runs_unguarded(self, tmp_path, monkeypatch):
        # Platform guard: no SIGALRM (or not the main thread) means no
        # backstop, not a crash -- the round still runs to completion.
        monkeypatch.setattr(costs_module, "_alarm_available", lambda: False)
        factory = ClientFactory([final("RIGHT")] * 3)
        monkeypatch.setattr(rlm_module, "get_client", factory)

        result = run_governed_round(make_config(tmp_path), CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert len(result.entries) == 3


if __name__ == "__main__":
    pytest.main([__file__])


class TestSplitClaim:
    """R17: two invocations must not dispatch runs into one split directory.

    Interleaved manifest appends would tear lines and double-charge runs, and
    neither invocation would know the other existed. The claim is taken on
    every governed round, not only concurrent ones -- a second *sequential*
    invocation of the same split does exactly the same damage.
    """

    def test_a_split_claimed_by_this_live_process_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)
        round_path = round_dir(config.out_dir, config.round_index)
        round_path.mkdir(parents=True, exist_ok=True)

        with claim_split(round_path):
            with pytest.raises(SplitClaimedError) as excinfo:
                run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert str(round_path) in str(excinfo.value)
        assert str(os.getpid()) in str(excinfo.value)

    def test_a_claim_left_by_a_dead_pid_is_reclaimed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)
        round_path = round_dir(config.out_dir, config.round_index)
        claim_dir = round_path / ".claim"
        claim_dir.mkdir(parents=True)
        # A pid that cannot be alive: a crashed round must not lock its split
        # forever, or an operator's only recovery is deleting files by hand.
        (claim_dir / "pid").write_text("999999999\n")

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert not claim_dir.exists()

    def test_the_claim_is_released_after_a_normal_round(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)

        run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert not (round_dir(config.out_dir, config.round_index) / ".claim").exists()

    def test_the_claim_is_released_when_the_round_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)
        round_path = round_dir(config.out_dir, config.round_index)

        def explode(*args, **kwargs):
            raise RuntimeError("round blew up")

        monkeypatch.setattr(costs_module, "_run_slice", explode)
        with pytest.raises(RuntimeError, match="round blew up"):
            run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert not (round_path / ".claim").exists()

    def test_a_claim_that_never_named_an_owner_is_evicted(self, tmp_path, monkeypatch):
        """A claim directory with no owner inside can only be debris.

        Ownership is published by a single rename of a directory that already
        contains its pid, so a live claim always names someone. An empty one is
        either external tampering or a process killed mid-staging, and honouring
        it forever would lock the split with nobody holding it.
        """
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)
        round_path = round_dir(config.out_dir, config.round_index)
        (round_path / ".claim").mkdir(parents=True)

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert not (round_path / ".claim").exists()

    def test_release_leaves_a_claim_another_process_now_holds(self, tmp_path):
        """Eviction hands the split to someone else; the evicted process must
        not delete the new owner's claim on its way out."""
        round_path = tmp_path / "split"
        round_path.mkdir()

        with claim_split(round_path):
            # Someone else took over while we were running.
            (round_path / ".claim" / "pid").write_text("4242\n")

        assert (round_path / ".claim" / "pid").read_text().strip() == "4242"

    def test_acquisition_leaves_no_staging_debris(self, tmp_path):
        round_path = tmp_path / "split"
        round_path.mkdir()

        with claim_split(round_path):
            pass

        assert [child.name for child in round_path.iterdir()] == []

    def test_a_refused_claim_leaves_no_staging_debris(self, tmp_path):
        round_path = tmp_path / "split"
        round_path.mkdir()

        with claim_split(round_path):
            with pytest.raises(SplitClaimedError):
                with claim_split(round_path):
                    pass
            # Only the live claim itself remains; the refused attempt cleaned up.
            assert sorted(c.name for c in round_path.iterdir()) == [".claim"]

    def test_a_sequential_round_still_completes_under_the_claim(self, tmp_path, monkeypatch):
        """The guard is not gated on concurrency; the default path is unaffected."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        config = make_config(tmp_path)

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert len(result.entries) == 3
        assert result.skipped_run_ids == []


class TestChildErrorVerdict:
    """A child's trace crosses the process boundary but its verdict does not;
    the parent rebuilds it from the error string. A provider content-filter
    refusal must come back as CONTENT_FILTERED, not RESOURCE_TERMINATED."""

    def test_a_content_filter_refusal_keeps_its_label(self):
        detail = (
            "BadRequestError: Error code: 400 - {'error': {'code': 'content_filter', "
            "'message': \"Response content blocked by label 'MultiSeverity_ViolenceScore'.\"}}"
        )
        verdict = costs_module._error_verdict(detail, "partial")
        assert verdict.cause is VerifierCause.CONTENT_FILTERED
        assert verdict.passed is False
        assert verdict.produced == "partial"
        assert verdict.detail == detail

    def test_any_other_error_is_a_termination(self):
        verdict = costs_module._error_verdict("TimeoutExceededError: 1900.0s of 1800.0s", "")
        assert verdict.cause is VerifierCause.RESOURCE_TERMINATED
