"""Live recursion test (U4): does Kimi-K2.5 under ``H0*`` issue sub-calls?

Paid, opt-in, and bounded. Runs only when the azure live gate is open
(``AZURE_API_KEY``, ``AZURE_FOUNDRY_ENDPOINT``, ``SHRLM_RUN_LIVE=1``, and the
``SHRLM_VERIFIED_PRICING`` attestation matching the Kimi config); never in CI.
Worst-case spend is ``len(RECURSION_INSTANCE_IDS) x LIVE_MAX_BUDGET_USD``.

The assertion is deliberately the disjunction: at least one of the selected
~110k-char instances issued at least one sub-call. The addendum tells the model
to read directly when a regex over ``context`` would pin the answer, so a
faithful model may still solve a graph in-REPL; a zero result on every instance
is a finding to report, not a threshold to loosen. The per-instance table is
printed whether the assertion passes or fails.

Run with ``-s`` to see the table::

    SHRLM_RUN_LIVE=1 uv run pytest -q tests/experiment/test_recursion_live.py -s
"""

import copy
from pathlib import Path
from typing import Any

import pytest

from shrlm.environments.graphwalks import GraphWalksVerifier
from shrlm.experiment.config import backend_kwargs_for, load_config
from shrlm.experiment.live_gates import live_skip_reason
from shrlm.optimization.driver import RoundConfig, build_round_rlm, execute_run
from shrlm.optimization.walker import walk
from shrlm.rlm_harness import H0_STAR
from shrlm.runner import run_metrics
from tests.experiment.recursion_instances import RECURSION_INSTANCE_IDS, load_recursion_instances

KIMI_CONFIG = Path("configs/experiment_kimiK25.toml")

# Per-run limits. The budget is the smoke live ceiling; the timeout allows a
# 110k-char graph plus a handful of sub-calls under thinking mode.
LIVE_MAX_BUDGET_USD = 0.20
LIVE_MAX_TIMEOUT_SECONDS = 600.0


def _skip_reason() -> str | None:
    # The attestation must be checked against THIS config's rate card, not the
    # shipped smoke profile's -- they differ (0.6/3.0 vs 0.1/0.3).
    return live_skip_reason(
        runner_backend="azure_foundry", config=load_config("full", path=KIMI_CONFIG)
    )


_LIVE_SKIP = _skip_reason()


def _live_round_config(out_dir: Path) -> RoundConfig:
    """H0* plus the Kimi config's real runner kwargs, under tight per-run caps."""
    config = load_config("full", path=KIMI_CONFIG)
    assert config.loop.initial_harness == "H0*"
    return RoundConfig(
        round_index=1,
        harness=H0_STAR,
        instances=load_recursion_instances(config),
        verifier=GraphWalksVerifier(),
        out_dir=out_dir,
        backend="azure_foundry",
        backend_kwargs=copy.deepcopy(backend_kwargs_for(config, "runner")),
        attempts=1,
        max_iterations=config.caps.max_iterations,
        max_budget=LIVE_MAX_BUDGET_USD,
        max_timeout=LIVE_MAX_TIMEOUT_SECONDS,
    )


def _sub_call_counts(completion: Any) -> tuple[int, int]:
    """Sub-calls counted two ways: per-turn trace metrics, and the rebuilt tree."""
    from_metrics = int(run_metrics(completion)["sub_call_count"])
    _root, stats = walk(completion)
    from_tree = int(stats.n_rlm_children + stats.n_llm_leaves)
    return from_metrics, from_tree


@pytest.fixture(scope="module")
def recursion_rows(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Run every selected instance once and collect one row each (paid)."""
    config = _live_round_config(tmp_path_factory.mktemp("recursion_live"))
    harnessed = build_round_rlm(config)
    rows: list[dict[str, Any]] = []
    for instance in config.instances:
        outcome = execute_run(
            harnessed,
            instance,
            model_name=config.backend_kwargs["model_name"],
            verifier=config.verifier,
        )
        completion = outcome.completion
        from_metrics, from_tree = _sub_call_counts(completion)
        verdict = outcome.verdict
        rows.append(
            {
                "instance_id": instance["id"],
                "prompt_chars": len(instance["prompt"]),
                "sub_calls_metrics": from_metrics,
                "sub_calls_tree": from_tree,
                "iterations": len(completion.metadata["iterations"]),
                "passed": bool(verdict.passed) if verdict else None,
                "cause": (verdict.cause.value if verdict and verdict.cause else "pass"),
                "cost_usd": completion.usage_summary.total_cost,
                "usage_lower_bound": outcome.usage_lower_bound,
            }
        )
    _print_table(rows)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'instance':<28} {'chars':>7} {'subcalls':>8} {'tree':>5} {'iters':>5} "
        f"{'cause':<20} {'cost':>7} lb"
    )
    print("\n" + header)
    for row in rows:
        print(
            f"{row['instance_id']:<28} {row['prompt_chars']:>7} {row['sub_calls_metrics']:>8} "
            f"{row['sub_calls_tree']:>5} {row['iterations']:>5} {row['cause']:<20} "
            f"{row['cost_usd']:>7.4f} {'*' if row['usage_lower_bound'] else ''}"
        )


@pytest.mark.skipif(_LIVE_SKIP is not None, reason=_LIVE_SKIP or "live gates satisfied")
class TestH0StarRecursesLive:
    def test_every_selected_instance_produced_a_row(self, recursion_rows):
        assert [row["instance_id"] for row in recursion_rows] == list(RECURSION_INSTANCE_IDS)
        for row in recursion_rows:
            assert row["cost_usd"] is None or row["cost_usd"] <= LIVE_MAX_BUDGET_USD * 1.05

    def test_trace_metrics_and_rebuilt_tree_agree_on_sub_calls(self, recursion_rows):
        for row in recursion_rows:
            assert row["sub_calls_metrics"] == row["sub_calls_tree"], row

    def test_at_least_one_instance_issued_a_sub_call(self, recursion_rows):
        # KTD5: the disjunction. A total of zero is the plan's stop condition
        # and is reported through the printed table, not hidden.
        total = sum(row["sub_calls_metrics"] for row in recursion_rows)
        assert total >= 1, (
            "H0* issued zero sub-calls on every selected instance; "
            "see the table above and POST_MORTEM.md follow-ups (S6 per-prompt cap)"
        )
