"""Live recursion smoke test: does the runner issue sub-calls on long OOLONG-synth?

Paid, opt-in, and bounded -- the OOLONG analogue of
``tests/experiment/test_recursion_live.py``. Runs only when the azure live gate
is open (``AZURE_API_KEY``, ``AZURE_FOUNDRY_ENDPOINT``, ``SHRLM_RUN_LIVE=1``,
and the ``SHRLM_VERIFIED_PRICING`` attestation matching the OOLONG config);
never in CI. Worst-case spend is ``n_instances x LIVE_MAX_BUDGET_USD``.

The point: before trusting the full loop to OOLONG-synth, confirm the runtime
actually decomposes on the long instances -- ``mean_sub_calls > 0``. OOLONG's
aggregation questions over a 64K-128K token context cannot be answered by a
single in-REPL scan, so an ``H0*`` root has a real reason to delegate; a zero
result on every instance is a finding to report (see POST_MORTEM.md), not a
threshold to loosen. The per-instance table prints whether the assertion passes.

    SHRLM_RUN_LIVE=1 uv run pytest -q tests/experiment/test_oolong_recursion_live.py -s
"""

import copy
from pathlib import Path
from typing import Any

import pytest

from shrlm.environments.oolong import OolongVerifier
from shrlm.experiment.config import backend_kwargs_for, load_config
from shrlm.experiment.live_gates import CONFIG_ENV_KEY, live_config_path, live_skip_reason
from shrlm.optimization.driver import RoundConfig, build_round_rlm, execute_run
from shrlm.optimization.walker import walk
from shrlm.rlm_harness import HARNESSES
from shrlm.runner import run_metrics
from tests.experiment.oolong_recursion_instances import load_oolong_recursion_instances

OOLONG_CONFIG = Path("configs/experiment_oolong.toml")

N_INSTANCES = 4
LIVE_MAX_BUDGET_USD = 0.40
LIVE_MAX_TIMEOUT_SECONDS = 900.0


def _config() -> Any:
    """The env-selected experiment config (KTD7); the OOLONG Kimi TOML by
    default -- ``SHRLM_EXPERIMENT_CONFIG=configs/experiment_oolong_gptoss.toml``
    points this tier at the gpt-oss deployment."""
    return load_config("full", path=live_config_path(OOLONG_CONFIG))


def _skip_reason() -> str | None:
    # Checked against the SELECTED config's runner backend and rate card.
    try:
        config = _config()
    except Exception as exc:  # bad SHRLM_EXPERIMENT_CONFIG must skip, not error collection
        return f"failed to load config selected via {CONFIG_ENV_KEY}: {exc!r}"
    return live_skip_reason(config=config)


_LIVE_SKIP = _skip_reason()


def _round_config(out_dir: Path, instances: list[dict[str, Any]]) -> RoundConfig:
    config = _config()
    return RoundConfig(
        round_index=1,
        harness=HARNESSES[config.loop.initial_harness],
        instances=instances,
        verifier=OolongVerifier(task_set="synth"),
        out_dir=out_dir,
        backend=config.backends.runner.backend,
        backend_kwargs=copy.deepcopy(backend_kwargs_for(config, "runner")),
        attempts=1,
        max_iterations=config.caps.max_iterations,
        max_budget=LIVE_MAX_BUDGET_USD,
        max_timeout=LIVE_MAX_TIMEOUT_SECONDS,
    )


def _sub_call_counts(completion: Any) -> tuple[int, int]:
    from_metrics = int(run_metrics(completion)["sub_call_count"])
    _root, stats = walk(completion)
    return from_metrics, int(stats.n_rlm_children + stats.n_llm_leaves)


@pytest.fixture(scope="module")
def recursion_rows(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    config = _config()
    instances = load_oolong_recursion_instances(config, n=N_INSTANCES)
    round_config = _round_config(tmp_path_factory.mktemp("oolong_recursion"), instances)
    harnessed = build_round_rlm(round_config)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        outcome = execute_run(
            harnessed,
            instance,
            model_name=round_config.backend_kwargs["model_name"],
            verifier=round_config.verifier,
        )
        completion = outcome.completion
        from_metrics, from_tree = _sub_call_counts(completion)
        verdict = outcome.verdict
        rows.append(
            {
                "instance_id": instance["id"],
                "context_len": instance["context_len"],
                "prompt_chars": len(instance["prompt"]),
                "sub_calls_metrics": from_metrics,
                "sub_calls_tree": from_tree,
                "iterations": len(completion.metadata["iterations"]),
                "passed": bool(verdict.passed) if verdict else None,
                "cause": (verdict.cause.value if verdict and verdict.cause else "pass"),
                "cost_usd": completion.usage_summary.total_cost,
                "input_tokens": completion.usage_summary.total_input_tokens,
                "output_tokens": completion.usage_summary.total_output_tokens,
                "wall_s": completion.execution_time,
            }
        )
    _print_table(rows)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'instance':<44} {'ctxlen':>7} {'chars':>8} {'subcalls':>8} {'tree':>5} "
        f"{'iters':>5} {'cause':<16} {'cost':>7} {'in_tok':>8} {'out_tok':>8} {'wall_s':>7}"
    )
    print("\n" + header)
    for row in rows:
        print(
            f"{row['instance_id']:<44} {row['context_len']:>7} {row['prompt_chars']:>8} "
            f"{row['sub_calls_metrics']:>8} {row['sub_calls_tree']:>5} {row['iterations']:>5} "
            f"{row['cause']:<16} {row['cost_usd'] or 0.0:>7.4f} {row['input_tokens']:>8} "
            f"{row['output_tokens']:>8} {row['wall_s']:>7.1f}"
        )


@pytest.mark.live
@pytest.mark.skipif(_LIVE_SKIP is not None, reason=_LIVE_SKIP or "live gates satisfied")
class TestOolongRecursesLive:
    def test_every_selected_instance_produced_a_row(self, recursion_rows):
        assert len(recursion_rows) == N_INSTANCES
        for row in recursion_rows:
            assert row["cost_usd"] is None or row["cost_usd"] <= LIVE_MAX_BUDGET_USD * 1.05

    def test_trace_metrics_and_rebuilt_tree_agree_on_sub_calls(self, recursion_rows):
        for row in recursion_rows:
            assert row["sub_calls_metrics"] == row["sub_calls_tree"], row

    def test_mean_sub_calls_is_positive(self, recursion_rows):
        total = sum(row["sub_calls_metrics"] for row in recursion_rows)
        assert total >= 1, (
            "H0* issued zero sub-calls on every selected long OOLONG-synth instance; "
            "the aggregation task did not trigger decomposition -- see the table above "
            "before trusting the full loop to this environment."
        )
