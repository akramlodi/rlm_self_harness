"""Tier 1 of the U8 smoke: the whole scaffold end to end for $0, in CI (R13).

One invocation of the real ``run_experiment`` + ``run_evaluation`` +
``build_report`` chain, on the shipped ``[smoke]`` config profile (R2), over
BOTH environments at BOTH lengths -- driven entirely offline:

* Splits are the checked-in fixtures under ``fixtures/smoke_splits``,
  pre-materialized into the experiment directory with a matching manifest, so
  ``materialize_splits`` verifies hashes and no loader ever runs. The loaders
  handed to both entry points raise if called, which is the assertion.
* The runtime seam is the repo-standard ``rlm.core.rlm.get_client``
  monkeypatch with the cost-stubbed scripted client from
  ``tests.optimization.test_driver`` -- a plain ``MockLM`` persists no cost and
  the governed (breaker-wrapped) mining and eval stages refuse cost-less runs.
* Attributor and proposer are injected ``MockLM``s.
* Zero network access is enforced, not assumed: ``socket.socket.connect`` is
  wrapped to allow only loopback (the RLM runtime's own LM handler talks to
  ``LocalREPL`` over a local TCP socket) and to raise on anything else, and the
  two dataset fetch seams raise on contact.

What it asserts is wiring, never statistics (KTD9): the directory contract, a
``stage_usage.jsonl`` carrying every stage with nonzero synthetic tokens, a
report that renders and projects, and -- statically, without network -- that
the tier-2 live budgets stay under the $5 ceiling and that tier 2 declines to
spend anything without its flag and key.
"""

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from examples import experiment_smoke
from shrlm.experiment.config import ExperimentConfig, load_config
from shrlm.experiment.evaluation import (
    DEFAULT_CONDITIONS,
    EVAL_DIR,
    EVAL_SUMMARY_FILENAME,
    run_evaluation,
)
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    FROZEN_DIR,
    FROZEN_HARNESS_FILENAME,
    PROPOSALS_MARKER_FILENAME,
    ROUND_MARKER_FILENAME,
    experiment_round_dir,
    run_experiment,
)
from shrlm.experiment.report import REPORT_FILENAME, build_report, render_markdown, write_report
from shrlm.experiment.splits import (
    MANIFEST_FILE,
    SPLITS_DIR,
    LoaderFn,
    dataset_revision_for,
    split_file_name,
)
from shrlm.experiment.usage import STAGE_USAGE_FILE, read_stage_usage
from tests.experiment.test_orchestrator import attribution, proposer_batch
from tests.mock_lm import MockLM
from tests.optimization.test_driver import ClientFactory, final

FIXTURES = Path(__file__).parent / "fixtures" / "smoke_splits"

# The split files the smoke profile's plan calls for, per environment.
FIXTURE_ROLES: dict[str, dict[str, tuple[str, ...]]] = {
    "graphwalks": {"short": ("held_in", "held_out", "test"), "long": ("test",)},
    "oolong_pairs": {"short": ("test",), "long": ("test",)},
}

# The evaluation grid, in the order the runner executes it: environments
# sorted, lengths short then long, conditions in DEFAULT_CONDITIONS order.
EVAL_SET_ORDER: tuple[tuple[str, str], ...] = (
    ("graphwalks", "short"),
    ("graphwalks", "long"),
    ("oolong_pairs", "short"),
    ("oolong_pairs", "long"),
)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

STAGES = ("mining", "attribution", "proposal", "validation", "eval")

# Two distinct failure mechanisms across the three held-in mining runs give two
# patterns, hence two candidates (the smoke profile's k=2) on two different
# surfaces, hence a merged re-evaluation -- the widest round the profile allows.
MECHANISMS = ("incomplete_coverage", "skipped_verification", "skipped_verification")
MERGE_TEXT = "Cover every input chunk and verify before answering. [smoke]"

WRONG_ANSWER = "Final Answer: [not-a-node]"


# ---------------------------------------------------------------------------
# Offline scaffolding: fixture splits, refusing loaders, network guard
# ---------------------------------------------------------------------------


def fixture_instances(environment: str, length: str, role: str) -> list[dict[str, Any]]:
    path = FIXTURES / split_file_name(environment, length, role)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold_answer(instance: dict[str, Any]) -> str:
    """The exact answer the environment's real verifier passes for this instance."""
    if "answer_nodes" in instance:
        return "Final Answer: [" + ", ".join(instance["answer_nodes"]) + "]"
    pairs = ", ".join(f"({a}, {b})" for a, b in instance["gold_pairs"])
    return f"Final Answer: {pairs}"


def pre_materialize_splits(config: ExperimentConfig, out_dir: Path) -> None:
    """Copy the fixture splits in and write the manifest ``materialize_splits`` verifies.

    Written here rather than checked in so the manifest always carries the
    configured dataset revision pins: an edit to a pin in
    ``configs/experiment.toml`` is a live-loader concern, not a reason for the
    offline smoke to go red.
    """
    splits_dir = out_dir / SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)
    environments: dict[str, Any] = {}
    for environment, lengths in FIXTURE_ROLES.items():
        files: dict[str, Any] = {}
        for length, roles in lengths.items():
            for role in roles:
                file_name = split_file_name(environment, length, role)
                content = (FIXTURES / file_name).read_bytes()
                (splits_dir / file_name).write_bytes(content)
                files[file_name] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "count": len(content.decode().splitlines()),
                }
        environments[environment] = {
            "revision": dataset_revision_for(config, environment),
            "files": files,
        }
    manifest = {"sample_seed": config.splits.seed, "environments": environments}
    (splits_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def refusing_loader(environment: str) -> LoaderFn:
    """A loader that must never run: the splits are already on disk."""

    def load(config: ExperimentConfig, length: str, limit: int, seed: int) -> list[dict[str, Any]]:
        raise AssertionError(
            f"the {environment!r} loader ran for the {length} split; the mock smoke reads "
            "pre-materialized fixture splits and must never touch a dataset"
        )

    return load


LOADERS: dict[str, LoaderFn] = {
    environment: refusing_loader(environment) for environment in FIXTURE_ROLES
}


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow loopback (the runtime's LM handler socket); refuse everything else."""
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK_HOSTS:
            raise AssertionError(f"the mock smoke attempted a network connection to {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    def refuse_fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the mock smoke attempted a dataset download")

    monkeypatch.setattr("shrlm.environments.graphwalks.fetch_rows", refuse_fetch)
    monkeypatch.setattr("shrlm.environments.oolong_pairs.iter_dataset_rows", refuse_fetch)


# ---------------------------------------------------------------------------
# The scripted round: what every run answers, in execution order
# ---------------------------------------------------------------------------


def optimization_script() -> list[str]:
    """Mining (all failing) then validation: baseline fails, every subject passes.

    Validation order is baseline, candidate 1, candidate 2, merged; each
    subject runs held-in then held-out at v=1. Pass/fail here is scripted only
    to drive the round to its widest shape -- the smoke asserts no outcome.
    """
    held_in = fixture_instances("graphwalks", "short", "held_in")
    held_out = fixture_instances("graphwalks", "short", "held_out")
    subject_instances = held_in + held_out

    script = [final(WRONG_ANSWER) for _ in held_in]  # mining: 3 failures
    script += [final(WRONG_ANSWER) for _ in subject_instances]  # baseline
    for _ in range(3):  # candidate 1, candidate 2, merged
        script += [final(gold_answer(instance)) for instance in subject_instances]
    return script


def evaluation_script() -> list[str]:
    """Every condition x test set x instance, answered correctly, in order."""
    per_condition = [
        final(gold_answer(instance))
        for environment, length in EVAL_SET_ORDER
        for instance in fixture_instances(environment, length, "test")
    ]
    return per_condition * len(DEFAULT_CONDITIONS)


def n_optimization_runs() -> int:
    return len(optimization_script())


def n_evaluation_runs() -> int:
    return len(evaluation_script())


# ---------------------------------------------------------------------------
# One shared smoke run: the whole pipeline, executed once for the module
# ---------------------------------------------------------------------------


class SmokeRun:
    """What one full mock smoke produced, for the assertions below to read."""

    def __init__(self, config: ExperimentConfig, out_dir: Path, factory: ClientFactory) -> None:
        self.config = config
        self.out_dir = out_dir
        self.factory = factory


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> SmokeRun:
    """Run the smoke profile end to end once, offline, and hand back the artifacts."""
    import rlm.core.rlm as rlm_module

    out_dir = tmp_path_factory.mktemp("smoke") / "exp"
    config = load_config("smoke")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key-never-used")
        block_network(monkeypatch)
        factory = ClientFactory(optimization_script() + evaluation_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)

        pre_materialize_splits(config, out_dir)
        run_experiment(
            config,
            out_dir,
            attributor_lm=MockLM(responses=[attribution(mechanism) for mechanism in MECHANISMS]),
            proposer_lm=MockLM(responses=[proposer_batch((0, MERGE_TEXT), (1, MERGE_TEXT))]),
            loaders=LOADERS,
        )
        run_evaluation(config, DEFAULT_CONDITIONS, out_dir, loaders=LOADERS)
    return SmokeRun(config, out_dir, factory)


# ---------------------------------------------------------------------------
# The pipeline completed, offline, at the scripted scale
# ---------------------------------------------------------------------------


class TestSmokePipeline:
    def test_every_scripted_run_executed_and_nothing_else_did(self, smoke: SmokeRun):
        assert smoke.factory.total_calls == n_optimization_runs() + n_evaluation_runs()
        assert smoke.factory.script == []  # the script is exactly consumed

    def test_directory_contract(self, smoke: SmokeRun):
        out = smoke.out_dir
        assert (out / CONFIG_FILENAME).exists()
        assert (out / SPLITS_DIR / MANIFEST_FILE).exists()
        assert (out / STAGE_USAGE_FILE).exists()
        assert (out / FROZEN_DIR / FROZEN_HARNESS_FILENAME).exists()

        round_path = experiment_round_dir(out, 1)
        assert (round_path / "mining" / "round_01" / "bundle.json").exists()
        assert (round_path / "mining" / "round_01" / "runs.jsonl").exists()
        assert (round_path / PROPOSALS_MARKER_FILENAME).exists()
        assert (round_path / "validation" / "round_01" / "decision.json").exists()
        assert (round_path / ROUND_MARKER_FILENAME).exists()

        assert (out / EVAL_DIR / EVAL_SUMMARY_FILENAME).exists()
        for condition in DEFAULT_CONDITIONS:
            for environment, length in EVAL_SET_ORDER:
                path = out / EVAL_DIR / condition / f"{environment}_{length}" / "round_00"
                assert (path / "runs.jsonl").exists(), path
                assert (path / "instances.jsonl").exists(), path

    def test_both_environments_at_both_lengths_were_evaluated(self, smoke: SmokeRun):
        summary = json.loads((smoke.out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_text())
        assert set(summary["test_sets"]) == {
            f"{environment}_{length}" for environment, length in EVAL_SET_ORDER
        }
        for condition in DEFAULT_CONDITIONS:
            aggregates = summary["conditions"][condition]["test_sets"]
            assert {
                (aggregate["environment"], aggregate["length"]) for aggregate in aggregates.values()
            } == set(EVAL_SET_ORDER)
            assert all(aggregate["n_runs"] > 0 for aggregate in aggregates.values())

    def test_stage_usage_carries_every_stage_with_nonzero_tokens(self, smoke: SmokeRun):
        totals = read_stage_usage(smoke.out_dir / STAGE_USAGE_FILE)
        records = [
            json.loads(line)
            for line in (smoke.out_dir / STAGE_USAGE_FILE).read_text().splitlines()
            if line.strip()
        ]
        stage_of = {record["stage_work_id"]: record["stage"] for record in records}
        assert set(stage_of.values()) == set(STAGES)
        for work_id, usage in totals.items():
            assert usage.input_tokens > 0, work_id
            assert usage.output_tokens > 0, work_id
            assert usage.wall_seconds > 0.0, work_id
            assert not usage.lower_bound, work_id


# ---------------------------------------------------------------------------
# The report renders over the smoke output (U7 on U8's artifacts)
# ---------------------------------------------------------------------------


class TestSmokeReport:
    def test_report_covers_every_stage_and_projects_the_full_experiment(self, smoke: SmokeRun):
        report = build_report(smoke.config, smoke.out_dir)

        assert {stage.stage for stage in report.stages} == set(STAGES)
        for stage in report.stages:
            assert stage.input_tokens > 0, stage.stage
            assert stage.output_tokens > 0, stage.stage
        assert {(bucket.environment, bucket.length) for bucket in report.buckets} >= set(
            EVAL_SET_ORDER
        )
        assert report.point.long_measured
        assert report.point.total_tokens > 0.0
        assert report.scenarios and all(scenario.usd_point > 0.0 for scenario in report.scenarios)

    def test_report_renders_and_persists(self, smoke: SmokeRun):
        report = build_report(smoke.config, smoke.out_dir)
        path = write_report(report)
        assert path == smoke.out_dir / REPORT_FILENAME
        assert json.loads(path.read_text())["profile"] == "smoke"

        markdown = render_markdown(report)
        for stage in STAGES:
            assert stage in markdown


# ---------------------------------------------------------------------------
# Tier 2, checked without spending: budget arithmetic and the decline path
# ---------------------------------------------------------------------------


class TestLiveSmokeStructuralChecks:
    """Tier 2's own artifact checks, run over tier 1's artifacts.

    The live smoke is never executed here, but the structural assertions it
    makes afterwards read persisted bytes only -- so running them against the
    mock experiment directory proves they address real keys and paths before
    anyone spends money on discovering a typo.
    """

    def test_directory_contract_and_costs_and_sampling_args_check_out(self, smoke: SmokeRun):
        experiment_smoke.check_directory_contract(smoke.out_dir)
        assert experiment_smoke.check_costs_present(smoke.out_dir) == (
            n_optimization_runs() + n_evaluation_runs()
        )
        persisted = experiment_smoke.check_sampling_args(smoke.out_dir, smoke.config)
        assert persisted["temperature"] == smoke.config.decoding.temperature
        assert persisted["extra_body"]["top_k"] == smoke.config.decoding.top_k

    def test_long_coverage_and_measured_spend_read_the_smoke_artifacts(self, smoke: SmokeRun):
        coverage = experiment_smoke.long_run_coverage(smoke.out_dir)
        assert set(coverage) == {"graphwalks", "oolong_pairs"}
        for environment, counts in coverage.items():
            assert counts["runs"] > 0, environment
            assert counts["uncapped"] == counts["runs"], environment
        assert experiment_smoke.measured_spend(smoke.out_dir) > 0.0

    def test_a_cost_less_run_is_refused(self, tmp_path):
        round_path = tmp_path / "opt" / "round_01" / "mining" / "round_01"
        round_path.mkdir(parents=True)
        (round_path / "runs.jsonl").write_text(
            json.dumps({"run_id": "inst__a01", "cost": None}) + "\n"
        )
        with pytest.raises(experiment_smoke.SmokeError, match="no cost"):
            experiment_smoke.check_costs_present(tmp_path)

    def test_a_missing_artifact_is_named(self, tmp_path):
        with pytest.raises(experiment_smoke.SmokeError, match=CONFIG_FILENAME):
            experiment_smoke.check_directory_contract(tmp_path)


class TestLiveSmokeGuards:
    def test_configured_live_budgets_stay_under_the_five_dollar_ceiling(self):
        config = experiment_smoke.live_config()

        # 7 breakers x ($0.50 cumulative + $0.20 per-run overshoot) = $4.90.
        assert experiment_smoke.breaker_count(config) == 7
        assert experiment_smoke.spend_ceiling(config) == pytest.approx(4.90)
        assert experiment_smoke.spend_ceiling(config) < experiment_smoke.SPEND_CEILING_USD
        assert experiment_smoke.check_budget_arithmetic(config) == pytest.approx(4.90)

    def test_live_profile_keeps_the_smoke_scale_and_semantics(self):
        config = experiment_smoke.live_config()
        smoke_profile = load_config("smoke")

        assert config.profile == "smoke"
        assert config.loop == smoke_profile.loop
        assert config.splits == smoke_profile.splits
        assert config.decoding == smoke_profile.decoding
        assert config.promotion == smoke_profile.promotion
        assert config.caps.max_budget == experiment_smoke.LIVE_MAX_BUDGET_USD
        assert config.caps.candidate_budget == experiment_smoke.LIVE_CANDIDATE_BUDGET_USD

    def test_a_budget_over_the_ceiling_is_refused(self):
        from dataclasses import replace

        config = experiment_smoke.live_config()
        reckless = replace(config, caps=replace(config.caps, candidate_budget=5.0))
        with pytest.raises(experiment_smoke.SmokeError, match="ceiling"):
            experiment_smoke.check_budget_arithmetic(reckless)

    def test_declines_without_the_live_flag(self, capsys, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "present-but-unused")
        block_network(monkeypatch)

        assert experiment_smoke.main([]) == 1
        assert "Nothing was spent" in capsys.readouterr().out

    def test_declines_without_the_api_key(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        block_network(monkeypatch)

        assert experiment_smoke.main(["--live"]) == 1
        out = capsys.readouterr().out
        assert "OPENROUTER_API_KEY" in out
        assert "Nothing was spent" in out
