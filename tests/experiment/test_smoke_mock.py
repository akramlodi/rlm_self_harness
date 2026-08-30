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
the tier-2 live budgets stay under the $5 ceiling (governed runs and
ungoverned probe/attribution/proposal calls alike), that tier 2 fails rather
than passes when an environment lands no uncapped long run, and that tier 2
declines to spend anything without its flag and key.
"""

import hashlib
import json
import re
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from examples import experiment_smoke
from examples import run_experiment as run_experiment_cli
from shrlm.experiment.config import CONFIG_PATH, ExperimentConfig, load_config
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

# Sentinel credential for the openrouter backend the shipped config selects.
# The driver's fail-fast demands the variable before any run, and the hygiene
# test below asserts the VALUE never reaches a persisted byte.
OPENROUTER_KEY_SENTINEL = "mock-openrouter-key-sentinel-51c9"


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
        monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY_SENTINEL)
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

    def test_no_credential_sentinel_reaches_any_persisted_byte(self, smoke: SmokeRun):
        """R3: credentials come from the environment only. The whole pipeline
        ran with a sentinel value in OPENROUTER_API_KEY; no file the
        experiment persisted may contain it."""
        scanned = 0
        for path in sorted(smoke.out_dir.rglob("*")):
            if not path.is_file():
                continue
            scanned += 1
            content = path.read_text(errors="replace")
            assert OPENROUTER_KEY_SENTINEL not in content, path
        assert scanned > 10  # config, splits, manifests, traces, usage, eval


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


SummaryTransform = Callable[[dict[str, Any]], None]


def doctored_experiment(smoke: SmokeRun, tmp_path: Path, transform: SummaryTransform) -> Path:
    """A copy of the mock experiment whose eval summary ``transform`` rewrote.

    The tier-2 checks read persisted bytes only, so a doctored copy is how the
    live smoke's failure paths are exercised without a live run. The report is
    removed so a later assertion that it exists means *this* invocation wrote
    it.
    """
    out_dir = tmp_path / "exp"
    shutil.copytree(smoke.out_dir, out_dir)
    summary_path = out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    transform(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / REPORT_FILENAME).unlink(missing_ok=True)
    return out_dir


def cap_long_runs(environment: str) -> SummaryTransform:
    """Flag every one of an environment's long aggregates as a lower bound."""

    def transform(summary: dict[str, Any]) -> None:
        for condition in summary["conditions"].values():
            for aggregate in condition["test_sets"].values():
                if aggregate["environment"] == environment and aggregate["length"] == "long":
                    aggregate["usage_lower_bound"] = True

    return transform


def drop_long_sets(environment: str) -> SummaryTransform:
    """Remove an environment's long aggregates, as a tripped breaker would."""

    def transform(summary: dict[str, Any]) -> None:
        for condition in summary["conditions"].values():
            for set_id in [
                set_id
                for set_id, aggregate in condition["test_sets"].items()
                if aggregate["environment"] == environment and aggregate["length"] == "long"
            ]:
                del condition["test_sets"][set_id]

    return transform


def stub_live_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the three spending stages so ``run_live`` runs over the copy.

    Everything downstream of them -- the structural checks, the coverage read,
    the report, the spend accounting, the KTD6 gate -- runs for real against
    the persisted artifacts, which is the ordering under test.
    """
    monkeypatch.setattr(
        experiment_smoke,
        "probe",
        lambda config: {
            "provider": None,
            "usage_cost": None,
            "client_cost": 0.000001,
            "cost_source": "synthesized",
            "input_tokens": 1,
            "output_tokens": 1,
            "sampling_args": {},
        },
    )
    monkeypatch.setattr(
        experiment_smoke,
        "run_experiment",
        lambda config, out_dir: SimpleNamespace(stopped="max_rounds", final_harness_hash="stub"),
    )
    monkeypatch.setattr(
        experiment_smoke,
        "run_evaluation",
        lambda config, conditions, out_dir: SimpleNamespace(conditions=[]),
    )


class TestLiveSmokeStructuralChecks:
    """Tier 2's own artifact checks, run over tier 1's artifacts.

    The live smoke is never executed here, but the structural assertions it
    makes afterwards read persisted bytes only -- so running them against the
    mock experiment directory proves they address real keys and paths before
    anyone spends money on discovering a typo.
    """

    def test_directory_contract_and_costs_and_sampling_args_check_out(self, smoke: SmokeRun):
        experiment_smoke.check_directory_contract(smoke.out_dir)
        # No run is terminated in the mock tier, so every run carries a cost.
        assert experiment_smoke.check_costs_present(smoke.out_dir) == (
            n_optimization_runs() + n_evaluation_runs(),
            0,
        )
        persisted = experiment_smoke.check_sampling_args(smoke.out_dir, smoke.config)
        assert persisted["temperature"] == smoke.config.decoding.temperature
        assert persisted["top_p"] == smoke.config.decoding.top_p
        # The Qwen knobs ride in extra_body; no azure-only chat_template_kwargs
        # ride an openrouter role.
        assert persisted["extra_body"]["top_k"] == smoke.config.decoding.top_k
        assert persisted["extra_body"]["min_p"] == smoke.config.decoding.min_p
        assert "chat_template_kwargs" not in persisted["extra_body"]

    def test_stage_coverage_check_reads_the_smoke_artifacts(self, smoke: SmokeRun):
        counts = experiment_smoke.check_stage_coverage(smoke.out_dir)
        assert set(counts) == set(experiment_smoke.REQUIRED_STAGES)
        assert all(count > 0 for count in counts.values())

    def test_stage_coverage_names_a_missing_stage(self, smoke: SmokeRun, tmp_path):
        out_dir = tmp_path / "exp"
        shutil.copytree(smoke.out_dir, out_dir)
        usage_path = out_dir / STAGE_USAGE_FILE
        kept = [
            line
            for line in usage_path.read_text().splitlines()
            if line.strip() and json.loads(line)["stage"] != "proposal"
        ]
        usage_path.write_text("".join(line + "\n" for line in kept))
        with pytest.raises(experiment_smoke.SmokeError, match="proposal"):
            experiment_smoke.check_stage_coverage(out_dir)

    def test_stage_coverage_tolerates_an_individual_zero_token_work_item(
        self, smoke: SmokeRun, tmp_path
    ):
        """A 0/0 work id is legitimately reachable (a test set skipped after a
        shared breaker tripped; an all-RESOURCE_TERMINATED work id rebuilds
        zero counts) and must not fail a stage that also shows positive-token
        evidence of hitting the backend."""
        out_dir = tmp_path / "exp"
        shutil.copytree(smoke.out_dir, out_dir)
        usage_path = out_dir / STAGE_USAGE_FILE
        zero_record = {
            "stage": "eval",
            "stage_work_id": "eval__breaker-skipped-set",
            "attempt_index": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "wall_seconds": 0.0,
            "cache_hits": 0,
            "lower_bound": False,
        }
        with usage_path.open("a") as handle:
            handle.write(json.dumps(zero_record) + "\n")

        counts = experiment_smoke.check_stage_coverage(out_dir)
        assert counts["eval"] >= 2  # the real record(s) plus the 0/0 work id

    def test_stage_coverage_fails_a_stage_with_only_zero_token_work(
        self, smoke: SmokeRun, tmp_path
    ):
        """No work id with positive input AND output tokens means the stage's
        role never measurably hit the backend."""
        out_dir = tmp_path / "exp"
        shutil.copytree(smoke.out_dir, out_dir)
        usage_path = out_dir / STAGE_USAGE_FILE
        records = [json.loads(line) for line in usage_path.read_text().splitlines() if line.strip()]
        for record in records:
            if record["stage"] == "proposal":
                record["input_tokens"] = 0
                record["output_tokens"] = 0
        usage_path.write_text("".join(json.dumps(record) + "\n" for record in records))

        with pytest.raises(experiment_smoke.SmokeError, match="proposal"):
            experiment_smoke.check_stage_coverage(out_dir)

    def test_long_coverage_and_measured_spend_read_the_smoke_artifacts(self, smoke: SmokeRun):
        coverage = experiment_smoke.long_run_coverage(smoke.config, smoke.out_dir)
        assert set(coverage) == {"graphwalks", "oolong_pairs"}
        for environment, counts in coverage.items():
            assert counts["runs"] > 0, environment
            assert counts["uncapped"] == counts["runs"], environment
        assert experiment_smoke.measured_spend(smoke.out_dir) > 0.0
        # Every environment has an uncapped long run, so the KTD6 gate passes.
        experiment_smoke.check_long_run_coverage(smoke.config, coverage)

    def test_a_cost_less_completed_run_is_refused(self, tmp_path):
        round_path = tmp_path / "opt" / "round_01" / "mining" / "round_01"
        round_path.mkdir(parents=True)
        (round_path / "runs.jsonl").write_text(
            json.dumps({"run_id": "inst__a01", "cost": None}) + "\n"
        )
        with pytest.raises(experiment_smoke.SmokeError, match="no cost"):
            experiment_smoke.check_costs_present(tmp_path)

    def test_a_terminated_run_without_cost_is_counted_not_refused(self, tmp_path):
        """A run the experiment's own timeout or budget killed reports no cost
        and is already flagged a lower bound; only a *completed* cost-less run
        means the route went blind."""
        round_path = tmp_path / "opt" / "round_01" / "mining" / "round_01"
        round_path.mkdir(parents=True)
        (round_path / "runs.jsonl").write_text(
            json.dumps({"run_id": "paid__a01", "cost": 0.01})
            + "\n"
            + json.dumps({"run_id": "killed__a01", "cost": None, "usage_lower_bound": True})
            + "\n"
        )
        assert experiment_smoke.check_costs_present(tmp_path) == (2, 1)

    def test_a_missing_artifact_is_named(self, tmp_path):
        with pytest.raises(experiment_smoke.SmokeError, match=CONFIG_FILENAME):
            experiment_smoke.check_directory_contract(tmp_path)


class TestLiveSmokeLongRunGate:
    """KTD6: a smoke whose long sample is entirely censored must go red.

    The plan's Definition of Done wants at least one *uncapped* long run per
    environment. These exercise the two ways an environment can end up with
    none -- every long run terminated by a cap, and no long aggregate at all
    (a breaker that tripped before the long sets ran) -- and pin the ordering
    the failure must respect: the report is written and printed first.
    """

    def test_an_all_capped_environment_is_named_and_fails(self, smoke: SmokeRun, tmp_path: Path):
        out_dir = doctored_experiment(smoke, tmp_path, cap_long_runs("oolong_pairs"))

        coverage = experiment_smoke.long_run_coverage(smoke.config, out_dir)
        assert coverage["oolong_pairs"]["runs"] > 0
        assert coverage["oolong_pairs"]["uncapped"] == 0
        assert coverage["graphwalks"]["uncapped"] > 0

        with pytest.raises(experiment_smoke.SmokeError) as excinfo:
            experiment_smoke.check_long_run_coverage(smoke.config, coverage)
        message = str(excinfo.value)
        assert "oolong_pairs" in message
        assert "graphwalks" not in message
        assert "FRESH --out-dir" in message
        assert "LIVE_MAX_BUDGET_USD" in message

    def test_an_environment_with_no_long_aggregate_at_all_fails(
        self, smoke: SmokeRun, tmp_path: Path
    ):
        """A missing environment key is a censored long sample, not a skip."""
        out_dir = doctored_experiment(smoke, tmp_path, drop_long_sets("oolong_pairs"))

        coverage = experiment_smoke.long_run_coverage(smoke.config, out_dir)
        assert coverage["oolong_pairs"] == {"runs": 0, "uncapped": 0}
        with pytest.raises(experiment_smoke.SmokeError, match="oolong_pairs"):
            experiment_smoke.check_long_run_coverage(smoke.config, coverage)

    def test_run_live_writes_and_prints_the_report_before_it_fails(
        self, smoke: SmokeRun, tmp_path: Path, capsys, monkeypatch
    ):
        """The operator wakes to a failed run AND every measured number."""
        out_dir = doctored_experiment(smoke, tmp_path, cap_long_runs("oolong_pairs"))
        report_path = out_dir / REPORT_FILENAME
        assert not report_path.exists()
        stub_live_stages(monkeypatch)

        with pytest.raises(experiment_smoke.SmokeError, match="no uncapped long run"):
            experiment_smoke.run_live(experiment_smoke.live_config(), out_dir)

        out = capsys.readouterr().out
        assert report_path.exists(), "the report must be on disk before the failure"
        assert json.loads(report_path.read_text())["profile"] == "smoke"
        for stage in STAGES:  # the rendered markdown reached stdout
            assert stage in out
        assert f"Wrote {report_path}" in out
        assert "Total measured spend" in out
        assert "SMOKE PASSED" not in out

    def test_run_live_passes_the_gate_when_a_long_run_is_uncapped(
        self, smoke: SmokeRun, tmp_path: Path, capsys, monkeypatch
    ):
        """The same path, undoctored: the gate is not unconditionally red."""
        out_dir = doctored_experiment(smoke, tmp_path, lambda summary: None)
        stub_live_stages(monkeypatch)

        assert experiment_smoke.run_live(experiment_smoke.live_config(), out_dir) == 0
        assert "SMOKE PASSED" in capsys.readouterr().out


class TestLiveSmokeGuards:
    def test_configured_live_budgets_stay_under_the_five_dollar_ceiling(self):
        config = experiment_smoke.live_config()

        # Governed: t=1 x (1 mining + 2 fixed + k=2) + 2 conditions = 7 breakers,
        # each admitting $0.24 cumulative + one $0.20 per-run overshoot = $3.08.
        assert experiment_smoke.breaker_count(config) == 7
        governed = 7 * (config.caps.candidate_budget + config.caps.max_budget)
        assert governed == pytest.approx(3.08)

        # Ungoverned: 2 probe + (1 x 2 live instances x 1 attempt x 3 x 3)
        # attribution + (1 x 3 x 3) proposal = 29 calls, each priced at the
        # char-cap-derived input bound plus max_output_tokens out, at the
        # configured LIST tier -- computed from the config so a pricing flip
        # moves the assertions with it.
        price = config.pricing.list_price
        per_call = (
            experiment_smoke.UNGOVERNED_INPUT_TOKENS * price.input_per_million
            + config.decoding.max_output_tokens * price.output_per_million
        ) / 1_000_000
        assert experiment_smoke.ungoverned_call_count(config) == 29
        assert experiment_smoke.ungoverned_call_ceiling(config) == pytest.approx(per_call)
        # Literal sanity pin at the shipped Qwen list rate ($0.10 / $0.30 per
        # 1M): (49,152 x 0.10 + 4,096 x 0.30) / 1e6 = $0.0061440 per call.
        assert per_call == pytest.approx(0.0061440)
        ungoverned = 29 * experiment_smoke.ungoverned_call_ceiling(config)
        assert ungoverned == pytest.approx(0.178176)

        assert experiment_smoke.spend_ceiling(config) == pytest.approx(governed + ungoverned)
        # The standalone --probe invocation (run FIRST) spends two more
        # ungoverned calls on top of the --live run's own probe.
        probe_reserve = experiment_smoke.standalone_probe_reserve_usd(config)
        assert probe_reserve == pytest.approx(2 * experiment_smoke.ungoverned_call_ceiling(config))
        # The proven figure is cumulative: this invocation plus the standalone
        # probe reserve plus the $0.60 U4 pytest live reserve, under the one
        # $5 ceiling. Literal sanity pin: 3.08 + 0.178176 + 0.012288 + 0.60.
        cumulative = (
            experiment_smoke.spend_ceiling(config)
            + probe_reserve
            + experiment_smoke.PYTEST_LIVE_RESERVE_USD
        )
        assert cumulative < 5.0
        assert cumulative < experiment_smoke.SPEND_CEILING_USD
        assert experiment_smoke.check_budget_arithmetic(config) == pytest.approx(cumulative)
        assert experiment_smoke.check_budget_arithmetic(config) == pytest.approx(3.870464)

    def test_the_ceiling_scales_with_the_round_count(self):
        """Every per-round breaker and per-round LM call is armed t times."""
        from dataclasses import replace

        config = experiment_smoke.live_config()
        three_rounds = replace(config, loop=replace(config.loop, t=3))

        # 3 x (1 mining + 2 fixed + k=2) + 2 conditions.
        assert experiment_smoke.breaker_count(three_rounds) == 17
        # 2 probe + 3 x (6 attribution + 3 proposal) x 3 transport retries.
        assert experiment_smoke.ungoverned_call_count(three_rounds) == 2 + 3 * 27
        assert experiment_smoke.spend_ceiling(three_rounds) > experiment_smoke.spend_ceiling(config)
        # A three-round smoke is not affordable under the $5 ceiling, and the
        # arithmetic says so rather than discovering it while spending.
        with pytest.raises(experiment_smoke.SmokeError, match="ceiling"):
            experiment_smoke.check_budget_arithmetic(three_rounds)

    def test_the_ceiling_covers_the_ungoverned_calls_and_the_reserve(self):
        """The proof is not a proof if the probe, the LM stages, or the U4
        pytest tier are treated as free."""
        config = experiment_smoke.live_config()
        governed_only = experiment_smoke.breaker_count(config) * (
            config.caps.candidate_budget + config.caps.max_budget
        )
        assert experiment_smoke.spend_ceiling(config) > governed_only
        assert experiment_smoke.check_budget_arithmetic(config) == pytest.approx(
            experiment_smoke.spend_ceiling(config)
            + experiment_smoke.standalone_probe_reserve_usd(config)
            + experiment_smoke.PYTEST_LIVE_RESERVE_USD
        )

    def test_live_profile_keeps_the_smoke_semantics_and_documents_its_shrinks(self):
        """Semantics -- decoding, promotion, loop shape, environments' datasets
        -- are byte-identical to the smoke profile; only the documented caps
        and instance counts move (identity keys, so a fresh out-dir)."""
        config = experiment_smoke.live_config()
        smoke_profile = load_config("smoke")

        assert config.profile == "smoke"
        assert config.loop == smoke_profile.loop
        assert config.decoding == smoke_profile.decoding
        assert config.promotion == smoke_profile.promotion
        assert config.backends == smoke_profile.backends
        assert config.caps.max_budget == experiment_smoke.LIVE_MAX_BUDGET_USD
        assert config.caps.candidate_budget == experiment_smoke.LIVE_CANDIDATE_BUDGET_USD
        assert config.caps.max_timeout == experiment_smoke.LIVE_MAX_TIMEOUT_SECONDS
        # The live instance-count shrinks, and nothing else about the splits.
        assert config.splits.n_in == experiment_smoke.LIVE_HELD_IN == 2
        assert config.splits.test_short == experiment_smoke.LIVE_TEST_SHORT == 2
        assert config.splits.test_long == experiment_smoke.LIVE_TEST_LONG == 1
        assert config.splits.n_ho == smoke_profile.splits.n_ho
        assert config.splits.seed == smoke_profile.splits.seed
        oolong = config.environments.oolong_pairs
        assert oolong.n_short == experiment_smoke.LIVE_OOLONG_SHORT == 2
        assert oolong.n_long == experiment_smoke.LIVE_OOLONG_LONG == 1
        smoke_oolong = smoke_profile.environments.oolong_pairs
        assert oolong.dataset_repo == smoke_oolong.dataset_repo
        assert oolong.dataset_revision == smoke_oolong.dataset_revision
        assert oolong.task_ids == smoke_oolong.task_ids
        assert config.environments.graphwalks == smoke_profile.environments.graphwalks

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

    def test_declines_without_the_openrouter_key_and_names_it(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        block_network(monkeypatch)

        assert experiment_smoke.main(["--live"]) == 1
        out = capsys.readouterr().out
        assert "OPENROUTER_API_KEY" in out
        assert "Nothing was spent" in out

    def test_probe_declines_without_the_openrouter_key(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        block_network(monkeypatch)

        assert experiment_smoke.main(["--probe"]) == 1
        out = capsys.readouterr().out
        assert "OPENROUTER_API_KEY" in out
        # Only the configured backend's credentials are demanded.
        assert "AZURE_API_KEY" not in out
        assert "Nothing was spent" in out


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
