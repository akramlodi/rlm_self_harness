"""The evaluation runner: fixed harness conditions over frozen test splits (R8).

``run_evaluation`` is a thin loop -- condition x test set x one governed
``run_round`` -- over artifacts the rest of the scaffold already produced. It
mints nothing: split instances come only from the persisted
``splits/<env>_<length>_test.jsonl`` files (R8: never re-sampled here), the
``sh_rlm`` condition's harness comes only from the orchestrator's frozen
``sh_rlm/harness.json`` envelope, and every run persists through the same
persist-first driver the optimization loop uses.

Directory contract under ``out_dir``::

    config.json                  # identity-checked before anything runs (R3)
    splits/                      # materialize_splits output (verified, not redrawn)
    stage_usage.jsonl            # one StageMeter record per condition x set attempt
    eval/
        eval_summary.json        # the aggregate (new artifact; see below)
        <condition>/
            work/                # materialized modules for frozen harnesses
            <env>_<length>/round_00/{harness.json,instances.jsonl,runs.jsonl,runs/}

Conditions are named harness *sources*, resolved from one mapping
(``CONDITIONS``) rather than branched on at the call sites: ``b1`` is the
registry incumbent ``H0``, ``sh_rlm`` is the frozen envelope rematerialized
through ``materialize_harness`` and re-hashed against the freeze-time hash
(a tampered envelope raises before any run executes). Adding ``h1`` -- or a
fine-tuned ``f1`` reading a different frozen path -- is one entry in that
mapping; nothing else in this module changes.

Byte-identical instances across conditions (R8) is enforced, not assumed. Each
split file is the single source for every condition's round, the driver
byte-compares ``instances.jsonl`` on resume, and after each round this module
compares the persisted ``instances.jsonl`` bytes against the split file's
bytes and records their sha256 in the summary -- so the cross-condition
guarantee is checkable from the artifact alone.

Spend control (R12): one ``CandidateSpendBreaker`` per condition, cumulative
across that condition's test sets, exactly as ``evaluate_subject`` shares one
breaker across a subject's splits. A tripped breaker skips the remaining runs
and records ``over_budget`` plus the skipped run ids per set -- eval is
measurement, so a budget-stopped condition is reported, not raised past the
sibling conditions' results (the mining stage raises instead, because a
truncated mining round cannot feed a meaningful proposal stage).

Repetitions (KTD8): ``operational.eval_repetitions`` becomes the round's
``attempts`` -- unseeded temperature samples, no per-attempt seed plumbing.

Aggregation lands in a NEW artifact, ``eval/eval_summary.json``. It mirrors
``validation.split_aggregate``'s shape (pass counts, costs, sub-calls) and
adds the U4 manifest token/time keys; it is deliberately not
``validation.summary.json``, whose byte-compare on resume makes any schema
change a resume-breaker (KTD4). The payload is a pure function of the
persisted rounds -- no timestamps -- so a resumed invocation that executes
nothing rewrites identical bytes, while a resumed invocation that completes
skipped runs grows it.

Single-threaded, main thread only: the SIGALRM hard-deadline backstop in
``shrlm.optimization.costs`` binds only there, and this module never spawns
threads around run execution.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shrlm.environments.graphwalks import GraphWalksVerifier
from shrlm.environments.oolong_pairs import OolongPairsVerifier
from shrlm.experiment.config import (
    ExperimentConfig,
    identity_hash,
    round_config_kwargs,
    validation_caps,
)
from shrlm.experiment.orchestrator import (
    FROZEN_DIR,
    FROZEN_HARNESS_FILENAME,
    INITIAL_INCUMBENT,
    WORK_DIR,
    check_identity,
)
from shrlm.experiment.splits import (
    LENGTHS,
    MANIFEST_FILE,
    LoaderFn,
    materialize_splits,
    split_file_name,
)
from shrlm.experiment.usage import (
    STAGE_USAGE_FILE,
    StageMeter,
    UsageTotals,
    aggregate_manifest_usage,
)
from shrlm.harness_identity import harness_hash
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.candidates import CandidateRejection, materialize_harness
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import INSTANCES_FILE, RoundConfig, load_manifest
from shrlm.optimization.types import Verifier
from shrlm.optimization.validation import EVAL_ROUND_INDEX, split_aggregate
from shrlm.rlm_harness import HARNESSES, Harness

EVAL_DIR = "eval"
EVAL_SUMMARY_FILENAME = "eval_summary.json"
EVAL_SUMMARY_FORMAT = "shrlm-eval-summary/v1"

STAGE_EVAL = "eval"

# The split role every condition is evaluated on; held-in/held-out belong to
# the optimization loop and are never read here.
ROLE_TEST = "test"

CONDITION_B1 = "b1"
CONDITION_SH_RLM = "sh_rlm"

# The limit kwargs ``governed_limits`` owns; the config's own values for them
# are dropped so the caps (merged tighten-only against the harness's S6
# policy) are the only source of a run's limits.
_GOVERNED_KEYS = ("attempts", "max_iterations", "max_depth", "max_budget", "max_timeout")


class EvaluationPersistenceError(RuntimeError):
    """Persisted evaluation state contradicts itself or the configuration."""


# ---------------------------------------------------------------------------
# Conditions: named harness sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryHarnessSource:
    """A condition served by a harness in the module registry (e.g. B1 = H0)."""

    registry_name: str

    def describe(self) -> dict[str, str]:
        return {"kind": "registry", "registry_name": self.registry_name}

    def load(self, out_dir: Path, work_dir: Path) -> Harness:
        if self.registry_name not in HARNESSES:
            raise EvaluationPersistenceError(
                f"harness {self.registry_name!r} is not in the registry "
                f"({sorted(HARNESSES)}); an evaluation condition must name a real harness"
            )
        return HARNESSES[self.registry_name]


@dataclass(frozen=True)
class FrozenHarnessSource:
    """A condition served by a persisted ``harness.json`` envelope.

    The envelope is rematerialized through ``materialize_harness`` and the
    rebuilt harness re-hashed against the hash the freeze recorded, so an
    envelope that was edited after freezing -- or one that does not round-trip
    -- is refused instead of silently evaluated as something else.
    """

    relative_path: str

    def describe(self) -> dict[str, str]:
        return {"kind": "frozen", "path": self.relative_path}

    def load(self, out_dir: Path, work_dir: Path) -> Harness:
        path = out_dir / self.relative_path
        if not path.exists():
            raise EvaluationPersistenceError(
                f"{path} does not exist; this condition evaluates the frozen harness, so "
                "the optimization experiment must have completed and frozen it first"
            )
        envelope = json.loads(path.read_text())
        recorded = str(envelope["hash"])
        work_dir.mkdir(parents=True, exist_ok=True)
        harness = materialize_harness(envelope["harness"], work_dir / f"_frozen_{recorded[:16]}.py")
        rebuilt = harness_hash(harness)
        if rebuilt != recorded:
            raise EvaluationPersistenceError(
                f"{path} records harness hash {recorded}, but rematerializing its envelope "
                f"produced {rebuilt}; the frozen harness does not round-trip (the envelope "
                "was modified, or its surfaces are not reconstructible). Refusing to "
                "evaluate a condition whose identity cannot be verified."
            )
        return harness


HarnessSource = RegistryHarnessSource | FrozenHarnessSource

# The condition registry. Every condition the scaffold can evaluate is one
# entry here; H1 (lambda-RLM) and F1 (fine-tuned baseline) land as further
# entries without reshaping this module's API.
CONDITIONS: dict[str, HarnessSource] = {
    CONDITION_B1: RegistryHarnessSource(INITIAL_INCUMBENT),
    CONDITION_SH_RLM: FrozenHarnessSource(f"{FROZEN_DIR}/{FROZEN_HARNESS_FILENAME}"),
}

DEFAULT_CONDITIONS: tuple[str, ...] = (CONDITION_B1, CONDITION_SH_RLM)

DEFAULT_VERIFIERS: dict[str, Verifier] = {
    "graphwalks": GraphWalksVerifier(),
    "oolong_pairs": OolongPairsVerifier(),
}


def resolve_conditions(conditions: Sequence[str]) -> list[tuple[str, HarnessSource]]:
    """Resolve condition names to their harness sources, in the caller's order."""
    if not conditions:
        raise ValueError("run_evaluation needs at least one condition")
    seen: set[str] = set()
    resolved: list[tuple[str, HarnessSource]] = []
    for condition_id in conditions:
        if condition_id not in CONDITIONS:
            raise ValueError(
                f"unknown evaluation condition {condition_id!r}; known conditions are "
                f"{sorted(CONDITIONS)}"
            )
        if condition_id in seen:
            raise ValueError(
                f"duplicate evaluation condition {condition_id!r}: condition directories "
                "are keyed by id, so repeats would mix two conditions' runs"
            )
        seen.add(condition_id)
        resolved.append((condition_id, CONDITIONS[condition_id]))
    return resolved


# ---------------------------------------------------------------------------
# Test sets: whatever the persisted splits manifest froze (R8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestSet:
    """One frozen test split: which environment and length, and its file."""

    environment: str
    length: str

    @property
    def set_id(self) -> str:
        return f"{self.environment}_{self.length}"

    @property
    def file_name(self) -> str:
        return split_file_name(self.environment, self.length, ROLE_TEST)


def test_sets(splits_dir: Path) -> list[TestSet]:
    """Every test split the persisted manifest records, in (env, length) order.

    Derived from the manifest rather than from the config's split plan: only
    materialized environments have files on disk, and eval reads files, never
    the plan.
    """
    manifest_path = splits_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise EvaluationPersistenceError(
            f"{manifest_path} does not exist; evaluation reads only persisted split files"
        )
    manifest = json.loads(manifest_path.read_text())
    sets = [
        TestSet(environment=environment, length=length)
        for environment in sorted(manifest["environments"])
        for length in LENGTHS
        if split_file_name(environment, length, ROLE_TEST)
        in manifest["environments"][environment]["files"]
    ]
    if not sets:
        raise EvaluationPersistenceError(
            f"{manifest_path} records no {ROLE_TEST} split; there is nothing to evaluate"
        )
    return sets


def read_instances(path: Path) -> list[dict[str, Any]]:
    """One persisted split file's instances, in file order."""
    instances = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not instances:
        raise EvaluationPersistenceError(f"{path} holds no instances; cannot evaluate on it")
    return instances


def verify_instance_identity(set_path: Path, split_path: Path) -> str:
    """Check the round consumed the split file verbatim; return its sha256.

    The driver already refuses to resume a round whose ``instances.jsonl``
    differs from the configured instances; this makes the cross-condition half
    of R8 explicit -- every condition's round holds the split file's exact
    bytes, so one recorded sha256 identifies the instances all conditions saw.
    """
    persisted_path = round_dir(set_path, EVAL_ROUND_INDEX) / INSTANCES_FILE
    persisted = persisted_path.read_bytes()
    expected = split_path.read_bytes()
    if persisted != expected:
        raise EvaluationPersistenceError(
            f"{persisted_path} does not match the frozen split {split_path} verbatim; "
            "every condition must consume byte-identical instances (R8)"
        )
    return hashlib.sha256(expected).hexdigest()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionEvaluation:
    """One condition's evaluation across every test set."""

    condition_id: str
    harness_hash: str
    outcome: str
    spent: float
    test_sets: dict[str, dict[str, Any]]

    @property
    def over_budget(self) -> bool:
        return self.outcome == OUTCOME_OVER_BUDGET

    def to_payload(self, source: HarnessSource) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "source": source.describe(),
            "harness_hash": self.harness_hash,
            "outcome": self.outcome,
            "spent": self.spent,
            "test_sets": self.test_sets,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """What one ``run_evaluation`` invocation measured and persisted."""

    out_dir: Path
    eval_dir: Path
    summary_path: Path
    conditions: list[ConditionEvaluation]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class _Evaluation:
    """One invocation's shared state; ``run`` is the whole condition x set loop."""

    config: ExperimentConfig
    conditions: list[tuple[str, HarnessSource]]
    out_dir: Path
    verifiers: dict[str, Verifier]
    loaders: dict[str, LoaderFn] | None
    caps: ValidationCaps = field(init=False)
    usage_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.caps = validation_caps(self.config)
        self.usage_path = self.out_dir / STAGE_USAGE_FILE

    def run(self) -> EvaluationResult:
        check_identity(self.config, self.out_dir)
        splits_dir = materialize_splits(self.config, self.out_dir, loaders=self.loaders)
        sets = test_sets(splits_dir)
        eval_dir = self.out_dir / EVAL_DIR

        evaluations: list[ConditionEvaluation] = []
        payloads: dict[str, Any] = {}
        for condition_id, source in self.conditions:
            evaluation = self._evaluate_condition(condition_id, source, splits_dir, sets)
            evaluations.append(evaluation)
            payloads[condition_id] = evaluation.to_payload(source)

        summary = {
            "format": EVAL_SUMMARY_FORMAT,
            "profile": self.config.profile,
            "identity_hash": identity_hash(self.config),
            "eval_repetitions": self.config.operational.eval_repetitions,
            "candidate_budget": self.caps.candidate_budget,
            "test_sets": [test_set.set_id for test_set in sets],
            "conditions": payloads,
        }
        summary_path = eval_dir / EVAL_SUMMARY_FILENAME
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return EvaluationResult(
            out_dir=self.out_dir,
            eval_dir=eval_dir,
            summary_path=summary_path,
            conditions=evaluations,
            summary=summary,
        )

    def _evaluate_condition(
        self,
        condition_id: str,
        source: HarnessSource,
        splits_dir: Path,
        sets: list[TestSet],
    ) -> ConditionEvaluation:
        """One condition over every test set, under one cumulative breaker."""
        condition_dir = self.out_dir / EVAL_DIR / condition_id
        harness = source.load(self.out_dir, condition_dir / WORK_DIR)
        limits = governed_limits(condition_id, harness.runtime_policy, self.caps)
        if isinstance(limits, CandidateRejection):
            raise EvaluationPersistenceError(
                f"evaluation condition {condition_id!r} violates the experiment-owned caps "
                f"({limits.reason}); a fixed condition that cannot run under the eval limits "
                "is a misconfigured experiment, not a rejectable candidate"
            )
        breaker = CandidateSpendBreaker(self.caps)
        aggregates = {
            test_set.set_id: self._evaluate_set(
                condition_id, harness, limits, breaker, splits_dir, test_set
            )
            for test_set in sets
        }
        return ConditionEvaluation(
            condition_id=condition_id,
            harness_hash=harness_hash(harness),
            outcome=OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
            spent=breaker.spent,
            test_sets=aggregates,
        )

    def _evaluate_set(
        self,
        condition_id: str,
        harness: Harness,
        limits: dict[str, Any],
        breaker: CandidateSpendBreaker,
        splits_dir: Path,
        test_set: TestSet,
    ) -> dict[str, Any]:
        """One condition x test set: a governed round, metered and aggregated."""
        split_path = splits_dir / test_set.file_name
        instances = read_instances(split_path)
        set_path = self.out_dir / EVAL_DIR / condition_id / test_set.set_id
        kwargs = round_config_kwargs(self.config)
        for key in _GOVERNED_KEYS:
            kwargs.pop(key)
        round_config = RoundConfig(
            round_index=EVAL_ROUND_INDEX,
            harness=harness,
            instances=instances,
            verifier=self.verifier_for(test_set.environment),
            out_dir=set_path,
            attempts=self.config.operational.eval_repetitions,
            **kwargs,
            **limits,
        )
        with StageMeter(
            stage=STAGE_EVAL,
            stage_work_id=f"{EVAL_DIR}/{condition_id}/{test_set.set_id}",
            round_index=EVAL_ROUND_INDEX,
            out_path=self.usage_path,
        ) as meter:
            known = len(load_manifest(set_path, EVAL_ROUND_INDEX))
            result = run_governed_round(round_config, breaker)
            meter.add(aggregate_manifest_usage(result.entries[known:]))

        usage = aggregate_manifest_usage(result.entries)
        return {
            "environment": test_set.environment,
            "length": test_set.length,
            "round_path": f"{test_set.set_id}/round_{EVAL_ROUND_INDEX:02d}",
            "split_file": test_set.file_name,
            "instances_sha256": verify_instance_identity(set_path, split_path),
            "n_instances": len(instances),
            "attempts": self.config.operational.eval_repetitions,
            "outcome": result.outcome,
            "skipped_run_ids": list(result.skipped_run_ids),
            **split_aggregate(set_path),
            **_usage_payload(usage),
        }

    def verifier_for(self, environment: str) -> Verifier:
        if environment not in self.verifiers:
            raise EvaluationPersistenceError(
                f"no verifier registered for environment {environment!r}; known "
                f"environments are {sorted(self.verifiers)}"
            )
        return self.verifiers[environment]


def _usage_payload(usage: UsageTotals) -> dict[str, Any]:
    """The U4 manifest-derived usage keys ``split_aggregate`` does not carry."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "wall_seconds": usage.wall_seconds,
        "usage_lower_bound": usage.lower_bound,
    }


def run_evaluation(
    config: ExperimentConfig,
    conditions: Sequence[str],
    out_dir: Path | str,
    *,
    verifiers: dict[str, Verifier] | None = None,
    loaders: dict[str, LoaderFn] | None = None,
) -> EvaluationResult:
    """Evaluate fixed harness conditions on the frozen test splits (R8).

    For each condition (in the caller's order) and each persisted test split:
    one ``run_governed_round`` of ``operational.eval_repetitions`` attempts per
    instance into ``eval/<condition>/<env>_<length>/``, under one cumulative
    spend breaker per condition (R12). Aggregates land in
    ``eval/eval_summary.json``.

    Idempotent and crash-safe over ``out_dir``: the config identity is verified
    first (R3), the persisted splits are verified rather than redrawn, and
    persisted runs are never re-executed -- a re-invocation of a finished
    evaluation makes zero model calls and rewrites byte-identical summary bytes.

    Args:
        config: The loaded experiment configuration (one profile).
        conditions: Condition names to evaluate, in evaluation order; each must
            be a key of ``CONDITIONS`` (``DEFAULT_CONDITIONS`` is the scaffold's
            pair, ``b1`` and ``sh_rlm``).
        out_dir: The experiment directory (the one the orchestrator wrote).
        verifiers: Per-environment verifiers; defaults to ``DEFAULT_VERIFIERS``.
        loaders: Optional environment-loader overrides for ``materialize_splits``
            (tests inject offline loaders here).

    Returns:
        The ``EvaluationResult`` with each condition's aggregates and the
        persisted summary payload.

    Raises:
        ValueError: On an unknown, duplicated, or empty condition list.
        IdentityMismatchError: The configured identity hash does not match the
            experiment's persisted one -- raised before any run executes.
        EvaluationPersistenceError: A frozen envelope is missing or does not
            round-trip to its recorded hash, a condition violates the
            experiment-owned caps, an environment has no verifier, or a
            persisted round does not hold the frozen split's exact bytes.
    """
    evaluation = _Evaluation(
        config=config,
        conditions=resolve_conditions(conditions),
        out_dir=Path(out_dir),
        verifiers=dict(verifiers) if verifiers is not None else dict(DEFAULT_VERIFIERS),
        loaders=loaders,
    )
    return evaluation.run()


__all__ = [
    "CONDITIONS",
    "CONDITION_B1",
    "CONDITION_SH_RLM",
    "DEFAULT_CONDITIONS",
    "DEFAULT_VERIFIERS",
    "EVAL_DIR",
    "EVAL_SUMMARY_FILENAME",
    "EVAL_SUMMARY_FORMAT",
    "ConditionEvaluation",
    "EvaluationPersistenceError",
    "EvaluationResult",
    "FrozenHarnessSource",
    "HarnessSource",
    "RegistryHarnessSource",
    "TestSet",
    "resolve_conditions",
    "run_evaluation",
    "test_sets",
]
