"""The evaluation runner: fixed inference methods over frozen test splits (R8).

``run_evaluation`` is a thin loop -- condition x test set x one governed
method execution -- over artifacts the rest of the scaffold already produced.
It mints nothing: split instances come only from the persisted
``splits/<env>_<length>_test.jsonl`` files (R8: never re-sampled here), and the
``sh_rlm`` condition's harness comes only from the orchestrator's frozen
``sh_rlm/harness.json`` envelope. Each loaded ``EvaluationMethod`` owns its
persist-first runner and aggregate reader.

Directory contract under ``out_dir``::

    config.json                  # identity-checked before anything runs (R3)
    splits/                      # materialize_splits output (verified, not redrawn)
    stage_usage.jsonl            # one StageMeter record per condition x set attempt
    eval/
        eval_summary.json        # the aggregate (new artifact; see below)
        <condition>/
            work/                # materialized modules for frozen harnesses
            <env>_<length>/round_00/{harness.json|method.json,instances.jsonl,runs.jsonl,runs/}

Conditions are named method sources, resolved from one mapping (``CONDITIONS``)
rather than branched on at call sites. A source loads an ``EvaluationMethod``
that owns method identity, cap validation, governed execution, and aggregation.
The four conditions are ``b1`` (the registry incumbent ``H0``), ``h0_star``
(the shipped reference ``H0*``), ``lambda_rlm`` (the pinned upstream λ-RLM
method), and ``sh_rlm`` (the frozen envelope rematerialized and re-hashed
against its freeze-time hash). Harness and non-harness methods join through
the same protocol, not a condition-id branch.

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
from typing import Any, Protocol

from rlm.core.types import RLMChatCompletion
from shrlm.baselines.lambda_rlm import (
    LAMBDA_RLM_METHOD_KIND,
    LAMBDA_RLM_SOURCE_SHA256,
    LAMBDA_RLM_UPSTREAM_REPOSITORY,
    LAMBDA_RLM_UPSTREAM_REVISION,
    LambdaBaselineConfig,
    lambda_method_hash,
)
from shrlm.baselines.lambda_runner import (
    METHOD_FILE,
    LambdaRoundConfig,
    run_governed_lambda_round,
)
from shrlm.environments.graphwalks import GraphWalksVerifier
from shrlm.environments.oolong_pairs import OolongPairsVerifier
from shrlm.experiment.config import (
    GOVERNED_ROUND_KEYS,
    ExperimentConfig,
    identity_hash,
    round_config_kwargs,
    validation_caps,
)
from shrlm.experiment.errors import ExperimentError
from shrlm.experiment.orchestrator import (
    FROZEN_DIR,
    FROZEN_HARNESS_FILENAME,
    INITIAL_INCUMBENT,
    WORK_DIR,
    check_identity,
    rematerialize_harness_envelope,
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
    read_jsonl,
)
from shrlm.harness_identity import harness_hash
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.candidates import CandidateRejection
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    GovernedRoundResult,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import (
    INSTANCES_FILE,
    RoundConfig,
    load_manifest,
    verify_trace,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verifier
from shrlm.optimization.validation import EVAL_ROUND_INDEX, split_aggregate
from shrlm.rlm_harness import HARNESSES, Harness

EVAL_DIR = "eval"
EVAL_SUMMARY_FILENAME = "eval_summary.json"
EVAL_SUMMARY_FORMAT = "shrlm-eval-summary/v2"

STAGE_EVAL = "eval"

# The split role every condition is evaluated on; held-in/held-out belong to
# the optimization loop and are never read here.
ROLE_TEST = "test"

CONDITION_B1 = "b1"
CONDITION_H0_STAR = "h0_star"
CONDITION_SH_RLM = "sh_rlm"
CONDITION_LAMBDA_RLM = "lambda_rlm"


class EvaluationPersistenceError(ExperimentError):
    """Persisted evaluation state contradicts itself or the configuration."""


# ---------------------------------------------------------------------------
# Conditions: named sources of executable evaluation methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSetRequest:
    """Method-neutral inputs for one condition x test-set execution."""

    config: ExperimentConfig
    condition_id: str
    instances: list[dict[str, Any]]
    verifier: Verifier
    out_dir: Path
    attempts: int
    breaker: CandidateSpendBreaker
    caps: ValidationCaps


class EvaluationMethod(Protocol):
    """A loaded inference method that evaluation can identify, run, and aggregate."""

    @property
    def method_kind(self) -> str: ...

    @property
    def method_hash(self) -> str: ...

    def validate_caps(self, condition_id: str, caps: ValidationCaps) -> None: ...

    def run_set(self, request: EvaluationSetRequest) -> GovernedRoundResult: ...

    def aggregate_set(self, set_path: Path) -> dict[str, Any]: ...


class ConditionSource(Protocol):
    """Persistent provenance capable of loading one executable evaluation method."""

    def describe(self) -> dict[str, str]: ...

    def load(self, out_dir: Path, work_dir: Path) -> EvaluationMethod: ...


@dataclass(frozen=True)
class HarnessEvaluationMethod:
    """EvaluationMethod adapter for the existing editable RLM harness runtime."""

    harness: Harness

    @property
    def method_kind(self) -> str:
        return "harness"

    @property
    def method_hash(self) -> str:
        return harness_hash(self.harness)

    def limits_for(
        self,
        condition_id: str,
        caps: ValidationCaps,
    ) -> dict[str, Any]:
        """Resolve the harness policy under experiment-owned caps or fail loudly."""
        limits = governed_limits(condition_id, self.harness.runtime_policy, caps)
        if isinstance(limits, CandidateRejection):
            raise EvaluationPersistenceError(
                f"evaluation condition {condition_id!r} violates the experiment-owned caps "
                f"({limits.reason}); a fixed condition that cannot run under the eval limits "
                "is a misconfigured experiment, not a rejectable candidate"
            )
        return limits

    def validate_caps(self, condition_id: str, caps: ValidationCaps) -> None:
        self.limits_for(condition_id, caps)

    def run_set(self, request: EvaluationSetRequest) -> GovernedRoundResult:
        kwargs = round_config_kwargs(request.config)
        for key in GOVERNED_ROUND_KEYS:
            kwargs.pop(key)
        round_config = RoundConfig(
            round_index=EVAL_ROUND_INDEX,
            harness=self.harness,
            instances=request.instances,
            verifier=request.verifier,
            out_dir=request.out_dir,
            attempts=request.attempts,
            **kwargs,
            **self.limits_for(request.condition_id, request.caps),
        )
        return run_governed_round(round_config, request.breaker)

    def aggregate_set(self, set_path: Path) -> dict[str, Any]:
        aggregate = split_aggregate(set_path)
        recorded_hash = str(aggregate.pop("harness_hash"))
        if recorded_hash != self.method_hash:
            raise EvaluationPersistenceError(
                f"{set_path} records harness hash {recorded_hash}, but the loaded "
                f"evaluation method hashes to {self.method_hash}"
            )
        return aggregate


@dataclass(frozen=True)
class LambdaEvaluationMethod:
    """Evaluation adapter for the pinned, non-harness λ-RLM baseline."""

    method: LambdaBaselineConfig = field(default_factory=LambdaBaselineConfig)

    @property
    def method_kind(self) -> str:
        return LAMBDA_RLM_METHOD_KIND

    @property
    def method_hash(self) -> str:
        return lambda_method_hash(self.method)

    def validate_caps(self, condition_id: str, caps: ValidationCaps) -> None:
        # λ-RLM has no editable runtime-policy surface to merge. Its runner
        # receives the experiment's exact per-run budget and timeout below and
        # validates those values against the shared breaker before execution.
        return None

    def run_set(self, request: EvaluationSetRequest) -> GovernedRoundResult:
        kwargs = round_config_kwargs(request.config)
        round_config = LambdaRoundConfig(
            round_index=EVAL_ROUND_INDEX,
            instances=request.instances,
            verifier=request.verifier,
            out_dir=request.out_dir,
            method=self.method,
            backend=kwargs["backend"],
            backend_kwargs=kwargs["backend_kwargs"],
            attempts=request.attempts,
            max_budget=request.caps.max_budget,
            max_timeout=request.caps.max_timeout,
        )
        return run_governed_lambda_round(round_config, request.breaker)

    def aggregate_set(self, set_path: Path) -> dict[str, Any]:
        """Aggregate λ-RLM's verified method artifact, manifest, and traces.

        λ-RLM has no RLM root/child distinction. For the common comparison
        field, every model call made by the method is therefore counted as a
        sub-call; this includes its task-detection call as well as bounded leaf
        calls. Skill loading is an SH-RLM harness feature and is always zero.
        """
        path = round_dir(set_path, EVAL_ROUND_INDEX)
        method_path = path / METHOD_FILE
        if not method_path.exists():
            raise EvaluationPersistenceError(
                f"{method_path} does not exist; a λ-RLM round must persist its method identity"
            )
        envelope = json.loads(method_path.read_text())
        recorded_hash = str(envelope.get("hash"))
        if recorded_hash != self.method_hash:
            raise EvaluationPersistenceError(
                f"{method_path} records method hash {recorded_hash}, but the loaded "
                f"evaluation method hashes to {self.method_hash}"
            )

        entries = load_manifest(set_path, EVAL_ROUND_INDEX)
        total_model_calls = 0
        for entry in entries:
            trace_path = verify_trace(path, entry)
            completion = RLMChatCompletion.from_dict(json.loads(trace_path.read_text()))
            total_model_calls += sum(
                usage.total_calls
                for usage in completion.usage_summary.model_usage_summaries.values()
            )

        n_runs = len(entries)
        pass_count = sum(1 for entry in entries if entry["passed"])
        total_cost = float(sum(entry["cost"] for entry in entries if entry.get("cost") is not None))
        return {
            "n_runs": n_runs,
            "pass_count": pass_count,
            "pass_rate": pass_count / n_runs if n_runs else None,
            "n_resource_terminated": sum(
                1
                for entry in entries
                if entry.get("cause") == VerifierCause.RESOURCE_TERMINATED.value
            ),
            "total_cost": total_cost,
            "mean_cost": total_cost / n_runs if n_runs else None,
            "total_sub_calls": total_model_calls,
            "mean_sub_calls": total_model_calls / n_runs if n_runs else None,
            "total_skill_loads": 0,
            "mean_skill_loads": 0.0 if n_runs else None,
        }


@dataclass(frozen=True)
class RegistryHarnessSource:
    """A condition served by a harness in the module registry (e.g. B1 = H0)."""

    registry_name: str

    def describe(self) -> dict[str, str]:
        return {"kind": "registry", "registry_name": self.registry_name}

    def load(self, out_dir: Path, work_dir: Path) -> EvaluationMethod:
        if self.registry_name not in HARNESSES:
            raise EvaluationPersistenceError(
                f"harness {self.registry_name!r} is not in the registry "
                f"({sorted(HARNESSES)}); an evaluation condition must name a real harness"
            )
        return HarnessEvaluationMethod(HARNESSES[self.registry_name])


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

    def load(self, out_dir: Path, work_dir: Path) -> EvaluationMethod:
        path = out_dir / self.relative_path
        if not path.exists():
            raise EvaluationPersistenceError(
                f"{path} does not exist; this condition evaluates the frozen harness, so "
                "the optimization experiment must have completed and frozen it first"
            )
        return HarnessEvaluationMethod(
            rematerialize_harness_envelope(
                path,
                work_dir,
                module_prefix="_frozen_",
                error=EvaluationPersistenceError,
            )
        )


@dataclass(frozen=True)
class LambdaMethodSource:
    """The pinned upstream λ-RLM method and its complete source provenance."""

    method: LambdaBaselineConfig = field(default_factory=LambdaBaselineConfig)

    def describe(self) -> dict[str, str]:
        return {
            "kind": "pinned_upstream",
            "method_kind": LAMBDA_RLM_METHOD_KIND,
            "repository": LAMBDA_RLM_UPSTREAM_REPOSITORY,
            "revision": LAMBDA_RLM_UPSTREAM_REVISION,
            "source_sha256": LAMBDA_RLM_SOURCE_SHA256,
        }

    def load(self, out_dir: Path, work_dir: Path) -> EvaluationMethod:
        return LambdaEvaluationMethod(self.method)


# The complete fixed-weight comparison grid. Ordering keeps the two RLM
# references first, then the hand-designed λ-RLM method, then learned SH-RLM.
CONDITIONS: dict[str, ConditionSource] = {
    CONDITION_B1: RegistryHarnessSource(INITIAL_INCUMBENT),
    CONDITION_H0_STAR: RegistryHarnessSource("H0*"),
    CONDITION_LAMBDA_RLM: LambdaMethodSource(),
    CONDITION_SH_RLM: FrozenHarnessSource(f"{FROZEN_DIR}/{FROZEN_HARNESS_FILENAME}"),
}

DEFAULT_CONDITIONS: tuple[str, ...] = (
    CONDITION_B1,
    CONDITION_H0_STAR,
    CONDITION_LAMBDA_RLM,
    CONDITION_SH_RLM,
)

DEFAULT_VERIFIERS: dict[str, Verifier] = {
    "graphwalks": GraphWalksVerifier(),
    "oolong_pairs": OolongPairsVerifier(),
}


def resolve_conditions(conditions: Sequence[str]) -> list[tuple[str, ConditionSource]]:
    """Resolve condition names to their method sources, in the caller's order."""
    if not conditions:
        raise ValueError("run_evaluation needs at least one condition")
    seen: set[str] = set()
    resolved: list[tuple[str, ConditionSource]] = []
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
    """Every test split the persisted manifest records, longest length first.

    Derived from the manifest rather than from the config's split plan: only
    materialized environments have files on disk, and eval reads files, never
    the plan.

    Long sets run before short ones because a condition's spend breaker can
    trip partway through, and what it truncates should be the cheap, plentiful
    measurements rather than the scarce ones. Long-run cost is superlinear in
    context length (KTD6), so a long per-run mean is both the dominant term in
    every projection and the input the report's validity gate requires at
    least one uncapped sample of; a short mean is neither. Ordering within a
    length stays alphabetical by environment, and results are keyed by
    (environment, length), so this changes only what survives a truncated
    condition -- never any measurement's value.
    """
    manifest_path = splits_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise EvaluationPersistenceError(
            f"{manifest_path} does not exist; evaluation reads only persisted split files"
        )
    manifest = json.loads(manifest_path.read_text())
    sets = [
        TestSet(environment=environment, length=length)
        for length in sorted(LENGTHS, key=LENGTHS.index, reverse=True)
        for environment in sorted(manifest["environments"])
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
    instances = read_jsonl(path)
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
    method_kind: str
    method_hash: str
    outcome: str
    spent: float
    test_sets: dict[str, dict[str, Any]]

    @property
    def over_budget(self) -> bool:
        return self.outcome == OUTCOME_OVER_BUDGET

    def to_payload(self, source: ConditionSource) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "source": source.describe(),
            "method_kind": self.method_kind,
            "method_hash": self.method_hash,
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
    conditions: list[tuple[str, ConditionSource]]
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
        source: ConditionSource,
        splits_dir: Path,
        sets: list[TestSet],
    ) -> ConditionEvaluation:
        """One condition over every test set, under one cumulative breaker."""
        condition_dir = self.out_dir / EVAL_DIR / condition_id
        method = source.load(self.out_dir, condition_dir / WORK_DIR)
        method.validate_caps(condition_id, self.caps)
        breaker = CandidateSpendBreaker(self.caps)
        aggregates = {
            test_set.set_id: self._evaluate_set(condition_id, method, breaker, splits_dir, test_set)
            for test_set in sets
        }
        return ConditionEvaluation(
            condition_id=condition_id,
            method_kind=method.method_kind,
            method_hash=method.method_hash,
            outcome=OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
            spent=breaker.spent,
            test_sets=aggregates,
        )

    def _evaluate_set(
        self,
        condition_id: str,
        method: EvaluationMethod,
        breaker: CandidateSpendBreaker,
        splits_dir: Path,
        test_set: TestSet,
    ) -> dict[str, Any]:
        """One condition x test set: a governed round, metered and aggregated."""
        split_path = splits_dir / test_set.file_name
        instances = read_instances(split_path)
        set_path = self.out_dir / EVAL_DIR / condition_id / test_set.set_id
        request = EvaluationSetRequest(
            config=self.config,
            condition_id=condition_id,
            instances=instances,
            verifier=self.verifier_for(test_set.environment),
            out_dir=set_path,
            attempts=self.config.operational.eval_repetitions,
            breaker=breaker,
            caps=self.caps,
        )
        with StageMeter(
            stage=STAGE_EVAL,
            stage_work_id=f"{EVAL_DIR}/{condition_id}/{test_set.set_id}",
            round_index=EVAL_ROUND_INDEX,
            out_path=self.usage_path,
        ) as meter:
            known = len(load_manifest(set_path, EVAL_ROUND_INDEX))
            try:
                result = method.run_set(request)
            finally:
                # Re-read from disk rather than from the (possibly unbound)
                # result: a stage that raised still persisted every run it
                # paid for, and usage the meter never sees is usage the
                # report silently undercounts (R5).
                meter.add(
                    aggregate_manifest_usage(load_manifest(set_path, EVAL_ROUND_INDEX)[known:])
                )

        usage = aggregate_manifest_usage(result.entries)
        return {
            "environment": test_set.environment,
            "length": test_set.length,
            "round_path": f"{test_set.set_id}/round_{EVAL_ROUND_INDEX:02d}",
            "split_file": test_set.file_name,
            "instances_sha256": verify_instance_identity(set_path, split_path),
            "n_instances": len(instances),
            "attempts": self.config.operational.eval_repetitions,
            "method_kind": method.method_kind,
            "method_hash": method.method_hash,
            "outcome": result.outcome,
            "skipped_run_ids": list(result.skipped_run_ids),
            **method.aggregate_set(set_path),
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
    """Evaluate fixed inference-method conditions on frozen test splits (R8).

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
            be a key of ``CONDITIONS`` (``DEFAULT_CONDITIONS`` contains
            ``b1``, ``h0_star``, ``lambda_rlm``, and ``sh_rlm``).
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
    "CONDITION_H0_STAR",
    "CONDITION_LAMBDA_RLM",
    "CONDITION_SH_RLM",
    "DEFAULT_CONDITIONS",
    "DEFAULT_VERIFIERS",
    "EVAL_DIR",
    "EVAL_SUMMARY_FILENAME",
    "EVAL_SUMMARY_FORMAT",
    "ConditionEvaluation",
    "ConditionSource",
    "EvaluationMethod",
    "EvaluationPersistenceError",
    "EvaluationResult",
    "EvaluationSetRequest",
    "FrozenHarnessSource",
    "HarnessEvaluationMethod",
    "LambdaEvaluationMethod",
    "LambdaMethodSource",
    "RegistryHarnessSource",
    "TestSet",
    "resolve_conditions",
    "run_evaluation",
    "test_sets",
]
