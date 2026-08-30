"""The outer round orchestrator: T optimization rounds, crash-safe, then freeze (R7).

``run_experiment`` chains the existing stage functions -- governed mining
(``run_governed_round``), evidence mining (``mine_round`` + ``write_bundle``),
proposal (``propose_round``), and validation (``validate_round``) -- into
complete rounds under the T/patience stopping rule, and freezes the final
incumbent to ``<out_dir>/sh_rlm/harness.json``. Stage modules stay untouched
(KTD2): everything here composes them.

Directory contract under ``out_dir``::

    config.json               # profile + identity hash (R3), checked every invocation
    splits/                   # materialize_splits output (verified on re-invocation)
    stage_usage.jsonl         # one StageMeter record per stage attempt (R5)
    opt/round_NN/
        mining/round_NN/      # run_round + mine_round artifacts, incl. bundle.json
            evidence_complete.json  # stage marker: the whole bundle triplet landed
        proposals/            # <candidate_id>/proposal.json directories
        proposals_complete.json  # stage marker: the proposal stage's sealed outcome
        validation/round_NN/  # validate_round artifacts, incl. decision.json
        work/                 # scratch for generated-source modules
        round.json            # stage marker: the round's final outcome
    sh_rlm/harness.json       # the frozen final harness (write_harness_json envelope)
    analysis/<UTC stamp>/     # post-round analysis snapshots (KTD4); never read
                              # by the loop, and never experiment state

Identity enforcement (R3). ``identity_hash(config)`` is persisted in the root
``config.json`` on the first invocation and compared on every later one;
a mismatch raises ``IdentityMismatchError`` before any run executes, so a
changed temperature, repetition count, model, or dataset pin can never
silently mix state with an existing experiment.

Crash-safe resume. Every stage boundary is decidable from disk:

1. Mining is complete when its manifest holds every configured run id --
   ``run_governed_round`` already resumes persist-first, so re-invocation
   executes only the missing runs (under the cumulative spend breaker, R12;
   a tripped breaker raises ``MiningBudgetExceededError`` after persisting the
   partial, resumable state).
2. The evidence stage is complete when ``evidence_complete.json`` exists;
   resume then reads ``bundle.json`` back instead of re-mining (the attribution
   cache, persisted at the configured operational path, makes any re-mine
   replay-free anyway). ``bundle.json`` itself is NOT the marker:
   ``shrlm.optimization.bundle.write_bundle`` publishes it atomically *before*
   it writes ``records.jsonl`` and ``attributions.jsonl``, so a crash in that
   window leaves a readable bundle beside missing or truncated audit files.
   The marker is written by this module only after all three artifacts are
   re-read and their line counts checked against what mining produced. An
   unmarked bundle is re-mined with the persisted ``created_at``, so the
   re-mine reproduces the same bytes and ``write_bundle``'s non-clobber guard
   confirms it rather than refusing the resume.
3. The proposal stage is complete when ``proposals_complete.json`` exists.
   ``propose_round`` runs over a PERSISTED ``ProposalCache``
   (``operational.proposal_cache_path``), so a crash-and-rerun replays the
   same responses and cannot mint a divergent candidate set; the marker seals
   the (possibly empty) candidate set so an empty proposal round is
   distinguishable from an interrupted one.
4. The round is complete when ``round.json`` exists. It records the outcome
   (promoted or not, which hash), distinguishing "round complete, no
   promotion" from "round incomplete". For a completed round, resume never
   re-executes anything: the incumbent is rebuilt from the promotion ledger by
   rematerializing the promoted harness's persisted ``harness.json`` envelope
   through ``materialize_harness`` and verifying its content hash -- the
   round-trip the gate test in ``tests/experiment/test_orchestrator.py`` pins,
   merged promotions included.

Operational cache paths (``operational.attribution_cache_path`` /
``proposal_cache_path``) resolve relative to ``out_dir`` unless absolute, so
an experiment directory is self-contained and resumable as a unit.

Deferred redesign: a resumable mining budget allocation. ``run_governed_round``
charges every already-persisted run into each fresh ``CandidateSpendBreaker``,
so once a round's cumulative persisted mining spend exceeds
``caps.candidate_budget``, every re-invocation re-charges it, trips before the
pending runs execute, and raises ``MiningBudgetExceededError`` again -- the
round cannot be completed in that output directory, and because caps are
identity keys (R3) the budget cannot be raised in place. That is the breaker's
contract working as designed (cumulative spend over persisted runs; a
per-invocation budget would let an operator spend the cap again on every
re-invocation), so the fix that would make such a round resumable -- persisting
a per-round budget allocation (spend already accounted for, remaining
allowance) and charging an invocation only for what it adds -- is deliberately
NOT smuggled in here: it changes what ``candidate_budget`` means for every
stage that shares a breaker (validation and evaluation included) and belongs in
its own change. Until then the failure is transparent rather than silent: the
error states that the round is uncompletable in this directory, why, and the
operator's two options (a larger budget in a fresh out_dir, or a smaller
scale).

Post-round analysis (KTD4). After every EXECUTED round the loop refreshes the
aggregation snapshots under ``analysis/`` from what it has just persisted, so a
long study's outputs never go stale. That refresh is strictly one-way: it reads
persisted artifacts, writes only under ``analysis/``, and cannot fail, block, or
alter a round -- ``_run_post_round_analysis`` swallows every analysis failure,
including its own imports. It does NOT swallow an operator interrupt: a
``KeyboardInterrupt`` is recorded in the snapshot's provenance and then re-raised,
because Ctrl-C means stop the experiment, not skip the analysis. Nothing in the
loop ever reads ``analysis/`` back.

Single-threaded, main thread only: the SIGALRM hard-deadline backstop in
``shrlm.optimization.costs`` binds only there (POSIX main thread), and this
module never spawns threads around run execution.
"""

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from shrlm.environments.graphwalks import GraphWalksSubVerifier, GraphWalksVerifier
from shrlm.environments.oolong import (
    OolongSubVerifier,
    OolongVerifier,
    continuous_score,
)
from shrlm.experiment.config import (
    GOVERNED_ROUND_KEYS,
    ExperimentConfig,
    backend_kwargs_for,
    evaluation_config_kwargs,
    identity_hash,
    promotion_config,
    proposer_config,
    round_config_kwargs,
    validation_caps,
)
from shrlm.experiment.errors import ExperimentError
from shrlm.experiment.splits import LoaderFn, materialize_splits, split_file_name
from shrlm.experiment.usage import (
    STAGE_USAGE_FILE,
    StageMeter,
    UsageTotals,
    aggregate_manifest_usage,
    read_jsonl,
)
from shrlm.harness_identity import harness_hash, write_harness_json
from shrlm.optimization.attribution import AttributionCache, LLMAttributor
from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    BUNDLE_FILENAME,
    RECORDS_FILENAME,
    round_dir,
    write_bundle,
)
from shrlm.optimization.candidates import CandidateRejection, materialize_harness
from shrlm.optimization.costs import (
    CandidateSpendBreaker,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import (
    RoundConfig,
    build_round_rlm,
    execute_run,
    load_manifest,
    mine_round,
)
from shrlm.optimization.mining import WeaknessMiner
from shrlm.optimization.promotion import DECISION_PROMOTED, PromotionConfig
from shrlm.optimization.proposal import (
    ProposalBudgetExhausted,
    ProposalCache,
    load_passing_behaviors,
    propose_round,
)
from shrlm.optimization.types import SubVerifier, Verifier
from shrlm.optimization.validation import (
    SPLIT_HELDIN,
    EvaluationConfig,
    ValidationRound,
    ValidationSplits,
    load_promotion_ledger,
    validate_round,
)
from shrlm.rlm_harness import HARNESSES, Harness

if TYPE_CHECKING:  # analysis_io imports this module, so its types are annotations only
    from shrlm.experiment.analysis_io import Snapshot

CONFIG_FILENAME = "config.json"
CONFIG_FORMAT = "shrlm-experiment-config/v1"

OPT_DIR = "opt"
MINING_DIR = "mining"
PROPOSALS_DIR = "proposals"
VALIDATION_DIR = "validation"
WORK_DIR = "work"

EVIDENCE_MARKER_FILENAME = "evidence_complete.json"
EVIDENCE_MARKER_FORMAT = "shrlm-evidence-complete/v1"
PROPOSALS_MARKER_FILENAME = "proposals_complete.json"
PROPOSALS_MARKER_FORMAT = "shrlm-proposals-complete/v1"
ROUND_MARKER_FILENAME = "round.json"
ROUND_MARKER_FORMAT = "shrlm-experiment-round/v1"

FROZEN_DIR = "sh_rlm"
FROZEN_HARNESS_FILENAME = "harness.json"

STAGE_MINING = "mining"
STAGE_ATTRIBUTION = "attribution"
STAGE_PROPOSAL = "proposal"
STAGE_VALIDATION = "validation"

STOP_MAX_ROUNDS = "max_rounds"
STOP_PATIENCE = "patience"

# The registry floor. ``config.loop.initial_harness`` decides which registry
# harness the loop actually starts from (default: this); evaluation's B1
# condition stays bound to this constant so B1 always means the H0 floor.
INITIAL_INCUMBENT = "H0"

# The optimization loop mines and validates one environment's held-in/held-out
# splits. Which environment (and its verifier pair) is ``config.loop.environment``
# -- ``graphwalks`` (default) or ``oolong_synth`` -- resolved by
# ``resolve_env_binding``. The length label is ``short`` for both (OOLONG-synth's
# held-in/held-out/test partition is one length-diverse pool, not a short/long
# pair).
SPLIT_LENGTH = "short"
ROLE_HELD_IN = "held_in"
ROLE_HELD_OUT = "held_out"

# The default source environment. The loop itself reads
# ``resolve_env_binding(config).name`` (``config.loop.environment``); this
# constant is the fallback the report/analysis pipeline
# (``shrlm.experiment.report``) uses to locate the optimization split bucket,
# and it stays "graphwalks" -- the shipped default and the environment those
# analyses were written for.
SPLIT_ENVIRONMENT = "graphwalks"

# OOLONG-real generalization check (non-gated): its split role and metering
# stage. Written under ``opt/round_NN/real_check/`` and, for the final
# incumbent, ``real_check/final/``.
REAL_CHECK_DIR = "real_check"
REAL_CHECK_ENVIRONMENT = "oolong_real"
REAL_CHECK_LENGTH = "short"
ROLE_REAL_CHECK = "check"
REAL_CHECK_SUMMARY_FILENAME = "summary.json"
REAL_CHECK_SUMMARY_FORMAT = "shrlm-oolong-real-check/v1"
STAGE_REAL_CHECK = "real_check"


# The verifier dotted paths handed to validation subject workers, per environment.
GRAPHWALKS_VERIFIER_FACTORY = "shrlm.environments.graphwalks:GraphWalksVerifier"
OOLONG_SYNTH_VERIFIER_FACTORY = "shrlm.environments.oolong:make_synth_verifier"


@dataclass(frozen=True)
class EnvBinding:
    """Which environment the optimization loop mines and validates this run.

    ``name`` / ``length`` locate the held-in/held-out/test split files;
    ``verifier`` / ``sub_verifier`` are the defaults ``run_experiment`` installs
    when the caller passes neither; ``verifier_factory`` is the dotted path a
    parallel validation stage rebuilds the verifier from in its child processes.
    """

    name: str
    length: str
    verifier: Verifier
    sub_verifier: SubVerifier | None
    verifier_factory: str


def resolve_env_binding(config: ExperimentConfig) -> EnvBinding:
    """The ``EnvBinding`` for ``config.loop.environment``.

    ``config.load_config`` has already validated the value against
    ``SELECTABLE_ENVIRONMENTS``; an unexpected value here is a programming error.
    """
    environment = config.loop.environment
    if environment == "graphwalks":
        return EnvBinding(
            name="graphwalks",
            length=SPLIT_LENGTH,
            verifier=GraphWalksVerifier(),
            sub_verifier=GraphWalksSubVerifier(),
            verifier_factory=GRAPHWALKS_VERIFIER_FACTORY,
        )
    if environment == "oolong_synth":
        return EnvBinding(
            name="oolong_synth",
            length=SPLIT_LENGTH,
            verifier=OolongVerifier(task_set="synth"),
            sub_verifier=OolongSubVerifier(),
            verifier_factory=OOLONG_SYNTH_VERIFIER_FACTORY,
        )
    raise ValueError(f"resolve_env_binding: unsupported loop.environment {environment!r}")


class ExperimentPersistenceError(ExperimentError):
    """Persisted experiment state contradicts itself or the configuration."""


class IdentityMismatchError(ExperimentPersistenceError):
    """The configured identity hash does not match the experiment's (R3)."""


class MiningBudgetExceededError(ExperimentError):
    """The mining spend breaker tripped; partial state is persisted and resumable."""


@dataclass(frozen=True)
class RoundOutcome:
    """One completed round's recorded outcome (the ``round.json`` payload)."""

    round_index: int
    promoted: bool
    promoted_harness_hash: str | None
    has_ledger: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": ROUND_MARKER_FORMAT,
            "round": self.round_index,
            "promoted": self.promoted,
            "promoted_harness_hash": self.promoted_harness_hash,
            "has_ledger": self.has_ledger,
        }


@dataclass(frozen=True)
class ExperimentResult:
    """What one ``run_experiment`` invocation concluded."""

    out_dir: Path
    rounds: list[RoundOutcome]
    stopped: str
    final_harness: Harness
    final_harness_hash: str
    frozen_path: Path


# ---------------------------------------------------------------------------
# Persistence primitives
# ---------------------------------------------------------------------------


def _persist_once(path: Path, payload: dict[str, Any], diverging: str) -> None:
    """Write a JSON payload exactly once, in the bundle's non-clobbering style.

    Byte-identical rewrites are no-ops; different bytes raise ``diverging``.
    The write goes through a same-directory tmp file and ``os.replace`` so a
    crash mid-write never leaves a truncated marker.
    """
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() == text:
            return
        raise ExperimentPersistenceError(diverging)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _load_marker(path: Path, expected_format: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("format") != expected_format:
        raise ExperimentPersistenceError(f"{path} is not a {expected_format} document")
    return payload


def _interrupted_bundle_created_at(bundle_path: Path) -> str | None:
    """The ``created_at`` of a bundle an interrupted evidence stage left behind.

    ``write_bundle`` refuses any rewrite that differs from the persisted bundle
    in a single byte, timestamp included, so a re-mine over an unmarked bundle
    must reproduce the timestamp already on disk -- otherwise the very crash
    window this resume path exists for would raise instead of healing. No
    bundle on disk means a fresh stage, which mints its own timestamp.
    """
    if not bundle_path.exists():
        return None
    return str(json.loads(bundle_path.read_text())["created_at"])


def _evidence_marker_payload(
    mining_round_path: Path, round_index: int, *, n_records: int, n_attributions: int
) -> dict[str, Any]:
    """Verify the whole evidence triplet landed; describe it for the marker.

    ``bundle.json`` is published (atomically) before ``records.jsonl`` and
    ``attributions.jsonl`` are written, so only a re-read of all three proves
    the stage finished: both JSON-lines files are parsed and their line counts
    checked against what mining produced, and the bundle id is recorded so the
    marker names the evidence it certifies.
    """
    bundle_id = str(json.loads((mining_round_path / BUNDLE_FILENAME).read_text())["bundle_id"])
    for file_name, expected in (
        (RECORDS_FILENAME, n_records),
        (ATTRIBUTIONS_FILENAME, n_attributions),
    ):
        actual = len(read_jsonl(mining_round_path / file_name))
        if actual != expected:
            raise ExperimentPersistenceError(
                f"{mining_round_path / file_name} holds {actual} line(s), but round "
                f"{round_index}'s mining produced {expected}; refusing to mark evidence "
                "complete over a missing or truncated audit file."
            )
    return {
        "format": EVIDENCE_MARKER_FORMAT,
        "round": round_index,
        "bundle_id": bundle_id,
        "n_records": n_records,
        "n_attributions": n_attributions,
    }


def check_identity(config: ExperimentConfig, out_dir: Path | str) -> str:
    """Persist-or-verify the experiment's config identity (R3). Returns the hash.

    Raises:
        IdentityMismatchError: When ``<out_dir>/config.json`` records a
            different identity hash -- a behavior-changing config value moved
            under an existing experiment, and resuming would silently mix two
            experiments' state.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = identity_hash(config)
    path = out / CONFIG_FILENAME
    if path.exists():
        recorded = _load_marker(path, CONFIG_FORMAT)
        if str(recorded["identity_hash"]) != expected:
            raise IdentityMismatchError(
                f"{path} records identity hash {recorded['identity_hash']}, but the "
                f"configured experiment hashes to {expected}; a behavior-changing config "
                "value changed under an existing experiment. Refusing to resume -- start "
                "a fresh out_dir (or restore the original configuration)."
            )
        return expected
    _persist_once(
        path,
        {"format": CONFIG_FORMAT, "profile": config.profile, "identity_hash": expected},
        f"{path} changed while being written",  # unreachable: guarded by exists() above
    )
    return expected


def _operational_path(out_dir: Path, raw: str) -> Path:
    """An operational cache path, resolved relative to the experiment directory."""
    path = Path(raw)
    return path if path.is_absolute() else out_dir / path


def _read_split(splits_dir: Path, environment: str, length: str, role: str) -> list[dict[str, Any]]:
    path = splits_dir / split_file_name(environment, length, role)
    instances = read_jsonl(path)
    if not instances:
        raise ExperimentPersistenceError(f"{path} holds no instances; cannot run a round on it")
    return instances


def experiment_round_dir(out_dir: Path | str, round_index: int) -> Path:
    """One optimization round's directory: ``<out_dir>/opt/round_NN``."""
    return round_dir(Path(out_dir) / OPT_DIR, round_index)


def _freeze_harness(harness: Harness, out_dir: Path) -> Path:
    """Write the final harness envelope, refusing to clobber a diverging one."""
    path = out_dir / FROZEN_DIR / FROZEN_HARNESS_FILENAME
    final_hash = harness_hash(harness)
    if path.exists():
        recorded = json.loads(path.read_text())
        if str(recorded["hash"]) != final_hash:
            raise ExperimentPersistenceError(
                f"{path} already freezes harness {recorded['hash']}, but this run "
                f"concluded with {final_hash}; refusing to overwrite a frozen harness."
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_harness_json(harness, path)
    return path


# ---------------------------------------------------------------------------
# Persisted harness envelopes -> live harnesses
# ---------------------------------------------------------------------------


def rematerialize_harness_envelope(
    envelope_path: Path,
    work_dir: Path,
    *,
    module_prefix: str,
    error: type[ExperimentError],
    expected_hash: str | None = None,
) -> Harness:
    """Rebuild a persisted ``harness.json`` envelope and verify its identity.

    The one place a persisted harness becomes a live one again -- the
    orchestrator's resume path and the evaluation runner's frozen condition
    both go through it. The envelope is rematerialized via
    ``materialize_harness`` into ``work_dir/<module_prefix><hash[:16]>.py`` and
    the rebuilt harness re-hashed against the hash the envelope records, so an
    envelope edited after it was written -- or one whose surfaces do not
    reconstruct -- is refused rather than silently used as something else.

    Args:
        envelope_path: The persisted ``write_harness_json`` envelope.
        work_dir: Where the generated module is materialized; created if absent.
        module_prefix: Filename prefix for that module (the caller's context).
        error: The exception class raised on any mismatch. Callers pass their
            own: the same failed check means a tampered freeze in one context
            and contradictory round state in the other.
        expected_hash: When given, the hash the caller independently expects;
            an envelope recording a different one is refused before anything is
            materialized. ``None`` verifies only that the envelope round-trips
            to its own recorded hash.

    Returns:
        The rematerialized harness, hash-verified.
    """
    envelope = json.loads(envelope_path.read_text())
    recorded = str(envelope["hash"])
    if expected_hash is not None and recorded != expected_hash:
        raise error(
            f"{envelope_path} records harness hash {recorded}, but {expected_hash} was "
            "expected; refusing to rebuild a harness from a contradictory envelope"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    harness = materialize_harness(
        envelope["harness"], work_dir / f"{module_prefix}{recorded[:16]}.py"
    )
    rebuilt = harness_hash(harness)
    if rebuilt != recorded:
        raise error(
            f"rematerializing {envelope_path} produced harness hash {rebuilt}, not the "
            f"recorded {recorded}; the envelope does not round-trip (it was modified, or "
            "its surfaces are not reconstructible). Refusing to use a harness whose "
            "identity cannot be verified."
        )
    return harness


# ---------------------------------------------------------------------------
# Resume: rebuild the incumbent from the persisted ledger chain
# ---------------------------------------------------------------------------


def _rematerialize_promoted(
    validation_round_path: Path, promoted_hash: str, work_dir: Path
) -> Harness:
    """Rebuild a completed round's promoted harness from its persisted envelope.

    The promoted subject's ``harness.json`` (written by ``run_round`` under its
    held-in split directory) is located through the ledger record's audit
    links, rematerialized via ``materialize_harness``, and hash-verified --
    the round-trip the gate test pins, covering merged promotions whose live
    harness exists nowhere else on disk.
    """
    records, decision = load_promotion_ledger(validation_round_path)
    promoted = [record for record in records if record["decision"] == DECISION_PROMOTED]
    if len(promoted) != 1:
        raise ExperimentPersistenceError(
            f"{validation_round_path} ledger holds {len(promoted)} promoted record(s) for a "
            "round marked promoted; exactly one is required to rebuild the incumbent"
        )
    record = promoted[0]
    if str(decision["promoted_harness_hash"]) != promoted_hash:
        raise ExperimentPersistenceError(
            f"{validation_round_path} decision names promoted hash "
            f"{decision['promoted_harness_hash']}, but the round marker recorded "
            f"{promoted_hash}; the persisted round state contradicts itself"
        )
    envelope_path = validation_round_path / str(record["links"]["splits"][SPLIT_HELDIN]["harness"])
    return rematerialize_harness_envelope(
        envelope_path,
        work_dir,
        module_prefix="_promoted_",
        error=ExperimentPersistenceError,
        expected_hash=promoted_hash,
    )


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _Experiment:
    """One invocation's shared state; ``run`` is the whole loop."""

    config: ExperimentConfig
    out_dir: Path
    verifier: Verifier
    sub_verifier: SubVerifier | None
    attributor_lm: BaseLM | None
    proposer_lm: BaseLM | None
    loaders: dict[str, LoaderFn] | None
    verifier_factory: str | None = None
    client_factory: tuple[str, dict[str, Any]] | None = None
    caps: ValidationCaps = field(init=False)
    pconfig: PromotionConfig = field(init=False)
    binding: EnvBinding = field(init=False)
    splits: ValidationSplits = field(init=False)
    splits_dir: Path = field(init=False)
    usage_path: Path = field(init=False)
    prior_history: list[tuple[list[dict[str, Any]], dict[str, Any]]] = field(init=False)

    def __post_init__(self) -> None:
        self.caps = validation_caps(self.config)
        self.pconfig = promotion_config(self.config)
        self.binding = resolve_env_binding(self.config)
        self.usage_path = self.out_dir / STAGE_USAGE_FILE
        self.prior_history = []
        self.verifier_factory = self._resolve_verifier_factory()

    def _resolve_verifier_factory(self) -> str | None:
        """The dotted path validation children rebuild the verifier from (KTD6).

        Only a parallel validation stage (``operational.validation_workers > 1``)
        needs one. A verifier that is the selected environment's default type
        maps to that environment's factory path; an injected verifier must come
        with an explicit ``verifier_factory``, and the mismatch is a
        configuration error raised here -- before any directory is created or any
        run executes -- not deep inside stage 4.
        """
        if self.verifier_factory is not None:
            return self.verifier_factory
        if self.config.operational.validation_workers <= 1:
            return None
        if type(self.verifier) is GraphWalksVerifier:
            return GRAPHWALKS_VERIFIER_FACTORY
        if isinstance(self.verifier, OolongVerifier) and self.verifier.task_set == "synth":
            return OOLONG_SYNTH_VERIFIER_FACTORY
        raise ValueError(
            f"operational.validation_workers={self.config.operational.validation_workers} "
            "evaluates validation subjects in child processes, which rebuild the verifier "
            f"from a dotted path; the injected verifier {type(self.verifier).__name__} has "
            "none. Pass run_experiment(verifier_factory=...) or set validation_workers = 1."
        )

    # -- lazily built caller-held clients (only when their stage must run) --

    def _attributor(self) -> BaseLM:
        if self.attributor_lm is None:
            endpoint = self.config.backends.attributor
            self.attributor_lm = get_client(
                endpoint.backend, backend_kwargs_for(self.config, "attributor")
            )
        return self.attributor_lm

    def _proposer(self) -> BaseLM:
        if self.proposer_lm is None:
            endpoint = self.config.backends.proposer
            self.proposer_lm = get_client(
                endpoint.backend, backend_kwargs_for(self.config, "proposer")
            )
        return self.proposer_lm

    # -- the loop --

    def run(self) -> ExperimentResult:
        check_identity(self.config, self.out_dir)
        splits_dir = materialize_splits(self.config, self.out_dir, loaders=self.loaders)
        self.splits_dir = splits_dir
        self.splits = ValidationSplits(
            heldin=_read_split(splits_dir, self.binding.name, self.binding.length, ROLE_HELD_IN),
            heldout=_read_split(splits_dir, self.binding.name, self.binding.length, ROLE_HELD_OUT),
        )

        incumbent: Harness = HARNESSES[self.config.loop.initial_harness]
        rounds: list[RoundOutcome] = []
        without_promotion = 0
        stopped = STOP_MAX_ROUNDS
        # A replay phase refreshes the analyses ONCE, when it catches up --
        # never per replayed round. Every replayed round is already finished,
        # so a per-round refresh would run one full pass over the tree for each
        # of them before any new work happened, and only the last of those
        # passes would carry current information.
        replay_unanalyzed = False
        for round_index in range(1, self.config.loop.t + 1):
            round_path = experiment_round_dir(self.out_dir, round_index)
            marker_path = round_path / ROUND_MARKER_FILENAME
            replayed = marker_path.exists()
            if replayed:
                outcome = self._replay_round(round_path, round_index)
            else:
                outcome = self._execute_round(round_path, round_index, incumbent)
            rounds.append(outcome)
            if replayed:
                replay_unanalyzed = True
            else:
                _run_post_round_analysis(self.out_dir)
                replay_unanalyzed = False
            if outcome.promoted:
                incumbent = self._promoted_harness(round_path, round_index, outcome)
                without_promotion = 0
            else:
                without_promotion += 1
            if outcome.has_ledger:
                validation_round_path = round_dir(round_path / VALIDATION_DIR, round_index)
                self.prior_history.append(load_promotion_ledger(validation_round_path))
            # Non-gated OOLONG-real generalization check on the round-end
            # incumbent (after any promotion is applied). Never touches the
            # outcome, the ledger, or the patience counter.
            if not replayed and self._real_check_due(round_index):
                self._real_generalization_check(round_index, f"round_{round_index:02d}", incumbent)
            if without_promotion >= self.config.loop.patience:
                stopped = STOP_PATIENCE
                break

        # The loop ended while still replaying: a resume that executed nothing
        # (the experiment was already finished, or stopped on patience) still
        # gets its one catch-up refresh here.
        if replay_unanalyzed:
            _run_post_round_analysis(self.out_dir)

        if self._real_check_enabled():
            self._real_generalization_check(self.config.loop.t + 1, "final", incumbent)

        frozen_path = _freeze_harness(incumbent, self.out_dir)
        return ExperimentResult(
            out_dir=self.out_dir,
            rounds=rounds,
            stopped=stopped,
            final_harness=incumbent,
            final_harness_hash=harness_hash(incumbent),
            frozen_path=frozen_path,
        )

    def _replay_round(self, round_path: Path, round_index: int) -> RoundOutcome:
        """A completed round's recorded outcome; nothing re-executes."""
        payload = _load_marker(round_path / ROUND_MARKER_FILENAME, ROUND_MARKER_FORMAT)
        if int(payload["round"]) != round_index:
            raise ExperimentPersistenceError(
                f"{round_path} marker records round {payload['round']}, expected {round_index}"
            )
        hash_value = payload["promoted_harness_hash"]
        return RoundOutcome(
            round_index=round_index,
            promoted=bool(payload["promoted"]),
            promoted_harness_hash=str(hash_value) if hash_value is not None else None,
            has_ledger=bool(payload["has_ledger"]),
        )

    def _promoted_harness(
        self, round_path: Path, round_index: int, outcome: RoundOutcome
    ) -> Harness:
        """The promoted incumbent, rebuilt from the persisted validation round."""
        assert outcome.promoted_harness_hash is not None  # promoted implies a hash
        validation_round_path = round_dir(round_path / VALIDATION_DIR, round_index)
        return _rematerialize_promoted(
            validation_round_path, outcome.promoted_harness_hash, round_path / WORK_DIR
        )

    # -- one round, stage by stage --

    def _execute_round(
        self, round_path: Path, round_index: int, incumbent: Harness
    ) -> RoundOutcome:
        mining_parent = round_path / MINING_DIR
        mining_round_path = round_dir(mining_parent, round_index)

        self._mine_runs(mining_parent, round_index, incumbent)
        bundle = self._evidence_bundle(mining_parent, mining_round_path, round_index)
        proposals_dir = self._proposals(
            round_path, mining_round_path, round_index, incumbent, bundle
        )
        validation = self._validate(round_path, round_index, incumbent, proposals_dir)

        outcome = RoundOutcome(
            round_index=round_index,
            promoted=validation.promoted,
            promoted_harness_hash=validation.promoted_harness_hash,
            has_ledger=validation.ledger is not None,
        )
        _persist_once(
            round_path / ROUND_MARKER_FILENAME,
            outcome.to_payload(),
            f"{round_path / ROUND_MARKER_FILENAME} already records a diverging outcome for "
            f"round {round_index}; refusing to rewrite a completed round's history.",
        )
        return outcome

    def _mine_runs(self, mining_parent: Path, round_index: int, incumbent: Harness) -> None:
        """Stage 1: the held-in mining runs, governed by the spend breaker (R12)."""
        limits = governed_limits("incumbent", incumbent.runtime_policy, self.caps)
        if isinstance(limits, CandidateRejection):
            raise ExperimentPersistenceError(
                f"the incumbent violates the experiment-owned caps ({limits.reason}); an "
                "incumbent that cannot run under the mining limits is a misconfigured "
                "experiment, not a rejectable candidate"
            )
        kwargs = round_config_kwargs(self.config)
        for governed_key in GOVERNED_ROUND_KEYS:
            kwargs.pop(governed_key)
        mining_config = RoundConfig(
            round_index=round_index,
            harness=incumbent,
            instances=self.splits.heldin,
            verifier=self.verifier,
            out_dir=mining_parent,
            attempts=self.config.loop.m,
            **kwargs,
            **limits,
        )
        with StageMeter(
            stage=STAGE_MINING,
            stage_work_id=f"round_{round_index:02d}/{STAGE_MINING}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            known = len(load_manifest(mining_parent, round_index))
            breaker = CandidateSpendBreaker(self.caps)
            try:
                result = run_governed_round(mining_config, breaker)
            finally:
                # Re-read from disk rather than from the (possibly unbound)
                # result: a stage that raised still persisted every run it
                # paid for, and usage the meter never sees is usage the
                # report silently undercounts (R5).
                meter.add(
                    aggregate_manifest_usage(load_manifest(mining_parent, round_index)[known:])
                )
            if result.over_budget:
                raise MiningBudgetExceededError(
                    f"round {round_index} mining tripped the cumulative spend breaker at "
                    f"{result.spent:.6f} USD against candidate_budget "
                    f"{self.caps.candidate_budget}; {len(result.skipped_run_ids)} run(s) were "
                    f"skipped. Every completed run is persisted, but this round cannot be "
                    f"completed in {self.out_dir}: each invocation charges the already-persisted "
                    "mining spend to a fresh breaker, so the breaker trips again before the "
                    "pending runs execute, and caps.candidate_budget is identity-protected (R3) "
                    "-- raising it under this directory is refused by the identity check. "
                    "Options: (a) raise caps.candidate_budget and run in a FRESH out_dir, or "
                    "(b) lower the scale (splits.n_in, loop.m, caps.max_budget) and run in a "
                    "fresh out_dir. A resumable per-round budget allocation is a documented "
                    "deferral -- see this module's docstring."
                )

    def _evidence_bundle(
        self, mining_parent: Path, mining_round_path: Path, round_index: int
    ) -> dict[str, Any]:
        """Stage 2: mine the persisted round into the evidence triplet.

        Gated on this module's ``evidence_complete.json``, never on
        ``bundle.json`` alone: ``write_bundle`` publishes the bundle before the
        records and attributions files, so the bundle is a completion marker
        for nothing (see the module docstring's resume contract). An unmarked
        bundle is re-mined with its persisted ``created_at`` so the rewrite
        reproduces it byte for byte.
        """
        bundle_path = mining_round_path / BUNDLE_FILENAME
        marker_path = mining_round_path / EVIDENCE_MARKER_FILENAME
        if marker_path.exists():
            _load_marker(marker_path, EVIDENCE_MARKER_FORMAT)
            return json.loads(bundle_path.read_text())
        cache_path = _operational_path(self.out_dir, self.config.operational.attribution_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with StageMeter(
            stage=STAGE_ATTRIBUTION,
            stage_work_id=f"round_{round_index:02d}/{STAGE_ATTRIBUTION}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            attributor_lm = self._attributor()
            meter.watch(attributor_lm)
            miner = WeaknessMiner(
                verifier=self.verifier,
                attributor=LLMAttributor(
                    attributor_lm, cache=AttributionCache(path=str(cache_path))
                ),
                sub_verifier=self.sub_verifier,
            )
            result = mine_round(
                mining_parent,
                round_index,
                miner,
                split_id=split_file_name(
                    self.binding.name, self.binding.length, ROLE_HELD_IN
                ).removesuffix(".jsonl"),
                created_at=_interrupted_bundle_created_at(bundle_path),
            )
            if result.errors:
                raise ExperimentPersistenceError(
                    f"round {round_index} mining checkpointed with "
                    f"{len(result.errors)} unattributed run(s) (attributor transport "
                    "failures); refusing to persist a bundle built on missing "
                    "attributions -- re-invoke to retry them (completed attributions "
                    "replay from the cache)."
                )
            write_bundle(
                result.bundle,
                result.records,
                out_dir=str(mining_parent),
                raw_attributions=result.raw_attributions,
            )
        _persist_once(
            marker_path,
            _evidence_marker_payload(
                mining_round_path,
                round_index,
                n_records=len(result.records),
                n_attributions=len(result.raw_attributions),
            ),
            f"{marker_path} already marks a diverging evidence bundle for round "
            f"{round_index}; the persisted attribution cache should have made this "
            "impossible -- refusing to mix two mining outcomes.",
        )
        return json.loads(bundle_path.read_text())

    def _proposals(
        self,
        round_path: Path,
        mining_round_path: Path,
        round_index: int,
        incumbent: Harness,
        bundle: dict[str, Any],
    ) -> Path:
        """Stage 3: propose candidates, sealed by ``proposals_complete.json``."""
        proposals_dir = round_path / PROPOSALS_DIR
        marker_path = round_path / PROPOSALS_MARKER_FILENAME
        if marker_path.exists():
            _load_marker(marker_path, PROPOSALS_MARKER_FORMAT)
            return proposals_dir
        cache_path = _operational_path(self.out_dir, self.config.operational.proposal_cache_path)
        with StageMeter(
            stage=STAGE_PROPOSAL,
            stage_work_id=f"round_{round_index:02d}/{STAGE_PROPOSAL}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            proposer_lm = self._proposer()
            meter.watch(proposer_lm)
            try:
                result = propose_round(
                    bundle,
                    incumbent,
                    proposer_lm,
                    proposals_dir,
                    round_index=round_index,
                    passing_behaviors=load_passing_behaviors(mining_round_path),
                    prior_history=self.prior_history,
                    config=proposer_config(self.config),
                    cache=ProposalCache(path=str(cache_path)),
                    workdir=round_path / WORK_DIR,
                )
            except ProposalBudgetExhausted as exc:
                # The proposer spent its output budget on reasoning (R6/KTD3):
                # deterministic for the prompt, so re-asking only re-bills it,
                # and letting it escape would re-ask on every resume. Seal the
                # stage as a failure with zero candidates; validation then sees
                # an empty proposals directory and the round closes unpromoted,
                # exactly as a round whose proposer wrote nothing.
                print(
                    f"round {round_index}: proposal stage failed with zero candidates "
                    f"({exc}); sealing {marker_path.name} and continuing",
                    file=sys.stderr,
                )
                payload = {
                    "format": PROPOSALS_MARKER_FORMAT,
                    "round": round_index,
                    "candidate_ids": [],
                    "prompt_sha256": None,
                    "skipped_patterns": [],
                    "n_materialization_failures": 0,
                    "stage_failure": {
                        "kind": "budget_exhausted",
                        "error": str(exc),
                        "n_attempts": len(exc.attempts),
                    },
                }
            else:
                payload = {
                    "format": PROPOSALS_MARKER_FORMAT,
                    "round": round_index,
                    "candidate_ids": sorted(written.candidate_id for written in result.written),
                    "prompt_sha256": result.prompt_sha256,
                    "skipped_patterns": list(result.skipped_patterns),
                    "n_materialization_failures": len(result.materialization_failures),
                }
        _persist_once(
            marker_path,
            payload,
            f"{marker_path} already seals a diverging candidate set for round "
            f"{round_index}; the persisted proposal cache should have made this "
            "impossible -- refusing to mix two proposal outcomes.",
        )
        return proposals_dir

    def _validate(
        self, round_path: Path, round_index: int, incumbent: Harness, proposals_dir: Path
    ) -> ValidationRound:
        """Stage 4: one whole ``validate_round``, idempotent over its directory.

        With ``operational.validation_workers > 1`` the subjects evaluate in
        child processes; the meter still charges the delta of every persisted
        validation manifest under the round (``_validation_usage`` re-reads the
        disk), so child-persisted runs are metered exactly like in-process ones.
        """
        validation_parent = round_path / VALIDATION_DIR
        validation_round_path = round_dir(validation_parent, round_index)
        eval_config = EvaluationConfig(
            splits=self.splits,
            verifier=self.verifier,
            out_dir=validation_parent,
            round_index=round_index,
            verifier_factory=self.verifier_factory,
            client_factory=self.client_factory,
            **evaluation_config_kwargs(self.config),
        )
        with StageMeter(
            stage=STAGE_VALIDATION,
            stage_work_id=f"round_{round_index:02d}/{STAGE_VALIDATION}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            before = _validation_usage(validation_round_path)
            try:
                validation = validate_round(
                    incumbent,
                    proposals_dir,
                    eval_config,
                    self.pconfig,
                    loader_timeout_seconds=self.config.operational.loader_timeout_seconds,
                )
            finally:
                # A crashed validation stage still persisted (and paid for)
                # every run its manifests hold; charge the delta either way.
                meter.add(_validation_usage(validation_round_path) - before)
        return validation

    # -- OOLONG-real generalization check (non-gated, never feeds promotion) --

    def _real_check_enabled(self) -> bool:
        return (
            self.binding.name == "oolong_synth"
            and self.config.operational.real_check_every_n_rounds > 0
        )

    def _real_check_due(self, round_index: int) -> bool:
        if not self._real_check_enabled():
            return False
        return round_index % self.config.operational.real_check_every_n_rounds == 0

    def _real_generalization_check(
        self, round_index: int, tag: str, incumbent: Harness
    ) -> None:
        """Evaluate the current incumbent on the OOLONG-real check set.

        A generalization probe, deliberately outside the promotion machinery:
        real D&D transcripts are not cleanly decomposable and would add a second
        uncontrolled noise source on top of the gate's known sensitivity. The
        summary it writes is never read by the loop -- not by ``RoundOutcome``,
        the promotion ledger, ``prior_history``, ``tau_*``, or the patience
        counter. Crash-safe (skips a completed ``summary.json``) and fully
        guarded: any failure is a stderr warning and a return, never a round
        failure -- same contract as ``_run_post_round_analysis``.
        """
        check_dir = self.out_dir / OPT_DIR / REAL_CHECK_DIR / tag
        summary_path = check_dir / REAL_CHECK_SUMMARY_FILENAME
        if summary_path.exists():
            return
        try:
            instances = _read_split(
                self.splits_dir,
                REAL_CHECK_ENVIRONMENT,
                REAL_CHECK_LENGTH,
                ROLE_REAL_CHECK,
            )
        except Exception as error:  # noqa: BLE001 -- isolation from the loop is the point
            sys.stderr.write(
                f"OOLONG-real check skipped for {tag}: no check split ({error})\n"
            )
            return

        verifier = OolongVerifier(task_set="real")
        model_name = self.config.backends.runner.model
        kwargs = round_config_kwargs(self.config)
        kwargs.pop("attempts", None)
        round_config = RoundConfig(
            round_index=round_index,
            harness=incumbent,
            instances=instances,
            verifier=verifier,
            out_dir=check_dir,
            **kwargs,
        )
        check_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        with StageMeter(
            stage=STAGE_REAL_CHECK,
            stage_work_id=f"{REAL_CHECK_DIR}/{tag}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            try:
                harnessed = build_round_rlm(round_config)
                for instance in instances:
                    outcome = execute_run(
                        harnessed, instance, model_name=model_name, verifier=verifier
                    )
                    summary = outcome.completion.usage_summary
                    cost = summary.total_cost
                    meter.add(
                        UsageTotals(
                            input_tokens=int(summary.total_input_tokens),
                            output_tokens=int(summary.total_output_tokens),
                            cost=float(cost) if cost is not None else 0.0,
                            lower_bound=bool(outcome.usage_lower_bound),
                        )
                    )
                    verdict = outcome.verdict
                    rows.append(
                        {
                            "instance_id": instance["id"],
                            "answer_kind": instance["answer_kind"],
                            "question_type": instance.get("question_type"),
                            "n_episodes": instance.get("n_episodes"),
                            "passed": bool(verdict.passed) if verdict else None,
                            "score": continuous_score(instance, outcome.completion.response),
                            "cost_usd": cost,
                        }
                    )
            except Exception as error:  # noqa: BLE001 -- must never break the round
                sys.stderr.write(
                    f"OOLONG-real check failed for {tag}: {type(error).__name__}: {error}\n"
                )
                return
        _write_real_check_summary(summary_path, tag, round_index, harness_hash(incumbent), rows)


def _write_real_check_summary(
    summary_path: Path,
    tag: str,
    round_index: int,
    incumbent_hash: str,
    rows: list[dict[str, Any]],
) -> None:
    """Persist the per-answer-kind aggregate of one OOLONG-real check pass."""
    by_kind: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = by_kind.setdefault(row["answer_kind"], {"n": 0, "score_sum": 0.0, "passed": 0})
        bucket["n"] += 1
        bucket["score_sum"] += float(row["score"])
        bucket["passed"] += 1 if row["passed"] else 0
    per_kind = {
        kind: {
            "n": int(bucket["n"]),
            "mean_score": bucket["score_sum"] / bucket["n"] if bucket["n"] else 0.0,
            "pass_rate": bucket["passed"] / bucket["n"] if bucket["n"] else 0.0,
        }
        for kind, bucket in sorted(by_kind.items())
    }
    n_total = len(rows)
    payload = {
        "format": REAL_CHECK_SUMMARY_FORMAT,
        "tag": tag,
        "round_index": round_index,
        "incumbent_hash": incumbent_hash,
        "n": n_total,
        "mean_score": sum(float(row["score"]) for row in rows) / n_total if n_total else 0.0,
        "pass_rate": sum(1 for row in rows if row["passed"]) / n_total if n_total else 0.0,
        "per_answer_kind": per_kind,
        "rows": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp_path = summary_path.with_name(summary_path.name + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, summary_path)


def _validation_usage(validation_round_path: Path) -> UsageTotals:
    """Aggregate every persisted validation manifest under one round (KTD4)."""
    total = UsageTotals()
    if not validation_round_path.exists():
        return total
    for manifest_path in sorted(validation_round_path.rglob("runs.jsonl")):
        total = total + aggregate_manifest_usage(read_jsonl(manifest_path))
    return total


# ---------------------------------------------------------------------------
# Post-round analysis: best-effort, isolated, one snapshot per invocation (KTD4)
# ---------------------------------------------------------------------------

# The name a failure of the BATCH ITSELF is recorded under, as opposed to a
# failure of one of the aggregations it runs (each of those is recorded under
# its own tool name by ``Snapshot.run_tool``). It names the hook rather than
# any analysis because that is what failed: the batch never got as far as
# deciding which analyses to run.
POST_ROUND_BATCH_TOOL = "post_round_analysis"


def _post_round_analyses(
    out_dir: Path, snapshot: "Snapshot"
) -> list[tuple[str, Callable[[], Any]]]:
    """The aggregations this invocation should run, in the order they run.

    Every import here is function-local and must stay that way: ``analysis_io``
    and ``rounds`` both import THIS module for its layout constants and
    filenames (KTD1), so an import of either at module scope is a circular
    import that fails while ``orchestrator`` is still half-initialized. The
    dependency direction is analyses -> loop, never the reverse, and keeping
    the import inside the call is what preserves it.

    Two aggregations are conditional, and their absence is silent because it is
    the normal mid-experiment state rather than a fault:

    * the evaluation phase runs only once ``eval/eval_summary.json`` exists --
      the evaluation runner writes it, never this loop, so it is absent for the
      whole optimization run;
    * the frequency diff runs only once two rounds have persisted a bundle.

    Args:
        out_dir: The experiment directory to analyze.
        snapshot: The batch's allocated ``analysis_io.Snapshot``; every
            aggregation writes into that one directory (KTD2).

    Returns:
        ``(tool name, thunk)`` pairs. Nothing has run yet -- the caller runs
        each one under the snapshot's own guard.
    """
    from shrlm.experiment.collapse_and_attribution import (
        PHASE_EVALUATION,
        PHASE_OPTIMIZATION,
        TOOL_NAME_EVALUATION,
        TOOL_NAME_OPTIMIZATION,
        eval_summary_path,
        run_collapse_and_attribution,
    )
    from shrlm.experiment.incumbent_quality import TOOL_NAME as TOOL_NAME_INCUMBENT_QUALITY
    from shrlm.experiment.incumbent_quality import run_incumbent_quality
    from shrlm.experiment.pattern_frequency_diff import (
        MIN_DIFFABLE_BUNDLES,
        run_pattern_frequency_diff,
    )
    from shrlm.experiment.pattern_frequency_diff import TOOL_NAME as TOOL_NAME_DIFF
    from shrlm.experiment.rounds import discover_rounds
    from shrlm.experiment.surface_activity import TOOL_NAME as TOOL_NAME_SURFACE_ACTIVITY
    from shrlm.experiment.surface_activity import run_surface_activity

    # One discovery pass for the whole batch: every analysis below, and the
    # bundle list, ask the same question of the same unchanging tree, so a
    # separate walk per analysis would only re-derive the same answer several
    # times over.
    inventory = discover_rounds(out_dir)

    analyses: list[tuple[str, Callable[[], Any]]] = [
        (
            TOOL_NAME_OPTIMIZATION,
            lambda: run_collapse_and_attribution(
                out_dir, snapshot, phase=PHASE_OPTIMIZATION, inventory=inventory
            ),
        ),
        (
            TOOL_NAME_INCUMBENT_QUALITY,
            lambda: run_incumbent_quality(out_dir, snapshot, inventory=inventory),
        ),
        (
            TOOL_NAME_SURFACE_ACTIVITY,
            lambda: run_surface_activity(out_dir, snapshot, inventory=inventory),
        ),
    ]

    # The bundles come from the shared inventory rather than a local walk, so
    # the diff reads exactly the rounds the loop wrote (KTD1). Incomplete ones
    # are NOT filtered out here: the aggregation flags every bundle it is given
    # and compares only the evidence-complete ones (KTD5), which is how a
    # partial round stays visible instead of silently vanishing.
    bundles = [
        record.mining_round_path / BUNDLE_FILENAME
        for record in inventory.rounds
        if (record.mining_round_path / BUNDLE_FILENAME).is_file()
    ]
    # A frequency diff needs a pair to compare. Below this the aggregation
    # refuses its arguments (correctly), which for a first round is the normal
    # state and not a failure worth recording -- so it is left out of the
    # batch instead.
    if len(bundles) >= MIN_DIFFABLE_BUNDLES:
        analyses.append(
            (
                TOOL_NAME_DIFF,
                lambda: run_pattern_frequency_diff(bundles, snapshot, inventory=inventory),
            )
        )
    if eval_summary_path(out_dir).is_file():
        analyses.append(
            (
                TOOL_NAME_EVALUATION,
                lambda: run_collapse_and_attribution(
                    out_dir, snapshot, phase=PHASE_EVALUATION, inventory=inventory
                ),
            )
        )
    return analyses


def _run_post_round_analysis(out_dir: Path) -> None:
    """Refresh the analysis snapshot from what the loop has persisted (KTD4).

    Best-effort in the strongest sense: no analysis FAILURE in here may reach
    the experiment. A broken aggregation, an unreadable artifact, an
    unallocatable snapshot, even an ImportError raised while loading the
    analysis modules -- each ends in a warning on stderr and a return. The
    experiment's outcome must never depend on whether its analyses ran, which
    is why the whole body sits under one blanket guard and why the individual
    tools sit under a second one (``Snapshot.run_tool``) inside it: one failing
    aggregation must not cost the others their output either.

    An INTERRUPT is the one thing that does travel outward. ``KeyboardInterrupt``
    and ``SystemExit`` are not analysis failures -- they are the operator
    stopping the process -- so they take the same audit-trail path as any other
    escape (see ``_publish_failed_batch``) and are then re-raised. Catching them
    with everything else would silently turn a Ctrl-C during the hook into
    "skip this snapshot and run another round".

    The batch allocates exactly ONE snapshot (KTD2) and publishes it
    unconditionally -- ``publish`` writes ``provenance.json`` either way and
    withholds only the completion marker when a tool failed, so a partial batch
    leaves a readable audit trail that no reader will mistake for the latest
    results. "Unconditionally" includes the failures that happen OUTSIDE any
    tool's own guard, between allocating the directory and publishing it: those
    take the same path through ``_publish_failed_batch``, because a stamped
    directory holding neither provenance nor a completion marker would be an
    empty snapshot with nothing on disk saying why -- the one outcome this
    module and ``analysis_io`` both promise never to produce.

    Args:
        out_dir: The experiment directory; the snapshot lands under its
            ``analysis/``.
    """
    # A local annotation is never evaluated at runtime (PEP 526), so this one
    # names the TYPE_CHECKING-only import without quoting it.
    snapshot: Snapshot | None = None
    try:
        # Function-local because ``analysis_io`` imports THIS module: at module
        # scope this is a circular import that fails while ``orchestrator`` is
        # still half-initialized (the direction ``_post_round_analyses``'s
        # docstring spells out). The position is structurally required, not
        # merely defensive.
        from shrlm.experiment.analysis_io import PROVENANCE_FILENAME, allocate_snapshot

        snapshot = allocate_snapshot(out_dir)
        failed = [
            name
            for name, call in _post_round_analyses(out_dir, snapshot)
            if not snapshot.run_tool(name, call)
        ]
        published = snapshot.publish()
    except (KeyboardInterrupt, SystemExit) as interrupt:
        # Ahead of the blanket arm because these are NOT ``Exception``: without
        # this, Ctrl-C between the ``mkdir`` and ``publish`` left exactly the
        # stamped-but-empty directory this function promises never to produce.
        # The snapshot explains itself, and then the interrupt goes on stopping
        # the experiment -- swallowing it would make Ctrl-C mean "skip the
        # analysis and keep running", which is not what anybody typed it for.
        sys.stderr.write(
            f"post-round analysis interrupted for {out_dir}: {type(interrupt).__name__}\n"
        )
        _publish_failed_batch(snapshot, interrupt)
        raise
    except Exception as error:  # noqa: BLE001 -- isolation from the loop is the point
        sys.stderr.write(
            f"post-round analysis skipped for {out_dir}: {type(error).__name__}: {error}\n"
        )
        _publish_failed_batch(snapshot, error)
        return
    if not published:
        sys.stderr.write(
            f"post-round analysis: {', '.join(failed)} failed; {snapshot.path} is unpublished, "
            f"see {snapshot.path / PROVENANCE_FILENAME}\n"
        )


def _publish_failed_batch(snapshot: "Snapshot | None", error: BaseException) -> None:
    """Leave provenance behind for a batch that failed outside any tool's guard.

    ``allocate_snapshot`` claims its directory before a single aggregation
    runs, so anything that raises between the claim and ``publish`` -- the
    function-local import of the analysis modules, building the list of
    analyses, or a ``KeyboardInterrupt`` landing anywhere in there -- used to
    leave a stamped directory with no ``provenance.json`` and no
    ``published.json``. Recording the failure as a tool outcome and
    publishing gives that directory the audit trail every other failure mode
    gets, and (because the outcome is a failed one) withholds the completion
    marker, so it can never be read as the latest results.

    Nothing here may reach the loop either, so the recovery write is guarded in
    turn: if the reason the batch failed was that this snapshot cannot be
    written to at all, saying so on stderr is the end of it.
    """
    if snapshot is None:
        return
    try:
        snapshot.record_tool(
            POST_ROUND_BATCH_TOOL, ok=False, error=f"{type(error).__name__}: {error}"
        )
        snapshot.publish()
    except Exception as write_error:  # noqa: BLE001 -- isolation from the loop is the point
        sys.stderr.write(
            f"post-round analysis: could not record that failure under {snapshot.path}: "
            f"{type(write_error).__name__}: {write_error}\n"
        )


def run_experiment(
    config: ExperimentConfig,
    out_dir: Path | str,
    *,
    verifier: Verifier | None = None,
    sub_verifier: SubVerifier | None = None,
    attributor_lm: BaseLM | None = None,
    proposer_lm: BaseLM | None = None,
    loaders: dict[str, LoaderFn] | None = None,
    verifier_factory: str | None = None,
    client_factory: tuple[str, dict[str, Any]] | None = None,
) -> ExperimentResult:
    """Run (or resume) one whole optimization experiment and freeze the result (R7).

    For t in 1..T: governed mining on the held-in split (``loop.m`` attempts
    per instance, under the cumulative spend breaker, R12) -> ``mine_round``
    into ``bundle.json`` -> ``propose_round`` (prior promotion ledgers threaded
    as history, responses replayed from the persisted proposal cache) ->
    ``validate_round`` -> incumbent update / patience counter. Stops at T
    rounds or after ``loop.patience`` consecutive no-promotion rounds, then
    freezes the final incumbent to ``<out_dir>/sh_rlm/harness.json``.

    Idempotent and crash-safe over ``out_dir``: re-invocation verifies the
    config identity (R3), replays completed rounds from their markers without
    executing anything, and resumes an interrupted round at its exact stage
    boundary -- persisted runs are never re-run, and persisted caches replay
    attribution and proposal responses.

    Args:
        config: The loaded experiment configuration (one profile).
        out_dir: The experiment directory; created if absent.
        verifier: The environment verifier for mining and validation runs;
            defaults to ``GraphWalksVerifier()`` (the held-in/held-out splits
            are GraphWalks source-short).
        sub_verifier: Optional sub-call verifier for grounded attribution.
        attributor_lm: The attributor client; defaults to the configured
            backend via ``rlm.clients.get_client``.
        proposer_lm: The proposer client; same default mechanism.
        loaders: Optional environment-loader overrides for
            ``materialize_splits`` (tests inject offline loaders here).
        verifier_factory: Dotted path of a zero-argument callable returning
            ``verifier``, for the child processes a parallel validation stage
            (``operational.validation_workers > 1``) spawns. Defaults to the
            ``GraphWalksVerifier`` path when ``verifier`` is the default; an
            injected verifier needs it whenever workers exceed 1.
        client_factory: Test-only seam for those children: a dotted path plus
            per-subject-id args the child installs on ``rlm.core.rlm.get_client``
            (see ``shrlm.optimization.subject_worker``).

    Returns:
        The ``ExperimentResult`` with every round's outcome and the frozen
        harness identity.

    Raises:
        IdentityMismatchError: The configured identity hash does not match the
            experiment's persisted one -- raised before any run executes.
        MiningBudgetExceededError: The mining spend breaker tripped; partial
            state is persisted and the invocation refuses to continue.
        ExperimentPersistenceError: Persisted state contradicts itself or the
            configuration.
    """
    # The sub-verifier pairs with the verifier: when the caller takes the
    # selected environment's default for one, it gets that environment's default
    # for the other (``resolve_env_binding`` keyed on ``config.loop.environment``
    # -- GraphWalksVerifier/GraphWalksSubVerifier for "graphwalks",
    # OolongVerifier/OolongSubVerifier for "oolong_synth"). experiment_kimi ran
    # with neither passed and therefore no grounding at all (bundle config
    # sub_verifier_enabled=False). The ablation is still one call away -- pass
    # the environment's verifier with sub_verifier=None.
    if verifier is None:
        binding = resolve_env_binding(config)
        verifier = binding.verifier
        if sub_verifier is None:
            sub_verifier = binding.sub_verifier
    experiment = _Experiment(
        config=config,
        out_dir=Path(out_dir),
        verifier=verifier,
        sub_verifier=sub_verifier,
        attributor_lm=attributor_lm,
        proposer_lm=proposer_lm,
        loaders=loaders,
        verifier_factory=verifier_factory,
        client_factory=client_factory,
    )
    return experiment.run()


__all__ = [
    "CONFIG_FILENAME",
    "EVIDENCE_MARKER_FILENAME",
    "FROZEN_DIR",
    "FROZEN_HARNESS_FILENAME",
    "POST_ROUND_BATCH_TOOL",
    "PROPOSALS_MARKER_FILENAME",
    "ROUND_MARKER_FILENAME",
    "STOP_MAX_ROUNDS",
    "STOP_PATIENCE",
    "ExperimentPersistenceError",
    "ExperimentResult",
    "IdentityMismatchError",
    "MiningBudgetExceededError",
    "RoundOutcome",
    "check_identity",
    "experiment_round_dir",
    "rematerialize_harness_envelope",
    "run_experiment",
]
