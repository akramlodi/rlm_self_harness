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
        proposals/            # <candidate_id>/proposal.json directories
        proposals_complete.json  # stage marker: the proposal stage's sealed outcome
        validation/round_NN/  # validate_round artifacts, incl. decision.json
        work/                 # scratch for generated-source modules
        round.json            # stage marker: the round's final outcome
    sh_rlm/harness.json       # the frozen final harness (write_harness_json envelope)

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
2. The evidence stage is complete when ``bundle.json`` exists; resume reads it
   back instead of re-mining (the attribution cache, persisted at the
   configured operational path, makes any re-mine replay-free anyway).
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

Single-threaded, main thread only: the SIGALRM hard-deadline backstop in
``shrlm.optimization.costs`` binds only there (POSIX main thread), and this
module never spawns threads around run execution.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from shrlm.environments.graphwalks import GraphWalksVerifier
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
from shrlm.optimization.bundle import BUNDLE_FILENAME, round_dir, write_bundle
from shrlm.optimization.candidates import CandidateRejection, materialize_harness
from shrlm.optimization.costs import (
    CandidateSpendBreaker,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import RoundConfig, load_manifest, mine_round
from shrlm.optimization.mining import WeaknessMiner
from shrlm.optimization.promotion import DECISION_PROMOTED, PromotionConfig
from shrlm.optimization.proposal import ProposalCache, load_passing_behaviors, propose_round
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

CONFIG_FILENAME = "config.json"
CONFIG_FORMAT = "shrlm-experiment-config/v1"

OPT_DIR = "opt"
MINING_DIR = "mining"
PROPOSALS_DIR = "proposals"
VALIDATION_DIR = "validation"
WORK_DIR = "work"

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

# The incumbent every experiment starts from (registry entry, not a copy).
INITIAL_INCUMBENT = "H0"

# The optimization loop mines and validates the source-short splits.
SPLIT_ENVIRONMENT = "graphwalks"
SPLIT_LENGTH = "short"
ROLE_HELD_IN = "held_in"
ROLE_HELD_OUT = "held_out"


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


def _read_split(splits_dir: Path, role: str) -> list[dict[str, Any]]:
    path = splits_dir / split_file_name(SPLIT_ENVIRONMENT, SPLIT_LENGTH, role)
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
    caps: ValidationCaps = field(init=False)
    pconfig: PromotionConfig = field(init=False)
    splits: ValidationSplits = field(init=False)
    usage_path: Path = field(init=False)
    prior_history: list[tuple[list[dict[str, Any]], dict[str, Any]]] = field(init=False)

    def __post_init__(self) -> None:
        self.caps = validation_caps(self.config)
        self.pconfig = promotion_config(self.config)
        self.usage_path = self.out_dir / STAGE_USAGE_FILE
        self.prior_history = []

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
        self.splits = ValidationSplits(
            heldin=_read_split(splits_dir, ROLE_HELD_IN),
            heldout=_read_split(splits_dir, ROLE_HELD_OUT),
        )

        incumbent: Harness = HARNESSES[INITIAL_INCUMBENT]
        rounds: list[RoundOutcome] = []
        without_promotion = 0
        stopped = STOP_MAX_ROUNDS
        for round_index in range(1, self.config.loop.t + 1):
            round_path = experiment_round_dir(self.out_dir, round_index)
            marker_path = round_path / ROUND_MARKER_FILENAME
            if marker_path.exists():
                outcome = self._replay_round(round_path, round_index)
            else:
                outcome = self._execute_round(round_path, round_index, incumbent)
            rounds.append(outcome)
            if outcome.promoted:
                incumbent = self._promoted_harness(round_path, round_index, outcome)
                without_promotion = 0
            else:
                without_promotion += 1
            if outcome.has_ledger:
                validation_round_path = round_dir(round_path / VALIDATION_DIR, round_index)
                self.prior_history.append(load_promotion_ledger(validation_round_path))
            if without_promotion >= self.config.loop.patience:
                stopped = STOP_PATIENCE
                break

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
            result = run_governed_round(mining_config, breaker)
            meter.add(aggregate_manifest_usage(result.entries[known:]))
            if result.over_budget:
                raise MiningBudgetExceededError(
                    f"round {round_index} mining tripped the spend breaker at "
                    f"{result.spent:.6f} USD (candidate_budget "
                    f"{self.caps.candidate_budget}); {len(result.skipped_run_ids)} run(s) "
                    "were skipped. Completed runs are persisted; the round is resumable "
                    "only under a configuration whose budget admits it."
                )

    def _evidence_bundle(
        self, mining_parent: Path, mining_round_path: Path, round_index: int
    ) -> dict[str, Any]:
        """Stage 2: mine the persisted round into ``bundle.json`` (or read it back)."""
        bundle_path = mining_round_path / BUNDLE_FILENAME
        if bundle_path.exists():
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
                    SPLIT_ENVIRONMENT, SPLIT_LENGTH, ROLE_HELD_IN
                ).removesuffix(".jsonl"),
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
        _persist_once(
            marker_path,
            {
                "format": PROPOSALS_MARKER_FORMAT,
                "round": round_index,
                "candidate_ids": sorted(written.candidate_id for written in result.written),
                "prompt_sha256": result.prompt_sha256,
                "skipped_patterns": list(result.skipped_patterns),
                "n_materialization_failures": len(result.materialization_failures),
            },
            f"{marker_path} already seals a diverging candidate set for round "
            f"{round_index}; the persisted proposal cache should have made this "
            "impossible -- refusing to mix two proposal outcomes.",
        )
        return proposals_dir

    def _validate(
        self, round_path: Path, round_index: int, incumbent: Harness, proposals_dir: Path
    ) -> ValidationRound:
        """Stage 4: one whole ``validate_round``, idempotent over its directory."""
        validation_parent = round_path / VALIDATION_DIR
        validation_round_path = round_dir(validation_parent, round_index)
        eval_config = EvaluationConfig(
            splits=self.splits,
            verifier=self.verifier,
            out_dir=validation_parent,
            round_index=round_index,
            **evaluation_config_kwargs(self.config),
        )
        with StageMeter(
            stage=STAGE_VALIDATION,
            stage_work_id=f"round_{round_index:02d}/{STAGE_VALIDATION}",
            round_index=round_index,
            out_path=self.usage_path,
        ) as meter:
            before = _validation_usage(validation_round_path)
            validation = validate_round(
                incumbent,
                proposals_dir,
                eval_config,
                self.pconfig,
                loader_timeout_seconds=self.config.operational.loader_timeout_seconds,
            )
            meter.add(_validation_usage(validation_round_path) - before)
        return validation


def _validation_usage(validation_round_path: Path) -> UsageTotals:
    """Aggregate every persisted validation manifest under one round (KTD4)."""
    total = UsageTotals()
    if not validation_round_path.exists():
        return total
    for manifest_path in sorted(validation_round_path.rglob("runs.jsonl")):
        total = total + aggregate_manifest_usage(read_jsonl(manifest_path))
    return total


def run_experiment(
    config: ExperimentConfig,
    out_dir: Path | str,
    *,
    verifier: Verifier | None = None,
    sub_verifier: SubVerifier | None = None,
    attributor_lm: BaseLM | None = None,
    proposer_lm: BaseLM | None = None,
    loaders: dict[str, LoaderFn] | None = None,
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
    experiment = _Experiment(
        config=config,
        out_dir=Path(out_dir),
        verifier=verifier if verifier is not None else GraphWalksVerifier(),
        sub_verifier=sub_verifier,
        attributor_lm=attributor_lm,
        proposer_lm=proposer_lm,
        loaders=loaders,
    )
    return experiment.run()


__all__ = [
    "CONFIG_FILENAME",
    "FROZEN_DIR",
    "FROZEN_HARNESS_FILENAME",
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
