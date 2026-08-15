"""Script 2: incumbent quality over rounds -- feeds Graph 2 (quality over time).

``incumbent_quality_over_rounds`` walks every round's ``promotions.jsonl``
(via ``shrlm.experiment.promotion_rounds``) and tracks one running "current
incumbent" state -- held-in and held-out pass counts and run counts -- that
only moves on a ``promoted`` decision. Every round still emits exactly one
row, so the series is a clean step function even through rounds that
promoted nothing.

Seeding the baseline
    The state starts from the first round that actually scored a candidate:
    every scored row's ``rule.<split>.baseline_pass_count`` is the same
    incumbent-before-this-round figure (all candidates in a round are scored
    against the same evaluated baseline), so any one of them seeds it. In the
    ordinary case that is round 1, where baseline == H0, matching the
    proposal's framing; a round 1 that happened to be loader-rejection-only
    (no candidate survived the loader, so nothing was ever scored) is handled
    by carrying the seed forward to the first round that does score
    something, rather than crashing -- that round's row (and any before it)
    reports null pass rates with an annotation explaining why.

Non-promoting rounds
    A round that promotes nothing leaves the incumbent state untouched. When
    the reason is structural -- a loader-gate rejection, or an evaluation
    that never finished scoring because it went over budget (``rule`` is
    null on that row) -- the row's ``annotation`` names every such candidate
    and its reason, for plotting as a marker. An ordinary scored-but-rejected
    candidate (failed the promotion rule or band; ``rule`` is populated)
    contributes no annotation, per the spec this script implements: it is
    the expected, unremarkable outcome of a round that did not improve on
    the incumbent, not a structural anomaly worth flagging.

The all-candidates table
    ``incumbent_quality_over_rounds`` tracks only the running incumbent -- one
    number per split per round. ``all_candidate_quality_over_rounds`` is the
    second table the plotting script's optional scatter overlay needs: one
    row per *scored* candidate record (``rule`` populated -- an unscored
    loader rejection or over-budget stop has no measured pass rate to plot),
    reading its own ``candidate_pass_count`` rather than the shared
    ``baseline_pass_count``. This includes the merged harness's own
    re-evaluation record (a genuine scored candidate, just with no single
    surface), and constituent records that lost the promotion despite being
    individually scored.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shrlm.experiment.promotion_rounds import iter_promotion_rounds
from shrlm.optimization.promotion import DECISION_PROMOTED

INCUMBENT_QUALITY_FILENAME = "incumbent_quality.csv"
INCUMBENT_QUALITY_CANDIDATES_FILENAME = "incumbent_quality_candidates.csv"

SPLIT_HELDIN = "heldin"
SPLIT_HELDOUT = "heldout"


@dataclass(frozen=True)
class _IncumbentState:
    """The running incumbent's pass counts on both splits."""

    heldin_pass_count: int
    heldin_n_runs: int
    heldout_pass_count: int
    heldout_n_runs: int

    @property
    def heldin_pass_rate(self) -> float | None:
        return self.heldin_pass_count / self.heldin_n_runs if self.heldin_n_runs else None

    @property
    def heldout_pass_rate(self) -> float | None:
        return self.heldout_pass_count / self.heldout_n_runs if self.heldout_n_runs else None

    @classmethod
    def from_rule(cls, rule: dict[str, dict], *, key: str) -> "_IncumbentState":
        """Build from a decision's ``rule`` dict, reading ``<key>_pass_count``."""
        return cls(
            heldin_pass_count=rule[SPLIT_HELDIN][f"{key}_pass_count"],
            heldin_n_runs=rule[SPLIT_HELDIN]["n_runs"],
            heldout_pass_count=rule[SPLIT_HELDOUT][f"{key}_pass_count"],
            heldout_n_runs=rule[SPLIT_HELDOUT]["n_runs"],
        )


@dataclass(frozen=True)
class IncumbentQualityRow:
    """One round's incumbent-quality reading, emitted regardless of outcome."""

    round_index: int
    heldin_pass_rate: float | None
    heldout_pass_rate: float | None
    incumbent_changed: bool
    annotation: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "heldin_pass_rate": self.heldin_pass_rate,
            "heldout_pass_rate": self.heldout_pass_rate,
            "incumbent_changed": self.incumbent_changed,
            "annotation": self.annotation,
        }


@dataclass(frozen=True)
class CandidateQualityRow:
    """One scored candidate's own pass rates in one round (the scatter table)."""

    round_index: int
    subject_id: str
    decision: str
    surface: str | None
    heldin_pass_rate: float | None
    heldout_pass_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "subject_id": self.subject_id,
            "decision": self.decision,
            "surface": self.surface,
            "heldin_pass_rate": self.heldin_pass_rate,
            "heldout_pass_rate": self.heldout_pass_rate,
        }


def _structural_rejection_annotation(records: list[dict]) -> str | None:
    """Name every non-promoted, unscored (``rule`` null) candidate and why."""
    parts = [
        f"{record['subject_id']}: {'; '.join(record.get('reasons', ())) or record['decision']}"
        for record in records
        if record["decision"] != DECISION_PROMOTED and record.get("rule") is None
    ]
    return "; ".join(parts) if parts else None


def incumbent_quality_over_rounds(out_dir: Path | str) -> list[IncumbentQualityRow]:
    """One row per validation round: the incumbent's pass rates and whether it changed."""
    rows: list[IncumbentQualityRow] = []
    state: _IncumbentState | None = None

    for round_index, records, _decision in iter_promotion_rounds(out_dir):
        scored = [record for record in records if record.get("rule") is not None]
        if state is None:
            if not scored:
                rows.append(
                    IncumbentQualityRow(
                        round_index=round_index,
                        heldin_pass_rate=None,
                        heldout_pass_rate=None,
                        incumbent_changed=False,
                        annotation=(
                            "no candidate was scored this round (loader-rejection-only or "
                            "over-budget); the H0 baseline has not been observed yet"
                        ),
                    )
                )
                continue
            state = _IncumbentState.from_rule(scored[0]["rule"], key="baseline")

        promoted_record = next(
            (record for record in records if record["decision"] == DECISION_PROMOTED), None
        )
        if promoted_record is not None:
            state = _IncumbentState.from_rule(promoted_record["rule"], key="candidate")
            incumbent_changed = True
            annotation = None
        else:
            incumbent_changed = False
            annotation = _structural_rejection_annotation(records)

        rows.append(
            IncumbentQualityRow(
                round_index=round_index,
                heldin_pass_rate=state.heldin_pass_rate,
                heldout_pass_rate=state.heldout_pass_rate,
                incumbent_changed=incumbent_changed,
                annotation=annotation,
            )
        )
    return rows


def all_candidate_quality_over_rounds(out_dir: Path | str) -> list[CandidateQualityRow]:
    """One row per scored candidate record, across every round (the scatter table)."""
    rows: list[CandidateQualityRow] = []
    for round_index, records, _decision in iter_promotion_rounds(out_dir):
        for record in records:
            rule = record.get("rule")
            if rule is None:
                continue
            state = _IncumbentState.from_rule(rule, key="candidate")
            rows.append(
                CandidateQualityRow(
                    round_index=round_index,
                    subject_id=record["subject_id"],
                    decision=record["decision"],
                    surface=record.get("surface"),
                    heldin_pass_rate=state.heldin_pass_rate,
                    heldout_pass_rate=state.heldout_pass_rate,
                )
            )
    return rows


def _write_csv(path: Path, rows: Sequence[IncumbentQualityRow] | Sequence[CandidateQualityRow]) -> None:
    with path.open("w", newline="") as handle:
        if not rows:
            return
        fieldnames = list(rows[0].to_dict())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``incumbent_quality.csv`` and ``incumbent_quality_candidates.csv``."""
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.incumbent_quality",
        description="Per-round incumbent held-in/held-out pass rates (feeds Graph 2).",
    )
    parser.add_argument("out_dir", help="the experiment directory to read (holds opt/)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rows = incumbent_quality_over_rounds(out_dir)
    if not rows:
        sys.stderr.write(f"no validation rounds with a promotion ledger found under {out_dir}\n")
        return 1

    quality_path = out_dir / INCUMBENT_QUALITY_FILENAME
    _write_csv(quality_path, rows)
    sys.stdout.write(f"Wrote {quality_path}\n")

    candidates_path = out_dir / INCUMBENT_QUALITY_CANDIDATES_FILENAME
    _write_csv(candidates_path, all_candidate_quality_over_rounds(out_dir))
    sys.stdout.write(f"Wrote {candidates_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "INCUMBENT_QUALITY_CANDIDATES_FILENAME",
    "INCUMBENT_QUALITY_FILENAME",
    "CandidateQualityRow",
    "IncumbentQualityRow",
    "all_candidate_quality_over_rounds",
    "incumbent_quality_over_rounds",
    "main",
]
