"""Mechanism-frequency diff over two or more mining bundles (proposal section 3.5).

``diff_bundle_pair`` full-outer-joins two ``bundle.json``s' ``patterns[]`` on
the four-tuple failure signature (``verifier_cause``, ``failing_level``,
``causal_status``, ``agent_mechanism`` -- ``FailureSignature.key()``) and
reports, per signature, whether it was resolved, newly introduced, or
persisted (improved / worsened / unchanged) between the two rounds. This is
the direct counterpart to the proposal's "report the frequency of each mined
failure pattern before and after optimization" -- the underlying data
(``patterns[].instance_support``, ``totals.n_runs``) was already captured by
mining; nothing here is a new instrumentation point, only an analysis pass
over what ``bundle.json`` already persists.

The denominator, honestly stated
    ``FailurePattern.instance_support`` counts distinct *instances* a
    signature was seen on; ``bundle.json`` has no matching distinct-instance
    denominator to normalize it against -- ``MiningTotals.n_runs`` is a
    *run* count (``len(runs)`` mined that round), not an instance count.
    When mining used one attempt per instance (the common case), the two
    coincide and ``instance_support / n_runs`` is a true per-instance rate.
    Under repeated attempts per instance, ``n_runs`` over-counts the
    denominator relative to ``instance_support``'s instance-level numerator,
    so the computed rate is a proportional, comparable-across-rounds figure
    rather than an exact "fraction of instances affected" -- consistent with
    how the rest of this codebase treats a known-imprecise-but-honest
    denominator (see ``totals`` on ``EvidenceBundle``, or the bundle's own
    ``known_substrate_biases``), not silently presented as exact.

Which bundles are compared, and why the excluded ones are still written down
    A frequency diff across a round whose mining never finished compares a
    full round against a partial one and reports the shortfall as improvement.
    So the comparison runs only over bundles whose round is EVIDENCE-COMPLETE
    (KTD9) -- but excluding a bundle silently is the same failure in the other
    direction, because the excluded round then simply disappears from the
    output. Every bundle handed to this analysis therefore gets a row in
    ``pattern_frequency_diff_bundles.csv`` naming its round, its completeness,
    whether it was included, and -- when it was not -- why (KTD5's flag,
    don't refuse).

    Exclusion turns on KNOWN incompleteness only. A bundle that discovery
    cannot match to a round of this experiment (a checked-in fixture, a copy
    lifted out of the tree) has completeness ``unknown``, and unknown never
    collapses into false: it is compared, and its row says the completeness was
    never established. Refusing to compare on unknown would be treating an
    unmeasured round as a failed one, which is precisely what KTD9 forbids.
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrlm.experiment.analysis_io import (
    Snapshot,
    add_snapshot_parent_argument,
    allocate_snapshot,
    tristate,
    write_csv,
    write_json,
)
from shrlm.experiment.rounds import ExperimentInventory, RoundRecord, discover_rounds
from shrlm.optimization.bundle import BUNDLE_FILENAME

# A signature is present with rate exactly equal (floating-point noise only,
# not "close enough") counted as persisted_unchanged rather than a spurious
# improved/worsened flip.
RATE_EPSILON = 1e-9

FILESYSTEM_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

STATUS_RESOLVED = "resolved"
STATUS_NEW = "new"
STATUS_PERSISTED_IMPROVED = "persisted_improved"
STATUS_PERSISTED_WORSENED = "persisted_worsened"
STATUS_PERSISTED_UNCHANGED = "persisted_unchanged"

SIGNATURE_FIELDS = ("verifier_cause", "failing_level", "causal_status", "agent_mechanism")

# The name this analysis is recorded under in a snapshot's provenance.
TOOL_NAME = "pattern_frequency_diff"

# Declared field order (KTD8): a pair with no signature on either side still
# writes a header-only CSV, which says "no shared failure patterns" where an
# empty file would say "this analysis never ran".
DIFF_FIELDNAMES = (
    "signature",
    "status",
    "support_rate_before",
    "support_rate_after",
    "delta",
    "delta_pct",
    "grounded_fraction_before",
    "grounded_fraction_after",
    "below_support_floor_before",
    "below_support_floor_after",
)

# The completeness table: one row per bundle handed to this analysis, whether
# or not it was compared (KTD5).
BUNDLES_FILENAME = "pattern_frequency_diff_bundles.csv"
BUNDLE_FIELDNAMES = (
    "label",
    "bundle_path",
    "round_index",
    "round_complete",
    "evidence_complete",
    "included",
    "exclusion_reason",
)


@dataclass(frozen=True)
class _BundleSummary:
    """One loaded bundle, reduced to what the diff needs."""

    path: Path
    label: str
    n_runs: int
    patterns_by_key: dict[tuple[str, str, str, str], dict[str, Any]]


@dataclass(frozen=True)
class BundleCompletenessRow:
    """One input bundle's completeness, and whether the diff used it.

    ``round_index`` and both flags are empty when discovery could not match the
    bundle to a round of this experiment -- completeness unknown, which is a
    reason to say so, never a reason to drop the bundle.
    """

    label: str
    bundle_path: str
    round_index: int | None
    round_complete: bool | None
    evidence_complete: bool | None
    included: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bundle_path": self.bundle_path,
            "round_index": self.round_index,
            "round_complete": tristate(self.round_complete),
            "evidence_complete": tristate(self.evidence_complete),
            "included": tristate(self.included),
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class PatternFrequencyDiffRow:
    """One signature's before/after reading in a diffed bundle pair."""

    signature_key: tuple[str, str, str, str]
    status: str
    support_rate_before: float
    support_rate_after: float
    delta: float
    delta_pct: float | None
    grounded_fraction_before: float | None
    grounded_fraction_after: float | None
    below_support_floor_before: bool | None
    below_support_floor_after: bool | None

    @property
    def signature_str(self) -> str:
        return "|".join(f"{field_name}={value}" for field_name, value in zip(SIGNATURE_FIELDS, self.signature_key, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature_str,
            "status": self.status,
            "support_rate_before": self.support_rate_before,
            "support_rate_after": self.support_rate_after,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "grounded_fraction_before": self.grounded_fraction_before,
            "grounded_fraction_after": self.grounded_fraction_after,
            "below_support_floor_before": self.below_support_floor_before,
            "below_support_floor_after": self.below_support_floor_after,
        }


def _signature_key(pattern: dict[str, Any]) -> tuple[str, str, str, str]:
    """The fixed four-tuple two bundles' patterns are joined on.

    Unpacked into four names rather than returned straight from the generator:
    a comprehension over ``SIGNATURE_FIELDS`` has the type of a
    variable-length tuple, not the four-tuple the join key actually is, and
    the previous spelling needed a ``# type: ignore[return-value]`` to paper
    over the difference. Unpacking keeps ``SIGNATURE_FIELDS`` as the single
    source of field order while making the arity real -- a signature that grew
    or lost a field raises here instead of silently producing a key of the
    wrong shape.
    """
    signature = pattern["signature"]
    verifier_cause, failing_level, causal_status, agent_mechanism = (
        signature[field_name] for field_name in SIGNATURE_FIELDS
    )
    return verifier_cause, failing_level, causal_status, agent_mechanism


def _safe_label(raw: str) -> str:
    return FILESYSTEM_SAFE_PATTERN.sub("_", raw).strip("_") or "bundle"


def load_bundle_summary(path: Path | str, explicit_label: str | None = None) -> _BundleSummary:
    """Reduce one ``bundle.json`` to its signature->pattern map plus a naming label.

    Label precedence: ``config.round_index`` when present (the normal case --
    every mining round stamps this), else an explicit ``--labels`` entry, else
    the bundle's parent directory name (``round_NN`` in the standard layout,
    far more useful than every bundle sharing the literal filename
    ``bundle.json``), else a positional fallback the caller assigns.
    """
    path = Path(path)
    with path.open() as handle:
        bundle = json.load(handle)

    config = bundle.get("config", {})
    round_index = config.get("round_index")
    if round_index is not None:
        label = f"round_{round_index}"
    elif explicit_label is not None:
        label = explicit_label
    elif path.parent.name:
        label = path.parent.name
    else:
        label = path.stem

    patterns_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for pattern in bundle.get("patterns", []):
        key = _signature_key(pattern)
        if key in patterns_by_key:
            raise ValueError(
                f"{path} has two patterns sharing signature {key}; mining clusters each "
                "signature into exactly one pattern, so this indicates a corrupt or "
                "hand-edited bundle, not a case this diff can resolve silently"
            )
        patterns_by_key[key] = pattern

    return _BundleSummary(
        path=path,
        label=_safe_label(label),
        n_runs=int(bundle.get("totals", {}).get("n_runs", 0)),
        patterns_by_key=patterns_by_key,
    )


def _rate(pattern: dict[str, Any] | None, n_runs: int) -> float:
    if pattern is None or n_runs <= 0:
        return 0.0
    return pattern["instance_support"] / n_runs


def diff_bundle_pair(before: _BundleSummary, after: _BundleSummary) -> list[PatternFrequencyDiffRow]:
    """Full outer join of two bundles' patterns on the failure signature."""
    all_keys = sorted(set(before.patterns_by_key) | set(after.patterns_by_key))
    rows: list[PatternFrequencyDiffRow] = []
    for key in all_keys:
        pattern_before = before.patterns_by_key.get(key)
        pattern_after = after.patterns_by_key.get(key)
        rate_before = _rate(pattern_before, before.n_runs)
        rate_after = _rate(pattern_after, after.n_runs)
        present_before = pattern_before is not None and rate_before > 0
        present_after = pattern_after is not None and rate_after > 0
        delta = rate_after - rate_before

        if present_before and not present_after:
            status = STATUS_RESOLVED
        elif not present_before and present_after:
            status = STATUS_NEW
        elif abs(delta) < RATE_EPSILON:
            status = STATUS_PERSISTED_UNCHANGED
        elif delta < 0:
            status = STATUS_PERSISTED_IMPROVED
        else:
            status = STATUS_PERSISTED_WORSENED

        rows.append(
            PatternFrequencyDiffRow(
                signature_key=key,
                status=status,
                support_rate_before=rate_before,
                support_rate_after=rate_after,
                delta=delta,
                delta_pct=(delta / rate_before) if rate_before > 0 else None,
                grounded_fraction_before=pattern_before["grounded_fraction"] if pattern_before else None,
                grounded_fraction_after=pattern_after["grounded_fraction"] if pattern_after else None,
                below_support_floor_before=(
                    pattern_before["below_support_floor"] if pattern_before else None
                ),
                below_support_floor_after=(
                    pattern_after["below_support_floor"] if pattern_after else None
                ),
            )
        )
    return rows


def summarize(rows: list[PatternFrequencyDiffRow]) -> dict[str, int]:
    counts = {
        STATUS_RESOLVED: 0,
        STATUS_NEW: 0,
        STATUS_PERSISTED_IMPROVED: 0,
        STATUS_PERSISTED_WORSENED: 0,
        STATUS_PERSISTED_UNCHANGED: 0,
    }
    for row in rows:
        counts[row.status] += 1
    counts["total_signatures"] = len(rows)
    return counts


def _pairs_to_diff(bundles: list[_BundleSummary]) -> list[tuple[_BundleSummary, _BundleSummary]]:
    """Every consecutive pair, plus first-vs-last when more than two bundles are given."""
    pairs = [(bundles[i], bundles[i + 1]) for i in range(len(bundles) - 1)]
    if len(bundles) > 2:
        pairs.append((bundles[0], bundles[-1]))
    return pairs


def _dedupe_labels(bundles: list[_BundleSummary]) -> list[_BundleSummary]:
    """Disambiguate a repeated label (e.g. two bundles under differently-named
    but identically-round-indexed directories) with a positional suffix."""
    seen: dict[str, int] = {}
    deduped: list[_BundleSummary] = []
    for index, bundle in enumerate(bundles):
        count = seen.get(bundle.label, 0)
        seen[bundle.label] = count + 1
        label = bundle.label if count == 0 else f"{bundle.label}_{index}"
        deduped.append(
            _BundleSummary(
                path=bundle.path, label=label, n_runs=bundle.n_runs, patterns_by_key=bundle.patterns_by_key
            )
        )
    return deduped


def _matching_round(bundle_path: Path, inventory: ExperimentInventory) -> RoundRecord | None:
    """The discovered round a bundle belongs to, or None when it belongs to none.

    Matched on the path the loop itself would have written the bundle to
    (``<mining round>/bundle.json``, from discovery) rather than on the
    ``config.round_index`` the bundle carries: a bundle copied out of the tree,
    or one from a different experiment, records a round index just as
    convincingly as an in-tree one, and treating it as that round's evidence
    would attach another experiment's completeness to it.
    """
    resolved = bundle_path.resolve()
    for record in inventory.rounds:
        if (record.mining_round_path / BUNDLE_FILENAME).resolve() == resolved:
            return record
    return None


def bundle_completeness(
    bundles: Sequence[_BundleSummary], inventory: ExperimentInventory
) -> list[BundleCompletenessRow]:
    """One completeness row per bundle, marking which ones the diff may use.

    Only a bundle whose round is KNOWN incomplete is excluded; unknown
    completeness is reported and included (see this module's docstring).
    """
    rows: list[BundleCompletenessRow] = []
    for bundle in bundles:
        record = _matching_round(bundle.path, inventory)
        excluded = record is not None and not record.evidence_complete
        reason = (
            f"round {record.round_index} is not evidence-complete: its mining evidence "
            "marker is absent or disagrees with the persisted records, so a frequency "
            "comparison against it would read the missing evidence as change"
            if excluded and record is not None
            else None
        )
        rows.append(
            BundleCompletenessRow(
                label=bundle.label,
                bundle_path=str(bundle.path),
                round_index=None if record is None else record.round_index,
                round_complete=None if record is None else record.round_complete,
                evidence_complete=None if record is None else record.evidence_complete,
                included=not excluded,
                exclusion_reason=reason,
            )
        )
    return rows


def run_pattern_frequency_diff(
    bundle_paths: Sequence[Path | str],
    snapshot: Snapshot,
    labels: Sequence[str] | None = None,
    *,
    inventory: ExperimentInventory | None = None,
) -> list[Path]:
    """Load every bundle, flag its completeness, diff every includable pair.

    Writes into an already-allocated snapshot (KTD2) rather than a bare output
    directory: the diff's own filenames are derived from bundle labels, so two
    runs over re-mined bundles carrying the same round indices would otherwise
    land on the same names -- the exact clobber the snapshot layer exists to
    make impossible. The bundles are recorded as provenance sources before
    anything is written, because a diff is only interpretable against the exact
    bundle bytes it compared.

    The completeness table is written first and unconditionally, so a run whose
    bundles leave fewer than two comparable rounds still says what it found and
    why it compared nothing -- an empty snapshot would say only that the
    analysis ran.

    Args:
        bundle_paths: Two or more ``bundle.json`` paths, chronological.
        snapshot: The batch's allocated snapshot; everything is written there.
        labels: One label per bundle, for bundles carrying no round index.
        inventory: The caller's already-computed discovery over the snapshot's
            experiment directory, used to match each bundle to its round.
            Discovery does real per-round IO and nothing rewrites the tree
            mid-batch, so a caller that already holds one passes it. Omitted,
            it is discovered here.

    Returns every written CSV path: the completeness table, then one per
    diffed pair.
    """
    if len(bundle_paths) < 2:
        raise ValueError(f"need at least 2 bundle paths to diff, got {len(bundle_paths)}")
    if labels is not None and len(labels) != len(bundle_paths):
        raise ValueError(f"--labels has {len(labels)} entries but {len(bundle_paths)} bundles were given")

    bundles = [
        load_bundle_summary(path, explicit_label=labels[i] if labels else None)
        for i, path in enumerate(bundle_paths)
    ]
    bundles = _dedupe_labels(bundles)
    snapshot.record_sources(bundle.path for bundle in bundles)

    if inventory is None:
        inventory = discover_rounds(snapshot.out_dir)
    completeness = bundle_completeness(bundles, inventory)
    snapshot.record_rounds(row.round_index for row in completeness if row.round_index is not None)
    bundles_path = snapshot.path / BUNDLES_FILENAME
    write_csv(bundles_path, completeness, fieldnames=BUNDLE_FIELDNAMES)
    written: list[Path] = [bundles_path]

    for row in completeness:
        if row.included:
            continue
        sys.stderr.write(f"excluded {row.label} from the comparison: {row.exclusion_reason}\n")
    includable = [bundle for bundle, row in zip(bundles, completeness, strict=True) if row.included]
    if len(includable) < 2:
        sys.stderr.write(
            f"{len(includable)} of {len(bundles)} bundles are comparable; no pair was diffed. "
            f"See {bundles_path}\n"
        )
        return written

    for before, after in _pairs_to_diff(includable):
        rows = diff_bundle_pair(before, after)
        summary = summarize(rows)

        stem = f"pattern_frequency_diff_{before.label}_vs_{after.label}"
        csv_path = snapshot.path / f"{stem}.csv"
        write_csv(csv_path, rows, fieldnames=DIFF_FIELDNAMES)
        written.append(csv_path)

        summary_path = snapshot.path / f"{stem}.json"
        write_json(
            summary_path,
            snapshot.stamp_payload(
                {
                    "before": {
                        "path": str(before.path),
                        "label": before.label,
                        "n_runs": before.n_runs,
                    },
                    "after": {
                        "path": str(after.path),
                        "label": after.label,
                        "n_runs": after.n_runs,
                    },
                    **summary,
                }
            ),
        )

        sys.stdout.write(f"{before.label} vs {after.label}: {summary}\n")
        sys.stdout.write(f"  Wrote {csv_path}\n")
        sys.stdout.write(f"  Wrote {summary_path}\n")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.pattern_frequency_diff",
        description=(
            "Diff mined failure-pattern frequency across two or more bundle.json files "
            "(proposal section 3.5's before/after-optimization comparison)."
        ),
    )
    # The experiment directory is required even though the bundles carry every
    # number this analysis reads: it is what anchors the snapshot location and
    # the identity lookup (KTD2). Without it a diff would be a table with no
    # answer to "which experiment is this about" -- which is the whole failure
    # mode the snapshot layer exists to close.
    parser.add_argument(
        "out_dir", help="the experiment directory the bundles came from (anchors the snapshot)"
    )
    parser.add_argument(
        "bundle_paths",
        nargs="+",
        help="two or more bundle.json paths, in chronological order (round 0 first)",
    )
    add_snapshot_parent_argument(parser)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "one label per bundle path, used when a bundle's config.round_index is "
            "missing (falls back to the bundle's parent directory name if omitted)"
        ),
    )
    args = parser.parse_args(argv)

    if len(args.bundle_paths) < 2:
        sys.stderr.write("need at least 2 bundle paths to diff\n")
        return 1

    out_dir = Path(args.out_dir)
    snapshot = allocate_snapshot(out_dir, parent=args.snapshot_parent)
    # Isolated through ``run_tool``, exactly as the three sibling aggregation
    # CLIs are: a hand-rolled ``except ValueError`` here caught the diff's own
    # argument checks and nothing else, so a malformed bundle (a ``KeyError``
    # out of the signature extraction) or an unreadable bundle path (an
    # ``OSError``) escaped as a traceback and left the allocated directory with
    # no provenance at all. Every failure is now recorded against the tool
    # name, and the snapshot stays on disk as the audit trail while never
    # being selectable as "the latest results".
    snapshot.run_tool(
        TOOL_NAME,
        lambda: run_pattern_frequency_diff(args.bundle_paths, snapshot, labels=args.labels),
    )
    if not snapshot.publish():
        sys.stderr.write(snapshot.failure_message(TOOL_NAME))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "BUNDLES_FILENAME",
    "BUNDLE_FIELDNAMES",
    "DIFF_FIELDNAMES",
    "RATE_EPSILON",
    "STATUS_NEW",
    "STATUS_PERSISTED_IMPROVED",
    "STATUS_PERSISTED_UNCHANGED",
    "STATUS_PERSISTED_WORSENED",
    "STATUS_RESOLVED",
    "TOOL_NAME",
    "BundleCompletenessRow",
    "PatternFrequencyDiffRow",
    "bundle_completeness",
    "diff_bundle_pair",
    "load_bundle_summary",
    "main",
    "run_pattern_frequency_diff",
    "summarize",
]
