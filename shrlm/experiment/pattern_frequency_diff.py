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
"""

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class _BundleSummary:
    """One loaded bundle, reduced to what the diff needs."""

    path: Path
    label: str
    n_runs: int
    patterns_by_key: dict[tuple[str, str, str, str], dict[str, Any]]


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
    signature = pattern["signature"]
    return tuple(signature[field_name] for field_name in SIGNATURE_FIELDS)  # type: ignore[return-value]


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


def _write_csv(path: Path, rows: list[PatternFrequencyDiffRow]) -> None:
    fieldnames = [
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
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


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


def run_pattern_frequency_diff(
    bundle_paths: Sequence[Path | str], out_dir: Path | str, labels: Sequence[str] | None = None
) -> list[Path]:
    """Load every bundle, diff every pair (see ``_pairs_to_diff``), write CSV + summary JSON each.

    Returns the list of written CSV paths, one per diffed pair.
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for before, after in _pairs_to_diff(bundles):
        rows = diff_bundle_pair(before, after)
        summary = summarize(rows)

        csv_path = out_dir / f"pattern_frequency_diff_{before.label}_vs_{after.label}.csv"
        _write_csv(csv_path, rows)
        written.append(csv_path)

        summary_path = out_dir / f"pattern_frequency_diff_{before.label}_vs_{after.label}.json"
        summary_payload = {
            "before": {"path": str(before.path), "label": before.label, "n_runs": before.n_runs},
            "after": {"path": str(after.path), "label": after.label, "n_runs": after.n_runs},
            **summary,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")

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
    parser.add_argument(
        "bundle_paths",
        nargs="+",
        help="two or more bundle.json paths, in chronological order (round 0 first)",
    )
    parser.add_argument("--out", dest="out_dir", required=True, help="directory to write CSV/JSON into")
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

    try:
        run_pattern_frequency_diff(args.bundle_paths, args.out_dir, labels=args.labels)
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "RATE_EPSILON",
    "STATUS_NEW",
    "STATUS_PERSISTED_IMPROVED",
    "STATUS_PERSISTED_UNCHANGED",
    "STATUS_PERSISTED_WORSENED",
    "STATUS_RESOLVED",
    "PatternFrequencyDiffRow",
    "diff_bundle_pair",
    "load_bundle_summary",
    "main",
    "run_pattern_frequency_diff",
    "summarize",
]
