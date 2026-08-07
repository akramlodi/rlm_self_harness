"""
Assemble, serialize, and reproduce the evidence bundle B_t.

The bundle is the output of weakness mining and the input to harness proposal.
It deliberately does not prescribe an edit: it separates verifier-level failure
from agent-level mechanism so the proposer can target a reusable weakness
rather than patch a coarse outcome. That property is enforced twice --
structurally, because no field can hold an edit, and by a lint over the
free-text fields.
"""

import hashlib
import json
import os
from datetime import datetime

from shrlm.optimization.types import (
    EvidenceBundle,
    FailurePattern,
    FailureRecord,
    IntegrityReport,
    MiningConfig,
    MiningTotals,
)

# Phrases that indicate a free-text field has drifted from describing a failure
# into recommending a fix. Kept short on purpose: a longer list would start
# rejecting legitimate mechanism descriptions, and this is a lint rather than a
# proof.
PRESCRIPTION_MARKERS: tuple[str, ...] = (
    "we recommend",
    "should be changed",
    "should instead",
    "the fix is",
    "to fix this",
    "instead of",
    "would be better",
)

BUNDLE_FILENAME = "bundle.json"
RECORDS_FILENAME = "records.jsonl"
ATTRIBUTIONS_FILENAME = "attributions.jsonl"


def compute_bundle_id(config: MiningConfig, records: list[FailureRecord]) -> str:
    """
    A deterministic identity for the bundle.

    Excludes the creation timestamp, so two runs of the same round over the
    same records produce the same id and any difference is a real one.
    """
    material = json.dumps(
        {
            "config": config.to_dict(),
            "records": sorted(record.instance_id for record in records),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_integrity_report(
    records: list[FailureRecord], digest_coverages: list[float]
) -> IntegrityReport:
    return IntegrityReport(
        total_suspected_lost_subcalls=sum(
            record.stats.suspected_lost_subcalls for record in records
        ),
        n_records_with_lost_subcalls=sum(
            1 for record in records if record.stats.suspected_lost_subcalls > 0
        ),
        n_records_unreliable_block_attribution=sum(
            1 for record in records if not record.stats.block_attribution_reliable
        ),
        n_indeterminate_nodes=sum(record.stats.n_indeterminate for record in records),
        mean_digest_coverage=(
            sum(digest_coverages) / len(digest_coverages) if digest_coverages else 1.0
        ),
    )


def build_evidence_bundle(
    config: MiningConfig,
    records: list[FailureRecord],
    patterns: list[FailurePattern],
    marginals: dict[str, dict[str, int]],
    totals: MiningTotals,
    digest_coverages: list[float],
    created_at: str | None = None,
) -> EvidenceBundle:
    """Assemble B_t and refuse to emit it if any field prescribes an edit."""
    bundle = EvidenceBundle(
        bundle_id=compute_bundle_id(config, records),
        created_at=created_at or datetime.now().isoformat(),
        config=config,
        totals=totals,
        patterns=patterns,
        marginals=marginals,
        integrity=build_integrity_report(records, digest_coverages),
    )
    assert_no_prescription(bundle)
    return bundle


def assert_no_prescription(bundle: EvidenceBundle) -> None:
    """
    Reject a bundle whose free text recommends a change.

    The separation between the evaluation system and the optimizer is a claim
    the paper makes about the method; this turns it into something the build
    can check.
    """
    for pattern in bundle.patterns:
        fields = [*pattern.shared_symptoms, *pattern.verifier_evidence]
        for text in fields:
            lowered = text.lower()
            for marker in PRESCRIPTION_MARKERS:
                if marker in lowered:
                    raise ValueError(
                        f"Evidence bundle prescribes a harness edit "
                        f"({marker!r} in {text!r}). The bundle must describe failures only."
                    )


def write_bundle(
    bundle: EvidenceBundle,
    records: list[FailureRecord],
    out_dir: str,
    raw_attributions: list[dict] | None = None,
) -> str:
    """
    Write the round's artifacts.

    The bundle is pretty-printed because it is read by a person and diffed
    across rounds; the records and raw responses are JSONL because they are
    streamed and appended.
    """
    round_dir = os.path.join(out_dir, f"round_{bundle.config.round_index:02d}")
    os.makedirs(round_dir, exist_ok=True)

    bundle_path = os.path.join(round_dir, BUNDLE_FILENAME)
    with open(bundle_path, "w") as handle:
        json.dump(bundle.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    with open(os.path.join(round_dir, RECORDS_FILENAME), "w") as handle:
        for record in records:
            json.dump(record.to_dict(), handle, sort_keys=True, default=str)
            handle.write("\n")

    if raw_attributions is not None:
        with open(os.path.join(round_dir, ATTRIBUTIONS_FILENAME), "w") as handle:
            for entry in raw_attributions:
                json.dump(entry, handle, sort_keys=True, default=str)
                handle.write("\n")

    return bundle_path


def bundle_without_timestamp(bundle: EvidenceBundle) -> dict:
    """The bundle as it should be compared: everything except when it was made."""
    payload = bundle.to_dict()
    payload.pop("created_at")
    return payload
