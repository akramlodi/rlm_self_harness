"""
Assemble, serialize, and reproduce the evidence bundle B_t.

The bundle is the output of weakness mining and the input to harness proposal.
It deliberately does not prescribe an edit: it separates verifier-level failure
from agent-level mechanism so the proposer can target a reusable weakness
rather than patch a coarse outcome. That property is enforced twice --
structurally, because no field can hold an edit, and by a lint over the
prose mining itself composes (quoted model output is exempt; see
``assert_no_prescription``).
"""

import hashlib
import json
import os
from datetime import datetime

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import (
    KNOWN_SUBSTRATE_BIASES,
    AttributionErrorKind,
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


def _is_transport_error(record: FailureRecord) -> bool:
    """Whether a record's attribution failed at the transport level.

    Reads the typed ``attribution_error_kind`` mining stamps on the record,
    falling back to the ``transport failure`` prefix of ``attribution_error``
    for legacy records that predate the field.
    """
    if record.attribution_error_kind is not None:
        return record.attribution_error_kind is AttributionErrorKind.TRANSPORT
    return record.attribution_error.startswith("transport failure")


def build_integrity_report(
    records: list[FailureRecord], digest_coverages: list[float]
) -> IntegrityReport:
    """A pure function of the records: every count is recomputable from
    records.jsonl. Transport failures are recognized by the typed
    ``attribution_error_kind`` mining stamps on each record (with a legacy
    prefix fallback; see ``_is_transport_error``), which keeps the count
    derivable from the records rather than from process state."""
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
        n_unattributed=sum(1 for record in records if record.attribution_failed),
        n_ungrounded=sum(1 for record in records if not record.level_grounded),
        n_resource_terminated=sum(
            1 for record in records if record.verdict.cause is VerifierCause.RESOURCE_TERMINATED
        ),
        n_transport_errors=sum(1 for record in records if _is_transport_error(record)),
        known_substrate_biases=list(KNOWN_SUBSTRATE_BIASES),
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
    Reject a bundle whose mining-generated prose recommends a change.

    The separation between the evaluation system and the optimizer is a claim
    the paper makes about the method; this turns it into something the build
    can check.

    Scope: only text mining itself composes is linted -- today that is
    ``shared_symptoms``, the arithmetic statements over tree statistics.
    Fields that quote model output verbatim are exempt: ``verifier_evidence``
    embeds ``verdict.produced`` / ``verdict.gold``, so a wrong answer that
    happens to contain "instead of" is evidence to preserve, not a
    recommendation, and linting it would crash bundle emission on the model's
    own words.
    """
    for pattern in bundle.patterns:
        for text in pattern.shared_symptoms:
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
    overwrite: bool = False,
    bundle_dir: str | None = None,
) -> str:
    """
    Write the round's artifacts, refusing to clobber audited evidence.

    The bundle is pretty-printed because it is read by a person and diffed
    across rounds; the records and raw responses are JSONL because they are
    streamed and appended. ``bundle.json`` is written atomically (to a
    same-directory ``.tmp`` file, then ``os.replace``), so a crash mid-write
    can never leave a truncated bundle behind.

    ``bundle_dir``, when given, names the directory the triplet
    (``bundle.json``, ``records.jsonl``, ``attributions.jsonl``) lands in
    DIRECTLY -- unlike ``out_dir``, which is an experiment directory that
    internally appends ``round_NN``. This is how two bundles coexist under
    one round (the sub-verification ablation writes its second mode into
    ``round_NN/bundles/<label>/``); only the triplet moves, every shared
    round artifact (manifest, traces, digests, prompts, the attribution
    cache) stays at the round root. The default keeps today's layout: the
    triplet lands at ``<out_dir>/round_NN``. The non-clobber guard below is
    applied within whichever destination is used, so each destination's
    evidence is protected independently.

    An existing ``bundle.json`` in the destination directory is overwritten only
    when the new content is byte-identical (an idempotent re-write). A
    different bundle_id means a different round was already persisted here;
    the same id with different bytes means a re-mine diverged -- typically a
    cold attribution cache resampling responses, or a new ``created_at``
    timestamp (pass the recorded one to reproduce). Both cases raise instead
    of silently replacing what may already have been audited, as does an
    existing file that no longer parses as a bundle (corruption from outside
    this writer).

    ``overwrite=True`` skips the guard entirely and deliberately replaces
    whatever ``bundle.json`` already holds -- the explicit escape hatch for a
    poisoned round (e.g. a re-mine after an attributor outage produced
    divergent content). It replaces evidence that may already have been
    audited, so it must be an operator's stated intent, never a default.
    """
    destination = (
        bundle_dir
        if bundle_dir is not None
        else os.path.join(out_dir, f"round_{bundle.config.round_index:02d}")
    )
    os.makedirs(destination, exist_ok=True)

    bundle_path = os.path.join(destination, BUNDLE_FILENAME)
    payload = json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n"
    if not overwrite and os.path.exists(bundle_path):
        with open(bundle_path) as handle:
            existing_text = handle.read()
        try:
            existing_payload = json.loads(existing_text)
            existing_id = str(existing_payload["bundle_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"{bundle_path} exists but does not parse as a bundle "
                f"({type(error).__name__}: {error}); the file is corrupt -- e.g. a "
                "truncated or interrupted write. Delete the corrupt file (or pass "
                "overwrite=True) to let a re-mine regenerate it."
            ) from error
        if existing_id != bundle.bundle_id:
            raise ValueError(
                f"{bundle_path} already holds bundle {existing_id}, which is not "
                f"bundle {bundle.bundle_id}; refusing to overwrite one round's "
                "evidence with another's."
            )
        if existing_text != payload:
            timestamp_only = existing_payload == {
                **bundle.to_dict(),
                "created_at": existing_payload.get("created_at"),
            }
            reason = (
                "only created_at differs; pass the recorded created_at to reproduce it"
                if timestamp_only
                else "a re-mine produced different content (e.g. a cold attribution cache)"
            )
            raise ValueError(
                f"{bundle_path} already holds bundle {bundle.bundle_id} with different "
                f"bytes -- the rewrite diverges from the persisted evidence ({reason}). "
                "Refusing to replace audited evidence; pass overwrite=True to replace "
                "it deliberately."
            )
    tmp_path = bundle_path + ".tmp"
    with open(tmp_path, "w") as handle:
        handle.write(payload)
    os.replace(tmp_path, bundle_path)

    with open(os.path.join(destination, RECORDS_FILENAME), "w") as handle:
        for record in records:
            json.dump(record.to_dict(), handle, sort_keys=True, default=str)
            handle.write("\n")

    if raw_attributions is not None:
        with open(os.path.join(destination, ATTRIBUTIONS_FILENAME), "w") as handle:
            for entry in raw_attributions:
                json.dump(entry, handle, sort_keys=True, default=str)
                handle.write("\n")

    return bundle_path


def bundle_without_timestamp(bundle: EvidenceBundle) -> dict:
    """The bundle as it should be compared: everything except when it was made."""
    payload = bundle.to_dict()
    payload.pop("created_at")
    return payload
