"""
Propose minimal, targeted harness edits from a mined evidence bundle.

Stage 2 of the self-harness loop (docs/handoff-harness-proposal.md,
docs/harness-proposal-interface.md). Given a mining round's ``bundle.json``, the
current (incumbent) harness, which runs already pass, and the prior validation
rounds' promotion ledgers, a fixed proposer model is shown the mined failure
patterns and asked to propose up to K candidate edits, each targeting exactly
one pattern on exactly one eligible surface from
``shrlm.optimization.taxonomy.MECHANISM_SURFACES``. The model declares unique
selections before their matching replacements. Every candidate is written as a ``shrlm-proposal/v1``
``proposal.json`` (``docs/harness-proposal-interface.md``), ready for
``shrlm.optimization.candidates.load_candidates``.

This module follows ``attribution.py``'s shape throughout: closed-vocabulary
validation with named violations and a re-ask loop, a JSONL replay cache keyed
over everything that changes what the model is asked, and a full per-attempt
audit trail. It differs in one respect attribution does not need: an accepted
response must still *materialize* into a live, one-surface-diff ``Harness``
before it becomes a proposal, so a second failure mode exists past JSON
validation -- a candidate whose JSON was well-formed but whose edit does not
materialize cleanly is dropped with a recorded reason rather than failing the
whole round (mirrors ``WeaknessMiner``: one bad candidate must not abort a
round).

Non-text surfaces are authored declaratively, never freehand JSON:
    - S1-S5 (prompt text): the model writes full replacement text.
    - S6 (runtime policy): the model writes a policy dict restricted to the
      harness's own known keys.
    - S7 / S9 (callables): the model writes one undecorated top-level Python
      function; it is written to a scratch file and imported so
      ``inspect.getsource`` recovers it exactly, exactly like the stage-3
      loader's own callable materialization (``candidates.py``, reused here
      directly rather than re-implemented).
    - S8 (repl helpers): the model writes one named helper function, added or
      replacing one entry in ``repl_helpers`` or ``sub_repl_helpers``.
    - S10 (skills): the model writes one named skill (name / description /
      body), added or replacing the incumbent skill of the same name, like
      an S8 helper; sending the name with description and body both empty
      removes that skill instead. Data, not code: index fields are brace-free
      so the assembled prompt still survives ``str.format``, bodies are stored
      raw for the runner's fixed loader to return verbatim, and every field is
      scanned for a runtime limit ``check_stated_limits`` governs
      (R5, R6, R7, R14). The merge against the incumbent library is dry-run
      at validation time so a cap overflow or unknown removal name is
      re-asked with coaching, not silently dropped at materialization.

A materialized candidate's one-surface-diff is checked with
``candidates.changed_surfaces`` -- the real gate's own diffing function -- so
"does this pass the surface_diff gate" is answered with the gate's own code,
never a reimplementation that could drift from it.
"""

import ast
import hashlib
import json
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from string import Formatter
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.llm_observation import ObservationPersistenceError, observation_refs, observe_call
from rlm.environments.base_env import RESERVED_TOOL_NAMES
from rlm.utils.exceptions import TokenLimitExceededError
from shrlm.harness_identity import (
    HARNESS_FORMAT,
    HarnessSerializationError,
    canonical_json,
    hash_of_serialization,
    serialize_harness,
)
from shrlm.optimization.behavior import BEHAVIOR_SCHEMA
from shrlm.optimization.candidates import (
    BEHAVIOR_FIELDS,
    CANDIDATE_MODULE_PREAMBLE,
    DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
    SURFACE_SERIALIZATION_KEYS,
    CandidateRejection,
    behavioral_difference_violation,
    changed_surfaces,
    import_surface_module,
    load_candidate,
    select_preflight_profile,
    validate_preflight_profile,
)
from shrlm.optimization.driver import (
    INSTANCES_FILE,
    MANIFEST_FILE,
    canonical_manifest_entries,
)
from shrlm.optimization.history import (
    HISTORY_BUDGET_CHARS,
    HISTORY_SCHEMA,
    compact_history,
    prior_attempts,
    revision_violation,
    select_predecessor,
    surface_fingerprint,
)
from shrlm.optimization.llm_observation_store import observation_recorder, verify_observations
from shrlm.optimization.response_cache import ResponseCache
from shrlm.optimization.skill_edit import (
    SKILLS_EDIT_FORMAT,
    SkillEditRejection,
    _merge_skill,
    _validate_skill_edit,
)
from shrlm.optimization.taxonomy import (
    CAPABILITY_VERSION,
    MECHANISM_SURFACE,
    AgentMechanism,
    eligible_surfaces,
    render_surface_block,
)
from shrlm.rlm_harness import (
    SKILL_BODY_MAX_CHARS,
    SKILL_BODY_MIN_STEPS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_LOADER_NAME,
    SKILL_MAX_ENTRIES,
    SKILL_NAME_MAX_CHARS,
    SKILL_RECORD_FIELDS,
    SKILL_TOTAL_MAX_CHARS,
    Harness,
    SkillEntry,
    build_runtime_policy,
)
from shrlm.runner import declared_metadata_bound

PROPOSAL_FORMAT = "shrlm-proposal/v1"
TEXT_CONTRACT = "literal-text/v1"
RESPONSE_FORMAT_VERSION = "proposal-selection/v2"
# ``HARNESS_FORMAT`` is imported from ``shrlm.harness_identity`` (the single
# declaration site) and re-exported here for the proposal writer.
PROPOSAL_FILENAME = "proposal.json"

# 1.8.0: environment-summary patterns (synthetic, mechanism OTHER) are named
# and scoped -- informational counts of environment-terminated runs; the
# proposer may, at its own judgment, propose harness-side handling against
# one, with skipping as the expected default.
# 1.7.0: proposal-quality guidance from the 2026-09-01 dsv4f round-1 postmortem
# (all three candidates rejected): causal-reach triage, recoverable-over-veto,
# the zero-regression promotion contract, resource cost as a regression axis,
# below-floor patterns as hypotheses, candidate diversity, and the
# resource_terminated-means-efficiency note. 1.4.0: the S10 removal form is
# documented, the S10 bullet is compact with the pedagogy appended only when a
# pattern targets S10, and the S10 pattern block carries a full-library
# inventory line. 1.3.0: S10 edit is one skill added or replaced by name
# (S8-style), not a whole-list rewrite.
# 1.9.0: the prior-edit history covers every completed round, not only rounds
# that reached validation, renders each attempted edit's predicted effect, and
# says that a candidate identical to the current surface is refused before
# validation (see VALIDATOR_VERSION 1.5.0).
PROMPT_VERSION = "4.2.0"
# Version of the validation logic in this module (validate_candidate_spec,
# _validate_edit_shape, _validate_single_def, skill_edit._validate_skill_edit).
# Folded into the cache key so a validator change cannot replay stale responses
# judged under different rules. 1.3.0: the S10 merge is dry-run against the
# incumbent at validation (over-cap and unknown-removal edits are re-asked) and
# the removal form is accepted. 1.2.0: S10 is one named skill, merged by name.
# 1.3.1: strict-parse failures get a bounded unescaped-quote repair pass
# (_repair_unescaped_quotes) before rejection -- responses judged invalid
# under 1.3.0 may parse under it, so cached rejections must not replay.
# 1.5.0: a batch in which no candidate materializes as a change to the
# incumbent (a no-op edit that reproduces the current surface) is rejected and
# re-asked with the reason, and exhaustion whose final attempt failed only at
# materialization returns an empty result instead of raising. The 2026-09-10
# OOLONG-Pairs run lost rounds 4-6 to a proposer that re-emitted the incumbent's
# own S9 three rounds in a row; under 1.4.0 that was counted, never re-asked.
VALIDATOR_VERSION = "4.2.0"

DEFAULT_K = 4
# Raised from 3 on 2026-08-24: stealth/ox-alpha exhausted 3 attempts twice in
# one round (an out-of-surface pattern_index, then malformed JSON), and each
# exhaustion aborts the whole experiment. Rejected attempts are cached, so a
# resume replays them verbatim -- the attempt budget is the only re-sampling
# a run ever gets. max_attempts is in the proposer's cache-key material, so
# this change orphans rows cached under 3 rather than replaying them. A
# compliant proposer rarely needs attempt 2; extra headroom only spends when
# validation already failed.
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_TRANSPORT_RETRIES = 3
DEFAULT_TRANSPORT_BACKOFF_SECONDS = 0.5

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

# Deterministic programming/contract errors that must propagate rather than be
# retried as transient transport failures. Same rationale and set as
# attribution.py's NON_TRANSPORT_ERRORS.
NON_TRANSPORT_ERRORS = (
    TypeError,
    AttributeError,
    KeyError,
    ValueError,
    ObservationPersistenceError,
)

# S1-S5: the Harness field each string surface fills.
TEXT_SURFACE_FIELDS: dict[str, str] = {
    "S1": "repl_contract",
    "S2": "decomposition_instruction",
    "S3": "execution_instruction",
    "S4": "verification_instruction",
    "S5": "recovery_instruction",
}
# S7 / S9: the Harness field each callable surface fills.
CODE_SURFACE_FIELDS: dict[str, str] = {"S7": "metadata", "S9": "answer_middleware"}
REPL_HELPER_DICTS: tuple[str, str] = ("repl_helpers", "sub_repl_helpers")
S6_KEYS: tuple[str, ...] = tuple(sorted(build_runtime_policy()))

EDIT_KIND_TEXT = "text"
EDIT_KIND_POLICY = "policy"
EDIT_KIND_CODE = "code"
EDIT_KIND_REPL_HELPER = "repl_helper"
EDIT_KIND_SKILLS = "skills"

# S10 edit bounds (R7) live in ``shrlm.rlm_harness`` so the runner enforces the
# same numbers at harness construction (U7) without importing this module; the
# caps the prompt cites are re-exported here under their original names. The S10
# record validators (shape, bounds, R5/R6/R14) live in ``skill_edit``.

# The one edit shape each surface accepts -- fixed by the harness's own field
# types, not a proposer choice.
SURFACE_EDIT_KIND: dict[str, str] = {
    **dict.fromkeys(TEXT_SURFACE_FIELDS, EDIT_KIND_TEXT),
    "S6": EDIT_KIND_POLICY,
    "S7": EDIT_KIND_CODE,
    "S9": EDIT_KIND_CODE,
    "S8": EDIT_KIND_REPL_HELPER,
    "S10": EDIT_KIND_SKILLS,
}


class ProposalRejection(Exception):
    """A response that did not satisfy the output contract.

    Raised both for a single field-level violation (caught and folded into the
    re-ask loop) and, carrying the full ``attempts`` audit trail, when no
    attempt in a round ever validates -- exactly attribution.py's
    AttributionRejection split.
    """

    def __init__(self, message: str, attempts: list["ProposalAttempt"] | None = None):
        super().__init__(message)
        self.attempts: list[ProposalAttempt] = attempts or []


class ProposalTransportError(Exception):
    """The LM call itself failed (network, rate limit, server error) after
    bounded retries. The model never produced a judgable response, so the
    caller should checkpoint the round rather than record a rejected batch."""

    def __init__(self, message: str, attempts: list["ProposalAttempt"] | None = None):
        super().__init__(message)
        self.attempts: list[ProposalAttempt] = attempts or []


class ProposalBudgetExhausted(Exception):
    """The client raised ``TokenLimitExceededError``: the proposer spent its
    whole output budget on reasoning and returned no content (R6/KTD3).

    The parallel of ``attribution.AttributionBudgetExhausted``. Deterministic
    for the prompt at temperature 0, so it is neither a transport glitch to
    re-send nor a rejected attempt to re-ask: it surfaces immediately with the
    attempts made so far, and the proposal stage records a round with zero
    candidates.
    """

    def __init__(self, message: str, attempts: list["ProposalAttempt"] | None = None):
        super().__init__(message)
        self.attempts: list[ProposalAttempt] = attempts or []


class MaterializationFailure(Exception):
    """A schema-valid candidate whose edit did not materialize cleanly.

    Distinct from ProposalRejection: the JSON was fine, but building the live
    Harness failed (unserializable S6 value, generated source that does not
    yield a clean one-surface diff, ...). Isolated per candidate so it cannot
    abort the rest of the batch.
    """

    def __init__(self, pattern_index: int, reason: str):
        super().__init__(reason)
        self.pattern_index = pattern_index
        self.reason = reason


@dataclass(frozen=True)
class ProposalAttempt:
    """One attempt in the re-ask loop, kept for the audit trail."""

    attempt: int
    cached: bool
    raw_response: str
    accepted: bool
    violation: str = ""
    admissions: list[dict[str, Any]] = field(default_factory=list)
    llm_observations: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "attempt": self.attempt,
            "cached": self.cached,
            "raw_response": self.raw_response,
            "accepted": self.accepted,
            "violation": self.violation,
        }
        if self.llm_observations is not None:
            result["llm_observations"] = self.llm_observations
        if self.admissions:
            result["admissions"] = self.admissions
        return result


@dataclass(frozen=True)
class CandidateSpec:
    """One validated (but not yet materialized) candidate from the model."""

    pattern_index: int
    pattern: dict[str, Any]
    surface: str
    edit: dict[str, Any]
    predicted_effect: str
    regression_risks: list[str]
    incumbent_behavior: str = ""
    observed_failure: str = ""
    behavioral_change: str = ""
    text_contract: str = ""
    revision: dict[str, Any] | None = None
    revision_unchanged: bool | None = None


@dataclass(frozen=True)
class WrittenProposal:
    """One candidate successfully written to ``<proposals_dir>/<candidate_id>``."""

    candidate_id: str
    path: Path
    surface: str
    pattern_index: int


@dataclass(frozen=True)
class MaterializationFailureRecord:
    """One validated candidate that failed to materialize (R4).

    Carries the candidate's own predicted effect so the persisted marker and
    the next round's history can say what direction was refused, not only
    which surface.
    """

    pattern_index: int
    surface: str
    reason: str
    predicted_effect: str = ""
    behavior: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_index": self.pattern_index,
            "surface": self.surface,
            "reason": self.reason,
            "predicted_effect": self.predicted_effect,
            "behavior": self.behavior,
        }


@dataclass(frozen=True)
class ProposalRoundResult:
    """The full audit record of one ``propose_round`` call."""

    written: list[WrittenProposal]
    skipped_patterns: list[int]
    materialization_failures: list[MaterializationFailureRecord]
    attempts: list[ProposalAttempt]
    prompt_sha256: str
    preflight_failures: list[dict[str, Any]] = field(default_factory=list)
    preflight_profile: str = "generic/v1"
    evidence_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposerConfig:
    """Everything that changes what the proposer produces. Hashed into the
    cache key so a config change cannot silently reuse proposals made under
    different conditions. Transport retry knobs are deliberately excluded:
    they change when a response is obtained, never what response is produced.
    """

    k: int = DEFAULT_K
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    prompt_version: str = PROMPT_VERSION
    validator_version: str = VALIDATOR_VERSION
    transport_retries: int = DEFAULT_TRANSPORT_RETRIES
    transport_backoff_seconds: float = DEFAULT_TRANSPORT_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.transport_retries < 1:
            raise ValueError(f"transport_retries must be >= 1, got {self.transport_retries}")
        if self.transport_backoff_seconds < 0:
            raise ValueError(
                f"transport_backoff_seconds must be >= 0, got {self.transport_backoff_seconds}"
            )


class ProposalCache(ResponseCache):
    """Persistent text and observation cache; existing response keys stay unchanged."""


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _pattern_mechanism(pattern: dict[str, Any]) -> AgentMechanism | None:
    try:
        return AgentMechanism(pattern.get("signature", {}).get("agent_mechanism"))
    except ValueError:
        return None


def _pattern_surface(pattern: dict[str, Any]) -> str | None:
    """The PRIMARY surface a pattern's mechanism implicates (the bundle's
    recorded ``surface``), or None for OTHER / an unrecognized value."""
    mechanism = _pattern_mechanism(pattern)
    if mechanism is None:
        return None
    surface = MECHANISM_SURFACE.get(mechanism)
    return surface.value if surface is not None else None


def _pattern_surfaces(pattern: dict[str, Any]) -> list[str]:
    """Every surface the proposer may target for this pattern, primary first
    (``MECHANISM_SURFACES``); empty for an unrecognized mechanism."""
    if "selectable" in pattern:
        return pattern.get("eligible_surfaces", []) if pattern["selectable"] else []
    mechanism = _pattern_mechanism(pattern)
    if mechanism is None:
        return []
    return [
        surface.value
        for surface in eligible_surfaces(
            mechanism,
            pattern.get("route_support"),
            causal_status=pattern["signature"].get("causal_status"),
        )
    ]


def _serialized_record_chars(record: dict[str, Any]) -> int:
    """One serialized S10 record's character weight against the total cap."""
    return sum(len(record[field]) for field in SKILL_RECORD_FIELDS)


def _skill_inventory_line(skills: list[dict[str, Any]]) -> str:
    """A compact full-library S10 inventory: every existing name with its char
    weight, plus the used entry slots and total chars. Merge-by-name needs the
    complete name list and the remaining budget even when the displayed value
    above it is truncated at 4000 chars; at the caps (8 entries) this line
    stays bounded."""
    used = sum(_serialized_record_chars(record) for record in skills)
    names = (
        ", ".join(f"{record['name']} ({_serialized_record_chars(record):,})" for record in skills)
        or "(empty)"
    )
    return (
        f"S10 inventory: {len(skills)}/{SKILL_MAX_ENTRIES} entries, "
        f"{used:,}/{SKILL_TOTAL_MAX_CHARS:,} chars used; names: {names}"
    )


def _render_verifier_contract(verifier_config: dict[str, Any] | None) -> str:
    """One line naming what the environment's verifier accepts.

    Without it the proposer guesses the answer contract from failure symptoms:
    experiment_kimi's ``r03-c04-s1`` told the model to submit ``"[id1, id2]"``
    -- a correct reading of the quotes symptom, implemented against a marker
    rule it could not see -- and scored 0/96.
    """
    if not verifier_config:
        return "Verifier contract: not recorded for this bundle."
    parts = [f"{key}={verifier_config[key]}" for key in sorted(verifier_config)]
    return "Verifier contract (what the environment accepts): " + ", ".join(parts)


def literal_surface_text(template: str, marker: str) -> str:
    """Decode stored templates for proposal display, without inserting tools."""
    parts = []
    for literal, field_name, format_spec, conversion in Formatter().parse(template):
        parts.append(literal)
        if field_name is not None:
            if field_name != "custom_tools_section" or format_spec or conversion:
                raise ValueError(f"unsupported incumbent template field: {field_name}")
            parts.append(marker)
    return "".join(parts)


def text_slot_marker(serialization: dict[str, Any]) -> str:
    """Choose a round-stable marker absent from every incumbent literal span."""
    literals = [
        literal_surface_text(serialization["surfaces"][SURFACE_SERIALIZATION_KEYS[s][0]], "")
        for s in TEXT_SURFACE_FIELDS
    ]
    number = 0
    while True:
        suffix = f"_{number}" if number else ""
        marker = f"<<custom_tools_section{suffix}>>"
        if all(marker not in text for text in literals):
            return marker
        number += 1


def render_current_surfaces(
    addressable: Sequence[tuple[int, dict[str, Any]]],
    incumbent_serialization: dict[str, Any],
) -> str:
    eligible = {surface for _, pattern in addressable for surface in _pattern_surfaces(pattern)}
    marker = text_slot_marker(incumbent_serialization)
    blocks = [
        "Complete current surfaces (shared by all patterns; replacements must preserve useful behavior):",
        f"Text contract {TEXT_CONTRACT}: S1-S5 are literal text. Do not double braces. "
        f"The reserved marker {marker} inserts custom tools; preserve it where present. "
        "All other braces are literal, including {custom_tools_section}. "
        "JSON string escaping is still required for response transport.",
    ]
    for surface in sorted(eligible, key=lambda value: int(value[1:])):
        current = {
            key: literal_surface_text(incumbent_serialization["surfaces"][key], marker)
            if surface in TEXT_SURFACE_FIELDS
            else incumbent_serialization["surfaces"][key]
            for key in SURFACE_SERIALIZATION_KEYS[surface]
        }
        blocks.append(f"current {surface} value:\n" + json.dumps(current, indent=2, sort_keys=True))
    if "S10" in eligible:
        blocks.append(_skill_inventory_line(incumbent_serialization["surfaces"]["S10_skills"]))
    return "\n\n".join(blocks)


CALLABLE_CONTRACT = (
    """Candidate callable contract:
S9 takes exactly (answer, repl_inventory) and returns AnswerDecision.accept(answer)
or AnswerDecision.redirect(nudge). There is no AnswerDecision.reject method.
repl_inventory contains redacted type/length tuples, e.g. {'context': ('str', 19006)},
never variable contents. Do not call string methods on those tuples.
Reserve S9 for answer-visible defects. For semantic aggregation or predicate
mistakes, including those diagnosed as other, prefer S3/S4: S9 cannot inspect
the records or verify their classifications.
Available imports for candidate functions:
"""
    + CANDIDATE_MODULE_PREAMBLE
    + """
New text instructions use the literal text contract shown with current surfaces.
The host handles template encoding; write Python/JSON examples with literal braces.
"""
)


# The history outcome of a candidate that validated but failed to materialize
# (its edit reproduced the incumbent). Synthesized by the orchestrator from the
# proposals marker; never written to a validation ledger.
HISTORY_NOT_MATERIALIZED = "not_materialized"


def _render_history_block(
    prior_history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
    patterns: Sequence[dict[str, Any]] = (),
) -> str:
    """Compact recent facts before optional verbose context; archive stays complete."""
    if not prior_history:
        return "No prior rounds exist yet; this is the first proposal round."
    rendered, _ = compact_history(
        prior_history, mechanisms={p.get("signature", {}).get("agent_mechanism") for p in patterns}
    )
    return rendered


PROPOSER_INTRO = """\
You are proposing minimal, targeted edits to the harness of a recursive language \
model (RLM). A recursive language model keeps its input in a REPL variable, writes \
code to split that input into pieces, issues sub-calls over the pieces, and combines \
the results. You are shown failure patterns mined from the model's own past runs, \
each already clustered by a verifier-grounded signature. Each pattern lists the \
harness surfaces on which its mechanism can be addressed, primary first; a \
candidate edits exactly ONE of those surfaces, and you choose which. Prefer the \
surface where a minimal edit most directly removes the mechanism -- an answer \
format problem is an S9 normalization or an S1 contract line before it is a \
decomposition instruction; a procedure the root keeps re-deriving is an S10 skill.

The ten editable surfaces:
"""

PROPOSER_TASK = """\
Propose at most one edit per surface in this round. All admitted edits will be \
evaluated as one combined candidate, with one shared promotion decision. The \
candidate limit is a maximum, not a quota. When patterns compete for a surface, \
choose the best-supported minimal edit. Do not move an edit to a weaker surface \
just to fill the batch. If only one surface warrants a change, propose one edit; \
if none does, return empty selections and candidates lists.
Select in priority order, removing each chosen surface and pattern from further \
consideration. The host retains the first candidate in candidate order that \
passes all local gates for a surface and pattern; later contenders cannot replace \
that owner. Invalid candidates do not reserve a surface.

Before replacement text, describe incumbent_behavior (what the relevant execution \
actually did at its latest relevant state, distinguishing it from instructions), \
observed_failure (what remains wrong after later checks or recovery and what is \
unverified), and behavioral_change (a changed action/value at that operation and \
a minimal example for which the old and new behavior differ). Each is a short string, at most 600 characters. \
Repeating an existing instruction more emphatically is insufficient justification. \
If there is no concrete difference, withdraw the candidate instead of supplying \
"none", "no change", "no-op", or "unchanged". Local checks enforce shape and literal \
changes, not semantic novelty; you must assess the behavioral difference.
Minimal means the smallest effective change. Replace or clearly scope a misleading \
worked example when its return contract conflicts with the procedure needed here; \
do not append another rule while leaving an incompatible example as the default. \
Preserve an explicitly scoped example when it remains correct for simpler tasks.

For each pattern you choose to address, propose exactly one minimal edit on one of \
its eligible surfaces, naming that surface in the candidate's "surface" field -- \
change only what is needed to address that specific mechanism, never a broad rewrite. You do not have to address every pattern: skip a pattern if no \
minimal edit plausibly addresses it (weak support, or the failure looks like a \
model-capability limit rather than a harness gap). Do not invent a mechanism or a \
surface; only cite patterns from the list below by their bracketed index.

Before proposing against a pattern, ask whether its mechanism is something the \
harness's own text or policy plausibly caused. A failure whose evidence points to \
the environment, the provider, or the run's resource limits is not a harness gap; \
skip it, or address only the harness-side handling of it. One exception cuts the \
other way: a pattern whose verifier cause is resource_terminated reflects a run \
that exhausted a resource limit -- its evidence names whether time or spend ran \
out -- and that IS harness-addressable: the remedy is efficiency (fewer \
iterations, less redundant work, tighter decomposition per answer), not more \
checking.

A pattern whose support is below the floor is a hypothesis, not evidence. Spend a \
candidate on it only when the edit would be harmless even if the pattern is \
spurious.

A pattern whose symptoms begin with "environment summary (synthetic)" is not a \
mined failure mechanism: it is a count of runs the environment terminated \
(provider refusals and the like), with no agent behavior recorded. The \
terminations themselves cannot be addressed by any harness edit. Such a pattern \
exists only so you can see how often the environment intervened; the expected \
default is to skip it. Propose a candidate against it only when, in your own \
judgment, a harness-side handling edit is worthwhile -- for example, changing how \
sub-call errors are treated during recovery -- and hold that candidate to the \
same no-harm bar as any other.
"""

TASK_REASONING_GUIDANCE = """Task-derived reasoning (use only considerations supported by held-in evidence):
Which information must survive each step, and which task conditions must the final
computation enforce? Consider counts, dates, identity, ordering, provenance, units,
and asymmetric roles only when the task needs them. A set discards multiplicity
and cannot implement an exactly-one predicate; aggregate counts cannot restore
lost dates; joining by ID cannot verify semantic labels. Derive the procedure
from this task's conditions rather than applying a benchmark recipe.

Compare with actual execution, not only the incumbent instruction text. Identify
successful checks and discarded/replaced intermediate results before naming an
unresolved operation. Complete parsed IDs do not prove correct labels or complete
original parsing. Do not assert unverified labels were correct. Recovery already
performed is not a new final-answer fix; explain what remains wrong afterward.

Challenge each proposed change: If this edit were followed perfectly, could the
demonstrated failure still happen for the same reason? If yes, revise the causal
claim or withdraw it. A coverage reminder cannot resolve wrong-but-valid labels.
Describe the changed action/value at the unresolved operation in behavioral_change;
use a tiny synthetic counterexample, not a saved task answer. Running an identical
parser expression again changes no deterministic result. Sorting or deduplicating
invalid elements does not remove them. Identify the changed operation that would.
If a later computation replaces the cited intermediate state, diagnose that latest
state or explain why the earlier defect still affects it.
"""


PROPOSER_QUALITY = """\
Candidate quality rules:
- Promotion evaluates one combined candidate on held-out runs only, requiring \
strictly more exact passes and the configured cost band. For each \
candidate, predicted_effect must also state which currently-passing behaviors the \
edit deliberately leaves untouched and why the edit cannot plausibly harm them. An \
edit whose upside on the failing pattern is bought with plausible harm to passing \
runs will be rejected in validation; no-harm comes first.
- Prefer edits that guide behavior toward recovery over hard accept/reject rules. \
Any edit that can veto or discard an output must state the path by which a correct \
output still gets through; a rule with no escape path usually trades one failure \
class for another.
- Runs operate under fixed time and spend limits, and some already end by \
exhausting them. An edit that increases iterations, sub-calls, or output length \
spends from that same budget; treat added work as a regression risk to justify, \
and prefer edits that are work-neutral or work-reducing.
- When multiple candidates are proposed, prefer candidates that differ in \
mechanism family and in risk profile, rather than variants of the same kind of \
intervention.
"""

# EDIT_FORMATS and SKILLS_PEDAGOGY are ``%``-interpolated (never ``str.format``)
# precisely so the literal JSON/template braces they carry need no doubling --
# keep any addition to them ``%``-safe.
EDIT_FORMATS = (
    """\
Edit formats, exactly one of the following depending on the surface named for a \
pattern:
- S1-S5 (prompt text): {"kind": "text", "new_text": "<the full replacement surface \
text>"}
- S6 (runtime policy): {"kind": "policy", "runtime_policy": {"enabled": true, ...}} \
-- legal keys: %(s6_keys)s. Only "enabled": true activates any numeric/boolean field \
you set; an edit that leaves "enabled" false or absent changes nothing.
- S7 metadata / S9 answer_middleware (callables): {"kind": "code", "source": \
"def <name>(...):\\n    ...\\n"} -- exactly one undecorated top-level function \
definition, matching the surface's current signature.
- S8 repl helper (one named helper function): {"kind": "repl_helper", "dict": \
"repl_helpers" or "sub_repl_helpers", "name": "<function name>", "source": \
"def <name>(...):\\n    ...\\n"} -- the function name in source must equal "name". \
The name "%(skill_loader)s" is reserved for the S10 skill loader the runner installs.
"""
    + SKILLS_EDIT_FORMAT
)

# The S10 skill-writing pedagogy: how to fill a body -- distill the shown traces
# into a procedure, not a knowledge dump. Rendered only when at least one
# addressable pattern targets S10, so bundles with no S10 pattern do not pay its
# prompt cost. The merge semantics live once, in ``SKILLS_EDIT_FORMAT``.
SKILLS_PEDAGOGY = """\
When proposing an S10 edit, distill the shown failure pattern's traces into a \
reusable skill.

A skill is a procedural anchor, not a knowledge dump. Its job is to stabilize \
action: setup steps, tool sequences, checks, and pitfalls. Do not paste raw \
trajectory content -- compress it. Verbose process residue (exploration, dead \
ends, debugging noise) wastes the load budget and the sub-call prompt the body \
is forwarded into.

Write ONE skill in this format:

## Use When
- {concrete triggering conditions}
## Don't Use When
- {conditions where this guidance doesn't apply -- be honest about scope}

## Steps
1. {ordered, concrete, actionable -- REPL and tool sequences, not vague advice}

## Pitfalls
- {failure signal -> mitigation, taken from the FAILED runs in the pattern}

## Verify
- {how to confirm success at runtime -- actually run something, don't just \
inspect statically}

Rules:
- Generalize: placeholders instead of task-specific paths/data, but keep \
concrete command patterns. Over-specific skills break on the next task.
- Environment setup (imports, missing modules), REPL and sub-call sequences, \
output formats, and checks are the highest-value things to encode. Skills \
won't fix wrong algorithms -- don't try to encode the answer, encode the \
process. A skill is guidance text loaded by %(skill_loader)s, not a REPL \
helper (those are S8).
- Adapt-don't-copy: write steps as guidance to be checked against the current \
task, not a script to follow blindly.
- Make the description retrieval-friendly: if it's vague, it will be confused \
with similar skills in the index and never loaded correctly.
"""

RESPONSE_FORMAT = """\
Respond with one fenced JSON object using format proposal-selection/v2. Write
selections first: choose at most %(k)s interventions, one per surface and pattern,
ranked by evidence for the unresolved operation. Then write exactly one matching
candidate per selection. No extra model call is needed. To withdraw all, return
empty selections and candidates lists. Example shape:

```json
{
  "format": "proposal-selection/v2",
  "selections": [{"pattern_index": 0, "surface": "S3", "reason": "<capability, unresolved operation and intended caller/recovery point>", "evidence_refs": ["<copy this pattern\'s admitted operation_ref>"]}],
  "candidates": [{
    "pattern_index": 0,
    "surface": "S3",
    "revision": null,
    "incumbent_behavior": "<what the relevant execution actually did; distinguish instructions from execution>",
    "observed_failure": "<unresolved operation and verification limits in held-in evidence>",
    "behavioral_change": "<precise changed action and why it addresses the demonstrated cause>",
    "edit": "<one edit object in the surface's format above>",
    "predicted_effect": "<predicted behavior>",
    "regression_risks": ["<what it might break>"]
  }]
}
```
Selection reasons and each explanation field must contain 1-600 characters.
Each selection's evidence_refs lists 1-12 admitted operation_ref IDs from
its own selectable pattern. Copy identities from the evidence inventory; inventory-only
rows cannot authorize edits, even on an otherwise eligible surface. Each surface's
predecessors entry supplies the revision round and subject_id when required.
Newly supported S8/S5 routes must cite all route_support refs;
S8 must specify the deterministic input/output contract and intended caller.
A visible error followed by recovery is not itself an unresolved defect.
Unattributed patterns are inventory-only and cannot authorize an edit. Coverage
basis marked not_established or contradicted is an unestablished hypothesis;
legacy "coverage basis not assessed" is not proof of input loss. Missing answer
elements or an absent coverage check alone cannot establish skipped inputs.
For a revisited intervention, revision must be {"round": 6, "subject_id": "prior-id",
"explanation": "<changed operation, current evidence of new applicability, or revised joint hypothesis>"}.
Use null only for a new intervention. Unchanged members in a changed batch need
an explicit prior reference and revised joint hypothesis; they have no individual
performance outcome. Repeating an identical rejected evaluated harness under
the same incumbent is refused locally before validation.
"""


def render_prompt(
    patterns: list[dict[str, Any]],
    incumbent_serialization: dict[str, Any],
    passing_behaviors: Sequence[dict[str, Any]],
    prior_history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
    k: int,
    verifier_config: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    evidence_audit: dict[str, Any] | None = None,
) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
    """The one system prompt for a round, and the addressable patterns shown.

    The addressable list is returned alongside the prompt because the round
    loop needs it again (to compute which addressable patterns were never
    proposed, i.e. ``skipped_patterns``) without re-deriving it.
    """
    attempt_index = prior_attempts(prior_history)
    from shrlm.optimization.proposal_evidence import (
        EvidenceBudgetExceeded,
        bounded_excerpt,
        pack_evidence,
        withhold_choices,
    )

    inventory = [
        {
            "index": index,
            "signature": pattern["signature"],
            "predecessors": {
                surface: {key: prior.get(key) for key in ("round", "subject_id", "decision")}
                for surface in (f"S{i}" for i in range(1, 11))
                if (
                    prior := select_predecessor(
                        surface=surface,
                        mechanism=pattern["signature"]["agent_mechanism"],
                        attempts=attempt_index,
                    )
                )
                is not None
            },
            "support": pattern.get("support"),
            "instance_support": pattern.get("instance_support"),
            "instance_ids": pattern.get("instance_ids", [])
            if pattern.get("instance_support") is None
            else [],
            "actionability": pattern.get("actionability"),
            "below_min_support": pattern.get("below_support_floor"),
            "symptoms": bounded_excerpt(str(pattern.get("shared_symptoms", "")), 300),
        }
        for index, pattern in enumerate(patterns)
        if _pattern_mechanism(pattern) is not None
    ]
    context = dict(evidence or {})
    context["passing_ids"] = [str(run["instance_id"]) for run in passing_behaviors]
    try:
        evidence_text, audit = pack_evidence(inventory, context, k=k)
    except EvidenceBudgetExceeded as exc:
        raise ProposalRejection(str(exc)) from exc
    history_text, withheld = (
        compact_history(
            prior_history,
            choices=audit["choices"],
            mechanisms={p.get("signature", {}).get("agent_mechanism") for p in patterns},
            incumbent_hash=hash_of_serialization(incumbent_serialization),
        )
        if prior_history
        else (_render_history_block(prior_history), [])
    )
    evidence_text = withhold_choices(evidence_text, audit, withheld)
    if evidence_audit is not None:
        evidence_audit.update(audit)
        evidence_audit["history_chars"] = len(history_text)
    addressable = [
        (int(index), {**patterns[int(index)], **choice})
        for index, choice in audit["choices"].items()
    ]
    sections = [
        PROPOSER_INTRO + render_surface_block(),
        PROPOSER_TASK,
        PROPOSER_QUALITY,
        TASK_REASONING_GUIDANCE,
        _render_verifier_contract(verifier_config),
        CALLABLE_CONTRACT,
        render_current_surfaces(addressable, incumbent_serialization),
        evidence_text,
        "Prior edit history (budgeted prior candidates with their surfaces, "
        "predicted effect, and outcome; do not repeat an approach already rejected "
        "for the same reason). A candidate identical to the current surface is "
        "refused before validation and must not be re-proposed: a not_materialized "
        "entry below means the incumbent already contained that edit.\n"
        "A rejected but potentially_promising direction may merit a materially different "
        "refinement. Keep its rejection reasons and contrary metrics in view; do not replay "
        "the same edit or attribute a combined gain to one member. These descriptive "
        "diagnostics do not change promotion; v=1 is not a reliable causal estimate.\n"
        + history_text,
        EDIT_FORMATS
        % {
            "s6_keys": list(S6_KEYS),
            "skill_loader": SKILL_LOADER_NAME,
            "name_cap": SKILL_NAME_MAX_CHARS,
            "description_cap": SKILL_DESCRIPTION_MAX_CHARS,
            "body_cap": SKILL_BODY_MAX_CHARS,
            "entry_cap": SKILL_MAX_ENTRIES,
            "total_cap": SKILL_TOTAL_MAX_CHARS,
            "min_steps": SKILL_BODY_MIN_STEPS,
        },
    ]
    if (verifier_config or {}).get("environment") == "oolong_pairs":
        from shrlm.environments.oolong_pairs import ANSWER_CONTRACT

        sections.append(ANSWER_CONTRACT)
    if any("S10" in _pattern_surfaces(pattern) for _, pattern in addressable):
        sections.append(SKILLS_PEDAGOGY % {"skill_loader": SKILL_LOADER_NAME})
    eligible = {surface for _, pattern in addressable for surface in _pattern_surfaces(pattern)}
    sections.append(RESPONSE_FORMAT % {"k": min(k, len(eligible))})
    rendered = "\n\n".join(sections)
    if evidence_audit is not None:
        evidence_audit["system_prompt_chars"] = len(rendered)
    return rendered, addressable


def prompt_sha256(rendered_prompt: str) -> str:
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Response extraction and validation
# ---------------------------------------------------------------------------


def _repair_unescaped_quotes(candidate: str, first_error: json.JSONDecodeError) -> Any | None:
    """Parse JSON whose string values hold unescaped quotes or raw control
    characters (newlines/tabs), or None when no clean repair exists.

    Observed on stealth/ox-alpha (2026-08-24): proposals quoting harness
    identifiers verbatim (``answer["ready"] = True``) inside ``new_text``
    break strict JSON at the same character on every sample, so re-asking
    cannot converge. The decoder stops the string at the stray quote and then
    fails expecting a delimiter; escaping the quote that prematurely closed
    the string (the one just before the error position) and re-parsing fixes
    exactly one such quote per pass. Bounded, monotic passes: each escape
    must move the error position forward, anything else aborts. The repaired
    text preserves every literal character the model wrote -- only the
    escaping changes.
    """
    control_escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    error = first_error
    for _ in range(200):
        if error.msg.startswith("Invalid control character"):
            escape = control_escapes.get(candidate[error.pos])
            if escape is None:
                return None
            repaired = candidate[: error.pos] + escape + candidate[error.pos + 1 :]
        elif error.msg.startswith("Expecting"):
            close = candidate.rfind('"', 0, error.pos)
            if close <= 0 or candidate[close - 1] == "\\":
                return None
            repaired = candidate[:close] + '\\"' + candidate[close + 1 :]
        else:
            return None
        try:
            return json.loads(repaired, object_pairs_hook=ProposalObject)
        except json.JSONDecodeError as exc:
            if exc.pos <= error.pos:
                return None
            candidate, error = repaired, exc
    return None


class ProposalObject(dict[str, Any]):
    """Retain duplicate-key errors until they can be assigned to a member."""

    def __init__(self, pairs: list[tuple[str, Any]]):
        super().__init__(pairs)
        keys = [key for key, _ in pairs]
        self.duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})


def duplicate_json_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return list(getattr(value, "duplicate_keys", [])) + [
            key for child in value.values() for key in duplicate_json_keys(child)
        ]
    if isinstance(value, list):
        return [key for child in value for key in duplicate_json_keys(child)]
    return []


def extract_proposal_response(text: str) -> dict[str, Any]:
    """Parse the live selection contract; historical arrays are not reinterpreted."""
    match = JSON_BLOCK_PATTERN.search(text)
    candidate = match.group(1) if match else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start or text.lstrip().startswith("["):
            raise ProposalRejection("no JSON selection object found in the response")
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate, object_pairs_hook=ProposalObject)
    except json.JSONDecodeError as exc:
        parsed = _repair_unescaped_quotes(candidate, exc)
        if parsed is None:
            raise ProposalRejection(f"response was not valid JSON: {exc}") from None
    if not isinstance(parsed, dict) or parsed.get("format") != RESPONSE_FORMAT_VERSION:
        raise ProposalRejection(f"expected a JSON object with format {RESPONSE_FORMAT_VERSION}")
    if getattr(parsed, "duplicate_keys", []):
        raise ProposalRejection("duplicate JSON keys in response envelope")
    if not all(isinstance(parsed.get(key), list) for key in ("selections", "candidates")):
        raise ProposalRejection("selections and candidates must be lists")
    return parsed


def _validate_single_def(source: Any, label: str) -> str:
    """Enforce the loader's own callable-source rule and return the def name.

    Mirrors ``candidates._single_function_name``: exactly one undecorated
    top-level function definition. Re-implemented locally (not imported, it
    is a private helper) so a broken candidate is caught and re-asked before
    ever touching disk, with the same violation shape the real gate gives.
    """
    if not isinstance(source, str) or not source.strip():
        raise ProposalRejection(f"{label}: source must be a non-empty string")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ProposalRejection(f"{label}: source does not parse: {error}") from None
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].decorator_list
    ):
        raise ProposalRejection(
            f"{label}: source must be exactly one undecorated top-level function definition"
        )
    return tree.body[0].name


def _validate_edit_shape(index: int, surface: str, edit: dict[str, Any]) -> dict[str, Any]:
    kind = edit.get("kind")
    label = f"pattern_index {index} ({surface})"
    expected_kind = SURFACE_EDIT_KIND[surface]
    if kind != expected_kind:
        raise ProposalRejection(f"{label}: needs edit.kind={expected_kind!r}, got {kind!r}")

    if kind == EDIT_KIND_TEXT:
        new_text = edit.get("new_text")
        if not isinstance(new_text, str) or not new_text.strip():
            raise ProposalRejection(f"{label}: edit.new_text must be a non-empty string")
        return {"kind": kind, "new_text": new_text}

    if kind == EDIT_KIND_POLICY:
        policy = edit.get("runtime_policy")
        if not isinstance(policy, dict):
            raise ProposalRejection(f"{label}: edit.runtime_policy must be an object")
        unknown = set(policy) - set(S6_KEYS)
        if unknown:
            raise ProposalRejection(
                f"{label}: edit.runtime_policy has unknown keys {sorted(unknown)}; "
                f"legal keys: {list(S6_KEYS)}"
            )
        merged = build_runtime_policy()
        merged.update(policy)
        try:
            json.dumps(merged)
        except (TypeError, ValueError) as error:
            raise ProposalRejection(
                f"{label}: edit.runtime_policy is not JSON-serializable: {error}"
            ) from None
        return {"kind": kind, "runtime_policy": merged}

    if kind == EDIT_KIND_CODE:
        def_name = _validate_single_def(edit.get("source"), label)
        return {"kind": kind, "source": edit["source"], "def_name": def_name}

    if kind == EDIT_KIND_REPL_HELPER:
        dict_name = edit.get("dict")
        if dict_name not in REPL_HELPER_DICTS:
            raise ProposalRejection(
                f"{label}: edit.dict must be one of {REPL_HELPER_DICTS}, got {dict_name!r}"
            )
        def_name = _validate_single_def(edit.get("source"), label)
        name = edit.get("name")
        if not isinstance(name, str) or not name:
            raise ProposalRejection(f"{label}: edit.name must be a non-empty string")
        if name != def_name:
            raise ProposalRejection(
                f"{label}: edit.name {name!r} must match the function name defined in "
                f"source ({def_name!r})"
            )
        if name in RESERVED_TOOL_NAMES:
            raise ProposalRejection(
                f"{label}: edit.name {name!r} collides with a reserved REPL name"
            )
        if name == SKILL_LOADER_NAME:
            raise ProposalRejection(
                f"{label}: edit.name {name!r} is the harness-reserved S10 skill loader, "
                "installed by the runner; it cannot be bound as an S8 helper"
            )
        return {"kind": kind, "dict": dict_name, "name": name, "source": edit["source"]}

    if kind == EDIT_KIND_SKILLS:
        try:
            record = _validate_skill_edit(label, edit)
        except SkillEditRejection as exc:
            raise ProposalRejection(str(exc)) from None
        return {"kind": kind, **record}

    raise ProposalRejection(
        f"{label}: unknown edit.kind {kind!r}"
    )  # pragma: no cover - unreachable


def validate_candidate_spec(item: Any, patterns: list[dict[str, Any]]) -> CandidateSpec:
    """Validate one array item against the closed vocabulary and the pattern
    list, exactly like ``attribution.validate`` validates one response."""
    if not isinstance(item, dict):
        raise ProposalRejection(f"each candidate must be a JSON object, got {type(item).__name__}")

    index = item.get("pattern_index")
    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(patterns)):
        raise ProposalRejection(
            f"pattern_index must be an integer in [0, {len(patterns)}), got {index!r}"
        )
    pattern = patterns[index]
    eligible = _pattern_surfaces(pattern)
    if not eligible:
        raise ProposalRejection(
            f"pattern_index {index} has mechanism "
            f"{pattern.get('signature', {}).get('agent_mechanism')!r}, which maps to no "
            "editable surface; do not propose a candidate for it"
        )
    surface = item.get("surface", eligible[0])
    if surface not in eligible:
        raise ProposalRejection(
            f"pattern_index {index}: surface {surface!r} is not eligible for mechanism "
            f"{pattern.get('signature', {}).get('agent_mechanism')!r}; choose one of "
            f"{eligible}"
        )

    edit = item.get("edit")
    if not isinstance(edit, dict):
        raise ProposalRejection(f"pattern_index {index}: edit must be an object")
    validated_edit = _validate_edit_shape(index, surface, edit)

    effect = item.get("predicted_effect")
    if not isinstance(effect, str) or not effect.strip():
        raise ProposalRejection(
            f"pattern_index {index}: predicted_effect must be a non-empty string"
        )

    risks = item.get("regression_risks", [])
    if not isinstance(risks, list) or not all(isinstance(risk, str) for risk in risks):
        raise ProposalRejection(
            f"pattern_index {index}: regression_risks must be a list of strings"
        )
    violation = behavioral_difference_violation(item, required=True)
    if violation:
        raise ProposalRejection(f"pattern_index {index}: {violation}")
    revision = item.get("revision")
    if revision is not None and (
        not isinstance(revision, dict)
        or set(revision) != {"round", "subject_id", "explanation"}
        or type(revision["round"]) is not int
        or revision["round"] < 0
        or not isinstance(revision["subject_id"], str)
        or not 1 <= len(revision["subject_id"]) <= 128
        or not isinstance(revision["explanation"], str)
        or not 1 <= len(revision["explanation"].strip()) <= 600
    ):
        raise ProposalRejection(
            "revision must be null or a prior round/subject_id and 1-600 character explanation"
        )

    return CandidateSpec(
        pattern_index=index,
        pattern=pattern,
        surface=surface,
        edit=validated_edit,
        predicted_effect=effect,
        regression_risks=list(risks),
        **{name: item[name] for name in BEHAVIOR_FIELDS},
        text_contract=TEXT_CONTRACT,
        revision=revision,
    )


def _skill_merge_coaching(skills: list[SkillEntry]) -> str:
    """What the proposer needs to fix a doomed S10 merge: the caps, the current
    totals, every existing name (with its char weight), and the two legal ways
    to free budget."""
    used = sum(
        len(getattr(entry, field_name)) for entry in skills for field_name in SKILL_RECORD_FIELDS
    )
    entries = (
        ", ".join(
            f"{entry.name} "
            f"({sum(len(getattr(entry, field_name)) for field_name in SKILL_RECORD_FIELDS):,})"
            for entry in skills
        )
        or "(empty)"
    )
    return (
        f"Current library: {len(skills)}/{SKILL_MAX_ENTRIES} entries, "
        f"{used:,}/{SKILL_TOTAL_MAX_CHARS:,} chars. Existing entries (chars): {entries}. "
        'Reuse an existing name to replace that skill, or remove one (its "name" with '
        '"description": "" and "body": "") to free budget.'
    )


def _dry_run_skill_merge(spec: CandidateSpec, incumbent: Harness) -> None:
    """Dry-run an S10 edit's merge against the incumbent library at validation
    time, so an entry-cap or total-length overflow (or a removal naming no
    existing entry) is a re-askable ``ProposalRejection`` with coaching --
    never only a silent ``MaterializationFailure`` the proposer is not told
    about. Materialization keeps the same ``_merge_skill`` check as a
    backstop."""
    record = {field_name: spec.edit[field_name] for field_name in SKILL_RECORD_FIELDS}
    try:
        _merge_skill(incumbent.skills, record)
    except SkillEditRejection as exc:
        raise ProposalRejection(
            f"pattern_index {spec.pattern_index} (S10): {exc}. "
            + _skill_merge_coaching(incumbent.skills)
        ) from None


# ---------------------------------------------------------------------------
# Materialization: CandidateSpec -> live Harness
# ---------------------------------------------------------------------------


def _import_candidate_function(source: str, def_name: str, workdir: Path, surface: str) -> Any:
    """Write generated source into a real module and import it.

    Reuses ``candidates.import_surface_module`` (content-addressed module
    names, so re-importing identical content is a cache hit) and the loader's
    own import preamble, so the source lands in exactly the vocabulary the
    stage-3 gate will later accept. ``workdir`` is scratch space, not the
    proposals directory -- nothing needs this file to persist, since the
    source text is captured into ``proposal.json`` and stage 3 regenerates
    its own ``surfaces.py`` from that JSON independently.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    text = source if source.endswith("\n") else source + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    module_path = workdir / f"_candidate_{surface.lower()}_{digest}.py"
    module_path.write_text(CANDIDATE_MODULE_PREAMBLE + "\n" + text)
    module = import_surface_module(module_path)
    return getattr(module, def_name)


def materialize_candidate_harness(
    incumbent: Harness, spec: CandidateSpec, workdir: Path
) -> Harness:
    """Build the live edited Harness for one validated spec.

    ``dataclasses.replace`` on the incumbent, per the interface doc's rule:
    never assemble the envelope JSON by hand, build a real Harness and
    serialize it.
    """
    edit = spec.edit
    kind = edit["kind"]

    if kind == EDIT_KIND_TEXT:
        field_name = TEXT_SURFACE_FIELDS[spec.surface]
        marker = text_slot_marker(serialize_harness(incumbent))
        encoded = edit["new_text"].replace("{", "{{").replace("}", "}}")
        encoded = encoded.replace(marker, "{custom_tools_section}")
        return replace(incumbent, **{field_name: encoded})

    if kind == EDIT_KIND_POLICY:
        return replace(incumbent, runtime_policy=dict(edit["runtime_policy"]))

    if kind == EDIT_KIND_CODE:
        fn = _import_candidate_function(edit["source"], edit["def_name"], workdir, spec.surface)
        if spec.surface == "S7":
            # Pinned to the incumbent's own bound: this proposer only ever
            # edits S7's logic, never its truncation bound, which is what
            # keeps an S7 edit a legal one-surface diff (see interface doc).
            fn.declared_bound = declared_metadata_bound(incumbent.metadata)
        field_name = CODE_SURFACE_FIELDS[spec.surface]
        return replace(incumbent, **{field_name: fn})

    if kind == EDIT_KIND_REPL_HELPER:
        fn = _import_candidate_function(edit["source"], edit["name"], workdir, spec.surface)
        updated = dict(getattr(incumbent, edit["dict"]))
        updated[edit["name"]] = fn
        return replace(incumbent, **{edit["dict"]: updated})

    if kind == EDIT_KIND_SKILLS:
        try:
            merged = _merge_skill(
                incumbent.skills,
                {field_name: edit[field_name] for field_name in SKILL_RECORD_FIELDS},
            )
        except SkillEditRejection as exc:
            raise MaterializationFailure(spec.pattern_index, str(exc)) from None
        return replace(incumbent, skills=merged)

    raise AssertionError(f"unreachable edit kind {kind!r}")  # validated upstream


def build_candidate(
    incumbent: Harness,
    incumbent_serialization: dict[str, Any],
    spec: CandidateSpec,
    workdir: Path,
) -> tuple[Harness, dict[str, Any]]:
    """Materialize and verify one spec, or raise ``MaterializationFailure``.

    Verification reuses ``candidates.changed_surfaces`` -- the real gate's own
    diffing function -- to confirm the result is exactly a one-surface diff on
    the declared surface, so a proposal that would fail the loader's
    ``surface_diff`` gate is caught here instead of burning a stage-3 slot.
    """
    try:
        harness = materialize_candidate_harness(incumbent, spec, workdir)
        serialization = serialize_harness(harness)
    except HarnessSerializationError as error:
        raise MaterializationFailure(spec.pattern_index, str(error)) from error
    changed = changed_surfaces(incumbent_serialization, serialization)
    if changed != [spec.surface]:
        raise MaterializationFailure(
            spec.pattern_index,
            f"declared surface {spec.surface} but the materialized harness changes "
            f"{changed or 'no surface'}",
        )
    return harness, serialization


# ---------------------------------------------------------------------------
# Writing proposal.json
# ---------------------------------------------------------------------------


def _candidate_id(round_index: int, position: int, surface: str) -> str:
    return f"r{round_index:02d}-c{position:02d}-{surface.lower()}"


def proposal_history_fields(
    spec: CandidateSpec,
    incumbent_serialization: dict[str, Any],
    serialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve known intent on refused attempts without inventing an effective edit."""
    return {
        **{name: getattr(spec, name) for name in BEHAVIOR_FIELDS},
        "mechanism": spec.pattern["signature"]["agent_mechanism"],
        "incumbent_hash": hash_of_serialization(incumbent_serialization),
        "revision": spec.revision,
        **(
            {"effective_edit_fingerprint": surface_fingerprint(serialization, spec.surface)}
            if serialization is not None
            else {}
        ),
    }


def write_proposal(
    proposals_dir: Path,
    candidate_id: str,
    incumbent_serialization: dict[str, Any],
    spec: CandidateSpec,
    serialization: dict[str, Any],
    model_name: str,
    prompt_sha: str,
) -> Path:
    """Write one ``shrlm-proposal/v1`` proposal.json (docs/harness-proposal-interface.md).

    Non-clobbering: an existing file is left alone if byte-identical
    (canonical JSON), and raises if it differs -- the same guard
    ``bundle.write_bundle`` uses for audited evidence.
    """
    named_serialization = {**serialization, "name": candidate_id}
    envelope = {
        "format": HARNESS_FORMAT,
        "name": candidate_id,
        "hash": hash_of_serialization(named_serialization),
        "harness": named_serialization,
    }
    payload = {
        "format": PROPOSAL_FORMAT,
        "candidate_id": candidate_id,
        "base_harness_hash": hash_of_serialization(incumbent_serialization),
        "target_signature": dict(spec.pattern["signature"]),
        "surface": spec.surface,
        "harness": envelope,
        "predicted_effect": spec.predicted_effect,
        **{name: getattr(spec, name) for name in BEHAVIOR_FIELDS if getattr(spec, name)},
        "regression_risks": list(spec.regression_risks),
        "revision": spec.revision,
        "revision_unchanged": spec.revision_unchanged,
        "effective_edit_fingerprint": surface_fingerprint(serialization, spec.surface),
        "changed_skill": spec.edit.get("name") if spec.surface == "S10" else None,
        "activation_applicable": spec.surface != "S6"
        or (
            bool(serialization["surfaces"]["S6_runtime_policy"].get("retry_on_syntax_error"))
            and any(
                incumbent_serialization["surfaces"]["S6_runtime_policy"].get(key)
                != serialization["surfaces"]["S6_runtime_policy"].get(key)
                for key in ("enabled", "retry_on_syntax_error", "max_retries")
            )
        ),
        "provenance": {
            "model": model_name,
            "prompt_sha256": prompt_sha,
            **(
                {"text_contract": spec.text_contract, "response_format": RESPONSE_FORMAT_VERSION}
                if spec.text_contract
                else {}
            ),
        },
    }
    directory = Path(proposals_dir) / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROPOSAL_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if canonical_json(existing) != canonical_json(payload):
            raise ValueError(
                f"{path} already exists with different content; refusing to clobber it"
            )
        return path
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)
    return path


# ---------------------------------------------------------------------------
# The round entry point
# ---------------------------------------------------------------------------


def load_passing_behaviors(round_path: Path | str) -> list[dict[str, Any]]:
    """Runs with ``passed: true`` from one mining round's ``runs.jsonl``.

    Returns ``[]`` when the file is missing or every run failed (the
    committed example round is 0/8) -- the prompt renderer turns that into an
    explicit "nothing to preserve yet" line rather than an empty section.
    Results follow persisted instance order, then attempt.
    """
    round_path = Path(round_path)
    manifest_path = round_path / MANIFEST_FILE
    if not manifest_path.exists():
        return []
    instances = [
        json.loads(line)
        for line in (round_path / INSTANCES_FILE).read_text().splitlines()
        if line.strip()
    ]
    runs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    return [run for run in canonical_manifest_entries(runs, instances) if run.get("passed")]


def config_material(config: ProposerConfig, lm: BaseLM) -> dict[str, Any]:
    return {
        "model": lm.model_name,
        "sampling_args": lm.sampling_args,
        "k": config.k,
        "max_attempts": config.max_attempts,
        "prompt_version": config.prompt_version,
        "validator_version": config.validator_version,
        "text_contract": TEXT_CONTRACT,
        "response_format": RESPONSE_FORMAT_VERSION,
    }


def config_sha256(config: ProposerConfig, lm: BaseLM) -> str:
    payload = json.dumps(config_material(config, lm), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_key(prompt_sha: str, bundle_id: str, config_sha: str, attempt: int) -> str:
    material = f"{prompt_sha}|{bundle_id}|{config_sha}|{attempt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _completion_with_retry(
    lm: BaseLM,
    messages: list[dict[str, str]],
    config: ProposerConfig,
    attempts: list[ProposalAttempt],
) -> str:
    """Call the LM, retrying transient failures with exponential backoff.

    Identical retry/backoff/exception-classification policy to
    ``attribution._completion_with_retry``: a ``TokenLimitExceededError`` is
    deterministic for the prompt and propagates at once as
    ProposalBudgetExhausted rather than being re-sent.
    """
    last_error: Exception | None = None
    for retry in range(config.transport_retries):
        try:
            return lm.completion(messages)
        except NON_TRANSPORT_ERRORS:
            raise
        except TokenLimitExceededError as exc:
            raise ProposalBudgetExhausted(
                f"output budget exhausted on the proposer response: {exc}",
                attempts=list(attempts),
            ) from exc
        except Exception as exc:
            last_error = exc
            if retry + 1 < config.transport_retries:
                time.sleep(config.transport_backoff_seconds * (2**retry))
    raise ProposalTransportError(
        f"LM call failed after {config.transport_retries} tries: "
        f"{type(last_error).__name__}: {last_error}",
        attempts=list(attempts),
    ) from last_error


def publish_proposal_result(
    state: dict[str, Any],
    proposals_dir: Path,
    incumbent_serialization: dict[str, Any],
    model_name: str,
    *,
    replay: bool = False,
) -> ProposalRoundResult:
    """Publish only checkpointed survivors, including after an interrupted write."""
    expected = {item["candidate_id"] for item in state["survivors"]}
    unexpected = {path.name for path in proposals_dir.iterdir() if path.is_dir()} - expected
    if unexpected:
        raise ValueError(
            f"proposal directories outside the frozen survivor set: {sorted(unexpected)}"
        )
    written = []
    for item in state["survivors"]:
        spec = CandidateSpec(**item["spec"])
        path = write_proposal(
            proposals_dir,
            item["candidate_id"],
            incumbent_serialization,
            spec,
            item["serialization"],
            model_name,
            state["prompt_sha256"],
        )
        written.append(
            WrittenProposal(item["candidate_id"], path, spec.surface, spec.pattern_index)
        )
    return ProposalRoundResult(
        written=written,
        skipped_patterns=state["skipped_patterns"],
        materialization_failures=[
            MaterializationFailureRecord(**r) for r in state["materialization_failures"]
        ],
        attempts=[
            ProposalAttempt(**(a | {"cached": True} if replay else a)) for a in state["attempts"]
        ],
        prompt_sha256=state["prompt_sha256"],
        preflight_failures=state["preflight_failures"],
        preflight_profile=state["preflight_profile"],
        evidence_audit=state.get("evidence_audit", {}),
    )


def validate_batch_members(
    items: list[Any],
    patterns: list[dict[str, Any]],
    incumbent: Harness,
    failed_slots: dict[int, int] | None,
    failed_sources: list[Any],
    selections: list[Any],
) -> tuple[list[tuple[int, CandidateSpec]], list[dict[str, Any]]]:
    """Validate members independently; ownership starts only after all local gates."""
    identities = []
    for item in items:
        index = item.get("pattern_index") if isinstance(item, dict) else None
        if type(index) is not int or not 0 <= index < len(patterns):
            identities.append((None, None))
            continue
        eligible = _pattern_surfaces(patterns[index])
        surface = item.get("surface", eligible[0] if eligible else None)
        identities.append((index, surface if isinstance(surface, str) else None))
    slots = []
    failures = []
    for position, (item, (index, surface)) in enumerate(
        zip(items, identities, strict=True), start=1
    ):
        if failed_slots is not None:
            position = failed_slots.get(index, position)
        try:
            duplicates = duplicate_json_keys(item)
            if duplicates:
                raise ProposalRejection(f"duplicate JSON keys in candidate: {duplicates}")
            matching = [
                selection
                for selection in selections
                if isinstance(selection, dict)
                and type(selection.get("pattern_index")) is int
                and selection["pattern_index"] == index
                and selection.get("surface") == surface
            ]
            if len(matching) != 1:
                raise ProposalRejection("candidate must match exactly one selection")
            selection = matching[0]
            if duplicate_json_keys(selection):
                raise ProposalRejection("duplicate JSON keys in selection")
            reason = selection.get("reason")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
                raise ProposalRejection("selection reason must contain 1-600 characters")
            refs = selection.get("evidence_refs", [])
            if (
                not isinstance(refs, list)
                or not refs
                or len(refs) > 12
                or not all(isinstance(ref, str) for ref in refs)
            ):
                raise ProposalRejection(
                    "selection evidence_refs must be a nonempty list of at most 12 operation refs"
                )
            if index is None or not patterns[index].get("selectable"):
                raise ProposalRejection(
                    "pattern is not selectable: no admitted complete evidence and history"
                )
            if surface not in _pattern_surfaces(patterns[index]):
                raise ProposalRejection("surface is not one of this pattern's selectable choices")
            required = (
                patterns[index].get("route_support", {}).get(surface, [])
                if index is not None
                else []
            )
            if required and not set(required).issubset(refs):
                raise ProposalRejection("supported route must cite its admitted evidence_refs")
            admitted_refs = patterns[index].get("admitted_refs", []) if index is not None else []
            if refs and not set(refs).issubset(admitted_refs):
                raise ProposalRejection(
                    "selection evidence_refs must belong to this pattern's admitted operations"
                )
            if not isinstance(item, dict) or "surface" not in item:
                raise ProposalRejection("candidate must explicitly name its selected surface")
            if item.get("edit") is None:
                raise ProposalRejection("selection has no matching candidate replacement")
            if "revision" not in item or "evidence_refs" not in selection:
                raise ProposalRejection(
                    "proposal-selection/v2 requires candidate revision and selection evidence_refs"
                )
            if failed_slots is not None and index not in failed_slots:
                raise ProposalRejection("repair must target only an original failed pattern")
            spec = validate_candidate_spec(item, patterns)
            for original in failed_sources:
                old_surface = original.get("surface")
                if (
                    original["pattern_index"] == index
                    and old_surface != surface
                    and original.get("behavioral_change") == spec.behavioral_change
                ):
                    raise ProposalRejection("retargeting requires a revised behavioral_change")
            if spec.edit["kind"] == EDIT_KIND_SKILLS:
                _dry_run_skill_merge(spec, incumbent)
            slots.append((position, spec))
        except ProposalRejection as exc:
            failures.append(
                {
                    "pattern_index": index,
                    "surface": surface,
                    "position": position,
                    "gate": "proposal",
                    "reason": str(exc),
                    "predicted_effect": item.get("predicted_effect", "unavailable")
                    if isinstance(item, dict)
                    else "unavailable",
                    "behavior": {
                        **{
                            name: item[name]
                            for name in BEHAVIOR_FIELDS
                            if isinstance(item, dict) and isinstance(item.get(name), str)
                        },
                        **(
                            {"mechanism": patterns[index]["signature"]["agent_mechanism"]}
                            if index is not None and 0 <= index < len(patterns)
                            else {}
                        ),
                    },
                }
            )
    return slots, failures


def persist_proposal_failure(
    workdir: Path, error: Exception, attempts: list[ProposalAttempt]
) -> None:
    """Retain the terminal attempt audit without marking the stage successful."""
    pending = workdir / "proposal_failure.tmp"
    pending.write_text(
        canonical_json({"error": str(error), "attempts": [a.to_dict() for a in attempts]}) + "\n"
    )
    pending.replace(workdir / "proposal_failure.json")


def propose_round(
    bundle: dict[str, Any],
    incumbent: Harness,
    lm: BaseLM,
    proposals_dir: Path | str,
    *,
    round_index: int = 0,
    passing_behaviors: Sequence[dict[str, Any]] = (),
    prior_history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]] = (),
    config: ProposerConfig | None = None,
    cache: ProposalCache | None = None,
    workdir: Path | str | None = None,
    preflight_profile: str | None = None,
    caps: dict[str, int | float] | None = None,
    loader_timeout_seconds: float = DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
    evidence: dict[str, Any] | None = None,
) -> ProposalRoundResult:
    """Propose and write up to ``config.k`` candidates for one mining round.

    Args:
        bundle: A parsed ``bundle.json`` (``EvidenceBundle.to_dict()``'s
            shape) -- the artifact stage 1 actually writes to disk.
        incumbent: The harness the round's candidates must target
            (``base_harness_hash``) and diff from on exactly one surface.
        lm: The fixed proposer client, e.g.
            ``get_client("openrouter", {"model_name": "qwen/qwen3-30b-a3b-instruct-2507"})``.
        proposals_dir: Where ``<candidate_id>/proposal.json`` directories are written.
        round_index: Folded into each ``candidate_id``.
        passing_behaviors: Runs to preserve, from ``load_passing_behaviors``.
        prior_history: Prior rounds' ``(records, decision)`` pairs, one per
            round, from ``shrlm.optimization.validation.load_promotion_ledger``.
        config: Proposer configuration; defaults applied when omitted.
        cache: Response cache; a fresh in-memory one when omitted.
        workdir: Scratch directory for generated-source modules; a fresh
            temporary directory when omitted.

    Returns:
        A ``ProposalRoundResult`` covering every written proposal, every
        addressable pattern the model chose not to propose for, every
        candidate that validated but failed to materialize, and the full
        re-ask attempt trail.

    Raises:
        ProposalRejection: No attempt validated within ``config.max_attempts``.
        ProposalTransportError: The LM stayed unreachable after bounded retries.
        ProposalBudgetExhausted: The proposer spent its output budget without
            producing content; raised once, never re-asked.
    """
    config = config or ProposerConfig()
    cache = cache or ProposalCache()
    proposals_dir = Path(proposals_dir)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    workdir = (
        Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="shrlm-proposal-"))
    )

    recorder = observation_recorder(
        workdir / "proposal_result.json",
        {"stage": "proposal", "round": round_index},
        namespace="llm_calls/proposal",
    )
    patterns = [dict(pattern) for pattern in bundle.get("patterns", [])]
    attempt_index = prior_attempts(prior_history)
    incumbent_serialization = serialize_harness(incumbent)
    evidence_audit: dict[str, Any] = {}
    rendered_prompt, addressable = render_prompt(
        patterns,
        incumbent_serialization,
        passing_behaviors,
        prior_history,
        config.k,
        verifier_config=(bundle.get("config") or {}).get("verifier_config"),
        evidence=evidence,
        evidence_audit=evidence_audit,
    )
    for index, pattern in enumerate(patterns):
        pattern.update(evidence_audit["choices"].get(str(index), {"selectable": False}))
    system_sha = prompt_sha256(rendered_prompt)
    cfg_sha = config_sha256(config, lm)
    bundle_id = str(bundle.get("bundle_id", ""))

    profile = preflight_profile or select_preflight_profile(
        (bundle.get("config") or {}).get("verifier_config")
    )
    validate_preflight_profile(profile)
    workdir.mkdir(parents=True, exist_ok=True)
    from shrlm.optimization.proposal_evidence import (
        DIAGNOSTIC_HISTORY_VERSION,
        EVIDENCE_SELECTOR_VERSION,
    )

    contract = {
        "text_contract": TEXT_CONTRACT,
        "text_slot_marker": text_slot_marker(incumbent_serialization),
        "response_format": RESPONSE_FORMAT_VERSION,
        "evidence_selector_version": EVIDENCE_SELECTOR_VERSION,
        "diagnostic_history_version": DIAGNOSTIC_HISTORY_VERSION,
        "capability_version": CAPABILITY_VERSION,
        "history_schema": HISTORY_SCHEMA,
        "history_budget_chars": HISTORY_BUDGET_CHARS,
        "behavior_contract": BEHAVIOR_SCHEMA,
        "prior_attempts_sha256": prompt_sha256(canonical_json(attempt_index)),
        "prompt_sha256": system_sha,
        "config_sha256": cfg_sha,
        "base_hash": hash_of_serialization(incumbent_serialization),
        "bundle_id": bundle_id,
        "profile": profile,
        "caps": caps,
        "loader_timeout_seconds": loader_timeout_seconds,
        "round": round_index,
    }
    contract_path = workdir / "proposal_contract.json"
    finalized = list(proposals_dir.glob("*/proposal.json"))
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "proposal prompt/source/profile contract changed; refusing paid replay"
            )
    elif finalized:
        raise ValueError("finalized unsealed proposals have no matching preflight contract")
    else:
        pending_contract = contract_path.with_suffix(".tmp")
        pending_contract.write_text(json.dumps(contract, sort_keys=True) + "\n")
        pending_contract.replace(contract_path)
    for path in finalized:
        payload = json.loads(path.read_text())
        if (
            payload["provenance"]["prompt_sha256"] != system_sha
            or payload["base_harness_hash"] != contract["base_hash"]
            or hash_of_serialization(payload["harness"]["harness"]) != payload["harness"]["hash"]
        ):
            raise ValueError(f"finalized proposal source mismatch: {path}")

    result_path = workdir / "proposal_result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["sha256"] != prompt_sha256(canonical_json(saved["result"])):
            raise ValueError("proposal result checkpoint hash mismatch")
        verify_observations(saved, result_path.parent)
        if saved["contract"] != contract:
            raise ValueError("proposal result checkpoint contract mismatch")
        return publish_proposal_result(
            saved["result"], proposals_dir, incumbent_serialization, lm.model_name, replay=True
        )
    if finalized:
        raise ValueError("finalized proposals have no frozen result checkpoint")

    attempts: list[ProposalAttempt] = []
    rejection = ""
    materialized: list[tuple[int, CandidateSpec, dict[str, Any]]] = []
    materialization_failures: list[MaterializationFailureRecord] = []
    preflight_failures: list[dict[str, Any]] = []
    failed_slots: dict[int, int] = {}
    failed_sources: list[Any] = []
    repairing = False
    for attempt in range(config.max_attempts if addressable else 0):
        user = (
            "Propose your candidates now."
            if not rejection
            else f"Your previous response was rejected: {rejection}\nRespond again with a corrected selection object."
        )
        if repairing:
            retained = [
                {
                    "pattern_index": spec.pattern_index,
                    "surface": spec.surface,
                    "hash": hash_of_serialization(serialization),
                }
                for _, spec, serialization in materialized
            ]
            user = (
                "One repair response only. Return replacements only for the failed original "
                "patterns; omit a failed member to withdraw it. You may choose another eligible, "
                "unoccupied surface for that same pattern. Explain the revised behavioral change "
                "when retargeting. Keep retained edits unchanged; no new patterns. "
                "Return selections followed by candidates. Select in priority order and remove "
                "each selected surface from further consideration. Occupied surfaces cannot "
                "be selected again. Do not move an edit to an unsuitable surface just to fill "
                "the batch.\nOccupied surfaces: "
                + canonical_json(sorted(r["surface"] for r in retained))
                + "\nEligible unoccupied surfaces: "
                + canonical_json(
                    {
                        str(index): [
                            s
                            for s in _pattern_surfaces(patterns[index])
                            if s not in {r["surface"] for r in retained}
                        ]
                        for index in failed_slots
                    }
                )
                + "\nCopyable failed-pattern choices (use only free surfaces): "
                + canonical_json(
                    {str(index): evidence_audit["choices"][str(index)] for index in failed_slots}
                )
                + "\nRetained: "
                + canonical_json(retained)
                + "\nFailed sources: "
                + canonical_json(failed_sources)
                + "\nFailures: "
                + rejection
            )
        request_sha = prompt_sha256(
            canonical_json({"system": system_sha, "user": user, "profile": profile})
        )
        key = _cache_key(request_sha, bundle_id, cfg_sha, attempt)
        response = cache.get(key)
        cached = response is not None
        evidence_audit.setdefault("attempt_prompt_chars", []).append(
            len(rendered_prompt) + len(user)
        )
        observed = None
        try:
            with observe_call(
                "proposal_repair" if attempt else "proposal",
                lm.model_name,
                recorder=recorder,
                attempt=attempt + 1,
            ) as observed:
                budget_failure = cache.get(key + ":budget_exhausted")
                if budget_failure is not None:
                    cached = True
                    observed.replay(cache.get_observations(key + ":budget_exhausted"))
                    raise ProposalBudgetExhausted(budget_failure, attempts)
                if response is None:
                    response = _completion_with_retry(
                        lm,
                        [
                            {"role": "system", "content": rendered_prompt},
                            {"role": "user", "content": user},
                        ],
                        config,
                        attempts,
                    )
                else:
                    observed.replay(cache.get_observations(key))
        except (ProposalBudgetExhausted, ProposalTransportError) as exc:
            if isinstance(exc, ProposalBudgetExhausted) and not cached:
                cache.put(
                    key + ":budget_exhausted",
                    str(exc),
                    observations=observed.responses if observed else None,
                )
            attempts.append(
                ProposalAttempt(
                    attempt + 1,
                    cached,
                    "",
                    False,
                    str(exc),
                    llm_observations=observation_refs(observed),
                )
            )
            exc.attempts = list(attempts)
            if not repairing or isinstance(exc, ProposalTransportError):
                persist_proposal_failure(workdir, exc, attempts)
                raise
            break
        if not cached:
            cache.put(key, response, observations=observed.responses if observed else None)
        refs = observation_refs(observed)

        try:
            parsed = extract_proposal_response(response)
            selections = parsed["selections"]
            raw_items = parsed["candidates"]
            if max(len(raw_items), len(selections)) > config.k:
                raise ProposalRejection(
                    f"response proposed {max(len(raw_items), len(selections))} entries, more than the allowed {config.k}"
                )
        except ProposalRejection as exc:
            rejection = str(exc)
            attempts.append(
                ProposalAttempt(
                    attempt + 1, cached, response, False, rejection, llm_observations=refs
                )
            )
            if repairing:
                break
            continue

        # An orphan selection is an attributable member failure, not a reason
        # to discard independent replacements. Retain it in the repair sources.
        for selection in selections:
            if not isinstance(selection, dict) or not any(
                isinstance(item, dict)
                and item.get("pattern_index") == selection.get("pattern_index")
                and item.get("surface") == selection.get("surface")
                for item in raw_items
            ):
                raw_items.append(
                    {
                        "pattern_index": selection.get("pattern_index"),
                        "surface": selection.get("surface"),
                    }
                    if isinstance(selection, dict)
                    else selection
                )
        slots, local_failures = validate_batch_members(
            raw_items,
            patterns,
            incumbent,
            failed_slots if repairing else None,
            failed_sources,
            selections,
        )
        if repairing and local_failures and not slots:
            rejection = "; ".join(failure["reason"] for failure in local_failures)
            attempts.append(
                ProposalAttempt(
                    attempt + 1, cached, response, False, rejection, llm_observations=refs
                )
            )
            break
        admissions = []
        new_materialization_failures = []
        new_preflight_failures = local_failures
        new_failed_slots = {
            failure["pattern_index"]: failure["position"]
            for failure in reversed(local_failures)
            if failure["pattern_index"] in {index for index, _ in addressable}
        }
        for position, spec in slots:
            identity = {
                "position": position,
                "pattern_index": spec.pattern_index,
                "surface": spec.surface,
            }
            owner = next(
                (
                    {
                        "position": pos,
                        "pattern_index": other.pattern_index,
                        "surface": other.surface,
                    }
                    for pos, other, _ in materialized
                    if other.surface == spec.surface or other.pattern_index == spec.pattern_index
                ),
                None,
            )
            if owner is not None:
                admissions.append({**identity, "status": "occupied", "owner": owner})
                new_preflight_failures.append(
                    {
                        **identity,
                        "gate": "occupancy",
                        "reason": f"surface {spec.surface} or pattern {spec.pattern_index} is occupied by {owner}",
                        "owner": owner,
                        "predicted_effect": spec.predicted_effect,
                        "behavior": proposal_history_fields(spec, incumbent_serialization),
                    }
                )
                new_failed_slots.setdefault(spec.pattern_index, position)
                continue
            stage = workdir / f"attempt_{attempt + 1:02d}"
            try:
                _, serialization = build_candidate(incumbent, incumbent_serialization, spec, stage)
            except MaterializationFailure as exc:
                new_materialization_failures.append(
                    MaterializationFailureRecord(
                        spec.pattern_index,
                        spec.surface,
                        exc.reason,
                        spec.predicted_effect,
                        proposal_history_fields(spec, incumbent_serialization),
                    )
                )
                new_failed_slots.setdefault(spec.pattern_index, position)
                continue
            violation = revision_violation(
                spec.revision,
                surface=spec.surface,
                mechanism=spec.pattern["signature"]["agent_mechanism"],
                fingerprint=surface_fingerprint(serialization, spec.surface),
                attempts=attempt_index,
            )
            if violation:
                new_preflight_failures.append(
                    {
                        "pattern_index": spec.pattern_index,
                        "surface": spec.surface,
                        "position": position,
                        "gate": "revision",
                        "reason": violation,
                        "required_predecessor": {
                            key: value
                            for key, value in (
                                select_predecessor(
                                    surface=spec.surface,
                                    mechanism=spec.pattern["signature"]["agent_mechanism"],
                                    fingerprint=surface_fingerprint(serialization, spec.surface),
                                    attempts=attempt_index,
                                )
                                or {}
                            ).items()
                            if key in {"round", "subject_id", "decision"}
                        },
                        "predicted_effect": spec.predicted_effect,
                        "behavior": proposal_history_fields(
                            spec, incumbent_serialization, serialization
                        ),
                    }
                )
                new_failed_slots.setdefault(spec.pattern_index, position)
                continue
            if spec.revision is not None:
                predecessor = next(
                    record
                    for record in attempt_index
                    if record["round"] == spec.revision["round"]
                    and record.get("subject_id") == spec.revision["subject_id"]
                )
                previous_fingerprint = predecessor.get("effective_edit_fingerprint")
                spec = replace(
                    spec,
                    revision_unchanged=(
                        previous_fingerprint == surface_fingerprint(serialization, spec.surface)
                    )
                    if previous_fingerprint
                    else None,
                )
            path = write_proposal(
                stage / "proposals",
                _candidate_id(round_index, position, spec.surface),
                incumbent_serialization,
                spec,
                serialization,
                lm.model_name,
                system_sha,
            )
            checked = load_candidate(
                path,
                incumbent,
                caps=caps,
                timeout_seconds=loader_timeout_seconds,
                incumbent_serialization=incumbent_serialization,
                preflight_profile=profile,
            )
            if isinstance(checked, CandidateRejection):
                new_preflight_failures.append(
                    {
                        "pattern_index": spec.pattern_index,
                        "surface": spec.surface,
                        "position": position,
                        "gate": checked.gate,
                        "reason": checked.reason,
                        "predicted_effect": spec.predicted_effect,
                        "behavior": proposal_history_fields(
                            spec, incumbent_serialization, serialization
                        ),
                    }
                )
                new_failed_slots.setdefault(spec.pattern_index, position)
            else:
                materialized.append((position, spec, serialization))
                admissions.append({**identity, "status": "admitted"})
        # A valid repair replaces the failed slots (omission explicitly withdraws them).
        materialization_failures = new_materialization_failures
        preflight_failures = new_preflight_failures
        owned_patterns = {spec.pattern_index for _, spec, _ in materialized}
        failed_slots = {
            index: position
            for index, position in new_failed_slots.items()
            if index not in owned_patterns
        }
        failed_sources = [
            item
            for item in raw_items
            if isinstance(item, dict)
            and type(item.get("pattern_index")) is int
            and item["pattern_index"] in failed_slots
        ]
        reasons = [
            f"pattern {r.pattern_index} {r.surface}: {r.reason} (already unchanged or unmaterializable)"
            for r in materialization_failures
        ]
        reasons.extend(
            f"pattern {r['pattern_index']} {r['surface']} {r['gate']}: {r['reason']}"
            for r in preflight_failures
        )
        rejection = "; ".join(reasons)
        has_failures = bool(materialization_failures or preflight_failures)
        attempts.append(
            ProposalAttempt(
                attempt + 1, cached, response, not has_failures, rejection, admissions, refs
            )
        )
        if repairing or not failed_slots:
            break
        repairing = True
    else:
        if not repairing and addressable:
            error = ProposalRejection(
                f"no valid proposal batch after {config.max_attempts} attempts: {rejection}",
                attempts=attempts,
            )
            persist_proposal_failure(workdir, error, attempts)
            raise error

    all_indices = {index for index, _ in addressable}
    proposed_indices = {spec.pattern_index for _, spec, _ in materialized} | set(failed_slots)
    state = {
        "survivors": [
            {
                "candidate_id": _candidate_id(round_index, position, spec.surface),
                "spec": asdict(spec),
                "serialization": serialization,
            }
            for position, spec, serialization in sorted(materialized, key=lambda item: item[0])
        ],
        "skipped_patterns": sorted(all_indices - proposed_indices),
        "materialization_failures": [r.to_dict() for r in materialization_failures],
        "attempts": [a.to_dict() for a in attempts],
        "prompt_sha256": system_sha,
        "preflight_failures": preflight_failures,
        "preflight_profile": profile,
        "evidence_audit": evidence_audit,
    }
    # Freeze before the first final directory is written. A replay never re-runs
    # nondeterministic/time-limited gates to decide which paid batch to admit.
    pending = result_path.with_suffix(".tmp")
    pending.write_text(
        canonical_json(
            {"contract": contract, "result": state, "sha256": prompt_sha256(canonical_json(state))}
        )
        + "\n"
    )
    pending.replace(result_path)
    return publish_proposal_result(state, proposals_dir, incumbent_serialization, lm.model_name)
