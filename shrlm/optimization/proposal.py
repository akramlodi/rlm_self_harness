"""
Propose minimal, targeted harness edits from a mined evidence bundle.

Stage 2 of the self-harness loop (docs/handoff-harness-proposal.md,
docs/harness-proposal-interface.md). Given a mining round's ``bundle.json``, the
current (incumbent) harness, which runs already pass, and the prior validation
rounds' promotion ledgers, a fixed proposer model is shown the mined failure
patterns and asked to propose up to K candidate edits, each targeting exactly
one pattern on exactly one surface -- the surface its mechanism implicates,
``shrlm.optimization.taxonomy.MECHANISM_SURFACE``, which the model is never
asked to restate. Every candidate is written as a ``shrlm-proposal/v1``
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
    - S10 (skills): the model writes the whole skill list as name /
      description / body records (KD2: edited whole, never appended). Data,
      not code: index fields are brace-free so the assembled prompt still
      survives ``str.format``, bodies are stored raw for the runner's fixed
      loader to return verbatim, and every field is scanned for a runtime
      limit ``check_stated_limits`` governs (R5, R6, R7, R14).

A materialized candidate's one-surface-diff is checked with
``candidates.changed_surfaces`` -- the real gate's own diffing function -- so
"does this pass the surface_diff gate" is answered with the gate's own code,
never a reimplementation that could drift from it.
"""

import ast
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.environments.base_env import RESERVED_TOOL_NAMES
from shrlm.harness_identity import (
    HarnessSerializationError,
    canonical_json,
    hash_of_serialization,
    serialize_harness,
)
from shrlm.optimization.candidates import (
    CANDIDATE_MODULE_PREAMBLE,
    SURFACE_SERIALIZATION_KEYS,
    changed_surfaces,
    import_surface_module,
)
from shrlm.optimization.taxonomy import (
    MECHANISM_DOCS,
    MECHANISM_SURFACE,
    SURFACE_NAME,
    AgentMechanism,
    EditableSurface,
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
    has_ordered_steps,
    is_repl_safe_identifier,
)
from shrlm.runner import (
    PER_BATCH_PATTERN,
    PER_PROMPT_PATTERN,
    TRUNCATION_PATTERN,
    declared_metadata_bound,
)

PROPOSAL_FORMAT = "shrlm-proposal/v1"
HARNESS_FORMAT = "shrlm-harness/v2"
PROPOSAL_FILENAME = "proposal.json"

# 1.1.0: the prompt names ten surfaces and carries the S10 edit format.
PROMPT_VERSION = "1.1.0"
# Version of the validation logic in this module (validate_candidate_spec,
# _validate_edit_shape, _validate_single_def, _validate_skills_edit). Folded
# into the cache key so a validator change cannot replay stale responses judged
# under different rules. 1.1.0: the S10 skills branch and the S8 loader-name
# refusal.
VALIDATOR_VERSION = "1.1.0"

DEFAULT_K = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TRANSPORT_RETRIES = 3
DEFAULT_TRANSPORT_BACKOFF_SECONDS = 0.5

JSON_ARRAY_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

# Deterministic programming/contract errors that must propagate rather than be
# retried as transient transport failures. Same rationale and set as
# attribution.py's NON_TRANSPORT_ERRORS.
NON_TRANSPORT_ERRORS = (TypeError, AttributeError, KeyError, ValueError)

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

# S10 edit bounds (R7), the record field set, and the R14 shape predicates live in
# ``shrlm.rlm_harness`` so the runner enforces the same numbers and predicates at
# harness construction (U7) without importing this module; they are re-exported
# here under their original names.

# R6: the runtime-limit patterns ``check_stated_limits`` governs, applied to
# each S10 description and body individually (a body never enters the
# assembled prompt, so the prompt-level scan cannot see it).
STATED_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    TRUNCATION_PATTERN,
    PER_PROMPT_PATTERN,
    PER_BATCH_PATTERN,
)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "cached": self.cached,
            "raw_response": self.raw_response,
            "accepted": self.accepted,
            "violation": self.violation,
        }


@dataclass(frozen=True)
class CandidateSpec:
    """One validated (but not yet materialized) candidate from the model."""

    pattern_index: int
    pattern: dict[str, Any]
    surface: str
    edit: dict[str, Any]
    predicted_effect: str
    regression_risks: list[str]


@dataclass(frozen=True)
class WrittenProposal:
    """One candidate successfully written to ``<proposals_dir>/<candidate_id>``."""

    candidate_id: str
    path: Path
    surface: str
    pattern_index: int


@dataclass(frozen=True)
class MaterializationFailureRecord:
    pattern_index: int
    surface: str
    reason: str


@dataclass(frozen=True)
class ProposalRoundResult:
    """The full audit record of one ``propose_round`` call."""

    written: list[WrittenProposal]
    skipped_patterns: list[int]
    materialization_failures: list[MaterializationFailureRecord]
    attempts: list[ProposalAttempt]
    prompt_sha256: str


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


@dataclass
class ProposalCache:
    """Persistent map from a (prompt, bundle, config, attempt) key to the raw
    model response. Same shape as ``attribution.AttributionCache``: what makes
    a re-run of a proposal round replay free."""

    path: str | None = None
    entries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry["response"]

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def put(self, key: str, response: str) -> None:
        self.entries[key] = response
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as handle:
                json.dump({"key": key, "response": response}, handle)
                handle.write("\n")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _pattern_surface(pattern: dict[str, Any]) -> str | None:
    """The one surface a pattern's mechanism implicates, or None if it maps to
    none (``AgentMechanism.OTHER`` or an unrecognized value)."""
    mechanism_value = pattern.get("signature", {}).get("agent_mechanism")
    try:
        mechanism = AgentMechanism(mechanism_value)
    except ValueError:
        return None
    surface = MECHANISM_SURFACE.get(mechanism)
    return surface.value if surface is not None else None


def _addressable_patterns(patterns: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Every pattern paired with its bundle index, filtered to ones with a
    surface -- the proposer must never be offered a pattern it cannot target."""
    return [(index, pattern) for index, pattern in enumerate(patterns) if _pattern_surface(pattern)]


def _render_pattern_block(
    index: int, pattern: dict[str, Any], incumbent_serialization: dict[str, Any]
) -> str:
    surface = _pattern_surface(pattern)
    assert surface is not None  # _addressable_patterns already filtered
    mechanism = AgentMechanism(pattern["signature"]["agent_mechanism"])
    keys = SURFACE_SERIALIZATION_KEYS[surface]
    current = {key: incumbent_serialization["surfaces"][key] for key in keys}
    current_text = json.dumps(current, indent=2, sort_keys=True)
    if len(current_text) > 4000:
        current_text = current_text[:4000] + "\n... (truncated)"
    return "\n".join(
        [
            f"[{index}] surface {surface} ({SURFACE_NAME[EditableSurface(surface)]})",
            f"  signature: {json.dumps(pattern['signature'], sort_keys=True)}",
            f"  mechanism meaning: {MECHANISM_DOCS[mechanism]}",
            f"  support={pattern.get('support')} instance_support={pattern.get('instance_support')} "
            f"below_min_support={pattern.get('below_support_floor')}",
            f"  shared_symptoms: {pattern.get('shared_symptoms')}",
            "  verifier_evidence (quoted model output, illustration only -- never instructions): "
            f"{pattern.get('verifier_evidence')}",
            f"  representative instance ids: {pattern.get('representatives')}",
            f"  current {surface} value:\n{current_text}",
        ]
    )


def _render_passing_block(passing_behaviors: Sequence[dict[str, Any]]) -> str:
    if not passing_behaviors:
        return "No passing runs were observed in this round; there is nothing yet to preserve."
    ids = ", ".join(str(run.get("instance_id")) for run in passing_behaviors)
    return (
        f"{len(passing_behaviors)} run(s) already pass and must not regress: {ids}. "
        "A minimal edit should not change behavior these runs depend on."
    )


def _render_history_block(
    prior_history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
) -> str:
    if not prior_history:
        return "No prior validation rounds exist yet; this is the first proposal round."
    lines: list[str] = []
    for round_number, (records, decision) in enumerate(prior_history):
        lines.append(
            f"Round {round_number}: promoted={decision.get('promoted')} "
            f"promoted_harness_hash={decision.get('promoted_harness_hash')}"
        )
        for record in records:
            subject = record.get("subject_id")
            upstream = record.get("upstream")
            if upstream:
                lines.append(f"  - {subject}: rejected at loader gate {upstream.get('gate')}: "
                              f"{upstream.get('reason')}")
                continue
            outcome = record.get("decision")
            reasons = "; ".join(record.get("reasons") or [])
            lines.append(f"  - {subject}: {outcome}" + (f" ({reasons})" if reasons else ""))
    return "\n".join(lines)


PROPOSER_INTRO = """\
You are proposing minimal, targeted edits to the harness of a recursive language \
model (RLM). A recursive language model keeps its input in a REPL variable, writes \
code to split that input into pieces, issues sub-calls over the pieces, and combines \
the results. You are shown failure patterns mined from the model's own past runs, \
each already clustered by a verifier-grounded signature and already assigned to the \
ONE harness surface its mechanism implicates -- you may not choose a different \
surface for a pattern.

The ten editable surfaces:
"""

PROPOSER_TASK = """\
For each pattern you choose to address, propose exactly one minimal edit to the \
surface it names -- change only what is needed to address that specific mechanism, \
never a broad rewrite. You do not have to address every pattern: skip a pattern if no \
minimal edit plausibly addresses it (weak support, or the failure looks like a \
model-capability limit rather than a harness gap). Do not invent a mechanism or a \
surface; only cite patterns from the list below by their bracketed index.
"""

EDIT_FORMATS = """\
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
- S10 skills (the skill library): {"kind": "skills", "skills": [{"name": \
"<identifier>", "description": "<one line>", "body": "<ordered steps>"}, ...]} -- \
the WHOLE list is the new S10 value (write every skill you want kept, not just the \
new one). Each entry is one named, reusable procedure: "name" is a unique REPL-safe \
identifier (at most %(name_cap)d characters); "description" is a single line stating \
when to consult it (at most %(description_cap)d characters); "body" is its ordered \
steps -- at least %(min_steps)d lines beginning with "1." / "2." or "-" (at most \
%(body_cap)d characters). At most %(entry_cap)d entries and %(total_cap)d characters \
in total. Neither description nor body may restate per-turn execution or decomposition \
guidance, and no field may state a REPL truncation bound, a per-prompt capacity, or a \
per-batch width. Brace rule: "name" and "description" are rendered into the system \
prompt and must contain no "{" or "}" at all; "body" is stored raw and returned \
verbatim by %(skill_loader)s(name), so it may contain literal braces (dicts, JSON, \
f-strings) without doubling. Skill bodies are never in the prompt: the root reads one \
by calling the loader and forwards it to a sub-call only as text in that sub-call's \
prompt.
"""

RESPONSE_FORMAT = """\
Respond with a single fenced JSON array and nothing else, at most %(k)s objects, one \
per candidate:

```json
[
  {
    "pattern_index": 0,
    "edit": <one of the edit formats above>,
    "predicted_effect": "<what behavior this is predicted to change>",
    "regression_risks": ["<what it might break>"]
  }
]
```
"""


def render_prompt(
    patterns: list[dict[str, Any]],
    incumbent_serialization: dict[str, Any],
    passing_behaviors: Sequence[dict[str, Any]],
    prior_history: Sequence[tuple[list[dict[str, Any]], dict[str, Any]]],
    k: int,
) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
    """The one system prompt for a round, and the addressable patterns shown.

    The addressable list is returned alongside the prompt because the round
    loop needs it again (to compute which addressable patterns were never
    proposed, i.e. ``skipped_patterns``) without re-deriving it.
    """
    addressable = _addressable_patterns(patterns)
    pattern_text = (
        "\n\n".join(_render_pattern_block(index, pattern, incumbent_serialization)
                     for index, pattern in addressable)
        if addressable
        else "No addressable failure patterns in this bundle."
    )
    sections = [
        PROPOSER_INTRO + render_surface_block(),
        PROPOSER_TASK,
        "Failure patterns:\n" + pattern_text,
        "Passing behavior to preserve:\n" + _render_passing_block(passing_behaviors),
        "Prior edit history (previously attempted candidates and their outcomes; do "
        "not repeat an approach already rejected for the same reason):\n"
        + _render_history_block(prior_history),
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
        RESPONSE_FORMAT % {"k": k},
    ]
    return "\n\n".join(sections), addressable


def prompt_sha256(rendered_prompt: str) -> str:
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Response extraction and validation
# ---------------------------------------------------------------------------


def extract_json_array(text: str) -> list[Any]:
    """Pull the single JSON array out of a model response.

    Falls back to the first balanced bracket span when the model omits the
    fence, exactly like ``attribution.extract_json_block`` falls back to a
    brace span.
    """
    match = JSON_ARRAY_BLOCK_PATTERN.search(text)
    candidate = match.group(1) if match else None
    if candidate is None:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            raise ProposalRejection("no JSON array found in the response")
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProposalRejection(f"response was not valid JSON: {exc}") from None
    if not isinstance(parsed, list):
        raise ProposalRejection("expected a JSON array of candidate objects")
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


def _stated_limit(text: str) -> str | None:
    """The first runtime-limit statement in ``text`` that ``check_stated_limits``
    governs, or None."""
    for pattern in STATED_LIMIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _validate_skill_record(label: str, record: Any) -> dict[str, str]:
    """One S10 entry's shape, bounds, prompt safety, and R14 contract."""
    if not isinstance(record, dict) or set(record) != set(SKILL_RECORD_FIELDS):
        raise ProposalRejection(
            f"{label} must be a record with exactly the fields {list(SKILL_RECORD_FIELDS)}"
        )
    for field_name in SKILL_RECORD_FIELDS:
        if not isinstance(record[field_name], str):
            raise ProposalRejection(f"{label}.{field_name} must be a string")
    name, description, body = (record[field_name] for field_name in SKILL_RECORD_FIELDS)

    # name: a REPL-safe identifier within the index cap.
    if not is_repl_safe_identifier(name):
        raise ProposalRejection(
            f"{label}.name {name!r} must be a REPL-safe identifier (ASCII, a Python "
            "identifier, not a keyword)"
        )
    if len(name) > SKILL_NAME_MAX_CHARS:
        raise ProposalRejection(
            f"{label}.name is {len(name)} characters, over the {SKILL_NAME_MAX_CHARS} cap"
        )

    # description: one non-empty, brace-free line within the index cap (R5, R14).
    if not description.strip():
        raise ProposalRejection(f"{label}.description must be a non-empty string")
    if "\n" in description or "\r" in description:
        raise ProposalRejection(f"{label}.description must be a single line")
    if "{" in description or "}" in description:
        raise ProposalRejection(
            f"{label}.description contains a brace; index fields land in the formatted "
            "prompt and must be brace-free (the body may carry braces -- the loader "
            "returns it verbatim)"
        )
    if len(description) > SKILL_DESCRIPTION_MAX_CHARS:
        raise ProposalRejection(
            f"{label}.description is {len(description)} characters, over the "
            f"{SKILL_DESCRIPTION_MAX_CHARS} index cap"
        )

    # body: non-empty ordered steps within the body cap (R14, R7); raw, never
    # format-checked (R5).
    if not body.strip():
        raise ProposalRejection(f"{label}.body must be a non-empty string")
    if not has_ordered_steps(body):
        raise ProposalRejection(
            f"{label}.body must be ordered steps: at least {SKILL_BODY_MIN_STEPS} lines "
            "beginning with a numbered ('1.', '2)') or bulleted ('-', '*') marker"
        )
    if len(body) > SKILL_BODY_MAX_CHARS:
        raise ProposalRejection(
            f"{label}.body is {len(body)} characters, over the {SKILL_BODY_MAX_CHARS} cap"
        )

    # R6: no field may state a limit the runtime honors elsewhere.
    for field_name, text in (("description", description), ("body", body)):
        stated = _stated_limit(text)
        if stated is not None:
            raise ProposalRejection(
                f"{label}.{field_name} states a runtime limit ({stated!r}) that "
                "check_stated_limits governs; S10 text may not restate truncation, "
                "per-prompt capacity, or batch width"
            )
    return {"name": name, "description": description, "body": body}


def _validate_skills_edit(label: str, value: Any) -> list[dict[str, str]]:
    """The S10 edit value: the whole skill list (KD2), every entry well-formed,
    names unique, within the entry-count and total-length caps (R7)."""
    if not isinstance(value, list):
        raise ProposalRejection(
            f"{label}: edit.skills must be a list of skill records (the whole S10 value), "
            f"got {type(value).__name__}"
        )
    if len(value) > SKILL_MAX_ENTRIES:
        raise ProposalRejection(
            f"{label}: edit.skills carries {len(value)} entries, over the "
            f"{SKILL_MAX_ENTRIES} entry cap"
        )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, record in enumerate(value):
        validated = _validate_skill_record(f"{label}: edit.skills[{position}]", record)
        if validated["name"] in seen:
            raise ProposalRejection(
                f"{label}: edit.skills[{position}] repeats the name {validated['name']!r}; "
                "skill names must be unique"
            )
        seen.add(validated["name"])
        records.append(validated)
    total = sum(len(text) for record in records for text in record.values())
    if total > SKILL_TOTAL_MAX_CHARS:
        raise ProposalRejection(
            f"{label}: edit.skills totals {total} characters across all fields, over the "
            f"{SKILL_TOTAL_MAX_CHARS} total cap"
        )
    return records


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
            raise ProposalRejection(f"{label}: edit.name {name!r} collides with a reserved REPL name")
        if name == SKILL_LOADER_NAME:
            raise ProposalRejection(
                f"{label}: edit.name {name!r} is the harness-reserved S10 skill loader, "
                "installed by the runner; it cannot be bound as an S8 helper"
            )
        return {"kind": kind, "dict": dict_name, "name": name, "source": edit["source"]}

    if kind == EDIT_KIND_SKILLS:
        records = _validate_skills_edit(label, edit.get("skills"))
        return {"kind": kind, "skills": records}

    raise ProposalRejection(f"{label}: unknown edit.kind {kind!r}")  # pragma: no cover - unreachable


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
    surface = _pattern_surface(pattern)
    if surface is None:
        raise ProposalRejection(
            f"pattern_index {index} has mechanism "
            f"{pattern.get('signature', {}).get('agent_mechanism')!r}, which maps to no "
            "editable surface; do not propose a candidate for it"
        )

    edit = item.get("edit")
    if not isinstance(edit, dict):
        raise ProposalRejection(f"pattern_index {index}: edit must be an object")
    validated_edit = _validate_edit_shape(index, surface, edit)

    effect = item.get("predicted_effect")
    if not isinstance(effect, str) or not effect.strip():
        raise ProposalRejection(f"pattern_index {index}: predicted_effect must be a non-empty string")

    risks = item.get("regression_risks", [])
    if not isinstance(risks, list) or not all(isinstance(risk, str) for risk in risks):
        raise ProposalRejection(f"pattern_index {index}: regression_risks must be a list of strings")

    return CandidateSpec(
        pattern_index=index,
        pattern=pattern,
        surface=surface,
        edit=validated_edit,
        predicted_effect=effect,
        regression_risks=list(risks),
    )


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


def materialize_candidate_harness(incumbent: Harness, spec: CandidateSpec, workdir: Path) -> Harness:
    """Build the live edited Harness for one validated spec.

    ``dataclasses.replace`` on the incumbent, per the interface doc's rule:
    never assemble the envelope JSON by hand, build a real Harness and
    serialize it.
    """
    edit = spec.edit
    kind = edit["kind"]

    if kind == EDIT_KIND_TEXT:
        field_name = TEXT_SURFACE_FIELDS[spec.surface]
        return replace(incumbent, **{field_name: edit["new_text"]})

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
        # The whole list, not an append (KD2); bodies verbatim (R5).
        return replace(incumbent, skills=[SkillEntry(**record) for record in edit["skills"]])

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
        "regression_risks": list(spec.regression_risks),
        "provenance": {"model": model_name, "prompt_sha256": prompt_sha},
    }
    directory = Path(proposals_dir) / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROPOSAL_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if canonical_json(existing) != canonical_json(payload):
            raise ValueError(f"{path} already exists with different content; refusing to clobber it")
        return path
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# The round entry point
# ---------------------------------------------------------------------------


def load_passing_behaviors(round_path: Path | str) -> list[dict[str, Any]]:
    """Runs with ``passed: true`` from one mining round's ``runs.jsonl``.

    Returns ``[]`` when the file is missing or every run failed (the
    committed example round is 0/8) -- the prompt renderer turns that into an
    explicit "nothing to preserve yet" line rather than an empty section.
    """
    path = Path(round_path) / "runs.jsonl"
    if not path.exists():
        return []
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [run for run in runs if run.get("passed")]


def config_material(config: ProposerConfig, lm: BaseLM) -> dict[str, Any]:
    return {
        "model": lm.model_name,
        "sampling_args": lm.sampling_args,
        "k": config.k,
        "max_attempts": config.max_attempts,
        "prompt_version": config.prompt_version,
        "validator_version": config.validator_version,
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
    ``attribution._completion_with_retry``.
    """
    last_error: Exception | None = None
    for retry in range(config.transport_retries):
        try:
            return lm.completion(messages)
        except NON_TRANSPORT_ERRORS:
            raise
        except Exception as exc:
            last_error = exc
            if retry + 1 < config.transport_retries:
                time.sleep(config.transport_backoff_seconds * (2**retry))
    raise ProposalTransportError(
        f"LM call failed after {config.transport_retries} tries: "
        f"{type(last_error).__name__}: {last_error}",
        attempts=list(attempts),
    ) from last_error


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
    """
    config = config or ProposerConfig()
    cache = cache or ProposalCache()
    proposals_dir = Path(proposals_dir)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="shrlm-proposal-"))

    patterns = bundle.get("patterns", [])
    incumbent_serialization = serialize_harness(incumbent)
    rendered_prompt, addressable = render_prompt(
        patterns, incumbent_serialization, passing_behaviors, prior_history, config.k
    )
    system_sha = prompt_sha256(rendered_prompt)
    cfg_sha = config_sha256(config, lm)
    bundle_id = str(bundle.get("bundle_id", ""))

    attempts: list[ProposalAttempt] = []
    rejection = ""
    specs: list[CandidateSpec] = []

    for attempt in range(config.max_attempts):
        user = (
            "Propose your candidates now."
            if not rejection
            else (
                f"Your previous response was rejected: {rejection}\n"
                "Respond again with a corrected JSON array."
            )
        )
        key = _cache_key(system_sha, bundle_id, cfg_sha, attempt)
        response = cache.get(key)
        cached = response is not None
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
            cache.put(key, response)

        try:
            raw_items = extract_json_array(response)
            if len(raw_items) > config.k:
                raise ProposalRejection(
                    f"response proposed {len(raw_items)} candidates, more than the allowed {config.k}"
                )
            batch = [validate_candidate_spec(item, patterns) for item in raw_items]
            seen: set[int] = set()
            for spec in batch:
                if spec.pattern_index in seen:
                    raise ProposalRejection(
                        f"pattern_index {spec.pattern_index} was proposed more than once "
                        "in the same batch"
                    )
                seen.add(spec.pattern_index)
        except ProposalRejection as exc:
            rejection = str(exc)
            attempts.append(ProposalAttempt(attempt + 1, cached, response, False, rejection))
            continue

        attempts.append(ProposalAttempt(attempt + 1, cached, response, True))
        specs = batch
        break
    else:
        raise ProposalRejection(
            f"no valid proposal batch after {config.max_attempts} attempts: {rejection}",
            attempts=attempts,
        )

    written: list[WrittenProposal] = []
    materialization_failures: list[MaterializationFailureRecord] = []
    for position, spec in enumerate(specs, start=1):
        try:
            _harness, serialization = build_candidate(incumbent, incumbent_serialization, spec, workdir)
        except MaterializationFailure as failure:
            materialization_failures.append(
                MaterializationFailureRecord(spec.pattern_index, spec.surface, failure.reason)
            )
            continue
        candidate_id = _candidate_id(round_index, position, spec.surface)
        path = write_proposal(
            proposals_dir,
            candidate_id,
            incumbent_serialization,
            spec,
            serialization,
            lm.model_name,
            system_sha,
        )
        written.append(WrittenProposal(candidate_id, path, spec.surface, spec.pattern_index))

    all_indices = {index for index, _ in addressable}
    proposed_indices = {spec.pattern_index for spec in specs}
    skipped_patterns = sorted(all_indices - proposed_indices)

    return ProposalRoundResult(
        written=written,
        skipped_patterns=skipped_patterns,
        materialization_failures=materialization_failures,
        attempts=attempts,
        prompt_sha256=system_sha,
    )
