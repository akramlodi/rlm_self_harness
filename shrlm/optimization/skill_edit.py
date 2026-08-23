"""
S10 record validation shared by the proposal-time belt and the candidate gate.

The S10 (skills) surface is data, not code: the whole skill list as name /
description / body records (KD2). Both layers that check those records live
here so neither ``proposal.py`` nor ``candidates.py`` carries the S10
vocabulary itself:

    - ``_validate_skills_edit`` / ``_validate_skill_record``: the proposal-time
      belt -- record shape, the R7 bounds, REPL-safe names, R5 prompt safety of
      index fields, R14 ordered-steps bodies, and the R6 stated-limit scan over
      each description and body individually (a body never enters the
      assembled prompt, so the prompt-level scan cannot see it). Violations
      raise ``SkillEditRejection``; ``proposal._validate_edit_shape`` translates
      it into its own ``ProposalRejection`` so the re-ask loop sees one type.
    - ``_skills_violation``: the shape-only check the candidate gate runs over
      a serialized S10 list, returned as a value (the gate's rejections are
      values, never exceptions).

``SKILLS_EDIT_FORMAT`` is the S10 bullet of the proposer prompt's edit-format
block, kept beside the validator that enforces it. The bounds themselves live
in ``shrlm.rlm_harness`` so the runner enforces the same numbers at harness
construction (U7) without importing this package.
"""

import re
from typing import Any

from shrlm.rlm_harness import (
    SKILL_BODY_MAX_CHARS,
    SKILL_BODY_MIN_STEPS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_MAX_ENTRIES,
    SKILL_NAME_MAX_CHARS,
    SKILL_RECORD_FIELDS,
    SKILL_TOTAL_MAX_CHARS,
    has_ordered_steps,
    is_repl_safe_identifier,
    skill_description_violation,
)
from shrlm.runner import PER_BATCH_PATTERN, PER_PROMPT_PATTERN, TRUNCATION_PATTERN

# R6: the runtime-limit patterns ``check_stated_limits`` governs, applied to
# each S10 description and body individually (a body never enters the
# assembled prompt, so the prompt-level scan cannot see it).
STATED_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    TRUNCATION_PATTERN,
    PER_PROMPT_PATTERN,
    PER_BATCH_PATTERN,
)

# The S10 bullet of the proposer's edit-format block (``proposal.EDIT_FORMATS``
# appends it verbatim); ``%``-interpolated there with the caps above.
SKILLS_EDIT_FORMAT = """\
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


class SkillEditRejection(Exception):
    """An S10 edit value that did not satisfy the record contract.

    Local to this module so it needs nothing from ``proposal.py``; the proposal
    layer re-raises it as ``ProposalRejection`` with the same message.
    """


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
        raise SkillEditRejection(
            f"{label} must be a record with exactly the fields {list(SKILL_RECORD_FIELDS)}"
        )
    for field_name in SKILL_RECORD_FIELDS:
        if not isinstance(record[field_name], str):
            raise SkillEditRejection(f"{label}.{field_name} must be a string")
    name, description, body = (record[field_name] for field_name in SKILL_RECORD_FIELDS)

    # name: a REPL-safe identifier within the index cap.
    if not is_repl_safe_identifier(name):
        raise SkillEditRejection(
            f"{label}.name {name!r} must be a REPL-safe identifier (ASCII, a Python "
            "identifier, not a keyword)"
        )
    if len(name) > SKILL_NAME_MAX_CHARS:
        raise SkillEditRejection(
            f"{label}.name is {len(name)} characters, over the {SKILL_NAME_MAX_CHARS} cap"
        )

    # description: one non-empty, brace-free line within the index cap (R5, R14).
    violation = skill_description_violation(description)
    if violation is not None:
        raise SkillEditRejection(f"{label}.description {violation}")
    if len(description) > SKILL_DESCRIPTION_MAX_CHARS:
        raise SkillEditRejection(
            f"{label}.description is {len(description)} characters, over the "
            f"{SKILL_DESCRIPTION_MAX_CHARS} index cap"
        )

    # body: non-empty ordered steps within the body cap (R14, R7); raw, never
    # format-checked (R5).
    if not body.strip():
        raise SkillEditRejection(f"{label}.body must be a non-empty string")
    if not has_ordered_steps(body):
        raise SkillEditRejection(
            f"{label}.body must be ordered steps: at least {SKILL_BODY_MIN_STEPS} lines "
            "beginning with a numbered ('1.', '2)') or bulleted ('-', '*') marker"
        )
    if len(body) > SKILL_BODY_MAX_CHARS:
        raise SkillEditRejection(
            f"{label}.body is {len(body)} characters, over the {SKILL_BODY_MAX_CHARS} cap"
        )

    # R6: no field may state a limit the runtime honors elsewhere.
    for field_name, text in (("description", description), ("body", body)):
        stated = _stated_limit(text)
        if stated is not None:
            raise SkillEditRejection(
                f"{label}.{field_name} states a runtime limit ({stated!r}) that "
                "check_stated_limits governs; S10 text may not restate truncation, "
                "per-prompt capacity, or batch width"
            )
    return {"name": name, "description": description, "body": body}


def _validate_skills_edit(label: str, value: Any) -> list[dict[str, str]]:
    """The S10 edit value: the whole skill list (KD2), every entry well-formed,
    names unique, within the entry-count and total-length caps (R7)."""
    if not isinstance(value, list):
        raise SkillEditRejection(
            f"{label}: edit.skills must be a list of skill records (the whole S10 value), "
            f"got {type(value).__name__}"
        )
    if len(value) > SKILL_MAX_ENTRIES:
        raise SkillEditRejection(
            f"{label}: edit.skills carries {len(value)} entries, over the "
            f"{SKILL_MAX_ENTRIES} entry cap"
        )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, record in enumerate(value):
        validated = _validate_skill_record(f"{label}: edit.skills[{position}]", record)
        if validated["name"] in seen:
            raise SkillEditRejection(
                f"{label}: edit.skills[{position}] repeats the name {validated['name']!r}; "
                "skill names must be unique"
            )
        seen.add(validated["name"])
        records.append(validated)
    total = sum(len(text) for record in records for text in record.values())
    if total > SKILL_TOTAL_MAX_CHARS:
        raise SkillEditRejection(
            f"{label}: edit.skills totals {total} characters across all fields, over the "
            f"{SKILL_TOTAL_MAX_CHARS} total cap"
        )
    return records


def _skills_violation(skills: Any) -> str | None:
    """The first shape violation in a serialized S10 list, or None when well-formed.

    Shape only — a list of records, each carrying string ``name``,
    ``description`` and ``body``, with unique names — so ``SkillEntry(**record)``
    cannot raise at materialization. The edit-kind bounds (R7) and the prompt
    safety of index fields (R5) are the proposal layer's and
    ``check_harness``'s concerns, not this gate's.
    """
    if not isinstance(skills, list):
        return "surfaces['S10_skills'] must be a list of skill records"
    seen: set[str] = set()
    for index, record in enumerate(skills):
        label = f"surfaces['S10_skills'][{index}]"
        if not isinstance(record, dict) or set(record) != set(SKILL_RECORD_FIELDS):
            return f"{label} must be a record with exactly the fields {list(SKILL_RECORD_FIELDS)}"
        for field in SKILL_RECORD_FIELDS:
            if not isinstance(record[field], str):
                return f"{label}[{field!r}] must be a string"
        if record["name"] in seen:
            return f"{label} repeats the name {record['name']!r}; skill names must be unique"
        seen.add(record["name"])
    return None
