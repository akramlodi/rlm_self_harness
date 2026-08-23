"""
S10 record validation shared by the proposal-time belt and the candidate gate.

The S10 (skills) surface is data, not code: a list of name / description /
body records. The *edit* is one named skill, added or replacing the same
name, like an S8 helper -- not a whole-list rewrite. Both layers that check
those records live here so neither ``proposal.py`` nor ``candidates.py``
carries the S10 vocabulary itself:

    - ``_validate_skill_edit`` / ``_validate_skill_record``: the proposal-time
      belt -- one skill's shape, the per-field R7 bounds, REPL-safe names, R5
      prompt safety of index fields, R14 ordered-steps bodies, and the R6
      stated-limit scan over description and body individually (a body never
      enters the assembled prompt, so the prompt-level scan cannot see it).
      Violations raise ``SkillEditRejection``; ``proposal._validate_edit_shape``
      translates it into its own ``ProposalRejection`` so the re-ask loop sees
      one type.
    - ``_merge_skill``: apply that one skill to the incumbent list; the
      merged library must still satisfy the entry-count and total-length caps.
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
    SkillEntry,
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
# appends it verbatim); ``%``-interpolated there with the caps above. The schema
# sentence is the validator contract. The writing guidance after it is how to
# fill a body: distill the shown traces into a procedure, not a knowledge dump.
SKILLS_EDIT_FORMAT = """\
- S10 skills (the skill library): {"kind": "skills", "name": "<identifier>", \
"description": "<one line>", "body": "<ordered steps>"} -- one skill, added or \
replacing the existing skill of the same name (like S8); every other skill stays. \
"name" is a REPL-safe identifier (at most %(name_cap)d characters); "description" \
is a single line stating when to consult it (at most %(description_cap)d characters); \
"body" is its ordered steps -- at least %(min_steps)d lines beginning with "1." / \
"2." or "-" (at most %(body_cap)d characters). The merged library is at most \
%(entry_cap)d entries and %(total_cap)d characters in total. Neither description \
nor body may restate per-turn execution or decomposition guidance, and no field \
may state a REPL truncation bound, a per-prompt capacity, or a per-batch width. \
Brace rule: "name" and "description" are rendered into the system prompt and must \
contain no "{" or "}" at all; "body" is stored raw and returned verbatim by \
%(skill_loader)s(name), so it may contain literal braces (dicts, JSON, f-strings) \
without doubling. Skill bodies are never in the prompt: the root reads one by \
calling the loader and forwards it to a sub-call only as text in that sub-call's \
prompt.

When proposing an S10 edit, distill the shown failure pattern's traces into a \
reusable skill.

A skill is a procedural anchor, not a knowledge dump. Its job is to stabilize \
action: setup steps, tool sequences, checks, and pitfalls. Do not paste raw \
trajectory content -- compress it. Verbose process residue (exploration, dead \
ends, debugging noise) wastes the load budget and the sub-call prompt the body \
is forwarded into.

Write ONE skill in this format. It is added, or it replaces the existing skill \
of the same name; every other skill stays.

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


def _validate_skill_edit(label: str, edit: dict[str, Any]) -> dict[str, str]:
    """One S10 skill to add or replace by name (like an S8 helper).

    A leftover ``skills`` list is refused by name so a whole-list rewrite
    cannot silently become "ignore the list, take these three fields."
    """
    if "skills" in edit:
        raise SkillEditRejection(
            f"{label}: S10 is one skill added or replacing the same name, like an S8 "
            "helper; do not send a whole list under edit.skills"
        )
    unknown = set(edit) - {"kind", *SKILL_RECORD_FIELDS}
    if unknown:
        raise SkillEditRejection(
            f"{label}: unknown edit fields {sorted(unknown)}; legal fields: kind, "
            f"{', '.join(SKILL_RECORD_FIELDS)}"
        )
    record = {field: edit.get(field) for field in SKILL_RECORD_FIELDS}
    return _validate_skill_record(f"{label}: edit", record)


def _merge_skill(incumbent: list[SkillEntry], record: dict[str, str]) -> list[SkillEntry]:
    """Add ``record`` or replace the incumbent skill of the same name.

    The merged library must stay inside the entry-count and total-length caps
    (R7). A ninth distinct name or a total that overflows is a rejection, not
    a silent drop of an older skill.
    """
    incoming = SkillEntry(**record)
    merged: list[SkillEntry] = []
    replaced = False
    for entry in incumbent:
        if entry.name == incoming.name:
            merged.append(incoming)
            replaced = True
        else:
            merged.append(entry)
    if not replaced:
        merged.append(incoming)
    if len(merged) > SKILL_MAX_ENTRIES:
        raise SkillEditRejection(
            f"S10 skills would carry {len(merged)} entries after adding "
            f"{incoming.name!r}, over the {SKILL_MAX_ENTRIES} entry cap"
        )
    total = sum(len(getattr(entry, field)) for entry in merged for field in SKILL_RECORD_FIELDS)
    if total > SKILL_TOTAL_MAX_CHARS:
        raise SkillEditRejection(
            f"S10 skills would total {total} characters across all fields after adding "
            f"{incoming.name!r}, over the {SKILL_TOTAL_MAX_CHARS} total cap"
        )
    return merged


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
