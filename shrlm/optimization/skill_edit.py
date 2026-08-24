"""
S10 record validation shared by the proposal-time belt and the candidate gate.

The S10 (skills) surface is data, not code: a list of name / description /
body records. The *edit* is one named skill, added or replacing the same
name, like an S8 helper -- not a whole-list rewrite. A record whose
``description`` and ``body`` are both exactly empty is the removal form: it
deletes the named entry instead. Both layers that check those records live
here so neither ``proposal.py`` nor ``candidates.py`` carries the S10
vocabulary itself:

    - ``_validate_skill_edit`` / ``_validate_skill_record``: the proposal-time
      belt -- one skill's shape, the per-field R7 bounds, REPL-safe names, R5
      prompt safety of index fields, R14 ordered-steps bodies, and the R6
      stated-limit scan over description and body individually (a body never
      enters the assembled prompt, so the prompt-level scan cannot see it).
      Violations raise ``SkillEditRejection``; ``proposal._validate_edit_shape``
      translates it into its own ``ProposalRejection`` so the re-ask loop sees
      one type.
    - ``_merge_skill``: apply that one skill (or removal) to the incumbent
      list; the merged library is then run through the runner's own
      ``check_skills`` so the entry-count and total-length caps (R7) have a
      single enforcement site rather than a re-implementation here.
    - ``_skills_violation``: the shape-only check the candidate gate runs over
      a serialized S10 list, returned as a value (the gate's rejections are
      values, never exceptions).

``SKILLS_EDIT_FORMAT`` is the compact S10 bullet of the proposer prompt's
edit-format block, kept beside the validator that enforces it; the longer
skill-writing pedagogy lives in ``proposal.py`` beside the other surfaces'
prompt guidance and is appended only when a pattern actually targets S10. The
bounds themselves live in ``shrlm.rlm_harness`` so the runner enforces the
same numbers at harness construction (U7) without importing this package.
"""

import re
from dataclasses import replace
from typing import Any

from shrlm.rlm_harness import (
    H0,
    SKILL_BODY_MAX_CHARS,
    SKILL_BODY_MIN_STEPS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_NAME_MAX_CHARS,
    SKILL_RECORD_FIELDS,
    SkillEntry,
    has_ordered_steps,
    is_repl_safe_identifier,
    skill_description_violation,
)
from shrlm.runner import (
    PER_BATCH_PATTERN,
    PER_PROMPT_PATTERN,
    TRUNCATION_PATTERN,
    check_skills,
)

# R6: the runtime-limit patterns ``check_stated_limits`` governs, applied to
# each S10 description and body individually (a body never enters the
# assembled prompt, so the prompt-level scan cannot see it).
STATED_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    TRUNCATION_PATTERN,
    PER_PROMPT_PATTERN,
    PER_BATCH_PATTERN,
)

# The compact S10 bullet of the proposer's edit-format block
# (``proposal.EDIT_FORMATS`` appends it verbatim); ``%``-interpolated there with
# the caps above, so literal braces are safe. The schema sentence is the
# validator contract; the skill-writing pedagogy lives in
# ``proposal.SKILLS_PEDAGOGY`` and is appended only when a pattern targets S10.
SKILLS_EDIT_FORMAT = """\
- S10 skills (the skill library): {"kind": "skills", "name": "<identifier>", \
"description": "<one line>", "body": "<ordered steps>"} -- one skill, added or \
replacing the existing skill of the same name (like S8); every other skill stays. \
To remove an existing skill instead, send its "name" with "description": "" and \
"body": "" (both exactly empty); removing a name not in the library is rejected. \
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


def _validate_skill_record(label: str, record: dict[str, str]) -> dict[str, str]:
    """One S10 entry's shape, bounds, prompt safety, and R14 contract.

    ``record`` carries exactly ``SKILL_RECORD_FIELDS`` as strings --
    ``_validate_skill_edit`` enforces that before calling.
    """
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


def _is_removal(record: dict[str, str]) -> bool:
    """The S10 removal form: ``description`` and ``body`` both exactly empty."""
    return record["description"] == "" and record["body"] == ""


def _validate_skill_edit(label: str, edit: dict[str, Any]) -> dict[str, str]:
    """One S10 skill to add or replace by name (like an S8 helper), or the
    removal form -- ``description`` and ``body`` both empty strings -- which
    deletes the named entry.

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
    missing = set(SKILL_RECORD_FIELDS) - set(edit)
    if missing:
        raise SkillEditRejection(
            f"{label}: missing edit fields {sorted(missing)}; an S10 edit carries kind, "
            f"{', '.join(SKILL_RECORD_FIELDS)}"
        )
    record = {field: edit[field] for field in SKILL_RECORD_FIELDS}
    for field_name in SKILL_RECORD_FIELDS:
        if not isinstance(record[field_name], str):
            raise SkillEditRejection(f"{label}: edit.{field_name} must be a string")
    if _is_removal(record):
        if not record["name"]:
            raise SkillEditRejection(
                f"{label}: edit.name must name the skill to remove (removal form: "
                'description and body both "")'
            )
        return record
    return _validate_skill_record(f"{label}: edit", record)


def _merge_skill(incumbent: list[SkillEntry], record: dict[str, str]) -> list[SkillEntry]:
    """Apply one validated S10 record to the incumbent library.

    Add ``record`` or replace the incumbent skill of the same name; the removal
    form (``description`` and ``body`` both empty) deletes the named entry, and
    removing a name the library does not carry is a rejection naming the
    existing entries. The merged library is then run through the runner's own
    ``check_skills`` -- the single site enforcing the entry-count and
    total-length caps (R7) -- so an overflow is a rejection naming the cap and
    the totals, never a silent drop of an older skill.
    """
    name = record["name"]
    if _is_removal(record):
        existing = [entry.name for entry in incumbent]
        if name not in existing:
            raise SkillEditRejection(
                f"S10 removal names {name!r}, which is not in the library; existing "
                f"entries: {', '.join(existing) if existing else '(none)'}"
            )
        merged = [entry for entry in incumbent if entry.name != name]
    else:
        incoming = SkillEntry(**record)
        merged = []
        replaced = False
        for entry in incumbent:
            if entry.name == incoming.name:
                merged.append(incoming)
                replaced = True
            else:
                merged.append(entry)
        if not replaced:
            merged.append(incoming)
    try:
        check_skills(replace(H0, skills=merged))
    except ValueError as error:
        raise SkillEditRejection(f"S10 edit {name!r} rejected at merge: {error}") from None
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
