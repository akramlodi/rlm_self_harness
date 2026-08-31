"""
OOLONG environment: instances, a mining-side Verifier, and a SubVerifier.

OOLONG (https://arxiv.org/abs/2511.02817, Bertsch et al., "Oolong: Evaluating
Long Context Reasoning and Aggregation Capabilities") asks a *distributional*
question over a long context of many discrete labelled examples -- count a label
class, name the user/date with the most instances, decide whether a label was
more common before or after a date. The answer is a property of the whole input,
so unlike GraphWalks it cannot be shortcut by a regex-and-scan single pass: a
faithful solver has to classify every chunk and aggregate across all of them.
That is what makes OOLONG the environment where the recursion taxonomy and the
sub-call surfaces actually have something to exercise (experiment_kimi's
POST_MORTEM: GraphWalks-short fits one REPL BFS, so 0 of 5,104 runs recursed).

Two task sets, both under ``oolongbench`` on Hugging Face:

* **OOLONG-synth** (``oolongbench/oolong-synth``): constructed from in-context
  learning datasets, cleanly decomposable, numeric / label / user / date /
  month-year / comparison answers. This feeds the optimization loop -- mining,
  attribution, proposal, validation, promotion. Its
  ``context_window_text_with_labels`` field carries the gold label on every
  example line, which is exactly what ``OolongSubVerifier`` grounds against.
* **OOLONG-real** (``oolongbench/oolong-real``, config ``dnd``): the same
  question shapes asked over real Critical Role Dungeons & Dragons transcripts
  with human-annotated gold. numeric / string / list-of-strings answers. Used
  only as a periodic generalization check that is run but *never* fed into the
  promotion gate -- real transcripts are not cleanly decomposable and would add
  a second uncontrolled noise source on top of the gate's known sensitivity.
  Not decomposable the synth way, so there is deliberately no sub-verifier for
  it (``sub_verifier=None``).

Scoring, verbatim from the paper (section 2.3):

* numeric answers -> partial credit ``score = 0.75 ** |y - y_hat|`` (0.0 when the
  prediction carries no number);
* label / user id / date / month-year / comparison -> exact match after
  normalization;
* OOLONG-real list-of-strings -> set overlap (Jaccard) for the score, and exact
  set equality for the binary "pass".

The extraction contract is made visible two ways so experiment_kimi's POST_MORTEM
Finding 1 (an *unstated* verifier contract turned 84% of correct answers into
"failures") cannot recur: ``ANSWER_FORMAT_CONTRACT`` is appended to every prompt
the loaders build, and ``OolongVerifier.config()`` returns ``extraction_rule`` +
``answer_kinds``, which ``shrlm.optimization.mining`` folds into the bundle's
``verifier_config`` and ``shrlm.optimization.proposal._render_verifier_contract``
renders into the proposer's pattern block.

The parsing/scoring helpers are re-implemented here rather than imported from
``shrlm.environments.oolong_pairs`` or ``examples/`` -- OOLONG-Pairs is a
different benchmark and the demo modules are entry points, not library code.
Resource exceptions never reach the Verifier: its signature receives only a
produced string, and RESOURCE_TERMINATED is owned by the experiment driver.
"""

from __future__ import annotations

import ast
import hashlib
import queue
import random
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict

SYNTH_DATASET_REPO = "oolongbench/oolong-synth"
REAL_DATASET_REPO = "oolongbench/oolong-real"

# The context lengths OOLONG-synth is defined over, in TOKENS (powers of two).
# The loader can only filter existing upstream rows to these, never mint new
# lengths.
SYNTH_CONTEXT_LENGTHS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)

# OOLONG-synth ``task_group`` values (section 2.2: counting < user < timeline in
# increasing difficulty).
SYNTH_TASK_GROUPS: tuple[str, ...] = ("counting", "user", "timeline")

# Upstream ``answer_type`` -> this module's internal answer-kind vocabulary. The
# internal names drive extraction and scoring; ``label``/``user``/``date``/
# ``month_year``/``comparison`` all score on normalized exact match, ``numeric``
# on the paper's partial-credit curve.
ANSWER_KIND_FROM_SYNTH: dict[str, str] = {
    "ANSWER_TYPE.NUMERIC": "numeric",
    "ANSWER_TYPE.LABEL": "label",
    "ANSWER_TYPE.USER": "user",
    "ANSWER_TYPE.DATE": "date",
    "ANSWER_TYPE.MONTH_YEAR": "month_year",
    "ANSWER_TYPE.COMPARISON": "comparison",
}

SYNTH_ANSWER_KINDS: tuple[str, ...] = tuple(sorted(set(ANSWER_KIND_FROM_SYNTH.values())))
REAL_ANSWER_KINDS: tuple[str, ...] = ("numeric", "string", "list")

# Appended to every OOLONG prompt the loaders build. OOLONG's own question text
# already prescribes a shape ("Give your final answer in the form 'Label:
# answer'", "Answer: number", "User: [X]", "Date: MM/DD/YYYY", "Return a comma
# separated list"); this states the ONE normalizing wrapper the verifier applies
# on top, so a correct answer in a slightly-off shell is graded correct.
ANSWER_FORMAT_CONTRACT = (
    "\n\n---\n"
    "Answer in the exact form the question above asks for, and put that answer on the final "
    "line (optionally prefixed with 'FINAL: '). Grading normalizes formatting first: a number "
    "is read as digits only and a trailing '%' is ignored; a label, user id, comparison "
    "phrase, date, or month/year is matched case-insensitively with surrounding "
    "quotes/brackets/asterisks and any 'Answer:'/'Label:'/'User:'/'Date:' prefix stripped; a "
    "list is read as the set of its comma-separated items. Put only the answer token(s) on "
    "that final line."
)

# The short id ``OolongVerifier.config()`` exposes into the mined bundle's
# ``verifier_config`` (and through it the proposer's pattern block).
EXTRACTION_RULE = (
    "final-line-or-marker;prefixes-stripped;quotes-brackets-asterisks-stripped;"
    "numeric-digits-only-percent-ignored;comparison-phrase-normalized;date-normalized;"
    "list-as-lowercased-set"
)

_MARKER_LINE_RE = re.compile(r"^\s*(?:final|answer|label|user|date)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WRAP_CHARS = "\"'`[]*{}()"
_NONE_MARKER_RE = re.compile(r"^\s*(?:none|n/?a|no\s+\w+|empty|\[\s*\]|\(\s*\))\s*$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Accepted date / month-year input formats, normalized to isoformat / "YYYY-MM".
_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y", "%d/%m/%Y")
_MONTH_YEAR_FORMATS = ("%B %Y", "%b %Y", "%m/%Y", "%Y-%m")

# Comparison phrases collapse to one of three canonical tokens. Both the synth
# COMPARISON answers ("more common than" / "less common than" / "same frequency
# as") and the timeline variant ("more common" / "less common" / "the same
# frequency") map here.
_COMPARISON_CANON = ("more common", "less common", "same frequency")


# =============================================================================
# Answer parsing, gold normalization, scoring (per the dataset card / paper)
# =============================================================================


@dataclass(frozen=True)
class ParsedAnswer:
    """One parsed model answer.

    ``empty`` distinguishes "the final line carried an explicit empty marker"
    (an intentional empty answer -> NO_ANSWER against a non-empty gold) from
    ``None`` at the call site, which means "no answer token at all" ->
    WRONG_FORMAT. Mirrors the None-vs-``[]`` split in
    ``shrlm.environments.graphwalks.extract_answer_nodes``.
    """

    kind: str
    value: Any
    empty: bool = False


def _candidate_line(response: str) -> str | None:
    """The substring to parse an answer out of: the last ``FINAL:``/``Answer:``/
    ``Label:``/``User:``/``Date:`` marked line if any, else the last non-empty
    line. Returns None for an all-whitespace response."""
    lines = [line for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        marker = _MARKER_LINE_RE.match(line)
        if marker is not None:
            return marker.group(1).strip()
    return lines[-1].strip()


def _strip_wrap(text: str) -> str:
    """Strip whitespace and wrapping quote/bracket/asterisk characters, repeatedly."""
    prev = None
    out = text.strip()
    while out != prev:
        prev = out
        out = out.strip().strip(_WRAP_CHARS).strip()
    return out


def _normalize_scalar(text: str, *, casefold: bool) -> str:
    out = _WS_RE.sub(" ", _strip_wrap(text)).strip()
    return out.casefold() if casefold else out


def _normalize_comparison(text: str) -> str:
    lowered = _WS_RE.sub(" ", _strip_wrap(text)).strip().casefold()
    if "more" in lowered and "common" in lowered:
        return "more common"
    if "less" in lowered and "common" in lowered:
        return "less common"
    if "same" in lowered or "no change" in lowered or "equal" in lowered:
        return "same frequency"
    return lowered


def _parse_number(text: str) -> float | int | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _parse_date(text: str) -> str | None:
    cleaned = _strip_wrap(text)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_month_year(text: str) -> str | None:
    cleaned = _strip_wrap(text)
    for fmt in _MONTH_YEAR_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return f"{parsed.year:04d}-{parsed.month:02d}"
        except ValueError:
            continue
    return None


def _parse_list(text: str) -> list[str]:
    parts = re.split(r"[,\n;]", text)
    return sorted(
        {item for item in (_normalize_scalar(part, casefold=True) for part in parts) if item}
    )


def extract_oolong_answer(response: str, answer_kind: str) -> ParsedAnswer | None:
    """Parse a model response into a normalized answer for ``answer_kind``.

    Tolerances are serialization, not reasoning (POST_MORTEM Finding 1): the
    ``FINAL:``/``Answer:`` marker is optional, wrapping quotes/brackets/asterisks
    are stripped, a number is read as digits only with a trailing ``%`` ignored,
    a comparison phrase collapses to one of three canonical tokens, and a date /
    month-year is normalized. Returns None when nothing of the required kind can
    be read (WRONG_FORMAT), and a ``ParsedAnswer(empty=True)`` when the final
    line is an explicit empty marker (NO_ANSWER against a non-empty gold).
    """
    candidate = _candidate_line(response)
    if candidate is None:
        return None
    if answer_kind != "list" and _NONE_MARKER_RE.match(candidate):
        return ParsedAnswer(kind=answer_kind, value=None, empty=True)

    if answer_kind == "numeric":
        number = _parse_number(candidate)
        if number is None:
            return None
        return ParsedAnswer(kind=answer_kind, value=number)
    if answer_kind == "comparison":
        return ParsedAnswer(kind=answer_kind, value=_normalize_comparison(candidate))
    if answer_kind == "date":
        parsed = _parse_date(candidate)
        return ParsedAnswer(kind=answer_kind, value=parsed) if parsed is not None else None
    if answer_kind == "month_year":
        parsed = _parse_month_year(candidate)
        return ParsedAnswer(kind=answer_kind, value=parsed) if parsed is not None else None
    if answer_kind == "list":
        if _NONE_MARKER_RE.match(candidate):
            return ParsedAnswer(kind=answer_kind, value=[], empty=True)
        items = _parse_list(candidate)
        return ParsedAnswer(kind=answer_kind, value=items) if items else None

    # label / user / string: normalized exact match. User ids are digit strings,
    # so casefold is a no-op there but harmless.
    casefold = answer_kind in ("label", "string")
    value = _normalize_scalar(candidate, casefold=casefold)
    if not value:
        return ParsedAnswer(kind=answer_kind, value="", empty=True)
    return ParsedAnswer(kind=answer_kind, value=value)


def _literal_first(answer_raw: str) -> Any:
    """``ast.literal_eval`` a ``"['spam']"`` / ``"[4]"`` gold and take ``[0]``;
    fall back to the raw string. Ported from
    ``training/environments/oolong/oolong/env.py::_synth_score``."""
    text = answer_raw.strip()
    date_match = re.search(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)", text)
    if date_match is not None:
        year, month, day = (int(group) for group in date_match.groups())
        return date(year, month, day)
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def normalize_gold(answer_raw: str, answer_kind: str) -> Any:
    """Normalize a gold answer into the same space ``extract_oolong_answer``
    produces, so scoring compares like with like.

    OOLONG-synth golds are stringified one-element lists (``"['spam']"``,
    ``"[4]"``, ``"[datetime.date(2023, 1, 6)]"``); OOLONG-real golds are bare
    (``"84"``, ``"Attack"``, ``"Acrobatics, Constitution Save"``).
    """
    raw = answer_raw.strip()
    if answer_kind == "list":
        return sorted(set(_parse_list(raw)))

    first = _literal_first(raw)
    if answer_kind == "numeric":
        try:
            value = float(first)
        except (TypeError, ValueError):
            number = _parse_number(str(first))
            return number if number is not None else str(first)
        return int(value) if value.is_integer() else value
    if answer_kind == "date":
        if isinstance(first, date):
            return first.isoformat()
        parsed = _parse_date(str(first))
        return parsed if parsed is not None else str(first).strip()
    if answer_kind == "month_year":
        parsed = _parse_month_year(str(first))
        return parsed if parsed is not None else str(first).strip()
    if answer_kind == "comparison":
        return _normalize_comparison(str(first))
    casefold = answer_kind in ("label", "string")
    return _normalize_scalar(str(first), casefold=casefold)


def _gold_is_empty(gold: Any, answer_kind: str) -> bool:
    if answer_kind == "list":
        return not gold
    return gold in (None, "")


def score_oolong(parsed: ParsedAnswer, gold: Any, answer_kind: str) -> dict[str, Any]:
    """``{"score": float in [0, 1], "exact": bool}`` for one parsed answer.

    numeric -> ``0.75 ** |round(y) - round(y_hat)|`` (paper section 2.3), 0.0
    when ``parsed.value`` is not a number. list -> Jaccard for the score, exact
    set equality for ``exact``. Everything else -> normalized exact match.
    """
    if answer_kind == "numeric":
        if not isinstance(parsed.value, (int, float)):
            return {"score": 0.0, "exact": False}
        try:
            delta = abs(round(float(gold)) - round(float(parsed.value)))
        except (TypeError, ValueError):
            return {"score": 0.0, "exact": False}
        return {"score": 0.75**delta, "exact": delta == 0}
    if answer_kind == "list":
        predicted, golden = set(parsed.value or []), set(gold or [])
        union = predicted | golden
        score = len(predicted & golden) / len(union) if union else 1.0
        return {"score": score, "exact": predicted == golden}
    exact = parsed.value == gold
    return {"score": 1.0 if exact else 0.0, "exact": exact}


def serialize_answer(value: Any, answer_kind: str) -> str:
    """Deterministic rendering for the Verdict's gold/produced fields.

    The Verdict gold feeds the digest sha and the attribution cache key, so an
    order-dependent rendering would split cache entries for byte-identical
    problems -- lists are sorted here for the same reason
    ``graphwalks.serialize_nodes`` sorts.
    """
    if answer_kind == "list":
        items = sorted(value) if isinstance(value, (list, set, tuple)) else [str(value)]
        return "[" + ", ".join(str(item) for item in items) + "]"
    if value is None:
        return ""
    return str(value)


# =============================================================================
# Verifier
# =============================================================================


class OolongVerifier:
    """
    Deterministic outcome for a whole run: full credit (score == 1.0).

    Partial credit is reported in ``detail`` (``score=0.83 exact=False``) the way
    ``GraphWalksVerifier`` reports ``f1=...`` -- the promotion gate compares pass
    counts, so a binary pass keeps that machinery untouched.

    Cause mapping:

    * no answer token on the final line -> WRONG_FORMAT.
    * an explicit empty marker against a non-empty gold -> NO_ANSWER.
    * a ``list`` answer with missing items only -> INCOMPLETE, extra only ->
      SPURIOUS, both -> MIXED_SET_ERROR (the GraphWalks set-cause mapping).
    * every other scored miss -> WRONG_VALUE.
    """

    PASS_SCORE_THRESHOLD: float = 1.0
    GOLD_ORDERING: str = "normalized"

    def __init__(self, task_set: str = "synth") -> None:
        if task_set not in ("synth", "real"):
            raise ValueError(f"task_set must be 'synth' or 'real', got {task_set!r}")
        self.task_set = task_set

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig.verifier_config -> the proposer."""
        kinds = SYNTH_ANSWER_KINDS if self.task_set == "synth" else REAL_ANSWER_KINDS
        return {
            "environment": f"oolong_{self.task_set}",
            "pass_score_threshold": self.PASS_SCORE_THRESHOLD,
            "extraction_rule": EXTRACTION_RULE,
            "answer_kinds": list(kinds),
            "gold_ordering": self.GOLD_ORDERING,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        answer_kind = str(instance["answer_kind"])
        gold = normalize_gold(str(instance["answer_raw"]), answer_kind)
        gold_str = serialize_answer(gold, answer_kind)

        parsed = extract_oolong_answer(produced, answer_kind)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold_str,
                produced=produced,
                detail=f"no {answer_kind} answer token on the final line to parse",
            )
        if parsed.empty and not _gold_is_empty(gold, answer_kind):
            return Verdict(
                passed=False,
                cause=VerifierCause.NO_ANSWER,
                gold=gold_str,
                produced=produced,
                detail="final line carried an explicit empty marker",
            )

        result = score_oolong(parsed, gold, answer_kind)
        produced_str = serialize_answer(parsed.value, answer_kind)
        detail = f"score={result['score']:.3f} exact={result['exact']} kind={answer_kind}"

        if result["exact"] or result["score"] >= self.PASS_SCORE_THRESHOLD:
            return Verdict(
                passed=True, cause=None, gold=gold_str, produced=produced_str, detail=detail
            )

        if answer_kind == "list":
            predicted, golden = set(parsed.value or []), set(gold or [])
            missing, extra = golden - predicted, predicted - golden
            if missing and extra:
                cause = VerifierCause.MIXED_SET_ERROR
            elif missing:
                cause = VerifierCause.INCOMPLETE
            elif extra:
                cause = VerifierCause.SPURIOUS
            else:
                cause = VerifierCause.WRONG_VALUE
        else:
            cause = VerifierCause.WRONG_VALUE
        return Verdict(
            passed=False, cause=cause, gold=gold_str, produced=produced_str, detail=detail
        )


# =============================================================================
# SubVerifier (OOLONG-synth only)
# =============================================================================

# One OOLONG example line as trec/spam-style datasets render it, with the gold
# label suffix present in ``context_window_text_with_labels``.
_LABELED_LINE_RE = re.compile(r"^(?P<body>.*?)\s*\|\|\s*Label:\s*(?P<label>.+?)\s*$")
# The child-prompt asks the SubVerifier recognizes. Everything else is
# uncheckable -> None (never False): a formatting quirk must not flip the failing
# level to CHILD.
_CLASSIFY_RE = re.compile(r"\b(classif|label each|assign (?:a )?label|what (?:is|are) the label)", re.IGNORECASE)
_COUNT_RE = re.compile(
    r"\bhow many\b.*\blabel(?:led|ed)?\s+['\"`]?(?P<label>[\w ]+?)['\"`]?[\s.?]|"
    r"\bcount\b.*\blabel(?:led|ed)?\s+['\"`]?(?P<label2>[\w ]+?)['\"`]?[\s.?]",
    re.IGNORECASE,
)


def _norm_line_body(body: str) -> str:
    """Normalize an example line for use as a gold-label-map key.

    Strips a ``Date: ... || User: ... || Instance:`` wrapper down to the
    instance text when present (trec/spam form), otherwise uses the whole body;
    collapses whitespace either way so the child prompt's copy of the line
    matches regardless of incidental spacing.
    """
    instance_match = re.search(r"\|\|\s*Instance:\s*(?P<text>.+)$", body)
    text = instance_match.group("text") if instance_match is not None else body
    return _WS_RE.sub(" ", text).strip().casefold()


def build_label_map(labels_text: str) -> dict[str, str]:
    """``{normalized example-line text: gold label}`` from ``context_window_text_with_labels``.

    Empty when no line carries a ``|| Label:`` suffix -- the source dataset uses
    a shape this parser does not recognize, so the sub-call is left uncheckable.
    """
    mapping: dict[str, str] = {}
    for line in labels_text.splitlines():
        match = _LABELED_LINE_RE.match(line.strip())
        if match is None:
            continue
        key = _norm_line_body(match.group("body"))
        if key:
            mapping[key] = match.group("label").strip().casefold()
    return mapping


def parse_sub_task(
    prompt: str, label_map: dict[str, str]
) -> tuple[str, list[str], str | None] | None:
    """Best-effort parse of a model-authored child prompt into a groundable
    sub-task over labelled example lines.

    Recognizes exactly two shapes -- "classify these lines" and "how many of
    these lines have label X" -- because those are the sub-tasks whose answer
    can be recomputed from the gold labels alone. Returns
    ``(kind, matched_line_keys, label_or_None)`` or None for anything else
    (uncheckable, never a wrong verdict).
    """
    matched: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = _norm_line_body(stripped)
        if key in label_map:
            matched.append(key)
    if len(matched) < 2:
        return None

    count_match = _COUNT_RE.search(prompt)
    if count_match is not None:
        label = (count_match.group("label") or count_match.group("label2") or "").strip().casefold()
        known = set(label_map.values())
        if label in known:
            return ("count", matched, label)
        return None
    if _CLASSIFY_RE.search(prompt):
        return ("classify", matched, None)
    return None


def _parse_child_labels(response: str) -> list[str] | None:
    """A child's per-line classification: a bracketed/comma list, or one label
    per line. Lowercased. None when nothing list-shaped is present."""
    bracket = re.search(r"\[(.*?)\]", response, re.DOTALL)
    source = bracket.group(1) if bracket is not None else response
    parts = re.split(r"[,\n]", source)
    labels = [_normalize_scalar(part, casefold=True) for part in parts]
    labels = [label for label in labels if label]
    return labels or None


class OolongSubVerifier:
    """
    Post-hoc, deterministic check of one OOLONG-synth sub-call against its own prompt.

    Grounds on the per-example gold labels OOLONG-synth ships in
    ``context_window_text_with_labels``: when the root hands a child a slice of
    example lines and asks it to classify them or count a label within the
    slice, the answer is recomputable from the gold labels and the child's
    result can be graded in isolation. That feeds
    ``shrlm.optimization.grounding.derive_failing_level`` -- CHILD if any
    checkable slice was misclassified, ROOT if every checkable child was
    correct, UNDETERMINED if none were checkable, NO_RECURSION if there were no
    sub-calls at all (the derive_failing_level path is reused unchanged,
    including its zero-descendant grounding fix).

    Returns None -- uncheckable, never False -- for an errored node, a
    non-string prompt/response, a source dataset whose labelled-line shape is
    unrecognized, a child prompt that is not a slice classify/count task, and a
    child response with no parseable per-line labels or count. The body is
    wrapped defensively because a raise here would kill a whole mining round.
    """

    def __call__(self, instance: dict[str, Any], node: CallNode) -> bool | None:
        try:
            return self._check(instance, node)
        except Exception:
            return None

    def _check(self, instance: dict[str, Any], node: CallNode) -> bool | None:
        if node.kind is NodeKind.ERRORED or node.error_kind is not None:
            return None
        if not isinstance(node.prompt, str) or not isinstance(node.response, str):
            return None

        label_map = build_label_map(str(instance.get("labels_text", "")))
        if not label_map:
            return None
        parsed = parse_sub_task(node.prompt, label_map)
        if parsed is None:
            return None
        kind, matched, label = parsed

        if kind == "count":
            expected_count = sum(1 for key in matched if label_map[key] == label)
            child = extract_oolong_answer(node.response, "numeric")
            if child is None or not isinstance(child.value, (int, float)):
                return None
            return int(child.value) == expected_count

        expected_labels = [label_map[key] for key in matched]
        child_labels = _parse_child_labels(node.response)
        if child_labels is None or len(child_labels) != len(expected_labels):
            return None
        return child_labels == expected_labels


def continuous_score(instance: dict[str, Any], produced: str) -> float:
    """The paper's continuous score in [0, 1] for one produced answer.

    ``OolongVerifier`` collapses this to a binary pass at 1.0 so the promotion
    gate keeps comparing pass counts; the OOLONG-real generalization check
    reports the mean of this instead (partial credit is the point of the numeric
    scheme). A response with no parseable answer scores 0.0.
    """
    answer_kind = str(instance["answer_kind"])
    parsed = extract_oolong_answer(produced, answer_kind)
    if parsed is None or parsed.empty:
        return 0.0
    gold = normalize_gold(str(instance["answer_raw"]), answer_kind)
    return float(score_oolong(parsed, gold, answer_kind)["score"])


def make_synth_verifier() -> OolongVerifier:
    """Zero-arg factory for validation child processes (EvaluationConfig.verifier_factory)."""
    return OolongVerifier(task_set="synth")


def make_synth_sub_verifier() -> OolongSubVerifier:
    return OolongSubVerifier()


def make_real_verifier() -> OolongVerifier:
    return OolongVerifier(task_set="real")


# =============================================================================
# Dataset streaming (the module's only network seam)
# =============================================================================

DEFAULT_SCAN_DEADLINE_SECONDS: float = 60.0


def _bounded_iter(
    rows: Iterator[dict[str, Any]], deadline_seconds: float, repo: str
) -> Iterator[dict[str, Any]]:
    """Wrap a streaming iterator so a stalled ``next()`` raises instead of hanging.

    ``datasets`` exposes no per-call timeout for its streaming iterator, so rows
    are pulled on a daemon thread and each read is bounded by a blocking queue
    ``get``. A stall on the first row is caught as reliably as one between rows.
    An exception from ``rows`` is re-raised unchanged in the caller's thread.
    Same rationale as ``shrlm.environments.oolong_pairs._bounded_iter``; kept
    local so this module does not import OOLONG-Pairs.
    """
    result_q: queue.Queue[Any] = queue.Queue(maxsize=1)
    sentinel = object()

    def _pump() -> None:
        try:
            for row in rows:
                result_q.put(row)
            result_q.put(sentinel)
        except Exception as exc:  # noqa: BLE001 -- forwarded to the consumer thread
            result_q.put(exc)

    threading.Thread(target=_pump, daemon=True).start()
    while True:
        try:
            item = result_q.get(timeout=deadline_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"OOLONG dataset stream stalled: no row from {repo} within "
                f"{deadline_seconds}s. Aborting rather than blocking indefinitely."
            ) from exc
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _load_dataset_streaming(repo: str, config_name: str | None, split: str, revision: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for the OOLONG dataset. Install with:\n"
            '    uv pip install -e ".[oolong]"'
        ) from exc
    if config_name is None:
        return load_dataset(repo, split=split, streaming=True, revision=revision)
    return load_dataset(repo, config_name, split=split, streaming=True, revision=revision)


def iter_synth_rows(
    split: str, revision: str | None, deadline_seconds: float = DEFAULT_SCAN_DEADLINE_SECONDS
) -> Iterator[dict[str, Any]]:
    """Stream raw ``oolongbench/oolong-synth`` rows. Monkeypatched in tests."""
    raw = iter(_load_dataset_streaming(SYNTH_DATASET_REPO, None, split, revision))
    return _bounded_iter(raw, deadline_seconds, SYNTH_DATASET_REPO)


def iter_real_rows(
    config_name: str,
    split: str,
    revision: str | None,
    deadline_seconds: float = DEFAULT_SCAN_DEADLINE_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Stream raw ``oolongbench/oolong-real`` rows. Monkeypatched in tests."""
    raw = iter(_load_dataset_streaming(REAL_DATASET_REPO, config_name, split, revision))
    return _bounded_iter(raw, deadline_seconds, REAL_DATASET_REPO)


# =============================================================================
# Length-stratified sampling
# =============================================================================


def _round_robin(items: list[dict[str, Any]], key_of: Any, rng: random.Random) -> list[dict[str, Any]]:
    """Reorder ``items`` so consecutive elements cycle through distinct ``key_of``
    values as far as the pools allow. ``items`` is shuffled first, so the result
    is deterministic for a given ``rng`` state."""
    rng.shuffle(items)
    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(key_of(item), []).append(item)
    order = sorted(groups, key=str)
    out: list[dict[str, Any]] = []
    while any(groups[key] for key in order):
        for key in order:
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def stratified_sample(
    rows: list[dict[str, Any]],
    length_of: Any,
    group_of: Any,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Draw ``limit`` rows spread as evenly as the pools allow across length
    buckets, and within a bucket across ``group_of`` values.

    Length diversity is required, not optional (task requirement 4): the loop
    must see both "decomposition unnecessary" (short) and "decomposition
    necessary" (long) instances in the same pool so the taxonomy learns the
    decision boundary. Deterministic for a given ``(rows, seed)``. Returns
    ``min(limit, len(rows))`` rows.
    """
    rng = random.Random(seed)
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(length_of(row), []).append(row)
    for length in buckets:
        buckets[length] = _round_robin(buckets[length], group_of, rng)

    ordered_lengths = sorted(buckets)
    picked: list[dict[str, Any]] = []
    while len(picked) < limit and any(buckets[length] for length in ordered_lengths):
        for length in ordered_lengths:
            if buckets[length]:
                picked.append(buckets[length].pop(0))
                if len(picked) >= limit:
                    break
    return picked


# =============================================================================
# Instance loading -- OOLONG-synth
# =============================================================================


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return slug or "x"


def _require_length_coverage(
    inventory: dict[int, int], requested: tuple[int, ...], limit: int, repo: str, split: str
) -> None:
    missing = sorted(length for length in requested if inventory.get(length, 0) == 0)
    total = sum(inventory.values())
    if missing or total < limit:
        raise ValueError(
            f"insufficient OOLONG rows from {repo}:{split} for a length-diverse pool of "
            f"{limit}: found {inventory} (rows per requested context length), "
            f"missing lengths {missing}. Widen max_scan, adjust context_lengths, or lower "
            "the split size."
        )


def row_to_synth_instance(
    row: dict[str, Any], sample_seed: int, sample_index: int
) -> dict[str, Any]:
    """Pure transform from an ``oolong-synth`` row to a mining instance.

    The id is content-derived -- task slug, context length, and
    ``sha256(context_window_text + question)[:16]`` -- so the same problem keeps
    the same id across rounds and seeds. ``labels_text`` (the labelled context)
    rides along for ``OolongSubVerifier``; the model-facing ``prompt`` never
    contains it.
    """
    answer_type = str(row["answer_type"])
    if answer_type not in ANSWER_KIND_FROM_SYNTH:
        raise ValueError(
            f"unknown OOLONG-synth answer_type {answer_type!r}; known: "
            f"{sorted(ANSWER_KIND_FROM_SYNTH)}"
        )
    context = str(row["context_window_text"]).rstrip()
    question = str(row["question"]).strip()
    prompt = f"{context}\n\n{question}{ANSWER_FORMAT_CONTRACT}"
    digest = hashlib.sha256(f"{context}␟{question}".encode()).hexdigest()[:16]
    context_len = int(row["context_len"])
    instance_id = f"oolong-synth-{_slug(row['task'])}-{context_len}-{digest}"
    if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(f"derived instance id {instance_id!r} is not filesystem-safe")
    return {
        "id": instance_id,
        "question": question,
        "prompt": prompt,
        "answer_kind": ANSWER_KIND_FROM_SYNTH[answer_type],
        "answer_raw": str(row["answer"]),
        "task_group": str(row["task_group"]),
        "task": str(row["task"]),
        "context_len": context_len,
        "source_dataset": str(row["dataset"]),
        "context_window_id": int(row["context_window_id"]),
        "labels_text": str(row["context_window_text_with_labels"]),
        "sample_seed": sample_seed,
        "sample_index": sample_index,
    }


def load_oolong_synth(
    context_lengths: tuple[int, ...],
    task_groups: tuple[str, ...],
    subsets: tuple[str, ...],
    limit: int,
    seed: int,
    split: str = "test",
    revision: str | None = None,
    max_scan: int = 20_000,
) -> list[dict[str, Any]]:
    """Stream ``oolongbench/oolong-synth``, filter, length-stratify, build ``limit`` instances.

    ``context_lengths`` MUST span both short (model can solve without
    decomposing) and long (aggregation breaks down) -- the selection draws a
    balanced share per length. ``task_groups`` / ``subsets`` empty means "all".
    Deterministic for a given (args, revision): stream order is pinned by
    ``revision`` and only the stratified draw uses ``seed``.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    unknown = sorted(set(context_lengths) - set(SYNTH_CONTEXT_LENGTHS))
    if unknown:
        raise ValueError(
            f"context length(s) {unknown} are not OOLONG-synth power-of-two lengths "
            f"{SYNTH_CONTEXT_LENGTHS}"
        )
    wanted_lengths = set(context_lengths)
    wanted_groups = set(task_groups)
    wanted_subsets = set(subsets)

    collected: list[dict[str, Any]] = []
    inventory: dict[int, int] = dict.fromkeys(context_lengths, 0)
    for index, row in enumerate(iter_synth_rows(split, revision)):
        if index >= max_scan:
            break
        context_len = int(row["context_len"])
        if context_len not in wanted_lengths:
            continue
        if wanted_groups and str(row["task_group"]) not in wanted_groups:
            continue
        if wanted_subsets and str(row["dataset"]) not in wanted_subsets:
            continue
        collected.append(row)
        inventory[context_len] += 1

    _require_length_coverage(inventory, context_lengths, limit, SYNTH_DATASET_REPO, split)
    selected = stratified_sample(
        collected,
        length_of=lambda row: int(row["context_len"]),
        group_of=lambda row: str(row["task_group"]),
        limit=limit,
        seed=seed,
    )
    return [
        row_to_synth_instance(row, sample_seed=seed, sample_index=index)
        for index, row in enumerate(selected)
    ]


# =============================================================================
# Instance loading -- OOLONG-real
# =============================================================================

_LIST_QUESTION_RE = re.compile(r"comma[- ]separated list|return a list|as a list", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def infer_real_answer_kind(question: str, answer_raw: str) -> str:
    """numeric / string / list for an OOLONG-real row, from the question wording
    and the gold shape (there is no ``answer_type`` field on the real split)."""
    if _LIST_QUESTION_RE.search(question):
        return "list"
    if _INT_RE.match(answer_raw.strip()):
        return "numeric"
    return "string"


def row_to_real_instance(
    row: dict[str, Any], sample_seed: int, sample_index: int
) -> dict[str, Any]:
    """Pure transform from an ``oolong-real`` row to a generalization-check instance.

    ``context_window_text`` is already a self-contained prompt; only the answer
    contract is appended. id is content-derived from the built prompt.
    """
    context = str(row["context_window_text"]).rstrip()
    question = str(row["question"]).strip()
    prompt = f"{context}\n\n{question}{ANSWER_FORMAT_CONTRACT}"
    answer_raw = str(row["answer"])
    answer_kind = infer_real_answer_kind(question, answer_raw)
    digest = hashlib.sha256(f"{context}␟{question}".encode()).hexdigest()[:16]
    episodes = [int(episode) for episode in row.get("episodes", [])]
    instance_id = f"oolong-real-{_slug(row['question_type'])}-{digest}"
    if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(f"derived instance id {instance_id!r} is not filesystem-safe")
    return {
        "id": instance_id,
        "question": question,
        "prompt": prompt,
        "answer_kind": answer_kind,
        "answer_raw": answer_raw,
        "question_type": str(row["question_type"]),
        "n_episodes": len(episodes),
        "episodes": episodes,
        "context_window_id": str(row["context_window_id"]),
        "sample_seed": sample_seed,
        "sample_index": sample_index,
    }


def load_oolong_real(
    episode_counts: tuple[int, ...],
    question_types: tuple[str, ...],
    limit: int,
    seed: int,
    split: str = "test",
    revision: str | None = None,
    config_name: str = "dnd",
    max_scan: int = 20_000,
) -> list[dict[str, Any]]:
    """Stream ``oolongbench/oolong-real``, filter, stratify by episode count, build ``limit``.

    The generalization check wants length diversity too (task requirement 4):
    ``episode_counts`` empty means "whatever the stream carries", otherwise the
    draw is spread across the requested transcript counts. Never feeds the
    promotion gate.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    wanted_counts = set(episode_counts)
    wanted_types = set(question_types)

    collected: list[dict[str, Any]] = []
    inventory: dict[int, int] = {}
    for index, row in enumerate(iter_real_rows(config_name, split, revision)):
        if index >= max_scan:
            break
        n_episodes = len(row.get("episodes", []) or [])
        if wanted_counts and n_episodes not in wanted_counts:
            continue
        if wanted_types and str(row["question_type"]) not in wanted_types:
            continue
        collected.append(row)
        inventory[n_episodes] = inventory.get(n_episodes, 0) + 1

    if len(collected) < limit:
        raise ValueError(
            f"insufficient OOLONG-real rows from {REAL_DATASET_REPO}:{config_name}:{split} "
            f"for a check set of {limit}: found {inventory} (rows per episode count). Widen "
            "max_scan, adjust episode_counts, or lower n_check."
        )
    selected = stratified_sample(
        collected,
        length_of=lambda row: len(row.get("episodes", []) or []),
        group_of=lambda row: str(row["question_type"]),
        limit=limit,
        seed=seed,
    )
    return [
        row_to_real_instance(row, sample_seed=seed, sample_index=index)
        for index, row in enumerate(selected)
    ]


# =============================================================================
# Config wiring
# =============================================================================


def load_oolong_synth_from_config(
    config: Any,
    n: int,
    seed: int,
    context_lengths: tuple[int, ...] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """``load_oolong_synth`` with repo/pin/scan facts taken from the experiment config."""
    synth = config.environments.oolong.synth
    return load_oolong_synth(
        context_lengths=context_lengths
        if context_lengths is not None
        else tuple(synth.context_lengths),
        task_groups=tuple(synth.task_groups),
        subsets=tuple(synth.subsets),
        limit=n,
        seed=seed,
        split=split if split is not None else synth.split,
        revision=synth.dataset_revision,
        max_scan=synth.max_scan,
    )


def load_oolong_real_from_config(
    config: Any,
    n: int,
    seed: int,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """``load_oolong_real`` with repo/pin/scan facts taken from the experiment config."""
    real = config.environments.oolong.real
    return load_oolong_real(
        episode_counts=tuple(real.episode_counts),
        question_types=tuple(real.question_types),
        limit=n,
        seed=seed,
        split=split if split is not None else real.split,
        revision=real.dataset_revision,
        config_name=real.config_name,
        max_scan=real.max_scan,
    )
