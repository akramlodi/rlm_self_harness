"""
RULER environment: instances, a mining-side Verifier, and a SubVerifier.

RULER (NVIDIA, https://arxiv.org/abs/2404.06654, https://github.com/NVIDIA/RULER)
is a synthetic long-context suite. This module implements 3 of its 4 official
task families, fully in-process (stdlib-only: ``random``/``hashlib``/``re``,
no network, no download):

* **Retrieval (NIAH)**: single-needle (kept only for harness/test sanity --
  excluded from ``DEFAULT_TASK_TYPES`` because frontier models saturate it
  near 100% even at long lengths, which is exactly why vanilla RULER is not
  hard enough), multi-key (one queried key among several hard, same-shaped
  distractor keys), multi-value (many distinct values recorded for the SAME
  key -- a list answer), and multi-query (several keys queried at once,
  answered as ``key=value`` pairs rather than bare values, so a response
  cannot get credit for the right value set paired with the wrong keys).
* **Multi-hop tracing**: Variable Tracking (VT) -- a chain of
  ``VAR = VAR`` assignments; the answer is every variable name transitively
  bound to a target value, scattered among distractor chains bound to other
  values.
* **Aggregation**: Common Words Extraction (CWE, top-10 by count) and
  Frequent Words Extraction (FWE, top-3 by count over a Zipf-weighted draw).

RULER's official QA family (SQuAD/HotpotQA-backed) is deliberately OUT of
scope: it needs external downloads and doesn't add decomposition-forcing
value the other three families don't already provide, at a cost the project
does not want to pay for a mining-loop environment.

Answer format and Verdict/VerifierCause mapping follow the same convention
every other environment in this repo uses (``graphwalks.py``,
``oolong.py``, ``oolong_pairs.py``): a trailing answer line, an optional
``FINAL:`` marker, and a strict distinction between "no candidate line at
all" (``None`` -> WRONG_FORMAT) and "an explicit empty/none marker"
(``ParsedAnswer(empty=True)`` -> NO_ANSWER against a non-empty gold).

Sub-verification (the highest-leverage part of this environment): a
``CallNode`` carries only opaque ``prompt``/``response`` text, never
structural offset metadata connecting a child's prompt back to a slice of
the parent's context (confirmed against ``shrlm.optimization.types.CallNode``
and every existing SubVerifier in this repo, which all ground by re-parsing
a child's own prompt text). Because RULER is fully synthetic, the generator
retains complete ground truth about every literal string in the haystack
(``tracked_units``, the RULER analogue of ``oolong``'s hidden ``labels_text``
map), so instead of GraphWalks'/OOLONG's best-effort regex-recovery of an
organically-worded sub-task, this environment follows the (unmerged)
``mrcrv2.py`` precedent and INJECTS an explicit machine-parseable self-report
contract (``RULER_SUBCALL_CONTRACT``) into every prompt: any sub-call the
root delegates must end its own response with one ``LOCAL FINDING: <item> =
<value>`` line per item it was asked to check. ``RulerSubVerifier`` then
grounds each claim by (1) determining what is actually present in the
child's OWN prompt via literal-substring matching against ``tracked_units``,
and (2) comparing that recomputed ground truth to the child's self-reported
claim. One unified contract and one unified SubVerifier cover all seven task
types via a per-``task_type`` ground-truth resolver, since the substring-
matching mechanism itself is identical across families.

Resource exceptions never reach the Verifier: its signature receives only a
produced string, and RESOURCE_TERMINATED is owned by the experiment driver.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict

# =============================================================================
# Task types
# =============================================================================

TASK_TYPES: tuple[str, ...] = (
    "niah_single",
    "niah_multikey",
    "niah_multivalue",
    "niah_multiquery",
    "variable_tracking",
    "common_words_extraction",
    "frequent_words_extraction",
)

# niah_single deliberately excluded: this is the shape frontier models
# saturate near 100% even at long lengths (the reason vanilla RULER is "not
# hard enough"). It stays implemented for harness/test sanity-checking only.
DEFAULT_TASK_TYPES: tuple[str, ...] = tuple(t for t in TASK_TYPES if t != "niah_single")

ANSWER_KIND_FOR_TASK: dict[str, str] = {
    "niah_single": "scalar",
    "niah_multikey": "scalar",
    "niah_multivalue": "list",
    "niah_multiquery": "list",
    "variable_tracking": "list",
    "common_words_extraction": "list",
    "frequent_words_extraction": "list",
}

# =============================================================================
# Shared vocabulary and haystack building
# =============================================================================

_WORD_POOL: tuple[str, ...] = (
    "harbor",
    "cinder",
    "willow",
    "granite",
    "meadow",
    "compass",
    "ember",
    "thistle",
    "canyon",
    "prairie",
    "obsidian",
    "sable",
    "brine",
    "ridge",
    "orchard",
    "tundra",
    "copper",
    "basalt",
    "clover",
    "delta",
    "fern",
    "glacier",
    "horizon",
    "ivory",
    "jasper",
    "kestrel",
    "lichen",
    "marrow",
    "nectar",
    "opal",
    "pebble",
    "quill",
    "raven",
    "sienna",
    "talon",
    "umber",
    "violet",
    "wren",
    "yarrow",
    "zephyr",
    "anchor",
    "bramble",
    "cascade",
    "driftwood",
    "estuary",
    "foxglove",
    "gable",
    "hollow",
    "isthmus",
    "juniper",
)

_FILLER_TEMPLATES: tuple[str, ...] = (
    "A brief note about the {a} near the {b} was recorded on this page.",
    "Someone once observed that the {a} resembled a distant {b}.",
    "The committee filed a report on the {a} shortly after the {b}.",
    "Nothing of consequence happened near the {a} or the {b} that day.",
    "A traveler once described the {a} as quieter than the {b}.",
    "Records from that period rarely mention the {a} or the {b} at all.",
)


def _filler_sentence(rng: random.Random) -> str:
    template = rng.choice(_FILLER_TEMPLATES)
    a, b = rng.sample(_WORD_POOL, k=2)
    return template.format(a=a, b=b)


def _sample_distinct_words(rng: random.Random, n: int) -> list[str]:
    """``n`` distinct word-like tokens, extending the bare pool with numeric
    suffixes once it is exhausted (never raises for ``n`` beyond the pool)."""
    if n <= len(_WORD_POOL):
        return rng.sample(_WORD_POOL, k=n)
    words = list(_WORD_POOL)
    rng.shuffle(words)
    extra = [f"{rng.choice(_WORD_POOL)}{index}" for index in range(n - len(_WORD_POOL))]
    return (words + extra)[:n]


def _sample_keys(rng: random.Random, n: int) -> list[str]:
    """``n`` distinct key phrases (a word plus a random numeric suffix, so
    keys stay distinct even across many instances sharing the same seed)."""
    seen: set[str] = set()
    keys: list[str] = []
    while len(keys) < n:
        key = f"{rng.choice(_WORD_POOL)}-{rng.randrange(1_000, 9_999)}"
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _scatter(rng: random.Random, filler: list[str], mandatory: list[str]) -> list[str]:
    """Interleave ``mandatory`` lines roughly evenly across a shuffled copy of
    ``filler``, so they land scattered through the haystack instead of
    clustered at one end."""
    if not mandatory:
        return filler
    pool = list(filler)
    rng.shuffle(pool)
    chunk = max(1, len(pool) // (len(mandatory) + 1))
    out: list[str] = []
    remaining = list(mandatory)
    next_insert = chunk
    for index, line in enumerate(pool):
        out.append(line)
        if remaining and index + 1 == next_insert:
            out.append(remaining.pop(0))
            next_insert += chunk
    out.extend(remaining)
    return out


def _build_haystack(rng: random.Random, target_chars: int, mandatory_lines: list[str]) -> str:
    """Generic filler padded to roughly ``target_chars``, with
    ``mandatory_lines`` (needles / assignments) scattered through it. Used by
    the sentence-based families (NIAH, Variable Tracking); CWE/FWE build a
    flat word stream instead (see ``_gen_cwe``/``_gen_fwe``)."""
    mandatory_chars = sum(len(line) + 1 for line in mandatory_lines)
    filler: list[str] = []
    filler_chars = 0
    budget = max(0, target_chars - mandatory_chars)
    while filler_chars < budget:
        line = _filler_sentence(rng)
        filler.append(line)
        filler_chars += len(line) + 1
    lines = _scatter(rng, filler, mandatory_lines)
    return "\n".join(lines)


# =============================================================================
# Retrieval (NIAH)
# =============================================================================


def _needle_line(key: str, value: str) -> str:
    return f"The special magic number for {key} is: {value}."


def _sample_unique_values(rng: random.Random, n: int) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    while len(values) < n:
        value = str(rng.randint(100_000, 999_999))
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _gen_niah(
    rng: random.Random,
    n_queried_keys: int,
    n_distractor_keys: int,
    values_per_queried_key: int,
) -> tuple[list[str], list[dict[str, str]], dict[str, list[str]], list[str]]:
    """Shared NIAH generator: ``n_queried_keys`` keys each get
    ``values_per_queried_key`` needle(s); ``n_distractor_keys`` extra keys get
    exactly one needle each as pure noise. Returns ``(mandatory_lines,
    tracked_units, gold_by_key, queried_keys)``."""
    all_keys = _sample_keys(rng, n_queried_keys + n_distractor_keys)
    queried_keys, distractor_keys = all_keys[:n_queried_keys], all_keys[n_queried_keys:]

    mandatory_lines: list[str] = []
    tracked_units: list[dict[str, str]] = []
    gold_by_key: dict[str, list[str]] = {}

    for key in queried_keys:
        values = _sample_unique_values(rng, values_per_queried_key)
        gold_by_key[key] = values
        for value in values:
            line = _needle_line(key, value)
            mandatory_lines.append(line)
            tracked_units.append({"item": key, "literal_text": line, "value": value})

    for key in distractor_keys:
        value = _sample_unique_values(rng, 1)[0]
        line = _needle_line(key, value)
        mandatory_lines.append(line)
        tracked_units.append({"item": key, "literal_text": line, "value": value})

    rng.shuffle(mandatory_lines)
    return mandatory_lines, tracked_units, gold_by_key, queried_keys


def _gen_niah_single(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    mandatory, tracked, gold_by_key, queried = _gen_niah(rng, 1, 0, 1)
    body = _build_haystack(rng, target_chars, mandatory)
    key = queried[0]
    question = f'What is the special magic number for "{key}"? Report only the number.'
    return body, question, gold_by_key[key][0], "scalar", tracked


def _gen_niah_multikey(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    n_distractor_keys: int | None = None,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_distractor_keys is None:
        n_distractor_keys = max(1, round(8 * distractor_density_multiplier))
    mandatory, tracked, gold_by_key, queried = _gen_niah(rng, 1, n_distractor_keys, 1)
    body = _build_haystack(rng, target_chars, mandatory)
    key = queried[0]
    question = (
        f'What is the special magic number for "{key}"? The text above records several '
        "other similar-looking magic numbers for OTHER keys -- report only the number for "
        "exactly this key. Report only the number."
    )
    return body, question, gold_by_key[key][0], "scalar", tracked


def _gen_niah_multivalue(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    n_values_per_key: int = 4,
    n_distractor_keys: int | None = None,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_distractor_keys is None:
        n_distractor_keys = max(0, round(4 * distractor_density_multiplier))
    mandatory, tracked, gold_by_key, queried = _gen_niah(
        rng, 1, n_distractor_keys, n_values_per_key
    )
    body = _build_haystack(rng, target_chars, mandatory)
    key = queried[0]
    question = (
        f'The text above records the special magic number for "{key}" multiple times, each '
        f'time with a different value. List ALL distinct magic numbers recorded for "{key}".'
    )
    return body, question, sorted(gold_by_key[key]), "list", tracked


def _gen_niah_multiquery(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    n_queried_keys: int = 4,
    n_distractor_keys: int | None = None,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_distractor_keys is None:
        n_distractor_keys = max(0, round(8 * distractor_density_multiplier))
    mandatory, tracked, gold_by_key, queried = _gen_niah(rng, n_queried_keys, n_distractor_keys, 1)
    body = _build_haystack(rng, target_chars, mandatory)
    keys_listed = ", ".join(f'"{key}"' for key in queried)
    question = (
        "For EACH of the following keys, report its special magic number from the text "
        f"above, formatted as key=value, one per line or comma-separated: {keys_listed}."
    )
    gold = sorted(f"{key}={gold_by_key[key][0]}" for key in queried)
    return body, question, gold, "list", tracked


# =============================================================================
# Multi-hop tracing (Variable Tracking)
# =============================================================================


def _build_chain(chain_index: int, length: int, value: str) -> tuple[list[str], list[str]]:
    names = [f"VAR_{chain_index}_{step}" for step in range(length)]
    lines = [f"{names[0]} = {value}"]
    lines.extend(f"{names[step]} = {names[step - 1]}" for step in range(1, length))
    return names, lines


def _gen_variable_tracking(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    chain_length: int = 4,
    n_distractor_chains: int | None = None,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_distractor_chains is None:
        n_distractor_chains = max(1, round(3 * distractor_density_multiplier))

    target_value = str(rng.randint(100_000, 999_999))
    target_names, target_lines = _build_chain(0, chain_length, target_value)
    mandatory_lines = list(target_lines)
    tracked_units = [
        {"item": name, "literal_text": line, "value": ""}
        for name, line in zip(target_names, target_lines, strict=True)
    ]

    used_values = {target_value}
    for chain_index in range(1, n_distractor_chains + 1):
        distractor_value = str(rng.randint(100_000, 999_999))
        while distractor_value in used_values:
            distractor_value = str(rng.randint(100_000, 999_999))
        used_values.add(distractor_value)
        names, lines = _build_chain(chain_index, chain_length, distractor_value)
        mandatory_lines.extend(lines)
        tracked_units.extend(
            {"item": name, "literal_text": line, "value": ""}
            for name, line in zip(names, lines, strict=True)
        )

    rng.shuffle(mandatory_lines)
    body = _build_haystack(rng, target_chars, mandatory_lines)
    question = (
        f"One variable above was directly assigned the value {target_value}, and other "
        'variables were transitively assigned that same value through a chain of "VAR = '
        'VAR" assignments. List ALL variable names (in any order, comma-separated) whose '
        f"value equals {target_value}."
    )
    return body, question, sorted(target_names), "list", tracked_units


# =============================================================================
# Aggregation (CWE / FWE)
# =============================================================================


def _gen_cwe(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    n_top: int = 10,
    n_noise: int | None = None,
    avg_chars_per_occurrence: int = 8,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_noise is None:
        n_noise = max(n_top, round(30 * distractor_density_multiplier))

    words = _sample_distinct_words(rng, n_top + n_noise)
    top_words, noise_words = words[:n_top], words[n_top:]
    total_occurrences = max(n_top + n_noise, target_chars // avg_chars_per_occurrence)

    # Noise words all get the SAME count (a fixed, even share of half the
    # budget); top words get a strictly higher, strictly decreasing count
    # derived from double the noise share plus a growing increment -- this
    # guarantees, by construction and regardless of ``target_chars`` scale,
    # that every top word outranks every noise word: no probabilistic top-10
    # tie is possible.
    noise_share = max(1, (total_occurrences // 2) // n_noise)
    top_floor = noise_share * 2
    top_remaining = max(0, total_occurrences - noise_share * n_noise)
    top_counts = [top_floor + (top_remaining // n_top) + (n_top - index) for index in range(n_top)]

    occurrences: list[str] = []
    for word, count in zip(top_words, top_counts, strict=True):
        occurrences.extend([word] * count)
    for word in noise_words:
        occurrences.extend([word] * noise_share)
    rng.shuffle(occurrences)

    body = ", ".join(occurrences)
    question = (
        "Above is a long list of words separated by commas. List the 10 words that occur "
        "most frequently in that list, in any order, comma-separated."
    )
    gold = sorted(top_words)
    tracked_units = [{"item": word, "literal_text": word, "value": ""} for word in words]
    return body, question, gold, "list", tracked_units


def _gen_fwe(
    rng: random.Random,
    target_chars: int,
    *,
    distractor_density_multiplier: float = 1.0,
    n_top: int = 3,
    n_distinct_words: int | None = None,
    zipf_exponent: float = 1.3,
    avg_chars_per_occurrence: int = 8,
    **_ignored: Any,
) -> tuple[str, str, Any, str, list[dict[str, str]]]:
    if n_distinct_words is None:
        n_distinct_words = max(n_top + 5, round(20 * distractor_density_multiplier))

    words = _sample_distinct_words(rng, n_distinct_words)
    weights = [1.0 / (rank**zipf_exponent) for rank in range(1, n_distinct_words + 1)]
    n_occurrences = max(n_distinct_words * 3, target_chars // avg_chars_per_occurrence)
    occurrences = rng.choices(words, weights=weights, k=n_occurrences)

    # Gold is whatever the realized draw actually made most frequent -- not
    # the theoretical Zipf rank -- so grading is a tautology against the
    # generated text rather than a probabilistic approximation of it.
    counts = Counter(occurrences)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    gold = sorted(word for word, _ in ranked[:n_top])

    shuffled = list(occurrences)
    rng.shuffle(shuffled)
    body = ", ".join(shuffled)
    question = (
        "Above is a long list of words separated by commas, drawn from a small vocabulary "
        "with skewed frequency. List the 3 words that occur most frequently in that list, "
        "in any order, comma-separated."
    )
    tracked_units = [{"item": word, "literal_text": word, "value": ""} for word in words]
    return body, question, gold, "list", tracked_units


_TASK_GENERATORS: dict[str, Callable[..., tuple[str, str, Any, str, list[dict[str, str]]]]] = {
    "niah_single": _gen_niah_single,
    "niah_multikey": _gen_niah_multikey,
    "niah_multivalue": _gen_niah_multivalue,
    "niah_multiquery": _gen_niah_multiquery,
    "variable_tracking": _gen_variable_tracking,
    "common_words_extraction": _gen_cwe,
    "frequent_words_extraction": _gen_fwe,
}


# =============================================================================
# Answer contract
# =============================================================================

ANSWER_FORMAT_CONTRACT = (
    "\n\n---\n"
    "Answer the question above. If the answer is a single item, put it alone on the final "
    "line (optionally prefixed with 'FINAL: '). If the answer is a list of items, put them "
    "on the final line as a comma-separated list (optionally prefixed with 'FINAL: '). If "
    "there is truly no valid answer, put 'NONE' on the final line."
)

# The sub-call grounding contract (this environment's own addition, not a
# shared harness surface): if the root delegates a slice of the text to a
# sub-call, that sub-call must report a checkable local claim. See the module
# docstring for why this is injected rather than best-effort parsed.
RULER_SUBCALL_CONTRACT = (
    "\n\n---\n"
    "If you delegate any part of the text above to a sub-call (llm_query/rlm_query), tell "
    "it EXACTLY which item(s) to look for in its slice (a key name, a variable name, or a "
    "target word), and instruct it to end its own response with one line per item, in the "
    "form:\n"
    "  LOCAL FINDING: <item> = <value-or-count-found-in-this-slice>\n"
    "or, if the item does not appear anywhere in its slice:\n"
    "  LOCAL FINDING: <item> = NONE\n"
    "where <item> is copied verbatim from the name/key/word you asked it to check, and the "
    "value/count is read ONLY from what is literally present in the slice you gave it, not "
    "the whole text above."
)

_FINAL_LINE_RE = re.compile(r"^\s*(?:final)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
_NONE_MARKER_RE = re.compile(r"^\s*(?:none|n/?a|no\s+\w+|empty)\s*$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_WRAP_CHARS = "\"'`[]*{}()"


def _candidate_line(response: str) -> str | None:
    """The substring to parse an answer out of: the last ``FINAL:``-marked
    line if any, else the last non-empty line. None for an all-whitespace
    response. Mirrors ``oolong._candidate_line``/``mrcrv2._candidate_line``."""
    lines = [line for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        marker = _FINAL_LINE_RE.match(line)
        if marker is not None:
            return marker.group(1).strip()
    return lines[-1].strip()


def _strip_wrap(text: str) -> str:
    prev = None
    out = text.strip()
    while out != prev:
        prev = out
        out = out.strip().strip(_WRAP_CHARS).strip()
    return out


@dataclass(frozen=True)
class ParsedAnswer:
    """One parsed model answer. ``empty`` distinguishes "the final line
    carried an explicit empty/none marker" (NO_ANSWER against a non-empty
    gold) from ``None`` at the call site, which means "no answer token at
    all" (WRONG_FORMAT). Mirrors ``oolong.ParsedAnswer``."""

    value: Any
    empty: bool = False


def extract_ruler_answer(response: str, answer_kind: str) -> ParsedAnswer | None:
    """Parse a model response into a normalized answer for ``answer_kind``
    ("scalar" or "list"). Returns None when nothing parseable is on the
    candidate line (WRONG_FORMAT), and ``ParsedAnswer(empty=True)`` for an
    explicit none/empty marker (NO_ANSWER against a non-empty gold)."""
    candidate = _candidate_line(response)
    if candidate is None:
        return None
    if _NONE_MARKER_RE.match(candidate):
        return ParsedAnswer(value=[] if answer_kind == "list" else "", empty=True)

    if answer_kind == "list":
        items = [_strip_wrap(item) for item in re.split(r"[,\n;]", candidate)]
        items = [item for item in items if item]
        if not items:
            return ParsedAnswer(value=[], empty=True)
        return ParsedAnswer(value=sorted(set(items)))

    value = _strip_wrap(candidate)
    if not value:
        return ParsedAnswer(value="", empty=True)
    return ParsedAnswer(value=value)


def serialize_gold(value: Any, answer_kind: str) -> str:
    """Render the Verdict's gold/produced fields SORTED, so equal answers
    always serialize identically (the digest-sha/attribution-cache reason
    ``graphwalks.serialize_nodes`` documents)."""
    if answer_kind == "list":
        items = sorted(value) if isinstance(value, (list, set, tuple)) else [str(value)]
        return "[" + ", ".join(str(item) for item in items) + "]"
    return "" if value in (None, "") else str(value)


# =============================================================================
# Dataset generation
# =============================================================================


def generate_ruler_instance(
    task_type: str,
    target_tokens: int,
    seed: int,
    index: int,
    *,
    chars_per_token: float = 4.0,
    distractor_density_multiplier: float = 1.0,
    **family_knobs: Any,
) -> dict[str, Any]:
    """Generate one RULER instance.

    Deterministic for a given ``(task_type, target_tokens, seed, index)``:
    the per-instance RNG is seeded from all four, so two calls with the same
    arguments produce byte-identical output and different ``index`` values
    produce different (but still reproducible) instances from one ``seed``.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown RULER task_type {task_type!r}; expected one of {TASK_TYPES}")
    if target_tokens < 1:
        raise ValueError(f"target_tokens must be >= 1, got {target_tokens}")

    rng = random.Random(f"{seed}:{task_type}:{index}:{target_tokens}")
    target_chars = int(target_tokens * chars_per_token)
    generator = _TASK_GENERATORS[task_type]
    body, question, gold, answer_kind, tracked_units = generator(
        rng,
        target_chars,
        distractor_density_multiplier=distractor_density_multiplier,
        **family_knobs,
    )

    prompt = body + "\n\n---\n" + question + ANSWER_FORMAT_CONTRACT + RULER_SUBCALL_CONTRACT
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    instance_id = f"ruler-{task_type}-{target_tokens}-{digest}"
    if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(f"derived instance id {instance_id!r} is not filesystem-safe")

    return {
        "id": instance_id,
        "question": question,
        "prompt": prompt,
        "task_type": task_type,
        "answer_kind": answer_kind,
        "gold": gold,
        "tracked_units": tracked_units,
        "target_tokens": target_tokens,
        "prompt_chars": len(prompt),
        "sample_seed": seed,
        "sample_index": index,
    }


def generate_ruler_instances(
    task_types: tuple[str, ...],
    target_tokens: int,
    limit: int,
    seed: int,
    **generator_kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ``limit`` RULER instances, balanced across ``task_types``
    (per-type share, remainder topped up with a seeded shuffle). Deterministic
    for a given ``(task_types, target_tokens, limit, seed, generator_kwargs)``;
    ids are content-derived and unique since each task type draws from its
    own ``index`` namespace."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not task_types:
        raise ValueError("task_types must not be empty")
    unknown = sorted(set(task_types) - set(TASK_TYPES))
    if unknown:
        raise ValueError(f"unknown RULER task_type(s) {unknown}; expected one of {TASK_TYPES}")

    rng = random.Random(seed)
    per_type = limit // len(task_types)
    counts = dict.fromkeys(task_types, per_type)
    remainder = limit - per_type * len(task_types)
    if remainder:
        topped_up = list(task_types)
        rng.shuffle(topped_up)
        for task_type in topped_up[:remainder]:
            counts[task_type] += 1

    instances = [
        generate_ruler_instance(task_type, target_tokens, seed, index, **generator_kwargs)
        for task_type in task_types
        for index in range(counts[task_type])
    ]
    rng.shuffle(instances)
    return instances


def load_ruler_from_config(
    config: Any,
    n: int,
    seed: int,
    length: str = "short",
) -> list[dict[str, Any]]:
    """``generate_ruler_instances`` with task-type/target-tokens facts taken
    from the experiment config, keyed by split length ("short" or "long")."""
    env = config.environments.ruler
    if length == "short":
        target_tokens = env.short_target_tokens
    elif length == "long":
        target_tokens = env.long_target_tokens
    else:
        raise ValueError(f"unknown split length {length!r}; expected 'short' or 'long'")
    return generate_ruler_instances(
        task_types=tuple(env.task_types),
        target_tokens=target_tokens,
        limit=n,
        seed=seed,
        chars_per_token=env.chars_per_token,
        distractor_density_multiplier=env.distractor_density_multiplier,
    )


# =============================================================================
# Verifier
# =============================================================================


class RulerVerifier:
    """
    Deterministic outcome for a whole run: exact-match (set equality for a
    list answer, string equality for a scalar answer).

    Deliberately NOT RULER's own official recall-only metric: ``Verdict.passed``
    feeds cross-condition pass-count comparisons in the promotion gate, and
    every existing verifier in this repo (including ``OolongVerifier``, whose
    underlying score is continuous) collapses to a binary pass at a threshold
    for exactly that reason. This sacrifices literal comparability to
    published RULER leaderboard numbers, an acceptable tradeoff since the
    goal here is a mining-loop target, not benchmark reproduction.

    Cause mapping:

    * no candidate answer line at all -> WRONG_FORMAT.
    * an explicit empty/none marker against a non-empty gold -> NO_ANSWER.
    * a ``list`` answer with missing items only -> INCOMPLETE, extra only ->
      SPURIOUS, both -> MIXED_SET_ERROR (the GraphWalks/OOLONG set-cause
      mapping).
    * a ``scalar`` answer that mismatches -> WRONG_VALUE.
    """

    PASS_THRESHOLD: float = 1.0
    EXTRACTION_RULE: str = "final-line-or-marker;list-as-comma-separated-set;none-marker-explicit"
    GOLD_ORDERING: str = "sorted"

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig by the experiment driver."""
        return {
            "environment": "ruler",
            "pass_threshold": self.PASS_THRESHOLD,
            "extraction_rule": self.EXTRACTION_RULE,
            "gold_ordering": self.GOLD_ORDERING,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        answer_kind = str(instance["answer_kind"])
        gold = instance["gold"]
        gold_str = serialize_gold(gold, answer_kind)

        parsed = extract_ruler_answer(produced, answer_kind)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold_str,
                produced=produced,
                detail="no candidate answer line to parse",
            )

        gold_is_empty = (not gold) if answer_kind == "list" else gold in (None, "")
        if parsed.empty and not gold_is_empty:
            return Verdict(
                passed=False,
                cause=VerifierCause.NO_ANSWER,
                gold=gold_str,
                produced=produced,
                detail="final line carried an explicit empty/none marker",
            )

        if answer_kind == "list":
            pred_set, gold_set = set(parsed.value or []), set(gold or [])
            produced_str = serialize_gold(sorted(pred_set), "list")
            missing, extra = gold_set - pred_set, pred_set - gold_set
            detail = f"missing={len(missing)} extra={len(extra)}"
            if not missing and not extra:
                return Verdict(
                    passed=True, cause=None, gold=gold_str, produced=produced_str, detail=detail
                )
            cause = (
                VerifierCause.MIXED_SET_ERROR
                if missing and extra
                else VerifierCause.INCOMPLETE
                if missing
                else VerifierCause.SPURIOUS
            )
            return Verdict(
                passed=False, cause=cause, gold=gold_str, produced=produced_str, detail=detail
            )

        produced_str = str(parsed.value)
        if produced_str == str(gold):
            return Verdict(
                passed=True, cause=None, gold=gold_str, produced=produced_str, detail="exact match"
            )
        return Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold=gold_str,
            produced=produced_str,
            detail=f"expected {gold_str}",
        )


def make_ruler_verifier() -> RulerVerifier:
    """Zero-arg factory for validation child processes (EvaluationConfig.verifier_factory)."""
    return RulerVerifier()


# =============================================================================
# SubVerifier
# =============================================================================

_LOCAL_FINDING_RE = re.compile(
    r"LOCAL FINDING:\s*(?P<item>.+?)\s*=\s*(?P<value>.+?)\s*$", re.MULTILINE
)


def parse_local_findings(response: str) -> dict[str, str]:
    """Every ``LOCAL FINDING: <item> = <value>`` line in a sub-call's
    response, first occurrence per item wins."""
    findings: dict[str, str] = {}
    for match in _LOCAL_FINDING_RE.finditer(response):
        item = match.group("item").strip()
        if item and item not in findings:
            findings[item] = match.group("value").strip()
    return findings


def _normalize_local_value(value: str) -> str:
    return _WS_RE.sub(" ", _strip_wrap(value)).strip().casefold()


def _resolve_niah_item(instance: dict[str, Any], item: str, child_prompt: str) -> str | None:
    """Ground truth for a NIAH ``item`` (a key): the sorted, comma-joined set
    of values whose needle sentence is literally present in ``child_prompt``,
    or "NONE" if the key is real but none of its needles are in this slice,
    or ``None`` if ``item`` names no real key at all."""
    units = [unit for unit in instance["tracked_units"] if unit["item"] == item]
    if not units:
        return None
    present = [unit for unit in units if unit["literal_text"] in child_prompt]
    if not present:
        return "NONE"
    return ", ".join(sorted({unit["value"] for unit in present}))


def _resolve_vt_item(instance: dict[str, Any], item: str, child_prompt: str) -> str | None:
    """Ground truth for a Variable Tracking ``item`` (a variable name):
    presence-only -- is this variable's OWN assignment line literally in the
    child's slice -- not a transitive-binding claim, since a child slice may
    not contain enough of the chain to determine transitive binding (the
    direct analogue of ``GraphWalksSubVerifier``'s one-hop-not-whole-path
    recomputation)."""
    units = [unit for unit in instance["tracked_units"] if unit["item"] == item]
    if not units:
        return None
    return "PRESENT" if units[0]["literal_text"] in child_prompt else "ABSENT"


def _resolve_word_count_item(instance: dict[str, Any], item: str, child_prompt: str) -> str | None:
    """Ground truth for a CWE/FWE ``item`` (a word): its literal, word-
    boundary-bounded occurrence count within the child's own prompt text,
    recomputed from scratch -- exact regardless of what else the slice
    contains."""
    known = {unit["item"] for unit in instance["tracked_units"]}
    if item not in known:
        return None
    count = len(re.findall(r"(?<!\w)" + re.escape(item) + r"(?!\w)", child_prompt))
    return str(count)


_LOCAL_GROUND_TRUTH_RESOLVERS: dict[str, Callable[[dict[str, Any], str, str], str | None]] = {
    "niah_single": _resolve_niah_item,
    "niah_multikey": _resolve_niah_item,
    "niah_multivalue": _resolve_niah_item,
    "niah_multiquery": _resolve_niah_item,
    "variable_tracking": _resolve_vt_item,
    "common_words_extraction": _resolve_word_count_item,
    "frequent_words_extraction": _resolve_word_count_item,
}


class RulerSubVerifier:
    """
    Post-hoc, deterministic check of one sub-call against ITS OWN prompt and
    its self-reported ``LOCAL FINDING:`` line(s) (see
    ``RULER_SUBCALL_CONTRACT``).

    Returns None -- uncheckable, never False -- for: an errored node, a
    non-string prompt/response, a response with no parseable ``LOCAL
    FINDING:`` line, an unrecognized ``task_type``, and a claimed item that
    names no real key/variable/word in this instance (the sub-call may have
    been asked about a different sub-task entirely). The body is wrapped
    defensively because a raise here would kill a whole mining round.
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

        findings = parse_local_findings(node.response)
        if not findings:
            return None
        resolver = _LOCAL_GROUND_TRUTH_RESOLVERS.get(str(instance["task_type"]))
        if resolver is None:
            return None

        checked = 0
        for item, claimed in findings.items():
            truth = resolver(instance, item, node.prompt)
            if truth is None:
                continue
            checked += 1
            if _normalize_local_value(claimed) != _normalize_local_value(truth):
                return False
        return True if checked > 0 else None


def make_ruler_sub_verifier() -> RulerSubVerifier:
    return RulerSubVerifier()
