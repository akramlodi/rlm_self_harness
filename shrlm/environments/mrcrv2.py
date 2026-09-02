"""
MRCRv2 environment: instances, a mining-side Verifier, and a SubVerifier.

MRCR (Multi-Round Co-reference Resolution) presents a long multi-turn
conversation containing several near-duplicate "needle" turns that all answer
the SAME recurring query with different content each time. The task is: given
the query and an instance index i, return the content of the i-th needle turn
matching that query, out of n_needles total. Distractor turns (other queries,
other content) surround the needles, so the task cannot be shortcut by a naive
keyword search -- the model has to count occurrences in order, not just find a
match.

Unlike GraphWalks/OOLONG this environment has no external dataset: every
instance is generated in-process (stdlib-only: ``random``/``hashlib``/``re``/
``difflib``), parameterized by ``target_tokens`` (approximated as
``chars / chars_per_token``, the same char-proxy convention GraphWalks already
uses via ``prompt_chars`` rather than a real tokenizer) and ``n_needles``. Two
shapes are used: short (~64K tokens, 2 needles) for mining + validation, and
long (~2M tokens, 8 needles) held out for frozen-harness eval only -- the same
split shape as GraphWalks (short mined/validated, long eval-only), not
OOLONG-synth's single length-diverse pool.

Grounding (the SubVerifier) needs a sub-call's own LOCAL, checkable claim, not
just the root's final answer. Nothing in the shared harness surfaces (S1-S9)
requires a sub-call to expose one, so this environment adds that requirement
itself, the same way OOLONG's ``ANSWER_FORMAT_CONTRACT`` (oolong.py) states the
root's answer contract: a ``SUBCALL_LOCAL_FINDING_CONTRACT`` block appended to
every instance's prompt, asking any sub-call the root spawns to end its own
response with one of two fixed-shape lines. The SubVerifier then grounds each
sub-call two ways, both by literal-substring matching against the sub-call's
OWN prompt -- the same mechanism ``GraphWalksSubVerifier`` (edges present in
the prompt) and ``OolongSubVerifier`` (labelled lines present in the prompt)
already use, since a ``CallNode`` carries only ``prompt``/``response`` text, no
structural offset metadata for a model-authored child call:

  1. which of the instance's needles were actually copied into this sub-call's
     prompt (by literal content match) -- that is "what this slice contains";
  2. what the sub-call claims about that slice, parsed from its
     ``LOCAL FINDING:`` line.

Resource exceptions never reach the Verifier: its signature receives only a
produced string, and RESOURCE_TERMINATED is owned by the experiment driver.
"""

from __future__ import annotations

import difflib
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict

# =============================================================================
# Generator vocabulary and templates
# =============================================================================

# Query templates: each instance picks ONE of these (formatted with a random
# topic word) as its recurring query. Distractor turns draw from the same pool
# with a DIFFERENT topic word, so they are lexically similar but never equal to
# the instance's actual query -- the model must match the query verbatim, not
# just the general shape.
_QUERY_TEMPLATES: tuple[str, ...] = (
    "What is your favorite {topic}?",
    "Describe your ideal {topic} in detail.",
    "Give me a recommendation for a {topic}.",
    "Write a short note about a memorable {topic}.",
    "What {topic} would you suggest for a beginner?",
)

_TOPICS: tuple[str, ...] = (
    "restaurant", "novel", "hiking trail", "coding language", "board game",
    "coffee blend", "movie", "city to visit", "recipe", "workout routine",
    "podcast", "museum exhibit", "song", "camera lens", "houseplant",
    "chess opening", "tea", "bicycle route", "puzzle", "documentary",
)

# Content vocabulary: needle/distractor "answers" are built from this pool so
# every generated content blob is a distinctive, non-overlapping sequence of
# words (SequenceMatcher-scorable) rather than templated boilerplate that could
# accidentally alias between needles.
_CONTENT_WORDS: tuple[str, ...] = (
    "amber", "quartz", "lantern", "harbor", "cinder", "willow", "granite",
    "meadow", "compass", "ember", "thistle", "canyon", "prairie", "obsidian",
    "sable", "brine", "ridge", "orchard", "tundra", "copper", "basalt",
    "clover", "delta", "fern", "glacier", "horizon", "ivory", "jasper",
    "kestrel", "lichen", "marrow", "nectar", "opal", "pebble", "quill",
    "raven", "sienna", "talon", "umber", "violet", "wren", "yarrow", "zephyr",
)

# The recurring per-instance answer contract, appended to every built prompt.
# Mirrors ``oolong.ANSWER_FORMAT_CONTRACT``'s role: states the ONE normalizing
# rule the verifier applies, so a correct answer in a slightly-off shell still
# grades correct.
ANSWER_FORMAT_CONTRACT = (
    "\n\n---\n"
    "Answer with the EXACT content of the requested needle turn, copied verbatim, on the "
    "final line (optionally prefixed with 'FINAL: '). If no such instance exists, put "
    "'FINAL: NONE' on the final line. Grading strips surrounding whitespace and the "
    "optional 'FINAL:' prefix, then compares the remainder to the gold content."
)

# The sub-call grounding contract (this environment's own addition, not a
# shared harness surface): if the root delegates a slice of the conversation to
# a sub-call, that sub-call must report a checkable local claim.
SUBCALL_LOCAL_FINDING_CONTRACT = (
    "\n\n---\n"
    "If you delegate any part of this conversation to a sub-call (llm_query/rlm_query), "
    "instruct it to end its own response with exactly one line of the form:\n"
    '  LOCAL FINDING: found instance <k> of query "<query text>" in this slice\n'
    "or, if the query never appears in its slice:\n"
    '  LOCAL FINDING: no match for query "<query text>" in this slice\n'
    "where <query text> is copied verbatim from the query you gave it, and <k> is the "
    "count of times that exact query's turn appears WITHIN the slice you gave it (not "
    "the whole conversation)."
)

_FINAL_LINE_RE = re.compile(r"^\s*(?:final)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
_NONE_MARKER_RE = re.compile(r"^\s*(?:none|n/?a|not\s+found|no\s+match)\s*$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

_LOCAL_FINDING_FOUND_RE = re.compile(
    r'LOCAL FINDING:\s*found instance\s+(?P<k>\d+)\s+of query\s+"(?P<query>[^"]*)"\s+in this slice',
    re.IGNORECASE,
)
_LOCAL_FINDING_NONE_RE = re.compile(
    r'LOCAL FINDING:\s*no match for query\s+"(?P<query>[^"]*)"\s+in this slice',
    re.IGNORECASE,
)

EXTRACTION_RULE = "final-line-or-marker;whitespace-normalized;sequence-similarity-ratio"


# =============================================================================
# Generation
# =============================================================================


@dataclass(frozen=True)
class Needle:
    """One needle turn's ground truth: its content, which query/instance-index
    it answers, and its character span in the built conversation.

    ``char_start``/``char_end`` are recorded for provenance/analysis (they
    answer "where in the raw conversation is this needle"), but the
    SubVerifier below grounds by literal-content substring matching against a
    sub-call's own prompt rather than by offset arithmetic -- a ``CallNode``
    carries no offset metadata for a model-authored child prompt, only text.
    """

    query: str
    instance_index: int  # 1-based: this is the k-th needle turn for `query`
    content: str
    char_start: int
    char_end: int


def _sample_content(rng: random.Random, min_words: int = 8, max_words: int = 20) -> str:
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(_CONTENT_WORDS) for _ in range(n))


def _sample_query(rng: random.Random, exclude_topic: str | None = None) -> str:
    template = rng.choice(_QUERY_TEMPLATES)
    topics = [t for t in _TOPICS if t != exclude_topic] or list(_TOPICS)
    return template.format(topic=rng.choice(topics))


def _render_turn(role_label: str, text: str) -> str:
    return f"{role_label}: {text}"


def generate_mrcrv2_instance(
    target_tokens: int,
    n_needles: int,
    seed: int,
    index: int,
    *,
    chars_per_token: float = 4.0,
    min_distractor_words: int = 6,
    max_distractor_words: int = 16,
) -> dict[str, Any]:
    """Generate one MRCRv2 instance.

    Deterministic for a given ``(target_tokens, n_needles, seed, index)``: the
    per-instance RNG is seeded from all four, so two calls with the same
    arguments produce byte-identical output and different ``index`` values
    produce different (but still reproducible) instances from one ``seed``.
    """
    if n_needles < 1:
        raise ValueError(f"n_needles must be >= 1, got {n_needles}")
    if target_tokens < 1:
        raise ValueError(f"target_tokens must be >= 1, got {target_tokens}")

    rng = random.Random(f"{seed}:{index}:{target_tokens}:{n_needles}")
    target_chars = int(target_tokens * chars_per_token)

    query = _sample_query(rng)
    needle_contents = set()
    while len(needle_contents) < n_needles:
        needle_contents.add(_sample_content(rng))
    needle_contents = list(needle_contents)
    rng.shuffle(needle_contents)

    turns: list[str] = []
    needles: list[Needle] = []
    needles_placed = 0
    # Interleave needle turns among distractor turns: a needle is placed at a
    # random point in each of n_needles equal-sized windows across the build,
    # so needles are scattered rather than clustered at the start or end.
    # ``conversation`` is built incrementally (not assembled from ``turns`` at
    # the end) so every needle's char_start/char_end is measured against the
    # ACTUAL joined string, including the "\n\n" turn separator -- computing
    # offsets from a guessed separator width and joining afterward drifts them
    # out of sync with the real string by one separator's width per turn.
    conversation = ""
    total_chars_budget = target_chars
    while len(conversation) < total_chars_budget or needles_placed < n_needles:
        place_needle = needles_placed < n_needles and (
            len(conversation) >= total_chars_budget
            or rng.random() < (1.0 / max(1, n_needles))
        )
        if place_needle:
            content = needle_contents[needles_placed]
            turn_text = _render_turn("user", query) + "\n" + _render_turn("assistant", content)
        else:
            distractor_query = _sample_query(rng, exclude_topic=None)
            if distractor_query == query:
                continue
            distractor_content = _sample_content(
                rng, min_distractor_words, max_distractor_words
            )
            turn_text = (
                _render_turn("user", distractor_query)
                + "\n"
                + _render_turn("assistant", distractor_content)
            )

        separator = "\n\n" if turns else ""
        char_start = len(conversation) + len(separator)
        conversation += separator + turn_text
        turns.append(turn_text)

        if place_needle:
            needles.append(
                Needle(
                    query=query,
                    instance_index=needles_placed + 1,
                    content=content,
                    char_start=char_start,
                    char_end=len(conversation),
                )
            )
            needles_placed += 1

        if len(conversation) >= total_chars_budget and needles_placed >= n_needles:
            break
    target_instance_index = rng.randint(1, n_needles)
    gold_needle = next(n for n in needles if n.instance_index == target_instance_index)

    question = (
        f'This conversation contains {n_needles} turn(s) where the user asked exactly: '
        f'"{query}"\n'
        f"Return the content of the assistant's response to the "
        f"{_ordinal(target_instance_index)} occurrence of that exact question "
        f"(counting from the start of the conversation)."
    )
    prompt = (
        conversation
        + "\n\n---\n"
        + question
        + ANSWER_FORMAT_CONTRACT
        + SUBCALL_LOCAL_FINDING_CONTRACT
    )

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    instance_id = f"mrcrv2-{n_needles}n-{digest}"

    return {
        "id": instance_id,
        "question": question,
        "prompt": prompt,
        "query": query,
        "n_needles": n_needles,
        "target_instance_index": target_instance_index,
        "gold_content": gold_needle.content,
        "needles": [
            {
                "query": n.query,
                "instance_index": n.instance_index,
                "content": n.content,
                "char_start": n.char_start,
                "char_end": n.char_end,
            }
            for n in needles
        ],
        "target_tokens": target_tokens,
        "prompt_chars": len(prompt),
        "sample_seed": seed,
        "sample_index": index,
    }


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def generate_mrcrv2_instances(
    target_tokens: int,
    n_needles: int,
    limit: int,
    seed: int,
    **generator_kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ``limit`` MRCRv2 instances. Deterministic for a given
    ``(target_tokens, n_needles, limit, seed)``; ids are content-derived and
    unique because each instance draws from a distinct ``index``."""
    return [
        generate_mrcrv2_instance(target_tokens, n_needles, seed, index, **generator_kwargs)
        for index in range(limit)
    ]


# =============================================================================
# Answer parsing and scoring
# =============================================================================


def _candidate_line(response: str) -> str | None:
    """The substring to parse an answer out of: the last ``FINAL:``-marked
    line if any, else the last non-empty line. None for an all-whitespace
    response. Mirrors ``oolong._candidate_line``."""
    lines = [line for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        marker = _FINAL_LINE_RE.match(line)
        if marker is not None:
            return marker.group(1).strip()
    return lines[-1].strip()


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).strip()


def extract_mrcrv2_answer(response: str) -> tuple[str, bool] | None:
    """Parse a model response into ``(normalized_text, empty)``.

    ``empty=True`` means the final line carried an explicit "none/not found"
    marker (NO_ANSWER against a non-empty gold). Returns None when there is no
    candidate line at all (WRONG_FORMAT) -- same None-vs-empty split as
    ``graphwalks.extract_answer_nodes`` / ``oolong.extract_oolong_answer``.
    """
    candidate = _candidate_line(response)
    if candidate is None:
        return None
    normalized = _normalize(candidate)
    if not normalized or _NONE_MARKER_RE.match(normalized):
        return "", True
    return normalized, False


def score_mrcrv2(produced: str, gold: str) -> float:
    """Sequence-similarity ratio in [0, 1] between the normalized produced
    text and the gold needle content (``difflib.SequenceMatcher.ratio``, the
    same metric family the MRCR reference implementation uses)."""
    return difflib.SequenceMatcher(None, produced, gold).ratio()


# =============================================================================
# Verifier
# =============================================================================


class Mrcrv2Verifier:
    """
    Deterministic outcome for a whole run: near-exact recall of the requested
    needle's content.

    ``PASS_SCORE_THRESHOLD = 1.0`` matches ``OolongVerifier``'s own threshold
    exactly (deliberately -- comparability under the promotion rule's pass-count
    comparison is the point), even though the underlying score is a continuous
    sequence-similarity ratio rather than an exact-match boolean: MRCR's task IS
    verbatim recall, so requiring ratio == 1.0 to pass is the task's own
    semantics, not an arbitrary tightening. The continuous ratio is still
    reported in ``detail`` for partial-credit visibility, the same pattern
    ``OolongVerifier`` uses (its docstring: "Partial credit is reported in
    detail... the promotion gate compares pass counts, so a binary pass keeps
    that machinery untouched.").

    Cause mapping:

    * no candidate line at all -> WRONG_FORMAT.
    * an explicit none/not-found marker against a non-empty gold -> NO_ANSWER.
    * every other miss (score < threshold) -> WRONG_VALUE; MRCR's answer is a
      single free-text field, not a set, so none of GraphWalks/OOLONG's
      INCOMPLETE/SPURIOUS/MIXED_SET_ERROR set-arithmetic causes apply.
    """

    PASS_SCORE_THRESHOLD: float = 1.0
    EXTRACTION_RULE: str = EXTRACTION_RULE
    GOLD_ORDERING: str = "verbatim"

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig by the experiment driver."""
        return {
            "environment": "mrcrv2",
            "pass_score_threshold": self.PASS_SCORE_THRESHOLD,
            "extraction_rule": self.EXTRACTION_RULE,
            "scoring_metric": "difflib.SequenceMatcher.ratio",
            "gold_ordering": self.GOLD_ORDERING,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold = str(instance["gold_content"])
        gold_normalized = _normalize(gold)

        parsed = extract_mrcrv2_answer(produced)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold_normalized,
                produced=produced,
                detail="no candidate answer line to parse",
            )
        candidate, empty = parsed
        if empty:
            return Verdict(
                passed=False,
                cause=VerifierCause.NO_ANSWER,
                gold=gold_normalized,
                produced=candidate,
                detail="final line carried an explicit none/not-found marker",
            )

        ratio = score_mrcrv2(candidate, gold_normalized)
        detail = f"ratio={ratio:.3f}"
        if ratio >= self.PASS_SCORE_THRESHOLD:
            return Verdict(
                passed=True, cause=None, gold=gold_normalized, produced=candidate, detail=detail
            )
        return Verdict(
            passed=False,
            cause=VerifierCause.WRONG_VALUE,
            gold=gold_normalized,
            produced=candidate,
            detail=detail,
        )


# =============================================================================
# SubVerifier
# =============================================================================


def needles_in_slice(needles: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """Which of the instance's needles were copied (by literal content match)
    into a sub-call's own prompt -- the substring-matching proxy for "what this
    slice contains", since a ``CallNode`` carries no structural offset data for
    a model-authored child prompt."""
    return [needle for needle in needles if str(needle["content"]) in prompt]


def parse_local_finding(response: str) -> tuple[str, int | None] | None:
    """Best-effort parse of a sub-call's trailing ``LOCAL FINDING:`` line into
    ``(query, k)`` where ``k`` is None for an explicit no-match claim. Returns
    None when no recognized LOCAL FINDING line is present -- uncheckable, not
    wrong; a sub-call that just dumped raw text back to the root exposes no
    checkable local claim at all."""
    found = _LOCAL_FINDING_FOUND_RE.search(response)
    if found is not None:
        return found.group("query").strip(), int(found.group("k"))
    none_match = _LOCAL_FINDING_NONE_RE.search(response)
    if none_match is not None:
        return none_match.group("query").strip(), None
    return None


class Mrcrv2SubVerifier:
    """
    Post-hoc, deterministic check of one sub-call against ITS OWN prompt.

    A sub-call's local claim ("found instance k of query Q in this slice" / "no
    match") is graded against what the instance's own needle records say is
    actually present in that sub-call's prompt (by literal-content substring
    match, see ``needles_in_slice``) -- never against the environment's
    ``target_instance_index``, which is the ROOT's problem, not any one child's.

    Returns None -- uncheckable, never False -- for: an errored node, a
    non-string prompt/response, a response with no parseable LOCAL FINDING
    line, and a claimed query that does not match any needle's query (the
    sub-call may have been asked about a different sub-task entirely, e.g. a
    root that never delegates a needle-bearing slice at all). That last case
    matters the same way it does in GraphWalks/OOLONG: a sub-call that
    correctly reports "no match" for a query that never appears anywhere in
    the whole instance must not masquerade as a checked "no needles in slice"
    claim it never actually made a checkable statement about.
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

        parsed = parse_local_finding(node.response)
        if parsed is None:
            return None
        claimed_query, claimed_k = parsed

        needles = instance.get("needles", [])
        known_queries = {str(needle["query"]) for needle in needles}
        if claimed_query not in known_queries:
            return None

        in_slice = needles_in_slice(needles, node.prompt)
        expected_count = sum(
            1 for needle in in_slice if str(needle["query"]) == claimed_query
        )

        if claimed_k is None:
            return expected_count == 0
        return claimed_k == expected_count


def make_mrcrv2_verifier() -> Mrcrv2Verifier:
    """Zero-arg factory for validation child processes (EvaluationConfig.verifier_factory)."""
    return Mrcrv2Verifier()


def make_mrcrv2_sub_verifier() -> Mrcrv2SubVerifier:
    return Mrcrv2SubVerifier()


# =============================================================================
# Config wiring
# =============================================================================


def load_mrcrv2_from_config(
    config: Any,
    n: int,
    seed: int,
    length: str = "short",
) -> list[dict[str, Any]]:
    """``generate_mrcrv2_instances`` with target-tokens/n-needles facts taken
    from the experiment config, keyed by split length ("short" or "long")."""
    env = config.environments.mrcrv2
    if length == "short":
        target_tokens, n_needles = env.short_target_tokens, env.short_n_needles
    elif length == "long":
        target_tokens, n_needles = env.long_target_tokens, env.long_n_needles
    else:
        raise ValueError(f"unknown split length {length!r}; expected 'short' or 'long'")
    return generate_mrcrv2_instances(
        target_tokens=target_tokens,
        n_needles=n_needles,
        limit=n,
        seed=seed,
        chars_per_token=env.chars_per_token,
    )
