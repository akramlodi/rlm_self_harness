"""
GraphWalks environment: instances, a mining-side Verifier, and a SubVerifier.

GraphWalks (https://huggingface.co/datasets/openai/graphwalks) asks for the
exact node set produced by a graph operation -- "BFS from node X at depth D" or
"parents of node X" -- over a directed edge list, answered as a trailing
"Final Answer: [n1, n2, ...]" line. Set-valued gold with a deterministic
extraction rule makes it a natural fit for the weakness miner: the terminal
verifier cause is arithmetic on two sets, and each sub-call the harness spawns
is a one-hop sub-problem checkable in isolation.

The parsing and scoring helpers here re-implement the logic of
``examples/graphwalks_example.py`` rather than importing it: the example is a
script under ``examples/``, not a package module, so importing it would couple
library code to a demo entry point. One deliberate difference is documented on
``extract_answer_nodes``: the example returns ``[]`` both for "no Final Answer
line" and for a parsed empty list, while the Verifier and SubVerifier need to
distinguish a format failure from an explicit empty-set answer.

Resource exceptions never reach the Verifier: its signature receives only a
produced string, and RESOURCE_TERMINATED is owned by the experiment driver.
"""

import hashlib
import random
import re
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict

DATASET_REPO = "openai/graphwalks"
DATASET_FILE = "graphwalks_128k_and_shorter.parquet"

# One directed edge as the dataset renders it: "a1b2 -> c3d4".
_EDGE_RE = re.compile(r"(\w+)\s*->\s*(\w+)")
# The bracketed list inside a "Final Answer: [...]" line. Non-greedy so the
# FIRST bracket pair on the line is the answer list, per the dataset card's
# "Final Answer: [n1, n2, ...]" format: a greedy match would swallow any later
# bracketed text (e.g. "Final Answer: [a, b] (excluding [c])") into the list
# and grade a correct answer wrong.
_ANSWER_LIST_RE = re.compile(r"\[(.*?)\]")
# Characters stripped from each parsed item: stray brackets from redundant
# nesting, and quotes from a Python-repr'd list. Node ids are \w+ and can carry
# neither, so stripping cannot merge or split a real id.
_ITEM_STRIP_CHARS = "[]'\""
# An operation line of a dataset prompt. Every GraphWalks prompt opens with a
# worked example whose operation line matches this too, so the instance's
# question is the LAST match, never the first (``extract_question``).
_QUESTION_RE = re.compile(r"^\s*(Perform a BFS\b.*|Find the parents\b.*)$", re.MULTILINE)

# Best-effort patterns over model-authored child prompts (see GraphWalksSubVerifier).
_PARENTS_QUERY_RE = re.compile(r"\bparents of (?:the )?(?:node )?['\"`]?(\w+)", re.IGNORECASE)
_CHILDREN_LIST_RE = re.compile(r"\bchildren of[^\[\]\n]*\[([^\]]*)\]", re.IGNORECASE)
_FRONTIER_LIST_RE = re.compile(r"\bfrontier[^\[\]\n]*\[([^\]]*)\]", re.IGNORECASE)
_CHILDREN_NODE_RE = re.compile(r"\bchildren of (?:the )?(?:node )?['\"`]?(\w+)", re.IGNORECASE)
_EXCLUDE_LIST_RE = re.compile(r"\bexclud\w*[^\[\]\n]*\[([^\]]*)\]", re.IGNORECASE)
_EXCLUDE_WORD_RE = re.compile(r"\bexclud", re.IGNORECASE)


# =============================================================================
# Answer parsing and scoring (per the dataset card)
# =============================================================================


def extract_answer_nodes(response: str) -> list[str] | None:
    """
    Parse the node list out of a response's trailing answer line.

    The dataset card's format is a last line ``Final Answer: [n1, n2, ...]``.
    Extraction is deliberately more tolerant than the card on two points that
    are serialization, not graph reasoning (experiment_kimi/POST_MORTEM.md
    section 3): the ``Final Answer:`` marker is optional -- a bracketed list
    on the last line is the answer whether or not the marker precedes it --
    and quote characters around an item are stripped, so ``['a', 'b']`` (a
    Python list's ``str()``) parses to ``["a", "b"]``. Under the strict rule
    84% of the H0 baseline's failures were correct node sets wearing quotes or
    missing the marker, and the optimization loop spent five rounds moving
    bytes across that boundary instead of improving the harness.

    Returns None when the last line carries no bracketed list at all -- the
    response is not in the required format -- and an empty list when a bracket
    pair parsed but held no items, i.e. the model explicitly answered the empty
    set. Callers rely on that distinction: the Verifier maps no-parse to
    WRONG_FORMAT but a parsed ``[]`` against a non-empty gold to NO_ANSWER, and
    the SubVerifier treats an unparseable child response as uncheckable rather
    than wrong.

    The FIRST bracket pair on the line is the answer list; anything after its
    closing bracket (e.g. a parenthetical "(excluding [c])") is commentary and
    ignored. Redundantly nested brackets like "Final Answer: [[a, b]]" are
    tolerated by stripping stray bracket characters from each item -- node ids
    are ``\\w+`` and can never legitimately contain brackets or quotes -- so
    that parse yields ``["a", "b"]``.
    """
    line = response.strip().split("\n")[-1]
    match = _ANSWER_LIST_RE.search(line)
    if match is None:
        return None
    items = (item.strip().strip(_ITEM_STRIP_CHARS).strip() for item in match.group(1).split(","))
    return [item for item in items if item]


def score(predicted: list[str], golden: list[str]) -> dict[str, float]:
    """
    Precision/recall/F1 over node sets, verbatim per the dataset card.

    The card's code returns f1=1.0 whenever precision + recall == 0, which
    covers the intended both-empty case but also the degenerate empty
    prediction against a non-empty gold. The Verifier therefore decides
    pass/fail on exact set equality -- identical to F1 == 1.0 in every
    non-degenerate case -- and never lets that branch turn a miss into a pass.
    """
    pred_set, gold_set = set(predicted), set(golden)
    n_overlap = len(pred_set & gold_set)
    n_golden, n_sampled = len(gold_set), len(pred_set)
    recall = n_overlap / n_golden if n_golden > 0 else 0.0
    precision = n_overlap / n_sampled if n_sampled > 0 else 0.0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def serialize_nodes(nodes: set[str]) -> str:
    """Render a node set SORTED, so equal sets always serialize identically.

    The Verdict's gold field feeds the digest sha and through it the
    attribution cache; an order-dependent rendering would split cache entries
    for byte-identical problems.
    """
    return "[" + ", ".join(sorted(nodes)) + "]"


# =============================================================================
# Dataset loading
# =============================================================================


def row_to_instance(row: dict[str, Any], sample_seed: int, sample_index: int) -> dict[str, Any]:
    """
    Pure transform from a dataset row to a mining instance.

    The id is derived from CONTENT -- problem_type plus sha256(prompt)[:16] --
    so the same problem keeps the same id across rounds and across differently
    seeded samples. The seed and index are provenance, carried as separate
    fields and deliberately not folded into the id.
    """
    prompt = str(row["prompt"])
    problem_type = str(row["problem_type"])
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"{problem_type}-{digest}",
        "question": extract_question(prompt),
        "prompt": prompt,
        "answer_nodes": [str(node) for node in row["answer_nodes"]],
        "problem_type": problem_type,
        "sample_seed": sample_seed,
        "sample_index": sample_index,
    }


def extract_question(prompt: str) -> str:
    """The operation line of the prompt ("Perform a BFS..."/"Find the parents..."),
    falling back to the first non-empty line when no operation line is found.

    The LAST operation line wins. Dataset prompts embed a worked example
    ("Perform a BFS from node abcd with depth 1.") before the real
    ``Operation:`` block; taking the first match handed the attributor the
    example's question for every instance of experiment_kimi and 41% of its
    attributions blamed the model for "using the wrong node"
    (POST_MORTEM.md section 5).
    """
    matches = _QUESTION_RE.findall(prompt)
    if matches:
        return matches[-1].strip()
    for line in prompt.splitlines():
        if line.strip():
            return line.strip()
    return ""


def sample_rows(
    rows: list[dict[str, Any]],
    problem_types: tuple[str, ...],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Seeded sample over pre-filtered rows, balanced across problem types.

    With a limit and more than one problem type, an equal share is drawn per
    type so a small sample still exercises both operations; otherwise a plain
    seeded shuffle-and-truncate. Deterministic for a given (rows, seed).

    The per-type share alone can undershoot the limit -- a limit that is not a
    multiple of the type count leaves a remainder, and a type whose pool is
    smaller than its share leaves a shortfall -- so the balanced draw is topped
    up from the leftover rows with the same seeded RNG. Exactly
    ``min(limit, available)`` rows are always returned, where ``available``
    counts the rows of the requested types, with the per-type balance kept as
    close as the pools allow.
    """
    rng = random.Random(seed)
    if limit is not None and len(problem_types) > 1:
        per_type = max(1, limit // len(problem_types))
        picked: list[dict[str, Any]] = []
        leftover: list[dict[str, Any]] = []
        for problem_type in problem_types:
            pool = [row for row in rows if row["problem_type"] == problem_type]
            rng.shuffle(pool)
            picked.extend(pool[:per_type])
            leftover.extend(pool[per_type:])
        shortfall = min(limit, len(picked) + len(leftover)) - len(picked)
        if shortfall > 0:
            rng.shuffle(leftover)
            picked.extend(leftover[:shortfall])
        rng.shuffle(picked)
        return picked[:limit]

    pool = list(rows)
    rng.shuffle(pool)
    return pool[:limit] if limit is not None else pool


# Default bound applied by ``fetch_rows`` below; see that function's docstring
# for exactly what it does and does not cover.
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS: float = 60.0


def fetch_rows(
    dataset_repo: str,
    dataset_file: str,
    revision: str | None,
    download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Download one dataset parquet from Hugging Face and read it into rows.

    ``revision=None`` follows the repo's default branch; a pinned commit sha
    freezes the rows against upstream edits (the dataset shipped a ground-truth
    correction on 2026-02-27, so unpinned loads can silently drift).

    This is the loader's only network-touching seam, monkeypatched in tests,
    and it runs inside ``materialize_splits`` on the unattended
    ``examples/experiment_smoke.py --live`` path -- outside the SIGALRM
    hard-deadline backstop, which only bounds run execution, not split
    materialization. Without a bound here, a stalled Hugging Face connection
    hangs the whole invocation forever with no failure signal. (Note this is
    unrelated to ``config.operational.loader_timeout_seconds``, which bounds a
    candidate-materialization subprocess in the validation path -- see
    ``shrlm/optimization/validation.py`` -- and is not reused here.)

    ``download_timeout_seconds`` is applied two ways, both via
    ``huggingface_hub``:

    * as ``hf_hub_download``'s ``etag_timeout``, bounding the initial
      metadata (HEAD) request that resolves the revision to a concrete file;
    * as the ``HF_HUB_DOWNLOAD_TIMEOUT`` setting (mutated on
      ``huggingface_hub.constants`` for the duration of this call and
      restored after), bounding every read of the actual parquet transfer
      that follows.

    Both are PER-OPERATION timeouts applied by the underlying HTTP client,
    not a cap on total wall-clock duration: a connection that goes fully
    silent for longer than ``download_timeout_seconds`` raises loudly, but
    one that keeps dribbling at least one byte within every window does not
    trip this bound and could still run arbitrarily long. That is the failure
    mode this guards against -- a hung connection, not a slow one. It also
    only covers the plain-HTTP download path; if the optional ``hf_xet``
    accelerator package is installed, ``huggingface_hub`` may use a separate
    client that this constant does not govern.

    The imports live inside the function so the module imports without the
    ``graphwalks`` extra; only actually loading the dataset requires it.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for the GraphWalks dataset. Install with:\n"
            '    uv pip install -e ".[graphwalks]"'
        ) from exc

    previous_timeout = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = download_timeout_seconds
    try:
        path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=dataset_file,
            revision=revision,
            etag_timeout=download_timeout_seconds,
        )
    finally:
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = previous_timeout
    return pq.read_table(path).to_pylist()


def load_graphwalks(
    problem_types: tuple[str, ...] = ("bfs", "parents"),
    max_chars: int | None = 128_000,
    limit: int | None = None,
    seed: int = 0,
    dataset_repo: str = DATASET_REPO,
    dataset_file: str = DATASET_FILE,
    min_chars: int = 0,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """
    Download a GraphWalks parquet, filter, sample, and build instances.

    The defaults reproduce the original hardcoded behavior exactly: the
    128k-and-shorter file at the repo's default revision, capped at 128k
    prompt chars with no lower bound. The long split is the same code under
    different arguments -- ``dataset_file="graphwalks_256k_to_1mil.parquet"``
    with a ``min_chars`` floor and ``max_chars=None`` (the file's own 256k-1M
    range is the only upper bound; there is no configured long cap to apply).
    Both bounds are inclusive.
    """
    rows = fetch_rows(dataset_repo, dataset_file, revision)
    rows = [
        row
        for row in rows
        if row["prompt_chars"] >= min_chars
        and (max_chars is None or row["prompt_chars"] <= max_chars)
        and row["problem_type"] in problem_types
    ]
    selected = sample_rows(rows, problem_types, limit, seed)
    return [
        row_to_instance(row, sample_seed=seed, sample_index=index)
        for index, row in enumerate(selected)
    ]


# =============================================================================
# Verifier
# =============================================================================


class GraphWalksVerifier:
    """
    Deterministic outcome for a whole run: exact node-set match.

    Cause mapping, decided entirely by set arithmetic on the parsed answer:

    * no bracketed node list on the trailing line (including a prose
      fallback answer) -> WRONG_FORMAT. The response never entered the
      environment's answer channel, so nothing finer can be measured. The
      ``Final Answer:`` marker itself is optional and item quotes are
      stripped; see ``extract_answer_nodes``.
    * a parsed ``[]`` against a non-empty gold -> NO_ANSWER. The format was
      honored but no answer nodes were produced; ``extract_answer_nodes``
      returning ``[]`` rather than None is what separates this from the case
      above.
    * missing nodes only -> INCOMPLETE; extra nodes only -> SPURIOUS; both
      -> MIXED_SET_ERROR.
    * both gold and prediction empty -> pass.

    A pass requires the exact gold set, i.e. F1 == 1.0 under the dataset
    card's scorer in every non-degenerate case (see ``score``).
    """

    PASS_F1_THRESHOLD: float = 1.0
    EXTRACTION_RULE: str = "trailing-bracket-list;marker-optional;quotes-stripped"
    GOLD_ORDERING: str = "sorted"

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig by the experiment driver."""
        return {
            "environment": "graphwalks",
            "pass_f1_threshold": self.PASS_F1_THRESHOLD,
            "extraction_rule": self.EXTRACTION_RULE,
            "gold_ordering": self.GOLD_ORDERING,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold_set = {str(node) for node in instance["answer_nodes"]}
        gold = serialize_nodes(gold_set)

        parsed = extract_answer_nodes(produced)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold,
                produced=produced,
                detail="no bracketed node list on the trailing line to parse",
            )

        pred_set = set(parsed)
        produced_nodes = serialize_nodes(pred_set)
        missing = gold_set - pred_set
        extra = pred_set - gold_set
        metrics = score(sorted(pred_set), sorted(gold_set))
        detail = (
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} missing={len(missing)} extra={len(extra)}"
        )

        if not missing and not extra:
            return Verdict(
                passed=True, cause=None, gold=gold, produced=produced_nodes, detail=detail
            )

        if not pred_set:
            cause = VerifierCause.NO_ANSWER
        elif missing and extra:
            cause = VerifierCause.MIXED_SET_ERROR
        elif missing:
            cause = VerifierCause.INCOMPLETE
        else:
            cause = VerifierCause.SPURIOUS
        return Verdict(passed=False, cause=cause, gold=gold, produced=produced_nodes, detail=detail)


# =============================================================================
# SubVerifier
# =============================================================================


def one_hop(
    edges: list[tuple[str, str]],
    frontier: set[str],
    excluded: set[str],
    reverse: bool,
) -> set[str]:
    """One relational hop over an edge slice, mirroring the example's verify_child:
    forward follows edges out of the frontier, reverse follows them backward."""
    if reverse:
        edges = [(v, u) for (u, v) in edges]
    return {v for (u, v) in edges if u in frontier} - excluded


def parse_subproblem(
    prompt: str,
) -> tuple[list[tuple[str, str]], set[str], set[str], bool] | None:
    """
    Best-effort parse of a model-authored child prompt into a one-hop query.

    Under H0 the root is instructed to hand each hop to a child as an edge
    slice plus either "children of <frontier>" (forward) or "parents of
    <node>" (reverse), but the exact wording is the model's. This recognizes
    the shapes that phrasing can take -- a "parents of X" query, a bracketed
    frontier list near "children of"/"frontier", or a single-node "children
    of X" -- and returns None for anything else rather than guessing. If the
    prompt mentions an exclusion but its node list does not parse, the whole
    parse is None: grading without the exclusion could mark a correct child
    wrong.
    """
    edges = _EDGE_RE.findall(prompt)
    if not edges:
        return None

    parents_match = _PARENTS_QUERY_RE.search(prompt)
    if parents_match is not None:
        frontier = {parents_match.group(1)}
        reverse = True
    else:
        list_match = _CHILDREN_LIST_RE.search(prompt) or _FRONTIER_LIST_RE.search(prompt)
        if list_match is not None:
            frontier = _split_nodes(list_match.group(1))
        else:
            node_match = _CHILDREN_NODE_RE.search(prompt)
            if node_match is None:
                return None
            frontier = {node_match.group(1)}
        reverse = False
    if not frontier:
        return None

    excluded_match = _EXCLUDE_LIST_RE.search(prompt)
    if excluded_match is not None:
        excluded = _split_nodes(excluded_match.group(1))
    elif _EXCLUDE_WORD_RE.search(prompt):
        return None
    else:
        excluded = set()

    return edges, frontier, excluded, reverse


def _split_nodes(text: str) -> set[str]:
    """Split a bracketed list's interior into node ids, tolerating quotes."""
    return {item.strip().strip("'\"`") for item in text.split(",") if item.strip().strip("'\"`")}


class GraphWalksSubVerifier:
    """
    Post-hoc, deterministic check of one sub-call against ITS OWN prompt.

    The node's prompt is parsed for an edge slice and a one-hop query, the hop
    is recomputed exactly, and the child's parsed "Final Answer" is compared
    against it. Depth > 1 nodes are graded against the slice in their own
    prompt only; the ``instance`` argument is deliberately unused, because a
    child's sub-problem is defined by its parent's decomposition, not by the
    environment.

    Returns None -- uncheckable, never False -- for: a non-string (message
    list) prompt, an errored node, a prompt that does not parse as a one-hop
    query, and a child response with no parseable "Final Answer" line. That
    last one matters: a child that answered correctly in the wrong format must
    not masquerade as a wrong child answer, which would flip the failing level
    to CHILD on a formatting quirk. The body is wrapped defensively because a
    raise here would kill a whole mining round.
    """

    def __call__(self, instance: dict[str, Any], node: CallNode) -> bool | None:
        try:
            return self._check(node)
        except Exception:
            return None

    def _check(self, node: CallNode) -> bool | None:
        if node.kind is NodeKind.ERRORED or node.error_kind is not None:
            return None
        if not isinstance(node.prompt, str) or not isinstance(node.response, str):
            return None

        parsed = parse_subproblem(node.prompt)
        if parsed is None:
            return None
        child_out = extract_answer_nodes(node.response)
        if child_out is None:
            return None

        edges, frontier, excluded, reverse = parsed
        return set(child_out) == one_hop(edges, frontier, excluded, reverse)
