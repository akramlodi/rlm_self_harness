"""
Example: GraphWalks multi-hop graph reasoning with a recursive, self-verifying RLM.

GraphWalks (https://huggingface.co/datasets/openai/graphwalks) is OpenAI's long-context
benchmark where the model is given a directed graph as an edge list and must answer a
"BFS from node X at depth D" or "parents of node X" query with an exact set of node ids.

It is a good fit for RLM's recursive sub-call model because the underlying operation is a
*deterministic, checkable* graph algorithm rather than an open-ended judgment call:

  1. The root RLM parses the edge list out of `context` with code and computes the exact
     answer itself (a BFS frontier expansion, or a reverse-adjacency lookup) -- no LLM
     needed for the traversal itself.
  2. It then extracts the small *induced subgraph* relevant to the query (the nodes/edges
     actually touched by the traversal) and hands that -- not the whole graph -- to a
     child via `rlm_query()`, asking it to solve the same, now much smaller, operation.
  3. Because both the parent's code and the child's answer are working over the same
     small, self-contained subgraph, the child's answer is directly checkable against the
     parent's own computed result before a final answer is submitted.

This example downloads the real dataset from Hugging Face (`openai/graphwalks`,
`graphwalks_128k_and_shorter.parquet`), filters down to short prompts (<=128K characters
by default), runs each sampled problem through an `RLM` (Gemini backend, local Python REPL,
`max_depth=2` so `rlm_query()` spawns a real child RLM), and grades the response against
ground truth using the extraction/F1 scoring described on the dataset card.

Usage:
    uv pip install -e ".[graphwalks]"
    GEMINI_API_KEY=... python -m examples.graphwalks_example --n 4

    # Options:
    python -m examples.graphwalks_example --n 8 --problem-type bfs --model gemini-2.5-flash \\
        --max-chars 32000 --verbose --log-dir ./logs/graphwalks
"""

import argparse
import os
import random
import re
import sys

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.exceptions import (
    BudgetExceededError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)

load_dotenv()

DATASET_REPO = "openai/graphwalks"
DATASET_FILE = "graphwalks_128k_and_shorter.parquet"


# =============================================================================
# Dataset loading
# =============================================================================


def load_graphwalks(
    problem_types: tuple[str, ...] = ("bfs", "parents"),
    max_chars: int = 128_000,
    limit: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Download graphwalks_128k_and_shorter.parquet from HF and filter it.

    The parquet file itself buckets prompts by an approximate token budget (its rows
    span ~2.6K to ~220K *characters*), so we still apply an explicit `max_chars` filter
    on the `prompt_chars` column to get the actually-short subset.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Missing dependency for the GraphWalks dataset. Install with:\n"
            '    uv pip install -e ".[graphwalks]"'
        ) from e

    path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    rows = pq.read_table(path).to_pylist()
    rows = [
        r for r in rows if r["prompt_chars"] <= max_chars and r["problem_type"] in problem_types
    ]

    rng = random.Random(seed)
    if limit is not None and len(problem_types) > 1:
        # Balanced sample across problem types so a small --n still shows both.
        per_type = max(1, limit // len(problem_types))
        picked = []
        for pt in problem_types:
            pool = [r for r in rows if r["problem_type"] == pt]
            rng.shuffle(pool)
            picked.extend(pool[:per_type])
        rng.shuffle(picked)
        return picked[:limit]

    rng.shuffle(rows)
    return rows[:limit] if limit is not None else rows


# =============================================================================
# Grading (per https://huggingface.co/datasets/openai/graphwalks)
# =============================================================================


def extract_answer_nodes(response: str) -> list[str]:
    """Pull the node list out of a response's trailing 'Final Answer: [...]' line.

    Same intent as the extraction snippet on the dataset card, but reads the list
    directly out of the brackets rather than via `str.strip("[]")` on the whole
    "Final Answer: [...]" match (which leaves the "Final Answer: [" prefix glued to
    the first item since strip() only trims from the very ends of the string).
    """
    line = response.strip().split("\n")[-1]
    if "Final Answer:" not in line:
        return []
    match = re.search(r"\[(.*)\]", line)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def score(predicted: list[str], golden: list[str]) -> dict[str, float]:
    """Precision/recall/F1 over node sets, verbatim per the dataset card's grading code."""
    pred_set, gold_set = set(predicted), set(golden)
    n_overlap = len(pred_set & gold_set)
    n_golden, n_sampled = len(gold_set), len(pred_set)
    recall = n_overlap / n_golden if n_golden > 0 else 0.0
    precision = n_overlap / n_sampled if n_sampled > 0 else 0.0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Deterministic sub-call verification
#
# Exposed to the root RLM's REPL as a custom tool (see `custom_tools=` below) so it can
# check a child's rlm_query() answer exactly, instead of eyeballing set equality.
# =============================================================================


def verify_child(chunk_edges, frontier_in, child_out, excluded, reverse=False):
    """Did the child correctly take ONE relational hop over ITS slice of edges?

    reverse=False -> follow edges forward  (BFS: children of the frontier)
    reverse=True  -> follow edges backward (parents of the frontier)
    """
    frontier_in, excluded = set(frontier_in), set(excluded)

    if reverse:
        chunk_edges = [(v, u) for (u, v) in chunk_edges]

    expected = {v for (u, v) in chunk_edges if u in frontier_in}
    expected -= excluded

    return set(child_out) == expected


# =============================================================================
# Task strategy handed to the root RLM as `root_prompt`
# =============================================================================

TASK_STRATEGY = r"""
This is a GraphWalks task. `context` holds the full problem exactly as given, including a
directed edge list ("A -> B" per line), the requested operation ("Perform a BFS from node
X with depth D" or "Find the parents of node X"), a few worked examples, and the exact
required output format (a final line "Final Answer: [n1, n2, ...]", or
"Final Answer: []" if the result is empty).

Do NOT trace the graph by eye -- it can have dozens of nodes and hundreds of edges. Follow
this procedure in the REPL:

1. Parse the edge list out of `context` with code (e.g. a regex over lines matching
   `(\w+) -> (\w+)`) into a forward adjacency dict {node: set(children)} and a reverse
   adjacency dict {node: set(parents)}. Also parse the target node (and depth, for BFS)
   out of the "Operation:" section.
2. Compute the exact result in plain Python -- this is a deterministic graph algorithm,
   not something to guess: a BFS frontier expansion (only nodes reachable at EXACTLY the
   requested depth, never the start node, never re-including nodes already seen at an
   earlier depth) for BFS operations, or a direct reverse-adjacency lookup for parents
   operations. For BFS, keep each depth's frontier (`frontier_0 = {start}`, `frontier_1`,
   ...) and the running `visited` set (union of all frontiers so far) -- you need both for
   the cross-check in step 3.
3. Before finalizing, cross-check EACH hop with a checkable sub-call, verified exactly
   with `verify_child` (available in your REPL) rather than eyeballing set equality:
   - For a BFS hop from `frontier_d` to `frontier_{d+1}`: build `chunk_edges`, the list of
     (u, v) tuples pulled straight from your parsed edge list where `u` is in
     `frontier_d` (do not include unrelated edges). Render `frontier_d` and `chunk_edges`
     as a small, self-contained sub-problem and call `rlm_query(prompt)`, asking a child
     to compute the children of `frontier_d` over just that edge list, ending in the same
     "Final Answer: [...]" format. Parse its response into `child_out` with
     `extract_answer_nodes`, then call
     `verify_child(chunk_edges, frontier_d, child_out, excluded=visited_before_this_hop, reverse=False)`.
     It returns True only if `child_out` is EXACTLY the correct one-hop result (with nodes
     already visited at an earlier depth excluded).
   - For a "parents" operation: build `chunk_edges` as the direct incoming edges to the
     target (parent -> target tuples), call `rlm_query(prompt)` asking a child to find the
     parents of the target over just those edges, parse its response into `child_out`, and
     call `verify_child(chunk_edges, {target}, child_out, excluded=set(), reverse=True)`.
   - If `verify_child` returns False, the child got that hop wrong -- fall back to your own
     code-computed result for that hop (already computed in step 2); do not use the
     child's answer.
   - If it returns True for every hop, you've cross-validated your answer.
4. Only after this cross-check, set `answer["content"]` to your final response, ending in
   the exact line format `Final Answer: [n1, n2, ...]` (or `Final Answer: []` if empty),
   and set `answer["ready"] = True`.
"""


# =============================================================================
# Runner
# =============================================================================


def run_example(
    row: dict,
    model: str,
    max_depth: int,
    max_iterations: int,
    verbose: bool,
    log_dir: str | None,
) -> tuple[str | None, list[str], dict[str, float], float]:
    """Run one GraphWalks problem through the RLM. Returns (raw_response, predicted, metrics, seconds)."""
    rlm = RLM(
        backend="gemini",
        backend_kwargs={
            "model_name": model,
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=RLMLogger(log_dir=log_dir) if log_dir else None,
        verbose=verbose,
        custom_tools={
            "verify_child": {
                "tool": verify_child,
                "description": (
                    "verify_child(chunk_edges, frontier_in, child_out, excluded, reverse=False) -> bool. "
                    "Deterministically checks whether a child rlm_query() answer is EXACTLY the correct "
                    "one-hop result over chunk_edges (a list of (u, v) edge tuples): the children of "
                    "frontier_in if reverse=False, or the parents of frontier_in if reverse=True, with "
                    "any node ids in excluded removed. Use this instead of comparing sets by eye."
                ),
            },
        },
    )

    try:
        result = rlm.completion(row["prompt"], root_prompt=TASK_STRATEGY)
        response = result.response
        seconds = result.execution_time
    except (
        BudgetExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
        ErrorThresholdExceededError,
    ) as e:
        # BudgetExceededError has no partial_answer field; the others do.
        response = getattr(e, "partial_answer", None)
        seconds = 0.0
        print(f"  ! limit hit: {type(e).__name__}: {e}")

    predicted = extract_answer_nodes(response) if response else []
    metrics = score(predicted, row["answer_nodes"])
    return response, predicted, metrics, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4, help="Number of problems to sample.")
    parser.add_argument(
        "--problem-type", choices=["bfs", "parents", "both"], default="both"
    )
    parser.add_argument(
        "--max-chars", type=int, default=128_000, help="Max prompt_chars to keep (short subset)."
    )
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument(
        "--max-depth", type=int, default=2, help="2 lets rlm_query() spawn a real child RLM."
    )
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Print full RLM iteration trace.")
    parser.add_argument("--log-dir", default=None, help="If set, write JSONL trajectory logs here.")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Set it (e.g. in a .env file) and re-run.")
        sys.exit(1)

    problem_types = ("bfs", "parents") if args.problem_type == "both" else (args.problem_type,)

    print(f"Downloading/filtering {DATASET_FILE} from {DATASET_REPO}...")
    rows = load_graphwalks(
        problem_types=problem_types, max_chars=args.max_chars, limit=args.n, seed=args.seed
    )
    if not rows:
        print("No rows matched the given filters.")
        sys.exit(1)
    print(f"Loaded {len(rows)} problem(s) (<= {args.max_chars:,} chars, types={problem_types})\n")

    results = []
    for i, row in enumerate(rows):
        print(
            f"=== [{i + 1}/{len(rows)}] {row['problem_type']} | "
            f"{row['prompt_chars']:,} chars | golden={row['answer_nodes']} ==="
        )
        _response, predicted, metrics, seconds = run_example(
            row, args.model, args.max_depth, args.max_iterations, args.verbose, args.log_dir
        )
        print(f"Predicted: {predicted}")
        print(
            f"F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  ({seconds:.1f}s)\n"
        )
        results.append({"problem_type": row["problem_type"], **metrics})

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for pt in problem_types:
        subset = [r for r in results if r["problem_type"] == pt]
        if subset:
            avg_f1 = sum(r["f1"] for r in subset) / len(subset)
            print(f"  {pt:8s}  n={len(subset):<3d}  avg F1={avg_f1:.3f}")
    avg_f1_all = sum(r["f1"] for r in results) / len(results)
    print(f"  {'overall':8s}  n={len(results):<3d}  avg F1={avg_f1_all:.3f}")


if __name__ == "__main__":
    main()
