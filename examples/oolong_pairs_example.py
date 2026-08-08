"""
Example: OOLONG-Pairs semantic-pairing reasoning with a recursive RLM.

OOLONG-Pairs (RLM paper, https://arxiv.org/abs/2512.24601, Appendix D.1) is a benchmark the
paper's authors constructed themselves -- there is no public OOLONG-Pairs dataset. It starts
from OpenAI's OOLONG `trec_coarse` split (Bertsch et al. 2025): long lists of TREC
general-knowledge questions, each tagged with a User ID, where every question's answer has a
hidden ground-truth semantic label (one of: description and abstract concept, entity, human
being, numeric value, location, abbreviation). The paper then defines 20 fixed pairwise
questions over those same entries -- e.g. "list all pairs of user IDs where both users have
at least one instance with a numeric value or location" -- and the model must return the
exact set of valid (user_id_1, user_id_2) pairs, not just a count.

`examples/oolong_pairs/` (Stage 1) recreates this benchmark: it downloads the real OOLONG
data, parses out the ground-truth labels, and computes gold pairs programmatically for each
of the 20 tasks (`examples/oolong_pairs/tasks.py`, verbatim from Appendix D.1). This file
(Stage 2) is the RLM harness that answers those questions.

Unlike GraphWalks, there is no deterministic algorithm for the underlying *classification*
here -- mapping a TREC question to one of six semantic categories is a judgment call, not
something code can compute. The root RLM

  1. inspects `context` (the raw, unlabeled entries + the exact task question) with code,
  2. splits the entries into contiguous, configurably-sized chunks,
  3. delegates ONLY the semantic classification of each chunk to a child via `rlm_query()`
     (children never see the task question and never filter, count, or pair anything),
  4. merges every child's per-entry classifications into one dataset-wide mapping,
  5. evaluates the specific task's predicate (parsed from the task question's own English
     wording -- "at least one", "exactly two", a date cutoff, symmetric vs. one-user/
     other-user, etc.) against that mapping in plain Python, and
  6. returns the resulting pairs in the required "(user_id_1, user_id_2)" format.

There IS, however, a deterministic sub-verification signal available for step 3: we (the
harness, not the model) know the real OOLONG ground-truth labels, so `verify_child`
(`examples/oolong_pairs/verification.py`) recomputes -- via the SAME task predicate used to
build the gold answer -- what a chunk's classifications imply about any user that chunk
happens to fully resolve, and diffs that against ground truth with TP/FP/FN/precision/
recall/F1, exposed to the root as a custom tool. It's called immediately after every
`rlm_query()` per the updated TASK_STRATEGY below, purely observationally: the result is
logged but never changes the child's answer, triggers a retry, or otherwise alters execution
-- see the module docstring in `verification.py` for the (fairly fundamental) limitation on
what a single chunk's verification can and can't tell you.

This example downloads the real OOLONG data from Hugging Face (`oolongbench/oolong-synth`),
samples a handful of `trec_coarse` context windows at small context lengths, runs each
(window, task) problem through an `RLM` (Gemini backend, local Python REPL, `max_depth=2` so
`rlm_query()` spawns real child RLMs), and grades the response against the gold pairs
computed in Stage 1.

Usage:
    uv pip install -e ".[oolong]"
    GEMINI_API_KEY=... python -m examples.oolong_pairs_example --n 4

    # Options:
    python -m examples.oolong_pairs_example --n 6 --tasks 1,2,6 --model gemini-2.5-flash \\
        --context-lengths 1024,2048 --chunk-size 20 --verbose --log-dir ./logs/oolong_pairs
"""

import argparse
import os
import re
import sys

from dotenv import load_dotenv

from examples.oolong_pairs.dataset import ALL_CONTEXT_LENGTHS, generate_oolong_pairs
from examples.oolong_pairs.verification import verify_child as _verify_child
from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.exceptions import (
    BudgetExceededError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)

load_dotenv()

DEFAULT_CHUNK_SIZE = 25


# =============================================================================
# Grading
# =============================================================================


def extract_answer_pairs(response: str) -> list[tuple[int, int]]:
    """Pull every "(user_id_1, user_id_2)" pair out of a response, in the order they appear.

    Order within a pair is preserved (not sorted), since the task explicitly requires the
    lower id first -- an out-of-order pair should not silently score as correct.
    """
    return [(int(a), int(b)) for a, b in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", response)]


def score(predicted: list[tuple[int, int]], golden: list[tuple[int, int]]) -> dict[str, float]:
    """Precision/recall/F1 over the set of predicted vs. golden pairs."""
    pred_set, gold_set = set(predicted), set(golden)
    n_overlap = len(pred_set & gold_set)
    n_golden, n_sampled = len(gold_set), len(pred_set)
    recall = n_overlap / n_golden if n_golden > 0 else 1.0 if n_sampled == 0 else 0.0
    precision = n_overlap / n_sampled if n_sampled > 0 else 0.0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Deterministic sub-call verification
#
# Exposed to the root RLM's REPL as a custom tool (see `custom_tools=` below) so it can
# check each rlm_query() classification call against ground truth, instead of trusting it
# blindly. See examples/oolong_pairs/verification.py for the full explanation of what this
# can and can't tell you for a given chunk.
# =============================================================================


def _make_verify_child(gold_by_user: dict, task_id: int):
    """Bind a row's ground truth + task id into a 2-arg tool the model can call directly.

    The model must never see `gold_by_user` itself (that would leak the answer) -- only this
    closure, which takes just `child_scope` and `child_output` and returns the verification
    dict, is exposed via `custom_tools`.
    """

    def verify_child(child_scope: list, child_output: list) -> dict:
        return _verify_child(child_scope, child_output, gold_by_user, task_id)

    return verify_child


# =============================================================================
# Task strategy handed to the root RLM as `root_prompt`
# =============================================================================

TASK_STRATEGY_TEMPLATE = r"""
This is an OOLONG-Pairs task. `context` holds a list of general-knowledge question entries,
one per line, in the form:
    Date: <date> || User: <user_id> || Instance: <question text>
followed by a blank line and then the exact task question you must answer -- a request to
list all pairs of user IDs (no duplicate pairs, lower id first) satisfying some predicate
over each user's HIDDEN semantic label(s): description and abstract concept, entity, human
being, numeric value, location, abbreviation. The entries do NOT provide these labels -- you
(and your sub-calls) must infer each entry's label from the semantics of its question text.

There is no deterministic classifier for this -- semantic labeling is a judgment call, not
something a keyword regex can reliably get right. Delegate EVERY entry's classification to a
child RLM via `rlm_query()`; do not try to guess labels yourself from keywords.

Follow this procedure in the REPL:

1. Inspect `context` with code: print the header and a handful of entry lines, then parse
   out ALL entries with a regex over lines matching
   `Date: (.+?) \|\| User: (\d+) \|\| Instance: (.+)`. Keep them in their original order.
   Also isolate the task question itself (the paragraph after the entries, starting
   "In the above data, ...") into its own variable -- this tells you exactly which predicate
   to evaluate in step 5. Print the parsed entry count, the user id range, and the task
   question so you can read it carefully.
2. Split the parsed entries into contiguous chunks of about {chunk_size} entries each (the
   last chunk may be smaller). Do not shuffle or reorder entries -- chunk boundaries don't
   matter for correctness, but each entry must end up in exactly one chunk.
3. For EVERY chunk, call `rlm_query(prompt)` with a prompt containing ONLY that chunk's raw
   entry lines (Date/User/Instance -- never the task question) and ask the child to classify
   EACH line into exactly one of the six labels, returning ONE JSON object per input line so
   no per-instance information is lost, e.g.:
   `[{{"user_id": 1234, "date": "Feb 02, 2023", "label": "entity"}}, ...]`
   Ask for ONLY that JSON array in the response (no prose, no markdown fences) so you can
   parse it directly with `json.loads` (strip code fences defensively if present anyway).
   The child's ONLY job is classification -- it must never filter, count, or pair up users;
   that all happens later, in your own code.

   Immediately after each `rlm_query()` call returns and you've parsed its JSON into
   `child_output`, call `verify_child(chunk, child_output)` (available in your REPL; `chunk`
   is that same list of (date, user_id, instance) tuples you just sent). Append the result to
   a running list, e.g. `verification_log.append(verify_child(chunk, child_output))`, and
   print a short summary for that chunk: the returned `pass` (True/False/None -- None means
   this chunk couldn't resolve enough users to judge any pair, which is expected and NOT an
   error), and if `pass` is False, its `fp`/`fn` pairs. This is purely observational: do NOT
   change `child_output`, retry the call, or otherwise alter what you do with it based on the
   verification result -- keep using the child's classifications exactly as returned, and
   simply carry on to the next chunk.
4. Collect every child's parsed JSON list and concatenate them into one list covering every
   entry in the dataset. Do NOT deduplicate or collapse per user yet -- keep one record per
   original entry. A user can have several instances, possibly with different labels, and
   several tasks care about exact per-label counts or per-label dates, not just which labels
   a user has "some" instance of.
5. From the merged list, build `by_user: dict[user_id, list[(label, date)]]`, parsing each
   date with `datetime.strptime(date_str, "%b %d, %Y").date()`. Re-read the task question you
   isolated in step 1 carefully and note:
   - Is it symmetric ("both users have...") or asymmetric ("one user has..., and the other
     user has...")?
   - For each label condition: is it "at least one", "at least N", or "exactly N" instances?
   - Is there a date filter ("before"/"after" some date, applying to ALL of a user's
     instances of one specific label)?
   Write plain Python code implementing exactly that predicate over `by_user`, then enumerate
   every valid pair of distinct users: no duplicate pairs, lower user id first within each
   pair, and the final list deduplicated and sorted (e.g. `sorted(set(pairs))`).
6. Before finalizing, print a one-line summary of `verification_log` (e.g. how many chunks
   passed / failed / were inconclusive) so it's visible in your trajectory -- this is a
   record only, not a gate: proceed to submit your answer regardless of what it says.
7. Only after computing the final pair list, set `answer["content"]` to the pairs in the
   exact required format -- one pair per line, `(user_id_1, user_id_2)`, nothing else (or the
   single line `No valid pairs found.` if the list is empty) -- and set
   `answer["ready"] = True`.

Important: recursive `rlm_query()` calls must ONLY perform semantic classification of the
entries handed to them. They must never see the task question, and must never filter, count,
compare dates, or return pairs -- all of that happens in your own REPL code in steps 4-5,
over the merged mapping. `verify_child` is purely observational (step 3) -- it never changes
what you do with a child's output.
"""


# =============================================================================
# Runner
# =============================================================================


def run_example(
    row: dict,
    model: str,
    max_depth: int,
    max_iterations: int,
    chunk_size: int,
    verbose: bool,
    log_dir: str | None,
) -> tuple[str | None, list[tuple[int, int]], dict[str, float], float]:
    """Run one OOLONG-Pairs problem through the RLM. Returns (raw_response, predicted, metrics, seconds)."""
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
                "tool": _make_verify_child(row["gold_by_user"], row["task_id"]),
                "description": (
                    "verify_child(child_scope, child_output) -> dict. Deterministically "
                    "checks a chunk's rlm_query() classification output against the real "
                    "OOLONG ground truth, for whichever users that chunk fully resolves. "
                    "child_scope: the list of (date, user_id, instance) tuples you sent to "
                    "the child. child_output: the child's parsed JSON classification list. "
                    "Returns tp/fp/fn (pairs), precision/recall/f1, and pass (True/False, or "
                    "None if this chunk didn't resolve enough users to judge any pair -- "
                    "expected and not an error). Observational only -- use it to log "
                    "verification results, not to alter what you do with a child's output."
                ),
            },
        },
    )

    try:
        result = rlm.completion(
            row["prompt"],
            root_prompt=TASK_STRATEGY_TEMPLATE.format(chunk_size=chunk_size),
        )
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

    predicted = extract_answer_pairs(response) if response else []
    metrics = score(predicted, row["gold_pairs"])
    return response, predicted, metrics, seconds


def _parse_task_ids(raw: str) -> list[int] | None:
    if raw == "all":
        return None
    return [int(t) for t in raw.split(",") if t.strip()]


def _parse_context_lengths(raw: str) -> tuple[int, ...]:
    return tuple(int(c) for c in raw.split(",") if c.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4, help="Number of (window, task) problems to sample.")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated OOLONG-Pairs task ids (1-20) to sample from, or 'all'.",
    )
    parser.add_argument(
        "--context-lengths",
        default="1024,2048",
        help=f"Comma-separated OOLONG context lengths to sample windows from (subset of {ALL_CONTEXT_LENGTHS}).",
    )
    parser.add_argument(
        "--n-windows", type=int, default=2, help="Distinct context windows to sample per context length."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Approx. number of entries per rlm_query() classification chunk (Stage 2 Step 2).",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
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

    task_ids = _parse_task_ids(args.tasks)
    context_lengths = _parse_context_lengths(args.context_lengths)

    print(f"Downloading/sampling OOLONG trec_coarse windows from {args.split} split...")
    rows = generate_oolong_pairs(
        task_ids=task_ids,
        context_lengths=context_lengths,
        n_windows=args.n_windows,
        split=args.split,
        seed=args.seed,
    )[: args.n]
    if not rows:
        print("No rows matched the given filters.")
        sys.exit(1)
    print(f"Loaded {len(rows)} problem(s) (context_lengths={context_lengths}, tasks={args.tasks})\n")

    results = []
    for i, row in enumerate(rows):
        print(
            f"=== [{i + 1}/{len(rows)}] task {row['task_id']} | "
            f"context_len={row['context_len']:,} | {row['num_users']} users | "
            f"gold={row['gold_pairs']} ==="
        )
        _response, predicted, metrics, seconds = run_example(
            row,
            args.model,
            args.max_depth,
            args.max_iterations,
            args.chunk_size,
            args.verbose,
            args.log_dir,
        )
        print(f"Predicted: {predicted}")
        print(
            f"F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  ({seconds:.1f}s)\n"
        )
        results.append({"task_id": row["task_id"], **metrics})

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task_id in sorted({r["task_id"] for r in results}):
        subset = [r for r in results if r["task_id"] == task_id]
        avg_f1 = sum(r["f1"] for r in subset) / len(subset)
        print(f"  task {task_id:<3d}  n={len(subset):<3d}  avg F1={avg_f1:.3f}")
    avg_f1_all = sum(r["f1"] for r in results) / len(results)
    print(f"  {'overall':8s}  n={len(results):<3d}  avg F1={avg_f1_all:.3f}")


if __name__ == "__main__":
    main()
