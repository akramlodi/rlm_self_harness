"""
Example: RULER long-context retrieval/tracing/aggregation with a recursive RLM.

RULER (NVIDIA, https://arxiv.org/abs/2404.06654) is a synthetic long-context suite.
This example demonstrates the 3 task families implemented in
`shrlm.environments.ruler` (Retrieval/NIAH, Variable Tracking, and Aggregation)
fully in-process -- there is no dataset to download, every instance is generated
on the fly.

Each instance's prompt already carries two contracts appended by the generator
(see `shrlm.environments.ruler.ANSWER_FORMAT_CONTRACT` /
`RULER_SUBCALL_CONTRACT`): how to format the final answer, and how any sub-call
the root delegates work to must self-report a checkable `LOCAL FINDING:` line.
Grading uses the same `RulerVerifier` the mining/eval pipeline uses, so a score
here is directly comparable to what an optimization round would measure.

Usage:
    GEMINI_API_KEY=... uv run python -m examples.ruler_example --n 4

    # Options:
    uv run python -m examples.ruler_example --n 8 --task-type variable_tracking \\
        --target-tokens 20000 --model gemini-2.5-flash --max-depth 2 --verbose
"""

import argparse
import os
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
from shrlm.environments.ruler import (
    DEFAULT_TASK_TYPES,
    RulerVerifier,
    generate_ruler_instances,
)

load_dotenv()

TASK_STRATEGY = r"""
This is a RULER long-context task. `context` holds the full problem text: a long
haystack (needle sentences, variable assignment lines, or a comma-separated word
stream, depending on the task) followed by a question, an answer-format contract,
and a sub-call self-report contract.

Do NOT try to read the whole haystack by eye -- it can be tens of thousands of
words. Follow this procedure:

1. Read the question carefully: it names either a specific key/word/variable to
   look up, or asks for an aggregate (top-K most frequent words, or every
   variable transitively equal to a value).
2. If `context` is short enough to search directly with code (e.g. a regex over
   lines, or `str.count()` over the word stream), just do that -- there is no
   need to recurse on an instance that fits comfortably in one pass.
3. If `context` is too large for one pass, split it into contiguous chunks and
   hand each chunk to a child via `rlm_query()`, WITH the exact item(s) to check
   named explicitly in the child's prompt. The prompt's own sub-call contract
   requires that child to end its response with one line per item, in the form
   `LOCAL FINDING: <item> = <value-or-count-or-NONE>` -- follow that contract
   exactly when you phrase the child's task, and parse each child's
   `LOCAL FINDING:` line(s) back out of its response with a regex.
4. Combine the children's local findings into the final answer: for a
   single-key lookup, take the one non-NONE value found; for a multi-value or
   multi-query lookup, take the union across all children; for aggregation
   (top-K words), sum each word's per-chunk counts across children and rank;
   for variable tracking, take the union of variable names any child reported
   PRESENT.
5. Only after that, set `answer["content"]` to your final response, ending in
   the exact final-line format the prompt's answer contract describes, and set
   `answer["ready"] = True`.
"""


def run_example(
    instance: dict,
    model: str,
    max_depth: int,
    max_iterations: int,
    verbose: bool,
    log_dir: str | None,
) -> tuple[str | None, float]:
    """Run one RULER instance through the RLM. Returns (raw_response, seconds)."""
    rlm = RLM(
        backend="gemini",
        backend_kwargs={"model_name": model, "api_key": os.getenv("GEMINI_API_KEY")},
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=RLMLogger(log_dir=log_dir) if log_dir else None,
        verbose=verbose,
    )
    try:
        result = rlm.completion(instance["prompt"], root_prompt=TASK_STRATEGY)
        return result.response, result.execution_time
    except (
        BudgetExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
        ErrorThresholdExceededError,
    ) as e:
        print(f"  ! limit hit: {type(e).__name__}: {e}")
        return getattr(e, "partial_answer", None), 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4, help="Number of instances to sample.")
    parser.add_argument("--task-type", choices=[*DEFAULT_TASK_TYPES, "all"], default="all")
    parser.add_argument(
        "--target-tokens", type=int, default=8_000, help="Approximate haystack size."
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

    task_types = DEFAULT_TASK_TYPES if args.task_type == "all" else (args.task_type,)
    instances = generate_ruler_instances(
        task_types, target_tokens=args.target_tokens, limit=args.n, seed=args.seed
    )
    print(
        f"Generated {len(instances)} instance(s) (~{args.target_tokens:,} tokens, types={task_types})\n"
    )

    verifier = RulerVerifier()
    results = []
    for i, instance in enumerate(instances):
        print(
            f"=== [{i + 1}/{len(instances)}] {instance['task_type']} | "
            f"{instance['prompt_chars']:,} chars | gold={instance['gold']} ==="
        )
        response, seconds = run_example(
            instance, args.model, args.max_depth, args.max_iterations, args.verbose, args.log_dir
        )
        verdict = verifier(instance, response or "")
        status = "PASS" if verdict.passed else f"FAIL ({verdict.cause})"
        print(f"Produced: {verdict.produced}")
        print(f"{status}  {verdict.detail}  ({seconds:.1f}s)\n")
        results.append({"task_type": instance["task_type"], "passed": verdict.passed})

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task_type in task_types:
        subset = [r for r in results if r["task_type"] == task_type]
        if subset:
            n_pass = sum(r["passed"] for r in subset)
            print(f"  {task_type:28s}  n={len(subset):<3d}  pass={n_pass}/{len(subset)}")
    n_pass_all = sum(r["passed"] for r in results)
    print(f"  {'overall':28s}  n={len(results):<3d}  pass={n_pass_all}/{len(results)}")


if __name__ == "__main__":
    main()
