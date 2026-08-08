"""
Deterministic sub-verification for OOLONG-Pairs recursive `rlm_query()` calls.

Sibling to GraphWalks' `verify_child` (`examples/graphwalks_example.py`), and built the same
way: recompute the correct answer for a sub-call in plain Python from data already available,
then diff it against what the child returned with set operations -- rather than asking
another LLM to judge the LLM.

DECOMPOSITION LIMITATION (read this before extending): in `examples/oolong_pairs_example.py`,
a recursive child NEVER solves a pairs sub-problem. Per TASK_STRATEGY, it only classifies the
raw entries in one contiguous chunk (step 3) and returns per-entry `{"user_id", "date",
"label"}` records -- it never sees the task question, and never filters, counts, or pairs
anything. A pair (u, v) can only be judged once BOTH users' COMPLETE entry histories are
known: most task predicates need more than "at least one" instance to be judged safely
(exact counts, or "all instances of label X must be before/after date D"), so a partial view
of a user is not enough. A single contiguous chunk essentially never contains every entry for
every user it touches, since chunking is oblivious to user identity -- so in general there is
no sound way to define "the expected pairs for this child" from its chunk alone.

Rather than inventing a new, user-scoped decomposition to make full pair-level verification
always possible (explicitly out of scope for this iteration), `verify_child` below verifies
what IS deterministically knowable from a single chunk: for whichever users this chunk
happens to fully resolve -- every entry that user has anywhere in the window is inside this
one chunk, which is common for low-frequency users since chunking is not user-aware -- it
applies THE SAME task predicate (`tasks.compute_gold_pairs`) used to build the OOLONG-Pairs
gold answer to (a) the real ground-truth labels and (b) the child's own classifications, and
diffs the resulting pair sets with TP/FP/FN/precision/recall/F1. When a chunk resolves fewer
than 2 users, no pair can be judged from it at all, and `verify_child` says so explicitly
(`pass: None`, plus a `note`) instead of reporting a vacuous, misleading "PASS".

Full pair-level correctness of the ROOT's final merged answer is judged separately, once,
over the complete dataset, by the existing global evaluation in `score()` -- unaffected by
this module. This is purely an additional, observational, per-child signal: nothing here
changes a child's answer, retries it, or feeds back into the decomposition.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any

from examples.oolong_pairs.tasks import compute_gold_pairs

# user_id -> [(label, date), ...] -- one entry per instance that user has.
GoldByUser = dict[int, list[tuple[str, date]]]


def _prf1(expected: set[tuple], actual: set[tuple]) -> dict[str, Any]:
    """TP/FP/FN/precision/recall/F1 between two sets of tuples, via set operations.

    Symmetric vacuous-truth convention: if both sets are empty, precision/recall/F1 are all
    1.0 (nothing to find, nothing wrongly found).
    """
    tp = actual & expected
    fp = actual - expected
    fn = expected - actual
    precision = len(tp) / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = len(tp) / len(expected) if expected else (1.0 if not actual else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else (1.0 if not expected and not actual else 0.0)
    )
    return {
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pass": not fp and not fn,
    }


def _parse_child_date(raw: object) -> date | None:
    """Parse a child-returned date string ("Feb 02, 2023"); None if malformed/missing."""
    try:
        return datetime.strptime(str(raw).strip(), "%b %d, %Y").date()
    except (ValueError, TypeError):
        return None


def verify_child(
    child_scope: list[tuple[str, int | str, str]],
    child_output: list[dict],
    gold_data: GoldByUser,
    task: int,
) -> dict[str, Any]:
    """Deterministically verify one `rlm_query()` classification sub-call.

    Args:
        child_scope: the exact entries handed to the child, as (date, user_id, instance)
            triples -- the same rows sliced out of `context` in TASK_STRATEGY step 2.
        child_output: the child's parsed JSON classification list, e.g.
            `[{"user_id": 1234, "date": "Feb 02, 2023", "label": "entity"}, ...]`.
        gold_data: the FULL window's ground-truth `user_id -> [(label, date), ...]` mapping
            (Stage 1's `by_user`, i.e. a row's `gold_by_user`) -- not just this chunk's slice.
        task: the OOLONG-Pairs task id (1-20) currently being answered -- the SAME task
            passed to `tasks.compute_gold_pairs` when Stage 1 computed the gold answer.

    Returns:
        A dict with `tp`/`fp`/`fn` (pair lists), `precision`/`recall`/`f1`, `pass`, and
        `resolved_users` (which users this chunk fully resolved). When fewer than 2 users are
        fully resolved, `precision`/`recall`/`f1`/`pass` are all `None` and a `note` explains
        why -- see the module docstring; this is an expected decomposition limitation, not an
        error.
    """
    scope_counts = Counter(int(user_id) for _date, user_id, _instance in child_scope)
    resolved_users = {
        user_id
        for user_id, count in scope_counts.items()
        if count == len(gold_data.get(user_id, []))
    }

    if len(resolved_users) < 2:
        return {
            "tp": [],
            "fp": [],
            "fn": [],
            "precision": None,
            "recall": None,
            "f1": None,
            "pass": None,
            "resolved_users": sorted(resolved_users),
            "note": (
                "Fewer than 2 users in this chunk have their COMPLETE entry history inside "
                "this single chunk, so no pair can be soundly judged from it alone. This is "
                "an expected decomposition limitation (see verify_child's docstring), not a "
                "child error."
            ),
        }

    # The child's own classifications, restricted to fully-resolved users only -- comparing
    # a predicate elsewhere would be unfair, since the child never saw those users' other
    # instances.
    child_by_user: GoldByUser = {}
    for rec in child_output:
        if not isinstance(rec, dict) or "user_id" not in rec:
            continue
        try:
            user_id = int(rec["user_id"])
        except (TypeError, ValueError):
            continue
        if user_id not in resolved_users:
            continue
        parsed_date = _parse_child_date(rec.get("date"))
        if parsed_date is None:
            continue
        label = str(rec.get("label", "")).strip()
        child_by_user.setdefault(user_id, []).append((label, parsed_date))

    # Same task predicate, applied to (a) real ground truth and (b) the child's own
    # classifications, restricted to pairs where BOTH members are fully resolved here.
    gold_pairs = compute_gold_pairs(task, gold_data)
    actual_pairs = compute_gold_pairs(task, child_by_user)
    expected = {(u, v) for u, v in gold_pairs if u in resolved_users and v in resolved_users}
    actual = {(u, v) for u, v in actual_pairs if u in resolved_users and v in resolved_users}

    result = _prf1(expected, actual)
    result["resolved_users"] = sorted(resolved_users)
    return result
