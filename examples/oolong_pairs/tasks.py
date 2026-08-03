"""
The 20 OOLONG-Pairs tasks, verbatim from Appendix D.1 of the RLM paper
(https://arxiv.org/abs/2512.24601), plus the programmatic gold-answer logic for each.

`TASK_TEXTS` holds the exact task wording (copied from `docs/oolong-pairs-details.md`,
which is itself a direct transcription of the appendix) -- do not paraphrase or reword
these. `compute_gold_pairs` independently computes the correct answer for a task from the
ground-truth (label, date) instances per user, using the OOLONG `trec_coarse` labels that
are hidden from the RLM but known to us from the dataset's labeled context field. This is
grading logic only -- the RLM itself must infer labels from the raw text via `rlm_query()`,
per the task's own instruction that "the data does not provide the labels".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

# The 6 TREC-coarse labels used throughout OOLONG / OOLONG-Pairs, exactly as they appear
# in the dataset's `context_window_text_with_labels` field and in each task's wording.
LABELS: tuple[str, ...] = (
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation",
)

# Shared suffix, verbatim and identical across all 20 tasks.
_LABEL_SUFFIX = (
    "Each of the questions can be labelled as one of the labels (the data does not provide "
    "the labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines."
)

# Exact task wording from Appendix D.1. Do not modify.
TASK_TEXTS: dict[int, str] = {
    1: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a numeric value or "
        f"location. {_LABEL_SUFFIX}"
    ),
    2: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with an entity or human being. "
        f"{_LABEL_SUFFIX}"
    ),
    3: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a description and abstract "
        f"concept or abbreviation. {_LABEL_SUFFIX}"
    ),
    4: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a human being or location, "
        "and all instances that are a human being for both users must be after January 6, "
        f"2023. {_LABEL_SUFFIX}"
    ),
    5: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with an entity or numeric value, "
        "and all instances that are an entity for both users must be before March 15, 2023. "
        f"{_LABEL_SUFFIX}"
    ),
    6: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        f"first) where both users have at least one instance with a location or abbreviation. "
        f"{_LABEL_SUFFIX}"
    ),
    7: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a description and abstract "
        "concept or numeric value, and all instances that are a numeric value for both users "
        f"must be after February 1, 2023. {_LABEL_SUFFIX}"
    ),
    8: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a human being or description "
        f"and abstract concept. {_LABEL_SUFFIX}"
    ),
    9: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with an entity or location, and "
        "all instances that are a location for both users must be after April 10, 2023. "
        f"{_LABEL_SUFFIX}"
    ),
    10: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) where both users have at least one instance with a numeric value or "
        "abbreviation, and all instances that are an abbreviation for both users must be "
        f"before May 20, 2023. {_LABEL_SUFFIX}"
    ),
    11: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with entity and one with "
        f"abbreviation, and the other user has exactly one instance with entity. {_LABEL_SUFFIX}"
    ),
    12: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least two instances with numeric value, and the "
        "other user has at least one instance with location and at least one instance with "
        f"human being. {_LABEL_SUFFIX}"
    ),
    13: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has exactly one instance with description and abstract "
        "concept, and the other user has at least one instance with abbreviation and at "
        f"least one instance with entity. {_LABEL_SUFFIX}"
    ),
    14: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with human being and at least "
        "one instance with numeric value, and the other user has exactly two instances with "
        f"location. {_LABEL_SUFFIX}"
    ),
    15: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with entity, at least one "
        "instance with location, and at least one instance with abbreviation, and the other "
        f"user has exactly one instance with numeric value. {_LABEL_SUFFIX}"
    ),
    16: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with description and abstract "
        "concept and at least one instance with human being, and the other user has at least "
        "two instances with entity and exactly one instance with abbreviation. "
        f"{_LABEL_SUFFIX}"
    ),
    17: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has exactly one instance with numeric value, and the "
        "other user has at least one instance with location and at least one instance with "
        f"description and abstract concept. {_LABEL_SUFFIX}"
    ),
    18: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with abbreviation and exactly "
        "one instance with human being, and the other user has at least one instance with "
        f"entity and at least one instance with numeric value. {_LABEL_SUFFIX}"
    ),
    19: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least two instances with location and at least one "
        "instance with entity, and the other user has exactly one instance with description "
        f"and abstract concept and exactly one instance with abbreviation. {_LABEL_SUFFIX}"
    ),
    20: (
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID "
        "first) such that one user has at least one instance with numeric value and at least "
        "one instance with human being, and the other user has at least one instance with "
        "location, at least one instance with entity, and exactly one instance with "
        f"abbreviation. {_LABEL_SUFFIX}"
    ),
}

# Per-user (label, date) instances -- the ground-truth semantic mapping a task predicate is
# evaluated against.
UserInstances = list[tuple[str, date]]


def _count(instances: UserInstances, label: str) -> int:
    return sum(1 for l, _ in instances if l == label)


def _has(instances: UserInstances, label: str) -> bool:
    return _count(instances, label) >= 1


def _dates(instances: UserInstances, label: str) -> list[date]:
    return [d for l, d in instances if l == label]


# --- Tasks 1-10: symmetric "both users satisfy the same eligibility predicate" ---


def _eligible_1(i: UserInstances) -> bool:
    return _has(i, "numeric value") or _has(i, "location")


def _eligible_2(i: UserInstances) -> bool:
    return _has(i, "entity") or _has(i, "human being")


def _eligible_3(i: UserInstances) -> bool:
    return _has(i, "description and abstract concept") or _has(i, "abbreviation")


def _eligible_4(i: UserInstances) -> bool:
    base = _has(i, "human being") or _has(i, "location")
    clean = all(d > date(2023, 1, 6) for d in _dates(i, "human being"))
    return base and clean


def _eligible_5(i: UserInstances) -> bool:
    base = _has(i, "entity") or _has(i, "numeric value")
    clean = all(d < date(2023, 3, 15) for d in _dates(i, "entity"))
    return base and clean


def _eligible_6(i: UserInstances) -> bool:
    return _has(i, "location") or _has(i, "abbreviation")


def _eligible_7(i: UserInstances) -> bool:
    base = _has(i, "description and abstract concept") or _has(i, "numeric value")
    clean = all(d > date(2023, 2, 1) for d in _dates(i, "numeric value"))
    return base and clean


def _eligible_8(i: UserInstances) -> bool:
    return _has(i, "human being") or _has(i, "description and abstract concept")


def _eligible_9(i: UserInstances) -> bool:
    base = _has(i, "entity") or _has(i, "location")
    clean = all(d > date(2023, 4, 10) for d in _dates(i, "location"))
    return base and clean


def _eligible_10(i: UserInstances) -> bool:
    base = _has(i, "numeric value") or _has(i, "abbreviation")
    clean = all(d < date(2023, 5, 20) for d in _dates(i, "abbreviation"))
    return base and clean


# --- Tasks 11-20: asymmetric "one user satisfies role A, the other satisfies role B" ---


def _role_11a(i: UserInstances) -> bool:
    return _has(i, "entity") and _has(i, "abbreviation")


def _role_11b(i: UserInstances) -> bool:
    return _count(i, "entity") == 1


def _role_12a(i: UserInstances) -> bool:
    return _count(i, "numeric value") >= 2


def _role_12b(i: UserInstances) -> bool:
    return _has(i, "location") and _has(i, "human being")


def _role_13a(i: UserInstances) -> bool:
    return _count(i, "description and abstract concept") == 1


def _role_13b(i: UserInstances) -> bool:
    return _has(i, "abbreviation") and _has(i, "entity")


def _role_14a(i: UserInstances) -> bool:
    return _has(i, "human being") and _has(i, "numeric value")


def _role_14b(i: UserInstances) -> bool:
    return _count(i, "location") == 2


def _role_15a(i: UserInstances) -> bool:
    return _has(i, "entity") and _has(i, "location") and _has(i, "abbreviation")


def _role_15b(i: UserInstances) -> bool:
    return _count(i, "numeric value") == 1


def _role_16a(i: UserInstances) -> bool:
    return _has(i, "description and abstract concept") and _has(i, "human being")


def _role_16b(i: UserInstances) -> bool:
    return _count(i, "entity") >= 2 and _count(i, "abbreviation") == 1


def _role_17a(i: UserInstances) -> bool:
    return _count(i, "numeric value") == 1


def _role_17b(i: UserInstances) -> bool:
    return _has(i, "location") and _has(i, "description and abstract concept")


def _role_18a(i: UserInstances) -> bool:
    return _has(i, "abbreviation") and _count(i, "human being") == 1


def _role_18b(i: UserInstances) -> bool:
    return _has(i, "entity") and _has(i, "numeric value")


def _role_19a(i: UserInstances) -> bool:
    return _count(i, "location") >= 2 and _has(i, "entity")


def _role_19b(i: UserInstances) -> bool:
    return _count(i, "description and abstract concept") == 1 and _count(i, "abbreviation") == 1


def _role_20a(i: UserInstances) -> bool:
    return _has(i, "numeric value") and _has(i, "human being")


def _role_20b(i: UserInstances) -> bool:
    return _has(i, "location") and _has(i, "entity") and _count(i, "abbreviation") == 1


def _symmetric_pairs(
    by_user: dict[int, UserInstances], eligible: Callable[[UserInstances], bool]
) -> list[tuple[int, int]]:
    """Every pair of distinct users that both individually satisfy `eligible`."""
    elig_users = sorted(u for u, instances in by_user.items() if eligible(instances))
    return [(a, b) for idx, a in enumerate(elig_users) for b in elig_users[idx + 1 :]]


def _asymmetric_pairs(
    by_user: dict[int, UserInstances],
    role_a: Callable[[UserInstances], bool],
    role_b: Callable[[UserInstances], bool],
) -> list[tuple[int, int]]:
    """Every pair {u, v} where one of u/v satisfies role_a and the other satisfies role_b."""
    role_a_users = {u for u, instances in by_user.items() if role_a(instances)}
    role_b_users = {u for u, instances in by_user.items() if role_b(instances)}
    pairs = set()
    for u in role_a_users:
        for v in role_b_users:
            if u == v:
                continue
            pairs.add((u, v) if u < v else (v, u))
    return sorted(pairs)


_GOLD_FNS: dict[int, Callable[[dict[int, UserInstances]], list[tuple[int, int]]]] = {
    1: lambda by_user: _symmetric_pairs(by_user, _eligible_1),
    2: lambda by_user: _symmetric_pairs(by_user, _eligible_2),
    3: lambda by_user: _symmetric_pairs(by_user, _eligible_3),
    4: lambda by_user: _symmetric_pairs(by_user, _eligible_4),
    5: lambda by_user: _symmetric_pairs(by_user, _eligible_5),
    6: lambda by_user: _symmetric_pairs(by_user, _eligible_6),
    7: lambda by_user: _symmetric_pairs(by_user, _eligible_7),
    8: lambda by_user: _symmetric_pairs(by_user, _eligible_8),
    9: lambda by_user: _symmetric_pairs(by_user, _eligible_9),
    10: lambda by_user: _symmetric_pairs(by_user, _eligible_10),
    11: lambda by_user: _asymmetric_pairs(by_user, _role_11a, _role_11b),
    12: lambda by_user: _asymmetric_pairs(by_user, _role_12a, _role_12b),
    13: lambda by_user: _asymmetric_pairs(by_user, _role_13a, _role_13b),
    14: lambda by_user: _asymmetric_pairs(by_user, _role_14a, _role_14b),
    15: lambda by_user: _asymmetric_pairs(by_user, _role_15a, _role_15b),
    16: lambda by_user: _asymmetric_pairs(by_user, _role_16a, _role_16b),
    17: lambda by_user: _asymmetric_pairs(by_user, _role_17a, _role_17b),
    18: lambda by_user: _asymmetric_pairs(by_user, _role_18a, _role_18b),
    19: lambda by_user: _asymmetric_pairs(by_user, _role_19a, _role_19b),
    20: lambda by_user: _asymmetric_pairs(by_user, _role_20a, _role_20b),
}


def compute_gold_pairs(task_id: int, by_user: dict[int, UserInstances]) -> list[tuple[int, int]]:
    """Programmatically compute the gold (user_id_1, user_id_2) pairs for `task_id`.

    `by_user` maps each user id to the list of (label, date) instances belonging to that
    user, built from the OOLONG ground-truth labels. Pairs are deduplicated, lower id first,
    and returned in sorted order.
    """
    if task_id not in _GOLD_FNS:
        raise ValueError(f"Unknown OOLONG-Pairs task_id: {task_id} (expected 1-20)")
    return _GOLD_FNS[task_id](by_user)
