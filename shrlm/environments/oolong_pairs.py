"""
OOLONG-Pairs environment: instances and a mining-side Verifier.

OOLONG-Pairs (RLM paper, https://arxiv.org/abs/2512.24601, Appendix D.1) has no
public dataset: it is recreated by filtering the upstream OOLONG data
(``oolongbench/oolong-synth`` on Hugging Face, ``trec_coarse`` subset) for
distinct context windows and asking 20 fixed pairwise questions over each. The
model must return the exact set of valid ``(user_id_1, user_id_2)`` pairs, one
per line, lower id first. Set-valued gold with a deterministic extraction rule
makes it a drop-in peer of GraphWalks for the weakness miner.

The generation, task, and grading logic here is PORTED (copied and adapted)
from ``examples/oolong_pairs/dataset.py``, ``examples/oolong_pairs/tasks.py``,
and ``examples/oolong_pairs_example.py`` rather than imported: those are demo
modules under ``examples/``, and importing them would couple library code to a
demo entry point. The 20 task texts are verbatim from Appendix D.1 -- do not
paraphrase or reword them. One deliberate difference is documented on
``extract_answer_pairs``: the example returns ``[]`` both for "nothing
parseable" and for an explicit empty answer, while the Verifier needs to
distinguish a format failure from an explicitly empty pair set.

Tokens, not characters: the upstream ``context_len`` field counts TOKENS.
OOLONG defines its context windows at the power-of-two token lengths
``ALL_CONTEXT_LENGTHS`` enumerates (1024 through 1048576), and nothing in the
ported code ever measures characters against the field -- it is used purely as
an upstream label filter. The configured ``context_length_short=8192`` and
``context_length_long=262144`` therefore select 8k-token and 262k-token
windows. The rebuilt prompt strips each entry's ``|| Label: ...`` suffix, so
its CHARACTER length is on the order of a few (roughly chars-per-token, ~4)
times ``context_len``, slightly below that for the stripped labels.

Determinism: window selection is seed-independent and pinned to upstream
order. ``collect_windows`` streams the dataset (at a pinned ``revision``) and
keeps the FIRST ``n_per_length`` distinct windows per requested length, in
stream order, deduplicated by ``(context_window_id, context_len)``, scanning
at most ``max_scan`` raw rows. The seed only shuffles the resulting
(window, task) rows. Same (task_ids, context_lengths, n, seed, max_scan,
split, revision) -> byte-identical instance list, ids included.

Lengths cannot be minted: the generator FILTERS existing upstream rows by
their ``context_len``, so a requested length is only satisfiable if upstream
ships enough distinct windows at it. ``verify_window_coverage`` checks that
before any instances are produced, and an unsatisfiable request raises naming
the available inventory.

There is deliberately NO sub-verifier for this environment: mapping a TREC
question to one of six semantic categories is a judgment call, not something
code can recompute (the stance documented in ``examples/oolong_pairs_example
.py``). Mining-side code already treats absent sub-verification as
all-ungrounded.

Resource exceptions never reach the Verifier: its signature receives only a
produced string, and RESOURCE_TERMINATED is owned by the experiment driver.
"""

import hashlib
import queue
import random
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from shrlm.experiment.config import OolongPairsConfig
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict

DATASET_REPO = "oolongbench/oolong-synth"
OOLONG_SUBSET = "trec_coarse"

# All context lengths OOLONG-Pairs is defined over (Appendix D.1), in TOKENS.
# The generator can only filter upstream rows to these lengths, never invent
# new ones.
ALL_CONTEXT_LENGTHS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)

# The 6 TREC-coarse labels used throughout OOLONG / OOLONG-Pairs, exactly as
# they appear in the dataset's `context_window_text_with_labels` field and in
# each task's wording.
LABELS: tuple[str, ...] = (
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation",
)

# Matches a raw OOLONG entry line, with or without the trailing label:
#   "Date: Feb 02, 2023 || User: 76153 || Instance: What is ... ?"
#   "Date: Feb 02, 2023 || User: 76153 || Instance: ... ? || Label: entity"
_ENTRY_RE = re.compile(
    r"^Date: (?P<date>.+?) \|\| User: (?P<user_id>\d+) \|\| Instance: (?P<instance>.+?)"
    r"(?: \|\| Label: (?P<label>.+))?$"
)

_HEADER_TEMPLATE = (
    "The following lines contain {n} general-knowledge questions, one per line. Each line "
    "has a Date, a User ID, and an Instance (the question text). Each question's answer can "
    "be described as one of 6 categories: 'abbreviation', 'entity', 'human being', "
    "'numeric value', 'location', 'description and abstract concept'.\n\n"
)

# One "(user_id_1, user_id_2)" pair anywhere in a response.
_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
# The explicit empty answer the task format prescribes ("No valid pairs
# found."), matched loosely enough to catch minor rephrasings of the marker.
_NO_PAIRS_RE = re.compile(r"\bno\s+(?:valid\s+)?pairs\b", re.IGNORECASE)


# =============================================================================
# The 20 tasks, verbatim from Appendix D.1 (ported from examples/oolong_pairs/
# tasks.py). Do not modify the wording.
# =============================================================================

# Shared suffix, verbatim and identical across all 20 tasks.
_LABEL_SUFFIX = (
    "Each of the questions can be labelled as one of the labels (the data does not provide "
    "the labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines."
)

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

# Per-user (label, date) instances -- the ground-truth semantic mapping a task
# predicate is evaluated against.
UserInstances = list[tuple[str, date]]


def _count(instances: UserInstances, label: str) -> int:
    return sum(1 for lbl, _ in instances if lbl == label)


def _has(instances: UserInstances, label: str) -> bool:
    return _count(instances, label) >= 1


def _dates(instances: UserInstances, label: str) -> list[date]:
    return [d for lbl, d in instances if lbl == label]


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


# Fail-fast ceiling on the number of gold pairs one (window, task) may produce.
# Pair construction is quadratic in the number of eligible users in a window,
# and nothing else bounds that count: `max_scan` limits raw dataset ROWS, not
# users or pairs. The shipped config requests `context_length_long = 262144`
# tokens, and several tasks (1-10) have single-label-OR eligibility, so a
# schema-valid long window can plausibly carry tens of thousands of eligible
# users -- whose full symmetric closure would be tens of millions of pairs,
# enough to exhaust memory during split materialization. 1,000,000 is chosen
# to sit orders of magnitude above any plausible legitimate window's gold-set
# size (real windows realistically produce gold sets in the tens to low
# thousands) while still catching a runaway window's pair count before the
# O(n^2) pair list is allocated.
MAX_GOLD_PAIRS_PER_WINDOW: int = 1_000_000


def _check_pair_budget(
    n_pairs_upper_bound: int,
    max_gold_pairs: int,
    task_id: int,
    window_id: Any,
    eligible_description: str,
) -> None:
    """Raise before allocating the pair list if it would exceed the budget."""
    if n_pairs_upper_bound <= max_gold_pairs:
        return
    raise ValueError(
        f"window {window_id!r} task {task_id}: gold-pair construction over "
        f"{eligible_description} would produce up to {n_pairs_upper_bound} pairs, "
        f"exceeding MAX_GOLD_PAIRS_PER_WINDOW={max_gold_pairs}. Refusing to "
        "materialize the full pair list; this usually means the window has an "
        "unexpectedly large number of task-eligible users."
    )


def _symmetric_pairs(
    by_user: dict[int, UserInstances],
    eligible: Callable[[UserInstances], bool],
    task_id: int,
    window_id: Any,
    max_gold_pairs: int = MAX_GOLD_PAIRS_PER_WINDOW,
) -> list[tuple[int, int]]:
    """Every pair of distinct users that both individually satisfy `eligible`.

    Fails fast -- before allocating the O(n^2) pair list -- if the number of
    eligible users would produce more than `max_gold_pairs` pairs; see
    `MAX_GOLD_PAIRS_PER_WINDOW`.
    """
    elig_users = sorted(u for u, instances in by_user.items() if eligible(instances))
    n = len(elig_users)
    _check_pair_budget(n * (n - 1) // 2, max_gold_pairs, task_id, window_id, f"{n} eligible users")
    return [(a, b) for idx, a in enumerate(elig_users) for b in elig_users[idx + 1 :]]


def _asymmetric_pairs(
    by_user: dict[int, UserInstances],
    role_a: Callable[[UserInstances], bool],
    role_b: Callable[[UserInstances], bool],
    task_id: int,
    window_id: Any,
    max_gold_pairs: int = MAX_GOLD_PAIRS_PER_WINDOW,
) -> list[tuple[int, int]]:
    """Every pair {u, v} where one of u/v satisfies role_a and the other satisfies role_b.

    Fails fast -- before the O(|A|*|B|) double loop -- if the role-set sizes'
    product would exceed `max_gold_pairs`; see `MAX_GOLD_PAIRS_PER_WINDOW`.
    """
    role_a_users = {u for u, instances in by_user.items() if role_a(instances)}
    role_b_users = {u for u, instances in by_user.items() if role_b(instances)}
    _check_pair_budget(
        len(role_a_users) * len(role_b_users),
        max_gold_pairs,
        task_id,
        window_id,
        f"{len(role_a_users)} role-A and {len(role_b_users)} role-B eligible users",
    )
    pairs = set()
    for u in role_a_users:
        for v in role_b_users:
            if u == v:
                continue
            pairs.add((u, v) if u < v else (v, u))
    return sorted(pairs)


_GoldFn = Callable[[dict[int, UserInstances], Any, int], list[tuple[int, int]]]

_GOLD_FNS: dict[int, _GoldFn] = {
    1: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_1, 1, window_id, max_gold_pairs
    ),
    2: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_2, 2, window_id, max_gold_pairs
    ),
    3: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_3, 3, window_id, max_gold_pairs
    ),
    4: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_4, 4, window_id, max_gold_pairs
    ),
    5: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_5, 5, window_id, max_gold_pairs
    ),
    6: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_6, 6, window_id, max_gold_pairs
    ),
    7: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_7, 7, window_id, max_gold_pairs
    ),
    8: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_8, 8, window_id, max_gold_pairs
    ),
    9: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_9, 9, window_id, max_gold_pairs
    ),
    10: lambda by_user, window_id, max_gold_pairs: _symmetric_pairs(
        by_user, _eligible_10, 10, window_id, max_gold_pairs
    ),
    11: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_11a, _role_11b, 11, window_id, max_gold_pairs
    ),
    12: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_12a, _role_12b, 12, window_id, max_gold_pairs
    ),
    13: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_13a, _role_13b, 13, window_id, max_gold_pairs
    ),
    14: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_14a, _role_14b, 14, window_id, max_gold_pairs
    ),
    15: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_15a, _role_15b, 15, window_id, max_gold_pairs
    ),
    16: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_16a, _role_16b, 16, window_id, max_gold_pairs
    ),
    17: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_17a, _role_17b, 17, window_id, max_gold_pairs
    ),
    18: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_18a, _role_18b, 18, window_id, max_gold_pairs
    ),
    19: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_19a, _role_19b, 19, window_id, max_gold_pairs
    ),
    20: lambda by_user, window_id, max_gold_pairs: _asymmetric_pairs(
        by_user, _role_20a, _role_20b, 20, window_id, max_gold_pairs
    ),
}


def compute_pairs(
    task_id: int,
    by_user: dict[int, UserInstances],
    window_id: Any = "<unknown>",
    max_gold_pairs: int = MAX_GOLD_PAIRS_PER_WINDOW,
) -> list[tuple[int, int]]:
    """Compute the pairs specified by the public task rule for ``task_id``.

    ``by_user`` maps each user id to ``(label, date)`` instances. The mapping
    may contain dataset labels while materializing gold, or model-predicted
    labels while running a paper-style symbolic pairwise baseline. The task
    rules are public prompt semantics; this function never reads a gold answer.
    Pairs are deduplicated, lower id first, and returned in sorted order.

    `window_id` and `max_gold_pairs` back the `MAX_GOLD_PAIRS_PER_WINDOW`
    fail-fast guard (see `_symmetric_pairs`/`_asymmetric_pairs`); `window_id`
    is used only in the resulting error message and defaults to a placeholder
    for direct callers (e.g. tests) with no real window on hand.
    """
    if task_id not in _GOLD_FNS:
        raise ValueError(f"Unknown OOLONG-Pairs task_id: {task_id} (expected 1-20)")
    return _GOLD_FNS[task_id](by_user, window_id, max_gold_pairs)


def compute_gold_pairs(
    task_id: int,
    by_user: dict[int, UserInstances],
    window_id: Any = "<unknown>",
    max_gold_pairs: int = MAX_GOLD_PAIRS_PER_WINDOW,
) -> list[tuple[int, int]]:
    """Backward-compatible grading name for :func:`compute_pairs`."""
    return compute_pairs(task_id, by_user, window_id, max_gold_pairs)


# =============================================================================
# Window parsing and prompt building (ported from examples/oolong_pairs/dataset.py)
# =============================================================================


@dataclass
class OolongEntry:
    user_id: int
    date: date
    instance: str
    label: str | None


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw.strip(), "%b %d, %Y").date()


def parse_entries(labeled_text: str) -> list[OolongEntry]:
    """Parse `context_window_text_with_labels` into ordered entries."""
    entries = []
    for line in labeled_text.splitlines():
        match = _ENTRY_RE.match(line.strip())
        if not match:
            continue
        entries.append(
            OolongEntry(
                user_id=int(match.group("user_id")),
                date=_parse_date(match.group("date")),
                instance=match.group("instance").strip(),
                label=match.group("label").strip() if match.group("label") else None,
            )
        )
    return entries


def build_prompt(entries: list[OolongEntry], task_text: str) -> str:
    """The RLM-facing prompt: unlabeled entries (no `label`) followed by the task text."""
    header = _HEADER_TEMPLATE.format(n=len(entries))
    lines = [
        f"Date: {e.date.strftime('%b %d, %Y')} || User: {e.user_id} || Instance: {e.instance}"
        for e in entries
    ]
    return header + "\n".join(lines) + "\n\n" + task_text


def build_gold_mapping(entries: list[OolongEntry]) -> dict[int, UserInstances]:
    """Ground-truth user_id -> [(label, date), ...] instances, used only for grading."""
    by_user: dict[int, UserInstances] = {}
    for e in entries:
        by_user.setdefault(e.user_id, []).append((e.label, e.date))
    return by_user


# =============================================================================
# Dataset streaming and window coverage
# =============================================================================


# Wall-clock ceiling on a single `next()` call against the upstream stream
# (see `_bounded_iter`). `collect_windows` runs during `materialize_splits`,
# which sits on the unattended, real-money `examples/experiment_smoke.py
# --live` path and is OUTSIDE the SIGALRM hard-deadline backstop (that only
# bounds run execution, not split materialization -- and
# `config.operational.loader_timeout_seconds` sounds like it covers this too
# but actually bounds an unrelated candidate-materialization subprocess). 60
# seconds is generous for a single row under normal network jitter or
# upstream retries, while still failing well before "indefinitely" if the
# stream has genuinely stalled.
DEFAULT_SCAN_DEADLINE_SECONDS: float = 60.0


def _bounded_iter(
    rows: Iterator[dict[str, Any]], deadline_seconds: float
) -> Iterator[dict[str, Any]]:
    """Wrap `rows` so a stalled `next()` call raises loudly instead of hanging.

    The `datasets` library exposes no per-call timeout for its streaming
    iterator, so this pulls rows on a background thread and enforces
    `deadline_seconds` with a blocking queue read. Because the read (not a
    post-hoc check between yielded rows) is what is bounded, a stall on the
    very FIRST row is caught just as reliably as a stall between rows. An
    exception raised by `rows` itself is re-raised unchanged in the caller's
    thread.
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

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    while True:
        try:
            item = result_q.get(timeout=deadline_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"OOLONG dataset stream stalled: no row received from {DATASET_REPO} "
                f"within {deadline_seconds}s. Aborting rather than blocking the "
                "invocation indefinitely."
            ) from exc
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def iter_dataset_rows(
    split: str,
    revision: str | None,
    deadline_seconds: float = DEFAULT_SCAN_DEADLINE_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Stream raw upstream rows -- the module's only network seam.

    Tests monkeypatch this with an iterator over synthetic rows. The imports
    live inside the function so the module imports without the ``oolong``
    extra; only actually streaming the dataset requires it. Rows are pulled
    through `_bounded_iter`, which bounds each `next()` call to
    `deadline_seconds` -- see its docstring for why.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for the OOLONG dataset. Install with:\n"
            '    uv pip install -e ".[oolong]"'
        ) from exc
    raw_rows = iter(load_dataset(DATASET_REPO, split=split, streaming=True, revision=revision))
    return _bounded_iter(raw_rows, deadline_seconds)


def _check_context_lengths(context_lengths: tuple[int, ...]) -> None:
    """Reject lengths the upstream data cannot supply, before any streaming."""
    if not context_lengths:
        raise ValueError("context_lengths must not be empty")
    if len(set(context_lengths)) != len(context_lengths):
        raise ValueError(f"context_lengths carries duplicates: {context_lengths}")
    unknown = sorted(set(context_lengths) - set(ALL_CONTEXT_LENGTHS))
    if unknown:
        raise ValueError(
            f"context length(s) {unknown} are not defined by OOLONG; the generator filters "
            f"existing upstream rows and cannot mint new lengths. Defined lengths (tokens): "
            f"{ALL_CONTEXT_LENGTHS}"
        )


def collect_windows(
    context_lengths: tuple[int, ...],
    n_per_length: int,
    split: str,
    max_scan: int,
    revision: str | None,
) -> dict[int, list[dict[str, Any]]]:
    """Collect the first ``n_per_length`` distinct trec_coarse windows per length.

    Deterministic and seed-independent: windows are kept in upstream stream
    order (pinned by ``revision``), deduplicated by ``(context_window_id,
    context_len)`` since upstream repeats the same window across its own
    tasks, scanning at most ``max_scan`` raw rows and stopping early once
    every requested length is satisfied.
    """
    _check_context_lengths(context_lengths)
    if n_per_length < 1:
        raise ValueError(f"n_per_length must be >= 1, got {n_per_length}")

    wanted_lengths = set(context_lengths)
    seen_windows: set[tuple[Any, int]] = set()
    windows_by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in context_lengths}
    for i, row in enumerate(iter_dataset_rows(split, revision)):
        if i >= max_scan:
            break
        if row["dataset"] != OOLONG_SUBSET:
            continue
        context_len = row["context_len"]
        if context_len not in wanted_lengths:
            continue
        if len(windows_by_length[context_len]) >= n_per_length:
            continue
        key = (row["context_window_id"], context_len)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        windows_by_length[context_len].append(row)
        if all(len(windows) >= n_per_length for windows in windows_by_length.values()):
            break
    return windows_by_length


def _require_coverage(
    windows_by_length: dict[int, list[dict[str, Any]]],
    n_windows: int,
    split: str,
    max_scan: int,
) -> dict[int, int]:
    """The shared coverage gate: raise naming the inventory on any shortfall."""
    inventory = {length: len(windows) for length, windows in windows_by_length.items()}
    short = sorted(length for length, count in inventory.items() if count < n_windows)
    if short:
        raise ValueError(
            f"insufficient distinct trec_coarse windows at context length(s) {short}: "
            f"need {n_windows} per length, but the first {max_scan} rows of "
            f"{DATASET_REPO}:{split} ship only {inventory} (distinct windows per "
            f"requested length). The generator filters existing upstream rows and "
            f"cannot mint new lengths."
        )
    return inventory


def verify_window_coverage(
    context_lengths: tuple[int, ...],
    n_windows: int,
    split: str = "validation",
    max_scan: int = 50_000,
    revision: str | None = None,
) -> dict[int, int]:
    """Verify upstream ships ``n_windows`` distinct windows at every length.

    The coverage pre-step for split materialization: run it before instances
    are produced. Returns the distinct-window inventory found per requested
    length (counting stops at ``n_windows`` per length once every length is
    satisfied). An unsatisfiable length raises naming the available inventory
    within the first ``max_scan`` scanned rows.
    """
    windows_by_length = collect_windows(context_lengths, n_windows, split, max_scan, revision)
    return _require_coverage(windows_by_length, n_windows, split, max_scan)


# =============================================================================
# Instance loading
# =============================================================================


def build_instance(
    window: dict[str, Any],
    task_id: int,
    sample_seed: int,
    sample_index: int,
) -> dict[str, Any]:
    """Pure transform from one (upstream window, task) pair to a mining instance.

    The id is derived from CONTENT -- task id, upstream ``context_window_id``,
    and sha256(prompt)[:16] -- so the same problem keeps the same id across
    rounds and across differently seeded samples. The seed and index are
    provenance, carried as separate fields and deliberately not folded into
    the id. ``gold_pairs`` is stored as lists (not tuples) so the instance
    JSON round-trips through ``instances.jsonl`` byte-identically.
    """
    entries = parse_entries(window["context_window_text_with_labels"])
    if not entries:
        raise ValueError(
            f"window {window['context_window_id']!r} parsed to zero entries; the upstream "
            "labeled text does not match the documented entry format"
        )
    by_user = build_gold_mapping(entries)
    task_text = TASK_TEXTS[task_id]
    prompt = build_prompt(entries, task_text)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    instance_id = f"oolong-t{task_id:02d}-w{window['context_window_id']}-{digest}"
    if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(
            f"derived instance id {instance_id!r} is not filesystem-safe "
            f"({FILESYSTEM_SAFE_ID_PATTERN.pattern}); upstream context_window_id "
            f"{window['context_window_id']!r} carries unsafe characters"
        )
    return {
        "id": instance_id,
        "question": task_text,
        "prompt": prompt,
        "gold_pairs": [
            [a, b] for a, b in compute_gold_pairs(task_id, by_user, window["context_window_id"])
        ],
        "task_id": task_id,
        "context_len": int(window["context_len"]),
        "context_window_id": window["context_window_id"],
        "num_entries": len(entries),
        "num_users": len(by_user),
        "sample_seed": sample_seed,
        "sample_index": sample_index,
    }


def load_oolong_pairs(
    task_ids: tuple[int, ...],
    context_lengths: tuple[int, ...],
    n: int,
    seed: int,
    max_scan: int = 50_000,
    split: str = "validation",
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Stream upstream windows, verify coverage, and build ``n`` instances.

    Requires ``ceil(n / len(task_ids))`` distinct windows at EVERY requested
    length -- enough that any single length alone could satisfy ``n`` -- and
    raises naming the available inventory otherwise (see
    ``verify_window_coverage``). The (window, task) rows are then shuffled
    with ``seed`` and truncated to ``n``; window selection itself is
    seed-independent (see ``collect_windows``), so the same arguments always
    yield the identical instance list, ids included.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    for task_id in task_ids:
        if task_id not in TASK_TEXTS:
            raise ValueError(f"Unknown OOLONG-Pairs task_id: {task_id} (expected 1-20)")

    n_windows = -(-n // len(task_ids))  # ceil division
    windows_by_length = collect_windows(context_lengths, n_windows, split, max_scan, revision)
    _require_coverage(windows_by_length, n_windows, split, max_scan)

    rows = [
        (window, task_id)
        for length in context_lengths
        for window in windows_by_length[length]
        for task_id in task_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [
        build_instance(window, task_id, sample_seed=seed, sample_index=index)
        for index, (window, task_id) in enumerate(rows[:n])
    ]


def load_oolong_pairs_from_config(
    config: OolongPairsConfig,
    n: int,
    seed: int,
    context_lengths: tuple[int, ...] | None = None,
    split: str = "validation",
) -> list[dict[str, Any]]:
    """``load_oolong_pairs`` with repo/pin/scan facts taken from the experiment config.

    ``context_lengths`` defaults to the configured (short, long) pair; pass an
    explicit tuple to load a single length (e.g. only the short split).
    """
    lengths = (
        context_lengths
        if context_lengths is not None
        else (config.context_length_short, config.context_length_long)
    )
    return load_oolong_pairs(
        task_ids=config.task_ids,
        context_lengths=lengths,
        n=n,
        seed=seed,
        max_scan=config.max_scan,
        split=split,
        revision=config.dataset_revision,
    )


# =============================================================================
# Answer parsing and scoring (ported from examples/oolong_pairs_example.py)
# =============================================================================


def extract_answer_pairs(response: str) -> list[tuple[int, int]] | None:
    """Pull every "(user_id_1, user_id_2)" pair out of a response, in order.

    Order within a pair is preserved (not sorted), since the task explicitly
    requires the lower id first -- an out-of-order pair must not silently
    score as correct.

    Returns None when the response carries neither a parseable pair nor the
    format's explicit empty marker ("No valid pairs found.") -- the response
    never entered the answer channel -- and ``[]`` when the empty marker is
    present, i.e. the model explicitly answered the empty set. This is the
    one deliberate divergence from the example's ``extract_answer_pairs``,
    which returns ``[]`` for both: the Verifier maps no-parse to WRONG_FORMAT
    but an explicit empty answer against a non-empty gold to NO_ANSWER.
    """
    pairs = [(int(a), int(b)) for a, b in _PAIR_RE.findall(response)]
    if pairs:
        return pairs
    if _NO_PAIRS_RE.search(response):
        return []
    return None


def score(predicted: list[tuple[int, int]], golden: list[tuple[int, int]]) -> dict[str, float]:
    """Precision/recall/F1 over the set of predicted vs. golden pairs.

    Both predicted and golden empty (a correct answer for a window with no
    qualifying pairs) scores precision=recall=f1=1.0. This is the one
    deliberate divergence from the example's verbatim scorer, which computed
    f1=0.0 for that case: precision landed at 0.0 (an empty prediction has no
    overlap to divide by), which kept `recall + precision > 0` true and
    routed past the `else 1.0` branch, scoring a correct answer as a total
    miss. An empty prediction against a non-empty gold still scores 0. The
    Verifier decides pass/fail on exact set equality and reports these
    metrics as detail only.
    """
    pred_set, gold_set = set(predicted), set(golden)
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    n_overlap = len(pred_set & gold_set)
    n_golden, n_sampled = len(gold_set), len(pred_set)
    recall = n_overlap / n_golden if n_golden > 0 else 1.0 if n_sampled == 0 else 0.0
    precision = n_overlap / n_sampled if n_sampled > 0 else 0.0
    # The both-empty perfect case already returned above, so every remaining
    # zero-overlap answer is genuinely wrong -- spurious pairs against an empty
    # gold, or an empty answer against a non-empty gold -- and scores 0.0.
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def serialize_pairs(pairs: set[tuple[int, int]]) -> str:
    """Render a pair set SORTED, so equal sets always serialize identically.

    The Verdict's gold field feeds the digest sha and through it the
    attribution cache; an order-dependent rendering would split cache entries
    for byte-identical problems.
    """
    return "[" + ", ".join(f"({a}, {b})" for a, b in sorted(pairs)) + "]"


# =============================================================================
# Verifier
# =============================================================================


class OolongPairsVerifier:
    """
    Deterministic outcome for a whole run: exact pair-set match.

    Cause mapping, decided entirely by set arithmetic on the parsed answer:

    * neither a parseable "(a, b)" pair nor the explicit "No valid pairs"
      marker anywhere in the response -> WRONG_FORMAT. The response never
      entered the environment's answer channel, so nothing finer can be
      measured.
    * an explicit empty answer against a non-empty gold -> NO_ANSWER. The
      format was honored but no pairs were produced; ``extract_answer_pairs``
      returning ``[]`` rather than None is what separates this from the case
      above.
    * missing pairs only -> INCOMPLETE; extra pairs only -> SPURIOUS; both
      -> MIXED_SET_ERROR. An out-of-order pair (higher id first) is counted
      as one missing gold pair plus one extra pair, per the task's explicit
      lower-id-first requirement.
    * both gold and prediction empty -> pass.

    A pass requires the exact gold set, i.e. F1 == 1.0 under the example's
    scorer in every non-degenerate case (see ``score``).

    There is deliberately no companion SubVerifier (see the module
    docstring): child correctness here is a semantic-labeling judgment no
    deterministic check can recompute.
    """

    PASS_F1_THRESHOLD: float = 1.0
    EXTRACTION_RULE: str = "user-id-pair-tuples"
    GOLD_ORDERING: str = "sorted"

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig by the experiment driver."""
        return {
            "environment": "oolong_pairs",
            "pass_f1_threshold": self.PASS_F1_THRESHOLD,
            "extraction_rule": self.EXTRACTION_RULE,
            "gold_ordering": self.GOLD_ORDERING,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        # JSON round-tripping turns pair tuples into lists; normalize both.
        gold_set = {(int(a), int(b)) for a, b in instance["gold_pairs"]}
        gold = serialize_pairs(gold_set)

        parsed = extract_answer_pairs(produced)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold,
                produced=produced,
                detail="no '(user_id_1, user_id_2)' pair and no 'No valid pairs' marker to parse",
            )

        pred_set = set(parsed)
        produced_pairs = serialize_pairs(pred_set)
        missing = gold_set - pred_set
        extra = pred_set - gold_set
        metrics = score(sorted(pred_set), sorted(gold_set))
        detail = (
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} missing={len(missing)} extra={len(extra)}"
        )

        if not missing and not extra:
            return Verdict(
                passed=True, cause=None, gold=gold, produced=produced_pairs, detail=detail
            )

        if not pred_set:
            cause = VerifierCause.NO_ANSWER
        elif missing and extra:
            cause = VerifierCause.MIXED_SET_ERROR
        elif missing:
            cause = VerifierCause.INCOMPLETE
        else:
            cause = VerifierCause.SPURIOUS
        return Verdict(passed=False, cause=cause, gold=gold, produced=produced_pairs, detail=detail)
