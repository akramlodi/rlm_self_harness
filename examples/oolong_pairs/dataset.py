"""
Stage 1 -- generate the OOLONG-Pairs benchmark.

There is no public OOLONG-Pairs dataset. The RLM paper (Appendix D.1) builds it by taking
the OOLONG `trec_coarse` split (Bertsch et al. 2025) -- contiguous windows of TREC
general-knowledge questions, each tagged with a User ID and a hidden ground-truth semantic
label -- and asking 20 fixed pairwise questions (see `tasks.py`) over the *same* underlying
windows. So this module downloads the real OOLONG data (`oolongbench/oolong-synth` on
Hugging Face) and recreates the benchmark from it:

  1. Stream the dataset and pick out distinct `trec_coarse` context windows at the
     requested context lengths (windows repeat across the original OOLONG's own tasks --
     e.g. "most common label", "count" -- since those are different questions over the same
     entries, so we dedupe by `context_window_id`).
  2. Parse each window's `context_window_text_with_labels` field into per-entry
     (user_id, date, label) records -- this is the ground truth, used only to compute gold
     answers, never shown to the RLM.
  3. Build the RLM-facing prompt from the *unlabeled* entries plus one of the 20 exact task
     texts from `tasks.py`.
  4. Compute the gold (user_id_1, user_id_2) pairs for that task programmatically via
     `tasks.compute_gold_pairs`.

`generate_oolong_pairs(...)` is the reusable entry point; each returned row is:
    {"task_id": int, "prompt": str, "gold_pairs": list[tuple[int, int]], ...}
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, datetime

from examples.oolong_pairs.tasks import TASK_TEXTS, compute_gold_pairs

DATASET_REPO = "oolongbench/oolong-synth"
OOLONG_SUBSET = "trec_coarse"

# All context lengths OOLONG-Pairs is defined over (Appendix D.1).
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

# Matches a raw OOLONG entry line, with or without the trailing label:
#   "Date: Feb 02, 2023 || User: 76153 || Instance: What is one of the languages of the Sioux ?"
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


@dataclass
class OolongEntry:
    user_id: int
    date: date
    instance: str
    label: str | None


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw.strip(), "%b %d, %Y").date()


def _parse_entries(labeled_text: str) -> list[OolongEntry]:
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


def _build_prompt(entries: list[OolongEntry], task_text: str) -> str:
    """The RLM-facing prompt: unlabeled entries (no `label`) followed by the task text."""
    header = _HEADER_TEMPLATE.format(n=len(entries))
    lines = [
        f"Date: {e.date.strftime('%b %d, %Y')} || User: {e.user_id} || Instance: {e.instance}"
        for e in entries
    ]
    return header + "\n".join(lines) + "\n\n" + task_text


def _build_gold_mapping(entries: list[OolongEntry]) -> dict[int, list[tuple[str, date]]]:
    """Ground-truth user_id -> [(label, date), ...] instances, used only for grading."""
    by_user: dict[int, list[tuple[str, date]]] = {}
    for e in entries:
        by_user.setdefault(e.user_id, []).append((e.label, e.date))
    return by_user


def _stream_context_windows(
    context_lengths: tuple[int, ...],
    split: str,
    n_per_length: int,
    max_scan: int,
) -> list[dict]:
    """Stream `oolongbench/oolong-synth` and collect distinct trec_coarse context windows.

    Multiple dataset rows can share the same underlying window (they differ only in which
    original OOLONG question -- e.g. "most common label" -- was asked over it), so we dedupe
    by (context_window_id, context_len) and keep the first row seen per window.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency for the OOLONG dataset. Install with:\n"
            '    uv pip install -e ".[oolong]"'
        ) from e

    stream = load_dataset(DATASET_REPO, split=split, streaming=True)
    wanted_lengths = set(context_lengths)
    seen_windows: set[tuple[int, int]] = set()
    windows_by_length: dict[int, list[dict]] = {length: [] for length in wanted_lengths}

    for i, ex in enumerate(stream):
        if i >= max_scan:
            break
        if ex.get("dataset") != OOLONG_SUBSET:
            continue
        context_len = ex.get("context_len")
        if context_len not in wanted_lengths:
            continue
        key = (ex.get("context_window_id"), context_len)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        windows_by_length[context_len].append(ex)
        if all(len(windows_by_length[length]) >= n_per_length for length in wanted_lengths):
            break

    return [ex for length in context_lengths for ex in windows_by_length[length]]


def generate_oolong_pairs(
    task_ids: list[int] | None = None,
    context_lengths: tuple[int, ...] = (1024, 2048),
    n_windows: int = 4,
    split: str = "validation",
    seed: int = 0,
    max_scan: int = 50_000,
) -> list[dict]:
    """Recreate the OOLONG-Pairs benchmark: (context window x task) rows with gold answers.

    Args:
        task_ids: Which of the 20 tasks (1-20) to generate. Defaults to all 20.
        context_lengths: Which OOLONG context lengths to sample windows from (subset of
            `ALL_CONTEXT_LENGTHS`).
        n_windows: Number of distinct context windows to sample per context length.
        split: HF dataset split ("validation" or "test").
        seed: Shuffle seed for reproducible sampling.
        max_scan: Safety cap on how many raw dataset rows to stream through while looking
            for matching windows.

    Returns:
        A list of rows, one per (window, task) pair, each with:
            {"task_id", "task_text", "prompt", "gold_pairs", "context_len",
             "context_window_id", "num_entries", "num_users"}
    """
    task_ids = task_ids if task_ids is not None else sorted(TASK_TEXTS)
    for task_id in task_ids:
        if task_id not in TASK_TEXTS:
            raise ValueError(f"Unknown OOLONG-Pairs task_id: {task_id} (expected 1-20)")

    windows = _stream_context_windows(
        context_lengths=context_lengths,
        split=split,
        n_per_length=n_windows,
        max_scan=max_scan,
    )
    if not windows:
        raise RuntimeError(
            f"No trec_coarse context windows found for context_lengths={context_lengths} "
            f"in split={split!r} within the first {max_scan} rows scanned."
        )

    rng = random.Random(seed)
    rng.shuffle(windows)

    rows = []
    for window in windows:
        entries = _parse_entries(window["context_window_text_with_labels"])
        by_user = _build_gold_mapping(entries)
        for task_id in task_ids:
            task_text = TASK_TEXTS[task_id]
            rows.append(
                {
                    "task_id": task_id,
                    "task_text": task_text,
                    "prompt": _build_prompt(entries, task_text),
                    "gold_pairs": compute_gold_pairs(task_id, by_user),
                    # Ground truth, for grading and sub-verification only -- never put this
                    # (or anything derived from it) into `prompt` / the RLM-visible context.
                    "gold_by_user": by_user,
                    "context_len": window["context_len"],
                    "context_window_id": window["context_window_id"],
                    "num_entries": len(entries),
                    "num_users": len(by_user),
                }
            )
    rng.shuffle(rows)
    return rows
