"""Stage-level usage metering: append-only ``stage_usage.jsonl`` records (R5; KTD4).

Every pipeline stage (mining runs, attribution, proposal, validation, eval)
appends one JSON line per execution attempt to a ``stage_usage.jsonl``
sidecar. The sidecar is new and append-only: nothing here touches
``summary.json`` (whose byte-compare on resume makes any schema change a
resume-breaker) or rewrites an existing record.

Two capture paths (KTD4)
    Run-executing stages (mining, validation, eval) cannot snapshot a client:
    the RLM runtime constructs a fresh client per completion
    (``rlm/core/rlm.py``), so their usage is aggregated from the per-run
    manifest keys via ``aggregate_manifest_usage`` and handed to the meter
    with ``StageMeter.add``. Caller-held clients (proposer, attributor) are
    registered with ``StageMeter.watch``, which snapshots
    ``BaseLM.get_usage_summary()`` on registration and diffs it at exit --
    the summary is cumulative per client, so the diff is exactly the stage's
    share even when sequential stages reuse one client.

The incremental contract and the exactly-once rule
    A record's usage covers only the work performed during that attempt.
    A resumed run-stage must therefore hand ``add`` the aggregate of the
    manifest lines *new in this attempt* (aggregate-after minus
    aggregate-before via ``UsageTotals.__sub__``), never the whole manifest.
    Under that contract, ``read_stage_usage`` counts each execution attempt
    exactly once: records are grouped by ``stage_work_id``; within a group,
    the *latest* record per ``attempt_index`` wins (a re-appended duplicate
    of the same attempt supersedes its predecessor), and the surviving
    records -- distinct attempts doing disjoint work -- are summed.
    ``attempt_index`` is assigned by the meter at entry as the number of
    records the sidecar already holds for its ``stage_work_id``, so the id
    is stable across resumes without caller bookkeeping. An attempt that
    crashed before its append leaves no record and is undercounted -- a
    lower bound, never a double count.

Lower bounds
    A record is flagged ``lower_bound`` when any contributed usage was itself
    a lower bound (e.g. terminated runs' manifest lines) or when the stage
    body raised -- a terminated attempt may have consumed usage the meter
    never saw (R5: terminated runs are lower bounds, never zeros).

Durability of the sidecar
    The sidecar is the accounting of an experiment that may hold thousands of
    paid runs, so a kill mid-append must never brick the directory. Each record
    is appended with one ``os.write`` to an ``O_APPEND`` descriptor
    (``append_record``): the kernel places an append-mode write at the end of
    the file in a single step, so a record cannot be interleaved with, or
    truncated by, another append. Should a write still be cut short (a torn
    line), the next append fences it off with a leading newline and
    ``read_usage_records`` quarantines it -- counted, skipped, never raised on.
    The stage totals it fed are then unattributable, so ``read_stage_usage``
    reports every total as a ``lower_bound`` while the torn line remains.
"""

import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from rlm.clients.base_lm import BaseLM

STAGE_USAGE_FILE = "stage_usage.jsonl"

# The manifest keys _persist_run added in U4; absent on pre-U4 lines.
_MANIFEST_USAGE_KEYS = ("input_tokens", "output_tokens", "execution_time")


@dataclass(frozen=True)
class UsageTotals:
    """One aggregate of usage: tokens, USD, wall clock, cache hits.

    ``lower_bound`` propagates through arithmetic: a sum containing any
    lower-bound contribution is itself a lower bound.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    wall_seconds: float = 0.0
    cache_hits: int = 0
    lower_bound: bool = False

    def __add__(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cache_hits=self.cache_hits + other.cache_hits,
            lower_bound=self.lower_bound or other.lower_bound,
        )

    def __sub__(self, other: "UsageTotals") -> "UsageTotals":
        """The increment from ``other`` (a prior aggregate) to ``self``.

        Both aggregates are cumulative over the same monotonically growing
        source (a client's usage summary, or a manifest that is only ever
        appended to), so a negative component means the snapshots were taken
        from different sources -- an accounting bug, refused loudly.
        """
        result = UsageTotals(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cost=self.cost - other.cost,
            wall_seconds=self.wall_seconds - other.wall_seconds,
            cache_hits=self.cache_hits - other.cache_hits,
            lower_bound=self.lower_bound or other.lower_bound,
        )
        negatives = [
            name
            for name in ("input_tokens", "output_tokens", "cost", "wall_seconds", "cache_hits")
            if getattr(result, name) < 0
        ]
        if negatives:
            raise ValueError(
                f"usage diff went negative on {negatives}: the later aggregate is smaller "
                "than the earlier one, so the two snapshots do not share a source."
            )
        return result


def snapshot_usage(lm: BaseLM) -> UsageTotals:
    """A client's cumulative usage right now, as ``UsageTotals``.

    ``get_usage_summary`` is cumulative per client, so two snapshots around a
    stage diff (via ``__sub__``) to exactly that stage's usage. A backend that
    reports no cost contributes zero USD, mirroring ``split_aggregate``'s
    treatment of cost-less runs.
    """
    summary = lm.get_usage_summary()
    cost = summary.total_cost
    return UsageTotals(
        input_tokens=summary.total_input_tokens,
        output_tokens=summary.total_output_tokens,
        cost=float(cost) if cost is not None else 0.0,
    )


def aggregate_manifest_usage(entries: Iterable[Mapping[str, Any]]) -> UsageTotals:
    """Sum per-run manifest lines into one ``UsageTotals``.

    The capture path for run-executing stages (KTD4). Pre-U4 lines carry no
    token/time keys: they contribute zero and force ``lower_bound`` -- their
    usage is unknown, not free. A ``cost`` of ``None`` contributes zero USD
    without flagging, mirroring ``split_aggregate``'s documented stance that
    a cost-less backend measures nothing. Entries flagged
    ``usage_lower_bound`` (terminated runs) propagate the flag.
    """
    total = UsageTotals()
    for entry in entries:
        pre_u4 = any(key not in entry for key in _MANIFEST_USAGE_KEYS)
        cost = entry.get("cost")
        total = total + UsageTotals(
            input_tokens=int(entry.get("input_tokens", 0)),
            output_tokens=int(entry.get("output_tokens", 0)),
            cost=float(cost) if cost is not None else 0.0,
            wall_seconds=float(entry.get("execution_time", 0.0)),
            lower_bound=bool(entry.get("usage_lower_bound", False)) or pre_u4,
        )
    return total


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON-lines file's records in file order; ``[]`` when it is absent.

    The package's single JSON-lines reader: the usage sidecar, the report's
    manifests, the persisted split files, and the smoke checks all parse
    through it, so "absent" and "blank line" mean one thing everywhere. It
    imposes no emptiness policy -- callers that require records (a split with
    no instances, say) check and raise with their own error type.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class SidecarRecords:
    """One sidecar's parsed records, plus how many lines failed to parse."""

    records: list[dict[str, Any]]
    torn_lines: int


def read_usage_records(path: Path) -> SidecarRecords:
    """The sidecar's records, quarantining any line that does not parse.

    The usage sidecar is the one JSON-lines file in the package that must stay
    readable through a crash: every ``StageMeter.__enter__``, report build, and
    smoke spend check parses the whole file, so a single torn line raised on
    would make an experiment directory unusable for work already paid for. A
    line that does not parse is therefore skipped and counted rather than
    raised on -- ``append_record`` fences it off so it can never swallow a
    later record -- and the count lets ``read_stage_usage`` mark the totals it
    can no longer account for as lower bounds. Split files and run manifests
    keep the strict ``read_jsonl`` reader: there a corrupt line is a stop sign.
    """
    if not path.exists():
        return SidecarRecords(records=[], torn_lines=0)
    records: list[dict[str, Any]] = []
    torn = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return SidecarRecords(records=records, torn_lines=torn)


def _fence_prefix(path: Path) -> str:
    """A newline when the sidecar's last line is unterminated, else nothing.

    An unterminated last line is a torn append. Without the fence the next
    record would be concatenated onto it, turning one damaged line into two
    lost records; with it, the damage stays confined to the torn line.
    """
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return "" if handle.read(1) == b"\n" else "\n"


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one record to the sidecar as a single atomic write."""
    payload = _fence_prefix(path) + json.dumps(record, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


class StageMeter:
    """Context manager metering one stage execution attempt (R5).

    On exit -- normal or exceptional -- it appends exactly one record to the
    append-only sidecar: stage name, stable ``stage_work_id`` (caller-supplied,
    e.g. ``"round03/mining"``), round index, ``attempt_index``, input/output
    tokens, USD, wall seconds, cache hits, ``resumed`` flag, ``lower_bound``
    flag, timestamp. Wall clock is ``time.perf_counter`` around the body.

    Usage comes in through the two KTD4 paths: ``add`` for caller-computed
    aggregates (run stages; see the module docstring's incremental contract)
    and ``watch`` for caller-held clients (proposer/attributor), snapshot at
    registration and diffed at exit. ``resumed`` defaults to
    ``attempt_index > 0`` -- a prior record for the same ``stage_work_id``
    means this attempt re-executes interrupted work -- and an explicit bool
    overrides that inference. An exception in the body still appends the
    record (whatever usage was registered, flagged ``lower_bound``) and then
    propagates.
    """

    def __init__(
        self,
        *,
        stage: str,
        stage_work_id: str,
        round_index: int,
        out_path: Path | str,
        resumed: bool | None = None,
    ) -> None:
        self.stage = stage
        self.stage_work_id = stage_work_id
        self.round_index = round_index
        self.out_path = Path(out_path)
        self._resumed = resumed
        self._added: list[UsageTotals] = []
        self._watched: list[tuple[BaseLM, UsageTotals]] = []
        self._cache_hits = 0
        self._started: float | None = None
        self._attempt_index: int | None = None

    def __enter__(self) -> "StageMeter":
        self._attempt_index = sum(
            1
            for record in read_usage_records(self.out_path).records
            if record["stage_work_id"] == self.stage_work_id
        )
        self._started = time.perf_counter()
        return self

    def watch(self, lm: BaseLM) -> None:
        """Register a caller-held client: snapshot now, diff at exit."""
        if self._started is None:
            raise RuntimeError("StageMeter.watch called outside the context body")
        self._watched.append((lm, snapshot_usage(lm)))

    def add(self, totals: UsageTotals) -> None:
        """Register a caller-computed aggregate (the run-stage capture path)."""
        if self._started is None:
            raise RuntimeError("StageMeter.add called outside the context body")
        self._added.append(totals)

    def add_cache_hits(self, count: int) -> None:
        """Record cache hits (proposal/attribution caches), caller-counted."""
        if count < 0:
            raise ValueError(f"cache hit count must be non-negative, got {count}")
        self._cache_hits += count

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._started is None or self._attempt_index is None:
            raise RuntimeError("StageMeter exited without entering")
        wall_seconds = time.perf_counter() - self._started
        usage = UsageTotals()
        for totals in self._added:
            usage = usage + totals
        for lm, before in self._watched:
            usage = usage + (snapshot_usage(lm) - before)
        record = {
            "stage": self.stage,
            "stage_work_id": self.stage_work_id,
            "round_index": self.round_index,
            "attempt_index": self._attempt_index,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost": usage.cost,
            "wall_seconds": wall_seconds,
            "cache_hits": self._cache_hits + usage.cache_hits,
            "resumed": self._resumed if self._resumed is not None else self._attempt_index > 0,
            "lower_bound": usage.lower_bound or exc_type is not None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        append_record(self.out_path, record)
        return False


def read_stage_usage(path: Path | str) -> dict[str, UsageTotals]:
    """Aggregate a sidecar by ``stage_work_id``, counting each attempt exactly once.

    The exactly-once rule (module docstring): per ``stage_work_id``, the
    latest record for each ``attempt_index`` supersedes any earlier duplicate,
    and the surviving records sum -- their usage is disjoint by the
    incremental contract. Wall seconds sum across attempts too: the total is
    time actually spent, not time the finished stage would have taken.

    A quarantined torn line (see ``read_usage_records``) carried usage that
    cannot be attributed to any ``stage_work_id``, so while one is present
    every total is reported as a ``lower_bound`` rather than as an exact
    figure.
    """
    sidecar = read_usage_records(Path(path))
    torn = sidecar.torn_lines > 0
    latest: dict[str, dict[int, dict[str, Any]]] = {}
    for record in sidecar.records:
        work_id = str(record["stage_work_id"])
        latest.setdefault(work_id, {})[int(record["attempt_index"])] = record
    totals: dict[str, UsageTotals] = {}
    for work_id, attempts in latest.items():
        total = UsageTotals(lower_bound=torn)
        for _, record in sorted(attempts.items()):
            total = total + UsageTotals(
                input_tokens=int(record["input_tokens"]),
                output_tokens=int(record["output_tokens"]),
                cost=float(record["cost"]),
                wall_seconds=float(record["wall_seconds"]),
                cache_hits=int(record["cache_hits"]),
                lower_bound=bool(record["lower_bound"]),
            )
        totals[work_id] = total
    return totals
