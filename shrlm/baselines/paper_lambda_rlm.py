"""Paper-faithful OOLONG-Pairs reconstruction for the pinned λ-RLM source.

The paper's Algorithm 5 specifies

    SPLIT -> MAP(sub_M_cls) -> PARSE -> FILTER(phi) -> CROSS

but the released upstream ``rlm/lambda_rlm.py`` does not implement that
specialized path.  This module preserves the vendored source byte-for-byte and
adds the missing pairwise realization as a subclass.  Non-pairwise instances
continue through the pinned upstream implementation unchanged.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import RLMChatCompletion
from shrlm.baselines.upstream import lambda_rlm as upstream_lambda
from shrlm.environments.oolong_pairs import (
    LABELS,
    OolongEntry,
    UserInstances,
    compute_pairs,
    parse_entries,
)

PAPER_RECONSTRUCTION_VERSION = "oolong-pairs-algorithm-5/v2"
PAIRWISE_AUDIT_FORMAT = "shrlm-lambda-pairwise-audit/v1"

LABEL_TO_CODE: dict[str, str] = {
    "description and abstract concept": "D",
    "entity": "E",
    "human being": "H",
    "numeric value": "N",
    "location": "L",
    "abbreviation": "A",
}
CODE_TO_LABEL: dict[str, str] = {code: label for label, code in LABEL_TO_CODE.items()}

_HEADER_COUNT_RE = re.compile(r"The following lines contain (?P<count>\d+) general-knowledge")
_CLASSIFICATION_LINE_RE = re.compile(r"^(?P<index>\d+)\s*[|:\t,]\s*(?P<code>[DEHNLA])$")


@dataclass(frozen=True)
class ClassificationBatch:
    """One bounded MAP input in the paper's pairwise combinator chain."""

    entries: tuple[tuple[int, OolongEntry], ...]
    prompt: str


@dataclass(frozen=True)
class PairwiseExecutionTrace:
    """Auditable call-shape and progress record for the last pairwise run."""

    phase: str
    task_id: int
    records: int
    users: int
    batches: int
    completed_batches: int
    model_call_upper_bound: int
    pairs: int | None


@dataclass(frozen=True)
class ClassificationAttemptAudit:
    """One raw classifier response and its parse result."""

    attempt: int
    response: str
    rejection: str | None


@dataclass(frozen=True)
class ClassificationBatchAudit:
    """Persistable evidence for one MAP batch, without duplicating its prompt."""

    batch_index: int
    global_indices: tuple[int, ...]
    attempts: tuple[ClassificationAttemptAudit, ...]
    predictions: dict[int, str]


def parse_oolong_prompt(prompt: str) -> list[OolongEntry]:
    """Parse every unlabeled record and reject truncated or malformed prompts."""
    header = _HEADER_COUNT_RE.search(prompt)
    if header is None:
        raise ValueError("OOLONG-Pairs prompt is missing its declared record count")
    declared = int(header.group("count"))
    entries = parse_entries(prompt)
    if len(entries) != declared:
        raise ValueError(
            f"OOLONG-Pairs prompt declares {declared} records but exactly {len(entries)} "
            "valid record lines were parsed"
        )
    if any(entry.label is not None for entry in entries):
        raise ValueError("OOLONG-Pairs model input unexpectedly exposes dataset labels")
    return entries


def classification_prompt(entries: Sequence[OolongEntry]) -> str:
    """Build a compact, strictly parseable semantic-classification request.

    Numbers items by their position within THIS batch (0..len(entries)-1),
    not by their position in the full instance: a live run against a
    262k-token instance (thousands of records, indices in the low
    thousands) showed a real model mistranscribing a large global index
    across a long list. Keeping the number the model has to echo back small
    and batch-local removes that failure mode at the source; the caller
    maps the local index back to the record's global index afterward.
    """
    if not entries:
        raise ValueError("classification batch must not be empty")
    items = "\n".join(f"{position}\t{entry.instance}" for position, entry in enumerate(entries))
    return (
        "Classify the expected ANSWER TYPE of every general-knowledge question below.\n"
        "Use exactly one code from this fixed menu:\n"
        "D = description and abstract concept\n"
        "E = entity\n"
        "H = human being\n"
        "N = numeric value\n"
        "L = location\n"
        "A = abbreviation\n\n"
        "Return exactly one line per input, in the identical order and with no explanation:\n"
        "<index>|<code>\n"
        "Do not answer the questions. Do not omit, duplicate, or renumber an index.\n\n"
        f"Questions:\n{items}"
    )


def build_classification_batches(
    entries: Sequence[OolongEntry],
    *,
    max_records: int,
    max_chars: int,
) -> list[ClassificationBatch]:
    """Split on record boundaries under simultaneous input/output bounds."""
    if max_records < 1:
        raise ValueError("pairwise_max_batch_records must be >= 1")
    if max_chars < 1:
        raise ValueError("pairwise_max_batch_chars must be >= 1")

    batches: list[ClassificationBatch] = []
    current: list[tuple[int, OolongEntry]] = []
    for indexed_entry in enumerate(entries):
        candidate = [*current, indexed_entry]
        candidate_prompt = classification_prompt([entry for _, entry in candidate])
        if len(candidate) > max_records or len(candidate_prompt) > max_chars:
            if not current:
                raise ValueError(
                    f"OOLONG-Pairs record {indexed_entry[0]} cannot fit the configured "
                    f"classification batch limit of {max_chars} characters"
                )
            batches.append(
                ClassificationBatch(
                    tuple(current), classification_prompt([entry for _, entry in current])
                )
            )
            current = [indexed_entry]
            if len(classification_prompt([entry for _, entry in current])) > max_chars:
                raise ValueError(
                    f"OOLONG-Pairs record {indexed_entry[0]} cannot fit the configured "
                    f"classification batch limit of {max_chars} characters"
                )
        else:
            current = candidate
    if current:
        batches.append(
            ClassificationBatch(
                tuple(current), classification_prompt([entry for _, entry in current])
            )
        )
    return batches


def parse_classifications(
    response: str,
    expected_indices: Sequence[int],
) -> dict[int, str]:
    """Parse one MAP result with exact coverage and a closed label vocabulary."""
    expected = set(expected_indices)
    parsed: dict[int, str] = {}
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line or line in {"```", "```text"}:
            continue
        match = _CLASSIFICATION_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed OOLONG classification line: {raw_line!r}")
        index = int(match.group("index"))
        if index not in expected:
            raise ValueError(f"unexpected OOLONG classification index {index}")
        if index in parsed:
            raise ValueError(f"duplicate OOLONG classification index {index}")
        parsed[index] = CODE_TO_LABEL[match.group("code")]
    missing = sorted(expected - set(parsed))
    if missing:
        raise ValueError(f"missing OOLONG classification indices: {missing}")
    return parsed


def aggregate_predictions(
    entries: Sequence[OolongEntry],
    predictions: dict[int, str],
) -> dict[int, UserInstances]:
    """Merge labels globally by user before applying the pair predicate."""
    if set(predictions) != set(range(len(entries))):
        raise ValueError("classification predictions do not cover every OOLONG record exactly")
    by_user: dict[int, UserInstances] = {}
    for index, entry in enumerate(entries):
        label = predictions[index]
        if label not in LABELS:
            raise ValueError(f"unknown OOLONG label for record {index}: {label!r}")
        by_user.setdefault(entry.user_id, []).append((label, entry.date))
    return by_user


def format_pairs(pairs: Sequence[tuple[int, int]]) -> str:
    """Return the verifier's canonical lower-first, sorted, newline format."""
    if not pairs:
        return "No valid pairs found."
    return "\n".join(f"({left}, {right})" for left, right in pairs)


class PaperLambdaRLM(upstream_lambda.LambdaRLM):
    """Pinned λ-RLM plus the paper's missing OOLONG-Pairs Algorithm 5."""

    def __init__(
        self,
        *args: Any,
        task_id: int | None = None,
        pairwise_max_batch_records: int = 256,
        pairwise_max_batch_chars: int = 80_000,
        pairwise_max_concurrency: int = 8,
        pairwise_max_attempts: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if task_id is not None and task_id not in range(1, 21):
            raise ValueError(f"OOLONG-Pairs task_id must be in 1-20, got {task_id}")
        if pairwise_max_batch_records < 1:
            raise ValueError("pairwise_max_batch_records must be >= 1")
        if pairwise_max_batch_chars < 1:
            raise ValueError("pairwise_max_batch_chars must be >= 1")
        if pairwise_max_batch_chars > self.context_window_chars:
            raise ValueError("pairwise_max_batch_chars must not exceed context_window_chars")
        if pairwise_max_concurrency < 1:
            raise ValueError("pairwise_max_concurrency must be >= 1")
        if pairwise_max_attempts < 1:
            raise ValueError("pairwise_max_attempts must be >= 1")
        self.task_id = task_id
        self.pairwise_max_batch_records = pairwise_max_batch_records
        self.pairwise_max_batch_chars = pairwise_max_batch_chars
        self.pairwise_max_concurrency = pairwise_max_concurrency
        self.pairwise_max_attempts = pairwise_max_attempts
        self.last_pairwise_trace: PairwiseExecutionTrace | None = None
        self.last_pairwise_audit: tuple[ClassificationBatchAudit, ...] | None = None

    def completion(self, prompt: str) -> RLMChatCompletion:
        if self.task_id is None:
            return super().completion(prompt)
        return self.pairwise_completion(prompt)

    def pairwise_completion(self, prompt: str) -> RLMChatCompletion:
        """Execute SPLIT -> MAP(M_cls) -> PARSE -> FILTER(phi) -> CROSS."""
        if not isinstance(prompt, str):
            raise TypeError(
                f"PaperLambdaRLM.completion requires a string prompt, got {type(prompt).__name__}"
            )
        assert self.task_id is not None
        started = time.perf_counter()
        entries = parse_oolong_prompt(prompt)
        batches = build_classification_batches(
            entries,
            max_records=self.pairwise_max_batch_records,
            max_chars=self.pairwise_max_batch_chars,
        )
        users = len({entry.user_id for entry in entries})
        self.last_pairwise_trace = PairwiseExecutionTrace(
            phase="classification",
            task_id=self.task_id,
            records=len(entries),
            users=users,
            batches=len(batches),
            completed_batches=0,
            model_call_upper_bound=len(batches) * self.pairwise_max_attempts,
            pairs=None,
        )
        if self.verbose:
            print(
                f"[λ-RLM/pairwise] task={self.task_id} records={len(entries)} "
                f"users={users} batches={len(batches)} concurrency={self.pairwise_max_concurrency}"
            )

        client = upstream_lambda.get_client(self.backend, self.backend_kwargs)
        batch_audits = asyncio.run(self.classify_batches(client, batches))
        self.last_pairwise_audit = tuple(batch_audits)
        predictions: dict[int, str] = {}
        for batch_audit in batch_audits:
            predictions.update(batch_audit.predictions)

        self.last_pairwise_trace = PairwiseExecutionTrace(
            phase="symbolic_cross",
            task_id=self.task_id,
            records=len(entries),
            users=users,
            batches=len(batches),
            completed_batches=len(batches),
            model_call_upper_bound=len(batches) * self.pairwise_max_attempts,
            pairs=None,
        )
        by_user = aggregate_predictions(entries, predictions)
        pairs = compute_pairs(self.task_id, by_user, window_id="model-predicted")
        response = format_pairs(pairs)
        self.last_pairwise_trace = PairwiseExecutionTrace(
            phase="complete",
            task_id=self.task_id,
            records=len(entries),
            users=users,
            batches=len(batches),
            completed_batches=len(batches),
            model_call_upper_bound=len(batches) * self.pairwise_max_attempts,
            pairs=len(pairs),
        )
        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "unknown"),
            prompt=prompt,
            response=response,
            usage_summary=client.get_usage_summary(),
            execution_time=time.perf_counter() - started,
            metadata={
                "pairwise_audit": {
                    "format": PAIRWISE_AUDIT_FORMAT,
                    "reconstruction_version": PAPER_RECONSTRUCTION_VERSION,
                    "execution": asdict(self.last_pairwise_trace),
                    "actual_model_calls": sum(
                        len(batch_audit.attempts) for batch_audit in batch_audits
                    ),
                    "label_counts": {
                        label: sum(prediction == label for prediction in predictions.values())
                        for label in LABELS
                    },
                    "batches": [asdict(batch_audit) for batch_audit in batch_audits],
                }
            },
        )

    async def classify_batches(
        self,
        client: BaseLM,
        batches: Sequence[ClassificationBatch],
    ) -> list[ClassificationBatchAudit]:
        """Classify every batch, retrying a batch's own call on a rejection.

        A rejection (a malformed line, an index outside this batch, a
        duplicate, or missing coverage) re-asks with the violation appended,
        up to ``pairwise_max_attempts`` -- the same call-then-validate retry
        shape ``optimization/attribution.py`` uses for the attributor. This
        keeps one flaky batch out of ~dozens from failing an entire long
        run's already-sunk cost; it still fails loud once attempts are
        exhausted. Results carry global indices (mapped from each batch's
        local numbering) so the caller can merge them directly.
        """
        semaphore = asyncio.Semaphore(self.pairwise_max_concurrency)

        async def classify(
            batch_index: int,
            batch: ClassificationBatch,
        ) -> ClassificationBatchAudit:
            expected_local = range(len(batch.entries))
            rejection = ""
            attempts: list[ClassificationAttemptAudit] = []
            async with semaphore:
                for attempt in range(1, self.pairwise_max_attempts + 1):
                    request = (
                        batch.prompt
                        if not rejection
                        else (
                            f"{batch.prompt}\n\nYour previous response was rejected: "
                            f"{rejection}\nRespond again, following the format exactly."
                        )
                    )
                    response = await client.acompletion(request)
                    try:
                        local_predictions = parse_classifications(response, expected_local)
                    except ValueError as exc:
                        rejection = str(exc)
                        attempts.append(ClassificationAttemptAudit(attempt, response, rejection))
                        continue
                    attempts.append(ClassificationAttemptAudit(attempt, response, None))
                    if self.verbose:
                        print(f"[λ-RLM/pairwise] classified batch {batch_index + 1}/{len(batches)}")
                    return ClassificationBatchAudit(
                        batch_index=batch_index,
                        global_indices=tuple(index for index, _ in batch.entries),
                        attempts=tuple(attempts),
                        predictions={
                            batch.entries[local][0]: label
                            for local, label in local_predictions.items()
                        },
                    )
                raise ValueError(
                    f"OOLONG-Pairs classification batch {batch_index} still rejected after "
                    f"{self.pairwise_max_attempts} attempts: {rejection}"
                )

        return list(
            await asyncio.gather(*(classify(index, batch) for index, batch in enumerate(batches)))
        )


__all__ = [
    "CODE_TO_LABEL",
    "LABEL_TO_CODE",
    "PAIRWISE_AUDIT_FORMAT",
    "PAPER_RECONSTRUCTION_VERSION",
    "ClassificationAttemptAudit",
    "ClassificationBatch",
    "ClassificationBatchAudit",
    "PairwiseExecutionTrace",
    "PaperLambdaRLM",
    "aggregate_predictions",
    "build_classification_batches",
    "classification_prompt",
    "format_pairs",
    "parse_classifications",
    "parse_oolong_prompt",
]
