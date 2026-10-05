"""Document-aware λ-RLM adaptation for OBLIQ-Bench Math long ranking.

The pinned upstream λ-RLM implements a generic QA map/reduce pipeline.  That
pipeline may compress away document identifiers, which makes it a poor fit for
OBLIQ's ranked-retrieval contract.  This module keeps the λ-RLM decomposition
shape but makes document IDs the conserved host-side representation:

``document batches -> local rankings -> candidate union -> global reductions``.

It is deliberately a separately named method, not a modification of the
paper/upstream baseline.
"""

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.clients.base_lm import BaseLM
from rlm.core.types import RLMChatCompletion
from shrlm.baselines.paper_lambda_rlm import PaperLambdaRLM
from shrlm.environments.obliq_bench_math import RANK_K

OBLIQ_LAMBDA_VERSION = "2"
OBLIQ_LAMBDA_AUDIT_FORMAT = "shrlm-obliq-lambda-audit/v2"
DEFAULT_OBLIQ_MAX_BATCH_DOCUMENTS = 64
DEFAULT_OBLIQ_MAX_BATCH_CHARS = 25_000
DEFAULT_OBLIQ_MAX_ATTEMPTS = 3
DOCUMENT_HEADER_RE = re.compile(r"(?m)^\[([^\]\n]+)\]\n")
CORPUS_MARKER = "CANDIDATE CORPUS:\n"
FINAL_REMINDER_MARKER = "\n\nFINAL OUTPUT REMINDER:"
RANKED_INDICES_RE = re.compile(r"RANKED_INDICES:\s*(\[[^\n]*\])\s*")


@dataclass(frozen=True)
class ObliqDocument:
    """One candidate document recovered from the benchmark prompt."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class ObliqRankingBatch:
    """One document-aligned model request."""

    documents: tuple[ObliqDocument, ...]


@dataclass(frozen=True)
class ObliqRankingAttempt:
    """Auditable response and validation outcome for one model request."""

    attempt: int
    response: str
    rejection: str | None
    selected_indices: tuple[int, ...]
    dropped_values: tuple[str, ...]


@dataclass(frozen=True)
class ObliqRankingBatchAudit:
    """All attempts and accepted IDs for one batch."""

    stage: str
    batch_index: int
    document_ids: tuple[str, ...]
    attempts: tuple[ObliqRankingAttempt, ...]
    selected_indices: tuple[int, ...]
    selected_ids: tuple[str, ...]
    degraded: bool


def parse_obliq_documents(prompt: str) -> list[ObliqDocument]:
    """Recover candidate documents without splitting inside problem text."""
    if not isinstance(prompt, str):
        raise TypeError(f"OBLIQ prompt must be a string, got {type(prompt).__name__}")
    marker_index = prompt.find(CORPUS_MARKER)
    if marker_index < 0:
        raise ValueError("OBLIQ prompt is missing the CANDIDATE CORPUS marker")
    corpus = prompt[marker_index + len(CORPUS_MARKER) :]
    reminder_index = corpus.rfind(FINAL_REMINDER_MARKER)
    if reminder_index < 0:
        raise ValueError("OBLIQ prompt is missing the final output reminder")
    corpus = corpus[:reminder_index]

    matches = list(DOCUMENT_HEADER_RE.finditer(corpus))
    if not matches:
        raise ValueError("OBLIQ prompt contains no bracketed candidate documents")
    documents = [
        ObliqDocument(
            doc_id=match.group(1),
            text=corpus[match.end() : matches[index + 1].start()].strip()
            if index + 1 < len(matches)
            else corpus[match.end() :].strip(),
        )
        for index, match in enumerate(matches)
    ]
    ids = [document.doc_id for document in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("OBLIQ prompt contains duplicate candidate document IDs")
    if any(not document.text for document in documents):
        raise ValueError("OBLIQ prompt contains an empty candidate document")
    return documents


def build_obliq_batches(
    documents: Sequence[ObliqDocument],
    *,
    max_documents: int,
    max_chars: int,
) -> list[ObliqRankingBatch]:
    """Pack whole documents into bounded batches, preserving their order."""
    if max_documents < 1:
        raise ValueError("max_documents must be >= 1")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    batches: list[ObliqRankingBatch] = []
    current: list[ObliqDocument] = []
    current_chars = 0
    for document in documents:
        document_chars = len(document.doc_id) + len(document.text) + 4
        if document_chars > max_chars:
            raise ValueError(
                f"OBLIQ document {document.doc_id!r} has {document_chars} characters, "
                f"exceeding max_batch_chars={max_chars}"
            )
        if current and (
            len(current) >= max_documents or current_chars + document_chars > max_chars
        ):
            batches.append(ObliqRankingBatch(tuple(current)))
            current = []
            current_chars = 0
        current.append(document)
        current_chars += document_chars
    if current:
        batches.append(ObliqRankingBatch(tuple(current)))
    if not batches:
        raise ValueError("cannot build OBLIQ ranking batches from no documents")
    return batches


def ranking_prompt(
    query: str,
    batch: ObliqRankingBatch,
    *,
    stage: str,
) -> str:
    """Build a strict batch-local ranking request."""
    if stage not in {"map", "reduce"}:
        raise ValueError(f"unknown OBLIQ ranking stage {stage!r}")
    candidates = "\n\n".join(
        f"[{index}]\n{document.text}" for index, document in enumerate(batch.documents)
    )
    return (
        f"OBLIQ_LAMBDA_{stage.upper()}\n"
        "This is closed-corpus ranking. Compare the source problem with every candidate "
        "below. Select candidates that use the same underlying proof technique or aha "
        "insight, not merely the same topic. Return at most 10 candidate indices, "
        "best first. Indices are local to this request and must be integers shown in "
        "brackets below. Do not return document names, unseen indices, or explanations. "
        "Your entire response must be one line of valid JSON syntax: "
        "RANKED_INDICES: [7, 2, 19] or RANKED_INDICES: [].\n\n"
        f"SOURCE QUERY:\n{query}\n\nCANDIDATES:\n{candidates}"
    )


def project_ranking_response(
    response: str,
    batch_size: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Project a parseable model ranking onto valid batch-local indices.

    Removing impossible values, duplicates, and values beyond ``RANK_K`` is a
    deterministic typed composition step.  Every removal is returned for the
    audit rather than silently changing the model output.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    match = RANKED_INDICES_RE.fullmatch(response.strip())
    if match is None:
        raise ValueError("response is not one RANKED_INDICES JSON list")
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError(f"RANKED_INDICES is not valid JSON: {error.msg}") from error
    if not isinstance(parsed, list):
        raise ValueError("RANKED_INDICES must contain a JSON list")

    selected: list[int] = []
    dropped: list[str] = []
    seen: set[int] = set()
    for value in parsed:
        if not isinstance(value, int) or isinstance(value, bool):
            dropped.append(repr(value))
        elif not 0 <= value < batch_size:
            dropped.append(str(value))
        elif value in seen:
            dropped.append(str(value))
        elif len(selected) >= RANK_K:
            dropped.append(str(value))
        else:
            selected.append(value)
            seen.add(value)
    if parsed and not selected:
        raise ValueError("response contains no valid indices from this batch")
    return tuple(selected), tuple(dropped)


class ObliqLambdaRLM(PaperLambdaRLM):
    """λ-RLM adaptation whose intermediate values remain ranked document IDs."""

    def __init__(
        self,
        *args: Any,
        obliq_max_batch_documents: int = DEFAULT_OBLIQ_MAX_BATCH_DOCUMENTS,
        obliq_max_batch_chars: int = DEFAULT_OBLIQ_MAX_BATCH_CHARS,
        obliq_max_concurrency: int = 8,
        obliq_max_attempts: int = DEFAULT_OBLIQ_MAX_ATTEMPTS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.task_id is not None:
            raise ValueError("ObliqLambdaRLM does not accept an OOLONG-Pairs task_id")
        if obliq_max_batch_documents < 1:
            raise ValueError("obliq_max_batch_documents must be >= 1")
        if obliq_max_batch_chars < 1:
            raise ValueError("obliq_max_batch_chars must be >= 1")
        if obliq_max_batch_chars > self.context_window_chars:
            raise ValueError("obliq_max_batch_chars must not exceed context_window_chars")
        if obliq_max_concurrency < 1:
            raise ValueError("obliq_max_concurrency must be >= 1")
        if obliq_max_attempts < 1:
            raise ValueError("obliq_max_attempts must be >= 1")
        self.obliq_max_batch_documents = obliq_max_batch_documents
        self.obliq_max_batch_chars = obliq_max_batch_chars
        self.obliq_max_concurrency = obliq_max_concurrency
        self.obliq_max_attempts = obliq_max_attempts

    def completion(self, prompt: str) -> RLMChatCompletion:
        started = time.perf_counter()
        documents = parse_obliq_documents(prompt)
        map_batches = build_obliq_batches(
            documents,
            max_documents=self.obliq_max_batch_documents,
            max_chars=self.obliq_max_batch_chars,
        )
        client = upstream_lambda.get_client(self.backend, self.backend_kwargs)
        map_audits = asyncio.run(self.rank_batches(client, map_batches, stage="map"))
        candidates = self.candidate_documents(documents, map_audits)
        all_audits = list(map_audits)

        reduction_rounds = 0
        while candidates:
            reduction_rounds += 1
            batches = build_obliq_batches(
                candidates,
                max_documents=self.obliq_max_batch_documents,
                max_chars=self.obliq_max_batch_chars,
            )
            audits = asyncio.run(self.rank_batches(client, batches, stage="reduce"))
            all_audits.extend(audits)
            reduced = self.candidate_documents(candidates, audits)
            if len(batches) == 1:
                candidates = reduced
                break
            if len(reduced) >= len(candidates):
                raise RuntimeError(
                    "OBLIQ λ-RLM reduction did not shrink the candidate set; "
                    "decrease obliq_max_batch_documents or inspect model responses"
                )
            candidates = reduced

        response = "RANKED: " + json.dumps([document.doc_id for document in candidates])
        retried_batches = sum(len(audit.attempts) > 1 for audit in all_audits)
        degraded_batches = sum(audit.degraded for audit in all_audits)
        normalized_attempts = sum(
            bool(attempt.dropped_values) for audit in all_audits for attempt in audit.attempts
        )
        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "unknown"),
            prompt=prompt,
            response=response,
            usage_summary=client.get_usage_summary(),
            execution_time=time.perf_counter() - started,
            metadata={
                "obliq_lambda_audit": {
                    "format": OBLIQ_LAMBDA_AUDIT_FORMAT,
                    "version": OBLIQ_LAMBDA_VERSION,
                    "documents": len(documents),
                    "map_batches": len(map_batches),
                    "reduction_rounds": reduction_rounds,
                    "final_candidates": len(candidates),
                    "scoreable": bool(candidates),
                    "retried_batches": retried_batches,
                    "degraded_batches": degraded_batches,
                    "normalized_attempts": normalized_attempts,
                    "batches": [asdict(audit) for audit in all_audits],
                }
            },
        )

    def candidate_documents(
        self,
        documents: Sequence[ObliqDocument],
        audits: Sequence[ObliqRankingBatchAudit],
    ) -> list[ObliqDocument]:
        """Return selected documents in ranking order with stable deduplication."""
        by_id = {document.doc_id: document for document in documents}
        selected = dict.fromkeys(doc_id for audit in audits for doc_id in audit.selected_ids)
        return [by_id[doc_id] for doc_id in selected]

    async def rank_batches(
        self,
        client: BaseLM,
        batches: Sequence[ObliqRankingBatch],
        *,
        stage: str,
    ) -> list[ObliqRankingBatchAudit]:
        """Rank batches concurrently and retry only contract violations."""
        semaphore = asyncio.Semaphore(self.obliq_max_concurrency)

        async def rank(
            batch_index: int,
            batch: ObliqRankingBatch,
        ) -> ObliqRankingBatchAudit:
            base_prompt = ranking_prompt(
                self.query,
                batch,
                stage=stage,
            )
            rejection = ""
            attempts: list[ObliqRankingAttempt] = []
            async with semaphore:
                for attempt in range(1, self.obliq_max_attempts + 1):
                    request = (
                        base_prompt
                        if not rejection
                        else (
                            f"{base_prompt}\n\nYour previous response was rejected: {rejection}. "
                            f"Valid indices are 0 through {len(batch.documents) - 1}. "
                            "Respond again using the required one-line JSON format."
                        )
                    )
                    response = await client.acompletion(request)
                    try:
                        selected_indices, dropped_values = project_ranking_response(
                            response,
                            len(batch.documents),
                        )
                    except ValueError as error:
                        rejection = str(error)
                        attempts.append(ObliqRankingAttempt(attempt, response, rejection, (), ()))
                        continue
                    selected_ids = tuple(
                        batch.documents[index].doc_id for index in selected_indices
                    )
                    attempts.append(
                        ObliqRankingAttempt(
                            attempt,
                            response,
                            None,
                            selected_indices,
                            dropped_values,
                        )
                    )
                    return ObliqRankingBatchAudit(
                        stage=stage,
                        batch_index=batch_index,
                        document_ids=tuple(document.doc_id for document in batch.documents),
                        attempts=tuple(attempts),
                        selected_indices=selected_indices,
                        selected_ids=selected_ids,
                        degraded=False,
                    )
            return ObliqRankingBatchAudit(
                stage=stage,
                batch_index=batch_index,
                document_ids=tuple(document.doc_id for document in batch.documents),
                attempts=tuple(attempts),
                selected_indices=(),
                selected_ids=(),
                degraded=True,
            )

        return list(
            await asyncio.gather(*(rank(index, batch) for index, batch in enumerate(batches)))
        )


__all__ = [
    "DEFAULT_OBLIQ_MAX_ATTEMPTS",
    "DEFAULT_OBLIQ_MAX_BATCH_CHARS",
    "DEFAULT_OBLIQ_MAX_BATCH_DOCUMENTS",
    "OBLIQ_LAMBDA_AUDIT_FORMAT",
    "OBLIQ_LAMBDA_VERSION",
    "ObliqDocument",
    "ObliqLambdaRLM",
    "ObliqRankingBatch",
    "build_obliq_batches",
    "parse_obliq_documents",
    "ranking_prompt",
    "project_ranking_response",
]
