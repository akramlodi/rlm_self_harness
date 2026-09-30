"""Deterministic tests for the separately named OBLIQ λ-RLM adaptation."""

import json

import pytest

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from shrlm.baselines.lambda_rlm import (
    LambdaBaselineConfig,
    ObliqLambdaBaselineConfig,
    lambda_method_envelope,
    lambda_method_hash,
)
from shrlm.baselines.obliq_lambda_rlm import (
    ObliqLambdaRLM,
    build_obliq_batches,
    parse_obliq_documents,
    validate_ranking_response,
)
from shrlm.environments.obliq_bench_math import build_prompt
from tests.optimization.test_driver import ClientFactory


def test_parses_and_batches_only_at_document_boundaries() -> None:
    prompt = build_prompt(
        [("id-a", "first problem"), ("id-b", "second problem"), ("id-c", "third")],
        "source problem",
    )

    documents = parse_obliq_documents(prompt)
    batches = build_obliq_batches(documents, max_documents=2, max_chars=1_000)

    assert [(document.doc_id, document.text) for document in documents] == [
        ("id-a", "first problem"),
        ("id-b", "second problem"),
        ("id-c", "third"),
    ]
    assert [[document.doc_id for document in batch.documents] for batch in batches] == [
        ["id-a", "id-b"],
        ["id-c"],
    ]


def test_rejects_unknown_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="outside this batch"):
        validate_ranking_response('RANKED: ["invented"]', frozenset({"id-a"}))
    with pytest.raises(ValueError, match="duplicate"):
        validate_ranking_response('RANKED: ["id-a", "id-a"]', frozenset({"id-a"}))


def test_map_union_is_globally_reranked(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = build_prompt(
        [
            ("id-a", "first problem"),
            ("id-b", "second problem"),
            ("id-c", "third problem"),
            ("id-d", "fourth problem"),
        ],
        "source problem",
    )
    factory = ClientFactory(
        [
            'RANKED: ["id-a", "id-b"]',
            'RANKED: ["id-c", "id-d"]',
            'RANKED: ["id-a"]',
            'RANKED: ["id-c"]',
            'RANKED: ["id-c", "id-a"]',
        ]
    )
    monkeypatch.setattr(upstream_lambda, "get_client", factory)
    method = ObliqLambdaRLM(
        backend="openai",
        backend_kwargs={"model_name": "test"},
        environment="local",
        context_window_chars=10_000,
        query="find analogues",
        pairwise_max_batch_chars=1_000,
        obliq_max_batch_documents=2,
        obliq_max_batch_chars=1_000,
        obliq_max_concurrency=1,
    )

    completion = method.completion(prompt)

    assert completion.response == 'RANKED: ["id-c", "id-a"]'
    audit = completion.metadata["obliq_lambda_audit"]
    assert audit["map_batches"] == 2
    assert audit["reduction_rounds"] == 2
    assert audit["final_candidates"] == 2
    assert factory.total_calls == 5


def test_adaptation_has_distinct_identity_without_changing_upstream_identity() -> None:
    generic = LambdaBaselineConfig()
    adapted = ObliqLambdaBaselineConfig()

    assert (
        lambda_method_hash(generic)
        == "2ff8b52c4482fdebcea6d6c69ba848ece98f8cb0ef0f68ffcfbcfbaf00137072"
    )
    assert lambda_method_hash(adapted) != lambda_method_hash(generic)
    envelope = lambda_method_envelope(adapted)
    assert envelope["kind"] == "lambda_rlm_obliq_adaptation"
    assert envelope["method"]["configuration"]["obliq_max_batch_documents"] == 64
    assert envelope["method"]["configuration"]["obliq_max_batch_chars"] == 25_000
    assert envelope["method"]["configuration"]["obliq_max_attempts"] == 3
    assert envelope["method"]["adaptation"]["scope"] == ("OBLIQ document-boundary map-rank-reduce")
    json.dumps(envelope, allow_nan=False)
