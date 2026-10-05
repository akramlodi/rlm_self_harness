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
    project_ranking_response,
    ranking_prompt,
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


def test_projects_indices_and_audits_impossible_values() -> None:
    selected, dropped = project_ranking_response(
        'RANKED_INDICES: [2, "invented", 2, 9, 0]',
        3,
    )

    assert selected == (2, 0)
    assert dropped == ("'invented'", "2", "9")


def test_rejects_unparseable_or_entirely_invalid_indices() -> None:
    with pytest.raises(ValueError, match="one RANKED_INDICES"):
        project_ranking_response('RANKED: ["id-a"]', 1)
    with pytest.raises(ValueError, match="no valid indices"):
        project_ranking_response("RANKED_INDICES: [4]", 1)


def test_ranking_prompt_exposes_local_indices_without_document_ids() -> None:
    documents = parse_obliq_documents(
        build_prompt([("long-document-id", "candidate text")], "source problem")
    )

    request = ranking_prompt(
        "find analogues",
        build_obliq_batches(documents, max_documents=2, max_chars=1_000)[0],
        stage="map",
    )

    assert "[0]\ncandidate text" in request
    assert "long-document-id" not in request


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
            "RANKED_INDICES: [0, 1, 99]",
            "RANKED_INDICES: [0, 1]",
            "RANKED_INDICES: [0]",
            "RANKED_INDICES: [0]",
            "RANKED_INDICES: [1, 0]",
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
    assert audit["normalized_attempts"] == 1
    assert audit["batches"][0]["attempts"][0]["dropped_values"] == ("99",)
    assert factory.total_calls == 5


def test_exhausted_batch_is_audited_without_discarding_other_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "not a ranking",
            "still not a ranking",
            "RANKED_INDICES: [99]",
            "RANKED_INDICES: [0]",
            "RANKED_INDICES: [0]",
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

    assert completion.response == 'RANKED: ["id-c"]'
    audit = completion.metadata["obliq_lambda_audit"]
    assert audit["scoreable"] is True
    assert audit["retried_batches"] == 1
    assert audit["degraded_batches"] == 1
    assert audit["batches"][0]["degraded"] is True
    assert audit["batches"][1]["selected_indices"] == (0,)


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
    assert envelope["method"]["adaptation"]["version"] == "2"
    json.dumps(envelope, allow_nan=False)
