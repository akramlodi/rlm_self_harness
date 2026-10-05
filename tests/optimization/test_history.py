"""History keeps identity and uncertainty independent of prompt compaction."""

import json
from typing import Any

from shrlm.optimization.history import HISTORY_BUDGET_CHARS, prior_evaluations, revision_violation
from shrlm.optimization.proposal import _render_history_block


def test_history_budget_keeps_qualified_gains_and_all_attempt_index():
    history = []
    for index in range(200):
        record = {
            "subject_id": f"candidate-{index}",
            "decision": "rejected",
            "surface": "S6",
            "incumbent_hash": "base",
            "harness_hash": f"hash-{index}",
            "links": {"summary": "saved"},
            "predicted_effect": "x" * 600,
            "behavioral_change": "y" * 600,
            "diagnostics": {"exact_passes": 0, "quality": {"mean": 0.7193}},
            "diagnostic_progress": {
                "status": "potentially_promising" if index == 6 else "no_measured_improvement",
                "baseline": {"quality": {"mean": 0.6194}},
                "candidate": {"quality": {"mean": 0.7193}},
            },
        }
        history.append(
            (
                [record],
                {
                    "round": index,
                    "baseline_diagnostics": {"exact_passes": 1, "quality": {"mean": 0.6194}},
                },
            )
        )
    text = _render_history_block(history)
    assert len(text) <= HISTORY_BUDGET_CHARS
    assert "candidate-6" in text and "0.6194" in text and "0.7193" in text
    assert "not_assessed" in text
    audit = json.loads(text.splitlines()[0])
    assert audit["attempts_omitted"] > 0 and audit["status_totals"] == {"rejected": 200}
    assert len(prior_evaluations(history)) == 200


def test_revision_requires_resolvable_predecessor_even_when_not_rendered():
    attempts = [
        {
            "round": 6,
            "subject_id": "policy",
            "surface": "S6",
            "mechanism": "iteration_budget_exhaustion",
            "effective_edit_fingerprint": "same",
            "behavioral_change": "Retry syntax errors without changing the prompt.",
        }
    ]
    kwargs: dict[str, Any] = dict(
        surface="S6", mechanism="iteration_budget_exhaustion", fingerprint="same", attempts=attempts
    )
    assert "revision" in revision_violation(None, **kwargs)
    assert "Retry syntax errors without changing the prompt." in revision_violation(None, **kwargs)
    assert (
        revision_violation(
            {
                "round": 6,
                "subject_id": "policy",
                "explanation": "Changed joint intervention; this member is unchanged.",
            },
            **kwargs,
        )
        is None
    )
    assert "resolve" in revision_violation(
        {"round": 7, "subject_id": "missing", "explanation": "changed"}, **kwargs
    )


def test_recent_rounds_and_incumbent_promotion_survive_verbose_old_history():
    history = []
    for index in range(1, 7):
        record = {
            "subject_id": f"edit-{index}",
            "surface": "S4",
            "mechanism": "skipped_verification",
            "decision": "promoted" if index == 2 else "rejected",
            "links": {"summary": "saved"},
            "behavioral_change": "detail " * 6000,
            "diagnostics": {"exact_passes": index, "quality": {"mean": index / 10}},
            "diagnostic_progress": {
                "status": "potentially_promising" if index == 1 else "not_assessed"
            },
        }
        history.append(
            (
                [record],
                {
                    "round": index,
                    "promoted": index == 2,
                    "promoted_harness_hash": "current" if index == 2 else None,
                },
            )
        )
    rendered = _render_history_block(history)
    assert len(rendered) <= HISTORY_BUDGET_CHARS
    for index in (1, 2, 4, 5, 6):
        assert f"edit-{index}" in rendered


def test_evaluated_predecessor_beats_refused_rewrite_but_exact_match_wins():
    from shrlm.optimization.history import select_predecessor

    measured = {
        "round": 3,
        "subject_id": "kept",
        "surface": "S4",
        "mechanism": "skipped_verification",
        "decision": "bundled",
        "effective_edit_fingerprint": "kept-hash",
    }
    refused = {
        **measured,
        "subject_id": "rewrite",
        "decision": "preflight_rejected",
        "effective_edit_fingerprint": "other-hash",
    }
    kwargs: dict[str, Any] = dict(
        surface="S4", mechanism="skipped_verification", attempts=[measured, refused]
    )
    assert select_predecessor(**kwargs) == measured
    violation = revision_violation(None, fingerprint="new", **kwargs)
    assert violation is not None and "subject kept" in violation
    assert select_predecessor(**kwargs, fingerprint="other-hash") == refused
    violation = revision_violation(None, fingerprint="other-hash", **kwargs)
    assert violation is not None and "subject rewrite" in violation


def test_required_identity_is_whole_or_its_choice_is_withheld():
    from shrlm.optimization.history import compact_history

    records = [
        {"subject_id": "a" * 500, "surface": "S3", "decision": "bundled"},
        {"subject_id": "b" * 500, "surface": "S4", "decision": "bundled"},
    ]
    choices = {
        str(i): {"predecessors": {r["surface"]: {"round": 1, "subject_id": r["subject_id"]}}}
        for i, r in enumerate(records)
    }
    text, withheld = compact_history([(records, {"round": 1})], choices=choices, budget=1900)
    assert len(text) <= 1900
    assert withheld == ["1"]
    assert records[0]["subject_id"] in text
    assert records[1]["subject_id"] not in text


def test_ambiguous_predecessor_withholds_only_the_affected_choice():
    from shrlm.optimization.history import compact_history

    records = [
        {"subject_id": "duplicate", "surface": "S3", "decision": "bundled"},
        {"subject_id": "duplicate", "surface": "S3", "decision": "preflight_rejected"},
        {"subject_id": "unique", "surface": "S4", "decision": "bundled"},
    ]
    choices = {
        str(i): {"predecessors": {r["surface"]: {"round": 1, "subject_id": r["subject_id"]}}}
        for i, r in enumerate((records[0], records[2]))
    }
    text, withheld = compact_history([(records, {"round": 1})], choices=choices)
    assert withheld == ["0"]
    assert any(
        row["subject_id"] == "unique" for row in json.loads(text.splitlines()[1])["attempts"]
    )


def test_conditional_route_predecessor_is_visible_in_frozen_choice():
    from shrlm.harness_identity import serialize_harness
    from shrlm.optimization.proposal import render_prompt
    from shrlm.rlm_harness import H0
    from tests.optimization.test_proposal import make_pattern, synthetic_evidence

    pattern = make_pattern("lossy_aggregation")
    record = {
        "subject_id": "helper",
        "surface": "S8",
        "mechanism": "lossy_aggregation",
        "decision": "bundled",
    }
    audit = {}
    prompt, _ = render_prompt(
        [pattern],
        serialize_harness(H0),
        [],
        [([record], {"round": 3})],
        4,
        evidence=synthetic_evidence([pattern]),
        evidence_audit=audit,
    )
    assert audit["choices"]["0"]["predecessors"]["S8"] == {
        "round": 3,
        "subject_id": "helper",
        "decision": "bundled",
    }
    assert '"subject_id": "helper"' in prompt


def test_old_promising_batch_is_related_through_members():
    from shrlm.optimization.history import compact_history

    history = [
        (
            [
                {
                    "subject_id": "member",
                    "surface": "S4",
                    "mechanism": "skipped_verification",
                    "decision": "bundled",
                },
                {
                    "subject_id": "batch",
                    "decision": "rejected",
                    "links": {"summary": "saved"},
                    "merge": {"constituent_ids": ["member"]},
                    "diagnostic_progress": {"status": "potentially_promising"},
                    "diagnostics": {"quality": {"mean": 0.8}},
                },
            ],
            {"round": 1},
        )
    ] + [([], {"round": i}) for i in range(2, 5)]
    text, _ = compact_history(history, mechanisms={"skipped_verification"})
    facts = json.loads(text.splitlines()[1])
    assert [r["round"] for r in facts["rounds"]] == [1, 2, 3, 4]
    assert facts["rounds"][0]["outcomes"][0]["quality_signal"] == "potentially_promising"


def test_activation_keeps_detector_counts_and_intended_behavior_uncertainty():
    from shrlm.optimization.behavior import BEHAVIOR_SCHEMA, activation_for_surface
    from shrlm.optimization.history import compact_history

    activation = activation_for_surface(
        "S9",
        {
            "schema": BEHAVIOR_SCHEMA,
            "n_runs": 10,
            "answer_redirects": {"count": 3, "n_measured_runs": 8},
        },
    )
    text, _ = compact_history(
        [
            (
                [{"subject_id": "middleware", "decision": "bundled", "activation": activation}],
                {"round": 1},
            )
        ]
    )
    row = json.loads(text.splitlines()[1])["attempts"][0]
    assert row["activation"] == activation
