"""History keeps identity and uncertainty independent of prompt compaction."""

import json

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
    kwargs = dict(
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
