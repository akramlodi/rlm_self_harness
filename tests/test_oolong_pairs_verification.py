"""Tests for the OOLONG-Pairs deterministic sub-verifier (examples/oolong_pairs/verification.py)."""

from datetime import date

from examples.oolong_pairs.verification import _prf1, verify_child


class TestPrf1:
    def test_toy_pairs_example(self):
        """Spec example: a toy gold set, restricted to the pairs relevant to a child that
        only classified user 1's entries, checked against a child output with one correct
        and one incorrect pair -- TP=[(1,2)], FP=[(1,4)], FN=[(1,3)], P=R=F1=0.5."""
        gold_pairs = [(1, 2), (1, 3), (2, 4)]
        child_users = {1}
        expected = {p for p in gold_pairs if p[0] in child_users or p[1] in child_users}
        actual = {(1, 2), (1, 4)}

        result = _prf1(expected, actual)

        assert result["tp"] == [(1, 2)]
        assert result["fp"] == [(1, 4)]
        assert result["fn"] == [(1, 3)]
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5
        assert result["pass"] is False

    def test_exact_match_passes(self):
        expected = {(1, 2), (3, 4)}
        actual = {(1, 2), (3, 4)}

        result = _prf1(expected, actual)

        assert result["tp"] == [(1, 2), (3, 4)]
        assert result["fp"] == []
        assert result["fn"] == []
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["pass"] is True

    def test_both_empty_is_vacuous_pass(self):
        result = _prf1(set(), set())

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["pass"] is True

    def test_no_overlap_scores_zero(self):
        result = _prf1({(1, 2)}, {(3, 4)})

        assert result["tp"] == []
        assert result["fp"] == [(3, 4)]
        assert result["fn"] == [(1, 2)]
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["pass"] is False


class TestVerifyChild:
    def _gold(self) -> dict:
        # Task 1: eligible if a user has >=1 'numeric value' or 'location' instance.
        return {
            1: [("numeric value", date(2023, 1, 1))],
            2: [("location", date(2023, 1, 2))],
            3: [("entity", date(2023, 1, 3))],  # not eligible for task 1
        }

    def test_fully_resolved_chunk_correct_child_passes(self):
        gold = self._gold()
        # This chunk contains ALL of user 1's and user 2's entries (1 each) -- both resolved.
        child_scope = [("Jan 01, 2023", 1, "q1"), ("Jan 02, 2023", 2, "q2")]
        child_output = [
            {"user_id": 1, "date": "Jan 01, 2023", "label": "numeric value"},
            {"user_id": 2, "date": "Jan 02, 2023", "label": "location"},
        ]

        result = verify_child(child_scope, child_output, gold, task=1)

        assert result["resolved_users"] == [1, 2]
        assert result["tp"] == [(1, 2)]
        assert result["fp"] == []
        assert result["fn"] == []
        assert result["pass"] is True

    def test_misclassified_child_fails(self):
        gold = self._gold()
        child_scope = [("Jan 01, 2023", 1, "q1"), ("Jan 02, 2023", 2, "q2")]
        # Child mislabels user 2's instance, so user 2 no longer looks eligible.
        child_output = [
            {"user_id": 1, "date": "Jan 01, 2023", "label": "numeric value"},
            {"user_id": 2, "date": "Jan 02, 2023", "label": "abbreviation"},
        ]

        result = verify_child(child_scope, child_output, gold, task=1)

        assert result["pass"] is False
        assert result["fn"] == [(1, 2)]
        assert result["tp"] == []

    def test_unresolved_chunk_is_inconclusive(self):
        # User 1 has TWO entries total in the ground truth, but this chunk only has one --
        # user 1 is not fully resolved, so fewer than 2 users are resolved overall.
        gold = {1: [("numeric value", date(2023, 1, 1)), ("location", date(2023, 1, 5))]}
        child_scope = [("Jan 01, 2023", 1, "q1")]
        child_output = [{"user_id": 1, "date": "Jan 01, 2023", "label": "numeric value"}]

        result = verify_child(child_scope, child_output, gold, task=1)

        assert result["pass"] is None
        assert result["precision"] is None
        assert result["resolved_users"] == []
        assert "note" in result
