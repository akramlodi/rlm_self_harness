"""Tests for the OOLONG-Pairs paper reconstruction (Algorithm 5).

Covers the pieces the released upstream ``lambda_rlm.py`` does not implement:
symbolic record parsing, token/output-bounded batching, strict classification
parsing, cross-chunk aggregation, and the bounded SPLIT -> MAP -> PARSE ->
FILTER -> CROSS call shape. Everything runs offline against a fake classifier
client; no network or real model calls.
"""

import re
from datetime import date

import pytest

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from shrlm.baselines.paper_lambda_rlm import (
    LABEL_TO_CODE,
    PAIRWISE_AUDIT_FORMAT,
    PAPER_RECONSTRUCTION_VERSION,
    PaperLambdaRLM,
    aggregate_predictions,
    build_classification_batches,
    classification_prompt,
    format_pairs,
    parse_classifications,
    parse_oolong_prompt,
)
from shrlm.environments.oolong_pairs import TASK_TEXTS, OolongEntry, build_prompt
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_entry(user_id: int, day: int, question: str) -> OolongEntry:
    return OolongEntry(user_id=user_id, date=date(2023, 2, day), instance=question, label=None)


class FakeClassifierLM(MockLM):
    """Classifies every question in a batch from its own prompt content.

    Deterministic and order-independent: it reads the "<index>\\t<question>"
    lines straight out of the prompt it receives, so it produces the correct
    answer regardless of which batch asyncio.gather happens to run first.
    """

    def __init__(self, label_by_question: dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self._label_by_question = label_by_question
        self.prompts_seen: list[str] = []

    async def acompletion(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        items_block = prompt.split("Questions:\n", 1)[1]
        lines = []
        for line in items_block.splitlines():
            if not line.strip():
                continue
            index_str, question = line.split("\t", 1)
            label = self._label_by_question[question]
            lines.append(f"{index_str}|{LABEL_TO_CODE[label]}")
        return "\n".join(lines)


def build_paper_method(
    task_id: int,
    *,
    label_by_question: dict[str, str],
    max_batch_records: int = 256,
    max_batch_chars: int = 80_000,
) -> tuple[PaperLambdaRLM, FakeClassifierLM]:
    method = PaperLambdaRLM(
        backend="openrouter",
        backend_kwargs={"model_name": "test-model"},
        query=TASK_TEXTS[task_id],
        task_id=task_id,
        pairwise_max_batch_records=max_batch_records,
        pairwise_max_batch_chars=max_batch_chars,
    )
    client = FakeClassifierLM(label_by_question=label_by_question)
    return method, client


# ---------------------------------------------------------------------------
# parse_oolong_prompt
# ---------------------------------------------------------------------------


class TestParseOolongPrompt:
    def test_extracts_unlabeled_entries_only(self):
        entries = [make_entry(1, 2, "How many moons does Mars have ?")]
        prompt = build_prompt(entries, TASK_TEXTS[1])

        parsed = parse_oolong_prompt(prompt)

        assert len(parsed) == 1
        assert parsed[0].user_id == 1
        assert parsed[0].instance == "How many moons does Mars have ?"
        assert parsed[0].label is None

    def test_rejects_declared_count_mismatch(self):
        entries = [make_entry(1, 2, "Q1 ?"), make_entry(2, 3, "Q2 ?")]
        prompt = build_prompt(entries, TASK_TEXTS[1])
        tampered = prompt.replace("contain 2 general-knowledge", "contain 3 general-knowledge")

        with pytest.raises(ValueError, match="declares 3 records"):
            parse_oolong_prompt(tampered)

    def test_rejects_leaked_dataset_labels(self):
        entries = [make_entry(1, 2, "Q1 ?")]
        prompt = build_prompt(entries, TASK_TEXTS[1])
        leaked = prompt.replace("|| Instance: Q1 ?", "|| Instance: Q1 ? || Label: numeric value")

        with pytest.raises(ValueError, match="exposes dataset labels"):
            parse_oolong_prompt(leaked)

    def test_rejects_missing_header(self):
        with pytest.raises(ValueError, match="missing its declared record count"):
            parse_oolong_prompt("Date: Feb 02, 2023 || User: 1 || Instance: Q ?")


# ---------------------------------------------------------------------------
# build_classification_batches
# ---------------------------------------------------------------------------


class TestBuildClassificationBatches:
    def entries(self, n: int) -> list[OolongEntry]:
        return [make_entry(u, 2, f"Question {u} ?") for u in range(n)]

    def test_splits_on_max_records(self):
        batches = build_classification_batches(self.entries(5), max_records=2, max_chars=10_000)

        assert [len(b.entries) for b in batches] == [2, 2, 1]
        # Every record appears exactly once, in order.
        flat = [index for batch in batches for index, _ in batch.entries]
        assert flat == list(range(5))

    def test_splits_on_char_budget(self):
        entries = self.entries(4)
        one_record_prompt_len = len(classification_prompt([entries[0]]))

        batches = build_classification_batches(
            entries, max_records=100, max_chars=one_record_prompt_len + 5
        )

        assert all(len(b.entries) == 1 for b in batches)
        assert len(batches) == 4

    def test_batch_indices_are_local_not_global(self):
        # A live run against a 262k-token instance showed a model
        # mistranscribing a large global index (e.g. 3328) across a long
        # classification list. Batches must number items by their own
        # position, never by the record's position in the full instance,
        # so the model is never asked to echo back a large number.
        entries = self.entries(2)
        batches = build_classification_batches(entries, max_records=1, max_chars=10_000)

        assert len(batches) == 2
        assert batches[1].entries[0][0] == 1  # global index of the 2nd record
        assert "0\tQuestion 1 ?" in batches[1].prompt  # but the model sees local 0
        assert "1\tQuestion 1 ?" not in batches[1].prompt

    def test_single_record_exceeding_budget_raises(self):
        entries = self.entries(1)
        with pytest.raises(ValueError, match="cannot fit"):
            build_classification_batches(entries, max_records=100, max_chars=5)

    def test_rejects_non_positive_limits(self):
        with pytest.raises(ValueError, match="max_batch_records"):
            build_classification_batches(self.entries(1), max_records=0, max_chars=100)
        with pytest.raises(ValueError, match="max_batch_chars"):
            build_classification_batches(self.entries(1), max_records=1, max_chars=0)


class TestClassificationPrompt:
    def test_rejects_empty_batch(self):
        with pytest.raises(ValueError, match="must not be empty"):
            classification_prompt([])


# ---------------------------------------------------------------------------
# parse_classifications
# ---------------------------------------------------------------------------


class TestParseClassifications:
    def test_happy_path(self):
        result = parse_classifications("0|E\n1|N", [0, 1])
        assert result == {0: "entity", 1: "numeric value"}

    def test_tolerates_pipe_colon_tab_comma_separators(self):
        result = parse_classifications("0: E\n1\tN\n2,H", [0, 1, 2])
        assert result == {0: "entity", 1: "numeric value", 2: "human being"}

    def test_malformed_line_raises(self):
        with pytest.raises(ValueError, match="malformed OOLONG classification line"):
            parse_classifications("0|E\nnot a classification line", [0])

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="malformed OOLONG classification line"):
            parse_classifications("0|Z", [0])

    def test_unexpected_index_raises(self):
        with pytest.raises(ValueError, match="unexpected OOLONG classification index"):
            parse_classifications("0|E\n1|N", [0])

    def test_duplicate_index_raises(self):
        with pytest.raises(ValueError, match="duplicate OOLONG classification index"):
            parse_classifications("0|E\n0|N", [0])

    def test_missing_index_raises(self):
        with pytest.raises(ValueError, match="missing OOLONG classification indices"):
            parse_classifications("0|E", [0, 1])


# ---------------------------------------------------------------------------
# aggregate_predictions
# ---------------------------------------------------------------------------


class TestAggregatePredictions:
    def test_merges_same_user_across_chunk_boundaries(self):
        entries = [
            make_entry(7, 2, "Q0 ?"),
            make_entry(9, 3, "Q1 ?"),
            make_entry(7, 4, "Q2 ?"),
        ]
        predictions = {0: "entity", 1: "location", 2: "abbreviation"}

        by_user = aggregate_predictions(entries, predictions)

        assert {label for label, _ in by_user[7]} == {"entity", "abbreviation"}
        assert by_user[9] == [("location", date(2023, 2, 3))]

    def test_incomplete_coverage_raises(self):
        entries = [make_entry(1, 2, "Q0 ?"), make_entry(2, 3, "Q1 ?")]
        with pytest.raises(ValueError, match="do not cover every OOLONG record"):
            aggregate_predictions(entries, {0: "entity"})


# ---------------------------------------------------------------------------
# format_pairs
# ---------------------------------------------------------------------------


class TestFormatPairs:
    def test_empty_returns_explicit_marker(self):
        assert format_pairs([]) == "No valid pairs found."

    def test_formats_in_supplied_order(self):
        assert format_pairs([(1, 2), (3, 4)]) == "(1, 2)\n(3, 4)"


# ---------------------------------------------------------------------------
# PaperLambdaRLM construction guards
# ---------------------------------------------------------------------------


class TestPaperLambdaRLMConstruction:
    def test_rejects_out_of_range_task_id(self):
        with pytest.raises(ValueError, match="task_id must be in 1-20"):
            PaperLambdaRLM(backend="openrouter", backend_kwargs={}, task_id=21)

    def test_rejects_batch_chars_larger_than_context_window(self):
        with pytest.raises(ValueError, match="must not exceed context_window_chars"):
            PaperLambdaRLM(
                backend="openrouter",
                backend_kwargs={},
                task_id=1,
                context_window_chars=1_000,
                pairwise_max_batch_chars=2_000,
            )


# ---------------------------------------------------------------------------
# End-to-end pairwise_completion
# ---------------------------------------------------------------------------


class TestPairwiseCompletionEndToEnd:
    def test_bounded_calls_and_correct_symbolic_pairs(self, monkeypatch: pytest.MonkeyPatch):
        # Task 1: pairs of users who both have >= 1 numeric-value-or-location
        # instance. User 1 (numeric) and user 2 (location) are eligible; user
        # 3 (entity) is not.
        entries = [
            make_entry(1, 2, "How many moons does Mars have ?"),
            make_entry(2, 3, "Where is the Eiffel Tower located ?"),
            make_entry(3, 4, "What is jazz ?"),
        ]
        label_by_question = {
            "How many moons does Mars have ?": "numeric value",
            "Where is the Eiffel Tower located ?": "location",
            "What is jazz ?": "description and abstract concept",
        }
        method, client = build_paper_method(
            1, label_by_question=label_by_question, max_batch_records=2
        )
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)
        prompt = build_prompt(entries, TASK_TEXTS[1])

        completion = method.completion(prompt)

        assert completion.response == "(1, 2)"
        # 3 records at max_batch_records=2 -> exactly 2 batches -> 2 calls.
        assert len(client.prompts_seen) == 2
        assert method.last_pairwise_trace is not None
        assert method.last_pairwise_trace.phase == "complete"
        assert method.last_pairwise_trace.pairs == 1
        assert method.last_pairwise_trace.model_call_upper_bound == 6
        assert completion.metadata is not None
        audit = completion.metadata["pairwise_audit"]
        assert audit["format"] == PAIRWISE_AUDIT_FORMAT
        assert audit["reconstruction_version"] == PAPER_RECONSTRUCTION_VERSION
        assert audit["actual_model_calls"] == 2
        assert audit["execution"]["model_call_upper_bound"] == 6
        assert audit["label_counts"] == {
            "description and abstract concept": 1,
            "entity": 0,
            "human being": 0,
            "numeric value": 1,
            "location": 1,
            "abbreviation": 0,
        }
        assert audit["batches"][0]["global_indices"] == (0, 1)
        assert audit["batches"][0]["attempts"] == (
            {"attempt": 1, "response": "0|N\n1|L", "rejection": None},
        )
        assert audit["batches"][0]["predictions"] == {
            0: "numeric value",
            1: "location",
        }

    def test_no_eligible_pairs_returns_explicit_marker(self, monkeypatch: pytest.MonkeyPatch):
        entries = [make_entry(1, 2, "What is jazz ?")]
        label_by_question = {"What is jazz ?": "description and abstract concept"}
        method, client = build_paper_method(1, label_by_question=label_by_question)
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)

        completion = method.completion(build_prompt(entries, TASK_TEXTS[1]))

        assert completion.response == "No valid pairs found."

    def test_same_user_spanning_multiple_chunks_is_aggregated_before_predicate(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Task 11 (asymmetric): one user needs >=1 entity AND >=1
        # abbreviation; the other needs EXACTLY one entity. User 7's two
        # qualifying instances are forced into separate batches
        # (max_batch_records=1), so the pair only exists if predictions are
        # aggregated globally by user before FILTER/CROSS runs.
        entries = [
            make_entry(7, 2, "Which company is IBM ?"),  # -> abbreviation
            make_entry(3, 3, "Who is the mayor ?"),  # -> entity (role b)
            make_entry(7, 4, "Name the river ?"),  # -> entity (role a, other half)
        ]
        label_by_question = {
            "Which company is IBM ?": "abbreviation",
            "Who is the mayor ?": "entity",
            "Name the river ?": "entity",
        }
        method, client = build_paper_method(
            11, label_by_question=label_by_question, max_batch_records=1
        )
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)

        completion = method.completion(build_prompt(entries, TASK_TEXTS[11]))

        assert completion.response == "(3, 7)"
        assert len(client.prompts_seen) == 3

    def test_malformed_classification_response_exhausts_retries_then_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        entries = [make_entry(1, 2, "Q0 ?")]

        class BrokenLM(MockLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.call_count = 0

            async def acompletion(self, prompt: str) -> str:
                self.call_count += 1
                return "not a valid classification line"

        method = PaperLambdaRLM(
            backend="openrouter",
            backend_kwargs={"model_name": "test-model"},
            query=TASK_TEXTS[1],
            task_id=1,
            pairwise_max_attempts=2,
        )
        client = BrokenLM()
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)

        with pytest.raises(ValueError, match="batch 0 still rejected after 2 attempts"):
            method.completion(build_prompt(entries, TASK_TEXTS[1]))
        # Every attempt was spent -- not one call, not unbounded retrying.
        assert client.call_count == 2

    def test_transient_batch_rejection_is_retried_and_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A model that fumbles the format once (a real live-run failure mode:
        # a mistranscribed index or a stray line) must not fail the whole
        # run -- the batch gets one more try before anything is given up on.
        entries = [
            make_entry(1, 2, "How many moons does Mars have ?"),
            make_entry(2, 3, "Where is the Eiffel Tower located ?"),
        ]
        label_by_question = {
            "How many moons does Mars have ?": "numeric value",
            "Where is the Eiffel Tower located ?": "location",
        }

        class RejectsOnceThenCorrectLM(MockLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.call_count = 0

            async def acompletion(self, prompt: str) -> str:
                self.call_count += 1
                if self.call_count == 1:
                    return "not a valid classification line"
                item_lines = [line for line in prompt.splitlines() if re.match(r"^\d+\t", line)]
                lines = []
                for line in item_lines:
                    index_str, question = line.split("\t", 1)
                    label = label_by_question[question]
                    lines.append(f"{index_str}|{LABEL_TO_CODE[label]}")
                return "\n".join(lines)

        method = PaperLambdaRLM(
            backend="openrouter",
            backend_kwargs={"model_name": "test-model"},
            query=TASK_TEXTS[1],
            task_id=1,
        )
        client = RejectsOnceThenCorrectLM()
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)

        completion = method.completion(build_prompt(entries, TASK_TEXTS[1]))

        assert completion.response == "(1, 2)"
        assert client.call_count == 2
        assert completion.metadata is not None
        attempts = completion.metadata["pairwise_audit"]["batches"][0]["attempts"]
        assert attempts[0]["rejection"] == (
            "malformed OOLONG classification line: 'not a valid classification line'"
        )
        assert attempts[1]["rejection"] is None

    def test_large_instance_call_bound_stays_linear(self, monkeypatch: pytest.MonkeyPatch):
        # 500 records at 50/batch must cost exactly 10 classification calls --
        # not one recursive/QA call per record and not O(n^2).
        n_records = 500
        max_records = 50
        entries = [make_entry(u, 2, f"Question {u} ?") for u in range(n_records)]
        label_by_question = {f"Question {u} ?": "entity" for u in range(n_records)}
        method, client = build_paper_method(
            2, label_by_question=label_by_question, max_batch_records=max_records
        )
        monkeypatch.setattr(upstream_lambda, "get_client", lambda *a, **k: client)

        completion = method.completion(build_prompt(entries, TASK_TEXTS[2]))

        assert len(client.prompts_seen) == n_records // max_records
        assert method.last_pairwise_trace.model_call_upper_bound == (
            n_records // max_records * method.pairwise_max_attempts
        )
        # Task 2 is eligible on entity or human being; all 500 users qualify,
        # so C(500, 2) pairs come out purely from the symbolic CROSS step.
        assert len(completion.response.splitlines()) == n_records * (n_records - 1) // 2

    def test_task_id_none_falls_back_to_upstream_completion(self, monkeypatch: pytest.MonkeyPatch):
        from tests.optimization.test_driver import ClientFactory

        factory = ClientFactory(["2", "RIGHT"])
        monkeypatch.setattr(upstream_lambda, "get_client", factory)
        method = PaperLambdaRLM(
            backend="openrouter",
            backend_kwargs={"model_name": "test-model"},
            query="Which answer is correct?",
            task_id=None,
        )

        completion = method.completion("A short context containing the answer.")

        assert completion.response == "RIGHT"
        assert method.last_pairwise_trace is None
