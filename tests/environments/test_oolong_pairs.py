"""Tests for the OOLONG-Pairs environment: coverage, loader, Verifier, driver fit.

Everything runs offline: the module's only network seam,
``iter_dataset_rows``, is monkeypatched with synthetic upstream rows shaped
like ``oolongbench/oolong-synth`` (``dataset``, ``context_len``,
``context_window_id``, ``context_window_text_with_labels``).
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest

import rlm.core.rlm as rlm_module
import shrlm.environments.oolong_pairs as oolong_pairs
from shrlm.environments.oolong_pairs import (
    OOLONG_SUBSET,
    OolongPairsVerifier,
    build_instance,
    collect_windows,
    compute_gold_pairs,
    extract_answer_pairs,
    load_oolong_pairs,
    load_oolong_pairs_from_config,
    score,
    verify_window_coverage,
)
from shrlm.experiment.config import OolongPairsConfig
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.driver import RoundConfig, run_round
from shrlm.optimization.driver import _validate_config as validate_round_config
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Synthetic upstream rows
# ---------------------------------------------------------------------------


def entry(user_id: int, date_str: str, question: str, label: str) -> str:
    return f"Date: {date_str} || User: {user_id} || Instance: {question} || Label: {label}"


# Three users, one entry each: 101 numeric value, 202 location, 303 abbreviation.
# Task 1 (numeric value or location) -> [(101, 202)].
# Task 6 (location or abbreviation) -> [(202, 303)].
WINDOW_LINES = [
    entry(101, "Feb 02, 2023", "How many moons does Mars have ?", "numeric value"),
    entry(202, "Mar 03, 2023", "Where is the Eiffel Tower located ?", "location"),
    entry(303, "Apr 04, 2023", "What does TNT stand for ?", "abbreviation"),
]


def make_row(
    window_id: Any,
    context_len: int,
    lines: list[str] | None = None,
    dataset: str = OOLONG_SUBSET,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "context_len": context_len,
        "context_window_id": window_id,
        "context_window_text_with_labels": "\n".join(lines if lines is not None else WINDOW_LINES),
    }


def stub_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace the network seam with an in-memory stream; record its arguments."""
    calls: dict[str, Any] = {}

    def fake_iter(split: str, revision: str | None):
        calls["split"] = split
        calls["revision"] = revision
        return iter(rows)

    monkeypatch.setattr(oolong_pairs, "iter_dataset_rows", fake_iter)
    return calls


def sized_window_lines(
    context_len: int, tokens_per_entry: int = 16, label: str = "entity"
) -> list[str]:
    """Synthetic window text sized like a real ``context_len``-TOKEN window.

    Each labeled line is padded to ~4 characters per token, mirroring the
    real data's rough chars-per-token ratio, so ``context_len // tokens_per_
    entry`` entries give a labeled text of ~4 * context_len characters.

    ``label`` defaults to "entity", which keeps task 1's gold empty (task 1
    is eligible on numeric value or location) -- previously the only way this
    helper avoided a window of tens of thousands of task-eligible users
    exploding into millions of pairs. That explosion is now caught by
    ``MAX_GOLD_PAIRS_PER_WINDOW`` (see ``TestGoldPairBudget``), so callers
    that want to exercise that guard can pass an eligible label instead.
    """
    target_chars = tokens_per_entry * 4
    lines = []
    for i in range(context_len // tokens_per_entry):
        question = f"What is creature number {i:06d} ?"
        line = entry(100_000 + i, "Feb 02, 2023", question, label)
        lines.append(line + " " * max(0, target_chars - len(line) - 1))
    return lines


# ---------------------------------------------------------------------------
# Gold-pair logic
# ---------------------------------------------------------------------------


class TestComputeGoldPairs:
    def test_symmetric_task(self):
        from datetime import date

        by_user = {
            1: [("numeric value", date(2023, 2, 2))],
            2: [("location", date(2023, 3, 3))],
            3: [("entity", date(2023, 4, 4))],
        }
        assert compute_gold_pairs(1, by_user) == [(1, 2)]

    def test_asymmetric_task(self):
        from datetime import date

        d = date(2023, 2, 2)
        by_user = {
            7: [("entity", d), ("abbreviation", d)],  # role 11a
            3: [("entity", d)],  # role 11b: exactly one entity
            9: [("location", d)],  # neither role
        }
        assert compute_gold_pairs(11, by_user) == [(3, 7)]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="task_id"):
            compute_gold_pairs(21, {})


# ---------------------------------------------------------------------------
# Finding 9: unbounded quadratic gold-pair construction
# ---------------------------------------------------------------------------


class TestGoldPairBudget:
    def test_symmetric_guard_fires_before_allocating(self):
        from datetime import date

        # 3 users all eligible for task 1 (numeric value or location) -> 3
        # pairs, which exceeds a max_gold_pairs of 2.
        by_user = {u: [("numeric value", date(2023, 1, 1))] for u in range(3)}
        with pytest.raises(ValueError, match=r"window 'w-huge' task 1\b.*exceeding"):
            compute_gold_pairs(1, by_user, window_id="w-huge", max_gold_pairs=2)

    def test_asymmetric_guard_fires_before_allocating(self):
        from datetime import date

        d = date(2023, 1, 1)
        # role_11a (entity and abbreviation): users 1, 2.
        # role_11b (exactly one entity instance): users 3, 4.
        # Upper bound on pairs is 2 * 2 = 4, which exceeds max_gold_pairs=3.
        by_user = {
            1: [("entity", d), ("abbreviation", d)],
            2: [("entity", d), ("abbreviation", d)],
            3: [("entity", d)],
            4: [("entity", d)],
        }
        with pytest.raises(ValueError, match=r"window 'w-asym' task 11\b.*exceeding"):
            compute_gold_pairs(11, by_user, window_id="w-asym", max_gold_pairs=3)

    def test_normal_window_is_unaffected(self):
        from datetime import date

        by_user = {
            1: [("numeric value", date(2023, 1, 1))],
            2: [("location", date(2023, 1, 1))],
            3: [("entity", date(2023, 1, 1))],
        }
        # Well under the default MAX_GOLD_PAIRS_PER_WINDOW ceiling.
        assert compute_gold_pairs(1, by_user, window_id="w-normal") == [(1, 2)]

    def test_broad_eligibility_long_window_fails_fast_not_oom(self):
        """Regression for finding 9: a schema-valid 262144-token window where
        every user is eligible under an OR-only predicate must fail fast with
        a named error, not attempt to materialize tens of millions of pairs.
        """
        long_len = 262144
        lines = sized_window_lines(long_len, label="numeric value")
        window = make_row("w-explosive", long_len, lines)
        with pytest.raises(ValueError, match=r"window 'w-explosive' task 1\b.*exceeding"):
            build_instance(window, task_id=1, sample_seed=0, sample_index=0)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


class TestExtractAnswerPairs:
    def test_no_pairs_and_no_marker_is_none(self):
        assert extract_answer_pairs("I could not work out any of the labels.") is None

    def test_explicit_empty_marker_is_empty_list(self):
        assert extract_answer_pairs("No valid pairs found.") == []

    def test_pairs_parsed_in_order_preserving_pair_order(self):
        parsed = extract_answer_pairs("(202, 101)\n(101, 303)")
        assert parsed == [(202, 101), (101, 303)]

    def test_pairs_embedded_in_prose(self):
        parsed = extract_answer_pairs("The valid pairs are (101, 202) and (202, 303).")
        assert parsed == [(101, 202), (202, 303)]


# ---------------------------------------------------------------------------
# Finding 7: unbounded upstream stream
# ---------------------------------------------------------------------------


class TestBoundedIter:
    def test_rows_pass_through_unchanged(self):
        rows = iter([{"a": 1}, {"a": 2}, {"a": 3}])
        assert list(oolong_pairs._bounded_iter(rows, deadline_seconds=5.0)) == [
            {"a": 1},
            {"a": 2},
            {"a": 3},
        ]

    def test_stall_before_first_row_raises_timeout(self):
        def slow_rows():
            time.sleep(0.5)
            yield {"a": 1}

        with pytest.raises(TimeoutError, match="stalled"):
            list(oolong_pairs._bounded_iter(slow_rows(), deadline_seconds=0.05))

    def test_stall_between_rows_raises_timeout(self):
        def slow_rows():
            yield {"a": 1}
            time.sleep(0.5)
            yield {"a": 2}

        collected = []
        with pytest.raises(TimeoutError, match="stalled"):
            for row in oolong_pairs._bounded_iter(slow_rows(), deadline_seconds=0.05):
                collected.append(row)
        assert collected == [{"a": 1}]

    def test_upstream_exception_propagates_unchanged(self):
        def boom():
            yield {"a": 1}
            raise RuntimeError("upstream broke")

        with pytest.raises(RuntimeError, match="upstream broke"):
            list(oolong_pairs._bounded_iter(boom(), deadline_seconds=5.0))


# ---------------------------------------------------------------------------
# Window collection and coverage
# ---------------------------------------------------------------------------


class TestWindowCoverage:
    def test_dedupes_repeated_windows_and_skips_other_subsets(self, monkeypatch):
        stub_rows(
            monkeypatch,
            [
                make_row("w1", 1024),
                make_row("w1", 1024),  # same window, different upstream task
                make_row("w9", 1024, dataset="banking77"),
                make_row("w2", 1024),
            ],
        )
        windows = collect_windows((1024,), 2, "validation", 100, None)
        assert [w["context_window_id"] for w in windows[1024]] == ["w1", "w2"]

    def test_max_scan_bounds_the_scan(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 1024), make_row("w2", 1024)])
        with pytest.raises(ValueError, match="insufficient distinct"):
            verify_window_coverage((1024,), 2, max_scan=1)

    def test_satisfiable_lengths_return_inventory(self, monkeypatch):
        stub_rows(
            monkeypatch,
            [make_row("w1", 1024), make_row("w2", 1024), make_row("w3", 2048)],
        )
        assert verify_window_coverage((1024, 2048), 1, max_scan=100) == {1024: 1, 2048: 1}

    def test_shortfall_raises_naming_available_inventory(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 8192)])
        with pytest.raises(ValueError) as excinfo:
            verify_window_coverage((8192, 262144), 3, max_scan=100)
        message = str(excinfo.value)
        assert "262144" in message
        assert "need 3" in message
        assert "{8192: 1, 262144: 0}" in message

    def test_unknown_length_raises_before_streaming(self, monkeypatch):
        def boom(split: str, revision: str | None):
            raise AssertionError("must not stream for an unmintable length")

        monkeypatch.setattr(oolong_pairs, "iter_dataset_rows", boom)
        with pytest.raises(ValueError, match="cannot mint new lengths"):
            verify_window_coverage((5000,), 1)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoadOolongPairs:
    def test_instances_pass_driver_validation(self, monkeypatch, tmp_path):
        stub_rows(monkeypatch, [make_row("w1", 1024), make_row("w2", 1024)])
        instances = load_oolong_pairs((1, 6), (1024,), n=4, seed=7)
        assert len(instances) == 4
        validate_round_config(
            RoundConfig(
                round_index=1,
                harness=H0,
                instances=instances,
                verifier=OolongPairsVerifier(),
                out_dir=tmp_path,
            )
        )
        for instance in instances:
            assert FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance["id"])
            assert instance["sample_seed"] == 7
            assert instance["gold_pairs"]
            # Byte-stable through the driver's instances.jsonl round trip.
            assert json.loads(json.dumps(instance, sort_keys=True)) == instance
        assert [i["sample_index"] for i in instances] == [0, 1, 2, 3]

    def test_gold_pairs_match_the_task_predicates(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        instances = load_oolong_pairs((1, 6), (1024,), n=2, seed=0)
        by_task = {i["task_id"]: i for i in instances}
        assert by_task[1]["gold_pairs"] == [[101, 202]]
        assert by_task[6]["gold_pairs"] == [[202, 303]]

    def test_same_seed_and_config_is_deterministic(self, monkeypatch):
        rows = [make_row("w1", 1024), make_row("w2", 1024)]
        stub_rows(monkeypatch, list(rows))
        first = load_oolong_pairs((1, 2, 6), (1024,), n=5, seed=3)
        stub_rows(monkeypatch, list(rows))
        second = load_oolong_pairs((1, 2, 6), (1024,), n=5, seed=3)
        assert first == second
        assert [i["id"] for i in first] == [i["id"] for i in second]

    def test_different_seed_changes_the_sample(self, monkeypatch):
        rows = [make_row(f"w{k}", 1024) for k in range(4)]
        stub_rows(monkeypatch, list(rows))
        first = load_oolong_pairs(tuple(range(1, 11)), (1024,), n=10, seed=0)
        stub_rows(monkeypatch, list(rows))
        second = load_oolong_pairs(tuple(range(1, 11)), (1024,), n=10, seed=1)
        assert [i["id"] for i in first] != [i["id"] for i in second]

    def test_ids_are_content_derived_not_seed_derived(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        first = load_oolong_pairs((1,), (1024,), n=1, seed=0)
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        second = load_oolong_pairs((1,), (1024,), n=1, seed=99)
        assert first[0]["id"] == second[0]["id"]

    def test_insufficient_windows_raises_naming_inventory(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        # n=4 over 2 tasks needs ceil(4/2)=2 windows; only 1 ships.
        with pytest.raises(ValueError, match="insufficient distinct"):
            load_oolong_pairs((1, 6), (1024,), n=4, seed=0)

    def test_bad_arguments_fail_fast(self, monkeypatch):
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        with pytest.raises(ValueError, match="task_id"):
            load_oolong_pairs((1, 42), (1024,), n=1, seed=0)
        with pytest.raises(ValueError, match="n must be"):
            load_oolong_pairs((1,), (1024,), n=0, seed=0)
        with pytest.raises(ValueError, match="task_ids"):
            load_oolong_pairs((), (1024,), n=1, seed=0)

    def test_unsafe_window_id_raises(self):
        with pytest.raises(ValueError, match="filesystem-safe"):
            build_instance(make_row("w/../1", 1024), 1, sample_seed=0, sample_index=0)

    def test_from_config_forwards_pin_and_lengths(self, monkeypatch):
        config = OolongPairsConfig(
            dataset_repo="oolongbench/oolong-synth",
            subset="trec_coarse",
            dataset_revision="pin-abc123",
            task_ids=(1, 6),
            context_length_short=1024,
            context_length_long=2048,
            max_scan=100,
            n_short=4,
            n_long=2,
        )
        calls = stub_rows(monkeypatch, [make_row("w1", 1024), make_row("w2", 2048)])
        instances = load_oolong_pairs_from_config(config, n=2, seed=0)
        assert calls["revision"] == "pin-abc123"
        assert calls["split"] == "validation"
        assert len(instances) == 2

        calls = stub_rows(monkeypatch, [make_row("w1", 1024)])
        short_only = load_oolong_pairs_from_config(config, n=2, seed=0, context_lengths=(1024,))
        assert {i["context_len"] for i in short_only} == {1024}


class TestPromptMagnitude:
    def test_prompt_chars_track_the_requested_token_length(self, monkeypatch):
        short_len, long_len = 8192, 262144
        stub_rows(monkeypatch, [make_row("ws", short_len, sized_window_lines(short_len))])
        short = load_oolong_pairs((1,), (short_len,), n=1, seed=0)
        stub_rows(monkeypatch, [make_row("wl", long_len, sized_window_lines(long_len))])
        long = load_oolong_pairs((1,), (long_len,), n=1, seed=0)

        # Upstream context_len counts tokens; at ~4 chars/token the rebuilt
        # prompt lands within a small factor of that, and the long/short char
        # ratio matches the 32x token ratio.
        assert 2 * short_len <= len(short[0]["prompt"]) <= 8 * short_len
        assert 2 * long_len <= len(long[0]["prompt"]) <= 8 * long_len
        ratio = len(long[0]["prompt"]) / len(short[0]["prompt"])
        assert 16 <= ratio <= 64


# ---------------------------------------------------------------------------
# Finding 20: degenerate score on the both-empty case
# ---------------------------------------------------------------------------


class TestScore:
    def test_both_empty_scores_perfect(self):
        assert score([], []) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_normal_window_unaffected(self):
        metrics = score([(1, 2)], [(1, 2), (2, 3)])
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
        assert 0.0 < metrics["f1"] < 1.0

    def test_empty_prediction_against_nonempty_gold_scores_zero(self):
        metrics = score([], [(1, 2)])
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_spurious_pairs_against_empty_gold_score_zero(self):
        metrics = score([(1, 2)], [])
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def make_instance(gold_pairs: list[list[int]]) -> dict[str, Any]:
    return {
        "id": "oolong-t01-w1-deadbeef00000000",
        "question": "task text",
        "prompt": "prompt text",
        "gold_pairs": gold_pairs,
        "task_id": 1,
        "sample_seed": 0,
        "sample_index": 0,
    }


class TestOolongPairsVerifier:
    def setup_method(self):
        self.verifier = OolongPairsVerifier()
        self.instance = make_instance([[101, 202], [202, 303]])

    def test_exact_pair_set_passes(self):
        verdict = self.verifier(self.instance, "(101, 202)\n(202, 303)")
        assert verdict.passed and verdict.cause is None
        assert verdict.gold == "[(101, 202), (202, 303)]"
        assert verdict.produced == "[(101, 202), (202, 303)]"
        assert "f1=1.000" in verdict.detail

    def test_gold_survives_json_round_trip(self):
        instance = json.loads(json.dumps(self.instance))
        verdict = self.verifier(instance, "(202, 303)\n(101, 202)")
        assert verdict.passed

    def test_unparseable_output_is_wrong_format_not_a_crash(self):
        verdict = self.verifier(self.instance, "I gave up on labeling the entries.")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.WRONG_FORMAT

    def test_explicit_empty_answer_is_no_answer(self):
        verdict = self.verifier(self.instance, "No valid pairs found.")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_missing_only_is_incomplete(self):
        verdict = self.verifier(self.instance, "(101, 202)")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.INCOMPLETE

    def test_extra_only_is_spurious(self):
        verdict = self.verifier(self.instance, "(101, 202)\n(202, 303)\n(101, 303)")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.SPURIOUS

    def test_wrong_pair_is_mixed_set_error(self):
        verdict = self.verifier(self.instance, "(101, 202)\n(200, 300)")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.MIXED_SET_ERROR

    def test_out_of_order_pair_does_not_score_as_correct(self):
        verdict = self.verifier(self.instance, "(202, 101)\n(202, 303)")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.MIXED_SET_ERROR

    def test_empty_gold_and_explicit_empty_answer_passes(self):
        verdict = self.verifier(make_instance([]), "No valid pairs found.")
        assert verdict.passed
        assert verdict.gold == "[]"
        # Finding 20 regression: a correct empty answer must not be scored as
        # a miss in the reported detail metrics either.
        assert "f1=1.000" in verdict.detail

    def test_config_names_the_environment(self):
        config = self.verifier.config()
        assert config["environment"] == "oolong_pairs"
        assert config["pass_f1_threshold"] == 1.0
        assert set(config) == {
            "environment",
            "pass_f1_threshold",
            "extraction_rule",
            "gold_ordering",
        }


# ---------------------------------------------------------------------------
# Integration: run_round over OOLONG-Pairs instances with a MockLM
# ---------------------------------------------------------------------------


def final(content: str) -> str:
    """A ```repl``` block that returns ``content`` from the answer variable."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


class TestRunRoundIntegration:
    def test_round_over_two_instances_persists_valid_traces(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        stub_rows(monkeypatch, [make_row("w1", 1024)])
        instances = load_oolong_pairs((1, 6), (1024,), n=2, seed=0)

        gold_answer = "\n".join(f"({a}, {b})" for a, b in instances[0]["gold_pairs"])
        script = [final(gold_answer), final("(1, 2)")]

        def fake_get_client(backend: str, backend_kwargs: dict[str, Any] | None) -> MockLM:
            return MockLM(response_fn=lambda prompt: script.pop(0))

        monkeypatch.setattr(rlm_module, "get_client", fake_get_client)

        config = RoundConfig(
            round_index=1,
            harness=H0,
            instances=instances,
            verifier=OolongPairsVerifier(),
            out_dir=tmp_path,
            backend="openai",
            backend_kwargs={"model_name": "oolong-test"},
            max_iterations=3,
        )
        entries = run_round(config)

        assert [entry["instance_id"] for entry in entries] == [i["id"] for i in instances]
        verdicts = [entry["verdict"] for entry in entries]
        assert verdicts[0]["passed"] is True
        assert verdicts[1]["passed"] is False
        assert verdicts[1]["cause"] in {cause.value for cause in VerifierCause}
        round_path = tmp_path / "round_01"
        manifest_lines = (round_path / "runs.jsonl").read_text().splitlines()
        assert len(manifest_lines) == 2
        for entry in entries:
            trace_path = round_path / entry["trace_path"]
            assert trace_path.exists()
            json.loads(trace_path.read_text())
