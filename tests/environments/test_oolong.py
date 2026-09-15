"""Tests for the OOLONG environment: extraction tolerance, gold normalization,
scoring (all three OOLONG-real paths), the Verifier's cause mapping, the
SubVerifier's grounding (including the no-decomposition-possible case), and both
loaders over a monkeypatched dataset stream.

Everything runs offline: the module's only network seams, ``iter_synth_rows``
and ``iter_real_rows``, are monkeypatched with synthetic rows shaped like
``oolongbench/oolong-synth`` / ``oolongbench/oolong-real``.
"""

from typing import Any

import pytest

import shrlm.environments.oolong as oolong
from shrlm.environments.oolong import (
    ANSWER_FORMAT_CONTRACT,
    OolongSubVerifier,
    OolongVerifier,
    build_label_map,
    continuous_score,
    extract_oolong_answer,
    infer_real_answer_kind,
    load_oolong_real,
    load_oolong_synth,
    normalize_gold,
    parse_sub_task,
    score_oolong,
)
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind

# ---------------------------------------------------------------------------
# Synthetic upstream rows
# ---------------------------------------------------------------------------

WINDOW_LINES = [
    "Date: Jan 01, 2023 || User: 11 || Instance: claim your free prize now || Label: spam",
    "Date: Jan 02, 2023 || User: 22 || Instance: see you at lunch tomorrow || Label: ham",
    "Date: Jan 03, 2023 || User: 11 || Instance: urgent account verification || Label: spam",
    "Date: Jan 04, 2023 || User: 33 || Instance: the meeting notes are attached || Label: ham",
]
LABELS_TEXT = "\n".join(WINDOW_LINES)
CONTEXT_TEXT = "\n".join(line.split(" || Label: ")[0] for line in WINDOW_LINES)


def synth_row(
    *,
    context_len: int = 4096,
    task_group: str = "counting",
    task: str = "TASK_TYPE.NUMERIC_ONE_CLASS",
    answer_type: str = "ANSWER_TYPE.NUMERIC",
    answer: str = "[2]",
    question: str = "How many data points should be classified as label 'spam'?",
    dataset: str = "spam",
    window_id: int = 10000,
) -> dict[str, Any]:
    return {
        "context_len": context_len,
        "dataset": dataset,
        "context_window_text": CONTEXT_TEXT,
        "context_window_text_with_labels": LABELS_TEXT,
        "question": question,
        "task_group": task_group,
        "task": task,
        "answer": answer,
        "answer_type": answer_type,
        "context_window_id": window_id,
    }


def real_row(
    *,
    question_type: str = "singledoc_rolls",
    question: str = "Total number of rolls in this episode?",
    answer: str = "84",
    episodes: tuple[int, ...] = (1,),
    window_id: str = "w-abc",
) -> dict[str, Any]:
    return {
        "id": f"id-{window_id}-{question_type}",
        "context_window_id": window_id,
        "context_window_text": "[START OF EPISODE]\n...transcript...\n[END OF EPISODE]",
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "episodes": list(episodes),
        "campaign": "campaign1",
    }


def stub_synth(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_iter(split: str, revision: str | None, deadline_seconds: float = 60.0):
        calls["split"] = split
        calls["revision"] = revision
        return iter(rows)

    monkeypatch.setattr(oolong, "iter_synth_rows", fake_iter)
    return calls


def stub_real(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_iter(
        config_name: str, split: str, revision: str | None, deadline_seconds: float = 60.0
    ):
        calls["config_name"] = config_name
        calls["split"] = split
        calls["revision"] = revision
        return iter(rows)

    monkeypatch.setattr(oolong, "iter_real_rows", fake_iter)
    return calls


def make_node(
    prompt: str | dict[str, Any],
    response: str,
    kind: NodeKind = NodeKind.RLM_CHILD,
    error_kind: str | None = None,
) -> CallNode:
    return CallNode(
        node_id="r/i0/b0/c0",
        parent_id="r",
        kind=kind,
        depth=1,
        model="mock",
        prompt=prompt,
        response=response,
        prompt_chars=len(str(prompt)),
        response_chars=len(response),
        execution_time=0.1,
        error_kind=error_kind,
    )


# ---------------------------------------------------------------------------
# Extraction and tolerance
# ---------------------------------------------------------------------------


class TestExtractOolongAnswer:
    def test_marker_line_wins_over_trailing_prose(self):
        parsed = extract_oolong_answer("FINAL: spam\nThanks for asking!", "label")
        assert parsed is not None and parsed.value == "spam"

    def test_last_nonempty_line_when_no_marker(self):
        assert extract_oolong_answer("reasoning...\nham", "label").value == "ham"

    def test_numeric_reads_last_number(self):
        assert extract_oolong_answer("maybe 3 or\nAnswer: 5", "numeric").value == 5

    def test_numeric_no_number_is_none(self):
        assert extract_oolong_answer("I could not compute this", "numeric") is None

    def test_comparison_phrases_collapse(self):
        assert extract_oolong_answer("Answer: spam is less common than ham", "comparison").value == (
            "less common"
        )
        assert extract_oolong_answer("more common", "comparison").value == "more common"
        assert extract_oolong_answer("the same frequency as", "comparison").value == (
            "same frequency"
        )

    def test_date_normalizes_multiple_formats(self):
        assert extract_oolong_answer("Date: 01/06/2023", "date").value == "2023-01-06"
        assert extract_oolong_answer("January 6, 2023", "date").value == "2023-01-06"

    def test_unparseable_date_is_none(self):
        assert extract_oolong_answer("Date: sometime in spring", "date") is None

    def test_explicit_empty_marker_flags_empty(self):
        parsed = extract_oolong_answer("FINAL: none", "label")
        assert parsed is not None and parsed.empty is True


class TestExtractionTolerance:
    """Serialization is not reasoning: wrapping quotes/brackets/asterisks, the
    ``Answer:`` prefix, a trailing ``%`` on a number, and case are all
    normalized away (POST_MORTEM Finding 1)."""

    def test_wrapping_quotes_and_brackets_stripped(self):
        assert extract_oolong_answer("FINAL: ['spam']", "label").value == "spam"
        assert extract_oolong_answer('Label: "ham"', "label").value == "ham"
        assert extract_oolong_answer("**spam**", "label").value == "spam"

    def test_percent_sign_ignored_on_numeric(self):
        assert extract_oolong_answer("Answer: 42%", "numeric").value == 42

    def test_thousands_separators_tolerated(self):
        assert extract_oolong_answer("FINAL: 1,234", "numeric").value == 1234

    def test_user_id_case_and_brackets(self):
        assert extract_oolong_answer("User: [76063]", "user").value == "76063"

    def test_list_read_as_lowercased_set(self):
        parsed = extract_oolong_answer("FINAL: Attack, Damage , attack", "list")
        assert parsed is not None and parsed.value == ["attack", "damage"]

    def test_list_empty_marker(self):
        parsed = extract_oolong_answer("none", "list")
        assert parsed is not None and parsed.value == [] and parsed.empty is True


class TestNormalizeGold:
    def test_stringified_one_element_list(self):
        assert normalize_gold("['spam']", "label") == "spam"
        assert normalize_gold("[4]", "numeric") == 4

    def test_datetime_date_repr(self):
        assert normalize_gold("[datetime.date(2023, 1, 6)]", "date") == "2023-01-06"

    def test_comparison_gold_normalized(self):
        assert normalize_gold("['less common than']", "comparison") == "less common"

    def test_bare_real_values(self):
        assert normalize_gold("84", "numeric") == 84
        assert normalize_gold("Attack", "string") == "attack"
        assert normalize_gold("Acrobatics, Constitution Save", "list") == [
            "acrobatics",
            "constitution save",
        ]


class TestScoreOolong:
    def test_numeric_partial_credit_curve(self):
        assert score_oolong(extract_oolong_answer("4", "numeric"), 4, "numeric") == {
            "score": 1.0,
            "exact": True,
        }
        near = score_oolong(extract_oolong_answer("6", "numeric"), 4, "numeric")
        assert near["exact"] is False and abs(near["score"] - 0.75**2) < 1e-9

    def test_numeric_non_number_scores_zero(self):
        assert score_oolong(
            oolong.ParsedAnswer(kind="numeric", value="lots"), 4, "numeric"
        ) == {"score": 0.0, "exact": False}

    def test_label_exact_match(self):
        assert score_oolong(extract_oolong_answer("spam", "label"), "spam", "label")["exact"]
        assert not score_oolong(extract_oolong_answer("ham", "label"), "spam", "label")["exact"]

    def test_list_set_overlap_and_exact(self):
        got = extract_oolong_answer("attack, damage", "list")
        assert score_oolong(got, ["attack", "damage"], "list") == {"score": 1.0, "exact": True}
        partial = score_oolong(got, ["attack", "damage", "acrobatics"], "list")
        assert partial["exact"] is False and abs(partial["score"] - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TestOolongVerifier:
    def synth(self, **kw: Any) -> dict[str, Any]:
        return {"answer_kind": kw.get("answer_kind", "numeric"), "answer_raw": kw.get("answer_raw", "[2]")}

    def test_exact_numeric_passes(self):
        verdict = OolongVerifier()(self.synth(), "reasoning\nFINAL: 2")
        assert verdict.passed is True and verdict.cause is None

    def test_wrong_format_when_no_number(self):
        verdict = OolongVerifier()(self.synth(), "I was unable to determine a count")
        assert verdict.passed is False and verdict.cause is VerifierCause.WRONG_FORMAT

    def test_no_answer_when_explicit_empty_against_nonempty_gold(self):
        verdict = OolongVerifier()(self.synth(answer_kind="label", answer_raw="['spam']"), "FINAL: none")
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_wrong_value_for_scalar_miss(self):
        verdict = OolongVerifier()(self.synth(), "FINAL: 9")
        assert verdict.cause is VerifierCause.WRONG_VALUE
        assert "score=" in verdict.detail

    def test_list_cause_mapping(self):
        inst = {"answer_kind": "list", "answer_raw": "attack, damage"}
        assert OolongVerifier("real")(inst, "FINAL: attack").cause is VerifierCause.INCOMPLETE
        assert (
            OolongVerifier("real")(inst, "FINAL: attack, damage, extra").cause
            is VerifierCause.SPURIOUS
        )
        assert (
            OolongVerifier("real")(inst, "FINAL: attack, wrong").cause
            is VerifierCause.MIXED_SET_ERROR
        )

    def test_config_names_the_grading_facts(self):
        config = OolongVerifier("synth").config()
        assert config["environment"] == "oolong_synth"
        assert config["pass_score_threshold"] == 1.0
        assert "numeric-digits-only" in config["extraction_rule"]
        assert "numeric" in config["answer_kinds"]
        assert OolongVerifier("real").config()["environment"] == "oolong_real"

    def test_continuous_score_reports_partial_credit(self):
        inst = {"answer_kind": "numeric", "answer_raw": "10"}
        assert continuous_score(inst, "FINAL: 10") == 1.0
        assert abs(continuous_score(inst, "FINAL: 12") - 0.75**2) < 1e-9
        assert continuous_score(inst, "no answer here") == 0.0


# ---------------------------------------------------------------------------
# SubVerifier
# ---------------------------------------------------------------------------

CLASSIFY_CHILD = (
    "Classify each of the following lines as spam or ham, one label per line:\n"
    "claim your free prize now\n"
    "see you at lunch tomorrow\n"
    "urgent account verification\n"
    "Return the labels as a comma-separated list."
)
COUNT_CHILD = (
    "How many of these lines are labelled spam?\n"
    "claim your free prize now\n"
    "see you at lunch tomorrow\n"
    "urgent account verification"
)


class TestBuildLabelMap:
    def test_parses_labelled_lines(self):
        mapping = build_label_map(LABELS_TEXT)
        assert mapping["claim your free prize now"] == "spam"
        assert mapping["see you at lunch tomorrow"] == "ham"

    def test_unrecognized_shape_is_empty(self):
        assert build_label_map("just some text with no label suffix\nanother line") == {}


class TestParseSubTask:
    def test_classify_recognized(self):
        mapping = build_label_map(LABELS_TEXT)
        kind, lines, label = parse_sub_task(CLASSIFY_CHILD, mapping)
        assert kind == "classify" and len(lines) == 3 and label is None

    def test_count_recognized_with_label(self):
        mapping = build_label_map(LABELS_TEXT)
        kind, lines, label = parse_sub_task(COUNT_CHILD, mapping)
        assert kind == "count" and label == "spam" and len(lines) == 3

    def test_non_slice_task_is_none(self):
        # No labelled lines from the map -> not groundable.
        assert parse_sub_task("Summarize the sentiment of the whole document.", build_label_map(LABELS_TEXT)) is None

    def test_fewer_than_two_matched_lines_is_none(self):
        mapping = build_label_map(LABELS_TEXT)
        assert parse_sub_task("Classify:\nclaim your free prize now", mapping) is None


class TestOolongSubVerifier:
    inst = {"labels_text": LABELS_TEXT}

    def test_correct_classify_child_passes(self):
        node = make_node(CLASSIFY_CHILD, "Final labels: [spam, ham, spam]")
        assert OolongSubVerifier()(self.inst, node) is True

    def test_wrong_classify_child_fails(self):
        node = make_node(CLASSIFY_CHILD, "[spam, spam, spam]")
        assert OolongSubVerifier()(self.inst, node) is False

    def test_correct_count_child_passes(self):
        node = make_node(COUNT_CHILD, "FINAL: 2")
        assert OolongSubVerifier()(self.inst, node) is True

    def test_wrong_count_child_fails(self):
        node = make_node(COUNT_CHILD, "FINAL: 3")
        assert OolongSubVerifier()(self.inst, node) is False

    def test_no_decomposition_possible_is_uncheckable(self):
        # A child prompt that is not a slice classify/count task: uncheckable,
        # never False -- so derive_failing_level stays NO_RECURSION/UNDETERMINED
        # rather than flipping to CHILD on a formatting quirk.
        node = make_node("Aggregate the label distribution and report the mode.", "FINAL: spam")
        assert OolongSubVerifier()(self.inst, node) is None

    def test_source_dataset_without_labels_is_uncheckable(self):
        node = make_node(CLASSIFY_CHILD, "[spam, ham, spam]")
        assert OolongSubVerifier()({"labels_text": "no labels here"}, node) is None

    def test_errored_node_is_uncheckable(self):
        node = make_node(CLASSIFY_CHILD, "boom", kind=NodeKind.ERRORED, error_kind="lm_error")
        assert OolongSubVerifier()(self.inst, node) is None

    def test_dict_prompt_is_uncheckable(self):
        node = make_node({"role": "user", "content": CLASSIFY_CHILD}, "[spam, ham, spam]")
        assert OolongSubVerifier()(self.inst, node) is None

    def test_child_without_parseable_labels_is_uncheckable(self):
        node = make_node(CLASSIFY_CHILD, "I looked at the lines and formed an opinion.")
        assert OolongSubVerifier()(self.inst, node) is None

    def test_pathological_prompt_never_raises(self):
        node = make_node("\x00\xff" * 10000, "FINAL: 2")
        assert OolongSubVerifier()(self.inst, node) is None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestLoadOolongSynth:
    def _pool(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ctx in (1024, 4096, 16384):
            for group, task, atype, ans in (
                ("counting", "TASK_TYPE.NUMERIC_ONE_CLASS", "ANSWER_TYPE.NUMERIC", "[2]"),
                ("user", "TASK_TYPE.MOST_FREQ", "ANSWER_TYPE.USER", "[11]"),
                ("timeline", "TASK_TYPE.RELATIVE_FREQ", "ANSWER_TYPE.COMPARISON", "['less common than']"),
            ):
                for k in range(3):
                    rows.append(
                        synth_row(
                            context_len=ctx,
                            task_group=group,
                            task=task,
                            answer_type=atype,
                            answer=ans,
                            question=f"q {ctx} {group} {k}",
                            window_id=ctx * 10 + k,
                        )
                    )
        return rows

    def test_length_stratified_and_group_balanced(self, monkeypatch):
        stub_synth(monkeypatch, self._pool())
        instances = load_oolong_synth(
            context_lengths=(1024, 4096, 16384),
            task_groups=("counting", "user", "timeline"),
            subsets=(),
            limit=9,
            seed=0,
        )
        assert len(instances) == 9
        by_len: dict[int, int] = {}
        for inst in instances:
            by_len[inst["context_len"]] = by_len.get(inst["context_len"], 0) + 1
        # Balanced share per length bucket.
        assert set(by_len) == {1024, 4096, 16384}
        assert max(by_len.values()) - min(by_len.values()) <= 1

    def test_ids_are_content_derived_and_seed_independent(self, monkeypatch):
        stub_synth(monkeypatch, self._pool())
        first = load_oolong_synth((1024, 4096, 16384), ("counting", "user", "timeline"), (), 6, seed=0)
        stub_synth(monkeypatch, self._pool())
        second = load_oolong_synth((1024, 4096, 16384), ("counting", "user", "timeline"), (), 6, seed=99)
        overlap = set(i["id"] for i in first) & set(i["id"] for i in second)
        assert overlap
        for inst in first:
            assert FILESYSTEM_SAFE_ID_PATTERN.fullmatch(inst["id"])
            assert inst["prompt"].endswith(ANSWER_FORMAT_CONTRACT)
            assert "|| Label:" in inst["labels_text"]
            assert "|| Label:" not in inst["prompt"]

    def test_missing_length_raises_naming_inventory(self, monkeypatch):
        stub_synth(monkeypatch, [synth_row(context_len=1024)])
        with pytest.raises(ValueError, match="insufficient OOLONG rows"):
            load_oolong_synth((1024, 65536), ("counting",), (), limit=4, seed=0)

    def test_unknown_context_length_rejected(self, monkeypatch):
        stub_synth(monkeypatch, [synth_row()])
        with pytest.raises(ValueError, match="power-of-two"):
            load_oolong_synth((3000,), (), (), limit=1, seed=0)

    def test_split_and_revision_are_plumbed(self, monkeypatch):
        calls = stub_synth(monkeypatch, self._pool())
        load_oolong_synth(
            (1024,), ("counting",), (), limit=1, seed=0, split="validation", revision="pinned"
        )
        assert calls == {"split": "validation", "revision": "pinned"}


class TestLoadOolongReal:
    def _pool(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for n_ep in (1, 2, 3):
            for qtype, q, ans in (
                ("singledoc_rolls", "Total number of rolls in this episode?", "84"),
                ("singledoc_spells", "How many spells were cast during this episode?", "15"),
                (
                    "singledoc_rolls",
                    "What is the most common roll type? Return a comma separated list.",
                    "Attack",
                ),
            ):
                for k in range(2):
                    rows.append(
                        real_row(
                            question_type=qtype,
                            question=q,
                            answer=ans,
                            episodes=tuple(range(1, n_ep + 1)),
                            window_id=f"w{n_ep}-{qtype}-{k}",
                        )
                    )
        return rows

    def test_infer_answer_kind(self):
        assert infer_real_answer_kind("Return a comma separated list.", "Attack") == "list"
        assert infer_real_answer_kind("Total number of rolls?", "84") == "numeric"
        assert infer_real_answer_kind("Most common roll type?", "Attack") == "string"

    def test_loads_stratified_by_episode_count(self, monkeypatch):
        calls = stub_real(monkeypatch, self._pool())
        instances = load_oolong_real(
            episode_counts=(1, 2, 3),
            question_types=(),
            limit=6,
            seed=0,
            split="test",
            revision="pin",
            config_name="dnd",
        )
        assert len(instances) == 6
        assert calls == {"config_name": "dnd", "split": "test", "revision": "pin"}
        for inst in instances:
            assert FILESYSTEM_SAFE_ID_PATTERN.fullmatch(inst["id"])
            assert inst["answer_kind"] in ("numeric", "string", "list")
            assert inst["prompt"].endswith(ANSWER_FORMAT_CONTRACT)

    def test_insufficient_rows_raises(self, monkeypatch):
        stub_real(monkeypatch, [real_row()])
        with pytest.raises(ValueError, match="insufficient OOLONG-real rows"):
            load_oolong_real((1,), (), limit=5, seed=0)
