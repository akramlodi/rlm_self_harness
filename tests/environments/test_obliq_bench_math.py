"""Tests for the OBLIQ-Bench Math environment: parsing, NDCG, Verifier, loader.

Everything runs offline: the module's only network seam, ``_download``, is
monkeypatched to read small synthetic fixture files shaped like the real
``corpus.jsonl``/``queries.jsonl``/``qrels.tsv``/``per_query_excluded_ids.json``
(mirrors ``test_oolong_pairs.py``'s ``iter_dataset_rows`` stubbing).
"""

import json
from pathlib import Path

import pytest

import shrlm.environments.obliq_bench_math as obliq_bench_math
from shrlm.environments.obliq_bench_math import (
    ObliqBenchMathVerifier,
    build_prompt,
    extract_ranked_ids,
    load_obliq_bench_math,
    ndcg_at_10,
    recorded_ndcg,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict

# ---------------------------------------------------------------------------
# extract_ranked_ids
# ---------------------------------------------------------------------------


class TestExtractRankedIds:
    def test_no_marker_is_none(self):
        assert extract_ranked_ids("I could not find any matching problems.") is None

    def test_unparseable_json_is_none(self):
        assert extract_ranked_ids("RANKED: [not json") is None

    def test_non_list_json_is_none(self):
        assert extract_ranked_ids('RANKED: {"a": 1}') is None

    def test_non_string_items_is_none(self):
        assert extract_ranked_ids("RANKED: [1, 2, 3]") is None

    def test_explicit_empty_marker_is_empty_list(self):
        assert extract_ranked_ids("RANKED: []") == []

    def test_parses_ordered_list(self):
        assert extract_ranked_ids('RANKED: ["id_a", "id_b", "id_c"]') == ["id_a", "id_b", "id_c"]

    def test_parses_python_list_with_single_quotes(self):
        assert extract_ranked_ids("RANKED: ['id_a', 'id_b']") == ["id_a", "id_b"]

    def test_python_expression_is_not_executed(self):
        response = 'RANKED: [__import__("os").system("echo unsafe")]'
        assert extract_ranked_ids(response) is None

    def test_last_occurrence_wins_over_scratch_reasoning(self):
        response = (
            'The format example is RANKED: ["id_x"].\n'
            "After scanning the corpus, my real answer is:\n"
            'RANKED: ["id_a", "id_b"]'
        )
        assert extract_ranked_ids(response) == ["id_a", "id_b"]


# ---------------------------------------------------------------------------
# ndcg_at_10
# ---------------------------------------------------------------------------


class TestNdcgAt10:
    def test_perfect_ranking_is_one(self):
        gold = {"a", "b", "c"}
        assert ndcg_at_10(["a", "b", "c"], gold) == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        assert ndcg_at_10(["x", "y"], {"a", "b", "c"}) == 0.0

    def test_worse_ordering_scores_lower_than_perfect(self):
        gold = {"a", "b", "c"}
        assert ndcg_at_10(["z", "a", "b", "c"], gold) == pytest.approx(0.7328286204777911)

    def test_ideal_dcg_caps_at_rank_k_even_with_more_gold(self):
        gold = {f"g{i:02d}" for i in range(15)}
        top_ten = sorted(gold)[:10]
        assert ndcg_at_10(top_ten, gold) == pytest.approx(1.0)

    def test_entries_past_rank_ten_are_ignored(self):
        gold = {"a", "b", "c"}
        base = ndcg_at_10(["a", "b", "c"], gold)
        padded = ndcg_at_10(["a", "b", "c"] + [f"noise{i}" for i in range(20)], gold)
        assert padded == pytest.approx(base)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_bracketed_ids_and_query_and_format_contract(self):
        prompt = build_prompt(
            [("doc_a", "problem A text"), ("doc_b", "problem B text")], "QUERY TEXT"
        )
        assert "[doc_a]" in prompt
        assert "problem A text" in prompt
        assert "[doc_b]" in prompt
        assert "QUERY TEXT" in prompt
        assert "RANKED:" in prompt
        assert prompt.startswith("BENCHMARK RETRIEVAL TASK:")
        assert "not a request to invent" in prompt
        assert prompt.index("BENCHMARK RETRIEVAL TASK:") < prompt.index("QUERY TEXT")
        assert prompt.index("QUERY TEXT") < prompt.index("[doc_a]")
        assert prompt.rindex("FINAL OUTPUT REMINDER:") > prompt.index("[doc_b]")


# ---------------------------------------------------------------------------
# ObliqBenchMathVerifier
# ---------------------------------------------------------------------------


class TestObliqBenchMathVerifier:
    def setup_method(self):
        self.verifier = ObliqBenchMathVerifier()
        self.instance = {"gold_relevant_ids": ["a", "b", "c"]}

    def test_perfect_ranking_passes(self):
        verdict = self.verifier(self.instance, 'RANKED: ["a", "b", "c"]')
        assert verdict.passed and verdict.cause is None
        assert "ndcg@10=1.000" in verdict.detail

    def test_unparseable_output_is_wrong_format(self):
        verdict = self.verifier(self.instance, "I gave up.")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.WRONG_FORMAT

    def test_explicit_empty_answer_is_no_answer(self):
        verdict = self.verifier(self.instance, "RANKED: []")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_missing_only_is_incomplete(self):
        verdict = self.verifier(self.instance, 'RANKED: ["a"]')
        assert not verdict.passed
        assert verdict.cause is VerifierCause.INCOMPLETE

    def test_extra_only_is_spurious(self):
        # 6 irrelevant hits ranked ahead of all 3 gold ids pushes ndcg@10
        # below the 0.5 threshold with missing == empty (extra-only).
        response = 'RANKED: ["z1", "z2", "z3", "z4", "z5", "z6", "a", "b", "c"]'
        verdict = self.verifier(self.instance, response)
        assert not verdict.passed
        assert verdict.cause is VerifierCause.SPURIOUS

    def test_missing_and_extra_is_mixed_set_error(self):
        verdict = self.verifier(self.instance, 'RANKED: ["a", "z"]')
        assert not verdict.passed
        assert verdict.cause is VerifierCause.MIXED_SET_ERROR

    def test_reordering_within_gold_set_still_passes(self):
        # NDCG with one uniform relevance grade is order-invariant among
        # tied-relevance items: any permutation of exactly the gold set
        # scores ndcg@10 == 1.0, so this is never a WRONG_VALUE failure.
        verdict = self.verifier(self.instance, 'RANKED: ["c", "b", "a"]')
        assert verdict.passed
        assert "ndcg@10=1.000" in verdict.detail

    def test_duplicate_ids_are_deduplicated_before_scoring(self):
        verdict = self.verifier(self.instance, 'RANKED: ["a", "a", "b", "c"]')
        assert verdict.passed
        assert verdict.produced == '["a", "b", "c"]'

    def test_config_names_the_environment(self):
        config = self.verifier.config()
        assert config["environment"] == "obliq_bench_math"
        assert config["rank_k"] == 10
        assert set(config) == {
            "environment",
            "pass_ndcg_threshold",
            "rank_k",
            "extraction_rule",
        }


class TestRecordedNdcg:
    def test_reads_ndcg_out_of_detail(self):
        verdict = Verdict(
            passed=True, cause=None, gold="[]", produced="[]", detail="ndcg@10=0.732 hits=3/13"
        )
        assert recorded_ndcg(verdict) == pytest.approx(0.732)

    def test_wrong_format_cause_returns_none(self):
        verdict = Verdict(
            passed=False,
            cause=VerifierCause.WRONG_FORMAT,
            gold="[]",
            produced="",
            detail="no marker",
        )
        assert recorded_ndcg(verdict) is None


# ---------------------------------------------------------------------------
# load_obliq_bench_math (offline, via a monkeypatched _download)
# ---------------------------------------------------------------------------

CORPUS_ROWS = [
    {"_id": "doc_a", "text": "problem A"},
    {"_id": "doc_b", "text": "problem B"},
    {"_id": "doc_c", "text": "problem C"},
    {"_id": "doc_d", "text": "problem D"},
    {"_id": "doc_e", "text": "problem E"},
    {"_id": "doc_f", "text": "problem F"},
]
QUERY_ROWS = [
    {"_id": "q1", "text": "query one text"},
    {"_id": "q2", "text": "query two text"},
]
QRELS_ROWS = [("q1", "doc_a", 2), ("q1", "doc_b", 2), ("q2", "doc_d", 2)]
EXCLUDED_IDS = {"q1": ["doc_c"]}


def write_fixture_files(tmp_path: Path) -> dict[str, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n".join(json.dumps(row) for row in CORPUS_ROWS) + "\n")

    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text("\n".join(json.dumps(row) for row in QUERY_ROWS) + "\n")

    qrels_path = tmp_path / "qrels.tsv"
    lines = ["query-id\tcorpus-id\tscore"]
    lines += [f"{q}\t{c}\t{s}" for q, c, s in QRELS_ROWS]
    qrels_path.write_text("\n".join(lines) + "\n")

    excluded_path = tmp_path / "excluded.json"
    excluded_path.write_text(json.dumps(EXCLUDED_IDS))

    return {
        obliq_bench_math._CORPUS_FILE: corpus_path,
        obliq_bench_math._QUERIES_FILE: queries_path,
        obliq_bench_math._QRELS_FILE: qrels_path,
        obliq_bench_math._EXCLUDED_IDS_FILE: excluded_path,
    }


def stub_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_map = write_fixture_files(tmp_path)

    def fake_download(filename: str, revision: str | None) -> str:
        return str(file_map[filename])

    monkeypatch.setattr(obliq_bench_math, "_download", fake_download)


class TestLoadObliqBenchMath:
    def test_full_corpus_excludes_only_the_per_query_id(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math(query_ids=["q1"])
        assert len(instances) == 1
        instance = instances[0]
        assert instance["id"] == "q1"
        assert instance["gold_relevant_ids"] == ["doc_a", "doc_b"]
        assert instance["excluded_ids"] == ["doc_c"]
        assert instance["pool_size"] == 5  # 6 corpus docs minus the excluded one
        assert "[doc_c]" not in instance["prompt"]
        assert "[doc_a]" in instance["prompt"] and "[doc_b]" in instance["prompt"]

    def test_query_with_no_exclusions_keeps_full_corpus(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math(query_ids=["q2"])
        assert instances[0]["pool_size"] == len(CORPUS_ROWS)
        assert instances[0]["excluded_ids"] == []

    def test_query_ids_overrides_n_and_preserves_order(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math(n=1, query_ids=["q2", "q1"])
        assert [i["id"] for i in instances] == ["q2", "q1"]

    def test_n_none_selects_every_query(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math()
        assert {i["id"] for i in instances} == {"q1", "q2"}

    def test_seeded_sampling_is_reproducible(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        first = load_obliq_bench_math(n=1, seed=7)
        second = load_obliq_bench_math(n=1, seed=7)
        assert [i["id"] for i in first] == [i["id"] for i in second]

    def test_candidate_pool_size_keeps_every_gold_id(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math(query_ids=["q1"], candidate_pool_size=1)
        instance = instances[0]
        # candidate_pool_size (1) is smaller than the 2 gold ids -- both gold
        # ids must still appear; the cap only ceilings the negative count.
        assert instance["pool_size"] == 2
        assert "[doc_a]" in instance["prompt"] and "[doc_b]" in instance["prompt"]
        assert "[doc_c]" not in instance["prompt"]  # excluded id never enters the pool

    def test_candidate_pool_size_never_leaks_excluded_ids(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        instances = load_obliq_bench_math(query_ids=["q1"], candidate_pool_size=10)
        assert "doc_c" not in instances[0]["prompt"]

    def test_candidate_pool_size_is_reproducible_per_seed(self, monkeypatch, tmp_path):
        stub_download(monkeypatch, tmp_path)
        first = load_obliq_bench_math(query_ids=["q1"], candidate_pool_size=3, seed=3)
        second = load_obliq_bench_math(query_ids=["q1"], candidate_pool_size=3, seed=3)
        assert first[0]["prompt"] == second[0]["prompt"]
