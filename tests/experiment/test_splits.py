"""Tests for split materialization: disjoint partition, determinism, hash
pinning, and the dataset-drift guard.

Everything runs offline: the GraphWalks network seam (``fetch_rows``) is
monkeypatched with synthetic pools large enough for the shipped full-profile
sizes (24/40/40 short, 150 long), so the real config drives the real wiring
end to end. Loader-injection tests bypass GraphWalks entirely via the
``loaders`` mapping.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import shrlm.environments.graphwalks as graphwalks
import shrlm.environments.oolong_pairs as oolong_pairs
from shrlm.experiment.config import ExperimentConfig, load_config
from shrlm.experiment.splits import (
    MANIFEST_FILE,
    SplitsPersistenceError,
    instance_lines,
    materialize_splits,
    sha256_text,
    split_file_name,
    split_plan,
)

BFS_OPERATION = "Perform a BFS from node a with depth 1."
PARENTS_OPERATION = "Find the parents of node d."


def build_pool(n: int, prompt_chars_base: int, tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(n):
        problem_type = "bfs" if index % 2 == 0 else "parents"
        operation = BFS_OPERATION if problem_type == "bfs" else PARENTS_OPERATION
        prompt = (
            "You are given a directed graph as an edge list:\n"
            "a -> b\nb -> d\nc -> d\n"
            f"{operation}\n"
            f"{tag} variant {index}\n"
            "End your response with a line 'Final Answer: [n1, n2, ...]'."
        )
        rows.append(
            {
                "prompt": prompt,
                "problem_type": problem_type,
                "answer_nodes": ["b", "c"],
                "prompt_chars": prompt_chars_base + index,
            }
        )
    return rows


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config()


@pytest.fixture
def fetch_calls(monkeypatch, config: ExperimentConfig) -> list[tuple[str, str, str | None]]:
    """Monkeypatch the GraphWalks download seam; return the recorded calls."""
    short_pool = build_pool(120, 1_000, "short")
    long_pool = build_pool(160, 300_000, "long")
    short_file = config.environments.graphwalks.dataset_file_short
    calls: list[tuple[str, str, str | None]] = []

    def fake_fetch(
        dataset_repo: str, dataset_file: str, revision: str | None
    ) -> list[dict[str, Any]]:
        calls.append((dataset_repo, dataset_file, revision))
        pool = short_pool if dataset_file == short_file else long_pool
        return [dict(row) for row in pool]

    monkeypatch.setattr(graphwalks, "fetch_rows", fake_fetch)

    def oolong_entry(user_id: int, question: str, label: str) -> str:
        return f"Date: Feb 02, 2023 || User: {user_id} || Instance: {question} || Label: {label}"

    def oolong_row(window_id: str, context_len: int) -> dict[str, Any]:
        lines = [
            oolong_entry(101, "How many moons does Mars have ?", "numeric value"),
            oolong_entry(202, "Where is the Eiffel Tower located ?", "location"),
            oolong_entry(303, "What does TNT stand for ?", "abbreviation"),
        ]
        return {
            "dataset": config.environments.oolong_pairs.subset,
            "context_len": context_len,
            "context_window_id": window_id,
            "context_window_text_with_labels": "\n".join(lines),
        }

    short_len = config.environments.oolong_pairs.context_length_short
    long_len = config.environments.oolong_pairs.context_length_long
    oolong_rows = [oolong_row(f"s{i}", short_len) for i in range(4)] + [
        oolong_row(f"l{i}", long_len) for i in range(10)
    ]

    def fake_iter(split: str, revision: str | None):
        return iter([dict(row) for row in oolong_rows])

    monkeypatch.setattr(oolong_pairs, "iter_dataset_rows", fake_iter)
    return calls


def read_ids(splits_dir: Path, environment: str, length: str, role: str) -> list[str]:
    path = splits_dir / split_file_name(environment, length, role)
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def splits_bytes(splits_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(splits_dir.iterdir())}


class TestPartition:
    def test_source_short_partition_is_disjoint_and_exactly_sized(
        self, fetch_calls, config, tmp_path
    ):
        splits_dir = materialize_splits(config, tmp_path)
        held_in = read_ids(splits_dir, "graphwalks", "short", "held_in")
        held_out = read_ids(splits_dir, "graphwalks", "short", "held_out")
        test = read_ids(splits_dir, "graphwalks", "short", "test")
        assert (len(held_in), len(held_out), len(test)) == (24, 40, 40)
        combined = held_in + held_out + test
        assert len(set(combined)) == len(combined), "roles share instance ids"

    def test_source_long_is_test_only_and_sized(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        long_test = read_ids(splits_dir, "graphwalks", "long", "test")
        assert len(long_test) == 150
        assert not (splits_dir / split_file_name("graphwalks", "long", "held_in")).exists()

    def test_wiring_routes_both_dataset_files_under_the_pinned_revision(
        self, fetch_calls, config, tmp_path
    ):
        materialize_splits(config, tmp_path)
        env = config.environments.graphwalks
        assert fetch_calls == [
            (env.dataset_repo, env.dataset_file_short, env.dataset_revision),
            (env.dataset_repo, env.dataset_file_long, env.dataset_revision),
        ]

    def test_split_plan_is_config_derived(self, config):
        plan = split_plan(config)
        assert plan["graphwalks"]["short"] == {"held_in": 24, "held_out": 40, "test": 40}
        assert plan["graphwalks"]["long"] == {"test": 150}
        oolong = config.environments.oolong_pairs
        assert plan["oolong_pairs"] == {
            "short": {"test": oolong.n_short},
            "long": {"test": oolong.n_long},
        }


class TestDeterminism:
    def test_same_seed_produces_byte_identical_files(self, fetch_calls, config, tmp_path):
        first = materialize_splits(config, tmp_path / "a")
        second = materialize_splits(config, tmp_path / "b")
        assert splits_bytes(first) == splits_bytes(second)

    def test_different_seed_produces_different_membership(self, fetch_calls, config, tmp_path):
        first = materialize_splits(config, tmp_path / "a")
        reseeded = replace(config, splits=replace(config.splits, seed=1))
        second = materialize_splits(reseeded, tmp_path / "b")
        assert read_ids(first, "graphwalks", "short", "held_in") != read_ids(
            second, "graphwalks", "short", "held_in"
        )


class TestIdempotency:
    def test_reinvocation_with_matching_hashes_is_a_noop(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        before = splits_bytes(splits_dir)
        assert len(fetch_calls) == 2
        materialize_splits(config, tmp_path)
        assert len(fetch_calls) == 2, "a verified re-invocation must not reload the dataset"
        assert splits_bytes(splits_dir) == before

    def test_tampered_split_file_raises(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        target = splits_dir / split_file_name("graphwalks", "short", "held_out")
        target.write_text(target.read_text() + '{"id": "intruder"}\n')
        with pytest.raises(SplitsPersistenceError, match="sha256"):
            materialize_splits(config, tmp_path)

    def test_missing_split_file_raises(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        (splits_dir / split_file_name("graphwalks", "long", "test")).unlink()
        with pytest.raises(SplitsPersistenceError, match="missing on disk"):
            materialize_splits(config, tmp_path)

    def test_seed_change_against_existing_manifest_raises(self, fetch_calls, config, tmp_path):
        materialize_splits(config, tmp_path)
        reseeded = replace(config, splits=replace(config.splits, seed=1))
        with pytest.raises(SplitsPersistenceError, match="seed"):
            materialize_splits(reseeded, tmp_path)


class TestInterruptedMaterialization:
    """A crash between the split writes and the manifest write must not wedge
    the directory: the files are unaccounted for, but a re-draw reproduces them
    byte for byte (the loaders are seeded), so they are reused, not refused."""

    def counting_loader(self, calls: list[str]):
        def loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            calls.append(length)
            return make_fake_instances(length, limit)

        return loader

    def test_files_left_by_an_interrupted_materialization_are_reused(self, config, tmp_path):
        calls: list[str] = []
        loaders = {"graphwalks": self.counting_loader(calls)}
        splits_dir = materialize_splits(config, tmp_path, loaders=loaders)
        before = splits_bytes(splits_dir)

        # The crash: every split file is on disk, the manifest that accounts
        # for them never landed.
        (splits_dir / MANIFEST_FILE).unlink()

        materialize_splits(config, tmp_path, loaders=loaders)

        assert calls == ["short", "long", "short", "long"], "the retry must re-draw"
        assert splits_bytes(splits_dir) == before, "the re-draw must reuse the identical files"

    def test_an_unaccounted_file_that_diverges_still_raises(self, config, tmp_path):
        calls: list[str] = []
        loaders = {"graphwalks": self.counting_loader(calls)}
        splits_dir = materialize_splits(config, tmp_path, loaders=loaders)
        (splits_dir / MANIFEST_FILE).unlink()
        target = splits_dir / split_file_name("graphwalks", "short", "held_in")
        target.write_text(target.read_text().replace("p0", "tampered"))

        with pytest.raises(SplitsPersistenceError, match="fresh draw does not reproduce"):
            materialize_splits(config, tmp_path, loaders=loaders)


class TestRevisionGuard:
    def test_manifest_records_the_configured_revision(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        manifest = json.loads((splits_dir / MANIFEST_FILE).read_text())
        assert manifest["sample_seed"] == config.splits.seed
        recorded = manifest["environments"]["graphwalks"]
        assert recorded["revision"] == config.environments.graphwalks.dataset_revision

    def test_manifest_hashes_match_the_files(self, fetch_calls, config, tmp_path):
        splits_dir = materialize_splits(config, tmp_path)
        manifest = json.loads((splits_dir / MANIFEST_FILE).read_text())
        files = manifest["environments"]["graphwalks"]["files"]
        assert set(files) == {
            split_file_name("graphwalks", "short", "held_in"),
            split_file_name("graphwalks", "short", "held_out"),
            split_file_name("graphwalks", "short", "test"),
            split_file_name("graphwalks", "long", "test"),
        }
        for file_name, entry in files.items():
            content = (splits_dir / file_name).read_text()
            assert entry["sha256"] == sha256_text(content)
            assert entry["count"] == len(content.splitlines())

    def test_loader_under_a_different_revision_than_the_manifest_raises(
        self, fetch_calls, config, tmp_path
    ):
        materialize_splits(config, tmp_path)
        drifted = replace(
            config,
            environments=replace(
                config.environments,
                graphwalks=replace(config.environments.graphwalks, dataset_revision="deadbeef"),
            ),
        )
        with pytest.raises(SplitsPersistenceError, match="revision"):
            materialize_splits(drifted, tmp_path)


def make_fake_instances(length: str, limit: int) -> list[dict[str, Any]]:
    return [{"id": f"{length}-i{index:03d}", "prompt": f"p{index}"} for index in range(limit)]


class TestLoaderInjection:
    def test_loader_receives_config_length_limit_seed(self, config, tmp_path):
        calls: list[tuple[str, int, int]] = []

        def fake_loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            assert cfg is config
            calls.append((length, limit, seed))
            return make_fake_instances(length, limit)

        materialize_splits(config, tmp_path, loaders={"graphwalks": fake_loader})
        assert calls == [("short", 104, config.splits.seed), ("long", 150, config.splits.seed)]

    def test_short_loader_return_raises(self, config, tmp_path):
        def starved_loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            return make_fake_instances(length, limit - 1)

        with pytest.raises(ValueError, match="exactly 104"):
            materialize_splits(config, tmp_path, loaders={"graphwalks": starved_loader})

    def test_duplicate_instance_ids_raise(self, config, tmp_path):
        def duplicating_loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            instances = make_fake_instances(length, limit)
            instances[1]["id"] = instances[0]["id"]
            return instances

        with pytest.raises(ValueError, match="duplicate instance ids"):
            materialize_splits(config, tmp_path, loaders={"graphwalks": duplicating_loader})

    def test_unknown_environment_raises(self, config, tmp_path):
        def loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            return make_fake_instances(length, limit)

        with pytest.raises(ValueError, match="unknown environment 'bogus'"):
            materialize_splits(config, tmp_path, loaders={"bogus": loader})

    def test_split_file_lines_are_canonical_instance_renderings(self, config, tmp_path):
        instances_by_length = {
            length: make_fake_instances(length, limit)
            for length, limit in (("short", 104), ("long", 150))
        }

        def fake_loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            return instances_by_length[length]

        splits_dir = materialize_splits(config, tmp_path, loaders={"graphwalks": fake_loader})
        held_in = (splits_dir / split_file_name("graphwalks", "short", "held_in")).read_text()
        assert held_in == instance_lines(instances_by_length["short"][:24])


class TestOolongSynthEnvironment:
    """``loop.environment == "oolong_synth"`` mines/validates the OOLONG-synth
    pool; GraphWalks is never materialized."""

    @pytest.fixture
    def oolong_config(self) -> ExperimentConfig:
        return load_config("full", path=Path("configs/experiment_oolong.toml"))

    def test_split_plan_holds_only_the_selected_environment(self, oolong_config):
        plan = split_plan(oolong_config)
        assert set(plan) == {"oolong_synth", "oolong_real"}  # real_check_every_n_rounds = 2 > 0
        assert plan["oolong_synth"]["short"] == {
            "held_in": oolong_config.splits.n_in,
            "held_out": oolong_config.splits.n_ho,
            "test": oolong_config.splits.test_short,
        }
        assert plan["oolong_real"]["short"] == {
            "check": oolong_config.environments.oolong.real.n_check
        }

    def test_real_check_absent_from_plan_when_disabled(self, oolong_config):
        disabled = replace(
            oolong_config,
            operational=replace(oolong_config.operational, real_check_every_n_rounds=0),
        )
        assert set(split_plan(disabled)) == {"oolong_synth"}

    def test_materializes_oolong_only_and_skips_graphwalks(self, oolong_config, tmp_path):
        seen: list[str] = []

        def fake_loader(
            cfg: ExperimentConfig, length: str, limit: int, seed: int
        ) -> list[dict[str, Any]]:
            seen.append(length)
            return [
                {"id": f"oolong-{length}-i{index:03d}", "prompt": f"p{index}"}
                for index in range(limit)
            ]

        splits_dir = materialize_splits(
            oolong_config,
            tmp_path,
            loaders={
                "graphwalks": pytest.fail,  # must never be called on an OOLONG run
                "oolong_synth": fake_loader,
                "oolong_real": fake_loader,
            },
        )
        for role in ("held_in", "held_out", "test"):
            assert (splits_dir / split_file_name("oolong_synth", "short", role)).exists()
        assert (splits_dir / split_file_name("oolong_real", "short", "check")).exists()
        assert not (splits_dir / split_file_name("graphwalks", "short", "held_in")).exists()
        manifest = json.loads((splits_dir / MANIFEST_FILE).read_text())
        assert set(manifest["environments"]) == {"oolong_synth", "oolong_real"}


class TestOolongPairsEnvironment:
    def test_split_plan_partitions_the_finite_short_pool_and_keeps_long_for_test(self, config):
        pairs_config = replace(
            config,
            loop=replace(config.loop, environment="oolong_pairs"),
            splits=replace(config.splits, n_in=15, n_ho=15, test_short=10, test_long=40),
        )

        assert split_plan(pairs_config) == {
            "oolong_pairs": {
                "short": {"held_in": 15, "held_out": 15, "test": 10},
                "long": {"test": 40},
            }
        }

    def test_split_plan_rejects_counts_above_the_pinned_inventory(self, config):
        short_overflow = replace(
            config,
            loop=replace(config.loop, environment="oolong_pairs"),
            splits=replace(config.splits, n_in=15, n_ho=15, test_short=11, test_long=41),
        )

        with pytest.raises(ValueError, match=r"short roles require 41.*pool to 40"):
            split_plan(short_overflow)

        long_overflow = replace(
            short_overflow,
            splits=replace(short_overflow.splits, test_short=10),
        )
        with pytest.raises(ValueError, match=r"long test requires 41.*pool to 40"):
            split_plan(long_overflow)

    def test_materializes_only_the_selected_pair_pools(self, tmp_path):
        config = load_config(
            "full", path=Path("configs/experiment_oolong_pairs_DeepSeekV4Flash.toml")
        )

        splits_dir = materialize_splits(
            config,
            tmp_path,
            loaders={
                "oolong_pairs": lambda cfg, length, limit, seed: make_fake_instances(length, limit)
            },
        )

        expected_counts = {
            "oolong_pairs_short_held_in.jsonl": 10,
            "oolong_pairs_short_held_out.jsonl": 10,
            "oolong_pairs_short_test.jsonl": 20,
            "oolong_pairs_long_test.jsonl": 40,
        }
        manifest = json.loads((splits_dir / MANIFEST_FILE).read_text())
        files = manifest["environments"]["oolong_pairs"]["files"]
        assert {name: details["count"] for name, details in files.items()} == expected_counts
