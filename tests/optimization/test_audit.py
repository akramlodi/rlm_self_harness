"""Tests for the U8 audit walk: prove the round's evidence chain link by link.

The audit consumes only what the round persisted -- ``harness.json``,
``instances.jsonl``, ``runs.jsonl``, the trace files, the content-addressed
digests, the rendered attributor prompt(s), and the bundle triplet
(``bundle.json``, ``records.jsonl``, ``attributions.jsonl``) -- and verifies
every join between them. The contract under test is destructive: a full
MockLM round audits clean, and deleting or corrupting any single artifact
makes the audit fail naming exactly the broken link, never passing silently.

The round itself is driven through the same offline seam as the driver tests:
``rlm.core.rlm.get_client`` is patched to a scripted ``MockLM`` factory, and
the attributor is a ``MockLM`` with a canned valid attribution.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

import rlm.core.rlm as rlm_module
from shrlm.optimization.attribution import AttributionCache, AttributorConfig, LLMAttributor
from shrlm.optimization.audit import AuditReport, audit_round, main, run_audited_round
from shrlm.optimization.driver import round_dir
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from tests.mock_lm import MockLM
from tests.optimization.test_driver import (
    BoomVerifier,
    ClientFactory,
    final,
    full_script,
    make_instances,
    make_miner,
    make_round_config,
)

Mutator = Callable[[Path, MiningResult], None]


# ---------------------------------------------------------------------------
# One audited MockLM round: run -> mine -> bundle -> audit, all offline
# ---------------------------------------------------------------------------


@pytest.fixture
def audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, MiningResult, AuditReport]:
    """A complete audited round: 3 runs (1 pass, 1 fail, 1 termination)."""
    factory = ClientFactory(full_script())
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_round_config(tmp_path)
    result, report = run_audited_round(config, make_miner(BoomVerifier()), split_id="held_in_v1")
    return round_dir(config.out_dir, config.round_index), result, report


class TestHappyPath:
    def test_full_round_audits_clean(self, audited):
        _, _, report = audited
        assert report.ok
        assert report.broken == ()
        assert all(link.failures == 0 for link in report.links)

    def test_stats_are_recomputed_from_the_artifacts(self, audited):
        _, result, report = audited
        stats = report.stats
        assert stats.n_runs == 3
        assert stats.n_passed == 1
        assert stats.pass_rate == pytest.approx(1 / 3)
        assert stats.n_records == len(result.records) == 2
        # WRONG_VALUE and RESOURCE_TERMINATED give two distinct signatures.
        assert stats.n_patterns == 2
        # No sub-verifier ran, so no record carries a checkable child verdict.
        assert stats.grounding_coverage == pytest.approx(0.0)

    def test_reaudit_from_disk_alone_is_clean(self, audited):
        round_path, _, _ = audited
        report = audit_round(round_path.parent, 1)
        assert report.ok

    def test_every_named_link_is_reported(self, audited):
        _, _, report = audited
        names = [link.link for link in report.links]
        assert len(names) == len(set(names))
        for expected in (
            "harness_file",
            "harness_hash",
            "manifest_trace",
            "bundle_id",
            "record_trace",
            "record_digest",
            "attribution_prompt",
        ):
            assert expected in names


# ---------------------------------------------------------------------------
# Breakage: each deleted or corrupted artifact names its own link
# ---------------------------------------------------------------------------


def _delete_failure_trace(round_path: Path, result: MiningResult) -> None:
    trace_path = result.records[0].trace_path
    assert trace_path is not None
    (round_path / trace_path).unlink()


def _corrupt_trace_byte(round_path: Path, result: MiningResult) -> None:
    trace_path = result.records[0].trace_path
    assert trace_path is not None
    target = round_path / trace_path
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))


def _delete_digest_text(round_path: Path, result: MiningResult) -> None:
    sha = result.records[0].digest_sha256
    (round_path / "digests" / f"{sha}.txt").unlink()


def _delete_harness_json(round_path: Path, result: MiningResult) -> None:
    (round_path / "harness.json").unlink()


def _delete_attributor_prompt(round_path: Path, result: MiningResult) -> None:
    (round_path / "attributor_prompt.txt").unlink()


def _delete_bundle_json(round_path: Path, result: MiningResult) -> None:
    (round_path / "bundle.json").unlink()


def _delete_digests_dir(round_path: Path, result: MiningResult) -> None:
    """Pre-U7 rounds persisted no digest texts at all."""
    shutil.rmtree(round_path / "digests")


def _strip_trace_links(round_path: Path, result: MiningResult) -> None:
    """Pre-U7 records carried no run_id / trace_path / trace_sha256."""
    records_path = round_path / "records.jsonl"
    stripped = []
    for line in records_path.read_text().splitlines():
        payload = json.loads(line)
        payload["run_id"] = None
        payload["trace_path"] = None
        payload["trace_sha256"] = None
        stripped.append(json.dumps(payload, sort_keys=True))
    records_path.write_text("\n".join(stripped) + "\n")


def _empty_unattributed_attempts(round_path: Path, result: MiningResult) -> None:
    """An unattributed entry with no audit trail is an unusable attribution."""
    attributions_path = round_path / "attributions.jsonl"
    entries = [json.loads(line) for line in attributions_path.read_text().splitlines()]
    entries[0]["attributed"] = False
    entries[0]["error"] = "no valid attribution after 3 attempts"
    entries[0]["attempts"] = []
    attributions_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    )


BREAKAGES: list[tuple[str, Mutator, set[str]]] = [
    ("deleted trace file", _delete_failure_trace, {"manifest_trace", "record_trace"}),
    ("corrupted trace byte", _corrupt_trace_byte, {"manifest_trace", "record_trace"}),
    ("deleted digest text", _delete_digest_text, {"record_digest", "attribution_digest"}),
    ("deleted harness.json", _delete_harness_json, {"harness_file"}),
    ("deleted attributor prompt", _delete_attributor_prompt, {"attribution_prompt"}),
    ("deleted bundle.json", _delete_bundle_json, {"bundle_file"}),
    ("pre-U7: no digests dir", _delete_digests_dir, {"record_digest", "attribution_digest"}),
    ("pre-U7: unlinked records", _strip_trace_links, {"record_run", "record_trace"}),
    ("emptied unattributed attempts", _empty_unattributed_attempts, {"attribution_attempts"}),
]


class TestBreakage:
    @pytest.mark.parametrize(
        ("label", "mutate", "expected_links"),
        BREAKAGES,
        ids=[label for label, _, _ in BREAKAGES],
    )
    def test_each_broken_artifact_names_its_link(self, audited, label, mutate, expected_links):
        round_path, result, _ = audited
        mutate(round_path, result)

        report = audit_round(round_path.parent, 1)

        assert not report.ok
        assert {broken.link for broken in report.broken} == expected_links

    @pytest.mark.parametrize(
        "extra_fields",
        [
            {"attribution_error_kind": "transport"},
            {"error": "transport failure: LM unreachable"},
        ],
        ids=["typed kind", "legacy prefix"],
    )
    def test_transport_failed_entry_is_exempt_from_the_attempts_demand(self, audited, extra_fields):
        round_path, _, _ = audited
        attributions_path = round_path / "attributions.jsonl"
        entries = [json.loads(line) for line in attributions_path.read_text().splitlines()]
        entries[0].update(attributed=False, attempts=[], **extra_fields)
        attributions_path.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
        )

        report = audit_round(round_path.parent, 1)

        assert "attribution_attempts" not in report.broken_link_names()

    def test_broken_link_names_the_offending_artifact(self, audited):
        round_path, result, _ = audited
        _delete_failure_trace(round_path, result)

        report = audit_round(round_path.parent, 1)

        record = result.records[0]
        assert record.run_id is not None and record.trace_path is not None
        subjects = {broken.subject for broken in report.broken}
        assert record.run_id in subjects
        messages = " ".join(broken.message for broken in report.broken)
        assert record.trace_path in messages


# ---------------------------------------------------------------------------
# Malformed artifacts: missing required keys are broken links, not crashes
# ---------------------------------------------------------------------------


def _drop_jsonl_key(path: Path, index: int, key: str) -> None:
    lines = path.read_text().splitlines()
    payload = json.loads(lines[index])
    del payload[key]
    lines[index] = json.dumps(payload, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")


def _drop_manifest_run_id(round_path: Path, result: MiningResult) -> None:
    _drop_jsonl_key(round_path / "runs.jsonl", 0, "run_id")


def _drop_manifest_trace_path(round_path: Path, result: MiningResult) -> None:
    _drop_jsonl_key(round_path / "runs.jsonl", 0, "trace_path")


def _drop_manifest_trace_sha(round_path: Path, result: MiningResult) -> None:
    _drop_jsonl_key(round_path / "runs.jsonl", 0, "trace_sha256")


def _drop_instance_id(round_path: Path, result: MiningResult) -> None:
    _drop_jsonl_key(round_path / "instances.jsonl", 0, "id")


def _drop_record_instance_id(round_path: Path, result: MiningResult) -> None:
    _drop_jsonl_key(round_path / "records.jsonl", 0, "instance_id")


def _drop_bundle_config(round_path: Path, result: MiningResult) -> None:
    bundle_path = round_path / "bundle.json"
    payload = json.loads(bundle_path.read_text())
    del payload["config"]
    bundle_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


MALFORMED: list[tuple[str, Mutator, str]] = [
    ("manifest line missing run_id", _drop_manifest_run_id, "manifest_file"),
    ("manifest line missing trace_path", _drop_manifest_trace_path, "manifest_trace"),
    ("manifest line missing trace_sha256", _drop_manifest_trace_sha, "manifest_trace"),
    ("instance line missing id", _drop_instance_id, "instances_file"),
    ("record missing instance_id", _drop_record_instance_id, "record_instance"),
    ("bundle missing config", _drop_bundle_config, "bundle_id"),
]


class TestMalformedArtifacts:
    @pytest.mark.parametrize(
        ("label", "mutate", "expected_link"),
        MALFORMED,
        ids=[label for label, _, _ in MALFORMED],
    )
    def test_missing_required_key_is_a_named_broken_link(
        self, audited, label, mutate, expected_link
    ):
        round_path, result, _ = audited
        mutate(round_path, result)

        report = audit_round(round_path.parent, 1)  # must report, never raise

        assert not report.ok
        assert expected_link in report.broken_link_names()

    def test_malformed_manifest_line_names_the_line(self, audited):
        round_path, result, _ = audited
        _drop_manifest_run_id(round_path, result)

        report = audit_round(round_path.parent, 1)

        broken = [b for b in report.broken if b.link == "manifest_file"]
        assert broken and broken[0].subject == "line 1"
        assert "run_id" in broken[0].message

    def test_malformed_manifest_still_exits_one_from_the_cli(self, audited, capsys):
        round_path, result, _ = audited
        _drop_manifest_trace_path(round_path, result)
        exit_code = main([str(round_path.parent), "1"])
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "manifest_trace" in out


# ---------------------------------------------------------------------------
# Pipeline seams: re-invocation, operational visibility, cache stamping
# ---------------------------------------------------------------------------


def make_transport_down_miner() -> WeaknessMiner:
    """A miner whose attributor LM raises on every call: pure transport failure."""
    return WeaknessMiner(
        verifier=BoomVerifier(),
        attributor=LLMAttributor(
            MockLM(responses=[]),  # every completion raises IndexError
            config=AttributorConfig(transport_backoff_seconds=0.0),
        ),
    )


class TestPipelineSeams:
    def test_run_audited_round_is_reinvocable_over_the_same_out_dir(self, tmp_path, monkeypatch):
        """A second identical invocation reuses the persisted created_at, so
        the re-mined bundle is byte-identical and the no-clobber guard passes."""
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path)
        result1, report1 = run_audited_round(
            config, make_miner(BoomVerifier()), split_id="held_in_v1"
        )
        assert report1.ok
        bundle_path = round_dir(config.out_dir, config.round_index) / "bundle.json"
        before = bundle_path.read_bytes()

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        result2, report2 = run_audited_round(
            make_round_config(tmp_path), make_miner(BoomVerifier()), split_id="held_in_v1"
        )

        assert report2.ok
        assert idle.total_calls == 0
        assert result2.bundle.created_at == result1.bundle.created_at
        assert bundle_path.read_bytes() == before

    def test_overwrite_bundle_is_the_explicit_escape_for_a_divergent_remine(
        self, tmp_path, monkeypatch
    ):
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path)
        run_audited_round(config, make_miner(BoomVerifier()), split_id="held_in_v1")

        divergent = (
            "```json\n"
            + json.dumps(
                {
                    "causal_status": "causal",
                    "agent_mechanism": "premature_termination",
                    "failing_level": "root",
                    "evidence_node_ids": ["r"],
                    "symptom_summary": "stopped before checking the answer",
                }
            )
            + "\n```"
        )
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        with pytest.raises(ValueError, match="diverge"):
            run_audited_round(
                make_round_config(tmp_path),
                make_miner(BoomVerifier(), responses=[divergent] * 8),
                split_id="held_in_v1",
            )

        result, report = run_audited_round(
            make_round_config(tmp_path),
            make_miner(BoomVerifier(), responses=[divergent] * 8),
            split_id="held_in_v1",
            overwrite_bundle=True,
        )
        assert report.ok
        mechanisms = {pattern.signature.agent_mechanism.value for pattern in result.bundle.patterns}
        assert "premature_termination" in mechanisms

    def test_all_transport_failure_round_surfaces_the_counts(self, tmp_path, monkeypatch, capsys):
        """A round mined with the LM down still audits ok (unattributed records
        are legitimate evidence) but the zero-signature outcome must be visible
        in the stats and in MiningResult.errors, never silent."""
        factory = ClientFactory(full_script())
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path)

        result, report = run_audited_round(
            config, make_transport_down_miner(), split_id="held_in_v1"
        )

        assert report.ok
        assert report.stats.n_records == 2
        assert report.stats.n_patterns == 0
        assert report.stats.n_unattributed == 2
        assert report.stats.n_transport_errors == 2
        assert len(result.errors) == 2
        assert result.bundle.patterns == []

        exit_code = main([str(config.out_dir), "1"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "n_unattributed=2" in out
        assert "n_transport_errors=2" in out
        assert "every failure record is unattributed" in out

    def test_clean_round_reports_zero_unattributed(self, audited):
        _, _, report = audited
        assert report.stats.n_unattributed == 0
        assert report.stats.n_transport_errors == 0

    def test_all_pass_round_with_file_backed_cache_audits_clean(self, tmp_path, monkeypatch):
        """put() never runs on an all-pass round: no cache file, no stamped
        path, and the audit's attribution_cache link stays unbroken."""
        factory = ClientFactory([final("RIGHT"), final("RIGHT")])
        monkeypatch.setattr(rlm_module, "get_client", factory)
        config = make_round_config(tmp_path, instances=make_instances()[:2])

        cache_path = tmp_path / "caches" / "attribution.jsonl"
        miner = WeaknessMiner(
            verifier=BoomVerifier(),
            attributor=LLMAttributor(
                MockLM(responses=[]), cache=AttributionCache(path=str(cache_path))
            ),
        )
        result, report = run_audited_round(config, miner, split_id="held_in_v1")

        assert report.ok
        assert result.bundle.config.attribution_cache_path is None
        assert not cache_path.exists()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


class TestCli:
    def test_clean_round_exits_zero_and_prints_the_walk(self, audited, capsys):
        round_path, _, _ = audited
        exit_code = main([str(round_path.parent), "1"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "harness_hash" in out
        assert "OK" in out

    def test_broken_round_exits_nonzero_naming_the_link(self, audited, capsys):
        round_path, result, _ = audited
        _delete_failure_trace(round_path, result)
        exit_code = main([str(round_path.parent), "1"])
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "record_trace" in out


if __name__ == "__main__":
    pytest.main([__file__])
