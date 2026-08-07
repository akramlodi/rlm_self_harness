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
from shrlm.optimization.audit import AuditReport, audit_round, main, run_audited_round
from shrlm.optimization.driver import round_dir
from shrlm.optimization.mining import MiningResult
from tests.optimization.test_driver import (
    BoomVerifier,
    ClientFactory,
    full_script,
    make_config,
    make_miner,
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
    config = make_config(tmp_path)
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
