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
from shrlm.harness_identity import hash_of_serialization
from shrlm.optimization.attribution import AttributionCache, AttributorConfig, LLMAttributor
from shrlm.optimization.audit import (
    AuditReport,
    audit_round,
    bundle_dir_for,
    main,
    run_audited_round,
    stored_harness_hash,
)
from shrlm.optimization.driver import mine_round, round_dir
from shrlm.optimization.mining import MiningResult, WeaknessMiner
from tests.mock_lm import MockLM
from tests.optimization.test_driver import (
    BoomVerifier,
    ClientFactory,
    NoneSubVerifier,
    final,
    full_script,
    make_instances,
    make_miner,
    make_round_config,
    run_dual_mined_round,
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

    def test_stored_harness_hash_delegates_to_hash_of_serialization(self):
        # The audit-side entrypoint must agree byte-for-byte with the
        # canonical hasher that minted the envelope, name field ignored.
        serialization = {"name": "informational", "orchestrator": True, "surfaces": {"S1": "x"}}
        envelope = {"harness": serialization}
        assert stored_harness_hash(envelope) == hash_of_serialization(serialization)
        renamed = {"harness": {**serialization, "name": "renamed"}}
        assert stored_harness_hash(renamed) == stored_harness_hash(envelope)

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
    for prompt_file in round_path.glob("attributor_prompt*.txt"):
        prompt_file.unlink()


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
# Content-addressed prompts: a re-mine never invalidates an earlier audit
# ---------------------------------------------------------------------------


class TestPromptPersistenceAcrossMines:
    def test_a_second_mining_pass_keeps_the_first_bundles_prompt_link(self, audited):
        """The U2 regression: re-mining the same round with a different prompt
        variant (here grounded, via a stub sub-verifier over trees with no
        sub-calls) must not clobber the prompt file the first bundle's
        attributions.jsonl entries hash-link to."""
        round_path, _, first_report = audited
        assert first_report.ok

        mine_round(
            out_dir=round_path.parent,
            round_index=1,
            miner=make_miner(BoomVerifier(), sub_verifier=NoneSubVerifier()),
            split_id="held_in_v1",
        )

        report = audit_round(round_path.parent, 1)
        assert "attribution_prompt" not in report.broken_link_names()
        assert report.ok

    def test_legacy_round_with_only_the_unsuffixed_prompt_still_resolves(self, audited):
        """Rounds mined before content-addressing persisted one plain
        attributor_prompt.txt; the resolver's fallback must keep them clean."""
        round_path, result, _ = audited
        assert len(result.attributor_prompts) == 1
        sha, text = next(iter(result.attributor_prompts.items()))
        (round_path / f"attributor_prompt_{sha[:16]}.txt").unlink()
        (round_path / "attributor_prompt.txt").write_text(text)

        report = audit_round(round_path.parent, 1)
        assert report.ok


# ---------------------------------------------------------------------------
# Labeled bundle destinations: two modes' bundles coexist under one round
# ---------------------------------------------------------------------------


class TestLabeledBundleDestinations:
    @pytest.fixture
    def dual(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """One MockLM round mined twice: ungrounded at the round root,
        grounded (stub sub-verifier) into ``bundles/grounded/``, sharing one
        round-root-relative attribution cache."""
        cache_path = tmp_path / "caches" / "attribution.jsonl"
        outcome = run_dual_mined_round(
            monkeypatch,
            script=full_script(),
            first_config=make_round_config(tmp_path),
            first_miner=make_miner(BoomVerifier(), cache=AttributionCache(path=str(cache_path))),
            second_config=make_round_config(tmp_path),
            second_miner_factory=lambda: make_miner(
                BoomVerifier(),
                sub_verifier=NoneSubVerifier(),
                cache=AttributionCache(path=str(cache_path)),
            ),
            second_label="grounded",
        )
        return (
            round_dir(tmp_path, 1),
            outcome.first_result,
            outcome.first_report,
            outcome.second_result,
            outcome.second_report,
        )

    def test_both_bundles_coexist_and_audit_clean(self, dual):
        round_path, result_root, report_root, result_sub, report_sub = dual
        assert report_root.ok
        assert report_sub.ok
        assert result_root.bundle.bundle_id != result_sub.bundle.bundle_id
        for triplet_dir in (round_path, bundle_dir_for(round_path, "grounded")):
            assert (triplet_dir / "bundle.json").is_file()
            assert (triplet_dir / "records.jsonl").is_file()
            assert (triplet_dir / "attributions.jsonl").is_file()

    def test_subdirectory_bundle_reaudits_from_disk_with_shared_root_artifacts(self, dual):
        round_path, _, _, result_sub, _ = dual
        report = audit_round(round_path.parent, 1, bundle_label="grounded")
        assert report.ok
        # Every shared link really was checked against round-root artifacts:
        # manifest runs, traces, digests, prompts, and the shared cache path.
        checked = {link.link: link.checked for link in report.links}
        assert checked["record_run"] > 0
        assert checked["record_digest"] > 0
        assert checked["attribution_prompt"] > 0
        assert checked["attribution_cache"] == 1
        assert result_sub.bundle.config.attribution_cache_path is not None

    def test_root_bundle_audit_is_unchanged_by_the_subdirectory_bundle(self, dual):
        round_path, result_root, _, _, _ = dual
        report = audit_round(round_path.parent, 1)
        assert report.ok
        bundle = json.loads((round_path / "bundle.json").read_text())
        assert bundle["bundle_id"] == result_root.bundle.bundle_id

    def test_reinvocation_is_byte_idempotent_per_destination(self, dual, monkeypatch, tmp_path):
        """created_at reuse keys on the bundle being rewritten: the labeled
        re-mine reads bundles/grounded/bundle.json, never the root bundle
        (whose created_at differs), so each destination reproduces itself."""
        round_path, _, _, _, _ = dual
        sub_bundle_path = bundle_dir_for(round_path, "grounded") / "bundle.json"
        root_bundle_path = round_path / "bundle.json"
        sub_before = sub_bundle_path.read_bytes()
        root_before = root_bundle_path.read_bytes()
        cache_path = tmp_path / "caches" / "attribution.jsonl"

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        _, report_sub = run_audited_round(
            make_round_config(tmp_path),
            make_miner(
                BoomVerifier(),
                sub_verifier=NoneSubVerifier(),
                cache=AttributionCache(path=str(cache_path)),
            ),
            split_id="held_in_v1",
            bundle_label="grounded",
        )
        _, report_root = run_audited_round(
            make_round_config(tmp_path),
            make_miner(BoomVerifier(), cache=AttributionCache(path=str(cache_path))),
            split_id="held_in_v1",
        )

        assert report_sub.ok and report_root.ok
        assert idle.total_calls == 0
        assert sub_bundle_path.read_bytes() == sub_before
        assert root_bundle_path.read_bytes() == root_before

    def test_cli_audits_the_labeled_bundle(self, dual, capsys):
        round_path, _, _, _, _ = dual
        exit_code = main([str(round_path.parent), "1", "--bundle-label", "grounded"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "OK" in out

        (bundle_dir_for(round_path, "grounded") / "bundle.json").unlink()
        exit_code = main([str(round_path.parent), "1", "--bundle-label", "grounded"])
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "bundle_file" in out


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

    def test_bad_bundle_label_exits_two_with_a_one_line_error(self, audited, capsys):
        """An unsafe --bundle-label is an operator error: one line on stderr
        and exit code 2, never a raw ValueError traceback."""
        round_path, _, _ = audited
        exit_code = main([str(round_path.parent), "1", "--bundle-label", ".hidden"])
        captured = capsys.readouterr()
        assert exit_code == 2
        assert "not filesystem-safe" in captured.err
        assert len(captured.err.strip().splitlines()) == 1
        assert "Traceback" not in captured.err + captured.out


if __name__ == "__main__":
    pytest.main([__file__])
