"""Tests for the validation subject worker: one subject per child process (U2).

The worker is exercised the way the dispatcher uses it -- a real child
``python -m shrlm.optimization.subject_worker <request>`` -- with the runtime
scripted through the child-side seam in ``subject_worker_support`` (KTD9).
What is pinned: the child persists the same ``summary.json`` the in-process
path writes, refuses a harness envelope that does not round-trip to its
expected hash before touching any split directory, reports an unresolvable
factory as a result document rather than a bare crash, and resumes
persist-first (a second run makes zero model calls).
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shrlm.harness_identity import harness_hash, serialize_harness
from shrlm.optimization.subject_worker import (
    CLIENT_CALLS_FILENAME,
    REQUEST_FILENAME,
    RESULT_FILENAME,
    build_request,
    read_result,
    run_subject_worker,
    write_request,
)
from shrlm.optimization.validation import (
    SPLIT_HELDIN,
    SPLIT_HELDOUT,
    SUMMARY_FILENAME,
    load_summary,
    subject_dir,
)
from shrlm.rlm_harness import H0
from tests.optimization.subject_worker_support import SCRIPTED_FACTORY, write_script
from tests.optimization.test_validation import (
    CAPS,
    GOLD_VERIFIER_FACTORY,
    final,
    make_splits,
)


def request_for(
    tmp_path: Path,
    subject_id: str,
    script: list[str],
    *,
    harness: Any = H0,
    expected_hash: str | None = None,
    verifier_factory: str = GOLD_VERIFIER_FACTORY,
    client_factory: str = SCRIPTED_FACTORY,
) -> Path:
    out_dir = tmp_path / "validation"
    subject_path = subject_dir(out_dir, 1, subject_id)
    script_path = write_script(tmp_path / "scripts" / f"{subject_id}.json", script)
    splits = make_splits()
    request = build_request(
        subject_id=subject_id,
        harness_serialization=serialize_harness(harness),
        expected_hash=expected_hash or harness_hash(harness),
        splits={SPLIT_HELDIN: splits.heldin, SPLIT_HELDOUT: splits.heldout},
        caps=CAPS,
        repetitions=2,
        backend="openai",
        backend_kwargs={"model_name": "validation-test"},
        out_dir=out_dir,
        round_index=1,
        verifier_factory=verifier_factory,
        client_factory=(client_factory, {"script_path": str(script_path)}),
    )
    return write_request(subject_path, request)


def run_child(request_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "shrlm.optimization.subject_worker", str(request_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def client_calls(subject_path: Path) -> int:
    return int(json.loads((subject_path / CLIENT_CALLS_FILENAME).read_text())["calls"])


class TestHappyPath:
    def test_child_persists_the_scripted_summary_and_exits_zero(self, tmp_path):
        # 2 instances x 2 splits x 2 reps = 8 runs: held-in 3/4, held-out 4/4.
        script = [final("RIGHT"), final("RIGHT"), final("WRONG"), final("RIGHT")] + [
            final("RIGHT")
        ] * 4
        request_path = request_for(tmp_path, "cand-a", script)
        completed = run_child(request_path)
        assert completed.returncode == 0, completed.stderr

        subject_path = request_path.parent
        result = read_result(subject_path)
        assert result is not None and result["ok"] is True
        assert result["subject_id"] == "cand-a"
        assert Path(result["summary_path"]) == subject_path / SUMMARY_FILENAME

        summary = load_summary(subject_path)
        assert summary["subject_id"] == "cand-a"
        assert summary["harness_hash"] == harness_hash(H0)
        assert summary["splits"][SPLIT_HELDIN]["pass_count"] == 3
        assert summary["splits"][SPLIT_HELDOUT]["pass_count"] == 4
        assert client_calls(subject_path) == 8
        # The status line is echoed to stdout for the worker log.
        assert json.loads(completed.stdout.strip().splitlines()[-1])["ok"] is True

    def test_second_run_resumes_with_zero_model_calls(self, tmp_path):
        script = [final("RIGHT")] * 8
        request_path = request_for(tmp_path, "cand-b", script)
        assert run_child(request_path).returncode == 0
        first = load_summary(request_path.parent)
        assert client_calls(request_path.parent) == 8

        # Re-run against the SAME script file: nothing is popped on resume.
        assert run_child(request_path).returncode == 0
        assert client_calls(request_path.parent) == 0
        assert load_summary(request_path.parent) == first

    def test_request_round_trips_through_disk(self, tmp_path):
        request_path = request_for(tmp_path, "cand-c", [final("RIGHT")] * 8)
        payload = json.loads(request_path.read_text())
        assert request_path.name == REQUEST_FILENAME
        assert payload["caps"] == asdict(CAPS)
        assert payload["client_factory"][0] == SCRIPTED_FACTORY
        assert payload["harness"]["hash"] == harness_hash(H0)


class TestRefusals:
    def test_tampered_envelope_is_refused_before_any_split_directory_exists(self, tmp_path):
        request_path = request_for(tmp_path, "cand-t", [final("RIGHT")] * 8, expected_hash="0" * 64)
        completed = run_child(request_path)
        assert completed.returncode == 1
        result = read_result(request_path.parent)
        assert result is not None and result["ok"] is False
        assert "rematerializes to hash" in result["error"]
        assert not (request_path.parent / SPLIT_HELDIN).exists()
        assert not (request_path.parent / SPLIT_HELDOUT).exists()
        assert not (request_path.parent / SUMMARY_FILENAME).exists()

    def test_unresolvable_verifier_factory_is_reported_as_a_result(self, tmp_path):
        request_path = request_for(
            tmp_path,
            "cand-v",
            [final("RIGHT")] * 8,
            verifier_factory="tests.optimization.no_such_module:Verifier",
        )
        completed = run_child(request_path)
        assert completed.returncode == 1
        result = read_result(request_path.parent)
        assert result is not None and result["ok"] is False
        assert "no_such_module" in result["error"]
        assert not (request_path.parent / SUMMARY_FILENAME).exists()

    def test_wrong_format_request_is_a_result_not_a_crash(self, tmp_path):
        subject_path = tmp_path / "bad"
        subject_path.mkdir()
        request_path = subject_path / REQUEST_FILENAME
        request_path.write_text(json.dumps({"format": "something-else"}))
        result = run_subject_worker(request_path)
        assert result["ok"] is False
        assert "shrlm-subject-worker-request/v1" in result["error"]
        assert (subject_path / RESULT_FILENAME).exists()

    def test_write_request_unlinks_a_stale_result(self, tmp_path):
        request_path = request_for(tmp_path, "cand-s", [final("RIGHT")] * 8)
        stale = request_path.parent / RESULT_FILENAME
        stale.write_text(json.dumps({"format": "shrlm-subject-worker-result/v1", "ok": True}))
        write_request(request_path.parent, json.loads(request_path.read_text()))
        assert not stale.exists()

    def test_missing_result_file_reads_as_none(self, tmp_path):
        assert read_result(tmp_path) is None
        (tmp_path / RESULT_FILENAME).write_text("{not json")
        assert read_result(tmp_path) is None
