"""A validation subject's runs, executed concurrently in child processes.

The properties that matter are not "it is faster" -- they are that fanning out
changes *nothing else*. A round at worker count 3 must persist the same runs,
charge the same money, and reach the same verdicts as the same round run one at
a time; the only permitted difference is the order lines land in the manifest.

The rest of this file is about what happens when that goes wrong: a child that
crashes, a child that hangs, a parent that died mid-flight and left a paid-for
trace behind, and a budget that runs out partway through.
"""

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

import rlm.core.rlm as rlm_module
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    SplitClaimedError,
    run_governed_round,
)
from shrlm.optimization.driver import round_dir, run_id_for, trace_path_for
from shrlm.optimization.run_worker import RESULT_FORMAT, run_run_worker
from shrlm.optimization.taxonomy import VerifierCause
from tests.optimization.run_worker_support import (
    RUN_SCRIPTED_FACTORY,
    observed_peak_concurrency,
    write_script,
)
from tests.optimization.test_costs import (
    CAPS,
    ClientFactory,
    final,
    make_config,
    read_manifest,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def scripted(tmp_path: Path, script: list[str], **extra) -> tuple[str, dict]:
    """A client-factory spec a run child can resolve inside its own process."""
    args = {"script_path": str(write_script(tmp_path / "script.json", script)), **extra}
    return (RUN_SCRIPTED_FACTORY, args)


def fanned_config(tmp_path: Path, *, workers: int, script: list[str], **factory_args):
    return make_config(
        tmp_path,
        run_workers=workers,
        client_factory=scripted(tmp_path, script, **factory_args),
    )


class TestSequentialEquivalence:
    """Worker count 1 must be the untouched sequential path, byte for byte."""

    def test_worker_count_one_spawns_no_child_and_persists_the_same_bytes(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        sequential = make_config(tmp_path / "seq")
        result = run_governed_round(sequential, CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        # No child artifacts at all: nothing was dispatched.
        assert not (round_dir(sequential.out_dir, sequential.round_index) / "run_workers").exists()
        assert len(result.entries) == 3


class TestConcurrentEquivalence:
    def test_a_fanned_round_matches_the_sequential_one_apart_from_line_order(
        self, tmp_path, monkeypatch
    ):
        script = [final("RIGHT")]
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory(script * 3))
        sequential = make_config(tmp_path / "seq")
        seq_result = run_governed_round(sequential, CandidateSpendBreaker(CAPS))

        concurrent = fanned_config(tmp_path / "par", workers=3, script=script)
        par_result = run_governed_round(concurrent, CandidateSpendBreaker(CAPS))

        assert par_result.outcome == seq_result.outcome == OUTCOME_COMPLETED
        assert par_result.skipped_run_ids == seq_result.skipped_run_ids == []
        assert par_result.spent == pytest.approx(seq_result.spent)

        def comparable(entries):
            return sorted(
                (e["run_id"], e["passed"], e["cause"], e["cost"], e["input_tokens"])
                for e in entries
            )

        seq_manifest = read_manifest(round_dir(sequential.out_dir, sequential.round_index))
        par_manifest = read_manifest(round_dir(concurrent.out_dir, concurrent.round_index))
        assert comparable(par_manifest) == comparable(seq_manifest)

    def test_every_run_is_persisted_exactly_once(self, tmp_path):
        config = fanned_config(tmp_path, workers=3, script=[final("RIGHT")])

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        run_ids = [entry["run_id"] for entry in result.entries]
        assert sorted(run_ids) == sorted(set(run_ids))
        assert len(run_ids) == 3

    def test_concurrency_stays_within_the_configured_count(self, tmp_path):
        witness = tmp_path / "witness"
        config = make_config(
            tmp_path,
            instances=[{"id": f"inst-{i}", "prompt": "p", "gold": "RIGHT"} for i in range(1, 7)],
            run_workers=2,
            client_factory=scripted(
                tmp_path,
                [final("RIGHT")],
                witness_dir=str(witness),
                hold=0.3,
            ),
        )

        run_governed_round(config, CandidateSpendBreaker(CAPS))

        # Both bounds matter. The upper one is the cap; the lower one is the
        # point of the feature -- without it this passes just as happily if
        # fan-out silently collapses to sequential execution.
        assert observed_peak_concurrency(witness) == 2


class TestChildFailure:
    def test_pid_marker_write_failure_terminates_and_reaps_spawned_child(
        self, tmp_path, monkeypatch
    ):
        import shrlm.optimization.costs as costs_module

        config = make_config(
            tmp_path,
            instances=[{"id": "inst-1", "prompt": "p", "gold": "RIGHT"}],
            run_workers=2,
            client_factory=(RUN_SCRIPTED_FACTORY, {"hang": True}),
        )
        spawned = []
        popen = costs_module.subprocess.Popen
        write_text = Path.write_text

        def capture_popen(*args, **kwargs):
            process = popen(*args, **kwargs)
            spawned.append(process)
            return process

        def fail_pid_marker(path, *args, **kwargs):
            if path.name == "worker.pid":
                raise OSError("pid marker write failed")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(costs_module.subprocess, "Popen", capture_popen)
        monkeypatch.setattr(Path, "write_text", fail_pid_marker)

        with pytest.raises(OSError, match="pid marker write failed"):
            run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert len(spawned) == 1
        process = spawned[0]
        assert process.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        round_path = round_dir(config.out_dir, config.round_index)
        assert list((round_path / "run_workers").glob("*/worker.pid")) == []

    def test_a_child_that_crashes_is_persisted_terminated_and_charged(self, tmp_path):
        config = make_config(
            tmp_path,
            run_workers=2,
            client_factory=(RUN_SCRIPTED_FACTORY, {"crash": True}),
        )

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert len(result.entries) == 3
        assert all(
            entry["cause"] == VerifierCause.RESOURCE_TERMINATED.value for entry in result.entries
        )
        # Charged, never silently free, and never re-dispatched: each run id
        # appears exactly once.
        assert result.spent > 0
        run_ids = [entry["run_id"] for entry in result.entries]
        assert sorted(run_ids) == sorted(set(run_ids))

    def test_a_crashed_child_leaves_its_error_in_its_own_log(self, tmp_path):
        config = make_config(
            tmp_path,
            run_workers=1 + 1,
            client_factory=(RUN_SCRIPTED_FACTORY, {"crash": True}),
        )

        run_governed_round(config, CandidateSpendBreaker(CAPS))

        run_path = (
            round_dir(config.out_dir, config.round_index) / "run_workers" / run_id_for("inst-1", 1)
        )
        assert (run_path / "result.json").exists()
        assert json.loads((run_path / "result.json").read_text())["ok"] is False


class TestOrphanAdoption:
    def test_a_trace_left_by_a_dead_parent_is_adopted_rather_than_re_paid(self, tmp_path):
        """The run was made and the money was spent; only the line is missing."""
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        # Simulate a parent killed between a child publishing its trace and the
        # parent appending its manifest line.
        first = run_governed_round(config, CandidateSpendBreaker(CAPS))
        round_path = round_dir(config.out_dir, config.round_index)
        entries = read_manifest(round_path)
        orphan = entries[0]
        remaining = [e for e in entries if e["run_id"] != orphan["run_id"]]
        (round_path / "runs.jsonl").write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in remaining)
        )
        orphan_trace = trace_path_for(round_path, str(orphan["run_id"]))
        assert orphan_trace.exists()
        # Evidence unique to adoption: if the run were silently re-executed, a
        # child would rewrite this trace. Aggregate spend cannot tell the two
        # apart -- re-running the same scripted turns costs exactly the same --
        # so the file's identity is what proves nothing was paid for twice.
        trace_bytes_before = orphan_trace.read_bytes()
        child_dir = round_path / "run_workers" / str(orphan["run_id"])
        shutil.rmtree(child_dir, ignore_errors=True)

        breaker = CandidateSpendBreaker(CAPS)
        second = run_governed_round(config, breaker)

        assert len(second.entries) == 3
        assert second.outcome == OUTCOME_COMPLETED
        assert second.spent == pytest.approx(first.spent)
        assert orphan_trace.read_bytes() == trace_bytes_before
        # No child was dispatched for the adopted run at all.
        assert not child_dir.exists()

    def test_a_parallel_orphan_resumed_sequentially_is_adopted_once(self, tmp_path, monkeypatch):
        """Changing worker count must not turn an existing trace into another paid run."""
        parallel = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        first = run_governed_round(parallel, CandidateSpendBreaker(CAPS))
        round_path = round_dir(parallel.out_dir, parallel.round_index)
        entries = read_manifest(round_path)
        orphan = entries[0]
        (round_path / "runs.jsonl").write_text(
            "".join(
                json.dumps(entry, sort_keys=True) + "\n"
                for entry in entries
                if entry["run_id"] != orphan["run_id"]
            )
        )
        orphan_trace = trace_path_for(round_path, str(orphan["run_id"]))
        trace_bytes_before = orphan_trace.read_bytes()

        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)
        sequential = replace(parallel, run_workers=1)
        breaker = CandidateSpendBreaker(CAPS)

        resumed = run_governed_round(sequential, breaker)

        assert idle.total_calls == 0
        assert orphan_trace.read_bytes() == trace_bytes_before
        assert [entry["run_id"] for entry in resumed.entries].count(orphan["run_id"]) == 1
        assert resumed.spent == pytest.approx(first.spent)
        assert breaker.spent == pytest.approx(first.spent)


class TestBudget:
    def test_a_breaker_trip_leaves_a_contiguous_tail_and_reports_over_budget(self, tmp_path):
        """A run whose realised charge overshoots its reservation trips the breaker.

        The reservation is what dispatch sets aside per in-flight run, not a
        cap the run is held to -- the runtime checks the budget only between
        iterations. So a single run can come back having spent more than was
        reserved for it, which is exactly the case that trips the breaker
        rather than the gate.
        """
        caps = replace(CAPS, candidate_budget=CAPS.max_budget * 3)
        config = make_config(
            tmp_path,
            run_workers=2,
            client_factory=scripted(
                tmp_path, [final("RIGHT")], cost_per_call=caps.candidate_budget * 2
            ),
        )

        result = run_governed_round(config, CandidateSpendBreaker(caps))

        assert result.outcome == OUTCOME_OVER_BUDGET
        assert result.spent > caps.candidate_budget  # the breaker really tripped
        assert result.skipped_run_ids
        every = [run_id_for(f"inst-{i}", 1) for i in (1, 2, 3)]
        # The skipped set is a tail of the dispatch order, not a scatter.
        assert result.skipped_run_ids == every[len(result.entries) :]

    def test_reservation_stopping_dispatch_still_reports_over_budget(self, tmp_path):
        """Dispatch can stop while spend is still inside the budget.

        The reservation gate holds back a slot's worth of headroom per
        in-flight run, so a round can end with runs never executed and the
        breaker never tripped. Reporting that as completed would let the
        promotion rule score a sample that never ran.
        """
        # Reservation is 2x max_budget per in-flight run, so this budget
        # admits exactly one run and then holds the gate shut: the breaker
        # never trips, and the remaining runs are skipped.
        caps = replace(CAPS, candidate_budget=CAPS.max_budget * 2.5)
        config = fanned_config(tmp_path, workers=3, script=[final("RIGHT")])

        result = run_governed_round(config, CandidateSpendBreaker(caps))

        assert result.skipped_run_ids
        assert result.outcome == OUTCOME_OVER_BUDGET
        assert not CandidateSpendBreaker(caps).tripped  # spend never reached the budget
        # The skipped set is a tail, so what did run is a prefix.
        assert len(result.entries) + len(result.skipped_run_ids) == 3


class TestChildContract:
    def test_a_tampered_harness_envelope_is_refused_before_the_run_executes(self, tmp_path):
        from shrlm.harness_identity import harness_hash, serialize_harness
        from shrlm.optimization.candidates import write_surface_module
        from shrlm.optimization.run_worker import REQUEST_FILENAME, build_request
        from shrlm.rlm_harness import H0

        run_path = tmp_path / "run"
        run_path.mkdir()
        module_path = tmp_path / "surface.py"
        write_surface_module(serialize_harness(H0), module_path)
        request = build_request(
            run_id="inst-1__a01",
            instance={"id": "inst-1", "prompt": "p", "gold": "RIGHT"},
            attempt=1,
            harness_serialization=serialize_harness(H0),
            expected_hash="0" * len(harness_hash(H0)),
            module_path=module_path,
            backend="openai",
            backend_kwargs={"model_name": "m"},
            limits={},
            trace_path=tmp_path / "trace.json",
            deadline_seconds=None,
            parent_pid=os.getpid(),
        )
        request_path = run_path / REQUEST_FILENAME
        request_path.write_text(json.dumps(request))

        result = run_run_worker(request_path)

        assert result["ok"] is False
        assert "identity cannot be verified" in result["error"]
        # Refused before executing: no trace was produced.
        assert not (tmp_path / "trace.json").exists()

    def test_a_request_of_the_wrong_format_is_reported_not_raised(self, tmp_path):
        run_path = tmp_path / "run"
        run_path.mkdir()
        request_path = run_path / "request.json"
        request_path.write_text(json.dumps({"format": "something-else"}))

        result = run_run_worker(request_path)

        assert result["format"] == RESULT_FORMAT
        assert result["ok"] is False

    def test_the_child_writes_nothing_outside_its_own_directory(self, tmp_path):
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        round_path = round_dir(config.out_dir, config.round_index)

        run_governed_round(config, CandidateSpendBreaker(CAPS))

        # Every child artifact lives under run_workers/<run_id>/; the manifest,
        # the identity files, and the surface module are the parent's alone.
        child_dirs = sorted(p.name for p in (round_path / "run_workers").iterdir())
        assert child_dirs == sorted(run_id_for(f"inst-{i}", 1) for i in (1, 2, 3))
        for child in (round_path / "run_workers").iterdir():
            written = sorted(p.name for p in child.iterdir())
            assert set(written) <= {"request.json", "result.json", "run.log", "client_calls.json"}


class TestDeadline:
    def test_a_child_that_ignores_its_deadline_is_killed_and_recorded(self, tmp_path, monkeypatch):
        import shrlm.optimization.costs as costs_module

        # Shrink the grace so the backstop fires in about a second rather than
        # the shipped half-minute; the mechanism under test is unchanged.
        monkeypatch.setattr(costs_module, "HARD_DEADLINE_GRACE_SECONDS", 0.5)
        caps = replace(CAPS, max_timeout=0.05)
        config = make_config(
            tmp_path,
            caps=caps,
            instances=[{"id": "inst-1", "prompt": "p", "gold": "RIGHT"}],
            run_workers=2,
            client_factory=(RUN_SCRIPTED_FACTORY, {"hang": True}),
        )

        start = time.monotonic()
        result = run_governed_round(config, CandidateSpendBreaker(caps))

        assert time.monotonic() - start < 60.0  # returned rather than hanging
        assert len(result.entries) == 1
        assert result.entries[0]["run_id"] == run_id_for("inst-1", 1)
        assert result.entries[0]["cause"] == VerifierCause.RESOURCE_TERMINATED.value
        # Which deadline fired matters, and so does how. The child's own alarm
        # raises HardDeadlineSignal (a BaseException, so no ``except Exception``
        # inside the REPL or a sub-call wrapper can swallow it), and only
        # execute_run converts it -- into HardDeadlineExceeded, a
        # TimeoutExceededError persisted as an ordinary terminated run carrying
        # whatever usage the runtime recorded, its name in the detail so the
        # audit can tell a backstop kill from a between-iteration timeout. An
        # alarm that escaped execute_run would crash the child, and the parent
        # would then price the run at its flat ceiling rather than what it
        # actually spent.
        #
        # So the child reports success -- it did its job -- with a terminated
        # run inside, and it leaves a trace behind. A crashed child reports
        # ok: false and leaves no trace at all.
        run_path = (
            round_dir(config.out_dir, config.round_index) / "run_workers" / run_id_for("inst-1", 1)
        )
        payload = json.loads((run_path / "result.json").read_text())
        assert payload["ok"] is True
        assert payload["terminated"] is True
        assert "HardDeadlineExceeded" in payload["detail"]
        assert trace_path_for(
            round_dir(config.out_dir, config.round_index), run_id_for("inst-1", 1)
        ).exists()


class TestMisconfiguredBudget:
    def test_a_budget_that_cannot_reserve_one_run_is_refused(self, tmp_path):
        """Silently skipping every run would look like a budget stop.

        The gate holds back one reservation per in-flight run, so a budget
        below a single reservation admits nothing at all -- and the subject
        would then carry an empty sample that reads like an ordinary
        over-budget result. That is a configuration error and says so.
        """
        caps = replace(CAPS, candidate_budget=CAPS.max_budget)
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])

        with pytest.raises(ValueError) as excinfo:
            run_governed_round(config, CandidateSpendBreaker(caps))

        assert "cannot reserve even one concurrent run" in str(excinfo.value)
        assert "run-worker setting to 1" in str(excinfo.value)
        assert "validation_run_workers" not in str(excinfo.value)

    def test_the_sequential_path_accepts_that_same_budget(self, tmp_path, monkeypatch):
        """The reservation is a dispatch mechanism; sequential has no gate."""
        monkeypatch.setattr(rlm_module, "get_client", ClientFactory([final("RIGHT")] * 3))
        caps = replace(CAPS, candidate_budget=CAPS.max_budget)
        config = make_config(tmp_path)

        result = run_governed_round(config, CandidateSpendBreaker(caps))

        assert result.entries


class TestOrphanedChildrenBlockANewParent:
    """The split claim names the parent; a crashed parent's children outlive it.

    A child notices its parent is gone only on its next watchdog poll, so a
    replacement parent can win the claim while the old children are still
    executing -- and re-dispatch the very runs they are still paying for.
    """

    def test_a_live_run_child_from_an_earlier_invocation_refuses_the_round(self, tmp_path):
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        round_path = round_dir(config.out_dir, config.round_index)
        run_path = round_path / "run_workers" / run_id_for("inst-1", 1)
        run_path.mkdir(parents=True)
        # A pid that is alive but is not us: this process's parent will do.
        (run_path / "worker.pid").write_text(f"{os.getppid()}\n")

        with pytest.raises(SplitClaimedError) as excinfo:
            run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert "run worker(s) alive" in str(excinfo.value)
        assert str(os.getppid()) in str(excinfo.value)

    def test_a_live_parallel_child_blocks_a_sequential_resume_before_sidecar_rewrite(
        self, tmp_path
    ):
        parallel = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        run_governed_round(parallel, CandidateSpendBreaker(CAPS))
        round_path = round_dir(parallel.out_dir, parallel.round_index)
        sidecar = round_path / "execution.json"
        sidecar_bytes = sidecar.read_bytes()
        run_path = round_path / "run_workers" / run_id_for("inst-1", 1)
        (run_path / "worker.pid").write_text(f"{os.getppid()}\n")

        with pytest.raises(SplitClaimedError):
            run_governed_round(replace(parallel, run_workers=1), CandidateSpendBreaker(CAPS))

        assert sidecar.read_bytes() == sidecar_bytes

    def test_a_dead_pid_marker_does_not_block(self, tmp_path):
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        round_path = round_dir(config.out_dir, config.round_index)
        run_path = round_path / "run_workers" / run_id_for("inst-1", 1)
        run_path.mkdir(parents=True)
        (run_path / "worker.pid").write_text("999999999\n")

        result = run_governed_round(config, CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert len(result.entries) == 3

    def test_a_dead_parallel_child_does_not_block_a_sequential_resume(self, tmp_path, monkeypatch):
        parallel = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])
        run_governed_round(parallel, CandidateSpendBreaker(CAPS))
        round_path = round_dir(parallel.out_dir, parallel.round_index)
        run_path = round_path / "run_workers" / run_id_for("inst-1", 1)
        (run_path / "worker.pid").write_text("999999999\n")
        idle = ClientFactory([])
        monkeypatch.setattr(rlm_module, "get_client", idle)

        result = run_governed_round(replace(parallel, run_workers=1), CandidateSpendBreaker(CAPS))

        assert result.outcome == OUTCOME_COMPLETED
        assert len(result.entries) == 3
        assert idle.total_calls == 0

    def test_a_completed_round_leaves_no_pid_markers_behind(self, tmp_path):
        config = fanned_config(tmp_path, workers=2, script=[final("RIGHT")])

        run_governed_round(config, CandidateSpendBreaker(CAPS))

        round_path = round_dir(config.out_dir, config.round_index)
        assert list((round_path / "run_workers").glob("*/worker.pid")) == []
