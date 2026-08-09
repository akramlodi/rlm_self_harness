"""Live smoke for the proposal-validation stage: one hand-written candidate, end to end.

One ``validate_round`` call against OpenRouter ``qwen/qwen3-30b-a3b-instruct-2507``:
a hand-written S2 candidate (a decomposition-instruction edit to ``H0``) is
written as a real ``shrlm-proposal/v1`` artifact, gated by the loader, and
evaluated with the incumbent over two fabricated splits of 2 instances each,
1 repetition -- 8 model runs total (baseline 4 + candidate 4; one candidate
can never merge, and a single accepted candidate promotes without
re-evaluation, so 8 is the hard ceiling).

The smoke asserts artifact structure and ledger completeness, NEVER model
behavior: whether the candidate is accepted or rejected is the model's
business; what must hold is that every run persisted, every summary
aggregated, every ledger record landed, every audit link resolves, and total
spend stays under the stated ceiling.

Re-running is nearly free: persisted runs are replayed from disk and the
ledger rewrite is a byte-identical no-op.

Usage:
    OPENROUTER_API_KEY=... uv run python -m examples.validation_live_smoke
    # or put the key in .env. Artifacts land under --out-dir
    # (default ./validation_smoke): proposals/ and validation/round_00/.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from shrlm.harness_identity import harness_hash, hash_of_serialization, serialize_harness
from shrlm.optimization.candidates import HARNESS_FORMAT, PROPOSAL_FILENAME, PROPOSAL_FORMAT
from shrlm.optimization.costs import ValidationCaps
from shrlm.optimization.driver import sha256_file
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.optimization.validation import (
    BASELINE_ID,
    DECISION_FILENAME,
    PROMOTIONS_FILENAME,
    EvaluationConfig,
    ValidationRound,
    ValidationSplits,
    load_promotion_ledger,
    validate_round,
)
from shrlm.rlm_harness import H0

load_dotenv()

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
CANDIDATE_ID = "smoke-s2-restate"

# The stated ceiling: 8 tiny runs of a cheap model cost well under this; the
# caps below make it unreachable anyway (8 runs x $0.02 per-run budget = $0.16).
SPEND_CEILING_USD = 0.25

# Experiment-owned limits (R3): per-run budget/timeout plus the per-candidate
# cumulative breaker. Tight enough that a runaway run cannot breach the ceiling.
CAPS = ValidationCaps(
    max_depth=2,
    max_iterations=4,
    max_budget=0.02,
    max_timeout=180.0,
    candidate_budget=0.10,
)


def make_splits() -> ValidationSplits:
    """Two fabricated instances per split: trivial, deterministic-to-grade sums."""

    def instance(instance_id: str, a: int, b: int) -> dict[str, Any]:
        return {
            "id": instance_id,
            "prompt": f"Compute {a} + {b} and answer with just the number.",
            "gold": str(a + b),
        }

    return ValidationSplits(
        heldin=[instance("hi-1", 17, 25), instance("hi-2", 130, 70)],
        heldout=[instance("ho-1", 61, 8), instance("ho-2", 240, 60)],
    )


@dataclass
class SubstringVerifier:
    """Passes when the gold string appears in the produced answer.

    Deliberately lenient: the smoke never asserts on pass counts, so the
    verifier only needs to be deterministic, not strict.
    """

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold = str(instance["gold"])
        if gold in (produced or ""):
            return Verdict(passed=True, cause=None, gold=gold, produced=produced)
        return Verdict(passed=False, cause=VerifierCause.WRONG_VALUE, gold=gold, produced=produced)


def write_hand_written_candidate(proposals_dir: Path) -> None:
    """One S2 edit to H0, serialized into a real ``shrlm-proposal/v1`` artifact."""
    candidate = replace(
        H0,
        name=CANDIDATE_ID,
        decomposition_instruction=H0.decomposition_instruction
        + "\n\nBefore decomposing, restate the task in one sentence to confirm "
        "what is being asked.",
    )
    serialization = serialize_harness(candidate)
    proposal = {
        "format": PROPOSAL_FORMAT,
        "candidate_id": CANDIDATE_ID,
        "base_harness_hash": harness_hash(H0),
        "target_signature": {
            "verifier_cause": "wrong_value",
            "failing_level": "root",
            "causal_status": "causal",
            "agent_mechanism": "lossy_aggregation",
        },
        "surface": "S2",
        "harness": {
            "format": HARNESS_FORMAT,
            "name": serialization["name"],
            "hash": hash_of_serialization(serialization),
            "harness": serialization,
        },
        "predicted_effect": "The root restates the task before splitting it, reducing "
        "misread-question failures.",
        "regression_risks": ["One extra sentence of context per run; marginally higher cost."],
        "provenance": {"model": "hand-written", "prompt_sha256": "0" * 64},
    }
    candidate_dir = proposals_dir / CANDIDATE_ID
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / PROPOSAL_FILENAME).write_text(json.dumps(proposal, indent=2))


def assert_links_resolve(round_path: Path, links: dict[str, Any], expected_hash: str) -> None:
    """Walk one subject's ledger links down to sha-verified traces, audit-style."""
    assert (round_path / links["summary"]).is_file(), f"missing summary {links['summary']}"
    for split_links in links["splits"].values():
        nested = round_path / split_links["round_dir"]
        harness_file = round_path / split_links["harness"]
        assert nested.is_dir(), f"missing round dir {nested}"
        assert harness_file.is_file(), f"missing harness identity {harness_file}"
        recorded = json.loads(harness_file.read_text())["hash"]
        assert recorded == expected_hash, f"{harness_file} names {recorded}, not {expected_hash}"
        manifest = [
            json.loads(line)
            for line in (nested / "runs.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for entry in manifest:
            trace = nested / entry["trace_path"]
            assert trace.is_file(), f"missing trace {trace}"
            assert sha256_file(trace) == entry["trace_sha256"], f"trace sha mismatch for {trace}"


def check_artifacts(result: ValidationRound) -> tuple[int, float]:
    """Structure and completeness only -- never model behavior. Returns (runs, spend)."""
    assert result.loader_rejections == [], (
        f"the hand-written candidate must load cleanly, got {result.loader_rejections}"
    )
    assert result.evaluation is not None and result.ledger is not None

    subjects = [result.evaluation.baseline, *result.evaluation.candidates]
    if result.merge_evaluation is not None:  # unreachable with one candidate, checked anyway
        subjects.append(result.merge_evaluation)
    total_runs = 0
    total_spend = 0.0
    for subject in subjects:
        assert subject.summary_path.is_file(), f"missing {subject.summary_path}"
        total_spend += float(subject.summary["spent"])
        for split_id, split_summary in subject.summary["splits"].items():
            n_runs = int(split_summary["n_runs"])
            total_runs += n_runs
            assert n_runs == 2, f"{subject.subject_id}/{split_id} persisted {n_runs} runs, not 2"
    assert total_runs <= 8, f"the smoke's ceiling is 8 runs, got {total_runs}"

    round_path = result.round_path
    assert (round_path / PROMOTIONS_FILENAME).is_file()
    assert (round_path / DECISION_FILENAME).is_file()
    records, decision = load_promotion_ledger(round_path)
    recorded_ids = [record["subject_id"] for record in records]
    assert CANDIDATE_ID in recorded_ids, f"ledger misses {CANDIDATE_ID}: {recorded_ids}"
    assert decision["baseline"]["subject_id"] == BASELINE_ID
    assert_links_resolve(
        round_path, decision["baseline"]["links"], decision["baseline"]["harness_hash"]
    )
    for record in records:
        assert record["decision"], f"record for {record['subject_id']} carries no decision"
        assert record["reasons"] is not None
        if record["links"] is not None:
            assert_links_resolve(round_path, record["links"], record["harness_hash"])
    return total_runs, total_spend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="./validation_smoke",
        help="Where proposals/ and validation/round_00/ land.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set. Set it (e.g. in a .env file) and re-run.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    proposals_dir = out_dir / "proposals"
    write_hand_written_candidate(proposals_dir)
    config = EvaluationConfig(
        splits=make_splits(),
        verifier=SubstringVerifier(),
        caps=CAPS,
        out_dir=out_dir / "validation",
        round_index=0,
        repetitions=1,
        backend="openrouter",
        # The key comes from the environment, never from kwargs (it would be logged).
        backend_kwargs={"model_name": MODEL},
    )

    print(f"Validating {CANDIDATE_ID} against {MODEL} (8 runs max)...")
    result = validate_round(H0, proposals_dir, config)

    total_runs, total_spend = check_artifacts(result)
    print(f"Round artifacts: {result.round_path}")
    print(f"Runs persisted: {total_runs} (ceiling 8)")
    print(f"Plan: {result.plan.kind}; promoted: {result.promoted}")
    if result.promoted:
        print(f"Promoted harness hash: {result.promoted_harness_hash}")
    print(f"Total spend: ${total_spend:.4f} (ceiling ${SPEND_CEILING_USD})")
    assert total_spend < SPEND_CEILING_USD, (
        f"spend ${total_spend:.4f} breached the ${SPEND_CEILING_USD} ceiling"
    )
    print("SMOKE PASSED: artifact structure and ledger complete, links resolve.")


if __name__ == "__main__":
    main()
