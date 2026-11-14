"""Compute a deterministic, disjoint partition of all 151 OBLIQ-Bench Math
queries across up to 3 optimization stages + a shared final-eval pool.

Budget (from explicit instruction):
  - 3 stages x (mining=15 + validation=15 + promotion=1) = 93
  - shared full-evaluation test pool = 40 (identical queries across every
    stage, so stages are comparable on the same final benchmark regardless
    of candidate_pool_size)
  - remainder -> "other_test_short" (not used by any stage right now)

Partition is built by a single deterministic shuffle of the sorted 151 query
ids (MASTER_SEED), sliced sequentially into named buckets -- guarantees zero
overlap by construction, unlike relying on two independently-seeded samples
from the same pool (which would almost certainly overlap heavily: each
per-stage sample is a sizeable fraction of only 151 total queries).

Each stage's "validation" (held_out) bucket has 3 ids moved to the front so
screening's default_screening_subset (which just takes the first 3 of
held_out, in materialized order) picks 1 hard + 2 medium instances by a
gold-set-size proxy -- smaller gold set = fewer correct targets = harder
retrieval. The 3 are chosen from across the FULL 15-instance bucket (hard =
smallest gold-set size overall; medium = the two closest to that bucket's
median size, excluding whichever instance was already picked as hard) --
not just whichever 3 instances happened to land first before reordering.
This is a heuristic, not validated against real trace data (there is no run
yet to validate against, unlike oolong_pairs' curated task order).
"""

import json
from pathlib import Path
from random import Random

MASTER_SEED = 0
HERE = Path(__file__).parent
ALL_IDS_PATH = HERE / "all_query_ids.json"
OUT_PATH = HERE / "partition.json"

PER_STAGE_MINING = 15
PER_STAGE_VALIDATION = 15
PER_STAGE_PROMOTION = 1
N_STAGES_RESERVED = 3  # budget for stage1, stage2, and a possible stage3
FULL_EVAL_TEST = 40


def main() -> None:
    entries = json.loads(ALL_IDS_PATH.read_text())
    by_id = {e["id"]: e["n_gold"] for e in entries}
    pool = sorted(by_id)
    assert len(pool) == 151, f"expected 151 queries, got {len(pool)}"

    shuffled = pool[:]
    Random(MASTER_SEED).shuffle(shuffled)

    cursor = 0

    def take(n: int) -> list[str]:
        nonlocal cursor
        chunk = shuffled[cursor : cursor + n]
        cursor += n
        return chunk

    def reorder_hard_medium_first(validation: list[str]) -> list[str]:
        """Move 1 hard (smallest gold set) + 2 medium (closest to this
        bucket's median gold-set size) instances to the front, selected
        from across the whole bucket -- not just whichever 3 happen to sit
        first before reordering."""
        by_size = sorted(validation, key=lambda qid: by_id[qid])
        hard = by_size[0]
        median_size = by_id[by_size[len(by_size) // 2]]
        remaining = [qid for qid in by_size if qid != hard]
        remaining.sort(key=lambda qid: abs(by_id[qid] - median_size))
        medium = remaining[:2]
        head = [hard, *medium]
        tail = [qid for qid in validation if qid not in head]
        return head + tail

    partition: dict[str, list[str]] = {}
    for stage in (1, 2, 3):
        mining = take(PER_STAGE_MINING)
        validation = reorder_hard_medium_first(take(PER_STAGE_VALIDATION))
        promotion = take(PER_STAGE_PROMOTION)
        partition[f"stage{stage}_mining"] = mining
        partition[f"stage{stage}_validation"] = validation
        partition[f"stage{stage}_promotion"] = promotion

    partition["full_eval_test"] = take(FULL_EVAL_TEST)
    partition["other_test_short"] = shuffled[cursor:]  # whatever remains

    # Stage 1 calibration override, 2026-11: round 1's real mining results
    # showed the gold-set-size proxy picked the WRONG "hard" instance --
    # q02312 (n_gold=2) scored a trivial 1.000 NDCG once it produced any
    # answer at all. Swap in q01424 (n_gold=37, already in stage1_validation)
    # as the new hard pick -- real traces show large gold sets are where H0
    # actually struggles (q00740/q01307/q01630/q01588 all hit <0.7 NDCG on
    # n_gold=37 instances). q01217 (n_gold=6) is swapped for q01528, round 1
    # mining's single worst performer (0/2 pass, mean NDCG 0.000, both
    # attempts an explicit empty ranking) -- real difficulty signal beats the
    # proxy, so q01528 moves mining->validation and q01217 moves the other
    # way to keep both buckets at 15 and disjoint. q00564 (n_gold=6) is left
    # as the remaining medium pick.
    # Screening widened from 3 to 6 instances (orchestrator.py's
    # _screening_subset): two per gold-set-size band (<5, 13, 37) instead of
    # 3 narrowly-clustered picks. Within each band the pick is the hardest by
    # real full-validation terminal-zero mean (empty/wrong_format scored 0),
    # not by gold-set size alone -- q02312 (n_gold=2) looked like a "hard"
    # pick but is actually mostly easy with one unlucky miss (1.0, 0.0, 1.0,
    # mean 0.667); q02144 (also n_gold=2) fails consistently (0.0, 0.0,
    # wrong_format, mean 0.0) and is the genuinely hard one. Same logic for
    # q01368 (n_gold=3, mean 0.099) over q02887 (mean 1.0); q00239/q00844
    # (n_gold=13, means 0.0/0.363) over q01528 (mean 0.384); q00104/q01424
    # (n_gold=37, means 0.481/0.698) over q01905/q00570 (means 0.814/1.0).
    s1_validation, s1_mining = partition["stage1_validation"], partition["stage1_mining"]
    assert "q02312" in s1_validation and "q01424" in s1_validation
    assert "q01217" in s1_validation and "q01528" in s1_mining
    s1_mining[s1_mining.index("q01528")] = "q01217"
    s1_validation.remove("q01217")
    s1_validation.append("q01528")
    head = ["q02144", "q01368", "q00239", "q00844", "q00104", "q01424"]
    for qid in head:
        assert qid in s1_validation, f"{qid} missing from stage1_validation"
    partition["stage1_validation"] = head + [
        qid for qid in s1_validation if qid not in head
    ]
    partition["stage1_mining"] = s1_mining

    total_allocated = sum(len(v) for v in partition.values())
    assert total_allocated == 151, f"partition covers {total_allocated}, expected 151"
    all_allocated_ids = [qid for ids in partition.values() for qid in ids]
    assert len(set(all_allocated_ids)) == len(all_allocated_ids), "overlap detected!"

    OUT_PATH.write_text(json.dumps(partition, indent=2, sort_keys=True))

    print("Partition (disjoint, verified, 151/151 covered):")
    for key in sorted(partition):
        ids = partition[key]
        gold_sizes = [by_id[i] for i in ids]
        print(f"  {key:22s} n={len(ids):3d}  gold_sizes(first 5)={gold_sizes[:5]}")
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
