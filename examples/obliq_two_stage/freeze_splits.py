"""Pre-materialize one stage's obliq_bench_math splits directory using the
explicit, disjoint partition from plan_partition.py, bypassing
materialize_splits' own seeded sampling (which has no way to guarantee
disjoint sets across separately-configured stages drawing from the same
151-query pool).

Writes splits/<env>_<length>_<role>.jsonl + splits/manifest.json in exactly
the format shrlm.experiment.splits.materialize_splits/verify_environment
expect, reusing its own instance_lines/sha256_text/write_split_file helpers
so the bytes it freezes are indistinguishable from what the framework would
have written itself -- materialize_splits sees "already frozen" on first
real invocation and only verifies, never re-draws.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shrlm.environments.obliq_bench_math import load_obliq_bench_math  # noqa: E402
from shrlm.experiment.splits import (  # noqa: E402
    MANIFEST_FILE,
    SPLITS_DIR,
    split_file_name,
    write_split_file,
)
from shrlm.optimization.driver import instance_lines, sha256_file  # noqa: E402

DATASET_REVISION = "4ebee29f68ceeb62ca00bd73d5478acdd3cd7764"
ENVIRONMENT = "obliq_bench_math"
LENGTH = "short"


def freeze_stage_splits(
    out_dir: Path,
    seed: int,
    candidate_pool_size: int,
    mining_ids: list[str],
    validation_ids: list[str],
    promotion_ids: list[str],
    test_ids: list[str],
) -> None:
    """Write one stage's splits/manifest.json for a fresh --out-dir."""
    role_ids = {
        "held_in": mining_ids,
        "held_out": validation_ids,
        "promotion": promotion_ids,
        "test": test_ids,
    }
    all_needed_ids = [qid for ids in role_ids.values() for qid in ids]
    instances_by_id = {
        inst["id"]: inst
        for inst in load_obliq_bench_math(
            query_ids=all_needed_ids,
            candidate_pool_size=candidate_pool_size,
            revision=DATASET_REVISION,
        )
    }

    splits_dir = out_dir / SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    for role, ids in role_ids.items():
        subset = [instances_by_id[qid] for qid in ids]
        file_name = split_file_name(ENVIRONMENT, LENGTH, role)
        content = instance_lines(subset)
        write_split_file(splits_dir / file_name, content, ENVIRONMENT)
        files[file_name] = {"sha256": sha256_file(splits_dir / file_name), "count": len(ids)}

    manifest_path = splits_dir / MANIFEST_FILE
    manifest = {
        "sample_seed": seed,
        "environments": {ENVIRONMENT: {"revision": DATASET_REVISION, "files": files}},
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(content)
    print(f"froze splits for {out_dir} (seed={seed}, pool={candidate_pool_size})")
    for file_name, entry in files.items():
        print(f"  {file_name}: count={entry['count']} sha256={entry['sha256'][:12]}...")
