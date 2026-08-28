"""The fixed GraphWalks instances the live recursion test runs against (U3).

Selection is by content-addressed id, not by sampling: a GraphWalks id is
``<problem_type>-<sha256(prompt)[:16]>`` (``row_to_instance``), so the same
dataset revision always yields the same prompt for the same id. The four ids
below are the largest held-in instances of ``experiment_kimi`` -- the graphs
where the orchestrator addendum's stated ~100K-characters-per-prompt ceiling is
closest to binding, and so where an ``H0*`` root has the strongest stated
reason to delegate.
"""

from typing import Any

from shrlm.environments.graphwalks import load_graphwalks, row_to_instance
from shrlm.experiment.config import ExperimentConfig

# (id, prompt chars, gold-set size) as observed in experiment_kimi round 1.
RECURSION_INSTANCE_IDS: tuple[str, ...] = (
    "bfs-a1c463dbfd91f84c",  # 110,363 chars, gold 181 nodes
    "bfs-e1b3ed79dbe72c0f",  # 110,363 chars, gold 72 nodes
    "parents-7cfd78c8d93d5bfc",  # 110,309 chars, gold 7 nodes
    "bfs-2b1afb378791d043",  # 110,363 chars, gold 1 node
)


def load_recursion_instances(
    config: ExperimentConfig, ids: tuple[str, ...] = RECURSION_INSTANCE_IDS
) -> list[dict[str, Any]]:
    """Load exactly ``ids`` from the configured short GraphWalks file, in id order.

    Reads the dataset repo, file, pinned revision, and character window from
    ``config.environments.graphwalks`` so the ids resolve to the prompts the
    experiment saw. Raises if any id is absent (a revision drift or a typo
    would otherwise silently shrink the test) or if a returned instance's id
    no longer recomputes from its prompt.
    """
    graphwalks = config.environments.graphwalks
    pool = load_graphwalks(
        problem_types=graphwalks.problem_types,
        max_chars=graphwalks.max_chars,
        min_chars=graphwalks.min_chars,
        limit=None,
        seed=config.splits.seed,
        dataset_repo=graphwalks.dataset_repo,
        dataset_file=graphwalks.dataset_file_short,
        revision=graphwalks.dataset_revision,
    )
    by_id = {instance["id"]: instance for instance in pool}
    missing = [instance_id for instance_id in ids if instance_id not in by_id]
    if missing:
        raise LookupError(
            f"recursion instance id(s) {missing} not found in {graphwalks.dataset_file_short} "
            f"at revision {graphwalks.dataset_revision}; {len(by_id)} instances loaded"
        )
    selected = [by_id[instance_id] for instance_id in ids]
    for instance in selected:
        recomputed = row_to_instance(
            {
                "prompt": instance["prompt"],
                "problem_type": instance["problem_type"],
                "answer_nodes": instance["answer_nodes"],
            },
            sample_seed=instance["sample_seed"],
            sample_index=instance["sample_index"],
        )["id"]
        if recomputed != instance["id"]:
            raise ValueError(
                f"instance {instance['id']} does not recompute from its prompt ({recomputed})"
            )
    return selected
