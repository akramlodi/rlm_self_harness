"""One-off: drive round 1's repair branch against the already-persisted,
hand-patched merged/baseline summaries (Plan B -- see conversation).

Not a permanent script. Constructs ``_Experiment`` directly and calls
``_refine()`` without going through ``run()``'s identity check, since this
out-dir's persisted identity predates ``[refinement]`` being enabled. Splits
(held_in/held_out) and the new ``promotion`` role were already verified
in place by hand before this runs; this script only drives the actual
mining -> attribution -> proposal -> validation spend.
"""

import json
from pathlib import Path

from shrlm.experiment.config import load_config
from shrlm.experiment.orchestrator import (
    ROLE_HELD_IN,
    ROLE_HELD_OUT,
    ROLE_PROMOTION,
    ExperimentPersistenceError,
    _Experiment,
    _read_split,
    experiment_round_dir,
    rematerialize_harness_envelope,
    resolve_env_binding,
)
from shrlm.experiment.splits import SPLITS_DIR
from shrlm.optimization.validation import ValidationSplits
from shrlm.rlm_harness import HARNESSES

CONFIG_PATH = "configs/experiment_oolong_pairs_blended_DeepSeekV4Flash.toml"
OUT_DIR = Path("experiment_real_runs/oolong_pairs_h0_init_blended_production")
ROUND_INDEX = 1


def main() -> None:
    config = load_config("full", path=CONFIG_PATH)
    binding = resolve_env_binding(config)
    experiment = _Experiment(
        config=config,
        out_dir=OUT_DIR,
        verifier=binding.verifier,
        sub_verifier=binding.sub_verifier,
        attributor_lm=None,
        proposer_lm=None,
        loaders=None,
    )

    splits_dir = OUT_DIR / SPLITS_DIR
    experiment.splits_dir = splits_dir
    experiment.splits = ValidationSplits(
        heldin=[
            instance
            for length in binding.lengths
            for instance in _read_split(splits_dir, binding.name, length, ROLE_HELD_IN)
        ],
        heldout=[
            instance
            for length in binding.lengths
            for instance in _read_split(splits_dir, binding.name, length, ROLE_HELD_OUT)
        ],
    )
    experiment.promotion_instances = [
        instance
        for length in binding.lengths
        for instance in _read_split(splits_dir, binding.name, length, ROLE_PROMOTION)
    ]
    print(
        f"heldin={len(experiment.splits.heldin)} "
        f"heldout={len(experiment.splits.heldout)} "
        f"promotion={len(experiment.promotion_instances)}",
        flush=True,
    )

    round_path = experiment_round_dir(OUT_DIR, ROUND_INDEX)
    envelope_path = (
        round_path
        / "validation"
        / f"round_{ROUND_INDEX:02d}"
        / "merged"
        / "heldout"
        / "round_00"
        / "harness.json"
    )
    work_dir = round_path / "work"
    branch_base = rematerialize_harness_envelope(
        envelope_path,
        work_dir,
        module_prefix="_refine_branch_base_",
        error=ExperimentPersistenceError,
    )
    incumbent = HARNESSES[config.loop.initial_harness]
    print(f"branch_base rematerialized, incumbent={config.loop.initial_harness}", flush=True)

    validation = experiment._refine(round_path, ROUND_INDEX, incumbent, branch_base)
    print(f"refinement validation promoted={validation.promoted}", flush=True)
    if validation.ledger is not None:
        print(json.dumps(validation.ledger.decision, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
