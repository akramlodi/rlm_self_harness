"""Long OOLONG-synth instances the live recursion test runs against.

Selection is deterministic without a network round trip at import time: the
loader is seeded and its stream order is pinned by the configured dataset
revision, so ``load_oolong_recursion_instances(config)`` always returns the same
long-context instances for the same config. These are the lengths where
aggregation genuinely breaks down (published models score <50% even at 128K),
so an ``H0*`` root has the strongest reason to decompose.
"""

from typing import Any

from shrlm.environments.oolong import load_oolong_synth
from shrlm.experiment.config import ExperimentConfig

# Only the >= 64K context lengths: the regime where a single in-REPL pass over
# the whole context is implausible.
LONG_CONTEXT_LENGTHS: tuple[int, ...] = (65536, 131072)


def load_oolong_recursion_instances(
    config: ExperimentConfig, n: int = 4
) -> list[dict[str, Any]]:
    """Load ``n`` long OOLONG-synth instances from the configured synth split.

    Uses only the context lengths in ``LONG_CONTEXT_LENGTHS`` that the config's
    ``context_lengths`` also lists, so the pool is genuinely long. Raises if the
    config declares no long length or the split cannot supply ``n``.
    """
    synth = config.environments.oolong.synth
    lengths = tuple(
        length for length in LONG_CONTEXT_LENGTHS if length in tuple(synth.context_lengths)
    )
    if not lengths:
        raise LookupError(
            f"config context_lengths {tuple(synth.context_lengths)} contains none of "
            f"{LONG_CONTEXT_LENGTHS}; the recursion smoke test needs long instances"
        )
    return load_oolong_synth(
        context_lengths=lengths,
        task_groups=tuple(synth.task_groups),
        subsets=tuple(synth.subsets),
        limit=n,
        seed=config.splits.seed,
        split=synth.split,
        revision=synth.dataset_revision,
        max_scan=synth.max_scan,
    )
