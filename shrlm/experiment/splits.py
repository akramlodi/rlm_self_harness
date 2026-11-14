"""One-shot, hash-pinned split materialization (R11).

``materialize_splits`` draws one seeded sample per environment per length and
partitions it disjointly by slicing in order: each optimization length's
sample splits into held-in / held-out / test, while evaluation-only lengths
(e.g. OOLONG-Pairs 262k) are test-only sets. Each subset is written
as ``splits/<env>_<length>_<role>.jsonl`` -- one canonically serialized
instance per line, the same rendering ``instances.jsonl`` uses in
``shrlm.optimization.driver`` -- next to a ``manifest.json`` recording the
sample seed and, per environment, the pinned dataset revision and each file's
sha256 and instance count.

Splits are materialized exactly once. Re-invocation over an existing manifest
verifies -- seed unchanged, configured revision equal to the recorded one
(the dataset-drift guard: upstream HF edits must never silently change a
frozen split), every file byte-identical to its recorded sha256 -- and is
otherwise a no-op; any mismatch raises ``SplitsPersistenceError``. An
environment interrupted *mid*-materialization (files on disk, no manifest
entry yet) is redrawn on the next invocation and its files reused when the
fresh render reproduces them byte for byte (``write_split_file``), so a crash
between the file writes and the manifest write never wedges the directory.
Downstream consumers (orchestrator, eval runner) read only the persisted files,
so every condition sees byte-identical splits (R8).

Loaders are an injectable mapping ``{environment_name: loader_fn}`` where a
loader is ``loader(config, length, limit, seed) -> instances``; each
environment registers one entry in ``DEFAULT_LOADERS``. An environment is
only materialized when its loader is present, and a manifest grows
per-environment, so registering another environment later extends the splits
directory without touching already-frozen files.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from random import Random
from typing import Any

from shrlm.environments.obliq_bench_math import (
    _load_queries,
    load_obliq_bench_math,
    load_obliq_bench_math_from_config,
)
from shrlm.environments.oolong import (
    load_oolong_real_from_config,
    load_oolong_synth_from_config,
)
from shrlm.environments.oolong_pairs import load_oolong_pairs_from_config
from shrlm.experiment.config import ExperimentConfig
from shrlm.experiment.errors import ExperimentError

# ``instance_lines`` is the driver's own canonical renderer, imported rather
# than restated: the driver byte-compares a resumed round's ``instances.jsonl``
# against these exact bytes, so the two renderings must never drift apart.
from shrlm.optimization.driver import instance_lines, sha256_file

SPLITS_DIR = "splits"
MANIFEST_FILE = "manifest.json"

LENGTHS: tuple[str, ...] = ("short", "long")

# OOLONG-Pairs' three blended optimization lengths (held_in/held_out/test),
# in mining/validation order; "long" is the separate, test-only length and is
# never part of this tuple -- see ``split_plan``'s oolong_pairs branch.
OOLONG_PAIRS_OPTIMIZATION_LENGTHS: tuple[str, ...] = ("short", "mid16k", "mid32k")

# OOLONG-Pairs' full length vocabulary for ``load_oolong_pairs_split``'s
# validity check: the three blended optimization lengths plus the separate
# test-only "long". Deliberately NOT the shared ``LENGTHS`` -- that constant
# is also the fixed short/long reporting-bucket vocabulary every other
# environment's cost/scenario reporting (``report.py``, ``scenarios.py``,
# ``evaluation.py``) assumes is exactly two long, so widening it here would
# have silently broken those unrelated environments' reporting.
OOLONG_PAIRS_LENGTHS: tuple[str, ...] = (*OOLONG_PAIRS_OPTIMIZATION_LENGTHS, "long")

# OBLIQ-Bench Math's three blended optimization pool sizes (held_in/held_out/
# test), mirroring OOLONG_PAIRS_OPTIMIZATION_LENGTHS: each pool size draws its
# own disjoint set of queries (see load_obliq_bench_math_split's docstring),
# combined into one mining/validation pool. Only bound when
# ``environments.obliq_bench_math.blend_pool_sizes`` is true -- the default
# single-pool-size shape keeps using the plain "short" length unchanged.
OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES: tuple[str, ...] = ("pool200", "pool500", "pool1000")
OBLIQ_BENCH_MATH_POOL_SIZE_VALUES: dict[str, int] = {
    "pool200": 200,
    "pool500": 500,
    "pool1000": 1000,
}

# The 40-query frozen OBLIQ-Bench Math test split already used for the
# shrlm/H0/H0*/lambda-RLM full-corpus comparison
# (scratch/run_obliq_fullcorpus_testsplit*.py). The blended config's own
# "test" role draws from this same set (see load_obliq_bench_math_split), so
# this optimization loop's held-out final evaluation overlaps with queries
# that already have those four methods' baselines measured.
OBLIQ_BENCH_MATH_FROZEN_TEST_SPLIT_PATH = Path(
    "experiment_obliq_bench/calib_stage1/splits/obliq_bench_math_short_test.jsonl"
)

# loader(config, length, limit, seed) -> exactly ``limit`` instance dicts.
LoaderFn = Callable[[ExperimentConfig, str, int, int], list[dict[str, Any]]]

# OOLONG-Pairs' held_in/held_out/test roles are assigned by slicing one
# seeded draw in order (``materialize_environment``): instances[0:n_in] ->
# held_in, instances[n_in:n_in+n_ho] -> held_out, the rest -> test. With
# seed=0 that slicing concentrated the three hardest task_ids in the whole
# 20-task pool (19, 11, 18 -- individually F1 ~0.22-0.73 across mining and
# validation rounds 1-2) into held_in, while held_out drew only the easiest
# (4, 6, 8, 10 -- individually F1 ~0.80-0.94) -- a seed-luck difficulty
# confound, not a deliberate design: round 2's baseline scored macro F1
# 0.706 on held-in mining versus 0.857 on held-out validation, the SAME
# frozen H0 harness, purely from which task_ids landed where. This order
# interleaves the known difficulty spread across held_in and held_out
# instead (task_ids with no measurement yet are appended in task_id order,
# unprioritized). Applied as a stable sort in ``load_oolong_pairs_split``
# after the loader's own seeded draw, so it is deterministic and
# seed-independent, not a second source of randomness.
OOLONG_PAIRS_CURATED_TASK_ORDER: tuple[int, ...] = (
    # held_in (8 slots, n_in=8): keeps some of the previously held-in hard
    # task_ids (12, 15, 17, 19) and mixes in easy ones (1, 3, 8) plus one
    # never-measured task_id (2).
    1,
    2,
    3,
    8,
    12,
    15,
    17,
    19,
    # held_out (8 slots, n_ho=8): keeps the previously held-out easy
    # task_ids (4, 6, 10) and now also draws hard ones (11, 18) plus three
    # never-measured task_ids (5, 7, 9) -- held_out is no longer an
    # easy-only population.
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    18,
    # test (4 slots, test_short=4): never-measured task_ids, untouched by
    # promotion decisions either way.
    13,
    14,
    16,
    20,
)


# The screening-tournament probe (shrlm.experiment.orchestrator): a small,
# fixed, difficulty-informed subset of held_out used to cheaply rank several
# independently-proposed candidates before committing full-validation spend
# to just the best one. Derived from real pooled held_out trace data (two
# real runs' baseline evaluations, terminal failures correctly counted as
# zero rather than excluded): task_id 11 is unambiguously the hardest held_out
# task_id at every length (macro F1 0.517 vs. the next-worst 0.684); task_id 6
# is the easiest (macro F1 0.909). Mixes hard+easy per length rather than
# always the same extreme, so the screen isn't dominated by one idiosyncratic
# task_id's noise; weighted toward the harder end in general since an easy
# instance where the baseline already scores ~0.95+ has little room to show a
# candidate is better or worse.
OOLONG_PAIRS_SCREENING_TASK_IDS: dict[int, tuple[int, ...]] = {
    8192: (11,),  # hardest held_out task_id alone
    16384: (11, 6),  # hard + easy
    32768: (11, 18, 6),  # two hard (11, 18), one easy (6)
}


def oolong_pairs_screening_subset(heldout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The curated screening probe: held_out instances matching (context_len, task_id)
    in ``OOLONG_PAIRS_SCREENING_TASK_IDS``. Real oolong_pairs instances always carry
    both fields (``shrlm/environments/oolong_pairs.py``); a held_out population without
    them -- a test fixture standing in for oolong_pairs with a generic loader, say --
    has no curated difficulty signal to use, so this falls back to
    ``default_screening_subset`` instead of crashing.

    A held_out population smaller than the full profile's own (fewer n_in/n_ho
    than the 8/8 the curated task-id blocks were sized for) can carry
    context_len/task_id yet still match nothing -- the curated block's task_ids
    simply never land in a smaller held_out slice. When that happens but
    context_len IS present on every instance, prefer one instance per bound
    length over the fully-generic first-n_screen default, so screening still
    samples across the blend rather than collapsing onto whichever length
    happens to sort first.
    """
    wanted = {
        (length, task_id)
        for length, task_ids in OOLONG_PAIRS_SCREENING_TASK_IDS.items()
        for task_id in task_ids
    }
    matched = [
        instance
        for instance in heldout
        if "context_len" in instance
        and "task_id" in instance
        and (int(instance["context_len"]), int(instance["task_id"])) in wanted
    ]
    if matched:
        return matched
    if heldout and all("context_len" in instance for instance in heldout):
        seen_lengths: set[int] = set()
        one_per_length: list[dict[str, Any]] = []
        for instance in heldout:
            length = int(instance["context_len"])
            if length not in seen_lengths:
                seen_lengths.add(length)
                one_per_length.append(instance)
        return one_per_length
    return default_screening_subset(heldout)


def default_screening_subset(
    heldout: list[dict[str, Any]], n_screen: int = 3
) -> list[dict[str, Any]]:
    """Generic screening-probe fallback for environments without a curated
    difficulty constant: the first ``n_screen`` held_out instances, in
    materialized order. Not difficulty-informed -- a reasonable default, not
    validated against real trace data the way ``oolong_pairs_screening_subset``
    is.
    """
    return heldout[:n_screen]


# The OBLIQ-Bench Math screening-tournament probe, mirroring
# ``OOLONG_PAIRS_SCREENING_TASK_IDS``: 2 held_out queries per pool size (6
# total across pool200/500/1000) -- the hardest-scoring query and the
# median-scoring query, not an even or rank-weighted split. Derived from
# round 1's real baseline validation data (experiment_real_runs/
# experiment_obliq_bench_math_openrouter_dsv4f_blended_h0init/opt/round_01/
# validation/round_01/baseline/heldout, 8 held_out queries x 3 attempts per
# pool size, 2026-11-12): per-query mean NDCG@10, terminal failures counted
# as zero. Within each pool size the 8 queries were ranked by that mean and
# the lowest-scoring (hardest) plus the index-4-of-8 (lower-median) query
# selected -- ties at the hardest end (two pool1000 queries both scored
# 0.0) broken by the loader's own id order.
OBLIQ_BENCH_MATH_SCREENING_QUERY_IDS: dict[int, tuple[str, ...]] = {
    200: ("q03180__pool200", "q00604__pool200"),  # hardest 0.0000, median 0.3333
    500: ("q00880__pool500", "q02765__pool500"),  # hardest 0.0000, median 0.2600
    1000: ("q01210__pool1000", "q00740__pool1000"),  # hardest 0.0000, median 0.3697
}


def obliq_bench_math_screening_subset(
    heldout: list[dict[str, Any]], n_screen: int = 6
) -> list[dict[str, Any]]:
    """The screening probe for OBLIQ-Bench Math: held_out instances matching
    an id in ``OBLIQ_BENCH_MATH_SCREENING_QUERY_IDS`` -- one hardest and one
    median query per pool size, 6 total at the shipped 3-pool-size shape.
    Falls back to ``default_screening_subset`` when none of ``heldout``'s
    ids match the curated set (a held_out population drawn under a
    different seed/corpus, or the single-pool-size shape, has no curated
    difficulty signal to use), exactly as ``oolong_pairs_screening_subset``
    falls back for OOLONG-Pairs.
    """
    wanted = {qid for ids in OBLIQ_BENCH_MATH_SCREENING_QUERY_IDS.values() for qid in ids}
    matched = [instance for instance in heldout if instance.get("id") in wanted]
    if matched:
        return matched
    return default_screening_subset(heldout, n_screen=n_screen)


class SplitsPersistenceError(ExperimentError):
    """Persisted splits contradict the configuration or were modified on disk."""


def load_oolong_pairs_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """The OOLONG-Pairs loader wiring: config -> ``load_oolong_pairs`` arguments.

    Four named lengths, not two: ``short``/``mid16k``/``mid32k`` are the
    blended optimization lengths (8192/16384/32768 at the shipped config),
    ``long`` is the separate 262144-token length reserved for final
    evaluation only.

    The loader's own seeded draw returns every requested task_id (when
    ``limit`` equals the full configured task_id pool, as it does for the
    blended optimization lengths) in a seed-shuffled order; this reorders
    that result by ``OOLONG_PAIRS_CURATED_TASK_ORDER`` (a stable sort, ties
    broken by the loader's own order) before ``materialize_environment``
    slices it into held_in/held_out/test, so which task_ids land in which
    role is a deliberate, known-difficulty-mixed choice instead of whatever
    the seed's shuffle happened to produce.
    """
    env = config.environments.oolong_pairs
    if length not in OOLONG_PAIRS_LENGTHS:
        raise ValueError(f"unknown split length {length!r}; expected one of {OOLONG_PAIRS_LENGTHS}")
    context_length = {
        "short": env.context_length_short,
        "mid16k": env.context_length_mid16k,
        "mid32k": env.context_length_mid32k,
        "long": env.context_length_long,
    }[length]
    instances = load_oolong_pairs_from_config(
        env, n=limit, seed=seed, context_lengths=(context_length,)
    )
    priority = {task_id: index for index, task_id in enumerate(OOLONG_PAIRS_CURATED_TASK_ORDER)}
    default = len(OOLONG_PAIRS_CURATED_TASK_ORDER)
    return sorted(instances, key=lambda instance: priority.get(instance["task_id"], default))


def load_oolong_synth_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """OOLONG-synth loader wiring. Only the ``short`` length is used: the
    held_in / held_out / test partition is one length-diverse pool (the loader
    draws a balanced share per ``context_lengths`` bucket), so there is no
    separate long split to route."""
    if length != "short":
        raise ValueError(f"oolong_synth has only a 'short' pool; got length {length!r}")
    return load_oolong_synth_from_config(config, n=limit, seed=seed)


def load_oolong_real_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """OOLONG-real loader wiring for the periodic generalization check set."""
    if length != "short":
        raise ValueError(f"oolong_real has only a 'short' check set; got length {length!r}")
    return load_oolong_real_from_config(config, n=limit, seed=seed)


def load_obliq_bench_math_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """OBLIQ-Bench Math loader wiring.

    ``"short"`` (the default, single-pool-size shape): held_in / held_out /
    test is one seeded query-id pool over the dataset's 151 queries, at
    ``environments.obliq_bench_math.candidate_pool_size`` -- there is no
    context-length axis to split on.

    ``"pool200"``/``"pool500"``/``"pool1000"`` (the blended shape, only bound
    when ``blend_pool_sizes`` is true): each pool size draws a DISJOINT set of
    queries from the other two, not the same queries at three difficulties.
    Two disjoint pools feed this, by role:

    - ``test`` (the last ``splits.test_short`` queries in role order -- see
      ``OBLIQ_BENCH_MATH_FROZEN_TEST_SPLIT_PATH``'s docstring): drawn from the
      40-query frozen test split already used for the earlier shrlm/H0/H0*/
      lambda-RLM full-corpus comparison (``scratch/run_obliq_fullcorpus_testsplit*.py``),
      so this optimization loop's own held-out final evaluation overlaps with
      queries that already have those baselines measured, instead of a fresh
      eval set needing all four reference runs repeated.
    - ``held_in``/``held_out`` (everything else): a full, seeded shuffle of
      every OTHER query in the corpus (the frozen 40 excluded entirely, so no
      query ever serves double duty as both a mining/validation instance and
      a reserved test instance).

    Both pools are sliced the same way: one seeded shuffle, computed
    identically by every independent per-length call (deterministic from
    ``seed`` alone, no shared mutable state needed), then each length takes
    its own position's slice from ``OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES``
    (pool200 gets ``shuffled[0:n]``, pool500 ``shuffled[n:2n]``, pool1000
    ``shuffled[2n:3n]``) -- the same disjoint-slicing idea
    ``materialize_environment`` already uses for held_in/held_out/test within
    one length, just applied across lengths too. Raises if either pool's
    three-length combined need exceeds its available query count. A round's
    instance list must have unique ids regardless (``_validate_config``
    enforces this), so each instance's ``id`` here is still suffixed with its
    pool size (``f"{query_id}__pool{size}"``) for safety even though the
    underlying query ids are now already disjoint across lengths --
    verification only ever reads ``gold_relevant_ids``/``excluded_ids`` off
    the instance dict directly, never re-looks-up by id, so the suffix is
    inert everywhere except the id itself.
    """
    if length == "short":
        return load_obliq_bench_math_from_config(config, n=limit, seed=seed)
    if length in OBLIQ_BENCH_MATH_POOL_SIZE_VALUES:
        env = config.environments.obliq_bench_math
        assert env is not None
        pool_size = OBLIQ_BENCH_MATH_POOL_SIZE_VALUES[length]
        position = OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES.index(length)
        n_lengths = len(OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES)

        test_short = config.splits.test_short
        non_test_limit = limit - test_short

        frozen_test_ids = sorted(
            {
                json.loads(line)["id"]
                for line in OBLIQ_BENCH_MATH_FROZEN_TEST_SPLIT_PATH.read_text().splitlines()
            }
        )
        needed_test = test_short * n_lengths
        if needed_test > len(frozen_test_ids):
            raise ValueError(
                f"blended obliq_bench_math test role needs {needed_test} disjoint queries "
                f"({test_short} x {n_lengths} pool sizes) from the frozen 40-query test "
                f"split, but it only has {len(frozen_test_ids)}"
            )
        shuffled_test = frozen_test_ids.copy()
        Random(seed).shuffle(shuffled_test)
        test_ids = shuffled_test[position * test_short : (position + 1) * test_short]

        all_query_ids = sorted(_load_queries(env.dataset_revision))
        non_test_pool = [q for q in all_query_ids if q not in set(frozen_test_ids)]
        needed_non_test = non_test_limit * n_lengths
        if needed_non_test > len(non_test_pool):
            raise ValueError(
                f"blended obliq_bench_math held_in/held_out needs {needed_non_test} "
                f"disjoint queries ({non_test_limit} x {n_lengths} pool sizes) outside the "
                f"frozen test split, but only {len(non_test_pool)} remain"
            )
        shuffled_non_test = non_test_pool.copy()
        Random(seed).shuffle(shuffled_non_test)
        non_test_ids = shuffled_non_test[position * non_test_limit : (position + 1) * non_test_limit]

        # Role order (materialize_environment slices positionally): held_in,
        # held_out, [promotion], test -- test is always last, so the frozen
        # test ids must come last here too.
        query_ids = non_test_ids + test_ids
        instances = load_obliq_bench_math(
            seed=seed,
            query_ids=query_ids,
            candidate_pool_size=pool_size,
            revision=env.dataset_revision,
        )
        # "context_len" (not "pool_size") is the generic stratification key
        # ``shrlm.optimization.validation`` and ``shrlm.optimization.promotion``
        # already read for every environment's per-length primary_quality
        # breakdown and per-length tau_regression lookup (see
        # load_oolong_pairs_split's own "context_len" field) -- tagging it
        # here with the pool size lets a ``[promotion.tau_regression]``
        # sub-table keyed "200"/"500"/"1000" work for OBLIQ-Bench Math with
        # no changes to that shared machinery.
        return [
            {**instance, "id": f"{instance['id']}__pool{pool_size}", "context_len": pool_size}
            for instance in instances
        ]
    raise ValueError(
        f"unknown obliq_bench_math split length {length!r}; expected 'short' or one of "
        f"{OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES}"
    )


DEFAULT_LOADERS: dict[str, LoaderFn] = {
    "oolong_pairs": load_oolong_pairs_split,
    "oolong_synth": load_oolong_synth_split,
    "oolong_real": load_oolong_real_split,
    "obliq_bench_math": load_obliq_bench_math_split,
}


def split_plan(config: ExperimentConfig) -> dict[str, dict[str, dict[str, int]]]:
    """``{environment: {length: {role: size}}}``, entirely config-derived.

    Role order within a length is the partition order: the seeded sample is
    sliced sequentially, so held-in comes first, then held-out, then test.

    The plan holds only the environments a run actually uses:
    ``loop.environment == "oolong_pairs"`` partitions its finite short pool for optimization
    and keeps its long pool test-only; ``"oolong_synth"`` materializes the
    OOLONG-synth pool and -- when
    ``operational.real_check_every_n_rounds > 0`` -- the OOLONG-real check set,
    and nothing else; ``"obliq_bench_math"`` materializes one seeded query-id
    pool (no length axis, same single-pool shape as ``oolong_synth``) and
    nothing else. ``materialize_splits`` skips any registered loader absent
    from the plan, so a run never touches a dataset it does not use.
    """
    splits = config.splits
    # Dormant: still materialized whenever refinement is enabled, but
    # refinement validation now reuses `held_out` instead of this split (see
    # RefinementSettings' docstring and Orchestrator._validate_refinement).
    # Kept rather than deleted as a paper trail for that earlier design.
    promotion_roles = (
        {"promotion": config.refinement.promotion_size} if config.refinement.enabled else {}
    )
    if config.loop.environment == "oolong_synth":
        plan: dict[str, dict[str, dict[str, int]]] = {
            "oolong_synth": {
                "short": {
                    "held_in": splits.n_in,
                    "held_out": splits.n_ho,
                    **promotion_roles,
                    "test": splits.test_short,
                },
            },
        }
        if config.operational.real_check_every_n_rounds > 0:
            plan["oolong_real"] = {
                "short": {"check": config.environments.oolong.real.n_check},
            }
        return plan
    if config.loop.environment == "oolong_pairs":
        oolong_pairs = config.environments.oolong_pairs
        # Blended mining/validation: the same n_in/n_ho/test_short roles at
        # each of the three optimization lengths (8192/16384/32768 at the
        # shipped config), not one fixed length -- held_in and held_out for
        # mining/promotion draw from all three, combined. ``long`` (262144)
        # stays the separate, test-only length: never mined or validated,
        # only reserved for final-harness evaluation.
        optimization_total = (
            splits.n_in
            + splits.n_ho
            + splits.test_short
            + (config.refinement.promotion_size if config.refinement.enabled else 0)
        )
        if optimization_total > oolong_pairs.n_short:
            raise ValueError(
                "oolong_pairs optimization roles require "
                f"{optimization_total} instances at each blended length, but "
                "environments.oolong_pairs.n_short limits the pinned pool "
                f"(at every length) to {oolong_pairs.n_short}"
            )
        if splits.test_long > oolong_pairs.n_long:
            raise ValueError(
                "oolong_pairs long test requires "
                f"{splits.test_long} instances, but environments.oolong_pairs.n_long "
                f"limits the pinned pool to {oolong_pairs.n_long}"
            )
        optimization_roles = {
            "held_in": splits.n_in,
            "held_out": splits.n_ho,
            **promotion_roles,
            "test": splits.test_short,
        }
        return {
            "oolong_pairs": {
                **{
                    length: dict(optimization_roles) for length in OOLONG_PAIRS_OPTIMIZATION_LENGTHS
                },
                "long": {"test": splits.test_long},
            }
        }
    if config.loop.environment == "obliq_bench_math":
        obliq = config.environments.obliq_bench_math
        assert obliq is not None
        optimization_roles = {
            "held_in": splits.n_in,
            "held_out": splits.n_ho,
            **promotion_roles,
            "test": splits.test_short,
        }
        if obliq.blend_pool_sizes:
            # Blended mining/validation: the same held_in/held_out/test query
            # ids, evaluated at each of 200/500/1000 candidate-pool
            # difficulty -- see OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES and
            # load_obliq_bench_math_split's docstring. Mirrors oolong_pairs'
            # blended-length branch above exactly; there is no separate
            # test-only tier the way oolong_pairs has "long".
            return {
                "obliq_bench_math": {
                    length: dict(optimization_roles)
                    for length in OBLIQ_BENCH_MATH_OPTIMIZATION_POOL_SIZES
                }
            }
        return {
            "obliq_bench_math": {
                "short": optimization_roles,
            },
        }
    raise ValueError(f"unsupported loop.environment {config.loop.environment!r} in split_plan")


def split_file_name(environment: str, length: str, role: str) -> str:
    return f"{environment}_{length}_{role}.jsonl"


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def dataset_revision_for(config: ExperimentConfig, environment: str) -> str:
    """The configured dataset revision pin for one environment.

    ``oolong_synth`` / ``oolong_real`` are sub-tables of ``environments.oolong``
    rather than top-level attributes, so they are resolved explicitly.
    """
    if environment == "oolong_synth":
        return str(config.environments.oolong.synth.dataset_revision)
    if environment == "oolong_real":
        return str(config.environments.oolong.real.dataset_revision)
    if not hasattr(config.environments, environment):
        raise ValueError(
            f"unknown environment {environment!r} in loaders; the config defines "
            "no such environment table"
        )
    env: Any = getattr(config.environments, environment)
    return str(env.dataset_revision)


def write_split_file(path: Path, content: str, environment: str) -> None:
    """Persist one split file, reusing an interrupted attempt's identical bytes.

    Split files are written before the manifest entry that accounts for them,
    so a crash (or a bounded-fetch timeout) between the two leaves unaccounted
    files on disk. Loaders are seeded and deterministic, so the retry's draw
    renders byte-identical content: a file that matches those bytes IS the
    interrupted attempt's own output and is reused, which is what keeps an
    interrupted materialization from wedging the splits directory forever.
    Different bytes are a genuine divergence -- a file from another
    configuration, or one edited on disk -- and are refused.
    """
    if not path.exists():
        path.write_text(content)
        return
    if path.read_text() == content:
        return
    raise SplitsPersistenceError(
        f"{path} exists with bytes a fresh draw does not reproduce, and the manifest has "
        f"no entry for {environment!r}; an interrupted materialization would have left "
        "byte-identical content (the loaders are seeded and deterministic), so this file "
        "came from a different configuration or was modified. Refusing to overwrite it."
    )


def _task_id_of(instance: dict[str, Any]) -> int:
    return int(instance["task_id"])


def _can_draw_oolong_pairs_with_promotion_safely(
    config: ExperimentConfig, roles: dict[str, int]
) -> bool:
    """Whether ``_draw_oolong_pairs_with_promotion``'s invariant holds here.

    It requires held_in+held_out+test to total exactly ``len(task_ids)`` --
    true for the shipped blended profile (8+8+4=20), not for every profile
    (e.g. the single-length profile's held_in alone is 20, a full window by
    itself; held_out+test+promotion share the second window instead). Only
    the proven-safe shape gets the corrected draw; anything else falls back
    to the original behavior unchanged rather than guessing at a shape this
    fix was not built and verified for.

    Still exercised whenever refinement is enabled (the promotion role is
    still materialized -- see ``split_plan``), even though
    ``Orchestrator._validate_refinement`` no longer reads the resulting
    ``promotion`` split: this function fixes a real materialization bug for
    *any* role beyond held_in/held_out/test, not something specific to
    refinement's now-dormant use of it.
    """
    if "promotion" not in roles:
        return False
    env = config.environments.oolong_pairs
    primary_total = sum(size for role, size in roles.items() if role != "promotion")
    return primary_total == len(env.task_ids)


def _draw_oolong_pairs_with_promotion(
    config: ExperimentConfig,
    length: str,
    roles: dict[str, int],
    loader: LoaderFn,
    seed: int,
) -> list[dict[str, Any]]:
    """Draw one oolong_pairs length's roles when ``promotion`` is among them.

    ``load_oolong_pairs`` requests ``ceil(n / len(task_ids))`` windows and then
    shuffles-and-truncates the resulting (window, task_id) rows to exactly
    ``n``. When ``n`` equals ``len(task_ids)`` this always returns one
    instance per task_id (one window fully covers it, nothing to truncate).
    The moment ``n`` exceeds ``len(task_ids)`` -- exactly what happens once a
    ``promotion`` role is added on top of held_in/held_out/test -- a second
    window is pulled in and the truncation becomes a random subset of 2x the
    task_ids: some task_ids end up with two instances, others with zero. The
    positional, curated-order slice that assigns roles by task_id then
    misassigns every task_id after the first duplicate, and a task_id can
    vanish from every role entirely.

    Fixed by drawing in two deterministic, non-colliding passes instead of
    one: the primary pass requests exactly ``len(task_ids)`` instances (one
    window, zero duplication by construction) for held_in/held_out/test; the
    promotion instances are drawn from a separate, full-pool call and
    filtered to the complementary-window counterparts of held_out's task_ids
    -- the same disjoint, difficulty-matched construction used by hand for
    this fork's first promotion split.
    """
    env = config.environments.oolong_pairs
    n_tasks = len(env.task_ids)
    primary_roles = {role: size for role, size in roles.items() if role != "promotion"}
    promotion_size = roles["promotion"]
    primary_total = sum(primary_roles.values())
    assert primary_total == n_tasks, (
        "caller must only invoke this helper when held_in+held_out+test totals exactly "
        "n_tasks -- see can_draw_oolong_pairs_with_promotion_safely"
    )

    primary = loader(config, length, primary_total, seed)
    priority = {task_id: index for index, task_id in enumerate(OOLONG_PAIRS_CURATED_TASK_ORDER)}
    primary = sorted(primary, key=lambda instance: priority[_task_id_of(instance)])
    primary_task_ids = [_task_id_of(instance) for instance in primary]
    if sorted(primary_task_ids) != sorted(env.task_ids):
        raise ValueError(
            f"oolong_pairs {length} primary draw did not cover every task_id exactly "
            f"once (got {sorted(primary_task_ids)}, expected {sorted(env.task_ids)}); "
            "this should be structurally impossible when primary_total == n_tasks"
        )
    primary_ids = {str(instance["id"]) for instance in primary}

    held_out_start = primary_roles.get("held_in", 0)
    held_out_size = primary_roles.get("held_out", 0)
    held_out_task_ids = primary_task_ids[held_out_start : held_out_start + held_out_size]

    full_pool = loader(config, length, n_tasks * 2, seed)
    complementary = [
        instance for instance in full_pool if str(instance["id"]) not in primary_ids
    ]
    by_task_id: dict[int, dict[str, Any]] = {}
    for instance in complementary:
        by_task_id.setdefault(_task_id_of(instance), instance)
    missing = [task_id for task_id in held_out_task_ids if task_id not in by_task_id]
    if missing:
        raise ValueError(
            f"oolong_pairs {length} has no complementary-window instance for held_out "
            f"task_id(s) {missing}; cannot draw a disjoint promotion split from them"
        )
    promotion_pool = [by_task_id[task_id] for task_id in held_out_task_ids]
    if len(promotion_pool) < promotion_size:
        raise ValueError(
            f"oolong_pairs {length} promotion role needs {promotion_size} instances but "
            f"only {len(promotion_pool)} complementary-window held_out task_ids exist"
        )
    promotion = promotion_pool[:promotion_size]

    ordered: list[dict[str, Any]] = []
    cursor = 0
    for role, size in roles.items():
        if role == "promotion":
            ordered.extend(promotion)
        else:
            ordered.extend(primary[cursor : cursor + size])
            cursor += size
    return ordered


def materialize_environment(
    config: ExperimentConfig,
    splits_dir: Path,
    environment: str,
    loader: LoaderFn,
    plan: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, Any]:
    """Draw, partition, and persist one environment's splits; return its
    manifest entry (revision plus per-file sha256 and count)."""
    files: dict[str, dict[str, Any]] = {}
    for length, roles in plan.items():
        total = sum(roles.values())
        if environment == "oolong_pairs" and _can_draw_oolong_pairs_with_promotion_safely(
            config, roles
        ):
            instances = _draw_oolong_pairs_with_promotion(config, length, roles, loader, seed)
        else:
            instances = loader(config, length, total, seed)
        if len(instances) != total:
            raise ValueError(
                f"loader for {environment!r} returned {len(instances)} instances for the "
                f"{length} split, but the plan requires exactly {total}; a short pool "
                "cannot silently shrink a preregistered split"
            )
        ids = [str(instance["id"]) for instance in instances]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"duplicate instance ids in the {environment!r} {length} sample; "
                "roles are partitioned by slicing, so ids must be unique for the "
                "partition to be disjoint"
            )
        start = 0
        for role, size in roles.items():
            subset = instances[start : start + size]
            start += size
            file_name = split_file_name(environment, length, role)
            content = instance_lines(subset)
            write_split_file(splits_dir / file_name, content, environment)
            files[file_name] = {"sha256": sha256_text(content), "count": size}
    return {"revision": dataset_revision_for(config, environment), "files": files}


def verify_environment(
    splits_dir: Path, environment: str, recorded: dict[str, Any], revision: str
) -> None:
    """Check one persisted environment: revision pin and per-file sha256."""
    if recorded["revision"] != revision:
        raise SplitsPersistenceError(
            f"splits manifest records dataset revision {recorded['revision']!r} for "
            f"{environment!r}, but the config pins {revision!r}; a revision change "
            "would redefine frozen splits (dataset-drift guard)"
        )
    for file_name, entry in recorded["files"].items():
        path = splits_dir / file_name
        if not path.exists():
            raise SplitsPersistenceError(
                f"{path} is recorded in the splits manifest but missing on disk"
            )
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise SplitsPersistenceError(
                f"{path} sha256 {actual} does not match the manifest's {entry['sha256']}; "
                "split files are frozen once materialized"
            )


def materialize_splits(
    config: ExperimentConfig,
    out_dir: Path | str,
    loaders: dict[str, LoaderFn] | None = None,
) -> Path:
    """Materialize (or verify) split files under ``out_dir/splits``.

    For each environment in ``loaders`` (``DEFAULT_LOADERS`` when None): a
    manifest entry means the environment is already frozen -- verify revision
    and hashes, load nothing; no entry means draw and persist it now. Returns
    the splits directory.
    """
    if loaders is None:
        loaders = DEFAULT_LOADERS
    plan = split_plan(config)
    seed = config.splits.seed
    splits_dir = Path(out_dir) / SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = splits_dir / MANIFEST_FILE
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"sample_seed": seed, "environments": {}}
    )
    if manifest["sample_seed"] != seed:
        raise SplitsPersistenceError(
            f"splits manifest records sample seed {manifest['sample_seed']}, but the "
            f"config sets {seed}; a seed change would resample frozen splits"
        )

    for environment in sorted(loaders):
        if environment not in plan and environment in DEFAULT_LOADERS:
            # A recognized loader the selected ``loop.environment`` does not use
            # (e.g. oolong_real on an OOLONG-Pairs run). Skip it entirely -- its
            # dataset is never downloaded and no split file is written. An
            # unrecognized name still falls through to dataset_revision_for,
            # which raises "unknown environment".
            continue
        revision = dataset_revision_for(config, environment)
        recorded = manifest["environments"].get(environment)
        if recorded is not None:
            verify_environment(splits_dir, environment, recorded, revision)
            continue
        manifest["environments"][environment] = materialize_environment(
            config, splits_dir, environment, loaders[environment], plan[environment], seed
        )

    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.exists() or manifest_path.read_text() != content:
        manifest_path.write_text(content)
    return splits_dir
