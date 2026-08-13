"""One-shot, hash-pinned split materialization (R11; R9 GraphWalks half).

``materialize_splits`` draws one seeded sample per environment per length and
partitions it disjointly by slicing in order: the source-short sample splits
into held-in / held-out / test (24/40/40 at the shipped config), while
source-long and both target lengths are test-only sets. Each subset is written
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
loader is ``loader(config, length, limit, seed) -> instances``. The GraphWalks
loader is registered in ``DEFAULT_LOADERS``; the OOLONG-Pairs environment (a
parallel unit) wires in by adding its own one-line entry. An environment is
only materialized when its loader is present, and a manifest grows
per-environment, so registering a second environment later extends the splits
directory without touching the frozen GraphWalks files.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shrlm.environments.graphwalks import load_graphwalks
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

# loader(config, length, limit, seed) -> exactly ``limit`` instance dicts.
LoaderFn = Callable[[ExperimentConfig, str, int, int], list[dict[str, Any]]]


class SplitsPersistenceError(ExperimentError):
    """Persisted splits contradict the configuration or were modified on disk."""


def load_graphwalks_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """The GraphWalks loader wiring: config -> ``load_graphwalks`` arguments.

    Short reads the 128k-and-shorter file under the configured
    [min_chars, max_chars] window; long reads the native 256k-1M file (KTD5)
    under the same floor with no upper cap -- the file's own range is the cap.
    Both are pinned to the configured dataset revision.
    """
    env = config.environments.graphwalks
    if length not in LENGTHS:
        raise ValueError(f"unknown split length {length!r}; expected one of {LENGTHS}")
    dataset_file = env.dataset_file_short if length == "short" else env.dataset_file_long
    max_chars = env.max_chars if length == "short" else None
    return load_graphwalks(
        problem_types=env.problem_types,
        max_chars=max_chars,
        limit=limit,
        seed=seed,
        dataset_repo=env.dataset_repo,
        dataset_file=dataset_file,
        min_chars=env.min_chars,
        revision=env.dataset_revision,
    )


def load_oolong_pairs_split(
    config: ExperimentConfig, length: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """The OOLONG-Pairs loader wiring: config -> ``load_oolong_pairs`` arguments."""
    env = config.environments.oolong_pairs
    if length not in LENGTHS:
        raise ValueError(f"unknown split length {length!r}; expected one of {LENGTHS}")
    context_length = env.context_length_short if length == "short" else env.context_length_long
    return load_oolong_pairs_from_config(env, n=limit, seed=seed, context_lengths=(context_length,))


DEFAULT_LOADERS: dict[str, LoaderFn] = {
    "graphwalks": load_graphwalks_split,
    "oolong_pairs": load_oolong_pairs_split,
}


def split_plan(config: ExperimentConfig) -> dict[str, dict[str, dict[str, int]]]:
    """``{environment: {length: {role: size}}}``, entirely config-derived.

    Role order within a length is the partition order: the seeded sample is
    sliced sequentially, so held-in comes first, then held-out, then test.
    """
    splits = config.splits
    oolong = config.environments.oolong_pairs
    return {
        "graphwalks": {
            "short": {"held_in": splits.n_in, "held_out": splits.n_ho, "test": splits.test_short},
            "long": {"test": splits.test_long},
        },
        "oolong_pairs": {
            "short": {"test": oolong.n_short},
            "long": {"test": oolong.n_long},
        },
    }


def split_file_name(environment: str, length: str, role: str) -> str:
    return f"{environment}_{length}_{role}.jsonl"


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def dataset_revision_for(config: ExperimentConfig, environment: str) -> str:
    """The configured dataset revision pin for one environment."""
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
