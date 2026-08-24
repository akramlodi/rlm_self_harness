"""Persist-first execution for the pinned λ-RLM comparison baseline.

The optimization driver runs editable ``Harness`` objects and identifies them
with ``harness.json``. λ-RLM is a different inference method, not an RLM
harness surface, so this runner keeps its construction separate while sharing
the round's canonical instance, trace, and manifest formats.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm.core.types import ClientBackend
from shrlm.baselines.lambda_rlm import (
    LambdaBaselineConfig,
    lambda_input,
    lambda_method_envelope,
    write_lambda_method_json,
)
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN, round_dir
from shrlm.optimization.driver import (
    INSTANCES_FILE,
    TRACES_DIR,
    RoundPersistenceError,
    instance_lines,
    load_manifest,
    persist_run,
    run_id_for,
    verify_trace,
)
from shrlm.optimization.types import Verifier

METHOD_FILE = "method.json"

# These values intentionally match the core round driver's safety policy.
# Backend kwargs are copied into model traces, so credentials must stay in the
# environment rather than becoming persisted experiment artifacts.
SENSITIVE_KWARG_FRAGMENTS = ("key", "token", "secret", "password", "authorization")
BACKEND_ENV_KEYS: dict[str, str] = {"openrouter": "OPENROUTER_API_KEY"}


@dataclass(frozen=True)
class LambdaRoundConfig:
    """Everything needed to execute and persist one λ-RLM evaluation round."""

    round_index: int
    instances: list[dict[str, Any]]
    verifier: Verifier
    out_dir: Path | str
    method: LambdaBaselineConfig = field(default_factory=LambdaBaselineConfig)
    backend: ClientBackend = "openrouter"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1


def validate_lambda_round(config: LambdaRoundConfig) -> None:
    """Reject invalid input before constructing a client or making a model call."""
    if config.attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {config.attempts}")
    if not config.instances:
        raise ValueError("a λ-RLM round needs at least one instance")

    seen: set[str] = set()
    for instance in config.instances:
        instance_id = str(instance["id"])
        if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
            raise ValueError(
                f"instance id {instance_id!r} is not filesystem-safe; ids become trace "
                f"file names and must match {FILESYSTEM_SAFE_ID_PATTERN.pattern}"
            )
        if instance_id in seen:
            raise ValueError(
                f"duplicate instance id {instance_id!r}: run ids derive from "
                "(instance id, attempt); use attempts for repeated runs"
            )
        seen.add(instance_id)
        lambda_input(instance)

    for name in config.backend_kwargs:
        lowered = name.lower()
        if any(fragment in lowered for fragment in SENSITIVE_KWARG_FRAGMENTS):
            raise ValueError(
                f"backend_kwargs may not carry credential material ({name!r}): kwargs "
                "can enter persisted traces; supply credentials through the environment"
            )


def prepare_lambda_round(config: LambdaRoundConfig) -> Path:
    """Create or verify the round's method and instance identity artifacts."""
    path = round_dir(config.out_dir, config.round_index)
    path.mkdir(parents=True, exist_ok=True)
    (path / TRACES_DIR).mkdir(exist_ok=True)

    method_path = path / METHOD_FILE
    expected_method = lambda_method_envelope(config.method)
    if method_path.exists():
        recorded_method = json.loads(method_path.read_text())
        if recorded_method != expected_method:
            raise RoundPersistenceError(
                f"{method_path} does not match the configured λ-RLM method; refusing "
                "to mix two method configurations in one round"
            )
    else:
        write_lambda_method_json(config.method, method_path)

    instances_path = path / INSTANCES_FILE
    expected_instances = instance_lines(config.instances)
    if instances_path.exists():
        if instances_path.read_text() != expected_instances:
            raise RoundPersistenceError(
                f"{instances_path} does not match the configured instances; resuming "
                "requires the identical instance list, verbatim"
            )
    else:
        instances_path.write_text(expected_instances)

    return path


def require_lambda_backend_credential(config: LambdaRoundConfig) -> None:
    """Fail before a paid pending run when a known backend credential is absent."""
    env_key = BACKEND_ENV_KEYS.get(config.backend)
    if env_key is not None and not os.environ.get(env_key):
        raise RuntimeError(
            f"backend {config.backend!r} requires the {env_key} environment variable; "
            "refusing to start a paid λ-RLM round"
        )


def run_lambda_round(
    config: LambdaRoundConfig,
    *,
    stop_after: int | None = None,
) -> list[dict[str, Any]]:
    """Run missing λ-RLM attempts and persist each completion immediately.

    Reinvocation with the same configuration verifies every recorded trace and
    skips its run id. This makes a partially completed round resumable without
    paying for completed attempts again.
    """
    validate_lambda_round(config)
    path = prepare_lambda_round(config)

    existing = load_manifest(config.out_dir, config.round_index)
    for entry in existing:
        verify_trace(path, entry)
    done = {str(entry["run_id"]) for entry in existing}
    pending = [
        (instance, attempt)
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if run_id_for(str(instance["id"]), attempt) not in done
    ]

    entries = list(existing)
    if not pending or stop_after == 0:
        return entries
    require_lambda_backend_credential(config)

    executed = 0
    for instance, attempt in pending:
        if stop_after is not None and executed >= stop_after:
            break

        instance_id = str(instance["id"])
        run_id = run_id_for(instance_id, attempt)
        model_input = lambda_input(instance)
        method = config.method.build(
            backend=config.backend,
            backend_kwargs=dict(config.backend_kwargs),
            query=model_input.query,
        )
        completion = method.completion(model_input.prompt)
        verdict = config.verifier(instance, completion.response)
        entries.append(
            persist_run(
                path,
                run_id,
                instance_id,
                attempt,
                completion,
                verdict,
            )
        )
        executed += 1

    return entries


__all__ = [
    "LambdaRoundConfig",
    "METHOD_FILE",
    "prepare_lambda_round",
    "require_lambda_backend_credential",
    "run_lambda_round",
    "validate_lambda_round",
]
