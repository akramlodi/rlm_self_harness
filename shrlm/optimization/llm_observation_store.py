"""Immutable call observations, anchored to the artifact that references them."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from rlm.core.llm_observation import ObservationPersistenceError, ObservationRecorder


def observation_recorder(
    artifact: Path, owner: dict[str, Any], *, namespace: str | None = None
) -> ObservationRecorder:
    anchor = artifact.parent.resolve()
    prefix = Path(namespace or f"{artifact.stem}.llm_calls")
    if not (anchor / prefix).resolve().is_relative_to(anchor):
        raise ObservationPersistenceError("observation namespace is outside its artifact directory")

    def save(record: dict[str, Any]) -> dict[str, Any]:
        relative = prefix / record["call_id"] / f"{record['event']}-{uuid4().hex}.json"
        destination = anchor / relative
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=destination.parent, suffix=".partial", delete=False
            ) as file:
                staged = Path(file.name)
                file.write(payload)
            os.replace(staged, destination)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        reference = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "call_id": record["call_id"],
        }
        if "attempt_id" in record:
            reference["attempt_id"] = record["attempt_id"]
        return reference

    return ObservationRecorder(owner, save)


def read_observation(reference: dict[str, Any], anchor: Path) -> dict[str, Any]:
    """Read and verify one reference; never repair missing data by calling an LM."""
    try:
        relative = Path(reference["path"])
        path = (anchor / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(anchor.resolve()):
            raise ObservationPersistenceError("observation reference resolves outside its owner")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise ObservationPersistenceError(f"observation sha256 mismatch: {path}")
        record = json.loads(payload)
        for key in ("call_id", "attempt_id"):
            if key in reference and reference[key] != record.get(key):
                raise ObservationPersistenceError(f"observation {key} mismatch: {path}")
        return record
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ObservationPersistenceError(f"cannot read observation: {error}") from error


def verify_observations(payload: Any, anchor: Path) -> None:
    """Verify additive references in traces/attempts; legacy objects are untouched."""
    seen: set[tuple[str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for reference in value.get("llm_observations") or []:
                identity = (reference.get("path", ""), reference.get("sha256", ""))
                if identity not in seen:
                    read_observation(reference, anchor)
                    seen.add(identity)
            for key, child in value.items():
                # Inputs, generated text, and observation bodies are not trace containers.
                if key not in {"llm_observations", "prompt", "raw_response", "response", "content"}:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def rebase_observations(payload: Any, source: Path, destination: Path) -> Any:
    """Copy references into a containing artifact's coordinate system."""
    verify_observations(payload, source)
    prefix = source.resolve().relative_to(destination.resolve())

    def copy(value: Any) -> Any:
        if isinstance(value, dict):
            result = {key: copy(child) for key, child in value.items() if key != "llm_observations"}
            if "llm_observations" in value:
                result["llm_observations"] = [
                    {**ref, "path": (prefix / ref["path"]).as_posix()}
                    for ref in value["llm_observations"] or []
                ]
            return result
        if isinstance(value, list):
            return [copy(child) for child in value]
        return value

    return copy(payload)


def discover_observations(artifact: Path) -> list[dict[str, Any]] | None:
    """Index committed events left by a worker that never published its trace."""
    directory = artifact.parent / f"{artifact.stem}.llm_calls"
    if not directory.exists():
        return None
    references = []
    try:
        for path in sorted(directory.glob("*/*.json")):
            payload = path.read_bytes()
            record = json.loads(payload)
            reference = {
                "path": path.relative_to(artifact.parent).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "call_id": record["call_id"],
            }
            if "attempt_id" in record:
                reference["attempt_id"] = record["attempt_id"]
            read_observation(reference, artifact.parent)
            references.append(reference)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ObservationPersistenceError(
            f"cannot index interrupted observations: {error}"
        ) from error
    return references
