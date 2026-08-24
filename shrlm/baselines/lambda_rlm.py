"""Experiment adapter for the pinned upstream λ-RLM baseline."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlm.core.types import ClientBackend
from shrlm.baselines.upstream.lambda_rlm import LambdaRLM

LAMBDA_RLM_UPSTREAM_REPOSITORY = "https://github.com/lambda-calculus-LLM/lambda-RLM"
LAMBDA_RLM_UPSTREAM_REVISION = "c70c6db2cf8498eaf05fa6999dc41088827158f4"
LAMBDA_RLM_SOURCE_SHA256 = "3f0e0521f92e1e124e76aa4f717a7bf29c95386ff42b3faf6057d4fa320f42e6"
LAMBDA_RLM_METHOD_FORMAT = "shrlm-method/v1"
LAMBDA_RLM_METHOD_KIND = "lambda_rlm"
LAMBDA_RLM_DISPLAY_NAME = "λ-RLM"


@dataclass(frozen=True)
class LambdaInput:
    """The non-gold input passed to λ-RLM for one experiment instance."""

    prompt: str
    query: str


@dataclass(frozen=True)
class LambdaBaselineConfig:
    """The λ-RLM parameters fixed for this experimental baseline."""

    context_window_chars: int = 100_000
    accuracy_target: float = 0.80
    a_leaf: float = 0.95
    a_compose: float = 0.90

    def build(
        self,
        *,
        backend: ClientBackend,
        backend_kwargs: dict[str, Any],
        query: str,
    ) -> LambdaRLM:
        """Construct the upstream method with experiment-supplied model settings."""
        if not query.strip():
            raise ValueError("λ-RLM requires a non-empty query")

        return LambdaRLM(
            backend=backend,
            backend_kwargs=dict(backend_kwargs),
            environment="local",
            context_window_chars=self.context_window_chars,
            accuracy_target=self.accuracy_target,
            a_leaf=self.a_leaf,
            a_compose=self.a_compose,
            query=query,
        )


def serialize_lambda_method(config: LambdaBaselineConfig) -> dict[str, Any]:
    """Serialize everything that changes the fixed λ-RLM inference method."""
    return {
        "kind": LAMBDA_RLM_METHOD_KIND,
        "display_name": LAMBDA_RLM_DISPLAY_NAME,
        "upstream": {
            "repository": LAMBDA_RLM_UPSTREAM_REPOSITORY,
            "revision": LAMBDA_RLM_UPSTREAM_REVISION,
            "source_sha256": LAMBDA_RLM_SOURCE_SHA256,
        },
        "runtime": {"environment": "local"},
        "configuration": asdict(config),
    }


def lambda_method_hash(config: LambdaBaselineConfig) -> str:
    """Return the stable SHA-256 identity of one λ-RLM method configuration."""
    serialization = serialize_lambda_method(config)
    canonical = json.dumps(
        serialization,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lambda_method_envelope(config: LambdaBaselineConfig) -> dict[str, Any]:
    """Build the complete persisted identity envelope for one λ-RLM method."""
    return {
        "format": LAMBDA_RLM_METHOD_FORMAT,
        "kind": LAMBDA_RLM_METHOD_KIND,
        "display_name": LAMBDA_RLM_DISPLAY_NAME,
        "hash": lambda_method_hash(config),
        "method": serialize_lambda_method(config),
    }


def write_lambda_method_json(
    config: LambdaBaselineConfig,
    path: str | Path,
) -> dict[str, Any]:
    """Persist the reproducible identity envelope for a λ-RLM evaluation round."""
    envelope = lambda_method_envelope(config)
    Path(path).write_text(json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return envelope


def lambda_input(instance: Mapping[str, Any]) -> LambdaInput:
    """Read only prompt and question fields from a persisted instance."""
    prompt = instance["prompt"]
    query = instance["question"]

    if not isinstance(prompt, str):
        raise TypeError("λ-RLM instance prompt must be a string")
    if not isinstance(query, str):
        raise TypeError("λ-RLM instance question must be a string")
    if not prompt.strip():
        raise ValueError("λ-RLM instance has an empty prompt")
    if not query.strip():
        raise ValueError("λ-RLM instance has an empty question")

    # Preserve the persisted strings exactly. Validation uses strip only to
    # detect empty values; it does not rewrite the model input.
    return LambdaInput(prompt=prompt, query=query)


__all__ = [
    "LAMBDA_RLM_DISPLAY_NAME",
    "LAMBDA_RLM_METHOD_FORMAT",
    "LAMBDA_RLM_METHOD_KIND",
    "LAMBDA_RLM_SOURCE_SHA256",
    "LAMBDA_RLM_UPSTREAM_REPOSITORY",
    "LAMBDA_RLM_UPSTREAM_REVISION",
    "LambdaBaselineConfig",
    "LambdaInput",
    "lambda_method_envelope",
    "lambda_method_hash",
    "lambda_input",
    "serialize_lambda_method",
    "write_lambda_method_json",
]
