"""Experiment adapter for the pinned upstream λ-RLM baseline."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlm.core.types import ClientBackend
from shrlm.baselines.upstream.lambda_rlm import LambdaRLM

LAMBDA_RLM_UPSTREAM_REPOSITORY = (
    "https://github.com/lambda-calculus-LLM/lambda-RLM"
)
LAMBDA_RLM_UPSTREAM_REVISION = (
    "c70c6db2cf8498eaf05fa6999dc41088827158f4"
)


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
    "LAMBDA_RLM_UPSTREAM_REPOSITORY",
    "LAMBDA_RLM_UPSTREAM_REVISION",
    "LambdaBaselineConfig",
    "LambdaInput",
    "lambda_input",
]
