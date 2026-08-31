"""Shared pytest configuration: the provider test matrix (KTD5).

Every offline test that takes a ``provider`` argument runs once per
``ProviderCase`` row below, with the row id as the parametrize id. Rows carry
every provider-specific fact a test needs -- backend, model, credentials,
config-table text, an offline client builder, the sampling-arg contract, the
cost provenance, and a fake-response factory -- so test bodies never branch on
a provider name (branching on the row id is forbidden; a guard test enforces it).
Provider-specific divergence belongs in a row field or in an
``xfail(strict=True)`` attached from ``pytest_collection_modifyitems``.

Three rows: ``azure_kimi``, ``azure_gptoss``, ``openrouter_qwen``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Sentinel credentials for env stubs; nothing offline ever sends them.
SENTINEL_ENV: dict[str, str] = {
    "AZURE_API_KEY": "sentinel-key",
    "AZURE_FOUNDRY_ENDPOINT": "https://sentinel.services.ai.azure.com",
    "OPENROUTER_API_KEY": "sentinel-key",
}


class ClientBuilder(Protocol):
    def __call__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        response: Any = None,
        sampling_args: dict[str, Any] | None = None,
    ) -> Any: ...


def fake_response(
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 500,
    cost: float | None = None,
    finish_reason: str = "stop",
    content: str | None = "hello from provider",
    reasoning_content: str | None = None,
    completion_tokens_details: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """A fresh fake ``chat.completions`` response.

    ``SimpleNamespace`` (never ``MagicMock``) so ``hasattr``/``getattr``
    checks behave like a pydantic model: only the fields set here exist.
    ``reasoning_content`` lands on the message and
    ``completion_tokens_details`` on the usage block only when given, so rows
    that do not set them yield a response without those attributes.
    """
    usage_kwargs: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        "cost": cost,
        "model_extra": None,
    }
    if completion_tokens_details is not None:
        usage_kwargs["completion_tokens_details"] = SimpleNamespace(**completion_tokens_details)
    message_kwargs: dict[str, Any] = {"content": content, "model_extra": None}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content
    choice = SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(**message_kwargs))
    return SimpleNamespace(
        choices=[choice], usage=SimpleNamespace(**usage_kwargs), error=None, model_extra=None
    )


@dataclass(frozen=True, eq=False)
class ProviderCase:
    """One row of the provider matrix."""

    id: str
    backend: str
    model: str
    env_keys: tuple[str, ...]
    # The client's mandatory ``pricing`` kwarg; None for providers that report
    # costs themselves (no pricing rides along).
    pricing: dict[str, float] | None
    make_client: ClientBuilder
    # Top-level ``chat.completions.create`` kwargs the shipped decoding must
    # (and must not) produce through this provider's client.
    expected_sampling_keys: frozenset[str]
    forbidden_sampling_keys: frozenset[str]
    # The exact ``extra_body`` the shipped ``[decoding]`` table sends through a
    # role switched to this provider.
    expected_extra_body: dict[str, Any]
    # ``cost_source`` every persisted run records under this provider.
    cost_source: str
    # The shipped config whose roles select this provider.
    config_path: Path
    # ``decoding.max_output_tokens`` in ``config_path`` -- the output cap that
    # must reach the wire as ``max_completion_tokens`` from the shipped file.
    config_max_output_tokens: int
    response_defaults: dict[str, Any]
    # ``[backends.azure_foundry]`` overrides the shipped provider table needs
    # for this row (``thinking`` / ``reasoning_effort``); empty when the
    # shipped table already fits. Applied by ``all_roles_text`` (text surgery)
    # and ``smoke_config_for`` (dataclass replace) alike.
    azure_table: dict[str, Any] = field(default_factory=dict)
    # Whether ``response()`` carries the reasoning-model fields
    # (``message.reasoning_content`` and ``usage.completion_tokens_details``).
    expects_reasoning_fields: bool = False

    def role_table_text(self, role: str) -> str:
        """The exact ``[backends.<role>]`` TOML table for this provider."""
        return f'[backends.{role}]\nbackend = "{self.backend}"\nmodel = "{self.model}"'

    def response(self, **overrides: Any) -> SimpleNamespace:
        """A fresh fake response per call, shaped for this provider."""
        return fake_response(**{**self.response_defaults, **overrides})


def _azure_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    sampling_args: dict[str, Any] | None = None,
    pricing: dict[str, float],
) -> Any:
    from tests.clients.test_azure_foundry import _make_client

    kwargs: dict[str, Any] = {}
    if sampling_args is not None:
        kwargs["sampling_args"] = sampling_args
    return _make_client(monkeypatch, response=response, pricing=dict(pricing), **kwargs)


def _openrouter_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    sampling_args: dict[str, Any] | None = None,
) -> Any:
    from tests.clients.test_openai_transport import make_client

    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL_ENV["OPENROUTER_API_KEY"])
    return make_client(response=response, sampling_args=sampling_args)


AZURE_KIMI_PRICING: dict[str, float] = {"input_per_million": 0.60, "output_per_million": 3.00}
AZURE_GPTOSS_PRICING: dict[str, float] = {"input_per_million": 0.15, "output_per_million": 0.60}

# The OpenAI client renames max_tokens -> max_completion_tokens itself.
_OPENAI_SURFACE_KEYS = frozenset({"temperature", "top_p", "max_completion_tokens"})

AZURE_KIMI = ProviderCase(
    id="azure_kimi",
    backend="azure_foundry",
    model="Kimi-K2.5",
    env_keys=("AZURE_API_KEY", "AZURE_FOUNDRY_ENDPOINT"),
    pricing=AZURE_KIMI_PRICING,
    make_client=partial(_azure_client, pricing=AZURE_KIMI_PRICING),
    expected_sampling_keys=_OPENAI_SURFACE_KEYS,
    forbidden_sampling_keys=frozenset({"max_tokens", "reasoning_effort", "pricing"}),
    # Instant mode rides chat_template_kwargs; the shipped decoding knobs
    # top_k/min_p ride extra_body for every backend.
    expected_extra_body={"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"thinking": False}},
    cost_source="synthesized",
    config_path=CONFIG_DIR / "experiment_kimiK25.toml",
    config_max_output_tokens=8192,
    response_defaults={"content": "hello from foundry"},
)

AZURE_GPTOSS = ProviderCase(
    id="azure_gptoss",
    backend="azure_foundry",
    model="gpt-oss-120b",
    env_keys=("AZURE_API_KEY", "AZURE_FOUNDRY_ENDPOINT"),
    pricing=AZURE_GPTOSS_PRICING,
    make_client=partial(_azure_client, pricing=AZURE_GPTOSS_PRICING),
    # reasoning_effort is a TOP-LEVEL request parameter, never extra_body.
    expected_sampling_keys=_OPENAI_SURFACE_KEYS | {"reasoning_effort"},
    forbidden_sampling_keys=frozenset({"max_tokens", "pricing"}),
    # Same convention as the Kimi row: the shipped [decoding] (top_k/min_p)
    # after role surgery, so extra_body carries those two knobs and NO
    # chat_template_kwargs (thinking = true sends nothing). The gpt-oss TOML
    # itself sets neither top_k nor min_p; the config_path tests cover that.
    expected_extra_body={"top_k": 20, "min_p": 0.0},
    cost_source="synthesized",
    config_path=CONFIG_DIR / "experiment_oolong_gptoss.toml",
    config_max_output_tokens=16384,
    response_defaults={
        "content": "hello from gpt-oss",
        "reasoning_content": "thinking about hi",
        "completion_tokens_details": {"reasoning_tokens": 100},
    },
    azure_table={"thinking": True, "reasoning_effort": "medium"},
    expects_reasoning_fields=True,
)

OPENROUTER_QWEN = ProviderCase(
    id="openrouter_qwen",
    backend="openrouter",
    model="qwen/qwen3-30b-a3b-instruct-2507",
    env_keys=("OPENROUTER_API_KEY",),
    pricing=None,
    make_client=_openrouter_client,
    expected_sampling_keys=_OPENAI_SURFACE_KEYS,
    forbidden_sampling_keys=frozenset({"max_tokens", "reasoning_effort", "pricing"}),
    # Empty provider_order: no routing dict; no azure-only chat_template_kwargs.
    expected_extra_body={"top_k": 20, "min_p": 0.0},
    cost_source="provider",
    config_path=CONFIG_DIR / "experiment.toml",
    config_max_output_tokens=4096,
    response_defaults={"content": "ok", "cost": 0.001},
)

PROVIDER_CASES: tuple[ProviderCase, ...] = (AZURE_KIMI, AZURE_GPTOSS, OPENROUTER_QWEN)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "provider" in metafunc.fixturenames:
        metafunc.parametrize(
            "provider", [pytest.param(case, id=case.id) for case in PROVIDER_CASES]
        )


# Provider-specific divergence hooks go here as xfail(strict=True) marks keyed
# on (test name, row id); none exist for the three-row table.
_ROW_XFAILS: dict[tuple[str, str], str] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        callspec = getattr(item, "callspec", None)
        provider = callspec.params.get("provider") if callspec is not None else None
        if provider is None:
            continue
        reason = _ROW_XFAILS.get((item.originalname, provider.id))
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))

    # Live (paid) items are DESELECTED -- not merely skipped -- unless the
    # explicit opt-in is present and we are not in CI (KTD8). The modules'
    # own module-level skipif gates stay in place: once selected, `-rs` names
    # the exact missing gate (credential, flag, or pricing attestation).
    import os

    if os.environ.get("SHRLM_RUN_LIVE") == "1" and not os.environ.get("CI"):
        return
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (deselected if item.get_closest_marker("live") else kept).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


__all__: list[str] = [
    "AZURE_GPTOSS",
    "AZURE_GPTOSS_PRICING",
    "AZURE_KIMI",
    "AZURE_KIMI_PRICING",
    "OPENROUTER_QWEN",
    "PROVIDER_CASES",
    "SENTINEL_ENV",
    "ProviderCase",
    "fake_response",
]
