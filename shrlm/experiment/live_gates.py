"""The U4 live-tier gate (KTD8): when may a pytest run spend real money?

Live tests run ONLY when every one of these holds:

* every credential environment variable the CONFIGURED runner backend reads
  (``shrlm.optimization.driver._BACKEND_ENV_KEYS``; for ``azure_foundry``
  that is ``AZURE_API_KEY`` and ``AZURE_FOUNDRY_ENDPOINT``, for
  ``openrouter`` it is ``OPENROUTER_API_KEY``) is set (non-empty),
* ``SHRLM_RUN_LIVE`` equals exactly ``"1"``,
* when -- and only when -- the runner backend is ``azure_foundry`` (the
  cost-SYNTHESIZING backend), ``SHRLM_VERIFIED_PRICING`` attests the
  deployment's verified rates and matches the configured
  ``[pricing.list_price]`` (format
  ``"<input_per_million>/<output_per_million>"``, e.g. ``"0.60/3.00"``) --
  the plan's pricing-verification prerequisite: the $5 budget proof is
  meaningless when the configured rates differ from what Azure actually
  bills, so the user checks the portal and exports the figure they saw. An
  openrouter runner reports provider costs directly, so no attestation is
  demanded of it (the promised R4/KTD9 fallback stays runnable),

AND they are skipped whenever a ``CI`` environment variable is present with
any non-empty value -- CI wins over everything, including a fully
credentialed environment with the opt-in flag set, and is checked before the
config is ever read. Credential presence alone must never spend: the
explicit ``SHRLM_RUN_LIVE=1`` opt-in is what separates "this machine could
pay" from "this invocation is allowed to pay".

The helper takes the environment as an argument so the decision itself is
unit-testable offline (see ``tests/experiment/test_smoke_live.py``), and an
optional ``runner_backend`` override (None means "read the smoke profile's
configured runner backend") so the backend-conditional branches are testable
without rewriting the shipped config. The live test modules evaluate it once
at import time against ``os.environ`` and hang a ``pytest.mark.skipif`` off
the result, so ``-rs`` output names the exact missing gate.
"""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from shrlm.experiment.config import ExperimentConfig


LIVE_FLAG = "SHRLM_RUN_LIVE"
VERIFIED_PRICING_KEY = "SHRLM_VERIFIED_PRICING"


def pricing_attestation_mismatch(
    attested: str | None, input_per_million: float, output_per_million: float
) -> str | None:
    """Why the pricing attestation fails against configured rates, or ``None``.

    ``attested`` is the ``SHRLM_VERIFIED_PRICING`` value, expected as
    ``"<in>/<out>"`` in USD per 1M tokens. Absent, malformed, or mismatched
    attestations all return a reason -- the live tiers fail fast rather than
    spending against an unverified budget proof. The reason never echoes
    credentials (pricing is non-secret).
    """
    expected = f"{input_per_million}/{output_per_million}"
    if not attested:
        return (
            f"live gate not met: {VERIFIED_PRICING_KEY} not set -- verify the "
            f"deployment's rates in the Azure portal, then export "
            f"{VERIFIED_PRICING_KEY}='{expected}' to attest they match the "
            "configured [pricing.list_price]"
        )
    parts = attested.split("/")
    try:
        attested_pair = (float(parts[0]), float(parts[1])) if len(parts) == 2 else None
    except ValueError:
        attested_pair = None
    if attested_pair is None:
        return (
            f"live gate not met: {VERIFIED_PRICING_KEY}={attested!r} is not "
            "'<input_per_million>/<output_per_million>'"
        )
    if attested_pair != (input_per_million, output_per_million):
        return (
            f"live gate not met: {VERIFIED_PRICING_KEY}={attested!r} does not match "
            f"the configured [pricing.list_price] {expected} -- reconcile the config "
            "with the verified portal rate before any live spend"
        )
    return None


def live_skip_reason(
    env: Mapping[str, str] | None = None,
    runner_backend: str | None = None,
    config: "ExperimentConfig | None" = None,
) -> str | None:
    """The reason the live tier must be skipped, or ``None`` when it may run.

    Checks are ordered so the returned reason names the decisive gate: CI
    first (it overrides everything else, KTD8, and wins before any config
    read), then the configured runner backend's credentials (each missing
    variable named, values never echoed), then the explicit opt-in flag, then
    -- for the cost-synthesizing ``azure_foundry`` backend only -- the pricing
    attestation.

    ``runner_backend`` of None means "read the smoke profile's configured
    runner backend"; offline tests pass a backend name to exercise the
    conditional branches directly. ``config`` is the experiment config whose
    ``[pricing.list_price]`` the attestation is checked against; None means
    the shipped smoke profile. A live test that spends under a different
    config (``configs/experiment_kimiK25.toml``) must pass that config, or
    the gate compares the attestation to the wrong rate card.

    When reading the real environment, ``load_dotenv()`` runs first so the
    gate sees exactly what the clients see (every ``rlm.clients`` module
    loads ``.env`` into ``os.environ`` at import): otherwise the decision
    would depend on whether a client module happened to be imported before
    this evaluation. An explicit ``env`` mapping (the offline unit tests)
    bypasses that.
    """
    if env is None:
        load_dotenv()
        env = os.environ
    if env.get("CI"):
        return "CI is set: live (paid) tests never run in CI, even fully credentialed (KTD8)"

    from shrlm.experiment.config import load_config
    from shrlm.optimization.driver import _BACKEND_ENV_KEYS

    if runner_backend is None:
        if config is None:
            config = load_config(profile="smoke")
        runner_backend = config.backends.runner.backend
    if runner_backend not in _BACKEND_ENV_KEYS:
        raise ValueError(
            f"runner backend {runner_backend!r} has no credential mapping in "
            f"_BACKEND_ENV_KEYS ({sorted(_BACKEND_ENV_KEYS)}); refusing to gate a "
            "live spend against unknown credentials."
        )
    missing = [key for key in _BACKEND_ENV_KEYS[runner_backend] if not env.get(key)]
    if missing:
        return f"live gate not met: {', '.join(missing)} not set (KTD8)"
    if env.get(LIVE_FLAG) != "1":
        return (
            f"live gate not met: {LIVE_FLAG} != '1' -- credential presence alone "
            "must never spend (KTD8)"
        )
    if runner_backend != "azure_foundry":
        # Only the cost-synthesizing backend needs the pricing attestation;
        # openrouter reports provider costs directly.
        return None
    if config is None:
        config = load_config(profile="smoke")
    pricing = config.pricing.list_price
    return pricing_attestation_mismatch(
        env.get(VERIFIED_PRICING_KEY),
        pricing.input_per_million,
        pricing.output_per_million,
    )
