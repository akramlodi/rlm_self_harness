"""Scenario pricing and the recommendation policy for the cost/time report (R6).

Split out of ``shrlm.experiment.report`` (which still holds measurement, run
counts, and the projections this module prices): every priced way to buy the
full experiment's projected tokens -- both API pricing tiers and every
configured GPU profile -- plus the policy that turns those prices into a
single recommendation or a labeled provisional comparison.

Recommendation policy (explicit and tested)
    Recommend the cheapest scenario that passes validity:

    1. no lower-bound flag on any contributing long-run mean,
    2. long coverage present for every configured environment,
    3. GPU scenarios eligible only when their throughput input carries
       validated provenance -- a profile declaring
       ``provenance_validated = false`` is scenario-only and can never be the
       recommendation.

    Otherwise the comparison is still emitted, labeled provisional, naming
    each failing gate. Scenarios declaring ``changes_numerics = true``
    (INT4-style) are surfaced with that flag and deprioritized -- excluded
    from candidacy unless quantization is explicitly accepted, never silently
    dropped from the table. Both gates read the profile's declared booleans,
    never its free-text ``provenance`` / ``precision_note``: those are the
    human-readable justification the report prints, and editing a
    justification must not move a decision.

``ReportInputError`` lives here (not in ``report.py``) so this module never
imports back from ``report.py`` at runtime: ``report.py`` depends on
``scenarios.py`` (for scenario pricing and the recommendation), never the
reverse. ``report.py`` re-exports it unchanged, so existing callers and the
raised type are identical to before the split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from shrlm.experiment.config import ExperimentConfig, GpuScenario, PricingTier
from shrlm.experiment.errors import ExperimentError
from shrlm.experiment.splits import LENGTHS

if TYPE_CHECKING:
    from shrlm.experiment.report import Projection, RunBucket

SHORT, LONG = LENGTHS

KIND_API = "api"
KIND_GPU = "gpu"

SCENARIO_API_PROMO = "api_promo"
SCENARIO_API_LIST = "api_list"

STATUS_RECOMMENDED = "recommended"
STATUS_PROVISIONAL = "provisional"

GATE_LONG_LOWER_BOUND = "long_run_lower_bound"
GATE_LONG_COVERAGE = "long_coverage"
GATE_NO_ELIGIBLE_SCENARIO = "no_eligible_scenario"

SECONDS_PER_HOUR = 3600.0
TOKENS_PER_PRICE_UNIT = 1e6


class ReportInputError(ExperimentError):
    """The experiment directory does not hold the measurements a report needs."""


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One priced way to buy the full experiment's tokens."""

    name: str
    kind: str
    usd_point: float
    usd_pessimistic: float
    eligible: bool
    ineligible_reason: str | None
    changes_numerics: bool
    provenance: str
    detail: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "usd_point": self.usd_point,
            "usd_pessimistic": self.usd_pessimistic,
            "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason,
            "changes_numerics": self.changes_numerics,
            "provenance": self.provenance,
            "detail": self.detail,
        }


def api_usd(tier: PricingTier, projection: Projection) -> float:
    return (
        projection.input_tokens / TOKENS_PER_PRICE_UNIT * tier.input_per_million
        + projection.output_tokens / TOKENS_PER_PRICE_UNIT * tier.output_per_million
    )


def gpu_hours(profile: GpuScenario, projection: Projection) -> float:
    """Tokens divided by the context-bucketed throughput, in hours."""
    seconds = 0.0
    for context, tokens in projection.tokens_by_context().items():
        if tokens == 0.0:
            continue
        if context not in profile.throughput_tokens_per_second:
            raise ReportInputError(
                f"gpu scenario {profile.name!r} has no {context!r} throughput; every "
                f"scenario must declare throughput_tokens_per_second for {list(LENGTHS)}"
            )
        seconds += tokens / profile.throughput_tokens_per_second[context]
    return seconds / SECONDS_PER_HOUR


def api_scenario(
    name: str, tier: PricingTier, point: Projection, pessimistic: Projection
) -> Scenario:
    return Scenario(
        name=name,
        kind=KIND_API,
        usd_point=api_usd(tier, point),
        usd_pessimistic=api_usd(tier, pessimistic),
        eligible=True,
        ineligible_reason=None,
        changes_numerics=False,
        provenance=(
            f"configured pricing tier: ${tier.input_per_million}/1M in, "
            f"${tier.output_per_million}/1M out"
        ),
        detail={
            "input_per_million": tier.input_per_million,
            "output_per_million": tier.output_per_million,
        },
    )


def gpu_scenario(profile: GpuScenario, point: Projection, pessimistic: Projection) -> Scenario:
    hours_point = gpu_hours(profile, point)
    hours_pessimistic = gpu_hours(profile, pessimistic)
    validated = profile.provenance_validated
    return Scenario(
        name=profile.name,
        kind=KIND_GPU,
        usd_point=hours_point * profile.hourly_rate_usd,
        usd_pessimistic=hours_pessimistic * profile.hourly_rate_usd,
        eligible=validated,
        ineligible_reason=(
            None
            if validated
            else (
                "provenance is marked unvalidated, so its throughput input cannot support a "
                "recommendation; this profile is scenario-only"
            )
        ),
        changes_numerics=profile.changes_numerics,
        provenance=profile.provenance,
        detail={
            "hourly_rate_usd": profile.hourly_rate_usd,
            "gpu_hours_point": hours_point,
            "gpu_hours_pessimistic": hours_pessimistic,
            "usd_point_low": hours_point * profile.sensitivity_range[0],
            "usd_point_high": hours_point * profile.sensitivity_range[1],
            "sensitivity_range": list(profile.sensitivity_range),
            "precision_note": profile.precision_note,
            "throughput_tokens_per_second": dict(profile.throughput_tokens_per_second),
        },
    )


def build_scenarios(
    config: ExperimentConfig, point: Projection, pessimistic: Projection
) -> list[Scenario]:
    """Both API tiers plus every configured GPU profile, cheapest first."""
    scenarios = [
        api_scenario(SCENARIO_API_PROMO, config.pricing.promo, point, pessimistic),
        api_scenario(SCENARIO_API_LIST, config.pricing.list_price, point, pessimistic),
        *(gpu_scenario(profile, point, pessimistic) for profile in config.gpu_scenarios),
    ]
    return sorted(scenarios, key=lambda scenario: (scenario.usd_point, scenario.name))


# ---------------------------------------------------------------------------
# Recommendation policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """The verdict: a named scenario, or a provisional comparison plus gaps."""

    status: str
    scenario: str | None
    usd: float | None
    failing_gates: list[dict[str, str]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scenario": self.scenario,
            "usd": self.usd,
            "failing_gates": self.failing_gates,
        }


def configured_environments(config: ExperimentConfig) -> tuple[str, ...]:
    """The environment names the config declares, in declaration order."""
    return tuple(field.name for field in fields(config.environments))


def validity_gates(config: ExperimentConfig, buckets: Sequence[RunBucket]) -> list[dict[str, str]]:
    """Every failing validity gate, in policy order (see the module docstring)."""
    gates: list[dict[str, str]] = []
    for bucket in buckets:
        if bucket.length == LONG and bucket.lower_bound:
            gates.append(
                {
                    "gate": GATE_LONG_LOWER_BOUND,
                    "detail": (
                        f"the {bucket.environment}/{bucket.length} per-run mean is a lower "
                        "bound (terminated runs contributed to it), so every extrapolation "
                        "resting on it understates the long-context cost"
                    ),
                }
            )
    measured_long = {bucket.environment for bucket in buckets if bucket.length == LONG}
    for environment in configured_environments(config):
        if environment not in measured_long:
            gates.append(
                {
                    "gate": GATE_LONG_COVERAGE,
                    "detail": (
                        f"no measured {LONG} runs for environment {environment!r}; long "
                        "coverage is required for every configured environment before a "
                        "recommendation can be made"
                    ),
                }
            )
    return gates


def recommend(
    scenarios: Sequence[Scenario],
    gates: Sequence[dict[str, str]],
    *,
    accept_quantization: bool,
) -> Recommendation:
    """The decision function: gates first, then cheapest eligible scenario.

    Inputs: the priced scenarios, the failing validity gates, and whether
    quantization is explicitly accepted. Verdict: ``provisional`` with the
    failing gates named whenever any gate fails or no scenario is eligible;
    otherwise ``recommended`` naming the cheapest scenario that is eligible
    (validated provenance) and precision-preserving (or quantization-accepted).
    """
    failing = list(gates)
    if failing:
        return Recommendation(
            status=STATUS_PROVISIONAL, scenario=None, usd=None, failing_gates=failing
        )
    candidates = [
        scenario
        for scenario in scenarios
        if scenario.eligible and (accept_quantization or not scenario.changes_numerics)
    ]
    if not candidates:
        return Recommendation(
            status=STATUS_PROVISIONAL,
            scenario=None,
            usd=None,
            failing_gates=[
                {
                    "gate": GATE_NO_ELIGIBLE_SCENARIO,
                    "detail": (
                        "every scenario is either unvalidated or changes model numerics; "
                        "validate a profile's throughput or accept quantization explicitly"
                    ),
                }
            ],
        )
    cheapest = min(candidates, key=lambda scenario: (scenario.usd_point, scenario.name))
    return Recommendation(
        status=STATUS_RECOMMENDED,
        scenario=cheapest.name,
        usd=cheapest.usd_point,
        failing_gates=[],
    )


__all__ = [
    "GATE_LONG_COVERAGE",
    "GATE_LONG_LOWER_BOUND",
    "GATE_NO_ELIGIBLE_SCENARIO",
    "KIND_API",
    "KIND_GPU",
    "SCENARIO_API_LIST",
    "SCENARIO_API_PROMO",
    "STATUS_PROVISIONAL",
    "STATUS_RECOMMENDED",
    "Recommendation",
    "ReportInputError",
    "Scenario",
    "api_scenario",
    "api_usd",
    "build_scenarios",
    "configured_environments",
    "gpu_hours",
    "gpu_scenario",
    "recommend",
    "validity_gates",
]
