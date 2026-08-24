"""Typed, fail-fast loading of ``configs/experiment.toml`` (R1; KTD1).

This module is the only file in the repository that reads the experiment
TOML. Every experiment parameter lives in that one file; the scaffold code
hardcodes none of them. ``load_config(profile="full" | "smoke")`` parses the
file with stdlib ``tomllib`` into frozen dataclasses, rejecting unknown
top-level keys, unknown keys inside any table, and missing mandatory keys --
in particular ``caps.max_budget``, ``caps.max_timeout``, and
``caps.candidate_budget`` (R12) -- with a ``ValueError`` naming the key.

Profiles (R2)
    The ``[smoke]`` table overrides scale counts only: instance counts,
    repetition counts, candidate width, round count, and spend caps
    (``SMOKE_SCALE_KEYS`` is the exhaustive set). A smoke key outside that set
    is rejected at load time, so the smoke profile can never change semantics
    -- decoding, promotion rule, environments, and backends are byte-identical
    to the full profile.

Identity hash (R3; KTD3)
    ``identity_hash`` is a sha256 over a canonical JSON rendering of every
    behavior-changing value: decoding/sampling args, split sizes and seeds,
    loop counts (m, v, k, t, patience), promotion thresholds and bands, caps
    (including per-round attempts), environment definitions with their dataset
    revision pins, the runner/attributor/proposer backends including the
    optional provider tables (OpenRouter routing, Azure Foundry thinking
    mode), and the evaluation repetition count
    (``IDENTITY_OPERATIONAL_KEYS``; see below). The remaining operational keys
    -- cache paths, loader timeout, the validation worker count, pricing and
    GPU tables, report settings -- are excluded: they change what a run costs,
    how long it takes, or how it is summarized, never what it does or how much
    of it lands on disk.

eval_repetitions -> attempts (KTD8)
    ``operational.eval_repetitions`` is the evaluation-stage repetition count
    per instance per condition. At evaluation time it is forwarded as the
    round's ``attempts`` (``RoundConfig.attempts``). Attempts are unseeded
    temperature samples, so a repetition draws another sample from the same
    per-run behavior distribution without changing that distribution -- but it
    does decide how many runs an evaluation persists and what
    ``eval/eval_summary.json`` claims the evaluated plan was. Lowering it under
    an existing experiment directory would reuse the runs a higher count
    already persisted while rewriting the summary as if the smaller plan had
    been evaluated, so the effective count is identity-protected
    (``IDENTITY_OPERATIONAL_KEYS``) exactly like ``caps.attempts``. The smoke
    profile may still shrink it (``SMOKE_SCALE_KEYS``): switching profiles
    already moves the identity hash, and each profile owns its own out_dir.

Factory helpers
    ``round_config_kwargs`` and ``evaluation_config_kwargs`` return kwargs for
    ``RoundConfig`` and ``EvaluationConfig``, minus the per-round arguments
    (harness, instances, splits, verifier, out_dir, round_index) their callers
    own. ``validation_caps``, ``promotion_config``, and ``proposer_config``
    construct the real objects. Call sites never touch raw TOML.
"""

import tomllib
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

from shrlm.harness_identity import canonical_json_sha256
from shrlm.optimization.costs import ValidationCaps
from shrlm.optimization.promotion import Band, PromotionConfig
from shrlm.optimization.proposal import ProposerConfig

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "experiment.toml"

PROFILES: tuple[str, ...] = ("full", "smoke")

CLIENT_ROLES: tuple[str, ...] = ("runner", "attributor", "proposer")

TOP_LEVEL_KEYS: tuple[str, ...] = (
    "decoding",
    "splits",
    "loop",
    "promotion",
    "caps",
    "environments",
    "backends",
    "pricing",
    "gpu_scenarios",
    "report",
    "operational",
    "smoke",
)

# The config sections that enter the identity hash; everything else is
# operational (R3). Order is irrelevant -- the JSON rendering sorts keys.
IDENTITY_SECTIONS: tuple[str, ...] = (
    "decoding",
    "splits",
    "loop",
    "promotion",
    "caps",
    "environments",
    "backends",
)

# Identity-covered keys that live inside an otherwise operational section.
# ``operational.eval_repetitions`` is the evaluation stage's effective attempt
# count: it decides how many runs land on disk and what the eval summary claims
# was evaluated, so lowering it under an existing experiment directory must be
# refused like any other identity change (see the module docstring on KTD8).
IDENTITY_OPERATIONAL_KEYS: tuple[str, ...] = ("eval_repetitions",)

# The exhaustive set of dotted keys the [smoke] table may override (R2):
# scale counts (instances, repetitions, candidate width, rounds) and spend
# caps. Anything else would change semantics between profiles.
SMOKE_SCALE_KEYS: frozenset[str] = frozenset(
    {
        "splits.n_in",
        "splits.n_ho",
        "splits.test_short",
        "splits.test_long",
        "loop.m",
        "loop.v",
        "loop.k",
        "loop.t",
        "caps.attempts",
        "caps.max_budget",
        "caps.max_timeout",
        "caps.candidate_budget",
        "operational.eval_repetitions",
        "operational.validation_workers",
        "environments.oolong_pairs.n_short",
        "environments.oolong_pairs.n_long",
    }
)


# ---------------------------------------------------------------------------
# Sections, one frozen dataclass per TOML table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodingConfig:
    """Section 3.0 decoding invariant, identical for root and sub-calls.

    ``top_k`` and ``min_p`` are optional: the Kimi-K2.5 instant-mode card
    specifies neither, and an absent knob is omitted from ``extra_body``
    entirely (the Foundry v1 route's handling of unknown body params is
    unconfirmed), never sent as a default.
    """

    temperature: float
    top_p: float
    max_output_tokens: int
    top_k: int | None = None
    min_p: float | None = None


@dataclass(frozen=True)
class SplitsConfig:
    """Source-environment split sizes and the partition seed (section 3.1)."""

    n_in: int
    n_ho: int
    test_short: int
    test_long: int
    seed: int


@dataclass(frozen=True)
class LoopConfig:
    """Optimization-loop counts (section 3.2): m mining repetitions, v
    validation repetitions, k candidates per round, t max rounds, patience."""

    m: int
    v: int
    k: int
    t: int
    patience: int


@dataclass(frozen=True)
class PromotionSettings:
    """Preregistered promotion thresholds and bands, as raw TOML values.

    ``promotion_config`` turns these into a real ``PromotionConfig``; a
    ``sub_call_band`` of None means the paper's unconstrained rule.
    """

    tau_regression: float
    tau_improvement: float
    cost_band: tuple[float, float]
    sub_call_band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("cost_band", "sub_call_band"):
            value = getattr(self, name)
            if value is not None and len(value) != 2:
                raise ValueError(f"promotion.{name} must be [lower, upper], got {value!r}")


@dataclass(frozen=True)
class CapsConfig:
    """Experiment-owned run limits (KTD3). The first three are mandatory in
    the TOML (R12); ``max_depth`` may stay None when S6 declares it."""

    max_budget: float
    max_timeout: float
    candidate_budget: float
    attempts: int
    max_iterations: int
    max_depth: int | None = None


@dataclass(frozen=True)
class GraphWalksConfig:
    """Source environment: dataset files, revision pin, and filters."""

    dataset_repo: str
    dataset_file_short: str
    dataset_file_long: str
    dataset_revision: str
    max_chars: int
    min_chars: int
    problem_types: tuple[str, ...]


@dataclass(frozen=True)
class OolongPairsConfig:
    """Target environment: tasks, context lengths, and test-set sizes."""

    dataset_repo: str
    subset: str
    dataset_revision: str
    task_ids: tuple[int, ...]
    context_length_short: int
    context_length_long: int
    max_scan: int
    n_short: int
    n_long: int


@dataclass(frozen=True)
class EnvironmentsConfig:
    graphwalks: GraphWalksConfig
    oolong_pairs: OolongPairsConfig


@dataclass(frozen=True)
class EndpointConfig:
    """One model endpoint: the client backend name and the model it serves."""

    backend: str
    model: str


@dataclass(frozen=True)
class OpenRouterConfig:
    """Provider routing, forwarded as ``extra_body["provider"]``. An empty
    order tuple means no restriction."""

    provider_order: tuple[str, ...]
    allow_fallbacks: bool


@dataclass(frozen=True)
class AzureFoundryConfig:
    """Azure AI Foundry provider settings. ``thinking = False`` selects
    Kimi-K2.5 instant mode, forwarded as
    ``extra_body["chat_template_kwargs"] = {"thinking": False}``."""

    thinking: bool


@dataclass(frozen=True)
class BackendsConfig:
    """Role endpoints plus optional per-provider tables (KTD5): a provider
    table may be absent when no role uses that backend, and an openrouter role
    with no ``[backends.openrouter]`` table simply sends no provider routing.
    An azure_foundry role, by contrast, REQUIRES the
    ``[backends.azure_foundry]`` table: thinking mode is declared, never
    defaulted, so ``load_config`` refuses the absent table."""

    runner: EndpointConfig
    attributor: EndpointConfig
    proposer: EndpointConfig
    openrouter: OpenRouterConfig | None = None
    azure_foundry: AzureFoundryConfig | None = None


@dataclass(frozen=True)
class PricingTier:
    """USD per 1M tokens."""

    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class PricingConfig:
    promo: PricingTier
    list_price: PricingTier


@dataclass(frozen=True)
class GpuScenario:
    """One local-serving cost scenario for the report's GPU-hour projection.

    ``provenance_validated`` and ``changes_numerics`` are the two booleans the
    report's recommendation policy reads: an unvalidated profile is
    scenario-only (it can never be recommended), and one whose numerics change
    is a candidate only when quantization is explicitly accepted. They are
    declared per profile in the TOML rather than inferred from the prose, so
    editing a justification can never move the decision. ``provenance`` and
    ``precision_note`` stay as the human-readable justification the report
    prints alongside each verdict.
    """

    name: str
    hourly_rate_usd: float
    provenance: str
    provenance_validated: bool
    sensitivity_range: tuple[float, float]
    precision_note: str
    changes_numerics: bool
    throughput_tokens_per_second: dict[str, float]


@dataclass(frozen=True)
class ReportConfig:
    """Report-only assumptions: merge frequency and the evaluation grid."""

    p_merge: float
    eval_conditions: int


@dataclass(frozen=True)
class OperationalConfig:
    """Keys that never change run behavior: timeouts, cache paths, the
    evaluation repetition count (see the module docstring on KTD8), and the
    validation worker count.

    ``validation_workers`` caps how many validation subjects (the baseline
    plus candidates) evaluate concurrently, each in its own child process. It
    changes wall clock and peak request rate only -- every subject still runs
    under its own breaker, deadline, and persist-first manifests -- so it is
    excluded from the identity hash and may change under an existing
    experiment directory. ``1`` (the default) is the sequential in-process
    path with no child process at all.
    """

    loader_timeout_seconds: float
    attribution_cache_path: str
    proposal_cache_path: str
    eval_repetitions: int
    validation_workers: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("operational.eval_repetitions", self.eval_repetitions)
        _require_positive_int("operational.validation_workers", self.validation_workers)


def _require_positive_int(label: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")


@dataclass(frozen=True)
class ExperimentConfig:
    """The fully parsed experiment configuration for one profile."""

    profile: str
    decoding: DecodingConfig
    splits: SplitsConfig
    loop: LoopConfig
    promotion: PromotionSettings
    caps: CapsConfig
    environments: EnvironmentsConfig
    backends: BackendsConfig
    pricing: PricingConfig
    gpu_scenarios: tuple[GpuScenario, ...]
    report: ReportConfig
    operational: OperationalConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_S = TypeVar("_S")


def build_section(cls: type[_S], table: dict[str, Any], context: str) -> _S:
    """Construct one section dataclass, rejecting unknown and missing keys.

    ``fields`` is typed for dataclass instances and dataclass types, which a
    bare ``TypeVar`` cannot express; every call site passes one of this
    module's frozen section dataclasses, so the ``cast`` is the sanctioned
    narrowing (AGENTS.md prefers it to a ``type: ignore``).
    """
    section_fields = fields(cast(Any, cls))
    names = {section_field.name for section_field in section_fields}
    unknown = sorted(set(table) - names)
    if unknown:
        raise ValueError(f"unknown key(s) in [{context}]: {unknown}")
    required = {
        section_field.name
        for section_field in section_fields
        if section_field.default is MISSING and section_field.default_factory is MISSING
    }
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"missing mandatory key(s) in [{context}]: {missing}")
    return cls(**table)


def check_keys(
    table: dict[str, Any],
    expected: tuple[str, ...],
    context: str,
    optional: tuple[str, ...] = (),
) -> None:
    """Reject a table whose sub-tables differ from the expected exact set.

    ``optional`` keys may be present or absent: the unknown-key check runs
    against ``expected`` plus ``optional``; the missing-key check against
    ``expected`` only.
    """
    unknown = sorted(set(table) - set(expected) - set(optional))
    if unknown:
        raise ValueError(f"unknown key(s) in [{context}]: {unknown}")
    missing = sorted(set(expected) - set(table))
    if missing:
        raise ValueError(f"missing mandatory key(s) in [{context}]: {missing}")


def tuplify(table: dict[str, Any], key: str) -> None:
    """Convert a TOML list value to a tuple in place. A missing key is left
    for ``build_section`` to report, so the error names the table and key."""
    if key in table:
        table[key] = tuple(table[key])


def flatten_table(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested TOML tables into dotted keys mapped to leaf values."""
    flat: dict[str, Any] = {}
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_table(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def apply_override(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set one dotted-key override inside the raw parsed TOML tree."""
    node = raw
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def load_config(profile: str = "full", path: Path | str = CONFIG_PATH) -> ExperimentConfig:
    """Parse ``configs/experiment.toml`` into an ``ExperimentConfig``.

    Raises ``ValueError`` for an unknown profile, an unknown top-level key, a
    missing table, an unknown or missing key inside any table (mandatory caps
    included, R12), or a ``[smoke]`` key outside ``SMOKE_SCALE_KEYS``.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    unknown = sorted(set(raw) - set(TOP_LEVEL_KEYS))
    if unknown:
        raise ValueError(f"unknown top-level key(s) in {path}: {unknown}")
    missing_tables = sorted(set(TOP_LEVEL_KEYS) - set(raw))
    if missing_tables:
        raise ValueError(f"missing table(s) in {path}: {missing_tables}")

    smoke_overrides = flatten_table(raw.pop("smoke"))
    disallowed = sorted(set(smoke_overrides) - SMOKE_SCALE_KEYS)
    if disallowed:
        raise ValueError(
            f"[smoke] may override scale counts only ({sorted(SMOKE_SCALE_KEYS)}); "
            f"disallowed key(s): {disallowed}"
        )
    if profile == "smoke":
        for dotted, value in smoke_overrides.items():
            apply_override(raw, dotted, value)

    promotion_table = dict(raw["promotion"])
    tuplify(promotion_table, "cost_band")
    tuplify(promotion_table, "sub_call_band")

    env_table = raw["environments"]
    check_keys(env_table, ("graphwalks", "oolong_pairs"), "environments")
    graphwalks_table = dict(env_table["graphwalks"])
    tuplify(graphwalks_table, "problem_types")
    oolong_table = dict(env_table["oolong_pairs"])
    tuplify(oolong_table, "task_ids")

    backends_table = raw["backends"]
    check_keys(
        backends_table,
        ("runner", "attributor", "proposer"),
        "backends",
        optional=("openrouter", "azure_foundry"),
    )
    openrouter: OpenRouterConfig | None = None
    if "openrouter" in backends_table:
        openrouter_table = dict(backends_table["openrouter"])
        tuplify(openrouter_table, "provider_order")
        openrouter = build_section(OpenRouterConfig, openrouter_table, "backends.openrouter")
    azure_foundry: AzureFoundryConfig | None = None
    if "azure_foundry" in backends_table:
        azure_foundry = build_section(
            AzureFoundryConfig, backends_table["azure_foundry"], "backends.azure_foundry"
        )
    else:
        azure_roles = sorted(
            role for role in CLIENT_ROLES if backends_table[role].get("backend") == "azure_foundry"
        )
        if azure_roles:
            raise ValueError(
                f"missing [backends.azure_foundry] table: role(s) {azure_roles} use the "
                "azure_foundry backend, whose thinking mode must be declared explicitly "
                "(an absent table would send no chat_template_kwargs, silently defaulting "
                "Kimi to thinking mode -- ~10x output cost and <think> markup in outputs)."
            )

    pricing_table = raw["pricing"]
    check_keys(pricing_table, ("promo", "list_price"), "pricing")

    scenarios: list[GpuScenario] = []
    for index, scenario_table in enumerate(raw["gpu_scenarios"]):
        scenario = dict(scenario_table)
        tuplify(scenario, "sensitivity_range")
        scenarios.append(build_section(GpuScenario, scenario, f"gpu_scenarios[{index}]"))

    return ExperimentConfig(
        profile=profile,
        decoding=build_section(DecodingConfig, raw["decoding"], "decoding"),
        splits=build_section(SplitsConfig, raw["splits"], "splits"),
        loop=build_section(LoopConfig, raw["loop"], "loop"),
        promotion=build_section(PromotionSettings, promotion_table, "promotion"),
        caps=build_section(CapsConfig, raw["caps"], "caps"),
        environments=EnvironmentsConfig(
            graphwalks=build_section(GraphWalksConfig, graphwalks_table, "environments.graphwalks"),
            oolong_pairs=build_section(
                OolongPairsConfig, oolong_table, "environments.oolong_pairs"
            ),
        ),
        backends=BackendsConfig(
            runner=build_section(EndpointConfig, backends_table["runner"], "backends.runner"),
            attributor=build_section(
                EndpointConfig, backends_table["attributor"], "backends.attributor"
            ),
            proposer=build_section(EndpointConfig, backends_table["proposer"], "backends.proposer"),
            openrouter=openrouter,
            azure_foundry=azure_foundry,
        ),
        pricing=PricingConfig(
            promo=build_section(PricingTier, pricing_table["promo"], "pricing.promo"),
            list_price=build_section(
                PricingTier, pricing_table["list_price"], "pricing.list_price"
            ),
        ),
        gpu_scenarios=tuple(scenarios),
        report=build_section(ReportConfig, raw["report"], "report"),
        operational=build_section(OperationalConfig, raw["operational"], "operational"),
    )


# ---------------------------------------------------------------------------
# Identity hash (R3; KTD3)
# ---------------------------------------------------------------------------


def identity_hash(config: ExperimentConfig) -> str:
    """Sha256 over the canonical JSON of every behavior-changing value.

    The subset is ``IDENTITY_SECTIONS`` plus ``IDENTITY_OPERATIONAL_KEYS`` from
    ``[operational]``; see the module docstring for what is included and why
    the remaining operational keys are excluded.
    """
    subset: dict[str, Any] = {name: asdict(getattr(config, name)) for name in IDENTITY_SECTIONS}
    subset["operational"] = {
        key: getattr(config.operational, key) for key in IDENTITY_OPERATIONAL_KEYS
    }
    return canonical_json_sha256(subset)


# ---------------------------------------------------------------------------
# Factory helpers: config -> kwargs / objects for the existing constructors
# ---------------------------------------------------------------------------


def sampling_args(config: ExperimentConfig, role: str) -> dict[str, Any]:
    """The decoding config as a client ``sampling_args`` dict for one role.

    ``top_k`` and ``min_p`` ride in ``extra_body`` (the OpenAI surface has no
    top-level parameter for them) only when set -- a None knob is omitted
    entirely. Provider-specific ``extra_body`` branches on the role's backend:
    an openrouter role with a non-empty ``provider_order`` gets the routing
    dict; an azure_foundry role with ``thinking = False`` gets the instant-mode
    ``chat_template_kwargs``. ``max_tokens`` stays ``max_tokens`` -- the OpenAI
    client performs the ``max_completion_tokens`` rename itself.
    """
    if role not in CLIENT_ROLES:
        raise ValueError(f"unknown client role {role!r}; expected one of {CLIENT_ROLES}")
    decoding = config.decoding
    extra_body: dict[str, Any] = {}
    if decoding.top_k is not None:
        extra_body["top_k"] = decoding.top_k
    if decoding.min_p is not None:
        extra_body["min_p"] = decoding.min_p
    endpoint: EndpointConfig = getattr(config.backends, role)
    if endpoint.backend == "openrouter":
        routing = config.backends.openrouter
        if routing is not None and routing.provider_order:
            extra_body["provider"] = {
                "order": list(routing.provider_order),
                "allow_fallbacks": routing.allow_fallbacks,
            }
    elif endpoint.backend == "azure_foundry":
        foundry = config.backends.azure_foundry
        if foundry is not None and foundry.thinking is False:
            extra_body["chat_template_kwargs"] = {"thinking": False}
    return {
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "max_tokens": decoding.max_output_tokens,
        "extra_body": extra_body,
    }


def backend_kwargs_for(config: ExperimentConfig, role: str) -> dict[str, Any]:
    """Client ``backend_kwargs`` for one role: runner, attributor, or proposer.

    An azure_foundry role additionally carries the client's mandatory
    ``pricing`` from ``pricing.list_price``, nested under the single top-level
    key ``"pricing"`` -- deliberate: the driver's sensitive-kwarg scan matches
    top-level names and ``token`` substrings, and pricing is non-secret and
    trace-safe.
    """
    if role not in CLIENT_ROLES:
        raise ValueError(f"unknown client role {role!r}; expected one of {CLIENT_ROLES}")
    endpoint: EndpointConfig = getattr(config.backends, role)
    kwargs: dict[str, Any] = {
        "model_name": endpoint.model,
        "sampling_args": sampling_args(config, role),
    }
    if endpoint.backend == "azure_foundry":
        tier = config.pricing.list_price
        kwargs["pricing"] = {
            "input_per_million": tier.input_per_million,
            "output_per_million": tier.output_per_million,
        }
    return kwargs


# The ``round_config_kwargs`` keys that ``shrlm.optimization.costs.governed_limits``
# owns. A stage that governs its round drops these from the kwargs and lets the
# merged (tighten-only) limits supply them, so the caps are a run's only source
# of limits.
GOVERNED_ROUND_KEYS: tuple[str, ...] = (
    "attempts",
    "max_iterations",
    "max_depth",
    "max_budget",
    "max_timeout",
)


def round_config_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Kwargs for ``shrlm.optimization.driver.RoundConfig``, minus the
    per-round arguments (round_index, harness, instances, verifier, out_dir).

    ``attempts`` is the caps default; mining and evaluation stages override it
    with ``loop.m`` and ``operational.eval_repetitions`` respectively.
    """
    caps = config.caps
    return {
        "backend": config.backends.runner.backend,
        "backend_kwargs": backend_kwargs_for(config, "runner"),
        "attempts": caps.attempts,
        "max_iterations": caps.max_iterations,
        "max_depth": caps.max_depth,
        "max_budget": caps.max_budget,
        "max_timeout": caps.max_timeout,
    }


def validation_caps(config: ExperimentConfig) -> ValidationCaps:
    """The experiment-owned ``ValidationCaps`` (all five limits mandatory)."""
    caps = config.caps
    if caps.max_depth is None:
        raise ValueError(
            "caps.max_depth must be set in configs/experiment.toml to build ValidationCaps: "
            "validation runs under experiment-owned limits, never under S6's declaration."
        )
    return ValidationCaps(
        max_depth=caps.max_depth,
        max_iterations=caps.max_iterations,
        max_budget=caps.max_budget,
        max_timeout=caps.max_timeout,
        candidate_budget=caps.candidate_budget,
    )


def evaluation_config_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Kwargs for ``shrlm.optimization.validation.EvaluationConfig``, minus
    the per-round arguments (splits, verifier, out_dir, round_index)."""
    return {
        "caps": validation_caps(config),
        "repetitions": config.loop.v,
        "backend": config.backends.runner.backend,
        "backend_kwargs": backend_kwargs_for(config, "runner"),
        "workers": config.operational.validation_workers,
    }


def promotion_config(config: ExperimentConfig) -> PromotionConfig:
    """The preregistered ``PromotionConfig``. An absent ``sub_call_band``
    keeps ``PromotionConfig``'s unconstrained default band."""
    settings = config.promotion
    kwargs: dict[str, Any] = {
        "tau_regression": settings.tau_regression,
        "tau_improvement": settings.tau_improvement,
        "cost_band": Band(lower=settings.cost_band[0], upper=settings.cost_band[1]),
    }
    if settings.sub_call_band is not None:
        kwargs["sub_call_band"] = Band(
            lower=settings.sub_call_band[0], upper=settings.sub_call_band[1]
        )
    return PromotionConfig(**kwargs)


def proposer_config(config: ExperimentConfig) -> ProposerConfig:
    """The ``ProposerConfig`` with the experiment's candidate width. Prompt and
    validator versions stay module-owned; transport knobs keep their defaults
    (they change when a response is obtained, never what is produced)."""
    return ProposerConfig(k=config.loop.k)
