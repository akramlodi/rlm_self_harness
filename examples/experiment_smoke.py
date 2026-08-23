"""Tier 2 of the experiment smoke: one shrunk live round plus a small eval, bounded (R13).

Tier 1 is the $0 mock smoke in ``tests/experiment/test_smoke_mock.py``, which
runs the identical pipeline over checked-in split fixtures with a scripted
client and never touches the network. This file is tier 2: the same ``[smoke]``
config profile (R2) against the CONFIGURED runner backend -- the shipped
config serves Kimi-K2.5 from Azure AI Foundry -- with the configured
``sampling_args``, run once by hand to turn the scaffold's plumbing into real
token, time, and cost numbers.

It is opt-in twice over: nothing runs without the configured backend's
environment variables (``_BACKEND_ENV_KEYS``; for ``azure_foundry`` that is
BOTH ``AZURE_API_KEY`` and ``AZURE_FOUNDRY_ENDPOINT``) AND an explicit
``--live`` flag, and the decline path spends nothing and names each missing
variable. CI never runs it.

What it asserts (KTD9, following ``examples/validation_live_smoke.py``):
artifact structure, a usable cost on live responses (provider-reported for
openrouter; the client's own synthesized cost, ``cost_source ==
"synthesized"``, for azure_foundry -- Goal Capsule stop condition (a)), no
reasoning leakage from the instant-mode probe, the persisted
``run_metadata.backend_kwargs`` showing the intended sampling args, stage
coverage across all three model roles (runner, attributor, proposer), and a
report carrying a full-experiment estimate. It NEVER asserts accuracy, pass
counts, or promotion outcomes -- at smoke scale the tau pass-count rule is
meaningless, so model behavior is the model's business.

Spend control (KTD7; hard ceiling $5, CUMULATIVE across every live tier)
    The $5.00 ceiling covers everything the provider-switch plan spends live,
    added together: the U4 pytest live tier (one client check plus a minimal
    driver round, reserved below as ``PYTEST_LIVE_RESERVE_USD``), this
    script's probe (its two calls are counted inside the ungoverned class),
    and the full ``--live`` run. ``check_budget_arithmetic`` proves the whole
    sum before a cent is spent. Every paid call falls in exactly one of two
    classes; the ceiling adds both, then the reserve.

    Governed calls -- every run executed under a ``CandidateSpendBreaker``:

        per-run budget      $0.20   clears one full 262,144-token window of
                                    input ($0.157 at $0.60/1M) plus one
                                    max-length 4,096-token output ($0.012 at
                                    $3.00/1M) with ~1.2x headroom, and the
                                    measured-profile long run (~$0.09, below)
                                    with ~2.2x
        per-breaker budget  $0.24   cumulative, per spend breaker
        breakers            7       t x (1 mining + baseline + k candidates +
                                    merged) + 1 per evaluation condition
        governed ceiling    7 x ($0.24 + $0.20) = $3.08

    A breaker trips only *after* a run pushes cumulative spend past its
    budget, so each breaker's true ceiling is its budget plus one per-run
    budget -- that is the $0.44 term. The per-round breakers are armed afresh
    in each of ``loop.t`` rounds, hence the ``t`` factor (the shipped smoke
    profile runs t = 1, so it is latent there and load-bearing anywhere else).

    Ungoverned calls -- paid model calls no breaker ever sees:

        probe               2       the raw completion plus the client's own
        attribution         18      t x n_in x m records x max_attempts x
                                    transport_retries (n_in = 2 live, below)
        proposal            9       t x max_attempts x transport_retries
        per call            $0.0418 = UNGOVERNED_INPUT_TOKENS (49,152) x
                                    $0.60/1M in + 4,096 x $3.00/1M out
        ungoverned ceiling  29 x $0.0417792 = $1.2116

    Reserved for the U4 pytest live tier (client check ~$0.01 + minimal
    driver round <$0.50, rounded up):

        U4 reserve          $0.60

        cumulative ceiling  $3.08 + $1.2116 + $0.60 = $4.8916 < $5.00

    Why ``UNGOVERNED_INPUT_TOKENS`` and not the full context window: at the
    Kimi list price a full 262,144-token input bound prices 29 calls at $4.92
    on its own. The ungoverned calls are single, non-REPL completions whose
    prompts this repo constructs mechanically and char-caps: the probe prompt
    is a constant in this file (~30 chars); an attribution prompt is the
    attributor system prompt (~6.5k chars) plus a digest hard-capped at
    12,000 budget chars plus a sub-call table bounded at 40 rows of two
    200-char previews (~18.4k chars) plus a ~2k header, ~39k chars total; a
    proposal prompt is a fixed template plus char-capped skill surfaces
    (SKILL_TOTAL_MAX_CHARS = 16,000) plus at most n_in x m pattern blocks,
    well under that. A BPE token never encodes less than one character, so
    49,152 tokens (>25% above the largest constructed prompt's char cap) is a
    true bound on the input side, and ``max_tokens`` bounds the output.
    ``caps.max_budget`` is deliberately NOT the per-call bound here: at $0.20
    x 29 calls it would claim $5.80 of the ceiling on its own.

    Caps AND the live instance counts are identity keys (R3), so editing any
    ``LIVE_*`` constant below means a fresh ``--out-dir``: an existing
    experiment refuses to resume under changed caps or split sizes.

Live scale (LIVE_* overrides; identity note above)
    The live smoke shrinks the smoke profile's counts, not its semantics:
    n_in 3 -> 2 (cuts the worst-case attribution transport calls from 27 to
    18 -- ungoverned headroom the proof needs), test_short 4 -> 2 and
    oolong n_short 4 -> 2 with test_long 2 -> 1 and oolong n_long 2 -> 1 (so
    one evaluation condition's cumulative spend before its final long set --
    ~4 short runs x ~$0.02 + one GraphWalks long ~$0.09, ~$0.17 -- clears the
    $0.24 breaker budget instead of tripping it before the OOLONG long run).
    Decoding, promotion rule, environments, loop shape (m, v, k, t), and
    n_ho are the shipped smoke profile, byte-identical.

Long runs (KTD6, re-derived for Kimi-K2.5 pricing $0.60/$3.00 per 1M)
    Both environments contribute long instances. The per-run budget is sized
    so at least one uncapped long run completes per environment: one full
    pass of the 262,144-token OOLONG window ($0.157 in) plus one max-length
    4,096-token output ($0.012) is ~$0.17 against the $0.20 budget, and the
    measured profile is far smaller -- prior-provider smoke means
    ($0.0024-$0.0134/run) imply ~130k effective input tokens on a long run,
    ~$0.09 at Kimi rates. The old sizing figure -- ~3 context passes plus 30
    max-length outputs -- reprices to ~$0.84 at Kimi rates and is
    deliberately NOT cleared: a budget clearing it would put the governed
    ceiling alone past $5 (7 breakers x >$1). A run that heavy is
    budget-terminated, lands flagged ``usage_lower_bound`` -- named, never
    silently zeroed -- and the KTD6 gate below adjudicates. GraphWalks long
    instances range up to ~1M tokens; a maximal draw exceeds the per-run
    budget by construction and lands as a lower bound, which the gate
    likewise catches if the whole long sample ends up censored.

    An environment whose long runs were *all* capped (or all skipped by a
    tripped breaker) fails the smoke: long-run cost is superlinear, so a fully
    censored long sample cannot support the API-vs-GPU recommendation, and a
    green smoke over one would certify nothing. The failure is raised only
    after the report is built, written, and printed, so the operator keeps
    every measured number.

Usage:
    # ~$0.01: the R10 execution-note probe, run this FIRST.
    AZURE_API_KEY=... AZURE_FOUNDRY_ENDPOINT=... \\
        uv run python examples/experiment_smoke.py --probe

    # bounded by the arithmetic above, up to a few hours: the full tier-2 smoke.
    AZURE_API_KEY=... AZURE_FOUNDRY_ENDPOINT=... \\
        uv run python examples/experiment_smoke.py --live

The live run materializes real splits, so it downloads the pinned GraphWalks
parquets (including the 256k-1M long file) and streams the OOLONG-Pairs
upstream dataset -- install the ``graphwalks`` and ``oolong`` extras first.
Artifacts land under ``--out-dir`` (default ``./experiment_smoke``); the run is
resumable, so a re-invocation completes only what is missing.
"""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from rlm.clients.openai import _merge_extra_body, _normalize_sampling_args
from shrlm.experiment.config import (
    CLIENT_ROLES,
    ExperimentConfig,
    backend_kwargs_for,
    load_config,
    sampling_args,
)
from shrlm.experiment.evaluation import (
    DEFAULT_CONDITIONS,
    EVAL_DIR,
    EVAL_SUMMARY_FILENAME,
    ROLE_TEST,
    run_evaluation,
)
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    FROZEN_DIR,
    FROZEN_HARNESS_FILENAME,
    PROPOSALS_MARKER_FILENAME,
    ROUND_MARKER_FILENAME,
    experiment_round_dir,
    run_experiment,
)
from shrlm.experiment.report import build_report, render_markdown, write_report
from shrlm.experiment.splits import LENGTHS, MANIFEST_FILE, SPLITS_DIR, split_plan
from shrlm.experiment.usage import (
    STAGE_USAGE_FILE,
    read_jsonl,
    read_stage_usage,
    read_usage_records,
)
from shrlm.optimization.attribution import DEFAULT_MAX_ATTEMPTS as ATTRIBUTION_MAX_ATTEMPTS
from shrlm.optimization.attribution import (
    DEFAULT_TRANSPORT_RETRIES as ATTRIBUTION_TRANSPORT_RETRIES,
)
from shrlm.optimization.driver import _BACKEND_ENV_KEYS
from shrlm.optimization.proposal import DEFAULT_MAX_ATTEMPTS as PROPOSAL_MAX_ATTEMPTS
from shrlm.optimization.proposal import DEFAULT_TRANSPORT_RETRIES as PROPOSAL_TRANSPORT_RETRIES

load_dotenv()

# The hard ceiling from the plan's Goal Capsule stop condition (c). It is
# CUMULATIVE across every live tier of the provider-switch plan: the U4 pytest
# live tier (reserved below), this script's probe, and the full --live run.
SPEND_CEILING_USD = 5.0

# Worst case reserved for the U4 pytest live tier (one client check ~$0.01
# plus a minimal driver round <$0.50, rounded up). Counted into the proof so
# $5 stays a ceiling on the plan's total live spend, never a per-tier bound.
PYTEST_LIVE_RESERVE_USD = 0.60

# Live-smoke caps, sized by the module docstring's re-derived Kimi arithmetic.
# They replace the smoke profile's pilot-sized caps; decoding, promotion rule,
# environments, and loop shape stay the shipped smoke profile. The per-run
# budget is load-bearing for KTD6 (it must clear one full 262,144-token input
# pass plus one max-length output, ~$0.17), so the ceiling is fitted by the
# candidate budget and the LIVE_* instance counts below.
LIVE_MAX_BUDGET_USD = 0.20
LIVE_CANDIDATE_BUDGET_USD = 0.24
LIVE_MAX_TIMEOUT_SECONDS = 1200.0

# Live-smoke instance counts (identity keys like the caps, R3 -- edits mean a
# fresh --out-dir). n_in bounds the worst-case attribution transport calls;
# the test-set counts keep one evaluation condition's cumulative spend inside
# its breaker budget while still owing each environment its KTD6 long run.
LIVE_HELD_IN = 2  # splits.n_in: 3 -> 2
LIVE_TEST_SHORT = 2  # splits.test_short: 4 -> 2
LIVE_TEST_LONG = 1  # splits.test_long: 2 -> 1
LIVE_OOLONG_SHORT = 2  # environments.oolong_pairs.n_short: 4 -> 2
LIVE_OOLONG_LONG = 1  # environments.oolong_pairs.n_long: 2 -> 1

# Spend breakers armed per round: one for mining, one per validation subject
# (baseline + k candidates + the merged harness). Evaluation arms one more per
# condition, once for the whole invocation.
MINING_BREAKERS = 1
VALIDATION_FIXED_BREAKERS = 2  # baseline and the merged re-evaluation

# The ungoverned paid calls. ``probe`` issues two (the raw completion, then the
# client's own), and neither the attributor's nor the proposer's completions
# are wrapped in a CandidateSpendBreaker -- they are per-stage LM calls, not
# governed runs -- so the ceiling must allow for them explicitly. The attempt
# and retry counts are imported from the stage modules whose defaults the
# orchestrator actually constructs, so a change there moves this arithmetic.
PROBE_CALLS = 2

# Input-token bound for one ungoverned completion. Every ungoverned prompt is
# constructed mechanically and char-capped by this repo (see the module
# docstring's accounting: the largest, an attribution prompt, tops out ~39k
# chars), and one BPE token never encodes less than one character, so this is
# a true bound with >25% margin -- unlike the model's full 262,144-token
# context window, which at Kimi list prices would price the ungoverned class
# alone past the $5 ceiling. The output side is bounded by
# decoding.max_output_tokens (sent as ``max_tokens``).
UNGOVERNED_INPUT_TOKENS = 49_152

# The length whose per-run means the extrapolation is most sensitive to (KTD6).
LONG = LENGTHS[1]

PROBE_PROMPT = "Reply with the single word OK."

# A probe completion grossly longer than this trivial prompt warrants is a
# reasoning signal (thinking tokens billed as completion tokens) even when no
# marker survives into the payload: the expected answer is a handful of
# tokens, so this ceiling is already >10x generous.
PROBE_COMPLETION_TOKEN_CEILING = 64


class SmokeError(RuntimeError):
    """The live smoke's own precondition or structural check failed."""


# ---------------------------------------------------------------------------
# Configuration and budget arithmetic
# ---------------------------------------------------------------------------


def live_config() -> ExperimentConfig:
    """The shipped ``[smoke]`` profile with live-smoke caps and counts (R2).

    Semantics -- decoding, promotion rule, environments, backends, loop shape
    -- are byte-identical to the smoke profile; only spend caps and instance
    counts move (see the module docstring's "Live scale" section). Caps and
    split sizes are identity keys (R3), so this config refuses to resume an
    experiment directory the plain smoke profile created, and vice versa.
    """
    config = load_config("smoke")
    return replace(
        config,
        caps=replace(
            config.caps,
            max_budget=LIVE_MAX_BUDGET_USD,
            max_timeout=LIVE_MAX_TIMEOUT_SECONDS,
            candidate_budget=LIVE_CANDIDATE_BUDGET_USD,
        ),
        splits=replace(
            config.splits,
            n_in=LIVE_HELD_IN,
            test_short=LIVE_TEST_SHORT,
            test_long=LIVE_TEST_LONG,
        ),
        environments=replace(
            config.environments,
            oolong_pairs=replace(
                config.environments.oolong_pairs,
                n_short=LIVE_OOLONG_SHORT,
                n_long=LIVE_OOLONG_LONG,
            ),
        ),
    )


def breaker_count(config: ExperimentConfig, conditions: int = len(DEFAULT_CONDITIONS)) -> int:
    """How many independent spend breakers one smoke invocation arms.

    Mining, the two fixed validation subjects, and the ``k`` candidates are
    armed once per optimization round, so they carry the ``loop.t`` factor;
    the evaluation conditions are armed once for the whole invocation.
    """
    per_round = MINING_BREAKERS + VALIDATION_FIXED_BREAKERS + config.loop.k
    return config.loop.t * per_round + conditions


def ungoverned_call_count(config: ExperimentConfig) -> int:
    """Every paid model call that no ``CandidateSpendBreaker`` governs.

    The probe's two calls, one attribution attempt per mining run (worst case:
    every run fails and every attempt is retried to its bound), and the
    proposal batch's attempts, both per round.
    """
    attribution = (
        config.loop.t
        * config.splits.n_in
        * config.loop.m
        * ATTRIBUTION_MAX_ATTEMPTS
        * ATTRIBUTION_TRANSPORT_RETRIES
    )
    proposal = config.loop.t * PROPOSAL_MAX_ATTEMPTS * PROPOSAL_TRANSPORT_RETRIES
    return PROBE_CALLS + attribution + proposal


def ungoverned_call_ceiling(config: ExperimentConfig) -> float:
    """The most one ungoverned completion can cost, in USD.

    ``UNGOVERNED_INPUT_TOKENS`` of input at the configured *list* (never
    promo) input price, plus ``decoding.max_output_tokens`` of output at the
    list output price. Both sides are hard limits on a single completion --
    the input side because every ungoverned prompt is char-capped by
    construction (see the constant's comment and the module docstring) and a
    token never encodes less than a character, the output side because
    ``max_tokens`` is sent on every call -- which is what makes this a bound
    rather than an estimate. See the module docstring for why
    ``caps.max_budget`` is not used here.
    """
    price = config.pricing.list_price
    input_usd = UNGOVERNED_INPUT_TOKENS * price.input_per_million / 1_000_000
    output_usd = config.decoding.max_output_tokens * price.output_per_million / 1_000_000
    return input_usd + output_usd


def spend_ceiling(config: ExperimentConfig, conditions: int = len(DEFAULT_CONDITIONS)) -> float:
    """The worst-case USD one smoke invocation can spend, governed and not.

    Each breaker admits its cumulative ``candidate_budget`` and trips only
    after the run that crosses it, so one more per-run ``max_budget`` can land
    on top of every breaker. The probe, attribution, and proposal calls are
    governed by no breaker at all and are priced per call on top.
    """
    per_breaker = config.caps.candidate_budget + config.caps.max_budget
    governed = breaker_count(config, conditions) * per_breaker
    ungoverned = ungoverned_call_count(config) * ungoverned_call_ceiling(config)
    return governed + ungoverned


def check_budget_arithmetic(config: ExperimentConfig) -> float:
    """Refuse to spend anything unless the worst case stays under the ceiling.

    The proven figure is CUMULATIVE across the plan's live tiers: this
    invocation's worst case (``spend_ceiling``) plus the U4 pytest live
    tier's reserve, against the one $5 ceiling. Returns the cumulative
    ceiling.
    """
    cumulative = spend_ceiling(config) + PYTEST_LIVE_RESERVE_USD
    if cumulative >= SPEND_CEILING_USD:
        raise SmokeError(
            f"the configured budgets admit up to ${cumulative:.2f} cumulatively "
            f"({breaker_count(config)} breakers x (${config.caps.candidate_budget} + "
            f"${config.caps.max_budget}) + {ungoverned_call_count(config)} ungoverned call(s) "
            f"x ${ungoverned_call_ceiling(config):.4f} + the ${PYTEST_LIVE_RESERVE_USD:.2f} "
            f"U4 pytest live reserve), which does not clear the ${SPEND_CEILING_USD} "
            "ceiling; lower caps.candidate_budget (raising caps.max_budget is what the KTD6 "
            "long runs need, so cut the cumulative budget first) before running live."
        )
    return cumulative


# ---------------------------------------------------------------------------
# The live probe (U1 execution note): usage.cost and extra_body, for ~$0.001
# ---------------------------------------------------------------------------


def runner_client(config: ExperimentConfig) -> BaseLM:
    """The configured runner endpoint's client (credentials from the env only)."""
    return get_client(config.backends.runner.backend, backend_kwargs_for(config, "runner"))


def raw_completion(lm: Any, config: ExperimentConfig, prompt: str) -> Any:
    """One chat completion built exactly as ``OpenAIClient.completion`` builds it.

    The client returns only the message content, but the probe must read the
    response's ``usage.cost`` and ``provider`` fields, so it issues the request
    through the client's own request builders rather than reconstructing (and
    drifting from) the parameter shape by hand.
    """
    args = sampling_args(config, "runner")
    return lm.client.chat.completions.create(
        model=config.backends.runner.model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=_merge_extra_body({}, args),
        **_normalize_sampling_args(args),
    )


def check_probe_reasoning(payload: dict[str, Any]) -> None:
    """FAIL (never record) on any reasoning signal in the probe response (R10).

    The decoding config requests instant (non-thinking) mode via
    ``chat_template_kwargs``; a route that ignores it changes cost and
    REPL-output parsing materially, which is the KTD6 contingency -- surfaced
    here, before anything larger spends. Detected signals: ``<think>`` markup
    in content, a non-empty ``reasoning_content``/``reasoning`` field on the
    message (``model_extra`` fields appear inline in ``model_dump``), a
    nonzero reasoning-token count in usage details, or completion tokens
    grossly exceeding what the trivial probe prompt warrants.
    """
    choices = payload.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    signals: list[str] = []
    if "<think>" in content:
        signals.append("content carries <think> markup")
    for key, value in message.items():
        if "reasoning" in key and value:
            signals.append(f"message.{key} is non-empty")
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details, dict) and details.get("reasoning_tokens"):
        signals.append(f"usage reports {details['reasoning_tokens']} reasoning token(s)")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > PROBE_COMPLETION_TOKEN_CEILING:
        signals.append(
            f"{completion_tokens} completion tokens for a trivial prompt "
            f"(ceiling {PROBE_COMPLETION_TOKEN_CEILING})"
        )
    if signals:
        raise SmokeError(
            "the probe response shows reasoning signal(s): "
            + "; ".join(signals)
            + ". Instant (non-thinking) mode is not being honored -- reasoning tokens "
            "change cost and REPL-output parsing materially (KTD6 contingency). Stop and "
            "surface before running anything larger; do not strip <think> content."
        )


def probe(config: ExperimentConfig) -> dict[str, Any]:
    """Two tiny live calls: cost usable, sampling args accepted, no reasoning.

    The raw completion is issued through the client's own request builders, so
    a rejected decoding argument surfaces as an HTTP error right here; the
    second call travels the client's own ``completion`` path -- the path every
    persisted run's cost actually travels.

    Raises:
        SmokeError: No usable cost -- the spend breaker, the cost band, and
            ``acceptance_inputs`` all stand on it, so the whole scaffold stops
            here rather than after spending more. For openrouter that means a
            non-null raw ``usage.cost``; for azure_foundry it means the
            client's own synthesized cost (positive ``total_cost`` with
            ``cost_source == "synthesized"``). Also raised on any reasoning
            signal (``check_probe_reasoning``).
    """
    backend = config.backends.runner.backend
    lm = runner_client(config)
    response = raw_completion(lm, config, PROBE_PROMPT)
    payload = response.model_dump()
    check_probe_reasoning(payload)
    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    provider = payload.get("provider")

    # The path every persisted run's cost actually travels: the client's own
    # extraction (and, for azure_foundry, its synthesis), not the raw field.
    client_content = lm.completion(PROBE_PROMPT)
    if "<think>" in client_content:
        raise SmokeError(
            "the client-path probe completion carries <think> markup; instant mode is not "
            "being honored (KTD6 contingency). Stop and surface before running anything "
            "larger."
        )
    last = lm.get_last_usage()
    client_cost = last.total_cost
    cost_source = getattr(last, "cost_source", None)

    if backend == "azure_foundry":
        if client_cost is None or client_cost <= 0 or cost_source != "synthesized":
            raise SmokeError(
                f"the azure_foundry client read total_cost={client_cost!r} with "
                f"cost_source={cost_source!r} for {config.backends.runner.model}; the spend "
                "breaker, the promotion cost band, and acceptance_inputs all require a "
                "positive synthesized cost (token counts x configured [pricing]). Stop and "
                "fix the deployment or pricing before running anything larger."
            )
    elif cost is None or client_cost is None:
        raise SmokeError(
            f"OpenRouter returned usage.cost={cost!r} (client read {client_cost!r}) for "
            f"{config.backends.runner.model} via provider {provider!r}. The spend breaker, "
            "the promotion cost band, and acceptance_inputs all require a non-null cost "
            "(BYOK and zero-cost routes do not report one). Stop and re-route before "
            "running anything larger."
        )
    return {
        "provider": provider,
        "usage_cost": float(cost) if cost is not None else None,
        "client_cost": float(client_cost),
        "cost_source": cost_source,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "sampling_args": sampling_args(config, "runner"),
    }


def check_provider(config: ExperimentConfig, provider: str | None) -> str:
    """The provider check: allow-list for openrouter, deployment note otherwise.

    OpenRouter routes per request, and a provider that silently drops
    ``extra_body`` sampling args makes the backbone non-stationary; pinning
    ``backends.openrouter.provider_order`` is what keeps the preregistered
    decoding invariant testable. Until it is pinned, the smoke records what it
    observed so the pilot can pin it. A non-openrouter backend (azure_foundry)
    serves exactly one deployment, so there is nothing to allow-list; the
    returned line names the backend and the configured deployment for stdout
    only -- it is never persisted, and the endpoint host is never printed.
    """
    backend = config.backends.runner.backend
    if backend != "openrouter":
        return (
            f"backend {backend!r} serves the single configured deployment "
            f"{config.backends.runner.model!r}; provider allow-lists apply to openrouter only"
        )
    routing = config.backends.openrouter
    allowed = routing.provider_order if routing is not None else ()
    if not allowed:
        return (
            f"provider allow-list is empty (backends.openrouter.provider_order): recording "
            f"the observed provider {provider!r} rather than enforcing it -- pin it before "
            "the main run"
        )
    if provider not in allowed:
        raise SmokeError(
            f"the response was served by provider {provider!r}, which is not in the "
            f"configured allow-list {list(allowed)}; a routed-away provider can silently "
            "drop extra_body sampling args and make the backbone non-stationary."
        )
    return f"provider {provider!r} is in the configured allow-list {list(allowed)}"


# ---------------------------------------------------------------------------
# Structural checks over the persisted experiment
# ---------------------------------------------------------------------------


def manifests(out_dir: Path) -> list[Path]:
    """Every persisted run manifest under the experiment directory."""
    return sorted(out_dir.rglob("runs.jsonl"))


def check_directory_contract(out_dir: Path) -> None:
    """The U5/U6 directory contract, from the root markers down to a trace."""
    required = [
        out_dir / CONFIG_FILENAME,
        out_dir / SPLITS_DIR / MANIFEST_FILE,
        out_dir / STAGE_USAGE_FILE,
        out_dir / FROZEN_DIR / FROZEN_HARNESS_FILENAME,
        experiment_round_dir(out_dir, 1) / ROUND_MARKER_FILENAME,
        out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SmokeError(f"the experiment directory is missing required artifact(s): {missing}")
    if not manifests(out_dir):
        raise SmokeError(f"{out_dir} holds no run manifest; nothing was measured")


def check_costs_present(out_dir: Path) -> tuple[int, int]:
    """Every *completed* run carries a non-null cost (stop condition (a), at scale).

    A run the experiment terminated itself -- per-run budget or timeout -- has
    no cost to report: the runtime raised before any usage came back, which is
    why ``_partial_completion`` persists it with ``usage_lower_bound`` set.
    Those are expected, already flagged, and counted separately here.

    The condition this guards is the other one: a run that ran to completion
    and still carries no cost means the runner backend stopped producing a
    usable cost -- provider-reported or synthesized client-side (the
    manifest's ``cost_source`` key records which) -- and every spend breaker
    downstream of it is blind. Conflating the two would fail an honest run
    over its own timeout while hiding the failure that actually matters.
    """
    blind: list[str] = []
    terminated = 0
    total = 0
    for path in manifests(out_dir):
        for entry in read_jsonl(path):
            total += 1
            if entry.get("cost") is not None:
                continue
            if entry.get("usage_lower_bound"):
                terminated += 1
                continue
            blind.append(f"{path}:{entry['run_id']}")
    if blind:
        raise SmokeError(
            f"{len(blind)} completed run(s) carry no cost, e.g. {blind[:3]}; the runner "
            "backend stopped producing a usable cost (provider-reported or synthesized -- "
            "see the manifest's cost_source key) and the spend breaker is now blind."
        )
    return total, terminated


def check_sampling_args(out_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    """The persisted ``run_metadata.backend_kwargs`` show the intended decoding.

    Reads one trace (they all carry the same run metadata) and compares its
    sampling args against ``sampling_args(config, "runner")`` key by key.
    """
    expected = sampling_args(config, "runner")
    manifest_path = manifests(out_dir)[0]
    entry = read_jsonl(manifest_path)[0]
    trace = json.loads((manifest_path.parent / entry["trace_path"]).read_text())
    metadata = trace.get("metadata")
    if metadata is None:
        raise SmokeError(f"trace for {entry['run_id']} carries no trajectory metadata")
    persisted = metadata["run_metadata"]["backend_kwargs"].get("sampling_args")
    if persisted is None:
        raise SmokeError(
            f"trace for {entry['run_id']} persists no sampling_args in "
            "run_metadata.backend_kwargs; the decoding invariant is unverifiable"
        )
    mismatched = {
        key: (value, persisted.get(key))
        for key, value in expected.items()
        if persisted.get(key) != value
    }
    if mismatched:
        raise SmokeError(
            f"persisted sampling args differ from the configured ones "
            f"(key: (configured, persisted)): {mismatched}"
        )
    return persisted


# Every metered stage a green live smoke must have driven. Runner runs cover
# mining/validation/eval; attribution and proposal are the other two model
# roles, so their presence is what proves all three roles hit the backend.
REQUIRED_STAGES = ("mining", "attribution", "proposal", "validation", "eval")


def check_stage_coverage(out_dir: Path) -> dict[str, int]:
    """All three model roles measurably ran: stages metered, artifacts present.

    A green smoke must prove "all three roles on the configured backend" with
    artifacts, not config claims: every stage in ``REQUIRED_STAGES`` has at
    least one ``stage_usage.jsonl`` record with positive input AND output
    tokens (the attributor and proposer records are the attribution/proposal
    roles' usage), the mining bundle exists with a non-empty pattern list (the
    attribution stage's artifact), and the proposals marker exists (the
    proposal stage's artifact). A live round whose mining runs all pass never
    reaches attribution or proposal -- that fails here loudly, by stage name:
    a smoke that exercised only one role certifies nothing about the others.

    Returns:
        Stage name -> number of usage records, for the caller to print.
    """
    records = read_usage_records(out_dir / STAGE_USAGE_FILE).records
    stage_of = {record["stage_work_id"]: str(record["stage"]) for record in records}
    counts: dict[str, int] = {}
    for stage in stage_of.values():
        counts[stage] = counts.get(stage, 0) + 1
    missing = sorted(set(REQUIRED_STAGES) - set(counts))
    if missing:
        raise SmokeError(
            f"stage_usage.jsonl records no usage for stage(s) {missing}; a green smoke "
            "must drive every stage (a round whose mining runs all pass never exercises "
            "the attributor or proposer -- re-run in a fresh --out-dir)."
        )
    for work_id, usage in read_stage_usage(out_dir / STAGE_USAGE_FILE).items():
        if usage.input_tokens <= 0 or usage.output_tokens <= 0:
            raise SmokeError(
                f"stage work {work_id!r} ({stage_of.get(work_id)}) metered no tokens "
                f"({usage.input_tokens} in / {usage.output_tokens} out); its role never "
                "measurably hit the backend."
            )
    round_path = experiment_round_dir(out_dir, 1)
    bundle_path = round_path / "mining" / "round_01" / "bundle.json"
    if not bundle_path.exists():
        raise SmokeError(f"{bundle_path} is missing; attribution left no bundle artifact")
    if not json.loads(bundle_path.read_text()).get("patterns"):
        raise SmokeError(
            f"{bundle_path} holds no mined pattern; attribution produced an empty artifact "
            "and the proposal stage had nothing real to work from."
        )
    proposals_marker = round_path / PROPOSALS_MARKER_FILENAME
    if not proposals_marker.exists():
        raise SmokeError(f"{proposals_marker} is missing; the proposal stage left no artifact")
    return counts


def long_test_environments(config: ExperimentConfig) -> tuple[str, ...]:
    """Every environment the config plans a long test split for.

    Each one owes the run at least one uncapped long run (KTD6), so this is
    the set ``long_run_coverage`` is seeded with -- an environment the
    evaluation summary never mentions is a fully censored long sample, not an
    environment to skip over.
    """
    plan = split_plan(config)
    return tuple(
        sorted(
            environment
            for environment, lengths in plan.items()
            if ROLE_TEST in lengths.get(LONG, {})
        )
    )


def long_run_coverage(config: ExperimentConfig, out_dir: Path) -> dict[str, dict[str, int]]:
    """Per environment: how many long evaluation runs landed, and how many uncapped.

    KTD6 wants at least one *uncapped* long run per environment -- a
    terminated (lower-bound) long run understates the per-run mean the
    extrapolation keys off, and a run the spend breaker skipped never happened
    at all. Seeded from the config, so every environment that owes a long run
    appears with zeros even when the summary holds no aggregate for it.
    """
    summary = json.loads((out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_text())
    coverage: dict[str, dict[str, int]] = {
        environment: {"runs": 0, "uncapped": 0} for environment in long_test_environments(config)
    }
    for condition in summary["conditions"].values():
        for aggregate in condition["test_sets"].values():
            if aggregate["length"] != LONG:
                continue
            entry = coverage.setdefault(aggregate["environment"], {"runs": 0, "uncapped": 0})
            entry["runs"] += int(aggregate["n_runs"])
            if not aggregate["usage_lower_bound"]:
                entry["uncapped"] += int(aggregate["n_runs"])
    return coverage


def check_long_run_coverage(config: ExperimentConfig, coverage: dict[str, dict[str, int]]) -> None:
    """Refuse to pass a smoke whose long sample is entirely censored (KTD6).

    The plan's Definition of Done and Success Criteria require at least one
    uncapped long run per environment. Long-run cost is superlinear, so a long
    sample in which every run was terminated by a cap (or skipped by a tripped
    breaker) supports no API-vs-GPU recommendation at all -- passing on one
    would be exactly the silent pass this script exists to prevent.

    Called only after the report has been built, written, and printed: the
    measured numbers are worth keeping even when the run cannot be certified.

    Raises:
        SmokeError: Any environment landed zero uncapped long runs.
    """
    censored = sorted(
        environment for environment, counts in coverage.items() if counts["uncapped"] == 0
    )
    if not censored:
        return
    detail = "; ".join(
        f"{environment}: {coverage[environment]['runs']} long run(s) recorded, 0 uncapped"
        for environment in censored
    )
    raise SmokeError(
        f"no uncapped long run landed for {len(censored)} environment(s) -- {detail}. "
        "KTD6 and the plan's Definition of Done require at least one uncapped long run per "
        "environment: long-run cost is superlinear, so a fully capped (lower-bound) or "
        "skipped long sample cannot support the API-vs-GPU recommendation, and the report "
        "above -- complete and written to disk, every measured number in it still valid -- "
        "may not be read as one. To fix: raise LIVE_MAX_BUDGET_USD (now "
        f"${config.caps.max_budget}) so a long run completes, and LIVE_CANDIDATE_BUDGET_USD "
        f"(now ${config.caps.candidate_budget}) if a condition went over budget before its "
        f"long sets ran, keep spend_ceiling under the ${SPEND_CEILING_USD:.2f} ceiling, and "
        "re-run in a FRESH --out-dir: caps are identity keys (R3), so an existing experiment "
        "refuses to resume under changed caps."
    )


def measured_spend(out_dir: Path) -> float:
    """Total USD across every metered stage (runs plus proposer/attributor calls)."""
    return sum(total.cost for total in read_stage_usage(out_dir / STAGE_USAGE_FILE).values())


# ---------------------------------------------------------------------------
# The two entry points
# ---------------------------------------------------------------------------


def run_probe(config: ExperimentConfig) -> int:
    print(f"Probing {config.backends.runner.model} on {config.backends.runner.backend}...")
    result = probe(config)
    raw_cost = "n/a" if result["usage_cost"] is None else f"${result['usage_cost']:.6f}"
    print(f"  sampling args accepted: {result['sampling_args']}")
    print(
        f"  cost: client read ${result['client_cost']:.6f} "
        f"(source {result['cost_source']!r}; raw usage.cost {raw_cost})"
    )
    print(f"  tokens:      {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"  {check_provider(config, result['provider'])}")
    print(
        "PROBE PASSED: a usable cost is reported, no reasoning signal, and the configured "
        "sampling args are accepted."
    )
    return 0


def run_live(config: ExperimentConfig, out_dir: Path) -> int:
    ceiling = check_budget_arithmetic(config)
    print(
        f"Live smoke: profile {config.profile}, model {config.backends.runner.model}, "
        f"cumulative worst case ${ceiling:.2f} = {breaker_count(config)} breaker(s) x "
        f"${config.caps.candidate_budget + config.caps.max_budget:.2f} + "
        f"{ungoverned_call_count(config)} ungoverned call(s) x "
        f"${ungoverned_call_ceiling(config):.4f} + ${PYTEST_LIVE_RESERVE_USD:.2f} U4 reserve "
        f"(ceiling ${SPEND_CEILING_USD:.2f})."
    )

    probed = probe(config)
    print(
        f"Probe: client cost ${probed['client_cost']:.6f} "
        f"(source {probed['cost_source']!r}); {check_provider(config, probed['provider'])}"
    )

    print(f"Running one optimization round into {out_dir} (this downloads the pinned splits)...")
    experiment = run_experiment(config, out_dir)
    print(f"  stopped on {experiment.stopped}; frozen harness {experiment.final_harness_hash}")

    print(f"Evaluating conditions {list(DEFAULT_CONDITIONS)} on the frozen test splits...")
    evaluation = run_evaluation(config, DEFAULT_CONDITIONS, out_dir)
    for condition in evaluation.conditions:
        print(f"  {condition.condition_id}: outcome {condition.outcome}, ${condition.spent:.4f}")

    check_directory_contract(out_dir)
    n_runs, n_terminated = check_costs_present(out_dir)
    persisted_args = check_sampling_args(out_dir, config)
    stage_counts = check_stage_coverage(out_dir)
    print(
        "Stage coverage (all three roles measurably ran): "
        + ", ".join(f"{stage}={stage_counts[stage]}" for stage in REQUIRED_STAGES)
    )
    completed = n_runs - n_terminated
    note = (
        ""
        if not n_terminated
        else f" {n_terminated} run(s) the experiment terminated carry no cost and are lower bounds."
    )
    print(f"Artifacts: {n_runs} persisted run(s), {completed} carrying a non-null cost.{note}")
    print(f"Persisted run_metadata.backend_kwargs.sampling_args: {persisted_args}")

    # Read once, reported here and adjudicated after the report is on disk.
    coverage = long_run_coverage(config, out_dir)
    for environment, counts in sorted(coverage.items()):
        note = (
            ""
            if counts["uncapped"]
            else "  WARNING: no uncapped long run -- this smoke FAILS after the report below"
        )
        print(
            f"Long coverage {environment}: {counts['runs']} run(s), {counts['uncapped']} uncapped.{note}"
        )

    report = build_report(config, out_dir)
    path = write_report(report)
    sys.stdout.write(render_markdown(report))
    print(f"\nWrote {path}")
    if report.point.total_tokens <= 0.0 or not report.scenarios:
        raise SmokeError(
            "the report projected no full-experiment tokens or priced no scenario; the "
            "measured per-run means never reached the extrapolation"
        )
    cheapest = report.scenarios[0]
    print(
        f"Full-experiment estimate: {report.point.total_tokens:,.0f} tokens, cheapest "
        f"scenario {cheapest.name} at ${cheapest.usd_point:,.2f} "
        f"({report.recommendation.status})"
    )

    spent = measured_spend(out_dir)
    print(f"Total measured spend: ${spent:.4f} (ceiling ${SPEND_CEILING_USD:.2f})")
    if spent >= SPEND_CEILING_USD:
        raise SmokeError(f"spend ${spent:.4f} breached the ${SPEND_CEILING_USD} ceiling")

    # Last, and only here: the report exists on disk and has been printed in
    # full, so a censored long sample fails the run without costing the
    # operator a single measured number.
    check_long_run_coverage(config, coverage)
    print(
        "SMOKE PASSED: artifacts complete, costs present, report carries a full-experiment estimate."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python examples/experiment_smoke.py",
        description="Live (tier 2) smoke for the experiment scaffold; opt-in and bounded.",
    )
    parser.add_argument(
        "--live", action="store_true", help="run the full live smoke (spends real money)"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run only the ~$0.001 probe: usage.cost present, sampling args accepted",
    )
    parser.add_argument(
        "--out-dir", default="./experiment_smoke", help="where the experiment artifacts land"
    )
    args = parser.parse_args(argv)

    if not (args.live or args.probe):
        print(
            "Declining to run: this smoke spends real money. Re-run with --probe (~$0.01) "
            f"or --live (bounded under the cumulative ${SPEND_CEILING_USD:.2f} ceiling). "
            "Nothing was spent."
        )
        return 1

    config = live_config()
    required = sorted(
        {
            env_key
            for role in CLIENT_ROLES
            for env_key in _BACKEND_ENV_KEYS.get(getattr(config.backends, role).backend, ())
        }
    )
    missing = [env_key for env_key in required if not os.getenv(env_key)]
    if missing:
        print(
            f"Declining to run: {', '.join(missing)} "
            f"{'are' if len(missing) > 1 else 'is'} not set (the configured backends "
            "read credentials from the environment only). Set them (e.g. in a .env file) "
            "and re-run. Nothing was spent."
        )
        return 1

    from shrlm.experiment.live_gates import pricing_attestation_mismatch

    pricing_reason = pricing_attestation_mismatch(
        os.getenv("SHRLM_VERIFIED_PRICING"),
        config.pricing.list_price.input_per_million,
        config.pricing.list_price.output_per_million,
    )
    if pricing_reason is not None:
        print(f"Declining to run: {pricing_reason}. Nothing was spent.")
        return 1

    if args.probe and not args.live:
        return run_probe(config)
    return run_live(config, Path(args.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
