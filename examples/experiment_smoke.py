"""Tier 2 of the experiment smoke: one shrunk live round plus a small eval, bounded (R13).

Tier 1 is the $0 mock smoke in ``tests/experiment/test_smoke_mock.py``, which
runs the identical pipeline over checked-in split fixtures with a scripted
client and never touches the network. This file is tier 2: the same ``[smoke]``
config profile (R2) against OpenRouter's ``qwen/qwen3-30b-a3b-instruct-2507``
with the configured ``sampling_args``, run once by hand to turn the scaffold's
plumbing into real token, time, and cost numbers.

It is opt-in twice over: nothing runs without BOTH ``OPENROUTER_API_KEY`` and
an explicit ``--live`` flag, and the decline path spends nothing. CI never runs
it.

What it asserts (KTD9, following ``examples/validation_live_smoke.py``):
artifact structure, ``usage.cost`` present and non-null on live responses
(Goal Capsule stop condition (a)), the persisted
``run_metadata.backend_kwargs`` showing the intended sampling args, the
serving provider against the configured allow-list, and a report carrying a
full-experiment estimate. It NEVER asserts accuracy, pass counts, or promotion
outcomes -- at smoke scale (v=1, 3 held-in instances) the tau pass-count rule
is meaningless, so model behavior is the model's business.

Spend control (KTD7; hard ceiling $5)
    The smoke profile's own caps are sized for a pilot, not for a $5 ceiling,
    so this script tightens them to live-smoke values and proves the
    arithmetic before spending anything (``check_budget_arithmetic``). Every
    paid call falls in exactly one of two classes, and the ceiling adds both.

    Governed calls -- every run executed under a ``CandidateSpendBreaker``:

        per-run budget      $0.20   worst-case long run (below) x ~1.7 headroom
        per-breaker budget  $0.35   cumulative, per spend breaker
        breakers            7       t x (1 mining + baseline + k candidates +
                                    merged) + 1 per evaluation condition
        governed ceiling    7 x ($0.35 + $0.20) = $3.85

    A breaker trips only *after* a run pushes cumulative spend past its
    budget, so each breaker's true ceiling is its budget plus one per-run
    budget -- that is the $0.55 term. The per-round breakers are armed afresh
    in each of ``loop.t`` rounds, hence the ``t`` factor (the shipped smoke
    profile runs t = 1, so it is latent there and load-bearing anywhere else).

    Ungoverned calls -- paid model calls no breaker ever sees:

        probe               2       the raw completion plus the client's own
        attribution         27      t x n_in x m records x max_attempts x
                                    transport_retries
        proposal            9       t x max_attempts x transport_retries
        per call            $0.0274 a full 262,144-token context window in at
                                    the configured list input price, plus
                                    max_output_tokens out at the list output
                                    price
        ungoverned ceiling  38 x $0.0274 = $1.04

        ceiling             $3.85 + $1.04 = $4.89 < $5.00

    ``caps.max_budget`` is deliberately NOT the per-call bound for the
    ungoverned classes: at $0.20 x 38 calls it would claim $7.60 of a $5
    ceiling on its own, and the only way to fit under $5 would be to cut the
    per-run budget below the worst-case long run -- guaranteeing the censored
    long sample KTD6 exists to prevent. The priced bound above is a real
    bound on one completion: the provider cannot be given more input than the
    model's context window, and ``max_tokens`` caps the output. It is also
    many times larger than any prompt these stages actually build -- an
    attribution digest is char-bounded and the probe prompt is a constant in
    this file -- which is what leaves the class allowances room to absorb the
    bounded transport retries that are already counted into them.

    Caps are identity keys (R3), so editing the three ``LIVE_*`` constants
    below means a fresh ``--out-dir``: an existing experiment refuses to
    resume under changed caps.

Long runs (KTD6)
    Both environments contribute long instances, with the per-run budget sized
    from a worst-case back-of-envelope so at least one uncapped long run
    completes per environment: an OOLONG-Pairs long window is 262,144 context
    tokens; assume the context passes the model ~3x (root scan, sub-calls,
    re-reads) = ~786k input tokens at the $0.10/1M list tier = $0.079, plus
    30 iterations x 4,096 output tokens at $0.30/1M = $0.037, i.e. ~$0.12
    against a $0.20 per-run budget. Runs the breaker or the per-run budget
    does cap land as lower bounds and the report labels the recommendation
    provisional -- named, never silently zeroed.

    An environment whose long runs were *all* capped (or all skipped by a
    tripped breaker) fails the smoke: long-run cost is superlinear, so a fully
    censored long sample cannot support the API-vs-GPU recommendation, and a
    green smoke over one would certify nothing. The failure is raised only
    after the report is built, written, and printed, so the operator keeps
    every measured number.

Usage:
    # ~$0.001: the U1 execution-note probe, run this FIRST.
    OPENROUTER_API_KEY=... uv run python examples/experiment_smoke.py --probe

    # ~$1-5, up to a few hours: the full tier-2 smoke.
    OPENROUTER_API_KEY=... uv run python examples/experiment_smoke.py --live

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
    ROUND_MARKER_FILENAME,
    experiment_round_dir,
    run_experiment,
)
from shrlm.experiment.report import build_report, render_markdown, write_report
from shrlm.experiment.splits import LENGTHS, MANIFEST_FILE, SPLITS_DIR, split_plan
from shrlm.experiment.usage import STAGE_USAGE_FILE, read_jsonl, read_stage_usage
from shrlm.optimization.attribution import DEFAULT_MAX_ATTEMPTS as ATTRIBUTION_MAX_ATTEMPTS
from shrlm.optimization.attribution import (
    DEFAULT_TRANSPORT_RETRIES as ATTRIBUTION_TRANSPORT_RETRIES,
)
from shrlm.optimization.proposal import DEFAULT_MAX_ATTEMPTS as PROPOSAL_MAX_ATTEMPTS
from shrlm.optimization.proposal import DEFAULT_TRANSPORT_RETRIES as PROPOSAL_TRANSPORT_RETRIES

load_dotenv()

# The hard ceiling from the plan's Goal Capsule stop condition (c).
SPEND_CEILING_USD = 5.0

# Live-smoke caps, sized by the module docstring's back-of-envelope. They
# replace the smoke profile's pilot-sized caps; everything else -- decoding,
# promotion rule, environments, scale counts -- is the shipped smoke profile.
# The per-run budget is load-bearing for KTD6 (it must clear the ~$0.12
# worst-case long run), so the ceiling is fitted by the candidate budget.
LIVE_MAX_BUDGET_USD = 0.20
LIVE_CANDIDATE_BUDGET_USD = 0.35
LIVE_MAX_TIMEOUT_SECONDS = 1200.0

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

# The runner/attributor/proposer model's context window
# (qwen/qwen3-30b-a3b-instruct-2507). No single completion can be given more
# input than this, so it bounds an ungoverned call's input side; the output
# side is bounded by decoding.max_output_tokens (sent as ``max_tokens``).
UNGOVERNED_CONTEXT_TOKENS = 262_144

# The length whose per-run means the extrapolation is most sensitive to (KTD6).
LONG = LENGTHS[1]

PROBE_PROMPT = "Reply with the single word OK."


class SmokeError(RuntimeError):
    """The live smoke's own precondition or structural check failed."""


# ---------------------------------------------------------------------------
# Configuration and budget arithmetic
# ---------------------------------------------------------------------------


def live_config() -> ExperimentConfig:
    """The shipped ``[smoke]`` profile with live-smoke spend caps (R2)."""
    config = load_config("smoke")
    return replace(
        config,
        caps=replace(
            config.caps,
            max_budget=LIVE_MAX_BUDGET_USD,
            max_timeout=LIVE_MAX_TIMEOUT_SECONDS,
            candidate_budget=LIVE_CANDIDATE_BUDGET_USD,
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

    A full context window of input at the configured *list* (never promo)
    input price, plus ``decoding.max_output_tokens`` of output at the list
    output price. Both sides are hard limits on a single completion, which is
    what makes this a bound rather than an estimate -- see the module
    docstring for why ``caps.max_budget`` is not used here.
    """
    price = config.pricing.list_price
    input_usd = UNGOVERNED_CONTEXT_TOKENS * price.input_per_million / 1_000_000
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
    """Refuse to spend anything unless the worst case stays under the ceiling."""
    ceiling = spend_ceiling(config)
    if ceiling >= SPEND_CEILING_USD:
        raise SmokeError(
            f"the configured budgets admit up to ${ceiling:.2f} "
            f"({breaker_count(config)} breakers x (${config.caps.candidate_budget} + "
            f"${config.caps.max_budget}) + {ungoverned_call_count(config)} ungoverned call(s) "
            f"x ${ungoverned_call_ceiling(config):.4f}), which does not clear the "
            f"${SPEND_CEILING_USD} ceiling; lower caps.candidate_budget (raising "
            "caps.max_budget is what the KTD6 long runs need, so cut the cumulative budget "
            "first) before running live."
        )
    return ceiling


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
    args = sampling_args(config)
    return lm.client.chat.completions.create(
        model=config.backends.runner.model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=_merge_extra_body({}, args),
        **_normalize_sampling_args(args),
    )


def probe(config: ExperimentConfig) -> dict[str, Any]:
    """One ~$0.001 live call: cost present, sampling args accepted, provider seen.

    Raises:
        SmokeError: The response carries no usable ``usage.cost`` -- the spend
            breaker, the cost band, and ``acceptance_inputs`` all stand on it,
            so the whole scaffold stops here rather than after spending more.
    """
    lm = runner_client(config)
    response = raw_completion(lm, config, PROBE_PROMPT)
    payload = response.model_dump()
    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    provider = payload.get("provider")

    # The path every persisted run's cost actually travels: the client's own
    # extraction, not the raw field.
    lm.completion(PROBE_PROMPT)
    client_cost = lm.get_last_usage().total_cost

    if cost is None or client_cost is None:
        raise SmokeError(
            f"OpenRouter returned usage.cost={cost!r} (client read {client_cost!r}) for "
            f"{config.backends.runner.model} via provider {provider!r}. The spend breaker, "
            "the promotion cost band, and acceptance_inputs all require a non-null cost "
            "(BYOK and zero-cost routes do not report one). Stop and re-route before "
            "running anything larger."
        )
    return {
        "provider": provider,
        "usage_cost": float(cost),
        "client_cost": float(client_cost),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "sampling_args": sampling_args(config),
    }


def check_provider(config: ExperimentConfig, provider: str | None) -> str:
    """The provider allow-list check (empty allow-list records rather than fails).

    OpenRouter routes per request, and a provider that silently drops
    ``extra_body`` sampling args makes the backbone non-stationary; pinning
    ``backends.openrouter.provider_order`` is what keeps the preregistered
    decoding invariant testable. Until it is pinned, the smoke records what it
    observed so the pilot can pin it.
    """
    allowed = config.backends.openrouter.provider_order
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


def check_costs_present(out_dir: Path) -> int:
    """Every persisted run carries a non-null cost (stop condition (a), at scale)."""
    cost_less: list[str] = []
    total = 0
    for path in manifests(out_dir):
        for entry in read_jsonl(path):
            total += 1
            if entry.get("cost") is None:
                cost_less.append(f"{path}:{entry['run_id']}")
    if cost_less:
        raise SmokeError(
            f"{len(cost_less)} persisted run(s) carry no cost, e.g. {cost_less[:3]}; "
            "OpenRouter stopped reporting usage.cost for this route and the spend "
            "breaker is now blind."
        )
    return total


def check_sampling_args(out_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    """The persisted ``run_metadata.backend_kwargs`` show the intended decoding.

    Reads one trace (they all carry the same run metadata) and compares its
    sampling args against ``sampling_args(config)`` key by key.
    """
    expected = sampling_args(config)
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
    print(f"  sampling args accepted: {result['sampling_args']}")
    print(f"  usage.cost:  ${result['usage_cost']:.6f} (client read ${result['client_cost']:.6f})")
    print(f"  tokens:      {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"  {check_provider(config, result['provider'])}")
    print("PROBE PASSED: usage.cost is present and the configured sampling args are accepted.")
    return 0


def run_live(config: ExperimentConfig, out_dir: Path) -> int:
    ceiling = check_budget_arithmetic(config)
    print(
        f"Live smoke: profile {config.profile}, model {config.backends.runner.model}, "
        f"worst case ${ceiling:.2f} = {breaker_count(config)} breaker(s) x "
        f"${config.caps.candidate_budget + config.caps.max_budget:.2f} + "
        f"{ungoverned_call_count(config)} ungoverned call(s) x "
        f"${ungoverned_call_ceiling(config):.4f} (ceiling ${SPEND_CEILING_USD:.2f})."
    )

    probed = probe(config)
    print(
        f"Probe: usage.cost ${probed['usage_cost']:.6f}; {check_provider(config, probed['provider'])}"
    )

    print(f"Running one optimization round into {out_dir} (this downloads the pinned splits)...")
    experiment = run_experiment(config, out_dir)
    print(f"  stopped on {experiment.stopped}; frozen harness {experiment.final_harness_hash}")

    print(f"Evaluating conditions {list(DEFAULT_CONDITIONS)} on the frozen test splits...")
    evaluation = run_evaluation(config, DEFAULT_CONDITIONS, out_dir)
    for condition in evaluation.conditions:
        print(f"  {condition.condition_id}: outcome {condition.outcome}, ${condition.spent:.4f}")

    check_directory_contract(out_dir)
    n_runs = check_costs_present(out_dir)
    persisted_args = check_sampling_args(out_dir, config)
    print(f"Artifacts: {n_runs} persisted run(s), all carrying a non-null cost.")
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
            "Declining to run: this smoke spends real money. Re-run with --probe (~$0.001) "
            f"or --live (up to ${SPEND_CEILING_USD:.2f}). Nothing was spent."
        )
        return 1
    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "Declining to run: OPENROUTER_API_KEY is not set. Set it (e.g. in a .env file) "
            "and re-run. Nothing was spent."
        )
        return 1

    config = live_config()
    if args.probe and not args.live:
        return run_probe(config)
    return run_live(config, Path(args.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
