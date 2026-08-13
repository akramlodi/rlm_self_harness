"""Markdown rendering for the cost/time report (R6).

Split out of ``shrlm.experiment.report``, which still assembles the
``CostReport`` this module renders: ``render_markdown`` turns one built
report into the same markdown the report command writes to stdout -- the
measured stage table and per-run means, the extrapolation (point and
pessimistic projections), the priced scenario table, and the recommendation
or provisional comparison -- for pasting into the paper as well as terminal
reading.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from shrlm.experiment.scenarios import STATUS_RECOMMENDED

if TYPE_CHECKING:
    from shrlm.experiment.report import CostReport


def usd(value: float) -> str:
    return f"${value:,.2f}" if abs(value) >= 0.01 else f"${value:.6f}"


def count(value: float) -> str:
    return f"{value:,.0f}"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(report: CostReport) -> str:
    """The whole report as markdown, for stdout and for pasting into the paper."""
    lines: list[str] = [
        f"# Cost/time report -- {report.out_dir} (profile: {report.profile})",
        "",
        "## Measured",
        "",
    ]
    lines += table(
        ("stage", "runs", "input tok", "output tok", "USD", "wall s", "cache hits", "flags"),
        (
            (
                stage.stage,
                count(stage.runs),
                count(stage.input_tokens),
                count(stage.output_tokens),
                usd(stage.cost),
                f"{stage.wall_seconds:,.1f}",
                count(stage.cache_hits),
                " ".join(
                    flag
                    for flag, active in (
                        ("lower-bound", stage.lower_bound),
                        (f"resumed x{stage.resumed_attempts}", stage.resumed_attempts > 0),
                    )
                    if active
                )
                or "-",
            )
            for stage in report.stages
        ),
    )
    lines += ["", "### Per-run means (the extrapolation basis)", ""]
    lines += table(
        ("environment", "length", "runs", "mean in", "mean out", "mean USD", "mean s", "flags"),
        (
            (
                bucket.environment,
                bucket.length,
                count(bucket.n_runs),
                f"{bucket.mean_input_tokens:,.1f}",
                f"{bucket.mean_output_tokens:,.1f}",
                usd(bucket.mean_cost),
                f"{bucket.mean_wall_seconds:,.1f}",
                " ".join(
                    flag
                    for flag, active in (
                        ("lower-bound", bucket.lower_bound),
                        ("thin", bucket.thin),
                    )
                    if active
                )
                or "-",
            )
            for bucket in report.buckets
        ),
    )
    disk = report.disk
    lines += [
        "",
        f"Disk: {disk['measured_bytes']:,.0f} B over {disk['measured_runs']:,.0f} run(s) "
        f"= {disk['bytes_per_run']:,.0f} B/run; projected "
        f"{disk['projected_bytes'] / 1e9:,.3f} GB for the full experiment.",
        "",
        "## Extrapolation",
        "",
        f"Runs/round = {count(report.run_counts.runs_per_round)} "
        f"(m*n_in + v(K+1)(n_in+n_ho) + p_merge*v(n_in+n_ho)); "
        f"{report.run_counts.rounds} round(s) = "
        f"{count(report.run_counts.optimization_runs)} optimization runs. "
        f"Eval grid = {report.run_counts.eval_conditions} condition(s) x "
        f"(short + long) x repetitions = {count(report.run_counts.eval_runs)} runs.",
        "",
    ]
    for projection in (report.point, report.pessimistic):
        lines += [f"### {projection.label} projection", ""]
        lines += table(
            ("leg", "context", "runs", "cost drift", "input tok", "output tok", "basis"),
            (
                (
                    leg.name,
                    leg.context,
                    count(leg.runs),
                    "-" if leg.drift_multiplier == 1.0 else f"x{leg.drift_multiplier:,.0f}",
                    count(leg.input_tokens),
                    count(leg.output_tokens),
                    leg.basis,
                )
                for leg in projection.legs
            ),
        )
        lines += [
            "",
            f"Total: {count(projection.input_tokens)} in / "
            f"{count(projection.output_tokens)} out"
            + ("" if projection.long_measured else "  **(short-only: long unmeasured)**"),
            "",
        ]
    lines += ["## Scenarios", ""]
    lines += table(
        ("scenario", "kind", "USD (point)", "USD (pessimistic)", "eligible", "notes"),
        (
            (
                scenario.name,
                scenario.kind,
                usd(scenario.usd_point),
                usd(scenario.usd_pessimistic),
                "yes" if scenario.eligible else "no",
                "; ".join(
                    note
                    for note in (
                        scenario.ineligible_reason,
                        "changes model numerics" if scenario.changes_numerics else None,
                        scenario.provenance,
                    )
                    if note
                ),
            )
            for scenario in report.scenarios
        ),
    )
    lines += ["", "## Recommendation", ""]
    if report.recommendation.status == STATUS_RECOMMENDED:
        assert report.recommendation.usd is not None
        lines.append(
            f"**{report.recommendation.scenario}** at "
            f"{usd(report.recommendation.usd)} -- the cheapest scenario passing every "
            "validity gate."
        )
    else:
        lines.append("**Provisional comparison** -- no recommendation. Missing:")
        lines += [
            f"- `{gate['gate']}`: {gate['detail']}" for gate in report.recommendation.failing_gates
        ]
    if report.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in report.warnings]
    return "\n".join(lines) + "\n"


__all__ = ["count", "render_markdown", "table", "usd"]
