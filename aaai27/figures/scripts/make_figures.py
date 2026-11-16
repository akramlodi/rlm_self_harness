#!/usr/bin/env python3
"""Build the result figures of the AAAI-27 paper from aaai27/figures/data/*.csv.

Usage (from the repository root or anywhere):
    python3 aaai27/figures/scripts/make_figures.py

Writes vector PDFs next to the data folder:
    aaai27/figures/fig_pilot_gate.pdf     (pilot run, DeepSeek-V4-Flash)
    aaai27/figures/fig_long_context.pdf   (preliminary 256K test, DeepSeek-V3.2)

AAAI forbids pgfplots, so figures are pre-rendered and included with
\\includegraphics. Text is set by LaTeX with newtxtext/newtxmath (the paper's
fonts) and embedded as Type 1, never Type 3. Needs matplotlib, numpy and a
TeX installation with newtx; pass --no-usetex to fall back to STIX fonts
(TrueType, still no Type 3).
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(HERE)
DATA = os.path.join(FIG_DIR, "data")

# Categorical slots (fixed order) and text inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#2b2f36", "#5f6570", "#d9dce1"

LENGTHS = ("8192", "16384", "32768")
LEN_LABEL = {"8192": "8K", "16384": "16K", "32768": "32K"}


def setup(usetex: bool) -> None:
    rc = {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "hatch.color": "white",
        "hatch.linewidth": 0.5,
    }
    if usetex:
        rc.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        })
    else:
        rc.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    plt.rcParams.update(rc)


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def tex(s: str) -> str:
    """Escape the few characters that LaTeX would otherwise interpret."""
    if plt.rcParams["text.usetex"]:
        return s.replace("_", r"\_").replace("%", r"\%")
    return s


# ---------------------------------------------------------------------------
# Figure: pilot run (DeepSeek-V4-Flash, study configuration)
# ---------------------------------------------------------------------------
def fig_pilot_gate() -> None:
    rows = read_csv("pilot_validation.csv")
    val = defaultdict(dict)  # round -> {"baseline": row, "cand": row}
    scr = defaultdict(dict)  # round -> {subject: row}
    for r in rows:
        rnd = int(r["round"])
        if r["stage"] == "validation":
            val[rnd]["baseline" if r["subject"] == "baseline" else "cand"] = r
        else:
            scr[rnd][r["subject"]] = r
    rounds = sorted(val)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                             gridspec_kw={"width_ratios": [1.45, 1.0, 1.0], "wspace": 0.42})

    # (a) per-stratum delta of each validated candidate against its incumbent,
    #     with the veto line -tau_l and the improvement threshold for the macro.
    ax = axes[0]
    strata = list(LENGTHS) + ["macro"]
    xs = np.arange(len(strata))
    width = 0.17
    offsets = (np.arange(len(rounds)) - (len(rounds) - 1) / 2) * width
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for off, rnd in zip(offsets, rounds):
        b, c = val[rnd]["baseline"], val[rnd]["cand"]
        promoted = c["decision"] == "promoted"
        deltas = [float(c[f"f1_{l}"]) - float(b[f"f1_{l}"]) for l in LENGTHS]
        deltas.append(float(c["f1_macro"]) - float(b["f1_macro"]))
        col = BLUE if promoted else ORANGE
        ax.vlines(xs + off, 0, deltas, color=col, lw=1.2, zorder=2)
        ax.scatter(xs + off, deltas, s=16, marker=markers[rnd], zorder=3,
                   facecolor=col if promoted else "white", edgecolor=col, lw=1.0,
                   label=f"R{rnd} {c['surface']}")
    c0 = val[rounds[0]]["cand"]
    for i, l in enumerate(LENGTHS):
        tau = float(c0[f"tau_{l}"])
        ax.hlines(-tau, i - 0.42, i + 0.42, color=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=1)
    ax.hlines(float(c0["tau_improvement"]), 3 - 0.42, 3 + 0.42, color=INK, lw=0.9,
              ls=(0, (1, 1)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.text(2.47, -float(c0["tau_32768"]) - 0.03, r"$-\tau_\ell$", ha="right", va="top",
            fontsize=6.3, color=INK)
    ax.set_xticks(xs, [LEN_LABEL.get(s, "Macro") for s in strata])
    ax.set_ylabel(r"$\Delta$ held-out F1 vs.\ incumbent" if plt.rcParams["text.usetex"]
                  else r"$\Delta$ held-out F1 vs. incumbent")
    ax.set_ylim(-0.78, 0.26)
    ax.grid(axis="x", visible=False)
    ax.set_title(tex("(a) Validation gate per stratum"), loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handletextpad=0.2, columnspacing=0.7)

    # (b) screening delta (3 tasks x 1 attempt) vs full-validation delta.
    ax = axes[1]
    lim = (-0.5, 0.6)
    ax.plot(lim, lim, color=GRID, lw=0.8, zorder=1)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    for rnd in rounds:
        c = val[rnd]["cand"]
        # The candidate was screened in the round its id names (r03-... in round 4).
        srnd = int(c["subject"][1:3])
        s_d = float(scr[srnd][c["subject"]]["f1_macro"]) - float(scr[srnd]["baseline"]["f1_macro"])
        v_d = float(c["f1_macro"]) - float(val[rnd]["baseline"]["f1_macro"])
        promoted = c["decision"] == "promoted"
        col = BLUE if promoted else ORANGE
        ax.scatter([s_d], [v_d], s=22, marker=markers[rnd], zorder=3,
                   facecolor=col if promoted else "white", edgecolor=col, lw=1.0)
        dx, ha = (0.03, "left") if rnd != 1 else (-0.03, "right")
        ax.text(s_d + dx, v_d, f"R{rnd}", ha=ha, va="center", fontsize=6.3, color=INK)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$\Delta$ macro F1, screening")
    ax.set_ylabel(r"$\Delta$ macro F1, validation")
    ax.set_title(tex("(b) Screening vs. validation"), loc="left")

    # (c) where the rejected candidates failed: outcome mix of the 36 runs.
    ax = axes[2]
    labels, ok, term, fmt = [], [], [], []
    for rnd in rounds:
        for key in ("baseline", "cand"):
            r = val[rnd][key]
            if key == "baseline" and rnd > 1:
                continue  # reused incumbent runs: same numbers as the previous candidate
            name = r"$\mathrm{H}_0$" if key == "baseline" else f"R{rnd} {r['surface']}"
            n = int(r["n_runs"])
            t = int(r["n_resource_terminated"])
            w = int(r["n_wrong_format"])
            labels.append(name)
            term.append(t / n)
            fmt.append(w / n)
            ok.append((n - t - w) / n)
    y = np.arange(len(labels))[::-1]
    h = 0.62
    ax.barh(y, ok, height=h, color="#a9b0bb", edgecolor="white", lw=0.8, label="answered")
    ax.barh(y, fmt, left=ok, height=h, color=YELLOW, edgecolor="white", lw=0.8,
            label=tex("wrong_format"))
    ax.barh(y, term, left=np.array(ok) + np.array(fmt), height=h, color=ORANGE,
            edgecolor="white", lw=0.8, hatch="////", label="terminated")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_xlabel("share of validation runs")
    ax.grid(axis="y", visible=False)
    ax.set_title(tex("(c) Run outcomes"), loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.17), ncol=3,
              handlelength=1.2, columnspacing=0.6, handletextpad=0.3)

    out = os.path.join(FIG_DIR, "fig_pilot_gate.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure: preliminary 256K long-context test (DeepSeek-V3.2)
# ---------------------------------------------------------------------------
def fig_long_context() -> None:
    rows = read_csv("prelim_long256k.csv")
    name = {"H0": r"$\mathrm{H}_0$", "H0*": r"$\mathrm{H}_0^{*}$",
            "lambda-RLM": r"$\lambda$-RLM", "SH-RLM": "SH-RLM"}
    color = {"H0": AQUA, "H0*": YELLOW, "lambda-RLM": ORANGE, "SH-RLM": BLUE}
    marker = {"H0": "s", "H0*": "^", "lambda-RLM": "D", "SH-RLM": "o"}

    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.6),
                             gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.75})

    # (a) quality against cost per run.
    ax = axes[0]
    for r in rows:
        c = r["condition"]
        ax.scatter(float(r["cost_per_run_usd"]), float(r["mean_f1"]), s=24, marker=marker[c],
                   facecolor=color[c], edgecolor="white", lw=0.8, zorder=3)
        dx, ha = (0.04, "left")
        if c == "SH-RLM":
            dx, ha = (-0.04, "right")
        ax.text(float(r["cost_per_run_usd"]) + dx, float(r["mean_f1"]), name[c],
                ha=ha, va="center", fontsize=6.3)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.8)
    ax.set_xticks([0, 0.4, 0.8], ["0", ".4", ".8"])
    ax.set_yticks([0, 0.4, 0.8], ["0", ".4", ".8"])
    ax.set_xlabel(r"cost per run (\$)" if plt.rcParams["text.usetex"] else "cost per run ($)")
    ax.set_ylabel("mean F1 at 256K")
    ax.set_title(tex("(a) Quality vs. cost"), loc="left")

    # (b) SH-RLM minus each reference, task-level paired bootstrap 95% CI.
    ax = axes[1]
    refs = [r for r in rows if r["condition"] != "SH-RLM"]
    y = np.arange(len(refs))[::-1]
    for yi, r in zip(y, refs):
        c = r["condition"]
        d, lo, hi = float(r["diff_vs_shrlm"]), float(r["ci_lo"]), float(r["ci_hi"])
        ax.hlines(yi, lo, hi, color=color[c], lw=2.0, zorder=2)
        ax.scatter([d], [yi], s=18, marker=marker[c], facecolor=color[c],
                   edgecolor=INK, lw=0.5, zorder=3)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.set_yticks(y, [name[r["condition"]] for r in refs])
    ax.set_ylim(-0.6, len(refs) - 0.4)
    ax.set_xlim(-0.3, 0.6)
    ax.set_xticks([-0.2, 0, 0.2, 0.4], ["$-$.2", "0", ".2", ".4"])
    ax.set_xlabel(r"SH-RLM $-$ reference, F1")
    ax.grid(axis="y", visible=False)
    ax.set_title(tex("(b) Paired difference"), loc="left")

    out = os.path.join(FIG_DIR, "fig_long_context.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    setup(usetex="--no-usetex" not in sys.argv)
    fig_pilot_gate()
    fig_long_context()


if __name__ == "__main__":
    main()
