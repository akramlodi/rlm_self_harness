#!/usr/bin/env python3
"""Write the result figures as plain TikZ (no pgfplots, no external files).

Usage:
    python3 aaai27/figures/scripts/make_tikz.py

Reads aaai27/figures/data/*.csv and writes
    aaai27/figures/fig_pilot_gate.tex
    aaai27/figures/fig_long_context.tex
Each file is a complete figure environment, \\input from the paper. All
coordinates are computed here, so the .tex needs only tikz and xcolor.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(HERE)
DATA = os.path.join(FIG_DIR, "data")
LENGTHS = ("8192", "16384", "32768")

COLORS = r"""\definecolor{vizblue}{HTML}{2A78D6}
\definecolor{vizorange}{HTML}{EB6834}
\definecolor{vizaqua}{HTML}{1BAF7A}
\definecolor{vizyellow}{HTML}{EDA100}
\definecolor{vizgrey}{HTML}{A9B0BB}
\definecolor{vizink}{HTML}{2B2F36}
\definecolor{vizink2}{HTML}{5F6570}
\definecolor{vizgrid}{HTML}{D9DCE1}
"""

MARK = {1: "circle", 2: "square", 3: "triangle", 4: "diamond"}


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def f(x: float) -> str:
    return f"{x:.3f}"


def marker(kind: str, x: float, y: float, filled: bool, color: str, r: float = 0.075) -> str:
    fill = f"fill={color}" if filled else "fill=white"
    style = f"draw={color}, {fill}, line width=0.6pt"
    if kind == "circle":
        return f"\\path[{style}] ({f(x)},{f(y)}) circle ({f(r)});"
    if kind == "square":
        s = r * 0.9
        return f"\\path[{style}] ({f(x - s)},{f(y - s)}) rectangle ({f(x + s)},{f(y + s)});"
    if kind == "triangle":
        s = r * 1.15
        return (f"\\path[{style}] ({f(x)},{f(y + s)}) -- ({f(x - s)},{f(y - 0.75 * s)}) -- "
                f"({f(x + s)},{f(y - 0.75 * s)}) -- cycle;")
    s = r * 1.2  # diamond
    return (f"\\path[{style}] ({f(x)},{f(y + s)}) -- ({f(x + s)},{f(y)}) -- "
            f"({f(x)},{f(y - s)}) -- ({f(x - s)},{f(y)}) -- cycle;")


def pilot_data():
    val = defaultdict(dict)
    scr = defaultdict(dict)
    for r in read_csv("pilot_validation.csv"):
        rnd = int(r["round"])
        if r["stage"] == "validation":
            val[rnd]["baseline" if r["subject"] == "baseline" else "cand"] = r
        else:
            scr[rnd][r["subject"]] = r
    return val, scr


def panel_a(val) -> str:
    rounds = sorted(val)
    W, H = 5.0, 3.3
    y0, y1 = -0.8, 0.3
    sy = H / (y1 - y0)
    Y = lambda v: (v - y0) * sy  # noqa: E731
    xs = [0.65 + i * 1.2 for i in range(4)]
    out = [r"\begin{tikzpicture}[font=\scriptsize, line cap=round, baseline=0pt]"]
    for t in (-0.6, -0.4, -0.2, 0.0, 0.2):
        out.append(f"\\draw[vizgrid, line width=0.3pt] (0,{f(Y(t))}) -- ({f(W)},{f(Y(t))});")
        lab = "0" if t == 0 else (f"$-{abs(t):.1f}$".replace("0.", ".") if t < 0 else f"{t:.1f}".replace("0.", "."))
        out.append(f"\\node[anchor=east, text=vizink2] at (-0.05,{f(Y(t))}) {{{lab}}};")
    out.append(f"\\draw[vizink2, line width=0.4pt] (0,{f(Y(0))}) -- ({f(W)},{f(Y(0))});")
    out.append(f"\\draw[vizink2, line width=0.4pt] (0,0) -- (0,{f(H)});")
    c0 = val[rounds[0]]["cand"]
    for i, l in enumerate(LENGTHS):
        tau = float(c0[f"tau_{l}"])
        out.append(f"\\draw[vizink, line width=0.6pt, dash pattern=on 2pt off 1pt] "
                   f"({f(xs[i] - 0.5)},{f(Y(-tau))}) -- ({f(xs[i] + 0.5)},{f(Y(-tau))});")
    out.append(f"\\node[anchor=north east, text=vizink, inner sep=1pt] at ({f(xs[2] + 0.55)},{f(Y(-float(c0['tau_32768'])) - 0.02)}) {{$-\\tau_\\ell$}};")
    ti = float(c0["tau_improvement"])
    out.append(f"\\draw[vizink, line width=0.6pt, dash pattern=on 0.6pt off 0.8pt] "
               f"({f(xs[3] - 0.5)},{f(Y(ti))}) -- ({f(xs[3] + 0.5)},{f(Y(ti))});")
    width = 0.2
    for j, rnd in enumerate(rounds):
        off = (j - 1.5) * width
        b, c = val[rnd]["baseline"], val[rnd]["cand"]
        promoted = c["decision"] == "promoted"
        col = "vizblue" if promoted else "vizorange"
        ds = [float(c[f"f1_{l}"]) - float(b[f"f1_{l}"]) for l in LENGTHS]
        ds.append(float(c["f1_macro"]) - float(b["f1_macro"]))
        for i, d in enumerate(ds):
            x = xs[i] + off
            out.append(f"\\draw[{col}, line width=0.9pt] ({f(x)},{f(Y(0))}) -- ({f(x)},{f(Y(d))});")
            out.append(marker(MARK[rnd], x, Y(d), promoted, col))
    for i, lab in enumerate(["8K", "16K", "32K", "Macro"]):
        out.append(f"\\node[anchor=north, text=vizink2] at ({f(xs[i])},-0.03) {{{lab}}};")
    out.append(f"\\node[rotate=90, anchor=south] at (-0.62,{f(H / 2)}) {{$\\Delta$ held-out F1 vs.\\ incumbent}};")
    out.append(f"\\node[anchor=south west, inner sep=0pt] at (-0.55,{f(H + 0.08)}) {{(a) Validation gate per stratum}};")
    # legend
    lx, ly = 0.0, -0.55
    for j, rnd in enumerate(rounds):
        c = val[rnd]["cand"]
        promoted = c["decision"] == "promoted"
        col = "vizblue" if promoted else "vizorange"
        x = lx + j * 1.3
        out.append(marker(MARK[rnd], x + 0.1, ly, promoted, col))
        out.append(f"\\node[anchor=west, inner sep=1pt] at ({f(x + 0.2)},{f(ly)}) {{R{rnd} {c['surface']}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def panel_b(val, scr) -> str:
    S = 3.3
    lo, hi = -0.5, 0.6
    s = S / (hi - lo)
    P = lambda v: (v - lo) * s  # noqa: E731
    out = [r"\begin{tikzpicture}[font=\scriptsize, line cap=round, baseline=0pt]"]
    for t in (-0.4, -0.2, 0.0, 0.2, 0.4):
        out.append(f"\\draw[vizgrid, line width=0.3pt] (0,{f(P(t))}) -- ({f(S)},{f(P(t))});")
        out.append(f"\\draw[vizgrid, line width=0.3pt] ({f(P(t))},0) -- ({f(P(t))},{f(S)});")
        lab = "0" if t == 0 else (f"$-{abs(t):.1f}$".replace("0.", ".") if t < 0 else f"{t:.1f}".replace("0.", "."))
        out.append(f"\\node[anchor=east, text=vizink2] at (-0.05,{f(P(t))}) {{{lab}}};")
        out.append(f"\\node[anchor=north, text=vizink2] at ({f(P(t))},-0.03) {{{lab}}};")
    out.append(f"\\draw[vizgrid, line width=0.7pt] (0,0) -- ({f(S)},{f(S)});")
    out.append(f"\\draw[vizink2, line width=0.4pt] (0,{f(P(0))}) -- ({f(S)},{f(P(0))});")
    out.append(f"\\draw[vizink2, line width=0.4pt] ({f(P(0))},0) -- ({f(P(0))},{f(S)});")
    out.append(f"\\draw[vizink2, line width=0.4pt] (0,0) -- (0,{f(S)}) (0,0) -- ({f(S)},0);")
    for rnd in sorted(val):
        c = val[rnd]["cand"]
        srnd = int(c["subject"][1:3])
        sd = float(scr[srnd][c["subject"]]["f1_macro"]) - float(scr[srnd]["baseline"]["f1_macro"])
        vd = float(c["f1_macro"]) - float(val[rnd]["baseline"]["f1_macro"])
        promoted = c["decision"] == "promoted"
        col = "vizblue" if promoted else "vizorange"
        out.append(marker(MARK[rnd], P(sd), P(vd), promoted, col))
        anchor, dx = ("east", -0.1) if rnd == 1 else ("west", 0.1)
        out.append(f"\\node[anchor={anchor}, inner sep=1pt] at ({f(P(sd) + dx)},{f(P(vd))}) {{R{rnd}}};")
    out.append(f"\\node[rotate=90, anchor=south] at (-0.5,{f(S / 2)}) {{$\\Delta$ macro F1, validation}};")
    out.append(f"\\node[anchor=north] at ({f(S / 2)},-0.3) {{$\\Delta$ macro F1, screening}};")
    out.append(f"\\node[anchor=south west, inner sep=0pt] at (-0.55,{f(S + 0.08)}) {{(b) Screening vs.\\ validation}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def panel_c(val) -> str:
    W, H = 3.1, 3.3
    rows = []
    for rnd in sorted(val):
        for key in ("baseline", "cand"):
            if key == "baseline" and rnd > 1:
                continue
            r = val[rnd][key]
            n = int(r["n_runs"])
            t = int(r["n_resource_terminated"])
            w = int(r["n_wrong_format"])
            name = "$\\mathrm{H}_0$" if key == "baseline" else f"R{rnd} {r['surface']}"
            rows.append((name, (n - t - w) / n, w / n, t / n))
    out = [r"\begin{tikzpicture}[font=\scriptsize, baseline=0pt]"]
    step = H / len(rows)
    bh = step * 0.62
    for i, (name, ok, fmt, term) in enumerate(rows):
        yc = H - (i + 0.5) * step
        x = 0.0
        for share, style in ((ok, "fill=vizgrey"), (fmt, "fill=vizyellow"),
                             (term, "fill=vizorange")):
            if share > 0:
                out.append(f"\\path[{style}, draw=white, line width=0.5pt] ({f(x * W)},{f(yc - bh / 2)}) "
                           f"rectangle ({f((x + share) * W)},{f(yc + bh / 2)});")
                if style == "fill=vizorange":
                    out.append(f"\\begin{{scope}}\\clip ({f(x * W)},{f(yc - bh / 2)}) rectangle "
                               f"({f((x + share) * W)},{f(yc + bh / 2)});")
                    k = x * W - bh
                    while k < (x + share) * W:
                        out.append(f"\\draw[white, line width=0.4pt] ({f(k)},{f(yc - bh / 2)}) -- "
                                   f"({f(k + bh)},{f(yc + bh / 2)});")
                        k += 0.09
                    out.append("\\end{scope}")
            x += share
        out.append(f"\\node[anchor=east, text=vizink2] at (-0.05,{f(yc)}) {{{name}}};")
    out.append(f"\\draw[vizink2, line width=0.4pt] (0,0) -- ({f(W)},0) (0,0) -- (0,{f(H)});")
    for t, lab in ((0, "0"), (0.5, ".5"), (1, "1")):
        out.append(f"\\node[anchor=north, text=vizink2] at ({f(t * W)},-0.03) {{{lab}}};")
    out.append(f"\\node[anchor=north] at ({f(W / 2)},-0.3) {{share of validation runs}};")
    out.append(f"\\node[anchor=south west, inner sep=0pt] at (-0.9,{f(H + 0.08)}) {{(c) Run outcomes}};")
    ly = -0.9
    for j, (lab, col) in enumerate((("answered", "vizgrey"), ("malformed", "vizyellow"),
                                    ("terminated", "vizorange"))):
        x = (-0.9, 0.45, 1.95)[j]
        out.append(f"\\path[fill={col}] ({f(x)},{f(ly - 0.06)}) rectangle ({f(x + 0.2)},{f(ly + 0.06)});")
        out.append(f"\\node[anchor=west, inner sep=1pt] at ({f(x + 0.22)},{f(ly)}) {{{lab}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def fig_pilot() -> str:
    val, scr = pilot_data()
    return "\n".join([
        "% Generated by aaai27/figures/scripts/make_tikz.py from figures/data/pilot_validation.csv.",
        "% Do not edit by hand; edit the data or the script and regenerate.",
        r"\begin{figure*}[t]",
        r"\centering",
        COLORS.strip(),
        panel_a(val),
        r"\hfill",
        panel_b(val, scr),
        r"\hfill",
        panel_c(val),
        r"\caption{Pilot run (DeepSeek-V4-Flash, OOLONG-Pairs, 12 validation tasks",
        r"$\times$ 3 attempts per subject). Filled blue markers are promoted",
        r"candidates and hollow orange markers are rejected ones; R$t$~S$j$ is the round",
        r"and edited surface. (a)~Held-out F1 change against the round's incumbent per",
        r"stratum and macro-averaged. Dashed lines are the veto margins $-\tau_\ell$; the",
        r"dotted line is $\tau_{\mathrm{imp}}$. (b)~Macro-F1 change in screening (3 tasks",
        r"$\times$ 1 attempt) against full validation; the grey line is $y{=}x$.",
        r"(c)~Outcome of the 36 validation runs: a scored answer, a malformed answer",
        r"(\texttt{wrong\_format}, F1 $=0$), or termination by the 3,600\,s cap (F1",
        r"$=0$). Incumbent rows after round 1 reuse the previous candidate's runs and",
        r"are omitted.}",
        r"\label{fig:pilot}",
        r"\end{figure*}",
        "",
    ])


def fig_long() -> str:
    rows = read_csv("prelim_long256k.csv")
    name = {"H0": r"$\mathrm{H}_0$", "H0*": r"$\mathrm{H}_0^{*}$",
            "lambda-RLM": r"$\lambda$-RLM", "SH-RLM": r"\textsc{SH-RLM}"}
    col = {"H0": "vizaqua", "H0*": "vizyellow", "lambda-RLM": "vizorange", "SH-RLM": "vizblue"}
    mk = {"H0": "square", "H0*": "triangle", "lambda-RLM": "diamond", "SH-RLM": "circle"}
    W, H = 2.6, 2.6
    a = [r"\begin{tikzpicture}[font=\scriptsize]"]
    sx, sy = W / 0.85, H / 0.8
    for t in (0.0, 0.4, 0.8):
        lab = "0" if t == 0 else f"{t:.1f}".replace("0.", ".")
        a.append(f"\\draw[vizgrid, line width=0.3pt] (0,{f(t * sy)}) -- ({f(W)},{f(t * sy)});")
        a.append(f"\\draw[vizgrid, line width=0.3pt] ({f(t * sx)},0) -- ({f(t * sx)},{f(H)});")
        a.append(f"\\node[anchor=east, text=vizink2] at (-0.05,{f(t * sy)}) {{{lab}}};")
        a.append(f"\\node[anchor=north, text=vizink2] at ({f(t * sx)},-0.03) {{{lab}}};")
    a.append(f"\\draw[vizink2, line width=0.4pt] (0,0) -- ({f(W)},0) (0,0) -- (0,{f(H)});")
    for r in rows:
        c = r["condition"]
        x, y = float(r["cost_per_run_usd"]) * sx, float(r["mean_f1"]) * sy
        a.append(marker(mk[c], x, y, True, col[c], r=0.08))
        anchor, dx = ("east", -0.12) if c == "SH-RLM" else ("west", 0.12)
        dy = 0.08 if c == "SH-RLM" else (-0.06 if c == "H0" else 0)
        a.append(f"\\node[anchor={anchor}, inner sep=1pt] at ({f(x + dx)},{f(y + dy)}) {{{name[c]}}};")
    a.append(f"\\node[rotate=90, anchor=south] at (-0.38,{f(H / 2)}) {{mean F1 at 256K}};")
    a.append(f"\\node[anchor=north] at ({f(W / 2)},-0.3) {{cost per run (\\$)}};")
    a.append(f"\\node[anchor=south west, inner sep=0pt] at (-0.45,{f(H + 0.08)}) {{(a) Quality vs.\\ cost}};")
    a.append(r"\end{tikzpicture}")

    b = [r"\begin{tikzpicture}[font=\scriptsize]"]
    lo, hi = -0.3, 0.6
    s = W / (hi - lo)
    X = lambda v: (v - lo) * s  # noqa: E731
    refs = [r for r in rows if r["condition"] != "SH-RLM"]
    for t in (-0.2, 0.0, 0.2, 0.4):
        lab = "0" if t == 0 else (f"$-{abs(t):.1f}$".replace("0.", ".") if t < 0 else f"{t:.1f}".replace("0.", "."))
        b.append(f"\\draw[vizgrid, line width=0.3pt] ({f(X(t))},0) -- ({f(X(t))},{f(H)});")
        b.append(f"\\node[anchor=north, text=vizink2] at ({f(X(t))},-0.03) {{{lab}}};")
    b.append(f"\\draw[vizink2, line width=0.4pt] ({f(X(0))},0) -- ({f(X(0))},{f(H)});")
    b.append(f"\\draw[vizink2, line width=0.4pt] (0,0) -- ({f(W)},0) (0,0) -- (0,{f(H)});")
    step = H / len(refs)
    for i, r in enumerate(refs):
        c = r["condition"]
        y = H - (i + 0.5) * step
        d, l, h = float(r["diff_vs_shrlm"]), float(r["ci_lo"]), float(r["ci_hi"])
        b.append(f"\\draw[{col[c]}, line width=1.4pt] ({f(X(l))},{f(y)}) -- ({f(X(h))},{f(y)});")
        b.append(marker(mk[c], X(d), y, True, col[c], r=0.08).replace(f"draw={col[c]}", "draw=vizink"))
        b.append(f"\\node[anchor=east, text=vizink2] at (-0.05,{f(y)}) {{{name[c]}}};")
    b.append(f"\\node[anchor=north] at ({f(W / 2)},-0.3) {{\\textsc{{SH-RLM}} $-$ reference, F1}};")
    b.append(f"\\node[anchor=south west, inner sep=0pt] at (-0.75,{f(H + 0.08)}) {{(b) Paired difference}};")
    b.append(r"\end{tikzpicture}")

    return "\n".join([
        "% Generated by aaai27/figures/scripts/make_tikz.py from figures/data/prelim_long256k.csv.",
        "% Do not edit by hand; edit the data or the script and regenerate.",
        r"\begin{figure}[t]",
        r"\centering",
        COLORS.strip(),
        r"\resizebox{\linewidth}{!}{%",
        "\n".join(a),
        r"\hspace{0.4cm}",
        "\n".join(b),
        "}",
        r"\caption{256K OOLONG-Pairs test (DeepSeek-V3.2; 9 tasks $\times$ 3",
        r"attempts per condition). (a)~Mean F1 against mean cost per run.",
        r"(b)~\shrlm minus each reference with task-level paired bootstrap 95\%",
        r"intervals; right of 0 favors \shrlm. Data as in \Cref{tab:long}.}",
        r"\label{fig:long}",
        r"\end{figure}",
        "",
    ])


def main() -> None:
    for name, text in (("fig_pilot_gate.tex", fig_pilot()), ("fig_long_context.tex", fig_long())):
        path = os.path.join(FIG_DIR, name)
        with open(path, "w") as fh:
            fh.write(text)
        print("wrote", path)


if __name__ == "__main__":
    main()
