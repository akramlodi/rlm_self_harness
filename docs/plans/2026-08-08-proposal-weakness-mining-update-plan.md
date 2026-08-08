# Update Plan: Weakness Mining paragraph in paper/proposal.tex

Created: 2026-08-08

## Goal

Revise the Weakness Mining paragraph in §Self-Harness Optimization (`sec:sho`) so it describes the stage *as implemented* — at the minimum depth a technical reviewer needs to understand the design and re-implement it. Keep the φ(r_i) equation; keep the paragraph's length within roughly 1.5× the current text. Do not import implementation detail (module names, file formats, hashes, CLI commands) into the paper.

## Why the paragraph is now inaccurate or under-specified

The implementation resolved five things the current text either overstates or leaves unimplementable. Each is a required content delta:

| # | Current text says | Implementation reality | Required change |
|---|---|---|---|
| D1 | "Each sub-call is additionally scored by the environment's synthesized sub-verifier, so a failed run carries a per-child correctness label" | Scoring is **post-hoc over recorded traces**, and child prompts are **model-authored** under the sparse harness — a sub-call is scored only when its prompt parses as a checkable sub-problem; otherwise it is *uncheckable*, not wrong. A record whose sub-calls are all uncheckable is **ungrounded**: its failing level falls back to model judgment. | State that sub-verification is applied offline to recorded sub-calls; per-child labels are three-valued (correct / incorrect / uncheckable); the failing level is a *checked fact* only when at least one child was checkable, and records are marked grounded vs. ungrounded accordingly. This distinction is load-bearing for the Appendix-B ablation. |
| D2 | The structured record contains "the agent behavior causally associated with it, the implicated harness mechanism" (source unstated) | Two of the four signature components are **derived** (verifier cause from the deterministic verifier; failing level from sub-verifier grounding), and two are **judged by the same fixed model** (causal status, agent mechanism) labeling a *bounded, mechanically compressed digest* of the trace — never the raw trajectory — with outputs **validated against a closed taxonomy** (re-asked on violation; unvalidatable responses recorded as unattributed rather than coerced). Each mechanism maps to exactly one of the nine declared surfaces. | Add two sentences: (a) the derived-vs-model-judged split and the mechanical digest (no second model summarization step upstream of mining); (b) the closed mechanism vocabulary, each member mapping to one editable surface, with off-vocabulary responses rejected rather than coerced. |
| D3 | "clustered by the verifier-grounded signature" (method unstated) | Clustering is **exact agreement on the 4-tuple** — a deterministic group-by, not embedding similarity — and the bundle ranks patterns by **distinct-instance support** (repeated attempts on one flaky instance cannot inflate rank), then a fixed actionability score. | Add one sentence after the equation: clusters form by exact agreement on φ; patterns are ordered by distinct-instance support, then actionability. |
| D4 | "records verifier outcomes and recursive execution traces" (failures only, implicitly) | **Every** run is persisted — passes, failures, and runs terminated by resource limits (recorded with a resource-terminated outcome rather than dropped) — so pass rates and exclusions are recomputable from the round's artifacts. | One clause: all runs, including resource-terminated ones, enter the round record; the bundle carries integrity counts (unattributed, ungrounded, terminated) so downstream stages can weigh the evidence. |
| D5 | Evidence bundle = "recurring, actionable failure patterns" (contents unstated) | Each pattern carries cluster size, mechanically computed shared symptoms, representative instances with supporting excerpts, and full provenance (harness identity, configuration, prompts); the bundle **describes weaknesses and never prescribes edits** (evaluator/optimizer separation). | One clause on pattern contents + the no-prescription property; a short clause that every artifact in the chain is persisted and auditable. Keep to one sentence total — this is the auditability claim, not a systems description. |

## Edit specification

**Location:** the single paragraph beginning `First, \textbf{Weakness Mining}` plus the `\phi(r_i)` equation block in `paper/proposal.tex` (§sec:sho, first of the three stage paragraphs).

**Keep unchanged:** the equation; the held-in/source-environment framing; the opening sentence's structure ("runs the current harness on short held-in instances...").

**Rewrite shape** (suggested order, one point per sentence, per the deltas above):
1. Opening sentence as-is, extended with D4's all-runs clause.
2. D1: post-hoc sub-verification, three-valued labels, grounded vs. ungrounded.
3. D2a: structured record — two components derived, two judged by the fixed model over a bounded mechanical digest.
4. D2b: closed mechanism vocabulary keyed to the nine surfaces; off-vocabulary rejected as unattributed.
5. Equation (unchanged).
6. D3: exact-match clustering; distinct-instance-support-then-actionability ordering.
7. D5: bundle contents, no-prescription property, one auditability clause.

**Tone constraints:**
- Reviewer-implementable, not system documentation: name *properties and decisions* (three-valued labels, closed vocabulary, exact-match clustering, mechanical digest), never artifacts (`bundle.json`, `runs.jsonl`, sha256, module names).
- No results or observations from test rounds (rejection rates, grounding coverage) — those belong in the eventual results section, not the method.
- Keep LaTeX conventions of the file: `\sh{}`, `\rlm{}` macros, `\cref` for references, no new packages.

### Suggested replacement text (directional draft — edit freely)

> First, **Weakness Mining** runs the current harness on short held-in instances from a source environment, recording the verifier outcome and full recursive execution trace of every run, including runs terminated by resource limits. Each recorded sub-call is then scored post hoc by the environment's synthesized sub-verifier; because sub-problems are authored by the model itself, a sub-call may be correct, incorrect, or uncheckable, and the level at which the error signal first appears is treated as a checked fact only when at least one sub-call was checkable — records without any checkable sub-call are marked ungrounded and their failing level falls to the proposer's judgment, the distinction exercised by the ablation of \cref{app:ablations}. Each failure is converted into a structured record: the verifier-level cause and the failing level are derived from the verifier and sub-verifier, while the causal status of the associated agent behavior and the implicated mechanism are judged by the same fixed model from a bounded, mechanically compressed digest of the trace, with judgments validated against a closed vocabulary of mechanisms — each mapping to exactly one editable surface — and unvalidatable judgments recorded as unattributed rather than coerced. Failures are then clustered by exact agreement on the verifier-grounded signature
>
> φ(r_i) = (verifier cause, failing level, causal status, agent mechanism),
>
> and the resulting evidence bundle ranks each recurring pattern by the number of distinct instances it affects and its estimated actionability, carrying cluster size, mechanically computed shared symptoms, and representative trace excerpts. The bundle describes weaknesses without prescribing edits, and every artifact behind it — traces, per-child labels, digests, judgments — is persisted for audit.

(~40% longer than the current paragraph; trim the auditability clause first if space is tight.)

## Adjacent ripple edits (flagged only — separate decision, not in this edit)

- **§Timeline, "Completed work":** now understates the completed stage (trace clustering → the full mining stage with sub-verifier grounding and auditable bundles); "harness versioning" is listed as Week-2 future work but harness identity/serialization already exists.
- **§Ablations, sub-verification paragraph:** under the sparse initial harness, checkability can be near zero even *with* the sub-verifier (model-authored child prompts may not parse) — the ablation text may want one clause acknowledging that grounding coverage is itself a measured quantity, so the with/without comparison stays honest.
- **§Analysis:** "frequency of each mined failure pattern" — consider stating frequency in distinct-instance terms to match the bundle's ranking.

## Verification

- The revised paragraph states all five deltas (D1–D5) and still reads as method, not implementation documentation.
- `\phi(r_i)` equation and all macros/citations unchanged; document compiles (`pdflatex` or the repo's usual build) with no new warnings.
- A reviewer reading only this paragraph plus \cref{fig:edit-surfaces} could re-implement the stage's contract: what is recorded, what is checked vs. judged, what the signature is, how clusters form and rank, what the bundle contains.
