# Paper revision verification

Date: September 28, 2026. Implementation reference: `070d1a8493d79999ce4a5b01b19a2d0dc85b7637`. This records application of the [paper accuracy audit](2026-09-28-paper-optimization-accuracy-audit.md), including the author's decisions to keep the current split, joint batch promotion, and inclusive verifier-pass gate, and to remove child-verifier descriptions from the papers.

## Deliverables

- Both active manuscripts, [expanded](../../paper/proposal.tex) and [short](../../paper/neurIPS_short_paper.tex), use shared [main method](../../paper/optimization_method.tex), [study protocol](../../paper/experiment_protocol.tex), and [implementation appendix](../../paper/optimization_appendix.tex) sections.
- [Figure 2](../../paper/Figure_2.tex) shows admission/composition before held-out batch validation; its call tree has no correctness labels. [Figure 2B](../../paper/Figure_2B.tex) delegates to this maintained figure.
- [Figure 1](../../paper/Figure_1.tex) and [Figure A1](../../paper/Figure_A1.tex) identify their sparse H0 reference and distinguish it from the actual H0*R starting harness. Both appendices include the single full S1 listing.
- [The collaborator summary](../../meta-harness-improvements.md) describes the implemented changes and their limitations, including qualified historical dense-metric signals.
- [The project checklist](../../paper/checklist.tex) supplies 16 answers for the current methods/protocol draft, with incomplete study/release/author-attestation items explicitly marked rather than invented.

`proposal-old.tex` was initially converted to a compatibility entry point, then removed from the working tree during this task. That deletion was preserved. The historical draft remains available in git at the implementation reference above. The two active manuscripts are the maintained build targets. The vendor example, bibliography, and style file were not edited. Pre-existing deletions of research reports and paper notes/PDFs were left intact.

## Audit coverage

| Audit findings | Applied correction |
|---|---|
| #1–2 | Joint held-out comparison, fresh incumbent, inclusive pass-count inequalities, resource bands, and batch attribution in the main method, diagram, and appendix. |
| #3 | H0/H0*/H0*R distinguished; H0*R hand guidance disclosed; exact saved `initial` versus final comparison specified. |
| #4 | Child-verifier method, diagram labels, algorithm operations, metrics, and ablations removed. Final-answer verification and uncertain model diagnoses retained. |
| #5–7 | OOLONG-Pairs source and reconstruction; 20/10/10/40 tasks; retained task-level split and overlap; adaptive validation feedback; short/long claims scoped correctly. |
| #8–10 | Upcoming study separated from historical pilots; no unsupported preregistration, completed transfer, seeded-attempt, or significance claims. Three final-evaluation attempts kept separate from `v=1`. |
| #11 | `m*n_in + 2*v*n_ho` replaces obsolete validation/merge counts. Explicit condition counts, dated rates, usage coverage, and additional optimizer calls replace stale dollar projections. |
| #12–14 | Attribution uncertainty and coverage basis; many-to-many routes; distinct-instance ranking; whole-operation evidence, local definitions/corrections, contrasts, and component budgets. |
| #15–16 | Selection before replacement, admission ownership, abstention, retained-sibling repair, literal text, host escaping, materialization, and bounded preflight. |
| #17–18 | Bounded history and revision links, generic comparable metrics, potentially promising rejections, activation versus intended behavior, and unassessed coverage. Historical F1 examples remain in the collaborator summary, not as results of the forthcoming study. |
| #19 | Completed failures versus missing attempts, partial subjects, propagated operational failures, cache/resume identity, and denominators. |
| #20–21 | Actual scope/capabilities of S1–S10, root-local hooks, recursive versus bare calls, skill invocation and history-dependent token costs. |
| #22 | Provider-returned observation artifacts, availability, cache provenance, unknown token counts, and separation from optimizer input. |
| #23 | Appendix algorithm includes attribution, composition before validation, actual promotion-based patience reset, and freeze after loop termination. |
| #24 | Fresh measured quality curves, combined surface promotions, configuration-derived environment labels, diagnostic coverage, and precise collapse proxy. |
| #25–26 | Optional ablations and training clearly prospective; a final-surface counterfactual is defined; stale schedule and unsupported backbone/hardware promises removed. |
| #27–28 | Missing short-paper input eliminated, full-contract references supplied, duplicate appendix/figure content consolidated, and historical draft retired. |
| #29 | Checklist filled and included in both active papers. Future results, release artifacts, resource totals, license inventory, and author attestations remain explicitly pending. |

## Verification performed

1. Checked documented task counts, repetitions, edit/round ceilings, patience, initial harness, promotion thresholds/bands, and decoding against parsed TOML. Checked listed attribution/proposal/taxonomy/digest/evidence versions and attempt ceilings against current source constants.
2. Recomputed reference harness serialization hashes. H0*R is `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`, recorded in the appendix as a registry reference, not a substitute for the forthcoming run's saved initial artifact.
3. Checked every nonempty line of the full H0 S1 listing against the implementation's literal template. Checked the capability and routing tables against the runtime, taxonomy, and conditional operation-support rules.
4. Recursively expanded both active manuscripts: all includes resolve, all 26 labels are unique, all references resolve, and all 15 cited bibliography keys exist. Collaborator-summary links resolve. Scans found no obsolete child-verifier account, dual-split/per-edit gate, old run totals, seeded-run claim, or unanswered checklist placeholder in either expanded manuscript.
5. Built both active papers with `latexmk`/pdfLaTeX and BibTeX to convergence. Both are **20 pages**, including the appendix and full template checklist; the main text occupies four pages. Final logs have **zero undefined references/citations, duplicate labels, overfull boxes, or fatal errors**. Benign underfull-box warnings remain, as does the expanded draft's existing missing-author warning.
6. Inspected rendered loop figures, main protocol, runtime/capability table, routing table, promotion equation and algorithm, and version/observation sections. Shortened a diagram label that crowded its neighbor and rebuilt both papers.
7. `git diff --check` passed. No application code/configuration was changed by this task; application tests and paid model experiments were not run.

The machine's BasicTeX installation lacked Helvetica metrics. The TeX Live `helvetic` package was downloaded into a temporary TEXMF tree, leaving the repository style and global installation unchanged. An initial offline Tectonic attempt lacked cached `tabularx`; the successful builds used the installed pdfLaTeX toolchain instead.

From `paper/`, the successful build commands were:

```bash
env TEXMFHOME=/tmp/rlm-paper-revision-20260928/texmf \
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/rlm-paper-revision-20260928 proposal.tex
env TEXMFHOME=/tmp/rlm-paper-revision-20260928/texmf \
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/rlm-paper-revision-20260928 neurIPS_short_paper.tex
```

On a complete TeX installation the temporary `TEXMFHOME` override is unnecessary. PDFs, logs, extracted text, verification JSON, and inspected page images are in `/tmp/rlm-paper-revision-20260928/`; they are local verification artifacts, not committed study results.

| Final PDF | SHA-256 |
|---|---|
| `proposal.pdf` | `98da6192678cbd6e69907ef89359eeaa568d2cd54bf1e047acd764dc9e3a736a` |
| `neurIPS_short_paper.pdf` | `2d8ee4d50997d2e2ea150960592b5c69d5a29f6c1a02a39c2457fc99dc5a33d0` |

This verification establishes consistency with the inspected implementation and successful document assembly. It is not an independent scientific review, a venue-compliance check, a new experiment, or verification of the literature's substantive claims. The forthcoming study still needs its own frozen artifact identities and completed measurements before reporting efficacy.
