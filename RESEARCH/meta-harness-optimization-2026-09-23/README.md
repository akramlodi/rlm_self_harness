# Research: general improvements to harness optimization

September 23, 2026.

Start with the [executive summary](executive_summary.md), or read the [full report](full_report.md).

| Artifact | Contents |
| --- | --- |
| [Full report](full_report.md) | Evidence, six prioritized recommendations, surface examples, research grounding, and evaluation approach |
| [Edit-by-edit appendix](appendices/edit-by-edit.md) | Analysis of all 16 tested edits and 11 distinct untested drafts |
| [Prompt templates](appendices/prompt-templates.md) | Task-agnostic mining/proposal prompts, capability cards, and illustrative interventions |
| [Census](data/census.json) | Measured selection constraints, prompt/history sizes, candidate mapping, and source hashes |
| [Surface table](data/surface-funnel.csv) | Per-surface eligibility, evidence expansion, testing, and promotions |
| [Mining record inventory](data/mining-records.json) | All 239 held-in failure diagnoses, with provenance |
| [Runtime probes](data/runtime-probes.json) | Verified runtime/S9 behavior and isolated validation of the two rejected S10 drafts |
| [Sources](sources/README.md) | Primary papers, local evidence quality, and unresolved questions |
| [Research scope](research_notes/scope.md) | Questions, constraints, and methodology |

Reproduce the census and local probes without model calls:

```bash
uv run python RESEARCH/meta-harness-optimization-2026-09-23/data/build_census.py
```

The narrative and prompt templates are authored analysis. The script rebuilds the quantitative census, record inventory, surface CSV, and runtime probes; it does not rewrite the narrative.

The [original report with exact edit appendices](../../docs/analysis/oolong-pairs-2026-09-23/report.md) remains the record of what was proposed and promoted. This research proposes changes to the general optimizer. It does not modify the paper, optimizer, or stopped experiment.
