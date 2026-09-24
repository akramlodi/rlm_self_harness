# OOLONG-Pairs harness edit audit — September 23, 2026

Start with [the full report](report.md). It explains the five promoted edits, eleven tested but unpromoted edits, and eleven distinct untested drafts from all eight rounds of `experiment_oolong_pairs_dsv4f_20260916_131837`.

The report includes exact replacement text, code, policies, and skill drafts in Appendices A–C. It compares exact passes, F1, missing/extra pairs, and cost, and distinguishes observed behavior from proposed explanations.

| File or directory | Contents |
| --- | --- |
| [report.md](report.md) | Human-readable analysis and all exact-edit appendices |
| [rounds.csv](rounds.csv) | Eight within-round baseline/candidate comparisons |
| [audit.json](audit.json) | All proposals, before/after surfaces, response violations, and 160 paired validation attempts |
| [edits/](edits/) | 27 exact edit JSON payloads and 16 stored-surface diffs |
| [trace-evidence.json](trace-evidence.json) | Selected complete operations and outputs, with trace hashes and coordinates |
| [runtime-retry-audit.json](runtime-retry-audit.json) | Persisted retry metrics from rounds 6 and 8; nested metric objects are not unique-call counts |
| [extract_audit.py](extract_audit.py) | Rebuilds the inventory, metrics, edit files, and report appendices from local experiment artifacts |

Rebuild the generated artifacts without model calls:

```bash
uv run python docs/analysis/oolong-pairs-2026-09-23/extract_audit.py
```

The prose analysis and selected trace/retry audits are preserved independently of that rebuild. The original-versus-edited long-context evaluation is separate: its [saved matched comparison](../../../experiment_oolong_pairs_dsv4f_20260916_131837/eval/paired_20260923T140326Z/comparison.md) was set up for the same 16 tasks with three attempts per task, and was stopped at the user’s request after three edited attempts on one task. All evaluation and monitoring processes have exited. The report records this partial result without claiming general long-context improvement. The repository paper has not been changed.
