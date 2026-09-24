# Sources and evidence quality

Accessed September 23, 2026. External searches used public research topics, not private experiment content. No private artifacts were uploaded.

Quality scale used here: **A** = directly inspected primary artifact or reproducible local observation; **B** = primary research supporting a design precedent in another setting; **C** = secondary synthesis used only for context; **D/E** = weak or unverified material, not relied on. These grades describe source proximity; they are not probabilities that a causal explanation is correct. Recommendations remain untested hypotheses.

## Repository evidence

| Source | Owner/date | Grade | What it establishes |
| --- | --- | --- | --- |
| [Original complete edit report](../../../docs/analysis/oolong-pairs-2026-09-23/report.md) and [audit JSON](../../../docs/analysis/oolong-pairs-2026-09-23/audit.json) | This repository, September 23, 2026 | A | Edit inventory, within-round outcomes, exact replacements and trace-supported limitations |
| [Reconstructed census](../data/census.json) | This investigation, September 23, 2026 | A | Eligibility/expansion counts, history sizes, all sixteen candidate mappings, inspected-source hashes |
| [239 mining records](../data/mining-records.json) | Experiment artifacts, analyzed September 23, 2026 | A | What the attributor actually reported; does not establish that those diagnoses are true |
| [Runtime probes](../data/runtime-probes.json) | This investigation, September 23, 2026 | A | Actual retry, refusal, and answer-decision semantics with synthetic inputs |
| [Earlier cross-experiment investigation](../../../docs/analysis/2026-09-15-task-agnostic-proposer-review.md) | This repository, September 15, 2026 | C for historical synthesis | Recurrence across older tasks/models; not a controlled performance comparison |

The census records repository HEAD and SHA-256 agreement between the current and frozen copies of ten relevant implementation files. The experiment directory remains the authoritative source; copied summaries are for navigation and reproducibility.

## External primary research

1. **Agrawal, Lakshya A., et al. (July 25, 2025). _GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning_, arXiv:2507.19457v1.** [Paper](https://arxiv.org/pdf/2507.19457v1). Grade B. Relevant locations: §3.2, printed p. 6, module selection and execution feedback; §4, printed p. 7, train/validation/test access. Used for the principle of component-specific feedback and deliberate component access. This investigation does not claim its round-robin schedule or reported gains transfer to these ten surfaces.
2. **Yuksekgonul, Mert, et al. (June 11, 2024). _TextGrad: Automatic “Differentiation” via Text_, arXiv:2406.07496v1.** [Paper](https://arxiv.org/html/2406.07496v1). Grade B. Relevant locations: §2 and Appendix B. Used for directing feedback to editable components with constraints and history; not as evidence that generated causal explanations are calibrated.
3. **Zhang, Shaokun, et al. (February 17, 2024). _Training Language Model Agents without Modifying Language Models_, arXiv:2402.11359v1.** [Original paper](https://arxiv.org/html/2402.11359v1). Grade B. Relevant locations: §2.1 and Appendix D.1. Later versions use the title _Offline Training of Language Model Agents with Functions as Learnable Weights_. Used for incremental function updates grounded in execution history and future-useful functions; no benchmark gain is projected onto this repository.

All substantive external claims in the main report link to the primary paper at their point of use. No secondary paper summaries, search snippets, or social-media performance claims supply the recommendations. GEPA's HTML rendering exceeded the browser fetch limit; its PDF supplied the inspected text instead.

## Unresolved questions

- Will capability-aware routing and evidence allocation increase effective behavior changes, not merely valid candidate counts?
- Will the model discover and call S8 helpers or load S10 procedures without companion instruction edits?
- How much of the observed gain is semantic improvement, completion recovery, or stochastic variation?
- Can a smaller history retain relevant negative evidence without encouraging duplicate hypotheses?
- Which candidate fixtures predict downstream performance across different environments?

The proposed offline replay and subsequent unchanged validation protocol are needed to answer these. No paid model replay or new evaluation was run for this research.
