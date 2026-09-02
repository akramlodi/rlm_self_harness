# Evaluation baselines

This package contains baseline-specific evaluation integrations.

## B1 / H₀

The sparse starting harness used by the Self-Harness optimization loop.

Implementation: `shrlm/rlm_harness.py`

## H₀*

The shipped upstream RLM reference harness reconstructed through the same
editable surfaces as H₀.

Implementation: `shrlm/rlm_harness.py`

## λ-RLM

A separate hand-designed inference method using a typed functional runtime.
For OOLONG-Pairs, the released source is missing the paper's specialized
Algorithm 5, so the local implementation reconstructs its bounded
SPLIT→MAP→PARSE→FILTER→CROSS path and identifies it explicitly as a paper
reconstruction. Other tasks continue through the pinned upstream runtime.
Each pairwise run persists its raw batch responses, retry rejections, parsed
record labels, label counts, and bounded call totals under
`metadata.pairwise_audit` in the run trace.

- Byte-identical upstream implementation: `shrlm/baselines/upstream/lambda_rlm.py`
- OOLONG-Pairs paper reconstruction: `shrlm/baselines/paper_lambda_rlm.py`
- Local evaluation adapter: `shrlm/baselines/lambda_rlm.py`
- License and provenance: `third_party/lambda-RLM/`

## SH-RLM

SH-RLM is the system under test, not a baseline. Its final harness is produced
dynamically by optimization and frozen under the experiment output directory.
