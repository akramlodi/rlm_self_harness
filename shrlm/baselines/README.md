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

- Upstream implementation: `shrlm/baselines/vendor/lambda_rlm.py`
- Local evaluation adapter: `shrlm/baselines/lambda_rlm.py`
- License and provenance: `third_party/lambda-RLM/`

## SH-RLM

SH-RLM is the system under test, not a baseline. Its final harness is produced
dynamically by optimization and frozen under the experiment output directory.