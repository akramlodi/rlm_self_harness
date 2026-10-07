# Harness proposal interface

The proposer selects evidenced interventions and then supplies replacements in one response. The host materializes each edit against the incumbent, checks it locally, and validates the surviving edits as one combined harness. Admission does not establish semantic correctness.

## Model response

New responses use `proposal-selection/v3`, with `selections` followed by `candidates`. Each selection names `pattern_index`, `surface`, a 1–600-character `reason`, and 1–12 admitted `evidence_refs`. Each candidate matches one selection and includes `revision`, `incumbent_behavior`, `observed_failure`, `behavioral_change`, `edit`, `predicted_effect`, and `regression_risks`.

Select at most `k` edits, on distinct surfaces, using the existing evidence-dependent routes. Normally a pattern authorizes only one edit. Invalid candidates do not permanently occupy a surface; retained valid candidates keep their identity during the existing repair attempt.

An optional `activation_pair` label permits exactly two edits for one pattern: one **S8 helper or S10 skill**, and one **S2 decomposition or S3 execution instruction** that invokes it. Both selections and candidates carry the same identifier (1–64 letters, digits, underscores, or hyphens). The pair consumes two slots. Both routes must already be eligible; the exception introduces neither new routes nor a surface quota. A skill removal cannot be paired. Explain why current discovery/instructions are insufficient, the caller, and the capability's input/output contract in the existing explanation fields.

If either member fails admission, both are withdrawn. The existing repair attempt can replace both or omit both, retaining the label and capability/caller roles; it can retarget to eligible unoccupied surfaces. It cannot silently turn the pair into a singleton. Unrelated valid edits remain intact.

For example, selections for `(pattern 0, S8, activation_pair="count-operation")` and `(pattern 0, S3, activation_pair="count-operation")` must have two matching candidates. This describes membership only; complete reasons, references, and edits are still required.

## Text and persisted proposals

Instruction replacements use `literal-text/v1`: write literal instruction text, and let the host encode template braces once. The existing custom-tools marker and preflight checks remain authoritative. S8 supplies one named helper, S10 one named skill, and all materializers retain their single-surface checks.

New `proposal.json` envelopes use `shrlm-proposal/v2`. For paired candidates the host writes reciprocal metadata:

```json
"activation_pair": {"id": "count-operation", "partner_candidate_id": "r01-c02-s3"}
```

Frozen results preserve these identities for replay. `load_candidates` checks individually admitted members for reciprocal membership, matching signature/incumbent, exactly two members, and allowed surface roles before evaluation. Missing, corrupt, or rejected partners exclude the pair. Use this batch loader at the evaluation boundary; `load_candidate` alone performs local preflight and cannot establish pair completeness.

Current readers accept genuine unpaired `shrlm-proposal/v1` artifacts. Pair metadata is invalid in v1; older v1-only readers reject all new v2 envelopes rather than accidentally evaluating an orphan. Harness serialization itself is unchanged. Prompt, validator, evidence-selector, and compact-history identities were updated to avoid reusing older-contract responses.

## Evidence, intent, and history

The final proposer evidence includes `failing_level_detail`, `causal_status_detail`, and `agent_mechanism_detail`, bounded to 600 characters each including truncation markers. Missing legacy details say `not recorded`. Complete cited operations, verification limits, and the total 32,000-character evidence budget remain in force. A representative explanation is a model assessment, not proof that all records in a broad `other` bucket share that cause.

New diagnoses assess the cited defect as `unresolved`, `recovered`, or `unknown`; an omitted live field becomes `unknown`. Legacy absence remains `not_assessed`. Existing causal-status and coverage-basis checks still apply. Recovered-only patterns remain visible but cannot authorize edits; complete unresolved representatives take priority over uncertain representatives. Inventory `resolution_counts` describes the available diagnosis records, not independently verified causes. Recovered operations can appear as contrast. A residual cost/failure requires evidence of that separate consequence.

After complete failure cores, the packer tries a relevant complete passing held-in contrast before optional operations, preferring the same task instance. Shared operation names are a relevance heuristic, not proof of equivalent semantics or correct intermediates. Missing or oversized contrast is reported explicitly.

`predicted_effect` should state a task condition, changed operation, and expected benefit. `regression_risks` should state relevant already-working behavior, why it should survive, and uncertainty or cost. Evidence IDs are provenance, not constants for replacement instructions. These descriptions and pair membership survive bounded history compaction; they remain predictions. Batch results do not assign individual contribution or prove protected behavior was preserved.

This interface adds no regression bank or held-in scoring stage and changes no validation samples, repeats, promotion thresholds, or experiment settings. See [the implementation verification](analysis/2026-10-04-grounded-coordinated-proposals-results.md) for deterministic and live observations.
