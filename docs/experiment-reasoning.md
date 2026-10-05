# Inspect saved experiment reasoning

New experiment calls automatically save the reasoning fields returned by the provider. This covers mining runs, validation, final evaluation (including Lambda adapters), attribution, proposals, repairs, child calls, compaction, and answer fallbacks. No configuration change is needed. Request settings, budgets, attempts, concurrency, scoring, and promotion rules stay unchanged.

These are provider-returned observations, not a reconstruction of hidden reasoning. A provider may return full text, a summary, opaque data, or nothing under the existing request settings.

## Output locations

Each response is saved before normalization, response validation, or execution. Existing artifacts link to immutable JSON files through an optional `llm_observations` list.

| Artifact you start from | Observation location relative to that artifact's directory |
|---|---|
| Any mining, validation, evaluation, or Lambda `runs/<run_id>.json` | `<run_id>.llm_calls/<call_id>/` |
| Mining `attributions.jsonl`, in each attempt | `llm_calls/attribution/<call_id>/` |
| `work/proposal_result.json`, in each attempt | `llm_calls/proposal/<call_id>/` |
| `work/proposal_failure.json`, for a terminal proposal failure | `llm_calls/proposal/<call_id>/` |
| `proposals_complete.json`, in each attempt | `work/llm_calls/proposal/<call_id>/` |

A run's top-level list indexes all observations, including calls that never produced a completed iteration. Iterations and successful child completions also carry their own references. Nested completions use the outer run trace's directory as their anchor. Failed socket results retain call references; the run index includes their events even where the existing trajectory omitted a failed child.

Each call directory contains separate `started-*.json`, `response-*.json`, and `finished-*.json` files. A client retry can produce multiple response files under one call ID. Do not infer event order from filenames; use `recorded_at` and the response ordinal in `attempt_id`. A started call without a finished event is incomplete. A killed process can leave committed observations before a trace exists; interrupted-run recovery indexes those files without rerunning the model.

## Read an observation

References include `path`, `sha256`, `call_id`, and, for response events, `attempt_id`. Resolve paths from the owning artifact's directory, never the working directory. The reader verifies the hash and rejects paths that escape the owner.

```python
import json
from pathlib import Path
from shrlm.optimization.llm_observation_store import read_observation

trace_path = Path("experiment/.../runs/task__a01.json")
trace = json.loads(trace_path.read_text())
for reference in trace.get("llm_observations", []):
    record = read_observation(reference, trace_path.parent)
    if record["event"] == "response":
        print(record["purpose"], record["availability"])
        print(json.dumps(record.get("reasoning", {}), indent=2))
```

For an attribution, read each line's `attempts`; for a proposal result, read `result.attempts`. A sealed proposal marker's references are rebased to the marker's directory. Reasoning remains separate from `raw_response`, the final answer, executable code, and prompt history.

## Interpret the fields

Records use `format: rlm-llm-observation/v1`. They identify the owning run or stage, purpose, requested model, available provider response ID/model, parent call, and available depth, iteration, batch slot, or meta attempt. Raw selected assistant content is retained before normalization. Original reasoning field names are preserved: OpenAI-compatible `reasoning_content`, `reasoning`, and `reasoning_details`; Anthropic thinking blocks; Gemini thought parts and signatures; and Azure's recognized inline analysis blocks. Opaque bytes use explicit base64 encoding.

| `availability` | Meaning |
|---|---|
| `returned` | Readable reasoning text or a summary was supplied; `reasoning_kinds` distinguishes them. |
| `opaque_only` | Only encrypted/redacted/signature data was supplied. |
| `not_returned` | An instrumented client received a response without readable or opaque reasoning. Present empty fields remain in the payload. |
| `unsupported_client` | A string completion succeeded through an uninstrumented/custom client. |
| `legacy_unavailable` | A text-only cache entry was replayed without captured observations. |
| `no_response` | The observed call failed before a provider response was captured. |

A missing `reasoning_tokens` count is null, not zero. When supplied, the count is diagnostic: existing usage summaries and cost ledgers remain authoritative, and reasoning tokens are not added to billed output tokens again.

## Cache, failures, and historical experiments

New attribution and proposal cache entries include portable observation payloads. A cache hit copies them into the current experiment with `cached: true` and `source_call_id`; it does not call the provider or add spend. Proposal budget-exhaustion cache entries preserve the failed response too. Legacy text-only entries remain free to replay, with `legacy_unavailable` recorded locally.

Historical traces without references remain readable and byte-identical. Completed stages are not rewritten or backfilled. A missing or corrupt referenced file is a persistence error, not a reason to call the model again.

If saving an observation fails, the provider's available usage is accounted once and the error propagates through the experiment's persistence boundary. It is not classified as an incorrect answer or retried as a transport failure. Worker error reports preserve this classification and available usage. Already committed files remain inspectable. Atomic publication protects against partial files; it does not promise durability through a machine or power failure.

Captured reasoning is for inspection only. It is not added to weakness digests, proposal evidence/history, verifiers, or future model prompts. In particular, saving held-out reasoning does not make it available to mining. Saving it does not turn on a provider reasoning mode or enable streaming support.
