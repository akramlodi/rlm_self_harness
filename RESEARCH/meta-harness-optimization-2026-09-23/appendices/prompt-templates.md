# Appendix B — task-agnostic prompt templates and intervention examples

These are draft replacements or refinements for portions of the meta-harness prompts, not instructions to concatenate every block. Keep the current budget, expose relevant blocks only, and version any implemented schema changes. No block below depends on OOLONG categories, record layouts, answer IDs, or predicates.

## 1. Host-supplied capability cards

Generate these from maintained contracts and test them against the implementation. Showing key names alone is insufficient. [Current implementation](../../../rlm/environments/local_repl.py), [runner](../../../shrlm/runner.py), [surface definitions](../../../shrlm/rlm_harness.py).

```text
S6 runtime policy
Trigger: root call dispatch; some fields also affect RLM construction.
max_retries + retry_on_syntax_error: extra calls for syntax-classified error
  strings only, with the same prompt. Not a timeout retry or a limit on loops
  written by the model. max_retries is additional attempts, not total attempts.
max_batch_width: refuses an oversized batch; does not split, queue, or throttle it.
max_prompt_chars: refuses an oversized prompt; does not shorten it.
max_depth: becomes the RLM recursion-depth limit and is inherited by children;
  it remains subject to the experiment-owned ceiling.
The local policy object itself is not propagated to child environments.
Do not propose an arbitrary callable through a JSON-only policy field.
Required justification: identify the activating event and the behavior afterward.

S7 execution-result metadata
Input: formatted execution output and redacted variable type/length inventory.
Effect: returns bounded text for the next root turn.
Cannot: read unprinted variables, repair a child result, change global history,
  or affect a child's metadata formatting through this root-only hook.
Required justification: identify information currently lost in this output.

S8 REPL helper
Input: arguments supplied when the model calls the helper.
Effect: a pure or otherwise bounded operation in the configured root or child
  namespace. The helper is not automatically called when installed.
Use a clear name and docstring; specify the caller and preserved information.
Do not shadow reserved runtime functions or require hidden task answers.

S9 answer middleware
Input: detected final answer and redacted variable type/length inventory.
accept(text): accepts that text, including an explicitly transformed answer.
redirect(nudge): resets readiness and asks the root for another turn.
Cannot: inspect record values or independently verify their semantic labels.
Required justification: an answer-visible defect under a stable answer contract,
  plus examples of valid answers that remain accepted.

S10 skill
Effect: exposes a name/description; the body is available only when loaded.
Use for a conditional multi-step procedure, not a second global instruction.
Required justification: a recognizable retrieval trigger, required inputs,
  concrete steps, and evidence that this procedure would be useful.
Do not claim a stored skill changes behavior if the model never loads it.
```

For S1–S5, include the additional fact that their shared system prompt reaches recursive children. Have an instruction refer to the current subtask's contract rather than impose the root task's final format on every child. This avoids a general class of incompatible intermediate-return requirements.

## 2. Refine diagnosis around observations and recovery

Replace overlapping causal reminders with a compact reporting instruction:

```text
Diagnose the earliest supported operation whose wrong result remains relevant
to the failed outcome. Report what was observed separately from what you infer.

For that operation, state:
- the required property and the universe it applies to;
- the observed operation/result and its trace coordinates;
- whether its result was later discarded, corrected, retained, or is unknown;
- the downstream consequence that is actually supported;
- what remains unverified.

Distinguish lost information, incorrect semantic judgment, incorrect task-condition
implementation, runtime failure, and an already-recovered incident. Missing output
elements alone cannot distinguish these. A successful parse proves syntax, not
semantics. If the needed operation is absent from the evidence, say unknown.

Do not propose a remedy. A successful recovery may be recorded as a reusable
procedure example, but must not be labeled an unresolved cause of the final error.
```

This is a proposed extension of the existing attribution output, not a claim that the current schema accepts all these fields. The host can verify coordinates and required fields; it cannot validate semantic causality merely because the explanation sounds precise.

## 3. Select an intervention form and surface together

```text
For each well-supported unresolved operation, briefly compare the viable forms:
an instruction, an executable helper/policy, or a conditional reusable procedure.
Discard forms that cannot see the necessary information, run at the right point,
or perform the claimed action under the capability cards.

Choose the smallest effective change, then its surface. Explain why this form is
preferable to the nearest alternative. Do not generate replacements for losing
alternatives. One evidence-supported edit is sufficient; do not fill a quota.

For a helper or skill, identify how it will be discovered and used at the relevant
operation. For an always-on instruction, explain its scope at root and child
levels. Prefer replacing a conflicting example to appending another rule.

Before writing the replacement, provide one tiny distinguishing example:
what the incumbent does, what the candidate would do differently, and why that
difference addresses the observed operation. Also give a valid case preserved.
```

This asks the model to choose implementation form, not to populate a surface quota. Keep the distinguishing example short and grounded in held-in or synthetic evidence.

## 4. An S8 example, followed by the abstraction that should generate it

Illustrative function, not a deployed candidate:

```python
def merge_grouped_values(parts):
    """Merge key-to-list batches, preserving repeated keys and value multiplicity."""
    merged = {}
    for part in parts:
        for key, values in part.items():
            if not isinstance(values, list):
                raise ValueError("Each grouped value must be a list")
            merged.setdefault(key, []).extend(values)
    return merged
```

Distinguishing fixture:

```text
Input: [{"A": [1, 1]}, {"A": [2], "B": [3]}]
Overwrite merge: {"A": [2], "B": [3]}
Candidate merge: {"A": [1, 1, 2], "B": [3]}
Preserved case: disjoint keys produce the same groups under both operations.
```

This is useful only when concatenation is the required semantics. It does not establish that values are accurate, fix duplicated processing upstream, or handle tasks that demand replacement rather than accumulation. A schema mismatch raises an error rather than silently deleting information.

The proposer instruction that could discover similar helpers:

```text
Look for deterministic operations the trace repeatedly implements or gets wrong:
parsing, joining, merging, counting, preserving order, or validating a schema.
Can a small parameterized helper prevent the observed error at that operation?
Define its input/output contract from the task's requirements; do not assume
deduplication, set semantics, or replacement. Show a distinguishing fixture and
where execution would call it. Keep semantic judgment outside the helper unless
an actual deterministic verifier is available.
```

The fixture also has valid analogues in grouped measurements, document extraction, and graph edge accumulation. The meta-prompt need not mention any of those domains.

## 5. An S3 example: preserve information until it is no longer needed

Illustrative replacement idea:

```text
Before each aggregation, identify which remaining task conditions need individual
items, multiplicity, ordering, or provenance. Retain those properties until the
conditions have been evaluated. If a transformation discards one, show why no
remaining computation needs it. Derive the checks from the current subtask;
do not assume every task needs the same intermediate representation.
```

The prompt that should generate it:

```text
Trace each required task condition to the data used by the operation that enforces
it. Name any information discarded before its last required use. Propose a change
at that transformation, with a small boundary case that behaves differently.
Do not substitute a generic coverage reminder for the missing condition.
```

A boundary case might distinguish “exactly one” from “at least one,” ordered from unordered output, or an asymmetric relation from a symmetric one. Choose the case from the task contract, not from a fixed benchmark template.

## 6. S5 and S10: repair a failed operation without discarding good work

S5 instruction form:

```text
When a sub-result is unusable, identify the failure class before retrying. Keep
verified independent results. Retry affected work only after changing the cause
of failure, and track repeated attempts so the same unsuccessful action is not
repeated indefinitely. Rebuild dependent results when their inputs change. Before
consuming the combined result, check the completeness required by this subtask;
do not silently treat a failed piece as an empty successful answer.
```

S10 procedure form should have a narrower retrieval trigger, such as “load when a batch mixes valid results with failed or schema-incompatible results.” Its body can then specify inspection, isolation, cause-specific repair, dependency invalidation, and completion checks. A child timeout may require reducing work or choosing a different operation; it is not evidence that a syntax retry will help.

The general prompt:

```text
Find a successful recovery sequence or a repeated unresolved recovery problem in
the held-in evidence. Identify its preconditions and the work it preserves. If it
requires several conditional steps and is useful only in that situation, propose
a skill with a precise load trigger. If it is a short general response to errors,
use the recovery instruction instead. Do not turn one repaired sample into a rule
to rerun all completed work; justify the dependency scope.
```

## 7. S7 and S9: target what the hook can observe

```text
For S7, point to the output segment that was omitted or overwhelmed and the later
decision that needed it. Preserve that segment within the existing bound, and
test short output, long output, and a terminal diagnostic. Do not summarize values
available only in the REPL inventory: it contains types and lengths, not values.

For S9, distinguish a reversible presentation change from an uncertain content
repair. If a transformation is demonstrably safe under the answer contract,
accept the transformed text. Otherwise use a specific nudge only when another
turn can plausibly repair the detected issue. Preserve valid alternative formats
and valid empty answers. Do not impose a task-specific format when this hook
cannot observe which task is being answered.
```

The correct result can be “no supported S7/S9 edit.” A missing surface promotion is not itself evidence of a defect.

## 8. A meaningful revision to a previous attempt

```text
Name the closest prior intervention, if any. Separate its observed activation,
batch outcome, dense-quality diagnostics, cost, and uncertainty from its proposed
explanation. State what this new intervention changes in response to that evidence.

A rejected but diagnostically promising batch may justify a different refinement;
it does not justify replaying an identical policy. A combined gain does not belong
to one member. If no relevant behavior was observed or no comparable metric exists,
say so. Do not classify missing measurements as a negative result.
```

The host should build the compact factual history. The model supplies the proposed revision, not fabricated measurements. Unknown metrics remain unknown; exact counts and dense scores retain their definitions and denominators.

## 9. Repair only the broken part

```text
These candidate IDs and surfaces are retained: {retained}.
These fields failed validation: {field_errors}.
Return corrections only for the failed fields or withdraw the affected candidate.
Do not regenerate retained edits. If two candidates compete for one surface,
choose one supported contender or withdraw them; do not submit both again.
If changing the target, recheck that target's capability and occupancy first.
```

This would require a narrow repair-response shape rather than accepting partial candidate objects in the current schema. It is especially useful for the observed 600-character selection-reason failures. Do not solve them by silently truncating explanations or weakening capability checks.
