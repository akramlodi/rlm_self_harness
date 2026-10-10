# Exact promoted edits and proposal claims

## Round 1: S2

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_01/work/proposal_result.json`


**incumbent_behavior**: The root split 188 items into 7 chunks of 30, issued 7 sub-calls, but the merge loop at iteration 12 processed only chunks 0-3 before a TypeError (NoneType not callable) terminated execution. No later iteration re-processed chunks 4-6. The S2 example shows a CHUNK=300 loop with no coverage verification or retry guidance.


**observed_failure**: Final answer missed user 16357 and all 8 pairs involving it because chunks 4-6 (items 120-187) were never parsed. The merge loop stopped after chunk 3; the 'Missing labels: 0' check at iteration 16 ran on a partial all_labels list that never received those chunks' data, so it reported no missing labels incorrectly.


**behavioral_change**: After the merge loop, the model will verify that all chunks were processed by checking len(parsed_chunks) == len(chunks) and that all_labels has no None entries. If coverage is incomplete, it will re-run only the missing chunks instead of proceeding with partial data. This directly prevents the demonstrated failure where partial merge was accepted as complete.


**predicted_effect**: After the merge loop, the model will explicitly check that all input items received a label. If chunks were skipped due to an error or incomplete loop, the coverage check will detect the gap and trigger re-processing of only the missing chunks, preventing the partial-data final answer seen in the failure.


### Exact replacement

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n" + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.

**Coverage check after merging**: After you merge results from all chunks into
a single data structure (e.g., a list of labels), verify that every input item
has been assigned. Print the count of items with missing assignments. If any
items are missing, identify which chunks they belong to and re-run only those
chunks with a corrected prompt. Do not proceed to the final answer until all
items have been assigned.
````

## Round 2: S4

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_02/work/proposal_result.json`


**incumbent_behavior**: The root classified 188 questions into 6 categories via 4 child RLM calls, merged results into all_classifications dict, checked that all 188 indices were present (coverage check passed), then immediately proceeded to extract user IDs and generate pairs. No verification of classification correctness was performed.


**observed_failure**: Only 6 users identified with 'description and abstract concept' questions, while gold answer shows many more such users exist. The coverage check (all 188 indices present) passed, but the classification labels themselves were wrong for many questions. The verifier returned wrong_format because the answer was incomplete.


**behavioral_change**: After merging child results, the root will run a verification step that samples 5-10 classified items and prints the question text alongside its assigned label for manual inspection. If any sample looks misclassified, the root re-runs the classification with a refined prompt. This catches systematic labeling errors before they propagate to pair generation.


**predicted_effect**: The root will spot-check classification quality before proceeding to pair generation. If the child RLM systematically mislabels 'description and abstract concept' questions as another category, the spot-check will catch it and trigger re-classification with a corrected prompt, preventing the incomplete answer seen in the failure run.


### Exact replacement

````text
After merging results from all chunks into a single data structure (e.g., a list of labels), verify that every input item has been assigned. Print the count of items with missing assignments. If any items are missing, identify which chunks they belong to and re-run only those chunks with a corrected prompt. Do not proceed to the final answer until all items have been assigned.

**Classification quality check**: After confirming full coverage, sample 5-10 items from each category (or from the smallest categories) and print the original question text alongside its assigned label. Review these samples: if any clearly belong to a different category, re-run the classification with a refined prompt that addresses the observed confusion. Do not proceed to the final answer until the classification passes this spot-check.
````

## Round 3: S3

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_03/work/proposal_result.json`


**incumbent_behavior**: The root at iteration 8 generates pairs by iterating user1_candidates (entity+abbreviation) as u1 and user2_candidates (exactly one entity) as u2, with u1 != u2. This only produces pairs where the entity+abbreviation user is listed first. At iteration 9, the root re-runs with the same logic (entity_and_abbrev x exactly_one_entity), producing the same 139 pairs. The final answer at iteration 10 outputs these 139 pairs, missing 44 gold pairs where the exactly-one-entity user should be first.


**observed_failure**: 44 pairs present in gold_answer are absent from produced_answer (e.g., (14630, 32435), (14630, 48114)). The pair-generation code only considers one ordering (entity+abbreviation user first), but the task requires all pairs where one user has entity+abbreviation and the other has exactly one entity, regardless of which user is listed first. The code never generates the reverse ordering.


**behavioral_change**: After generating pairs in one direction, the root will also generate pairs in the reverse direction (exactly_one_entity x entity_and_abbrev) and deduplicate by sorting each pair. For example, if user A has entity+abbreviation and user B has exactly one entity, the code will produce both (A, B) and (B, A) then keep only the sorted version. This ensures all 183 gold pairs are generated instead of only 139.


**predicted_effect**: The root will generate pairs in both orderings when the task has asymmetric conditions, producing the complete set of 183 gold pairs instead of the incomplete 139. The coverage check (|A| * |B| minus overlap) will catch any missing pairs before final answer submission.


### Exact replacement

````text
After merging results from all chunks into a single data structure (e.g., a list of labels), verify that every input item has been assigned. Print the count of items with missing assignments. If any items are missing, identify which chunks they belong to and re-run only those chunks with a corrected prompt. Do not proceed to the final answer until all items have been assigned.

**Pair generation for asymmetric conditions**: When the task requires pairs where one user satisfies condition A and the other satisfies condition B (with no ordering constraint), generate pairs in BOTH directions: first iterate A × B, then iterate B × A. Deduplicate by sorting each pair so the lower ID comes first. Print the total count and verify it equals |A| * |B| (minus self-pairs if A and B overlap). Do not assume the first-listed condition must be the first element of each pair.
````

## Round 4: S5

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_04/work/proposal_result.json`


**incumbent_behavior**: S5_recovery_instruction is empty string. The root encountered a TypeError when sorting mixed int/str user IDs (iteration 8), then manually normalized to int (iteration 9) and regenerated pairs. However, the run still ended incomplete — the verifier rejected the output.


**observed_failure**: The root's pair generation failed with TypeError on mixed types. Manual recovery (normalize to int) produced 351 pairs, but the verifier still found 84 missing pairs (F1=0.893). The recovery did not address the underlying coverage gap — the root had only 27 users with entity/human labels vs. 37 initially counted, indicating some users were lost during classification merging.


**behavioral_change**: When a TypeError occurs during pair generation (e.g., sorting mixed types), the root will: (1) print the types causing the error, (2) normalize all IDs to int, (3) re-run pair generation, and (4) verify the pair count matches |A| * |B| (minus overlap). This addresses the demonstrated failure where the root recovered the type error but did not verify coverage completeness afterward.


**predicted_effect**: The root will recover from type errors in pair generation by normalizing IDs and then verifying pair count against the expected formula. This catches cases where the user sets A and B are incomplete due to earlier classification errors, prompting the root to re-check its classification coverage before finalizing.


### Exact replacement

````text
If a code block raises a TypeError during pair generation (e.g., sorting mixed int/str IDs), print the types of the first few elements causing the error, normalize all IDs to int, and re-run pair generation. After regenerating pairs, verify the total pair count equals |A| * |B| (minus self-pairs if A and B overlap). If the count does not match, re-check which users are in sets A and B before proceeding.
````

## Round 6: S9

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_06/work/proposal_result.json`


**incumbent_behavior**: The root classified all 188 questions into 6 categories, printed category counts, spot-checked labels, then set answer['content'] = json.dumps(total_counts) and answer['ready'] = True. The final submitted answer was a category-count dictionary, not the required list of user-ID pairs. The S9 identity function accepted this dictionary unchanged.


**observed_failure**: The verifier rejected the output as wrong_format: no '(user_id_1, user_id_2)' pair and no 'No valid pairs found' marker. The root never constructed or verified the pair list; it stopped after classification quality checks. The S9 middleware accepted the non-conforming answer without intervention.


**behavioral_change**: S9 now checks whether the answer text contains at least one parenthesized integer pair (e.g., '(12345, 67890)') or the literal empty marker 'No valid pairs found.'. If neither is present, S9 returns redirect(nudge='The answer must list user-ID pairs in (id1, id2) format or state "No valid pairs found."') instead of accept. For example, a category-count dictionary would be redirected; a correct pair list or the empty marker would pass through unchanged.


**predicted_effect**: When the root submits a category-count dictionary or any other non-pair output, S9 redirects with a nudge explaining the required format. The root then has an opportunity to construct the pair list in a subsequent turn. Correct pair lists and the empty marker are accepted unchanged. This prevents wrong-format submissions from reaching the verifier.


### Exact replacement

````text
def accept_answer(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
    """Accept if answer contains at least one (int, int) pair or the empty marker; otherwise redirect."""
    if 'No valid pairs found.' in answer:
        return AnswerDecision.accept(answer)
    if re.search(r'\(\d+,\s*\d+\)', answer):
        return AnswerDecision.accept(answer)
    return AnswerDecision.redirect(nudge='The answer must list user-ID pairs in (id1, id2) format or state "No valid pairs found."')
````

## Round 7: S10

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_07/work/proposal_result.json`


**incumbent_behavior**: Root built user data from refined_classifications (iteration 19) without comparing against original input lines. Reported 58 unique users vs gold's 82+. Never checked whether all original lines were classified or whether any users were missing from the classified set.


**observed_failure**: Missing 7 group A users and 6 group B users from gold answer. The root's refined_classifications dictionary did not contain entries for all original input lines, but no coverage check against the original input was ever performed. The root compared two classification runs (labels vs labels3) but never verified against the source data.


**behavioral_change**: Adds a skill that provides concrete REPL steps to verify coverage: count original input lines, count classified items, identify which original lines are missing from classifications, and re-run only those missing chunks. This makes the abstract 'verify coverage' instruction actionable with specific code patterns.


**predicted_effect**: The root will have a concrete, loadable procedure for verifying coverage against original input. When classifications are incomplete (as in the held-in run where 58 users vs 82+ gold users indicates missing lines), the skill's step 3-4 will identify the missing items and trigger re-processing of only those chunks, preventing the downstream loss of entire user groups.


### Exact replacement

````text
## Use When
- You have classified items from multiple chunks and need to confirm all original input items received a label
- You are about to aggregate per-user statistics and need to know the coverage is complete

## Don't Use When
- The task has no per-item classification step (e.g., single-pass summarization)
- You already verified coverage by comparing against the original input in the current iteration

## Steps
1. Store the original input lines in a variable before splitting into chunks: `original_lines = [ln for ln in context.split(chr(10)) if ln.strip()]`
2. After merging all child results into a single dictionary (e.g., `all_labels` mapping line text to label), count: `len(original_lines)` vs `len(all_labels)`
3. If counts differ, find the missing items: `missing = [ln for ln in original_lines if ln not in all_labels]` and print them with their indices
4. Identify which chunk each missing item belongs to by checking the original chunk boundaries, then re-run only those chunks with a corrected prompt
5. After re-running, merge the new results into `all_labels` and repeat the count check until `len(all_labels) == len(original_lines)`

## Pitfalls
- Comparing by line text fails if lines are not unique; use a line index or hash instead: `all_labels[i] = label` where i is the line index within the chunk
- If child results return counts instead of per-item labels, verify the total count matches the chunk size before merging
- Do not skip this check even if child results look complete - partial coverage is invisible without explicit comparison

## Verify
- Run `assert len(all_labels) == len(original_lines), f'Missing {len(original_lines) - len(all_labels)} items'` in the REPL
- Print the first 3 missing items if the assertion fails, then re-run those chunks
````

## Round 9: S4

Source: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/opt/round_09/work/proposal_result.json`


**incumbent_behavior**: Root classified 188 lines into 6 categories via 6 child RLM calls, merged results, found 33 users with 'numeric value' or 'location', verified all 56 users accounted for, then spot-checked 5 random target users (all plausible). No check was performed on the 23 non-target users to verify they were correctly excluded.


**observed_failure**: User 48114 was classified as having 'numeric value' or 'location' (false positive), producing 32 extra pairs in the final answer. The spot-check only sampled from target users, so the misclassification of a non-target user was never detected. F1=0.969 due to 32 extra pairs (precision=0.939).


**behavioral_change**: After the existing spot-check of target-category users, the root will also sample 5-10 users from the excluded set (those NOT in the target categories) and print their questions alongside their assigned labels. If any excluded user's questions clearly belong to a target category, the root re-runs classification with a refined prompt. This catches false positives like user 48114 before pair generation.


**predicted_effect**: The root will sample from both target and excluded users after classification. In the held-in run, user 48114's questions (e.g., 'What does JESSICA mean?' classified as 'abbreviation' but also having 'numeric value' or 'location') would be caught during the excluded-user cross-check, triggering re-classification with a refined prompt before pair generation. This prevents false positive pairs from entering the final answer.


### Exact replacement

````text
After merging results from all chunks into a single data structure (e.g., a list of labels), verify that every input item has been assigned. Print the count of items with missing assignments. If any items are missing, identify which chunks they belong to and re-run only those chunks with a corrected prompt. Do not proceed to the final answer until all items have been assigned.

**Classification quality check**: After confirming full coverage, sample 5-10 items from each category (or from the smallest categories) and print the original question text alongside its assigned label. Review these samples: if any clearly belong to a different category, re-run the classification with a refined prompt that addresses the observed confusion. Do not proceed to the final answer until the classification passes this spot-check.

**Cross-check excluded users**: After identifying the set of users that satisfy the target condition (e.g., have 'numeric value' or 'location'), also sample 5-10 users from the excluded set and print their questions alongside their assigned labels. If any excluded user's questions clearly belong to a target category, re-run the classification with a refined prompt. Do not proceed to the final answer until the excluded set passes this cross-check.
````
