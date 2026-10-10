# Grounded and coordinated proposals: plan review

Reviewed [the implementation plan](../plans/2026-10-04-1144-feat-grounded-coordinated-proposals-plan.md) using `ce-doc-review` in non-interactive mode. The planning request authorizes corrections needed to implement its established contract: preserve diagnosis detail, permit bounded capability/caller pairs, describe behavior patterns, and distinguish recovered defects. Regression-stage design remains excluded.

Document type: `unified-plan`. Source revision: `e323f4328e0aa37b2cb5ff67cabd4729d1b0916c`. Final plan SHA-256: `fa5972bb1f584bb4804e6cc292407bbf90592ca5d04559f17d20790deaa1bb62`.

## Coverage

| Lens | Scope | Execution |
| --- | --- | --- |
| Coherence | Consistent pair limits, requirements, dependencies, and verification counts | Completed inline |
| Feasibility | Evidence projection, member repair identity, candidate loading, format readers, existing paths | Completed inline |
| Scope | Reuse existing fields and batch validation; exclude regression-stage work and general dependency machinery | Completed inline |
| Adversarial | Missing partners, older readers, scripted-test limitations, and bounded paid calls | Completed inline |

All lenses were applied sequentially by the same agent, following the repository's instruction to run delegated work in the main thread. These are not independent reviewer endorsements. No cross-model job was started under that instruction. No Compound Packs were configured. No additional product, UI, or credential-handling review was needed for the settled scope.

## Original findings and resolved dispositions

### 1. Older readers could ignore pair requirements

**Evidence:** The original plan required a response-contract update but did not explicitly change the persisted proposal envelope. `shrlm/optimization/candidates.py` accepts the current format, validates known fields, and does not reject arbitrary additional metadata; `load_candidates` gates members independently. A v1 proposal with new pair metadata would remain acceptable to an older loader.

**Consequence:** One dependent member could be evaluated alone despite the pair-integrity requirement.

**Disposition:** Applied. New writes use a v2 proposal envelope, with legacy unpaired v1 read support. Added coverage for the old-format boundary and both history-reader formats. This fulfills the existing admission and persistence contract; it adds no evaluation stage.

### 2. Pair cardinality was ambiguous

**Evidence:** “One bounded activation pair” and “allow exactly one” could mean either two members per pair or one pair per round. The governing constraint was one edit per surface and the existing edit-slot limit.

**Consequence:** Implementers could add an unnecessary global restriction on otherwise disjoint interventions.

**Disposition:** Applied. State exactly two members per pair and permit distinct pairs when surface, pattern, and slot constraints allow them. No additional mechanism is introduced.

### 3. The named interface document is absent

**Evidence:** The code references `docs/harness-proposal-interface.md`, but filesystem and tracked-file checks found no such document in this checkout. The original unit listed it as an existing document to update.

**Consequence:** The unit misrepresented its documentation deliverable and could send implementation into an unnecessary search or restore.

**Disposition:** Applied. Explicitly declare a new focused interface document at the code-referenced path. No deleted experiment artifacts are restored.

### 4. Live comparison scope needed to match the eight-execution limit

**Evidence:** The plan specified two task families, four harness variants, target/protected behavior, and at most eight live executions without naming which family would run live.

**Consequence:** Applying the full matrix live would exceed the stated execution limit.

**Disposition:** Applied. Live coverage is one family × four variants × two inputs. Both families and renamed variants retain offline coverage. The cumulative $2 allowance and conservative retry/nested-call accounting still govern.

## Final result

Document review complete (non-interactive mode).

Applied 4 fixes. No proposed fixes, user decisions, or FYI findings remain. Product scope and priorities are preserved.

Checked the final plan's repository paths, source links, five unit IDs, and declared new files. All existing paths and source links resolve. No source/configuration changes, implementation tests, paid calls, or experiments were performed during planning.

Review complete.
