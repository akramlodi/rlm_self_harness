#!/bin/bash
# Show what a candidate harness proposal actually changed, relative to the
# incumbent it was proposed against.
#
# A proposal.json carries a *full* shrlm-harness/v2 envelope, not a delta (see
# shrlm/docs/harness-proposal-interface.md). The incumbent is referenced only by
# `base_harness_hash`, so the original and the candidate never appear in the same
# file. This script pairs them up and diffs them.
#
# Usage:
#   scripts/diff_proposal.sh <proposals_dir>/<candidate_id>
#   scripts/diff_proposal.sh <round_dir> <candidate_id>
#
#   scripts/diff_proposal.sh experiment_kimi/opt/round_02/proposals/r02-c02-s2
#   scripts/diff_proposal.sh experiment_kimi/opt/round_02 r02-c02-s2
#
# Options:
#   --summary        Only print the surface table, the declared-vs-actual surface
#                    check, and the performance block; skip the unified diff.
#   --all-surfaces   List every surface, changed or not. The default prints only
#                    the surfaces that actually differ -- a full harness carries
#                    a dozen-plus surfaces and an edit touches one, so the
#                    unchanged rows are noise that hides the signal.
#   --no-perf        Skip the held-in/held-out performance block.
#   --raw            Diff the envelopes verbatim, including `name` and `hash`,
#                    which are expected to differ, and without eliding long
#                    context lines.
#
# The performance block pairs the edit with what it did: candidate vs baseline
# pass rate on both splits, the pass-count deltas the promotion rule actually
# reads, and the cost ratio against the configured cost band. Sourced from each
# subject's summary.json and, when the round has closed, the promotions.jsonl
# record carrying the decision and its stated reasons.
#
# Sources, in preference order:
#   incumbent  <round_dir>/mining/*/harness.json
#              <round_dir>/validation/*/baseline/heldin/round_00/harness.json
#   candidate  <round_dir>/validation/*/<id>/heldin/round_00/harness.json
#              <round_dir>/proposals/<id>/proposal.json  (the .harness field)
#
# The validation copy is what the evaluation driver actually ran; the proposal
# copy is what was proposed. They should be identical -- the loader's round_trip
# gate enforces it -- so preferring the validation copy costs nothing and shows
# the real subject when one exists.

set -euo pipefail

SUMMARY_ONLY=0
RAW=0
ALL_SURFACES=0
NO_PERF=0
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --summary)      SUMMARY_ONLY=1 ;;
        --raw)          RAW=1 ;;
        --all-surfaces) ALL_SURFACES=1 ;;
        --no-perf)      NO_PERF=1 ;;
        -h|--help) sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)        echo "diff_proposal: unknown option: $arg" >&2; exit 2 ;;
        *)         ARGS+=("$arg") ;;
    esac
done

case "${#ARGS[@]}" in
    1)
        PROPOSAL_DIR="${ARGS[0]%/}"
        CANDIDATE_ID="$(basename "$PROPOSAL_DIR")"
        ROUND_DIR="$(cd "$PROPOSAL_DIR/../.." && pwd)"
        ;;
    2)
        ROUND_DIR="$(cd "${ARGS[0]}" && pwd)"
        CANDIDATE_ID="${ARGS[1]}"
        PROPOSAL_DIR="$ROUND_DIR/proposals/$CANDIDATE_ID"
        ;;
    *)
        echo "usage: diff_proposal.sh <proposals_dir>/<candidate_id> [options]" >&2
        echo "       diff_proposal.sh <round_dir> <candidate_id> [options]" >&2
        echo "options: --summary --all-surfaces --no-perf --raw" >&2
        exit 2
        ;;
esac

first_match() {
    local path
    for path in "$@"; do
        if [ -f "$path" ]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

INCUMBENT_SRC="$(first_match "$ROUND_DIR"/mining/*/harness.json \
                             "$ROUND_DIR"/validation/*/baseline/heldin/round_00/harness.json \
                             "$ROUND_DIR"/validation/*/baseline/heldout/round_00/harness.json || true)"
CANDIDATE_SRC="$(first_match "$ROUND_DIR"/validation/*/"$CANDIDATE_ID"/heldin/round_00/harness.json \
                             "$ROUND_DIR"/validation/*/"$CANDIDATE_ID"/heldout/round_00/harness.json \
                             "$PROPOSAL_DIR"/proposal.json || true)"

if [ -z "$INCUMBENT_SRC" ]; then
    echo "diff_proposal: no incumbent harness.json under $ROUND_DIR/{mining,validation/*/baseline}" >&2
    exit 1
fi
if [ -z "$CANDIDATE_SRC" ]; then
    echo "diff_proposal: no candidate harness for '$CANDIDATE_ID' under $ROUND_DIR" >&2
    exit 1
fi

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

# Normalize both sides to the same shape: an envelope with sorted keys. A
# proposal.json wraps its envelope in `.harness`; a harness.json is one already.
#
# Unless --raw was asked for, drop `name` and `hash` first. `name` is
# informational and excluded from the hash; `hash` differs by construction
# whenever any surface does. Neither is the edit, and removing them before the
# diff -- rather than filtering the diff afterwards -- keeps the output free of
# hunks that carry no changed line.
extract() {
    python3 - "$1" "$2" "$RAW" <<'EXTRACT'
import json, sys

src, dest, raw = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
with open(src) as fh:
    doc = json.load(fh)
if doc.get("format", "").startswith("shrlm-proposal/"):
    doc = doc["harness"]
if not raw:
    for scope in (doc, doc.get("harness", {})):
        scope.pop("name", None)
        scope.pop("hash", None)
with open(dest, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
EXTRACT
}

extract "$INCUMBENT_SRC" "$TMPDIR_RUN/incumbent.json"
extract "$CANDIDATE_SRC" "$TMPDIR_RUN/candidate.json"

echo "candidate : $CANDIDATE_ID"
echo "incumbent : ${INCUMBENT_SRC#"$ROUND_DIR"/}"
echo "candidate : ${CANDIDATE_SRC#"$ROUND_DIR"/}"
echo

# Per-surface table, plus the declared-vs-actual surface check that the loader's
# surface_diff gate performs.
python3 - "$TMPDIR_RUN/incumbent.json" "$TMPDIR_RUN/candidate.json" "$PROPOSAL_DIR/proposal.json" "$ALL_SURFACES" <<'SURFACES'
import json, os, sys

inc = json.load(open(sys.argv[1]))["harness"]["surfaces"]
cand = json.load(open(sys.argv[2]))["harness"]["surfaces"]

show_all = sys.argv[4] == "1"

changed, unchanged = [], []
for key in sorted(set(inc) | set(cand)):
    (unchanged if inc.get(key) == cand.get(key) else changed).append(key)

# Only the changed surfaces by default: an edit touches one surface out of a
# dozen-plus, so listing the rest buries the one line that matters.
for key in changed:
    print(f"  DIFF  {key}")
if show_all:
    for key in unchanged:
        print(f"  same  {key}")
elif unchanged:
    print(f"  ({len(unchanged)} unchanged, --all-surfaces to list)")
if not changed:
    print("  (no surface differs)")

# S8 serializes as two keys but counts as one surface; S10 as S10_skills.
def surface_of(key):
    return "S8" if key.startswith("S8_") else key.split("_", 1)[0]

actual = sorted({surface_of(k) for k in changed})
print()
print("surfaces changed :", ", ".join(actual) if actual else "(none)")

path = sys.argv[3]
if os.path.exists(path):
    proposal = json.load(open(path))
    declared = proposal.get("surface")
    print("surface declared :", declared)
    if not actual:
        print("CHECK            : FAIL - modifies no surface")
    elif len(actual) > 1:
        print("CHECK            : FAIL - modifies", len(actual), "surfaces:", ", ".join(actual))
    elif actual[0] != declared:
        print(f"CHECK            : FAIL - declared {declared}, edits {actual[0]}")
    else:
        print("CHECK            : ok - exactly one surface, and it is the declared one")
    print("predicted effect :", proposal.get("predicted_effect", ""))
    signature = proposal.get("target_signature", {})
    if signature:
        print("target signature :", json.dumps(signature, sort_keys=True))
SURFACES

# Performance: what the edit actually did on each split. The surface table says
# what changed; this says whether it helped. Both come from what the driver ran,
# never from the proposal's own predicted_effect.
if [ "$NO_PERF" -eq 0 ]; then
python3 - "$ROUND_DIR" "$CANDIDATE_ID" <<'PERF'
import json, os, glob, sys

round_dir, cand_id = sys.argv[1], sys.argv[2]

vdirs = sorted(glob.glob(os.path.join(round_dir, "validation", "*")))
vdir = next((d for d in vdirs if os.path.isdir(d)), None)
if vdir is None:
    print("\n--- performance ---\n  (no validation directory yet)")
    raise SystemExit(0)

def summary(subject):
    path = os.path.join(vdir, subject, "summary.json")
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None

base, cand = summary("baseline"), summary(cand_id)
print("\n--- performance (candidate vs baseline) ---")
if cand is None:
    print(f"  (no summary.json for {cand_id}; still running or never completed)")
    raise SystemExit(0)
if base is None:
    print("  (no baseline summary.json; cannot compare)")
    raise SystemExit(0)

for tag, doc in (("baseline", base), (cand_id, cand)):
    if doc.get("outcome") != "completed":
        print(f"  NOTE {tag} outcome={doc.get('outcome')!r} -- figures below are partial")

print(f"  {'split':<9}{'inst':>5}{'runs':>6}{'baseline':>11}{'candidate':>11}{'delta':>9}{'pass-count':>14}")
for split in ("heldin", "heldout"):
    b = base.get("splits", {}).get(split)
    c = cand.get("splits", {}).get(split)
    if not b or not c:
        print(f"  {split:<9}(missing)")
        continue
    br, cr = b.get("pass_rate"), c.get("pass_rate")
    # The promotion rule reads pass COUNTS, not rates -- show both so a delta
    # that looks tiny as a rate is still legible as the integer the gate saw.
    bpass = b.get("pass_count", 0)
    cpass = c.get("pass_count", 0)
    counts = "%d/%d (%+d)" % (cpass, bpass, cpass - bpass)
    print(f"  {split:<9}{c.get('n_instances',0):>5}{c.get('n_runs',0):>6}"
          f"{br:>11.4f}{cr:>11.4f}{cr-br:>+9.4f}{counts:>14}")

    term = c.get("n_resource_terminated", 0)
    if term:
        print(f"  {'':<9}{split} resource-terminated runs: {term} of {c.get('n_runs',0)}")

bc, cc = base.get("spent"), cand.get("spent")
if bc and cc:
    print(f"  spend    baseline ${bc:.2f}   candidate ${cc:.2f}")

# The promotions ledger is the authority on the outcome: it records the bands,
# the taus, and the reasons the rule gave. Present only once the round closed.
ledger = os.path.join(vdir, "promotions.jsonl")
rec = None
if os.path.exists(ledger):
    for line in open(ledger):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("subject_id") == cand_id or r.get("harness_hash") == cand.get("harness_hash"):
            rec = r
if rec is None:
    print("  decision : (round not closed yet)")
    raise SystemExit(0)

print(f"  decision : {rec.get('decision')}"
      f"   tau_regression={rec.get('tau_regression')} tau_improvement={rec.get('tau_improvement')}")
for metric, band in (rec.get("band") or {}).items():
    if band.get("baseline") in (None, 0) and band.get("candidate") in (None, 0):
        continue
    ratio = (band["candidate"] / band["baseline"]) if band.get("baseline") else float("nan")
    print(f"  band     {metric}: {ratio:.2f}x of baseline "
          f"(allowed {band.get('lower')}-{band.get('upper')}) "
          f"{'within' if band.get('within') else 'OUTSIDE'}")
for reason in rec.get("reasons") or []:
    print(f"  reason   {reason}")
PERF
fi

[ "$SUMMARY_ONLY" -eq 1 ] && exit 0

echo
echo "--- diff (incumbent -> candidate) ---"

# Changed lines only. Each surface serializes to one line, so a -/+ pair IS the
# edit -- surrounding context lines are neighbouring surfaces that did not
# change, and the ---/+++/@@ headers name temp files that no longer exist by the
# time anyone reads the output. --raw restores the full unified diff.
if [ "$RAW" -eq 1 ]; then
    diff -u "$TMPDIR_RUN/incumbent.json" "$TMPDIR_RUN/candidate.json" || true
else
    diff -U0 "$TMPDIR_RUN/incumbent.json" "$TMPDIR_RUN/candidate.json" \
        | grep -E '^[-+]' | grep -vE '^(---|\+\+\+)' || true
fi
