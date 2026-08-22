"""Tests for the mechanical trace digest in shrlm/optimization/digest.py.

The digest is the only view of a trace the attributor ever sees, so three
properties are load-bearing: it never leaks REPL locals, it announces its own
truncation, and it is a deterministic function of the tree and config -- the
sha256 is what makes the attribution cache meaningful.
"""

import hashlib
import json
from typing import Any

from shrlm.optimization.attribution import LLMAttributor
from shrlm.optimization.digest import (
    DIGEST_VERSION,
    DigestConfig,
    TraceDigest,
    build_digest,
    head_tail,
    render_child_table,
)
from shrlm.optimization.grounding import GroundingResult
from shrlm.optimization.mining import WeaknessMiner
from shrlm.optimization.taxonomy import AgentMechanism, EditableSurface, VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict
from shrlm.optimization.walker import compute_tree_stats, walk
from tests.mock_lm import MockLM
from tests.optimization.fixtures import (
    NESTED_CHILD_PROMPT,
    PROCEDURE_CODE,
    ROOT_MODEL,
    SECRET_LOCAL,
    SKILL_INDEX,
    as_completion,
    code_block,
    completion_dict,
    fallback_run,
    iteration_entry,
    make_verdict,
    nested_run,
    run_metadata,
    shallow_run,
    skilled_run,
    usage,
)


def digest_of_nested_run(cfg: DigestConfig | None = None) -> TraceDigest:
    root, stats = walk(as_completion(nested_run()))
    return build_digest(
        instance_id="run-nested",
        question="which letter does the middle slice name?",
        root=root,
        stats=stats,
        verdict=make_verdict(),
        cfg=cfg,
    )


def call_node(node_id: str, prompt: str, response: str, depth: int = 1) -> CallNode:
    return CallNode(
        node_id=node_id,
        parent_id="r" if depth else None,
        kind=NodeKind.LLM_LEAF if depth else NodeKind.ROOT,
        depth=depth,
        model="mock-model",
        prompt=prompt,
        response=response,
        prompt_chars=len(prompt),
        response_chars=len(response),
        execution_time=0.1,
    )


def bulky_tree() -> CallNode:
    """A root with two large sub-calls, far bigger than a small char budget."""
    root = call_node("r", prompt="Q" * 500, response="A", depth=0)
    root.kind = NodeKind.ROOT
    root.children = [
        call_node("r/i0/b0/c0", prompt="x" * 8000, response="y" * 4000),
        call_node("r/i0/b0/c1", prompt="p" * 3000, response="q" * 2000),
    ]
    return root


class TestNoLocalsLeak:
    def test_repl_locals_never_appear_in_the_digest(self):
        # The fixture plants a distinctive value in the REPL locals of two
        # runs; the walker drops locals and the digest must not resurrect them.
        assert SECRET_LOCAL in str(nested_run())  # the fixture really carries it
        assert SECRET_LOCAL not in digest_of_nested_run().text


class TestDeterminism:
    def test_same_input_produces_byte_identical_text_and_sha(self):
        first = digest_of_nested_run()
        second = digest_of_nested_run()
        assert first.text == second.text
        assert first.sha256 == second.sha256

    def test_sha_is_the_sha256_of_the_text(self):
        digest = digest_of_nested_run()
        assert digest.sha256 == hashlib.sha256(digest.text.encode("utf-8")).hexdigest()


class TestTruncation:
    def test_head_tail_returns_short_text_verbatim(self):
        assert head_tail("short", 100) == "short"

    def test_head_tail_announces_what_it_dropped(self):
        text = "a" * 300
        rendered = head_tail(text, 100)
        assert "[200 chars omitted]" in rendered
        assert len(rendered) < len(text)

    def test_forced_truncation_is_announced_in_the_digest_text(self):
        cramped = digest_of_nested_run(DigestConfig(char_budget=600))
        assert "chars omitted]" in cramped.text

    def test_generous_budget_shows_focused_excerpts_in_full(self):
        # The sub-call table is bounded by construction at PREVIEW_CHARS per
        # preview, so the 300-char child prompt is always elided there; the
        # focused excerpts own half the budget and carry it whole when room
        # allows.
        roomy = digest_of_nested_run(DigestConfig(char_budget=200_000))
        assert NESTED_CHILD_PROMPT in roomy.text


class TestCoverage:
    def test_coverage_is_the_surviving_fraction_of_available_chars(self):
        root = bulky_tree()
        stats = compute_tree_stats(root)
        digest = build_digest(
            instance_id="run-bulky",
            question="q",
            root=root,
            stats=stats,
            verdict=make_verdict(),
            cfg=DigestConfig(char_budget=1000),
        )
        assert digest.chars_available > 0
        assert digest.chars_kept <= digest.chars_available
        assert digest.coverage == digest.chars_kept / digest.chars_available
        assert 0.0 < digest.coverage < 1.0

    def test_coverage_never_exceeds_one(self):
        assert digest_of_nested_run().coverage <= 1.0


class TestRenderedSections:
    def test_header_carries_the_verifier_outcome(self):
        digest = digest_of_nested_run()
        assert "instance_id: run-nested" in digest.text
        assert "verifier_cause: wrong_value" in digest.text
        assert "collapse_ratio: 0.750" in digest.text

    def test_wide_tree_degrades_to_a_per_depth_aggregate(self):
        root = call_node("r", prompt="Q" * 100, response="A", depth=0)
        root.kind = NodeKind.ROOT
        root.children = [
            call_node(f"r/i0/b0/c{i}", prompt=f"piece {i}", response=f"answer {i}")
            for i in range(41)
        ]
        stats = compute_tree_stats(root)
        digest = build_digest(
            instance_id="run-wide",
            question="q",
            root=root,
            stats=stats,
            verdict=make_verdict(),
        )
        assert "aggregated by depth" in digest.text


def wide_tree(sub_verdicts: list[bool | None] | None = None, n_children: int = 41) -> CallNode:
    """A tree wide enough to force the per-depth aggregate table.

    ``sub_verdicts`` assigns verdicts positionally to the children; None (the
    default) leaves every child unverdicted, which is exactly the shape an
    ablated (no sub-verifier) mining pass produces.
    """
    root = call_node("r", prompt="Q" * 100, response="A", depth=0)
    root.kind = NodeKind.ROOT
    root.children = [
        call_node(f"r/i0/b0/c{i}", prompt=f"piece {i}", response=f"answer {i}")
        for i in range(n_children)
    ]
    if sub_verdicts is not None:
        for node, verdict in zip(root.children, sub_verdicts, strict=True):
            node.sub_verdict = verdict
    return root


def digest_of_tree(root: CallNode, instance_id: str = "run-wide") -> TraceDigest:
    return build_digest(
        instance_id=instance_id,
        question="q",
        root=root,
        stats=compute_tree_stats(root),
        verdict=make_verdict(),
    )


class TestAggregateVerdictHonesty:
    """The per-depth aggregate must not fabricate a sub-verifier statistic.

    An ablated round computes no verdicts, so its aggregate line must say so
    (``n/a``) rather than assert a zero count indistinguishable from a
    sub-verifier that ran and passed everything.
    """

    def test_ablated_wide_tree_renders_na_never_zero(self):
        digest = digest_of_tree(wide_tree())
        assert "sub_verifier_failed=n/a" in digest.text
        assert "sub_verifier_failed=0" not in digest.text

    def test_all_passing_verdicts_render_zero_not_na(self):
        digest = digest_of_tree(wide_tree(sub_verdicts=[True] * 41))
        assert "sub_verifier_failed=0" in digest.text
        assert "sub_verifier_failed=n/a" not in digest.text

    def test_mixed_verdicts_render_the_failure_count(self):
        # Two failures among a mix of passes and uncheckable Nones: any
        # verdict set at a depth keeps that depth's count numeric (per-depth
        # rule), and only explicit False verdicts are counted as failures.
        verdicts: list[bool | None] = [False, False, None] + [True] * 38
        digest = digest_of_tree(wide_tree(sub_verdicts=verdicts))
        assert "sub_verifier_failed=2" in digest.text
        assert "sub_verifier_failed=n/a" not in digest.text

    def test_mixed_depths_render_na_only_for_the_all_none_depth(self):
        # The rule is per-depth, not tree-level: depth 1 carries verdicts
        # (one explicit failure), so its line counts numerically, while the
        # depth-2 grandchildren are all sub_verdict None -- no sub-verifier
        # statistic was ever computed there, so that line must say n/a
        # rather than borrow depth 1's "verdicts exist" and fabricate a 0.
        root = wide_tree(sub_verdicts=[False] + [True] * 40)
        root.children[0].children = [
            call_node(f"r/i0/b0/c0/g{i}", prompt=f"deep {i}", response=f"deep answer {i}", depth=2)
            for i in range(3)
        ]
        digest = digest_of_tree(root)
        depth_lines = {
            line.split(":")[0]: line
            for line in digest.text.splitlines()
            if line.startswith("depth ")
        }
        assert "sub_verifier_failed=1" in depth_lines["depth 1"]
        assert "sub_verifier_failed=n/a" in depth_lines["depth 2"]

    def test_ablated_and_all_passing_digests_have_different_shas(self):
        # The n/a rendering is what separates "never ran" from "ran and all
        # passed" in bytes -- and different bytes mean different attribution
        # cache keys, which is the cache-invalidation mechanism for affected
        # records.
        ablated = digest_of_tree(wide_tree())
        all_passing = digest_of_tree(wide_tree(sub_verdicts=[True] * 41))
        assert ablated.text != all_passing.text
        assert ablated.sha256 != all_passing.sha256

    def test_grounded_narrow_digest_is_untouched_by_the_aggregate_rule(self):
        # A verdict-bearing narrow tree stays on the per-call table: no n/a
        # anywhere in the digest, and the table section is byte-identical to
        # what the table renderer produces -- grounded digest bytes (and hence
        # cached attributions) survive this change.
        root = wide_tree(sub_verdicts=[True, False], n_children=2)
        digest = digest_of_tree(root, instance_id="run-narrow")
        table_text, _, aggregated = render_child_table(root, DigestConfig())
        assert not aggregated
        assert table_text in digest.text
        assert "n/a" not in digest.text


# Digest sha256 of each pre-S10 fixture as ``digest_of_fixture`` renders it,
# computed at DIGEST_VERSION 1.1.0 (commit 3803be8) before the skills lines
# existed. A trace whose run_metadata carries no skill index -- every pre-S10
# trace, and every trace under an empty S10 -- must still render to exactly
# these bytes, so pre-S10 bundles and attribution caches (keyed on digest
# bytes, not on DIGEST_VERSION) are untouched by the format change.
PRE_S10_DIGEST_SHA256 = {
    "nested": "37edddf95a35e910d47724eb1361e2c040cdce8ed11322e3876328736c1abd50",
    "shallow": "58c16eca024a23a886cb0d56d196a8f4bd04436c8a6ebd775279db330b65cc64",
    "fallback": "e2d1b9f7a3b10652858cbe6e33bbedcf03edd3c1bc168727917a0459f2ca41cf",
}


def digest_of_fixture(name: str, run: dict[str, Any]) -> TraceDigest:
    root, stats = walk(as_completion(run))
    return build_digest(
        instance_id=f"run-{name}",
        question="q",
        root=root,
        stats=stats,
        verdict=make_verdict(),
    )


def skilled_child_run() -> dict[str, Any]:
    """A root that delegated to an ``rlm_query`` child which loaded a skill itself.

    The child's own environment recorded the load at its depth (1), and the
    child's run_metadata carries the same index the root's does, because the
    loader is installed in both REPL namespaces (KTD7).
    """
    child_block = code_block(
        code="proc = load_skill('check_slice_coverage')\nprint('covered')",
        stdout="covered\n",
        skill_loads=[{"skill": "check_slice_coverage", "depth": 1}],
    )
    child = {
        "root_model": ROOT_MODEL,
        "prompt": "check the slices are all covered",
        "response": "covered",
        "usage_summary": usage(),
        "execution_time": 0.3,
        "metadata": {
            "run_metadata": run_metadata(max_depth=2, skill_index=SKILL_INDEX),
            "iterations": [
                iteration_entry(
                    index=1,
                    response="Loading the procedure.\n```repl\n...\n```",
                    code_blocks=[child_block],
                    final_answer="covered",
                )
            ],
        },
    }
    root_block = code_block(
        code="verdict = rlm_query('check the slices are all covered')\nprint(verdict)",
        stdout="covered\n",
        rlm_calls=[child],
    )
    return completion_dict(
        prompt="what is the total across all slices?",
        response="41",
        iterations=[
            iteration_entry(
                index=1,
                response="Delegating the check.\n```repl\n...\n```",
                code_blocks=[root_block],
                final_answer="41",
            )
        ],
        max_depth=2,
        skill_index=SKILL_INDEX,
    )


class TestSkillLines:
    """The available_skills / loaded_skills pair: the S10 mechanism's observable.

    Rendered only when the trace's run-start record carries a skill index --
    i.e. a loader was installed, i.e. S10 was non-empty -- so the attributor
    can tell "no skill covered this" from "a skill covered it and was never
    consulted", while an empty-S10 trace renders exactly as before.
    """

    def test_empty_s10_trace_renders_byte_identically_to_the_pre_s10_digest(self):
        for name, run in [
            ("nested", nested_run()),
            ("shallow", shallow_run()),
            ("fallback", fallback_run()),
        ]:
            digest = digest_of_fixture(name, run)
            assert "available_skills:" not in digest.text
            assert "loaded_skills:" not in digest.text
            assert digest.sha256 == PRE_S10_DIGEST_SHA256[name], name

    def test_one_load_renders_the_loaded_skills_line_naming_it(self):
        run = skilled_run([{"skill": "merge_slice_totals", "depth": 0}], skill_index=SKILL_INDEX)
        text = digest_of_fixture("skilled", run).text
        assert "available_skills: merge_slice_totals, check_slice_coverage" in text
        assert "loaded_skills: merge_slice_totals (depth 0)" in text

    def test_no_load_under_a_non_empty_index_says_none_rather_than_nothing(self):
        text = digest_of_fixture("skilled", skilled_run([], skill_index=SKILL_INDEX)).text
        assert "available_skills: merge_slice_totals, check_slice_coverage" in text
        assert "loaded_skills: (none)" in text

    def test_the_pair_sits_in_the_run_header_before_the_tree_statistics(self):
        text = digest_of_fixture("skilled", skilled_run([], skill_index=SKILL_INDEX)).text
        header = text.split("## Tree statistics")[0]
        assert "available_skills:" in header and "loaded_skills:" in header

    def test_a_child_load_is_listed_at_its_own_depth(self):
        text = digest_of_fixture("child", skilled_child_run()).text
        assert "loaded_skills: check_slice_coverage (depth 1)" in text

    def test_repeated_loads_of_one_skill_at_one_depth_are_listed_once(self):
        loads = [{"skill": "merge_slice_totals", "depth": 0}] * 3
        text = digest_of_fixture("skilled", skilled_run(loads, skill_index=SKILL_INDEX)).text
        assert "loaded_skills: merge_slice_totals (depth 0)\n" in text

    def test_loaded_and_unloaded_digests_differ_in_bytes_and_sha(self):
        # Different bytes mean different attribution cache keys: the loader
        # event is exactly what separates the two records for the attributor.
        unloaded = digest_of_fixture("skilled", skilled_run([], skill_index=SKILL_INDEX))
        loaded = digest_of_fixture(
            "skilled",
            skilled_run([{"skill": "merge_slice_totals", "depth": 0}], skill_index=SKILL_INDEX),
        )
        assert unloaded.text != loaded.text
        assert unloaded.sha256 != loaded.sha256


UNGROUNDED = GroundingResult(failing_level=None, grounded=False, verdicts={})


def _skill_names(digest_text: str, prefix: str) -> list[str] | None:
    """The names on the ``available_skills:`` / ``loaded_skills:`` line, or None if absent."""
    for line in digest_text.splitlines():
        if line.startswith(prefix):
            rest = line[len(prefix) :].strip()
            if rest == "(none)":
                return []
            return [item.split(" (")[0] for item in rest.split(", ")]
    return None


def scripted_mechanism(digest_text: str) -> AgentMechanism:
    """The ``unconsulted_procedure`` rule of MECHANISM_DOCS, scripted over digest text.

    No code-level precedence exists (the rule is prompt text), so this stands
    in for an attributor that follows it literally: the S6 terminal signal is
    claimed first; then, under a non-empty index, an available skill that was
    never loaded; then, with no index, the root re-running a procedure it had
    already carried out; otherwise nothing S10 can address.
    """
    if "terminated_by_fallback: True" in digest_text:
        return AgentMechanism.ITERATION_BUDGET_EXHAUSTION
    available = _skill_names(digest_text, "available_skills: ")
    loaded = _skill_names(digest_text, "loaded_skills: ") or []
    if available is not None:
        if any(name not in loaded for name in available):
            return AgentMechanism.UNCONSULTED_PROCEDURE
        return AgentMechanism.OTHER
    if digest_text.count(PROCEDURE_CODE) >= 2:
        return AgentMechanism.UNCONSULTED_PROCEDURE
    return AgentMechanism.OTHER


def scripted_response(messages: Any) -> str:
    digest_text = messages[-1]["content"]
    payload = {
        "causal_status": "causal",
        "agent_mechanism": scripted_mechanism(digest_text).value,
        "failing_level": "no_recursion",
        "evidence_node_ids": ["r"],
        "symptom_summary": "scripted attributor following the documented rule",
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def attributed_surface(run: dict[str, Any]) -> EditableSurface | None:
    """Digest ``run``, attribute it with the scripted attributor, resolve the surface."""
    root, stats = walk(as_completion(run))
    digest = build_digest(
        instance_id="run-skilled",
        question="what is the total across all slices?",
        root=root,
        stats=stats,
        verdict=make_verdict(),
    )
    attributor = LLMAttributor(MockLM(response_fn=scripted_response))
    result = attributor.attribute(digest, root, make_verdict(), UNGROUNDED)
    return result.signature.surface()


class TestSkillsMechanismResolution:
    """The S10 mechanism, end to end through the real attributor and validator.

    A scripted attributor applies the documented rule to the rendered digest;
    the test is that the digest carries enough for the rule to be applied, and
    that the resulting signature resolves to S10 exactly when it should.
    """

    def test_available_and_never_loaded_resolves_to_s10(self):
        run = skilled_run([], skill_index=SKILL_INDEX[:1])
        assert attributed_surface(run) is EditableSurface.SKILLS

    def test_the_same_run_with_the_skill_loaded_does_not(self):
        run = skilled_run(
            [{"skill": "merge_slice_totals", "depth": 0}], skill_index=SKILL_INDEX[:1]
        )
        assert attributed_surface(run) is not EditableSurface.SKILLS

    def test_budget_exhaustion_independently_explains_the_failure_so_s6_wins(self):
        # Same unloaded skill, but the run ended by fallback: the conservative
        # precedence rule hands the terminal signal to S6, not S10.
        run = skilled_run([], skill_index=SKILL_INDEX[:1], fallback=True)
        assert attributed_surface(run) is EditableSurface.RUNTIME_POLICY

    def test_removing_the_budget_signal_lets_s10_claim_the_same_record(self):
        exhausted = skilled_run([], skill_index=SKILL_INDEX[:1], fallback=True)
        committed = skilled_run([], skill_index=SKILL_INDEX[:1], fallback=False)
        assert attributed_surface(exhausted) is EditableSurface.RUNTIME_POLICY
        assert attributed_surface(committed) is EditableSurface.SKILLS

    def test_no_index_and_a_repeated_procedure_resolves_to_s10_via_the_fallback(self):
        run = skilled_run(repeat_procedure=True)
        assert attributed_surface(run) is EditableSurface.SKILLS

    def test_no_index_and_no_repetition_is_not_s10(self):
        assert attributed_surface(skilled_run()) is not EditableSurface.SKILLS


class _FailingVerifier:
    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        return Verdict(passed=False, cause=VerifierCause.WRONG_VALUE, gold="42", produced=produced)


class TestDigestVersion:
    def test_version_bumped_for_the_skill_lines(self):
        # 1.1.0 was the n/a aggregate rendering; 1.2.0 adds the
        # available_skills / loaded_skills pair under a non-empty index.
        assert DIGEST_VERSION == "1.2.0"

    def test_digest_version_is_recorded_per_bundle(self):
        lm = MockLM(response_fn=scripted_response)
        miner = WeaknessMiner(verifier=_FailingVerifier(), attributor=LLMAttributor(lm))
        result = miner.mine(
            [({"id": "inst-1", "question": "q"}, as_completion(shallow_run()))],
            round_index=1,
            harness_version="H0",
            split_id="held_in_v1",
        )
        assert result.bundle.config.digest_version == DIGEST_VERSION == "1.2.0"
        assert result.bundle.to_dict()["config"]["digest_version"] == "1.2.0"

    def test_attribution_cache_key_does_not_include_digest_version(self):
        # DIGEST_VERSION reaches bundle ids via MiningConfig.digest_version
        # only; it is deliberately NOT part of the attribution cache key.
        # Asserted on the hashed material itself (not on a monkeypatched
        # module attribute, which a from-import would dodge): config_sha256
        # hashes exactly this key set, and no digest_version is among them,
        # so no bump -- however imported -- can reach the key. Invalidation
        # rides on digest bytes instead.
        attributor = LLMAttributor(MockLM())
        material = attributor.config_material()
        assert set(material) == {
            "model",
            "sampling_args",
            "max_attempts",
            "prompt_version",
            "taxonomy_version",
            "validator_version",
        }
        # And the material really is what the sha is computed over.
        payload = json.dumps(material, sort_keys=True, default=str)
        assert attributor.config_sha256() == hashlib.sha256(payload.encode("utf-8")).hexdigest()
