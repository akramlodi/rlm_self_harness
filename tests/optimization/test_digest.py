"""Tests for the mechanical trace digest in shrlm/optimization/digest.py.

The digest is the only view of a trace the attributor ever sees, so three
properties are load-bearing: it never leaks REPL locals, it announces its own
truncation, and it is a deterministic function of the tree and config -- the
sha256 is what makes the attribution cache meaningful.
"""

import hashlib
import json

from shrlm.optimization.attribution import LLMAttributor
from shrlm.optimization.digest import (
    DIGEST_VERSION,
    DigestConfig,
    TraceDigest,
    build_digest,
    head_tail,
    render_child_table,
)
from shrlm.optimization.types import CallNode, NodeKind
from shrlm.optimization.walker import compute_tree_stats, walk
from tests.mock_lm import MockLM
from tests.optimization.fixtures import (
    NESTED_CHILD_PROMPT,
    SECRET_LOCAL,
    as_completion,
    make_verdict,
    nested_run,
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


class TestDigestVersion:
    def test_version_bumped_for_the_na_rendering(self):
        assert DIGEST_VERSION == "1.1.0"

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
