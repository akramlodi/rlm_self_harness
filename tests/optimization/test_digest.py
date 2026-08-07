"""Tests for the mechanical trace digest in shrlm/optimization/digest.py.

The digest is the only view of a trace the attributor ever sees, so three
properties are load-bearing: it never leaks REPL locals, it announces its own
truncation, and it is a deterministic function of the tree and config -- the
sha256 is what makes the attribution cache meaningful.
"""

import hashlib

from shrlm.optimization.digest import (
    DigestConfig,
    TraceDigest,
    build_digest,
    head_tail,
)
from shrlm.optimization.types import CallNode, NodeKind
from shrlm.optimization.walker import compute_tree_stats, walk
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
