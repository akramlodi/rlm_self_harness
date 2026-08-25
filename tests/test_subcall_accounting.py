"""Recursive sub-call spend must reach both the budget check and the reported usage.

A child RLM builds its own ``LMHandler`` and its own clients, so the parent's
handler aggregate never sees what the child spent. Two consequences, both
measured on live experiment data before these tests existed:

* the persisted per-run cost excluded child spend -- a median of 45% of a
  decomposing run's true cost, up to 96%, so the promotion cost band
  systematically under-measured exactly the harnesses that decompose;
* ``max_budget`` never counted child spend, because the budget check assigned
  the handler total over the sub-call increments ``_subcall`` had accumulated.

The tests below pin both halves, plus the reset that the old clobbering
assignment was silently providing: ``_cumulative_cost`` is per completion, and
a driver that reuses one RLM across a round's runs must not carry one run's
spend into the next.

Every class and type here is resolved through the ``live`` fixture rather than
imported at module scope. ``tests/test_imports.py::test_no_circular_imports``
deletes ``rlm.core.*`` from ``sys.modules`` and re-imports them, so a name bound
at collection time can end up pointing at a stale module: the parent RLM would
be the old class, whose ``_subcall`` reads the old module's globals, and a
monkeypatch applied to the live module would never reach it.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def live():
    """The ``rlm.core.rlm`` module object that is current for this test."""
    import rlm.core.rlm as module

    return module


def usage(
    live, model: str, *, calls: int = 1, inp: int = 0, out: int = 0, cost: float | None = None
):
    return live.UsageSummary(
        model_usage_summaries={
            model: live.ModelUsageSummary(
                total_calls=calls,
                total_input_tokens=inp,
                total_output_tokens=out,
                total_cost=cost,
            )
        }
    )


def iteration(live):
    return live.RLMIteration(prompt="p", response="r", code_blocks=[])


def handler_reporting(summary) -> Mock:
    handler = Mock()
    handler.get_usage_summary.return_value = summary
    return handler


def an_rlm(live, **kwargs):
    return live.RLM(backend="openai", backend_kwargs={"model_name": "m"}, **kwargs)


class TestUsageMerge:
    """Merging two summaries sums per model; it never clobbers one with the other."""

    def test_same_model_sums_rather_than_replacing(self, live):
        left = usage(live, "m", calls=2, inp=10, out=5, cost=0.01)
        right = usage(live, "m", calls=3, inp=20, out=7, cost=0.02)

        merged = left.merged_with(right)

        entry = merged.model_usage_summaries["m"]
        assert entry.total_calls == 5
        assert entry.total_input_tokens == 30
        assert entry.total_output_tokens == 12
        assert merged.total_cost == pytest.approx(0.03)

    def test_distinct_models_are_both_kept(self, live):
        merged = usage(live, "a", cost=0.01).merged_with(usage(live, "b", cost=0.02))

        assert set(merged.model_usage_summaries) == {"a", "b"}
        assert merged.total_cost == pytest.approx(0.03)

    def test_absent_cost_stays_absent(self, live):
        merged = usage(live, "m", calls=1).merged_with(usage(live, "m", calls=1))

        assert merged.model_usage_summaries["m"].total_calls == 2
        assert merged.total_cost is None

    def test_a_reported_cost_survives_merging_with_an_unpriced_call(self, live):
        merged = usage(live, "m", cost=0.01).merged_with(usage(live, "m"))

        assert merged.total_cost == pytest.approx(0.01)

    def test_synthesized_provenance_wins(self, live):
        priced = live.UsageSummary(
            model_usage_summaries={
                "m": live.ModelUsageSummary(1, 0, 0, total_cost=0.01, cost_source="provider")
            }
        )
        synthesized = live.UsageSummary(
            model_usage_summaries={
                "m": live.ModelUsageSummary(1, 0, 0, total_cost=0.02, cost_source="synthesized")
            }
        )

        assert priced.merged_with(synthesized).cost_source == "synthesized"

    def test_merging_leaves_both_operands_untouched(self, live):
        left = usage(live, "m", calls=1, cost=0.01)
        right = usage(live, "m", calls=1, cost=0.02)

        left.merged_with(right)

        assert left.model_usage_summaries["m"].total_calls == 1
        assert right.model_usage_summaries["m"].total_calls == 1


class TestBudgetCountsSubcallSpend:
    def test_child_spend_pushes_the_run_over_budget(self, live):
        """Handler alone is under budget; handler plus child spend is over it."""
        rlm = an_rlm(live, max_budget=0.10)
        rlm._subcall_usage = usage(live, "m", cost=0.08)
        handler = handler_reporting(usage(live, "m", cost=0.05))

        with pytest.raises(live.BudgetExceededError) as excinfo:
            rlm._check_iteration_limits(iteration(live), 0, handler)

        assert excinfo.value.spent == pytest.approx(0.13)

    def test_handler_spend_alone_still_trips_the_budget(self, live):
        rlm = an_rlm(live, max_budget=0.01)
        handler = handler_reporting(usage(live, "m", cost=0.05))

        with pytest.raises(live.BudgetExceededError) as excinfo:
            rlm._check_iteration_limits(iteration(live), 0, handler)

        assert excinfo.value.spent == pytest.approx(0.05)

    def test_within_budget_when_the_combined_total_fits(self, live):
        rlm = an_rlm(live, max_budget=0.10)
        rlm._subcall_usage = usage(live, "m", cost=0.02)
        handler = handler_reporting(usage(live, "m", cost=0.03))

        rlm._check_iteration_limits(iteration(live), 0, handler)

        assert rlm._cumulative_cost == pytest.approx(0.05)

    def test_repeated_checks_do_not_accumulate(self, live):
        """The budget is a snapshot of this completion, not a running sum of checks."""
        rlm = an_rlm(live, max_budget=0.10)
        rlm._subcall_usage = usage(live, "m", cost=0.02)
        handler = handler_reporting(usage(live, "m", cost=0.03))

        for index in range(5):
            rlm._check_iteration_limits(iteration(live), index, handler)

        assert rlm._cumulative_cost == pytest.approx(0.05)

    def test_child_tokens_count_against_the_token_limit(self, live):
        rlm = an_rlm(live, max_tokens=100)
        rlm._subcall_usage = usage(live, "m", inp=40, out=40)
        handler = handler_reporting(usage(live, "m", inp=30, out=30))

        with pytest.raises(live.TokenLimitExceededError) as excinfo:
            rlm._check_iteration_limits(iteration(live), 0, handler)

        assert excinfo.value.tokens_used == 140


class TestSubcallAccumulation:
    @staticmethod
    def _leaf(live, monkeypatch) -> Mock:
        leaf = Mock()
        leaf.model_name = "m"
        leaf.completion.return_value = "answer"
        leaf.get_last_usage.return_value = live.ModelUsageSummary(
            total_calls=1, total_input_tokens=10, total_output_tokens=5, total_cost=0.04
        )
        monkeypatch.setattr(live, "get_client", lambda *a, **k: leaf)
        return leaf

    def test_depth_capped_subcall_records_its_leaf_spend(self, live, monkeypatch):
        """The max-depth branch returns a plain LM completion; its spend still counts."""
        self._leaf(live, monkeypatch)

        parent = an_rlm(live, max_depth=1)
        result = parent._subcall("prompt")

        assert result.usage_summary.total_cost == pytest.approx(0.04)
        assert parent._subcall_usage.total_cost == pytest.approx(0.04)

    def test_accumulation_is_cumulative_across_subcalls(self, live, monkeypatch):
        self._leaf(live, monkeypatch)

        parent = an_rlm(live, max_depth=1)
        parent._subcall("one")
        parent._subcall("two")

        assert parent._subcall_usage.total_cost == pytest.approx(0.08)
        assert parent._subcall_usage.model_usage_summaries["m"].total_calls == 2


class TestChildBudgetTermination:
    def test_a_child_that_blew_its_budget_still_charges_the_parent(self, live, monkeypatch):
        """The child raised instead of returning, so only its exception carries the figure."""
        # Built before the patch: afterwards ``live.RLM`` is the child stub.
        parent = an_rlm(live, max_depth=3)

        class ExplodingChild:
            def __init__(self, *args, **kwargs):
                pass

            def completion(self, prompt, root_prompt=None):
                raise live.BudgetExceededError(spent=0.07, budget=0.05, message="child over budget")

            def close(self):
                pass

        monkeypatch.setattr(live, "RLM", ExplodingChild)

        result = parent._subcall("prompt")

        assert "budget" in result.response.lower()
        assert parent._subcall_usage.total_cost == pytest.approx(0.07)

    def test_a_child_killed_by_the_token_limit_still_charges_the_parent(self, live, monkeypatch):
        """Only the attached summary carries this; the exception has no spend figure."""
        parent = an_rlm(live, max_depth=3)
        spent = usage(live, "m", calls=3, inp=90, out=10, cost=0.06)

        class ThrottledChild:
            def __init__(self, *args, **kwargs):
                # What the child's completion context published before it died.
                self.last_completion_usage = spent

            def completion(self, prompt, root_prompt=None):
                raise live.TokenLimitExceededError(tokens_used=100, token_limit=50)

            def close(self):
                pass

        monkeypatch.setattr(live, "RLM", ThrottledChild)

        parent._subcall("prompt")

        assert parent._subcall_usage.total_cost == pytest.approx(0.06)
        assert parent._subcall_usage.total_input_tokens == 90

    def test_a_child_that_died_without_any_usage_charges_nothing(self, live, monkeypatch):
        """A plain crash carries neither an attached summary nor a spend figure."""
        parent = an_rlm(live, max_depth=3)

        class BrokenChild:
            def __init__(self, *args, **kwargs):
                pass

            last_completion_usage = None

            def completion(self, prompt, root_prompt=None):
                raise RuntimeError("child blew up")

            def close(self):
                pass

        monkeypatch.setattr(live, "RLM", BrokenChild)

        parent._subcall("prompt")

        assert parent._subcall_usage.model_usage_summaries == {}


class TestTerminatedChildUsage:
    """The completion context publishes its usage total on the way out."""

    def test_the_published_total_is_preferred_over_the_spend_figure(self, live):
        published = usage(live, "m", calls=2, inp=10, out=5, cost=0.09)
        child = SimpleNamespace(last_completion_usage=published)
        error = live.BudgetExceededError(spent=0.07, budget=0.05)

        recovered = live._terminated_child_usage(child, error, "m")

        assert recovered is published
        assert recovered.total_cost == pytest.approx(0.09)

    def test_the_spend_figure_is_the_fallback_when_nothing_was_published(self, live):
        child = SimpleNamespace(last_completion_usage=None)
        error = live.BudgetExceededError(spent=0.07, budget=0.05)

        recovered = live._terminated_child_usage(child, error, "m")

        assert recovered.total_cost == pytest.approx(0.07)
        assert recovered.model_usage_summaries["m"].total_calls == 0

    def test_a_child_with_neither_yields_nothing(self, live):
        child = SimpleNamespace(last_completion_usage=None)

        assert live._terminated_child_usage(child, RuntimeError("boom"), "m") is None


class TestPerCompletionReset:
    def test_reset_clears_both_accumulators(self, live):
        """One RLM is reused across a round's runs; spend must not carry over."""
        rlm = an_rlm(live, max_budget=0.10)
        rlm._subcall_usage = usage(live, "m", cost=0.09)
        rlm._cumulative_cost = 0.09

        rlm._reset_completion_accounting()

        assert rlm._cumulative_cost == 0.0
        assert rlm._subcall_usage.total_cost is None
        assert rlm._subcall_usage.model_usage_summaries == {}


class TestConcurrentSubcallAccounting:
    """``rlm_query_batched`` runs sub-calls on a thread pool, so the accumulator
    is written from several threads at once. An unguarded read-modify-write
    loses a child's spend silently -- the exact failure this accounting exists
    to prevent, and one that only appears under load.
    """

    def test_every_concurrent_child_survives_the_merge(self, live):
        from concurrent.futures import ThreadPoolExecutor

        parent = an_rlm(live, max_budget=1000.0)
        children = 64
        per_child = usage(live, "m", calls=1, inp=10, out=5, cost=0.01)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda _: parent._record_subcall_usage(per_child), range(children)))

        assert parent._subcall_usage.model_usage_summaries["m"].total_calls == children
        assert parent._subcall_usage.total_cost == pytest.approx(children * 0.01)

    def test_the_running_cost_tracks_every_child(self, live):
        """``_cumulative_cost`` is what a sibling's budget headroom is derived
        from, so it has to advance with each child, not once per iteration."""
        from concurrent.futures import ThreadPoolExecutor

        parent = an_rlm(live, max_budget=1000.0)
        per_child = usage(live, "m", calls=1, cost=0.02)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: parent._record_subcall_usage(per_child), range(50)))

        assert parent._cumulative_cost == pytest.approx(50 * 0.02)


class TestSubcallBudgetHeadroom:
    def test_a_second_child_in_one_iteration_sees_its_siblings_spend(self, live, monkeypatch):
        """Several children can be spawned inside a single iteration. Without a
        running total each would be handed the same headroom and the budget
        would only bite at the next iteration boundary -- far too late.
        """
        handed_out: list[float | None] = []

        class RecordingChild:
            last_completion_usage = None

            def __init__(self, *args, **kwargs):
                handed_out.append(kwargs.get("max_budget"))
                self.last_completion_usage = None

            def completion(self, prompt, root_prompt=None):
                return live.RLMChatCompletion(
                    root_model="m",
                    prompt=prompt,
                    response="done",
                    usage_summary=usage(live, "m", calls=1, cost=0.30),
                    execution_time=0.0,
                )

            def close(self):
                pass

        parent = an_rlm(live, max_depth=3, max_budget=1.00)
        monkeypatch.setattr(live, "RLM", RecordingChild)

        parent._subcall("one")
        parent._subcall("two")
        parent._subcall("three")

        assert handed_out == [pytest.approx(1.00), pytest.approx(0.70), pytest.approx(0.40)]

    def test_an_exhausted_budget_stops_further_children(self, live, monkeypatch):
        class GreedyChild:
            last_completion_usage = None

            def __init__(self, *args, **kwargs):
                self.last_completion_usage = None

            def completion(self, prompt, root_prompt=None):
                return live.RLMChatCompletion(
                    root_model="m",
                    prompt=prompt,
                    response="done",
                    usage_summary=usage(live, "m", calls=1, cost=0.60),
                    execution_time=0.0,
                )

            def close(self):
                pass

        parent = an_rlm(live, max_depth=3, max_budget=1.00)
        monkeypatch.setattr(live, "RLM", GreedyChild)

        parent._subcall("one")
        parent._subcall("two")
        third = parent._subcall("three")

        assert "budget exhausted" in third.response.lower()
