"""A run that hits a resource limit must report what it actually cost.

A terminated completion returns no ``RLMChatCompletion``, so its usage used to
die with the ``LMHandler`` the completion context held: the driver synthesized
an empty summary and salvaged a cost only from ``BudgetExceededError.spent``.
Every other termination persisted ``cost: null`` and was then priced at the
per-run ceiling.

Estimating the figure after the fact does not work. Limit checks run *before*
the terminating turn is priced and logged, so the trajectory is missing exactly
the turn the money is in -- on the live run ``bfs-84c18f2383c7c0c3__a04`` the
logged turns sum to 1.8% of what the run really cost.

The fix is a publication point, not an estimate: the completion context's
``finally`` hands over the handler's own total on the way out. That block runs
during exception propagation, after every completed call is recorded and before
the handler is stopped, which makes it the one place where "the run is over" is
true however it ended -- including a limit raised inside a client, which never
passes through the iteration checks at all.

Classes are resolved through the ``live`` fixture because
``tests/test_imports.py::test_no_circular_imports`` re-imports ``rlm.core.*``;
see ``tests/test_subcall_accounting.py`` for the full explanation.
"""

import pytest

from tests.mock_lm import MockLM


@pytest.fixture
def live():
    """The ``rlm.core.rlm`` module object that is current for this test."""
    import rlm.core.rlm as module

    return module


class RaisingLM(MockLM):
    """A client that raises on its ``nth`` call, after recording the earlier ones.

    Stands in for a limit the provider enforces inside the call -- a token cap
    rejected server-side, say. The runtime's between-iteration checks never see
    it, which is the path an exception-attribute scheme would miss. The failing
    call is counted before it raises, matching a client that issued the request
    and then had it rejected.
    """

    def __init__(self, error: Exception, *, raise_on: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._error = error
        self._raise_on = raise_on

    def completion(self, prompt):
        if self._call_count + 1 >= self._raise_on:
            self._call_count += 1
            raise self._error
        return super().completion(prompt)


class NeverCallsLM(MockLM):
    """A client that fails before the request is ever issued or counted.

    The genuinely-zero-call case: a connection refused before anything left the
    process. Its usage is empty rather than merely cheap, which is what the
    driver's ceiling pricing keys on.
    """

    def completion(self, prompt):
        raise RuntimeError("connection refused before the request was sent")


def an_rlm(live, **kwargs):
    return live.RLM(
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        environment="local",
        **kwargs,
    )


class TestPublicationOnTermination:
    def test_a_client_raised_limit_still_publishes_what_was_recorded(self, live, monkeypatch):
        """The path the iteration checks never see -- KTD1's reason for existing."""
        client = RaisingLM(live.TokenLimitExceededError(tokens_used=9, token_limit=1), raise_on=2)
        monkeypatch.setattr(live, "get_client", lambda *a, **k: client)

        rlm = an_rlm(live, max_iterations=5)
        with pytest.raises(live.TokenLimitExceededError):
            rlm.completion("prompt")

        published = rlm.last_completion_usage
        assert published is not None
        assert published.total_calls == 2
        assert published.total_input_tokens == 20

    def test_a_request_that_was_issued_then_rejected_still_counts(self, live, monkeypatch):
        """The client counted the call before raising, so the spend is real."""
        client = RaisingLM(RuntimeError("rejected server-side"), raise_on=1)
        monkeypatch.setattr(live, "get_client", lambda *a, **k: client)

        rlm = an_rlm(live, max_iterations=5)
        with pytest.raises(RuntimeError):
            rlm.completion("prompt")

        assert rlm.last_completion_usage.total_calls == 1

    def test_a_completion_that_made_no_calls_publishes_nothing_recorded(self, live, monkeypatch):
        """No calls means no evidence of spend -- the breaker must keep its ceiling.

        This is the branch ``driver._partial_completion`` keys its ceiling
        pricing on: a published summary recording zero calls is not a free run,
        it is a run nothing is known about.
        """
        monkeypatch.setattr(live, "get_client", lambda *a, **k: NeverCallsLM())

        rlm = an_rlm(live, max_iterations=5)
        with pytest.raises(RuntimeError):
            rlm.completion("prompt")

        assert rlm.last_completion_usage.total_calls == 0
        assert rlm.last_completion_usage.total_cost is None

    def test_the_published_figure_does_not_leak_into_the_next_completion(self, live, monkeypatch):
        """One RLM is reused across a round's runs; run two must not inherit run one."""
        client = MockLM(responses=["no answer here", "still nothing"])
        monkeypatch.setattr(live, "get_client", lambda *a, **k: client)

        rlm = an_rlm(live, max_iterations=1)
        rlm.completion("first")
        after_first = rlm.last_completion_usage.total_calls

        rlm._reset_completion_accounting()

        assert after_first > 0
        assert rlm.last_completion_usage is None

    def test_nothing_is_published_before_the_first_completion(self, live):
        assert an_rlm(live).last_completion_usage is None

    def test_the_published_total_is_read_only(self, live):
        rlm = an_rlm(live)
        with pytest.raises(AttributeError):
            rlm.last_completion_usage = None


class TestErrorNodeClassificationIsUnaffected:
    """R4: charging the parent must not disturb how trace analysis reads a node.

    ``walker.classify_node`` distinguishes a genuine answer beginning "Error: "
    from a real error node by the conjunct ``response.startswith(ERROR_PREFIX)
    and not usage``. Recording a lost child's spend must therefore go into the
    parent's accumulator, never onto the completion the sub-call hands back.
    """

    def test_a_failed_subcalls_completion_still_carries_empty_usage(self, live, monkeypatch):
        from shrlm.optimization.walker import NodeKind, WalkContext, classify_node

        parent = live.RLM(
            backend="openai", backend_kwargs={"model_name": "mock-model"}, max_depth=3
        )

        class ExplodingChild:
            last_completion_usage = None

            def __init__(self, *args, **kwargs):
                self.last_completion_usage = live.UsageSummary(
                    model_usage_summaries={
                        "mock-model": live.ModelUsageSummary(2, 10, 5, total_cost=0.06)
                    }
                )

            def completion(self, prompt, root_prompt=None):
                raise live.BudgetExceededError(spent=0.07, budget=0.05)

            def close(self):
                pass

        monkeypatch.setattr(live, "RLM", ExplodingChild)
        result = parent._subcall("prompt")

        # The parent was charged...
        assert parent._subcall_usage.total_cost == pytest.approx(0.06)
        # ...but the node the walker reads still looks like the error it is.
        assert result.usage_summary.model_usage_summaries == {}
        kind, _ = classify_node(
            result.to_dict(),
            depth=1,
            ctx=WalkContext(
                root_max_depth=3,
                recursion_available=True,
                block_attribution_reliable=True,
            ),
        )
        assert kind is NodeKind.ERRORED
