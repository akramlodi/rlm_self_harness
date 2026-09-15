"""Per-thread context for recursive sub-calls.

``rlm_query_batched`` runs child RLMs on a thread pool. Each child is spawned
through ``RLM._subcall`` on its own worker thread, and every sibling computes
its budget from the same parent ``_cumulative_cost`` snapshot (child spend is
folded in only when a child completes). Without a share, a batch of N children
could each claim the whole remainder -- N x the cap. The batch sets the share
here, on the worker thread, right before the sub-call; ``RLM._subcall`` reads
it. A thread-local rather than a new ``subcall_fn`` argument so every existing
two-argument callback (tests, custom environments) keeps working unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_state = threading.local()


def budget_fraction() -> float:
    """The share of the parent's remaining budget the current sub-call may use."""
    return float(getattr(_state, "budget_fraction", 1.0))


@contextmanager
def budget_share(fraction: float) -> Iterator[None]:
    """Scope within which sub-calls on this thread get ``fraction`` of the remainder."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"budget share must be in (0, 1], got {fraction}")
    previous = getattr(_state, "budget_fraction", 1.0)
    _state.budget_fraction = fraction
    try:
        yield
    finally:
        _state.budget_fraction = previous
