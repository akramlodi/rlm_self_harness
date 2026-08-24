"""Child-process test seam for the validation run worker.

The in-process tests script the runtime by monkeypatching
``rlm.core.rlm.get_client``; a run child sees neither the monkeypatch nor the
parent's script position. This module gives the run worker's request file a
dotted path it can resolve *inside the child*, so a fanned-out round replays
exactly the turns a sequential one would.

Each child is a fresh process executing exactly one run, so each reads the
script from the start -- which is precisely the sequential path's per-run
behaviour, where every run gets a fresh harness and its own turns.

Factory args::

    {"script_path": "<path to a JSON list of scripted turns>",
     "cost_per_call": 0.001,          # optional
     "crash": true,                   # raise instead of completing
     "hang": true,                    # never return, to exercise deadlines
     "witness_dir": "<dir>"}          # optional, records concurrent liveness
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from tests.optimization.test_validation import COST_PER_CALL, ClientFactory


class _CrashingFactory:
    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> Any:
        raise RuntimeError("scripted crash: this run's worker must fail")


class _WitnessingFactory:
    """A scripted factory that records how many children were alive at once.

    Each child writes a marker on entry and removes it on exit, noting the
    peak it observed. The parent takes the maximum across children, which is
    what proves the slot cap actually bounds concurrency rather than merely
    being configured.
    """

    def __init__(self, script: list[str], cost_per_call: float, witness_dir: Path, hold: float):
        self._inner = ClientFactory(script, cost_per_call)
        self._witness_dir = witness_dir
        self._hold = hold

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> Any:
        client = self._inner(backend, backend_kwargs)
        original = client.completion
        witness_dir = self._witness_dir
        hold = self._hold

        def completion(prompt):
            live = witness_dir / "live"
            live.mkdir(parents=True, exist_ok=True)
            marker = live / str(os.getpid())
            marker.write_text("1")
            try:
                peak = len(list(live.iterdir()))
                peaks = witness_dir / "peak"
                peaks.mkdir(parents=True, exist_ok=True)
                (peaks / str(os.getpid())).write_text(str(peak))
                if hold:
                    time.sleep(hold)
                return original(prompt)
            finally:
                marker.unlink(missing_ok=True)

        client.completion = completion  # type: ignore[method-assign]
        return client

    @property
    def total_calls(self) -> int:
        return getattr(self._inner, "total_calls", 0)


class _HangingFactory:
    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> Any:
        class _Hanging:
            model_name = "hanging"

            def completion(self, prompt):
                while True:
                    time.sleep(0.01)

            def get_usage_summary(self):
                from rlm.core.types import UsageSummary

                return UsageSummary(model_usage_summaries={})

            def get_last_usage(self):
                from rlm.core.types import ModelUsageSummary

                return ModelUsageSummary(0, 0, 0, total_cost=None)

        return _Hanging()


def run_scripted_client_factory(args: dict[str, Any]) -> Any:
    if args.get("crash"):
        return _CrashingFactory()
    if args.get("hang"):
        return _HangingFactory()
    script = json.loads(Path(args["script_path"]).read_text())
    cost = float(args.get("cost_per_call", COST_PER_CALL))
    if "witness_dir" in args:
        return _WitnessingFactory(
            list(script), cost, Path(args["witness_dir"]), float(args.get("hold", 0.0))
        )
    return ClientFactory(list(script), cost)


def observed_peak_concurrency(witness_dir: Path) -> int:
    """The largest number of simultaneously-executing children any child saw."""
    peaks = witness_dir / "peak"
    if not peaks.exists():
        return 0
    return max(int(path.read_text()) for path in peaks.iterdir())


def write_script(path: Path, script: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script))
    return path


RUN_SCRIPTED_FACTORY = "tests.optimization.run_worker_support:run_scripted_client_factory"
