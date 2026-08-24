"""Child-process test seam for the validation subject worker (KTD9).

The in-process tests script the runtime by monkeypatching
``rlm.core.rlm.get_client``; a child process sees neither the monkeypatch nor
the parent's script position. This module gives the worker's request file a
dotted path it can resolve *inside the child*: ``scripted_client_factory``
reads a per-subject script from disk and returns the same ``ClientFactory``
the in-process tests use, so a parallel round replays exactly the turns a
sequential one would.

Per-subject args (the request's ``client_factory`` args for this subject)::

    {"script_path": "<path to a JSON list of scripted turns>",
     "cost_per_call": 0.001}          # optional, defaults to COST_PER_CALL

``{"crash": true}`` selects the failure fixture instead: a client that raises
on the first completion, so that subject's worker exits non-zero after
persisting nothing.
"""

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any

from tests.optimization.test_validation import COST_PER_CALL, ClientFactory, ScriptedLM


class _CrashingFactory:
    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> Any:
        raise RuntimeError("scripted crash: this subject's worker must fail")


def crashing_client_factory(args: dict[str, Any]) -> _CrashingFactory:
    return _CrashingFactory()


class _TrackingFactory(ClientFactory):
    """A ``ClientFactory`` that records how many sibling children are alive.

    With ``concurrency_dir`` set, every child leaves a ``pid`` marker there on
    creation and removes it at exit; each completion counts the markers and
    records the maximum it saw in ``<concurrency_dir>/max/<pid>``. ``hold``
    sleeps that long inside the first completion so siblings overlap.
    """

    def __init__(
        self, script: list[str], cost_per_call: float, concurrency_dir: Path | None, hold: float
    ):
        super().__init__(script, cost_per_call)
        self.concurrency_dir = concurrency_dir
        self.hold = hold
        self.max_seen = 0
        self._held = False
        if concurrency_dir is not None:
            (concurrency_dir / "max").mkdir(parents=True, exist_ok=True)
            (concurrency_dir / str(os.getpid())).write_text("alive")
            atexit.register(self._leave)

    def _leave(self) -> None:
        assert self.concurrency_dir is not None
        (self.concurrency_dir / "max" / str(os.getpid())).write_text(str(self.max_seen))
        marker = self.concurrency_dir / str(os.getpid())
        if marker.exists():
            marker.unlink()

    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> ScriptedLM:
        client = super().__call__(backend, backend_kwargs)
        original = client.completion

        def completion(prompt: Any) -> str:
            if self.concurrency_dir is not None:
                alive = [p for p in self.concurrency_dir.iterdir() if p.is_file()]
                self.max_seen = max(self.max_seen, len(alive))
            if self.hold and not self._held:
                self._held = True
                time.sleep(self.hold)
            return original(prompt)

        client.completion = completion  # type: ignore[method-assign]
        return client


def scripted_client_factory(args: dict[str, Any]) -> Any:
    if args.get("crash"):
        return _CrashingFactory()
    script = json.loads(Path(args["script_path"]).read_text())
    concurrency_dir = Path(args["concurrency_dir"]) if "concurrency_dir" in args else None
    return _TrackingFactory(
        list(script),
        float(args.get("cost_per_call", COST_PER_CALL)),
        concurrency_dir,
        float(args.get("hold", 0.0)),
    )


def max_concurrency(concurrency_dir: Path) -> int:
    """The largest number of simultaneously alive children any child observed."""
    return max(int(path.read_text()) for path in (concurrency_dir / "max").iterdir())


def write_script(path: Path, script: list[str]) -> Path:
    """Persist one subject's scripted turns where a child can read them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script))
    return path


SCRIPTED_FACTORY = "tests.optimization.subject_worker_support:scripted_client_factory"
