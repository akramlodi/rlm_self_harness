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

``crashing_client_factory`` is the failure fixture: it raises on the first
completion, so the subject's worker exits non-zero after persisting nothing.
"""

import json
from pathlib import Path
from typing import Any

from tests.optimization.test_validation import COST_PER_CALL, ClientFactory


def scripted_client_factory(args: dict[str, Any]) -> ClientFactory:
    script = json.loads(Path(args["script_path"]).read_text())
    return ClientFactory(list(script), float(args.get("cost_per_call", COST_PER_CALL)))


class _CrashingFactory:
    def __call__(self, backend: str, backend_kwargs: dict[str, Any] | None) -> Any:
        raise RuntimeError("scripted crash: this subject's worker must fail")


def crashing_client_factory(args: dict[str, Any]) -> _CrashingFactory:
    return _CrashingFactory()


def write_script(path: Path, script: list[str]) -> Path:
    """Persist one subject's scripted turns where a child can read them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script))
    return path


SCRIPTED_FACTORY = "tests.optimization.subject_worker_support:scripted_client_factory"
CRASHING_FACTORY = "tests.optimization.subject_worker_support:crashing_client_factory"
