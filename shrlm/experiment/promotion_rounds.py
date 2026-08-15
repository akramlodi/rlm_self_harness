"""Shared discovery over a completed run's promotion ledgers, round by round.

``surface_activity.py`` (Graph 1: harness complexity/lever-touch over time)
and ``incumbent_quality.py`` (Graph 2: incumbent quality over time) both walk
the same on-disk structure -- one ``promotions.jsonl`` per validation round,
under ``<out_dir>/opt/round_NN/validation/round_NN/`` (the nesting is the
orchestrator's own layout: ``experiment_round_dir`` then ``round_dir`` again
under ``VALIDATION_DIR``) -- so the glob-and-parse logic that finds them
lives here once instead of twice.

``round_index`` is not a field inside any ``promotions.jsonl`` record (see
``shrlm.optimization.validation._ledger_record``); it is recoverable only
from the directory name, which is why this module resolves it from disk
rather than trusting anything in the record payloads.
"""

from collections.abc import Iterator
from pathlib import Path

from shrlm.experiment.orchestrator import OPT_DIR, VALIDATION_DIR
from shrlm.optimization.bundle import round_dir
from shrlm.optimization.validation import PROMOTIONS_FILENAME, load_promotion_ledger

ROUND_DIR_PATTERN = "round_*"


def discover_validation_rounds(out_dir: Path | str) -> list[tuple[int, Path]]:
    """Every round with a persisted promotion ledger, in ascending round order.

    Returns ``(round_index, ledger_round_path)`` pairs. A round the
    orchestrator never wrote a ledger for (the pathological empty-proposals
    round noted in ``validate_round``'s docstring) is simply absent -- callers
    should not assume the returned indices are contiguous from 1.
    """
    opt_root = Path(out_dir) / OPT_DIR
    found: dict[int, Path] = {}
    if not opt_root.is_dir():
        return []
    for opt_round_path in sorted(opt_root.glob(ROUND_DIR_PATTERN)):
        try:
            round_index = int(opt_round_path.name.removeprefix("round_"))
        except ValueError:
            continue
        ledger_round_path = round_dir(opt_round_path / VALIDATION_DIR, round_index)
        if (ledger_round_path / PROMOTIONS_FILENAME).exists():
            found[round_index] = ledger_round_path
    return sorted(found.items())


def iter_promotion_rounds(
    out_dir: Path | str,
) -> Iterator[tuple[int, list[dict], dict]]:
    """``(round_index, records, decision)`` for every ledgered round, in order.

    ``records`` are the ``promotions.jsonl`` lines (one per candidate, plus
    the merged harness's own record when the round's plan merged);
    ``decision`` is the round's ``decision.json`` payload.
    """
    for round_index, ledger_round_path in discover_validation_rounds(out_dir):
        records, decision = load_promotion_ledger(ledger_round_path)
        yield round_index, records, decision


__all__ = ["discover_validation_rounds", "iter_promotion_rounds"]
