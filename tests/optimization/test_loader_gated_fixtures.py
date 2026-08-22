"""The committed loader-gated proposal artifacts stay valid under the current envelope (R9).

Two ``proposal.json`` documents in the repo are read by the candidate loader's
gates rather than by an analysis: the smoke run's promoted-surface proposal
under ``experiment_smoke/opt/round_01/proposals/`` and the live validation
smoke's hand-written candidate under ``examples/validation_rounds/proposals/``.
Both must carry the ``shrlm-harness/v2`` envelope with the full surface key
set, store a hash that recomputes from their stored serialization, and be
byte-for-byte what the production writers emit -- never hand-edited JSON.

The committed round trees (``experiment_smoke/opt/round_01/{mining,validation}``
and ``examples/mining_rounds/``) are real-run, pre-S10 evidence and are
deliberately not covered here; ``tests/experiment/test_rounds.py`` pins the
pre-S10 ``examples/mining_rounds/round_00/harness.json`` on purpose.
"""

import json
import shutil
from pathlib import Path

import pytest

from examples import validation_live_smoke
from shrlm.harness_identity import harness_hash, hash_of_serialization, serialize_harness
from shrlm.optimization.candidates import (
    HARNESS_FORMAT,
    MODULE_FILENAME,
    PROPOSAL_FILENAME,
    SURFACE_SERIALIZATION_KEYS,
    LoadedCandidate,
    load_candidate,
)
from shrlm.optimization.proposal import EDIT_KIND_TEXT, CandidateSpec, write_proposal
from shrlm.rlm_harness import H0

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PROPOSAL = (
    REPO_ROOT
    / "experiment_smoke"
    / "opt"
    / "round_01"
    / "proposals"
    / "r01-c01-s4"
    / PROPOSAL_FILENAME
)
VALIDATION_PROPOSAL = (
    REPO_ROOT
    / "examples"
    / "validation_rounds"
    / "proposals"
    / "smoke-s2-restate"
    / PROPOSAL_FILENAME
)
LOADER_GATED_PROPOSALS = (SMOKE_PROPOSAL, VALIDATION_PROPOSAL)
EXPECTED_SURFACE_KEYS = {key for keys in SURFACE_SERIALIZATION_KEYS.values() for key in keys}


@pytest.fixture(params=LOADER_GATED_PROPOSALS, ids=lambda path: path.parent.name)
def proposal_path(request) -> Path:
    return request.param


def test_fixture_carries_the_v2_tag_and_eleven_surface_keys(proposal_path):
    payload = json.loads(proposal_path.read_text())
    envelope = payload["harness"]
    assert envelope["format"] == HARNESS_FORMAT == "shrlm-harness/v2"
    keys = set(envelope["harness"]["surfaces"])
    assert len(keys) == 11
    assert keys == EXPECTED_SURFACE_KEYS


def test_fixture_stored_hash_recomputes_from_its_stored_serialization(proposal_path):
    payload = json.loads(proposal_path.read_text())
    envelope = payload["harness"]
    assert hash_of_serialization(envelope["harness"]) == envelope["hash"]
    assert payload["base_harness_hash"] == harness_hash(H0)


def test_fixture_passes_the_candidate_loader_and_its_module_is_the_gate_render(
    proposal_path, tmp_path
):
    """Loader-gated means exactly this: ``load_candidate`` against H0 accepts
    it, and the committed ``surfaces.py`` is what the gate subprocess writes."""
    copy_dir = tmp_path / proposal_path.parent.name
    copy_dir.mkdir()
    copied = copy_dir / PROPOSAL_FILENAME
    shutil.copy(proposal_path, copied)

    payload = json.loads(proposal_path.read_text())
    result = load_candidate(copied, H0)

    assert isinstance(result, LoadedCandidate), result
    assert result.surface == payload["surface"]
    assert result.harness_hash == payload["harness"]["hash"]
    committed_module = proposal_path.parent / MODULE_FILENAME
    assert (copy_dir / MODULE_FILENAME).read_text() == committed_module.read_text()


def test_smoke_proposal_is_byte_identical_to_the_production_writer_output(tmp_path):
    """``write_proposal`` over the document's own content reproduces the file."""
    payload = json.loads(SMOKE_PROPOSAL.read_text())
    serialization = payload["harness"]["harness"]
    spec = CandidateSpec(
        pattern_index=0,
        pattern={"signature": payload["target_signature"]},
        surface=payload["surface"],
        edit={
            "kind": EDIT_KIND_TEXT,
            "new_text": serialization["surfaces"]["S4_verification_instruction"],
        },
        predicted_effect=payload["predicted_effect"],
        regression_risks=list(payload["regression_risks"]),
    )
    written = write_proposal(
        tmp_path,
        payload["candidate_id"],
        serialize_harness(H0),
        spec,
        serialization,
        payload["provenance"]["model"],
        payload["provenance"]["prompt_sha256"],
    )
    assert written.read_bytes() == SMOKE_PROPOSAL.read_bytes()


def test_validation_smoke_script_constructs_the_committed_proposal_and_it_loads(tmp_path):
    """The example script's writer is the production writer for its fixture."""
    validation_live_smoke.write_hand_written_candidate(tmp_path)
    written = tmp_path / validation_live_smoke.CANDIDATE_ID / PROPOSAL_FILENAME

    assert written.read_bytes() == VALIDATION_PROPOSAL.read_bytes()
    result = load_candidate(written, H0)
    assert isinstance(result, LoadedCandidate), result
    assert result.surface == "S2"
