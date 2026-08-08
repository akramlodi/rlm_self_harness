"""The stage-2 boundary: load and gate ``shrlm-proposal/v1`` candidate proposals.

Stage 2 (the proposer, another developer's work) hands stage 3 a directory of
candidates, one ``proposal.json`` per candidate directory. Each proposal is the
paper's audit record — candidate identity, base-harness hash, targeted failure
signature, predicted behavioral effect, regression risks, proposer provenance —
wrapped around a full ``shrlm-harness/v1`` envelope (KTD1: a whole serialized
harness, never a surface delta; the one-surface rule is enforced by diffing the
candidate serialization against the incumbent's, which is stricter than
trusting a declared delta). The schema stage 2 targets is documented in
``docs/harness-proposal-interface.md``; this module is the enforcement.

Gate order is KTD2's, and it is load-bearing:

1. **Text-level gates, no candidate code in the process.** Schema validation,
   envelope hash recompute, base-hash-is-the-incumbent, exactly-one-surface
   diff, and the S6-within-caps comparison all operate on the JSON alone. A
   candidate rejected here never had a byte of its source written to disk,
   imported, or executed.
2. **Materialization plus ``check_harness``, inside a subprocess with a
   wall-clock timeout.** ``check_harness`` calls candidate callables (the S9
   probe, the S7 boundedness probe), so it sits inside the boundary: a
   candidate whose source raises, hangs, or fails an invariant check surfaces
   as a structured rejection while the host never blocks and never runs the
   code first. The subprocess also proves the round trip — the materialized
   harness re-serializes byte-identically to the envelope — before the host
   touches anything.
3. **Host materialization.** Only after the subprocess verdict does the host
   build the live ``Harness``: the edited surface comes from the envelope (and,
   for callable surfaces, from the generated module file), every unchanged
   surface reuses the incumbent's live objects (KTD2).

Callable surfaces are materialized *file-backed*: their source is written to a
real module file (``surfaces.py``) under the candidate's directory and imported
from there, so ``inspect.getsource`` recovers the exact source and the harness
re-serializes and re-hashes — a bare ``exec``-built callable cannot
re-serialize and would crash the evaluation driver's harness-hash step. The
module's import preamble (``CANDIDATE_MODULE_PREAMBLE``) is the entire
vocabulary candidate source may reference; it is a legibility contract and
defense-in-depth, not a security boundary — the REPL already executes
model-written code by design.

Rejections are values, never exceptions: expected-invalid input comes back as a
``CandidateRejection`` naming the gate and the violation, in the style of
``attribution.py``'s validate-with-named-violation loop, so the promotion
ledger can record every candidate including the ones that never ran.
"""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from shrlm.harness_identity import (
    HarnessSerializationError,
    hash_of_serialization,
    serialize_harness,
)
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import (
    AgentMechanism,
    CausalStatus,
    FailingLevel,
    VerifierCause,
)
from shrlm.rlm_harness import Harness
from shrlm.runner import check_harness

PROPOSAL_FORMAT = "shrlm-proposal/v1"
HARNESS_FORMAT = "shrlm-harness/v1"
PROPOSAL_FILENAME = "proposal.json"
MODULE_FILENAME = "surfaces.py"

DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS = 60.0

# Gate identifiers, in the order the gates run. Recorded on every rejection so
# the ledger can say not just why a candidate died but how far it got.
GATE_SCHEMA = "schema"
GATE_ENVELOPE_HASH = "envelope_hash"
GATE_BASE_HASH = "base_hash"
GATE_SURFACE_DIFF = "surface_diff"
GATE_CAPS = "caps"
GATE_MATERIALIZATION = "materialization"
GATE_HARNESS_CHECK = "harness_check"
GATE_ROUND_TRIP = "round_trip"

# Which serialization keys belong to which declared surface. S8 is one surface
# with two builders (see ``shrlm.rlm_harness.SURFACES``), so a candidate that
# edits both helper dicts still edits exactly one surface. A test asserts this
# grouping covers ``serialize_harness``'s surface keys exactly.
SURFACE_SERIALIZATION_KEYS: dict[str, tuple[str, ...]] = {
    "S1": ("S1_repl_contract",),
    "S2": ("S2_decomposition_instruction",),
    "S3": ("S3_execution_instruction",),
    "S4": ("S4_verification_instruction",),
    "S5": ("S5_recovery_instruction",),
    "S6": ("S6_runtime_policy",),
    "S7": ("S7_metadata",),
    "S8": ("S8_repl_helpers", "S8_sub_repl_helpers"),
    "S9": ("S9_answer_middleware",),
}

# The Harness field each string surface fills (builder convention: build_<x>
# fills <x>).
_STRING_SURFACE_FIELDS: dict[str, tuple[str, str]] = {
    "S1": ("repl_contract", "S1_repl_contract"),
    "S2": ("decomposition_instruction", "S2_decomposition_instruction"),
    "S3": ("execution_instruction", "S3_execution_instruction"),
    "S4": ("verification_instruction", "S4_verification_instruction"),
    "S5": ("recovery_instruction", "S5_recovery_instruction"),
}

# The four phi components of a target signature, each a closed vocabulary.
_SIGNATURE_FIELDS: dict[str, type] = {
    "verifier_cause": VerifierCause,
    "failing_level": FailingLevel,
    "causal_status": CausalStatus,
    "agent_mechanism": AgentMechanism,
}

# The import preamble of every generated surface module — the entire vocabulary
# candidate surface source may reference (documented in the contract doc). It
# covers what the incumbent surfaces themselves need (``AnswerDecision``,
# ``default_metadata_builder``) plus a small stdlib allowance. Anything outside
# it fails at import or probe time inside the gate subprocess and comes back as
# a structured rejection.
CANDIDATE_MODULE_PREAMBLE = '''\
"""Materialized candidate surfaces. Generated by shrlm.optimization.candidates.

Derived from a candidate proposal's serialized harness; edit the proposal, not
this file. The imports below are the entire vocabulary candidate surface source
may reference (see docs/harness-proposal-interface.md).
"""

import json
import math
import re
import textwrap
from typing import Any

from rlm.core.types import AnswerDecision
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH, default_metadata_builder
'''


class CandidateMaterializationError(RuntimeError):
    """A candidate serialization could not be turned into a live harness."""


@dataclass(frozen=True)
class CandidateRejection:
    """One candidate refused at one named gate, with the violation spelled out.

    A value, not an exception: expected-invalid stage-2 output must land in the
    promotion ledger with its reason, exactly like an unattributed record in
    the mining stage.
    """

    candidate_id: str
    gate: str
    reason: str
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "gate": self.gate,
            "reason": self.reason,
            "path": self.path,
        }


@dataclass(frozen=True)
class LoadedCandidate:
    """A candidate that survived every gate, ready for evaluation.

    ``harness`` is live: the edited surface materialized from the envelope,
    every other surface the incumbent's own object. ``harness_hash`` is the
    envelope hash, which the host re-derives from the materialized harness
    before returning — the same figure the evaluation driver's
    ``_prepare_round_dir`` will record.
    """

    candidate_id: str
    path: Path
    surface: str
    proposal: dict[str, Any]
    harness: Harness
    harness_hash: str
    module_path: Path


@dataclass(frozen=True)
class _CallableSlot:
    """One callable surface entry destined for the generated module file."""

    label: str
    alias: str
    source: str
    key: tuple[str, str]
    declared_bound: int | None = None


def _canonical(value: Any) -> str:
    """Canonical JSON text, the byte form every comparison here runs on."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Schema gate (text-level)
# ---------------------------------------------------------------------------


def _enum_violation(payload: dict[str, Any], field: str, enum_cls: type) -> str | None:
    value = payload.get(field)
    if not isinstance(value, str):
        return f"target_signature.{field} must be a string, got {type(value).__name__}"
    try:
        enum_cls(value)
    except ValueError:
        legal = ", ".join(member.value for member in enum_cls)
        return f"target_signature.{field} was {value!r}, not in the vocabulary. Legal: {legal}"
    return None


def _surfaces_violation(surfaces: Any) -> str | None:
    """Shape-check the nine serialized surfaces so later gates cannot raise."""
    expected = {key for keys in SURFACE_SERIALIZATION_KEYS.values() for key in keys}
    if not isinstance(surfaces, dict) or set(surfaces) != expected:
        return f"harness.harness.surfaces must carry exactly the keys {sorted(expected)}"
    for _, key in _STRING_SURFACE_FIELDS.values():
        if not isinstance(surfaces[key], str):
            return f"surfaces[{key!r}] must be a string"
    if not isinstance(surfaces["S6_runtime_policy"], dict):
        return "surfaces['S6_runtime_policy'] must be a dict"
    metadata = surfaces["S7_metadata"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("source"), str)
        or "declared_bound" not in metadata
    ):
        return "surfaces['S7_metadata'] must carry a string 'source' and a 'declared_bound'"
    for key in SURFACE_SERIALIZATION_KEYS["S8"]:
        helpers = surfaces[key]
        if not isinstance(helpers, dict):
            return f"surfaces[{key!r}] must be a dict"
        for name, entry in helpers.items():
            violation = _helper_entry_violation(f"surfaces[{key!r}][{name!r}]", entry)
            if violation:
                return violation
    middleware = surfaces["S9_answer_middleware"]
    if not isinstance(middleware, dict) or not isinstance(middleware.get("source"), str):
        return "surfaces['S9_answer_middleware'] must carry a string 'source'"
    return None


def _helper_entry_violation(label: str, entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return f"{label} must be a dict"
    inner = entry.get("tool") if "tool" in entry else entry
    if not isinstance(inner, dict):
        return f"{label}['tool'] must be a dict"
    kind = inner.get("kind")
    if kind == "callable":
        if not isinstance(inner.get("source"), str):
            return f"{label} callable entry must carry a string 'source'"
    elif kind == "value":
        if "value" not in inner:
            return f"{label} value entry must carry a 'value'"
    else:
        return f"{label} must declare kind 'callable' or 'value', got {kind!r}"
    return None


def _schema_violation(payload: Any) -> str | None:
    """The first schema violation in ``payload``, or None when it conforms.

    Unknown extra fields are tolerated (a v2 is additive); everything the
    loader and the ledger rely on is demanded here so no later gate can hit a
    KeyError on expected-invalid input.
    """
    if not isinstance(payload, dict):
        return f"proposal must be a JSON object, got {type(payload).__name__}"
    if payload.get("format") != PROPOSAL_FORMAT:
        return f"format must be {PROPOSAL_FORMAT!r}, got {payload.get('format')!r}"
    candidate_id = payload.get("candidate_id")
    if not isinstance(candidate_id, str) or not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(candidate_id):
        return (
            f"candidate_id {candidate_id!r} must be a filesystem-safe string matching "
            f"{FILESYSTEM_SAFE_ID_PATTERN.pattern}"
        )
    if not isinstance(payload.get("base_harness_hash"), str):
        return "base_harness_hash must be a string"
    surface = payload.get("surface")
    if surface not in SURFACE_SERIALIZATION_KEYS:
        return f"surface must be one of {sorted(SURFACE_SERIALIZATION_KEYS)}, got {surface!r}"
    signature = payload.get("target_signature")
    if not isinstance(signature, dict):
        return "target_signature must be an object with the four phi components"
    for field, enum_cls in _SIGNATURE_FIELDS.items():
        violation = _enum_violation(signature, field, enum_cls)
        if violation:
            return violation
    effect = payload.get("predicted_effect")
    if not isinstance(effect, str) or not effect.strip():
        return "predicted_effect must be a non-empty string"
    risks = payload.get("regression_risks")
    if not isinstance(risks, list) or not all(isinstance(risk, str) for risk in risks):
        return "regression_risks must be a list of strings"
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return "provenance must be an object"
    for field in ("model", "prompt_sha256"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            return f"provenance.{field} must be a non-empty string"
    envelope = payload.get("harness")
    if not isinstance(envelope, dict) or envelope.get("format") != HARNESS_FORMAT:
        return f"harness must be a {HARNESS_FORMAT!r} envelope"
    if not isinstance(envelope.get("hash"), str):
        return "harness.hash must be a string"
    serialization = envelope.get("harness")
    if not isinstance(serialization, dict):
        return "harness.harness must be the serialized harness object"
    if not isinstance(serialization.get("name"), str):
        return "harness.harness.name must be a string"
    if not isinstance(serialization.get("orchestrator"), bool):
        return "harness.harness.orchestrator must be a boolean"
    return _surfaces_violation(serialization.get("surfaces"))


# ---------------------------------------------------------------------------
# Diff and caps gates (text-level)
# ---------------------------------------------------------------------------


def changed_surfaces(
    base_serialization: dict[str, Any], candidate_serialization: dict[str, Any]
) -> list[str]:
    """The surface ids whose serialized content differs, in S1..S9 order."""
    base = base_serialization["surfaces"]
    candidate = candidate_serialization["surfaces"]
    return [
        surface_id
        for surface_id, keys in SURFACE_SERIALIZATION_KEYS.items()
        if any(_canonical(base[key]) != _canonical(candidate[key]) for key in keys)
    ]


def cap_violations(policy: dict[str, Any], caps: dict[str, int | float]) -> list[str]:
    """S6 values that exceed the experiment-owned caps (comparison only).

    The tighten-only *merge* is the cost governor's job (U2); this is the
    loader-level comparison R2 demands. A disabled policy binds nothing, so its
    values are inert and pass trivially.
    """
    if not policy.get("enabled"):
        return []
    violations = []
    for key in sorted(caps):
        value = policy.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            violations.append(f"S6 {key}={value!r} is not numeric, so it cannot honor the cap")
        elif value > caps[key]:
            violations.append(f"S6 {key}={value} exceeds the experiment-owned cap {caps[key]}")
    return violations


# ---------------------------------------------------------------------------
# Materialization: serialization -> module file -> live Harness
# ---------------------------------------------------------------------------


def _helper_callable_source(entry: dict[str, Any]) -> str | None:
    inner = entry.get("tool") if "tool" in entry else entry
    if isinstance(inner, dict) and inner.get("kind") == "callable":
        return inner["source"]
    return None


def _callable_slots(serialization: dict[str, Any]) -> list[_CallableSlot]:
    """Every callable surface entry, in deterministic module order."""
    surfaces = serialization["surfaces"]
    metadata = surfaces["S7_metadata"]
    bound = metadata["declared_bound"]
    if isinstance(bound, bool) or not isinstance(bound, int):
        raise CandidateMaterializationError(
            f"S7_metadata declared_bound must be an integer, got {bound!r}"
        )
    slots = [
        _CallableSlot(
            label="S7_metadata",
            alias="_candidate_s7_metadata",
            source=metadata["source"],
            key=("S7_metadata", ""),
            declared_bound=bound,
        )
    ]
    for surface_key in SURFACE_SERIALIZATION_KEYS["S8"]:
        for index, name in enumerate(sorted(surfaces[surface_key])):
            source = _helper_callable_source(surfaces[surface_key][name])
            if source is not None:
                slots.append(
                    _CallableSlot(
                        label=f"{surface_key}[{name!r}]",
                        alias=f"_candidate_{surface_key.lower()}_{index}",
                        source=source,
                        key=(surface_key, name),
                    )
                )
    slots.append(
        _CallableSlot(
            label="S9_answer_middleware",
            alias="_candidate_s9_answer_middleware",
            source=surfaces["S9_answer_middleware"]["source"],
            key=("S9_answer_middleware", ""),
        )
    )
    return slots


def _single_function_name(slot: _CallableSlot) -> str:
    """The def name of a slot's source, demanding exactly one undecorated def.

    The structural demand is what keeps the module file legible and the round
    trip exact: ``inspect.getsource`` on the imported function returns exactly
    the def block, so anything beyond one plain top-level ``def`` (imports,
    module-level statements, decorators) could not re-serialize byte-for-byte.
    """
    try:
        tree = ast.parse(slot.source)
    except SyntaxError as error:
        raise CandidateMaterializationError(f"{slot.label}: source does not parse: {error}") from (
            error
        )
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].decorator_list
    ):
        raise CandidateMaterializationError(
            f"{slot.label}: callable source must be exactly one undecorated top-level "
            "function definition"
        )
    return tree.body[0].name


def render_surface_module(serialization: dict[str, Any]) -> str:
    """The generated module text: preamble, then one aliased def per callable.

    Each def is followed immediately by its alias binding (and, for S7, its
    ``declared_bound``), so two sources sharing a def name cannot cross wires:
    the alias captures the object before any later def shadows the name.
    """
    parts = [CANDIDATE_MODULE_PREAMBLE]
    for slot in _callable_slots(serialization):
        def_name = _single_function_name(slot)
        source = slot.source if slot.source.endswith("\n") else slot.source + "\n"
        block = f"\n# {slot.alias}: {slot.label}\n{source}"
        if slot.declared_bound is not None:
            block += f"{def_name}.declared_bound = {slot.declared_bound}\n"
        block += f"{slot.alias} = {def_name}\n"
        parts.append(block)
    return "".join(parts)


def import_surface_module(module_path: Path) -> ModuleType:
    """Import the generated module under a content-addressed module name.

    The name hashes path and bytes, so reloading a rewritten file yields a
    fresh module while re-importing identical content is a cache hit, and two
    candidates' modules can never collide in ``sys.modules``.
    """
    resolved = module_path.resolve()
    data = resolved.read_bytes()
    digest = hashlib.sha256(str(resolved).encode("utf-8") + b"\x00" + data).hexdigest()[:16]
    module_name = f"_shrlm_candidate_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover - file exists by construction
        raise CandidateMaterializationError(f"cannot build an import spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _rebuild_helpers(
    surface_key: str,
    surfaces: dict[str, Any],
    module: ModuleType,
    aliases: dict[tuple[str, str], str],
) -> dict[str, Any]:
    """One S8 helper dict, callables from the module, data verbatim."""
    helpers: dict[str, Any] = {}
    for name, entry in surfaces[surface_key].items():
        if "tool" in entry:
            inner = entry["tool"]
            value = (
                getattr(module, aliases[(surface_key, name)])
                if inner.get("kind") == "callable"
                else inner["value"]
            )
            helpers[name] = {"tool": value, "description": entry.get("description")}
        elif entry.get("kind") == "callable":
            helpers[name] = getattr(module, aliases[(surface_key, name)])
        else:
            helpers[name] = entry["value"]
    return helpers


def surface_field_values(
    serialization: dict[str, Any], surface_id: str, module: ModuleType | None
) -> dict[str, Any]:
    """The ``Harness`` field assignments one surface id materializes to."""
    surfaces = serialization["surfaces"]
    if surface_id in _STRING_SURFACE_FIELDS:
        field, key = _STRING_SURFACE_FIELDS[surface_id]
        return {field: surfaces[key]}
    if surface_id == "S6":
        return {"runtime_policy": dict(surfaces["S6_runtime_policy"])}
    aliases = {slot.key: slot.alias for slot in _callable_slots(serialization)}
    if module is None:  # pragma: no cover - callers import the module first
        raise CandidateMaterializationError(f"surface {surface_id} needs the surface module")
    if surface_id == "S7":
        return {"metadata": getattr(module, aliases[("S7_metadata", "")])}
    if surface_id == "S8":
        return {
            "repl_helpers": _rebuild_helpers("S8_repl_helpers", surfaces, module, aliases),
            "sub_repl_helpers": _rebuild_helpers("S8_sub_repl_helpers", surfaces, module, aliases),
        }
    if surface_id == "S9":
        return {"answer_middleware": getattr(module, aliases[("S9_answer_middleware", "")])}
    raise CandidateMaterializationError(f"unknown surface id {surface_id!r}")


def materialize_harness(serialization: dict[str, Any], module_path: Path) -> Harness:
    """Build a full live harness from a serialization alone (the gate's view).

    Writes and imports the surface module, then assembles every surface from
    the serialized values — no incumbent objects involved, which is what lets
    the gate subprocess run self-contained on the proposal file.
    """
    module_path.write_text(render_surface_module(serialization))
    module = import_surface_module(module_path)
    fields: dict[str, Any] = {}
    for surface_id in SURFACE_SERIALIZATION_KEYS:
        fields.update(surface_field_values(serialization, surface_id, module))
    return Harness(name=serialization["name"], orchestrator=serialization["orchestrator"], **fields)


# ---------------------------------------------------------------------------
# The subprocess gate
# ---------------------------------------------------------------------------


def run_subprocess_gate(proposal_path: str) -> dict[str, Any]:
    """Materialize, ``check_harness``, and round-trip one proposal. Child-side.

    Runs inside the gate subprocess (``python -m shrlm.optimization.candidates
    <proposal.json>``): everything after the first line may execute candidate
    code, which is exactly why this function is never called in the host. Every
    failure — including an arbitrary exception out of candidate source — is a
    verdict dict, so the child's stdout is always one JSON line.
    """
    payload = json.loads(Path(proposal_path).read_text())
    serialization = payload["harness"]["harness"]
    module_path = Path(proposal_path).parent / MODULE_FILENAME
    try:
        harness = materialize_harness(serialization, module_path)
    except Exception as error:
        return {
            "ok": False,
            "gate": GATE_MATERIALIZATION,
            "reason": f"{type(error).__name__}: {error}",
        }
    try:
        check_harness(harness)
    except Exception as error:
        return {
            "ok": False,
            "gate": GATE_HARNESS_CHECK,
            "reason": f"{type(error).__name__}: {error}",
        }
    try:
        reserialized = serialize_harness(harness)
    except HarnessSerializationError as error:
        return {"ok": False, "gate": GATE_ROUND_TRIP, "reason": str(error)}
    if _canonical(reserialized) != _canonical(serialization):
        return {
            "ok": False,
            "gate": GATE_ROUND_TRIP,
            "reason": "the materialized harness does not re-serialize byte-identically to the "
            "envelope; callable sources must survive a getsource round trip",
        }
    return {"ok": True}


def _run_gate_subprocess(proposal_path: Path, timeout_seconds: float) -> dict[str, Any]:
    """Host-side wrapper: run the gate in a child under a wall-clock timeout."""
    command = [sys.executable, "-m", "shrlm.optimization.candidates", str(proposal_path)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "gate": GATE_MATERIALIZATION,
            "reason": f"materialization/check timed out after {timeout_seconds}s of wall clock; "
            "candidate source must not block",
        }
    stderr_tail = process.stderr[-500:] if process.stderr else ""
    if process.returncode != 0:
        return {
            "ok": False,
            "gate": GATE_MATERIALIZATION,
            "reason": f"gate subprocess exited {process.returncode}: {stderr_tail}",
        }
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    try:
        verdict = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict) or "ok" not in verdict:
        return {
            "ok": False,
            "gate": GATE_MATERIALIZATION,
            "reason": f"gate subprocess produced no verdict; stderr: {stderr_tail}",
        }
    return verdict


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


def load_candidate(
    proposal_path: Path | str,
    incumbent: Harness,
    *,
    caps: dict[str, int | float] | None = None,
    timeout_seconds: float = DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
) -> LoadedCandidate | CandidateRejection:
    """Gate one ``proposal.json`` against the incumbent, KTD2 order.

    Args:
        proposal_path: The candidate's ``proposal.json``.
        incumbent: The harness the candidate must declare as its base and
            differ from on exactly one surface.
        caps: Optional experiment-owned S6 maxima (e.g. ``{"max_depth": 2}``);
            an enabled candidate policy may not exceed any of them.
        timeout_seconds: Wall-clock bound on the materialization/check
            subprocess.

    Returns:
        A ``LoadedCandidate`` (live harness, edited surface, envelope hash) or
        a ``CandidateRejection`` naming the gate and the violation. Expected-
        invalid input never raises.
    """
    proposal_path = Path(proposal_path)
    fallback_id = proposal_path.parent.name

    def rejection(candidate_id: str, gate: str, reason: str) -> CandidateRejection:
        return CandidateRejection(
            candidate_id=candidate_id, gate=gate, reason=reason, path=str(proposal_path)
        )

    try:
        payload = json.loads(proposal_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return rejection(fallback_id, GATE_SCHEMA, f"proposal.json unreadable: {error}")

    violation = _schema_violation(payload)
    if violation:
        raw_id = payload.get("candidate_id") if isinstance(payload, dict) else None
        candidate_id = raw_id if isinstance(raw_id, str) and raw_id else fallback_id
        return rejection(candidate_id, GATE_SCHEMA, violation)

    candidate_id = payload["candidate_id"]
    envelope = payload["harness"]
    serialization = envelope["harness"]

    recomputed = hash_of_serialization(serialization)
    if recomputed != envelope["hash"]:
        return rejection(
            candidate_id,
            GATE_ENVELOPE_HASH,
            f"envelope declares hash {envelope['hash']} but its serialization hashes to "
            f"{recomputed}; the envelope was tampered with or built inconsistently",
        )

    incumbent_serialization = serialize_harness(incumbent)
    incumbent_hash = hash_of_serialization(incumbent_serialization)
    if payload["base_harness_hash"] != incumbent_hash:
        return rejection(
            candidate_id,
            GATE_BASE_HASH,
            f"candidate declares base {payload['base_harness_hash']} but the incumbent is "
            f"{incumbent_hash}; proposals must target the current harness",
        )

    if serialization["orchestrator"] != incumbent_serialization["orchestrator"]:
        return rejection(
            candidate_id,
            GATE_SURFACE_DIFF,
            "candidate changes the orchestrator scalar, which is not an editable surface",
        )
    changed = changed_surfaces(incumbent_serialization, serialization)
    if not changed:
        return rejection(
            candidate_id,
            GATE_SURFACE_DIFF,
            "candidate modifies no surface: its serialization is identical to the base",
        )
    if len(changed) > 1:
        return rejection(
            candidate_id,
            GATE_SURFACE_DIFF,
            f"candidate modifies {len(changed)} surfaces ({', '.join(changed)}); exactly one "
            "declared surface may change",
        )
    if changed != [payload["surface"]]:
        return rejection(
            candidate_id,
            GATE_SURFACE_DIFF,
            f"candidate declares surface {payload['surface']} but modifies {changed[0]}",
        )
    surface_id = changed[0]

    if caps:
        violations = cap_violations(serialization["surfaces"]["S6_runtime_policy"], caps)
        if violations:
            return rejection(candidate_id, GATE_CAPS, "; ".join(violations))

    verdict = _run_gate_subprocess(proposal_path, timeout_seconds)
    if not verdict.get("ok"):
        return rejection(
            candidate_id,
            str(verdict.get("gate", GATE_MATERIALIZATION)),
            str(verdict.get("reason", "gate subprocess rejected the candidate")),
        )

    # The subprocess vetted materialization and invariants; now build the live
    # harness the evaluation will run: edited surface from the envelope,
    # unchanged surfaces the incumbent's own objects (KTD2).
    module_path = proposal_path.parent / MODULE_FILENAME
    try:
        module = import_surface_module(module_path) if surface_id in ("S7", "S8", "S9") else None
        fields = surface_field_values(serialization, surface_id, module)
        harness = replace(incumbent, name=serialization["name"], **fields)
        materialized_hash = hash_of_serialization(serialize_harness(harness))
    except Exception as error:  # defensive: the subprocess already vetted this path
        return rejection(candidate_id, GATE_MATERIALIZATION, f"{type(error).__name__}: {error}")
    if materialized_hash != envelope["hash"]:
        return rejection(
            candidate_id,
            GATE_ROUND_TRIP,
            f"host materialization hashes to {materialized_hash}, not the envelope's "
            f"{envelope['hash']}",
        )

    return LoadedCandidate(
        candidate_id=candidate_id,
        path=proposal_path,
        surface=surface_id,
        proposal=payload,
        harness=harness,
        harness_hash=envelope["hash"],
        module_path=module_path,
    )


def load_candidates(
    proposals_dir: Path | str,
    incumbent: Harness,
    *,
    caps: dict[str, int | float] | None = None,
    timeout_seconds: float = DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
) -> tuple[list[LoadedCandidate], list[CandidateRejection]]:
    """Gate every candidate directory under ``proposals_dir``, in sorted order.

    Each immediate subdirectory is one candidate and must hold a
    ``proposal.json`` whose ``candidate_id`` equals the directory name — the
    ledger links candidates by id, so a mismatch is an auditability hole and is
    rejected rather than repaired. Nothing is silently dropped: a directory
    without a proposal file is itself a rejection.

    Returns:
        ``(loaded, rejections)`` covering every candidate directory exactly
        once.
    """
    loaded: list[LoadedCandidate] = []
    rejections: list[CandidateRejection] = []
    for entry in sorted(Path(proposals_dir).iterdir()):
        if not entry.is_dir():
            continue
        proposal_path = entry / PROPOSAL_FILENAME
        if not proposal_path.exists():
            rejections.append(
                CandidateRejection(
                    candidate_id=entry.name,
                    gate=GATE_SCHEMA,
                    reason=f"candidate directory carries no {PROPOSAL_FILENAME}",
                    path=str(entry),
                )
            )
            continue
        result = load_candidate(
            proposal_path, incumbent, caps=caps, timeout_seconds=timeout_seconds
        )
        if isinstance(result, LoadedCandidate) and result.candidate_id != entry.name:
            result = CandidateRejection(
                candidate_id=entry.name,
                gate=GATE_SCHEMA,
                reason=f"candidate_id {result.candidate_id!r} does not match its directory name "
                f"{entry.name!r}",
                path=str(proposal_path),
            )
        if isinstance(result, LoadedCandidate):
            loaded.append(result)
        else:
            rejections.append(result)
    return loaded, rejections


if __name__ == "__main__":
    print(json.dumps(run_subprocess_gate(sys.argv[1])))


__all__ = [
    "CANDIDATE_MODULE_PREAMBLE",
    "DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS",
    "HARNESS_FORMAT",
    "MODULE_FILENAME",
    "PROPOSAL_FILENAME",
    "PROPOSAL_FORMAT",
    "SURFACE_SERIALIZATION_KEYS",
    "CandidateMaterializationError",
    "CandidateRejection",
    "LoadedCandidate",
    "cap_violations",
    "changed_surfaces",
    "import_surface_module",
    "load_candidate",
    "load_candidates",
    "materialize_harness",
    "render_surface_module",
    "run_subprocess_gate",
    "surface_field_values",
]
