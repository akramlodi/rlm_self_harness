"""
Attribute a failed run to a closed-vocabulary failure signature.

The attributor is asked for two of the four signature components. The terminal
verifier cause comes from the verifier, and the failing level is derived from
sub-verdicts whenever a sub-verifier is available. That split is the point: it
is what makes an attribution more than a model's reading of a trace.

Output is validated against the enums rather than parsed leniently. An
off-vocabulary label would silently create a singleton cluster, so a rejected
response is re-asked with the violation named, and a response that never
validates is recorded as unattributed rather than coerced to OTHER.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from rlm.clients.base_lm import BaseLM
from shrlm.optimization.digest import TraceDigest
from shrlm.optimization.grounding import GroundingResult
from shrlm.optimization.taxonomy import (
    TAXONOMY_VERSION,
    AgentMechanism,
    CausalStatus,
    FailingLevel,
    render_failing_level_block,
    render_taxonomy_block,
)
from shrlm.optimization.types import (
    AttributionDetail,
    CallNode,
    FailureSignature,
    Verdict,
    iter_nodes,
)

PROMPT_VERSION = "1.0.0"
DEFAULT_MAX_ATTEMPTS = 3

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

ATTRIBUTOR_SYSTEM_PROMPT = """\
You are analyzing one failed run of a recursive language model, in order to \
describe *why* it failed in terms that generalize across runs.

A recursive language model keeps its input in a REPL variable, writes code to \
split that input into pieces, issues sub-calls over the pieces, and combines \
the results. The run below was rejected by a deterministic verifier.

Your job is diagnosis only. Do NOT propose, name, hint at, or describe any \
change to the system. Do not say what should be done differently. Describe \
only what happened and which reusable behavior produced it.

Choose exactly one value from each vocabulary below. Use the literal string \
value. If nothing fits, choose the "other"/"unattributed" member and explain \
in the corresponding detail field -- do not stretch a member that does not fit.

{taxonomy}
{failing_level}

Respond with a single fenced JSON block and nothing else:

```json
{{
  "causal_status": "<value>",
  "agent_mechanism": "<value>",
  "causal_status_detail": "<free text, required only for unattributed>",
  "agent_mechanism_detail": "<free text, required only for other>",{failing_level_field}
  "evidence_node_ids": ["<node_id from the sub-call table>"],
  "symptom_summary": "<one sentence describing the observed behavior>"
}}
```

Every entry in evidence_node_ids must be a node_id that appears in the run \
below. Do not invent identifiers.
"""

FAILING_LEVEL_FIELD = '\n  "failing_level": "<value>",'


class AttributionRejection(Exception):
    """A response that did not satisfy the output contract."""


@dataclass(frozen=True)
class AttributorConfig:
    """
    Everything that changes what the attributor produces.

    Hashed into the cache key, so a config change cannot silently reuse
    attributions made under different conditions.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    prompt_version: str = PROMPT_VERSION
    taxonomy_version: str = TAXONOMY_VERSION


@dataclass
class AttributionCache:
    """
    Persistent map from (prompt, digest, config) to the raw model response.

    Temperature zero against a hosted API is not determinism. This is what
    actually makes a mining round reproducible: a re-run with unchanged inputs
    replays byte-identical responses and costs nothing.
    """

    path: str | None = None
    entries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry["response"]

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def put(self, key: str, response: str) -> None:
        self.entries[key] = response
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as handle:
                json.dump({"key": key, "response": response}, handle)
                handle.write("\n")


def extract_json_block(text: str) -> dict[str, Any]:
    """
    Pull the single JSON object out of a model response.

    Falls back to the first balanced brace span when the model omits the fence,
    which is common enough that rejecting it would waste a retry on a response
    that is otherwise correct.
    """
    match = JSON_BLOCK_PATTERN.search(text)
    candidate = match.group(1) if match else None

    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise AttributionRejection("no JSON object found in the response")
        candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AttributionRejection(f"response was not valid JSON: {exc}") from None

    if not isinstance(parsed, dict):
        raise AttributionRejection("expected a JSON object")
    return parsed


def parse_enum(value: Any, enum_cls: type, field_name: str):
    """Look a label up in its closed vocabulary, listing the legal values on failure."""
    if not isinstance(value, str):
        raise AttributionRejection(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return enum_cls(value)
    except ValueError:
        legal = ", ".join(member.value for member in enum_cls)
        raise AttributionRejection(
            f"{field_name} was {value!r}, which is not in the vocabulary. Legal values: {legal}"
        ) from None


class LLMAttributor:
    """
    Attributes failed runs using a fixed language model.

    Takes a BaseLM instance rather than a backend name so that tests inject a
    stub with no patching, and so the attributor is never itself an RLM -- the
    mining stage must stay outside the system it is measuring.
    """

    def __init__(
        self,
        lm: BaseLM,
        config: AttributorConfig | None = None,
        cache: AttributionCache | None = None,
    ):
        self.lm = lm
        self.config = config or AttributorConfig()
        self.cache = cache or AttributionCache()

    def system_prompt(self, grounded: bool) -> str:
        """
        Render the instructions.

        The vocabularies are generated from the enums, so the prompt cannot
        drift from the code. The failing-level vocabulary appears only when no
        sub-verifier supplied it.
        """
        return ATTRIBUTOR_SYSTEM_PROMPT.format(
            taxonomy=render_taxonomy_block(),
            failing_level="" if grounded else "\n" + render_failing_level_block(),
            failing_level_field="" if grounded else FAILING_LEVEL_FIELD,
        )

    def prompt_sha256(self, grounded: bool) -> str:
        return hashlib.sha256(self.system_prompt(grounded).encode("utf-8")).hexdigest()

    def config_sha256(self) -> str:
        payload = json.dumps(
            {
                "model": self.lm.model_name,
                "sampling_args": self.lm.sampling_args,
                "max_attempts": self.config.max_attempts,
                "prompt_version": self.config.prompt_version,
                "taxonomy_version": self.config.taxonomy_version,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_key(self, digest: TraceDigest, grounded: bool, attempt: int) -> str:
        material = (
            f"{self.prompt_sha256(grounded)}|{digest.sha256}|{self.config_sha256()}|{attempt}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def validate(
        self,
        payload: dict[str, Any],
        root: CallNode,
        grounding: GroundingResult,
    ) -> tuple[CausalStatus, AgentMechanism, FailingLevel | None, AttributionDetail]:
        """Check a parsed response against the closed vocabularies and the tree."""
        causal_status = parse_enum(payload.get("causal_status"), CausalStatus, "causal_status")
        agent_mechanism = parse_enum(
            payload.get("agent_mechanism"), AgentMechanism, "agent_mechanism"
        )

        failing_level = None
        if not grounding.grounded:
            failing_level = parse_enum(payload.get("failing_level"), FailingLevel, "failing_level")

        evidence = payload.get("evidence_node_ids") or []
        if not isinstance(evidence, list):
            raise AttributionRejection("evidence_node_ids must be a list of node ids")

        known = {node.node_id for node in iter_nodes(root)}
        unknown = [node_id for node_id in evidence if node_id not in known]
        if unknown:
            raise AttributionRejection(
                f"evidence_node_ids contains identifiers that do not appear in the run: {unknown}"
            )

        detail = AttributionDetail(
            symptom_summary=str(payload.get("symptom_summary", "")),
            evidence_node_ids=[str(node_id) for node_id in evidence],
            failing_level_detail=str(payload.get("failing_level_detail", "")),
            causal_status_detail=str(payload.get("causal_status_detail", "")),
            agent_mechanism_detail=str(payload.get("agent_mechanism_detail", "")),
        )
        return causal_status, agent_mechanism, failing_level, detail

    def attribute(
        self,
        digest: TraceDigest,
        root: CallNode,
        verdict: Verdict,
        grounding: GroundingResult,
    ) -> tuple[FailureSignature, AttributionDetail]:
        """
        Produce a signature for one failed run.

        Raises AttributionRejection if no attempt validates. The caller records
        that as an unattributed record rather than guessing a label, so an
        unusable response stays visible in the bundle totals.
        """
        if verdict.passed or verdict.cause is None:
            raise ValueError("Only failed runs are attributed; this verdict passed")

        system = self.system_prompt(grounding.grounded)
        rejection = ""

        for attempt in range(self.config.max_attempts):
            user = digest.text
            if rejection:
                user = (
                    f"{digest.text}\n\n"
                    f"Your previous response was rejected: {rejection}\n"
                    f"Respond again, correcting only that problem."
                )

            key = self.cache_key(digest, grounding.grounded, attempt)
            response = self.cache.get(key)
            if response is None:
                response = self.lm.completion(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )
                self.cache.put(key, response)

            try:
                payload = extract_json_block(response)
                causal_status, mechanism, level, detail = self.validate(payload, root, grounding)
            except AttributionRejection as exc:
                rejection = str(exc)
                continue

            failing_level = grounding.failing_level if grounding.grounded else level
            assert failing_level is not None  # validate() guarantees one or the other
            signature = FailureSignature(
                verifier_cause=verdict.cause,
                failing_level=failing_level,
                causal_status=causal_status,
                agent_mechanism=mechanism,
            )
            return signature, detail

        raise AttributionRejection(
            f"no valid attribution after {self.config.max_attempts} attempts: {rejection}"
        )
