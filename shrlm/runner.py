"""Assemble a harnessed RLM from a ``Harness``, enforce what is enforceable, monitor the rest.

One entry point, ``build_harnessed_rlm``, wires the ten surfaces into the
runtime: S1-S5 concatenate into the root system prompt in phase order, S6 becomes
the runtime policy plus the ``max_depth`` scalar, S7 becomes the metadata seam,
S8 becomes ``custom_tools`` / ``custom_sub_tools``, S9 becomes the
answer-detection seam, and S10's index joins the system prompt while its bodies
sit behind the fixed ``load_skill`` loader the runner installs into both tool
dicts whenever the list is non-empty (KTD9). The ``orchestrator`` scalar comes
from the harness.

Two derivations run before anything is constructed, so the prompt cannot state a
limit the runtime does not honor:

- S1's truncation sentence is derived from the active S7 builder's
  ``declared_bound`` (the convention documented in ``shrlm.rlm_harness``) and the
  assembled prompt must state exactly that figure.
- S6's capacity sentence is derived from the active policy and injected into the
  metadata user message; where the prompt text also states a per-prompt capacity
  or a per-batch fan-out, it must agree with the policy.

Invariant protection is two-tier, and the tiers are not interchangeable.

**Structural** (this module, at construction, raises):

- *I1 as a growth property.* The wiring version is not implementable: at the S7
  seam ``format_iteration`` receives an ``RLMIteration`` whose
  ``code_blocks[i].result.locals`` is a full copy of the REPL namespace, and the
  prompt lives inside that same dict as ``context_0`` / ``context``. One object
  carries both, so "wired to receive the prompt" is not a distinguishable
  construction-time decision. U1's redaction already makes the values
  unreachable; this module adds the probe -- call the builder with synthetic REPL
  states at 1x and 100x prompt length and assert the output does not grow.
- *Plumbing present.* The programmatic sub-call path and the answer-from-variable
  return path are always injected, and no surface may shadow them, so I2 and I3
  cannot be violated structurally.
- *Non-program-rewriting* (KTD1 condition 1). S9 is handed only the detected
  answer and the redacted inventory, and its return type admits only "unchanged"
  or "redirect". It has no channel through which to rewrite the root's program.
  This is the mechanically enforceable half; the judgment half stays with the
  human reviewing a proposed edit.
- *Incidental hazards.* ``other_backends`` silently swaps the child model, and
  tracing callbacks and hard token/budget/timeout caps are experiment-owned; a
  harness that supplies any of them is rejected.

**Behavioral** (trace-monitored, reported and never prevented): an S2 or S9 edit
can discourage recursion in prose ("just answer directly") or push the root to
verbalize without touching any invariant mechanically. Prose is not mechanically
checkable, so I2 and I3 *in practice* are preregistered constraints observed
through U1's trace metrics -- ``run_metrics`` aggregates them per run and
``acceptance_inputs`` exposes the cost-aware gate's inputs. Nothing here raises
on a low sub-call count; that judgment belongs to the optimization loop.
"""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlm.core.rlm import RLM
from rlm.core.types import ANSWER_SUBMITTED, AnswerDecision, RLMChatCompletion
from rlm.environments.base_env import SkillLoader, validate_custom_tools
from rlm.logger import RLMLogger
from rlm.utils.prompts import DEFAULT_CAPACITY_SENTENCE, ORCHESTRATOR_ADDENDUM
from shrlm.rlm_harness import (
    SKILL_LOADER_DESCRIPTION,
    SKILL_LOADER_NAME,
    AnswerMiddlewareFn,
    Harness,
    MetadataFn,
    SkillEntry,
    assemble_system_prompt,
)

# ---------------------------------------------------------------------------
# Constants: the probe, the derived sentences, and the ownership boundaries
# ---------------------------------------------------------------------------

# The synthetic prompt used by the I1 boundedness probe, and the factor it grows
# by. 100x is far past any real head-truncation bound, so a builder that copies
# any constant fraction of the REPL state through is caught.
PROBE_PROMPT_CHARS = 50_000
PROBE_GROWTH_FACTOR = 100
# Growth allowance. The shipped default appends "... + [N chars...]", whose
# decimal width grows logarithmically with the discarded remainder; that is
# bookkeeping, not prompt content.
PROBE_SLACK_CHARS = 64

PROBE_ANSWER = "boundedness probe answer"

# The REPL names that carry I2 (sub-calls issued by code) and I3 (the answer
# accumulated in a variable and returned from it). No surface may shadow them.
#
# This is a semantic subset of ``base_env.RESERVED_TOOL_NAMES``, not a copy of
# it: reserved names like ``SHOW_VARS`` are protected for other reasons and
# carry no invariant. ``validate_custom_tools`` already rejects shadowing of
# every reserved name; listing these separately is what produces the invariant-
# specific error instead of a generic one. If the runtime ever gains another
# sub-call primitive or answer channel, add it here as well as to the reserved
# set — otherwise it is still protected, but the failure stops explaining why.
REQUIRED_REPL_PLUMBING: tuple[str, ...] = (
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
    "answer",
)

# The S6 policy's own fields. Anything else in the dict is either an incidental
# hazard (below) or a typo; both are rejected.
#
# Deliberately absent: per-turn and cumulative sub-call ceilings. The runtime has
# no counter enforcing them, so accepting them here would hand a proposer a knob
# it could set, validate, and promote while nothing changed — a silently inert
# edit is worse than a rejected one. Add them back together with the counters.
POLICY_FIELDS: frozenset[str] = frozenset(
    {
        "enabled",
        "max_prompt_chars",
        "max_batch_width",
        "max_depth",
        "retry_on_syntax_error",
        "max_retries",
        "validate_sub_output",
    }
)

# Depth routing silently swaps the child model (``rlm/core/lm_handler.py:188``),
# which would confound every measured harness effect.
BACKEND_ROUTING_KEYS: frozenset[str] = frozenset({"other_backends", "other_backend_kwargs"})

# Tracing callbacks and hard caps belong to the experiment, not to a harness: a
# harness that could set them could buy a pass-rate win by spending budget the
# acceptance gate is supposed to price.
EXPERIMENT_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "max_budget",
        "max_tokens",
        "max_timeout",
        "max_errors",
        "on_subcall_start",
        "on_subcall_complete",
        "on_iteration_start",
        "on_iteration_complete",
    }
)

# ``RLM`` kwargs this module fills from the harness. A caller that also passes one
# is either fighting the harness or unaware of it; both are errors.
HARNESS_OWNED_KWARGS: frozenset[str] = frozenset(
    {
        "custom_system_prompt",
        "orchestrator",
        "runtime_policy",
        "metadata_builder",
        "answer_middleware",
        "custom_tools",
        "custom_sub_tools",
        "capacity_sentence",
        "logger",
    }
)

TRUNCATION_SENTENCE = "REPL outputs over {bound} characters are truncated"
TRUNCATION_PATTERN = re.compile(r"REPL outputs over (\S+) characters are truncated")
CAPACITY_SENTENCE = "Each sub-LLM call can handle roughly {bound} characters at once."
PER_PROMPT_PATTERN = re.compile(r"(~[\d.]+[KM]?) characters per prompt")
PER_BATCH_PATTERN = re.compile(r"~(\d+) prompts per batch")


# ---------------------------------------------------------------------------
# Derived sentences
# ---------------------------------------------------------------------------


def format_char_bound(chars: int) -> str:
    """Render a character bound the way the prompts state it: ``20000`` -> ``~20K``."""
    if chars % 1000 == 0:
        return f"~{chars // 1000}K"
    return f"~{chars}"


def declared_metadata_bound(builder: MetadataFn) -> int:
    """Read an S7 builder's declared bound, or fail loudly.

    Args:
        builder: The active S7 builder.

    Returns:
        The largest execution-result size the builder lets through.

    Raises:
        ValueError: If the builder omits ``declared_bound`` or declares a
            non-positive one. Guessing a default here is what would let S1 and S7
            drift, which is the exact failure this convention exists to prevent.
    """
    bound = getattr(builder, "declared_bound", None)
    if not isinstance(bound, int) or isinstance(bound, bool) or bound <= 0:
        raise ValueError(
            f"S7 builder {getattr(builder, '__name__', builder)!r} must set a positive "
            "integer `declared_bound` attribute stating the largest execution-result "
            "size it lets through; the runner derives S1's truncation sentence from it."
        )
    return bound


def effective_system_prompt(harness: Harness) -> str:
    """The system prompt the root model will actually see, addendum included.

    ``assemble_system_prompt`` concatenates S1-S5; ``build_rlm_system_prompt``
    appends ``ORCHESTRATOR_ADDENDUM`` at completion time when the harness sets
    ``orchestrator=True``. The stated-limit checks have to read both, so this
    reproduces the concatenation exactly.
    """
    text = assemble_system_prompt(harness)
    if harness.orchestrator:
        return f"{text}\n\n{ORCHESTRATOR_ADDENDUM}"
    return text


def derive_truncation_sentence(harness: Harness) -> str:
    """The truncation sentence implied by the harness's active S7 bound."""
    bound = declared_metadata_bound(harness.metadata)
    return TRUNCATION_SENTENCE.format(bound=format_char_bound(bound))


def derive_capacity_sentence(runtime_policy: dict[str, Any]) -> str:
    """The sub-call capacity sentence implied by the active S6 policy.

    A policy that declares no per-prompt cap leaves the shipped sentence in place,
    which is what keeps H0* byte-identical to the reference harness.
    """
    max_prompt_chars = runtime_policy.get("max_prompt_chars")
    if not runtime_policy.get("enabled") or max_prompt_chars is None:
        return DEFAULT_CAPACITY_SENTENCE
    return CAPACITY_SENTENCE.format(bound=format_char_bound(int(max_prompt_chars)))


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def probe_repl_state(scale: int) -> tuple[str, dict[str, tuple[str, int]]]:
    """A synthetic (execution result, redacted inventory) pair at ``scale`` x prompt length."""
    length = PROBE_PROMPT_CHARS * scale
    stdout = f"\n{'p' * length}\n\nREPL variables: ['context_0', 'context', 'answer', 'buffer']\n"
    inventory = {
        "context_0": ("str", length),
        "context": ("str", length),
        "answer": ("dict", 2),
        "buffer": ("str", length),
    }
    return stdout, inventory


def check_metadata_boundedness(builder: MetadataFn) -> None:
    """I1 as a growth property: S7's output must not grow with the prompt.

    Raises:
        ValueError: If the builder's output grows between the 1x and 100x probe,
            or exceeds its own declared bound at 100x.
    """
    bound = declared_metadata_bound(builder)
    small = builder(*probe_repl_state(1))
    large = builder(*probe_repl_state(PROBE_GROWTH_FACTOR))
    growth = len(large) - len(small)
    if growth > PROBE_SLACK_CHARS:
        raise ValueError(
            f"S7 builder {getattr(builder, '__name__', builder)!r} violates I1: its output "
            f"grew by {growth} characters when the synthetic REPL state grew "
            f"{PROBE_GROWTH_FACTOR}x ({len(small)} -> {len(large)}). What carries across "
            "turns must be bounded independently of the prompt."
        )
    if len(large) > bound + PROBE_SLACK_CHARS:
        raise ValueError(
            f"S7 builder {getattr(builder, '__name__', builder)!r} returned {len(large)} "
            f"characters, over its declared bound of {bound}. The declared bound is what "
            "S1's truncation sentence states, so it has to be true."
        )


def check_stated_limits(harness: Harness) -> None:
    """The prompt may not state a truncation bound or a capacity the runtime will not honor.

    Raises:
        ValueError: If the assembled prompt states no truncation figure, states one
            other than the active S7 bound, or states a per-prompt capacity or
            per-batch fan-out that the active S6 policy contradicts.
    """
    prompt = effective_system_prompt(harness)
    expected = derive_truncation_sentence(harness)
    stated = set(TRUNCATION_PATTERN.findall(prompt))
    expected_figure = format_char_bound(declared_metadata_bound(harness.metadata))
    if stated != {expected_figure}:
        raise ValueError(
            f"S1 must state the truncation bound the active S7 builder honors. Expected "
            f"exactly {expected!r}; the assembled prompt states {sorted(stated) or 'nothing'}."
        )

    policy = harness.runtime_policy
    if not policy.get("enabled"):
        return

    max_prompt_chars = policy.get("max_prompt_chars")
    if max_prompt_chars is not None:
        expected_capacity = format_char_bound(int(max_prompt_chars))
        stated_capacity = set(PER_PROMPT_PATTERN.findall(prompt))
        if stated_capacity - {expected_capacity}:
            raise ValueError(
                f"The prompt states a sub-call capacity of {sorted(stated_capacity)} per "
                f"prompt, but S6 enforces max_prompt_chars={max_prompt_chars} "
                f"({expected_capacity}). The model would be told a false fact about its "
                "own environment."
            )

    max_batch_width = policy.get("max_batch_width")
    if max_batch_width is not None:
        stated_width = set(PER_BATCH_PATTERN.findall(prompt))
        if stated_width - {str(max_batch_width)}:
            raise ValueError(
                f"The prompt states a batch fan-out of {sorted(stated_width)} prompts per "
                f"batch, but S6 enforces max_batch_width={max_batch_width}."
            )


def check_runtime_policy(runtime_policy: dict[str, Any]) -> None:
    """S6 holds the harness's numbers and switches, and nothing else.

    Raises:
        ValueError: If the policy carries a backend-routing key, an experiment-owned
            cap or callback, or a field S6 does not define.
    """
    routing = sorted(set(runtime_policy) & BACKEND_ROUTING_KEYS)
    if routing:
        raise ValueError(
            f"S6 may not declare {routing}: depth routing silently swaps the model used "
            "for sub-calls, which would confound every measured harness effect."
        )
    owned = sorted(set(runtime_policy) & EXPERIMENT_OWNED_KEYS)
    if owned:
        raise ValueError(
            f"S6 may not declare {owned}: tracing callbacks and hard token/budget/timeout "
            "caps are experiment-owned, not harness-owned."
        )
    unknown = sorted(set(runtime_policy) - POLICY_FIELDS)
    if unknown:
        raise ValueError(
            f"S6 policy carries unknown field(s) {unknown}; S6's fields are "
            f"{sorted(POLICY_FIELDS)}."
        )


def check_plumbing(harness: Harness) -> None:
    """The sub-call path and the answer-from-variable path may not be shadowed by S8.

    Raises:
        ValueError: If either helper dict binds a reserved REPL name. The
            environment's own collision check is a backstop; the harness is
            rejected before a client is ever constructed.
    """
    for label, helpers in (
        ("repl_helpers", harness.repl_helpers),
        ("sub_repl_helpers", harness.sub_repl_helpers),
    ):
        shadowed = sorted(set(helpers) & set(REQUIRED_REPL_PLUMBING))
        if shadowed:
            raise ValueError(
                f"S8 {label} may not shadow {shadowed}: the programmatic sub-call path "
                "and the answer-from-variable return path are always injected, so I2 and "
                "I3 cannot be removed through a surface."
            )
        validate_custom_tools(helpers)


def check_answer_middleware(middleware: AnswerMiddlewareFn) -> None:
    """S9 may inspect the answer and redirect; it may not rewrite the root's program.

    The mechanical half of KTD1 condition 1 is the input and the return type: two
    declared inputs (the detected answer and the redacted inventory) and an
    ``AnswerDecision``, which admits only "unchanged" or "redirect". The probe
    calls the middleware once, so a stateful middleware must tolerate that.

    Raises:
        ValueError: If the middleware declares anything other than two positional
            inputs.
        TypeError: If it returns anything other than an ``AnswerDecision``.
    """
    parameters = list(inspect.signature(middleware).parameters.values())
    positional = [
        p
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = [
        p
        for p in parameters
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(positional) != 2 or variadic:
        raise ValueError(
            "S9 middleware must declare exactly two inputs, (answer, repl_inventory); "
            f"{getattr(middleware, '__name__', middleware)!r} declares "
            f"{[p.name for p in parameters]}. Anything more is a channel into the root's "
            "program."
        )
    decision = middleware(PROBE_ANSWER, dict(probe_repl_state(1)[1]))
    if not isinstance(decision, AnswerDecision):
        raise TypeError(
            "S9 middleware must return an AnswerDecision (accept or redirect); "
            f"{getattr(middleware, '__name__', middleware)!r} returned "
            f"{type(decision).__name__}."
        )


def check_harness(harness: Harness) -> None:
    """Run every structural check against ``harness``. Raises on the first failure."""
    check_runtime_policy(harness.runtime_policy)
    check_metadata_boundedness(harness.metadata)
    check_stated_limits(harness)
    check_plumbing(harness)
    check_answer_middleware(harness.answer_middleware)


# ---------------------------------------------------------------------------
# S10 scaffold: the skill loader
# ---------------------------------------------------------------------------


class UnknownSkillError(LookupError):
    """``load_skill`` was asked for a name the harness's skill index does not carry."""


def build_skill_loader(skills: list[SkillEntry]) -> SkillLoader:
    """Build the fixed ``load_skill(name) -> str`` loader over a harness's S10 entries (KTD9).

    The loader is runtime scaffold, not surface content: the runner installs it
    under ``SKILL_LOADER_NAME`` in both the root ``custom_tools`` and the child
    ``custom_sub_tools`` whenever ``Harness.skills`` is non-empty, and never
    otherwise. It is never serialized into the S8 helper dicts (``serialize_harness``
    does not see it, so installing it moves no hash), never proposable, and its
    name is harness-reserved against S8.

    Hand-off contract: a skill body reaches a sub-call only when the root
    interpolates it into the sub-call prompt -- ``llm_query`` / ``rlm_query``
    accept any string, and no further pass-through exists. An ``rlm_query`` child
    below ``max_depth`` is a full RLM with the same system prompt (so it sees the
    index) and the same loader (installed from ``custom_sub_tools``), so it may
    call ``load_skill`` itself; an ``llm_query`` sub-call and a max-depth leaf are
    bare completions with neither. Docker/Daytona-style isolated environments skip
    host callables, so the loader is not installed there; the local REPL is the
    environment the experiment runs.

    Every successful load is recorded by the local REPL environment as a trace
    fact (skill name, environment depth) beside its sub-call events -- see
    ``rlm.environments.base_env.SkillLoader``.

    Args:
        skills: The harness's S10 entries, in index order.

    Returns:
        A ``SkillLoader`` whose call returns the named body verbatim -- braces
        and all; bodies never enter the format slot -- and raises
        ``UnknownSkillError`` naming the available skills for an unknown name.
    """
    bodies = {entry.name: entry.body for entry in skills}
    index = [{"name": entry.name, "description": entry.description} for entry in skills]

    def load_skill(name: str) -> str:
        """Return the full procedure of the named skill, verbatim.

        A loaded procedure reaches a sub-call only as text you put in that
        sub-call's prompt (``llm_query`` / ``rlm_query`` accept any string);
        ``rlm_query`` children can also call ``load_skill`` themselves.

        Raises:
            UnknownSkillError: For a name not in the skill index; the message
                lists the available names.
        """
        try:
            return bodies[name]
        except KeyError:
            available = ", ".join(repr(known) for known in bodies) or "(none)"
            raise UnknownSkillError(
                f"unknown skill {name!r}; available skills: {available}"
            ) from None

    return SkillLoader(load=load_skill, index=index)


def _install_skill_loader(helpers: dict[str, Any], loader: Callable[[str], str]) -> dict[str, Any]:
    """A copy of one S8 helper dict with the loader merged in last, under its reserved name.

    The harness's own dict is never mutated: the loader is not surface content
    and must not leak into anything that serializes the harness.
    """
    merged = dict(helpers)
    merged[SKILL_LOADER_NAME] = {"tool": loader, "description": SKILL_LOADER_DESCRIPTION}
    return merged


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessRun:
    """One completion under one harness, with the trace metrics it produced."""

    harness: Harness
    completion: RLMChatCompletion
    metrics: dict[str, Any]


@dataclass(frozen=True)
class HarnessedRLM:
    """A constructed RLM plus the derived values the structural checks pinned down."""

    harness: Harness
    rlm: RLM
    logger: RLMLogger
    system_prompt: str
    metadata_bound: int
    capacity_sentence: str

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> HarnessRun:
        """Run one completion and aggregate its trace metrics."""
        completion = self.rlm.completion(prompt, root_prompt=root_prompt)
        return HarnessRun(
            harness=self.harness,
            completion=completion,
            metrics=run_metrics(completion),
        )


def build_harnessed_rlm(
    harness: Harness,
    *,
    backend: str = "openai",
    backend_kwargs: dict[str, Any] | None = None,
    log_dir: str | None = None,
    **rlm_kwargs: Any,
) -> HarnessedRLM:
    """Assemble a harnessed RLM, enforcing every structural invariant first.

    Args:
        harness: The ten-surface assignment to run.
        backend: The client backend, experiment-owned.
        backend_kwargs: The client kwargs, experiment-owned.
        log_dir: Optional directory for the JSONL trajectory. The in-memory
            trajectory is always captured -- the behavioral tier is unreadable
            without it.
        **rlm_kwargs: Remaining experiment-owned ``RLM`` kwargs (``max_iterations``,
            ``max_budget``, and so on).

    Returns:
        The constructed ``HarnessedRLM``.

    Raises:
        ValueError: If any structural check fails, if a caller passes a
            harness-owned kwarg, or if ``max_depth`` is declared by both S6 and the
            caller.
        TypeError: If S9 returns something other than an ``AnswerDecision``.
    """
    check_harness(harness)

    routing = sorted(set(rlm_kwargs) & BACKEND_ROUTING_KEYS)
    if routing:
        raise ValueError(
            f"{routing} must stay unset: depth routing silently swaps the model used for "
            "sub-calls, which would confound every measured harness effect."
        )
    owned = sorted(set(rlm_kwargs) & HARNESS_OWNED_KWARGS)
    if owned:
        raise ValueError(
            f"{owned} come from the harness, not from the caller; edit the surface instead."
        )

    policy = dict(harness.runtime_policy)
    policy_max_depth = policy.get("max_depth") if policy.get("enabled") else None
    if policy_max_depth is not None and "max_depth" in rlm_kwargs:
        raise ValueError(
            "max_depth is declared by both S6 and the caller; S6 owns it once the policy "
            "is enabled."
        )
    if policy_max_depth is not None:
        rlm_kwargs["max_depth"] = int(policy_max_depth)

    system_prompt = assemble_system_prompt(harness)
    capacity_sentence = derive_capacity_sentence(policy)
    logger = RLMLogger(log_dir=log_dir)

    # S8 as the harness declares it, plus -- only when S10 is non-empty -- the
    # fixed skill loader merged in last, for the root and for every child (KTD9,
    # KTD7). An empty S10 installs nothing, so H0/H0* render no tool line and
    # the formatted prompt stays byte-identical to the shipped reference.
    custom_tools: dict[str, Any] = harness.repl_helpers
    custom_sub_tools: dict[str, Any] = harness.sub_repl_helpers
    if harness.skills:
        loader = build_skill_loader(harness.skills)
        custom_tools = _install_skill_loader(harness.repl_helpers, loader)
        custom_sub_tools = _install_skill_loader(harness.sub_repl_helpers, loader)

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        custom_system_prompt=system_prompt,
        orchestrator=harness.orchestrator,
        runtime_policy=policy,
        metadata_builder=harness.metadata,
        answer_middleware=harness.answer_middleware,
        custom_tools=custom_tools,
        custom_sub_tools=custom_sub_tools,
        capacity_sentence=capacity_sentence,
        logger=logger,
        **rlm_kwargs,
    )
    return HarnessedRLM(
        harness=harness,
        rlm=rlm,
        logger=logger,
        system_prompt=effective_system_prompt(harness),
        metadata_bound=declared_metadata_bound(harness.metadata),
        capacity_sentence=capacity_sentence,
    )


# ---------------------------------------------------------------------------
# Behavioral tier: observed, never prevented
# ---------------------------------------------------------------------------


def run_metrics(completion: RLMChatCompletion) -> dict[str, Any]:
    """Aggregate one run's U1 per-turn trace metrics into the lineage record's summary.

    Nothing here is a gate. A run whose ``sub_call_count`` is zero because an S2
    edit told the root to answer directly is reported, not refused: prose is not
    mechanically checkable, and whether the optimization loop blocks on it is a
    question for the loop.

    Args:
        completion: A completion produced under a logger (``build_harnessed_rlm``
            always attaches one).

    Returns:
        The per-run summary, including ``cost`` -- the per-run cost total --
        plus the U4 usage keys ``input_tokens``, ``output_tokens``, and
        ``execution_time`` (all carried by the completion; additive, KTD4).

    Raises:
        ValueError: If the completion carries no trajectory, which means no logger
            was attached and the behavioral tier is blind.
    """
    if completion.metadata is None:
        raise ValueError(
            "Completion carries no trajectory; the behavioral tier needs the logger that "
            "build_harnessed_rlm attaches."
        )
    turns = completion.metadata["iterations"]
    per_turn = [turn["trace_metrics"] for turn in turns if "trace_metrics" in turn]
    return {
        "turns": len(turns),
        "sub_call_count": sum(turn["sub_call_count"] for turn in per_turn),
        # A trace persisted before the skill loader existed carries no count;
        # no loader means no loads, so the honest back-fill is zero (R16).
        "skill_load_count": sum(int(turn.get("skill_load_count", 0)) for turn in per_turn),
        "syntax_error_turns": sum(1 for turn in per_turn if turn["syntax_error"]),
        "truncation_events": sum(1 for turn in per_turn if turn["truncation_event"]),
        "answer_events": [turn["answer_event"] for turn in per_turn if turn["answer_event"]],
        "answer_from_variable": any(turn["answer_event"] == ANSWER_SUBMITTED for turn in per_turn),
        "cost": completion.usage_summary.total_cost,
        "input_tokens": completion.usage_summary.total_input_tokens,
        "output_tokens": completion.usage_summary.total_output_tokens,
        "execution_time": completion.execution_time,
        "per_turn": per_turn,
    }


def acceptance_inputs(baseline: HarnessRun, candidate: HarnessRun) -> dict[str, Any]:
    """The measurable inputs to the cost-aware acceptance gate (R6).

    The gate itself -- pass-rate non-regression AND sub-call/cost within a
    preregistered band -- lives in the optimization loop and is out of scope here.
    This returns only what two logged runs can supply. Pass-rate comes from the
    task grader; ``*_answered`` is the per-run flag it grades.

    Raises:
        ValueError: If either run lacks a cost total. A cost-aware gate with a
            missing cost is a pass-rate gate wearing a costume.
    """
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        if run.metrics["cost"] is None:
            raise ValueError(
                f"{label} run has no cost total; the acceptance gate is cost-aware and "
                "cannot be evaluated from pass-rate alone (R6)."
            )
    return {
        "baseline_cost": baseline.metrics["cost"],
        "candidate_cost": candidate.metrics["cost"],
        "cost_delta": candidate.metrics["cost"] - baseline.metrics["cost"],
        "baseline_sub_calls": baseline.metrics["sub_call_count"],
        "candidate_sub_calls": candidate.metrics["sub_call_count"],
        "sub_call_delta": candidate.metrics["sub_call_count"] - baseline.metrics["sub_call_count"],
        "baseline_answered": baseline.metrics["answer_from_variable"],
        "candidate_answered": candidate.metrics["answer_from_variable"],
    }


__all__ = [
    "HarnessRun",
    "HarnessedRLM",
    "UnknownSkillError",
    "acceptance_inputs",
    "build_harnessed_rlm",
    "build_skill_loader",
    "check_harness",
    "run_metrics",
]
