"""The RLM harness: three frozen invariants, ten editable surfaces, two starting harnesses.

Read this module as a figure. Nothing here is clever and nothing is indirect:
each surface is one builder function, each builder is a pure function of its
declared inputs, and the two starting harnesses are literal tables of what those
builders return.

Layout:
    1. The three invariants — frozen architecture, no surface may edit them.
    2. The ten builders, S1 through S10, in turn-loop order.
    3. ``Harness``, ``SURFACES``, and the two constants ``H0`` and ``H0_STAR``
       (written H0* in the plan; ``*`` is not a legal identifier).

Naming convention, relied on by the runner and by the surface tests: a builder
named ``build_<x>`` produces the ``Harness`` field named ``<x>``.

Declared-bound convention, also relied on by the runner: every S7 builder carries
a ``declared_bound`` attribute stating the largest execution-result size it lets
through to the next turn. The runner derives S1's truncation sentence from it and
probes the builder against it, so an S7 edit cannot leave S1 stating a bound the
runtime does not honor.

Brace convention for the prompt surfaces (S1-S5): every prompt surface's return
value is concatenated into the string that ``rlm.utils.prompts`` passes through
``system_prompt.format(custom_tools_section=...)``. The only replacement field
allowed in that string is ``{custom_tools_section}``; every other brace must be
doubled or the run dies with ``KeyError``/``IndexError`` before the first turn.
Prompt-surface text is therefore stored format-ready, and ``escape_braces``
converts ordinary text (a JSON or dict example, say) into that form. S10's index
fields (``name`` and ``description``) join that string too and follow the same
convention; skill bodies never enter it -- they are returned verbatim by the
harness-installed loader -- so they carry no brace convention at all.
"""

import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rlm.core.types import AnswerDecision
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH, default_metadata_builder
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# 1. The three invariants (frozen architecture)
#
# These are the three flaws of the paper's Algorithm 2, inverted. A harness that
# edits one of them is no longer an RLM, so no surface exposes them.
# ---------------------------------------------------------------------------

# Algorithm 2 Flaw #1: the prompt P is copied into the root history.
I1_PROMPT_AS_VARIABLE = (
    "I1 - Prompt-as-variable: the prompt lives in the environment as a REPL "
    "variable and is never copied into root context."
)
# Algorithm 2 Flaw #3: sub-calls are verbalized single actions rather than code.
I2_PROGRAMMATIC_SUBCALLS = (
    "I2 - Programmatic sub-calls: sub-calls are issued by code, in loops, over "
    "slices of the prompt - not as verbalized single actions."
)
# Algorithm 2 Flaw #2: the final answer is verbalized directly by the root.
I3_OUTPUTS_IN_VARIABLES = (
    "I3 - Outputs-in-variables: the final answer is accumulated in a REPL "
    "variable and returned from it, not verbalized directly by the root."
)

INVARIANTS: tuple[str, str, str] = (
    I1_PROMPT_AS_VARIABLE,
    I2_PROGRAMMATIC_SUBCALLS,
    I3_OUTPUTS_IN_VARIABLES,
)

# ---------------------------------------------------------------------------
# Types and the brace helper
# ---------------------------------------------------------------------------

# S7's signature, matching ``rlm.utils.parsing.MetadataBuilder``.
MetadataFn = Callable[[str, dict[str, tuple[str, int]]], str]
# S9's signature, matching ``RLM(answer_middleware=...)``.
AnswerMiddlewareFn = Callable[[str, dict[str, tuple[str, int]]], AnswerDecision]

CUSTOM_TOOLS_SLOT = "{custom_tools_section}"

# S10's loader. The runner installs ``load_skill(name) -> str`` into both the root
# and child REPL namespaces whenever ``Harness.skills`` is non-empty (KTD9). It is
# harness scaffold, not surface content: never serialized, never proposable, and
# this name is reserved against S8 helpers.
SKILL_LOADER_NAME = "load_skill"

# The fixed wrapper that precedes the S10 index in the assembled prompt. Purely
# declarative by design: it names the loader and states what it returns, nothing
# about when to call it -- "when to consult" belongs to each entry's
# ``description`` (R14), which is proposable surface content; this sentence is not.
SKILL_INDEX_PREAMBLE = (
    f"Skills available in the REPL: `{SKILL_LOADER_NAME}(name: str) -> str` returns "
    "the full procedure of a skill listed below."
)


def escape_braces(text: str) -> str:
    """Make ``text`` safe for ``str.format``, keeping ``{custom_tools_section}`` live.

    Every brace is doubled, then the one legal replacement field is restored. Use
    this on any prompt-surface text that contains a dict or JSON example; without
    it the run raises ``KeyError``/``IndexError`` at prompt assembly, which shows
    up as an infrastructure error rather than as a badly scoring edit.
    """
    doubled = text.replace("{", "{{").replace("}", "}}")
    return doubled.replace("{{custom_tools_section}}", CUSTOM_TOOLS_SLOT)


# ---------------------------------------------------------------------------
# 2. The ten surfaces
# ---------------------------------------------------------------------------

# S1's H0 value: the factual contract and nothing else. Names in the REPL, the
# ``answer`` protocol, one ```repl``` block per turn, print-only stdout, the
# truncation sentence. No strategy, no orchestration advice.
MINIMAL_REPL_CONTRACT = textwrap.dedent(
    """\
    You are a Recursive Language Model (RLM): a language model whose prompt-related information lives in a Python REPL that you drive turn by turn. You will be queried turn-by-turn until you have an answer to the query.

    Available in the REPL:
    - `context`: the information related to the prompt (typically `str` or `list[str]`).
    - `llm_query(prompt: str, model: str | None = None) -> str`: a single sub-LLM completion.
    - `llm_query_batched(prompts: list[str], model=None) -> list[str]`: several sub-LLM completions in parallel; same order out as in.
    - `rlm_query(prompt, model=None)` / `rlm_query_batched(prompts, model=None)`: recursive RLM sub-calls. Fall back to `llm_query` / `llm_query_batched` when recursion is disabled.
    - `SHOW_VARS() -> str`: list every variable currently in the REPL.
    - `answer`: dict initialized to `{{"content": "", "ready": False}}`. To submit, set `answer["content"]` to the final answer and `answer["ready"] = True` inside a ```repl``` block.
    {custom_tools_section}

    Write code in ```repl``` blocks, one block per turn; the REPL persists across turns. The REPL is NOT a Jupyter cell - only `print(...)` output (stdout) is shown back to you between turns; a bare expression on the last line is silently discarded. REPL outputs over ~20K characters are truncated."""
)


def build_repl_contract() -> str:
    """S1 - prompt assembly: the factual contract the root model is handed.

    Names available in the REPL, the ``answer`` protocol, one ```repl``` block
    per turn, print-only stdout, and the truncation sentence. Facts about the
    environment only; strategy belongs to S2-S5.

    Returns:
        Format-ready prompt text whose only replacement field is
        ``{custom_tools_section}``.
    """
    return MINIMAL_REPL_CONTRACT


def build_decomposition_instruction() -> str:
    """S2 - turns 1-2, decomposition: how to probe ``context`` and plan the split.

    Returns:
        Format-ready prompt text, or ``""`` to contribute nothing.
    """
    return (
        "Start by probing `context` to understand what you have, then decide how "
        "the task breaks into steps."
    )


def build_execution_instruction() -> str:
    """S3 - per-turn execution: what to print, when to offload, how to aggregate.

    Returns:
        Format-ready prompt text, or ``""`` to contribute nothing.
    """
    return (
        "On each turn, run one block of code, print what you need to see, and "
        "build the result up in REPL variables."
    )


def build_verification_instruction() -> str:
    """S4 - pre-submission verification: what to check before flipping ``ready``.

    Returns:
        Format-ready prompt text, or ``""`` to contribute nothing.
    """
    return 'Check your candidate answer before setting `answer["ready"] = True`.'


def build_recovery_instruction() -> str:
    """S5 - sub-call failure recovery: what to do with an error or unusable result.

    Returns:
        Format-ready prompt text, or ``""`` to contribute nothing.
    """
    return (
        "If a sub-call errors or returns something you cannot use, adjust the "
        "approach and try again."
    )


def build_runtime_policy() -> dict[str, Any]:
    """S6 - runtime enforcement: every number and switch the runtime obeys.

    The floor disables the whole policy and leaves every field unset, so "turn on
    a limit" is one legible edit against one surface.

    Returns:
        A fresh policy dict for ``RLM(runtime_policy=...)``.
    """
    return {
        "enabled": False,
        "max_prompt_chars": None,
        "max_batch_width": None,
        "max_depth": None,
        "retry_on_syntax_error": None,
        "max_retries": None,
        "validate_sub_output": None,
    }


def build_metadata(
    stdout: str,
    repl_inventory: dict[str, tuple[str, int]],
    max_character_length: int = DEFAULT_MAX_CHARACTER_LENGTH,
) -> str:
    """S7 - turn-to-turn memory: what one executed block leaves behind for the next turn.

    The floor is the shipped 20K head-truncation, byte for byte. ``repl_inventory``
    is the redacted ``{name: (type_name, length)}`` view of the namespace, never
    the values, so I1 holds however this surface is edited.

    Args:
        stdout: The formatted execution result for one code block.
        repl_inventory: The redacted namespace inventory.
        max_character_length: Per-block cap on the formatted execution result.

    Returns:
        The entry that carries into the next turn.
    """
    return default_metadata_builder(stdout, repl_inventory, max_character_length)


# S7's declared-bound convention. Every S7 builder carries ``declared_bound``: the
# largest number of characters of one block's execution result it will ever let
# through to the next turn. The runner reads it to write S1's truncation sentence
# and probes the builder against it, so the stated bound and the honored bound
# cannot drift (see ``shrlm.runner``). A builder without it is rejected at
# construction rather than assumed to be the shipped default.
build_metadata.declared_bound = DEFAULT_MAX_CHARACTER_LENGTH


def build_repl_helpers() -> dict[str, Any]:
    """S8 - REPL namespace construction: what the root REPL gets beyond the defaults.

    Keys must avoid ``rlm.environments.base_env.RESERVED_TOOL_NAMES``; the
    environment's collision check is a backstop, not the harness's design.

    Returns:
        A fresh ``{name: value}`` dict for ``RLM(custom_tools=...)``.
    """
    return {}


def build_sub_repl_helpers() -> dict[str, Any]:
    """S8 - REPL namespace construction: the same, for sub-call environments.

    Returns:
        A fresh ``{name: value}`` dict for ``RLM(custom_sub_tools=...)``.
    """
    return {}


def accept_answer(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
    """The S9 floor: accept every detected answer unchanged (identity)."""
    return AnswerDecision.accept(answer)


def build_answer_middleware() -> AnswerMiddlewareFn:
    """S9 - answer detection: programmatic inspection of the detected answer.

    The middleware runs when the root sets ``answer["ready"] = True``; it either
    accepts the answer (the floor) or redirects, suppressing the answer and
    pushing a nudge back to the model.

    Returns:
        A callable ``(answer, repl_inventory) -> AnswerDecision``.
    """
    return accept_answer


@dataclass(frozen=True)
class SkillEntry:
    """One S10 entry: a named, reusable procedure.

    ``name`` identifies it (unique within the list, a REPL-safe identifier);
    ``description`` is one line stating when to consult it; ``body`` is the
    procedure text, ordered steps. ``name`` and ``description`` are rendered into
    the assembled prompt as the index and are therefore format-ready, brace-free
    text; ``body`` is returned verbatim by the loader and never enters the prompt.
    """

    name: str
    description: str
    body: str


def build_skills() -> list[SkillEntry]:
    """S10 - reusable procedure, available across turns: the skill library.

    The floor carries no skills, so the assembled prompt carries no index and the
    runner installs no loader; every procedure the loop wants the root to have on
    hand must be proposed and promoted into this list.

    Returns:
        A fresh list of ``SkillEntry`` records, empty at the floor.
    """
    return []


# ---------------------------------------------------------------------------
# 3. The registry and the two starting harnesses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One editable position in the RLM turn loop."""

    id: str
    phase: str
    governs: str
    builders: tuple[Callable[..., Any], ...]


SURFACES: dict[str, Surface] = {
    "S1": Surface(
        "S1",
        "prompt assembly",
        "the factual REPL contract",
        (build_repl_contract,),
    ),
    "S2": Surface(
        "S2",
        "turns 1-2, decomposition",
        "probing `context` and planning the decomposition",
        (build_decomposition_instruction,),
    ),
    "S3": Surface(
        "S3",
        "per-turn execution",
        "what to print, when to offload, how to aggregate",
        (build_execution_instruction,),
    ),
    "S4": Surface(
        "S4",
        "pre-submission verification",
        'what to check before flipping `answer["ready"]`',
        (build_verification_instruction,),
    ),
    "S5": Surface(
        "S5",
        "sub-call failure recovery",
        "what to do when a sub-call errors or returns something unusable",
        (build_recovery_instruction,),
    ),
    "S6": Surface(
        "S6",
        "runtime enforcement",
        "every number and switch",
        (build_runtime_policy,),
    ),
    "S7": Surface(
        "S7",
        "turn-to-turn memory",
        "what carries across turns",
        (build_metadata,),
    ),
    "S8": Surface(
        "S8",
        "REPL namespace construction",
        "proposer-written functions and data injected into the REPL; the "
        "harness-installed skill loader is scaffold and belongs to S10",
        (build_repl_helpers, build_sub_repl_helpers),
    ),
    "S9": Surface(
        "S9",
        "answer detection",
        "programmatic inspection of the detected answer, with redirect",
        (build_answer_middleware,),
    ),
    "S10": Surface(
        "S10",
        "reusable procedure, available across turns",
        "the skill library: one named, reusable procedure per entry -- `name` "
        "identifies it, a one-line `description` states when to consult it, and "
        "`body` is its ordered steps; neither description nor body may restate "
        "per-turn execution or decomposition guidance",
        (build_skills,),
    ),
}


@dataclass(frozen=True)
class Harness:
    """One complete assignment of the ten surfaces, plus the orchestrator scalar.

    Field names follow the builder convention: ``build_<x>`` fills ``<x>``.
    """

    name: str
    orchestrator: bool
    # S1-S5: prompt surfaces, concatenated by ``assemble_system_prompt``.
    repl_contract: str
    decomposition_instruction: str
    execution_instruction: str
    verification_instruction: str
    recovery_instruction: str
    # S6-S9: runtime surfaces, handed to ``RLM(...)`` by the runner.
    runtime_policy: dict[str, Any] = field(default_factory=build_runtime_policy)
    metadata: MetadataFn = build_metadata
    repl_helpers: dict[str, Any] = field(default_factory=build_repl_helpers)
    sub_repl_helpers: dict[str, Any] = field(default_factory=build_sub_repl_helpers)
    answer_middleware: AnswerMiddlewareFn = accept_answer
    # S10: the skill library. Only its name/description index reaches the prompt.
    skills: list[SkillEntry] = field(default_factory=build_skills)


def render_skill_index(skills: list[SkillEntry]) -> str:
    """Render S10's prompt contribution: the fixed preamble plus one index line per skill.

    Name and description only -- never the body, which the loader returns at run
    time. An empty list renders ``""`` so it contributes no bytes.

    Args:
        skills: The harness's S10 entries.

    Returns:
        Format-ready text, or ``""`` when there are no skills.
    """
    if not skills:
        return ""
    lines = [SKILL_INDEX_PREAMBLE]
    lines.extend(f"- `{entry.name}`: {entry.description}" for entry in skills)
    return "\n".join(lines)


def assemble_system_prompt(harness: Harness) -> str:
    """Concatenate the harness's prompt surfaces S1-S5, then the S10 index, into one system prompt.

    Empty surfaces contribute nothing, so a harness whose only prompt surface is
    S1 yields S1 byte for byte -- that is what makes H0* identical to the shipped
    reference. The S10 index (preamble plus name/description lines, never bodies)
    is appended after S5 only when the list is non-empty. The result still carries
    the ``{custom_tools_section}`` slot and is formatted by
    ``rlm.utils.prompts.build_rlm_system_prompt``.

    Args:
        harness: The harness whose prompt surfaces to assemble.

    Returns:
        Format-ready system prompt text.
    """
    parts = [
        harness.repl_contract,
        harness.decomposition_instruction,
        harness.execution_instruction,
        harness.verification_instruction,
        harness.recovery_instruction,
        render_skill_index(harness.skills),
    ]
    return "\n\n".join(part for part in parts if part)


# H0 - the mechanism floor and the optimization loop's starting point. Eight of
# the ten surfaces are empty, disabled, or one generic line; every clause H0
# lacks relative to H0_STAR is one the loop must rediscover from its own traces.
H0 = Harness(
    name="H0",
    orchestrator=False,
    repl_contract=build_repl_contract(),
    decomposition_instruction=build_decomposition_instruction(),
    execution_instruction=build_execution_instruction(),
    verification_instruction=build_verification_instruction(),
    recovery_instruction=build_recovery_instruction(),
    runtime_policy=build_runtime_policy(),
    metadata=build_metadata,
    repl_helpers=build_repl_helpers(),
    sub_repl_helpers=build_sub_repl_helpers(),
    answer_middleware=build_answer_middleware(),
    skills=build_skills(),
)

# H0* - the shipped reference harness. S1 is ``RLM_SYSTEM_PROMPT`` verbatim and
# the runtime appends ``ORCHESTRATOR_ADDENDUM`` itself (``orchestrator=True``),
# so S2-S5 are empty: their content arrives through the addendum. S6-S10 are H0's.
H0_STAR = Harness(
    name="H0*",
    orchestrator=True,
    repl_contract=RLM_SYSTEM_PROMPT,
    decomposition_instruction="",
    execution_instruction="",
    verification_instruction="",
    recovery_instruction="",
    runtime_policy=build_runtime_policy(),
    metadata=build_metadata,
    repl_helpers=build_repl_helpers(),
    sub_repl_helpers=build_sub_repl_helpers(),
    answer_middleware=build_answer_middleware(),
    skills=build_skills(),
)

HARNESSES: dict[str, Harness] = {"H0": H0, "H0*": H0_STAR}
