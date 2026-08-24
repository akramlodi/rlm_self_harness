"""
Lambda-RLM (λ-RLM): Deterministic Recursive Language Model.

Replaces the open-ended LLM-driven while loop of the original RLM with a
pre-verified combinator chain (Φ) that guarantees termination and bounded cost.

Algorithm (Algorithm 1: λ-RLM Complete System):
  Phase 1: InitREPL         - same as original RLM; prompt lives in REPL env, not LLM context
  Phase 2: Task Detection   - exactly 1 LLM call (digit menu selection)
  Phase 3: Optimal Planning - 0 LLM calls (pure math: compute k*, τ*, d)
  Phase 4: Cost Estimate    - deterministic formula Ĉ
  Phase 5: Execute Φ        - pre-verified combinators only; guaranteed termination

Contrast with original RLM:
  ORIGINAL RLM:                         λ-RLM:
  ─────────────────────────             ─────────────────────────
  state ← InitREPL(P)         SAME      state ← InitREPL(P)
  state ← AddFunction(sub_M)  SAME      state ← RegisterSubCall(sub_M)
                               NEW      state ← RegisterLibrary(L)
  while True:                  GONE     // no while loop
    code ← LLM(hist)           GONE     // no LLM-generated code
    (state,out) ← REPL(code)   GONE     // no arbitrary execution
    if Final is set: break     GONE     // no uncertain termination
                               NEW      (k*,τ*,⊕,π) ← Plan(...)
                               NEW      state ← REPL("Φ = BuildExecutor(...)")
                               NEW      state ← REPL("result = Φ(P)")
                               NEW      return state[result]

Φ(P) runs entirely inside the REPL as Python (not LLM) code:
  if |P| <= τ*: return sub_M(template(P))         # the ONLY neural call (leaf)
  else:         Split(P,k*) → Filter? → Map(Φ) → Reduce(⊕)
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter

# Deep recursive Φ trees on very large inputs can exceed Python's default limit.
sys.setrecursionlimit(5000)
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import ClientBackend, EnvironmentType, RLMChatCompletion
from rlm.environments.local_repl import LocalREPL
from rlm.logger import RLMLogger


# ─── Task types ───────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    SUMMARIZATION  = "summarization"
    QA             = "qa"
    TRANSLATION    = "translation"
    CLASSIFICATION = "classification"
    EXTRACTION     = "extraction"
    ANALYSIS       = "analysis"
    GENERAL        = "general"


# ─── Composition operators (⊕) ────────────────────────────────────────────────

class ComposeOp(str, Enum):
    CONCATENATE       = "concatenate"        # free: ordered string join
    MERGE_SUMMARIES   = "merge_summaries"    # 1 LLM call: hierarchical merge
    SELECT_RELEVANT   = "select_relevant"    # 1 LLM call: synthesise answers
    MAJORITY_VOTE     = "majority_vote"      # deterministic: frequency count
    MERGE_EXTRACTIONS = "merge_extractions"  # deterministic: deduplication
    COMBINE_ANALYSIS  = "combine_analysis"   # 1 LLM call: combine insights


# CompositionTable[τ_type] → ⊕
COMPOSITION_TABLE: dict[TaskType, ComposeOp] = {
    TaskType.SUMMARIZATION:  ComposeOp.MERGE_SUMMARIES,
    TaskType.QA:             ComposeOp.SELECT_RELEVANT,
    TaskType.TRANSLATION:    ComposeOp.CONCATENATE,
    TaskType.CLASSIFICATION: ComposeOp.MAJORITY_VOTE,
    TaskType.EXTRACTION:     ComposeOp.MERGE_EXTRACTIONS,
    TaskType.ANALYSIS:       ComposeOp.COMBINE_ANALYSIS,
    TaskType.GENERAL:        ComposeOp.MERGE_SUMMARIES,
}

# Relative cost of one ⊕ reduction (c_⊕ in the cost formula)
C_COMPOSE: dict[ComposeOp, float] = {
    ComposeOp.CONCATENATE:       0.01,  # near-free
    ComposeOp.MERGE_SUMMARIES:   2.0,   # one LLM call
    ComposeOp.SELECT_RELEVANT:   1.5,   # one LLM call
    ComposeOp.MAJORITY_VOTE:     0.05,  # near-free
    ComposeOp.MERGE_EXTRACTIONS: 0.05,  # near-free
    ComposeOp.COMBINE_ANALYSIS:  2.0,   # one LLM call
}

C_IN: float = 1.0  # relative cost per input char


# ─── PlanTable[τ_type] → combinator sequence π ────────────────────────────────

@dataclass
class PipelineFlags:
    use_filter: bool = False  # include relevance Filter step before Map


PLAN_TABLE: dict[TaskType, PipelineFlags] = {
    TaskType.SUMMARIZATION:  PipelineFlags(use_filter=False),
    TaskType.QA:             PipelineFlags(use_filter=True),
    TaskType.TRANSLATION:    PipelineFlags(use_filter=False),
    TaskType.CLASSIFICATION: PipelineFlags(use_filter=False),
    TaskType.EXTRACTION:     PipelineFlags(use_filter=True),
    TaskType.ANALYSIS:       PipelineFlags(use_filter=False),
    TaskType.GENERAL:        PipelineFlags(use_filter=False),
}


# ─── Leaf prompt templates (Template[τ_type].Fmt(P)) ──────────────────────────

TASK_TEMPLATES: dict[TaskType, str] = {
    TaskType.SUMMARIZATION:  "Summarize the following text concisely:\n\n{text}",
    TaskType.QA:             "Using the following context, answer: {query}\n\nContext:\n{text}",
    TaskType.TRANSLATION:    "Translate the following text:\n\n{text}",
    TaskType.CLASSIFICATION: "Classify the following text:\n\n{text}",
    TaskType.EXTRACTION:     "Extract all key information from:\n\n{text}",
    TaskType.ANALYSIS:       "Analyze the following text and provide insights:\n\n{text}",
    TaskType.GENERAL:        "Process the following and provide a response:\n\n{text}",
}


# ─── Task detection prompt (Phase 2) ──────────────────────────────────────────
# This is the single menu-selection LLM call (not code generation).

_TASK_DETECTION_PROMPT = """\
Based on the metadata below, select the single most appropriate task type.

Metadata: {metadata}

Reply with ONLY a single digit (no other text):
1. summarization - condense/summarize content
2. qa - answer a question using context
3. translation - translate text
4. classification - categorize/label text
5. extraction - extract specific facts or entities
6. analysis - deep analysis or evaluation
7. general - mixed or other

Single digit:"""

_TASK_DIGIT_MAP: dict[int, TaskType] = {
    1: TaskType.SUMMARIZATION,
    2: TaskType.QA,
    3: TaskType.TRANSLATION,
    4: TaskType.CLASSIFICATION,
    5: TaskType.EXTRACTION,
    6: TaskType.ANALYSIS,
    7: TaskType.GENERAL,
}


# ─── Plan result (Phases 3 & 4) ───────────────────────────────────────────────

@dataclass
class LambdaPlan:
    task_type:     TaskType
    compose_op:    ComposeOp
    pipeline:      PipelineFlags
    k_star:        int    # optimal branching factor k*
    tau_star:      int    # leaf chunk size τ* (chars)
    depth:         int    # recursion depth d = ⌈log_k*(n/K)⌉
    cost_estimate: float  # Ĉ: relative cost estimate
    n:             int    # total input length (chars)


# ─── LambdaRLM ────────────────────────────────────────────────────────────────

class LambdaRLM:
    """
    λ-RLM: Deterministic Recursive Language Model.

    Replaces the open-ended LLM while-loop with a fixed combinator chain Φ:
      Split(P, k*) → Filter? → Map(Φ) → Reduce(⊕)

    Guarantees: termination, bounded cost, no LLM-generated arbitrary code.

    Args:
        backend:               LLM client backend (same options as RLM).
        backend_kwargs:        Backend kwargs, e.g. {"model_name": "gpt-4o"}.
        environment:           REPL env type; only "local" is used by λ-RLM.
        environment_kwargs:    Extra kwargs forwarded to LocalREPL.
        context_window_chars:  Model context window in chars (default 400k ≈ 100k tokens).
        accuracy_target:       Minimum accuracy α for the accuracy constraint (0–1).
        a_leaf:                Estimated single-call accuracy A(K) (default 0.95).
        a_compose:             Estimated per-level composition accuracy A_⊕ (default 0.90).
        query:                 For QA/Extraction tasks: the question string.
        verbose:               Print plan info to stdout.
        logger:                Optional RLMLogger (currently unused, reserved for future).
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        context_window_chars: int = 100_000,
        accuracy_target: float = 0.80,
        a_leaf: float = 0.95,
        a_compose: float = 0.90,
        query: str | None = None,
        verbose: bool = False,
        logger: RLMLogger | None = None,
    ):
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.environment_type = environment
        self.environment_kwargs = environment_kwargs or {}
        self.context_window_chars = context_window_chars
        self.accuracy_target = accuracy_target
        self.a_leaf = a_leaf
        self.a_compose = a_compose
        self.query = query
        self.verbose = verbose
        self.logger = logger

    # ── Public API ────────────────────────────────────────────────────────────

    def completion(self, prompt: str) -> RLMChatCompletion:
        """
        Run λ-RLM on prompt. Returns RLMChatCompletion (drop-in for RLM.completion).

        Phases:
          1. InitREPL    — LocalREPL created; context stored as context_0, NOT in LLM context
          2. Task detect — 1 LLM call: digit menu selection, no code generation
          3. Plan        — 0 LLM calls: k*, τ*, d computed analytically
          4. Cost est.   — deterministic Ĉ formula (logged if verbose)
          5. Execute Φ   — BuildExecutor + Φ(P) run inside REPL; only leaves call LLM
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"LambdaRLM.completion requires a string prompt, got {type(prompt).__name__}"
            )

        # ── Parse the prompt into (context_text, effective_query) ────────────
        # Benchmark prompts have the form:
        #   "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        # We split out the context (what to recursively split) from the question
        # (what to pass to every leaf node as `query`).
        context_text: str = prompt
        effective_query: str = self.query or ""

        if not effective_query:
            # Attempt to extract "Question: ..." from the prompt.
            q_marker = "\nQuestion: "
            q_idx = prompt.rfind(q_marker)
            if q_idx != -1:
                ans_idx = prompt.rfind("\nAnswer:", q_idx)
                q_end = ans_idx if ans_idx > q_idx else len(prompt)
                effective_query = prompt[q_idx + len(q_marker):q_end].strip()
                # Strip boilerplate so context_0 holds ONLY the document text.
                ctx_start = len("Context:\n") if prompt.startswith("Context:\n") else 0
                context_text = prompt[ctx_start:q_idx].strip()

        time_start = time.perf_counter()
        client: BaseLM = get_client(self.backend, self.backend_kwargs)

        # LMHandler: socket server that routes llm_query() calls from inside the REPL
        with LMHandler(client) as handler:

            # ── Phase 1: InitREPL ─────────────────────────────────────────────
            # Context lives in the REPL environment (context_0), NOT in LLM context.
            env_kwargs: dict[str, Any] = {
                "lm_handler_address": handler.address,
                "context_payload": context_text,  # → stored as context_0 in REPL locals
                "depth": 1,
            }
            env_kwargs.update(self.environment_kwargs)
            repl = LocalREPL(**env_kwargs)

            try:
                # ── Phase 2: Task Detection (1 LLM call) ─────────────────────
                repl.execute_code("lrlm_peek = context_0[:500]\nlrlm_n = len(context_0)")
                peek_text = str(repl.locals.get("lrlm_peek", context_text[:500]))
                n = int(repl.locals.get("lrlm_n", len(context_text)))

                metadata = (
                    f"length={n}, "
                    f"query={repr(effective_query[:100])}, "
                    f"preview={repr(peek_text[:150])}"
                )
                raw = client.completion(
                    _TASK_DETECTION_PROMPT.format(metadata=metadata)
                ).strip()
                task_type = self._parse_task_type(raw)

                if self.verbose:
                    print(f"[λ-RLM] n={n:,} chars  |  task={task_type.value}  |  q={effective_query[:60]!r}")

                # ── Phase 3: Optimal Planning (0 LLM calls, pure math) ────────
                plan = self._plan(task_type, n)

                if self.verbose:
                    print(
                        f"[λ-RLM] k*={plan.k_star}  τ*={plan.tau_star}  "
                        f"d={plan.depth}  ⊕={plan.compose_op.value}"
                    )
                    print(f"[λ-RLM] Ĉ ≈ {plan.cost_estimate:.1f} (relative units)")

                # ── Phase 5: Execute Combinator Chain Φ in REPL ──────────────
                self._register_library(repl, plan, effective_query)
                phi_code = self._build_phi_code(plan, effective_query)
                phi_result = repl.execute_code(phi_code)

                if phi_result.stderr:
                    raise RuntimeError(f"λ-RLM Φ execution error:\n{phi_result.stderr}")

                result: str = str(repl.locals.get("lambda_rlm_result", "")).strip()
                if not result:
                    result = phi_result.stdout.strip() or "No result produced."

                time_end = time.perf_counter()
                usage = client.get_usage_summary()

                return RLMChatCompletion(
                    root_model=self.backend_kwargs.get("model_name", "unknown"),
                    prompt=prompt,
                    response=result,
                    usage_summary=usage,
                    execution_time=time_end - time_start,
                )

            finally:
                repl.cleanup()

    # ── Internal: Phase 2 helper ──────────────────────────────────────────────

    def _parse_task_type(self, response: str) -> TaskType:
        """Extract first digit from LLM response and map to TaskType."""
        for ch in response:
            if ch.isdigit():
                return _TASK_DIGIT_MAP.get(int(ch), TaskType.GENERAL)
        return TaskType.GENERAL

    # ── Internal: Phase 3 — Optimal Planning ─────────────────────────────────

    def _plan(self, task_type: TaskType, n: int) -> LambdaPlan:
        """
        Phase 3: compute k*, τ*, d analytically (0 LLM calls).

        Minimises T(n) = k·T(n/k) + C_⊕(k):
            k* = ⌈√(n · c_in / c_⊕)⌉

        Subject to accuracy constraint A(K)^d · A_⊕^d ≥ α:
            if unsatisfied, increment k* (reducing d) until satisfied or k* hits ceiling.

        τ* = min(K, ⌊n/k*⌋)
        Ĉ  = k*^d · C(τ*) + d · C_⊕(k*) + C(500)  [probe cost]
        """
        K = self.context_window_chars
        compose_op = COMPOSITION_TABLE[task_type]
        pipeline   = PLAN_TABLE[task_type]
        c_compose  = C_COMPOSE[compose_op]

        # Fits in one context window — no splitting.
        if n <= K:
            return LambdaPlan(
                task_type=task_type,
                compose_op=compose_op,
                pipeline=pipeline,
                k_star=1,
                tau_star=n,
                depth=0,
                cost_estimate=C_IN * n + C_IN * 500,
                n=n,
            )

        # k* = ⌈√(n · c_in / c_⊕)⌉  — minimises T(n) = k·T(n/k) + C_⊕(k)
        # Capped at 20 to avoid exponential leaf-call blowup on very large inputs.
        _K_STAR_MAX = 20
        if c_compose > 0.1:
            k_star = min(_K_STAR_MAX, max(2, math.ceil(math.sqrt(n * C_IN / c_compose))))
        else:
            # Near-free composition: flat fan-out, split into K-sized chunks.
            k_star = min(_K_STAR_MAX, max(2, math.ceil(n / K)))

        # d = ⌈log_{k*}(n/K)⌉
        d = max(1, math.ceil(math.log(n / K) / math.log(k_star)))

        # Accuracy constraint: while A(K)^d · A_⊕^d < α, increase k* (reduces d).
        max_k = max(2, n // K)
        while (
            (self.a_leaf ** d) * (self.a_compose ** d) < self.accuracy_target
            and k_star < max_k
        ):
            k_star += 1
            d = max(1, math.ceil(math.log(n / K) / math.log(k_star)))

        # τ* = min(K, ⌊n/k*⌋)
        tau_star = min(K, max(1, n // k_star))

        # Ĉ = k*^d · C(τ*) + d · C_⊕(k*) + C(500)
        cost_estimate = (
            (k_star ** d) * C_IN * tau_star
            + d * c_compose * k_star
            + C_IN * 500
        )

        return LambdaPlan(
            task_type=task_type,
            compose_op=compose_op,
            pipeline=pipeline,
            k_star=k_star,
            tau_star=tau_star,
            depth=d,
            cost_estimate=cost_estimate,
            n=n,
        )

    # ── Internal: Phase 5a — RegisterLibrary ─────────────────────────────────

    def _register_library(self, repl: LocalREPL, plan: LambdaPlan, query: str = "") -> None:
        """
        RegisterLibrary: inject pre-verified combinators L into repl.globals.

        Registered names: _Split, _Peek, _Reduce, _FilterRelevant.
        These are pure Python — no arbitrary LLM code.
        _Reduce may call repl._llm_query for merge-type operators (each call is
        a single bounded LLM invocation, not an open-ended loop).

        Note: these names start with _ so they are NOT saved back to repl.locals
        after execute_code — they live only in repl.globals for the lifetime
        of this REPL session.
        """
        # ── Peek(P, start, length) ────────────────────────────────────────────
        def _peek(text: str, start: int, length: int) -> str:
            return text[start: start + length]

        # ── Split(P, k) → [P₁, ..., P_k]  (word-boundary aware) ─────────────
        def _split(text: str, k: int) -> list[str]:
            if k <= 1:
                return [text]
            n = len(text)
            chunk_size = max(1, n // k)
            chunks: list[str] = []
            start = 0
            for i in range(k):
                if start >= n:
                    break
                if i == k - 1:
                    chunks.append(text[start:])
                    break
                end = start + chunk_size
                # Snap to nearest word boundary within ±20% of chunk_size.
                if end < n:
                    margin = max(1, chunk_size // 5)
                    boundary = text.rfind(" ", max(start, end - margin), min(n, end + margin))
                    if boundary > start:
                        end = boundary + 1
                chunks.append(text[start:end])
                start = end
            return [c for c in chunks if c]

        # ── Reduce(⊕, [R₁…R_k']) ─────────────────────────────────────────────
        # Direct reference to _llm_query avoids an extra dict lookup.
        # _llm_query sends a socket request to LMHandler → single LLM call.
        _llm = repl._llm_query
        compose_op = plan.compose_op

        if compose_op == ComposeOp.CONCATENATE:
            def _reduce(parts: list[str]) -> str:
                return "\n\n".join(parts)

        elif compose_op == ComposeOp.MERGE_SUMMARIES:
            def _reduce(parts: list[str]) -> str:
                if len(parts) == 1:
                    return parts[0]
                merged = "\n\n---\n\n".join(parts)
                return _llm(
                    "Merge these partial summaries into one concise, coherent summary. "
                    "Preserve all key facts and findings:\n\n" + merged
                )

        elif compose_op == ComposeOp.SELECT_RELEVANT:
            def _reduce(parts: list[str]) -> str:
                candidates = [
                    p for p in parts
                    if p.strip()
                    and "not found" not in p.lower()
                    and "no information" not in p.lower()
                    and "not mentioned" not in p.lower()
                ] or parts
                if len(candidates) == 1:
                    return candidates[0]
                merged = "\n\n---\n\n".join(candidates)
                return _llm(
                    f"Question: {query}\n\n"
                    "Synthesise these partial answers into one complete, accurate answer:\n\n"
                    + merged
                )

        elif compose_op == ComposeOp.MAJORITY_VOTE:
            def _reduce(parts: list[str]) -> str:
                if not parts:
                    return ""
                normalized = [p.strip().lower() for p in parts]
                winner = Counter(normalized).most_common(1)[0][0]
                for p in parts:
                    if p.strip().lower() == winner:
                        return p.strip()
                return parts[0]

        elif compose_op == ComposeOp.MERGE_EXTRACTIONS:
            def _reduce(parts: list[str]) -> str:
                seen: set[str] = set()
                lines: list[str] = []
                for part in parts:
                    for line in part.splitlines():
                        line = line.strip()
                        if line and line not in seen:
                            seen.add(line)
                            lines.append(line)
                return "\n".join(lines)

        elif compose_op == ComposeOp.COMBINE_ANALYSIS:
            def _reduce(parts: list[str]) -> str:
                if len(parts) == 1:
                    return parts[0]
                merged = "\n\n---\n\n".join(parts)
                return _llm(
                    "Combine these partial analyses into one comprehensive, "
                    "well-structured analysis:\n\n" + merged
                )

        else:
            def _reduce(parts: list[str]) -> str:
                return "\n\n".join(parts)

        # ── FilterRelevant(query, [(chunk, preview)]) → [chunk] ──────────────
        # Used only when pipeline.use_filter=True (QA, Extraction).
        # Each call is one bounded LLM invocation (YES/NO), not a loop.
        def _filter_relevant(query_text: str, items: list[tuple[str, str]]) -> list[str]:
            kept: list[str] = []
            for chunk, preview in items:
                resp = _llm(
                    f"Question: {query_text}\n\n"
                    "Does this excerpt contain information relevant to answering the question?\n"
                    "Reply YES or NO only.\n\nExcerpt:\n" + preview
                ).strip().upper()
                if resp.startswith("Y"):
                    kept.append(chunk)
            # Fallback: if nothing passes the filter, retain all chunks.
            return kept if kept else [chunk for chunk, _ in items]

        # Inject into repl.globals (accessible in all execute_code calls on this REPL).
        # Names start with _ → not persisted back to repl.locals, but always in globals.
        repl.globals["_Split"]          = _split
        repl.globals["_Peek"]           = _peek
        repl.globals["_Reduce"]         = _reduce
        repl.globals["_FilterRelevant"] = _filter_relevant

    # ── Internal: Phase 5b — BuildExecutor + execute ─────────────────────────

    def _build_phi_code(self, plan: LambdaPlan, query: str = "") -> str:
        """
        Build the Φ executor as a Python code string for execute_code().

        Φ is defined as a recursive Python function (not LLM-generated code).
        It references only combinators already registered in repl.globals:
          _Split, _Peek, _Reduce, _FilterRelevant  (from _register_library)
          llm_query                                 (auto-registered by LocalREPL)
          context_0                                 (from LocalREPL.load_context)

        Result is stored in `lambda_rlm_result` (no _ prefix) so LocalREPL's
        execute_code saves it back to repl.locals after exec() completes.

        Correctness notes:
        - Functions defined inside exec(code, combined, combined) use `combined`
          as __globals__, so recursive _Phi calls and all combinator lookups work.
        - _chunks, _raw, _pairs are function-local variables; no globals needed.
        - FINAL_VAR is intentionally NOT called here; result is read directly
          from repl.locals["lambda_rlm_result"] after execute_code returns.
        """
        tau  = plan.tau_star
        k    = plan.k_star
        peek_len = max(50, tau // 10)
        template = TASK_TEMPLATES[plan.task_type]

        # Leaf call: sub_M(Template[τ_type].Fmt(P)) — the ONLY neural call in Φ.
        if plan.task_type == TaskType.QA and query:
            leaf_expr = (
                f"llm_query({repr(template)}.format(text=P, query={repr(query)}))"
            )
        elif plan.task_type == TaskType.QA:
            # No query available — use a generic QA prompt without {query} placeholder.
            fallback_tpl = "Answer based on the following context:\n\n{text}"
            leaf_expr = f"llm_query({repr(fallback_tpl)}.format(text=P))"
        else:
            leaf_expr = f"llm_query({repr(template)}.format(text=P))"

        # Split + optional filter (inside the else branch of _Phi).
        # Indented 8 spaces = inside `else:` block of `def _Phi(P):`.
        if plan.pipeline.use_filter and query:
            split_block = (
                f"        _raw = _Split(P, {k})\n"
                f"        _pairs = [(_raw[i], _Peek(_raw[i], 0, {peek_len}))"
                f" for i in range(len(_raw))]\n"
                f"        _chunks = _FilterRelevant({repr(query)}, _pairs)\n"
            )
        else:
            split_block = f"        _chunks = _Split(P, {k})\n"

        # Full Φ code.  lambda_rlm_result (no _ prefix) is saved to repl.locals.
        return (
            f"def _Phi(P):\n"
            f"    if len(P) <= {tau}:\n"
            f"        return {leaf_expr}\n"
            f"    else:\n"
            f"{split_block}"
            f"        return _Reduce([_Phi(c) for c in _chunks])\n"
            f"\n"
            f"lambda_rlm_result = _Phi(context_0)\n"
        )
