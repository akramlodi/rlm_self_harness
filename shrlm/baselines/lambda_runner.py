"""Persist-first execution for the pinned λ-RLM comparison baseline.

The optimization driver runs editable ``Harness`` objects and identifies them
with ``harness.json``. λ-RLM is a different inference method, not an RLM
harness surface, so this runner keeps its construction separate while sharing
the round's canonical instance, trace, and manifest formats.
"""

import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.clients.base_lm import BaseLM
from rlm.core.llm_observation import (
    ACTIVE_RECORDER,
    ObservationPersistenceError,
    observation_session,
    observe_call,
)
from rlm.core.types import (
    ClientBackend,
    ModelUsageSummary,
    RLMChatCompletion,
    UsageSummary,
)
from rlm.utils.exceptions import BudgetExceededError, TimeoutExceededError
from shrlm.baselines.lambda_rlm import (
    PAPER_RECONSTRUCTION_VERSION,
    LambdaBaselineConfig,
    lambda_input,
    lambda_method_envelope,
    write_lambda_method_json,
)
from shrlm.baselines.paper_lambda_rlm import ClassificationRejectedError, PaperLambdaRLM
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN, round_dir
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    GovernedRoundResult,
    HardDeadlineExceeded,
    call_with_hard_deadline,
    hard_deadline_seconds,
)
from shrlm.optimization.driver import (
    INSTANCES_FILE,
    TRACES_DIR,
    RoundPersistenceError,
    instance_lines,
    load_manifest,
    persist_run,
    run_id_for,
    verify_trace,
)
from shrlm.optimization.llm_observation_store import discover_observations, observation_recorder
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict, Verifier

METHOD_FILE = "method.json"
LAMBDA_SUBCALL_AUDIT_FORMAT = "shrlm-lambda-subcall-audit/v1"
SUBCALL_PROMPT_PREVIEW_CHARS = 500

# These values intentionally match the core round driver's safety policy.
# Backend kwargs are copied into model traces, so credentials must stay in the
# environment rather than becoming persisted experiment artifacts.
SENSITIVE_KWARG_FRAGMENTS = ("key", "token", "secret", "password", "authorization")
BACKEND_ENV_KEYS: dict[str, str] = {"openrouter": "OPENROUTER_API_KEY"}


def lambda_prompt_text(prompt: str | dict[str, Any]) -> str:
    """Return a stable text representation for one audited model request."""
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False)


def lambda_subcall_phase(prompt: str | dict[str, Any]) -> str:
    """Classify a pinned λ-RLM request without changing its execution."""
    text = lambda_prompt_text(prompt)
    if text.startswith("Based on the metadata below, select the single most appropriate task"):
        return "task_detection"
    if text.startswith("OBLIQ_LAMBDA_MAP"):
        return "obliq_map_rank"
    if text.startswith("OBLIQ_LAMBDA_REDUCE"):
        return "obliq_reduce_rank"
    if "Does this excerpt contain information relevant to answering the question?" in text:
        return "relevance_filter"
    if "Synthesise these partial answers into one complete, accurate answer:" in text:
        return "reduce_select_relevant"
    if text.startswith("Using the following context, answer:"):
        return "leaf_qa"
    if text.startswith("Answer based on the following context:"):
        return "leaf_qa"
    if text.startswith("Summarize the following text concisely:"):
        return "leaf_summarization"
    if text.startswith("Translate the following text:"):
        return "leaf_translation"
    if text.startswith("Classify the following text:"):
        return "leaf_classification"
    if text.startswith("Extract all key information from:"):
        return "leaf_extraction"
    if text.startswith("Analyze the following text and provide insights:"):
        return "leaf_analysis"
    if text.startswith("Process the following and provide a response:"):
        return "leaf_general"
    if text.startswith("Merge these partial summaries into one concise, coherent summary."):
        return "reduce_summaries"
    if text.startswith("Combine these partial analyses into one comprehensive"):
        return "reduce_analysis"
    if text.startswith("Classify the expected ANSWER TYPE"):
        return "pairwise_classification"
    return "unknown"


def lambda_filter_decision(phase: str, response: str) -> str | None:
    """Record the upstream filter's raw decision before its retain-all fallback."""
    if phase != "relevance_filter":
        return None
    return "yes" if response.strip().upper().startswith("Y") else "no"


@dataclass(frozen=True)
class LambdaSubcallAudit:
    """One observed model request made by the pinned λ-RLM implementation."""

    sequence: int
    mode: str
    phase: str
    prompt_chars: int
    prompt_preview: str
    response: str
    response_chars: int
    filter_decision: str | None
    usage_after: dict[str, Any]
    error: str | None


class BudgetGuardClient(BaseLM):
    """A transparent client proxy that enforces one λ-RLM run's cost cap.

    The check happens after each completed model call, matching the core RLM's
    post-call budget semantics: the crossing call is paid and reported, while
    every later call is refused before reaching the backend. ``budget_error``
    retains the crossing exception because λ-RLM leaf calls pass through a
    socket handler that may surface it as a generic runtime error.
    """

    def __init__(self, delegate: BaseLM, max_budget: float | None):
        if max_budget is not None and (
            isinstance(max_budget, bool)
            or not isinstance(max_budget, int | float)
            or max_budget <= 0
        ):
            raise ValueError("max_budget must be a positive number or None")
        super().__init__(
            model_name=delegate.model_name,
            timeout=delegate.timeout,
            sampling_args=delegate.sampling_args,
        )
        self.observation_recorder = ACTIVE_RECORDER.get()
        self.delegate = delegate
        self.max_budget = None if max_budget is None else float(max_budget)
        self.spent: float | None = None
        self.budget_error: BudgetExceededError | None = None
        self.subcall_audits: list[LambdaSubcallAudit] = []
        self.audit_lock = threading.Lock()
        self.next_sequence = 1

    def reserve_sequence(self) -> int:
        """Allocate a stable call-start sequence across sync and async requests."""
        with self.audit_lock:
            sequence = self.next_sequence
            self.next_sequence += 1
        return sequence

    def record_subcall(
        self,
        *,
        sequence: int,
        mode: str,
        prompt: str | dict[str, Any],
        response: str = "",
        error: Exception | None = None,
    ) -> None:
        """Record one completed or failed delegated call for later persistence."""
        prompt_text = lambda_prompt_text(prompt)
        phase = lambda_subcall_phase(prompt)
        audit = LambdaSubcallAudit(
            sequence=sequence,
            mode=mode,
            phase=phase,
            prompt_chars=len(prompt_text),
            prompt_preview=prompt_text[:SUBCALL_PROMPT_PREVIEW_CHARS],
            response=response,
            response_chars=len(response),
            filter_decision=lambda_filter_decision(phase, response),
            usage_after=self.delegate.get_usage_summary().to_dict(),
            error=None if error is None else f"{type(error).__name__}: {error}",
        )
        with self.audit_lock:
            self.subcall_audits.append(audit)

    def subcall_audit_dict(self) -> dict[str, Any]:
        """Return all observed calls in invocation order with compact diagnostics."""
        with self.audit_lock:
            calls = sorted(self.subcall_audits, key=lambda audit: audit.sequence)
        phase_counts = Counter(audit.phase for audit in calls)
        filter_counts = Counter(
            audit.filter_decision for audit in calls if audit.filter_decision is not None
        )
        return {
            "format": LAMBDA_SUBCALL_AUDIT_FORMAT,
            "calls_observed": len(calls),
            "phase_counts": dict(sorted(phase_counts.items())),
            "filter_decisions": dict(sorted(filter_counts.items())),
            "calls": [asdict(audit) for audit in calls],
        }

    def enforce_budget(self) -> None:
        """Refresh cumulative spend and raise once it exceeds the configured cap."""
        if self.budget_error is not None:
            raise self.budget_error

        self.spent = self.delegate.get_usage_summary().total_cost
        if self.max_budget is not None and self.spent is not None and self.spent > self.max_budget:
            self.budget_error = BudgetExceededError(
                spent=self.spent,
                budget=self.max_budget,
            )
            raise self.budget_error

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self.enforce_budget()
        sequence = self.reserve_sequence()
        try:
            with observe_call("lambda", self.model_name, recorder=self.observation_recorder):
                response = self.delegate.completion(prompt)
        except Exception as error:
            self.record_subcall(
                sequence=sequence,
                mode="sync",
                prompt=prompt,
                error=error,
            )
            raise
        self.record_subcall(
            sequence=sequence,
            mode="sync",
            prompt=prompt,
            response=response,
        )
        self.enforce_budget()
        return response

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        self.enforce_budget()
        sequence = self.reserve_sequence()
        try:
            with observe_call("lambda", self.model_name, recorder=self.observation_recorder):
                response = await self.delegate.acompletion(prompt)
        except Exception as error:
            self.record_subcall(
                sequence=sequence,
                mode="async",
                prompt=prompt,
                error=error,
            )
            raise
        self.record_subcall(
            sequence=sequence,
            mode="async",
            prompt=prompt,
            response=response,
        )
        self.enforce_budget()
        return response

    def get_usage_summary(self) -> UsageSummary:
        return self.delegate.get_usage_summary()

    def get_last_usage(self) -> ModelUsageSummary:
        return self.delegate.get_last_usage()


ClientFactory = Callable[[ClientBackend, dict[str, Any] | None], BaseLM]


@dataclass
class LambdaClientGuard:
    """The temporarily installed λ-RLM client factory and its one run client."""

    delegate_factory: ClientFactory
    max_budget: float | None
    client: BudgetGuardClient | None = None

    def __call__(
        self,
        backend: ClientBackend,
        backend_kwargs: dict[str, Any] | None,
    ) -> BaseLM:
        if self.client is not None:
            raise RuntimeError(
                "the pinned λ-RLM created more than one client for one completion; "
                "its upstream execution contract changed"
            )
        delegate = self.delegate_factory(backend, backend_kwargs)
        self.client = BudgetGuardClient(delegate, self.max_budget)
        return self.client

    @property
    def budget_error(self) -> BudgetExceededError | None:
        return None if self.client is None else self.client.budget_error

    @property
    def usage_summary(self) -> UsageSummary | None:
        return None if self.client is None else self.client.get_usage_summary()

    @property
    def subcall_audit(self) -> dict[str, Any] | None:
        return None if self.client is None else self.client.subcall_audit_dict()


def attach_lambda_subcall_audit(
    completion: RLMChatCompletion,
    guard: LambdaClientGuard | None,
) -> None:
    """Add wrapper-observed subcalls to a completion without touching upstream code."""
    audit = None if guard is None else guard.subcall_audit
    if audit is None:
        return
    metadata = dict(completion.metadata or {})
    if "lambda_subcall_audit" in metadata:
        raise RuntimeError("λ-RLM completion already contains a lambda_subcall_audit")
    metadata["lambda_subcall_audit"] = audit
    completion.metadata = metadata


@contextmanager
def guarded_lambda_client(max_budget: float | None) -> Iterator[LambdaClientGuard]:
    """Install one budget-guarded upstream client factory for a λ-RLM call.

    The evaluation runner is intentionally single-threaded. Keeping this
    process-global replacement scoped to a context preserves the pinned source
    bytes while ensuring the original factory is restored on every exit path.
    """
    original: ClientFactory = upstream_lambda.get_client
    guard = LambdaClientGuard(original, max_budget)
    upstream_lambda.get_client = guard
    try:
        yield guard
    finally:
        upstream_lambda.get_client = original


@dataclass(frozen=True)
class LambdaRoundConfig:
    """Everything needed to execute and persist one λ-RLM evaluation round."""

    round_index: int
    instances: list[dict[str, Any]]
    verifier: Verifier
    out_dir: Path | str
    method: LambdaBaselineConfig = field(default_factory=LambdaBaselineConfig)
    backend: ClientBackend = "openrouter"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    max_budget: float | None = None
    max_timeout: float | None = None


def validate_lambda_round(config: LambdaRoundConfig) -> None:
    """Reject invalid input before constructing a client or making a model call."""
    if config.attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {config.attempts}")
    if not config.instances:
        raise ValueError("a λ-RLM round needs at least one instance")

    seen: set[str] = set()
    for instance in config.instances:
        instance_id = str(instance["id"])
        if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(instance_id):
            raise ValueError(
                f"instance id {instance_id!r} is not filesystem-safe; ids become trace "
                f"file names and must match {FILESYSTEM_SAFE_ID_PATTERN.pattern}"
            )
        if instance_id in seen:
            raise ValueError(
                f"duplicate instance id {instance_id!r}: run ids derive from "
                "(instance id, attempt); use attempts for repeated runs"
            )
        seen.add(instance_id)
        lambda_input(instance)

    for name in config.backend_kwargs:
        lowered = name.lower()
        if any(fragment in lowered for fragment in SENSITIVE_KWARG_FRAGMENTS):
            raise ValueError(
                f"backend_kwargs may not carry credential material ({name!r}): kwargs "
                "can enter persisted traces; supply credentials through the environment"
            )
    for name in ("max_budget", "max_timeout"):
        value = getattr(config, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number or None")


def prepare_lambda_round(config: LambdaRoundConfig) -> Path:
    """Create or verify the round's method and instance identity artifacts."""
    path = round_dir(config.out_dir, config.round_index)
    path.mkdir(parents=True, exist_ok=True)
    (path / TRACES_DIR).mkdir(exist_ok=True)

    method_path = path / METHOD_FILE
    expected_method = lambda_method_envelope(config.method)
    if method_path.exists():
        recorded_method = json.loads(method_path.read_text())
        if recorded_method != expected_method:
            raise RoundPersistenceError(
                f"{method_path} does not match the configured λ-RLM method; refusing "
                "to mix two method configurations in one round"
            )
    else:
        write_lambda_method_json(config.method, method_path)

    instances_path = path / INSTANCES_FILE
    expected_instances = instance_lines(config.instances)
    if instances_path.exists():
        if instances_path.read_text() != expected_instances:
            raise RoundPersistenceError(
                f"{instances_path} does not match the configured instances; resuming "
                "requires the identical instance list, verbatim"
            )
    else:
        instances_path.write_text(expected_instances)

    return path


def require_lambda_backend_credential(config: LambdaRoundConfig) -> None:
    """Fail before a paid pending run when a known backend credential is absent."""
    env_key = BACKEND_ENV_KEYS.get(config.backend)
    if env_key is not None and not os.environ.get(env_key):
        raise RuntimeError(
            f"backend {config.backend!r} requires the {env_key} environment variable; "
            "refusing to start a paid λ-RLM round"
        )


def lambda_resource_usage(
    guard: LambdaClientGuard | None,
    error: Exception,
) -> UsageSummary:
    """Best observed usage for a λ-RLM call interrupted by a resource limit."""
    observed = None if guard is None else guard.usage_summary
    if observed is not None:
        return observed

    spent = getattr(error, "spent", None)
    if isinstance(spent, int | float) and not isinstance(spent, bool):
        return UsageSummary(
            model_usage_summaries={
                "unknown": ModelUsageSummary(
                    total_calls=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost=float(spent),
                )
            }
        )
    return UsageSummary(model_usage_summaries={})


def lambda_resource_completion(
    config: LambdaRoundConfig,
    prompt: str,
    error: Exception,
    elapsed_seconds: float,
    guard: LambdaClientGuard | None = None,
) -> RLMChatCompletion:
    """Build the auditable partial trace for a resource-terminated λ-RLM run."""
    partial_answer = getattr(error, "partial_answer", None)
    return RLMChatCompletion(
        root_model=str(config.backend_kwargs.get("model_name", "unknown")),
        prompt=prompt,
        response=partial_answer or "",
        usage_summary=lambda_resource_usage(guard, error),
        execution_time=elapsed_seconds,
        error=f"{type(error).__name__}: {error}",
    )


def lambda_resource_verdict(
    completion: RLMChatCompletion,
    error: Exception,
) -> Verdict:
    """The deterministic verdict for a run stopped by an experiment limit."""
    return Verdict(
        passed=False,
        cause=VerifierCause.RESOURCE_TERMINATED,
        gold="",
        produced=completion.response,
        detail=f"{type(error).__name__}: {error}",
    )


def lambda_format_completion(
    config: LambdaRoundConfig,
    prompt: str,
    error: ClassificationRejectedError,
    elapsed_seconds: float,
    guard: LambdaClientGuard,
    method: PaperLambdaRLM,
) -> RLMChatCompletion:
    """Preserve an exhausted classifier repair as an auditable failed run."""
    execution = method.last_pairwise_trace
    usage_summary = guard.usage_summary
    assert usage_summary is not None
    return RLMChatCompletion(
        root_model=str(config.backend_kwargs.get("model_name", "unknown")),
        prompt=prompt,
        response="",
        usage_summary=usage_summary,
        execution_time=elapsed_seconds,
        metadata={
            "pairwise_failure": {
                "reconstruction_version": PAPER_RECONSTRUCTION_VERSION,
                "execution": None if execution is None else asdict(execution),
                "failed_batch": error.audit_dict(),
            }
        },
        error=f"{type(error).__name__}: {error}",
    )


def lambda_format_verdict(
    verifier: Verifier,
    instance: dict[str, Any],
    error: ClassificationRejectedError,
) -> Verdict:
    """Use the environment verifier to build the schema-correct format verdict."""
    base = verifier(instance, "")
    if base.cause is not VerifierCause.WRONG_FORMAT:
        raise RuntimeError(
            "the verifier did not classify an empty response as wrong_format after "
            "an exhausted OOLONG-Pairs classification repair"
        )
    return Verdict(
        passed=False,
        cause=VerifierCause.WRONG_FORMAT,
        gold=base.gold,
        produced=base.produced,
        detail=f"{base.detail}; {type(error).__name__}: {error}",
    )


def run_lambda_round(
    config: LambdaRoundConfig,
    *,
    stop_after: int | None = None,
) -> list[dict[str, Any]]:
    """Run missing λ-RLM attempts and persist each completion immediately.

    Reinvocation with the same configuration verifies every recorded trace and
    skips its run id. This makes a partially completed round resumable without
    paying for completed attempts again.
    """
    validate_lambda_round(config)
    path = prepare_lambda_round(config)

    existing = load_manifest(config.out_dir, config.round_index)
    for entry in existing:
        verify_trace(path, entry)
    done = {str(entry["run_id"]) for entry in existing}
    pending = [
        (instance, attempt)
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if run_id_for(str(instance["id"]), attempt) not in done
    ]

    entries = list(existing)
    if not pending or stop_after == 0:
        return entries
    require_lambda_backend_credential(config)

    executed = 0
    for instance, attempt in pending:
        if stop_after is not None and executed >= stop_after:
            break

        instance_id = str(instance["id"])
        run_id = run_id_for(instance_id, attempt)
        model_input = lambda_input(instance)
        run_started = time.perf_counter()
        guard: LambdaClientGuard | None = None
        method: PaperLambdaRLM | None = None
        usage_lower_bound = False
        recorder = observation_recorder(
            path / TRACES_DIR / f"{run_id}.json",
            {"run_id": run_id, "attempt": attempt, "method": "lambda"},
        )
        try:
            with observation_session(recorder), guarded_lambda_client(config.max_budget) as guard:
                method = config.method.build(
                    backend=config.backend,
                    backend_kwargs=dict(config.backend_kwargs),
                    query=model_input.query,
                    task_id=model_input.task_id,
                )
                completion = method.completion(model_input.prompt)
                if guard.budget_error is not None:
                    raise guard.budget_error
            verdict = config.verifier(instance, completion.response)
        except Exception as caught:
            recorder.check()
            if isinstance(caught, ObservationPersistenceError):
                raise
            resource_error: Exception | None = None
            if guard is not None and guard.budget_error is not None:
                # LMHandler serializes leaf-call exceptions into an error
                # response. Recover the typed exception retained by the guard.
                resource_error = guard.budget_error
            elif isinstance(caught, BudgetExceededError | TimeoutExceededError):
                resource_error = caught
            if resource_error is not None:
                completion = lambda_resource_completion(
                    config,
                    model_input.prompt,
                    resource_error,
                    time.perf_counter() - run_started,
                    guard,
                )
                verdict = lambda_resource_verdict(completion, resource_error)
                usage_lower_bound = True
            elif isinstance(caught, ClassificationRejectedError):
                assert guard is not None
                assert method is not None
                completion = lambda_format_completion(
                    config,
                    model_input.prompt,
                    caught,
                    time.perf_counter() - run_started,
                    guard,
                    method,
                )
                verdict = lambda_format_verdict(config.verifier, instance, caught)
            else:
                raise
        attach_lambda_subcall_audit(completion, guard)
        completion.llm_observations = list(recorder.references)
        entries.append(
            persist_run(
                path,
                run_id,
                instance_id,
                attempt,
                completion,
                verdict,
                usage_lower_bound=usage_lower_bound,
            )
        )
        executed += 1

    return entries


def persist_interrupted_lambda_run(
    config: LambdaRoundConfig,
    error: Exception,
) -> dict[str, Any] | None:
    """Persist the first pending run after a deadline escaped its run slice."""
    validate_lambda_round(config)
    path = prepare_lambda_round(config)
    existing = load_manifest(config.out_dir, config.round_index)
    for entry in existing:
        verify_trace(path, entry)
    done = {str(entry["run_id"]) for entry in existing}

    for instance in config.instances:
        for attempt in range(1, config.attempts + 1):
            instance_id = str(instance["id"])
            run_id = run_id_for(instance_id, attempt)
            if run_id in done:
                continue
            model_input = lambda_input(instance)
            completion = lambda_resource_completion(
                config,
                model_input.prompt,
                error,
                elapsed_seconds=0.0,
            )
            completion.llm_observations = discover_observations(
                path / TRACES_DIR / f"{run_id}.json"
            )
            verdict = lambda_resource_verdict(completion, error)
            return persist_run(
                path,
                run_id,
                instance_id,
                attempt,
                completion,
                verdict,
                usage_lower_bound=True,
            )
    return None


def validate_lambda_governance(
    config: LambdaRoundConfig,
    breaker: CandidateSpendBreaker,
) -> None:
    """Require λ-RLM's per-run limits to match the experiment-owned caps."""
    expected = {
        "max_budget": breaker.caps.max_budget,
        "max_timeout": breaker.caps.max_timeout,
    }
    for name, cap in expected.items():
        value = getattr(config, name)
        if value != cap:
            raise ValueError(
                f"governed λ-RLM requires {name}={cap!r} from ValidationCaps, got {value!r}"
            )


def run_lambda_slice(
    config: LambdaRoundConfig,
    *,
    stop_after: int,
    deadline: float | None,
    known: int,
) -> list[dict[str, Any]]:
    """Execute one resumable λ-RLM slice under the hard deadline."""
    try:
        return call_with_hard_deadline(
            lambda: run_lambda_round(config, stop_after=stop_after),
            deadline,
        )
    except HardDeadlineExceeded as error:
        persisted = load_manifest(config.out_dir, config.round_index)
        if len(persisted) == known:
            entry = persist_interrupted_lambda_run(config, error)
            if entry is not None:
                persisted.append(entry)
        return persisted


def run_governed_lambda_round(
    config: LambdaRoundConfig,
    breaker: CandidateSpendBreaker,
) -> GovernedRoundResult:
    """Execute λ-RLM persist-first, one run at a time, under shared cost caps."""
    validate_lambda_round(config)
    validate_lambda_governance(config, breaker)
    namespace = str(round_dir(config.out_dir, config.round_index))
    deadline = hard_deadline_seconds(config.max_timeout)

    entries = run_lambda_slice(
        config,
        stop_after=0,
        deadline=deadline,
        known=len(load_manifest(config.out_dir, config.round_index)),
    )
    for entry in entries:
        breaker.charge(entry, namespace=namespace)

    while not breaker.tripped:
        known = len(entries)
        entries = run_lambda_slice(
            config,
            stop_after=1,
            deadline=deadline,
            known=known,
        )
        if len(entries) == known:
            break
        for entry in entries[known:]:
            breaker.charge(entry, namespace=namespace)

    done = {str(entry["run_id"]) for entry in entries}
    skipped = [
        run_id
        for instance in config.instances
        for attempt in range(1, config.attempts + 1)
        if (run_id := run_id_for(str(instance["id"]), attempt)) not in done
    ]
    return GovernedRoundResult(
        entries=entries,
        outcome=OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
        spent=breaker.spent,
        skipped_run_ids=skipped,
    )


__all__ = [
    "BudgetGuardClient",
    "LambdaClientGuard",
    "LambdaRoundConfig",
    "METHOD_FILE",
    "guarded_lambda_client",
    "lambda_resource_completion",
    "lambda_resource_usage",
    "lambda_resource_verdict",
    "persist_interrupted_lambda_run",
    "prepare_lambda_round",
    "require_lambda_backend_credential",
    "run_governed_lambda_round",
    "run_lambda_round",
    "run_lambda_slice",
    "validate_lambda_governance",
    "validate_lambda_round",
]
