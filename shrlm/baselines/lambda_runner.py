"""Persist-first execution for the pinned λ-RLM comparison baseline.

The optimization driver runs editable ``Harness`` objects and identifies them
with ``harness.json``. λ-RLM is a different inference method, not an RLM
harness surface, so this runner keeps its construction separate while sharing
the round's canonical instance, trace, and manifest formats.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import shrlm.baselines.upstream.lambda_rlm as upstream_lambda
from rlm.clients.base_lm import BaseLM
from rlm.core.types import (
    ClientBackend,
    ModelUsageSummary,
    RLMChatCompletion,
    UsageSummary,
)
from rlm.utils.exceptions import BudgetExceededError, TimeoutExceededError
from shrlm.baselines.lambda_rlm import (
    LambdaBaselineConfig,
    lambda_input,
    lambda_method_envelope,
    write_lambda_method_json,
)
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
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict, Verifier

METHOD_FILE = "method.json"

# These values intentionally match the core round driver's safety policy.
# Backend kwargs are copied into model traces, so credentials must stay in the
# environment rather than becoming persisted experiment artifacts.
SENSITIVE_KWARG_FRAGMENTS = ("key", "token", "secret", "password", "authorization")
BACKEND_ENV_KEYS: dict[str, str] = {"openrouter": "OPENROUTER_API_KEY"}


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
        self.delegate = delegate
        self.max_budget = None if max_budget is None else float(max_budget)
        self.spent: float | None = None
        self.budget_error: BudgetExceededError | None = None

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
        response = self.delegate.completion(prompt)
        self.enforce_budget()
        return response

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        self.enforce_budget()
        response = await self.delegate.acompletion(prompt)
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
        usage_lower_bound = False
        try:
            with guarded_lambda_client(config.max_budget) as guard:
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
            resource_error: Exception | None = None
            if guard is not None and guard.budget_error is not None:
                # LMHandler serializes leaf-call exceptions into an error
                # response. Recover the typed exception retained by the guard.
                resource_error = guard.budget_error
            elif isinstance(caught, BudgetExceededError | TimeoutExceededError):
                resource_error = caught
            if resource_error is None:
                raise

            completion = lambda_resource_completion(
                config,
                model_input.prompt,
                resource_error,
                time.perf_counter() - run_started,
                guard,
            )
            verdict = lambda_resource_verdict(completion, resource_error)
            usage_lower_bound = True
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
