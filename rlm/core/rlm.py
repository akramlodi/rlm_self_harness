import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ANSWER_REDIRECTED,
    ANSWER_SUBMITTED,
    AnswerDecision,
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.environments import (
    BaseEnv,
    SkillLoader,
    SupportsPersistence,
    extract_tool_value,
    get_environment,
)
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils.exceptions import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.utils.parsing import (
    MetadataBuilder,
    build_repl_inventory,
    find_code_blocks,
    format_iteration,
    has_syntax_error,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys
from rlm.utils.token_utils import count_tokens, get_context_limit


def _skill_index_of(custom_tools: dict[str, Any] | None) -> list[dict[str, str]] | None:
    """The skill index carried by an installed ``SkillLoader`` tool, or None if there is none."""
    for entry in (custom_tools or {}).values():
        value = extract_tool_value(entry)
        if isinstance(value, SkillLoader):
            return [dict(item) for item in value.index]
    return None


def _terminated_child_usage(child: Any, error: Exception, model: str) -> UsageSummary | None:
    """What a child RLM spent before an exception ended it, or None.

    A child that raised returns no completion, so its usage reaches the parent
    only through the total its completion context published on the way out.
    That figure is the accurate one and covers every termination path, so it is
    preferred. ``BudgetExceededError.spent`` is the fallback for a child that
    died before publishing -- a cost with no token breakdown, recorded as a
    zero-call cost-only record rather than dropped.
    """
    published = getattr(child, "last_completion_usage", None)
    if isinstance(published, UsageSummary):
        return published
    spent = getattr(error, "spent", None)
    if isinstance(spent, int | float) and not isinstance(spent, bool):
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost=float(spent),
                )
            }
        )
    return None


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        compaction_threshold_pct: float = 0.85,
        max_concurrent_subcalls: int = 4,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None] | None = None,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
        sampling_args: dict[str, Any] | None = None,
        sub_sampling_args: dict[str, Any] | None = None,
        orchestrator: bool = True,
        user_prologue: str | None = None,
        metadata_builder: MetadataBuilder | None = None,
        answer_middleware: Callable[[str, dict[str, tuple[str, int]]], AnswerDecision]
        | None = None,
        runtime_policy: dict[str, Any] | None = None,
        capacity_sentence: str | None = None,
    ):
        """
        Args:
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of recursion. When depth >= max_depth, falls back to plain LM completion.
            max_iterations: The maximum number of iterations of the RLM.
            max_budget: Maximum budget in USD. Execution stops if exceeded. Requires cost-tracking backend (e.g., OpenRouter).
            max_timeout: Maximum execution time in seconds. Execution stops if exceeded, returning best answer if available.
            max_tokens: Maximum total tokens (input + output). Execution stops if exceeded, returning best answer if available.
            max_errors: Maximum consecutive errors before stopping. Execution stops if exceeded, returning best answer if available.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends that the environments can use to make sub-calls.
            other_backend_kwargs: The kwargs to pass to the other client backends (ordered to match other_backends).
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output in rich to console.
            persistent: If True, reuse the environment across completion() calls for multi-turn conversations.
            custom_tools: Dict of custom functions/tools available in the REPL. Keys are function names,
                values are callable functions. These are injected into the REPL globals.
            custom_sub_tools: Dict of custom tools for sub-agents (llm_query calls). If None, inherits
                from custom_tools. Pass an empty dict {} to disable tools for sub-agents.
            compaction: If True, keep full root model history in REPL variable `history` and compact
                when root context reaches compaction_threshold_pct of the model's context limit.
            compaction_threshold_pct: When compaction is on, trigger summarization when root
                message token count reaches this fraction of the model context limit (default 0.85).
            max_concurrent_subcalls: Maximum number of parallel threads for rlm_query_batched subcalls.
                Each child RLM runs in its own thread. Default 4.
            on_subcall_start: Callback fired when a child RLM starts. Args: (depth, model, prompt_preview).
            on_subcall_complete: Callback fired when a child RLM completes. Args: (depth, model, duration, error_or_none).
            on_iteration_start: Callback fired when an iteration starts. Args: (depth, iteration_num).
            on_iteration_complete: Callback fired when an iteration completes. Args: (depth, iteration_num, duration).
            metadata_builder: Optional S7 seam. Called as
                ``builder(execution_result, repl_inventory)`` once per executed code
                block to produce what carries into the next turn. ``repl_inventory``
                is the redacted ``{name: (type_name, length)}`` view of the REPL
                namespace — never the values. Defaults to the shipped 20K
                head-truncation.
            answer_middleware: Optional S9 seam. Called as
                ``middleware(final_answer, repl_inventory)`` when a final answer is
                detected; returns an ``AnswerDecision`` that either accepts the
                answer (identity, the default when unset) or redirects — suppressing
                the answer, resetting the REPL answer dict, and injecting a nudge.
            runtime_policy: Optional S6 policy dict handed to the ``local``
                environment: ``enabled``, ``max_batch_width``, ``max_prompt_chars``,
                ``retry_on_syntax_error``, ``max_retries``, ``validate_sub_output``.
                Applies to this RLM's own environment; child RLMs spawned by
                ``rlm_query`` build their own. Defaults to unbounded pass-through.
            capacity_sentence: Optional S6 sentence stating sub-call capacity in the
                metadata user message. Defaults to the shipped ~100k tokens string.
        """
        # Sampling args plumbed into backend_kwargs / other_backend_kwargs
        # before the clients are constructed, so they reach the chat-completions
        # call (e.g. temperature, top_p, max_tokens, seed). ``sampling_args``
        # applies to the root model (depth=0); ``sub_sampling_args`` to
        # depth=1 sub-LLM calls. If ``sub_sampling_args`` is set without an
        # ``other_backends``, we mirror the root backend so depth=1 routes
        # through a separate client with its own sampling args.
        if sampling_args is not None:
            backend_kwargs = dict(backend_kwargs or {})
            existing = dict(backend_kwargs.get("sampling_args") or {})
            existing.update(sampling_args)
            backend_kwargs["sampling_args"] = existing
        if sub_sampling_args is not None:
            if other_backends is None:
                other_backends = [backend]
                other_backend_kwargs = [dict(backend_kwargs or {})]
            else:
                other_backend_kwargs = [dict(kw or {}) for kw in (other_backend_kwargs or [{}])]
            first = dict(other_backend_kwargs[0])
            existing = dict(first.get("sampling_args") or {})
            existing.update(sub_sampling_args)
            first["sampling_args"] = existing
            other_backend_kwargs[0] = first

        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        # Validate other_backends: currently only support one additional backend
        if other_backends is not None:
            if len(other_backends) != 1:
                raise ValueError(
                    "We currently only support one additional backend for the recursive sub-calls! "
                    "This model will be the model used for recursive sub-calls, but this will change in the future"
                )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        # Custom tools: functions available in the REPL environment
        self.custom_tools = custom_tools
        # Sub-tools: if None, inherit from custom_tools; if {}, no tools for sub-agents
        self.custom_sub_tools = custom_sub_tools if custom_sub_tools is not None else custom_tools

        self.compaction = compaction
        self.compaction_threshold_pct = compaction_threshold_pct
        self.max_concurrent_subcalls = max_concurrent_subcalls

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_budget = max_budget
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.orchestrator = orchestrator
        # Harness seams. All default to None, i.e. the shipped behavior.
        self.metadata_builder = metadata_builder
        self.answer_middleware = answer_middleware
        self.runtime_policy = runtime_policy
        self.capacity_sentence = capacity_sentence
        # Optional user-prologue message inserted between the metadata user
        # message and the iter-0 turn prompt. Mirrors RLMTrainEnv's
        # ``user_prologue`` so canonical inference can match envs that
        # depend on a task-specific tips message (e.g. BC+).
        self.user_prologue = user_prologue
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Event callbacks for live tree display
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

        # Tracking (cumulative across all calls including children)
        self._cumulative_cost: float = 0.0
        # Spend by recursive children, which run their own handlers and are
        # therefore invisible to this RLM's handler aggregate. Reset per
        # completion alongside ``_cumulative_cost``.
        self._subcall_usage: UsageSummary = UsageSummary(model_usage_summaries={})
        # What the last completion recorded, published by the completion context
        # on its way out (see ``last_completion_usage``). None until one runs.
        self._last_completion_usage: UsageSummary | None = None
        self._consecutive_errors: int = 0
        self._last_error: str | None = None
        self._best_partial_answer: str | None = None
        self._completion_start_time: float | None = None  # Set when completion() starts

        # Persistence support
        self.persistent = persistent
        self._persistent_env: SupportsPersistence | None = None

        # Validate persistence support at initialization
        if self.persistent:
            self._validate_persistent_environment_support()

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=other_backends,
                # The run-start record of what skills were available: read off
                # the installed loader, so it is present exactly when one is.
                skill_index=_skill_index_of(self.custom_tools),
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        Spawn an LM handler and environment for a single completion call.

        When persistent=True, the environment is reused across calls.
        When persistent=False (default), creates fresh environment each call.
        """
        # Create client and wrap in handler
        client: BaseLM = get_client(self.backend, self.backend_kwargs)

        # Create other_backend_client if provided (for depth=1 routing)
        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(self.other_backends[0], self.other_backend_kwargs[0])

        lm_handler = LMHandler(client, other_backend_client=other_backend_client)

        # Register other clients to be available as sub-call options (by model name).
        # Reuse other_backend_client for the first entry so each (backend, kwargs)
        # pair is instantiated exactly once.
        if other_backend_client is not None:
            lm_handler.register_client(other_backend_client.model_name, other_backend_client)
            for backend, kwargs in zip(
                self.other_backends[1:],
                self.other_backend_kwargs[1:],
                strict=True,
            ):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        # Environment: reuse if persistent, otherwise create fresh
        if self.persistent and self._persistent_env is not None:
            environment = self._persistent_env
            # Defensive check: ensure environment supports persistence methods
            if not self._env_supports_persistence(environment):
                raise RuntimeError(
                    f"Persistent environment of type '{type(environment).__name__}' does not "
                    f"implement required methods (update_handler_address, add_context, get_context_count). "
                    f"This should have been caught at initialization."
                )
            environment.update_handler_address((lm_handler.host, lm_handler.port))
            environment.add_context(prompt)
        else:
            env_kwargs = self.environment_kwargs.copy()
            env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
            env_kwargs["context_payload"] = prompt
            env_kwargs["depth"] = self.depth + 1  # Environment depth is RLM depth + 1
            # For environments that support recursive RLM calls, pass the subcall
            # callback when max_depth > 1. local/ipython invoke it in-process;
            # docker invokes it via its host-side proxy (/rlm_query endpoints).
            if self.environment_type in ("local", "ipython", "docker") and self.max_depth > 1:
                env_kwargs["subcall_fn"] = self._subcall
            # Pass custom tools to the environment
            if self.custom_tools is not None:
                env_kwargs["custom_tools"] = self.custom_tools
            if self.custom_sub_tools is not None:
                env_kwargs["custom_sub_tools"] = self.custom_sub_tools
            if self.compaction and self.environment_type in ("local", "docker"):
                env_kwargs["compaction"] = True
            env_kwargs["max_concurrent_subcalls"] = self.max_concurrent_subcalls
            # S6 runtime policy is enforced by the local REPL's sub-call helpers.
            if self.runtime_policy is not None and self.environment_type == "local":
                env_kwargs["runtime_policy"] = self.runtime_policy
            environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

            if self.persistent:
                self._persistent_env = environment

        try:
            yield lm_handler, environment
        finally:
            # Publish the run's total before the only object holding it dies.
            # This block runs during exception propagation too, after every
            # completed call has been recorded, which makes it the one point
            # where "the run is over" is true however it ended -- including a
            # limit raised inside a client, which never reaches the iteration
            # checks. A terminated completion returns nothing, so without this
            # its usage would be unrecoverable and the run would read as free.
            self._last_completion_usage = lm_handler.get_usage_summary().merged_with(
                self._subcall_usage
            )
            lm_handler.stop()
            if not self.persistent and hasattr(environment, "cleanup"):
                environment.cleanup()

    def _setup_prompt(
        self,
        prompt: str | dict[str, Any],
        root_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
            custom_tools=self.custom_tools,
            root_prompt=root_prompt,
            orchestrator=self.orchestrator,
            capacity_sentence=self.capacity_sentence,
        )
        if self.user_prologue:
            message_history.append({"role": "user", "content": self.user_prologue})
        if self.compaction:
            message_history[0]["content"] += (
                "\n\nThe full conversation history (trajectory segments and any summaries) "
                "is available in the REPL variable `history` as a list."
            )
        return message_history

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """
        Recursive Language Model completion call. This is the main entry point for querying an RLM, and
        can replace a regular LM completion call.

        Spawns its own environment and LM handler for the duration of this call.

        Args:
            prompt: A single string or dictionary of messages to pass as context to the model.
            root_prompt: We allow the RLM's root LM to see a (small) prompt that the user specifies. A common example of this
            is if the user is asking the RLM to answer a question, we can pass the question as the root prompt.
        Returns:
            A final answer as a string.
        """
        time_start = time.perf_counter()
        self._completion_start_time = time_start

        # Reset tracking state for this completion
        self._consecutive_errors = 0
        self._last_error = None
        self._best_partial_answer = None
        self._reset_completion_accounting()
        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        if self.logger:
            self.logger.clear_iterations()

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt, root_prompt=root_prompt)

            compaction_count = 0
            try:
                for i in range(self.max_iterations):
                    # Check timeout before each iteration
                    self._check_timeout(i, time_start)

                    # Compaction: check if context needs summarization
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        current_tokens, threshold_tokens, max_tokens = self._get_compaction_status(
                            message_history
                        )
                        self.verbose.print_compaction_status(
                            current_tokens, threshold_tokens, max_tokens
                        )
                        if current_tokens >= threshold_tokens:
                            compaction_count += 1
                            self.verbose.print_compaction()
                            message_history = self._compact_history(
                                lm_handler, environment, message_history, compaction_count
                            )

                    context_count = (
                        environment.get_context_count()
                        if isinstance(environment, SupportsPersistence)
                        else 1
                    )
                    history_count = (
                        environment.get_history_count()
                        if isinstance(environment, SupportsPersistence)
                        else 0
                    )
                    # Fully prefixed trajectory: persist the per-turn user prompt
                    # into message_history so the model sees a single continuous
                    # [system, metadata, user_0, assistant_0, repl_0, user_1, ...]
                    # chain across turns.
                    message_history.append(
                        build_user_prompt(
                            root_prompt,
                            i,
                            context_count,
                            history_count,
                            max_iterations=self.max_iterations,
                        )
                    )

                    cost_before = lm_handler.get_usage_summary().total_cost

                    iteration: RLMIteration = self._completion_turn(
                        prompt=message_history,
                        lm_handler=lm_handler,
                        environment=environment,
                    )

                    # Check error/budget/token limits after each iteration
                    self._check_iteration_limits(iteration, i, lm_handler)

                    # The REPL signals completion by populating
                    # ``answer["content"]`` and setting ``answer["ready"] = True``.
                    # Each environment surfaces that on ``REPLResult.final_answer``.
                    final_answer = None
                    answer_inventory: dict[str, tuple[str, int]] = {}
                    for block in iteration.code_blocks:
                        if getattr(block.result, "final_answer", None) is not None:
                            final_answer = block.result.final_answer
                            # Only the middleware reads the inventory; building it
                            # otherwise walks the namespace for a value nobody uses.
                            if self.answer_middleware is not None:
                                answer_inventory = build_repl_inventory(block.result.locals)
                            break

                    # S9: programmatic inspection of the detected answer.
                    answer_event: str | None = None
                    nudge: str | None = None
                    if final_answer is not None:
                        answer_event = ANSWER_SUBMITTED
                        if self.answer_middleware is not None:
                            final_answer, nudge = self._apply_answer_middleware(
                                final_answer, answer_inventory, environment
                            )
                            if final_answer is None:
                                answer_event = ANSWER_REDIRECTED

                    iteration.final_answer = final_answer

                    # Store as best partial answer (most recent response with content)
                    if iteration.response and iteration.response.strip():
                        self._best_partial_answer = iteration.response

                    turn_metrics = self._build_turn_metrics(
                        iteration, answer_event, cost_before, lm_handler
                    )
                    # Format the iteration for the next prompt. Done before logging
                    # so the turn record carries the truncation event.
                    new_messages: list[dict[str, str]] | None = None
                    if final_answer is None:
                        new_messages = format_iteration(
                            iteration,
                            metadata_builder=self.metadata_builder,
                            metrics=turn_metrics,
                        )
                    iteration.trace_metrics = turn_metrics

                    # If logger is used, log the iteration.
                    if self.logger:
                        self.logger.log(iteration)

                    # Verbose output for this iteration
                    self.verbose.print_iteration(iteration, i + 1)

                    if final_answer is not None:
                        time_end = time.perf_counter()
                        usage = lm_handler.get_usage_summary().merged_with(self._subcall_usage)
                        self.verbose.print_final_answer(final_answer)
                        self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                        # Store message history in persistent environment
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)

                        return RLMChatCompletion(
                            root_model=self.backend_kwargs.get("model_name", "unknown")
                            if self.backend_kwargs
                            else "unknown",
                            prompt=prompt,
                            response=final_answer,
                            usage_summary=usage,
                            execution_time=time_end - time_start,
                            metadata=self.logger.get_trajectory() if self.logger else None,
                        )

                    # Update message history with the new messages.
                    assert new_messages is not None
                    message_history.extend(new_messages)
                    # S9 redirect: push the nudge back to the model and keep going.
                    if nudge is not None:
                        message_history.append({"role": "user", "content": nudge})
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        environment.append_compaction_entry(new_messages)

            except KeyboardInterrupt:
                self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
                raise CancellationError(
                    partial_answer=self._best_partial_answer,
                    message="Execution cancelled by user (Ctrl+C)",
                ) from None

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary().merged_with(self._subcall_usage)
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())

            # Store message history in persistent environment
            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
                metadata=self.logger.get_trajectory() if self.logger else None,
            )

    def _apply_answer_middleware(
        self,
        final_answer: str,
        answer_inventory: dict[str, tuple[str, int]],
        environment: BaseEnv,
    ) -> tuple[str | None, str | None]:
        """
        Run the S9 middleware on a detected final answer.

        Returns ``(answer, nudge)``. An accepted decision returns
        ``(answer, None)``; a redirect returns ``(None, nudge)`` after resetting
        the environment's answer dict so the model's namespace no longer reads
        ``ready: True``.
        """
        decision = self.answer_middleware(final_answer, answer_inventory)
        if decision.accepted:
            if decision.answer is None:
                raise ValueError("Answer middleware accepted an answer but returned answer=None")
            return decision.answer, None

        if not hasattr(environment, "reset_answer"):
            raise RuntimeError(
                f"Answer middleware redirected, but environment "
                f"'{type(environment).__name__}' does not implement reset_answer(). "
                "Without it the REPL namespace still reads answer['ready'] = True and "
                "the run burns to max_iterations."
            )
        environment.reset_answer()
        return None, decision.nudge

    @staticmethod
    def _build_turn_metrics(
        iteration: RLMIteration,
        answer_event: str | None,
        cost_before: float | None,
        lm_handler: LMHandler,
    ) -> dict[str, Any]:
        """
        Collect the per-turn trace metrics (KTD5).

        ``truncation_event`` is filled in by ``format_iteration`` when the
        metadata step actually shortens a block's output.
        """
        cost_after = lm_handler.get_usage_summary().total_cost
        if cost_before is None or cost_after is None:
            turn_cost = None
        else:
            turn_cost = cost_after - cost_before

        return {
            "sub_call_count": sum(len(block.result.rlm_calls) for block in iteration.code_blocks),
            "skill_load_count": sum(
                len(block.result.skill_loads) for block in iteration.code_blocks
            ),
            "syntax_error": any(
                has_syntax_error(block.result.stderr) for block in iteration.code_blocks
            ),
            "answer_event": answer_event,
            "truncation_event": False,
            "cost": turn_cost,
        }

    def _check_timeout(self, iteration: int, time_start: float) -> None:
        """Raise TimeoutExceededError if the timeout has been exceeded."""
        if self.max_timeout is None:
            return
        elapsed = time.perf_counter() - time_start
        if elapsed > self.max_timeout:
            self.verbose.print_limit_exceeded(
                "timeout",
                f"{elapsed:.1f}s of {self.max_timeout:.1f}s",
            )
            raise TimeoutExceededError(
                elapsed=elapsed,
                timeout=self.max_timeout,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Timeout exceeded after iteration {iteration}: "
                    f"{elapsed:.1f}s of {self.max_timeout:.1f}s limit"
                ),
            )

    def _check_iteration_limits(
        self, iteration: RLMIteration, iteration_num: int, lm_handler: LMHandler
    ) -> None:
        """Check error tracking, budget, and token limits after an iteration.

        Raises ErrorThresholdExceededError, BudgetExceededError, or TokenLimitExceededError
        if the respective limits are exceeded.
        """
        # Track errors from code execution (check stderr for errors)
        iteration_had_error = False
        for code_block in iteration.code_blocks:
            if code_block.result and code_block.result.stderr:
                iteration_had_error = True
                self._last_error = code_block.result.stderr
                break

        if iteration_had_error:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0  # Reset on success

        # Check error threshold
        if self.max_errors is not None and self._consecutive_errors >= self.max_errors:
            self.verbose.print_limit_exceeded(
                "errors",
                f"{self._consecutive_errors} consecutive errors (limit: {self.max_errors})",
            )
            raise ErrorThresholdExceededError(
                error_count=self._consecutive_errors,
                threshold=self.max_errors,
                last_error=self._last_error,
                partial_answer=self._best_partial_answer,
                message=(
                    "Error threshold exceeded: "
                    f"{self._consecutive_errors} consecutive errors "
                    f"(limit: {self.max_errors})"
                ),
            )

        if self.max_budget is None and self.max_tokens is None:
            return

        # The handler sees only this RLM's own calls; recursive children spend
        # through their own handlers, so their total is folded in here or it
        # never counts against either cap at all. Both checks read this one
        # snapshot -- nothing between them can move it.
        current_usage = lm_handler.get_usage_summary().merged_with(self._subcall_usage)

        # Check budget
        if self.max_budget is not None:
            current_cost = current_usage.total_cost or 0.0
            self._cumulative_cost = current_cost
            if self._cumulative_cost > self.max_budget:
                self.verbose.print_budget_exceeded(self._cumulative_cost, self.max_budget)
                raise BudgetExceededError(
                    spent=self._cumulative_cost,
                    budget=self.max_budget,
                    message=(
                        f"Budget exceeded after iteration {iteration_num + 1}: "
                        f"spent ${self._cumulative_cost:.6f} "
                        f"of ${self.max_budget:.6f} budget"
                    ),
                )

        # Check token limit, over the same combined view as the budget.
        if self.max_tokens is not None:
            total_tokens = current_usage.total_input_tokens + current_usage.total_output_tokens
            if total_tokens > self.max_tokens:
                self.verbose.print_limit_exceeded(
                    "tokens",
                    f"{total_tokens:,} of {self.max_tokens:,} tokens",
                )
                raise TokenLimitExceededError(
                    tokens_used=total_tokens,
                    token_limit=self.max_tokens,
                    partial_answer=self._best_partial_answer,
                    message=(
                        f"Token limit exceeded after iteration {iteration_num + 1}: "
                        f"{total_tokens:,} of {self.max_tokens:,} tokens"
                    ),
                )

    def _reset_completion_accounting(self) -> None:
        """Clear per-completion spend state.

        ``run_round`` reuses one RLM for every run in a round, so without this
        a run would inherit its predecessor's spend.
        """
        self._cumulative_cost = 0.0
        self._subcall_usage = UsageSummary(model_usage_summaries={})
        self._last_completion_usage = None

    @property
    def last_completion_usage(self) -> UsageSummary | None:
        """What the most recent completion on this instance recorded, or None.

        Read-only, and the only way a caller can price a completion that raised
        instead of returning: a terminated run has no ``RLMChatCompletion`` to
        carry its usage. Reset per completion, so it never reports the
        predecessor's figure on a reused instance.
        """
        return self._last_completion_usage

    def _record_subcall_usage(self, usage_summary: UsageSummary | None) -> None:
        """Fold one child completion's usage into this completion's total."""
        if usage_summary is None:
            return
        self._subcall_usage = self._subcall_usage.merged_with(usage_summary)

    def _get_compaction_status(self, message_history: list[dict[str, Any]]) -> tuple[int, int, int]:
        """Return (current_tokens, threshold_tokens, max_tokens) for compaction."""
        model_name = (
            self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"
        )
        max_tokens = get_context_limit(model_name)
        current_tokens = count_tokens(message_history, model_name)
        threshold_tokens = int(self.compaction_threshold_pct * max_tokens)
        return current_tokens, threshold_tokens, max_tokens

    def _should_compact(self, message_history: list[dict[str, Any]]) -> bool:
        """True when root message history is at or over the compaction threshold."""
        current_tokens, threshold_tokens, _ = self._get_compaction_status(message_history)
        return current_tokens >= threshold_tokens

    def _compact_history(
        self,
        lm_handler: LMHandler,
        environment: BaseEnv,
        message_history: list[dict[str, Any]],
        compaction_count: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Summarize current trajectory, append summary to REPL history, and return
        a short message_history with the summary as the new starting point.
        """
        summary_prompt = message_history + [
            {
                "role": "user",
                "content": (
                    "Summarize your progress so far. Include:\n"
                    "1. Which steps/sub-tasks you have completed and which remain.\n"
                    "2. Any concrete intermediate results (numbers, values, variable names) "
                    "you computed — preserve these exactly.\n"
                    "3. What your next action should be.\n"
                    "Be concise (1–3 paragraphs) but preserve all key results and your "
                    "current position in the task."
                ),
            }
        ]
        summary = lm_handler.completion(summary_prompt)
        if hasattr(environment, "append_compaction_entry"):
            environment.append_compaction_entry({"type": "summary", "content": summary})
        # Keep system + initial assistant (metadata), then summary + continue
        new_history = message_history[:2] + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    f"Your conversation has been compacted {compaction_count} time(s). "
                    "Continue from the above summary. Do NOT repeat work you have already "
                    "completed. Use SHOW_VARS() to check which REPL variables exist, "
                    "and check `history` for full context. "
                    "Your next action:"
                ),
            },
        ]
        return new_history

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        response = client.completion(message)
        return response

    def _subcall(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        """
        Handle a subcall from the environment, potentially spawning a child RLM.

        This method is passed as a callback to LocalREPL to enable recursive RLM calls.
        When depth allows, it spawns a child RLM with its own REPL. At max depth,
        it falls back to a plain LM completion.

        Args:
            prompt: The prompt to process.
            model: Optional model name. If specified, the child RLM will use this model
                instead of inheriting the parent's default backend.

        Returns:
            The full RLMChatCompletion from either a child RLM or plain LM completion.
            On error, returns a completion with the error message as the response.
        """
        next_depth = self.depth + 1

        # Determine which backend/kwargs to use (model override or parent's default)
        if model is not None:
            child_backend_kwargs = (self.backend_kwargs or {}).copy()
            child_backend_kwargs["model_name"] = model
        else:
            child_backend_kwargs = self.backend_kwargs
        resolved_model = model or (child_backend_kwargs or {}).get("model_name", "unknown")

        # If we'd hit/exceed the cap, do a normal LM completion (no REPL)
        if next_depth >= self.max_depth:
            # Use other_backend if available, otherwise use main backend
            if self.other_backends and self.other_backend_kwargs:
                client = get_client(self.other_backends[0], self.other_backend_kwargs[0])
            else:
                client = get_client(self.backend, child_backend_kwargs or {})
            root_model = model or client.model_name
            start_time = time.perf_counter()
            try:
                response = client.completion(prompt)
                end_time = time.perf_counter()
                model_usage = client.get_last_usage()
                usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
                # This branch returns before the recursive path's accounting, so
                # a depth-capped sub-call would otherwise spend invisibly.
                self._record_subcall_usage(usage_summary)
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=response,
                    usage_summary=usage_summary,
                    execution_time=end_time - start_time,
                )
            except Exception as e:
                end_time = time.perf_counter()
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=f"Error: LM query failed at max depth - {e}",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=end_time - start_time,
                )

        # Calculate remaining budget for child (if budget tracking enabled)
        remaining_budget = None
        if self.max_budget is not None:
            remaining_budget = self.max_budget - self._cumulative_cost
            if remaining_budget <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=(
                        "Error: Budget exhausted "
                        f"(spent ${self._cumulative_cost:.6f} of ${self.max_budget:.6f})"
                    ),
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Calculate remaining timeout for child (if timeout tracking enabled)
        remaining_timeout = None
        if self.max_timeout is not None and self._completion_start_time is not None:
            elapsed = time.perf_counter() - self._completion_start_time
            remaining_timeout = self.max_timeout - elapsed
            if remaining_timeout <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=f"Error: Timeout exhausted ({elapsed:.1f}s of {self.max_timeout:.1f}s)",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Resolve the model name for callbacks
        prompt_preview = prompt[:80] if len(prompt) > 80 else prompt

        # Fire subcall start callback
        if self.on_subcall_start:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt_preview)
            except Exception:
                pass  # Don't let callback errors break execution

        subcall_start = time.perf_counter()
        error_msg: str | None = None

        # Spawn a child RLM with its own LocalREPL
        child = RLM(
            backend=self.backend,
            backend_kwargs=child_backend_kwargs,
            environment=self.environment_type,
            environment_kwargs=self.environment_kwargs,
            depth=next_depth,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_budget=remaining_budget,
            max_timeout=remaining_timeout,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            custom_system_prompt=self.system_prompt,
            # Without this the child defaults to orchestrator=True and receives
            # ORCHESTRATOR_ADDENDUM even when the root does not. That would put the
            # authors' hand-tuned orchestration guidance back into every recursive
            # call of a harness whose whole purpose is to start without it.
            orchestrator=self.orchestrator,
            capacity_sentence=self.capacity_sentence,
            other_backends=self.other_backends,
            other_backend_kwargs=self.other_backend_kwargs,
            # Give child its own logger so its trajectory is captured in metadata
            logger=RLMLogger() if self.logger else None,
            verbose=False,
            # Propagate custom tools to children (sub_tools become the child's tools)
            custom_tools=self.custom_sub_tools,
            custom_sub_tools=self.custom_sub_tools,
            # Propagate concurrency settings to children
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            # Propagate callbacks to children for nested tracking
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
        )
        try:
            result = child.completion(prompt, root_prompt=None)
            self._record_subcall_usage(result.usage_summary)
            return result
        except BudgetExceededError as e:
            self._record_subcall_usage(_terminated_child_usage(child, e, resolved_model))
            error_msg = f"Budget exceeded - {e}"
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM budget exceeded - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        except Exception as e:
            # Token, timeout and error-threshold terminations land here. They
            # carry no spend figure of their own, so the usage the root attached
            # on its way out is the only record of what the child burned.
            self._record_subcall_usage(_terminated_child_usage(child, e, resolved_model))
            error_msg = str(e)
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM completion failed - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        finally:
            # Ensure child resources are cleaned up
            child.close()
            # Fire subcall complete callback
            if self.on_subcall_complete:
                try:
                    duration = time.perf_counter() - subcall_start
                    self.on_subcall_complete(next_depth, str(resolved_model), duration, error_msg)
                except Exception:
                    pass  # Don't let callback errors break execution

    def _validate_persistent_environment_support(self) -> None:
        """
        Validate that the configured environment type supports persistent mode.

        Persistent mode requires environments to implement:
        - update_handler_address(address): Update LM handler address between calls
        - add_context(payload, index): Add new context for multi-turn conversations
        - get_context_count(): Return the number of loaded contexts

        Currently 'local', 'ipython', and 'docker' support these methods.

        Raises:
            ValueError: If the environment type does not support persistent mode.
        """
        # Known environments that support persistence
        persistent_supported_environments = {"local", "ipython", "docker"}

        if self.environment_type not in persistent_supported_environments:
            raise ValueError(
                f"persistent=True is not supported for environment type '{self.environment_type}'. "
                f"Persistent mode requires environments that implement update_handler_address(), "
                f"add_context(), and get_context_count(). "
                f"Supported environments: {sorted(persistent_supported_environments)}"
            )

    @staticmethod
    def _env_supports_persistence(env: BaseEnv) -> bool:
        """Check if an environment instance supports persistent mode methods."""
        return isinstance(env, SupportsPersistence)

    def close(self) -> None:
        """Clean up persistent environment. Call when done with multi-turn conversations."""
        if self._persistent_env is not None:
            if hasattr(self._persistent_env, "cleanup"):
                self._persistent_env.cleanup()
            self._persistent_env = None

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
