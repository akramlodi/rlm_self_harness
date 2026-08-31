"""Custom exceptions for RLM execution limits and cancellation."""


class BudgetExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum budget."""

    def __init__(self, spent: float, budget: float, message: str | None = None):
        self.spent = spent
        self.budget = budget
        super().__init__(message or f"Budget exceeded: spent ${spent:.6f} of ${budget:.6f} budget")


class TimeoutExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum timeout."""

    def __init__(
        self,
        elapsed: float,
        timeout: float,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.elapsed = elapsed
        self.timeout = timeout
        self.partial_answer = partial_answer
        super().__init__(message or f"Timeout exceeded: {elapsed:.1f}s of {timeout:.1f}s limit")


class TokenLimitExceededError(Exception):
    """Raised when the RLM execution exceeds the maximum token limit."""

    def __init__(
        self,
        tokens_used: int,
        token_limit: int,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.tokens_used = tokens_used
        self.token_limit = token_limit
        self.partial_answer = partial_answer
        super().__init__(
            message or f"Token limit exceeded: {tokens_used:,} of {token_limit:,} tokens"
        )


class ErrorThresholdExceededError(Exception):
    """Raised when the RLM encounters too many consecutive errors."""

    def __init__(
        self,
        error_count: int,
        threshold: int,
        last_error: str | None = None,
        partial_answer: str | None = None,
        message: str | None = None,
    ):
        self.error_count = error_count
        self.threshold = threshold
        self.last_error = last_error
        self.partial_answer = partial_answer
        super().__init__(
            message
            or f"Error threshold exceeded: {error_count} consecutive errors (limit: {threshold})"
        )


class HardDeadlineSignal(BaseException):
    """A wall-clock backstop fired on this thread (SIGALRM handler).

    Derives from ``BaseException`` on purpose: the alarm can land while the
    main thread is inside model code (``LocalREPL.execute``) or a sub-call
    wrapper, both of which catch ``Exception`` and turn it into an in-band
    error string. A plain ``Exception`` was swallowed there and the run kept
    going past its deadline (observed 2026-08-29: 5011s against an 1800s cap).
    Only the run driver's limit handler converts this into a persisted,
    resource-terminated run; nothing else should catch it.
    """

    def __init__(self, deadline: float, message: str | None = None):
        self.deadline = deadline
        super().__init__(
            message
            or (
                f"hard wall-clock deadline exceeded: the run slice did not return "
                f"within {deadline:.1f}s; candidate code likely hung inside a live call"
            )
        )


class HardDeadlineExceeded(TimeoutExceededError):
    """The hard wall-clock backstop fired: a run slice never returned control.

    Subclasses ``TimeoutExceededError`` so the run driver persists it exactly
    like the runtime's own timeout -- a failing RESOURCE_TERMINATED run -- while
    the error string still names this class, which is how a backstop kill is
    told apart from an ordinary between-iteration timeout in the audit. It is
    raised by the driver when it catches ``HardDeadlineSignal``, and by the
    governed-round slice when the signal escapes ``run_round`` entirely.
    """

    def __init__(self, deadline: float, message: str | None = None):
        super().__init__(
            elapsed=deadline,
            timeout=deadline,
            message=message
            or (
                f"hard wall-clock deadline exceeded: the run slice did not return "
                f"within {deadline:.1f}s; candidate code likely hung inside a live call"
            ),
        )


class CancellationError(Exception):
    """Raised when the RLM execution is cancelled by the user."""

    def __init__(self, partial_answer: str | None = None, message: str | None = None):
        self.partial_answer = partial_answer
        super().__init__(message or "Execution cancelled by user")
