"""Call-local provider observations, separate from model-visible responses.

The runtime supplies a recorder; clients publish only fields from responses
they actually received. No recorder means no capture and no filesystem work.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

OBSERVATION_FORMAT = "rlm-llm-observation/v1"
ObservationSink = Callable[[dict[str, Any]], dict[str, Any]]


class ObservationPersistenceError(RuntimeError):
    """Capture failed; never retry the model or score this as a task failure."""


ACTIVE_RECORDER: ContextVar[ObservationRecorder | None] = ContextVar("llm_recorder", default=None)
ACTIVE_CALL: ContextVar[ObservedCall | None] = ContextVar("llm_call", default=None)


def json_value(value: Any) -> Any:
    """Encode allowlisted SDK fields, including opaque byte signatures."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="python", exclude_unset=True))
    # Test doubles and older SDK field containers have no model_dump.
    if hasattr(value, "__dict__"):
        return json_value(vars(value))
    raise TypeError(f"Unsupported observation value: {type(value).__name__}")


def field_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def selected_fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    missing = object()
    result = {}
    for name in names:
        item = field_value(value, name, missing)
        if item is not missing:
            result[name] = json_value(item)
    return result


def reasoning_kinds(reasoning: dict[str, Any]) -> list[str]:
    kinds: set[str] = set()
    for name, value in reasoning.items():
        if name in ("reasoning", "reasoning_content") and isinstance(value, str) and value:
            kinds.add("text")
        if isinstance(value, list):
            for block in value:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type", "")
                if kind == "reasoning.summary" or block.get("thought"):
                    if block.get("summary") or block.get("text"):
                        kinds.add("summary")
                elif block.get("text") or block.get("thinking"):
                    kinds.add("text")
                if kind in ("reasoning.encrypted", "redacted_thinking") or any(
                    block.get(key) for key in ("data", "signature", "thought_signature")
                ):
                    kinds.add("opaque")
    return sorted(kinds)


def response_observation(
    response: Any, provider: str, model: str, response_format: str = "openai"
) -> dict[str, Any]:
    """Extract existing non-streaming provider fields without normalizing text."""
    reasoning: dict[str, Any] = {}
    usage = field_value(response, "usage")
    tokens = field_value(field_value(usage, "completion_tokens_details"), "reasoning_tokens")
    finish_reason = None
    if response_format == "anthropic":
        content = field_value(response, "content", []) or []
        raw_content = json_value(content)
        reasoning["thinking_blocks"] = [
            json_value(block)
            for block in content
            if field_value(block, "type") in ("thinking", "redacted_thinking")
        ]
        finish_reason = field_value(response, "stop_reason")
    elif response_format == "gemini":
        candidates = field_value(response, "candidates", []) or []
        first = candidates[0] if candidates else None
        parts = field_value(field_value(first, "content"), "parts", []) or []
        raw_content = json_value(parts)
        reasoning["thought_parts"] = [
            selected_fields(part, ("text", "thought", "thought_signature"))
            for part in parts
            if field_value(part, "thought") or field_value(part, "thought_signature")
        ]
        tokens = field_value(field_value(response, "usage_metadata"), "thoughts_token_count")
        finish_reason = field_value(first, "finish_reason")
    else:
        choices = field_value(response, "choices", []) or []
        first = choices[0] if choices else None
        message = field_value(first, "message")
        raw_content = json_value(field_value(message, "content"))
        reasoning = selected_fields(
            message, ("reasoning_content", "reasoning", "reasoning_details")
        )
        finish_reason = field_value(first, "finish_reason")
    kinds = reasoning_kinds(reasoning)
    availability = (
        "returned"
        if set(kinds) & {"text", "summary"}
        else "opaque_only"
        if kinds
        else "not_returned"
    )
    return {
        "provider": provider,
        "model": model,
        "response_model": field_value(response, "model", field_value(response, "model_version")),
        "provider_response_id": field_value(response, "id", field_value(response, "response_id")),
        "finish_reason": json_value(finish_reason),
        "raw_content": raw_content,
        "reasoning": reasoning,
        "reasoning_kinds": kinds,
        "reasoning_tokens": tokens
        if isinstance(tokens, int) and not isinstance(tokens, bool)
        else None,
        "availability": availability,
    }


@dataclass
class ObservedCall:
    recorder: ObservationRecorder
    purpose: str
    model: str | None
    coordinates: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid4().hex)
    parent_call_id: str | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "format": OBSERVATION_FORMAT,
            "event": event,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "purpose": self.purpose,
            "model": self.model,
            "owner": self.recorder.owner,
            "coordinates": self.coordinates,
            "recorded_at": datetime.now(UTC).isoformat(),
            "cached": False,
            **fields,
        }
        reference = self.recorder.publish(record)
        self.references.append(reference)
        if event == "response":
            self.responses.append(record)
        return record

    def receive(self, payload: dict[str, Any]) -> None:
        self.emit(
            "response", **{**payload, "attempt_id": f"{self.call_id}:{len(self.responses) + 1}"}
        )

    def replay(self, responses: list[dict[str, Any]] | None) -> None:
        for record in responses or [{"availability": "legacy_unavailable"}]:
            payload = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "format",
                    "event",
                    "call_id",
                    "attempt_id",
                    "parent_call_id",
                    "purpose",
                    "owner",
                    "coordinates",
                    "recorded_at",
                }
            }
            self.receive({**payload, "cached": True, "source_call_id": record.get("call_id")})


class ObservationRecorder:
    """One run/stage owner; call bindings stay local to threads and tasks."""

    def __init__(self, owner: dict[str, Any] | None = None, sink: ObservationSink | None = None):
        self.owner = dict(owner or {})
        self.sink = sink
        self.references: list[dict[str, Any]] = []
        self.failure: ObservationPersistenceError | None = None
        self.lock = Lock()

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def publish(self, record: dict[str, Any]) -> dict[str, Any]:
        self.check()
        try:
            reference = self.sink(record) if self.sink else {"observation": record}
        except Exception as error:
            failure = ObservationPersistenceError(f"LLM observation could not be saved: {error}")
            with self.lock:
                self.failure = failure
            raise failure from error
        with self.lock:
            self.references.append(reference)
        return reference

    @contextmanager
    def call(
        self, purpose: str, model: str | None = None, **coordinates: Any
    ) -> Iterator[ObservedCall]:
        self.check()
        parent = ACTIVE_CALL.get()
        call = ObservedCall(
            self,
            purpose,
            model,
            coordinates,
            parent_call_id=parent.call_id if parent and parent.recorder is self else None,
        )
        call.emit("started")
        recorder_token = ACTIVE_RECORDER.set(self)
        call_token = ACTIVE_CALL.set(call)
        try:
            yield call
        except BaseException as error:
            if self.failure is None:
                if not call.responses:
                    call.receive(
                        {"availability": "no_response", "error_type": type(error).__name__}
                    )
                call.emit("finished", outcome="error", error_type=type(error).__name__)
            self.check()
            raise
        else:
            self.check()
            if not call.responses:
                call.receive({"availability": "unsupported_client"})
            call.emit("finished", outcome="returned")
        finally:
            ACTIVE_CALL.reset(call_token)
            ACTIVE_RECORDER.reset(recorder_token)


@contextmanager
def observation_session(recorder: ObservationRecorder | None) -> Iterator[None]:
    token = ACTIVE_RECORDER.set(recorder)
    try:
        yield
        if recorder:
            recorder.check()
    finally:
        ACTIVE_RECORDER.reset(token)


@contextmanager
def observe_call(
    purpose: str,
    model: str | None = None,
    *,
    recorder: ObservationRecorder | None = None,
    **coordinates: Any,
) -> Iterator[ObservedCall | None]:
    recorder = recorder or ACTIVE_RECORDER.get()
    if recorder is None:
        yield None
    else:
        with recorder.call(purpose, model, **coordinates) as call:
            yield call


def observation_refs(call: ObservedCall | None) -> list[dict[str, Any]] | None:
    return list(call.references) if call else None


def capture_response(
    response: Any,
    *,
    provider: str,
    model: str,
    account_usage: Callable[[], None],
    response_format: str = "openai",
    extra_reasoning: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Save before response processing; a failed write must not erase paid usage."""
    call = ACTIVE_CALL.get()
    if call is None:
        return
    try:
        payload = response_observation(response, provider, model, response_format)
        if extra_reasoning:
            payload["reasoning"].update(extra_reasoning())
            kinds = reasoning_kinds(payload["reasoning"])
            payload["reasoning_kinds"] = kinds
            if set(kinds) & {"text", "summary"}:
                payload["availability"] = "returned"
        call.receive(payload)
    except Exception as error:
        failure = (
            error
            if isinstance(error, ObservationPersistenceError)
            else ObservationPersistenceError(str(error))
        )
        call.recorder.failure = failure
        try:
            account_usage()
        except Exception as accounting_error:
            failure.add_note(f"Response accounting/validation also failed: {accounting_error}")
        raise failure from error
