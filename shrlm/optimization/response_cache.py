"""Text-response cache with optional portable provider observations."""

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResponseCache:
    path: str | None = None
    entries: dict[str, str] = field(default_factory=dict)
    observations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path) as handle:
                for line in handle:
                    if line.strip():
                        entry = json.loads(line)
                        key = entry["key"]
                        self.entries[key] = entry["response"]
                        self.observations.pop(key, None)
                        if "observations" in entry:
                            self.observations[key] = entry["observations"]

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def get_observations(self, key: str) -> list[dict[str, Any]] | None:
        return self.observations.get(key)

    def put(
        self, key: str, response: str, *, observations: list[dict[str, Any]] | None = None
    ) -> None:
        self.entries[key] = response
        self.observations.pop(key, None)
        entry: dict[str, Any] = {"key": key, "response": response}
        if observations is not None:
            self.observations[key] = observations
            entry["observations"] = observations
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as handle:
                json.dump(entry, handle)
                handle.write("\n")
