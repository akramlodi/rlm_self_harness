"""Self-harnessing RLM: the ten editable surfaces and the starting harnesses."""

from shrlm.rlm_harness import (
    H0,
    H0_STAR,
    HARNESSES,
    INVARIANTS,
    SURFACES,
    Harness,
    SkillEntry,
    Surface,
    assemble_system_prompt,
    escape_braces,
)
from shrlm.runner import (
    HarnessedRLM,
    HarnessRun,
    acceptance_inputs,
    build_harnessed_rlm,
    check_harness,
    run_metrics,
)

__all__ = [
    "H0",
    "H0_STAR",
    "HARNESSES",
    "INVARIANTS",
    "SURFACES",
    "Harness",
    "HarnessRun",
    "HarnessedRLM",
    "SkillEntry",
    "Surface",
    "acceptance_inputs",
    "assemble_system_prompt",
    "build_harnessed_rlm",
    "check_harness",
    "escape_braces",
    "run_metrics",
]
