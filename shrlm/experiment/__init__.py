"""Experiment scaffolding: typed configuration and usage metering for the Self-Harness experiment."""

from shrlm.experiment.config import CONFIG_PATH, ExperimentConfig, identity_hash, load_config
from shrlm.experiment.errors import ExperimentError
from shrlm.experiment.usage import (
    STAGE_USAGE_FILE,
    StageMeter,
    UsageTotals,
    aggregate_manifest_usage,
    read_stage_usage,
    snapshot_usage,
)

__all__ = [
    "CONFIG_PATH",
    "STAGE_USAGE_FILE",
    "ExperimentConfig",
    "ExperimentError",
    "StageMeter",
    "UsageTotals",
    "aggregate_manifest_usage",
    "identity_hash",
    "load_config",
    "read_stage_usage",
    "snapshot_usage",
]
