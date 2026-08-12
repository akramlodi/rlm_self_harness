"""Tests for shrlm.experiment.config: the single TOML owner of every
experiment parameter, its fail-fast loader, the identity hash, and the
factory helpers that feed the existing optimization constructors."""

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest

from shrlm.experiment.config import (
    CONFIG_PATH,
    OperationalConfig,
    backend_kwargs_for,
    evaluation_config_kwargs,
    identity_hash,
    load_config,
    promotion_config,
    proposer_config,
    round_config_kwargs,
    validation_caps,
)
from shrlm.optimization.costs import ValidationCaps
from shrlm.optimization.driver import RoundConfig
from shrlm.optimization.promotion import Band, PromotionConfig
from shrlm.optimization.proposal import ProposerConfig
from shrlm.optimization.validation import EvaluationConfig, ValidationSplits

DUMMY_VERIFIER = cast(Any, lambda *args, **kwargs: None)


def shipped_text() -> str:
    return CONFIG_PATH.read_text()


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    return path


def drop_caps_key(text: str, key: str) -> str:
    """Remove one ``key = value`` line from the [caps] table only."""
    out: list[str] = []
    in_caps = False
    dropped = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_caps = stripped == "[caps]"
        if in_caps and not dropped and stripped.startswith(f"{key} "):
            dropped = True
            continue
        out.append(line)
    assert dropped, f"no line for {key} found in [caps]"
    return "".join(out)


# ---------------------------------------------------------------------------
# Loading the shipped file
# ---------------------------------------------------------------------------


def test_full_profile_ships_draft_defaults() -> None:
    config = load_config()
    assert config.profile == "full"
    assert config.splits.n_in == 24
    assert config.splits.n_ho == 40
    assert config.splits.test_short == 40
    assert config.splits.test_long == 150
    assert (config.loop.m, config.loop.v, config.loop.k, config.loop.t) == (2, 4, 4, 15)
    assert config.loop.patience == 3
    assert config.decoding.temperature == 0.7
    assert config.decoding.top_p == 0.8
    assert config.decoding.top_k == 20
    assert config.decoding.max_output_tokens == 4096
    assert config.promotion.cost_band == (0.5, 2.0)
    assert config.environments.graphwalks.dataset_file_short == (
        "graphwalks_128k_and_shorter.parquet"
    )
    assert config.environments.oolong_pairs.task_ids == tuple(range(1, 21))


def test_operational_and_provider_defaults() -> None:
    config = load_config()
    assert config.operational.eval_repetitions == 3
    assert isinstance(config.operational.eval_repetitions, int)
    assert config.backends.openrouter.provider_order == ()
    assert config.backends.runner.model == "qwen/qwen3-30b-a3b-instruct-2507"


def test_eval_repetitions_must_be_a_positive_integer() -> None:
    operational = load_config().operational
    with pytest.raises(ValueError, match="eval_repetitions"):
        dataclasses.replace(operational, eval_repetitions=0)
    with pytest.raises(ValueError, match="eval_repetitions"):
        dataclasses.replace(operational, eval_repetitions=cast(int, 1.5))
    assert isinstance(operational, OperationalConfig)


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        load_config("dev")


# ---------------------------------------------------------------------------
# Smoke profile (R2)
# ---------------------------------------------------------------------------


def test_smoke_overrides_scale_counts_only() -> None:
    full = load_config("full")
    smoke = load_config("smoke")
    assert smoke.splits.n_in == 3
    assert smoke.splits.n_ho == 3
    assert smoke.splits.test_short == 4
    assert smoke.splits.test_long == 2
    assert (smoke.loop.m, smoke.loop.v, smoke.loop.k, smoke.loop.t) == (1, 1, 2, 1)
    assert smoke.environments.oolong_pairs.n_short == 4
    assert smoke.caps.max_budget == 0.5
    # Semantics are byte-identical to the full profile.
    assert smoke.decoding == full.decoding
    assert smoke.promotion == full.promotion
    assert smoke.backends == full.backends
    assert smoke.loop.patience == full.loop.patience
    assert smoke.environments.graphwalks == full.environments.graphwalks
    assert smoke.splits.seed == full.splits.seed


def test_smoke_key_outside_scale_set_is_rejected(tmp_path: Path) -> None:
    text = shipped_text() + "\n[smoke.decoding]\ntemperature = 0.1\n"
    path = write_config(tmp_path, text)
    with pytest.raises(ValueError, match=r"decoding\.temperature"):
        load_config("smoke", path=path)


# ---------------------------------------------------------------------------
# Fail-fast validation (R12, unknown keys)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["max_budget", "max_timeout", "candidate_budget"])
def test_missing_mandatory_cap_raises_naming_the_key(tmp_path: Path, key: str) -> None:
    path = write_config(tmp_path, drop_caps_key(shipped_text(), key))
    with pytest.raises(ValueError, match=key):
        load_config(path=path)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    path = write_config(tmp_path, shipped_text() + "\n[bogus]\nx = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(path=path)


def test_unknown_key_inside_a_table_raises(tmp_path: Path) -> None:
    # tomllib itself rejects a duplicated [decoding] header, so append the key
    # by rewriting the original table line instead.
    text = shipped_text().replace("temperature = 0.7", "temperature = 0.7\nbeam_width = 4")
    path = write_config(tmp_path, text)
    with pytest.raises(ValueError, match="beam_width"):
        load_config(path=path)


# ---------------------------------------------------------------------------
# Identity hash (R3; KTD3)
# ---------------------------------------------------------------------------


def test_identity_hash_changes_with_behavior_changing_values() -> None:
    config = load_config()
    base = identity_hash(config)
    variants = [
        dataclasses.replace(config, decoding=dataclasses.replace(config.decoding, temperature=0.9)),
        dataclasses.replace(config, caps=dataclasses.replace(config.caps, attempts=2)),
        dataclasses.replace(config, loop=dataclasses.replace(config.loop, v=5)),
        dataclasses.replace(config, loop=dataclasses.replace(config.loop, patience=4)),
        dataclasses.replace(
            config, promotion=dataclasses.replace(config.promotion, tau_regression=0.5)
        ),
        dataclasses.replace(
            config,
            environments=dataclasses.replace(
                config.environments,
                graphwalks=dataclasses.replace(
                    config.environments.graphwalks, dataset_revision="deadbeef"
                ),
            ),
        ),
        dataclasses.replace(config, splits=dataclasses.replace(config.splits, seed=7)),
    ]
    hashes = [identity_hash(variant) for variant in variants]
    assert base not in hashes
    assert len(set(hashes)) == len(hashes)


def test_identity_hash_ignores_operational_values() -> None:
    config = load_config()
    base = identity_hash(config)
    pricier_gpu = dataclasses.replace(config.gpu_scenarios[0], hourly_rate_usd=9.99)
    assert identity_hash(dataclasses.replace(config, gpu_scenarios=(pricier_gpu,))) == base
    more_repetitions = dataclasses.replace(config.operational, eval_repetitions=7)
    assert identity_hash(dataclasses.replace(config, operational=more_repetitions)) == base
    cheaper = dataclasses.replace(
        config.pricing, promo=dataclasses.replace(config.pricing.promo, input_per_million=0.01)
    )
    assert identity_hash(dataclasses.replace(config, pricing=cheaper)) == base


def test_identity_hash_differs_between_profiles() -> None:
    assert identity_hash(load_config("full")) != identity_hash(load_config("smoke"))


# ---------------------------------------------------------------------------
# Factory helpers against the real constructors
# ---------------------------------------------------------------------------


def test_promotion_config_factory() -> None:
    promotion = promotion_config(load_config())
    assert isinstance(promotion, PromotionConfig)
    assert promotion.tau_regression == 0.0
    assert promotion.tau_improvement == 0.0
    assert promotion.cost_band == Band(lower=0.5, upper=2.0)
    # Absent sub_call_band keeps the unconstrained default.
    assert promotion.sub_call_band == Band()


def test_validation_caps_factory() -> None:
    caps = validation_caps(load_config())
    assert isinstance(caps, ValidationCaps)
    assert caps.max_budget == 5.0
    assert caps.max_timeout == 1800.0
    assert caps.candidate_budget == 60.0
    assert caps.max_depth == 1
    assert caps.max_iterations == 30


def test_proposer_config_factory() -> None:
    proposer = proposer_config(load_config())
    assert isinstance(proposer, ProposerConfig)
    assert proposer.k == 4


def test_sampling_args_route_top_k_via_extra_body() -> None:
    args = backend_kwargs_for(load_config(), "runner")["sampling_args"]
    assert args["temperature"] == 0.7
    assert args["top_p"] == 0.8
    assert args["extra_body"]["top_k"] == 20
    assert args["extra_body"]["min_p"] == 0.0
    # Empty provider order means no provider restriction is sent.
    assert "provider" not in args["extra_body"]
    # The OpenAI client owns the max_completion_tokens rename, not the config.
    assert args["max_tokens"] == 4096
    assert "max_completion_tokens" not in args


def test_round_config_kwargs_accepted_by_round_config(tmp_path: Path) -> None:
    config = load_config("smoke")
    round_config = RoundConfig(
        round_index=0,
        harness=cast(Any, object()),
        instances=[{"id": "i0", "prompt": "p"}],
        verifier=DUMMY_VERIFIER,
        out_dir=tmp_path,
        **round_config_kwargs(config),
    )
    assert round_config.backend == "openrouter"
    assert round_config.attempts == config.caps.attempts
    assert round_config.max_budget == 0.5
    assert round_config.max_timeout == 300.0
    assert round_config.backend_kwargs["model_name"] == config.backends.runner.model


def test_evaluation_config_kwargs_accepted_by_evaluation_config(tmp_path: Path) -> None:
    config = load_config("smoke")
    splits = ValidationSplits(
        heldin=[{"id": "a", "prompt": "p"}],
        heldout=[{"id": "b", "prompt": "q"}],
    )
    evaluation = EvaluationConfig(
        splits=splits,
        verifier=DUMMY_VERIFIER,
        out_dir=tmp_path,
        round_index=0,
        **evaluation_config_kwargs(config),
    )
    assert evaluation.repetitions == config.loop.v == 1
    assert evaluation.caps.candidate_budget == 3.0
    assert evaluation.backend == "openrouter"


def test_unknown_client_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown client role"):
        backend_kwargs_for(load_config(), "judge")
