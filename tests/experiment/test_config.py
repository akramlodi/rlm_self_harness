"""Tests for shrlm.experiment.config: the single TOML owner of every
experiment parameter, its fail-fast loader, the identity hash, and the
factory helpers that feed the existing optimization constructors."""

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest

from shrlm.experiment.config import (
    CLIENT_ROLES,
    CONFIG_PATH,
    OperationalConfig,
    backend_kwargs_for,
    check_keys,
    evaluation_config_kwargs,
    identity_hash,
    load_config,
    promotion_config,
    proposer_config,
    round_config_kwargs,
    sampling_args,
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


def drop_table(text: str, header: str) -> str:
    """Remove one ``[table]`` (header, comments, and keys) from the text."""
    out: list[str] = []
    in_table = False
    dropped = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = stripped == f"[{header}]"
            if in_table:
                dropped = True
        if not in_table:
            out.append(line)
    assert dropped, f"no [{header}] table found"
    return "".join(out)


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
    assert config.decoding.temperature == 0.6
    assert config.decoding.top_p == 0.95
    assert config.decoding.top_k is None
    assert config.decoding.min_p is None
    assert config.decoding.max_output_tokens == 4096
    assert config.promotion.cost_band == (0.5, 1.25)
    assert config.environments.graphwalks.dataset_file_short == (
        "graphwalks_128k_and_shorter.parquet"
    )
    assert config.environments.oolong_pairs.task_ids == tuple(range(1, 21))


def test_operational_and_provider_defaults() -> None:
    config = load_config()
    assert config.operational.eval_repetitions == 3
    assert isinstance(config.operational.eval_repetitions, int)
    assert config.backends.openrouter is not None
    assert config.backends.openrouter.provider_order == ()
    assert config.backends.azure_foundry is not None
    assert config.backends.azure_foundry.thinking is False


@pytest.mark.parametrize("profile", ["full", "smoke"])
def test_shipped_backends_are_azure_foundry_kimi_for_all_roles(profile: str) -> None:
    config = load_config(profile)
    for role in CLIENT_ROLES:
        endpoint = getattr(config.backends, role)
        assert endpoint.backend == "azure_foundry"
        assert endpoint.model == "Kimi-K2.5"


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
    text = shipped_text().replace("temperature = 0.6", "temperature = 0.6\nbeam_width = 4")
    path = write_config(tmp_path, text)
    with pytest.raises(ValueError, match="beam_width"):
        load_config(path=path)


def test_unknown_key_inside_azure_foundry_table_raises(tmp_path: Path) -> None:
    text = shipped_text().replace("\nthinking = false\n", "\nthinking = false\nreasoning = 1\n")
    path = write_config(tmp_path, text)
    with pytest.raises(ValueError, match="reasoning"):
        load_config(path=path)


# ---------------------------------------------------------------------------
# Optional provider tables (KTD5)
# ---------------------------------------------------------------------------


def test_absent_openrouter_table_loads_when_roles_use_azure_foundry(tmp_path: Path) -> None:
    path = write_config(tmp_path, drop_table(shipped_text(), "backends.openrouter"))
    config = load_config(path=path)
    assert config.backends.openrouter is None
    assert config.backends.runner.backend == "azure_foundry"


def openrouter_runner_text() -> str:
    """Shipped text with the runner role switched to the openrouter backend."""
    return shipped_text().replace(
        '[backends.runner]\nbackend = "azure_foundry"\nmodel = "Kimi-K2.5"',
        '[backends.runner]\nbackend = "openrouter"\nmodel = "qwen/qwen3-30b-a3b-instruct-2507"',
    )


def test_openrouter_role_without_openrouter_table_loads_and_omits_provider(
    tmp_path: Path,
) -> None:
    text = drop_table(openrouter_runner_text(), "backends.openrouter")
    config = load_config(path=write_config(tmp_path, text))
    assert config.backends.runner.backend == "openrouter"
    assert config.backends.openrouter is None
    args = sampling_args(config, "runner")
    assert "provider" not in args["extra_body"]


def test_openrouter_role_with_provider_order_injects_provider(tmp_path: Path) -> None:
    text = openrouter_runner_text().replace("provider_order = []", 'provider_order = ["deepinfra"]')
    config = load_config(path=write_config(tmp_path, text))
    args = sampling_args(config, "runner")
    assert args["extra_body"]["provider"] == {
        "order": ["deepinfra"],
        "allow_fallbacks": True,
    }
    # azure_foundry roles do not pick up the openrouter routing.
    assert "provider" not in sampling_args(config, "attributor")["extra_body"]


def test_check_keys_optional_parameter() -> None:
    check_keys({"a": 1}, ("a",), "ctx")
    # A present optional key passes; a missing optional key passes.
    check_keys({"a": 1, "b": 2}, ("a",), "ctx", optional=("b",))
    check_keys({"a": 1}, ("a",), "ctx", optional=("b",))
    # An unknown key still fails, and a missing expected key still fails.
    with pytest.raises(ValueError, match="unknown key"):
        check_keys({"a": 1, "c": 3}, ("a",), "ctx", optional=("b",))
    with pytest.raises(ValueError, match="missing mandatory"):
        check_keys({"b": 2}, ("a",), "ctx", optional=("b",))


# ---------------------------------------------------------------------------
# Identity hash (R3; KTD3)
# ---------------------------------------------------------------------------


def test_identity_hash_changes_with_behavior_changing_values() -> None:
    config = load_config()
    base = identity_hash(config)
    assert len(base) == 64
    assert set(base) <= set("0123456789abcdef")
    variants = [
        dataclasses.replace(config, decoding=dataclasses.replace(config.decoding, temperature=0.9)),
        dataclasses.replace(config, decoding=dataclasses.replace(config.decoding, top_k=20)),
        dataclasses.replace(
            config,
            backends=dataclasses.replace(
                config.backends,
                runner=dataclasses.replace(config.backends.runner, model="Kimi-K2.5-alt"),
            ),
        ),
        dataclasses.replace(
            config,
            backends=dataclasses.replace(
                config.backends,
                azure_foundry=dataclasses.replace(
                    cast(Any, config.backends.azure_foundry), thinking=True
                ),
            ),
        ),
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
    relocated_cache = dataclasses.replace(
        config.operational, attribution_cache_path="/tmp/attribution.jsonl"
    )
    assert identity_hash(dataclasses.replace(config, operational=relocated_cache)) == base
    patient_loader = dataclasses.replace(config.operational, loader_timeout_seconds=9999.0)
    assert identity_hash(dataclasses.replace(config, operational=patient_loader)) == base
    cheaper = dataclasses.replace(
        config.pricing, promo=dataclasses.replace(config.pricing.promo, input_per_million=0.01)
    )
    assert identity_hash(dataclasses.replace(config, pricing=cheaper)) == base


def test_identity_hash_covers_eval_repetitions_in_both_directions() -> None:
    """The effective evaluation attempt count is identity-protected (KTD8).

    It decides how many runs an evaluation persists and what the eval summary
    claims was evaluated, so raising *or* lowering it under an existing
    experiment directory must move the hash -- lowering in particular would
    otherwise reuse a higher count's persisted attempts while rewriting the
    summary as if the smaller plan had been run.
    """
    config = load_config()
    base = identity_hash(config)
    assert config.operational.eval_repetitions == 3
    lowered = dataclasses.replace(
        config, operational=dataclasses.replace(config.operational, eval_repetitions=1)
    )
    raised = dataclasses.replace(
        config, operational=dataclasses.replace(config.operational, eval_repetitions=7)
    )
    assert identity_hash(lowered) != base
    assert identity_hash(raised) != base
    assert identity_hash(lowered) != identity_hash(raised)


def test_smoke_profile_still_shrinks_eval_repetitions() -> None:
    """Identity protection must not cost the smoke profile its scale key (R2)."""
    assert load_config("smoke").operational.eval_repetitions == 1
    assert load_config("full").operational.eval_repetitions == 3


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
    assert promotion.cost_band == Band(lower=0.5, upper=1.25)
    # Absent sub_call_band keeps the unconstrained default.
    assert promotion.sub_call_band == Band()


def test_validation_caps_factory() -> None:
    caps = validation_caps(load_config())
    assert isinstance(caps, ValidationCaps)
    assert caps.max_budget == 0.5
    assert caps.max_timeout == 1800.0
    assert caps.candidate_budget == 60.0
    assert caps.max_depth == 3
    assert caps.max_iterations == 30


def test_proposer_config_factory() -> None:
    proposer = proposer_config(load_config())
    assert isinstance(proposer, ProposerConfig)
    assert proposer.k == 4


def test_shipped_sampling_args_carry_instant_mode_and_omit_absent_knobs() -> None:
    kwargs = backend_kwargs_for(load_config(), "runner")
    args = kwargs["sampling_args"]
    assert args["temperature"] == 0.6
    assert args["top_p"] == 0.95
    # Absent decoding knobs never enter extra_body (the Foundry v1 route's
    # handling of unknown body params is unconfirmed).
    assert "top_k" not in args["extra_body"]
    assert "min_p" not in args["extra_body"]
    assert "provider" not in args["extra_body"]
    # Instant-mode routing for the azure_foundry backend.
    assert args["extra_body"]["chat_template_kwargs"] == {"thinking": False}
    # The OpenAI client owns the max_completion_tokens rename, not the config.
    assert args["max_tokens"] == 4096
    assert "max_completion_tokens" not in args


def test_azure_foundry_backend_kwargs_carry_list_price_pricing() -> None:
    config = load_config()
    for role in CLIENT_ROLES:
        kwargs = backend_kwargs_for(config, role)
        assert kwargs["pricing"] == {"input_per_million": 0.60, "output_per_million": 3.00}
        assert kwargs["model_name"] == "Kimi-K2.5"


def test_surgered_top_k_and_min_p_still_route_via_extra_body(tmp_path: Path) -> None:
    text = shipped_text().replace("temperature = 0.6", "temperature = 0.6\ntop_k = 20\nmin_p = 0.0")
    config = load_config(path=write_config(tmp_path, text))
    args = sampling_args(config, "runner")
    assert args["extra_body"]["top_k"] == 20
    assert args["extra_body"]["min_p"] == 0.0


def test_sampling_args_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown client role"):
        sampling_args(load_config(), "judge")


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
    assert round_config.backend == "azure_foundry"
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
    assert evaluation.backend == "azure_foundry"


def test_unknown_client_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown client role"):
        backend_kwargs_for(load_config(), "judge")
