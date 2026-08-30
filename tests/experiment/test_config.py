"""Tests for shrlm.experiment.config: the single TOML owner of every
experiment parameter, its fail-fast loader, the identity hash, and the
factory helpers that feed the existing optimization constructors."""

import dataclasses
import re
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
from tests.conftest import AZURE_KIMI, OPENROUTER_QWEN, ProviderCase

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
    # t = 3 for the 2026-08-23 Qwen comparison run (was 15; identity key).
    assert (config.loop.m, config.loop.v, config.loop.k, config.loop.t) == (2, 4, 4, 3)
    assert config.loop.patience == 3
    assert config.decoding.temperature == 0.7
    assert config.decoding.top_p == 0.8
    assert config.decoding.top_k == 20
    assert config.decoding.min_p == 0.0
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
def test_shipped_backends_are_openrouter_qwen_for_all_roles(profile: str) -> None:
    config = load_config(profile)
    for role in CLIENT_ROLES:
        endpoint = getattr(config.backends, role)
        assert endpoint.backend == "openrouter"
        assert endpoint.model == "qwen/qwen3-30b-a3b-instruct-2507"


def test_eval_repetitions_must_be_a_positive_integer() -> None:
    operational = load_config().operational
    with pytest.raises(ValueError, match="eval_repetitions"):
        dataclasses.replace(operational, eval_repetitions=0)
    with pytest.raises(ValueError, match="eval_repetitions"):
        dataclasses.replace(operational, eval_repetitions=cast(int, 1.5))
    assert isinstance(operational, OperationalConfig)


def test_validation_workers_defaults_to_one_and_is_identity_exempt() -> None:
    """``operational.validation_workers`` scales wall clock, never behavior (R4, R5)."""
    config = load_config()
    assert config.operational.validation_workers == 1
    assert load_config("smoke").operational.validation_workers == 1
    base = identity_hash(config)
    wider = dataclasses.replace(config.operational, validation_workers=5)
    assert identity_hash(dataclasses.replace(config, operational=wider)) == base


def test_validation_workers_must_be_a_positive_integer(tmp_path: Path) -> None:
    operational = load_config().operational
    with pytest.raises(ValueError, match="validation_workers"):
        dataclasses.replace(operational, validation_workers=0)
    with pytest.raises(ValueError, match="validation_workers"):
        dataclasses.replace(operational, validation_workers=cast(int, True))
    text = shipped_text().replace("validation_workers = 1", "validation_workers = 0")
    with pytest.raises(ValueError, match="validation_workers"):
        load_config(path=write_config(tmp_path, text))


def test_smoke_may_override_validation_workers(tmp_path: Path) -> None:
    # The shipped [smoke.operational] table already exists; append the key to it
    # instead of declaring the table twice.
    text = shipped_text().replace(
        "[smoke.operational]\neval_repetitions = 1",
        "[smoke.operational]\neval_repetitions = 1\nvalidation_workers = 3",
    )
    path = write_config(tmp_path, text)
    assert load_config("smoke", path=path).operational.validation_workers == 3
    assert load_config("full", path=path).operational.validation_workers == 1
    assert identity_hash(load_config("full", path=path)) == identity_hash(load_config("full"))


def test_environment_selector_defaults_to_graphwalks_and_is_identity_covered(
    tmp_path: Path,
) -> None:
    config = load_config()
    assert config.loop.environment == "graphwalks"
    switched = dataclasses.replace(config.loop, environment="oolong_synth")
    assert identity_hash(dataclasses.replace(config, loop=switched)) != identity_hash(config)


def test_unknown_environment_selector_is_rejected(tmp_path: Path) -> None:
    text = shipped_text().replace('environment = "graphwalks"', 'environment = "oolong_pairs"')
    with pytest.raises(ValueError, match=r"\[loop\] environment must be one of"):
        load_config(path=write_config(tmp_path, text))


def test_oolong_environment_table_parses() -> None:
    oolong = load_config("full", path=Path("configs/experiment_oolong.toml")).environments.oolong
    assert oolong.synth.dataset_repo == "oolongbench/oolong-synth"
    assert 131072 in oolong.synth.context_lengths
    assert oolong.real.config_name == "dnd"
    assert oolong.real.n_check == 20


def test_real_check_cadence_is_identity_exempt() -> None:
    config = load_config()
    assert config.operational.real_check_every_n_rounds == 0
    base = identity_hash(config)
    with_check = dataclasses.replace(config.operational, real_check_every_n_rounds=3)
    assert identity_hash(dataclasses.replace(config, operational=with_check)) == base


def test_real_check_cadence_must_be_a_non_negative_integer() -> None:
    operational = load_config().operational
    with pytest.raises(ValueError, match="real_check_every_n_rounds"):
        dataclasses.replace(operational, real_check_every_n_rounds=-1)
    with pytest.raises(ValueError, match="real_check_every_n_rounds"):
        dataclasses.replace(operational, real_check_every_n_rounds=cast(int, 1.5))


def test_validation_run_workers_defaults_to_one_and_is_identity_exempt() -> None:
    """Run-level fan-out changes wall clock and request rate, never behavior.

    The parent still owns every shared file, appends every manifest line, and
    charges every run through one breaker, so a round produces the same runs at
    any worker count -- which is why this may change under an existing out-dir.
    """
    config = load_config()
    assert config.operational.validation_run_workers == 1
    assert load_config("smoke").operational.validation_run_workers == 1
    base = identity_hash(config)
    wider = dataclasses.replace(config.operational, validation_run_workers=4)
    assert identity_hash(dataclasses.replace(config, operational=wider)) == base


def test_validation_run_workers_must_be_a_positive_integer(tmp_path: Path) -> None:
    operational = load_config().operational
    with pytest.raises(ValueError, match="validation_run_workers"):
        dataclasses.replace(operational, validation_run_workers=0)
    with pytest.raises(ValueError, match="validation_run_workers"):
        dataclasses.replace(operational, validation_run_workers=cast(int, True))
    text = shipped_text().replace("validation_run_workers = 1", "validation_run_workers = 0")
    with pytest.raises(ValueError, match="validation_run_workers"):
        load_config(path=write_config(tmp_path, text))


def test_smoke_may_override_validation_run_workers(tmp_path: Path) -> None:
    text = shipped_text().replace(
        "[smoke.operational]\neval_repetitions = 1",
        "[smoke.operational]\neval_repetitions = 1\nvalidation_run_workers = 2",
    )
    path = write_config(tmp_path, text)
    assert load_config("smoke", path=path).operational.validation_run_workers == 2
    assert load_config("full", path=path).operational.validation_run_workers == 1
    assert identity_hash(load_config("full", path=path)) == identity_hash(load_config("full"))


def test_the_shipped_profile_keeps_subject_workers_at_one() -> None:
    """The two knobs multiply, so the shipped profile must not raise both.

    Run-level fan-out is the one that shortens a round; leaving subject workers
    above 1 alongside it would multiply the in-flight request rate against a
    provider that already logs retries at three concurrent runs.
    """
    assert load_config().operational.validation_workers == 1


def test_the_run_worker_count_reaches_the_evaluation_config() -> None:
    """A knob that stops at the config object changes nothing."""
    config = load_config()
    wider = dataclasses.replace(config.operational, validation_run_workers=3)
    kwargs = evaluation_config_kwargs(dataclasses.replace(config, operational=wider))
    assert kwargs["run_workers"] == 3


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


def test_unknown_key_inside_azure_foundry_table_raises(tmp_path: Path) -> None:
    text = shipped_text().replace("\nthinking = false\n", "\nthinking = false\neffort_level = 1\n")
    path = write_config(tmp_path, text)
    with pytest.raises(ValueError, match="effort_level"):
        load_config(path=path)


def with_azure_table(text: str, *, thinking: bool, reasoning_effort: str | None) -> str:
    """``text`` with the shipped ``[backends.azure_foundry]`` table rewritten."""
    anchor = "\nthinking = false\n"
    assert anchor in text
    lines = [f"thinking = {'true' if thinking else 'false'}"]
    if reasoning_effort is not None:
        lines.append(f'reasoning_effort = "{reasoning_effort}"')
    return text.replace(anchor, "\n" + "\n".join(lines) + "\n")


class TestReasoningEffort:
    """[backends.azure_foundry] reasoning_effort (R3-R5, KTD1-KTD2)."""

    def test_effort_reaches_sampling_args_top_level_without_chat_template_kwargs(
        self, tmp_path: Path
    ) -> None:
        text = with_azure_table(all_azure_roles_text(), thinking=True, reasoning_effort="medium")
        config = load_config(path=write_config(tmp_path, text))
        assert config.backends.azure_foundry is not None
        assert config.backends.azure_foundry.reasoning_effort == "medium"
        args = sampling_args(config, "runner")
        assert args["reasoning_effort"] == "medium"
        assert "chat_template_kwargs" not in args["extra_body"]
        assert "reasoning_effort" not in args["extra_body"]

    def test_thinking_true_without_effort_emits_neither_knob(self, tmp_path: Path) -> None:
        text = with_azure_table(all_azure_roles_text(), thinking=True, reasoning_effort=None)
        config = load_config(path=write_config(tmp_path, text))
        args = sampling_args(config, "runner")
        assert "reasoning_effort" not in args
        assert "chat_template_kwargs" not in args["extra_body"]

    def test_thinking_false_without_effort_keeps_instant_mode(self, tmp_path: Path) -> None:
        text = with_azure_table(all_azure_roles_text(), thinking=False, reasoning_effort=None)
        config = load_config(path=write_config(tmp_path, text))
        args = sampling_args(config, "runner")
        assert args["extra_body"]["chat_template_kwargs"] == {"thinking": False}
        assert "reasoning_effort" not in args

    def test_thinking_false_with_effort_is_refused_naming_both_keys(self, tmp_path: Path) -> None:
        text = with_azure_table(all_azure_roles_text(), thinking=False, reasoning_effort="low")
        with pytest.raises(ValueError, match="thinking = false.*reasoning_effort"):
            load_config(path=write_config(tmp_path, text))

    def test_effort_outside_the_enum_is_refused_listing_the_values(self, tmp_path: Path) -> None:
        text = with_azure_table(all_azure_roles_text(), thinking=True, reasoning_effort="xhigh")
        with pytest.raises(ValueError, match="low.*medium.*high.*none.*xhigh"):
            load_config(path=write_config(tmp_path, text))

    def test_openrouter_roles_never_carry_the_effort(self, tmp_path: Path) -> None:
        # Shipped config: every role openrouter; the dormant azure table sets an effort.
        text = with_azure_table(shipped_text(), thinking=True, reasoning_effort="high")
        config = load_config(path=write_config(tmp_path, text))
        for role in CLIENT_ROLES:
            assert "reasoning_effort" not in sampling_args(config, role)

    def test_explicit_none_hashes_identically_to_absent(self, tmp_path: Path) -> None:
        base = load_config(path=write_config(tmp_path, all_azure_roles_text()))
        explicit = dataclasses.replace(
            base,
            backends=dataclasses.replace(
                base.backends,
                azure_foundry=dataclasses.replace(
                    cast(Any, base.backends.azure_foundry), reasoning_effort=None
                ),
            ),
        )
        assert identity_hash(explicit) == identity_hash(base)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/experiment.toml",
        "configs/experiment_kimiK25.toml",
        "configs/experiment_ox.toml",
        "configs/experiment_oolong.toml",
        "configs/experiment_oolong_gptoss.toml",
    ],
)
@pytest.mark.parametrize("profile", ["full", "smoke"])
def test_every_shipped_config_loads_under_the_current_schema(
    config_path: str, profile: str
) -> None:
    config = load_config(profile, path=Path(config_path))
    assert len(identity_hash(config)) == 64


# ---------------------------------------------------------------------------
# Optional provider tables (KTD5)
# ---------------------------------------------------------------------------


def role_text(text: str, role: str, provider: ProviderCase) -> str:
    """``text`` with one role's shipped (openrouter Qwen) table switched to
    ``provider``'s table."""
    before = OPENROUTER_QWEN.role_table_text(role)
    assert before in text, f"no openrouter table found for role {role!r}"
    return text.replace(before, provider.role_table_text(role))


def azure_role_text(text: str, role: str) -> str:
    """``text`` with one role's table switched to the azure_foundry backend
    (the Kimi row of the provider matrix)."""
    return role_text(text, role, AZURE_KIMI)


def all_roles_text(provider: ProviderCase) -> str:
    """Shipped text with every role switched to ``provider``'s table."""
    text = shipped_text()
    for role in CLIENT_ROLES:
        text = role_text(text, role, provider)
    return text


def all_azure_roles_text() -> str:
    """Shipped text with every role switched to the azure_foundry backend."""
    return all_roles_text(AZURE_KIMI)


def test_matrix_row_table_text_is_the_shipped_openrouter_table() -> None:
    """The Qwen row's table text IS the shipped role table, so the matrix
    surgery replaces exactly what ships (an edit to either side fails here)."""
    text = shipped_text()
    for role in CLIENT_ROLES:
        assert OPENROUTER_QWEN.role_table_text(role) in text
    assert all_roles_text(OPENROUTER_QWEN) == text


def test_matrix_row_roles_load_and_route(provider: ProviderCase, tmp_path: Path) -> None:
    """Provider matrix (KTD5): every role switched to the row loads, selects
    the row's backend and model, carries pricing exactly when the row says
    so, and sends the row's ``extra_body``."""
    config = load_config(path=write_config(tmp_path, all_roles_text(provider)))
    for role in CLIENT_ROLES:
        endpoint = getattr(config.backends, role)
        assert (endpoint.backend, endpoint.model) == (provider.backend, provider.model)
        kwargs = backend_kwargs_for(config, role)
        assert kwargs["model_name"] == provider.model
        assert ("pricing" in kwargs) == (provider.pricing is not None)
        if provider.pricing is not None:
            assert set(kwargs["pricing"]) == set(provider.pricing)
        assert kwargs["sampling_args"]["extra_body"] == provider.expected_extra_body


def test_matrix_row_shipped_config_selects_the_row(provider: ProviderCase) -> None:
    """Each row's ``config_path`` is a shipped config whose runner is the row."""
    config = load_config(path=provider.config_path)
    assert (config.backends.runner.backend, config.backends.runner.model) == (
        provider.backend,
        provider.model,
    )
    kwargs = backend_kwargs_for(config, "runner")
    assert ("pricing" in kwargs) == (provider.pricing is not None)


def test_matrix_response_factory_yields_fresh_plain_objects(provider: ProviderCase) -> None:
    first, second = provider.response(), provider.response()
    assert first is not second
    assert first.usage is not second.usage
    # No row sets reasoning fields yet: neither attribute exists.
    assert not hasattr(first.choices[0].message, "reasoning_content")
    assert not hasattr(first.usage, "completion_tokens_details")
    assert provider.response(content="x").choices[0].message.content == "x"


_BRANCH_ON_PROVIDER = re.compile(r"if\s+provider" + r"\.id\b")


def test_no_test_body_branches_on_the_provider_id() -> None:
    """Divergence lives in matrix row fields or collection-time xfails (KTD5),
    never in a test body."""
    tests_dir = Path(__file__).resolve().parent.parent
    hits = [
        f"{path.relative_to(tests_dir.parent)}:{number}"
        for path in sorted(tests_dir.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if _BRANCH_ON_PROVIDER.search(line)
    ]
    assert hits == [], f"test bodies branch on provider.id: {hits}"


def test_absent_openrouter_table_loads_when_roles_use_azure_foundry(tmp_path: Path) -> None:
    path = write_config(tmp_path, drop_table(all_azure_roles_text(), "backends.openrouter"))
    config = load_config(path=path)
    assert config.backends.openrouter is None
    assert config.backends.runner.backend == "azure_foundry"


def test_absent_azure_foundry_table_with_azure_roles_raises(tmp_path: Path) -> None:
    """Thinking mode must be declared, never defaulted: an azure_foundry role
    with no [backends.azure_foundry] table would silently send no
    chat_template_kwargs and let Kimi default to thinking mode."""
    path = write_config(tmp_path, drop_table(all_azure_roles_text(), "backends.azure_foundry"))
    with pytest.raises(ValueError, match=r"backends\.azure_foundry") as excinfo:
        load_config(path=path)
    message = str(excinfo.value)
    assert "thinking" in message
    assert "runner" in message and "attributor" in message and "proposer" in message


def test_absent_azure_foundry_table_with_all_openrouter_roles_loads(tmp_path: Path) -> None:
    text = drop_table(shipped_text(), "backends.azure_foundry")
    config = load_config(path=write_config(tmp_path, text))
    assert config.backends.azure_foundry is None
    assert config.backends.runner.backend == "openrouter"


def test_openrouter_role_without_openrouter_table_loads_and_omits_provider(
    tmp_path: Path,
) -> None:
    text = drop_table(shipped_text(), "backends.openrouter")
    config = load_config(path=write_config(tmp_path, text))
    assert config.backends.runner.backend == "openrouter"
    assert config.backends.openrouter is None
    args = sampling_args(config, "runner")
    assert "provider" not in args["extra_body"]


def test_openrouter_role_with_provider_order_injects_provider(tmp_path: Path) -> None:
    text = azure_role_text(shipped_text(), "attributor").replace(
        "provider_order = []", 'provider_order = ["deepinfra"]'
    )
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
        dataclasses.replace(config, decoding=dataclasses.replace(config.decoding, top_k=None)),
        dataclasses.replace(
            config,
            backends=dataclasses.replace(
                config.backends,
                runner=dataclasses.replace(config.backends.runner, model="Kimi-K2.5"),
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
        dataclasses.replace(
            config,
            backends=dataclasses.replace(
                config.backends,
                azure_foundry=dataclasses.replace(
                    cast(Any, config.backends.azure_foundry), reasoning_effort="high"
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


def test_shipped_sampling_args_route_qwen_knobs_and_omit_provider() -> None:
    kwargs = backend_kwargs_for(load_config(), "runner")
    args = kwargs["sampling_args"]
    assert args["temperature"] == 0.7
    assert args["top_p"] == 0.8
    # top_k and min_p have no top-level OpenAI parameter, so they ride
    # extra_body; the empty provider order sends no provider restriction, and
    # no azure-only chat_template_kwargs ride an openrouter role.
    assert args["extra_body"]["top_k"] == 20
    assert args["extra_body"]["min_p"] == 0.0
    assert "provider" not in args["extra_body"]
    assert "chat_template_kwargs" not in args["extra_body"]
    # The OpenAI client owns the max_completion_tokens rename, not the config.
    assert args["max_tokens"] == 4096
    assert "max_completion_tokens" not in args


def test_shipped_openrouter_backend_kwargs_omit_pricing() -> None:
    """OpenRouter reports provider costs directly, so no pricing rides along."""
    config = load_config()
    for role in CLIENT_ROLES:
        kwargs = backend_kwargs_for(config, role)
        assert "pricing" not in kwargs
        assert kwargs["model_name"] == "qwen/qwen3-30b-a3b-instruct-2507"


def drop_optional_decoding_knobs(text: str) -> str:
    """Remove the top_k and min_p lines from the [decoding] table."""
    assert "top_k = 20\nmin_p = 0.0\n" in text
    return text.replace("top_k = 20\nmin_p = 0.0\n", "")


def test_azure_surgered_sampling_args_carry_instant_mode_and_omit_absent_knobs(
    tmp_path: Path,
) -> None:
    """The Kimi card specifies neither top_k nor min_p: with the knobs absent,
    nothing enters extra_body (the Foundry v1 route's handling of unknown body
    params is unconfirmed) and instant mode rides chat_template_kwargs."""
    text = drop_optional_decoding_knobs(all_azure_roles_text())
    config = load_config(path=write_config(tmp_path, text))
    args = backend_kwargs_for(config, "runner")["sampling_args"]
    assert "top_k" not in args["extra_body"]
    assert "min_p" not in args["extra_body"]
    assert "provider" not in args["extra_body"]
    # Instant-mode routing for the azure_foundry backend.
    assert args["extra_body"]["chat_template_kwargs"] == {"thinking": False}
    # The OpenAI client owns the max_completion_tokens rename, not the config.
    assert args["max_tokens"] == 4096
    assert "max_completion_tokens" not in args


def test_azure_surgered_backend_kwargs_carry_list_price_pricing(tmp_path: Path) -> None:
    config = load_config(path=write_config(tmp_path, all_azure_roles_text()))
    for role in CLIENT_ROLES:
        kwargs = backend_kwargs_for(config, role)
        assert kwargs["pricing"] == {"input_per_million": 0.10, "output_per_million": 0.30}
        assert kwargs["model_name"] == "Kimi-K2.5"


def test_surgered_absent_top_k_and_min_p_are_omitted(tmp_path: Path) -> None:
    text = drop_optional_decoding_knobs(shipped_text())
    config = load_config(path=write_config(tmp_path, text))
    assert config.decoding.top_k is None
    assert config.decoding.min_p is None
    args = sampling_args(config, "runner")
    assert "top_k" not in args["extra_body"]
    assert "min_p" not in args["extra_body"]


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
    assert evaluation.workers == config.operational.validation_workers == 1


def test_unknown_client_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown client role"):
        backend_kwargs_for(load_config(), "judge")


# ---------------------------------------------------------------------------
# [loop] initial_harness -- the registry harness the loop starts from
# ---------------------------------------------------------------------------


def _with_initial_harness(text: str, value: str) -> str:
    """Rewrite the shipped ``[loop] initial_harness`` line to ``value``."""
    assert 'initial_harness = "H0"\n' in text
    return text.replace('initial_harness = "H0"\n', f'initial_harness = "{value}"\n', 1)


def test_initial_harness_defaults_to_h0() -> None:
    assert load_config().loop.initial_harness == "H0"
    assert load_config("smoke").loop.initial_harness == "H0"


class TestGptOssOolongConfig:
    """configs/experiment_oolong_gptoss.toml (U2: R1, R2)."""

    PATH = Path("configs/experiment_oolong_gptoss.toml")
    KIMI_PATH = Path("configs/experiment_oolong.toml")

    @pytest.mark.parametrize("profile", ["full", "smoke"])
    def test_loads_with_every_role_on_azure_gpt_oss(self, profile: str) -> None:
        config = load_config(profile, path=self.PATH)
        for role in CLIENT_ROLES:
            endpoint = getattr(config.backends, role)
            assert (endpoint.backend, endpoint.model) == ("azure_foundry", "gpt-oss-120b")

    def test_every_role_carries_the_gpt_oss_rate_card(self) -> None:
        config = load_config("full", path=self.PATH)
        for role in CLIENT_ROLES:
            kwargs = backend_kwargs_for(config, role)
            assert kwargs["pricing"] == {"input_per_million": 0.15, "output_per_million": 0.60}
        assert config.pricing.promo == config.pricing.list_price

    def test_decoding_and_reasoning_knob_reach_sampling_args(self) -> None:
        config = load_config("full", path=self.PATH)
        args = sampling_args(config, "runner")
        assert args["temperature"] == 0
        assert args["top_p"] == 1.0
        assert args["max_tokens"] == 16384
        assert args["reasoning_effort"] == "medium"
        assert "chat_template_kwargs" not in args["extra_body"]

    def test_inherits_the_kimi_profile_except_the_swapped_tables(self) -> None:
        gptoss = load_config("full", path=self.PATH)
        kimi = load_config("full", path=self.KIMI_PATH)
        for section in ("caps", "splits", "loop", "promotion", "environments"):
            assert getattr(gptoss, section) == getattr(kimi, section)
        assert gptoss.operational == kimi.operational

    def test_identity_differs_from_the_kimi_profile(self) -> None:
        gptoss = load_config("full", path=self.PATH)
        kimi = load_config("full", path=self.KIMI_PATH)
        assert identity_hash(gptoss) != identity_hash(kimi)


def test_kimi_config_starts_from_h0_star() -> None:
    config = load_config("full", path=Path("configs/experiment_kimiK25.toml"))
    assert config.loop.initial_harness == "H0*"


def test_initial_harness_outside_the_registry_is_rejected_at_load(tmp_path: Path) -> None:
    path = write_config(tmp_path, _with_initial_harness(shipped_text(), "H9"))
    with pytest.raises(ValueError, match=r"initial_harness.*H0\*.*H9"):
        load_config("full", path=path)


def test_initial_harness_is_an_identity_key(tmp_path: Path) -> None:
    base = identity_hash(load_config())
    star = write_config(tmp_path, _with_initial_harness(shipped_text(), "H0*"))
    assert identity_hash(load_config("full", path=star)) != base


def test_explicit_h0_hashes_identically_to_the_default(tmp_path: Path) -> None:
    explicit = write_config(tmp_path, _with_initial_harness(shipped_text(), "H0"))
    assert identity_hash(load_config("full", path=explicit)) == identity_hash(load_config())
