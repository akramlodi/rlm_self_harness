import json
from dataclasses import replace
from pathlib import Path

import pytest

from examples.dataset_qualification import (
    DEFAULT_STUDY_CONFIG,
    QualificationThresholds,
    load_study_config,
    main,
    prepare_study_directory,
    qualification_verdict,
)


def method_summary(score: float, coverage: float = 1.0, failures: float = 0.0) -> dict:
    return {
        "mean_metric_all_runs": score,
        "metric_coverage_rate": coverage,
        "technical_failure_rate": failures,
    }


def test_shipped_qualification_config_is_strict_and_bounded() -> None:
    config = load_study_config(DEFAULT_STUDY_CONFIG)

    assert config.study.methods == ("H0", "H0*", "lambda_rlm")
    assert config.study.attempts == 2
    assert config.study.max_budget == pytest.approx(0.25)
    assert config.obliq_math.candidate_pool_size is None
    assert len(config.obliq_math.query_ids) == 10
    assert config.oolong_pairs.n == 10


def test_preflight_loads_no_dataset_and_spends_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "examples.dataset_qualification.load_obliq_bench_math",
        lambda **kwargs: pytest.fail("preflight loaded OBLIQ"),
    )
    monkeypatch.setattr(
        "examples.dataset_qualification.load_oolong_pairs",
        lambda **kwargs: pytest.fail("preflight loaded OOLONG-Pairs"),
    )

    assert main(["--config", str(DEFAULT_STUDY_CONFIG), "--out-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "worst-case spend: $30.00" in output
    assert "nothing loaded and nothing spent" in output
    assert list(tmp_path.iterdir()) == []


def test_qualification_requires_every_preregistered_gate() -> None:
    thresholds = QualificationThresholds(0.9, 0.1, 0.05, 0.1, 0.9)
    methods = {
        "H0": method_summary(0.2),
        "H0*": method_summary(0.5),
        "lambda_rlm": method_summary(0.3),
    }

    result = qualification_verdict(methods, thresholds)

    assert result["qualified"] is True
    degraded = dict(methods)
    degraded["lambda_rlm"] = method_summary(0.3, coverage=0.5)
    result = qualification_verdict(degraded, thresholds)
    assert result["qualified"] is False
    assert result["checks"]["metric_coverage"] is False


def test_study_directory_refuses_configuration_drift(tmp_path: Path) -> None:
    config = load_study_config(DEFAULT_STUDY_CONFIG)
    prepare_study_directory(tmp_path, config)
    recorded = json.loads((tmp_path / "study.json").read_text())
    assert recorded["study"]["attempts"] == 2
    assert len(recorded["study"]["experiment_config_sha256"]) == 64

    changed = replace(config, study=replace(config.study, attempts=3))
    with pytest.raises(ValueError, match="different qualification configuration"):
        prepare_study_directory(tmp_path, changed)
