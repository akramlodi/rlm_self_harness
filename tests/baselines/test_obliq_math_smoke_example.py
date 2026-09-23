import json
from types import SimpleNamespace

import pytest

import examples.obliq_bench_math_smoke as smoke


def entry(instance_id: str, ndcg: float | None, *, passed: bool = True) -> dict:
    detail = "BudgetExceededError" if ndcg is None else f"ndcg@10={ndcg:.3f} hits={int(passed)}/1"
    cause = None if passed else ("resource_terminated" if ndcg is None else "mixed_set_error")
    return {
        "run_id": f"{instance_id}__a01",
        "instance_id": instance_id,
        "passed": passed,
        "cause": cause,
        "cost": 0.01,
        "input_tokens": 100,
        "output_tokens": 10,
        "execution_time": 1.0,
        "verdict": {
            "passed": passed,
            "cause": cause,
            "gold": '["id_a"]',
            "produced": '["id_a"]' if passed else "",
            "detail": detail,
        },
    }


def test_method_selector_supports_lambda_and_harness_alias() -> None:
    assert smoke.selected_method(smoke.parse_args([])) == "H0*"
    assert smoke.selected_method(smoke.parse_args(["--method", "lambda_rlm"])) == "lambda_rlm"
    assert smoke.selected_method(smoke.parse_args(["--harness", "H0*R"])) == "H0*R"
    assert smoke.selected_methods(smoke.parse_args(["--methods", "lambda_rlm,H0"])) == (
        "H0",
        "lambda_rlm",
    )

    with pytest.raises(SystemExit):
        smoke.parse_args(["--method", "lambda_rlm", "--harness", "H0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--method", "H0", "--methods", "H0,H0*"])
    with pytest.raises(ValueError, match="duplicates"):
        smoke.selected_methods(smoke.parse_args(["--methods", "H0,H0"]))


def test_lambda_method_uses_lambda_runner_without_calling_harness_runner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = {
        "id": "q1",
        "question": 'Closed-corpus retrieval. End with RANKED: ["id_a"].',
        "source_query": "source problem",
        "prompt": "prompt with [id_a]",
        "gold_relevant_ids": ["id_a"],
        "excluded_ids": [],
        "pool_size": 1,
    }
    monkeypatch.setattr(smoke, "load_obliq_bench_math", lambda **kwargs: [instance])

    captured = {}

    def fake_lambda_run(config, breaker):
        captured["config"] = config
        captured["breaker"] = breaker
        config.out_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(entries=[entry("q1", 1.0)])

    monkeypatch.setattr(smoke, "run_governed_lambda_round", fake_lambda_run)
    monkeypatch.setattr(
        smoke,
        "run_round",
        lambda config: pytest.fail("the harness runner must not execute for lambda_rlm"),
    )

    out_dir = tmp_path / "lambda"
    result = smoke.main(
        [
            "--live",
            "--method",
            "lambda_rlm",
            "--query-ids",
            "q1",
            "--candidate-pool-size",
            "1",
            "--config",
            "configs/experiment.toml",
            "--max-budget",
            "0.10",
            "--max-timeout",
            "30",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert result == 0
    assert captured["config"].instances == [instance]
    assert captured["config"].max_budget == 0.10
    assert captured["config"].max_timeout == 30.0
    assert captured["breaker"].caps.candidate_budget == 0.10
    assert (out_dir / "summary.json").is_file()


def test_matched_methods_share_instances_and_write_comparison(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = [
        {
            "id": "q1",
            "question": "question",
            "source_query": "source",
            "prompt": "prompt",
            "gold_relevant_ids": ["id_a"],
            "excluded_ids": [],
            "pool_size": 1,
        }
    ]
    monkeypatch.setattr(smoke, "load_obliq_bench_math", lambda **kwargs: instances)
    seen_instances = []

    def fake_round(config):
        seen_instances.append(config.instances)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        ndcg = 0.2 if config.harness.name == "H0" else 0.8
        return [entry("q1", ndcg, passed=ndcg >= 0.5)]

    def fake_lambda(config, breaker):
        seen_instances.append(config.instances)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(entries=[entry("q1", 0.6)])

    monkeypatch.setattr(smoke, "run_round", fake_round)
    monkeypatch.setattr(smoke, "run_governed_lambda_round", fake_lambda)

    out_dir = tmp_path / "matched"
    result = smoke.main(
        [
            "--live",
            "--methods",
            "H0,H0*,lambda_rlm",
            "--query-ids",
            "q1",
            "--candidate-pool-size",
            "1",
            "--config",
            "configs/experiment.toml",
            "--max-budget",
            "0.10",
            "--max-timeout",
            "30",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert result == 0
    assert seen_instances == [instances, instances, instances]
    assert (out_dir / "h0" / "summary.json").is_file()
    assert (out_dir / "h0_star" / "summary.json").is_file()
    assert (out_dir / "lambda_rlm" / "summary.json").is_file()

    comparison = json.loads((out_dir / "comparison.json").read_text())
    assert comparison["per_instance"][0]["H0"]["ndcg_at_10"] == 0.2
    assert comparison["per_instance"][0]["H0*"]["ndcg_at_10"] == 0.8
    assert comparison["per_instance"][0]["lambda_rlm"]["ndcg_at_10"] == 0.6
