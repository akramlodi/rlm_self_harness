"""Both-modes end-to-end sub-verification ablation test (U4).

This is the controlled comparison the ablation depends on, proven offline: one
persisted MockLM round is mined twice -- grounded (a stub SubVerifier returning
mixed verdicts) into the round root, and ablated (``sub_verifier=None``) into
``bundles/ablated/`` -- over ONE shared on-disk attribution cache at the round
root. The two bundles must coexist, audit clean against the same shared round
artifacts, diverge in bundle id, and differ in their persisted configs by
EXACTLY the mode flag, so any future config field that accidentally varies
with the mode breaks this file by construction.

The round's two failing runs are built to hit both grounding outcomes:

* ``inst-childfail`` -- two llm_query children, one of which the stub
  sub-verifier rejects: grounded mode derives ``failing_level=child`` as a
  checkable fact, ablated mode asks the attributor.
* ``inst-uncheck`` -- the same tree shape but every child verdict is None
  (uncheckable in isolation): UNDETERMINED, so even the grounded pass falls
  back to the ungrounded prompt variant for this record.

Cache semantics follow from the bytes: ``inst-childfail``'s digest carries
True/False in its sub_verdict column when grounded and ``n/a`` when ablated,
so the two passes' cache keys differ and the ablated pass pays its own LM
call. ``inst-uncheck``'s digest and prompt are byte-identical across modes,
so the ablated pass legitimately replays the grounded pass's cached response
-- that hit is the cache doing its job on identical inputs, not
cross-contamination.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from shrlm.optimization.attribution import AttributionCache
from shrlm.optimization.audit import AuditReport, audit_round, bundle_dir_for
from shrlm.optimization.driver import round_dir
from shrlm.optimization.mining import MiningResult
from shrlm.optimization.taxonomy import FailingLevel
from tests.mock_lm import MockLM
from tests.optimization.test_driver import (
    CANNED_ATTRIBUTION,
    BoomVerifier,
    final,
    make_miner,
    make_round_config,
    run_dual_mined_round,
)

# ---------------------------------------------------------------------------
# The round: one pass, two failures with sub-calls
# ---------------------------------------------------------------------------


def ablation_instances() -> list[dict[str, Any]]:
    return [
        {"id": "inst-pass", "question": "q pass", "prompt": "context pass", "gold": "RIGHT"},
        {"id": "inst-childfail", "question": "q cf", "prompt": "context cf", "gold": "RIGHT"},
        {"id": "inst-uncheck", "question": "q un", "prompt": "context un", "gold": "RIGHT"},
    ]


def delegating_response() -> str:
    """Root iteration 1: fan out to two plain-LM sub-calls."""
    return (
        "Delegating to two sub-calls.\n"
        "```repl\n"
        "part_one = llm_query('summarize part one')\n"
        "part_two = llm_query('summarize part two')\n"
        "print(part_one)\n"
        "print(part_two)\n"
        "```"
    )


def ablation_script() -> list[str]:
    """Scripted runtime turns, in pop order: each failing run burns one
    delegating iteration (whose two llm_query calls pop the next two entries)
    and then commits a wrong answer."""
    return [
        final("RIGHT"),  # inst-pass, iteration 1
        delegating_response(),  # inst-childfail, iteration 1
        "part one says A",  # ... its first llm_query child
        "part two says B",  # ... its second llm_query child
        final("WRONG"),  # inst-childfail, iteration 2
        delegating_response(),  # inst-uncheck, iteration 1
        "part one says A",
        "part two says B",
        final("WRONG"),  # inst-uncheck, iteration 2
    ]


class MixedSubVerifier:
    """The grounded mode's stub: mixed verdicts for one tree, all-None for the
    other. ``inst-childfail``'s 'part one' child is judged wrong (False) and
    its 'part two' child right (True), so ``derive_failing_level`` lands on a
    checkable CHILD; every ``inst-uncheck`` child is uncheckable (None), so
    that record stays UNDETERMINED and ungrounded even in grounded mode."""

    def __call__(self, instance: dict[str, Any], node: Any) -> bool | None:
        if instance["id"] == "inst-uncheck":
            return None
        prompt = node.prompt if isinstance(node.prompt, str) else str(node.prompt)
        return "part one" not in prompt


# The attributor responses. The grounded variant must NOT carry failing_level
# (validate() never reads it there; the level is the sub-verifier's fact) and
# the ungrounded variant MUST carry it (validate() rejects its absence) --
# CANNED_ATTRIBUTION, the driver tests' canned ungrounded response, serves as
# the latter.
GROUNDED_ATTRIBUTION = (
    "```json\n"
    + json.dumps(
        {
            "causal_status": "causal",
            "agent_mechanism": "lossy_aggregation",
            "evidence_node_ids": ["r"],
            "symptom_summary": "a sub-call returned a wrong local result",
        }
    )
    + "\n```"
)


class AblationRound:
    """Everything the assertions need from the two mining passes."""

    def __init__(
        self,
        round_path: Path,
        grounded_lm: MockLM,
        ablated_lm: MockLM,
        result_grounded: MiningResult,
        report_grounded: AuditReport,
        result_ablated: MiningResult,
        report_ablated: AuditReport,
    ):
        self.round_path = round_path
        self.grounded_lm = grounded_lm
        self.ablated_lm = ablated_lm
        self.result_grounded = result_grounded
        self.report_grounded = report_grounded
        self.result_ablated = result_ablated
        self.report_ablated = report_ablated

    @property
    def ablated_dir(self) -> Path:
        return bundle_dir_for(self.round_path, "ablated")

    def bundle_config(self, triplet_dir: Path) -> dict[str, Any]:
        return json.loads((triplet_dir / "bundle.json").read_text())["config"]

    def attributions_by_id(self, triplet_dir: Path) -> dict[str, dict[str, Any]]:
        lines = (triplet_dir / "attributions.jsonl").read_text().splitlines()
        return {entry["instance_id"]: entry for entry in map(json.loads, lines)}


@pytest.fixture(scope="module")
def ablation(tmp_path_factory: pytest.TempPathFactory) -> AblationRound:
    """One MockLM round, mined grounded into the round root and ablated into
    ``bundles/ablated/``, both passes over one shared round-root cache.

    Module-scoped: every test below reads the two persisted passes without
    mutating them, and the round is the expensive part of this file.
    """
    tmp_path = tmp_path_factory.mktemp("ablation")
    with pytest.MonkeyPatch.context() as monkeypatch:
        config = make_round_config(
            tmp_path, instances=ablation_instances(), max_budget=None, max_iterations=3
        )
        round_path = round_dir(tmp_path, config.round_index)
        cache_file = round_path / "attribution_cache.jsonl"

        # Grounded pass first: mines from the fresh round, pays both
        # attribution calls. The ablated pass may then call out only for the
        # record whose digest bytes actually changed with the mode.
        grounded_miner = make_miner(
            BoomVerifier(),
            responses=[GROUNDED_ATTRIBUTION, CANNED_ATTRIBUTION],
            sub_verifier=MixedSubVerifier(),
            cache=AttributionCache(path=str(cache_file)),
        )
        outcome = run_dual_mined_round(
            monkeypatch,
            script=ablation_script(),
            first_config=config,
            first_miner=grounded_miner,
            second_config=make_round_config(
                tmp_path, instances=ablation_instances(), max_budget=None
            ),
            second_miner_factory=lambda: make_miner(
                BoomVerifier(),
                responses=[CANNED_ATTRIBUTION],
                cache=AttributionCache(path=str(cache_file)),
            ),
            second_label="ablated",
        )

    return AblationRound(
        round_path=round_path,
        grounded_lm=grounded_miner.attributor.lm,
        ablated_lm=outcome.second_miner.attributor.lm,
        result_grounded=outcome.first_result,
        report_grounded=outcome.first_report,
        result_ablated=outcome.second_result,
        report_ablated=outcome.second_report,
    )


# ---------------------------------------------------------------------------
# 1. Identity: the modes mint different bundles, correctly stamped
# ---------------------------------------------------------------------------


class TestModeIdentity:
    def test_bundle_ids_diverge(self, ablation):
        assert ablation.result_grounded.bundle.bundle_id != ablation.result_ablated.bundle.bundle_id

    def test_modes_are_stamped_true_and_false(self, ablation):
        assert ablation.result_grounded.bundle.config.sub_verifier_enabled is True
        assert ablation.result_ablated.bundle.config.sub_verifier_enabled is False
        # And as persisted, not just in memory.
        assert ablation.bundle_config(ablation.round_path)["sub_verifier_enabled"] is True
        assert ablation.bundle_config(ablation.ablated_dir)["sub_verifier_enabled"] is False

    def test_configs_differ_in_exactly_the_mode_flag(self, ablation):
        """The controlled-comparison guard: 'nothing else varies'. Any config
        field that starts varying with the mode -- a prompt pin, a cache path,
        a version -- breaks this assertion by construction."""
        grounded = ablation.bundle_config(ablation.round_path)
        ablated = ablation.bundle_config(ablation.ablated_dir)
        assert set(grounded) == set(ablated)
        differing = {key for key in grounded if grounded[key] != ablated[key]}
        assert differing == {"sub_verifier_enabled"}


# ---------------------------------------------------------------------------
# 2. Accounting: level_grounded and n_ungrounded are correct per mode
# ---------------------------------------------------------------------------


class TestGroundingAccounting:
    def test_grounded_pass_grounds_the_checkable_record(self, ablation):
        records = {record.instance_id: record for record in ablation.result_grounded.records}
        childfail = records["inst-childfail"]
        assert childfail.level_grounded is True
        assert childfail.signature is not None
        # The level is derived from the False child verdict, not asked for.
        assert childfail.signature.failing_level is FailingLevel.CHILD
        assert False in childfail.sub_verdicts.values()

    def test_grounded_pass_leaves_the_all_none_record_ungrounded(self, ablation):
        records = {record.instance_id: record for record in ablation.result_grounded.records}
        uncheck = records["inst-uncheck"]
        assert uncheck.level_grounded is False
        assert set(uncheck.sub_verdicts.values()) == {None}
        # The attributor supplied the level (the canned ungrounded response).
        assert uncheck.signature is not None
        assert uncheck.signature.failing_level is FailingLevel.ROOT
        assert ablation.result_grounded.bundle.integrity.n_ungrounded == 1

    def test_ablated_pass_grounds_nothing(self, ablation):
        assert len(ablation.result_ablated.records) == 2
        for record in ablation.result_ablated.records:
            assert record.level_grounded is False
            assert record.sub_verdicts == {}
        integrity = ablation.result_ablated.bundle.integrity
        assert integrity.n_ungrounded == ablation.result_ablated.bundle.totals.n_failures == 2

    def test_audit_grounding_coverage_separates_the_modes(self, ablation):
        # Coverage counts records with at least one non-null child verdict:
        # 1 of 2 grounded (the all-None record does not count), 0 of 2 ablated.
        assert ablation.report_grounded.stats.grounding_coverage == pytest.approx(0.5)
        assert ablation.report_ablated.stats.grounding_coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Audits: both walks clean, all prompt variants and the shared cache resolve
# ---------------------------------------------------------------------------


class TestDualAudits:
    def test_both_pipeline_audits_pass(self, ablation):
        assert ablation.report_grounded.ok
        assert ablation.report_ablated.ok

    def test_both_walks_reaudit_clean_from_disk_alone(self, ablation):
        assert audit_round(ablation.round_path.parent, 1, bundle_label=None).ok
        assert audit_round(ablation.round_path.parent, 1, bundle_label="ablated").ok

    def test_grounded_pass_mixed_prompt_variants_all_resolve_as_files(self, ablation):
        """The grounded pass renders TWO variants -- the grounded prompt for
        the checkable record and the ungrounded fallback for the all-None one
        -- and every variant must be persisted content-addressed."""
        prompts = ablation.result_grounded.attributor_prompts
        assert len(prompts) == 2
        for sha, text in prompts.items():
            prompt_path = ablation.round_path / f"attributor_prompt_{sha[:16]}.txt"
            assert prompt_path.is_file()
            assert prompt_path.read_text() == text
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == sha
        # The ablated pass's single (ungrounded) variant is one of the two.
        assert set(ablation.result_ablated.attributor_prompts) <= set(prompts)

    def test_shared_cache_path_link_audits_in_both_walks(self, ablation):
        grounded = ablation.bundle_config(ablation.round_path)
        ablated = ablation.bundle_config(ablation.ablated_dir)
        # Both bundles stamp the identical round-root-relative path...
        assert grounded["attribution_cache_path"] == "attribution_cache.jsonl"
        assert ablated["attribution_cache_path"] == "attribution_cache.jsonl"
        assert (ablation.round_path / "attribution_cache.jsonl").is_file()
        # ...and both walks actually checked (and passed) that link.
        for report in (ablation.report_grounded, ablation.report_ablated):
            checked = {link.link: link for link in report.links}
            assert checked["attribution_cache"].checked == 1
            assert checked["attribution_cache"].failures == 0
            assert checked["attribution_prompt"].checked == 2


# ---------------------------------------------------------------------------
# 4. Cache isolation: keys separate the modes; identical bytes legitimately share
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    def test_lm_call_counts_pin_the_isolation_semantics(self, ablation):
        # Grounded pass: two failure records, two fresh attribution calls.
        assert ablation.grounded_lm._call_count == 2
        # Ablated pass: inst-childfail's digest bytes changed with the mode
        # (True/False sub_verdict column -> n/a) and its prompt variant
        # changed (grounded -> ungrounded), so its key misses and one fresh
        # call is paid; inst-uncheck's key hits (see below), costing nothing.
        assert ablation.ablated_lm._call_count == 1

    def test_grounded_record_is_never_served_across_modes(self, ablation):
        """For the record the grounded pass actually GROUNDED, both cache-key
        ingredients differ across modes -- the digest sha (verdict column
        bytes) and the prompt sha (grounded vs ungrounded variant) -- so the
        grounded response cannot leak into the ablated pass."""
        grounded_entry = ablation.attributions_by_id(ablation.round_path)["inst-childfail"]
        ablated_entry = ablation.attributions_by_id(ablation.ablated_dir)["inst-childfail"]
        assert grounded_entry["digest_sha256"] != ablated_entry["digest_sha256"]
        assert grounded_entry["prompt_sha256"] != ablated_entry["prompt_sha256"]
        assert grounded_entry["level_grounded"] is True
        assert ablated_entry["level_grounded"] is False
        # The ablated pass paid a fresh LM call for it: not served from cache.
        assert ablated_entry["attempts"][0]["cached"] is False

    def test_uncheckable_record_legitimately_shares_its_cache_entry(self, ablation):
        """inst-uncheck is UNDETERMINED in grounded mode too: its digest bytes
        (all-n/a verdict column) and its prompt variant (ungrounded fallback)
        are byte-identical across modes, so the ablated pass's cache HIT here
        is the cache working on identical inputs -- the response replayed was
        produced under the exact same digest, prompt, and config, so nothing
        mode-specific crossed over. This is NOT contamination."""
        grounded_entry = ablation.attributions_by_id(ablation.round_path)["inst-uncheck"]
        ablated_entry = ablation.attributions_by_id(ablation.ablated_dir)["inst-uncheck"]
        assert grounded_entry["digest_sha256"] == ablated_entry["digest_sha256"]
        assert grounded_entry["prompt_sha256"] == ablated_entry["prompt_sha256"]
        # The grounded pass paid the call; the ablated pass replayed it.
        assert grounded_entry["attempts"][0]["cached"] is False
        assert ablated_entry["attempts"][0]["cached"] is True
        # And the hit is corroborated by the call count: the ablated LM was
        # scripted with exactly one response, all of it spent on inst-childfail.
        assert ablation.ablated_lm._call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
