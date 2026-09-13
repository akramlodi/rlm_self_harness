"""Released DeepMind MRCR v2 environment.

Instances come from the public CSV release, pinned to the eval_hub revision in
the experiment config.  The CSV's position fields are retained as provenance,
but the sub-verifier independently derives the repeated turns from ``queries``
and cross-checks the derived target against the official ``answer``.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind, Verdict

HASH_LENGTH = 12
PASS_SCORE_THRESHOLD = 1.0
SUBCALL_LOCAL_FINDING_CONTRACT = (
    "\n\n---\nIf you delegate a transcript slice, instruct the child to end with exactly: "
    'LOCAL FINDING: found <k> target turns in this slice, or LOCAL FINDING: no target turns in this slice.'
)
_TURN_RE = re.compile(r"(?:\A|\n\n)User: (?P<user>.*?)\n\nAssistant: (?P<assistant>.*?)(?=\n\nUser:|\Z)", re.DOTALL)
_FINAL_RE = re.compile(
    r"Prepend\s+(?P<hash>.{12})\s+to the\s+(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth)\s+(?P<query>.+?)\.",
    re.DOTALL,
)
_ORDINALS = {name: index for index, name in enumerate(("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"), 1)}
_FOUND_RE = re.compile(r"LOCAL FINDING:\s*found\s+(?P<count>\d+)\s+target turns in this slice", re.I)
_NONE_RE = re.compile(r"LOCAL FINDING:\s*no target turns in this slice", re.I)


def mrcr_v2_metric(prediction: str, target: str) -> float:
    """Exact port of eval_hub/mrcr_v2/run_evaluation.py."""
    if not isinstance(prediction, str) or not prediction:
        return 0.0
    target = target.strip()
    if len(target) < HASH_LENGTH:
        return 0.0
    random_hash = target[:HASH_LENGTH]
    target_ref = target[HASH_LENGTH:].strip()
    start_index = prediction.strip().rfind(random_hash)
    if start_index == -1:
        return 0.0
    prediction_content = prediction.strip()[start_index + HASH_LENGTH :].strip()
    return difflib.SequenceMatcher(a=target_ref, b=prediction_content).ratio()


def derive_needles(query: str, answer: str, expected_needles: int) -> dict[str, Any]:
    """Derive and independently validate target turns from a released prompt.

    The final request supplies the recurring user query and ordinal.  Every
    matching turn is then compared with the official answer payload; a mismatch
    is explicit metadata, never silently resolved by choosing a different turn.
    """
    matches = list(_FINAL_RE.finditer(query))
    final = matches[-1] if matches else None
    if final is None:
        return {"parse_status": "final-query-unparseable", "needles": []}
    # The follow-up says "in a archaic style" while the original turn says
    # "in archaic style".  This is the released template's wording change.
    target_query = "Write a " + final.group("query").strip().replace(" in a ", " in ") + "."
    expected_ordinal = _ORDINALS[final.group("ordinal").lower()]
    turns = []
    # Do not parse every assistant turn with a generic delimiter regex: generated
    # prose may itself quote ``User:``.  The official analysis utility anchors
    # the exact recurring user request first; retain that robust strategy.
    marker = f"User: {target_query}\n\nAssistant:"
    for match in re.finditer(re.escape(marker), query):
        content_start = match.end()
        content_end = query.find("\n\nUser:", content_start)
        if content_end == -1:
            content_end = len(query)
        assistant = query[content_start:content_end].strip()
        turns.append({"query": target_query, "instance_index": len(turns) + 1, "content": assistant,
                      "char_start": content_start, "char_end": content_end})
    target_ref = answer.strip()[HASH_LENGTH:].strip()
    ratios = [difflib.SequenceMatcher(a=target_ref, b=str(turn["content"])).ratio() for turn in turns]
    best_index = max(range(len(ratios)), key=ratios.__getitem__) + 1 if ratios else None
    status = "ok"
    if len(turns) != expected_needles:
        status = f"needle-count-mismatch:{len(turns)}"
    elif best_index != expected_ordinal:
        status = f"parse-mismatch:ordinal={expected_ordinal},best={best_index}"
    return {"parse_status": status, "target_query": target_query, "target_instance_index": expected_ordinal,
            "best_similarity_index": best_index, "best_similarity_ratio": max(ratios, default=0.0), "needles": turns}


def row_to_instance(row: dict[str, str], sample_seed: int, sample_index: int, expected_needles: int) -> dict[str, Any]:
    prompt, answer = row["queries"], row["answer"]
    derived = derive_needles(prompt, answer, expected_needles)
    digest = hashlib.sha256(f"{prompt}\u241f{answer}".encode()).hexdigest()[:16]
    return {"id": f"DeepMind_mrcrv2-{digest}", "prompt": prompt + SUBCALL_LOCAL_FINDING_CONTRACT,
            "queries": prompt, "answer": answer, "gold_content": answer.strip()[HASH_LENGTH:].strip(),
            "random_hash": answer.strip()[:HASH_LENGTH], "sample_seed": sample_seed, "sample_index": sample_index,
            "official_answer_context_position": row.get("answer_context_position", ""),
            "official_relevant_context_positions": row.get("relevant_context_positions", ""), **derived}


def ensure_dataset(cache_path: Path, url: str) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def load_deepmind_mrcrv2(
    *, cache_path: str, download_url: str, expected_needles: int, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Load a deterministic sample from the pinned released CSV cache."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    csv.field_size_limit(sys.maxsize)
    path = ensure_dataset(Path(cache_path), download_url)
    # The released CSV remains intact in ``cache_path``.  A row is eligible for
    # split sampling only when the independently derived turn records agree
    # with the official answer and ordinal; anomalous rows remain auditable in
    # the source cache rather than being silently repaired or discarded. Stream
    # the 3 GB CSV: retaining all parsed prompts would exceed ordinary EC2 RAM.
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    clean_count = 0
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            instance = row_to_instance(row, seed, index, expected_needles)
            if instance["parse_status"] != "ok":
                continue
            clean_count += 1
            if len(sample) < limit:
                sample.append(instance)
            else:
                replacement = rng.randrange(clean_count)
                if replacement < limit:
                    sample[replacement] = instance
    if clean_count < limit:
        raise ValueError(
            f"requested {limit} clean MRCR rows but only {clean_count} released rows "
            "passed the answer-anchored parser audit"
        )
    return sample


class DeepMindMrcrv2Verifier:
    PASS_SCORE_THRESHOLD = PASS_SCORE_THRESHOLD

    def config(self) -> dict[str, Any]:
        return {"environment": "DeepMind_mrcrv2", "pass_score_threshold": self.PASS_SCORE_THRESHOLD,
                "scoring_metric": "eval_hub.mrcr_v2_metric (exact port)", "promotion_mapping": "pass iff score == 1.0"}

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        score = mrcr_v2_metric(produced, str(instance["answer"]))
        return Verdict(passed=score >= self.PASS_SCORE_THRESHOLD,
                       cause=None if score >= self.PASS_SCORE_THRESHOLD else VerifierCause.WRONG_VALUE,
                       gold=str(instance["gold_content"]), produced=produced, detail=f"ratio={score:.6f}")


class DeepMindMrcrv2SubVerifier:
    def __call__(self, instance: dict[str, Any], node: CallNode) -> bool | None:
        if node.kind is NodeKind.ERRORED or node.error_kind is not None or not isinstance(node.prompt, str) or not isinstance(node.response, str):
            return None
        if str(instance.get("parse_status")) != "ok":
            return None
        found = _FOUND_RE.search(node.response)
        no_match = _NONE_RE.search(node.response)
        if found is None and no_match is None:
            return None
        count = sum(1 for needle in instance["needles"] if str(needle["content"]) in node.prompt)
        return count == (int(found.group("count")) if found else 0)


def make_deepmind_mrcrv2_verifier() -> DeepMindMrcrv2Verifier:
    return DeepMindMrcrv2Verifier()


def make_deepmind_mrcrv2_sub_verifier() -> DeepMindMrcrv2SubVerifier:
    return DeepMindMrcrv2SubVerifier()


def load_deepmind_mrcrv2_from_config(config: Any, n: int, seed: int, length: str = "short") -> list[dict[str, Any]]:
    env = config.environments.DeepMind_mrcrv2
    if length == "short":
        cache_path, bucket, needles = env.cache_path_short, env.context_bucket_short, env.num_needles_short
    elif length == "long":
        cache_path, bucket, needles = env.cache_path_long, env.context_bucket_long, env.num_needles_long
    else:
        raise ValueError(f"unknown length {length!r}")
    filename = f"mrcr_v2p1_{needles}needle_{bucket}_dynamic_fewshot_text_style_fast.csv"
    return load_deepmind_mrcrv2(cache_path=cache_path, download_url=f"{env.dataset_base_url}/{filename}", expected_needles=needles, limit=n, seed=seed)
