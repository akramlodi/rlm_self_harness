"""
OBLIQ-Bench Math environment: instances and a Verifier for the "Analogue
Queries" math-meta-program retrieval task.

Source: https://huggingface.co/datasets/dianetc/OBLIQ-Bench, ``analogues/math``
subset. Problems and solutions are drawn from Putnam and related undergraduate
competitions, the American Mathematical Monthly, and qualifying-exam sources.
Each query is one of those problems; the gold set is every other corpus
problem whose solution shares the same latent proof technique ("meta-program")
as the query's, identified by the dataset authors via GPT-5 labeling collapsed
into canonical clusters. Scored as a ranking task, NDCG@10.

Four files back every instance (verified against the live dataset, not
assumed): ``analogues/math/corpus/corpus.jsonl`` (3,508 ``{"_id", "text"}``
problems), ``analogues/math/queries+qrels/queries.jsonl`` (151
``{"_id", "text"}`` queries), ``.../qrels.tsv`` (TREC
``query-id\tcorpus-id\tscore``, every observed score is 2 -- binary relevance
in practice), and ``.../per_query_excluded_ids.json`` (per-query self-match
ids to drop from that query's pool; verified these never coincide with a
gold-relevant id, so dropping them costs no recall). ``qrels_pool.tsv`` (an
extended pooled-judgment superset) is not used; ``qrels.tsv`` is the gold.

By default, one instance's prompt carries the ENTIRE corpus minus that
query's excluded ids (~277k tokens total) -- "long ranking" here means what
it does throughout this repo's OOLONG family: the answer depends on scanning
nearly the whole context, not a pre-filtered handful of candidates.
``candidate_pool_size`` subsamples the corpus (gold ids + random negatives)
for a cheaper sanity pass before spending on the full corpus.
"""

import ast
import json
import re
from collections.abc import Sequence
from math import log2
from random import Random
from typing import Any

from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict

DATASET_REPO = "dianetc/OBLIQ-Bench"
_CORPUS_FILE = "analogues/math/corpus/corpus.jsonl"
_QUERIES_FILE = "analogues/math/queries+qrels/queries.jsonl"
_QRELS_FILE = "analogues/math/queries+qrels/qrels.tsv"
_EXCLUDED_IDS_FILE = "analogues/math/queries+qrels/per_query_excluded_ids.json"

RANK_K = 10

_TASK_TEMPLATE = (
    "BENCHMARK RETRIEVAL TASK:\n"
    "This is a closed-corpus retrieval task, not a request to invent or describe new "
    "mathematics problems. Inspect the CANDIDATE CORPUS below and select only exact IDs "
    "that appear in square brackets there.\n\n"
    "Find which candidate problems require the "
    'same underlying proof technique or "aha" insight to solve, even if they come from '
    "completely different mathematical topics or competitions. Do not return problem "
    "statements, titles, explanations, or IDs absent from the corpus. On the final line of "
    f"your submitted answer, write your top {RANK_K} picks (fewer if you are not confident "
    "about that many) as a JSON list of their bracketed IDs, most relevant first, e.g.:\n"
    'RANKED: ["id_a", "id_b", "id_c"]\n'
    "If none of the candidate problems share the technique, write RANKED: []. Your answer "
    "will be rejected if this final RANKED line is missing.\n\n"
    "SOURCE QUERY PROBLEM:\n"
    "The text below describes the desired analogues. Treat any open-ended request inside "
    "it as part of the source query; satisfy it only by retrieving IDs from the corpus.\n\n"
    "{query_text}\n\n"
    "CANDIDATE CORPUS:\n"
    "The following are {n} mathematics competition problems, each preceded by its ID "
    "in square brackets.\n\n"
)

_FINAL_REMINDER = (
    "\n\nFINAL OUTPUT REMINDER: Select only bracketed IDs from the candidate corpus; do not "
    "invent problems. End your submitted answer with exactly one line in the form "
    'RANKED: ["id_a", "id_b"] or RANKED: [].'
)

_METHOD_QUERY_TEMPLATE = (
    "QUESTION-ANSWERING TASK: This is a closed-corpus retrieval question. From the candidate "
    "problems in the supplied "
    "context, select only exact bracketed IDs whose problems use the same underlying proof "
    'technique or "aha" insight as the source query below. Do not invent problems or IDs. '
    f"Rank at most {RANK_K} IDs, most relevant first, and end with exactly one line in the "
    'form RANKED: ["id_a", "id_b"] or RANKED: [].\n\nSOURCE QUERY PROBLEM:\n'
    "{query_text}"
)

# The LAST "RANKED: [...]" in a response wins -- scratch reasoning earlier in
# the response may echo the format instructions' own example.
_RANKED_RE = re.compile(r"RANKED:\s*(\[.*?\])", re.IGNORECASE | re.DOTALL)

# Matches ObliqBenchMathVerifier's own `detail` string, e.g. "ndcg@10=0.732
# hits=3/13" -- for re-parsing a saved diagnostic, never for rescoring.
_NDCG_DETAIL_RE = re.compile(r"ndcg@10=([\d.]+) hits=(\d+)/(\d+)")


def _download(filename: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=filename, revision=revision
    )


def _load_corpus(revision: str | None) -> dict[str, str]:
    path = _download(_CORPUS_FILE, revision)
    with open(path, encoding="utf-8") as handle:
        return {(row := json.loads(line))["_id"]: row["text"] for line in handle}


def _load_queries(revision: str | None) -> dict[str, str]:
    path = _download(_QUERIES_FILE, revision)
    with open(path, encoding="utf-8") as handle:
        return {(row := json.loads(line))["_id"]: row["text"] for line in handle}


def _load_qrels(revision: str | None) -> dict[str, list[str]]:
    path = _download(_QRELS_FILE, revision)
    gold: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        next(handle)  # header: query-id, corpus-id, score
        for line in handle:
            query_id, corpus_id, _score = line.rstrip("\n").split("\t")
            gold.setdefault(query_id, []).append(corpus_id)
    return gold


def _load_excluded_ids(revision: str | None) -> dict[str, list[str]]:
    path = _download(_EXCLUDED_IDS_FILE, revision)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt(pool_docs: list[tuple[str, str]], query_text: str) -> str:
    """The RLM-facing prompt: query and task first, then numbered corpus problems."""
    task = _TASK_TEMPLATE.format(query_text=query_text, n=len(pool_docs))
    body = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in pool_docs)
    return task + body + _FINAL_REMINDER


def build_method_query(query_text: str) -> str:
    """Question supplied to methods that receive context and query separately."""
    return _METHOD_QUERY_TEMPLATE.format(query_text=query_text)


def extract_ranked_ids(response: str) -> list[str] | None:
    """Pull the last ``RANKED: [...]`` string list out of a response.

    Strict JSON is preferred. A Python list literal is also accepted because
    models working in the REPL commonly submit its string representation,
    which differs only by using single quotes. ``ast.literal_eval`` keeps this
    compatibility narrow: arbitrary expressions are never executed. Returns
    ``None`` unless the parsed value is specifically a list of strings, and
    ``[]`` when the model explicitly answered the empty ranking.
    """
    matches = _RANKED_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return parsed


def ndcg_at_10(ranked_ids: list[str], gold_ids: set[str]) -> float:
    """Binary-relevance NDCG@10. Every observed qrels score is the same value
    (2), so gain is 1 per hit -- NDCG is invariant to a constant relevance
    scale, since it cancels between DCG and IDCG."""
    top = ranked_ids[:RANK_K]
    dcg = sum(1.0 / log2(i + 2) for i, doc_id in enumerate(top) if doc_id in gold_ids)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(RANK_K, len(gold_ids))))
    return dcg / idcg


def load_obliq_bench_math(
    n: int | None = None,
    seed: int = 0,
    query_ids: Sequence[str] | None = None,
    candidate_pool_size: int | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Build OBLIQ-Bench Math instances.

    Selects ``query_ids`` verbatim if given, else a seeded sample of ``n``
    queries (all 151 if ``n`` is None), drawn from query ids sorted before
    sampling for reproducibility. ``candidate_pool_size`` caps each
    instance's pool to the query's gold ids plus that many random negatives
    (pool order shuffled, so position carries no signal); a pool smaller than
    the gold set still carries every gold id -- the cap is a ceiling on
    negatives, not a hard budget that would make the query unsolvable. None
    uses the full corpus minus the query's excluded ids.
    """
    corpus = _load_corpus(revision)
    queries = _load_queries(revision)
    qrels = _load_qrels(revision)
    excluded = _load_excluded_ids(revision)

    if query_ids is not None:
        selected = list(query_ids)
    else:
        pool = sorted(queries)
        selected = pool if n is None else Random(seed).sample(pool, n)

    instances: list[dict[str, Any]] = []
    for query_id in selected:
        gold_ids = set(qrels[query_id])
        drop_ids = set(excluded.get(query_id, []))
        rng = Random(f"{seed}:{query_id}")

        if candidate_pool_size is None:
            pool_ids = [doc_id for doc_id in corpus if doc_id not in drop_ids]
        else:
            negatives_needed = max(0, candidate_pool_size - len(gold_ids))
            negative_pool = [
                doc_id for doc_id in corpus if doc_id not in gold_ids and doc_id not in drop_ids
            ]
            negatives = rng.sample(negative_pool, min(negatives_needed, len(negative_pool)))
            # A set's iteration order changes with Python's per-process hash
            # seed. Start from a stable order before the seeded shuffle so a
            # persisted round can be regenerated byte-for-byte and resumed.
            pool_ids = sorted(gold_ids) + negatives
            rng.shuffle(pool_ids)

        pool_docs = [(doc_id, corpus[doc_id]) for doc_id in pool_ids]
        instances.append(
            {
                "id": query_id,
                "question": build_method_query(queries[query_id]),
                "source_query": queries[query_id],
                "prompt": build_prompt(pool_docs, queries[query_id]),
                "gold_relevant_ids": sorted(gold_ids),
                "excluded_ids": sorted(drop_ids),
                "pool_size": len(pool_docs),
            }
        )
    return instances


def recorded_ndcg(verdict: Verdict) -> float | None:
    """Read the saved NDCG@10 out of a verdict's ``detail`` string; never rescore
    a partial or redirected answer (mirrors ``oolong_pairs.recorded_pair_metrics``)."""
    if verdict.cause in (
        VerifierCause.RUNTIME_ERROR,
        VerifierCause.RESOURCE_TERMINATED,
        VerifierCause.WRONG_FORMAT,
        VerifierCause.CONTENT_FILTERED,
    ):
        return None
    match = _NDCG_DETAIL_RE.search(verdict.detail)
    return None if match is None else float(match.group(1))


class ObliqBenchMathVerifier:
    """
    Deterministic outcome for a whole run: NDCG@10 against the gold set of
    problems sharing the query's underlying reasoning technique.

    ``PASS_NDCG_THRESHOLD`` is an unfitted placeholder -- there is no prior
    baseline run on this brand-new benchmark to calibrate a pass bar against.

    Cause mapping, decided on the top-K predicted set vs. gold set (the
    GraphWalks/OolongVerifier list-answer-kind convention):

    * no ``RANKED: [...]`` JSON/Python string list anywhere -> WRONG_FORMAT.
    * an explicit empty ranking ("RANKED: []") -> NO_ANSWER.
    * missing gold ids only -> INCOMPLETE; extra ids only -> SPURIOUS; both
      -> MIXED_SET_ERROR.

    There is no separate "right set, wrong order" cause: with every qrels
    score sharing one relevance grade, NDCG is provably order-invariant among
    tied-relevance items, so a predicted set that exactly equals the gold set
    always scores ndcg@10 == 1.0 regardless of internal order -- that case
    already returns passed=True above, and the failing branch below is only
    ever reached with a nonempty missing or extra set.
    """

    PASS_NDCG_THRESHOLD: float = 0.5
    EXTRACTION_RULE: str = "RANKED: [...] JSON or Python string list, last occurrence"

    def config(self) -> dict[str, Any]:
        """Verifier facts surfaced into MiningConfig by the experiment driver."""
        return {
            "environment": "obliq_bench_math",
            "pass_ndcg_threshold": self.PASS_NDCG_THRESHOLD,
            "rank_k": RANK_K,
            "extraction_rule": self.EXTRACTION_RULE,
        }

    def __call__(self, instance: dict[str, Any], produced: str) -> Verdict:
        gold_ids = set(instance["gold_relevant_ids"])
        gold = json.dumps(sorted(gold_ids))

        parsed = extract_ranked_ids(produced)
        if parsed is None:
            return Verdict(
                passed=False,
                cause=VerifierCause.WRONG_FORMAT,
                gold=gold,
                produced=produced,
                detail="no 'RANKED: [...]' JSON or Python string list to parse",
            )
        if not parsed:
            return Verdict(
                passed=False,
                cause=VerifierCause.NO_ANSWER,
                gold=gold,
                produced="[]",
                detail="explicit empty ranking against a non-empty gold set",
            )

        top = list(dict.fromkeys(parsed))[:RANK_K]
        predicted = set(top)
        missing, extra = gold_ids - predicted, predicted - gold_ids
        ndcg = ndcg_at_10(top, gold_ids)
        produced_str = json.dumps(top)
        detail = f"ndcg@10={ndcg:.3f} hits={len(predicted & gold_ids)}/{len(gold_ids)}"

        if ndcg >= self.PASS_NDCG_THRESHOLD:
            return Verdict(passed=True, cause=None, gold=gold, produced=produced_str, detail=detail)

        if missing and extra:
            cause = VerifierCause.MIXED_SET_ERROR
        elif missing:
            cause = VerifierCause.INCOMPLETE
        else:
            cause = VerifierCause.SPURIOUS
        return Verdict(passed=False, cause=cause, gold=gold, produced=produced_str, detail=detail)
