"""Paired document-cluster bootstrap for the frozen v3 A/B/C/D predictions.

The same sampled document IDs are used for both language pairs and all arms.
This estimates uncertainty due to selecting Europarl documents; it does not
capture variation from a new training seed, annotator, or LLM run.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("en-de", "en-fr")
ARMS = ("A", "B", "C", "D")
CONTRASTS = {"context_B_minus_A": ("B", "A"),
             "llm_on_sentence_C_minus_A": ("C", "A"),
             "llm_on_context_D_minus_B": ("D", "B"),
             "context_with_llm_D_minus_C": ("D", "C")}
METRICS = ("bleu", "chrf", "comet")


def percentile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def read_data(path, split):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"split", "pair", "doc_id", "src_sent_id", "tgt_sent_id", "tgt"}
    required.update(f"{a}_{suffix}" for a in ARMS for suffix in ("prediction", "comet"))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required fields: {sorted(required - set(rows[0] if rows else []))}")
    if any(row["split"] != split or row["pair"] not in PAIRS for row in rows):
        raise ValueError("Unexpected split or language pair")
    keys = [(r["pair"], r["doc_id"], r["src_sent_id"], r["tgt_sent_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated translation key")
    if any(not r["tgt"] or any(not r[f"{a}_prediction"] for a in ARMS) for r in rows):
        raise ValueError("Empty reference or prediction")
    for r in rows:
        for arm in ARMS:
            score = float(r[f"{arm}_comet"])
            if not 0 <= score <= 1:
                raise ValueError("COMET score out of [0, 1]")
            r[f"{arm}_comet"] = score
    docs = {pair: defaultdict(list) for pair in PAIRS}
    for r in rows:
        docs[r["pair"]][r["doc_id"]].append(r)
    doc_ids = sorted(set(docs[PAIRS[0]]) | set(docs[PAIRS[1]]))
    if set(docs[PAIRS[0]]) != set(docs[PAIRS[1]]):
        raise ValueError("Pair document sets differ; preserve the joint bootstrap design")
    return rows, docs, doc_ids


def pair_scores(rows, metric, bleu, chrf):
    if metric == "comet":
        return {a: sum(r[f"{a}_comet"] for r in rows) / len(rows) for a in ARMS}
    reference = [r["tgt"] for r in rows]
    scorer = bleu if metric == "bleu" else chrf
    return {a: scorer.corpus_score([r[f"{a}_prediction"] for r in rows], [reference]).score
            for a in ARMS}


def precompute_document_stats(docs, ids, bleu, chrf):
    """Cache additive corpus statistics so each resample does no retokenization."""
    stats = {metric: {pair: {arm: {} for arm in ARMS} for pair in PAIRS} for metric in METRICS}
    for pair in PAIRS:
        for did in ids:
            rows = docs[pair][did]
            refs = [[r["tgt"] for r in rows]]
            for arm in ARMS:
                scores = [r[f"{arm}_comet"] for r in rows]
                stats["comet"][pair][arm][did] = (sum(scores), len(scores))
                hyps = [r[f"{arm}_prediction"] for r in rows]
                for metric, scorer in (("bleu", bleu), ("chrf", chrf)):
                    segments = scorer._extract_corpus_statistics(hyps, refs)
                    stats[metric][pair][arm][did] = tuple(map(sum, zip(*segments)))
    return stats


def cached_scores(draw_counts, stats, metric, pair, bleu, chrf):
    result = {}
    for arm in ARMS:
        first = next(iter(stats[metric][pair][arm].values()))
        totals = [0] * len(first)
        for did, repetitions in draw_counts.items():
            values = stats[metric][pair][arm][did]
            for i, value in enumerate(values):
                totals[i] += value * repetitions
        if metric == "comet":
            result[arm] = totals[0] / totals[1]
        else:
            scorer = bleu if metric == "bleu" else chrf
            result[arm] = scorer._compute_score_from_stats(totals).score
    return result


def summarize(scores):
    return {name: (sum(scores[pair][after] - scores[pair][before] for pair in PAIRS) / len(PAIRS))
            for name, (after, before) in CONTRASTS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--postedit-model", choices=("gpt6", "qwen"), default="gpt6")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 200:
        parser.error("Use at least 200 bootstrap replications for percentile intervals")
    model_suffix = "_gpt6" if args.postedit_model == "gpt6" else ""
    path = args.input or ROOT / f"outputs/analysis/comet_{args.split}_scaled_v3{model_suffix}/all_examples.csv"
    output = args.output or ROOT / f"outputs/metrics/document_bootstrap_{args.split}_scaled_v3{model_suffix}.json"
    rows, docs, ids = read_data(path, args.split)
    bleu, chrf = BLEU(), CHRF()
    stats = precompute_document_stats(docs, ids, bleu, chrf)
    whole = Counter({did: 1 for did in ids})
    originals = {metric: {pair: cached_scores(whole, stats, metric, pair, bleu, chrf)
                          for pair in PAIRS} for metric in METRICS}
    # Check the cached sufficient-statistics implementation against corpus_score.
    for metric in METRICS:
        for pair in PAIRS:
            direct = pair_scores([r for did in ids for r in docs[pair][did]], metric, bleu, chrf)
            if any(abs(direct[a] - originals[metric][pair][a]) > 1e-10 for a in ARMS):
                raise AssertionError(f"Cached {metric}/{pair} differs from direct corpus_score")
    observed = {metric: summarize(originals[metric]) for metric in METRICS}
    bootstrap = {metric: {contrast: [] for contrast in CONTRASTS} for metric in METRICS}
    rng = random.Random(args.seed)
    for iteration in range(args.reps):
        sampled = Counter(rng.choices(ids, k=len(ids)))
        for metric in METRICS:
            scores = {pair: cached_scores(sampled, stats, metric, pair, bleu, chrf) for pair in PAIRS}
            for contrast, delta in summarize(scores).items():
                bootstrap[metric][contrast].append(delta)
        if (iteration + 1) % 100 == 0:
            print(f"{iteration + 1}/{args.reps} document resamples", flush=True)
    results = {metric: {contrast: {"observed_delta": observed[metric][contrast],
                                    "ci95_percentile": [percentile(values, .025), percentile(values, .975)]}
                        for contrast, values in bootstrap[metric].items()} for metric in METRICS}
    report = {"dataset": "europarl_scaled_v3", "split": args.split,
              "postedit_model": args.postedit_model,
              "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "documents": len(ids), "examples": len(rows), "pair_counts": {p: sum(map(len, docs[p].values())) for p in PAIRS},
              "bootstrap_unit": "document ID, same draw for both pairs and all arms",
              "macro": "equal mean of en-de and en-fr corpus scores (or mean COMET)",
              "interval": "2.5th and 97.5th percentiles of paired document resamples",
              "repetitions": args.reps, "seed": args.seed,
              "sacrebleu_signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())},
              "caveat": "Conditional on one frozen dataset/model seed/prompt; not a human-quality or retraining uncertainty interval.",
              "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for metric, contrasts in results.items():
        for name, result in contrasts.items():
            lo, hi = result["ci95_percentile"]
            print(f"{metric} {name}: {result['observed_delta']:+.4f} [ {lo:+.4f}, {hi:+.4f} ]")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
