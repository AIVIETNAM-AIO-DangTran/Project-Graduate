"""Exploratory pronoun-cue analysis of the frozen v3 validation experiment.

Uses saved predictions only: sentence adapter A, context adapter B with the
correct previous sentence, the shuffled previous sentence, and empty context.
The cue is an English-source heuristic, not a gold discourse annotation.
"""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/europarl_scaled_v3/validation.csv"
SENTENCE = ROOT / "outputs/predictions/sentence_lora_balanced_scaled_v3_validation.csv"
CONTEXT = ROOT / "outputs/predictions/context_lora_balanced_scaled_v3_validation.csv"
PERTURBATION = ROOT / "outputs/analysis/context_perturbation/context_perturbation_validation_scaled_v3.csv"
PERTURBATION_METRICS = ROOT / "outputs/metrics/context_perturbation_validation_scaled_v3.json"
OUT = ROOT / "outputs/analysis/context_perturbation"
KEYS = ("pair", "doc_id", "src_sent_id", "tgt_sent_id")
PAIRS = ("en-de", "en-fr")
ARMS = ("sentence", "correct", "shuffled", "empty")
CONTRASTS = {"correct_minus_sentence": ("correct", "sentence"),
             "correct_minus_shuffled": ("correct", "shuffled"),
             "correct_minus_empty": ("correct", "empty")}
# Predefined before inspecting individual translations. "that" and "this"
# are intentionally omitted: they are often conjunctions or determiners.
PRONOUN = re.compile(r"\b(?:it|its|they|them|their|theirs|he|him|his|she|her|hers)\b", re.I)
REPS = 1000
SEED = 20261104


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not set(KEYS).issubset(rows[0]):
        raise ValueError(f"Empty or malformed table: {path}")
    output = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        if key in output:
            raise ValueError(f"Duplicate example: {key} in {path}")
        output[key] = row
    return output


def load():
    data, a, b, controls = map(read_table, (DATA, SENTENCE, CONTEXT, PERTURBATION))
    if not (data.keys() == a.keys() == b.keys() == controls.keys()):
        raise ValueError("Four tables do not have identical example keys")
    meta = json.loads(PERTURBATION_METRICS.read_text(encoding="utf-8"))
    if meta["config"]["split"] != "validation" or meta["config"]["limit_per_pair"]:
        raise ValueError("Context perturbation must cover full validation")
    if (meta["config"]["dataset_sha256"] != sha(DATA)
            or meta["config"]["baseline_sha256"] != sha(CONTEXT)):
        raise ValueError("Perturbation input hashes differ from current files")
    rows = []
    for key, item in data.items():
        if item["split"] != "validation" or item["pair"] not in PAIRS:
            raise ValueError(f"Unexpected split or pair: {key}")
        for table in (a, b, controls):
            if any(table[key][name] != item[name] for name in ("src", "tgt", "previous_src")):
                raise ValueError(f"Text/reference mismatch for {key}")
        if b[key]["prediction"] != controls[key]["correct_prediction"]:
            raise ValueError(f"Correct-context prediction mismatch for {key}")
        if any(not value for value in (item["src"], item["tgt"], a[key]["prediction"],
                                     b[key]["prediction"], controls[key]["shuffled_prediction"],
                                     controls[key]["empty_prediction"])):
            raise ValueError(f"Empty source, reference or prediction for {key}")
        rows.append({**{name: item[name] for name in KEYS},
                     "src": item["src"], "previous_src": item["previous_src"],
                     "tgt": item["tgt"], "cue": bool(PRONOUN.search(item["src"])),
                     "sentence": a[key]["prediction"],
                     "correct": b[key]["prediction"],
                     "shuffled": controls[key]["shuffled_prediction"],
                     "empty": controls[key]["empty_prediction"]})
    return rows


def bootstrap(rows, metric_scores):
    """Paired document bootstrap for mean sentence chrF, per pair and macro."""
    rng = random.Random(SEED)
    doc_stats = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in rows:
        for contrast, (after, before) in CONTRASTS.items():
            cell = doc_stats[(row["pair"], row["doc_id"])][contrast]
            cell[0] += metric_scores[id(row)][after] - metric_scores[id(row)][before]
            cell[1] += 1
    pairs = {}
    resamples = {}
    for pair in PAIRS:
        docs = sorted(doc for p, doc in doc_stats if p == pair)
        if len(docs) < 2:
            raise ValueError(f"Need multiple documents for {pair}")
        pairs[pair] = {}
        resamples[pair] = {}
        for contrast in CONTRASTS:
            observed = sum(doc_stats[(pair, d)][contrast][0] for d in docs) / sum(
                doc_stats[(pair, d)][contrast][1] for d in docs)
            pairs[pair][contrast] = observed
            resamples[pair][contrast] = []
        for _ in range(REPS):
            counts = Counter(rng.choices(docs, k=len(docs)))
            for contrast in CONTRASTS:
                numerator = sum(doc_stats[(pair, d)][contrast][0] * n for d, n in counts.items())
                denominator = sum(doc_stats[(pair, d)][contrast][1] * n for d, n in counts.items())
                resamples[pair][contrast].append(numerator / denominator)
    def interval(values):
        values = sorted(values)
        return [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]
    report = {pair: {contrast: {"delta": pairs[pair][contrast],
                               "ci_95": interval(resamples[pair][contrast])}
                     for contrast in CONTRASTS} for pair in PAIRS}
    report["macro"] = {contrast: {"delta": sum(pairs[p][contrast] for p in PAIRS) / len(PAIRS),
                                  "ci_95": interval([sum(resamples[p][contrast][i] for p in PAIRS) / len(PAIRS)
                                                     for i in range(REPS)])}
                       for contrast in CONTRASTS}
    return report


def analyze(rows):
    if set(row["pair"] for row in rows) != set(PAIRS):
        raise ValueError("Each subset must contain both language pairs")
    bleu, chrf = BLEU(), CHRF()
    per_sentence = {id(row): {arm: chrf.sentence_score(row[arm], [row["tgt"]]).score
                               for arm in ARMS} for row in rows}
    report = {"num_examples": len(rows), "pair_counts": dict(Counter(r["pair"] for r in rows)),
              "num_documents": len({r["doc_id"] for r in rows}),
              "pairs": {}, "macro": {}}
    for pair in PAIRS:
        subset = [r for r in rows if r["pair"] == pair]
        refs = [[r["tgt"] for r in subset]]
        report["pairs"][pair] = {arm: {
            "bleu": bleu.corpus_score([r[arm] for r in subset], refs).score,
            "chrf": chrf.corpus_score([r[arm] for r in subset], refs).score,
            "changed_from_correct": sum(r[arm] != r["correct"] for r in subset)}
            for arm in ARMS}
    for arm in ARMS:
        report["macro"][arm] = {metric: sum(report["pairs"][p][arm][metric] for p in PAIRS) / len(PAIRS)
                                for metric in ("bleu", "chrf")}
    report["mean_sentence_chrf_document_bootstrap"] = bootstrap(rows, per_sentence)
    return report, per_sentence


def main():
    rows = load()
    groups = {"pronoun_cue": [r for r in rows if r["cue"]],
              "no_pronoun_cue": [r for r in rows if not r["cue"]]}
    report = {"design": {
        "split": "validation", "subset": "current English src contains one of it/its/they/them/their/theirs/he/him/his/she/her/hers",
        "caveat": "Heuristic cue does not establish that preceding context is needed; exploratory analysis only.",
        "bootstrap_unit": "document within language pair", "bootstrap_reps": REPS, "seed": SEED,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (DATA, SENTENCE, CONTEXT, PERTURBATION,
                                                                     PERTURBATION_METRICS)}}, "groups": {}}
    for name, subset in groups.items():
        result, _ = analyze(subset)
        report["groups"][name] = result
        print(f"\n=== {name}: N={result['num_examples']}; docs={result['num_documents']} ===")
        for pair in PAIRS:
            m = result["pairs"][pair]
            print(f"{pair} N={result['pair_counts'][pair]}: A={m['sentence']['bleu']:.2f}, "
                  f"B={m['correct']['bleu']:.2f}, shuffled={m['shuffled']['bleu']:.2f}, "
                  f"empty={m['empty']['bleu']:.2f} BLEU")
        for contrast, value in result["mean_sentence_chrf_document_bootstrap"]["macro"].items():
            low, high = value["ci_95"]
            print(f"{contrast}: mean sentence chrF {value['delta']:+.2f} [{low:+.2f}, {high:+.2f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "context_cues_validation_scaled_v3.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
