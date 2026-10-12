"""Build a blind A/B/C/D validation review sheet from existing COMET results.

Each item compares two translations of the same sentence. The answer key is
separate from the sheet that an evaluator should see while judging.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/analysis/comet_validation_scaled_v3/all_examples.csv"
DEFAULT_OUTPUT = ROOT / "outputs/analysis/abcd_validation_scaled_v3/manual_review"
SEED = 20261012
KEYS = ("pair", "doc_id", "src_sent_id", "tgt_sent_id")
PAIRS = ("en-de", "en-fr")
COMPARISONS = {
    "context": ("A", "B"),
    "llm_sentence": ("A", "C"),
    "llm_context": ("B", "D"),
}
CUE = re.compile(r"\b(?:it|its|they|them|their|these|those|this|that|he|him|his|she|her)\b", re.I)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return tuple(row[column] for column in KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*KEYS, "split", "previous_src", "src", "tgt"}
        for arm in "ABCD":
            required.update((f"{arm}_prediction", f"{arm}_comet"))
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"COMET score file is missing columns: {sorted(missing)}")
        rows = list(reader)
    if (len(rows) != 1500 or len({identity(row) for row in rows}) != len(rows)
            or {row["pair"] for row in rows} != set(PAIRS)
            or any(row["split"] != "validation" for row in rows)):
        raise ValueError("Expected 1500 unique scaled_v3 validation sentences in two language pairs")
    if any(not all(row[f"{arm}_prediction"] for arm in "ABCD") for row in rows):
        raise ValueError("Empty model prediction")

    rng = random.Random(SEED)
    chosen = []
    # Per pair and comparison: four randomly sampled cue sentences and four
    # without cues, plus the largest positive and negative COMET differences.
    # COMET extremes are diagnostic only; do not treat them as random.
    for name, (before, after) in COMPARISONS.items():
        for pair in PAIRS:
            changed = [row for row in rows if row["pair"] == pair
                       and row[f"{before}_prediction"] != row[f"{after}_prediction"]]
            ranked = sorted(changed, key=lambda row: float(row[f"{after}_comet"])
                            - float(row[f"{before}_comet"]))
            if len(ranked) < 10 or identity(ranked[0]) == identity(ranked[-1]):
                raise ValueError(f"Not enough changed sentences for {name}/{pair}")
            extremes = [(ranked[0], "lowest_comet_delta"), (ranked[-1], "highest_comet_delta")]
            extreme_ids = {identity(row) for row, _ in extremes}
            chosen.extend((name, before, after, pair, row, label) for row, label in extremes)
            for cue, label in ((True, "random_cue"), (False, "random_no_cue")):
                pool = [row for row in changed if identity(row) not in extreme_ids
                        and bool(CUE.search(row["src"])) == cue]
                if len(pool) < 4:
                    raise ValueError(f"Not enough changed {label} sentences for {name}/{pair}")
                chosen.extend((name, before, after, pair, row, label)
                              for row in rng.sample(pool, 4))
    rng.shuffle(chosen)

    blind, answer = [], []
    for n, (name, before, after, pair, row, selection) in enumerate(chosen, start=1):
        review_id = f"ABCD-{n:03d}"
        before_at = 1 + rng.randrange(2)
        candidates = {before_at: row[f"{before}_prediction"],
                      3 - before_at: row[f"{after}_prediction"]}
        blind.append({"review_id": review_id, "pair": pair,
                      "previous_src": row["previous_src"], "src": row["src"],
                      "reference": row["tgt"],
                      "candidate_1": candidates[1], "candidate_2": candidates[2],
                      "context_relevant_0_1_unsure": "",
                      "pronoun_issue_0_1_unsure": "",
                      "cross_sentence_reference_0_1_unsure": "",
                      "term_issue_0_1_unsure": "",
                      "meaning_winner_1_2_tie": "", "fluency_winner_1_2_tie": "",
                      "critical_error_candidate_1_0_1": "",
                      "critical_error_candidate_2_0_1": "", "notes": ""})
        answer.append({"review_id": review_id, "comparison": name,
                       "before_arm": before, "after_arm": after, "pair": pair,
                       "doc_id": row["doc_id"], "src_sent_id": row["src_sent_id"],
                       "tgt_sent_id": row["tgt_sent_id"], "selection": selection,
                       "cue_regex_matched": int(bool(CUE.search(row["src"]))),
                       "before_candidate": before_at, "after_candidate": 3 - before_at,
                       "comet_delta_after_minus_before":
                       float(row[f"{after}_comet"]) - float(row[f"{before}_comet"])})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blind_path = output / "blind_review.csv"
    key_path = output / "answer_key.csv"
    if blind_path.exists() or key_path.exists():
        raise FileExistsError("Review already exists; use a new --output-dir to preserve annotations")
    save_csv(blind_path, blind)
    save_csv(key_path, answer)
    metadata = {"dataset": "scaled_v3", "split": "validation", "seed": SEED,
                "source_path": str(args.scores.resolve()),
                "source_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                "num_items": len(blind), "per_pair_comparison": 10,
                "selection": "4 random cue + 4 random non-cue + 2 COMET diagnostic extremes",
                "comparisons": COMPARISONS,
                "note": "Cue regex is a sampling proxy; context relevance must be judged manually."
                        " COMET extremes are not representative."}
    (output / "review_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review: {blind_path}\nAnswer key: {key_path}")
    print(f"Rows: {len(blind)}; input SHA256: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()
