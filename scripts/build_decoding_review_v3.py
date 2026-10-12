"""Create a blinded, reproducible validation review of beam vs sampled decoding.

The answer key is deliberately stored separately from the annotation sheet.
Review only validation predictions; never use this sheet to tune on test data.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = (ROOT / "outputs/decoding/scaled_v3/validation_comet_comparison"
                  / "all_examples.csv")
DEFAULT_OUTPUT = ROOT / "outputs/analysis/decoding_validation_scaled_v3/manual_review"
KEYS = ("pair", "doc_id", "src_sent_id", "tgt_sent_id")
PAIRS = ("en-de", "en-fr")
METHODS = ("top_k", "top_p")
SEED = 20261012


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*KEYS, "split", "src", "previous_src", "tgt", "beam_prediction"}
        for method in METHODS:
            required.update((f"{method}_prediction", f"{method}_comet_delta"))
        if required - set(reader.fieldnames or ()):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or ())) } ")
        data = list(reader)
    if len(data) != 1500 or set(row["pair"] for row in data) != set(PAIRS):
        raise ValueError("Expected full 1500-row, two-pair scaled_v3 validation table")
    if any(row["split"] != "validation" for row in data):
        raise ValueError("Only validation examples can be reviewed here")
    if len({tuple(row[column] for column in KEYS) for row in data}) != len(data):
        raise ValueError("Duplicate sentence keys")

    randomizer = random.Random(SEED)
    selected = []
    # Per pair/method: eight random changed sentences plus two largest positive
    # and two largest negative COMET changes. Extremes are diagnostic examples,
    # never treated as a representative sample.
    for method in METHODS:
        for pair in PAIRS:
            changed = [row for row in data if row["pair"] == pair
                       and row[f"{method}_prediction"] != row["beam_prediction"]]
            positives = sorted(changed, key=lambda row: float(row[f"{method}_comet_delta"]),
                               reverse=True)[:2]
            negatives = sorted(changed, key=lambda row: float(row[f"{method}_comet_delta"]))[:2]
            extremes = positives + negatives
            extreme_keys = {tuple(row[column] for column in KEYS) for row in extremes}
            pool = [row for row in changed
                    if tuple(row[column] for column in KEYS) not in extreme_keys]
            if len(pool) < 8 or len(extreme_keys) != 4:
                raise ValueError(f"Insufficient distinct changed examples for {method}/{pair}")
            selected.extend((method, row, label) for label, group in
                            (("largest_positive", positives), ("largest_negative", negatives),
                             ("random_changed", randomizer.sample(pool, 8))) for row in group)
    randomizer.shuffle(selected)

    review, key = [], []
    for number, (method, row, selection) in enumerate(selected, start=1):
        review_id = f"V3-{number:03d}"
        if randomizer.randrange(2):
            one, two, beam_position = row[f"{method}_prediction"], row["beam_prediction"], 2
        else:
            one, two, beam_position = row["beam_prediction"], row[f"{method}_prediction"], 1
        review.append({
            "review_id": review_id, "pair": row["pair"],
            "previous_src": row["previous_src"], "src": row["src"],
            "reference": row["tgt"], "candidate_1": one, "candidate_2": two,
            "context_relevant_0_1_unsure": "", "pronoun_issue_0_1_unsure": "",
            "cross_sentence_reference_0_1_unsure": "", "term_issue_0_1_unsure": "",
            "meaning_winner_1_2_tie": "", "fluency_winner_1_2_tie": "",
            "critical_error_candidate_1_0_1": "", "critical_error_candidate_2_0_1": "",
            "notes": "",
        })
        key.append({"review_id": review_id, "method": method, "pair": row["pair"],
                    "doc_id": row["doc_id"], "src_sent_id": row["src_sent_id"],
                    "tgt_sent_id": row["tgt_sent_id"], "selection": selection,
                    "beam_candidate": beam_position, "sampled_candidate": 3 - beam_position,
                    "comet_delta_sampled_minus_beam": row[f"{method}_comet_delta"]})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_path, key_path = output / "blind_review.csv", output / "answer_key.csv"
    if review_path.exists() or key_path.exists():
        raise FileExistsError("Review files already exist; use a fresh --output-dir to avoid overwriting annotations")
    write_rows(review_path, review)
    write_rows(key_path, key)
    metadata = {"dataset": "scaled_v3", "split": "validation", "seed": SEED,
                "source_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                "rows": len(review), "per_pair_method": 12,
                "selection": "8 random changed + 2 largest positive + 2 largest negative per pair/method",
                "note": "The extreme cases are diagnostic, not population-representative."}
    (output / "review_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Blind review: {review_path}\nAnswer key (keep separate): {key_path}")
    print(f"Rows: {len(review)}; source SHA256: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()
