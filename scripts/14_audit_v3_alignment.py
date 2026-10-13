"""Verify frozen v3 CSV structure and draw a documented alignment-review sample.

Semantic equivalence and previous-sentence provenance require a reviewer and,
for the latter, the original Europarl XML. The output CSV keeps those fields blank.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("en-de", "en-fr")
SPLITS = ("train", "validation", "test")
KEYS = ("pair", "doc_id", "src_sent_id", "tgt_sent_id")
CUE = re.compile(r"\b(?:it|its|they|them|their|these|those|this|that|he|him|his|she|her)\b", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/europarl_scaled_v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis/alignment_audit_scaled_v3")
    parser.add_argument("--per-pair-split", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.per_pair_split < 1:
        parser.error("--per-pair-split must be positive")
    manifest = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    all_rows, seen_docs, seen_source, counts, hashes = {}, {}, {}, {}, {}
    for split in SPLITS:
        path = args.data_dir / f"{split}.csv"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[split] != manifest["csv_sha256"][split]:
            raise ValueError(f"CSV hash differs from frozen manifest: {path}")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        required = {*KEYS, "split", "context_group_id", "previous_src", "src", "tgt"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Missing columns in {path}")
        keys = [tuple(r[k] for k in KEYS) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate pair alignment in {path}")
        if any(r["split"] != split or r["pair"] not in PAIRS or
               not all(r[c].strip() for c in ("previous_src", "src", "tgt", "context_group_id")) or
               r["src"].strip().casefold() == r["previous_src"].strip().casefold() for r in rows):
            raise ValueError(f"Missing, malformed, or repeated context in {path}")
        seen_docs[split] = {r["doc_id"] for r in rows}
        seen_source[split] = {r["src"].strip().casefold() for r in rows}
        counts[split] = dict(Counter(r["pair"] for r in rows))
        all_rows[split] = rows
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if seen_docs[left] & seen_docs[right] or seen_source[left] & seen_source[right]:
                raise ValueError(f"Document/current source overlap: {left} vs {right}")
    rng = random.Random(args.seed)
    selected = []
    for split in SPLITS:
        for pair in PAIRS:
            groups = defaultdict(list)
            for r in all_rows[split]:
                if r["pair"] == pair:
                    groups[r["doc_id"]].append(r)
            ids = sorted(groups)
            rng.shuffle(ids)
            if len(ids) < args.per_pair_split:
                raise ValueError(f"Too few documents: {split}/{pair}")
            for doc_id in ids[:args.per_pair_split]:
                pool = groups[doc_id]
                cue_pool = [r for r in pool if CUE.search(r["src"])]
                row = rng.choice(cue_pool or pool)
                selected.append({"audit_id": f"ALIGN-{len(selected)+1:03d}", "split": split,
                                 "pair": pair, "doc_id": doc_id,
                                 "src_sent_id": row["src_sent_id"], "tgt_sent_id": row["tgt_sent_id"],
                                 "previous_src": row["previous_src"], "src": row["src"], "tgt": row["tgt"],
                                 "semantic_equivalent_1_0_unsure": "",
                                 "previous_source_adjacent_1_0_unsure": "",
                                 "notes": ""})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "alignment_review_blank.csv"
    with sample_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    metadata = {"dataset": "europarl_scaled_v3", "csv_sha256": hashes,
                "structural_checks": "PASS (manifest hashes, unique keys, no missing context, no document/source overlap)",
                "counts": counts, "documents": {k: len(v) for k, v in seen_docs.items()},
                "sample_size": len(selected), "per_pair_split": args.per_pair_split,
                "distinct_sampled_documents_per_pair_split": args.per_pair_split,
                "seed": args.seed,
                "sampling": "one cue sentence (where available) from each randomly drawn document",
                "limitation": "No semantic or raw-XML adjacency claim without bilingual human/archival review."}
    (args.output_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {sum(map(len, all_rows.values()))} rows; {metadata['documents']}")
    print(f"Review sheet: {sample_path} ({len(selected)} rows)")


if __name__ == "__main__":
    main()
