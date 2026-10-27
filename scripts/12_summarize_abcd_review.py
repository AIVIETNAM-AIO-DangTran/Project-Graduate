"""Unblind completed A/B/C/D review, keeping random and COMET extreme rows separate."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "gpt6": ROOT / "outputs/analysis/abcd_validation_scaled_v3_gpt6/manual_review",
    "qwen": ROOT / "outputs/analysis/abcd_validation_scaled_v3/manual_review",
}
COMPARISONS = ("context", "llm_sentence", "llm_context")
ARMS = {"context": ("A", "B"), "llm_sentence": ("A", "C"),
        "llm_context": ("B", "D")}
PAIRS = ("en-de", "en-fr")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def counts(items, key):
    return {name: sum(item[key] == name for item in items)
            for name in ("after", "before", "tie")}


def summarize(items):
    return {"n": len(items), "meaning": counts(items, "meaning"),
            "fluency": counts(items, "fluency"),
            "critical_errors": {"before": sum(item["before_error"] for item in items),
                                "after": sum(item["after_error"] for item in items)},
            "context_relevant": sum(item["context_relevant"] == "1" for item in items),
            "pronoun_issue": sum(item["pronoun_issue"] == "1" for item in items),
            "term_issue": sum(item["term_issue"] == "1" for item in items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postedit-model", choices=("gpt6", "qwen"), default="gpt6")
    parser.add_argument("--reviewer-type", choices=("human", "ai"),
                        help="Default: human for GPT-6, AI for historical Qwen")
    parser.add_argument("--review", type=Path,
                        help="Completed annotated CSV; defaults to completed_review.csv for GPT-6")
    parser.add_argument("--answer-key", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base = BASE[args.postedit_model]
    reviewer_type = args.reviewer_type or ("human" if args.postedit_model == "gpt6" else "ai")
    args.review = args.review or base / ("completed_review.csv" if reviewer_type == "human" else "ai_review.csv")
    args.answer_key = args.answer_key or base / "answer_key.csv"
    args.output_dir = args.output_dir or base / ("human_results" if reviewer_type == "human" else "")
    review = read_csv(args.review)
    key = read_csv(args.answer_key)
    if len(review) != 60 or len(key) != 60:
        raise ValueError("Expected exactly 60 judgments and 60 key rows")
    reviewed = {row["review_id"]: row for row in review}
    mapping = {row["review_id"]: row for row in key}
    if (len(reviewed) != 60 or len(mapping) != 60 or set(reviewed) != set(mapping)):
        raise ValueError("Review IDs do not align or are duplicated")
    combined = []
    for rid, row in reviewed.items():
        k = mapping[rid]
        if (row["pair"] != k["pair"] or k["comparison"] not in COMPARISONS
                or (k["before_arm"], k["after_arm"]) != ARMS.get(k["comparison"])):
            raise ValueError(f"Mismatched language pair/comparison: {rid}")
        before, after = k["before_candidate"], k["after_candidate"]
        if (sorted((before, after)) != ["1", "2"] or
                row["candidate_1"] == row["candidate_2"]):
            raise ValueError(f"Invalid candidate alignment: {rid}")
        for field in ("meaning_winner_1_2_tie", "fluency_winner_1_2_tie"):
            if row[field] not in ("1", "2", "tie"):
                raise ValueError(f"Unscored or invalid {field}: {rid}")
        for field in ("critical_error_candidate_1_0_1", "critical_error_candidate_2_0_1"):
            if row[field] not in ("0", "1"):
                raise ValueError(f"Unscored or invalid {field}: {rid}")
        def outcome(value):
            return "tie" if value == "tie" else "before" if value == before else "after"
        combined.append({"review_id": rid, "comparison": k["comparison"],
                         "pair": k["pair"], "selection": k["selection"],
                         "meaning": outcome(row["meaning_winner_1_2_tie"]),
                         "fluency": outcome(row["fluency_winner_1_2_tie"]),
                         "before_error": int(row[f"critical_error_candidate_{before}_0_1"]),
                         "after_error": int(row[f"critical_error_candidate_{after}_0_1"]),
                         "context_relevant": row["context_relevant_0_1_unsure"],
                         "pronoun_issue": row["pronoun_issue_0_1_unsure"],
                         "term_issue": row["term_issue_0_1_unsure"]})
    summary = {"evaluation_type": ("independent human annotation" if reviewer_type == "human"
                                   else "AI-assisted, not independent human annotation"),
               "postedit_model": args.postedit_model,
               "dataset": "scaled_v3", "split": "validation", "total_items": 60,
               "sampling_note": "Random cue and non-cue rows are exploratory samples of changed outputs; COMET extremes are diagnostic only.",
               "comparisons": {}}
    if any(sum(item["comparison"] == name and item["pair"] == pair
               for item in combined) != 10 for name in COMPARISONS for pair in PAIRS):
        raise ValueError("Expected 10 judged sentences for every comparison/language pair")
    lines = [f"# A/B/C/D validation: blinded {reviewer_type} review", "",
             ("Human judgments; record reviewer identity and proficiency separately." if reviewer_type == "human"
              else "AI judgments; not independent human annotation.") + " All rows compare different translations.",
             "Cue/non-cue is a regex-based sampling split; human-assessed context relevance is a separate label.",
             "Counts from COMET extremes must not be reported as representative win rates.", ""]
    for name in COMPARISONS:
        lines += [f"## {name}", "",
                  "| Selection | Pair | After better meaning | Before better meaning | Tie | N |",
                  "|---|---|---:|---:|---:|---:|"]
        summary["comparisons"][name] = {}
        for subset in ("random", "extremes", "all"):
            summary["comparisons"][name][subset] = {}
            for pair in (*PAIRS, "both"):
                part = [item for item in combined if item["comparison"] == name
                        and (pair == "both" or item["pair"] == pair)
                        and (subset == "all" or
                             (item["selection"].startswith("random_") if subset == "random"
                              else not item["selection"].startswith("random_")))]
                result = summarize(part)
                summary["comparisons"][name][subset][pair] = result
                m = result["meaning"]
                lines.append(f"| {subset} | {pair} | {m['after']} | {m['before']} | {m['tie']} | {result['n']} |")
        lines.append("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{reviewer_type}_review_summary.md"
    json_path = args.output_dir / f"{reviewer_type}_review_summary.json"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== RANDOM CHANGED VALIDATION ROWS: MEANING ===")
    for name in COMPARISONS:
        result = summary["comparisons"][name]["random"]["both"]
        print(f"{name}: after={result['meaning']['after']}, before={result['meaning']['before']}, "
              f"tie={result['meaning']['tie']}, N={result['n']}")
    print(f"Summary: {report_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
