"""Rank changed Top-k/Top-p validation translations by paired COMET delta."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("pair", "doc_id", "src_sent_id", "tgt_sent_id")
TEXT = ("split", "src", "previous_src", "tgt")
PAIRS = ("en-de", "en-fr")
TOL = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), set(reader.fieldnames or ())


def key(row):
    return tuple(row[field] for field in KEYS)


def index(rows, path):
    mapped = {key(row): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Duplicate sentence keys: {path}")
    return mapped


def verify(dataset):
    data_dir = ROOT / "data/processed" / f"europarl_{dataset}"
    comparison = ROOT / "outputs/decoding" / dataset / "validation_comet_comparison"
    scores_path = comparison / "all_examples.csv"
    baseline = ROOT / "outputs/predictions" / f"context_lora_balanced_{dataset}_validation.csv"
    report = json.loads((comparison / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    data_path = data_dir / "validation.csv"
    dataset_hash = sha256(data_path)
    provenance = report["provenance"]
    if (report["dataset"] != dataset or report["split"] != "validation"
            or manifest["csv_sha256"]["validation"] != dataset_hash
            or provenance["dataset"] != dataset_hash
            or provenance["beam"] != sha256(baseline)):
        raise ValueError("Dataset or beam differs from COMET experiment")
    canonical, _ = read_csv(data_path)
    scored, columns = read_csv(scores_path)
    needed = {*KEYS, *TEXT, "beam_prediction", "beam_comet"}
    for method in ("top_k", "top_p"):
        needed.update({f"{method}_prediction", f"{method}_comet", f"{method}_comet_delta"})
    if needed - columns:
        raise ValueError(f"Missing score table columns: {sorted(needed - columns)}")
    truth = index(canonical, data_path)
    scores = index(scored, scores_path)
    if set(truth) != set(scores):
        raise ValueError("COMET scores do not cover every validation sentence")
    beam, _ = read_csv(baseline)
    beam_map = index(beam, baseline)
    if set(beam_map) != set(truth):
        raise ValueError("Beam predictions do not match validation keys")
    for method in ("top_k", "top_p"):
        info = provenance[method]
        folder = Path(info["directory"])
        if not folder.is_absolute():
            folder = ROOT / folder
        prediction_path = folder / "predictions.csv"
        if (info["predictions_sha256"] != sha256(prediction_path)
                or info["metrics_sha256"] != sha256(folder / "metrics.json")):
            raise ValueError(f"{method} prediction or metric file changed")
        selected, _ = read_csv(prediction_path)
        selection_map = index(selected, prediction_path)
        if set(selection_map) != set(truth):
            raise ValueError(f"{method} has different sentence keys")
        for row_key, row in scores.items():
            if row[f"{method}_prediction"] != selection_map[row_key]["selected_prediction"]:
                raise ValueError(f"{method} selected text differs from scored text: {row_key}")
    for row_key, row in scores.items():
        if (any(row[field] != truth[row_key][field] for field in TEXT)
                or row["beam_prediction"] != beam_map[row_key]["prediction"]
                or row["pair"] not in PAIRS):
            raise ValueError(f"Dataset text or beam mismatch: {row_key}")
        for method in ("top_k", "top_p"):
            actual = float(row[f"{method}_comet"]) - float(row["beam_comet"])
            saved = float(row[f"{method}_comet_delta"])
            if not math.isfinite(saved) or not math.isclose(actual, saved, abs_tol=1e-7):
                raise ValueError(f"Invalid {method} COMET delta: {row_key}")
    for method in ("beam", "top_k", "top_p"):
        for pair in PAIRS:
            values = [float(row[f"{method}_comet"]) for row in scored if row["pair"] == pair]
            expected = report["summary"][method]["pairs"][pair]["comet_mean"]
            if not values or not math.isclose(sum(values) / len(values), expected, abs_tol=1e-7):
                raise ValueError(f"{method} {pair} COMET scores differ from metrics.json")
    return scored, report


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, method, pair=None):
    part = [row for row in rows if pair is None or row["pair"] == pair]
    changed = [row for row in part if row[f"{method}_prediction"] != row["beam_prediction"]]
    values = [float(row[f"{method}_comet_delta"]) for row in changed]
    return {"n": len(part), "changed": len(changed),
            "helped": sum(value > TOL for value in values),
            "hurt": sum(value < -TOL for value in values),
            "tied": sum(abs(value) <= TOL for value in values),
            "mean_delta_changed": sum(values) / len(values) if values else 0.0}


def short(text, limit=150):
    value = " ".join(text.split()).replace("|", "\\|")
    return value if len(value) <= limit else value[:limit - 1] + "…"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("multidoc_v2", "scaled_v3"), default="multidoc_v2")
    args = parser.parse_args()
    rows, metrics = verify(args.dataset)
    output_dir = ROOT / "outputs/analysis" / f"decoding_validation_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Top-k / Top-p validation error analysis", "",
             f"Ranked by paired COMET score on identical Europarl {args.dataset} sentences. "
             "These rankings identify examples for human review; COMET does not establish that "
             "the preceding sentence caused an improvement.", ""]
    for method in ("top_k", "top_p"):
        lines.extend([f"## {method}", "", "| Pair | N | Changed | Helped | Hurt | Tied | Mean delta (changed) |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for pair in (*PAIRS, None):
            result = summary(rows, method, pair)
            lines.append(f"| {pair or 'All'} | {result['n']} | {result['changed']} | "
                         f"{result['helped']} | {result['hurt']} | {result['tied']} | "
                         f"{result['mean_delta_changed']:+.4f} |")
        lines.append("")
        changed = [row for row in rows if row[f"{method}_prediction"] != row["beam_prediction"]]
        for outcome, sorted_rows in (
                ("helped", sorted(changed, key=lambda r: float(r[f"{method}_comet_delta"]), reverse=True)),
                ("hurt", sorted(changed, key=lambda r: float(r[f"{method}_comet_delta"])))):
            ranked = [row for row in sorted_rows
                      if (float(row[f"{method}_comet_delta"]) > TOL if outcome == "helped"
                          else float(row[f"{method}_comet_delta"]) < -TOL)]
            output = []
            for row in ranked:
                output.append({**{column: row[column] for column in (*KEYS, *TEXT)},
                               "beam_prediction": row["beam_prediction"],
                               "selected_prediction": row[f"{method}_prediction"],
                               "beam_comet": row["beam_comet"],
                               "selected_comet": row[f"{method}_comet"],
                               "comet_delta": row[f"{method}_comet_delta"]})
            write_csv(output_dir / f"{method}_{outcome}_examples.csv", output)
            lines.extend([f"### Most {outcome} examples", ""])
            for row in output[:5]:
                lines.extend([f"**{row['pair']} / {row['doc_id']} / {row['src_sent_id']}** "
                              f"(COMET {float(row['comet_delta']):+.4f})", "",
                              f"- Previous: {short(row['previous_src'])}",
                              f"- Source: {short(row['src'])}",
                              f"- Reference: {short(row['tgt'])}",
                              f"- Beam: {short(row['beam_prediction'])}",
                              f"- Selected: {short(row['selected_prediction'])}", ""])
            if not output:
                lines.append("No changed examples in this category.\n")
    lines.extend(["## Interpretation", "",
                  "Compare meaning and omissions manually in the source, reference and two outputs. "
                  "Report both corpus BLEU/chrF and COMET; a positive COMET delta on one "
                  "sentence is an inspection cue, not a significance test.", ""])
    report_path = output_dir / "qualitative_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("=== CHANGED TRANSLATIONS BY COMET ===")
    for method in ("top_k", "top_p"):
        result = summary(rows, method)
        print(f"{method}: changed={result['changed']}, helped={result['helped']}, "
              f"hurt={result['hurt']}, tied={result['tied']}")
    print(f"Report: {report_path}\nExample CSVs: {output_dir}")


if __name__ == "__main__":
    main()
