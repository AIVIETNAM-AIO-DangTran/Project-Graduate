"""Compare beam, Top-k and Top-p reranked translations with validation COMET.

Run with .venv-comet. Reuses the already scored arm B sentence COMET values,
after verifying the exact dataset, predictions, checkpoint, and row alignment.
"""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from evaluate_comet import (
    KEYS, MODEL_NAME, PAIRS, PROJECT_ROOT, file_hash, load_metric,
    read_table, validate_scores, write_csv, write_json,
)

SPLIT = "validation"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("multidoc_v2", "scaled_v3"), default="scaled_v3")
    parser.add_argument("--top-k-dir", type=Path, default=None)
    parser.add_argument("--top-p-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def check_content(reference, other, label):
    if len(other) != len(reference) or not other.index.equals(reference.index):
        raise ValueError(f"{label}: sentence keys differ from the validation set")
    for column in ("split", "src", "previous_src", "tgt"):
        if not other[column].equals(reference[column]):
            raise ValueError(f"{label}: {column} differs from the validation set")


def inputs(args):
    data_dir = PROJECT_ROOT / "data/processed" / f"europarl_{args.dataset}"
    decoding_dir = PROJECT_ROOT / "outputs/decoding" / args.dataset
    baseline_path = PROJECT_ROOT / "outputs/predictions" / f"context_lora_balanced_{args.dataset}_validation.csv"
    existing_metrics = PROJECT_ROOT / "outputs/metrics" / f"comet_validation_{args.dataset}_metrics.json"
    existing_scores = PROJECT_ROOT / "outputs/analysis" / f"comet_validation_{args.dataset}" / "all_examples.csv"
    data_path = data_dir / "validation.csv"
    manifest = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    dataset_hash = file_hash(data_path)
    if dataset_hash != manifest["csv_sha256"][SPLIT]:
        raise ValueError(f"Validation CSV differs from the {args.dataset} manifest")
    truth = read_table(data_path, SPLIT)
    beam = read_table(baseline_path, SPLIT, predictions=True)
    check_content(truth, beam, "Beam baseline")
    baseline_hash = file_hash(baseline_path)
    existing = json.loads(existing_metrics.read_text(encoding="utf-8"))
    if (existing.get("metric_model") != MODEL_NAME or existing.get("dataset") != args.dataset
            or existing.get("split") != SPLIT or existing.get("smoke_test")
            or existing.get("num_examples_per_arm") != len(truth)
            or existing["inputs"]["dataset"]["sha256"] != dataset_hash
            or existing["inputs"]["B"]["sha256"] != baseline_hash):
        raise ValueError("Existing arm B COMET report has different provenance")
    saved = read_table(existing_scores, SPLIT)
    check_content(truth, saved, "Existing COMET score table")
    if not saved["B_prediction"].equals(beam["prediction"]):
        raise ValueError("Existing COMET beam text differs from the baseline")
    b_scores = pd.to_numeric(saved["B_comet"], errors="raise")
    if not b_scores.map(math.isfinite).all():
        raise ValueError("Existing beam COMET scores are not finite")
    table = truth.reset_index(drop=True).copy()
    table["beam_prediction"] = beam["prediction"].to_numpy()
    table["beam_comet"] = b_scores.to_numpy()
    for pair in PAIRS:
        expected = existing["arms"]["B"]["pairs"][pair]["comet_mean"]
        measured = table.loc[table["pair"].eq(pair), "beam_comet"].mean()
        if not math.isclose(measured, expected, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Existing beam COMET mean differs for {pair}")
    provenance = {"dataset": dataset_hash, "beam": baseline_hash,
                  "existing_comet_report": file_hash(existing_metrics),
                  "existing_comet_scores": file_hash(existing_scores)}
    folders = {
        "top_k": args.top_k_dir or decoding_dir / "validation_top_k_k50_n3_t0p8_seed42",
        "top_p": args.top_p_dir or decoding_dir / "validation_top_p_p0p9_n3_t0p8_seed42",
    }
    for method, folder in folders.items():
        folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
        report_path, prediction_path = folder / "metrics.json", folder / "predictions.csv"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        if (config["dataset"] != f"europarl_{args.dataset}" or config["split"] != SPLIT
                or config["method"] != method or config.get("limit_per_pair") != 0
                or report.get("smoke_test") or config["dataset_sha256"] != dataset_hash
                or config["baseline_sha256"] != baseline_hash
                or config["input_format"] != "src </s> previous_src"
                or (method == "top_k" and config["top_p"] != 1.0)
                or (method == "top_p" and config["top_k"] != 0)):
            raise ValueError(f"{method}: sampling metrics have different provenance")
        adapter_weights = PROJECT_ROOT / "outputs/checkpoints" / f"context_lora_balanced_{args.dataset}" / "final_adapter" / "adapter_model.safetensors"
        if (Path(config["adapter"]) != adapter_weights.parent or
                config["adapter_sha256"] != file_hash(adapter_weights)):
            raise ValueError(f"{method}: adapter differs from the frozen context model")
        if report["reranker"]["candidate_sha256"] != file_hash(folder / "candidates.jsonl"):
            raise ValueError(f"{method}: candidates changed after reranking")
        selected = read_table(prediction_path, SPLIT)
        check_content(truth, selected, method)
        if (selected["beam_prediction"].ne(beam["prediction"]).any()
                or selected["selected_prediction"].eq("").any()):
            raise ValueError(f"{method}: beam or selected translation differs")
        for pair in PAIRS:
            if report["pairs"][pair]["num_examples"] != int(table["pair"].eq(pair).sum()):
                raise ValueError(f"{method}: incomplete {pair} scoring")
        table[f"{method}_prediction"] = selected["selected_prediction"].to_numpy()
        provenance[method] = {"directory": str(folder), "predictions_sha256": file_hash(prediction_path),
                              "metrics_sha256": file_hash(report_path), "generation": config,
                              "reranker": report["reranker"]}
    return table, existing, provenance, decoding_dir


def summarize(table):
    result = {}
    for label in ("beam", "top_k", "top_p"):
        by_pair = {}
        for pair in PAIRS:
            subset = table.loc[table["pair"].eq(pair)]
            by_pair[pair] = {"num_examples": len(subset),
                             "comet_mean": float(subset[f"{label}_comet"].mean()),
                             "mean_delta_vs_beam": float(subset[f"{label}_comet_delta"].mean()) if label != "beam" else 0.0,
                             "changed_vs_beam": int(subset[f"{label}_prediction"].ne(subset["beam_prediction"]).sum()) if label != "beam" else 0}
        result[label] = {"pairs": by_pair,
                         "macro_comet": sum(by_pair[p]["comet_mean"] for p in PAIRS) / len(PAIRS),
                         "macro_delta_vs_beam": sum(by_pair[p]["mean_delta_vs_beam"] for p in PAIRS) / len(PAIRS)}
    return result


def main():
    args = arguments()
    table, existing, provenance, decoding_dir = inputs(args)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the COMET environment")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    metric, checkpoint = load_metric()
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != existing["checkpoint_sha256"]:
        raise ValueError("COMET checkpoint differs from the earlier A/B/C/D evaluation")
    for method in ("top_k", "top_p"):
        samples = [{"src": row.src, "mt": getattr(row, f"{method}_prediction"), "ref": row.tgt}
                   for row in table.itertuples(index=False)]
        print(f"Scoring {method}: {len(samples)} sentences")
        predicted = metric.predict(samples, batch_size=args.batch_size,
                                   gpus=1 if args.device == "cuda" else 0,
                                   accelerator="gpu" if args.device == "cuda" else "cpu",
                                   devices=[0] if args.device == "cuda" else 1,
                                   num_workers=0, length_batching=False)
        table[f"{method}_comet"] = validate_scores(predicted.scores, len(table))
        table[f"{method}_comet_delta"] = table[f"{method}_comet"] - table["beam_comet"]
    summary = summarize(table)
    output_dir = decoding_dir / "validation_comet_comparison"
    scores_path, metrics_path = output_dir / "all_examples.csv", output_dir / "metrics.json"
    write_csv(scores_path, table)
    write_json(metrics_path, {"metric_model": MODEL_NAME, "checkpoint_sha256": checkpoint_hash,
                              "split": SPLIT, "dataset": args.dataset, "provenance": provenance,
                              "device": args.device, "batch_size": args.batch_size,
                              "versions": {name: version(name) for name in
                                           ("unbabel-comet", "torch", "transformers", "pytorch-lightning", "pandas")},
                              "macro_definition": "unweighted average of en-de and en-fr means",
                              "summary": summary, "sentence_scores": str(scores_path)})
    print("\n=== COMET VALIDATION: BEAM VS TOP-K VS TOP-P ===")
    for label, row in summary.items():
        print(f"{label}: macro={row['macro_comet']:.4f}, delta={row['macro_delta_vs_beam']:+.4f}")
        for pair, values in row["pairs"].items():
            print(f"  {pair}: {values['comet_mean']:.4f}, delta={values['mean_delta_vs_beam']:+.4f}, "
                  f"changed={values['changed_vs_beam']}, N={values['num_examples']}")
    print(f"Metrics: {metrics_path}\nSentence scores: {scores_path}")


if __name__ == "__main__":
    main()
