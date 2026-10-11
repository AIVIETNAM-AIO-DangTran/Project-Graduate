"""Score aligned A/B/C/D predictions with Unbabel/wmt22-comet-da."""

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))

import pandas as pd

MODEL_NAME = "Unbabel/wmt22-comet-da"
KEYS = ["pair", "doc_id", "src_sent_id", "tgt_sent_id"]
PAIRS = ("en-de", "en-fr")
ARMS = {
    "A": "sentence_lora_balanced",
    "B": "context_lora_balanced",
    "C": "sentence_lora_balanced_llm",
    "D": "context_lora_balanced_llm",
}
COMPARISONS = {
    "context_without_llm": ("A", "B"),
    "llm_sentence": ("A", "C"),
    "llm_context": ("B", "D"),
    "context_with_llm": ("C", "D"),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "multidoc_v2", "scaled_v3"), default="multidoc_v2")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-per-pair", type=int, default=0,
                        help="Positive value runs a separately saved smoke subset; 0 scores all examples.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit_per_pair < 0:
        parser.error("batch-size must be positive and limit-per-pair must be >= 0.")
    return args


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def read_table(path, split, predictions=False):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {*KEYS, "split", "src", "previous_src", "tgt"}
    if predictions:
        required.add("prediction")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.empty or table[list(required)].eq("").any().any():
        raise ValueError(f"{path}: empty table or required fields.")
    if table.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate alignment keys.")
    if not table["split"].eq(split).all() or set(table["pair"]) != set(PAIRS):
        raise ValueError(f"{path}: expected split={split}, with en-de and en-fr.")
    return table.set_index(KEYS, drop=False).sort_index()


def arms_for_dataset(dataset):
    if dataset == "pilot":
        return dict(ARMS)
    if dataset in ("multidoc_v2", "scaled_v3"):
        return {
            "A": f"sentence_lora_balanced_{dataset}",
            "B": f"context_lora_balanced_{dataset}",
            "C": f"sentence_lora_balanced_{dataset}_llm",
            "D": f"context_lora_balanced_{dataset}_llm",
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def load_aligned(split, limit_per_pair=0, dataset="pilot"):
    data_dir = PROJECT_ROOT / "data/processed" / (
        "europarl_pilot" if dataset == "pilot" else f"europarl_{dataset}"
    )
    dataset_path = data_dir / f"{split}.csv"
    dataset_hash = file_hash(dataset_path)
    if dataset != "pilot":
        summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
        if summary["csv_sha256"][split] != dataset_hash:
            raise ValueError(f"Dataset differs from builder manifest: {dataset_path}")
    canonical = read_table(dataset_path, split)
    result = canonical.reset_index(drop=True).copy()
    input_info = {"dataset": {"path": str(dataset_path), "sha256": dataset_hash}}
    content = ["split", "src", "previous_src", "tgt"]
    for arm, directory in arms_for_dataset(dataset).items():
        path = PROJECT_ROOT / f"outputs/predictions/{directory}_{split}.csv"
        table = read_table(path, split, predictions=True)
        if not table.index.equals(canonical.index) or not table[content].equals(canonical[content]):
            raise ValueError(f"Arm {arm}: IDs, source, context, or references do not match {dataset_path}.")
        result[f"{arm}_prediction"] = table["prediction"].to_numpy()
        if dataset != "pilot":
            metric_path = PROJECT_ROOT / f"outputs/metrics/{directory}_{split}_metrics.json"
            report = json.loads(metric_path.read_text(encoding="utf-8"))
            if report.get("split") != split or report.get("smoke_test", False):
                raise ValueError(f"Arm {arm}: mismatched split or smoke metrics: {metric_path}")
            if arm in ("A", "B") and report.get("dataset_sha256") != dataset_hash:
                raise ValueError(f"Arm {arm}: incorrect dataset hash in {metric_path}")
            if arm in ("C", "D"):
                baseline = "A" if arm == "C" else "B"
                if (report.get("config", {}).get("input_sha256") !=
                        input_info[baseline]["sha256"] or
                        report.get("num_examples") != len(canonical)):
                    raise ValueError(f"Arm {arm}: post-edit input/hash does not match {baseline}")
        if arm in ("C", "D"):
            baseline = "A" if arm == "C" else "B"
            if "mbart_prediction" not in table or not (
                table["mbart_prediction"].to_numpy() == result[f"{baseline}_prediction"].to_numpy()
            ).all():
                raise ValueError(f"Arm {arm}: saved draft does not match arm {baseline}.")
        input_info[arm] = {"path": str(path), "sha256": file_hash(path)}
    if limit_per_pair:
        result = result.groupby("pair", sort=False).head(limit_per_pair)
    return result.reset_index(drop=True), input_info


def build_samples(table, arm):
    # Every arm receives the same CURRENT source and reference, even when its MT system used context.
    return [{"src": row["src"], "mt": row[f"{arm}_prediction"], "ref": row["tgt"]}
            for _, row in table.iterrows()]


def validate_scores(scores, count):
    values = [float(score) for score in scores]
    if len(values) != count or not all(math.isfinite(score) for score in values):
        raise RuntimeError("COMET returned the wrong number of scores or non-finite scores.")
    return values


def load_metric():
    from comet import download_model
    from comet.models import RegressionMetric

    checkpoint = Path(download_model(MODEL_NAME))
    # The official COMET loader selects this class for wmt22-comet-da.
    # Explicit weights_only=False, when supported, loads the trusted official research
    # checkpoint's hyperparameters on modern Lightning/PyTorch without a global override.
    kwargs = {"checkpoint_path": str(checkpoint), "load_pretrained_weights": False,
              "map_location": "cpu", "strict": False}
    if "weights_only" in inspect.signature(RegressionMetric.load_from_checkpoint).parameters:
        kwargs["weights_only"] = False
    metric = RegressionMetric.load_from_checkpoint(**kwargs)
    metric.eval()
    return metric, checkpoint


def summarize(table, arm_directories=ARMS):
    arms = {}
    comparisons = {}
    for arm in arm_directories:
        pairs = {pair: {"num_examples": int(table["pair"].eq(pair).sum()),
                        "comet_mean": float(table.loc[table["pair"].eq(pair), f"{arm}_comet"].mean())}
                 for pair in PAIRS}
        arms[arm] = {
            "model": arm_directories[arm], "pairs": pairs,
            "macro_average_comet": sum(m["comet_mean"] for m in pairs.values()) / len(pairs),
            "overall_mean_comet": float(table[f"{arm}_comet"].mean()),
        }
    for name, (before, after) in COMPARISONS.items():
        delta = table[f"{after}_comet"] - table[f"{before}_comet"]
        table[f"{name}_comet_delta"] = delta
        comparisons[name] = {
            "before": before, "after": after,
            "macro_comet_delta": arms[after]["macro_average_comet"] - arms[before]["macro_average_comet"],
            "pairs": {pair: float(delta[table["pair"].eq(pair)].mean()) for pair in PAIRS},
            "higher_by_comet": int(delta.gt(1e-6).sum()),
            "lower_by_comet": int(delta.lt(-1e-6).sum()),
            "tied_by_comet": int(delta.abs().le(1e-6).sum()),
        }
    return arms, comparisons


def save_ranked_examples(table, directory):
    for name in ("llm_sentence", "llm_context"):
        column = f"{name}_comet_delta"
        helped = table[table[column] > 1e-6].sort_values(column, ascending=False).head(10)
        hurt = table[table[column] < -1e-6].sort_values(column).head(10)
        write_csv(directory / f"{name}_helped_examples.csv", helped)
        write_csv(directory / f"{name}_hurt_examples.csv", hurt)


def main():
    args = parse_arguments()
    # Validate all four prediction files before downloading/loading the metric.
    table, inputs = load_aligned(args.split, args.limit_per_pair, args.dataset)
    arm_directories = arms_for_dataset(args.dataset)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the COMET environment or select --device cpu.")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"Device: {device_name}; dataset: {args.dataset}; split: {args.split}; examples per arm: {len(table)}")
    print(f"Pair counts: {table['pair'].value_counts().to_dict()}")
    print(f"Loading metric: {MODEL_NAME}")
    started = time.perf_counter()
    metric, checkpoint = load_metric()
    checkpoint_sha256 = file_hash(checkpoint)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for arm in arm_directories:
        print(f"\nScoring arm {arm}: {arm_directories[arm]}")
        prediction = metric.predict(
            build_samples(table, arm), batch_size=args.batch_size,
            gpus=1 if args.device == "cuda" else 0,
            accelerator="gpu" if args.device == "cuda" else "cpu",
            devices=[0] if args.device == "cuda" else 1,
            num_workers=0, length_batching=False,
        )
        table[f"{arm}_comet"] = validate_scores(prediction.scores, len(table))
    arms, comparisons = summarize(table, arm_directories)
    suffix = f"_smoke{args.limit_per_pair}" if args.limit_per_pair else ""
    label = f"comet_{args.split}" + (f"_{args.dataset}" if args.dataset != "pilot" else "") + suffix
    analysis_dir = PROJECT_ROOT / "outputs/analysis" / label
    score_path = analysis_dir / "all_examples.csv"
    report_path = PROJECT_ROOT / "outputs/metrics" / f"{label}_metrics.json"
    report = {
        "metric_model": MODEL_NAME, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256, "split": args.split, "dataset": args.dataset,
        "smoke_test": bool(args.limit_per_pair), "num_examples_per_arm": len(table),
        "batch_size": args.batch_size, "num_workers": 0, "length_batching": False,
        "precision": "float32", "float32_matmul_precision": "highest", "seed": 42,
        "device": args.device, "device_name": device_name, "inputs": inputs,
        "versions": {package: version(package) for package in
                     ("unbabel-comet", "torch", "transformers", "pytorch-lightning", "pandas")},
        "score_scale": "raw, unscaled; higher is better; no clipping",
        "macro_definition": "unweighted mean of the en-de and en-fr mean scores",
        "evaluation_input": "same current source sentence and reference for every arm; no previous sentence",
        "arms": arms, "comparisons": comparisons,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else None,
        "score_table": str(score_path),
    }
    write_csv(score_path, table)
    save_ranked_examples(table, analysis_dir)
    write_json(report_path, report)
    smoke_label = " SMOKE" if args.limit_per_pair else ""
    print(f"\n=== COMET {args.split.upper()}{smoke_label} COMPARISON ===")
    for arm, result in arms.items():
        print(f"\nArm {arm}: {result['model']}")
        for pair, scores in result["pairs"].items():
            print(f"{pair}: COMET={scores['comet_mean']:.4f}, N={scores['num_examples']}")
        print(f"Macro COMET={result['macro_average_comet']:.4f}")
    print("\n=== PAIRED COMET DELTAS ===")
    for name, result in comparisons.items():
        print(f"{name} ({result['after']} - {result['before']}): {result['macro_comet_delta']:+.4f}")
    print(f"\nMetrics: {report_path}\nSentence scores and ranked examples: {analysis_dir}")


if __name__ == "__main__":
    main()
