"""Compare LoRA adapters on validation or test with identical beam decoding."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))

import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
PAIR_CODES = {"en-de": "de_DE", "en-fr": "fr_XX"}
MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_BEAMS = 4
VARIANTS = {
    "sentence_balanced_multidoc_v2": {
        "directory": "sentence_lora_balanced_multidoc_v2",
        "format": "src",
        "truncation_side": "right",
        "dataset": "europarl_multidoc_v2",
        "loss_mode": "balanced",
    },
    "context_balanced_multidoc_v2": {
        "directory": "context_lora_balanced_multidoc_v2",
        "format": "src </s> previous_src",
        "truncation_side": "right",
        "dataset": "europarl_multidoc_v2",
        "loss_mode": "balanced",
    },
    "sentence_balanced_scaled_v3": {
        "directory": "sentence_lora_balanced_scaled_v3",
        "format": "src",
        "truncation_side": "right",
        "dataset": "europarl_scaled_v3",
        "loss_mode": "balanced",
    },
    "context_balanced_scaled_v3": {
        "directory": "context_lora_balanced_scaled_v3",
        "format": "src </s> previous_src",
        "truncation_side": "right",
        "dataset": "europarl_scaled_v3",
        "loss_mode": "balanced",
    },
    "sentence_unweighted_scaled_v3": {
        "directory": "sentence_lora_unweighted_scaled_v3",
        "format": "src",
        "truncation_side": "right",
        "dataset": "europarl_scaled_v3",
        "loss_mode": "unweighted",
    },
    "context_unweighted_scaled_v3": {
        "directory": "context_lora_unweighted_scaled_v3",
        "format": "src </s> previous_src",
        "truncation_side": "right",
        "dataset": "europarl_scaled_v3",
        "loss_mode": "unweighted",
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("multidoc_balanced", "scaled_balanced", "scaled_ablation", *VARIANTS), default="scaled_ablation",
        help="One adapter, both balanced adapters, or all four scaled v3 ablation adapters.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    return parser.parse_args()


def load_split(split, data_dir):
    dataset_path = data_dir / f"{split}.csv"
    data = pd.read_csv(dataset_path)
    required = {"pair", "split", "doc_id", "src_sent_id", "tgt_sent_id", "src", "previous_src", "tgt"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError(f"{split} data is empty or contains missing text.")
    if set(data["pair"].unique()) != set(PAIR_CODES):
        raise ValueError(f"Expected {split} pairs en-de and en-fr.")
    if not data["split"].eq(split).all():
        raise ValueError(f"Every row must belong to {split}.")
    if data.duplicated(["pair", "doc_id", "src_sent_id", "tgt_sent_id"]).any():
        raise ValueError(f"Duplicate example key in {dataset_path}.")
    manifest_path = data_dir / "summary.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("csv_sha256", {}).get(split)
        actual = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if expected and expected != actual:
            raise ValueError(f"{dataset_path} differs from summary.json.")
    return data, dataset_path


def check_training_metadata(settings, data_dir):
    directory = settings["directory"]
    metadata_path = PROJECT_ROOT / "outputs/checkpoints" / directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("input_format") != settings["format"] or
            metadata.get("source_truncation_side") != settings["truncation_side"] or
            metadata.get("max_length") != MAX_LENGTH or
            metadata.get("loss_mode") != settings["loss_mode"] or
            metadata.get("training_mode") != "full_training" or
            metadata.get("seed") != 42 or
            metadata.get("epochs") != 3.0):
        raise ValueError(f"Training configuration mismatch: {metadata_path}")
    for split in ("train", "validation"):
        csv_path = data_dir / f"{split}.csv"
        expected = metadata.get("dataset_files", {}).get(split, {}).get("sha256")
        actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Adapter {directory} was trained with different {split}.csv")
    return metadata_path


def build_inputs(batch, variant, tokenizer):
    input_format = VARIANTS[variant]["format"]
    if input_format == "src":
        return batch["src"].tolist()
    if input_format == "src </s> previous_src":
        return [
            f"{src} {tokenizer.eos_token} {previous}"
            for src, previous in zip(batch["src"], batch["previous_src"])
        ]
    if input_format == "previous_src </s> src":
        return [
            f"{previous} {tokenizer.eos_token} {src}"
            for src, previous in zip(batch["src"], batch["previous_src"])
        ]
    raise ValueError(f"Unsupported input format: {input_format}")


def evaluate_variant(variant, data, dataset_hash, split, dataset_path):
    settings = VARIANTS[variant]
    directory = settings["directory"]
    adapter_path = PROJECT_ROOT / "outputs/checkpoints" / directory / "final_adapter"
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "en_XX"
    tokenizer.truncation_side = settings["truncation_side"]
    tokenizer.padding_side = "right"

    print(f"\nEvaluating: {variant}")
    print(f"Adapter: {adapter_path}")
    print(f"Input: {settings['format']}; truncate: {tokenizer.truncation_side}")

    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    prediction_frames = []
    pair_metrics = {}
    try:
        for pair, target_code in PAIR_CODES.items():
            pair_data = data[data["pair"] == pair].reset_index(drop=True)
            predictions = []
            for start in tqdm(
                range(0, len(pair_data), BATCH_SIZE), desc=f"{variant} {pair}"
            ):
                batch = pair_data.iloc[start : start + BATCH_SIZE]
                tokenizer.src_lang = "en_XX"
                inputs = tokenizer(
                    build_inputs(batch, variant, tokenizer),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to("cuda")
                with torch.inference_mode():
                    ids = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        max_length=MAX_LENGTH,
                    )
                predictions.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))

            references = pair_data["tgt"].tolist()
            bleu = BLEU()
            chrf = CHRF()
            pair_metrics[pair] = {
                "num_examples": len(pair_data),
                "sacrebleu": bleu.corpus_score(predictions, [references]).score,
                "chrf": chrf.corpus_score(predictions, [references]).score,
                "bleu_signature": str(bleu.get_signature()),
                "chrf_signature": str(chrf.get_signature()),
            }
            pair_data["prediction"] = predictions
            prediction_frames.append(pair_data)

        report = {
            "model": directory,
            "base_model": MODEL_NAME,
            "adapter_path": str(adapter_path),
            "split": split,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "input_format": settings["format"],
            "source_truncation_side": settings["truncation_side"],
            "source_max_length": MAX_LENGTH,
            "decoding": {
                "method": "beam_search", "num_beams": NUM_BEAMS,
                "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            },
            "sacrebleu_version": sacrebleu.__version__,
            "pairs": pair_metrics,
            "macro_average_sacrebleu": sum(m["sacrebleu"] for m in pair_metrics.values()) / 2,
            "macro_average_chrf": sum(m["chrf"] for m in pair_metrics.values()) / 2,
            "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        prediction_path = PROJECT_ROOT / f"outputs/predictions/{directory}_{split}.csv"
        metric_path = PROJECT_ROOT / f"outputs/metrics/{directory}_{split}_metrics.json"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            prediction_path, index=False, encoding="utf-8"
        )
        metric_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Predictions: {prediction_path}")
        print(f"Metrics: {metric_path}")
        return report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.variant == "multidoc_balanced":
        variants = ["sentence_balanced_multidoc_v2", "context_balanced_multidoc_v2"]
    elif args.variant == "scaled_balanced":
        variants = ["sentence_balanced_scaled_v3", "context_balanced_scaled_v3"]
    elif args.variant == "scaled_ablation":
        variants = ["sentence_balanced_scaled_v3", "context_balanced_scaled_v3",
                    "sentence_unweighted_scaled_v3", "context_unweighted_scaled_v3"]
    else:
        variants = [args.variant]
    dataset_names = {VARIANTS[v]["dataset"] for v in variants}
    if len(dataset_names) != 1:
        raise ValueError("A comparison must use one dataset.")
    data_dir = PROJECT_ROOT / "data/processed" / dataset_names.pop()
    # Check all requested adapters before spending time on inference.
    for variant in variants:
        directory = VARIANTS[variant]["directory"]
        adapter_path = PROJECT_ROOT / "outputs/checkpoints" / directory / "final_adapter"
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / filename).is_file():
                raise FileNotFoundError(f"Missing adapter file: {adapter_path / filename}")
        check_training_metadata(VARIANTS[variant], data_dir)
    data, dataset_path = load_split(args.split, data_dir)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Split: {args.split}; dataset: {dataset_path}")
    print(f"Pair counts: {data['pair'].value_counts().to_dict()}")
    reports = []
    for variant in variants:
        torch.manual_seed(42)
        reports.append(evaluate_variant(
            variant, data, dataset_hash, args.split, dataset_path
        ))

    print(f"\n=== {args.split.upper()} COMPARISON ===")
    for report in reports:
        print(f"\nModel: {report['model']}")
        for pair, metrics in report["pairs"].items():
            print(
                f"{pair}: BLEU={metrics['sacrebleu']:.2f}, "
                f"chrF={metrics['chrf']:.2f}, N={metrics['num_examples']}"
            )
        print(
            f"Macro BLEU={report['macro_average_sacrebleu']:.2f}, "
            f"Macro chrF={report['macro_average_chrf']:.2f}"
        )


if __name__ == "__main__":
    main()
