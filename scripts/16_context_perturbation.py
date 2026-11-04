"""Test whether the frozen v3 context adapter uses its preceding sentence.

Compare its saved correct-context beam predictions with fresh inferences using
an unrelated previous sentence or no previous sentence. No model is trained and
no paid API is called. Run on validation before considering the frozen test set.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

MODEL = "facebook/mbart-large-50-many-to-many-mmt"
ADAPTER = ROOT / "outputs/checkpoints/context_lora_balanced_scaled_v3/final_adapter"
METADATA = ADAPTER.parent / "run_metadata.json"
DATA_DIR = ROOT / "data/processed/europarl_scaled_v3"
KEY = ["pair", "doc_id", "src_sent_id", "tgt_sent_id"]
PAIRS = {"en-de": "de_DE", "en-fr": "fr_XX"}
BEAMS = 4
MAX_LENGTH = 128
BATCH_SIZE = 4


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(split, limit):
    data_path = DATA_DIR / f"{split}.csv"
    baseline_path = ROOT / f"outputs/predictions/context_lora_balanced_scaled_v3_{split}.csv"
    manifest = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if manifest["csv_sha256"][split] != sha256(data_path):
        raise ValueError("Dataset hash differs from summary.json")
    if (metadata.get("input_format") != "src </s> previous_src"
            or metadata.get("source_truncation_side") != "right"
            or metadata.get("max_length") != MAX_LENGTH
            or metadata.get("training_mode") != "full_training"):
        raise ValueError("Adapter metadata does not match the evaluation setup")
    for name in ("train", "validation"):
        if metadata["dataset_files"][name]["sha256"] != sha256(DATA_DIR / f"{name}.csv"):
            raise ValueError(f"Adapter was trained on another {name} dataset")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER / name).is_file():
            raise FileNotFoundError(ADAPTER / name)

    data = pd.read_csv(data_path, keep_default_na=False, dtype=str)
    baseline = pd.read_csv(baseline_path, keep_default_na=False, dtype=str)
    required = set(KEY + ["split", "src", "previous_src", "tgt"])
    if not required.issubset(data) or not (required | {"prediction"}).issubset(baseline):
        raise ValueError("Dataset or saved baseline is missing required columns")
    if len(data) != len(baseline) or data.empty or data[KEY].duplicated().any() or baseline[KEY].duplicated().any():
        raise ValueError("Incomplete or duplicate dataset/baseline keys")
    if set(data.pair) != set(PAIRS) or not data.split.eq(split).all():
        raise ValueError("Unexpected pairs or split")
    aligned = data.merge(baseline, on=KEY, how="left", validate="one_to_one", suffixes=("", "_saved"), indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("Saved baseline does not cover the entire dataset")
    for field in ("split", "src", "previous_src", "tgt"):
        if not aligned[field].eq(aligned[f"{field}_saved"]).all():
            raise ValueError(f"Saved baseline {field} differs from dataset")
    if aligned.prediction.eq("").any() or aligned.src.eq("").any() or aligned.tgt.eq("").any():
        raise ValueError("Blank source, reference, or baseline prediction")
    if limit:
        aligned = pd.concat([aligned[aligned.pair == pair].head(limit) for pair in PAIRS], ignore_index=True)
    return aligned, data_path, baseline_path


def shuffled_context(data, seed):
    """Rotate document IDs within each pair; donor remains in the same split."""
    result = pd.Series(index=data.index, dtype=str)
    donors = pd.Series(index=data.index, dtype=str)
    for pair_index, pair in enumerate(PAIRS):
        rows = data[data.pair == pair]
        documents = sorted(rows.doc_id.unique())
        if len(documents) < 2:
            raise ValueError(f"Need two documents in {pair} to shuffle context")
        rng = random.Random(seed + pair_index)
        offset = rng.randrange(1, len(documents))
        mapping = {doc: documents[(i + offset) % len(documents)] for i, doc in enumerate(documents)}
        pool = {doc: rows.loc[rows.doc_id == doc, "previous_src"].tolist() for doc in documents}
        positions = {doc: 0 for doc in documents}
        for idx, row in rows.iterrows():
            donor = mapping[row.doc_id]
            values = pool[donor]
            text = values[positions[donor] % len(values)]
            positions[donor] += 1
            result.at[idx] = text
            donors.at[idx] = donor
    if result.isna().any() or donors.isna().any() or (donors == data.doc_id).any():
        raise AssertionError("Context shuffle failed to exclude the source document")
    return result, donors


def generate(data, previous, model, tokenizer):
    predictions = pd.Series(index=data.index, dtype=str)
    for pair, code in PAIRS.items():
        rows = data[data.pair == pair]
        for start in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"{pair}"):
            batch = rows.iloc[start:start + BATCH_SIZE]
            tokenizer.src_lang = "en_XX"
            # Keep the trained separator and the target sentence at the front.
            texts = [f"{src} {tokenizer.eos_token} {previous.loc[index]}"
                     for index, src in zip(batch.index, batch.src)]
            encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH).to("cuda")
            with torch.inference_mode():
                ids = model.generate(**encoded,
                                     forced_bos_token_id=tokenizer.convert_tokens_to_ids(code),
                                     num_beams=BEAMS, do_sample=False, max_length=MAX_LENGTH)
            predictions.loc[batch.index] = tokenizer.batch_decode(ids, skip_special_tokens=True)
    if predictions.isna().any() or predictions.eq("").any():
        raise ValueError("Generation returned missing or empty predictions")
    return predictions


def score(data):
    metrics = {"pairs": {}, "macro": {}}
    for pair in PAIRS:
        rows = data[data.pair == pair]
        refs = rows.tgt.tolist()
        bleu, chrf = BLEU(), CHRF()
        current = {"num_examples": len(rows), "bleu_signature": None, "chrf_signature": None}
        for variant in ("correct", "shuffled", "empty"):
            preds = rows[f"{variant}_prediction"].tolist()
            current[variant] = {"bleu": bleu.corpus_score(preds, [refs]).score,
                                "chrf": chrf.corpus_score(preds, [refs]).score,
                                "changed_from_correct": sum(a != b for a, b in zip(preds, rows.correct_prediction))}
            current["bleu_signature"] = str(bleu.get_signature())
            current["chrf_signature"] = str(chrf.get_signature())
        metrics["pairs"][pair] = current
    for variant in ("correct", "shuffled", "empty"):
        metrics["macro"][variant] = {metric: sum(metrics["pairs"][p][variant][metric] for p in PAIRS) / len(PAIRS)
                                      for metric in ("bleu", "chrf")}
    for control in ("shuffled", "empty"):
        metrics["macro"][f"correct_minus_{control}"] = {
            metric: metrics["macro"]["correct"][metric] - metrics["macro"][control][metric]
            for metric in ("bleu", "chrf")}
    return metrics


def document_intervals(data, seed, reps=1000):
    """Paired document bootstrap of macro mean sentence chrF differences."""
    chrf = CHRF()
    for variant in ("correct", "shuffled", "empty"):
        data[f"{variant}_sentence_chrf"] = [
            chrf.sentence_score(pred, [ref]).score
            for pred, ref in zip(data[f"{variant}_prediction"], data.tgt)
        ]
    documents = sorted(data.doc_id.unique())
    if set(data.loc[data.pair == "en-de", "doc_id"]) != set(data.loc[data.pair == "en-fr", "doc_id"]):
        raise ValueError("Both language pairs must have the same document IDs")
    rng = random.Random(seed + 1000)
    grouped = {}
    for pair in PAIRS:
        for doc in documents:
            rows = data[(data.pair == pair) & (data.doc_id == doc)]
            grouped[(pair, doc)] = {
                "count": len(rows),
                **{control: (rows.correct_sentence_chrf - rows[f"{control}_sentence_chrf"]).sum()
                   for control in ("shuffled", "empty")},
            }

    def contrasts(draw):
        counts = Counter(draw)
        result = {}
        for control in ("shuffled", "empty"):
            pair_deltas = []
            for pair in PAIRS:
                numerator = denominator = 0
                for doc, repetitions in counts.items():
                    stats = grouped[(pair, doc)]
                    numerator += stats[control] * repetitions
                    denominator += stats["count"] * repetitions
                pair_deltas.append(numerator / denominator)
            result[f"correct_minus_{control}"] = sum(pair_deltas) / len(pair_deltas)
        return result

    observed = contrasts(documents)
    samples = {key: [] for key in observed}
    for _ in range(reps):
        draw = rng.choices(documents, k=len(documents))
        values = contrasts(draw)
        for key in samples:
            samples[key].append(values[key])
    result = {}
    for key, values in samples.items():
        ordered = sorted(values)
        result[key] = {"observed": observed[key], "ci_95": [ordered[int(.025 * (reps - 1))],
                                                         ordered[int(.975 * (reps - 1))]]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-pair", type=int, default=0, help="Small smoke subset; not for conclusions")
    args = parser.parse_args()
    if args.limit_per_pair < 0:
        parser.error("--limit-per-pair must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data, data_path, baseline_path = load_inputs(args.split, args.limit_per_pair)
    shuffled, donor_docs = shuffled_context(data, args.seed)
    data = data.copy()
    data["correct_prediction"] = data["prediction"]
    data["shuffled_previous_src"] = shuffled
    data["shuffled_donor_doc_id"] = donor_docs
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL)
    tokenizer.src_lang = "en_XX"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    model = MBartForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload().eval()
    torch.manual_seed(args.seed)
    data["shuffled_prediction"] = generate(data, shuffled, model, tokenizer)
    data["empty_prediction"] = generate(data, pd.Series("", index=data.index), model, tokenizer)
    report = score(data)
    report["document_bootstrap_mean_sentence_chrf"] = (
        None if args.limit_per_pair else document_intervals(data, args.seed)
    )
    report["config"] = {
        "split": args.split, "seed": args.seed, "limit_per_pair": args.limit_per_pair,
        "dataset_sha256": sha256(data_path), "baseline_sha256": sha256(baseline_path),
        "adapter": str(ADAPTER), "model": MODEL,
        "input_format": "src </s> previous_src", "truncation_side": "right",
        "max_length": MAX_LENGTH, "num_beams": BEAMS,
        "shuffling": "same-pair, different-document deterministic rotation; validation only",
        "empty_input_format": "src </s>", "correct_predictions": "saved frozen beam baseline",
        "sacrebleu_version": sacrebleu.__version__,
        "bootstrap_reps": 1000,
    }
    suffix = f"_smoke{args.limit_per_pair}" if args.limit_per_pair else ""
    stem = f"context_perturbation_{args.split}_scaled_v3{suffix}"
    prediction_path = ROOT / "outputs/analysis/context_perturbation" / f"{stem}.csv"
    metric_path = ROOT / "outputs/metrics" / f"{stem}.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    data[KEY + ["previous_src", "shuffled_previous_src", "shuffled_donor_doc_id", "src", "tgt",
                "correct_prediction", "shuffled_prediction", "empty_prediction"]].to_csv(
                    prediction_path, index=False, encoding="utf-8")
    metric_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== CONTEXT PERTURBATION: {args.split.upper()} ===")
    for pair, result in report["pairs"].items():
        print(f"{pair}: N={result['num_examples']}")
        for variant in ("correct", "shuffled", "empty"):
            m = result[variant]
            print(f"  {variant}: BLEU={m['bleu']:.2f}; chrF={m['chrf']:.2f}; changed={m['changed_from_correct']}")
    for name, delta in report["macro"].items():
        print(f"macro {name}: BLEU={delta['bleu']:+.2f}; chrF={delta['chrf']:+.2f}")
    if report["document_bootstrap_mean_sentence_chrf"]:
        for name, value in report["document_bootstrap_mean_sentence_chrf"].items():
            low, high = value["ci_95"]
            print(f"macro mean sentence chrF {name}: {value['observed']:+.2f} [{low:+.2f}, {high:+.2f}]")
    print(f"Predictions: {prediction_path}\nMetrics: {metric_path}")


if __name__ == "__main__":
    main()
