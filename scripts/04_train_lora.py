import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".cache" / "huggingface"),
)

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DATA_DIR = PROJECT_ROOT / "data/processed/europarl_scaled_v3"

LANGUAGE_CODES = {
    "en": "en_XX",
    "de": "de_DE",
    "fr": "fr_XX",
}

MAX_LENGTH = 128


def normalized_pair_weights(pair_counts):
    """Normalize by the training-distribution mean, not each microbatch."""
    total = sum(pair_counts.values())
    raw = {pair: math.sqrt(total / count) for pair, count in pair_counts.items()}
    mean_weight = sum(pair_counts[pair] * raw[pair] for pair in raw) / total
    normalized = {pair: weight / mean_weight for pair, weight in raw.items()}
    return raw, mean_weight, normalized


def reduce_pair_loss(loss_per_sample, weights, training):
    if training:
        # Fixed globally normalized weights remain effective for batch size 1.
        return (loss_per_sample * weights).mean()
    # Keep exactly the previous validation objective/checkpoint selection.
    return (loss_per_sample * weights).sum() / weights.sum().clamp(min=1e-8)


def verify_weighting():
    """Check gradients, batch-size-1 weighting, and accumulation on CPU."""
    _, _, weights = normalized_pair_weights({"en-de": 800, "en-fr": 400})
    assert abs((800 * weights["en-de"] + 400 * weights["en-fr"]) / 1200 - 1) < 1e-12
    assert abs(weights["en-fr"] / weights["en-de"] - math.sqrt(2)) < 1e-12
    # A weighted singleton must have a scaled gradient, not gradient 1.
    for pair, factor in weights.items():
        sample = torch.tensor([2.0], requires_grad=True)
        loss = reduce_pair_loss(sample, torch.tensor([factor]), training=True)
        loss.backward()
        assert torch.allclose(sample.grad, torch.tensor([factor]))
    # Eight accumulated singleton gradients must equal a full-batch gradient.
    factors = torch.tensor([weights["en-de"], weights["en-fr"]] * 4)
    values = torch.arange(1.0, 9.0, requires_grad=True)
    reduce_pair_loss(values, factors, training=True).backward()
    batch_gradient = values.grad.clone()
    values = torch.arange(1.0, 9.0, requires_grad=True)
    for i in range(8):
        (reduce_pair_loss(values[i:i+1], factors[i:i+1], training=True) / 8).backward()
    assert torch.allclose(values.grad, batch_gradient)
    print("Loss checks: PASS (singleton weights and gradient accumulation)")
    print(f"Expected normalized weights for 800/400: {weights}")


class PairAwareCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        pairs = [feature["pair"] for feature in features]

        model_features = [
            {
                key: value
                for key, value in feature.items()
                if key != "pair"
            }
            for feature in features
        ]

        batch = self.base_collator(model_features)
        batch["pair"] = pairs
        return batch


class BalancedLossSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, pair_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_weights = pair_weights or {}

        # Our balanced loss is a per-microbatch mean.
        # Let Trainer scale it for gradient accumulation.
        self.loss_is_scaled_for_ga = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        pairs = inputs.pop("pair", None)

        if pairs is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="none",
        )

        loss_per_token = loss_function(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        loss_per_token = loss_per_token.reshape(
            labels.size(0),
            labels.size(1),
        )

        valid_token_mask = (labels != -100).float()

        loss_per_sample = (
            (loss_per_token * valid_token_mask).sum(dim=1)
            / valid_token_mask.sum(dim=1).clamp(min=1)
        )

        weights = torch.tensor(
            [
                self.pair_weights[pair]
                for pair in pairs
            ],
            device=loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )

        loss = reduce_pair_loss(loss_per_sample, weights, training=model.training)

        if return_outputs:
            return loss, outputs

        return loss


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--input-mode", choices=("sentence", "context_v1"), default="context_v1")
    parser.add_argument("--loss-mode", choices=("balanced", "unweighted"), default="balanced")
    parser.add_argument("--check-loss-only", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest complete checkpoint in --output-dir.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Directory containing train.csv and validation.csv; relative to project root.",
    )
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--check-data-only", action="store_true",
        help="Validate train/validation files and exit before loading a tokenizer or model.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Positive value runs a smoke test. "
            "-1 trains for the configured epochs."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()
    if args.max_length < 4:
        parser.error("--max-length must be at least 4.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.max_steps != -1 and args.max_steps <= 0:
        parser.error("--max-steps must be -1 or positive.")
    if args.resume and args.max_steps != -1:
        parser.error("--resume is only supported for full-epoch training.")
    return args


def find_resume_checkpoint(output_dir):
    """Select a checkpoint that includes optimizer, schedule, RNG and model."""
    if (output_dir / "final_adapter" / "adapter_model.safetensors").is_file() and (
        output_dir / "run_metadata.json"
    ).is_file():
        raise FileExistsError(f"Training already completed: {output_dir}")
    required = (
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
        "scheduler.pt", "trainer_state.json", "rng_state.pth",
    )
    candidates = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint-"):
                continue
            number = path.name.removeprefix("checkpoint-")
            if not number.isdecimal() or not all((path / name).is_file() for name in required):
                continue
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            if state.get("global_step") == int(number):
                candidates.append((int(number), path))
    if not candidates:
        raise FileNotFoundError(
            f"No complete checkpoint in {output_dir}. "
            "Keep this directory and choose a new --output-dir to start over."
        )
    return max(candidates)[1]


def inspect_training_data(data_dir):
    """Read only train/validation; check structure, provenance and split isolation.

    These checks do not establish semantic equivalence of bilingual sentences.
    """
    required = {"pair", "split", "doc_id", "src_sent_id", "tgt_sent_id",
                "previous_src", "src", "tgt"}
    split_info = {}
    documents = {}
    sources = {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.csv"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty dataset: {path}")
        keys = set()
        for line, row in enumerate(rows, start=2):
            if any(not (row.get(column) or "").strip() for column in required):
                raise ValueError(f"{path}: blank required field on line {line}")
            if row["split"] != split or row["pair"] not in ("en-de", "en-fr"):
                raise ValueError(f"{path}: unexpected split or language pair on line {line}")
            key = tuple(row[column] for column in ("pair", "doc_id", "src_sent_id", "tgt_sent_id"))
            if key in keys:
                raise ValueError(f"{path}: duplicate example key on line {line}")
            keys.add(key)
        counts = Counter(row["pair"] for row in rows)
        if set(counts) != {"en-de", "en-fr"}:
            raise ValueError(f"{path}: expected both en-de and en-fr")
        documents[split] = {row["doc_id"] for row in rows}
        sources[split] = {" ".join(row["src"].split()) for row in rows}
        split_info[split] = {
            "path": str(path), "sha256": digest, "num_examples": len(rows),
            "num_documents": len(documents[split]), "pair_counts": dict(counts),
        }
    if documents["train"] & documents["validation"]:
        raise ValueError("Train/validation document leakage detected.")
    if sources["train"] & sources["validation"]:
        raise ValueError("Train/validation current-source overlap detected.")

    # Verify the builder's manifest when present, without opening test.csv.
    summary_path = data_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = summary.get("csv_sha256", {})
        for split, info in split_info.items():
            if split in expected and expected[split] != info["sha256"]:
                raise ValueError(f"{split}.csv differs from the builder's summary.json.")
    print(f"Data directory: {data_dir}")
    for split, info in split_info.items():
        print(f"{split}: {info['num_examples']} examples, {info['num_documents']} documents, "
              f"pairs={info['pair_counts']}")
        print(f"{split} SHA256: {info['sha256']}")
    print("Data checks: PASS (structure, train/validation isolation, available manifest hashes)")
    return split_info


def preprocess_example(example, tokenizer, input_mode, max_length=MAX_LENGTH):
    target_language = example["pair"].split("-")[1]

    tokenizer.src_lang = LANGUAGE_CODES["en"]
    tokenizer.tgt_lang = LANGUAGE_CODES[target_language]
    tokenizer.truncation_side = "right"

    # Keep the current sentence first so truncation removes
    # previous-sentence context before the sentence to translate.
    contextual_source = (
        f"{example['src']} "
        f"{tokenizer.eos_token} "
        f"{example['previous_src']}"
    )
    if input_mode == "sentence":
        contextual_source = example["src"]

    encoded = tokenizer(
        contextual_source,
        text_target=example["tgt"],
        max_length=max_length,
        truncation=True,
    )

    return encoded


def main():
    args = parse_arguments()
    verify_weighting()
    if args.check_loss_only:
        return

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    dataset_info = inspect_training_data(data_dir)
    if args.check_data_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(42)

    experiment = "sentence_lora" if args.input_mode == "sentence" else "context_lora"
    experiment += "_" + args.loss_mode
    if data_dir == (PROJECT_ROOT / "data/processed/europarl_multidoc_v2").resolve():
        experiment += "_multidoc_v2"
    requested_output = args.output_dir or Path("outputs/checkpoints") / experiment
    output_dir = (PROJECT_ROOT / requested_output).resolve()
    resume_checkpoint = find_resume_checkpoint(output_dir) if args.resume else None
    if resume_checkpoint:
        state = json.loads((resume_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        steps_per_epoch = math.ceil(dataset_info["train"]["num_examples"] / 8)
        expected_steps = math.ceil(args.epochs * steps_per_epoch)
        if state.get("max_steps") != expected_steps:
            raise ValueError(
                f"Checkpoint expects {state.get('max_steps')} total steps; "
                f"current configuration expects {expected_steps}. Keep the original training settings."
            )
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_test = args.max_steps > 0

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Output: {output_dir}")
    print(f"Smoke test: {smoke_test}")
    print(f"Input mode: {args.input_mode}")
    print(f"Loss mode: {args.loss_mode}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
    print(f"Source/target max length: {args.max_length}")

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(data_dir / "train.csv"),
            "validation": str(
                data_dir / "validation.csv"
            ),
        },
    )

    tokenizer = MBart50TokenizerFast.from_pretrained(
        MODEL_NAME
    )

    removable_columns = [
        column
        for column in dataset["train"].column_names
        if column != "pair"
    ]

    tokenized_dataset = dataset.map(
        lambda example: preprocess_example(
            example,
            tokenizer,
            args.input_mode,
            args.max_length,
        ),
        remove_columns=removable_columns,
        load_from_cache_file=False,
        desc="Tokenizing Europarl",
    )

    pair_counts = Counter(
        tokenized_dataset["train"]["pair"]
    )
    raw_weights, normalization, normalized_weights = normalized_pair_weights(pair_counts)
    pair_weights = normalized_weights if args.loss_mode == "balanced" else {
        pair: 1.0 for pair in pair_counts
    }

    print(f"Training pair counts: {dict(pair_counts)}")
    print(f"Pair weights: {pair_weights}")
    print(f"Training-distribution mean raw weight: {normalization:.6f}")

    base_model = (
        MBartForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
        )
    )

    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )
    model.print_trainable_parameters()

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    collator = PairAwareCollator(base_collator)

    strategy = "no" if smoke_test else "epoch"

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        warmup_steps=2 if smoke_test else 45,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        fp16=True,
        logging_steps=5 if smoke_test else 25,
        eval_strategy=strategy,
        save_strategy=strategy,
        save_total_limit=2,
        load_best_model_at_end=not smoke_test,
        metric_for_best_model=(
            "eval_loss" if not smoke_test else None
        ),
        greater_is_better=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
    )

    trainer = BalancedLossSeq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        pair_weights=pair_weights,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    final_adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    trainer.save_state()

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    metadata = {
        "base_model": MODEL_NAME,
        "experiment": experiment,
        "data_directory": str(data_dir),
        "dataset_files": dataset_info,
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "input_mode": args.input_mode,
        "input_format": "src" if args.input_mode == "sentence" else "src </s> previous_src",
        "source_truncation_side": "right",
        "target_truncation_side": "right",
        "loss_mode": args.loss_mode,
        "training_objective": "mean(normalized_pair_weight * mean_valid_token_CE_per_sentence)",
        "validation_objective": "legacy_per_batch_weighted_mean_of_sentence_CE",
        "raw_pair_weights": raw_weights,
        "weight_normalization": normalization,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "seed": 42,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "training_mode": (
            "smoke_test"
            if smoke_test
            else "full_training"
        ),
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "source_max_length": args.max_length,
        "target_max_length": args.max_length,
        "pair_counts": dict(pair_counts),
        "pair_weights": pair_weights,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "v_proj",
            ],
        },
        "train_metrics": train_result.metrics,
        "elapsed_minutes": elapsed_minutes,
        "peak_allocated_vram_gb": peak_vram,
    }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"Elapsed: {elapsed_minutes:.2f} minutes")
    print(f"Peak allocated VRAM: {peak_vram:.2f} GB")
    print(f"Adapter: {final_adapter_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
