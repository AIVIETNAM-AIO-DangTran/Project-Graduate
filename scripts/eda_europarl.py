import os
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".cache" / "huggingface"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import MBart50TokenizerFast


MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DATA_DIR = PROJECT_ROOT / "data/processed/europarl_multidoc_v2"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/eda_multidoc_v2"
METRIC_DIR = PROJECT_ROOT / "outputs/metrics/europarl_multidoc_v2"

SPLITS = ("train", "validation", "test")
PAIRS = ("en-de", "en-fr")
LANGUAGE_CODES = {
    "en": "en_XX",
    "de": "de_DE",
    "fr": "fr_XX",
}
TEXT_COLUMNS = ("previous_src", "src", "tgt")
TOKEN_COLUMNS = (
    "previous_src_tokens",
    "src_tokens",
    "tgt_tokens",
    "context_src_tokens",
)
TRUNCATION_LIMITS = (128, 256)
TOKEN_BATCH_SIZE = 64


def load_splits():
    frames = []
    manifest = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset split: {path}")
        expected = manifest["csv_sha256"][split]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset differs from multidoc v2 manifest: {path}")

        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    required = {"pair", "split", *TEXT_COLUMNS}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return data


def count_words(series):
    return (
        series.fillna("")
        .astype(str)
        .str.split()
        .str.len()
        .astype(int)
    )


def get_token_lengths(tokenizer, texts, language_code):
    tokenizer.src_lang = language_code
    lengths = []

    for start in range(0, len(texts), TOKEN_BATCH_SIZE):
        batch = texts[start : start + TOKEN_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    return lengths


def add_length_columns(data, tokenizer):
    result = data.copy()

    for column in TEXT_COLUMNS:
        result[f"{column}_words"] = count_words(result[column])

    for column in TOKEN_COLUMNS:
        result[column] = 0

    for pair in PAIRS:
        indices = result.index[result["pair"] == pair]
        subset = result.loc[indices]

        result.loc[indices, "previous_src_tokens"] = get_token_lengths(
            tokenizer,
            subset["previous_src"].astype(str).tolist(),
            LANGUAGE_CODES["en"],
        )

        result.loc[indices, "src_tokens"] = get_token_lengths(
            tokenizer,
            subset["src"].astype(str).tolist(),
            LANGUAGE_CODES["en"],
        )

        target_language = pair.split("-")[1]
        result.loc[indices, "tgt_tokens"] = get_token_lengths(
            tokenizer,
            subset["tgt"].astype(str).tolist(),
            LANGUAGE_CODES[target_language],
        )

        # Put the current sentence first so right-side truncation preserves
        # the sentence that must be translated. The previous sentence is
        # supplied after an mBART EOS separator as document context.
        context_inputs = [
            f"{current} {tokenizer.eos_token} {previous}"
            for current, previous in zip(
                subset["src"].astype(str),
                subset["previous_src"].astype(str),
            )
        ]

        result.loc[indices, "context_src_tokens"] = get_token_lengths(
            tokenizer,
            context_inputs,
            LANGUAGE_CODES["en"],
        )

    for column in TOKEN_COLUMNS:
        result[column] = result[column].astype(int)

    return result


def iter_scopes(data):
    for split in ("all", *SPLITS):
        split_data = data if split == "all" else data[data["split"] == split]

        for pair in ("all", *PAIRS):
            scoped = (
                split_data
                if pair == "all"
                else split_data[split_data["pair"] == pair]
            )

            if not scoped.empty:
                yield split, pair, scoped


def build_length_summary(data):
    rows = []
    columns = (
        "previous_src_words",
        "src_words",
        "tgt_words",
        *TOKEN_COLUMNS,
    )

    for split, pair, scoped in iter_scopes(data):
        for column in columns:
            values = scoped[column]
            rows.append(
                {
                    "split": split,
                    "pair": pair,
                    "measure": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": int(values.min()),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                    "max": int(values.max()),
                }
            )

    return pd.DataFrame(rows)


def build_truncation_summary(data):
    rows = []
    input_columns = {
        "sentence": "src_tokens",
        "context": "context_src_tokens",
    }

    for split, pair, scoped in iter_scopes(data):
        for input_type, column in input_columns.items():
            for limit in TRUNCATION_LIMITS:
                num_truncated = int((scoped[column] > limit).sum())
                rows.append(
                    {
                        "split": split,
                        "pair": pair,
                        "input_type": input_type,
                        "max_length": limit,
                        "num_examples": len(scoped),
                        "num_truncated": num_truncated,
                        "truncated_percent": (
                            100.0 * num_truncated / len(scoped)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_tables(data, length_summary, truncation_summary):
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    pair_counts = (
        data.groupby(["split", "pair"], sort=False)
        .size()
        .rename("num_examples")
        .reset_index()
    )

    outlier_columns = [
        "pair",
        "split",
        "doc_id",
        "previous_src",
        "src",
        "tgt",
        "previous_src_tokens",
        "src_tokens",
        "tgt_tokens",
        "context_src_tokens",
    ]
    outlier_columns = [
        column for column in outlier_columns if column in data.columns
    ]
    outliers = (
        data.sort_values("context_src_tokens", ascending=False)
        .head(20)[outlier_columns]
    )

    pair_counts.to_csv(
        METRIC_DIR / "eda_pair_counts.csv",
        index=False,
        encoding="utf-8",
    )
    length_summary.to_csv(
        METRIC_DIR / "eda_length_summary.csv",
        index=False,
        encoding="utf-8",
    )
    truncation_summary.to_csv(
        METRIC_DIR / "eda_truncation_summary.csv",
        index=False,
        encoding="utf-8",
    )
    outliers.to_csv(
        METRIC_DIR / "eda_longest_context_examples.csv",
        index=False,
        encoding="utf-8",
    )

    return pair_counts


def save_figures(data, pair_counts, truncation_summary):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=pair_counts,
        x="split",
        y="num_examples",
        hue="pair",
        order=list(SPLITS),
        hue_order=list(PAIRS),
    )
    plt.title("Europarl multidoc v2 examples by split and language pair")
    plt.xlabel("Split")
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_pair_counts.png", dpi=200)
    plt.close()

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    histogram_specs = (
        ("previous_src_tokens", "Previous-source token length"),
        ("src_tokens", "Current-source token length"),
        ("tgt_tokens", "Target token length"),
        ("context_src_tokens", "Context-input token length"),
    )

    for axis, (column, title) in zip(axes.flat, histogram_specs):
        sns.histplot(
            data=data,
            x=column,
            hue="pair",
            bins=35,
            element="step",
            stat="density",
            common_norm=False,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("mBART tokens")
        axis.set_ylabel("Density")

    figure.suptitle("Europarl multidoc v2 token-length distributions", y=1.01)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "02_token_length_distributions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=data,
        x="pair",
        y="context_src_tokens",
        hue="split",
        order=list(PAIRS),
        hue_order=list(SPLITS),
        showfliers=False,
    )
    plt.title("Context-input token length by pair and split")
    plt.xlabel("Language pair")
    plt.ylabel("mBART tokens")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_context_length_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="src_tokens",
        y="tgt_tokens",
        hue="pair",
        style="split",
        alpha=0.6,
    )
    maximum = int(max(data["src_tokens"].max(), data["tgt_tokens"].max()))
    plt.plot([0, maximum], [0, maximum], "--", color="gray", linewidth=1)
    plt.title("Current-source versus target token length")
    plt.xlabel("Source mBART tokens")
    plt.ylabel("Target mBART tokens")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_source_target_tokens.png", dpi=200)
    plt.close()

    plot_data = truncation_summary[
        (truncation_summary["split"] == "all")
        & (truncation_summary["pair"] != "all")
        & (truncation_summary["input_type"] == "context")
    ].copy()
    plot_data["max_length"] = plot_data["max_length"].astype(str)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_data,
        x="max_length",
        y="truncated_percent",
        hue="pair",
        hue_order=list(PAIRS),
    )
    plt.title("Context inputs exceeding each maximum length")
    plt.xlabel("Maximum input length")
    plt.ylabel("Examples truncated (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_context_truncation_rate.png", dpi=200)
    plt.close()


def print_summary(data, pair_counts, length_summary, truncation_summary):
    print("\n=== DATASET COUNTS ===")
    print(pair_counts.to_string(index=False))

    context_lengths = length_summary[
        (length_summary["split"] == "all")
        & (length_summary["pair"] == "all")
        & (length_summary["measure"] == "context_src_tokens")
    ]

    print("\n=== OVERALL CONTEXT TOKEN LENGTH ===")
    print(
        context_lengths[
            ["count", "mean", "p50", "p90", "p95", "p99", "max"]
        ].to_string(index=False)
    )

    overall_truncation = truncation_summary[
        (truncation_summary["split"] == "all")
        & (truncation_summary["pair"] == "all")
    ]

    print("\n=== OVERALL TRUNCATION RATES ===")
    print(
        overall_truncation[
            [
                "input_type",
                "max_length",
                "num_examples",
                "num_truncated",
                "truncated_percent",
            ]
        ].to_string(index=False)
    )

    context_rates = overall_truncation[
        overall_truncation["input_type"] == "context"
    ].set_index("max_length")["truncated_percent"]

    if context_rates.loc[128] <= 2.0:
        recommendation = 128
    elif context_rates.loc[256] <= 2.0:
        recommendation = 256
    else:
        recommendation = ">256 or controlled context truncation"

    print(f"\nRecommended context max length: {recommendation}")
    print(f"Analyzed examples: {len(data)}")
    print(f"Tables: {METRIC_DIR}")
    print(f"Figures: {FIGURE_DIR}")


def main():
    print(f"Loading Europarl multidoc v2 data from: {DATA_DIR}")
    data = load_splits()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)

    print("Computing word and mBART token lengths...")
    data = add_length_columns(data, tokenizer)

    length_summary = build_length_summary(data)
    truncation_summary = build_truncation_summary(data)
    pair_counts = save_tables(data, length_summary, truncation_summary)
    save_figures(data, pair_counts, truncation_summary)
    print_summary(data, pair_counts, length_summary, truncation_summary)


if __name__ == "__main__":
    main()
