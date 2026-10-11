# Context-aware multilingual machine translation (CIS 600)

This project evaluates document context and local LLM post-editing for English–German and English–French translation. It uses `facebook/mbart-large-50-many-to-many-mmt` with LoRA adapters trained on Europarl v2, and `qwen2.5:7b` through Ollama for optional post-editing. The experimental setup follows the [project proposal](#scope-and-limitations); the earlier small pilot is retained only for data provenance and is not used for the final results.

## Experiment

| Arm | mBART input | LLM post-editing |
| --- | --- | --- |
| A | Current English sentence | No |
| B | Current English sentence, `</s>`, preceding English sentence | No |
| C | Same as A | Yes |
| D | Same as B | Yes |

A and B use the same pair-balanced LoRA training objective. C and D use Qwen2.5:7b at temperature 0 to post-edit the frozen A/B predictions. The reference translation is never sent to the LLM. mBART decoding uses four-beam search with maximum length 128. See the JSON reports for full model, data, decoding, and metric provenance.

The frozen Europarl multidoc v2 dataset has 1,200 training examples from 40 documents, 150 validation examples from 10 documents, and 150 test examples from 10 documents. Each split has both pairs: en-de 800/100/100 and en-fr 400/50/50 (train/validation/test). The pilot CSVs are retained because the dataset builder uses their document IDs to exclude legacy documents.

## Main results

Macro scores are unweighted means of the en-de and en-fr scores, so the pairs each contribute 50% despite their different example counts. BLEU and chrF are calculated at the corpus level within each language pair; COMET is averaged over sentence scores within each pair.

| Split | Arm | SacreBLEU | chrF | COMET |
| --- | --- | ---: | ---: | ---: |
| Validation | A | 31.08 | 59.26 | 0.8645 |
| Validation | B | 31.66 | 59.52 | 0.8653 |
| Validation | C | 31.58 | 59.21 | 0.8615 |
| Validation | D | 31.49 | 59.60 | 0.8692 |
| Test | A | 31.05 | 58.37 | 0.8521 |
| Test | B | 31.30 | 58.55 | 0.8528 |
| Test | C | 30.30 | 58.01 | 0.8539 |
| Test | D | 31.24 | 58.60 | 0.8582 |

Raw pair scores and metric signatures are in `outputs/metrics/`. A targeted qualitative review is in `qualitative_analysis.md`. The observed differences are small; these tables alone do not establish statistical significance or that context was used to resolve a particular ambiguity.

### Sampling ablation (validation only)

Using arm B's adapter, the optional decoding experiment generates three candidates with top-k (k=50) or top-p (p=0.9) sampling at temperature 0.8, then asks Qwen2.5:7b to select among them and the frozen beam output without access to references. Both sampling methods reduced macro BLEU relative to beam search (31.66 to 30.86 for top-k and 31.39 for top-p). Macro COMET rose from 0.8653 to 0.8671 and 0.8676, respectively. One large improvement accounts for most of the top-p COMET gain. This is a validation ablation, not an additional test-set result. Details: `outputs/decoding/multidoc_v2/` and `outputs/analysis/decoding_validation_multidoc_v2/`.

## Repository layout

- `data/processed/europarl_multidoc_v2/`: frozen train, validation, test CSVs and manifest with SHA-256 hashes.
- `data/processed/europarl_pilot/`: legacy pilot CSVs used only to exclude its documents when rebuilding v2.
- `scripts/`: download, build, train, evaluate, post-edit, sampling, COMET, and EDA programs.
- `notebooks/01_europarl_eda.ipynb` and `notebooks/02_experiment_analysis.ipynb`: saved-data exploration and experiment analysis.
- `outputs/checkpoints/*_multidoc_v2/final_adapter/`: final LoRA weights; the base mBART model is downloaded separately.
- `outputs/predictions/`, `outputs/metrics/`, `outputs/analysis/`, `outputs/figures/`, `outputs/postediting/`, `outputs/decoding/`: recorded results.

## Reproducing the analysis

The main training environment was Windows PowerShell with CUDA; `requirements-lock.txt` records its installed Python packages. COMET used a separate environment (`unbabel-comet` 2.2.7, PyTorch 2.8.0+cu128, Transformers 4.57.6), and Ollama 0.40.2 with model digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`. Exact environment and input hashes are saved in result JSON files.

The processed CSVs allow analysis without downloading raw Europarl. From the repository root, with the main environment active:

```powershell
python scripts/eda_europarl.py
jupyter nbconvert --to notebook --execute notebooks/01_europarl_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_experiment_analysis.ipynb --inplace
```

To run model inference using the included final adapters, the evaluation script checks the included `run_metadata.json` files, the adapter configuration, and the frozen dataset hashes before generating predictions:

```powershell
python scripts/evaluate_lora_validation.py --variant multidoc_balanced --split validation
python scripts/evaluate_lora_validation.py --variant multidoc_balanced --split test
```

For the optional sampling analysis, run the `generate`, `rerank`, then `score` stages of `scripts/sample_decoding_v2.py` for each method. It requires the context adapter, the frozen beam predictions, and a local Ollama server. `scripts/evaluate_decoding_comet_v2.py` scores the resulting validation predictions in the separate COMET environment.

## Scope and limitations

The raw OPUS Europarl archives, virtual environments, Hugging Face caches, and intermediate training checkpoints are not stored in Git. To rebuild the data from raw sources, download the archives with `scripts/download_europarl.py` and follow the data builder's arguments and manifest; the legacy pilot files are also required. The v2 training metadata files are included alongside the final adapter directories. A larger controlled v3 experiment is described in [`docs/scaled_v3_experiment.md`](docs/scaled_v3_experiment.md).

The test split has only 150 examples across 10 source documents. We report descriptive scores and examples, without a paired uncertainty interval or a dedicated human evaluation of cross-sentence phenomena. No improvement should be interpreted as a causal effect of document context without additional targeted evidence.
