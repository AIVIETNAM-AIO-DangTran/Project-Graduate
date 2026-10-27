# Context-aware multilingual machine translation (CIS 600)

This project studies English → German/French translation on document-separated Europarl subsets. It compares sentence and previous-source context inputs, balanced and unweighted LoRA training, and LLM post-editing. mBART-50 is the base translator. The completed larger **scaled v3** experiment first used local Qwen2.5:7b; GPT-6 Astra is now the primary LLM post-editor. Qwen remains a historical comparator, and their outputs are saved separately. The smaller multidoc v2 and pilot runs remain historical experiments.

## Scaled v3 study

The v3 splits contain 12,000 train pairs from 400 documents, 1,500 validation pairs from 100 documents, and 1,500 test pairs from 100 documents. Every split is en-de/en-fr in a 2:1 ratio. This is a controlled subset, not all of Europarl. The sentence and context arms use current source `src` and `src </s> previous_src` respectively. Balanced/unweighted LoRA ablations share three epochs and 128-token maximum. Beam search (four beams) was selected on validation. The reference translation is never supplied to the post-editor.

| Arm | mBART input | LLM post-edit |
| --- | --- | --- |
| A | Current English sentence | No |
| B | Current plus preceding English source | No |
| C | Same as A | GPT-6 Astra (primary); Qwen historical |
| D | Same as B | GPT-6 Astra (primary); Qwen historical |

The first table below preserves the historical Qwen run; the GPT-6 C/D table below it is the primary post-editing result. See [GPT-6 post-editing](docs/gpt6_postedit.md) for the reproducible workflow.

Held-out **test** macro scores average the two language pairs equally, despite their unequal row counts. See [full v3 results](docs/scaled_v3_results.md) for pair scores, validation results, ablations, and limitations.

| Model | SacreBLEU | chrF | COMET |
| --- | ---: | ---: | ---: |
| A, balanced | 33.61 | 61.76 | 0.8700 |
| B, balanced | 33.51 | 61.69 | 0.8695 |
| Sentence, unweighted | 33.52 | 61.66 | — |
| Context, unweighted | 33.46 | 61.60 | — |
| C, A + Qwen | 32.89 | 61.37 | 0.8704 |
| D, B + Qwen | 33.25 | 61.52 | 0.8722 |

Within the historical Qwen run, A has the best BLEU/chrF and D the highest COMET. Its differences are small and mixed; paired document-level intervals are reported, but no independent human study establishes a stable quality gain across splits and metrics. The v3 Top-k/Top-p experiment was assessed on validation only: sampled candidates plus Qwen reranking reduced BLEU relative to beam, so beam remained the test decoder.

### Completed GPT-6 Astra post-editing follow-up

The same frozen A/B beam outputs were post-edited with GPT-6 Astra. These C/D outputs are separate from the Qwen results above. Held-out **test** macro scores:

| Model | SacreBLEU | chrF | COMET |
| --- | ---: | ---: | ---: |
| C, A + GPT-6 | 37.08 | 64.58 | 0.8879 |
| D, B + GPT-6 | 37.17 | 64.64 | 0.8888 |

GPT-6 improved all three metrics over A/B on both validation and test. For D−B, the test document-bootstrap 95% intervals are [+3.06, +4.26] BLEU, [+2.55, +3.37] chrF and [+0.0166, +0.0221] COMET. D−C is only +0.00083 macro COMET on test, despite a positive conditional interval, while the underlying B−A context effect remains uncertain. Paired same-row comparison with the historical Qwen run gives test GPT-6 minus Qwen macro COMET +0.01752 [+0.01510, +0.02035] for C and +0.01656 [+0.01436, +0.01940] for D under document resampling. A bilingual human review is pending. See [full results](docs/scaled_v3_results.md) and [GPT-6 workflow](docs/gpt6_postedit.md); no further paid calls are required for these completed results.

## Files and use

The scripts are numbered by workflow stage. Steps 08–10 form an optional validation-only decoding branch; step 11 supports either the A/B/C/D or decoding review. The numbers are for navigation, not Python imports. Analysis commands default to scaled v3 unless `--dataset` or `--data-dir` is set.

| Stage | Script | Purpose |
| --- | --- | --- |
| 01 | `scripts/01_download_europarl.py` | Download the Europarl archive |
| 02 | `scripts/02_build_europarl_multidoc.py` | Build document-separated splits |
| 03 | `scripts/03_eda_europarl.py` | Explore data and token lengths |
| 04 | `scripts/04_train_lora.py` | Train balanced or unweighted LoRA |
| 05 | `scripts/05_evaluate_lora.py` | Generate BLEU/chrF predictions |
| 06 | `scripts/06_postedit_lora.py` | Post-edit with GPT-6 API or legacy local Ollama |
| 07 | `scripts/07_evaluate_comet.py` | Score A/B/C/D with COMET, keeping GPT-6 separate |
| 08 | `scripts/08_sample_decoding.py` | Generate and rerank Top-k/Top-p candidates |
| 09 | `scripts/09_evaluate_decoding_comet.py` | Score sampled decoding with COMET |
| 10 | `scripts/10_analyze_decoding_effect.py` | Compare decoding errors |
| 11 | `scripts/11_build_manual_review.py` | Build a blinded review sheet |
| 12 | `scripts/12_summarize_abcd_review.py` | Summarize A/B/C/D judgments |
| 13 | `scripts/13_document_bootstrap.py` | Paired document-level confidence intervals |
| 14 | `scripts/14_audit_v3_alignment.py` | Recheck v3 CSVs and draw alignment-review samples |

- `notebooks/01_europarl_eda.ipynb`: v3 EDA. `notebooks/02_experiment_analysis.ipynb`: v3 validation/test results, selected by `EXPERIMENT_SPLIT`, with GPT-6 post-editing by default. Set `POSTEDIT_MODEL=qwen` to inspect the historical Qwen run. Executed copies are in `outputs/notebooks/` (`01_scaled_v3_eda.ipynb`, `02_scaled_v3_validation.ipynb`, `02_scaled_v3_test.ipynb`). Earlier v2 results remain in Git history.
- `docs/scaled_v3_experiment.md`: design and data-building commands.
- `docs/scaled_v3_results.md`: full v3 results and interpretation.
- `docs/human_review_protocol.md`: blinded bilingual review rubric; the current AI review is exploratory.
- `demo_v3.py` and `docs/demo_v3.md`: local A/B/C/D demo and installation commands.
- `data/processed/europarl_scaled_v3/` and `outputs/`: working-project datasets, adapters, predictions and reports; check their availability in the local Windows project before reproduction. The complete frozen v3 artifacts are included in this repository.
- `data/processed/europarl_pilot/` and `data/processed/europarl_multidoc_v2/`: small exclusion records required to reproduce the frozen v3 document selection. Model training and evaluation use `europarl_scaled_v3/`; earlier v2 results remain in Git history.

The executed notebooks under `outputs/notebooks/` can be opened or rerun from the project folder or a nested notebook directory. Their validation/test copies are fixed to their named split; use `notebooks/02_experiment_analysis.ipynb` with `EXPERIMENT_SPLIT` to regenerate a different split.

The main run used Windows PowerShell and CUDA; Qwen ran through Ollama, and COMET in a separate `.venv-comet` environment. See the metadata/metric JSON files for hashes and settings, and `requirements-lock.txt` for the earlier main environment snapshot. To run the v3 notebooks:

```powershell
$env:EXPERIMENT_SPLIT = "validation"
jupyter nbconvert --to notebook --execute notebooks\02_experiment_analysis.ipynb --output-dir outputs\notebooks --output 02_scaled_v3_validation.ipynb
$env:EXPERIMENT_SPLIT = "test"
jupyter nbconvert --to notebook --execute notebooks\02_experiment_analysis.ipynb --output-dir outputs\notebooks --output 02_scaled_v3_test.ipynb
```

Raw archives, virtual environments, Hugging Face caches and intermediate training checkpoints are not committed. The v3 findings are descriptive for this controlled sample and one training seed.

Paired bootstrap results for both v3 splits are in `outputs/metrics/document_bootstrap_{validation,test}_scaled_v3.json`; the 30-row source/reference spot check and flagged examples are in `outputs/analysis/alignment_audit_scaled_v3/`. The test D–B COMET interval is positive but validation and BLEU/chrF remain mixed, so no stable across-metric improvement is claimed. The bilingual human review is still pending.
