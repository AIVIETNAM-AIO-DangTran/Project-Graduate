# Context-aware multilingual machine translation (CIS 600)

This project studies English → German/French translation on document-separated Europarl subsets. It compares sentence and previous-source context inputs, balanced and unweighted LoRA training, and optional local Qwen2.5:7b post-editing. mBART-50 is the base translator. The completed larger **scaled v3** experiment is the main study; the smaller multidoc v2 and pilot runs remain historical experiments.

## Scaled v3 study

The v3 splits contain 12,000 train pairs from 400 documents, 1,500 validation pairs from 100 documents, and 1,500 test pairs from 100 documents. Every split is en-de/en-fr in a 2:1 ratio. This is a controlled subset, not all of Europarl. The sentence and context arms use current source `src` and `src </s> previous_src` respectively. Balanced/unweighted LoRA ablations share three epochs and 128-token maximum. Beam search (four beams) was selected on validation. The reference translation is never supplied to the post-editor.

| Arm | mBART input | LLM post-edit |
| --- | --- | --- |
| A | Current English sentence | No |
| B | Current plus preceding English source | No |
| C | Same as A | Qwen2.5:7b |
| D | Same as B | Qwen2.5:7b |

Held-out **test** macro scores below average the two language pairs equally, despite their unequal row counts. See [full v3 results](docs/scaled_v3_results.md) for pair scores, validation results, ablations, and limitations.

| Model | SacreBLEU | chrF | COMET |
| --- | ---: | ---: | ---: |
| A, balanced | 33.61 | 61.76 | 0.8700 |
| B, balanced | 33.51 | 61.69 | 0.8695 |
| Sentence, unweighted | 33.52 | 61.66 | — |
| Context, unweighted | 33.46 | 61.60 | — |
| C, A + Qwen | 32.89 | 61.37 | 0.8704 |
| D, B + Qwen | 33.25 | 61.52 | 0.8722 |

The best BLEU/chrF is A, while D has the highest COMET. Differences are small and mixed; no document-level uncertainty interval or independent human study establishes a stable quality gain. The v3 Top-k/Top-p experiment was assessed on validation only: sampled candidates plus Qwen reranking reduced BLEU relative to beam, so beam remained the test decoder.

## Files and use

- `scripts/`: build, EDA, train, evaluate, Qwen post-edit, COMET and sampling analysis. Analysis entry points default to scaled v3; pass `--dataset` or `--data-dir` explicitly to examine a different dataset.
- `scripts/build_manual_review.py`: creates a blinded validation sheet and separate answer key with `--kind abcd` or `--kind decoding`. Existing saved review files do not need regeneration.
- `notebooks/01_europarl_eda.ipynb`: v3 EDA. `notebooks/02_experiment_analysis.ipynb`: v3 validation/test results, selected by `EXPERIMENT_SPLIT`. Executed copies are in `outputs/notebooks/` (`01_scaled_v3_eda.ipynb`, `02_scaled_v3_validation.ipynb`, `02_scaled_v3_test.ipynb`). Earlier v2 results remain in Git history.
- `docs/scaled_v3_experiment.md`: design and data-building commands.
- `docs/scaled_v3_results.md`: full v3 results and interpretation.
- `demo_v3.py` and `docs/demo_v3.md`: local A/B/C/D demo and installation commands.
- `data/processed/europarl_scaled_v3/` and `outputs/`: working-project datasets, adapters, predictions and reports; check their availability in the local Windows project before reproduction. The complete frozen v3 artifacts are included in this repository.
- `data/processed/europarl_pilot/` and `data/processed/europarl_multidoc_v2/`: small exclusion records required to reproduce the frozen v3 document selection. Model training and evaluation use `europarl_scaled_v3/`; earlier v2 results remain in Git history.

The main run used Windows PowerShell and CUDA; Qwen ran through Ollama, and COMET in a separate `.venv-comet` environment. See the metadata/metric JSON files for hashes and settings, and `requirements-lock.txt` for the earlier main environment snapshot. To run the v3 notebooks:

```powershell
$env:EXPERIMENT_SPLIT = "validation"
jupyter nbconvert --to notebook --execute notebooks\02_experiment_analysis.ipynb --output-dir outputs\notebooks --output 02_scaled_v3_validation.ipynb
$env:EXPERIMENT_SPLIT = "test"
jupyter nbconvert --to notebook --execute notebooks\02_experiment_analysis.ipynb --output-dir outputs\notebooks --output 02_scaled_v3_test.ipynb
```

Raw archives, virtual environments, Hugging Face caches and intermediate training checkpoints are not committed. The v3 findings are descriptive for this controlled sample and one training seed.
