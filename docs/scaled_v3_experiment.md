# Scaled Europarl v3 experiment

**Status (October 2026):** The dataset, four balanced/unweighted models, validation/test A/B/C/D evaluation, and validation-only Top-k/Top-p ablations have been completed. The sections below preserve the original run plan and commands; see [the completed results](scaled_v3_results.md). Document-level uncertainty and independent human evaluation remain open.

This is the next, larger controlled experiment. It does not replace the published multidoc v2 results. The proposal permits controlled Europarl subsets; evaluating every sentence of the raw corpus is a different and much more expensive study. The existing v2 test was already evaluated in full (all 150 rows).

## Design and gates

1. Freeze a document-stratified v3 dataset before training. Exclude both the pilot and all v2 documents, so the new test set is separate from previous experiments. Keep the manifest and CSV hashes.
2. Inspect the new dataset, including a manual bilingual-alignment sample and truncation rates. Choose a source length once on validation, then keep that setting for all arms.
3. Train A (current sentence) and B (current sentence plus previous source) from the same base mBART model using the same balanced-loss schedule and 3 epochs. Keep the two output directories distinct.
4. Run A/B beam decoding on validation, then C/D local LLM post-editing and COMET on validation. Make any design decisions here and freeze them.
5. Run all four arms once on the v3 test and score BLEU, chrF and COMET by pair. Report both language pairs and document-aware uncertainty. Do not use test scores to select prompts or hyperparameters.

The target plan below selects 400/100/100 documents for train/validation/test. The per-document quotas are 20 en-de and 10 en-fr training pairs and 10 en-de and 5 en-fr pairs in each evaluation split: **12,000/1,500/1,500 translation examples** if enough eligible bilingual documents exist. The builder fails without writing a partial dataset if a quota cannot be met. This is 10 times the v2 example count, still a controlled subset of Europarl. Running time is hardware-dependent; do not assume the old 1,200-example time scales exactly.

## 1. Build and inspect data

First synchronize the updated scripts from the Git checkout to the original working directory (which contains the raw Europarl archives and Python environments):

```powershell
cd D:\CIS600\Project-Graduate-upload
git pull --ff-only origin main
Copy-Item .\scripts\02_build_europarl_multidoc.py, .\scripts\03_eda_europarl.py, .\scripts\05_evaluate_lora.py, .\scripts\06_postedit_lora.py, .\scripts\07_evaluate_comet.py D:\CIS600\context-mt\scripts\ -Force
cd D:\CIS600\context-mt
```

Run the remaining commands from `D:\CIS600\context-mt` in the project's normal Python environment. The ignored raw Europarl archives in `data/raw/europarl/` and the existing pilot/v2 processed CSVs are required.

```powershell
python scripts\02_build_europarl_multidoc.py `
  --output-dir data/processed/europarl_scaled_v3 `
  --exclude-dir data/processed/europarl_multidoc_v2 `
  --train-documents 400 `
  --validation-documents 100 `
  --test-documents 100 `
  --candidate-multiplier 4

python scripts\04_train_lora.py `
  --data-dir data/processed/europarl_scaled_v3 `
  --input-mode sentence --loss-mode balanced `
  --check-data-only

python scripts\03_eda_europarl.py --dataset scaled_v3
```

Before training, check the printed counts, document isolation, `data/processed/europarl_scaled_v3/summary.json`, source lengths, and a manually inspected sample of en-de/en-fr source–target alignments from each split. If the builder cannot find 400/100/100 eligible documents, preserve the error and rerun with a **new output directory** and a documented smaller plan; never edit the frozen v2 CSVs.

## 2. Train A and B

The commands below use the same 128-token maximum as v2. Review the new EDA truncation report before starting. If a larger maximum is needed, use it for both arms and update the evaluator consistently before running validation.

```powershell
python scripts\04_train_lora.py `
  --data-dir data/processed/europarl_scaled_v3 `
  --input-mode sentence --loss-mode balanced --max-length 128 `
  --epochs 3 --output-dir outputs/checkpoints/sentence_lora_balanced_scaled_v3

python scripts\04_train_lora.py `
  --data-dir data/processed/europarl_scaled_v3 `
  --input-mode context_v1 --loss-mode balanced --max-length 128 `
  --epochs 3 --output-dir outputs/checkpoints/context_lora_balanced_scaled_v3
```

Training creates `final_adapter/` and `run_metadata.json`. The evaluator checks both metadata files and the dataset hashes. Use `--resume` only when the selected output directory contains a complete interrupted checkpoint.

## 3. Validation first

```powershell
python scripts\05_evaluate_lora.py --variant scaled_balanced --split validation

python scripts\06_postedit_lora.py `
  --dataset scaled_v3 --split validation --variant both --model qwen2.5:7b

.\.venv-comet\Scripts\python.exe scripts\07_evaluate_comet.py `
  --dataset scaled_v3 --split validation --device cuda --batch-size 4
```

Review the A/B/C/D results and prompt behavior, then freeze the settings. The existing v2 Top-k/Top-p analysis remains a validation-only side experiment; it is not necessary to rerun sampling over all v3 test sentences.

## 4. One final held-out test

```powershell
python scripts\05_evaluate_lora.py --variant scaled_balanced --split test

python scripts\06_postedit_lora.py `
  --dataset scaled_v3 --split test --variant both --model qwen2.5:7b

.\.venv-comet\Scripts\python.exe scripts\07_evaluate_comet.py `
  --dataset scaled_v3 --split test --device cuda --batch-size 4
```

Save the v3 manifest, run metadata, predictions and metrics. Do not compare raw v2 and v3 BLEU as if their references and document samples were identical. For the report, give paired A–B, A–C, B–D, C–D comparisons on the same v3 examples, preferably bootstrapping **documents** rather than independent sentences.

## Further ablations from the reference course project

- Train one or both arms with `--loss-mode unweighted` to measure the effect of balanced loss on the smaller en-fr pair. Use fresh checkpoint directories and add explicit evaluation variants; do not call balanced loss beneficial merely because the PDF introduces it.
- The PDF shows direct top-k, top-p and joint top-k+top-p sampling. The current project instead uses three sampled candidates plus a frozen beam candidate with blind LLM reranking. A direct-sampling and joint-sampling ablation could be added on validation, with the same seed and decoding budget. Judge accuracy, completeness and consistency, not only diversity.
- A stronger context-specific check labels pronouns, entity references and repeated terminology across adjacent sentences and compares paired A/B or C/D translations by document. A small blinded human review is more diagnostic than a tiny corpus-level BLEU difference.

The reference PDF is a teaching project based on OPUS-100. Its example balanced-loss formula divides by the sum of weights within each batch; with microbatch size one that would cancel the pair weight. This repository instead uses globally normalized fixed pair weights and gradient accumulation so pair weighting affects optimization.
