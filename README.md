# Context-aware multilingual machine translation (CIS 600)

This project studies English-to-German and English-to-French translation on a document-separated Europarl subset. mBART-50 with LoRA supplies sentence (A) and previous-source context (B) drafts. GPT-6 Astra post-edits those drafts (C/D) and reranks the separate validation-only Top-k/Top-p candidate experiments.

## Scaled v3 study

The frozen data contain 12,000 train pairs (400 documents), 1,500 validation pairs (100 documents) and 1,500 held-out test pairs (100 documents). The en-de/en-fr counts are 2:1 in every split. This is a controlled subset of Europarl, not every Europarl sentence. A uses `src`; B uses `src </s> previous_src`. Four adapters compare balanced versus unweighted loss in each input mode, using the same three-epoch schedule and 128-token maximum.

| Balanced arm | Decoder input | Post-edit |
| --- | --- | --- |
| A | Current sentence | None |
| B | Current sentence, separator, previous sentence | None |
| C | A's frozen beam draft | GPT-6 Astra |
| D | B's frozen beam draft | GPT-6 Astra |

Held-out **test** macro scores (equal weight for en-de and en-fr):

| Arm | SacreBLEU | chrF | COMET |
| --- | ---: | ---: | ---: |
| A | 33.61 | 61.76 | 0.8700 |
| B | 33.51 | 61.69 | 0.8695 |
| C | 37.08 | 64.58 | 0.8879 |
| D | 37.17 | 64.64 | 0.8888 |

Paired whole-document intervals show substantial gains for GPT-6 post-editing over both mBART drafts, while context by itself does not show a stable improvement. See [full results](docs/scaled_v3_results.md) and [GPT-6 post-edit workflow](docs/gpt6_postedit.md).

## Validation-only sampling extension

Three candidates per row were generated with Top-k (k=50) or Top-p (p=0.9), temperature 0.8 and seed 42. GPT-6 selected among each set of candidates plus the frozen B beam translation. The reference was never included in the prompt. Macro COMET for beam / Top-k+GPT-6 / Top-p+GPT-6 is **0.8700 / 0.8726 / 0.8727**; macro BLEU is **33.95 / 33.72 / 33.64**, and chrF **61.61 / 61.57 / 61.54**. The experiment is validation-only; beam remains the main held-out test decoder. See [decoding workflow](docs/gpt6_decoding.md).

## Repository map

- `scripts/01_download_europarl.py` through `05_evaluate_lora.py`: data, EDA, LoRA and beam evaluations.
- `scripts/06_postedit_lora.py`, `07_evaluate_comet.py`: GPT-6 post-editing and A/B/C/D scoring.
- `scripts/08_sample_decoding.py`, `15_rerank_decoding_gpt6.py`, `09_evaluate_decoding_comet.py`: sample candidates, GPT-6 rerank, validation COMET.
- `scripts/10_analyze_decoding_effect.py`: rank changed Top-k/Top-p translations using saved COMET scores.
- `scripts/13_document_bootstrap.py`, `14_audit_v3_alignment.py`: paired uncertainty and corpus audit.
- `notebooks/01_europarl_eda.ipynb`, `02_experiment_analysis.ipynb`: v3 EDA and experiment analysis. Executed copies reside in `outputs/notebooks/`.
- `demo_v3.py`: local Streamlit A/B/C/D demo. Live GPT-6 calls are billed separately.
- `data/processed/europarl_scaled_v3/`: frozen data and manifest. Pilot/v2 processed data are retained only as exclusion records for reproducing the v3 document split.

Main experiments used Windows PowerShell and CUDA. COMET runs in `.venv-comet`. Raw archives, virtual environments, model caches, and intermediate checkpoints are not committed. This study uses one training seed and one post-edit configuration; automatic metrics describe the measured corpus and do not establish discourse-level correctness for individual translations.
