# Europarl scaled v3: completed results

## Scope and design

The controlled v3 corpus has 12,000 training, 1,500 validation and 1,500 held-out test translation pairs, with 400/100/100 document-separated source documents. Each split contains en-de and en-fr in a 2:1 example ratio. All macro scores average the two language pairs equally. The four mBART-50 LoRA training arms cross sentence/context input with balanced/unweighted loss. The primary A/B/C/D comparison uses balanced training and four-beam decoding. C and D are GPT-6 Astra post-edits of the same frozen A and B beam drafts respectively. References were not sent to GPT-6.

## Primary A/B/C/D results

| Split | Arm | Macro BLEU | Macro chrF | Macro COMET |
| --- | --- | ---: | ---: | ---: |
| Validation | A: sentence | 33.99 | 61.76 | 0.8699 |
| Validation | B: context | 33.95 | 61.61 | 0.8700 |
| Validation | C: A + GPT-6 | 36.97 | 64.24 | 0.8881 |
| Validation | D: B + GPT-6 | 36.92 | 64.21 | 0.8896 |
| Test | A: sentence | 33.61 | 61.76 | 0.8700 |
| Test | B: context | 33.51 | 61.69 | 0.8695 |
| Test | C: A + GPT-6 | 37.08 | 64.58 | 0.8879 |
| Test | D: B + GPT-6 | 37.17 | 64.64 | 0.8888 |

On held-out test, C-A is +3.48 BLEU, +2.83 chrF and +0.0180 COMET; D-B is +3.66 BLEU, +2.96 chrF and +0.0192 COMET. Whole-document paired bootstrap with 1,000 resamples gives D-B intervals of [+3.06, +4.26] BLEU, [+2.55, +3.37] chrF and [+0.0166, +0.0221] COMET. B-A intervals include zero; adding one preceding source sentence did not yield a stable independent gain. D-C is +0.0008 COMET on test, while its BLEU/chrF intervals include zero. These conditional intervals do not quantify retraining or human-judgment uncertainty. COMET and paired intervals: `outputs/metrics/comet_{validation,test}_scaled_v3_gpt6_metrics.json` and `outputs/metrics/document_bootstrap_{validation,test}_scaled_v3_gpt6.json`.

The unweighted sentence/context test arms scored 33.52/33.46 macro BLEU and 61.66/61.60 macro chrF; balanced counterparts scored 33.61/33.51 and 61.76/61.69. Small mixed differences do not show a consistent advantage for weighting. COMET was not run for the unweighted arms.

## Top-k and Top-p: GPT-6 candidate reranking

This branch is **validation only**. The frozen B adapter produced three sampled candidates per example at temperature 0.8 and seed 42, with Top-k k=50 or Top-p p=0.9. GPT-6 chose one of those or the frozen beam prediction, without reference access or rewriting. The candidates and beam were the same as in the earlier decoding run. This compares sampling **plus reranking** with beam, not raw sampling.

| Validation decoder | Macro BLEU | Macro chrF | Macro COMET | Changed vs beam |
| --- | ---: | ---: | ---: | ---: |
| Four-beam baseline | 33.95 | 61.61 | 0.8700 | 0 |
| Top-k + GPT-6 rerank | 33.72 | 61.57 | 0.8726 | 362/1,500 |
| Top-p + GPT-6 rerank | 33.64 | 61.54 | 0.8727 | 508/1,500 |

Top-k COMET is 0.8721 for en-de (N=1,000) and 0.8731 for en-fr (N=500); Top-p is 0.8721 and 0.8732. The beam counterparts are 0.8698 and 0.8702. Top-k gains +0.0026 macro COMET against beam but loses 0.23 BLEU and 0.04 chrF; Top-p gains +0.0027 COMET but loses 0.31 BLEU and 0.07 chrF. No document-bootstrap interval has been computed for these decoding differences, and no sampled-decoder test score is claimed. Beam remains the primary held-out test decoder. Source outputs in `outputs/decoding/scaled_v3/validation_top_k_k50_n3_t0p8_seed42_gpt6/`, `validation_top_p_p0p9_n3_t0p8_seed42_gpt6/` and `validation_comet_comparison_gpt6/`.

## Quality controls and open work

The v3 builder checks document isolation and structural alignment, and EDA found 177/15,000 context inputs (1.18%) above the 128-token cap. A 30-pair AI-assisted spot check flagged one likely English-German negation discrepancy in a held-out reference; this is a review flag, not a corpus-level error rate. The frozen test references were not edited. The AI-assisted alignment spot check is exploratory and does not establish a corpus-wide error rate or translation-quality judgment. Raw v2 versus v3 scores refer to different data and should not be treated as paired improvements.
