# Europarl scaled v3: results and interpretation

## Experimental setup

This is a controlled Europarl subset, not a run on every Europarl sentence. The frozen splits contain 12,000 train, 1,500 validation and 1,500 test translation pairs, drawn from 400, 100 and 100 documents respectively. Each split has two English-source language pairs in a 2:1 ratio: en-de (8,000/1,000/1,000) and en-fr (4,000/500/500). The builder checks document isolation and alignment structure; spot checking semantic equivalence remains a separate human task. Excluded pilot/v2 documents keep this experiment distinct from the earlier small study.

All four mBART-50 LoRA models used three epochs and the same 128-token maximum. The sentence models translate `src`; the context models receive `src </s> previous_src`, truncating from the right. The EDA found 177 of 15,000 context inputs (1.18%) exceeding 128 tokens; none exceeded 256. The two balanced models use fixed pair weights; the two unweighted models are the loss ablation. Evaluation uses four-beam decoding. C and D post-edit the respective A and B predictions with local Qwen2.5:7b at temperature zero, without sending references to the LLM.

The tables below are transcribed from the completed local v3 runs. Macro scores give en-de and en-fr **equal weight**, despite the 2:1 example ratio. BLEU and chrF are corpus scores within each pair, then averaged across pairs; COMET scores are averaged within each pair, then across pairs. Values are rounded, so subtracting the displayed entries can differ slightly from the underlying results.

## Validation

| Model | en-de BLEU | en-fr BLEU | Macro BLEU | Macro chrF | Macro COMET |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Sentence, balanced | 29.17 | 38.82 | 33.99 | 61.76 | 0.8699 |
| B · Context, balanced | 29.11 | 38.78 | 33.95 | 61.61 | 0.8700 |
| Sentence, unweighted | — | — | 33.97 | 61.81 | — |
| Context, unweighted | — | — | 34.11 | 61.74 | — |
| C · A + Qwen | 28.20 | 37.93 | 33.06 | 61.35 | 0.8687 |
| D · B + Qwen | 28.69 | 38.72 | 33.70 | 61.66 | 0.8705 |

Qwen changed 1,230/1,500 C predictions and 844/1,500 D predictions. On the frozen validation split it lowered macro BLEU in both arms. The COMET differences were small: B−A ≈ +0.0001, C−A ≈ −0.0011, D−B ≈ +0.0005.

### Top-k and Top-p decoding (validation only)

The context adapter B was decoded with either four-beam search or three sampled candidates plus the frozen beam candidate. Qwen selected a candidate without seeing a reference. Top-k used k=50; Top-p used p=0.9; both used temperature 0.8, seed 42 and three sampled candidates per source. This compares **sampling plus reranking** against beam, rather than raw sampling alone.

| Decoding | Macro BLEU | Macro chrF | Macro COMET | Outputs changed |
| --- | ---: | ---: | ---: | ---: |
| Beam | 33.95 | 61.61 | 0.8700 | 0 |
| Top-k + rerank | 33.46 | 61.33 | 0.8706 | 290/1,500 |
| Top-p + rerank | 33.36 | 61.18 | 0.8698 | 421/1,500 |

For the changed cases, COMET favored top-k 138 times and beam 152 times; for top-p it favored sampling 202 times and beam 219 times. We kept **beam as the primary decoder** after validation. There is no v3 test-set sampling result.

An exploratory blind AI-assisted review of 16 random changed translations per comparison found mixed meaning judgments: context B versus A 7/8/1 better/worse/tied, C versus A 9/5/2, D versus B 7/5/4. The sample is small and AI judged; it is not an independent human assessment or an estimate of population error rates. Other selected examples were enriched by context cues and COMET extremes and should not be reported as random samples.

## Held-out test

| Model | en-de BLEU | en-de chrF | en-de COMET | en-fr BLEU | en-fr chrF | en-fr COMET | Macro BLEU | Macro chrF | Macro COMET |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Sentence, balanced | 29.19 | 59.01 | 0.8674 | 38.02 | 64.51 | 0.8725 | 33.61 | 61.76 | 0.8700 |
| B · Context, balanced | 28.70 | 58.77 | 0.8663 | 38.33 | 64.61 | 0.8727 | 33.51 | 61.69 | 0.8695 |
| Sentence, unweighted | 28.99 | 58.90 | — | 38.04 | 64.42 | — | 33.52 | 61.66 | — |
| Context, unweighted | 28.75 | 58.75 | — | 38.17 | 64.45 | — | 33.46 | 61.60 | — |
| C · A + Qwen | 28.69 | 58.62 | 0.8666 | 37.09 | 64.12 | 0.8743 | 32.89 | 61.37 | 0.8704 |
| D · B + Qwen | 28.56 | 58.60 | 0.8681 | 37.94 | 64.44 | 0.8763 | 33.25 | 61.52 | 0.8722 |

Compared on the same held-out examples, B−A is −0.10 macro BLEU, −0.07 chrF and about −0.0005 COMET. C−A is −0.72 BLEU, −0.39 chrF and about +0.0004 COMET. D−B is −0.26 BLEU, −0.17 chrF and about +0.0027 COMET. Qwen changed 1,230 C outputs and 806 D outputs. The highest macro COMET is D, while the highest macro BLEU and chrF are A. Balanced training beats the corresponding unweighted model by about 0.09 macro BLEU and 0.10 chrF for sentence input, and about 0.05 BLEU and 0.09 chrF for context input. The pair-level pattern is mixed; these differences do not establish a consistent quality gain.

## Paired document-level uncertainty

We drew 2,000 bootstrap resamples of **whole document IDs** from each split (100 documents, seed 20261013). Each draw keeps the en-de/en-fr sentences from the same sampled documents together and compares matched A, B and D predictions. Intervals are percentile 95% intervals for the macro difference; BLEU and chrF are recomputed as corpus scores within each language pair. Full-precision outputs, hashes and SacreBLEU signatures are in `outputs/metrics/document_bootstrap_{validation,test}_scaled_v3.json`. These intervals condition on the single frozen training seed and Qwen run; they do not quantify retraining or annotator uncertainty.

| Split and contrast | Macro BLEU delta [95% interval] | Macro chrF delta [95% interval] | Macro COMET delta [95% interval] |
| --- | ---: | ---: | ---: |
| Validation B − A | −0.047 [−0.368, +0.334] | −0.148 [−0.467, +0.116] | +0.0001 [−0.0013, +0.0016] |
| Validation D − B | −0.244 [−0.953, +0.070] | +0.052 [−0.279, +0.351] | +0.0005 [−0.0037, +0.0043] |
| Test B − A | −0.094 [−0.418, +0.211] | −0.071 [−0.267, +0.114] | −0.0005 [−0.0016, +0.0006] |
| Test D − B | −0.259 [−0.579, +0.027] | −0.169 [−0.388, +0.026] | +0.0027 [+0.0008, +0.0046] |

The test D−B COMET interval excludes zero under this document-resampling method, but its validation interval includes zero and the BLEU/chrF intervals include zero. This does not establish a stable gain across metrics, splits or training seeds. The intervals are descriptive and were not adjusted for the number of comparisons.

## Alignment and blinded review

`scripts/14_audit_v3_alignment.py` checked manifest hashes, uniqueness and document/current-source isolation for all 15,000 rows, then sampled 30 pairs (five distinct documents per language pair per split). The sample is in `outputs/analysis/alignment_audit_scaled_v3/`. An AI-assisted preliminary reading marked 27 pairs apparently aligned, two uncertain and one likely English–German negation discrepancy in a test reference (ALIGN-022). This is a warning about reference fidelity, **not** a measured dataset-wide error rate. Raw XML adjacency and bilingual human verification are still required. The original frozen test CSV was not edited after inspection.

The 60-row blinded validation review sheet remains unfilled by an independent human. `docs/human_review_protocol.md` defines the rubric and keeps the current AI judgments separate; `outputs/analysis/abcd_validation_scaled_v3/manual_review/ai_review_diagnostic.md` summarizes the existing AI labels by context, pronoun and terminology categories. Those selected cases are exploratory.

## Interpretation and remaining checks

- The v3 experiment completes the A/B/C/D comparison and the balanced-loss ablation on a larger, document-separated controlled subset. It does not show a stable advantage from simply appending the previous source sentence. The post-editor trades lower BLEU/chrF for a small COMET gain in D.
- Document-level paired percentile intervals are now reported above, but retraining uncertainty, multiple comparisons and reference-quality uncertainty remain. Avoid broad causal or generalization claims.
- A blind **human** evaluation with the defined rubric for pronouns, entities, terminology, omitted content and prior-sentence contamination would strengthen the context claim. Existing AI-assisted reviews are exploratory.
- V2 and v3 use different held-out references and documents. Their raw BLEU scores must not be read as a matched v2→v3 improvement. The final demo uses beam and balanced v3 adapters; displaying a translation is not a test-set metric.

Primary local artifacts: `data/processed/europarl_scaled_v3/summary.json`, the four `outputs/checkpoints/*_scaled_v3/run_metadata.json` files, `outputs/metrics/*_scaled_v3_*_metrics.json`, `outputs/metrics/comet_{validation,test}_scaled_v3_metrics.json`, `outputs/decoding/scaled_v3/`, `outputs/analysis/decoding_validation_scaled_v3/`, and `notebooks/02_experiment_analysis.ipynb`. Verify each file in the working project before submission; older GitHub snapshots can lack the large v3 artifacts.
