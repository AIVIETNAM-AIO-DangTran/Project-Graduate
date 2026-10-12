# AI-assisted blinded review: scaled v3 validation decoding

This is an **AI assessment**, not an independent human evaluation. The 48 rows were judged without the answer key; the key was opened only for aggregation. Comparisons are beam versus a Qwen-reranked sampled output from the same context-LoRA model. Rows were selected among translations that differ from beam. Repeated source sentences across top-k and top-p are counted as separate method comparisons.

## Selection

For each method and language pair, 8 changed rows were drawn at random and 4 diagnostic extremes were added (2 highest and 2 lowest paired COMET deltas). The 32 random rows provide a small exploratory check **conditional on the translation changing**. The 16 COMET extremes are intentionally nonrepresentative and should be used only as error examples. No test sentence was reviewed.

## Meaning judgments on random changed rows

| Method | Pair | Sampled better | Beam better | Tie | N |
|---|---|---:|---:|---:|---:|
| Top-k | en-de | 4 | 3 | 1 | 8 |
| Top-k | en-fr | 2 | 4 | 2 | 8 |
| Top-k | Both | 6 | 7 | 3 | 16 |
| Top-p | en-de | 5 | 3 | 0 | 8 |
| Top-p | en-fr | 2 | 5 | 1 | 8 |
| Top-p | Both | 7 | 8 | 1 | 16 |

Fluency on those same random changed rows: Top-k sampled/beam/tie = 6/8/2; Top-p = 6/9/1. Both methods have 3 sampled and 3 beam outputs marked with a critical error in their respective random subsamples. These counts are far too small for claims about population-level win rates.

## All 48 rows, including deliberately selected COMET extremes

| Method | Sampled better meaning | Beam better meaning | Tie | N |
|---|---:|---:|---:|---:|
| Top-k | 11 | 10 | 3 | 24 |
| Top-p | 11 | 12 | 1 | 24 |

Do not turn this second table into an estimated win rate. The extremes were selected by COMET score and some underlying sentences appear under both methods.

## Qualitative patterns

- Both sampling methods sometimes repair a beam semantic error: for “It misleads the media and the outside world,” the sampled text contains *irreführend*, while beam uses *misst* (“measures”). The sampled German sentence still has grammatical errors.
- Sampling can create severe errors. For “I think that is mad!”, a sampled French version says “Je pense que c'est de l'aile !”, which fails to convey the intended meaning.
- Both candidates can be wrong. For “the Court is unable to give an unqualified positive opinion,” one German candidate incorrectly drops the negation while the other keeps it in an incoherent construction. In another example both miss “last but one paragraph.”
- There are recurring entity and terminology issues (*clergynes*, *EMPA*, and *Cassa del Mezzogiorno*) and referent errors (*these*, *it*). The preceding sentence sometimes clarifies the referent, but this review does not isolate a causal effect of source context: both alternatives used the same context-aware model.

## Decision and limitations

Keep **beam search as the default decoding configuration** for the final v3 test and demo. The full validation evaluation found a small Top-k COMET gain (+0.0006) alongside lower macro BLEU/chrF, and Top-p did not improve macro COMET, BLEU, or chrF. The exploratory AI judgments do not supply strong counterevidence. Report both sampling methods as validation ablations and include concrete examples of gains and failures.

Do not describe these labels as human annotation. For a defensible human study, recruit a German/French-proficient reviewer, retain the blind candidate order, and ideally ask a second reviewer on an overlap set. The `ai_review.csv` file contains sentence-level judgments and notes; `answer_key.csv` maps candidates to methods after judging.
