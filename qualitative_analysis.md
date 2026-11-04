# Europarl multidoc v2: test results and targeted error analysis

## Scope and provenance

This note summarizes the frozen A/B/C/D experiment on the Europarl multidoc v2 test split: 150 aligned examples, 100 English–German and 50 English–French, drawn from 10 source documents (10 German and 5 French examples per document). The four arms are A, sentence-level mBART50 LoRA; B, preceding-source context mBART50 LoRA; C, A followed by Qwen2.5:7b post-editing; and D, B followed by the same post-editor. A/B use balanced pair loss. The COMET input file `all_examples(1).csv` supplies the same source, reference, predictions and sentence scores for all arms. The aggregate BLEU/chrF and COMET values below come from the saved test command output provided with the file.

The CSV has 150 unique (pair, document, source sentence, target sentence) keys, no empty required cells and only test rows. All four stored paired COMET deltas agree with the underlying scores. These checks establish table consistency; they do not verify the underlying Europarl alignments or certify individual references.

## Aggregate test results

| Arm | Macro SacreBLEU | Macro chrF | Macro COMET |
| --- | ---: | ---: | ---: |
| A: sentence | 31.05 | 58.37 | 0.8521 |
| B: context | **31.30** | 58.55 | 0.8528 |
| C: sentence + LLM | 30.30 | 58.01 | 0.8539 |
| D: context + LLM | 31.24 | **58.60** | **0.8582** |

Each macro value is the unweighted mean of the English–German and English–French scores; this gives the 50-example French pair the same macro weight as the 100-example German pair. BLEU and chrF are corpus metrics computed by pair, whereas COMET is averaged over sentence scores by pair. The COMET differences below use unrounded scores, so they may differ slightly from subtracting the four-decimal display values.

| Paired comparison | en-de mean COMET difference | en-fr mean COMET difference | Macro COMET difference | Changed predictions | Positive / negative / tied sentence differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| B − A: contextual input | +0.00174 | −0.00035 | +0.0007 | 76/150 | 41 / 35 / 74 |
| C − A: LLM on sentence model | +0.00528 | −0.00166 | +0.0018 | 117/150 | 59 / 58 / 33 |
| D − B: LLM on context model | +0.01077 | +0.00008 | +0.0054 | 76/150 | 49 / 27 / 74 |
| D − C: context in LLM pipeline | +0.00722 | +0.00139 | +0.0043 | 112/150 | 70 / 42 / 38 |

By document, B − A has a positive mean COMET difference in 5 of 10 documents; D − B is positive in 8 of 10. A descriptive document-cluster bootstrap (10,000 resamples of the 10 documents with replacement, seed 42, pair-balanced macro) gave percentile intervals of approximately [−0.0028, +0.0044] for B − A and [+0.0019, +0.0096] for D − B. These exploratory intervals depend on only 10 documents, the particular selected corpus and COMET's assessment; they are not a general population guarantee. No model, prompt or decoding setting was selected using these test results.

## RQ1: effect of preceding-source context

Context gives a small, mixed aggregate effect: B − A is +0.25 macro BLEU, +0.18 macro chrF and +0.0007 macro COMET. Its COMET gain is concentrated in English–German (+0.00174), while English–French is approximately unchanged (−0.00035). A and B produce identical predictions for 74/150 cases. These data do not support a broad claim that the preceding sentence reliably resolves discourse phenomena.

Targeted examples from the complete test table:

| Pair / document / source ID | A to B observation | Interpretation |
| --- | --- | --- |
| en-de / ep-07-06-18-017.xml / 9 | A mistranslates “keep up with the evolving world”; B produces the idiom “Schritt halten.” COMET B − A: +0.257. | Clear sentence-quality improvement, although the source itself contains the information needed; this is not direct proof of context use. |
| en-de / ep-06-11-13-014.xml / 27 | In response to the previous invitation to a press conference, A translates “you are welcome to come”; B outputs only “Herr Präsident!” COMET B − A: −0.134. | Strong omission and a missed opportunity to carry the event from the previous sentence into the current translation. |
| en-de / ep-09-05-05-010.xml / 455 | A attempts the long statement about the “Silva Peneda legacy”; B stops after thanking him. COMET B − A: −0.156. | Context can coincide with a serious loss of content; the shorter translation should not be called better simply because it reads fluently. |
| en-fr / ep-07-06-18-017.xml / 109 | For “question of recovery,” A uses “recouvrement” like the reference; B uses “relèvement.” COMET B − A: −0.146. | Lexical ambiguity in the Europarl domain. Without a human audit of wider context, the reference establishes a mismatch but not an unequivocal semantic error by B. |

A simple English pronoun/demonstrative regex matches 114 of 150 current source sentences, making that subset broad and overlapping. Its membership alone does not establish that a translation requires previous-sentence context. Claims about pronoun resolution or term consistency require targeted human labels and source/reference inspection.

## RQ2: LLM post-editing of contextual translation

D − B increases macro COMET by +0.0054, primarily through English–German (+0.01077); English–French changes by about +0.00008. However, macro BLEU decreases from 31.30 to 31.24, and chrF increases only from 58.55 to 58.60. For English–German, D increases BLEU from 25.44 to 26.57 and chrF from 55.44 to 56.36; English–French falls from 37.15 to 35.91 BLEU and from 61.67 to 60.84 chrF. The post-editor changes 76/150 contextual predictions.

Useful human-readable examples:

| Pair / document / source ID | B to D observation | Assessment |
| --- | --- | --- |
| en-de / ep-09-05-05-010.xml / 455 | D restores the clause about Silva Peneda's contribution and legacy that B omitted. COMET D − B: +0.174. | Clear gain in coverage, although D still has awkward German (“Haushaltskammer,” “Silva Peneda Erbe”). This repair does not by itself demonstrate use of the previous sentence. |
| en-de / ep-11-03-23-004.xml / 41 | D corrects the proposition “zero tolerance applies to all Members” to “Nulltoleranz gilt für alle Mitglieder dieses Hauses.” COMET D − B: +0.084. | Clear local meaning and formulation improvement; no cross-sentence inference is required. |
| en-fr / ep-09-05-05-010.xml / 641 | For “a stitch in time saves nine,” D contains Chinese characters “九针” in the French output. COMET D − B: +0.062. | A COMET increase accompanies an invalid mixed-script translation. Do not count this as a verified improvement. |
| en-de / ep-09-01-13-004.xml / 100 | D changes one compound in an otherwise near-identical sentence; COMET D − B: −0.104. Neither output matches the reference's “Kompromissänderungsanträge.” | Large sentence-score movement from a small lexical change; do not call this a large semantic regression without German review. |
| en-fr / ep-11-03-23-004.xml / 8 | D changes “député” to “membre” for a Member of the European Parliament. COMET D − B: −0.022. | Loss of institution-specific role precision; the reference uses “député.” |
| en-fr / ep-09-01-13-004.xml / 198 | D changes “document gratuit” to “document libre” for “free document.” COMET D − B: −0.021. | A plausible substantive regression: the context and reference indicate zero cost rather than freedom. |

The German reference for en-de / ep-97-04-10.xml / 398 stops at “und die Nrn.” while both B and D translate the source's full amendment-number list. This is an incomplete reference. Exclude this case from qualitative claims about lost or invented list items; retain the original automatic score and disclose the reference limitation.

## Report-ready interpretation

**RQ1.** On this 10-document Europarl test subset, adding one preceding source sentence yields a small aggregate advantage over the sentence-level model (+0.25 macro BLEU, +0.18 macro chrF, +0.0007 macro COMET). The differences vary across documents and language pairs, and 74/150 predictions do not change. Individual cases show both improvements and content omissions. Therefore the evidence supports a modest dataset-specific effect, not a claim that the model consistently uses document context to resolve pronouns or maintain terminology.

**RQ2.** LLM post-editing of the context model improves macro COMET by +0.0054, largely on English–German, and repairs some severe omissions. The lexical metrics disagree: macro BLEU falls by 0.05 and macro chrF rises by 0.05. Manual inspection also identifies a French output containing Chinese characters and some terminology regressions. Post-editing is beneficial in selected cases but is not a uniformly reliable quality improvement.

## Limitations and next reporting step

- Only 10 source documents underlie these 150 test examples; sentences within a document are not independent observations. Keep the pair and document breakdown beside aggregate scores.
- COMET and the single Europarl reference are useful diagnostics, but neither is a substitute for human assessment. A truncated German reference and a mixed-script output illustrate opposite failure modes.
- Selected examples were inspected because their score changes were large. The examples are illustrative, not a random manually annotated estimate of overall error frequency.
- A controlled comparison with unweighted loss, Top-k/Top-p candidate generation, reranking and a demonstration interface were optional in the preliminary proposal and were not part of these four arms.
- The observed examples are illustrative. Keep the already frozen test scores unchanged and avoid population-level claims about specific error types.
