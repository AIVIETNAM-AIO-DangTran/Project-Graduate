# Blind translation review protocol (scaled v3)

## Purpose and materials

Assess whether context (B versus A) or **GPT-6** post-editing (C versus A, D versus B) fixes meaning, pronoun/reference and terminology errors on the **validation** split. Generate a separate 60-item blind worksheet at `outputs/analysis/abcd_validation_scaled_v3_gpt6/manual_review/blind_review.csv` with `python scripts/11_build_manual_review.py --kind abcd --postedit-model gpt6`. Give annotators only that file and this protocol. Keep `answer_key.csv`, COMET scores and model identities hidden until annotation is complete.

The sheet has 10 changed translations for each language pair and comparison: four sampled from sentences matching an English context-cue regex, four from sentences without those cues, plus one highest and one lowest COMET change. **Only the first eight are random within their respective changed-output strata; the extremes are diagnostic examples.** None is a random sample of all test translations. The candidate order is masked.

## Annotator instructions

Use one German-proficient and one French-proficient reviewer at minimum; preferably two independent reviewers per pair. Read the current English source (`src`), previous English source (`previous_src`) and target-language reference. Compare `candidate_1` and `candidate_2` without looking up the arm identity. The reference helps catch omissions, but a valid paraphrase can still be correct when its surface wording differs from the reference. For each row, fill in:

- `context_relevant_0_1_unsure`: 1 only when information in the previous sentence is needed to interpret or translate the current one; otherwise 0. A cue word by itself is insufficient.
- `pronoun_issue_0_1_unsure`: 1 when a candidate gets a person, number, gender or pronoun referent wrong or unclear; otherwise 0.
- `cross_sentence_reference_0_1_unsure`: 1 when the current sentence refers to an entity/event in the previous sentence and a candidate mishandles it; otherwise 0.
- `term_issue_0_1_unsure`: 1 when a term is mistranslated or inconsistent with the preceding sentence; otherwise 0.
- `meaning_winner_1_2_tie`: `1`, `2` or `tie`; prioritize accuracy, completeness and absence of added information over stylistic preference.
- `fluency_winner_1_2_tie`: `1`, `2` or `tie` for grammaticality/naturalness, separately from meaning.
- `critical_error_candidate_1_0_1` and `critical_error_candidate_2_0_1`: 1 for an altered negation, number, named entity, large omission, or insertion of previous-sentence content; otherwise 0.
- `notes`: quote a short decisive phrase, especially when choosing a meaning winner or critical error.

For uncertain binary fields enter `unsure` instead of guessing. If both candidates are good or both make equivalent mistakes, use `tie`. Work independently before adjudication; record reviewer ID, language proficiency and date in a separate local record. Preserve individual completed files. Do not use the AI-assisted review as a key.

## Summarizing and limitations

After both reviewers complete the sheets, calculate agreement on meaning wins and critical errors, reconcile disputed items, then unblind with `answer_key.csv`. `scripts/12_summarize_abcd_review.py` summarizes **one** completed review with the fixed 60-row layout. Save each annotator's completed CSV separately and run it with `--postedit-model gpt6 --reviewer-type human --review <completed-file> --answer-key <answer-key> --output-dir <new-directory>`. Use separate output directories for each reviewer; do not merge individual judgments before computing agreement. Report results separately by language, comparison and sampling stratum. Never extrapolate a percentage from the four selected COMET extremes to the population, and do not present the current AI review as human judgments.

This review is optional evidence about error types, not a replacement for document-level paired metric uncertainty. It is limited to selected changed outputs on validation. Reusing it for prompt tuning would make it unsuitable for a fresh validation claim.

## Commands after the review

```powershell
python scripts\11_build_manual_review.py --kind abcd --postedit-model gpt6
# Share only outputs\analysis\abcd_validation_scaled_v3_gpt6\manual_review\blind_review.csv
# After a bilingual reviewer returns a completed copy:
python scripts\12_summarize_abcd_review.py --postedit-model gpt6 --reviewer-type human `
  --review path\to\completed_review.csv `
  --output-dir outputs\analysis\abcd_validation_scaled_v3_gpt6\manual_review\reviewer_1
```

These scripts read saved validation results and never call GPT-6. Running the builder a second time into an existing directory deliberately fails rather than overwriting annotations. For the decoding comparison, run `python scripts/11_build_manual_review.py --kind decoding`. Its default score table and output directory use GPT-6 reranking.
