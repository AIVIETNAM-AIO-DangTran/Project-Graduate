# Existing AI-assisted blind review: diagnostic counts

This summary uses the repository's existing `ai_review.csv` and `answer_key.csv` (60 validation comparisons). It is **not** an independent bilingual human study. Only changed outputs were sampled. The 16 random rows per comparison are stratified by language pair and a simple English cue regex; the other four are COMET extremes and are excluded from the random rows below.

| Comparison | Random changed rows: after / before / tie | Context-relevant rows among all 20: after / before / tie | Pronoun-issue rows among all 20: after / before / tie | Term-issue rows among all 20: after / before / tie |
| --- | --- | --- | --- | --- |
| Context B vs A | 7 / 8 / 1 | 4 / 3 / 1 (n=8) | 2 / 1 / 0 (n=3) | 3 / 4 / 0 (n=7) |
| Qwen C vs A | 9 / 5 / 2 | 6 / 4 / 1 (n=11) | 4 / 1 / 1 (n=6) | 7 / 4 / 0 (n=11) |
| Qwen D vs B | 7 / 5 / 4 | 5 / 5 / 3 (n=13) | 2 / 0 / 2 (n=4) | 3 / 2 / 3 (n=8) |

`after` denotes B, C or D in the respective comparison. The phenomenon subsets overlap and are small. These are counts of AI labels, not accuracy estimates or a claim of context improvement. An independent reviewer should use the still-blank `blind_review.csv` and the rubric in `docs/human_review_protocol.md` without seeing this file or the answer key.
