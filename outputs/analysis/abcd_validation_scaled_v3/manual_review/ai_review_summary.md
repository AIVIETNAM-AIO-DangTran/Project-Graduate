# A/B/C/D validation: blinded AI review

AI judgments; not independent human annotation. All rows compare different translations. 
Cue/non-cue is a regex-based sampling split; human-assessed context relevance is a separate label.
Counts from COMET extremes must not be reported as representative win rates.

## context

| Selection | Pair | After better meaning | Before better meaning | Tie | N |
|---|---|---:|---:|---:|---:|
| random | en-de | 5 | 3 | 0 | 8 |
| random | en-fr | 2 | 5 | 1 | 8 |
| random | both | 7 | 8 | 1 | 16 |
| extremes | en-de | 1 | 1 | 0 | 2 |
| extremes | en-fr | 2 | 0 | 0 | 2 |
| extremes | both | 3 | 1 | 0 | 4 |
| all | en-de | 6 | 4 | 0 | 10 |
| all | en-fr | 4 | 5 | 1 | 10 |
| all | both | 10 | 9 | 1 | 20 |

## llm_sentence

| Selection | Pair | After better meaning | Before better meaning | Tie | N |
|---|---|---:|---:|---:|---:|
| random | en-de | 5 | 2 | 1 | 8 |
| random | en-fr | 4 | 3 | 1 | 8 |
| random | both | 9 | 5 | 2 | 16 |
| extremes | en-de | 1 | 1 | 0 | 2 |
| extremes | en-fr | 1 | 1 | 0 | 2 |
| extremes | both | 2 | 2 | 0 | 4 |
| all | en-de | 6 | 3 | 1 | 10 |
| all | en-fr | 5 | 4 | 1 | 10 |
| all | both | 11 | 7 | 2 | 20 |

## llm_context

| Selection | Pair | After better meaning | Before better meaning | Tie | N |
|---|---|---:|---:|---:|---:|
| random | en-de | 4 | 2 | 2 | 8 |
| random | en-fr | 3 | 3 | 2 | 8 |
| random | both | 7 | 5 | 4 | 16 |
| extremes | en-de | 1 | 1 | 0 | 2 |
| extremes | en-fr | 1 | 1 | 0 | 2 |
| extremes | both | 2 | 2 | 0 | 4 |
| all | en-de | 5 | 3 | 2 | 10 |
| all | en-fr | 4 | 4 | 2 | 10 |
| all | both | 9 | 7 | 4 | 20 |
