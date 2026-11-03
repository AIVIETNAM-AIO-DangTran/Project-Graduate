# GPT-6 reranking of saved Top-k / Top-p candidates

This is a **validation-only** extension of the frozen scaled-v3 context
adapter. The existing three sampled candidates plus the beam candidate are
read from each `candidates.jsonl`; the mBART generation stage is not rerun.
GPT-6 chooses an index and cannot rewrite a candidate. References are withheld
from the API. Outputs are written beside, rather than over, the Qwen runs.

Run from `D:\CIS600\context-mt` after copying the updated scripts into
`scripts\`. Activate `.venv` and set `OPENAI_API_KEY` in the **current**
PowerShell session as for the successful GPT-6 post-edit run. Never print or
commit the key. The scripts require the original full validation candidates
under `outputs/decoding/scaled_v3/`, the unchanged scaled-v3 manifest, the
context LoRA adapter and the beam predictions.

First run a two-per-pair smoke subset for each method (at most eight API calls
total). Each run is checkpointed and has an independent output directory:

```powershell
cd D:\CIS600\context-mt
foreach ($method in @("top_k", "top_p")) {
    python scripts\15_rerank_decoding_gpt6.py --stage rerank --method $method --limit-per-pair 2
    if ($LASTEXITCODE -ne 0) { throw "GPT-6 smoke rerank failed: $method" }
    python scripts\15_rerank_decoding_gpt6.py --stage score --method $method --limit-per-pair 2
    if ($LASTEXITCODE -ne 0) { throw "GPT-6 smoke score failed: $method" }
}
```

Inspect the output and usage before the full run. The full run makes up to
3,000 GPT-6 requests (one per row per method unless candidates collapse to
one unique option). It incurs API usage. `--max-api-calls 50` is an optional
per-invocation cap: reissue the same command to resume at the next row. Never
use `--limit-per-pair` for the full run:

```powershell
foreach ($method in @("top_k", "top_p")) {
    python scripts\15_rerank_decoding_gpt6.py --stage rerank --method $method
    if ($LASTEXITCODE -ne 0) { throw "GPT-6 rerank failed: $method" }
    python scripts\15_rerank_decoding_gpt6.py --stage score --method $method
    if ($LASTEXITCODE -ne 0) { throw "GPT-6 score failed: $method" }
}
```

BLEU/chrF and prediction tables are in directories ending `_gpt6`:

- `outputs/decoding/scaled_v3/validation_top_k_k50_n3_t0p8_seed42_gpt6/`
- `outputs/decoding/scaled_v3/validation_top_p_p0p9_n3_t0p8_seed42_gpt6/`

Then run COMET in the separate COMET environment:

```powershell
.\.venv-comet\Scripts\python.exe scripts\09_evaluate_decoding_comet.py `
  --dataset scaled_v3 --reranker gpt6 `
  --top-k-dir outputs/decoding/scaled_v3/validation_top_k_k50_n3_t0p8_seed42_gpt6 `
  --top-p-dir outputs/decoding/scaled_v3/validation_top_p_p0p9_n3_t0p8_seed42_gpt6 `
  --device cuda --batch-size 4
```

COMET writes `outputs/decoding/scaled_v3/validation_comet_comparison_gpt6/`.
This reuses the verified frozen B beam scores; it does not make GPT-6 API
calls. Compare beam, Top-k + GPT-6 and Top-p + GPT-6 on **validation**.
Do not describe these validation scores as held-out test results. The existing
Qwen decoding comparison remains an explicitly labeled historical result.
