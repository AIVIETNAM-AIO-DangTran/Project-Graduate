# GPT-6 Astra post-editing follow-up

This follow-up changes only the LLM post-editor. Use the frozen balanced mBART v3 A/B predictions, the same source/context fields and post-edit prompt, and the same Europarl validation/test references. `gpt-6-astra` is an OpenAI API model; it cannot be pulled with Ollama. ChatGPT subscription access does not establish API access or cover API usage. GPT-6 was subsequently evaluated on both frozen v3 splits; the results below replace the earlier pending status.

`scripts/06_postedit_lora.py` now defaults to `--provider openai --model gpt-6-astra`. It sends a single source/draft per Responses API request with low reasoning effort and strict JSON output. It does not send `tgt` or the reference translation. The script saves each completed row so interrupted runs can resume, logs reported token usage, and never writes the API key. New results use `_llm_gpt6_` in filenames; the old `_llm_` Qwen results remain unchanged. An exact model change or input hash change requires a distinct output directory, rather than resuming an incompatible run.

## Completed v3 follow-up (27 October 2026)

| Split | Arm | Macro BLEU | Macro chrF | Macro COMET |
| --- | --- | ---: | ---: | ---: |
| Validation | C: sentence + GPT-6 | 36.97 | 64.24 | 0.8881 |
| Validation | D: context + GPT-6 | 36.92 | 64.21 | 0.8896 |
| Test | C: sentence + GPT-6 | 37.08 | 64.58 | 0.8879 |
| Test | D: context + GPT-6 | 37.17 | 64.64 | 0.8888 |

Both C−A and D−B improve BLEU, chrF, and COMET on validation and test. A 1,000-replication paired document bootstrap for the test gives D−B intervals of [+3.06, +4.26] BLEU, [+2.55, +3.37] chrF, and [+0.0166, +0.0221] COMET. With all four arms included, the same document resamples give test C−A macro COMET +0.01796 [+0.01560, +0.02075] and D−C +0.00083 [+0.00012, +0.00153]. The D−C difference is small; the underlying B−A intervals include zero. See [full v3 results](scaled_v3_results.md) and the local JSON reports. These intervals are conditional on one frozen training seed and one LLM run; independent bilingual review is still pending.

The historical Qwen results remain separate. GPT-6 test macro COMET is 0.8879/0.8888 for C/D versus Qwen's 0.8704/0.8722. The 1,500 matched test rows give paired GPT-6 minus Qwen macro COMET differences of +0.01752 [+0.01510, +0.02035] for C and +0.01656 [+0.01436, +0.01940] for D under 1,000 shared-document bootstrap resamples. Provider/runtime and decoding behavior may differ, so this is a matched pipeline comparison rather than a model-weights-only effect.

## Windows PowerShell setup

Run from `D:\CIS600\context-mt` after copying the updated scripts. Ensure that `scripts\05_evaluate_lora.py --variant scaled_balanced --split validation` has already created the frozen A/B predictions and metrics.

Set the key for this PowerShell session only. Do not paste it into chat, Git, notebook outputs, or command arguments:

```powershell
$secure = Read-Host "OpenAI API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr); Remove-Variable secure, ptr }
```

Smoke run: two examples per language pair per arm (eight paid API requests):

```powershell
python scripts\06_postedit_lora.py `
  --dataset scaled_v3 --split validation --variant both `
  --provider openai --model gpt-6-astra --limit-per-pair 2
```

Inspect the sample translations and `outputs\metrics\*_llm_gpt6_validation_smoke2_metrics.json`. If the API rejects the model, check the API project's access to `gpt-6-astra`; do not try `ollama pull` or relabel Qwen predictions as GPT-6.

Full validation (1,500 examples per arm; 3,000 billable post-edit requests):

```powershell
python scripts\06_postedit_lora.py `
  --dataset scaled_v3 --split validation --variant both `
  --provider openai --model gpt-6-astra

if ($LASTEXITCODE -ne 0) { throw "GPT-6 validation post-edit did not finish." }

.\.venv-comet\Scripts\python.exe scripts\07_evaluate_comet.py `
  --dataset scaled_v3 --split validation --postedit-model gpt6 `
  --device cuda --batch-size 4
```

After reviewing validation quality and cost, use the same frozen configuration once on the held-out test:

```powershell
python scripts\06_postedit_lora.py `
  --dataset scaled_v3 --split test --variant both `
  --provider openai --model gpt-6-astra

if ($LASTEXITCODE -ne 0) { throw "GPT-6 test post-edit did not finish." }

.\.venv-comet\Scripts\python.exe scripts\07_evaluate_comet.py `
  --dataset scaled_v3 --split test --postedit-model gpt6 `
  --device cuda --batch-size 4
```

The GPT-6 COMET files are `outputs/metrics/comet_{validation,test}_scaled_v3_gpt6_metrics.json`; the Qwen files retain their old names. Compare A/B with C/D within each GPT-6 report and compare the saved Qwen report separately. Differences in provider, model, and decoding behavior prevent a perfectly controlled model-only ablation; state this limitation in the report. Use document-level confidence intervals and bilingual review before claiming better translation quality.

For paired document-level intervals after COMET, run `python scripts\13_document_bootstrap.py --split validation --postedit-model gpt6` (and then `--split test`). The old Qwen run remains available with `--postedit-model qwen`.

The GPT-6 Astra API charges by input and output tokens; check the current pricing and your project's spending limit before starting 3,000 requests. The script records usage from the returned Responses metadata, but no cost is fixed in advance. To reproduce the historical Qwen run, pass `--provider ollama --model qwen2.5:7b` to step 06 and `--postedit-model qwen` to step 07.
