# Local v3 translation demo

`demo_v3.py` is a Streamlit interface for English → German/French. It shows A (sentence) and B (current sentence + previous source) with the two **balanced scaled-v3 adapters**. Checking the GPT-6 option also shows C (A post-edited) and D (B post-edited). The default is the four-beam decoder chosen on validation. The app sends **no reference translations** to GPT-6.

Run it in the original Windows project, where the trained v3 adapters and `.venv` already exist. First pull the latest repository files into the GitHub clone and copy the app into the working project:

```powershell
cd D:\CIS600\Project-Graduate-upload
git pull --ff-only origin main
Copy-Item .\demo_v3.py D:\CIS600\context-mt\demo_v3.py -Force

cd D:\CIS600\context-mt
.\.venv\Scripts\Activate.ps1
python -m pip install streamlit
python -m streamlit run demo_v3.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag prevents Streamlit's file watcher from inspecting optional Transformers image modules such as ZoeDepth on this installation. When editing app code, restart Streamlit to see changes. This demo translates text and does not require `torchvision`.

The demo imports the post-edit prompt and response parser from the renamed `scripts/06_postedit_lora.py`; copy that script into the Windows working project along with `demo_v3.py` if it is missing. The first launch downloads/loads the same base mBART model used for evaluation and reads the two local adapter directories:

- `outputs/checkpoints/sentence_lora_balanced_scaled_v3/final_adapter/`
- `outputs/checkpoints/context_lora_balanced_scaled_v3/final_adapter/`

It checks each adapter's `run_metadata.json` for balanced loss, full training, the expected input format and length 128. Both adapters share one base model; choosing A/B changes the active adapter. Paste the previous English sentence and the current English sentence, select German or French, and click **Translate A and B**. The demo reports when an input is truncated to 128 mBART tokens.

To show C/D with GPT-6 Astra, set `OPENAI_API_KEY` in the shell that launches Streamlit and check **Show GPT-6 post-edits**. Each click makes two billable OpenAI Responses API calls. The demo reuses the evaluation prompt, structured output contract, and reasoning setting. If the API request fails, A/B remain visible. Run one trial before presenting. The completed Qwen experiment and its scores are historical; this live demo does not reproduce its saved C/D outputs.

The app is a **live demonstration**, not an evaluation. It does not compute BLEU, chrF or COMET for typed text. Its default adapters require CUDA and have only been trained and evaluated for Europarl-style inputs. Because the GitHub clone may not yet contain the v3 adapter weights or v3 CSVs, run from `D:\CIS600\context-mt`, not from the clone, until those artifacts have been synchronized.
