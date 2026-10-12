# Local v3 translation demo

`demo_v3.py` is a Streamlit interface for English → German/French. It shows A (sentence) and B (current sentence + previous source) with the two **balanced scaled-v3 adapters**. Checking the Qwen option also shows C (A post-edited) and D (B post-edited). The default is the four-beam decoder chosen on validation. The app sends **no reference translations** to Qwen.

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

The `--server.fileWatcherType none` flag prevents Streamlit's file watcher from inspecting optional Transformers image modules such as ZoeDepth on this installation. When editing app code, restart Streamlit to see changes. This demo translates text and does not require `torchvision`.\n\nThe first launch downloads/loads the same base mBART model used for evaluation and reads the two local adapter directories:

- `outputs/checkpoints/sentence_lora_balanced_scaled_v3/final_adapter/`
- `outputs/checkpoints/context_lora_balanced_scaled_v3/final_adapter/`

It checks each adapter's `run_metadata.json` for balanced loss, full training, the expected input format and length 128. Both adapters share one base model; choosing A/B changes the active adapter. Paste the previous English sentence and the current English sentence, select German or French, and click **Translate A and B**. The demo reports when an input is truncated to 128 mBART tokens.

To show C/D, start the Ollama application locally and make sure `qwen2.5:7b` is available (the project's model digest was recorded in the evaluation artifacts); then check **Show Qwen post-edits**. Its request uses the evaluation system prompt, schema and decoding options. If the Qwen request fails, A/B remain visible. Model loading and Qwen calls may take time, especially on first use; run a trial translation before presenting.

The app is a **live demonstration**, not an evaluation. It does not compute BLEU, chrF or COMET for typed text. Its default adapters require CUDA and have only been trained and evaluated for Europarl-style inputs. Because the GitHub clone may not yet contain the v3 adapter weights or v3 CSVs, run from `D:\CIS600\context-mt`, not from the clone, until those artifacts have been synchronized.
