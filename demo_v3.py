"""Local Streamlit demo of the frozen Europarl v3 balanced LoRA arms."""

import json
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import streamlit as st
import torch
from peft import PeftModel
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from scripts.postedit_lora import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    api_call,
    parse_translation,
)

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
ADAPTERS = {
    "sentence": "sentence_lora_balanced_scaled_v3",
    "context": "context_lora_balanced_scaled_v3",
}
TARGETS = {"German (en-de)": ("de_DE", "German"), "French (en-fr)": ("fr_XX", "French")}


def check_adapter(name, expected_format):
    directory = PROJECT_ROOT / "outputs" / "checkpoints" / name
    adapter = directory / "final_adapter"
    metadata = directory / "run_metadata.json"
    for file in (adapter / "adapter_config.json", adapter / "adapter_model.safetensors", metadata):
        if not file.is_file():
            raise FileNotFoundError(f"Missing v3 artifact: {file}")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if (info.get("input_format") != expected_format or info.get("loss_mode") != "balanced"
            or info.get("max_length") != 128 or info.get("training_mode") != "full_training"):
        raise ValueError(f"Adapter metadata differs from the frozen demo setup: {metadata}")
    return adapter


@st.cache_resource(show_spinner="Loading mBART and the two v3 adapters...")
def load_models():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. This demo expects the project's GPU environment.")
    sentence = check_adapter(ADAPTERS["sentence"], "src")
    context = check_adapter(ADAPTERS["context"], "src </s> previous_src")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "en_XX"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    base = MBartForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float16)
    base.to("cuda")
    model = PeftModel.from_pretrained(base, str(sentence), adapter_name="sentence")
    model.load_adapter(str(context), adapter_name="context", is_trainable=False)
    model.eval()
    return tokenizer, model, threading.Lock()


def translate(tokenizer, model, lock, src, previous, target_code, arm):
    source = src if arm == "sentence" else f"{src} {tokenizer.eos_token} {previous}"
    # The tokenizer is constant after initialization; adapter selection and generation
    # share a lock because the model is cached across Streamlit sessions.
    with lock:
        tokenizer.src_lang = "en_XX"
        untruncated_length = len(tokenizer(source, truncation=False)["input_ids"])
        encoded = tokenizer(
            source, return_tensors="pt", truncation=True, max_length=128
        ).to("cuda")
        model.set_adapter(arm)
        with torch.inference_mode():
            ids = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                num_beams=4,
                do_sample=False,
                max_length=128,
            )
        prediction = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    return prediction, untruncated_length > 128


def postedit(src, previous, prediction, target_language, use_context):
    supplied = {
        "target_language": target_language,
        "current_source": src,
        "draft_translation": prediction,
    }
    if use_context:
        supplied["previous_source"] = previous
    response = api_call(
        "http://127.0.0.1:11434",
        "/api/chat",
        120,
        {
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(supplied, ensure_ascii=False)},
            ],
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 2048, "num_predict": 512},
        },
    )
    return parse_translation(response)


def main():
    st.set_page_config(page_title="Europarl v3 translation demo", page_icon="🌐")
    st.title("Europarl v3 · translation demo")
    st.caption("mBART-50 + LoRA | four-beam decoding | optional local Qwen post-editing")
    with st.form("translation"):
        language = st.selectbox("Target language", list(TARGETS))
        previous = st.text_area("Previous English sentence (for context)", height=100)
        current = st.text_area("Current English sentence to translate", height=130)
        use_llm = st.checkbox("Show Qwen post-edits (C and D)", value=False)
        submitted = st.form_submit_button("Translate A and B")
    if not submitted:
        return
    src, prev = current.strip(), previous.strip()
    if not src or not prev:
        st.error("Please enter both the current and previous English sentences.")
        return
    target_code, target_language = TARGETS[language]
    try:
        tokenizer, model, lock = load_models()
        with st.spinner("Generating A and B..."):
            outputs = {
                arm: translate(tokenizer, model, lock, src, prev, target_code, arm)
                for arm in ("sentence", "context")
            }
    except Exception as exc:
        st.error(f"Cannot load or run the v3 adapters: {exc}")
        return

    columns = st.columns(2)
    for col, arm, title in zip(columns, ("sentence", "context"), ("A · Sentence", "B · Context")):
        with col:
            st.subheader(title)
            st.write(outputs[arm][0])
            if outputs[arm][1]:
                st.warning("Source exceeded 128 mBART tokens and was truncated on the right.")

    if use_llm:
        for col, arm, title in zip(columns, ("sentence", "context"), ("C · A + Qwen", "D · B + Qwen")):
            with col:
                st.subheader(title)
                try:
                    with st.spinner("Post-editing with local Ollama..."):
                        st.write(postedit(src, prev, outputs[arm][0], target_language, arm == "context"))
                except Exception as exc:
                    st.error(f"Qwen unavailable or returned an invalid answer: {exc}")
    st.caption("Live translations are demonstrations. BLEU, chrF and COMET require held-out references and are reported separately.")


if __name__ == "__main__":
    main()
