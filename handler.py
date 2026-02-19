import runpod
import torch
import base64
import os
import subprocess
import tempfile
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_STT = "/model_stt"
MODEL_TRANS = "/model_trans"
PIPER_BIN = "/piper_bin/piper"
VOICE_DIR = "/piper_voices"

print("--- Starting Service ---")

# 1. Load Whisper
print("Loading Whisper...")
stt_model = WhisperModel(MODEL_STT, device=DEVICE, compute_type="float16")

# 2. Load MADLAD
print("Loading MADLAD-400...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_TRANS)
trans_model = T5ForConditionalGeneration.from_pretrained(MODEL_TRANS).to(DEVICE)
print("✅ Models Loaded")

# MADLAD Language Codes
LANG_MAP = {
    'bn': '<2bn>', 'ben': '<2bn>',
    'es': '<2es>', 'spa': '<2es>',
    'de': '<2de>', 'deu': '<2de>',
    'fr': '<2fr>', 'fra': '<2fr>',
    'en': '<2en>', 'eng': '<2en>',
    'ar': '<2ar>', 'ara': '<2ar>',
    'hi': '<2hi>', 'hin': '<2hi>'
}

def run_piper(text, lang):
    short_lang = lang[:2]
    model_path = os.path.join(VOICE_DIR, f"{short_lang}.onnx")
    
    # ফলব্যাক: যদি ভাষা না থাকে তবে ইংলিশে বলবে (ক্র্যাশ এড়াতে)
    if not os.path.exists(model_path):
        print(f"⚠️ Voice {short_lang} not found, using English.")
        model_path = os.path.join(VOICE_DIR, "en.onnx")
        
    output_path = "/tmp/out.wav"
    
    # Piper Command: echo text | piper ...
    cmd = f'echo "{text}" | {PIPER_BIN} --model {model_path} --output_file {output_path}'
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"🔥 Piper Error: {e}")
        return None

def handler(job):
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    tgt_lang = job_input.get("tgt_lang", "bn").lower()

    if not audio_b64: return {"error": "No audio"}

    try:
        # A. Decode Audio
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # B. STT (Speech to Text)
        segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
        text_in = " ".join([s.text for s in segments]).strip()
        print(f"🗣️ Heard: {text_in}")

        # C. Translation
        token = LANG_MAP.get(tgt_lang, '<2en>')
        inputs = tokenizer(f"{token} {text_in}", return_tensors="pt").input_ids.to(DEVICE)
        outputs = trans_model.generate(inputs, max_new_tokens=200)
        text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Translated: {text_out}")

        # D. TTS (Text to Speech)
        audio_out = run_piper(text_out, tgt_lang)

        os.remove(tmp_path)
        
        return {
            "original_text": text_in,
            "translated_text": text_out,
            "audio_out": audio_out
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})