import runpod
import torch
import base64
import os
import subprocess
import tempfile
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# কনফিগারেশন
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_STT_PATH = "/model_stt"
MODEL_TRANS_PATH = "/model_trans"
PIPER_BINARY = "/usr/local/bin/piper_bin/piper"
VOICE_DIR = "/piper_voices"

print("--- Initializing AI Models ---")

# 1. Load Whisper (STT)
print("Loading Faster-Whisper...")
stt_model = WhisperModel(MODEL_STT_PATH, device=DEVICE, compute_type="float16")

# 2. Load MADLAD-400 (Translation)
print("Loading MADLAD-400...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_TRANS_PATH)
trans_model = T5ForConditionalGeneration.from_pretrained(MODEL_TRANS_PATH).to(DEVICE)

print("✅ All Models Ready!")

# ভাষা কোড ম্যাপিং (ISO to MADLAD format <2xx>)
LANG_MAP = {
    'bn': '<2bn>', 'ben': '<2bn>',
    'es': '<2es>', 'spa': '<2es>',
    'de': '<2de>', 'deu': '<2de>',
    'fr': '<2fr>', 'fra': '<2fr>',
    'hi': '<2hi>', 'hin': '<2hi>',
    'ar': '<2ar>', 'ara': '<2ar>',
    'ru': '<2ru>', 'rus': '<2ru>',
    'en': '<2en>', 'eng': '<2en>'
}

def run_tts_piper(text, lang_code):
    """Piper TTS (Binary) দিয়ে অডিও তৈরি"""
    # ২ অক্ষরের কোড নিশ্চিত করা (ben -> bn)
    short_code = lang_code[:2]
    
    voice_file = os.path.join(VOICE_DIR, f"{short_code}.onnx")
    # যদি স্পেসিফিক ভয়েস না থাকে, ডিফল্ট ইংলিশ ব্যবহার হবে (ক্র্যাশ এড়াতে)
    if not os.path.exists(voice_file):
        print(f"⚠️ Voice not found for {lang_code}, using English.")
        voice_file = os.path.join(VOICE_DIR, "en.onnx")

    output_wav = "/tmp/output.wav"
    
    # Piper Command Execution
    cmd = f'echo "{text}" | {PIPER_BINARY} --model {voice_file} --output_file {output_wav}'
    try:
        subprocess.run(cmd, shell=True, check=True)
        with open(output_wav, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"🔥 TTS Error: {e}")
        return None

def handler(job):
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    tgt_lang = job_input.get("tgt_lang", "bn").lower() # টার্গেট ল্যাঙ্গুয়েজ

    if not audio_b64: return {"error": "No audio provided"}

    try:
        # --- ধাপ ১: অডিও সেভ করা ---
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # --- ধাপ ২: Speech to Text (Whisper) ---
        segments, info = stt_model.transcribe(temp_audio_path, beam_size=5)
        detected_text = " ".join([segment.text for segment in segments]).strip()
        print(f"🗣️ Detected ({info.language}): {detected_text}")

        # --- ধাপ ৩: Translation (MADLAD) ---
        # টার্গেট ভাষার কোড ফরম্যাট করা (যেমন: <2bn>)
        madlad_token = LANG_MAP.get(tgt_lang, '<2en>')
        input_ids = tokenizer(f"{madlad_token} {detected_text}", return_tensors="pt").input_ids.to(DEVICE)
        
        outputs = trans_model.generate(input_ids, max_new_tokens=200)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Translated to {tgt_lang}: {translated_text}")

        # --- ধাপ ৪: Text to Speech (Piper) ---
        tts_audio = run_tts_piper(translated_text, tgt_lang)

        # ক্লিনআপ
        os.remove(temp_audio_path)

        return {
            "original_text": detected_text,
            "translated_text": translated_text,
            "audio_out": tts_audio,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})