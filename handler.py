import runpod
import torch
import base64
import os
import subprocess
import tempfile
import urllib.request
import tarfile
import shutil
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- কনফিগারেশন ও পাথ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATHS = {
    "stt": "/model_stt",
    "trans": "/model_trans",
    "voices": "/piper_voices",
    "bin_exe": "/piper_bin/piper"
}

def setup_system():
    """সার্ভার চালু হলে এটি ফাইল চেক ও ডাউনলোড করবে"""
    print("--- ⚙️ SYSTEM INITIALIZATION START ---")
    
    # ফোল্ডার তৈরি
    for p in PATHS.values():
        if not p.endswith("piper") and not os.path.exists(p): os.makedirs(p)

    # 1. Piper Binary ডাউনলোড
    if not os.path.exists(PATHS["bin_exe"]):
        print("📥 Downloading Piper TTS Engine...")
        url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
        tar_p = "/tmp/piper.tar.gz"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(tar_p, 'wb') as f: shutil.copyfileobj(r, f)
        with tarfile.open(tar_p, "r:gz") as tar: tar.extractall(path="/tmp")
        if not os.path.exists("/piper_bin"): os.makedirs("/piper_bin")
        shutil.move("/tmp/piper/piper", PATHS["bin_exe"])
        os.chmod(PATHS["bin_exe"], 0o755)
        print("✅ Piper Engine Ready.")

    # 2. Voices ডাউনলোড (আপাতত বাংলা, ইংরেজি, স্প্যানিশ, জার্মান)
    VOICES = {
        "bn": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/bn/bn_IN/arijit/medium/bn_IN-arijit-medium.onnx",
        "en": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
        "es": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "de": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
    }
    
    for lang, url in VOICES.items():
        v_path = os.path.join(PATHS["voices"], f"{lang}.onnx")
        if not os.path.exists(v_path):
            print(f"📥 Downloading Voice: {lang}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as r, open(v_path, 'wb') as f: shutil.copyfileobj(r, f)
            req_j = urllib.request.Request(url + ".json", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_j) as r, open(v_path + ".json", 'wb') as f: shutil.copyfileobj(r, f)
    
    print("--- ⚙️ SYSTEM INITIALIZATION DONE ---")

# --- মডেল লোডিং (Runtime) ---
setup_system()

print("🚀 Loading Whisper (STT)...")
stt_model = WhisperModel("medium", device=DEVICE, compute_type="float16", download_root=PATHS["stt"])

print("🚀 Loading MADLAD-400 (Translation)... This will take a few minutes the first time.")
tokenizer = T5Tokenizer.from_pretrained('google/madlad400-3b-mt', cache_dir=PATHS["trans"])
trans_model = T5ForConditionalGeneration.from_pretrained(
    'google/madlad400-3b-mt', 
    cache_dir=PATHS["trans"], 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(DEVICE)

print("✅ ALL SYSTEMS READY & LOADED")

# --- লজিক ---
LANG_MAP = {'bn': '<2bn>', 'ben': '<2bn>', 'es': '<2es>', 'spa': '<2es>', 'de': '<2de>', 'deu': '<2de>', 'en': '<2en>', 'eng': '<2en>'}

def run_tts(text, lang_code):
    short_lang = lang_code[:2]
    v_path = os.path.join(PATHS["voices"], f"{short_lang}.onnx")
    if not os.path.exists(v_path): v_path = os.path.join(PATHS["voices"], "en.onnx")
    
    out_path = "/tmp/out.wav"
    try:
        # shell=True বাদ দিয়ে সরাসরি ইনপুট পাস করা হলো যাতে কোনো কোটেশন এরর না আসে
        subprocess.run(
            [PATHS["bin_exe"], "--model", v_path, "--output_file", out_path],
            input=text.encode('utf-8'),
            check=True
        )
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"🔥 TTS Error: {e}")
        return None

def handler(job):
    job_input = job.get('input', {})
    audio_b64 = job_input.get("audio")
    tgt_lang = job_input.get("tgt_lang", "bn").lower()

    if not audio_b64: return {"error": "No audio provided"}

    try:
        # 1. Decode Audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(base64.b64decode(audio_b64))
            tmp_path = tmp.name

        # 2. STT
        segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
        text_in = " ".join([s.text for s in segments]).strip()
        print(f"🗣️ Heard: {text_in}")

        # 3. Translate
        token = LANG_MAP.get(tgt_lang, '<2en>')
        inputs = tokenizer(f"{token} {text_in}", return_tensors="pt").input_ids.to(DEVICE)
        outputs = trans_model.generate(inputs, max_new_tokens=200)
        text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Translated ({tgt_lang}): {text_out}")

        # 4. TTS
        audio_out = run_tts(text_out, tgt_lang)
        
        os.remove(tmp_path)
        
        return {
            "original_text": text_in,
            "translated_text": text_out,
            "audio_out": audio_out,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})