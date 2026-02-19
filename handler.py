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

# কনফিগারেশন
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATHS = {
    "stt": "/model_stt",
    "trans": "/model_trans",
    "voices": "/piper_voices",
    "bin_dir": "/piper_bin",
    "bin_exe": "/piper_bin/piper"
}

# ফোল্ডার তৈরি
for p in PATHS.values():
    if not p.endswith("piper"): # ফাইল পাথ বাদ দিয়ে
        if not os.path.exists(p): os.makedirs(p)

# --- ১. ডাউনলোডার ফাংশন (RunTime) ---
def download_safe(url, path):
    if os.path.exists(path): return # অলরেডি থাকলে স্কিপ
    print(f"📥 Downloading: {url.split('/')[-1]}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Download Error: {e}")

def setup_piper():
    if os.path.exists(PATHS["bin_exe"]): return
    
    print("⚙️ Setting up Piper binary...")
    tar_path = "/tmp/piper.tar.gz"
    url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
    
    download_safe(url, tar_path)
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path="/tmp")
    
    if os.path.exists(PATHS["bin_exe"]): os.remove(PATHS["bin_exe"])
    shutil.move("/tmp/piper/piper", PATHS["bin_exe"])
    os.chmod(PATHS["bin_exe"], 0o755)
    print("✅ Piper Ready.")

def setup_voices():
    # বাংলা সহ প্রধান ভাষা
    VOICES = {
        "bn": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/bn/bn_IN/arijit/medium/bn_IN-arijit-medium.onnx",
        "en": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
        "es": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "de": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        "fr": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
        "ar": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
        "hi": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/srivastava/medium/hi_IN-srivastava-medium.onnx"
    }
    for lang, url in VOICES.items():
        dest = f"{PATHS['voices']}/{lang}.onnx"
        download_safe(url, dest)
        download_safe(url + ".json", dest + ".json")

# --- ২. মডেল ইনিশিয়ালাইজেশন ---
print("--- 🚀 STARTUP: Checking Models ---")
setup_piper()
setup_voices()

print("Loading Whisper (STT)...")
stt_model = WhisperModel("medium", device=DEVICE, compute_type="float16", download_root=PATHS["stt"])

print("Loading MADLAD (Trans)...")
tokenizer = T5Tokenizer.from_pretrained('google/madlad400-3b-mt', cache_dir=PATHS["trans"])
trans_model = T5ForConditionalGeneration.from_pretrained('google/madlad400-3b-mt', cache_dir=PATHS["trans"]).to(DEVICE)
print("✅ SYSTEM READY")

# --- ৩. মেইন প্রসেসিং ---
LANG_MAP = {
    'bn': '<2bn>', 'ben': '<2bn>', 'es': '<2es>', 'spa': '<2es>',
    'de': '<2de>', 'deu': '<2de>', 'fr': '<2fr>', 'fra': '<2fr>',
    'en': '<2en>', 'eng': '<2en>', 'ar': '<2ar>', 'ara': '<2ar>',
    'hi': '<2hi>', 'hin': '<2hi>'
}

def run_piper_tts(text, lang):
    short_lang = lang[:2]
    model_path = os.path.join(PATHS["voices"], f"{short_lang}.onnx")
    if not os.path.exists(model_path): model_path = os.path.join(PATHS["voices"], "en.onnx")
    
    out_path = "/tmp/out.wav"
    cmd = f'echo "{text}" | {PATHS["bin_exe"]} --model {model_path} --output_file {out_path}'
    try:
        subprocess.run(cmd, shell=True, check=True)
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def handler(job):
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    tgt_lang = job_input.get("tgt_lang", "bn").lower()

    if not audio_b64: return {"error": "No audio"}

    try:
        audio_data = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # STT
        segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
        text_in = " ".join([s.text for s in segments]).strip()

        # Trans
        token = LANG_MAP.get(tgt_lang, '<2en>')
        inputs = tokenizer(f"{token} {text_in}", return_tensors="pt").input_ids.to(DEVICE)
        outputs = trans_model.generate(inputs, max_new_tokens=200)
        text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # TTS
        audio_out = run_piper_tts(text_out, tgt_lang)
        os.remove(tmp_path)

        return {"original_text": text_in, "translated_text": text_out, "audio_out": audio_out}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})