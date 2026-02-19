import os
import urllib.request
import tarfile
import shutil
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ফোল্ডার তৈরি
PATHS = {
    "stt": "/model_stt",
    "trans": "/model_trans",
    "voices": "/piper_voices",
    "bin": "/piper_bin"
}

for p in PATHS.values():
    if not os.path.exists(p): os.makedirs(p)

# --- স্মার্ট ডাউনলোডার ফাংশন (Browser Headers সহ) ---
def download_safe(url, path):
    print(f"📥 Downloading: {url.split('/')[-1]}...")
    try:
        # GitHub যাতে ব্লক না করে, তাই আমরা Mozilla (Browser) সেজে রিকোয়েস্ট পাঠাবো
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("✅ Success.")
    except Exception as e:
        print(f"❌ Failed: {e}")
        # ফেইল হলে প্রসেস থামিয়ে দেবো যাতে বিল্ড লগ দেখে বোঝা যায়
        raise e

# 1. Piper Binary ডাউনলোড (Anti-Block)
piper_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
tar_path = "piper.tar.gz"
download_safe(piper_url, tar_path)

print("📦 Extracting Piper...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path="/tmp")

# বাইনারি মুভ করা
shutil.move("/tmp/piper/piper", PATHS["bin"] + "/piper")
os.chmod(PATHS["bin"] + "/piper", 0o755) # এক্সিকিউশন পারমিশন
os.remove(tar_path)

# 2. Faster-Whisper (STT)
print("Downloading Whisper...")
model = WhisperModel("medium", device="cpu", download_root=PATHS["stt"])

# 3. MADLAD-400 (Translation)
print("Downloading Translation Model...")
model_id = 'google/madlad400-3b-mt'
T5Tokenizer.from_pretrained(model_id, cache_dir=PATHS["trans"])
T5ForConditionalGeneration.from_pretrained(model_id, cache_dir=PATHS["trans"])

# 4. Piper Voices (TTS)
print("Downloading Voices...")
# বাংলা, ইংরেজি, স্প্যানিশ, জার্মান, ফ্রেঞ্চ, আরবি, হিন্দি
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

print("\n🎉 ALL DOWNLOADS COMPLETE 🎉")