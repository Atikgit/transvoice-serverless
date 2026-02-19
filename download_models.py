import os
import urllib.request
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ডিরেক্টরি তৈরি
DIRS = ["/model_stt", "/model_trans", "/piper_voices", "/usr/local/bin/piper_bin"]
for d in DIRS:
    if not os.path.exists(d): os.makedirs(d)

print("--- 1. Downloading Faster-Whisper (STT) ---")
# 'large-v3' মডেলটি সেরা কোয়ালিটির জন্য। ফাস্ট চাইলে 'medium' দিতে পারেন।
model = WhisperModel("large-v3", device="cpu", compute_type="int8", download_root="/model_stt")
print("✅ STT Model Downloaded.")

print("--- 2. Downloading MADLAD-400 (Translation) ---")
model_id = 'google/madlad400-3b-mt'
T5Tokenizer.from_pretrained(model_id, cache_dir="/model_trans")
T5ForConditionalGeneration.from_pretrained(model_id, cache_dir="/model_trans")
print("✅ Translation Model Downloaded.")

print("--- 3. Downloading Piper TTS Voices ---")
# ভয়েস লিস্ট (আপনি চাইলে আরও বাড়াতে পারেন)
VOICES = {
    "bn": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/bn/bn_IN/arijit/medium/bn_IN-arijit-medium.onnx",
    "en": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
    "es": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
    "de": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "fr": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
    "ar": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "ru": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/dmitry/medium/ru_RU-dmitry-medium.onnx",
    "hi": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/srivastava/medium/hi_IN-srivastava-medium.onnx"
}

def download_file(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        urllib.request.urlretrieve(url + ".json", path + ".json") # JSON কনফিগ জরুরি
        print(f"🔹 Downloaded: {path}")
    except Exception as e:
        print(f"❌ Failed: {path} - {e}")

for lang, url in VOICES.items():
    download_file(url, f"/piper_voices/{lang}.onnx")

print("--- All Downloads Complete ---")