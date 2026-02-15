import os
import urllib.request
import tarfile
import shutil

# --- ১০০+ ভাষার বা কাস্টম মডেলের লিংক এখানে যোগ করুন ---
# এই লিংকগুলো Sherpa-ONNX এর অফিসিয়াল রিলিজ পেজ থেকে নেওয়া
TTS_MODELS = {
    "ben": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2",
    "ara": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ara.tar.bz2",
    "eng": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2",
    "hin": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-hin.tar.bz2",
    "spa": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-spa.tar.bz2",
    "fra": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-fra.tar.bz2",
    "deu": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-deu.tar.bz2",
    # ... এখানে আপনি আরও লিংক যোগ করতে পারেন
}

BASE_DIR = "/tts_models"

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

print(f"Starting download of {len(TTS_MODELS)} TTS models...")

for lang, url in TTS_MODELS.items():
    try:
        filename = url.split("/")[-1]
        file_path = os.path.join(BASE_DIR, filename)
        
        print(f"Downloading [{lang}]: {filename}...")
        urllib.request.urlretrieve(url, file_path)
        
        print(f"Extracting [{lang}]...")
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(path=BASE_DIR)
        
        # ক্লিনআপ: জিপ ফাইল ডিলিট করে জায়গা বাঁচানো
        os.remove(file_path)
        print(f"✅ [{lang}] Success!")
        
    except Exception as e:
        # কোনো একটি ফেইল করলে আমরা শুধু ওয়ার্নিং দেব, বিল্ড আটকাবো না
        print(f"❌ FAILED [{lang}]: {str(e)}")
        # Continue to next language...

print("--- All Downloads Processed ---")