import os
import urllib.request
import tarfile
import shutil

# --- ১০০+ ভাষার বা কাস্টম মডেলের লিংক এখানে যোগ করুন ---
# এই লিংকগুলো Sherpa-ONNX এর অফিসিয়াল রিলিজ পেজ থেকে নেওয়া
TTS_MODELS = {
    "ben": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2",
    "ara": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ara.tar.bz2",
    "hin": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-hin.tar.bz2",
    "spa": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-spa.tar.bz2",
    "fra": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-fra.tar.bz2",
    "deu": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-deu.tar.bz2",
    "ita": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ita.tar.bz2",
    "rus": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-rus.tar.bz2",
    "kor": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-kor.tar.bz2",
    "jpn": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-jpn.tar.bz2",
    "por": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-por.tar.bz2",
    "tur": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-tur.tar.bz2",
    "vie": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-vie.tar.bz2",
    "urd": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-urd.tar.bz2",
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