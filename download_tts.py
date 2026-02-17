import os
import urllib.request
import json
import tarfile
import ssl

# SSL সার্টিফিকেটের সমস্যা এড়ানোর জন্য
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# ১. ম্যানুয়াল ব্যাকআপ লিংক (যেগুলো API তে পাওয়া যাচ্ছে না)
# ==========================================
MANUAL_FALLBACK = {
    "ben": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2",
    "urd": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-urd.tar.bz2",
    "jpn": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-jpn.tar.bz2",
    "kor": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-kor.tar.bz2"
}

# আপনার কাঙ্খিত ভাষার লিস্ট (ISO Code)
TARGET_LANGS = {
    'ben': ['bn', 'ben'],         # Bengali
    'hin': ['hi', 'hin'],         # Hindi
    'ara': ['ar', 'ara', 'arb'],  # Arabic
    'urd': ['ur', 'urd'],         # Urdu
    'vie': ['vi', 'vie'],         # Vietnamese
    'tur': ['tr', 'tur'],         # Turkish
    'spa': ['es', 'spa'],         # Spanish
    'fra': ['fr', 'fra'],         # French
    'deu': ['de', 'deu'],         # German
    'eng': ['en', 'eng'],         # English
    'jpn': ['ja', 'jpn'],         # Japanese
    'kor': ['ko', 'kor'],         # Korean
    'ind': ['id', 'ind']          # Indonesian
}

BASE_DIR = "/tts_models"
if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)

def get_release_assets():
    """GitHub API থেকে রিয়েল-টাইম ফাইলের লিস্ট নিয়ে আসা"""
    print("🔍 Fetching latest model list from GitHub API...")
    url = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            return {asset['name']: asset['browser_download_url'] for asset in data['assets']}
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {}

def download_and_extract(url, lang_code):
    filename = url.split("/")[-1]
    file_path = os.path.join(BASE_DIR, filename)
    print(f"📥 Downloading [{lang_code}]: {filename}...")
    
    try:
        # User-Agent হেডার যোগ করা হয়েছে যাতে 403 Forbidden না আসে
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            import shutil
            shutil.copyfileobj(response, out_file)
            
        print(f"📦 Extracting [{lang_code}]...")
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(path=BASE_DIR)
        os.remove(file_path)
        print(f"✅ [{lang_code}] Success!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

# ==========================================
# মেইন প্রসেস শুরু
# ==========================================
assets = get_release_assets()
if not assets:
    print("⚠️ No assets found from API. Using manual fallback only.")

print(f"Found {len(assets)} available models in release.")

for lang, codes in TARGET_LANGS.items():
    found = False
    
    # ধাপ ১: স্মার্ট সার্চ (API থেকে খোঁজা)
    if assets:
        # ১.১ Piper মডেল খোঁজা (বেস্ট কোয়ালিটি)
        for code in codes:
            piper_match = next((name for name in assets if f"vits-piper-{code}" in name), None)
            if piper_match:
                if download_and_extract(assets[piper_match], lang):
                    found = True
                    break
                
        # ১.২ Piper না পেলে MMS মডেল খোঁজা
        if not found:
            for code in codes:
                mms_match = next((name for name in assets if f"vits-mms-{code}" in name), None)
                if mms_match:
                    if download_and_extract(assets[mms_match], lang):
                        found = True
                        break
    
    # ==========================================
    # ধাপ ২: ম্যানুয়াল ফলব্যাক (কোডটি এখানে বসানো হয়েছে)
    # ==========================================
    if not found and lang in MANUAL_FALLBACK:
        print(f"🔗 Attempting manual fallback for {lang}...")
        if download_and_extract(MANUAL_FALLBACK[lang], lang):
            found = True

    # ফাইনাল চেক
    if not found:
        print(f"⚠️ Skipping [{lang}]: No model found in release or manual list.")

print("--- Setup Finished ---")