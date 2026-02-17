import os
import urllib.request
import json
import tarfile

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

MANUAL_LINKS = {
    "ben": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2",
    "urd": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-urd.tar.bz2"
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
            # শুধু ফাইলের নাম এবং ডাউনলোড লিংকের ডিকশনারি রিটার্ন করবে
            return {asset['name']: asset['browser_download_url'] for asset in data['assets']}
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {}

def download_and_extract(url, lang_code):
    filename = url.split("/")[-1]
    file_path = os.path.join(BASE_DIR, filename)
    print(f"📥 Downloading [{lang_code}]: {filename}...")
    
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"📦 Extracting [{lang_code}]...")
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(path=BASE_DIR)
        os.remove(file_path)
        print(f"✅ [{lang_code}] Success!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

# মেইন প্রসেস
assets = get_release_assets()
if not assets:
    print("⚠️ No assets found from API. Check internet connection.")
    exit(1)

print(f"Found {len(assets)} available models in release.")

for lang, codes in TARGET_LANGS.items():
    found = False
    
    # ১. প্রথমে Piper মডেল খোঁজা (বেস্ট কোয়ালিটি)
    for code in codes:
        # যেমন: vits-piper-en_US...
        piper_match = next((name for name in assets if f"vits-piper-{code}" in name), None)
        if piper_match:
            download_and_extract(assets[piper_match], lang)
            found = True
            break
            
    # ২. Piper না পেলে MMS মডেল খোঁজা
    if not found:
        for code in codes:
            # যেমন: vits-mms-ben...
            mms_match = next((name for name in assets if f"vits-mms-{code}" in name), None)
            if mms_match:
                download_and_extract(assets[mms_match], lang)
                found = True
                break
    
    if not found:
        print(f"⚠️ Skipping [{lang}]: No model found in release assets matching codes {codes}")

print("--- Setup Finished ---")