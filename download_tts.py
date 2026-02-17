import os
import urllib.request
import json
import tarfile

TARGET_LANGS = {
    'ben': ['bn', 'ben'], 'hin': ['hi', 'hin'], 'ara': ['ar', 'ara'], 
    'urd': ['ur', 'urd'], 'vie': ['vi', 'vie'], 'tur': ['tr', 'tur'], 
    'spa': ['es', 'spa'], 'fra': ['fr', 'fra'], 'deu': ['de', 'deu'], 
    'eng': ['en', 'eng'], 'jpn': ['ja', 'jpn'], 'kor': ['ko', 'kor']
}

# যে লিঙ্কগুলো API তে পাওয়া যাচ্ছে না সেগুলোর ব্যাকআপ
MANUAL_LINKS = {
    "ben": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2",
    "urd": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-urd.tar.bz2",
    "jpn": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-jpn.tar.bz2",
    "kor": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-kor.tar.bz2"
}

BASE_DIR = "/tts_models"
if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)

def download_and_extract(url, lang):
    file_path = os.path.join(BASE_DIR, url.split("/")[-1])
    try:
        print(f"📥 Downloading [{lang}] -> {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        
        print(f"📦 Extracting [{lang}]...")
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(path=BASE_DIR)
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"❌ Error downloading {lang}: {e}")
        return False

# ১. API থেকে সার্চ করা
api_url = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
assets = {}
try:
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())
        assets = {a['name']: a['browser_download_url'] for a in data['assets']}
except: pass

# ২. মেইন ডাউনলোড লুপ
for lang, codes in TARGET_LANGS.items():
    downloaded = False
    # প্রথমে API তে খোঁজা
    for code in codes:
        match = next((url for name, url in assets.items() if f"-{code}-" in name or f"-{code}." in name), None)
        if match:
            if download_and_extract(match, lang):
                downloaded = True
                break
    
    # না পেলে ম্যানুয়াল লিঙ্ক ব্যবহার
    if not downloaded and lang in MANUAL_LINKS:
        download_and_extract(MANUAL_LINKS[lang], lang)

print("--- Build Process Ready ---")