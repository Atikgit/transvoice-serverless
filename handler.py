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

# পাথ সেটআপ
PATHS = {
    "stt": "/model_stt",
    "trans": "/model_trans",
    "voices": "/piper_voices",
    "bin_exe": "/piper_bin/piper"
}

def setup_all():
    """সার্ভার চালু হওয়ার পর এটি একবারই রান হবে"""
    for p in PATHS.values():
        if not p.endswith("piper") and not os.path.exists(p): os.makedirs(p)

    # Piper Binary Setup
    if not os.path.exists(PATHS["bin_exe"]):
        print("📥 Downloading Piper Binary...")
        url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
        tar_p = "/tmp/piper.tar.gz"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(tar_p, 'wb') as f: shutil.copyfileobj(r, f)
        with tarfile.open(tar_p, "r:gz") as tar: tar.extractall(path="/tmp")
        if not os.path.exists("/piper_bin"): os.makedirs("/piper_bin")
        shutil.move("/tmp/piper/piper", PATHS["bin_exe"])
        os.chmod(PATHS["bin_exe"], 0o755)

    # Voices Setup
    VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/bn/bn_IN/arijit/medium/bn_IN-arijit-medium.onnx"
    v_path = os.path.join(PATHS["voices"], "bn.onnx")
    if not os.path.exists(v_path):
        print("📥 Downloading Bengali Voice...")
        req = urllib.request.Request(VOICE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(v_path, 'wb') as f: shutil.copyfileobj(r, f)
        req_j = urllib.request.Request(VOICE_URL + ".json", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_j) as r, open(v_path + ".json", 'wb') as f: shutil.copyfileobj(r, f)

# ইনিশিয়ালাইজেশন
setup_all()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Whisper & MADLAD...")
stt_model = WhisperModel("medium", device=DEVICE, compute_type="float16", download_root=PATHS["stt"])
tokenizer = T5Tokenizer.from_pretrained('google/madlad400-3b-mt', cache_dir=PATHS["trans"])
trans_model = T5ForConditionalGeneration.from_pretrained('google/madlad400-3b-mt', cache_dir=PATHS["trans"]).to(DEVICE)
print("✅ ALL SYSTEMS READY")

def handler(job):
    # আপনার আগের হ্যান্ডলার লজিক এখানে থাকবে...
    return {"status": "success", "message": "It works!"}

runpod.serverless.start({"handler": handler})