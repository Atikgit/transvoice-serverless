import runpod
import torch
import torchaudio
import base64
import io
import os
import glob
import sherpa_onnx
import soundfile as sf
from transformers import SeamlessM4Tv2Model, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/model_data"
TTS_BASE_PATH = "/tts_models"

processor = None
model = None
tts_engines = {}

# ভাষা কোড ম্যাপিং (RunPod থেকে আসা ৩-অক্ষর -> মডেলের ২-অক্ষর)
ISO_MAP = {
    'ben': 'bn', 'hin': 'hi', 'ara': 'ar', 'urd': 'ur', 
    'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it',
    'jpn': 'ja', 'kor': 'ko', 'vie': 'vi', 'ind': 'id',
    'tur': 'tr', 'eng': 'en'
}

def load_seamless():
    global processor, model
    if model is None:
        print("--- Loading SeamlessM4T v2 Large ---")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("--- SeamlessM4T Loaded Successfully ---")

def get_tts_engine(lang_code):
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    print(f"DEBUG: Searching TTS for [{lang_code}]")
    
    # ফোল্ডার খোঁজার চেষ্টা (Piper বা MMS যাই হোক)
    search_codes = [lang_code]
    if lang_code in ISO_MAP:
        search_codes.append(ISO_MAP[lang_code])
    
    folder_path = None
    for code in search_codes:
        pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{code}*")
        found_folders = glob.glob(pattern)
        if found_folders:
            folder_path = found_folders[0]
            break

    if not folder_path or not os.path.exists(folder_path):
        print(f"ERROR: No model folder found for code {lang_code} in {TTS_BASE_PATH}")
        return None

    print(f"DEBUG: Found folder -> {folder_path}")

    # ফোল্ডারের ভেতর ফাইলগুলো খুঁজে বের করা
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    tokens_file = os.path.join(folder_path, "tokens.txt")
    data_dir = os.path.join(folder_path, "espeak-ng-data")

    if not onnx_files or not os.path.exists(tokens_file):
        print(f"ERROR: Missing .onnx or tokens.txt in {folder_path}")
        print(f"Folder Content: {os.listdir(folder_path)}")
        return None

    try:
        # 'argument ids' এরর ঠেকাতে সরাসরি কনফিগ সেট করা
        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=onnx_files[0],
            tokens=tokens_file,
            data_dir=data_dir if os.path.exists(data_dir) else "",
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits_config),
            max_num_sentences=1
        )
        
        engine = sherpa_onnx.OfflineTts(tts_config)
        tts_engines[lang_code] = engine
        print(f"SUCCESS: Loaded TTS engine for {lang_code}")
        return engine
    except Exception as e:
        print(f"CRITICAL ERROR loading engine: {str(e)}")
        return None

def handler(job):
    load_seamless()
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng").lower()
    tgt_lang = job_input.get("tgt_lang", "ben").lower()

    if not audio_b64:
        return {"error": "No audio provided"}

    try:
        # ১. অডিও ডিকোড ও রিস্যাম্পল
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        if orig_freq != 16000:
            audio_data = torchaudio.transforms.Resample(orig_freq, 16000)(audio_data)

        # ২. টেক্সট ট্রান্সলেশন
        inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text = processor.decode(tokens[0].tolist(), skip_special_tokens=True)
        print(f"DEBUG: Translation Result -> {translated_text}")

        # ৩. ভয়েস জেনারেশন
        engine = get_tts_engine(tgt_lang)
        if engine:
            audio = engine.generate(translated_text, sid=0, speed=1.0)
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            audio_out = base64.b64encode(out_io.getvalue()).decode('utf-8')
            return {"audio_out": audio_out, "text_out": translated_text}
        
        return {"error": f"TTS engine failed for {tgt_lang}", "text_out": translated_text}

    except Exception as e:
        print(f"HANDLER CRASH: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})