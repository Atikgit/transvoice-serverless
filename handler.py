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

# কনফিগারেশন
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/model_data"
TTS_BASE_PATH = "/tts_models"

print(f"--- Using Sherpa-ONNX Version: {sherpa_onnx.__version__} ---")

processor = None
model = None
tts_engines = {}

def load_models():
    global processor, model
    if model is None:
        print("Loading SeamlessM4T...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("SeamlessM4T Loaded.")

def get_tts_engine(lang_code):
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    print(f"Finding TTS for: {lang_code}")
    
    # ভাষা ম্যাপিং (Official Image স্টান্ডার্ড)
    iso_map = {'spa': 'es', 'deu': 'de', 'ben': 'bn', 'hin': 'hi', 'ara': 'ar', 'rus': 'ru', 'eng': 'en'}
    search_code = iso_map.get(lang_code, lang_code)
    
    # ফোল্ডার খোঁজা
    pattern = os.path.join(TTS_BASE_PATH, f"*{search_code}*")
    folders = glob.glob(pattern)
    if not folders:
        print(f"❌ No folder found for {lang_code}")
        return None
    
    target_folder = folders[0]
    onnx_file = glob.glob(os.path.join(target_folder, "*.onnx"))[0]
    tokens_file = os.path.join(target_folder, "tokens.txt")
    
    # Piper মডেলের জন্য data_dir খুবই গুরুত্বপূর্ণ
    data_dir = os.path.join(target_folder, "espeak-ng-data")
    
    try:
        # অফিশিয়াল ইমেজে এই কনফিগারেশনটি কাজ করবেই
        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=onnx_file,
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
        print(f"✅ Engine Ready: {lang_code}")
        return engine
    except Exception as e:
        print(f"🔥 Error loading {lang_code}: {e}")
        return None

def handler(job):
    load_models()
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    tgt_lang = job_input.get("tgt_lang", "ben").lower()
    src_lang = job_input.get("src_lang", "eng").lower()

    if not audio_b64: return {"error": "No audio"}

    try:
        # 1. Translation
        audio_bytes = base64.b64decode(audio_b64)
        audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != 16000:
            audio_tensor = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)

        inputs = processor(audios=audio_tensor, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        text = processor.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"Text: {text}")

        # 2. TTS Generation
        engine = get_tts_engine(tgt_lang)
        if engine:
            audio = engine.generate(text, sid=0, speed=1.0)
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            return {
                "audio_out": base64.b64encode(out_io.getvalue()).decode('utf-8'),
                "text_out": text
            }
        else:
            return {"text_out": text, "error": "TTS Failed"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})