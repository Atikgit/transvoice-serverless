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

def load_seamless():
    global processor, model
    if model is None:
        print("Loading SeamlessM4T Engine...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)

def get_tts_engine(lang_code):
    """অটোমেটিক ফোল্ডার এবং অনক্স ফাইল ডিটেকশন লজিক"""
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    # ফোল্ডার খুঁজে বের করা (যেমন: vits-piper-es-* বা vits-mms-ben-*)
    search_pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{lang_code}*")
    folders = glob.glob(search_pattern)
    
    if not folders:
        # বিকল্প কোড চেক (যেমন: ben -> bn)
        alt_code = "bn" if lang_code == "ben" else "ar" if lang_code == "ara" else lang_code
        search_pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{alt_code}*")
        folders = glob.glob(search_pattern)

    if not folders:
        print(f"❌ No TTS folder found for: {lang_code}")
        return None

    folder_path = folders[0]
    # ফোল্ডারের ভেতর প্রথম .onnx ফাইলটি খুঁজে নেওয়া
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    if not onnx_files:
        return None

    try:
        config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=onnx_files[0],
            tokens=os.path.join(folder_path, "tokens.txt"),
            data_dir=os.path.join(folder_path, "espeak-ng-data")
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=config)
        )
        engine = sherpa_onnx.OfflineTts(tts_config)
        tts_engines[lang_code] = engine
        return engine
    except Exception as e:
        print(f"Error loading {lang_code}: {e}")
        return None

def handler(job):
    load_seamless()
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng")
    tgt_lang = job_input.get("tgt_lang", "ben")

    try:
        # 1. Decode & Resample
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        if orig_freq != 16000:
            audio_data = torchaudio.transforms.Resample(orig_freq, 16000)(audio_data)

        # 2. Translation (Text Output)
        inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text = processor.decode(tokens[0].tolist(), skip_special_tokens=True)

        # 3. TTS (Sherpa-ONNX)
        engine = get_tts_engine(tgt_lang)
        if engine:
            audio = engine.generate(translated_text, sid=0, speed=1.1)
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            return {"audio_out": base64.b64encode(out_io.getvalue()).decode('utf-8'), "text_out": translated_text}
        
        return {"error": f"TTS model not found for {tgt_lang}", "text_out": translated_text}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})