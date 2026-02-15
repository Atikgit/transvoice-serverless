import runpod
import torch
import torchaudio
import base64
import io
import os
import sherpa_onnx
import soundfile as sf
import numpy as np
from transformers import SeamlessM4Tv2Model, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", "/model_data")
TTS_BASE_PATH = os.environ.get("TTS_PATH", "/tts_models")

processor = None
model = None
tts_engines = {} # Loaded TTS models cache

# --- Language Map to Model Path ---
# Seamless Code -> Sherpa Model Folder
TTS_MODEL_MAP = {
    'ben': 'vits-mms-ben',
    'arb': 'vits-mms-ara', # Seamless 'arb' -> Sherpa 'ara'
    'eng': 'vits-piper-en_US-amy-low',
    'hin': 'vits-mms-hin',
    'spa': 'vits-mms-spa',
    # এখানে আরও ভাষা যোগ করতে হবে...
}

def load_seamless():
    global processor, model
    if model is None:
        print("Loading SeamlessM4T (Translation Engine)...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("SeamlessM4T Loaded!")

def get_tts_engine(lang_code):
    """Load specific Sherpa-ONNX model for the language on demand"""
    if lang_code not in TTS_MODEL_MAP:
        print(f"TTS Model not found for {lang_code}, falling back to English")
        lang_code = 'eng'
    
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    model_folder = TTS_MODEL_MAP[lang_code]
    base = os.path.join(TTS_BASE_PATH, model_folder)
    
    print(f"Loading TTS Model for {lang_code} from {base}...")
    
    # Sherpa Config setup
    try:
        # Check if it's a VITS-MMS model
        if "mms" in model_folder:
            config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{base}/model.onnx",
                tokens=f"{base}/tokens.txt",
                data_dir=f"{base}/espeak-ng-data"
            )
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(vits=config)
            )
        # Check if it's a Piper model (Better quality usually)
        elif "piper" in model_folder:
            config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{base}/en_US-amy-low.onnx", # Name varies, adjust per model
                tokens=f"{base}/tokens.txt",
                data_dir=f"{base}/espeak-ng-data"
            )
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(vits=config)
            )
        else:
             # Generic Fallback
             return None

        app = sherpa_onnx.OfflineTts(tts_config)
        tts_engines[lang_code] = app
        return app
    except Exception as e:
        print(f"Failed to load TTS for {lang_code}: {e}")
        return None

def handler(job):
    load_seamless()
    
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng")
    tgt_lang = job_input.get("tgt_lang", "ben")

    try:
        # 1. Decode Input Audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        if orig_freq != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq, 16000)
            audio_data = resampler(audio_data)

        # 2. STT + Translation (Text Generation ONLY)
        # আমরা Seamless কে বলছি অডিও না বানাতে, শুধু টেক্সট দিতে
        audio_inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs, 
                tgt_lang=tgt_lang,
                generate_speech=False # IMPORTANT: শুধু টেক্সট চাই
            )
            
        translated_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        print(f"Translated Text ({tgt_lang}): {translated_text}")

        # 3. TTS Generation (Sherpa-ONNX)
        tts_app = get_tts_engine(tgt_lang)
        
        if tts_app:
            # Generate Audio
            audio = tts_app.generate(translated_text, sid=0, speed=1.1)
            
            # Sherpa output is raw float32, need to convert to wav bytes
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            out_b64 = base64.b64encode(out_io.getvalue()).decode('utf-8')
            
            return {"audio_out": out_b64, "text_out": translated_text}
        else:
            return {"error": f"TTS model missing for {tgt_lang}"}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})