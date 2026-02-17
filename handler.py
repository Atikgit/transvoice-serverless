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

# 1. ভাষা কোড ম্যাপিং (RunPod 3-letter -> Model 2-letter)
ISO_MAP = {
    'ben': 'bn', 'hin': 'hi', 'ara': 'ar', 'urd': 'ur', 
    'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it',
    'jpn': 'ja', 'kor': 'ko', 'vie': 'vi', 'ind': 'id',
    'tur': 'tr', 'eng': 'en'
}

def load_seamless():
    global processor, model
    if model is None:
        print("Loading SeamlessM4T Engine...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("SeamlessM4T Loaded.")

def get_tts_engine(lang_code):
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    print(f"🔍 Searching TTS for: {lang_code}...")

    # ১. প্রথমে সরাসরি ৩ অক্ষরের কোড দিয়ে ফোল্ডার খোঁজা
    search_pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{lang_code}*")
    folders = glob.glob(search_pattern)

    # ২. না পেলে ২ অক্ষরের কোড দিয়ে খোঁজা (যেমন spa -> es)
    if not folders and lang_code in ISO_MAP:
        short_code = ISO_MAP[lang_code]
        search_pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{short_code}*")
        folders = glob.glob(search_pattern)

    if not folders:
        print(f"❌ No TTS folder found for {lang_code}")
        return None

    folder_path = folders[0] # প্রথম ফোল্ডারটি নেওয়া
    print(f"📂 Found Model Folder: {folder_path}")

    # ৩. ফোল্ডারের ভিতর .onnx ফাইল খোঁজা
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    if not onnx_files:
        print("❌ No .onnx file found inside folder.")
        return None
    
    try:
        model_file = onnx_files[0]
        tokens_file = os.path.join(folder_path, "tokens.txt")
        data_dir = os.path.join(folder_path, "espeak-ng-data")
        
        config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model_file,
            tokens=tokens_file,
            data_dir=data_dir if os.path.exists(data_dir) else ""
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=config)
        )
        engine = sherpa_onnx.OfflineTts(tts_config)
        tts_engines[lang_code] = engine
        print(f"✅ Loaded Engine for {lang_code}")
        return engine
    except Exception as e:
        print(f"❌ Error loading engine: {e}")
        return None

def handler(job):
    load_seamless()
    job_input = job['input']
    
    audio_b64 = job_input.get("audio")
    # ক্লায়েন্ট থেকে ভাষা কোড ছোট হাতের অক্ষরে নেওয়া
    src_lang = job_input.get("src_lang", "eng").lower() 
    tgt_lang = job_input.get("tgt_lang", "ben").lower()

    if not audio_b64:
        return {"error": "No audio provided"}

    try:
        # A. Audio Decoding
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        if orig_freq != 16000:
            audio_data = torchaudio.transforms.Resample(orig_freq, 16000)(audio_data)

        # B. Translation (Text)
        # Seamless মডেল সাধারণত 3 অক্ষরের কোড সাপোর্ট করে (যেমন ben, spa)
        inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text = processor.decode(tokens[0].tolist(), skip_special_tokens=True)
        
        print(f"📝 Translated ({src_lang}->{tgt_lang}): {translated_text}")

        # C. TTS Generation
        engine = get_tts_engine(tgt_lang)
        audio_out_b64 = None
        
        if engine:
            # স্পিড ১.১ রাখা হয়েছে ন্যাচারাল সাউন্ডের জন্য
            audio = engine.generate(translated_text, sid=0, speed=1.1)
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            audio_out_b64 = base64.b64encode(out_io.getvalue()).decode('utf-8')
        else:
            print(f"⚠️ Skipping TTS: Model not found for {tgt_lang}")

        return {
            "text_out": translated_text,
            "audio_out": audio_out_b64,
            "status": "success"
        }

    except Exception as e:
        print(f"🔥 Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})