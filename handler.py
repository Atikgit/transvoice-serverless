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

# গ্লোবাল ভ্যারিয়েবল
processor = None
model = None
tts_engines = {}

def load_models():
    global processor, model
    if model is None:
        print("--- LOADING SEAMLESSM4T V2 ENGINE ---")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("--- SEAMLESSM4T LOADED ---")

def get_tts_engine(lang_code):
    if lang_code in tts_engines:
        return tts_engines[lang_code]

    print(f"DEBUG: Finding models for [{lang_code}]")
    
    # ৩-অক্ষর থেকে ২-অক্ষর ম্যাপিং (spa -> es, deu -> de)
    iso_map = {'spa': 'es', 'deu': 'de', 'ben': 'bn', 'hin': 'hi', 'ara': 'ar', 'rus': 'ru', 'eng': 'en'}
    search_codes = [lang_code, iso_map.get(lang_code, lang_code)]
    
    folder_path = None
    for code in search_codes:
        pattern = os.path.join(TTS_BASE_PATH, f"vits-*-{code}*")
        found = glob.glob(pattern)
        if found:
            folder_path = found[0]
            break

    if not folder_path:
        print(f"❌ ERROR: No TTS folder found for '{lang_code}' in {TTS_BASE_PATH}")
        return None

    # ফাইলের অস্তিত্ব নিবিড়ভাবে চেক করা
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    tokens_file = os.path.join(folder_path, "tokens.txt")
    data_dir = os.path.join(folder_path, "espeak-ng-data")

    if not onnx_files or not os.path.exists(tokens_file):
        print(f"❌ ERROR: Missing .onnx or tokens.txt in {folder_path}")
        return None

    try:
        # 'argument ids' এরর এড়াতে নির্দিষ্টভাবে প্যারামিটার পাস করা
        print(f"🛠️ Initializing engine: {os.path.basename(onnx_files[0])}")
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
        print(f"✅ SUCCESS: {lang_code} Engine is Ready!")
        return engine
    except Exception as e:
        print(f"🔥 TTS INITIALIZATION FAILED: {str(e)}")
        return None

def handler(job):
    load_models()
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng").lower()
    tgt_lang = job_input.get("tgt_lang", "ben").lower()

    try:
        # ১. অডিও ডিকোড ও রিস্যাম্পল
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, samplerate = torchaudio.load(io.BytesIO(audio_bytes))
        if samplerate != 16000:
            audio_data = torchaudio.transforms.Resample(samplerate, 16000)(audio_data)

        # ২. অনুবাদ (SeamlessM4T)
        inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        print(f"DEBUG: Translated Text -> {translated_text}")

        # ৩. ভয়েস (Sherpa-ONNX)
        engine = get_tts_engine(tgt_lang)
        if engine:
            audio = engine.generate(translated_text, sid=0, speed=1.1)
            out_io = io.BytesIO()
            sf.write(out_io, audio.samples, audio.sample_rate, format='wav')
            return {
                "audio_out": base64.b64encode(out_io.getvalue()).decode('utf-8'),
                "text_out": translated_text
            }
        
        return {"error": f"TTS not available for {tgt_lang}", "text_out": translated_text}

    except Exception as e:
        print(f"🔥 HANDLER ERROR: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})