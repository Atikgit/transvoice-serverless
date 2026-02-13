import runpod
import torch
import torchaudio
import base64
import io
import os
from transformers import SeamlessM4Tv2Model, AutoProcessor

# 1. Device & Model Path Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", "/model_data")

processor = None
model = None

def load_models():
    global processor, model
    if model is None:
        print(f"Loading models from {MODEL_PATH}...")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback if local path fails (Safety net)
            processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(DEVICE)

def handler(job):
    # Ensure models are loaded
    load_models()
    
    job_input = job['input']
    
    # 2. Input Parsing
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng")
    tgt_lang = job_input.get("tgt_lang", "ben")
    
    # হাইব্রিড স্পিকার লজিক (Male/Female Fix)
    voice_tone = job_input.get("voice_tone", "female").lower()
    
    # [UPDATED ID MAPPING]
    # ID 0 = সাধারণত ভারী/পুরুষালি ভয়েস
    # ID 1 = সাধারণত চিকন/মেয়েলি ভয়েস (SeamlessM4T v2 Preset)
    if "male" in voice_tone:
        speaker_id = 0  
    else:
        speaker_id = 1  # 4 এর বদলে 1 ব্যবহার করা হলো

    # 4. Audio Processing
    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Resample to 16kHz (Required by model)
        if orig_freq != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq, 16000)
            audio_data = resampler(audio_data)
        
        # Prepare inputs
        audio_inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
        
        # 5. Generate Audio (With Speaker ID Force)
        # speaker_id পাস করার ফলে এটি ইনপুট অডিওর নয়েজ বা টোন কপি করবে না
        # এতে 'Robotic Voice' সমস্যা দূর হবে এবং স্পিড বাড়বে
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs, 
                tgt_lang=tgt_lang,
                speaker_id=speaker_id 
            )[0]
        
        # 6. Output Conversion
        out_io = io.BytesIO()
        torchaudio.save(out_io, output_tokens.cpu(), 16000, format="wav")
        out_b64 = base64.b64encode(out_io.getvalue()).decode('utf-8')
        
        return {"audio_out": out_b64}

    except Exception as e:
        print(f"Processing Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})