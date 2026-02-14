import runpod
import torch
import torchaudio
import base64
import io
import os
from transformers import SeamlessM4Tv2Model, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", "/model_data")

processor = None
model = None

def load_models():
    global processor, model
    if model is None:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        # FP16 (Half) ব্যবহার করা হয়েছে স্পিড বাড়ানোর জন্য
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE).half()
        print("Optimized FP16 Model Loaded!")

def handler(job):
    load_models()
    job_input = job['input']
    
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng")
    tgt_lang = job_input.get("tgt_lang", "ben")
    voice_tone = job_input.get("voice_tone", "female").lower()
    
    # Speaker ID: এখানে ক্লোনিং বাদ দিয়ে সরাসরি আইডি ব্যবহার করা হচ্ছে
    # ID 6 = ক্লিন পুরুষ ভয়েস, ID 10 = ক্লিন মহিলা ভয়েস
    speaker_id = 6 if "male" in voice_tone else 10

    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
        
        if orig_freq != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq, 16000)
            audio_data = resampler(audio_data)
        
        # ইনপুট ডাটাকেও half precision এ নেওয়া হয়েছে
        audio_inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE).half()
        
        # inference_mode() কোনো গ্রাডিয়েন্ট ক্যালকুলেট করে না, তাই এটি দ্রুততম
        with torch.inference_mode():
            output_tokens = model.generate(
                **audio_inputs, 
                tgt_lang=tgt_lang,
                speaker_id=speaker_id
            )[0]
        
        out_io = io.BytesIO()
        # আউটপুট সেভ করার সময় আবার float এ নিতে হয় অডিও ফাইলের জন্য
        torchaudio.save(out_io, output_tokens.cpu().float(), 16000, format="wav")
        
        return {"audio_out": base64.b64encode(out_io.getvalue()).decode('utf-8')}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})