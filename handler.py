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
        print(f"Loading models from {MODEL_PATH}...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = SeamlessM4Tv2Model.from_pretrained(MODEL_PATH).to(DEVICE)
        print("Models loaded successfully!")

def handler(job):
    load_models()
    job_input = job['input']
    audio_b64 = job_input.get("audio")
    src_lang = job_input.get("src_lang", "eng")
    tgt_lang = job_input.get("tgt_lang", "ben")
    
    audio_bytes = base64.b64decode(audio_b64)
    audio_data, orig_freq = torchaudio.load(io.BytesIO(audio_bytes))
    
    if orig_freq != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq, 16000)
        audio_data = resampler(audio_data)
    
    audio_inputs = processor(audios=audio_data, src_lang=src_lang, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0]
    
    out_io = io.BytesIO()
    torchaudio.save(out_io, output_tokens.cpu(), 16000, format="wav")
    return {"audio_out": base64.b64encode(out_io.getvalue()).decode('utf-8')}

runpod.serverless.start({"handler": handler})