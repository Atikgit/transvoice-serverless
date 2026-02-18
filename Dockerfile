# ১. রানপডের অফিশিয়াল ইমেজ (এটি ১০০% ডাউনলোড হবে)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ২. সিস্টেম টুলস ও espeak-ng ইন্সটল (Piper মডেলের জন্য বাধ্যতামূলক)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ৩. পাইথন লাইব্রেরি ইন্সটল
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy

# ৪. SeamlessM4T (Translation) মডেল ডাউনলোড
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# ৫. TTS মডেল ডাউনলোড (আপনার আগের স্ক্রিপ্ট)
COPY download_tts.py .
RUN python3 download_tts.py

# ৬. হ্যান্ডলার সেটআপ
ENV MODEL_PATH=/model_data
ENV TTS_BASE_PATH=/tts_models

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]