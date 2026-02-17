FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. সিস্টেম টুলস ও espeak-ng ইন্সটল (এটি মিসিং ছিল!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip espeak-ng && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন লাইব্রেরি ইন্সটল
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy && \
    rm -rf /root/.cache/pip

# ৩. SeamlessM4T (Translation) মডেল ডাউনলোড
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# ৪. TTS মডেল ডাউনলোড (আপনার আগের স্ক্রিপ্টটিই থাকবে)
COPY download_tts.py .
RUN python3 download_tts.py

# ৫. হ্যান্ডলার সেটআপ
ENV MODEL_PATH=/model_data
ENV TTS_PATH=/tts_models

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]