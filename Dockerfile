FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. সিস্টেম টুলস ইনস্টল
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন লাইব্রেরি ইনস্টল
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy && \
    rm -rf /root/.cache/pip

# ৩. SeamlessM4T (Translation) মডেল ডাউনলোড
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# ৪. TTS মডেল ডাউনলোড (নতুন পদ্ধতি)
# প্রথমে স্ক্রিপ্টটি কপি করছি
COPY download_tts.py .
# এরপর স্ক্রিপ্টটি রান করছি (এটি /tts_models ফোল্ডারে সব সেভ করবে)
RUN python3 download_tts.py

ENV MODEL_PATH=/model_data
ENV TTS_PATH=/tts_models

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]