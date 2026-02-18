FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. প্রয়োজনীয় সিস্টেম টুলস এবং espeak-ng নিশ্চিত করা
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip espeak-ng libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন ডিপেন্ডেন্সি
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy && \
    rm -rf /root/.cache/pip

# ৩. মেইন মডেল ডাউনলোড
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# ৪. TTS মডেল ডাউনলোড (৯টি সফল ভাষা সহ)
COPY download_tts.py .
RUN python3 download_tts.py

ENV MODEL_PATH=/model_data
ENV TTS_PATH=/tts_models

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]