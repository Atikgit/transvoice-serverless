FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. সিস্টেম টুলস (espeak-ng ছাড়া Piper চলবে না)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন লাইব্রেরি
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ৩. পাইথন দিয়ে সব ডাউনলোড করা হবে (এতে ব্লক খাবে না)
COPY download_models.py .
RUN python3 download_models.py

# ৪. এনভায়রনমেন্ট ভেরিয়েবল সেটআপ
ENV MODEL_STT=/model_stt
ENV MODEL_TRANS=/model_trans
ENV PIPER_BINARY=/piper_bin/piper
ENV VOICE_DIR=/piper_voices
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ৫. হ্যান্ডলার
COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]