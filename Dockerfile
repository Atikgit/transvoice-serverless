FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. সিস্টেম লাইব্রেরি ইন্সটল
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন প্যাকেজ ইন্সটল
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ৩. শুধু কোড কপি করা (ডাউনলোড এখন হবে না)
COPY handler.py .

# ৪. সার্ভার শুরু হওয়ার কমান্ড
CMD [ "python3", "-u", "/handler.py" ]