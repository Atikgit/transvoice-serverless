FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# ১. সিস্টেম টুলস (শুধু ইন্সটল, কোনো ডাউনলোড নেই)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ২. পাইথন লাইব্রেরি
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ৩. শুধু হ্যান্ডলার কপি হবে (মডেল রানটাইমে নামবে)
COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]