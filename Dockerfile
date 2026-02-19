FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. eSpeak-NG এবং অডিও টুলস ইন্সটল
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. পাইথন প্যাকেজ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. কোড কপি
COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]