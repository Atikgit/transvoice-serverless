# Base Image (CUDA supported)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. System Dependencies
# Piper এর জন্য espeak-ng এবং অডিওর জন্য ffmpeg বাধ্যতামূলক
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip ffmpeg espeak-ng libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Setup Piper TTS (Linux Binary)
# এটি পাইথন প্যাকেজ নয়, সরাসরি সফটওয়্যার, তাই কোনো এরর দেবে না
RUN wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz && \
    tar -xf piper.tar.gz && \
    rm piper.tar.gz && \
    mv piper /usr/local/bin/piper_bin

# 4. Download ALL AI Models (STT, Translate, Voices)
COPY download_models.py .
RUN python3 download_models.py

# 5. Environment Variables
ENV MODEL_STT=/model_stt
ENV MODEL_TRANS=/model_trans
ENV PIPER_BINARY=/usr/local/bin/piper_bin/piper
ENV VOICE_DIR=/piper_voices

# 6. Copy Handler
COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]