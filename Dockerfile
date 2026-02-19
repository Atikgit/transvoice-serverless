FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. System Dependencies (espeak-ng is MUST)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Download EVERYTHING (Binary + Models) using Python
# এটি wget এর মতো ব্লক খাবে না
COPY download_models.py .
RUN python3 download_models.py

# 4. Environment Variables
ENV MODEL_STT=/model_stt
ENV MODEL_TRANS=/model_trans
ENV PIPER_BINARY=/piper_bin/piper
ENV VOICE_DIR=/piper_voices
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 5. Handler
COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]