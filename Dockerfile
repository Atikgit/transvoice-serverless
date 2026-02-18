# 1. Base Image: Official Sherpa-ONNX (CUDA support included)
# এটিতে espeak-ng এবং সব পাথ আগে থেকেই সেট করা আছে
FROM ghcr.io/k2-fsa/sherpa-onnx:cuda

WORKDIR /

# 2. System dependencies for SeamlessM4T and Audio processing
# git, wget, ffmpeg এগুলো লাগবে
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    ffmpeg \
    libsndfile1 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Python Dependencies for SeamlessM4T & RunPod
# Sherpa-ONNX এর পাইথন প্যাকেজ এই ইমেজে আগে থেকেই থাকে, তাই শুধু বাকিগুলো লাগবে
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Download SeamlessM4T Model (Translation)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# 5. Download TTS Models (Sherpa-ONNX)
# আপনার আগের download_tts.py ফাইলটিই এখানে ব্যবহার করুন
COPY download_tts.py .
RUN python3 download_tts.py

# 6. Setup Handler
ENV MODEL_PATH=/model_data
ENV TTS_BASE_PATH=/tts_models
# Sherpa-ONNX এর বাইনারি পাথ এনভায়রনমেন্টে যোগ করা
ENV PATH="/usr/local/bin:${PATH}"

COPY handler.py .

# 7. Start Command
CMD [ "python3", "-u", "/handler.py" ]