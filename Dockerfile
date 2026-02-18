FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# espeak-ng এখানে থাকা বাধ্যতামূলক
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip espeak-ng && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy

# মডেল ডাউনলোড
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

COPY download_tts.py .
RUN python3 download_tts.py

COPY handler.py .
CMD [ "python3", "-u", "/handler.py" ]