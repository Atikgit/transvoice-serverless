# Base Image (PyTorch + CUDA)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Dependencies (Sherpa-ONNX added)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sherpa-onnx soundfile numpy && \
    rm -rf /root/.cache/pip

# 3. Download SeamlessM4T Model (Translation Engine)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

# 4. Create Directory for TTS Models
RUN mkdir -p /tts_models

# 5. Download Sherpa-ONNX TTS Models (Example: Bengali, Arabic, English, Hindi)
# আপনি এখানে আপনার ১০০+ ভাষার লিংক যোগ করতে পারবেন। আমি প্রধান কয়েকটি দিচ্ছি।
RUN cd /tts_models && \
    # Bengali (VITS)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ben.tar.bz2 && \
    tar -xvf vits-mms-ben.tar.bz2 && rm vits-mms-ben.tar.bz2 && \
    # Arabic (VITS)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-ara.tar.bz2 && \
    tar -xvf vits-mms-ara.tar.bz2 && rm vits-mms-ara.tar.bz2 && \
    # English (US - Amy)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2 && \
    tar -xvf vits-piper-en_US-amy-low.tar.bz2 && rm vits-piper-en_US-amy-low.tar.bz2 && \
    # Hindi (VITS)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-hin.tar.bz2 && \
    tar -xvf vits-mms-hin.tar.bz2 && rm vits-mms-hin.tar.bz2 && \
    # Spanish (VITS)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-spa.tar.bz2 && \
    tar -xvf vits-mms-spa.tar.bz2 && rm vits-mms-spa.tar.bz2

ENV MODEL_PATH=/model_data
ENV TTS_PATH=/tts_models

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]