FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model using HuggingFace CLI (More stable for 10GB+ files)
RUN huggingface-cli download facebook/seamless-m4t-v2-large --local-dir /container_cache/seamless-m4t-v2-large --local-dir-use-symlinks False

# Set environment variable for the model path
ENV MODEL_PATH=/container_cache/seamless-m4t-v2-large

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]