FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model only (Avoids OOM error during build)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/seamless-m4t-v2-large')"

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]