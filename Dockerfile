FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build time
RUN python3 -c "from transformers import AutoProcessor, SeamlessM4Tv2Model; AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large'); SeamlessM4Tv2Model.from_pretrained('facebook/seamless-m4t-v2-large')"

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]