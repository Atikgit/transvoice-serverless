FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SeamlessM4T v2 Large model
RUN python3 -c "from transformers import AutoProcessor, SeamlessM4Tv2Model; \
    AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large'); \
    SeamlessM4Tv2Model.from_pretrained('facebook/seamless-m4t-v2-large')"

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]