FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime

WORKDIR /

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('facebook/seamless-m4t-v2-large', local_dir='/model_data', local_dir_use_symlinks=False)"

ENV MODEL_PATH=/model_data
ENV TRANSFORMERS_CACHE=/model_data

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]