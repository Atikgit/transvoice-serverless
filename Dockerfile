FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# সিস্টেমে প্রয়োজনীয় টুলস ইন্সটল করা
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# requirements.txt কপি এবং লাইব্রেরি ইন্সটল করা
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SeamlessM4T v2 Large মডেল প্রি-ডাউনলোড করা
RUN python3 -c "from transformers import AutoProcessor, SeamlessM4Tv2Model; \
    AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large'); \
    SeamlessM4Tv2Model.from_pretrained('facebook/seamless-m4t-v2-large')"

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]