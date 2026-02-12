FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel

WORKDIR /

# সিস্টেমে প্রয়োজনীয় কিছু টুলস ইন্সটল করা
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# মডেল ডাউনলোড করার সময় এরর এড়াতে ক্যাশ ডিরেক্টরি সেট করা
ENV TRANSFORMERS_CACHE=/container_cache
RUN mkdir -p /container_cache

# মডেল প্রি-ডাউনলোড (বিল্ড টাইমে)
RUN python3 -c "from transformers import AutoProcessor, SeamlessM4Tv2Model; AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large'); SeamlessM4Tv2Model.from_pretrained('facebook/seamless-m4t-v2-large')"

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]