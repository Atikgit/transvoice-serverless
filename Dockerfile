FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

# মডেল আগে থেকে ডাউনলোড করে রাখা (ইমেজ বিল্ড করার সময়)
RUN python3 -c 'from transformers import AutoProcessor, SeamlessM4Tv2Model; AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large"); SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")'

COPY handler.py .

CMD [ "python3", "-u", "/handler.py" ]