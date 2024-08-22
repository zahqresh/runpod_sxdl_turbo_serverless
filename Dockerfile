# Include Python
FROM python:3.11.1-buster

# Define your working directory
WORKDIR /

# Install all libs from requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install runpod
RUN pip install runpod
RUN pip install torch
RUN pip install uuid
RUN pip install requests
RUN pip install diffusers
RUN pip install accelerate
RUN pip install googletrans==3.1.0a0
RUN pip install numba
RUN pip install ipython
RUN pip install boto3
RUN pip install transformers
RUN pip install transformers[sentencepiece]
RUN pip install -U "huggingface_hub[cli]"
RUN huggingface-cli login --token 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
# Add your file
ADD handler.py .

# Call your file when your container starts
CMD [ "python", "-u", "/handler.py" ]
