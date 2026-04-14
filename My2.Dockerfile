FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps 
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git curl wget \
    && apt-get clean

# Python setup
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# --- JAX  ---
RUN pip install "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Prevent JAX from grabbing all GPU memory
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# --- Environments ---
RUN pip install \
    crafter \
    ale_py==0.9.0 \
    autorom[accept-rom-license]==0.6.1

# Dreamer dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# App
WORKDIR /app
COPY . .

ENTRYPOINT ["sh", "entrypoint.sh"]
