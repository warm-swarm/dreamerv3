FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04

RUN echo "Building the image, stay tuned..."

RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb wget \
  && apt-get clean

# Prepare python
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

# env
RUN pip install jax[cuda]==0.5.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install crafter

# Source
RUN mkdir /app
WORKDIR /app
COPY . .

ENTRYPOINT ["sh", "entrypoint.sh"]
