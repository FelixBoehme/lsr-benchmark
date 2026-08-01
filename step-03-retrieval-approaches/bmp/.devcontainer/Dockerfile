FROM python:3.12-slim

ARG BMP_COMMIT=c0a17ffc094976da4db2a4e8717e3ce9012c2ce5

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cargo git rustc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/
COPY bmp-final-block.patch /tmp/bmp-final-block.patch

RUN git clone https://github.com/pisa-engine/BMP.git /tmp/BMP \
    && cd /tmp/BMP \
    && git checkout "${BMP_COMMIT}" \
    && git apply /tmp/bmp-final-block.patch \
    && pip install --no-cache-dir /tmp/BMP/python \
    && pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt \
    && rm -rf /tmp/BMP /tmp/bmp-final-block.patch /root/.cargo /root/.cache
