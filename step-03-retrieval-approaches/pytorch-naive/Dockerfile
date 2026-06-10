FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base-amd64

FROM python:3.11-slim AS base-arm64

RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/*

FROM base-${TARGETARCH}

ADD requirements.txt /tmp/

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/* \
	&& python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

ADD search-pytorch-index.py /index-and-retrieve.py
