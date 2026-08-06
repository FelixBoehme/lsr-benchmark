FROM qdrant/qdrant:v1.15.5 AS qdrant

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY requirements.txt requirements-dev.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt
