FROM milvusdb/milvus:v2.6.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-pip \
    && chmod -R a+rX /milvus/bin /milvus/configs /milvus/lib \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/

RUN pip3 install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt
