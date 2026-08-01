FROM python:3.12-slim AS builder

ARG BMP_COMMIT=c0a17ffc094976da4db2a4e8717e3ce9012c2ce5

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cargo git rustc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY bmp-final-block.patch /tmp/bmp-final-block.patch

RUN git clone https://github.com/pisa-engine/BMP.git /tmp/BMP \
    && cd /tmp/BMP \
    && git checkout "${BMP_COMMIT}" \
    && git apply /tmp/bmp-final-block.patch \
    && pip wheel --no-cache-dir --wheel-dir /wheels /tmp/BMP/python \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

FROM python:3.12-slim

COPY --from=builder /wheels /wheels
COPY --from=builder /tmp/BMP/LICENSE /usr/share/licenses/bmp/LICENSE

RUN pip install --no-cache-dir --no-index /wheels/*.whl \
    && rm -rf /wheels

COPY bmp_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
