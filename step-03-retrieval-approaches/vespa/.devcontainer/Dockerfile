FROM ubuntu:20.04 AS cpp-runtime

RUN cp "$(find /usr/lib -name 'libstdc++.so.6.*' -print -quit)" /libstdc++.so.6

FROM vespaengine/vespa:8

USER root

COPY --from=cpp-runtime /libstdc++.so.6 /opt/compat-lib/libstdc++.so.6

ENV HOME=/tmp
ENV LD_LIBRARY_PATH=/opt/compat-lib

RUN dnf install -y git \
    && dnf clean all \
    && python3 -m ensurepip

COPY requirements.txt requirements-dev.txt /tmp/

RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt

USER vespa

ENTRYPOINT []
