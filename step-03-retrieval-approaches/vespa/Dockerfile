FROM vespaengine/vespa:8

ARG TARGETARCH
ARG TIREX_TRACKER_VERSION=0.2.23
ARG TIREX_TRACKER_SHA256=7267bad3e5fb353663e7b03437c794c0e09b1df5bb809a8dfc7baecd7838eed5

USER root

ENV HOME=/tmp
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH

RUN dnf install -y \
        cmake \
        curl \
        gcc-toolset-13-gcc \
        gcc-toolset-13-gcc-c++ \
        git \
        ninja-build \
        python3.11-devel \
    && dnf clean all \
    && python3 -m ensurepip \
    && ln -s /usr/lib64/libbz2.so.1 /usr/lib64/libbz2.so.1.0

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade --no-cache-dir pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN architecture="${TARGETARCH:-$(uname -m)}" \
    && case "$architecture" in \
        amd64 | x86_64) tracker_architecture=x86_64 ;; \
        arm64 | aarch64) tracker_architecture=aarch64 ;; \
        *) echo "Unsupported architecture: $architecture" >&2; exit 1 ;; \
    esac \
    && curl -fsSL \
        "https://github.com/tira-io/tirex-tracker/archive/refs/tags/${TIREX_TRACKER_VERSION}.tar.gz" \
        -o /tmp/tirex-tracker.tar.gz \
    && echo "${TIREX_TRACKER_SHA256}  /tmp/tirex-tracker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/tirex-tracker.tar.gz -C /tmp \
    && cmake \
        -S "/tmp/tirex-tracker-${TIREX_TRACKER_VERSION}/c" \
        -B /tmp/tirex-tracker-build \
        -G Ninja \
        --preset=conf-release-full-shared-lib-static-deps \
    && cmake --build /tmp/tirex-tracker-build --target tirex_tracker -j "$(nproc)" \
    && cp \
        /tmp/tirex-tracker-build/src/libtirex_tracker.so \
        "/usr/local/lib/python3.11/site-packages/tirex_tracker/libtirex_tracker_linux_${tracker_architecture}.so" \
    && rm -rf \
        /tmp/tirex-tracker-build \
        "/tmp/tirex-tracker-${TIREX_TRACKER_VERSION}" \
        /tmp/tirex-tracker.tar.gz \
    && python3 -c "import tirex_tracker"

COPY vespa_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py \
    && chmod a+rx /var/lib/private

USER vespa

WORKDIR /tmp

ENTRYPOINT ["python3", "/index-and-retrieve.py"]
