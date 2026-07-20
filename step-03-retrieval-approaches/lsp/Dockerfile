# docker build -t mam10eks/lsp:0.0.1 .
FROM ubuntu:24.04

RUN apt-get update \
	&& apt-get install -y python3 python3-pip git curl build-essential \
	&& apt-get clean

# LSP is compiled from source at image build time and needs the Rust nightly
# toolchain (the exact version is pinned by rust-toolchain.toml in the LSP repo).
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain none --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

ADD requirements.txt /tmp/requirements.txt

# On amd64 we target x86-64-v3 (AVX2, portable across modern x86 servers) instead
# of target-cpu=native so the published image runs on any recent x86 host.
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3"; \
    fi \
    && pip3 install -r /tmp/requirements.txt --break-system-packages \
    && rm -Rf /tmp/requirements.txt /root/.cargo/registry /root/.cargo/git \
    && pip3 cache purge

ADD build-and-search-lsp-index.py /index-and-retrieve.py
RUN chmod +x /index-and-retrieve.py
