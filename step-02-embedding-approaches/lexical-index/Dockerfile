FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN curl -I https://github.com/reneuir/lsr-benchmark.git || true

RUN pip3 install -r /tmp/requirements.txt

RUN python3 -c "import pyterrier as pt; pt.init();"

ADD run-pyterrier.py /run-pyterrier.py

