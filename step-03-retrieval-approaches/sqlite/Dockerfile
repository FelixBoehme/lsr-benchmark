FROM ubuntu:jammy

RUN apt-get update \
    && apt-get install -y git python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt \
	&& rm -Rf /tmp/requirements.txt \
	&& pip3 cache purge

ADD run-sqlite.py /index-and-retrieve.py
ADD sqlite_retrieval.py /sqlite_retrieval.py
