# Taken from the docker container of seismic. it does not seems to require any change. 
# docker build -t mam10eks/kannolo:0.0.1 .
FROM ubuntu:24.04

RUN apt-get update \
	&& apt-get install -y python3 python3-pip git \
	&& apt-get clean

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt --break-system-packages \
    && rm -Rf /tmp/requirements.txt \
    && pip3 cache purge

ADD build-and-search-kannolo-index.py /index-and-retrieve.py
