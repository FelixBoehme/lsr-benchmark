FROM ubuntu:22.04

RUN apt-get update \
	&& apt-get install -y python3 python3-pip git \
	&& pip3 install numpy \
	&& apt-get clean \
	&& pip3 cache purge

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt \
	&& rm -Rf /tmp/requirements.txt \
	&& pip3 cache purge

ADD numpy_exhaustive_search.py /index-and-retrieve.py
