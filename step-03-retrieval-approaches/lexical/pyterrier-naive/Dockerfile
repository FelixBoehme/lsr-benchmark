FROM ubuntu:jammy

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip \
  && pip3 install -r /tmp/requirements.txt \
  && python3 -c "import pyterrier as pt; pt.init();"

COPY run-pyterrier.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
