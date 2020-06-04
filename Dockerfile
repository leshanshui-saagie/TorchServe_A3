FROM ubuntu:18.04

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    python3-dev \
    openjdk-11-jdk \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

## Flask utils
RUN pip install Flask flask-restplus Flask-SSLify Flask-Admin gunicorn hdfs
## Torch utils
RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir torchvision
RUN pip install --no-cache-dir torchtext
RUN pip install --no-cache-dir transformers

USER root
ENV TEMP=/home/model-server/tmp
COPY serve-api.py /
WORKDIR /
EXPOSE 23333
CMD ["gunicorn", "-b", "0.0.0.0:23333", "serve-api"]
