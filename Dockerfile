FROM ubuntu:18.04
ENV PYTHONUNBUFFERED TRUE

## Basic installation of jdk, git, python, curl ....
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
    git \
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

## Torch Serve
RUN cd / && git clone https://github.com/pytorch/serve.git
RUN cd /serve \
    && pip install . \
    && cd model-archiver \
    && pip install .


## Create a user as model-server
RUN useradd -m model-server && mkdir -p /home/model-server/tmp
## Configure entrypoint, change owner of created files to model-server
RUN chown -R model-server /home/model-server
COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

## Preparing expose and configuration for users
USER model-server
WORKDIR /home/model-server
# EXPOSE 2334 2335

CMD ["torchserve", "--start"]

#RUN torchserve --start --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store
#USER root
#ENV TEMP=/home/model-server/tmp
#COPY serve-api.py /
#WORKDIR /
#EXPOSE 23333
#CMD ["gunicorn", "-b", "0.0.0.0:23333", "serve-api"]
