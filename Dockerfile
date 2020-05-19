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
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && chown -R model-server /home/model-server
COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

## Preparing expose and configuration for users

USER model-server
WORKDIR /home/model-server

## Configuring model
RUN mkdir /home/model-server/kernels && chown -R model-server /home/model-server/kernels
COPY lang_model.pt /home/model-server/kernels/lang_model.pt
COPY lang_handler.py /home/model-server/kernels/lang_handler.py
COPY lang_clf.py /home/model-server/kernels/lang_clf.py

RUN torch-model-archiver --model-name ELDALangClf --version 1.0 \ 
    --model-file /home/model-server/kernels/lang_clf.py \
    --serialized-file /home/model-server/kernels/lang_model.pt \
    --handler /home/model-server/kernels/lang_handler.py \
    && mv ELDALangClf.mar /home/model-server/model-store/

EXPOSE 2334 2335

ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]

## Use them as sandbox for testing
CMD ["torchserve", "--start", "--ts-config", "/home/model-server/config.properties", "--model-store", "/home/model-server/model-store", "--models", "ELDALangClf=ELDALangClf.mar"]
