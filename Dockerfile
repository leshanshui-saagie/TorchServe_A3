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
RUN pip install --no-cache-dir psutil torch torchvision torchtext transformers
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
# USER model-server
# WORKDIR /home/model-server
USER root
ENV TEMP=/home/model-server/tmp
COPY serve-api.py /
WORKDIR /

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
EXPOSE 8079 8080 8081
CMD ["serve"]
