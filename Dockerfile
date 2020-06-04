FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*     
ENV PYTHONUNBUFFERED TRUE
# FOR DEBUGGING
CMD ["ls", "-a"]
