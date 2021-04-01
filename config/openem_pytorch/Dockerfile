# Use nvidia pytorch image
FROM nvcr.io/nvidia/pytorch:21.03-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
      apt-get install -y --no-install-recommends libxrender-dev python3-pip \
      software-properties-common libboost-python-dev
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN rm -fr /var/lib/apt/lists/*
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
RUN apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io && \
      rm -fr /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir opencv-python>=4.5 tator>=0.10.4  docker>=4.2.0 tables>=3.6.1 \
      boto3>=1.16.63 torch>=1.7.1 torchvision>=0.8.2
run mkdir -p /root/.cache/torch/hub/checkpoints
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
      -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth

COPY scripts /scripts
COPY train /train

WORKDIR /
