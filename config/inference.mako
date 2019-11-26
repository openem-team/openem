<%!
import version
%>

# Use nvidia tensorflow image
FROM nvcr.io/nvidia/tensorflow:19.10-py3
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsm6 libxext6 \
    libxrender-dev && \
    rm -fr /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opencv-python==4.1.1.26 

ENV deploy_dir /deploy_dir

COPY deploy_python /openem
WORKDIR /openem
RUN pip3 install .

COPY externals/keras_retinanet /keras_retinanet
WORKDIR /keras_retinanet
RUN pip3 install .

COPY scripts /scripts

WORKDIR /

# Add repo version to image as last step
RUN echo ${version.Git.pretty} > /git_version.txt

