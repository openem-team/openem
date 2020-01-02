<%!
import version
import multiArch
%>

# Use nvidia tensorflow image
% if multiArch.arch == "x86_64":
FROM nvcr.io/nvidia/tensorflow:19.11-tf1-py3
% elif multiArch.arch == "aarch64":
FROM nvcr.io/nvidia/l4t-base:r32.3.1
% else:
# No go if not on aarch64 or x86_64..
% endif

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsm6 libxext6 \
    libxrender-dev python3-pip && \
    rm -fr /var/lib/apt/lists/*

% if multiArch.arch == "x86_64":
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir opencv-python==4.1.1.26 scikit-image==0.14.2
% elif multiArch.arch == "aarch64":
# ARM64 needs to have tensorflow installed
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
# Release notes for version info to tie x86 container to jetpack release:
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel

# Added 'build-essential', 'python3-devel' for container compatibility
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
    build-essential libpython3-dev && \
    rm -fr /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -U pip testresources setuptools
# Jetson Jetpack already has xavier-optimized opencv2.

# Install pre-built wheels
# These take awhile to build, so its best to do this outside of docker
WORKDIR /packages
COPY arm_packages/numpy-1.16.1-cp36-cp36m-linux_aarch64.whl numpy-1.16.1-cp36-cp36m-linux_aarch64.whl
COPY arm_packages/scikit-image-0.14.2-cp36-cp36m-linux_aarch64.whl scikit-image-0.14.2-cp36-cp36m-linux_aarch64.whl
RUN pip3 install --no-cache-dir numpy-1.16.1-cp36-cp36m-linux_aarch64.whl
RUN pip3 install --no-cache-dir scikit-image-0.14.2-cp36-cp36m-linux_aarch64.whl
WORKDIR /
RUN rm -fr /packages

RUN pip3 install --no-cache-dir -U \
         future==0.17.1 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.0.5\
	 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf
# Install latest tensorflow 1.XX
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu<2
% endif

COPY deploy_python /openem
WORKDIR /openem
RUN pip3 install --no-cache-dir .

COPY externals/keras_retinanet /keras_retinanet
WORKDIR /keras_retinanet
RUN pip3 install --no-cache-dir .

COPY scripts /scripts
COPY train /train

WORKDIR /

# Add repo version to image as last step
RUN echo ${version.Git.pretty} > /git_version.txt
