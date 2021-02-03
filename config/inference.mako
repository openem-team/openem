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
    libxrender-dev python3-pip software-properties-common \
    libboost-python-dev && \
    rm -fr /var/lib/apt/lists/*

% if multiArch.arch == "x86_64":

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
RUN apt-get install -y --no-install-recommends docker-ce \
            docker-ce-cli containerd.io && rm -fr /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN echo "force 0.1.4"
RUN pip3 install --no-cache-dir opencv-python==4.1.1.26 scikit-image==0.14.2 tator>=0.9.2 \
      pytator>=0.1.4 docker>=4.2.0 tables>=3.6.1 natsort>=7.0.1 boto3>=1.16.63 torch>=1.7.1 \
      torchvision>=0.8.2
run mkdir -p /root/.cache/torch/hub/checkpoints
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
% elif multiArch.arch == "aarch64":

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=arm64] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
RUN apt-get install -y --no-install-recommends docker-ce \
            docker-ce-cli containerd.io && rm -fr /var/lib/apt/lists/*

# ARM64 needs to have tensorflow installed
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
# Release notes for version info to tie x86 container to jetpack release:
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel

ENV DEBIAN_FRONTEND=noninteractive
# Added 'build-essential', 'python3-devel' for container compatibility
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
    build-essential libpython3-dev python3-scipy python3-matplotlib \
    python3-pil python3-pywt && \
    rm -fr /var/lib/apt/lists/*

# Make sure pip can see the packages we installed from apt-get above
WORKDIR /usr/local/lib/python3.6/dist-packages
RUN ln -s /usr/lib/python3/dist-packages/scipy
RUN ln -s /usr/lib/python3/dist-packages/PIL
RUN ln -s /usr/lib/python3/dist-packages/matplotlib
RUN ln -s /usr/lib/python3/dist-packages/pywt
WORKDIR /
RUN pip3 install --no-cache-dir -U pip testresources setuptools
# Jetson Jetpack already has xavier-optimized opencv2.

# Install pre-built wheels
# These take awhile to build, so its best to do this outside of docker
WORKDIR /packages
COPY config/arm_packages/numpy-1.16.1-cp36-cp36m-linux_aarch64.whl numpy-1.16.1-cp36-cp36m-linux_aarch64.whl
COPY config/arm_packages/scikit_image-0.14.2-cp36-cp36m-linux_aarch64.whl scikit_image-0.14.2-cp36-cp36m-linux_aarch64.whl
COPY config/arm_packages/h5py-2.10.0-cp36-cp36m-linux_aarch64.whl h5py-2.10.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install --no-cache-dir numpy-1.16.1-cp36-cp36m-linux_aarch64.whl
RUN pip3 install --no-cache-dir scikit_image-0.14.2-cp36-cp36m-linux_aarch64.whl
RUN pip3 install --no-cache-dir h5py-2.10.0-cp36-cp36m-linux_aarch64.whl
WORKDIR /
RUN rm -fr /packages

RUN pip3 install --no-cache-dir -U \
         future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5\
	 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf
# Install latest tensorflow 1.XX
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 "tensorflow-gpu<2"

# Install jetpack components
# Grab trusted keyring from host (ugh)
COPY config/arm_packages/nvidia-l4t-apt-source.list /etc/apt/sources.list.d
COPY config/arm_packages/trusted.gpg /etc/apt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnvinfer6 libopencv-python automake autoconf libtool m4 && \
    rm -fr /var/lib/apt/lists/*

# But a cpp version of protobuf on instead of pure python which is what
# ARM gets you by default
RUN pip3 uninstall -y protobuf
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
WORKDIR /protobuf_build
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protobuf-python-3.11.2.tar.gz
RUN tar -xf protobuf-python-3.11.2.tar.gz
WORKDIR /protobuf_build/protobuf-3.11.2
RUN ./autogen.sh
RUN ./configure --prefix=/usr
RUN make -j16
RUN make install
RUN ldconfig
WORKDIR /protobuf_build/protobuf-3.11.2/python
# make c++11 work
RUN sed -i '205s/if v:/if True:/' setup.py
RUN python3 setup.py build --cpp_implementation
RUN python3 setup.py install --cpp_implementation

WORKDIR /
RUN rm -fr protobuf_build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pandas python3-tqdm && \
    rm -fr /var/lib/apt/lists/*

% endif

# Graph solver
WORKDIR /
RUN git clone https://github.com/cvisionai/graph.git && git -C graph checkout 35dc69c9ab25639
RUN mkdir -p /graph/build
WORKDIR /graph/build
RUN cmake ..
RUN make
RUN make -j8 install

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
