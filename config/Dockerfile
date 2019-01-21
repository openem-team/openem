FROM nvidia/cuda
MAINTAINER CVision AI <info@cvisionai.com>

# Configure cuDNN
ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# System packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7
RUN apt-get install -y curl git swig pkg-config zip g++ zlib1g-dev unzip \
    cmake autoconf automake libtool make mercurial

# Install miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Install conda packages
RUN conda install -y keras-gpu==2.2.4
RUN conda install -y pandas==0.23.4
RUN conda install -y opencv==3.4.2
RUN conda install -y scikit-image==0.14.0
RUN conda install -y scikit-learn==0.20.1

# Install bazel
RUN curl -LO https://github.com/bazelbuild/bazel/releases/download/0.18.0/bazel-0.18.0-installer-linux-x86_64.sh 
RUN bash bazel-0.18.0-installer-linux-x86_64.sh --prefix=/bazel
RUN rm bazel-0.18.0-installer-linux-x86_64.sh
ENV PATH=/bazel/bin:${PATH}

# Build TensorFlow C++ API
RUN mkdir config
RUN git clone -b v1.12.0 https://github.com/tensorflow/tensorflow.git
WORKDIR tensorflow
COPY config/tf_config.txt /config/tf_config.txt
RUN ./configure < /config/tf_config.txt
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/compat
RUN bazel build --config=opt --verbose_failures //tensorflow:libtensorflow_cc.so

# Build OpenCV with contrib modules
WORKDIR /
RUN git clone -b 3.4.1 https://github.com/opencv/opencv.git opencv_src
RUN git clone -b 3.4.1 https://github.com/opencv/opencv_contrib.git
WORKDIR opencv_src
RUN mkdir build
WORKDIR build
RUN cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DCMAKE_INSTALL_PREFIX=/opencv \
    ..
RUN make -j8 install

# Build and install protobuf
WORKDIR /
RUN git clone -b v3.6.0 https://github.com/protocolbuffers/protobuf.git
WORKDIR protobuf
RUN git submodule update --init --recursive
RUN ./autogen.sh
RUN ./configure --prefix=/usr
RUN make
RUN make check
RUN make install
RUN ldconfig

# Build and install eigen
WORKDIR /
RUN hg clone -r ea85a5993547 http://bitbucket.org/eigen/eigen
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make -j8 install

# Build and install openem
WORKDIR /
RUN mkdir /openem_src
WORKDIR /openem_src
COPY deploy deploy
COPY train train
COPY doc doc
COPY examples examples
COPY CMakeLists.txt CMakeLists.txt
RUN mkdir build
WORKDIR build
COPY config/tensorflow-config.cmake /config/tensorflow-config.cmake
RUN cmake \
    -DTensorflow_DIR=/config \
    -DOpenCV_DIR=/opencv/share/OpenCV \
    -DPYTHON_LIBRARY=/miniconda/lib/libpython3.6m.so \
    -DPYTHON_INCLUDE_DIR=/miniconda/include \
    -DCMAKE_INSTALL_PREFIX=/openem \
    ..
RUN make -j8 install
