# Top Layer just gets an updated cuda image
FROM nvidia/cuda AS cvbase0
MAINTAINER CVision AI <info@cvisionai.com>

# Configure cuDNN
ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# System packages
# Combine apt-get update and apt-get install to fix cache bug.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && apt-get install -y --no-install-recommends curl git swig pkg-config zip g++ zlib1g-dev unzip \
    cmake autoconf automake libtool make mercurial ffmpeg libavcodec-dev \
    libavformat-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev \
    libv4l-dev python3 python3-pip python3-dev python3-setuptools && apt-get clean && rm -fr /var/lib/apt/lists/*

# Temporary layer to do utility wrangling
# Eigen, protobuf
FROM cvbase0 AS cvwrangler
# Build and install protobuf
WORKDIR /
RUN git clone -b v3.6.0 https://github.com/protocolbuffers/protobuf.git
WORKDIR protobuf
RUN git submodule update --init --recursive
RUN ./autogen.sh
RUN ./configure --prefix=/opt
RUN make -j8 && make -j8 check && make -j8 install
RUN ldconfig
WORKDIR /
RUN rm -rf protobuf

# Build and install eigen
WORKDIR /
RUN hg clone -r ea85a5993547 http://bitbucket.org/eigen/eigen
WORKDIR eigen
RUN mkdir build
WORKDIR build
RUN cmake -DINCLUDE_INSTALL_DIR=/opt/include .. && make -j8 install
WORKDIR /
RUN rm -rf eigen

# cvbase is now clean of temporary layers used in construction.
FROM cvbase0 AS cvbase 
copy --from=cvwrangler /opt /opt

# Install python packages
RUN pip3 --no-cache-dir install numpy==1.16.1 scikit-image==0.14.0 scikit-learn==0.20.1 pandas==0.23.4 opencv-python==3.4.2.17 ##


FROM cvbase AS cvtensorflow
# Install bazel
RUN curl -LO https://github.com/bazelbuild/bazel/releases/download/0.18.0/bazel-0.18.0-installer-linux-x86_64.sh 
RUN bash bazel-0.18.0-installer-linux-x86_64.sh --prefix=/bazel
RUN rm bazel-0.18.0-installer-linux-x86_64.sh #
ENV PATH=/bazel/bin:${PATH}

# Build TensorFlow C++ API
RUN mkdir config
RUN git clone -b v1.12.0 https://github.com/tensorflow/tensorflow.git
WORKDIR tensorflow
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH
COPY config/tf_config.txt /config/tf_config.txt
RUN ./configure < /config/tf_config.txt
RUN bazel build --config=opt --config=cuda --verbose_failures //tensorflow:libtensorflow_cc.so
RUN bazel build --config=opt --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN cp /tmp/tensorflow_pkg/`find /tmp/tensorflow_pkg/ -name "*.whl"` /tensorflow/tf.whl 
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

FROM cvbase AS cvopencv
# Build OpenCV with contrib modules
WORKDIR /
RUN git clone -b 3.4.1 https://github.com/opencv/opencv.git opencv_src
RUN git clone -b 3.4.1 https://github.com/opencv/opencv_contrib.git
WORKDIR opencv_src
RUN mkdir build
WORKDIR build
RUN cmake \
    -DWITH_FFMPEG=ON \
    -DWITH_LIBV4L=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DCMAKE_INSTALL_PREFIX=/opencv \
    ..
RUN make -j8 install
WORKDIR /
RUN rm -rf opencv_src
RUN rm -rf opencv_contrib



FROM cvbase as cvopenem

# Naive Implementation copies everything:
# COPY --from=cvtensorflow /tensorflow /tensorflow
# Bazel hides things in /root which isn't very pleasant. 
# Only copy in what we need to build
RUN mkdir -p /tensorflow/lib
RUN mkdir -p /tensorflow/include
RUN mkdir -p /tensorflow/include/bazel-genfiles

COPY --from=cvtensorflow /tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so /tensorflow/lib
COPY --from=cvtensorflow /tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so /tensorflow/lib
COPY --from=cvtensorflow /tensorflow/tensorflow /tensorflow/include/tensorflow
COPY --from=cvtensorflow /tensorflow/third_party /tensorflow/include/third_party
COPY --from=cvtensorflow /tensorflow/bazel-genfiles/tensorflow /tensorflow/include/bazel-genfiles/tensorflow
# Copy abseil to include dir
COPY --from=cvtensorflow /tensorflow/bazel-tensorflow/external/com_google_absl/absl /usr/include/absl
COPY --from=cvtensorflow /tensorflow/tf.whl /tensorflow/tf.whl
RUN pip3 install /tensorflow/tf.whl

COPY --from=cvopencv /opencv /opencv

# Build and install openem
WORKDIR /
RUN mkdir /openem_src
WORKDIR /openem_src
COPY deploy deploy
COPY train train
COPY doc doc
COPY examples examples
COPY CMakeLists.txt CMakeLists.txt
COPY LICENSE.md LICENSE.md
RUN mkdir build
WORKDIR build
COPY config/tensorflow-config.cmake /config/tensorflow-config.cmake
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:${LD_LIBRARY_PATH}
RUN cmake \
    -DTensorflow_DIR=/config \
    -DOpenCV_DIR=/opencv/share/OpenCV \
    -DPYTHON_LIBRARY=/miniconda/lib/libpython3.6m.so \
    -DPYTHON_INCLUDE_DIR=/miniconda/include/python3.6m \
    -DCMAKE_INSTALL_PREFIX=/openem \
    ..
RUN make -j8 install
WORKDIR /
RUN rm -rf openem_src
RUN rm -rf opencv

FROM cvbase as cvopenem_deploy
COPY --from=cvopenem /tensorflow /tensorflow
COPY --from=cvopenem /openem /openem

# Add libraries to path
ENV LD_LIBRARY_PATH=/tensorflow/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set up environment variables for command line invokation
ENV find_ruler_model /openem_example_data/deploy/find_ruler/find_ruler.pb
ENV detect_model /openem_example_data/deploy/detect/detect.pb
ENV classify_model /openem_example_data/deploy/classify/classify.pb
ENV count_model /openem_example_data/deploy/count/count.pb
ENV video_paths "/openem_example_data/deploy/video/test_video_000.mp4 /openem_example_data/deploy/video/test_video_001.mp4 /openem_example_data/deploy/video/test_video_002.mp4"
ENV video_out "--no_video"

# Define run command
WORKDIR /openem/examples/deploy/python
CMD ["sh", "-c", "python video.py ${find_ruler_model} ${detect_model} ${classify_model} ${count_model} ${video_paths} ${video_out}"]
