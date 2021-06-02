<%!
import version
import multiArch
%>

# Use generic tensorflow image
FROM tensorflow/tensorflow:1.15.2-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsm6 libxext6 \
    libxrender-dev python3-pip software-properties-common \
    libboost-python-dev curl git build-essential cmake && \
    rm -fr /var/lib/apt/lists/*

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
RUN apt-get install -y --no-install-recommends docker-ce \
            docker-ce-cli containerd.io && rm -fr /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN echo "force 0.1.1"
RUN pip3 install --no-cache-dir opencv-python==4.1.1.26 scikit-image==0.14.2 tator>=0.10.8 docker>=4.2.0 tables>=3.6.1 natsort>=7.0.1 tqdm
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
