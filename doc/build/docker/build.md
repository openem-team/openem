# Using OpenEM with Docker

## OpenEM Lite image

The OpenEM docker image in version 0.1.3 and upwards has been retooled to be
a slimmer image based on NVIDIA's GPU cloud offerings. The `openem_lite` image
can be used for both training and inference of OpenEM models. 

### Installing the image

The docker image is provided from dockerhub, and can be installed with:
```shell
docker pull cvisionai/openem_lite:latest
```

### Building the image

* Follow instructions [here][NvidiaDocker] to install nvidia-docker.
* From the openem config directory run the following command:

```shell
make openem_lite
```

### Running outside of docker

Versions 0.1.3 and later of OpenEM do not have a hard requirement of using
the supplied docker image. It is possible to install the openem deployment
library outside of docker. 

```shell
$> cd deploy_python
$> pip3 install .
$> python3
%python3> import openem
```


[NvidiaDocker]: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
