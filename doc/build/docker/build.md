## Build instructions for OpenEM with Docker

The docker image has only been built on Ubuntu 18.04 LTS. If you simply want
to use the docker image you can get the latest release with:

```shell
docker pull cvisionai/openem
```

### Building the image

* Follow instructions [here][NvidiaDocker] to install nvidia-docker.
* From the openem root directory run the following command:

```shell
nvidia-docker build -t openem -f config/Dockerfile .
```

The resulting image will have the OpenEM binary distribution in /openem.

[NvidiaDocker]: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
