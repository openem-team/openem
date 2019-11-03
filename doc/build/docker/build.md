## Build instructions for OpenEM with Docker

The docker image has only been built on Ubuntu 18.04 LTS. If you simply want
to use the docker image you can get the latest release with:

```shell
docker pull cvisionai/openem:latest
```

### Building the image

* Follow instructions [here][NvidiaDocker] to install nvidia-docker.
* From the openem config directory run the following command:

```shell
make openem-image
```

This will generate the dockerfile from the template and execute the build. If
not initialized, it will setup any submodules required for the project.

The resulting image will have the OpenEM binary distribution in /openem.

[NvidiaDocker]: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
