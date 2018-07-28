## Build and Test Instructions for Windows

### Install build tools

Before starting you will need to install the following:

* [Visual Studio](visual_studio.md) 2017
* [CMake][CMake] 3.8.2 or later
* [Git][Git] 2.15 or later
* [CUDA][CUDA] 9.2
* [cuDNN][cuDNN] 7.1

Other configurations are possible but may require some additional
tweaking/troubleshooting.

### Set up third party libraries

* [TensorFlow](tensorflow.md) 1.8.0
* [OpenCV](opencv.md) 3.4.1

### Run CMake

First navigate to the top level OpenEM directory in a Visual Studio
command prompt, which you can access from the start menu at
Visual Studio 2017 > Visual Studio Tools > VC > x64 Native Tools Command
Prompt for VS 2017.  Now you can create a build directory and invoke
cmake as follows:

```shell
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64" ^
-DTensorflow_DIR=C:/local/tensorflow/lib/cmake ^
-DOpenCV_DIR=C:/local/opencv/opencv/build ^
-DCMAKE_INSTALL_PREFIX=C:/local/openem/build/inst
```

### Building

```shell
cmake --build . --config Release --target INSTALL
```

[CMake]: https://cmake.org/
[Git]: https://git-scm.com/download/win
[CUDA]: https://developer.nvidia.com/cuda-downloads
[cuDNN]: https://developer.nvidia.com/cudnn
