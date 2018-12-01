## Build and Test Instructions for Deployment Library on Windows

### Install build tools

Before starting you will need to install the following:

* [Visual Studio](visual_studio.md) 2017
* [CMake][CMake] 3.8.2 or later
* [Git][Git] 2.15 or later
* [CUDA][CUDA] 9.2
* [cuDNN][cuDNN] 7.1
* [SWIG][SWIG] 3.0.12
* [Doxygen][Doxygen] 1.8

Other configurations are possible but may require some additional
tweaking/troubleshooting.  SWIG is only necessary if you want to build
Python or C# bindings.  Doxygen is only necessary if you want to build
the documentation target.

### Set up third party libraries

* [TensorFlow](tensorflow.md) 1.8.0
* [OpenCV](opencv.md) 3.4.1

It is up to you whether you want to use static or dynamic libraries.
OpenEM does not include third party library headers in any examples
or header files, so if you choose to use static libraries OpenEM will
act as a standalone dependency.

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
-DSWIG_EXECUTABLE=C:/local/swigwin-3.0.12/swig.exe ^
-DCMAKE_INSTALL_PREFIX=C:/local/openem/build/inst
```

Your paths may differ depending on where you installed the libraries.
To disable python bindings, include the option -DBUILD_PYTHON=OFF.
To disable C# bindings, include the option -DBUILD_CSHARP=OFF.
To disable the documentation target, include the option -DBUILD_DOCS=OFF.
To build a shared library, include the option -DBUILD_SHARED_LIBS=ON.
Otherwise, a static library will be built.

### Building

To build the main libraries and examples:

```shell
cmake --build . --config Release --target INSTALL
```
To build the documentation target:

```shell
cmake --build . --config Release --target doc
```

You may also try building using the Visual Studio solutions generated
by CMake.

[CMake]: https://cmake.org/
[Git]: https://git-scm.com/download/win
[CUDA]: https://developer.nvidia.com/cuda-downloads
[cuDNN]: https://developer.nvidia.com/cudnn
[SWIG]: http://www.swig.org/download.html
[Doxygen]: https://www.stack.nl/~dimitri/doxygen/download.html
