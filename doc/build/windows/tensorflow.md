## Setting up Tensorflow on Windows

### Download Tensorflow binaries

Download the Windows Tensorflow binaries kindly provided [here][Download].

### Extract the binaries

Extract the package using [7-zip][7zip] to an appropriately named directory, 
such as C:\local\tensorflow.  The path to this directory will be needed to 
build OpenEM.

### Clone the repository

From a git command window, type the following:

```shell
git clone -b v1.8.0 https://github.com/tensorflow/tensorflow.git
```

### Make a directory for building

From a Visual Studio x64 Native Tools command window, do the 
following:

```shell
cd tensorflow
mkdir cmake_build
cd cmake_build
```

### Invoke cmake

Figure out where Anaconda/Miniconda is installed, and use its
base path to supply the python executable path in the below command.
The python executable is needed to generate a version file.

```shell
cmake ../tensorflow/contrib/cmake -G "Visual Studio 14 2015 Win64" ^
-DCMAKE_BUILD_TYPE=Release ^
-DPYTHON_EXECUTABLE=C:/Users/jtaka/Miniconda3/envs/tensorflow/python.exe ^
-Dtensorflow_ENABLE_GPU=ON ^
-DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0" ^
-DCUDA_HOST_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\cl.exe" ^
-DCUDNN_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0" ^
-Dtensorflow_BUILD_PYTHON_BINDINGS=OFF ^
-Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
-Dtensorflow_BUILD_SHARED_LIB=ON
```

### Build the library

To avoid running out of heap space during compilation, the below
command includes options that disable multicore compilation.  Modify
only if you are feeling adventurous. 

```shell
MSBuild ^
/m:1 ^
/p:CL_MPCount=1 ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 ALL_BUILD.vcxproj ^
/filelogger
```
[Download]: https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.8.0/cpp/libtensorflow-gpu-windows-x86_64-1.8.0-sse2cuda92cudnn71.7z
[7zip]: https://www.7-zip.org/
