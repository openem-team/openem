# Python Environment for Training Library

This document describes how to set up a python environment to use the training library.  It specifies specific module versions that were used to test the training library, but more recent versions of these modules may also work.  The training library has been tested on Windows 10 x64 and Ubuntu 18.04 LTS.

## Installing Miniconda

[Miniconda][Miniconda] is a cross platform distribution of python that includes a utility for managing packages called conda.  It allows for maintenance of multiple python environments that each have different modules or libraries installed.  Download the version for python 3 and install it for your operating system of choice.

## Creating a new environment (optional)

To create a new environment, use this command:

```shell
conda create --name openem
```

This will create a new environment called openem.  To start using it:

```shell
source activate openem
```

If you do not do these steps then modules will be installed in the base environment, which is activated by default. This is fine if you do not plan to use Miniconda for other purposes than OpenEM training.

## Install modules

Versions are included here for reference, different versions may work but have not been tested.

```shell
conda install keras-gpu==2.2.4
conda install pandas==0.23.4
conda install opencv==3.4.2
conda install scikit-image==0.14.0
conda install scikit-learn==0.20.1
```

[Miniconda]: https://conda.io/miniconda.html
