## OpenEM: Open Source Electronic Monitoring Library

### Introduction

**OpenEM** is a library that provides advanced video analytics for 
fisheries electronic monitoring (EM) data.  It currently supports detection,
classification, counting and measurement of fish during landing or discard.  
This functionality is available via a deployment library with
pretrained models available in our example data (see tutorial).  The base
library is written in C++, with bindings available for both Python and C#.
Examples are included for all three languages.

The current release also includes a training library for all OpenEM
functionality. The library is distributed as a native Windows library and
as a Docker image.

**Documentation** is hosted at [read the docs](https://openem.readthedocs.io/en/latest/).

Click the image below to see a video of OpenEM in action:

[![OpenEM detection example](https://img.youtube.com/vi/EZ1Xyg_mnhM/0.jpg)](https://youtu.be/EZ1Xyg_mnhM)

### Case Studies

**OpenEM** has been used in multiple EM projects. We provide in depth case studies and tutorials showing the power and flexibility of the library. 

**EDF West Coast Activity Recognition**

For this project, CVision AI developed an algorithm using data provided by EDF to identify segments of video with specific activities of interest. You can read more about the overall project [here](http://blogs.edf.org/edfish/2021/03/01/computer-assisted-monitoring-technologies-are-set-to-revolutionize-fisheries/). You can follow the tutorial located [here](case-studies/EDF-Activity-Recognition.md)

### Distributions

Get the latest Windows library from our [GitHub releases page][Releases].

The docker image can be obtained with:

```shell
docker pull cvisionai/openem:latest
```

[Releases]: https://github.com/openem-team/openem/releases

