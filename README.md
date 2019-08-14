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

Click the image below to see a video of OpenEM in action:

[![OpenEM detection example](https://img.youtube.com/vi/EZ1Xyg_mnhM/0.jpg)](https://youtu.be/EZ1Xyg_mnhM)

### Distributions

Get the latest Windows library from our [GitHub releases page][Releases].

The docker image can be obtained with:

```shell
docker pull cvisionai/openem:latest
```

### Contents

* [Building](doc/build.md)
* [Tutorial](doc/tutorial.md)
* [Data Collection Guidelines](doc/data_collection.md)
* [Annotation Guidelines](doc/annotation.md)
* [Library reference](https://jrtcppv.bitbucket.io)

[Releases]: https://github.com/openem-team/openem/releases

