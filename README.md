## OpenEM: Open Source Electronic Monitoring Library

### Introduction

**OpenEM** is a library that provides advanced video analytics for 
fisheries electronic monitoring (EM) data.  It currently supports detection,
classification, counting and measurement of fish during landing or discard.  
This functionality is currently only available via a deployment library with
pretrained models available in our example data (see tutorial).  The base
library is written in C++, with bindings available for both Python and C#.
Examples are included for all three languages.

There are immediate plans to develop a training library so that users
can build their own models on their own data.  Currently builds have only 
been tested on Windows.  We plan to support both Ubuntu and macOS in the 
future.

Click the image below to see a video of OpenEM in action:

[![OpenEM detection example](https://img.youtube.com/vi/EZ1Xyg_mnhM/0.jpg)](https://youtu.be/EZ1Xyg_mnhM)

### Contents

* [Building](doc/build.md)
* [Tutorial](doc/tutorial.md)
* [Data Collection Guidelines](doc/data_collection.md)
* [Annotation Guidelines](doc/annotation.md)
* [Library reference](https://jrtcppv.bitbucket.io)

