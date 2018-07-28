## OpenEM: Open Source Electronic Monitoring Library

### Introduction

**OpenEM** is a library that provides advanced video analytics for 
fisheries electronic monitoring (EM) data.  It currently supports detection,
classification and measurement of fish during landing or discard.  This
functionality is currently only available via a deployment library with
pretrained models available in our example data (see tutorial).  The base
library is written in C++, with bindings available for both Python and C#.
Examples are included for all three languages.

There are immediate plans to develop a training library so that users
can build their own models on their own data, and to add counting 
functionality.  Currently builds have only been tested on Windows.  We plan
to support both Ubuntu and macOS in the future.

### Contents

* [Building and Testing](doc/build.md)
* [Tutorial](doc/tutorial.md)
* [Data Collection Guidelines](doc/data_collection.md)
* [Annotation Guidelines](doc/annotation.md)
* [Library reference](https://jrtcppv.bitbucket.io)

