## Tutorial

This tutorial will guide you through some examples.  Before you get 
started, make sure you have built the library, including its examples by
following the [build instructions](build.md) for your operating
system.  You can also download the library from the 
[release section][Releases] if available for your operating system.

First, you will need to download the [OpenEM example data][ExampleData].  The
example data includes model files for the deployment library in protobuf 
format, as well as input data that can be used to run the examples.  You can
try running the examples on your own data as well.  The examples should accept
input data in various file formats and resolutions.  They will expect 8-bit 
color image/video data, however.

After you have built your library, you should end up with a subdirectory
called examples/deploy in the top level install directory.  This contains
the examples for the main library in the cc directory, plus python and csharp
if you built the bindings to the main library.  Source files for these examples 
are located [here][ExampleSources] for your inspection.  In addition, there is a
script that will run all of the examples for you if you point it to the
location of the example data.  This script is called [run_all.py][RunAll].

To run this script, you will need to have Python installed.  We recommend
downloading the latest version of [Anaconda][Anaconda].  Open a command prompt
that has access to python and change directories to the examples/deploy 
subdirectory.  Now invoke the [run_all.py][RunAll] script to see how to run it:

```shell
python run_all.py -h
```

This will show you the command line options for this script, and give you 
an explanation of each of the available examples to run.  The simplest way
to invoke the script is as follows:

```shell
python run_all.py <path to OpenEM example data>
```

Doing so will run all available examples in all languages for which you 
built the software.

Once you are able to run the examples, you are encouraged to inspect the 
source code for the language that you plan to use for your application.

[Releases]: https://github.com/openem-team/openem/releases
[ExampleData]:https://drive.google.com/drive/folders/18silAFzXaP27VHLS0texHJz1ZxSMGhjx?usp=sharing
[ExampleSources]: ../examples/deploy
[Anaconda]: https://www.anaconda.com/download/
[RunAll]: ../examples/deploy/run_all.py

