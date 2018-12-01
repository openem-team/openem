## Tutorial

This tutorial will guide you through some examples.  Before you get 
started, make sure you have built the library, including its examples by
following the [build instructions](build.md) for your operating
system.  You can also download the library from the 
[release section][Releases] if available for your operating system.

# Deployment Library

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

# Training Library

To build your own models, you will first need to create an appropriate python environment according to the training environment [instructions][TrainingEnvironment]. You will then need to modify the configuration file included with this repository at train/train.ini. This file is included as an example but you will need to modify some paths in it to get it working. Start by making a copy of this file and modify the paths section as follows:

```shell
[Paths]
# Path to directory that contains training data.
TrainDir=<Path to OpenEM example data>/train
# Path to directory for storing intermediate outputs.
WorkDir=<Path where you want to store working files>
# Path to directory for storing final model outputs.
ModelDir=<Path where you want to store models>
```

TrainDir is the path to the example training data.  WorkDir is where temporary files are stored during training. ModelDir contains model outputs that can be used directly by the deployment library.  Once you have modified your copy of train.ini to use the right paths on your system, you can do the following:

```shell
python train.py train.ini preprocess
```

Where train.ini is your modified copy. This command will go through the videos and annotations and start dumping images into the working directory. It will take a few hours to complete.

Next, you can do some training. To train the detection model you can do the following command:

```shell
python train.py train.ini detect
```

This will start training the detection model.  This will take longer, potentially a couple days.  If you want to monitor the training outside of the command line, you can use Tensorboard. This is a program that serves a webpage for monitoring losses during training runs. Use the following command:

```shell
tensorboard --logdir <path to WorkDir>/tensorboard --port 10000
```

Then you can open a web browser on the same machine and go to 127.0.0.1:10000. This will display a live view of the training results. You can also use a different machine on the same network and modify the IP address accordingly.

Once training completes, a new model will be converted to protobuf format at the location specified in train.ini as the ModelDir, subdirectory detect. The file here, detect.pb, is the same format used in the example data for the deployment library.


# Building Datasets

Now that you have done training using the example data, you can try doing the same with your own data.  Follow the [data collection][DataCollection] and [annotation][Annotation] guidelines to build your own training set. Once you have a dataset, you can modify the train.ini file's Data section to include new species to match your data, then repeat the same training process you went through with the example data.

[Releases]: https://github.com/openem-team/openem/releases
[ExampleData]:https://drive.google.com/drive/folders/18silAFzXaP27VHLS0texHJz1ZxSMGhjx?usp=sharing
[ExampleSources]: ../examples/deploy
[Anaconda]: https://www.anaconda.com/download/
[RunAll]: ../examples/deploy/run_all.py
[TrainingEnvironment]: ./training_environment.md

