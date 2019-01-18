# Tutorial

This tutorial will guide you through some examples.  Before you get 
started, make sure you have built the library, including its examples by
following the [build instructions](build.md) for your operating
system.  You can also download the library from the 
[release section][Releases] if available for your operating system.

## Deployment Library

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

## Training Library

To build your own models, you will first need to create an appropriate python environment according to the training environment [instructions][TrainingEnvironment]. You will then need to modify the configuration file included with this repository at train/train.ini. This file is included as an example but you will need to modify some paths in it to get it working. Start by making a copy of this file and modify the paths section as follows:

```shell
[Paths]
# Path to directory that contains training data.
TrainDir=<Path to OpenEM example data>/train
# Path to directory for storing intermediate outputs.
WorkDir=<Path where you want to store working files>
# Path to directory for storing final model outputs.
ModelDir=<Path where you want to store models>
# Path to directory that contains test data.
TestDir=<Path to OpenEM example data>/test
```

TrainDir is the path to the example training data.  WorkDir is where temporary files are stored during training. ModelDir contains model outputs that can be used directly by the deployment library.  Once you have modified your copy of train.ini to use the right paths on your system, you can do the following:

```shell
python train.py train.ini extract_images
```

Where train.ini is your modified copy. This command will go through the videos and annotations and start dumping images into the working directory. It will take a few hours to complete. Images are dumped in:

```shell
<WorkDir>/train_imgs
```

Next, you can do some training. To train the find ruler model you can do the following command:

```shell
python train.py train.ini find_ruler_train
```

This will start training the find ruler model. This will take a while. If you want to monitor the training outside of the command line, you can use Tensorboard. This is a program that serves a webpage for monitoring losses during training runs. Use the following command:

```shell
tensorboard --logdir <path to WorkDir>/tensorboard --port 10000
```

Then you can open a web browser on the same machine and go to 127.0.0.1:10000. This will display a live view of the training results. You can also use a different machine on the same network and modify the IP address accordingly. All training steps output tensorboard files, so you can monitor training of any of the openem models using this utility.

Once training completes, a new model will be converted to protobuf format and saved at:

```shell
<ModelDir>/deploy/find_ruler/find_ruler.pb
```

This file is the same format used in the example data for the deployment library.

Now that we have a model for finding rulers, we can run the algorithm on all of our extracted images. Run the following command:

```shell
python train.py train.ini find_ruler_predict
```

This will use the model that we just trained to find the ruler endpoints. The outputs of this are stored at:

```shell
<WorkDir>/inference/find_ruler.csv
```

This file has a simple format, which is just a csv containing the video ID and (x, y) location in pixels of the ruler endpoints. Note that this makes the assumption that the ruler is not moving within a particular video. If it is, you will need to split up your videos into segments in which the ruler is stationary (only for training purposes).

It is possible to train only particular models in openem. Suppose we always know the position of the ruler in our videos and do not need the find ruler algorithm. In this case, we can manually create our own find ruler inference file that contains the same information and store it in the path above. So for example, if we know the ruler is always horizontal spanning the entire video frame, we would use the same (x, y) coordinates for every video in the csv.

The next step is extracting the regions of interest for the videos as determined in the previous step. Run the following:

```shell
python train.py train.ini extract_rois
```

This will dump the ROI image corresponding to each extracted image into:

```shell
<WorkDir>/train_rois
```

Now we are ready to train the detection model. Run the following:

```shell
python train.py train.ini detect_train
```

This training will likely take a couple days. As before you can monitor progress using tensorboard.

By default, the model weights saved to protobuf format are those that correspond to the epoch that yielded the lowest validation loss during training. For various reasons we may wish to choose a different epoch. In this tutorial, we will choose a different epoch for the detect model so that it will be more likely to work on fish when they are covered by a hand. To do this use the following command:

```shell
python select_epoch.py train.ini detect 8
```

This will save over the previously saved detection model in protobuf format, using the model weights from epoch 8. This epoch was selected for this tutorial after some experimentation. You can use the select_epoch.py script to select the epoch of any of the four openem models.

Now we can do detection on all of the ROI images:

```shell
python train.py train.ini detect_predict
```

This will create a new inference output at:

```shell
<WorkDir>/inference/detect.csv
```

As with the find ruler output, if you have a situation where you do not need detection (you always know where the fish is) then you can create this file manually and continue with the next steps.

Next we can extract the detection images:

```shell
python train.py train.ini extract_dets
```

This will dump all of the detection outputs into:

```shell
<WorkDir>/train_dets
```

And finally we can repeat the train/predict cycle for the classifier and counting algorithms:

```shell
python train.py train.ini classify_train
python train.py train.ini classify_predict
python train.py train.ini count_train
```

As with other training steps, these will take a while and can be monitored with TensorBoard. We should now have protobuf models in our designated model directory for all four models.

To test our newly trained models, we can use the test videos included with the openem example data. Run the following command:

```shell
python train.py train.ini test_predict
```

This will run the algorithm models end to end on all videos in the TestDir as specified in train.ini. The outputs will be in:

```shell
<WorkDir>/test
```

One csv will be output for each test video, and will contain the keyframe and species of each fish found. We can compare these outputs to the truth datacontained in the example data with the following command:

```shell
python train.py train.ini test_eval
```

# Building Datasets

Now that you have done training using the example data, you can try doing the same with your own data.  Follow the [data collection][DataCollection] and [annotation][Annotation] guidelines to build your own training set. Once you have a dataset, you can modify the train.ini file's Data section to include new species to match your data, then repeat the same training process you went through with the example data.

[Releases]: https://github.com/openem-team/openem/releases
[ExampleData]:https://drive.google.com/drive/folders/18silAFzXaP27VHLS0texHJz1ZxSMGhjx?usp=sharing
[ExampleSources]: ../examples/deploy
[Anaconda]: https://www.anaconda.com/download/
[RunAll]: ../examples/deploy/run_all.py
[TrainingEnvironment]: ./training_environment.md
[DataCollection]: ./data_collection.md
[Annotation]: ./annotation.md

