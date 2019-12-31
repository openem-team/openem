# Quick Start

This document is a quick start guide in how to use the OpenEM package. The
instructions in this guide still work for newer versions, but usage of the
python inference library is incouraged.

## Example data

This tutorial requires the OpenEM example data which can be downloaded via BitTorrent [here][ExampleData].

## Installation

OpenEM is distributed as a native Windows library or as a Docker image. See below for your selected option.


    Warning: The windows binary releases have been deprecated as of
    version 0.1.3.

    Refer to the python deployment library.

### Windows

* Download the library from our [releases][Releases] page.
* Follow the [instructions][TrainingEnvironment] to set up a Python environment.
* Open an Anaconda command prompt.
* Navigate to where you downloaded OpenEM.

### Docker

* Make sure you have installed nvidia-docker 2 as described [here][NvidiaDocker].
* Pull the docker image from Docker Hub:

```shell
docker pull cvisionai/openem:latest
```

* Start a bash session in the image with the volume containing the example data mounted. The default train.ini file assumes a directory structure as follows:

```
working-dir
|- openem_example_data
|- openem_work
|- openem_model
```

The openem_work and openem_model directories may be empty, and openem_example_data is the example data downloaded at the beginning of this tutorial. The following command will start the bash shell within the container with working-dir mounted to /data. The openem library is located at /openem.

```shell
nvidia-docker run --name openem --rm -ti -v <Path to working-dir>:/data cvisionai/openem bash
```

If using any X11 code, it is important to also enable X11 connections within the docker image:
```shell
nvidia-docker run --name openem --rm -ti -v <Path to working-dir>:/data -v"$HOME/.Xauthority:/root/.Xauthority:rw" --env=DISPLAY --net=host cvisionai/openem bash
```

Note: Instead of `$HOME/.Xauthority` one should use the authority file listed from executing: `xauth info`

### Launching additional shell into a running container
If the container was launched with `--name openem`, then the following command
launches another bash process in the running container:

`docker exec --env=DISPLAY -it openem bash`

Substitute `openem` for what ever you named your container. If you didn't name
your container, then you need to find your running container via `docker ps`
and use:

`docker exec --env=DISPLAY -it <hash_of_container> bash`

## Running the deployment library (0.1.3 and later)

In version 0.1.3 the deployment library has changed from a C++ library with variable language bindings, to
a single python library.

In versions 0.1.3 and later there is a unit test for each inference module that runs against the example data
provided above. To run the test suite; launch the openem-lite image with the example data mounted and the
`deploy_dir` environment variable set appropriately.

The included `Makefile` in config facilitates this by forwarding the host's `work_dir` environment variable to
the container's `deploy_dir` variable. In the config directory with `work_dir` set to
`/path/to/the/openem_example_data` run `make inference_bash`.

The `inference_bash` target launches the nvidia container with recommended settings on device 0; forwarding
port 10001 for potential tensorboard usage.

### Running the tests in the container

The unit tests can be used to verify the underlying computing envioronment for inference and serve as a
regression test against modifications to optomize image preprocessing or result post processing. The unit tests
are also an example usage of the python deployment API.

* Whilst in the container navigate to `/deploy_python`
* Type:

```shell
python -m test
```

* The results of the tests will print out.
* On machines with limited memory resources, it may be required to run each unit test individually, this can
  be done by replacing `test` with `test.CountTest` or `test.DetectionTest`

## Running the deployment library demo (0.1.2 and earlier)

* Navigate to examples/deploy/python.
* Type:

```shell
python video.py -h
```

* This will show you the command line arguments to process a series of videos end to end. The command will look something like:

```shell
python video.py \
    <path to openem_example_data>/find_ruler/find_ruler.pb \
    <path to openem_example_data>/detect/detect.pb \
    <path to openem_example_data>/classify/classify.pb \
    <path to openem_example_data>/count/count.pb \
    <path to video 1> <path to video 2> <path to video 3>
```

* The output will be a csv file with the same base name and location as each video.

### Running with Docker

* If you do not want to enter a docker bash shell and instead want to process a video directly, you can use the following command:

```shell
nvidia-docker run --rm -ti -v \
    <path to openem_example_data>/deploy:/openem_models \
    -e find_ruler_model=/openem_models/find_ruler/find_ruler.pb \
    -e detect_model=/openem_models/detect/detect.pb \
    -e classify_model=/openem_models/classify/classify.pb \
    -e count_model=/openem_models/count/count.pb \
    -e video_paths="<path to video 1> <path to video 2> <path to video 3>" \
    -e CUDA_VISIBLE_DEVICES=0 cvisionai/openem
```

## Deployment library (0.1.2 and earlier)

Navigate to examples/deploy.  This directory contains the examples for the main library in the cc directory, plus python and csharp if you built the bindings to the main library.  Source files for these examples are located [here][ExampleSources] for your inspection.  In addition, there is a script that will run all of the examples for you if you point it to the location of the example data.  This script is called [run_all.py][RunAll].

Now invoke the [run_all.py][RunAll] script to see how to run it:

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

## Training library

To train a model from the example data you will need to modify the configuration file included with the distribution at train/train.ini. This file is included as an example so you will need to modify some paths in it to get it working. Start by making a copy of this file and modify the paths section as follows:

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

### Extracting imagery

TrainDir is the path to the example training data. WorkDir is where temporary files are stored during training. ModelDir contains model outputs that can be used directly by the deployment library.  Once you have modified your copy of train.ini to use the right paths on your system, you can do the following:

```shell
python train.py train.ini extract_images
```

Where train.ini is your modified copy. This command will go through the videos and annotations and start dumping images into the working directory. It will take a few hours to complete. Images are dumped in:

```shell
<WorkDir>/train_imgs
```

### Ruler Training
Next, you can do some training. To train the find ruler model you can do the following command:

```shell
python train.py train.ini find_ruler_train
```

This will start training the find ruler model. This will take a while. If you want to monitor the training outside of the command line, you can use Tensorboard. This is a program that serves a webpage for monitoring losses during training runs. Use the following command:

```shell
tensorboard --logdir <path to WorkDir>/tensorboard --port 10000
```

![Tensorboard example](https://user-images.githubusercontent.com/47112112/60043826-fca98000-968e-11e9-848f-10347587f832.png)

Then you can open a web browser on the same machine and go to 127.0.0.1:10000. This will display a live view of the training results. You can also use a different machine on the same network and modify the IP address accordingly. All training steps output tensorboard files, so you can monitor training of any of the openem models using this utility.

Once training completes, a new model will be converted to protobuf format and saved at:

```shell
<ModelDir>/deploy/find_ruler/find_ruler.pb
```

This file is the same format used in the example data for the deployment library.

#### Running Inference on train/val data

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

### Extract ROIs for Detection Training

The next step is extracting the regions of interest for the videos as determined in the previous step. Run the following:

```shell
python train.py train.ini extract_rois
```

This will dump the ROI image corresponding to each extracted image into:

```shell
<WorkDir>/train_rois
```

### Training Detector

OpenEM supports two detector models. One is the [Single Shot Detector](https://arxiv.org/abs/1512.02325) the other is [RetinaNet](https://arxiv.org/abs/1708.02002). Both models can be trained using the `train.py` tool within OpenEM. The underlying RetinaNet implementation is a forked version of [keras_retinanet](https://github.com/cvisionai/keras_retinanet).

#### Train RetinaNet Detector

To properly train the retinanet model, additional steps are required to generate intermediate artifacts for the underlying retinanet train scripts. These
intermediate artifacts are stored in `<work_dir>/retinanet`. In `train.ini`, additional parameters are supported for retinanet specifically:

```shell
# Random seed for validation split
ValRandomSeed=200
# Validation poulation (0 to 1.0)
ValPopulation=0.2
# Backbone for retinanet implementation
Backbone=resnet152
```

`ValPopulation` is used by the `retinanet_split` to generate a validation population from the overall training population.

##### Generate required retinanet artifacts

```shell
# Generate a csv file compatible with retinanet representing the entire training population (incl. species.csv)
python3 train.py /path/to/train.ini retinanet_prep

# Split population based on ValPopulation and ValRandomSeed
python3 train.py /path/to/train.ini retinanet_split
```

##### Initiate retinanet training

```shell
python3 train.py /path/to/train.ini retinanet_train
```

At this point you will see retinanet training output including losses and current epoch. Tensorboard can be run from `<openem_work>/retinanet/train_log`
and model files are stored in `<openem_work>/retinanet/train_snapshots`.

##### Converting keras-style model to static protobuf format

The keras training script results in a series of h5 files, one for each training epoch. To convert a given epoch to the protobuf format, utilize the
`/scripts/convertToPb.py` script within the openem-lite container.

An example invocation is:
```shell

# Create detection model folder
mkdir -p /data/openem_model/deploy/detect/

# Output epoch 17 to the model area
python3 /scripts/convertToPb.py --resnet /data/openem_work/retinanet/train_snapshots/resnet152_csv_17.h5 /data/openem_model/deploy/detect/detect_retinanet.pb
```

##### Running inference with the protobuf graph output

Similar to the SSD procedure `train.py` can be used to generate detection results on the training population (training + validation).

```shell
# Generate a detect.csv from retinanet detections
python3 train.py /path/to/train.ini retinanet_predict
```

**Note:** If training both SSD and RetinaNet, care should be taken not to overwrite the respective `detect.csv` files.

###### Executing in production environment

The `scripts/infer.py` file can be used as an example or starting point for production runs of inference. The inputs of the inference script support
both openem flavor CSV and retinanet inputs. This can be used to generate detection results on just validation imagery or batches of test imagery.

##### Extracting Retinanet Detection Images

The procedure to extract the detection images for retinanet is identical to the [SSD procedure](#Extracting-detection-imagery).

#### Train Single Shot Detector

Once all ROIs are extracted, run the following:

```shell
python train.py train.ini detect_train
```

This training will likely take a couple days. As before you can monitor progress using tensorboard.

By default, the model weights saved to protobuf format are those that correspond to the epoch that yielded the lowest validation loss during training. For various reasons we may wish to choose a different epoch. In this tutorial, we will choose a different epoch for the detect model so that it will be more likely to work on fish when they are covered by a hand. To do this use the following command:

```shell
python select_epoch.py train.ini detect 8
```

This will save over the previously saved detection model in protobuf format, using the model weights from epoch 8. This epoch was selected for this tutorial after some experimentation. You can use the select_epoch.py script to select the epoch of any of the four openem models.

Besides using it for selecting an earlier epoch, select_epoch.py can also be used when you wish to terminate a training run early. For example, you may find that a training run has converged (losses are no longer decreasing) after 20 epochs even though you have set the number of epochs to 50. If you stop the training run, the script will not get to the point where it writes the best model to disk in protobuf format. This script will allow you to write the latest epoch to protobuf format manually.

Now we can do detection on all of the ROI images:

```shell
python train.py train.ini detect_predict
```

This will create a new inference output at:

```shell
<WorkDir>/inference/detect.csv
```

### Extracting detection imagery

As with the find ruler output, if you have a situation where you do not need detection (you always know where the fish is) then you can create this file manually and continue with the next steps.

Next we can extract the detection images:

```shell
python train.py train.ini extract_dets
```

This will dump all of the detection outputs into:

```shell
<WorkDir>/train_dets
```

### Training the rest of the algorithm pipeline
And finally we can repeat the train/predict cycle for the classifier and counting algorithms:

```shell
python train.py train.ini classify_train
python train.py train.ini classify_predict
python train.py train.ini count_train
```

As with other training steps, these will take a while and can be monitored with TensorBoard. We should now have protobuf models in our designated model directory for all four models.

### Testing complete algorithm chain
To test our newly trained models, we can use the test videos included with the openem example data. Run the following command:

```shell
python train.py train.ini test_predict
```

This will run the algorithm models end to end on all videos in the TestDir as specified in train.ini. The outputs will be in:

```shell
<WorkDir>/test
```

One csv will be output for each test video, and will contain the keyframe and species of each fish found. We can compare these outputs to the truth data contained in the example data with the following command:

```shell
python train.py train.ini test_eval
```

# Building Datasets

Now that you have done training using the example data, you can try doing the same with your own data.  Follow the [data collection][DataCollection] and [annotation][Annotation] guidelines to build your own training set. Once you have a dataset, you can modify the train.ini file's Data section to include new species to match your data, then repeat the same training process you went through with the example data.

[Releases]: https://github.com/openem-team/openem/releases
[ExampleData]: http://academictorrents.com/download/b2a418e07b033bbb37ff46d030d9633d365c148e.torrent
[ExampleSources]: ../examples/deploy
[Anaconda]: https://www.anaconda.com/download/
[RunAll]: ../examples/deploy/run_all.py
[TrainingEnvironment]: ./training_environment.md
[NvidiaDocker]: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
[DataCollection]: ./data_collection.md
[Annotation]: ./annotation.md
