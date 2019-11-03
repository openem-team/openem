# Annotation Guidelines

This document describes the data layout for building your own models with OpenEM. Training routines in OpenEM expect the following directory layout:

```shell
your-top-level-directory
├── test
│   ├── truth
│   │   ├── test_video_0.csv
│   │   ├── test_video_1.csv
│   │   └── test_video_2.csv
│   └── videos
│       ├── test_video_0.mp4
│       ├── test_video_1.mp4
│       └── test_video_2.mp4
└── train
    ├── length.csv
    ├── cover.csv
    ├── masks
    │   ├── images
    │   │   ├── 00000.jpg
    │   │   ├── 00001.jpg
    │   │   └── 00002.jpg
    │   └── masks
    │       ├── 00000.png
    │       ├── 00001.png
    │       └── 00002.png
    └── videos
        ├── train_video_0.mp4
        ├── train_video_1.mp4
        └── train_video_2.mp4
```

Many of the annotations require video frame numbers. It is important to point out that most video players do not have frame level accuracy, so attempting to convert timestamps in a typical video player to frame numbers will likely be inaccurate. Therefore we recommend using a frame accurate video annotator such as [Tator][Tator], or converting your video to a series of images before annotating.

## Train directory

**videos** contains video files in mp4 format. The content of these videos should follow the [data collection guidelines][CollectionGuidelines]. We refer to the basename of each video file as the *video ID*, a unique identifier for each video.  In the directory layout above, the video ID for the videos are train_video_0, train_video_1, and train_video_2.

**masks/images** contains single frames from the videos. Each image in this directory has a corresponding binary mask image in **masks/masks**. The high values (RGB value of [255, 255, 255]) in the mask correspond to the ruler, and it is zeros elsewhere.

**length.csv** contains length annotations of fish in the videos. Each row corresponds to an individual fish, specifically the video frame containing the clearest view of each fish. This file is also used to train the counting algorithm, so exactly one frame should be annotated per individual fish. The columns of this file are:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* Choose one of the following annotation styles:
  * *x1, y1, x2, y2*: xy-coordinates of the tip and tail of the fish in pixels.
  * *x,y,width,height,theta*: xy-coordinates of box surrounding the fish in pixels.
* *species_id*: The one-based index of the species as listed in the ini file, as described in the [tutorial][Tutorial]. If this value is zero, it indicates that no fish are present. While length.csv can be used to include no fish example frames, it is encouraged to instead include them in cover.csv. Both are used when training the detection model, but only cover.csv is used when training the classification model.

![Length coordinates of a clearly visible fish.](https://user-images.githubusercontent.com/7937658/49332082-acdd5d00-f574-11e8-8a7e-23a9f9dd1f1b.png)
![Box coordinates of a clearly visible fish.](https://user-images.githubusercontent.com/47112112/59935158-31ad9c80-941b-11e9-86b5-b0e0979b686e.png)

**cover.csv** contains examples of frames that contain no fish, fish covered by a hand or other obstruction, and fish that can be clearly viewed.  The columns of this file are:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* *cover*: 0 for no fish, 1 for covered fish, 2 for clear view of fish.

![Example of image with no fish.](https://user-images.githubusercontent.com/7937658/49332090-c54d7780-f574-11e8-985a-87ac99c56d8c.png)

![Example of image with covered fish.](https://user-images.githubusercontent.com/7937658/49332093-d4342a00-f574-11e8-8e52-6b2988aced75.png)

![Example of image with clear fish.](https://user-images.githubusercontent.com/7937658/49332096-e3b37300-f574-11e8-9e36-64ba90b0e17e.png)

## Test directory

**videos** contains videos that are different from the videos in the train directory but collected in a similar way.

**truth** contains a csv corresponding to each video. Each row corresponds to a fish in the video. The columns in this file are:

* *frame*: The keyframe for each fish.
* *species*: Species of each fish.
* *length*: Length of each fish in pixels.

## Skipping training steps

It is possible to train only some models in openem. For example, you may wish to only train the detect model or only the classify model. During training, there are steps in which model outputs are predicted for use in the next model in the pipeline. In each of these cases, the outputs are written to one of:

```shell
<WorkDir>/inference/find_ruler.csv
<WorkDir>/inference/detect.csv
<WorkDir>/inference/classify.csv
```

The name of the file corresponds to the model that generated it. If you would like to skip training one of these models but a model you wish to train depends on one of these files, you will need to generate the missing file manually as if it were part of the training set annotations. Below is a description of each file:

**find_ruler.csv** contains the ruler position in each video.  The columns of this file are:

* *video_id*: The basename of the video.
* *x1, y1, x2, y2*: xy-coordinates of the ends of the ruler in pixels.

![Ruler coordinates for a video.](https://user-images.githubusercontent.com/7937658/49332099-f6c64300-f574-11e8-89b2-b95e85d26b6e.png)

**detect.csv** contains the bounding box around fish in the video frames. The columns of this file are:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* *x, y, w, h*: The top, left, width, and height of the bounding box in pixels. The origin is at the top left of the video frame.
* *det_conf*: Detection confidence, between 0 and 1. This should be 1 for manually annotation.
* *det_species*: The one-based index of the species as listed in the ini file.

**classify.csv** contains the cover and species for the highest confidence detection in each frame. It has the following columns:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* *no_fish, covered, clear*: Cover category. One of these should be set to 1.0, others should be zero.
* *speces__, species_\**: Species category. The species__ corresponds to background (not a fish). One of these should be set to 1.0, others should be zero.

[Tator]: https://github.com/cvisionai/Tator/releases
[CollectionGuidelines]: ./data_collection.md
[Tutorial]: ./tutorial.md
