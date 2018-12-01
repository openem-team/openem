## Annotation Guidelines

This document describes the data layout for building your own models with OpenEM. Training routines in OpenEM expect the following directory layout:

```shell
your-top-level-directory
└── train
    ├── cover.csv
    ├── length.csv
    ├── ruler_position.csv
    └── videos
        ├── 00WK7DR6FyPZ5u3A.mp4
        ├── 01wO3HNwawJYADQw.mp4
        ├── 02p3Yn87z0b5grhL.mp4
```

Many of the annotations require video frame numbers. It is important to point out that most video players do not have frame level accuracy, so attempting to convert timestamps in a typical video player to frame numbers will likely be inaccurate. Therefore we recommend using the frame accurate video annotator [Tator][Tator], which guarantees frame level accuracy when seeking and allows line annotations which are useful for generating frame level fish length.

**videos** contains video files in mp4 format. The content of these videos should follow the [data collection guidelines][CollectionGuidelines]. We refer to the basename of each video file as the *video ID*, a unique identifier for each video.  In the directory layout above, the video ID for the videos are 00WK7DR6FyPZ5u3A, 01wO3HNwawJYADQw, and 02p3Yn87z0b5grhL.

**length.csv** contains length annotations of fish in the videos. Each row corresponds to an individual fish, specifically the video frame containing the clearest view of each fish. This file is also used to train the counting algorithm, so exactly one frame should be annotated per individual fish. The columns of this file are:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* *x1, y1, x2, y2*: xy-coordinates of the tip and tail of the fish in pixels.
* *species_id*: The one-based index of the species as listed in the ini file, as described in the [tutorial][Tutorial]. If this value is zero, it indicates that no fish are present. While length.csv can be used to include no fish example frames, it is encouraged to instead include them in cover.csv. Both are used when training the detection model, but only cover.csv is used when training the classification model.

**cover.csv** contains examples of frames that contain no fish, fish covered by a hand or other obstruction, and fish that can be clearly viewed.  The columns of this file are:

* *video_id*: The basename of the video.
* *frame*: The zero-based frame number in the video.
* *cover*: 0 for no fish, 1 for covered fish, 2 for clear view of fish.

**ruler_position.csv** contains the ruler position in each video.  The columns of this file are:

* *video_id*: The basename of the video.
* *x1, y1, x2, y2*: xy-coordinates of the ends of the ruler in pixels.

[Tator]: https://github.com/cvisionai/Tator/releases
[CollectionGuidelines]: ./data_collection.md
[Tutorial]: ./tutorial.md
