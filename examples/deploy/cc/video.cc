/// @file
/// @brief End to end example on a video clip.
/// @copyright Copyright (C) 2018 CVision AI.
/// @license This file is part of OpenEM, released under GPLv3.
//  OpenEM is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <sstream>

#include "find_ruler.h"
#include "detect.h"
#include "classify.h"
#include "count.h"
#include "video.h"

// Declare namespace alias for shorthand.
namespace em = openem;

/// Finds ROI in a video.
/// @param mask_finder_path Path to find_ruler model file.
/// @param vid_path Path to the video.
/// @param roi Rect specifying the ROI.
/// @param transform Transform specifying rectification matrix.
/// @return Error code.
em::ErrorCode FindRoi(
    const std::string& mask_finder_path,
    const std::string& vid_path, 
    em::Rect* roi, 
    std::vector<double>* transform);

/// Finds and classifies detections for all frames in a video.
/// @param detect_path Path to detect model file.
/// @param classify_path Path to classify model file.
/// @param vid_path Path to the video.
/// @param roi Region of interest output from FindRoi.
/// @param transform Transform output from FindRoi.
/// @param detections Detections for each frame.
/// @param scores Cover and species scores for each detection.
em::ErrorCode DetectAndClassify(
    const std::string& detect_path,
    const std::string& classify_path,
    const std::string& vid_path,
    const em::Rect& roi,
    const std::vector<double>& transform,
    std::vector<std::vector<em::detect::Detection>>* detections, 
    std::vector<std::vector<em::classify::Classification>>* scores);

/// Writes a csv file containing fish species and frame numbers.
/// @param count_path Path to count model file.
/// @param out_path Path to output csv file.
/// @param roi Region of interest, needed for image width and height.
/// @param detections Detections for each frame.
/// @param scores Cover and species scores for each detection.
em::ErrorCode WriteCounts(
    const std::string& count_path,
    const std::string& out_path,
    const em::Rect& roi,
    const std::vector<std::vector<em::detect::Detection>>& detections,
    const std::vector<std::vector<em::classify::Classification>>& scores);

/// Writes a new video with bounding boxes around detections.
/// @param vid_path Path to the original video.
/// @param out_path Path to the output video.
/// @param roi Region of interest output from FindRoi.
/// @param transform Transform output from FindRoi.
/// @param detections Detections for each frame.
/// @param scores Cover and species scores for each detection.
em::ErrorCode WriteVideo(
    const std::string& vid_path,
    const std::string& out_path,
    const em::Rect& roi,
    const std::vector<double>& transform,
    const std::vector<std::vector<em::detect::Detection>>& detections,
    const std::vector<std::vector<em::classify::Classification>>& scores);

int main(int argc, char* argv[]) {

  // Check input arguments.
  if (argc < 6) {
    std::cout << "Expected at least four arguments: " << std::endl;
    std::cout << "  Path to pb file with find_ruler model." << std::endl;
    std::cout << "  Path to pb file with detect model." << std::endl;
    std::cout << "  Path to pb file with classify model." << std::endl;
    std::cout << "  Path to pb file with count model." << std::endl;
    std::cout << "  Path to one or more video files." << std::endl;
  }

  for (int vid_idx = 5; vid_idx < argc; ++vid_idx) {
    // Find the roi.
    std::cout << "Finding region of interest..." << std::endl;
    em::Rect roi;
    std::vector<double> transform;
    em::ErrorCode status = FindRoi(argv[1], argv[vid_idx], &roi, &transform);
    if (status != em::kSuccess) return -1;

    // Find detections and classify them.
    std::cout << "Performing detection and classification..." << std::endl;
    std::vector<std::vector<em::detect::Detection>> detections;
    std::vector<std::vector<em::classify::Classification>> scores;
    status = DetectAndClassify(
        argv[2], 
        argv[3], 
        argv[vid_idx], 
        roi, 
        transform, 
        &detections, 
        &scores);
    if (status != em::kSuccess) return -1;

    // Write fish counts to file.
    std::cout << "Writing counts to file..." << std::endl;
    std::stringstream ss1;
    ss1 << "fish_counts_" << vid_idx - 4 << ".csv";
    status = WriteCounts(
        argv[4],
        ss1.str(),
        roi,
        detections,
        scores);

    // Write annotated video to file.
    std::cout << "Writing video to file..." << std::endl;
    std::stringstream ss;
    ss << "annotated_video_" << vid_idx - 4 << ".avi";
    status = WriteVideo(
        argv[vid_idx],
        ss.str(),
        roi,
        transform,
        detections,
        scores);
    if (status != em::kSuccess) return -1;
  }
  return 0;
}

//
// Implementations
//

em::ErrorCode FindRoi(
    const std::string& mask_finder_path,
    const std::string& vid_path, 
    em::Rect* roi, 
    std::vector<double>* transform) {
  // Determined by experimentation with GPU having 8GB memory.
  static const int kMaxImg = 8;

  // Create and initialize the mask finder.
  em::find_ruler::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(mask_finder_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to intialize mask finder!" << std::endl;
    return status;
  }

  // Decode the first 100 frames and find the mask that corresponds
  // to the largest ruler area.
  em::VideoReader reader;
  status = reader.Init(vid_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to open video " << vid_path << "!" << std::endl;
    return status;
  }
  std::vector<em::Image> masks;
  em::Image best_mask;
  double max_mask_sum = 0;
  bool vid_end = false;
  for (int i = 0; i < 100 / kMaxImg; ++i) {
    for (int j = 0; j < kMaxImg; ++j) {
      em::Image img;
      status = reader.GetFrame(&img);
      if (status != em::kSuccess) {
        vid_end = true;
        break;
      }
      status = mask_finder.AddImage(img);
      if (status != em::kSuccess) {
        std::cout << "Failed to add frame to mask finder!" << std::endl;
        return status;
      }
    }
    status = mask_finder.Process(&masks);
    if (status != em::kSuccess) {
      std::cout << "Failed to process mask finder!" << std::endl;
      return status;
    }
    for (const auto& mask : masks) {
      double mask_sum = mask.Sum()[0];
      if (mask_sum > max_mask_sum) {
        max_mask_sum = mask_sum;
        best_mask = mask;
      }
    }
    if (vid_end) break;
  }

  // Now that we have the best mask, use this to compute the ROI.
  best_mask.Resize(reader.Width(), reader.Height());
  *transform = em::find_ruler::RulerOrientation(best_mask);
  em::Image r_mask = em::find_ruler::Rectify(best_mask, *transform);
  *roi = em::find_ruler::FindRoi(r_mask);
  return em::kSuccess;
}

em::ErrorCode DetectAndClassify(
    const std::string& detect_path,
    const std::string& classify_path,
    const std::string& vid_path,
    const em::Rect& roi,
    const std::vector<double>& transform,
    std::vector<std::vector<em::detect::Detection>>* detections, 
    std::vector<std::vector<em::classify::Classification>>* scores) {
  // Determined by experimentation with GPU having 8GB memory.
  static const int kMaxImg = 32;

  // Create and initialize the detector.
  em::detect::Detector detector;
  em::ErrorCode status = detector.Init(detect_path, 0.5);
  if (status != em::kSuccess) {
    std::cout << "Failed to intialize detector!" << std::endl;
    return status;
  }

  // Create and initialize the classifier.
  em::classify::Classifier classifier;
  status = classifier.Init(classify_path, 0.5);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize classifier!" << std::endl;
    return status;
  }

  // Initialize the video reader.
  em::VideoReader reader;
  status = reader.Init(vid_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to open video " << vid_path << "!" << std::endl;
    return status;
  }

  // Iterate through frames.
  bool vid_end = false;
  while (true) {

    // Find detections.
    std::vector<std::vector<em::detect::Detection>> dets;
    std::vector<em::Image> imgs;
    for (int i = 0; i < kMaxImg; ++i) {
      em::Image img;
      status = reader.GetFrame(&img);
      if (status != em::kSuccess) {
        vid_end = true;
        break;
      }
      img = em::find_ruler::Rectify(img, transform);
      img = em::find_ruler::Crop(img, roi);
      imgs.push_back(img);
      status = detector.AddImage(img);
      if (status != em::kSuccess) {
        std::cout << "Failed to add frame to detector!" << std::endl;
        return status;
      }
    }
    status = detector.Process(&dets);
    if (status != em::kSuccess) {
      std::cout << "Failed to process detector!" << std::endl;
      return status;
    }
    detections->insert(detections->end(), dets.begin(), dets.end());

    // Classify detections.
    for (int i = 0; i < dets.size(); ++i) {
      std::vector<em::classify::Classification> score_batch;
      for (int j = 0; j < dets[i].size(); ++j) {
        em::Image det_img = em::detect::GetDetImage(
            imgs[i], 
            dets[i][j].location);
        status = classifier.AddImage(det_img);
        if (status != em::kSuccess) {
          std::cout << "Failed to add frame to classifier!" << std::endl;
          return status;
        }
      }
      status = classifier.Process(&score_batch);
      if (status != em::kSuccess) {
        std::cout << "Failed to process classifier!" << std::endl;
        return status;
      }
      scores->push_back(std::move(score_batch));
    }
    if (vid_end) break;
  }
  return em::kSuccess;
}

em::ErrorCode WriteCounts(
    const std::string& count_path,
    const std::string& out_path,
    const em::Rect& roi,
    const std::vector<std::vector<em::detect::Detection>>& detections,
    const std::vector<std::vector<em::classify::Classification>>& scores) {

  // Create and initialize keyframe finder.
  em::count::KeyframeFinder finder;
  em::ErrorCode status = finder.Init(count_path, roi[2], roi[3]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize keyframe finder!" << std::endl;
    return status;
  }

  // Process the keyframe finder.
  std::vector<int> keyframes;
  status = finder.Process(scores, detections, &keyframes);
  if (status != em::kSuccess) {
    std::cout << "Failed to process keyframe finder!" << std::endl;
    return status;
  }
  return em::kSuccess;
}

em::ErrorCode WriteVideo(
    const std::string& vid_path,
    const std::string& out_path,
    const em::Rect& roi,
    const std::vector<double>& transform,
    const std::vector<std::vector<em::detect::Detection>>& detections,
    const std::vector<std::vector<em::classify::Classification>>& scores) {

  // Initialize the video reader.
  em::VideoReader reader;
  em::ErrorCode status = reader.Init(vid_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to read video " << vid_path << "!" << std::endl;
    return status;
  }

  // Initialize the video writer.
  em::VideoWriter writer;
  status = writer.Init(
      out_path, 
      reader.FrameRate(), 
      em::kWmv2, 
      {reader.Width(), reader.Height()});
  if (status != em::kSuccess) {
    std::cout << "Failed to write video " << out_path << "!" << std::endl;
    return status;
  }

  // Iterate through frames.
  for (int i = 0; i < detections.size(); ++i) {
    em::Image frame;
    status = reader.GetFrame(&frame);
    if (status != em::kSuccess) {
      std::cout << "Error retrieving video frame!" << std::endl;
      return status;
    }
    frame.DrawRect(roi, {255, 0, 0}, 1, transform);
    for (int j = 0; j < detections[i].size(); ++j) {
      em::Color det_color;
      double clear = scores[i][j].cover[2];
      double hand = scores[i][j].cover[1];
      if (j == 0) {
        if (clear > hand) {
          frame.DrawText("Clear", {0, 0}, {0, 255, 0});
          det_color = {0, 255, 0};
        } else {
          frame.DrawText("Hand", {0, 0}, {0, 0, 255});
          det_color = {0, 0, 255};
        }
      }
      frame.DrawRect(detections[i][j].location, det_color, 2, transform, roi);
    }
    status = writer.AddFrame(frame);
    if (status != em::kSuccess) {
      std::cout << "Error adding frame to video!" << std::endl;
      return status;
    }
  }
  return em::kSuccess;
}
