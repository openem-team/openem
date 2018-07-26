/// @file
/// @brief End to end example on a video clip.

#include <iostream>

#include "find_ruler.h"
#include "detect.h"
#include "classify.h"
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
    std::vector<std::vector<em::Rect>>* detections, 
    std::vector<std::vector<std::vector<float>>>* scores);

int main(int argc, char* argv[]) {

  // Check input arguments.
  if (argc < 5) {
    std::cout << "Expected at least four arguments: " << std::endl;
    std::cout << "  Path to pb file with find_ruler model." << std::endl;
    std::cout << "  Path to pb file with detect model." << std::endl;
    std::cout << "  Path to pb file with classify model." << std::endl;
    std::cout << "  Path to one or more video files." << std::endl;
  }

  // Find the roi.
  std::cout << "Finding region of interest..." << std::endl;
  em::Rect roi;
  std::vector<double> transform;
  em::ErrorCode status = FindRoi(argv[1], argv[4], &roi, &transform);
  if (status != em::kSuccess) return -1;

  // Find detections and classify them.
  std::cout << "Performing detection and classification..." << std::endl;
  std::vector<std::vector<em::Rect>> detections;
  std::vector<std::vector<std::vector<float>>> scores;
  status = DetectAndClassify(
      argv[2], argv[3], argv[4], roi, transform, &detections, &scores);
}

//
// Implementations
//

em::ErrorCode FindRoi(
    const std::string& mask_finder_path,
    const std::string& vid_path, 
    em::Rect* roi, 
    std::vector<double>* transform) {

  // Create and initialize the mask finder.
  em::find_ruler::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(mask_finder_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to intialize mask finder!" << std::endl;
    return status;
  }

  // Decode the first 100 frames and find the mask that corresponds
  // to the largest ruler area.
  em::VideoReader cap;
  status = cap.Init(vid_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to open video " << vid_path << "!" << std::endl;
    return status;
  }
  int max_img = mask_finder.MaxImages();
  std::vector<em::Image> masks;
  em::Image best_mask;
  double max_mask_sum = 0;
  bool vid_end = false;
  for (int i = 0; i < 100 / max_img; ++i) {
    for (int j = 0; j < max_img; ++j) {
      em::Image img;
      status = cap.GetFrame(&img);
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
  best_mask.Resize(cap.Width(), cap.Height());
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
    std::vector<std::vector<em::Rect>>* detections, 
    std::vector<std::vector<std::vector<float>>>* scores) {

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
    std::cout << "Failed to intialize classifier!" << std::endl;
    return status;
  }

  // Initialize the video reader.
  em::VideoReader cap;
  status = cap.Init(vid_path);
  if (status != em::kSuccess) {
    std::cout << "Failed to open video " << vid_path << "!" << std::endl;
    return status;
  }

  // Iterate through frames.
  bool vid_end = false;
  while (true) {

    // Find detections.
    std::vector<std::vector<em::Rect>> dets;
    std::vector<em::Image> imgs;
    for (int i = 0; i < detector.MaxImages(); ++i) {
      em::Image img;
      status = cap.GetFrame(&img);
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
    detector.Process(&dets);
    detections->insert(detections->end(), dets.begin(), dets.end());

    // Classify detections.
    for (int i = 0; i < dets.size(); ++i) {
      std::vector<std::vector<float>> score_batch;
      for (int j = 0; j < dets[i].size(); ++j) {
        em::Image det_img = em::detect::GetDetImage(imgs[i], dets[i][j]);
        classifier.AddImage(det_img);
      }
      classifier.Process(&score_batch);
      scores->push_back(std::move(score_batch));
    }
    if (vid_end) break;
  }
  return em::kSuccess;
}

