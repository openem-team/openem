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
/// @param vid_path Path to the video.
/// @param roi Rect specifying the ROI.
/// @param transform Transform specifying rectification matrix.
/// @return Error code.
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
        std::cout << "Failed to add frame to processing queue!" << std::endl;
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

int main(int argc, char* argv[]) {

  // Check input arguments.
  if (argc < 5) {
    std::cout << "Expected at least four arguments: " << std::endl;
    std::cout << "  Path to pb file with find_ruler model." << std::endl;
    std::cout << "  Path to pb file with detect model." << std::endl;
    std::cout << "  Path to pb file with classify model." << std::endl;
    std::cout << "  Path to video file." << std::endl;
  }

  // Find the roi.
  std::cout << "Finding region of interest..." << std::endl;
  em::Rect roi;
  std::vector<double> transform;
  em::ErrorCode status = FindRoi(argv[1], argv[4], &roi, &transform);
  if (status != em::kSuccess) return -1;

  // Create and initialize the detector.  Allocate 30% of GPU memory.
  /*em::detect::Detector detector;
  em::ErrorCode status = detector.Init(argv[2], 0.3);
  if (status != em::kSuccess) {
    std::cout << */
}

