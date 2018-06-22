/// @file
/// @brief Example of find_ruler deployment.

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "find_ruler.h"

int main(int argc, char* argv[]) {

  // Declare namespace aliases for shorthand.
  namespace em = openem;
  namespace fr = openem::find_ruler;
  namespace ph = std::placeholders;

  // Check input arguments.
  if (argc < 3) {
    std::cout << "Expected at least arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Paths to one or more image files" << std::endl;
    return -1;
  }

  // Create and initialize the mask finder.
  fr::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(argv[1]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize ruler mask finder!" << std::endl;
    return -1;
  }

  // Load in images.
  std::vector<cv::Mat> imgs;
  for (int i = 2; i < argc; ++i) {
    imgs.push_back(cv::imread(argv[i], CV_LOAD_IMAGE_COLOR));
  }

  // Add images to processing queue.
  for (const auto& img : imgs) {
    status = mask_finder.AddImage(img);
    if (status != em::kSuccess) {
      std::cout << "Failed to add image for processing!" << std::endl;
      return -1;
    }
  }

  // Process the loaded images.
  std::vector<cv::Mat> masks;
  status = mask_finder.Process(&masks);
  if (status != em::kSuccess) {
    std::cout << "Failed to process images!" << std::endl;
    return -1;
  }

  for (int i = 0; i < masks.size(); ++i) {
    cv::imshow("Ruler mask", masks[i]);
    cv::waitKey(0);

    // Check if the ruler is present.
    bool present = fr::RulerPresent(masks[i]);
    if (!present) {
      std::cout << "Could not find ruler in image!  Exiting..." << std::endl;
      continue;
    }

    // Find orientation and region of interest based on the mask.
    double orientation = fr::RulerOrientation(masks[i]);
    cv::Mat r_mask = fr::Rectify(masks[i], orientation);
    cv::Rect roi = fr::FindRoi(masks[i]);

    // Rectify, crop, and display the image.
    cv::Mat r_img = fr::Rectify(imgs[i], orientation);
    cv::Mat c_img = fr::Crop(imgs[i], roi);
  }
  return 0;
}


