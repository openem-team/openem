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
  if(argc != 3) {
    std::cout << "Expected 2 arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Path to image file" << std::endl;
    return -1;
  }

  // Create and initialize the mask finder.
  fr::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(argv[1]);
  if(status != em::kSuccess) {
    std::cout << "Failed to initialize ruler mask finder!" << std::endl;
    return -1;
  }

  // Load in an image.
  cv::Mat img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

  // Add the image to the processing queue.
  status = mask_finder.AddImage(img);
  //status = mask_finder.AddImage(img);
  //status = mask_finder.AddImage(img);
  if(status != em::kSuccess) {
    std::cout << "Failed to add image for processing!" << std::endl;
    return -1;
  }

  // Process the single loaded image.
  std::vector<cv::Mat> masks;
  status = mask_finder.Process(&masks);
  if(status != em::kSuccess) {
    std::cout << "Failed to process images!" << std::endl;
    return -1;
  }
  cv::imshow("Ruler mask", masks.back());
  cv::waitKey(0);

  // Check if the ruler is present.
  bool present = fr::RulerPresent(masks.back());
  if(!present) {
    std::cout << "Could not find ruler in image!  Exiting..." << std::endl;
    return 0;
  }

  // Find orientation and region of interest based on the mask.
  double orientation = fr::RulerOrientation(masks.back());
  cv::Mat r_mask = fr::Rectify(masks.back(), orientation);
  cv::Rect roi = fr::FindRoi(masks.back());

  // Rectify, crop, and display the image.
  cv::Mat r_img = fr::Rectify(img, orientation);
  cv::Mat c_img = fr::Crop(img, roi);
  cv::imshow("Region of interest", c_img);
  cv::waitKey(0);
  return 0;
}


