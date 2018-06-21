///@file
///@brief Example of find_ruler deployment.

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "find_ruler.h"

/// Copies a vector of images to another vector of images.  This will
/// be the callback executed when processing completes.
/// @param src Source vector of images.
/// @param dst Destination vector of images.
/// @param ready Set to true when copy is complete.
void CopyImages(
    const std::vector<cv::Mat>& src, 
    std::vector<cv::Mat>* dst,
    std::atomic<bool>* ready) {
  *dst = src;
  *ready = true;
}

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

  // The callback used to initialize the mask finder must match 
  // a specific signature, so we define local variables and bind
  // them to the callback using std::bind.
  std::vector<cv::Mat> masks;
  std::atomic<bool> ready = false;
  auto CopyImagesBound = std::bind(CopyImages, ph::_1, &masks, &ready);

  // Create and initialize the mask finder.
  fr::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(argv[1], CopyImagesBound);
  if(status != em::kSuccess) {
    std::cout << "Failed to initialize ruler mask finder!" << std::endl;
    return -1;
  }

  // Load in an image.
  cv::Mat img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

  // Add the image to the processing queue.
  status = mask_finder.AddImage(img);
  if(status != em::kSuccess) {
    std::cout << "Failed to add image for processing!" << std::endl;
    return -1;
  }

  // Initiate processing.
  status = mask_finder.Process();
  if(status != em::kSuccess) {
    std::cout << "Failed to initiate processing!" << std::endl;
    return -1;
  }

  // Wait until processing is complete.
  while(!ready) std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Check if the ruler is present.
  bool present = fr::RulerPresent(masks[0]);
  if(!present) {
    std::cout << "Could not find ruler in image!  Exiting..." << std::endl;
    return 0;
  }

  // Find orientation and region of interest based on the mask.
  double orientation = fr::RulerOrientation(masks[0]);
  cv::Mat r_mask = fr::Rectify(masks[0], orientation);
  cv::Rect roi = fr::FindRoi(masks[0]);

  // Rectify, crop, and display the image.
  cv::Mat r_img = fr::Rectify(img, orientation);
  cv::Mat c_img = fr::Crop(img, roi);
  cv::imshow("Region of interest", c_img);
  cv::waitKey(0);
  return 0;
}


