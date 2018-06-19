///@file
///@brief Example of find_ruler deployment.

#include "find_ruler.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
  namespace fr = openem::find_ruler;
  if(argc != 3) {
    std::cout << "Expected 2 arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Path to image file" << std::endl;
    return -1;
  }
  fr::RulerMaskFinder mask_finder;
  int status = mask_finder.Init(argv[1]);
  if(status != 0) {
    std::cout << "Failed to initialize ruler mask finder!" << std::endl;
    return -1;
  }
  cv::Mat img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  cv::Mat mask = mask_finder.GetMask(img);
  bool present = fr::RulerPresent(mask);
  if(!present) {
    std::cout << "Could not find ruler in image!  Exiting..." << std::endl;
    return 0;
  }
  double orientation = fr::RulerOrientation(mask);
  cv::Mat r_mask = fr::Rectify(mask, orientation);
  cv::Rect roi = fr::FindRoi(mask);
  cv::Mat r_img = fr::Rectify(img, orientation);
  cv::Mat c_img = fr::Crop(img, roi);
  cv::imshow("Region of interest", c_img);
  cv::waitKey(0);
  return 0;
}


