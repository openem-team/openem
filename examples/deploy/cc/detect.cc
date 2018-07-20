/// @file
/// @brief Example of detect deployment.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "detect.h"

int main(int argc, char* argv[]) {

  // Declare namespace aliases for shorthand.
  namespace em = openem;
  namespace dt = openem::detect;

  // Check input arguments.
  if (argc < 3) {
    std::cout << "Expected at least two arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Paths to one or more image files" << std::endl;
    return -1;
  }

  // Create and initialize detector.
  dt::Detector detector;
  em::ErrorCode status = detector.Init(argv[1]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize detector!" << std::endl;
    return -1;
  }

  // Load in images.
  std::vector<em::Image> imgs;
  auto size = detector.ImageSize();
  for (int i = 2; i < argc; ++i) {
    em::Image img;
    status = img.FromFile(argv[i]);
    if (status != em::kSuccess) {
      std::cout << "Failed to load image " << argv[i] << "!" << std::endl;
      return -1;
    }
    img.Resize(size.first, size.second);
    imgs.push_back(std::move(img));
  }

  // Add images to processing queue.
  for (const auto& img : imgs) {
    status = detector.AddImage(img);
    if (status != em::kSuccess) {
      std::cout << "Failed to add image for processing!" << std::endl;
      return -1;
    }
  }

  // Process the loaded images.
  std::vector<std::vector<std::array<int, 4>>> detections;
  status = detector.Process(&detections);
  if (status != em::kSuccess) {
    std::cout << "Error when attempting to do detection!" << std::endl;
    return -1;
  }

  // Display the detections on the image.
  for (int i = 0; i < detections.size(); ++i) {
    em::Image img = imgs[i];
    std::vector<std::array<int, 4>> dets = detections[i];
    if (dets.size() == 0) {
      std::cout << "No detections found for image " << i << std::endl;
      continue;
    }
    cv::Mat disp_img = *(reinterpret_cast<cv::Mat*>(img.MatPtr()));
    for (auto det : dets) {
      cv::Rect rect(det[0], det[1], det[2], det[3]);
      cv::rectangle(disp_img, rect, cv::Scalar(0, 0, 255));
    }
    cv::imshow("Detections", disp_img);
    cv::waitKey(0);
  }
}

