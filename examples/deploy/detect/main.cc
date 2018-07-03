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
  std::vector<cv::Mat> imgs;
  for (int i = 2; i < argc; ++i) {
    cv::Size size = detector.ImageSize();
    cv::Mat img = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
    cv::resize(img, img, size);
    if (img.total() == 0) {
      std::cout << "Failed to load image " << argv[i] << "!" << std::endl;
      return -1;
    }
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
  std::vector<std::vector<cv::Rect>> detections;
  status = detector.Process(&detections);
  if (status != em::kSuccess) {
    std::cout << "Error when attempting to do detection!" << std::endl;
    return -1;
  }

  // Display the detections on the image.
  for (int i = 0; i < detections.size(); ++i) {
    cv::Mat img = imgs[i];
    std::vector<cv::Rect> dets = detections[i];
    if (dets.size() == 0) {
      std::cout << "No detections found for image " << i << std::endl;
      continue;
    }
    for (auto det : dets) {
      cv::rectangle(img, det, cv::Scalar(0, 0, 255));
    }
    cv::imshow("Detections", img);
    cv::waitKey(0);
  }
}

