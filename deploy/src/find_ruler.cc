///@file
///@brief Implementation for finding rulers in images.

#include "find_ruler.h"

#include <iostream>

namespace openem { namespace find_ruler {

namespace tf = tensorflow;

RulerMaskFinder::RulerMaskFinder() 
    : session_(nullptr) {
}

int RulerMaskFinder::Init(const std::string& model_path) {
  tf::GraphDef graph_def;
  tf::Status status = tf::NewSession(tf::SessionOptions(), &session_);
  if(!status.ok()) {
    std::cout << "Error: Unable to create Tensorflow session!" << std::endl;
    return -1;
  }
  status = tf::ReadBinaryProto(tf::Env::Default(), model_path, &graph_def);
  if(!status.ok()) {
    std::cout << "Error: Failed to read Tensorflow model!" << std::endl;
    return -1;
  }
  status = session_->Create(graph_def);
  if(!status.ok()) {
    std::cout << "Error: Failed to create Tensorflow graph!" << std::endl;
    return -1;
  }
  return 0;
}

cv::Mat RulerMaskFinder::GetMask(const cv::Mat& image) {
  cv::Mat mask;
  return mask;
}

bool RulerPresent(const cv::Mat& mask) {
  return false;
}

double RulerOrientation(const cv::Mat& mask) {
  double orientation = 0.0;
  return orientation;
}

cv::Mat Rectify(const cv::Mat& image, const double orientation) {
  cv::Mat r_image;
  return r_image;
}

cv::Rect FindRoi(const cv::Mat& mask, int h_margin, int v_margin) {
  cv::Rect roi;
  return roi;
}

cv::Mat Crop(const cv::Mat& image, const cv::Rect& roi) {
  cv::Mat c_image;
  return c_image;
}

}} // namespace openem::find_ruler

