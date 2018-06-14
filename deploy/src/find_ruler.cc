///@file
///@brief Implementation for finding rulers in images.

#include "find_ruler.h"

namespace openem { namespace find_ruler {

RulerMaskFinder::RulerMaskFinder() {
}

void RulerMaskFinder::Init(const std::string& model_path) {
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

