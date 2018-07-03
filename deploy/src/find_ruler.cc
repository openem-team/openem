/// @file
/// @brief Implementation for finding rulers in images.

#include "find_ruler.h"

#include <opencv2/imgproc.hpp>
#include "model.h"
#include "util.h"

namespace openem {
namespace find_ruler {

/// Implementation details for RulerMaskFinder.
class RulerMaskFinder::RulerMaskFinderImpl {
 public:
   /// Stores and processes the model.
   detail::Model model_;
};

RulerMaskFinder::RulerMaskFinder() : impl_(new RulerMaskFinderImpl()) {}

RulerMaskFinder::~RulerMaskFinder() {}

ErrorCode RulerMaskFinder::Init(
    const std::string& model_path, double gpu_fraction) {
  return impl_->model_.Init(model_path, gpu_fraction);
}

int RulerMaskFinder::MaxImages() {
  return impl_->model_.MaxImages();
}

ErrorCode RulerMaskFinder::AddImage(const cv::Mat& image) {
  auto preprocess = std::bind(
      &util::Preprocess, 
      std::placeholders::_1, 
      std::placeholders::_2, 
      std::placeholders::_3, 
      1.0 / 128.0, 
      cv::Scalar(-1.0, -1.0, -1.0),
      true);
  return impl_->model_.AddImage(image, preprocess);
}

ErrorCode RulerMaskFinder::Process(std::vector<cv::Mat>* masks) {
  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(&outputs);
  if (status != kSuccess) return status;

  // Copy model outputs into mask images.
  util::TensorToMatVec(outputs.back(), masks, 255.0, 0.0, CV_8UC1);

  // Do additional processing on the masks.
  for (auto& mask : *masks) {
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    cv::medianBlur(mask, mask, 5);
  }
  return kSuccess;
}

bool RulerPresent(const cv::Mat& mask) {
  return cv::sum(mask)[0] > 1000.0;
}

cv::Mat RulerOrientation(const cv::Mat& mask) {
  // Find center of rotation of the mask.
  cv::Moments m = cv::moments(mask);
  double centroid_x = m.m10 / m.m00;
  double centroid_y = m.m01 / m.m00;
  cv::Point2f centroid(centroid_x, centroid_y);

  // Find transform to translate image to center of rotation.
  double center_x = static_cast<double>(mask.cols) / 2.0;
  double center_y = static_cast<double>(mask.rows) / 2.0;
  cv::Point2f center(center_x, center_y);
  double diff_x = center_x - centroid_x;
  double diff_y = center_y - centroid_y;
  double t[3][3] = {
      {1.0, 0.0, diff_x}, 
      {0.0, 1.0, diff_y}, 
      {0.0, 0.0, 1.0}};
  cv::Mat t_matrix(3, 3, CV_64F, t);
  cv::Mat row = t_matrix.row(2);

  // Rotate image, saving off transform with smallest second central moment.
  double min_moment = 1e99;
  cv::Mat rotated, best;
  for (double ang = -90.0; ang < 90.0; ang += 1.0) {
    cv::Mat r_matrix = cv::getRotationMatrix2D(centroid, ang, 1.0);
    cv::vconcat(r_matrix, row, r_matrix);
    r_matrix = t_matrix * r_matrix;
    r_matrix = r_matrix.rowRange(0, 2);
    cv::warpAffine(mask, rotated, r_matrix, mask.size());
    cv::Moments moments = cv::moments(rotated);
    if (moments.mu02 < min_moment) {
      best = r_matrix.clone();
      min_moment = moments.mu02;
    }
  }
  return best;
}

cv::Mat Rectify(const cv::Mat& image, const cv::Mat& transform) {
  cv::Mat r_image;
  cv::warpAffine(image, r_image, transform, image.size());
  return r_image;
}

cv::Rect FindRoi(const cv::Mat& mask, int h_margin) {
  cv::Rect roi = cv::boundingRect(mask);

  // Find unconstrained bounding rect.
  double aspect = 
    static_cast<double>(mask.rows) / 
    static_cast<double>(mask.cols);
  double v_center = 
    static_cast<double>(roi.y) + static_cast<double>(roi.height) / 2;
  double x0, x1, y0, y1;
  x0 = roi.x - h_margin;
  x1 = roi.x + roi.width + h_margin;
  double v_width = (x1 - x0) * aspect / 2.0;
  y0 = v_center - v_width;
  y1 = v_center + v_width;

  // Constrain the bounding rect.
  x0 = x0 >= 0 ? x0 : 0;
  x1 = x1 < mask.cols ? x1 : mask.cols - 1;
  y0 = y0 >= 0 ? y0 : 0;
  y1 = y1 < mask.rows ? y1 : mask.rows - 1;

  // Convert to roi.
  roi.x = std::lround(x0);
  roi.width = std::lround(x1 - x0 + 1);
  roi.y = std::lround(y0);
  roi.height = std::lround(y1 - y0 + 1);
  return roi;
}

cv::Mat Crop(const cv::Mat& image, const cv::Rect& roi) {
  return image(roi);
}

} // namespace find_ruler
} // namespace openem

