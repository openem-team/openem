/// @file
/// @brief Implementation for finding rulers in images.
/// @copyright Copyright (C) 2018 CVision AI.
/// @license This file is part of OpenEM, released under GPLv3.
//  OpenEM is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

#include "find_ruler.h"

#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include "detail/model.h"
#include "detail/util.h"

namespace openem {
namespace find_ruler {

/// Implementation details for RulerMaskFinder.
class RulerMaskFinder::RulerMaskFinderImpl {
 public:
   /// Stores and processes the model.
   detail::ImageModel model_;
};

RulerMaskFinder::RulerMaskFinder() : impl_(new RulerMaskFinderImpl()) {}

RulerMaskFinder::~RulerMaskFinder() {}

ErrorCode RulerMaskFinder::Init(
    const std::string& model_path, double gpu_fraction) {
  return impl_->model_.Init(model_path, gpu_fraction);
}

std::pair<int, int> RulerMaskFinder::ImageSize() {
  cv::Size size = impl_->model_.ImageSize();
  return {size.width, size.height};
}

ErrorCode RulerMaskFinder::AddImage(const Image& image) {
  const cv::Mat* mat = detail::MatFromImage(&image);
  auto preprocess = std::bind(
      &detail::Preprocess, 
      std::placeholders::_1, 
      std::placeholders::_2, 
      std::placeholders::_3, 
      1.0 / 128.0, 
      cv::Scalar(-1.0, -1.0, -1.0),
      true);
  return impl_->model_.AddImage(*mat, preprocess);
}

ErrorCode RulerMaskFinder::Process(std::vector<Image>* masks) {
  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(
      "input_1", 
      {"output_node0:0"},
      &outputs);
  if (status != kSuccess) return status;

  // Copy model outputs into mask images.
  detail::TensorToImageVec(outputs.back(), masks, 255.0, 0.0, CV_8UC1);

  // Do additional processing on the masks.
  for (auto& mask : *masks) {
    cv::Mat* mat = detail::MatFromImage(&mask);
    cv::threshold(*mat, *mat, 127, 255, cv::THRESH_BINARY);
    cv::medianBlur(*mat, *mat, 5);
  }
  return kSuccess;
}

bool RulerPresent(const Image& mask) {
  const cv::Mat* mat = detail::MatFromImage(&mask);
  return cv::sum(*mat)[0] > 1000.0;
}

PointPair RulerEndpoints(const Image& mask) {
  // Find center of rotation of the mask.
  const cv::Mat* mat = detail::MatFromImage(&mask);
  cv::Moments m = cv::moments(*mat);
  double centroid_x = m.m10 / m.m00;
  double centroid_y = m.m01 / m.m00;
  cv::Point2f centroid(centroid_x, centroid_y);

  // Find transform to translate image to center of rotation.
  double center_x = static_cast<double>(mat->cols) / 2.0;
  double center_y = static_cast<double>(mat->rows) / 2.0;
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
    cv::warpAffine(*mat, rotated, r_matrix, mat->size());
    cv::Moments moments = cv::moments(rotated);
    if (moments.mu02 < min_moment) {
      best = r_matrix.clone();
      min_moment = moments.mu02;
    }
  }

  // Find the endpoints using the best transform.
  cv::Mat col_sum;
  cv::warpAffine(*mat, rotated, best, mat->size());
  rotated.convertTo(rotated, CV_32F);
  cv::reduce(rotated, col_sum, 0, CV_REDUCE_SUM);
  col_sum.convertTo(col_sum, CV_64F);
  std::vector<double> cum_sum(col_sum.begin<double>(), col_sum.end<double>());
  std::partial_sum(cum_sum.begin(), cum_sum.end(), cum_sum.begin(), std::plus<double>());
  for (auto& elem : cum_sum) elem /= cum_sum.back();
  auto left_it = std::upper_bound(cum_sum.begin(), cum_sum.end(), 0.06);
  int left_col = left_it - cum_sum.begin();
  auto right_it = std::lower_bound(cum_sum.begin(), cum_sum.end(), 0.94);
  int right_col = right_it - cum_sum.begin();
  left_col -= (right_col - left_col) * 0.1;
  right_col += (right_col - left_col) * 0.1;
  std::vector<cv::Point2f> endpoints;
  endpoints.push_back(cv::Point2f(left_col, static_cast<float>(mat->rows) / 2.0));
  endpoints.push_back(cv::Point2f(right_col, static_cast<float>(mat->rows) / 2.0));
  cv::Mat inverse;
  cv::invertAffineTransform(best, inverse);
  cv::vconcat(inverse, row, inverse);
  cv::perspectiveTransform(endpoints, endpoints, inverse);
  return PointPair(
      {endpoints[0].x, endpoints[0].y},
      {endpoints[1].x, endpoints[1].y}
  );
}

Image Rectify(const Image& image, const PointPair& endpoints) {
  Image r_image;
  cv::Mat* r_mat = detail::MatFromImage(&r_image);
  const cv::Mat* mat = detail::MatFromImage(&image);
  cv::Mat t_mat = detail::EndpointsToTransform(
      endpoints.first.first,
      endpoints.first.second,
      endpoints.second.first,
      endpoints.second.second,
      image.Height(),
      image.Width());
  cv::warpAffine(*mat, *r_mat, t_mat, r_mat->size());
  return std::move(r_image);
}

Rect FindRoi(const Image& mask, int h_margin) {
  const cv::Mat* mat = detail::MatFromImage(&mask);
  cv::Rect roi = cv::boundingRect(*mat);

  // Find unconstrained bounding rect.
  double aspect = 
    static_cast<double>(mat->rows) / 
    static_cast<double>(mat->cols);
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
  x1 = x1 < mat->cols ? x1 : mat->cols - 1;
  y0 = y0 >= 0 ? y0 : 0;
  y1 = y1 < mat->rows ? y1 : mat->rows - 1;

  // Convert to roi.
  return {
    std::lround(x0),
    std::lround(y0),
    std::lround(x1 - x0 + 1),
    std::lround(y1 - y0 + 1)};
}

Image Crop(const Image& image, const Rect& roi) {
  Image image_out;
  const cv::Mat* mat_in = detail::MatFromImage(&image);
  cv::Mat* mat_out = detail::MatFromImage(&image_out);
  *mat_out = mat_in->operator()(cv::Rect(roi[0], roi[1], roi[2], roi[3]));
  *mat_out = mat_out->clone(); // To ensure continuous.
  return std::move(image_out);
}

} // namespace find_ruler
} // namespace openem

