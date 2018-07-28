/// @file
/// @brief Implementation for detecting fish in images.
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

#include "detect.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "detail/model.h"
#include "detail/util.h"

namespace openem {
namespace detect {

namespace tf = tensorflow;

namespace {

/// Decodes bounding box.
/// @param loc Bounding box parameters, one box per row.
/// @param anchors Anchor box parameters, one box per row.
/// @param variances Variances corresponding to each anchor box param.
/// @return Decoded bounding boxes.
std::vector<cv::Rect> DecodeBoxes(
    const cv::Mat& loc, 
    const cv::Mat& anchors, 
    const cv::Mat& variances,
    const cv::Size& img_size);
  
} // namespace

/// Implementation details for Detector.
class Detector::DetectorImpl {
 public:
  /// Stores and processes the model.
  detail::Model model_;
};

namespace {

std::vector<cv::Rect> DecodeBoxes(
    const cv::Mat& loc, 
    const cv::Mat& anchors, 
    const cv::Mat& variances,
    const cv::Size& img_size) {
  std::vector<cv::Rect> decoded;
  cv::Mat anchor_width, anchor_height, anchor_center_x, anchor_center_y;
  cv::Mat decode_width, decode_height, decode_center_x, decode_center_y;
  cv::Mat decode_x0, decode_y0, decode_x1, decode_y1;
  anchor_width = anchors.col(2) - anchors.col(0);
  anchor_height = anchors.col(3) - anchors.col(1);
  anchor_center_x = 0.5 * (anchors.col(2) + anchors.col(0));
  anchor_center_y = 0.5 * (anchors.col(3) + anchors.col(1));
  decode_center_x = loc.col(0).mul(anchor_width).mul(variances.col(0));
  decode_center_x += anchor_center_x;
  decode_center_y = loc.col(1).mul(anchor_height).mul(variances.col(1));
  decode_center_y += anchor_center_y;
  cv::exp(loc.col(2).mul(variances.col(2)), decode_width);
  decode_width = decode_width.mul(anchor_width);
  cv::exp(loc.col(3).mul(variances.col(3)), decode_height);
  decode_height = decode_height.mul(anchor_height);
  decode_x0 = (decode_center_x - 0.5 * decode_width) * img_size.width;
  decode_y0 = (decode_center_y - 0.5 * decode_height) * img_size.height;
  decode_x1 = (decode_center_x + 0.5 * decode_width) * img_size.width;
  decode_y1 = (decode_center_y + 0.5 * decode_height) * img_size.height;
  decode_x0.setTo(0, decode_x0 < 0);
  decode_x0.setTo(img_size.width, decode_x0 >= img_size.width);
  decode_y0.setTo(0, decode_y0 < 0);
  decode_y0.setTo(img_size.height, decode_y0 >= img_size.height);
  decode_x1.setTo(0, decode_x1 < 0);
  decode_x1.setTo(img_size.width, decode_x1 >= img_size.width);
  decode_y1.setTo(0, decode_y1 < 0);
  decode_y1.setTo(img_size.height, decode_y1 >= img_size.height);
  for (int i = 0; i < loc.rows; ++i) {
    decoded.emplace_back(
        decode_x0.at<float>(i),
        decode_y0.at<float>(i),
        decode_x1.at<float>(i) - decode_x0.at<float>(i) + 1,
        decode_y1.at<float>(i) - decode_y0.at<float>(i) + 1);
  }
  return decoded;
}

} // namespace

Detector::Detector() : impl_(new DetectorImpl()) {}

Detector::~Detector() {}

ErrorCode Detector::Init(
    const std::string& model_path, double gpu_fraction) {
  return impl_->model_.Init(model_path, gpu_fraction);
}

std::pair<int, int> Detector::ImageSize() {
  cv::Size size = impl_->model_.ImageSize();
  return {size.width, size.height};
}

ErrorCode Detector::AddImage(const Image& image) {
  const cv::Mat* mat = detail::MatFromImage(&image);
  auto preprocess = std::bind(
      &detail::Preprocess,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3,
      1.0,
      cv::Scalar(-103.939, -116.779, -123.68),
      false);
  return impl_->model_.AddImage(*mat, preprocess);
}

ErrorCode Detector::Process(
    std::vector<std::vector<std::array<int, 4>>>* detections) {
  constexpr int kBackgroundClass = 0;

  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(
      &outputs, 
      "input_1",
      {"output_node0:0"});
  if (status != kSuccess) return status;

  // Convert to mat vector.
  std::vector<cv::Mat> pred;
  detail::TensorToMatVec(outputs.back(), &pred, 1.0, 0.0, CV_32F);

  // Iterate through results for each image.
  int pred_stop = 4;
  int conf_stop = outputs.back().dim_size(2) - 8;
  int anc_stop = conf_stop + 4;
  int var_stop = anc_stop + 4;
  cv::Mat loc, conf, variances, anchors;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> indices;
  double min, max;
  for (const auto& p : pred) {
    loc = p(cv::Range::all(), cv::Range(0, pred_stop));
    conf = p(cv::Range::all(), cv::Range(pred_stop, conf_stop));
    cv::minMaxLoc(loc, &min, &max);
    anchors = p(cv::Range::all(), cv::Range(conf_stop, anc_stop));
    variances = p(cv::Range::all(), cv::Range(anc_stop, var_stop));
    boxes = DecodeBoxes(loc, anchors, variances, impl_->model_.ImageSize());
    std::vector<std::array<int, 4>> dets;
    for (int c = 0; c < conf.cols; ++c) {
      if (c == kBackgroundClass) continue;
      cv::Mat& c_conf = conf.col(c);
      scores.assign(c_conf.begin<float>(), c_conf.end<float>());
      cv::dnn::NMSBoxes(boxes, scores, 0.01, 0.45, indices, 1.0, 200);
      for (int idx : indices) {
        dets.emplace_back(std::array<int, 4>{
            boxes[idx].x,
            boxes[idx].y,
            boxes[idx].width,
            boxes[idx].height});
      }
    }
    detections->push_back(std::move(dets));
  }
  return kSuccess;
}

Image GetDetImage(const Image& image, const Rect& det) {
  int x = det[0];
  int y = det[1];
  int w = det[2];
  int h = det[3];
  int diff = w - h;
  y -= diff / 2;
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if ((x + w) > image.Width()) w = image.Width() - x;
  if ((y + h) > image.Height()) h = image.Height() - y;
  return image.GetSub({x, y, w, h});
}

} // namespace detect
} // namespace openem

