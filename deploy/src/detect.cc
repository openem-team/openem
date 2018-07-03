/// @file
/// @brief Implementation for detecting fish in images.

#include "detect.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "model.h"
#include "util.h"

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

int Detector::MaxImages() {
  return impl_->model_.MaxImages();
}

cv::Size Detector::ImageSize() {
  return impl_->model_.ImageSize();
}

ErrorCode Detector::AddImage(const cv::Mat& image) {
  auto preprocess = std::bind(
      &util::Preprocess,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3,
      1.0,
      cv::Scalar(-103.939, -116.779, -123.68));
  return impl_->model_.AddImage(image, preprocess);
}

ErrorCode Detector::Process(std::vector<std::vector<cv::Rect>>* detections) {
  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(&outputs);
  if (status != kSuccess) return status;

  // Convert to mat vector.
  std::vector<cv::Mat> pred;
  util::TensorToMatVec(outputs.back(), &pred, 1.0, 0.0, CV_32F);

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
    std::vector<cv::Rect> dets;
    for (int c = 0; c < conf.cols; ++c) {
      cv::Mat& c_conf = conf.col(c);
      scores.assign(c_conf.begin<float>(), c_conf.end<float>());
      cv::dnn::NMSBoxes(boxes, scores, 0.01, 0.45, indices, 1.0, 200);
      for (int idx : indices) {
        dets.push_back(boxes[idx]);
      }
    }
    detections->push_back(std::move(dets));
  }
  return kSuccess;
}

} // namespace detect
} // namespace openem

