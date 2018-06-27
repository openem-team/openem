/// @file
/// @brief Implementation for detecting fish in images.

#include "detect.h"

#include <opencv2/imgproc.hpp>
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
cv::Mat DecodeBoxes(
    const cv::Mat& loc, 
    const cv::Mat& anchors, 
    const cv::Mat& variances) {
  cv::Mat anchor_width, anchor_height, anchor_center_x, anchor_center_y;
  cv::Mat decode_width, decode_height, decode_center_x, decode_center_y;
  anchor_width = anchors.col(2) - anchors.col(0);
  anchor_height = anchors.col(3) - anchors.col(1);
  anchor_center_x = 0.5 * (anchors.col(2) + anchors.col(0));
  anchor_center_y = 0.5 * (anchors.col(3) + anchors.col(1));
  decode_center_x = loc.col(0) * anchor_width * variances.col(0);
  decode_center_x += anchor_center_x;
  decode_center_y = loc.col(1) * anchor_width * variances.col(1);
  cv::exp(loc.col(2) * variances.col(2), decode_width);
  decode_width *= anchor_width;
  cv::exp(loc.col(3) * variances.col(3), decode_height);
  decode_height *= anchor_height;
  cv::Mat decode_bbox(loc.rows, 4, CV_32F);
  decode_bbox.col(0) = decode_center_x - 0.5 * decode_width;
  decode_bbox.col(1) = decode_center_y - 0.5 * decode_height;
  decode_bbox.col(2) = decode_center_x + 0.5 * decode_width;
  decode_bbox.col(3) = decode_center_y + 0.5 * decode_height;
  return decode_bbox;
}
  
} // namespace

/// Implementation details for Detector.
class Detector::DetectorImpl {
 public:
  /// Stores and processes the model.
  detail::Model model_;
};

Detector::Detector() : impl_(new DetectorImpl()) {}

Detector::~Detector() {}

ErrorCode Detector::Init(
    const std::string& model_path, double gpu_fraction) {
  return impl_->model_.Init(model_path, gpu_fraction);
}

int Detector::MaxImages() {
  return impl_->model_.MaxImages();
}

ErrorCode Detector::AddImage(const cv::Mat& image) {
  auto preprocess = std::bind(
      &util::Preprocess,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3,
      1.0 / 128.0,
      -1.0);
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
  cv::Mat loc, conf, variances, anchors, decoded;
  for (const auto& p : pred) {
    std::vector<cv::Rect> dets;
    loc = p(cv::Range::all(), cv::Range(0, pred_stop));
    conf = p(cv::Range::all(), cv::Range(pred_stop, conf_stop));
    anchors = p(cv::Range::all(), cv::Range(conf_stop, anc_stop));
    variances = p(cv::Range::all(), cv::Range(anc_stop, var_stop));
    decoded = DecodeBoxes(loc, anchors, variances);
    for (int c = 0; c < conf.cols; ++c) {
      cv::Mat over_thresh;
      cv::threshold(conf.col(c), over_thresh, 0.01, 1.0, cv::THRESH_BINARY);
      int total = std::lround(cv::sum(over_thresh)[0]);
      cv::Mat boxes(total, 4, CV_32F);
      cv::Mat confs(total, 1, CV_32F);
      for (int i = 0, k = 0; i < total; ++i) {
        if (over_thresh.at<float>(i) > 0.5) {
          boxes.row(k) = decoded.row(i);
          confs.row(k) = conf.row(i);
          ++k;
        }
      }
      tf::Tensor t_boxes = util::MatToTensor(boxes, {boxes.rows, boxes.cols});
      tf::Tensor t_confs = util::MatToTensor(confs, {confs.rows});
    }
    detections->push_back(std::move(dets));
  }
  return kSuccess;
}

} // namespace detect
} // namespace openem

