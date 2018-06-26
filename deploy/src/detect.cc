/// @file
/// @brief Implementation for detecting fish in images.

#include "detect.h"

#include "model.h"
#include "util.h"

namespace openem {
namespace detect {

namespace tf = tensorflow;

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
  for (const auto& p : pred) {
    std::vector<cv::Rect> dets;
    detections->push_back(std::move(dets));
  }
  return kSuccess;
}

} // namespace detect
} // namespace openem

