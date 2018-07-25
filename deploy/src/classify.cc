/// @file
/// @brief Implementation for classifying cropped images of fish.

#include "classify.h"

#include "detail/model.h"
#include "detail/util.h"

namespace openem {
namespace classify {

/// Implementation details for Classifier.
class Classifier::ClassifierImpl {
 public:
  /// Stores and processes the model.
  detail::Model model_;
};

Classifier::Classifier() : impl_(new ClassifierImpl()) {}

Classifier::~Classifier() {}

ErrorCode Classifier::Init(
    const std::string& model_path,
    double gpu_fraction) {
  return impl_->model_.Init(model_path, gpu_fraction);
}

int Classifier::MaxImages() {
  return impl_->model_.MaxImages();
}

std::pair<int, int> Classifier::ImageSize() {
  cv::Size size = impl_->model_.ImageSize();
  return {size.width, size.height};
}

ErrorCode Classifier::AddImage(const Image& image) {
  const cv::Mat* mat = detail::MatFromImage(&image);
  auto preprocess = std::bind(
      &detail::Preprocess,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3,
      127.5,
      cv::Scalar(-1.0, -1.0, -1.0),
      false);
  return impl_->model_.AddImage(*mat, preprocess);
}

ErrorCode Classifier::Process(std::vector<std::vector<float>>* scores) {
  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(&outputs);
  if (status != kSuccess) return status;

  // Convert to mat vector.
  std::vector<cv::Mat> pred;
  detail::TensorToMatVec(outputs.back(), &pred, 1.0, 0.0, CV_32F);

  // Clear input results.
  scores->clear();

  // Iterate through results for each image.
  for(const auto& p : pred) {
    scores->emplace_back(p.begin<float>(), p.end<float>());
  }
  return kSuccess;
}

} // namespace classify
} // namespace openem

