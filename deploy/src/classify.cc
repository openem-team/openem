/// @file
/// @brief Implementation for classifying cropped images of fish.
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
      1.0 / 127.5,
      cv::Scalar(-1.0, -1.0, -1.0),
      true);
  return impl_->model_.AddImage(*mat, preprocess);
}

ErrorCode Classifier::Process(std::vector<std::vector<float>>* scores) {
  // Run the model.
  std::vector<tensorflow::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(
      &outputs, 
      "data", 
      {"cat_species_1:0", "cat_cover_1:0"});
  if (status != kSuccess) return status;

  // Convert to mat vector.
  std::vector<cv::Mat> species;
  detail::TensorToMatVec(outputs[0], &species, 1.0, 0.0, CV_32F);
  std::vector<cv::Mat> quality;
  detail::TensorToMatVec(outputs[1], &quality, 1.0, 0.0, CV_32F);

  // Clear input results.
  scores->clear();

  // Iterate through results for each image.
  for(int i = 0; i < species.size(); ++i) {
    std::vector<float> vec(quality[i].begin<float>(), quality[i].end<float>());
    vec.insert(vec.end(), species[i].begin<float>(), species[i].end<float>());
    scores->push_back(std::move(vec));
  }
  return kSuccess;
}

} // namespace classify
} // namespace openem

