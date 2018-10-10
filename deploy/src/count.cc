/// @file
/// @brief Implementation for counting fish.
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

#include "count.h"

#include <tensorflow/cc/framework/ops.h>
#include "detail/model.h"
#include "detail/util.h"

namespace openem {
namespace count {

namespace tf = tensorflow;

/// Implementation details for KeyframeFinder.
class KeyframeFinder::KeyframeFinderImpl {
 public:
  /// Stores and processes the model.
  detail::Model model_;

  /// Stores image width.  Needed for feature normalization.
  float width_;

  /// Stores image height.  Needed for feature normalization.
  float height_;

  /// Model input size.
  std::vector<int> input_size_;
};

KeyframeFinder::KeyframeFinder() : impl_(new KeyframeFinderImpl()) {}

KeyframeFinder::~KeyframeFinder() {}

ErrorCode KeyframeFinder::Init(
    const std::string& model_path, 
    int img_width,
    int img_height,
    double gpu_fraction) {
  impl_->width_ = static_cast<float>(img_width);
  impl_->height_ = static_cast<float>(img_height);
  ErrorCode status = impl_->model_.Init(model_path, gpu_fraction);
  if (status != kSuccess) return status;
  impl_->input_size_ = impl_->model_.InputSize();
  if (impl_->input_size_.size() != 3) return kErrorNumInputDims;
  return kSuccess;
}

ErrorCode KeyframeFinder::Process(
    const std::vector<std::vector<classify::Classification>>& classifications,
    const std::vector<std::vector<detect::Detection>>& detections,
    std::vector<int>* keyframes) {
  constexpr float kKeyframeThresh = 0.2;
  constexpr int kKeyframeOffset = 32;

  // Get tensor size and do size checks.
  if (classifications.size() != detections.size()) return kErrorLenMismatch;
  int seq_len = impl_->input_size_[1];
  int fea_len = impl_->input_size_[2];
  tf::TensorShape shape({1, seq_len, fea_len});
  tf::Tensor seq_tensor = tf::Input::Initializer(0.0f, shape).tensor;
  auto seq = seq_tensor.flat<float>();

  // Copy data from each detection/classification into sequence tensor.
  for (int n = 0, offset = 0; n < detections.size(); ++n) {

    // If this frame has no detections, continue.
    if (detections[n].empty()) {
      offset += fea_len;
      continue;
    }

    // More size checking.
    const auto& c = classifications[n][0];
    const auto& d = detections[n][0];
    int num_species = static_cast<int>(c.species.size());
    int num_cover = static_cast<int>(c.cover.size());
    int num_loc = static_cast<int>(d.location.size());
    int num_fea = num_species + num_cover + num_loc + 2;
    if (fea_len != (2 * num_fea)) return kErrorBadSeqLength;

    // Normalize the bounding boxes to image size.
    std::array<float, 4> norm_loc = {
      static_cast<float>(d.location[0]) / impl_->width_,
      static_cast<float>(d.location[1]) / impl_->height_,
      static_cast<float>(d.location[2]) / impl_->width_,
      static_cast<float>(d.location[3]) / impl_->height_};

    // Copy twice for now, this is how the existing model works.
    for (int m = 0; m < 2; m++) {
      std::copy_n(c.species.data(), num_species, seq.data() + offset);
      offset += num_species;
      std::copy_n(c.cover.data(), num_cover, seq.data() + offset);
      offset += num_cover;
      std::copy_n(norm_loc.data(), num_loc, seq.data() + offset);
      offset += num_loc;
      seq(offset) = d.confidence;
      offset++;
      seq(offset) = static_cast<float>(d.species);
      offset++;
    }
  }
  // Process the model.
  std::vector<tf::Tensor> outputs;
  ErrorCode status = impl_->model_.Process(
      seq_tensor,
      "input_1",
      {"cumsum_values_1:0"},
      &outputs);

  // Find values over a threshold.
  keyframes->clear();
  auto out = outputs[0].tensor<float, 2>();
  for (int i = 0; i < detections.size(); ++i) {
    if (out(0, i) > kKeyframeThresh) {
      keyframes->push_back(i + kKeyframeOffset);
    }
  }
  return kSuccess;
}

} // namespace count
} // namespace openem

