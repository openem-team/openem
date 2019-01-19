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
  constexpr int kKeyframeOffset = 32;
  constexpr int kMinSpacing = 8;
  constexpr float kPeakThresh = 0.05;
  constexpr float kAreaThresh = 0.16;

  // Get tensor size and do size checks.
  if (classifications.size() != detections.size()) return kErrorLenMismatch;
  int seq_len = impl_->input_size_[1];
  int fea_len = impl_->input_size_[2];
  tf::TensorShape shape({1, seq_len, fea_len});
  tf::Tensor seq_tensor = tf::Input::Initializer(0.0f, shape).tensor;
  auto seq = seq_tensor.flat<float>();

  // Copy data from each detection/classification into sequence tensor.
  int k = 0;
  int m = 0;
  bool at_end = false;
  keyframes->clear();
  while(true) {

    // Fill sequence with zeros.
    std::fill_n(seq.data(), seq.size(), 0.0);

    // Add padding.
    k -= kKeyframeOffset;

    // Fill up a sequence.
    for (int n = 0, offset = 0; n < seq_len; ++n, ++k) {

      // Check if we are at end of data.
      if (k == detections.size() + kKeyframeOffset) {
        at_end = true;
        break;
      }

      // If we are doing padding then just continue.
      if (k < 0 || k >= detections.size()) {
        offset += fea_len;
        continue;
      }

      // If this frame has no detections, continue.
      if (detections[k].empty()) {
        offset += fea_len;
        continue;
      }

      // More size checking.
      const auto& c = classifications[k][0];
      const auto& d = detections[k][0];
      int num_species = static_cast<int>(c.species.size());
      int num_cover = static_cast<int>(c.cover.size());
      int num_loc = static_cast<int>(d.location.size());
      int num_fea = num_species + num_cover + num_loc + 2;
      if (fea_len != (2 * num_fea) && fea_len != num_fea) {
        return kErrorBadSeqLength;
      }

      // Normalize the bounding boxes to image size.
      std::array<float, 4> norm_loc = {
        static_cast<float>(d.location[0]) / impl_->width_,
        static_cast<float>(d.location[1]) / impl_->height_,
        static_cast<float>(d.location[2]) / impl_->width_,
        static_cast<float>(d.location[3]) / impl_->height_};

      // Copy based on number of expected inputs.
      int num_models = fea_len / num_fea;
      for (int m = 0; m < num_models; m++) {
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
    auto out = outputs[0].tensor<float, 2>();
    const int out_size = seq_len - 2 * kKeyframeOffset;
    while (true) {

      // Get max value and index.
      int max_idx = 0;
      float max_val = 0.0;
      float max_clear = 0.0;
      int clear_idx = -1;
      for (int idx = 0; idx < out_size; ++idx) {
        if (out(0, idx) > max_val) {
          max_idx = idx;
          max_val = out(0, idx);
        }
      }

      // If below threshold, we are done.
      if (max_val < kPeakThresh) break;

      // Get the sum around the max.
      float sum = 0.0;
      for (
          int idx = (max_idx - kMinSpacing);
          idx <= (max_idx + kMinSpacing);
          ++idx) {
        if (idx < 0) continue;
        if (idx >= out_size) break;
        sum += out(0, idx);
      }

      // If sum is over threshold, find the best detection.
      if (sum > kAreaThresh) {
        for (
            int idx = (max_idx - kMinSpacing);
            idx <= (max_idx + kMinSpacing);
            ++idx) {
          if (idx < 0) continue;
          if (idx >= out_size) break;
          if ((m + idx) >= classifications.size()) break;
          if (classifications[m + idx].size() > 0) {
            if (classifications[m + idx][0].cover[2] > max_clear) {
              max_clear = classifications[m + idx][0].cover[2];
              clear_idx = idx;
            }
          }
        }

        // If a detection is found, add it to keyframes.
        if (clear_idx != -1) {
          keyframes->push_back(m + clear_idx);
        }
      }

      // Zero out area around the keyframe.
      for (
          int idx = (max_idx - kMinSpacing);
          idx <= (max_idx + kMinSpacing);
          ++idx) {
        if (idx < 0) continue;
        if (idx >= out_size) break;
        out(0, idx) = 0;
      }
    }
    m += out_size;

    if (at_end) break;

    // Add padding.
    k -= kKeyframeOffset;
  }
  std::sort(keyframes->begin(), keyframes->end());
  return kSuccess;
}

} // namespace count
} // namespace openem

