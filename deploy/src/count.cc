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

#include "detail/model.h"
#include "detail/util.h"

namespace openem {
namespace count {

/// Implementation details for KeyframeFinder.
class KeyframeFinder::KeyframeFinderImpl {
 public:
  /// Stores and processes the model.
  detail::Model model_;

  /// Stores image width.  Needed for feature normalization.
  double width_;

  /// Stores image height.  Needed for feature normalization.
  double height_;

  /// Model input size.
  int input_size_;
};

KeyframeFinder::KeyframeFinder() : impl_(new KeyframeFinderImpl()) {}

KeyframeFinder::~KeyframeFinder() {}

ErrorCode KeyframeFinder::Init(
    const std::string& model_path, 
    int img_width,
    int img_height,
    double gpu_fraction) {
  impl_->width_ = img_width;
  impl_->height_ = img_height;
  return impl_->model_.Init(model_path, gpu_fraction);
}

ErrorCode KeyframeFinder::Process(
    const std::vector<classify::Classification>& classifications,
    const std::vector<detect::Detection>& detections,
    std::vector<int>* keyframes) {
  return kSuccess;
}

} // namespace count
} // namespace openem

