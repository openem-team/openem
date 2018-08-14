/// @file
/// @brief Interface for counting fish.
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

#ifndef OPENEM_DEPLOY_COUNT_H_
#define OPENEM_DEPLOY_COUNT_H_

#include <memory>
#include <string>
#include <vector>

#include "detect.h"
#include "classify.h"
#include "error_codes.h"

namespace openem {
namespace count {

/// Class for finding keyframes.
class KeyframeFinder {
 public:
  /// Constructor.
  KeyframeFinder();

  /// Destructor.
  ~KeyframeFinder();

  /// Initializes the keyframe finder.
  /// @param model_path Path to protobuf file containing model.
  /// @param img_width Width of image input to detector.
  /// @param img_height Height of image input to detector.
  /// @param gpu_fraction Fraction of GPU memory that may be allocated to
  /// this object.
  /// @return Error code.
  ErrorCode Init(
      const std::string& model_path, 
      int img_width,
      int img_height,
      double gpu_fraction=1.0);

  /// Finds keyframes in a given sequence.
  /// @param classifications Sequence of outputs from classifier.
  /// @param detections Sequence of outputs from detector.
  /// @param keyframes Vector of keyframes in the sequence.  The length of
  /// this vector is the number of fish in the sequence.  The values 
  /// are the indices of the classification and detection vectors, not
  /// the frame numbers.
  /// @return Error code.
  ErrorCode Process(
      const std::vector<classify::Classification>& classifications,
      const std::vector<detect::Detection>& detections,
      std::vector<int>* keyframes);
 private:
  /// Forward declaration of implementation class.
  class KeyframeFinderImpl;

  /// Pointer to implementation.
  std::unique_ptr<KeyframeFinderImpl> impl_;
};

} // namespace count
} // namespace openem

#endif // OPENEM_DEPLOY_COUNT_H_

