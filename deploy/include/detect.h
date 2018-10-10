/// @file
/// @brief Interface for detecting fish in images.
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

#ifndef OPENEM_DEPLOY_DETECT_H_
#define OPENEM_DEPLOY_DETECT_H_

#include <memory>
#include <vector>
#include <string>

#include "image.h"
#include "error_codes.h"

namespace openem {
namespace detect {

/// Contains information for a single detection.
struct Detection {
  /// Location of the detection in the frame.
  Rect location;

  /// Confidence score.
  float confidence;

  /// Species index based on highest confidence.
  int species;
};

/// Class for detecting fish in images.
class Detector {
 public:
  /// Constructor.
  Detector();

  /// Destructor.
  ~Detector();

  /// Initializes the detector.
  /// @param model_path Path to protobuf file containing model.
  /// @param gpu_fraction Fraction of GPU memory that may be allocated to
  /// this object.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, double gpu_fraction=1.0);

  /// Expected image size.  Not valid until the model has been initialized.
  /// @return Expected image size (rows, columns).
  std::pair<int, int> ImageSize();

  /// Adds an image to batch for processing.  
  ///
  /// This function launches a new thread to do image preprocessing 
  /// and immediately returns.  Each call to this function consumes 
  /// additional GPU memory which is cleared when Process is called.  
  /// All images are processed as one batch, so for speed it is 
  /// recommended to call AddImage as many times as possible before 
  /// Process without exceeding available GPU memory.  It is the 
  /// responsibility of the caller to know an appropriate batch size 
  /// for their processing hardware.
  /// @param image Input image for which mask will be found.
  /// @return Error code.
  ErrorCode AddImage(const Image& image);

  /// Finds fish in batched images by performing object detection
  /// with Single Shot Detector (SSD).
  /// @param detections Vector of detections for each image.
  /// @return Error code.
  ErrorCode Process(std::vector<std::vector<Detection>>* detections);
 private:
  /// Forward declaration of implementation class.
  class DetectorImpl;

  /// Pointer to implementation.
  std::unique_ptr<DetectorImpl> impl_;
};

/// Retrieves a detection image.
///
/// Note this is not a simple crop to the detection bounding box.
/// The image extracted uses the x coordinate and width of the
/// input detection, but the y coordinate and height are adjusted
/// such that a square image is extracted.
/// @param image Image used to extract the detection.
/// @param det Detection for which image should be extracted.
Image GetDetImage(const Image& image, const Rect& det);

} // namespace detect
} // namespace openem

#endif // OPENEM_DEPLOY_DETECT_H_

