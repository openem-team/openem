/// @file
/// @brief Interface for classifying cropped images of fish.
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

#ifndef OPENEM_DEPLOY_CLASSIFY_H_
#define OPENEM_DEPLOY_CLASSIFY_H_

#include <memory>
#include <vector>
#include <string>

#include "image.h"
#include "error_codes.h"

namespace openem {
namespace classify {

/// Class for determining fish species and whether the image is covered
/// by a hand, clear, or not a fish.
class Classifier {
 public:
  /// Constructor.
  Classifier();

  /// Destructor.
  ~Classifier();

  /// Initializes the classifier.
  /// @param model_path Path to protobuf file containing model.
  /// @param gpu_fraction Fraction of GPU memory that may be allocated to
  /// this object.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, double gpu_fraction=1.0);

  /// Expected image size.  Not valid until the model has been initialized.
  /// @return Expected image size.
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
  /// @param image Input image that will be classified.
  /// @return Error code.
  ErrorCode AddImage(const Image& image);

  /// Determines fish species and whether the fish is covered by a hand,
  /// clear, or not a fish.
  /// @param scores Vector of double vectors.  Each double vector
  /// corresponds to one of the images in the image queue.  The first
  /// three numbers in the double vector correspond to:
  /// * No fish in the image.
  /// * Fish is covered by a hand.
  /// * Fish is not covered.
  /// The remaining vector elements correspond to the species used to train
  /// the loaded model.
  ErrorCode Process(std::vector<std::vector<float>>* scores);
 private:
  /// Forward declaration of implementation class.
  class ClassifierImpl;

  /// Pointer to implementation.
  std::unique_ptr<ClassifierImpl> impl_;
}; 

} // namespace openem
} // namespace openem

#endif // OPENEM_DEPLOY_CLASSIFY_H_

