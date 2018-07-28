/// @file
/// @brief Interface for tensorflow models.
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
//  along with this OpenEM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OPENEM_DEPLOY_MODEL_H_
#define OPENEM_DEPLOY_MODEL_H_

#include <future>
#include <queue>
#include <mutex>

#include <opencv2/core.hpp>
#include <tensorflow/core/public/session.h>
#include "error_codes.h"

namespace openem {
namespace detail {

/// Stores and runs a tensorflow model with multithreaded preprocessing.
/// This class is intended to be an implementation detail only.
class Model {
 public:
  /// Constructor.
  Model();

  /// Loads a model from a protobuf file and initializes the tensorflow
  /// session.
  /// @param model_path Path to protobuf file containing the model.
  /// @param gpu_fraction Fraction fo GPU allowed to be used by this object.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, double gpu_fraction);

  /// Input image size.  Images that are added without having this size
  /// will be resized.  This only returns a valid value if the model
  /// has been initialized.
  cv::Size ImageSize();

  /// Adds an image to batch for processing.  This function launches 
  /// a new thread to do image preprocessing and immediately returns.
  /// @param image Input image for which mask will be found.
  /// @param preprocess Function object to do preprocessing.
  /// @return Error code.
  ErrorCode AddImage(
      const cv::Mat& image,
      std::function<tensorflow::Tensor(const cv::Mat&, int, int)> preprocess);

  /// Processes the model on the current batch.
  /// @param outputs Output of the model.
  /// @param input_name Name of input tensor.
  /// @param output_names Name of output tensors.
  /// @return Error code.
  ErrorCode Process(
      std::vector<tensorflow::Tensor>* outputs, 
      const std::string& input_name,
      const std::vector<std::string>& output_names);
 private:
  /// Tensorflow session.
  std::unique_ptr<tensorflow::Session> session_;

  /// Input image width.
  int width_;

  /// Input image height.
  int height_;

  /// Indicates whether the model has been initialized.
  bool initialized_;

  /// Queue of futures containing preprocessed images.
  std::queue<std::future<tensorflow::Tensor>> preprocessed_;

  /// Mutex for handling concurrent access to image queue.
  std::mutex mutex_;
};

} // namespace detail
} // namespace openem

#endif // OPENEM_DEPLOY_MODEL_H_

