/// @file
/// @brief Implementation for tensorflow models.
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

#include "detail/model.h"

#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/array_ops.h>
#include "detail/util.h"

namespace openem {
namespace detail {

namespace tf = tensorflow;

/// Implementation details for Model.
class Model::ModelImpl {
 public:
  /// Tensorflow session.
  std::unique_ptr<tensorflow::Session> session_;

  /// Input size.
  std::vector<int> input_size_;

  /// Indicates whether the model has been initialized.
  bool initialized_;
};

/// Implementation details for ImageModel.
class ImageModel::ImageModelImpl {
 public:
  /// Model object.
  Model model_;

  /// Image width.
  int width_;

  /// Image height.
  int height_;

  /// Queue of futures containing preprocessed images.
  std::queue<std::future<tensorflow::Tensor>> preprocessed_;

  /// Mutex for handling concurrent access to image queue.
  std::mutex mutex_;
};

Model::Model() : impl_(new ModelImpl()) {}

Model::~Model() {}

ErrorCode Model::Init(
    const std::string& model_path, double gpu_fraction) {
  impl_->initialized_ = false;

  // Read in the graph.
  tf::GraphDef graph_def;
  tf::Status status = tf::ReadBinaryProto(
      tf::Env::Default(), model_path, &graph_def);
  if (!status.ok()) return kErrorLoadingModel;

  // Get graph input size.
  ErrorCode status1 = detail::InputSize(graph_def, &(impl_->input_size_));
  if (status1 != kSuccess) return status1;
  
  // Create a new tensorflow session.
  tf::Session* session;
  status1 = GetSession(&session, gpu_fraction);
  if (status1 != kSuccess) return status1;
  impl_->session_.reset(session);

  // Create the tensorflow graph.
  status = impl_->session_->Create(graph_def);
  if (!status.ok()) return kErrorTfGraph;
  impl_->initialized_ = true;
  return kSuccess;
}

std::vector<int> Model::InputSize() {
  return impl_->input_size_;
}

bool Model::Initialized() {
  return impl_->initialized_;
}

ErrorCode Model::Process(
    const tf::Tensor& input,
    const std::string& input_name,
    const std::vector<std::string>& output_names,
    std::vector<tf::Tensor>* outputs) {
  tf::Status status = impl_->session_->Run(
      {{input_name, input}}, 
      output_names, 
      {},
      outputs);
  if (!status.ok()) return kErrorRunSession;
  return kSuccess;
}

ImageModel::ImageModel() : impl_(new ImageModelImpl()) {}

ImageModel::~ImageModel() {}

ErrorCode ImageModel::Init(
    const std::string& model_path, double gpu_fraction) {
  // Do model initialization.
  ErrorCode status = impl_->model_.Init(model_path, gpu_fraction);
  if (status != kSuccess) return status;

  // Get graph input size.
  ErrorCode status1 = detail::ImageSize(
      impl_->model_.InputSize(), &(impl_->width_), &(impl_->height_));
  if (status != kSuccess) return status;
  return kSuccess;
}

cv::Size ImageModel::ImageSize() {
  return cv::Size(impl_->width_, impl_->height_);
}

ErrorCode ImageModel::AddImage(
    const cv::Mat& image,
    std::function<tf::Tensor(const cv::Mat&, int, int)> preprocess) {
  if (!impl_->model_.Initialized()) return kErrorBadInit;
  if (!image.isContinuous()) return kErrorNotContinuous;
  auto f = std::async(preprocess, image, impl_->width_, impl_->height_);
  impl_->mutex_.lock();
  impl_->preprocessed_.push(std::move(f));
  impl_->mutex_.unlock();
  return kSuccess;
}

ErrorCode ImageModel::Process(
    const std::string& input_name,
    const std::vector<std::string>& output_names,
    std::vector<tf::Tensor>* outputs) {
  if (!(impl_->model_.Initialized())) return kErrorBadInit;

  // Copy image queue contents into input tensor.
  impl_->mutex_.lock();
  tf::Tensor input = FutureQueueToTensor(
      &impl_->preprocessed_, 
      impl_->width_, 
      impl_->height_);
  impl_->mutex_.unlock();

  // Run the model.
  return impl_->model_.Process(input, input_name, output_names, outputs);
}

} // namespace detail
} // namespace openem

