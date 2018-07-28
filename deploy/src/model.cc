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

Model::Model()
  : session_(nullptr),
    width_(0),
    height_(0),
    initialized_(false),
    preprocessed_(),
    mutex_() {
}

ErrorCode Model::Init(
    const std::string& model_path, double gpu_fraction) {
  initialized_ = false;

  // Read in the graph.
  tf::GraphDef graph_def;
  tf::Status status = tf::ReadBinaryProto(
      tf::Env::Default(), model_path, &graph_def);
  if (!status.ok()) return kErrorLoadingModel;
  
  // Get graph input size.
  ErrorCode status1 = InputSize(
      graph_def, &(width_), &(height_));
  if (status1 != kSuccess) return status1;

  // Create a new tensorflow session.
  tf::Session* session;
  status1 = GetSession(&session, gpu_fraction);
  if (status1 != kSuccess) return status1;
  session_.reset(session);

  // Create the tensorflow graph.
  status = session_->Create(graph_def);
  if (!status.ok()) return kErrorTfGraph;
  initialized_ = true;
  return kSuccess;
}

cv::Size Model::ImageSize() {
  return cv::Size(width_, height_);
}

ErrorCode Model::AddImage(
    const cv::Mat& image,
    std::function<tf::Tensor(const cv::Mat&, int, int)> preprocess) {
  if (!initialized_) return kErrorBadInit;
  if (!image.isContinuous()) return kErrorNotContinuous;
  auto f = std::async(preprocess, image, width_, height_);
  mutex_.lock();
  preprocessed_.push(std::move(f));
  mutex_.unlock();
  return kSuccess;
}

ErrorCode Model::Process(
    std::vector<tf::Tensor>* outputs, 
    const std::string& input_name,
    const std::vector<std::string>& output_names) {
  if (!initialized_) return kErrorBadInit;

  // Copy image queue contents into input tensor.
  mutex_.lock();
  tf::Tensor input = FutureQueueToTensor(
      &preprocessed_, 
      width_, 
      height_);
  mutex_.unlock();

  // Run the model.
  tf::Status status = session_->Run(
      {{input_name, input}}, 
      output_names, 
      {},
      outputs);
  if (!status.ok()) return kErrorRunSession;
  return kSuccess;
}

} // namespace detail
} // namespace openem

