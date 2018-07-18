/// @file
/// @brief Implementation for tensorflow models.

#include "model.h"

#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/array_ops.h>
#include "util.h"

namespace openem {
namespace detail {

namespace tf = tensorflow;

Model::Model()
  : session_(nullptr),
    width_(0),
    height_(0),
    batch_size_(64),
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

int Model::MaxImages() {
  return batch_size_;
}

cv::Size Model::ImageSize() {
  return cv::Size(width_, height_);
}

ErrorCode Model::AddImage(
    const cv::Mat& image,
    std::function<tf::Tensor(const cv::Mat&, int, int)> preprocess) {
  if (!initialized_) return kErrorBadInit;
  if (!image.isContinuous()) return kErrorNotContinuous;
  if (preprocessed_.size() >= MaxImages()) return kErrorMaxBatchSize;
  auto f = std::async(preprocess, image, width_, height_);
  mutex_.lock();
  preprocessed_.push(std::move(f));
  mutex_.unlock();
  return kSuccess;
}

ErrorCode Model::Process(std::vector<tf::Tensor>* outputs) {
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
      {{"input_1", input}}, 
      {"output_node0:0"}, 
      {},
      outputs);
  if (!status.ok()) return kErrorRunSession;
  return kSuccess;
}

} // namespace detail
} // namespace openem

