/// @file
/// @brief Implementation for utility functions.

#include "util.h"

#include <opencv2/imgproc.hpp>

namespace openem {
namespace util {

namespace tf = tensorflow;

ErrorCode GetSession(tf::Session** session, double gpu_fraction) {
  tf::SessionOptions options;
  tf::GPUOptions* gpu_options = options.config.mutable_gpu_options();
  gpu_options->set_allow_growth(true);
  gpu_options->set_per_process_gpu_memory_fraction(gpu_fraction);
  tf::Status status = tf::NewSession(options, session);
  if (!status.ok()) return kErrorTfSession;
  return kSuccess;
}

ErrorCode InputSize(const tf::GraphDef& graph_def, int* width, int* height) {
  bool found = false;
  for (auto p : graph_def.node(0).attr()) {
    if (p.first == "shape") {
      found = true;
      auto shape = p.second.shape();
      if (shape.dim_size() != 4) return kErrorGraphDims;
      *width = static_cast<int>(shape.dim(2).size());
      *height = static_cast<int>(shape.dim(1).size());
    }
  }
  if (!found) return kErrorNoShape;
  return kSuccess;
}

tf::Tensor MatToTensor(const cv::Mat& mat) {
  tf::Tensor tensor(tf::DT_FLOAT, tf::TensorShape({1, mat.rows, mat.cols, 3}));
  auto flat = tensor.flat<float>();
  std::copy_n(mat.ptr<float>(), flat.size(), flat.data());
  return tensor;
}

tf::Tensor FutureQueueToTensor(
    std::queue<std::future<tf::Tensor>>* queue,
    int width,
    int height) {
  const int num_img = queue->size();
  tf::Tensor tensor(tf::DT_FLOAT, tf::TensorShape({num_img, height, width, 3}));
  auto flat = tensor.flat<float>();
  int offset = 0;
  for (int n = 0; n < num_img; ++n) {
    tf::Tensor elem = queue->front().get();
    auto elem_flat = elem.flat<float>();
    std::copy_n(elem_flat.data(), elem_flat.size(), flat.data() + offset);
    offset += elem_flat.size();
    queue->pop();
  }
  return tensor;
}

void TensorToMatVec(
    const tensorflow::Tensor& tensor, 
    std::vector<cv::Mat>* vec,
    double scale,
    double bias,
    int dtype) {
  vec->clear();
  const int num_img = tensor.dim_size(0);
  const int height = tensor.dim_size(1);
  const int width = tensor.dim_size(2);
  auto flat = tensor.flat<float>();
  int offset = 0;
  cv::Mat mat(height, width, CV_32FC1);
  float* mat_ptr = mat.ptr<float>();
  for (int n = 0; n < num_img; ++n) {
    std::copy_n(flat.data() + offset, mat.total(), mat_ptr);
    offset += mat.total();
    vec->emplace_back(height, width, dtype);
    mat.convertTo(vec->back(), dtype, scale, bias);
  }
}

tf::Tensor Preprocess(
    const cv::Mat& image, 
    int width, 
    int height,
    double scale,
    double bias) {

  // Start by resizing the image if necessary.
  cv::Mat p_image;
  if ((image.rows != height) || (image.cols != width)) {
    cv::resize(image, p_image, cv::Size(width, height));
  } else {
    p_image = image.clone();
  }

  // Convert to RGB as required by the model.
  cv::cvtColor(p_image, p_image, CV_BGR2RGB);

  // Do image scaling.
  p_image.convertTo(p_image, CV_32F, scale, bias);

  // Copy into tensor.
  return MatToTensor(p_image);
}

} // namespace util
} // namespace openem
