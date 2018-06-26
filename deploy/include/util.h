/// @file
/// @brief Interface for utility functions.

#ifndef OPENEM_DEPLOY_UTIL_H_
#define OPENEM_DEPLOY_UTIL_H_

#include <queue>
#include <future>

#include <opencv2/core.hpp>
#include <tensorflow/core/public/session.h>
#include "error_codes.h"

namespace openem {
namespace util {

/// Gets a Tensorflow session.
ErrorCode GetSession(tensorflow::Session** session, double gpu_fraction);

/// Gets graph input size.
ErrorCode InputSize(
    const tensorflow::GraphDef& graph_def, 
    int* width, 
    int* height);

/// Copy a Mat to a Tensor.
tensorflow::Tensor MatToTensor(const cv::Mat& mat);

/// Copy queue of future Tensors to Tensor.
tensorflow::Tensor FutureQueueToTensor(
    std::queue<std::future<tensorflow::Tensor>>* queue,
    int width,
    int height);

/// Copy a tensor to a vector of Mat.
void TensorToMatVec(
    const tensorflow::Tensor& tensor, 
    std::vector<cv::Mat>* vec,
    double scale,
    double bias);

} // namespace openem
} // namespace util

#endif // OPENEM_DEPLOY_UTIL_H_
