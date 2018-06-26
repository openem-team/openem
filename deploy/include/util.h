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
/// @param session Pointer to session address.
/// @param gpu_fraction Fraction (0.0-1.0) indicating how much GPU memory
/// can be allocated to this session.
/// @return Error code.
ErrorCode GetSession(tensorflow::Session** session, double gpu_fraction);

/// Gets graph input size.
/// @param graph_def Graph definition.
/// @param width Width dimension of input layer.
/// @param height Height dimension of input layer.
/// @return Error code.
ErrorCode InputSize(
    const tensorflow::GraphDef& graph_def, 
    int* width, 
    int* height);

/// Copy a Mat to a Tensor.
/// @param mat Mat to be converted.
/// @return Tensor containing mat data.
tensorflow::Tensor MatToTensor(const cv::Mat& mat);

/// Copy queue of future Tensors to Tensor.  The queue is emptied 
/// in the process.
/// @param queue Queue of futures returning a tensor.
/// @param width Expected width dimension of returned tensor.
/// @param height Expected height dimension of returned tensor.
/// @return Tensor containing data from all futures.
tensorflow::Tensor FutureQueueToTensor(
    std::queue<std::future<tensorflow::Tensor>>* queue,
    int width,
    int height);

/// Copy a tensor to a vector of Mat.
/// @param tensor Input tensor.
/// @param vec Vector of Mat containing the tensor data.
/// @param scale Scale factor applied to the tensor data.
/// @param bias Bias applied to the tensor data.
/// @param dtype OpenCV data type.
void TensorToMatVec(
    const tensorflow::Tensor& tensor, 
    std::vector<cv::Mat>* vec,
    double scale,
    double bias,
    int dtype);

/// Does preprocessing on an image.
/// @param image Image to preprocess.
/// @param width Required width of the image.
/// @param height Required height of the image.
/// @param scale Scale factor applied to image after conversion to float.
/// @param bias Bias applied to image after scaling.
/// @return Preprocessed image as a tensor.
tensorflow::Tensor Preprocess(
    const cv::Mat& image, 
    int width, 
    int height, 
    double scale, 
    double bias);

} // namespace openem
} // namespace util

#endif // OPENEM_DEPLOY_UTIL_H_

