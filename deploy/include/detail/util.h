/// @file
/// @brief Interface for utility functions.
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

#ifndef OPENEM_DEPLOY_UTIL_H_
#define OPENEM_DEPLOY_UTIL_H_

#include <queue>
#include <future>

#include <opencv2/core.hpp>
#include <tensorflow/core/public/session.h>
#include "image.h"
#include "error_codes.h"

namespace openem {
namespace detail {

/// Gets pointer to cv::Mat from Image.
/// @param image Pointer to an image.
/// @return Pointer to underlying cv::Mat.
cv::Mat* MatFromImage(Image* image);

/// Gets const pointer to cv::Mat from Image.
/// @param image Pointer to an image.
/// @return Pointer to underlying cv::Mat.
const cv::Mat* MatFromImage(const Image* image);

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

/// Copy an Image to a Tensor.
/// @param image Image to be converted.
/// @param shape Shape of the output tensor.
/// @return Tensor containing mat data.
tensorflow::Tensor ImageToTensor(
    const Image& image, 
    const tensorflow::TensorShape& shape);

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

/// Copy a tensor to a vector of Image.
/// @param tensor Input tensor.
/// @param vec Vector of Image containing the tensor data.
/// @param scale Scale factor applied to the tensor data.
/// @param bias Bias applied to the tensor data.
/// @param dtype OpenCV data type.
void TensorToImageVec(
    const tensorflow::Tensor& tensor, 
    std::vector<Image>* vec,
    double scale,
    double bias,
    int dtype);

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
/// @param image Image to preprocess.  Channels are BGR.
/// @param width Required width of the image.
/// @param height Required height of the image.
/// @param scale Scale factor applied to image after conversion to float.
/// @param bias Bias applied to image after scaling.
/// @param rgb True if image needs to be converted to RGB.
/// @return Preprocessed image as a tensor.
tensorflow::Tensor Preprocess(
    const cv::Mat& image, 
    int width, 
    int height, 
    double scale, 
    const cv::Scalar& bias,
    bool rgb);

} // namespace detail 
} // namespace openem

#endif // OPENEM_DEPLOY_UTIL_H_

