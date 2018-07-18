/// @file
/// @brief Contains definitions of error codes.

#ifndef OPENEM_DEPLOY_ERROR_CODES_H_
#define OPENEM_DEPLOY_ERROR_CODES_H_

namespace openem {

/// All openem error codes.
enum ErrorCode {
  kSuccess = 0,        ///< No error.
  kErrorLoadingModel,  ///< Failed to read tensorflow model.
  kErrorGraphDims,     ///< Wrong number of dimensions in graph input node.
  kErrorNoShape,       ///< No shape attribute in graph input node.
  kErrorTfSession,     ///< Failed to create tensorflow session.
  kErrorTfGraph,       ///< Failed to create tensorflow graph.
  kErrorBadInit,       ///< Attempted to use uninitialized object.
  kErrorNotContinuous, ///< cv::Mat expected to be continuous.
  kErrorRunSession,    ///< Failed to run session.
  kErrorMaxBatchSize,  ///< Exceeded max batch size.
  kErrorReadingImage,  ///< Failed to read image.
  kErrorImageSize,     ///< Invalid image data size for given dimensions.
  kErrorNumChann       ///< Invalid number of image channels.
};

} // namespace openem

#endif // OPENEM_DEPLOY_ERROR_CODES_H_

