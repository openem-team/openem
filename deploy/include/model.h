/// @file
/// @brief Interface for tensorflow models.

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

  /// Maximum image batch size.  AddImage may only be called this 
  /// many times before a call to Process is required, otherwise
  /// AddImage will return an error.
  /// @return Maximum image batch size.
  int MaxImages();

  /// Input image size.  Images that are added without having this size
  /// will be resized.  This only returns a valid value if the model
  /// has been initialized.
  cv::Size ImageSize();

  /// Adds an image to batch for processing.  This function launches 
  /// a new thread to do image preprocessing and immediately returns.
  /// The input image is assumed to be 8-bit, 3-channel with colors 
  /// in the default channel order for OpenCV, which is BGR.  It must
  /// also have continuous storage, i.e. image.isContinuous() returns
  /// true.
  /// @param image Input image for which mask will be found.
  /// @param preprocess Function object to do preprocessing.
  /// @return Error code.
  ErrorCode AddImage(
      const cv::Mat& image,
      std::function<tensorflow::Tensor(const cv::Mat&, int, int)> preprocess);

  /// Processes the model on the current batch.
  /// @param outputs Output of the model.
  /// @return Error code.
  ErrorCode Process(std::vector<tensorflow::Tensor>* outputs);
 private:
  /// Tensorflow session.
  std::unique_ptr<tensorflow::Session> session_;

  /// Input image width.
  int width_;

  /// Input image height.
  int height_;

  /// Batch size, computed from available memory.
  int batch_size_;

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

