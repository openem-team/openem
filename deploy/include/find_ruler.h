/// @file
/// @brief Interface for finding rulers in images.

#ifndef OPENEM_DEPLOY_FIND_RULER_H_
#define OPENEM_DEPLOY_FIND_RULER_H_

#include <memory>
#include <functional>
#include <future>
#include <queue>
#include <mutex>

#include <opencv2/core.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "error_codes.h"

namespace openem {
namespace find_ruler {

/// Function signature for user-defined callback.
using UserCallback = std::function<void(const std::vector<cv::Mat>&)>;

/// Class for finding ruler masks from raw images.
class RulerMaskFinder {
 public:
  /// Constructor.
  RulerMaskFinder();

  /// Initializes U-Net model and registers user-defined callback.
  /// @param model_path Path to protobuf file containing model.
  /// @param callback Pointer to function that will execute whenever
  /// processing of an image batch completes.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, UserCallback callback);

  /// Maximum image batch size.  AddImage may only be called this 
  /// many times before a call to Process is required, otherwise
  /// AddImage will return an error.
  /// @return Maximum image batch size.
  int MaxImages();

  /// Adds an image to batch for processing.  This function launches 
  /// a new thread to do image preprocessing and immediately returns.
  /// The input image is assumed to be 8-bit, 3-channel with colors 
  /// in the default channel order for OpenCV, which is BGR.  It must
  /// also have continuous storage, i.e. image.isContinuous() returns
  /// true.
  /// @param image Input image for which mask will be found.
  /// @return Error code.
  ErrorCode AddImage(const cv::Mat& image);

  /// Finds the ruler mask on batched images by performing 
  /// segmentation with U-Net.  This function launches a new thread 
  /// to execute the model and immediately returns.  The user-defined
  /// callback registered by Init is called once the thread completes.
  /// @return Error code.
  ErrorCode Process();
 private:
  /// Tensorflow session.
  std::unique_ptr<tensorflow::Session> session_;

  /// Input image width.
  uint64_t width_;

  /// Input image height.
  uint64_t height_;

  /// Indicates whether the model has been initialized.
  bool initialized_;

  /// Queue of futures containing preprocessed images.
  std::queue<std::future<tensorflow::Tensor>> preprocessed_;

  /// Mutex for handling concurrent access to image queue.
  std::mutex mutex_;

  /// User defined callback, executed when Process completes.
  UserCallback callback_;
};

/// Determines if a ruler is present in a mask.  If so, finds
/// the orientation of the ruler and stores it for subsequent
/// rectification.
/// @param mask Mask image.
/// @return True if ruler is present, false otherwise.
bool RulerPresent(const cv::Mat& mask);

/// Finds ruler orientation in a mask.
/// @param mask Mask image.
/// @return Orientation angle in degrees.  Zero corresponds to 
/// horizontal, positive values are clockwise rotation from zero.
double RulerOrientation(const cv::Mat& mask);

/// Rectifies an image by applying specified rotation.  This is
/// meant to make the ruler horizontal in the image.
/// @param image Image to be rectified.  May be a mask image.
/// @param orientation Orientation found with RulerOrientation.
/// @return Rectified image.
cv::Mat Rectify(const cv::Mat& image, const double orientation);

/// Finds region of interest (ROI) on a rectified mask image.
/// @param mask Rectified mask image.
/// @param h_margin Horizontal margin from edge of ruler.
/// @param v_margin Vertical margin from edge of ruler.
/// @return Region of interest.
cv::Rect FindRoi(const cv::Mat& mask, int h_margin=0, int v_margin=200);

/// Crops an image using the specified ROI.
/// @param image Rectified image.
/// @param roi Region of interest found with FindRoi.
/// @return Cropped image.
cv::Mat Crop(const cv::Mat& image, const cv::Rect& roi);

} // namespace find_ruler
} // namespace openem

#endif // OPENEM_DEPLOY_FIND_RULER_H_

