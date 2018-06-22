/// @file
/// @brief Interface for finding rulers in images.

#ifndef OPENEM_DEPLOY_FIND_RULER_H_
#define OPENEM_DEPLOY_FIND_RULER_H_

#include <memory>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include "error_codes.h"

namespace openem {
namespace find_ruler {

/// Class for finding ruler masks from raw images.
class RulerMaskFinder {
 public:
  /// Constructor.
  RulerMaskFinder();

  /// Destructor.
  ~RulerMaskFinder();

  /// Initializes the mask finder.
  /// @param model_path Path to protobuf file containing model.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path);

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
  /// segmentation with U-Net.  The output parameter masks is cleared,
  /// then filled with one segmentation mask per image previously
  /// added with AddImage.  Each mask has data type CV_32F with 
  /// values between 0.0 and 1.0.  Values near 1.0 indicate pixels
  /// with ruler present.  The size of the mask is determined by 
  /// the loaded model; images are resized appropriately when they
  /// are preprocessed.
  /// @param masks Output masks.
  /// @return Error code.
  ErrorCode Process(std::vector<cv::Mat>* masks);
 private:
  /// Forward declaration of implementation class.
  class RulerMaskFinderImpl;

  /// Pointer to implementation class.
  std::unique_ptr<RulerMaskFinderImpl> impl_;
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

