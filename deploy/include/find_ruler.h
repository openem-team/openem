///@file
///@brief Interface for finding rulers in images.

#ifndef OPENEM_DEPLOY_FIND_RULER_H_
#define OPENEM_DEPLOY_FIND_RULER_H_

#include <opencv2/core.hpp>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

namespace openem { namespace find_ruler {

/// Class for finding ruler masks from raw images.
class RulerMaskFinder {
public:
  /// Constructor.
  RulerMaskFinder();

  /// Initializes U-Net model and sets weights.
  /// @param model_path Path to protobuf file containing model.
  /// @return 0 on success, -1 on error.
  int Init(const std::string& model_path);

  /// Finds the ruler mask by performing segmentation with U-Net.
  /// @param image Input image for which mask will be found.
  /// @return Ruler mask.
  cv::Mat GetMask(const cv::Mat& image);
private:
  /// Tensorflow session.
  std::unique_ptr<tensorflow::Session> session_;

  /// Input image width.
  uint64_t width_;

  /// Input image height.
  uint64_t height_;
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

}} // namespace openem::find_ruler

#endif // OPENEM_DEPLOY_FIND_RULER_H_

