/// @file
/// @brief Interface for finding rulers in images.

#ifndef OPENEM_DEPLOY_FIND_RULER_H_
#define OPENEM_DEPLOY_FIND_RULER_H_

#include <memory>
#include <vector>
#include <string>

#include "image.h"
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
  /// @param gpu_fraction Fraction of GPU allowed to be used by this
  /// object.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, double gpu_fraction=1.0);

  /// Expected image size.  Not valid until the model has been initialized.
  /// @return Expected image size (width, height).
  std::pair<int, int> ImageSize();

  /// Adds an image to batch for processing.  
  ///
  /// This function launches a new thread to do image preprocessing 
  /// and immediately returns.  Each call to this function consumes 
  /// additional GPU memory which is cleared when Process is called.  
  /// All images are processed as one batch, so for speed it is 
  /// recommended to call AddImage as many times as possible before 
  /// Process without exceeding available GPU memory.  It is the 
  /// responsibility of the caller to know an appropriate batch size 
  /// for their processing hardware.
  /// @param image Input image for which mask will be found.
  /// @return Error code.
  ErrorCode AddImage(const Image& image);

  /// Finds the ruler mask on batched images by performing 
  /// segmentation with U-Net.  The output parameter is cleared,
  /// then filled with one segmentation mask per image previously
  /// added with AddImage.  Each image has the same size as is
  /// returned from the ImageSize function.  Values of 255 indicate pixels
  /// with ruler present.  The size of the mask is determined by 
  /// the loaded model; images are resized appropriately when they
  /// are preprocessed.
  /// @param masks Output masks.
  /// @return Error code.
  ErrorCode Process(std::vector<Image>* masks);
 private:
  /// Forward declaration of implementation class.
  class RulerMaskFinderImpl;

  /// Pointer to implementation class.
  std::unique_ptr<RulerMaskFinderImpl> impl_;
};

/// Determines if a ruler is present in a mask.
/// @param mask Mask image.
/// @return True if ruler is present, false otherwise.
bool RulerPresent(const Image& mask);

/// Finds ruler orientation in a mask.
/// @param mask Mask image.
/// @return Affine transformation matrix.
std::vector<double> RulerOrientation(const Image& mask);

/// Rectifies an image by applying specified rotation.  This is
/// meant to make the ruler horizontal in the image.
/// @param image Image to be rectified.  May be a mask image.
/// @param transform Transform found with RulerOrientation.
/// @return Rectified image.
Image Rectify(const Image& image, const std::vector<double>& transform);

/// Finds region of interest (ROI) on a rectified mask image.  The ROI
/// will retain the same aspect ratio as the original image.  Size of 
/// the ROI is parameterized by the horizontal margin from the edge of
/// the ruler.
/// @param mask Rectified mask image.
/// @param h_margin Horizontal margin from edge of ruler.
/// @return Region of interest.
Rect FindRoi(const Image& mask, int h_margin=0);

/// Crops an image using the specified ROI.
/// @param image Rectified image.
/// @param roi Region of interest found with FindRoi.
/// @return Cropped image.
Image Crop(const Image& image, const Rect& roi);

} // namespace find_ruler
} // namespace openem

#endif // OPENEM_DEPLOY_FIND_RULER_H_

