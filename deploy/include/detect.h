/// @file
/// @brief Interface for detecting fish in images.

#ifndef OPENEM_DEPLOY_DETECT_H_
#define OPENEM_DEPLOY_DETECT_H_

#include <memory>
#include <vector>
#include <string>

#include "image.h"
#include "error_codes.h"

namespace openem {
namespace detect {

/// Class for detecting fish in images.
class Detector {
 public:
  /// Constructor.
  Detector();

  /// Destructor.
  ~Detector();

  /// Initializes the detector.
  /// @param model_path Path to protobuf file containing model.
  /// @param gpu_fraction Fraction of GPU memory that may be allocated to
  /// this object.
  /// @return Error code.
  ErrorCode Init(const std::string& model_path, double gpu_fraction=1.0);

  /// Expected image size.  Not valid until the model has been initialized.
  /// @return Expected image size (rows, columns).
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

  /// Finds fish in batched images by performing object detection
  /// with Single Shot Detector (SSD).
  ErrorCode Process(std::vector<std::vector<Rect>>* detections);
 private:
  /// Forward declaration of implementation class.
  class DetectorImpl;

  /// Pointer to implementation.
  std::unique_ptr<DetectorImpl> impl_;
};

/// Retrieves a detection image.
///
/// Note this is not a simple crop to the detection bounding box.
/// The image extracted uses the x coordinate and width of the
/// input detection, but the y coordinate and height are adjusted
/// such that a square image is extracted.
/// @param image Image used to extract the detection.
/// @param det Detection for which image should be extracted.
Image GetDetImage(const Image& image, const Rect& det);

} // namespace detect
} // namespace openem

#endif // OPENEM_DEPLOY_DETECT_H_

