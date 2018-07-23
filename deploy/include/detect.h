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

  /// Maximum image batch size.  AddImage may only be called this 
  /// many times before a call to Process is required, otherwise
  /// AddImage will return an error.
  /// @return Maximum image batch size.
  int MaxImages();

  /// Expected image size.  Not valid until the model has been initialized.
  /// @return Expected image size (rows, columns).
  std::pair<int, int> ImageSize();

  /// Adds an image to batch for processing.  This function launches 
  /// a new thread to do image preprocessing and immediately returns.
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

} // namespace detect
} // namespace openem

#endif // OPENEM_DEPLOY_DETECT_H_

