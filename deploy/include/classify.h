/// @file
/// @brief Interface for classifying cropped images of fish.

#ifndef OPENEM_DEPLOY_CLASSIFY_H_
#define OPENEM_DEPLOY_CLASSIFY_H_

#include <memory>
#include <vector>
#include <string>

#include "image.h"
#include "error_codes.h"

namespace openem {
namespace classify {

/// Class for determining fish species and whether the image is covered
/// by a hand, clear, or not a fish.
class Classifier {
 public:
  /// Constructor.
  Classifier();

  /// Destructor.
  ~Classifier();

  /// Initializes the classifier.
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
  /// @return Expected image size.
  std::pair<int, int> ImageSize();

  /// Adds an image to batch for processing.  This function launches 
  /// a new thread to do image preprocessing and immediately returns.
  /// @param image Input image that will be classified.
  /// @return Error code.
  ErrorCode AddImage(const Image& image);

  /// Determines fish species and whether the fish is covered by a hand,
  /// clear, or not a fish.
  /// @param scores Vector of double vectors.  Each double vector
  /// corresponds to one of the images in the image queue.  The first
  /// three numbers in the double vector correspond to:
  /// * If the fish is covered.
  /// * If the fish is not covered.
  /// * If the fish is actually not a fish.
  /// The remaining vector elements correspond to the species used to train
  /// the loaded model.
  ErrorCode Process(std::vector<std::vector<float>>* scores);
 private:
  /// Forward declaration of implementation class.
  class ClassifierImpl;

  /// Pointer to implementation.
  std::unique_ptr<ClassifierImpl> impl_;
}; 

} // namespace openem
} // namespace openem

#endif // OPENEM_DEPLOY_CLASSIFY_H_

