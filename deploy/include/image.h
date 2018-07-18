/// @file
/// @brief Interface for image class.

#ifndef OPENEM_DEPLOY_IMAGE_H_
#define OPENEM_DEPLOY_IMAGE_H_

#include <string>
#include <vector>
#include <memory>

#include "error_codes.h"

namespace openem {

/// @brief Class for holding image data and dimensions.
///
/// This class is a thin wrapper around a cv::Mat.  We avoid using
/// cv::Mat directly for two reasons:
/// 1) It makes generating bindings difficult.
/// 2) We do not want to force application code to use OpenCV.  OpenCV
///    can be statically linked into an OpenEM shared library, allowing
///    it to be used as a standalone dependency by an application.
class Image {
 public:
  /// Constructor.  Creates an empty image container.
  Image();

  /// Move constructor.
  Image(Image&& other);

  /// Move assignment operator.
  Image& operator=(Image&& other);

  /// Copy constructor.
  Image(const Image& other);

  /// Copy assignment operator.
  Image& operator=(const Image& other);

  /// Destructor.
  ~Image();

  /// Loads an image file.
  /// @param image_path Path to image file.
  /// @param color If true, load a color image.
  ErrorCode FromFile(const std::string& image_path, bool color=true);

  /// Creates an image from existing data.  Data is copied.
  ///
  /// This function will check to make sure the size of data 
  /// is appropriate for the given width, height and number of
  /// channels.  If it is not, an error code will be returned.
  /// Data must have a memory layout such that the address of 
  /// element (r, c, ch) is computed as:
  /// data.data() + (r * width * channels) + (c * channels) + ch
  /// So data is stored row by column by channel, with channel
  /// being the fastest changing index.  This layout is compatible
  /// with OpenCV Mats, Numpy ndarrays, Win32 independent device bitmaps,
  /// and other dense array types.  For color images, the channel
  /// order must match the OpenCV default, which is BGR.
  ErrorCode FromData(
      const std::vector<uint8_t>& data, 
      int width, 
      int height, 
      int channels);

  /// Returns pointer to image data.
  const uint8_t* Data();

  /// Returns image width.
  int Width();

  /// Returns image height.
  int Height();

  /// Returns number of image channels.
  int Channels();

  /// Resizes the image to the specified width and height.
  void Resize(int width, int height);

  /// Returns pointer that can be can be converted to a pointer to the 
  /// underlying cv::Mat via reinterpret_cast.
  ///
  /// This is intended for internal use by other OpenEM implementations or
  /// by applications that use OpenCV to avoid unnecessarily copying 
  /// data.
  void* MatPtr();
 private:
  /// Forward declaration of implementation class.
  class ImageImpl;

  /// Pointer to implementation.
  std::unique_ptr<ImageImpl> impl_;
};

} // namespace openem

#endif // OPENEM_DEPLOY_IMAGE_H_

