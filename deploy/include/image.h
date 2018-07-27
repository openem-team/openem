/// @file
/// @brief Interface for image class.

#ifndef OPENEM_DEPLOY_IMAGE_H_
#define OPENEM_DEPLOY_IMAGE_H_

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "error_codes.h"

namespace openem {

/// Type for storing x, y, w, h.
using Rect = std::array<int, 4>;

/// Type for storing RGB color.
using Color = std::array<uint8_t, 3>;

/// Class for holding image data.
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

  /// Returns copy of image data.
  std::vector<uint8_t> DataCopy();

  /// Returns image width.
  int Width() const;

  /// Returns image height.
  int Height() const;

  /// Returns number of image channels.
  int Channels() const;

  /// Resizes the image to the specified width and height.
  void Resize(int width, int height);

  /// Returns sum of the image.
  /// @return Image sum.  Size is equal to number of channels.
  std::vector<double> Sum() const;

  /// Gets a subimage specified by the give Rect.
  /// @param rect Rect for which subimage is requested.
  Image operator()(const Rect& rect) const;

  /// Draws a rectangle on top of the image.
  /// @param rect Rectangle to draw.
  /// @param color Color of the rectangle.
  /// @param linewidth Width of the line used to draw rectangle.
  /// @param transform Transform from find_ruler::RulerOrientation.  Use
  /// this to draw a rect on the original image where a ruler was found.
  /// @param roi ROI from find_ruler::FindRoi.  Use this to draw a rect
  /// on the original image where a ruler was found.
  void DrawRect(
      const Rect& rect, 
      const Color& color={0, 0, 255}, 
      int linewidth=2,
      const std::vector<double>& transform={1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      const Rect& roi={0, 0, 0, 0});

  /// Draws text in the image.
  /// @param text Text to write.
  /// @param loc Bottom left text corner location on the image.
  /// @param color Color of the text.
  void DrawText(
      const std::string& text,
      const std::pair<int, int>& loc={0, 0},
      const Color& color={0, 0, 255},
      double scale=10.0);

  /// Displays the image in a named window.
  ///
  /// Window will be displayed until user closes it.
  /// @param window_name Name of the window.
  void Show(const std::string& window_name="");

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

