/// @file
/// @brief Interface for video IO classes.

#ifndef OPENEM_DEPLOY_VIDEO_H_
#define OPENEM_DEPLOY_VIDEO_H_

#include "image.h"

namespace openem {

/// Class for video playback.
///
/// This class is a thin wrapper around a cv::VideoCapture.  This is
/// provided to remain SWIG-friendly and for compatibility with
/// the OpenEM image class.
class VideoReader {
 public:
  /// Constructor.
  VideoReader();

  /// Destructor.
  ~VideoReader();

  /// Initializes the reader with the given video.
  /// @param video_path Full qualified path to video file.
  ErrorCode Init(const std::string& video_path);

  /// Gets the next video frame.
  /// @param frame Next video frame.
  /// @return Error code.
  ErrorCode GetFrame(Image* frame);

  /// Returns video width.
  int Width() const;

  /// Returns video height.
  int Height() const;
 private:
  /// Forward declaration of implementation class.
  class VideoReaderImpl;

  /// Pointer to implementation.
  std::unique_ptr<VideoReaderImpl> impl_;
};

} // namespace openem

#endif // OPENEM_DEPLOY_VIDEO_H_
