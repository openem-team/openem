/// @file
/// @brief Interface for video IO classes.
/// @copyright Copyright (C) 2018 CVision AI.
/// @license This file is part of OpenEM, released under GPLv3.
//  OpenEM is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OPENEM_DEPLOY_VIDEO_H_
#define OPENEM_DEPLOY_VIDEO_H_

#include "image.h"

namespace openem {

/// Enum specifying codec.
enum Codec {
  kRaw,  ///< No codec.
  kH264, ///< H264.
  kH265, ///< H265.
  kMjpg, ///< MJPEG.
  kMrle, ///< Microsoft RLE.
  kWmv2  ///< Windows media video 8.
};

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

  /// Returns frame rate.
  double FrameRate() const;
 private:
  /// Forward declaration of implementation class.
  class VideoReaderImpl;

  /// Pointer to implementation.
  std::unique_ptr<VideoReaderImpl> impl_;
};

/// Class for video recording.
///
/// This class is a thin wrapper around a cv::VideoWriter.  This is
/// provided to remain SWIG-friendly and for compatibility with
/// the OpenEM image class.
class VideoWriter {
 public:
  /// Constructor.
  VideoWriter();

  /// Destructor.
  ~VideoWriter();

  /// Initializes the writer with the given video path, fps, codec, and 
  /// frame size.
  /// @param video_path Full qualified path to output video file.
  /// @param fps Frame rate.
  /// @param codec Codec.
  /// @param size Size of the video.
  /// @return Error code.
  ErrorCode Init(
      const std::string& video_path, 
      double fps, 
      Codec codec, 
      const std::pair<int, int>& size);

  /// Adds a frame for writing.
  /// @param frame Video frame to be written.
  /// @return Error code.
  ErrorCode AddFrame(const Image& frame);
 private:
   /// Forward declaration of implementation class.
   class VideoWriterImpl;

   /// Pointer to implementation.
   std::unique_ptr<VideoWriterImpl> impl_;
};

} // namespace openem

#endif // OPENEM_DEPLOY_VIDEO_H_

