/// @file
/// @brief Implementation for video IO classes.
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

#include "video.h"

#include <opencv2/videoio.hpp>
#include "detail/util.h"

namespace openem {

/// Implementation details for VideoReader.
class VideoReader::VideoReaderImpl {
 public:
  /// Constructor.
  VideoReaderImpl();

  /// Under the hood it's just a cv::VideoCapture.
  cv::VideoCapture cap_;
};

/// Implementation details for VideoWriter.
class VideoWriter::VideoWriterImpl {
 public:
  /// Constructor.
  VideoWriterImpl();

  /// Video writer doesn't store the size.
  cv::Size size_;

  /// Under the hood it's just a cv::VideoWriter.
  cv::VideoWriter writer_;
};

VideoReader::VideoReaderImpl::VideoReaderImpl() : cap_() {}

VideoReader::VideoReader() : impl_(new VideoReaderImpl()) {}

VideoReader::~VideoReader() {}

ErrorCode VideoReader::Init(const std::string& video_path) {
  bool ok = impl_->cap_.open(video_path);
  if (!ok) return kErrorVidFileOpen;
  return kSuccess;
}

ErrorCode VideoReader::GetFrame(Image* frame) {
  if (!impl_->cap_.isOpened()) return kErrorVidNotOpen;
  cv::Mat* mat = detail::MatFromImage(frame);
  bool ok = impl_->cap_.read(*mat);
  if (!ok) return kErrorVidFrame;
  return kSuccess;
}

int VideoReader::Width() const {
  return static_cast<int>(impl_->cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoReader::Height() const {
  return static_cast<int>(impl_->cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double VideoReader::FrameRate() const {
  return impl_->cap_.get(cv::CAP_PROP_FPS);
}

VideoWriter::VideoWriterImpl::VideoWriterImpl() : writer_() {}

VideoWriter::VideoWriter() : impl_(new VideoWriterImpl()) {}

VideoWriter::~VideoWriter() {}

ErrorCode VideoWriter::Init(
    const std::string& video_path, 
    double fps, 
    Codec codec, 
    const std::pair<int, int>& size) {
  int fourcc = 0;
  switch (codec) {
    case kH264:
      fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
      break;
    case kH265:
      fourcc = cv::VideoWriter::fourcc('H', '2', '6', '5');
      break;
    case kMjpg:
      fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      break;
    case kMrle:
      fourcc = cv::VideoWriter::fourcc('M', 'R', 'L', 'E');
      break;
    case kWmv2:
      fourcc = cv::VideoWriter::fourcc('W', 'M', 'V', '2');
  }
  impl_->size_ = cv::Size(size.first, size.second);
  bool ok = impl_->writer_.open(video_path, fourcc, fps, impl_->size_);
  if (!ok) return kErrorVidFileOpen;
  return kSuccess;
}

ErrorCode VideoWriter::AddFrame(const Image& frame) {
  const cv::Mat* mat = detail::MatFromImage(&frame);
  if (impl_->size_ != mat->size()) return kErrorVidFrame;
  impl_->writer_.write(*mat);
  return kSuccess;
}

} // namespace openem

