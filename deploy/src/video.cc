/// @file
/// @brief Implementation for video IO classes.

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

} // namespace openem

