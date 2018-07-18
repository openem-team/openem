/// @file
/// @brief Implementation for image class.

#include <opencv2/core.hpp>

namespace openem {

/// Implementation details for Image.
class Image::ImageImpl {
 public:
  /// Constructor from an existing cv::Mat.
  ImageImpl(const cv::Mat& mat);

  /// Under the hood it's just a cv::Mat.
  cv::Mat mat_;
};

Image::ImageImpl(const cv::Mat& mat) : mat_(mat) {}

Image::Image() : impl_(new ImageImpl()) {}

Image::Image(Image&& other) : impl_(new ImageImpl(other->impl_.mat_)) {}

Image& Image::operator(Image&& other) {
  impl_->mat_ = other.impl_->mat_;
}

Image::Image(const Image& other) 
  : impl_(new ImageImpl(other->impl_.mat_.clone())) {
}

Image& Image::operator=(const Image& other) {
  impl_->mat_ = other.impl_->mat_.clone();
}

Image Image::operator() 

Image::~Image() {}

ErrorCode Image::FromFile(const std::string& image_path) {
  impl_->mat_ = cv::imread(image_path);
  if (!impl_->mat_->data) return kErrorReadingImage;
  return kSuccess;
}

ErrorCode Image::FromData(
    const std::vector<uint8_t>& data, 
    int width, 
    int height, 
    int channels) {
  if (data.size() != width * height * channels) return kErrorImageSize;
  int dtype;
  if (channels == 1) {
    dtype = CV_8UC1;
  } else if (channels == 3) {
    dtype = CV_8UC3;
  } else {
    return kErrorNumChann;
  }
  impl_->mat_.resize(height, width, dtype);
  std::memcpy(impl_->mat_.data, data.data(), data.size() * sizeof(uint8_t));
}

const uint8_t* Image::Data() {
  return impl_->mat_.data;
}

int Image::Width() {
  return impl_->mat_.cols;
}

int Image::Height() {
  return impl_->mat_.rows;
}

int Image::Channels() {
  return impl_->mat_.channels();
}

void* Image::MatPtr() {
  return &(impl_->mat_);
}

} // namespace openem

