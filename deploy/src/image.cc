/// @file
/// @brief Implementation for image class.

#include "image.h"

#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace openem {

/// Implementation details for Image.
class Image::ImageImpl {
 public:
  /// Default constructor.
  ImageImpl();

  /// Constructor from an existing cv::Mat.
  ImageImpl(const cv::Mat& mat);

  /// Under the hood it's just a cv::Mat.
  cv::Mat mat_;
};

Image::ImageImpl::ImageImpl() : mat_() {}

Image::ImageImpl::ImageImpl(const cv::Mat& mat) : mat_(mat) {}

Image::Image() : impl_(new ImageImpl()) {}

Image::Image(Image&& other) : impl_(new ImageImpl(other.impl_->mat_)) {}

Image& Image::operator=(Image&& other) {
  impl_->mat_ = other.impl_->mat_;
  return *this;
}

Image::Image(const Image& other) 
  : impl_(new ImageImpl(other.impl_->mat_.clone())) {
}

Image& Image::operator=(const Image& other) {
  impl_->mat_ = other.impl_->mat_.clone();
  return *this;
}

Image::~Image() {}

ErrorCode Image::FromFile(const std::string& image_path, bool color) {
  if (color) {
    impl_->mat_ = cv::imread(image_path, cv::IMREAD_COLOR);
  } else {
    impl_->mat_ = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  }
  if (!impl_->mat_.data) return kErrorReadingImage;
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
  impl_->mat_.convertTo(impl_->mat_, dtype);
  Resize(width, height);
  std::memcpy(impl_->mat_.data, data.data(), data.size() * sizeof(uint8_t));
  return kSuccess;
}

const uint8_t* Image::Data() {
  return impl_->mat_.data;
}

std::vector<uint8_t> Image::DataCopy() {
  uint64_t size = Width() * Height() * Channels();
  return std::vector<uint8_t>(impl_->mat_.data, impl_->mat_.data + size);
}

int Image::Width() const {
  return impl_->mat_.cols;
}

int Image::Height() const {
  return impl_->mat_.rows;
}

int Image::Channels() const {
  return impl_->mat_.channels();
}

void Image::Resize(int width, int height) {
  cv::resize(impl_->mat_, impl_->mat_, cv::Size(width, height));
}

std::vector<double> Image::Sum() const {
  cv::Scalar sum = cv::sum(impl_->mat_);
  const int num_chan = Channels();
  std::vector<double> out(num_chan);
  for (int i = 0; i < num_chan; ++i) {
    out[i] = sum[i];
  }
  return out;
}

Image Image::operator()(const Rect& rect) const {
  Image img;
  cv::Rect r(rect[0], rect[1], rect[2], rect[3]);
  img.impl_->mat_ = impl_->mat_(r);
  img.impl_->mat_ = img.impl_->mat_.clone(); // To ensure continuous.
  return img;
}

void Image::DrawRect(
    const Rect& rect, 
    const Color& color, 
    int linewidth,
    const std::vector<double>& transform,
    const Rect& roi) {
  cv::Scalar c(color[0], color[1], color[2]);
  cv::Mat t(2, 3, CV_64F);
  std::memcpy(t.data, transform.data(), transform.size() * sizeof(double));
  cv::invertAffineTransform(t, t);
  double x = static_cast<double>(rect[0]);
  double y = static_cast<double>(rect[1]);
  double w = static_cast<double>(rect[2]);
  double h = static_cast<double>(rect[3]);
  double x0 = x + roi[0];
  double x1 = x0 + w;
  double y0 = y + roi[1];
  double y1 = y0 + h;
  std::vector<cv::Point2d> v(4);
  v[0] = cv::Point2d(x0, y0);
  v[1] = cv::Point2d(x0, y1);
  v[2] = cv::Point2d(x1, y1);
  v[3] = cv::Point2d(x1, y0);
  cv::transform(v, v, t);
  for (int i = 0; i < 4; ++i) {
    cv::line(impl_->mat_, v[i], v[(i+1)%4], c, linewidth);
  }
}

void Image::DrawText(
    const std::string& text,
    const std::pair<int, int>& loc,
    const Color& color,
    double scale) {
  cv::putText(
      impl_->mat_,
      text,
      cv::Point2d(loc.first, loc.second),
      cv::FONT_HERSHEY_SIMPLEX,
      scale,
      cv::Scalar(color[0], color[1], color[2]));
}

void Image::Show(const std::string& window_name) {
  cv::imshow(window_name, impl_->mat_);
  cv::waitKey(0);
}

void* Image::MatPtr() {
  return &(impl_->mat_);
}

} // namespace openem

