/// @file
/// @brief Implementation for finding rulers in images.

#include "find_ruler.h"

#include <future>
#include <queue>
#include <mutex>

#include <opencv2/imgproc.hpp>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/array_ops.h>
#include "util.h"

namespace openem {
namespace find_ruler {

namespace tf = tensorflow;

namespace {

/// Does preprocessing on an image.
/// @param image Image to preprocess.
/// @param width Required width of the image.
/// @param height Required height of the image.
/// @return Preprocessed image.
tf::Tensor Preprocess(const cv::Mat& image, int width, int height);

} // namespace

/// Implementation details for RulerMaskFinder.
class RulerMaskFinder::RulerMaskFinderImpl {
 public:
  /// Constructor.
  RulerMaskFinderImpl();

  /// Tensorflow session.
  std::unique_ptr<tf::Session> session_;

  /// Input image width.
  int width_;

  /// Input image height.
  int height_;

  /// Batch size, computed from available memory.
  int batch_size_;

  /// Indicates whether the model has been initialized.
  bool initialized_;

  /// Queue of futures containing preprocessed images.
  std::queue<std::future<tf::Tensor>> preprocessed_;

  /// Mutex for handling concurrent access to image queue.
  std::mutex mutex_;
};

//
// Implementations
//

namespace {

tf::Tensor Preprocess(
    const cv::Mat& image, 
    int width, 
    int height) {

  // Start by resizing the image if necessary.
  cv::Mat p_image;
  if ((image.rows != height) || (image.cols != width)) {
    cv::resize(image, p_image, cv::Size(width, height));
  } else {
    p_image = image.clone();
  }

  // Convert to RGB as required by the model.
  cv::cvtColor(p_image, p_image, CV_BGR2RGB);

  // Do image scaling.
  p_image.convertTo(p_image, CV_32F, 1.0 / 128.0, -1.0);

  // Copy into tensor.
  return util::MatToTensor(p_image);
}

} // namespace

RulerMaskFinder::RulerMaskFinderImpl::RulerMaskFinderImpl()
    : session_(nullptr),
      width_(0),
      height_(0),
      batch_size_(64),
      initialized_(false),
      mutex_() {
}

RulerMaskFinder::RulerMaskFinder() : impl_(new RulerMaskFinderImpl()) {}

RulerMaskFinder::~RulerMaskFinder() {}

ErrorCode RulerMaskFinder::Init(
    const std::string& model_path, double gpu_fraction) {
  impl_->initialized_ = false;

  // Read in the graph.
  tf::GraphDef graph_def;
  tf::Status status = tf::ReadBinaryProto(
      tf::Env::Default(), model_path, &graph_def);
  if (!status.ok()) return kErrorLoadingModel;
  
  // Get graph input size.
  ErrorCode status1 = util::InputSize(
      graph_def, &(impl_->width_), &(impl_->height_));
  if (status1 != kSuccess) return status1;

  // Create a new tensorflow session.
  tf::Session* session;
  status1 = util::GetSession(&session, gpu_fraction);
  if (status1 != kSuccess) return status1;
  impl_->session_.reset(session);

  // Create the tensorflow graph.
  status = impl_->session_->Create(graph_def);
  if (!status.ok()) return kErrorTfGraph;
  impl_->initialized_ = true;
  return kSuccess;
}

int RulerMaskFinder::MaxImages() {
  return impl_->batch_size_;
}

ErrorCode RulerMaskFinder::AddImage(const cv::Mat& image) {
  if (!impl_->initialized_) return kErrorBadInit;
  if (!image.isContinuous()) return kErrorNotContinuous;
  if (impl_->preprocessed_.size() >= MaxImages()) return kErrorMaxBatchSize;
  auto f = std::async(
      Preprocess, 
      image, 
      impl_->width_, 
      impl_->height_);
  impl_->mutex_.lock();
  impl_->preprocessed_.push(std::move(f));
  impl_->mutex_.unlock();
  return kSuccess;
}

ErrorCode RulerMaskFinder::Process(std::vector<cv::Mat>* masks) {
  if (!impl_->initialized_) return kErrorBadInit;

  // Copy image queue contents into input tensor.
  impl_->mutex_.lock();
  tf::Tensor input = util::FutureQueueToTensor(
      &(impl_->preprocessed_), 
      impl_->width_, 
      impl_->height_);
  impl_->mutex_.unlock();

  // Run the model.
  std::vector<tf::Tensor> outputs;
  tf::Status status = impl_->session_->Run(
      {{"input_1", input}}, 
      {"output_node0:0"}, 
      {},
      &outputs);
  if (!status.ok()) return kErrorRunSession;

  // Copy model outputs into mask images.
  util::TensorToMatVec(outputs.back(), masks, 255.0, 0.0);

  // Do additional processing on the masks.
  for (auto& mask : *masks) {
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    cv::medianBlur(mask, mask, 5);
  }
  return kSuccess;
}

bool RulerPresent(const cv::Mat& mask) {
  return cv::sum(mask)[0] > 1000.0;
}

cv::Mat RulerOrientation(const cv::Mat& mask) {
  // Find center of rotation of the mask.
  cv::Moments m = cv::moments(mask);
  double centroid_x = m.m10 / m.m00;
  double centroid_y = m.m01 / m.m00;
  cv::Point2f centroid(centroid_x, centroid_y);

  // Find transform to translate image to center of rotation.
  double center_x = static_cast<double>(mask.cols) / 2.0;
  double center_y = static_cast<double>(mask.rows) / 2.0;
  cv::Point2f center(center_x, center_y);
  double diff_x = center_x - centroid_x;
  double diff_y = center_y - centroid_y;
  double t[3][3] = {
      {1.0, 0.0, diff_x}, 
      {0.0, 1.0, diff_y}, 
      {0.0, 0.0, 1.0}};
  cv::Mat t_matrix(3, 3, CV_64F, t);
  cv::Mat row = t_matrix.row(2);

  // Rotate image, saving off transform with smallest second central moment.
  double min_moment = 1e99;
  cv::Mat rotated, best;
  for (double ang = -90.0; ang < 90.0; ang += 1.0) {
    cv::Mat r_matrix = cv::getRotationMatrix2D(centroid, ang, 1.0);
    cv::vconcat(r_matrix, row, r_matrix);
    r_matrix = t_matrix * r_matrix;
    r_matrix = r_matrix.rowRange(0, 2);
    cv::warpAffine(mask, rotated, r_matrix, mask.size());
    cv::Moments moments = cv::moments(rotated);
    if (moments.mu02 < min_moment) {
      best = r_matrix.clone();
      min_moment = moments.mu02;
    }
  }
  return best;
}

cv::Mat Rectify(const cv::Mat& image, const cv::Mat& transform) {
  cv::Mat r_image;
  cv::warpAffine(image, r_image, transform, image.size());
  return r_image;
}

cv::Rect FindRoi(const cv::Mat& mask, int h_margin) {
  cv::Rect roi = cv::boundingRect(mask);

  // Find unconstrained bounding rect.
  double aspect = 
    static_cast<double>(mask.rows) / 
    static_cast<double>(mask.cols);
  double v_center = 
    static_cast<double>(roi.y) + static_cast<double>(roi.height) / 2;
  double x0, x1, y0, y1;
  x0 = roi.x - h_margin;
  x1 = roi.x + roi.width + h_margin;
  double v_width = (x1 - x0) * aspect / 2.0;
  y0 = v_center - v_width;
  y1 = v_center + v_width;

  // Constrain the bounding rect.
  x0 = x0 >= 0 ? x0 : 0;
  x1 = x1 < mask.cols ? x1 : mask.cols - 1;
  y0 = y0 >= 0 ? y0 : 0;
  y1 = y1 < mask.rows ? y1 : mask.rows - 1;

  // Convert to roi.
  roi.x = std::lround(x0);
  roi.width = std::lround(x1 - x0 + 1);
  roi.y = std::lround(y0);
  roi.height = std::lround(y1 - y0 + 1);
  return roi;
}

cv::Mat Crop(const cv::Mat& image, const cv::Rect& roi) {
  return image(roi);
}

} // namespace find_ruler
} // namespace openem

