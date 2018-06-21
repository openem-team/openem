/// @file
/// @brief Implementation for finding rulers in images.

#include "find_ruler.h"

#include <opencv2/imgproc.hpp>

namespace openem { namespace find_ruler {

namespace tf = tensorflow;

namespace {

/// Does preprocessing on an image.
/// @param image Image to preprocess.
/// @param width Required width of the image.
/// @param height Required height of the image.
/// @return Preprocessed image.
tf::Tensor Preprocess(
    const cv::Mat& image, 
    int width, 
    int height) {

  // Start by resizing the image if necessary.
  cv::Mat p_image;
  if ((image.rows != height) || (image.cols != width)) {
    cv::resize(image, p_image, cv::Size(width, height));
  }
  else {
    p_image = image.clone();
  }

  // Convert to RGB as required by the model.
  cv::cvtColor(p_image, p_image, CV_BGR2RGB);

  // Do image scaling.
  p_image.convertTo(p_image, CV_32F, 1.0 / 128.0, -1.0);

  // Copy into tensor.
  tf::Tensor tensor(tf::DT_FLOAT, tf::TensorShape({1, height, width, 3}));
  auto flat = tensor.flat<float>();
  std::copy_n(p_image.ptr<float>(), p_image.total(), flat.data());
  return tensor;
}

} // namespace

RulerMaskFinder::RulerMaskFinder() 
    : session_(nullptr),
      width_(0),
      height_(0),
      initialized_(false),
      callback_(nullptr) {
}

ErrorCode RulerMaskFinder::Init(
    const std::string& model_path, 
    UserCallback callback) {
  initialized_ = false;

  // Read in the graph.
  tf::GraphDef graph_def;
  tf::Status status = tf::ReadBinaryProto(
      tf::Env::Default(), model_path, &graph_def);
  if (!status.ok()) return kErrorLoadingModel;

  // Find the input shape based on graph definition.  For now we avoid
  // using protobuf map functions so we don't have to link with 
  // protobuf.
  bool found = false;
  for (auto p : graph_def.node(0).attr()) {
    if (p.first == "shape") {
      found = true;
      auto shape = p.second.shape();
      if (shape.dim_size() != 4) return kErrorGraphDims;
      width_ = shape.dim(2).size();
      height_ = shape.dim(1).size();
    }
  }
  if (!found) return kErrorNoShape;

  // Create a new tensorflow session.
  tf::Session* session;
  status = tf::NewSession(tf::SessionOptions(), &session);
  if (!status.ok()) return kErrorTfSession;
  session_.reset(session);

  // Create the tensorflow graph.
  status = session_->Create(graph_def);
  if (!status.ok()) return kErrorTfGraph;
  initialized_ = true;
  return kSuccess;
}

ErrorCode RulerMaskFinder::AddImage(const cv::Mat& image) {
  if (!initialized_) return kErrorBadInit;
  if (!image.isContinuous()) return kErrorNotContinuous;
  auto f = std::async(std::launch::async, Preprocess, image, width_, height_);
  mutex_.lock();
  preprocessed_.push(std::move(f));
  mutex_.unlock();
  return kSuccess;
}

ErrorCode RulerMaskFinder::Process() {
  if (!initialized_) return kErrorBadInit;
  preprocessed_ = {};
  return kSuccess;
}

bool RulerPresent(const cv::Mat& mask) {
  return false;
}

double RulerOrientation(const cv::Mat& mask) {
  double orientation = 0.0;
  return orientation;
}

cv::Mat Rectify(const cv::Mat& image, const double orientation) {
  cv::Mat r_image;
  return r_image;
}

cv::Rect FindRoi(const cv::Mat& mask, int h_margin, int v_margin) {
  cv::Rect roi;
  return roi;
}

cv::Mat Crop(const cv::Mat& image, const cv::Rect& roi) {
  cv::Mat c_image;
  return c_image;
}

}} // namespace openem::find_ruler

