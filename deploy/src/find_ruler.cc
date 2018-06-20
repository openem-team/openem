/// @file
/// @brief Implementation for finding rulers in images.

#include "find_ruler.h"

namespace openem { namespace find_ruler {

namespace tf = tensorflow;

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
  if ((image.rows != height_) || (image.cols != width_)) {
    cv::Mat im;
  }
  else {
  }
  return kSuccess;
}

ErrorCode RulerMaskFinder::Process() {
  if (!initialized_) return kErrorBadInit;
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

