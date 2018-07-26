/// @file
/// @brief Example of find_ruler deployment.

#include <iostream>

#include "find_ruler.h"

int main(int argc, char* argv[]) {

  // Declare namespace aliases for shorthand.
  namespace em = openem;
  namespace fr = openem::find_ruler;

  // Check input arguments.
  if (argc < 3) {
    std::cout << "Expected at least two arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Paths to one or more image files" << std::endl;
    return -1;
  }

  // Create and initialize the mask finder.
  fr::RulerMaskFinder mask_finder;
  em::ErrorCode status = mask_finder.Init(argv[1]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize ruler mask finder!" << std::endl;
    return -1;
  }

  // Load in images.
  std::vector<em::Image> imgs;
  for (int i = 2; i < argc; ++i) {
    em::Image img;
    status = img.FromFile(argv[i]);
    if (status != em::kSuccess) {
      std::cout << "Failed to load image " << argv[i] << "!" << std::endl;
      return -1;
    }
    imgs.push_back(std::move(img));
  }

  // Add images to processing queue.
  for (const auto& img : imgs) {
    status = mask_finder.AddImage(img);
    if (status != em::kSuccess) {
      std::cout << "Failed to add image for processing!" << std::endl;
      return -1;
    }
  }

  // Process the loaded images.
  std::vector<em::Image> masks;
  status = mask_finder.Process(&masks);
  if (status != em::kSuccess) {
    std::cout << "Failed to process images!" << std::endl;
    return -1;
  }

  for (int i = 0; i < masks.size(); ++i) {
    // Resize the masks back into the same size as the images.
    masks[i].Resize(imgs[i].Width(), imgs[i].Height());

    // Check if the ruler is present.
    bool present = fr::RulerPresent(masks[i]);
    if (!present) {
      std::cout << "Could not find ruler in image!  Exiting..." << std::endl;
      continue;
    }

    // Find orientation and region of interest based on the mask.
    std::vector<double> transform = fr::RulerOrientation(masks[i]);
    em::Image r_mask = fr::Rectify(masks[i], transform);
    std::array<int, 4> roi = fr::FindRoi(r_mask);

    // Rectify, crop, and display the image.
    em::Image r_img = fr::Rectify(imgs[i], transform);
    em::Image c_img = fr::Crop(r_img, roi);
    c_img.Show();
  }
  return 0;
}


