/// @file
/// @brief Example of classify deployment.

#include <iostream>

#include "classify.h"

int main(int argc, char* argv[]) {

  // Declare namespace aliases for shorthand.
  namespace em = openem;
  namespace cl = openem::classify;

  // Check input arguments.
  if (argc < 3) {
    std::cout << "Expected at least two arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Paths to one or more image files" << std::endl;
    return -1;
  }

  // Create and initialize classifier.
  cl::Classifier classifier;
  em::ErrorCode status = classifier.Init(argv[1]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize classifier!" << std::endl;
    return -1;
  }

  // Load in images.
  std::vector<em::Image> imgs;
  auto size = classifier.ImageSize();
  for (int i = 2; i < argc; ++i) {
    em::Image img;
    status = img.FromFile(argv[i]);
    if (status != em::kSuccess) {
      std::cout << "Failed to load image " << argv[i] << "!" << std::endl;
      return -1;
    }
    img.Resize(size.first, size.second);
    imgs.push_back(std::move(img));
  }

  // Add images to processing queue.
  for (const auto& img : imgs) {
    status = classifier.AddImage(img);
    if (status != em::kSuccess) {
      std::cout << "Failed to add image for processing!" << std::endl;
      return -1;
    }
  }

  // Process the loaded images.
  std::vector<std::vector<float>> scores;
  status = classifier.Process(&scores);
  if (status != em::kSuccess) {
    std::cout << "Error when attempting to do classification!" << std::endl;
    return -1;
  }

  // Display the images and print scores to console.
  for (int i = 0; i < scores.size(); ++i) {
    const std::vector<float>& score = scores[i];
    std::cout << "*******************************************" << std::endl;
    std::cout << "Fish cover scores:" << std::endl;
    std::cout << "No fish:        " << score[0] << std::endl;
    std::cout << "Hand over fish: " << score[1] << std::endl;
    std::cout << "Fish clear:     " << score[2] << std::endl;
    std::cout << "*******************************************" << std::endl;
    std::cout << "Fish species scores:" << std::endl;
    std::cout << "Fourspot:   " << score[3] << std::endl;
    std::cout << "Grey sole:  " << score[4] << std::endl;
    std::cout << "Other:      " << score[5] << std::endl;
    std::cout << "Plaice:     " << score[6] << std::endl;
    std::cout << "Summer:     " << score[7] << std::endl;
    std::cout << "Windowpane: " << score[8] << std::endl;
    std::cout << "Winter:     " << score[9] << std::endl;
    std::cout << std::endl;
    imgs[i].Show();
  }
}

