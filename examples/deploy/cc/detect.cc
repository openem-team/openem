/// @file
/// @brief Example of detect deployment.
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

#include <iostream>

#include "detect.h"

int main(int argc, char* argv[]) {

  // Declare namespace aliases for shorthand.
  namespace em = openem;
  namespace dt = openem::detect;

  // Check input arguments.
  if (argc < 3) {
    std::cout << "Expected at least two arguments: " << std::endl;
    std::cout << "  Path to protobuf file containing model" << std::endl;
    std::cout << "  Paths to one or more image files" << std::endl;
    return -1;
  }

  // Create and initialize detector.
  dt::Detector detector;
  em::ErrorCode status = detector.Init(argv[1]);
  if (status != em::kSuccess) {
    std::cout << "Failed to initialize detector!" << std::endl;
    return -1;
  }

  // Load in images.
  std::vector<em::Image> imgs;
  auto size = detector.ImageSize();
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
    status = detector.AddImage(img);
    if (status != em::kSuccess) {
      std::cout << "Failed to add image for processing!" << std::endl;
      return -1;
    }
  }

  // Process the loaded images.
  std::vector<std::vector<std::array<int, 4>>> detections;
  status = detector.Process(&detections);
  if (status != em::kSuccess) {
    std::cout << "Error when attempting to do detection!" << std::endl;
    return -1;
  }

  // Display the detections on the image.
  for (int i = 0; i < detections.size(); ++i) {
    em::Image img = imgs[i];
    std::vector<std::array<int, 4>> dets = detections[i];
    if (dets.size() == 0) {
      std::cout << "No detections found for image " << i << std::endl;
      continue;
    }
    for (auto det : dets) {
      img.DrawRect(det);
    }
    img.Show();
  }
}

