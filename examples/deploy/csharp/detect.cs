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

using System;

/// <summary>
/// Example demonstrating Detector class.
/// </summary>
class Program {

  /// <summary>
  /// Main program.
  /// </summary>
  static int Main(string[] args) {

    // Check input arguments.
    if (args.Length < 2) {
      Console.WriteLine("Expected at least two arguments:");
      Console.WriteLine("  Path to protobuf file containing model");
      Console.WriteLine("  Paths to one or more image files");
      return -1;
    }
    
    // Create and initialize detector.
    Detector detector = new Detector();
    ErrorCode status = detector.Init(args[0]);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to initialize detector!");
      return -1;
    }

    // Load in images.
    VectorImage imgs = new VectorImage();
    for (int i = 1; i < args.Length; i++) {
      Image img = new Image();
      status = img.FromFile(args[i]);
      if (status != ErrorCode.kSuccess) {
        Console.WriteLine("Failed to load image {0}!", args[i]);
        return -1;
      }
      imgs.Add(img);
    }

    // Add images to processing queue.
    foreach (var img in imgs) {
      status = detector.AddImage(img);
      if (status != ErrorCode.kSuccess) {
        Console.WriteLine("Failed to add image for processing!");
        return -1;
      }
    }

    // Process the loaded images.
    VectorVectorDetection detections = new VectorVectorDetection();
    status = detector.Process(detections);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to process images!");
      return -1;
    }

    // Display the detections on the image.
    for (int i = 0; i < detections.Count; ++i) {
      foreach (var det in detections[i]) {
        imgs[i].DrawRect(det.location);
      }
      imgs[i].Show();
    }

    return 0;
  }
}

