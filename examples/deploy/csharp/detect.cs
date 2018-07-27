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
    PairIntInt img_size = detector.ImageSize();
    for (int i = 1; i < args.Length; i++) {
      Image img = new Image();
      status = img.FromFile(args[i]);
      if (status != ErrorCode.kSuccess) {
        Console.WriteLine("Failed to load image {0}!", args[i]);
        return -1;
      }
      img.Resize(img_size.first, img_size.second);
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
    VectorVectorRect detections = new VectorVectorRect();
    status = detector.Process(detections);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to process images!");
      return -1;
    }

    // Display the detections on the image.
    for (int i = 0; i < detections.Count; ++i) {
      foreach (var det in detections[i]) {
        imgs[i].DrawRect(det);
      }
      imgs[i].Show();
    }

    return 0;
  }
}

