using System;

/// <summary>
/// Example demonstrating FindRuler class.
/// </summary>
class Program {

  /// <summary>
  /// Main program.
  /// </summary>
  static int Main(string[] args) {

    // Check input arguments.
    if(args.Length < 2) {
      Console.WriteLine("Expected at least two arguments:");
      Console.WriteLine("  Path to protobuf file containing model");
      Console.WriteLine("  Paths to one or more image files");
      return -1;
    }
    
    // Create and initialize the mask finder.
    RulerMaskFinder mask_finder = new RulerMaskFinder();
    ErrorCode status = mask_finder.Init(args[0]);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to initialize ruler mask finder!");
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
      status = mask_finder.AddImage(img);
      if (status != ErrorCode.kSuccess) {
        Console.WriteLine("Failed to add image for processing!");
        return -1;
      }
    }

    // Process the loaded images.
    VectorImage masks = new VectorImage();
    status = mask_finder.Process(masks);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to process images!");
      return -1;
    }

    for (int i = 0; i < masks.Count; ++i) {
      // Resize the masks back into the same size as the images.
      masks[i].Resize(imgs[i].Width(), imgs[i].Height());

      // Check if the ruler is present.
      bool present = openem.RulerPresent(masks[i]);
      if (!present) {
        Console.WriteLine("Could not find ruler in image!  Skipping...");
        continue;
      }

      // Find orientation and region of interest based on the mask.
      VectorDouble transform = openem.RulerOrientation(masks[i]);
      Image r_mask = openem.Rectify(masks[i], transform);
      Rect roi = openem.FindRoi(r_mask);

      // Rectify, crop, and display the image.
      Image r_img = openem.Rectify(imgs[i], transform);
      Image c_img = openem.Crop(r_img, roi);
      c_img.Show();
    }

    return 0;
  }
}

