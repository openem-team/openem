using System;

/// <summary>
/// Example demonstrating Classifier class.
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

    // Create and initialize classifier.
    Classifier classifier = new Classifier();
    ErrorCode status = classifier.Init(args[0]);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to initialize classifier!");
      return -1;
    }

    // Load in images.
    VectorImage imgs = new VectorImage();
    PairIntInt img_size = classifier.ImageSize();
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
      status = classifier.AddImage(img);
      if (status != ErrorCode.kSuccess) {
        Console.WriteLine("Failed to add image for processing!");
        return -1;
      }
    }

    // Process the loaded images.
    VectorVectorFloat scores = new VectorVectorFloat();
    status = classifier.Process(scores);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to process images!");
      return -1;
    }

    // Display the images and print scores to console.
    for (int i = 0; i < scores.Count; ++i) {
      Console.WriteLine("*******************************************");
      Console.WriteLine("Fish cover scores:");
      Console.WriteLine("No fish:        {0}", scores[i][0]);
      Console.WriteLine("Hand over fish: {0}", scores[i][1]);
      Console.WriteLine("Fish clear:     {0}", scores[i][2]);
      Console.WriteLine("*******************************************");
      Console.WriteLine("Fish species scores:");
      Console.WriteLine("Fourspot:   {0}", scores[i][3]);
      Console.WriteLine("Grey sole:  {0}", scores[i][4]);
      Console.WriteLine("Other:      {0}", scores[i][5]);
      Console.WriteLine("Plaice:     {0}", scores[i][6]);
      Console.WriteLine("Summer:     {0}", scores[i][7]);
      Console.WriteLine("Windowpane: {0}", scores[i][8]);
      Console.WriteLine("Winter:     {0}", scores[i][9]);
      Console.WriteLine("");
      imgs[i].Show();
    }
    return 0;
  }
}
