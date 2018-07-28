using System;
using System.Collections.Generic;

/// <summary>
/// End to end example on a video clip.
/// </summary>
class Program {
  /// <summary>Finds ROI in a video.</summary>
  /// <param name="mask_finder_path">Path to find_ruler model file.</param>
  /// <param name="vid_path">Path to the video.</param>
  /// <param name="roi">Rect specifying the ROI.</param>
  /// <param name="transform">
  /// Transform specifying rectification matrix.
  /// </param>
  /// <returns>Error code.</returns>
  static ErrorCode FindRoi(
      string mask_finder_path,
      string vid_path,
      out Rect roi,
      out VectorDouble transform) {
    roi = new Rect();
    transform = new VectorDouble();
    return ErrorCode.kSuccess;
  }

  /// <summary>
  /// Finds and classifies detections for all frames in a video.
  /// </summary>
  /// <param name="detect_path">Path to detect model file.</param>
  /// <param name="classify_path">Path to classify model file.</param>
  /// <param name="roi">Region of interest output from FindRoi.</param>
  /// <param name="transform">Transform output from FindRoi.</param>
  /// <param name="detections">Detections for each frame.</param>
  /// <param name="scores">Cover and species scores for each detection.</param>
  static ErrorCode DetectAndClassify(
      string detect_path,
      string classify_path,
      string vid_path,
      Rect roi,
      VectorDouble transform,
      out List<VectorRect> detections,
      out List<VectorVectorFloat> scores) {
    detections = new List<VectorRect>();
    scores = new List<VectorVectorFloat>();
    return ErrorCode.kSuccess;
  }

  /// <summary>
  /// Writes a new video with bounding boxes around detections.
  /// </summary>
  /// <param name="vid_path">Path to the original video.</param>
  /// <param name="out_path">Path to the output video.</param>
  /// <param name="roi">Region of interest output from FindRoi.</param>
  /// <param name="transform">Transform output from FindRoi.</param>
  /// <param name="detections">Detections for each frame.</param>
  /// <param name="scores">Cover and species scores for each detection.</param>
  static ErrorCode WriteVideo(
      string vid_path,
      string out_path,
      Rect roi,
      VectorDouble transform,
      List<VectorRect> detections,
      List<VectorVectorFloat> scores) {
    return ErrorCode.kSuccess;
  }

  /// <summary>
  /// Main program.
  /// </summary>
  static int Main(string[] args) {

    // Check input arguments.
    if (args.Length < 4) {
      Console.WriteLine("Expected at least four arguments: ");
      Console.WriteLine("  Path to pb file with find_ruler model.");
      Console.WriteLine("  Path to pb file with detect model.");
      Console.WriteLine("  Path to pb file with classify model.");
      Console.WriteLine("  Path to one or more video files.");
    }

    for (int vid_idx = 3; vid_idx < args.Length; ++vid_idx) {
      // Find the roi.
      Console.WriteLine("Finding region of interest...");
      Rect roi;
      VectorDouble transform;
      ErrorCode status = FindRoi(args[0], args[3], out roi, out transform);
      if (status != ErrorCode.kSuccess) return -1;

      // Find detections and classify them.
      Console.WriteLine("Performing detection and classification...");
      List<VectorRect> detections;
      List<VectorVectorFloat> scores;
      status = DetectAndClassify(
          args[1], 
          args[2], 
          args[vid_idx], 
          roi, 
          transform, 
          out detections,
          out scores);
      if (status != ErrorCode.kSuccess) return -1;

      // Write annotated video to file.
      Console.WriteLine("Writing video to file...");
      status = WriteVideo(
          args[vid_idx],
          "annotated_video_{vid_idx - 3}.avi",
          roi,
          transform,
          detections,
          scores);
      if (status != ErrorCode.kSuccess) return -1;
    }
    return 0;
  }
}
