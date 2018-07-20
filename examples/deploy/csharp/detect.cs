using System;
using System.Windows.Forms;

class Program {
  static int Main(string[] args) {

    // Check input arguments.
    if(args.Length < 2) {
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
    vector_image imgs = new vector_image();
    pair_int_int img_size = detector.ImageSize();
    for (int i = 1; i < args.Length; i++) {
      Image img = new Image();
      status = img.FromFile(args[i]);
      if (status != ErrorCode.kSuccess) {
        Console.Write("Failed to load image {0}!", args[i]);
        return -1;
      }
      img.Resize(img_size.first, img_size.second);
      imgs.Add(img);
    }

    // Add images to processing queue.
    foreach (var img in imgs) {
      status = detector.AddImage(img);
      if (status != ErrorCode.kSuccess) {
        Console.Write("Failed to add image for processing!");
        return -1;
      }
    }

    // Process the loaded images.
    vector_vector_rect detections = new vector_vector_rect();
    status = detector.Process(detections);
    if (status != ErrorCode.kSuccess) {
      Console.Write("Failed to process images!");
      return -1;
    }

    // Display the detections on the image.
    for (int i = 0; i < detections.Count; ++i) {
      vector_uint8 img_data = imgs[i].DataCopy();
      int w = imgs[i].Width();
      int h = imgs[i].Height();
      int ch = imgs[i].Channels();
      int nbytes = w * h * ch;
      System.Drawing.Bitmap disp_img = new System.Drawing.Bitmap(w, h,
          System.Drawing.Imaging.PixelFormat.Format24bppRgb);
      int k = 0;
      for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
          disp_img.SetPixel(c, r, System.Drawing.Color.FromArgb(
              img_data[k + 2],
              img_data[k + 1],
              img_data[k]));
          k += 3;
        }
      }
      using (System.Drawing.Graphics g = 
          System.Drawing.Graphics.FromImage(disp_img)) {
        System.Drawing.Color red = System.Drawing.Color.Red;
        System.Drawing.Pen pen = new System.Drawing.Pen(red);
        foreach (var det in detections[i]) {
          g.DrawRectangle(pen, det[0], det[1], det[2], det[3]);
        }
      }
      System.Drawing.Size size = new System.Drawing.Size(w, h);
      Form form = new Form();
      form.Text = "Ruler Finder Example";
      form.ClientSize = size;
      PictureBox box = new PictureBox();
      box.Size = size;
      box.Image = disp_img;
      box.Dock = DockStyle.Fill;
      form.Controls.Add(box);
      form.ShowDialog();
    }

    return 0;
  }
}

