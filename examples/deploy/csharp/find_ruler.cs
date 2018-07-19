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
    
    // Create and initialize the mask finder.
    RulerMaskFinder mask_finder = new RulerMaskFinder();
    ErrorCode status = mask_finder.Init(args[0]);
    if (status != ErrorCode.kSuccess) {
      Console.WriteLine("Failed to initialize ruler mask finder!");
      return -1;
    }

    // Load in images.
    vector_image imgs = new vector_image();
    for (int i = 1; i < args.Length; i++) {
      Image img = new Image();
      status = img.FromFile(args[i]);
      if (status != ErrorCode.kSuccess) {
        Console.Write("Failed to load image {0}!", args[i]);
        return -1;
      }
      imgs.Add(img);
    }

    // Add images to processing queue.
    foreach (var img in imgs) {
      status = mask_finder.AddImage(img);
      if (status != ErrorCode.kSuccess) {
        Console.Write("Failed to add image for processing!");
        return -1;
      }
    }

    // Process the loaded images.
    vector_image masks = new vector_image();
    status = mask_finder.Process(masks);
    if (status != ErrorCode.kSuccess) {
      Console.Write("Failed to process images!");
      return -1;
    }

    for (int i = 0; i < masks.Count; ++i) {
      // Resize the masks back into the same size as the images.
      masks[i].Resize(imgs[i].Width(), imgs[i].Height());

      // Check if the ruler is present.
      bool present = openem.RulerPresent(masks[i]);
      if (!present) {
        Console.Write("Could not find ruler in image!  Skipping...");
        continue;
      }

      // Find orientation and region of interest based on the mask.
      vector_double transform = openem.RulerOrientation(masks[i]);
      Image r_mask = openem.Rectify(masks[i], transform);
      rect roi = openem.FindRoi(r_mask);

      // Rectify, crop, and display the image.
      Image r_img = openem.Rectify(imgs[i], transform);
      Image c_img = openem.Crop(r_img, roi);
      vector_uint8 img_data = c_img.DataCopy();
      int w = c_img.Width();
      int h = c_img.Height();
      int ch = c_img.Channels();
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

