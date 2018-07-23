using System.Windows.Forms;

/// <summary>
/// Utility functions for working with OpenEM images.
/// </summary>
class Util {
  /// <summary>
  /// Converts an OpenEM image to a Bitmap.
  /// </summary>
  /// <param name="img"> OpenEM image object. </param>
  /// <returns> System.Drawing.Bitmap object. </returns>
  public static System.Drawing.Bitmap ImageToBitmap(Image img) {
    vector_uint8 img_data = img.DataCopy();
    int w = img.Width();
    int h = img.Height();
    int ch = img.Channels();
    int nbytes = w * h * ch;
    System.Drawing.Bitmap new_img = new System.Drawing.Bitmap(w, h,
        System.Drawing.Imaging.PixelFormat.Format24bppRgb);
    int k = 0;
    for (int r = 0; r < h; r++) {
      for (int c = 0; c < w; c++) {
        new_img.SetPixel(c, r, System.Drawing.Color.FromArgb(
            img_data[k + 2],
            img_data[k + 1],
            img_data[k]));
        k += 3;
      }
    }
    return new_img;
  }

  /// <summary>
  /// Displays a Bitmap in a Windows Form.
  /// </summary>
  /// <param name="title"> Title of the display window. </param>
  /// <param name="img"> Bitmap object. </param>
  public static void ShowBitmap(string title, System.Drawing.Bitmap img) {
    Form form = new Form();
    form.Text = title;
    form.ClientSize = img.Size;
    PictureBox box = new PictureBox();
    box.Size = img.Size;
    box.Image = img;
    box.Dock = DockStyle.Fill;
    form.Controls.Add(box);
    form.ShowDialog();
  }
}

