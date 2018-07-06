## Data Collection Guidelines

# Region of interest

The region of interest (ROI) is the part of the video frame that may contain
fish.  Although the ROI may be the entire video frame, typically the ROI is only
part of it.  Some algorithms in OpenEM, such as detection, will perform better
if the input images are cropped to the ROI.  Therefore, many of the data
collection guidelines are driven by recommendations for the ROI, not the entire
video frame.

# Environment

The example data is all taken during the daytime.  Algorithms in OpenEM can work
under other conditions, such as with artificial lighting at night or at
dawn/dusk for users who wish to train models on their own data.  Keep in mind,
however, that generally speaking lower variability in appearance will lead to
better algorithm performance.

# Camera characteristics

Video data is expected to have three channels of color (RGB).  Camera
resolution is driven by the resolution of the region of interest.  The
resolution of the ROI should be near 720 x 360 pixels, but lower resolution may
still yield acceptable results.

# Camera placement

Video should be taken from the overhead perspective, perpendicular to the
broadside of any fish in the field of view.  We recommend that the camera be
aligned with perpendicular within 20 degrees.  If possible, the region of
interest should be located near the center of the camera field of view to
minimize lens distortion.

# Use of rulers

OpenEM has functionality that allows for automatic determination of the region
of interest.  This functionality requires the use of a ruler that will span the
region of interest whenever a fish may require detection.  See figure 1 for
examples of a region of interest spanned by a ruler.

# Fish movement

Each fish that is to be detected, counted, classified, or measured should be
moved through the ROI in the following way:

* The fish is oriented head to tail horizontally within the ROI.  The ROI itself
  may be rotated within the video frame, but fish within the ROI should be
oriented along one of its primary axes.

* The fish should pass through the ROI along a linear path.

* At some point while passing through the ROI, the camera should have an
  unobstructed view of the fish (no hands or other objects in front of it).

