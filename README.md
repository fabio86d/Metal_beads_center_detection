# Interactive metal beads detection with MSER algorithm

![](demo_images/metal_bead_detection.png)

# Brief Description
The repository allows the extraction of image centers from X-ray images of fiducials (metal beads), typically used for validation of image registration procedures.

The algorithm makes use of a blob detection algorithm (maximally stable extremal regions (MSER)) provided by the open source computer vision python library “OpenCV” (https://www.learnopencv.com/blob-detection-using-opencv-python-c/) to interactively detect the beads centers from the center of fitted ellipses to detected blobs.
