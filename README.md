# Char-Seg

Contour-based Handwritten Digit Segmentation using OpenCV

### Usage
- 1 
  - Correct the skew 
  - BGR to Grayscale
  - Remove horizontal lines
  - Remove Vertical lines
  - Apply medium bluring
  - Apply non-localized means for final Denoising of the image
  - Binarize using adaptive threshold method
```text
python pre-process.py --input Testing.jpg
```

- 2 
  - BGR 2 Grayscale
  - Gaussian Blur
  - Dilation for thinning
  - Finding Connect Components by Contours

```text
python bounding-boxes.py
```

### Result

![This is bounding boxes](box.jpg)

### Future Works
- Different denoising methods with respective PSNR values for benchmarking
- Image Croping
- Image Clustering for labelling
