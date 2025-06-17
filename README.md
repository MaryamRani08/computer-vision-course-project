# Project Assignments

## Overview

In this part, different image processing techniques are applied.The work includes reading image data, preprocessing images, analyzing pixel intensity values, detecting structures, and visualizing results.

## Progress

### Done Exercise 1 - Box Detection

Exercise 1 has been completed. It includes box detection from distance images and 3D point cloud data using RANSAC, morphological filtering, connected component analysis, and geometric dimension estimation.

**Completed tasks:**

- Data loading and visualization
- Distance image and point cloud preprocessing
- RANSAC-based floor plane detection
- Box top plane detection
- Binary mask filtering
- Largest connected component extraction
- Box dimension estimation


### Done Exercise 2 - Writer Retrieval

Exercise 2 has been completed. It includes writer identification and retrieval using local descriptors, VLAD encoding, normalization, and Exemplar SVM classification.

**Completed tasks:**

- Loaded training and test labels
- Used pre-computed local descriptors
- Generated visual codebook using MiniBatch K-Means
- Created VLAD encodings for train and test data
- Applied power normalization and L2 normalization
- Evaluated results using Top-1 Accuracy and mAP
- Improved performance using Exemplar SVM

**Results:**

| Method | Top-1 Accuracy | mAP |
|---|---:|---:|
| VLAD Encoding | 0.8225 | 0.6313 |
| VLAD + Exemplar SVM | 0.8856 | 0.7529 |

### Done Exercise 3 - Selective Search

Exercise 3 has been completed. It includes object proposal generation using Selective Search with Felzenszwalb segmentation, similarity-based region merging, and bounding box proposal filtering.

**Completed tasks:**

- Generated initial image segmentations
- Extracted regions and neighboring regions
- Calculated color, texture, size, and fill similarities
- Merged regions using hierarchical grouping
- Generated rectangular object proposals
- Filtered noisy, duplicate, small, and distorted boxes
- Saved final result images with proposal boxes

