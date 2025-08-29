# Project Assignments

## Overview

In this part, different image processing techniques are applied.The work includes reading image data, preprocessing images, analyzing pixel intensity values, detecting structures, and visualizing results.

## Progress

---

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


---

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

---

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

---

### Done Exercise 4 - Demosaicing and HDR

Exercise 4 has been completed. It includes demosaicing raw Bayer images, applying gamma correction and white balance, verifying sensor linearity, generating HDR output, and saving the final processed JPG image.

**Completed tasks:**

- Bayer pattern investigation
- Raw image demosaicing
- Gamma correction
- White balance
- Sensor linearity verification
- HDR image generation
- Tone mapping
- Final raw-to-JPG processing

---

### Done Exercise 5 - Face Recognition

Exercise 5 has been completed for the main face recognition system. It includes face detection, tracking, alignment, face identification, open-set recognition, face clustering, re-identification, and DIR curve evaluation.

**Completed tasks:**

- Implemented face detection, tracking, and alignment
- Extracted face embeddings using FaceNet
- Built a labeled gallery for face identification
- Implemented k-NN based face recognition
- Added posterior probability and distance-based prediction
- Added open-set recognition for unknown faces
- Implemented face clustering using k-means
- Added person re-identification using cluster centers
- Implemented DIR curve evaluation

---

### Done Part 2 - Exercise 1: Advanced RANSAC for Box Detection

Part 2 Exercise 1 has been completed. It extends the original box detection task by implementing and comparing advanced RANSAC-based methods for robust plane fitting and box dimension estimation.

**Completed tasks:**

- Extended the baseline RANSAC box detection pipeline
- Implemented MLESAC for cost-based model selection
- Tested MLESAC with different floor and box-top thresholds
- Compared RANSAC and MLESAC results
- Implemented Preemptive RANSAC with fixed computational budget
- Tested different `M` and `B` parameter values
- Compared runtime and accuracy trade-offs
- Discussed advantages and disadvantages of MLESAC and Preemptive RANSAC
- Added report PDFs for implementation results and discussion

---

---

### Done Part 2 - Exercise 2: Writer Retrieval

Part 2 Exercise 2 has been completed for tasks **e–g**. It extends the writer identification pipeline with custom SIFT feature extraction, Generalized Max Pooling, and Multi-VLAD with PCA whitening.

**Completed tasks:**

- Implemented SIFT feature extraction from images
- Applied Hellinger normalization to SIFT descriptors
- Integrated custom features using the `--from_images` option
- Implemented Generalized Max Pooling using ridge regression
- Tested GMP with `--gmp --gamma 1`
- Built multiple VLAD codebooks using different runs
- Applied PCA whitening on Multi-VLAD encodings
- Compared Multi-VLAD, PCA, and E-SVM performance
- Added implementation report PDF

---

### Done Part 2 - Exercise 3: Selective Search Detection Pipeline

Part 2 Exercise 3 has been completed. It extends the Selective Search assignment by adding a simple object detection pipeline using generated region proposals, labeled samples, feature extraction, and SVM classification.

**Completed tasks:**

- Generated Selective Search region proposals
- Saved proposal results in `balloon_proposals.json`
- Labeled proposals using ground-truth annotations
- Created positive and negative training samples
- Extracted features from proposal regions
- Trained an SVM classifier
- Saved the trained model as `svm_model.joblib`
- Added QnA/report PDF for the detection pipeline
- Excluded large generated `features.json` file from GitHub because it exceeds the normal GitHub file size limit

---

### Done Part 2 - Exercise 4: Additional HDR from JPG Images

Part 2 Exercise 4 has been completed for the additional HDR task. It extends the demosaicing and HDR assignment by generating HDR images from JPG exposure stacks instead of RAW sensor data.

**Completed tasks:**

- Loaded JPG exposure stack
- Estimated the camera gamma/response curve
- Linearized JPG images before HDR fusion
- Scaled images using exposure times
- Combined JPG images into an HDR image
- Applied white balance in linear space
- Applied logarithmic tone mapping
- Saved the final HDR-from-JPG output

**Note:** Tasks 1–8 are documented in **Part 1 - Exercise 4**. This section only covers the additional Task 9.

