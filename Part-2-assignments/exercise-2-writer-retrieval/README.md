# Part 2 - Exercise 2: Writer Retrieval

This folder contains the solution for **Part 2 - Exercise 2** of the Computer Vision Course Project.

This exercise extends the previous writer retrieval pipeline by implementing advanced feature extraction and encoding methods for writer identification.

The implemented tasks are:

- **Task e:** SIFT + Hellinger feature extraction
- **Task f:** Generalized Max Pooling
- **Task g:** Multi-VLAD with PCA whitening

## Overview

The goal of this exercise is to improve and analyze the writer identification system using additional methods beyond the basic VLAD pipeline.

The original writer retrieval system used pre-computed local descriptors, codebook generation, VLAD encoding, normalization, and Exemplar SVM classification.

In this part, the pipeline is extended with:

- Own SIFT descriptor extraction from images
- Hellinger normalization
- Generalized Max Pooling instead of standard VLAD sum pooling
- Multiple VLAD codebooks
- PCA whitening for dimensionality reduction and decorrelation

## Implemented Tasks

### Task e - SIFT + Hellinger Features

In this task, SIFT descriptors are extracted directly from the original images instead of using the provided pre-computed `.pkl.gz` descriptors.

Implementation details:

- SIFT keypoints are detected from the input images
- All keypoint angles are set to `0`
- SIFT descriptors are L1-normalized
- Signed square-root normalization is applied
- The feature extraction pipeline is integrated using the `--from_images` flag

This flag routes both descriptor loading and VLAD encoding through the custom `computeDescs` method.

The custom SIFT-based pipeline was verified on a subset of the dataset. Full evaluation was omitted because extracting features directly from images is significantly slower than loading the provided descriptor files.

### Task f - Generalized Max Pooling

In this task, the standard VLAD sum-pooling step is replaced with Generalized Max Pooling.

For each cluster, ridge regression is used to compute the pooled residual vector. The ridge regression coefficient vector is used as the new pooled representation.

This method helps balance frequent and rare local descriptors and reduces the dominance of repeated visual patterns.

Run configuration:

--gmp --gamma 1

Results using GMP with power normalization:

Method |Top-1 Accuracy|	mAP
GMP + PowerNorm|0.8008|	0.5888

On this setup, GMP gave performance comparable to standard sum-pooling. Since power normalization was already applied, the improvement from GMP was modest.

### Task g - Multi-VLAD + PCA Whitening

In this task, multiple VLAD codebooks are generated using different random seeds and descriptor subsets.

The process includes:

Training multiple codebooks
Encoding each image using each codebook
Concatenating the resulting VLAD descriptors
Applying PCA whitening
Training PCA only on the training encodings
Transforming the test encodings using the trained PCA model

Results from the 3-run example:

Method	|Top-1 Accuracy| mAP
Multi-VLAD + PCA	|0.7703	|0.5560
E-SVM on Multi-VLAD + PCA|	0.7650	|0.5535

In this experiment, Multi-VLAD with PCA whitening did not outperform the stronger single-VLAD with Exemplar SVM setup from the previous part.

Adding E-SVM after PCA also did not improve the results, likely because PCA whitening already decorrelates and regularizes the descriptor space.

#### Results Summary
Task	
e	|SIFT + Hellinger	Verified on subset	Full evaluation omitted
f	|GMP + PowerNorm	0.8008	0.5888
g	|Multi-VLAD + PCA	0.7703	0.5560
g	|E-SVM on Multi-VLAD + PCA	0.7650	0.5535
Report

The implementation report is included as a PDF.
