# Exercise 2 - Writer Retrieval

This folder contains the solution for **Exercise 2 - Writer Retrieval** of the Computer Vision Course Project.

The goal of this exercise is to build a writer identification and retrieval system using the ICDAR2017 Historical Writer Identification dataset. The system represents handwritten document images using local descriptors, VLAD encoding, and exemplar SVM classification.

## Overview

In this exercise, a writer retrieval pipeline is implemented using pre-computed local descriptors provided with the assignment. The task is to identify and retrieve documents written by the same writer.

The main pipeline includes:

- Loading training and test labels
- Loading local descriptors from `.pkl.gz` files
- Generating a visual codebook using MiniBatch K-Means
- Encoding images using VLAD
- Applying VLAD normalization
- Computing pairwise distances between image descriptors
- Evaluating retrieval performance using Top-1 Accuracy and mAP
- Improving retrieval performance using Exemplar SVM classification

## Dataset

The exercise uses the **ICDAR2017 Historical Writer Identification Dataset**.

The provided data includes:

- Training label file
- Test label file
- Pre-computed local feature descriptors for each image
- Code skeleton file for implementation

Each image is represented by a set of local descriptors. These descriptors are encoded into a global image representation using VLAD.

## Assignment Tasks

### a) Codebook Generation

A visual codebook is generated from randomly selected local descriptors from the training set.

Implementation details:

- Random descriptors are selected from training images
- A total of 100 training images are used for descriptor sampling
- MiniBatch K-Means is used for clustering
- The codebook contains 100 cluster centers
- The generated codebook is saved as:

```text
mus.pkl.gz

The shape of the codebook is:

(100, D)

where D is the dimensionality of the local descriptors.

### b) VLAD Encoding 

Using the generated codebook, each image is converted into a global VLAD descriptor.

VLAD encoding includes:

Assigning each local descriptor to the nearest cluster center
Computing residuals between descriptors and cluster centers
Aggregating residuals for each cluster
Concatenating all residuals into one global descriptor

The generated VLAD encodings are saved as:

enc_train.pkl.gz
enc_test.pkl.gz
### c) VLAD Normalization

Power normalization and L2 normalization are applied to improve the descriptor representation.

This step reduces the effect of visual burstiness, where repeated local patterns dominate the similarity score.

Power normalization can be activated using:

--powernorm
### d). Exemplar SVM Classification

To further improve retrieval performance, Exemplar SVM classification is applied.

For each test descriptor:

The test descriptor is used as the positive sample
All training descriptors are used as negative samples
A Linear SVM is trained
The normalized SVM weight vector is used as the new descriptor

This creates a more discriminative representation for writer retrieval.

### Results
Method	Top-1 Accuracy	mAP
VLAD Encoding	0.8225	0.6313
VLAD + Exemplar SVM	0.8856	0.7529

Conclusion

The implemented writer retrieval system successfully identifies handwritten document images based on visual features.

The basic VLAD representation gives strong results. After applying Exemplar SVM classification, both Top-1 Accuracy and mAP improve significantly.

This shows that Exemplar SVM creates a better global descriptor and improves writer retrieval performance compared to the original VLAD representation.

