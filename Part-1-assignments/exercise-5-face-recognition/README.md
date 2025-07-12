# Exercise 5 - Face Recognition

This folder contains the solution for **Exercise 5 - Face Recognition** of the Computer Vision Course Project.

The goal of this exercise is to develop a basic video-based face recognition system. The system can detect, track, align, identify, and re-identify faces using facial embeddings and machine learning techniques.

## Overview

This exercise focuses on building a complete face recognition pipeline.

The system contains two main parts:

- **Training:** Collects facial data from videos and stores learned models.
- **Testing:** Uses trained models to identify or re-identify faces in video data.

The implementation supports:

- Face detection
- Face tracking
- Face alignment
- Face embedding extraction
- Face identification
- Face verification
- Open-set recognition
- Face clustering
- DIR curve evaluation

## Implementation Details

### 1. Face Detection, Tracking, and Alignment

Face preprocessing is implemented in `face_detector.py`.

This part includes:

- Detecting the largest face in a frame
- Tracking faces across video frames
- Re-initializing tracking when tracking quality is poor
- Aligning detected faces to a fixed image size
- Preparing faces for feature extraction

Face tracking is used to avoid running face detection on every frame, which improves runtime performance.

### 2. Face Identification and Verification

Face recognition is implemented in `face_recognition.py`.

This part uses facial embeddings extracted from aligned face images. The system compares embeddings using distance-based matching.

Implemented features include:

- Extracting deep face embeddings
- Building a labeled face gallery
- k-nearest neighbor based face identification
- Posterior probability calculation
- Distance-based verification
- Open-set recognition using unknown-class decision thresholds

### 3. Face Clustering and Re-Identification

Unsupervised face clustering is used for re-identification when identity labels are not available.

This part includes:

- Storing unlabeled face embeddings
- Implementing k-means clustering
- Assigning faces to clusters
- Re-identifying faces by finding the nearest cluster center

### 4. Evaluation

Evaluation is implemented using DIR curves.

This part includes:

- Computing identification rate
- Selecting similarity thresholds
- Evaluating open-set face recognition
- Plotting the Detection and Identification Rate curve

## Files

```text
exercise-5-face-recognition/
│
├── classifier.py
├── config.py
├── dir_curve.py
├── evaluation.py
├── face_detector.py
├── face_recognition.py
├── osr_learning.py
├── requirements.txt
├── test_osr_learning.py
├── test.py
└── training.py