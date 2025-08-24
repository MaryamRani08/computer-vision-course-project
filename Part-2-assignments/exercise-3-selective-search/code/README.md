# Part 2 - Exercise 3: Selective Search Detection Pipeline

This folder contains the solution for **Part 2 - Exercise 3** of the Computer Vision Course Project.

This exercise extends the Selective Search implementation by building a simple object detection pipeline. The pipeline generates region proposals, labels them using ground-truth annotations, extracts features, trains an SVM classifier, and performs inference on test images.

## Overview

The goal of this exercise is to move from object proposal generation to a basic object detection system.

The implemented pipeline includes:

- Selective Search region proposal generation
- Proposal labeling using COCO annotations
- Positive and negative sample creation
- Feature extraction from proposal regions
- SVM classifier training
- Inference on test images
- Visualization of predicted detections

The exercise uses a small balloon detection dataset with training, validation, and test splits.

## Implementation Details

### 1. Selective Search

The Selective Search algorithm is used to generate object proposal boxes.

The implementation uses:

- Initial segmentation
- Region extraction
- Region similarity calculation
- Hierarchical region merging
- Final bounding box proposal generation

The result is a set of candidate object regions for each input image.

### 2. Proposal Generation

The script `generate_proposals.py` generates region proposals for the balloon dataset.

Generated proposals are saved in:

results/balloon_proposals.json


This file stores candidate bounding boxes for the dataset images.

### 3. Proposal Labeling

The script `label_proposals.py` compares generated proposals with ground-truth bounding boxes from COCO annotation files.

Proposals are labeled as:

* Positive samples if they overlap strongly with a ground-truth object
* Negative samples if they have low overlap
* Ignored samples if they fall between the positive and negative thresholds

The labeled proposals are saved in:

```text
results/labeled_proposals.json
```

### 4. Feature Extraction

The script `extract_features.py` extracts features from the labeled proposal regions.

The extracted feature vectors are saved in:

```text
results/features.json
```

These features are later used to train the classifier.

### 5. SVM Classifier Training

The script `train_classifier.py` trains an SVM classifier using the extracted features.

The trained model is saved as:

```text
results/svm_model.joblib
```

The classifier learns to distinguish balloon regions from background regions.

### 6. Inference

The script `inference.py` applies the trained classifier to proposal regions in test images.

