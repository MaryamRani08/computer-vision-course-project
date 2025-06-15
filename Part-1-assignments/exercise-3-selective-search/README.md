# Exercise 3 - Selective Search

This folder contains the solution for **Exercise 3 - Selective Search** of the Computer Vision Course Project.

The goal of this exercise is to implement the Selective Search algorithm for object proposal generation. Selective Search is used to find possible object regions in an image before applying an object detection or classification model.

## Overview

In this exercise, initial image regions are generated using the Felzenszwalb segmentation algorithm. These regions are then merged step by step using different similarity measures to create larger object proposal regions.

The algorithm is tested on images from three different domains:

- Art History
- Christian Archaeology
- Classical Archaeology

The final output is a set of bounding boxes showing possible object regions in each input image.

## Implementation Details

The Selective Search pipeline includes the following steps:

### 1. Initial Segmentation

Initial regions are generated using the Felzenszwalb segmentation algorithm.

This creates small image regions that are used as the starting point for Selective Search.

### 2. Region Extraction

After segmentation, all regions are extracted from the image.

For each region, important information is calculated, including:

- Bounding box
- Region size
- Color histogram
- Texture histogram
- Region labels

### 3. Similarity Calculation

Neighboring regions are compared using different similarity measures.

The implemented similarity measures include:

- Color similarity
- Texture similarity
- Size similarity
- Fill similarity

These similarities help decide which regions should be merged.

### 4. Neighbor Extraction

Neighboring regions are identified and stored.

Only neighboring regions are considered for merging during the hierarchical grouping process.

### 5. Region Merging

The most similar neighboring regions are merged iteratively.

After each merge:

- Old similarities are removed
- New region information is calculated
- New similarities are added
- Final region proposals are updated

### 6. Bounding Box Proposal Generation

Selective Search merges regions with arbitrary shapes.  
To generate rectangular object proposals, bounding boxes are created around each merged region using the minimum and maximum x/y coordinates.

### 7. Proposal Filtering

The generated boxes are filtered to remove unnecessary or poor-quality proposals.

Filtering removes:

- Duplicate boxes
- Very small boxes
- Distorted or unusual shaped boxes
- Noisy proposals

## Results

The algorithm was tested on multiple images from the provided datasets.  
The red rectangles in the output images show the generated object proposals.

### Sample Output Images


results/
├── adoration1.jpg
├── ajax3.jpg
├── annunciation1.jpg
├── baptism1.jpg
├── ca-annun1.jpg
├── ca-annun2.jpg
├── ca-annun3.jpg
├── leading1.jpg
└── pursuit2.jpg

### Discussion

# Why is Selective Search needed if Felzenszwalb already gives segmentation?

Felzenszwalb produces only one segmentation of the image, which may miss objects at different scales. Selective Search improves this by merging regions using multiple cues such as color, texture, size, and fill, producing more flexible object candidates.

# How are proposal boxes filtered?

Proposal boxes are filtered based on duplicate boxes, minimum area, and unusual shape ratios. This helps remove noisy or unnecessary proposals and keeps stronger object candidates.

Additional filtering could include:

Removing highly overlapping boxes
Keeping boxes with strong edge information
Adjusting thresholds for different datasets
Filtering based on objectness score

# How are rectangles obtained from merged regions?

Selective Search merges regions of arbitrary shapes. Rectangular proposals are obtained by computing the bounding box around each region using the minimum and maximum x/y pixel coordinates.
