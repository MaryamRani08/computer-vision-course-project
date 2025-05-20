# Exercise-1 Box Detection using RANSAC and Image Processing

This folder contains the solution for **Exercise 1** of the Computer Vision Course Project.

The goal of this assignment is to estimate the size of a box from a distance image and a registered 3D point cloud. The task uses image processing, point cloud analysis, plane detection, RANSAC, morphological filtering, and geometric measurement techniques.

## Overview

In this assignment, a Time-of-Flight dataset is used to detect a box placed on the floor.  
Each dataset example contains:

- ToF amplitude image
- Distance image
- 3D point cloud

The amplitude image, distance image, and point cloud are registered, meaning that the same pixel position corresponds to the same physical point in all three representations.

The main objective is to detect the floor plane and the top plane of the box, then calculate the box dimensions from the detected planes and box corners.

## Assignment Tasks

### 1. Load and Visualize the Data

The provided `.mat` files contain the input data used for testing the implementation.

The data includes:

- `A` - amplitude image
- `D` - distance image
- `PC` - point cloud

The first step is to load the data and visualize the amplitude image, distance image, and 3D point cloud.

### 2. Preprocess the Input Data

Different filters are tested to improve the quality of the input data before plane detection.

Possible preprocessing steps include:

- Noise reduction
- Image smoothing
- Invalid point removal
- Point cloud subsampling for faster visualization
- Filtering of amplitude and distance data

### 3. Implement RANSAC for Plane Detection

RANSAC is implemented from scratch to detect dominant planes in the point cloud.

The implementation takes the following parameters:

- Point cloud containing 3D point vectors
- Distance threshold for evaluating inliers
- Maximum number of iterations

RANSAC is first used to detect the floor plane.  
The detected inliers are visualized as a binary mask, where:

- `1` represents floor pixels
- `0` represents non-floor pixels

### 4. Filter the Floor Mask

Morphological operations are applied to improve the quality of the detected floor mask.

Common operations include:

- Binary opening
- Binary closing
- Noise removal
- Hole filling

The filtered mask is then used to separate floor points from non-floor points.

### 5. Detect the Top Plane of the Box

After removing the floor points, RANSAC is applied again to the remaining point cloud.

The goal is to detect the dominant plane that represents the top surface of the box.

A binary mask is created for the box top plane.

### 6. Extract the Largest Connected Component

The detected box mask may contain noise or unrelated objects.  
To improve the result, the largest connected component is selected as the final box top region.

This step helps remove false detections and keeps only the main box surface.

### 7. Estimate Box Dimensions

After detecting the floor plane and box top plane, the box dimensions are calculated.

The height is estimated using the distance between the floor plane and the top plane of the box.

The length and width are estimated by analyzing the 3D coordinates of the detected box corners.

## Technologies Used

- Python
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

