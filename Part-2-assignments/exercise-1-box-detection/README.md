# Part 2- Exercise 1 - Advanced RANSAC for Box Detection

This folder contains the solution for **Part2 Exercise 1** of the Computer Vision Course Project.

The goal of this exercise is to extend the original box detection task by implementing and evaluating advanced RANSAC-based methods:

- Maximum Likelihood Estimation Sample Consensus (MLESAC)
- Preemptive RANSAC
- Parameter sensitivity analysis
- Runtime and accuracy comparison

This exercise builds on the box detection pipeline where amplitude images, distance images, and 3D point cloud data are used to estimate the dimensions of a box.

## Overview

In the original box detection task, RANSAC is used to detect two dominant planes:

1. The floor plane
2. The top plane of the box

After detecting these planes, the box dimensions are estimated from the fitted plane models and the 3D point cloud data.

In this part, the basic RANSAC method is improved and compared with MLESAC and Preemptive RANSAC to analyze robustness, parameter sensitivity, and runtime behavior.

## Implemented Methods

### 1. Baseline RANSAC

The baseline RANSAC implementation is used to detect planes in the 3D point cloud.

The main steps include:

- Loading amplitude image, distance image, and point cloud data
- Removing invalid 3D points
- Detecting the floor plane using RANSAC
- Creating a binary floor mask
- Cleaning the floor mask using morphological operations
- Removing floor points from the point cloud
- Detecting the box top plane using RANSAC
- Estimating box width, length, and height

### 2. MLESAC

MLESAC extends the standard RANSAC scoring strategy.

Instead of only counting the number of inliers, MLESAC uses a cost-based scoring function. Inlier points contribute their distance error, while outliers receive a fixed penalty.

This makes the model selection less sensitive to the distance threshold.

Tested threshold values:

| Floor Threshold | Box Top Threshold |
|---:|---:|
| 0.02 | 0.005 |
| 0.05 | 0.01 |
| 0.10 | 0.02 |

The results show that MLESAC is more stable when the threshold changes because it uses residual errors instead of a simple binary inlier/outlier decision.

### 3. Preemptive RANSAC

Preemptive RANSAC is implemented to reduce runtime and provide a fixed computational budget.

Instead of evaluating unlimited random hypotheses, a fixed number of hypotheses is generated first. These hypotheses are evaluated in batches, and weaker hypotheses are removed after each evaluation step.

Tested parameters:

M = 500, 1000, 2000
B = 50, 100, 200 

Where:
M is the number of initial hypotheses
B is the batch size used before pruning hypotheses

This allows comparison between accuracy and runtime for different time budgets.

Results and Reports

The implementation results are currently included in the report PDFs.

#### Discussion Report

The reports include:

RANSAC baseline results
MLESAC comparison results
Threshold sensitivity experiments
Preemptive RANSAC results
Runtime and parameter comparison
Discussion of advantages and disadvantages
Key Observations
##### MLESAC

MLESAC gives more stable results compared to standard RANSAC because it uses a continuous cost function.

Advantages:

More robust to threshold changes
Uses residual errors more effectively
Produces smoother model selection
Easy to integrate into an existing RANSAC pipeline

Disadvantages:

Slightly slower than standard RANSAC
Requires an additional penalty parameter
Still depends on threshold selection
Gives limited improvement when the data is already clean
##### Preemptive RANSAC

Preemptive RANSAC provides better control over runtime by using a fixed number of hypotheses.

Observations:

Increasing M improves the chance of finding a better model
Runtime increases as M increases
Very small B causes frequent pruning and sorting overhead
Very large B may remove good hypotheses too early
Intermediate B values provide a better balance between speed and accuracy

