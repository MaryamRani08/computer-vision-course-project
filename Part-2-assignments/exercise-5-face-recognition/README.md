# Part 2 - Exercise 5: Face Recognition Challenge

This folder contains the solution for **Part 2 - Exercise 5** of the Computer Vision Course Project.

This part focuses on the **additional exercise** for open-set face recognition. The main face recognition system, including face detection, tracking, alignment, identification, clustering, re-identification, and DIR curve evaluation, is documented in **Part 1 - Exercise 5**.

## Overview

The goal of this exercise is to develop an open-set face recognition method that can classify known identities while also rejecting unknown identities.

The challenge includes:

- Known Classes (KCs)
- Known Unknown Classes (KUCs)
- Unknown Unknown Classes (UUCs)
- Single Pseudo Label training
- Multi Pseudo Label training

The implementation is mainly done in:

```text
src/cvproj_exc/osr_learning.py
````

## Individual Challenge Task

### Single Pseudo Label Training

Single Pseudo Label training groups all known unknown samples into one shared pseudo unknown class.

This allows the classifier to learn a boundary between known identities and unknown samples.

Main function:

```python
spl_training()
```

### Multi Pseudo Label Training

Multi Pseudo Label training assigns known unknown samples to multiple pseudo unknown labels.

This gives the model more flexibility when learning unknown-class regions and can help improve open-set rejection.

Main function:

```python
mpl_training()
```

## Dataset

The challenge uses:

```text
data/challenge_train_data.csv
```

This file contains:

* 128-dimensional feature vectors
* Known class labels
* Known unknown class labels
* Label `-1` for unknown-class samples
* Labels `>= 0` for known-class samples

The hidden test set is used for final benchmark evaluation.

## Evaluation Metrics

The challenge methods are evaluated using:

* AUCROC
* DIR@FAR = 1%
* DIR@FAR = 10%
* Balanced rank-1 identification rate
* Average prediction time per sample
* Model fitting time

