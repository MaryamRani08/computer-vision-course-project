# Part 2 - Exercise 4: Additional HDR from JPG Images

This folder contains the solution for **Part 2 - Exercise 4** of the Computer Vision Course Project.

This part focuses only on the **additional HDR exercise**, where HDR images are generated from JPG exposure stacks instead of raw sensor data.

The main Exercise 4 tasks, including Bayer pattern investigation, demosaicing, gamma correction, white balance, sensor linearity, RAW-based HDR, iCAM06, and `process_raw`, are already included in **Part 1 - Exercise 4**.

## Overview

Raw images have a mostly linear relationship between scene brightness and pixel values.  
JPG images, however, are processed by the camera and include non-linear transformations such as gamma correction and tone mapping.

Because of this, JPG images must first be converted back into a linear-light representation before they can be combined into an HDR image.

This exercise implements HDR generation from JPG images by estimating the camera response curve and reversing the non-linear transformation.

## Implemented Additional Task

### Task 9 - HDR from JPG Images

The additional task was completed.

The implemented pipeline includes:

- Loading a JPG exposure stack
- Reading RGB images and luminance values
- Estimating the camera gamma/response curve
- Linearizing JPG images
- Scaling images according to exposure time
- Combining the exposure stack into an HDR image
- Applying white balance in linear space
- Compressing dynamic range using logarithmic tone mapping
- Saving the final HDR result as a JPG image
