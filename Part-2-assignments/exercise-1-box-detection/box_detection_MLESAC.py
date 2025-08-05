import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import median_filter, binary_opening, binary_closing
import cv2
    
def visualize_image(image, title="Image", cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()

def visualize_point_cloud(point_cloud, sample_rate=10):

    # Subsample the point cloud for faster plotting.
    subsample = point_cloud[::sample_rate, ::sample_rate, :]
    X = subsample[:, :, 0].ravel()
    Y = subsample[:, :, 1].ravel()
    Z = subsample[:, :, 2].ravel()

    # Remove invalid points. 
    mask = Z != 0

    # Plot the 3D point cloud.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[mask], Y[mask], Z[mask], s=1)  # scatter plot of valid points
    ax.set_title("3D Point Cloud")
    plt.show()
    

def ransac_algo_MLESAC(point_cloud, thres, max_iter, use_mlesac=False, gamma=1.0):
    np.random.seed(42)
    flat_point = point_cloud.reshape(-1, 3)
    valid_pt = flat_point[flat_point[:, 2] != 0]
    total_pt = valid_pt.shape[0]

    best_inliers = []
    best_plane = None
    best_score = np.inf if use_mlesac else -1

    for _ in range(max_iter):
        idx = np.random.choice(valid_pt.shape[0], 3, replace=False)
        p1, p2, p3 = valid_pt[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) == 0:
            continue

        normal = normal / np.linalg.norm(normal)
        dot_prod = np.dot(normal, p1)

        distances = np.abs(np.dot(valid_pt, normal) - dot_prod)

        if use_mlesac:
            # Calculate MLESAC cost function 
            cost_func = np.sum(np.where(distances < thres, distances, gamma))
            if cost_func < best_score:
                best_score = cost_func
                best_inliers = valid_pt[distances < thres]
                best_plane = (normal, dot_prod)
        else:
            inliers = valid_pt[distances < thres]
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
                best_plane = (normal, dot_prod)
                if best_score == total_pt:
                    break

    return best_plane, np.array(best_inliers)

def create_and_visualize_binary_mask(point_cloud, inliers, title="Binary Mask", width=None, height=None):

    # Initialize an empty mask (all zeros = outliers)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Flatten the point cloud and keep original indices.
    flat_pc = point_cloud.reshape(-1, 3)
    indices = np.arange(flat_pc.shape[0])  # Generate array of indices that correspond to the rows (points) in the flattened point cloud.
    valid_mask = flat_pc[:, 2] != 0  # Filter valid points.

    # Filter only valid 3D points.
    valid_pc = flat_pc[valid_mask]
    valid_indices = indices[valid_mask]

    # Create a lookup dictionary to map 3D inlier points to their corresponding pixel indices.
    point_to_index = {
        tuple(np.round(p, 4)): idx for p, idx in zip(valid_pc, valid_indices)
    }

    # Match the inliers to valid points via the above dictionary.
    inlier_indices = []
    for point in inliers:
        key = tuple(np.round(point, 4))
        if key in point_to_index:
            inlier_indices.append(point_to_index[key])

    # Map the inlier indices back to (i, j) pixel coordinates and mark as inliers (1) in the mask.
    for flat_idx in inlier_indices:
        i = flat_idx // width  # row index
        j = flat_idx % width   # column index
        mask[i, j] = 1  

    # Visualize raw and cleaned mask side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Raw mask. 
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f"{title} (Raw)")
    axes[0].axis('off')  
    
    # Apply morphological filters to clean the mask. 
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))  # dilation, erosion
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, np.ones((5, 5)))  # erosion, dilation

    # Cleaned mask.
    axes[1].imshow(cleaned_mask, cmap='gray')
    axes[1].set_title(f"Cleaned {title}")
    axes[1].axis('off') 

    plt.show()

    return cleaned_mask

def convert_points_from_mask_to_numpy_array(point_cloud, mask, height, width):
    points = []

    for i in range(height):
        for j in range(width):
            if mask[i, j]:  
                x, y, z = point_cloud[i, j]
                if z != 0:  # valid points only
                    points.append([x, y, z])

    return np.array(points)

def measure_box_dimensions(box_top_points, box_plane, floor_plane):
    min_vals = np.min(box_top_points, axis=0)
    max_vals = np.max(box_top_points, axis=0)

    width = max_vals[0] - min_vals[0] 
    length = max_vals[1] - min_vals[1]  
    # height = abs(box_plane[1] - floor_plane[1])  # distance between the top and floor planes
    normal = box_plane[0]  # take the normal of one plane (both should be parallel)
    d1 = box_plane[1]
    d2 = floor_plane[1]
    height = abs(d2 - d1) / np.linalg.norm(normal)


    return width, length, height


def main():
  x = 1
  file_path = f'example{x}kinect.mat'
  data = scipy.io.loadmat(file_path)
  amplitude_img = data[f'amplitudes{x}']
  distance_img = data[f'distances{x}']
  point_cloud = data[f'cloud{x}']

  # Display the amplitude image (2D).
  visualize_image(amplitude_img, title="Amplitude Image")

  # Display the distance image (2D).
  visualize_image(distance_img, title="Distance Image", cmap = 'hot')

  # Display the point cloud (3D).
  visualize_point_cloud(point_cloud, sample_rate=10)

  # Flatten the point cloud into 2D and filter out the invalid points for RANSAC.
  flat_pc = point_cloud.reshape(-1, 3)
  valid_points = flat_pc[flat_pc[:, 2] != 0]  

#  Part (a): Floor plane detection with RANSAC/MLESAC:
  plane, inliers = ransac_algo_MLESAC(point_cloud, thres=0.10, max_iter=3000, use_mlesac=True, gamma=0.5)

  # Get shape of original point cloud.
  height, width, _ = point_cloud.shape

  # Creates a binary mask image for Box floor.
  floor_mask_cleaned = create_and_visualize_binary_mask(point_cloud, inliers, title="Floor Mask", width=width, height=height)

  # Get all the pixels that are not part of the floor.
  non_floor_mask = floor_mask_cleaned == 0  
  height, width = floor_mask_cleaned.shape
  
  # Extract the non-floor points (box and background) from the point cloud and convert to numpy array.
  non_floor_points = convert_points_from_mask_to_numpy_array(point_cloud, non_floor_mask, height, width)

  # Part (b): Box‐top plane detection from non‐floor points:
  box_plane, box_inliers = ransac_algo_MLESAC(non_floor_points, 0.2, 3000)

  # Creates a binary mask image for Box top.
  box_mask_cleaned = create_and_visualize_binary_mask(point_cloud, box_inliers, title="Box Top Mask", width=width, height=height)

  # Extract the box top points from the point cloud and convert to numpy array.
  box_top_points = convert_points_from_mask_to_numpy_array(point_cloud, box_mask_cleaned, height, width)

#  Part (c): Compute box width, length, and height
  box_width, box_length, box_height = measure_box_dimensions(box_top_points, box_plane, plane)

  print(f"Estimated Box Dimensions:")
  print(f"Width  ≈ {box_width:.3f} meters")
  print(f"Length ≈ {box_length:.3f} meters")
  print(f"Height ≈ {box_height:.3f} meters")
    
if __name__ == "__main__":
    main()