from __future__ import division
import skimage.feature
import skimage.color
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    segments = skimage.segmentation.felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)

    # merge the original image and the segmentation results
    im_with_segments = np.dstack([im_orig, segments.astype(np.float32)])

    return im_with_segments # image with segment information (4th channel)
    
def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
     # color histograms for both regions
    hist1 = r1['colour_hist']
    hist2 = r2['colour_hist']

    # histogram intersection = sum of minimum values of each bin
    hist_inter_color = np.minimum(hist1, hist2).sum()

    return hist_inter_color


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    # texture histograms for both regions
    hist1 = r1['text_hist']
    hist2 = r2['text_hist']

    # histogram intersection = sum of minimum values of each bin
    hist_text_color = np.minimum(hist1, hist2).sum()

    return hist_text_color


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    # size of both regions
    size1 = r1["size"]
    size2 = r2["size"]

    # calculate size similarity of regions in terms of their relative size in the image
    size_similarity = 1 - ((size1 + size2)/ imsize) 

    return size_similarity


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    # size of both regions
    size1 = r1["size"]
    size2 = r2["size"]

    # bounding box coordinates of both regions
    min_x = min(r1['min_x'], r2['min_x'])
    min_y = min(r1['min_y'], r2['min_y'])
    max_x = max(r1['max_x'], r2['max_x'])
    max_y = max(r1['max_y'], r2['max_y'])
    
    # Calculate the size of the combined bounding box
    box_size = (max_x - min_x) * (max_y - min_y)
    
    # calculate the fill similarity
    fill_similarity = 1 - ((box_size - size1 - size2) / imsize)
    
    return fill_similarity


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    #hist = np.array([])

    #combine histograms for each channel to create a single one of size 75
    hist = np.concatenate([
        np.histogram(img[..., channel].ravel(), bins=BINS, range=(0, 1), density=True)[0]
        for channel in range(img.shape[2])
    ])

    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    # LBP params
    points = 8  # standard LBP neighbors
    radius = 1  # radius of neighborhood size
    method = 'uniform'  # pattern for LBP 

    # loop through each color channel (R, G, B) 
    for c in range(img.shape[2]):
        # apply LBP to current color channel and store it in the corresponding slice
        ret[:, :, c] = skimage.feature.local_binary_pattern(img[:, :, c], P=points, R=radius, method=method)

    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    #hist = np.array([])

    # calculate texture gradient for the whole image
    text_gradient = calc_texture_gradient(img)
    
    # initialize histogram array for all channels (R, G, B)
    hist = np.concatenate([
        np.histogram(text_gradient[:, :, channel].ravel(), bins=BINS, density=True)[0]
        for channel in range(img.shape[2])
    ])
    
    # L1 normalization
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}

    # convert to HSV 
    hsv_img = skimage.color.rgb2hsv(img[:, :, :3]) # only use RGB channels 
    
    # extract region labels 
    labels = img[:, :, 3]
    region_labels = np.unique(labels).astype(int)
    
    for label in region_labels:
        mask = labels == label
        y, x = np.nonzero(mask)
            
        # bounding box
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
  
        # Extract region and HSV data
        region = img[:, :, :3] * mask[..., None]
        hsv_region = hsv_img * mask[..., None]

        # set bounding box dimensions
        bbox = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'rect': (min_x, min_y, max_x - min_x, max_y - min_y)
        }

        # feature histograms
        features = {
            'colour_hist': calc_colour_hist(hsv_region),
            'text_hist': calc_texture_hist(region)
        }

        # combine above into region descriptor
        R[label] = {
            **bbox,
            'size': len(y),
            'labels': [label],
            **features
        }
    return R

def extract_neighbours(regions):
    """
    Identify pairs of neighboring regions based on bounding box intersection.
    """
    # hint2: helper function to check if two bounding boxes intersect
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    neighbours = [] # hint1: list of neighbouring regions

    keys = list(regions.keys()) # get all region keys

    for i in range(len(keys)): 
        for j in range(i + 1, len(keys)): # compare each region with every other region
            key_i = keys[i] # get the key of the first region
            key_j = keys[j] # get the key of the second region

            # get the regions corresponding to the keys
            region_i = regions[key_i] 
            region_j = regions[key_j]

            # add the pair of regions to the neighbours list if they intersect
            if intersect(region_i, region_j): 
                neighbours.append( 
                    ((key_i, region_i), (key_j, region_j))
                )    

    return neighbours # list of tuples containing pairs of neighbouring regions

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}

    # calculate new bounding box coordinates
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])

    #  weighted average of color histogram
    colour_hist = ((r1["colour_hist"] * r1["size"]) + (r2["colour_hist"] * r2["size"])) / new_size

    # weighted average of texture histogram
    text_hist = ((r1["text_hist"] * r1["size"]) + (r2["text_hist"] * r2["size"])) / new_size

    # calculate the bounding box rectangle
    rect = (min_y, min_x, max_x - min_x, max_y - min_y)

    # create the new region dictionary containing merged information
    rt = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "size": new_size,
        "labels": r1["labels"] + r2["labels"],
        "rect": rect,
        "colour_hist": colour_hist,
        "text_hist": text_hist
    }
    
    return rt 

def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed

        to_remove = []  

        # iterate through the keys in S and check if i or j is in the key
        for key in S: 
            if i in key or j in key:  
                to_remove.append(key) # add the key to the list of keys to remove


        # Task 6: Remove old similarities of related regions

        for key in to_remove:
            del S[key]

        # Task 7: Calculate similarities with the new region

        for key in to_remove:

            # skip if the key is a pair of the new region
            if key in {(i, j), (j, i)}:
                continue

            # get the regions from the key
            region_a, region_b = key

            # if the region is one of the merged regions, use the new region
            if region_a in {i, j}:
                new_region = region_b
            else:
                new_region = region_a
            
            # calculate the similarity with the new region
            S[(t, new_region)] = calc_sim(R[t], R[new_region], imsize)
 

    # Task 8: Generating the final regions from R
    regions = []

    seen_rects = set()  # to avoid duplicates

    for k, r in R.items():
        rect = r["rect"]
        seen_rects.add(rect)

        region_info = {
            "rect": rect,
            "labels": r["labels"],
            "size": r["size"]
        }
        regions.append(region_info)

    return image, regions



