import numpy as np
import rawpy
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from pathlib import Path
from PIL import Image, ExifTags
from fractions import Fraction


def demosaic(raw_image):
    "Demosaicing is how we fill in the missing colors for each pixel by estimating them from nearby pixels"
    H, W = raw_image.shape

    padded_img = np.pad(raw_image, pad_width=1, mode='symmetric')  # symmetric padding

    red_mask = np.zeros_like(padded_img)
    green_mask = np.zeros_like(padded_img)
    blue_mask = np.zeros_like(padded_img)

    # BGGR pattern]
    blue_mask[::2, ::2] = True  
    green_mask[::2, 1::2] = True  
    green_mask[1::2, ::2] = True  
    red_mask[1::2, 1::2] = True 

    # convolution kernel for bilinear interpolation
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float64) / 4.0

    # color channels extraction
    blue_channel = padded_img * blue_mask
    green_channel = padded_img * green_mask
    red_channel = padded_img * red_mask

    # apply convolution to each channel
    blue_convolved = convolve(blue_channel, kernel, mode='mirror')
    green_convolved = convolve(green_channel, kernel, mode='mirror')
    red_convolved = convolve(red_channel, kernel, mode='mirror')

    # convolve the masks 
    blue_mask_convolved = convolve(blue_mask, kernel, mode='mirror')
    green_mask_convolved = convolve(green_mask, kernel, mode='mirror')
    red_mask_convolved = convolve(red_mask, kernel, mode='mirror')

    # normalize the interpolated channels-> correct brightness.
    #M: Normalize: Because some pixels have more neighbors than others, divide by the number of contributing neighbors
    blue_interpolated = np.divide(blue_convolved, blue_mask_convolved, where=blue_mask_convolved != 0)
    green_interpolated = np.divide(green_convolved, green_mask_convolved, where=green_mask_convolved != 0)
    red_interpolated = np.divide(red_convolved, red_mask_convolved, where=red_mask_convolved != 0)

    reconstructed_img = np.zeros((H, W, 3), dtype=np.float64)
    reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]  
    reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]  
    reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]  

    return reconstructed_img

def improve_luminosity(image, gamma):
   
    "percentile approach:ignore extreme outliers when defining the scaling range."
    "The majority of pixels get spread more evenly across 0-1 brightness." 

    a = np.percentile(image, 0.01)   # ignore darkest 0.01%
    b = np.percentile(image, 99.99)  # ignore brightest 0.01%
    image = (image - a) / (b - a)    # map these percentiles to 0 and 1
    image[image < 0] = 0
    image[image > 1] = 1

    # gamma correction
    image = np.power(image, gamma)

    # logarithmic enhancement 
    # image_log = np.log(image + 1) / np.log(2)  
    # image = np.clip(image_log, 0, 1)

    return image

def apply_white_balance(image): #M: In a typical photo, if you average all the colors together, they should form a neutral gray
    # calculate mean value of the image (mi)
    mi = np.mean(image)

    # calculate the mean of each channel (mc)
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    # calculate scaling factor for each channel
    scale_r = mi / mean_r
    scale_g = mi / mean_g
    scale_b = mi / mean_b

    # multiply values of channel c by mi / mc and handle out-of-bounds values
    image[:, :, 0] = np.clip(image[:, :, 0] * scale_r, 0, 1)
    image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 1)
    image[:, :, 2] = np.clip(image[:, :, 2] * scale_b, 0, 1)

    return image

def calculate_average_rgb(image):
    avg_r = np.mean(image[:, :, 0])  # red 
    avg_g = np.mean(image[:, :, 1])  # green 
    avg_b = np.mean(image[:, :, 2])  # blue 
    
    return avg_r, avg_g, avg_b

def process_images_for_linearity(image_files_cr3, exposure_times):
    avg_r_values = []
    avg_g_values = []
    avg_b_values = []

    # iterate through each image 
    for cr3_file in image_files_cr3:
        # loading CR3 image 
        raw = rawpy.imread(cr3_file)
        raw_image = raw.raw_image_visible  

        # demosaic the raw CR3 image
        demosaiced_image = demosaic(raw_image)

        # calculate average RGB values 
        raw_avg_r, raw_avg_g, raw_avg_b = calculate_average_rgb(demosaiced_image)

        # append averages to lists
        avg_r_values.append(raw_avg_r)
        avg_g_values.append(raw_avg_g)
        avg_b_values.append(raw_avg_b)

    # plot average RGB values 
    plt.figure(figsize=(10, 6))
    plt.plot(exposure_times, avg_r_values, label="Red Channel", marker='o', color='red')
    plt.plot(exposure_times, avg_g_values, label="Green Channel", marker='o', color='green')
    plt.plot(exposure_times, avg_b_values, label="Blue Channel", marker='o', color='blue')

    plt.xlabel('Exposure Time (seconds)')
    plt.ylabel('Average Pixel Value')
    plt.title('Task5: Sensor Linearity Check')
    plt.legend()
    plt.grid(True)

    # display the plot
    plt.show()

def combine_hdr_images(image_paths, exposure_times):
    hdr_image = None
    t = None  # threshold for replacement

    for idx, path in enumerate(image_paths):
        # load + convert raw image to float
        raw = rawpy.imread(path)
        raw_image = raw.raw_image_visible.astype(np.float64)

        if idx == 0:
            # load brightest raw data (longest exposure)
            hdr_image = raw_image
            t = 0.8 * np.max(hdr_image) #0.8·max(h) to avoid 'plateau' effect
        else:
            # multiply current raw image by exposure difference to the first photo
            scale = exposure_times[0] / exposure_times[idx]
            scaled_image = raw_image * scale

            # values in h which are above a threshold t get replaced by the corresponding values in current image
            mask = hdr_image > t
            hdr_image[mask] = scaled_image[mask]

    return hdr_image

def iCAM06_tone_mapping(hdr_image, output_range=4):

    hdr_image = hdr_image.astype(np.float64)

    # input intensity
    input_intensity = (1 / 61.0) * (20 * hdr_image[:, :, 0] + 40 * hdr_image[:, :, 1] + hdr_image[:, :, 2])
    
    # normalize RGB channels
    r = hdr_image[:, :, 0] / (input_intensity + 1e-8)
    g = hdr_image[:, :, 1] / (input_intensity + 1e-8)
    b = hdr_image[:, :, 2] / (input_intensity + 1e-8)

    # log base + details
    log_intensity = np.log(input_intensity + 1e-8)
    log_base = cv2.bilateralFilter(log_intensity.astype(np.float32), d=9, sigmaColor=0.5, sigmaSpace=5.0) 
    log_details = log_intensity - log_base

    # compression
    compression = np.log(output_range) / (np.max(log_base) - np.min(log_base) + 1e-8)
    log_offset = -np.max(log_base) * compression

    # output intensity
    output_intensity = np.exp(log_base * compression + log_offset + log_details)

    # RGB reconstruction
    tone_mapped = np.stack([r * output_intensity,
                            g * output_intensity,
                            b * output_intensity], axis=-1)

    icam_rgb = np.clip(tone_mapped, 0, 1)
    icam_rgb_uint8 = (icam_rgb * 255).astype(np.uint8)
    return icam_rgb_uint8  

#Task 8

def process_raw(in_path, out_path,
                gamma=0.5,
                do_white_balance=True):
    """
    Reads RAW CR3, demosaics, improves luminosity (gamma), 
    applies white balance, and saves JPG.
    """
    # 1) Load RAW
    raw = rawpy.imread(in_path)
    raw_image = np.array(raw.raw_image_visible)

    # 2) Demosaic
    rgb = demosaic(raw_image)  # uses demosaic()

    # 3) Improve luminosity with gamma & percentiles
    rgb = improve_luminosity(rgb, gamma)

    # 4) White balance 
    if do_white_balance:
        rgb = apply_white_balance(rgb)

    # 5) Scale to [0,255] and save
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(out_path, quality=98)  # high quality

    print(f"Saved processed image to {out_path}")


#Task 9


def read_jpg_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    # luminance (Rec.709)
    Y = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    return img, Y # Returns RGB in [0,1] and luminance Y

def estimate_gamma_from_stack(jpg_paths, times, p_lo=1.0, p_hi=99.0, min_samples=500):
    # jpg_paths: list of paths (strings or Path)
    # times: list/array of exposure times (seconds) matching jpg_paths
    times = np.asarray(times, np.float64)
    if len(jpg_paths) != len(times) or len(times) < 2:
        raise ValueError("Need at least two images and matching exposure times.")

    # sort by exposure time DESC (longest → shortest) so both steps use same order
    order = np.argsort(-times)
    jpg_paths = [jpg_paths[i] for i in order]
    times = times[order]

    def g_for_pair(i, j, lo, hi):
        #get luminace ignore rgb image
        _, Y1 = read_jpg_gray(jpg_paths[i]); _, Y2 = read_jpg_gray(jpg_paths[j])
        lo1, hi1 = np.percentile(Y1, [lo, hi]); lo2, hi2 = np.percentile(Y2, [lo, hi])

        #Creates a boolean mask of pixels that are not clipped in either image
        mask = (Y1 > lo1) & (Y1 < hi1) & (Y2 > lo2) & (Y2 < hi2)
        #1D array of all indices where mask is True
        idx = np.flatnonzero(mask)

        if idx.size < min_samples:
            return None
        #from each pixels randomly select exactly min_samples of those pixels
        selc = np.random.choice(idx, size=min_samples, replace=False)
        #compute the log luminance difference btw i ,j
        r = np.log(Y2.flat[selc] + 1e-8) - np.log(Y1.flat[selc] + 1e-8) 
         #Dividing gives an estimated gamma for each pixel, and taking the median reduces noise.
        return float(np.median(r) / (np.log(times[j]) - np.log(times[i])))

    # try extremes, then adjacent pairs, relax percentiles if needed
    for lo, hi in [(p_lo, p_hi), (0.5, 99.5), (0.1, 99.9)]:
        g = g_for_pair(0, len(times)-1, lo, hi)
        if g is not None:
            return g
        # If extreme fails, try adjacent exposure pairs
        for k in range(len(times)-1):
            g = g_for_pair(k, k+1, lo, hi)
            if g is not None:
                return g

    raise ValueError("No valid pixels to estimate gamma; check images/exposures/order.")


def linearize_jpg(img_rgb, gamma):
    "This function converts JPEG back to linear light so the math is physically correct."
    "Returns a float array in linear light space"
    return np.power(np.clip(img_rgb, 1e-6, 1.0), 1.0/gamma) #Reverses the gamma encoding 

def _lin_luma(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def hdr_from_jpgs(jpg_paths, times, gamma, t_ratio=0.7):
    "Returns the merged HDR image in linear light"

    ref_t = float(times[0])
    img0, _ = read_jpg_gray(jpg_paths[0])
    h = linearize_jpg(img0, gamma).astype(np.float64) #Reverses the gamma encoding by raising each value to the inverse gamma

    # Calculates a luminance threshold e.g 0.8
    Y = _lin_luma(h)
    t = t_ratio * Y.max()              

    for p, t_i in zip(jpg_paths[1:], times[1:]):
        im, _ = read_jpg_gray(p)
        li = linearize_jpg(im, gamma).astype(np.float64)
        scaled = li * (ref_t / float(t_i))
        # Recomputes luminance for the current merged HDR image h
        Yh = _lin_luma(h)
        mask = (Yh > t)                 # H×W mask
        h[mask, :] = scaled[mask, :]    # replace all channels together
    return h


def apply_white_balance_linear(img):
    mi = img.mean()
    out = img.copy().astype(np.float64)
    for c in range(3):
        mc = out[..., c].mean()
        if mc > 1e-9:
            out[..., c] *= (mi / mc)
    return out


def tone_map_log(rgb_hdr): 
    "compress dynamic range using a logarithmic curve so it can fit into a standard display's 0-255 range"
    x = np.clip(rgb_hdr, 1e-6, None)
    y = np.log(x)
    y -= y.min()
    y /= max(y.max(), 1e-9)
    return (y * 255.0).astype(np.uint8)



def main():

    #----------------------------Task 2------------------------------#
    # load the raw image using rawpy
    raw = rawpy.imread('exercise_4_data/02/IMG_4782.CR3')
    raw_image = np.array(raw.raw_image_visible)


    final_image = demosaic(raw_image)
    final_image2 = final_image.copy()
    
    # display the demosaiced image
    plt.imshow(final_image / final_image.max())  # normalize 
    plt.title('Task2: Demosaiced Image')
    plt.axis('off')
    plt.show()

    #----------------------------Task 3-------------------------------#
    gamma_corrected_image = improve_luminosity(final_image, 0.3) #stron brightnening
    #0.8 slight brightening, subtle change, 0.2 too much dark shadowed 0.5 Moderate brightening, more natural look

    #display gamma corrected image
    plt.imshow(gamma_corrected_image)  
    plt.title('Task3: Image after Luminosity Improvement')
    plt.axis('off')
    plt.show()

    #----------------------------Task 4-------------------------------#
    white_balanced_image = apply_white_balance(gamma_corrected_image)

    # display white balanced image
    plt.imshow(white_balanced_image)
    plt.title('Task4: Image after White Balance')
    plt.axis('off')
    plt.show()

    #---------------------------Task 5-------------------------------#
    #M: Sensor Linearty check
    image_files_cr3 = [
        'exercise_4_data/05/IMG_3044.CR3', 'exercise_4_data/05/IMG_3045.CR3', 'exercise_4_data/05/IMG_3046.CR3', 
        'exercise_4_data/05/IMG_3047.CR3', 'exercise_4_data/05/IMG_3048.CR3', 'exercise_4_data/05/IMG_3049.CR3'
     ]

    # given exposure times corresponding to the images
    exposure_times = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]

    # image processing + display the plot 
    process_images_for_linearity(image_files_cr3, exposure_times)

    #---------------------------Task 6-------------------------------#
    base_path = "exercise_4_data/06"
    image_files = [os.path.join(base_path, f"{i:02}.CR3") for i in range(11)]
    exposure_times = [1 / (2 ** i) for i in range(11)]  # 1s to 1/1024s
    #print(f'Exposure times: {exposure_times}')

    # HDR combination
    hdr_raw = combine_hdr_images(image_files, exposure_times)

    # demosaic
    demosaiced_hdr = demosaic(hdr_raw)

    # log tone mapping
    demosaiced_hdr = demosaiced_hdr.astype(np.float32)
    final_image = tone_map_log(demosaiced_hdr)
    log_hdr = np.clip(demosaiced_hdr, 1e-6, None)
    log_hdr = np.log(demosaiced_hdr) #no need to define again here just call function
    log_hdr -= log_hdr.min()
    log_hdr /= log_hdr.max()

    # gamma correction
    gamma_corr_img = improve_luminosity(log_hdr, gamma=0.3)
    # plt.imshow(gamma_corr_img)
    # plt.title('Task6: Image after Gamma Correction')
    # plt.axis('off')
    # plt.show()

    # white balance
    wb_image = apply_white_balance(gamma_corr_img)
    # plt.imshow(wb_image)
    # plt.title('Task6: Image after White Balance')
    # plt.axis('off')
    # plt.show()

    # apply a log scale + normalize the result in [0, 255]
    log_hdr = np.log1p(wb_image)
    log_hdr /= np.max(log_hdr)

    # for visualization
    final_image = (log_hdr * 255).astype(np.uint8)
    plt.imshow(final_image)
    plt.title('Task6: Initial HDR implementation')
    plt.axis('off')
    plt.show()

    # save the resultant image
    output_hdr_img = Image.fromarray(final_image)
    output_path = "exercise_4_data/06/hdr_output.jpg"
    output_hdr_img.save(output_path)

    #---------------------------Task 7-------------------------------#
    icam06_result = iCAM06_tone_mapping(wb_image,32)
    plt.imshow(icam06_result)
    plt.title("Task 7: After iCAM06 method")
    plt.axis('off')
    plt.show()


    #---------------------------Task 8-------------------------------#
    process_raw(
    in_path="exercise_4_data/02/IMG_4782.CR3",
    out_path="exercise_4_data/02/task8_output.jpg",
    gamma=0.5,        
    do_white_balance=True
    )

   #---------------------------Task 9-------------------------------#

    base = Path("ex4_additional_exercise_data")
    # jpgs = sorted([base / f for f in os.listdir(base) if f.lower().endswith(".jpg")])
    jpgs = sorted(p for p in base.glob("A45*.JPG"))


    times = [  # one per JPG, in the SAME order as 'jpgs'
            1.0, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 1/2048
            ]

    # quick sanity check:
    assert len(jpgs) == len(times), f"Found {len(jpgs)} JPGs but {len(times)} times."


    # sort both by exposure (longest→shortest)
    times_np = np.asarray(times, np.float64)
    order = np.argsort(-times_np)
    jpgs  = [jpgs[i] for i in order]
    times = times_np[order].tolist()

    # optional: reproducible gamma sampling
    np.random.seed(0)

    gamma   = estimate_gamma_from_stack(jpgs, times)
    print("Gamma estimate:", gamma)
    if gamma < 1.5 or gamma > 2.6:
        print(f"Gamma {gamma:.3f} out of expected range, forcing 2.2")
        gamma = 2.2


    hdr_lin = hdr_from_jpgs(jpgs, times, gamma)
    hdr_lin = apply_white_balance_linear(hdr_lin)
    output     = tone_map_log(hdr_lin)

    Image.fromarray(output).save(base / "hdr_from_jpg_task9.jpg", quality=99)
    print(f"[Task9] N={len(jpgs)} images, gamma={gamma:.3f}")




if __name__ == '__main__':
    main()
