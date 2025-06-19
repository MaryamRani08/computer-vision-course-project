import numpy as np
import rawpy
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

def demosaic(raw_image):
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

    # normalize the interpolated channels
    blue_interpolated = np.divide(blue_convolved, blue_mask_convolved, where=blue_mask_convolved != 0)
    green_interpolated = np.divide(green_convolved, green_mask_convolved, where=green_mask_convolved != 0)
    red_interpolated = np.divide(red_convolved, red_mask_convolved, where=red_mask_convolved != 0)

    reconstructed_img = np.zeros((H, W, 3), dtype=np.float64)
    reconstructed_img[:, :, 0] = red_interpolated[1:-1, 1:-1]  
    reconstructed_img[:, :, 1] = green_interpolated[1:-1, 1:-1]  
    reconstructed_img[:, :, 2] = blue_interpolated[1:-1, 1:-1]  

    return reconstructed_img

def improve_luminosity(image, gamma):
   
    # percentile approach
    a = np.percentile(image, 0.01)
    b = np.percentile(image, 99.99)
    image = (image - a) / (b - a)
    image[image < 0] = 0
    image[image > 1] = 1

    # gamma correction
    image = np.power(image, gamma)

    # logarithmic enhancement 
    # image_log = np.log(image + 1) / np.log(2)  
    # image = np.clip(image_log, 0, 1)

    return image

def apply_white_balance(image):
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
    gamma_corrected_image = improve_luminosity(final_image, 0.3)

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
    demosaiced_hdr = np.clip(demosaiced_hdr, 1e-6, None)
    log_hdr = np.log(demosaiced_hdr) 
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
    # normalize 
    final_image2 = (final_image2 / final_image2.max()) * 255
    final_image2 = final_image2.astype(np.uint8)  

    output_path = "exercise_4_data/02/demosaiced_image_task8.jpg"
    image_to_save = Image.fromarray(final_image2)
    image_to_save.save(output_path)

if __name__ == '__main__':
    main()
