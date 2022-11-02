import time
import random
import cv2
import os
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from math import sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

# Embedding strategy: DWT-SVD with local selection of blocks based on a spatial function and attacks

def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    csf = np.genfromtxt('utility/csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels


def edge_detection(original_image):

    image = original_image
    #image = cv2.imread(original_image, 0)

    block_size = 4

    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.show()

    edged = cv2.Canny(blurred, 30, 150)
    plt.imshow(edged, cmap='gray')
    plt.title('Edged Image')
    plt.show()

    # threshold the image by setting all pixel values less than 225
    # to 255 (white; foreground) and all pixel values >= 225 to 255
    # (black; background), thereby segmenting the image
    thresh = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)[1]
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')
    plt.show()



    edge_mask = np.zeros((512, 512))
    edge_detection_values = np.zeros((512, 512))

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            #avg_array.append(np.average(thresh[i:i + block_size, j:j + block_size]))
            surrounding_blocks_avgs = []
            thresh[i:i + block_size, j:j + block_size]
            if i - block_size >= 0:
                surrounding_blocks_avgs.append(np.average(thresh[i - block_size:i, j:j + block_size])) # top
            if i + block_size < image.shape[0]:
                surrounding_blocks_avgs.append(np.average(thresh[i + block_size:i + 2 * block_size, j:j + block_size])) # bottom
            if j - block_size >= 0:
                surrounding_blocks_avgs.append(np.average(thresh[i:i + block_size, j - block_size:j])) # left
            if j + block_size < image.shape[1]:
                surrounding_blocks_avgs.append(np.average(thresh[i:i + block_size, j + block_size:j + 2 * block_size])) # right
            if i - block_size >= 0 and j - block_size >= 0:
                surrounding_blocks_avgs.append(np.average(thresh[i - block_size:i, j - block_size:j])) # top left
            if i - block_size >= 0 and j + block_size < image.shape[1]:
                surrounding_blocks_avgs.append(np.average(thresh[i - block_size:i, j + block_size:j + 2 * block_size])) # top right
            if i + block_size < image.shape[0] and j - block_size >= 0:
                surrounding_blocks_avgs.append(np.average(thresh[i + block_size:i + 2 * block_size, j - block_size:j])) # bottom left
            if i + block_size < image.shape[0] and j + block_size < image.shape[1]:
                surrounding_blocks_avgs.append(np.average(thresh[i + block_size:i + 2 * block_size, j + block_size:j + 2 * block_size])) # bottom right

            surrounding_blocks_avgs = np.array(surrounding_blocks_avgs)

            #if block is not an edge, so white and surrounded by blacks then can be chosen
            if np.average(thresh[i:i + block_size, j:j + block_size]) == 255 and np.average(surrounding_blocks_avgs) < 200:
                edge_mask[i:i + block_size, j:j + block_size] = 0
                edge_detection_values[i:i + block_size, j:j + block_size] = np.average(surrounding_blocks_avgs)
            else:
                edge_mask[i:i + block_size, j:j + block_size] = 1
                edge_detection_values[i:i + block_size, j:j + block_size] = 255


    plt.imshow(edge_mask, cmap='gray')
    plt.title('Edge Mask')
    plt.show()

    plt.imshow(edge_detection_values, cmap='gray')
    plt.title('Edge Detection Values')
    plt.show()

    return edge_detection_values

def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def awgn(img, std, seed):
    mean = 0.0
    # np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def embedding(original_image, watermark_path="howimetyourmark.npy" ):

    original_image = cv2.imread(original_image, 0)

    watermark_size = 1024
    watermark_to_embed = np.load(watermark_path)


    alpha = 10  # 8 is the lower limit that can be used
    n_blocks_to_embed = 1024
    block_size = 4
    # spatial_functions = ['average', 'median', 'mean', 'max', 'min', 'gaussian', 'laplacian', 'sobel', 'prewitt', 'roberts']
    spatial_function = 'average'

    spatial_weight = 0.0  # 0: no spatial domain, 1: only spatial domain
    edge_detection_weight = 1
    attack_weight = 1.0 - spatial_weight - edge_detection_weight



    blocks_to_watermark = []

    blank_image = np.float64(np.zeros((512, 512)))

    start = time.time()

    #QF = [5,6, 7, 8,9, 10]
    #for qf in QF:
    #    attacked_image_tmp = jpeg_compression(original_image, qf)
     #   blank_image += np.abs(attacked_image_tmp - original_image)

    blur_sigma_values = [0.1, 0.5, 1, 2, [1, 1], [2, 1]]
    for sigma in blur_sigma_values:
        attacked_image_tmp = blur(original_image, sigma)
        blank_image += np.abs(attacked_image_tmp - original_image)

    kernel_size = [3, 5, 7, 9, 11]
    for k in kernel_size:
        attacked_image_tmp = median(original_image, k)
        blank_image += np.abs(attacked_image_tmp - original_image)

    awgn_std = [0.1, 0.5, 2, 5, 10]
    for std in awgn_std:
        attacked_image_tmp = awgn(original_image, std, 0)
        blank_image += np.abs(attacked_image_tmp - original_image)

    sharpening_sigma_values = [0.1, 0.5, 2, 100]
    sharpening_alpha_values = [0.1, 0.5, 1, 2]
    for sharpening_sigma in sharpening_sigma_values:
        for sharpening_alpha in sharpening_alpha_values:
            attacked_image_tmp = sharpening(original_image, sharpening_sigma, sharpening_alpha)
            blank_image += np.abs(attacked_image_tmp - original_image)

    resizing_scale_values = [0.5, 0.75, 0.9, 1.1, 1.5]
    for scale in resizing_scale_values:
        attacked_image_tmp = cv2.resize(original_image, (0, 0), fx=scale, fy=scale)
        attacked_image_tmp = cv2.resize(attacked_image_tmp, (512, 512))
        blank_image += np.abs(attacked_image_tmp - original_image)
    #plot blank image
    #plt.imshow(blank_image, cmap='gray')
    #plt.show()

    # end time
    end = time.time()
    #print("[EMBEDDING] Time of attacks for embedding: " + str(end - start))
    #print('[EMBEDDING] Spatial function:', spatial_function)

    edge_detection_values = edge_detection(original_image)

    for i in range(0, original_image.shape[0], block_size):
        for j in range(0, original_image.shape[1], block_size):

            if np.mean(original_image[i:i + block_size, j:j + block_size]) < 230 and np.mean(original_image[i:i + block_size, j:j + block_size]) > 10:
                if spatial_function == 'average':
                    spatial_value = np.average(original_image[i:i + block_size, j:j + block_size])
                elif spatial_function == 'median':
                    spatial_value = np.median(original_image[i:i + block_size, j:j + block_size])
                elif spatial_function == 'mean':
                    spatial_value = np.mean(original_image[i:i + block_size, j:j + block_size])

                block_tmp = {'locations': (i, j),
                             'spatial_value': spatial_value,
                             'attack_value': np.average(blank_image[i:i + block_size, j:j + block_size]),
                             'edge_detection_value': np.average(edge_detection_values[i:i + block_size, j:j + block_size])
                             }
                blocks_to_watermark.append(block_tmp)

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['spatial_value'], reverse=True)
    for i in range(len(blocks_to_watermark)):
        blocks_to_watermark[i]['merit'] = i*spatial_weight

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['attack_value'], reverse=False)
    for i in range(len(blocks_to_watermark)):
        blocks_to_watermark[i]['merit'] += i*attack_weight

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['edge_detection_value'], reverse=True)
    for i in range(len(blocks_to_watermark)):
        blocks_to_watermark[i]['merit'] += i*edge_detection_weight

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['merit'], reverse=True)


    # blocks_to_watermark = blocks_to_watermark[:n_blocks_to_embed]

    blank_image = np.float64(np.zeros((512, 512)))

    blocks_to_watermark_final = []
    for i in range(n_blocks_to_embed):
        tmp = blocks_to_watermark.pop()
        blocks_to_watermark_final.append(tmp)
        blank_image[tmp['locations'][0]:tmp['locations'][0] + block_size,
        tmp['locations'][1]:tmp['locations'][1] + block_size] = 1

    blocks_to_watermark_final = sorted(blocks_to_watermark_final, key=lambda k: k['locations'], reverse=False)
    #print(blocks_to_watermark_final)

####################################################################################################################

    divisions = original_image.shape[0] / block_size

    #shape_LL_tmp = np.floor(original_image.shape[0]/ (2*divisions))
    #shape_LL_tmp = np.uint8(shape_LL_tmp)
    watermarked_image=original_image.copy()

    watermark_to_embed = watermark_to_embed.reshape(32, 32)

    # loops trough x coordinates of blocks_to_watermark_final
    for i in range(len(blocks_to_watermark_final)):

        x = np.uint16(blocks_to_watermark_final[i]['locations'][0])
        y = np.uint16(blocks_to_watermark_final[i]['locations'][1])


        Sw = Sw + alpha*Swm
        Uw = Uw + alpha*Uwm
        Vw = Vw + alpha*Vwm

        LL_new = np.zeros((shape_LL_tmp, shape_LL_tmp))

        #LL_new = (Uc).dot(np.diag(Sw)).dot(Vc)
        LL_new = (Uw).dot(np.diag(Sw)).dot(Vw)

        #compute the new block
        Coefficients[0] = LL_new
        block_new = pywt.waverec2(Coefficients, wavelet='haar')
        #replace the block in the original image
        watermarked_image[x:x + block_size, y:y + block_size] = block_new


####################################################################################################################

    watermarked_image = np.uint8(watermarked_image)

    difference = (-watermarked_image + original_image) * np.uint8(blank_image)
    watermarked_image = original_image + difference
    watermarked_image += np.uint8(blank_image)

    # Compute quality
    w = wpsnr(original_image, watermarked_image)
    print('[EMBEDDING] wPSNR: %.2fdB' % w)

    return watermarked_image

np.set_printoptions(threshold=np.inf)

