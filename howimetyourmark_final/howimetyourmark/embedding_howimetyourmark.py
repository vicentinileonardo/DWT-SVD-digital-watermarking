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
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

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


    alpha = 2  # 8 is the lower limit that can be used
    n_blocks_to_embed = 512
    block_size = 4
    # spatial_functions = ['average', 'median', 'mean', 'max', 'min', 'gaussian', 'laplacian', 'sobel', 'prewitt', 'roberts']
    spatial_function = 'average'
    spatial_weight = 0.1  # 0: no spatial domain, 1: only spatial domain
    attack_weight = 1.0 - spatial_weight



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


    # find the min blocks (sum or mean of the 64 elements for each block) using sorting (min is best)

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
                             'attack_value': np.average(blank_image[i:i + block_size, j:j + block_size])
                             }
                blocks_to_watermark.append(block_tmp)

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['spatial_value'], reverse=True)
    for i in range(len(blocks_to_watermark)):
        blocks_to_watermark[i]['merit'] = i*spatial_weight

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['attack_value'], reverse=False)
    for i in range(len(blocks_to_watermark)):
        blocks_to_watermark[i]['merit'] += i*attack_weight

    blocks_to_watermark = sorted(blocks_to_watermark, key=lambda k: k['merit'], reverse=True)

    blank_image = np.float64(np.zeros((512, 512)))

    blocks_to_watermark_final = []
    for i in range(n_blocks_to_embed):
        tmp = blocks_to_watermark.pop()
        blocks_to_watermark_final.append(tmp)
        blank_image[tmp['locations'][0]:tmp['locations'][0] + block_size,
        tmp['locations'][1]:tmp['locations'][1] + block_size] = 1

    blocks_to_watermark_final = sorted(blocks_to_watermark_final, key=lambda k: k['locations'], reverse=False)

####################################################################################################################

    divisions = original_image.shape[0] / block_size

    shape_LL_tmp = np.floor(original_image.shape[0]/ (2*divisions))
    shape_LL_tmp = np.uint8(shape_LL_tmp)
    watermarked_image=original_image.copy()
    # loops trough x coordinates of blocks_to_watermark_final

    # svd of watermark_to_embed
    watermark_to_embed = watermark_to_embed.reshape(32,32)
    Uwm, Swm, Vwm = np.linalg.svd(watermark_to_embed)

    for i in range(len(blocks_to_watermark_final)):

        x = np.uint16(blocks_to_watermark_final[i]['locations'][0])
        y = np.uint16(blocks_to_watermark_final[i]['locations'][1])

        #get the block from the original image
        block = original_image[x:x + block_size, y:y + block_size]
        #compute the LL of the block
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]
        # SVD
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)
        Sw = Sc.copy()

        # embedding
        Sw = Sw + Swm[(i*shape_LL_tmp)%32: (shape_LL_tmp+(i*shape_LL_tmp)%32)] * alpha

        LL_new = np.zeros((shape_LL_tmp, shape_LL_tmp))
        LL_new = (Uc).dot(np.diag(Sw)).dot(Vc)
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

