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

# Detection, with these parameters, passes the 6 checks of the test_detection.py file on Google Drive

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

def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s

def extraction(input1, input2, input3):

    original_image = input1
    watermarked_image = input2
    attacked_image = input3

    alpha = 0.1  # 8 is the lower limit that can be used
    n_blocks_to_embed = 1024
    block_size = 4
    watermark_size = 1024

    # start time
    #start = time.time()

    blocks_with_watermark = []
    divisions = original_image.shape[0] / block_size
    watermark_extracted = np.float64(np.zeros(watermark_size))
    blank_image = np.float64(np.zeros((512, 512)))
    # compute difference between original and watermarked image

    difference = (watermarked_image - original_image)

    # fill blocks in differece where the difference is bigger o less than 0
    for i in range(0, original_image.shape[1], block_size):
        for j in range(0, original_image.shape[0], block_size):
            block_tmp = {'locations': (i, j)}
            if np.average(difference[i:i + block_size, j:j + block_size]) > 0:
                blank_image[i:i + block_size, j:j + block_size] = 1
                blocks_with_watermark.append(block_tmp)
            else:
                blank_image[i:i + block_size, j:j + block_size] = 0

    attacked_image-=np.uint8(blank_image)

####################################################################################################################


    #shape_LL_tmp = np.floor(original_image.shape[0] / divisions)
    #shape_LL_tmp = np.uint8(shape_LL_tmp)

    watermark_extracted = np.zeros(1024)

    watermark_extracted = watermark_extracted.reshape(32,32)

    #print(watermark_extracted)
    for i in range(len(blocks_with_watermark)):
        x = np.uint16(blocks_with_watermark[i]['locations'][0])
        y = np.uint16(blocks_with_watermark[i]['locations'][1])
        #get the block from the attacked image
        block = attacked_image[x:x + block_size, y:y + block_size]
        #compute the LL of the block
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]
        # SVD
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)
        #get the block from the original image
        block_ori = original_image[x:x + block_size, y:y + block_size]
        #compute the LL of the block
        Coefficients_ori = pywt.wavedec2(block_ori, wavelet='haar', level=1)
        LL_ori = Coefficients_ori[0]
        # SVD
        Uc_ori, Sc_ori, Vc_ori = np.linalg.svd(LL_ori)

        Udiff = (Uc - Uc_ori)/alpha
        Sdiff = (Sc - Sc_ori)/alpha
        Vdiff = (Vc - Vc_ori)/alpha

        #Sdiff = Sc_ori-Sc

        shape_LL_tmp = LL_tmp.shape[0]

        x_w = np.uint16((i % (watermark_extracted.shape[0] / shape_LL_tmp)) * shape_LL_tmp)
        y_w = np.uint16(((i // (watermark_extracted.shape[0] / shape_LL_tmp)) * shape_LL_tmp) % 32)

        wm_block = (Udiff).dot(np.diag(Sdiff)).dot(Vdiff) #se mai aggiungere T a vdiff

        print(wm_block)
        #block_limit = np.uint16(watermark_size/n_blocks_to_embed)
        #for px in range(0,block_limit):
        #    watermark_extracted[px + i * block_limit] = Sdiff[px]/ alpha


        watermark_extracted[x_w:x_w + shape_LL_tmp, y_w:y_w + shape_LL_tmp] = wm_block.copy()

    watermark_extracted = watermark_extracted.reshape(1024)
####################################################################################################################

    #end time
    #end = time.time()
    #print('[EXTRACTION] Time: %.2fs' % (end - start))

    #print(watermark_extracted)
    return watermark_extracted

def detection(input1, input2, input3):

    original_image = cv2.imread(input1, 0)
    watermarked_image = cv2.imread(input2, 0)
    attacked_image = cv2.imread(input3, 0)

    # start time
    #start = time.time()

    alpha = 0.1  # 8 is the lower limit that can be used
    n_blocks_to_embed = 1024
    block_size = 4
    watermark_size = 1024
    T = 15.86


    #extract watermark from watermarked image
    watermarked_image_dummy = watermarked_image.copy()
    watermark_extracted_wm = extraction(original_image, watermarked_image, watermarked_image_dummy)

    #starting extraction
    blocks_with_watermark = []
    divisions = original_image.shape[0] / block_size
    watermark_extracted = np.float64(np.zeros(watermark_size))
    blank_image = np.float64(np.zeros((512, 512)))
    # compute difference between original and watermarked image

    difference = (watermarked_image - original_image)

    # fill blocks in differece where the difference is bigger o less than 0
    for i in range(0, original_image.shape[1], block_size):
        for j in range(0, original_image.shape[0], block_size):
            block_tmp = {'locations': (i, j)}
            if np.average(difference[i:i + block_size, j:j + block_size]) > 0:
                blank_image[i:i + block_size, j:j + block_size] = 1
                blocks_with_watermark.append(block_tmp)
            else:
                blank_image[i:i + block_size, j:j + block_size] = 0

    attacked_image -= np.uint8(blank_image)

    ####################################################################################################################

    #shape_LL_tmp = np.floor(original_image.shape[0] / divisions)
    #shape_LL_tmp = np.uint8(shape_LL_tmp)

    watermark_extracted = np.zeros(1024)

    watermark_extracted = watermark_extracted.reshape(32, 32)

    # print(watermark_extracted)
    for i in range(len(blocks_with_watermark)):
        x = np.uint16(blocks_with_watermark[i]['locations'][0])
        y = np.uint16(blocks_with_watermark[i]['locations'][1])
        # get the block from the attacked image
        block = attacked_image[x:x + block_size, y:y + block_size]
        # compute the LL of the block
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]
        # SVD
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)
        # get the block from the original image
        block_ori = original_image[x:x + block_size, y:y + block_size]
        # compute the LL of the block
        Coefficients_ori = pywt.wavedec2(block_ori, wavelet='haar', level=1)
        LL_ori = Coefficients_ori[0]
        # SVD
        Uc_ori, Sc_ori, Vc_ori = np.linalg.svd(LL_ori)

        Udiff = Uc_ori - Uc
        Sdiff = Sc_ori - Sc
        Vdiff = Vc_ori - Vc

        # Sdiff = Sc_ori-Sc

        shape_LL_tmp = LL_tmp.shape[0]

        x_w = np.uint16((i % (watermark_extracted.shape[0] / shape_LL_tmp)) * shape_LL_tmp)
        y_w = np.uint16(((i // (watermark_extracted.shape[0] / shape_LL_tmp)) * shape_LL_tmp) % 32)

        wm_block = (Udiff).dot(np.diag(Sdiff)).dot(Vdiff)  # se mai aggiungere T a vdiff


        #block_limit = np.uint16(watermark_size / n_blocks_to_embed)
        #for px in range(0, block_limit):
        #    watermark_extracted[px + i * block_limit] = Sdiff[px] / alpha


        watermark_extracted[x_w:x_w + shape_LL_tmp, y_w:y_w + shape_LL_tmp] = wm_block.copy()

    watermark_extracted = watermark_extracted.reshape(1024)

####################################################################################################################
    #end of extraction

    sim = similarity(watermark_extracted_wm, watermark_extracted)
    if sim > T:
        watermark_status = 1
    else:
        watermark_status = 0

    output1 = watermark_status
    output2 = wpsnr(watermarked_image, attacked_image)

    # end time
    #end = time.time()
    #print('[DETECTION] Time: %.2fs' % (end - start))

    return output1, output2