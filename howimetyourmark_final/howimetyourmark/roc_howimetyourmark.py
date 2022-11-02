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


def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s

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

def random_attack(img):
  i = random.randint(1,6)
  if i==1:
    attacked = awgn(img, 5.0, 123)
  elif i==2:
    attacked = blur(img, [3, 2])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 5])
  elif i==5:
    attacked = resizing(img, 0.5)
  elif i==6:
    attacked = jpeg_compression(img, 75)
  return attacked

def embedding(original_image, watermark_path="howimetyourmark.npy" ):

    original_image = cv2.imread(original_image, 0)

    alpha = 12  # 8 is the lower limit that can be used
    n_blocks_to_embed = 1024
    block_size = 4
    # spatial_functions = ['average', 'median', 'mean', 'max', 'min', 'gaussian', 'laplacian', 'sobel', 'prewitt', 'roberts']
    spatial_function = 'average'
    spatial_weight = 0.5  # 0: no spatial domain, 1: only spatial domain
    attack_weight = 1.0 - spatial_weight

    watermark_size = 1024
    watermark_to_embed = np.load(watermark_path)

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

        for px in range(0, np.uint16(watermark_size/n_blocks_to_embed)):
            if watermark_to_embed[np.uint16(px + (i * np.uint16(watermark_size/n_blocks_to_embed)))] == 1:
                Sw[px] += alpha

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
    #w = wpsnr(original_image, watermarked_image)
    #print('[EMBEDDING] wPSNR: %.2fdB' % w)

    return watermarked_image

def extraction(input1, input2, input3):

    original_image = input1
    watermarked_image = input2
    attacked_image = input3

    alpha = 12  # 8 is the lower limit that can be used
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


    shape_LL_tmp = np.floor(original_image.shape[0] / divisions)
    shape_LL_tmp = np.uint8(shape_LL_tmp)

    watermark_extracted = np.zeros(1024)
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

        Sdiff = Sc_ori-Sc

        block_limit = np.uint16(watermark_size/n_blocks_to_embed)

        for px in range(0,block_limit):
            watermark_extracted[px + i * block_limit] = Sdiff[px]/ alpha

####################################################################################################################

    #end time
    #end = time.time()
    #print('[EXTRACTION] Time: %.2fs' % (end - start))

    return watermark_extracted

def compute_roc():
    # start time
    start = time.time()
    from sklearn.metrics import roc_curve, auc

    sample_images = []
    # loop for importing images from sample-images folder
    for filename in os.listdir('../sample-images'):
        if filename.endswith(".bmp"):
            path_tmp = os.path.join('../sample-images', filename)
            sample_images.append(path_tmp)

    sample_images.sort()

    # generate your watermark (if it is necessary)
    watermark_size = 1024
    watermark_path = "howimetyourmark.npy"
    watermark = np.load(watermark_path)

    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores = []
    labels = []

    for i in range(0, len(sample_images)):

        original_image = sample_images[i]
        watermarked_image = embedding(original_image, watermark_path)

        original_image = cv2.imread(original_image, 0)
        print(sample_images[i])
        #plot original and watermarked image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original image')
        plt.subplot(1, 2, 2)
        plt.imshow(watermarked_image, cmap='gray')
        plt.title('Watermarked image')
        plt.show()

        sample = 0
        while sample <= 9:
            # fakemark is the watermark for H0
            fakemark = np.random.uniform(0.0, 1.0, watermark_size)
            fakemark = np.uint8(np.rint(fakemark))

            # random attack to watermarked image (you can modify it)
            attacked_image = random_attack(watermarked_image)

            # extract attacked watermark
            w_ex_atk = extraction(original_image, watermarked_image, attacked_image)

            # compute similarity H1
            scores.append(similarity(watermark, w_ex_atk))
            labels.append(1)
            # compute similarity H0
            scores.append(similarity(fakemark, w_ex_atk))
            labels.append(0)
            sample += 1

    # print the scores and labels
    print('Scores:', scores)
    print('Labels:', labels)


    # compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    idx_tpr = np.where((fpr - 0.09) == min(i for i in (fpr - 0.09) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    # end time
    end = time.time()
    print('[COMPUTE ROC] Time: %0.2f seconds' % (end - start))

compute_roc()


