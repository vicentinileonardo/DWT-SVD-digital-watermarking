import time
import random
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
import embedding_howimetyourmark, detection_howimetyourmark


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



def compute_roc():
    # start time
    start = time.time()
    from sklearn.metrics import roc_curve, auc

    sample_images = []
    # loop for importing images from sample_images folder
    for filename in os.listdir('sample_images'):
        if filename.endswith(".bmp"):
            path_tmp = os.path.join('sample_images', filename)
            sample_images.append(path_tmp)

    sample_images.sort()

    # generate your watermark (if it is necessary)
    watermark_size = 1024
    watermark_path = "utilities/howimetyourmark.npy"
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
        watermarked_image = embedding_howimetyourmark.embedding(original_image, watermark_path)

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
            w_ex_atk = detection_howimetyourmark.extraction(original_image, watermarked_image, attacked_image)

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

    idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    # end time
    end = time.time()
    print('[COMPUTE ROC] Time: %0.2f seconds' % (end - start))

compute_roc()

#13.71 con alpha = 2, 512 blocks, 0.1 spatial
#13.45 con alpha = 2.25, 512 blocks, 0.33 spatial
#14.07 alpha=4, 64 blocks, 0.33 spatial

#14.86 alpha=6, 32 blocks, 0.33 spatial, fpr 0.05 0 attacchi
#14.82 alpha=5, 32blocks, 0.33 spatial, fpr 0.05 | (rifacendo la roc, 14.67) 0 attacchi, 1 embedding sotto i 66
#14.79 alpha=4.85, 32blocks, 0.33 spatial, fpr 0.05, | 1 embedding sotto i 66
#14.74 alpha=5.15, 32blocks, 0.33 spatial, fpr 0.05, | 2 attacchi
#14.79 alpha=5.25, 32blocks, 0.33 spatial, fpr 0.05, | 2 attacchi
#14.66 alpha=4.5, 64blocks, 0.33 spatial, fpr 0.05, tanti embedding sotto i 65
#14.80 alpha=5.25, 32blocks, 0.25 spatial, fpr 0.05
#14.84 alpha=5, 32blocks, 0.2 spatial, fpr 0.05
#14.83 alpha=5, 32blocks, 0.15 spatial, fpr 0.05
#15.08 alpha=6, 16blocks, 0.33 spatial, fpr 0.05
#14.96 alpha=5.15, 32blocks, 0.20 spatial, fpr 0.05
#14.88 alpha=5.15, 32blocks, 0.15 spatial, fpr 0.05
#15.02 alpha=5.75, 32blocks, 0.15 spatial, fpr 0.05
#14.80 alpha=5.75, 32blocks, 0.33 spatial, fpr 0.05
#14.88 alpha=5.85 32blocks, 0.33 spatial, fpr 0.05


#14.57 alpha=6, 32blocks, 0.50 spatial, fpr 0.07
#14.63 alpha=6, 32blocks, 0.40 spatial, fpr 0.07
#14.66 alpha=6, 32blocks, 0.30 spatial, fpr 0.07
#14.66 alpha=6, 32blocks, 0.25 spatial, fpr 0.07
#14.77 alpha=6, 32blocks, 0.20 spatial, fpr 0.07
#14.72 alpha=6, 32blocks, 0.15 spatial, fpr 0.07
#14.70 alpha=6, 32blocks, 0.10 spatial, fpr 0.07
#14.83 alpha=6, 32blocks, 0.05 spatial, fpr 0.07
#15.20 alpha=6, 32blocks, 0.00 spatial, fpr 0.07


