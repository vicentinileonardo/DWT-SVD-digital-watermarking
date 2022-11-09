import embedding_howimetyourmark, detection_howimetyourmark
import time
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from math import sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

np.set_printoptions(threshold=np.inf)

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

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s

def test_detection(original_image, watermarked_image):
  original = original_image
  watermarked = watermarked_image

  watermarked_image_array = cv2.imread(watermarked_image, 0)

  # 1. TIME REQUIREMENT: the detection should run in < 5 seconds
  start_time = time.time()
  tr, w = detection_howimetyourmark.detection(original, watermarked, watermarked)
  end_time = time.time()
  if (end_time - start_time) > 5:
    print('ERROR! Takes too much to run: ' + str(end_time - start_time))
  else:
    print('OK 1 time requirement')

  # 2. THE WATERMARK MUST BE FOUND IN THE WATERMARKED IMAGE
  if tr == 0:
    print('ERROR! Watermark not found in watermarked image')
  else:
    print('OK 2 watermark in watermarked')

  # 3. THE WATERMARK MUST NOT BE FOUND IN ORIGINAL
  tr, w = detection_howimetyourmark.detection(original, watermarked, original)
  if tr == 1:
    print('ERROR! Watermark found in original')
  else:
    print('OK 3 watermark not found in the original image')

  # 4. CHECK DESTROYED IMAGES
  img = watermarked_image_array.copy()
  attacked = []
  c = 0
  ws = []

  attacked.append(embedding_howimetyourmark.blur(img, 15))
  attacked.append(embedding_howimetyourmark.awgn(img, 50, 123))
  attacked.append(resizing(img, 0.1))

  for i, a in enumerate(attacked):
    aName = 'attacked-%d.bmp' % i
    cv2.imwrite(aName, a)
    tr, w = detection_howimetyourmark.detection(original, watermarked, aName)
    if tr == 1:
      c += 1
      ws.append(w)
  if c > 0:
    print('ERROR! Watermark found in %d destroyed images with ws %s' % (c, str(ws)))
  else:
    print('OK 4 watermark not found in destroyed images')

  # 5. CHECK UNRELATED IMAGES
  files = [os.path.join('TESTImages', f) for f in os.listdir('TESTImages')]
  c = 0
  for f in files:
    unr = cv2.imread(f, 0)
    tr, w = detection_howimetyourmark.detection(original, watermarked, f)
    if tr == 1:
      c += 1
  if c > 0:
    print('ERROR! Watermark found in %d unrelated images' % c)
  else:
    print('OK 5 watermark not found in unrelated images')

def check_mark(X, X_star):
  X_star = np.rint(abs(X_star)).astype(int)
  res = [1 for a, b in zip(X, X_star) if a==b]
  if sum(res) != 1024:
    print('The marks are different, please check your code')
  print(sum(res))

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')
  return attacked




'''ATTACKS PARAMETERS'''
# brute force attack
successful_attacks = []
attacks = ["jpeg_compression","median","blur", "awgn",  "sharpening",  "resizing"]
#attacks = ["resizing"]
##attacks = ["jpeg_compression", "awgn", "blur"]

# setting parameter ranges

# awgn
awgn_std_values = [2.0, 4.0, 10.0, 20.0]
# awgn_seed_values = []
awgn_mean_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# jpeg_compression
jpeg_compression_QF_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 29, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# blur
blur_sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                     2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                     3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
                     [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                     [2, 1], [2, 2]

                     ]

# sharpening
sharpening_sigma_values = [0.01, 0.1, 1, 5, 9, 30, 40, 50, 100]
sharpening_alpha_values = [0.01, 0.1, 1, 5, 9, 30, 40, 50, 100]

# median
median_kernel_size_values = [[1, 3], [1, 5],
                             [3, 1], [3, 3], [3, 5],
                             [5, 1], [5, 3], [5, 5],
                             [7, 1], [7, 3], [7, 5],
                             [9, 1], [9, 3], [9, 5]]

# resizing
resizing_scale_values = [0.01, 0.05, 0.1, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

'''ATTACKS'''

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
  x, y = img.shape
  _x = int(x*scale)
  _y = int(y*scale)

  attacked = cv2.resize(img, (_x, _y))
  attacked = cv2.resize(attacked, (x, y))

  return attacked


def plot_attack(original_image, watermarked_image, attacked_image):
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_image, cmap='gray')
    plt.subplot(132)
    plt.title('Watermarked')
    plt.imshow(watermarked_image, cmap='gray')
    plt.subplot(133)
    plt.title('Attacked')
    plt.imshow(attacked_image, cmap='gray')
    plt.show()


def print_successful_attacks(successful_attacks, image_name='lena.bmp'):
    import json
    output_file = open('Successful_attacks_' + image_name + '.txt', 'w', encoding='utf-8')
    output_file.write(image_name + "\n")

    output_file.write("alpha: " + str(alpha) + "\n")
    output_file.write("block_size: " + str(block_size) + "\n")
    output_file.write("n_blocks_to_embed: " + str(n_blocks_to_embed) + "\n")
    output_file.write("spatial_function: " + str(spatial_function) + "\n")
    output_file.write("spatial_weight: " + str(spatial_weight) + "\n")
    output_file.write("attack_weight: " + str(attack_weight) + "\n")

    for dic in successful_attacks:
        json.dump(dic, output_file)
        output_file.write("\n")

    output_file.close()


def bf_attack(original_image_path, watermarked_image_path):


  original_image = cv2.imread(original_image_path, 0)
  watermarked_image = cv2.imread(watermarked_image_path, 0)


  current_best_wpsnr = 0

  for attack in attacks:
    ########## JPEG ##########
    if attack == 'jpeg_compression':
      for QF_value in reversed(jpeg_compression_QF_values):
        watermarked_to_attack = watermarked_image.copy()
        attacked_image = jpeg_compression(watermarked_to_attack, QF_value)

        cv2.imwrite('attacked_image.bmp', attacked_image)
        watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

        current_attack = {}
        current_attack["Attack_name"] = 'JPEG_Compression'
        current_attack["QF"] = QF_value

        current_attack["WPSNR"] = tmp_wpsnr

        if watermark_status == 0:
          if tmp_wpsnr >= 35.0:
            successful_attacks.append(current_attack)
            if tmp_wpsnr > current_best_wpsnr:
              current_best_wpsnr = tmp_wpsnr
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
            plot_attack(original_image, watermarked_image, attacked_image)
            # break
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')
        else:
          print('[' + str(current_attack) + ']',
                '[watermark_status = ' + str(watermark_status) + '] - FAILED')

    ########## BLUR ##########
    if attack == 'blur':
      for sigma_value in blur_sigma_values:
        watermarked_to_attack = watermarked_image.copy()
        attacked_image = blur(watermarked_to_attack, sigma_value)

        cv2.imwrite('attacked_image.bmp', attacked_image)
        watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

        current_attack = {}
        current_attack["Attack_name"] = 'blur'
        current_attack["sigma"] = sigma_value

        current_attack["WPSNR"] = tmp_wpsnr

        if watermark_status == 0:
          if tmp_wpsnr >= 35.0:
            successful_attacks.append(current_attack)
            if tmp_wpsnr > current_best_wpsnr:
              current_best_wpsnr = tmp_wpsnr
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
            plot_attack(original_image, watermarked_image, attacked_image)
            # break
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')
        else:
          print('[' + str(current_attack) + ']',
                '[watermark_status = ' + str(watermark_status) + '] - FAILED')

    ########## AWGN ##########
    if attack == 'awgn':
      for std_value in awgn_std_values:
        for mean_value in awgn_mean_values:
          watermarked_to_attack = watermarked_image.copy()
          attacked_image = awgn(watermarked_to_attack, std_value, mean_value)

          cv2.imwrite('attacked_image.bmp', attacked_image)
          watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

          current_attack = {}
          current_attack["Attack_name"] = 'awgn'
          current_attack["std"] = std_value
          current_attack["mean"] = mean_value

          current_attack["WPSNR"] = tmp_wpsnr

          if watermark_status == 0:
            if tmp_wpsnr >= 35.0:
              successful_attacks.append(current_attack)
              if tmp_wpsnr > current_best_wpsnr:
                current_best_wpsnr = tmp_wpsnr
              print('[' + str(current_attack) + ']',
                    '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
              plot_attack(original_image, watermarked_image, attacked_image)
              # break
            else:
              print('[' + str(current_attack) + ']',
                    '[watermark_status = ' + str(watermark_status) + '] - FAILED')
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')

    ########## SHARPENING ##########
    if attack == 'sharpening':
      for sigma_value in sharpening_sigma_values:
        for alpha_value in sharpening_alpha_values:
          watermarked_to_attack = watermarked_image.copy()
          attacked_image = sharpening(watermarked_to_attack, sigma_value, alpha_value)

          cv2.imwrite('attacked_image.bmp', attacked_image)
          watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

          current_attack = {}
          current_attack["Attack_name"] = 'Sharpening'
          current_attack["sigma"] = sigma_value
          current_attack["alpha"] = alpha_value


          current_attack["WPSNR"] = tmp_wpsnr

          if watermark_status == 0:
            if tmp_wpsnr >= 35.0:
              successful_attacks.append(current_attack)
              if tmp_wpsnr > current_best_wpsnr:
                current_best_wpsnr = tmp_wpsnr
              print('[' + str(current_attack) + ']',
                    '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
              plot_attack(original_image, watermarked_image, attacked_image)
              break
            else:
              print('[' + str(current_attack) + ']',
                    '[watermark_status = ' + str(watermark_status) + '] - FAILED')
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')

    ########## MEDIAN ##########
    if attack == 'median':
      for kernel_size_value in median_kernel_size_values:
        watermarked_to_attack = watermarked_image.copy()
        attacked_image = median(watermarked_to_attack, kernel_size_value)

        cv2.imwrite('attacked_image.bmp', attacked_image)
        watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

        current_attack = {}
        current_attack["Attack_name"] = 'median'
        current_attack["kernel_size_value"] = kernel_size_value

        current_attack["WPSNR"] = tmp_wpsnr

        if watermark_status == 0:
          if tmp_wpsnr >= 35.0:
            successful_attacks.append(current_attack)
            if tmp_wpsnr > current_best_wpsnr:
              current_best_wpsnr = tmp_wpsnr
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
            plot_attack(original_image, watermarked_image, attacked_image)
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')
        else:
          print('[' + str(current_attack) + ']',
                '[watermark_status = ' + str(watermark_status) + '] - FAILED')

    ########## RESIZING ##########
    if attack == 'resizing':
      for scale_value in resizing_scale_values:
        watermarked_to_attack = watermarked_image.copy()
        attacked_image = resizing(watermarked_to_attack, scale_value)

        cv2.imwrite('attacked_image.bmp', attacked_image)
        watermark_status, tmp_wpsnr = detection_howimetyourmark.detection(original_image_path, watermarked_image_path, 'attacked_image.bmp')

        current_attack = {}
        current_attack["Attack_name"] = 'resizing'
        current_attack["scale"] = scale_value

        current_attack["WPSNR"] = tmp_wpsnr

        if watermark_status == 0:
          if tmp_wpsnr >= 35.0:
            successful_attacks.append(current_attack)
            if tmp_wpsnr > current_best_wpsnr:
              current_best_wpsnr = tmp_wpsnr
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - !!!SUCCESS!!!')
            plot_attack(original_image, watermarked_image, attacked_image)
          else:
            print('[' + str(current_attack) + ']',
                  '[watermark_status = ' + str(watermark_status) + '] - FAILED')
        else:
          print('[' + str(current_attack) + ']',
                '[watermark_status = ' + str(watermark_status) + '] - FAILED')




###############################################################################################################

##EMBEDDING
original_image_path_1 = 'utilities/originals/lena.bmp'
#original_image_path_2 = 'utilities/originals/tree.bmp'
#original_image_path_3 = 'utilities/originals/rollercoaster.bmp'

original = cv2.imread(original_image_path_1, 0)

watermarked = embedding_howimetyourmark.embedding(original_image_path_1, 'howimetyourmark.npy')
#cv2.imwrite('utilities/watermarked/howimetyourmark/howimetyourmark_buildings.bmp', watermarked)
#cv2.imwrite('utilities/watermarked/howimetyourmark/howimetyourmark_tree.bmp', watermarked)
#cv2.imwrite('utilities/watermarked/howimetyourmark/howimetyourmark_rollercoaster.bmp', watermarked)



#test_detection(original_image_path_1, 'utilities/watermarked/howimetyourmark/tree_howimetyourmark.bmp')
#watermark_ex = detection_howimetyourmark.extraction(original, watermarked, watermarked)
#check_mark(np.load('howimetyourmark.npy'), watermark_ex)

#bf_attack(original_image_path_1, 'utilities/watermarked/howimetyourmark/buildings_howimetyourmark.bmp')
