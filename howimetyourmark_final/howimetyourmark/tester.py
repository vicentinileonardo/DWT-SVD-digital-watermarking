import embedding_howimetyourmark, detection_howimetyourmark
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


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
  # attacked.append(resizing(img, 0.1))

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


watermarked = embedding_howimetyourmark.embedding('../sample-images/0000.bmp', 'howimetyourmark.npy')
cv2.imwrite('watermarked.bmp', watermarked)

original = cv2.imread('../sample-images/0000.bmp', 0)
watermarked = cv2.imread('watermarked.bmp', 0)
w = detection_howimetyourmark.wpsnr(original, watermarked)
print('[EMBEDDING] wPSNR: %.2fdB' % w)



#attack
attacked = jpeg_compression(watermarked, 99)
cv2.imwrite('attacked.bmp', attacked)
plt.imshow(attacked)
plt.show()

#
start = time.time()
dec, wpsnr = detection_howimetyourmark.detection('../sample-images/0000.bmp', 'watermarked.bmp', 'attacked.bmp')
print('time consumed: ', time.time() - start)

print(dec)
print(wpsnr)

test_detection('../sample-images/0000.bmp', 'watermarked.bmp')
watermark_ex = detection_howimetyourmark.extraction(original, watermarked, watermarked)
check_mark(np.load('howimetyourmark.npy'), watermark_ex)

# print 2 watermarks
print('watermark 1: ', np.load('howimetyourmark.npy'))
print('watermark 2: ', watermark_ex)
