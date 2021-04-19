import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from skimage.feature import hog
from skimage.io import imread
from skimage import color
dirImg = os.listdir("posImg")
dirMaskImg = os.listdir("posImgMasks")
histograms_Full = []
histograms_Masked = []
fd, hog_image = hog(imread('posImg/FudanPed00001.png'), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.axis("off")
hog = cv2.HOGDescriptor()
plt.imshow(hog_image, cmap="gray")

gray = [color.rgb2gray(imread('posImg/' + i)) for i in dirImg]
plt.imshow( gray[51])
plt.show()
masked = []
hist = []
for i in range(len(dirImg)):
    img = cv2.imread('posImg/' + dirImg[i], 0)
    mask = cv2.imread('posImgMasks/' + dirMaskImg[i], 0)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked.append(masked_img)
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], masked_img, [256], [0, 256])
    hist.append(hog.compute(masked_img, (8,8), (8,8), ((0,0),)))
print(hist)

#hog_img = []
#hog_feat = []

#for image in masked:
 #   fd, hog_image = hog(image)
   # hog_img.append(hog_image)
  #  hog_feat.append(fd)
#plt.imshow(hog_image[51])
#plt.show()
    #print(dirImg[i])

img = cv2.imread('posImg/FudanPed00001.png', 0)
mask2 = cv2.imread('FudanPed00001_mask.png', 0)
mask = np.zeros(img.shape[:2], np.uint8)
mask[302:431, 160:182] = 255
masked_img = cv2.bitwise_and(img,img, mask=mask2)

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],masked_img,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask2,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()