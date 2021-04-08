import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import ticker
import os
from skimage.feature import hog
from skimage.io import imread
from skimage import color


def show_image_histogram_2d(image, bins=32, tick_spacing=5):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    channels_mapping = {0: 'B', 1: 'G', 2: 'R'}
    for i, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
        hist = cv2.calcHist([image], channels, None, [bins] * 2, [1, 256] * 2)

        channel_x = channels_mapping[channels[0]]
        channel_y = channels_mapping[channels[1]]

        ax = axes[i]
        ax.set_xlim([0, bins - 1])
        ax.set_ylim([0, bins - 1])

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')
        ax.set_title(f'2D Color Histogram for {channel_x} and '
                     f'{channel_y}')

        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))

        im = ax.imshow(hist)

    fig.colorbar(im, ax=axes.ravel().tolist(),
                 orientation='orizontal')
    fig.suptitle(f'2D Color Histograms with {bins} bins',
                 fontsize=16)
    plt.show()



dirImg = os.listdir("croppedMasked")
dirMaskImg = os.listdir("posImgMasks")
histograms_Full = []
histograms_Masked = []
fd, hog_image = hog(imread('croppedMasked/FudanPed00002.png'), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.axis("off")
histograms_Masked.append(hog_image)
hogD = cv2.HOGDescriptor()
#plt.imshow(hog_image, cmap="gray")

#gray = [color.rgb2gray(imread('croppedMask/' + i)) for i in dirImg]
#plt.imshow( gray[51])
#plt.show()
imgs = []
hist = []
_hog = []
hog2 = []
fds = []
mas = cv2.imread('croppedMasked/FudanPed00002.png', 0)
for i in range(len(dirImg)):
    img = cv2.imread('croppedMasked/' + dirImg[i])
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog2.append(hog_image)

    #mask = cv2.imread('posImgMasks/' + dirMaskImg[i], 0)
    #masked_img = cv2.bitwise_and(img, img, mask=mask)
    hist_full = cv2.calcHist([img], [0], None, [256], [1, 256])
    fds.append(fd)
    imgs.append(img)
    hist.append(hist_full)
    _hog.append(np.transpose(hogD.compute(img, (8,8), (8,8), ((0,0),))))
print(hist[1])
print(hist[1].shape[0])
print(fds[1])
#show_image_histogram_2d(imgs[1])
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

hist_full = cv2.calcHist(hist[2],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],None,[256],[0,256])

for i in range(len(imgs)):
    plt.subplot(221), plt.imshow(imgs[i], 'gray')
    plt.subplot(222), plt.imshow(hog2[i])
    plt.subplot(223), plt.plot(fds[i])
    plt.subplot(224), plt.plot(hist[i])
    plt.xlim([0,256])
    plt.show()
