import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from os import listdir, mkdir
from os.path import isfile, join, isdir
"""
img = cv.imread(
    '/Users/patrickspafford/Desktop/Machine Learning/finalProject/iron/rdj.jpg')
plt.imshow(img)
blue, green, red = cv.split(img)
pca = PCA(20)


red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)
"""


def save_pca_image(source_path, output_path, principalComponents=10):
    img = cv.imread(source_path)
    blue, green, red = cv.split(img)
    pca = PCA(principalComponents)
    red_inverted = pca.inverse_transform(pca.fit_transform(red))
    green_inverted = pca.inverse_transform(pca.fit_transform(green))
    blue_inverted = pca.inverse_transform(pca.fit_transform(blue))
    img_compressed = (
        np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)
    cv.imwrite(output_path, img_compressed)


def transform_pca(frame, principalComponents=10):
    img = frame
    blue, green, red = cv.split(img)
    pca = PCA(principalComponents)
    red_inverted = pca.inverse_transform(pca.fit_transform(red))
    green_inverted = pca.inverse_transform(pca.fit_transform(green))
    blue_inverted = pca.inverse_transform(pca.fit_transform(blue))
    img_compressed = (
        np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)
    return img_compressed


def save_pca_directory(source_folder, output_folder, principalComponents=10):
    image_files = [f for f in listdir(
        source_folder) if isfile(join(source_folder, f))]
    if not isdir(output_folder):
        mkdir(output_folder)
    for image in image_files:
        source_path = join(source_folder, image)
        output_path = join(output_folder, image)
        save_pca_image(source_path, output_path, principalComponents)


def save_multiple_pca_dir(source_folder, output_folder, pca_step=5, pca_min=1, pca_max=50):
    if not isdir(output_folder):
        mkdir(output_folder)
    for p in range(pca_min, pca_max, pca_step):
        output_folder_for_pca = join(output_folder, f'PCA={p}/')
        save_pca_directory(source_folder, output_folder_for_pca, p)


"""
save_multiple_pca_dir('/Users/patrickspafford/Desktop/Machine Learning/finalProject/iron/',
                      '/Users/patrickspafford/Desktop/Machine Learning/finalProject/iron_pca/', 50, 1, 500)
"""
