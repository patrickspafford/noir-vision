import cv2 as cv
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir


def save_kmeans_image(source_path, k, output_path):
    ori_image = cv.imread(source_path)
    img = cv.cvtColor(ori_image, cv.COLOR_BGR2RGB)
    vectorized_img = img.reshape((-1, 3))
    new_vectorized_img = np.float32(vectorized_img)
    stopping_criteria = (cv.TERM_CRITERIA_EPS +
                         cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    _, label, center = cv.kmeans(
        new_vectorized_img, k, None, stopping_criteria, attempts, cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    misshapen_result = center[label.flatten()]
    result = misshapen_result.reshape((img.shape))
    cv.imwrite(output_path, result)


def save_kmeans_directory(source_folder, k, output_folder):
    image_files = [f for f in listdir(
        source_folder) if isfile(join(source_folder, f))]
    if not isdir(output_folder):
        mkdir(output_folder)
    for image in image_files:
        source_path = join(source_folder, image)
        output_path = join(output_folder, image)
        save_kmeans_image(source_path, k, output_path)


def save_multiple_kmeans_dir(source_folder, output_folder, k_step, k_min, k_max):
    if not isdir(output_folder):
        mkdir(output_folder)
    for k in range(k_min, k_max, k_step):
        output_folder_for_k = join(output_folder, f'K={k}/')
        save_kmeans_directory(source_folder, k, output_folder_for_k)


"""
save_multiple_kmeans_dir('/Users/patrickspafford/Desktop/Machine Learning/finalProject/iron',
                         '/Users/patrickspafford/Desktop/Machine Learning/finalProject/ironK', 2, 2, 5)
"""
