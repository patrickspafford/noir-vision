import cv2 as cv
import random
from os import makedirs, listdir
from os.path import join, exists
from kmeans2 import transform_kmeans
from pca import transform_pca
from imutils.paths import list_files
from detection_baseline import find_people_in_frame


def run_live_trained_svm(output_path, seconds_per_frame, classifier, preprocessor, preprocessor_k=5):
    cam = cv.VideoCapture(0)
    try:
        if not exists(output_path):
            makedirs(output_path)
    except OSError:
        print('Error when creating directory of extracted frames.')
    count = 0
    while(True):
        _, frame = cam.read()
        if count > 0 and count % (30 * seconds_per_frame) == 0:
            if preprocessor:
                frame = preprocessor(frame, preprocessor_k)
            frame = classifier(frame)
            name = join(output_path, 'frame_' + str(count) + '.jpg')
            cv.imwrite(name, frame)
        count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()


def run_offline_trained_svm(source_path, output_path, classifier, preprocessor,  preprocessor_k=5, seconds_per_frame=1):
    cam = cv.VideoCapture(source_path)
    try:
        if not exists(output_path):
            makedirs(output_path)
    except OSError:
        print('Error when creating directory of extracted frames.')
    count = 0
    while(True):
        ret, frame = cam.read()
        if ret:
            if preprocessor:
                frame = preprocessor(frame, preprocessor_k)
            frame = classifier(frame)
            name = join(output_path, 'frame_' + str(count) + '.jpg')
            cv.imwrite(name, frame)
            count += 30 * seconds_per_frame
            cam.set(1, count)
        else:
            break
    cam.release()
    cv.destroyAllWindows()


"""
for image in list_files('/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir'):
    run_offline_trained_svm(image, f'/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir_result/no_pre/result_{random.randint(1, 2000)}',
                            find_people_in_frame, None, 20, 1)

for image in list_files('/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir'):
    run_offline_trained_svm(image, f'/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir_result/kmeans/result_{random.randint(1, 2000)}',
                            find_people_in_frame, transform_kmeans, 20, 1)
"""

for image in list_files('/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir'):
    run_offline_trained_svm(image, f'/Users/patrickspafford/Desktop/Machine Learning/finalProject/ir_result/pca/result_{random.randint(1, 2000)}',
                            find_people_in_frame, transform_pca, 100, 1)
