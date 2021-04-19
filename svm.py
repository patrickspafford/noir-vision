import cv2 as cv
from os import makedirs
from os.path import join, exists
from kmeans2 import transform_kmeans
from pca import transform_pca
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
        if count % (30 * seconds_per_frame) == 0:
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


run_live_trained_svm('/Users/patrickspafford/Desktop/Machine Learning/finalProject/test2',
                     3, find_people_in_frame, transform_kmeans, 10)
