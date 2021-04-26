
import cv2 as cv
from os import makedirs
from os.path import join, exists


def extract_images_from_video(source_path, output_path, seconds_per_frame=1):
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
            name = join(output_path, 'frame_' + str(count) + '.jpg')
            print('Creating...' + name)
            cv.imwrite(name, frame)
            count += 30 * seconds_per_frame
            cam.set(1, count)
        else:
            break
    cam.release()
    cv.destroyAllWindows()


def get_images_from_live_video(output_path, seconds_per_frame=1):
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
            name = join(output_path, 'frame_' + str(count) + '.jpg')
            print('Creating...' + name)
            cv.imwrite(name, frame)
        count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()


"""
def run_trained_svm(output_path, seconds_per_frame, classifier, preprocessor, preprocessor_k=5):
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
            if (classifier(frame)):
                name = join(output_path, 'frame_' + str(count) + '.jpg')
                print('Found person in...' + name)
                cv.imwrite(name, frame)
        count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()
"""


"""
extract_images_from_video('/Users/patrickspafford/Desktop/Machine Learning/finalProject/sample_640x360.mp4',
                          '/Users/patrickspafford/Desktop/Machine Learning/finalProject/samples', seconds_per_frame=1)
"""
get_images_from_live_video(
    '/Users/patrickspafford/Desktop/Machine Learning/finalProject/samples', 2)
