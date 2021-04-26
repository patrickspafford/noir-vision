import joblib
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import svm
from collections import Counter
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
import pickle
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
import cv2
import imutils
import numpy as np
import cv2
import time
from imutils.object_detection import non_max_suppression as nms
import cv2 as cv
from os import makedirs
from os.path import join, exists



def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}x{height}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)

        joblib.dump(data, pklname)


# modify to fit your system
data_path = '/Users/jamescourson/Desktop/noir-vision/svm/dataset'
# os.listdir(data_path)

base_name = 'human_detector'
width = 150
height = 75

include = {'person', 'notPerson'}

resize_all(src=data_path, pklname=base_name, width=width, height=height, include=include)


data = joblib.load(f'{base_name}_{width}x{height}px.pkl')

# print('number of samples: ', len(data['data']))
# print('keys: ', list(data.keys()))
# print('description: ', data['description'])
# print('image shape: ', data['data'][0].shape)
# print('labels:', np.unique(data['label']))

# Counter(data['label'])

X = np.array(data['data'])
y = np.array(data['label'])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=True,
    random_state=42,
)


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

# call fit_transform on each transform converting X_train step by step

X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)


X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

y_pred = sgd_clf.predict(X_test_prepared)
print(np.array(y_pred == y_test)[:25])
print('')
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))


# This is the pipline to create the classifier object names "CLF"

HOG_pipeline = Pipeline([
    ('grayify', RGB2GrayTransformer()),
    ('hogify', HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys')
    ),
    ('scalify', StandardScaler()),
    ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
])

clf = HOG_pipeline.fit(X_train, y_train)


""" RUN if you want to optimize the SVM hyper parameters, must use "grid_res" for classifier object  """
# param_grid = [
#     {
#         'hogify__orientations': [8, 9],
#         'hogify__cells_per_block': [(2, 2), (3, 3)],
#         'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
#     },
#     {
#         'hogify__orientations': [8],
#          'hogify__cells_per_block': [(3, 3)],
#          'hogify__pixels_per_cell': [(8, 8)],
#          'classify': [
#              SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
#              svm.SVC(kernel='linear')
#          ]
#     }
# ]
#
# grid_search = GridSearchCV(HOG_pipeline,
#                            param_grid,
#                            cv=3,
#                            n_jobs=-1,
#                            scoring='accuracy',
#                            verbose=1,
#                            return_train_score=True)
#
# grid_res = grid_search.fit(X_train, y_train)






def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



# loop over the image pyramid
def searchImage(image):
    (winW, winH) = (int(128/2), 128)
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)


# image input is a frame that will be fed to the classifier to determine whether it is a person or not
# Returns coordiantes for a rectangle
def searchImage(image):
    (winW, winH) = (int((128/2)), int(128))
    i=0
    j=0
    corVec = []
    for resized in pyramid(image, scale=2):
        i+=1
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            j+=1
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            path = '/Users/jamescourson/Desktop/noir-vision/svm/positivePred'
            pathneg = '/Users/jamescourson/Desktop/noir-vision/svm/negPred'
#             while(True):
#                 cv2.imshow('img1',image[y:y + winH, x:x + winW]) #display the captured image
#                 if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
#                     cv2.destroyAllWindows()
#                     break

#             cv2.imwrite(os.path.join(path , f'{i}_{j}.jpg'), image[y:y + winH, x:x + winW])
            img = image[y:y + winH, x:x + winW]
            img = cv2.resize(img, (75,150))
#             print("a")
            img = np.array([img])

            X_test_gray = grayify.transform(img)

            X_test_hog = hogify.transform(X_test_gray)

            X_test_prepared = scalify.transform(X_test_hog)
            guess = sgd_clf.predict(X_test_prepared)
            # print(guess[0])
            if guess[0] == 'pe':
#                 while(True):
#                     cv2.imshow('img1',img[0]) #display the captured image
#                     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
#                         cv2.imwrite(os.path.join(path , f'{i}_{j}.jpg'), img[0])
#                         cv2.destroyAllWindows()
#                         break
                # cv2.imwrite(os.path.join(path , f'{i}_{j}.jpg'), img[0])
                corVec.append((x, y, x + winW, y + winH))

            else:
                # cv2.imwrite(os.path.join(pathneg , f'{i}_{j}.jpg'), img[0])
                pass

    return corVec




"""Feed in image path to path variable and will display rectangles over the areas the classifier thinks is a human"""
path = '/Users/jamescourson/Desktop/noir-vision/svm/desertman.jpeg'
image = cv2.imread(path)

print("one frame")
coor = searchImage(image)
#print(coor)

# Window name in which image is displayed
window_name = 'Image'
coor = np.array(coor)
print(coor)
merged_rects = nms(coor, probs=None, overlapThresh=0.2)
#print(merged_rects)


for i in range(len(coor)):
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (coor[i][0], coor[i][1])

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (coor[i][2], coor[i][3])

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
        cv2.destroyAllWindows()
        break






""" This section is for a live video and will save the video to the path found in the result variable below"""
# source_path = '/Users/jamescourson/Desktop/noir-vision/svm/man_walking.mov'
# video = cv2.VideoCapture(source_path)
#
#
# # We need to check if camera
# # is opened previously or not
# if (video.isOpened() == False):
#     print("Error reading video file")
#
# # We need to set resolutions.
# # so, convert them from float to integer.
# frame_width = int(video.get(3))
# frame_height = int(video.get(4))
#
# size = (frame_width, frame_height)
#
# # Below VideoWriter object will create
# # a frame of above defined The output
# # is stored in 'filename.avi' file.
# result = cv2.VideoWriter('/Users/jamescourson/Desktop/noir-vision/svm/desert.mp4',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)
#
#
#
# while(True):
#     ret, frame = video.read()
#
#     if ret == True:
#
#         coor = searchImage(frame)
#         # Window name in which image is displayed
#         window_name = 'Image'
#         coor = np.array(coor)
#
#         merged_rects = nms(coor, probs=None, overlapThresh=0.3)
#         merged_rects = nms(merged_rects, probs=None, overlapThresh=0.1)
#         # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#         # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#
#         for i in range(len(coor)):
#             # Start coordinate, here (5, 5)
#             # represents the top left corner of rectangle
#             start_point = (coor[i][0], coor[i][1])
#
#             # Ending coordinate, here (220, 220)
#             # represents the bottom right corner of rectangle
#             end_point = (coor[i][2], coor[i][3])
#
#             # Blue color in BGR
#             color = (255, 0, 0)
#
#             # Line thickness of 2 px
#             thickness = 2
#
#             # Using cv2.rectangle() method
#             # Draw a rectangle with blue line borders of thickness of 2 px
#             frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
#
#         # Write the frame into the
#         # file 'filename.avi'
#
#         result.write(frame)
#
#
#         # # Display the frame
#         # # saved in the file
#         cv2.imshow('Frame', frame)
#
#         # Press S on keyboard
#         # to stop the process
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # When everything done, release
# # the video capture and video
# # write objects
# video.release()
# result.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()
#
# print("The video was successfully saved")
#








# def get_images_from_live_video(output_path, seconds_per_frame=1):
#     cam = cv2.VideoCapture(0)
#     try:
#         if not exists(output_path):
#             makedirs(output_path)
#     except OSError:
#         print('Error when creating directory of extracted frames.')
#     count = 0
#     while(True):
#         _, frame = cam.read()
#         if count % (30 * seconds_per_frame) == 0:
#
#
#             coor = searchImage(frame)
#             # Window name in which image is displayed
#             window_name = 'Image'
#             coor = np.array(coor)
#
#             merged_rects = nms(coor, probs=None, overlapThresh=0.7)
#             # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#             # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#
#             for i in range(len(coor)):
#                 # Start coordinate, here (5, 5)
#                 # represents the top left corner of rectangle
#                 start_point = (coor[i][0], coor[i][1])
#
#                 # Ending coordinate, here (220, 220)
#                 # represents the bottom right corner of rectangle
#                 end_point = (coor[i][2], coor[i][3])
#
#                 # Blue color in BGR
#                 color = (255, 0, 0)
#
#                 # Line thickness of 2 px
#                 thickness = 2
#
#                 # Using cv2.rectangle() method
#                 # Draw a rectangle with blue line borders of thickness of 2 px
#                 frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
#                 cv2.imshow("h", frame)
#
#
#         count += 1
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     cam.release()
#     cv.destroyAllWindows()




# get_images_from_live_video('/Users/jamescourson/Desktop/noir-vision/svm/Video_out', 1)
#
# def extract_images_from_video(source_path, seconds_per_frame=1):
#     cam = cv2.VideoCapture(source_path)
#
#     count = 0
#     while(True):
#         _, frame = cam.read()
#         if count % (30 * seconds_per_frame) == 0:
#
#
#             coor = searchImage(frame)
#             # Window name in which image is displayed
#             window_name = 'Image'
#             coor = np.array(coor)
#
#             merged_rects = nms(coor, probs=None, overlapThresh=0.3)
#             merged_rects = nms(merged_rects, probs=None, overlapThresh=0.1)
#             # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#             # merged_rects = nms(merged_rects, probs=None, overlapThresh=0.01)
#
#             for i in range(len(coor)):
#                 # Start coordinate, here (5, 5)
#                 # represents the top left corner of rectangle
#                 start_point = (coor[i][0], coor[i][1])
#
#                 # Ending coordinate, here (220, 220)
#                 # represents the bottom right corner of rectangle
#                 end_point = (coor[i][2], coor[i][3])
#
#                 # Blue color in BGR
#                 color = (255, 0, 0)
#
#                 # Line thickness of 2 px
#                 thickness = 2
#
#                 # Using cv2.rectangle() method
#                 # Draw a rectangle with blue line borders of thickness of 2 px
#                 frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
#                 cv2.imshow("h", frame)
#
#
#         count += 1
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     cam.release()
#     cv.destroyAllWindows()
#
# # get_images_from_live_video('/Users/jamescourson/Desktop/noir-vision/svm/Video_out', 1)
# extract_images_from_video('/Users/jamescourson/Desktop/noir-vision/svm/Screen Recording 2021-04-19 at 10.26.50 AM.mov',.1)
#
#
