from imutils.object_detection import non_max_suppression as nms
from imutils.paths import list_images
from numpy import array
from argparse import ArgumentParser
from imutils import resize
from os.path import join, exists
from os import mkdir
from cv2 import HOGDescriptor, HOGDescriptor_getDefaultPeopleDetector, imwrite, rectangle, imread

"""
ap = ArgumentParser()
ap.add_argument('-i', '--images', required=True,
                help="The path to the images/frames directory")
ap.add_argument('-o', '--output', required=True,
                help="The path to the folder for resulting images with rectangles added")
args = vars(ap.parse_args())
hog = HOGDescriptor()
hog.setSVMDetector(HOGDescriptor_getDefaultPeopleDetector())
if not exists(args['output']):
    mkdir(args['output'])
for imagePath in list_images(args['images']):
    image = imread(imagePath)
    image = resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
    rects = array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    merged_rects = nms(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in merged_rects:
        rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    filename = imagePath[imagePath.rfind('/') + 1:]
    imwrite(join(args['output'], filename), image)
"""


def run_detection_baseline(inputFolder, outputFolder):
    hog = HOGDescriptor()
    hog.setSVMDetector(HOGDescriptor_getDefaultPeopleDetector())
    if not exists(outputFolder):
        mkdir(outputFolder)
    for imagePath in list_images(inputFolder):
        image = imread(imagePath)
        image = resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        (rects, _) = hog.detectMultiScale(
            image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
        rects = array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
        merged_rects = nms(rects, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in merged_rects:
            rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        filename = imagePath[imagePath.rfind('/') + 1:]
        imwrite(join(outputFolder, filename), image)


def find_people_in_frame(inputFrame, detector=HOGDescriptor_getDefaultPeopleDetector):
    hog = HOGDescriptor()
    hog.setSVMDetector(detector())
    image = resize(inputFrame, width=min(400, inputFrame.shape[1]))
    orig = image.copy()
    (rects, _) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
    rects = array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    merged_rects = nms(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in merged_rects:
        rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return image
