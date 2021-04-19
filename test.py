from obj_detector import ObjectDetector
import numpy as np
import cv2
import argparse
from imutils.paths import list_images
import dlib

detector = ObjectDetector(loadPath= 'testSmallTorso')
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
win = dlib.image_window()
win.set_image(detector._detector)
dlib.hit_enter_to_continue()
for imagePath in list_images("posImg/"):
    image = cv2.imread(imagePath)
    detector.detect(image, "test")
