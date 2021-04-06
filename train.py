from obj_detector import ObjectDetector
import numpy as np
import cv2
import argparse

annFile = "annotationsNew.npy"
imFile = "imagesNew.npy"

print ("[INFO] loading annotations and images")
annots = np.load("annotationsNew.npy")
imagePaths = np.load("imagesNew.npy")

detector = ObjectDetector()
print("[INFO] creating & saving object detector")

detector.fit(imagePaths,annots,visualize=True, savePath="test")

imagePath = "posImg/FudanPed00020.png"
image = cv2.imread(imagePath)

detector.detect(image, "test.svm")