import cv2
import numpy as np
import os
from time import time
from vision import Vision
import PNGImages
import matplotlib.pyplot as plt

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread("PNGImages/FudanPed00001.png")
print(img)

print("Hi")

cv2.imshow('Me', img)
webcam = cv2.VideoCapture(0)
cv2.waitKey()

## to use video file as input
# webcam = cv2.VideoCapture('video.mp4')

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangle

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('Me', frame)

    key = cv2.waitKey(1)

    if key == 81 or key ==113:
        break

webcam.release()

print('\nFinished now\n')


