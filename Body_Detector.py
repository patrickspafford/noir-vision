import cv2
import numpy as np
import os
from time import time
from PIL import Image
import matplotlib.pyplot as plt

def draw_rect(img, rect):
    line_color = (0, 255, 0)
    line_type = cv2.LINE_4
    for (x, y, w, h) in rect:
        topL = (x, y)
        bottomR = (x + w, y + h)
        cv2.rectangle(img, topL, bottomR, line_color, 2)
        return img
cascade_body = cv2.CascadeClassifier('cascade12/cascade.xml')

img = cv2.imread("posImg/FudanPed00008.png")
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rectangles = cascade_body.detectMultiScale(g_img)

new_img = draw_rect(img, rectangles)
cv2.imshow('trained', new_img)

#img = cv2.imread("pos2/FudanPed00001.png")
#print(img)

print("Hi")

#cv2.imshow('Me', img)
webcam = cv2.VideoCapture(0)

cv2.waitKey(0)

## to use video file as input



while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    body_coordinates = cascade_body.detectMultiScale(grayscaled_img)

    #draw rectangle

    for (x, y, w, h) in body_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('Me', frame)

    key = cv2.waitKey(1)

    if key == 81 or key ==113:
        break

webcam.release()

print('\nFinished now\n')


