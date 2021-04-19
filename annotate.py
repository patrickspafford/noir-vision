import numpy as np
import cv2
import argparse
from imutils.paths import list_images
import box_selector
#parse arguments
annotatePath = "testImg"
#annotations and image paths
annotations = []
imPaths = []
#loop through each image and collect annotations
for imagePath in list_images(annotatePath):
    #load image and create a BoxSelector instance
    cv2.destroyAllWindows()
    image = cv2.imread(imagePath)
    bs = box_selector.BoxSelector(image,imagePath)
    cv2.imshow(imagePath,image)
    cv2.waitKey(0)
    #order the points suitable for the Object detector
    pt1,pt2 = bs.roiPts
    (x,y,xb,yb) = [pt1[0],pt1[1],pt2[0],pt2[1]]
    annotations.append([int(x),int(y),int(xb),int(yb)])
    imPaths.append(imagePath)



#save annotations and image paths to disk
print(annotations)
annotations = np.array(annotations)
imPaths = np.array(imPaths,dtype="unicode")
np.save("annotationsNew",annotations)
np.save("imagesNew",imPaths)