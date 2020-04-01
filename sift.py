import cv2
import numpy as np
from matplotlib import pyplot as plt

img_right = "/Users/aubinheissler/Desktop/ETS/cours/SYS_809/Projet/Image_calibration/Stereo/rapproche/Droite/cote.JPG"
img_left = "/Users/aubinheissler/Desktop/ETS/cours/SYS_809/Projet/Image_calibration/Stereo/rapproche/Gauche/cote.JPG"

Left = cv2.imread(img_left)
Right = cv2.imread(img_right)

Left_gray = cv2.cvtColor(Left, cv2.COLOR_BGR2GRAY)
Right_gray = cv2.cvtColor(Right, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# kp = sift.detect(Left_gray, None)
# img = cv2.drawKeypoints(Left_gray, kp, Left_gray)
# find the keypoints and descriptors with SIFT

kp1, des1 = sift.detectAndCompute(Left_gray, None)
kp2, des2 = sift.detectAndCompute(Right_gray, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(Left_gray, kp1, Right_gray, kp2, good, None, flags=2,)

plt.imshow(img3)
plt.show()
