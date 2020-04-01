# -*- coding: latin-1 -*-
"""
@brief
Module qui permet de générer une image pour un casque de réalité virtuelle à
partir d'une caméra stéréoscopique

Projet réalisé dans le cadre du projet de fin session du cours SYS809-vision
par ordinateur.
"""
import sys
import cv2

import numpy as np
from matplotlib import pyplot as plt

from calibration import StereoCalibration

data = "/Users/aubinheissler/Desktop/ETS/cours/SYS_809/Projet/data_calib"
img_right = "/Users/aubinheissler/Desktop/ETS/cours/SYS_809/Projet/Bouteille/Gauche/face.JPG"
img_left = "/Users/aubinheissler/Desktop/ETS/cours/SYS_809/Projet/Bouteille/Droite/face.JPG"

Left = cv2.imread(img_left)
Right = cv2.imread(img_right)
size = Left.shape
img = [Left, Right]

LeftGray = cv2.cvtColor(Left, cv2.COLOR_RGB2GRAY)
RightGray = cv2.cvtColor(Right, cv2.COLOR_RGB2GRAY)
imgGray = [LeftGray, RightGray]

# importe calibration data
calib = StereoCalibration(input_folder=data)

# Rectification
img_rectified = calib.rectify((img[0], img[1]))


def Display_compare(img_base, img_transform):
    """
    @brief  :  affiche un plot de quatre images
    @param  : img_base : liste de deux images, gauche, droite
              img_transform : liste de deux images modifiés, gauche droite
    @return : none
    """
    plt.figure()
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img_base[0], cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(img_base[1], cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(img_transform[0], cv2.COLOR_BGR2RGB))
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(img_transform[1], cv2.COLOR_BGR2RGB))
    plt.show()


def fusion_2_img(images, display=False):
    """
    @brief  : fusionne deux images en les ajoutants une à coté de l'autre
    @params : images  : liste de deux images, gauche, droite
              display : permet l'affichage de l'image obtenue
    @return : une image
    """
    size = images[0].shape
    Fusion = np.zeros((size[0], size[1]*2, 3), dtype=np.uint8)

    for i in range(size[0]):
        for j in range(size[1]):
            Fusion[i][j] = images[0][i][j]
            Fusion[i][j+size[1]] = images[1][i][j]

    if display is True:
        while(1):
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.imshow("test", Fusion)
            k = cv2.waitKey(27)
            if k == 27:
                break
    return Fusion


def findTransform(images):
    """
    @brief : trouve la transformation géometrique entre deux images en
    fonction des critères ECC
    @params : listes d'images, gauche droite
    @return : couple de matrice des transformations entre les deux images
    """
    number_of_iterations = 50
    termination_eps = 0.0001
    Motion = cv2.MOTION_AFFINE

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    cc, r = cv2.findTransformECC(imgGray[0], imgGray[1], warp, Motion, criteria, None, 1)

    l = np.array([[warp[0, 0], -warp[0, 1], -warp[0, 2]], [-warp[1, 0], warp[1, 1], -warp[1, 2]]])
    return l, r


def projection(images, warps):
    """
    @brief  : effectue une transformations affine sur une paire d'images
    @params : images : liste d'images gauche, droite
              warps  : liste de matrice 3x2 de transformation affine
    @return : liste d'images transformées
    """
    Left_warp = cv2.warpAffine(images[0], warps[0], (size[1], size[0]), flags=cv2.WARP_INVERSE_MAP)
    Right_warp = cv2.warpAffine(images[1], warps[1], (size[1], size[0]), flags=cv2.WARP_INVERSE_MAP)
    return Left_warp, Right_warp


warp = findTransform(img)
img_proj = projection(img, warp)
fusion = fusion_2_img(img_proj, True)
cv2.imwrite("test5.JPG", fusion)

Display_compare(img, img_proj)
