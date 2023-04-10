import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from rotation_outil import get_ellipse
import math

MEDIAN_KERNEL_SIZE = 21

#the optimal thresh by volume seems like 133

def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\training_data\14.jpeg"

    border, E = extract_tableau(img_path)

    dessiner(border, "border")


    #test(r"C:\Users\onurb\PycharmProjects\Projet-Image\training_data")


def extract_tableau(img_path):
    img = cv.imread(img_path)

    # Apply inpainting to remove the reflection
    #mask = cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1]
    #img = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

    threshed = binary_normal_or_inverse(img, 133)

    dessiner(threshed, "threshed")

    border, E = extract_border(img, threshed)

    return border, E


def binary_normal_or_inverse(img, seuil):
    """
    :param img: base gray img
    :return: binarized image by the color of the board
    """
    #find color
    board_color = get_board_color(img, seuil)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #binarize
    """if board_color == 1:
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        ret2, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        #ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        #gray = cv.medianBlur(gray, ksize=MEDIAN_KERNEL_SIZE)
        th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 7, 0)
        th = cv.ximgproc.niBlackThreshold(_src=gray,maxValue=255,type=cv.THRESH_BINARY_INV,blockSize=7,k=0.1,binarizationMethod=cv.ximgproc.BINARIZATION_NIBLACK)
    else:
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        ret2, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        #gray = cv.medianBlur(gray, ksize=MEDIAN_KERNEL_SIZE)
        th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 0)
        th = cv.ximgproc.niBlackThreshold(_src=gray, maxValue=255,type=cv.THRESH_BINARY,blockSize=7,k=0.1,binarizationMethod=cv.ximgproc.BINARIZATION_NIBLACK)"""

    #gray = cv.GaussianBlur(gray, (5, 5), 0)
    th = cv.ximgproc.niBlackThreshold(_src=gray, maxValue=255, type=cv.THRESH_BINARY_INV, blockSize=27, k=0,
                                      binarizationMethod=cv.ximgproc.BINARIZATION_NIBLACK)
    return th



def get_board_color(img, seuil):
    """
    :param img: binary image
    :param seuil: the seuil (0-255) defining what is white or not white
    :return: 0 if colired board 1 if white board
    """
    #transform to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cutted = img[hsv.shape[0] // 3: -hsv.shape[0] // 3, hsv.shape[1] // 3: -hsv.shape[1] // 3]


    h, s, v = cv.split(cutted)

    hist, bins = np.histogram(v.ravel(), 256, [0, 256])

    #hist = cv.calcHist([img], [0], None, [256], [0, 256])

    afficher_hist(hist)

    #get the max point
    m_index = np.argmax(hist)
    print(m_index)

    #afficher_hist(hist)

    if m_index > seuil:
        return 1
    else:
        return 0


def extract_border(base_img, threshed):
    cutted = base_img.copy()

    largest_connected, coordinates = cut_largest_connected_component(threshed)
    dessiner(largest_connected, "largest")
    #rotate the image
    E = get_ellipse(largest_connected)


    #fill the holes in the mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
    mask = cv.morphologyEx(largest_connected, cv.MORPH_CLOSE, kernel)
    mask = (mask == 255)

    #extract the board on the original image
    cutted[~mask] = 0

    cutted = cutted[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]

    return cutted, E


def cut_largest_connected_component(img):
    """
    :param img:
    :return: Largest connected components of the binary image and it's invere
    """
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img)

    max_label = 1
    max_size = stats[1, cv.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv.CC_STAT_AREA] > max_size:
            max_label = i
            max_size = stats[i, cv.CC_STAT_AREA]


    x1 = stats[max_label, cv.CC_STAT_LEFT]
    y1 = stats[max_label, cv.CC_STAT_TOP]
    # en bas à droite
    x2 = x1 + stats[max_label, cv.CC_STAT_WIDTH]
    y2 = y1 + stats[max_label, cv.CC_STAT_HEIGHT]



    cutted_mask = (labels == max_label).astype("uint8") * 255
    #cut_img = cutted_mask[y1:y2, x1:x2]


    #find the contour
    # Find the contours in the binary image
    contours, hierarchy = cv.findContours(cutted_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #fill the largest largest component
    cutted_mask = cv.drawContours(cutted_mask, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

    return cutted_mask, (x1,y1,x2,y2)

def dessiner(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()

def afficher_hist(hist):
    x_axe = [x for x in range(256)]
    plt.figure()
    plt.plot(x_axe, hist)
    plt.show()


#to delete

def mean_index(directory):
    for i, file in enumerate(os.listdir(directory)):
        if i == 0:
            continue
        img = cv.imread(os.path.join(directory, file), cv.COLOR_BGR2HSV)
        img = img[img.shape[0] // 3: -img.shape[0] // 3, img.shape[1] // 3: -img.shape[1] // 3]

        print(img.shape)

        h, s, v = cv.split(img)
        dessiner(img, "img")
        #print(img)

        get_mx_index(v)



def get_mx_index(v):

    hist, bins = np.histogram(v.ravel(), 256, [0, 256])
    # Plot the histogram
    """plt.plot(hist)
    plt.xlim([0, 256])
    plt.xlabel('Lightness Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Lightness Value')
    plt.show()"""

    #afficher_hist(hist)

    #get the max point
    m_index = np.argmax(hist)
    print(m_index)


def test(directory):

    for i, file in enumerate(os.listdir(directory)):
        if i == 0:
            continue

        print(file)
        img_path = os.path.join(directory, file)

        border, E = extract_tableau(img_path)

        dessiner(border, "border")


if __name__ == '__main__':
    main()
    #mean_index(r"C:\Users\onurb\PycharmProjects\Projet-Image\training_data")