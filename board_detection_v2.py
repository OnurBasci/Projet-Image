import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math



def main():
    img_path = r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\ImagesProjetL3\0.jpg"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    dessiner(img, 'image')

    threshed = binarisation(img)

    dessiner(threshed, 'thresh')

    largest_connected = cut_largest_connected_component(threshed)

    dessiner(largest_connected, "connected")

def binarisation(img):
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return th2


def get_board_color(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])


def cut_largest_connected_component(img):
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

    cut_img = (labels == max_label).astype("uint8") * 255
    cut_img = cut_img[y1:y2, x1:x2]

    return cut_img

def dessiner(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()