import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"

    border = extract_tableau(img_path)

    dessiner(border, "border")


def extract_tableau(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    threshed = binary_normal_or_inverse(img, 128)

    border = extract_border(img, threshed)

    return border


def binary_normal_or_inverse(img, seuil):
    """
    :param img: base gray img
    :return: binarized image by the color of the board
    """
    #find color
    board_color = get_board_color(img, seuil)

    #binarize
    if board_color == 0:
        ret2, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        #ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        #th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)
    else:
        ret2, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        #th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)
    return th



def get_board_color(img, seuil):
    """
    :param img: binary image
    :param seuil: the seuil (0-255) defining what is white or not white
    :return: 0 if colired board 1 if white board
    """
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

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

    #fill the holes in the mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
    mask = cv.morphologyEx(largest_connected, cv.MORPH_CLOSE, kernel)
    mask = (mask == 255)

    #extract the board on the original image
    cutted[~mask] = 0

    cutted = cutted[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]

    return cutted


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


if __name__ == '__main__':
    main()