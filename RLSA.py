import numpy as np
import cv2 as cv
from board_detection_v2 import extract_tableau, dessiner, binary_normal_or_inverse
from filter import filter_small_big_components
from rotation_outil import rotate_img

def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"

    rlsa_img, border = apply_rlsa(img_path)
    dessiner(rlsa_img, "rlsa")


def apply_rlsa(img_path):
    border_img, E = extract_tableau(img_path)
    print(f"E, {E}")
    dessiner(border_img, "border")

    #rotate the image
    border_img = rotate_img(border_img, E)
    dessiner(border_img, "rotated")


    # Apply a threshold to create a binary image
    canny = cv.Canny(border_img, 50, 200, None, 3)
    #dessiner(canny, "canny")


    ret, binary_image = cv.threshold(canny, 127, 255, cv.THRESH_BINARY)
    binary_image = filter_small_big_components(binary_image)

    #get the good threh for rlsa
    t_seuil = get_RLSA_thresh_value(binary_image)
    print(f"tseuil, {t_seuil}")

    # binary_image = binary_normal_or_inverse(border_img, 128)
    dessiner(binary_image, "binarie")

    # rlsa direct
    """dessiner(binary_image,"binary")
    # Apply RLSA to extract text lines
    rlsa_image = rlsa(binary_image, 10)

    dessiner(rlsa_image, "rlsa")"""

    # rlsa with filter
    #filtered = filter_small_big_components(binary_image)
    #dessiner(filtered,"filtre")

    rlsa_image = rlsa(binary_image, t_seuil)
    #dessiner(rlsa_image, "rlsa filtered")
    return rlsa_image, border_img


def rlsa(binary_image, t_seuil):
    mask = binary_image.copy()

    for i in range(binary_image.shape[0]):
        counter = 0
        white_found = False
        for j in range(binary_image.shape[1]):

            #check if a writing is found
            if binary_image[i,j] == 255:
                white_found = True
            if white_found:
                if binary_image[i,j] == 0:
                    counter += 1
                else:
                    mask[i, j - counter: j] = 255
                    counter = 0

                if counter > t_seuil:
                    counter = 0
                    white_found = False

    return mask


def get_RLSA_thresh_value(bin_img):
    """
    :param bin_img:
    :return: an integer defining the threshold valu
    """
    length_of_composants = []

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bin_img)

    for i in range(1, num_labels):
        #get box coordinates
        x1 = stats[i, cv.CC_STAT_LEFT]
        y1 = stats[i, cv.CC_STAT_TOP]
        x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]

        length = y2 - y1
        length_of_composants.append(length)



    #get the median of the list
    med = length_of_composants[len(length_of_composants)//2]

    print(length_of_composants)
    #a formule to define
    return int((3 * med - 2))



if __name__ == '__main__':
    main()