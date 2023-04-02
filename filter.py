import numpy as np
import cv2 as cv
from board_detection_v2 import extract_tableau, dessiner, binary_normal_or_inverse

MAX_LENGTH_RATIO = 0.1
MIN_LENGTH_RATIO = 0.03

def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"
    border_img = extract_tableau(img_path)

    # Apply a threshold to create a binary image
    ret, binary_image = cv.threshold(border_img, 127, 255, cv.THRESH_BINARY)

    dessiner(binary_image, "binary")

    filtered = filter_small_big_components(binary_image)

    dessiner(filtered, "filtered")




def filter_small_big_components(bin_img):
    filtered_img = bin_img.copy()

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filtered_img)

    for i in range(2, num_labels):
        #get the ratio of the component
        y1 = stats[i, cv.CC_STAT_TOP]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]

        component_length = y2 - y1

        img_length = bin_img.shape[0]

        ratio = component_length/img_length

        #put 0 to big or small components
        if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
            filtered_img[(labels == i)] = 0

    return filtered_img




def length_ratio(base_img, connected_component):
    return base_img.shape[0]/connected_component.shape[0]

if __name__ == '__main__':
    main()