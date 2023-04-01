import numpy as np
import cv2 as cv
from board_detection_v2 import extract_tableau, dessiner, binary_normal_or_inverse

def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"
    border_img = extract_tableau(img_path)
    dessiner(border_img, "border")

    # Apply a threshold to create a binary image
    ret, binary_image = cv.threshold(border_img, 127, 255, cv.THRESH_BINARY)
    #binary_image = binary_normal_or_inverse(border_img, 128)
    dessiner(binary_image,"binary")
    # Apply RLSA to extract text lines
    rlsa_image = rlsa(binary_image, 10)

    dessiner(rlsa_image, "rlsa")

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


if __name__ == '__main__':
    main()