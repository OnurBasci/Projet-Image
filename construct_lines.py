import numpy as np
import cv2 as cv
from RLSA import dessiner, apply_rlsa

MEDIAN_KERNEL_SIZE = 7


def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"
    construct_lines(img_path)



def construct_lines(img_path):
    rlsa = apply_rlsa(img_path)
    dessiner(rlsa,"rlsa")
    #apply median to remove lines
    med = cv.medianBlur(rlsa, ksize = MEDIAN_KERNEL_SIZE)

    dessiner(med, "median")

    words = get_lines(med)
    print(words)
def get_lines(rlsa_images):
    """
    :param rlsa_images: gets an image and extract all of the connected components
    :return: list of connected components
    """
    lines = []
    words = []
    line = []

    line_find = False

    filtered_img = rlsa_images.copy()

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filtered_img)

    for i in range(num_labels):
        #get box coordinates
        x1 = stats[i, cv.CC_STAT_LEFT]
        y1 = stats[i, cv.CC_STAT_TOP]
        x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]

        # get the component image
        word = (labels == i) * rlsa_images
        words.append([word,(x1,y1,x2,y2)])

        # dessiner(word, "word")
        if i == 0:
            line.append(word)
            line_find = True

        #get the length of the world
        last_word_lengt = words[i-1][3] - words[i-1][1]


        #if the distance between the word is les

        if line_find:
            line.append(word)


    return lines



if __name__ == '__main__':
    main()