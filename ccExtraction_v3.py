import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#from ratio import get_size_ratio_2

np.set_printoptions(threshold=sys.maxsize)


ACCEPTANCE_SMALL_THRESH = 0.0001
ACCEPTANCE_BIG_THRESH = 0.05


def main():
    #final_path = r"C:\Users\onurb\pycharm_projects\Projet_UE_2023_Brouillon\filter\final_directory"
    #image_directory = r"C:\Users\onurb\pycharm_projects\Projet_UE_2023_Brouillon\filter\selFiles-1"


    #extractionComponentsDirectory(image_directory, final_path)
    #create a directory to put filtered images
    dir_path = r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\composants"
    create_directory(dir_path)

    img_path = r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\2.tif"

    extractConnectedComponents(img_path, dir_path)



def extractionComponentsDirectory(directory_path, final_path):
    """
    this function given from a directory path containing images, for each image, creates a directory containing
    the connected components of the image
    """
    for file in os.listdir(directory_path):
        print(file)
        #create directory
        image_dir = os.path.join(final_path, file[:-4])
        os.mkdir(image_dir)

        #put the connected components
        img_path = os.path.join(directory_path, file)
        extractConnectedComponents(img_path, image_dir)


# Cette fonction recupere la matrice des coordonnées des c.c selon l'image et le seuillage utilisé
def extractConnectedComponents(img_path, dir_path):
    """
    :param img_path:
    :param dir_path: Le répertoire à sauvegarder les images filtrées
    :return: void
    """

    image = cv.imread(img_path)  # image à importer
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # On passe l'image en nuances de gris
    #plt.imshow(img)
    #plt.show()

    # Test avec un seuillage Niblack
    # Blur
    img_blur = cv.GaussianBlur(img, (5, 5), 0)

    # Niblack + blur
    img_niblack_blur = cv.ximgproc.niBlackThreshold(_src=img_blur,
                                                    maxValue=255,
                                                    type=cv.THRESH_BINARY_INV,
                                                    blockSize=7,
                                                    k=0.1,
                                                    binarizationMethod=cv.ximgproc.BINARIZATION_NIBLACK)

    # connectivity = 8
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_niblack_blur)  # , connectivity)

    #plt.imshow(img_niblack_blur)
    #plt.title("Image binarisée")
    #plt.show()

    composant_index = 0

    for i in range(1, num_labels):
        # en haut à gauche
        x1 = stats[i, cv.CC_STAT_LEFT]
        y1 = stats[i, cv.CC_STAT_TOP]
        # en bas à droite
        x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]
        #print(x1, y1, x2, y2)

        #seperate the connected component
        cut_img = (labels == i).astype("uint8") * 255
        cut_img = cut_img[y1:y2, x1:x2]
        # plt.imshow(cut_img)
        # plt.show()

        #print(get_size_ratio_2(img_niblack_blur, cut_img))
        #if (ACCEPTANCE_BIG_THRESH > get_size_ratio_2(img_niblack_blur, cut_img) > ACCEPTANCE_SMALL_THRESH):
        print("hello")
        composant_index += 1
        #plt.imshow(cut_img)
        #plt.show()
        #save image
        cut_img_name = f"composant{composant_index}_{x1}_{y1}_{x2}_{y2}.png"
        cv.imwrite(os.path.join(dir_path, cut_img_name), cut_img)


def create_directory(path_name):
    # create directory if it doesn't exist already
    if not (os.path.isdir(path_name)):
        os.mkdir(path_name)



def dessiner(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()