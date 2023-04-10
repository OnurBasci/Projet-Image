import os

import numpy as np
import math
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
#from scipy.ndimage import rotate


#source: http://raphael.candelier.fr/?blog=Image%20Moments

def main():
    np.set_printoptions(threshold=sys.maxsize)


    img_path = r"C:\Users\onurb\pycharm_projects\Projet_UE_2023_Brouillon\classification\modele\modele1.png"
    img = cv.imread(img_path)

    final = rotation_scale_adjustments(img)
    #save
    #cv.imwrite(r"C:\Users\onurb\pycharm_projects\Projet_UE_2023_Brouillon\classification\rotated_components\composant28177_485_1285_857_1309_867.png", final)


def rotation_scale_adjustments(img):
    """
    This function takes an image and rotates and adjust the images to make it rotation invariant for the GFD comparison
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    E = get_ellipse(gray)
    print(E)
    draw_major_axis(img, E)
    dessiner(img, "original")

    # rotate image
    # rotated = rotate(img, math.degrees(math.pi/2 - E['theta']))
    rotated = rotate_img(gray, E)
    dessiner(rotated, "rotated")



    # apply threshold for gray level (low level =10)
    ret, thresh_rot = cv.threshold(rotated, 10, 255, cv.THRESH_BINARY)

    # dessiner(thresh_rot, "thresh")
    cutted = cut_largest_connected_component(thresh_rot)
    dessiner(cutted, "cutted")

    # resize
    resized = cv.resize(cutted, (75, 39))
    dessiner(resized, "resized")

    return resized


def get_ellipse(img, direct=True):
    # Default values
    if not direct:
        direct = True

    # Computing moments
    # Object's pixels coordinates
    I = np.argwhere(img)
    i, j = I[:, 0], I[:, 1]

    # Function handle to compute the moments
    moment = lambda p, q: np.sum((i ** p) * (j ** q) * img[i, j])

    # Prepare the output
    E = {}

    # Get the Moments
    E['m00'] = moment(0, 0)
    E['m10'] = moment(1, 0)
    E['m01'] = moment(0, 1)
    E['m11'] = moment(1, 1)
    E['m02'] = moment(0, 2)
    E['m20'] = moment(2, 0)

    # Ellipse properties
    # Barycenter
    E['x'] = E['m10'] / E['m00']
    E['y'] = E['m01'] / E['m00']

    # Central moments (intermediary step)
    a = E['m20'] / E['m00'] - E['x'] ** 2
    b = 2 * (E['m11'] / E['m00'] - E['x'] * E['y'])
    c = E['m02'] / E['m00'] - E['y'] ** 2

    # Orientation (radians)
    E['theta'] = 1 / 2 * np.arctan(b / (a - c)) + (a < c) * np.pi / 2

    # Minor and major axis
    E['w'] = np.sqrt(8 * (a + c - np.sqrt(b ** 2 + (a - c) ** 2))) / 2
    E['l'] = np.sqrt(8 * (a + c + np.sqrt(b ** 2 + (a - c) ** 2))) / 2

    return E


def draw_major_axis(img, E, color=(0, 255, 0), thickness=2):
    # Calculate the endpoints of the major axis
    x1 = int(E['x'] - E['w'] / 2 * math.cos(E['theta']))
    y1 = int(E['y'] - E['w'] / 2 * math.sin(E['theta']))
    x2 = int(E['x'] + E['w'] / 2 * math.cos(E['theta']))
    y2 = int(E['y'] + E['w'] / 2 * math.sin(E['theta']))

    # Draw the line on the image
    cv.line(img, (y1, x1), (y2, x2), color, thickness)
    #cv.line(img, (32, 12), (41, 12), color, thickness)
    print((x1,y1))
    print((x2, y2))
    print(img.shape)


def rotate_img(img, E):
    #check the rotation
    rotated = rotate(img, math.degrees(math.pi / 2 - E['theta']))
    """if E['theta'] <= 0:
        rotated = rotate(img, math.degrees(math.pi / 2 - E['theta']))
    else:
        rotated = rotate(img, math.degrees(-(math.pi/2 + E['theta'])))"""

    return rotated



def get_index_matrix(img):
    indices = np.where(img == 255)
    coords = zip(indices[0], indices[1])
    return coords



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

#from rotated models we get 39 and 75 as avarage size
def find_mean_size(directory):
    counter = 0
    mean_x = 0
    mean_y = 0
    for path in os.listdir(directory):
        img = cv.imread(os.path.join(directory, path))
        mean_x += img.shape[0]
        mean_y += img.shape[1]
        counter+=1
        print(img.shape)

    mean_x /= counter
    mean_y /= counter

    return (mean_x, mean_y)

#from rotated models we get 39 and 75 as avarage size
def resize_model_images(directory):

    for path in os.listdir(directory):
        img = cv.imread(os.path.join(directory, path))

        resized = cv.resize(img, (75,39))
        dessiner(resized, "resized")



def dessiner(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()

if __name__ == '__main__':
    #resize_model_images(r"C:\Users\onurb\pycharm_projects\Projet_UE_2023_Brouillon\classification\rotated_cutted_models")
    main()