from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import json
from construct_lines import construct_lines
import sys

np.set_printoptions(threshold=sys.maxsize)

from ens_figure import point_dans_figures
from test import create_polygon_mask


def dessinerbin(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=1, cmap="gray")
    plt.show()


def dessiner(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()


def somme_img(m1, m2):
    return m1 + m2



def get_terrain_figure(json_path, img_path):
    img = cv.imread(img_path)
    size = img.shape

    with open(json_path, 'r') as json_file: #ImagesProjetL3\\0.json "ImagesProjetL3\\0.jpg"
        json_load = json.load(json_file)

    data = json_load['shapes']
    figures = []
    for elem in data:
        figure = []
        for points in elem['points']:
            x = points[0]
            y = points[1]
            point = (x, y)
            figure.append(point)
        figures.append(figure)

        mask_terrain = []

        for figure in figures:
            mask_terrain.append(create_polygon_mask(figure, size[0], size[1]))

    base, lines, coords = construct_lines(img_path)

    mask_figures = []
    for coord in coords:
        mask_figures.append(create_polygon_mask(coord, size[0], size[1]))

    return mask_terrain, mask_figures




#True positif: there is a text, the prediction is text --> the number of values passing the threshold of iou
#False positif: there is not a text, the prediction is text --> the number of
#True Negatif: there is no text, the prediction no text
#false negatif: there is a text, the prediction no text (1 - tp)  tp + fn = 1


def true_positif_false_negatif(mask_terrain, mask_figures, seuil):
    print("Calcul de vrai positif et faux négatif")
    l_taux = [] #une liste de taux contenant iou

    for maskT in mask_terrain:
        max_taux = 0
        for maskf in mask_figures:
            #get logical and for the intersection and logical or for the union
            inter_mask = np.logical_and(maskT, maskf)*255
            union_mask = np.logical_or(maskT, maskf)*255

            count_inter = np.count_nonzero(inter_mask)
            count_union = np.count_nonzero(union_mask)

            #pass if there is no intersection
            if count_inter == 0:
                continue
            current_taux = count_inter/count_union



            """if count_union == 0:
                print(f"count inter : {count_inter}")
                print(f"count union : {count_union}")"""


            if current_taux > max_taux:
                max_taux = current_taux

        l_taux.append(max_taux)

    print(f"liste de taux {l_taux}")

    tp = 0
    for taux in l_taux:
        if taux > seuil:
            tp += 1
    tp = tp/len(l_taux)
    fn = 1 - tp

    return tp, fn



def faux_positif(mask_terrain, mask_figures, seuil):
    print("Calcul de faux positif")

    fp_taux = []

    for maskf in mask_figures:
        max_taux = 0
        for maskT in mask_terrain:
            # get logical and for the intersection and logical or for the union
            inter_mask = np.logical_and(maskT, maskf) * 255
            union_mask = np.logical_or(maskT, maskf) * 255

            count_inter = np.count_nonzero(inter_mask)
            count_union = np.count_nonzero(union_mask)

            # pass if there is no intersection
            if count_inter == 0:
                continue

            current_taux = count_inter / count_union

            if current_taux > max_taux:
                max_taux = current_taux

        fp_taux.append(max_taux)

    print(f"list of faux positif {fp_taux}")
    fp = 0
    for taux in fp_taux:
        if taux < seuil:
            fp += 1

    if len(fp_taux) == 0:
        return 0

    return fp / len(fp_taux)

#demo

mask_terrain, mask_figures = get_terrain_figure("ImagesProjetL3\\11.json", "ImagesProjetL3\\11.jpg")

tp, fn = true_positif_false_negatif(mask_terrain, mask_figures, 0.3)

fp = faux_positif(mask_terrain, mask_figures, 0.3)

print(f"vrai postif {tp}, faux negatif {fn}, faux positif {fp}")