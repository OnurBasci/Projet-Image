from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import json
from construct_lines import construct_lines

from ens_figure import point_dans_figures
from test import create_polygon_mask
with open('ImagesProjetL3\\0.json', 'r') as json_file:
	json_load = json.load(json_file)
        
def dessinerbin(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=1, cmap="gray")
    plt.show()

def dessiner(img, name):
    plt.figure(name)
    plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    plt.show()


def somme_img(m1,m2):
    return m1 + m2
img_path= r"ImagesProjetL3\\0.jpg"
img = cv.imread(img_path)
size=img.shape

data = json_load['shapes']
figures =[]
for elem in data:
    figure = []
    for points in elem['points']:
        x= points[0]
        y= points[1]
        point = (x,y)
        figure.append(point)
    figures.append(figure)

mask_terrain= []


for figure in figures:
    mask_terrain.append(create_polygon_mask(figure,size[0],size[1]))

base , lines ,coords = construct_lines(img_path)

mask_figures= []
for coord in coords:
    mask_figures.append(create_polygon_mask(coord,size[0],size[1]))

for maskT in mask_terrain:
     taux= 0
     for mask in mask_figures:
        img_comp = maskT + mask
        union= 0
        inter= 0
        for x in img_comp:
            for pixel in x:
                if pixel == 2:
                    union += 1
                    inter += 1
                if pixel == 1:
                    union += 1
        taux_img = inter/union
        if taux_img > taux:
            taux = taux_img        
"""
i=0
for mask in mask_terrain:
    i=i+1
    dessinerbin(mask,i)

i=0
for mask in mask_figures:
    i=i+1
    dessinerbin(mask,i)
"""