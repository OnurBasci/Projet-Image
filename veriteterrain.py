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
mask_figures= []
"""
for figure in figures:
    mask_figures.append(create_polygon_mask(figure,size[0],size[1]))
    print(figure)
"""
i=0
for mask in mask_figures:
    i=i+1
    dessinerbin(mask,i)


base , lines ,coords = construct_lines(img_path)
print(coords)
for coord in coords:
    mask_figures.append(create_polygon_mask(coord,size[0],size[1]))

i=0
for mask in mask_figures:
    i=i+1
    dessinerbin(mask,i)