import numpy as np
import cv2 as cv
import json

from ens_figure import point_dans_figures
with open('ImagesProjetL3\\0.json', 'r') as json_file:
	json_load = json.load(json_file)

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

img_path= r"ImagesProjetL3\\0.jpg"
img= cv.imread(img_path)

size= img.shape
res= []
for x in range (size[0]):
    for y in range (size[1]):
        if point_dans_figures((x,y),figures) :
             res.append((x,y))
