import json

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

    

