import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

def main():
	img = cv.imread(r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\median_images\m4.tif")
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#cv.imwrite(r"D:\Pycharm_project\Image_Tps\ImagesProjetL3\4_gray.png", gray)

	line_detection(gray, img)
	#calcul_bord(img)
	#line_detection(contour, img)
	#extraction_tableau(gray)
	#seuillage(gray, 100)


def extraction_tableau(img):
	"""
	:param img: Numpy array de 2 dimension
	:return: Np array represantant l'image avec le tableau extrait
	"""

	hist = cv.calcHist([img],[0],None,[256],[0,256])
	#hist2 = get_histogram(img)

	a,b = get_intervalle_tableau(hist)

	print(a,b)

	afficher_hist(hist)


	img = extraction(img, a, b)
	print(img)

	#g = np.array([[1,2],[3,4],[5,6]])
	#print((2<((g<5) * g))*g)


	dessiner(img)

def seuillage(img, seuil):
	img_seuil = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] > seuil:
				img_seuil[i,j] = 255
			else:
				img_seuil[i, j] = 0
	return img_seuil

def get_histogram(img):
	hist = np.zeros(255)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			hist[img[i,j]-1] += 1

	return hist

def get_intervalle_tableau(hist):
	moyenne = sum(hist)
	moyenne/= 256
	#moyenne = max(hist)/3
	print(moyenne)
	pique = max(hist)
	m_index = np.where(hist == pique)[0][0]

	#trouver l'intevalle
	b = m_index+1
	a = m_index-1
	a_Found = False
	b_Found = False
	while not(a_Found or b_Found):
		if hist[a] < moyenne:
			a_Found = True
		else:
			if a > 0:
				a -= 1
		if hist[b] < moyenne:
			b_Found = True
		else:
			if b < 255:
				b += 1

	return a, b

def extraction(img, a, b):
	im = (a < ((img < b) * img)) * img

	#im = img.copy()

	"""for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] > a and img[i,j] < b:
				img[i,j] = 255
			else:
				img[i,j] = 0"""

	"""im[img >= b] = 0
	im[img <= a] = 0
	im[((img > a)*img) < b] = 255"""

	#im = (im  b) * im
	return im

def dessiner(img):
	plt.figure()
	plt.imshow(img, vmin=0, vmax=255)
	plt.show()

def afficher_hist(hist):
	x_axe = [x for x in range(256)]
	plt.figure()
	plt.plot(x_axe, hist)
	plt.show()


def calcul_bord(img):
	ddepth = cv.CV_16S
	blur = cv.GaussianBlur(img, (3, 3), 0)
	gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

	dst = cv.Laplacian(gray, ddepth, ksize=3)
	abs_dst = cv.convertScaleAbs(dst)

	dessiner(abs_dst)
	return abs_dst

def line_detection(gray, img_base):
	dst = cv.Canny(gray, 50, 200, None, 3)

	# Copy edges to the images that will display the results in BGR
	cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
	cdstP = np.copy(cdst)
	cdstGray = np.copy(cdst)

	cdst_edges = np.copy(cdst)

	lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
			pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
			cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

	linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

	#min_max(linesP)


	#afficher lignes
	if linesP is not None:
		for i in range(0, len(linesP)):
			l = linesP[i][0]
			cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

	#afficher les lignes gris
	gray_lines = extract_gray_lines(linesP, img_base)

	#print(gray_lines[0][0])
	if gray_lines is not None:
		for i in range(0, len(gray_lines)):
			l = gray_lines[i][0]
			cv.line(cdstGray, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

	dessiner(cdstP)
	dessiner(cdstGray)
	#dessiner(cdst_edges)

	cv.waitKey()

def extract_gray_lines(linesP, img_base):
	gray_line_coordinates = []
	#print(linesP)
	for pixel_cordinates in linesP:
		first_point = (pixel_cordinates[0][0], pixel_cordinates[0][1])
		second_point = (pixel_cordinates[0][2], pixel_cordinates[0][3])
		print(img_base[first_point[1], first_point[0]])
		if is_close(img_base[first_point[1], first_point[0]]) and is_close(img_base[second_point[1], second_point[0]]):
			print(img_base[first_point[1], first_point[0]])
			gray_line_coordinates.append(pixel_cordinates)

	return gray_line_coordinates


def is_close(pixel_color):
	seuil = 50
	gray_color = (90,110,89)
	#distance euclidan
	dist = math.sqrt(math.pow(abs(pixel_color[0] - gray_color[0]), 2) + math.pow(abs(pixel_color[1] - gray_color[1]),2) + math.pow(abs(pixel_color[2] - gray_color[2]),2))

	if dist < seuil:
		return True
	else:
		return False


def min_max(linesP, cdst_edges):
	x1_values = [linesP[i][0][0] for i in range(len(linesP))]
	# find min x value
	min_x = min(x1_values)
	index = x1_values.index(min_x)
	min_x_p1 = (min(x1_values), linesP[index][0][1])
	min_x_p2 = (linesP[index][0][2], linesP[index][0][3])

	# find max x value
	max_x = max(x1_values)
	index = x1_values.index(max_x)
	max_x_p1 = (max(x1_values), linesP[index][0][1])
	max_x_p2 = (linesP[index][0][2], linesP[index][0][3])

	y1_values = [linesP[i][0][1] for i in range(len(linesP))]

	# find min y_values
	min_y = min(y1_values)
	index = y1_values.index(min_y)
	min_y_p1 = (linesP[index][0][0], min(y1_values))
	min_y_p2 = (linesP[index][0][2], linesP[index][0][3])

	# find max_y values
	max_y = max(y1_values)
	index = y1_values.index(max_y)
	max_y_p1 = (linesP[index][0][0], max(y1_values))
	max_y_p2 = (linesP[index][0][2], linesP[index][0][3])

	cv.line(cdst_edges, min_x_p1, min_x_p2, (0, 0, 255), 3, cv.LINE_AA)
	cv.line(cdst_edges, max_x_p1, max_x_p2, (0, 0, 255), 3, cv.LINE_AA)
	cv.line(cdst_edges, min_y_p1, min_y_p2, (0, 0, 255), 3, cv.LINE_AA)
	cv.line(cdst_edges, max_y_p1, max_y_p2, (0, 0, 255), 3, cv.LINE_AA)

if __name__ == '__main__':
	main()