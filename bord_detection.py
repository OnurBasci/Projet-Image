import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os

#premièrement dillatation
#

#MEDIAN_KERNEL_SIZE_COEFFICIENT = 50
KERNEL_SIZE = 31
ACCEPTABLE_ANGLE = 0.3



def main():
    img = cv.imread(r"C:\\Users\\onurb\\PycharmProjects\\Projet-Image\\training_data\\80.jpg")
    #img = cv.imread(r"C:\\Users\\onurb\\pycharm_projects\\Image_Tps\\ImagesProjetL3\\ImagesProjetL3\\13.jpg")
    #dessiner(img, "image")
    print(img.shape[0])
    med = median(img)
    gray = cv.cvtColor(med, cv.COLOR_BGR2GRAY)
    line_detection(gray,img)

    #dir = r"C:\Users\onurb\PycharmProjects\Projet-Image\training_data"
    #test(dir)


def extract_tableau(img_path):
    img = cv.imread(img_path)
    med = median(img)
    gray = cv.cvtColor(med, cv.COLOR_BGR2GRAY)
    tableau, board_limit = line_detection(gray, img)
    return tableau, board_limit

def median(img):
    """
    :param img: image de base en 1 ou 3 channels
    :return: image avec le filtre median appliqué
    """
    #kernel_size = get_kernel_size(img, MEDIAN_KERNEL_SIZE_COEFFICIENT)
    #print(kernel_size)
    med = cv.medianBlur(img, ksize = KERNEL_SIZE)
    #dessiner(med, "med")
    return med

def get_kernel_size(img, size_coef):
    if (img.shape[0]//size_coef)%2 == 0:
        return (img.shape[0]//size_coef) + 1
    return img.shape[0]//size_coef

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological closing to fill in small holes in the object
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    # Perform morphological erosion to shrink the object
    #erosion = cv.erode(closing, kernel, iterations=5)
    erosion = cv.dilate(img, kernel, iterations=1)

    # Display the original and processed images side by side
    combined = np.hstack([img, erosion])
    #dessiner(combined, "erosion")
    return erosion


def line_detection(gray,base_img):
    dst = cv.Canny(gray, 50, 200, None, 3)
    #dessiner(dst, "canny")

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    cdst_gray = np.copy(cdst)
    cdst_filter = np.copy(cdst)
    cdst_filter1 = np.copy(cdst)

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

    #dessiner(cdst, "lines")

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
            #print(f"coordinates (x1, y1): {l[1]}, {l[0]}, color : {base_img[l[1], l[0]]}")
            #print(f"coordinates (x1, y1): {l[1]}, {l[0]}, color : {base_img[l[1], l[0]]}")

    """lines_gray = filter_gray_lines(linesP, base_img)

    if lines_gray is not None:
        for i in range(0, len(lines_gray)):
            l = linesP[i][0]
            cv.line(cdst_gray, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
            #print(f"coordinates (x1, y1): {l[1]}, {l[0]}, color : {base_img[l[1], l[0]]}")
            #print(f"coordinates (x1, y1): {l[1]}, {l[0]}, color : {base_img[l[1], l[0]]}")"""


    #dessiner(cdstP, "probability")
    #dessiner(cdst_gray,"filtered")

    #filter the most centered ones

    #if lines are found
    border_lines = []
    if linesP is not None:
        border_lines = filter2(linesP, base_img.shape[0], base_img.shape[1])

    """
    l1,l2,l3,l4 = border_lines
    cv.line(cdst_filter1, (l1[0], l1[1]), (l1[2], l1[3]), (0, 0, 255), 3, cv.LINE_AA)
    cv.line(cdst_filter1, (l2[0], l2[1]), (l2[2], l2[3]), (0, 0, 255), 3, cv.LINE_AA)
    cv.line(cdst_filter1, (l3[0], l3[1]), (l3[2], l3[3]), (0, 0, 255), 3, cv.LINE_AA)
    cv.line(cdst_filter1, (l4[0], l4[1]), (l4[2], l4[3]), (0, 0, 255), 3, cv.LINE_AA)
    dessiner(cdst_filter1, "filter1")"""


    #la fin des lignes
    endpoints = []
    #draw lines of the edges
    for line in border_lines:
        endpoint = draw_line((line[0], line[1]), (line[2], line[3]),cdst_filter)
        endpoints.append(endpoint)

    #dessiner(cdst_filter, "filter")

    extracted_img, board_limit = extraction_tableau(base_img, endpoints)
    #dessiner(extracted_img, "extracted")

    #cv.imshow("Source", gray)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    #cv.waitKey()

    return extracted_img, board_limit


"""def filter_gray_lines(lines, base_img):
    gray_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if is_gray(base_img[y1, x1]) and is_gray(base_img[y2, x2]):
            gray_lines.append(line)
    return gray_lines



def is_gray(pixel, tolerance=GRAY_TOLERANCE):
    #print(pixel)
    r, g, b = pixel
    if abs(float(r) - float(g)) < tolerance and abs(float(g) - float(b)) < tolerance and abs(float(b) - float(r)) < tolerance:
        return True
    else:
        return False

def majoritory_gray(img, pixel_coordinates, tolerance = GRAY_TOLERANCE):
    neighborhood = img[max(pixel_coordinates[0] - 5, 0):min(pixel_coordinates[0] + 6, img.shape[0]),
                   max(pixel_coordinates[1] - 5, 0):min(pixel_coordinates[1] + 6, img.shape[1])]
    gray_counter = 0
    print("neigbors")
    print(neighborhood)
    for row in neighborhood:
        for pixel in row:
            if is_gray(pixel):
                gray_counter += 1

    if gray_counter >= 80:
        return True
    else:
        return False"""



def filter2(linesP, height, length):
    up_border = get_upper_border(linesP, height//2)
    bottom_border = get_bottom_border(linesP, height//2)
    left_border = get_left_border(linesP, length//2, up_border[1], bottom_border[1])
    right_border = get_right_border(linesP, length//2, up_border[1], bottom_border[1])

    print(f"upper : {up_border}")
    print(f"upper : {bottom_border}")
    print(f"upper : {left_border}")
    print(f"upper : {right_border}")

    return up_border, bottom_border, left_border, right_border


def get_upper_border(lines, middle_height):
    max_p1 = (0,0)
    max_p2 = (0,0)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #check if the line is on the top part and if it's horizontal
        if y1 < middle_height and y2 < middle_height and is_horizontal((x1, y1), (x2, y2)):
            #chek if the point is the maximum
            if y1 > max_p1[1] and y2 > max_p2[1]:
                max_p1 = (x1, y1)
                max_p2 = (x2, y2)
    return max_p1[0], max_p1[1], max_p2[0], max_p2[1]

def get_bottom_border(lines, middle_height):
    min_p1 = (0, middle_height * 2)
    min_p2 = (0, middle_height * 2)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # check if the line is on the bottom part and if it's horizontal
        if y1 > middle_height and y2 > middle_height and is_horizontal((x1, y1), (x2, y2)):
            # chek if the point is the minimum
            if y1 < min_p1[1] and y2 < min_p2[1]:
                min_p1 = (x1, y1)
                min_p2 = (x2, y2)
    return min_p1[0], min_p1[1], min_p2[0], min_p2[1]


def get_left_border(lines, middle_length, up_border, down_border):
    max_p1 = (0,0)
    max_p2 = (0,0)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #check if the line is on the top part and if it's horizontal
        if x1 < middle_length and x2 < middle_length and is_vertical((x1, y1), (x2, y2)) and down_border > y1 > up_border:
            #chek if the point is the maximum
            if x1 > max_p1[0] and x2 > max_p2[0]:
                max_p1 = (x1, y1)
                max_p2 = (x2, y2)
    return max_p1[0], max_p1[1], max_p2[0], max_p2[1]

def get_right_border(lines, midddle_length, up_border, down_border):
    min_p1 = (midddle_length * 2, 0)
    min_p2 = (midddle_length * 2, 0)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # check if the line is on the bottom part and if it's horizontal
        if x1 > midddle_length and x2 > midddle_length and is_vertical((x1, y1), (x2, y2)) and down_border > y1 > up_border:
            # chek if the point is the minimum
            if x1 < min_p1[0] and x2 < min_p2[0]:
                min_p1 = (x1, y1)
                min_p2 = (x2, y2)
    return min_p1[0], min_p1[1], min_p2[0], min_p2[1]


def is_horizontal(p1, p2, threshold=ACCEPTABLE_ANGLE):
    """
    Check if a line defined by two points is approximately horizontal.
    """
    # calculate the absolute difference between the y-coordinates
    diff = abs(p1[1] - p2[1])

    # calculate the length of the line
    length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # calculate the ratio of the absolute difference and the length
    ratio = diff / length

    # check if the ratio is below the threshold
    return ratio < threshold

def is_horizontal2(p1, p2, threshold=ACCEPTABLE_ANGLE): #environ 20 degre
    """Check if the line formed by two points is approximately horizontal.
    """
    x1, y1 = p1
    x2, y2 = p2
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    angle = 0.0
    if dx != 0:
        angle = math.atan(dy / dx)
    return angle < threshold

def is_vertical2(p1, p2, threshold=ACCEPTABLE_ANGLE): #environ 20 degre
    """Check if the line formed by two points is approximately vertical.
    """
    x1, y1 = p1
    x2, y2 = p2
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    angle = 0.0
    if dy != 0:
        angle = math.atan(dx / dy)
    return angle < threshold


def is_vertical(p1, p2, threshold=ACCEPTABLE_ANGLE):
    """
    Check if a line defined by two points is approximately vertical.
    """
    # calculate the absolute difference between the x-coordinates
    diff = abs(p1[0] - p2[0])

    # calculate the length of the line
    length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # calculate the ratio of the absolute difference and the length
    ratio = diff / length

    # check if the ratio is below the threshold
    return ratio < threshold


def draw_line(p1, p2, img):
    """
    draw the line passing through the two points that extends to the edges of the image
    """
    x1, y1 = p1
    x2, y2 = p2

    #check if there is a line
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return []
    # check if it is not a vertical line
    if x1 - x2 == 0:
        cv.line(img, (x1, 0), (x1, img.shape[0]), (255, 0, 0), 3, cv.LINE_AA)
        return [(x1, 0), (x1, img.shape[0])]
    elif y2 - y1 == 0: #horizonta
        cv.line(img, (0, y1), (img.shape[1], y1), (255, 0, 0), 3, cv.LINE_AA)
        return[(0, y1), (img.shape[1], y1)]
    print(f"point {p1, p2}")
    # calculate the equation of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    print(f"m: {m}, b {b}")

    # calculate the intersection points with the four edges of the image
    left_pt = (0, int(b))
    right_pt = (img.shape[1] - 1, int(m * (img.shape[1] - 1) + b))
    top_pt = (int(-b / m), 0)
    bottom_pt = (int((img.shape[0] - 1 - b) / m), img.shape[0] - 1)

    # find the endpoints of the line that extend to the edges of the image
    endpoints = []
    if left_pt[1] >= 0 and left_pt[1] < img.shape[0]:
        endpoints.append(left_pt)
    if right_pt[1] >= 0 and right_pt[1] < img.shape[0]:
        endpoints.append(right_pt)
    if top_pt[0] >= 0 and top_pt[0] < img.shape[1]:
        endpoints.append(top_pt)
    if bottom_pt[0] >= 0 and bottom_pt[0] < img.shape[1]:
        endpoints.append(bottom_pt)

    # draw the line passing through the two points that extends to the edges of the image
    if len(endpoints) == 2:
        cv.line(img, endpoints[0], endpoints[1], (255, 0, 0), 3, cv.LINE_AA)

    return endpoints


def extraction_tableau(img, end_points):
    """Extraction de tableau en mettant 0 pour les pixels qui sont pas sur les tableau par rapport aux bords trouvés"""
    extracted_img = img.copy()

    if len(end_points) == 0:
        return img, [0,img.shape[0],0,img.shape[1]]

    print(end_points)
    top_line = end_points[0]
    bottom_line = end_points[1]
    left_line = end_points[2]
    right_line = end_points[3]

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if len(top_line) == 2 and is_over_line((x, y), top_line[0], top_line[1]):
                extracted_img[y,x] = [0,0,0]
                continue
            elif len(bottom_line) == 2 and is_under_line((x, y), bottom_line[0], bottom_line[1]):
                extracted_img[y,x] = [0,0,0]
                continue
            elif len(left_line) == 2 and is_left_of_line((x, y), left_line[0], left_line[1]):
                extracted_img[y,x] = [0,0,0]
                continue
            elif len(right_line) == 2 and is_right_of_line((x, y), right_line[0], right_line[1]):
                extracted_img[y,x] = [0,0,0]
                continue

    #get the bord limits
    vertical_min = 0 if len(top_line) < 2 else min(top_line[0][1], top_line[1][1])
    vertical_max = img.shape[0] if len(bottom_line) < 2 else max(bottom_line[0][1], bottom_line[1][1])
    horizontal_min = 0 if len(left_line) < 2 else min(left_line[0][0], left_line[1][0])
    horizontal_max = img.shape[1] if len(right_line) < 2 else max(right_line[0][0], right_line[1][0])
    print("hello")
    print([vertical_min, vertical_max, horizontal_min, horizontal_max])
    return extracted_img, [vertical_min, vertical_max, horizontal_min, horizontal_max]
def is_under_line(pixel, line_p1, line_p2):
    """
    Checks if a given pixel is below a line defined by two points.
    Returns True if the pixel is below the line, and False otherwise.
    """
    x, y = pixel
    x1, y1 = line_p1
    x2, y2 = line_p2
    if x2 == x1:  # Vertical line
        return x <= x1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return y > slope * x + y_intercept


def is_over_line(pixel, line_p1, line_p2):
    """
    Checks if a given pixel is over a line defined by two points.
    Returns True if the pixel is over the line, and False otherwise.
    """
    x, y = pixel
    x1, y1 = line_p1
    x2, y2 = line_p2
    if x2 == x1:  # Vertical line
        return x >= x1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return y < slope * x + y_intercept

def is_left_of_line(pixel, line_p1, line_p2):
    """
    Checks if a given pixel is to the left of a line defined by two points.
    Returns True if the pixel is to the left of the line, and False otherwise.
    """
    x, y = pixel
    x1, y1 = line_p1
    x2, y2 = line_p2
    if y2 == y1:  # Horizontal line
        return y <= y1
    elif x2 == x1:
        return x <= x1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return x < (y - y_intercept) / slope

def is_right_of_line(pixel, line_p1, line_p2):
    """
    Checks if a given pixel is to the left of a line defined by two points.
    Returns True if the pixel is to the left of the line, and False otherwise.
    """
    x, y = pixel
    x1, y1 = line_p1
    x2, y2 = line_p2
    if y2 == y1:  # Horizontal line
        return y >= y1
    elif x2 == x1: #horizontal line
        return x >= x1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return x > (y - y_intercept) / slope

def dessiner(img, name):
	plt.figure(name)
	plt.imshow(img, vmin=0, vmax=255)
	plt.show()

def test(directory):

    for i, file in enumerate(os.listdir(directory)):
        if i == 0:
            continue

        img = cv.imread(os.path.join(directory, file))
        # img = cv.imread(r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\ImagesProjetL3\13.jpg")
        dessiner(img, "image")
        med = median(img)
        gray = cv.cvtColor(med, cv.COLOR_BGR2GRAY)
        line_detection(gray, img)

if __name__ == '__main__':
    main()