import cv2
import numpy as np

# Load the image
img = cv2.imread(r"C:\Users\onurb\pycharm_projects\Image_Tps\ImagesProjetL3\ImagesProjetL3\12.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find lines using the Hough transform
lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

# Create arrays to store the angles of the lines
horizontal_angles = []
vertical_angles = []

# Loop through the lines
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Calculate the angle of the line
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # Classify the line as horizontal or vertical based on its angle
    if angle < 45 or angle > 135:
        vertical_angles.append(angle)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        horizontal_angles.append(angle)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Find the median angles of the horizontal and vertical lines
median_horizontal_angle = np.median(horizontal_angles)
median_vertical_angle = np.median(vertical_angles)

# Calculate the angle of the sides of the rectangle
if median_horizontal_angle > 0:
    side_angle = median_horizontal_angle - 90
else:
    side_angle = median_horizontal_angle + 90

# Find the intersections of the lines
x_intersect = int(np.cos(median_horizontal_angle * np.pi / 180) * np.cos(median_vertical_angle * np.pi / 180))
y_intersect = int(np.sin(median_horizontal_angle * np.pi / 180) * np.sin(median_vertical_angle * np.pi / 180))

# Draw a circle at the intersection of the lines
cv2.circle(img, (x_intersect, y_intersect), 5, (255, 0, 0), -1)

# Show the image with the lines and intersection point marked
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
