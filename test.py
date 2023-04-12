import numpy as np
from PIL import Image, ImageDraw

def create_polygon_mask(points, width, height):
    # Create a new black image
    img = Image.new('L', (width, height), 0)
    # Create a new white drawing object
    draw = ImageDraw.Draw(img)
    # Draw the polygon using the provided points
    draw.polygon(points, fill=1)
    # Convert the image to a numpy array
    mask = np.array(img)
    # Invert the mask so that the polygon is 1 and the background is 0
    #mask = 1 - mask
    return mask



def main():
    a=1

if __name__ == '__main__':
    main()