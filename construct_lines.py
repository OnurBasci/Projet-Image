import numpy as np
import cv2 as cv
from RLSA import dessiner, apply_rlsa
from filter import filter_small_big_components

MEDIAN_KERNEL_SIZE = 7


def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"
    construct_lines(img_path)



def construct_lines(img_path):
    rlsa, border_image = apply_rlsa(img_path)
    dessiner(rlsa,"rlsa")
    #apply median to remove lines
    med = cv.medianBlur(rlsa, ksize = MEDIAN_KERNEL_SIZE)

    dessiner(med, "median")

    #filter
    filter = filter_small_big_components(med)

    dessiner(filter, "median")

    lines, words = get_lines(filter)
    print(lines)
    draw_lines(lines, border_image)


def get_lines(rlsa_images):
    """
    :param rlsa_images: gets an image and extract all of the connected components
    :return: list of connected components
    """
    lines = []
    words = []
    line = []

    line_find = False

    filtered_img = rlsa_images.copy()

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filtered_img)

    for i in range(1, num_labels):
        #get box coordinates
        x1 = stats[i, cv.CC_STAT_LEFT]
        y1 = stats[i, cv.CC_STAT_TOP]
        x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]


        # get the component image
        word = (labels == i) * rlsa_images
        words.append([word,(x1,y1,x2,y2)])

        # dessiner(word, "word")
        if i == 1:
            line.append([word,(x1,y1,x2,y2)])
            continue

        #get the length of the world
        last_word_lengt = words[i-2][1][3] - words[i-2][1][1]

        distance_between_words = 3*last_word_lengt
        #print((x1,y1,x2,y2))
        #print((words[i-1][1][0], words[i-1][1][1], words[i-1][1][2], words[i-1][1][3]))

        #check if the words are on the same line and the distance between words are less than 2 times of the length
        center_y_coor_of_curr_word = (y2 + y1)/2
        #print(f"{words[i-2][1][0]}, {x1}, {words[i-2][1][0] + distance_between_words}")

        line.append([word, (x1, y1, x2, y2)])
        if words[i-2][1][1] - 10 < center_y_coor_of_curr_word < words[i-2][1][3] + 10 and words[i-2][1][0] < x1 < words[i-2][1][0] + distance_between_words:
            continue
        else:
            lines.append(line)
            line = []
    return lines, words

def draw_lines(lines, base_img):
    mask = np.zeros((base_img.shape[0], base_img.shape[1]), dtype=np.uint8)
    base = base_img.copy()

    for line in lines:
        start_pos = (line[0][1][0], line[0][1][1])
        end_pos = (line[-1][1][2], line[-1][1][3])

        print(start_pos, end_pos)

        mask[start_pos[1] : end_pos[1], start_pos[0] : end_pos[0]] = 255
        cv.rectangle(base, (start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]), (0, 255, 0), 3)


    dessiner(mask,"mask")
    dessiner(base, "base")

    return mask

if __name__ == '__main__':
    main()