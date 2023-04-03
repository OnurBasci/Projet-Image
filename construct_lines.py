import numpy as np
import cv2 as cv
from RLSA import dessiner, apply_rlsa
from filter import filter_small_big_components

MEDIAN_KERNEL_SIZE = 7


def main():
    img_path = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3\13.jpg"
    #buff = r"C:\Users\onurb\PycharmProjects\Projet-Image\component3.png"
    construct_lines(img_path)
    #buffer(buff)



def construct_lines(img_path):
    rlsa, border_image = apply_rlsa(img_path)
    print((border_image.shape[0], border_image.shape[1]))
    dessiner(rlsa,"rlsa")
    #apply median to remove lines
    med = cv.medianBlur(rlsa, ksize = MEDIAN_KERNEL_SIZE)

    dessiner(med, "median")

    #filter
    filter = filter_small_big_components(med)

    dessiner(filter, "median")

    #lines, words = get_lines(filter)
    lines, contents = get_lines_v2(filter)

    remove_subsets(lines, contents, border_image)

    draw_lines(lines, border_image)


def buffer(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    lines, contents = get_lines_v2(img)
    #print(lines)
    remove_subsets(lines, contents, img)
    draw_lines(lines, img)


def get_lines(rlsa_images):
    """
    :param rlsa_images: gets an image and extract all of the connected components
    :return: list of connected components
    """
    lines = []
    words = []
    line = []

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

        #print(
        #    f"curr : {(x1, y1, x2, y2)} last {(words[i - 2][1][0], words[i - 2][1][1], words[i - 2][1][2], words[i - 2][1][3])}")
        #print(words[i-2][1][1] - last_word_lengt//2 < center_y_coor_of_curr_word < words[i-2][1][3] + last_word_lengt//2)

        if words[i-2][1][1] - last_word_lengt//2 < center_y_coor_of_curr_word < words[i-2][1][3] + last_word_lengt//2 and words[i-2][1][0] < x1 < words[i-2][1][2] + distance_between_words:
            line.append([word, (x1, y1, x2, y2)])
            #print(f"curr : {(x1,y1,x2,y2)} last {(words[i-2][1][0], words[i-2][1][1], words[i-2][1][2], words[i-2][1][3])}")
            #dessiner(word, "curr")
            #dessiner(words[i-2][0], "last")
            #print(line)
            continue
        else:
            lines.append(line)
            line = []
            line.append([word, (x1, y1, x2, y2)])
    return lines, words


def get_lines_v2(rlsa_image):

    lines = []
    contents = []
    line = []

    discovered = []

    masks, labels = get_masks(rlsa_image)

    for i, mask in enumerate(masks):
        Found = True
        mask_buf = mask.copy()


        #a list representing the contents (label) of the line
        content = {i+1}


        while Found:
            #print("hello")
            line.append(mask_buf)
            word_length = mask_buf[1][3] - mask_buf[1][1]
            distance_between_words = 2 * word_length
            #print("hello")
            #get the box from the end of the word to 3 times more
            buf_img = rlsa_image[mask_buf[1][1]: mask_buf[1][3], mask_buf[1][2]: mask_buf[1][2] + distance_between_words]

            #dessiner(mask_buf[0], "mask")
            #dessiner(buf_img, "buf")

            # Check if there is a word next to the current word
            coordinates = np.where(buf_img == 255)

            #dessiner(mask_buf[0], "mask_buf")

            if coordinates[0].size > 0:
                #get the next mask
                white_point = (mask_buf[1][1] + coordinates[0][0], mask_buf[1][2] + coordinates[1][0])

                # Get the label of the connected component that contains the given point
                label = labels[white_point]

                content.add(label)

                # if already found recalculate
                #if label in discovered:
                #    remove_multiple_apparence(lines, contents, label)

                # if already discovered remove the line
                """if label in discovered:
                    print(label)
                    remove_multiple_apparence(lines, contents, i + 1)"""


                #dessiner(masks[label][0], "labeled")
                #discovered.append(label)
                #dessiner(mask_buf[0], "mask buf")
                mask_buf = masks[label - 1]

                continue
            else:
                #if the line is the biggest then add it
                if check_biggest(contents, content):
                    #add the content
                    contents.append(content)

                    lines.append(line)
                Found = False
                line = []

    return lines, contents


def check_biggest(contents, content):
    biggest = True
    for c in contents:
        if content.issubset(c):
            biggest = False

    return biggest

def remove_subsets(lines, contents, base_img):
    """
    this function is to remove multiple box surrounding same object
    """
    to_delete_index = []

    #find the indexes
    for i, c in enumerate(contents):
        for c2 in contents:
            if c is c2:
                continue
            if c.issubset(c2):
                to_delete_index.append(i)
                break

    #remove the elements
    for index in sorted(to_delete_index, reverse=True):
        #print(index)
        del lines[index]


def get_masks(rlsa_image):
    filtered_img = rlsa_image.copy()

    masks = []

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filtered_img)

    for i in range(1, num_labels):
        #get box coordinates
        x1 = stats[i, cv.CC_STAT_LEFT]
        y1 = stats[i, cv.CC_STAT_TOP]
        x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]


        # get the component image
        mask = (labels == i) * rlsa_image

        masks.append([mask, (x1,y1,x2,y2)])

    return masks, labels


def draw_lines(lines, base_img):
    mask = np.zeros((base_img.shape[0], base_img.shape[1]), dtype=np.uint8)
    base = base_img.copy()

    print(lines)

    for line in lines:
        left_top = (line[0][1][0], line[0][1][1])
        left_down = (line[0][1][0], line[0][1][3])
        right_top = (line[-1][1][2], line[-1][1][1])
        right_down = (line[-1][1][2], line[-1][1][3])

        #(start_pos, end_pos)

        print((left_top, right_down))

        #draw polygon
        pts = np.array([[left_top[0], left_top[1]], [left_down[0], left_down[1]], [right_down[0], right_down[1]], [right_top[0], right_top[1]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(base, [pts], True, (0, 255, 255), 2)

        mask[left_top[1] : right_down[1], left_top[0] : right_down[0]] = 255
        #cv.rectangle(base, (start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]), (0, 255, 0), 3)


    dessiner(mask,"mask")
    dessiner(base, "base")

    return mask

if __name__ == '__main__':
    main()