import sys
import os
from veriteterain2 import get_terrain_figure, true_positif_false_negatif, faux_positif


def get_average_succes_rates(dir_path):
    """
    input: path of a directroy containing json files and images
    """
    avarage_tp = 0
    avarage_fn = 0
    avarage_fp = 0

    found1 = False
    found2 = False
    json_file_name = ""
    image_file_name = ""

    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            found1 = True
            json_file_name = file
        elif file.endswith(".jpg") or file.endswith(".jpeg"):
            found2 = True
            image_file_name = file
        if found1 and found2:
            print(json_file_name)
            print(image_file_name)
            mask_terrain, mask_figures = get_terrain_figure(os.path.join(dir_path, json_file_name), os.path.join(dir_path, image_file_name))

            found1 = False
            found2 = False
