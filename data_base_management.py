import os
import random
import shutil

TRAINING_DATA = [60, 16, 10, 46, 70, 51, 23, 24, 71, 29, 11, 74, 2, 22, 68, 40, 19, 13, 66, 37, 28, 77, 27, 79, 59, 76, 31, 20, 78, 72, 44, 67, 21, 39, 41, 57, 49, 18, 52, 25, 75, 8, 9, 48, 12, 17, 65, 0, 7]

def main():
    src = r"C:\Users\onurb\PycharmProjects\Projet-Image\ImagesProjetL3"
    dst = r"C:\Users\onurb\PycharmProjects\Projet-Image\training_data"
    #get_training_data(src, dst)
    get_test_data(src, r"C:\Users\onurb\PycharmProjects\Projet-Image\test_data")

def get_training_data(data_path, training_directory):
    #get 49 random numbers (49 is 60 percent of the training set/81 images
    #file_numbers = random.sample(range(81), 49)

    for i, file in enumerate(os.listdir(data_path)):
        print(i)
        if i in TRAINING_DATA:
            shutil.copyfile(os.path.join(data_path, file), os.path.join(training_directory, file))


def get_test_data(src, dst):
    # get the test data from the training data

    for i, file in enumerate(os.listdir(src)):
        if not(i in TRAINING_DATA):
            shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))


if __name__ == '__main__':
    main()