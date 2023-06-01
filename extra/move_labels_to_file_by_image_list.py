"""
Move label text files from one folder to another, based on a list of images (from images folder)
"""

import glob
import shutil
import ntpath


labels_origin_folder_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/all_images/labels'
labels_new_folder_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/cups/labels'

# create list of images
images_to_move_folder = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/cups/images'
images_list_to_move = glob.glob(images_to_move_folder + '/*')

for image in images_list_to_move:
    head, tail = ntpath.split(image)

    org = labels_origin_folder_path + "/" + tail[:-3] + 'txt'
    des = labels_new_folder_path + "/" + tail[:-3] + 'txt'
    shutil.move(org, des)

