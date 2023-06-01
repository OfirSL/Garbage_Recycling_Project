"""
Move label text files from one folder to another, based on a list of images (from images folder)
"""

import glob
import shutil
import ntpath

original_folder_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/waste segergation.v15i.yolov8/all'
labels_origin_folder_path = original_folder_path + '/labels/'
image_origin_folder_path = original_folder_path + '/images/'
new_folder_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/waste segergation.v15i.yolov8/not_used'
labels_new_folder_path = new_folder_path + '/labels/'
image_new_folder_path = new_folder_path + '/images/'

labels_to_move = ['0','2','3','5']

# create list of images
labels_list = glob.glob(labels_origin_folder_path + '/*')

for label_file in labels_list:
    with open(label_file) as f:
        class_ = f.read(1)

    if class_ in labels_to_move:
        head, tail = ntpath.split(label_file)
        org_label = label_file
        det_label = labels_new_folder_path + tail
        shutil.move(org_label, det_label)

        org_image = "".join(glob.glob(image_origin_folder_path + tail[:-3] + '*'))
        head_image, tail_image = ntpath.split(org_image)
        des_image = image_new_folder_path + tail_image
        shutil.move(org_image, des_image)


