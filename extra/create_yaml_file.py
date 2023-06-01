import yaml

yaml_file_name = 'custom. yaml'

custom = """path:  /Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/all_images_for_YOLO
train: /Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/all_images_for_YOLO/train
test: /Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/all_images_for_YOLO/test
valid: /Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/all_images_for_YOLO/val

#Classes
nc: 7

#classes names
names: ['blue_paper', 'green_general', 'orange_packaging', 'purple_glass', 'cardboard', 'brown_organic', 'deposit_bottles']
"""

def write_yaml(data):
    """ A function to write YAML file"""
    with open(yaml_file_name, 'w') as f:
        yaml.dump(data, f)

# write A python object to a file
write_yaml(custom)
