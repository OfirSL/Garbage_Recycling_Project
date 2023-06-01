import splitfolders

input_folder = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/dataset_for_YOLO2'
output = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/dataset_for_YOLO2'
splitfolders.ratio(input_folder, output=output, seed=13, ratio=(.8, .1, .1), group_prefix=None, move=False)
