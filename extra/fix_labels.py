""" convert yolo labels in all the text files in a folder according to the "convert_dict" """

import glob

labels = {0: 'blue_paper', 1: 'green_general', 2: 'orange_packaging', 3: 'purple_glass',
          4: 'cardboard', 5: 'brown_organic', 6: 'deposit_bottles', 7: 'electronics',
          8: 'clothing', 9: 'batteries', 10: 'medicines', 11: 'Light_bulbs'}

# names: [0'Light bulb', 1'battery', 2'clothes', 3'e waste', 4'glass', 5'metal', 6'organic', 7'paper', 8'plastic']
convert_dict = {'0': '11', '1': '9', '2': '8', '3': '7', '4': '3', '5': '2', '6': '5', '7': '0', '8': '2'}

# number_str = '10'
# convert_dict = {'0': number_str, '1': number_str, '2': number_str, '3': number_str, '4': number_str, '5': number_str}
folder_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/waste segergation.v15i.yolov8/all/labels'

file_list = glob.glob(folder_path + '/*')

for file in file_list:
    replaced_content = ""
    with open(file) as f:
        for line in f:
            original_class = line[0]
            if original_class == '\n':
                continue
            new_class = convert_dict[original_class]
            print(f'original: {line}')
            new_line = new_class + line[1:]
            print(f'new: {new_line}')
            replaced_content += new_line

    with open(file, 'w') as f:
        f.write(replaced_content)

