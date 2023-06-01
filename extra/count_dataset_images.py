import glob
train, val, test, all = 1, 1, 1, 1
dataset_path = '/Users/ofir/Documents/Data_Science/ITC/Data_Science_Course_Oct_2022/Final Project/Datasets/dataset_for_YOLO_all'
train_path = dataset_path + '/train/labels/'
val_path = dataset_path + '/val/labels/'
test_path = dataset_path + '/val/labels/'

labels = {0: 'blue_paper', 1: 'green_general', 2: 'orange_packaging', 3: 'purple_glass',
          4: 'cardboard', 5: 'brown_organic', 6: 'deposit_bottles', 7: 'electronics',
          8: 'clothing', 9: 'batteries', 10: 'medicines', 11: 'Light_bulbs'}

"train"
labels_count_train = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}
if train:
    train_list = glob.glob(train_path + '*')
    for file in train_list:
        with open(file) as f:
            for line in f:
                class_ = line[:2].strip()
            labels_count_train[class_] += 1

    print('\n\ntrain set count:')
    for class__ in labels_count_train:
        class_name = labels[int(class__)]
        print(f'{class_name}: {labels_count_train[class__]}')

"val"
labels_count_val = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}
if val:
    val_list = glob.glob(val_path + '*')
    for file in val_list:
        with open(file) as f:
            for line in f:
                class_ = line[:2].strip()
            labels_count_val[class_] += 1

    print('\n\nval set count:')
    for class__ in labels_count_val:
        class_name = labels[int(class__)]
        print(f'{class_name}: {labels_count_val[class__]}')

"test"
labels_count_test = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}
if test:
    test_list = glob.glob(val_path + '*')
    for file in test_list:
        with open(file) as f:
            for line in f:
                class_ = line[:2].strip()
            labels_count_test[class_] += 1

    print('\n\ntest set count:')
    for class__ in labels_count_test:
        class_name = labels[int(class__)]
        print(f'{class_name}: {labels_count_test[class__]}')

"all datasets"
labels_count_all = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}
if all:
    print('\n\nall datasets count:')
    sum = 0
    for class__ in labels_count_test:
        class_name = labels[int(class__)]
        labels_count_all[class__] = labels_count_test[class__] + labels_count_train[class__] + labels_count_val[class__]
        print(f'{class_name}: {labels_count_all[class__]}')
        sum += labels_count_all[class__]

    print(f'\ntotal number of labels: {sum}')


