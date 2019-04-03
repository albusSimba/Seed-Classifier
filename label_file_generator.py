import numpy as np
import csv
from os.path import join
from os import listdir

def label_csv_generator(data_dir, save_name):
    label_collection_dir = join(data_dir, "train")
    label_collection = listdir(label_collection_dir)
    # label_collection.remove('.DS_Store')
    label_collection.sort()

    save_file = open(save_name + '.csv', 'w', newline='')
    writer = csv.writer(save_file, delimiter=',')
    writer.writerow(['img_id', 'class', 'label'])
    for i in range(len(label_collection)):
        label = label_collection[i]
        img_dir = join(label_collection_dir, label)
        for img in listdir(img_dir):
            if len(img) > 0:
                writer.writerow([img, label, i])

    save_file.close()

data_dir = 'Data'
save_name = 'labels'

label_csv_generator(data_dir, save_name)