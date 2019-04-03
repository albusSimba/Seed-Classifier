import pandas as pd
import numpy as np
from os.path import join
from image_preprocessing import image_to_array
from sklearn.utils import shuffle


def compile_images_labels(data_dir, label_csv, save_name, new_height, new_width, generate_test_data=0.2):
    labelCSV = pd.read_csv(label_csv)
    sample_size = len(labelCSV)

    images = np.zeros([sample_size, new_height, new_width, 3])
    labels = np.zeros([sample_size, 1])

    for i in range(sample_size):
        print(str(i) + "/" + str(sample_size))
        img_dir = join(data_dir, labelCSV['class'][i])
        img_arr = image_to_array(img_dir, labelCSV['img_id'][i], new_height, new_width, normalize=True)
        label = labelCSV['label'][i]
        images[i] = img_arr
        labels[i] = label

    images, labels = shuffle(images, labels)
    n = int(len(labels) * generate_test_data)
    test_data, test_labels = images[:n], labels[:n]
    train_data, train_labels = images[n:], labels[n:]

    np.save(save_name + "_data.npy", train_data)
    print("Training data saved successfully...")
    np.save(save_name + "_labels.npy", train_labels)
    print("Training labels saved successfully...")
    np.save("Test_data.npy", test_data)
    print("Test data saved successfully...")
    np.save("Test_labels.npy", test_labels)
    print("Test labels saved successfully...")


# directory list
training_image_dir = 'Data/train'
test_image_dir = 'Data/test'
labels_csv_dir = 'Data/labels.csv'

# parameters
resize_height = 224
resize_width = 224

# compile_images_labels(training_image_dir, labels_csv_dir, labels_encoding_dir, 'training', resize_height, resize_width)
# compile_images_labels(training_image_dir, labels_csv_dir, 'all_classes', resize_height, resize_width)
compile_images_labels(training_image_dir, labels_csv_dir, 'training', resize_height, resize_width)
