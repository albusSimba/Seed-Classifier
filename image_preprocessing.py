import math
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from os.path import join, isdir
from os import listdir, mkdir
import numpy as np


def plot_images(img_dir, label, num_to_plot):
    '''
    Plot the images of a class
    :param img_dir: the path of the training data folder
    :param label: the label of images to display
    :param num_to_plot: number of images to display
    :return:
    '''
    # find the labeled images
    img_dir = join(img_dir, label)
    img_ids = listdir(img_dir)
    if '.DS_Store' in img_ids:
        img_ids.remove('.DS_Store')

    # plot the pictures
    nb_cols = 4
    nb_rows = math.ceil(num_to_plot/4)

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(6, 6))

    n = 0
    for i in range(nb_rows):
        for j in range(nb_cols):
            img_to_display = join(img_dir, img_ids[n])
            axs[i,j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(cv2.imread(img_to_display))
            n += 1
            if(n==num_to_plot):
                break
    plt.show()


def image_to_array(img_dir, file_name, new_height, new_width, normalize = False):
    '''
    Return an array representing the input image
    :param img_dir: the path of the folder containing the image
    :param file_name: the file name of the image
    :param new_height: height the image resized to
    :param new_width: width the image resized to
    :param normalize: if True, the values will be normalized to be 0-1
    :return:
    '''
    img = image.load_img(join(img_dir, file_name), target_size=(new_height, new_width))
    img = image.img_to_array(img) # the array shape is 224x224x3
    if normalize == True:
        img = img/255
    return img


def create_mask_for_img(image):
    '''
    Create a mask for the plant using HSV color-space
    :param image: an image ready by cv2.imread
    :return: image mask
    '''
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_img(image):
    '''
    Segment the plant from the background
    :param image: an image ready by cv2.imread
    :return: segmented plant image
    '''
    mask = create_mask_for_img(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    '''
    sharpen the image
    :param image: an image read by cv2.imread
    :return: sharpened image
    '''
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def batch_image_segmentation(img_dir, label, save_dir):
    '''
    Take in a batch of images of the same class, segment the plants from the background and save the images in the save_dir
    :param img_dir: the path of the folder containing targeted images
    :param label: class of targeted plant
    :param save_dir: the path to save the segmented images
    :return: None
    '''
    img_dir = join(img_dir, label)
    img_ids = listdir(img_dir)
    if '.DS_Store' in img_ids:
        img_ids.remove('.DS_Store')
    save_folder = join(save_dir, label)
    if not isdir(save_folder):
        mkdir(save_folder)
    for img_id in img_ids:
        print(label, "  ", img_id)
        img = cv2.imread(join(img_dir, img_id))
        img_segmented = segment_img(img)
        img_sharpen = sharpen_image(img_segmented)
        cv2.imwrite(join(save_folder, img_id), img_sharpen)

'''
# batch segmentation
training_image_dir = '/Users/shengguili/Documents/Projects/Kaggle/Plant_seedling_classification/Data/train'
save_dir = '/Users/shengguili/Documents/Projects/Kaggle/Plant_seedling_classification/Data/train_segmented'
labels = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]
for label in labels:
    batch_image_segmentation(training_image_dir, label, save_dir)
'''