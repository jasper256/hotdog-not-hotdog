import cv2 as cv
import pickle
import numpy as np
import os
from decouple import config
from random import shuffle


def retrieve_img_arrays(DIR, CATEGORY, LABEL, IMG_SIZE):
    path = os.path.join(DIR, CATEGORY)
    # Opencv defualts to opening images in BGR as opposed to RGB
    # Training the model on RGB images will save us time when we later open the user's image using Pillow, which defaults to RGB
    data = [(cv.cvtColor(cv.resize(img_array, (IMG_SIZE, IMG_SIZE)), cv.COLOR_BGR2RGB), LABEL)
            for img_array in 
            [cv.imread(os.path.join(path, img))
            for img in os.listdir(path)]]
    return data

# Use environment variables for train and test data locations
TRAIN_DATA_DIR = config("TRAIN_DATA_DIR")
TEST_DATA_DIR = config("TEST_DATA_DIR")

CATEGORIES = ("nothotdog", "hotdog")
IMG_SIZE = 100 # images will be square, (IMG_SIZE px x IMG_SIZE px)

train = retrieve_img_arrays(TRAIN_DATA_DIR, CATEGORIES[0], 0, IMG_SIZE) + retrieve_img_arrays(TRAIN_DATA_DIR, CATEGORIES[1], 1, IMG_SIZE)
test = retrieve_img_arrays(TEST_DATA_DIR, CATEGORIES[0], 0, IMG_SIZE) + retrieve_img_arrays(TEST_DATA_DIR, CATEGORIES[1], 1, IMG_SIZE)

# Randomize order of train and test datasets
shuffle(train)
shuffle(test)

# Separate images from labels
train_imgs = np.array([data[0] for data in train])
train_labels = np.array([data[1] for data in train])
test_imgs = np.array([data[0] for data in test])
test_labels = np.array([data[1] for data in test])

# Srink pixel values from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Save processed data
with open("train_imgs.pickle", "wb") as f:
    pickle.dump(train_imgs, f)
with open("train_labels.pickle", "wb") as f:
    pickle.dump(train_labels, f)
with open("test_imgs.pickle", "wb") as f:
    pickle.dump(test_imgs, f)
with open("test_labels.pickle", "wb") as f:
    pickle.dump(test_labels, f)