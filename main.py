import cv2
import helpers
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

image_number = 750
selected_image = IMAGE_LIST[image_number][0]
label = IMAGE_LIST[image_number][1]
print(selected_image.shape, label)
plt.imshow(selected_image)
plt.show()