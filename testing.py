# This file was just used to test the pipeline with 1 image
import cv2
import helpers
import numpy as np
import test_functions

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"

# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
# Print out 1. The shape of the image and 2. The image's label
image_number = 750

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[image_number][0]
label = IMAGE_LIST[image_number][1]
print("Example yellow light:", selected_image.shape, label)
# plt.imshow(selected_image)

tests = test_functions.Tests()

# Test for one_hot_encode function
print("** Testing one-hot encoding function:")
tests.test_one_hot(helpers.one_hot_encode)


# Standardize all training images
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

# Display a standardized image and its label
# print(STANDARDIZED_LIST[image_number][1])
# plt.imshow(STANDARDIZED_LIST[image_number][0])
# plt.show()

# Convert and image to HSV colorspace
# Visualize the individual color channels
test_im = STANDARDIZED_LIST[image_number][0]
test_label = STANDARDIZED_LIST[image_number][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

z = v[9:21,7:24]

# Plot the original image and the three channels
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
# ax1.set_title('Standardized image')
# ax1.imshow(test_im)
# ax2.set_title('H channel')
# ax2.imshow(h, cmap='gray')
# ax3.set_title('S channel')
# ax3.imshow(s, cmap='gray')
# ax4.set_title('V channel')
# ax4.imshow(v, cmap='gray')
# plt.suptitle("Example yellow light channels")

# plt.show()

cropped_mask, feature = helpers.create_feature(test_im)


objects = list(reversed(range(len(feature))))
# objects = range(len(feature))
y_pos = np.arange(len(objects))
counts = feature
width = 1
# plt.barh(y_pos, counts, width)
# plt.yticks(y_pos, objects)
# plt.show()

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
# ax1.set_title('Standardized image')
# ax1.imshow(test_im)
# ax2.set_title('V channel')
# ax2.imshow(v, cmap='gray')
# ax3.set_title('Cropped mask')
# ax3.imshow(cropped_mask, cmap='gray')
# ax4.set_title('Bright frequency')
# ax4.barh(y_pos, counts, width)
# ax4.set_yticks(y_pos, objects)
# ax4.imshow()
# plt.suptitle("Example yellow light feature")

# plt.show()