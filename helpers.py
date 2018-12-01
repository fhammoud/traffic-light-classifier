# Helper functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    # Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    
    resized_im = cv2.resize(standard_im, (32,32))
    
    return resized_im

# One hot encode an image label
# Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    # Create a one-hot encoded label that works for all classes of traffic lights
    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    else:
        one_hot_encoded = [0, 0, 1]
    
    return one_hot_encoded

# Standardize image list
def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
# This feature should use HSV colorspace values
def create_feature(rgb_image):
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Create and return a feature value and/or vector
    # HSV channels
    h = hsv_image[:,:,0]
    s = hsv_image[:,:,1]
    v = hsv_image[:,:,2]

    # Mask bright values and crop
    # These values can be tweaked to get best results
    mask = cv2.inRange(v, 200, 256)
    cropped_mask = mask[5:27, 10:22]

    # Count bright pixels cropped mask image
    feature = []
    for i in range(len(cropped_mask)):
        sum = 0
        for j in range(len(cropped_mask[0])):
            if (cropped_mask[i,j] == 255):
                sum = sum + 1
        feature.append(sum)
    
    return cropped_mask, feature

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    # Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    
    final_img, feature = create_feature(rgb_image)
    
    predicted_label = []
    
    max_index = 0
    for i in range(1,len(feature)):
        if feature[i] > feature[max_index]:
            max_index = i
        
    if 0 <= max_index < 8:
        predicted_label = [1, 0, 0]
    elif 8 <= max_index < 16:
        predicted_label = [0, 1, 0]
    else:
        predicted_label = [0, 0, 1]
    
    return predicted_label

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels