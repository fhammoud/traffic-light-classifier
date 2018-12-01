import helpers
import random
import test_functions

tests = test_functions.Tests()

# Image data directories
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Test for one_hot_encode function
print("** Testing one-hot encoding function:")
tests.test_one_hot(helpers.one_hot_encode)

# Find all misclassified images in a given test set
MISCLASSIFIED = helpers.get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print()
print('** Calculating accuracy in test set:')
print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
print()

if(len(MISCLASSIFIED) > 0):
    # Test code for red as green
    print("** Testing Red as Green")
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
