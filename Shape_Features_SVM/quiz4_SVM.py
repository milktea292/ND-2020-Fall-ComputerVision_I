# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame
# ___________________________________________
# Adam Czajka, Jin Huang, November 2019


import cv2
import math as mt
import numpy as np
from sklearn import svm
from skimage import measure
from sys import platform as sys_pf
import warnings
warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

# Read the image into grayscale
sample = cv2.imread('breakfast2.png')

# sample_small = cv2.resize(sample, (640, 480))
# cv2.imshow('Grey scale image',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert the original image to HSV
# and take H channel for further calculations
sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample_h = sample_hsv[:, :, 0]

# Show the H channel of the image
# sample_small = cv2.resize(sample_h, (640, 480))
# cv2.imshow('H channel of the image',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert the original image to grayscale
sample_grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Show the grey scale image
# sample_small = cv2.resize(sample_grey, (640, 480))
# cv2.imshow('Grey scale image',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# sample_small = cv2.resize(binary_image, (640, 480))
# cv2.imshow('Image after Otsu''s thresholding',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Morphological operations
# Step 1: Fill the holes.
im_floodfill = binary_image.copy()
h, w = binary_image.shape[ : 2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
binary_image = binary_image | im_floodfill_inv

# sample_small = cv2.resize(binary_image, (640, 480))
# cv2.imshow('Image after filling',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 2: Apply opening, closing and erosion
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5),np.uint8))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3),np.uint8))
binary_image = cv2.erode(binary_image,kernel= np.ones((3, 3),np.uint8),iterations = 9)

# sample_small = cv2.resize(binary_image, (640, 480))
# cv2.imshow('Image after morphological transformation',sample_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Find connected pixels and compose them into objects
labels = measure.label(binary_image)

# Calculate features for each object; since we want to differentiate
# between circular and oval shapes, the major and mino_r axes may help; we
# will use also the centroid to annotate the final result
properties = measure.regionprops(labels)

# Calculate features for each object:
# - perimeter (dimension 1)
# - intensity of the H channel (dimension 2)
features = np.zeros((len(properties), 2))

for i in range(0, len(properties)):
    features[i, 0] = properties[i].perimeter
    patch = sample_h[int(np.round(properties[i].centroid[0] - properties[i].major_axis_length / 2)) : int(np.round(properties[i].centroid[0] + properties[i].major_axis_length / 2)),
                    int(np.round(properties[i].centroid[1] - properties[i].major_axis_length / 2)) : int(np.round(properties[i].centroid[1] + properties[i].major_axis_length / 2))]
    features[i, 1] = np.mean(patch[ : ])

# Calculate ground truth labels for each object 
thrF1 = 250
thrF2 = 50
total_samples = 0
labels = np.zeros((len(features), 1))

for i in range(0, len(features)):
    if (features[i, 0] > thrF1):
        total_samples = total_samples + 1
        labels[i] = 1

    if ((features[i, 0] < thrF1) and (features[i, 1] > thrF2)):
        total_samples = total_samples + 1
        labels[i] = 2

    if ((features[i, 0] < thrF1) and (features[i, 1] < thrF2)):
        total_samples = total_samples + 1
        labels[i] = 3

print("Shape of features: ", features.shape)
print("Shape of labels: ", labels.shape)






# And now your tasks:


''' 
*** Task 0: It's good to look at the documentation of the SVM implementation in scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

There are links to example codes at the bottom of this webpage. 
''' 



'''
*** Task 1: Prepare train and test data

Split "features" into:
- "features_train" consisting of N first rows of "features" (i.e., features for N first objects)
- "features_test" consisting of features for the rest of the objects

Choose N to split "features" almost equally (note that the number of all objects in this example is odd).

Now split the "labels" into "labels_train" and "labels_test", so that these new vectors correspond to "features_train" and "features_test".
'''
N = round(len(features)/2)
features_train = features[0:N,:]
features_test  = features[N:len(labels)]


labels_train = labels[0:N]
labels_test  = labels[N:len(labels)]

''' 
*** Task 2: Create a linear SVM. 
Use "SVC" function from the sklearn.svm class. 
Set only the kernel type ("linear"), and leave other options as defaults.
'''
print("Training the SVM on the train set ...")
clf = svm.SVC(kernel="linear")


''' 
*** Task 3: Use your train samples (with appropriate labels) to train your SVM. 
Use "fit" function available in your SVM model.
'''
print("Predicting class labels on the test set ...")
clf = clf.fit(features_train, labels_train)

print("Predicting class labels on the test set ...")
''' 
*** Task 4: Use your test samples (with appropriate labels) to test your SVM. 
Use "predict" function available in your trained SVM model.
Assign predicted labels to "labels_predicted".
'''
labels_predicted = clf.predict(features_test)
labels_test = np.transpose(labels_test).flatten()

# That's it! Let's see how the SVM performed on this tiny test set:
print("Predicted labels:    ", labels_predicted)
print("Ground-truth labels: ", labels_test)
print("Prediction errors: ", np.sum(np.nonzero(labels_predicted-labels_test)))
