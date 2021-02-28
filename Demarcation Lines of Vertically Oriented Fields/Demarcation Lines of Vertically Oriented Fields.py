import cv2
import numpy as np
import matplotlib.pyplot as plt

def CannyEdgeDetector(image):

	# Compute the median of the pixel intensities
	med = np.median(image)

	# Apply Canny edge detection using the computed median 
    # in defining lower and upper thresholds for hysteresis
	lower = int(max(0, 0.8 * med))
	upper = int(min(255, 1.2 * med))
	edgeImage = cv2.Canny(image, lower, upper)

	# Return the edged image
	return edgeImage

# Load our pattern
gray = cv2.imread('pattern.png',cv2.IMREAD_GRAYSCALE)

# This is what happens when we apply edge detector directly on this image
edgeImageCanny = CannyEdgeDetector(gray)
images = np.hstack((gray, edgeImageCanny))
cv2.imshow("Not what we wanted ...",images)

# Let's do it better:

'''
    Step 1: build the Gabor kernel that will enhance for us vertically oriented patches:
    cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)

    where:
    ksize  - size of kernel in pixels (n, n), i.e., size of our neighborhood
    sigma  - size of the Gaussian envelope, i.e., how wide is our Gaussian "hat"
    theta  - orientation of the normal to the filter's oscilation pattern; e.g., theta = 0.0 means vertical stripes
    lambda - wavelength of the sinusoidal oscilation; this together with sigma 
             is resposinble for frequencies enhanced by this filter
    gamma  - spatial aspect ratio; keep it 1
    phi    - phase offset; keep it 0
    ktype  - type and range of values that each pixel in the gabor kernel can hold; keep it cv2.CV_32F

'''

# ***TASK*** Select parameters of your Gabor kernel here:
ksize = 9       # try something between 5 and 15
sigma = 4       # try something between 2.0 and 4.0
theta = 0.0     # keep it 0.0 if you want to focus on vertically-oriented patterns 
lbd = 2       # try something between 2.0 and 4.0
gamma = 1.0     # keep it 1.0
psi = 0.0       # keep it 0.0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lbd, gamma, psi, ktype=cv2.CV_32F)

# Normalize the kernel and remove the DC component
kernel /= kernel.sum()
kernel -= kernel.mean()

# Curious how it looks? Here we go:
xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, kernel ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
plt.show()

# Step 2: image filtering
res1 = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
cv2.imshow("Filtering result",res1)

# Step 3: image binarization (let's use an idea with maximization of the Fisher ratio, implemeted by Otsu)
th2, res2 = cv2.threshold(res1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Otsu's binarization",res2)

# Step 4: morphological operations
# ***TASK*** Choose the type among cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_ERODE or cv2.MORPH_DILATE
# (or a sequence of those, in the order you think makes sense)
type = cv2.MORPH_OPEN
se_size = 15  # size of your structuring element (morphological operation kernel) -- try something between 5 and 15
se = np.ones((se_size,se_size), np.uint8)
res3 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel=se)
cv2.imshow("Morphological operations",res3)

# Step 4: check if your result allows to draw clear boundaries demarcating all vertically-oriented fields
res4 = CannyEdgeDetector(res3)
images = np.hstack((gray, res4))
cv2.imshow("My final result",images)

cv2.waitKey(0)
cv2.destroyAllWindows()
