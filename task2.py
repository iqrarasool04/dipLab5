import cv2
import numpy as np
import matplotlib.pyplot as plt

#load the image
image = cv2.imread("wiki.jpg", cv2.IMREAD_GRAYSCALE)

#histogram before equalization
histogramBefore = cv2.calcHist([image],[0], None, [256], [0, 256])

#implementation of own equalization
cdf = histogramBefore.cumsum()
cdfNormalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
equalizedImage = cdfNormalized[image]
equalizedImage = np.uint8(equalizedImage)

#histogram after own equalization
histogramAfterOwn = cv2.calcHist([equalizedImage], [0], None, [256], [0, 256])

#opencv equalization
equalizedImageCV = cv2.equalizeHist(image)

#histogram after opencv equalization
histogramAfterCV = cv2.calcHist([equalizedImageCV], [0], None, [256], [0, 256])

#plot histogram and images
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title('Original Histogram')
plt.plot(histogramBefore)

plt.subplot(2, 2, 2)
plt.title('Histogram After Own Equalization')
plt.plot(histogramAfterOwn)

plt.subplot(2, 2, 3)
plt.title("Histogram After OpenCV's Equalization")
plt.plot(histogramAfterCV)

plt.subplot(2, 2, 4)
plt.title('Images')
plt.imshow(np.hstack([image, equalizedImage, equalizedImageCV]), cmap='gray')

plt.show()
