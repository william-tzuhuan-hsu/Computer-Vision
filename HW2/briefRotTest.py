import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import skimage.color
import scipy.ndimage
import matplotlib.pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
# img_gray = skimage.color.rgb2gray(img)

hist = []

for i in range(36):
	
	print("Compute angle: {angle}".format(angle=i*10))
	#Rotate Image
	rot_img = scipy.ndimage.rotate(img, angle=i*10)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, rot_img, opts)
	print(len(matches))

	#Update histogram
	hist.append(len(matches))
	# pass # comment out when code is ready

idx = [i*10 for i in range(36)]
#Display histogram
plt.hist(idx, 36, weights=hist)
plt.show()
