import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from planarH import computeH, computeH_norm, computeH_ransac, construct_A, compositeH
from matchPics import matchPics
import math



#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# compute the matches 
matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

# use the matched points to compute ransac
x1 = locs1[[i[0] for i in matches]]
x2 = locs2[[i[1] for i in matches]]

# computeH_norm(x1, x2)

bestH, inliers = computeH_ransac(x1, x2, opts)

# print(bestH)

temp = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
harry = compositeH(bestH, temp, cv_desk)
cv2.imshow('harry.jpg', harry)
cv2.waitKey(0)
