import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from planarH import computeH_ransac, compositeH
from matchPics import matchPics
from opts import get_opts
# from numpy.linalg import LinAlgError

def CaptureImage(img, ratio):
    # calculate the correct dimension for the image
    H = img.shape[0]
    W = int(H/ratio)
    # print("Height")
    # print(H)
    # print("Width")
    # print(W)

    # crop the image from the original frame
    return img[50:320, int(img.shape[1]/2-W/2):int(img.shape[1]/2+W/2)]

opts = get_opts()

#Write script for Q3.1
print("Importing AR video")
ar_source = loadVid("../data/ar_source.mov")

print("Importing book video")
book = loadVid("../data/book.mov")

cv_cover = cv2.imread('../data/cv_cover.jpg')
cover_ratio = cv_cover.shape[0]/cv_cover.shape[1]

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# make the video to write to
# video = cv2.VideoWriter('Panda.avi', fourcc, 30, (book.shape[2], book.shape[1]))
video = cv2.VideoWriter('Panda.avi', fourcc, 1, (book.shape[2], book.shape[1]))

# range through all the frames in the video
for i in range(len(book)):
    print("Frame number: {frame}".format(frame=i))
    ar_frame = ar_source[i]
    book_frame = book[i]

    # crop the frame from ar_source
    cropped_image = CaptureImage(ar_frame, cover_ratio)

    # Compute Ransac of the two image
    # compute the matches 
    matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)

    # use the matched points to compute ransac
    x1 = locs1[[i[0] for i in matches]]
    x2 = locs2[[i[1] for i in matches]]

    # computeH_norm(x1, x2)
    # handling the case where we can't find homography
    try:
        print("In trying loop")
        bestH, inliers = computeH_ransac(x1, x2, opts)

    except np.linalg.LinAlgError as err:
        # if "converge" in str(err):
        print("Not converging")
        bestH = np.eye(3)
        # else:
        #     raise err

    # bestH, inliers = computeH_ransac(x1, x2, opts)
    temp = cv2.resize(cropped_image, (cv_cover.shape[1], cv_cover.shape[0]))
    harry = compositeH(bestH, temp, book_frame)
    cv2.imshow('Test', harry)
    # cv2.waitKey(0)
    video.write(harry)



