import numpy as np
import util
import helper
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import math
import scipy.ndimage
import submission


'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''



if __name__ == "__main__":
    
    img1 = plt.imread('../data/im1.png')
    img2 = plt.imread('../data/im2.png')

    temple = np.load("../data/templeCoords.npz")
    x1 = temple['x1']
    y1 = temple['y1']
    # print(x1.shape)
    x2 = np.zeros(len(x1))
    y2 = np.zeros(len(x1))
    # print(x2.shape)
    F_load = np.load("../result/Q2_1.npz")
    F = F_load['F']


    for i in range(len(x1)):
        # print(x1[i], y1[i])
        x2[i], y2[i] = submission.epipolarCorrespondence(img1, img2, F, x1[i][0], y1[i][0])

    # # find 3D points 

    matrix_pair = np.load("../data/intrinsics.npz")
    K1 = matrix_pair['K1']
    K2 = matrix_pair['K2']
    C1 = np.hstack((K1, np.array([[0],[0],[0]])))
    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    ma = np.load("../result/q3_3.npz")
    C2 = ma['C2']
    M2 = ma['M2']

    pts1 = np.hstack((x1, y1))
    print(pts1[0:5])
    pts2 = np.vstack((x2, y2)).T
    # print(x2[0:5])
    # print(y2[0:5])
    print(pts2[0:5])

    w , _ = submission.triangulate(C1, pts1, C2, pts2)
    print(w[0:5])


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(w[:,0], w[:,1], w[:,2])

    plt.show()

    np.savez("../result/q4_2.npz", F=F, M1=M1, M2=M2, C1=C1, C2=C2)


