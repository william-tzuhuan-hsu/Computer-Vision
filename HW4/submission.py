"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import helper
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import math
import scipy.ndimage


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # treat pts1 as left
    # treat pts2 as right
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)
    M = float(M)

    # Pr.T * F * Pl = 0
    # Replace pass by your implementation
    # pass
    # normalize the points in pts1 and pts2
    pts1 = pts1/M
    pts2 = pts2/M
    
    # compute F using the normalized points
    A = construct_A(pts1, pts2)
    # print(A)
    # solve for A by SVD and obtain unormalized F
    _, _, Vh = np.linalg.svd(A)
    eig_vec = Vh[-1,:]

    F_unormalized = eig_vec.reshape(3, 3)
    # print(F_unormalized)
    
    # local minimization
    F_unormalized = util.refineF(F_unormalized, pts1, pts2)

    # un-normalize the matrix by calculating (T.T) @ F @ T
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = T.T @ F_unormalized @ T

    return F


def construct_A(pts1, pts2):
    N = len(pts1)
    xl, yl = pts1[:,0], pts1[:,1]
    xr, yr = pts2[:,0], pts2[:,1]
    A = np.zeros((N, 9))
    A[:, 0] = xr*xl
    A[:, 1] = xr*yl
    A[:, 2] = xr
    A[:, 3] = yr*xl
    A[:, 4] = yr*yl
    A[:, 5] = yr
    A[:, 6] = xl
    A[:, 7] = yl
    A[:, 8] = 1
    # print(A.shape)

    return A


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)

    xl, yl = pts1[:,0], pts1[:,1]
    xr, yr = pts2[:,0], pts2[:,1]
    N = len(pts1)

    # print(C1)

    cross_l = np.array([[np.zeros(N), np.repeat(-1, N), yl], 
                        [np.repeat(1, N), np.zeros(N), -xl], 
                         [-yl, xl, np.zeros(N)]])
    cross_r = np.array([[np.zeros(N), np.repeat(-1, N), yr], 
                         [np.repeat(1, N), np.zeros(N), -xr],
                         [-yr, xr, np.zeros(N)]])
    # print(cross_l.shape)
    # print(cross_l[:,:,0])

    Pl = np.einsum("ijk, jl -> ilk", cross_l, C1)
    Pr = np.einsum("ijk, jl -> ilk", cross_r, C2)
    # print(Pl.shape)

    A_n = np.vstack((Pl[:2,:,:], Pr[:2,:,:]))
    # print("A:")
    # print(A_n[:,:,0])
    # print(A_n.shape)

    # for each point we solve for wi using svd
    W_n = []
    W_nonhomo = []
    for i in range(N):
        A = A_n[:,:,i]
        _, _, Vh = np.linalg.svd(A)
        p = Vh[-1,:]
        P = p[:3]/p[3]
        W_n.append(p)
        W_nonhomo.append(P)

    # calculate the error 
    # project all the points onto 2D homogeneous coordinate
    W_n = np.array(W_n)
    W_nonhomo = np.array(W_nonhomo)
    # print(W_n.shape)
    # print(C1.shape)
    print("Shape of w nonhomo: {homo}".format(homo=W_nonhomo.shape))

    pts1_calc = (C1 @ W_n.T)
    pts2_calc = (C2 @ W_n.T)
    
    pts1_calc = pts1_calc[:-1]/pts1_calc[-1]
    pts2_calc = pts2_calc[:-1]/pts2_calc[-1]
    # print(pts1_calc.shape)
    # print(pts1[:10])
    # print(pts1_calc[:10])

    error = np.square(np.linalg.norm(pts1 - pts1_calc.T))
    error += np.square(np.linalg.norm(pts2 - pts2_calc.T))
    print(error)

    return W_nonhomo, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    
    H = im2.shape[0]
    W = im2.shape[1]
    pl = np.array([x1, y1, 1])
    # make splines for im2
    spline_list = []
    for i in range(3):
        im2_spline = RectBivariateSpline([i for i in range(H)], [j for j in range(W)], im2[:,:,i])
        spline_list.append(im2_spline)
    # calculate epipolar lines in image2
    line_vect = F @ pl
    # print(line_vect)

    # calculate the range on the epipolar line that we're calculating 
    # d = (pl @ F @ pl.T) / math.sqrt(2 * np.square(np.norm(line_vect)))
    # find the point on the line within 2D 
    d = 80
    y_low, y_high = y1-d/2, y1+d/2
    y_centers = np.linspace(y_low, y_high, endpoint=True, num=int(y_high-y_low+1))
    x_centers = findX(line_vect, y_centers)
    # print(x_centers)
    # print(x1)

    # make the patches from image 1 and image 2
    window_size = 10
    # image 1:
    patch_im1 = im1[int(y1-window_size/2):int(y1+window_size/2), int(x1-window_size/2):int(x1+window_size/2)]
    best_error = np.inf
    best_idx = 0

    for i in range(len(y_centers)):
        # image 2:
        # construct the grid around x center and y center
        x_coord = np.linspace(x_centers[i]-window_size/2, x_centers[i]+window_size/2, endpoint=False, num=window_size)
        y_coord = np.linspace(y_centers[i]-window_size/2, y_centers[i]+window_size/2, endpoint=False, num=window_size)
        x_grid, y_grid = np.meshgrid(x_coord, y_coord)
        # print("y_coord: {coord}".format(coord = y_coord))
        # print("x_coord: {coord}".format(coord = x_coord))
        error = 0
        # get the patch on every channel
        for j in range(3):
            patch_im2 = spline_list[j].ev(y_grid, x_grid)
            # print(patch_im2.shape)
            # calculate error and increment
            diff_matrix = patch_im1[:,:,j]-patch_im2
            masked_diff = scipy.ndimage.filters.gaussian_filter(diff_matrix, 2)
            # print(np.square(np.linalg.norm(masked_diff)))
            error += np.square(np.linalg.norm(masked_diff))
        # print(error)
        # if the error is lower than the best error update the error and index
        # print(error)
        if error < best_error:
            # print("Best error:{best_error}".format(best_error=best_error))
            # print("Update error")
            best_error = error
            best_idx = i
        
            # return the point with the lowest norm

    return int(x_centers[best_idx]), int(y_centers[best_idx])


def findX(line_vect, y_centers):
    a, b, c = line_vect
    x_list = []
    
    for i in range(len(y_centers)):
        x = (-b*y_centers[i]-c)/a
        x_list.append(x)
    
    return np.array(x_list)

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


if __name__ == "__main__":

    # Q2.1
    pair = np.load("../data/some_corresp.npz")
    # print(pair.files)
    pts1 = pair['pts1']
    pts2 = pair['pts2']
    # shape = (110, 2)
    img1 = plt.imread('../data/im1.png')
    img2 = plt.imread('../data/im2.png')
    # shape = (480, 640, 3)
    M = img1.shape[1]
    # construct_A(pts1, pts2)
    F = eightpoint(pts1, pts2, M)
    # print(F)
    np.savez("../result/q2_1.npz", F=F, M=M)
    helper.displayEpipolarF(img1, img2, F)

    # Q3.1
    matrix_pair = np.load("../data/intrinsics.npz")
    K1 = matrix_pair['K1']
    K2 = matrix_pair['K2']
    E = essentialMatrix(F, K1, K2)
    np.savez("../result/q3_1.npz", E=E)

    # Q4.1
    # epipolarCorrespondence(img1, img2, F, 240, 320)
    # helper.epipolarMatchGUI(img1, img2, F)
    np.savez("../result/q4_1.npz", F=F, pts1=pts1, pts2=pts2)

    # Q4.2


