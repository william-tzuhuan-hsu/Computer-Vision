import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import copy

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
#     """
#     :param It: template image
#     :param It1: Current image
#     :param rect: Current position of the car (top left, bot right coordinates)
#     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
#     :param num_iters: number of iterations of the optimization
#     :param p0: Initial movement vector [dp_x0, dp_y0]
#     :return: p: movement vector [dp_x, dp_y]
#     """

    # put your implementation here
    p = copy.deepcopy(p0)
    # initializing p0 and get all the parameters
    H = It.shape[0]
    W = It.shape[1]
    n = p.shape[0]

    # make the spline for It and It1
    It_spline = RectBivariateSpline([i for i in range(H)], [j for j in range(W)], It)
    It1_spline = RectBivariateSpline([i for i in range(H)], [j for j in range(W)], It1)

    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    x_coord = np.linspace(x1, x2, endpoint=True, num=int(x2-x1+1))
    y_coord = np.linspace(y1, y2, endpoint=True, num=int(y2-y1+1))
    x_grid, y_grid = np.meshgrid(x_coord, y_coord)

    for i in range(int(num_iters)):
        print(i)

        # 1.Warp the current image It using p to compute I(W(x,p))
        x_grid_w = x_grid+p[0]
        y_grid_w = y_grid+p[1]

        # evaulate the warped points
        Iw = It1_spline.ev(y_grid_w, x_grid_w)
        Tx = It_spline.ev(y_grid, x_grid)
        # construct b
        b = (Tx-Iw).flatten()
        # print(b.shape)

        N = b.shape[0]
        # print(N)
        # print(N)
        # compute for matrix A
        # compute gradient
        It1_delta_x = It1_spline.ev(y_grid_w, x_grid_w, dx=0, dy=1).flatten()
        It1_delta_y = It1_spline.ev(y_grid_w, x_grid_w, dx=1, dy=0).flatten()
        # print(mask.shape)
        A = np.zeros((N, n))
        # print(A.shape)

        for j in range(N):
            # A_temp = np.array([It1_delta_x[j]*x_grid[j], It1_delta_y[j]*x_grid[j], It1_delta_x[j]*y_grid[j], It1_delta_y[j]*y_grid[j], It1_delta_x[j], It1_delta_y[j]])
            A_temp = np.array([It1_delta_x[j], It1_delta_y[j]])
            A[j, :] = A_temp

        # print(A[0])
        # print(mask.shape)

        
        delta_p = np.linalg.inv(A.T @ A) @ A.T @ b
        # print(np.square(delta_p).sum()) 
        if np.square(delta_p).sum() < threshold:
            print("Aborting")
            print(np.square(delta_p).sum()) 
            return p

        p += delta_p

    return p
