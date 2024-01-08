import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # initializing p0 and get all the parameters
    H = It.shape[0]
    W = It.shape[1]
    n = 6

    # make the spline for It and It1
    It1_spline = RectBivariateSpline([i for i in range(H)], [j for j in range(W)], It1)


    p = M.flatten()
    print(p)
    x1, y1, x2, y2 = 0, 0, W, H
    x_coord = np.linspace(x1, x2, endpoint=False, num=W)
    y_coord = np.linspace(y1, y2, endpoint=False, num=H)

    for i in range(int(num_iters)):
        # print(i)

        x_grid, y_grid = np.meshgrid(x_coord, y_coord)
        # 1.Warp the current image It using p to compute I(W(x,p))
        x_grid_w = x_grid*p[0] + y_grid*p[1] + p[2]
        y_grid_w = x_grid*p[3] + y_grid*p[4] + p[5]
        # print(x_grid_w>=0)
        # before evaluating we have to find the points lie inside of the image
        mask = (x_grid_w>=0) & (x_grid_w<W) & (y_grid_w>=0) & (y_grid_w<H)
        # print(np.unique(mask))
        # update the grid
        x_grid = x_grid[mask]
        y_grid = y_grid[mask]

        x_grid_w = x_grid_w[mask]
        y_grid_w = y_grid_w[mask]

        # evaulate the warped points
        Iw = It1_spline.ev(y_grid_w, x_grid_w)

        # construct b
        b = It[mask].flatten()-Iw

        N = x_grid_w.shape[0]
        # print(N)
        # compute for matrix A
        # compute gradient
        It1_delta_x = It1_spline.ev(y_grid_w, x_grid_w, dx=0, dy=1).flatten()
        It1_delta_y = It1_spline.ev(y_grid_w, x_grid_w, dx=1, dy=0).flatten()
        # print(mask.shape)
        A = np.zeros((N, n))
        # print(A.shape)

        # for x, y in zip(x_grid, y_grid):
        #     x = int(x)
        #     y = int(y)
        #     idx = y*W + x
        #     # print(idx)
        #     A_temp = np.array([It1_delta_x[idx]*x, It1_delta_y[idx]*x, It1_delta_x[idx]*y, It1_delta_y[idx]*y, It1_delta_x[idx], It1_delta_y[idx]])
        #     A[idx, :] = A_temp
        for j in range(N):
            # A_temp = np.array([It1_delta_x[j]*x_grid[j], It1_delta_y[j]*x_grid[j], It1_delta_x[j]*y_grid[j], It1_delta_y[j]*y_grid[j], It1_delta_x[j], It1_delta_y[j]])
            A_temp = np.array([It1_delta_x[j]*x_grid[j], It1_delta_x[j]*y_grid[j], It1_delta_x[j], It1_delta_y[j]*x_grid[j], It1_delta_y[j]*y_grid[j], It1_delta_y[j]])
            A[j, :] = A_temp

        # print(A[0])
        # print(mask.shape)

        
        delta_p = np.linalg.inv(A.T @ A) @ A.T @ b
        print(np.square(delta_p).sum())
        if np.sqrt(np.square(delta_p).sum()) < threshold:
            print("Aborting")
            return p.reshape(2, 3)

        p += delta_p

    p.reshape(2, 3)
    return M
