import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # initializing p0 and get all the parameters
    H = It.shape[0]
    W = It.shape[1]
    N = H*W
    n = 6

    # make the spline for It and It1
    It_spline = RectBivariateSpline([i for i in range(H)], [j for j in range(W)], It)


    p = M.flatten()
    print(p)
    x1, y1, x2, y2 = 0, 0, W, H
    x_coord = np.linspace(x1, x2, endpoint=False, num=W)
    y_coord = np.linspace(y1, y2, endpoint=False, num=H)
    x_grid, y_grid = np.meshgrid(x_coord, y_coord)
    x_point, y_point = x_grid.flatten(), y_grid.flatten()

    # compute gradient
    It_delta_x = It_spline.ev(y_grid, x_grid, dx=0, dy=1).flatten()
    It_delta_y = It_spline.ev(y_grid, x_grid, dx=1, dy=0).flatten()
    # print(mask.shape)
    A = np.zeros((N, n))
    # print(A.shape)

    for j in range(N):
        # A_temp = np.array([It1_delta_x[j]*x_grid[j], It1_delta_y[j]*x_grid[j], It1_delta_x[j]*y_grid[j], It1_delta_y[j]*y_grid[j], It1_delta_x[j], It1_delta_y[j]])
        A_temp = np.array([It_delta_x[j]*x_point[j], It_delta_x[j]*y_point[j], It_delta_x[j], It_delta_y[j]*x_point[j], It_delta_y[j]*y_point[j], It_delta_y[j]])
        A[j, :] = A_temp

    for i in range(int(num_iters)):
        # print(i)

        # 1.Warp the current image It1 back using p to compute I(W(x,p))
        x_grid_w = x_grid*p[0] + y_grid*p[1] + p[2]
        y_grid_w = x_grid*p[3] + y_grid*p[4] + p[5]
        # print(x_grid_w>=0)
        # before evaluating we have to find the points lie inside of the image
        mask = (x_grid_w>=0) & (x_grid_w<W) & (y_grid_w>=0) & (y_grid_w<H)

        # update the grid
        x_grid_w = x_grid_w[mask]
        y_grid_w = y_grid_w[mask]

        N_iter = x_grid_w.shape[0]

        # evaulate the warped points
        Iw = It_spline.ev(y_grid_w, x_grid_w)

        # construct b
        b = Iw-It[mask].flatten()

        mask = mask.flatten()
        # mask A as well
        A_iter = A[mask].reshape((N_iter, n))

        # print(N)
        # compute for matrix A

        # for x, y in zip(x_grid, y_grid):
        #     x = int(x)
        #     y = int(y)
        #     idx = y*W + x
        #     # print(idx)
        #     A_temp = np.array([It1_delta_x[idx]*x, It1_delta_y[idx]*x, It1_delta_x[idx]*y, It1_delta_y[idx]*y, It1_delta_x[idx], It1_delta_y[idx]])
        #     A[idx, :] = A_temp


        # print(A[0])
        # print(mask.shape)

        
        delta_p = np.linalg.inv(A_iter.T @ A_iter) @ A_iter.T @ b
        print(delta_p)
        if np.sqrt(np.square(delta_p).sum()) < threshold:
            print("Aborting")
            return p.reshape(2, 3)

        delta_M = np.array([[1+delta_p[0], delta_p[1], delta_p[2]],
                            delta_p[3], 1+delta_p[4], delta_p[5]])
        delta_M = np.concatenate(delta_M, [0, 0, 1], axis=0)
        M = p.reshape(2, 3)
        M = np.concatenate(M, [0, 0, 1], axis=0)
        M = M * np.linalg.pinv(delta_M)
        p = M[:, :2].flatten()

    M = p.reshape(2, 3)
    return M

