import helper
import numpy as np
import submission

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

if __name__ == "__main__":
    # 1.
    correspondence = np.load("../data/some_corresp.npz")
    pts1 = correspondence['pts1']
    pts2 = correspondence['pts2']

    E_load =  np.load("../result/q3_1.npz")
    E = E_load['E']
    M2s = helper.camera2(E)
    # print("E:")
    # print(E)

    for i in range(M2s.shape[-1]):
        M2 = M2s[:,:,i]
        print(M2)

    matrix_pair = np.load("../data/intrinsics.npz")
    K1 = matrix_pair['K1']
    K2 = matrix_pair['K2']

    # calculate error for each M2 and pick the right one
    # calculate C1
    C1 = np.hstack((K1, np.array([[0],[0],[0]])))
    # calculate M2 and C2
    best_M2 = M2s[:,:,0]
    best_error = 100000000
    best_3D = None
    best_C2 = C1

    for i in range(M2s.shape[-1]):
        M2 = M2s[:,:,i]
        C2 = K2 @ M2
        P, error = submission.triangulate(C1, pts1, C2, pts2)
        # print("breaking")

        # break
        # print(error)
        print(np.min(P[:, 2]))
        print(np.min(P[:, 2]).dtype)
        if error < best_error and np.min(P[:, 2]) >= 0:

            best_error = error
            best_M2 = M2
            best_3D = P
            best_C2 = C2
        
    # 3. Save the correct M2, C2, and P to q3_3.npz
    np.savez("../result/q3_3.npz", M2=best_M2, C2=best_C2, P=best_3D)
