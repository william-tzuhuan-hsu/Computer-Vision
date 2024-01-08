import numpy as np
import cv2
import math


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	# first swap the x and y in x1 and x2
	# x1 = [(x1[i][1], x1[i][0]) for i in range(len(x1))]
	# x2 = [(x2[i][1], x2[i][0]) for i in range(len(x2))]
	# x1[:, [0, 1]] = x1[:, [1, 0]]
	# x2[:, [0, 1]] = x2[:, [1, 0]]
	
	# construct A
	A = construct_A(x1, x2)
	# print("A has shape of {shape}".format(shape=A.shape))
	U, S, Vh = np.linalg.svd(A)
	# eig = math.sqrt(S[9][9])
	# print("U has shape of {shape}".format(shape=U.shape))
	# print("S has shape of {shape}".format(shape=S.shape))
	# print("Vh has shape of {shape}".format(shape=Vh.shape))
	
	# handle the case where eigen value can't be calculated
	# try:
	# 	U, S, Vh = np.linalg.svd(A)

	# except np.linalg.LinAlgError as err:
	# 	if "converge" in str(err):
	# 		print("Not converging")
	# 		return np.eye(3)
	# 	else:
	# 		raise err
	
	# print("ATA")
	# print("EIGVAL")
	# print(eigval_ata)
	# print("EIGVEC")
	# print(eigvec_ata[:,0])
	# print("S")
	# print(S)
	# print("Vh")
	# print(Vh.T[:,8])
	eig_vec = Vh[-1,:]
	# eig_vec[3], eig_vec[4] = eig_vec[4], eig_vec[3]
	# solution is sigma9 and column9 in matrix Vh
	# print(eig_vec)
	return eig_vec.reshape((3, 3))
	

	
	

def construct_A(x1, x2):
	N = len(x1)
	u_row = np.array([[x2[i][0], x2[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]] for i in range(N)])
	v_row = np.array([[0, 0, 0, x2[i][0], x2[i][1], 1, -x2[i][0]*x1[i][1], -x2[i][1]*x1[i][1], -x1[i][1]] for i in range(N)])

	A = np.zeros((2*N, 9), dtype=u_row.dtype)
	A[::2] = u_row
	A[1::2] = v_row
	
	return A

def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1_centroid = np.mean(x1, axis=0)
	x2_centroid = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1_cent = x1-x1_centroid
	x2_cent = x2-x2_centroid

	# print("Check whether the mean centering is correct.")
	# print(x1_cent.mean(axis=0))
	# print(x2_cent.mean(axis=0))

	#Calculate the scaling for x1 and x2
	x1_scale = math.sqrt(2)/np.max(np.linalg.norm(x1_cent, axis=1))
	x2_scale = math.sqrt(2)/np.max(np.linalg.norm(x2_cent, axis=1))

	# # add homogeneous coordinate
	# ones = np.ones(x1.shape[0])
	# ones = ones[..., np.newaxis]
	# x1_norm = np.hstack((x1_norm, ones))
	# x2_norm = np.hstack((x2_norm, ones))

	#Similarity transform 1
	translation_x1 = np.array([[1, 0, -x1_centroid[0]], 
							[0, 1, -x1_centroid[1]],
							[0, 0, 1]]) 
	scaling_x1 = np.array([[x1_scale, 0, 0], 
						[0, x1_scale, 0], 
						[0, 0, 1]])
	T1 = scaling_x1 @ translation_x1

	# applying transformation
	x1_norm = (T1 @ np.hstack((x1, np.ones(x1.shape[0])[..., np.newaxis])).T).T

	# print(np.max(np.linalg.norm(x1_norm, axis=1)))
	# #Similarity transform 2
	translation_x2 = np.array([[1, 0, -x2_centroid[0]], 
							[0, 1, -x2_centroid[1]], 
							[0, 0, 1]]) 
	scaling_x2 = np.array([[x2_scale, 0, 0], 
						[0, x2_scale, 0], 
						[0, 0, 1]])
	T2 = scaling_x2 @ translation_x2
	x2_norm = (T2 @ np.hstack((x2, np.ones(x2.shape[0])[..., np.newaxis])).T).T
	# print((x1_norm.T @ x1 @ np.linalg.inv(x1.T@x1)).shape)
	# T1 = (x1_norm.T @ x1 @ np.linalg.inv(x1.T@x1))
	# T2 = (x1_norm.T @ x1 @ np.linalg.inv(x1.T@x1))

	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H = np.linalg.inv(T1) @ H_norm @ T2

	return H




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	x1 = locs1[:, [1, 0]]
	x2 = locs2[:, [1, 0]]

	# # add homogeneous coordinate
	x1_h = np.hstack((x1, np.ones(x1.shape[0])[..., np.newaxis]))
	x2_h = np.hstack((x2, np.ones(x2.shape[0])[..., np.newaxis]))

	num_inliers = 0
	bestH = np.zeros((3, 3))
	inliers = np.zeros(len(locs1))
	# # H = computeH(x1, x1)
	# # print(H)

	for i in range(max_iters):
		# randomly select 4 points to compute homography
		idx = np.random.choice(np.arange(x1.shape[0]), 4)
		x1_s = x1[idx]
		# print("x1_s")
		# print(x1_s)
		x2_s = x2[idx]

		H_temp = computeH_norm(x1_s, x2_s)
		# print(H_temp)
		# apply the Homography
		x2_homo = (H_temp @ x2_h.T).T
		# convert back to nonhomogeneous coordinate
		x2_noh = x2_homo/x2_homo[:,-1,np.newaxis]
		# print(i)
		# print(x1_s)
		# test = (H_temp @ np.hstack((x2_s, np.ones(x2_s.shape[0])[..., np.newaxis])).T).T
		# print("Matrix after warping")
		# print(test/test[:,-1,np.newaxis])

		# calculate the distance of each point matches
		dist_matrix = np.sqrt(np.square(x2_noh[:,0]-x1[:,0])+np.square(x2_noh[:,1]-x1[:,1]))
		inliers_bool = dist_matrix < inlier_tol
		curr_inliers = inliers_bool.sum()

		if curr_inliers > num_inliers:
			num_inliers = curr_inliers
			bestH = H_temp
			inliers = inliers_bool.astype(int)

	# print(inliers)
	return bestH, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H1to2 = np.linalg.inv(H2to1)
	#Create mask of same size as template
	mask = np.ones(template.shape)
	#Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(mask, H1to2, (img.shape[1], img.shape[0]))
	# print(warped_mask.shape)
	#Warp template by appropriate homography
	warped_temp = cv2.warpPerspective(template, H1to2, (img.shape[1], img.shape[0]))
	# print(warped_temp.shape)
	#Use mask to combine the warped template and the image
	img[warped_mask!=0] = warped_temp[warped_mask!=0]

	return img

