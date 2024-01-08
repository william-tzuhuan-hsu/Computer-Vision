# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # construct the matrix for all the pixels
    W = res[0]
    H = res[1]
    image = np.zeros((H, W))
    cx, cy = W//2+center[0], H//2+center[1]
    # range through all the pixels
    for i in range(H):
        for j in range(W):
            x = (j - cx)*pxSize
            y = (i - cy)*pxSize
            if np.linalg.norm([x, y]) <= rad:
                z = np.sqrt(rad**2-x**2-y**2)
                n = np.array([x, y, z])
                image[i, j] = np.dot(n, light)

    plt.imshow(image, cmap="gray", origin='lower')
    # plt.show()
    plt.savefig("../result/render_circle"+str(np.sum(light*np.sqrt(3)))+".png")

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I = None
    s = None
    L = None
    from PIL import Image
    import skimage.color
    set = False
    # read in all the files
    files = sorted(os.listdir(path))
    for file in files:
        # save the source file 
        if file[-4:] == ".npy":
            L = np.load(path+file)
        else:
            # read in the image
            img = cv2.imread(path+file, cv2.IMREAD_UNCHANGED)
            s = (img.shape[0], img.shape[1])
            xyz_img = skimage.color.rgb2xyz(img)
            # take out the luminance channel
            luminance = xyz_img[:, :, 1].flatten()
            if set == False:
                I = luminance[:, None]
                set = True
            else:
                I = np.concatenate((I, luminance[:, None]), axis=1)


    return I.T, L.T, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # # 
    # print(f"I.shape: {I.shape}")
    # print(f"L.shape: {L.shape}")
    B = np.linalg.pinv(L @ L.T) @ L @ I

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    # since b = a dot n and a is the albedo, we could just calculate the magnitude of each vector
    albedos = np.linalg.norm(B, axis=0)
    # print(f"albedos.shape:{albedos.shape}")
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normalIm = (normals.T).reshape(s[0], s[1], 3)
    # print(normalIm.max()) # 0.9901097872540063
    # print(normalIm.min()) # -0.9999993230497296
    # make the value of normal between 0 and 1 to be able to display with greyscale
    normalIm = (normalIm+1)/2.0
    # print(normalIm.max()) # 0.9950548936270032
    # print(normalIm.min()) # 3.384751351975801e-07
    # normalIm = (normalIm*255).astype(np.uint16)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    n1 = normals[0, :]
    n2 = normals[1, :]
    n3 = normals[2, :]
    partial_x = (n1/n3).reshape(s)
    partial_y = (n2/n3).reshape(s)
    surface = integrateFrankot(partial_x, partial_y)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    # plot the surface according to the shape of surface
    H, W = surface.shape
    x = np.linspace(0, W, num=W).astype(int)
    y = np.linspace(0, H, num=H).astype(int)
    x_grid, y_grid = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(x_grid, y_grid, surface, cmap="coolwarm")
    plt.show()



    pass


if __name__ == '__main__':

    # # Put your main code here
    # # b:
    # center = np.array([0, 0, 0])
    # rad = 0.75 * 1.0e-02 # in m
    # light = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1]])/np.sqrt(3)
    # pxSize = 7.0e-06 # in m
    # res = np.array([3840, 2160])
    # for i in range(3):
    #     renderNDotLSphere(center, rad, light[i,:], pxSize, res)

    # # c:
    I, L, s = loadData("../data/")

    # d. 
    singular_val = np.linalg.svd(I, compute_uv=False)
    # print(singular_val)

    # e.
    B = estimatePseudonormalsCalibrated(I, L)

    # f.
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imshow(albedoIm, cmap="gray")
    plt.savefig("../result/albedoIm.png")
    plt.imshow(normalIm, cmap="rainbow", interpolation='nearest')
    plt.savefig("../result/normalIm.png")

    # g.
    surface = estimateShape(normals, s)
    # print(f"surface.shape: {surface.shape}")
    plotSurface(surface)



    

    
