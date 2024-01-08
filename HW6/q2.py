# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
import matplotlib.pyplot as plt

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    U, S, Vh = np.linalg.svd(I, full_matrices=False)
    singular = np.zeros(S.shape)
    singular[:3] = S[:3]

    # print(f"U.shape: {U.shape}")
    # print(f"S.shape: {S.shape}")
    # print(f"Vh.shape: {Vh.shape}")

    I_hat = U @ singular @ Vh
    
    B = Vh[:3, :]
    L = U[:3, :]

    return B, L


if __name__ == "__main__":

    # Put your main code here
    # a. 
    I, L, s = loadData("../data/")
    print(f"L: {L}")

    # b.
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)
    print(f"L_hat: {L_hat}")

    # c.
    albedos, normals = estimateAlbedosNormals(B_hat)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imshow(albedoIm, cmap="gray")
    plt.savefig("../result/albedoIm_hat.png")
    plt.imshow(normalIm, cmap="rainbow", interpolation='nearest')
    plt.savefig("../result/normalIm_hat.png")

    # d.
    # surface = estimateShape(normals, s)
    # plotSurface(surface)

    # e.
    integrity_normal = enforceIntegrability(normals, s)
    # surface = estimateShape(integrity_normal, s)
    # plotSurface(surface)

    # f.
    miu = 0
    v = 100
    lam = 1

    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [miu, v, lam]])
    surface = estimateShape((np.linalg.inv(G)).T @ integrity_normal, s)
    plotSurface(surface)