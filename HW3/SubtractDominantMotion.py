import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_erosion, binary_dilation

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    Iw = affine_transform(image1, M)
    diff = np.abs(Iw-image2)
    mask = diff > tolerance
    # mask = binary_erosion(mask)
    mask = binary_dilation(mask)

    return mask
