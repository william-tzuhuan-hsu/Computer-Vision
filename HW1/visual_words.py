import os, multiprocessing
from os.path import join, isfile

import random as rand
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # ----- TODO -----
    # check if the image have three channels
    if img.shape[-1] != 3:
        img = np.stack((img,)*3, axis=-1)

    # check if the image is within the range of [0, 1]
    if np.max(img) > 1.0:
        #normalize the image
        img = img / np.max(img)
    
    # convert image to lab color space
    img = skimage.color.rgb2lab(img)

    response_list = []
    # range through different scales
    for scale in filter_scales:
        # gaussian filter:
        # range through three channels
        for c in range(3):
            single_channel = img[:,:,c]
            # convolve the image with filter
            response = scipy.ndimage.gaussian_filter(single_channel, sigma=scale)
            # append to the response list
            response_list.append(response)

        # laplacian of gaussian
        for c in range(3):
            single_channel = img[:,:,c]
            # convolve the image with filter
            response = scipy.ndimage.gaussian_laplace(single_channel, sigma=scale)
            # append to the response list
            response_list.append(response)

        # derivative of x direction using sobel operator
        for c in range(3):
            # # sobel x direction 
            # k_sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # k_sob = k_sob*scale
            single_channel = img[:,:,c]
            # convolve the image with filter
            response = scipy.ndimage.gaussian_filter1d(single_channel, axis=1, sigma=scale)
            # append to the response list
            response_list.append(response)
        
        # derivative of y direction
        for c in range(3):
            # # sobel x direction 
            # k_sob = np.array([[1, 2, 1], [0, 0, 0], [-1, -2,-1]])
            # k_sob = k_sob*scale
            single_channel = img[:,:,c]
            # convolve the image with filter
            response = scipy.ndimage.gaussian_filter1d(single_channel, axis=0, sigma=scale)
            # append to the response list
            response_list.append(response)

    images = np.concatenate([i[:,:,np.newaxis] for i in response_list], axis=2)

    return images
     
#  you should read an image, extract the responses, and save to a temporary file.
def compute_dictionary_one_image(opts, file_name, img, alpha, F):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    ''' 
    # ----- TODO -----
    out_dir = opts.out_dir
    
    # extract the response of the filter and sample from it
    responses = extract_filter_responses(opts, img)

    # creat a list of random points for sampling
    x_index = [rand.randint(0,responses.shape[0]-1) for _ in range(alpha)]
    y_index = [rand.randint(0,responses.shape[1]-1) for _ in range(alpha)]

    # sample from the response
    sampled_response = []
    for i in range(alpha):
        sampled_response.append(responses[x_index[i], y_index[i],:])
    
    # concatenate all the data points into a 3D matrix (alpha X 3F)
    sampled_response = np.array(sampled_response)
    # sampled_response.shape = (100, 48)

    # write the result to a npy file
    file_name = join(out_dir, file_name.replace('.jpg', '.npy'))
    # make the directory to store response if it doesn't exist
    dir_name = file_name.split('/')[-2]
    # print(dir_name)
    os.makedirs(join(out_dir, dir_name), exist_ok=True)
    # save the reponses to a temporary file
    with open(file_name, 'wb') as f:
        np.save(f, sampled_response)
    
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    F = 4*len(opts.filter_scales)

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    # print(train_files)

    # create filter respones for all the files
    # print("Computing responses.")
    for img_name in train_files:
        img_path = join(opts.data_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        compute_dictionary_one_image(opts, img_name, img, alpha, F)

    # train for k means clustering and get the centers
    # print("Read in responses.")
    responses = []
    # read in the sampled response and concatenate it into a alpha*F X 3F matrix
    response_files = open((join(data_dir, 'train_files.txt'))).read().splitlines()
    for response_name in response_files:
        response_name = join(out_dir, response_name.replace('.jpg', '.npy'))
        # read in the saved response
        with open(response_name, 'rb') as f:
            response = np.load(f)
            responses.append(response)
    
    matrix = np.concatenate(responses, axis=0)

    # start k means clustering
    # print("Running K means!")
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(matrix)
    # print("Retrieving centers")
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    pass 

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    # # read in the dictionary
    # with open('dictionary.npy', 'rb') as f:
    #         dictionary = np.load(f)

    # get the dimension of the image
    H = img.shape[0]
    W = img.shape[1]
    # get the dimension of the centers
    K = opts.K

    # get response
    response = extract_filter_responses(opts, img)
    response = response.reshape((H*W, response.shape[2]))
    # make the new image
    wordmap = np.zeros(H*W)

    # compute the distance of the each pixel to all the centers
    dist_matrix = scipy.spatial.distance.cdist(response, dictionary, 'euclidean')

    # range through all the pixels and assign it to the correct
    for i in range(H*W):
        # set the shortest distance
        closest_dist = dist_matrix[i][0]
        # set the index
        close_idx = 0
        # range through k to check the distance and update the shortest
        for j in range(K):
            if dist_matrix[i][j] < closest_dist:
                closest_dist = dist_matrix[i][j]
                close_idx = j

        # assign K to that pixel
        wordmap[i] = close_idx
    # reshape wordmap
    wordmap = wordmap.reshape(H, W)
    
    return wordmap

