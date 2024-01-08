import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    # print(img)
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    H = im1.shape[0]
    print("find letters")
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap="Greys")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # find the rows using..RANSAC, counting, clustering, etc.
    # process the data and acquire the y_index
    y_centers = []
    for i in range(len(bboxes)):
        minr, minc, maxr, maxc = bboxes[i]
        y_centers.append((minr+maxr)/2)
        
    # print("clustering the bboxes")
    row_dictionary = {}
    for i in range(len(y_centers)):
        y_center = y_centers[i]

        # if the dictionary is empty, create the first cluster
        if bool(row_dictionary) == False:
            row_dictionary[y_center] = [i]
            continue

        keys = row_dictionary.keys()
        # print(keys)
        found_group = False
        for k in keys:
            # find the center of this group
            tmp_mean = [y_centers[i] for i in row_dictionary[k]]
            center = np.mean(tmp_mean)
            # if the cetner is close, append to this group
            if np.abs(y_center-center) < 0.075*H:
                # print(f"np.abs(y_center-center): {np.abs(y_center-center)}")
                # print(f"0.2*H: {0.2*H}")
                # print("Append to group")
                row_dictionary[k].append(i)
                found_group = True
            # else append to new group
        if found_group == False:
            row_dictionary[y_center] = [i]

    # print(row_dictionary)

    # check each row has been correctly cropped
    # for k, v in row_dictionary.items():
    #     print(k)
    #     plt.imshow(bw, cmap="Greys")
    #     for i in v:
    #         minr, minc, maxr, maxc = bboxes[i]
    #         rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                 fill=False, edgecolor='red', linewidth=2)
    #         plt.gca().add_patch(rect)
    #     plt.show()
        

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    sorted_index = []
    for k, v in row_dictionary.items():
        # get the list x values
        # minr, minc, maxr, maxc = bbox
        tmp_x_list = [bboxes[i][1] for i in v]
        tmp_index = [x for _, x in sorted(zip(tmp_x_list, v))]
        # print(tmp_index)

        for i in range(len(tmp_index)):
            sorted_index.append(tmp_index[i])

     
    data = []
    for i in sorted_index:
        minr, minc, maxr, maxc = bboxes[i]
        tmp_img = bw[minr:maxr, minc:maxc]
        # print(f"tmp_img: {tmp_img}")
        maxlen = int(np.max(tmp_img.shape)*1.5)
        # maxlen = int(np.max(tmp_img.shape))
        h_delta = maxlen-tmp_img.shape[0]
        w_delta = maxlen-tmp_img.shape[1]
        padding = ((h_delta//2, h_delta-h_delta//2), (w_delta//2, w_delta-w_delta//2))
        padded = np.pad(tmp_img, padding, mode='constant',constant_values=False)
        # print(f"padded: {padded}")
        dilated = skimage.morphology.dilation(padded, skimage.morphology.square(20))
        # print(f"dilated: {dilated}")
        rescaled = skimage.transform.rescale(dilated, 32/maxlen)
        rescaled = rescaled*1.0
        # print(f"rescaled: {rescaled}")
        # print(f"Find one: {np.where(rescaled==1)}")
        invert = skimage.util.invert(rescaled, signed_float=False)
        # print(invert)
        eroded = skimage.morphology.dilation(invert, skimage.morphology.square(1))
        # converting back to integer array
        # show the image
        # for i in range(invert.shape[0]):
        #     print(invert[i,:])
        if i < 5:
            plt.imshow(eroded, cmap="Greys")
            plt.show()
        # print(inted)
        data.append(eroded.T.flatten())
        

    data = np.array(data)
    
    print(f"data.shape: {data.shape}")
    # load the weights
    # run the crops through your neural network and print them out
    # import pickle
    # import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # params = pickle.load(open('q3_weights.pickle','rb'))

    ######################## try loading different models ###########################################
    # import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # import pickle
    # for model in os.listdir('../model'):
    #     params = pickle.load(open("../model/"+model,'rb'))


    # # a1 = forward(data, params, 'layer1', sigmoid)
    # # a2 = forward(a1, params, 'output', softmax)
    # # # loss
    # # result = np.argmax(a2, axis=1)
    # # check validation accuracy
    #     a1 = forward(data, params, 'layer1', sigmoid)
    #     a2 = forward(a1, params, 'output', softmax)
    #     # loss
    #     print(f"a2.shape: {a2.shape}")
    #     result = np.argmax(a2, axis=1)
    #     print(f"result: {result}")
    #     print(len(result))

    #     # plt.imshow(rescaled, cmap="Greys")
    #     # plt.show()
    #     growing_string = ""
    #     for i in range(len(letters[result])):
    #         growing_string += str(letters[result][i])
    #     print(growing_string)

    ########################### normal inference ###############################

    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    import pickle
    params = pickle.load(open("q3_weights.pickle",'rb'))


    # a1 = forward(data, params, 'layer1', sigmoid)
    # a2 = forward(a1, params, 'output', softmax)
    # # loss
    # result = np.argmax(a2, axis=1)
    # check validation accuracy
    a1 = forward(data, params, 'layer1', sigmoid)
    a2 = forward(a1, params, 'output', softmax)
    # loss
    print(f"a2.shape: {a2.shape}")
    result = np.argmax(a2, axis=1)
    print(f"result: {result}")
    print(len(result))

    # plt.imshow(rescaled, cmap="Greys")
    # plt.show()
    growing_string = ""
    for i in range(len(letters[result])):
        growing_string += str(letters[result][i])
    print(growing_string)