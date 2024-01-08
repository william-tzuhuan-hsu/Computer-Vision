import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    min_val = -np.sqrt(6/(in_size+out_size))
    max_val = np.sqrt(6/(in_size+out_size))
    # print(in_size)
    # print(out_size)
    # print(max_val)
    # print(min_val)
    W, b = np.random.uniform(min_val, max_val, (in_size, out_size)) , np.random.uniform(min_val, max_val, out_size)

    params['W' + name] = W
    params['b' + name] = b



############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    res += 0.

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    ##########################
    # pre-activation:
    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    # numerical stability 
    max_x = x.max(axis=1)
    center_x = x-max_x[:, None]
    norm_x = np.exp(center_x)
    # print(f"center_x: {center_x}")
    # print(f"norm_x: {norm_x}")
    # softmax
    sum_of_row = norm_x.sum(axis=1)
    # print(f"shape of sum_of_row: {sum_of_row.shape}")
    res = norm_x/sum_of_row[:, None]
    # print(res)
    res += 0.
    # print(f"Sum of res: {res.sum(axis=1)}")

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = 0.0, 0.0

    ##########################
    ##### your code here #####
    ##########################
    cnt = 0

    for i in range(y.shape[0]):
        y_i = y[i,:]
        pred_i = probs[i,:]
        # classifier
        f_x = np.zeros(y.shape[1])
        pred_idx = np.where(pred_i == pred_i.max())[0]
        f_x[pred_idx] = 1
        
        # calculate the loss first
        # print(f"pred_i: {pred_i}")
        # print(f"max of pred_i: {pred_i.max()}")
        # print(f"index of max: {pred_idx[0]}")
        # print(f"pred_i: {pred_i}")
        # print(f"np.log(pred_i, where=(pred_i!=0)): {np.log(pred_i, where=(pred_i!=0))}")
        # print(np.dot(np.log(pred_i, where=(pred_i!=0)), y_i))
        # print(np.dot(y_i, f_x))
        # loss -= np.dot(np.log(pred_i, where=(pred_i!=0)), y_i)
        cnt += np.dot(y_i, f_x)
    loss = -(y * np.log(probs)).sum()
    return loss, cnt/y.shape[0] 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop (L)
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first (dLdZ)
    dLdA = delta
    dAdZ = activation_deriv(post_act)
    dLdZ =  dLdA * dAdZ 
    # then compute the derivative W,b, and X
    # print(f"delta.shape = {delta.shape}")
    # print(f"grad_Act.shape = {grad_Act.shape}")
    # print(f"X.shape = {X.shape}")
        
    # # derivative of W = dLdZ * dZdW
    grad_W = X.T @ dLdZ
    # derivative of b = dLdZ * dZdb
    grad_b = (np.ones((1, dLdZ.shape[0])) @ dLdZ).flatten()
    # derivative of X = dLdZ * dZdX
    grad_X = dLdZ @ W.T

    ##########################
    ##### your code here #####
    ##########################
    

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X



############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    batch_num = int(x.shape[0]/batch_size)
    # print(f"batch_num: {batch_num}")
    # print(f"x.shape: {x.shape}")
    # generate random index 
    rand_idx = np.random.choice(np.arange(x.shape[0]), x.shape[0], replace=False)
    # print(rand_idx)

    for i in range(batch_num):
        batch_idx = rand_idx[i*batch_size:(i+1)*batch_size]
        batches.append((x[batch_idx, :], y[batch_idx, :]))

    return batches