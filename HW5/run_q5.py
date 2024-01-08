import numpy as np
import scipy.io
from nn import *
from collections import Counter
import util

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# max_iters = 10
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
np.random.seed(100)
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
params = {}

initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'layer2')
initialize_weights(32, 32, params, 'layer3')
initialize_weights(32, 1024, params, 'output')

# for all the weights, we initialized Mw
keys = params.keys()

for key in list(keys):
    # make a new list of parameters
    params["m_"+str(key)] = np.zeros(params[key].shape)

loss_list = []

print(params.keys())

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        a1 = forward(xb,params,'layer1', relu)
        a2 = forward(a1,params,'layer2', relu)
        a3 = forward(a2,params,'layer3', relu)
        a4 = forward(a3,params,'output', sigmoid)

        # loss
        total_loss += np.square(xb-a4).sum()

        # backpropagation
        delta1 = -2*(xb-a4)
        delta2 = backwards(delta1,params,'output', sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3', relu_deriv)
        delta4 = backwards(delta3,params,'layer2', relu_deriv)
        backwards(delta4, params,'layer1', relu_deriv)

        # now update the weights with Mw = 0.9Mw - alpha*partial_derivative
        for k, v in params.items():
            if "m_" in k:
                # calculate Mw
                params[k] = 0.9*params[k] - learning_rate*params['grad_'+k[2:]]
                # update weights
                params[k[2:]] += params[k]

    loss_list.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# print(loss_list)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([i+1 for i in range(max_iters)], loss_list)
ax.legend(['lost for each epoch'])
ax.grid(visible=True)
ax.set_title("Loss Across Epoch", fontsize=10, wrap=True)
plt.show()
plt.savefig("../result/autoencoder_loss.png")


#saving the dictionary
import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open("../model/q4.pickle", 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

params = pickle.load(open("../model/q4.pickle",'rb'))

# Q5.3.1
# visualize some results
a1 = forward(valid_x,params,'layer1', relu)
a2 = forward(a1,params,'layer2', relu)
a3 = forward(a2,params,'layer3', relu)
a4 = forward(a3,params,'output', sigmoid)

print(a4.shape)
pick = [(12, 65), (105, 124), (1221, 1263), (937, 992), (2807, 2884)]
fig, ax = plt.subplots(nrows=5, ncols=4)
fig.set_size_inches(10, 10.5, forward=True)
counter = 0
for i in pick:
    print(i)
    ax[counter, 0].imshow(valid_x[i[0]].reshape(32,32).T)
    ax[counter, 1].imshow(a4[i[0]].reshape(32,32).T)
    ax[counter, 2].imshow(valid_x[i[1]].reshape(32,32).T)
    ax[counter, 3].imshow(a4[i[1]].reshape(32,32).T)
    counter += 1
    
plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
psnr = 0

for i in range(a4.shape[0]):
    psnr += peak_signal_noise_ratio(valid_x[i], a4[i])

psnr /= a4.shape[0]
print(psnr)

