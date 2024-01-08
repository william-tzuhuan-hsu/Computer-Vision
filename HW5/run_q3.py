import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

print(f"train_x.shape: {train_x.shape}")
print(f"train_y.shape: {train_y.shape}")
max_iters = 100
# max_iters = 4
# pick a batch size, learning rate
batch_size = 128
learning_rate = 1e-3
hidden_size = 64

# Q3.1

# ##########################
# ##### your code here #####
# ##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)




# ##################################### training loop #####################################
# print("Start training loop!")
# # with default settings, you should get loss < 150 and accuracy > 80%

# for i in range(3):
#     # initialize model
#     params = {}
#     # initialize hidden layers
#     initialize_weights(1024, 64, params, 'layer1')
#     # add output layer
#     initialize_weights(64, 36, params, 'output')

#     # initialize the lists
#     acc_list = []
#     valid_acc_list = []
#     avg_loss_list = []

#     if i == 0:
#         learning_rate *= 0.1
#     else:
#         learning_rate *= 10

#     print(f"Learning rate: {learning_rate}")
#     for itr in range(max_iters):
#         total_loss = 0
#         total_acc = 0
#         total_valid_acc = 0
#         for xb,yb in batches:
            
#             total_loss = 0
#             # pass
#             # forward
#             a1 = forward(xb,params,'layer1', sigmoid)
#             a2 = forward(a1,params,'output', softmax)
#             # loss
#             loss, acc = compute_loss_and_acc(yb, a2)
#             # be sure to add loss and accuracy to epoch totals 
#             total_loss += loss
#             total_acc += acc

#             # backward
#             delta1 = a2 - yb
#             delta2 = backwards(delta1,params,'output',linear_deriv)
#             backwards(delta2,params,'layer1',sigmoid_deriv)

#             # apply gradient
#             params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
#             params['blayer1'] -= learning_rate*params['grad_blayer1']
#             params['Woutput'] -= learning_rate*params['grad_Woutput']
#             params['boutput'] -= learning_rate*params['grad_boutput']

#         # validation
#         a1 = forward(valid_x, params, 'layer1', sigmoid)
#         a2 = forward(a1, params, 'output', softmax)

#         # loss
#         valid_loss, valid_acc = compute_loss_and_acc(valid_y, a2)

            
#         total_valid_acc = total_valid_acc/len(batches)
#         valid_acc_list.append(valid_acc)
#         total_acc = total_acc/len(batches)
#         acc_list.append(total_acc)
#         avg_loss_list.append(total_loss/len(batches))
#         if itr % 2 == 0:
#             print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

#     # run on validation set and report accuracy! should be above 75%
#     valid_acc = None

#     a1 = forward(valid_x, params, 'layer1', sigmoid)
#     a2 = forward(a1, params, 'output', softmax)
#     # loss
#     valid_loss, valid_acc = compute_loss_and_acc(valid_y, a2)

#     np.savez("../result/q3_1"+str(learning_rate)+".npz", acc_list=acc_list, avg_loss_list=avg_loss_list, valid_acc_list=valid_acc_list, valid_acc=valid_acc)
#     print('Validation accuracy: ',valid_acc)

#     # plot the accuracy plot
#     plt.figure()
#     plt.ylim(0, 1)
#     plt.plot([i+1 for i in range(max_iters)], acc_list)
#     plt.plot([i+1 for i in range(max_iters)], valid_acc_list)
#     plt.legend(['Training Accuracy', 'Validation Accuracy'])
#     plt.grid(visible=True)
#     plt.title("Accuracy across different epoch")
#     plt.savefig('../result/accuracy'+str(learning_rate)+".png")
    
    # print("saving dictionary.")
    # if learning_rate == 1e-3:
    #     import pickle
    #     saved_params = {k:v for k,v in params.items() if '_' not in k}
    #     with open('q3_weights.pickle', 'wb') as handle:
    #         pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print("dictionary saved!")

# ######################## training loop with different learning rate######################################



########################## training loop with best learning rate #####################################
learning_rate = 1e-2
step = 1.3
for i in range(10):
    learning_rate = np.exp(np.log(learning_rate)-np.log(step))
    # initialize model
    params = {}
    # initialize hidden layers
    initialize_weights(1024, 64, params, 'layer1')
    # add output layer
    initialize_weights(64, 36, params, 'output')

    # initialize the lists
    acc_list = []
    valid_acc_list = []
    avg_loss_list = []

    print(f"Learning rate: {learning_rate}")
    for itr in range(max_iters):
        total_loss = 0
        total_acc = 0
        total_valid_acc = 0
        for xb,yb in batches:
            
            total_loss = 0
            # pass
            # forward
            a1 = forward(xb,params,'layer1', sigmoid)
            a2 = forward(a1,params,'output', softmax)
            # loss
            loss, acc = compute_loss_and_acc(yb, a2)
            # be sure to add loss and accuracy to epoch totals 
            total_loss += loss
            total_acc += acc

            # backward
            delta1 = a2 - yb
            delta2 = backwards(delta1,params,'output',linear_deriv)
            backwards(delta2,params,'layer1',sigmoid_deriv)

            # apply gradient
            params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
            params['blayer1'] -= learning_rate*params['grad_blayer1']
            params['Woutput'] -= learning_rate*params['grad_Woutput']
            params['boutput'] -= learning_rate*params['grad_boutput']

        # validation
        a1 = forward(valid_x, params, 'layer1', sigmoid)
        a2 = forward(a1, params, 'output', softmax)

        # loss
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, a2)

            
        total_valid_acc = total_valid_acc/len(batches)
        valid_acc_list.append(valid_acc)
        total_acc = total_acc/len(batches)
        acc_list.append(total_acc)
        avg_loss_list.append(total_loss/len(batches))
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%
    valid_acc = None

    a1 = forward(valid_x, params, 'layer1', sigmoid)
    a2 = forward(a1, params, 'output', softmax)
    # loss
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, a2)
    print(f"The valid accuracy is {valid_acc}")

    print("saving dictionary.")

    import pickle
    saved_params = {k:v for k,v in params.items() if '_' not in k}
    with open("q3_weights"+str(valid_acc)+".pickle", 'wb') as handle:
        pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("dictionary saved!")

########################## training loop with best learning rate #####################################




# Q3.2

# # read in the saved list
# lowest = np.load("../result/q3_10.01.npz")
# acc_list_low = lowest["acc_list"]
# valid_acc_list_low = lowest["valid_acc_list"]
# avg_loss_list_low = lowest["avg_loss_list"]

# med = np.load("../result/q3_10.001.npz")
# acc_list_med = med["acc_list"]
# valid_acc_list_med = med["valid_acc_list"]
# avg_loss_list_med = med["avg_loss_list"]

# high = np.load("../result/q3_10.0001.npz")
# acc_list_high = high["acc_list"]
# valid_acc_list_high = high["valid_acc_list"]
# avg_loss_list_high = high["avg_loss_list"]

# # print(avg_loss_list_high)

# plt.figure(figsize=(2000, 1000))
# fig, ax = plt.subplots(nrows=3, ncols=2)
# # fig.tight_layout(pad=2)
# fig.set_figheight(15)
# fig.set_figwidth(15)
# # plt.subplots_adjust(top=1.2)
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
#                     top=0.9, wspace=0.4,hspace=0.4)

# # low 
# ax[0, 0].set_ylim(0, 1)
# ax[0, 0].plot([i+1 for i in range(max_iters)], acc_list_low)
# ax[0, 0].plot([i+1 for i in range(max_iters)], valid_acc_list_low)
# ax[0, 0].legend(['Training Accuracy', 'Validation Accuracy'])
# ax[0, 0].grid(visible=True)
# ax[0, 0].set_title("Accuracy for Learning Rate = " + str(learning_rate*0.1), fontsize=10, wrap=True)

# ax[0, 1].plot([i+1 for i in range(max_iters)], avg_loss_list_low)
# ax[0, 1].set_ylim(0, 10)
# ax[0, 1].legend(['Average Cross Entropy Loss'])
# ax[0, 1].grid(visible=True)
# ax[0, 1].set_title("Average Cross Entropy Loss for Learning Rate = " + str(learning_rate*0.1), fontsize=10, wrap=True)

# # mid
# ax[1, 0].set_ylim(0, 1)
# ax[1, 0].plot([i+1 for i in range(max_iters)], acc_list_med)
# ax[1, 0].plot([i+1 for i in range(max_iters)], valid_acc_list_med)
# ax[1, 0].legend(['Training Accuracy', 'Validation Accuracy'])
# ax[1, 0].grid(visible=True)
# ax[1, 0].set_title("Accuracy for Learning Rate = " + str(learning_rate), fontsize=10)

# ax[1, 1].plot([i+1 for i in range(max_iters)], avg_loss_list_med)
# ax[1, 1].set_ylim(0, 10)
# ax[1, 1].legend(['Average Cross Entropy Loss'])
# ax[1, 1].grid(visible=True)
# ax[1, 1].set_title("Average Cross Entropy Loss for Learning Rate = " + str(learning_rate), fontsize=10, wrap=True)

# # high
# ax[2, 0].set_ylim(0, 1)
# ax[2, 0].plot([i+1 for i in range(max_iters)], acc_list_high)
# ax[2, 0].plot([i+1 for i in range(max_iters)], valid_acc_list_high)
# ax[2, 0].legend(['Training Accuracy', 'Validation Accuracy'])
# ax[2, 0].grid(visible=True)
# ax[2, 0].set_title("Accuracy for Learning Rate = " + str(learning_rate*10), fontsize=10, wrap=True)

# ax[2, 1].plot([i+1 for i in range(max_iters)], avg_loss_list_high)
# ax[2, 1].set_ylim(0, 10)
# ax[2, 1].legend(['Average Cross Entropy Loss'])
# ax[2, 1].grid(visible=True)
# ax[2, 1].set_title("Average Cross Entropy Loss for Learning Rate = " + str(learning_rate*10), fontsize=10, wrap=True)


# plt.savefig("../result/accuracy_and_loss.png")

# Q3.3
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid

# # load the paramteres
# import pickle
# params = pickle.load(open('q3_weights.pickle','rb'))

# # check the accuracy of the loaded dictionary
# a1 = forward(valid_x, params, 'layer1', sigmoid)
# a2 = forward(a1, params, 'output', softmax)
# # loss
# valid_loss, valid_acc = compute_loss_and_acc(valid_y, a2)
# print(f"The loaded dictionary have accuracy of {valid_acc}.")

# visualize weights here

# w = params['Wlayer1']
# w = w.reshape(32, 32, 64)

# fig = plt.figure(1, (8., 8.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                  axes_pad=0.0,  # pad between axes in inch.
#                  )

# for i in range(64):
#     grid[i].imshow(w[:, :, i])

# plt.savefig("../result/weight_grid_trained.png")


# tmp_dict = {}
# initialize_weights(1024, 64, tmp_dict, 'layer1')
# w_init = tmp_dict['Wlayer1']
# w_init = w_init.reshape(32, 32, 64)

# fig = plt.figure(1, (8., 8.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                  axes_pad=0.0,  # pad between axes in inch.
#                  )

# for i in range(64):
#     grid[i].imshow(w_init[:, :, i])

# plt.savefig("../result/weight_grid_initial.png")




# # Q3.4
# confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
# # (actual, predicted)

# # # compute comfusion matrix here
# # forward pass
# a1 = forward(valid_x, params, 'layer1', sigmoid)
# a2 = forward(a1, params, 'output', softmax)

# for i in range(a2.shape[0]):
#     prob_i = a2[i, :]

#     predict = np.argmax(prob_i)
#     print(valid_y[i, :])
#     actual = np.where(valid_y[i, :]==1)[0][0]
#     print(predict, actual)
#     confusion_matrix[actual, predict] += 1

# print(confusion_matrix)


# import string
# plt.imshow(confusion_matrix,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()