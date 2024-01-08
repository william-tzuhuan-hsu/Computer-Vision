import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import time
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

# for i in range(len(seq)-2):
# # for i in range(10):
#     mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
#     motion = np.where(mask==True)
#     # print(motion)
#     if (i+1) % 10 == 0:
#         plt.imshow(seq[:,:,i+1], cmap='gray')
#         plt.plot(motion[1],motion[0], '.')
#         # plt.show()
#         plt.savefig("../result/ant_{i}.png".format(i=i))
#         plt.close()
start_time = time.time()
for i in range(seq.shape[2]-1):
# for i in range(10):
    mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
    motion = np.where(mask==True)
    # print(motion)
    # if (i+1) % 10 == 0:
    #     plt.imshow(seq[:,:,i+1], cmap='gray')
    #     plt.plot(motion[1],motion[0], '.')
    #     # plt.show()
    #     plt.savefig("../result/ant_inv_{i}.png".format(i=i))
    #     plt.close()
end_time = time.time()

print(end_time-start_time)