import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

# (240, 320, 415)
# print(seq.shape)

rect_list = []
rect_list.append(rect)

p = np.zeros(2)
# range through all the frames in the video
for i in range(seq.shape[2]-1):
# for i in range(5):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    p = LucasKanade(It, It1, rect, threshold=threshold, num_iters=int(num_iters))
    # print(p)
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    rect_list.append(rect)
    # print(rect)
    if i+1 in [1, 100, 200, 300, 400]:
    # if i in [1, 20, 40, 60, 80, 100, 200, 300, 400, 414]:
        print(i)
        fig, ax = plt.subplots()
        ax.imshow(It1, cmap='gray')
        rect_draw = patches.Rectangle((rect[0], rect[1]), width=int(rect[2]-rect[0]+1), height=int(rect[3]-rect[1]+1), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_draw)
        # plt.imshow()
        print("Saving image at frame {i}.".format(i=i+1))
        plt.savefig('car_frame'+str(i+1)+'.png')

np.save("carseqrects.npy", np.array(rect_list))

        



# plt.imshow(seq[:, :, 0], interpolation='nearest')
# plt.show()
# plt.imshow(seq[116:151, 59:145, 0], interpolation='nearest')
# plt.show()
# plt.imshow(seq[116:151, 59:145, 10], interpolation='nearest')
# plt.show()

# test WarpPatchTranslation
# LucasKanade(seq[:, :, 0], seq[:, :, 1], rect, threshold=6.2, num_iters=1000)
