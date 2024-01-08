import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import copy

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect_origin = [280, 152, 330, 318]
rect = copy.deepcopy(rect_origin)
rect_list = np.load("girlseqrects.npy")

rect_list_WT = []
rect_list_WT.append(rect_origin)

frame_0 = seq[:,:,0]
It = seq[:,:,0]

p_start = np.zeros(2)
p_n = np.zeros(2)
# range through all the frames in the video
# for i in range(10):
for i in range(seq.shape[2]-1):
    print("Frame number: {indx}".format(indx=i+1))
    # current template 
    # current image
    It1 = seq[:,:,i+1]

    # calculate normal p
    p = LucasKanade(It, It1, rect, threshold=threshold, num_iters=int(num_iters), p0=p_start)
    # calculate accumulated p_n
    p_n = p + np.array([rect[0]-rect_origin[0], rect[1]-rect_origin[1]])

    # calculate p star
    p_star = LucasKanade(frame_0, It1, rect_origin, threshold=threshold, num_iters=int(num_iters), p0=p_n)

    # performing correction
    if np.sqrt(np.square((p_star - p_n)).sum()) < template_threshold:
        print("Update the template.")
        # if the change is smaller than the threshold
        # the result is not very different from the first
        # we can safely update the template by updating the image and rectangle
        # use the current image as template for next frame
        It = seq[:, :, i+1]
        # p_star_change = [p_star[0]-rect_origin[0], p_star[1]-]
        # update rectange using p star
        rect = [rect_origin[0]+p_star[0], rect_origin[1]+p_star[1], rect_origin[2]+p_star[0], rect_origin[3]+p_star[1]]
        # update p0 also since we will have a new starting point
        p_start = np.zeros(2)
        rect_list_WT.append(rect)

    else:
        print("Don't update the template.")
        # act conservatively, keep the template
        # rect = [rect_origin[0]+p_star[0], rect_origin[1]+p_star[1], rect_origin[2]+p_star[0], rect_origin[3]+p_star[1]]
        # rectangle = Original rect + delta p from p*
        p_start = p
        rect_list_WT.append([rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]])

    # print(rect)
    if i+1 in [1, 20, 40, 60, 80]:
        fig, ax = plt.subplots()
        ax.imshow(It1, cmap='gray')
        correct_rect_draw = patches.Rectangle((rect[0], rect[1]), width=int(rect[2]-rect[0]+1), height=int(rect[3]-rect[1]+1), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(correct_rect_draw)
        bad_rect = rect_list[i]
        non_rect_draw = patches.Rectangle((bad_rect[0], bad_rect[1]), width=int(bad_rect[2]-bad_rect[0]+1), height=int(bad_rect[3]-bad_rect[1]+1), linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(non_rect_draw)
        print("Saving image at frame {i}.".format(i=i+1))
        plt.savefig('girl_frame'+str(i+1)+'_withCorrection.png')

# print(rect_list_WT)
np.save("girlseqrects-wcrt.npy", np.array(rect_list_WT))



