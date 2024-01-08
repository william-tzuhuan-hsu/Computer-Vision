import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    import matplotlib.pyplot as plt

    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    image = skimage.color.rgb2gray(image) 

    # # blur the image
    # blurred_img = skimage.filters.gaussian(image, sigma=8)
    # # restore 
    # restored_image = skimage.restoration.denoise_bilateral(blurred_img)
    # erosion 
    # erode_image = skimage.morphology.dilation(image)
    # print(erode_image)

    print("Thresholding")
    # thresh = skimage.filters.threshold_otsu(image)
    thresh = skimage.filters.threshold_isodata(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(10))
    # print(f"bw: {bw}")


    # print("# remove artifacts connected to image border")
    cleared = skimage.segmentation.clear_border(bw)

    print("label image regions")
    label_image = skimage.measure.label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    # image_label_overlay = skimage.color.label2rgb(label_image, image=image, bg_label=0)

    # print("saving image")
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    # plt.savefig("../result/image_label_overlay.png")

    # region_prop_list = skimage.measure.regionprops(label_image)
    # group some of the region together:
    # for region in region_prop_list:


    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            bboxes.append(region.bbox)
            # minr, minc, maxr, maxc = region.bbox
            # rect = matplotlib.patches.Rectangle(
            # (minc, minr),
            # maxc - minc,
            # maxr - minr,
            # fill=False,
            # edgecolor='red',
            # linewidth=2,
        # )
        # ax.add_patch(rect)

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.savefig("../result/box"+str(thresh)+".png")


    return bboxes, bw
