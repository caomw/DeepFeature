#!/usr/bin/env python
import numpy as np
def non_maxima_supression(image,threshold = 11):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(threshold, threshold))
    dilate_image = cv2.dilate(image,kernel)  # max() with radius of threshold

    nms_img = np.zeros([image.shape[0],image.shape[1]])
    print nms_img.shape
    list_index = [(c,r) for c in range(image.shape[1]) for r in range(image.shape[0])]   #column first
    for c,r in list_index:
    	if dilate_image[r][c] == image[r][c]:
    		nms_img[r][c] = image[r][c]
    return nms_img
