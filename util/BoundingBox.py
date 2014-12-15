#!/usr/bin/env python
import cv2
import matplotlib.pylab as plt
import numpy as np
def bounding_box(rlsa_img_path,origin_img_path,scale,red = 0,green = 0,blue = 0,thresh_area = 0):
    """
    find bounding box based on the RLSA image

    Parameters
    ----------
    rlsa_img_path:
        the image path of RLSA image
    origin_img_path: 
        the image path of origin image
    red:
        the red channel of bounding box(default = 0)
    green:
        the green channel of bounding box(default = 0)
    blue:
        the blue channel of bounding box(default = 0)
    thresh_area:
        the thresh area to the bounding box
    scale:
        the image scale 
    Output
    ------
    bound_img: ndarray
        image with bounding box 
        the image data in the form of [Height Width  Channel]
    """
    image_data= cv2.imread(rlsa_img_path)
    imgray = cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    origin_img = cv2.imread(origin_img_path)
    height,width=origin_img.shape[:2]
    the_width  = int(np.ceil(scale*width))
    the_heigth = int(np.ceil(scale*height))
    img=cv2.resize(origin_img,(the_width,the_heigth),interpolation=cv2.INTER_CUBIC)
    cv2.drawContours(img, contours, -1, (red,green,blue), 3)
    bound_img = cv2.imread(origin_img_path)
    height,width=bound_img.shape[:2]
    the_width  = int(np.ceil(scale*width))
    the_heigth = int(np.ceil(scale*height))
    the_img=cv2.resize(bound_img,(the_width,the_heigth),interpolation=cv2.INTER_CUBIC)
    list_rectangele = []
    for item in contours:
        x,y,w,h = cv2.boundingRect(item)
        if w * h > thresh_area and h < 1.5*w and w * h < 0.25 * the_width * the_heigth:
            cv2.rectangle(the_img,(x,y),(x+w,y+h),
                (0,255,0),2)
            list_rectangele.append((x,y,w,h))
    return the_img,list_rectangele