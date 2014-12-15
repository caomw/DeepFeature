#!/usr/bin/env python
import numpy as np

def pad_image(img,pad_width = 0,pad_height = 0,padval=0):
    """
    pad the image

    Parameters
    ----------
    img: ndarray
        the image data in the form of [Height Width  Channel]
    pad_width: 
        the pad size on width(default = 0)
    pad_height:
        the pad size on height(default = 0)
    pad_value:
        the pad value(default = 0)
    Output
    ------
    img: ndarray
        image afer padding 
        the image data in the form of [Height Width  Channel]
    """
    pad_h_1 = int(np.floor(pad_height*0.5))
    pad_h_2 = int(np.ceil(pad_height*0.5))
    pad_w_1 = int(np.floor(pad_width*0.5))
    pad_w_2 = int(np.ceil(pad_width*0.5))
    padding = ((pad_h_1,pad_h_2),(pad_w_1,pad_w_2),(0,0))
    img = np.pad(img,padding,mode="constant",constant_values=(padval,padval))
    return img
