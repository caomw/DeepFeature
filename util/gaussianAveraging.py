#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
def gaussian_average(data,the_axis = 1):
    '''
    Average gaussian weights to an array
    
    Parameters
    ----------
        the_data    channel * height *  width
    
    Output
    ----------
        return the gaussian average data in chanel * width
    '''
    number,height,width = data.shape
    
    std = np.std(data,axis = the_axis) 
    coefficient = (1.0 / (math.pi ** 0.5)) / std 
    weights = np.zeros([number,height,width])
    mid = np.floor(height / 2)
    for z in range(number):
        for i in range(height):
            for j in range(width):
                weights[z][i][j] = coefficient[z][j] * math.exp((-0.5)*((i - mid )** 2) / (std[z][j] ** 2))
    multi = (data * weights)
    result = np.sum(multi,axis = the_axis)
    plt.imshow( result)
    plt.show()
    return result
