#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import util.NMS as NMS
def nmsLineResponse(vec, thresh, windowSize = 5):
    
    vec = vec - thresh
    length = len(vec) 

    responses = np.zeros([1,length])[0];
    for rr in range(length):
        if vec[rr] > 0 and vec[rr] == max(vec[max(0,rr - windowSize ):min(length,rr + windowSize + 1)]):
            responses[rr] = vec[rr] 
    
    start = 0
    count = -1
    for jj in range(len(responses)):
	if responses[start] != responses[jj]:
		if count > 0:
			the_max = responses[jj - 1]
			responses[start:jj] = 0
			responses[(jj+start)/2] = the_max
			count = -1
		start = jj
		count = 0
	else:
		count += 1
    
    return responses
    
def getSpaces(response_map, row, col_start, col_end, nSpaces, scale):
    r_nms = nmsLineResponse(-1 * response_map[row,col_start:col_end+1],-0.25, 10)
    indices = np.argsort(r_nms)[::-1]
    r_space = len([item for item in r_nms if item > 0])
    num = min(nSpaces, r_space);
    scores = np.array(r_nms)[indices[0:num + 1]];
    temp_list = (np.array(indices[0:num + 1]) +12.0) / scale
    idex = [min(col_end/scale,item) for item in temp_list] 
    return (idex,scores)


def findLinesFull(feature_map,scale,thresh = 0.8,space = 5):
    bboxes = []
    spaces = []
    chars  = []
    fullResponse = []
    feature_height,feature_width = feature_map.shape
    for i in range(feature_height):
        vec = feature_map[i,:]
        nms = nmsLineResponse(vec, thresh, 5)
        peaks = [index for index, e in enumerate(nms) if e > 0]
       
        separations = np.diff(peaks)
        if len(peaks) == 3 and (max(separations)*1.0 / min(separations)) >= 3:
            continue
        if len(peaks) > 2:
            median = np.median(separations)
            start = 0
            for j in range(len(separations)):
                if separations[j] > 5 * median:
                    if j - start >= 2:
                        rect = (round(peaks[start]/scale),round(i/scale),round((peaks[j]+24.0)/scale), \
                               round((i +24.0)/scale),np.mean(nms[peaks]))
                        #rect = (max(0,round((peaks[start]*1.0)/scale)),max(0,round(1.0*i/scale)),round((peaks[j]+24.0)/scale),round((i +24.0)/scale),np.mean(nms[peaks]))                        
                        #print rect,i,scale,np.mean(nms[peaks])
                        bboxes.append(rect)
                        aspectRatio =  (rect[2] - rect[0])*1.0 / (rect[3] - rect[1])
                        if aspectRatio > 20:
                            space = space * 4
                        elif aspectRatio > 10:
                            space = space * 2
                        charScores = nms[peaks[start:j+1]]
                        locations = (np.array(peaks[start:j+1]) - peaks[start]*1.0)/scale + 1
                        chars.append((locations,charScores))
                        temp_start = max(0,peaks[start])
                        temp_end = min(feature_width, peaks[j]+24)
                        fullResponse.append((feature_map[i,temp_start:temp_end],scale))
                        spaces.append(getSpaces(feature_map, i, temp_start, temp_end, space, scale))
                    
                    start = j + 1 
                    
                    
            j = j + 1
            if j - start >= 2:
                rect = (round(peaks[start]/scale),round(i/scale),round((peaks[j]+24.0)/scale) \
                        ,round((i +24.0)/scale),np.mean(nms[peaks]))
                        #rect = (max(0,round((peaks[start]*1.0)/scale)),max(0,round(1.0*i/scale)),round((peaks[j]+24.0)/scale),round((i +24.0)/scale),np.mean(nms[peaks]))                        
                #print rect,i,scale,np.mean(nms[peaks])
                bboxes.append(rect)
                aspectRatio =  (rect[2] - rect[0])*1.0 / (rect[3] - rect[1])
                if aspectRatio > 20:
                    space = space * 4
                elif aspectRatio > 10:
                    space = space * 2
                charScores = nms[peaks[start:j+1]]
                locations = (np.array(peaks[start:j+1]) - peaks[start]*1.0)/scale + 1
                chars.append((locations,charScores))
                temp_start = max(0,peaks[start])
                temp_end = min(feature_width, peaks[j]+24)
                fullResponse.append((feature_map[i,temp_start:temp_end],scale))
                spaces.append(getSpaces(feature_map, i, temp_start, temp_end, space, scale))
    return bboxes,spaces,chars,fullResponse
            
                    
def findBoxes(feature_map,height,width):
    feature_height,feature_width = feature_map.shape
    scale = 1.0 * (feature_height + 23) / height
    return findLinesFull(feature_map, scale);
        
        
        
        
    
    

    
        
  
       