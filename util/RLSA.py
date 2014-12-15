#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import skimage
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
def RLSA_BY_HORIZOANL_AND_BOUNDING(ostu_image,offset_x,offset_y,horizonal_spacing,thresh=0.5):
    h,w = ostu_image.shape
    for x in range(h):
            vec = ostu_image[x,:]
            nms = nmsLineResponse(vec, thresh,1)
            peaks = [index for index, e in enumerate(nms) if e > 0]
            if len(peaks) > 2:
                average = horizonal_spacing[x]
                for i in range(1,len(peaks)):
                    if peaks[i] - peaks[i-1] < average:
                        ostu_image[x,peaks[i-1]:peaks[i] + 1] = 1
    contours, hierarchy = cv2.findContours(np.array(ostu_image*255,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for item in contours:
        x,y,w,h = cv2.boundingRect(item)
        boxes.append((x+offset_x,y+offset_y,x+w+offset_x,y+h+offset_y))
    return boxes,ostu_image
    
def FIND(grey_image,feature_map,image_name,thresh = 0.5):
    
    h,w = feature_map.shape  
    grey_image = skimage.transform.resize(grey_image,[h+23,w+23])
    grey_image = grey_image * 255
    mask = (feature_map > thresh)*1
    hor_image = (feature_map > thresh)*1
    after_crop_image = copy.deepcopy(grey_image)
    h2,w2 = after_crop_image.shape
    ostu_image = np.zeros([h2,w2])
    #text_lines = []
    horizonal_spacing = np.zeros([h+23,1])
    binary_feature_map = (feature_map > thresh)*1
    for x in range(h):
        temp_horizonal_spacing = []
        vec = binary_feature_map[x,:]
        nms = nmsLineResponse(vec, thresh,1)
        peaks = [index for index, e in enumerate(nms) if e > 0]
        separations = np.diff(peaks)
        if 40 > len(peaks) > 2:
            temp_spacing = 0
            average = 0
            mean = np.mean(separations)
            std = np.std(separations)
            
            average = 3*mean - 0.5*std
            horizonal_spacing[x-12:x+13] = average
            for i in range(1,len(peaks)):
                if peaks[i] - peaks[i-1] < average:
                    hor_image[x,peaks[i-1]:peaks[i] + 1] = 1
                else:
                    temp_horizonal_spacing.append((peaks[i-1],peaks[i],x))
                    temp_spacing += separations[i-1]
            if len(temp_horizonal_spacing) > 0:
                horizonal_spacing[x-12:x+13] = temp_spacing*1.0/len(temp_horizonal_spacing)
    xxx =  cv2.imread(image_name)
    xxx =  skimage.transform.resize(xxx,[h2,w2,3])
    contours, hierarchy = cv2.findContours(np.array(hor_image*255,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    textline_boxs = []
    first_box = []
    for item in contours:
        x,y,ww,hh = cv2.boundingRect(item)
        if hh > 6:
            textline_boxs.append((x,y,x+ww+24,y+hh+24))
            cv2.rectangle(xxx, (int(x),int(y)), (int(x+ww+12), int(y+hh+12)), (0, 255, 0), 2)
        if hh > 6:
            score = np.mean(feature_map[y+h:max(y+1,y+hh-23),x:max(x+1,x+ww-23)])
            first_box.append((x,y,x+ww+24,y+hh+24,score))

        
    FINAL_BOX = []
    for item in textline_boxs: 
        col_1,row_1,col_2,row_2 = item
       
        #plt.imshow(after_crop_image[row_1:row_2+1,col_1:col_2+1])
        #plt.show()
        
        for_ostu =  after_crop_image[row_1:row_2+1,col_1:col_2+1]
        blur = cv2.GaussianBlur(for_ostu,(5,5),0)
        ret1,temp1 = cv2.threshold(np.array(blur,dtype=np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2,temp2 = cv2.threshold(np.array(blur,dtype=np.uint8),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)     
        if temp1 == None or temp2 == None:
            continue
        contours1, hierarchy = cv2.findContours(copy.deepcopy(temp1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy = cv2.findContours(copy.deepcopy(temp2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        standard = mask[row_1:row_2-22,col_1:col_2-22]
        hhh,www = temp1.shape
        normal = copy.deepcopy(np.array(temp1)[12:hhh-11,12:www-11])
        inv = copy.deepcopy(np.array(temp2)[12:hhh-11,12:www-11])
        if np.sum(standard^normal) < np.sum(standard^inv):
            #plt.rcParams['image.cmap'] = 'gray'
            #plt.imshow(temp1)
            #plt.title('normal')
            #plt.show()
            boxes,ostu_image[row_1:row_2+1,col_1:col_2+1] = RLSA_BY_HORIZOANL_AND_BOUNDING(temp1,col_1,row_1,horizonal_spacing)
        else:
            #plt.rcParams['image.cmap'] = 'gray'
            #plt.imshow(temp2)
            #plt.title('inv')
            #plt.show()
            boxes,ostu_image[row_1:row_2+1,col_1:col_2+1] = RLSA_BY_HORIZOANL_AND_BOUNDING(temp2,col_1,row_1,horizonal_spacing)
        for each in boxes:
            x1,y1,x2,y2 = each
            score = np.mean(feature_map[y1:max(y1+1,y2-23),x1:max(x1+1,x2-23)])
            FINAL_BOX.append((x1,y1,x2,y2,score))

    #plt.imshow(ostu_image)
    #plt.title('ostu_image')
    #plt.show()
    return FINAL_BOX,first_box
    '''
    
    for item in textline_boxs:
           x1,y1,x2,y2= item
           ret,mask = cv2.threshold(np.array(after_crop_image[row,col1:col2+1],dtype=np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
           ret,mask2 = cv2.threshold(np.array(after_crop_image[row,col1:col2+1],dtype=np.uint8),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
           if mask != None:
               ostu_image[row,col1:col2+1] = list(((mask > ret) *1))
    
    print '###############################'

            for y in range(peaks[0],peaks[len(peaks) -1] + 1):
                if  hor_image[x][y] == 255:
                    if y - c <= average and y - c > 0:
                        if np.sum(hor_image[x,c:y+1]) < (y + 1 - c)*255: 
                            line_centers.append((x,c,y))
                        hor_image[x,c:y+1] = 255 
                    c = y
    '''
