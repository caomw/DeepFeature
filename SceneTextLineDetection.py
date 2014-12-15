#!/usr/bin/env python
def SceneTextLineDetection(filename,store_dir,net,temp):
    import util.padImage as pd
    import util.findBoxes as findMe
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import sys
    import copy
    import skimage
    import util.NMS as nms
    import scipy.io as sio
    caffe_root = '/home/panhe/caffe/'  
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    filename = './data/ICDAR2013/img_34.jpg'
    image_name = filename
    image_data = caffe.io.load_image(image_name)
    height,width,channel = image_data.shape
    aspect_bboxes = []
    bboxes = []
    spaces = []
    chars  = []
    lineResponses = []
    plt.rcParams['image.interpolation'] = 'nearest'
    list_height = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    #aditional_set = set([0.15,0.25,0.5,0.75,1.0,1.25,1.5])  
    aditional_set = set([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]) 
    feature_list = []
    flag = 0
    for scale in list_height:
        scale_data = copy.deepcopy(image_data)
        new_height = scale
        new_width  = int(scale*width*1.0/height)
        if scale in aditional_set:
            new_height = int(scale*height)
            new_width  = int(scale*width)
        scale_data = skimage.transform.resize(scale_data,[new_height,new_width,3])
        ##scale_data = skimage.transform.resize(scale_data,[np.ceil(scale*height),np.ceil(scale*width),3])     
        #scale_data = pd.pad_image(scale_data,23,23,0)
        #plt.imshow(scale_data)
        #plt.show()
        scale_height,scale_width,scale_channel = scale_data.shape
        if scale_height < 24 or scale_width < 24:
            flag = 1
            continue
        if scale_height * scale_width > 480 * 640:
            output_data = np.zeros([scale_height - 23,scale_width - 23])
            patch_num_h = int(np.ceil((scale_height-23)*1.0/400))
            patch_num_w = int(np.ceil((scale_width -23)*1.0/400))

            offset = 12
            for index_height in range(0,patch_num_h):            
                for index_width in range(0,patch_num_w):            

                    
                    left   = offset + index_width*400
                    top    = offset + index_height*400
                    if index_width == patch_num_w - 1:
                        right = scale_width - 11
                        step_width = right - left + 1
                    else:
                        step_width = 400
                        right = left + step_width - 1
                    if index_height == patch_num_h - 1:
                        bottom = scale_height - 11
                        step_height = bottom - top + 1 
                    else:
                        step_height = 400
                        bottom = top + step_height - 1
                    patch = scale_data[top - offset:bottom + offset ,left - offset:right + offset,:]
                    patch_height,patch_width,patch_channel = patch.shape
                    net.set_input_dim(1,patch_channel,patch_height,patch_width)
                    patch = net.preprocess('data', patch)
                    patch = patch.reshape(1,patch.shape[0],patch.shape[1],patch.shape[2])
                    patch = net.forward(data=patch)
                    patch = patch['prob'][0][1]
                    output_data[top-offset:top-offset+step_height,left-offset:left-offset + step_width] = patch
        else:
            net.set_input_dim(1,scale_channel,scale_height,scale_width)          
            scale_data = net.preprocess('data', scale_data)
            scale_data = scale_data.reshape(1,scale_data.shape[0],scale_data.shape[1],scale_data.shape[2])           
            output_data = net.forward(data=scale_data)
            output_data = output_data['prob'][0][1]   #the first feature map   
        feature_map = copy.deepcopy(output_data)
        feature_list.append(feature_map)
        #sio.savemat('%'a.mat',{'data':feature_map})
        #plt.imshow(feature_map)
        #plt.show()
        b,s,c,r = findMe.findBoxes(feature_map,height,width)
        for item in b:
            x1,y1,x2,y2,score = item
            if 1 < (x2 - x1) / (y2 - y1) < 20:
                aspect_bboxes.append(item)
            bboxes.append(item)
        spaces += s
        chars  += c
        lineResponses += r
    if height > 2000:
        sio.savemat('%s%s%s'%('./matfile/extremLarge/',temp,'.mat'),{'feature_V0_1_15':feature_list}) 
    elif 2000 >height > 1000:
        sio.savemat('%s%s%s'%('./matfile/big/',temp,'.mat'),{'feature_V0_1_15':feature_list}) 
    else:
        if flag == 1:
            sio.savemat('%s%s%s'%('./matfile/tiny/',temp,'.mat'),{'feature_V0_1_15':feature_list})
        else:
            sio.savemat('%s%s%s'%('./matfile/small/',temp,'.mat'),{'feature_V0_1_15':feature_list})
    final= cv2.imread(image_name)        
    length = len(bboxes)
    result_file1 = open('%s%s%s%s'%('./temp/detection_all/','gt_',temp,'.txt'),'w')    
    TEMP= cv2.imread(image_name)        
    length = len(bboxes)
    print length
    for i in range(length):
        (startX, startY, endX, endY,score) = bboxes[i]
        result_file1.write('%d,%d,%d,%d\n'%(startX,startY,endX,endY))
        cv2.rectangle(TEMP, (int(startX),int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)         
    result_file1.close()
    cv2.imwrite('%s%s%s%s'%('./temp/pic/',temp,'_bounding_box_ALL','.jpg'),TEMP)    
    
    final= cv2.imread(image_name)  
    bboxes = nms.nms(np.array(bboxes),0.5) 
    length = len(bboxes)
    print length
    for i in range(length):
        (startX, startY, endX, endY,score) = bboxes[i]
        cv2.rectangle(final, (int(startX),int(startY)), (int(endX),int(endY)), (0, 255, 0), 2)  
    cv2.imwrite('%s%s%s%s'%('./temp/pic/',temp,'_bounding_box','.jpg'),final)
    result_file = open('%s%s%s%s'%('./temp/detection/','gt_',temp,'.txt'),'w')    
    for item in bboxes:
        startX, startY, endX, endY,score = item
        result_file.write('%d,%d,%d,%d\n'%(startX,startY,endX,endY))
    result_file.close()
    
    
    final= cv2.imread(image_name)        
    length = len(aspect_bboxes)
    result_file1 = open('%s%s%s%s'%('./temp/detection_all_with_ratio/','gt_',temp,'.txt'),'w')    
    TEMP= cv2.imread(image_name)        
    length = len(aspect_bboxes)
    print length
    for i in range(length):
        (startX, startY, endX, endY,score) = aspect_bboxes[i]
        result_file1.write('%d,%d,%d,%d\n'%(startX,startY,endX,endY))
        cv2.rectangle(TEMP, (int(startX),int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)         
    result_file1.close()    
    cv2.imwrite('%s%s%s%s'%('./temp/pic/',temp,'_bounding_box_ALL_ratio','.jpg'),TEMP)    
    final= cv2.imread(image_name)  
    aspect_bboxes = nms.nms(np.array(aspect_bboxes),0.5) 
    #FILTER_BOX = nms.nms(np.array(FILTER_BOX),0.6)
    length = len(aspect_bboxes)
    print length
    for i in range(length):
        (startX, startY, endX, endY,score) = aspect_bboxes[i]
        cv2.rectangle(final, (int(startX),int(startY)), (int(endX),int(endY)), (0, 255, 0), 2)  
    cv2.imwrite('%s%s%s%s'%('./temp/pic/',temp,'_bounding_box_ratio','.jpg'),final)
    result_file = open('%s%s%s%s'%('./temp/detection_ratio/','gt_',temp,'.txt'),'w')    
    for item in aspect_bboxes:
        startX, startY, endX, endY,score = item
        result_file.write('%d,%d,%d,%d\n'%(startX,startY,endX,endY))
    result_file.close()
    
        
        
        
  
       