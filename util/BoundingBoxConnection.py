#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cv2
def BoundingBoxConnection(bbox):
    number = len(bbox)
    record_matrix = np.zeros([number,number])
    list_index = [(x,y) for y in range(number - 1,-1,-1) for x in range(y)]
    #find pairs
    for pair1,pair2 in list_index:
        left1,top1,right1,bottom1 = bbox[pair1]
        left2,top2,right2,bottom2 = bbox[pair2]
        gravity1 = [ (left1+right1)/2.0,(top1+bottom1)/2.0]
        gravity2 = [ (left2+right2)/2.0,(top2+bottom2)/2.0]
        distance = np.sqrt((gravity1[0] - gravity2[0])*(gravity1[0] - gravity2[0]) + (gravity1[1] - gravity2[1])*(gravity1[1] - gravity2[1]))
        angle = np.arctan((gravity2[1] - gravity1[1]) / (gravity2[0] - gravity1[0] + 0.0000001))
        ratio = np.abs((bottom1-top1) * 1.0 / (bottom2 - top2))
        #the pair condition
        if np.abs(angle) < np.pi / 6 and ratio > 0.8 and ratio < 1.2:
            if( gravity2[1] > top1 and gravity2[1] < bottom1) or (gravity1[1] > top2 and gravity1[1] < bottom2):
                    record_matrix[pair1][pair2] = distance
    #compute the average distance of one box to others
    #plt.imshow(record_matrix)
    #plt.show()
    average_distance = np.zeros([1,number])
    for x in range(number):
        sum_distance = 0
        nozero_number = 0
        for y in range(number):
            if record_matrix[x][y] > 0:
                nozero_number += 1
                sum_distance += record_matrix[x][y]
        average_distance[:,x] = sum_distance / (nozero_number + 0.0000001)
    #filter out pairs whose distance greater than average
    list_groups = []
    list_each = []
    for x in range(number):
        list_each = [x]
        list_record = np.array(list(record_matrix[x]))
        idx = np.argsort(list_record)
        list_record = list_record[idx]
        for y in range(number):
            if  list_record[idx][y] < average_distance[:,x] and list_record[idx][y] > 0:
                list_each.append(y)
        list_groups.append(set(list_each))
    #find the group
    for x in range(number):
        temp_index = []
        union_group = []
        for y in range(len(list_groups)):
            if x in list_groups[y]:
                temp_index.append(list_groups[y])
                union_group += list(list_groups[y])
        list_groups.append(set(union_group))
        for item in temp_index:
            list_groups.remove(item)
    #groups each group with the index of bbox
    final_groups = []
    for item in list_groups:
        final_groups.append(list(item))
    #the return bounding box
    bounding_box = []
    for item in final_groups:
        x_left  = float('inf')
        y_left  = float('inf')
        x_right = 0
        y_right = 0
        sum_height = 0                    #for computing average height                    #for computing average width
        item_length = len(item)
        for index in range(item_length):
            sum_height += bbox[item[index]][3] - bbox[item[index]][1]
            x_left  = min(bbox[item[index]][0],x_left)
            y_left  = min(bbox[item[index]][1],y_left)
            x_right = max(bbox[item[index]][2],x_right)
            y_right = max(bbox[item[index]][3],y_right)
        bounding_box.append((x_left,y_left,x_right,y_right))
    return bounding_box
                