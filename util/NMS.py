#!/usr/bin/env python
import numpy as np
def nms(boxes,overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]   
    area = (x2 - x1) * (y2 - y1)
    s =  boxes[:,4]
    idxs = np.argsort(s)
    while len(idxs) > 0:
    # grab the last index in the indexes list, add the index
    # value to the list of picked indexes, then initialize
    # the suppression list (i.e. indexes that will be deleted)
    # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            #big_w = max(x2[i],x2[j]) -  min(x1[i], x1[j]) + 1
            #big_h = max(y2[i],y2[j]) -  min(y1[i], y1[j]) + 1
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            inter = float(w * h)
            #big = float(big_w * big_h)
            #print inter
            overlap = inter*1.0 / area[i]
            #print xx1,xx2,yy1,yy2
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
            # delete all indexes from the index list that are in the
            # suppression list
        idxs = np.delete(idxs, suppress)
    return boxes[pick]