#!/usr/bin/env python
def ComputingRectangleScores(rectangles,feature_map):
    """
    computing the score of the rectangle based on the feature map

    Parameters
    ----------
    rectangles:
        the rectangle or the bounding box in the form of (x,y,w,h)
    scale:
        the scale
    feature_map: 
        the image feature map on the scale
    Output
    ------
    score_box:
        bounding box on origin image in the form of (x,y,w,h,scores)
    """
    score_box = []
    score = 0
    max_h,max_w = feature_map.shape
    for rectangle in rectangles:
        x1,y1,x2,y2 = rectangle
        w = x2 -x1
        h = y2 -y1
        list_index = [(x1 + i,y1 + j) for i in range(w) for j in range(h)]
        for index_i,index_j in list_index:
            score += feature_map[min(index_j,max_h-1)][min(index_i,max_w - 1)]
        score /= ((w*h)*1.0)
        score_box.append([x1,y1,x2,y2,score])
        score = 0
    return score_box