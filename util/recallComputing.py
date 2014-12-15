#!/usr/bin/env python
from xml.etree import ElementTree
from xml.dom import minidom
import os
def recallComputing(detect_dir,ground_truth_dir,dst_xml,overlapThresh):
    """
    generate the bounding box xml file

    Parameters
    ----------
    root_dir:
        the all txt file folder 
    dst_xml: 
        the to-generate xml path
    Output
    ------
    """
    
    class FileFilter:
        fileList = []
        counter = 0
        def __init__(self):
            pass
        def FindFile(self,dirr,filtrate = 1):
            file_format = ['.txt']
            for s in os.listdir(dirr):
                newDir = os.path.join(dirr,s)
                if os.path.isfile(newDir):
                    if filtrate:
                        if newDir and (os.path.splitext(newDir)[1] in file_format):
                            self.fileList.append(newDir)
                            self.counter += 1
                        else:
                            self.fileList.append(newDir)
                            self.counter += 1
    class FileFilter2:
        fileList = []
        counter = 0
        def __init__(self):
            pass
        def FindFile(self,dirr,filtrate = 1):
            file_format = ['.txt']
            for s in os.listdir(dirr):
                newDir = os.path.join(dirr,s)
                if os.path.isfile(newDir):
                    if filtrate:
                        if newDir and (os.path.splitext(newDir)[1] in file_format):
                            self.fileList.append(newDir)
                            self.counter += 1
                        else:
                            self.fileList.append(newDir)
                            self.counter += 1
    files_ground_trouth = FileFilter()
    files_ground_trouth.FindFile(dirr = ground_truth_dir)
    files_detect = FileFilter2()
    files_detect.FindFile(dirr = detect_dir)

    #print  files_detect.fileList,files_ground_trouth.fileList
    result = ElementTree.Element('result')
    sum_all = 0
    detect_sum = 0
    pairs = zip(files_detect.fileList,files_ground_trouth.fileList)
    for dectect,ground in pairs:
            #print dectect,ground
            image = ElementTree.SubElement(result, 'image')
            imageName = ElementTree.SubElement(image, 'imageName')
            imageName.text = dectect.split('/')[3].split('.')[0]
            recall = ElementTree.SubElement(image, 'recall')
            detection_box = []
            file_object = open(dectect)
            for line in file_object.readlines():
                bounding = [int(s.strip()) for s in line.split(',') if s.strip().isdigit()]
                detection_box.append((bounding[0],bounding[1],bounding[2],bounding[3]))
            ground_box = []
            file_object_ground = open(ground)
            for line in file_object_ground.readlines():
                bounding = [int(s.strip()) for s in line.split(',') if s.strip().isdigit()]
                ground_box.append((bounding[0],bounding[1],bounding[2],bounding[3]))
            set_for_recall = set()
            for item_detect in detection_box:
                x1,y1,x2,y2 = item_detect
                for item_gt in ground_box:
                     gt_x1,gt_y1,gt_x2,gt_y2 = item_gt
                     xx1 = max(x1, gt_x1)
                     yy1 = max(y1, gt_y1)
                     xx2 = min(x2, gt_x2)
                     yy2 = min(y2, gt_y2)
                     # compute the width and height of the bounding box
                     w = max(0, xx2 - xx1 + 1)
                     h = max(0, yy2 - yy1 + 1)
                     big_w = max(x2,gt_x2) -  min(x1, gt_x1)
                     big_h = max(y2,gt_y2) -  min(y1, gt_y1)
                     inter = float(w * h)
                     area1 = (gt_x2 - gt_x1)*(gt_y2 -gt_y1)
                     area2 = (x2 - x1)*(y2 -y1)
                     #area = big_w*big_h
                     #print inter
                     overlap = inter*1.0 /  area1;
                     #print xx1,xx2,yy1,yy2
                     # if there is sufficient overlap, suppress the
                     # current bounding box
                     if overlap > overlapThresh:
                         set_for_recall.add((gt_x1,gt_y1,gt_x2,gt_y2))
            sum_all += len(ground_box)
            detect_sum += len(set_for_recall)
            temp_recall = len(set_for_recall)*1.0/len(ground_box)
            recall.text = '%6.3f'%(temp_recall)
    overal = ElementTree.SubElement(result, 'overal')
    recalls = ElementTree.SubElement(overal, 'recall')
    recalls.text = '%6.3f'%(detect_sum*1.0/sum_all)
    rough_string = ElementTree.tostring(result,'utf-8')
    reparsed = minidom.parseString(rough_string)
    text_file = open(dst_xml, "w")
    text_file.write(reparsed.toprettyxml(indent="  "))
    text_file.close()
