#!/usr/bin/env python
from xml.etree import ElementTree
from xml.dom import minidom
import os
def xmlGenerator(root_dir,dst_xml):
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
    files = FileFilter()
    files.FindFile(dirr = root_dir)

    tagset = ElementTree.Element('tagset')
    
    for each in files.fileList:
        image = ElementTree.SubElement(tagset, 'image')
        imageName = ElementTree.SubElement(image, 'imageName')
        imageName.text = each.split('/')[3].split('.')[0]
        taggedRectangles = ElementTree.SubElement(image, 'taggedRectangles')
        file_object = open(each)
        for line in file_object.readlines():
    
            bounding = [int(s.strip()) for s in line.split(',') if s.strip().isdigit()]
            taggedRectangle = ElementTree.SubElement(taggedRectangles, 'taggedRectangle')
            taggedRectangle.set('x','%d'%(bounding[0]))
            taggedRectangle.set('y','%d'%(bounding[1]))
            taggedRectangle.set('width','%d'%(bounding[2] - bounding[0]))
            taggedRectangle.set('height','%d'%(bounding[3] - bounding[1]))
            
    rough_string = ElementTree.tostring(tagset,'utf-8')
    reparsed = minidom.parseString(rough_string)
    text_file = open(dst_xml, "w")
    text_file.write(reparsed.toprettyxml(indent="  "))
    text_file.close()