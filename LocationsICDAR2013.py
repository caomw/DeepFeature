#!/usr/bin/env python
import os
import sys
import TextLocation as tl
import TextLocationTest as tlt
import SceneTextLineDetection as sss
root_dir = './data/ICDAR2013'
temp_dir = './temp/'
image_format = ['.jpg']
import time
class FileFilter:
    fileList = []
    counter = 0
    def __init__(self):
        pass
    def FindFile(self,dirr,filtrate = 1):
        global image_format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr,s)
            if os.path.isfile(newDir):
                if filtrate:
                    if newDir and (os.path.splitext(newDir)[1] in image_format):
                        self.fileList.append(newDir)
                        self.counter += 1
                else:
                    self.fileList.append(newDir)
                    self.counter += 1


images = FileFilter()
images.FindFile(dirr = root_dir)
caffe_root = '/home/panhe/caffe/'  
sys.path.insert(0, caffe_root + 'python')
import caffe
net = caffe.Net('model/textDeployEdit.prototxt','model/text.caffemodel')
net.set_phase_test()
net.set_mode_gpu()
net.set_device(1)
net.set_raw_scale('data', 255)
net.set_channel_swap('data', (2,1,0))
print (images.counter)
time_start = time.time()
print time_start
for k in images.fileList:
    create_dir = temp_dir + (k.split('/')[3]).split('.')[0]
    #if not os.path.exists(create_dir):
        #os.mkdir(create_dir)
    print k
    tlt.TextLocationTest(k,create_dir,net,(k.split('/')[3]).split('.')[0])
    #sss.SceneTextLineDetection(k,create_dir,net,(k.split('/')[3]).split('.')[0])
    #jpg_file = split_first[3]
    #print type(jpg_file)
    #print jpg_file
time_end = time.time()
print 'total time'
print (time_end - time_start)*1.0/60
print 'average time'
print (time_end - time_start)*1.0/60/images.counter