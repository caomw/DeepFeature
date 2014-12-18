#!/usr/bin/env python
import math
import xml.etree.cElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import time
import random
caffe_root = '/home/panhe/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import cv2
import caffe
import re
import skimage

#matplot setting
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

image_name = './word/2/156.jpg'
#net definition and setting 

net = caffe.Net('model/caseInSensitive.prototxt','model/textCaseInsensitive.caffemodel')
image_data = caffe.io.load_image(image_name)
print image_data.shape
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
#gray_data = rgb2gray(image_data)
'''
mean = np.mean(gray_data)
std = np.std(gray_data)
gray_data = (gray_data - mean)/std
maxVal = np.max(gray_data)
minVal = np.min(gray_data)
gray_data = (gray_data - minVal)/(maxVal - minVal)
'''
#image_data[:,:,0] = gray_data[:]
#image_data[:,:,2] = gray_data[:]
#image_data[:,:,1] = gray_data[:]
print image_data
plt.imshow(image_data)
plt.show()
new_height = 24
new_width = int(image_data.shape[1]* 24 /image_data.shape[0])
net.set_input_dim(1,3,new_height,new_width)
net.set_phase_test()
net.set_mode_gpu()
net.set_device(0)
net.set_raw_scale('data', 255)
net.set_channel_swap('data', (2,1,0))




#pad the image to enable text_saliency_map the same size with original image
def pad_image(img,pad_size,padval=0):
    """
    pad the image to enable text_saliency_map the same size with original image

    Parameters
    ----------
    img: ndarray
        the image data in the form of [Width Height Channel]
    pad_size: 
        the pad size

    Output
    ------
    img: ndarray
        image afer padding 
    """
    pad_h_1 = int(np.floor(pad_size)*0.5)
    pad_h_2 = int(np.ceil(pad_size)*0.5)
    pad_w_1 = int(np.floor(pad_size)*0.5)
    pad_w_2 = int(np.ceil(pad_size)*0.5)
    padding = ((pad_w_1,pad_w_2),(pad_h_1,pad_h_2),(0,0))
    img = np.pad(img,padding,mode="constant",constant_values=(padval,padval))
    return img



#print image_data.shape
#image_data = pad_image(image_data,23)
print image_data.shape
image_data = skimage.transform.resize(image_data,[new_height,new_width,3])
#image_data = (image_data > 0)*1.0
plt.imshow(image_data)
plt.show()
#print image_data.shape
image_data = net.preprocess('data', image_data) 
image_data = image_data.reshape(1,image_data.shape[0],image_data.shape[1],image_data.shape[2])

output_data = net.forward(data=image_data)
#print output_data['prob'][0].shape
response_data = output_data['prob'][0]
response_data = response_data.reshape(response_data.shape[0],response_data.shape[2])
plt.imshow(response_data[1:-1,:])
plt.title('feature map')
plt.show()

#Average gaussian weights along the column
def gaussian_average(data,the_axis = 1):
    '''
    Average gaussian weights along the column
    
    Parameters
    ----------
        the_data    channel * height *  width
    
    Output
    ----------
        return the gaussian average data in chanel * width
    '''
    number,height,width = data.shape

    response = np.zeros([number,width])

   
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
    return result
#####################################################
order_response = np.sort(response_data[0:-1,:], axis=0)
max_index = np.argmax(response_data[0:-1,:], axis=0)
print max_index
max_response = np.max(response_data[0:-1,:], axis=0)
confidence_score = order_response[-1,:] - order_response[-2,:]
print confidence_score
#####################################################
def nmsLineResponse(vec, thresh = 0, windowSize = 1):
    
    vec = vec - thresh
    length = len(vec) 

    responses = np.zeros([1,length])[0];
    for rr in range(length):
        if vec[rr] > 0 and vec[rr] == max(vec[max(0,rr - windowSize ):min(length,rr + windowSize + 1)]):
            responses[rr] = vec[rr] 
    '''
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
    '''
    climax_index =  [index for index, e in enumerate(responses) if e > 0]
    return responses,climax_index
 #####################################################
   
    
nms_response,climax_index = nmsLineResponse(confidence_score)
print climax_index,'#####'
LIST_LEXICON = []
tree = ET.ElementTree(file='word.xml')
root = tree.getroot()
for child_of_root in root:
    LIST_LEXICON.append(child_of_root.attrib['tag'])


#####################################################

def toLabel(word):
    new_word = np.zeros((1,len(word)))
    length = len(list(word))
    for i in range(length):
        order = ord(word[i])
        if order >=97: 
            #lowercase 27-52
            new_word[:,i] = order- 87;
        if(order>=65 and order<=90):
            #%upper case 1-26
            new_word[:,i] = order -55;
        if(order>=48 and order<=57):
            #%numbers 0-9
            new_word[:,i] = order - 48
    return new_word[0]
def matchScores(origscores, origionword, good_idx):
    scores = origscores[:,good_idx]
    word=toLabel(origionword)
    w= word.shape[0]
    if w < 1:
        matchScore = 0
        real_good_idx = 0
        return matchScore,real_good_idx
    s = scores.shape[1]
    print s,origscores.shape,scores.shape,word[0]
    os = origscores.shape[1]
    if w>s:
        matchScore = 0;
        real_good_idx = 0;
        return matchScore,real_good_idx
    scoreMat = np.zeros((w,s))
    scoreIdx = np.zeros((w,s))
    #Viterbi dynamic programming
    scoreMat[0,:]  = scores[word[0],:]
    for i in range(1,w):
        for j in range(i,s):
            maxPrev = np.max(scoreMat[i-1, i-1:j])
            maxPrevIdx = np.argmax(scoreMat[i-1, i-1:j])
            scoreMat[i,j] = scores[word[i], j] + maxPrev
            scoreIdx[i,j] = maxPrevIdx   
    
    matchScore = np.max(scoreMat[-1,:])
    lastidx = np.argmax(scoreMat[-1,:])
    temp_good_idx = np.zeros((w,1))
    temp_good_idx[-1] = lastidx
    real_good_idx = [good_idx[lastidx]]
    i = w - 1
    while i > 0:
        temp_good_idx[i-1,:] = scoreIdx[i, int(temp_good_idx[i,:])]+i-1
        real_good_idx.append(good_idx[int(temp_good_idx[i-1,:])])
        i = i -1
    gaps = []    
    length =len(real_good_idx)
    real_good_idx = real_good_idx[::-1]
    if length > 1:
        gaps = [real_good_idx[i]-real_good_idx[i-1] for i in range(1,length)]
            
        


    #penalize geometric inconsistency
    c_std = 0.08
    c_narrow = 0.6
    #inconsistent character spacing
    if len(gaps)>=4:
        std_loss = c_std*np.std(gaps);
    else:
        std_loss = 0;
    #very narrow characters
    narrow_loss = 0;
    if origionword !='I' and origionword !='i':
        if origscores.shape[1]/w<8:
            narrow_loss = (8-origscores.shape[1]/w)*c_narrow;
    #penalize excessive extra space on both sides
    matchScore = matchScore- std_loss-narrow_loss - ((min(real_good_idx) - 1)/os + (os-max(real_good_idx))/os);

    return matchScore,real_good_idx
#####################################################  
LIST_LEXICON = list(set(LIST_LEXICON))
    
#LIST_LEXICON = ['rowland','good','hello']
length = len(LIST_LEXICON)
matchScoreArray = np.zeros((1,length)); 
regex = r"[^\w\d]" # which is the same as \W btw
pat = re.compile(regex)
for i in range(length):
        regex_string = pat.sub('', LIST_LEXICON[i] )
        if len(regex_string)>2 and len(regex_string)<16:
            tempscore,real_good_idx= matchScores(response_data[0:-1,:],regex_string, climax_index)
            matchScoreArray[0,i] = tempscore
print np.max(matchScoreArray)
best_index =  np.argmax(matchScoreArray)
best_match = pat.sub('', LIST_LEXICON[best_index] )
print best_match
#####################################################
#Non Maxima Supression
def non_maxima_supress(image,threshold = 11):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(threshold, threshold))
    dilate_image = cv2.dilate(image,kernel)  # max() with radius of threshold

    nms_img = np.zeros([image.shape[0],image.shape[1]])
    print nms_img.shape
    list_index = [(c,r) for c in range(image.shape[1]) for r in range(image.shape[0])]   #column first
    for c,r in list_index:
    	if dilate_image[r][c] == image[r][c]:
    		nms_img[r][c] = image[r][c]
     
    return nms_img

def draw_break_line(location,origin_img):
	'''
	Draw the split line between the words
	Parameters
	----------
	location    one dimention array
	'''  
	print origin_img.shape
	length = location.shape[1]
	for i in range(length):
		if location[0][i] != 0:
			cv2.line(origin_img,(i,0),(i,origin_img.shape[0]),(255,0,0),1)
	plt.imshow( origin_img)
	plt.show()
'''
w,h = gaussian_data.shape
print w,h
#nms_img_data_test = np.sum(gaussian_data,axis = 0)
nms_img_data_test = gaussian_data[0][:]
print nms_img_data_test.shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
dilate_image = cv2.dilate(nms_img_data_test.reshape(1,h),kernel)
plt.imshow(dilate_image)
plt.title('dilate')
plt.savefig('./temp/dilate.png')
plt.show()
data = nms_img_data_test.reshape(1,h) - dilate_image
plt.imshow(data)
plt.title('nms')
plt.savefig('./temp/nms.png')
plt.show()
word_location = non_maxima_supress(nms_img_data_test.reshape(1,h),8)
length = word_location.shape[1]
origin_img =caffe.io.load_image('./data/crop1.jpg')
for i in range(length):
	if word_location[0][i] != 0:
		cv2.line(origin_img,(i,0),(i,origin_img.shape[0]),(255,0,0),1)
x1 = range(h)
plt.subplot(3,1,1)
plt.title('raw response')
plt.plot(x1,list(data[0]),'r-')
plt.subplot(3,1,2)
plt.title('NMS response')
plt.plot(x1,list(word_location[0]),'b-')
plt.subplot(3,1,3)
plt.title('split word')
plt.imshow( origin_img)
plt.subplots_adjust(hspace = 0.5)
plt.savefig('./temp/fig.png')
plt.show()

#draw_break_line(word_location,caffe.io.load_image('crop1.jpg'))
'''


